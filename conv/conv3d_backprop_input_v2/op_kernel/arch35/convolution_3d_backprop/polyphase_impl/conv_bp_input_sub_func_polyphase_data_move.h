/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_bp_input_sub_func_polyphase_data_move.h
 * \brief Polyphase 共用搬运和向量布局变换。
 */

#ifndef CONV_BP_INPUT_SUB_FUNC_POLYPHASE_DATA_MOVE_H
#define CONV_BP_INPUT_SUB_FUNC_POLYPHASE_DATA_MOVE_H

#include "../impl/conv_bp_input_sub_func_load_gm_to_l1.h"
#include "../impl/conv_bp_input_sub_func_store_l0c_fixpipe.h"
#include "../impl/conv_bp_input_sub_func_vector_intrinsics.h"

namespace Convolution3DBackpropFunc {

struct PolyphaseDataMove {
    static constexpr uint32_t C0 = 16U;
    static constexpr uint32_t PAD_TOP_AXIS = 2U;
    static constexpr uint32_t PAD_BOTTOM_AXIS = 3U;
    static constexpr AscendC::NdDmaConfig NDDMA_CONFIG = {false};

    // GM 步长和 destinationMatrixStride 以元素计，其余目标步长以 C0 块计。
    struct DnLayout {
        uint32_t matrices;
        uint32_t columns;
        uint32_t rows;
        uint64_t sourceMatrixStride;
        uint64_t sourceRowStride;
        uint32_t destinationMatrixStride;
        uint32_t destinationC0Stride;
        uint32_t destinationColumnStride;
    };

    // NDDMA 源和目标步长均以元素计。
    struct DmaAxis {
        uint32_t count;
        uint64_t sourceStride;
        uint32_t destinationStride;
    };

    struct Load3dShape {
        uint32_t width;
        uint32_t channels;
        uint32_t window;
        uint32_t mStart;
        uint32_t mSize;
        uint32_t kStart;
        uint32_t kSize;
        uint32_t destinationStride; // LoadData repeat 步长以 512 字节 fractal 计。
    };

    template <typename DataType>
    __aicore__ static inline void ZeroL1(const LocalTensor<DataType>& tensor)
    {
        AscendC::InitConstValueParams<DataType> params;
        params.repeatTimes = 1;
        params.blockNum = tensor.GetSize() * sizeof(DataType) / AscendC::GetDataBlockSizeInBytes();
        params.initValue = static_cast<DataType>(0);
        AscendC::Fill(tensor, params);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    template <typename DataType>
    __aicore__ static inline void GmToL1(const LocalTensor<DataType>& dst, const GlobalTensor<DataType>& src,
                                         const DnLayout& layout)
    {
        AscendC::Dn2NzParams params;
        params.dnNum = layout.matrices;
        params.nValue = layout.columns;
        params.dValue = layout.rows;
        params.srcDnMatrixStride = layout.sourceMatrixStride;
        params.srcDValue = layout.sourceRowStride;
        params.dstNzMatrixStride = layout.destinationMatrixStride;
        params.dstNzC0Stride = layout.destinationC0Stride;
        params.dstNzNStride = layout.destinationColumnStride;
        AscendC::DataCopy(dst, src, params);
    }

    template <typename DataType>
    __aicore__ static inline void GmToL1Matrix(const LocalTensor<DataType>& dst, const GlobalTensor<DataType>& src,
                                               uint32_t columns, uint32_t rows, uint64_t sourcePitch,
                                               uint32_t destinationPitch)
    {
        GmToL1(dst, src, {1U, columns, rows, sourcePitch, sourcePitch, destinationPitch * C0, destinationPitch, 1U});
    }

    template <typename DataType>
    __aicore__ static inline void GmToL1Window(const LocalTensor<DataType>& dst, const GlobalTensor<DataType>& src,
                                               int64_t firstColumn, uint32_t columns, uint32_t sourceWidth,
                                               uint32_t channels)
    {
        ZeroL1(dst);
        const int64_t last = firstColumn + columns - 1;
        const int64_t first = firstColumn > 0 ? firstColumn : 0;
        const int64_t sourceLast = static_cast<int64_t>(sourceWidth) - 1;
        const int64_t end = last < sourceLast ? last : sourceLast;
        if (first <= end) {
            const uint32_t offset = static_cast<uint32_t>(first - firstColumn) * C0;
            GmToL1Matrix(dst[offset], src[first], static_cast<uint32_t>(end - first + 1), channels, sourceWidth,
                         columns);
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    template <uint8_t axis, typename DataType, uint8_t dimensions>
    __aicore__ static inline void SetDmaAxes(AscendC::NdDmaParams<DataType, dimensions>& params,
                                             const DmaAxis (&axes)[dimensions])
    {
        params.loopInfo.loopSize[axis] = axes[axis].count;
        params.loopInfo.loopSrcStride[axis] = axes[axis].sourceStride;
        params.loopInfo.loopDstStride[axis] = axes[axis].destinationStride;
        if constexpr (axis + 1U < dimensions) {
            SetDmaAxes<axis + 1U, DataType, dimensions>(params, axes);
        }
    }

    template <uint8_t dimensions, typename DataType>
    __aicore__ static inline void GmToUb(const LocalTensor<DataType>& dst, const GlobalTensor<DataType>& src,
                                         const DmaAxis (&axes)[dimensions])
    {
        AscendC::NdDmaParams<DataType, dimensions> params{};
        SetDmaAxes<0U, DataType, dimensions>(params, axes);
        AscendC::DataCopy<DataType, dimensions, NDDMA_CONFIG>(dst, src, params);
    }

    template <typename DataType>
    __aicore__ static inline void GmToUbVector(const LocalTensor<DataType>& dst, const GlobalTensor<DataType>& src,
                                               uint32_t elements, uint8_t rightPadding)
    {
        AscendC::DataCopyExtParams copy{1U, static_cast<uint32_t>(elements * sizeof(DataType)), 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<DataType> pad{true, 0U, rightPadding, static_cast<DataType>(0)};
        AscendC::DataCopyPad(dst, src, copy, pad);
    }

    template <typename DataType>
    __aicore__ static inline void L1ToL0A(const LocalTensor<DataType>& dst, const LocalTensor<DataType>& src,
                                          const Load3dShape& shape)
    {
        AscendC::LoadData3DParamsV2<DataType> params;
        params.filterH = 1U;
        params.filterW = shape.window;
        params.filterSizeH = false;
        params.filterSizeW = false;
        params.dilationFilterH = 1U;
        params.dilationFilterW = 1U;
        params.channelSize = shape.channels;
        params.l1H = 1U;
        params.l1W = shape.width;
        params.padList[0] = 0U;
        params.padList[1] = 0U;
        params.padList[PAD_TOP_AXIS] = 0U;
        params.padList[PAD_BOTTOM_AXIS] = UINT8_MAX;
        params.strideH = 1U;
        params.strideW = 1U;
        params.mStartPt = shape.mStart;
        params.mExtension = shape.mSize;
        params.kStartPt = shape.kStart;
        params.kExtension = shape.kSize;
        params.enTranspose = false;
        params.enSmallK = false;
        params.fMatrixCtrl = false;
        AscendC::LoadDataRepeatParamWithStride repeat;
        repeat.repeatStride = 0U;
        repeat.repeatTime = 1U;
        repeat.repeatMode = 0U;
        repeat.dstStride = shape.destinationStride;
        AscendC::SetLoadDataRepeatWithStride(repeat);
        AscendC::LoadDataWithStride(dst, src, params);
    }

    template <typename DataType>
    __aicore__ static inline void L1ToL0B(const LocalTensor<DataType>& dst, const LocalTensor<DataType>& src,
                                          uint32_t sourcePitch, uint32_t nBlocks, uint32_t kStart, uint32_t kBlocks)
    {
        AscendC::LoadData2DParamsV2 params;
        params.ifTranspose = false;
        params.srcStride = sourcePitch;
        params.mStep = nBlocks;
        params.dstStride = nBlocks;
        params.mStartPosition = 0U;
        params.kStep = kBlocks;
        params.kStartPosition = kStart;
        AscendC::LoadData(dst, src, params);
    }

    template <typename DataType>
    __aicore__ static inline void UbToL1(const LocalTensor<DataType>& dst, const LocalTensor<DataType>& src,
                                         uint32_t elements)
    {
        AscendC::DataCopy(dst, src, elements);
    }

    template <typename DataType>
    __aicore__ static inline void UbToL1Blocks(const LocalTensor<DataType>& dst, const LocalTensor<DataType>& src,
                                               uint32_t elements)
    {
        AscendC::DataCopyParams params{
            1U, static_cast<uint16_t>(elements * sizeof(DataType) / AscendC::GetDataBlockSizeInBytes()), 0U, 0U};
        AscendC::DataCopy(dst, src, params);
    }

    template <typename DataType>
    __aicore__ static inline void UbToGm(const GlobalTensor<DataType>& dst, const LocalTensor<DataType>& src,
                                         uint32_t rows, uint32_t columns, uint32_t sourceGapBlocks,
                                         uint64_t destinationGapBytes)
    {
        AscendC::DataCopyExtParams copy;
        copy.blockCount = static_cast<uint16_t>(rows);
        copy.blockLen = columns * sizeof(DataType);
        copy.srcStride = sourceGapBlocks;
        copy.dstStride = destinationGapBytes;
        AscendC::DataCopyPad<DataType, AscendC::PaddingMode::Normal>(dst, src, copy);
    }

    // srcStride 以 C0 * sizeof(SrcType) 字节计，dstStride 以元素计。
    template <typename DstType, bool subBlock = false, typename SrcType>
    __aicore__ static inline void L0cToUb(LocalTensor<DstType>& dst, const LocalTensor<SrcType>& src, uint32_t m,
                                          uint32_t n, uint32_t srcStride, uint32_t dstStride, QuantMode_t quant)
    {
        FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> params;
        params.params.dnNum = 1U;
        params.params.srcNzMatrixStride = 0U;
        params.params.dstDnMatrixStride = 0U;
        params.params.srcNzC0Stride = 1U;
        params.mSize = static_cast<uint16_t>(m);
        params.nSize = static_cast<uint16_t>(n);
        params.srcStride = static_cast<uint16_t>(srcStride);
        params.dstStride = static_cast<uint16_t>(dstStride);
        params.quantPre = quant;
        params.dualDstCtl = 0U;
        params.subBlockId = subBlock;
        Fixpipe<DstType, SrcType, CFG_COLUMN_MAJOR_UB>(dst, src, params);
    }
};

/**
 * 将原始 ND=[N,Kcompact] 转为 B1 NZ=[Cout1,q,N1,C0,C0]。
 * 原 tap 按 r + s * t 排列；Gather 的 q - 1 - h 将 B1 行 h 映射到
 * 原 tap r + s * (q - 1 - h)，与递增的 Load3D 窗口一致。
 */
template <typename T>
__simd_vf__ inline void PolyphaseFilterDn2Nz(__ubuf__ uint16_t* index, __ubuf__ T* raw, __ubuf__ T* nz,
                                             uint16_t coutBlocks, uint16_t q, uint16_t kCompact)
{
    AscendC::Reg::RegTensor<uint16_t> indexReg;
    AscendC::Reg::RegTensor<uint16_t> tapIndex;
    AscendC::Reg::RegTensor<T> valueReg;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::LoadAlign(indexReg, index);
    for (uint16_t coutBlock = 0; coutBlock < coutBlocks; ++coutBlock) {
        for (uint16_t h = 0; h < q; ++h) {
            const uint32_t srcBase = static_cast<uint32_t>(coutBlock) * PolyphaseDataMove::C0 * q + (q - 1U - h);
            const uint32_t dstBase = (static_cast<uint32_t>(coutBlock) * q + h) * PolyphaseDataMove::C0 *
                                     PolyphaseDataMove::C0;
            // Gather 基址保持 32 字节对齐，tap 偏移放入元素索引。
            AscendC::Reg::Adds(tapIndex, indexReg, static_cast<uint16_t>(srcBase), fullMask);
            AscendC::Reg::Gather<T, T, uint16_t>(valueReg, raw, tapIndex, fullMask);
            AscendC::Reg::StoreAlign(nz + dstBase, valueReg, fullMask);
            AscendC::Reg::Adds(tapIndex, tapIndex,
                               static_cast<uint16_t>(
                                   (AscendC::VECTOR_REG_WIDTH / sizeof(uint16_t) / PolyphaseDataMove::C0) * kCompact),
                               fullMask);
            AscendC::Reg::Gather<T, T, uint16_t>(valueReg, raw, tapIndex, fullMask);
            AscendC::Reg::StoreAlign(nz + dstBase + (AscendC::VECTOR_REG_WIDTH / sizeof(uint16_t)), valueReg, fullMask);
        }
    }
}

// Gather 按 stride 将 phase-major 输出转为连续 W。
template <uint32_t STRIDE, typename T>
__simd_vf__ inline void GatherPhaseMajorToW(__ubuf__ uint16_t* index, __ubuf__ T* src, __ubuf__ T* dst, uint16_t nAlign,
                                            uint16_t mAlign, uint16_t outputRowAlign)
{
    AscendC::Reg::RegTensor<uint16_t> indexReg;
    AscendC::Reg::RegTensor<T> valueReg;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    const uint16_t repeats = outputRowAlign / (AscendC::VECTOR_REG_WIDTH / sizeof(uint16_t));
    for (uint16_t n = 0; n < nAlign; ++n) {
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            const uint32_t vectorOffset = static_cast<uint32_t>(repeat) *
                                          (AscendC::VECTOR_REG_WIDTH / sizeof(uint16_t));
            AscendC::Reg::LoadAlign(indexReg, index + vectorOffset);
            AscendC::Reg::Gather<T, T, uint16_t>(valueReg, src + static_cast<uint32_t>(n) * mAlign, indexReg, fullMask);
            AscendC::Reg::StoreAlign(dst + static_cast<uint32_t>(n) * outputRowAlign + vectorOffset, valueReg,
                                     fullMask);
        }
    }
}

constexpr AscendC::Reg::CastTrait POLYPHASE_INTRIN_CAST_FP32_TO_HALF = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

// 广播 FP32 bias，并将输出 bank 转为输出 dtype。
template <typename T>
__simd_vf__ inline void PolyphaseAddBiasCast(__ubuf__ float* source, __ubuf__ float* bias, __ubuf__ T* destination,
                                             uint16_t stride, uint16_t nCount, uint16_t nPitch, uint16_t mAlign)
{
    AscendC::Reg::RegTensor<float> sourceReg;
    AscendC::Reg::RegTensor<float> biasReg;
    AscendC::Reg::RegTensor<float> sumReg;
    AscendC::Reg::RegTensor<T> outputReg;
    for (uint16_t phase = 0U; phase < stride; ++phase) {
        for (uint16_t n = 0U; n < nCount; ++n) {
            uint32_t remaining = mAlign;
            const uint32_t row = (static_cast<uint32_t>(phase) * nPitch + n) * mAlign;
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(biasReg, bias + n);
            for (uint32_t offset = 0U; offset < mAlign; offset += (AscendC::VECTOR_REG_WIDTH / sizeof(float))) {
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
                AscendC::Reg::LoadAlign(sourceReg, source + row + offset);
                AscendC::Reg::Add(sumReg, sourceReg, biasReg, mask);
                AscendC::Reg::Cast<T, float, POLYPHASE_INTRIN_CAST_FP32_TO_HALF>(outputReg, sumReg, mask);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(destination + row + offset,
                                                                                    outputReg, mask);
            }
        }
    }
}

template <typename T>
__simd_vf__ inline void PolyphaseCastNoBias(__ubuf__ float* source, __ubuf__ T* destination, uint16_t stride,
                                            uint16_t nCount, uint16_t nPitch, uint16_t mAlign)
{
    AscendC::Reg::RegTensor<float> sourceReg;
    AscendC::Reg::RegTensor<T> outputReg;
    for (uint16_t phase = 0U; phase < stride; ++phase) {
        for (uint16_t n = 0U; n < nCount; ++n) {
            uint32_t remaining = mAlign;
            const uint32_t row = (static_cast<uint32_t>(phase) * nPitch + n) * mAlign;
            for (uint32_t offset = 0U; offset < mAlign; offset += (AscendC::VECTOR_REG_WIDTH / sizeof(float))) {
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
                AscendC::Reg::LoadAlign(sourceReg, source + row + offset);
                AscendC::Reg::Cast<T, float, POLYPHASE_INTRIN_CAST_FP32_TO_HALF>(outputReg, sourceReg, mask);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(destination + row + offset,
                                                                                    outputReg, mask);
            }
        }
    }
}

__simd_vf__ inline static void SegmentedOutputIndex(__ubuf__ uint16_t* index, uint16_t m, uint16_t stride, uint16_t row)
{
    AscendC::Reg::RegTensor<uint16_t> w;
    AscendC::Reg::RegTensor<uint16_t> u;
    AscendC::Reg::RegTensor<uint16_t> r;
    AscendC::Reg::RegTensor<uint16_t> product;
    AscendC::Reg::RegTensor<uint16_t> result;
    AscendC::Reg::RegTensor<uint16_t> divisor;
    AscendC::Reg::RegTensor<uint16_t> zero;
    auto all = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(divisor, stride, all);
    AscendC::Reg::Duplicate(zero, static_cast<uint16_t>(0), all);
    const uint16_t width = static_cast<uint16_t>(static_cast<uint32_t>(stride) * m);
    const uint16_t phasePitch = static_cast<uint16_t>(PolyphaseDataMove::C0 * m);
    for (uint32_t offset = 0U; offset < row; offset += (AscendC::VECTOR_REG_WIDTH / sizeof(uint16_t))) {
        AscendC::Reg::Arange<int16_t>(reinterpret_cast<AscendC::Reg::RegTensor<int16_t>&>(w),
                                      static_cast<int16_t>(offset));
        AscendC::Reg::MaskReg valid;
        AscendC::Reg::Compares<uint16_t, AscendC::CMPMODE::LT>(valid, w, width, all);
        AscendC::Reg::Select(w, w, zero, valid);
        AscendC::Reg::Div(u, w, divisor, all);
        AscendC::Reg::Muls(product, u, stride, all);
        AscendC::Reg::Sub(r, w, product, all);
        AscendC::Reg::Muls(result, r, phasePitch, all);
        AscendC::Reg::Add(result, result, u, all);
        AscendC::Reg::StoreAlign(index + offset, result, all);
    }
}

__simd_vf__ inline static void SegmentedParallelOutputIndex(__ubuf__ uint16_t* index, uint16_t m, uint16_t q,
                                                            uint16_t width, uint16_t row)
{
    AscendC::Reg::RegTensor<uint16_t> lane;
    AscendC::Reg::RegTensor<uint16_t> u;
    AscendC::Reg::RegTensor<uint16_t> v;
    AscendC::Reg::RegTensor<uint16_t> product;
    AscendC::Reg::RegTensor<uint16_t> result;
    AscendC::Reg::RegTensor<uint16_t> divisor;
    AscendC::Reg::RegTensor<uint16_t> zero;
    auto all = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(divisor, q, all);
    AscendC::Reg::Duplicate(zero, static_cast<uint16_t>(0), all);
    // host 已将输出索引限制在 uint16 范围。
    // 先裁剪 padding 的 v，再相乘，使中间值均适配 uint16。
    const uint16_t phasePitch = static_cast<uint16_t>(PolyphaseDataMove::C0 * static_cast<uint32_t>(m));
    for (uint32_t offset = 0U; offset < row; offset += (AscendC::VECTOR_REG_WIDTH / sizeof(uint16_t))) {
        AscendC::Reg::Arange<int16_t>(reinterpret_cast<AscendC::Reg::RegTensor<int16_t>&>(lane),
                                      static_cast<int16_t>(offset));
        AscendC::Reg::Div(u, lane, divisor, all);
        AscendC::Reg::Muls(product, u, q, all);
        AscendC::Reg::Sub(v, lane, product, all);
        AscendC::Reg::MaskReg valid;
        AscendC::Reg::Compares<uint16_t, AscendC::CMPMODE::LT>(valid, v, width, all);
        AscendC::Reg::Select(v, v, zero, valid);
        AscendC::Reg::Muls(result, v, phasePitch, all);
        AscendC::Reg::Add(result, result, u, all);
        AscendC::Reg::StoreAlign(index + offset, result, all);
    }
}

template <typename T>
__simd_vf__ inline void PolyphaseDirectDensePhase(__ubuf__ float* p, __ubuf__ float* bias, __ubuf__ T* y, uint16_t nc,
                                                  uint16_t kw, uint16_t m, uint16_t pitch, uint16_t count,
                                                  uint16_t phase, uint16_t stride, uint16_t kFirst, uint16_t kStep,
                                                  uint16_t taps, int32_t inputFirst, int32_t inputStep, int32_t halo,
                                                  bool hasBias)
{
    AscendC::Reg::RegTensor<int32_t> seq;
    AscendC::Reg::RegTensor<int32_t> ix;
    AscendC::Reg::RegTensor<uint16_t> outIx;
    AscendC::Reg::RegTensor<float> sum;
    AscendC::Reg::RegTensor<float> term;
    AscendC::Reg::RegTensor<float> b;
    AscendC::Reg::RegTensor<T> cast;
    AscendC::Reg::Arange<int32_t>(seq, static_cast<int32_t>(0));
    AscendC::Reg::Arange<int16_t>(reinterpret_cast<AscendC::Reg::RegTensor<int16_t>&>(outIx), static_cast<int16_t>(0));
    auto all16 = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::ShiftRights(outIx, outIx, static_cast<int16_t>(1), all16);
    AscendC::Reg::Muls(outIx, outIx, stride, all16);
    for (uint16_t c = 0; c < nc; ++c) {
        for (uint32_t off = 0; off < count; off += (AscendC::VECTOR_REG_WIDTH / sizeof(float))) {
            uint32_t remaining = count - off;
            auto mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::Duplicate(sum, 0.0f, mask);
            int32_t start = inputFirst + static_cast<int32_t>(off);
            uint16_t k = kFirst;
            for (uint16_t t = 0; t < taps; ++t, k += kStep, start -= inputStep) {
                AscendC::Reg::Adds(ix, seq, start, mask);
                AscendC::Reg::MaskReg lower;
                AscendC::Reg::MaskReg upper;
                AscendC::Reg::MaskReg valid;
                AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::GE>(lower, ix, static_cast<int32_t>(0), mask);
                AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::LT>(upper, ix, halo, mask);
                AscendC::Reg::And(valid, lower, upper, mask);
                AscendC::Reg::Gather(term, p + (static_cast<uint32_t>(c) * kw + k) * m,
                                     reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(ix), valid);
                AscendC::Reg::Add(sum, sum, term, mask);
            }
            if (hasBias) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(b, bias + c);
                AscendC::Reg::Add(sum, sum, b, mask);
            }
            AscendC::Reg::Cast<T, float, Convolution3DBackpropFunc::POLYPHASE_INTRIN_CAST_FP32_TO_HALF>(cast, sum,
                                                                                                        mask);
            AscendC::Reg::Scatter(y + static_cast<uint32_t>(c) * pitch + phase + static_cast<uint32_t>(off) * stride,
                                  cast, outIx, mask);
        }
    }
}

} // namespace Convolution3DBackpropFunc

#endif // CONV_BP_INPUT_SUB_FUNC_POLYPHASE_DATA_MOVE_H
