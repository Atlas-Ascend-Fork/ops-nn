/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LAYER_NORM_V3_REGBASE_TRANSPOSE_H
#define LAYER_NORM_V3_REGBASE_TRANSPOSE_H

#include "layer_norm_v3_common.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace LayerNormV3 {
// Keep parameter mode values in sync with the transpose host tiling.
constexpr uint32_t TRANSPOSE_PARAM_SAME = 0;
constexpr uint32_t TRANSPOSE_PARAM_BRC_P = 1;
constexpr uint32_t TRANSPOSE_PARAM_TILE_PARAMS = 2;
constexpr uint32_t TRANSPOSE_PARAM_FULL_PARAMS = 3;

using AscendC::Reg::Duplicate;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::LoadDist;
using AscendC::Reg::MaskReg;
using AscendC::Reg::Mul;
using AscendC::Reg::MulDstAdd;
using AscendC::Reg::Muls;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreAlign;
using AscendC::Reg::Sub;
using AscendC::Reg::UpdateMask;
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

// Gather uses element indices and a 32B-aligned base; count is at most VL_B32.
template <typename T>
__aicore__ inline void GatherRegForDtype(__ubuf__ T* src, RegTensor<float>& dst, RegTensor<uint32_t>& index,
                                         MaskReg& mask, uint32_t count)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Reg::Gather(dst, src, index, mask);
    } else {
        RegTensor<uint16_t> indexB16;
        RegTensor<T> packed, unpacked;
        MaskReg loadMask = UpdateMask<T>(count);
        AscendC::Reg::Pack(indexB16, index);
        AscendC::Reg::Gather(packed, src, indexB16, loadMask);
        AscendC::Reg::UnPack((RegTensor<uint32_t>&)unpacked, (RegTensor<uint16_t>&)packed);
        AscendC::Reg::Cast<float, T, castTraitB162B32>(dst, unpacked, mask);
    }
}

template <typename T, typename U, typename M, bool IsOutRstd, uint32_t PARAM_MODE>
class LayerNormV3RegbaseTranspose {
    static constexpr uint32_t TRANSPOSE_SIZE = NCHW_CONV_ADDR_LIST_SIZE;
    static constexpr uint32_t B32_TRANSPOSE_COLS = TRANSPOSE_SIZE / NUM_TWO;
    static constexpr uint32_t PARAM_BUFFER_COUNT = PARAM_MODE == TRANSPOSE_PARAM_TILE_PARAMS ? DOUBLE_BUFFER : 1;

public:
    __aicore__ inline LayerNormV3RegbaseTranspose(const LayerNormV3TilingDataRegBaseTranspose* tiling, TPipe* pipe)
    {
        td_ = tiling;
        pipe_ = pipe;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR lastout)
    {
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        yGm_.SetGlobalBuffer((__gm__ T*)y);
        gammaGm_.SetGlobalBuffer((__gm__ U*)gamma);
        betaGm_.SetGlobalBuffer((__gm__ U*)beta);
        meanGm_.SetGlobalBuffer((__gm__ M*)mean);
        lastoutGm_.SetGlobalBuffer((__gm__ M*)lastout);
        tileA_ = td_->tileA;
        r_ = td_->r;
        rAlign_ = td_->rAlign;
        paramAlign_ = td_->paramAlign;

        pipe_->InitBuffer(xQueue_, DOUBLE_BUFFER, tileA_ * r_ * sizeof(float));
        pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, tileA_ * rAlign_ * sizeof(T));
        pipe_->InitBuffer(tempBuf_, tileA_ * rAlign_ * sizeof(float));
        pipe_->InitBuffer(meanQueue_, DOUBLE_BUFFER, tileA_ * sizeof(float));
        pipe_->InitBuffer(lastoutQueue_, DOUBLE_BUFFER, tileA_ * sizeof(float));
        pipe_->InitBuffer(paramsQueue_, PARAM_BUFFER_COUNT, NUM_TWO * paramAlign_ * sizeof(U));
        uint32_t powerOfTwo = 1;
        while (powerOfTwo < r_) {
            powerOfTwo *= NUM_TWO;
        }
        scale_ = 1.0f / static_cast<float>(powerOfTwo);
        powerOfTwo_ = static_cast<float>(powerOfTwo);
    }

    __aicore__ inline void Process()
    {
        const uint64_t begin = GetBlockIdx() * td_->aBlockFactor;
        if (begin >= td_->a) {
            return;
        }
        const uint64_t rows = td_->a - begin < td_->aBlockFactor ? td_->a - begin : td_->aBlockFactor;
        const uint64_t tileCount = (rows + tileA_ - 1) / tileA_;
        const uint32_t tailA = rows - (tileCount - 1) * tileA_;
        NdDmaDci();
        if constexpr (PARAM_MODE != TRANSPOSE_PARAM_TILE_PARAMS) {
            CopyInParams(0, 0);
            paramsLocal_ = paramsQueue_.template DeQue<U>();
        }
        for (uint64_t idx = 0; idx < tileCount; ++idx) {
            const uint64_t row = begin + idx * tileA_;
            const uint32_t curA = idx + 1 == tileCount ? tailA : tileA_;
            if constexpr (PARAM_MODE == TRANSPOSE_PARAM_TILE_PARAMS) {
                CopyInParams(row, curA);
                paramsLocal_ = paramsQueue_.template DeQue<U>();
            }
            CopyInX(row, curA);
            ComputeMeanRstd(curA);
            ComputeOutput(row, curA);
            if constexpr (PARAM_MODE == TRANSPOSE_PARAM_TILE_PARAMS) {
                paramsQueue_.FreeTensor(paramsLocal_);
            }
            CopyOutY(row, curA);
            CopyOutMeanLastout(row, curA);
        }
        if constexpr (PARAM_MODE != TRANSPOSE_PARAM_TILE_PARAMS) {
            paramsQueue_.FreeTensor(paramsLocal_);
        }
    }

private:
    __aicore__ inline void CopyInParams(uint64_t row, uint32_t curA)
    {
        LocalTensor<U> params = paramsQueue_.template AllocTensor<U>();
        if constexpr (PARAM_MODE == TRANSPOSE_PARAM_SAME || PARAM_MODE == TRANSPOSE_PARAM_FULL_PARAMS) {
            const uint32_t rows = PARAM_MODE == TRANSPOSE_PARAM_FULL_PARAMS ? td_->paramRows : 1;
            DataCopyExtParams copy{1, static_cast<uint32_t>(rows * r_ * sizeof(U)), 0, 0, 0};
            DataCopyPadExtParams<U> pad{false, 0, 0, 0};
            DataCopyPad(params, gammaGm_, copy, pad);
            DataCopyPad(params[paramAlign_], betaGm_, copy, pad);
        } else if constexpr (PARAM_MODE == TRANSPOSE_PARAM_BRC_P) {
            static constexpr NdDmaConfig config = {false};
            NdDmaLoopInfo<NUM_TWO> loop;
            loop.loopSrcStride[0] = 1;
            loop.loopSrcStride[1] = 0;
            loop.loopDstStride[0] = 1;
            loop.loopDstStride[1] = td_->paramR;
            loop.loopSize[0] = td_->paramR;
            loop.loopSize[1] = r_ / td_->paramR;
            NdDmaParams<U, NUM_TWO> copy{loop, 0};
            DataCopy<U, NUM_TWO, config>(params, gammaGm_, copy);
            DataCopy<U, NUM_TWO, config>(params[paramAlign_], betaGm_, copy);
        } else {
            const int64_t startB = static_cast<int64_t>(row) % static_cast<int64_t>(td_->paramRows);
            const uint32_t first = td_->paramRows - startB < curA ? td_->paramRows - startB : curA;
            if (first == curA) {
                // A single continuous AR segment needs no NdDma layout conversion.
                DataCopyExtParams copy{1, static_cast<uint32_t>(curA * r_ * sizeof(U)), 0, 0, 0};
                DataCopyPadExtParams<U> pad{false, 0, 0, 0};
                const int64_t offset = startB * r_;
                DataCopyPad(params, gammaGm_[offset], copy, pad);
                DataCopyPad(params[paramAlign_], betaGm_[offset], copy, pad);
            } else {
                // NdDma preserves compact AR at unaligned period boundaries without padding.
                CopyParamRows(params, startB, 0, first);
                const uint32_t periods = (curA - first) / td_->paramRows;
                uint32_t copied = first;
                if (periods != 0) {
                    CopyParamRows(params, 0, copied, td_->paramRows, periods);
                    copied += periods * td_->paramRows;
                }
                if (copied < curA) {
                    CopyParamRows(params, 0, copied, curA - copied);
                }
            }
        }
        paramsQueue_.EnQue(params);
    }

    __aicore__ inline void CopyParamRows(const LocalTensor<U>& params, int64_t srcRow, uint32_t dstRow, uint32_t count,
                                         uint32_t periods = 1)
    {
        static constexpr NdDmaConfig config = {false};
        const uint32_t elements = count * r_;
        NdDmaLoopInfo<NUM_TWO> loop;
        loop.loopSrcStride[0] = 1;
        loop.loopSrcStride[1] = 0;
        loop.loopDstStride[0] = 1;
        loop.loopDstStride[1] = elements;
        loop.loopSize[0] = elements;
        loop.loopSize[1] = periods;
        NdDmaParams<U, NUM_TWO> copy{loop, 0};
        const int64_t offset = srcRow * r_;
        const uint32_t dstOffset = dstRow * r_;
        DataCopy<U, NUM_TWO, config>(params[dstOffset], gammaGm_[offset], copy);
        DataCopy<U, NUM_TWO, config>(params[paramAlign_ + dstOffset], betaGm_[offset], copy);
    }

    __aicore__ inline void CopyInX(uint64_t row, uint32_t curA)
    {
        static constexpr NdDmaConfig config = {false};
        NdDmaLoopInfo<NUM_TWO> loop;
        loop.loopSrcStride[0] = 1;
        loop.loopSrcStride[1] = r_;
        loop.loopDstStride[0] = tileA_ * sizeof(float) / sizeof(T);
        loop.loopDstStride[1] = 1;
        loop.loopSize[0] = r_;
        loop.loopSize[1] = curA;
        NdDmaParams<T, NUM_TWO> params = {loop, 0};
        LocalTensor<float> x = xQueue_.template AllocTensor<float>();
        LocalTensor<T> input = x.template ReinterpretCast<T>();
        if constexpr (!IsSameType<T, float>::value) {
            // Place B16 input in each FP32 row's upper half for safe forward expansion.
            input = input[tileA_];
        }
        DataCopy<T, NUM_TWO, config>(input, xGm_[row * r_], params);
        xQueue_.EnQue(x);
    }

    __aicore__ inline void ComputeMeanRstd(uint32_t curA)
    {
        xLocal_ = xQueue_.template DeQue<float>();
        meanLocal_ = meanQueue_.template AllocTensor<float>();
        lastoutLocal_ = lastoutQueue_.template AllocTensor<float>();
        CalculateMeanRstd(xLocal_, meanLocal_, lastoutLocal_, curA);
    }

    __aicore__ inline void ComputeOutput(uint64_t row, uint32_t curA)
    {
        LocalTensor<T> tempY = tempBuf_.Get<T>();
        Normalize(tempY, lastoutLocal_, row, curA);
        xQueue_.FreeTensor(xLocal_);
        CastOutput(meanLocal_, curA);
        CastOutput(lastoutLocal_, curA);
        meanQueue_.EnQue(meanLocal_);
        lastoutQueue_.EnQue(lastoutLocal_);
    }

    __aicore__ inline void CalculateMeanRstd(const LocalTensor<float>& x, const LocalTensor<float>& mean,
                                             const LocalTensor<float>& rstd, uint32_t curA)
    {
        LocalTensor<float> temp = tempBuf_.Get<float>();
        // RA source reuse needs no separate workspace or output-queue slot.
        LocalTensor<uint8_t> reduceTmp = temp.template ReinterpretCast<uint8_t>();
        uint32_t srcShape[NUM_TWO] = {r_, tileA_};
        CalcReduceInput<false>(x, temp, mean, curA);
        AscendC::ReduceSum<float, Pattern::Reduce::RA, true>(mean, temp, reduceTmp, srcShape, false);
        CalcMeanOrRstd<false>(mean, curA);
        CalcReduceInput<true>(x, temp, mean, curA);
        AscendC::ReduceSum<float, Pattern::Reduce::RA, true>(rstd, temp, reduceTmp, srcShape, false);
        CalcMeanOrRstd<true>(rstd, curA);
    }

    template <bool CALC_VARIANCE>
    __aicore__ inline void CalcReduceInput(const LocalTensor<float>& x, const LocalTensor<float>& temp,
                                           const LocalTensor<float>& mean, uint32_t curA)
    {
        __ubuf__ T* xAddr = (__ubuf__ T*)x.GetPhyAddr();
        __ubuf__ float* xSubMeanAddr = (__ubuf__ float*)x.GetPhyAddr();
        const uint32_t inputOffset = IsSameType<T, float>::value ? 0 : tileA_;
        const uint32_t inputStride = tileA_ * sizeof(float) / sizeof(T);
        __ubuf__ float* tempAddr = (__ubuf__ float*)temp.GetPhyAddr();
        __ubuf__ float* meanAddr = (__ubuf__ float*)mean.GetPhyAddr();
        const uint32_t stride = tileA_;
        const uint16_t rLoops = r_;
        const uint16_t aLoops = (curA + VL_B32 - 1) / VL_B32;
        const float scale = scale_;
        uint32_t maskCount = curA;
        __VEC_SCOPE__
        {
            RegTensor<float> xReg, meanReg, xSubMeanReg, reduceReg;
            MaskReg mask;
            for (uint16_t a = 0; a < aLoops; ++a) {
                mask = UpdateMask<float>(maskCount);
                const uint32_t aOffset = a * VL_B32;
                if constexpr (CALC_VARIANCE) {
                    LoadAlign(meanReg, meanAddr + aOffset);
                }
                for (uint16_t r = 0; r < rLoops; ++r) {
                    const uint32_t index = r * stride + aOffset;
                    LoadRegForDtype<T>(xAddr, xReg, mask, inputOffset + r * inputStride + aOffset);
                    if constexpr (CALC_VARIANCE) {
                        Sub(xSubMeanReg, xReg, meanReg, mask);
                        StoreAlign(xSubMeanAddr + index, xSubMeanReg, mask);
                        Mul(reduceReg, xSubMeanReg, xSubMeanReg, mask);
                        Muls(reduceReg, reduceReg, scale, mask);
                    } else {
                        Muls(reduceReg, xReg, scale, mask);
                    }
                    StoreAlign(tempAddr + index, reduceReg, mask);
                }
            }
        }
    }

    template <bool CALC_RSTD>
    __aicore__ inline void CalcMeanOrRstd(const LocalTensor<float>& result, uint32_t curA)
    {
        __ubuf__ float* resultAddr = (__ubuf__ float*)result.GetPhyAddr();
        const uint16_t aLoops = (curA + VL_B32 - 1) / VL_B32;
        const float powerOfTwo = powerOfTwo_;
        const float reduceNum = static_cast<float>(r_);
        const float epsilon = td_->epsilon;
        const float scaleFactor = powerOfTwo / reduceNum;
        uint32_t maskCount = curA;
        __VEC_SCOPE__
        {
            RegTensor<float> resultReg, rstdReg;
            MaskReg mask;
            for (uint16_t a = 0; a < aLoops; ++a) {
                mask = UpdateMask<float>(maskCount);
                const uint32_t aOffset = a * VL_B32;
                LoadAlign(resultReg, resultAddr + aOffset);
                Muls(resultReg, resultReg, scaleFactor, mask);
                if constexpr (CALC_RSTD && IsOutRstd) {
                    NormCommon::ComputeRstdNewtonRaphsonReg<false>(resultReg, rstdReg, mask, epsilon);
                    StoreAlign(resultAddr + aOffset, rstdReg, mask);
                } else {
                    StoreAlign(resultAddr + aOffset, resultReg, mask);
                }
            }
        }
    }

    __aicore__ inline void Normalize(const LocalTensor<T>& y, const LocalTensor<float>& lastout, uint64_t row,
                                     uint32_t curA)
    {
        __ubuf__ float* xSubMeanAddr = (__ubuf__ float*)xLocal_.GetPhyAddr();
        __ubuf__ float* lastoutAddr = (__ubuf__ float*)lastout.GetPhyAddr();
        __ubuf__ U* gamma = (__ubuf__ U*)paramsLocal_.GetPhyAddr();
        __ubuf__ U* beta = gamma + paramAlign_;
        __ubuf__ T* yAddr = (__ubuf__ T*)y.GetPhyAddr();
        const uint32_t stride = tileA_;
        const uint16_t rLoops = r_;
        const uint16_t aLoops = (curA + VL_B32 - 1) / VL_B32;
        const float epsilon = td_->epsilon;
        uint32_t paramStart = 0;
        uint32_t paramStep = 0;
        uint32_t paramRows = 1;
        if constexpr (PARAM_MODE == TRANSPOSE_PARAM_FULL_PARAMS) {
            paramRows = td_->paramRows;
            paramStart = row % td_->paramRows;
            paramStep = (VL_B32 % paramRows) * rLoops;
        }
        uint32_t maskCount = curA;
        __VEC_SCOPE__
        {
            RegTensor<float> xSubMeanReg, yReg, varianceReg, rstdReg, gammaReg, betaReg;
            RegTensor<uint32_t> paramRowOffsets, paramIndex;
            RegTensor<uint32_t> wrappedOffsets;
            MaskReg mask, wrapMask;
            MaskReg all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            if constexpr (PARAM_MODE == TRANSPOSE_PARAM_TILE_PARAMS || PARAM_MODE == TRANSPOSE_PARAM_FULL_PARAMS) {
                AscendC::Reg::Arange((RegTensor<int32_t>&)paramRowOffsets, 0);
                if constexpr (PARAM_MODE == TRANSPOSE_PARAM_FULL_PARAMS) {
                    // Reduce lane IDs modulo B exactly, including B smaller than a vector.
                    for (uint32_t factor = VL_B32 / NUM_TWO; factor != 0; factor /= NUM_TWO) {
                        const uint32_t divisor = paramRows * factor;
                        if (divisor < VL_B32) {
                            AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GE>(wrapMask, paramRowOffsets,
                                                                                        divisor, all);
                            AscendC::Reg::Adds(wrappedOffsets, paramRowOffsets, uint32_t(0) - divisor, all);
                            AscendC::Reg::Select(paramRowOffsets, wrappedOffsets, paramRowOffsets, wrapMask);
                        }
                    }
                    AscendC::Reg::Adds(paramRowOffsets, paramRowOffsets, paramStart, all);
                    AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GE>(wrapMask, paramRowOffsets, paramRows,
                                                                                all);
                    AscendC::Reg::Adds(wrappedOffsets, paramRowOffsets, uint32_t(0) - paramRows, all);
                    AscendC::Reg::Select(paramRowOffsets, wrappedOffsets, paramRowOffsets, wrapMask);
                }
                AscendC::Reg::Muls(paramRowOffsets, paramRowOffsets, static_cast<uint32_t>(rLoops), all);
            }
            for (uint16_t a = 0; a < aLoops; ++a) {
                mask = UpdateMask<float>(maskCount);
                const uint32_t aOffset = a * VL_B32;
                if constexpr (IsOutRstd) {
                    LoadAlign(rstdReg, lastoutAddr + aOffset);
                } else {
                    LoadAlign(varianceReg, lastoutAddr + aOffset);
                    NormCommon::ComputeRstdNewtonRaphsonReg<false>(varianceReg, rstdReg, mask, epsilon);
                }
                for (uint16_t r = 0; r < rLoops; ++r) {
                    const uint32_t index = r * stride + aOffset;
                    LoadAlign(xSubMeanReg, xSubMeanAddr + index);
                    Mul(yReg, xSubMeanReg, rstdReg, mask);
                    if constexpr (PARAM_MODE == TRANSPOSE_PARAM_TILE_PARAMS ||
                                  PARAM_MODE == TRANSPOSE_PARAM_FULL_PARAMS) {
                        // Batch bases stay 32B aligned; column offsets belong in the element indices.
                        const uint32_t paramBase = PARAM_MODE == TRANSPOSE_PARAM_FULL_PARAMS ? 0 : aOffset * rLoops;
                        AscendC::Reg::Adds(paramIndex, paramRowOffsets, static_cast<uint32_t>(r), mask);
                        const uint32_t active = curA - aOffset < VL_B32 ? curA - aOffset : VL_B32;
                        GatherRegForDtype(gamma + paramBase, gammaReg, paramIndex, mask, active);
                        GatherRegForDtype(beta + paramBase, betaReg, paramIndex, mask, active);
                    } else {
                        LoadRegForDtype<U, LoadDist::DIST_BRC_B32, LoadDist::DIST_BRC_B16>(gamma, gammaReg, mask, r);
                        LoadRegForDtype<U, LoadDist::DIST_BRC_B32, LoadDist::DIST_BRC_B16>(beta, betaReg, mask, r);
                    }
                    MulDstAdd(yReg, gammaReg, betaReg, mask);
                    StoreRegForDtype<T>(yAddr, yReg, mask, index);
                }
                if constexpr (PARAM_MODE == TRANSPOSE_PARAM_FULL_PARAMS) {
                    // Periods dividing a vector (for example B=2) reuse exactly the same indices.
                    if (paramStep != 0) {
                        AscendC::Reg::Adds(paramRowOffsets, paramRowOffsets, paramStep, all);
                        const uint32_t periodElements = paramRows * rLoops;
                        AscendC::Reg::CompareScalar<uint32_t, AscendC::CMPMODE::GE>(wrapMask, paramRowOffsets,
                                                                                    periodElements, all);
                        AscendC::Reg::Adds(wrappedOffsets, paramRowOffsets, uint32_t(0) - periodElements, all);
                        AscendC::Reg::Select(paramRowOffsets, wrappedOffsets, paramRowOffsets, wrapMask);
                    }
                }
            }
        }
    }

    __aicore__ inline void TransposeOutput(const LocalTensor<T>& src, const LocalTensor<T>& dst, uint32_t curA)
    {
        const uint32_t aRepeats = (curA + TRANSPOSE_SIZE - 1) / TRANSPOSE_SIZE;
        constexpr uint32_t cols = IsSameType<T, float>::value ? B32_TRANSPOSE_COLS : TRANSPOSE_SIZE;
        const uint32_t rRepeats = rAlign_ / cols;
        for (uint32_t r = 0; r < rRepeats; ++r) {
            TransDataTo5HDParams params;
            params.repeatTimes = aRepeats;
            params.srcRepStride = aRepeats == 1 ? 0 : sizeof(T) / sizeof(uint16_t);
            params.dstRepStride = aRepeats == 1 ? 0 : TRANSPOSE_SIZE * rRepeats;
            LocalTensor<T> srcList[TRANSPOSE_SIZE];
            LocalTensor<T> dstList[TRANSPOSE_SIZE];
            if constexpr (IsSameType<T, float>::value) {
                for (uint32_t j = 0; j < B32_TRANSPOSE_COLS; ++j) {
                    const uint32_t srcOffset = (r * B32_TRANSPOSE_COLS + j) * tileA_;
                    srcList[j] = src[srcOffset];
                    srcList[j + B32_TRANSPOSE_COLS] = src[srcOffset + B32_TRANSPOSE_COLS];
                    const uint32_t dstOffset = r * B32_TRANSPOSE_COLS + rAlign_ * j;
                    dstList[j * NUM_TWO] = dst[dstOffset];
                    dstList[j * NUM_TWO + 1] = dst[dstOffset + rAlign_ * B32_TRANSPOSE_COLS];
                }
            } else {
                for (uint32_t j = 0; j < TRANSPOSE_SIZE; ++j) {
                    srcList[j] = src[(r * TRANSPOSE_SIZE + j) * tileA_];
                    dstList[j] = dst[r * TRANSPOSE_SIZE + rAlign_ * j];
                }
            }
            TransDataTo5HD<T>(dstList, srcList, params);
        }
    }

    __aicore__ inline void CopyOutY(uint64_t row, uint32_t curA)
    {
        LocalTensor<T> src = tempBuf_.Get<T>();
        LocalTensor<T> dst = yQueue_.template AllocTensor<T>();
        TransposeOutput(src, dst, curA);
        yQueue_.EnQue(dst);
        dst = yQueue_.template DeQue<T>();
        DataCopyExtParams copy;
        copy.blockCount = curA;
        copy.blockLen = r_ * sizeof(T);
        copy.srcStride = 0;
        copy.dstStride = 0;
        DataCopyPad(yGm_[row * r_], dst, copy);
        yQueue_.FreeTensor(dst);
    }

    __aicore__ inline void CastOutput(const LocalTensor<float>& output, uint32_t count)
    {
        if constexpr (!IsSameType<M, float>::value) {
            __ubuf__ float* source = (__ubuf__ float*)output.GetPhyAddr();
            __ubuf__ M* destination = (__ubuf__ M*)output.GetPhyAddr();
            const uint16_t loops = (count + VL_B32 - 1) / VL_B32;
            __VEC_SCOPE__
            {
                RegTensor<float> outputReg;
                MaskReg mask;
                for (uint16_t idx = 0; idx < loops; ++idx) {
                    mask = UpdateMask<float>(count);
                    LoadAlign(outputReg, source + idx * VL_B32);
                    StoreRegForDtype<M>(destination, outputReg, mask, idx * VL_B32);
                }
            }
        }
    }

    __aicore__ inline void CopyOutMeanLastout(uint64_t row, uint32_t curA)
    {
        LocalTensor<float> mean = meanQueue_.template DeQue<float>();
        LocalTensor<float> lastout = lastoutQueue_.template DeQue<float>();
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(curA * sizeof(M)), 0, 0, 0};
        DataCopyPad(meanGm_[row], mean.template ReinterpretCast<M>(), dataCopyParams);
        DataCopyPad(lastoutGm_[row], lastout.template ReinterpretCast<M>(), dataCopyParams);
        meanQueue_.FreeTensor(mean);
        lastoutQueue_.FreeTensor(lastout);
    }

    const LayerNormV3TilingDataRegBaseTranspose* td_;
    TPipe* pipe_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQueue_;
    TQue<QuePosition::VECIN, PARAM_BUFFER_COUNT> paramsQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> meanQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> lastoutQueue_;
    TBuf<TPosition::VECCALC> tempBuf_;
    GlobalTensor<T> xGm_, yGm_;
    GlobalTensor<U> gammaGm_, betaGm_;
    GlobalTensor<M> meanGm_, lastoutGm_;
    LocalTensor<U> paramsLocal_;
    LocalTensor<float> xLocal_;
    LocalTensor<float> meanLocal_, lastoutLocal_;
    uint32_t tileA_, r_, rAlign_, paramAlign_;
    float scale_, powerOfTwo_;
};
} // namespace LayerNormV3
#endif // LAYER_NORM_V3_REGBASE_TRANSPOSE_H
