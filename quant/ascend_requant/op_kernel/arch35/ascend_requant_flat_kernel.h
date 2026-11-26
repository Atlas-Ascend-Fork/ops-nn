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
 * \file ascend_requant_flat_kernel.h
 * \brief Flat 1-D data path for AscendRequant: one contiguous x copy, one y copy and a req_scale staged
 *        by structural class (scalar / periodic pattern / contiguous tile) per tile.
 */

#ifndef ASCEND_REQUANT_FLAT_KERNEL_H
#define ASCEND_REQUANT_FLAT_KERNEL_H

#include "kernel_operator.h"
#include "ascend_requant_kernel.h"
#include "ascend_requant_flat_route.h"
#include "ascend_requant_tiling_data.h"

// One-off req_scale staging: int64 -> float with the baseline rule (CAST_RINT plus a 2^64 correction
// for scales whose int64 interpretation is negative).
template <typename ST>
__simd_vf__ inline void FlatScaleScalarVF(__ubuf__ float* dstAddr, __ubuf__ ST* srcAddr);

// Converts one prepared VL of raw req_scale into the float pattern and stores it as whole VL lanes:
// called with period == VL and groups == 1, so the single StoreAlign destination is the (32 B aligned)
// pattern base whatever the req_scale period is.
template <typename ST>
__simd_vf__ inline void FlatScalePatternVF(__ubuf__ float* dstAddr, __ubuf__ ST* srcAddr, uint32_t period,
                                           uint32_t groups);

// Per-tile elementwise requant. SCALE_MODE selects the req_scale operand:
// SCALAR -> one broadcast register from a single staged float,
// PERIODIC -> one prebuilt VL-wide float pattern register,
// CONTIG -> per-lane int64 scale staged for this tile with the baseline conversion chain.
template <bool DO_RELU, int64_t SCALE_MODE>
__simd_vf__ inline void FlatRequantVF(__ubuf__ int8_t* dstAddr, __ubuf__ int32_t* srcXAddr, __ubuf__ float* srcF32Addr,
                                      __ubuf__ uint64_t* srcU64Addr, uint32_t count, uint32_t oneRepeatSize,
                                      uint16_t repeatTimes);

// One contiguous GM -> UB block copy (blockLen in bytes, unpadded).
template <typename T>
__aicore__ inline void FlatCopyGmToUb(const AscendC::LocalTensor<T>& dst, const AscendC::GlobalTensor<T>& src,
                                      int64_t elemCount, int64_t elemOffset)
{
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = static_cast<uint32_t>(elemCount * static_cast<int64_t>(sizeof(T)));
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyPad(dst, src[elemOffset], extParams, padParams);
}

template <int64_t RANK, bool DO_RELU>
class AscendRequantFlatKernel {
    static constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(int32_t);
    static_assert(VL == static_cast<uint32_t>(ASCEND_REQUANT_FLAT_VEC_ELEMS),
                  "flat tiling classifies the req_scale period against the arch35 vector length");
    static_assert(static_cast<uint32_t>(ASCEND_REQUANT_FLAT_SCALE_INIT_BYTES) >=
                      VL * static_cast<uint32_t>(sizeof(uint64_t)),
                  "the periodic pattern expansion staging holds one VL of raw req_scale");

    using XT = int32_t;
    using ST = uint64_t;
    using YT = int8_t;

    AscendC::TPipe pipe_;
    const AscendRequantTilingData<RANK>* td_ = nullptr;
    AscendC::GlobalTensor<XT> gmX_;
    AscendC::GlobalTensor<ST> gmS_;
    AscendC::GlobalTensor<YT> gmY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufX_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufSTile_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufSInit_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufSPat_;
    int64_t route_ = ASCEND_REQUANT_ROUTE_LEGACY;
    int64_t flatElems_ = 0;
    int64_t tileElems_ = 0;
    int64_t period_ = 0;

public:
    __aicore__ inline void Init(GM_ADDR inputs[MAX_INPUT_SLOTS], GM_ADDR outputs[MAX_OUTPUT_SLOTS],
                                const AscendRequantTilingData<RANK>* td)
    {
        td_ = td;
        gmX_.SetGlobalBuffer((__gm__ XT*)inputs[0]);
        gmS_.SetGlobalBuffer((__gm__ ST*)inputs[1]);
        gmY_.SetGlobalBuffer((__gm__ YT*)outputs[0]);

        route_ = td_->flat.route;
        flatElems_ = td_->flat.flatElems;
        tileElems_ = td_->flat.tileElems;
        period_ = td_->flat.scalePeriod;

        pipe_.InitBuffer(bufX_, static_cast<uint32_t>(tileElems_ * static_cast<int64_t>(sizeof(XT)) +
                                                      ASCEND_REQUANT_FLAT_BUF_SLACK_BYTES));
        pipe_.InitBuffer(bufY_, static_cast<uint32_t>(tileElems_ * static_cast<int64_t>(sizeof(YT)) +
                                                      ASCEND_REQUANT_FLAT_BUF_SLACK_BYTES));
        if (route_ == ASCEND_REQUANT_ROUTE_CONTIG) {
            pipe_.InitBuffer(bufSTile_, static_cast<uint32_t>(tileElems_ * static_cast<int64_t>(sizeof(ST))));
        }
        pipe_.InitBuffer(bufSInit_, static_cast<uint32_t>(ASCEND_REQUANT_FLAT_SCALE_INIT_BYTES));
        pipe_.InitBuffer(bufSPat_, static_cast<uint32_t>(ASCEND_REQUANT_FLAT_PATTERN_BYTES));
        StageScale();
    }

    __aicore__ inline void Process()
    {
        int32_t evMTE2toV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        int32_t evVtoMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        int32_t evMTE3toMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        int64_t start = 0;
        int64_t end = 0;
        GetCoreRange(AscendC::GetBlockIdx(), td_->flat.tilesMain, td_->flat.coresTail, start, end);

        for (int64_t tile = start; tile < end; tile++) {
            int64_t first = tile * tileElems_;
            int64_t count = flatElems_ - first;
            if (count > tileElems_)
                count = tileElems_;

            if (tile != start)
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);

            CopyInX(first, count);
            if (route_ == ASCEND_REQUANT_ROUTE_CONTIG)
                CopyInScale(first, count);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);

            uint32_t elems = static_cast<uint32_t>(count);
            uint16_t repeatTimes = static_cast<uint16_t>(AscendC::CeilDivision(elems, VL));
            __ubuf__ YT* dstAddr = (__ubuf__ YT*)bufY_.Get<YT>().GetPhyAddr();
            __ubuf__ XT* srcXAddr = (__ubuf__ XT*)bufX_.Get<XT>().GetPhyAddr();
            __ubuf__ float* srcF32Addr = (__ubuf__ float*)bufSPat_.Get<float>().GetPhyAddr();
            if (route_ == ASCEND_REQUANT_ROUTE_CONTIG) {
                asc_vf_call<FlatRequantVF<DO_RELU, ASCEND_REQUANT_ROUTE_CONTIG>>(
                    dstAddr, srcXAddr, srcF32Addr, (__ubuf__ uint64_t*)bufSTile_.Get<ST>().GetPhyAddr(), elems, VL,
                    repeatTimes);
            } else if (route_ == ASCEND_REQUANT_ROUTE_SCALAR) {
                asc_vf_call<FlatRequantVF<DO_RELU, ASCEND_REQUANT_ROUTE_SCALAR>>(
                    dstAddr, srcXAddr, srcF32Addr, (__ubuf__ uint64_t*)srcF32Addr, elems, VL, repeatTimes);
            } else {
                asc_vf_call<FlatRequantVF<DO_RELU, ASCEND_REQUANT_ROUTE_PERIODIC>>(
                    dstAddr, srcXAddr, srcF32Addr, (__ubuf__ uint64_t*)srcF32Addr, elems, VL, repeatTimes);
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);

            CopyOutY(first, count);

            if (tile != end - 1)
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
        }
    }

private:
    __aicore__ inline void StageScale()
    {
        if (route_ == ASCEND_REQUANT_ROUTE_LEGACY || route_ == ASCEND_REQUANT_ROUTE_CONTIG)
            return;
        // tiling keeps this inside one VL of int64 for every non-CONTIG flat route.
        int64_t elems = td_->flat.scaleInitElems;
        if (elems < 1)
            elems = 1;

        FlatCopyGmToUb(bufSInit_.Get<ST>(), gmS_, elems, 0);

        __ubuf__ float* patAddr = (__ubuf__ float*)bufSPat_.Get<float>().GetPhyAddr();
        __ubuf__ ST* initAddr = (__ubuf__ ST*)bufSInit_.Get<ST>().GetPhyAddr();
        if (route_ == ASCEND_REQUANT_ROUTE_SCALAR) {
            int32_t evMTE2toV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            asc_vf_call<FlatScaleScalarVF<ST>>(patAddr, initAddr);
            return;
        }

        // PERIODIC: expand the broadcast map to one VL of raw req_scale on the scalar pipe, then convert
        // and store the whole VL pattern with a single 32 B aligned StoreAlign (see ExpandScalePattern).
        int32_t evMTE2toS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evMTE2toS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evMTE2toS);

        ExpandScalePattern();

        int32_t evStoV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(evStoV);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(evStoV);

        // period == VL: one full-mask pattern store at offset 0, so every StoreAlign destination is 32 B
        // aligned whatever the req_scale period is.
        asc_vf_call<FlatScalePatternVF<ST>>(patAddr, initAddr, static_cast<uint32_t>(VL), 1U);
    }

    // Lane l of the pattern is req_scale[map(l)] for the flat element l of every tile: tile bases are
    // 64-element aligned and the period divides 64, so all tiles share this pattern. Lanes are rewritten
    // from the highest down to keep reading the staged, not yet overwritten prefix.
    __aicore__ inline void ExpandScalePattern()
    {
        AscendC::LocalTensor<ST> init = bufSInit_.Get<ST>();
        int64_t period = period_;
        if (period < 1)
            period = 1;
        if (period > static_cast<int64_t>(VL))
            period = static_cast<int64_t>(VL);
        for (int64_t l = period - 1; l >= 0; --l) {
            int64_t idx = AscendRequantFlatScaleIndex(l, td_->maxBroShape, td_->inputStrides[1], RANK);
            init.SetValue(static_cast<uint32_t>(l), init.GetValue(static_cast<uint32_t>(idx)));
        }
        for (int64_t l = period; l < static_cast<int64_t>(VL); ++l) {
            init.SetValue(static_cast<uint32_t>(l), init.GetValue(static_cast<uint32_t>(l - period)));
        }
    }

    __aicore__ inline void CopyInX(int64_t first, int64_t count)
    {
        FlatCopyGmToUb(bufX_.Get<XT>(), gmX_, count, first);
    }

    // At most two contiguous int64 copies: the tile never exceeds one req_scale period.
    __aicore__ inline void CopyInScale(int64_t first, int64_t count)
    {
        int64_t offset = first % period_;
        int64_t head = period_ - offset;
        if (head > count)
            head = count;
        AscendC::LocalTensor<ST> dst = bufSTile_.Get<ST>();
        FlatCopyGmToUb(dst, gmS_, head, offset);
        if (count > head)
            FlatCopyGmToUb(dst[head], gmS_, count - head, 0);
    }

    __aicore__ inline void CopyOutY(int64_t first, int64_t count)
    {
        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(YT)));
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPad(gmY_[first], bufY_.Get<YT>(), extParams);
    }
};

// Route dispatcher: LEGACY keeps the untouched baseline kernel, every flat route runs the 1-D path.
// The two kernels are constructed one at a time because a TPipe owns the single UB pool.
template <int64_t RANK, bool DO_RELU>
class AscendRequantRouteKernel {
public:
    __aicore__ inline void Init(GM_ADDR inputs[MAX_INPUT_SLOTS], GM_ADDR outputs[MAX_OUTPUT_SLOTS],
                                const AscendRequantTilingData<RANK>* td)
    {
        for (int64_t i = 0; i < MAX_INPUT_SLOTS; i++)
            inputs_[i] = inputs[i];
        for (int64_t i = 0; i < MAX_OUTPUT_SLOTS; i++)
            outputs_[i] = outputs[i];
        td_ = td;
        flatRoute_ = (td->flat.route != ASCEND_REQUANT_ROUTE_LEGACY);
    }

    __aicore__ inline void Process()
    {
        if (flatRoute_) {
            AscendRequantFlatKernel<RANK, DO_RELU> kernel;
            kernel.Init(inputs_, outputs_, td_);
            kernel.Process();
        } else {
            AscendRequantKernel<RANK, DO_RELU> kernel;
            kernel.Init(inputs_, outputs_, td_);
            kernel.Process();
        }
    }

private:
    GM_ADDR inputs_[MAX_INPUT_SLOTS];
    GM_ADDR outputs_[MAX_OUTPUT_SLOTS];
    const AscendRequantTilingData<RANK>* td_ = nullptr;
    bool flatRoute_ = false;
};

template <typename ST>
__simd_vf__ inline void FlatScaleScalarVF(__ubuf__ float* dstAddr, __ubuf__ ST* srcAddr)
{
    AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo> i64Reg;
    AscendC::Reg::RegTensor<float> fp32SReg;
    AscendC::Reg::RegTensor<float> correctedSReg;
    uint32_t scalarOne = 1U;
    AscendC::Reg::MaskReg oneMask = AscendC::Reg::UpdateMask<uint32_t>(scalarOne);
    AscendC::Reg::MaskReg negMask;

    static constexpr float kUint64AsFloat = static_cast<float>(uint64_t(1) << 63) * 2.0f;

    AscendC::Reg::LoadAlign(i64Reg, (__ubuf__ int64_t*)srcAddr);
    AscendC::Reg::Cast<float, int64_t, kCastF32FromS64>(fp32SReg, i64Reg, oneMask);
    AscendC::Reg::Compares<int64_t, AscendC::CMPMODE::LT>(negMask, i64Reg, int64_t(0), oneMask);
    AscendC::Reg::Adds(correctedSReg, fp32SReg, kUint64AsFloat, negMask);
    AscendC::Reg::Select(fp32SReg, correctedSReg, fp32SReg, negMask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(dstAddr, fp32SReg, oneMask);
}

template <typename ST>
__simd_vf__ inline void FlatScalePatternVF(__ubuf__ float* dstAddr, __ubuf__ ST* srcAddr, uint32_t period,
                                           uint32_t groups)
{
    AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo> i64Reg;
    AscendC::Reg::RegTensor<float> fp32SReg;
    AscendC::Reg::RegTensor<float> correctedSReg;
    uint32_t scalarPeriod = period;
    AscendC::Reg::MaskReg periodMask = AscendC::Reg::UpdateMask<uint32_t>(scalarPeriod);
    AscendC::Reg::MaskReg negMask;

    static constexpr float kUint64AsFloat = static_cast<float>(uint64_t(1) << 63) * 2.0f;

    AscendC::Reg::LoadAlign(i64Reg, (__ubuf__ int64_t*)srcAddr);
    AscendC::Reg::Cast<float, int64_t, kCastF32FromS64>(fp32SReg, i64Reg, periodMask);
    AscendC::Reg::Compares<int64_t, AscendC::CMPMODE::LT>(negMask, i64Reg, int64_t(0), periodMask);
    AscendC::Reg::Adds(correctedSReg, fp32SReg, kUint64AsFloat, negMask);
    AscendC::Reg::Select(fp32SReg, correctedSReg, fp32SReg, negMask);
    for (uint32_t group = 0; group < groups; group++) {
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM>(dstAddr + group * period, fp32SReg,
                                                                            periodMask);
    }
}

template <bool DO_RELU, int64_t SCALE_MODE>
__simd_vf__ inline void FlatRequantVF(__ubuf__ int8_t* dstAddr, __ubuf__ int32_t* srcXAddr, __ubuf__ float* srcF32Addr,
                                      __ubuf__ uint64_t* srcU64Addr, uint32_t count, uint32_t oneRepeatSize,
                                      uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<int32_t> xReg;
    AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo> i64Reg;
    AscendC::Reg::RegTensor<float> fp32XReg;
    AscendC::Reg::RegTensor<float> fp32SReg;
    AscendC::Reg::RegTensor<float> correctedSReg;
    AscendC::Reg::RegTensor<float> scaledReg;
    AscendC::Reg::RegTensor<half> f16Reg;
    AscendC::Reg::RegTensor<int8_t> yReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::MaskReg negMask;
    AscendC::Reg::MaskReg storeMask;

    static constexpr float kUint64AsFloat = static_cast<float>(uint64_t(1) << 63) * 2.0f;

    constexpr uint32_t kCopyOutAlignElems = 32U / sizeof(int8_t);

    if constexpr (SCALE_MODE == ASCEND_REQUANT_ROUTE_SCALAR) {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(fp32SReg, srcF32Addr);
    } else if constexpr (SCALE_MODE == ASCEND_REQUANT_ROUTE_PERIODIC) {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(fp32SReg, srcF32Addr);
    }

    uint32_t sreg = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::Reg::UpdateMask<uint32_t>(sreg);

        AscendC::Reg::LoadAlign(xReg, srcXAddr + i * oneRepeatSize);
        AscendC::Reg::Cast<float, int32_t, kCastF32FromS32>(fp32XReg, xReg, mask);
        if constexpr (SCALE_MODE == ASCEND_REQUANT_ROUTE_CONTIG) {
            AscendC::Reg::LoadAlign(i64Reg, (__ubuf__ int64_t*)(srcU64Addr + i * oneRepeatSize));
            AscendC::Reg::Cast<float, int64_t, kCastF32FromS64>(fp32SReg, i64Reg, mask);
            AscendC::Reg::Compares<int64_t, AscendC::CMPMODE::LT>(negMask, i64Reg, int64_t(0), mask);
            AscendC::Reg::Adds(correctedSReg, fp32SReg, kUint64AsFloat, negMask);
            AscendC::Reg::Select(fp32SReg, correctedSReg, fp32SReg, negMask);
        }
        AscendC::Reg::Mul(scaledReg, fp32XReg, fp32SReg, mask);
        AscendC::Reg::Cast<half, float, kCastH16FromF32RndSat>(f16Reg, scaledReg, mask);
        AscendC::Reg::Cast<int8_t, half, kCastS8FromH16RndSat>(yReg, f16Reg, mask);
        constexpr bool kApplyMaxZero = DO_RELU;
        if constexpr (kApplyMaxZero) {
            AscendC::Reg::Maxs(yReg, yReg, int8_t(0), mask);
        }
        uint32_t storeCount = static_cast<uint32_t>(i + 1) * oneRepeatSize;
        uint32_t storeElems = (storeCount > count) ? (count - static_cast<uint32_t>(i) * oneRepeatSize) : oneRepeatSize;
        if (i + 1 == repeatTimes) {
            uint32_t aligned = (storeElems + kCopyOutAlignElems - 1U) / kCopyOutAlignElems * kCopyOutAlignElems;
            storeElems = (aligned < oneRepeatSize) ? aligned : oneRepeatSize;
        }
        uint32_t storeCnt = storeElems;
        storeMask = AscendC::Reg::UpdateMask<uint32_t>(storeCnt);
        AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(dstAddr + i * oneRepeatSize, yReg,
                                                                                  storeMask);
    }
}

#endif // ASCEND_REQUANT_FLAT_KERNEL_H
