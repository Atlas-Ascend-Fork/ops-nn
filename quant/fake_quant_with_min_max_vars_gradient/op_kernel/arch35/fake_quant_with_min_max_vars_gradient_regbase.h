/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fake_quant_with_min_max_vars_gradient_regbase.h
 * \brief fake_quant_with_min_max_vars_gradient kernel (regbase / arch35)
 *
 * Design:
 *  - min/max tensor inputs read via GetValue in Init
 *  - nudgedMin/nudgedMax computed on Device side (scalar, outside VF)
 *  - bothZero dispatched outside VF to ComputePassthrough / ComputeNormal
 *  - Vector compare+select for below_min/above_max (CMPMODE::GE/LE inversion)
 *  - Reg::Add accumulator + Reg::ReduceSum for vector reduce within VF
 *  - Per-iteration VF masks use the remaining element count, so tail tiles
 *    never feed uninitialized UB lanes into the accumulators
 *  - Deterministic multi-core reduce (two-stage, no atomics):
 *      stage 1: each block folds its tile sums into a running UB partial and
 *               writes it once to workspace: [blockIdx] for min,
 *               [numCore + blockIdx] for max (plain copy, fixed address)
 *      barrier: PipeBarrier<PIPE_MTE3> + SyncAll
 *      stage 2: block 0 reads all partials back and combines them in fixed
 *               ascending blockIdx order with Kahan compensation, then
 *               single-writes backprops_wrt_min / backprops_wrt_max
 *    SetAtomicAdd is intentionally NOT used: float atomic accumulation order
 *    is not controlled, which made the outputs vary run to run (MD5 mismatch
 *    across runs) and drift vs golden/third-party reductions.
 */

#ifndef FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_REGBASE_H_
#define FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_REGBASE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "fake_quant_with_min_max_vars_gradient_tilingdata.h"

namespace FakeQuantWithMinMaxVarsGradient {
using namespace AscendC;

constexpr uint32_t FQ_GRAD_BUFFER_NUM = 2;
constexpr uint32_t FQ_GRAD_FP32_SIGN_MASK = 0x80000000u;

template <uint64_t SchMode>
class FakeQuantWithMinMaxVarsGradientRegbase {
public:
    __aicore__ inline FakeQuantWithMinMaxVarsGradientRegbase(
        const FakeQuantWithMinMaxVarsGradientTilingData* tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR gradients, GM_ADDR x, GM_ADDR minGm, GM_ADDR maxGm, GM_ADDR backpropsWrtX,
                                GM_ADDR backpropsWrtMin, GM_ADDR backpropsWrtMax, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t dataCount);
    __aicore__ inline void InitAccumulators();
    __aicore__ inline void ComputePassthrough(int64_t dataCount);
    __aicore__ inline void ComputeNormal(int64_t dataCount);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t dataCount);
    __aicore__ inline void WriteBlockPartial();
    __aicore__ inline void FinalReduce();

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, FQ_GRAD_BUFFER_NUM> inQueueG_;
    TQue<QuePosition::VECIN, FQ_GRAD_BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, FQ_GRAD_BUFFER_NUM> outQueueY_;
    TBuf<TPosition::VECCALC> accBuf_; // [0]=running min partial, [VL]=running max partial
    TBuf<TPosition::VECCALC> wsBuf_;  // stage-2 staging: all 2*numCore block partials
    GlobalTensor<float> gGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    GlobalTensor<float> minOutGm_;
    GlobalTensor<float> maxOutGm_;
    GlobalTensor<float> wsGm_;

    const FakeQuantWithMinMaxVarsGradientTilingData* tilingData_;
    int32_t blockIdx_ = 0;
    int64_t blockOffset_ = 0;
    int64_t blockLen_ = 0;

    bool bothZero_ = false;
    float nudgedMin_ = 0.0f;
    float nudgedMax_ = 0.0f;
};

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::Init(GM_ADDR gradients, GM_ADDR x,
                                                                             GM_ADDR minGm, GM_ADDR maxGm,
                                                                             GM_ADDR backpropsWrtX,
                                                                             GM_ADDR backpropsWrtMin,
                                                                             GM_ADDR backpropsWrtMax, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();

    gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradients));
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(backpropsWrtX));
    minOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(backpropsWrtMin));
    maxOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(backpropsWrtMax));
    // WHY: the kernel workspace head is a CANN system-reserved region (used by SyncAll etc.);
    // user partials must start at GetUserWorkspace. Writing from the raw base would collide with
    // the reserved region and overrun the tiling-declared workspace (same lesson as the
    // per_channel_gradient sibling, PR review finding).
    wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(AscendC::GetUserWorkspace(workspace)));

    if (GetBlockIdx() == 0) {
        InitGlobalMemory<float>(maxOutGm_, 1, static_cast<float>(0));
        InitGlobalMemory<float>(minOutGm_, 1, static_cast<float>(0));
    }

    GlobalTensor<float> minTensor, maxTensor;
    minTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(minGm));
    maxTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(maxGm));
    float fMin = minTensor.GetValue(0);
    float fMax = maxTensor.GetValue(0);

    int64_t numBits = tilingData_->numBits;
    bool narrowRange = tilingData_->narrowRange;

    float quantMin = narrowRange ? 1.0f : 0.0f;
    float quantMax = static_cast<float>((1ULL << static_cast<uint32_t>(numBits)) - 1ULL);
    float scale = (fMax - fMin) / (quantMax - quantMin);
    float zpFromMin = quantMin - fMin / scale;

    float nudgedZp;
    if (zpFromMin <= quantMin) {
        nudgedZp = quantMin;
    } else if (zpFromMin >= quantMax) {
        nudgedZp = quantMax;
    } else {
        float val = zpFromMin + 0.5f;
        int intPart = (int)val;
        if (val < 0.0f && val != (float)intPart) {
            intPart -= 1;
        }
        nudgedZp = (float)intPart;
    }

    nudgedMin_ = (quantMin - nudgedZp) * scale;
    nudgedMax_ = (quantMax - nudgedZp) * scale;

    float absMin = (fMin >= 0.0f) ? fMin : -fMin;
    float absMax = (fMax >= 0.0f) ? fMax : -fMax;
    bothZero_ = (absMin + absMax) < 1e-10f;

    int64_t blockFactor = tilingData_->blockFactor;
    int64_t blockTail = tilingData_->blockTailFactor;
    int64_t numCore = tilingData_->numCore;

    if (blockIdx_ >= numCore) {
        blockOffset_ = 0;
        blockLen_ = 0;
    } else if (blockIdx_ == numCore - 1) {
        blockOffset_ = blockIdx_ * blockFactor;
        blockLen_ = blockTail;
    } else {
        blockOffset_ = blockIdx_ * blockFactor;
        blockLen_ = blockFactor;
    }

    // UB budget: 3 queues (g, x, y) x 2 buffers = 6 slots, plus the reduce buffers
    int64_t baseLen = tilingData_->baseLen;
    pipe_.InitBuffer(inQueueG_, FQ_GRAD_BUFFER_NUM, baseLen * sizeof(float));
    pipe_.InitBuffer(inQueueX_, FQ_GRAD_BUFFER_NUM, baseLen * sizeof(float));
    pipe_.InitBuffer(outQueueY_, FQ_GRAD_BUFFER_NUM, baseLen * sizeof(float));

    const uint32_t vlFloats = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    // two full-register load regions (min partial at [0], max partial at [VL])
    pipe_.InitBuffer(accBuf_, 2 * vlFloats * sizeof(float));
    const int64_t wsCore = (numCore > 0) ? numCore : 1;
    pipe_.InitBuffer(wsBuf_, 2 * wsCore * sizeof(float));

    SyncAll();
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::Process()
{
    int64_t totalLen = tilingData_->totalLen;

    if (totalLen == 0) {
        // uniform for all (one) launched blocks; outputs stay zeroed in Init
        return;
    }

    if (!bothZero_) {
        InitAccumulators();
    }

    int64_t baseLen = tilingData_->baseLen;
    int64_t loopNum = blockLen_ / baseLen;
    int64_t loopTail = blockLen_ % baseLen;
    int64_t base = blockOffset_;

    for (int64_t i = 0; i < loopNum; ++i) {
        int64_t off = base + i * baseLen;
        CopyIn(off, baseLen);
        if (bothZero_) {
            ComputePassthrough(baseLen);
        } else {
            ComputeNormal(baseLen);
        }
        CopyOut(off, baseLen);
    }
    if (loopTail > 0) {
        int64_t off = base + loopNum * baseLen;
        CopyIn(off, loopTail);
        if (bothZero_) {
            ComputePassthrough(loopTail);
        } else {
            ComputeNormal(loopTail);
        }
        CopyOut(off, loopTail);
    }

    if (bothZero_) {
        // min/max grads stay 0; keep barrier participation uniform across blocks
        SyncAll();
        return;
    }

    // stage 1: publish this block's partials at fixed workspace slots.
    // Only blocks within [0, numCore) own a workspace slot; if the framework ever
    // launches more blocks than numCore, the extras must not touch ws (would write
    // past the 2*numCore user area) — they still join the barrier below (review).
    if (blockIdx_ < tilingData_->numCore) {
        WriteBlockPartial();
        // drain this core's MTE3 before the cross-core barrier so the slots are visible
        AscendC::PipeBarrier<PIPE_MTE3>();
    }
    SyncAll();

    // stage 2: single writer combines all partials in fixed order
    if (blockIdx_ == 0) {
        FinalReduce();
    }
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::CopyIn(int64_t gmOffset, int64_t dataCount)
{
    LocalTensor<float> gLocal = inQueueG_.AllocTensor<float>();
    LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(dataCount * sizeof(float));
    params.srcStride = 0;
    params.dstStride = 0;
    params.rsv = 0;
    DataCopyPadExtParams<float> padParams = {false, 0, 0, 0};
    DataCopyPad(gLocal, gGm_[gmOffset], params, padParams);
    DataCopyPad(xLocal, xGm_[gmOffset], params, padParams);
    inQueueG_.EnQue(gLocal);
    inQueueX_.EnQue(xLocal);
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::CopyOut(int64_t gmOffset, int64_t dataCount)
{
    LocalTensor<float> yLocal = outQueueY_.DeQue<float>();
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(dataCount * sizeof(float));
    params.srcStride = 0;
    params.dstStride = 0;
    params.rsv = 0;
    DataCopyPad(yGm_[gmOffset], yLocal, params);
    outQueueY_.FreeTensor(yLocal);
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::InitAccumulators()
{
    LocalTensor<float> accLocal = accBuf_.Get<float>();
    __ubuf__ float* accMinAddr = (__ubuf__ float*)accLocal.GetPhyAddr();
    const uint32_t vlFloats = AscendC::VECTOR_REG_WIDTH / sizeof(float);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vZero;
        AscendC::Reg::Duplicate(vZero, 0.0f);
        AscendC::Reg::MaskReg vl1Mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(accMinAddr, vZero, vl1Mask);
        AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(accMinAddr + vlFloats, vZero, vl1Mask);
    }
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::ComputePassthrough(int64_t dataCount)
{
    LocalTensor<float> gLocal = inQueueG_.DeQue<float>();
    LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();

    __ubuf__ float* gAddr = (__ubuf__ float*)gLocal.GetPhyAddr();
    __ubuf__ float* yAddr = (__ubuf__ float*)yLocal.GetPhyAddr();

    const uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t vfLoopNum = static_cast<uint16_t>((dataCount + VL - 1) / VL);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vG;
        AscendC::Reg::MaskReg mask;

        uint32_t remaining = static_cast<uint32_t>(dataCount);
        for (uint16_t i = 0; i < vfLoopNum; ++i) {
            uint32_t procCount = (remaining > static_cast<uint32_t>(VL)) ? static_cast<uint32_t>(VL) : remaining;
            // UpdateMask consumes its by-reference count (post-update, see the sibling
            // per_channel_gradient usage), so pass a copy and decrement with the intact value.
            uint32_t maskCount = procCount;
            mask = AscendC::Reg::UpdateMask<float>(maskCount);
            remaining -= procCount;
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vG, gAddr + i * VL);
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + i * VL, vG, mask);
        }
    }

    outQueueY_.EnQue(yLocal);
    inQueueG_.FreeTensor(gLocal);
    inQueueX_.FreeTensor(xLocal);
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::ComputeNormal(int64_t dataCount)
{
    LocalTensor<float> gLocal = inQueueG_.DeQue<float>();
    LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();
    LocalTensor<float> accLocal = accBuf_.Get<float>();

    __ubuf__ float* gAddr = (__ubuf__ float*)gLocal.GetPhyAddr();
    __ubuf__ float* xAddr = (__ubuf__ float*)xLocal.GetPhyAddr();
    __ubuf__ float* yAddr = (__ubuf__ float*)yLocal.GetPhyAddr();
    __ubuf__ float* accMinAddr = (__ubuf__ float*)accLocal.GetPhyAddr();
    const uint32_t vlFloats = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    __ubuf__ float* accMaxAddr = accMinAddr + vlFloats;

    const uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t vfLoopNum = static_cast<uint16_t>((dataCount + VL - 1) / VL);
    const float nudgedMin = nudgedMin_;
    const float nudgedMax = nudgedMax_;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vG;
        AscendC::Reg::RegTensor<float> vX;
        AscendC::Reg::RegTensor<float> vZero;
        AscendC::Reg::RegTensor<float> vOne;
        AscendC::Reg::RegTensor<float> vMask;
        AscendC::Reg::RegTensor<float> vOut;
        AscendC::Reg::RegTensor<float> vBelow;
        AscendC::Reg::RegTensor<float> vAbove;
        AscendC::Reg::RegTensor<uint32_t> vSignMask;
        AscendC::Reg::RegTensor<uint32_t> vSignG;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::MaskReg geMask;
        AscendC::Reg::MaskReg leMask;
        AscendC::Reg::RegTensor<float> vAccMin;
        AscendC::Reg::RegTensor<float> vAccMax;

        AscendC::Reg::Duplicate(vZero, 0.0f);
        AscendC::Reg::Duplicate(vOne, 1.0f);
        AscendC::Reg::Duplicate(vSignMask, FQ_GRAD_FP32_SIGN_MASK);
        // per-tile accumulators must start from zero: the loop below reads them before the
        // running-partial reload (which happens only after ReduceSum) — PR review finding.
        AscendC::Reg::Duplicate(vAccMin, 0.0f);
        AscendC::Reg::Duplicate(vAccMax, 0.0f);

        AscendC::Reg::MaskReg preg1 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

        uint32_t remaining = static_cast<uint32_t>(dataCount);
        for (uint16_t i = 0; i < vfLoopNum; ++i) {
            uint32_t procCount = (remaining > static_cast<uint32_t>(VL)) ? static_cast<uint32_t>(VL) : remaining;
            // UpdateMask consumes its by-reference count (post-update), so pass a copy.
            uint32_t maskCount = procCount;
            mask = AscendC::Reg::UpdateMask<float>(maskCount);
            remaining -= procCount;

            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vG, gAddr + i * VL);
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vX, xAddr + i * VL);

            // mask = (x >= nudgedMin) && (x <= nudgedMax)
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GE>(geMask, vX, nudgedMin, mask);
            AscendC::Reg::Select<float>(vMask, vOne, vZero, geMask);
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LE>(leMask, vX, nudgedMax, mask);
            AscendC::Reg::Select<float>(vMask, vMask, vZero, leMask);

            // backprops_wrt_x = gradients * mask + sign-bit re-injection
            AscendC::Reg::Mul(vOut, vG, vMask, mask);
            AscendC::Reg::And(vSignG, (AscendC::Reg::RegTensor<uint32_t>&)vG, vSignMask, mask);
            AscendC::Reg::Or((AscendC::Reg::RegTensor<uint32_t>&)vOut, (AscendC::Reg::RegTensor<uint32_t>&)vOut, vSignG,
                             mask);
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + i * VL, vOut, mask);

            // below_min: x < nudgedMin -> Select(vG, vZero, ltMask)
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(geMask, vX, nudgedMin, mask);
            AscendC::Reg::Select<float>(vBelow, vG, vZero, geMask);

            // above_max: x > nudgedMax -> Select(vG, vZero, gtMask)
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GT>(leMask, vX, nudgedMax, mask);
            AscendC::Reg::Select<float>(vAbove, vG, vZero, leMask);

            // Vector accumulate across all vf iterations (use ALL mask)
            AscendC::Reg::Add(vAccMin, vAccMin, vBelow, preg1);
            AscendC::Reg::Add(vAccMax, vAccMax, vAbove, preg1);
        }

        // ReduceSum: VL lanes -> 1 scalar in lane 0
        AscendC::Reg::RegTensor<float> vSumMin;
        AscendC::Reg::RegTensor<float> vSumMax;
        AscendC::Reg::ReduceSum<float>(vSumMin, vAccMin, preg1);
        AscendC::Reg::ReduceSum<float>(vSumMax, vAccMax, preg1);

        // Fold the tile sums into this block's running partials (lane 0)
        AscendC::Reg::MaskReg vl1Mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vAccMin, accMinAddr);
        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vAccMax, accMaxAddr);
        AscendC::Reg::Add<float>(vAccMin, vAccMin, vSumMin, vl1Mask);
        AscendC::Reg::Add<float>(vAccMax, vAccMax, vSumMax, vl1Mask);
        AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(accMinAddr, vAccMin, vl1Mask);
        AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(accMaxAddr, vAccMax, vl1Mask);
    }

    outQueueY_.EnQue(yLocal);
    inQueueG_.FreeTensor(gLocal);
    inQueueX_.FreeTensor(xLocal);
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::WriteBlockPartial()
{
    LocalTensor<float> accLocal = accBuf_.Get<float>();
    const uint32_t vlFloats = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    const int64_t numCore = tilingData_->numCore;

    // make the VF-written running partials visible to the MTE3 copies
    const event_t eventVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVMte3);
    WaitFlag<HardEvent::V_MTE3>(eventVMte3);

    DataCopyExtParams pOne;
    pOne.blockCount = 1;
    pOne.blockLen = static_cast<uint32_t>(sizeof(float));
    pOne.srcStride = 0;
    pOne.dstStride = 0;
    pOne.rsv = 0;

    DataCopyPad(wsGm_[blockIdx_], accLocal, pOne);
    DataCopyPad(wsGm_[numCore + blockIdx_], accLocal[vlFloats], pOne);
}

template <uint64_t SchMode>
__aicore__ inline void FakeQuantWithMinMaxVarsGradientRegbase<SchMode>::FinalReduce()
{
    const int64_t numCore = tilingData_->numCore;
    LocalTensor<float> wsLocal = wsBuf_.Get<float>();
    LocalTensor<float> accLocal = accBuf_.Get<float>();
    const uint32_t vlFloats = AscendC::VECTOR_REG_WIDTH / sizeof(float);

    // copy all block partials in: ws[0..numCore) = min, ws[numCore..2*numCore) = max
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(2 * numCore * sizeof(float));
    params.srcStride = 0;
    params.dstStride = 0;
    params.rsv = 0;
    DataCopyPadExtParams<float> padParams = {false, 0, 0, 0};
    DataCopyPad(wsLocal, wsGm_[0], params, padParams);

    const event_t eventMte2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventMte2S);
    WaitFlag<HardEvent::MTE2_S>(eventMte2S);

    // Kahan-compensated sum in fixed ascending blockIdx order (deterministic,
    // and near-exact so the result stays within third-party distance to golden)
    float sumMin = 0.0f;
    float cMin = 0.0f;
    for (int64_t i = 0; i < numCore; ++i) {
        float y = wsLocal.GetValue(i) - cMin;
        float t = sumMin + y;
        cMin = (t - sumMin) - y;
        sumMin = t;
    }
    float sumMax = 0.0f;
    float cMax = 0.0f;
    for (int64_t i = 0; i < numCore; ++i) {
        float y = wsLocal.GetValue(numCore + i) - cMax;
        float t = sumMax + y;
        cMax = (t - sumMax) - y;
        sumMax = t;
    }

    accLocal.SetValue(0, sumMin);
    accLocal.SetValue(static_cast<int32_t>(vlFloats), sumMax);

    const event_t eventSMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventSMte3);
    WaitFlag<HardEvent::S_MTE3>(eventSMte3);

    DataCopyExtParams pOne;
    pOne.blockCount = 1;
    pOne.blockLen = static_cast<uint32_t>(sizeof(float));
    pOne.srcStride = 0;
    pOne.dstStride = 0;
    pOne.rsv = 0;

    DataCopyPad(minOutGm_[0], accLocal, pOne);
    DataCopyPad(maxOutGm_[0], accLocal[vlFloats], pOne);
}

} // namespace FakeQuantWithMinMaxVarsGradient

#endif // FAKE_QUANT_WITH_MIN_MAX_VARS_GRADIENT_REGBASE_H_
