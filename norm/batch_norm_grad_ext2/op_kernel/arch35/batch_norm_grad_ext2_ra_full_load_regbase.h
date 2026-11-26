/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_grad_ext2_ra_full_load_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_EXT2_RA_FULL_LOAD_REGBASE_H__
#define __BATCH_NORM_GRAD_EXT2_RA_FULL_LOAD_REGBASE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm_grad_ext2_common.h"
#include "batch_norm_grad_ext2_comp_sum.h"

namespace BatchNormGradExt2 {
using namespace AscendC;
using namespace BatchNormGradExt2::Comp;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MemType;

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1>
class BatchNormGradExt2RAFullLoad {
public:
    __aicore__ inline BatchNormGradExt2RAFullLoad(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(__gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* mean, __gm__ uint8_t* rstd,
                                __gm__ uint8_t* gamma, __gm__ uint8_t* dx, __gm__ uint8_t* dgamma,
                                __gm__ uint8_t* dbeta, __gm__ uint8_t* workspace,
                                const BatchNormGradExt2RAFullLoadTilingData* tilingData)
    {
        const BatchNormGradExt2BaseTilingData& baseTilingData = tilingData->baseTilingData;
        rDim_ = baseTilingData.r1Dim;
        aDim_ = baseTilingData.aDim;
        numBlocks = tilingData->numBlocks;
        mainBlockCount = tilingData->mainBlockCount;
        mainBlockFactor = tilingData->mainBlockFactor;
        tailBlockFactor = tilingData->tailBlockFactor;

        reduceLoopTimes = tilingData->reduceLoopTimes;
        reduceRecursionLoop = tilingData->reduceRecursionLoop;
        foldLoopStep1 = tilingData->foldLoopStep1;
        foldLoopStep2 = tilingData->foldLoopStep2;
        foldLoopStep3 = tilingData->foldLoopStep3;
        foldLoopOffset1 = tilingData->foldLoopOffset1;
        foldLoopOffset2 = tilingData->foldLoopOffset2;
        foldLoopOffset3 = tilingData->foldLoopOffset3;

        // calc ub tiling
        int64_t blockFactor = GetBlockIdx() < mainBlockCount ? mainBlockFactor : tailBlockFactor;
        int64_t mainALoopFactorAligned = tilingData->mainALoopFactorAligned;
        int64_t tailALoopFactorAligned = tilingData->tailALoopFactorAligned;
        aLoopFactor = GetBlockIdx() < mainBlockCount ? tilingData->mainALoopFactor : tilingData->tailALoopFactor;
        aLoopFactorAligned = GetBlockIdx() < mainBlockCount ? mainALoopFactorAligned : tailALoopFactorAligned;
        aLoopCount = blockFactor / aLoopFactor;
        aLoopTail = blockFactor - aLoopCount * aLoopFactor;

        int64_t offset = tilingData->mainBlockFactor * GetBlockIdx();
        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)dy + offset);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)x + offset);
        meanGm_.SetGlobalBuffer((__gm__ float*)mean + offset);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + offset);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)gamma + offset);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)dx + offset);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)dgamma + offset);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)dbeta + offset);

        int64_t feat_bytes = rDim_ * aLoopFactorAligned * sizeof(DY_TYPE);
        int64_t para_bytes = aLoopFactorAligned * sizeof(float);
        int64_t gamma_bytes = aLoopFactorAligned * sizeof(WEIGHT_TYPE);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, feat_bytes);
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, feat_bytes);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, para_bytes);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, para_bytes);
        pipe_->InitBuffer(gammaInQue_, BUFFER_NUM, gamma_bytes);
        pipe_->InitBuffer(dxOutQue_, BUFFER_NUM, feat_bytes);
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, gamma_bytes);
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, gamma_bytes);

        if (rDim_ > FOUR) {
            int64_t ws_bytes = (foldLoopStep1 + foldLoopStep2 + foldLoopStep3) * aLoopFactorAligned * sizeof(float);
            pipe_->InitBuffer(dbetaBinaryAddBuf_, ws_bytes);
            pipe_->InitBuffer(dgammaBinaryAddBuf_, ws_bytes);
        }
    }

    __aicore__ inline void Process()
    {
        if (aLoopFactor <= 0) {
            return;
        }
        if (GetBlockIdx() >= numBlocks) {
            return;
        }
        for (int64_t idx = 0; idx < aLoopCount; ++idx) {
            ProcessBlock(idx * aLoopFactor, aLoopFactor, aLoopFactorAligned);
        }
        if (aLoopTail > 0) {
            ProcessBlock(aLoopCount * aLoopFactor, aLoopTail, aLoopFactorAligned);
        }
    }

public:
    __aicore__ inline void ProcessBlock(const int64_t gmOffset, const int64_t factor, const int64_t stride)
    {
        /* Process one Block */
        // load dy
        LocalTensor<DY_TYPE> dy = dyInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(dy, dyGm_[gmOffset], rDim_, factor, stride, aDim_);
        dyInQue_.EnQue(dy);
        dy = dyInQue_.template DeQue<DY_TYPE>();

        // load x
        LocalTensor<DY_TYPE> x = xInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(x, xGm_[gmOffset], rDim_, factor, stride, aDim_);
        xInQue_.EnQue(x);
        x = xInQue_.template DeQue<DY_TYPE>();

        // load mean
        LocalTensor<float> mean = meanInQue_.template AllocTensor<float>();
        CopyIn(mean, meanGm_[gmOffset], 1, factor, 0, aDim_);
        meanInQue_.EnQue(mean);
        mean = meanInQue_.template DeQue<float>();

        // load rstd
        LocalTensor<float> rstd = rstdInQue_.template AllocTensor<float>();
        CopyIn(rstd, rstdGm_[gmOffset], 1, factor, 0, aDim_);
        rstdInQue_.EnQue(rstd);
        rstd = rstdInQue_.template DeQue<float>();

        // load gamma
        LocalTensor<WEIGHT_TYPE> gamma = gammaInQue_.template AllocTensor<WEIGHT_TYPE>();
        CopyIn(gamma, gammaGm_[gmOffset], 1, factor, 0, aDim_);
        gammaInQue_.EnQue(gamma);
        gamma = gammaInQue_.template DeQue<WEIGHT_TYPE>();

        // calc
        LocalTensor<WEIGHT_TYPE> dbeta = dbetaOutQue_.template AllocTensor<WEIGHT_TYPE>();
        LocalTensor<WEIGHT_TYPE> dgamma = dgammaOutQue_.template AllocTensor<WEIGHT_TYPE>();
        LocalTensor<DY_TYPE> dx = dxOutQue_.template AllocTensor<DY_TYPE>();
        float reciprocal = 1.0f / (float)rDim_;

        // 补偿累加 + dx：lane = 通道，R 轴沿 buffer 行方向累加到同一 lane
        // （不跨 lane，因此不需要 CompReduceLanes）。dbeta = sum(dy)，
        // dgamma = rstd * (sum(x*dy) - mean * sum(dy))，只在收尾舍入一次。
        CalcDbetaDgammaComp(dy, x, mean, rstd, dbeta, dgamma, factor, rDim_, stride);
        CalcDxBlock(dy, x, mean, rstd, gamma, dbeta, dgamma, dx, factor, rDim_, stride, reciprocal);

        dyInQue_.FreeTensor(dy);
        xInQue_.FreeTensor(x);
        meanInQue_.FreeTensor(mean);
        rstdInQue_.FreeTensor(rstd);
        gammaInQue_.FreeTensor(gamma);

        // store dbeta
        dbetaOutQue_.EnQue(dbeta);
        dbeta = dbetaOutQue_.template DeQue<WEIGHT_TYPE>();
        CopyOut(dbetaGm_[gmOffset], dbeta, factor);
        dbetaOutQue_.FreeTensor(dbeta);

        // store dgamma
        dgammaOutQue_.EnQue(dgamma);
        dgamma = dgammaOutQue_.template DeQue<WEIGHT_TYPE>();
        CopyOut(dgammaGm_[gmOffset], dgamma, factor);
        dgammaOutQue_.FreeTensor(dgamma);

        // store dx
        dxOutQue_.EnQue(dx);
        dx = dxOutQue_.template DeQue<DY_TYPE>();
        CopyOut(dxGm_[gmOffset], dx, rDim_, factor, aDim_, stride);
        dxOutQue_.FreeTensor(dx);
    }

    __aicore__ inline void CalcDbetaDgammaComp(const LocalTensor<DY_TYPE>& dyLocal, const LocalTensor<DY_TYPE>& xLocal,
                                               const LocalTensor<float>& mean, const LocalTensor<float>& rstd,
                                               const LocalTensor<WEIGHT_TYPE>& dbeta,
                                               const LocalTensor<WEIGHT_TYPE>& dgamma, int64_t factor, int64_t rDim,
                                               int64_t stride)
    {
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal.GetPhyAddr();
        __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xLocal.GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* dbetaAddr = (__local_mem__ WEIGHT_TYPE*)dbeta.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* dgammaAddr = (__local_mem__ WEIGHT_TYPE*)dgamma.GetPhyAddr();
        uint16_t outerLoop = static_cast<uint16_t>((factor + VL_FP32 - 1) / VL_FP32);

        __VEC_SCOPE__
        {
            MicroAPI::MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::RegTensor<float> s0Hi, s0Lo, s0Lo2, s1Hi, s1Lo, s1Lo2;
            MicroAPI::RegTensor<float> dyReg, xReg, prodHi, meanReg, rstdReg, dgFinal, dbFinal;
            uint32_t count = static_cast<uint32_t>(factor);
            for (uint16_t i = 0; i < outerLoop; ++i) {
                MicroAPI::MaskReg pMask = MicroAPI::UpdateMask<float>(count);
                uint32_t laneOffset = i * VL_FP32;
                LoadOneTensor<float>(meanAddr, meanReg, pMask, laneOffset);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, laneOffset);
                MicroAPI::Duplicate(s0Hi, 0.0f, all);
                MicroAPI::Duplicate(s0Lo, 0.0f, all);
                MicroAPI::Duplicate(s0Lo2, 0.0f, all);
                MicroAPI::Duplicate(s1Hi, 0.0f, all);
                MicroAPI::Duplicate(s1Lo, 0.0f, all);
                MicroAPI::Duplicate(s1Lo2, 0.0f, all);
                for (uint16_t j = 0; j < static_cast<uint16_t>(rDim); ++j) {
                    uint32_t offset = laneOffset + j * static_cast<uint32_t>(stride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, offset);
                    CompAddWide<DY_TYPE>(s0Hi, s0Lo, s0Lo2, dyReg, pMask);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, offset);
                    if constexpr (IsSameType<DY_TYPE, float>::value) {
                        CompAddProductWide<DY_TYPE>(s1Hi, s1Lo, s1Lo2, xReg, dyReg, pMask);
                    } else {
                        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(prodHi, xReg, dyReg, pMask);
                        CompAddWide<DY_TYPE>(s1Hi, s1Lo, s1Lo2, prodHi, pMask);
                    }
                }
                CompFoldLo2(s0Lo, s0Lo2, pMask);
                CompFoldLo2(s1Lo, s1Lo2, pMask);
                CompFinalizeDgammaDbeta(s1Hi, s1Lo, s0Hi, s0Lo, meanReg, rstdReg, pMask, dgFinal, dbFinal);
                StoreOneTensor<WEIGHT_TYPE>(dbetaAddr, dbFinal, pMask, laneOffset);
                StoreOneTensor<WEIGHT_TYPE>(dgammaAddr, dgFinal, pMask, laneOffset);
            }
        }
    }

    __aicore__ inline void CalcDxBlock(const LocalTensor<DY_TYPE>& dyLocal, const LocalTensor<DY_TYPE>& xLocal,
                                       const LocalTensor<float>& mean, const LocalTensor<float>& rstd,
                                       const LocalTensor<WEIGHT_TYPE>& gamma, const LocalTensor<WEIGHT_TYPE>& dbeta,
                                       const LocalTensor<WEIGHT_TYPE>& dgamma, const LocalTensor<DY_TYPE>& dxLocal,
                                       int64_t factor, int64_t rDim, int64_t stride, float reciprocal)
    {
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal.GetPhyAddr();
        __local_mem__ float* xAddr = (__local_mem__ float*)xLocal.GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* gammaAddr = (__local_mem__ WEIGHT_TYPE*)gamma.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* dbetaAddr = (__local_mem__ WEIGHT_TYPE*)dbeta.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* dgammaAddr = (__local_mem__ WEIGHT_TYPE*)dgamma.GetPhyAddr();
        __local_mem__ DY_TYPE* dxAddr = (__local_mem__ DY_TYPE*)dxLocal.GetPhyAddr();
        uint16_t outerLoop = static_cast<uint16_t>((factor + VL_FP32 - 1) / VL_FP32);
        float hNegDelta = ExactDivisorNegDelta(rDim); // -(R - fl32(R))，rDim 即精确整数 R

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> meanReg, rstdReg, gammaReg, dbetaReg, dgammaReg;
            MicroAPI::RegTensor<float> dyReg, xReg, u, p, q, t, dxReg;
            MicroAPI::RegTensor<float> rhiReg, rlo, dbetaOverR, pMag;
            MicroAPI::MaskReg pFinite;
            MicroAPI::MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            // 与 golden/torch 同构：dx = ((dy - dbeta/R) - (xhat*dgamma)/R) * gamma * rstd；
            // 除法用 (rhi + rlo) 的 FMA 形式（单次舍入，等价真除法）；rlo 按**精确整数 R** 补残差
            // （见 comp_sum.h：只修正到 fl32(R) 时整通道带 (R-fl32(R))/R 的同号偏差）。
            MicroAPI::Duplicate(rhiReg, reciprocal, all);
            CompExactDivisorReciprocalLo(rlo, rhiReg, static_cast<float>(rDim), hNegDelta, all);
            uint32_t count = static_cast<uint32_t>(factor);
            for (uint16_t i = 0; i < outerLoop; ++i) {
                MicroAPI::MaskReg pMask = MicroAPI::UpdateMask<float>(count);
                uint32_t laneOffset = i * VL_FP32;
                LoadOneTensor<float>(meanAddr, meanReg, pMask, laneOffset);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, laneOffset);
                LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, laneOffset);
                LoadOneTensor<WEIGHT_TYPE>(dbetaAddr, dbetaReg, pMask, laneOffset);
                LoadOneTensor<WEIGHT_TYPE>(dgammaAddr, dgammaReg, pMask, laneOffset);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dbetaOverR, dbetaReg, rlo, pMask);
                MicroAPI::MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(dbetaOverR, dbetaReg, rhiReg, pMask);
                for (uint16_t j = 0; j < static_cast<uint16_t>(rDim); ++j) {
                    uint32_t offset = laneOffset + j * static_cast<uint32_t>(stride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, offset);
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(u, dyReg, dbetaOverR, pMask);
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(p, xReg, rstdReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(p, p, dgammaReg, pMask);
                    MicroAPI::Abs(pMag, p, pMask);
                    MicroAPI::CompareScalar<float, CMPMODE::LE>(pFinite, pMag, FP32_MAX_FINITE, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(q, p, rlo, pFinite);
                    MicroAPI::MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(q, p, rhiReg, pMask);
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(t, u, q, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(t, t, gammaReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dxReg, t, rstdReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dxReg, pMask, offset);
                }
            }
        }
    }

private:
    TPipe* pipe_ = nullptr;
    int64_t rDim_;
    int64_t aDim_;
    int64_t numBlocks;
    int64_t mainBlockCount;
    int64_t mainBlockFactor;
    int64_t tailBlockFactor;

    int64_t aLoopFactor;
    int64_t aLoopFactorAligned;
    int64_t aLoopCount;
    int64_t aLoopTail;

    int64_t reduceLoopTimes;
    int64_t reduceRecursionLoop;
    int64_t foldLoopStep1;
    int64_t foldLoopStep2;
    int64_t foldLoopStep3;
    int64_t foldLoopOffset1;
    int64_t foldLoopOffset2;
    int64_t foldLoopOffset3;

    GlobalTensor<DY_TYPE> dyGm_, xGm_, dxGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, rstdGm_, workspaceGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dxOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;
    TBuf<TPosition::VECCALC> dbetaBinaryAddBuf_;
    TBuf<TPosition::VECCALC> dgammaBinaryAddBuf_;
};
} // namespace BatchNormGradExt2
#endif // __BATCH_NORM_GRAD_EXT2_RA_FULL_LOAD_REGBASE_H__
