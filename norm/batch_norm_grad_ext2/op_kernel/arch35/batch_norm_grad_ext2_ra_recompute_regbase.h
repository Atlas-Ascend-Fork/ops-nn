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
 * \file batch_norm_grad_ext2_ra_recompute_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_EXT2_RA_RECOMPUTE_REGBASE_H__
#define __BATCH_NORM_GRAD_EXT2_RA_RECOMPUTE_REGBASE_H__

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
class BatchNormGradExt2RARecompute {
public:
    __aicore__ inline BatchNormGradExt2RARecompute(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(__gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* mean, __gm__ uint8_t* rstd,
                                __gm__ uint8_t* gamma, __gm__ uint8_t* dx, __gm__ uint8_t* dgamma,
                                __gm__ uint8_t* dbeta, __gm__ uint8_t* workspace,
                                const BatchNormGradExt2RARecomputeTilingData* tilingData)
    {
        const BatchNormGradExt2BaseTilingData& baseTilingData = tilingData->baseTilingData;
        rDim_ = baseTilingData.r1Dim;
        aDim_ = baseTilingData.aDim;
        numBlocks = tilingData->numBlocks;
        mainBlockCount = tilingData->mainBlockCount;
        mainBlockFactor = tilingData->mainBlockFactor;
        tailBlockFactor = tilingData->tailBlockFactor;
        blockFactor = GetBlockIdx() >= mainBlockCount ? tailBlockFactor : mainBlockFactor;

        aLoopFactor = tilingData->aLoopFactor;
        aLoopFactorAligned = tilingData->aLoopFactorAligned;
        aLoopTimes = blockFactor / aLoopFactor;
        aLoopTail = blockFactor - aLoopTimes * aLoopFactor;
        rLoopFactor = tilingData->rLoopFactor;
        rLoopTimes = tilingData->rLoopTimes;
        rLoopTail = tilingData->rLoopTail;

        binaryFoldPoint = tilingData->binaryFoldPoint;
        binaryBlockCount = tilingData->binaryBlockCount;
        binaryTailBlock = tilingData->binaryTailBlock;
        cacheBufferCount = tilingData->cacheBufferCount;
        reciprocal = tilingData->reciprocal;
        // host 的 reciprocal 是 fl32(1/R)（double 除法后一次舍入）；这里补上精确整数 R 与
        // fl32(R) 之差，供 CompExactDivisorReciprocalLo 把 rhi + rlo 修正到 1/R。
        negDelta = ExactDivisorNegDelta(rDim_);
        resultCacheId = GetCacheID(binaryFoldPoint - 1);

        // init gm tensor
        int64_t offset = tilingData->mainBlockFactor * GetBlockIdx();
        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)dy + offset);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)x + offset);
        meanGm_.SetGlobalBuffer((__gm__ float*)mean + offset);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + offset);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)gamma + offset);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)dx + offset);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)dgamma + offset);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)dbeta + offset);

        // init queue
        int64_t feat_bytes = rLoopFactor * aLoopFactorAligned * sizeof(float);
        int64_t para_bytes = aLoopFactorAligned * sizeof(float);
        feat_bytes = RoundUpOneBlock(feat_bytes);
        para_bytes = RoundUpOneBlock(para_bytes);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, feat_bytes);
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, feat_bytes);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, para_bytes);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, para_bytes);
        pipe_->InitBuffer(gammaInQue_, BUFFER_NUM, para_bytes);
        pipe_->InitBuffer(dxOutQue_, BUFFER_NUM, feat_bytes);
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, para_bytes * cacheBufferCount);
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, para_bytes * cacheBufferCount);
        // 补偿累加器：dbeta/dgamma 各 hi/lo/lo2 三个字（见 comp_sum.h CompAddWide）
        accLaneGroups_ = CeilDiv(aLoopFactorAligned, static_cast<int64_t>(VL_FP32));
        pipe_->InitBuffer(compAccBuf_, 6 * accLaneGroups_ * static_cast<int64_t>(VL_FP32) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= numBlocks) {
            return;
        }

        for (int64_t i = 0; i < aLoopTimes; ++i) {
            ProcessOneRAFactor(i * aLoopFactor, aLoopFactor);
        }
        if (aLoopTail > 0) {
            ProcessOneRAFactor(aLoopTimes * aLoopFactor, aLoopTail);
        }
    }

public:
    // ---- 补偿（double-fp32 三字）累加：lane = A 轴通道，R 轴逐行累加到同一 lane ----
    __aicore__ inline __local_mem__ float* CompBase()
    {
        return (__local_mem__ float*)compAccBuf_.Get<float>().GetPhyAddr();
    }

    __aicore__ inline void InitCompAcc()
    {
        __local_mem__ float* base = CompBase();
        int64_t total = 6 * accLaneGroups_ * static_cast<int64_t>(VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> zero;
            MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(zero, 0.0f, all);
            for (uint16_t i = 0; i < static_cast<uint16_t>(total / VL_FP32); ++i) {
                MicroAPI::DataCopy(base + i * VL_FP32, zero, all);
            }
        }
    }

    __aicore__ inline void ProcessMainBlock(int64_t baseOffset, int64_t binaryLoopIdx, int64_t rLength, int64_t aLength)
    {
        int64_t offset = baseOffset + binaryLoopIdx * rLoopFactor * aDim_;
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(dyLocal, dyGm_[offset], rLength, aLength, aLoopFactorAligned, aDim_);
        dyInQue_.EnQue(dyLocal);
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();

        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(xLocal, xGm_[offset], rLength, aLength, aLoopFactorAligned, aDim_);
        xInQue_.EnQue(xLocal);
        xLocal = xInQue_.template DeQue<DY_TYPE>();

        AccumBlock(dyLocal, xLocal, rLength, aLength);

        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessFoldBlock(int64_t baseOffset, int64_t binaryLoopIdx, int64_t rLength, int64_t aLength)
    {
        int64_t offset = baseOffset + (binaryLoopIdx + binaryFoldPoint) * rLoopFactor * aDim_;
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(dyLocal, dyGm_[offset], rLength, aLength, aLoopFactorAligned, aDim_);
        dyInQue_.EnQue(dyLocal);
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();

        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(xLocal, xGm_[offset], rLength, aLength, aLoopFactorAligned, aDim_);
        xInQue_.EnQue(xLocal);
        xLocal = xInQue_.template DeQue<DY_TYPE>();

        AccumBlock(dyLocal, xLocal, rLength, aLength);

        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void AccumBlock(const LocalTensor<DY_TYPE>& dyLocal, const LocalTensor<DY_TYPE>& xLocal,
                                      int64_t rLength, int64_t aLength)
    {
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal.GetPhyAddr();
        __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xLocal.GetPhyAddr();
        __local_mem__ float* base = CompBase();
        int64_t groupStride = static_cast<int64_t>(VL_FP32);
        int64_t totalVec = accLaneGroups_ * groupStride;
        uint16_t laneGroups = static_cast<uint16_t>(CeilDiv(aLength, groupStride));
        __VEC_SCOPE__
        {
            MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            __local_mem__ float* dbHi = base;
            __local_mem__ float* dbLo = base + totalVec;
            __local_mem__ float* dbLo2 = base + DIGIT_TWO * totalVec;
            __local_mem__ float* dgHi = base + DIGIT_THREE * totalVec;
            __local_mem__ float* dgLo = base + 4 * totalVec;
            __local_mem__ float* dgLo2 = base + 5 * totalVec;
            RegTensor<float> s0Hi, s0Lo, s0Lo2, s1Hi, s1Lo, s1Lo2, dyReg, xReg, prodHi;
            uint32_t count = static_cast<uint32_t>(aLength);
            for (uint16_t i = 0; i < laneGroups; ++i) {
                MaskReg pMask = MicroAPI::UpdateMask<float>(count);
                uint32_t laneOffset = i * groupStride;
                MicroAPI::DataCopy(s0Hi, dbHi + laneOffset);
                MicroAPI::DataCopy(s0Lo, dbLo + laneOffset);
                MicroAPI::DataCopy(s0Lo2, dbLo2 + laneOffset);
                MicroAPI::DataCopy(s1Hi, dgHi + laneOffset);
                MicroAPI::DataCopy(s1Lo, dgLo + laneOffset);
                MicroAPI::DataCopy(s1Lo2, dgLo2 + laneOffset);
                for (uint16_t j = 0; j < static_cast<uint16_t>(rLength); ++j) {
                    uint32_t offset = laneOffset + j * static_cast<uint32_t>(aLoopFactorAligned);
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
                MicroAPI::DataCopy(dbHi + laneOffset, s0Hi, pMask);
                MicroAPI::DataCopy(dbLo + laneOffset, s0Lo, pMask);
                MicroAPI::DataCopy(dbLo2 + laneOffset, s0Lo2, pMask);
                MicroAPI::DataCopy(dgHi + laneOffset, s1Hi, pMask);
                MicroAPI::DataCopy(dgLo + laneOffset, s1Lo, pMask);
                MicroAPI::DataCopy(dgLo2 + laneOffset, s1Lo2, pMask);
            }
        }
    }

    __aicore__ inline void FinalizeDbetaDgamma(int64_t aLength, const LocalTensor<float>& dbetaLocal,
                                               const LocalTensor<float>& dgammaLocal)
    {
        __local_mem__ float* base = CompBase();
        __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor.GetPhyAddr();
        __local_mem__ float* dbOut = (__local_mem__ float*)dbetaLocal.GetPhyAddr();
        __local_mem__ float* dgOut = (__local_mem__ float*)dgammaLocal.GetPhyAddr();
        int64_t groupStride = static_cast<int64_t>(VL_FP32);
        int64_t totalVec = accLaneGroups_ * groupStride;
        int64_t resultOffset = resultCacheId * aLoopFactorAligned;
        uint16_t laneGroups = static_cast<uint16_t>(CeilDiv(aLength, groupStride));
        __VEC_SCOPE__
        {
            MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            __local_mem__ float* dbHi = base;
            __local_mem__ float* dbLo = base + totalVec;
            __local_mem__ float* dbLo2 = base + DIGIT_TWO * totalVec;
            __local_mem__ float* dgHi = base + DIGIT_THREE * totalVec;
            __local_mem__ float* dgLo = base + 4 * totalVec;
            __local_mem__ float* dgLo2 = base + 5 * totalVec;
            RegTensor<float> s0Hi, s0Lo, s0Lo2, s1Hi, s1Lo, s1Lo2, meanReg, rstdReg, dgFinal, dbFinal;
            uint32_t count = static_cast<uint32_t>(aLength);
            for (uint16_t i = 0; i < laneGroups; ++i) {
                MaskReg pMask = MicroAPI::UpdateMask<float>(count);
                uint32_t laneOffset = i * groupStride;
                MicroAPI::DataCopy(s0Hi, dbHi + laneOffset);
                MicroAPI::DataCopy(s0Lo, dbLo + laneOffset);
                MicroAPI::DataCopy(s0Lo2, dbLo2 + laneOffset);
                MicroAPI::DataCopy(s1Hi, dgHi + laneOffset);
                MicroAPI::DataCopy(s1Lo, dgLo + laneOffset);
                MicroAPI::DataCopy(s1Lo2, dgLo2 + laneOffset);
                CompFoldLo2(s0Lo, s0Lo2, pMask);
                CompFoldLo2(s1Lo, s1Lo2, pMask);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, laneOffset);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, laneOffset);
                CompFinalizeDgammaDbeta(s1Hi, s1Lo, s0Hi, s0Lo, meanReg, rstdReg, pMask, dgFinal, dbFinal);
                StoreOneTensor<float>(dbOut, dbFinal, pMask, static_cast<uint32_t>(resultOffset) + laneOffset);
                StoreOneTensor<float>(dgOut, dgFinal, pMask, static_cast<uint32_t>(resultOffset) + laneOffset);
            }
        }
    }

    __aicore__ inline void ProcessDx(int64_t baseOffset, int64_t blkIdx, int64_t rLength, int64_t aLength)
    {
        // ProcessDx
        int64_t offset = baseOffset + blkIdx * rLoopFactor * aDim_;
        dyTensor = dyInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(dyTensor, dyGm_[offset], rLength, aLength, aLoopFactorAligned, aDim_);
        dyInQue_.EnQue(dyTensor);
        dyTensor = dyInQue_.template DeQue<DY_TYPE>();
        xTensor = xInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(xTensor, xGm_[offset], rLength, aLength, aLoopFactorAligned, aDim_);
        xInQue_.EnQue(xTensor);
        xTensor = xInQue_.template DeQue<DY_TYPE>();

        dxTensor = dxOutQue_.template AllocTensor<DY_TYPE>();
        uint16_t outerLoopTimes = CeilDiv(aLength, VL_FP32);
        uint16_t innerLoopTimes = rLength;
        uint16_t outerLoopStride = VL_FP32;
        uint16_t innerLoopStride = aLoopFactorAligned;
        uint32_t factor = aLength;
        uint32_t gammaBetaOffset = resultCacheId * aLoopFactorAligned;
        __VEC_SCOPE__
        {
            __local_mem__ float* dyAddr = (__local_mem__ float*)dyTensor.GetPhyAddr();
            __local_mem__ float* xAddr = (__local_mem__ float*)xTensor.GetPhyAddr();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor.GetPhyAddr();
            __local_mem__ float* gammaAddr = (__local_mem__ float*)gammaTensor.GetPhyAddr();
            __local_mem__ float* dxAddr = (__local_mem__ float*)dxTensor.GetPhyAddr();
            __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbetaTensor.GetPhyAddr() + gammaBetaOffset;
            __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgammaTensor.GetPhyAddr() + gammaBetaOffset;
            AscendC::MicroAPI::MaskReg pMask;
            uint32_t count = factor;
            AscendC::MicroAPI::RegTensor<float> dyReg, xReg, meanReg, rstdReg, gammaReg, dxReg, dbetaReg, dgammaReg;
            AscendC::MicroAPI::RegTensor<float> u, p, q, t;
            AscendC::MicroAPI::RegTensor<float> rhiReg, rlo, dbetaOverR, pMag;
            AscendC::MicroAPI::MaskReg pFinite;
            AscendC::MicroAPI::MaskReg
                all = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            // 与 golden/torch 同构：dx = ((dy - dbeta/R) - (xhat*dgamma)/R) * gamma * rstd；
            // 除法用 (rhi + rlo) 的 FMA 形式（单次舍入，等价真除法）；rlo 按**精确整数 R** 补残差
            // （见 comp_sum.h：只修正到 fl32(R) 时整通道带 (R-fl32(R))/R 的同号偏差）。
            AscendC::MicroAPI::Duplicate(rhiReg, reciprocal, all);
            CompExactDivisorReciprocalLo(rlo, rhiReg, static_cast<float>(rDim_), negDelta, all);
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerLoopStride);
                LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(dbetaAddr, dbetaReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(dgammaAddr, dgammaReg, pMask, i * outerLoopStride);
                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dbetaOverR, dbetaReg, rlo,
                                                                                         pMask);
                AscendC::MicroAPI::MulAddDst<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dbetaOverR, dbetaReg,
                                                                                               rhiReg, pMask);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    uint32_t offset = i * outerLoopStride + j * innerLoopStride;
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, offset);
                    AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(u, dyReg, dbetaOverR,
                                                                                             pMask);
                    AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg,
                                                                                             pMask);
                    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(p, xReg, rstdReg, pMask);
                    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(p, p, dgammaReg, pMask);
                    AscendC::MicroAPI::Abs(pMag, p, pMask);
                    AscendC::MicroAPI::CompareScalar<float, AscendC::CMPMODE::LE>(pFinite, pMag, FP32_MAX_FINITE,
                                                                                  pMask);
                    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(q, p, rlo, pFinite);
                    AscendC::MicroAPI::MulAddDst<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(q, p, rhiReg, pMask);
                    AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(t, u, q, pMask);
                    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(t, t, gammaReg, pMask);
                    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dxReg, t, rstdReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dxReg, pMask, offset);
                }
            }
        }
        dyInQue_.FreeTensor(dyTensor);
        xInQue_.FreeTensor(xTensor);
        // Copy Out dxTensor
        dxOutQue_.EnQue(dxTensor);
        dxTensor = dxOutQue_.template DeQue<DY_TYPE>();
        CopyOut(dxGm_[offset], dxTensor, rLength, aLength, aDim_, aLoopFactorAligned);
        dxOutQue_.FreeTensor(dxTensor);
    }

    __aicore__ inline void ProcessOneRAFactor(int64_t gmOffset, int64_t aLength)
    {
        // ProcessOneRAFactor
        dbetaTensor = dbetaOutQue_.template AllocTensor<float>();
        dgammaTensor = dgammaOutQue_.template AllocTensor<float>();

        InitCompAcc();
        int64_t hasBinaryTailBlock = binaryTailBlock > 0 ? 1 : 0;
        int64_t loop1 = binaryBlockCount - binaryFoldPoint - hasBinaryTailBlock;
        for (int64_t binaryIdx = 0; binaryIdx < loop1; ++binaryIdx) {
            ProcessMainBlock(gmOffset, binaryIdx, rLoopFactor, aLength);
            ProcessFoldBlock(gmOffset, binaryIdx, rLoopFactor, aLength);
        }
        int64_t loop2 = hasBinaryTailBlock;
        for (int64_t binaryIdx = 0; binaryIdx < loop2; ++binaryIdx) {
            ProcessMainBlock(gmOffset, loop1, rLoopFactor, aLength);
            ProcessFoldBlock(gmOffset, loop1, binaryTailBlock, aLength);
        }
        for (int64_t binaryIdx = loop1 + loop2; binaryIdx < binaryFoldPoint; ++binaryIdx) {
            ProcessMainBlock(gmOffset, binaryIdx, rLoopFactor, aLength);
        }

        // Copy In gamma
        gammaTensor = gammaInQue_.template AllocTensor<WEIGHT_TYPE>();
        CopyIn(gammaTensor, gammaGm_[gmOffset], aLength);
        gammaInQue_.EnQue(gammaTensor);
        gammaTensor = gammaInQue_.template DeQue<WEIGHT_TYPE>();

        // Copy In mean
        meanTensor = meanInQue_.template AllocTensor<float>();
        CopyIn(meanTensor, meanGm_[gmOffset], aLength);
        meanInQue_.EnQue(meanTensor);
        meanTensor = meanInQue_.template DeQue<float>();

        // Copy In rstd
        rstdTensor = rstdInQue_.template AllocTensor<float>();
        CopyIn(rstdTensor, rstdGm_[gmOffset], aLength);
        rstdInQue_.EnQue(rstdTensor);
        rstdTensor = rstdInQue_.template DeQue<float>();

        FinalizeDbetaDgamma(aLength, dbetaTensor, dgammaTensor);

        for (int64_t i = 0; i < rLoopTimes; i++) {
            ProcessDx(gmOffset, i, rLoopFactor, aLength);
        }
        if (rLoopTail > 0) {
            ProcessDx(gmOffset, rLoopTimes, rLoopTail, aLength);
        }

        int64_t resultOffset = resultCacheId * aLoopFactorAligned;
        __VEC_SCOPE__
        {
            __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbetaTensor.GetPhyAddr() + resultOffset;
            __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgammaTensor.GetPhyAddr() + resultOffset;
            __local_mem__ WEIGHT_TYPE* dbetaTypeAddr = (__local_mem__ WEIGHT_TYPE*)dbetaTensor.GetPhyAddr();
            __local_mem__ WEIGHT_TYPE* dgammaTypeAddr = (__local_mem__ WEIGHT_TYPE*)dgammaTensor.GetPhyAddr();
            AscendC::MicroAPI::MaskReg
                pFull = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> dbetaReg, dgammaReg;
            LoadOneTensor<float>(dbetaAddr, dbetaReg, pFull, 0);
            LoadOneTensor<float>(dgammaAddr, dgammaReg, pFull, 0);
            StoreOneTensor<WEIGHT_TYPE>(dbetaTypeAddr, dbetaReg, pFull, 0);
            StoreOneTensor<WEIGHT_TYPE>(dgammaTypeAddr, dgammaReg, pFull, 0);
        }

        dbetaOutQue_.EnQue(dbetaTensor);
        dbetaTensor = dbetaOutQue_.template DeQue<float>();
        LocalTensor<WEIGHT_TYPE> dbetaType = dbetaTensor.template ReinterpretCast<WEIGHT_TYPE>();
        CopyOut(dbetaGm_[gmOffset], dbetaType, aLength);
        dgammaOutQue_.EnQue(dgammaTensor);
        dgammaTensor = dgammaOutQue_.template DeQue<float>();
        LocalTensor<WEIGHT_TYPE> dgammaType = dgammaTensor.template ReinterpretCast<WEIGHT_TYPE>();
        CopyOut(dgammaGm_[gmOffset], dgammaType, aLength);

        gammaInQue_.FreeTensor(gammaTensor);
        meanInQue_.FreeTensor(meanTensor);
        rstdInQue_.FreeTensor(rstdTensor);
        dbetaOutQue_.FreeTensor(dbetaTensor);
        dgammaOutQue_.FreeTensor(dgammaTensor);
    }

private:
    TPipe* pipe_ = nullptr;
    int64_t accLaneGroups_{0};
    TBuf<TPosition::VECCALC> compAccBuf_;
    int64_t rDim_;
    int64_t aDim_;
    int64_t numBlocks;
    int64_t mainBlockCount;
    int64_t mainBlockFactor;
    int64_t tailBlockFactor;
    int64_t blockFactor;

    int64_t aLoopFactor;
    int64_t aLoopFactorAligned;
    int64_t aLoopTimes;
    int64_t aLoopTail;
    int64_t rLoopFactor;
    int64_t rLoopTimes;
    int64_t rLoopTail;

    int64_t binaryFoldPoint;
    int64_t binaryBlockCount;
    int64_t binaryTailBlock;
    int64_t cacheBufferCount;
    int64_t resultCacheId;
    float reciprocal;
    float negDelta{0}; // -(R - fl32(R))：把 1/fl32(R) 两字倒数修正到 1/R

    // global tensor
    GlobalTensor<DY_TYPE> dyGm_, xGm_, dxGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, rstdGm_, workspaceGm_;

    // local tensor
    LocalTensor<float> dyMain, dyFold, xMain, xFold;
    LocalTensor<float> meanTensor, rstdTensor;
    LocalTensor<float> dbetaTensor, dgammaTensor;
    LocalTensor<WEIGHT_TYPE> gammaTensor;
    LocalTensor<DY_TYPE> dxTensor, dyTensor, xTensor;

    // queue
    TQue<QuePosition::VECIN, BUFFER_NUM> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dxOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;
};

} // namespace BatchNormGradExt2
#endif // __BATCH_NORM_GRAD_EXT2_RA_RECOMPUTE_REGBASE_H__
