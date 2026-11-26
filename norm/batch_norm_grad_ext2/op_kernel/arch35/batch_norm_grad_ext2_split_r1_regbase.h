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
 * \file batch_norm_grad_ext2_split_r1.h
 * \brief
 */

#ifndef BATCH_NORM_GRAD_EXT2_SPLIT_R1_REGBASE_H
#define BATCH_NORM_GRAD_EXT2_SPLIT_R1_REGBASE_H

#include "batch_norm_grad_ext2_common.h"
#include "batch_norm_grad_ext2_comp_sum.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace BatchNormGradExt2 {
using namespace AscendC;
using namespace BatchNormGradExt2::Comp;

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1>
class BatchNormGradExt2RARSplitR1 {
public:
    __aicore__ inline BatchNormGradExt2RARSplitR1(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline uint32_t CalcBinaryAddLastNum(uint32_t binaryAddQuotient)
    {
        if (binaryAddQuotient <= VL_FP32 * VL_FP32) {
            return binaryAddQuotient / VL_FP32;
        }
        return VL_FP32;
    }

    __aicore__ inline void Init(__gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* mean, __gm__ uint8_t* rstd,
                                __gm__ uint8_t* gamma, __gm__ uint8_t* dx, __gm__ uint8_t* dgamma,
                                __gm__ uint8_t* dbeta, __gm__ uint8_t* workspace,
                                const BatchNormGradExt2RARRecomputeTilingData* tilingData)
    {
        const BatchNormGradExt2BaseTilingData& baseTilingData = tilingData->baseTilingData;
        r1Dim_ = baseTilingData.r1Dim;
        aDim_ = baseTilingData.aDim;
        r0Dim_ = baseTilingData.r0Dim;
        blockNum_ = baseTilingData.blockNum;
        rAlign_ = tilingData->ubRDimFactor;

        binaryAddParam_.binaryAddQuotient = tilingData->generalBinAddTilingData.binaryAddQuotient;
        binaryAddParam_.binaryAddk = tilingData->generalBinAddTilingData.binaryAddk;
        binaryAddParam_.binaryAddLastNum = tilingData->generalBinAddTilingData.binaryAddLastNum;
        aTailCoreNum_ = baseTilingData.tailBlockNum;
        ubRDimLoopNum_ = tilingData->ubRDimLoopNum;

        nFactor_ = tilingData->ubRDimFactor / r0Dim_;
        tailNFactor_ = r1Dim_ % nFactor_ == 0 ? nFactor_ : r1Dim_ % nFactor_;
        ubRDimTailLoopNum_ = CeilDiv(r1Dim_, nFactor_) - ubRDimLoopNum_;

        gmOffset_ = GetBlockIdx() < baseTilingData.tailBlockNum ?
                        GetBlockIdx() * baseTilingData.tailBlockDim :
                        baseTilingData.tailBlockNum * baseTilingData.tailBlockDim +
                            (GetBlockIdx() - baseTilingData.tailBlockNum) * baseTilingData.formerBlockDim;

        aDimLoopNum_ = GetBlockIdx() < baseTilingData.tailBlockNum ? baseTilingData.tailBlockDim :
                                                                     baseTilingData.formerBlockDim;

        tailBinaryAddParam_.binaryAddQuotient = binaryAddParam_.binaryAddQuotient >> 1;
        tailBinaryAddParam_.binaryAddk = binaryAddParam_.binaryAddk > 1 ? binaryAddParam_.binaryAddk - 1 : 0;
        tailBinaryAddParam_.binaryAddLastNum = CalcBinaryAddLastNum(tailBinaryAddParam_.binaryAddQuotient);

        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dy) + gmOffset_ * r0Dim_);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(x) + gmOffset_ * r0Dim_);
        meanGm_.SetGlobalBuffer((__gm__ float*)(mean) + gmOffset_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)(rstd) + gmOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(gamma) + gmOffset_);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dx) + gmOffset_ * r0Dim_);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dgamma) + gmOffset_);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dbeta) + gmOffset_);

        dyBufSize_ = RoundUpTwoBlock(rAlign_ * sizeof(DY_TYPE));
        xBufSize_ = RoundUpTwoBlock(rAlign_ * sizeof(DY_TYPE));
        xBufElemNum_ = xBufSize_ / sizeof(DY_TYPE);
        halfXBufOffset_ = xBufElemNum_ / DIGIT_TWO;
        meanBufSize_ = ONE_BLK_SIZE;
        meanBufElemNum_ = meanBufSize_ / sizeof(float);
        gammaBufSize_ = ONE_BLK_SIZE;
        gammaBufElemNum_ = gammaBufSize_ / sizeof(WEIGHT_TYPE);
        binaryAddBufSize_ = RoundUpOneBlock(binaryAddParam_.binaryAddQuotient / VL_FP32 * sizeof(float));
        binaryAddBufElemNum_ = binaryAddBufSize_ / sizeof(float);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyBufSize_);
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, xBufSize_);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(gammaInQue_, BUFFER_NUM, gammaBufSize_);
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(binaryAddBuf_, binaryAddBufSize_ * BUFFER_NUM);
        pipe_->InitBuffer(cacheBuf_, CACHE_BUFF_SIZE);

        V_MTE3_EVENT = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        MTE3_MTE2_EVENT = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void PreProcess(uint64_t ni)
    {
        // PreProcess
        dBetaLocal_ = dbetaOutQue_.template AllocTensor<float>();
        dGammaLocal_ = dgammaOutQue_.template AllocTensor<float>();
        cacheLocal_ = cacheBuf_.Get<float>();
        InitCompAcc();

        meanLocal_ = meanInQue_.template AllocTensor<float>();
        PrepareMean(meanLocal_, ni, 1);
        meanInQue_.EnQue(meanLocal_);

        rstdLocal_ = rstdInQue_.template AllocTensor<float>();
        PrepareRstd(rstdLocal_, ni, 1);
        rstdInQue_.EnQue(rstdLocal_);

        gammaLocal_ = gammaInQue_.template AllocTensor<WEIGHT_TYPE>();
        PrepareInGamma(gammaLocal_, ni, 1);
        gammaInQue_.EnQue(gammaLocal_);

        meanLocal_ = meanInQue_.template DeQue<float>();
        rstdLocal_ = rstdInQue_.template DeQue<float>();
        gammaLocal_ = gammaInQue_.template DeQue<WEIGHT_TYPE>();
    }

    __aicore__ inline void PostProcess(uint64_t ni)
    {
        // PostProcess
        ReSaveDGammaDBeta(dGammaLocal_, dBetaLocal_);
        dbetaOutQue_.EnQue(dBetaLocal_);
        LocalTensor<WEIGHT_TYPE> dBetaLocal = dbetaOutQue_.template DeQue<WEIGHT_TYPE>();
        CopyOutDbeta(dBetaLocal, ni, 1);
        dbetaOutQue_.FreeTensor(dBetaLocal);

        dgammaOutQue_.EnQue(dGammaLocal_);
        LocalTensor<WEIGHT_TYPE> dGammaLocal = dgammaOutQue_.template DeQue<WEIGHT_TYPE>();
        CopyOutDgamma(dGammaLocal, ni, 1);
        dgammaOutQue_.FreeTensor(dGammaLocal);

        meanInQue_.FreeTensor(meanLocal_);
        rstdInQue_.FreeTensor(rstdLocal_);
        gammaInQue_.FreeTensor(gammaLocal_);
    }

    __aicore__ inline void ProcessMainBlock(uint64_t ni, uint64_t basicBlockIdx, uint64_t r1Factor)
    {
        uint64_t offset = ni * r0Dim_ + basicBlockIdx * nFactor_ * r0Dim_ * aDim_;
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        PrepareInDy(dyLocal, 0, offset, r1Factor);
        dyInQue_.EnQue(dyLocal);
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();

        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        PrepareInX(xLocal, 0, offset, r1Factor);
        xInQue_.EnQue(xLocal);
        xLocal = xInQue_.template DeQue<DY_TYPE>();

        AccumDbetaDgamma(dyLocal, xLocal, r1Factor * r0Dim_, 0);

        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessFoldBlock(uint64_t ni, uint64_t basicBlockIdx, uint64_t r1Factor,
                                            uint64_t r1TailFactor, uint64_t r1TailTailFactor)
    {
        uint64_t offset = ni * r0Dim_ + basicBlockIdx * nFactor_ * r0Dim_ * aDim_;
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        PrepareInDy(dyLocal, 0, offset, r1Factor);
        uint64_t tailOffset = offset + ubRDimLoopNum_ * nFactor_ * r0Dim_ * aDim_;
        PrepareInDy(dyLocal, halfXBufOffset_, tailOffset, r1TailFactor);
        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        PrepareInX(xLocal, 0, offset, r1Factor);
        PrepareInX(xLocal, halfXBufOffset_, tailOffset, r1TailFactor);
        dyInQue_.EnQue(dyLocal);
        xInQue_.EnQue(xLocal);
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        xLocal = xInQue_.template DeQue<DY_TYPE>();

        AccumDbetaDgamma(dyLocal, xLocal, r1Factor * r0Dim_, r1TailFactor * r0Dim_);

        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);

        uint64_t foldOffset = ni * r0Dim_ + basicBlockIdx * nFactor_ * r0Dim_ * aDim_ + r1Factor * r0Dim_ * aDim_;
        dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        PrepareInDy(dyLocal, 0, foldOffset, r1Factor);
        uint64_t foldTailOffset = foldOffset + ubRDimLoopNum_ * nFactor_ * r0Dim_ * aDim_;
        if (r1TailTailFactor > 0) {
            PrepareInDy(dyLocal, halfXBufOffset_, foldTailOffset, r1TailTailFactor);
        }
        xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        PrepareInX(xLocal, 0, foldOffset, r1Factor);
        if (r1TailTailFactor > 0) {
            PrepareInX(xLocal, halfXBufOffset_, foldTailOffset, r1TailTailFactor);
        }
        dyInQue_.EnQue(dyLocal);
        xInQue_.EnQue(xLocal);
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        xLocal = xInQue_.template DeQue<DY_TYPE>();

        AccumDbetaDgamma(dyLocal, xLocal, r1Factor * r0Dim_, r1TailTailFactor * r0Dim_);

        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessDx(uint64_t ni, uint64_t basicBlockIdx, uint64_t r1Factor)
    {
        // ProcessDx
        uint32_t bufferIdx = basicBlockIdx % BUFFER_NUM;
        uint64_t offset = ni * r0Dim_ + basicBlockIdx * nFactor_ * r0Dim_ * aDim_;
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        PrepareInDy(dyLocal, 0, offset, r1Factor);
        dyInQue_.EnQue(dyLocal);
        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        PrepareInX(xLocal, 0, offset, r1Factor);
        xInQue_.EnQue(xLocal);

        xLocal = xInQue_.template DeQue<DY_TYPE>();
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();

        CalcDx(xLocal, dyLocal, rstdLocal_, meanLocal_, gammaLocal_, dGammaLocal_, dBetaLocal_, r1Factor * r0Dim_,
               r1Dim_ * r0Dim_);

        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT + bufferIdx);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT + bufferIdx);
        CopyOutDx(xLocal, offset, r1Factor);
        SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
        WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= blockNum_) {
            return;
        }

        for (uint64_t ni = 0; ni < aDimLoopNum_; ni++) {
            PreProcess(ni);
            for (uint64_t basicBlockIdx = 0; basicBlockIdx < ubRDimTailLoopNum_; basicBlockIdx++) {
                if (basicBlockIdx < ubRDimTailLoopNum_ - 1) {
                    ProcessFoldBlock(ni, basicBlockIdx, nFactor_ / DIGIT_TWO, nFactor_ / DIGIT_TWO,
                                     nFactor_ / DIGIT_TWO);
                } else {
                    uint64_t tailRFactor = nFactor_ / DIGIT_TWO;
                    uint64_t tailTailRFactor = tailNFactor_ - tailRFactor;
                    if (tailNFactor_ < tailRFactor) {
                        tailRFactor = tailNFactor_;
                        tailTailRFactor = 0;
                    }
                    ProcessFoldBlock(ni, basicBlockIdx, nFactor_ / DIGIT_TWO, tailRFactor, tailTailRFactor);
                }
            }
            for (uint64_t basicBlockIdx = ubRDimTailLoopNum_; basicBlockIdx < ubRDimLoopNum_; basicBlockIdx++) {
                ProcessMainBlock(ni, basicBlockIdx, nFactor_);
            }
            FinalizeChannel();
            for (uint64_t basicBlockIdx = 0; basicBlockIdx < ubRDimLoopNum_ + ubRDimTailLoopNum_ - 1; basicBlockIdx++) {
                ProcessDx(ni, basicBlockIdx, nFactor_);
            }
            ProcessDx(ni, ubRDimLoopNum_ + ubRDimTailLoopNum_ - 1, tailNFactor_);
            PostProcess(ni);
        }
    }

    __aicore__ inline void PrepareInDy(LocalTensor<DY_TYPE>& dy, uint64_t dstOffset, uint64_t srcOffset,
                                       uint64_t r1Factor)
    {
        CopyInRAR(dy, dyGm_, dstOffset, srcOffset, r1Factor);
    }

    __aicore__ inline void PrepareInX(LocalTensor<DY_TYPE>& x, uint64_t dstOffset, uint64_t srcOffset,
                                      uint64_t r1Factor)
    {
        CopyInRAR(x, xGm_, dstOffset, srcOffset, r1Factor);
    }

    __aicore__ inline void PrepareMean(LocalTensor<float> mean, uint64_t offset, uint64_t a)
    {
        CopyInA(mean, meanGm_, offset, a);
    }

    __aicore__ inline void PrepareRstd(LocalTensor<float>& rstd, uint64_t offset, uint64_t a)
    {
        CopyInA(rstd, rstdGm_, offset, a);
    }

    __aicore__ inline void PrepareInGamma(LocalTensor<WEIGHT_TYPE>& gamma, uint64_t offset, uint64_t a)
    {
        CopyInA(gamma, gammaGm_, offset, a);
    }

    // ---- 核内补偿（double-fp32 三字）累加 ----
    // 一个通道一个通道地处理（ni），R 轴在 lane 内并行累加，因此不需要跨 lane 归约：
    // dbeta = sum(dy)、dgamma = rstd * (sum(x*dy) - mean * sum(dy))，收尾只舍入一次。
    // 累加器从 cacheBuf_（原多级缓存，已不再使用）中切出 6 个 VL。
    __aicore__ inline __local_mem__ float* CompBase() { return (__local_mem__ float*)cacheLocal_.GetPhyAddr(); }

    __aicore__ inline void InitCompAcc()
    {
        __local_mem__ float* base = CompBase();
        __VEC_SCOPE__
        {
            RegTensor<float> zero;
            MicroAPI::MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(zero, 0.0f, all);
            for (uint16_t i = 0; i < 6; ++i) {
                MicroAPI::DataCopy(base + i * VL_FP32, zero, all);
            }
        }
    }

    template <typename T>
    __aicore__ inline void AccumLanes(const __local_mem__ T* addr, int64_t baseOffset, int64_t len,
                                      RegTensor<float>& hi, RegTensor<float>& lo, RegTensor<float>& lo2)
    {
        uint32_t count = static_cast<uint32_t>(len);
        for (uint16_t chunk = 0; chunk < static_cast<uint16_t>(len); chunk += static_cast<uint16_t>(VL_FP32)) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(count);
            RegTensor<float> v;
            LoadOneTensor<T>(addr, v, mask, baseOffset + chunk);
            CompAddWide<DY_TYPE>(hi, lo, lo2, v, mask);
        }
    }

    template <typename T>
    __aicore__ inline void AccumProdBand(const __local_mem__ T* dyAddr, const __local_mem__ T* xAddr,
                                         int64_t baseOffset, int64_t len, RegTensor<float>& hi, RegTensor<float>& lo,
                                         RegTensor<float>& lo2)
    {
        uint32_t count = static_cast<uint32_t>(len);
        for (uint16_t chunk = 0; chunk < static_cast<uint16_t>(len); chunk += static_cast<uint16_t>(VL_FP32)) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(count);
            RegTensor<float> dy, x, prod;
            LoadOneTensor<T>(dyAddr, dy, mask, baseOffset + chunk);
            LoadOneTensor<T>(xAddr, x, mask, baseOffset + chunk);
            if constexpr (IsSameType<T, float>::value) {
                CompAddProductWide<DY_TYPE>(hi, lo, lo2, x, dy, mask);
            } else {
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(prod, x, dy, mask);
                CompAddWide<DY_TYPE>(hi, lo, lo2, prod, mask);
            }
        }
    }

    __aicore__ inline void AccumDbetaDgamma(const LocalTensor<DY_TYPE>& dyLocal, const LocalTensor<DY_TYPE>& xLocal,
                                            int64_t len0, int64_t len1)
    {
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal.GetPhyAddr();
        __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xLocal.GetPhyAddr();
        __local_mem__ float* base = CompBase();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            RegTensor<float> s0Hi, s0Lo, s0Lo2, s1Hi, s1Lo, s1Lo2;
            MicroAPI::DataCopy(s0Hi, base);
            MicroAPI::DataCopy(s0Lo, base + VL_FP32);
            MicroAPI::DataCopy(s0Lo2, base + DIGIT_TWO * VL_FP32);
            MicroAPI::DataCopy(s1Hi, base + DIGIT_THREE * VL_FP32);
            MicroAPI::DataCopy(s1Lo, base + 4 * VL_FP32);
            MicroAPI::DataCopy(s1Lo2, base + 5 * VL_FP32);
            AccumLanes<DY_TYPE>(dyAddr, 0, len0, s0Hi, s0Lo, s0Lo2);
            AccumProdBand<DY_TYPE>(dyAddr, xAddr, 0, len0, s1Hi, s1Lo, s1Lo2);
            if (len1 > 0) {
                AccumLanes<DY_TYPE>(dyAddr, halfXBufOffset_, len1, s0Hi, s0Lo, s0Lo2);
                AccumProdBand<DY_TYPE>(dyAddr, xAddr, halfXBufOffset_, len1, s1Hi, s1Lo, s1Lo2);
            }
            MicroAPI::DataCopy(base, s0Hi, all);
            MicroAPI::DataCopy(base + VL_FP32, s0Lo, all);
            MicroAPI::DataCopy(base + DIGIT_TWO * VL_FP32, s0Lo2, all);
            MicroAPI::DataCopy(base + DIGIT_THREE * VL_FP32, s1Hi, all);
            MicroAPI::DataCopy(base + 4 * VL_FP32, s1Lo, all);
            MicroAPI::DataCopy(base + 5 * VL_FP32, s1Lo2, all);
        }
    }

    __aicore__ inline void FinalizeChannel()
    {
        __local_mem__ float* base = CompBase();
        __local_mem__ float* outBeta = (__local_mem__ float*)dBetaLocal_.GetPhyAddr();
        __local_mem__ float* outGamma = (__local_mem__ float*)dGammaLocal_.GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal_.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdLocal_.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg one = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            RegTensor<float> s0Hi, s0Lo, s0Lo2, s1Hi, s1Lo, s1Lo2, meanReg, rstdReg, dgFinal, dbFinal;
            MicroAPI::DataCopy(s0Hi, base);
            MicroAPI::DataCopy(s0Lo, base + VL_FP32);
            MicroAPI::DataCopy(s0Lo2, base + DIGIT_TWO * VL_FP32);
            MicroAPI::DataCopy(s1Hi, base + DIGIT_THREE * VL_FP32);
            MicroAPI::DataCopy(s1Lo, base + 4 * VL_FP32);
            MicroAPI::DataCopy(s1Lo2, base + 5 * VL_FP32);
            CompFoldLo2(s0Lo, s0Lo2, all);
            CompFoldLo2(s1Lo, s1Lo2, all);
            // 通道内 R 方向共 r1Dim * r0Dim 个元素，只有前 min(R, VL) 条 lane 收过数据
            // （见 comp_sum.h CompReduceBeginWidth）。
            const uint32_t laneCount = CompLaneBound(static_cast<int64_t>(r1Dim_) * r0Dim_);
            CompReduceLanes(s0Hi, s0Lo, all, laneCount);
            CompReduceLanes(s1Hi, s1Lo, all, laneCount);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanAddr);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr);
            CompFinalizeDgammaDbeta(s1Hi, s1Lo, s0Hi, s0Lo, meanReg, rstdReg, one, dgFinal, dbFinal);
            StoreOneTensor<float>(outBeta, dbFinal, one, 0);
            StoreOneTensor<float>(outGamma, dgFinal, one, 0);
        }
    }

    __aicore__ inline void CalcDx(LocalTensor<DY_TYPE>& x, LocalTensor<DY_TYPE>& dy, LocalTensor<float>& rstd,
                                  LocalTensor<float>& mean, LocalTensor<WEIGHT_TYPE>& gamma, LocalTensor<float>& dgamma,
                                  LocalTensor<float>& dbeta, uint32_t xFactor, uint32_t r)
    {
        __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)x.GetPhyAddr();
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dy.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd.GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* gammaAddr = (__local_mem__ WEIGHT_TYPE*)gamma.GetPhyAddr();
        __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgamma.GetPhyAddr();
        __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbeta.GetPhyAddr();

        uint16_t VL = VL_FP32;
        uint16_t loopTimes = (xFactor + VL - 1) / VL;
        float hRecipValue = 1.0f / (float)r;
        float hNegDelta = ExactDivisorNegDelta(static_cast<int64_t>(r)); // -(R - fl32(R))
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::RegTensor<float> gammaValue;
            MicroAPI::RegTensor<float> dgammaValue;
            MicroAPI::RegTensor<float> dbetaValue;
            MicroAPI::MaskReg pregR;
            MicroAPI::RegTensor<float> dy, x, u, p, q, t, dx;
            MicroAPI::RegTensor<float> rhiReg, rlo, dbetaOverR, pMag;
            MicroAPI::MaskReg pFinite;
            MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            // 与 golden/torch 同构：dx = ((dy - dbeta/R) - (xhat*dgamma)/R) * gamma * rstd；
            // 除法用 (rhi + rlo) 的 FMA 形式（单次舍入，等价真除法），rlo 按**精确整数 R**
            // 补残差（见 comp_sum.h：只修正到 fl32(R) 时整通道带 (R-fl32(R))/R 的同号偏差）。
            MicroAPI::Duplicate(rhiReg, hRecipValue, preg);
            CompExactDivisorReciprocalLo(rlo, rhiReg, static_cast<float>(r), hNegDelta, preg);

            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float*)(rstdAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float*)(meanAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue,
                                                                        (__local_mem__ float*)(dgammaAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue, (__local_mem__ float*)(dbetaAddr));
            LoadOneElement<WEIGHT_TYPE>(gammaAddr, gammaValue, preg, 0);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dbetaOverR, dbetaValue, rlo, preg);
            MicroAPI::MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(dbetaOverR, dbetaValue, rhiReg, preg);

            uint32_t sregMask = (uint32_t)xFactor;
            for (uint16_t j = 0; j < loopTimes; j++) {
                pregR = MicroAPI::UpdateMask<float>(sregMask);

                LoadOneTensor<DY_TYPE>(xAddr, x, pregR, VL * j);
                LoadOneTensor<DY_TYPE>(dyAddr, dy, pregR, VL * j);
                MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(u, dy, dbetaOverR, pregR);
                MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(x, x, meanValue, pregR);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(p, x, rstdValue, pregR);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(p, p, dgammaValue, pregR);
                MicroAPI::Abs(pMag, p, pregR);
                MicroAPI::CompareScalar<float, CMPMODE::LE>(pFinite, pMag, FP32_MAX_FINITE, pregR);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(q, p, rlo, pFinite);
                MicroAPI::MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(q, p, rhiReg, pregR);
                MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(t, u, q, pregR);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(t, t, gammaValue, pregR);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dx, t, rstdValue, pregR);
                StoreOneTensor<DY_TYPE>(xAddr, dx, pregR, VL * j);
            }
        }
    }

    __aicore__ inline void ReSaveDGammaDBeta(LocalTensor<float>& dgamma, LocalTensor<float>& dbeta)
    {
        if constexpr (IsSameType<WEIGHT_TYPE, half>::value) {
            __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgamma.GetPhyAddr();
            __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbeta.GetPhyAddr();

            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<float> dgammaValue;
                MicroAPI::RegTensor<float> dbetaValue;
                MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue,
                                                                            (__local_mem__ float*)(dgammaAddr));
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue,
                                                                            (__local_mem__ float*)(dbetaAddr));
                StoreOneElement<WEIGHT_TYPE>(dbetaAddr, dbetaValue, pregMerge, 0);
                StoreOneElement<WEIGHT_TYPE>(dgammaAddr, dgammaValue, pregMerge, 0);
            }
        }
        return;
    }

    __aicore__ inline void CopyOutDx(LocalTensor<DY_TYPE>& dx, uint64_t offset, uint32_t r1Factor)
    {
        CopyOutRAR(dx, dxGm_, offset, r1Factor);
    }

    __aicore__ inline void CopyOutDgamma(LocalTensor<WEIGHT_TYPE>& dgamma, uint64_t meanOffset, uint32_t a)
    {
        CopyOutA(dgamma, dgammaGm_, meanOffset, a);
    }

    __aicore__ inline void CopyOutDbeta(LocalTensor<WEIGHT_TYPE>& dbeta, uint64_t meanOffset, uint32_t a)
    {
        CopyOutA(dbeta, dbetaGm_, meanOffset, a);
    }

    __aicore__ inline void CopyInRAR(LocalTensor<DY_TYPE>& localTensor, GlobalTensor<DY_TYPE>& globalTensor,
                                     uint64_t localOffset, uint64_t globalOffset, uint64_t r1Factor)
    {
        // RAR -> AR，R轴通过Compact Mode补pad到block对齐。
        DataCopyPadExtParams<DY_TYPE> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = r1Factor;
        copyInParams.blockLen = r0Dim_ * sizeof(DY_TYPE);
        copyInParams.srcStride = (aDim_ - 1) * r0Dim_ * sizeof(DY_TYPE);
        copyInParams.dstStride = 0;
        DataCopyPad<DY_TYPE, PaddingMode::Compact>(localTensor[localOffset], globalTensor[globalOffset], copyInParams,
                                                   dataCopyPadExtParams);
    }

    template <typename U>
    __aicore__ inline void CopyInA(LocalTensor<U>& localTensor, GlobalTensor<U>& globalTensor, uint64_t offset,
                                   uint64_t a)
    {
        DataCopyPadExtParams<U> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = a * sizeof(U);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad<U, PaddingMode::Compact>(localTensor, globalTensor[offset], copyInParams, dataCopyPadExtParams);
    }

    __aicore__ inline void CopyOutRAR(LocalTensor<DY_TYPE>& localTensor, GlobalTensor<DY_TYPE>& globalTensor,
                                      uint64_t offset, uint64_t r1Factor)
    {
        // RR -> RAR
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = r1Factor;
        copyOutParams.blockLen = r0Dim_ * sizeof(DY_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = (aDim_ - 1) * r0Dim_ * sizeof(DY_TYPE);
        DataCopyPad<DY_TYPE, PaddingMode::Compact>(globalTensor[offset], localTensor, copyOutParams);
    }

    __aicore__ inline void CopyOutA(LocalTensor<WEIGHT_TYPE>& localTensor, GlobalTensor<WEIGHT_TYPE>& globalTensor,
                                    uint64_t offset, uint64_t a)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = a * sizeof(WEIGHT_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad<WEIGHT_TYPE, PaddingMode::Compact>(globalTensor[offset], localTensor, copyOutParams);
    }

private:
    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECCALC> binaryAddBuf_;
    TBuf<TPosition::VECCALC> cacheBuf_;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;

    GlobalTensor<DY_TYPE> dyGm_, xGm_, dxGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, rstdGm_;

    event_t V_MTE3_EVENT;
    event_t MTE3_MTE2_EVENT;

    uint32_t r1Dim_{0};
    uint32_t aDim_{0};
    uint32_t r0Dim_{0};
    uint32_t rAlign_{0};
    uint32_t binaryAddQuotient_{0};
    uint32_t binaryAddK_{0};
    uint32_t binaryAddLastNum_{0};
    uint32_t aTailCoreNum_{0};
    uint32_t aDimTail_{0};
    uint32_t gmOffset_{0};
    uint32_t aDimLoopNum_{0};
    uint32_t dyBufSize_{0};
    uint32_t xBufSize_{0};
    uint32_t halfXBufOffset_{0};
    uint32_t xBufElemNum_{0};
    uint32_t meanBufSize_{0};
    uint32_t meanBufElemNum_{0};
    uint32_t gammaBufSize_{0};
    uint32_t gammaBufElemNum_{0};
    uint32_t binaryAddBufSize_{0};
    uint32_t binaryAddBufElemNum_{0};
    uint32_t blockNum_{0};
    uint64_t ubRDimLoopNum_{0};
    uint64_t nFactor_{0};
    uint64_t tailNFactor_{0};
    uint64_t ubRDimTailFactor_{0};
    uint64_t ubRDimTailTailFactor_{0};

    uint32_t tailBinaryAddQuotient_{0};
    uint32_t tailBinaryAddK_{0};
    uint32_t tailBinaryAddLastNum_{0};
    uint64_t ubRDimTailLoopNum_{0};
    BinaryAddParam binaryAddParam_;
    BinaryAddParam tailBinaryAddParam_;
    LocalTensor<float> dBetaLocal_;
    LocalTensor<float> dGammaLocal_;
    LocalTensor<float> cacheLocal_;
    LocalTensor<float> dBetaCacheLocal_;
    LocalTensor<float> dGammaCacheLocal_;
    LocalTensor<float> dBetaFoldCacheLocal_;
    LocalTensor<float> dGammaFoldCacheLocal_;
    LocalTensor<float> binaryAddTensor_;
    LocalTensor<float> rstdLocal_;
    LocalTensor<float> meanLocal_;
    LocalTensor<WEIGHT_TYPE> gammaLocal_;
};
} // namespace BatchNormGradExt2
#endif // BATCH_NORM_GRAD_EXT2_SPLIT_R1_REGBASE_H
