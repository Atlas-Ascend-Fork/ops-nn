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
 * \file batch_norm_grad_ext2_recompute_split_r0_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_EXT2_RECOMPUTE_SPLIT_R0_REGBASE_H_
#define __BATCH_NORM_GRAD_EXT2_RECOMPUTE_SPLIT_R0_REGBASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm_grad_ext2_common.h"
#include "batch_norm_grad_ext2_comp_sum.h"

namespace BNGRARRecomputeSplitR0 {
using namespace AscendC;
using namespace BatchNormGradExt2;
using namespace BatchNormGradExt2::Comp;
static constexpr int64_t UB_ADD_BUF = 3 * 1024;              // ub间二分累加固定buf
static constexpr uint32_t UB_ADD_LEN = 1024 / sizeof(float); // ub内二分累加每级存放个数

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1>
class BatchNormGradExt2RARRecomputeSplitR0 {
public:
    __aicore__ inline BatchNormGradExt2RARRecomputeSplitR0(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR mean, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx,
                                GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace,
                                const BatchNormGradExt2RARRecomputeTilingData* tilingData)
    {
        const BatchNormGradExt2BaseTilingData baseTilingData = tilingData->baseTilingData;
        r1Dim_ = baseTilingData.r1Dim;
        aDim_ = baseTilingData.aDim;
        r0Dim_ = baseTilingData.r0Dim;
        blockNum_ = baseTilingData.blockNum;
        /* ub内二分累加相关参数 */
        halfBinaryAddQuotient_ = tilingData->tailBinAddTilingData.binaryAddQuotient;
        halfBinaryAddK_ = tilingData->tailBinAddTilingData.binaryAddk;
        halfBinaryAddLastNum_ = tilingData->tailBinAddTilingData.binaryAddLastNum;
        binaryAddQuotient_ = tilingData->generalBinAddTilingData.binaryAddQuotient;
        binaryAddK_ = tilingData->generalBinAddTilingData.binaryAddk;
        binaryAddLastNum_ = tilingData->generalBinAddTilingData.binaryAddLastNum;
        /* A轴开多核的相关参数 */
        aDimPerCore_ = GetBlockIdx() < baseTilingData.tailBlockNum ? baseTilingData.tailBlockDim :
                                                                     baseTilingData.formerBlockDim;
        gmOffset_ = GetBlockIdx() < baseTilingData.tailBlockNum ?
                        (GetBlockIdx() * baseTilingData.tailBlockDim) :
                        (baseTilingData.tailBlockNum * baseTilingData.tailBlockDim +
                         (GetBlockIdx() - baseTilingData.tailBlockNum) * baseTilingData.formerBlockDim);
        /* ub切r0相关参数，涉及ub间二分累加 */
        ubRDimFactor_ = tilingData->ubRDimFactor;
        ubRDimLoopNum_ = tilingData->ubRDimLoopNum;
        ubRDimTailFactor_ = tilingData->ubRDimTailFactor;
        ubRDimTailFactorAlign_ = tilingData->ubRDimTailFactorAlign;
        ubRDimTailLoopNum_ = tilingData->ubRDimTailLoopNum;
        ubRDimTailTail_ = tilingData->ubRDimTailTail;
        ubRDimTailTailLoopNum_ = tilingData->ubRDimTailTailLoopNum;
        isTailLoop2_ = ubRDimTailTail_ > 0 && ubRDimTailTailLoopNum_ == BatchNormGradExt2::TWO;
        tailLoop_ = ubRDimTailTailLoopNum_ == BatchNormGradExt2::TWO ? ubRDimTailLoopNum_ / BatchNormGradExt2::TWO :
                                                                       ubRDimTailLoopNum_;
        tailLoop_ = ubRDimTailTail_ > 0 ? (tailLoop_ + 1) : tailLoop_;
        sumLoop_ = ubRDimLoopNum_ + tailLoop_;
        tailFactorAlign_ = RoundUpOneBlock(ubRDimTailFactor_ * sizeof(DY_TYPE)) / sizeof(DY_TYPE);

        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dy) + gmOffset_ * r0Dim_);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(x) + gmOffset_ * r0Dim_);
        meanGm_.SetGlobalBuffer((__gm__ float*)(mean) + gmOffset_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)(rstd) + gmOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(gamma) + gmOffset_);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dx) + gmOffset_ * r0Dim_);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dgamma) + gmOffset_);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dbeta) + gmOffset_);

        int64_t dyBufSize = tailFactorAlign_ * BatchNormGradExt2::TWO * sizeof(DY_TYPE);
        int64_t xBufSize = ubRDimTailFactorAlign_ * BatchNormGradExt2::TWO * sizeof(float);
        xBufElemNum_ = xBufSize / sizeof(DY_TYPE);
        int64_t binaryAddBufSize = RoundUpOneBlock(binaryAddQuotient_ / VL_FP32 * sizeof(float));

        pipe_.InitBuffer(dyInQue_, BUFFER_NUM, dyBufSize);
        pipe_.InitBuffer(xInQue_, BUFFER_NUM, xBufSize);
        pipe_.InitBuffer(meanInQue_, BUFFER_NUM, BatchNormGradExt2::ONE_BLK_SIZE);
        pipe_.InitBuffer(rstdInQue_, BUFFER_NUM, BatchNormGradExt2::ONE_BLK_SIZE);
        pipe_.InitBuffer(gammaInQue_, BUFFER_NUM, BatchNormGradExt2::ONE_BLK_SIZE);
        pipe_.InitBuffer(dbetaOutQue_, 1, BatchNormGradExt2::ONE_BLK_SIZE);
        pipe_.InitBuffer(dgammaOutQue_, 1, BatchNormGradExt2::ONE_BLK_SIZE);
        pipe_.InitBuffer(binaryAddBuf_, binaryAddBufSize);
        // 补偿累加器：dbeta/dgamma 各 hi/lo/lo2 三个字（见 comp_sum.h CompAddWide）
        pipe_.InitBuffer(compAccBuf_, 6 * BatchNormGradExt2::VL_FP32 * sizeof(float));

        V_MTE3_EVENT = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        MTE3_MTE2_EVENT = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= blockNum_) {
            return;
        }
        for (int64_t i = 0; i < aDimPerCore_; i++) {
            int64_t meanOffset = i;
            ProcessPre(meanOffset);
            CalcDbetaAndDgammaPerA(meanOffset);
            CalDxPerA(meanOffset);
            ProcessPost(meanOffset);
        }
    }

    __aicore__ inline void ProcessPre(int64_t meanOffset)
    {
        CopyInMeanAndRstd(meanOffset);
        CopyInGamma(meanOffset);
        mean_ = meanInQue_.template DeQue<float>();
        rstd_ = rstdInQue_.template DeQue<float>();
        gamma_ = gammaInQue_.template DeQue<WEIGHT_TYPE>();
        dgamma_ = dgammaOutQue_.template AllocTensor<float>();
        dbeta_ = dbetaOutQue_.template AllocTensor<float>();
        InitCompAcc();
    }

    __aicore__ inline void ProcessPost(int64_t meanOffset)
    {
        meanInQue_.FreeTensor(mean_);
        rstdInQue_.FreeTensor(rstd_);
        gammaInQue_.FreeTensor(gamma_);
        CopyOutDbetaAndDgamma(meanOffset);
    }

    __aicore__ inline void ProcessTail(int64_t offset1, int64_t offset2, uint32_t tailLen)
    {
        CopyInDyTwo(offset1, offset2, tailLen);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        CopyInXTwo(offset1, offset2, tailLen);
        LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
        // band0 = [0, ubRDimTailFactor_)，band1（尾）= [tailFactorAlign_, tailFactorAlign_ + tailLen)
        AccumBand(dyLocal, xLocal, 0, ubRDimTailFactor_, tailFactorAlign_, tailLen);
        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessHalfMain(int64_t offset1)
    {
        CopyInDyOne(offset1, ubRDimTailFactor_);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        CopyInXOne(offset1, ubRDimTailFactor_);
        LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
        AccumBand(dyLocal, xLocal, 0, ubRDimTailFactor_, 0, 0);
        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessMain(int64_t offset1)
    {
        CopyInDyOne(offset1, ubRDimFactor_);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        CopyInXOne(offset1, ubRDimFactor_);
        LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
        AccumBand(dyLocal, xLocal, 0, ubRDimFactor_, 0, 0);
        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessTailTail(int64_t offset1, int64_t offset2)
    {
        if (ubRDimTailTail_ > 0 && ubRDimTailTailLoopNum_ == 1) {
            ProcessTail(offset1, offset2, ubRDimTailTail_);
        }
        if (isTailLoop2_ && (ubRDimTailTail_ <= ubRDimTailFactor_)) {
            ProcessTail(offset1, offset2, ubRDimTailTail_);
            offset1 += ubRDimTailFactor_;
            ProcessHalfMain(offset1);
        }
        if (isTailLoop2_ && (ubRDimTailTail_ > ubRDimTailFactor_)) {
            ProcessTail(offset1, offset2, ubRDimTailFactor_);
            offset1 += ubRDimTailFactor_;
            offset2 += ubRDimTailFactor_;
            uint32_t tailLen = ubRDimTailTail_ - ubRDimTailFactor_;
            ProcessTail(offset1, offset2, tailLen);
        }
    }

    __aicore__ inline void CalcDbetaAndDgammaPerA(int64_t meanOffset)
    {
        int64_t offset = meanOffset * r0Dim_;
        int64_t tailOffset = ubRDimLoopNum_ * ubRDimFactor_;
        int64_t r1Offset = 0;
        int64_t offset1 = 0;
        int64_t offset2 = 0;
        for (int64_t k = 0; k < r1Dim_; k++) {
            r1Offset = k * r0Dim_ * aDim_;
            for (int64_t j = 0; j < ubRDimTailLoopNum_; j++) {
                offset1 = offset + r1Offset + ubRDimTailFactor_ * j;
                offset2 = offset1 + tailOffset;
                ProcessTail(offset1, offset2, ubRDimTailFactor_);
            }

            int64_t tailtailOffset = tailOffset + ubRDimTailLoopNum_ * ubRDimTailFactor_;
            offset1 = offset + r1Offset + ubRDimTailFactor_ * ubRDimTailLoopNum_;
            offset2 = offset + r1Offset + tailtailOffset;
            ProcessTailTail(offset1, offset2);

            for (int64_t j = tailLoop_; j < ubRDimLoopNum_; j++) {
                offset1 = offset + r1Offset + ubRDimFactor_ * j;
                ProcessMain(offset1);
            }
        }
        FinalizeChannel();
    }

    __aicore__ inline void CalDxPerA(int64_t meanOffset)
    {
        for (int64_t k = 0; k < r1Dim_; k++) {
            for (int64_t j = 0; j < sumLoop_; j++) {
                int64_t bufferIdx = j % BUFFER_NUM;
                int64_t offset1 = meanOffset * r0Dim_ + k * r0Dim_ * aDim_ + ubRDimFactor_ * j;
                uint32_t processR = ubRDimFactor_;
                bool isLast = j == (sumLoop_ - 1);
                if (unlikely(isLast)) {
                    processR = ubRDimTailTail_ == 0 ? ubRDimFactor_ : ubRDimTailTail_;
                }
                CopyInDyOne(offset1, processR);
                if (k > 0 || j >= BUFFER_NUM) {
                    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
                }
                CopyInXOne(offset1, processR);
                LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
                LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
                CalcDx(dyLocal, xLocal, processR);
                SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT + bufferIdx);
                WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT + bufferIdx);
                CopyOutDx(xLocal, processR, offset1);
                if (k < r1Dim_ - 1 || j < sumLoop_ - BUFFER_NUM) {
                    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
                }
                dyInQue_.FreeTensor(dyLocal);
                xInQue_.FreeTensor(xLocal);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInMeanAndRstd(int64_t offset)
    {
        mean_ = meanInQue_.template AllocTensor<float>();
        rstd_ = rstdInQue_.template AllocTensor<float>();
        DataCopyPadExtParams<float> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(mean_, meanGm_[offset], copyInParams, dataCopyPadExtParams);
        DataCopyPad(rstd_, rstdGm_[offset], copyInParams, dataCopyPadExtParams);
        meanInQue_.EnQue(mean_);
        rstdInQue_.EnQue(rstd_);
    }

    __aicore__ inline void CopyInGamma(int64_t offset)
    {
        LocalTensor<WEIGHT_TYPE> gammaLocal = gammaInQue_.template AllocTensor<WEIGHT_TYPE>();
        CopyIn(gammaLocal, gammaGm_, 0, offset, 1);
        gammaInQue_.EnQue(gammaLocal);
    }

    template <typename U>
    __aicore__ inline void CopyIn(LocalTensor<U>& localTensor, GlobalTensor<U>& globalTensor, uint32_t ubOffset,
                                  int64_t gmOffset, uint32_t copyLen)
    {
        DataCopyPadExtParams<U> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = copyLen * sizeof(U);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad<U, PaddingMode::Compact>(localTensor[ubOffset], globalTensor[gmOffset], copyInParams,
                                             dataCopyPadExtParams);
    }

    __aicore__ inline void CopyInDyTwo(int64_t offset1, int64_t offset2, uint32_t tailLen)
    {
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(dyLocal, dyGm_, 0, offset1, ubRDimTailFactor_);
        CopyIn(dyLocal, dyGm_, tailFactorAlign_, offset2, tailLen);
        dyInQue_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyInDyOne(int64_t offset1, uint32_t copyLen)
    {
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(dyLocal, dyGm_, 0, offset1, copyLen);
        dyInQue_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyInXTwo(int64_t offset1, int64_t offset2, uint32_t tailLen)
    {
        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        if constexpr (IsSameType<DY_TYPE, float>::value) {
            CopyIn(xLocal, xGm_, 0, offset1, ubRDimTailFactor_);
            CopyIn(xLocal, xGm_, tailFactorAlign_, offset2, tailLen);
        } else {
            CopyIn(xLocal, xGm_, xBufElemNum_ / BatchNormGradExt2::TWO, offset1, ubRDimTailFactor_);
            CopyIn(xLocal, xGm_, xBufElemNum_ / BatchNormGradExt2::TWO + tailFactorAlign_, offset2, tailLen);
        }
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInXOne(int64_t offset1, uint32_t copyLen)
    {
        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        if constexpr (IsSameType<DY_TYPE, float>::value) {
            CopyIn(xLocal, xGm_, 0, offset1, copyLen);
        } else {
            CopyIn(xLocal, xGm_, xBufElemNum_ / BatchNormGradExt2::TWO, offset1, copyLen);
        }
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void CastDbetaAndDgamma()
    {
        __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgamma_.GetPhyAddr();
        __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbeta_.GetPhyAddr();

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> dgammaValue;
            MicroAPI::RegTensor<float> dbetaValue;
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue,
                                                                        (__local_mem__ float*)(dgammaAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue, (__local_mem__ float*)(dbetaAddr));
            StoreOneElement<WEIGHT_TYPE>(dbetaAddr, dbetaValue, pregMerge, 0);
            StoreOneElement<WEIGHT_TYPE>(dgammaAddr, dgammaValue, pregMerge, 0);
        }
    }

    __aicore__ inline void CopyOutDbetaAndDgamma(int64_t offset)
    {
        if constexpr (IsSameType<WEIGHT_TYPE, half>::value) {
            CastDbetaAndDgamma();
        }
        dbetaOutQue_.EnQue(dbeta_);
        dgammaOutQue_.EnQue(dgamma_);
        LocalTensor<WEIGHT_TYPE> dbetaOut = dbetaOutQue_.template DeQue<WEIGHT_TYPE>();
        LocalTensor<WEIGHT_TYPE> dgammaOut = dgammaOutQue_.template DeQue<WEIGHT_TYPE>();
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = sizeof(WEIGHT_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad<WEIGHT_TYPE, PaddingMode::Compact>(dbetaGm_[offset], dbetaOut, copyOutParams);
        DataCopyPad<WEIGHT_TYPE, PaddingMode::Compact>(dgammaGm_[offset], dgammaOut, copyOutParams);
        dbetaOutQue_.FreeTensor(dbetaOut);
        dgammaOutQue_.FreeTensor(dgammaOut);
    }

    __aicore__ inline void CopyOutDx(LocalTensor<DY_TYPE>& dx, uint32_t copyLen, int64_t offset)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = copyLen * sizeof(DY_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad<DY_TYPE, PaddingMode::Compact>(dxGm_[offset], dx, copyOutParams);
    }

    // ---- 补偿（double-fp32 三字）累加 ----
    // 一个通道一个通道地处理（Process 的 A 循环），R 轴在 lane 内并行累加，不跨 lane 归约：
    // dbeta = sum(dy)、dgamma = rstd * (sum(x*dy) - mean * sum(dy))，收尾只舍入一次。
    __aicore__ inline __local_mem__ float* CompBase()
    {
        return (__local_mem__ float*)compAccBuf_.Get<float>().GetPhyAddr();
    }

    __aicore__ inline void InitCompAcc()
    {
        __local_mem__ float* base = CompBase();
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> zero;
            MicroAPI::MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(zero, 0.0f, all);
            for (uint16_t i = 0; i < 6; ++i) {
                MicroAPI::DataCopy(base + i * BatchNormGradExt2::VL_FP32, zero, all);
            }
        }
    }

    template <typename T>
    __aicore__ inline void AccumLanes(const __local_mem__ T* addr, int64_t baseOffset, int64_t len,
                                      MicroAPI::RegTensor<float>& hi, MicroAPI::RegTensor<float>& lo,
                                      MicroAPI::RegTensor<float>& lo2)
    {
        uint32_t count = static_cast<uint32_t>(len);
        for (uint16_t chunk = 0; chunk < static_cast<uint16_t>(len); chunk += static_cast<uint16_t>(VL_FP32)) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(count);
            MicroAPI::RegTensor<float> v;
            LoadOneTensor<T>(addr, v, mask, baseOffset + chunk);
            CompAddWide<DY_TYPE>(hi, lo, lo2, v, mask);
        }
    }

    template <typename T>
    __aicore__ inline void AccumProdBand(const __local_mem__ T* dyAddr, const __local_mem__ T* xAddr,
                                         int64_t baseOffset, int64_t len, MicroAPI::RegTensor<float>& hi,
                                         MicroAPI::RegTensor<float>& lo, MicroAPI::RegTensor<float>& lo2)
    {
        uint32_t count = static_cast<uint32_t>(len);
        for (uint16_t chunk = 0; chunk < static_cast<uint16_t>(len); chunk += static_cast<uint16_t>(VL_FP32)) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(count);
            MicroAPI::RegTensor<float> dy, x, prod;
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

    __aicore__ inline void AccumBand(const LocalTensor<DY_TYPE>& dyLocal, const LocalTensor<DY_TYPE>& xLocal,
                                     int64_t off0, int64_t len0, int64_t off1, int64_t len1)
    {
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal.GetPhyAddr();
        __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xLocal.GetPhyAddr();
        if constexpr (IsSameType<DY_TYPE, half>::value) {
            // fp16：输入 x 在 buffer 后半段（前半段留给 dx 的原地输出），偏移与 CalcDx 的读一致
            xAddr += xBufElemNum_ / BatchNormGradExt2::TWO;
        }
        __local_mem__ float* base = CompBase();
        constexpr int64_t VL = static_cast<int64_t>(BatchNormGradExt2::VL_FP32);
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::RegTensor<float> s0Hi, s0Lo, s0Lo2, s1Hi, s1Lo, s1Lo2;
            MicroAPI::DataCopy(s0Hi, base);
            MicroAPI::DataCopy(s0Lo, base + VL);
            MicroAPI::DataCopy(s0Lo2, base + DIGIT_TWO * VL);
            MicroAPI::DataCopy(s1Hi, base + DIGIT_THREE * VL);
            MicroAPI::DataCopy(s1Lo, base + 4 * VL);
            MicroAPI::DataCopy(s1Lo2, base + 5 * VL);
            AccumLanes<DY_TYPE>(dyAddr, off0, len0, s0Hi, s0Lo, s0Lo2);
            AccumProdBand<DY_TYPE>(dyAddr, xAddr, off0, len0, s1Hi, s1Lo, s1Lo2);
            if (len1 > 0) {
                AccumLanes<DY_TYPE>(dyAddr, off1, len1, s0Hi, s0Lo, s0Lo2);
                AccumProdBand<DY_TYPE>(dyAddr, xAddr, off1, len1, s1Hi, s1Lo, s1Lo2);
            }
            MicroAPI::DataCopy(base, s0Hi, all);
            MicroAPI::DataCopy(base + VL, s0Lo, all);
            MicroAPI::DataCopy(base + DIGIT_TWO * VL, s0Lo2, all);
            MicroAPI::DataCopy(base + DIGIT_THREE * VL, s1Hi, all);
            MicroAPI::DataCopy(base + 4 * VL, s1Lo, all);
            MicroAPI::DataCopy(base + 5 * VL, s1Lo2, all);
        }
    }

    __aicore__ inline void FinalizeChannel()
    {
        __local_mem__ float* base = CompBase();
        __local_mem__ float* outBeta = (__local_mem__ float*)dbeta_.GetPhyAddr();
        __local_mem__ float* outGamma = (__local_mem__ float*)dgamma_.GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)mean_.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd_.GetPhyAddr();
        constexpr int64_t VL = static_cast<int64_t>(BatchNormGradExt2::VL_FP32);
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg one = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::RegTensor<float> s0Hi, s0Lo, s0Lo2, s1Hi, s1Lo, s1Lo2, meanReg, rstdReg, dgFinal, dbFinal;
            MicroAPI::DataCopy(s0Hi, base);
            MicroAPI::DataCopy(s0Lo, base + VL);
            MicroAPI::DataCopy(s0Lo2, base + DIGIT_TWO * VL);
            MicroAPI::DataCopy(s1Hi, base + DIGIT_THREE * VL);
            MicroAPI::DataCopy(s1Lo, base + 4 * VL);
            MicroAPI::DataCopy(s1Lo2, base + 5 * VL);
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

    __aicore__ inline void CalcDx(LocalTensor<DY_TYPE>& dyLocal, LocalTensor<DY_TYPE>& xLocal, uint32_t r)
    {
        __local_mem__ DY_TYPE* xSrcAddr = (__local_mem__ DY_TYPE*)xLocal.GetPhyAddr();
        __local_mem__ DY_TYPE* xDstAddr = (__local_mem__ DY_TYPE*)xLocal.GetPhyAddr();
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal.GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)mean_.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd_.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* gammaAddr = (__local_mem__ WEIGHT_TYPE*)gamma_.GetPhyAddr();
        __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgamma_.GetPhyAddr();
        __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbeta_.GetPhyAddr();

        if constexpr (IsSameType<DY_TYPE, half>::value) {
            xSrcAddr += xBufElemNum_ / BatchNormGradExt2::TWO;
        }

        uint16_t loopTimes = CeilDiv(r, VL_FP32);
        float hRecipValue = 1.0f / (float)(r1Dim_ * r0Dim_);
        float hNegDelta = ExactDivisorNegDelta(r1Dim_ * r0Dim_); // -(R - fl32(R))
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> xValue;
            MicroAPI::RegTensor<float> xSubMean;
            MicroAPI::RegTensor<float> xHat;
            MicroAPI::RegTensor<float> gammaValue;
            MicroAPI::RegTensor<float> dgammaValue;
            MicroAPI::RegTensor<float> dbetaValue;
            MicroAPI::MaskReg pregR;
            MicroAPI::RegTensor<float> dy, u, p, q, t, dx;
            MicroAPI::RegTensor<float> rhiReg, rlo, dbetaOverR, pMag;
            MicroAPI::MaskReg pFinite;
            MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float*)(meanAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float*)(rstdAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue,
                                                                        (__local_mem__ float*)(dgammaAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue, (__local_mem__ float*)(dbetaAddr));
            LoadOneElement<WEIGHT_TYPE>(gammaAddr, gammaValue, preg, 0);
            // 与 golden/torch 同构：dx = ((dy - dbeta/R) - (xhat*dgamma)/R) * gamma * rstd；
            // 除法用 (rhi + rlo) 的 FMA 形式（单次舍入，等价真除法），rlo 按**精确整数 R**
            // 补残差（见 comp_sum.h：只修正到 fl32(R) 时整通道带 (R-fl32(R))/R 的同号偏差）。
            MicroAPI::Duplicate(rhiReg, hRecipValue, preg);
            CompExactDivisorReciprocalLo(rlo, rhiReg, static_cast<float>(r1Dim_ * r0Dim_), hNegDelta, preg);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dbetaOverR, dbetaValue, rlo, preg);
            MicroAPI::MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(dbetaOverR, dbetaValue, rhiReg, preg);
            uint32_t sregMask = r;
            for (uint16_t j = 0; j < loopTimes; j++) {
                pregR = MicroAPI::UpdateMask<float>(sregMask);
                LoadOneTensor<DY_TYPE>(xSrcAddr, xValue, pregR, VL_FP32 * j);
                LoadOneTensor<DY_TYPE>(dyAddr, dy, pregR, VL_FP32 * j);
                MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(u, dy, dbetaOverR, pregR);
                MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(xSubMean, xValue, meanValue, pregR);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xHat, xSubMean, rstdValue, pregR);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(p, xHat, dgammaValue, pregR);
                MicroAPI::Abs(pMag, p, pregR);
                MicroAPI::CompareScalar<float, CMPMODE::LE>(pFinite, pMag, FP32_MAX_FINITE, pregR);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(q, p, rlo, pFinite);
                MicroAPI::MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(q, p, rhiReg, pregR);
                MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(t, u, q, pregR);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(t, t, gammaValue, pregR);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dx, t, rstdValue, pregR);
                StoreOneTensor<DY_TYPE>(xDstAddr, dx, pregR, VL_FP32 * j);
            }
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;
    TBuf<TPosition::VECCALC> binaryAddBuf_;
    TBuf<TPosition::VECCALC> compAccBuf_;

    GlobalTensor<DY_TYPE> dyGm_, xGm_, dxGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, rstdGm_;

    LocalTensor<float> mean_;
    LocalTensor<float> rstd_;
    LocalTensor<WEIGHT_TYPE> gamma_;
    LocalTensor<float> dgamma_;
    LocalTensor<float> dbeta_;

    event_t V_MTE3_EVENT;
    event_t MTE3_MTE2_EVENT;

    int64_t r1Dim_{0};
    int64_t aDim_{0};
    int64_t r0Dim_{0};
    int64_t blockNum_{0};
    int64_t aDimPerCore_{0};
    int64_t gmOffset_{0};
    int64_t ubRDimLoopNum_{0};
    int64_t ubRDimTailTailLoopNum_{0};
    int64_t ubRDimTailLoopNum_{0};
    int64_t tailLoop_{0};
    int64_t sumLoop_{0};
    bool isTailLoop2_{true};

    uint32_t binaryAddQuotient_{0};
    uint32_t binaryAddK_{0};
    uint32_t binaryAddLastNum_{0};
    uint32_t halfBinaryAddQuotient_{0};
    uint32_t halfBinaryAddK_{0};
    uint32_t halfBinaryAddLastNum_{0};
    uint32_t ubRDimFactor_{0};
    uint32_t ubRDimTailFactor_{0};
    uint32_t ubRDimTailFactorAlign_{0};
    uint32_t ubRDimTailTail_{0};
    uint32_t xBufElemNum_{0};
    uint32_t tailFactorAlign_{0};
};
} // namespace BNGRARRecomputeSplitR0
#endif // __BATCH_NORM_GRAD_EXT2_RECOMPUTE_SPLIT_RO_REGBASE_H_
