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
 * \file batch_norm3d_grad_recompute_split_r0_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM3D_GRAD_RECOMPUTE_SPLIT_R0_REGBASE_H_
#define __BATCH_NORM3D_GRAD_RECOMPUTE_SPLIT_R0_REGBASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm3d_grad_common.h"

namespace BNGRARRecomputeSplitR0 {
using namespace AscendC;
using namespace BatchNorm3DGrad;
static constexpr int64_t UB_ADD_BUF = 3 * 1024; // dbeta/dgamma 单标量输出缓冲
// 通道级两肢（hi, lo）精确累加器：任何时刻 hi+lo 精确等于该 lane 已累加元素之和。
// 一个通道只开一对，Process 逐个通道串行处理，切通道时在 ProcessPre 里清零。
static constexpr uint32_t ACC_LEN = BatchNorm3DGrad::VL_FP32;
static constexpr uint32_t ACC_DBETA_HI = 0;
static constexpr uint32_t ACC_DBETA_LO = ACC_LEN;
static constexpr uint32_t ACC_DGAMMA_HI = ACC_LEN * BatchNorm3DGrad::TWO;
static constexpr uint32_t ACC_DGAMMA_LO = ACC_LEN * BatchNorm3DGrad::THREE;
static constexpr uint32_t ACC_BUF_LEN = ACC_LEN * BatchNorm3DGrad::FOUR;

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1>
class BatchNorm3DGradRARRecomputeSplitR0 {
public:
    __aicore__ inline BatchNorm3DGradRARRecomputeSplitR0(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR mean, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx,
                                GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace,
                                const BatchNorm3DGradRARRecomputeTilingData* tilingData)
    {
        const BatchNorm3DGradBaseTilingData baseTilingData = tilingData->baseTilingData;
        r1Dim_ = baseTilingData.r1Dim;
        aDim_ = baseTilingData.aDim;
        r0Dim_ = baseTilingData.r0Dim;
        blockNum_ = baseTilingData.blockNum;
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
        isTailLoop2_ = ubRDimTailTail_ > 0 && ubRDimTailTailLoopNum_ == BatchNorm3DGrad::TWO;
        tailLoop_ = ubRDimTailTailLoopNum_ == BatchNorm3DGrad::TWO ? ubRDimTailLoopNum_ / BatchNorm3DGrad::TWO :
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

        int64_t dyBufSize = tailFactorAlign_ * BatchNorm3DGrad::TWO * sizeof(DY_TYPE);
        int64_t xBufSize = ubRDimTailFactorAlign_ * BatchNorm3DGrad::TWO * sizeof(float);
        xBufElemNum_ = xBufSize / sizeof(DY_TYPE);

        pipe_.InitBuffer(dyInQue_, BUFFER_NUM, dyBufSize);
        pipe_.InitBuffer(xInQue_, BUFFER_NUM, xBufSize);
        pipe_.InitBuffer(meanInQue_, BUFFER_NUM, BatchNorm3DGrad::ONE_BLK_SIZE);
        pipe_.InitBuffer(rstdInQue_, BUFFER_NUM, BatchNorm3DGrad::ONE_BLK_SIZE);
        pipe_.InitBuffer(gammaInQue_, BUFFER_NUM, BatchNorm3DGrad::ONE_BLK_SIZE);
        pipe_.InitBuffer(dbetaOutQue_, 1, UB_ADD_BUF);
        pipe_.InitBuffer(dgammaOutQue_, 1, UB_ADD_BUF);
        pipe_.InitBuffer(accBuf_, ACC_BUF_LEN * sizeof(float));

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
        InitAccumulators();
        mean_ = meanInQue_.template DeQue<float>();
        rstd_ = rstdInQue_.template DeQue<float>();
        gamma_ = gammaInQue_.template DeQue<WEIGHT_TYPE>();
        dgamma_ = dgammaOutQue_.template AllocTensor<float>();
        dbeta_ = dbetaOutQue_.template AllocTensor<float>();
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
        CalcDbetaTailPre(dyLocal, tailLen);
        CopyInXTwo(offset1, offset2, tailLen);
        LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
        CalcDgammaTailPre(xLocal, dyLocal, tailLen);
        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessHalfMain(int64_t offset1)
    {
        CopyInDyOne(offset1, ubRDimTailFactor_);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        CalcDbetaPre(dyLocal, true);
        CopyInXOne(offset1, ubRDimTailFactor_);
        LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
        CalcDgammaPre(xLocal, dyLocal, true);
        dyInQue_.FreeTensor(dyLocal);
        xInQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ProcessMain(int64_t offset1)
    {
        CopyInDyOne(offset1, ubRDimFactor_);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        CalcDbetaPre(dyLocal, false);
        CopyInXOne(offset1, ubRDimFactor_);
        LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
        CalcDgammaPre(xLocal, dyLocal, false);
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
        for (int64_t k = 0; k < r1Dim_; k++) {
            r1Offset = k * r0Dim_ * aDim_;
            for (int64_t j = 0; j < ubRDimTailLoopNum_; j++) {
                int64_t offset1 = offset + r1Offset + ubRDimTailFactor_ * j;
                int64_t offset2 = offset1 + tailOffset;
                ProcessTail(offset1, offset2, ubRDimTailFactor_);
            }

            int64_t tailtailOffset = tailOffset + ubRDimTailLoopNum_ * ubRDimTailFactor_;
            int64_t offset1 = offset + r1Offset + ubRDimTailFactor_ * ubRDimTailLoopNum_;
            int64_t offset2 = offset + r1Offset + tailtailOffset;
            ProcessTailTail(offset1, offset2);

            for (int64_t j = tailLoop_; j < ubRDimLoopNum_; j++) {
                int64_t mainOffset = offset + r1Offset + ubRDimFactor_ * j;
                ProcessMain(mainOffset);
            }
        }
        FinalizeDbetaAndDgamma();
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
            CopyIn(xLocal, xGm_, xBufElemNum_ / BatchNorm3DGrad::TWO, offset1, ubRDimTailFactor_);
            CopyIn(xLocal, xGm_, xBufElemNum_ / BatchNorm3DGrad::TWO + tailFactorAlign_, offset2, tailLen);
        }
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInXOne(int64_t offset1, uint32_t copyLen)
    {
        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        if constexpr (IsSameType<DY_TYPE, float>::value) {
            CopyIn(xLocal, xGm_, 0, offset1, copyLen);
        } else {
            CopyIn(xLocal, xGm_, xBufElemNum_ / BatchNorm3DGrad::TWO, offset1, copyLen);
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

    // ==================== 精确（无误差）归约：通道级 (hi, lo) 两肢累加 ====================
    // 每个 lane 独立维护一对 (hi, lo)，任何时刻 hi + lo 精确等于该 lane 已累加元素之和。
    // 每个 UB 块开始时把 (hi, lo) 从 UB 读进寄存器，块内逐 VL 块做 Knuth TwoSum，块末写回 UB
    // （累加器不能跨 CopyIn 等待常驻寄存器；块内 ~240 个 VL 块，每块读写一次的代价可忽略）。

    // 累加器清零：Process 逐个通道串行处理，每通道开头重新开始
    __aicore__ inline void InitAccumulators()
    {
        LocalTensor<float> acc = accBuf_.Get<float>();
        __local_mem__ float* accAddr = (__local_mem__ float*)acc.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> zeroReg;
            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(zeroReg, 0.0f, pregMain);
            MicroAPI::DataCopy(accAddr + ACC_DBETA_HI, zeroReg, pregMain);
            MicroAPI::DataCopy(accAddr + ACC_DBETA_LO, zeroReg, pregMain);
            MicroAPI::DataCopy(accAddr + ACC_DGAMMA_HI, zeroReg, pregMain);
            MicroAPI::DataCopy(accAddr + ACC_DGAMMA_LO, zeroReg, pregMain);
        }
    }

    // dy 连续区间 [0, len) 逐 VL 块精确累加进 (hi, lo)
    __aicore__ inline void AccumulateDyRange(const __local_mem__ DY_TYPE* dyAddr, MicroAPI::RegTensor<float>& hi,
                                             MicroAPI::RegTensor<float>& lo, uint32_t len)
    {
        MicroAPI::RegTensor<float> val;
        MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        uint16_t fullTimes = len / VL_FP32;
        for (uint16_t i = 0; i < fullTimes; i++) {
            LoadOneTensor<DY_TYPE>(dyAddr, val, pregMain, i * VL_FP32);
            AccumulateWithError(hi, lo, val, pregMain);
        }
        uint32_t tailLen = len - static_cast<uint32_t>(fullTimes) * VL_FP32;
        if (tailLen > 0) {
            // 不足一个 VL 的 lane 先补 0，再仍然用满 mask 累加：TwoSum 里加 0 是精确恒等，
            // 这样 (hi, lo) 的状态与掩码模式无关，块与块之间衔接严格连续。
            MicroAPI::RegTensor<float> zeroReg;
            MicroAPI::MaskReg pregTail = MicroAPI::UpdateMask<float>(tailLen);
            MicroAPI::Duplicate(zeroReg, 0.0f, pregMain);
            LoadOneTensor<DY_TYPE>(dyAddr, val, pregTail, static_cast<uint32_t>(fullTimes) * VL_FP32);
            MicroAPI::Select(val, val, zeroReg, pregTail);
            AccumulateWithError(hi, lo, val, pregMain);
        }
    }

    // (x - mean) * rstd * dy 连续区间 [0, len) 逐 VL 块精确累加进 (hi, lo)
    // 每元素乘 rstd：与 golden / torch / CalDxPerA 对 xhat 的定义一致。
    // （若改成"通道末对整通道和乘一次 rstd"，会变成 round(round(S)*rstd) 的双舍入，实测掉 1 ULP。）
    // rstd/mean 都是每通道常量，按块广播装载一次（每个 __VEC_SCOPE__ 一次），不在 VL 循环里重复 load。
    __aicore__ inline void AccumulateXSubMeanMulDyRange(const __local_mem__ DY_TYPE* xAddr,
                                                        const __local_mem__ DY_TYPE* dyAddr,
                                                        MicroAPI::RegTensor<float>& meanValue,
                                                        MicroAPI::RegTensor<float>& rstdValue,
                                                        MicroAPI::RegTensor<float>& hi, MicroAPI::RegTensor<float>& lo,
                                                        uint32_t len)
    {
        MicroAPI::RegTensor<float> xValue;
        MicroAPI::RegTensor<float> dyValue;
        MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        uint16_t fullTimes = len / VL_FP32;
        for (uint16_t i = 0; i < fullTimes; i++) {
            LoadOneTensor<DY_TYPE>(xAddr, xValue, pregMain, i * VL_FP32);
            LoadOneTensor<DY_TYPE>(dyAddr, dyValue, pregMain, i * VL_FP32);
            MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(xValue, xValue, meanValue, pregMain);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xValue, xValue, rstdValue, pregMain);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xValue, xValue, dyValue, pregMain);
            AccumulateWithError(hi, lo, xValue, pregMain);
        }
        uint32_t tailLen = len - static_cast<uint32_t>(fullTimes) * VL_FP32;
        if (tailLen > 0) {
            MicroAPI::RegTensor<float> zeroReg;
            MicroAPI::MaskReg pregTail = MicroAPI::UpdateMask<float>(tailLen);
            MicroAPI::Duplicate(zeroReg, 0.0f, pregMain);
            LoadOneTensor<DY_TYPE>(xAddr, xValue, pregTail, static_cast<uint32_t>(fullTimes) * VL_FP32);
            LoadOneTensor<DY_TYPE>(dyAddr, dyValue, pregTail, static_cast<uint32_t>(fullTimes) * VL_FP32);
            MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(xValue, xValue, meanValue, pregTail);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xValue, xValue, rstdValue, pregTail);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xValue, xValue, dyValue, pregTail);
            MicroAPI::Select(xValue, xValue, zeroReg, pregTail);
            AccumulateWithError(hi, lo, xValue, pregMain);
        }
    }

    // fp32 输入：golden 元素项为 fp64 级，须元素项精确（TwoSum 减法误差 + 两次 TwoProduct），
    // 误差肢独立累加、块尾一次性并入 (hi,lo)。fp16 输入的 golden 是 fp32 级，不走这里。
    __aicore__ inline void AccumulateXSubMeanMulDyRangeExact(const __local_mem__ DY_TYPE* xAddr,
                                                             const __local_mem__ DY_TYPE* dyAddr,
                                                             MicroAPI::RegTensor<float>& meanValue,
                                                             MicroAPI::RegTensor<float>& rstdValue,
                                                             MicroAPI::RegTensor<float>& hi,
                                                             MicroAPI::RegTensor<float>& lo, uint32_t len)
    {
        MicroAPI::RegTensor<float> xValue, dyValue, termErr, errAcc;
        MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(errAcc, 0.0f, pregMain);
        uint16_t fullTimes = len / VL_FP32;
        for (uint16_t i = 0; i < fullTimes; i++) {
            LoadOneTensor<DY_TYPE>(xAddr, xValue, pregMain, i * VL_FP32);
            LoadOneTensor<DY_TYPE>(dyAddr, dyValue, pregMain, i * VL_FP32);
            BatchNorm3DGrad::ExactTermXSubMeanRstdMulDy(xValue, termErr, xValue, meanValue, rstdValue, dyValue,
                                                        pregMain);
            AccumulateWithError(hi, lo, xValue, pregMain);
            MicroAPI::Add<float, MicroAPI::MaskMergeMode::MERGING>(errAcc, errAcc, termErr, pregMain);
        }
        uint32_t tailLen = len - static_cast<uint32_t>(fullTimes) * VL_FP32;
        if (tailLen > 0) {
            MicroAPI::RegTensor<float> zeroReg;
            MicroAPI::MaskReg pregTail = MicroAPI::UpdateMask<float>(tailLen);
            MicroAPI::Duplicate(zeroReg, 0.0f, pregMain);
            LoadOneTensor<DY_TYPE>(xAddr, xValue, pregTail, static_cast<uint32_t>(fullTimes) * VL_FP32);
            LoadOneTensor<DY_TYPE>(dyAddr, dyValue, pregTail, static_cast<uint32_t>(fullTimes) * VL_FP32);
            BatchNorm3DGrad::ExactTermXSubMeanRstdMulDy(xValue, termErr, xValue, meanValue, rstdValue, dyValue,
                                                        pregTail);
            MicroAPI::Select(xValue, xValue, zeroReg, pregTail);
            MicroAPI::Select(termErr, termErr, zeroReg, pregTail);
            AccumulateWithError(hi, lo, xValue, pregMain);
            MicroAPI::Add<float, MicroAPI::MaskMergeMode::MERGING>(errAcc, errAcc, termErr, pregMain);
        }
        BatchNorm3DGrad::ClampResidual(errAcc, pregMain); // 项含 inf 时误差肢为 NaN，钳 0 保持 inf 传播
        AccumulateWithError(hi, lo, errAcc, pregMain);
    }

    // 尾块：UB 里是"两半拼接"的（[0, ubRDimTailFactor_) 来自 offset1，[tailFactorAlign_, +tailLen) 来自 offset2），
    // 两段分别累加，元素不多算也不少算
    __aicore__ inline void CalcDbetaTailPre(LocalTensor<DY_TYPE>& dy, uint32_t tailLen)
    {
        LocalTensor<float> acc = accBuf_.Get<float>();
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dy.GetPhyAddr();
        __local_mem__ float* accAddr = (__local_mem__ float*)acc.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> hi;
            MicroAPI::RegTensor<float> lo;
            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::DataCopy(hi, accAddr + ACC_DBETA_HI);
            MicroAPI::DataCopy(lo, accAddr + ACC_DBETA_LO);
            AccumulateDyRange(dyAddr, hi, lo, ubRDimTailFactor_);
            AccumulateDyRange(dyAddr + tailFactorAlign_, hi, lo, tailLen);
            MicroAPI::DataCopy(accAddr + ACC_DBETA_HI, hi, pregMain);
            MicroAPI::DataCopy(accAddr + ACC_DBETA_LO, lo, pregMain);
        }
    }

    // dbeta 主块：dy 连续单块 [0, processR)
    __aicore__ inline void CalcDbetaPre(LocalTensor<DY_TYPE>& dy, bool isHalf)
    {
        LocalTensor<float> acc = accBuf_.Get<float>();
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dy.GetPhyAddr();
        __local_mem__ float* accAddr = (__local_mem__ float*)acc.GetPhyAddr();
        uint32_t processR = isHalf ? ubRDimTailFactor_ : ubRDimFactor_;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> hi;
            MicroAPI::RegTensor<float> lo;
            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::DataCopy(hi, accAddr + ACC_DBETA_HI);
            MicroAPI::DataCopy(lo, accAddr + ACC_DBETA_LO);
            AccumulateDyRange(dyAddr, hi, lo, processR);
            MicroAPI::DataCopy(accAddr + ACC_DBETA_HI, hi, pregMain);
            MicroAPI::DataCopy(accAddr + ACC_DBETA_LO, lo, pregMain);
        }
    }

    __aicore__ inline void CalcDgammaTailPre(LocalTensor<DY_TYPE>& x, LocalTensor<DY_TYPE>& dy, uint32_t tailLen)
    {
        LocalTensor<float> acc = accBuf_.Get<float>();
        __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)x.GetPhyAddr();
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dy.GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)mean_.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd_.GetPhyAddr();
        __local_mem__ float* accAddr = (__local_mem__ float*)acc.GetPhyAddr();
        if constexpr (IsSameType<DY_TYPE, half>::value) {
            xAddr += xBufElemNum_ / BatchNorm3DGrad::TWO;
        }
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> hi;
            MicroAPI::RegTensor<float> lo;
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float*)(meanAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float*)(rstdAddr));
            MicroAPI::DataCopy(hi, accAddr + ACC_DGAMMA_HI);
            MicroAPI::DataCopy(lo, accAddr + ACC_DGAMMA_LO);
            if constexpr (IsSameType<DY_TYPE, float>::value) {
                AccumulateXSubMeanMulDyRangeExact(xAddr, dyAddr, meanValue, rstdValue, hi, lo, ubRDimTailFactor_);
                AccumulateXSubMeanMulDyRangeExact(xAddr + tailFactorAlign_, dyAddr + tailFactorAlign_, meanValue,
                                                  rstdValue, hi, lo, tailLen);
            } else {
                AccumulateXSubMeanMulDyRange(xAddr, dyAddr, meanValue, rstdValue, hi, lo, ubRDimTailFactor_);
                AccumulateXSubMeanMulDyRange(xAddr + tailFactorAlign_, dyAddr + tailFactorAlign_, meanValue, rstdValue,
                                             hi, lo, tailLen);
            }
            MicroAPI::DataCopy(accAddr + ACC_DGAMMA_HI, hi, pregMain);
            MicroAPI::DataCopy(accAddr + ACC_DGAMMA_LO, lo, pregMain);
        }
    }

    __aicore__ inline void CalcDgammaPre(LocalTensor<DY_TYPE>& x, LocalTensor<DY_TYPE>& dy, bool isHalf)
    {
        LocalTensor<float> acc = accBuf_.Get<float>();
        __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)x.GetPhyAddr();
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dy.GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)mean_.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd_.GetPhyAddr();
        __local_mem__ float* accAddr = (__local_mem__ float*)acc.GetPhyAddr();
        if constexpr (IsSameType<DY_TYPE, half>::value) {
            xAddr += xBufElemNum_ / BatchNorm3DGrad::TWO;
        }
        uint32_t processR = isHalf ? ubRDimTailFactor_ : ubRDimFactor_;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> hi;
            MicroAPI::RegTensor<float> lo;
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::MaskReg pregMain = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float*)(meanAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float*)(rstdAddr));
            MicroAPI::DataCopy(hi, accAddr + ACC_DGAMMA_HI);
            MicroAPI::DataCopy(lo, accAddr + ACC_DGAMMA_LO);
            if constexpr (IsSameType<DY_TYPE, float>::value) {
                AccumulateXSubMeanMulDyRangeExact(xAddr, dyAddr, meanValue, rstdValue, hi, lo, processR);
            } else {
                AccumulateXSubMeanMulDyRange(xAddr, dyAddr, meanValue, rstdValue, hi, lo, processR);
            }
            MicroAPI::DataCopy(accAddr + ACC_DGAMMA_HI, hi, pregMain);
            MicroAPI::DataCopy(accAddr + ACC_DGAMMA_LO, lo, pregMain);
        }
    }

    // 通道收尾：把 VL_FP32 个 lane 的两肢累加器精确合成一个标量。
    // 逐 lane 做标量 Knuth TwoSum（lane 数 < 64 只做一次/通道），最后 hi + lo 这一次舍入就是整条归约唯一一次舍入。
    __aicore__ inline void ReduceAccumulatorToScalar(uint32_t hiOffset, uint32_t loOffset, LocalTensor<float>& dst)
    {
        LocalTensor<float> acc = accBuf_.Get<float>();
        PipeBarrier<PIPE_ALL>();
        float hiScalar = 0.0f;
        float loScalar = 0.0f;
        for (uint32_t i = 0; i < VL_FP32; i++) {
            float value = acc.GetValue(hiOffset + i);
            float residual = acc.GetValue(loOffset + i);
            // total = fl(hiScalar + value)，error 为这次加法的精确误差；residual 量级 ~eps·S，平加即可
            volatile float total = hiScalar + value;
            volatile float back = total - hiScalar;
            volatile float error = (hiScalar - (total - back)) + (value - back);
            loScalar = loScalar + (error + residual);
            hiScalar = total;
        }
        // 残差只在有限域有意义：输入含 ±inf 时 hi 会溢出成 inf，两次相加的误差退化成 NaN，
        // 直接把 lo 加回去会把 inf 污染成 NaN。与 common.h 里 ClampResidual 的语义保持一致。
        if (!(loScalar <= BNG_MAX_FINITE_FP32 && loScalar >= -BNG_MAX_FINITE_FP32)) {
            loScalar = 0.0f;
        }
        dst.SetValue(0, hiScalar + loScalar);
        PipeBarrier<PIPE_ALL>();
    }

    // dbeta = Σdy，dgamma = Σ((x - mean)·rstd·dy)：rstd 在累加时按元素乘（与 golden/torch/CalDxPerA 的 xhat 定义一致）
    __aicore__ inline void FinalizeDbetaAndDgamma()
    {
        ReduceAccumulatorToScalar(ACC_DBETA_HI, ACC_DBETA_LO, dbeta_);
        ReduceAccumulatorToScalar(ACC_DGAMMA_HI, ACC_DGAMMA_LO, dgamma_);
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
            xSrcAddr += xBufElemNum_ / BatchNorm3DGrad::TWO;
        }

        uint16_t loopTimes = CeilDiv(r, VL_FP32);
        // A/B 实测（全量 550）：真除法 dbeta/R、dgamma/R 分开相减 = 12 FAIL，
        // 合成后乘 1/R = 13 FAIL（L1_319/L1_537 额外失败），故采用真除法口径。
        float countF = (float)(r1Dim_ * r0Dim_);
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
            MicroAPI::RegTensor<float> rstdMulSubDy;
            MicroAPI::MaskReg pregR;
            MicroAPI::RegTensor<float> dy;
            MicroAPI::RegTensor<float> mulDgamma;
            MicroAPI::RegTensor<float> addDbeta;
            MicroAPI::RegTensor<float> divH;
            MicroAPI::RegTensor<float> subDy;
            MicroAPI::RegTensor<float> dx;
            MicroAPI::RegTensor<float> countReg;
            // dx 双浮点分子（fp32 路径）用寄存器
            MicroAPI::RegTensor<float> s, bp, xe, e1, e2, xHatLo, tH, tE, q, b, be, corr, nLo, qb, bb, bbe, corrb, rem,
                remb;
            MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(countReg, countF, preg);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float*)(meanAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float*)(rstdAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue,
                                                                        (__local_mem__ float*)(dgammaAddr));
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue, (__local_mem__ float*)(dbetaAddr));
            LoadOneElement<WEIGHT_TYPE>(gammaAddr, gammaValue, preg, 0);
            uint32_t sregMask = r;
            for (uint16_t j = 0; j < loopTimes; j++) {
                pregR = MicroAPI::UpdateMask<float>(sregMask);
                LoadOneTensor<DY_TYPE>(xSrcAddr, xValue, pregR, VL_FP32 * j);
                LoadOneTensor<DY_TYPE>(dyAddr, dy, pregR, VL_FP32 * j);
                if constexpr (IsSameType<DY_TYPE, float>::value) {
                    // fp32 输入时 golden 的 dx 为 fp64 级：对消点项级舍入会吞掉结果，分子双浮点。
                    // N = dy - dbeta/R - xhat*dgamma/R，逐步带精确误差；末两级乘保持输出档。
                    // (xHat, xe) = exact((x-mean)*rstd)
                    MicroAPI::Sub(s, xValue, meanValue, pregR);
                    MicroAPI::Sub(bp, s, xValue, pregR);
                    MicroAPI::Sub(xe, s, bp, pregR);
                    MicroAPI::Sub(xe, xValue, xe, pregR);
                    MicroAPI::Add(bp, meanValue, bp, pregR);
                    MicroAPI::Neg(bp, bp, pregR);
                    MicroAPI::Add(xe, xe, bp, pregR);         // xe = 减法误差
                    MicroAPI::Mul(xe, xe, rstdValue, pregR);  // xe 归一到 xhat 量纲
                    MicroAPI::Mul(xHat, s, rstdValue, pregR); // xHat = fl(s*rstd)
                    AscendC::Reg::Move(e1, s);
                    MicroAPI::Neg(e2, xHat, pregR);
                    AscendC::Reg::FusedMulDstAdd(e1, rstdValue, e2, pregR); // e1 = s*rstd-xHat 精确
                    MicroAPI::Add(xe, xe, e1, pregR);                       // xHat 的总精确误差
                    AscendC::Reg::Move(xHatLo, xe);
                    // (tH, tE) = xhat*dgamma 精确
                    MicroAPI::Mul(tH, xHat, dgammaValue, pregR);
                    AscendC::Reg::Move(tE, xHat);
                    MicroAPI::Neg(e1, tH, pregR);
                    AscendC::Reg::FusedMulDstAdd(tE, dgammaValue, e1, pregR); // tE = xhat*dgamma-tH
                    MicroAPI::Mul(e1, xHatLo, dgammaValue, pregR);
                    MicroAPI::Add(tE, tE, e1, pregR); // + xe*dgamma
                    // divH = t/R 双浮点：q + corr
                    MicroAPI::Div(q, tH, countReg, pregR);
                    MicroAPI::Mul(b, q, countReg, pregR);
                    AscendC::Reg::Move(be, q);
                    MicroAPI::Neg(e1, b, pregR);
                    AscendC::Reg::FusedMulDstAdd(be, countReg, e1, pregR); // be = q*R-b 精确
                    MicroAPI::Sub(rem, tH, b, pregR);
                    MicroAPI::Sub(rem, rem, be, pregR);
                    MicroAPI::Add(rem, rem, tE, pregR);
                    MicroAPI::Div(corr, rem, countReg, pregR);
                    // dbeta/R 双浮点：qb + corrb
                    MicroAPI::Div(qb, dbetaValue, countReg, pregR);
                    MicroAPI::Mul(bb, qb, countReg, pregR);
                    AscendC::Reg::Move(bbe, qb);
                    MicroAPI::Neg(e1, bb, pregR);
                    AscendC::Reg::FusedMulDstAdd(bbe, countReg, e1, pregR);
                    MicroAPI::Sub(remb, dbetaValue, bb, pregR);
                    MicroAPI::Sub(remb, remb, bbe, pregR);
                    MicroAPI::Div(corrb, remb, countReg, pregR);
                    // N = (dy - qb) - q，TwoSum 逐步，lo 收全部修正
                    MicroAPI::Neg(e1, qb, pregR);
                    MicroAPI::Add(dx, dy, e1, pregR); // n1 = fl(dy-qb)
                    MicroAPI::Sub(bp, dx, dy, pregR);
                    MicroAPI::Sub(nLo, dx, bp, pregR);
                    MicroAPI::Sub(nLo, dy, nLo, pregR);
                    MicroAPI::Sub(bp, e1, bp, pregR);
                    MicroAPI::Add(nLo, nLo, bp, pregR); // err(dy-qb)
                    MicroAPI::Neg(e1, q, pregR);
                    MicroAPI::Add(e2, dx, e1, pregR); // n2 = fl(n1-q)
                    MicroAPI::Sub(bp, e2, dx, pregR);
                    MicroAPI::Sub(b, e2, bp, pregR);
                    MicroAPI::Sub(b, dx, b, pregR);
                    MicroAPI::Sub(bp, e1, bp, pregR);
                    MicroAPI::Add(b, b, bp, pregR); // err(n1-q)
                    MicroAPI::Add(nLo, nLo, b, pregR);
                    // N = dy - (qb+corrb) - (q+corr)：除法修正项是减不是加
                    MicroAPI::Sub(nLo, nLo, corrb, pregR);
                    MicroAPI::Sub(nLo, nLo, corr, pregR); // N = (n2=e2, nLo)
                    MicroAPI::Add(dx, e2, nLo, pregR);    // 双肢合一（结果档舍入）
                    MicroAPI::Mul(dx, dx, gammaValue, pregR);
                    MicroAPI::Mul(dx, dx, rstdValue, pregR);
                } else {
                    MicroAPI::Sub(xSubMean, xValue, meanValue, pregR);
                    MicroAPI::Mul(xHat, xSubMean, rstdValue, pregR);
                    MicroAPI::Mul(mulDgamma, xHat, dgammaValue, pregR);
                    // dx = (dy - doffset/R - xhat*dscale/R) * scale * rstd（真除法，与 golden/契约/三方同序）
                    MicroAPI::Div(divH, mulDgamma, countReg, pregR);
                    MicroAPI::Div(addDbeta, dbetaValue, countReg, pregR);
                    MicroAPI::Sub(subDy, dy, addDbeta, pregR);
                    MicroAPI::Sub(subDy, subDy, divH, pregR);
                    MicroAPI::Mul(dx, subDy, gammaValue, pregR);
                    MicroAPI::Mul(dx, dx, rstdValue, pregR);
                }
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
    TBuf<TPosition::VECCALC> accBuf_;

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

    uint32_t ubRDimFactor_{0};
    uint32_t ubRDimTailFactor_{0};
    uint32_t ubRDimTailFactorAlign_{0};
    uint32_t ubRDimTailTail_{0};
    uint32_t xBufElemNum_{0};
    uint32_t tailFactorAlign_{0};
};
} // namespace BNGRARRecomputeSplitR0
#endif // __BATCH_NORM3D_GRAD_RECOMPUTE_SPLIT_RO_REGBASE_H_
