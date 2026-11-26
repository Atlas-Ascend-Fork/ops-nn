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
 * \file batch_norm_grad_ext2_rar_split_core_r0.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_EXT2_SPLIT_CORE_R0_H__
#define __BATCH_NORM_GRAD_EXT2_SPLIT_CORE_R0_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm_grad_ext2_common.h"
#include "batch_norm_grad_ext2_comp_sum.h"
#include "batch_norm_grad_ext2_rar_split_core_r1.h"

namespace BatchNormGradExt2 {
using namespace AscendC;
using namespace AscendC::MicroAPI;

// Local only: this template uses CONST_TWO / BUFFER_NUM / BLOCK_SIZE from the (unmodified)
// rar_split_core_r1 header, which does not define CONST_THREE.
static constexpr int64_t CONST_THREE = 3;

template <typename DY_DTYPE, typename WEIGHT_DTYPE>
class BatchNormGradExt2RARSplitCoreR0 {
public:
    __aicore__ inline BatchNormGradExt2RARSplitCoreR0(TPipe* pipeIn,
                                                      const BatchNormGradExt2RARSplitCoreR0TilingData* tilingDataIn)
    {
        pipe_ = pipeIn;
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR mean, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx,
                                GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace)
    {
        blkIdx_ = GetBlockIdx();
        isLastCore_ = blkIdx_ == (tilingData_->usedCoreNums - 1) ? true : false;
        hRecipDen_ = tilingData_->r1Dim * tilingData_->r0Dim; // 精确归约长度 R（int64）
        // fl32(R) 在 R > 2^24 时不等于 R（本用例 26843545 -> 26843544），两者之差在这里一次算好，
        // 供 stage2 把 dscale/R 的商从「除以 fl32(R)」还原成「除以精确 R」。
        int64_t rLenExact = hRecipDen_;
        float rLenFp32 = static_cast<float>(rLenExact);
        hRecipVal_ = 1.0f / rLenFp32;
        hNegDelta_ = -static_cast<float>(rLenExact - static_cast<int64_t>(rLenFp32)); // R <= 2^53 时精确
        r0BlockOffset_ = blkIdx_ * tilingData_->r0Inner;

        meanGm_.SetGlobalBuffer((__gm__ float*)(mean));
        rstdGm_.SetGlobalBuffer((__gm__ float*)(rstd));
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)(gamma));
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)(dgamma));
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)(dbeta));

        int64_t wsStride = tilingData_->usedCoreNums * tilingData_->aDimAligned;
        dbetaTmpGm_.SetGlobalBuffer((__gm__ float*)(workspace));
        dbetaTmpLoGm_.SetGlobalBuffer((__gm__ float*)(workspace) + wsStride);
        dgammaTmpGm_.SetGlobalBuffer((__gm__ float*)(workspace) + wsStride * CONST_TWO);
        dgammaTmpLoGm_.SetGlobalBuffer((__gm__ float*)(workspace) + wsStride * CONST_THREE);

        dyGm_.SetGlobalBuffer((__gm__ DY_DTYPE*)(dy));
        xGm_.SetGlobalBuffer((__gm__ DY_DTYPE*)(x));
        dxGm_.SetGlobalBuffer((__gm__ DY_DTYPE*)(dx));
    }

    __aicore__ inline void Stage0InitBuffer()
    {
        // 二分相关buff初始化
        // r0InnerInnerStg0, r0InnerTailStg0 is aligned
        int64_t dyLen = tilingData_->r1InnerStg0 * tilingData_->r0InnerInnerStg0 * tilingData_->aInnerStg0;

        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));

        int64_t meanSize = tilingData_->aInnerAlignedStg0 * sizeof(float);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, meanSize);

        pipe_->InitBuffer(dbetaTmpOutQue_, BUFFER_NUM, meanSize * CONST_TWO);
        pipe_->InitBuffer(dgammaTmpOutQue_, BUFFER_NUM, meanSize * CONST_TWO);
        // double-fp32 累加器：S0/S1 各一组 hi/lo，按通道存放
        pipe_->InitBuffer(compAccBuf_, 4 * tilingData_->aInnerAlignedStg0 * sizeof(float));
    }

    __aicore__ inline void Stage1InitBuffer()
    {
        pipe_->Reset();
        int64_t aSizeIn = tilingData_->aInnerStg1 * sizeof(float) * tilingData_->usedCoreNums;
        pipe_->InitBuffer(wsInQue_, 1, 2 * aSizeIn);
        pipe_->InitBuffer(accBuf_, 2 * tilingData_->aInnerStg1 * sizeof(float));
        int64_t meanSize = tilingData_->aInnerStg1 * sizeof(float);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(dbetaTmpOutQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(dgammaTmpOutQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, tilingData_->aInnerStg1 * sizeof(WEIGHT_DTYPE));
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, tilingData_->aInnerStg1 * sizeof(WEIGHT_DTYPE));
    }

    __aicore__ inline void Stage2InitBuffer()
    {
        pipe_->Reset();

        int64_t dyLen = tilingData_->r1InnerStg2 * tilingData_->r0InnerInnerStg2 * tilingData_->aInnerStg2;
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));
        pipe_->InitBuffer(dxOutQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));

        int64_t meanSize = tilingData_->aInnerAlignedStg2 * sizeof(float);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(gammaInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(dbetaTmpInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(dgammaTmpInQue_, BUFFER_NUM, meanSize);
    }

    __aicore__ inline void Process()
    {
        // stage 0： 核内二分累加dbeta/dgamma
        Stage0InitBuffer();
        CalcDbetaDgammaInCore();

        // stage 1: 计算核间dbeta/dgamma
        SyncAll();
        if (blkIdx_ == 0) {
            Stage1InitBuffer();
            CalcDbetaDgamma();
        }

        // stage 2: 计算输出
        SyncAll();
        Stage2InitBuffer();
        CalcOutput();
    }

    // stage 0
    __aicore__ inline void CalcDbetaDgammaInCore()
    {
        for (int64_t i = 0; i < tilingData_->aOuterStg0; i++) {
            uint32_t curAInnerLen = i != (tilingData_->aOuterStg0 - 1) ? tilingData_->aInnerStg0 :
                                                                         tilingData_->aTailStg0;
            int64_t aOffset = i * tilingData_->aInnerStg0;

            CopyInMeanAndRstd(aOffset, curAInnerLen);
            LocalTensor<float> meanTensor = meanInQue_.template DeQue<float>();
            LocalTensor<float> rstdTensor = rstdInQue_.template DeQue<float>();
            __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor.GetPhyAddr();
            __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor.GetPhyAddr();

            InitCompAccA();
            CalcReduceSum(curAInnerLen, aOffset, meanLocal, rstdLocal);

            meanInQue_.FreeTensor(meanTensor);
            rstdInQue_.FreeTensor(rstdTensor);

            CopyOutDdTmp(curAInnerLen, aOffset);
        }
    }

    __aicore__ inline void CalcReduceSum(uint32_t curAInnerLen, int64_t aOffset, __local_mem__ float* meanLocal,
                                         __local_mem__ float* rstdLocal)
    {
        int64_t r0OuterInCore = tilingData_->r0InnerOuterStg0;
        int64_t r0InerIncore = tilingData_->r0InnerInnerStg0;
        int64_t r0TailIncore = tilingData_->r0InnerTailStg0;
        int64_t basicLoop = tilingData_->binAddBasicBlockLoop;
        int64_t mainLoop = tilingData_->binAddMainFoldCount;

        if (isLastCore_) {
            r0OuterInCore = tilingData_->r0TailOuterStg0;
            r0TailIncore = tilingData_->r0TailTailStg0;
            basicLoop = tilingData_->lastCoreBinAddBasicBlockLoop;
            mainLoop = tilingData_->lastCoreBinAddMainFoldCount;
        }

        for (int64_t basicBlockIdx = 0; basicBlockIdx < basicLoop; basicBlockIdx++) {
            int64_t r1OuterIdx = basicBlockIdx / r0OuterInCore;
            int64_t r0OuterIdx = basicBlockIdx % r0OuterInCore;
            int64_t dyOffset = r1OuterIdx * tilingData_->r1InnerStg0 * tilingData_->aDim * tilingData_->r0Dim +
                               aOffset * tilingData_->r0Dim + r0OuterIdx * tilingData_->r0InnerInnerStg0 +
                               r0BlockOffset_;
            uint32_t curR1Len = r1OuterIdx != (tilingData_->r1OuterStg0 - 1) ? tilingData_->r1InnerStg0 :
                                                                               tilingData_->r1TailStg0;
            uint32_t curR0Len = r0OuterIdx != (r0OuterInCore - 1) ? r0InerIncore : r0TailIncore;
            uint32_t curR0AlignedLen = curR0Len;
            if (r0OuterIdx == (r0OuterInCore - 1) && isLastCore_) {
                curR0AlignedLen = tilingData_->r0TailTailAlignedStg0;
            }

            ProcessMainBlock(meanLocal, rstdLocal, curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);

            if (basicBlockIdx < mainLoop) {
                int64_t r1OuterIdxFold = (basicBlockIdx + basicLoop) / r0OuterInCore;
                int64_t r0OuterIdxFold = (basicBlockIdx + basicLoop) % r0OuterInCore;
                int64_t dyOffsetFold = r1OuterIdxFold * tilingData_->r1InnerStg0 * tilingData_->aDim *
                                           tilingData_->r0Dim +
                                       aOffset * tilingData_->r0Dim + r0OuterIdxFold * tilingData_->r0InnerInnerStg0 +
                                       r0BlockOffset_;
                uint32_t curR1LenFold = r1OuterIdxFold != (tilingData_->r1OuterStg0 - 1) ? tilingData_->r1InnerStg0 :
                                                                                           tilingData_->r1TailStg0;
                uint32_t curR0LenFold = r0OuterIdxFold != (r0OuterInCore - 1) ? r0InerIncore : r0TailIncore;
                uint32_t curR0AlignedLenFold = curR0LenFold;
                if (r0OuterIdxFold == (r0OuterInCore - 1) && isLastCore_) {
                    curR0AlignedLenFold = tilingData_->r0TailTailAlignedStg0;
                }

                ProcessFoldBlock(meanLocal, rstdLocal, curAInnerLen, curR1LenFold, curR0LenFold, curR0AlignedLenFold,
                                 dyOffsetFold);
            }
        }

        if (basicLoop == 0) {
            int64_t dyOffset = aOffset * tilingData_->r0Dim + r0BlockOffset_;
            uint32_t r0AlignedLen = isLastCore_ ? tilingData_->r0TailTailAlignedStg0 : r0TailIncore;
            ProcessMainBlock(meanLocal, rstdLocal, curAInnerLen, tilingData_->r1TailStg0, r0TailIncore, r0AlignedLen,
                             dyOffset);
        }
    }

    __aicore__ inline void ProcessMainBlock(__local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
                                            const uint32_t curAInnerLen, const uint32_t curR1Len,
                                            const uint32_t curR0Len, const uint32_t curR0AlignedLen,
                                            const int64_t dyOffset)
    {
        CopyInDyAndX(curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);

        LocalTensor<DY_DTYPE> dyTensor = dyInQue_.DeQue<DY_DTYPE>();
        LocalTensor<DY_DTYPE> xTensor = xInQue_.DeQue<DY_DTYPE>();
        CalcDyAndX(dyTensor, xTensor, meanLocal, rstdLocal, curR0AlignedLen, curAInnerLen, curR1Len);
        xInQue_.FreeTensor(xTensor);
        dyInQue_.FreeTensor(dyTensor);
    }

    // fold cast fp32 加到main
    __aicore__ inline void ProcessFoldBlock(__local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
                                            const uint32_t curAInnerLen, const uint32_t curR1Len,
                                            const uint32_t curR0Len, const uint32_t curR0AlignedLen,
                                            const int64_t dyOffset)
    {
        CopyInDyAndX(curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);

        LocalTensor<DY_DTYPE> dyTensor = dyInQue_.DeQue<DY_DTYPE>();
        LocalTensor<DY_DTYPE> xTensor = xInQue_.DeQue<DY_DTYPE>();
        CalcDyAndX(dyTensor, xTensor, meanLocal, rstdLocal, curR0AlignedLen, curAInnerLen, curR1Len);
        xInQue_.FreeTensor(xTensor);
        dyInQue_.FreeTensor(dyTensor);
    }

    __aicore__ inline void CalcDyAndX(LocalTensor<DY_DTYPE>& dyTensor, LocalTensor<DY_DTYPE>& xTensor,
                                      __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
                                      uint32_t curR0AlignedLen, uint16_t curAInnerLen, uint16_t curR1Len)
    {
        __local_mem__ DY_DTYPE* dyLocal = (__local_mem__ DY_DTYPE*)dyTensor.GetPhyAddr();
        __local_mem__ DY_DTYPE* xLocal = (__local_mem__ DY_DTYPE*)xTensor.GetPhyAddr();
        __local_mem__ float* base = (__local_mem__ float*)compAccBuf_.Get<float>().GetPhyAddr();
        int64_t aAligned = tilingData_->aInnerAlignedStg0;
        __local_mem__ float* s0HiAddr = base;
        __local_mem__ float* s0LoAddr = base + aAligned;
        __local_mem__ float* s1HiAddr = base + CONST_TWO * aAligned;
        __local_mem__ float* s1LoAddr = base + CONST_THREE * aAligned;
        uint16_t loopNum = ops::CeilDiv(curR0AlignedLen, VL_FP32);

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            MaskReg pregMaskAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregOne = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            RegTensor<float> regDy, regX, laneS0Hi, laneS0Lo, laneS1Hi, laneS1Lo, prodHi, prodLo;
            for (uint16_t i = 0; i < curAInnerLen; i++) {
                uint32_t sreg = curR0AlignedLen;
                MicroAPI::Duplicate(laneS0Hi, 0.0f, pregMaskAll);
                MicroAPI::Duplicate(laneS0Lo, 0.0f, pregMaskAll);
                MicroAPI::Duplicate(laneS1Hi, 0.0f, pregMaskAll);
                MicroAPI::Duplicate(laneS1Lo, 0.0f, pregMaskAll);
                for (uint16_t j = 0; j < loopNum; j++) {
                    pregMask = UpdateMask<float>(sreg);
                    for (uint16_t k = 0; k < curR1Len; k++) {
                        uint32_t offset = i * curR1Len * curR0AlignedLen + k * curR0AlignedLen + j * VL_FP32;
                        LoadOneTensor<DY_DTYPE>(dyLocal, regDy, pregMask, offset);
                        LoadOneTensor<DY_DTYPE>(xLocal, regX, pregMask, offset);
                        Comp::CompAdd<DY_DTYPE>(laneS0Hi, laneS0Lo, regDy, pregMask);
                        if constexpr (IsSameType<DY_DTYPE, float>::value) {
                            Comp::CompAddProduct<DY_DTYPE>(laneS1Hi, laneS1Lo, regX, regDy, pregMask);
                        } else {
                            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(prodHi, regX, regDy, pregMask);
                            Comp::CompAdd<DY_DTYPE>(laneS1Hi, laneS1Lo, prodHi, pregMask);
                        }
                    }
                }
                // 本块只把 curR0AlignedLen 条 lane 填过数据（每个 lane 复用 curR1Len 次），
                // 其余恒 +0，归约树从对应层起跳（见 comp_sum.h CompReduceBeginWidth）。
                const uint32_t laneCount = Comp::CompLaneBound(static_cast<int64_t>(curR0AlignedLen) *
                                                               static_cast<int64_t>(curR1Len));
                Comp::CompReduceLanes(laneS0Hi, laneS0Lo, pregMaskAll, laneCount);
                Comp::CompReduceLanes(laneS1Hi, laneS1Lo, pregMaskAll, laneCount);
                DdAccumulateChannel(s0HiAddr + i, s0LoAddr + i, laneS0Hi, laneS0Lo, pregOne);
                DdAccumulateChannel(s1HiAddr + i, s1LoAddr + i, laneS1Hi, laneS1Lo, pregOne);
            }
        }
    }

    __aicore__ inline void DdAccumulateChannel(__local_mem__ float* hiAddr, __local_mem__ float* loAddr,
                                               RegTensor<float>& laneHi, RegTensor<float>& laneLo, MaskReg& one)
    {
        RegTensor<float> accHi, accLo;
        Comp::CompLoadLane0(hiAddr, loAddr, accHi, accLo, one);
        Comp::CompAddPair<DY_DTYPE>(accHi, accLo, laneHi, laneLo, one);
        Comp::CompStoreLane0(hiAddr, loAddr, accHi, accLo, one);
    }

    __aicore__ inline void InitCompAccA()
    {
        __local_mem__ float* base = (__local_mem__ float*)compAccBuf_.Get<float>().GetPhyAddr();
        // Clear exactly the 4 * aInnerAlignedStg0 floats of the accumulator.  The old form used
        // `4 * aInnerAlignedStg0 / VL_FP32` whole vectors, which is 0 as soon as
        // aInnerAlignedStg0 < VL_FP32 / 4 (A == 1 gives aInnerAlignedStg0 == 8, i.e. 32 < 64):
        // the per-channel accumulator then started from whatever UB held, which is one of the
        // reasons the A == 1 path produced garbage (2.686e23).
        uint32_t total = static_cast<uint32_t>(4 * tilingData_->aInnerAlignedStg0);
        uint16_t loopNum = static_cast<uint16_t>(ops::CeilDiv(total, VL_FP32));
        __VEC_SCOPE__
        {
            RegTensor<float> zero;
            MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(zero, 0.0f, all);
            uint32_t count = total;
            for (uint16_t i = 0; i < loopNum; ++i) {
                MaskReg pMask = MicroAPI::UpdateMask<float>(count);
                MicroAPI::DataCopy(base + i * VL_FP32, zero, pMask);
            }
        }
    }

    __aicore__ inline void CopyOutDdTmp(const uint32_t curAInnerLen, const int64_t aOffset)
    {
        int64_t gmOffset = blkIdx_ * tilingData_->aDimAligned + aOffset;
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = curAInnerLen * sizeof(float);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        __local_mem__ float* base = (__local_mem__ float*)compAccBuf_.Get<float>().GetPhyAddr();
        int64_t aAligned = tilingData_->aInnerAlignedStg0;

        // Same contract as the RAR-SplitCoreR1 stage 0: the queue is Alloc -> fill -> EnQue ->
        // DeQue -> DataCopyPad -> Free.  The previous code DeQue'd without a matching EnQue
        // (aivec 271 / "set and wait instructions do not match") and only copied the hi words,
        // leaving dbetaTmpLoGm_/dgammaTmpLoGm_ uninitialised while stage 1 read them.
        // The per-channel copy uses the broadcast/first-element form (one element in, one element
        // out) because the accumulator regions are only 2 * aInnerAlignedStg0 floats wide: a full
        // vector load would read past the TBuf when aInnerAlignedStg0 < VL_FP32 (A == 1).
        LocalTensor<float> dbetaOutTmpTensor = dbetaTmpOutQue_.template AllocTensor<float>();
        __local_mem__ float* dbOut = (__local_mem__ float*)dbetaOutTmpTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MaskReg pregOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            RegTensor<float> hi, lo;
            for (uint16_t j = 0; j < static_cast<uint16_t>(curAInnerLen); ++j) {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(hi, base + j);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(lo, base + aAligned + j);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dbOut + j, hi, pregOne);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dbOut + aAligned + j, lo,
                                                                                       pregOne);
            }
        }
        dbetaTmpOutQue_.EnQue(dbetaOutTmpTensor);
        dbetaOutTmpTensor = dbetaTmpOutQue_.template DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpGm_[gmOffset], dbetaOutTmpTensor, copyOutParams);
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpLoGm_[gmOffset], dbetaOutTmpTensor[aAligned], copyOutParams);
        dbetaTmpOutQue_.FreeTensor(dbetaOutTmpTensor);

        LocalTensor<float> dgammaOutTmpTensor = dgammaTmpOutQue_.template AllocTensor<float>();
        __local_mem__ float* dgOut = (__local_mem__ float*)dgammaOutTmpTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MaskReg pregOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            RegTensor<float> hi, lo;
            for (uint16_t j = 0; j < static_cast<uint16_t>(curAInnerLen); ++j) {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(hi, base + CONST_TWO * aAligned + j);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(lo, base + CONST_THREE * aAligned + j);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dgOut + j, hi, pregOne);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dgOut + aAligned + j, lo,
                                                                                       pregOne);
            }
        }
        dgammaTmpOutQue_.EnQue(dgammaOutTmpTensor);
        dgammaOutTmpTensor = dgammaTmpOutQue_.template DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpGm_[gmOffset], dgammaOutTmpTensor, copyOutParams);
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpLoGm_[gmOffset], dgammaOutTmpTensor[aAligned], copyOutParams);
        dgammaTmpOutQue_.FreeTensor(dgammaOutTmpTensor);
    }

    // stage 1：核间 double-fp32 合并 + 代数化收尾（dgamma = rstd*(S1 - mean*S0)，dbeta = S0）
    __aicore__ inline void CalcDbetaDgamma()
    {
        for (int64_t i = 0; i < tilingData_->aOuterStg1; i++) {
            uint16_t curAInnerLen = i != (tilingData_->aOuterStg1 - 1) ? tilingData_->aInnerStg1 :
                                                                         tilingData_->aTailStg1;
            int64_t offset = i * tilingData_->aInnerStg1;

            CopyInMeanAndRstd(offset, curAInnerLen);
            LocalTensor<float> mean = meanInQue_.template DeQue<float>();
            LocalTensor<float> rstd = rstdInQue_.template DeQue<float>();

            LocalTensor<float> dbetaTmpOut = dbetaTmpOutQue_.template AllocTensor<float>();
            LocalTensor<float> dgammaTmpOut = dgammaTmpOutQue_.template AllocTensor<float>();

            LocalTensor<float> tileDbeta = wsInQue_.template AllocTensor<float>();
            CopyInDdPair(offset, curAInnerLen, dbetaTmpGm_, dbetaTmpLoGm_, tileDbeta);
            wsInQue_.EnQue(tileDbeta);
            LocalTensor<float> tileDbetaLocal = wsInQue_.template DeQue<float>();
            FinishDbetaStage1(tileDbetaLocal, curAInnerLen, dbetaTmpOut);
            wsInQue_.FreeTensor(tileDbetaLocal);

            LocalTensor<float> tileDgamma = wsInQue_.template AllocTensor<float>();
            CopyInDdPair(offset, curAInnerLen, dgammaTmpGm_, dgammaTmpLoGm_, tileDgamma);
            wsInQue_.EnQue(tileDgamma);
            LocalTensor<float> tileDgammaLocal = wsInQue_.template DeQue<float>();
            FinishDgammaStage1(tileDgammaLocal, mean, rstd, curAInnerLen, dgammaTmpOut);
            wsInQue_.FreeTensor(tileDgammaLocal);

            meanInQue_.FreeTensor(mean);
            rstdInQue_.FreeTensor(rstd);

            LocalTensor<WEIGHT_DTYPE> dbetaOutTensor = dbetaOutQue_.template AllocTensor<WEIGHT_DTYPE>();
            LocalTensor<WEIGHT_DTYPE> dgammaOutTensor = dgammaOutQue_.template AllocTensor<WEIGHT_DTYPE>();
            __local_mem__ float* dbetaTmpOutLocal = (__local_mem__ float*)dbetaTmpOut.GetPhyAddr();
            __local_mem__ float* dgammaTmpOutLocal = (__local_mem__ float*)dgammaTmpOut.GetPhyAddr();
            __local_mem__ WEIGHT_DTYPE* dbetaOutLocal = (__local_mem__ WEIGHT_DTYPE*)dbetaOutTensor.GetPhyAddr();
            __local_mem__ WEIGHT_DTYPE* dgammaOutLocal = (__local_mem__ WEIGHT_DTYPE*)dgammaOutTensor.GetPhyAddr();
            uint16_t loopNum = ops::CeilDiv(static_cast<uint32_t>(curAInnerLen), VL_FP32);
            __VEC_SCOPE__
            {
                MaskReg pregMask;
                uint32_t sreg = curAInnerLen;
                RegTensor<float> regDbeta, regDgamma;
                for (uint16_t j = 0; j < loopNum; j++) {
                    pregMask = UpdateMask<float>(sreg);
                    uint32_t lane = j * VL_FP32;
                    LoadOneTensor<float>(dbetaTmpOutLocal, regDbeta, pregMask, lane);
                    LoadOneTensor<float>(dgammaTmpOutLocal, regDgamma, pregMask, lane);
                    StoreOneTensor<WEIGHT_DTYPE>(dbetaOutLocal, regDbeta, pregMask, lane);
                    StoreOneTensor<WEIGHT_DTYPE>(dgammaOutLocal, regDgamma, pregMask, lane);
                }
            }

            dbetaTmpOutQue_.EnQue(dbetaTmpOut);
            dgammaTmpOutQue_.EnQue(dgammaTmpOut);
            dbetaOutQue_.EnQue(dbetaOutTensor);
            dgammaOutQue_.EnQue(dgammaOutTensor);
            CopyOutDbetaDgammaAndTmp(offset, curAInnerLen);
        }
    }

    __aicore__ inline void CopyInDdPair(int64_t offset, int64_t curALen, const GlobalTensor<float>& hiGm,
                                        const GlobalTensor<float>& loGm, const LocalTensor<float>& tile)
    {
        DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = tilingData_->usedCoreNums;
        copyInParams.blockLen = curALen * sizeof(float);
        copyInParams.srcStride = (tilingData_->aDimAligned - curALen) * sizeof(float);
        copyInParams.dstStride = (tilingData_->aInnerStg1 - curALen) * sizeof(float) / BLOCK_SIZE;
        DataCopyPad<float, PaddingMode::Normal>(tile, hiGm[offset], copyInParams, padParams);
        DataCopyPad<float, PaddingMode::Normal>(tile[tilingData_->usedCoreNums * tilingData_->aInnerStg1], loGm[offset],
                                                copyInParams, padParams);
    }

    __aicore__ inline void FinishDbetaStage1(const LocalTensor<float>& tile, const uint16_t curALen,
                                             const LocalTensor<float>& dbetaOut)
    {
        int64_t coreNum = tilingData_->usedCoreNums;
        int64_t rowStride = tilingData_->aInnerStg1;
        __local_mem__ float* hiAddr = (__local_mem__ float*)tile.GetPhyAddr();
        __local_mem__ float* loAddr = hiAddr + coreNum * rowStride;
        __local_mem__ float* outAddr = (__local_mem__ float*)dbetaOut.GetPhyAddr();
        __local_mem__ float* accAddr = (__local_mem__ float*)accBuf_.Get<float>().GetPhyAddr();
        uint16_t outerLoopTimes = ops::CeilDiv(static_cast<uint32_t>(curALen), VL_FP32);
        __VEC_SCOPE__
        {
            MaskReg pMask;
            uint32_t count = curALen;
            RegTensor<float> accHi, accLo, hiReg, loReg, res;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                uint32_t lane = i * VL_FP32;
                MicroAPI::Duplicate(accHi, 0.0f, pMask);
                MicroAPI::Duplicate(accLo, 0.0f, pMask);
                for (uint16_t c = 0; c < static_cast<uint16_t>(coreNum); ++c) {
                    uint32_t rowOffset = c * rowStride + lane;
                    MicroAPI::DataCopy(hiReg, hiAddr + rowOffset);
                    MicroAPI::DataCopy(loReg, loAddr + rowOffset);
                    Comp::CompAddPair<DY_DTYPE>(accHi, accLo, hiReg, loReg, pMask);
                }
                Comp::CompRound(res, accHi, accLo, pMask);
                StoreOneTensor<float>(outAddr, res, pMask, lane);
                MicroAPI::DataCopy(accAddr + lane, accHi, pMask);
                MicroAPI::DataCopy(accAddr + tilingData_->aInnerStg1 + lane, accLo, pMask);
            }
        }
    }

    __aicore__ inline void FinishDgammaStage1(const LocalTensor<float>& tile, const LocalTensor<float>& meanTensor,
                                              const LocalTensor<float>& rstdTensor, const uint16_t curALen,
                                              const LocalTensor<float>& dgammaOut)
    {
        int64_t coreNum = tilingData_->usedCoreNums;
        int64_t rowStride = tilingData_->aInnerStg1;
        __local_mem__ float* hiAddr = (__local_mem__ float*)tile.GetPhyAddr();
        __local_mem__ float* loAddr = hiAddr + coreNum * rowStride;
        __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor.GetPhyAddr();
        __local_mem__ float* outAddr = (__local_mem__ float*)dgammaOut.GetPhyAddr();
        __local_mem__ float* accAddr = (__local_mem__ float*)accBuf_.Get<float>().GetPhyAddr();
        uint16_t outerLoopTimes = ops::CeilDiv(static_cast<uint32_t>(curALen), VL_FP32);
        __VEC_SCOPE__
        {
            MaskReg pMask;
            uint32_t count = curALen;
            RegTensor<float> accHi, accLo, hiReg, loReg, s0Hi, s0Lo, meanReg, rstdReg, dgFinal, dbFinal;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                uint32_t lane = i * VL_FP32;
                MicroAPI::Duplicate(accHi, 0.0f, pMask);
                MicroAPI::Duplicate(accLo, 0.0f, pMask);
                for (uint16_t c = 0; c < static_cast<uint16_t>(coreNum); ++c) {
                    uint32_t rowOffset = c * rowStride + lane;
                    MicroAPI::DataCopy(hiReg, hiAddr + rowOffset);
                    MicroAPI::DataCopy(loReg, loAddr + rowOffset);
                    Comp::CompAddPair<DY_DTYPE>(accHi, accLo, hiReg, loReg, pMask);
                }
                MicroAPI::DataCopy(s0Hi, accAddr + lane);
                MicroAPI::DataCopy(s0Lo, accAddr + tilingData_->aInnerStg1 + lane);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, lane);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, lane);
                Comp::CompFinalizeDgammaDbeta(accHi, accLo, s0Hi, s0Lo, meanReg, rstdReg, pMask, dgFinal, dbFinal);
                StoreOneTensor<float>(outAddr, dgFinal, pMask, lane);
            }
        }
    }

    __aicore__ inline void CopyOutDbetaDgammaAndTmp(int64_t offset, uint32_t curAInnerLen)
    {
        // 小shape输出
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = curAInnerLen * sizeof(WEIGHT_DTYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        LocalTensor<WEIGHT_DTYPE> dbeta = dbetaOutQue_.DeQue<WEIGHT_DTYPE>();
        DataCopyPad<WEIGHT_DTYPE, PaddingMode::Normal>(dbetaGm_[offset], dbeta, copyOutParams);
        dbetaOutQue_.FreeTensor(dbeta);

        LocalTensor<WEIGHT_DTYPE> dgamma = dgammaOutQue_.DeQue<WEIGHT_DTYPE>();
        DataCopyPad<WEIGHT_DTYPE, PaddingMode::Normal>(dgammaGm_[offset], dgamma, copyOutParams);
        dgammaOutQue_.FreeTensor(dgamma);

        // 拷贝到workspace 临时空间
        DataCopyExtParams copyOutParams2;
        copyOutParams2.blockCount = 1;
        copyOutParams2.blockLen = curAInnerLen * sizeof(float);
        copyOutParams2.srcStride = 0;
        copyOutParams2.dstStride = 0;

        LocalTensor<float> dbetaTmp = dbetaTmpOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpGm_[offset], dbetaTmp, copyOutParams2);
        dbetaTmpOutQue_.FreeTensor(dbetaTmp);

        LocalTensor<float> dgammaTmp = dgammaTmpOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpGm_[offset], dgammaTmp, copyOutParams2);
        dgammaTmpOutQue_.FreeTensor(dgammaTmp);
    }

    // stage 2
    __aicore__ inline void CalcOutput()
    {
        for (int64_t i = 0; i < tilingData_->aOuterStg2; i++) {
            uint16_t curAInnerLen = i != (tilingData_->aOuterStg2 - 1) ? tilingData_->aInnerStg2 :
                                                                         tilingData_->aTailStg2;
            int64_t aOffset = i * tilingData_->aInnerStg2;
            CopyInSmallShapes(aOffset, curAInnerLen);

            LocalTensor<float> mean = meanInQue_.template DeQue<float>();
            LocalTensor<float> rstd = rstdInQue_.template DeQue<float>();
            LocalTensor<WEIGHT_DTYPE> gamma = gammaInQue_.template DeQue<WEIGHT_DTYPE>();
            LocalTensor<float> dbetaTmp = dbetaTmpInQue_.template DeQue<float>();
            LocalTensor<float> dgammaTmp = dgammaTmpInQue_.template DeQue<float>();

            __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd.GetPhyAddr();
            __local_mem__ WEIGHT_DTYPE* gammaAddr = (__local_mem__ WEIGHT_DTYPE*)gamma.GetPhyAddr();
            __local_mem__ float* dbetaTmpAddr = (__local_mem__ float*)dbetaTmp.GetPhyAddr();
            __local_mem__ float* dgammaTmpAddr = (__local_mem__ float*)dgammaTmp.GetPhyAddr();

            int64_t r0OuterInCore = tilingData_->r0InnerOuterStg2;
            int64_t r0InerIncore = tilingData_->r0InnerInnerStg2;
            int64_t r0TailIncore = tilingData_->r0InnerTailStg2;
            if (isLastCore_) {
                r0OuterInCore = tilingData_->r0TailOuterStg2;
                r0TailIncore = tilingData_->r0TailTailStg2;
            }

            for (int64_t j = 0; j < tilingData_->r1OuterStg2; j++) {
                uint16_t curR1Len = j != (tilingData_->r1OuterStg2 - 1) ? tilingData_->r1InnerStg2 :
                                                                          tilingData_->r1TailStg2;
                for (int64_t k = 0; k < r0OuterInCore; k++) {
                    int64_t dyOffset = j * tilingData_->aDim * tilingData_->r0Dim * tilingData_->r1InnerStg2 +
                                       i * tilingData_->aInnerStg2 * tilingData_->r0Dim +
                                       k * tilingData_->r0InnerInnerStg2 + r0BlockOffset_;
                    uint32_t curR0Len = k != (r0OuterInCore - 1) ? r0InerIncore : r0TailIncore;
                    uint32_t curR0AlignedLen = curR0Len;
                    if (k == (r0OuterInCore - 1) && isLastCore_) {
                        curR0AlignedLen = tilingData_->r0TailTailAlignedStg2;
                    }

                    CopyInDyAndX(curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);

                    LocalTensor<DY_DTYPE> dyTensor = dyInQue_.DeQue<DY_DTYPE>();
                    __local_mem__ DY_DTYPE* dyLocal = (__local_mem__ DY_DTYPE*)dyTensor.GetPhyAddr();

                    LocalTensor<DY_DTYPE> xTensor = xInQue_.DeQue<DY_DTYPE>();
                    __local_mem__ DY_DTYPE* xLocal = (__local_mem__ DY_DTYPE*)xTensor.GetPhyAddr();

                    LocalTensor<DY_DTYPE> dxTensor = dxOutQue_.AllocTensor<DY_DTYPE>();
                    __local_mem__ DY_DTYPE* dxLocal = (__local_mem__ DY_DTYPE*)dxTensor.GetPhyAddr();

                    uint16_t loopNum = ops::CeilDiv(curR0Len, VL_FP32);

                    __VEC_SCOPE__
                    {
                        MaskReg
                            pregMaskAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

                        MaskReg pregMask;

                        RegTensor<float> regMean, regRstd, regGamma, regDbetaTmp, regDgammaTmp, regDy, regX;
                        RegTensor<float> regDbOverR, regDbOverR1, regDgC0, regDgC1, regZero;

                        for (uint16_t i = 0; i < curAInnerLen; i++) {
                            LoadsTensorForDtypeT<float>(meanAddr, regMean, pregMaskAll, i);
                            LoadsTensorForDtypeT<float>(rstdAddr, regRstd, pregMaskAll, i);
                            LoadsTensorForDtypeT<float>(dbetaTmpAddr, regDbetaTmp, pregMaskAll, i);
                            LoadsTensorForDtypeT<float>(dgammaTmpAddr, regDgammaTmp, pregMaskAll, i);
                            LoadsTensorForDtypeT<WEIGHT_DTYPE>(gammaAddr, regGamma, pregMaskAll, i);
                            // 每通道一次（A==1，摊到整条 R 上可忽略）。两个除法都按**精确整数 R**
                            // 求商（comp_sum.h：fl32(R) != R 时除法/倒数乘法带整通道同号偏差，
                            // 该偏差逐元素传出、不进 RMSE 平均，是 L1_065 偶发超线的根因）：
                            //   * dofset/R 只用第一字 c0（元素内仍是 1 条 Sub）；
                            //   * dscale/R 用 c0 + c1 两字（元素内 2 条 FMA，替换原 Mul+Sub）。
                            // 非有限 lane 在 helper 内部回退到原倒数乘法，c1 置 0。
                            Comp::CompExactDivisorQuotient(regDbOverR, regDbOverR1, regDbetaTmp, hRecipVal_,
                                                           static_cast<float>(hRecipDen_), hNegDelta_, pregMaskAll);
                            Comp::CompExactDivisorQuotient(regDgC0, regDgC1, regDgammaTmp, hRecipVal_,
                                                           static_cast<float>(hRecipDen_), hNegDelta_, pregMaskAll);
                            // MulAddDst 是 dst += a*b，而 t 里 dscale/R 这一项要**减**，取负存一次
                            Duplicate(regZero, 0.0f, pregMaskAll);
                            Sub(regDgC0, regZero, regDgC0, pregMaskAll);
                            Sub(regDgC1, regZero, regDgC1, pregMaskAll);

                            uint32_t sreg = curR0Len;
                            for (uint16_t j = 0; j < loopNum; j++) { // r0
                                pregMask = UpdateMask<float>(sreg);
                                for (uint16_t k = 0; k < curR1Len; k++) {
                                    uint32_t offset = i * curR1Len * curR0AlignedLen + k * curR0AlignedLen +
                                                      j * VL_FP32;
                                    LoadOneTensor<DY_DTYPE>(dyLocal, regDy, pregMask, offset);
                                    LoadOneTensor<DY_DTYPE>(xLocal, regX, pregMask, offset);

                                    Sub(regX, regX, regMean, pregMask);      // xhat = x - mean
                                    Mul(regX, regX, regRstd, pregMask);      // xhat * rstd
                                    Sub(regDy, regDy, regDbOverR, pregMask); // dy - dofset/R
                                    // t = (dy - dofset/R) - xhat*(dscale/R)。dscale/R 用两字常量
                                    // （regDgC0/regDgC1 已取负），两条 MulAddDst 都是单次舍入；
                                    // 连同上面的 Sub 仍是 3 条向量指令，与改动前的 Mul+Sub+Sub 同数。
                                    MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(regDy, regX, regDgC0, pregMask);
                                    MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(regDy, regX, regDgC1, pregMask);
                                    Mul(regX, regDy, regRstd, pregMask); // t * rstd
                                    Mul(regX, regX, regGamma, pregMask); // * scale

                                    StoreOneTensor<DY_DTYPE>(dxLocal, regX, pregMask, offset);
                                }
                            }
                        }
                    }
                    dyInQue_.FreeTensor(dyTensor);
                    xInQue_.FreeTensor(xTensor);

                    dxOutQue_.EnQue(dxTensor);
                    CopyOutDx(curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);
                }
            }

            dbetaTmpInQue_.FreeTensor(dbetaTmp);
            dgammaTmpInQue_.FreeTensor(dgammaTmp);
            meanInQue_.FreeTensor(mean);
            rstdInQue_.FreeTensor(rstd);
            gammaInQue_.FreeTensor(gamma);
        }
    }

    __aicore__ inline void CopyInDyAndX(uint32_t curAInnerLen, uint32_t curR1Len, uint32_t curR0Len,
                                        uint32_t curR0AlignedLen, int64_t gmOffset)
    {
        // RAR -> AR
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curAInnerLen;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->r0Dim * sizeof(DY_DTYPE);
        loopParams.loop1DstStride = curR0AlignedLen * curR1Len * sizeof(DY_DTYPE);

        DataCopyPadExtParams<DY_DTYPE> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = true;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0; // 默认补齐到block对齐，实际大小（curR0AlignedLen - curR0Len）;
        dataCopyPadExtParams.paddingValue = 0;

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curR1Len;
        copyInParams.blockLen = curR0Len * sizeof(DY_DTYPE);
        copyInParams.srcStride = (tilingData_->r0Dim * tilingData_->aDim - curR0Len) * sizeof(DY_DTYPE);
        copyInParams.dstStride = (curR0AlignedLen - curR0Len) * sizeof(DY_DTYPE) / BLOCK_SIZE;

        LocalTensor<DY_DTYPE> x = xInQue_.template AllocTensor<DY_DTYPE>();
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<DY_DTYPE, PaddingMode::Normal>(x, xGm_[gmOffset], copyInParams, dataCopyPadExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        xInQue_.EnQue(x);

        LocalTensor<DY_DTYPE> dy = dyInQue_.template AllocTensor<DY_DTYPE>();
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<DY_DTYPE, PaddingMode::Normal>(dy, dyGm_[gmOffset], copyInParams, dataCopyPadExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        dyInQue_.EnQue(dy);
    }

    __aicore__ inline void CopyOutDx(uint32_t curAInnerLen, uint32_t curR1Len, uint32_t curR0Len,
                                     uint32_t curR0AlignedLen, int64_t offset)
    {
        // AR-> RAR
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curAInnerLen;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = curR0AlignedLen * curR1Len * sizeof(DY_DTYPE);
        loopParams.loop1DstStride = tilingData_->r0Dim * sizeof(DY_DTYPE);

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = curR1Len;
        copyOutParams.blockLen = curR0Len * sizeof(DY_DTYPE);
        copyOutParams.srcStride = (curR0AlignedLen - curR0Len) * sizeof(DY_DTYPE) / BLOCK_SIZE;
        copyOutParams.dstStride = (tilingData_->r0Dim * tilingData_->aDim - curR0Len) * sizeof(DY_DTYPE);

        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

        LocalTensor<DY_DTYPE> dxTensor = dxOutQue_.DeQue<DY_DTYPE>();
        DataCopyPad<DY_DTYPE, PaddingMode::Normal>(dxGm_[offset], dxTensor, copyOutParams);
        dxOutQue_.FreeTensor(dxTensor);

        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }

    __aicore__ inline void CopyInMeanAndRstd(int64_t offset, uint32_t dataLen)
    {
        DataCopyPadExtParams<float> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = dataLen * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;

        LocalTensor<float> mean = meanInQue_.template AllocTensor<float>();
        DataCopyPad(mean, meanGm_[offset], copyInParams, dataCopyPadExtParams);
        meanInQue_.EnQue(mean);

        LocalTensor<float> rstd = rstdInQue_.template AllocTensor<float>();
        DataCopyPad(rstd, rstdGm_[offset], copyInParams, dataCopyPadExtParams);
        rstdInQue_.EnQue(rstd);
    }

    __aicore__ inline void CopyInSmallShapes(int64_t offset, uint32_t dataLen)
    {
        DataCopyPadExtParams<float> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = dataLen * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;

        LocalTensor<float> mean = meanInQue_.template AllocTensor<float>();
        DataCopyPad(mean, meanGm_[offset], copyInParams, dataCopyPadExtParams);
        meanInQue_.EnQue(mean);

        LocalTensor<float> rstd = rstdInQue_.template AllocTensor<float>();
        DataCopyPad(rstd, rstdGm_[offset], copyInParams, dataCopyPadExtParams);
        rstdInQue_.EnQue(rstd);

        LocalTensor<float> dbetaTmp = dbetaTmpInQue_.template AllocTensor<float>();
        DataCopyPad(dbetaTmp, dbetaTmpGm_[offset], copyInParams, dataCopyPadExtParams);
        dbetaTmpInQue_.EnQue(dbetaTmp);

        LocalTensor<float> dgammaTmp = dgammaTmpInQue_.template AllocTensor<float>();
        DataCopyPad(dgammaTmp, dgammaTmpGm_[offset], copyInParams, dataCopyPadExtParams);
        dgammaTmpInQue_.EnQue(dgammaTmp);

        DataCopyPadExtParams<WEIGHT_DTYPE> dataCopyPadExtParams1;
        dataCopyPadExtParams1.isPad = false;
        dataCopyPadExtParams1.leftPadding = 0;
        dataCopyPadExtParams1.rightPadding = 0;
        dataCopyPadExtParams1.paddingValue = 0;

        DataCopyExtParams copyInParams1;
        copyInParams1.blockCount = 1;
        copyInParams1.blockLen = dataLen * sizeof(WEIGHT_DTYPE);
        copyInParams1.srcStride = 0;
        copyInParams1.dstStride = 0;

        LocalTensor<WEIGHT_DTYPE> gamma = gammaInQue_.template AllocTensor<WEIGHT_DTYPE>();
        DataCopyPad(gamma, gammaGm_[offset], copyInParams1, dataCopyPadExtParams1);
        gammaInQue_.EnQue(gamma);
    }

private:
    const BatchNormGradExt2RARSplitCoreR0TilingData* tilingData_{nullptr};
    TPipe* pipe_{nullptr};

    GlobalTensor<DY_DTYPE> dyGm_;
    GlobalTensor<DY_DTYPE> xGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<float> dbetaTmpGm_;
    GlobalTensor<float> dgammaTmpGm_;
    GlobalTensor<float> dbetaTmpLoGm_, dgammaTmpLoGm_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> rstdInQue_;
    TBuf<TPosition::VECCALC> compAccBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dbetaTmpOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dgammaTmpOutQue_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dbetaTmpInQue_;
    TQue<QuePosition::VECIN, 1> wsInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> dgammaTmpInQue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dgammaOutQue_;

    GlobalTensor<WEIGHT_DTYPE> gammaGm_;
    GlobalTensor<WEIGHT_DTYPE> dgammaGm_;
    GlobalTensor<WEIGHT_DTYPE> dbetaGm_;
    GlobalTensor<DY_DTYPE> dxGm_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> gammaInQue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dxOutQue_;

    int64_t blkIdx_{0};
    float hRecipVal_{0};
    float hNegDelta_{0}; // -(R - fl32(R))，把「除以 fl32(R)」还原成「除以精确 R」
    int64_t hRecipDen_{0};
    bool isLastCore_{false};
    int64_t r0BlockOffset_{0};
};
} // namespace BatchNormGradExt2
#endif // __BATCH_NORM_GRAD_EXT2_SPLIT_CORE_R0_H__
