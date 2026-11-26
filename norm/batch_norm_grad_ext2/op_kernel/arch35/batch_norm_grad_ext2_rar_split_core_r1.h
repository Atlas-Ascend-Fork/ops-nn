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
 * \file batch_norm_grad_ext2_rar_split_core_r1.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_EXT2_SPLIT_CORE_R1_H__
#define __BATCH_NORM_GRAD_EXT2_SPLIT_CORE_R1_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm_grad_ext2_common.h"
#include "batch_norm_grad_ext2_comp_sum.h"

namespace BatchNormGradExt2 {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using namespace BatchNormGradExt2::Comp;

static constexpr int64_t CONST_ONE = 1;
static constexpr int64_t CONST_TWO = 2;
static constexpr int64_t ULONG_BIT_LEN = 64;
static constexpr int32_t BUFFER_NUM = 2;
static constexpr int32_t BUFFER_DEPTH = 1;
static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
static constexpr int32_t BLOCK_FP32_NUMS = BLOCK_SIZE / sizeof(float);

template <typename DY_DTYPE, typename WEIGHT_DTYPE>
class BatchNormGradExt2RARSplitCoreR1 {
public:
    __aicore__ inline BatchNormGradExt2RARSplitCoreR1(TPipe* pipeIn,
                                                      const BatchNormGradExt2RARSplitCoreR1TilingData* tilingDataIn)
    {
        pipe_ = pipeIn;
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR mean, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx,
                                GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace)
    {
        blkIdx_ = GetBlockIdx();
        isLastCore_ = blkIdx_ == (tilingData_->usedCoreNums - 1) ? true : false;

        meanGm_.SetGlobalBuffer((__gm__ float*)(mean));
        rstdGm_.SetGlobalBuffer((__gm__ float*)(rstd));
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)(gamma));
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)(dgamma));
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)(dbeta));

        // 4 个 workspace 区：dbeta hi/lo、dgamma hi/lo（stage0 落 dd 对；stage1 覆盖 hi 区为最终 fp32）
        int64_t wsStride = tilingData_->usedCoreNums * tilingData_->aDimAligned;
        dbetaTmpGm_.SetGlobalBuffer((__gm__ float*)(workspace));
        dbetaTmpLoGm_.SetGlobalBuffer((__gm__ float*)(workspace) + wsStride);
        dgammaTmpGm_.SetGlobalBuffer((__gm__ float*)(workspace) + wsStride * CONST_TWO);
        dgammaTmpLoGm_.SetGlobalBuffer((__gm__ float*)(workspace) + wsStride * 3);

        // stg0、stg2相同
        int64_t dyOffset = blkIdx_ * tilingData_->r1Inner * tilingData_->r0Dim * tilingData_->aDim;
        dyGm_.SetGlobalBuffer((__gm__ DY_DTYPE*)(dy) + dyOffset);
        xGm_.SetGlobalBuffer((__gm__ DY_DTYPE*)(x) + dyOffset);
        dxGm_.SetGlobalBuffer((__gm__ DY_DTYPE*)(dx) + dyOffset);

        hRecipVal_ = 1.0f / (float)(tilingData_->r1Dim * tilingData_->r0Dim);
        // 精确整数 R 与 fl32(R) 之差：供 Comp::CompExactDivisorReciprocalLo 把 rhi+rlo 从
        // 「1/fl32(R)」修正成「1/R」（R <= 2^24 或 R 可精确表示时 Δ = 0，结果逐位不变）。
        hNegDelta_ = Comp::ExactDivisorNegDelta(tilingData_->r1Dim * tilingData_->r0Dim);
    }

    __aicore__ inline void Stage0InitBuffer()
    {
        // 二分相关buff初始化
        int64_t dyLen = tilingData_->r0InnerStg0 * tilingData_->r1InnerInnerStg0 * tilingData_->aInnerStg0;
        int64_t meanSize = tilingData_->aInnerAlignedStg0 * sizeof(float);

        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));

        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, meanSize);

        // dbeta/dgamma 的 (hi, lo) 两字随 workspace 交给 stage1；lo2 在落盘前折回 lo
        pipe_->InitBuffer(dbetaTmpOutQue_, BUFFER_DEPTH, meanSize * CONST_TWO);
        pipe_->InitBuffer(dgammaTmpOutQue_, BUFFER_DEPTH, meanSize * CONST_TWO);

        // 补偿累加器：每个通道各 6 个标量（S0/S1 的 hi/lo/lo2），见 comp_sum.h CompAddWide
        accChannelAligned_ = tilingData_->aInnerAlignedStg0;
        // 多留一个 VL 的余量：清零按整向量写，aInnerAlignedStg0 不保证是 VL 的倍数
        pipe_->InitBuffer(compAccBuf_, (6 * accChannelAligned_ + static_cast<int64_t>(VL_FP32)) * sizeof(float));
    }

    __aicore__ inline void Stage1InitBuffer()
    {
        pipe_->Reset();

        // 每个 queue 装 hi 与 lo 两片（aInnerStg1 × usedCoreNums 各一片）
        int64_t aSizeIn = tilingData_->aInnerStg1 * sizeof(float) * tilingData_->usedCoreNums * CONST_TWO;
        pipe_->InitBuffer(dbetaTmpInQue_, BUFFER_NUM, aSizeIn);
        pipe_->InitBuffer(dgammaTmpInQue_, BUFFER_NUM, aSizeIn);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, tilingData_->aInnerStg1 * sizeof(float));
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, tilingData_->aInnerStg1 * sizeof(float));

        pipe_->InitBuffer(dbetaTmpOutQue_, BUFFER_NUM, tilingData_->aInnerStg1 * sizeof(float));
        pipe_->InitBuffer(dgammaTmpOutQue_, BUFFER_NUM, tilingData_->aInnerStg1 * sizeof(float));
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, tilingData_->aInnerStg1 * sizeof(WEIGHT_DTYPE));
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, tilingData_->aInnerStg1 * sizeof(WEIGHT_DTYPE));
    }

    __aicore__ inline void Stage2InitBuffer()
    {
        pipe_->Reset();

        int64_t dyLen = tilingData_->r0InnerStg2 * tilingData_->r1InnerInnerStg2 * tilingData_->aInnerStg2;
        int64_t meanSize = tilingData_->aInnerAlignedStg2 * sizeof(float);

        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));
        pipe_->InitBuffer(dxOutQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));

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

            CalcReduceSum(curAInnerLen, aOffset, meanLocal, rstdLocal);

            meanInQue_.FreeTensor(meanTensor);
            rstdInQue_.FreeTensor(rstdTensor);

            CopyOutDbetaDgammaTmp(curAInnerLen, aOffset);
        }
    }

    __aicore__ inline __local_mem__ float* CompBase()
    {
        return (__local_mem__ float*)compAccBuf_.Get<float>().GetPhyAddr();
    }

    __aicore__ inline void InitCompAcc()
    {
        __local_mem__ float* base = CompBase();
        __VEC_SCOPE__
        {
            RegTensor<float> zero;
            MaskReg all = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(zero, 0.0f, all);
            // 必须按 ceil 取整向量数：aInnerAlignedStg0 可能小于 VL（8 通道时 6*8/64 会截断成 0，
            // 累加器一个字节都不清零——历史踩过的坑）。
            int64_t totalVec = (6 * accChannelAligned_ + static_cast<int64_t>(VL_FP32) - 1) /
                               static_cast<int64_t>(VL_FP32);
            for (uint16_t i = 0; i < static_cast<uint16_t>(totalVec); ++i) {
                MicroAPI::DataCopy(base + i * VL_FP32, zero, all);
            }
        }
    }

    __aicore__ inline void CalcReduceSum(uint32_t curAInnerLen, int64_t aOffset, __local_mem__ float* meanLocal,
                                         __local_mem__ float* rstdLocal)
    {
        dbetaMainTmpTensor_ = dbetaTmpOutQue_.template AllocTensor<float>();
        dgammaMainTmpTensor_ = dgammaTmpOutQue_.template AllocTensor<float>();
        InitCompAcc();

        int64_t r1OuterInCore = tilingData_->r1InnerOuterStg0;
        int64_t r1InerIncore = tilingData_->r1InnerInnerStg0;
        int64_t r1TailIncore = tilingData_->r1InnerTailStg0;
        int64_t basicLoop = tilingData_->binAddBasicBlockLoop;
        int64_t mainLoop = tilingData_->binAddMainFoldCount;

        if (isLastCore_) {
            r1OuterInCore = tilingData_->r1TailOuterStg0;
            r1TailIncore = tilingData_->r1TailTailStg0;
            basicLoop = tilingData_->lastCoreBinAddBasicBlockLoop;
            mainLoop = tilingData_->lastCoreBinAddMainFoldCount;
        }

        for (int64_t basicBlockIdx = 0; basicBlockIdx < basicLoop; basicBlockIdx++) {
            int64_t r1OuterIdx = basicBlockIdx / tilingData_->r0OuterStg0;
            int64_t r0OuterIdx = basicBlockIdx % tilingData_->r0OuterStg0;

            uint32_t curR1Len = r1OuterIdx != (r1OuterInCore - 1) ? r1InerIncore : r1TailIncore;

            int64_t dyOffset = r1OuterIdx * tilingData_->aDim * tilingData_->r0Dim * r1InerIncore +
                               aOffset * tilingData_->r0Dim + r0OuterIdx * tilingData_->r0InnerStg0;

            uint32_t curR0Len = tilingData_->r0InnerStg0;
            uint32_t curR0AlignedLen = tilingData_->r0InnerStg0;
            if (r0OuterIdx == (tilingData_->r0OuterStg0 - 1)) {
                curR0Len = tilingData_->r0TailStg0;
                curR0AlignedLen = tilingData_->r0TailAlignedStg0;
            }

            ProcessMainBlock(curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);

            if (basicBlockIdx < mainLoop) {
                int64_t r1OuterIdxFold = (basicBlockIdx + basicLoop) / tilingData_->r0OuterStg0;
                int64_t r0OuterIdxFold = (basicBlockIdx + basicLoop) % tilingData_->r0OuterStg0;

                uint32_t curR1LenFold = r1OuterIdxFold != (r1OuterInCore - 1) ? r1InerIncore : r1TailIncore;

                int64_t dyOffsetFold = r1OuterIdxFold * tilingData_->aDim * tilingData_->r0Dim * r1InerIncore +
                                       aOffset * tilingData_->r0Dim + r0OuterIdxFold * tilingData_->r0InnerStg0;

                uint32_t curR0LenFold = tilingData_->r0InnerStg0;
                uint32_t curR0AlignedLenFold = tilingData_->r0InnerStg0;
                if (r0OuterIdxFold == (tilingData_->r0OuterStg0 - 1)) {
                    curR0LenFold = tilingData_->r0TailStg0;
                    curR0AlignedLenFold = tilingData_->r0TailAlignedStg0;
                }

                ProcessFoldBlock(curAInnerLen, curR1LenFold, curR0LenFold, curR0AlignedLenFold, dyOffsetFold);
            }
        }

        if (basicLoop == 0) { // 直接累加到补偿器
            int64_t dyOffset = aOffset * tilingData_->r0Dim;
            ProcessMainBlock(curAInnerLen, r1TailIncore, tilingData_->r0TailStg0, tilingData_->r0TailAlignedStg0,
                             dyOffset);
        }

        FinalizeStage0(curAInnerLen);

        dbetaTmpOutQue_.EnQue(dbetaMainTmpTensor_);
        dgammaTmpOutQue_.EnQue(dgammaMainTmpTensor_);
    }

    __aicore__ inline void ProcessMainBlock(const uint32_t curAInnerLen, const uint32_t curR1Len,
                                            const uint32_t curR0Len, const uint32_t curR0AlignedLen,
                                            const int64_t dyOffset)
    {
        CopyInDyAndX(curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);
        LocalTensor<DY_DTYPE> dyTensor = dyInQue_.DeQue<DY_DTYPE>();
        LocalTensor<DY_DTYPE> xTensor = xInQue_.DeQue<DY_DTYPE>();
        AccumBlock(dyTensor, xTensor, curAInnerLen, curR1Len, curR0Len, curR0AlignedLen);
        dyInQue_.FreeTensor(dyTensor);
        xInQue_.FreeTensor(xTensor);
    }

    // fold 块与 main 块同构：直接累加到同一组补偿器
    __aicore__ inline void ProcessFoldBlock(const uint32_t curAInnerLen, const uint32_t curR1Len,
                                            const uint32_t curR0Len, const uint32_t curR0AlignedLen,
                                            const int64_t dyOffset)
    {
        ProcessMainBlock(curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);
    }

    // lane 内沿 R 累加（每块每个通道一次），块内 CompReduceLanes 收到 lane0 后按
    // CompAddPairWide 合并进通道的常驻三字累加器。
    __aicore__ inline void AccumBlock(const LocalTensor<DY_DTYPE>& dyTensor, const LocalTensor<DY_DTYPE>& xTensor,
                                      uint32_t curAInnerLen, uint32_t curR1Len, uint32_t curR0Len,
                                      uint32_t curR0AlignedLen)
    {
        __local_mem__ DY_DTYPE* dyLocal = (__local_mem__ DY_DTYPE*)dyTensor.GetPhyAddr();
        __local_mem__ DY_DTYPE* xLocal = (__local_mem__ DY_DTYPE*)xTensor.GetPhyAddr();
        __local_mem__ float* base = CompBase();
        uint16_t loopNum = ops::CeilDiv(curR0Len, VL_FP32);

        __VEC_SCOPE__
        {
            MaskReg pregMaskAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg one = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            RegTensor<float> laneS0Hi, laneS0Lo, laneS0Lo2, laneS1Hi, laneS1Lo, laneS1Lo2;
            RegTensor<float> accS0Hi, accS0Lo, accS0Lo2, accS1Hi, accS1Lo, accS1Lo2;
            RegTensor<float> regDy, regX, prodHi;

            for (uint16_t i = 0; i < curAInnerLen; i++) {
                uint32_t sreg = curR0Len;
                MicroAPI::Duplicate(laneS0Hi, 0.0f, pregMaskAll);
                MicroAPI::Duplicate(laneS0Lo, 0.0f, pregMaskAll);
                MicroAPI::Duplicate(laneS0Lo2, 0.0f, pregMaskAll);
                MicroAPI::Duplicate(laneS1Hi, 0.0f, pregMaskAll);
                MicroAPI::Duplicate(laneS1Lo, 0.0f, pregMaskAll);
                MicroAPI::Duplicate(laneS1Lo2, 0.0f, pregMaskAll);
                for (uint16_t j = 0; j < loopNum; j++) {
                    MaskReg pregMask = UpdateMask<float>(sreg);
                    for (uint16_t k = 0; k < curR1Len; k++) {
                        uint32_t offset = i * curR1Len * curR0AlignedLen + k * curR0AlignedLen + j * VL_FP32;
                        LoadOneTensor<DY_DTYPE>(dyLocal, regDy, pregMask, offset);
                        LoadOneTensor<DY_DTYPE>(xLocal, regX, pregMask, offset);
                        CompAddWide<DY_DTYPE>(laneS0Hi, laneS0Lo, laneS0Lo2, regDy, pregMask);
                        if constexpr (IsSameType<DY_DTYPE, float>::value) {
                            CompAddProductWide<DY_DTYPE>(laneS1Hi, laneS1Lo, laneS1Lo2, regX, regDy, pregMask);
                        } else {
                            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(prodHi, regX, regDy, pregMask);
                            CompAddWide<DY_DTYPE>(laneS1Hi, laneS1Lo, laneS1Lo2, prodHi, pregMask);
                        }
                    }
                }
                CompFoldLo2(laneS0Lo, laneS0Lo2, pregMaskAll);
                CompFoldLo2(laneS1Lo, laneS1Lo2, pregMaskAll);
                // 本块只把 curR0Len 条 lane 填过数据（每个 lane 复用 curR1Len 次），其余恒 +0，
                // 归约树从对应层起跳（见 comp_sum.h CompReduceBeginWidth）。
                const uint32_t laneCount = CompLaneBound(static_cast<int64_t>(curR0Len) *
                                                         static_cast<int64_t>(curR1Len));
                CompReduceLanes(laneS0Hi, laneS0Lo, pregMaskAll, laneCount);
                CompReduceLanes(laneS1Hi, laneS1Lo, pregMaskAll, laneCount);
                // 合并到通道常驻三字累加器（lane0）
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accS0Hi, base + i);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accS0Lo, base + accChannelAligned_ + i);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accS0Lo2,
                                                                            base + CONST_TWO * accChannelAligned_ + i);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accS1Hi, base + 3 * accChannelAligned_ + i);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accS1Lo, base + 4 * accChannelAligned_ + i);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accS1Lo2,
                                                                            base + 5 * accChannelAligned_ + i);
                CompAddPairWide<DY_DTYPE>(accS0Hi, accS0Lo, accS0Lo2, laneS0Hi, laneS0Lo, one);
                CompAddPairWide<DY_DTYPE>(accS1Hi, accS1Lo, accS1Lo2, laneS1Hi, laneS1Lo, one);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(base + i, accS0Hi, one);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(base + accChannelAligned_ + i,
                                                                                       accS0Lo, one);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    base + CONST_TWO * accChannelAligned_ + i, accS0Lo2, one);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    base + 3 * accChannelAligned_ + i, accS1Hi, one);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    base + 4 * accChannelAligned_ + i, accS1Lo, one);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    base + 5 * accChannelAligned_ + i, accS1Lo2, one);
            }
        }
    }

    // 落盘前把 lo2 折回 lo，再把 (hi, lo) 两字写进输出 buffer（stage1 跨核合并使用）
    __aicore__ inline void FinalizeStage0(uint32_t curAInnerLen)
    {
        __local_mem__ float* base = CompBase();
        __local_mem__ float* dbetaOut = (__local_mem__ float*)dbetaMainTmpTensor_.GetPhyAddr();
        __local_mem__ float* dgammaOut = (__local_mem__ float*)dgammaMainTmpTensor_.GetPhyAddr();
        __VEC_SCOPE__
        {
            MaskReg all = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg one = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            RegTensor<float> accHi, accLo, accLo2;
            for (uint16_t i = 0; i < curAInnerLen; i++) {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accHi, base + i);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accLo, base + accChannelAligned_ + i);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accLo2,
                                                                            base + CONST_TWO * accChannelAligned_ + i);
                CompFoldLo2(accLo, accLo2, one);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dbetaOut + i, accHi, one);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dbetaOut + accChannelAligned_ + i, accLo, one);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accHi, base + 3 * accChannelAligned_ + i);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accLo, base + 4 * accChannelAligned_ + i);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accLo2, base + 5 * accChannelAligned_ + i);
                CompFoldLo2(accLo, accLo2, one);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dgammaOut + i, accHi, one);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dgammaOut + accChannelAligned_ + i, accLo, one);
            }
        }
    }

    __aicore__ inline void CopyOutDbetaDgammaTmp(const uint32_t curAInnerLen, const int64_t aOffset)
    {
        int64_t gmOffset = blkIdx_ * tilingData_->aDimAligned + aOffset;
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = curAInnerLen * sizeof(float);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        LocalTensor<float> dbetaOutTmpTensor = dbetaTmpOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpGm_[gmOffset], dbetaOutTmpTensor, copyOutParams);
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpLoGm_[gmOffset],
                                                dbetaOutTmpTensor[tilingData_->aInnerAlignedStg0], copyOutParams);
        dbetaTmpOutQue_.FreeTensor(dbetaOutTmpTensor);

        LocalTensor<float> dgammaOutTmpTensor = dgammaTmpOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpGm_[gmOffset], dgammaOutTmpTensor, copyOutParams);
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpLoGm_[gmOffset],
                                                dgammaOutTmpTensor[tilingData_->aInnerAlignedStg0], copyOutParams);
        dgammaTmpOutQue_.FreeTensor(dgammaOutTmpTensor);
    }

    __aicore__ inline int64_t GetCacheID(const int64_t idx)
    {
        return ScalarGetCountOfValue<1>(idx ^ (idx + CONST_ONE)) - CONST_ONE;
    }

    // stage 1: 跨核 dd 合并 + 代数化收尾（dbeta = S0，dgamma = rstd * (S1 - mean * S0)）
    __aicore__ inline void CalcDbetaDgamma()
    {
        for (int64_t i = 0; i < tilingData_->aOuterStg1; i++) {
            uint16_t curAInnerLen = i != (tilingData_->aOuterStg1 - 1) ? tilingData_->aInnerStg1 :
                                                                         tilingData_->aTailStg1;

            int64_t offset = i * tilingData_->aInnerStg1;
            CopyInDbetaDgammaTmp(offset, curAInnerLen);
            CopyInMeanAndRstd(offset, curAInnerLen);

            LocalTensor<float> dbetaTmpIn = dbetaTmpInQue_.template DeQue<float>();
            LocalTensor<float> dgammaTmpIn = dgammaTmpInQue_.template DeQue<float>();
            LocalTensor<float> mean = meanInQue_.template DeQue<float>();
            LocalTensor<float> rstd = rstdInQue_.template DeQue<float>();
            LocalTensor<float> dbetaTmpOutTensor = dbetaTmpOutQue_.AllocTensor<float>();
            LocalTensor<float> dgammaTmpOutTensor = dgammaTmpOutQue_.AllocTensor<float>();

            FinalizeStage1(curAInnerLen, dbetaTmpIn, dgammaTmpIn, mean, rstd, dbetaTmpOutTensor, dgammaTmpOutTensor);

            dbetaTmpInQue_.FreeTensor(dbetaTmpIn);
            dgammaTmpInQue_.FreeTensor(dgammaTmpIn);
            meanInQue_.FreeTensor(mean);
            rstdInQue_.FreeTensor(rstd);

            // 输出 dtype 转换
            LocalTensor<WEIGHT_DTYPE> dbetaOutTensor = dbetaOutQue_.AllocTensor<WEIGHT_DTYPE>();
            LocalTensor<WEIGHT_DTYPE> dgammaOutTensor = dgammaOutQue_.AllocTensor<WEIGHT_DTYPE>();

            __local_mem__ float* dbetaTmpOutLocal = (__local_mem__ float*)dbetaTmpOutTensor.GetPhyAddr();
            __local_mem__ float* dgammaTmpOutLocal = (__local_mem__ float*)dgammaTmpOutTensor.GetPhyAddr();
            __local_mem__ WEIGHT_DTYPE* dbetaOutLocal = (__local_mem__ WEIGHT_DTYPE*)dbetaOutTensor.GetPhyAddr();
            __local_mem__ WEIGHT_DTYPE* dgammaOutLocal = (__local_mem__ WEIGHT_DTYPE*)dgammaOutTensor.GetPhyAddr();

            uint16_t loopNum = ops::CeilDiv(static_cast<uint32_t>(curAInnerLen), VL_FP32);

            __VEC_SCOPE__
            {
                MaskReg pregMask;
                uint32_t sreg = curAInnerLen;
                RegTensor<float> regDbeta, regDgamma;

                for (uint16_t k = 0; k < loopNum; k++) {
                    pregMask = UpdateMask<float>(sreg);
                    uint32_t laneOffset = k * VL_FP32;
                    LoadOneTensor<float>(dbetaTmpOutLocal, regDbeta, pregMask, laneOffset);
                    LoadOneTensor<float>(dgammaTmpOutLocal, regDgamma, pregMask, laneOffset);
                    StoreOneTensor<WEIGHT_DTYPE>(dbetaOutLocal, regDbeta, pregMask, laneOffset);
                    StoreOneTensor<WEIGHT_DTYPE>(dgammaOutLocal, regDgamma, pregMask, laneOffset);
                }
            }

            dbetaTmpOutQue_.EnQue(dbetaTmpOutTensor);
            dgammaTmpOutQue_.EnQue(dgammaTmpOutTensor);

            dbetaOutQue_.EnQue(dbetaOutTensor);
            dgammaOutQue_.EnQue(dgammaOutTensor);

            CopyOutDbetaDgammaAndTmp(offset, curAInnerLen);
        }
    }

    __aicore__ inline void FinalizeStage1(uint16_t curAInnerLen, const LocalTensor<float>& dbetaTmpIn,
                                          const LocalTensor<float>& dgammaTmpIn, const LocalTensor<float>& mean,
                                          const LocalTensor<float>& rstd, const LocalTensor<float>& dbetaOut,
                                          const LocalTensor<float>& dgammaOut)
    {
        int64_t coreNum = tilingData_->usedCoreNums;
        int64_t aAligned = tilingData_->aInnerStg1;
        __local_mem__ float* dbHiAddr = (__local_mem__ float*)dbetaTmpIn.GetPhyAddr();
        __local_mem__ float* dbLoAddr = dbHiAddr + coreNum * aAligned;
        __local_mem__ float* dgHiAddr = (__local_mem__ float*)dgammaTmpIn.GetPhyAddr();
        __local_mem__ float* dgLoAddr = dgHiAddr + coreNum * aAligned;
        __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd.GetPhyAddr();
        __local_mem__ float* dbOut = (__local_mem__ float*)dbetaOut.GetPhyAddr();
        __local_mem__ float* dgOut = (__local_mem__ float*)dgammaOut.GetPhyAddr();
        uint16_t loopNum = ops::CeilDiv(static_cast<uint32_t>(curAInnerLen), VL_FP32);

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = curAInnerLen;
            RegTensor<float> s0Hi, s0Lo, s1Hi, s1Lo, hiReg, loReg, meanReg, rstdReg, dgFinal, dbFinal;
            for (uint16_t k = 0; k < loopNum; k++) {
                pregMask = UpdateMask<float>(sreg);
                uint32_t laneOffset = k * VL_FP32;
                MicroAPI::Duplicate(s0Hi, 0.0f, pregMask);
                MicroAPI::Duplicate(s0Lo, 0.0f, pregMask);
                for (uint16_t c = 0; c < static_cast<uint16_t>(coreNum); ++c) {
                    uint32_t rowOffset = c * aAligned + laneOffset;
                    LoadOneTensor<float>(dbHiAddr, hiReg, pregMask, rowOffset);
                    LoadOneTensor<float>(dbLoAddr, loReg, pregMask, rowOffset);
                    CompAddPair<DY_DTYPE>(s0Hi, s0Lo, hiReg, loReg, pregMask);
                }
                CompRound(dbFinal, s0Hi, s0Lo, pregMask);
                StoreOneTensor<float>(dbOut, dbFinal, pregMask, laneOffset);

                MicroAPI::Duplicate(s1Hi, 0.0f, pregMask);
                MicroAPI::Duplicate(s1Lo, 0.0f, pregMask);
                for (uint16_t c = 0; c < static_cast<uint16_t>(coreNum); ++c) {
                    uint32_t rowOffset = c * aAligned + laneOffset;
                    LoadOneTensor<float>(dgHiAddr, hiReg, pregMask, rowOffset);
                    LoadOneTensor<float>(dgLoAddr, loReg, pregMask, rowOffset);
                    CompAddPair<DY_DTYPE>(s1Hi, s1Lo, hiReg, loReg, pregMask);
                }
                LoadOneTensor<float>(meanAddr, meanReg, pregMask, laneOffset);
                LoadOneTensor<float>(rstdAddr, rstdReg, pregMask, laneOffset);
                CompFinalizeDgammaDbeta(s1Hi, s1Lo, s0Hi, s0Lo, meanReg, rstdReg, pregMask, dgFinal, dbFinal);
                StoreOneTensor<float>(dgOut, dgFinal, pregMask, laneOffset);
            }
        }
    }

    __aicore__ inline void CopyInDbetaDgammaTmp(int64_t offset, int64_t curALen)
    {
        DataCopyPadExtParams<float> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = tilingData_->usedCoreNums;
        copyInParams.blockLen = curALen * sizeof(float);
        copyInParams.srcStride = (tilingData_->aDimAligned - curALen) * sizeof(float);
        copyInParams.dstStride = (tilingData_->aInnerStg1 - curALen) * sizeof(float) / BLOCK_SIZE;

        int64_t inTile = tilingData_->aInnerStg1 * tilingData_->usedCoreNums;
        LocalTensor<float> dbetaTmpTensor = dbetaTmpInQue_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpTensor, dbetaTmpGm_[offset], copyInParams,
                                                dataCopyPadExtParams);
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpTensor[inTile], dbetaTmpLoGm_[offset], copyInParams,
                                                dataCopyPadExtParams);
        dbetaTmpInQue_.EnQue(dbetaTmpTensor);

        LocalTensor<float> dgammaTmpTensor = dgammaTmpInQue_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpTensor, dgammaTmpGm_[offset], copyInParams,
                                                dataCopyPadExtParams);
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpTensor[inTile], dgammaTmpLoGm_[offset], copyInParams,
                                                dataCopyPadExtParams);
        dgammaTmpInQue_.EnQue(dgammaTmpTensor);
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

            int64_t r1OuterInCore = tilingData_->r1InnerOuterStg2;
            int64_t r1InerIncore = tilingData_->r1InnerInnerStg2;
            int64_t r1TailIncore = tilingData_->r1InnerTailStg2;
            if (isLastCore_) {
                r1OuterInCore = tilingData_->r1TailOuterStg2;
                r1TailIncore = tilingData_->r1TailTailStg2;
            }

            for (int64_t j = 0; j < r1OuterInCore; j++) {
                uint16_t curR1Len = j != (r1OuterInCore - 1) ? r1InerIncore : r1TailIncore;
                for (int64_t k = 0; k < tilingData_->r0OuterStg2; k++) {
                    int64_t dyOffset = j * tilingData_->aDim * tilingData_->r0Dim * r1InerIncore +
                                       i * tilingData_->aInnerStg2 * tilingData_->r0Dim + k * tilingData_->r0InnerStg2;

                    uint32_t curR0Len = tilingData_->r0InnerStg2;
                    uint32_t curR0AlignedLen = tilingData_->r0InnerStg2;
                    if (k == (tilingData_->r0OuterStg2 - 1)) {
                        curR0Len = tilingData_->r0TailStg2;
                        curR0AlignedLen = tilingData_->r0TailAlignedStg2;
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
                        RegTensor<float> regU, regP, regQ, regT, regDx;
                        RegTensor<float> rhiReg, rlo, dbetaOverR, pMag;
                        MaskReg pFinite;
                        // 与 golden/torch 同构：dx = ((dy - dbeta/R) - (xhat*dgamma)/R) * gamma * rstd；
                        // 除法用 (rhi + rlo) 的 FMA 形式（单次舍入，等价真除法），rlo 按**精确整数 R**
                        // 补残差（见 comp_sum.h：只修正到 fl32(R) 时整通道会带 (R-fl32(R))/R 的同号偏差）。
                        Duplicate(rhiReg, hRecipVal_, pregMaskAll);
                        Comp::CompExactDivisorReciprocalLo(rlo, rhiReg,
                                                           static_cast<float>(tilingData_->r1Dim * tilingData_->r0Dim),
                                                           hNegDelta_, pregMaskAll);

                        for (uint16_t i = 0; i < curAInnerLen; i++) {
                            LoadsTensorForDtypeT<float>(meanAddr, regMean, pregMaskAll, i);
                            LoadsTensorForDtypeT<float>(rstdAddr, regRstd, pregMaskAll, i);
                            LoadsTensorForDtypeT<float>(dbetaTmpAddr, regDbetaTmp, pregMaskAll, i);
                            LoadsTensorForDtypeT<float>(dgammaTmpAddr, regDgammaTmp, pregMaskAll, i);
                            LoadsTensorForDtypeT<WEIGHT_DTYPE>(gammaAddr, regGamma, pregMaskAll, i);
                            Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dbetaOverR, regDbetaTmp, rlo,
                                                                                  pregMaskAll);
                            MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(dbetaOverR, regDbetaTmp, rhiReg,
                                                                               pregMaskAll);

                            uint32_t sreg = curR0Len;
                            for (uint16_t j = 0; j < loopNum; j++) { // r0
                                pregMask = UpdateMask<float>(sreg);
                                for (uint16_t k = 0; k < curR1Len; k++) {
                                    uint32_t offset = i * curR1Len * curR0AlignedLen + k * curR0AlignedLen +
                                                      j * VL_FP32;
                                    LoadOneTensor<DY_DTYPE>(dyLocal, regDy, pregMask, offset);
                                    LoadOneTensor<DY_DTYPE>(xLocal, regX, pregMask, offset);

                                    Sub<float, MicroAPI::MaskMergeMode::ZEROING>(regU, regDy, dbetaOverR, pregMask);
                                    Sub<float, MicroAPI::MaskMergeMode::ZEROING>(regX, regX, regMean, pregMask);
                                    Mul<float, MicroAPI::MaskMergeMode::ZEROING>(regP, regX, regRstd, pregMask);
                                    Mul<float, MicroAPI::MaskMergeMode::ZEROING>(regP, regP, regDgammaTmp, pregMask);
                                    Abs(pMag, regP, pregMask);
                                    CompareScalar<float, CMPMODE::LE>(pFinite, pMag, FP32_MAX_FINITE, pregMask);
                                    Mul<float, MicroAPI::MaskMergeMode::ZEROING>(regQ, regP, rlo, pFinite);
                                    MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(regQ, regP, rhiReg, pregMask);
                                    Sub<float, MicroAPI::MaskMergeMode::ZEROING>(regT, regU, regQ, pregMask);
                                    Mul<float, MicroAPI::MaskMergeMode::ZEROING>(regT, regT, regGamma, pregMask);
                                    Mul<float, MicroAPI::MaskMergeMode::ZEROING>(regDx, regT, regRstd, pregMask);

                                    StoreOneTensor<DY_DTYPE>(dxLocal, regDx, pregMask, offset);
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
    const BatchNormGradExt2RARSplitCoreR1TilingData* tilingData_{nullptr};
    TPipe* pipe_{nullptr};

    GlobalTensor<DY_DTYPE> dyGm_;
    GlobalTensor<DY_DTYPE> xGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<float> dbetaTmpGm_;
    GlobalTensor<float> dbetaTmpLoGm_;
    GlobalTensor<float> dgammaTmpGm_;
    GlobalTensor<float> dgammaTmpLoGm_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> rstdInQue_;
    TBuf<TPosition::VECCALC> compAccBuf_;
    int64_t accChannelAligned_{0};
    TBuf<TPosition::VECCALC> dbetaCacheBuffer_;
    TBuf<TPosition::VECCALC> dgammaCacheBuffer_;
    TBuf<TPosition::VECCALC> reduceInTmpBuffer_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dbetaTmpOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dgammaTmpOutQue_;

    LocalTensor<float> reduceInTmpTensor_;
    LocalTensor<float> dbetaMainTmpTensor_;
    LocalTensor<float> dgammaMainTmpTensor_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dbetaTmpInQue_;
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
    int64_t reduceTmpBufOffset_{0};
    float hRecipVal_{0};
    float hNegDelta_{0}; // -(R - fl32(R))：把 1/fl32(R) 两字倒数修正到 1/R
    bool isLastCore_{false};
};
} // namespace BatchNormGradExt2
#endif // __BATCH_NORM_GRAD_EXT2_SPLIT_CORE_R1_H__
