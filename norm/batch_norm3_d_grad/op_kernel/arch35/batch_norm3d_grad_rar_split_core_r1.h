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
 * \file batch_norm3d_grad_rar_split_core_r1.h
 * \brief
 */

#ifndef __BATCH_NORM3D_GRAD_SPLIT_CORE_R1_H__
#define __BATCH_NORM3D_GRAD_SPLIT_CORE_R1_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm3d_grad_common.h"

namespace BatchNorm3DGrad {
using namespace AscendC;
using namespace AscendC::MicroAPI;

static constexpr int64_t CONST_ONE = 1;
static constexpr int64_t CONST_TWO = 2;
static constexpr int64_t ULONG_BIT_LEN = 64;
static constexpr int32_t BUFFER_NUM = 2;
static constexpr int32_t BUFFER_DEPTH = 1;
static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
static constexpr int32_t BLOCK_FP32_NUMS = BLOCK_SIZE / sizeof(float);

template <typename DY_DTYPE, typename WEIGHT_DTYPE>
class BatchNorm3DGradRARSplitCoreR1 {
public:
    __aicore__ inline BatchNorm3DGradRARSplitCoreR1(TPipe* pipeIn,
                                                    const BatchNorm3DGradRARSplitCoreR1TilingData* tilingDataIn)
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

        // 跨核部分和按 hi/lo 两肢分平面落盘：dbetaHi | dbetaLo | dgammaHi | dgammaLo
        int64_t planeFloats = tilingData_->usedCoreNums * tilingData_->aDimAligned;
        dbetaTmpGm_.SetGlobalBuffer((__gm__ float*)(workspace));
        dbetaTmpLoGm_.SetGlobalBuffer((__gm__ float*)(workspace) + planeFloats);
        dgammaTmpGm_.SetGlobalBuffer((__gm__ float*)(workspace) + planeFloats * CONST_TWO);
        dgammaTmpLoGm_.SetGlobalBuffer((__gm__ float*)(workspace) + planeFloats * 3);

        // stg0、stg2相同
        int64_t dyOffset = blkIdx_ * tilingData_->r1Inner * tilingData_->r0Dim * tilingData_->aDim;
        dyGm_.SetGlobalBuffer((__gm__ DY_DTYPE*)(dy) + dyOffset);
        xGm_.SetGlobalBuffer((__gm__ DY_DTYPE*)(x) + dyOffset);
        dxGm_.SetGlobalBuffer((__gm__ DY_DTYPE*)(dx) + dyOffset);

        countF_ = static_cast<float>(tilingData_->r1Dim * tilingData_->r0Dim);
        vMte3Event_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        mte3VEvent_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        // 预置信号：首轮 CalcReduceSum 的 WaitFlag 直接通过
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3VEvent_);
    }

    __aicore__ inline void Stage0InitBuffer()
    {
        // 二分相关buff初始化
        int64_t dyLen = tilingData_->r0InnerStg0 * tilingData_->r1InnerInnerStg0 * tilingData_->aInnerStg0;
        int64_t meanSize = tilingData_->aInnerAlignedStg0 * sizeof(float);

        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));

        // fp16 增加dy cast结果的空间
        if constexpr (IsSameType<DY_DTYPE, float>::value) {
            pipe_->InitBuffer(reduceInTmpBuffer_, dyLen * sizeof(float));
        } else {
            pipe_->InitBuffer(reduceInTmpBuffer_, dyLen * sizeof(float) * CONST_TWO);
            reduceTmpBufOffset_ = dyLen;
        }

        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, meanSize);

        // [hi|lo] 双肢各占 meanSize
        pipe_->InitBuffer(dbetaTmpOutQue_, BUFFER_NUM, meanSize * CONST_TWO);
        pipe_->InitBuffer(dgammaTmpOutQue_, BUFFER_NUM, meanSize * CONST_TWO);

        // 核内reducesum结果使用 (hi,lo) 两槽位做精确补偿累加
        pipe_->InitBuffer(dbetaCacheBuffer_, meanSize * CONST_TWO);
        pipe_->InitBuffer(dgammaCacheBuffer_, meanSize * CONST_TWO);
    }

    __aicore__ inline void Stage1InitBuffer()
    {
        pipe_->Reset();

        // hi/lo 两平面各占 usedCoreNums 行
        int64_t aSizeIn = tilingData_->aInnerStg1 * sizeof(float) * tilingData_->usedCoreNums * CONST_TWO;
        pipe_->InitBuffer(dbetaTmpInQue_, BUFFER_NUM, aSizeIn);
        pipe_->InitBuffer(dgammaTmpInQue_, BUFFER_NUM, aSizeIn);

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

    __aicore__ inline void CalcReduceSum(uint32_t curAInnerLen, int64_t aOffset, __local_mem__ float* meanLocal,
                                         __local_mem__ float* rstdLocal)
    {
        LocalTensor<float> dbetaCacheTensor = dbetaCacheBuffer_.Get<float>();
        LocalTensor<float> dgammaCacheTensor = dgammaCacheBuffer_.Get<float>();
        reduceInTmpTensor_ = reduceInTmpBuffer_.Get<float>();

        dbetaMainTmpTensor_ = dbetaTmpOutQue_.template AllocTensor<float>();
        dgammaMainTmpTensor_ = dgammaTmpOutQue_.template AllocTensor<float>();
        dbetaCacheTensor_ = dbetaCacheTensor;
        dgammaCacheTensor_ = dgammaCacheTensor;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3VEvent_);
        InitAccumulator(dbetaCacheTensor_, tilingData_->aInnerAlignedStg0);
        InitAccumulator(dgammaCacheTensor_, tilingData_->aInnerAlignedStg0);

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

            ProcessMainBlock(meanLocal, rstdLocal, curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);

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

                ProcessFoldBlock(meanLocal, rstdLocal, curAInnerLen, curR1LenFold, curR0LenFold, curR0AlignedLenFold,
                                 dyOffsetFold);
            }
            // 块内 ReduceSum 的部分和用严格 TwoSum 累加进 (hi, lo)，取代原先未补偿的 binary-carry Add
            AccumulateBlockExact(dbetaCacheTensor_, dbetaMainTmpTensor_, tilingData_->aInnerAlignedStg0);
            AccumulateBlockExact(dbetaCacheTensor_, dbetaMainTmpTensor_[tilingData_->aInnerAlignedStg0],
                                 tilingData_->aInnerAlignedStg0);
            AccumulateBlockExact(dgammaCacheTensor_, dgammaMainTmpTensor_, tilingData_->aInnerAlignedStg0);
            AccumulateBlockExact(dgammaCacheTensor_, dgammaMainTmpTensor_[tilingData_->aInnerAlignedStg0],
                                 tilingData_->aInnerAlignedStg0);
        }

        if (basicLoop == 0) { // 直接输出到dbetaMainTmpTensor_，dgammaMainTmpTensor_
            int64_t dyOffset = aOffset * tilingData_->r0Dim;
            ProcessMainBlock(meanLocal, rstdLocal, curAInnerLen, r1TailIncore, tilingData_->r0TailStg0,
                             tilingData_->r0TailAlignedStg0, dyOffset);
            AccumulateBlockExact(dbetaCacheTensor_, dbetaMainTmpTensor_, tilingData_->aInnerAlignedStg0);
            AccumulateBlockExact(dbetaCacheTensor_, dbetaMainTmpTensor_[tilingData_->aInnerAlignedStg0],
                                 tilingData_->aInnerAlignedStg0);
            AccumulateBlockExact(dgammaCacheTensor_, dgammaMainTmpTensor_, tilingData_->aInnerAlignedStg0);
            AccumulateBlockExact(dgammaCacheTensor_, dgammaMainTmpTensor_[tilingData_->aInnerAlignedStg0],
                                 tilingData_->aInnerAlignedStg0);
        }

        // hi/lo 两肢直接落 GM，跨核合流前不舍入；唯一一次舍入在 stage1 末尾
        ClampAccumulatorLo(dbetaCacheTensor_, tilingData_->aInnerAlignedStg0);
        ClampAccumulatorLo(dgammaCacheTensor_, tilingData_->aInnerAlignedStg0);

        dbetaTmpOutQue_.FreeTensor(dbetaMainTmpTensor_);
        dgammaTmpOutQue_.FreeTensor(dgammaMainTmpTensor_);
    }

    __aicore__ inline void ProcessMainBlock(__local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
                                            const uint32_t curAInnerLen, const uint32_t curR1Len,
                                            const uint32_t curR0Len, const uint32_t curR0AlignedLen,
                                            const int64_t dyOffset)
    {
        CopyInDyAndX(curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);

        LocalTensor<DY_DTYPE> dyTensor = dyInQue_.DeQue<DY_DTYPE>();
        LocalTensor<DY_DTYPE> xTensor = xInQue_.DeQue<DY_DTYPE>();

        CalcDyAndX(dyTensor, xTensor, reduceInTmpTensor_, meanLocal, rstdLocal, curR0AlignedLen, curAInnerLen,
                   curR1Len);
        xInQue_.FreeTensor(xTensor);
        if constexpr (!IsSameType<DY_DTYPE, float>::value) {
            dyInQue_.FreeTensor(dyTensor);
        }

        uint32_t fusedRLen = curR0AlignedLen * curR1Len;

        // 块内补偿归约替代 AR ReduceSum 裸树：[hi|lo] 双肢写入 mainTmp 前后两半，
        // 块级部分和即正确舍入，跨块由 AccumulateBlockExact 精确累加
        Duplicate(dbetaMainTmpTensor_, float(0.0), tilingData_->aInnerAlignedStg0 * CONST_TWO);
        Duplicate(dgammaMainTmpTensor_, float(0.0), tilingData_->aInnerAlignedStg0 * CONST_TWO);

        __local_mem__ float* dbetaTerms;
        if constexpr (IsSameType<DY_DTYPE, float>::value) {
            dbetaTerms = (__local_mem__ float*)dyTensor.GetPhyAddr();
        } else {
            dbetaTerms = (__local_mem__ float*)reduceInTmpTensor_.GetPhyAddr() + reduceTmpBufOffset_;
        }
        __local_mem__ float* dgammaTerms = (__local_mem__ float*)reduceInTmpTensor_.GetPhyAddr();
        CompensatedRowReduceHiLo(
            dbetaTerms, (__local_mem__ float*)dbetaMainTmpTensor_.GetPhyAddr(),
            (__local_mem__ float*)dbetaMainTmpTensor_.GetPhyAddr() + tilingData_->aInnerAlignedStg0, curAInnerLen,
            fusedRLen, fusedRLen);
        CompensatedRowReduceHiLo(
            dgammaTerms, (__local_mem__ float*)dgammaMainTmpTensor_.GetPhyAddr(),
            (__local_mem__ float*)dgammaMainTmpTensor_.GetPhyAddr() + tilingData_->aInnerAlignedStg0, curAInnerLen,
            fusedRLen, fusedRLen);
        if constexpr (IsSameType<DY_DTYPE, float>::value) {
            dyInQue_.FreeTensor(dyTensor);
        }
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

        CalcDyAndX(dyTensor, xTensor, reduceInTmpTensor_, meanLocal, rstdLocal, curR0AlignedLen, curAInnerLen,
                   curR1Len);

        xInQue_.FreeTensor(xTensor);
        if constexpr (!IsSameType<DY_DTYPE, float>::value) {
            dyInQue_.FreeTensor(dyTensor);
        }

        uint32_t fusedRLen = curR0AlignedLen * curR1Len;

        LocalTensor<float> dbetaFoldTmpTensor = dbetaTmpOutQue_.template AllocTensor<float>();
        LocalTensor<float> dgammaFoldTmpTensor = dgammaTmpOutQue_.template AllocTensor<float>();

        // 块内补偿归约替代 AR ReduceSum 裸树：[hi|lo] 双肢写入 foldTmp 前后两半
        Duplicate(dbetaFoldTmpTensor, float(0.0), tilingData_->aInnerAlignedStg0 * CONST_TWO);
        Duplicate(dgammaFoldTmpTensor, float(0.0), tilingData_->aInnerAlignedStg0 * CONST_TWO);

        __local_mem__ float* dbetaTerms;
        if constexpr (IsSameType<DY_DTYPE, float>::value) {
            dbetaTerms = (__local_mem__ float*)dyTensor.GetPhyAddr();
        } else {
            dbetaTerms = (__local_mem__ float*)reduceInTmpTensor_.GetPhyAddr() + reduceTmpBufOffset_;
        }
        __local_mem__ float* dgammaTerms = (__local_mem__ float*)reduceInTmpTensor_.GetPhyAddr();
        CompensatedRowReduceHiLo(dbetaTerms, (__local_mem__ float*)dbetaFoldTmpTensor.GetPhyAddr(),
                                 (__local_mem__ float*)dbetaFoldTmpTensor.GetPhyAddr() + tilingData_->aInnerAlignedStg0,
                                 curAInnerLen, fusedRLen, fusedRLen);
        CompensatedRowReduceHiLo(
            dgammaTerms, (__local_mem__ float*)dgammaFoldTmpTensor.GetPhyAddr(),
            (__local_mem__ float*)dgammaFoldTmpTensor.GetPhyAddr() + tilingData_->aInnerAlignedStg0, curAInnerLen,
            fusedRLen, fusedRLen);
        if constexpr (IsSameType<DY_DTYPE, float>::value) {
            dyInQue_.FreeTensor(dyTensor);
        }

        // fold 部分单独精确累加进 (hi, lo)：先 hi 肢后 lo 肢
        AccumulateBlockExact(dbetaCacheTensor_, dbetaFoldTmpTensor, tilingData_->aInnerAlignedStg0);
        AccumulateBlockExact(dbetaCacheTensor_, dbetaFoldTmpTensor[tilingData_->aInnerAlignedStg0],
                             tilingData_->aInnerAlignedStg0);
        AccumulateBlockExact(dgammaCacheTensor_, dgammaFoldTmpTensor, tilingData_->aInnerAlignedStg0);
        AccumulateBlockExact(dgammaCacheTensor_, dgammaFoldTmpTensor[tilingData_->aInnerAlignedStg0],
                             tilingData_->aInnerAlignedStg0);

        dbetaTmpOutQue_.FreeTensor(dbetaFoldTmpTensor);
        dgammaTmpOutQue_.FreeTensor(dgammaFoldTmpTensor);
    }

    __aicore__ inline void CalcDyAndX(LocalTensor<DY_DTYPE>& dyTensor, LocalTensor<DY_DTYPE>& xTensor,
                                      LocalTensor<float>& reduceTmpBufferTensor, __local_mem__ float* meanLocal,
                                      __local_mem__ float* rstdLocal, uint32_t curR0AlignedLen, uint16_t curAInnerLen,
                                      uint16_t curR1Len)
    {
        __local_mem__ DY_DTYPE* dyLocal = (__local_mem__ DY_DTYPE*)dyTensor.GetPhyAddr();
        __local_mem__ DY_DTYPE* xLocal = (__local_mem__ DY_DTYPE*)xTensor.GetPhyAddr();
        __local_mem__ float* dgammaReduceTmpLocal = (__local_mem__ float*)reduceTmpBufferTensor.GetPhyAddr();

        __local_mem__ float* dbetaReduceTmpLocal = (__local_mem__ float*)reduceTmpBufferTensor.GetPhyAddr() +
                                                   reduceTmpBufOffset_;

        uint16_t loopNum = ops::CeilDiv(curR0AlignedLen, VL_FP32);

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            MaskReg pregMaskAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            RegTensor<float> regMean, regRstd, regDy, regX;

            for (uint16_t i = 0; i < curAInnerLen; i++) {
                uint32_t sreg = curR0AlignedLen;
                LoadsTensorForDtypeT<float>(meanLocal, regMean, pregMaskAll, i);
                LoadsTensorForDtypeT<float>(rstdLocal, regRstd, pregMaskAll, i);
                for (uint16_t j = 0; j < loopNum; j++) {
                    pregMask = UpdateMask<float>(sreg);
                    for (uint16_t k = 0; k < curR1Len; k++) {
                        uint32_t offset = i * curR1Len * curR0AlignedLen + k * curR0AlignedLen + j * VL_FP32;
                        LoadOneTensor<DY_DTYPE>(dyLocal, regDy, pregMask, offset);
                        LoadOneTensor<DY_DTYPE>(xLocal, regX, pregMask, offset);

                        Sub(regX, regX, regMean, pregMask);
                        Mul(regX, regX, regRstd, pregMask);
                        Mul(regX, regX, regDy, pregMask);
                        DataCopy((__local_mem__ float*)dgammaReduceTmpLocal + offset, regX, pregMask);

                        if constexpr (!IsSameType<DY_DTYPE, float>::value) {
                            DataCopy((__local_mem__ float*)dbetaReduceTmpLocal + offset, regDy, pregMask);
                        }
                    }
                }
            }
        }
    }

    // 残差肢写盘前守卫：hi 非有限时 lo 可能是 NaN（inf-inf），置 0 保持 inf+0=inf
    __aicore__ inline void ClampAccumulatorLo(const LocalTensor<float>& accTensor, uint32_t count)
    {
        uint16_t loopTimes = ops::CeilDiv(count, VL_FP32);
        uint32_t stride = tilingData_->aInnerAlignedStg0;
        __local_mem__ float* accAddr = (__local_mem__ float*)accTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pMask;
            uint32_t sreg = count;
            MicroAPI::RegTensor<float> lo;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(sreg);
                DataCopy(lo, accAddr + stride + i * VL_FP32);
                ClampResidual(lo, pMask);
                DataCopy(accAddr + stride + i * VL_FP32, lo, pMask);
            }
        }
    }

    __aicore__ inline void CopyOutDbetaDgammaTmp(const uint32_t curAInnerLen, const int64_t aOffset)
    {
        // acc 是裸 TBuf（不经队列），GM 写盘前须先等 V 管道的累加/Clamp 写落地
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3Event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3Event_);
        int64_t gmOffset = blkIdx_ * tilingData_->aDimAligned + aOffset;
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = curAInnerLen * sizeof(float);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        LocalTensor<float> dbetaAcc = dbetaCacheBuffer_.Get<float>();
        LocalTensor<float> dgammaAcc = dgammaCacheBuffer_.Get<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpGm_[gmOffset], dbetaAcc, copyOutParams);
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpLoGm_[gmOffset], dbetaAcc[tilingData_->aInnerAlignedStg0],
                                                copyOutParams);
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpGm_[gmOffset], dgammaAcc, copyOutParams);
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpLoGm_[gmOffset], dgammaAcc[tilingData_->aInnerAlignedStg0],
                                                copyOutParams);
        // 与下一轮 CalcReduceSum 的 V 写配对：GM 读走后才能复写 acc
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3VEvent_);
    }

    __aicore__ inline int64_t GetCacheID(const int64_t idx)
    {
        return ScalarGetCountOfValue<1>(idx ^ (idx + CONST_ONE)) - CONST_ONE;
    }

    // ---- 核内块间精确补偿累加（hi/lo 两槽位）----
    __aicore__ inline void InitAccumulator(const LocalTensor<float>& accTensor, uint32_t count)
    {
        uint16_t loopTimes = ops::CeilDiv(count, VL_FP32);
        __local_mem__ float* accAddr = (__local_mem__ float*)accTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pMask;
            uint32_t sreg = count;
            MicroAPI::RegTensor<float> zeroReg;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(sreg);
                MicroAPI::Duplicate(zeroReg, 0.0f, pMask);
                DataCopy(accAddr + i * VL_FP32, zeroReg, pMask);
                DataCopy(accAddr + tilingData_->aInnerAlignedStg0 + i * VL_FP32, zeroReg, pMask);
            }
        }
    }

    __aicore__ inline void AccumulateBlockExact(const LocalTensor<float>& accTensor,
                                                const LocalTensor<float>& valTensor, uint32_t count)
    {
        uint16_t loopTimes = ops::CeilDiv(count, VL_FP32);
        uint32_t stride = tilingData_->aInnerAlignedStg0;
        __local_mem__ float* accAddr = (__local_mem__ float*)accTensor.GetPhyAddr();
        __local_mem__ float* valAddr = (__local_mem__ float*)valTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pMask;
            uint32_t sreg = count;
            MicroAPI::RegTensor<float> hi, lo, val;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(sreg);
                DataCopy(hi, accAddr + i * VL_FP32);
                DataCopy(lo, accAddr + stride + i * VL_FP32);
                DataCopy(val, valAddr + i * VL_FP32);
                AccumulateWithError(hi, lo, val, pMask);
                DataCopy(accAddr + i * VL_FP32, hi, pMask);
                DataCopy(accAddr + stride + i * VL_FP32, lo, pMask);
            }
        }
    }

    // stage1 核间两肢（hi/lo 平面）精确求和：所有行统一累加后单次舍入
    __aicore__ inline void CompensatedRowSumTwoLimb(const LocalTensor<float>& dstTensor,
                                                    const LocalTensor<float>& srcTensor, uint16_t rowCount,
                                                    uint32_t stride, uint32_t count)
    {
        uint16_t loopTimes = ops::CeilDiv(count, VL_FP32);
        __local_mem__ float* srcAddr = (__local_mem__ float*)srcTensor.GetPhyAddr();
        __local_mem__ float* dstAddr = (__local_mem__ float*)dstTensor.GetPhyAddr();
        int64_t loOffset = static_cast<int64_t>(rowCount) * stride;
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pMask;
            uint32_t sreg = count;
            MicroAPI::RegTensor<float> hi, lo, val;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(sreg);
                MicroAPI::Duplicate(hi, 0.0f, pMask);
                MicroAPI::Duplicate(lo, 0.0f, pMask);
                for (uint16_t r = 0; r < rowCount; ++r) {
                    DataCopy(val, srcAddr + r * stride + i * VL_FP32);
                    AccumulateWithError(hi, lo, val, pMask);
                }
                for (uint16_t r = 0; r < rowCount; ++r) {
                    DataCopy(val, srcAddr + loOffset + r * stride + i * VL_FP32);
                    AccumulateWithError(hi, lo, val, pMask);
                }
                ClampResidual(lo, pMask);
                MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(hi, hi, lo, pMask);
                DataCopy(dstAddr + i * VL_FP32, hi, pMask);
            }
        }
    }

    __aicore__ inline void UpdateCache(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                       const int64_t cacheID, const uint32_t stride, const uint32_t count)
    {
        uint16_t outerLoopTimes = ops::CeilDiv(count, VL_FP32);
        uint16_t innerLoopTimes = cacheID;
        uint32_t outerLoopStride = VL_FP32;
        uint32_t innerLoopStride = stride;

        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ float* cah = (__local_mem__ float*)dstTensor.GetPhyAddr() + cacheID * stride;
        __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t sreg = static_cast<uint32_t>(count);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                DataCopy(aReg, (__local_mem__ float*)src + i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    DataCopy(bReg, (__local_mem__ float*)dst + i * outerLoopStride + j * innerLoopStride);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                }
                DataCopy((__local_mem__ float*)cah + i * outerLoopStride, aReg, pMask);
            }
        }
    }

    // stage 1
    __aicore__ inline void CalcDbetaDgamma()
    {
        for (int64_t i = 0; i < tilingData_->aOuterStg1; i++) {
            uint16_t curAInnerLen = i != (tilingData_->aOuterStg1 - 1) ? tilingData_->aInnerStg1 :
                                                                         tilingData_->aTailStg1;

            int64_t offset = i * tilingData_->aInnerStg1;
            CopyInDbetaDgammaTmp(offset, curAInnerLen);

            LocalTensor<float> dbetaTmpIn = dbetaTmpInQue_.template DeQue<float>();
            LocalTensor<float> dgammaTmpIn = dgammaTmpInQue_.template DeQue<float>();
            LocalTensor<float> dbetaTmpOutTensor = dbetaTmpOutQue_.AllocTensor<float>();
            LocalTensor<float> dgammaTmpOutTensor = dgammaTmpOutQue_.AllocTensor<float>();

            // 核间 hi/lo 双肢统一精确累加，整条归约只在这里做一次舍入
            CompensatedRowSumTwoLimb(dbetaTmpOutTensor, dbetaTmpIn, static_cast<uint16_t>(tilingData_->usedCoreNums),
                                     static_cast<uint32_t>(tilingData_->aInnerStg1), curAInnerLen);
            CompensatedRowSumTwoLimb(dgammaTmpOutTensor, dgammaTmpIn, static_cast<uint16_t>(tilingData_->usedCoreNums),
                                     static_cast<uint32_t>(tilingData_->aInnerStg1), curAInnerLen);

            dbetaTmpInQue_.FreeTensor(dbetaTmpIn);
            dgammaTmpInQue_.FreeTensor(dgammaTmpIn);

            // 小shape输出处理
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

                for (uint16_t i = 0; i < loopNum; i++) {
                    pregMask = UpdateMask<float>(sreg);
                    uint32_t offset = i * VL_FP32;
                    LoadOneTensor<float>(dbetaTmpOutLocal, regDbeta, pregMask, offset);
                    LoadOneTensor<float>(dgammaTmpOutLocal, regDgamma, pregMask, offset);
                    StoreOneTensor<WEIGHT_DTYPE>(dbetaOutLocal, regDbeta, pregMask, offset);
                    StoreOneTensor<WEIGHT_DTYPE>(dgammaOutLocal, regDgamma, pregMask, offset);
                }
            }

            dbetaTmpOutQue_.EnQue(dbetaTmpOutTensor);
            dgammaTmpOutQue_.EnQue(dgammaTmpOutTensor);

            dbetaOutQue_.EnQue(dbetaOutTensor);
            dgammaOutQue_.EnQue(dgammaOutTensor);

            CopyOutDbetaDgammaAndTmp(offset, curAInnerLen);
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

        int64_t loPlaneOffset = tilingData_->usedCoreNums * tilingData_->aInnerStg1;
        LocalTensor<float> dbetaTmpTensor = dbetaTmpInQue_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpTensor, dbetaTmpGm_[offset], copyInParams,
                                                dataCopyPadExtParams);
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpTensor[loPlaneOffset], dbetaTmpLoGm_[offset], copyInParams,
                                                dataCopyPadExtParams);
        dbetaTmpInQue_.EnQue(dbetaTmpTensor);

        LocalTensor<float> dgammaTmpTensor = dgammaTmpInQue_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpTensor, dgammaTmpGm_[offset], copyInParams,
                                                dataCopyPadExtParams);
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpTensor[loPlaneOffset], dgammaTmpLoGm_[offset], copyInParams,
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
                        RegTensor<float> regCount, regDbetaOverR, regDgammaOverR;

                        for (uint16_t i = 0; i < curAInnerLen; i++) {
                            LoadsTensorForDtypeT<float>(meanAddr, regMean, pregMaskAll, i);
                            LoadsTensorForDtypeT<float>(rstdAddr, regRstd, pregMaskAll, i);
                            LoadsTensorForDtypeT<float>(dbetaTmpAddr, regDbetaTmp, pregMaskAll, i);
                            LoadsTensorForDtypeT<float>(dgammaTmpAddr, regDgammaTmp, pregMaskAll, i);
                            LoadsTensorForDtypeT<WEIGHT_DTYPE>(gammaAddr, regGamma, pregMaskAll, i);
                            // dbeta/R 用真除法逐项算，避免先在 1e7 量级合成再乘倒数
                            MicroAPI::Duplicate(regCount, countF_, pregMaskAll);
                            MicroAPI::Div(regDbetaOverR, regDbetaTmp, regCount, pregMaskAll);
                            if constexpr (IsSameType<DY_DTYPE, float>::value) {
                                MicroAPI::Div(regDgammaOverR, regDgammaTmp, regCount, pregMaskAll);
                            }

                            uint32_t sreg = curR0Len;
                            for (uint16_t j = 0; j < loopNum; j++) { // r0
                                pregMask = UpdateMask<float>(sreg);
                                for (uint16_t k = 0; k < curR1Len; k++) {
                                    uint32_t offset = i * curR1Len * curR0AlignedLen + k * curR0AlignedLen +
                                                      j * VL_FP32;
                                    LoadOneTensor<DY_DTYPE>(dyLocal, regDy, pregMask, offset);
                                    LoadOneTensor<DY_DTYPE>(xLocal, regX, pregMask, offset);

                                    Sub(regX, regX, regMean, pregMask);
                                    Mul(regX, regX, regRstd, pregMask);
                                    if constexpr (IsSameType<DY_DTYPE, float>::value) {
                                        Mul(regX, regX, regDgammaOverR, pregMask);
                                    } else {
                                        Mul(regX, regX, regDgammaTmp, pregMask);
                                        MicroAPI::Div(regX, regX, regCount, pregMask);
                                    }
                                    Sub(regDy, regDy, regDbetaOverR, pregMask);
                                    Sub(regX, regDy, regX, pregMask);
                                    Mul(regX, regX, regGamma, pregMask);
                                    Mul(regX, regX, regRstd, pregMask);

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
    const BatchNorm3DGradRARSplitCoreR1TilingData* tilingData_{nullptr};
    TPipe* pipe_{nullptr};

    GlobalTensor<DY_DTYPE> dyGm_;
    GlobalTensor<DY_DTYPE> xGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<float> dbetaTmpGm_;
    GlobalTensor<float> dbetaTmpLoGm_;
    GlobalTensor<float> dgammaTmpGm_;
    GlobalTensor<float> dgammaTmpLoGm_;
    event_t vMte3Event_;
    event_t mte3VEvent_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> rstdInQue_;
    TBuf<TPosition::VECCALC> dbetaCacheBuffer_;
    TBuf<TPosition::VECCALC> dgammaCacheBuffer_;
    TBuf<TPosition::VECCALC> reduceInTmpBuffer_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dbetaTmpOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dgammaTmpOutQue_;

    LocalTensor<float> reduceInTmpTensor_;
    LocalTensor<float> dbetaMainTmpTensor_;
    LocalTensor<float> dgammaMainTmpTensor_;
    LocalTensor<float> dbetaCacheTensor_;
    LocalTensor<float> dgammaCacheTensor_;

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
    float countF_{0.0f};
    bool isLastCore_{false};
};
} // namespace BatchNorm3DGrad
#endif // __BATCH_NORM3D_GRAD_SPLIT_CORE_R1_H__
