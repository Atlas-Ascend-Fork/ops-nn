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
 * \file batch_norm3d_grad_ra_split_r_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM3D_GRAD_RA_SPLIT_R_REGBASE_H__
#define __BATCH_NORM3D_GRAD_RA_SPLIT_R_REGBASE_H__

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "batch_norm3d_grad_common.h"

namespace BatchNorm3DGrad {
using namespace AscendC;

template <typename DY_TYPE, typename WEIGHT_TYPE>
class BatchNorm3DGradRASplitR {
public:
    __aicore__ inline BatchNorm3DGradRASplitR(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR mean, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx,
                                GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace,
                                const BatchNorm3DGradRASplitRTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        tilingData_ = tilingData;
        int64_t offset = blockIdx_ * tilingData_->blockFactor * tilingData_->aDim;
        aFactorAlign_ = tilingData_->aFactorAlign;
        if (blockIdx_ == (tilingData_->usedCoreNum - 1)) {
            currBlockFactor_ = tilingData_->tailBlockFactor;
            binaryBlockCnt_ = tilingData_->lastCoreBlockCnt;
            binaryFoldPoint_ = tilingData_->lastCoreFoldPoint;
            binaryBlockTail_ = tilingData_->lastCoreLoopTail;
            dxLoopFactor_ = tilingData_->dxLastCoreFactor;
            dxLoopTail_ = tilingData_->dxLastCoreTail;
            dxLoopTimes_ = tilingData_->dxLastCoreTimes;
        } else {
            currBlockFactor_ = tilingData_->blockFactor;
            binaryBlockCnt_ = tilingData_->binaryBlockCnt;
            binaryFoldPoint_ = tilingData_->binaryFoldPoint;
            binaryBlockTail_ = tilingData_->binaryBlockTail;
            dxLoopFactor_ = tilingData_->dxLoopFactor;
            dxLoopTail_ = tilingData_->dxLoopTail;
            dxLoopTimes_ = tilingData_->dxLoopTimes;
        }
        currLoopFactor_ = (tilingData_->rLoopFactor > currBlockFactor_) ? currBlockFactor_ : tilingData_->rLoopFactor;

        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dy) + offset);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(x) + offset);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dx) + offset);
        meanGm_.SetGlobalBuffer((__gm__ float*)(mean));
        rstdGm_.SetGlobalBuffer((__gm__ float*)(rstd));
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(gamma));
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dgamma));
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dbeta));
        dbetaWorkSpace_.SetGlobalBuffer((__gm__ float*)(workspace));
        dgammaWorkSpace_.SetGlobalBuffer((__gm__ float*)(workspace) + 2 * tilingData_->usedCoreNum * tilingData_->aDim);
        countF_ = static_cast<float>(tilingData_->rDim);
    }

    __aicore__ inline void Process()
    {
        // stage 0： 核内二分累加dbeta/dgamma
        Stage0InitBuffer();
        CalcDbetaDgammaInCore();
        // stage 1: 计算核间dbeta/dgamma
        SyncAll();
        if (blockIdx_ == 0) {
            Stage1InitBuffer();
            ProcessDbetaDgamma();
        }
        // stage 2: 计算输出
        SyncAll();
        Stage2InitBuffer();
        ProcessDX();
    }

private:
    __aicore__ inline void Stage0InitBuffer()
    {
        int64_t rDimSize = currLoopFactor_ * aFactorAlign_;
        int64_t aDimSize = aFactorAlign_ * sizeof(float);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        // 现在直接逐元素精确累加进 (hi, lo)，不再需要 main/fold 的 fp32 临时缓冲
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dbetaWsOutQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dgammaWsOutQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dbetaLoOutQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dgammaLoOutQue_, BUFFER_NUM, aDimSize);
        // 补偿累加只需两个槽位：[0] hi（舍入后的和）、[1] lo（被舍掉的低位残差）
        pipe_->InitBuffer(dbetaCacheBuffer_, aDimSize * ACC_SLOT_NUM);
        pipe_->InitBuffer(dgammaCacheBuffer_, aDimSize * ACC_SLOT_NUM);
    }

    __aicore__ inline void CalcDbetaDgammaInCore()
    {
        for (uint32_t idx = 0; idx < tilingData_->aLoopTimes; idx++) {
            int64_t aLength = (idx == tilingData_->aLoopTimes - 1) ? tilingData_->aFactorTail : tilingData_->aFactor;
            dbetaWsTensor_ = dbetaWsOutQue_.AllocTensor<float>();
            dgammaWsTensor_ = dgammaWsOutQue_.AllocTensor<float>();
            dbetaLoTensor_ = dbetaLoOutQue_.AllocTensor<float>();
            dgammaLoTensor_ = dgammaLoOutQue_.AllocTensor<float>();
            dbetaCacheTensor_ = dbetaCacheBuffer_.Get<float>();
            dgammaCacheTensor_ = dgammaCacheBuffer_.Get<float>();
            int64_t baseOffset = idx * tilingData_->aFactor;
            LoadMeanRstdToUb(baseOffset, aLength);
            meanTensor_ = meanInQue_.template DeQue<float>();
            rstdTensor_ = rstdInQue_.template DeQue<float>();
            InitAccumulator(dbetaCacheTensor_, aLength);
            InitAccumulator(dgammaCacheTensor_, aLength);
            ProcessOneAFactor(baseOffset, aLength);
            meanInQue_.FreeTensor(meanTensor_);
            rstdInQue_.FreeTensor(rstdTensor_);
            FinalizeSlot(dbetaWsTensor_, dbetaCacheTensor_, aLength, 0);
            FinalizeSlot(dgammaWsTensor_, dgammaCacheTensor_, aLength, 0);
            FinalizeLoSlot(dbetaLoTensor_, dbetaCacheTensor_, aLength);
            FinalizeLoSlot(dgammaLoTensor_, dgammaCacheTensor_, aLength);
            dbetaWsOutQue_.EnQue(dbetaWsTensor_);
            dgammaWsOutQue_.EnQue(dgammaWsTensor_);
            dbetaLoOutQue_.EnQue(dbetaLoTensor_);
            dgammaLoOutQue_.EnQue(dgammaLoTensor_);
            int64_t wsOffset = baseOffset + blockIdx_ * tilingData_->aDim;
            StoreWsTwoLimbs(wsOffset, aLength);
        }
    }

    __aicore__ inline void ProcessOneAFactor(int64_t baseOffset, int64_t aLength)
    {
        int64_t foldCnt = binaryBlockCnt_ - binaryFoldPoint_;
        for (int64_t loopIdx = 0; loopIdx < binaryFoldPoint_; ++loopIdx) {
            int64_t mainOffset = baseOffset + loopIdx * currLoopFactor_ * tilingData_->aDim;
            // 只有main部分的场景
            LoadDyXToUb(mainOffset, currLoopFactor_, aLength, aFactorAlign_, tilingData_->aDim);
            LocalTensor<DY_TYPE> dyTensor = dyInQue_.DeQue<DY_TYPE>();
            LocalTensor<DY_TYPE> xTensor = xInQue_.DeQue<DY_TYPE>();
            ProcessMainBlock(currLoopFactor_, aLength, dyTensor, xTensor);
            dyInQue_.FreeTensor(dyTensor);
            xInQue_.FreeTensor(xTensor);
            if (loopIdx < foldCnt) {
                int64_t foldOffset = baseOffset + (loopIdx + binaryFoldPoint_) * currLoopFactor_ * tilingData_->aDim;
                int64_t rLength = (loopIdx == foldCnt - 1) ? binaryBlockTail_ : currLoopFactor_;
                LoadDyXToUb(foldOffset, rLength, aLength, aFactorAlign_, tilingData_->aDim);
                dyFoldTensor_ = dyInQue_.DeQue<DY_TYPE>();
                xFoldTensor_ = xInQue_.DeQue<DY_TYPE>();
                ProcessFoldBlock(rLength, aLength);
                dyInQue_.FreeTensor(dyFoldTensor_);
                xInQue_.FreeTensor(xFoldTensor_);
            }
        }
    }

    // 逐元素 Knuth TwoSum 累加：每个 VL 块把 (hi, lo) 留在寄存器里跨 r 行连续累加，
    // 整个块只回写一次。这样块内与块间都没有未记录的舍入（对齐 R0 模板的正确舍入范式）。
    __aicore__ inline void AccumulateDyExact(const LocalTensor<float>& accTensor, const LocalTensor<DY_TYPE>& dyTensor,
                                             uint32_t rLength, uint32_t aLength)
    {
        uint16_t loopTimes = CeilDiv(aLength, VL_FP32);
        __local_mem__ float* accAddr = (__local_mem__ float*)accTensor.GetPhyAddr();
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> hi, lo, val, scaleDownReg;
            MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(scaleDownReg, BNG_ACC_SCALE_DOWN, pregAll);
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                DataCopy(hi, accAddr + i * VL_FP32);
                DataCopy(lo, accAddr + aFactorAlign_ + i * VL_FP32);
                for (uint16_t j = 0; j < static_cast<uint16_t>(rLength); ++j) {
                    LoadOneTensor<DY_TYPE>(dyAddr, val, pMask, i * VL_FP32 + j * aFactorAlign_);
                    MicroAPI::Mul(val, val, scaleDownReg, pMask); // 2^-32 缩放域：累加器永不溢出卡死符号
                    AccumulateWithError(hi, lo, val, pMask);
                }
                DataCopy(accAddr + i * VL_FP32, hi, pMask);
                DataCopy(accAddr + aFactorAlign_ + i * VL_FP32, lo, pMask);
            }
        }
    }

    // Σ((x-mean)*rstd*dy)：rstd 在 dy 之前，逐元素 TwoSum。
    __aicore__ inline void AccumulateXSubMeanMulDyExact(const LocalTensor<float>& accTensor,
                                                        const LocalTensor<DY_TYPE>& xTensor,
                                                        const LocalTensor<DY_TYPE>& dyTensor, uint32_t rLength,
                                                        uint32_t aLength)
    {
        uint16_t loopTimes = CeilDiv(aLength, VL_FP32);
        __local_mem__ float* accAddr = (__local_mem__ float*)accTensor.GetPhyAddr();
        __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xTensor.GetPhyAddr();
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyTensor.GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor_.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor_.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> hi, lo, val, dyVal, meanReg, rstdReg;
            MicroAPI::RegTensor<float> termErr, errAcc, scaleDownReg;
            MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(scaleDownReg, BNG_ACC_SCALE_DOWN, pregAll);
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, i * VL_FP32);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * VL_FP32);
                DataCopy(hi, accAddr + i * VL_FP32);
                DataCopy(lo, accAddr + aFactorAlign_ + i * VL_FP32);
                if constexpr (IsSameType<DY_TYPE, float>::value) {
                    // fp32 输入：golden 元素项为 fp64 级，须元素项精确（误差肢独立累加，逐 i 块并入）
                    MicroAPI::Duplicate(errAcc, 0.0f, pregAll);
                    for (uint16_t j = 0; j < static_cast<uint16_t>(rLength); ++j) {
                        int32_t offset = i * VL_FP32 + j * aFactorAlign_;
                        LoadOneTensor<DY_TYPE>(xAddr, val, pMask, offset);
                        LoadOneTensor<DY_TYPE>(dyAddr, dyVal, pMask, offset);
                        BatchNorm3DGrad::ExactTermXSubMeanRstdMulDy(val, termErr, val, meanReg, rstdReg, dyVal, pMask);
                        MicroAPI::Mul(val, val, scaleDownReg, pMask); // 项与误差肢同步入 2^-32 缩放域
                        MicroAPI::Mul(termErr, termErr, scaleDownReg, pMask);
                        AccumulateWithError(hi, lo, val, pMask);
                        MicroAPI::Add<float, MicroAPI::MaskMergeMode::MERGING>(errAcc, errAcc, termErr, pMask);
                    }
                    BatchNorm3DGrad::ClampResidual(errAcc, pregAll); // 项含 inf 时误差肢为 NaN，钳 0 保持 inf 传播
                    AccumulateWithError(hi, lo, errAcc, pregAll);
                } else {
                    for (uint16_t j = 0; j < static_cast<uint16_t>(rLength); ++j) {
                        int32_t offset = i * VL_FP32 + j * aFactorAlign_;
                        LoadOneTensor<DY_TYPE>(xAddr, val, pMask, offset);
                        LoadOneTensor<DY_TYPE>(dyAddr, dyVal, pMask, offset);
                        MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(val, val, meanReg, pMask);
                        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(val, val, rstdReg, pMask);
                        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(val, val, dyVal, pMask);
                        MicroAPI::Mul(val, val, scaleDownReg, pMask); // 2^-32 缩放域
                        AccumulateWithError(hi, lo, val, pMask);
                    }
                }
                DataCopy(accAddr + i * VL_FP32, hi, pMask);
                DataCopy(accAddr + aFactorAlign_ + i * VL_FP32, lo, pMask);
            }
        }
    }

    __aicore__ inline void ProcessMainBlock(const int64_t rLength, const int64_t aLength,
                                            const LocalTensor<DY_TYPE> dyTensor, const LocalTensor<DY_TYPE> xTensor)
    {
        AccumulateDyExact(dbetaCacheTensor_, dyTensor, static_cast<uint32_t>(rLength), static_cast<uint32_t>(aLength));
        AccumulateXSubMeanMulDyExact(dgammaCacheTensor_, xTensor, dyTensor, static_cast<uint32_t>(rLength),
                                     static_cast<uint32_t>(aLength));
    }

    __aicore__ inline void ProcessFoldBlock(int64_t rLength, int64_t aLength)
    {
        // fold 块单独精确累加进 (hi, lo)，不再与 main 做逐元素 Add（那会引入一次未记录舍入）
        AccumulateDyExact(dbetaCacheTensor_, dyFoldTensor_, static_cast<uint32_t>(rLength),
                          static_cast<uint32_t>(aLength));
        AccumulateXSubMeanMulDyExact(dgammaCacheTensor_, xFoldTensor_, dyFoldTensor_, static_cast<uint32_t>(rLength),
                                     static_cast<uint32_t>(aLength));
    }

    // 归约累加器（hi, lo）清零：每个 A 分块重新开始
    __aicore__ inline void InitAccumulator(const LocalTensor<float>& accTensor, uint32_t aLength)
    {
        uint16_t loopTimes = CeilDiv(aLength, VL_FP32);
        __local_mem__ float* accAddr = (__local_mem__ float*)accTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> zeroReg;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                MicroAPI::Duplicate(zeroReg, 0.0f, pMask);
                DataCopy(accAddr + i * VL_FP32, zeroReg, pMask);
                DataCopy(accAddr + aFactorAlign_ + i * VL_FP32, zeroReg, pMask);
            }
        }
    }

    // (hi, lo) += value，lo 里保留每次加法的精确误差
    __aicore__ inline void AccumulateBlock(const LocalTensor<float>& accTensor, const LocalTensor<float>& valTensor,
                                           uint32_t aLength)
    {
        uint16_t loopTimes = CeilDiv(aLength, VL_FP32);
        __local_mem__ float* accAddr = (__local_mem__ float*)accTensor.GetPhyAddr();
        __local_mem__ float* valAddr = (__local_mem__ float*)valTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> hi, lo, val;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                DataCopy(hi, accAddr + i * VL_FP32);
                DataCopy(lo, accAddr + aFactorAlign_ + i * VL_FP32);
                DataCopy(val, valAddr + i * VL_FP32);
                AccumulateWithError(hi, lo, val, pMask);
                DataCopy(accAddr + i * VL_FP32, hi, pMask);
                DataCopy(accAddr + aFactorAlign_ + i * VL_FP32, lo, pMask);
            }
        }
    }

    // dst = hi + lo，整个归约只在这里做一次 fp32 舍入
    __aicore__ inline void FinalizeAccumulator(const LocalTensor<float>& dstTensor, const LocalTensor<float>& accTensor,
                                               uint32_t aLength)
    {
        uint16_t loopTimes = CeilDiv(aLength, VL_FP32);
        __local_mem__ float* accAddr = (__local_mem__ float*)accTensor.GetPhyAddr();
        __local_mem__ float* dstAddr = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> hi, lo;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                DataCopy(hi, accAddr + i * VL_FP32);
                DataCopy(lo, accAddr + aFactorAlign_ + i * VL_FP32);
                ClampResidual(lo, pMask);
                MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(hi, hi, lo, pMask);
                DataCopy(dstAddr + i * VL_FP32, hi, pMask);
            }
        }
    }

    // 只把指定槽位（0=hi, 1=lo）写出去，不做 hi+lo 合并：两肢都要进 workspace
    __aicore__ inline void FinalizeSlot(const LocalTensor<float>& dstTensor, const LocalTensor<float>& accTensor,
                                        uint32_t aLength, uint32_t slot)
    {
        uint16_t loopTimes = CeilDiv(aLength, VL_FP32);
        __local_mem__ float* accAddr = (__local_mem__ float*)accTensor.GetPhyAddr();
        __local_mem__ float* dstAddr = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> v;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                DataCopy(v, accAddr + slot * aFactorAlign_ + i * VL_FP32);
                DataCopy(dstAddr + i * VL_FP32, v, pMask);
            }
        }
    }

    // lo 肢写 workspace 前必须做非有限残差钳位：hi 溢出成 ±inf 时逐元素 TwoSum 的误差项会退化成 NaN，
    // 若把 NaN 写进 lo 平面，stage1 的跨核累加器会被 NaN 行污染（inf + NaN = NaN），
    // 把本该是 ±inf 的 dscale/doffset 变成 NaN。
    __aicore__ inline void FinalizeLoSlot(const LocalTensor<float>& dstTensor, const LocalTensor<float>& accTensor,
                                          uint32_t aLength)
    {
        uint16_t loopTimes = CeilDiv(aLength, VL_FP32);
        __local_mem__ float* accAddr = (__local_mem__ float*)accTensor.GetPhyAddr();
        __local_mem__ float* dstAddr = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> v;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                DataCopy(v, accAddr + aFactorAlign_ + i * VL_FP32);
                ClampResidual(v, pMask);
                DataCopy(dstAddr + i * VL_FP32, v, pMask);
            }
        }
    }

    // src 中 usedCoreNum 行（行步长 aFactorAlign_）按通道补偿归约成一行。
    // 行值是核内的 2^-32 缩放域部分和（核内累加器因此永不溢出卡死符号），
    // 合并后一次性按 2^32 放大回原域：真值超 fp32 域时在此以正确符号得 ±inf。
    __aicore__ inline void CompensatedRowSum(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                             uint32_t aLength)
    {
        uint16_t loopTimes = CeilDiv(aLength, VL_FP32);
        uint16_t rowCount = static_cast<uint16_t>(2 * tilingData_->usedCoreNum);
        __local_mem__ float* srcAddr = (__local_mem__ float*)srcTensor.GetPhyAddr();
        __local_mem__ float* dstAddr = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> hi, lo, val, upReg;
            MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(upReg, BNG_ACC_SCALE_UP, pregAll);
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                MicroAPI::Duplicate(hi, 0.0f, pMask);
                MicroAPI::Duplicate(lo, 0.0f, pMask);
                for (uint16_t r = 0; r < static_cast<uint16_t>(rowCount); ++r) {
                    DataCopy(val, srcAddr + r * aFactorAlign_ + i * VL_FP32);
                    AccumulateWithError(hi, lo, val, pMask);
                }
                ClampResidual(lo, pMask);
                MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(hi, hi, lo, pMask);
                MicroAPI::Mul(hi, hi, upReg, pMask); // 放大回原域；真溢出在此以正确符号得 ±inf
                DataCopy(dstAddr + i * VL_FP32, hi, pMask);
            }
        }
    }

    __aicore__ inline int64_t GetCacheID(const int64_t idx) { return ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1; }

    // stage1
    // dgamma 收尾：整通道部分和乘一次 rstd（等价于逐元素乘，但少 N 次舍入）
    __aicore__ inline void ScaleByRstd(const LocalTensor<float>& dgammaTensor, const LocalTensor<float>& rstdTensor,
                                       uint32_t aLength)
    {
        uint16_t loopTimes = CeilDiv(aLength, VL_FP32);
        __local_mem__ float* dgAddr = (__local_mem__ float*)dgammaTensor.GetPhyAddr();
        __local_mem__ float* rsAddr = (__local_mem__ float*)rstdTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> dg, rs;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                DataCopy(dg, dgAddr + i * VL_FP32);
                DataCopy(rs, rsAddr + i * VL_FP32);
                MicroAPI::Mul(dg, dg, rs, pMask);
                DataCopy(dgAddr + i * VL_FP32, dg, pMask);
            }
        }
    }

    __aicore__ inline void Stage1InitBuffer()
    {
        pipe_->Reset();

        int64_t aDimSize = 2 * aFactorAlign_ * tilingData_->usedCoreNum * sizeof(float);
        // 两肢 workspace 的落盘缓冲（hi 平面 + lo 平面），单缓冲即可：Load 完立即 DeQue 使用，
        // 双缓冲只会白占 UB。fp16 下 aFactorAlign_ 翻倍（128），72 核时双缓冲要 2*2*128*72*4=288KB，
        // 超过 ascend950 的 253952 B ub_size，会在运行期报 VEC_ERROR（L0_bound_032）。
        pipe_->InitBuffer(dbetaWsInQue_, 1, aDimSize);
        pipe_->InitBuffer(dgammaWsInQue_, 1, aDimSize);
        pipe_->InitBuffer(dbetaWsOutQue_, BUFFER_NUM, aFactorAlign_ * sizeof(float));
        pipe_->InitBuffer(dgammaWsOutQue_, BUFFER_NUM, aFactorAlign_ * sizeof(float));
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, aFactorAlign_ * sizeof(WEIGHT_TYPE));
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, aFactorAlign_ * sizeof(WEIGHT_TYPE));
    }

    __aicore__ inline void LoadDbetaDgammaFromWs(const int64_t offset, const int64_t rowSize, const uint32_t colSize,
                                                 const int64_t dstStride, const int64_t srcStride)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = rowSize;
        copyParams.blockLen = colSize * sizeof(float);
        copyParams.srcStride = (srcStride - colSize) * sizeof(float);
        copyParams.dstStride = (dstStride - colSize) * sizeof(float) / BLOCK_SIZE;
        DataCopyPadExtParams<float> PadParam{false, 0, 0, 0};

        LocalTensor<float> dbetaWsTensor = dbetaWsInQue_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaWsTensor, dbetaWorkSpace_[offset], copyParams, PadParam);
        dbetaWsInQue_.EnQue(dbetaWsTensor);

        LocalTensor<float> dgammaWsTensor = dgammaWsInQue_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaWsTensor, dgammaWorkSpace_[offset], copyParams, PadParam);
        dgammaWsInQue_.EnQue(dgammaWsTensor);
    }

    // stage0：每核把 (hi, lo) 两肢分别写到 [core] 与 [usedCoreNum + core] 两个平面
    __aicore__ inline void StoreWsTwoLimbs(const int64_t offset, const int64_t aLength)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = aLength * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        int64_t loOffset = offset + tilingData_->usedCoreNum * tilingData_->aDim;

        LocalTensor<float> dbetaHi = dbetaWsOutQue_.DeQue<float>();
        LocalTensor<float> dbetaLo = dbetaLoOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaWorkSpace_[offset], dbetaHi, copyParams);
        DataCopyPad<float, PaddingMode::Normal>(dbetaWorkSpace_[loOffset], dbetaLo, copyParams);
        dbetaWsOutQue_.FreeTensor(dbetaHi);
        dbetaLoOutQue_.FreeTensor(dbetaLo);

        LocalTensor<float> dgammaHi = dgammaWsOutQue_.DeQue<float>();
        LocalTensor<float> dgammaLo = dgammaLoOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaWorkSpace_[offset], dgammaHi, copyParams);
        DataCopyPad<float, PaddingMode::Normal>(dgammaWorkSpace_[loOffset], dgammaLo, copyParams);
        dgammaWsOutQue_.FreeTensor(dgammaHi);
        dgammaLoOutQue_.FreeTensor(dgammaLo);
    }

    // stage1：最终值只写 hi 槽位（dx 阶段按 rowSize=1 读取 baseOffset）
    __aicore__ inline void StoreWsFinal(const int64_t offset, const int64_t aLength)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = aLength * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        LocalTensor<float> dbetaWsTensor = dbetaWsOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaWorkSpace_[offset], dbetaWsTensor, copyParams);
        dbetaWsOutQue_.FreeTensor(dbetaWsTensor);

        LocalTensor<float> dgammaWsTensor = dgammaWsOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaWorkSpace_[offset], dgammaWsTensor, copyParams);
        dgammaWsOutQue_.FreeTensor(dgammaWsTensor);
    }

    __aicore__ inline void ProcessDbetaDgamma()
    {
        for (int64_t i = 0; i < tilingData_->aLoopTimes; i++) {
            int64_t baseOffset = i * tilingData_->aFactor;
            int64_t aLength = (i == tilingData_->aLoopTimes - 1) ? tilingData_->aFactorTail : tilingData_->aFactor;
            LoadDbetaDgammaFromWs(baseOffset, 2 * tilingData_->usedCoreNum, aLength, aFactorAlign_, tilingData_->aDim);
            LocalTensor<float> dbetaWsTensor = dbetaWsInQue_.template DeQue<float>();
            LocalTensor<float> dgammaWsTensor = dgammaWsInQue_.template DeQue<float>();
            LocalTensor<float> dbetaTmpTensor = dbetaWsOutQue_.AllocTensor<float>();
            LocalTensor<float> dgammaTmpTensor = dgammaWsOutQue_.AllocTensor<float>();
            // workspace空间上的dbeta, dgamma reduce成一行（补偿累加，避免核间求和再丢一次精度）
            CompensatedRowSum(dbetaTmpTensor, dbetaWsTensor, aLength);
            CompensatedRowSum(dgammaTmpTensor, dgammaWsTensor, aLength);
            dbetaWsInQue_.FreeTensor(dbetaWsTensor);
            dgammaWsInQue_.FreeTensor(dgammaWsTensor);
            // dgamma 在 stage0 已按元素乘过 rstd（与 golden/契约/三方乘序一致），
            // 此处不再整通道乘一次，避免二次舍入。
            LocalTensor<WEIGHT_TYPE> dbetaOutTensor = dbetaOutQue_.AllocTensor<WEIGHT_TYPE>();
            LocalTensor<WEIGHT_TYPE> dgammaOutTensor = dgammaOutQue_.AllocTensor<WEIGHT_TYPE>();
            if constexpr (IsSameType<WEIGHT_TYPE, float>::value) {
                DataCopy(dbetaOutTensor, dbetaTmpTensor, aFactorAlign_);
                DataCopy(dgammaOutTensor, dgammaTmpTensor, aFactorAlign_);
            } else {
                DbetaDgammaTypeConvers(aLength, dbetaTmpTensor, dgammaTmpTensor, dbetaOutTensor, dgammaOutTensor);
            }
            dbetaWsOutQue_.EnQue(dbetaTmpTensor);
            dgammaWsOutQue_.EnQue(dgammaTmpTensor);
            // 保存float32类型到dbeta, dgamma到workspace空间，计算dx时使用
            StoreWsFinal(baseOffset, aLength);
            dbetaOutQue_.EnQue(dbetaOutTensor);
            dgammaOutQue_.EnQue(dgammaOutTensor);
            StoreDbetaDgammaToGM(baseOffset, aLength);
        }
    }

    __aicore__ inline void DbetaDgammaTypeConvers(const uint32_t aLength, const LocalTensor<float> dbetaTmpTensor,
                                                  const LocalTensor<float> dgammaTmpTensor,
                                                  const LocalTensor<WEIGHT_TYPE> dbetaOutTensor,
                                                  const LocalTensor<WEIGHT_TYPE> dgammaOutTensor)
    {
        __local_mem__ float* dbetaTmpLocal = (__local_mem__ float*)dbetaTmpTensor.GetPhyAddr();
        __local_mem__ float* dgammaTmpLocal = (__local_mem__ float*)dgammaTmpTensor.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* dbetaOutLocal = (__local_mem__ WEIGHT_TYPE*)dbetaOutTensor.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* dgammaOutLocal = (__local_mem__ WEIGHT_TYPE*)dgammaOutTensor.GetPhyAddr();

        uint16_t loopNum = ops::CeilDiv(static_cast<uint32_t>(aLength), VL_FP32);

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = aLength;
            RegTensor<float> regDbeta, regDgamma;
            for (uint16_t i = 0; i < loopNum; i++) {
                pregMask = MicroAPI::UpdateMask<float>(sreg);
                int32_t offset = i * VL_FP32;
                LoadOneTensor<float>(dbetaTmpLocal, regDbeta, pregMask, offset);
                LoadOneTensor<float>(dgammaTmpLocal, regDgamma, pregMask, offset);
                StoreOneTensor<WEIGHT_TYPE>(dbetaOutLocal, regDbeta, pregMask, offset);
                StoreOneTensor<WEIGHT_TYPE>(dgammaOutLocal, regDgamma, pregMask, offset);
            }
        }
    }

    __aicore__ inline void StoreDbetaDgammaToGM(int64_t offset, uint32_t aLength)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = aLength * sizeof(WEIGHT_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        LocalTensor<WEIGHT_TYPE> dbetaOutTensor = dbetaOutQue_.DeQue<WEIGHT_TYPE>();
        DataCopyPad<WEIGHT_TYPE, PaddingMode::Normal>(dbetaGm_[offset], dbetaOutTensor, copyOutParams);
        dbetaOutQue_.FreeTensor(dbetaOutTensor);

        LocalTensor<WEIGHT_TYPE> dgammaOutTensor = dgammaOutQue_.DeQue<WEIGHT_TYPE>();
        DataCopyPad<WEIGHT_TYPE, PaddingMode::Normal>(dgammaGm_[offset], dgammaOutTensor, copyOutParams);
        dgammaOutQue_.FreeTensor(dgammaOutTensor);
    }

    // stage 2
    __aicore__ inline void Stage2InitBuffer()
    {
        pipe_->Reset();

        int64_t rDimSize = tilingData_->dxLoopFactor * aFactorAlign_;
        int64_t aDimSize = aFactorAlign_ * sizeof(float);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(dxOutQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(gammaInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dbetaWsInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dgammaWsInQue_, BUFFER_NUM, aDimSize);
    }

    __aicore__ inline void ProcessDX()
    {
        for (int64_t i = 0; i < tilingData_->aLoopTimes; i++) {
            int64_t baseOffset = i * tilingData_->aFactor;
            int64_t aLength = (i == tilingData_->aLoopTimes - 1) ? tilingData_->aFactorTail : tilingData_->aFactor;
            LoadMeanRstdToUb(baseOffset, aLength);
            meanTensor_ = meanInQue_.template DeQue<float>();
            rstdTensor_ = rstdInQue_.template DeQue<float>();
            LoadDbetaDgammaFromWs(baseOffset, 1, aLength, aFactorAlign_, aLength);
            LocalTensor<float> dbeta = dbetaWsInQue_.template DeQue<float>();
            LocalTensor<float> dgamma = dgammaWsInQue_.template DeQue<float>();
            LoadGamma(baseOffset, aLength);
            LocalTensor<WEIGHT_TYPE> gamma = gammaInQue_.template DeQue<WEIGHT_TYPE>();

            for (int64_t j = 0; j < dxLoopTimes_; j++) {
                int64_t offset = baseOffset + j * dxLoopFactor_ * tilingData_->aDim;
                int64_t rLength = (j == dxLoopTimes_ - 1) ? dxLoopTail_ : dxLoopFactor_;
                if (aLength <= EIGHT && aFactorAlign_ <= VL_FP32) {
                    CalDxVFSmallA(offset, rLength, aLength, dbeta, dgamma, gamma);
                } else {
                    CalDxVF(offset, rLength, aLength, dbeta, dgamma, gamma);
                }
                StoreDxToGM(offset, rLength, aLength, tilingData_->aDim, aFactorAlign_);
            }
            dbetaWsInQue_.FreeTensor(dbeta);
            dgammaWsInQue_.FreeTensor(dgamma);
            meanInQue_.FreeTensor(meanTensor_);
            rstdInQue_.FreeTensor(rstdTensor_);
            gammaInQue_.FreeTensor(gamma);
        }
    }

    __aicore__ inline void CalDxVF(const int64_t gmOffset, const uint16_t rLength, const uint16_t aLength,
                                   const LocalTensor<float>& dbetaTensor, const LocalTensor<float>& dgammaTensor,
                                   const LocalTensor<WEIGHT_TYPE>& gammaTensor)
    {
        LoadDyXToUb(gmOffset, rLength, aLength, aFactorAlign_, tilingData_->aDim);
        LocalTensor<DY_TYPE> dyTensor = dyInQue_.DeQue<DY_TYPE>();
        LocalTensor<DY_TYPE> xTensor = xInQue_.DeQue<DY_TYPE>();
        LocalTensor<DY_TYPE> dxTensor = dxOutQue_.template AllocTensor<DY_TYPE>();
        uint16_t outerLoopTimes = CeilDiv(aLength, VL_FP32);
        uint16_t innerLoopTimes = rLength;
        uint16_t outerLoopStride = VL_FP32;
        uint16_t innerLoopStride = aFactorAlign_;
        float countF = countF_;

        __VEC_SCOPE__
        {
            __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyTensor.GetPhyAddr();
            __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xTensor.GetPhyAddr();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor_.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor_.GetPhyAddr();
            __local_mem__ WEIGHT_TYPE* gammaAddr = (__local_mem__ WEIGHT_TYPE*)gammaTensor.GetPhyAddr();
            __local_mem__ DY_TYPE* dxAddr = (__local_mem__ DY_TYPE*)dxTensor.GetPhyAddr();
            __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbetaTensor.GetPhyAddr();
            __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgammaTensor.GetPhyAddr();
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> dyReg, xReg, meanReg, rstdReg, gammaReg, dxReg, dbetaReg, dgammaReg;
            MicroAPI::RegTensor<float> countReg, dbetaOverR, dgammaOverR;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerLoopStride);
                LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(dbetaAddr, dbetaReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(dgammaAddr, dgammaReg, pMask, i * outerLoopStride);
                // dbeta/R 用真除法（而非 fp32 倒数乘），每个 A 向量只算一次
                MicroAPI::Duplicate(countReg, countF, pMask);
                MicroAPI::Div(dbetaOverR, dbetaReg, countReg, pMask);
                if constexpr (IsSameType<DY_TYPE, float>::value) {
                    MicroAPI::Div(dgammaOverR, dgammaReg, countReg, pMask);
                }
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    uint32_t offset = i * outerLoopStride + j * innerLoopStride;
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, offset);
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    if constexpr (IsSameType<DY_TYPE, float>::value) {
                        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaOverR, pMask);
                    } else {
                        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaReg, pMask);
                        MicroAPI::Div<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, countReg, pMask);
                    }
                    // 逐项相减：dy - dbeta/R - (xhat*dgamma)/R，避免先在 1e7 量级合成再缩 R
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, dbetaOverR, pMask);
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dxReg, dyReg, gammaReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dxReg, dxReg, rstdReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dxReg, pMask, offset);
                }
            }
        }
        dyInQue_.FreeTensor(dyTensor);
        xInQue_.FreeTensor(xTensor);
        dxOutQue_.EnQue(dxTensor);
    }

    __aicore__ inline void CalDxVFSmallA(const int64_t gmOffset, const uint16_t rLength, const uint16_t aLength,
                                         const LocalTensor<float>& dbetaTensor, const LocalTensor<float>& dgammaTensor,
                                         const LocalTensor<WEIGHT_TYPE>& gammaTensor)
    {
        LoadDyXToUb(gmOffset, rLength, aLength, aFactorAlign_, tilingData_->aDim);
        LocalTensor<DY_TYPE> dyTensor = dyInQue_.DeQue<DY_TYPE>();
        LocalTensor<DY_TYPE> xTensor = xInQue_.DeQue<DY_TYPE>();
        LocalTensor<DY_TYPE> dxTensor = dxOutQue_.template AllocTensor<DY_TYPE>();

        // 基于aFactorAlign计算 R轴合并搬入的数量
        uint16_t mergeNum = VL_FP32 / aFactorAlign_;
        uint16_t loopTimes = CeilDiv(rLength, mergeNum);
        uint16_t loopStride = aFactorAlign_ * mergeNum;
        float countF = countF_;
        uint32_t meanOffset = 0;
        uint32_t rCount = rLength * aFactorAlign_;

        __VEC_SCOPE__
        {
            __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyTensor.GetPhyAddr();
            __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xTensor.GetPhyAddr();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor_.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor_.GetPhyAddr();
            __local_mem__ WEIGHT_TYPE* gammaAddr = (__local_mem__ WEIGHT_TYPE*)gammaTensor.GetPhyAddr();
            __local_mem__ DY_TYPE* dxAddr = (__local_mem__ DY_TYPE*)dxTensor.GetPhyAddr();
            __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbetaTensor.GetPhyAddr();
            __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgammaTensor.GetPhyAddr();
            MicroAPI::MaskReg pMask;
            MicroAPI::RegTensor<float> dyReg, xReg, meanReg, rstdReg, gammaReg, dxReg, dbetaReg, dgammaReg;
            MicroAPI::RegTensor<float> countReg, dbetaOverR, dgammaOverR;
            LoadOneTensorBrcVL(meanAddr, meanReg, meanOffset);
            LoadOneTensorBrcVL(rstdAddr, rstdReg, meanOffset);
            LoadOneTensorBrcVL(gammaAddr, gammaReg, meanOffset);
            LoadOneTensorBrcVL(dbetaAddr, dbetaReg, meanOffset);
            LoadOneTensorBrcVL(dgammaAddr, dgammaReg, meanOffset);
            // dbeta/R 用真除法（而非 fp32 倒数乘），每次调用只算一次
            uint32_t fullCount = rCount;
            pMask = MicroAPI::UpdateMask<float>(fullCount);
            MicroAPI::Duplicate(countReg, countF, pMask);
            MicroAPI::Div(dbetaOverR, dbetaReg, countReg, pMask);
            if constexpr (IsSameType<DY_TYPE, float>::value) {
                MicroAPI::Div(dgammaOverR, dgammaReg, countReg, pMask);
            }
            for (uint16_t i = 0; i < loopTimes; ++i) {
                uint32_t offset = i * loopStride;
                pMask = MicroAPI::UpdateMask<float>(rCount);
                LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, offset);
                LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, offset);
                MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                if constexpr (IsSameType<DY_TYPE, float>::value) {
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaOverR, pMask);
                } else {
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaReg, pMask);
                    MicroAPI::Div<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, countReg, pMask);
                }
                // 逐项相减：dy - dbeta/R - (xhat*dgamma)/R，避免先在 1e7 量级合成再缩 R
                MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, dbetaOverR, pMask);
                MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                if constexpr (IsSameType<DY_TYPE, float>::value) {
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dxReg, rstdReg, dyReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dxReg, dxReg, gammaReg, pMask);
                } else {
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dxReg, dyReg, gammaReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dxReg, dxReg, rstdReg, pMask);
                }
                StoreOneTensor<DY_TYPE>(dxAddr, dxReg, pMask, offset);
            }
        }
        dyInQue_.FreeTensor(dyTensor);
        xInQue_.FreeTensor(xTensor);
        dxOutQue_.EnQue(dxTensor);
    }

    __aicore__ inline void StoreDxToGM(const int64_t offset, const int64_t rowSize, const int64_t colSize,
                                       const int64_t dstStride, const int64_t srcStride)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = rowSize;
        copyParams.blockLen = colSize * sizeof(DY_TYPE);
        copyParams.srcStride = (srcStride - colSize) * sizeof(DY_TYPE) / BLOCK_SIZE;
        copyParams.dstStride = (dstStride - colSize) * sizeof(DY_TYPE);

        LocalTensor<DY_TYPE> dxTensor = dxOutQue_.template DeQue<DY_TYPE>();
        DataCopyPad<DY_TYPE, PaddingMode::Normal>(dxGm_[offset], dxTensor, copyParams);
        dxOutQue_.FreeTensor(dxTensor);
    }

    __aicore__ inline void LoadDyXToUb(const int64_t offset, const uint32_t rowSize, const uint32_t colSize,
                                       const int64_t dstStride, const int64_t srcStride)
    {
        DataCopyExtParams params;
        params.blockCount = rowSize;
        params.blockLen = colSize * sizeof(DY_TYPE);
        params.srcStride = (srcStride - colSize) * sizeof(DY_TYPE);
        params.dstStride = (dstStride - colSize) * sizeof(DY_TYPE) / BLOCK_SIZE;
        DataCopyPadExtParams<DY_TYPE> padParam{false, 0, 0, 0};

        LocalTensor<DY_TYPE> dyTensor = dyInQue_.template AllocTensor<DY_TYPE>();
        DataCopyPad(dyTensor, dyGm_[offset], params, padParam);
        dyInQue_.EnQue(dyTensor);

        LocalTensor<DY_TYPE> xTensor = xInQue_.template AllocTensor<DY_TYPE>();
        DataCopyPad(xTensor, xGm_[offset], params, padParam);
        xInQue_.EnQue(xTensor);
    }

    __aicore__ inline void LoadMeanRstdToUb(const int64_t offset, const uint32_t aLength)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = aLength * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<float> padParam{false, 0, 0, 0};

        meanTensor_ = meanInQue_.template AllocTensor<float>();
        DataCopyPad(meanTensor_, meanGm_[offset], copyParams, padParam);
        meanInQue_.EnQue(meanTensor_);

        rstdTensor_ = rstdInQue_.template AllocTensor<float>();
        DataCopyPad(rstdTensor_, rstdGm_[offset], copyParams, padParam);
        rstdInQue_.EnQue(rstdTensor_);
    }

    __aicore__ inline void LoadGamma(int64_t offset, uint32_t aLength)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = aLength * sizeof(WEIGHT_TYPE);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<WEIGHT_TYPE> PadParam{false, 0, 0, 0};
        LocalTensor<WEIGHT_TYPE> gammaTensor = gammaInQue_.template AllocTensor<WEIGHT_TYPE>();
        DataCopyPad(gammaTensor, gammaGm_[offset], copyParams, PadParam);
        gammaInQue_.EnQue(gammaTensor);
    }

private:
    const BatchNorm3DGradRASplitRTilingData* tilingData_{nullptr};
    TPipe* pipe_{nullptr};

    GlobalTensor<DY_TYPE> dyGm_, xGm_;
    GlobalTensor<float> meanGm_, rstdGm_;
    GlobalTensor<float> dbetaWorkSpace_, dgammaWorkSpace_; // workspace
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<DY_TYPE> dxGm_;
    LocalTensor<DY_TYPE> dyFoldTensor_, xFoldTensor_;
    LocalTensor<float> xMainTensor_, dyMainTensor_;
    LocalTensor<float> meanTensor_, rstdTensor_;
    LocalTensor<float> dbetaWsTensor_, dgammaWsTensor_;
    LocalTensor<float> dbetaLoTensor_, dgammaLoTensor_;
    LocalTensor<float> dbetaCacheTensor_, dgammaCacheTensor_;

    TQue<QuePosition::VECIN, 1> dyInQue_, xInQue_, meanInQue_, rstdInQue_;
    TBuf<TPosition::VECCALC> dyTmpQue_, xTmpQue_;
    TBuf<TPosition::VECCALC> dbetaCacheBuffer_, dgammaCacheBuffer_;
    TQue<QuePosition::VECOUT, 1> dbetaWsOutQue_, dgammaWsOutQue_;
    TQue<QuePosition::VECOUT, 1> dbetaLoOutQue_, dgammaLoOutQue_;
    TQue<QuePosition::VECIN, 1> dbetaWsInQue_, dgammaWsInQue_;
    TQue<QuePosition::VECOUT, 1> dbetaOutQue_, dgammaOutQue_;
    TQue<QuePosition::VECIN, 1> gammaInQue_;
    TQue<QuePosition::VECOUT, 1> dxOutQue_;

    static constexpr int64_t ULONG_BIT_LEN = 64;
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr uint32_t ACC_SLOT_NUM = 2; // hi/lo 两个补偿累加槽位
    static constexpr uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
    static constexpr int64_t EIGHT = 8;

    int64_t blockIdx_{0};
    int64_t currBlockFactor_{0};
    int64_t currLoopFactor_{0};
    int64_t binaryBlockCnt_{0};
    int64_t binaryFoldPoint_{0};
    int64_t binaryBlockTail_{0};
    int64_t dxLoopFactor_{0};
    int64_t dxLoopTail_{0};
    int64_t dxLoopTimes_{0};
    int64_t aFactorAlign_{0};
    float countF_{0.0f};
};
} // namespace BatchNorm3DGrad
#endif // __BATCH_NORM3D_GRAD_RA_SPLIT_R_REGBASE_H__
