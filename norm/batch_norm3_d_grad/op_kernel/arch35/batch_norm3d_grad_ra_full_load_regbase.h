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
 * \file batch_norm3d_grad_ra_full_load_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM3D_GRAD_RA_FULL_LOAD_REGBASE_H__
#define __BATCH_NORM3D_GRAD_RA_FULL_LOAD_REGBASE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm3d_grad_common.h"

namespace BatchNorm3DGrad {
using namespace AscendC;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MemType;

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1>
class BatchNorm3DGradRAFullLoad {
public:
    __aicore__ inline BatchNorm3DGradRAFullLoad(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(__gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* mean, __gm__ uint8_t* rstd,
                                __gm__ uint8_t* gamma, __gm__ uint8_t* dx, __gm__ uint8_t* dgamma,
                                __gm__ uint8_t* dbeta, __gm__ uint8_t* workspace,
                                const BatchNorm3DGradRAFullLoadTilingData* tilingData)
    {
        const BatchNorm3DGradBaseTilingData& baseTilingData = tilingData->baseTilingData;
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
        uint16_t outerLoop = (factor + 63) / 64;
        uint16_t innerLoop = rDim_;
        uint32_t outerStride = 64;
        uint32_t innerStride = stride;
        float countF = (float)rDim_;

        // dichotomy add
        uint16_t vfFoldLoopStep1 = foldLoopStep1;
        uint16_t vfFoldLoopStep2 = foldLoopStep2;
        uint16_t vfFoldLoopStep3 = foldLoopStep3;
        uint32_t vfFoldLoopOffset1 = foldLoopOffset1;
        uint32_t vfFoldLoopOffset2 = foldLoopOffset2;
        uint32_t vfFoldLoopOffset3 = foldLoopOffset3;
        uint32_t vfReduceLoopTimes = reduceLoopTimes;
        uint32_t vfReduceRecursionLoop = reduceRecursionLoop;

        if (rDim_ == ONE) {
            __VEC_SCOPE__
            {
                __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dy.GetPhyAddr();
                __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)x.GetPhyAddr();
                __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
                __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* gammaAddr = (__local_mem__ WEIGHT_TYPE*)gamma.GetPhyAddr();
                __local_mem__ float* dxAddr = (__local_mem__ float*)dx.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* dbetaAddr = (__local_mem__ WEIGHT_TYPE*)dbeta.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* dgammaAddr = (__local_mem__ WEIGHT_TYPE*)dgamma.GetPhyAddr();
                AscendC::MicroAPI::MaskReg pMask;
                uint32_t count = factor;

                AscendC::MicroAPI::RegTensor<float> dbetaReg, dgammaReg;
                AscendC::MicroAPI::RegTensor<float> countReg, dbetaOverR, dgammaOverR;
                AscendC::MicroAPI::RegTensor<float> xReg, dyReg;
                AscendC::MicroAPI::RegTensor<float> meanReg, rstdReg, gammaReg;
                for (uint16_t i = 0; i < outerLoop; ++i) {
                    pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                    LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerStride);
                    LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerStride);
                    LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride);
                    Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dbetaReg, dyReg, pMask);
                    StoreOneTensor<WEIGHT_TYPE>(dbetaAddr, dbetaReg, pMask, i * outerStride);
                    Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dgammaReg, xReg, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dgammaReg, dgammaReg, meanReg, pMask);
                    // 乘序与 golden/契约/三方一致：((x-mean)*rstd)*dy
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dgammaReg, dgammaReg, rstdReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dgammaReg, dgammaReg, dyReg, pMask);
                    StoreOneTensor<WEIGHT_TYPE>(dgammaAddr, dgammaReg, pMask, i * outerStride);

                    // dbeta/R、dgamma/R 用真除法逐项算，避免先在 1e7 量级合成再乘倒数
                    AscendC::MicroAPI::Duplicate(countReg, countF, pMask);
                    AscendC::MicroAPI::Div(dbetaOverR, dbetaReg, countReg, pMask);
                    AscendC::MicroAPI::Div(dgammaOverR, dgammaReg, countReg, pMask);
                    // dx = rstd * gamma * (dy - dbeta/R - xhat * (dgamma/R))
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, dbetaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    // 与 golden/契约/三方一致：(...)*scale*rstd，先乘 gamma 再乘 rstd
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, gammaReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, rstdReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dyReg, pMask, i * outerStride);
                }
            }
        } else if (rDim_ == TWO) {
            __VEC_SCOPE__
            {
                __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dy.GetPhyAddr();
                __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)x.GetPhyAddr();
                __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
                __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* gammaAddr = (__local_mem__ WEIGHT_TYPE*)gamma.GetPhyAddr();
                __local_mem__ float* dxAddr = (__local_mem__ float*)dx.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* dbetaAddr = (__local_mem__ WEIGHT_TYPE*)dbeta.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* dgammaAddr = (__local_mem__ WEIGHT_TYPE*)dgamma.GetPhyAddr();
                AscendC::MicroAPI::MaskReg pMask;
                uint32_t count = factor;

                AscendC::MicroAPI::RegTensor<float> dbetaReg, dgammaReg;
                AscendC::MicroAPI::RegTensor<float> countReg, dbetaOverR, dgammaOverR;
                AscendC::MicroAPI::RegTensor<float> xReg, dyReg;
                AscendC::MicroAPI::RegTensor<float> xReg2, dyReg2;
                AscendC::MicroAPI::RegTensor<float> meanReg, rstdReg, gammaReg;
                for (uint16_t i = 0; i < outerLoop; ++i) {
                    pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                    LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerStride);
                    LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerStride);
                    LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg2, pMask, i * outerStride + innerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg2, pMask, i * outerStride + innerStride);
                    // dbetaReg = dyReg + dyReg2
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dbetaReg, dyReg, dyReg2, pMask);
                    StoreOneTensor<WEIGHT_TYPE>(dbetaAddr, dbetaReg, pMask, i * outerStride);
                    // dgammaReg = (xReg - meanReg) * dyReg * rstdReg + (xReg2 - meanReg) * dyReg2 * rstdReg
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dyReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg2, xReg2, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg2, xReg2, dyReg2, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg2, xReg2, rstdReg, pMask);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dgammaReg, xReg, xReg2, pMask);
                    StoreOneTensor<WEIGHT_TYPE>(dgammaAddr, dgammaReg, pMask, i * outerStride);

                    // dbeta/R、dgamma/R 用真除法逐项算，避免先在 1e7 量级合成再乘倒数
                    AscendC::MicroAPI::Duplicate(countReg, countF, pMask);
                    AscendC::MicroAPI::Div(dbetaOverR, dbetaReg, countReg, pMask);
                    AscendC::MicroAPI::Div(dgammaOverR, dgammaReg, countReg, pMask);
                    // dx = rstd * gamma * (dy - dbeta/R - xhat * (dgamma/R))
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, dbetaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    // 与 golden/契约/三方一致：(...)*scale*rstd，先乘 gamma 再乘 rstd
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, gammaReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, rstdReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dyReg, pMask, i * outerStride);

                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride + innerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride + innerStride);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, dbetaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    // 与 golden/契约/三方一致：(...)*scale*rstd，先乘 gamma 再乘 rstd
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, gammaReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, rstdReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dyReg, pMask, i * outerStride + innerStride);
                }
            }
        } else if (rDim_ == THREE) {
            __VEC_SCOPE__
            {
                __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dy.GetPhyAddr();
                __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)x.GetPhyAddr();
                __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
                __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* gammaAddr = (__local_mem__ WEIGHT_TYPE*)gamma.GetPhyAddr();
                __local_mem__ float* dxAddr = (__local_mem__ float*)dx.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* dbetaAddr = (__local_mem__ WEIGHT_TYPE*)dbeta.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* dgammaAddr = (__local_mem__ WEIGHT_TYPE*)dgamma.GetPhyAddr();
                AscendC::MicroAPI::MaskReg pMask;
                uint32_t count = factor;

                AscendC::MicroAPI::RegTensor<float> dbetaReg, dgammaReg;
                AscendC::MicroAPI::RegTensor<float> countReg, dbetaOverR, dgammaOverR;
                AscendC::MicroAPI::RegTensor<float> xReg, dyReg;
                AscendC::MicroAPI::RegTensor<float> xReg2, dyReg2;
                AscendC::MicroAPI::RegTensor<float> xReg3, dyReg3;
                AscendC::MicroAPI::RegTensor<float> meanReg, rstdReg, gammaReg;
                for (uint16_t i = 0; i < outerLoop; ++i) {
                    pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                    LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerStride);
                    LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerStride);
                    LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg2, pMask, i * outerStride + innerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg2, pMask, i * outerStride + innerStride);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg3, pMask, i * outerStride + innerStride * 2);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg3, pMask, i * outerStride + innerStride * 2);
                    // dbetaReg = dyReg + dyReg2 + dyReg3
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dbetaReg, dyReg, dyReg2, pMask);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dbetaReg, dbetaReg, dyReg3, pMask);
                    StoreOneTensor<WEIGHT_TYPE>(dbetaAddr, dbetaReg, pMask, i * outerStride);
                    // dgammaReg = (xReg - meanReg) * dyReg * rstdReg
                    //           + (xReg2 - meanReg) * dyReg2 * rstdReg
                    //           + (xReg3 - meanReg) * dyReg3 * rstdReg
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dyReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg2, xReg2, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg2, xReg2, dyReg2, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg2, xReg2, rstdReg, pMask);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dgammaReg, xReg, xReg2, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg3, xReg3, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg3, xReg3, dyReg3, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg3, xReg3, rstdReg, pMask);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dgammaReg, dgammaReg, xReg3, pMask);
                    StoreOneTensor<WEIGHT_TYPE>(dgammaAddr, dgammaReg, pMask, i * outerStride);

                    // dbeta/R、dgamma/R 用真除法逐项算，避免先在 1e7 量级合成再乘倒数
                    AscendC::MicroAPI::Duplicate(countReg, countF, pMask);
                    AscendC::MicroAPI::Div(dbetaOverR, dbetaReg, countReg, pMask);
                    AscendC::MicroAPI::Div(dgammaOverR, dgammaReg, countReg, pMask);
                    // dx = rstd * gamma * (dy - dbeta/R - xhat * (dgamma/R))
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, dbetaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    // 与 golden/契约/三方一致：(...)*scale*rstd，先乘 gamma 再乘 rstd
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, gammaReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, rstdReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dyReg, pMask, i * outerStride);

                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride + innerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride + innerStride);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, dbetaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    // 与 golden/契约/三方一致：(...)*scale*rstd，先乘 gamma 再乘 rstd
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, gammaReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, rstdReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dyReg, pMask, i * outerStride + innerStride);

                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride + innerStride * 2);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride + innerStride * 2);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, dbetaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    // 与 golden/契约/三方一致：(...)*scale*rstd，先乘 gamma 再乘 rstd
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, gammaReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, rstdReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dyReg, pMask, i * outerStride + innerStride * 2);
                }
            }
        } else if (rDim_ == FOUR) {
            __VEC_SCOPE__
            {
                __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dy.GetPhyAddr();
                __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)x.GetPhyAddr();
                __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
                __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* gammaAddr = (__local_mem__ WEIGHT_TYPE*)gamma.GetPhyAddr();
                __local_mem__ float* dxAddr = (__local_mem__ float*)dx.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* dbetaAddr = (__local_mem__ WEIGHT_TYPE*)dbeta.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* dgammaAddr = (__local_mem__ WEIGHT_TYPE*)dgamma.GetPhyAddr();
                AscendC::MicroAPI::MaskReg pMask;
                uint32_t count = factor;

                AscendC::MicroAPI::RegTensor<float> dbetaReg, dgammaReg;
                AscendC::MicroAPI::RegTensor<float> countReg, dbetaOverR, dgammaOverR;
                AscendC::MicroAPI::RegTensor<float> xReg, dyReg;
                AscendC::MicroAPI::RegTensor<float> xReg2, dyReg2;
                AscendC::MicroAPI::RegTensor<float> xReg3, dyReg3;
                AscendC::MicroAPI::RegTensor<float> xReg4, dyReg4;
                AscendC::MicroAPI::RegTensor<float> meanReg, rstdReg, gammaReg;
                for (uint16_t i = 0; i < outerLoop; ++i) {
                    pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                    LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerStride);
                    LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerStride);
                    LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerStride);

                    // calc dbeta
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg3, pMask, i * outerStride + innerStride * 2);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, dyReg3, pMask);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg2, pMask, i * outerStride + innerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg4, pMask, i * outerStride + innerStride * 3);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg2, dyReg2, dyReg4, pMask);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dbetaReg, dyReg, dyReg2, pMask);
                    StoreOneTensor<WEIGHT_TYPE>(dbetaAddr, dbetaReg, pMask, i * outerStride);

                    // calc dgamma
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dyReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg3, pMask, i * outerStride + innerStride * 2);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg3, pMask, i * outerStride + innerStride * 2);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg3, xReg3, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg3, xReg3, dyReg3, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg3, xReg3, rstdReg, pMask);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, xReg3, pMask);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg2, pMask, i * outerStride + innerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg2, pMask, i * outerStride + innerStride);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg2, xReg2, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg2, xReg2, dyReg2, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg2, xReg2, rstdReg, pMask);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg4, pMask, i * outerStride + innerStride * 3);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg4, pMask, i * outerStride + innerStride * 3);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg4, xReg4, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg4, xReg4, dyReg4, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg4, xReg4, rstdReg, pMask);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg2, xReg2, xReg4, pMask);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dgammaReg, xReg, xReg2, pMask);
                    StoreOneTensor<WEIGHT_TYPE>(dgammaAddr, dgammaReg, pMask, i * outerStride);

                    // dbeta/R、dgamma/R 用真除法逐项算，避免先在 1e7 量级合成再乘倒数
                    AscendC::MicroAPI::Duplicate(countReg, countF, pMask);
                    AscendC::MicroAPI::Div(dbetaOverR, dbetaReg, countReg, pMask);
                    AscendC::MicroAPI::Div(dgammaOverR, dgammaReg, countReg, pMask);
                    // dx = rstd * gamma * (dy - dbeta/R - xhat * (dgamma/R))
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, dbetaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    // 与 golden/契约/三方一致：(...)*scale*rstd，先乘 gamma 再乘 rstd
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, gammaReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, rstdReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dyReg, pMask, i * outerStride);

                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride + innerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride + innerStride);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, dbetaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    // 与 golden/契约/三方一致：(...)*scale*rstd，先乘 gamma 再乘 rstd
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, gammaReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, rstdReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dyReg, pMask, i * outerStride + innerStride);

                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride + innerStride * 2);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride + innerStride * 2);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, dbetaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    // 与 golden/契约/三方一致：(...)*scale*rstd，先乘 gamma 再乘 rstd
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, gammaReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, rstdReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dyReg, pMask, i * outerStride + innerStride * 2);

                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride + innerStride * 3);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride + innerStride * 3);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, dbetaOverR, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    // 与 golden/契约/三方一致：(...)*scale*rstd，先乘 gamma 再乘 rstd
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, gammaReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, rstdReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dyReg, pMask, i * outerStride + innerStride * 3);
                }
            }
        } else {
            LocalTensor<float> dbetaws = dbetaBinaryAddBuf_.Get<float>();
            LocalTensor<float> dgammaws = dgammaBinaryAddBuf_.Get<float>();

            __VEC_SCOPE__
            {
                __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dy.GetPhyAddr();
                __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)x.GetPhyAddr();
                __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
                __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* gammaAddr = (__local_mem__ WEIGHT_TYPE*)gamma.GetPhyAddr();
                __local_mem__ float* dxAddr = (__local_mem__ float*)dx.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* dbetaAddr = (__local_mem__ WEIGHT_TYPE*)dbeta.GetPhyAddr();
                __local_mem__ WEIGHT_TYPE* dgammaAddr = (__local_mem__ WEIGHT_TYPE*)dgamma.GetPhyAddr();
                __local_mem__ float* dbetawsAddr = (__local_mem__ float*)dbetaws.GetPhyAddr();
                __local_mem__ float* dgammawsAddr = (__local_mem__ float*)dgammaws.GetPhyAddr();
                AscendC::MicroAPI::MaskReg pMask;
                uint32_t count = factor;

                AscendC::MicroAPI::RegTensor<float> dbetaReg, dgammaReg;
                AscendC::MicroAPI::RegTensor<float> countReg, dbetaOverR, dgammaOverR;
                AscendC::MicroAPI::RegTensor<float> xReg, yReg;
                AscendC::MicroAPI::RegTensor<float> meanReg, rstdReg, gammaReg;
                for (uint16_t i = 0; i < outerLoop; ++i) {
                    pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                    LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerStride);
                    LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerStride);
                    // lane 即 A 通道：跨 R 行用 VL 宽 (hi,lo) 直接累加，去掉 UB 多层树
                    AscendC::MicroAPI::RegTensor<float> hiD, loD, hiG, loG, valX;
                    AscendC::MicroAPI::Duplicate(hiD, 0.0f, pMask);
                    AscendC::MicroAPI::Duplicate(loD, 0.0f, pMask);
                    AscendC::MicroAPI::Duplicate(hiG, 0.0f, pMask);
                    AscendC::MicroAPI::Duplicate(loG, 0.0f, pMask);
                    for (uint16_t row = 0; row < innerLoop; ++row) {
                        uint32_t off = i * outerStride + row * innerStride;
                        LoadOneTensor<DY_TYPE>(dyAddr, yReg, pMask, off);
                        LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, off);
                        AccumulateWithError(hiD, loD, yReg, pMask);
                        Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(valX, xReg, meanReg, pMask);
                        Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(valX, valX, rstdReg, pMask);
                        Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(valX, valX, yReg, pMask);
                        AccumulateWithError(hiG, loG, valX, pMask);
                    }
                    ClampResidual(loD, pMask);
                    ClampResidual(loG, pMask);
                    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dbetaReg, hiD, loD, pMask);
                    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dgammaReg, hiG, loG,
                                                                                             pMask);
                    StoreOneTensor<WEIGHT_TYPE>(dbetaAddr, dbetaReg, pMask, i * outerStride);
                    StoreOneTensor<WEIGHT_TYPE>(dgammaAddr, dgammaReg, pMask, i * outerStride);

                    // dbeta/R、dgamma/R 用真除法逐项算，避免先在 1e7 量级合成再乘倒数
                    AscendC::MicroAPI::Duplicate(countReg, countF, pMask);
                    AscendC::MicroAPI::Div(dbetaOverR, dbetaReg, countReg, pMask);
                    AscendC::MicroAPI::Div(dgammaOverR, dgammaReg, countReg, pMask);
                    // dx = rstd * gamma * (dy - dbeta/R - xhat * (dgamma/R))
                    LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerStride);
                    for (uint16_t j = 0; j < innerLoop; ++j) {
                        LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride + j * innerStride);
                        Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                        Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                        Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaOverR, pMask);
                        LoadOneTensor<DY_TYPE>(dyAddr, yReg, pMask, i * outerStride + j * innerStride);
                        Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(yReg, yReg, dbetaOverR, pMask);
                        Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(yReg, yReg, xReg, pMask);
                        Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(yReg, yReg, rstdReg, pMask);
                        Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(yReg, yReg, gammaReg, pMask);
                        StoreOneTensor<DY_TYPE>(dxAddr, yReg, pMask, i * outerStride + j * innerStride);
                    }
                }
            }
        }
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
} // namespace BatchNorm3DGrad
#endif // __BATCH_NORM3D_GRAD_RA_FULL_LOAD_REGBASE_H__
