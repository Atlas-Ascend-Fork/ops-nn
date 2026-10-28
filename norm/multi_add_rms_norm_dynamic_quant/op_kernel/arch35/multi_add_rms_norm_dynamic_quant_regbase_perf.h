/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file multi_add_rms_norm_dynamic_quant_regbase_perf.h
 * \brief
 */
#ifndef MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_PERF_H_
#define MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_PERF_H_

#include "multi_add_rms_norm_dynamic_quant_regbase_common.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace MultiAddRmsNormDynamicQuant {
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

template <typename T_X, typename T_Y>
class KernelMultiAddRmsNormDynamicQuantRegbasePerf {
    using T_SMOOTH_SCALE = T_X;

public:
    __aicore__ inline KernelMultiAddRmsNormDynamicQuantRegbasePerf(TPipe* pipe) { pipe_ = pipe; }

    // 相对参照:去 beta 输入(multi_add 无 beta,hasBeta_ 恒 false);增 y 输出(RmsNorm 结果);x1 为 TensorList。
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooathScale1, GM_ADDR smooathScale2,
                                GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2,
                                const MultiAddRmsNormDynamicQuantRegbaseTilingData* tilingData)
    {
        InitTiling(numM_, numN_, baseM_, baseN_, baseNDtypeAlign_, baseNReduceAlign_, powerSplit_, mPerCore_,
                   mLastCore_, epsilon_, avgFactor_, tilingData);
        hasSmoothScale1_ = tilingData->hasSmoothScale1;
        hasSmoothScale2_ = tilingData->hasSmoothScale2;
        hasBeta_ = false; // multi_add 无 beta
        hasY2Scale2_ = hasSmoothScale2_;
        x1Num_ = tilingData->x1Num; // multi-add TensorList 长度(1~5)

        oriOverflowMode_ = GetOverflowMode<T_Y>();
        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();

        CalBlockTail();
        InitBuffer(x1, x2, gamma, smooathScale1, smooathScale2, y1, y2, x, y, scale1, scale2);
    }

    __aicore__ inline void CalBlockTail()
    {
        mCore_ = blockIdx_ == (blockNum_ - 1) ? mLastCore_ : mPerCore_;
        mOuterCnt_ = CeilDiv(mCore_, baseM_);
        tailMOuter_ = mCore_ - (mOuterCnt_ - 1) * baseM_;
        baseNB8Align_ = CeilAlign(baseN_, B8_BLOCK_NUM);
        baseNB32Align_ = CeilAlign(baseN_, B32_BLOCK_NUM);
    }

    __aicore__ inline void InitBuffer(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooathScale1,
                                      GM_ADDR smooathScale2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR y,
                                      GM_ADDR scale1, GM_ADDR scale2)
    {
        uint64_t gmOffset = blockIdx_ * mPerCore_ * numN_;
        uint64_t gmLen = mCore_ * numN_;
        uint64_t scalesGmOffset = blockIdx_ * mPerCore_;
        // x1 为 TensorList:逐个 tensor 取地址(参照 ascend910b kernel GetTensorAddr<T>(i, x1))
        for (uint32_t i = 0; i < x1Num_; ++i) {
            x1GmList_[i].SetGlobalBuffer(GetTensorAddr<T_X>(i, x1) + gmOffset, gmLen);
        }
        x2Gm_.SetGlobalBuffer((__gm__ T_X*)x2 + gmOffset, gmLen);
        gammaGm_.SetGlobalBuffer((__gm__ T_X*)gamma, numN_);
        y1Gm_.SetGlobalBuffer((__gm__ T_Y*)y1 + gmOffset, gmLen);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, gmLen);
        yGm_.SetGlobalBuffer((__gm__ T_X*)y + gmOffset, gmLen); // 增:y 输出(RmsNorm 结果)
        scale1Gm_.SetGlobalBuffer((__gm__ float*)scale1 + scalesGmOffset, mCore_);
        InitOptionalGmBuffers<T_X, T_Y, T_SMOOTH_SCALE>(smoothScale1Gm_, smoothScale2Gm_, y2Gm_, scale2Gm_, betaGm_,
                                                        smooathScale1, smooathScale2, y2, scale2, nullptr, gmOffset,
                                                        gmLen, scalesGmOffset, mCore_, numN_, hasSmoothScale1_,
                                                        hasSmoothScale2_, hasY2Scale2_, hasBeta_);

        InitUBBuffer();
    }

    __aicore__ inline void InitUBBuffer()
    {
        uint64_t ubFactorQuant = CeilAlign(numN_, BLOCK_SIZE / sizeof(T_SMOOTH_SCALE));
        uint64_t ubFactorRstd = CeilAlign(baseM_, B32_BLOCK_NUM);
        uint64_t firstVcaddResult = baseM_ * (((powerSplit_ + V_LENGTH - 1) / V_LENGTH + B32_BLOCK_NUM - 1) /
                                              B32_BLOCK_NUM * B32_BLOCK_NUM);
        pipe_->InitBuffer(inQueueX1_, 1, baseM_ * baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueX2_, 1, baseM_ * baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueX_, 1, baseM_ * baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueY_, DOUBLE_BUFFER, baseM_ * baseNDtypeAlign_ * sizeof(T_X)); // 增:y(RmsNorm 结果)输出
        pipe_->InitBuffer(x1AccBuf_,
                          baseM_ * baseNDtypeAlign_ * sizeof(float)); // 增:multi-add Σx1 累加缓冲(fp32,对齐A2)
        pipe_->InitBuffer(inQueueGamma_, 1, baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueY1_, DOUBLE_BUFFER, baseM_ * baseNB8Align_ * sizeof(T_Y));
        pipe_->InitBuffer(outQueueScale1_, DOUBLE_BUFFER, ubFactorRstd * sizeof(float));

        pipe_->InitBuffer(xOutTmpBuf_, baseM_ * baseNDtypeAlign_ * sizeof(float));
        pipe_->InitBuffer(y1TmpBuf_, baseM_ * baseNB32Align_ * sizeof(float));

        pipe_->InitBuffer(rstdBuf_, ubFactorRstd * sizeof(float));
        pipe_->InitBuffer(xReduceTmpBuf_, ubFactorRstd * sizeof(float));
        pipe_->InitBuffer(xTmpBuf_, firstVcaddResult * sizeof(float));
        if (hasSmoothScale1_) {
            pipe_->InitBuffer(inQueueSmoothScale1_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if (hasSmoothScale2_) {
            pipe_->InitBuffer(inQueueSmoothScale2_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if (hasY2Scale2_) {
            pipe_->InitBuffer(outQueueY2_, DOUBLE_BUFFER, baseM_ * baseNB8Align_ * sizeof(T_Y));
            pipe_->InitBuffer(outQueueScale2_, DOUBLE_BUFFER, ubFactorRstd * sizeof(float));
        }
        if (hasBeta_) {
            pipe_->InitBuffer(inQueueBeta_, 1, baseNDtypeAlign_ * sizeof(T_X));
        }
    }
    __aicore__ inline void Process()
    {
        CopyInParamToQueue(inQueueGamma_, gammaGm_, numN_);
        CopyInDynamicQuantCommon(inQueueSmoothScale1_, inQueueSmoothScale2_, smoothScale1Gm_, smoothScale2Gm_, numN_,
                                 hasSmoothScale1_, hasSmoothScale2_);
        LocalTensor<T_X> gammaLocal = inQueueGamma_.DeQue<T_X>();
        LocalTensor<T_SMOOTH_SCALE> smoothScale1Local;
        LocalTensor<T_SMOOTH_SCALE> smoothScale2Local;
        LocalTensor<T_X> betaLocal;
        PrepareOptionalParamLocals(inQueueSmoothScale1_, inQueueSmoothScale2_, inQueueBeta_, betaGm_, smoothScale1Local,
                                   smoothScale2Local, betaLocal, numN_, hasSmoothScale1_, hasSmoothScale2_, hasBeta_);

        for (uint64_t mOuterIdx = 0; mOuterIdx < mOuterCnt_; mOuterIdx++) {
            uint64_t realM = mOuterIdx == (mOuterCnt_ - 1) ? tailMOuter_ : baseM_;
            uint64_t mOuterOffset = mOuterIdx * baseM_;
            uint64_t gmOffset = mOuterOffset * baseN_;
            LocalTensor<float> scale1Local = outQueueScale1_.AllocTensor<float>();
            LocalTensor<float> scale2Local;
            if (hasY2Scale2_) {
                scale2Local = outQueueScale2_.AllocTensor<float>();
            }
            // 1.multi-add:CopyInXMutiMoveAlign 已按 A2 顺序累加出 x=(x1[0]+x2)+x1[1..](fp32)
            CopyInXMutiMoveAlign(gmOffset, realM);
            LocalTensor<float> xLocal1 = x1AccBuf_.Get<float>(); // x(fp32)
            LocalTensor<T_X> xOutLocal = outQueueX_.AllocTensor<T_X>();
            LocalTensor<float> xOutTmpLocal = xOutTmpBuf_.Get<float>();
            CalculateXAdd(xLocal1, xOutLocal, xOutTmpLocal, realM);
            outQueueX_.EnQue<T_X>(xOutLocal);
            CopyOutX(gmOffset, realM);

            // 2.二分累加计算SquareReduceSum、Rstd
            LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
            LocalTensor<float> xReduceLocal = xReduceTmpBuf_.Get<float>();
            NormCommon::NormCommonRegbase::CalculateSquareReduceSum<float>(
                xOutTmpLocal, xReduceLocal, xReduceTmpBuf_, static_cast<uint16_t>(realM),
                static_cast<uint32_t>(baseNDtypeAlign_), static_cast<uint32_t>(baseN_),
                static_cast<uint32_t>(powerSplit_), static_cast<uint32_t>(B32_BLOCK_NUM));
            NormCommon::ComputeRstdNewtonRaphson<true, true>(xReduceLocal, rstdLocal, realM, epsilon_, avgFactor_,
                                                             V_LENGTH);

            // y(RmsNorm 结果,量化前)= xOut * rstd * gamma,逐行(batch realM)计算并写回
            LocalTensor<T_X> yLocal = outQueueY_.AllocTensor<T_X>();
            ComputeYOutBatch(yLocal, xOutTmpLocal, rstdLocal, gammaLocal, realM);
            outQueueY_.EnQue<T_X>(yLocal);
            CopyOutYRms(gmOffset, realM);

            LocalTensor<T_Y> y1Local = outQueueY1_.AllocTensor<T_Y>();
            LocalTensor<float> y1TmpLocal = y1TmpBuf_.Get<float>();
            LocalTensor<T_Y> y2Local;
            if (hasY2Scale2_) {
                y2Local = outQueueY2_.AllocTensor<T_Y>();
            }
            SetOverflowMode<T_Y>(0);
            DispatchMutlScale<T_X, T_SMOOTH_SCALE, T_Y>(scale1Local, xOutTmpLocal, rstdLocal, gammaLocal, betaLocal,
                                                        smoothScale1Local, y1TmpLocal, baseN_, realM, baseN_,
                                                        baseNDtypeAlign_, hasSmoothScale1_, hasBeta_);
            PipeBarrier<PIPE_V>();
            ComputeMutlY<T_Y>(y1Local, scale1Local, y1TmpLocal, baseN_, realM);
            if (hasSmoothScale2_) {
                DispatchMutlScale<T_X, T_SMOOTH_SCALE, T_Y>(scale2Local, xOutTmpLocal, rstdLocal, gammaLocal, betaLocal,
                                                            smoothScale2Local, y1TmpLocal, baseN_, realM, baseN_,
                                                            baseNDtypeAlign_, hasSmoothScale2_, hasBeta_);
                PipeBarrier<PIPE_V>();
                ComputeMutlY<T_Y>(y2Local, scale2Local, y1TmpLocal, baseN_, realM);
            }
            SetOverflowMode<T_Y>(oriOverflowMode_);

            outQueueY1_.EnQue<T_Y>(y1Local);
            if (hasY2Scale2_) {
                outQueueY2_.EnQue<T_Y>(y2Local);
            }
            CopyOutY(gmOffset, realM);

            outQueueScale1_.EnQue<float>(scale1Local);
            if (hasY2Scale2_) {
                outQueueScale2_.EnQue<float>(scale2Local);
            }
            CopyOutScale(scale1Gm_, outQueueScale1_, mOuterOffset, realM);
            if (hasY2Scale2_) {
                CopyOutScale(scale2Gm_, outQueueScale2_, mOuterOffset, realM);
            }
        }
        inQueueGamma_.FreeTensor(gammaLocal);
        if (hasSmoothScale1_) {
            inQueueSmoothScale1_.FreeTensor(smoothScale1Local);
        }
        if (hasSmoothScale2_) {
            inQueueSmoothScale2_.FreeTensor(smoothScale2Local);
        }
        if (hasBeta_) {
            inQueueBeta_.FreeTensor(betaLocal);
        }
    }

private:
    // multi-add:批 realM 行,逐个 CopyIn x1[i](经 inQueueX1_ 完成 MTE2→V 同步)并累加到 x1AccBuf_,得 Σx1;x2 单独 CopyIn。
    __aicore__ inline void CopyInXMutiMoveAlign(uint64_t gmOffset, uint32_t realM)
    {
        DataCopyExtParams extParams{
            static_cast<uint16_t>(realM),                // blockCount
            static_cast<uint32_t>(baseN_ * sizeof(T_X)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(0),                    // dstStride
            0                                            // rsv
        };
        DataCopyPadExtParams<T_X> padParams{
            false,                   // isPad
            static_cast<uint8_t>(0), // leftPadding
            static_cast<uint8_t>(0), // rightPadding
            static_cast<T_X>(0.0)    // paddingValue
        };
        // 对齐 ascend910b(canndev normal/single_row_kernel):Σx1 全程 fp32 累加(逐个 Cast fp32 再 Add),
        // 避免 fp16 累加丢精度。acc=fp32(x1AccBuf_ 改 sizeof(float));cast 临时复用 CopyIn 期空闲的 xOutTmpBuf_(fp32)。
        // 累加次序必须与 ascend910b 一致:先 x1[0]+x2, 再逐个加 x1[1..]
        // (canndev normal_kernel CopyInX1X2AndAddX1X2「1.将x1的第一个tensor和x2相加 2.将x1剩余的相加」)。
        // fp32 加法不满足结合律, 写成 (Σx1)+x2 会在相消点产生 1~2 个 ULP 的偏差。
        LocalTensor<float> acc = x1AccBuf_.Get<float>();
        LocalTensor<float> accTmp = xOutTmpBuf_.Get<float>();
        uint32_t accCount = realM * baseNDtypeAlign_;
        LocalTensor<T_X> xLocal2 = inQueueX2_.AllocTensor<T_X>();
        DataCopyPad(xLocal2, x2Gm_[gmOffset], extParams, padParams);
        inQueueX2_.EnQue(xLocal2);
        for (uint32_t i = 0; i < x1Num_; ++i) {
            LocalTensor<T_X> x1Local = inQueueX1_.AllocTensor<T_X>();
            DataCopyPad(x1Local, x1GmList_[i][gmOffset], extParams, padParams);
            inQueueX1_.EnQue(x1Local);
            LocalTensor<T_X> x1In = inQueueX1_.DeQue<T_X>();
            if (i == 0) {
                Cast(acc, x1In, RoundMode::CAST_NONE, accCount); // acc = x1[0] (fp32)
                PipeBarrier<PIPE_V>();
                inQueueX1_.FreeTensor(x1In);
                LocalTensor<T_X> x2In = inQueueX2_.DeQue<T_X>();
                Cast(accTmp, x2In, RoundMode::CAST_NONE, accCount);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, accTmp, accCount); // acc = x1[0] + x2
                PipeBarrier<PIPE_V>();
                inQueueX2_.FreeTensor(x2In);
                continue;
            }
            Cast(accTmp, x1In, RoundMode::CAST_NONE, accCount);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, accTmp, accCount); // acc += x1[i] (fp32)
            PipeBarrier<PIPE_V>();
            inQueueX1_.FreeTensor(x1In);
        }
    }

    __aicore__ inline void CopyOutX(uint64_t offset, uint32_t realM)
    {
        LocalTensor<T_X> xLocal = outQueueX_.DeQue<T_X>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(realM),                // blockCount
            static_cast<uint32_t>(baseN_ * sizeof(T_X)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(0),                    // dstStride
            0                                            // rsv
        };
        DataCopyPad(xGm_[offset], xLocal, copyParams);
        outQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutY(uint64_t offset, uint32_t realM)
    {
        LocalTensor<T_Y> y1Local = outQueueY1_.DeQue<T_Y>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(realM),                // blockCount
            static_cast<uint32_t>(baseN_ * sizeof(T_Y)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(0),                    // dstStride
            0                                            // rsv
        };
        DataCopyPad(y1Gm_[offset], y1Local, copyParams);
        outQueueY1_.FreeTensor(y1Local);
        if (hasY2Scale2_) {
            LocalTensor<T_Y> y2Local = outQueueY2_.DeQue<T_Y>();
            DataCopyPad(y2Gm_[offset], y2Local, copyParams);
            outQueueY2_.FreeTensor(y2Local);
        }
    }

    // 增:y(RmsNorm 结果,量化前,dtype T_X)批 realM 行写回 yGm_。
    __aicore__ inline void CopyOutYRms(uint64_t offset, uint32_t realM)
    {
        LocalTensor<T_X> yLocal = outQueueY_.DeQue<T_X>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(realM),                // blockCount
            static_cast<uint32_t>(baseN_ * sizeof(T_X)), // blockLen
            static_cast<uint32_t>(0),                    // srcStride
            static_cast<uint32_t>(0),                    // dstStride
            0                                            // rsv
        };
        DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

    // 增:逐行(batch realM)调用共享 ComputeYOut,y = xOut * rstd * gamma。
    // xOutTmpLocal 为 fp32 的 xOut(=Σx1+x2),每行占 baseNDtypeAlign_;rstd 每行一个标量。
    __aicore__ inline void ComputeYOutBatch(LocalTensor<T_X>& yLocal, LocalTensor<float>& xOutTmpLocal,
                                            LocalTensor<float>& rstdLocal, LocalTensor<T_X>& gammaLocal, uint32_t realM)
    {
        for (uint32_t curA = 0; curA < realM; ++curA) {
            LocalTensor<T_X> ySub = yLocal[curA * baseNDtypeAlign_];
            LocalTensor<float> xSub = xOutTmpLocal[curA * baseNDtypeAlign_];
            ComputeYOut<T_X, T_X>(ySub, xSub, rstdLocal, gammaLocal, curA, static_cast<uint32_t>(baseN_));
        }
    }

    // x 已在 CopyInXMutiMoveAlign 内按 A2 顺序累加完成(fp32), 此处只做窄化输出与 fp32 留存。
    __aicore__ inline void CalculateXAdd(LocalTensor<float>& xLocal1, LocalTensor<T_X>& xOutLocal,
                                         LocalTensor<float>& xOutTmpLocal, uint32_t realM)
    {
        __local_mem__ float* x1InUb = (__local_mem__ float*)xLocal1.GetPhyAddr(); // x 已是 fp32
        __local_mem__ T_X* xOutInUb = (__local_mem__ T_X*)xOutLocal.GetPhyAddr();
        __local_mem__ float* xOutTmp = (__local_mem__ float*)xOutTmpLocal.GetPhyAddr();

        uint32_t sreg = realM * baseNDtypeAlign_;
        uint16_t loopCount = (sreg + V_LENGTH - 1) / V_LENGTH;

        __VEC_SCOPE__
        {
            RegTensor<float> xSum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * V_LENGTH;
                pregLoop = UpdateMask<float>(sreg);
                LoadRegForDtype<float>(x1InUb, xSum, pregLoop, offset); // x fp32 直接 load
                StoreRegForDtype<T_X>(xOutInUb, xSum, pregLoop, offset);
                DataCopy<float, StoreDist::DIST_NORM_B32>(xOutTmp + offset, xSum, pregLoop);
            }
        }
    }

    template <typename T_XPF32, typename T_GAMMA, typename T_SMOOTHSCALE = float, bool HAS_SMOOTH_SCALE = true,
              bool HAS_BETA = false, typename T_YB8>
    __aicore__ inline void ComputeMutlScale(LocalTensor<float>& scaleLocal, LocalTensor<T_XPF32>& xLocal,
                                            LocalTensor<float>& rstdLocal, LocalTensor<T_GAMMA>& gammaLocal,
                                            LocalTensor<T_GAMMA>& betaLocal,
                                            LocalTensor<T_SMOOTHSCALE>& smoothScaleLocal, LocalTensor<float>& yTmpLocal,
                                            uint32_t calCount, uint32_t realM, uint64_t baseN, uint64_t baseNDtypeAlign)
    {
        uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);
        uint16_t curAloops = static_cast<uint16_t>(realM);
        __local_mem__ T_XPF32* xAddr = (__ubuf__ T_XPF32*)xLocal.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
        __local_mem__ T_GAMMA* betaAddr;
        if constexpr (HAS_BETA) {
            betaAddr = (__ubuf__ T_GAMMA*)betaLocal.GetPhyAddr();
        }
        __local_mem__ T_SMOOTHSCALE* smoothScaleAddr;
        if constexpr (HAS_SMOOTH_SCALE) {
            smoothScaleAddr = (__ubuf__ T_SMOOTHSCALE*)smoothScaleLocal.GetPhyAddr();
        }
        __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
        __local_mem__ float* yTmpAddr = (__ubuf__ float*)yTmpLocal.GetPhyAddr();

        __VEC_SCOPE__
        {
            // VF0. Calc scale
            RegTensor<float> rstdReg, scaleReg;
            RegTensor<float> xRegFp32, yRegFp32, gammaRegFp32, betaRegFp32, smoothScaleRegFp32;
            MaskReg maskRegFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg maskRegOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg maskReg;

            for (uint16_t curA = 0; curA < curAloops; curA++) {
                Duplicate(scaleReg, static_cast<float>(-INFINITY), maskRegFull); // Abs before reducemax, scaleReg >= 0
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + static_cast<uint32_t>(curA));
                uint32_t sregElewiseNum = calCount;
                for (uint16_t idx = 0; idx < (uint16_t)repeatTimes; idx++) {
                    maskReg = UpdateMask<float>(sregElewiseNum);
                    NormCommon::LoadCastRegVF(xRegFp32, xAddr + curA * baseNDtypeAlign, idx, maskReg);
                    NormCommon::LoadCastRegVF(gammaRegFp32, gammaAddr, idx, maskReg);
                    if constexpr (HAS_BETA) {
                        NormCommon::LoadCastRegVF(betaRegFp32, betaAddr, idx, maskReg);
                    }
                    if constexpr (HAS_SMOOTH_SCALE) {
                        NormCommon::LoadCastRegVF(smoothScaleRegFp32, smoothScaleAddr, idx, maskReg);
                    }
                    Mul(xRegFp32, xRegFp32, rstdReg, maskReg);
                    Mul(xRegFp32, xRegFp32, gammaRegFp32, maskReg);
                    if constexpr (HAS_BETA) {
                        Add(xRegFp32, xRegFp32, betaRegFp32, maskReg);
                    }
                    if constexpr (HAS_SMOOTH_SCALE) {
                        // 物化 inp = y*smooth_scale(与 A2 的 ScaleTensor 同形态)。不走"只存 y、
                        // 由 ComputeMutlY 合成 k=(ss*qmax)/row_max"的尺度不变写法: k 是该写法
                        // 自造的中间量, row_max 落次正规区而 ss 为正常量级时会越过 fp32 上限
                        // 变 inf, 输出整体饱和(实测 ss~0.5、row_max~2e-38 -> k~3.2e39)。
                        // inp/scale 恒不溢出(|inp| <= row_max), 次正规分母的 DAZ 由 pre 同乘化解。
                        Mul(yRegFp32, xRegFp32, smoothScaleRegFp32, maskReg);
                        DataCopy<float>(yTmpAddr + curA * baseNB32Align_ + idx * V_LENGTH, yRegFp32, maskReg);
                        Abs(yRegFp32, yRegFp32, maskReg);               // VF abs is zeroing mode
                        Max(scaleReg, scaleReg, yRegFp32, maskRegFull); // Using full mask
                    } else {
                        DataCopy<float>(yTmpAddr + curA * baseNB32Align_ + idx * V_LENGTH, xRegFp32, maskReg);
                        Abs(yRegFp32, xRegFp32, maskReg);               // VF abs is zeroing mode
                        Max(scaleReg, scaleReg, yRegFp32, maskRegFull); // Using full mask
                    }
                }
                ReduceMax(scaleReg, scaleReg, maskRegFull);
                // 存入 UB 的是 row_max(正规数)而非 row_max/127(可能次正规)。
                // 矢量单元对次正规做 DAZ, 次正规值一旦落 UB, 重新加载时即被刷成 0,
                // 导致后续 Div 退化为除零、量化输出饱和。量化真实除数 scale=max/127
                // 在 ComputeMutlY 内由 max 还原后写回, GM 输出语义不变。
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + curA, scaleReg, maskRegOne);
            }
        }
    }

    template <typename T_YB8>
    __aicore__ inline void ComputeMutlY(LocalTensor<T_YB8>& yLocal, LocalTensor<float>& scaleLocal,
                                        LocalTensor<float>& yTmpLocal, uint32_t calCount, uint32_t realM)
    {
        uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);
        uint16_t curAloops = static_cast<uint16_t>(realM);

        __local_mem__ T_YB8* yAddr = (__ubuf__ T_YB8*)yLocal.GetPhyAddr();
        __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
        __local_mem__ float* yTmpAddr = (__ubuf__ float*)yTmpLocal.GetPhyAddr();

        __VEC_SCOPE__
        {
            // VF1. Calc y
            RegTensor<float> yRegFp32, yRegFp32Tmp, scaleReg;
            RegTensor<half> yRegFp16;
            RegTensor<T_YB8> yReg;
            MaskReg maskReg;
            // row_max 落 fp32 次正规区时, 矢量 Div 的分母会被 DAZ 当 0 -> 商 inf -> 输出整体饱和。
            // 用 PRE_SCALE(2 的幂, 精确)把分子分母**同时**搬正: (inp*PRE)/((row_max*PRE)/qmax)。
            // 分子恒不溢出 —— |inp| <= row_max, 故与分母同界。
            constexpr float SUBNORM_GUARD = 1.0e-30f;
            constexpr float PRE_SCALE = 1208925819614629174706176.0f; // 2^80
            float quantMaxV = static_cast<float>(1.0) / DIV_FACTOR_INT8;
            if constexpr (IsSameType<T_YB8, fp8_e4m3fn_t>::value) {
                quantMaxV = static_cast<float>(1.0) / DIV_FACTOR_FP8E4M3FN;
            } else if constexpr (IsSameType<T_YB8, fp8_e5m2_t>::value) {
                quantMaxV = static_cast<float>(1.0) / DIV_FACTOR_FP8E5M2;
            } else if constexpr (IsSameType<T_YB8, hifloat8_t>::value) {
                quantMaxV = static_cast<float>(1.0) / DIV_FACTOR_HIFP8;
            }
            RegTensor<float> preReg, rmScaled, denReg, bigReg, oneReg, qmaxReg;
            MaskReg maskRegAll = CreateMask<float, MaskPattern::ALL>();
            MaskReg maskRegOneY = CreateMask<float, MaskPattern::VL1>();
            Duplicate(bigReg, PRE_SCALE, maskRegAll);
            Duplicate(oneReg, static_cast<float>(1.0), maskRegAll);
            Duplicate(qmaxReg, quantMaxV, maskRegAll);
            for (uint16_t curA = 0; curA < curAloops; curA++) {
                uint32_t sregElewiseNum = calCount;
                DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, scaleAddr + curA); // 此处为 row_max
                MaskReg isSmall;
                CompareScalar<float, CMPMODE::LT>(isSmall, scaleReg, SUBNORM_GUARD, maskRegAll);
                Select(preReg, bigReg, oneReg, isSmall); // isSmall 真取 PRE_SCALE, 否则 1.0
                Mul(rmScaled, scaleReg, preReg, maskRegAll);
                // 分母 = (row_max*PRE)/qmax = scale*PRE
                Div(rmScaled, rmScaled, qmaxReg, maskRegAll);
                // row_max==0(整行 inp 恒为 0): 分母换 1, out = 0/1 = 0, 规避 0/0; 与算法公式一致
                MaskReg isZero;
                CompareScalar<float, CMPMODE::EQ>(isZero, rmScaled, static_cast<float>(0.0), maskRegAll);
                Select(denReg, oneReg, rmScaled, isZero);
                for (uint16_t idx = 0; idx < (uint16_t)repeatTimes; idx++) {
                    maskReg = UpdateMask<float>(sregElewiseNum);
                    DataCopy<float>(yRegFp32, yTmpAddr + curA * baseNB32Align_ + idx * V_LENGTH);
                    Mul(yRegFp32, yRegFp32, preReg, maskReg);
                    Div(yRegFp32, yRegFp32, denReg, maskReg);
                    if constexpr (IsSameType<T_YB8, int8_t>::value) {
                        Truncate<float, RoundMode::CAST_RINT>(yRegFp32Tmp, yRegFp32, maskReg);
                        Cast<half, float, castTraitFp322Fp16>(yRegFp16, yRegFp32Tmp, maskReg);
                        Cast<T_YB8, half, castTraitFp162Int8>(yReg, yRegFp16, maskReg);
                    } else if constexpr (IsSameType<T_YB8, fp8_e4m3fn_t>::value ||
                                         IsSameType<T_YB8, fp8_e5m2_t>::value) {
                        Cast<T_YB8, float, castTraitFp322Fp8>(yReg, yRegFp32, maskReg);
                    } else if constexpr (IsSameType<T_YB8, hifloat8_t>::value) {
                        Cast<T_YB8, float, castTraitFp322Hifp8>(yReg, yRegFp32, maskReg);
                    }
                    DataCopy<T_YB8, StoreDist::DIST_PACK4_B32>(yAddr + curA * baseNB8Align_ + idx * V_LENGTH, yReg,
                                                               maskReg);
                }
                // 还原真实 scale = row_max/quantMax 写回 UB。取 Div 形态 —— 与上面的分母同一个值,
                // 也与算法公式 scale = row_max(abs(inp))/127 字面一致。
                Div(scaleReg, scaleReg, qmaxReg, maskRegOneY);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + curA, scaleReg, maskRegOneY);
            }
        }
    }

    template <typename T_GAMMA, typename T_SMOOTHSCALE, typename T_YB8>
    __aicore__ inline void DispatchMutlScale(LocalTensor<float>& scaleLocal, LocalTensor<float>& xLocal,
                                             LocalTensor<float>& rstdLocal, LocalTensor<T_GAMMA>& gammaLocal,
                                             LocalTensor<T_GAMMA>& betaLocal,
                                             LocalTensor<T_SMOOTHSCALE>& smoothScaleLocal,
                                             LocalTensor<float>& yTmpLocal, uint32_t calCount, uint32_t realM,
                                             uint64_t baseN, uint64_t baseNDtypeAlign, bool hasSmoothScale,
                                             bool hasBeta)
    {
        if (hasSmoothScale) {
            if (hasBeta) {
                ComputeMutlScale<float, T_GAMMA, T_SMOOTHSCALE, true, true, T_YB8>(
                    scaleLocal, xLocal, rstdLocal, gammaLocal, betaLocal, smoothScaleLocal, yTmpLocal, calCount, realM,
                    baseN, baseNDtypeAlign);
            } else {
                ComputeMutlScale<float, T_GAMMA, T_SMOOTHSCALE, true, false, T_YB8>(
                    scaleLocal, xLocal, rstdLocal, gammaLocal, betaLocal, smoothScaleLocal, yTmpLocal, calCount, realM,
                    baseN, baseNDtypeAlign);
            }
        } else {
            if (hasBeta) {
                ComputeMutlScale<float, T_GAMMA, T_SMOOTHSCALE, false, true, T_YB8>(
                    scaleLocal, xLocal, rstdLocal, gammaLocal, betaLocal, smoothScaleLocal, yTmpLocal, calCount, realM,
                    baseN, baseNDtypeAlign);
            } else {
                ComputeMutlScale<float, T_GAMMA, T_SMOOTHSCALE, false, false, T_YB8>(
                    scaleLocal, xLocal, rstdLocal, gammaLocal, betaLocal, smoothScaleLocal, yTmpLocal, calCount, realM,
                    baseN, baseNDtypeAlign);
            }
        }
    }

private:
    TPipe* pipe_ = nullptr;
    // GM Buffer
    static constexpr uint32_t MAX_X1_NUM = 5; // x1 TensorList 最大长度(0~4 个 Add → 1~5 个 tensor)
    GlobalTensor<T_X> x1GmList_[MAX_X1_NUM];  // multi-add:x1 列表各 tensor
    GlobalTensor<T_X> x2Gm_;
    GlobalTensor<T_X> gammaGm_;
    GlobalTensor<T_X> betaGm_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_X> yGm_; // 增:y 输出(RmsNorm 结果)
    GlobalTensor<T_SMOOTH_SCALE> smoothScale1Gm_;
    GlobalTensor<T_SMOOTH_SCALE> smoothScale2Gm_;
    GlobalTensor<T_Y> y1Gm_;
    GlobalTensor<T_Y> y2Gm_;
    GlobalTensor<float> scale1Gm_;
    GlobalTensor<float> scale2Gm_;
    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX1_;
    TQue<QuePosition::VECIN, 1> inQueueX2_;
    TQue<QuePosition::VECIN, 1> inQueueGamma_;
    TQue<QuePosition::VECIN, 1> inQueueBeta_;
    TQue<QuePosition::VECIN, 1> inQueueSmoothScale1_;
    TQue<QuePosition::VECIN, 1> inQueueSmoothScale2_;
    TQue<QuePosition::VECOUT, 1> outQueueY1_;
    TQue<QuePosition::VECOUT, 1> outQueueY2_;
    TQue<QuePosition::VECOUT, 1> outQueueY_; // 增:y(RmsNorm 结果)输出
    TQue<QuePosition::VECOUT, 1> outQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueScale1_;
    TQue<QuePosition::VECOUT, 1> outQueueScale2_;
    TBuf<TPosition::VECCALC> rstdBuf_;
    TBuf<TPosition::VECCALC> y1TmpBuf_;
    TBuf<TPosition::VECCALC> xOutTmpBuf_;
    TBuf<TPosition::VECCALC> xReduceTmpBuf_;
    TBuf<TPosition::VECCALC> xTmpBuf_;
    TBuf<TPosition::VECCALC> x1AccBuf_; // 增:multi-add Σx1 累加缓冲

    // Tiling data
    uint64_t numN_{0};
    uint64_t numM_{0};
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t baseNDtypeAlign_{0};
    uint64_t baseNReduceAlign_{0};
    uint64_t baseNB32Align_{0};
    uint64_t powerSplit_{0};
    uint64_t mPerCore_{0};
    uint64_t mLastCore_{0};
    float epsilon_{0};
    float avgFactor_{0};
    uint32_t x1Num_{1}; // multi-add TensorList 长度(1~5)
    bool hasSmoothScale1_{false};
    bool hasSmoothScale2_{false};
    bool hasBeta_{false};
    bool hasY2Scale2_{false};
    // Platform
    int64_t blockIdx_{0};
    int64_t blockNum_{0};
    int64_t oriOverflowMode_{0};
    uint64_t mCore_;
    uint64_t mOuterCnt_;
    uint64_t tailMOuter_;
    uint64_t baseNB8Align_;
    // Other
};
} // namespace MultiAddRmsNormDynamicQuant
#endif // MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_PERF_H_
