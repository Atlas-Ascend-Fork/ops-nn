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
 * \file batch_norm_grad_ext2_full_load_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_EXT2_FULL_LOAD_REGBASE_H__
#define __BATCH_NORM_GRAD_EXT2_FULL_LOAD_REGBASE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm_grad_ext2_common.h"
#include "batch_norm_grad_ext2_comp_sum.h"

namespace BatchNormGradExt2 {
using namespace AscendC;
using namespace BatchNormGradExt2::Comp;

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1>
class BatchNormGradExt2RARFullLoad {
public:
    __aicore__ inline BatchNormGradExt2RARFullLoad(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(__gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* mean, __gm__ uint8_t* rstd,
                                __gm__ uint8_t* gamma, __gm__ uint8_t* dx, __gm__ uint8_t* dgamma,
                                __gm__ uint8_t* dbeta, __gm__ uint8_t* workspace,
                                const BatchNormGradExt2RARFullLoadTilingData* tilingData)
    {
        const BatchNormGradExt2BaseTilingData& baseTilingData = tilingData->baseTilingData;
        const BatchNormGradExt2BinaryAddTilingData& binaryAddTilingData = tilingData->binaryAddTilingData;
        r1Dim_ = baseTilingData.r1Dim;
        aDim_ = baseTilingData.aDim;
        r0Dim_ = baseTilingData.r0Dim;
        rAlign_ = baseTilingData.rAlign;
        binaryAddQuotient_ = binaryAddTilingData.binaryAddQuotient;
        binaryAddK_ = binaryAddTilingData.binaryAddk;
        binaryAddLastNum_ = binaryAddTilingData.binaryAddLastNum;
        aTailCoreNum_ = baseTilingData.tailBlockNum;
        aDimPerCore_ = GetBlockIdx() < baseTilingData.tailBlockNum ? baseTilingData.tailBlockDim :
                                                                     baseTilingData.formerBlockDim;
        gmOffset_ = GetBlockIdx() < baseTilingData.tailBlockNum ?
                        GetBlockIdx() * baseTilingData.tailBlockDim :
                        baseTilingData.tailBlockNum * baseTilingData.tailBlockDim +
                            (GetBlockIdx() - baseTilingData.tailBlockNum) * baseTilingData.formerBlockDim;
        aDimPerLoop_ = tilingData->formerUbDim;
        aDimLoopNum_ = GetBlockIdx() < baseTilingData.tailBlockNum ? tilingData->ubLoopOfTailBlock :
                                                                     tilingData->ubLoopOfFormerBlock;
        aDimLoopTail_ = GetBlockIdx() < baseTilingData.tailBlockNum ? tilingData->ubTailOfTailBlock :
                                                                      tilingData->ubTailOfFormerBlock;

        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dy) + gmOffset_ * r0Dim_);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(x) + gmOffset_ * r0Dim_);
        meanGm_.SetGlobalBuffer((__gm__ float*)(mean) + gmOffset_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)(rstd) + gmOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(gamma) + gmOffset_);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dx) + gmOffset_ * r0Dim_);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dgamma) + gmOffset_);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dbeta) + gmOffset_);

        dyBufSize_ = RoundUpOneBlock(rAlign_ * sizeof(DY_TYPE)) * aDimPerLoop_;
        dyBufElemNum_ = dyBufSize_ / sizeof(DY_TYPE);
        xBufSize_ = RoundUpTwoBlock(rAlign_ * sizeof(float)) * aDimPerLoop_;
        xBufElemNum_ = xBufSize_ / sizeof(DY_TYPE);
        meanBufSize_ = RoundUpOneBlock(aDimPerLoop_ * sizeof(float));
        meanBufElemNum_ = meanBufSize_ / sizeof(float);
        gammaBufSize_ = RoundUpOneBlock(aDimPerLoop_ * sizeof(WEIGHT_TYPE));
        gammaBufElemNum_ = gammaBufSize_ / sizeof(WEIGHT_TYPE);
        binaryAddBufSize_ = RoundUpOneBlock(binaryAddQuotient_ / VL_FP32 * sizeof(float));
        binaryAddBufElemNum_ = binaryAddBufSize_ / sizeof(float);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyBufSize_);
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, xBufSize_);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(gammaInQue_, BUFFER_NUM, gammaBufSize_);
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(binaryAddBuf_, binaryAddBufSize_ * BUFFER_NUM);

        V_MTE3_EVENT = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        MTE3_MTE2_EVENT = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < aDimLoopNum_; i++) {
            uint32_t a = i == aDimLoopNum_ - 1 ? aDimLoopTail_ : aDimPerLoop_;
            uint64_t offset = i * aDimPerLoop_ * r0Dim_;
            uint64_t meanOffset = i * aDimPerLoop_;
            uint32_t bufferIdx = i % BUFFER_NUM;

            LocalTensor<DY_TYPE> dy = dyInQue_.template AllocTensor<DY_TYPE>();
            PrepareInDy(dy, offset, a);
            dyInQue_.EnQue(dy);
            dy = dyInQue_.template DeQue<DY_TYPE>();

            // x 必须在 CalcXhat 原地改写之前读入：补偿累加用原始 x 计算 sum(x*dy)
            if (i >= BUFFER_NUM) {
                WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
            }
            LocalTensor<DY_TYPE> x = xInQue_.template AllocTensor<DY_TYPE>();
            PrepareInX(x, offset, a);
            xInQue_.EnQue(x);
            x = xInQue_.template DeQue<DY_TYPE>();

            LocalTensor<float> mean = meanInQue_.template AllocTensor<float>();
            PrepareMean(mean, meanOffset, a);
            meanInQue_.EnQue(mean);
            mean = meanInQue_.template DeQue<float>();

            LocalTensor<float> rstd = rstdInQue_.template AllocTensor<float>();
            PrepareRstd(rstd, meanOffset, a);
            rstdInQue_.EnQue(rstd);
            rstd = rstdInQue_.template DeQue<float>();

            LocalTensor<float> dbeta = dbetaOutQue_.template AllocTensor<float>();
            LocalTensor<float> dgamma = dgammaOutQue_.template AllocTensor<float>();
            CalcDbetaDgammaComp(dy, x, mean, rstd, dbeta, dgamma, a, r1Dim_ * r0Dim_, rAlign_);
            dbetaOutQue_.EnQue(dbeta);
            dgammaOutQue_.EnQue(dgamma);

            LocalTensor<WEIGHT_TYPE> gamma = gammaInQue_.template AllocTensor<WEIGHT_TYPE>();
            PrepareInGamma(gamma, meanOffset, a);
            gammaInQue_.EnQue(gamma);
            gamma = gammaInQue_.template DeQue<WEIGHT_TYPE>();

            CalcXhat(x, mean, a, r0Dim_ * r1Dim_, rAlign_);
            meanInQue_.FreeTensor(mean);

            CalcDx(x, dy, rstd, gamma, dgamma, dbeta, a, r1Dim_ * r0Dim_, rAlign_);
            SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT + bufferIdx);
            WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT + bufferIdx);

            dyInQue_.FreeTensor(dy);
            xInQue_.FreeTensor(x);
            rstdInQue_.FreeTensor(rstd);
            gammaInQue_.FreeTensor(gamma);

            LocalTensor<WEIGHT_TYPE> dbetaOut = dbetaOutQue_.template DeQue<WEIGHT_TYPE>();
            CopyOutDbeta(dbetaOut, meanOffset, a);
            dbetaOutQue_.FreeTensor(dbetaOut);

            LocalTensor<WEIGHT_TYPE> dgammaOut = dgammaOutQue_.template DeQue<WEIGHT_TYPE>();
            CopyOutDgamma(dgammaOut, meanOffset, a);
            dgammaOutQue_.FreeTensor(dgammaOut);

            CopyOutDx(x, offset, a);
            SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT + bufferIdx);
        }
    }

    __aicore__ inline void PrepareInDy(LocalTensor<DY_TYPE>& dy, uint64_t offset, uint64_t a)
    {
        CopyInRAR(dy, dyGm_, 0, offset, a);
    }

    __aicore__ inline void PrepareInX(LocalTensor<DY_TYPE>& x, int64_t offset, uint64_t a)
    {
        if constexpr (IsSameType<DY_TYPE, float>::value) {
            CopyInRAR(x, xGm_, 0, offset, a);
        } else {
            CopyInRAR(x, xGm_, xBufElemNum_ / 2, offset, a);
        }
    }

    __aicore__ inline void PrepareMean(LocalTensor<float>& mean, uint64_t offset, uint64_t a)
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

    __aicore__ inline void CalcXhat(LocalTensor<DY_TYPE>& x, LocalTensor<float>& mean, uint32_t a, uint32_t r,
                                    uint32_t rAlign)
    {
        uint16_t VL = VECTOR_LENGTH / sizeof(float);
        uint16_t loopTimes = (r + VL - 1) / VL;
        const __local_mem__ DY_TYPE* xSrcAddr = (__local_mem__ DY_TYPE*)x.GetPhyAddr();
        const __local_mem__ float* xDstAddr = (const __local_mem__ float*)x.GetPhyAddr();
        if constexpr (IsSameType<DY_TYPE, half>::value) {
            xSrcAddr += xBufElemNum_ / DIGIT_TWO;
        }
        const __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> meanValue;
            MicroAPI::MaskReg pregR;
            MicroAPI::RegTensor<float> xValue;
            MicroAPI::RegTensor<float> xHat;
            uint32_t sregMask;

            for (uint16_t i = 0; i < (uint16_t)a; i++) {
                DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanValue, (__local_mem__ float*)(meanAddr) + i);
                sregMask = (uint32_t)r;
                for (uint16_t j = 0; j < loopTimes; j++) {
                    pregR = MicroAPI::UpdateMask<float>(sregMask);
                    LoadOneTensor<DY_TYPE>(xSrcAddr, xValue, pregR, i * rAlign + VL * j);
                    Sub(xHat, xValue, meanValue, pregR);
                    DataCopy(((__local_mem__ float*)xDstAddr) + i * rAlign + VL * j, xHat, pregR);
                }
            }
        }
    }

    // 补偿累加 dbeta = sum(dy)、dgamma = rstd * (sum(x*dy) - mean * sum(dy))：
    // 每个通道一次遍历完成 S0 与 S1 的 double-fp32 三字累加（见 comp_sum.h CompAddWide），
    // 跨 lane 用 CompReduceLanes 归约到 lane0，最后 CompFinalizeDgammaDbeta 只做一次输出舍入。
    // 通道之间串行（a 小），R 方向按 VL 分段并行。
    __aicore__ inline void CalcDbetaDgammaComp(const LocalTensor<DY_TYPE>& dyLocal, const LocalTensor<DY_TYPE>& xLocal,
                                               const LocalTensor<float>& mean, const LocalTensor<float>& rstd,
                                               const LocalTensor<float>& dbeta, const LocalTensor<float>& dgamma,
                                               uint32_t a, uint32_t r, uint32_t rAlign)
    {
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal.GetPhyAddr();
        __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xLocal.GetPhyAddr();
        if constexpr (IsSameType<DY_TYPE, half>::value) {
            // fp16：输入 x 在 buffer 后半段（前半段留给原地改写后的 xhat），读偏移必须与 CalcXhat 一致
            xAddr += xBufElemNum_ / DIGIT_TWO;
        }
        __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd.GetPhyAddr();
        __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbeta.GetPhyAddr();
        __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgamma.GetPhyAddr();
        uint16_t outerLoopTimes = CeilDiv(r, VL_FP32);

        __VEC_SCOPE__
        {
            MicroAPI::MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg one = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::RegTensor<float> s0Hi, s0Lo, s0Lo2, s1Hi, s1Lo, s1Lo2;
            MicroAPI::RegTensor<float> satS0, satS1, satS0Red, satS1Red;
            MicroAPI::RegTensor<float> dyReg, xReg, prodHi, meanReg, rstdReg, dgFinal, dbFinal;
            for (uint16_t k = 0; k < static_cast<uint16_t>(a); ++k) {
                uint32_t aOffset = k * rAlign;
                uint32_t sreg = r;
                MicroAPI::Duplicate(s0Hi, 0.0f, all);
                MicroAPI::Duplicate(s0Lo, 0.0f, all);
                MicroAPI::Duplicate(s0Lo2, 0.0f, all);
                MicroAPI::Duplicate(s1Hi, 0.0f, all);
                MicroAPI::Duplicate(s1Lo, 0.0f, all);
                MicroAPI::Duplicate(s1Lo2, 0.0f, all);
                MicroAPI::Duplicate(satS0, 0.0f, all);
                MicroAPI::Duplicate(satS1, 0.0f, all);
                for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(sreg);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, mask, aOffset + i * VL_FP32);
                    CompAddWide<DY_TYPE>(s0Hi, s0Lo, s0Lo2, dyReg, mask);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, mask, aOffset + i * VL_FP32);
                    if constexpr (IsSameType<DY_TYPE, float>::value) {
                        CompAddProductWide<DY_TYPE>(s1Hi, s1Lo, s1Lo2, xReg, dyReg, mask);
                        // 缩放伴生累加（只有 fp32 输入可能饱和；fp16 上该分支不编译）
                        CompAddScaled(satS0, satS1, xReg, dyReg, mask);
                    } else {
                        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(prodHi, xReg, dyReg, mask);
                        CompAddWide<DY_TYPE>(s1Hi, s1Lo, s1Lo2, prodHi, mask);
                    }
                }
                CompFoldLo2(s0Lo, s0Lo2, all);
                CompFoldLo2(s1Lo, s1Lo2, all);
                // 通道内 R 方向共 r 个元素，只有前 min(r, VL) 条 lane 收过数据
                // （见 comp_sum.h CompReduceBeginWidth），归约树从对应层起跳。
                const uint32_t laneCount = CompLaneBound(static_cast<int64_t>(r));
                CompReduceLanes(s0Hi, s0Lo, all, laneCount);
                CompReduceLanes(s1Hi, s1Lo, all, laneCount);
                LoadOneElement<float>(meanAddr, meanReg, one, k);
                LoadOneElement<float>(rstdAddr, rstdReg, one, k);
                if constexpr (IsSameType<DY_TYPE, float>::value) {
                    MicroAPI::ReduceSum(satS0Red, satS0, all);
                    MicroAPI::ReduceSum(satS1Red, satS1, all);
                    CompFinalizeDgammaDbetaSat(s1Hi, s1Lo, s0Hi, s0Lo, satS0Red, satS1Red, meanReg, rstdReg, one,
                                               dgFinal, dbFinal);
                } else {
                    CompFinalizeDgammaDbeta(s1Hi, s1Lo, s0Hi, s0Lo, meanReg, rstdReg, one, dgFinal, dbFinal);
                }
                StoreOneElement<float>(dbetaAddr, dbFinal, one, k);
                StoreOneElement<float>(dgammaAddr, dgFinal, one, k);
            }
        }
    }

    __aicore__ inline void CalcDx(LocalTensor<DY_TYPE>& xhat, LocalTensor<DY_TYPE>& dy, LocalTensor<float> rstd,
                                  LocalTensor<WEIGHT_TYPE>& gamma, LocalTensor<float>& dgamma,
                                  LocalTensor<float>& dbeta, uint32_t a, uint32_t r, uint32_t rAlign)
    {
        __local_mem__ DY_TYPE* xhatAddr = (__local_mem__ DY_TYPE*)xhat.GetPhyAddr();
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dy.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* gammaAddr = (__local_mem__ WEIGHT_TYPE*)gamma.GetPhyAddr();
        __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgamma.GetPhyAddr();
        __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbeta.GetPhyAddr();

        uint16_t VL = VECTOR_LENGTH / sizeof(float);
        uint16_t loopTimes = (r + VL - 1) / VL;
        float hRecipValue = 1.0f / (float)r;
        float hNegDelta = ExactDivisorNegDelta(static_cast<int64_t>(r)); // -(R - fl32(R))
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> rstdValue;
            MicroAPI::RegTensor<float> gammaValue;
            MicroAPI::RegTensor<float> dgammaValue;
            MicroAPI::RegTensor<float> dbetaValue;
            MicroAPI::MaskReg pregR;
            MicroAPI::RegTensor<float> dy;
            MicroAPI::RegTensor<float> xhat;
            MicroAPI::RegTensor<float> dx, dxResid;
            MicroAPI::RegTensor<float> qbReg, qbLoReg, qsReg, qsLoReg;
            MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            // 与 golden/torch 同构的 dx 结合次序：
            //   dx = ((dy - dbeta/R) - ((x-mean)*rstd)*(dgamma/R)) * gamma * rstd
            // 两个除法取**精确两字商**（按精确整数 R 补残差），t 与后两个乘积全程 dd，收尾只舍入一次
            // （comp_sum.h CompTrainDxPairFromXc + common.h StoreDxSingleRounded）：fp32 单链的多次
            // 舍入会在 fp16 网格中点处与 golden 差 1 ULP，是 L1_014 rmse 假红的根因。
            const bool useTailWord = static_cast<int64_t>(r) > (1L << 24);
            MicroAPI::RegTensor<float> rhiReg, rloReg;
            MicroAPI::Duplicate(rhiReg, hRecipValue, preg);
            CompExactDivisorReciprocalLo(rloReg, rhiReg, static_cast<float>(r), hNegDelta, preg);
            for (uint16_t i = 0; i < (uint16_t)a; i++) {
                DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdValue, (__local_mem__ float*)(rstdAddr) + i);
                DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue, (__local_mem__ float*)(dgammaAddr) + i);
                DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue, (__local_mem__ float*)(dbetaAddr) + i);
                LoadOneElement<WEIGHT_TYPE>(gammaAddr, gammaValue, preg, i);
                // doffset/R、dscale/R 每通道一次；非有限 value 时函数内部回退到 value*recip
                CompRQuotientPair(qbReg, qbLoReg, dbetaValue, rhiReg, rloReg, preg);
                CompRQuotientPair(qsReg, qsLoReg, dgammaValue, rhiReg, rloReg, preg);
                if (!useTailWord) {
                    MicroAPI::Duplicate(qbLoReg, 0.0f, preg);
                    MicroAPI::Duplicate(qsLoReg, 0.0f, preg);
                }
                uint32_t sregMask = (uint32_t)r;
                for (uint16_t j = 0; j < loopTimes; j++) {
                    pregR = MicroAPI::UpdateMask<float>(sregMask);
                    DataCopy(xhat, (__local_mem__ float*)(xhatAddr) + i * rAlign + VL * j);
                    LoadOneTensor<DY_TYPE>(dyAddr, dy, pregR, i * rAlign + VL * j);
                    CompTrainDxPairFromXc(dx, dxResid, dy, xhat, rstdValue, gammaValue, qbReg, qbLoReg, qsReg, qsLoReg,
                                          pregR);
                    StoreDxSingleRounded<DY_TYPE>(xhatAddr, dx, dxResid, pregR, i * rAlign + VL * j);
                }
                if constexpr (IsSameType<WEIGHT_TYPE, half>::value) {
                    StoreOneElement<WEIGHT_TYPE>(dbetaAddr, dbetaValue, pregMerge, i);
                    StoreOneElement<WEIGHT_TYPE>(dgammaAddr, dgammaValue, pregMerge, i);
                }
            }
        }
    }

    __aicore__ inline void CopyOutDx(LocalTensor<DY_TYPE>& dx, uint64_t offset, uint32_t a)
    {
        CopyOutRAR(dx, dxGm_, offset, a);
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
                                     uint64_t localOffset, uint64_t globalOffset, uint64_t a)
    {
        // RAR -> AR，R轴通过Compact Mode补pad到block对齐。
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = a;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = r0Dim_ * sizeof(DY_TYPE);
        loopParams.loop1DstStride = rAlign_ * sizeof(DY_TYPE);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

        DataCopyPadExtParams<DY_TYPE> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = r1Dim_;
        copyInParams.blockLen = r0Dim_ * sizeof(DY_TYPE);
        copyInParams.srcStride = (aDim_ - 1) * r0Dim_ * sizeof(DY_TYPE);
        copyInParams.dstStride = 0;
        DataCopyPad<DY_TYPE, PaddingMode::Compact>(localTensor[localOffset], globalTensor[globalOffset], copyInParams,
                                                   dataCopyPadExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
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
                                      uint64_t offset, uint64_t a)
    {
        // AR -> RAR
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = a;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = rAlign_ * sizeof(DY_TYPE);
        loopParams.loop1DstStride = r0Dim_ * sizeof(DY_TYPE);
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = r1Dim_;
        copyOutParams.blockLen = r0Dim_ * sizeof(DY_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = (aDim_ - 1) * r0Dim_ * sizeof(DY_TYPE);
        DataCopyPad<DY_TYPE, PaddingMode::Compact>(globalTensor[offset], localTensor, copyOutParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
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

    TQue<QuePosition::VECIN, BUFFER_NUM> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;

    GlobalTensor<DY_TYPE> dyGm_, xGm_, dxGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, rstdGm_, workspaceGm_;

    event_t V_MTE3_EVENT;
    event_t MTE3_MTE2_EVENT;

    uint32_t r1Dim_;
    uint32_t aDim_;
    uint32_t r0Dim_;
    uint32_t rAlign_;
    uint32_t binaryAddQuotient_;
    uint32_t binaryAddK_;
    uint32_t binaryAddLastNum_;
    uint32_t aDimPerCore_;
    uint32_t aTailCoreNum_;
    uint32_t aDimTail_;
    uint32_t gmOffset_;
    uint32_t aDimPerLoop_;
    uint32_t aDimLoopNum_;
    uint32_t aDimLoopTail_;
    uint32_t dyBufSize_;
    uint32_t dyBufElemNum_;
    uint32_t xBufSize_;
    uint32_t xBufElemNum_;
    uint32_t meanBufSize_;
    uint32_t meanBufElemNum_;
    uint32_t gammaBufSize_;
    uint32_t gammaBufElemNum_;
    uint32_t binaryAddBufSize_;
    uint32_t binaryAddBufElemNum_;
};
} // namespace BatchNormGradExt2
#endif // __BATCH_NORM_GRAD_EXT2_FULL_LOAD_REGBASE_H__
