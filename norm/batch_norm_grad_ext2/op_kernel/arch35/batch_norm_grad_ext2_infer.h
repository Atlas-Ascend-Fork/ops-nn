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
 * \file batch_norm_grad_ext2_infer.h
 * \brief
 */

#ifndef BATCH_NORM_GRAD_EXT2_INFER_H
#define BATCH_NORM_GRAD_EXT2_INFER_H

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "batch_norm_grad_ext2_common.h"
#include "batch_norm_grad_ext2_comp_sum.h"

namespace BatchNormGradExt2 {
using namespace AscendC;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

static constexpr int64_t UB_ADD_BUF = 3 * 1024; // ub间二分累加固定buf
// R1 跨核切分（useRSplit_）允许的最大通道数：合并阶段一条向量覆盖全部通道。
static constexpr int64_t RS_MAX_A_DIM = 8;
// R1 跨核切分至少需要的 r1 行数：太少则切完每核不足一行、并行度反而不如按 A 切。
static constexpr int64_t RS_MIN_R1 = 16;

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1, bool EN_DGAMMA = true, bool EN_DBETA = true>
class BatchNormGradExt2InferDgammaDbetaSplitR0 {
public:
    __aicore__ inline BatchNormGradExt2InferDgammaDbetaSplitR0(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR mean, GM_ADDR var, GM_ADDR gamma, GM_ADDR dx,
                                GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace,
                                const BatchNormGradExt2InferTilingData* tilingData)
    {
        epsilon = tilingData->baseTilingData.epsilon;
        r1Dim_ = tilingData->baseTilingData.r1Dim;
        aDim_ = tilingData->baseTilingData.aDim;
        r0Dim_ = tilingData->baseTilingData.r0Dim;
        blockNum_ = tilingData->blockNum;
        /* ub内二分累加相关参数 */
        /* A轴开多核的相关参数 */
        aDimPerCore_ = GetBlockIdx() < tilingData->tailBlockNum ? tilingData->tailBlockDim : tilingData->formerBlockDim;
        // ---- dgamma/dbeta 的三条路径（条件与 host: batch_norm_grad_ext2_tiling_infer.cpp 严格一致）----
        // 1) lane = 通道（r1Dim*r0Dim <= 64）：每通道的 R 很短、通道极多时，旧路径逐通道串行，
        //    每通道一次 mean/rstd 搬入 + dbeta/dgamma 搬出 + 累加器初始化/finalize。
        useChannelLane_ = EN_DGAMMA && EN_DBETA && IsSameType<WEIGHT_TYPE, float>::value &&
                          (r1Dim_ * r0Dim_ <= static_cast<int64_t>(VL_FP32)) && (aDim_ >= r1Dim_ * r0Dim_) &&
                          (aDimPerCore_ > 0);
        // 2) R1 跨核切分：A 很短（<= RS_MAX_A_DIM）而 R1 很长时（如 L1_191 NCHW 128x2x8709x2），
        //    旧路径只有 aDim 个核有活（2/72），其余核空闲。tiling 侧把 blockNum 提到满核数，
        //    本核只处理 r1 的一段、但处理"全部通道"；每核每通道 4 个 dd 字落 workspace，
        //    SyncAll 之后 0 号核用补偿加法合并并 finalize。
        //    （非 rSplit 时 tiling 的 blockNum 恒 <= aDim，故 blockNum_ > aDim_ 是可靠判据。）
        useRSplit_ = EN_DGAMMA && EN_DBETA && !useChannelLane_ && (blockNum_ > aDim_) && (aDim_ <= RS_MAX_A_DIM) &&
                     (r1Dim_ >= RS_MIN_R1) && (aDim_ * static_cast<int64_t>(FOUR) <= blockNum_);
        gmOffset_ = useRSplit_ ? 0 :
                                 (GetBlockIdx() < tilingData->tailBlockNum ?
                                      (GetBlockIdx() * tilingData->tailBlockDim) :
                                      (tilingData->tailBlockNum * tilingData->tailBlockDim +
                                       (GetBlockIdx() - tilingData->tailBlockNum) * tilingData->formerBlockDim));
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
        varGm_.SetGlobalBuffer((__gm__ float*)(var) + gmOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(gamma) + gmOffset_);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dx) + gmOffset_ * r0Dim_);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dgamma) + gmOffset_);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dbeta) + gmOffset_);

        // r1Dim*r0Dim <= 64 时 dgamma/dbeta 需要的是"每通道很短的 R"：旧路径逐通道串行（每通道一次
        // mean/rstd 32B 搬入 + dbeta/dgamma 4B 搬出 + 累加器初始化/finalize），通道多时被延迟打死
        // （L1_046：14K 通道/核 × 313ns ≈ 4.5ms，而访存下限 208us）。此时改用 lane = 通道。
        // 只接管"dgammy+dbeta 都开、权重 fp32、A 比 R 长"的形状（其余保持旧路径，逐位不变）。
        if (useChannelLane_) {
            InitChannelLaneBuffers();
            return;
        }
        if (useRSplit_) {
            InitRSplitBuffers(workspace);
            return;
        }

        int64_t dyBufSize = tailFactorAlign_ * BatchNormGradExt2::TWO * sizeof(DY_TYPE);
        int64_t xBufSize = ubRDimTailFactorAlign_ * BatchNormGradExt2::TWO * sizeof(float);
        xBufElemNum_ = xBufSize / sizeof(DY_TYPE);

        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyBufSize);
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, xBufSize);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, BatchNormGradExt2::ONE_BLK_SIZE);
        pipe_->InitBuffer(varInQue_, BUFFER_NUM, BatchNormGradExt2::ONE_BLK_SIZE);
        pipe_->InitBuffer(dbetaOutQue_, 1, UB_ADD_BUF);
        pipe_->InitBuffer(dgammaOutQue_, 1, UB_ADD_BUF);
        pipe_->InitBuffer(compAccBuf_, BatchNormGradExt2::FOUR * VL_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= blockNum_) {
            return;
        }
        if (useChannelLane_) {
            ProcessChannelLane();
            return;
        }
        if (useRSplit_) {
            ProcessRSplit();
            return;
        }
        for (int64_t i = 0; i < aDimPerCore_; i++) {
            int64_t meanOffset = i;
            ProcessPre(meanOffset);
            CalcDbetaAndDgammaPerA(meanOffset);
            CopyOutDbetaAndDgamma(meanOffset);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // lane = 通道：每 tile 最多 VL_FP32 个通道，UB 内转置成 [r0][通道]，R 在 lane 内串行累加。
    // 逐元素算子与旧路径同一套（CompAdd / CompAddProduct，2 字补偿），精度口径不变；
    // mean/rstd、dbeta/dgamma 的搬运从"每通道 4B"变成"每 tile 256B"。
    // ---------------------------------------------------------------------------------------------
    __aicore__ inline void InitChannelLaneBuffers()
    {
        rLen_ = r1Dim_ * r0Dim_;
        const int64_t elems = static_cast<int64_t>(VL_FP32) * rLen_;
        pipe_->InitBuffer(dyLaneQue_, BatchNormGradExt2::TWO, elems * sizeof(DY_TYPE));
        pipe_->InitBuffer(xLaneQue_, BatchNormGradExt2::TWO, elems * sizeof(DY_TYPE));
        const int64_t vecBytes = static_cast<int64_t>(VL_FP32) * sizeof(float);
        pipe_->InitBuffer(meanLaneQue_, 1, vecBytes);
        pipe_->InitBuffer(varLaneQue_, 1, vecBytes);
        pipe_->InitBuffer(dgammaLaneQue_, 1, vecBytes);
        pipe_->InitBuffer(dbetaLaneQue_, 1, vecBytes);
    }

    __aicore__ inline void ProcessChannelLane()
    {
        const int64_t laneNum = static_cast<int64_t>(VL_FP32);
        for (int64_t chDone = 0; chDone < aDimPerCore_; chDone += laneNum) {
            int64_t curCh = aDimPerCore_ - chDone;
            if (curCh > laneNum) {
                curCh = laneNum;
            }
            CopyInChannelTile(chDone, curCh);

            LocalTensor<DY_TYPE> dyT = dyLaneQue_.template DeQue<DY_TYPE>();
            LocalTensor<DY_TYPE> xT = xLaneQue_.template DeQue<DY_TYPE>();
            LocalTensor<float> dgammaT = dgammaLaneQue_.template AllocTensor<float>();
            LocalTensor<float> dbetaT = dbetaLaneQue_.template AllocTensor<float>();
            LocalTensor<float> meanT;
            LocalTensor<float> varT;
            if constexpr (EN_DGAMMA) {
                meanT = meanLaneQue_.template DeQue<float>();
                varT = varLaneQue_.template DeQue<float>();
            }
            ComputeChannelTile(curCh, (__local_mem__ DY_TYPE*)dyT.GetPhyAddr(), (__local_mem__ DY_TYPE*)xT.GetPhyAddr(),
                               (__local_mem__ float*)meanT.GetPhyAddr(), (__local_mem__ float*)varT.GetPhyAddr(),
                               (__local_mem__ float*)dgammaT.GetPhyAddr(), (__local_mem__ float*)dbetaT.GetPhyAddr());
            dgammaLaneQue_.EnQue(dgammaT);
            dbetaLaneQue_.EnQue(dbetaT);
            dyLaneQue_.FreeTensor(dyT);
            xLaneQue_.FreeTensor(xT);
            if constexpr (EN_DGAMMA) {
                meanLaneQue_.FreeTensor(meanT);
                varLaneQue_.FreeTensor(varT);
            }

            DataCopyExtParams outParams;
            outParams.blockCount = 1;
            outParams.blockLen = static_cast<uint32_t>(curCh * sizeof(float));
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            LocalTensor<float> dg = dgammaLaneQue_.template DeQue<float>();
            DataCopyPad(dgammaGm_[chDone], dg, outParams);
            dgammaLaneQue_.FreeTensor(dg);
            LocalTensor<float> db = dbetaLaneQue_.template DeQue<float>();
            DataCopyPad(dbetaGm_[chDone], db, outParams);
            dbetaLaneQue_.FreeTensor(db);
        }
    }

    __aicore__ inline void CopyInChannelTile(int64_t chDone, int64_t curCh)
    {
        LocalTensor<DY_TYPE> dy = dyLaneQue_.template AllocTensor<DY_TYPE>();
        LocalTensor<DY_TYPE> x = xLaneQue_.template AllocTensor<DY_TYPE>();
        const int64_t chOff = chDone * r0Dim_;
        const int64_t perR1 = r0Dim_ * static_cast<int64_t>(VL_FP32);
        for (int64_t r1 = 0; r1 < r1Dim_; ++r1) {
            const int64_t srcOff = r1 * (aDim_ * r0Dim_) + chOff;
            CopyInTransposeTile<DY_TYPE>(dy[r1 * perR1], dyGm_[srcOff], curCh, r0Dim_, r0Dim_,
                                         static_cast<int64_t>(VL_FP32));
            CopyInTransposeTile<DY_TYPE>(x[r1 * perR1], xGm_[srcOff], curCh, r0Dim_, r0Dim_,
                                         static_cast<int64_t>(VL_FP32));
        }
        dyLaneQue_.EnQue(dy);
        xLaneQue_.EnQue(x);

        if constexpr (EN_DGAMMA) {
            DataCopyExtParams inParams;
            inParams.blockCount = 1;
            inParams.blockLen = static_cast<uint32_t>(curCh * sizeof(float));
            inParams.srcStride = 0;
            inParams.dstStride = 0;
            DataCopyPadExtParams<float> padParams;
            padParams.isPad = false;
            LocalTensor<float> mean = meanLaneQue_.template AllocTensor<float>();
            LocalTensor<float> var = varLaneQue_.template AllocTensor<float>();
            DataCopyPad(mean, meanGm_[chDone], inParams, padParams);
            DataCopyPad(var, varGm_[chDone], inParams, padParams);
            meanLaneQue_.EnQue(mean);
            varLaneQue_.EnQue(var);
        }
    }

    __aicore__ inline void ComputeChannelTile(int64_t curCh, __local_mem__ DY_TYPE* dy, __local_mem__ DY_TYPE* x,
                                              __local_mem__ float* meanAddr, __local_mem__ float* varAddr,
                                              __local_mem__ float* dgammaAddr, __local_mem__ float* dbetaAddr)
    {
        uint32_t chLanes = static_cast<uint32_t>(curCh);
        const uint32_t laneStride = static_cast<uint32_t>(VL_FP32);
        const uint32_t rLen = static_cast<uint32_t>(rLen_);
        __VEC_SCOPE__
        {
            MaskReg chMask = AscendC::MicroAPI::UpdateMask<float>(chLanes);
            MaskReg all = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            RegTensor<float> s0Hi, s0Lo, s1Hi, s1Lo, dyReg, xReg, prodHi;
            RegTensor<float> meanReg, varReg, rstdReg, dgFinal, dbFinal;
            AscendC::MicroAPI::Duplicate(s0Hi, 0.0f, all);
            AscendC::MicroAPI::Duplicate(s0Lo, 0.0f, all);
            if constexpr (EN_DGAMMA) {
                AscendC::MicroAPI::Duplicate(s1Hi, 0.0f, all);
                AscendC::MicroAPI::Duplicate(s1Lo, 0.0f, all);
            }
            for (uint16_t r = 0; r < static_cast<uint16_t>(rLen); ++r) {
                const uint32_t off = static_cast<uint32_t>(r) * laneStride;
                LoadOneTensor<DY_TYPE>(dy, dyReg, chMask, off);
                Comp::CompAdd<DY_TYPE>(s0Hi, s0Lo, dyReg, chMask);
                if constexpr (EN_DGAMMA) {
                    LoadOneTensor<DY_TYPE>(x, xReg, chMask, off);
                    if constexpr (IsSameType<DY_TYPE, float>::value) {
                        Comp::CompAddProduct<DY_TYPE>(s1Hi, s1Lo, xReg, dyReg, chMask);
                    } else {
                        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(prodHi, xReg, dyReg, chMask);
                        Comp::CompAdd<DY_TYPE>(s1Hi, s1Lo, prodHi, chMask);
                    }
                }
            }
            if constexpr (EN_DGAMMA) {
                LoadOneTensor<float>(meanAddr, meanReg, chMask, 0);
                LoadOneTensor<float>(varAddr, varReg, chMask, 0);
                Comp::RstdExactF32(varReg, rstdReg, chMask, epsilon);
                Comp::CompFinalizeDgammaDbeta(s1Hi, s1Lo, s0Hi, s0Lo, meanReg, rstdReg, chMask, dgFinal, dbFinal);
                StoreOneTensor<float>(dgammaAddr, dgFinal, chMask, 0);
                StoreOneTensor<float>(dbetaAddr, dbFinal, chMask, 0);
            } else {
                Comp::CompRound(dbFinal, s0Hi, s0Lo, chMask);
                StoreOneTensor<float>(dbetaAddr, dbFinal, chMask, 0);
            }
        }
    }

    __aicore__ inline void ProcessPre(int64_t meanOffset)
    {
        if constexpr (EN_DBETA) {
            dbeta_ = dbetaOutQue_.template AllocTensor<float>();
        }
        if constexpr (EN_DGAMMA) {
            CopyInMeanAndRstd(meanOffset);
            dgamma_ = dgammaOutQue_.template AllocTensor<float>();
            var_ = varInQue_.template DeQue<float>();
            mean_ = meanInQue_.template DeQue<float>();
        }
    }

    __aicore__ inline void ProcessTail(int64_t offset1, int64_t offset2, uint32_t tailLen)
    {
        CopyInDyTwo(offset1, offset2, tailLen);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        // S0 = sum(dy) 也是 dgamma 的输入（dgamma = rstd * (S1 - mean * S0)），
        // (EN_DGAMMA, !EN_DBETA) 实例化同样需要累加；可达配置 (true,true) 成本不变。
        if constexpr (EN_DBETA || EN_DGAMMA) {
            AccumDbeta(dyLocal, ubRDimTailFactor_, tailLen);
        }
        if constexpr (EN_DGAMMA) {
            CopyInXTwo(offset1, offset2, tailLen);
            LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
            AccumDgamma(dyLocal, xLocal, ubRDimTailFactor_, tailLen);
            xInQue_.FreeTensor(xLocal);
        }
        dyInQue_.FreeTensor(dyLocal);
    }

    __aicore__ inline void ProcessHalfMain(int64_t offset1)
    {
        CopyInDyOne(offset1, ubRDimTailFactor_);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        // S0 = sum(dy) 也是 dgamma 的输入（dgamma = rstd * (S1 - mean * S0)），
        // (EN_DGAMMA, !EN_DBETA) 实例化同样需要累加；可达配置 (true,true) 成本不变。
        if constexpr (EN_DBETA || EN_DGAMMA) {
            AccumDbeta(dyLocal, ubRDimTailFactor_, 0);
        }
        if constexpr (EN_DGAMMA) {
            CopyInXOne(offset1, ubRDimTailFactor_);
            LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
            AccumDgamma(dyLocal, xLocal, ubRDimTailFactor_, 0);
            xInQue_.FreeTensor(xLocal);
        }
        dyInQue_.FreeTensor(dyLocal);
    }

    __aicore__ inline void ProcessMain(int64_t offset1)
    {
        CopyInDyOne(offset1, ubRDimFactor_);
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        // S0 = sum(dy) 也是 dgamma 的输入（dgamma = rstd * (S1 - mean * S0)），
        // (EN_DGAMMA, !EN_DBETA) 实例化同样需要累加；可达配置 (true,true) 成本不变。
        if constexpr (EN_DBETA || EN_DGAMMA) {
            AccumDbeta(dyLocal, ubRDimFactor_, 0);
        }
        if constexpr (EN_DGAMMA) {
            CopyInXOne(offset1, ubRDimFactor_);
            LocalTensor<DY_TYPE> xLocal = xInQue_.template DeQue<DY_TYPE>();
            AccumDgamma(dyLocal, xLocal, ubRDimFactor_, 0);
            xInQue_.FreeTensor(xLocal);
        }
        dyInQue_.FreeTensor(dyLocal);
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
        AccumulateRows(meanOffset, 0, r1Dim_);
        FinalizeChannel();
    }

    // 累加单个通道在 r1 区间 [kBegin, kEnd) 上的和（S0=sum(dy)、S1=sum(dy*x)），结果留在
    // compAccBuf_ 的两字 lane 累加器里（未做 lane 归约 / finalize）。
    // rSplit 路径每个核只跑自己那段 r1，随后 ReduceLanesToPartial 落盘；老路径跑满 [0, r1Dim)。
    __aicore__ inline void AccumulateRows(int64_t meanOffset, int64_t kBegin, int64_t kEnd)
    {
        InitCompAcc();
        int64_t offset = meanOffset * r0Dim_; // A R0 里面的偏移
        int64_t tailOffset = ubRDimLoopNum_ * ubRDimFactor_;
        int64_t r1Offset = 0;
        int64_t offset1 = 0;
        int64_t offset2 = 0;
        for (int64_t k = kBegin; k < kEnd; k++) { // 遍历 r1
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
    }

    __aicore__ inline __local_mem__ float* CompAccAddr()
    {
        return (__local_mem__ float*)compAccBuf_.Get<float>().GetPhyAddr();
    }

    __aicore__ inline void InitCompAcc()
    {
        __local_mem__ float* acc = CompAccAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> zero;
            MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(zero, 0.0f, all);
            MicroAPI::DataCopy(acc, zero, all);
            MicroAPI::DataCopy(acc + VL_FP32, zero, all);
            MicroAPI::DataCopy(acc + DIGIT_TWO * VL_FP32, zero, all);
            MicroAPI::DataCopy(acc + DIGIT_THREE * VL_FP32, zero, all);
        }
    }

    // lane-parallel dd accumulation of one dy segment prefix plus its optional tail segment
    __aicore__ inline void AccumDbeta(const LocalTensor<DY_TYPE>& dyLocal, int64_t len0, int64_t len1)
    {
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal.GetPhyAddr();
        __local_mem__ float* acc = CompAccAddr();
        __VEC_SCOPE__
        {
            MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            RegTensor<float> laneHi, laneLo;
            MicroAPI::DataCopy(laneHi, acc);
            MicroAPI::DataCopy(laneLo, acc + VL_FP32);
            AccumulateLanes<DY_TYPE>(dyAddr, 0, len0, laneHi, laneLo);
            if (len1 > 0) {
                AccumulateLanes<DY_TYPE>(dyAddr, tailFactorAlign_, len1, laneHi, laneLo);
            }
            MicroAPI::DataCopy(acc, laneHi, all);
            MicroAPI::DataCopy(acc + VL_FP32, laneLo, all);
        }
    }

    __aicore__ inline void AccumDgamma(const LocalTensor<DY_TYPE>& dyLocal, const LocalTensor<DY_TYPE>& xLocal,
                                       int64_t len0, int64_t len1)
    {
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal.GetPhyAddr();
        __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xLocal.GetPhyAddr();
        __local_mem__ float* acc = CompAccAddr() + DIGIT_TWO * VL_FP32;
        __VEC_SCOPE__
        {
            MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            RegTensor<float> laneHi, laneLo;
            MicroAPI::DataCopy(laneHi, acc);
            MicroAPI::DataCopy(laneLo, acc + VL_FP32);
            AccumulateProdBand<DY_TYPE>(dyAddr, xAddr, 0, len0, laneHi, laneLo);
            if (len1 > 0) {
                AccumulateProdBand<DY_TYPE>(dyAddr, xAddr, tailFactorAlign_, len1, laneHi, laneLo);
            }
            MicroAPI::DataCopy(acc, laneHi, all);
            MicroAPI::DataCopy(acc + VL_FP32, laneLo, all);
        }
    }

    template <typename T>
    __aicore__ inline void AccumulateLanes(const __local_mem__ T* addr, int64_t baseOffset, int64_t len,
                                           RegTensor<float>& laneHi, RegTensor<float>& laneLo)
    {
        RegTensor<float> v;
        uint32_t count = static_cast<uint32_t>(len);
        for (uint16_t chunk = 0; chunk < static_cast<uint16_t>(len); chunk += static_cast<uint16_t>(VL_FP32)) {
            MaskReg mask = MicroAPI::UpdateMask<float>(count);
            LoadOneTensor<T>(addr, v, mask, baseOffset + chunk);
            Comp::CompAdd<T>(laneHi, laneLo, v, mask);
        }
    }

    template <typename T>
    __aicore__ inline void AccumulateProdBand(const __local_mem__ T* dyAddr, const __local_mem__ T* xAddr,
                                              int64_t baseOffset, int64_t len, RegTensor<float>& laneHi,
                                              RegTensor<float>& laneLo)
    {
        RegTensor<float> dy, x, ph, pl;
        uint32_t count = static_cast<uint32_t>(len);
        for (uint16_t chunk = 0; chunk < static_cast<uint16_t>(len); chunk += static_cast<uint16_t>(VL_FP32)) {
            MaskReg mask = MicroAPI::UpdateMask<float>(count);
            LoadOneTensor<T>(dyAddr, dy, mask, baseOffset + chunk);
            LoadOneTensor<T>(xAddr, x, mask, baseOffset + chunk);
            if constexpr (IsSameType<T, float>::value) {
                Comp::CompAddProduct<T>(laneHi, laneLo, x, dy, mask);
            } else {
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(ph, x, dy, mask);
                Comp::CompAdd<T>(laneHi, laneLo, ph, mask);
            }
        }
    }

    __aicore__ inline void FinalizeChannel()
    {
        __local_mem__ float* acc = CompAccAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)mean_.GetPhyAddr();
        __local_mem__ float* varAddr = (__local_mem__ float*)var_.GetPhyAddr();
        __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgamma_.GetPhyAddr();
        __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbeta_.GetPhyAddr();
        __VEC_SCOPE__
        {
            MaskReg one = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            RegTensor<float> s0Hi, s0Lo, s1Hi, s1Lo, meanReg, varReg, rstdReg, dgFinal, dbFinal;
            MicroAPI::DataCopy(s0Hi, acc);
            MicroAPI::DataCopy(s0Lo, acc + VL_FP32);
            MicroAPI::DataCopy(s1Hi, acc + DIGIT_TWO * VL_FP32);
            MicroAPI::DataCopy(s1Lo, acc + DIGIT_THREE * VL_FP32);
            // R 方向一次遍历的元素总数 = r1Dim * r0Dim；只有前 min(R, VL) 条 lane 收过数据，
            // 其余恒 +0（见 comp_sum.h CompReduceBeginWidth），归约树可直接从对应层起跳。
            const uint32_t laneCount = Comp::CompLaneBound(static_cast<int64_t>(r1Dim_) * r0Dim_);
            Comp::CompReduceLanes(s0Hi, s0Lo, all, laneCount);
            Comp::CompReduceLanes(s1Hi, s1Lo, all, laneCount);
            if constexpr (EN_DGAMMA) {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanAddr);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(varReg, varAddr);
                Comp::RstdExactF32(varReg, rstdReg, all, epsilon);
                Comp::CompFinalizeDgammaDbeta(s1Hi, s1Lo, s0Hi, s0Lo, meanReg, rstdReg, one, dgFinal, dbFinal);
                StoreOneTensor<float>(dgammaAddr, dgFinal, one, 0);
                if constexpr (EN_DBETA) {
                    StoreOneTensor<float>(dbetaAddr, dbFinal, one, 0);
                }
            } else {
                Comp::CompRound(dbFinal, s0Hi, s0Lo, one);
                if constexpr (EN_DBETA) {
                    StoreOneTensor<float>(dbetaAddr, dbFinal, one, 0);
                }
            }
        }
    }

    __aicore__ inline void CopyInMeanAndRstd(int64_t offset)
    {
        mean_ = meanInQue_.template AllocTensor<float>();
        var_ = varInQue_.template AllocTensor<float>();
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
        DataCopyPad(var_, varGm_[offset], copyInParams, dataCopyPadExtParams);
        meanInQue_.EnQue(mean_);
        varInQue_.EnQue(var_);
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
        // The two segments must use the same element offsets as the dy buffer (0 and
        // tailFactorAlign_): AccumulateProdBand receives ONE baseOffset and applies it to both the
        // dy and the x address.  Offsetting x by xBufElemNum_/TWO happens to equal tailFactorAlign_
        // only when sizeof(DY_TYPE) == sizeof(float); for fp16 it points at xBufElemNum_*sizeof,
        // i.e. at UB that the copy never wrote (that is what turned L0_024 dscale into -inf).
        CopyIn(xLocal, xGm_, 0, offset1, ubRDimTailFactor_);
        CopyIn(xLocal, xGm_, tailFactorAlign_, offset2, tailLen);
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInXOne(int64_t offset1, uint32_t copyLen)
    {
        LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
        // Same layout as CopyInXTwo / dy: segment 1 lives at element 0 (see the note above).
        CopyIn(xLocal, xGm_, 0, offset1, copyLen);
        xInQue_.EnQue(xLocal);
    }

    __aicore__ inline void CastDbetaAndDgamma()
    {
        __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgamma_.GetPhyAddr();
        __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbeta_.GetPhyAddr();

        __VEC_SCOPE__
        {
            MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            if constexpr (EN_DGAMMA) {
                MicroAPI::RegTensor<float> dgammaValue;
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue,
                                                                            (__local_mem__ float*)(dgammaAddr));
                StoreOneElement<WEIGHT_TYPE>(dgammaAddr, dgammaValue, pregMerge, 0);
            }
            if constexpr (EN_DBETA) {
                MicroAPI::RegTensor<float> dbetaValue;
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue,
                                                                            (__local_mem__ float*)(dbetaAddr));
                StoreOneElement<WEIGHT_TYPE>(dbetaAddr, dbetaValue, pregMerge, 0);
            }
        }
    }

    __aicore__ inline void CopyOutDbetaAndDgamma(int64_t offset)
    {
        if constexpr (EN_DGAMMA) {
            meanInQue_.FreeTensor(mean_);
            varInQue_.FreeTensor(var_);
        }

        if constexpr (IsSameType<WEIGHT_TYPE, half>::value) {
            CastDbetaAndDgamma();
        }

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = sizeof(WEIGHT_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        if constexpr (EN_DBETA) {
            dbetaOutQue_.EnQue(dbeta_);
            LocalTensor<WEIGHT_TYPE> dbetaOut = dbetaOutQue_.template DeQue<WEIGHT_TYPE>();
            DataCopyPad<WEIGHT_TYPE, PaddingMode::Compact>(dbetaGm_[offset], dbetaOut, copyOutParams);
            dbetaOutQue_.FreeTensor(dbetaOut);
        }
        if constexpr (EN_DGAMMA) {
            dgammaOutQue_.EnQue(dgamma_);
            LocalTensor<WEIGHT_TYPE> dgammaOut = dgammaOutQue_.template DeQue<WEIGHT_TYPE>();
            DataCopyPad<WEIGHT_TYPE, PaddingMode::Compact>(dgammaGm_[offset], dgammaOut, copyOutParams);
            dgammaOutQue_.FreeTensor(dgammaOut);
        }
    }

    // -------------------------------------------------------------------------------------------
    // R1 跨核切分（useRSplit_）：
    //   本核处理 r1 的 [r1Begin_, r1End_) 段、全部 aDim 个通道，每通道一条两字补偿 lane 累加器；
    //   收尾把 lane 归约成 4 个字（s0hi/s0lo/s1hi/s1lo）写 workspace[word][core][channel]；
    //   SyncAll 后 0 号核按通道向量（lane = 通道，一条向量覆盖全部通道）补偿合并并 finalize。
    //   落盘同样走 VECOUT 队列：FreeTensor 保证 MTE3 已把数据写进 GM，SyncAll 之后别的核才读得到。
    // -------------------------------------------------------------------------------------------
    __aicore__ inline void InitRSplitBuffers(GM_ADDR workspace)
    {
        wsGm_.SetGlobalBuffer((__gm__ float*)workspace);
        const int64_t dyBufSize = tailFactorAlign_ * TWO * sizeof(DY_TYPE);
        const int64_t xBufSize = ubRDimTailFactorAlign_ * TWO * sizeof(float);
        xBufElemNum_ = xBufSize / sizeof(DY_TYPE);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyBufSize);
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, xBufSize);
        pipe_->InitBuffer(compAccBuf_, FOUR * VL_FP32 * sizeof(float));
        // workspace 里每核每通道占 aDim_ 个 float，UB 侧按 32B 行距展开
        aDimPad_ = static_cast<int64_t>(RoundUpOneBlock(static_cast<uint32_t>(aDim_ * sizeof(float)))) /
                   static_cast<int64_t>(sizeof(float));
        aDimPad64_ = (aDim_ + static_cast<int64_t>(VL_FP32) - 1) / static_cast<int64_t>(VL_FP32) *
                     static_cast<int64_t>(VL_FP32);
        rowsPerCore_ = (r1Dim_ + blockNum_ - 1) / blockNum_;
        r1Begin_ = GetBlockIdx() * rowsPerCore_;
        r1End_ = r1Begin_ + rowsPerCore_ > r1Dim_ ? r1Dim_ : r1Begin_ + rowsPerCore_;
        activeCores_ = (r1Dim_ + rowsPerCore_ - 1) / rowsPerCore_;
        if (activeCores_ > blockNum_) {
            activeCores_ = blockNum_;
        }
        if (activeCores_ < 1) {
            activeCores_ = 1;
        }
        if (r1End_ > r1Begin_) {
            pipe_->InitBuffer(partOutQue_, 1, FOUR * aDimPad_ * sizeof(float));
        }
    }

    __aicore__ inline void ProcessRSplit()
    {
        if (r1End_ > r1Begin_) {
            LocalTensor<float> outT = partOutQue_.template AllocTensor<float>();
            __local_mem__ float* out = (__local_mem__ float*)outT.GetPhyAddr();
            for (int64_t c = 0; c < aDim_; ++c) {
                AccumulateRows(c, r1Begin_, r1End_);
                ReduceLanesToPartial(c, out);
            }
            partOutQue_.EnQue(outT);
            outT = partOutQue_.template DeQue<float>();
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(aDim_ * sizeof(float));
            params.srcStride = 0;
            params.dstStride = 0;
            const int64_t wsStride = blockNum_ * aDim_; // [word][core][channel]
            for (int64_t w = 0; w < FOUR; ++w) {
                DataCopyPad(wsGm_[w * wsStride + GetBlockIdx() * aDim_], outT[static_cast<uint32_t>(w * aDimPad_)],
                            params);
            }
            partOutQue_.FreeTensor(outT);
        }
        PipeBarrier<PIPE_ALL>();
        // 所有 block 都必须参与（blockDim 由 tiling 决定 = blockNum_）。
        SyncAll();
        if (GetBlockIdx() == 0) {
            pipe_->Reset();
            InitReduceBuffersRSplit();
            ReduceAndFinalizeRSplit();
        }
    }

    // 两字 lane 累加器 -> lane0 的 4 个字，按 [word][channel] 写进 out（out 是 VECOUT 队列缓冲）
    __aicore__ inline void ReduceLanesToPartial(int64_t c, __local_mem__ float* out)
    {
        __local_mem__ float* acc = CompAccAddr();
        const uint32_t laneCount = Comp::CompLaneBound((r1End_ - r1Begin_) * r0Dim_);
        const uint32_t ch = static_cast<uint32_t>(c);
        const uint32_t aPad = static_cast<uint32_t>(aDimPad_);
        __VEC_SCOPE__
        {
            MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MaskReg one = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            RegTensor<float> s0Hi, s0Lo, s1Hi, s1Lo;
            MicroAPI::DataCopy(s0Hi, acc);
            MicroAPI::DataCopy(s0Lo, acc + VL_FP32);
            MicroAPI::DataCopy(s1Hi, acc + DIGIT_TWO * VL_FP32);
            MicroAPI::DataCopy(s1Lo, acc + DIGIT_THREE * VL_FP32);
            Comp::CompReduceLanes(s0Hi, s0Lo, all, laneCount);
            Comp::CompReduceLanes(s1Hi, s1Lo, all, laneCount);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(out + ch, s0Hi, one);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(out + aPad + ch, s0Lo, one);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                out + static_cast<uint32_t>(DIGIT_TWO) * aPad + ch, s1Hi, one);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                out + static_cast<uint32_t>(DIGIT_THREE) * aPad + ch, s1Lo, one);
        }
    }

    __aicore__ inline void InitReduceBuffersRSplit()
    {
        pipe_->InitBuffer(partBuf_, (FOUR * blockNum_ * aDimPad_ + static_cast<int64_t>(VL_FP32)) * sizeof(float));
        pipe_->InitBuffer(meanBuf_, aDimPad64_ * sizeof(float));
        pipe_->InitBuffer(varBuf_, aDimPad64_ * sizeof(float));
        pipe_->InitBuffer(dgammaBuf_, aDimPad64_ * sizeof(float));
        pipe_->InitBuffer(dbetaBuf_, aDimPad64_ * sizeof(float));
    }

    __aicore__ inline void ReduceAndFinalizeRSplit()
    {
        LocalTensor<float> partT = partBuf_.Get<float>();
        DataCopyExtParams partParams;
        partParams.blockCount = static_cast<uint16_t>(blockNum_);
        partParams.blockLen = static_cast<uint32_t>(aDim_ * sizeof(float));
        partParams.srcStride = 0; // GM 上 [core][channel] 连续
        // UB 目的块天然按 32B 补齐，aDimPad_ 就是补齐后的行距（单位 float），故块间无额外 stride
        partParams.dstStride = 0;
        DataCopyPadExtParams<float> partPadParams;
        partPadParams.isPad = false;
        for (int64_t w = 0; w < FOUR; ++w) {
            DataCopyPad(partT[static_cast<uint32_t>(w * blockNum_ * aDimPad_)], wsGm_[w * blockNum_ * aDim_],
                        partParams, partPadParams);
        }

        LocalTensor<float> meanT = meanBuf_.Get<float>();
        LocalTensor<float> varT = varBuf_.Get<float>();
        LocalTensor<WEIGHT_TYPE> dgammaT = dgammaBuf_.Get<WEIGHT_TYPE>();
        LocalTensor<WEIGHT_TYPE> dbetaT = dbetaBuf_.Get<WEIGHT_TYPE>();
        DataCopyExtParams smallParams;
        smallParams.blockCount = 1;
        smallParams.blockLen = static_cast<uint32_t>(aDim_ * sizeof(float));
        smallParams.srcStride = 0;
        smallParams.dstStride = 0;
        DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        DataCopyPad(meanT, meanGm_, smallParams, padParams);
        DataCopyPad(varT, varGm_, smallParams, padParams);
        PipeBarrier<PIPE_ALL>();

        __local_mem__ float* meanPtr = (__local_mem__ float*)meanT.GetPhyAddr();
        __local_mem__ float* varPtr = (__local_mem__ float*)varT.GetPhyAddr();
        __local_mem__ float* dgPtr = (__local_mem__ float*)dgammaT.GetPhyAddr();
        __local_mem__ float* dbPtr = (__local_mem__ float*)dbetaT.GetPhyAddr();
        __local_mem__ float* part = (__local_mem__ float*)partT.GetPhyAddr();

        // rstd = fl32(1 / fl32(sqrt(fl32(var + eps))))，与 golden 的 _rstd_infer_fp32 逐位一致
        Comp::RstdExactF32Ub(varPtr, static_cast<uint32_t>(aDim_), epsilon);

        const uint32_t aPad = static_cast<uint32_t>(aDimPad_);
        uint32_t curALen = static_cast<uint32_t>(aDim_);
        const uint32_t wordStride = static_cast<uint32_t>(blockNum_ * aDimPad_);
        const uint16_t nCores = static_cast<uint16_t>(activeCores_);
        __VEC_SCOPE__
        {
            MaskReg pregMask = MicroAPI::UpdateMask<float>(curALen);
            RegTensor<float> hi0, lo0, hi1, lo1, ph, pl, meanReg, rstdReg, dgOut, dbOut;
            MicroAPI::DataCopy(hi0, part);
            MicroAPI::DataCopy(lo0, part + wordStride);
            MicroAPI::DataCopy(hi1, part + DIGIT_TWO * wordStride);
            MicroAPI::DataCopy(lo1, part + DIGIT_THREE * wordStride);
            for (uint16_t i = 1; i < nCores; ++i) {
                MicroAPI::DataCopy(ph, part + static_cast<uint32_t>(i) * aPad);
                MicroAPI::DataCopy(pl, part + wordStride + static_cast<uint32_t>(i) * aPad);
                Comp::CompAddPair<float>(hi0, lo0, ph, pl, pregMask);
                MicroAPI::DataCopy(ph, part + DIGIT_TWO * wordStride + static_cast<uint32_t>(i) * aPad);
                MicroAPI::DataCopy(pl, part + DIGIT_THREE * wordStride + static_cast<uint32_t>(i) * aPad);
                Comp::CompAddPair<float>(hi1, lo1, ph, pl, pregMask);
            }
            LoadOneTensor<float>(meanPtr, meanReg, pregMask, 0);
            LoadOneTensor<float>(varPtr, rstdReg, pregMask, 0);
            Comp::CompFinalizeDgammaDbeta(hi1, lo1, hi0, lo0, meanReg, rstdReg, pregMask, dgOut, dbOut);
            StoreOneTensor<WEIGHT_TYPE>(dgPtr, dgOut, pregMask, 0);
            StoreOneTensor<WEIGHT_TYPE>(dbPtr, dbOut, pregMask, 0);
        }
        PipeBarrier<PIPE_ALL>();

        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = static_cast<uint32_t>(aDim_ * sizeof(WEIGHT_TYPE));
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(dgammaGm_, dgammaT, outParams);
        DataCopyPad(dbetaGm_, dbetaT, outParams);
    }

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyInQue_;
    // R1 跨核切分路径（useRSplit_）
    bool useRSplit_{false};
    int64_t r1Begin_{0};
    int64_t r1End_{0};
    int64_t rowsPerCore_{0};
    int64_t activeCores_{1};
    int64_t aDimPad_{8};
    int64_t aDimPad64_{64};
    TQue<QuePosition::VECOUT, 1> partOutQue_;
    TBuf<TPosition::VECCALC> partBuf_;
    TBuf<TPosition::VECCALC> meanBuf_;
    TBuf<TPosition::VECCALC> varBuf_;
    TBuf<TPosition::VECCALC> dgammaBuf_;
    TBuf<TPosition::VECCALC> dbetaBuf_;
    GlobalTensor<float> wsGm_;
    // lane = 通道路径（r1Dim*r0Dim <= 64 时启用）
    bool useChannelLane_{false};
    int64_t rLen_{0};
    TQue<QuePosition::VECIN, BatchNormGradExt2::TWO> dyLaneQue_;
    TQue<QuePosition::VECIN, BatchNormGradExt2::TWO> xLaneQue_;
    TQue<QuePosition::VECIN, 1> meanLaneQue_;
    TQue<QuePosition::VECIN, 1> varLaneQue_;
    TQue<QuePosition::VECOUT, 1> dgammaLaneQue_;
    TQue<QuePosition::VECOUT, 1> dbetaLaneQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> varInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;
    TBuf<TPosition::VECCALC> compAccBuf_;

    GlobalTensor<DY_TYPE> dyGm_, xGm_, dxGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, varGm_;

    LocalTensor<float> mean_;
    LocalTensor<float> var_;
    LocalTensor<float> dgamma_;
    LocalTensor<float> dbeta_;

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
    float epsilon;

    uint32_t ubRDimFactor_{0};
    uint32_t ubRDimTailFactor_{0};
    uint32_t ubRDimTailFactorAlign_{0};
    uint32_t ubRDimTailTail_{0};
    uint32_t xBufElemNum_{0};
    uint32_t tailFactorAlign_{0};
};

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1, bool EN_DGAMMA = true, bool EN_DBETA = true>
class BatchNormGradExt2InferDgammaDbetaSplitR1 {
public:
    __aicore__ inline BatchNormGradExt2InferDgammaDbetaSplitR1(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(__gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* mean, __gm__ uint8_t* var,
                                __gm__ uint8_t* gamma, __gm__ uint8_t* dx, __gm__ uint8_t* dgamma,
                                __gm__ uint8_t* dbeta, __gm__ uint8_t* workspace,
                                const BatchNormGradExt2InferTilingData* tilingData)
    {
        r1Dim_ = tilingData->baseTilingData.r1Dim;
        aDim_ = tilingData->baseTilingData.aDim;
        r0Dim_ = tilingData->baseTilingData.r0Dim;
        epsilon = tilingData->baseTilingData.epsilon;
        blockNum_ = tilingData->blockNum;
        rAlign_ = tilingData->ubRDimFactor;

        aTailCoreNum_ = tilingData->tailBlockNum;
        ubRDimLoopNum_ = tilingData->ubRDimLoopNum;

        nFactor_ = tilingData->ubRDimFactor / r0Dim_;
        tailNFactor_ = r1Dim_ % nFactor_ == 0 ? nFactor_ : r1Dim_ % nFactor_;
        ubRDimTailLoopNum_ = CeilDiv(r1Dim_, nFactor_) - ubRDimLoopNum_;

        gmOffset_ = GetBlockIdx() < tilingData->tailBlockNum ?
                        GetBlockIdx() * tilingData->tailBlockDim :
                        tilingData->tailBlockNum * tilingData->tailBlockDim +
                            (GetBlockIdx() - tilingData->tailBlockNum) * tilingData->formerBlockDim;

        aDimLoopNum_ = GetBlockIdx() < tilingData->tailBlockNum ? tilingData->tailBlockDim : tilingData->formerBlockDim;

        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dy) + gmOffset_ * r0Dim_);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(x) + gmOffset_ * r0Dim_);
        meanGm_.SetGlobalBuffer((__gm__ float*)(mean) + gmOffset_);
        varGm_.SetGlobalBuffer((__gm__ float*)(var) + gmOffset_);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(gamma) + gmOffset_);
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

        // 与 SplitR0 同一判据：R(=r1Dim*r0Dim) 很短、A 比 R 长且权重 fp32 时，dgamma/dbeta 改成
        // "lane = 通道"（复用 SplitR0 类里的同一份实现）。tiling 若来自内置算子包，subTilingKey 仍会
        // 选 910001（本类），所以这个判断必须放在 kernel 侧，不能只依赖 host 路由。
        useChannelLane_ = EN_DGAMMA && EN_DBETA && IsSameType<WEIGHT_TYPE, float>::value &&
                          (static_cast<int64_t>(r1Dim_) * r0Dim_ <= static_cast<int64_t>(VL_FP32)) &&
                          (aDim_ >= r1Dim_ * r0Dim_) && (aDimLoopNum_ > 0);
        if (useChannelLane_) {
            dyRaw_ = dy;
            xRaw_ = x;
            meanRaw_ = mean;
            varRaw_ = var;
            gammaRaw_ = gamma;
            dxRaw_ = dx;
            dgammaRaw_ = dgamma;
            dbetaRaw_ = dbeta;
            wsRaw_ = workspace;
            tilingPtr_ = tilingData;
            return;
        }

        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyBufSize_);
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, xBufSize_);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(varInQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, meanBufSize_);
        pipe_->InitBuffer(compAccBuf_, BatchNormGradExt2::FOUR * VL_FP32 * sizeof(float));
    }

    __aicore__ inline void PreProcess(uint64_t ni)
    {
        if constexpr (EN_DGAMMA) {
            dGammaLocal_ = dgammaOutQue_.template AllocTensor<float>();
        }
        if constexpr (EN_DBETA) {
            dBetaLocal_ = dbetaOutQue_.template AllocTensor<float>();
        }
        InitCompAccS1();
        if constexpr (EN_DGAMMA) {
            meanLocal_ = meanInQue_.template AllocTensor<float>();
            PrepareMean(meanLocal_, ni, 1);
            meanInQue_.EnQue(meanLocal_);

            varLocal_ = varInQue_.template AllocTensor<float>();
            PrepareRstd(varLocal_, ni, 1);
            varInQue_.EnQue(varLocal_);

            meanLocal_ = meanInQue_.template DeQue<float>();
            varLocal_ = varInQue_.template DeQue<float>();
        }
    }

    __aicore__ inline void PostProcess(uint64_t ni)
    {
        // PostProcess
        ReSaveDGammaDBeta(dGammaLocal_, dBetaLocal_);

        if constexpr (EN_DBETA) {
            dbetaOutQue_.EnQue(dBetaLocal_);
            LocalTensor<WEIGHT_TYPE> dBetaLocal = dbetaOutQue_.template DeQue<WEIGHT_TYPE>();
            CopyOutDbeta(dBetaLocal, ni, 1);
            dbetaOutQue_.FreeTensor(dBetaLocal);
        }

        if constexpr (EN_DGAMMA) {
            dgammaOutQue_.EnQue(dGammaLocal_);
            LocalTensor<WEIGHT_TYPE> dGammaLocal = dgammaOutQue_.template DeQue<WEIGHT_TYPE>();
            CopyOutDgamma(dGammaLocal, ni, 1);
            dgammaOutQue_.FreeTensor(dGammaLocal);

            meanInQue_.FreeTensor(meanLocal_);
            varInQue_.FreeTensor(varLocal_);
        }
    }

    __aicore__ inline void FinalizeChannelS1()
    {
        __local_mem__ float* acc = CompAccAddrS1();
        __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal_.GetPhyAddr();
        __local_mem__ float* varAddr = (__local_mem__ float*)varLocal_.GetPhyAddr();
        __local_mem__ float* dgammaAddr = (__local_mem__ float*)dGammaLocal_.GetPhyAddr();
        __local_mem__ float* dbetaAddr = (__local_mem__ float*)dBetaLocal_.GetPhyAddr();
        __VEC_SCOPE__
        {
            MaskReg one = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            RegTensor<float> s0Hi, s0Lo, s1Hi, s1Lo, meanReg, varReg, rstdReg, dgFinal, dbFinal;
            MicroAPI::DataCopy(s0Hi, acc);
            MicroAPI::DataCopy(s0Lo, acc + VL_FP32);
            MicroAPI::DataCopy(s1Hi, acc + DIGIT_TWO * VL_FP32);
            MicroAPI::DataCopy(s1Lo, acc + DIGIT_THREE * VL_FP32);
            // 同 FinalizeChannel：R 方向总元素数 r1Dim * r0Dim，只有前 min(R, VL) 条 lane 非零。
            const uint32_t laneCount = Comp::CompLaneBound(static_cast<int64_t>(r1Dim_) * r0Dim_);
            Comp::CompReduceLanes(s0Hi, s0Lo, all, laneCount);
            if constexpr (EN_DGAMMA) {
                Comp::CompReduceLanes(s1Hi, s1Lo, all, laneCount);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanAddr);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(varReg, varAddr);
                Comp::RstdExactF32(varReg, rstdReg, all, epsilon);
                Comp::CompFinalizeDgammaDbeta(s1Hi, s1Lo, s0Hi, s0Lo, meanReg, rstdReg, one, dgFinal, dbFinal);
                StoreOneTensor<float>(dgammaAddr, dgFinal, one, 0);
                if constexpr (EN_DBETA) {
                    StoreOneTensor<float>(dbetaAddr, dbFinal, one, 0);
                }
            } else {
                Comp::CompRound(dbFinal, s0Hi, s0Lo, one);
                if constexpr (EN_DBETA) {
                    StoreOneTensor<float>(dbetaAddr, dbFinal, one, 0);
                }
            }
        }
    }

    __aicore__ inline __local_mem__ float* CompAccAddrS1()
    {
        return (__local_mem__ float*)compAccBuf_.Get<float>().GetPhyAddr();
    }

    __aicore__ inline void InitCompAccS1()
    {
        __local_mem__ float* acc = CompAccAddrS1();
        __VEC_SCOPE__
        {
            RegTensor<float> zero;
            MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::Duplicate(zero, 0.0f, all);
            MicroAPI::DataCopy(acc, zero, all);
            MicroAPI::DataCopy(acc + VL_FP32, zero, all);
            MicroAPI::DataCopy(acc + DIGIT_TWO * VL_FP32, zero, all);
            MicroAPI::DataCopy(acc + DIGIT_THREE * VL_FP32, zero, all);
        }
    }

    template <typename T>
    __aicore__ inline void AccumulateLanesS1(const __local_mem__ T* addr, int64_t baseOffset, int64_t len,
                                             RegTensor<float>& laneHi, RegTensor<float>& laneLo)
    {
        RegTensor<float> v;
        uint32_t count = static_cast<uint32_t>(len);
        for (uint16_t chunk = 0; chunk < static_cast<uint16_t>(len); chunk += static_cast<uint16_t>(VL_FP32)) {
            MaskReg mask = MicroAPI::UpdateMask<float>(count);
            LoadOneTensor<T>(addr, v, mask, baseOffset + chunk);
            Comp::CompAdd<T>(laneHi, laneLo, v, mask);
        }
    }

    template <typename T>
    __aicore__ inline void AccumulateProdBandS1(const __local_mem__ T* dyAddr, const __local_mem__ T* xAddr,
                                                int64_t baseOffset, int64_t len, RegTensor<float>& laneHi,
                                                RegTensor<float>& laneLo)
    {
        RegTensor<float> dy, x, ph, pl;
        uint32_t count = static_cast<uint32_t>(len);
        for (uint16_t chunk = 0; chunk < static_cast<uint16_t>(len); chunk += static_cast<uint16_t>(VL_FP32)) {
            MaskReg mask = MicroAPI::UpdateMask<float>(count);
            LoadOneTensor<T>(dyAddr, dy, mask, baseOffset + chunk);
            LoadOneTensor<T>(xAddr, x, mask, baseOffset + chunk);
            if constexpr (IsSameType<T, float>::value) {
                Comp::CompAddProduct<T>(laneHi, laneLo, x, dy, mask);
            } else {
                MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(ph, x, dy, mask);
                Comp::CompAdd<T>(laneHi, laneLo, ph, mask);
            }
        }
    }

    __aicore__ inline void AccumDbetaS1(const LocalTensor<DY_TYPE>& dyLocal, int64_t baseOffset, int64_t len,
                                        uint64_t unused)
    {
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal.GetPhyAddr();
        __local_mem__ float* acc = CompAccAddrS1();
        __VEC_SCOPE__
        {
            MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            RegTensor<float> laneHi, laneLo;
            MicroAPI::DataCopy(laneHi, acc);
            MicroAPI::DataCopy(laneLo, acc + VL_FP32);
            AccumulateLanesS1<DY_TYPE>(dyAddr, baseOffset, len, laneHi, laneLo);
            MicroAPI::DataCopy(acc, laneHi, all);
            MicroAPI::DataCopy(acc + VL_FP32, laneLo, all);
        }
    }

    __aicore__ inline void AccumDgammaS1(const LocalTensor<DY_TYPE>& dyLocal, const LocalTensor<DY_TYPE>& xLocal,
                                         int64_t baseOffset, int64_t len, uint64_t unused)
    {
        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal.GetPhyAddr();
        __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xLocal.GetPhyAddr();
        __local_mem__ float* acc = CompAccAddrS1() + DIGIT_TWO * VL_FP32;
        __VEC_SCOPE__
        {
            MaskReg all = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            RegTensor<float> laneHi, laneLo;
            MicroAPI::DataCopy(laneHi, acc);
            MicroAPI::DataCopy(laneLo, acc + VL_FP32);
            AccumulateProdBandS1<DY_TYPE>(dyAddr, xAddr, baseOffset, len, laneHi, laneLo);
            MicroAPI::DataCopy(acc, laneHi, all);
            MicroAPI::DataCopy(acc + VL_FP32, laneLo, all);
        }
    }

    __aicore__ inline void ProcessMainBlock(uint64_t ni, uint64_t basicBlockIdx, uint64_t r1Factor)
    {
        uint64_t offset = ni * r0Dim_ + basicBlockIdx * nFactor_ * r0Dim_ * aDim_;
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        PrepareInDy(dyLocal, 0, offset, r1Factor);
        dyInQue_.EnQue(dyLocal);
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();
        // 同上：S0 也是 dgamma 的输入，(EN_DGAMMA, !EN_DBETA) 也必须累加 S0。
        if constexpr (EN_DBETA || EN_DGAMMA) {
            AccumDbetaS1(dyLocal, 0, r1Factor * r0Dim_, 0UL);
        }
        if constexpr (EN_DGAMMA) {
            LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
            PrepareInX(xLocal, 0, offset, r1Factor);
            xInQue_.EnQue(xLocal);
            xLocal = xInQue_.template DeQue<DY_TYPE>();
            AccumDgammaS1(dyLocal, xLocal, 0, r1Factor * r0Dim_, 0UL);
            xInQue_.FreeTensor(xLocal);
        }
        dyInQue_.FreeTensor(dyLocal);
    }

    __aicore__ inline void ProcessFoldBlock(uint64_t ni, uint64_t basicBlockIdx, uint64_t r1Factor,
                                            uint64_t r1TailFactor, uint64_t r1TailTailFactor)
    {
        uint64_t offset = ni * r0Dim_ + basicBlockIdx * nFactor_ * r0Dim_ * aDim_;
        LocalTensor<DY_TYPE> dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        PrepareInDy(dyLocal, 0, offset, r1Factor);
        uint64_t tailOffset = offset + ubRDimLoopNum_ * nFactor_ * r0Dim_ * aDim_;
        PrepareInDy(dyLocal, halfXBufOffset_, tailOffset, r1TailFactor);
        dyInQue_.EnQue(dyLocal);
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();

        // 同上：S0 也是 dgamma 的输入，(EN_DGAMMA, !EN_DBETA) 也必须累加 S0。
        if constexpr (EN_DBETA || EN_DGAMMA) {
            AccumDbetaS1(dyLocal, 0, r1Factor * r0Dim_, 0UL);
            if (r1TailFactor > 0) {
                AccumDbetaS1(dyLocal, halfXBufOffset_, r1TailFactor * r0Dim_, 0UL);
            }
        }
        if constexpr (EN_DGAMMA) {
            LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
            PrepareInX(xLocal, 0, offset, r1Factor);
            PrepareInX(xLocal, halfXBufOffset_, tailOffset, r1TailFactor);
            xInQue_.EnQue(xLocal);
            xLocal = xInQue_.template DeQue<DY_TYPE>();
            AccumDgammaS1(dyLocal, xLocal, 0, r1Factor * r0Dim_, 0UL);
            if (r1TailFactor > 0) {
                AccumDgammaS1(dyLocal, xLocal, halfXBufOffset_, r1TailFactor * r0Dim_, 0UL);
            }
            xInQue_.FreeTensor(xLocal);
        }
        dyInQue_.FreeTensor(dyLocal);

        uint64_t foldOffset = ni * r0Dim_ + basicBlockIdx * nFactor_ * r0Dim_ * aDim_ + r1Factor * r0Dim_ * aDim_;
        dyLocal = dyInQue_.template AllocTensor<DY_TYPE>();
        PrepareInDy(dyLocal, 0, foldOffset, r1Factor);
        uint64_t foldTailOffset = foldOffset + ubRDimLoopNum_ * nFactor_ * r0Dim_ * aDim_;
        if (r1TailTailFactor > 0) {
            PrepareInDy(dyLocal, halfXBufOffset_, foldTailOffset, r1TailTailFactor);
        }
        dyInQue_.EnQue(dyLocal);
        dyLocal = dyInQue_.template DeQue<DY_TYPE>();

        // 同上：S0 也是 dgamma 的输入，(EN_DGAMMA, !EN_DBETA) 也必须累加 S0。
        if constexpr (EN_DBETA || EN_DGAMMA) {
            AccumDbetaS1(dyLocal, 0, r1Factor * r0Dim_, 0UL);
            if (r1TailTailFactor > 0) {
                AccumDbetaS1(dyLocal, halfXBufOffset_, r1TailTailFactor * r0Dim_, 0UL);
            }
        }
        if constexpr (EN_DGAMMA) {
            LocalTensor<DY_TYPE> xLocal = xInQue_.template AllocTensor<DY_TYPE>();
            PrepareInX(xLocal, 0, foldOffset, r1Factor);
            if (r1TailTailFactor > 0) {
                PrepareInX(xLocal, halfXBufOffset_, foldTailOffset, r1TailTailFactor);
            }
            xInQue_.EnQue(xLocal);
            xLocal = xInQue_.template DeQue<DY_TYPE>();
            AccumDgammaS1(dyLocal, xLocal, 0, r1Factor * r0Dim_, 0UL);
            if (r1TailTailFactor > 0) {
                AccumDgammaS1(dyLocal, xLocal, halfXBufOffset_, r1TailTailFactor * r0Dim_, 0UL);
            }
            xInQue_.FreeTensor(xLocal);
        }
        dyInQue_.FreeTensor(dyLocal);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= blockNum_) {
            return;
        }
        if (useChannelLane_) {
            BatchNormGradExt2InferDgammaDbetaSplitR0<DY_TYPE, WEIGHT_TYPE, BUFFER_NUM, EN_DGAMMA, EN_DBETA> op(pipe_);
            op.Init(dyRaw_, xRaw_, meanRaw_, varRaw_, gammaRaw_, dxRaw_, dgammaRaw_, dbetaRaw_, wsRaw_, tilingPtr_);
            op.Process();
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
            FinalizeChannelS1();
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
        CopyInA(rstd, varGm_, offset, a);
    }

    __aicore__ inline void PrepareInGamma(LocalTensor<WEIGHT_TYPE>& gamma, uint64_t offset, uint64_t a)
    {
        CopyInA(gamma, gammaGm_, offset, a);
    }

    __aicore__ inline void ReSaveDGammaDBeta(LocalTensor<float>& dgamma, LocalTensor<float>& dbeta)
    {
        if constexpr (IsSameType<WEIGHT_TYPE, half>::value) {
            __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgamma.GetPhyAddr();
            __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbeta.GetPhyAddr();

            __VEC_SCOPE__
            {
                MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
                if constexpr (EN_DGAMMA) {
                    MicroAPI::RegTensor<float> dgammaValue;
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dgammaValue,
                                                                                (__local_mem__ float*)(dgammaAddr));
                    StoreOneElement<WEIGHT_TYPE>(dgammaAddr, dgammaValue, pregMerge, 0);
                }
                if constexpr (EN_DBETA) {
                    MicroAPI::RegTensor<float> dbetaValue;
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dbetaValue,
                                                                                (__local_mem__ float*)(dbetaAddr));
                    StoreOneElement<WEIGHT_TYPE>(dbetaAddr, dbetaValue, pregMerge, 0);
                }
            }
        }
        return;
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
    // lane = 通道路径：委托给 SplitR0 类的同一份实现
    bool useChannelLane_{false};
    __gm__ uint8_t* dyRaw_{nullptr};
    __gm__ uint8_t* xRaw_{nullptr};
    __gm__ uint8_t* meanRaw_{nullptr};
    __gm__ uint8_t* varRaw_{nullptr};
    __gm__ uint8_t* gammaRaw_{nullptr};
    __gm__ uint8_t* dxRaw_{nullptr};
    __gm__ uint8_t* dgammaRaw_{nullptr};
    __gm__ uint8_t* dbetaRaw_{nullptr};
    __gm__ uint8_t* wsRaw_{nullptr};
    const BatchNormGradExt2InferTilingData* tilingPtr_{nullptr};
    TBuf<TPosition::VECCALC> compAccBuf_;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> varInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;

    GlobalTensor<DY_TYPE> dyGm_, xGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, varGm_;

    uint32_t r1Dim_{0};
    uint32_t aDim_{0};
    uint32_t r0Dim_{0};
    uint32_t rAlign_{0};
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
    uint32_t blockNum_{0};
    uint64_t ubRDimLoopNum_{0};
    uint64_t nFactor_{0};
    uint64_t tailNFactor_{0};
    uint64_t ubRDimTailFactor_{0};
    uint64_t ubRDimTailTailFactor_{0};

    uint64_t ubRDimTailLoopNum_{0};

    float epsilon;
    LocalTensor<float> dBetaLocal_;
    LocalTensor<float> dGammaLocal_;
    LocalTensor<float> varLocal_;
    LocalTensor<float> meanLocal_;
    LocalTensor<WEIGHT_TYPE> gammaLocal_;
};

template <typename T1, typename T2>
class BatchNormGradExt2InferDx {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline BatchNormGradExt2InferDx(TPipe* pipeIn, const BatchNormGradExt2InferDxTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
        pipe_ = pipeIn;
        epsilon = tilingData_->epsilon;
    }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR gamma, GM_ADDR runningVar, GM_ADDR dx)
    {
        dyGm_.SetGlobalBuffer((__gm__ T1*)dy);
        gammaGm_.SetGlobalBuffer((__gm__ T2*)gamma);
        runningVarGm_.SetGlobalBuffer((__gm__ T2*)runningVar);
        dxGm_.SetGlobalBuffer((__gm__ T1*)dx);

        pipe_->InitBuffer(gammaQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T2));
        pipe_->InitBuffer(runningVarQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T2));

        int64_t xShapeLen = tilingData_->tileBlockR0Len * tilingData_->tileBlockALen * tilingData_->tileBlockR1Len;
        pipe_->InitBuffer(dxQueue_, BUFFER_NUM, xShapeLen * sizeof(T1));
        pipe_->InitBuffer(dyQueue_, BUFFER_NUM, xShapeLen * sizeof(T1));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx >= tilingData_->usedCoreNums) {
            return;
        }

        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        int64_t paddingANumSizeT1 = tilingData_->tileBlockAPaddingNum * sizeof(T1) / BLOCK_SIZE;

        // pattern is [B0, A, B1]
        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curB1Idx = curIdx % tilingData_->r0Outer;
            int64_t curAIdx = (curIdx / tilingData_->r0Outer) / tilingData_->r1Outer;
            int64_t curB0Idx = (curIdx / tilingData_->r0Outer) % tilingData_->r1Outer;

            // ping、pang搬运首次或者tile块沿B轴换列时需要拷贝gamma、invstd
            // curIdx 整数倍(tilingData_->r1Outer * tilingData_->r0Outer) 表示一轮循环, 出现换行
            bool needCopy = (curIdx <= beginIdx + 1) || (curIdx % (tilingData_->r1Outer * tilingData_->r0Outer) <= 1);

            // Tile整尾块
            int64_t curTileB0Len = curB0Idx == (tilingData_->r1Outer - 1) ? tilingData_->tileBlockR1Tail :
                                                                            tilingData_->tileBlockR1Len;
            int64_t curTileALen = curAIdx == (tilingData_->aOuter - 1) ? tilingData_->tileBlockATail :
                                                                         tilingData_->tileBlockALen;
            int64_t curTileB1Len = curB1Idx == (tilingData_->r0Outer - 1) ? tilingData_->tileBlockR0Tail :
                                                                            tilingData_->tileBlockR0Len;

            int64_t ubStrideT1 = curAIdx == (tilingData_->aOuter - 1) ? paddingANumSizeT1 : 0;

            // x、y偏移一致，gamma、runningVar偏移一致
            int64_t weightOffset = curAIdx * tilingData_->tileBlockALen;
            int64_t dyOffset =
                // b0 offset
                curB0Idx * tilingData_->tileBlockR1Len * tilingData_->aDim * tilingData_->r0Dim +
                // a offset
                curAIdx * tilingData_->tileBlockALen * tilingData_->r0Dim +
                // b1 offset
                curB1Idx * tilingData_->tileBlockR0Len;

            CopyInDy(dyOffset, curTileB0Len, curTileALen, curTileB1Len, ubStrideT1);
            CopyInGammaVar(needCopy, weightOffset, curTileALen);
            Compute(curTileB0Len, curTileALen, curTileB1Len);
            CopyOutDx(dyOffset, curTileB0Len, curTileALen, curTileB1Len, ubStrideT1);
        }
    }

private:
    __aicore__ inline void CopyInDy(int64_t dyGmOffset, int64_t curTileB0Len, int64_t curTileALen, int64_t curTileB1Len,
                                    int64_t ubStrideT1)
    {
        LocalTensor<T1> dyLocal = dyQueue_.AllocTensor<T1>();
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curTileB0Len;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->aDim * tilingData_->r0Dim * sizeof(T1);
        loopParams.loop1DstStride = tilingData_->tileBlockR0Len * tilingData_->tileBlockALen * sizeof(T1);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPadExtParams<T1> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileALen;
        copyInParams.blockLen = curTileB1Len * sizeof(T1);
        copyInParams.srcStride = (tilingData_->r0Dim - curTileB1Len) * sizeof(T1);
        copyInParams.dstStride = (tilingData_->tileBlockR0Len - curTileB1Len) * sizeof(T1) / BLOCK_SIZE;
        DataCopyPad<T1, PaddingMode::Normal>(dyLocal, dyGm_[dyGmOffset], copyInParams, dataCopyPadExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        dyQueue_.EnQue(dyLocal);
    }

    __aicore__ inline void CopyInGammaVar(bool needCopy, int64_t offset, int64_t curTileALen)
    {
        LocalTensor<T2> gammaLocal = gammaQueue_.AllocTensor<T2>();
        LocalTensor<T2> varLocal = runningVarQueue_.AllocTensor<T2>();

        if (needCopy) {
            DataCopyExtParams extParam;
            extParam.blockCount = 1;

            // gamma
            extParam.blockLen = curTileALen * sizeof(T2);
            DataCopyPadExtParams<T2> padExtParam;
            padExtParam.isPad = false;
            DataCopyPad(gammaLocal, gammaGm_[offset], extParam, padExtParam);

            // runningVar
            extParam.blockLen = curTileALen * sizeof(T2);
            DataCopyPadExtParams<T2> padExtParams1;
            padExtParams1.isPad = false;
            DataCopyPad(varLocal, runningVarGm_[offset], extParam, padExtParams1);
        }

        gammaQueue_.EnQue(gammaLocal);
        runningVarQueue_.EnQue(varLocal);
    }

    __aicore__ inline void Compute(int64_t curTileB0Len, int64_t curTileALen, int64_t curTileB1Len)
    {
        LocalTensor<T1> dy = dyQueue_.DeQue<T1>();
        LocalTensor<T2> gamma = gammaQueue_.DeQue<T2>();
        LocalTensor<T2> runningVar = runningVarQueue_.DeQue<T2>();
        LocalTensor<T1> dx = dxQueue_.AllocTensor<T1>();

        __local_mem__ T1* dyLocal = (__local_mem__ T1*)dy.GetPhyAddr();
        __local_mem__ T2* gammaLocal = (__local_mem__ T2*)gamma.GetPhyAddr();
        __local_mem__ T2* varLocal = (__local_mem__ T2*)runningVar.GetPhyAddr();
        __local_mem__ T1* dxLocal = (__local_mem__ T1*)dx.GetPhyAddr();

        VFNormalize(dyLocal, gammaLocal, varLocal, dxLocal, curTileB0Len, curTileALen, curTileB1Len);

        dxQueue_.EnQue(dx);

        dyQueue_.FreeTensor<T1>(dy);
        gammaQueue_.FreeTensor<T2>(gamma);
        runningVarQueue_.FreeTensor<T2>(runningVar);
    }

    __aicore__ inline void VFNormalize(__local_mem__ T1* dyLocal, __local_mem__ T2* gammaLocal,
                                       __local_mem__ T2* varLocal, __local_mem__ T1* dxLocal, uint16_t curTileB0Len,
                                       uint16_t curTileALen, uint16_t curTileB1Len)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> dy;
            RegTensor<float> gamma;
            RegTensor<float> runningVar;
            RegTensor<float> dx;
            RegTensor<float> dxResid;
            RegTensor<float> invstd;

            MaskReg pregMask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

            uint16_t loopNum = ops::CeilDiv(curTileB1Len, VL_FP32);
            uint32_t tileBlockALenTmp = static_cast<uint32_t>(tilingData_->tileBlockALen);
            uint32_t tileBlockR0LenTmp = static_cast<uint32_t>(tilingData_->tileBlockR0Len);
            for (uint16_t i = 0; i < curTileALen; i++) {
                // loads runningVar  1->64
                LoadsTensorForDtypeT<T2>(varLocal, runningVar, pregMask, i);
                AscendC::MicroAPI::MaskReg
                    pregRstdAll7 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                Comp::RstdExactF32(runningVar, invstd, pregRstdAll7, epsilon);

                // load gamma 1->64
                LoadsTensorForDtypeT<T2>(gammaLocal, gamma, pregMask, i);
                for (uint16_t j = 0; j < curTileB0Len; j++) {
                    for (uint16_t k = 0; k < loopNum; k++) {
                        uint32_t dyOffset = (j * tileBlockALenTmp + i) * tileBlockR0LenTmp + k * VL_FP32;

                        // load dy
                        LoadOneTensor<T1>(dyLocal, dy, pregMask, dyOffset);

                        // compute：dx = (dy * scale) * rstd，次序与 golden/torch 对齐。
                        // 原来是 (dy * rstd) * scale：指令数完全相同，只是两次乘法交换，
                        // 但浮点舍入次序不同（L1_380 上旧次序有 736/3748080 个元素各差 1 个
                        // fp16 ULP，交换后该 draw 为 0；另 9 个 draw 不变差）。
                        // MulChain3Residual 同时给出 fp32 结果与 FMA 残差，收尾按"只舍一次"落盘。
                        Comp::MulChain3Residual<float>(dx, dxResid, dy, gamma, invstd, pregMask);

                        // copy out
                        StoreDxSingleRounded<T1>(dxLocal, dx, dxResid, pregMask, dyOffset);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOutDx(int64_t dxGmOffset, int64_t curTileB0Len, int64_t curTileALen,
                                     int64_t curTileB1Len, int64_t ubStrideT1)
    {
        LocalTensor<T1> dx = dxQueue_.DeQue<T1>();
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curTileB0Len;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->tileBlockR0Len * tilingData_->tileBlockALen * sizeof(T1);
        loopParams.loop1DstStride = tilingData_->aDim * tilingData_->r0Dim * sizeof(T1);
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileALen;
        copyInParams.blockLen = curTileB1Len * sizeof(T1);
        copyInParams.srcStride = (tilingData_->tileBlockR0Len - curTileB1Len) * sizeof(T1) / BLOCK_SIZE;
        copyInParams.dstStride = (tilingData_->r0Dim - curTileB1Len) * sizeof(T1);
        DataCopyPad<T1, PaddingMode::Normal>(dxGm_[dxGmOffset], dx, copyInParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        dxQueue_.FreeTensor(dx);
    }

private:
    const BatchNormGradExt2InferDxTilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dyQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> gammaQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> runningVarQueue_;

    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dxQueue_;

    GlobalTensor<T1> dyGm_;
    GlobalTensor<T2> gammaGm_;
    GlobalTensor<T2> runningVarGm_;
    GlobalTensor<T1> dxGm_;

    float epsilon;
};

template <typename T1, typename T2, bool SPLIT_R0>
class BatchNormGradExt2Infer {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline BatchNormGradExt2Infer(){};

    __aicore__ inline BatchNormGradExt2Infer(TPipe* pipeIn, const BatchNormGradExt2InferTilingData* tilingDataIn)
    {
        pipe_ = pipeIn;
        tilingData_ = tilingDataIn;
        enableDx = tilingData_->enableDx;
        enableDgamma = tilingData_->enableDgamma;
        enableDbeta = tilingData_->enableDbeta;
    }

    __aicore__ inline void Process(GM_ADDR dy, GM_ADDR x, GM_ADDR gamma, GM_ADDR save_mean, GM_ADDR runningVar,
                                   GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR usrWorkspace)
    {
        if (enableDx) {
            BatchNormGradExt2InferDx<T1, T2> opDx(pipe_, &tilingData_->baseTilingData);
            opDx.Init(dy, gamma, runningVar, dx);
            opDx.Process();
            PipeBarrier<PIPE_ALL>();
            pipe_->Reset();
        }
        if constexpr (SPLIT_R0) { // split R0
            if (enableDgamma && enableDbeta) {
                BatchNormGradExt2InferDgammaDbetaSplitR0<T1, T2, BUFFER_NUM, true, true> opDgammaDbeta(pipe_);
                opDgammaDbeta.Init(dy, x, save_mean, runningVar, gamma, dx, dgamma, dbeta, usrWorkspace, tilingData_);
                opDgammaDbeta.Process();
            } else if (enableDgamma && !enableDbeta) {
                BatchNormGradExt2InferDgammaDbetaSplitR0<T1, T2, BUFFER_NUM, true, false> opDgammaDbeta(pipe_);
                opDgammaDbeta.Init(dy, x, save_mean, runningVar, gamma, dx, dgamma, dbeta, usrWorkspace, tilingData_);
                opDgammaDbeta.Process();
            } else if (!enableDgamma && enableDbeta) {
                BatchNormGradExt2InferDgammaDbetaSplitR0<T1, T2, BUFFER_NUM, false, true> opDgammaDbeta(pipe_);
                opDgammaDbeta.Init(dy, x, save_mean, runningVar, gamma, dx, dgamma, dbeta, usrWorkspace, tilingData_);
                opDgammaDbeta.Process();
            }
        } else {
            if (enableDgamma && enableDbeta) {
                BatchNormGradExt2InferDgammaDbetaSplitR1<T1, T2, BUFFER_NUM, true, true> opDgammaDbeta(pipe_);
                opDgammaDbeta.Init(dy, x, save_mean, runningVar, gamma, dx, dgamma, dbeta, usrWorkspace, tilingData_);
                opDgammaDbeta.Process();
            } else if (enableDgamma && !enableDbeta) {
                BatchNormGradExt2InferDgammaDbetaSplitR1<T1, T2, BUFFER_NUM, true, false> opDgammaDbeta(pipe_);
                opDgammaDbeta.Init(dy, x, save_mean, runningVar, gamma, dx, dgamma, dbeta, usrWorkspace, tilingData_);
                opDgammaDbeta.Process();
            } else if (!enableDgamma && enableDbeta) {
                BatchNormGradExt2InferDgammaDbetaSplitR1<T1, T2, BUFFER_NUM, false, true> opDgammaDbeta(pipe_);
                opDgammaDbeta.Init(dy, x, save_mean, runningVar, gamma, dx, dgamma, dbeta, usrWorkspace, tilingData_);
                opDgammaDbeta.Process();
            }
        }
    }

private:
    const BatchNormGradExt2InferTilingData* tilingData_;
    TPipe* pipe_;

    bool enableDx;
    bool enableDgamma;
    bool enableDbeta;
};
} // namespace BatchNormGradExt2

#endif
