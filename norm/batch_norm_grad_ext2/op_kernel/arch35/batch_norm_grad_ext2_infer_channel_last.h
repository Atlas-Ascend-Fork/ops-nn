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
 * \file batch_norm_grad_ext2_infer_channel_last.h
 * \brief
 */

#ifndef BATCH_NORM_GRAD_EXT2_INFER_CHANNEL_LAST_H
#define BATCH_NORM_GRAD_EXT2_INFER_CHANNEL_LAST_H

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "batch_norm_grad_ext2_common.h"
#include "batch_norm_grad_ext2_comp_sum.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace BatchNormGradExt2 {
using namespace AscendC;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MemType;

using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::StoreDist;

// ---------------------------------------------------------------------------------------------
// 900000（infer channel last）dgamma/dbeta 归约的向量化开关。
//
//   0（默认，关闭）：lane = 通道（A），R 在 lane 内串行累加。A = 2~3 的小通道形状只有 2~3 条 lane
//                   有数据（64 条 lane 利用率 3~5%），而且每元素都是一条 20 余条指令的串行依赖链，
//                   实测 32 cycle/元素。开关关闭时代码路径与改造前逐位一致（新路径整体落在
//                   `if constexpr (kRVecLanes != 0)` 里，关闭时不参与编译）。
//   1（打开）      ：转置搬运后用 lane = R 的归约，通道退化成 lane 内串行循环（见
//                   BatchNormGradExt2InferChannelLastRecompute::AccumulateRVecLanes）。
//
// 翻转方式：把下面这个常量改成 1，重新编包即可（UB 账本、finalize、接口均不动）。
// 打开后并非所有 shape 都走新路径：host 还会按 R/C 形状门控写 tiling_data 的 rVecEnable，
// 只有 (R >= 750000) || (C >= 25) 才转置，R 短或通道少的 tile 仍走老路径：
//   bash build.sh --opkernel --ops=batch_norm_grad_ext2 --soc=ascend950 -j32
//   bash build.sh --ops=batch_norm_grad_ext2 --soc=ascend950 -j32 --vendor_name=batch_norm_grad_ext2
// ---------------------------------------------------------------------------------------------
static constexpr int32_t kRVecLanes = 0;

// 转置后每个通道占用的 r 跨距（元素个数）。必须向上对齐到整向量：末段的整向量 load 不能带掩码
// （见 LoadOneTensor），会读到 curRLen 之后的 [curRLen, rAlign) 区间，故这块也必须在缓冲区内。
// 上界：curRLen <= binAddRFactorStg1 = 128（host 侧 DEFAULT_BIN_ADD_R_FACTOR）=> rAlign <= 128。
__aicore__ inline int64_t RVecAlignLength(int64_t rLen)
{
    const int64_t vl = static_cast<int64_t>(VL_FP32);
    return ops::CeilDiv(rLen, vl) * vl;
}

template <typename T1, typename T2>
class BatchNormGradExt2InferChannelLastDx {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline BatchNormGradExt2InferChannelLastDx(){};

    __aicore__ inline BatchNormGradExt2InferChannelLastDx(
        const BatchNormGradExt2InferChannelLastDxTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }
    __aicore__ inline void InitStg0(GM_ADDR dy, GM_ADDR weight, GM_ADDR runningVar, GM_ADDR dx, TPipe* pipeIn)
    {
        pipe_ = pipeIn;

        dyGm_.SetGlobalBuffer((__gm__ T1*)dy);
        gammaGm_.SetGlobalBuffer((__gm__ T2*)weight);
        runningVarGm_.SetGlobalBuffer((__gm__ T2*)runningVar);
        dxGm_.SetGlobalBuffer((__gm__ T1*)dx);

        pipe_->InitBuffer(gammaQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T2));
        pipe_->InitBuffer(runningVarQueue_, BUFFER_NUM, tilingData_->tileBlockALen * sizeof(T2));

        int64_t xShapeLen = tilingData_->tileBlockBLen * tilingData_->tileBlockALen;
        pipe_->InitBuffer(dxQueue_, BUFFER_NUM, xShapeLen * sizeof(T1));
        pipe_->InitBuffer(dyQueue_, BUFFER_NUM, xShapeLen * sizeof(T1));
    }

    __aicore__ inline void ProcessStg0()
    {
        if (GetBlockIdx() >= tilingData_->usedCoreNums) {
            return;
        }
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        int64_t paddingANumSizeT1 = tilingData_->tileBlockAPaddingNum * sizeof(T1) / BLOCK_SIZE;

        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curColumIdx = ops::FloorDiv(curIdx, tilingData_->bOuter);
            int64_t curRowIdx = curIdx % tilingData_->bOuter;

            // ping、pang搬运首次或者tile块沿B轴换列时需要拷贝gamma、runningVar
            bool needCopy = (curIdx <= beginIdx + 1) || (curRowIdx <= 1);

            int64_t curTileBLen = curRowIdx == (tilingData_->bOuter - 1) ? tilingData_->tileBlockBTail :
                                                                           tilingData_->tileBlockBLen;

            int64_t curTileALen = tilingData_->tileBlockALen;
            int64_t ubStrideT1 = 0;
            int64_t ubStrideFloat = 0;

            if (curColumIdx == (tilingData_->aOuter - 1)) {
                curTileALen = tilingData_->tileBlockATail;
                ubStrideT1 = paddingANumSizeT1;
            }

            // x、y偏移一致，gamma、runningVar偏移一致
            int64_t weightOffset = curColumIdx * tilingData_->tileBlockALen;
            int64_t dyOffset = curRowIdx * tilingData_->aDim * tilingData_->tileBlockBLen + weightOffset;

            CopyInDy(dyOffset, curTileBLen, curTileALen, ubStrideT1);
            CopyInGammaVar(needCopy, weightOffset, curTileALen);
            Compute(curTileBLen, curTileALen);
            CopyOutDx(dyOffset, curTileBLen, curTileALen, ubStrideT1);
        }
    }

private:
    __aicore__ inline void CopyInDy(int64_t dyGmOffset, int64_t curTileBLen, int64_t curTileALen, int64_t ubStrideT1)
    {
        LocalTensor<T1> dyLocal = dyQueue_.AllocTensor<T1>();

        DataCopyExtParams extParam;
        extParam.blockLen = curTileALen * sizeof(T1);
        extParam.srcStride = (tilingData_->aDim - curTileALen) * sizeof(T1);
        extParam.dstStride = ubStrideT1;
        extParam.blockCount = curTileBLen;

        DataCopyPadExtParams<T1> padExtParam;
        padExtParam.isPad = false;

        DataCopyPad(dyLocal, dyGm_[dyGmOffset], extParam, padExtParam);
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

    __aicore__ inline void Compute(int64_t curTileBLen, int64_t curTileALen)
    {
        LocalTensor<T1> dy = dyQueue_.DeQue<T1>();
        LocalTensor<T2> gamma = gammaQueue_.DeQue<T2>();
        LocalTensor<T2> runningVar = runningVarQueue_.DeQue<T2>();
        LocalTensor<T1> dx = dxQueue_.AllocTensor<T1>();

        __local_mem__ T1* dyLocal = (__local_mem__ T1*)dy.GetPhyAddr();
        __local_mem__ T2* gammaLocal = (__local_mem__ T2*)gamma.GetPhyAddr();
        __local_mem__ T2* varLocal = (__local_mem__ T2*)runningVar.GetPhyAddr();
        __local_mem__ T1* dxLocal = (__local_mem__ T1*)dx.GetPhyAddr();

        VFNormalize(dyLocal, gammaLocal, varLocal, dxLocal, curTileBLen, curTileALen);

        dxQueue_.EnQue(dx);

        dyQueue_.FreeTensor<T1>(dy);
        gammaQueue_.FreeTensor<T2>(gamma);
        runningVarQueue_.FreeTensor<T2>(runningVar);
    }

    __aicore__ inline void VFNormalize(__local_mem__ T1* dyLocal, __local_mem__ T2* gammaLocal,
                                       __local_mem__ T2* varLocal, __local_mem__ T1* dxLocal, uint16_t curTileBLen,
                                       uint16_t curTileALen)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> dy;
            RegTensor<float> gamma;
            RegTensor<float> runningVar;
            RegTensor<float> dx;
            RegTensor<float> dxResid;
            RegTensor<float> invstd;

            MaskReg pregMaskFp32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            uint32_t tileBlockALenTmp = static_cast<uint32_t>(tilingData_->tileBlockALen);
            float epsilonTmp = tilingData_->epsilon;
            uint16_t loopNum = ops::CeilDiv(curTileALen, VL_FP32);
            for (uint16_t i = 0; i < loopNum; i++) {
                uint32_t offset = i * VL_FP32;

                // load runningVar, gamma
                LoadOneTensor<T2>(varLocal, runningVar, pregMaskFp32, offset);
                AscendC::MicroAPI::MaskReg
                    pregRstdAll1 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                Comp::RstdExactF32(runningVar, invstd, pregRstdAll1, epsilonTmp);

                LoadOneTensor<T2>(gammaLocal, gamma, pregMaskFp32, offset);

                for (uint16_t j = 0; j < curTileBLen; j++) {
                    uint32_t dyOffset = j * tileBlockALenTmp + offset;
                    // load dy
                    LoadOneTensor<T1>(dyLocal, dy, pregMaskFp32, dyOffset);

                    // compute
                    // 乘法顺序与 golden（fp64 单次舍入的 dy*scale*rstd）一致：先乘 scale 再乘 rstd。
                    // 反过来写（先乘 rstd）会在 scale<1、rstd>1 时让 dy*rstd 先溢出成 +-Inf，
                    // 而精确结果仍然有限（rangebound_highpre_331：约 1e7 个元素被误判为 -Inf）。
                    // MulChain3Residual 同时给出 fp32 结果与 FMA 残差，收尾时按"只舍一次"落盘
                    // （fp16 输出下，两次舍入只在 fp32 值恰好落在 fp16 中点时与 golden 差 1 ULP）。
                    Comp::MulChain3Residual<float>(dx, dxResid, dy, gamma, invstd, pregMaskFp32);

                    // copy out
                    StoreDxSingleRounded<T1>(dxLocal, dx, dxResid, pregMaskFp32, dyOffset);
                }
            }
        }
    }

    __aicore__ inline void CopyOutDx(int64_t dxGmOffset, int64_t curTileBLen, int64_t curTileALen, int64_t ubStrideT1)
    {
        LocalTensor<T1> dx = dxQueue_.DeQue<T1>();

        DataCopyExtParams extParams;
        extParams.blockLen = curTileALen * sizeof(T1);
        extParams.srcStride = ubStrideT1;
        extParams.dstStride = (tilingData_->aDim - curTileALen) * sizeof(T1);
        extParams.blockCount = curTileBLen;

        DataCopyPad(dxGm_[dxGmOffset], dx, extParams);

        dxQueue_.FreeTensor(dx);
    }

private:
    const BatchNormGradExt2InferChannelLastDxTilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dyQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> gammaQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> runningVarQueue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dxQueue_;

    GlobalTensor<T1> dyGm_;
    GlobalTensor<T2> gammaGm_;
    GlobalTensor<T2> runningVarGm_;

    GlobalTensor<T1> dxGm_;
};

template <typename T1, typename T2, int BUFFER_NUM = 1, bool EN_DGAMMA = true, bool EN_DBETA = true>
class BatchNormGradExt2InferChannelLastRecompute {
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
    // compAccBuf_ 的字数：dbeta/dgamma 各 hi/lo/lo2 三字；fp32 另加 satS0/satS1 两个
    // 缩放伴生字（2^-80 / 2^-160），fp16 归约不可能饱和故不分配（与 host UB 账本一致）。
    static constexpr int64_t COMP_ACC_WORDS = IsSameType<T1, float>::value ? 8 : 6;
    static constexpr int64_t SAT_WORD_S0 = 6;
    static constexpr int64_t SAT_WORD_S1 = 7;

public:
    __aicore__ inline BatchNormGradExt2InferChannelLastRecompute(){};

    __aicore__ inline BatchNormGradExt2InferChannelLastRecompute(
        TPipe* pipeIn, const BatchNormGradExt2InferChannelLastTilingData* tilingDataIn)
    {
        pipe_ = pipeIn;
        epsilon = tilingDataIn->dxTilingData.epsilon;
        tilingData_ = tilingDataIn;
        enableDx = tilingData_->enableDx;
        enableDgamma = tilingData_->enableDgamma;
        enableDbeta = tilingData_->enableDbeta;
    }

    __aicore__ inline void Process(GM_ADDR dy, GM_ADDR x, GM_ADDR save_mean, GM_ADDR runningVar, GM_ADDR dgamma,
                                   GM_ADDR dbeta, GM_ADDR usrWorkspace)
    {
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx >= tilingData_->usedCoreNumsStg1) {
            return;
        }
        InitStage1(dy, x, save_mean, runningVar, dgamma, dbeta);
        int64_t beginIdx = blockIdx * tilingData_->aOuterPerCoreStg1;
        int64_t endIdx = beginIdx + tilingData_->aOuterPerCoreStg1;
        endIdx = endIdx > tilingData_->aOuterStg1 ? tilingData_->aOuterStg1 : endIdx;

        // [R, A]
        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curALen = curIdx == (tilingData_->aOuterStg1 - 1) ? tilingData_->aTailStg1 :
                                                                        tilingData_->aInnerStg1;

            int64_t meanOffset = curIdx * tilingData_->aInnerStg1;
            uint16_t vlLoopNum = ops::CeilDiv(curALen, static_cast<int64_t>(VL_FP32));

            ComputeDbetaAndDgamma(curALen, meanOffset, vlLoopNum);
        }
    }

    __aicore__ inline void InitStage1(__gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* mean,
                                      __gm__ uint8_t* runningVar, __gm__ uint8_t* dgamma, __gm__ uint8_t* dbeta)
    {
        dyGm_.SetGlobalBuffer((__gm__ T1*)(dy));
        xGm_.SetGlobalBuffer((__gm__ T1*)(x));
        meanGm_.SetGlobalBuffer((__gm__ float*)(mean));
        varGm_.SetGlobalBuffer((__gm__ float*)(runningVar));
        dgammaGm_.SetGlobalBuffer((__gm__ float*)(dgamma));
        dbetaGm_.SetGlobalBuffer((__gm__ float*)(dbeta));

        int64_t dyLen = tilingData_->binAddRFactorStg1 * tilingData_->aInnerStg1;
        pipe_->InitBuffer(dyQueue_, BUFFER_NUM, dyLen * sizeof(T1));
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, dyLen * sizeof(T1));

        int64_t meanLen = tilingData_->aInnerStg1 * sizeof(float);
        // RstdExactF32Ub loads whole vectors, so the variance buffer is rounded up to whole
        // vectors; otherwise the tail chunk load reads up to VL-1 floats past the buffer.
        int64_t varLen = ops::CeilDiv(tilingData_->aInnerStg1, static_cast<int64_t>(VL_FP32)) * VL_FP32 * sizeof(float);
        pipe_->InitBuffer(meanQueue_, BUFFER_NUM, meanLen);
        pipe_->InitBuffer(varQueue_, BUFFER_NUM, varLen);
        pipe_->InitBuffer(dbetaQueue_, BUFFER_NUM, meanLen);
        pipe_->InitBuffer(dgammaQueue_, BUFFER_NUM, meanLen);

        // double-fp32 累加器：dbeta/dgamma 各 hi/lo/lo2 三个字（见 comp_sum.h CompAddWide 的说明：
        // 两字累加在长同号归约上残留数 ULP，第三字记 lo 的残差）。
        // fp32 输入再加一对缩放伴生累加器 satS0/satS1（2^-80 / 2^-160），供 finalize 在
        // 主累加器饱和成 +-Inf 时恢复 T 的符号与量级（rangebound_highpre_331 的 dscale 需要）；
        // fp16 的归约不可能饱和（|x*dy| <= 2^32、R <= 2^32），故不占这两段 UB。
        aInnerAligned_ = ops::CeilDiv(tilingData_->aInnerStg1, static_cast<int64_t>(VL_FP32)) * VL_FP32;
        pipe_->InitBuffer(compAccBuf_, COMP_ACC_WORDS * aInnerAligned_ * sizeof(float));
    }

    __aicore__ inline __local_mem__ float* CompAccBase()
    {
        return (__local_mem__ float*)compAccBuf_.Get<float>().GetPhyAddr();
    }

    __aicore__ inline void InitCompAcc()
    {
        __local_mem__ float* base = CompAccBase();
        uint32_t totalVec = static_cast<uint32_t>(COMP_ACC_WORDS * aInnerAligned_ / VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> zero;
            MaskReg all = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::Duplicate(zero, 0.0f, all);
            for (uint16_t i = 0; i < static_cast<uint16_t>(totalVec); ++i) {
                AscendC::MicroAPI::DataCopy(base + i * VL_FP32, zero, all);
            }
        }
    }

    __aicore__ inline void ComputeDbetaAndDgamma(uint32_t curALen, int64_t meanOffset, uint16_t vlLoopNum)
    {
        InitCompAcc();
        if constexpr (EN_DGAMMA) {
            CopyInMeanAndVar(meanOffset, curALen);
            mean_ = meanQueue_.DeQue<float>();
            var_ = varQueue_.DeQue<float>();
            Comp::RstdExactF32Ub((__local_mem__ float*)var_.GetPhyAddr(), static_cast<uint32_t>(curALen), epsilon);
        }

        int64_t blockOffset = tilingData_->aDimStg1 * tilingData_->binAddRFactorStg1;
        for (int64_t basicBlockIdx = 0; basicBlockIdx < tilingData_->binAddBasicBlockLoopStg1; basicBlockIdx++) {
            ProcessMainBlock(tilingData_->binAddRFactorStg1, curALen, meanOffset + basicBlockIdx * blockOffset,
                             vlLoopNum);
            if (basicBlockIdx < tilingData_->binAddMainFoldCountStg1) {
                ProcessFoldBlock(tilingData_->binAddRFactorStg1, curALen,
                                 meanOffset + (basicBlockIdx + tilingData_->binAddBasicBlockLoopStg1) * blockOffset,
                                 vlLoopNum);
            } else if (basicBlockIdx == tilingData_->binAddMainFoldCountStg1 && tilingData_->binAddRTailStg1 > 0 &&
                       tilingData_->binAddRTailStg1 != tilingData_->binAddRFactorStg1) {
                ProcessFoldBlock(tilingData_->binAddRTailStg1, curALen,
                                 meanOffset + (basicBlockIdx + tilingData_->binAddBasicBlockLoopStg1) * blockOffset,
                                 vlLoopNum);
            }
        }

        if (tilingData_->binAddBasicBlockLoopStg1 == 0) {
            ProcessMainBlock(tilingData_->binAddRTailStg1, curALen, meanOffset, vlLoopNum);
        }
        LocalTensor<T2> dbetaLocal;
        LocalTensor<T2> dgammaLocal;
        if constexpr (EN_DBETA) {
            dbetaLocal = dbetaQueue_.AllocTensor<T2>();
        }
        if constexpr (EN_DGAMMA) {
            dgammaLocal = dgammaQueue_.AllocTensor<T2>();
        }
        FinalizeDbetaDgamma(curALen, vlLoopNum, dbetaLocal, dgammaLocal);
        // mean_/var_ 必须活到 FinalizeDbetaDgamma 之后：该函数仍通过 GetPhyAddr() 读取这两块
        // 缓冲（mean/rstd）。提前释放会让队列把同一 ping-pong 缓冲交给第 +2 个 aInnerStg1 块
        // 的 CopyInMeanAndVar，finalize 读到尚未做 RstdExactF32Ub 变换的原始方差（以及该块的
        // mean），dscale 被放大 1e3~1e4 倍。900000 的散点坏通道即由此产生。
        if constexpr (EN_DGAMMA) {
            meanQueue_.FreeTensor(mean_);
            varQueue_.FreeTensor(var_);
        }
        if constexpr (EN_DBETA) {
            dbetaQueue_.EnQue(dbetaLocal);
        }
        if constexpr (EN_DGAMMA) {
            dgammaQueue_.EnQue(dgammaLocal);
        }
        CopyOutDbetaAndDgamma(meanOffset, curALen);
    }

    __aicore__ inline void FinalizeDbetaDgamma(const uint32_t curALen, const uint16_t vlLoopNum,
                                               const LocalTensor<T2>& dbetaOutLocal,
                                               const LocalTensor<T2>& dgammaOutLocal)
    {
        __local_mem__ float* base = CompAccBase();
        int64_t aAligned = aInnerAligned_;
        __local_mem__ float* dbHi = base;
        __local_mem__ float* dbLo = base + aAligned;
        __local_mem__ float* dbLo2 = base + DIGIT_TWO * aAligned;
        __local_mem__ float* dgHi = base + DIGIT_THREE * aAligned;
        __local_mem__ float* dgLo = base + 4 * aAligned;
        __local_mem__ float* dgLo2 = base + 5 * aAligned;
        __local_mem__ float* dbSat = base + SAT_WORD_S0 * aAligned;
        __local_mem__ float* dgSat = base + SAT_WORD_S1 * aAligned;
        __local_mem__ float* mean = (__local_mem__ float*)mean_.GetPhyAddr();
        __local_mem__ float* rstd = (__local_mem__ float*)var_.GetPhyAddr();
        __local_mem__ float* dbOut = (__local_mem__ float*)dbetaOutLocal.GetPhyAddr();
        __local_mem__ float* dgOut = (__local_mem__ float*)dgammaOutLocal.GetPhyAddr();

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = curALen;
            RegTensor<float> s0Hi, s0Lo, s0Lo2, s1Hi, s1Lo, s1Lo2, meanReg, rstdReg, dbFinal, dgFinal;
            RegTensor<float> satS0Reg, satS1Reg;
            for (uint16_t j = 0; j < vlLoopNum; ++j) {
                pregMask = UpdateMask<float>(sreg);
                uint32_t offset = j * VL_FP32;
                AscendC::MicroAPI::DataCopy(s0Hi, dbHi + offset);
                AscendC::MicroAPI::DataCopy(s0Lo, dbLo + offset);
                AscendC::MicroAPI::DataCopy(s0Lo2, dbLo2 + offset);
                if constexpr (EN_DGAMMA) {
                    AscendC::MicroAPI::DataCopy(s1Hi, dgHi + offset);
                    AscendC::MicroAPI::DataCopy(s1Lo, dgLo + offset);
                    AscendC::MicroAPI::DataCopy(s1Lo2, dgLo2 + offset);
                    Comp::CompFoldLo2(s0Lo, s0Lo2, pregMask);
                    Comp::CompFoldLo2(s1Lo, s1Lo2, pregMask);
                    LoadOneTensor<float>(mean, meanReg, pregMask, offset);
                    LoadOneTensor<float>(rstd, rstdReg, pregMask, offset);
                    if constexpr (IsSameType<T1, float>::value) {
                        // 主累加器饱和成 +-Inf（fp64 参考仍有限）时，用缩放伴生累加器恢复
                        // T = S1 - mean*S0 的符号与量级；非饱和 lane 由 CompFinalizeDgammaDbetaSat
                        // 内部的 Select 保留原 2 字结果，逐位不变。
                        AscendC::MicroAPI::DataCopy(satS0Reg, dbSat + offset);
                        AscendC::MicroAPI::DataCopy(satS1Reg, dgSat + offset);
                        Comp::CompFinalizeDgammaDbetaSat(s1Hi, s1Lo, s0Hi, s0Lo, satS0Reg, satS1Reg, meanReg, rstdReg,
                                                         pregMask, dgFinal, dbFinal);
                    } else {
                        Comp::CompFinalizeDgammaDbeta(s1Hi, s1Lo, s0Hi, s0Lo, meanReg, rstdReg, pregMask, dgFinal,
                                                      dbFinal);
                    }
                    if constexpr (EN_DBETA) {
                        StoreOneTensor<float>(dbOut, dbFinal, pregMask, offset);
                    }
                    StoreOneTensor<float>(dgOut, dgFinal, pregMask, offset);
                } else {
                    Comp::CompFoldLo2(s0Lo, s0Lo2, pregMask);
                    Comp::CompRound(dbFinal, s0Hi, s0Lo, pregMask);
                    StoreOneTensor<float>(dbOut, dbFinal, pregMask, offset);
                }
            }
        }
    }

    // 逐元素补偿累加：lane = 通道，R 轴在 lane 内累加，不需要跨 lane 归约
    __aicore__ inline void ProcessMainBlock(const int64_t curRLen, const int64_t curALen, const int64_t gmOffset,
                                            uint16_t vlLoopNum)
    {
        if constexpr (kRVecLanes != 0) {
            // host 形状门控（tiling_data.h 的 rVecEnable：(R >= 750000) || (C >= 25)）。
            // R 太短时每 tile 的转置 DMA + 跨 lane 归约固定开销摊不薄，逐位沿用下面的老路径。
            if (tilingData_->rVecEnable != 0) {
                ProcessMainBlockRVec(curRLen, curALen, gmOffset);
                return;
            }
        }
        CopyInDyAndX(gmOffset, curRLen, curALen);

        dyMain_ = dyQueue_.DeQue<T1>();
        if constexpr (EN_DGAMMA) {
            xMain_ = xQueue_.DeQue<T1>();
        }

        __local_mem__ T1* dy = (__local_mem__ T1*)dyMain_.GetPhyAddr();
        __local_mem__ T1* x = (__local_mem__ T1*)xMain_.GetPhyAddr();
        __local_mem__ float* base = CompAccBase();
        int64_t aAligned = aInnerAligned_;
        __local_mem__ float* dbHi = base;
        __local_mem__ float* dbLo = base + aAligned;
        __local_mem__ float* dbLo2 = base + DIGIT_TWO * aAligned;
        __local_mem__ float* dgHi = base + DIGIT_THREE * aAligned;
        __local_mem__ float* dgLo = base + 4 * aAligned;
        __local_mem__ float* dgLo2 = base + 5 * aAligned;
        __local_mem__ float* dbSat = base + SAT_WORD_S0 * aAligned;
        __local_mem__ float* dgSat = base + SAT_WORD_S1 * aAligned;

        uint16_t rLoopTimes = static_cast<uint16_t>(curRLen);
        uint32_t rLoopStride = tilingData_->aInnerStg1;

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = curALen;
            uint16_t loopNum = vlLoopNum;

            RegTensor<float> dyReg;
            RegTensor<float> xReg;
            RegTensor<float> dBetaHi, dBetaLo, dBetaLo2, dGammaHi, dGammaLo, dGammaLo2, prodHi;
            // 缩放伴生累加器（只有 fp32 输入可能饱和，fp16 下这些寄存器不参与运算）
            RegTensor<float> satS0Reg, satS1Reg;

            for (uint16_t j = 0; j < loopNum; ++j) {
                pregMask = UpdateMask<float>(sreg);
                uint32_t aOffset = j * VL_FP32;
                AscendC::MicroAPI::DataCopy(dBetaHi, dbHi + aOffset);
                AscendC::MicroAPI::DataCopy(dBetaLo, dbLo + aOffset);
                AscendC::MicroAPI::DataCopy(dBetaLo2, dbLo2 + aOffset);
                AscendC::MicroAPI::DataCopy(dGammaHi, dgHi + aOffset);
                AscendC::MicroAPI::DataCopy(dGammaLo, dgLo + aOffset);
                AscendC::MicroAPI::DataCopy(dGammaLo2, dgLo2 + aOffset);
                if constexpr (IsSameType<T1, float>::value && EN_DGAMMA) {
                    AscendC::MicroAPI::DataCopy(satS0Reg, dbSat + aOffset);
                    AscendC::MicroAPI::DataCopy(satS1Reg, dgSat + aOffset);
                }

                for (uint16_t i = 0; i < rLoopTimes; ++i) {
                    uint32_t dyOffset = i * rLoopStride + aOffset;
                    LoadOneTensor<T1>(dy, dyReg, pregMask, dyOffset);
                    Comp::CompAddWide<T1>(dBetaHi, dBetaLo, dBetaLo2, dyReg, pregMask);
                    if constexpr (EN_DGAMMA) {
                        LoadOneTensor<T1>(x, xReg, pregMask, dyOffset);
                        if constexpr (IsSameType<T1, float>::value) {
                            Comp::CompAddProductWide<T1>(dGammaHi, dGammaLo, dGammaLo2, xReg, dyReg, pregMask);
                            // satS0 += dy*2^-80 ; satS1 += (x*2^-80)*(dy*2^-80)（+5 条向量指令/元素，
                            // 只在主累加器会饱和时用于恢复 T 的符号与量级）
                            Comp::CompAddScaled(satS0Reg, satS1Reg, xReg, dyReg, pregMask);
                        } else {
                            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(prodHi, xReg, dyReg, pregMask);
                            Comp::CompAddWide<T1>(dGammaHi, dGammaLo, dGammaLo2, prodHi, pregMask);
                        }
                    }
                }

                AscendC::MicroAPI::DataCopy(dbHi + aOffset, dBetaHi, pregMask);
                AscendC::MicroAPI::DataCopy(dbLo + aOffset, dBetaLo, pregMask);
                AscendC::MicroAPI::DataCopy(dbLo2 + aOffset, dBetaLo2, pregMask);
                AscendC::MicroAPI::DataCopy(dgHi + aOffset, dGammaHi, pregMask);
                AscendC::MicroAPI::DataCopy(dgLo + aOffset, dGammaLo, pregMask);
                AscendC::MicroAPI::DataCopy(dgLo2 + aOffset, dGammaLo2, pregMask);
                if constexpr (IsSameType<T1, float>::value && EN_DGAMMA) {
                    AscendC::MicroAPI::DataCopy(dbSat + aOffset, satS0Reg, pregMask);
                    AscendC::MicroAPI::DataCopy(dgSat + aOffset, satS1Reg, pregMask);
                }
            }
        }

        if constexpr (EN_DGAMMA) {
            xQueue_.FreeTensor(xMain_);
        }
        dyQueue_.FreeTensor(dyMain_);
    }

    // fold 分支与 main 同构：直接累加到同一组 dd 累加器
    __aicore__ inline void ProcessFoldBlock(const int64_t curRLen, const int64_t curALen, const int64_t gmOffset,
                                            uint16_t vlLoopNum)
    {
        ProcessMainBlock(curRLen, curALen, gmOffset, vlLoopNum);
    }

    // ============================================================================================
    // lane = R 的向量化归约（kRVecLanes != 0 时才参与编译）
    //
    // 改造前后唯一的区别是"谁在 lane 上"：
    //   关闭：lane = 通道 A，R 在 lane 内串行。A = 2~3 时 64 条 lane 只用 2~3 条（3~5%），
    //         每元素还是一条 CompAddWide/CompAddProductWide 的串行链（实测 32 cycle/元素）。
    //   打开：lane = R（一条向量吃满 64 个 r，R 按 64 分段并行），通道退化成 lane 内串行循环。
    // 累加算子完全不换：每个 (通道, r 段) 仍是同一套三字 CompAddWide / CompAddProductWide /
    // CompAddScaled；一个通道的 r 段累加完 CompFoldLo2 -> CompReduceLanes 收到 lane0，再按
    // CompAddPairWide 并入 UB 里该通道的持久三字累加器。finalize / copy-out / host tiling 全不动。
    // ============================================================================================

    // 转置布局每个通道占 rAlign 个元素，缓冲区仍是 InitStage1 里按 binAddRFactorStg1 * aInnerStg1
    // 分配的那一块（dyLen，dy/x 各一块）。tiling 保证 curALen <= aInnerStg1 且
    // rAlign <= binAddRFactorStg1（host 侧 binAddRFactorStg1 恒为 128，见
    // batch_norm_grad_ext2_tiling_infer_channel_last.cpp:DEFAULT_BIN_ADD_R_FACTOR），所以恒不越界。
    // 走哪条路径由 host 的 rVecEnable 决定（见 ProcessMainBlock）。
    __aicore__ inline void ProcessMainBlockRVec(const int64_t curRLen, const int64_t curALen, const int64_t gmOffset)
    {
        CopyInDyAndXTransposed(gmOffset, curRLen, curALen);

        dyMain_ = dyQueue_.DeQue<T1>();
        if constexpr (EN_DGAMMA) {
            xMain_ = xQueue_.DeQue<T1>();
        }

        AccumulateRVecLanes(curRLen, curALen);

        if constexpr (EN_DGAMMA) {
            xQueue_.FreeTensor(xMain_);
        }
        dyQueue_.FreeTensor(dyMain_);
    }

    // 转置搬运：UB 里变成 [通道][rAlign]，r 方向连续。搬运字节数与老路径完全一样（同一块 tile），
    // 只是把 stride 描述从"行跨距 aInnerStg1、行内连续"换成"通道跨距 rAlign、通道内连续"。
    __aicore__ inline void CopyInDyAndXTransposed(const int64_t gmOffset, const int64_t curRLen, const int64_t curALen)
    {
        const int64_t rAlign = RVecAlignLength(curRLen);

        LocalTensor<T1> dy = dyQueue_.AllocTensor<T1>();
        CopyInTransposeTile<T1>(dy, dyGm_[gmOffset], curRLen, curALen, tilingData_->aDimStg1, rAlign);
        dyQueue_.EnQue(dy);

        if constexpr (EN_DGAMMA) {
            LocalTensor<T1> x = xQueue_.AllocTensor<T1>();
            CopyInTransposeTile<T1>(x, xGm_[gmOffset], curRLen, curALen, tilingData_->aDimStg1, rAlign);
            xQueue_.EnQue(x);
        }
    }

    // 逐通道把本块的 r 方向归约做完并并入 UB 里的持久累加器。
    // 输入 UB 布局：[channel][rAlign]（CopyInDyAndXTransposed）；累加器布局与老路径完全一致
    // （dbHi/dbLo/dbLo2/dgHi/dgLo/dgLo2/dbSat/dgSat，每通道一个字）。
    __aicore__ inline void AccumulateRVecLanes(const int64_t curRLen, const int64_t curALen)
    {
        __local_mem__ T1* dy = (__local_mem__ T1*)dyMain_.GetPhyAddr();
        __local_mem__ T1* x = (__local_mem__ T1*)xMain_.GetPhyAddr();
        __local_mem__ float* base = CompAccBase();
        const int64_t aAligned = aInnerAligned_;
        __local_mem__ float* dbHi = base;
        __local_mem__ float* dbLo = base + aAligned;
        __local_mem__ float* dbLo2 = base + DIGIT_TWO * aAligned;
        __local_mem__ float* dgHi = base + DIGIT_THREE * aAligned;
        __local_mem__ float* dgLo = base + 4 * aAligned;
        __local_mem__ float* dgLo2 = base + 5 * aAligned;
        __local_mem__ float* dbSat = base + SAT_WORD_S0 * aAligned;
        __local_mem__ float* dgSat = base + SAT_WORD_S1 * aAligned;

        const int64_t rAlign = RVecAlignLength(curRLen);
        const uint16_t rLoopTimes = static_cast<uint16_t>(ops::CeilDiv(curRLen, static_cast<int64_t>(VL_FP32)));
        // 只有前 min(curRLen, 64) 条 lane 收过数据（见 comp_sum.h CompReduceBeginWidth）。
        const uint32_t laneCount = Comp::CompLaneBound(curRLen);

        __VEC_SCOPE__
        {
            MaskReg all = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg one = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            MaskReg pregMask;
            RegTensor<float> dyReg, xReg, prodHi;
            RegTensor<float> s0Hi, s0Lo, s0Lo2, s1Hi, s1Lo, s1Lo2, satS0, satS1;
            RegTensor<float> satS0Red, satS1Red;

            // 通道循环只到 curALen：aInnerStg1 - curALen 条补齐通道（fp32 C=2/3 -> 8，fp16 -> 16）
            // 一条都不能进这个循环。理由有两条：
            //   1) 补齐通道的行区间在 UB 里没有数据（转置搬运只写 [0, curRLen)），读出来的是上一
            //      块残留的随机字节，一旦乘进 satS1 会让 lp64 参考里有限的通道变成 Inf/NaN；
            //   2) 即便把它们当 +0 累加，也会以 0+0=+0 的形式"写过"这些 lane：CompReduceBeginWidth
            //      与 CompAddWide 的 MERGING 低字都以"未激活的 lane 恰好是 Duplicate 播下的 +0、
            //      从未被部分掩码写过"为前提，lane 计数（CompLaneBound）也只统计真实数据。
            //      一旦补位参与，lo 的 +0/-0 语义与 lane 计数都会漂移，最终 CompRound 的舍入方向
            //      随之改变，而 dbeta/dgamma 又是逐位对拍的。
            for (uint16_t c = 0; c < static_cast<uint16_t>(curALen); ++c) {
                const uint32_t cOffset = static_cast<uint32_t>(c) * static_cast<uint32_t>(rAlign);
                AscendC::MicroAPI::Duplicate(s0Hi, 0.0f, all);
                AscendC::MicroAPI::Duplicate(s0Lo, 0.0f, all);
                AscendC::MicroAPI::Duplicate(s0Lo2, 0.0f, all);
                AscendC::MicroAPI::Duplicate(s1Hi, 0.0f, all);
                AscendC::MicroAPI::Duplicate(s1Lo, 0.0f, all);
                AscendC::MicroAPI::Duplicate(s1Lo2, 0.0f, all);
                if constexpr (IsSameType<T1, float>::value && EN_DGAMMA) {
                    AscendC::MicroAPI::Duplicate(satS0, 0.0f, all);
                    AscendC::MicroAPI::Duplicate(satS1, 0.0f, all);
                }

                // r 方向按 64 一段并行；末段用 UpdateMask 把 r >= curRLen 的 lane 掩掉
                // （那部分是转置搬运没写过的残留数据，掩掉后才保持 Duplicate 播下的 +0）。
                uint32_t sreg = static_cast<uint32_t>(curRLen);
                for (uint16_t j = 0; j < rLoopTimes; ++j) {
                    pregMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    const uint32_t offset = cOffset + j * VL_FP32;
                    LoadOneTensor<T1>(dy, dyReg, pregMask, offset);
                    Comp::CompAddWide<T1>(s0Hi, s0Lo, s0Lo2, dyReg, pregMask);
                    if constexpr (EN_DGAMMA) {
                        LoadOneTensor<T1>(x, xReg, pregMask, offset);
                        if constexpr (IsSameType<T1, float>::value) {
                            Comp::CompAddProductWide<T1>(s1Hi, s1Lo, s1Lo2, xReg, dyReg, pregMask);
                            Comp::CompAddScaled(satS0, satS1, xReg, dyReg, pregMask);
                        } else {
                            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(prodHi, xReg, dyReg, pregMask);
                            Comp::CompAddWide<T1>(s1Hi, s1Lo, s1Lo2, prodHi, pregMask);
                        }
                    }
                }

                // 三字 -> 两字（CompFoldLo2）后跨 lane 归约到 lane0，再并入该通道的持久三字累加器
                Comp::CompFoldLo2(s0Lo, s0Lo2, all);
                Comp::CompReduceLanes(s0Hi, s0Lo, all, laneCount);
                MergeChannelWide<T1>(dbHi + c, dbLo + c, dbLo2 + c, s0Hi, s0Lo, one);
                if constexpr (EN_DGAMMA) {
                    Comp::CompFoldLo2(s1Lo, s1Lo2, all);
                    Comp::CompReduceLanes(s1Hi, s1Lo, all, laneCount);
                    MergeChannelWide<T1>(dgHi + c, dgLo + c, dgLo2 + c, s1Hi, s1Lo, one);
                    if constexpr (IsSameType<T1, float>::value) {
                        // 缩放伴生累加器只用于恢复饱和通道的符号与量级，普通 ReduceSum 足够
                        // （与 full_load_regbase.h:284-287 的 satS0Red/satS1Red 处理一致）。
                        MicroAPI::ReduceSum(satS0Red, satS0, all);
                        MicroAPI::ReduceSum(satS1Red, satS1, all);
                        AccumulateSatChannel(dbSat + c, satS0Red, one);
                        AccumulateSatChannel(dgSat + c, satS1Red, one);
                    }
                }
            }
        }
    }

    // 把一个通道的块内归约结果（lane0 的 (hi, lo) 对）并入 UB 里的持久三字累加器。
    // 三字都用"广播单元素 load / 首元素 store"，只有 lane0 参与，不会碰到相邻通道的字
    // （相邻通道的字是独立通道的累加器，越界写会直接串通道）。
    template <typename T>
    __aicore__ inline void MergeChannelWide(__local_mem__ float* hiAddr, __local_mem__ float* loAddr,
                                            __local_mem__ float* lo2Addr, RegTensor<float>& laneHi,
                                            RegTensor<float>& laneLo, MaskReg& one)
    {
        RegTensor<float> accHi, accLo, accLo2;
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accHi, hiAddr);
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accLo, loAddr);
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(accLo2, lo2Addr);
        Comp::CompAddPairWide<T>(accHi, accLo, accLo2, laneHi, laneLo, one);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(hiAddr, accHi, one);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(loAddr, accLo, one);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(lo2Addr, accLo2, one);
    }

    // 缩放伴生累加器按通道累加：accent += lane0(reduced)
    __aicore__ inline void AccumulateSatChannel(__local_mem__ float* addr, RegTensor<float>& laneValue, MaskReg& one)
    {
        RegTensor<float> acc;
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(acc, addr);
        MicroAPI::Add<float, MicroAPI::MaskMergeMode::MERGING>(acc, acc, laneValue, one);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(addr, acc, one);
    }

    __aicore__ inline void CopyInDyAndX(int64_t gmOffset, int64_t curRLen, uint32_t curALen)
    {
        DataCopyExtParams params;
        params.blockCount = curRLen;
        params.blockLen = curALen * sizeof(T1);
        params.srcStride = (tilingData_->aDimStg1 - curALen) * sizeof(T1);
        params.dstStride = (tilingData_->aInnerStg1 * sizeof(T1) - params.blockLen) / BLOCK_SIZE;
        DataCopyPadExtParams<T1> padParams;
        padParams.isPad = false;

        LocalTensor<T1> dy = dyQueue_.AllocTensor<T1>();
        DataCopyPad(dy, dyGm_[gmOffset], params, padParams);
        dyQueue_.EnQue(dy);

        if constexpr (EN_DGAMMA) {
            LocalTensor<T1> x = xQueue_.AllocTensor<T1>();
            DataCopyPad(x, xGm_[gmOffset], params, padParams);
            xQueue_.EnQue(x);
        }
    }

    __aicore__ inline void CopyInMeanAndVar(int64_t gmOffset, uint32_t curALen)
    {
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = curALen * sizeof(T2);
        params.srcStride = (tilingData_->aDimStg1 - curALen) * sizeof(T2);
        params.dstStride = (tilingData_->aInnerStg1 * sizeof(T2) - params.blockLen) / BLOCK_SIZE;
        DataCopyPadExtParams<T2> padParams;
        padParams.isPad = false;

        LocalTensor<T2> mean = meanQueue_.AllocTensor<T2>();
        DataCopyPad(mean, meanGm_[gmOffset], params, padParams);
        meanQueue_.EnQue(mean);

        LocalTensor<T2> var = varQueue_.AllocTensor<T2>();
        DataCopyPad(var, varGm_[gmOffset], params, padParams);
        varQueue_.EnQue(var);
    }

    __aicore__ inline void CopyOutDbetaAndDgamma(int64_t gmOffset, uint32_t curALen)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = curALen * sizeof(T2);
        copyOutParams.srcStride = (tilingData_->aInnerStg1 - curALen) * sizeof(T2) / BLOCK_SIZE;
        copyOutParams.dstStride = 0;
        if constexpr (EN_DBETA) {
            LocalTensor<T2> dbeta = dbetaQueue_.DeQue<T2>();
            DataCopyPad<T2, PaddingMode::Normal>(dbetaGm_[gmOffset], dbeta, copyOutParams);
            dbetaQueue_.FreeTensor(dbeta);
        }
        if constexpr (EN_DGAMMA) {
            LocalTensor<T2> dgamma = dgammaQueue_.DeQue<T2>();
            DataCopyPad<T2, PaddingMode::Normal>(dgammaGm_[gmOffset], dgamma, copyOutParams);
            dgammaQueue_.FreeTensor(dgamma);
        }
    }

private:
    const BatchNormGradExt2InferChannelLastTilingData* tilingData_;
    TPipe* pipe_;

    float epsilon;
    bool enableDx;
    bool enableDgamma;
    bool enableDbeta;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dyQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> xQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> meanQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> varQueue_;

    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dbetaQueue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dgammaQueue_;

    // b16

    // bin add

    TBuf<TPosition::VECCALC> compAccBuf_;
    int64_t aInnerAligned_{0};

    LocalTensor<T1> xMain_;
    LocalTensor<T1> dyMain_;

    LocalTensor<float> mean_;
    LocalTensor<float> var_;

    GlobalTensor<T1> dyGm_;
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> meanGm_;
    GlobalTensor<T2> varGm_;
    GlobalTensor<T2> dbetaGm_;
    GlobalTensor<T2> dgammaGm_;
};

// ================================================================================================
// 900000（infer channel last）dgamma/dbeta 的 R 轴多核切分实现（aDim < VL_FP32 时启用）。
//
// 旧实现按 A（通道）轴切核：每核只拿 8 个通道、向量只有 8/64 条 lane 有效，A = 2~29 的形状在
// 72 核机器上只用 1~4 个核（aOuter = ceil(A/8)），dgamma/dbeta 因此比 dx 慢一个数量级。
// L1_008（A=29, R=141984）dgamma/dbeta 约 4.4ms；L1_202（A=2, R=5.46M）整例 22.5ms。
//
// 本实现把并行度放到 R（行）轴上：
//   * 每核负责一段连续的 R 行，全部 aDim 个通道一起归约；
//   * UB 内用 NDDMA 转置成 [channel][rAlign]，lane = R（64 条 lane 全用满），每个通道的 R 段仍走
//     同一套三字补偿累加（CompAddWide / CompAddProductWide / CompAddScaled），逐元素算子与旧实现
//     完全一致；
//   * 块内先 CompFoldLo2 -> CompReduceLanes 收 lane0，再并入该核的 (hi, lo, lo2) 累加器；
//   * 每核把 6 个分量（dbHi/dbLo、dgHi/dgLo、satS0/satS1）写到 workspace[word][core][channel]；
//   * SyncAll 后 0 号核跨核补偿合并（CompAddPairWide）并 Finalize
//     （dgamma = rstd * (S1 - mean * S0)，dbeta = S0，单次舍入）。
//
// 只改了归约的划分与 lane 方向，语义仍是"精确累加 + 单次舍入"：核内与跨核两级都是三字/两字补偿，
// 最终结果的差异只来自 ~2^-70 相对量级的残差分组（不进入 fp32 尾数）。
//
// 为什么参数在 kernel 侧推导而不是用 tiling 字段：本算子包与 CANN 内置算子同类型时，tiling 由内置
// 算子包先注册并生效（kernel 却是 vendor 的），因此 kernel 只能依赖两版 tiling 都一定正确的字段
// （aDimStg1、dx 阶段的 B 轴切分、enable*、workspace 公式），切分参数在本文件内重新推导。
// ================================================================================================
static constexpr int64_t RS_PARTIAL_WORDS = 6; // dbHi dbLo dgHi dgLo satS0 satS1
static constexpr int64_t RS_UB_BUDGET = 180 * 1024;
static constexpr int64_t RS_MAX_TILE_ROWS = 1024;

template <typename T1, typename T2>
class BatchNormGradExt2InferChannelLastRSplit {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline BatchNormGradExt2InferChannelLastRSplit() = default;

    __aicore__ inline BatchNormGradExt2InferChannelLastRSplit(
        TPipe* pipeIn, const BatchNormGradExt2InferChannelLastTilingData* tilingDataIn)
    {
        pipe_ = pipeIn;
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Process(GM_ADDR dy, GM_ADDR x, GM_ADDR saveMean, GM_ADDR runningVar, GM_ADDR dgamma,
                                   GM_ADDR dbeta, GM_ADDR usrWorkspace)
    {
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        dyGm_.SetGlobalBuffer((__gm__ T1*)dy);
        xGm_.SetGlobalBuffer((__gm__ T1*)x);
        meanGm_.SetGlobalBuffer((__gm__ float*)saveMean);
        varGm_.SetGlobalBuffer((__gm__ float*)runningVar);
        dgammaGm_.SetGlobalBuffer((__gm__ float*)dgamma);
        dbetaGm_.SetGlobalBuffer((__gm__ float*)dbeta);
        wsGm_.SetGlobalBuffer((__gm__ float*)usrWorkspace);

        ComputePlan();
        if (rows_ > 0) {
            InitAccumBuffers();
            Accumulate();
            StorePartials();
        }
        PipeBarrier<PIPE_ALL>();
        // 所有 block 都必须参与（blockDim 由 tiling 决定，可能大于本实现实际启用的核数）。
        SyncAll();
        if (blockIdx_ == 0) {
            pipe_->Reset();
            InitReduceBuffers();
            ReduceAndFinalize();
        }
    }

private:
    __aicore__ inline void ComputePlan()
    {
        aDim_ = tilingData_->aDimStg1;
        if (aDim_ < 1) {
            aDim_ = 1;
        }
        // R = R1 * R0（本模板已保证 r0Dim == 1），由 dx 阶段的 B 轴切分反推：
        // bOuter 个块，前 bOuter-1 块各 tileBlockBLen 行，最后一块 tileBlockBTail 行。
        const int64_t bOuter = tilingData_->dxTilingData.bOuter;
        rTotal_ = (bOuter - 1) * tilingData_->dxTilingData.tileBlockBLen + tilingData_->dxTilingData.tileBlockBTail;
        if (rTotal_ < 0) {
            rTotal_ = 0;
        }
        blockNum_ = static_cast<int64_t>(GetBlockNum());

        // workspace 预算与 GetWorkspaceSize 一致（blockNum * aDim * 4 个字）。本实现每核每通道需要
        // RS_PARTIAL_WORDS 个字，故最多可用 blockNum * 4 / 6 个核（72 核 -> 48 核）。
        int64_t rCores = (blockNum_ * 4) / RS_PARTIAL_WORDS;
        if (rCores > blockNum_) {
            rCores = blockNum_;
        }
        if (rCores < 1) {
            rCores = 1;
        }
        rCores_ = rCores;
        rowsPerCore_ = (rTotal_ + rCores - 1) / rCores;
        if (rowsPerCore_ < 1) {
            rowsPerCore_ = 1;
        }
        rBegin_ = blockIdx_ * rowsPerCore_;
        rows_ = 0;
        if (rBegin_ < rTotal_) {
            rows_ = rTotal_ - rBegin_;
            if (rows_ > rowsPerCore_) {
                rows_ = rowsPerCore_;
            }
        }
        // 有 R 任务的核数：rTotal < rCores * rowsPerCore 时尾部若干核没有数据（不写 workspace），
        // 0 号核跨核合并时只能读这些核的 partial。
        activeCores_ = (rTotal_ + rowsPerCore_ - 1) / rowsPerCore_;
        if (activeCores_ > rCores_) {
            activeCores_ = rCores_;
        }
        if (activeCores_ < 1) {
            activeCores_ = 1;
        }
        // 32B 对齐（UB 上的向量/搬运都按 32B 取址）
        aDimAligned_ = ops::CeilAlign(aDim_, static_cast<int64_t>(BLOCK_SIZE) / static_cast<int64_t>(sizeof(float)));
        aDimPad64_ = ops::CeilAlign(aDim_, static_cast<int64_t>(VL_FP32));
        aDimPad_ = ops::CeilAlign(aDim_, static_cast<int64_t>(BLOCK_SIZE) / static_cast<int64_t>(sizeof(float)));
    }

    __aicore__ inline void InitAccumBuffers()
    {
        // dy/x 两路、各 BUFFER_NUM 个 buffer
        const int64_t bytesPerRow = static_cast<int64_t>(BUFFER_NUM) * 2 * aDim_ * sizeof(T1);
        int64_t rowsMax = (RS_UB_BUDGET * 85 / 100) / bytesPerRow;
        int64_t rChunk = rowsMax / VL_FP32 * VL_FP32;
        if (rChunk > RS_MAX_TILE_ROWS) {
            rChunk = RS_MAX_TILE_ROWS;
        }
        if (rChunk < VL_FP32) {
            rChunk = VL_FP32;
        }
        rChunk_ = rChunk;
        const int64_t tileBytes = rChunk_ * aDim_ * sizeof(T1);
        pipe_->InitBuffer(dyQueue_, BUFFER_NUM, tileBytes);
        pipe_->InitBuffer(xQueue_, BUFFER_NUM, tileBytes);
        // 核内持久累加器：[word][channel]，word = dbHi/dbLo/dbLo2/dgHi/dgLo/dgLo2/satS0/satS1。
        // fp16 不使用 sat 两个字（归约不会饱和），但它俩保持在 0，跨核/最终汇总时按 0 处理。
        compAccLen_ = ops::CeilAlign(8 * aDimAligned_, static_cast<int64_t>(VL_FP32));
        pipe_->InitBuffer(compAccBuf_, compAccLen_ * sizeof(float));
        pipe_->InitBuffer(partOutQueue_, 1, compAccLen_ * sizeof(float));
        __local_mem__ float* base = (__local_mem__ float*)compAccBuf_.Get<float>().GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> zero;
            MaskReg all = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::Duplicate(zero, 0.0f, all);
            for (uint16_t i = 0; i < static_cast<uint16_t>(compAccLen_ / VL_FP32); ++i) {
                AscendC::MicroAPI::DataCopy(base + i * VL_FP32, zero, all);
            }
        }
    }

    __aicore__ inline void Accumulate()
    {
        int64_t done = 0;
        while (done < rows_) {
            int64_t curRows = rows_ - done;
            if (curRows > rChunk_) {
                curRows = rChunk_;
            }
            const int64_t gmOffset = (rBegin_ + done) * aDim_;
            const int64_t rAlign = ops::CeilAlign(curRows, static_cast<int64_t>(VL_FP32));

            LocalTensor<T1> dy = dyQueue_.AllocTensor<T1>();
            CopyInTransposeTile<T1>(dy, dyGm_[gmOffset], curRows, aDim_, aDim_, rAlign);
            dyQueue_.EnQue(dy);
            LocalTensor<T1> x = xQueue_.AllocTensor<T1>();
            CopyInTransposeTile<T1>(x, xGm_[gmOffset], curRows, aDim_, aDim_, rAlign);
            xQueue_.EnQue(x);

            ComputeTile(curRows, rAlign);
            done += curRows;
        }
    }

    __aicore__ inline void ComputeTile(int64_t curRows, int64_t rAlignIn)
    {
        LocalTensor<T1> dyT = dyQueue_.DeQue<T1>();
        LocalTensor<T1> xT = xQueue_.DeQue<T1>();
        __local_mem__ T1* dy = (__local_mem__ T1*)dyT.GetPhyAddr();
        __local_mem__ T1* x = (__local_mem__ T1*)xT.GetPhyAddr();
        LocalTensor<float> accT = compAccBuf_.Get<float>();
        __local_mem__ float* acc = (__local_mem__ float*)accT.GetPhyAddr();

        const uint32_t rAlign = static_cast<uint32_t>(rAlignIn);
        const uint16_t chunkNum = static_cast<uint16_t>((curRows + VL_FP32 - 1) / VL_FP32);
        const uint32_t laneCount = Comp::CompLaneBound(curRows);
        const uint16_t aDim = static_cast<uint16_t>(aDim_);
        const uint32_t aDimAligned = static_cast<uint32_t>(aDimAligned_);
        const uint32_t dbHi = 0;
        const uint32_t dbLo = 1 * aDimAligned;
        const uint32_t dbLo2 = 2 * aDimAligned;
        const uint32_t dgHi = 3 * aDimAligned;
        const uint32_t dgLo = 4 * aDimAligned;
        const uint32_t dgLo2 = 5 * aDimAligned;
        const uint32_t sat0 = 6 * aDimAligned;
        const uint32_t sat1 = 7 * aDimAligned;

        __VEC_SCOPE__
        {
            MaskReg all = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg one = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            MaskReg pregMask;
            RegTensor<float> dyReg, xReg, prodHi;
            RegTensor<float> s0Hi, s0Lo, s0Lo2, s1Hi, s1Lo, s1Lo2, satS0, satS1, satRed;
            RegTensor<float> accHi, accLo, accLo2;

            for (uint16_t c = 0; c < aDim; ++c) {
                const uint32_t cOffset = static_cast<uint32_t>(c) * rAlign;
                AscendC::MicroAPI::Duplicate(s0Hi, 0.0f, all);
                AscendC::MicroAPI::Duplicate(s0Lo, 0.0f, all);
                AscendC::MicroAPI::Duplicate(s0Lo2, 0.0f, all);
                if constexpr (IsSameType<T1, float>::value) {
                    AscendC::MicroAPI::Duplicate(satS0, 0.0f, all);
                }
                AscendC::MicroAPI::Duplicate(s1Hi, 0.0f, all);
                AscendC::MicroAPI::Duplicate(s1Lo, 0.0f, all);
                AscendC::MicroAPI::Duplicate(s1Lo2, 0.0f, all);
                if constexpr (IsSameType<T1, float>::value) {
                    AscendC::MicroAPI::Duplicate(satS1, 0.0f, all);
                }

                uint32_t sreg = static_cast<uint32_t>(curRows);
                for (uint16_t j = 0; j < chunkNum; ++j) {
                    pregMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    const uint32_t offset = cOffset + static_cast<uint32_t>(j) * VL_FP32;
                    LoadOneTensor<T1>(dy, dyReg, pregMask, offset);
                    Comp::CompAddWide<T1>(s0Hi, s0Lo, s0Lo2, dyReg, pregMask);
                    LoadOneTensor<T1>(x, xReg, pregMask, offset);
                    if constexpr (IsSameType<T1, float>::value) {
                        Comp::CompAddProductWide<T1>(s1Hi, s1Lo, s1Lo2, xReg, dyReg, pregMask);
                        Comp::CompAddScaled(satS0, satS1, xReg, dyReg, pregMask);
                    } else {
                        MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(prodHi, xReg, dyReg, pregMask);
                        Comp::CompAddWide<T1>(s1Hi, s1Lo, s1Lo2, prodHi, pregMask);
                    }
                }

                // dbeta 链：块内 lane 归约到 lane0 后并入本核持久累加器
                Comp::CompFoldLo2(s0Lo, s0Lo2, all);
                Comp::CompReduceLanes(s0Hi, s0Lo, all, laneCount);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(accHi, acc + dbHi + c);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(accLo, acc + dbLo + c);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(accLo2, acc + dbLo2 + c);
                Comp::CompAddPairWide<T1>(accHi, accLo, accLo2, s0Hi, s0Lo, one);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(acc + dbHi + c,
                                                                                                         accHi, one);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(acc + dbLo + c,
                                                                                                         accLo, one);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    acc + dbLo2 + c, accLo2, one);
                if constexpr (IsSameType<T1, float>::value) {
                    MicroAPI::ReduceSum(satRed, satS0, all);
                    AccumulateScalarChannel(acc + sat0 + c, satRed, one);
                }

                // dgamma 链
                Comp::CompFoldLo2(s1Lo, s1Lo2, all);
                Comp::CompReduceLanes(s1Hi, s1Lo, all, laneCount);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(accHi, acc + dgHi + c);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(accLo, acc + dgLo + c);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(accLo2, acc + dgLo2 + c);
                Comp::CompAddPairWide<T1>(accHi, accLo, accLo2, s1Hi, s1Lo, one);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(acc + dgHi + c,
                                                                                                         accHi, one);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(acc + dgLo + c,
                                                                                                         accLo, one);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    acc + dgLo2 + c, accLo2, one);
                if constexpr (IsSameType<T1, float>::value) {
                    MicroAPI::ReduceSum(satRed, satS1, all);
                    AccumulateScalarChannel(acc + sat1 + c, satRed, one);
                }
            }
        }

        dyQueue_.FreeTensor(dyT);
        xQueue_.FreeTensor(xT);
    }

    // 逐通道标量累加：acc += lane0(value)
    __aicore__ inline void AccumulateScalarChannel(__local_mem__ float* addr, RegTensor<float>& laneValue, MaskReg& one)
    {
        RegTensor<float> acc;
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(acc, addr);
        MicroAPI::Add<float, MicroAPI::MaskMergeMode::MERGING>(acc, acc, laneValue, one);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(addr, acc, one);
    }

    // 本核结果写到 workspace[word][core][channel]；lo2 先折回 lo（跨核合并走两字补偿）。
    // 落盘走 VECOUT 队列：FreeTensor 保证 MTE3 已经把数据写进 GM，SyncAll 之后别的核才读得到。
    __aicore__ inline void StorePartials()
    {
        LocalTensor<float> accT = compAccBuf_.Get<float>();
        __local_mem__ float* acc = (__local_mem__ float*)accT.GetPhyAddr();
        const uint32_t aDim = static_cast<uint32_t>(aDim_);
        const uint32_t aDimAligned = static_cast<uint32_t>(aDimAligned_);
        LocalTensor<float> outT = partOutQueue_.AllocTensor<float>();
        __local_mem__ float* out = (__local_mem__ float*)outT.GetPhyAddr();
        __VEC_SCOPE__
        {
            MaskReg one = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
            MaskReg copyMask;
            uint32_t copyLen = aDimAligned;
            copyMask = AscendC::MicroAPI::UpdateMask<float>(copyLen);
            RegTensor<float> lo, lo2, val;
            for (uint16_t c = 0; c < static_cast<uint16_t>(aDim); ++c) {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(lo, acc + 1 * aDimAligned + c);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(lo2, acc + 2 * aDimAligned + c);
                Comp::CompFoldLo2(lo, lo2, one);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(acc + 1 * aDimAligned + c, lo,
                                                                                       one);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(lo, acc + 4 * aDimAligned + c);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(lo2, acc + 5 * aDimAligned + c);
                Comp::CompFoldLo2(lo, lo2, one);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(acc + 4 * aDimAligned + c, lo,
                                                                                       one);
            }
            // 8 个 word 整块搬到队列 buffer（每段 32B 对齐）
            for (uint16_t w = 0; w < 8; ++w) {
                const uint32_t off = static_cast<uint32_t>(w) * aDimAligned;
                MicroAPI::DataCopy(val, acc + off);
                MicroAPI::DataCopy(out + off, val, copyMask);
            }
        }
        partOutQueue_.EnQue(outT);
        outT = partOutQueue_.DeQue<float>();
        out = (__local_mem__ float*)outT.GetPhyAddr();
        const int64_t wsStride = rCores_ * aDim_;
        const int64_t wordSrc[RS_PARTIAL_WORDS] = {0, 1, 3, 4, 6, 7};
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(aDim_ * sizeof(float));
        params.srcStride = 0;
        params.dstStride = 0;
        (void)out;
        for (int64_t w = 0; w < RS_PARTIAL_WORDS; ++w) {
            DataCopyPad(wsGm_[w * wsStride + blockIdx_ * aDim_], outT[wordSrc[w] * aDimAligned_], params);
        }
        partOutQueue_.FreeTensor(outT);
    }

    __aicore__ inline void InitReduceBuffers()
    {
        // UB 里每核一行，行距按 32B 对齐（DataCopyPad 目的块天然按 32B 补齐）
        aDimPad_ = ops::CeilAlign(aDim_ * static_cast<int64_t>(sizeof(float)), BLOCK_SIZE) / sizeof(float);
        // 末尾多留一个向量：reduce 里的整向量 DataCopy 会跨过最后一行的尾部（掩码掉，但地址要合法）
        pipe_->InitBuffer(partBuf_, (RS_PARTIAL_WORDS * rCores_ * aDimPad_ + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(meanBuf_, aDimPad64_ * sizeof(float));
        pipe_->InitBuffer(varBuf_, aDimPad64_ * sizeof(float));
        pipe_->InitBuffer(dgammaBuf_, aDimPad64_ * sizeof(float));
        pipe_->InitBuffer(dbetaBuf_, aDimPad64_ * sizeof(float));
    }

    __aicore__ inline void ReduceAndFinalize()
    {
        LocalTensor<float> partT = partBuf_.Get<float>();
        DataCopyExtParams partParams;
        partParams.blockCount = static_cast<uint16_t>(rCores_);
        partParams.blockLen = static_cast<uint32_t>(aDim_ * sizeof(float));
        partParams.srcStride = 0; // GM 上 [core][channel] 连续
        partParams.dstStride = static_cast<uint32_t>(
            (aDimPad_ * static_cast<int64_t>(sizeof(float)) - partParams.blockLen) / BLOCK_SIZE);
        DataCopyPadExtParams<float> partPadParams;
        partPadParams.isPad = false;
        for (int64_t w = 0; w < RS_PARTIAL_WORDS; ++w) {
            DataCopyPad(partT[w * rCores_ * aDimPad_], wsGm_[w * rCores_ * aDim_], partParams, partPadParams);
        }

        LocalTensor<float> meanT = meanBuf_.Get<float>();
        LocalTensor<float> varT = varBuf_.Get<float>();
        LocalTensor<float> dgammaT = dgammaBuf_.Get<float>();
        LocalTensor<float> dbetaT = dbetaBuf_.Get<float>();
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
        __local_mem__ float* dgammaPtr = (__local_mem__ float*)dgammaT.GetPhyAddr();
        __local_mem__ float* dbetaPtr = (__local_mem__ float*)dbetaT.GetPhyAddr();
        __local_mem__ float* part = (__local_mem__ float*)partT.GetPhyAddr();

        // rstd = fl32(1 / fl32(sqrt(fl32(var + eps))))，与 golden 的 _rstd_infer_fp32 逐位一致
        Comp::RstdExactF32Ub(varPtr, static_cast<uint32_t>(aDim_), tilingData_->dxTilingData.epsilon);

        const uint32_t aDimPad = static_cast<uint32_t>(aDimPad_);
        uint32_t rCores = static_cast<uint32_t>(activeCores_);
        uint32_t curALen = static_cast<uint32_t>(aDim_);
        const uint32_t wordStride = static_cast<uint32_t>(rCores_ * aDimPad_);
        const uint32_t iDbHi = 0;
        const uint32_t iDbLo = wordStride;
        const uint32_t iDgHi = 2 * wordStride;
        const uint32_t iDgLo = 3 * wordStride;
        const uint32_t iSat0 = 4 * wordStride;
        const uint32_t iSat1 = 5 * wordStride;

        __VEC_SCOPE__
        {
            MaskReg pregMask = AscendC::MicroAPI::UpdateMask<float>(curALen);
            MaskReg all = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            RegTensor<float> hi0, lo0, lo20, hi1, lo1, lo21, ph, pl, sat0, sat1;
            RegTensor<float> meanReg, rstdReg, dgOut, dbOut;

            AscendC::MicroAPI::DataCopy(hi0, part + iDbHi);
            AscendC::MicroAPI::DataCopy(lo0, part + iDbLo);
            AscendC::MicroAPI::Duplicate(lo20, 0.0f, all);
            AscendC::MicroAPI::DataCopy(hi1, part + iDgHi);
            AscendC::MicroAPI::DataCopy(lo1, part + iDgLo);
            AscendC::MicroAPI::Duplicate(lo21, 0.0f, all);
            AscendC::MicroAPI::DataCopy(sat0, part + iSat0);
            AscendC::MicroAPI::DataCopy(sat1, part + iSat1);

            for (uint16_t i = 1; i < static_cast<uint16_t>(rCores); ++i) {
                AscendC::MicroAPI::DataCopy(ph, part + iDbHi + i * aDimPad);
                AscendC::MicroAPI::DataCopy(pl, part + iDbLo + i * aDimPad);
                Comp::CompAddPairWide<float>(hi0, lo0, lo20, ph, pl, pregMask);
                AscendC::MicroAPI::DataCopy(ph, part + iDgHi + i * aDimPad);
                AscendC::MicroAPI::DataCopy(pl, part + iDgLo + i * aDimPad);
                Comp::CompAddPairWide<float>(hi1, lo1, lo21, ph, pl, pregMask);
                AscendC::MicroAPI::DataCopy(ph, part + iSat0 + i * aDimPad);
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(sat0, sat0, ph, pregMask);
                AscendC::MicroAPI::DataCopy(ph, part + iSat1 + i * aDimPad);
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(sat1, sat1, ph, pregMask);
            }
            Comp::CompFoldLo2(lo0, lo20, pregMask);
            Comp::CompFoldLo2(lo1, lo21, pregMask);

            LoadOneTensor<float>(meanPtr, meanReg, pregMask, 0);
            LoadOneTensor<float>(varPtr, rstdReg, pregMask, 0);
            if constexpr (IsSameType<T1, float>::value) {
                Comp::CompFinalizeDgammaDbetaSat(hi1, lo1, hi0, lo0, sat0, sat1, meanReg, rstdReg, pregMask, dgOut,
                                                 dbOut);
            } else {
                Comp::CompFinalizeDgammaDbeta(hi1, lo1, hi0, lo0, meanReg, rstdReg, pregMask, dgOut, dbOut);
            }
            StoreOneTensor<float>(dgammaPtr, dgOut, pregMask, 0);
            StoreOneTensor<float>(dbetaPtr, dbOut, pregMask, 0);
        }
        PipeBarrier<PIPE_ALL>();

        DataCopyPad(dgammaGm_, dgammaT, smallParams);
        DataCopyPad(dbetaGm_, dbetaT, smallParams);
    }

private:
    const BatchNormGradExt2InferChannelLastTilingData* tilingData_{nullptr};
    TPipe* pipe_{nullptr};

    GlobalTensor<T1> dyGm_;
    GlobalTensor<T1> xGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> varGm_;
    GlobalTensor<float> dgammaGm_;
    GlobalTensor<float> dbetaGm_;
    GlobalTensor<float> wsGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM> dyQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;
    // workspace partial 必须经 VECOUT 队列出队再写 GM：队列的 DeQue/FreeTensor 保证 MTE3 完成，
    // 否则 SyncAll 之后 0 号核读到的可能还是未落地的旧值（race：结果随运行变化）。
    TQue<QuePosition::VECOUT, 1> partOutQueue_;

    TBuf<TPosition::VECCALC> compAccBuf_;
    TBuf<TPosition::VECCALC> partBuf_;
    TBuf<TPosition::VECCALC> meanBuf_;
    TBuf<TPosition::VECCALC> varBuf_;
    TBuf<TPosition::VECCALC> dgammaBuf_;
    TBuf<TPosition::VECCALC> dbetaBuf_;

    int64_t blockIdx_{0};
    int64_t blockNum_{1};
    int64_t aDim_{1};
    int64_t aDimAligned_{8};
    int64_t aDimPad_{8};
    int64_t aDimPad64_{64};
    int64_t rTotal_{0};
    int64_t rCores_{1};
    int64_t activeCores_{1};
    int64_t rowsPerCore_{0};
    int64_t rBegin_{0};
    int64_t rows_{0};
    int64_t rChunk_{0};
    int64_t compAccLen_{0};
};

template <typename T1, typename T2>
class BatchNormGradExt2InferChannelLast {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;

    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline BatchNormGradExt2InferChannelLast(){};

    __aicore__ inline BatchNormGradExt2InferChannelLast(TPipe* pipeIn,
                                                        const BatchNormGradExt2InferChannelLastTilingData* tilingDataIn)
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
            BatchNormGradExt2InferChannelLastDx<T1, T2> opDx(&tilingData_->dxTilingData);
            opDx.InitStg0(dy, gamma, runningVar, dx, pipe_);
            opDx.ProcessStg0();
            PipeBarrier<PIPE_ALL>();
            pipe_->Reset();
        }
        if (enableDgamma && enableDbeta) {
            // A 在一个向量内（aDim < 64）时走 R 轴多核切分实现：旧实现按 A 轴切核，小通道形状只能
            // 用 1~4 个核、lane 利用率 3%~45%，dgamma/dbeta 会慢一个数量级。
            if (tilingData_->aDimStg1 < static_cast<int64_t>(VL_FP32)) {
                BatchNormGradExt2InferChannelLastRSplit<T1, T2> opDgammaDbetaRSplit(pipe_, tilingData_);
                opDgammaDbetaRSplit.Process(dy, x, save_mean, runningVar, dgamma, dbeta, usrWorkspace);
            } else {
                BatchNormGradExt2InferChannelLastRecompute<T1, T2, BUFFER_NUM, true, true> opDgammaDbeta(pipe_,
                                                                                                         tilingData_);
                opDgammaDbeta.Process(dy, x, save_mean, runningVar, dgamma, dbeta, usrWorkspace);
            }
        } else if (enableDgamma && !enableDbeta) {
            BatchNormGradExt2InferChannelLastRecompute<T1, T2, BUFFER_NUM, true, false> opDgammaDbeta(pipe_,
                                                                                                      tilingData_);
            opDgammaDbeta.Process(dy, x, save_mean, runningVar, dgamma, dbeta, usrWorkspace);
        } else if (!enableDgamma && enableDbeta) {
            BatchNormGradExt2InferChannelLastRecompute<T1, T2, BUFFER_NUM, false, true> opDgammaDbeta(pipe_,
                                                                                                      tilingData_);
            opDgammaDbeta.Process(dy, x, save_mean, runningVar, dgamma, dbeta, usrWorkspace);
        }
    }

private:
    const BatchNormGradExt2InferChannelLastTilingData* tilingData_;
    TPipe* pipe_;

    bool enableDx;
    bool enableDgamma;
    bool enableDbeta;
};
} // namespace BatchNormGradExt2

#endif
