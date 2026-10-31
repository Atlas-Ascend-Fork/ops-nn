/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fake_quant_with_min_max_vars_per_channel.h
 * \brief FakeQuantWithMinMaxVarsPerChannel kernel 实现（arch35 / DAV_3510 / v2.2 方案 A）
 *
 * v2.2 关键变更（基于 DESIGN v2.2 §3.5）：
 *   1. 删除 7 个行无关 TBuf（scaleBuf_/nudgedMinBuf_/nudgedMaxBuf_/boolBothZeroBuf_/
 *      scratchBuf_/oneVecBuf_/floorTmpBuf_）—— 共 ~42 KB UB 释放
 *   2. 删除 PrepareChannelScalarsForSegment 私有方法 —— 计算完全下沉 VF
 *   3. UB Init 仅保留 4 个 DB Queue：inQueueX_/outQueueY_/inQueueMin_/inQueueMax_
 *   4. ProcessSegment 精简为：CopyInMin/Max → DeQue → for baseN { CopyInX → Compute(min,max) → CopyOut }
 *   5. VF Compute 三阶段（同一 __VEC_SCOPE__ 内）：
 *      阶段 1 Load (x/min/max → RegTensor)
 *      阶段 2 Prepare 重算 (scale/scaleSafe/both/zpFromMin/nudgedZp/nudgedMin/nudgedMax 全 RegTensor)
 *      阶段 3 主公式 (clip/shift/div/floor/mul/add/bothMul)
 *   6. invQuantRange_ host 算好下发，避免 VF 内重复 Div
 *   7. CastTrait 仍集中在 _base.h（v2.1 已就绪）
 */
#ifndef _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_H_
#define _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "fake_quant_with_min_max_vars_per_channel_tiling_data.h"
#include "fake_quant_with_min_max_vars_per_channel_base.h"

namespace NsFakeQuantPerChannel {

using namespace AscendC;

template <typename T>
class FakeQuantPerChannelNativePC {
    static constexpr int32_t BUF_NUM = 2; // Double Buffer
    static constexpr int64_t VL = 64;     // VL = 64 (fp32 = 256B)

public:
    __aicore__ inline FakeQuantPerChannelNativePC() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR minIn, GM_ADDR maxIn, GM_ADDR y,
                                const FakeQuantWithMinMaxVarsPerChannelTilingData* td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessSegment(int64_t rowsThisBlk, int64_t baseRow);

    __aicore__ inline void CopyInMin(int64_t curLen, int64_t tailOff);
    __aicore__ inline void CopyInMax(int64_t curLen, int64_t tailOff);
    __aicore__ inline void CopyInX(int64_t curLen, int64_t xElemOff);
    __aicore__ inline void Compute(int64_t curLen, LocalTensor<T>& minSeg, LocalTensor<T>& maxSeg);
    __aicore__ inline void CopyOutY(int64_t curLen, int64_t yElemOff);

    TPipe pipe_;
    // 行相关 DB
    TQue<QuePosition::VECIN, BUF_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUF_NUM> outQueueY_;
    // 行无关 DB
    TQue<QuePosition::VECIN, BUF_NUM> inQueueMin_;
    TQue<QuePosition::VECIN, BUF_NUM> inQueueMax_;
    // [v2.2 删除] 7 个行无关 TBuf —— prepare 全部下沉 VF Compute
    //   scaleBuf_ / nudgedMinBuf_ / nudgedMaxBuf_ / boolBothZeroBuf_ /
    //   scratchBuf_ / oneVecBuf_ / floorTmpBuf_

    GlobalTensor<T> xGm_;
    GlobalTensor<T> minGm_;
    GlobalTensor<T> maxGm_;
    GlobalTensor<T> yGm_;

    // Tiling 缓存
    int64_t numCore_ = 0;
    int64_t blockAxis_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockTailFactor_ = 0;
    int64_t ubAxis_ = 0;
    int64_t baseN_ = 1;
    int64_t baseLen_ = 0;
    int64_t dim0_ = 0;
    int64_t dim1_ = 0;
    int64_t headNum_ = 0;
    int64_t tailDim_ = 0;
    float quantMin_ = 0.0f;
    float quantMax_ = 0.0f;
    float invQuantRange_ = 1.0f; // [v2.2 新增] 1.0f / (quantMax - quantMin)

    // 单核任务边界
    int64_t myHeadRowOffset_ = 0;
    int64_t myHeadRows_ = 0;
    int64_t myTailOffset_ = 0;
    int64_t myTailLen_ = 0;

    int64_t baseLenAligned64_ = 0; // align_up(baseLen, 64)
};

// =============================================================================
// Init（v2.2 方案 A：仅 4 个 Queue）
// =============================================================================
template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::Init(GM_ADDR x, GM_ADDR minIn, GM_ADDR maxIn, GM_ADDR y,
                                                            const FakeQuantWithMinMaxVarsPerChannelTilingData* td)
{
    // 1. 缓存 tiling
    numCore_ = td->numCore;
    blockAxis_ = td->blockAxis;
    blockFactor_ = td->blockFactor;
    blockTailFactor_ = td->blockTailFactor;
    ubAxis_ = td->ubAxis;
    baseN_ = td->baseN;
    baseLen_ = td->baseLen;
    dim0_ = td->dim0;
    dim1_ = td->dim1;
    headNum_ = td->headNum;
    tailDim_ = td->tailDim;
    quantMin_ = td->quantMin;
    quantMax_ = td->quantMax;

    if (baseN_ < 1) {
        baseN_ = 1;
    }
    if (baseLen_ < 1) {
        baseLen_ = VL;
    }

    baseLenAligned64_ = (baseLen_ + VL - 1) / VL * VL;

    // invQuantRange = 1/(qMax - qMin)；qMax-qMin >= 1（quant_max = 2^numBits - 1 >= 3）
    float range = quantMax_ - quantMin_;
    if (range <= 0.0f) {
        range = 1.0f;
    }
    invQuantRange_ = 1.0f / range;

    // 2. 计算本核任务边界
    int64_t blockIdx = GetBlockIdx();
    if (blockIdx >= numCore_) {
        myHeadRows_ = 0;
        myTailLen_ = 0;
        return;
    }

    if (blockAxis_ == 0) {
        myHeadRowOffset_ = blockIdx * blockFactor_;
        myHeadRows_ = (blockIdx == numCore_ - 1) ? blockTailFactor_ : blockFactor_;
        myTailOffset_ = 0;
        myTailLen_ = tailDim_;
    } else {
        myHeadRowOffset_ = 0;
        myHeadRows_ = headNum_;
        myTailOffset_ = blockIdx * blockFactor_;
        myTailLen_ = (blockIdx == numCore_ - 1) ? blockTailFactor_ : blockFactor_;
    }

    // 3. 绑定 GM
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    minGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(minIn));
    maxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(maxIn));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    // 4. 分配 UB（v2.2 方案 A：仅 4 个 Queue，无任何行无关 TBuf）
    pipe_.InitBuffer(inQueueX_, BUF_NUM, baseN_ * baseLen_ * sizeof(T));
    pipe_.InitBuffer(outQueueY_, BUF_NUM, baseN_ * baseLen_ * sizeof(T));
    pipe_.InitBuffer(inQueueMin_, BUF_NUM, baseLenAligned64_ * sizeof(T));
    pipe_.InitBuffer(inQueueMax_, BUF_NUM, baseLenAligned64_ * sizeof(T));
    // [v2.2 全部删除] 7 个行无关 TBuf 与 1 个 floorTmpBuf_ —— prepare 全在 VF Compute 内重算
}

// =============================================================================
// Process —— 三层嵌套外层（v2.2 不变）
// =============================================================================
template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::Process()
{
    if (myHeadRows_ <= 0 || myTailLen_ <= 0) {
        return;
    }

    int64_t nBlockNum = (myHeadRows_ + baseN_ - 1) / baseN_;
    for (int64_t nb = 0; nb < nBlockNum; ++nb) {
        int64_t rowsThisBlk = (nb == nBlockNum - 1) ? (myHeadRows_ - nb * baseN_) : baseN_;
        int64_t baseRow = myHeadRowOffset_ + nb * baseN_;
        ProcessSegment(rowsThisBlk, baseRow);
    }
}

// =============================================================================
// ProcessSegment（v2.2 方案 A）：精简为 CopyInMin/Max → DeQue → for baseN { ... }
// 删除：PrepareChannelScalarsForSegment 调用
// =============================================================================
template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::ProcessSegment(int64_t rowsThisBlk, int64_t baseRow)
{
    int64_t lenLoopNum = myTailLen_ / baseLen_;
    int64_t lenTail = myTailLen_ - lenLoopNum * baseLen_;

    for (int64_t li = 0; li < lenLoopNum; ++li) {
        int64_t tailOff = myTailOffset_ + li * baseLen_;
        CopyInMin(baseLen_, tailOff);
        CopyInMax(baseLen_, tailOff);
        LocalTensor<T> minSeg = inQueueMin_.template DeQue<T>();
        LocalTensor<T> maxSeg = inQueueMax_.template DeQue<T>();

        for (int64_t n = 0; n < rowsThisBlk; ++n) {
            int64_t xElemOff = (baseRow + n) * tailDim_ + tailOff;
            CopyInX(baseLen_, xElemOff);
            Compute(baseLen_, minSeg, maxSeg);
            CopyOutY(baseLen_, xElemOff);
        }

        inQueueMin_.FreeTensor(minSeg);
        inQueueMax_.FreeTensor(maxSeg);
    }

    if (lenTail > 0) {
        int64_t tailOff = myTailOffset_ + lenLoopNum * baseLen_;
        CopyInMin(lenTail, tailOff);
        CopyInMax(lenTail, tailOff);
        LocalTensor<T> minSeg = inQueueMin_.template DeQue<T>();
        LocalTensor<T> maxSeg = inQueueMax_.template DeQue<T>();

        for (int64_t n = 0; n < rowsThisBlk; ++n) {
            int64_t xElemOff = (baseRow + n) * tailDim_ + tailOff;
            CopyInX(lenTail, xElemOff);
            Compute(lenTail, minSeg, maxSeg);
            CopyOutY(lenTail, xElemOff);
        }

        inQueueMin_.FreeTensor(minSeg);
        inQueueMax_.FreeTensor(maxSeg);
    }
}

// =============================================================================
// CopyIn / CopyOut（v2.2 不变）
// =============================================================================
template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::CopyInX(int64_t curLen, int64_t xElemOff)
{
    LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();
    DataCopyExtParams cpyParams;
    cpyParams.blockCount = 1;
    cpyParams.blockLen = static_cast<uint32_t>(curLen * sizeof(T));
    cpyParams.srcStride = 0;
    cpyParams.dstStride = 0;
    cpyParams.rsv = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[xElemOff], cpyParams, padParams);
    inQueueX_.template EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::CopyInMin(int64_t curLen, int64_t tailOff)
{
    LocalTensor<T> mLocal = inQueueMin_.template AllocTensor<T>();
    DataCopyExtParams cpyParams;
    cpyParams.blockCount = 1;
    cpyParams.blockLen = static_cast<uint32_t>(curLen * sizeof(T));
    cpyParams.srcStride = 0;
    cpyParams.dstStride = 0;
    cpyParams.rsv = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(mLocal, minGm_[tailOff], cpyParams, padParams);
    inQueueMin_.template EnQue<T>(mLocal);
}

template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::CopyInMax(int64_t curLen, int64_t tailOff)
{
    LocalTensor<T> mLocal = inQueueMax_.template AllocTensor<T>();
    DataCopyExtParams cpyParams;
    cpyParams.blockCount = 1;
    cpyParams.blockLen = static_cast<uint32_t>(curLen * sizeof(T));
    cpyParams.srcStride = 0;
    cpyParams.dstStride = 0;
    cpyParams.rsv = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(mLocal, maxGm_[tailOff], cpyParams, padParams);
    inQueueMax_.template EnQue<T>(mLocal);
}

template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::CopyOutY(int64_t curLen, int64_t yElemOff)
{
    LocalTensor<T> yLocal = outQueueY_.template DeQue<T>();
    DataCopyExtParams cpyParams;
    cpyParams.blockCount = 1;
    cpyParams.blockLen = static_cast<uint32_t>(curLen * sizeof(T));
    cpyParams.srcStride = 0;
    cpyParams.dstStride = 0;
    cpyParams.rsv = 0;
    DataCopyPad(yGm_[yElemOff], yLocal, cpyParams);
    outQueueY_.FreeTensor(yLocal);
}

// =============================================================================
// Compute（v2.2 方案 A）—— VF Compute 三阶段全链路
//   阶段 1 Load (x/min/max → RegTensor)
//   阶段 2 Prepare 重算 (scale/scaleSafe/both/zpFromMin/nudgedZp/nudgedMin/nudgedMax)
//   阶段 3 主公式 (clip/shift/div/floor/mul/add/bothMul)
// =============================================================================
template <typename T>
__aicore__ inline void FakeQuantPerChannelNativePC<T>::Compute(int64_t curLen, LocalTensor<T>& minSeg,
                                                               LocalTensor<T>& maxSeg)
{
    LocalTensor<T> xLocal = inQueueX_.template DeQue<T>();
    LocalTensor<T> yLocal = outQueueY_.template AllocTensor<T>();

    __ubuf__ T* xAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
    __ubuf__ T* yAddr = reinterpret_cast<__ubuf__ T*>(yLocal.GetPhyAddr());
    __ubuf__ T* minAddr = reinterpret_cast<__ubuf__ T*>(minSeg.GetPhyAddr());
    __ubuf__ T* maxAddr = reinterpret_cast<__ubuf__ T*>(maxSeg.GetPhyAddr());

    constexpr uint32_t vfLen32 = VECTOR_REG_WIDTH / sizeof(float); // = 64 for DAV_3510
    uint32_t totalLen = static_cast<uint32_t>(curLen);
    uint32_t outerLoop = (totalLen + vfLen32 - 1) / vfLen32;
    uint32_t remain = totalLen;
    const float qMin = quantMin_;
    const float qMax = quantMax_;
    const float qRange = quantMax_ - quantMin_; // 0-ULP Div 用：scale = (max-min)/qRange
    // P1-03：分界取索引区间半程 qRange/2（原 (qMax+qMin)/2 在 narrow_range 时偏大 0.5，
    // 使 (qRange/2, (qMax+qMin)/2] 内的整数索引误走低分支而乘积越界）
    const float qHalf = (quantMax_ - quantMin_) * 0.5f;
    // 倒数次正规判定界：scaleSafe > 1/FLT_MIN_NORMAL ⟺ 对应倒数落入次正规区（P1-01）
    const float kInvSubScaleBound = 8.507059173023462e+37f;
    static constexpr AscendC::Reg::DivSpecificMode kDivMode = {AscendC::Reg::MaskMergeMode::ZEROING,
                                                               true}; // true = 0-ULP fp32 Div

    __VEC_SCOPE__
    {
        // === 输入 ===
        AscendC::Reg::RegTensor<float> vregX;
        AscendC::Reg::RegTensor<float> vregMin;
        AscendC::Reg::RegTensor<float> vregMax;
        // === prepare 常量 / 中间量 ===
        AscendC::Reg::RegTensor<float> vregZero;   // 全 0（Sub 自身得到）
        AscendC::Reg::RegTensor<float> vregOne;    // 全 1（Adds(zero, 1.0f)）
        AscendC::Reg::RegTensor<float> vregQRange; // 全 qMax-qMin（用作 0-ULP Div 的除数）
        AscendC::Reg::RegTensor<float> vregScale;
        AscendC::Reg::RegTensor<float> vregScaleAlt; // 溢出通道防溢出分解 scale（CODE.11）
        AscendC::Reg::RegTensor<float> vregScaleAltA;
        AscendC::Reg::RegTensor<float> vregScaleAltB;
        AscendC::Reg::RegTensor<float> vregInvScaleAlt;
        AscendC::Reg::RegTensor<float> vregScaleSafe;
        AscendC::Reg::RegTensor<float> vregDiffSaved;
        AscendC::Reg::RegTensor<float> vregDiffSafe;
        AscendC::Reg::RegTensor<float> vregInvScaleSafe;
        AscendC::Reg::RegTensor<float> vregKeep; // 1 = 非退化通道；0 = 退化通道（min==max 或 scale 次正规，输出全零）
        AscendC::Reg::RegTensor<float> vregTiny;       // 1 = scale < FLT_MIN_NORMAL（含负值）
        AscendC::Reg::RegTensor<float> vregNonNeg;     // 1 = scale >= 0
        AscendC::Reg::RegTensor<float> vregDegenerate; // 1 = 0 <= scale < FLT_MIN_NORMAL（退化）
        AscendC::Reg::RegTensor<float> vregZpFromMin;
        AscendC::Reg::RegTensor<float> vregZpClipped;
        AscendC::Reg::RegTensor<int32_t> vregZpInt;
        AscendC::Reg::RegTensor<float> vregNudgedZp;
        AscendC::Reg::RegTensor<float> vregNudgedMin;
        AscendC::Reg::RegTensor<float> vregNudgedMax;
        // === 主公式 ===
        AscendC::Reg::RegTensor<float> vregClamped;
        AscendC::Reg::RegTensor<float> vregNudgedTmp;  // CODE.11 ②nudge 收边临时
        AscendC::Reg::RegTensor<float> vregScaledAltA; // CODE.11 ③scaled 分解
        AscendC::Reg::RegTensor<float> vregScaledAltB;
        AscendC::Reg::RegTensor<float> vregScaledAlt;
        AscendC::Reg::RegTensor<float> vregIdxLo; // CODE.11 ④双侧重建
        AscendC::Reg::RegTensor<float> vregOutLo;
        AscendC::Reg::RegTensor<float> vregQMaxIdx;
        AscendC::Reg::RegTensor<float> vregIdxHi;
        AscendC::Reg::RegTensor<float> vregOutHi;
        AscendC::Reg::RegTensor<float> vregOutAlt;
        AscendC::Reg::RegTensor<float> vregOvfBound;   // CODE.11 溢出判定界（FLT_MAX+min）
        AscendC::Reg::RegTensor<float> vregOvfDelta;   // CODE.11 溢出判定差（max-bound，恒有限）
        AscendC::Reg::RegTensor<float> vregClampLo;    // P1-02 收边检测（nudgedMin+FLT_MAX）
        AscendC::Reg::RegTensor<float> vregClampHi;    // P1-02 收边检测（nudgedMax-FLT_MAX）
        AscendC::Reg::RegTensor<float> vregInvSubF;    // P1-01 倒数次正规标志（0/1）
        AscendC::Reg::RegTensor<float> vregOvfF;       // 溢出标志（0/1，浮点域）
        AscendC::Reg::RegTensor<float> vregReconFlag;  // 双侧重建标志（溢出∪倒数次正规）
        AscendC::Reg::RegTensor<float> vregScaledDivA; // P1-01 除法先行 scaled
        AscendC::Reg::RegTensor<float> vregScaledDivB;
        AscendC::Reg::RegTensor<float> vregScaledDiv;
        AscendC::Reg::RegTensor<float> vregQMaxB;    // qMax 广播（④高侧距离）
        AscendC::Reg::RegTensor<float> vregOutOther; // P1-06 兜底另一分支
        AscendC::Reg::RegTensor<float> vregChkPos;   // P1-06 越界检测
        AscendC::Reg::RegTensor<float> vregChkNeg;
        AscendC::Reg::RegTensor<float> vregPosF;
        AscendC::Reg::RegTensor<float> vregNegF;
        AscendC::Reg::RegTensor<float> vregBadF;
        AscendC::Reg::RegTensor<float> vregShift;
        AscendC::Reg::RegTensor<float> vregScaled;
        AscendC::Reg::RegTensor<int32_t> vregFlrInt;
        AscendC::Reg::RegTensor<float> vregFloored;
        AscendC::Reg::RegTensor<float> vregOut;
        AscendC::Reg::RegTensor<float> vregNan;   // NaN 传播载体（仅在 NaN 车道有效）
        AscendC::Reg::RegTensor<float> vregInf;   // K1: +inf 载体（FLT_MAX 加倍上溢构造）
        AscendC::Reg::RegTensor<float> vregNanEq; // K1: 相等非零通道 NaN 注入载体（仅该类车道为 NaN）
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::MaskReg maskTiny;
        AscendC::Reg::MaskReg maskNonNeg;
        AscendC::Reg::MaskReg maskNeg;
        AscendC::Reg::MaskReg maskOvf;
        AscendC::Reg::MaskReg maskHighIdx;
        AscendC::Reg::MaskReg maskClampLo;
        AscendC::Reg::MaskReg maskClampHi;
        AscendC::Reg::MaskReg maskInvSub;
        AscendC::Reg::MaskReg maskRecon;
        AscendC::Reg::MaskReg maskChkPos;
        AscendC::Reg::MaskReg maskChkNeg;
        AscendC::Reg::MaskReg maskBad;
        AscendC::Reg::MaskReg maskEq;    // K1: min==max（diff==0）
        AscendC::Reg::MaskReg maskMinNz; // K1: min!=0（diff==0 时即非全零通道）
        AscendC::Reg::MaskReg maskEqNz;  // K1: min==max!=0 相等非零通道

        for (uint16_t i = 0; i < static_cast<uint16_t>(outerLoop); ++i) {
            mask = AscendC::Reg::UpdateMask<float>(remain);

            // -----------------------------------------------------------------
            // 阶段 1：Load x / min / max
            // -----------------------------------------------------------------
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregX, xAddr + i * vfLen32);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregMin, minAddr + i * vfLen32);
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(vregMax, maxAddr + i * vfLen32);

            // -----------------------------------------------------------------
            // 阶段 2：Prepare 重算（替代 PrepareChannelScalarsForSegment）
            // -----------------------------------------------------------------

            // 2.0 构造常量 vregZero / vregOne / vregQRange（寄存器内）
            AscendC::Reg::Sub<float>(vregZero, vregMin, vregMin, mask);
            AscendC::Reg::Adds<float>(vregOne, vregZero, 1.0f, mask);
            AscendC::Reg::Adds<float>(vregQRange, vregZero, qRange, mask);
            AscendC::Reg::Adds<float>(vregQMaxB, vregZero, qMax, mask);

            // 2.1 scale = (max - min) / qRange
            //     用 0-ULP Div 替代 (max-min)*invR：避免 invR=1/(qMax-qMin) 在 fp32 上的
            //     round-to-nearest 误差与 (max-min)*invR 的舍入叠加，使 scale 精确等于
            //     `(max-min)/(qMax-qMin)` 的 IEEE 单步除法结果（与 TF 实现一致）。
            AscendC::Reg::Sub<float>(vregScale, vregMax, vregMin, mask);
            AscendC::Reg::Copy<float>(vregDiffSaved, vregScale, mask);
            AscendC::Reg::Div<float, &kDivMode>(vregScale, vregScale, vregQRange, mask);

            // 2.1b 量程溢出防护（CODE.11）：min/max 异号且 |min|+|max| 超过 fp32 上限时
            //     (max-min) 溢出为 +inf，后续 nudge 链（nudgedMin=(qMin-zp)*inf）产生 NaN，
            //     整通道输出 NaN。此时改用防溢出分解 scale = max/qRange - min/qRange
            //     （数学等价且各步有限）；溢出通道的 invScale 同步改用 1/scale。
            //     正常通道不命中该分支，scale/invScale 逐位不变。
            //     溢出判定不经过 (max-min)（其本身即溢出点，不同执行环境的饱和/inf
            //     行为不一致），用等价无溢出条件 max > FLT_MAX + min：min<0 时右端
            //     = FLT_MAX-|min|（有限），max>它 ⟺ max-min>FLT_MAX；min>=0 时
            //     右端 >= FLT_MAX >= max 恒 false。各步有限、不依赖 inf 语义。
            AscendC::Reg::Adds<float>(vregOvfBound, vregMin, 3.4028234663852886e+38f, mask);
            // reg 对 reg 比较无 Compares 标量接口，用差值符号等价判定：
            // max-(FLT_MAX+min) 恒有限（两端均在 [0, FLT_MAX]），>0 ⟺ max>bound ⟺ 溢出
            AscendC::Reg::Sub<float>(vregOvfDelta, vregMax, vregOvfBound, mask);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(maskOvf, vregOvfDelta, 0.0f, mask);
            AscendC::Reg::Div<float, &kDivMode>(vregScaleAltA, vregMax, vregQRange, mask);
            AscendC::Reg::Div<float, &kDivMode>(vregScaleAltB, vregMin, vregQRange, mask);
            AscendC::Reg::Sub<float>(vregScaleAlt, vregScaleAltA, vregScaleAltB, mask);
            AscendC::Reg::Select<float>(vregScale, vregScaleAlt, vregScale, maskOvf);

            // 2.2 退化通道判定（P1-01 补充：双侧次正规）：|scale| < FLT_MIN_NORMAL 即退化。
            //     正侧覆盖 min==max（scale==0）与正次正规（FTZ/倒数坍缩）；
            //     负侧覆盖反向区间×次正规量程（倒数溢出 ∓inf、int32 转换 UB——统一清零）。
            //     正常量程的反向区间（scale 显著负）不判退化，保持回退行为。
            //     正常通道 tiny*nonneg=0，keep/scaleSafe/diffSafe 与旧实现逐位一致。
            AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(maskTiny, vregScale, 1.17549435e-38f, mask);
            AscendC::Reg::Select<float>(vregTiny, vregOne, vregZero, maskTiny);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(maskNeg, vregScale, -1.17549435e-38f, mask);
            AscendC::Reg::Select<float>(vregNonNeg, vregOne, vregZero, maskNeg);
            AscendC::Reg::Mul<float>(vregDegenerate, vregTiny, vregNonNeg, mask);
            // keep = 1 - degenerate
            AscendC::Reg::Sub<float>(vregKeep, vregOne, vregDegenerate, mask);

            // 2.3 scaleSafe = scale*keep + degenerate：退化通道取 1.0（防止后续除法/倒数
            //     溢出产生 NaN 经 out*keep 泄漏，NaN*0=NaN）；正常通道逐位等于 scale
            AscendC::Reg::Mul<float>(vregScaleSafe, vregScale, vregKeep, mask);
            AscendC::Reg::Add<float>(vregScaleSafe, vregScaleSafe, vregDegenerate, mask);

            // 2.3b invScaleSafe = qRange / diffSafe
            //      TF Nudge() 独立计算 inv_scale = (qMax-qMin)/(max-min)，
            //      而非 1/scale，避免 scale 的 fp32 舍入误差在取倒数时被放大。
            //      diffSafe = diff*keep + degenerate：退化通道取 1.0 防倒数上溢，
            //      正常通道逐位等于 (max-min)
            AscendC::Reg::Mul<float>(vregDiffSafe, vregDiffSaved, vregKeep, mask);
            AscendC::Reg::Add<float>(vregDiffSafe, vregDiffSafe, vregDegenerate, mask);
            AscendC::Reg::Div<float, &kDivMode>(vregInvScaleSafe, vregQRange, vregDiffSafe, mask);
            // 溢出通道（2.1b）：diffSafe=inf 使 qRange/inf=0，改用 1/scale（scale 已是
            // 防溢出分解值），保持 shift*invScale 的量化步进正确
            AscendC::Reg::Div<float, &kDivMode>(vregInvScaleAlt, vregOne, vregScaleAlt, mask);
            AscendC::Reg::Select<float>(vregInvScaleSafe, vregInvScaleAlt, vregInvScaleSafe, maskOvf);

            // P1-01 倒数次正规检测：scaleSafe > 1/FLT_MIN ⟺ 对应倒数 qRange/(max-min)
            // （或溢出通道的 1/scaleAlt）落入次正规区，设备 FTZ 将其清零会使整通道
            // 坍缩到 nudgedMin。检测只依赖 scaleSafe（此类通道必为正常数，无 FTZ 风险）。
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(maskInvSub, vregScaleSafe, kInvSubScaleBound, mask);
            AscendC::Reg::Select<float>(vregInvSubF, vregOne, vregZero, maskInvSub);
            // 双侧重建适用通道 = 溢出 ∪ 倒数次正规（浮点域求并后转掩码）
            AscendC::Reg::Select<float>(vregOvfF, vregOne, vregZero, maskOvf);
            AscendC::Reg::Max<float>(vregReconFlag, vregOvfF, vregInvSubF, mask);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(maskRecon, vregReconFlag, 0.5f, mask);

            // 2.4 zp_from_min = qMin - min / scaleSafe
            //     用 0-ULP Div：min/scaleSafe 在 zfm 落到 N+0.5 半整数边界时极易抖动 1 ULP，
            //     与 floor(zfm+0.5) 叠加会让 nudged_zp 偏离 TF。
            AscendC::Reg::Div<float, &kDivMode>(vregZpFromMin, vregMin, vregScaleSafe, mask);
            AscendC::Reg::Muls<float>(vregZpFromMin, vregZpFromMin, -1.0f, mask);
            AscendC::Reg::Adds<float>(vregZpFromMin, vregZpFromMin, qMin, mask);

            // 2.5 nudged_zp = floor(clip(zp_from_min, qMin, qMax) + 0.5)
            AscendC::Reg::Maxs<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vregZpClipped, vregZpFromMin, qMin,
                                                                                   mask);
            AscendC::Reg::Mins<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(vregZpClipped, vregZpClipped, qMax,
                                                                                   mask);
            AscendC::Reg::Adds<float>(vregZpClipped, vregZpClipped, 0.5f, mask);
            AscendC::Reg::Cast<int32_t, float, kCastFp32ToInt32Floor>(vregZpInt, vregZpClipped, mask);
            AscendC::Reg::Cast<float, int32_t, kCastInt32ToFp32>(vregNudgedZp, vregZpInt, mask);

            // 2.6 nudged_min = (qMin - nudged_zp) * scaleSafe
            //     nudged_max = (qMax - nudged_zp) * scaleSafe
            AscendC::Reg::Muls<float>(vregNudgedMin, vregNudgedZp, -1.0f, mask);
            AscendC::Reg::Adds<float>(vregNudgedMin, vregNudgedMin, qMin, mask);
            AscendC::Reg::Mul<float>(vregNudgedMin, vregNudgedMin, vregScaleSafe, mask);

            AscendC::Reg::Muls<float>(vregNudgedMax, vregNudgedZp, -1.0f, mask);
            AscendC::Reg::Adds<float>(vregNudgedMax, vregNudgedMax, qMax, mask);
            AscendC::Reg::Mul<float>(vregNudgedMax, vregNudgedMax, vregScaleSafe, mask);

            // CODE.11 ②nudge 收边（P1-02 修正）：仅当计算值越过 ±FLT_MAX（∓inf）时按原
            // min/max 收边；有限计算值保持原值——zp 舍入可使有限 nudgedMin 低于 min 达
            // 0.5 步长，无条件收边会平移量化网格。越界检测不经 inf 字面量：
            // nudgedMin+FLT_MAX<0 ⟺ nudgedMin<-FLT_MAX（含 -inf）。
            AscendC::Reg::Adds<float>(vregClampLo, vregNudgedMin, 3.4028234663852886e+38f, mask);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(maskClampLo, vregClampLo, 0.0f, mask);
            AscendC::Reg::Select<float>(vregNudgedMin, vregMin, vregNudgedMin, maskClampLo);
            AscendC::Reg::Adds<float>(vregClampHi, vregNudgedMax, -3.4028234663852886e+38f, mask);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(maskClampHi, vregClampHi, 0.0f, mask);
            AscendC::Reg::Select<float>(vregNudgedMax, vregMax, vregNudgedMax, maskClampHi);

            // -----------------------------------------------------------------
            // 阶段 3：主公式
            // -----------------------------------------------------------------
            // clip：clamped = min(max(x, nudgedMin), nudgedMax)
            AscendC::Reg::Min<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregClamped, vregX, vregNudgedMax, mask);
            AscendC::Reg::Max<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregClamped, vregClamped, vregNudgedMin,
                                                                           mask);
            // shift = clamped - nudgedMin
            AscendC::Reg::Sub<float>(vregShift, vregClamped, vregNudgedMin, mask);
            // scaled = shift * invScaleSafe + 0.5
            //        与 TF 一致：使用独立计算的 inv_scale = qRange/(max-min)，
            //        而非 shift/scale（二次舍入误差）
            AscendC::Reg::Mul<float>(vregScaled, vregShift, vregInvScaleSafe, mask);
            AscendC::Reg::Adds<float>(vregScaled, vregScaled, 0.5f, mask);
            // CODE.11 ③溢出通道 scaled 分解：shift=(clamped-nudgedMin) 可达约 2×|nudgedMin|
            // 超过 fp32（如 ±3e38 量程内的大幅 x）；改用分解
            // scaledAlt = clamped*invScaleSafe - nudgedMin*invScaleSafe（各步 ≤ 量子数，
            // 有限），仅溢出通道取用，正常通道逐位不变
            AscendC::Reg::Mul<float>(vregScaledAltA, vregClamped, vregInvScaleSafe, mask);
            AscendC::Reg::Mul<float>(vregScaledAltB, vregNudgedMin, vregInvScaleSafe, mask);
            AscendC::Reg::Sub<float>(vregScaledAlt, vregScaledAltA, vregScaledAltB, mask);
            AscendC::Reg::Adds<float>(vregScaledAlt, vregScaledAlt, 0.5f, mask);
            AscendC::Reg::Select<float>(vregScaled, vregScaledAlt, vregScaled, maskOvf);
            // P1-01 倒数次正规通道：scaled 改用除法先行 clamped/scaleSafe - nudgedMin/scaleSafe + 0.5
            // （scaleSafe 为正常数，各步有限且不经过次正规倒数；对溢出通道的次正规
            // 1/scaleAlt 同样生效，覆盖 ③ 乘法在设备上被 FTZ 清零的情形）
            AscendC::Reg::Div<float, &kDivMode>(vregScaledDivA, vregClamped, vregScaleSafe, mask);
            AscendC::Reg::Div<float, &kDivMode>(vregScaledDivB, vregNudgedMin, vregScaleSafe, mask);
            AscendC::Reg::Sub<float>(vregScaledDiv, vregScaledDivA, vregScaledDivB, mask);
            AscendC::Reg::Adds<float>(vregScaledDiv, vregScaledDiv, 0.5f, mask);
            AscendC::Reg::Select<float>(vregScaled, vregScaledDiv, vregScaled, maskInvSub);
            // floor 语义：fp32 → int32 (CAST_FLOOR) → fp32 (UNKNOWN)
            AscendC::Reg::Cast<int32_t, float, kCastFp32ToInt32Floor>(vregFlrInt, vregScaled, mask);
            AscendC::Reg::Cast<float, int32_t, kCastInt32ToFp32>(vregFloored, vregFlrInt, mask);
            // out = floored * scaleSafe + nudgedMin
            AscendC::Reg::Mul<float>(vregOut, vregFloored, vregScaleSafe, mask);
            AscendC::Reg::Add<float>(vregOut, vregOut, vregNudgedMin, mask);
            // CODE.11 ④溢出通道双侧重建：floored*scaleSafe 在大索引处可越过 +FLT_MAX
            // （如 255×2.35e36）；i<=qRange/2 用 nMin+i*scale、i>qRange/2 用
            // nMax-(qRange-i)*scale（各步 ≤ qRange/2×scale ≤ FLT_MAX，数学等价），
            // 仅溢出通道取用
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(maskHighIdx, vregFloored, qHalf, mask);
            AscendC::Reg::Mul<float>(vregIdxLo, vregFloored, vregScaleSafe, mask);
            AscendC::Reg::Add<float>(vregOutLo, vregNudgedMin, vregIdxLo, mask);
            // P1-2: 高侧距离用 qRange-floored（floored 的索引域是 [0, qRange]，
            // narrow_range 时 qMax-floored 会多出 qMin=1 步）
            AscendC::Reg::Sub<float>(vregQMaxIdx, vregQRange, vregFloored, mask);
            AscendC::Reg::Mul<float>(vregIdxHi, vregQMaxIdx, vregScaleSafe, mask);
            AscendC::Reg::Sub<float>(vregOutHi, vregNudgedMax, vregIdxHi, mask);
            // floored>qHalf（mask=1）→ 取 src0=out_hi；否则取 src1=out_lo
            AscendC::Reg::Select<float>(vregOutAlt, vregOutHi, vregOutLo, maskHighIdx);
            AscendC::Reg::Select<float>(vregOut, vregOutAlt, vregOut, maskRecon);
            // P1-06 兜底 1：|out|>FLT_MAX（±inf，检测不经 inf 字面量）→ 取另一分支
            // （两侧乘积互补有限）；对称中点角落两分支可同时越界，交给兜底 2。
            AscendC::Reg::Select<float>(vregOutOther, vregOutLo, vregOutHi, maskHighIdx);
            AscendC::Reg::Adds<float>(vregChkPos, vregOut, -3.4028234663852886e+38f, mask);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(maskChkPos, vregChkPos, 0.0f, mask);
            AscendC::Reg::Select<float>(vregPosF, vregOne, vregZero, maskChkPos);
            AscendC::Reg::Adds<float>(vregChkNeg, vregOut, 3.4028234663852886e+38f, mask);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(maskChkNeg, vregChkNeg, 0.0f, mask);
            AscendC::Reg::Select<float>(vregNegF, vregOne, vregZero, maskChkNeg);
            AscendC::Reg::Max<float>(vregBadF, vregPosF, vregNegF, mask);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(maskBad, vregBadF, 0.5f, mask);
            AscendC::Reg::Select<float>(vregOut, vregOutOther, vregOut, maskBad);
            // P1-06 兜底 2：另一分支仍越界（对称中点角落）→ 取 clamped（与真值差 ≤1
            // 个量化步长，远好于 ±inf）
            AscendC::Reg::Adds<float>(vregChkPos, vregOut, -3.4028234663852886e+38f, mask);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(maskChkPos, vregChkPos, 0.0f, mask);
            AscendC::Reg::Select<float>(vregPosF, vregOne, vregZero, maskChkPos);
            AscendC::Reg::Adds<float>(vregChkNeg, vregOut, 3.4028234663852886e+38f, mask);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(maskChkNeg, vregChkNeg, 0.0f, mask);
            AscendC::Reg::Select<float>(vregNegF, vregOne, vregZero, maskChkNeg);
            AscendC::Reg::Max<float>(vregBadF, vregPosF, vregNegF, mask);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(maskBad, vregBadF, 0.5f, mask);
            AscendC::Reg::Select<float>(vregOut, vregClamped, vregOut, maskBad);
            // out *= keep（退化通道输出 0：min==max==0 全零通道，或量程极小使 fp32 scale
            // 下溢为 0/次正规——min==max≠0 相等通道同样先清零，随后由下方 K1 块覆盖为 NaN）
            AscendC::Reg::Mul<float>(vregOut, vregOut, vregKeep, mask);
            // NaN 语义（README 功能说明 / DESIGN §6）：输入 x 为 NaN 时 y 必须为 NaN。
            // 上方 floor 走 fp32→int32 取整会吞掉 NaN，故此处显式补回：
            //   clamped 已天然携带 NaN（Min/Max 不消解 NaN）；自减后 NaN 车道为 NaN、有限车道为 0，
            //   ±inf 输入经 clamp 后为有限值，不会误变为 NaN。全掩码加法，有限车道结果不变。
            AscendC::Reg::Sub<float>(vregNan, vregClamped, vregClamped, mask);
            AscendC::Reg::Add<float>(vregOut, vregOut, vregNan, mask);
            // K1（Issue #101）：min==max≠0 相等通道输出 NaN（对齐 TF oracle：Nudge 的
            // inv_scale = qRange/(max-min) = +inf → 0×inf = NaN，整通道 NaN）。
            // 与 min==max==0 全零通道区分（TF functor 显式清零，维持上方 keep 清零）。
            // 检测：diff==0（min==max）且 min≠0（diff==0 时 min≠0 ⟺ 非全零）。
            // NaN 须显式注入——floor 的 Cast<int32> 会吞 NaN，0-ULP Div 对 0/0 走
            // ZEROING 同样不产生 NaN，自然传播不可用；载体经 inf-inf 构造（inf 来自
            // FLT_MAX 加倍上溢，与 2.1b 的 (max-min) 溢出同机制），按通道掩码注入，
            // 避开 int cast 路径。次正规退化通道（diff≠0）与倒置通道不在此域，行为不变。
            AscendC::Reg::Compares<float, AscendC::CMPMODE::EQ>(maskEq, vregDiffSaved, 0.0f, mask);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::NE>(maskMinNz, vregMin, 0.0f, mask);
            AscendC::Reg::MaskAnd(maskEqNz, maskEq, maskMinNz, mask);
            AscendC::Reg::Adds<float>(vregInf, vregZero, 3.4028234663852886e+38f, mask);
            AscendC::Reg::Adds<float>(vregInf, vregInf, 3.4028234663852886e+38f, mask);
            AscendC::Reg::Sub<float>(vregNanEq, vregInf, vregInf, mask);
            AscendC::Reg::Select<float>(vregNanEq, vregNanEq, vregZero, maskEqNz);
            AscendC::Reg::Add<float>(vregOut, vregOut, vregNanEq, mask);
            // Store
            AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yAddr + i * vfLen32, vregOut, mask);
        }
    } // __VEC_SCOPE__

    outQueueY_.template EnQue<T>(yLocal);
    inQueueX_.FreeTensor(xLocal);
}

} // namespace NsFakeQuantPerChannel

#endif // _FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_H_
