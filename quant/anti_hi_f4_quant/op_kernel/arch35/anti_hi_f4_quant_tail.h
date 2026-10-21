/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file anti_hi_f4_quant_tail_axis.h
 * \brief AntiHiF4Quant -1 轴（tail axis）反量化 Kernel 实现
 *        将 HIFLOAT4(S1P2) + Scale(Ea/Eb/Ec) 反量化为 BF16
 *        对标 dynamic_hi_f4_quant_tail_axis.h 的 UB/Queue 结构，计算替换为反量化
 */

#ifndef ANTI_HIF4_QUANT_TAIL_AXIS_HIFLOAT4_H
#define ANTI_HIF4_QUANT_TAIL_AXIS_HIFLOAT4_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "anti_hi_f4_quant_tilingdata.h"
#include "anti_hi_f4_quant_common.h"

namespace AntiHiF4Quant {
using namespace AscendC;

// =====================================================================
// AntiHiF4QuantTail: -1 轴反量化 Kernel 主类
// Pipeline: CopyIn → ComputeAll(ExtractScale+Dequant) → CopyOut
// 简化计算流程:
//   1) DataCopyUnAlign 加载 8 个 uint32 scale → VF
//   2) DeInterleave → EbEa VF + EcEc VF (每个 block 相同值)
//   3) Ea: EbEa & 0xFF → Cast E6M2→BF16
//   4) Eb: ShiftRight(EbEa, ebShiftReg) → And(0b1)
//   5) Ec: ShiftRight(EcEc, ecShiftReg) → And(0b1)
//   6) pow = (Eb + Ec) << 1 → uint16 → fp16 → bf16
//   7) fp4 → BF16, Mul(pow), Mul(Ea) → result
// =====================================================================
template <typename ST_X, typename DT_Y>
class AntiHiF4QuantTail {
public:
    __aicore__ inline AntiHiF4QuantTail(const AntiHiF4QuantTailTilingData* __restrict tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe)
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR y);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn(const int64_t& gmOffset, const int64_t& scaleGmOffset);
    __aicore__ inline void ComputeAll(int64_t scaleRegLoop, int64_t outRegLoop);
    __aicore__ inline void CopyOut(const int64_t& gmOffset);

    // Compute 子阶段 — 直接在 VF 上完成 Eb/Ec 提取 + 反量化
    __aicore__ inline void ComputeDequantPerVF(uint16_t scaleRegLoop, __ubuf__ float* scaleReadAddr,
                                               __ubuf__ bfloat16_t* tmpScaleAddr);
    __aicore__ inline void ComputeOutPerVF(uint16_t outRegLoop, __ubuf__ uint8_t* xAddr,
                                           __ubuf__ bfloat16_t* tmpScaleAddr, __ubuf__ DT_Y* yWriteAddr);

private:
    const AntiHiF4QuantTailTilingData* __restrict tilingData_{nullptr};
    TPipe* pipe_{nullptr};
    int64_t blockIdx_{0};

    // Queues: x(HIFLOAT4 packed uint8), scale(uint32/float), y(BF16 or FP16 output)
    TQue<QuePosition::VECIN, DB_BUFFER> inQueueX_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueueScale_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueueY_;

    TBuf<TPosition::VECCALC> tempBuf_;     // 64 bytes: [8..15 dup2 | 0..15 seq]
    TBuf<TPosition::VECCALC> scaleTmpBuf_; // scale factor 暂存, 供 DIST_US_B16 重加载

    GlobalTensor<uint8_t> xGm_;   // x: HIFLOAT4 packed (fp4x2 → uint8)
    GlobalTensor<float> scaleGm_; // scale: float (4 bytes/group, packed Ea+Eb+Ec)
    GlobalTensor<DT_Y> yGm_;      // y: BF16 or FP16 output

    DataCopyExtParams dataCopyInParams_;
    DataCopyPadExtParams<uint8_t> padParams_;
    DataCopyExtParams dataCopyInScaleParams_;
    DataCopyPadExtParams<float> padScaleParams_;
    DataCopyExtParams dataCopyOutParams_;

    int64_t sysRegLen_ = 256;
    int64_t gmCopyOutRegStep_ = 0;
    int64_t gmCopyTmpBufScaleStep_ = 0; // DIST_US_B16 reload stride
    int64_t blockSize_ = 0;
};

// ==================== Init ====================
template <typename ST_X, typename DT_Y>
__aicore__ inline void AntiHiF4QuantTail<ST_X, DT_Y>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    blockSize_ = BLOCK_SIZE;

    // x: HIFLOAT4 packed, 每 group 32 字节 (64 fp4 → 32 uint8)
    int64_t xGmOffset = blockIdx_ * tilingData_->groupNum * DQ_HIFP4_GROUP_BYTE_NUM;
    xGm_.SetGlobalBuffer((__gm__ uint8_t*)x + xGmOffset);
    // scale: float, 每 group 4 字节 (1 个 float = 4 bytes)
    int64_t scaleGmOffset = blockIdx_ * tilingData_->groupNum * DQ_HIFP4_SCALE_FP32_NUM;
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale + scaleGmOffset);
    // y: BF16/FP16, 每 group 64 个元素 × 2 bytes = 128 字节
    int64_t yGmOffset = blockIdx_ * tilingData_->groupNum * DQ_HIFP4_BF16_PER_GROUP;
    yGm_.SetGlobalBuffer((__gm__ DT_Y*)y + yGmOffset);

    // Queue 初始化（对标 DynamicHiF4Quant，alignGroupNum 是 UB 单次处理的 group 数）
    int64_t xUbSize = tilingData_->alignGroupNum * DQ_HIFP4_GROUP_BYTE_NUM;     // 32 bytes * alignGroupNum
    int64_t scaleUbSize = tilingData_->alignGroupNum * DQ_HIFP4_SCALE_BYTE_NUM; // 4 bytes * alignGroupNum
    int64_t yUbSize = tilingData_->alignGroupNum * DQ_HIFP4_BF16_PER_GROUP *
                      sizeof(bfloat16_t); // 128 bytes * alignGroupNum

    pipe_->InitBuffer(inQueueX_, DB_BUFFER, xUbSize);
    pipe_->InitBuffer(inQueueScale_, DB_BUFFER, Ops::Base::CeilAlign(scaleUbSize, DQ_HIFP4_SCALE_ALIGN));
    pipe_->InitBuffer(outQueueY_, DB_BUFFER, yUbSize);
    pipe_->InitBuffer(tempBuf_, 64);
    pipe_->InitBuffer(scaleTmpBuf_, tilingData_->alignGroupNum * DQ_HIFP4_SCALE_ALIGN *
                                        sizeof(int16_t)); // scale factor 暂存, 4 轮 × 32B + 对齐

    LocalTensor<int16_t> tempLocal = tempBuf_.Get<int16_t>();
    for (uint16_t i = 0; i < 8; i++) {
        int16_t val = static_cast<int16_t>(i + 8);
        tempLocal.SetValue(i * 2, val);
        tempLocal.SetValue(i * 2 + 1, val);
    }
    for (uint16_t i = 0; i < 16; i++) {
        tempLocal.SetValue(16 + i, static_cast<int16_t>(i));
    }

    // DataCopy 参数初始化
    dataCopyInParams_.blockCount = 1;
    dataCopyInParams_.blockLen = 0;
    dataCopyInParams_.srcStride = 0;
    dataCopyInParams_.dstStride = 0;

    dataCopyInScaleParams_.blockCount = 1;
    dataCopyInScaleParams_.blockLen = 0;
    dataCopyInScaleParams_.srcStride = 0;
    dataCopyInScaleParams_.dstStride = 0;

    dataCopyOutParams_.blockCount = 1;
    dataCopyOutParams_.blockLen = 0;
    dataCopyOutParams_.srcStride = 0;
    dataCopyOutParams_.dstStride = 0;

    padParams_.isPad = false;
    padParams_.leftPadding = 0;
    padParams_.rightPadding = 0;

    padScaleParams_.isPad = false;
    padScaleParams_.leftPadding = 0;
    padScaleParams_.rightPadding = 0;

    gmCopyOutRegStep_ = sysRegLen_ / sizeof(int32_t) / DIGIT_FOUR; // 16
    gmCopyTmpBufScaleStep_ = sysRegLen_;                           // DIST_US_B16 每次加载 8 个 u16
}

// ==================== CopyIn ====================
template <typename ST_X, typename DT_Y>
__aicore__ inline void AntiHiF4QuantTail<ST_X, DT_Y>::CopyIn(const int64_t& gmOffset, const int64_t& scaleGmOffset)
{
    // 搬入 x (HIFLOAT4 packed uint8)
    auto xLocal = inQueueX_.template AllocTensor<uint8_t>();
    DataCopyPad(xLocal, xGm_[gmOffset], dataCopyInParams_, padParams_);
    inQueueX_.EnQue(xLocal);

    // 搬入 scale (float)
    auto scaleLocal = inQueueScale_.template AllocTensor<float>();
    DataCopyPad(scaleLocal, scaleGm_[scaleGmOffset], dataCopyInScaleParams_, padScaleParams_);
    inQueueScale_.EnQue(scaleLocal);
}

// ==================== ComputeDequantPerVF ====================
// 简化反量化: 1 次 scale 加载 → DeInterleave → 直接 shift 提取 Eb/Ec bit → 计算
// scale uint32 小端: [EA(8bit) | EB(8bit) | EC_lo(8bit) | EC_hi(8bit)]
// u16 视图: 偶 lane = EbEa, 奇 lane = EcEc
template <typename ST_X, typename DT_Y>
__aicore__ inline void AntiHiF4QuantTail<ST_X, DT_Y>::ComputeDequantPerVF(uint16_t scaleRegLoop,
                                                                          __ubuf__ float* scaleReadAddr,
                                                                          __ubuf__ bfloat16_t* tmpScaleAddr)
{
    __VEC_SCOPE__
    {
        using namespace AscendC::MicroAPI;

        MaskReg b16AllMask = CreateMask<uint16_t, MaskPattern::ALL>();

        MicroAPI::RegTensor<uint16_t> u16_1;
        Duplicate(u16_1, static_cast<uint16_t>(0b1));

        LocalTensor<int16_t> tempLocal = tempBuf_.Get<int16_t>();
        __ubuf__ int16_t* tempAddr = (__ubuf__ int16_t*)tempLocal.GetPhyAddr();
        MicroAPI::RegTensor<int16_t> ebShiftReg, ecShiftReg;
        DataCopy<int16_t, LoadDist::DIST_BLK>(ebShiftReg, tempAddr);
        DataCopy<int16_t, LoadDist::DIST_BLK>(ecShiftReg, tempAddr + 16);

        MicroAPI::RegTensor<float> scalesInReg;
        MicroAPI::RegTensor<uint16_t> ebEaReg, ecEcReg, regA, regB;
        MicroAPI::RegTensor<uint16_t> ebBitReg, ecBitReg, ebcReg;
        MicroAPI::RegTensor<bfloat16_t> eaReg;

        for (uint16_t loopIdx = 0; loopIdx < scaleRegLoop; loopIdx++) {
            DataCopy<float, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B32>(scalesInReg, scaleReadAddr,
                                                                                   DIGIT_EIGHT);

            DeInterleave(regA, regB, (MicroAPI::RegTensor<uint16_t>&)scalesInReg,
                         (MicroAPI::RegTensor<uint16_t>&)scalesInReg);
            Interleave<uint16_t, HighLowPart::LOWEST>(ebEaReg, regA, regA);
            Interleave<uint16_t, HighLowPart::LOWEST>(ecEcReg, regB, regB);

            Cast<bfloat16_t, fp8_e6m2_t, castTraitE6M2ToBF16>(eaReg, (MicroAPI::RegTensor<fp8_e6m2_t>&)ebEaReg,
                                                              b16AllMask);

            ShiftRight(ebBitReg, ebEaReg, ebShiftReg, b16AllMask);
            And(ebBitReg, ebBitReg, u16_1, b16AllMask);

            ShiftRight(ecBitReg, ecEcReg, ecShiftReg, b16AllMask);
            And(ecBitReg, ecBitReg, u16_1, b16AllMask);

            Add(ebcReg, ebBitReg, ecBitReg, b16AllMask);
            Adds(ebcReg, ebcReg, BF16_BIAS, b16AllMask);

            ShiftLefts(ebcReg, ebcReg, BF16_SHAFT_LEFT_NUM, b16AllMask);

            Mul(eaReg, eaReg, (MicroAPI::RegTensor<bfloat16_t>&)ebcReg, b16AllMask);

            DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(tmpScaleAddr + loopIdx * gmCopyTmpBufScaleStep_,
                                                                      eaReg, eaReg, b16AllMask);
        }
    }
}

template <typename ST_X, typename DT_Y>
__aicore__ inline void AntiHiF4QuantTail<ST_X, DT_Y>::ComputeOutPerVF(uint16_t outRegLoop, __ubuf__ uint8_t* xAddr,
                                                                      __ubuf__ bfloat16_t* tmpScaleAddr,
                                                                      __ubuf__ DT_Y* yWriteAddr)
{
    __VEC_SCOPE__
    {
        using namespace AscendC::MicroAPI;

        MaskReg b16AllMask = CreateMask<uint16_t, MaskPattern::ALL>();
        MicroAPI::RegTensor<fp4x2_e1m2_t> xInReg;
        MicroAPI::RegTensor<bfloat16_t> xBF16Reg, vregScaleTmp, vregOutbf16;
        MicroAPI::RegTensor<DT_Y> vregOut;
        for (uint16_t loopIdx = 0; loopIdx < outRegLoop; loopIdx++) {
            DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<uint8_t>&)xInReg, xAddr, DQ_HIFP4_BF16_PER_GROUP);
            Cast<bfloat16_t, fp4x2_e1m2_t, castTraitB4ToB16>(xBF16Reg, xInReg, b16AllMask);
            DataCopy<bfloat16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_US_B16>(
                vregScaleTmp, tmpScaleAddr, DQ_HIFP4_BF16_PER_GROUP);
            Mul(vregOutbf16, xBF16Reg, vregScaleTmp, b16AllMask);
            if constexpr (IsSameType<DT_Y, half>::value) {
                AscendC::MicroAPI::Cast<DT_Y, bfloat16_t, castTraitBF16ToFP16>((MicroAPI::RegTensor<DT_Y>&)vregOut,
                                                                               vregOutbf16, b16AllMask);
                MicroAPI::DataCopy<DT_Y, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM>(
                    yWriteAddr, (MicroAPI::RegTensor<DT_Y>&)vregOut, 128, b16AllMask);
            } else {
                MicroAPI::DataCopy<DT_Y, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_NORM>(
                    yWriteAddr, (MicroAPI::RegTensor<DT_Y>&)vregOutbf16, 128, b16AllMask);
            }
        }
    }
}

// ==================== ComputeAll ====================
template <typename ST_X, typename DT_Y>
__aicore__ inline void AntiHiF4QuantTail<ST_X, DT_Y>::ComputeAll(int64_t scaleRegLoop, int64_t outRegLoop)
{
    LocalTensor<uint8_t> x = inQueueX_.template DeQue<uint8_t>();
    LocalTensor<float> scale = inQueueScale_.template DeQue<float>();
    LocalTensor<DT_Y> y = outQueueY_.template AllocTensor<DT_Y>();
    LocalTensor<bfloat16_t> tmpscale = scaleTmpBuf_.Get<bfloat16_t>();

    auto xAddr = (__ubuf__ uint8_t*)x.GetPhyAddr();
    auto scaleReadAddr = (__ubuf__ float*)scale.GetPhyAddr();
    auto yWriteAddr = (__ubuf__ DT_Y*)y.GetPhyAddr();
    auto tmpScaleAddr = (__ubuf__ bfloat16_t*)tmpscale.GetPhyAddr();

    ComputeDequantPerVF(static_cast<uint16_t>(scaleRegLoop), scaleReadAddr, tmpScaleAddr);
    ComputeOutPerVF(static_cast<uint16_t>(outRegLoop), xAddr, tmpScaleAddr, yWriteAddr);
    outQueueY_.EnQue(y);
    inQueueX_.FreeTensor(x);
    inQueueScale_.FreeTensor(scale);
}

// ==================== CopyOut ====================
template <typename ST_X, typename DT_Y>
__aicore__ inline void AntiHiF4QuantTail<ST_X, DT_Y>::CopyOut(const int64_t& gmOffset)
{
    auto yLocalOut = outQueueY_.template DeQue<DT_Y>();
    DataCopyPad(yGm_[gmOffset], yLocalOut, dataCopyOutParams_);
    outQueueY_.FreeTensor(yLocalOut);
}

// ==================== Process ====================
template <typename ST_X, typename DT_Y>
__aicore__ inline void AntiHiF4QuantTail<ST_X, DT_Y>::Process()
{
    bool isLastBlockFlag = (blockIdx_ == tilingData_->usedCoreNum - 1);
    int64_t loopNum = isLastBlockFlag ? tilingData_->rowPerTailCore : tilingData_->rowPerHeadCore;
    int64_t tailNum = isLastBlockFlag ? tilingData_->multiRowNumTailCore : tilingData_->multiRowNumHeadCore;
    int64_t tailScaleRegLoop = isLastBlockFlag ? tilingData_->scaleLoopTail : tilingData_->scaleLoopTimes;
    int64_t tailOutRegLoop = isLastBlockFlag ? tilingData_->outLoopTail : tilingData_->outLoopTimes;
    // 满载块参数（对标 DynamicHiF4Quant 的 xRealFormer/yRealFormer）
    int64_t xRealFormer = tilingData_->alignGroupNum * DQ_HIFP4_GROUP_BYTE_NUM;
    int64_t scaleRealFormer = tilingData_->alignGroupNum * DQ_HIFP4_SCALE_BYTE_NUM;
    int64_t yRealFormer = tilingData_->alignGroupNum * DQ_HIFP4_BF16_PER_GROUP * sizeof(bfloat16_t);

    dataCopyInParams_.blockLen = xRealFormer;
    dataCopyInScaleParams_.blockLen = scaleRealFormer;
    dataCopyOutParams_.blockLen = yRealFormer;

    int64_t xGmOffset = 0;
    int64_t scaleGmOffset = 0;
    int64_t yGmOffset = 0;

    // 满载 UB 循环
    for (int64_t i = 0; i < loopNum - 1; ++i) {
        CopyIn(xGmOffset, scaleGmOffset);
        ComputeAll(tilingData_->scaleLoopEle, tilingData_->outLoopEle);
        CopyOut(yGmOffset);

        xGmOffset += xRealFormer;
        scaleGmOffset += tilingData_->alignGroupNum * DQ_HIFP4_SCALE_FP32_NUM;
        yGmOffset += tilingData_->alignGroupNum * DQ_HIFP4_BF16_PER_GROUP;
    }

    // 尾块
    dataCopyInParams_.blockLen = tailNum * DQ_HIFP4_GROUP_BYTE_NUM;
    dataCopyInScaleParams_.blockLen = tailNum * DQ_HIFP4_SCALE_BYTE_NUM;
    dataCopyOutParams_.blockLen = tailNum * DQ_HIFP4_BF16_PER_GROUP * sizeof(bfloat16_t);
    CopyIn(xGmOffset, scaleGmOffset);
    ComputeAll(tailScaleRegLoop, tailOutRegLoop);
    CopyOut(yGmOffset);
}

} // namespace AntiHiF4Quant
#endif
