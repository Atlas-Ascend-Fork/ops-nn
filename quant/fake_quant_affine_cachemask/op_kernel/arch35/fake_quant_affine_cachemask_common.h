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
 * \file fake_quant_affine_cachemask_common.h
 * \brief 共用常量、Cast trait 引用、VL 常量、mode 枚举
 *
 * 三种 mode 实现拆分到独立头：
 *   - fake_quant_affine_cachemask_pt.h
 *   - fake_quant_affine_cachemask_pc.h
 *   - fake_quant_affine_cachemask_ph.h
 *
 * 主头 fake_quant_affine_cachemask.h 串联以上三份并提供顶层
 * `FakeQuantAffineCachemaskOp<T, MODE, HAS_ZP>::Run(...)` 入口。
 */
#ifndef FAKE_QUANT_AFFINE_CACHEMASK_COMMON_H
#define FAKE_QUANT_AFFINE_CACHEMASK_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "fake_quant_affine_cachemask_tiling_data.h"
#include "fake_quant_affine_cachemask_tiling_key.h"

namespace NsFakeQuantAffineCachemask {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

constexpr int64_t MODE_PT = 0;
constexpr int64_t MODE_PC = 1;
constexpr int64_t MODE_PC_NDDMA = 2;
constexpr int64_t MODE_PH = 3;

// VL（一次 VF 处理的 fp32 数量）= VECTOR_REG_WIDTH / sizeof(fp32) = 256/4 = 64
constexpr int32_t VL_FP32 = 256 / sizeof(float);

// AscendC 提供的 CastTrait 常量（在 AscendC::namespace，定义于 dav_c310 vconv impl）
//   layoutZSatSMrgZRndR : fp32 → int32 / fp32 → fp16 / fp32 → bf16（带 Sat + RINT）
//   layoutZMrgZRndR     : int32 → fp32（指定 RoundMode::R，不允许 UNKNOWN）
//   layoutZMrgZ         : 同位宽 / fp16<->fp32 等无 round 的变换

template <typename T, typename ZpT>
struct CommonBuffers {
    TBuf<TPosition::VECCALC> tmpFp32Buf_;
    TBuf<TPosition::VECCALC> qFp32Buf_;
    TBuf<TPosition::VECCALC> m1Fp32Buf_;
    TBuf<TPosition::VECCALC> m2Fp32Buf_;
    TBuf<TPosition::VECCALC> halfMaskBuf_;

    GlobalTensor<T> inputGMX_;
    GlobalTensor<T> inputGMScale_;
    GlobalTensor<ZpT> inputGMZp_;
    GlobalTensor<T> outputGMY_;
    GlobalTensor<uint8_t> outputGMMask_;
};

// zero_point dtype 分支判据（REQUIREMENTS §2.1.1 / §4.6.4，与 PyTorch aten 逐位对齐）：
//   - 整数 zp  : qval = rint(x/s) + zp        （先 round 再加 zp）
//   - 浮点 zp  : qval = rint(x/s + zp)        （先加 zp 再 round，lrintf 路径）
template <typename ZpT>
constexpr bool ZP_IS_INTEGRAL = std::is_same_v<ZpT, int32_t> || std::is_same_v<ZpT, int16_t> ||
                                std::is_same_v<ZpT, int64_t>;

// 特殊值口径（spec.yaml formula notes / DESIGN §1.4）：x/s (+zp) 为非有限值（NaN/±inf）时，
// 经 int 转换须饱和到 INT32_MIN（等价 golden 的 non-finite → INT64_MIN），随后被
// min/max 钳到 quant_min、对应 mask = False。
// VF 的 Cast+RINT+Sat 对 NaN 不产生 INT32_MIN（实测为 0），故在 Cast 前显式做一次
// 非有限值检测并替换为 -FLT_MAX（其 RINT+Sat 结果即 INT32_MIN）：
//   有限 <=> (v > -FLT_MAX) && (v < +FLT_MAX)；NaN/±inf 均不满足，替换为 -FLT_MAX。
__aicore__ inline void ForceNonFiniteToSentinel(Reg::RegTensor<float>& vreg, Reg::MaskReg mask)
{
    constexpr float FLT_MAX_F = 3.4028234663852886e+38F;
    Reg::MaskReg mGt;
    Reg::MaskReg mLt;
    Reg::MaskReg mFinite;
    Reg::RegTensor<float> vregSentinel;
    Reg::Compares<float, CMPMODE::GT>(mGt, vreg, -FLT_MAX_F, mask);
    Reg::Compares<float, CMPMODE::LT>(mLt, vreg, FLT_MAX_F, mask);
    Reg::And(mFinite, mGt, mLt, mask);
    Reg::Duplicate<float>(vregSentinel, -FLT_MAX_F);
    Reg::Select(vreg, vreg, vregSentinel, mFinite); // mask 位为 1 取 src0（有限值），否则取哨兵
}

template <typename T, typename ZpT, int HAS_ZP>
__aicore__ inline void ComputeQuantMaskBody(Reg::RegTensor<float>& vregTmp, Reg::RegTensor<float>& vregSc,
                                            Reg::RegTensor<float>& vregZp, Reg::RegTensor<float>& vregQ,
                                            Reg::RegTensor<float>& vregM1, Reg::RegTensor<float>& vregM2,
                                            Reg::RegTensor<int32_t>& vregInt32, Reg::MaskReg mask, uint16_t i,
                                            __ubuf__ float* m1Ptr, __ubuf__ T* yPtr, float qMinF, float qMaxF,
                                            float qMinMinus1F, float qMaxPlus1F)
{
    // vregTmp 入参约定：整数 zp 为 x/s（round 前不加 zp）；浮点 zp 为 x/s + zp
    Reg::Cast<int32_t, float, layoutZSatSMrgZRndR>(vregInt32, vregTmp, mask);
    Reg::Cast<float, int32_t, layoutZMrgZRndR>(vregQ, vregInt32, mask);
    if constexpr (HAS_ZP == 1 && ZP_IS_INTEGRAL<ZpT>) {
        // 整数 zp：qval = rint(x/s) + zp（round 之后加，整数 zp 值在 fp32 中精确表示）
        Reg::Add<float>(vregQ, vregQ, vregZp, mask);
    }
    Reg::Adds<float, float>(vregM1, vregQ, -qMinMinus1F, mask);
    Reg::Mins<float, float>(vregM1, vregM1, 1.0f, mask);
    Reg::Maxs<float, float>(vregM1, vregM1, 0.0f, mask);
    Reg::Muls<float, float>(vregM2, vregQ, -1.0f, mask);
    Reg::Adds<float, float>(vregM2, vregM2, qMaxPlus1F, mask);
    Reg::Mins<float, float>(vregM2, vregM2, 1.0f, mask);
    Reg::Maxs<float, float>(vregM2, vregM2, 0.0f, mask);
    Reg::Mul<float>(vregM1, vregM1, vregM2, mask);
    Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(m1Ptr + i * VL_FP32, vregM1, mask);
    Reg::Maxs<float, float>(vregTmp, vregQ, qMinF, mask);
    Reg::Mins<float, float>(vregTmp, vregTmp, qMaxF, mask);
    if constexpr (HAS_ZP == 1) {
        Reg::Sub<float>(vregTmp, vregTmp, vregZp, mask);
    }
    Reg::Mul<float>(vregTmp, vregTmp, vregSc, mask);
    if constexpr (std::is_same<T, float>::value) {
        Reg::DataCopy<float, Reg::StoreDist::DIST_NORM_B32>(yPtr + i * VL_FP32, vregTmp, mask);
    } else {
        Reg::RegTensor<T> vregY;
        Reg::Cast<T, float, layoutZSatSMrgZRndR>(vregY, vregTmp, mask);
        Reg::DataCopy<T, Reg::StoreDist::DIST_PACK_B32>(yPtr + i * VL_FP32, vregY, mask);
    }
}

template <typename T>
__aicore__ inline void FinalizeCompute(AscendC::LocalTensor<float>& m1Buf, AscendC::LocalTensor<half>& hBuf,
                                       AscendC::LocalTensor<uint8_t>& mLocal, AscendC::LocalTensor<T>& yLocal,
                                       AscendC::LocalTensor<T>& xLocal,
                                       TQue<QuePosition::VECOUT, BUFFER_NUM>& outQueueMask,
                                       TQue<QuePosition::VECOUT, BUFFER_NUM>& outQueueY,
                                       TQue<QuePosition::VECIN, BUFFER_NUM>& inQueueX, int64_t dataCount)
{
    AscendC::Cast<half, float>(hBuf, m1Buf, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(dataCount));
    AscendC::Cast<uint8_t, half>(mLocal, hBuf, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(dataCount));
    outQueueMask.template EnQue<uint8_t>(mLocal);
    outQueueY.template EnQue<T>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

} // namespace NsFakeQuantAffineCachemask

#endif // FAKE_QUANT_AFFINE_CACHEMASK_COMMON_H
