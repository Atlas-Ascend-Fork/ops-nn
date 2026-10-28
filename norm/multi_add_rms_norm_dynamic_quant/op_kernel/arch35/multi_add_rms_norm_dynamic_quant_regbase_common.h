/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file multi_add_rms_norm_dynamic_quant_regbase_common.h
 * \brief
 */
#ifndef MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_COMMON_H_
#define MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_COMMON_H_

#include <cmath>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "../../rms_norm/arch35/rms_norm_regbase_common.h"

#ifndef FLOAT_OVERFLOW_MODE_CTRL
#define FLOAT_OVERFLOW_MODE_CTRL 60
#endif

namespace MultiAddRmsNormDynamicQuant {
using namespace AscendC;
using namespace AscendC::Reg;
using namespace NormCommon;
using namespace NormCommon::NormCommonRegbase;
using RmsNorm::castTraitFp322Fp8;
using RmsNorm::castTraitFp322Hifp8;
using RmsNorm::CopyInX;
using RmsNorm::CopyOutX;
using RmsNorm::CopyOutY;
using RmsNorm::GetOverflowMode;
using RmsNorm::SetOverflowMode;
using RmsNorm::YCopyOutImpl;

constexpr uint64_t ALIGN_512_FACTOR = 512;
constexpr uint64_t ALIGN_32_FACTOR = 32;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t B32_BLOCK_NUM = 8;
constexpr uint64_t B8_BLOCK_NUM = 32;
constexpr int32_t CONST_FACTOR_2 = 2;
constexpr uint32_t SUM_COUNT = 2;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_ONE = 1;
constexpr uint8_t UNIT_64_BYTE_POW = 3;

// multi-add:从 x1 TensorList 的 GM_ADDR 取第 index 个张量地址(参照 ascend910b base kernel GetTensorAddr)
template <typename T>
__aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> UNIT_64_BYTE_POW);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}

constexpr int32_t VL_SIZE = GetVRegSize();
constexpr int32_t V_LENGTH = (VL_SIZE / static_cast<int32_t>(sizeof(float)));
constexpr uint32_t NUM_TWO_DF = 2;  // 两段式(hi/lo)用
constexpr uint32_t NUM_FOUR_DF = 4; // 两段式累加缓冲: 每段留 2*V_LENGTH 地址空间(整 VL 载入余量)
constexpr float MAX_FINITE_FP32 = 3.402823466e+38F;
constexpr float DIV_FACTOR_INT8 = (static_cast<float>(1.0) / 127);
constexpr float DIV_FACTOR_FP8E4M3FN = (static_cast<float>(1.0) / 448);
constexpr float DIV_FACTOR_FP8E5M2 = (static_cast<float>(1.0) / 57344);
constexpr float DIV_FACTOR_HIFP8 = (static_cast<float>(1.0) / 32768);

constexpr AscendC::Reg::CastTrait castTraitFp322Fp16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::Reg::CastTrait castTraitFp162Int8 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,
};

template <typename TilingData>
__aicore__ inline void InitTiling(uint64_t& numM, uint64_t& numN, uint64_t& baseM, uint64_t& baseN,
                                  uint64_t& baseNDtypeAlign, uint64_t& baseNReduceAlign, uint64_t& powerSplit,
                                  uint64_t& mPerCore, uint64_t& mLastCore, float& epsilon, float& avgFactor,
                                  const TilingData* tilingData)
{
    numM = tilingData->numM;
    numN = tilingData->numN;
    baseM = tilingData->baseM;
    baseN = tilingData->baseN;
    baseNDtypeAlign = tilingData->baseNDtypeAlign;
    baseNReduceAlign = tilingData->baseNReduceAlign;
    powerSplit = tilingData->powerSplit;
    mPerCore = tilingData->mPerCore;
    mLastCore = tilingData->mLastCore;
    epsilon = tilingData->epsilon;
    avgFactor = tilingData->avgFactor;
}

__aicore__ inline void CopyOutScale(GlobalTensor<float>& scaleGm, TQue<QuePosition::VECOUT, 1>& outQueueScale,
                                    uint64_t gmOffset, uint32_t blockLen)
{
    LocalTensor<float> scaleLocal = outQueueScale.DeQue<float>();
    RmsNorm::DataCopyImpl<float>(scaleGm[gmOffset], scaleLocal, 1, blockLen);
    outQueueScale.FreeTensor(scaleLocal);
}

template <typename T, typename Queue>
__aicore__ inline void CopyInParamToQueue(Queue& inQueue, GlobalTensor<T>& srcGm, uint64_t numN)
{
    LocalTensor<T> local = inQueue.template AllocTensor<T>();
    RmsNorm::DataCopyImpl<T>(local, srcGm, 1, numN);
    inQueue.EnQue(local);
}

template <typename T_SMOOTH_SCALE, typename SmoothScale1Queue, typename SmoothScale2Queue>
__aicore__ inline void CopyInDynamicQuantCommon(SmoothScale1Queue& inQueueSmoothScale1,
                                                SmoothScale2Queue& inQueueSmoothScale2,
                                                GlobalTensor<T_SMOOTH_SCALE>& smoothScale1Gm,
                                                GlobalTensor<T_SMOOTH_SCALE>& smoothScale2Gm, uint64_t numN,
                                                bool hasSmoothScale1, bool hasSmoothScale2)
{
    if (hasSmoothScale1) {
        CopyInParamToQueue(inQueueSmoothScale1, smoothScale1Gm, numN);
    }
    if (hasSmoothScale2) {
        CopyInParamToQueue(inQueueSmoothScale2, smoothScale2Gm, numN);
    }
}

template <typename T_X, typename T_SMOOTH_SCALE, typename SmoothScale1Queue, typename SmoothScale2Queue,
          typename BetaQueue>
__aicore__ inline void PrepareOptionalParamLocals(SmoothScale1Queue& inQueueSmoothScale1,
                                                  SmoothScale2Queue& inQueueSmoothScale2, BetaQueue& inQueueBeta,
                                                  GlobalTensor<T_X>& betaGm,
                                                  LocalTensor<T_SMOOTH_SCALE>& smoothScale1Local,
                                                  LocalTensor<T_SMOOTH_SCALE>& smoothScale2Local,
                                                  LocalTensor<T_X>& betaLocal, uint64_t numN, bool hasSmoothScale1,
                                                  bool hasSmoothScale2, bool hasBeta)
{
    if (hasSmoothScale1) {
        smoothScale1Local = inQueueSmoothScale1.template DeQue<T_SMOOTH_SCALE>();
    }
    if (hasSmoothScale2) {
        smoothScale2Local = inQueueSmoothScale2.template DeQue<T_SMOOTH_SCALE>();
    }
    if (hasBeta) {
        CopyInParamToQueue(inQueueBeta, betaGm, numN);
        betaLocal = inQueueBeta.template DeQue<T_X>();
    }
}

template <typename T_X, typename T_Y, typename T_SMOOTH_SCALE>
__aicore__ inline void InitOptionalGmBuffers(GlobalTensor<T_SMOOTH_SCALE>& smoothScale1Gm,
                                             GlobalTensor<T_SMOOTH_SCALE>& smoothScale2Gm, GlobalTensor<T_Y>& y2Gm,
                                             GlobalTensor<float>& scale2Gm, GlobalTensor<T_X>& betaGm,
                                             GM_ADDR smoothScale1, GM_ADDR smoothScale2, GM_ADDR y2, GM_ADDR scale2,
                                             GM_ADDR beta, uint64_t gmOffset, uint64_t gmLen, uint64_t scalesGmOffset,
                                             uint64_t mCore, uint64_t numN, bool hasSmoothScale1, bool hasSmoothScale2,
                                             bool hasY2Scale2, bool hasBeta)
{
    if (hasSmoothScale1) {
        smoothScale1Gm.SetGlobalBuffer((__gm__ T_SMOOTH_SCALE*)smoothScale1, numN);
    }
    if (hasSmoothScale2) {
        smoothScale2Gm.SetGlobalBuffer((__gm__ T_SMOOTH_SCALE*)smoothScale2, numN);
    }
    if (hasY2Scale2) {
        y2Gm.SetGlobalBuffer((__gm__ T_Y*)y2 + gmOffset, gmLen);
        scale2Gm.SetGlobalBuffer((__gm__ float*)scale2 + scalesGmOffset, mCore);
    }
    if (hasBeta) {
        betaGm.SetGlobalBuffer((__gm__ T_X*)beta, numN);
    }
}

template <typename T_X, typename T_GAMMA, typename T_SMOOTH_SCALE = float, bool HAS_SMOOTH_SCALE = true,
          bool HAS_BETA = false, typename T_Y>
__aicore__ inline void ComputeYScale(LocalTensor<T_Y>& yLocal, LocalTensor<float>& scaleLocal, LocalTensor<T_X>& xLocal,
                                     LocalTensor<float>& rstdLocal, LocalTensor<T_GAMMA>& gammaLocal,
                                     LocalTensor<T_GAMMA>& betaLocal, LocalTensor<T_SMOOTH_SCALE>& smoothScaleLocal,
                                     LocalTensor<float>& yTmpLocal, uint32_t rstdScaleOffset, uint32_t calCount)
{
    uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);

    __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
    __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
    __local_mem__ T_SMOOTH_SCALE* smoothScaleAddr;
    if constexpr (HAS_SMOOTH_SCALE) {
        smoothScaleAddr = (__ubuf__ T_SMOOTH_SCALE*)smoothScaleLocal.GetPhyAddr();
    }
    __local_mem__ T_GAMMA* betaAddr;
    if constexpr (HAS_BETA) {
        betaAddr = (__ubuf__ T_GAMMA*)betaLocal.GetPhyAddr();
    }
    __local_mem__ T_Y* yAddr = (__ubuf__ T_Y*)yLocal.GetPhyAddr();
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

        Duplicate(scaleReg, static_cast<float>(-INFINITY), maskRegFull); // Abs before reducemax, scaleReg >= 0
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + rstdScaleOffset);
        for (uint16_t idx = 0; idx < (uint16_t)repeatTimes; idx++) {
            maskReg = UpdateMask<float>(calCount);
            NormCommon::LoadCastRegVF(xRegFp32, xAddr, idx, maskReg);
            NormCommon::LoadCastRegVF(gammaRegFp32, gammaAddr, idx, maskReg);
            if constexpr (HAS_SMOOTH_SCALE) {
                NormCommon::LoadCastRegVF(smoothScaleRegFp32, smoothScaleAddr, idx, maskReg);
            }
            if constexpr (HAS_BETA) {
                NormCommon::LoadCastRegVF(betaRegFp32, betaAddr, idx, maskReg);
            }
            Mul(xRegFp32, xRegFp32, rstdReg, maskReg);
            Mul(xRegFp32, xRegFp32, gammaRegFp32, maskReg);
            if constexpr (HAS_BETA) {
                Add(xRegFp32, xRegFp32, betaRegFp32, maskReg);
            }
            if constexpr (HAS_SMOOTH_SCALE) {
                Mul(yRegFp32, xRegFp32, smoothScaleRegFp32, maskReg);
                DataCopy<float>(yTmpAddr + idx * V_LENGTH, yRegFp32, maskReg);
                Abs(yRegFp32, yRegFp32, maskReg);               // VF abs is zeroing mode
                Max(scaleReg, scaleReg, yRegFp32, maskRegFull); // Using full mask
            } else {
                DataCopy<float>(yTmpAddr + idx * V_LENGTH, xRegFp32, maskReg);
                Abs(yRegFp32, xRegFp32, maskReg);               // VF abs is zeroing mode
                Max(scaleReg, scaleReg, yRegFp32, maskRegFull); // Using full mask
            }
        }
        ReduceMax(scaleReg, scaleReg, maskRegFull);
        if constexpr (IsSameType<T_Y, int8_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_INT8, maskRegOne);
        } else if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_FP8E4M3FN, maskRegOne);
        } else if constexpr (IsSameType<T_Y, fp8_e5m2_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_FP8E5M2, maskRegOne);
        } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
            Muls(scaleReg, scaleReg, DIV_FACTOR_HIFP8, maskRegOne);
        }
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + rstdScaleOffset, scaleReg, maskRegOne);
    }
    PipeBarrier<PIPE_V>();

    uint16_t repeatTimes2 = (uint16_t)CeilDivision(calCount, V_LENGTH);
    __VEC_SCOPE__
    {
        // VF1. Calc y
        RegTensor<float> yRegFp32, yRegFp32Tmp, scaleReg;
        RegTensor<half> yRegFp16;
        RegTensor<T_Y> yReg;
        MaskReg maskReg;

        DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, scaleAddr + rstdScaleOffset);
        for (uint16_t idx = 0; idx < (uint16_t)repeatTimes2; idx++) {
            maskReg = UpdateMask<float>(calCount);
            DataCopy<float>(yRegFp32, yTmpAddr + idx * V_LENGTH);
            Div(yRegFp32, yRegFp32, scaleReg, maskReg);
            if constexpr (IsSameType<T_Y, int8_t>::value) {
                Truncate<float, RoundMode::CAST_RINT>(yRegFp32Tmp, yRegFp32, maskReg);
                Cast<half, float, castTraitFp322Fp16>(yRegFp16, yRegFp32Tmp, maskReg);
                Cast<T_Y, half, castTraitFp162Int8>(yReg, yRegFp16, maskReg);
            } else if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value) {
                Cast<T_Y, float, castTraitFp322Fp8>(yReg, yRegFp32, maskReg);
            } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
                Cast<T_Y, float, castTraitFp322Hifp8>(yReg, yRegFp32, maskReg);
            }
            DataCopy<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + idx * V_LENGTH, yReg, maskReg);
        }
    }
}

template <typename T_X, typename T_GAMMA, typename T_SMOOTH_SCALE = float, bool HAS_SMOOTH_SCALE = true,
          bool HAS_BETA = false, typename T_Y>
__aicore__ inline void ComputeReduceMax(LocalTensor<float>& scaleLocal, LocalTensor<float>& yTmpLocal,
                                        LocalTensor<T_X>& xLocal, LocalTensor<float>& rstdLocal,
                                        LocalTensor<T_GAMMA>& gammaLocal, LocalTensor<T_GAMMA>& betaLocal,
                                        LocalTensor<T_SMOOTH_SCALE>& smoothScaleLocal, uint32_t rstdScaleOffset,
                                        uint32_t calCount)
{
    uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);

    __local_mem__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
    __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
    __local_mem__ T_GAMMA* betaAddr;
    if constexpr (HAS_BETA) {
        betaAddr = (__ubuf__ T_GAMMA*)betaLocal.GetPhyAddr();
    }
    __local_mem__ T_SMOOTH_SCALE* smoothScaleAddr;
    if constexpr (HAS_SMOOTH_SCALE) {
        smoothScaleAddr = (__ubuf__ T_SMOOTH_SCALE*)smoothScaleLocal.GetPhyAddr();
    }
    __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ float* yTmpAddr = (__ubuf__ float*)yTmpLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> rstdReg, scaleReg, scaleLastReg;
        RegTensor<float> xRegFp32, yRegFp32, gammaRegFp32, betaRegFp32, smoothScaleRegFp32;
        MaskReg maskRegFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskRegOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg maskReg;

        Duplicate(scaleReg, static_cast<float>(-INFINITY), maskRegFull); // Abs before reducemax, scaleReg >= 0
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + rstdScaleOffset);
        DataCopy<float>(scaleLastReg, scaleAddr + rstdScaleOffset);
        for (uint16_t idx = 0; idx < (uint16_t)repeatTimes; idx++) {
            maskReg = UpdateMask<float>(calCount);
            NormCommon::LoadCastRegVF(xRegFp32, xAddr, idx, maskReg);
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
                // 物化 inp = y*smooth_scale(与 A2 的 ScaleTensor 及 add_rms_norm_dynamic_quant 同形态)。
                // 不走"只存 y、由 ComputeY 以 k=(smooth_scale*quantMax)/row_max 合成"的尺度不变写法:
                // k 是该写法自造的中间量, row_max 落次正规区且 smooth_scale 为正常量级时
                // k = smooth_scale*quantMax/row_max 会越过 fp32 上限变 inf, 使输出整体饱和
                // (实测 smooth_scale~0.5、row_max~2e-38 时 k~3.2e39 > 3.4e38)。
                // 而 inp/scale 恒不溢出(|inp| <= row_max), 次正规分母的 DAZ 由 ComputeY 的 pre 同乘化解。
                Mul(yRegFp32, xRegFp32, smoothScaleRegFp32, maskReg);
                DataCopy<float>(yTmpAddr + idx * V_LENGTH, yRegFp32, maskReg); // 存 inp = y*smooth
                Abs(yRegFp32, yRegFp32, maskReg);                              // VF abs is zeroing mode
                Max(scaleReg, scaleReg, yRegFp32, maskRegFull);                // row_max = max|y*smooth|
            } else {
                DataCopy<float>(yTmpAddr + idx * V_LENGTH, xRegFp32, maskReg);
                Abs(yRegFp32, xRegFp32, maskReg);               // VF abs is zeroing mode
                Max(scaleReg, scaleReg, yRegFp32, maskRegFull); // Using full mask
            }
        }
        ReduceMax(scaleReg, scaleReg, maskRegFull);
        Max(scaleReg, scaleReg, scaleLastReg, maskRegOne);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + rstdScaleOffset, scaleReg, maskRegOne);
    }
}

// ======================= 两段式(double-float)基元 =======================
// 适用面只有 Sigma x^2 与由它得出的 rstd: 长链累加每级一次 fp32 舍入, 残差会被 rstd 乘回
// 每个元素, 故这两级用两段式(hi+lo)补偿。
// **不用于 row_max**: max 不做算术、不产生舍入误差, 补偿它没有收益; 而补偿式
// r = hi - s0*qmax 在 hi 非有限时退化为 inf-inf = NaN, 会把本该为 inf 的 scale 变成 NaN。
// 基元只用加/减/乘与单次舍入 FMA(FusedMulDstAdd) 表达, 不依赖 fp64。
// 注意: MulAddDst 不保证单次舍入, 不能用来取残差。

// 残差非有限即置 0。高位段溢出成 inf 时残差退化为 inf-inf=NaN(TwoProd 侧为 -inf),
// 把它加回去会把本该是 inf 的和污染成 NaN, 进而让 rstd/normed/scale 整列变 NaN。
// 补偿求和的意义只在有限域, 非有限值按 IEEE 自然传播即可 —— 与 l2_normalize_grad 的
// TwoSumVec、in_training_update_grad_gamma_beta 的 AddWithResidual 同一处理。
__simd_callee__ inline void DfClampResidual(RegTensor<float>& e, MaskReg& mask)
{
    RegTensor<float> absE, zeroReg;
    MaskReg finiteMask;
    Abs(absE, e, mask);
    Compares<float, CMPMODE::LE>(finiteMask, absE, MAX_FINITE_FP32, mask);
    Duplicate(zeroReg, 0.0F, mask);
    Select(e, e, zeroReg, finiteMask);
}

// TwoProd: p = fl(a*b), e = a*b - p(精确)。依赖 FMA 单次舍入。
__simd_callee__ inline void DfTwoProd(RegTensor<float>& p, RegTensor<float>& e, RegTensor<float>& a,
                                      RegTensor<float>& b, MaskReg& mask)
{
    RegTensor<float> negP;
    Mul(p, a, b, mask);
    Muls(negP, p, static_cast<float>(-1.0), mask);
    Muls(e, a, static_cast<float>(1.0), mask); // e = a(拷贝, 乘 1 精确)
    FusedMulDstAdd(e, b, negP, mask);          // e = a*b - p
    DfClampResidual(e, mask);
}

// TwoSum(Knuth): s = fl(a+b), e = (a+b) - s(精确), 不要求 |a|>=|b|。
__simd_callee__ inline void DfTwoSum(RegTensor<float>& s, RegTensor<float>& e, RegTensor<float>& a, RegTensor<float>& b,
                                     MaskReg& mask)
{
    RegTensor<float> bb, aa, da, db;
    Add(s, a, b, mask);
    Sub(bb, s, a, mask);
    Sub(aa, s, bb, mask);
    Sub(da, a, aa, mask);
    Sub(db, b, bb, mask);
    Add(e, da, db, mask);
    DfClampResidual(e, mask);
}

template <typename T_Y>
__aicore__ inline void ComputeScale(LocalTensor<float>& scaleLocal, uint32_t rstdScaleOffset)
{
    __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> scaleReg, qmaxReg;
        MaskReg maskRegOne = CreateMask<float, MaskPattern::VL1>();

        // scale = row_max/quantMax 取 Div 形态: 与 ComputeY 的分母是同一个值, 也与算法公式
        // scale = row_max(abs(inp))/127 字面一致。乘 1/quantMax 会与分母差 1 ULP(1/127 非 2 的幂),
        // 使输出的 scale 不再等于实际用于量化的分母。
        float quantMaxV = static_cast<float>(1.0) / DIV_FACTOR_INT8;
        if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value) {
            quantMaxV = static_cast<float>(1.0) / DIV_FACTOR_FP8E4M3FN;
        } else if constexpr (IsSameType<T_Y, fp8_e5m2_t>::value) {
            quantMaxV = static_cast<float>(1.0) / DIV_FACTOR_FP8E5M2;
        } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
            quantMaxV = static_cast<float>(1.0) / DIV_FACTOR_HIFP8;
        }
        DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, scaleAddr + rstdScaleOffset);
        Duplicate(qmaxReg, quantMaxV, maskRegOne);
        Div(scaleReg, scaleReg, qmaxReg, maskRegOne);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(scaleAddr + rstdScaleOffset, scaleReg, maskRegOne);
    }
}

// xLocal 存量化前的 inp(无 smooth 时即归一化后的 y, 有 smooth 时为 y*smooth, 均由
// ComputeReduceMax 落盘), scaleLocal[offset] 存 row_max(非 scale)。
// 算 out = inp/(row_max/quantMax), 即算法公式的 round(inp/scale)。
// 调用方须在本函数之后再调 ComputeScale 把 row_max 换算成真实 scale。
template <typename T_Y>
__aicore__ inline void ComputeY(LocalTensor<T_Y>& yLocal, LocalTensor<float>& scaleLocal, LocalTensor<float>& xLocal,
                                uint32_t rstdScaleOffset, uint32_t calCount)
{
    uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);

    __local_mem__ T_Y* yAddr = (__ubuf__ T_Y*)yLocal.GetPhyAddr();
    __local_mem__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    __local_mem__ float* xAddr = (__ubuf__ float*)xLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> yRegFp32, yRegFp32Tmp, scaleReg;
        RegTensor<half> yRegFp16;
        RegTensor<T_Y> yReg;
        MaskReg maskReg;
        MaskReg maskRegAll = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> preReg, rmScaled, denReg, bigReg, oneReg, qmaxReg;

        // row_max 落 fp32 次正规区时, 矢量 Div 的分母会被 DAZ 当 0 -> 商 inf -> 输出整体饱和。
        // 用 PRE_SCALE(2 的幂, 精确)把分子分母**同时**搬正: (inp*PRE)/((row_max*PRE)/qmax)。
        // 分子恒不溢出 —— |inp| <= row_max, 故 |inp*PRE| <= row_max*PRE, 与分母同界;
        // 这是不先合成 k=(ss*qmax)/row_max 的理由: 那个中间量在 row_max 次正规而 ss 为
        // 正常量级时会越过 fp32 上限变 inf(实测 ss~0.5、row_max~2e-38 -> k~3.2e39)。
        constexpr float SUBNORM_GUARD = 1.0e-30f;
        constexpr float PRE_SCALE = 1208925819614629174706176.0f; // 2^80
        float quantMaxV = static_cast<float>(1.0) / DIV_FACTOR_INT8;
        if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value) {
            quantMaxV = static_cast<float>(1.0) / DIV_FACTOR_FP8E4M3FN;
        } else if constexpr (IsSameType<T_Y, fp8_e5m2_t>::value) {
            quantMaxV = static_cast<float>(1.0) / DIV_FACTOR_FP8E5M2;
        } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
            quantMaxV = static_cast<float>(1.0) / DIV_FACTOR_HIFP8;
        }
        DataCopy<float, LoadDist::DIST_BRC_B32>(scaleReg, scaleAddr + rstdScaleOffset); // row_max
        Duplicate(bigReg, PRE_SCALE, maskRegAll);
        Duplicate(oneReg, static_cast<float>(1.0), maskRegAll);
        Duplicate(qmaxReg, quantMaxV, maskRegAll);
        MaskReg isSmall;
        CompareScalar<float, CMPMODE::LT>(isSmall, scaleReg, SUBNORM_GUARD, maskRegAll);
        Select(preReg, bigReg, oneReg, isSmall);
        Mul(rmScaled, scaleReg, preReg, maskRegAll);
        // 分母 = (row_max*PRE)/qmax = scale*PRE(与 ComputeScale 输出的 scale 同为 Div 形态)
        Div(rmScaled, rmScaled, qmaxReg, maskRegAll);
        MaskReg isZero;
        CompareScalar<float, CMPMODE::EQ>(isZero, rmScaled, static_cast<float>(0.0), maskRegAll);
        // 零行(整行 inp 恒为 0): 分母换 1 -> out = 0/1 = 0, 规避 0/0; 与算法公式一致
        Select(denReg, oneReg, rmScaled, isZero);
        for (uint16_t idx = 0; idx < (uint16_t)repeatTimes; idx++) {
            maskReg = UpdateMask<float>(calCount);
            DataCopy<float>(yRegFp32, xAddr + idx * V_LENGTH);
            Mul(yRegFp32, yRegFp32, preReg, maskReg);
            Div(yRegFp32, yRegFp32, denReg, maskReg);
            if constexpr (IsSameType<T_Y, int8_t>::value) {
                Truncate<float, RoundMode::CAST_RINT>(yRegFp32Tmp, yRegFp32, maskReg);
                Cast<half, float, castTraitFp322Fp16>(yRegFp16, yRegFp32Tmp, maskReg);
                Cast<T_Y, half, castTraitFp162Int8>(yReg, yRegFp16, maskReg);
            } else if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value) {
                Cast<T_Y, float, castTraitFp322Fp8>(yReg, yRegFp32, maskReg);
            } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
                Cast<T_Y, float, castTraitFp322Hifp8>(yReg, yRegFp32, maskReg);
            }
            DataCopy<T_Y, StoreDist::DIST_PACK4_B32>(yAddr + idx * V_LENGTH, yReg, maskReg);
        }
    }
}

// 增量输出:y = xOut * rstd * gamma(RmsNorm 结果,量化前),cast 回 T_X 写出。
// xLocal 为 fp32 的 xOut(=Σx1+x2),rstdLocal[rstdOffset] 为该行 rstd。
template <typename T_X, typename T_GAMMA>
__aicore__ inline void ComputeYOut(LocalTensor<T_X>& yLocal, LocalTensor<float>& xLocal, LocalTensor<float>& rstdLocal,
                                   LocalTensor<T_GAMMA>& gammaLocal, uint32_t rstdOffset, uint32_t calCount)
{
    uint16_t repeatTimes = (uint16_t)CeilDivision(calCount, V_LENGTH);

    __local_mem__ float* xAddr = (__ubuf__ float*)xLocal.GetPhyAddr();
    __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ T_GAMMA* gammaAddr = (__ubuf__ T_GAMMA*)gammaLocal.GetPhyAddr();
    __local_mem__ T_X* yAddr = (__ubuf__ T_X*)yLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        RegTensor<float> rstdReg, xRegFp32, gammaRegFp32;
        MaskReg maskReg;

        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + rstdOffset);
        for (uint16_t idx = 0; idx < (uint16_t)repeatTimes; idx++) {
            maskReg = UpdateMask<float>(calCount);
            DataCopy<float>(xRegFp32, xAddr + idx * V_LENGTH);
            NormCommon::LoadCastRegVF(gammaRegFp32, gammaAddr, idx, maskReg);
            Mul(xRegFp32, xRegFp32, rstdReg, maskReg);
            Mul(xRegFp32, xRegFp32, gammaRegFp32, maskReg);
            StoreRegForDtype<T_X>(yAddr, xRegFp32, maskReg, idx * V_LENGTH);
        }
    }
}

} // namespace MultiAddRmsNormDynamicQuant
#endif // _MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_COMMON_H_
