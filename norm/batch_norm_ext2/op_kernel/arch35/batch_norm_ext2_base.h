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
 * \file batch_norm_ext2_base.h
 * \brief
 */
#ifndef NORM_BATCH_NORM_EXT2_BASE_H
#define NORM_BATCH_NORM_EXT2_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace BatchNormExt2Ops {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::Reduce;
using AscendC::Reg::StoreAlign;

constexpr static int64_t DOUBLE_BUFFER = 2;
constexpr static int32_t BUFFER_DEPTH = 1;
static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
constexpr static uint32_t FLOAT_BYTES = 4;
constexpr static float POS_INF = 3.40282366920938E+38;
constexpr static float zero = 0.0f;
constexpr static int64_t NDDMA_THRESHOLD = 32;
constexpr static int64_t NDDMA_SECOND_DIM = 1;
constexpr static int64_t NDDMA_THIRD_DIM = 2;
constexpr static int64_t NDDMA_DIM_NUM = 3;
constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void LoadTensorForDtypeT(AscendC::MicroAPI::RegTensor<float>& dst, __ubuf__ T* src,
                                           AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign<float, LoadDist::DIST_NORM>(dst, (__ubuf__ float*)src + offset);
    } else { // fp16、bf16
        RegTensor<T> xFp16;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__ubuf__ T*)src + offset));
        Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void LoadTensorForDtypeTBrc(RegTensor<float>& dst, __ubuf__ T* src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dst, (__ubuf__ float*)src + offset);
    } else { // fp16、bf16
        RegTensor<T> xFp16;
        LoadAlign<T, LoadDist::DIST_BRC_B16>(xFp16, ((__ubuf__ T*)src + offset));
        Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void LoadTwoTensorForDtypeT(RegTensor<float>& dst1, RegTensor<float>& dst2, __ubuf__ T* src1,
                                              __ubuf__ T* src2, MaskReg& dst1Preg, MaskReg& dst2Preg,
                                              uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16R;
        RegTensor<half> xFp16Q;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__ubuf__ half*)(src1) + (src1Offset)));
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__ubuf__ half*)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xFp16R;
        RegTensor<bfloat16_t> xFp16Q;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__ubuf__ bfloat16_t*)(src1) + (src1Offset)));
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__ubuf__ bfloat16_t*)(src2) + (src2Offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        LoadAlign(dst1, ((__ubuf__ float*)(src1) + (src1Offset)));
        LoadAlign(dst2, ((__ubuf__ float*)(src2) + (src2Offset)));
    }
}

template <typename T>
__aicore__ inline void LoadTwoTensorForDtypeTBrc(RegTensor<float>& dst1, RegTensor<float>& dst2, __ubuf__ T* src1,
                                                 __ubuf__ T* src2, MaskReg& dst1Preg, MaskReg& dst2Preg,
                                                 uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16Q;
        RegTensor<half> xFp16R;
        LoadAlign<half, LoadDist::DIST_BRC_B16>(xFp16Q, ((__ubuf__ half*)(src1) + (src1Offset)));
        LoadAlign<half, LoadDist::DIST_BRC_B16>(xFp16R, ((__ubuf__ half*)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xFp16Q;
        RegTensor<bfloat16_t> xFp16R;
        LoadAlign<bfloat16_t, LoadDist::DIST_BRC_B16>(xFp16Q, ((__ubuf__ bfloat16_t*)(src1) + (src1Offset)));
        LoadAlign<bfloat16_t, LoadDist::DIST_BRC_B16>(xFp16R, ((__ubuf__ bfloat16_t*)(src2) + (src2Offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dst1, ((__ubuf__ float*)(src1) + (src1Offset)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dst2, ((__ubuf__ float*)(src2) + (src2Offset)));
    }
}

template <typename T>
__aicore__ inline void StoreTensorForDtypeT(__ubuf__ T* dst, AscendC::MicroAPI::RegTensor<float>& src,
                                            AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        AscendC::MicroAPI::RegTensor<T> xFp16;
        Cast<T, float, castTraitB322B16>(xFp16, src, preg);
        StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}

/*
 * 本地副本，派生自 norm/norm_common/op_kernel/reduce_common_regbase_part1.h:164-216
 * (NormCommon 命名空间下的 ComputeRstdNewtonRaphsonReg)。仅重构 rsqrt 种子与 Newton-Raphson 残差两处算式，
 * 语义与边界兜底保持不变。
 *
 * 动机：v = var + eps 落入 (8.507e37, 3.4e38] 时，原实现的中间量 1/v 与 rstd^2 下溢为 denormal，
 * 被 FTZ 清零，导致 rstd 退化为 0、y 精确等于 offset。重构后种子路径为 sqrt(v) 再取倒数，
 * 中间量恒落在 normal 区；残差项改用 (sqrt(v)*rstd)^2 表示，不再出现裸 1/v 与裸 rstd^2。
 * 本算子的训练路径（welford / full_reduce / ra_full_reduce / ra_welford，以及经
 * ComputeRstdNewtonRaphsonExt2 包装调用的 block_split_r / rar_block_split_r）已全部改用本副本。
 */
template <bool NEED_MAX = true>
__aicore__ inline void ComputeRstdNewtonRaphsonRegExt2(RegTensor<float>& var, RegTensor<float>& rstd, MaskReg& preg,
                                                       float epsilon)
{
    static constexpr float POS_INF = 3.40282366920938E+38;
    static constexpr float SCALAR1 = -0.5;
    static constexpr float SCALAR2 = 1.5;
    static constexpr float SCALAR3 = 0.5;
    static constexpr float SCALAR0 = -99.99;

    RegTensor<float> r; // rsqrt 种子（原实现此寄存器装 1/v）
    RegTensor<float> y; // sqrt(v)（原实现此寄存器装 rsqrt 种子）
    RegTensor<float> s;
    RegTensor<float> t;
    RegTensor<float> one;
    RegTensor<float> scalar1;
    RegTensor<float> t1;
    RegTensor<float> t4;
    RegTensor<float> scalarInf;
    RegTensor<float> scalarZero;
    MaskReg cmpRegZero;
    MaskReg cmpRegInf;

    Duplicate(scalarInf, POS_INF, preg);
    Duplicate(scalarZero, float(0.0), preg);
    Duplicate(one, float(1.0), preg);
    Duplicate(scalar1, SCALAR3, preg);
    Duplicate(t1, SCALAR2, preg);

    Adds(var, var, epsilon, preg);
    if constexpr (NEED_MAX) {
        Maxs(var, var, SCALAR0, preg);
    }
    // 改动 1：先开方后倒数，y = sqrt(v) 与 r = 1/sqrt(v) 均恒为 normal
    Sqrt(y, var, preg);
    Div(r, one, y, preg);
    Muls(t, var, SCALAR1, preg);
    Mul(t, t, r, preg);     // 改动 2：原为 y（原语义 -0.5*sqrt(v)）
    Mula(t1, t, r, preg);   // 改动 2：原为 y
    Mul(rstd, r, t1, preg); // 改动 2：原为 y；rstd 为一次 NR 结果
    // 改动 3：残差 s = 1 - v*rstd^2，改用 (sqrt(v)*rstd)^2，消掉裸 1/v 与裸 rstd^2
    Mul(t4, y, rstd, preg); // t4 = sqrt(v)*rstd，恒 normal
    Mul(t4, t4, t4, preg);  // t4 = v*rstd^2，恒 normal
    Muls(t4, t4, float(-1.0), preg);
    Adds(s, t4, float(1.0), preg);
    Mul(s, s, rstd, preg);
    Mula(rstd, s, scalar1, preg); // 二次 NR，与原实现等价
    Compares(cmpRegZero, var, POS_INF, preg);
    Select(rstd, scalarZero, rstd, cmpRegZero);
    Compares(cmpRegInf, var, float(0.0), preg);
    Select(rstd, scalarInf, rstd, cmpRegInf);
}

/*
 * 本地副本（LocalTensor 包装层），派生自 norm/norm_common/op_kernel/reduce_common_regbase_part1.h:308-320
 * (NormCommon 命名空间下的 ComputeRstdNewtonRaphson<NEED_MAX, NEED_AVG_FACTOR>)。
 * 与上游包装逐行对应：按 vectorLen 分块 UpdateMask -> DataCopy 载入 var -> (可选) Muls avgFactor
 * -> 调用已修复的 ComputeRstdNewtonRaphsonRegExt2 -> DataCopy 写回 rstd。
 * 唯一差别是内部转调本地修复版 Reg 实现，用于 block_split_r / rar_block_split_r 两条训练路径。
 */
template <bool NEED_MAX = true, bool NEED_AVG_FACTOR = false>
__aicore__ inline void ComputeRstdNewtonRaphsonExt2(__local_mem__ float* src, __local_mem__ float* dst,
                                                    uint32_t rowCount, float epsilon, float avgFactor = 1.0f,
                                                    uint32_t vectorLen = VL_FP32)
{
    uint16_t loopRows = static_cast<uint16_t>((rowCount + vectorLen - 1) / vectorLen);
    __VEC_SCOPE__
    {
        RegTensor<float> var;
        RegTensor<float> rstd;
        MaskReg pregLoop;

        uint32_t sreg = rowCount;
        for (uint16_t i = 0; i < loopRows; ++i) {
            pregLoop = UpdateMask<float>(sreg);
            AscendC::Reg::DataCopy(var, src + i * vectorLen);
            if constexpr (NEED_AVG_FACTOR) {
                Muls(var, var, avgFactor, pregLoop);
            }
            ComputeRstdNewtonRaphsonRegExt2<NEED_MAX>(var, rstd, pregLoop, epsilon);
            AscendC::Reg::DataCopy(dst + i * vectorLen, rstd, pregLoop);
        }
    }
}

template <bool NEED_MAX = true, bool NEED_AVG_FACTOR = false>
__aicore__ inline void ComputeRstdNewtonRaphsonExt2(LocalTensor<float> srcLocal, LocalTensor<float> dstLocal,
                                                    uint32_t rowCount, float epsilon, float avgFactor = 1.0f,
                                                    uint32_t vectorLen = VL_FP32)
{
    __local_mem__ float* src = (__local_mem__ float*)srcLocal.GetPhyAddr();
    __local_mem__ float* dst = (__local_mem__ float*)dstLocal.GetPhyAddr();
    ComputeRstdNewtonRaphsonExt2<NEED_MAX, NEED_AVG_FACTOR>(src, dst, rowCount, epsilon, avgFactor, vectorLen);
}

} // namespace BatchNormExt2Ops

#endif // NORM_BATCH_NORM_EXT2_BASE_H
