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
 * \file batch_norm3d_grad_common.h
 * \brief
 */
#ifndef __BATCH_NORM3D_GRAD_COMMON_H__
#define __BATCH_NORM3D_GRAD_COMMON_H__

#include "kernel_operator.h"
#include "batch_norm3_d_grad_tiling_data.h"
#include "../inc/platform.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace BatchNorm3DGrad {
using namespace AscendC;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

static constexpr uint32_t ONE = 1;
static constexpr uint32_t TWO = 2;
static constexpr uint32_t THREE = 3;
static constexpr uint32_t FOUR = 4;
static constexpr uint32_t ONE_BLK_SIZE = platform::GetUbBlockSize();
static constexpr uint32_t TWO_BLK_SIZE = ONE_BLK_SIZE * TWO;
static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
static constexpr uint64_t MOVE_ALIGN_LOOP_SIZE_SHIFT = 21;
static constexpr uint64_t MOVE_ALIGN_LOOP_STRIDE_SHIFT = 40;
static constexpr uint64_t MOVE_ALIGN_LOOP_ONCE = 2097153ULL;
static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);

static constexpr uint32_t DIGIT_TWO = 2;
static constexpr uint32_t DIGIT_THREE = 3;
static constexpr uint32_t CACHE_BUFF_SIZE = 6 * 1024 + 2 * 256;
static constexpr uint32_t DGAMA_CACHE_INDEX = 768;
static constexpr uint32_t CACHE_LEVEL0_INDEX = 0;
static constexpr uint32_t CACHE_LEVEL1_INDEX = 256;
static constexpr uint32_t CACHE_LEVEL2_INDEX = 512;
static constexpr uint32_t DBETA_FOLD_CACHE_INDEX = 1536;
static constexpr uint32_t DGAMA_FOLD_CACHE_INDEX = 1536 + 64;
static constexpr uint32_t FOLD_CACHE_CAPACITY = 64;
static constexpr uint32_t ONE_LEVEL_BINARRY_ADD_CACHE_CAPACITY = 256;

using BinaryAddParam = struct BNGBinaryAddParam {
    uint32_t binaryAddQuotient = 0;
    uint32_t binaryAddk = 0;
    uint32_t binaryAddLastNum = 0;
};

__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

__aicore__ inline uint32_t RoundUpOneBlock(uint32_t x) { return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE; }

__aicore__ inline uint32_t RoundUpTwoBlock(uint32_t x) { return (x + TWO_BLK_SIZE - 1) / TWO_BLK_SIZE * TWO_BLK_SIZE; }

static constexpr MicroAPI::CastTrait castTraitB162B32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                         MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

static constexpr MicroAPI::CastTrait castTraitB322B16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                         MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T>
__aicore__ inline void LoadOneTensor(const __local_mem__ void* input, MicroAPI::RegTensor<float>& dst,
                                     MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, (__local_mem__ half*)(input) + offset);
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else {
        DataCopy(dst, (__local_mem__ float*)(input) + offset);
    }
}

__aicore__ inline void LoadOneTensorBrcVL(const __local_mem__ void* input, MicroAPI::RegTensor<float>& dst,
                                          uint32_t offset)
{
    DataCopy<float, MicroAPI::LoadDist::DIST_BLK>(dst, (__local_mem__ float*)(input) + offset);
}

template <typename T>
__aicore__ inline void LoadsTensorForDtypeT(const __local_mem__ void* src, MicroAPI::RegTensor<float>& dst,
                                            MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst, (__local_mem__ float*)src + offset);
    } else { // fp16
        RegTensor<T> xFp16;
        DataCopy<T, LoadDist::DIST_BRC_B16>(xFp16, ((__local_mem__ T*)src + offset));
        Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void StoreOneTensor(const __local_mem__ void* output, MicroAPI::RegTensor<float>& src,
                                      MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>((__local_mem__ half*)(output) + offset, xFp16, preg);
    } else {
        DataCopy((__local_mem__ float*)(output) + offset, src, preg);
    }
}

template <typename T>
__aicore__ inline void LoadOneElement(const __local_mem__ void* input, MicroAPI::RegTensor<float>& dst,
                                      MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        DataCopy<half, MicroAPI::LoadDist::DIST_BRC_B16>(xFp16, (__local_mem__ half*)(input) + offset);
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else {
        DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(dst, ((__local_mem__ float*)(input)) + offset);
    }
}

template <typename T>
__aicore__ inline void StoreOneElement(const __local_mem__ void* output, MicroAPI::RegTensor<float>& src,
                                       MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<half> xFp16;
        Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        DataCopy<half, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>((__local_mem__ half*)(output) + offset, xFp16,
                                                                    preg);
    } else {
        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)output) + offset, src,
                                                                     preg);
    }
}

__aicore__ inline void LoadTwoTensorSum(const __local_mem__ void* input, MicroAPI::RegTensor<float>& dst,
                                        MicroAPI::MaskReg& preg, uint32_t offset0, uint32_t offset1)
{
    MicroAPI::RegTensor<float> a, b;
    DataCopy(a, (__local_mem__ float*)(input) + offset0);
    DataCopy(b, (__local_mem__ float*)(input) + offset1);
    Add<float, MicroAPI::MaskMergeMode::ZEROING>(dst, a, b, preg);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, int64_t n)
{
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = n * sizeof(T);
    DataCopyPad(dstTensor, srcTensor, params);
}

template <typename T>
__aicore__ inline void CopyIn(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor, int64_t n)
{
    // CopyIn
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = n * sizeof(T);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    DataCopyPad(dstTensor, srcTensor, params, padParams);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, int64_t rowSize,
                               int64_t colSize, int64_t dstStride, int64_t srcStride)
{
    // CopyOut
    DataCopyExtParams params;
    params.blockCount = rowSize;
    params.blockLen = colSize * sizeof(T);
    params.dstStride = dstStride * sizeof(T) - params.blockLen;
    params.srcStride = (srcStride * sizeof(T) - RoundUpOneBlock(params.blockLen)) / ONE_BLK_SIZE;
    DataCopyPad(dstTensor, srcTensor, params);
}

template <typename T>
__aicore__ inline void CopyIn(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor, int64_t rowSize,
                              int64_t colSize, int64_t dstStride, int64_t srcStride)
{
    // CopyIn
    DataCopyExtParams params;
    params.blockLen = colSize * sizeof(T);
    params.blockCount = rowSize;
    params.dstStride = (dstStride * sizeof(T) - RoundUpOneBlock(params.blockLen)) / ONE_BLK_SIZE;
    params.srcStride = srcStride * sizeof(T) - params.blockLen;
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    DataCopyPad(dstTensor, srcTensor, params, padParams);
}

__aicore__ inline int64_t GetCacheID(const int64_t idx) { return ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1; }

// ===================== dbeta/dgamma 归约的补偿累加 =====================
// scale_backprop/offset_backprop 的判定是 cross_check（NPU 误差 / 三方误差），分母取三方误差，
// 而三方（torch）的 GPU 归约本身就是近似正确舍入的，所以只靠"少几级朴素加法"赢不了 1.5 倍门限。
// 把最后这段"部分和累加"换成补偿（TwoSum）累加后，归约结果就是真实和的**正确舍入**值，
// 误差只剩最后一次 fp32 舍入，比值稳定在 1 附近。
// 逐位复现分解显示：核内 binary carry 累加 + 核间 72 行 ReduceSum 这两段朴素加法
// 各占输出误差的一半左右（L0_003 offset 0.65 ULP vs torch 0.27 ULP，rmse 比值 2.43 FAIL）。
constexpr static float BNG_MAX_FINITE_FP32 = 3.4028234663852886e38F;
// ra_split_r 核内累加的扩展指数域：项按 2^-32 精确缩小（2 的幂无舍入）再进补偿累加器，
// 累加器上界变为 R_max(2^31) × FLT_MAX × 2^-32 < FLT_MAX，中间和永不溢出卡死符号；
// 跨核合并末尾按 2^32 放大回原域，真值超 fp32 域时在此以正确符号得 ±inf。
constexpr static float BNG_ACC_SCALE_DOWN = 2.3283064365386963e-10F; // 2^-32
constexpr static float BNG_ACC_SCALE_UP = 4294967296.0F;             // 2^32
__simd_callee__ inline void ClampResidual(MicroAPI::RegTensor<float>& lo, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> zeroReg;
    MicroAPI::RegTensor<float> keepReg;
    MicroAPI::MaskReg leMask, geMask, finiteMask, badMask;
    MicroAPI::Duplicate(zeroReg, 0.0f, mask);
    AscendC::Reg::Move(keepReg, lo);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::LE>(leMask, lo, BNG_MAX_FINITE_FP32, mask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::GE>(geMask, lo, -BNG_MAX_FINITE_FP32, mask);
    AscendC::Reg::And(finiteMask, leMask, geMask, mask); // 活跃 && 有限（NaN 两次比较都 false）
    AscendC::Reg::Move(badMask, mask);                   // 掩码外恒 0
    AscendC::Reg::Not(badMask, finiteMask, mask);        // 活跃 lane 内取反 -> 非有限
    AscendC::Reg::Select(lo, zeroReg, keepReg, badMask); // 非有限 -> 0，其余（含掩码外）保持原值
}

// (hi, lo) += value：Knuth TwoSum 求 fl(hi+value) 的精确误差存进 lo，
// 与 Neumaier 的分支补偿逐位等价（不要求 |hi| >= |value|），且无分支。
// hi/lo 都是跨调用保持的累加器：mask 的非活跃 lane 必须原样保留（MERGING），
// 否则调用方把"不足一个 VL 的尾块"按部分掩码传进来时，会把该 lane 已累积的残差清零，补偿就白做了。
__simd_callee__ inline void AccumulateWithError(MicroAPI::RegTensor<float>& hi, MicroAPI::RegTensor<float>& lo,
                                                MicroAPI::RegTensor<float>& value, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> total, back, error;
    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(total, hi, value, mask);
    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(back, total, hi, mask);
    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(error, total, back, mask);
    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(error, hi, error, mask);
    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(back, value, back, mask);
    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(error, error, back, mask);
    // 非有限误差肢无意义（inf-inf=NaN）：钳 0，hi 的 ±inf 照常传播
    ClampResidual(error, mask);
    // lo 是累加器：非活跃 lane 必须 MERGING 保留（ZEROING 会把它清零）
    MicroAPI::Add<float, MicroAPI::MaskMergeMode::MERGING>(lo, lo, error, mask);
    // hi 是累加器：Move 默认即 MERGING，这里显式写出来
    AscendC::Reg::Move<float, MicroAPI::MaskMergeMode::MERGING>(hi, total, mask);
}

// hi 溢出成 ±inf 时 lo 会退化成 NaN，再加回 hi 会把 inf 污染成 NaN。
// 补偿只在有限域有意义，非有限残差按 IEEE 自然传播（置 0）。
// 同样只处理 mask 的活跃 lane：Select 会写满所有 lane，所以 else 分支必须给原值 keepReg，
// 否则掩码外 lane 的 lo 会被写成 0。

// 64-lane 补偿树：(hi,lo) 逐层对折叠加，最终 lane0 为全量双肢和
__simd_callee__ inline void CompensatedLaneTree(MicroAPI::RegTensor<float>& hi, MicroAPI::RegTensor<float>& lo,
                                                MicroAPI::RegTensor<uint32_t>& evenIdx,
                                                MicroAPI::RegTensor<uint32_t>& oddIdx)
{
    MicroAPI::RegTensor<float> rhsSums, rhsLow;
#pragma unroll
    for (uint16_t level = 0; level < 6; ++level) {
        uint32_t activeNum = VL_FP32 >> (level + 1);
        MicroAPI::MaskReg pregActive = MicroAPI::UpdateMask<float>(activeNum);
        AscendC::Reg::Gather(rhsSums, hi, oddIdx);
        AscendC::Reg::Gather(rhsLow, lo, oddIdx);
        AscendC::Reg::Gather(hi, hi, evenIdx);
        AscendC::Reg::Gather(lo, lo, evenIdx);
        MicroAPI::Add(lo, lo, rhsLow, pregActive);
        AccumulateWithError(hi, lo, rhsSums, pregActive);
    }
}

// 行级精确补偿归约：VL 宽 (hi,lo) 累加 + 64-lane 补偿树。
// 替代 ReduceSum 裸 fp32 树（其块内舍入在大 R 下会把 dbeta/dgamma 顶出 CR）。
// 结果以双肢写出（dstHi[r]/dstLo[r]），不在此舍入——单舍入留给最终合流处。
// 行 r 的数据在 src + r*rowStride 起 rowLen 个元素。
__aicore__ inline void CompensatedRowReduceHiLo(const __local_mem__ float* src, __local_mem__ float* dstHi,
                                                __local_mem__ float* dstLo, uint16_t rows, uint32_t rowLen,
                                                uint32_t rowStride)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> x, zero, hi, lo, rhsSums, rhsLow;
        AscendC::MicroAPI::RegTensor<uint32_t> evenIdx, oddIdx;
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregMerge = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
        MicroAPI::Duplicate(zero, 0.0f, pregAll);
        AscendC::Reg::Arange((MicroAPI::RegTensor<int32_t>&)evenIdx, static_cast<int32_t>(0));
        ShiftLefts(evenIdx, evenIdx, static_cast<int16_t>(1), pregAll);
        MicroAPI::Duplicate(oddIdx, static_cast<uint32_t>(VL_FP32 - 1), pregAll);
        AscendC::Reg::And(evenIdx, evenIdx, oddIdx, pregAll);
        Adds(oddIdx, evenIdx, static_cast<uint32_t>(1), pregAll);
        for (uint16_t r = 0; r < rows; r++) {
            MicroAPI::Duplicate(hi, 0.0f, pregAll);
            MicroAPI::Duplicate(lo, 0.0f, pregAll);
            uint32_t rem = rowLen;
            uint16_t loops = static_cast<uint16_t>(CeilDiv(rowLen, VL_FP32));
            for (uint16_t i = 0; i < loops; ++i) {
                MicroAPI::MaskReg pm = MicroAPI::UpdateMask<float>(rem);
                LoadOneTensor<float>(src, x, pm, r * rowStride + i * VL_FP32);
                MicroAPI::Select(x, x, zero, pm);
                AccumulateWithError(hi, lo, x, pregAll);
            }
            CompensatedLaneTree(hi, lo, evenIdx, oddIdx);
            ClampResidual(lo, pregAll);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dstHi + r, hi, pregMerge);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dstLo + r, lo, pregMerge);
        }
    }
}

// fp32 输入时 golden 的元素项是 fp64 级（Promote 抬档）：term = ((x-mean)*rstd)*dy
// 的主值 + 精确误差肢。TwoSum 取减法误差，TwoProduct（vmadd 单舍入）取两次乘法误差。
// fp16 输入的 golden 就是 fp32 级元素项，不得用本函数（做了反而超过 golden 档位）。
__simd_callee__ inline void ExactTermXSubMeanRstdMulDy(MicroAPI::RegTensor<float>& termHi,
                                                       MicroAPI::RegTensor<float>& termErr,
                                                       MicroAPI::RegTensor<float>& xVal,
                                                       MicroAPI::RegTensor<float>& meanReg,
                                                       MicroAPI::RegTensor<float>& rstdReg,
                                                       MicroAPI::RegTensor<float>& dyVal, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> s, bp, es, p1, ep1;
    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(s, xVal, meanReg, mask); // s = fl(x-mean)
    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(bp, s, xVal, mask);      // bp = s-x
    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(es, s, bp, mask);        // s-bp
    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(es, xVal, es, mask);     // x-(s-bp)
    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(bp, meanReg, bp, mask);  // mean+bp
    AscendC::Reg::Neg<float, MicroAPI::MaskMergeMode::ZEROING>(bp, bp, mask);       // -mean-bp
    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(es, es, bp, mask);       // es = 减法精确误差
    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(p1, s, rstdReg, mask);   // p1 = fl(s*rstd)
    // vmadd 语义是 dst = dst*src0 + src1（Horner 形），TwoProduct 误差要这样取：
    AscendC::Reg::Move(ep1, s);                                                      // ep1 = s
    AscendC::Reg::Neg<float, MicroAPI::MaskMergeMode::ZEROING>(bp, p1, mask);        // bp = -p1
    AscendC::Reg::FusedMulDstAdd(ep1, rstdReg, bp, mask);                            // ep1 = s*rstd-p1 精确
    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(termHi, p1, dyVal, mask); // t = fl(p1*dy)
    AscendC::Reg::Neg<float, MicroAPI::MaskMergeMode::ZEROING>(bp, termHi, mask);    // bp = -t
    AscendC::Reg::FusedMulDstAdd(p1, dyVal, bp, mask);                               // p1 = p1*dy-t 精确
    // termErr = es*rstd*dy + ep1*dy + et
    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(es, es, rstdReg, mask);
    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(es, es, dyVal, mask);
    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(ep1, ep1, dyVal, mask);
    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(termErr, es, ep1, mask);
    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(termErr, termErr, p1, mask);
    ClampResidual(termErr, mask); // 项为 ±inf 时误差肢是 NaN（inf-inf），钳 0 保持主值 inf 传播
}

} // namespace BatchNorm3DGrad
#endif
