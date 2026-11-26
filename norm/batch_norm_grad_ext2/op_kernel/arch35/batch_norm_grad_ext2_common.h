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
 * \file batch_norm_grad_ext2_common.h
 * \brief
 */
#ifndef __BATCH_NORM_GRAD_EXT2_COMMON_H__
#define __BATCH_NORM_GRAD_EXT2_COMMON_H__

#include "kernel_operator.h"
#include "batch_norm_grad_ext2_tiling_data.h"
#include "../inc/platform.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace BatchNormGradExt2 {
using namespace AscendC;

// ---------------------------------------------------------------------------------------------
// 转置搬运（NDDMA loop mode）：GM 上 [rowSize][srcRowStride] 的子块 -> UB 上 [colSize][dstStride]，
// 即   dst[c * dstStride + r] == src[r * srcRowStride + c]。
// 900000（infer channel last）与 910001/910002（infer）两族的 dgamma/dbeta 都用它把
// "通道在内、R 在外"的 tile 换成 "lane = 通道（或 R）" 的连续布局。
// 参数用法抄自 norm/layer_norm_grad/op_kernel/arch35/layer_norm_grad_base.h:1025-1040 的
// CopyInTranspose<T>(dstTensor, srcTensor, rowSize, colSize, dstStride)：
//   loop0（内层，源侧连续的一段）: loopSize=colSize, loopSrcStride=1,        loopDstStride=dstStride
//   loop1（外层，逐行）          : loopSize=rowSize, loopSrcStride=<行跨距>, loopDstStride=1
// 与 layer_norm 的唯一差别：那里的 tile 覆盖整行，源行跨距恰好等于 colSize；本算子按 A/R 切片，
// 所以把行跨距显式做成形参（其余逐字段一致）。源/目标跨距都是元素单位。
// ---------------------------------------------------------------------------------------------
template <typename T>
__aicore__ inline void CopyInTransposeTile(const LocalTensor<T>& dstTensor, const GlobalTensor<T>& srcTensor,
                                           int64_t rowSize, int64_t colSize, int64_t srcRowStride, int64_t dstStride)
{
    static constexpr uint32_t CONST_TWO_DIM = 2;
    static constexpr NdDmaConfig config = {false};
    NdDmaLoopInfo<CONST_TWO_DIM> copyLoopInfo;
    copyLoopInfo.loopSrcStride[0] = 1;
    copyLoopInfo.loopSrcStride[1] = static_cast<uint32_t>(srcRowStride);
    copyLoopInfo.loopDstStride[0] = static_cast<uint32_t>(dstStride);
    copyLoopInfo.loopDstStride[1] = 1;
    copyLoopInfo.loopSize[0] = static_cast<uint32_t>(colSize);
    copyLoopInfo.loopSize[1] = static_cast<uint32_t>(rowSize);
    NdDmaParams<T, CONST_TWO_DIM> params = {copyLoopInfo, 0};
    DataCopy<T, CONST_TWO_DIM, config>(dstTensor, srcTensor, params);
}

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
static constexpr uint32_t DIGIT_FOUR = 4;
static constexpr uint32_t DIGIT_FIVE = 5;
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

// ---------------------------------------------------------------------------------------------
// 单次舍入输出（dx 的 fp16 输出用）：
//
//   golden = fl16(在 fp64 里算完的精确值)——"只舍入一次"。任何 fp32 计算链都是先舍到 fp32 再
//   舍到 fp16（两次舍入），只有当 fp32 中间值**恰好落在 fp16 网格中点上**时两者才会差 1 个
//   fp16 ULP（实测 L1_469：1660/30985416 个元素，且 1660 个全部满足"低 13 位 == 0x1000"）。
//
//   base  = fl32(exact)   （调用方用两次 Mul 得到的当前结果）
//   resid = exact - base  （调用方用 FMA 残差算出：p=fl(dy*gamma)、e1=dy*gamma-p、
//                          c=fl(p*rstd)、resid=(p*rstd-c)+e1*rstd）
//
//   这里在中点 lane 上用 resid 的符号决定舍入方向（而不是 round-half-even），非中点 lane 保持
//   Cast 的逐位结果不变。中点判定用整数位段：fp16 只保留 fp32 尾数的高 10 位，故 base 恰为
//   中点 ⟺ 尾数低 13 位 == 0x1000。方向：resid > 0（精确值大于 base）按数值向上调，
//   否则向下调；调节量取 |base|*2^-12 ∈ [ULP16/4, ULP16/2)，必然改变 fp32 值又不会跨到
//   下一个 rung。subnormal 区（|base| < 2^-14）调节量 ≤ 2^-26 小于 subnormal ULP，天然不变。
//   对 fp32 输出（T = float）直接写 base——base 本身就是对 exact 的一次舍入。
// ---------------------------------------------------------------------------------------------
template <typename T>
__aicore__ inline void StoreDxSingleRounded(const __local_mem__ void* output, MicroAPI::RegTensor<float>& base,
                                            MicroAPI::RegTensor<float>& resid, MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<int32_t> mask13, midPattern, lowBits;
        MicroAPI::RegTensor<float> absC, delta, negDelta, signedDelta, cand;
        MicroAPI::RegTensor<half> outFp16;
        MaskReg isMid, residPos, residNonZero;
        MicroAPI::RegTensor<int32_t>& bits = reinterpret_cast<MicroAPI::RegTensor<int32_t>&>(base);
        MicroAPI::Duplicate(mask13, static_cast<int32_t>(0x1FFF), preg);
        MicroAPI::Duplicate(midPattern, static_cast<int32_t>(0x1000), preg);
        MicroAPI::And<int32_t, MicroAPI::MaskMergeMode::ZEROING>(lowBits, bits, mask13, preg);
        MicroAPI::Compare<int32_t, AscendC::CMPMODE::EQ>(isMid, lowBits, midPattern, preg);
        MicroAPI::Abs(absC, base, preg);
        MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(delta, absC, 0.000244140625f, preg);
        MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(negDelta, delta, -1.0f, preg);
        MicroAPI::CompareScalar<float, AscendC::CMPMODE::GT>(residPos, resid, 0.0f, preg);
        MicroAPI::Select(signedDelta, delta, negDelta, residPos);
        MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(cand, base, signedDelta, preg);
        // resid == 0（精确值恰好就是中点）必须保持 round-half-even：golden 也是把中点舍到偶数。
        // L1_088 有 6071 个元素属于"精确值 == 中点"，无此守卫时会被统一往下拨 1 ULP（npu 8696 vs
        // golden 8704）。
        MicroAPI::CompareScalar<float, AscendC::CMPMODE::NE>(residNonZero, resid, 0.0f, preg);
        MicroAPI::MaskAnd(isMid, isMid, residNonZero, preg);
        MicroAPI::Select(signedDelta, cand, base, isMid);
        Cast<half, float, castTraitB322B16>(outFp16, signedDelta, preg);
        DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>((__local_mem__ half*)(output) + offset, outFp16, preg);
    } else {
        StoreOneTensor<T>(output, base, preg, offset);
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

} // namespace BatchNormGradExt2
#endif
