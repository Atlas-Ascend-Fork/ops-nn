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
 * \file pp_matmul_common_tiling.h
 * \brief
 */

#ifndef __OP_HOST_PPMAT_MUL_COMMMON_TILING_H__
#define __OP_HOST_PPMAT_MUL_COMMMON_TILING_H__

#include "pp_matmul_info.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace pp_matmul {

constexpr uint64_t FP16_SIZE = 2;
constexpr uint64_t FP32_SIZE = 4;
constexpr uint64_t BLOCK_SIZE = 16;
constexpr uint64_t BLOCK_SIZE_INT8_K = 32;
constexpr uint64_t BASE_BLOCK_STEP = 2;
constexpr uint64_t AXES_ALIGN_SIZE = 512;
constexpr uint64_t AXES_ALIGN_SIZE_INT8 = 256;
constexpr uint64_t ND_SHAPE_SIZE = 2;
constexpr uint64_t NZ_SHAPE_SIZE = 4;
constexpr uint64_t CUBE_BLOCK_SIZE = 256;
constexpr uint64_t CUBE_BLOCK_SIZE_INT8 = 512;
constexpr uint64_t L1AB_PINGPONG_BUFFER_SIZE = 262144;
constexpr uint64_t L0AB_PINGPONG_BUFFER_SIZE_INT8 = 262144;        // 131072 * 2 = 256 KB
constexpr uint64_t L0AB_PINGPONG_BUFFER_SIZE_FP16 = 131072;        // 128 KB
constexpr uint64_t L1AB_PINGPONG_BUFFER_SIZE_INT8_SPARSE = 163840; // 160 * 1024
constexpr uint64_t UB_LIMIT_SIZE_910A = 131072;                    // 128 * 1024
constexpr uint64_t UB_LIMIT_SIZE_PERTOKEN_ARCH20 = 131072;         // 128 * 1024

template <uint64_t DIV>
inline uint64_t CeilDiv(uint64_t num)
{
    if (DIV == 0UL || num + DIV - 1UL < num) {
        return num;
    }
    return (num + DIV - 1UL) / DIV;
}

inline uint64_t CeilDiv(uint64_t dividend, uint64_t divisor)
{
    if (divisor == 0UL || dividend + divisor - 1UL < dividend) {
        return dividend;
    }
    return (dividend + divisor - 1UL) / divisor;
}

template <uint64_t RND>
inline uint64_t Round(uint64_t num)
{
    if (RND == 0UL || num + RND - 1UL < num) {
        return num;
    }
    return (num + RND - 1UL) / RND * RND;
}

inline uint64_t RoundUp(uint64_t num, uint64_t rnd)
{
    if (rnd == 0UL || num + rnd - 1UL < num) {
        return num;
    }
    return (num + rnd - 1UL) / rnd * rnd;
}

inline int64_t RoundUp(int64_t num, int64_t rnd)
{
    if (rnd == 0UL || static_cast<int64_t>(num + rnd - 1) < num) {
        return num;
    }
    return static_cast<int64_t>((num + rnd - 1) / rnd * rnd);
}

inline uint64_t RoundDown(uint64_t num, uint64_t rnd)
{
    if (rnd == 0UL) {
        return 0UL;
    }
    return num / rnd * rnd;
}

inline uint64_t GetN0TilingLimit(bool compressFlag, uint64_t tilingN, const platform_ascendc::SocVersion& platformType)
{
    if (compressFlag) {
        return std::min(tilingN * BLOCK_SIZE, AXES_ALIGN_SIZE_INT8);
    } else {
        return (platformType == platform_ascendc::SocVersion::ASCEND310P ||
                platformType == platform_ascendc::SocVersion::ASCEND910) ?
                   AXES_ALIGN_SIZE :
                   AXES_ALIGN_SIZE_INT8;
    }
}

template <typename OpShareType>
inline uint64_t GetN0TilingInit(const OpShareType& opShape, bool compressFlag, uint64_t tilingN)
{
    const uint64_t rnd = 16UL;
    if (compressFlag) {
        if (tilingN * BLOCK_SIZE > opShape.n) {
            return Round<rnd>(opShape.n);
        } else {
            return tilingN * BLOCK_SIZE;
        }
    } else {
        return BLOCK_SIZE;
    }
}

template <bool IS_PRIMARY_M_AXIS>
inline bool IsExceedTilingLimit(uint64_t secondaryBaseSize, uint64_t primaryBaseSize, uint64_t n0TilingLimit,
                                platform_ascendc::SocVersion platformType, uint64_t basicBlockSize,
                                const bool isPertokenArch20)
{
    return (IS_PRIMARY_M_AXIS && secondaryBaseSize > n0TilingLimit) ||
           (!IS_PRIMARY_M_AXIS && primaryBaseSize > n0TilingLimit) ||
           ((platformType == platform_ascendc::SocVersion::ASCEND910 && basicBlockSize > UB_LIMIT_SIZE_910A) ||
            (platformType == platform_ascendc::SocVersion::ASCEND310P && isPertokenArch20 == true &&
             basicBlockSize > UB_LIMIT_SIZE_PERTOKEN_ARCH20));
}

template <bool IS_PRIMARY_M_AXIS, typename OpShareType>
inline void SetOpShapeAxesInfo(OpShareType& opShape, uint64_t primaryBaseSize, uint64_t secondaryBaseSize)
{
    opShape.m0 = IS_PRIMARY_M_AXIS ? primaryBaseSize : secondaryBaseSize;
    opShape.n0 = IS_PRIMARY_M_AXIS ? secondaryBaseSize : primaryBaseSize;
}

template <typename HardwareType, typename OpShapeType, typename MatMulInfoType>
inline float CostFunc(const HardwareType& hwInfo, const OpShapeType& shape, const MatMulInfoType& mmInfo)
{
    float aCoef = 1;
    float bCoef = 1;
    float bwCoef = static_cast<float>(hwInfo.l2BandWidth) / static_cast<float>(hwInfo.hbmBandWidth);
    uint64_t mLoop = CeilDiv(shape.m, shape.m0);
    uint64_t nLoop = CeilDiv(shape.n, shape.n0);
    if (mLoop == 0UL || nLoop == 0UL) {
        return 1;
    }
    uint64_t coreNeed = shape.batchSize * mLoop * nLoop;
    uint64_t blockDim = std::min(coreNeed, hwInfo.coreNum);
    uint64_t mOnce = blockDim < nLoop ? shape.m0 : blockDim / nLoop * shape.m0;
    uint64_t nOnce = blockDim < nLoop ? hwInfo.coreNum * shape.n0 : shape.n;
    if (mOnce * shape.k * mmInfo.sizeInDtype > hwInfo.l2Size) {
        aCoef = bwCoef;
    }
    if (nOnce * shape.k * mmInfo.sizeInDtype > hwInfo.l2Size) {
        bCoef = bwCoef;
    }
    return 1 / (aCoef * static_cast<float>(shape.n0)) + 1 / (bCoef * static_cast<float>(shape.m0));
}

// OpShareType is OpShape, TilingType is PpTilingData, HardwareType is HardwareType, MatMulInfoType is MatMulInfo
template <bool IS_PRIMARY_M_AXIS, typename OpShareType, typename TilingType, typename HardwareType,
          typename MatMulInfoType>
void TilingFunc(OpShareType& opShape, TilingType& tilingParam, const HardwareType& hwInfo, const MatMulInfoType& mmInfo,
                bool compressFlag = false, const uint64_t tilingN = 1)
{
    float costMin = 1;
    const float const2 = 2.0;
    const uint64_t const16 = 16UL;
    uint64_t axisCnt = CeilDiv(IS_PRIMARY_M_AXIS ? opShape.n : opShape.m, const16);
    // The previous float expression pow(2, ceil(log2(axisCnt))) is replaced by this exact integer
    // round-up: both return the smallest power of two that is >= axisCnt for every axisCnt >= 1
    // (checked against the float formula over 1..65536), and the integer form cannot drift with the
    // rounding of log2 for large values. axisCnt == 0 is clamped to one block explicitly because a
    // zero roundBase would turn the RoundUp below into a division by zero.
    if (axisCnt == 0UL) {
        axisCnt = 1UL;
    }
    uint64_t pow2 = 1UL;
    while (pow2 < axisCnt) {
        pow2 <<= 1UL;
    }
    uint64_t roundBase = pow2 * const16;
    uint64_t primaryAxis = RoundUp(IS_PRIMARY_M_AXIS ? opShape.m : opShape.n, const16);
    uint64_t secondaryAxis = RoundUp(IS_PRIMARY_M_AXIS ? opShape.n : opShape.m, roundBase);
    float axes0Max = static_cast<float>(AXES_ALIGN_SIZE) / mmInfo.sizeInDtype;
    auto platformType = hwInfo.socVersion;
    if (mmInfo.isInt8 && (platformType == platform_ascendc::SocVersion::ASCEND310P ||
                          platformType == platform_ascendc::SocVersion::ASCEND910)) {
        axes0Max /= const2;
    }
    uint64_t n0TilingInit = GetN0TilingInit(opShape, compressFlag, tilingN);
    uint64_t n0TilingLimit = GetN0TilingLimit(compressFlag, tilingN, platformType);
    uint64_t primaryBaseSizeInit = IS_PRIMARY_M_AXIS ? BLOCK_SIZE : n0TilingInit;
    uint64_t secondaryBaseSizeInit = IS_PRIMARY_M_AXIS ? n0TilingInit : BLOCK_SIZE;
    bool isNpuArch2002 = platformType == platform_ascendc::SocVersion::ASCEND310P;
    for (uint64_t primaryBaseSize = primaryBaseSizeInit; primaryBaseSize <= primaryAxis && primaryBaseSize <= axes0Max;
         primaryBaseSize *= BASE_BLOCK_STEP) {
        for (uint64_t secondaryBaseSize = secondaryBaseSizeInit;
             secondaryBaseSize <= secondaryAxis && secondaryBaseSize <= axes0Max;
             secondaryBaseSize *= BASE_BLOCK_STEP) {
            uint64_t basicBlockSize = primaryBaseSize * secondaryBaseSize * FP32_SIZE;
            if (basicBlockSize > hwInfo.l0cSize) {
                continue;
            }
            if ((mmInfo.isInt8 || mmInfo.isCompress) &&
                IsExceedTilingLimit<IS_PRIMARY_M_AXIS>(secondaryBaseSize, primaryBaseSize, n0TilingLimit, platformType,
                                                       basicBlockSize, mmInfo.isPertokenArch20)) {
                continue;
            }
            SetOpShapeAxesInfo<IS_PRIMARY_M_AXIS>(opShape, primaryBaseSize, secondaryBaseSize);
            float cost = CostFunc<HardwareType, OpShareType, MatMulInfoType>(hwInfo, opShape, mmInfo);
            if (cost >= costMin) {
                continue;
            }
            costMin = cost;
            tilingParam.SetBaseOp(hwInfo.coreNum, hwInfo.l0cSize, opShape.m0, opShape.n0, mmInfo, isNpuArch2002);
        }
    }
}

template <typename PpTilingDataType>
uint64_t Swizzle(PpTilingDataType& tilingData)
{
    uint64_t swizzleDirect = 0UL;
    uint64_t swizzleCount = 1UL;
    float m0 = tilingData.opShape.m0;
    float n0 = tilingData.opShape.n0;
    float m = tilingData.opShape.m;
    float k = tilingData.opShape.k;
    float n = tilingData.opShape.n;
    float mincost = m * k + k * n;

    for (uint32_t i = 1; i <= tilingData.blockDim; ++i) {
        int c = static_cast<int32_t>((tilingData.blockDim + i - 1) / i);
        float cost;
        // B0 + A < A0 + B
        if (i * n0 + m < m0 * c + n) {
            swizzleDirect = 1UL; // Nz
            cost = n0 * i + m0 * c;
            if (cost <= mincost) {
                mincost = cost;
                swizzleCount = i;
            }
        } else {
            swizzleDirect = 0UL; // Zn
            cost = m0 * i + n0 * c;
            if (cost < mincost) {
                mincost = cost;
                swizzleCount = i;
            }
        }
    }
    tilingData.swizzleDirect = swizzleDirect;
    tilingData.swizzleCount = swizzleCount;
    return swizzleDirect;
}
} // namespace pp_matmul
} // namespace optiling
#endif
