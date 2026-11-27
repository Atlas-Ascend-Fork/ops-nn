/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>
#include "../op_kernel/arch35/rotate_dual_level_quant_tiling_data.h"

namespace RotateDualLevelQuantNS {
enum class TargetType { FP4_E2M1, FP8_E4M3FN, FP8_E5M2, UNSUPPORTED };

struct Attributes {
    int64_t axis = -1;
    std::string roundMode = "rint";
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;
    int64_t groupListType = 0;
    int64_t level0BlockSize = LEVEL0_BLOCK_SIZE;
    int64_t level1BlockSize = LEVEL1_BLOCK_SIZE;
};

struct ShapeInfo {
    std::vector<int64_t> level0Scale;
    std::vector<int64_t> level1Scale;
    uint64_t rows = 1;
    int64_t n = 0;
    int64_t rotK = 0;
    int64_t rotationsPerGroup = 1;
    bool empty = false;
    bool dynamic = false;
};

inline int64_t CeilDiv(int64_t n, int64_t divisor) { return n / divisor + (n % divisor != 0); }

inline float TargetMax(TargetType type)
{
    switch (type) {
        case TargetType::FP4_E2M1:
            return 6.0f;
        case TargetType::FP8_E4M3FN:
            return 448.0f;
        case TargetType::FP8_E5M2:
            return 57344.0f;
        default:
            return 0.0f;
    }
}

inline bool Invalid(std::string& reason, const char* message)
{
    reason = message;
    return false;
}

inline bool ValidateBlockSizes(int64_t level0BlockSize, int64_t level1BlockSize, std::string& reason)
{
    if (level0BlockSize != LEVEL0_BLOCK_SIZE) {
        return Invalid(reason, "level0_block_size must be 512");
    }
    if (level1BlockSize != LEVEL1_BLOCK_SIZE) {
        return Invalid(reason, "level1_block_size must be 32");
    }
    return true;
}

inline bool ValidateAttributes(const Attributes& attr, std::size_t rank, TargetType target, std::string& reason)
{
    if (!ValidateBlockSizes(attr.level0BlockSize, attr.level1BlockSize, reason)) {
        return false;
    }
    if (target == TargetType::UNSUPPORTED) {
        return Invalid(reason, "y must be FLOAT4_E2M1, FLOAT8_E4M3FN or FLOAT8_E5M2");
    }
    if (attr.axis != -1 && attr.axis != static_cast<int64_t>(rank) - 1) {
        return Invalid(reason, "axis must be -1 or the positive index of the last input dimension");
    }
    if (attr.trans || (attr.groupListType != 0 && attr.groupListType != 1)) {
        return Invalid(reason, "trans must be false and group_list_type must be 0 or 1");
    }
    if (attr.roundMode != "rint" && attr.roundMode != "round" && attr.roundMode != "floor") {
        return Invalid(reason, "round_mode must be rint, round or floor");
    }
    const bool fp4 = target == TargetType::FP4_E2M1;
    if ((!fp4 && attr.roundMode != "rint") ||
        (fp4 ? (attr.scaleAlg != 0 && attr.scaleAlg != 2) : (attr.scaleAlg != 0 && attr.scaleAlg != 1))) {
        return Invalid(reason, "FP4 supports scale_alg 0/2; FP8 supports scale_alg 0/1 and round_mode rint only");
    }
    if (!std::isfinite(attr.dstTypeMax) ||
        (attr.scaleAlg == 2 ? (attr.dstTypeMax < 6.0 || attr.dstTypeMax > 12.0) : attr.dstTypeMax != 0.0)) {
        return Invalid(reason, "dst_type_max must be in [6,12] for scale_alg 2, and zero otherwise");
    }
    return true;
}

// Shared by ACLNN, graph inference and tiling. -1 is accepted only during graph inference.
// Unknown rank (-2) is handled by graph inference before calling this function.
inline bool ValidateShapes(const std::vector<int64_t>& x, const std::vector<int64_t>& rotation,
                           const std::vector<int64_t>* groups, bool allowUnknown, ShapeInfo& info, std::string& reason)
{
    info = ShapeInfo{};
    if (x.empty() || x.size() > 7) {
        return Invalid(reason, "x rank must be in [1,7]");
    }
    const bool grouped = groups != nullptr;
    if (grouped ? (x.size() != 2 || groups->size() != 1 || rotation.size() < 3 || rotation.size() > 4) :
                  (rotation.size() < 2 || rotation.size() > 3)) {
        return Invalid(reason, "ordinary rotation is [K,K]/[N/K,K,K]; grouped inputs are x[M,N], "
                               "groups[E], rotation[E,K,K]/[E,N/K,K,K]");
    }
    for (const auto* shape : {&x, &rotation, groups}) {
        if (shape == nullptr) {
            continue;
        }
        for (int64_t dim : *shape) {
            if (dim < 0 && (!allowUnknown || dim != -1)) {
                return Invalid(reason, "shape dimensions must be nonnegative; only graph inference accepts -1");
            }
            info.dynamic = info.dynamic || dim == -1;
        }
    }
    info.n = x.back();
    info.rotK = rotation.back();
    const int64_t k = info.rotK;
    if (k != -1 && k != 32 && k != 64 && k != 128 && k != 256) {
        return Invalid(reason, "rotation K must be one of 32,64,128,256");
    }
    const int64_t otherK = rotation[rotation.size() - 2];
    if ((otherK != -1 && otherK != 32 && otherK != 64 && otherK != 128 && otherK != 256) ||
        (k != -1 && otherK != -1 && otherK != k)) {
        return Invalid(reason, "the last two rotation dimensions must be equal supported K values");
    }
    if (info.n != -1 && k != -1 && info.n % k != 0) {
        return Invalid(reason, "the last input dimension N must be divisible by rotation K");
    }
    if (grouped) {
        const int64_t e = groups->front();
        if ((e != -1 && (e < 1 || e > 1024)) ||
            (rotation.front() != -1 && (rotation.front() < 1 || rotation.front() > 1024)) ||
            (e != -1 && rotation.front() != -1 && e != rotation.front()) || info.n == 0) {
            return Invalid(reason, "grouped mode requires positive N and matching E in [1,1024]");
        }
    }
    const bool perBlockRotation = rotation.size() == (grouped ? 4U : 3U);
    if (perBlockRotation) {
        info.rotationsPerGroup = rotation[grouped ? 1 : 0];
        if (info.n != -1 && k != -1 && info.rotationsPerGroup != -1 && info.rotationsPerGroup != info.n / k) {
            return Invalid(reason, "the number of column-block rotations must equal N/K");
        }
    }
    // Check each dimension first: a later zero must not mask an oversized dimension.
    for (int64_t dim : x) {
        if (dim > std::numeric_limits<int32_t>::max()) {
            return Invalid(reason, "input dimensions must fit INT32");
        }
        info.empty = info.empty || dim == 0;
    }
    bool zeroLeadingDimension = false;
    bool unknownLeadingDimension = false;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        zeroLeadingDimension = zeroLeadingDimension || x[i] == 0;
        unknownLeadingDimension = unknownLeadingDimension || x[i] == -1;
    }
    // M depends only on x's leading dimensions. Unknown rotation/group metadata
    // or an unknown N must not hide an already known, overflowing M.
    if (!unknownLeadingDimension && !zeroLeadingDimension) {
        for (std::size_t i = 0; i + 1 < x.size(); ++i) {
            if (info.rows > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) / static_cast<uint64_t>(x[i])) {
                return Invalid(reason, "the flattened M dimension must fit INT32");
            }
            info.rows *= static_cast<uint64_t>(x[i]);
        }
    } else {
        info.rows = 0;
    }
    info.level0Scale = x;
    info.level0Scale.back() = info.n == -1 ? -1 : CeilDiv(info.n, LEVEL0_BLOCK_SIZE);
    info.level1Scale = x;
    info.level1Scale.back() = info.n == -1 ? -1 : CeilDiv(info.n, LEVEL1_BLOCK_SIZE * SCALE_PAIR_SIZE);
    info.level1Scale.push_back(SCALE_PAIR_SIZE);
    return true;
}
} // namespace RotateDualLevelQuantNS
