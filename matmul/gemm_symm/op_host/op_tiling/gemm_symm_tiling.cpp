/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <string>

#include "error_util.h"
#include "matmul/gemm_symm/op_kernel/arch35/gemm_symm_tiling_data.h"
#include "matmul/gemm_symm/op_kernel/arch35/gemm_symm_tiling_key.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kIndexA = 0;
constexpr uint32_t kIndexB = 1;
constexpr uint32_t kIndexC = 2;
constexpr uint32_t kAttrAlpha = 0;
constexpr uint32_t kAttrBeta = 1;
constexpr uint32_t kAttrTransA = 2;
constexpr uint32_t kAttrTransB = 3;
constexpr uint32_t kAttrSymmType = 4;
constexpr uint32_t kAttrImplMode = 5;
constexpr uint32_t kTileLow = 256;
constexpr uint32_t kTileHigh = 192;
constexpr int64_t kCubeMathFp32Add = 4;
constexpr size_t kRank2 = 2;
constexpr size_t kRank3 = 3;
const char* kSymmTypeFull = "full";

int64_t DimM(const gert::Shape& shape, bool transposed)
{
    const size_t rank = shape.GetDimNum();
    return transposed ? shape.GetDim(rank - 1) : shape.GetDim(rank - 2);
}

int64_t DimK(const gert::Shape& shape, bool transposed)
{
    const size_t rank = shape.GetDimNum();
    return transposed ? shape.GetDim(rank - 2) : shape.GetDim(rank - 1);
}

uint32_t TriangleTasks(uint32_t m, uint32_t tile)
{
    const uint32_t tiles = (m + tile - 1) / tile;
    return tiles * (tiles + 1) / 2;
}
} // namespace

namespace optiling {
namespace gemm_symm {

static ge::graphStatus GemmSymmDoTiling(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    if (platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND950) {
        OP_LOGE(context->GetNodeName(), "GemmSymm only supports Ascend950");
        return ge::GRAPH_FAILED;
    }
    const auto* aShape = context->GetInputShape(kIndexA);
    const auto* bShape = context->GetInputShape(kIndexB);
    const auto* cShape = context->GetInputShape(kIndexC);
    const auto* aDesc = context->GetInputDesc(kIndexA);
    const auto* bDesc = context->GetInputDesc(kIndexB);
    const auto* yDesc = context->GetOutputDesc(0);
    const auto* attrs = context->GetAttrs();
    if (aShape == nullptr || bShape == nullptr || aDesc == nullptr || bDesc == nullptr || attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "a, b or attrs is null");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dtype = aDesc->GetDataType();
    if ((dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) || bDesc->GetDataType() != dtype ||
        (yDesc != nullptr && yDesc->GetDataType() != dtype)) {
        OP_LOGE(context->GetNodeName(), "GemmSymm supports float16 and bfloat16, and a, b, y must match");
        return ge::GRAPH_FAILED;
    }
    const auto* transAAttr = attrs->GetAttrPointer<bool>(kAttrTransA);
    const auto* transBAttr = attrs->GetAttrPointer<bool>(kAttrTransB);
    const auto* alphaAttr = attrs->GetAttrPointer<float>(kAttrAlpha);
    const auto* betaAttr = attrs->GetAttrPointer<float>(kAttrBeta);
    const auto* symmType = attrs->GetAttrPointer<char>(kAttrSymmType);
    const bool transA = transAAttr != nullptr && *transAAttr;
    const bool transB = transBAttr != nullptr && *transBAttr;
    const float alpha = alphaAttr != nullptr ? *alphaAttr : 1.0F;
    const float beta = betaAttr != nullptr ? *betaAttr : 1.0F;
    if (symmType != nullptr && std::string(symmType) != kSymmTypeFull) {
        OP_LOGE(context->GetNodeName(), "symm_type only supports full");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& a = aShape->GetStorageShape();
    const gert::Shape& b = bShape->GetStorageShape();
    if ((a.GetDimNum() != kRank2 && a.GetDimNum() != kRank3) || b.GetDimNum() != a.GetDimNum()) {
        OP_LOGE(context->GetNodeName(), "a and b must be rank 2 or 3");
        return ge::GRAPH_FAILED;
    }
    const int64_t m = DimM(a, transA);
    const int64_t k = DimK(a, transA);
    const int64_t bK = DimK(b, !transB);
    const int64_t bM = DimM(b, !transB);
    if (m <= 0 || k < 0 || bK != k || bM != m || (a.GetDimNum() == kRank3 && b.GetDim(0) != a.GetDim(0))) {
        OP_LOGE(context->GetNodeName(), "b shape does not match a");
        return ge::GRAPH_FAILED;
    }
    const bool hasC = cShape != nullptr;
    const auto* cDesc = context->GetInputDesc(kIndexC);
    if (hasC && (cDesc == nullptr || cDesc->GetDataType() != dtype)) {
        OP_LOGE(context->GetNodeName(), "c must use the same dtype as a");
        return ge::GRAPH_FAILED;
    }
    if (beta != 0.0F && !hasC) {
        OP_LOGE(context->GetNodeName(), "c is required when beta is not 0");
        return ge::GRAPH_FAILED;
    }
    if (k == 0 && beta != 0.0F && !hasC) {
        OP_LOGE(context->GetNodeName(), "c is required when k is 0 and beta is not 0");
        return ge::GRAPH_FAILED;
    }
    const auto* yShape = context->GetOutputShape(0);
    if (yShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "y is null");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& y = yShape->GetStorageShape();
    if (y.GetDimNum() != a.GetDimNum() || DimM(y, false) != m || DimK(y, false) != m ||
        (a.GetDimNum() == kRank3 && y.GetDim(0) != a.GetDim(0))) {
        OP_LOGE(context->GetNodeName(), "y must be [..., m, m] and match the batch of a");
        return ge::GRAPH_FAILED;
    }
    const uint32_t batch = a.GetDimNum() == kRank3 ? static_cast<uint32_t>(a.GetDim(0)) : 1U;
    if (hasC) {
        const gert::Shape& c = cShape->GetStorageShape();
        if (c.GetDimNum() != a.GetDimNum() || DimM(c, false) != m || DimK(c, false) != m) {
            OP_LOGE(context->GetNodeName(), "c must be [..., m, m]");
            return ge::GRAPH_FAILED;
        }
        if (a.GetDimNum() == kRank3 && c.GetDim(0) != a.GetDim(0)) {
            OP_LOGE(context->GetNodeName(), "c batch must match a");
            return ge::GRAPH_FAILED;
        }
    }

    const auto* implMode = attrs->GetAttrPointer<int64_t>(kAttrImplMode);
    // High precision only when beta * C is applied. Otherwise the tile stays 256.
    const bool highPrec = implMode != nullptr && *implMode == kCubeMathFp32Add && hasC && beta != 0.0F;
    const uint32_t tile = highPrec ? kTileHigh : kTileLow;
    const uint32_t tasks = TriangleTasks(static_cast<uint32_t>(m), tile);
    const uint32_t aicNum = static_cast<uint32_t>(platform.GetCoreNumAic());
    const uint32_t usedCoreNum = std::max(1U, std::min(aicNum == 0 ? 1U : aicNum, tasks == 0 ? 1U : tasks));

    GemmSymmTilingData tiling{};
    tiling.m = static_cast<uint32_t>(m);
    tiling.k = static_cast<uint32_t>(k);
    tiling.batch = batch;
    tiling.usedCoreNum = usedCoreNum;
    tiling.transA = transA ? 1 : 0;
    tiling.transB = transB ? 1 : 0;
    tiling.hasC = hasC ? 1 : 0;
    tiling.alpha = alpha;
    tiling.beta = hasC ? beta : 0.0F;

    const size_t sizeTilingData = sizeof(GemmSymmTilingData);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->GetRawTilingData()->SetDataSize(sizeTilingData);
    auto* tilingPtr = static_cast<GemmSymmTilingData*>(context->GetRawTilingData()->GetData());
    *tilingPtr = tiling;
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(transA ? GEMM_SYMM_TRANS_TRUE : GEMM_SYMM_TRANS_FALSE,
                                             transB ? GEMM_SYMM_TRANS_TRUE : GEMM_SYMM_TRANS_FALSE,
                                             highPrec ? GEMM_SYMM_HIGH_PREC_TRUE : GEMM_SYMM_HIGH_PREC_FALSE));
    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace gemm_symm
} // namespace optiling

namespace optiling {
struct GemmSymmCompileInfo {};

static ge::graphStatus GemmSymmTilingFunc(gert::TilingContext* context) { return gemm_symm::GemmSymmDoTiling(context); }

static ge::graphStatus TilingParseForGemmSymm(gert::TilingParseContext* context)
{
    if (context == nullptr || context->GetCompiledInfo<GemmSymmCompileInfo>() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GemmSymm).Tiling(GemmSymmTilingFunc).TilingParse<GemmSymmCompileInfo>(TilingParseForGemmSymm);
} // namespace optiling
