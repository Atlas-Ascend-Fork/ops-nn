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
 * \file foreach_erfc_tiling_arch35.cpp
 * \brief Tiling implementation for foreach_erfc operator (arch35).
 */

#include <algorithm>
#include <limits>
#include "util/math_util.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "foreach_erfc_tiling_arch35.h"
#include "../../../foreach_utils/op_host/foreach_flat_regbase_tiling.h"
#include "../../op_kernel/arch35/foreach_erfc_tiling_key.h"
#include "foreach/foreach_erfc/op_kernel/arch35/foreach_erfc_scalar_tiling.h"

using namespace AscendC;

namespace optiling {
namespace {
constexpr int64_t INPUT_IDX = 0;
constexpr size_t MAX_DIM_NUM = 8U;
// 与 FOREACH 家族一致的粗粒度参考：每核至少处理 4096B 的数据块。
constexpr int64_t ERFC_FLAT_COARSE_BLOCK_BYTES = 4096;
// 给 Erfc 逐 repeat 尾部访问与 TPipe 对齐留出的 UB 余量。
constexpr int64_t ERFC_UB_RESERVE_BYTES = 512;
constexpr int64_t ERFC_UB_DOUBLE_BUFFER = 2;
constexpr int64_t ERFC_BUFFER_FLOWS = 2; // 输入 + 输出

static bool IsSupportedInputDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

static ge::graphStatus ValidateForeachErfc(gert::TilingContext* context, int32_t tensorNum)
{
    // 校验 x 与 y 的 Tensor 数量一致
    OP_CHECK_IF(context->GetComputeNodeOutputNum() != static_cast<uint64_t>(tensorNum),
                OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                    context->GetNodeName(), "x and y",
                    (std::to_string(tensorNum) + " and " + std::to_string(context->GetComputeNodeOutputNum())).c_str(),
                    "The tensor nums in {x, y} must be the same"),
                return ge::GRAPH_FAILED);

    ge::DataType listDtype = ge::DT_UNDEFINED;
    for (int32_t i = 0; i < tensorNum; i++) {
        auto inputDesc = context->GetDynamicInputDesc(INPUT_IDX, i);
        // 校验 x 的 desc 非空
        OP_CHECK_IF(
            inputDesc == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                context->GetNodeName(), "x",
                ("the descriptor of tensor x[" + std::to_string(i) + "] is missing in the tiling context").c_str()),
            return ge::GRAPH_FAILED);
        auto inputShape = context->GetDynamicInputShape(INPUT_IDX, i);
        // 校验 x 的 shape 非空
        OP_CHECK_IF(inputShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "x",
                        ("the shape of tensor x[" + std::to_string(i) + "] is missing in the tiling context").c_str()),
                    return ge::GRAPH_FAILED);
        const gert::Shape& inputStorageShape = inputShape->GetStorageShape();
        // 校验 x 的维度不超过 8 维
        OP_CHECK_IF(inputStorageShape.GetDimNum() > MAX_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(inputStorageShape.GetDimNum()).c_str(),
                        ("The shape dim of the " + std::to_string(i) +
                         "th tensor in the tensor list should be less than or equal to " + std::to_string(MAX_DIM_NUM))
                            .c_str()),
                    return ge::GRAPH_FAILED);
        // 校验 x 的元素数非负（GetShapeSize 溢出时返回 INT64_MIN 哨兵，一并拦截）
        OP_CHECK_IF(inputStorageShape.GetShapeSize() < 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(inputStorageShape.GetShapeSize()).c_str(),
                        ("The shape size of the " + std::to_string(i) +
                         "th tensor in the tensor list must be 0 or a positive number")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        for (size_t axis = 0; axis < inputStorageShape.GetDimNum(); ++axis) {
            OP_CHECK_IF(
                inputStorageShape.GetDim(axis) < 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "x", Ops::Base::ToString(inputStorageShape).c_str(),
                    ("tensor x[" + std::to_string(i) + "] has a negative dimension at axis " + std::to_string(axis))
                        .c_str()),
                return ge::GRAPH_FAILED);
        }
        ge::DataType inputDtype = inputDesc->GetDataType();
        // 校验 x 的 dtype 在支持范围内
        OP_CHECK_IF(!IsSupportedInputDtype(inputDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x",
                                              ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(),
                                              "DT_FLOAT, DT_FLOAT16 or DT_BF16"),
                    return ge::GRAPH_FAILED);
        if (listDtype == ge::DT_UNDEFINED) {
            listDtype = inputDtype;
        } else {
            // 校验 x 列表内各 Tensor 的 dtype 一致
            OP_CHECK_IF(inputDtype != listDtype,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            context->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(),
                            ("The dtypes of all tensors in the tensor list must be the same. "
                             "Currently, the dtype of the " +
                             std::to_string(i) + "th tensor is inconsistent with that (" +
                             ge::TypeUtils::DataTypeToSerialString(listDtype) + ") of other tensors")
                                .c_str()),
                        return ge::GRAPH_FAILED);
        }

        auto outputDesc = context->GetOutputDesc(i);
        // 校验 y 的 desc 非空
        OP_CHECK_IF(
            outputDesc == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                context->GetNodeName(), "y",
                ("the descriptor of tensor y[" + std::to_string(i) + "] is missing in the tiling context").c_str()),
            return ge::GRAPH_FAILED);
        auto outputShape = context->GetOutputShape(i);
        // 校验 y 的 shape 非空
        OP_CHECK_IF(outputShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "y",
                        ("the shape of tensor y[" + std::to_string(i) + "] is missing in the tiling context").c_str()),
                    return ge::GRAPH_FAILED);
        // 校验 y 的 dtype 与 x 一致
        OP_CHECK_IF(
            outputDesc->GetDataType() != listDtype,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(outputDesc->GetDataType()).c_str(),
                "The dtype of y must be the same as x"),
            return ge::GRAPH_FAILED);
        const gert::Shape& outputStorageShape = outputShape->GetStorageShape();
        OP_CHECK_IF(
            outputStorageShape.GetDimNum() > MAX_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context->GetNodeName(), "y", std::to_string(outputStorageShape.GetDimNum()).c_str(),
                ("the rank of tensor y[" + std::to_string(i) + "] must not exceed " + std::to_string(MAX_DIM_NUM))
                    .c_str()),
            return ge::GRAPH_FAILED);
        // 校验 y 的 shape size 不小于对应位置的 x（与 arch22 公共校验口径一致：仅拒绝 y 小于 x）
        OP_CHECK_IF(inputStorageShape.GetShapeSize() > outputStorageShape.GetShapeSize(),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        context->GetNodeName(), "x and y",
                        (std::to_string(inputStorageShape.GetShapeSize()) + " and " +
                         std::to_string(outputStorageShape.GetShapeSize()))
                            .c_str(),
                        ("The shape size of " + std::to_string(i) +
                         "th tensor in tensor list y should be greater than or equal to that of the tensor "
                         "in the same position of the another tensor list x")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// Retain Erfc's adaptive element partition while using the shared flat payload.
static void AssignFlatRanges(ForeachFlatTilingData* tiling, int64_t totalElements, uint16_t tensorCount,
                             int64_t coreCount, int64_t elementsPerBlock)
{
    const int64_t blocks = ForeachFlatRegbaseTiling::CeilDiv(totalElements, elementsPerBlock);
    const int64_t base = blocks / coreCount;
    const int64_t remainder = blocks % coreCount;
    int64_t cursor = 0;
    for (int64_t core = 0; core < coreCount; ++core) {
        const int64_t share = base + static_cast<int64_t>(core < remainder);
        const int64_t remaining = totalElements - cursor;
        const int64_t count = share <= remaining / elementsPerBlock ? share * elementsPerBlock : remaining;
        ForeachFlatRegbaseTiling::LocateFlatOffset(cursor, tensorCount, tiling->tensorDataCountList,
                                                   tiling->tensorStartList[core], tiling->tensorStartOffsetList[core]);
        ForeachFlatRegbaseTiling::LocateFlatOffset(cursor + count - 1, tensorCount, tiling->tensorDataCountList,
                                                   tiling->tensorEndList[core], tiling->tensorEndOffsetList[core]);
        cursor += count;
    }
}

static ge::graphStatus Tiling4ForeachErfc(gert::TilingContext* context)
{
    auto node = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, node);
    auto instances = node->GetInputInstanceInfo(INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, instances);
    const uint64_t tensorCount = instances->GetInstanceNum();
    OP_CHECK_IF(tensorCount > MAX_TENSOR_NUM_FOREACH_ERFC,
                OP_LOGE_FOR_INVALID_LISTSIZE(
                    context->GetNodeName(), "x", std::to_string(tensorCount).c_str(),
                    ("within the range [0, " + std::to_string(MAX_TENSOR_NUM_FOREACH_ERFC) + "]").c_str()),
                return ge::GRAPH_FAILED);
    if (ValidateForeachErfc(context, static_cast<int32_t>(tensorCount)) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto tiling = context->GetTilingData<ForeachFlatTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    auto workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    if (tensorCount == 0) {
        *tiling = ForeachFlatTilingData{};
        tiling->tileElems = ERFC_UB_RESERVE_BYTES / sizeof(float);
        tiling->tensorEndOffsetList[0] = -1;
        workspace[0] = 0;
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(FOREACH_ERFC_TPL_SCH_MODE_FP32));
        return context->SetLocalMemorySize(0);
    }

    // The common builder supplies the plain payload, checked accumulation and platform resources.
    if (ForeachFlatRegbaseTiling::Build<1>(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = context->GetDynamicInputDesc(INPUT_IDX, 0)->GetDataType();
    const int64_t elementBytes = ge::GetSizeByDataType(dtype);
    int64_t totalElements = 0;
    bool allScalar = true;
    for (uint64_t i = 0; i < tensorCount; ++i) {
        totalElements += tiling->tensorDataCountList[i];
        allScalar = allScalar && tiling->tensorDataCountList[i] == 1;
    }
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    uint64_t key = GET_TPL_TILING_KEY(FOREACH_ERFC_TPL_SCH_MODE_FP16);
    if (dtype == ge::DT_FLOAT) {
        key = GET_TPL_TILING_KEY(FOREACH_ERFC_TPL_SCH_MODE_FP32);
    } else if (dtype == ge::DT_BF16) {
        key = GET_TPL_TILING_KEY(FOREACH_ERFC_TPL_SCH_MODE_BF16);
    }
    context->SetTilingKey(key);
    workspace[0] = 0;

    if (allScalar && tensorCount <= 32 && platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950) {
        tiling->tileElems = FOREACH_ERFC_SCALAR_FLAG | static_cast<uint32_t>(tensorCount);
        context->SetBlockDim(1);
        return context->SetLocalMemorySize(0);
    }

    int64_t coreCount = 1;
    if (totalElements == 0) {
        tiling->tensorStartList[0] = 0;
        tiling->tensorEndList[0] = 0;
        tiling->tensorStartOffsetList[0] = 0;
        tiling->tensorEndOffsetList[0] = -1;
    } else {
        const int64_t availableCores = std::max<int64_t>(platform.GetCoreNumAiv(), 1);
        const int64_t lanes = std::min<int64_t>(availableCores, tensorCount);
        const int64_t perCore = ForeachFlatRegbaseTiling::CeilDiv(totalElements, lanes);
        const int64_t granularity = std::clamp<int64_t>(ERFC_FLAT_COARSE_BLOCK_BYTES / elementBytes, 1, perCore);
        const int64_t blocks = ForeachFlatRegbaseTiling::CeilDiv(totalElements, granularity);
        coreCount = std::min<int64_t>(std::min<int64_t>(availableCores, FOREACH_FLAT_MAX_CORE_COUNT), blocks);
        AssignFlatRanges(tiling, totalElements, static_cast<uint16_t>(tensorCount), coreCount, granularity);
    }
    context->SetBlockDim(static_cast<uint32_t>(coreCount));

    // Preserve the native Erfc queue budget, including BF16's two FP32 scratch generations.
    int64_t bytesPerElement = elementBytes * ERFC_UB_DOUBLE_BUFFER * ERFC_BUFFER_FLOWS;
    if (dtype == ge::DT_BF16) {
        bytesPerElement += sizeof(float) * ERFC_UB_DOUBLE_BUFFER;
    }
    const int64_t availableUb = std::max<int64_t>(static_cast<int64_t>(ubBytes) - ERFC_UB_RESERVE_BYTES, 0);
    const int64_t blockBytes = Ops::Base::GetUbBlockSize(context);
    const int64_t tileElements = Ops::Base::FloorAlign((availableUb / bytesPerElement) * elementBytes, blockBytes) /
                                 elementBytes;
    OP_CHECK_IF(tileElements < 1 || tileElements > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context, "invalid Erfc flat tile element count %ld", tileElements), return ge::GRAPH_FAILED);
    tiling->tileElems = static_cast<uint32_t>(tileElements);
    // Erfc allocates its vector queues statically through TPipe.
    return context->SetLocalMemorySize(0);
}

static ge::graphStatus TilingParseForForeachErfc([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(ForeachErfc).Tiling(Tiling4ForeachErfc).TilingParse<ForeachErfcCompileInfo>(TilingParseForForeachErfc);
} // namespace optiling
