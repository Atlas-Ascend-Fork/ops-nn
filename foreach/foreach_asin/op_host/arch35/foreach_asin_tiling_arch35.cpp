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
 * \file foreach_asin_tiling.cpp
 * \brief tiling implementation for foreach_asin
 */

#include <vector>
#include <algorithm>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "foreach/foreach_asin/op_kernel/arch35/foreach_asin_tiling_data.h"
#include "foreach/foreach_asin/op_kernel/arch35/foreach_asin_tiling_key.h"

using namespace AscendC;

namespace optiling {

constexpr int64_t DCACHE_SIZE = 128 * 1024;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16777216; // 16M
constexpr int64_t MIN_PER_CORE_ELEMENTS = 1024;
constexpr int32_t ALIGN_SIZE = 32;
constexpr size_t MAX_DIM_NUM = 8U;

struct ForeachAsinCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
};

static bool IsSupportedInputDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

static ge::graphStatus ValidateForeachAsin(gert::TilingContext* context)
{
    auto computeNodeInfoPtr = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfoPtr);
    auto instanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, instanceInfoPtr);
    uint64_t tensorNum = instanceInfoPtr->GetInstanceNum();
    // 校验 x 的 Tensor 数量不超过 256，空列表放行
    OP_CHECK_IF(tensorNum > static_cast<uint64_t>(FOREACH_ASIN_MAX_TENSOR_NUM),
                OP_LOGE_FOR_INVALID_TENSORNUM(
                    context->GetNodeName(), "x", static_cast<int64_t>(tensorNum),
                    ("within the range [0, " + std::to_string(FOREACH_ASIN_MAX_TENSOR_NUM) + "]").c_str()),
                return ge::GRAPH_FAILED);
    // 校验 x 与 y 的 Tensor 数量一致
    OP_CHECK_IF(context->GetComputeNodeOutputNum() != tensorNum,
                OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                    context->GetNodeName(), "x and y",
                    (std::to_string(tensorNum) + " and " + std::to_string(context->GetComputeNodeOutputNum())).c_str(),
                    "The tensor nums in {x, y} must be the same"),
                return ge::GRAPH_FAILED);

    ge::DataType listDtype = ge::DT_UNDEFINED;
    int64_t totalElements = 0;
    for (uint64_t i = 0; i < tensorNum; i++) {
        auto inputDesc = context->GetDynamicInputDesc(0, i);
        auto inputShape = context->GetDynamicInputShape(0, i);
        // 校验 x 的 desc 与 shape 非空
        OP_CHECK_IF(inputDesc == nullptr || inputShape == nullptr,
                    OP_LOGE(context, "The input %lu desc or shape is null.", i), return ge::GRAPH_FAILED);

        ge::DataType inputDtype = inputDesc->GetDataType();
        // 校验 x 的 dtype 在支持范围内
        OP_CHECK_IF(
            !IsSupportedInputDtype(inputDtype),
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x",
                                      ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(), "FP32, FP16 or BF16"),
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

        const gert::Shape& inputStorageShape = inputShape->GetStorageShape();
        // 校验 x 的维度不超过 8 维
        OP_CHECK_IF(inputStorageShape.GetDimNum() > MAX_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(inputStorageShape.GetDimNum()).c_str(),
                        ("The shape dim of the " + std::to_string(i) +
                         "th tensor in the tensor list should be less than or equal to " + std::to_string(MAX_DIM_NUM))
                            .c_str()),
                    return ge::GRAPH_FAILED);
        int64_t dataCount = inputStorageShape.GetShapeSize();
        // 校验 x 的元素数非负且累计不溢出
        OP_CHECK_IF(dataCount < 0 || totalElements > std::numeric_limits<int64_t>::max() - dataCount,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(dataCount).c_str(),
                        ("The shape size of the " + std::to_string(i) +
                         "th tensor in the tensor list must be 0 or a positive number, and the total element "
                         "count of the tensor list must not overflow")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        totalElements += dataCount;

        auto outputDesc = context->GetOutputDesc(i);
        auto outputShape = context->GetOutputShape(i);
        // 校验 y 的 desc 与 shape 非空
        OP_CHECK_IF(outputDesc == nullptr || outputShape == nullptr,
                    OP_LOGE(context, "The output %lu desc or shape is null.", i), return ge::GRAPH_FAILED);
        // 校验 y 的 dtype 与 x 一致
        OP_CHECK_IF(
            outputDesc->GetDataType() != listDtype,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(outputDesc->GetDataType()).c_str(),
                "The dtype of y must be the same as x"),
            return ge::GRAPH_FAILED);
        const gert::Shape& outputStorageShape = outputShape->GetStorageShape();
        // 校验 y 的维度不超过 8 维
        OP_CHECK_IF(outputStorageShape.GetDimNum() > MAX_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "y", std::to_string(outputStorageShape.GetDimNum()).c_str(),
                        ("The shape dim of the " + std::to_string(i) +
                         "th tensor in tensor list y must be less than or equal to " + std::to_string(MAX_DIM_NUM))
                            .c_str()),
                    return ge::GRAPH_FAILED);
        // 校验 y 的元素数不小于对应位置的 x
        OP_CHECK_IF(
            outputStorageShape.GetShapeSize() < dataCount,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                context->GetNodeName(), "y",
                (std::to_string(dataCount) + " and " + std::to_string(outputStorageShape.GetShapeSize())).c_str(),
                ("The shape size of the " + std::to_string(i) +
                 "th tensor in tensor list y should be greater than or equal to that of the tensor "
                 "in the same position of the another tensor list x")
                    .c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfoFallback(gert::TilingContext* context, int64_t& coreNum, int64_t& ubSize)
{
    auto compileInfo = reinterpret_cast<const ForeachAsinCompileInfo*>(context->GetCompileInfo());
    if (compileInfo != nullptr && compileInfo->coreNum > 0 && compileInfo->ubSize > 0) {
        coreNum = compileInfo->coreNum;
        ubSize = compileInfo->ubSize;
        return ge::GRAPH_SUCCESS;
    }
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ub = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        ubSize = static_cast<int64_t>(ub);
        if (coreNum > 0 && ubSize > 0) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

/**
 * @brief Tiling 主函数：计算切分参数并设置 tiling data
 */
static ge::graphStatus ForeachAsinTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    int64_t maxCoreNum = 0;
    int64_t ubSize = 0;
    OP_CHECK_IF(GetPlatformInfoFallback(context, maxCoreNum, ubSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Failed to get platform info"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((ubSize <= DCACHE_SIZE), OP_LOGE(context, "ub size less than DCache Size"), return ge::GRAPH_FAILED);
    int64_t ubSizeActual = ubSize - DCACHE_SIZE;

    OP_CHECK_IF(ValidateForeachAsin(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateForeachAsin failed"),
                return ge::GRAPH_FAILED);

    // 2. 获取输入 tensor 列表信息
    auto computeNodeInfoPtr = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfoPtr);
    auto instanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, instanceInfoPtr);
    uint64_t tensorNum = instanceInfoPtr->GetInstanceNum();

    if (tensorNum == 0) {
        ForeachAsinTilingData* emptyTilingData = context->GetTilingData<ForeachAsinTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, emptyTilingData);
        emptyTilingData->tensorNum = 0;
        emptyTilingData->tensorCumulativeOffset[0] = 0;
        emptyTilingData->totalElements = 0;
        emptyTilingData->perCoreElements = MIN_PER_CORE_ELEMENTS;
        emptyTilingData->needCoreNum = 1;
        context->SetTilingKey(GET_TPL_TILING_KEY(FOREACH_ASIN_TPL_DTYPE_FP32));
        context->SetBlockDim(emptyTilingData->needCoreNum);
        auto emptyWorkspaces = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, emptyWorkspaces);
        emptyWorkspaces[0] = ASCENDC_TOOLS_WORKSPACE;
        OP_LOGI(context, "ForeachAsin tiling: empty tensor list, skip computation.");
        return ge::GRAPH_SUCCESS;
    }

    // 3. 初始化 tiling data
    ForeachAsinTilingData* tilingData = context->GetTilingData<ForeachAsinTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingData->tensorNum = static_cast<int32_t>(tensorNum);
    tilingData->tensorCumulativeOffset[0] = 0;
    tilingData->totalElements = 0;

    // 4. 计算每个 tensor 的元素数和累计偏移量
    for (int32_t i = 0; i < tilingData->tensorNum; i++) {
        auto tensorShapePtr = context->GetDynamicInputShape(0, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, tensorShapePtr);
        auto tensorShape = tensorShapePtr->GetStorageShape();
        int64_t numel = tensorShape.GetShapeSize();
        tilingData->totalElements += numel;
        tilingData->tensorCumulativeOffset[i + 1] = tilingData->totalElements;
    }

    // 5. 计算核切分参数
    if (tilingData->totalElements == 0) {
        tilingData->perCoreElements = MIN_PER_CORE_ELEMENTS;
        tilingData->needCoreNum = 1;
    } else {
        tilingData->perCoreElements = std::max(MIN_PER_CORE_ELEMENTS,
                                               (tilingData->totalElements + maxCoreNum - 1) / maxCoreNum);
        // 对齐到 32
        tilingData->perCoreElements = ((tilingData->perCoreElements + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        tilingData->needCoreNum = static_cast<int32_t>((tilingData->totalElements + tilingData->perCoreElements - 1) /
                                                       tilingData->perCoreElements);
    }

    // 6. 根据数据类型确定 tiling key
    auto inputDesc = context->GetDynamicInputDesc(0, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dtype = inputDesc->GetDataType();

    uint64_t tilingKey = 0;
    if (dtype == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_ASIN_TPL_DTYPE_FP32);
    } else if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_ASIN_TPL_DTYPE_FP16);
    } else if (dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(FOREACH_ASIN_TPL_DTYPE_BF16);
    } else {
        OP_LOGE(context, "Unsupported dtype: %d", static_cast<int>(dtype));
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);

    // 7. 设置核数和 workspace
    context->SetBlockDim(tilingData->needCoreNum);
    auto workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = ASCENDC_TOOLS_WORKSPACE;

    // 8. 设置本地内存大小
    auto res = context->SetLocalMemorySize(ubSizeActual);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context, "SetLocalMemorySize ubSize = %ld failed.", ubSizeActual),
                return ge::GRAPH_FAILED);

    // 9. 打印 tiling 信息
    OP_LOGI(context,
            "ForeachAsin tiling: tensorNum=%d, totalElements=%ld, "
            "perCoreElements=%ld, needCoreNum=%d, tilingKey=%lu",
            tilingData->tensorNum, tilingData->totalElements, tilingData->perCoreElements, tilingData->needCoreNum,
            tilingKey);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief TilingParse 回调：获取平台信息
 */
static ge::graphStatus TilingParseForForeachAsin(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingParseForForeachAsin entering.");
    auto compileInfo = context->GetCompiledInfo<ForeachAsinCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context, "Failed to get core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context, "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachAsin)
    .Tiling(ForeachAsinTilingFunc)
    .TilingParse<ForeachAsinCompileInfo>(TilingParseForForeachAsin);

} // namespace optiling
