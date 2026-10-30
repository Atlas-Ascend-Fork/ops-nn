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
 * \file expand_into_jagged_permute_tiling.cpp
 * \brief expand_into_jagged_permute_tiling.cpp
 */
#include "expand_into_jagged_permute_tiling.h"
#include <string>
#include <vector>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr char OP_NAME[] = "ExpandIntoJaggedPermute";
constexpr char PERMUTE_NAME[] = "permute";
constexpr char INPUT_OFFSETS_NAME[] = "input_offsets";
constexpr char OUTPUT_OFFSETS_NAME[] = "output_offsets";
constexpr char OUTPUT_SIZE_NAME[] = "output_size";
constexpr size_t PERMUTE_INDEX = 0;
constexpr size_t INPUT_OFFSETS_INDEX = 1;
constexpr size_t OUTPUT_OFFSETS_INDEX = 2;
constexpr size_t OUTPUT_SIZE_ATTR = 0;

static void ReportTilingError(const char* opName, const std::string& reason)
{
    const char* safeOpName = opName == nullptr ? OP_NAME : opName;
    OP_LOGE_WITHOUT_REPORT(safeOpName, "%s", reason.c_str());
    REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                              (std::vector<const char*>{safeOpName, reason.c_str()}));
}

static ge::graphStatus ValidateInputDesc(const gert::TilingContext* context, size_t index, const char* tensorName)
{
    const auto* desc = context->GetInputDesc(index);
    if (desc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), tensorName);
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dataType = desc->GetDataType();
    if (dataType != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), tensorName, Ops::Base::ToString(dataType).c_str(), "INT32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReadInputShape(const gert::TilingContext* context, size_t index, const char* tensorName,
                                      const gert::StorageShape*& storageShape)
{
    storageShape = context->GetInputShape(index);
    if (storageShape == nullptr || context->GetInputTensor(index) == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), tensorName);
        return ge::GRAPH_FAILED;
    }
    const gert::Shape shape = storageShape->GetStorageShape();
    if (shape.GetDimNum() != 1) {
        const std::string actual = std::to_string(shape.GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), tensorName, actual.c_str(), "1D");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

template <typename T>
static T GetCeilInt(const T& value1, const T& value2)
{
    if (value2 == 0) {
        return value2;
    }
    return (value1 + value2 - 1) / value2;
}

static inline ge::graphStatus InputParamCheck(const gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    if (ValidateInputDesc(context, PERMUTE_INDEX, PERMUTE_NAME) != ge::GRAPH_SUCCESS ||
        ValidateInputDesc(context, INPUT_OFFSETS_INDEX, INPUT_OFFSETS_NAME) != ge::GRAPH_SUCCESS ||
        ValidateInputDesc(context, OUTPUT_OFFSETS_INDEX, OUTPUT_OFFSETS_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* permuteShape = nullptr;
    const gert::StorageShape* inputoffsetShape = nullptr;
    const gert::StorageShape* outputoffsetShape = nullptr;
    if (ReadInputShape(context, PERMUTE_INDEX, PERMUTE_NAME, permuteShape) != ge::GRAPH_SUCCESS ||
        ReadInputShape(context, INPUT_OFFSETS_INDEX, INPUT_OFFSETS_NAME, inputoffsetShape) != ge::GRAPH_SUCCESS ||
        ReadInputShape(context, OUTPUT_OFFSETS_INDEX, OUTPUT_OFFSETS_NAME, outputoffsetShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const int64_t permuteLength = permuteShape->GetStorageShape().GetDim(0);
    const int64_t inputOffsetLength = inputoffsetShape->GetStorageShape().GetDim(0);
    const int64_t outputOffsetLength = outputoffsetShape->GetStorageShape().GetDim(0);
    if (permuteLength + 1 != inputOffsetLength || permuteLength + 1 != outputOffsetLength) {
        const std::string actual = "permute=" + std::to_string(permuteLength) +
                                   ", input_offsets=" + std::to_string(inputOffsetLength) +
                                   ", output_offsets=" + std::to_string(outputOffsetLength);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "permute, input_offsets and output_offsets", actual.c_str(),
            "The lengths of input_offsets and output_offsets must both equal permute length + 1");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus Init(gert::TilingContext* context, ExpandIntoJaggedPermuteParam& param)
{
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        ReportTilingError(context->GetNodeName(), "GetWorkspaceSizes(1) returned nullptr");
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = sysWorkspaceSize;
    OP_LOGI(context, "get platform from compileInfo");

    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        ReportTilingError(context->GetNodeName(), "GetPlatformInfo() returned nullptr");
        return ge::GRAPH_FAILED;
    }
    auto ascendPlaform = platform_ascendc::PlatformAscendC(platformInfo);
    param.coreNum = static_cast<int64_t>(ascendPlaform.GetCoreNumAiv());
    uint64_t maxCoreMemery;
    ascendPlaform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxCoreMemery);
    param.maxCoreMemery = static_cast<int64_t>(maxCoreMemery);

    const gert::StorageShape* permuteShape = context->GetInputShape(0); // inputLen
    const gert::StorageShape* inputoffsetShape = context->GetInputShape(1);
    const gert::StorageShape* outputoffsetShape = context->GetInputShape(2);

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), OUTPUT_SIZE_NAME);
        return ge::GRAPH_FAILED;
    }
    const int64_t* outputSizePtr = attrs->GetAttrPointer<int64_t>(OUTPUT_SIZE_ATTR);
    if (outputSizePtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), OUTPUT_SIZE_NAME);
        return ge::GRAPH_FAILED;
    }
    param.outputSize = *outputSizePtr;
    if (param.outputSize < 0) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), OUTPUT_SIZE_NAME, std::to_string(param.outputSize).c_str(),
                                  "a non-negative integer");
        return ge::GRAPH_FAILED;
    }

    param.permute = permuteShape->GetStorageShape().GetDim(0);
    param.inputoffset = inputoffsetShape->GetStorageShape().GetDim(0);
    param.outputoffset = outputoffsetShape->GetStorageShape().GetDim(0);
    param.dtypeSize = ge::GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
    if (param.coreNum <= 0 || param.dtypeSize <= 0) {
        const std::string reason = "Invalid tiling metadata: coreNum=" + std::to_string(param.coreNum) +
                                   ", dtypeSize=" + std::to_string(param.dtypeSize) +
                                   "; expected both values to be positive";
        ReportTilingError(context->GetNodeName(), reason);
        return ge::GRAPH_FAILED;
    }

    param.oneTaskInputOffsetLen = SIZE;
    param.cacheLine = SIZE;
    param.numBlocks = param.coreNum;
    param.oneTaskLen = param.cacheLine / param.dtypeSize;
    param.taskNum = GetCeilInt(param.permute, param.oneTaskInputOffsetLen);
    param.numTail = param.permute - SIZE * (param.taskNum - 1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingKey(ExpandIntoJaggedPermuteParam& param)
{
    uint64_t tilingkey = 0;
    int64_t limitedSize = 512;
    if (param.inputoffset <= limitedSize) {
        param.tilingKey = tilingkey;
    } else {
        param.tilingKey = tilingkey + 1;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingData(gert::TilingContext* context, const ExpandIntoJaggedPermuteParam& param)
{
    ExpandIntoJaggedPermuteTilingData tilingData;
    tilingData.set_outputSize(param.outputSize);
    context->SetBlockDim(param.numBlocks);
    tilingData.set_blockFactor(param.taskNum);
    tilingData.set_inputLen(param.permute);
    tilingData.set_offsetLen(param.permute + 1);
    tilingData.set_lastTaskLen(param.numTail);
    tilingData.set_oneTaskOffsetLen(param.oneTaskInputOffsetLen);
    tilingData.set_oneTaskLen(param.oneTaskLen);
    auto rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr) {
        ReportTilingError(context->GetNodeName(), "GetRawTilingData() returned nullptr");
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static inline void PrintTilingData(const gert::TilingContext* context, const ExpandIntoJaggedPermuteParam& param)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print ExpandIntoJaggedPermute tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD("offsetLen", " param.offsetLen %ld", param.offsetLen);
    OP_LOGD("inputLen", " param.permute %ld", param.permute);
    OP_LOGD("oneTaskLen", " param.oneTaskLen %ld", param.oneTaskLen);
    OP_LOGD("oneTaskOffsetLen", " param.oneTaskInputOffsetLen %ld", param.oneTaskInputOffsetLen);
    OP_LOGD("numBlocks", " param.numBlocks %ld", param.numBlocks);
    OP_LOGD("taskNum", " param.taskNum %ld", param.taskNum);
    OP_LOGD("outputSize", " param.outputSize %ld", param.outputSize);
    OP_LOGD("numTail", " param.numTail %ld", param.numTail);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print ExpandIntoJaggedPermute tiling data end <<<<<<<<<<<<<<<<");
}

static ge::graphStatus TilingForExpandIntoJaggedPermute(gert::TilingContext* context)
{
    if (context == nullptr) {
        ReportTilingError(OP_NAME, "TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> ExpandIntoJaggedPermute Tiling start <<<<<<<<<<<<<<<<.");

    if (InputParamCheck(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    ExpandIntoJaggedPermuteParam param;
    if (Init(context, param) != ge::GRAPH_SUCCESS || SetTilingKey(param) != ge::GRAPH_SUCCESS ||
        SetTilingData(context, param) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    PrintTilingData(context, param);
    return context->SetTilingKey(param.tilingKey);
}

ge::graphStatus TilingPrepareExpandIntoJaggedPermute(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        ReportTilingError(OP_NAME, "TilingParseContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ExpandIntoJaggedPermute)
    .Tiling(TilingForExpandIntoJaggedPermute)
    .TilingParse<ExpandIntoJaggedPermuteCompileInfo>(TilingPrepareExpandIntoJaggedPermute);
} // namespace optiling
