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
 * \file sigmoid_cross_entropy_with_logits_grad_arch35_tiling.cpp
 * \brief
 */
#include "sigmoid_cross_entropy_with_logits_grad_arch35_tiling.h"
#include "register/op_def_registry.h"
#include "log/log.h"
#include "../../op_kernel/arch35/sigmoid_cross_entropy_with_logits_grad_tiling_data.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include <algorithm>
#include <limits>
#include <set>
#include <string>

using namespace ge;

namespace optiling {

static constexpr size_t IDX_PREDICT = 0;
static constexpr size_t IDX_TARGET = 1;
static constexpr size_t IDX_DOUT = 2;
static constexpr size_t WORKSPACE_NUM = 1;
static constexpr size_t MAX_DIM_NUM = 8;

static constexpr int64_t MIN_BITS_PER_CORE = 32768; // 4KB per core (32768 bits)
static constexpr int64_t ELEM_ALIGN = 512;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t UB_RESERVE_BYTES = 8 * 1024; // 为 InitBuffer 逐 buffer 对齐填充预留的余量

static bool GetPlatformInfo(gert::TilingContext* context, int64_t& coreNum, int64_t& ubSize)
{
    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return false;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (aivCoreNum == 0 || ubBytes == 0 || aivCoreNum > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        ubBytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    coreNum = static_cast<int64_t>(aivCoreNum);
    ubSize = static_cast<int64_t>(ubBytes);
    return true;
}

static bool ComputeSplit(int64_t totalElements, int64_t dtypeBytes, int64_t coreNum, int64_t ubSize, int64_t vRegSize,
                         SigmoidCrossEntropyWithLogitsGradTilingData& td)
{
    if (totalElements <= 0 || dtypeBytes <= 0 || coreNum <= 0 || ubSize <= 0 || vRegSize <= 0) {
        return false;
    }

    // 1.多核切分
    int64_t minElemsPerCore = std::max<int64_t>(1, MIN_BITS_PER_CORE / (dtypeBytes * 8));
    int64_t requestedCoreNum = Ops::Base::CeilDiv(totalElements, minElemsPerCore);
    int64_t usedCoreNum = std::max<int64_t>(1, std::min(requestedCoreNum, coreNum));
    int64_t elemsPerCore = Ops::Base::CeilAlign(Ops::Base::CeilDiv(totalElements, usedCoreNum), ELEM_ALIGN);
    int64_t launchCoreNum = Ops::Base::CeilDiv(totalElements, elemsPerCore);

    // 2.UB切分：UB 占用 = (3 输入队列 × BUFFER_NUM + 1 输出队列) × dtype 字节 + 5 个 float 临时 buffer
    int64_t perElemBytes = (3 * BUFFER_NUM + 1) * dtypeBytes + 5 * static_cast<int64_t>(sizeof(float));
    int64_t ubAvailableBytes = Ops::Base::FloorAlign(ubSize, vRegSize) - UB_RESERVE_BYTES;
    if (ubAvailableBytes <= 0) {
        return false;
    }
    int64_t alignElems = vRegSize / dtypeBytes;
    int64_t maxElemsPerTile = ubAvailableBytes / perElemBytes;
    if (maxElemsPerTile < alignElems) {
        return false;
    }
    int64_t elemsPerTile = std::min(elemsPerCore, Ops::Base::FloorAlign(maxElemsPerTile, alignElems));

    td.coreNum = launchCoreNum;
    td.elemsPerCore = elemsPerCore;
    td.elemsPerTile = elemsPerTile;
    return true;
}

static ge::graphStatus TilingForSigmoidCrossEntropyWithLogitsGrad(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for Arch35 architecture");
    auto nodeName = context->GetNodeName();
    if (nodeName != nullptr) {
        OP_LOGI(nodeName, "Enter SigmoidCrossEntropyWithLogitsGradTilingFunc");
    }
    auto predictShapePtr = context->GetInputShape(IDX_PREDICT);
    if (predictShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto predictShape = predictShapePtr->GetStorageShape();

    OP_CHECK_IF(predictShape.GetDimNum() > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    nodeName, "predict", std::to_string(predictShape.GetDimNum()).c_str(), "rank must be <= 8"),
                return ge::GRAPH_FAILED);

    int64_t totalElements = 1;
    for (size_t i = 0; i < predictShape.GetDimNum(); ++i) {
        int64_t dim = predictShape.GetDim(i);
        OP_CHECK_IF(dim < 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "predict", std::to_string(dim).c_str(),
                                                             "dimensions must be non-negative at tiling time"),
                    return ge::GRAPH_FAILED);
        if (dim != 0 && totalElements > std::numeric_limits<int64_t>::max() / dim) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "predict", Ops::Base::ToString(predictShape).c_str(),
                                                  "the total element count exceeds the supported int64 range");
            return ge::GRAPH_FAILED;
        }
        totalElements *= dim;
    }

    auto predictDesc = context->GetInputDesc(IDX_PREDICT);
    if (predictDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType inputDtype = predictDesc->GetDataType();

    const std::set<ge::DataType> supportedDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    OP_CHECK_IF(supportedDtypes.count(inputDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "predict", Ops::Base::ToString(inputDtype).c_str(),
                                                      "supported dtypes are FLOAT16, FLOAT, and BFLOAT16"),
                return ge::GRAPH_FAILED);

    auto targetDesc = context->GetInputDesc(IDX_TARGET);
    if (targetDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType targetDtype = targetDesc->GetDataType();
    OP_CHECK_IF(supportedDtypes.count(targetDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "target", Ops::Base::ToString(targetDtype).c_str(),
                                                      "supported dtypes are FLOAT16, FLOAT, and BFLOAT16"),
                return ge::GRAPH_FAILED);

    auto doutDesc = context->GetInputDesc(IDX_DOUT);
    if (doutDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType doutDtype = doutDesc->GetDataType();
    OP_CHECK_IF(supportedDtypes.count(doutDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "dout", Ops::Base::ToString(doutDtype).c_str(),
                                                      "supported dtypes are FLOAT16, FLOAT, and BFLOAT16"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(inputDtype != targetDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    nodeName, "predict and target",
                    (Ops::Base::ToString(inputDtype) + ", " + Ops::Base::ToString(targetDtype)).c_str(),
                    "their dtypes must be identical"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputDtype != doutDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    nodeName, "predict and dout",
                    (Ops::Base::ToString(inputDtype) + ", " + Ops::Base::ToString(doutDtype)).c_str(),
                    "their dtypes must be identical"),
                return ge::GRAPH_FAILED);

    auto targetShapePtr = context->GetInputShape(IDX_TARGET);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShapePtr);
    auto targetShape = targetShapePtr->GetStorageShape();
    OP_CHECK_IF(targetShape.GetDimNum() > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    nodeName, "target", std::to_string(targetShape.GetDimNum()).c_str(), "rank must be <= 8"),
                return ge::GRAPH_FAILED);

    auto doutShapePtr = context->GetInputShape(IDX_DOUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, doutShapePtr);
    auto doutShape = doutShapePtr->GetStorageShape();
    OP_CHECK_IF(doutShape.GetDimNum() > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    nodeName, "dout", std::to_string(doutShape.GetDimNum()).c_str(), "rank must be <= 8"),
                return ge::GRAPH_FAILED);

    bool targetShapeEqual = (predictShape.GetDimNum() == targetShape.GetDimNum());
    for (size_t i = 0; i < predictShape.GetDimNum() && targetShapeEqual; ++i) {
        if (predictShape.GetDim(i) != targetShape.GetDim(i)) {
            targetShapeEqual = false;
        }
    }
    const std::string targetShapeReason = "must exactly match predict shape " + Ops::Base::ToString(predictShape);
    OP_CHECK_IF(!targetShapeEqual,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "target", Ops::Base::ToString(targetShape).c_str(),
                                                      targetShapeReason.c_str()),
                return ge::GRAPH_FAILED);
    const auto predictElements = predictShape.GetShapeSize();
    const auto doutElements = doutShape.GetShapeSize();
    const std::string doutSizeReason = "element count must equal predict element count " +
                                       std::to_string(predictElements);
    OP_CHECK_IF(predictElements != doutElements,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(nodeName, "dout", std::to_string(doutElements).c_str(),
                                                          doutSizeReason.c_str()),
                return ge::GRAPH_FAILED);

    SigmoidCrossEntropyWithLogitsGradTilingData*
        tilingData = context->GetTilingData<SigmoidCrossEntropyWithLogitsGradTilingData>();
    if (tilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData->totalLength = totalElements;

    if (totalElements == 0) {
        tilingData->coreNum = 1;
        tilingData->elemsPerCore = 0;
        tilingData->elemsPerTile = 1;
        size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
        if (ws != nullptr) {
            ws[0] = 0;
        }
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    int64_t coreNum = 0;
    int64_t ubSize = 0;
    if (!GetPlatformInfo(context, coreNum, ubSize)) {
        OP_LOGE(nodeName, "failed to get platform info");
        return ge::GRAPH_FAILED;
    }
    int64_t vRegSize = Ops::Base::GetVRegSize(context);

    int64_t dtypeBytes = (inputDtype == ge::DT_FLOAT) ? 4 : 2;
    if (!ComputeSplit(totalElements, dtypeBytes, coreNum, ubSize, vRegSize, *tilingData)) {
        OP_LOGE(nodeName, "failed to compute tiling split");
        return ge::GRAPH_FAILED;
    }

    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    if (ws != nullptr) {
        ws[0] = 0;
    }
    context->SetBlockDim(tilingData->coreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseForSigmoidCrossEntropyWithLogitsGrad(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SigmoidCrossEntropyWithLogitsGrad)
    .Tiling(TilingForSigmoidCrossEntropyWithLogitsGrad)
    .TilingParse<SigmoidCrossEntropyWithLogitsGradCompileInfo>(ParseForSigmoidCrossEntropyWithLogitsGrad);

} // namespace optiling
