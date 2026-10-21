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
 * \file apply_ftrl_tiling.cpp
 * \brief
 */
#include <string>
#include <vector>

#include "apply_ftrl_tiling.h"

#include <graph/utils/type_utils.h>

#include <iostream>

#include "error_util.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_templates_registry.h"

using namespace ge;
using namespace ApplyFtrlOp;
using namespace ApplyFtrlOpTiling;

namespace optiling {
const size_t SYS_WORKSPACE = 16777216; // 16M
constexpr static int32_t SCALAR_INPUT_LR_INDEX = 4;
constexpr static int32_t INPUT_NUM = 8;
const static std::map<int32_t, string> TENSOR_INDEX_LIST = {{1, "accum"}, {2, "linear"}, {3, "grad"}};
const static std::map<int32_t, string> SCALAR_INDEX_LIST = {{4, "lr"}, {5, "l1"}, {6, "l2"}, {7, "lr_power"}};

namespace {
ge::graphStatus ReportTilingFailure(const char* opName, const std::string& reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    return ge::GRAPH_FAILED;
}
} // namespace

ge::graphStatus ApplyFtrlRegbaseTiling::CheckScalarShape(int32_t inputIdx)
{
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_IF(inputShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(tilingContext_->GetNodeName(), SCALAR_INDEX_LIST.at(inputIdx).c_str()),
                return ge::GRAPH_FAILED);
    auto storageShape = inputShape->GetStorageShape();
    std::string paramName = SCALAR_INDEX_LIST.at(inputIdx);
    OP_CHECK_IF((!storageShape.IsScalar() && storageShape.GetShapeSize() != 1),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext_->GetNodeName(), paramName.c_str(),
                                                      Ops::Base::ToString(storageShape).c_str(),
                                                      "The param must be a scalar(0D) or have shape size 1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyFtrlRegbaseTiling::CheckSameShape(int32_t inputIdx, const gert::Shape& input0Shape)
{
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_IF(inputShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(tilingContext_->GetNodeName(), TENSOR_INDEX_LIST.at(inputIdx).c_str()),
                return ge::GRAPH_FAILED);
    auto curStorageShape = inputShape->GetStorageShape();
    OP_CHECK_IF(curStorageShape != input0Shape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    tilingContext_->GetNodeName(), (string("var and ") + TENSOR_INDEX_LIST.at(inputIdx)).c_str(),
                    (Ops::Base::ToString(input0Shape) + " and " + Ops::Base::ToString(curStorageShape)).c_str(),
                    "Set accum, linear and grad to the same shape as var"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyFtrlRegbaseTiling::CheckSameDtype(int32_t inputIdx, const ge::DataType& input0Dtype)
{
    auto inputDesc = tilingContext_->GetInputDesc(inputIdx);
    const auto& paramName = inputIdx >= SCALAR_INPUT_LR_INDEX ? SCALAR_INDEX_LIST.at(inputIdx) :
                                                                TENSOR_INDEX_LIST.at(inputIdx);
    OP_CHECK_IF(inputDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(tilingContext_->GetNodeName(), paramName.c_str()),
                return ge::GRAPH_FAILED);
    auto curDtype = inputDesc->GetDataType();
    OP_CHECK_IF(
        curDtype != input0Dtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext_->GetNodeName(), (string("var and ") + paramName).c_str(),
                                               (ge::TypeUtils::DataTypeToSerialString(input0Dtype) + " and " +
                                                ge::TypeUtils::DataTypeToSerialString(curDtype))
                                                   .c_str(),
                                               "Set all input dtypes to the same dtype as var"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyFtrlRegbaseTiling::CheckShapeAndType()
{
    auto inputShape = tilingContext_->GetInputShape(0);
    OP_CHECK_IF(inputShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(tilingContext_->GetNodeName(), "var"),
                return ge::GRAPH_FAILED);
    auto inputStorageShape = inputShape->GetStorageShape();

    auto inputVarDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_IF(inputVarDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(tilingContext_->GetNodeName(), "var"),
                return ge::GRAPH_FAILED);
    auto inputDtype = inputVarDesc->GetDataType();
    for (const auto& pair : SCALAR_INDEX_LIST) {
        auto ret = CheckScalarShape(pair.first);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = CheckSameDtype(pair.first, inputDtype);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }
    for (const auto& pair : TENSOR_INDEX_LIST) {
        auto ret = CheckSameShape(pair.first, inputStorageShape);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ret = CheckSameDtype(pair.first, inputDtype);
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyFtrlRegbaseTiling::DoElewiseTiling()
{
    ElewiseBaseTiling eleBaseTiling(tilingContext_);
    auto varDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_IF(varDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(tilingContext_->GetNodeName(), "var"),
                return ge::GRAPH_FAILED);
    ge::DataType varDType = varDesc->GetDataType();
    ge::graphStatus ret = ge::GRAPH_FAILED;
    if (varDType == ge::DT_FLOAT) {
        ret = eleBaseTiling.DoTiling<ApplyFtrlOp::ApplyFtrlDag<float, float>::OpDag>(tiling_->elewiseTiling);
    } else if (varDType == ge::DT_FLOAT16) {
        ret = eleBaseTiling.DoTiling<ApplyFtrlOp::ApplyFtrlDag<half, float>::OpDag>(tiling_->elewiseTiling);
    } else if (varDType == ge::DT_BF16) {
        ret = eleBaseTiling.DoTiling<ApplyFtrlOp::ApplyFtrlDag<bfloat16_t, float>::OpDag>(tiling_->elewiseTiling);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext_->GetNodeName(), "var",
                                  ge::TypeUtils::DataTypeToSerialString(varDType).c_str(), "fp32, fp16 or bf16");
        ret = ge::GRAPH_FAILED;
    }
    return ret;
}

ge::graphStatus ApplyFtrlRegbaseTiling::SetTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Enter SetTilingData");
    size_t* currentWorkspace = tilingContext_->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ReportTilingFailure(
            tilingContext_->GetNodeName(),
            "SetTilingData: GetWorkspaceSizes(1) returned nullptr; required output-slot count=1. "
            "Check the TilingContext workspace container and its capacity.");
    }
    currentWorkspace[0] = SYS_WORKSPACE;
    tilingKey = GET_TPL_TILING_KEY(tiling_->elewiseTiling.scheMode);
    tilingContext_->SetTilingKey(tilingKey);
    uint32_t numBlocks = static_cast<uint32_t>(tiling_->elewiseTiling.blockNum);
    if (numBlocks == 0) {
        const auto& result = tiling_->elewiseTiling;
        return ReportTilingFailure(
            tilingContext_->GetNodeName(),
            "SetTilingData: ElewiseBaseTiling returned GRAPH_SUCCESS but produced invalid tiling data; blockNum=" +
                std::to_string(result.blockNum) + "; expected blockNum>0, dim0=" + std::to_string(result.dim0) +
                ", coreNum=" + std::to_string(result.coreNum) + ", blockFormer=" + std::to_string(result.blockFormer) +
                ", tilingKey=" + std::to_string(tilingKey) +
                ", shape=" + Ops::Base::ToString(tilingContext_->GetInputShape(0)->GetStorageShape()) +
                ", dtype=" + ge::TypeUtils::DataTypeToSerialString(tilingContext_->GetInputDesc(0)->GetDataType()) +
                ". Preserve input metadata and tiling logs for the opbase implementation owner.");
    }
    tilingContext_->SetBlockDim(numBlocks);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyFtrlRegbaseTiling::RunTiling()
{
    if (tilingContext_ == nullptr) {
        return ReportTilingFailure("ApplyFtrl", "RunTiling: TilingContext is nullptr. Check the GE tiling invocation.");
    }
    auto ret = CheckShapeAndType();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    auto rawTilingData = tilingContext_->GetRawTilingData();
    const auto requiredBytes = sizeof(ApplyFtrlRegbaseTilingData);
    if (rawTilingData == nullptr) {
        return ReportTilingFailure(
            tilingContext_->GetNodeName(),
            "RunTiling: raw tiling buffer is nullptr; requiredBytes=" + std::to_string(requiredBytes) +
                ". Check the GE TilingContext output buffer allocation.");
    }
    const auto actualBytes = rawTilingData->GetCapacity();
    if (actualBytes < requiredBytes) {
        return ReportTilingFailure(
            tilingContext_->GetNodeName(),
            "RunTiling: tiling buffer capacity is insufficient; requiredBytes=" + std::to_string(requiredBytes) +
                ", actualBytes=" + std::to_string(actualBytes) + ". Check the registered maximum tiling data size.");
    }
    tiling_ = tilingContext_->GetTilingData<ApplyFtrlRegbaseTilingData>();
    ret = DoElewiseTiling();
    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(tilingContext_, "DoElewiseTiling failed, rc=%u; propagate the lower-layer error", ret),
        return ret);
    return SetTilingData();
}

ge::graphStatus Tiling4ApplyFtrl(gert::TilingContext* context)
{
    auto nodeName = context;
    OP_LOGD(nodeName, "Tiling4ApplyFtrl running begin");
    ApplyFtrlRegbaseTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForApplyFtrl(gert::TilingParseContext* context)
{
    auto nodeName = context;
    OP_LOGD(nodeName, "TilingPrepareForApplyFtrl running begin");
    return ge::GRAPH_SUCCESS;
}

struct ApplyFtrlCompileInfo {};

IMPL_OP_OPTILING(ApplyFtrl).Tiling(Tiling4ApplyFtrl).TilingParse<ApplyFtrlCompileInfo>(TilingPrepareForApplyFtrl);

} // namespace optiling
