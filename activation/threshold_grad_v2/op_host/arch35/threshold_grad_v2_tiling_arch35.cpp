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
 * \file threshold_grad_v2_tiling_arch35.cpp
 * \brief threshold_grad_v2_tiling_arch35
 */

#include "threshold_grad_v2_tiling.h"
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/threshold_grad_v2_struct.h"
#include "../../op_kernel/arch35/threshold_grad_v2_dag.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;
using namespace ThresholdGradV2Op;

namespace optiling {
static constexpr uint64_t THRESHOLD_GRAD_V2_COMMON_TILING_PRIORITY = 0;
static constexpr uint32_t GRADIENTS_IDX = 0;
static constexpr uint32_t FEATURES_IDX = 1;
static constexpr uint32_t THRESHOLD_IDX = 2;

ge::graphStatus ThresholdGradV2Tiling::GetShapeAttrsInfo() { return ge::GRAPH_SUCCESS; }

bool ThresholdGradV2Tiling::IsCapable() { return true; }

ge::graphStatus ThresholdGradV2Tiling::DoOpTiling()
{
    auto gradientsDesc = context_->GetInputDesc(GRADIENTS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradientsDesc);
    ge::DataType gradientsDType = gradientsDesc->GetDataType();

    auto featuresDesc = context_->GetInputDesc(FEATURES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, featuresDesc);
    ge::DataType featuresDType = featuresDesc->GetDataType();

    auto thresholdDesc = context_->GetInputDesc(THRESHOLD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, thresholdDesc);
    ge::DataType thresholdDType = thresholdDesc->GetDataType();

    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputDtype = outputDesc->GetDataType();
    if ((gradientsDType != featuresDType) || (gradientsDType != thresholdDType) || (outputDtype != featuresDType)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "gradients, features, threshold, backprops",
            ge::TypeUtils::DataTypeToSerialString(gradientsDType) + ", " +
                ge::TypeUtils::DataTypeToSerialString(featuresDType) + ", " +
                ge::TypeUtils::DataTypeToSerialString(thresholdDType) + ", " +
                ge::TypeUtils::DataTypeToSerialString(outputDtype),
            "The dtypes of gradients, features, threshold and backprops must be the same");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus baseTilingResult = ge::GRAPH_FAILED;
    if (gradientsDType == ge::DT_FLOAT16) {
        BroadcastBaseTiling<ThresholdGradV2Dag<half>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<ThresholdGradV2Dag<half>::OpDag> failed"),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), THRESHOLD_GRAD_V2_TPL_FP16);
    } else if (gradientsDType == ge::DT_BF16) {
        BroadcastBaseTiling<ThresholdGradV2Dag<bfloat16_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<ThresholdGradV2Dag<bfloat16_t>::OpDag> failed"),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), THRESHOLD_GRAD_V2_TPL_BF16);
    } else if (gradientsDType == ge::DT_FLOAT) {
        BroadcastBaseTiling<ThresholdGradV2Dag<float>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<ThresholdGradV2Dag<float>::OpDag> failed"),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), THRESHOLD_GRAD_V2_TPL_FP32);
    } else if (gradientsDType == ge::DT_INT32) {
        BroadcastBaseTiling<ThresholdGradV2Dag<int32_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<ThresholdGradV2Int32Dag<int32_t>::OpDag> failed"),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), THRESHOLD_GRAD_V2_TPL_INT32);
    } else if (gradientsDType == ge::DT_INT8) {
        BroadcastBaseTiling<ThresholdGradV2Dag<int8_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(baseTilingResult == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<ThresholdGradV28BDag<int8_t>::OpDag> failed"),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), THRESHOLD_GRAD_V2_TPL_INT8);
    } else if (gradientsDType == ge::DT_UINT8) {
        BroadcastBaseTiling<ThresholdGradV2Dag<uint8_t>::OpDag> brcBaseTiling(context_);
        baseTilingResult = brcBaseTiling.DoTiling();
        OP_CHECK_IF(
            baseTilingResult == ge::GRAPH_FAILED,
            OP_LOGE(context_->GetNodeName(), "BroadcastBaseTiling<ThresholdGradV28BDag<uint8_t>::OpDag> failed"),
            return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode(), THRESHOLD_GRAD_V2_TPL_UINT8);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "gradients, features, threshold",
                                  ge::TypeUtils::DataTypeToSerialString(gradientsDType),
                                  "DT_FLOAT16, DT_BF16, DT_FLOAT,DT_INT32, DT_INT8, DT_UINT8");
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(context_->GetNodeName(), "[TilingData] tilingKey=%lu", tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ThresholdGradV2Tiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t ThresholdGradV2Tiling::GetTilingKey() const { return tilingKey; }

ge::graphStatus ThresholdGradV2Tiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus ThresholdGradV2Tiling::PostTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus ThresholdGradV2Tiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

ge::graphStatus TilingForThresholdGradV2(gert::TilingContext* context)
{
    OP_LOGD("ThresholdGradV2Tiling", "Enter TilingForThresholdGradV2");
    if (context == nullptr) {
        OP_LOGE("ThresholdGradV2Tiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = reinterpret_cast<const BroadcastCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for Arch35 architecture");
    OP_LOGD("ThresholdGradV2Tiling", "Enter new ThresholdGradV2Tiling");
    ThresholdGradV2Tiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForBroadcast(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<Ops::Base::BroadcastCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ThresholdGradV2)
    .Tiling(TilingForThresholdGradV2)
    .TilingParse<BroadcastCompileInfo>(TilingPrepareForBroadcast);
REGISTER_OPS_TILING_TEMPLATE(ThresholdGradV2, ThresholdGradV2Tiling, THRESHOLD_GRAD_V2_COMMON_TILING_PRIORITY);
} // namespace optiling
