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
 * \file index_check_tiling.cpp
 * \brief
 */
#include <string>
#include <vector>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info.h"
#include "index_check_tiling.h"
#include <algorithm>
#include "graph/utils/type_utils.h"

constexpr int64_t SCALE_SPACE = 20480;
constexpr int64_t MIN_UB_SIZE = 1024;
constexpr uint64_t BLOCK_BYTES = 32;
constexpr size_t MAX_DIM_NUM = 8;

namespace {
void ReportMissingTilingObject(const char* functionName, const char* objectName)
{
    const std::string reason = std::string(functionName) + ": framework-provided " + objectName +
                               " is nullptr; check framework context construction";
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {"IndexCheck", reason.c_str()};
    OP_LOGE_WITHOUT_REPORT("IndexCheck", "%s", reason.c_str());
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
}
} // namespace

namespace optiling {
class IndexCheckTiling {
public:
    explicit IndexCheckTiling(gert::TilingContext* context) : tilingContext_(context) {}
    ge::graphStatus Init();
    void CalcCoreAndBatch();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;

private:
    gert::TilingContext* tilingContext_ = nullptr;
    IndexCheckTilingData tilingData_;
    uint64_t workspaceSize_ = 0;
    ge::DataType indicesDtype_ = ge::DT_UNDEFINED;
    uint64_t ubSize_ = 0;
    uint64_t totalCoreNum_ = 0;
    uint64_t usedCoreNum_ = 0;
    uint64_t tensorId_ = 0;
    uint64_t maxBatchSize_ = 0;
    std::vector<uint64_t> tensorLens_;
};

ge::graphStatus IndexCheckTiling::Init()
{
    OP_LOGD(tilingContext_, "Tiling start.");
    if (tilingContext_ == nullptr) {
        ReportMissingTilingObject(__func__, "tilingContext");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = tilingContext_->GetCompileInfo<IndexCheckCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, ReportMissingTilingObject(__func__, "compileInfo"), return ge::GRAPH_FAILED);
    totalCoreNum_ = compileInfo->totalCoreNum;
    workspaceSize_ = compileInfo->workspaceSize;
    if (compileInfo->ubSizePlatform < static_cast<uint64_t>(SCALE_SPACE + MIN_UB_SIZE)) {
        const std::string reason = "Tiling requires platform UB bytes >= " + std::to_string(SCALE_SPACE + MIN_UB_SIZE) +
                                   "; actual=" + std::to_string(compileInfo->ubSizePlatform);
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {"IndexCheck", reason.c_str()};
        OP_LOGE_WITHOUT_REPORT("IndexCheck", "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
        return ge::GRAPH_FAILED;
    }
    ubSize_ = compileInfo->ubSizePlatform - static_cast<uint64_t>(SCALE_SPACE);
    OP_LOGD(tilingContext_, "totalCoreNum: %lu, ubSize: %lu, workspaceSize: %lu", totalCoreNum_, ubSize_,
            workspaceSize_);

    auto computeNodeInfoPtr = tilingContext_->GetComputeNodeInfo();
    OP_CHECK_IF(computeNodeInfoPtr == nullptr, ReportMissingTilingObject(__func__, "computeNodeInfoPtr"),
                return ge::GRAPH_FAILED);

    auto idxInstanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(1);
    OP_CHECK_IF(idxInstanceInfoPtr == nullptr, ReportMissingTilingObject(__func__, "idxInstanceInfoPtr"),
                return ge::GRAPH_FAILED);
    tensorId_ = idxInstanceInfoPtr->GetInstanceNum();
    OP_CHECK_IF(
        tensorId_ == 0,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("IndexCheck", "indices_list", "tensor count=0; expected [1, 8]"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tensorId_ > MAX_TENSOR_NUM,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    "IndexCheck", "indices_list", "tensor count=" + std::to_string(tensorId_) + "; expected [1, 8]"),
                return ge::GRAPH_FAILED);

    for (uint64_t i = 0; i < tensorId_; i++) {
        auto idxTensorDtypePtr = tilingContext_->GetDynamicInputDesc(1, i);
        OP_CHECK_IF(idxTensorDtypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT("IndexCheck", "indices_list"),
                    return ge::GRAPH_FAILED);
        auto idxTensorDtype = idxTensorDtypePtr->GetDataType();
        if (i == 0) {
            indicesDtype_ = idxTensorDtype;
            OP_CHECK_IF(
                indicesDtype_ != ge::DT_INT32 && indicesDtype_ != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE("IndexCheck", "indices_list",
                                          ge::TypeUtils::DataTypeToSerialString(indicesDtype_), "[DT_INT32, DT_INT64]"),
                return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(idxTensorDtype != indicesDtype_,
                        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                            "IndexCheck", "indices_list[0] and indices_list[" + std::to_string(i) + "]",
                            (ge::TypeUtils::DataTypeToSerialString(indicesDtype_) + " and " +
                             ge::TypeUtils::DataTypeToSerialString(idxTensorDtype))
                                .c_str(),
                            "indices tensors must have the same dtype"),
                        return ge::GRAPH_FAILED);
        }
    }

    auto boundsShapePtr = tilingContext_->GetInputShape(0);
    OP_CHECK_IF(boundsShapePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT("IndexCheck", "bounds"), return ge::GRAPH_FAILED);
    auto boundsShape = boundsShapePtr->GetStorageShape();
    OP_CHECK_IF(
        boundsShape.GetDimNum() != 1 || boundsShape.GetDim(0) != static_cast<int64_t>(tensorId_),
        OP_LOGE_FOR_INVALID_SHAPE("IndexCheck", "bounds", Ops::Base::ToString(boundsShape), std::to_string(tensorId_)),
        return ge::GRAPH_FAILED);

    tensorLens_.resize(MAX_TENSOR_NUM, 0);
    for (uint64_t i = 0; i < tensorId_; i++) {
        auto idxTensorShapePtr = tilingContext_->GetDynamicInputShape(1, i);
        OP_CHECK_IF(idxTensorShapePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT("IndexCheck", "indices_list"),
                    return ge::GRAPH_FAILED);
        auto idxTensorShape = idxTensorShapePtr->GetStorageShape();
        OP_CHECK_IF(idxTensorShape.GetDimNum() > MAX_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("IndexCheck", "indices_list[" + std::to_string(i) + "]",
                                                             std::to_string(idxTensorShape.GetDimNum()),
                                                             "rank must not exceed " + std::to_string(MAX_DIM_NUM)),
                    return ge::GRAPH_FAILED);
        uint64_t tensorLen = 1;
        for (uint32_t d = 0; d < idxTensorShape.GetDimNum(); d++) {
            tensorLen *= idxTensorShape.GetDim(d);
        }
        tensorLens_[i] = tensorLen;
        OP_LOGD(tilingContext_, "tensor[%lu] len: %lu", i, tensorLen);
    }

    CalcCoreAndBatch();

    OP_LOGD(tilingContext_, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void IndexCheckTiling::CalcCoreAndBatch()
{
    uint64_t maxTensorLen = 0;
    for (uint64_t i = 0; i < tensorId_; i++) {
        if (tensorLens_[i] > maxTensorLen) {
            maxTensorLen = tensorLens_[i];
        }
    }
    usedCoreNum_ = std::min(totalCoreNum_, maxTensorLen);
    if (usedCoreNum_ == 0) {
        usedCoreNum_ = 1;
    }

    uint64_t bytesPerElement;
    if (indicesDtype_ == ge::DT_INT64) {
        // int64 path is scalar-only (vector queues not instantiated): the
        // indices queue is the sole per-element buffer.
        bytesPerElement = sizeof(int64_t);
    } else {
        bytesPerElement = sizeof(int32_t) + sizeof(int32_t) + sizeof(float) + sizeof(float) + sizeof(int32_t) +
                          sizeof(int32_t);
    }
    uint64_t alignUnit = BLOCK_BYTES / sizeof(int32_t);
    uint64_t maxBatch = ubSize_ / bytesPerElement;
    maxBatchSize_ = (maxBatch / alignUnit) * alignUnit;
    if (maxBatchSize_ == 0) {
        maxBatchSize_ = alignUnit;
    }
}

ge::graphStatus IndexCheckTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext_, "set tiling data start.");
    tilingData_.params.set_usedCoreNum(usedCoreNum_);
    tilingData_.params.set_tensorId(tensorId_);
    tilingData_.params.set_maxBatchSize(maxBatchSize_);
    tilingData_.params.set_tensorLens(tensorLens_.data());

    auto rawTilingData = tilingContext_->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr, ReportMissingTilingObject(__func__, "rawTilingData"),
                return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());
    const auto blockDimRet = tilingContext_->SetBlockDim(usedCoreNum_);
    if (blockDimRet != ge::GRAPH_SUCCESS) {
        const std::string reason = "SetBlockDim failed during tiling; blockDim=" + std::to_string(usedCoreNum_) +
                                   "; rc=" + std::to_string(blockDimRet);
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {"IndexCheck", reason.c_str()};
        OP_LOGE_WITHOUT_REPORT("IndexCheck", "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
        return ge::GRAPH_FAILED;
    }
    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, ReportMissingTilingObject(__func__, "workspaces"), return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    TilingDataPrint();
    OP_LOGD(tilingContext_, "set tiling data end.");
    return ge::GRAPH_SUCCESS;
}

void IndexCheckTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_, "indicesDtype:           %d", static_cast<int32_t>(indicesDtype_));
    OP_LOGD(tilingContext_, "usedCoreNum:            %lu", usedCoreNum_);
    OP_LOGD(tilingContext_, "tensorId:               %lu", tensorId_);
    OP_LOGD(tilingContext_, "maxBatchSize:           %lu", maxBatchSize_);
    for (uint64_t i = 0; i < tensorId_; i++) {
        OP_LOGD(tilingContext_, "tensorLen[%lu]:         %lu", i, tensorLens_[i]);
    }
}

ge::graphStatus TilingIndexCheck(gert::TilingContext* context)
{
    if (context == nullptr) {
        ReportMissingTilingObject(__func__, "context");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "IndexCheck tiling begin");
    IndexCheckTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "IndexCheck tiling init failed; propagate the root cause.");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepareForIndexCheck(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        ReportMissingTilingObject(__func__, "context");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "TilingPrepareForIndexCheck start");
    auto compileInfo = context->GetCompiledInfo<IndexCheckCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, ReportMissingTilingObject(__func__, "compileInfo"), return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, ReportMissingTilingObject(__func__, "platformInfo"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    compileInfo->ubSizePlatform = ubSizePlatform;
    if (compileInfo->ubSizePlatform == 0) {
        const std::string reason = "TilingPrepareForIndexCheck could not obtain platform UB size; actual=0 bytes";
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {"IndexCheck", reason.c_str()};
        OP_LOGE_WITHOUT_REPORT("IndexCheck", "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "ub_size_platform: %lu", compileInfo->ubSizePlatform);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context, "total_ub_size: %lu", totalUbSize);
    OP_LOGD(context, "TilingPrepareForIndexCheck end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IndexCheck).Tiling(TilingIndexCheck).TilingParse<IndexCheckCompileInfo>(TilingPrepareForIndexCheck);
} // namespace optiling
