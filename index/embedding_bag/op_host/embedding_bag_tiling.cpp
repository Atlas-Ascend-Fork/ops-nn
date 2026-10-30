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
 * \file embedding_bag.cc
 * \brief
 */

#include "register/op_impl_registry.h"
#include <string>
#include <vector>
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_util.h"
#include "embedding_bag_tiling.h"
#include "embedding_bag_regbase_tiling.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {

namespace {
ge::graphStatus ReportTilingFailure(const char* opName, const std::string& reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {
        opName, "Operator tiling execution failed. Preserve the execution logs and contact operator support."};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    return ge::GRAPH_FAILED;
}
} // namespace

constexpr uint32_t WEIGHT_INPUT_INDEX = 0;
constexpr uint32_t INDICES_INPUT_INDEX = 1;
constexpr uint32_t OFFSETS_INPUT_INDEX = 2;
constexpr uint32_t PER_SAMPLER_WEIGHTS_INPUT_INDEX = 3;
constexpr uint32_t INDICES_MAX_MOVE_LENGTH = 1024;
constexpr uint32_t CHECK_TWO_DIM = 2;
constexpr uint32_t CHECK_ONE_DIM = 1;
constexpr uint32_t CHECK_ZERO_DIM = 0;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t MODE_INDEX = 0;
constexpr uint32_t INCLUDE_LAST_OFFSET_INDEX = 3;
constexpr uint32_t PADDING_IDX_INDEX = 4;
constexpr uint32_t ONE_TIMES = 1;
constexpr uint32_t DOUBLE_TIMES = 2;
constexpr uint32_t THREE_TIMES = 3;
constexpr uint32_t FOUR_TIMES = 4;
constexpr uint32_t DOUBLE_BYTE = 4;
constexpr uint32_t FOUR_BYTE = 4;
constexpr uint32_t EIGHT_BYTE = 8;
constexpr uint32_t RESERVED_UB = 15 * 1024;
constexpr uint32_t MAX_DIVIDE_UB_NUM = 3;
constexpr uint32_t OTHER_DIVIDE_UB_NUM = 3;
constexpr uint32_t MODE_SUM = 0;
constexpr uint32_t MODE_MEAN = 1;
constexpr uint32_t MODE_MAX = 2;
constexpr uint32_t TILINGKEY_FP32 = 1;
constexpr uint32_t TILINGKEY_FP16 = 2;
constexpr uint32_t TILINGKEY_BF16 = 3;

const std::string SUM = "sum";
const std::string MEAN = "mean";
const std::string MAX = "max";

class EmbeddingBagTiling {
public:
    explicit EmbeddingBagTiling(gert::TilingContext* context) : tilingContext_(context) {}
    ge::graphStatus Init();
    void GetUsedCore();
    void getTilingKeyAndComputeRepTime(ge::DataType weightDtype, std::string mode);
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    template <typename T1, typename T2>
    inline auto CeilDiv(T1 a, T2 b) -> T1 const
    {
        return T1(b == 0 ? a : (a + b - 1) / b);
    }
    template <typename T1, typename T2>
    inline auto FloorDiv(T1 a, T2 b) -> T1 const
    {
        return T1((a) / (b));
    }
    template <typename T1, typename T2>
    inline auto CeilAlign(T1 a, T2 b) -> T1 const
    {
        return T1(b == 0 ? a : (a + b - 1) / b * b);
    }
    template <typename T1, typename T2>
    inline auto FloorAlign(T1 a, T2 b) -> T1 const
    {
        return T1(b == 0 ? a : (a) / b * b);
    }

private:
    EmbeddingBagTilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;

    int64_t numEmbeddings_ = 0;
    int64_t numOffset_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t usedCoreNum_ = 0;
    int64_t paddingIdx_ = 0;
    int64_t formerOffsetNum_ = 0;
    int64_t tailOffsetNum_ = 0;
    int64_t computeRepTime_ = 0;
    int64_t indicesMaxMoveLength_ = 0;
    int64_t numIndices_ = 0;
    int64_t mode_ = 0;
    int64_t tilingKey_ = 0;
    uint64_t ubSizePlatForm_ = 0;
    int64_t hasPerSampleWeights_ = 0;
};

ge::graphStatus EmbeddingBagTiling::Init()
{
    auto compileInfo = tilingContext_->GetCompileInfo<EmbeddingBagCompileInfo>();
    if (compileInfo == nullptr) {
        return ReportTilingFailure(tilingContext_->GetNodeName(), "compileInfo is null");
    }
    indicesMaxMoveLength_ = INDICES_MAX_MOVE_LENGTH;

    const gert::StorageShape* weightShape = tilingContext_->GetInputShape(WEIGHT_INPUT_INDEX);
    if (weightShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(tilingContext_->GetNodeName(), "weight");
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* indicesShape = tilingContext_->GetInputShape(INDICES_INPUT_INDEX);
    if (indicesShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(tilingContext_->GetNodeName(), "indices");
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* offsetsShape = tilingContext_->GetInputShape(OFFSETS_INPUT_INDEX);
    if (offsetsShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(tilingContext_->GetNodeName(), "offsets");
        return ge::GRAPH_FAILED;
    }
    auto perSampleWeights = tilingContext_->GetOptionalInputTensor(PER_SAMPLER_WEIGHTS_INPUT_INDEX);
    if (weightShape->GetStorageShape().GetDimNum() != CHECK_TWO_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext_->GetNodeName(), "weight",
                                                 std::to_string(weightShape->GetStorageShape().GetDimNum()).c_str(),
                                                 "weight must be 2D");
        return ge::GRAPH_FAILED;
    }
    if (indicesShape->GetStorageShape().GetDimNum() != CHECK_ONE_DIM ||
        offsetsShape->GetStorageShape().GetDimNum() != CHECK_ONE_DIM) {
        std::string dims = std::to_string(indicesShape->GetStorageShape().GetDimNum()) + ", " +
                           std::to_string(offsetsShape->GetStorageShape().GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(tilingContext_->GetNodeName(), "indices, offsets", dims.c_str(),
                                                  "indices and offsets must be 1D");
        return ge::GRAPH_FAILED;
    }
    if (perSampleWeights != nullptr && perSampleWeights->GetShapeSize() != 0) {
        hasPerSampleWeights_ = 1;
    }
    numEmbeddings_ = weightShape->GetStorageShape().GetDim(DIM_INDEX1);
    numIndices_ = indicesShape->GetStorageShape().GetDim(DIM_INDEX0);
    numOffset_ = offsetsShape->GetStorageShape().GetDim(DIM_INDEX0);
    coreNum_ = compileInfo->totalCoreNum;
    ubSizePlatForm_ = compileInfo->ubSizePlatForm;

    auto attrs = tilingContext_->GetAttrs();
    if (attrs == nullptr) {
        return ReportTilingFailure(tilingContext_->GetNodeName(), "Get attrs failed");
    }
    const std::string mode = std::string(attrs->GetAttrPointer<char>(MODE_INDEX));
    OP_LOGW(tilingContext_, "%s mode is not supported", mode.c_str());
    if (mode == MEAN) {
        mode_ = MODE_MEAN;
    } else if (mode == SUM) {
        mode_ = MODE_SUM;
    } else if (mode == MAX) {
        mode_ = MODE_MAX;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "mode", mode.c_str(),
                                              "mode should be one of [sum, mean, max]");
        return ge::GRAPH_FAILED;
    }
    bool includeLastOffset = *attrs->GetAttrPointer<bool>(INCLUDE_LAST_OFFSET_INDEX);
    if (includeLastOffset) {
        numOffset_ -= 1;
    }
    OP_CHECK_IF((numOffset_ == 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext_->GetNodeName(), "offsets",
                                                      std::to_string(numOffset_).c_str(),
                                                      "offset size must be >=2 when include_last_offset is true"),
                return ge::GRAPH_FAILED);
    paddingIdx_ = *attrs->GetAttrPointer<int64_t>(PADDING_IDX_INDEX);
    GetUsedCore();

    ge::DataType weightDatatype = tilingContext_->GetInputDesc(WEIGHT_INPUT_INDEX)->GetDataType();

    getTilingKeyAndComputeRepTime(weightDatatype, mode);

    if (computeRepTime_ <= 0) {
        return ReportTilingFailure(tilingContext_->GetNodeName(),
                                   "computeRepTime " + std::to_string(computeRepTime_) + " should be greater than 0");
    }

    size_t sysWorkspaceSize = compileInfo->sysWorkspaceSize;
    size_t* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

void EmbeddingBagTiling::GetUsedCore()
{
    if (this->numOffset_ <= this->coreNum_) {
        this->usedCoreNum_ = this->numOffset_;
        this->formerOffsetNum_ = 1;
        this->tailOffsetNum_ = 1;
        return;
    }
    if (this->coreNum_ != 0) {
        this->formerOffsetNum_ = FloorDiv(this->numOffset_, this->coreNum_);
    }
    this->usedCoreNum_ = this->coreNum_;
    this->tailOffsetNum_ = this->numOffset_ - this->formerOffsetNum_ * (this->usedCoreNum_ - 1);
}

void EmbeddingBagTiling::getTilingKeyAndComputeRepTime(ge::DataType weightDtype, std::string mode)
{
    auto resiveUbSize = ubSizePlatForm_ - RESERVED_UB - formerOffsetNum_ * sizeof(float) * DOUBLE_TIMES -
                        INDICES_MAX_MOVE_LENGTH * sizeof(float) * DOUBLE_TIMES;

    if (weightDtype == ge::DT_FLOAT) {
        tilingKey_ = TILINGKEY_FP32;
    } else if (weightDtype == ge::DT_FLOAT16) {
        tilingKey_ = TILINGKEY_FP16;
    } else {
        tilingKey_ = TILINGKEY_BF16;
    }
    if (mode == MAX) {
        computeRepTime_ = resiveUbSize / (MAX_DIVIDE_UB_NUM * sizeof(float));
    } else {
        computeRepTime_ = resiveUbSize / (OTHER_DIVIDE_UB_NUM * sizeof(float));
    }
}

ge::graphStatus EmbeddingBagTiling::SetKernelTiling()
{
    tilingContext_->SetBlockDim(usedCoreNum_);
    tilingData_.set_formerOffsetNum(formerOffsetNum_);
    tilingData_.set_tailOffsetNum(tailOffsetNum_);
    tilingData_.set_numEmbeddings(numEmbeddings_);
    tilingData_.set_computeRepTime(computeRepTime_);
    tilingData_.set_indicesMaxMoveLength(indicesMaxMoveLength_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_paddingIdx(paddingIdx_);
    tilingData_.set_numIndices(numIndices_);
    tilingData_.set_mode(mode_);
    tilingData_.set_hasPerSampleWeights(hasPerSampleWeights_);
    tilingData_.set_tilingKey(tilingKey_);

    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->SetTilingKey(tilingKey_);
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void EmbeddingBagTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_, "start print             ");
    OP_LOGD(tilingContext_, "usedCoreNum_:            %d", usedCoreNum_);
    OP_LOGD(tilingContext_, "formerOffsetNum_:       %ld", formerOffsetNum_);
    OP_LOGD(tilingContext_, "tailOffsetNum_:         %ld", tailOffsetNum_);
    OP_LOGD(tilingContext_, "numEmbeddings_:         %ld", numEmbeddings_);
    OP_LOGD(tilingContext_, "computeRepTime_:        %ld", computeRepTime_);
    OP_LOGD(tilingContext_, "indicesMaxMoveLength_:  %ld", indicesMaxMoveLength_);
    OP_LOGD(tilingContext_, "paddingIdx_:            %ld", paddingIdx_);
    OP_LOGD(tilingContext_, "numIndices_:            %ld", numIndices_);
    OP_LOGD(tilingContext_, "mode_:                  %ld", mode_);
    OP_LOGD(tilingContext_, "hasPerSampleWeights_:   %ld", hasPerSampleWeights_);
    OP_LOGD(tilingContext_, "tilingKey_:             %ld", tilingKey_);
}

ge::graphStatus TilingEmbeddingBag(gert::TilingContext* context)
{
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context)) {
        return EmbeddingBagTilingForRegBase(context);
    }
    EmbeddingBagTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "tiling init fail");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.SetKernelTiling();
}

ge::graphStatus TilingPrepareForEmbeddingBag(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling Prepare For EmbeddingBag start");
    auto compileInfo = context->GetCompiledInfo<EmbeddingBagCompileInfo>();
    if (compileInfo == nullptr) {
        return ReportTilingFailure(context->GetNodeName(), "compileInfo is null");
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ReportTilingFailure(context->GetNodeName(), "platformInfo is null");
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    if (compileInfo->ubSizePlatForm <= 0) {
        return ReportTilingFailure(context->GetNodeName(), "Failed to get ub size");
    }
    OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu", compileInfo->ubSizePlatForm);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (compileInfo->sysWorkspaceSize < 0) {
        return ReportTilingFailure(context->GetNodeName(), "sysWorkspaceSize should be greater than or equal to zero");
    }
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context->GetNodeName(), "total ub size is %lu", totalUbSize);
    OP_LOGD(context->GetNodeName(), "Tiling prepare for EmbeddingBag end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EmbeddingBag)
    .Tiling(TilingEmbeddingBag)
    .TilingParse<EmbeddingBagCompileInfo>(TilingPrepareForEmbeddingBag);
} // namespace optiling
