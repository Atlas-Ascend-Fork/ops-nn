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
 * \file foreach_sign_regbase_tiling.cpp
 * \brief Sign-owned RegBase tiling implementation.
 */

#include "foreach_sign_regbase_tiling.h"

namespace optiling {

constexpr int64_t MAX_TENSOR_COUNT = 256;
constexpr size_t MAX_DIM_NUM = 8U;
constexpr int64_t DOUBLE_BUFFER = 2;

constexpr uint64_t TILING_KEY_HALF = 10001;
constexpr uint64_t TILING_KEY_FLOAT = 10002;
constexpr uint64_t TILING_KEY_INT = 10003;
constexpr uint64_t TILING_KEY_BF16 = 10004;
constexpr uint64_t TILING_KEY_INT8 = 10007;
constexpr uint64_t TILING_KEY_INT64 = 10010;

static bool IsSupportedInputDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_INT8 || dtype == ge::DT_INT32 ||
           dtype == ge::DT_INT64 || dtype == ge::DT_BF16;
}

ge::graphStatus ForeachSignRegbaseTiling::GetShapeAttrsInfo()
{
    auto computeNodeInfo = context_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, computeNodeInfo);
    auto xInstanceInfo = computeNodeInfo->GetInputInstanceInfo(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xInstanceInfo);
    int32_t tensorNum = static_cast<int32_t>(xInstanceInfo->GetInstanceNum());

    OP_CHECK_IF(
        tensorNum < 0 || tensorNum > MAX_TENSOR_COUNT,
        OP_LOGE_FOR_INVALID_TENSORNUM(context_->GetNodeName(), "x", static_cast<int64_t>(tensorNum),
                                      ("within the range [0, " + std::to_string(MAX_TENSOR_COUNT) + "]").c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(context_->GetComputeNodeOutputNum() != static_cast<uint64_t>(tensorNum),
                OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                    context_->GetNodeName(), "x and y",
                    (std::to_string(tensorNum) + " and " + std::to_string(context_->GetComputeNodeOutputNum())).c_str(),
                    "The tensor nums in {x, y} must be the same"),
                return ge::GRAPH_FAILED);

    ge::DataType listDtype = ge::DT_UNDEFINED;
    for (int32_t i = 0; i < tensorNum; i++) {
        auto inputDesc = context_->GetDynamicInputDesc(0, i);
        OP_CHECK_IF(inputDesc == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "inputDesc",
                                                             std::string("The input descriptor at tensor index ") +
                                                                 std::to_string(i) +
                                                                 " is unavailable; check Tiling context metadata"),
                    return ge::GRAPH_FAILED);
        auto inputShape = context_->GetDynamicInputShape(0, i);
        OP_CHECK_IF(inputShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "inputShape",
                                                             std::string("The input shape at tensor index ") +
                                                                 std::to_string(i) +
                                                                 " is unavailable; check Tiling context metadata"),
                    return ge::GRAPH_FAILED);
        const gert::Shape& inputStorageShape = inputShape->GetStorageShape();
        OP_CHECK_IF(inputStorageShape.GetDimNum() > MAX_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_->GetNodeName(), "x", std::to_string(inputStorageShape.GetDimNum()).c_str(),
                        ("The shape dim of the " + std::to_string(i) +
                         "th tensor in the tensor list should be less than or equal to " + std::to_string(MAX_DIM_NUM))
                            .c_str()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(inputStorageShape.GetShapeSize() < 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context_->GetNodeName(), "x", std::to_string(inputStorageShape.GetShapeSize()).c_str(),
                        ("The shape size of the " + std::to_string(i) +
                         "th tensor in the tensor list must be 0 or a positive number")
                            .c_str()),
                    return ge::GRAPH_FAILED);
        ge::DataType inputDtype = inputDesc->GetDataType();
        OP_CHECK_IF(!IsSupportedInputDtype(inputDtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(),
                        "supported dtypes are DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32, DT_INT64 and DT_BF16"),
                    return ge::GRAPH_FAILED);
        if (listDtype == ge::DT_UNDEFINED) {
            listDtype = inputDtype;
        } else {
            OP_CHECK_IF(inputDtype != listDtype,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(),
                            ("The dtypes of all tensors in the tensor list must be the same. "
                             "Currently, the dtype of the " +
                             std::to_string(i) + "th tensor is inconsistent with that (" +
                             ge::TypeUtils::DataTypeToSerialString(listDtype) + ") of other tensors")
                                .c_str()),
                        return ge::GRAPH_FAILED);
        }

        auto outputDesc = context_->GetOutputDesc(i);
        OP_CHECK_IF(outputDesc == nullptr, OP_LOGE(context_, "y[%d] descriptor is null", i), return ge::GRAPH_FAILED);
        auto outputShape = context_->GetOutputShape(i);
        OP_CHECK_IF(outputShape == nullptr, OP_LOGE(context_, "y[%d] shape is null", i), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            outputDesc->GetDataType() != listDtype,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(outputDesc->GetDataType()).c_str(),
                "The dtype of y must be the same as x"),
            return ge::GRAPH_FAILED);
        const gert::Shape& outputStorageShape = outputShape->GetStorageShape();
        OP_CHECK_IF(outputStorageShape != inputStorageShape,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "y", Ops::Base::ToString(outputStorageShape).c_str(),
                        ("The shape of the tensor y[" + std::to_string(i) +
                         "]:" + Ops::Base::ToString(outputStorageShape) + " must be same as that of the tensor x[" +
                         std::to_string(i) + "]:" + Ops::Base::ToString(inputStorageShape))
                            .c_str()),
                    return ge::GRAPH_FAILED);
    }

    totalTensorCount_ = static_cast<uint16_t>(tensorNum);
    totalDataCount_ = 0;
    dataType_ = listDtype;
    for (int32_t i = 0; i < tensorNum; i++) {
        tensorDataCountList_[i] = context_->GetDynamicInputShape(0, i)->GetStorageShape().GetShapeSize();
        totalDataCount_ += tensorDataCountList_[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachSignRegbaseTiling::DoOpTiling()
{
    OP_CHECK_IF(aicoreParams_.numBlocks == 0UL || aicoreParams_.ubSize == 0UL,
                OP_LOGE(context_->GetNodeName(), "invalid platform resources: coreNum=%lu, ubSize=%lu",
                        aicoreParams_.numBlocks, aicoreParams_.ubSize),
                return ge::GRAPH_FAILED);
    if (totalDataCount_ > 0) {
        if (ForeachRegbaseTiling::DoOpTiling() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        numBlocks_ = 1;
        tensorStartList_[0] = 0;
        tensorEndList_[0] = 0;
        tensorStartOffsetList_[0] = 0;
        tensorEndOffsetList_[0] = -1;
        foreachSoloTilingData_.set_tensorDataCountList(tensorDataCountList_);
        foreachSoloTilingData_.set_tensorStartList(tensorStartList_);
        foreachSoloTilingData_.set_tensorEndList(tensorEndList_);
        foreachSoloTilingData_.set_tensorStartOffsetList(tensorStartOffsetList_);
        foreachSoloTilingData_.set_tensorEndOffsetList(tensorEndOffsetList_);
    }

    int64_t sizePerElem = ge::GetSizeByDataType(dataType_);
    int64_t inputsTensorUbSize = 0;
    if (sizePerElem > 0) {
        const int64_t blockSize = Ops::Base::GetUbBlockSize(context_);
        const int64_t ubSizePerNumber = sizePerElem * DOUBLE_BUFFER + DOUBLE_BUFFER * sizePerElem;
        inputsTensorUbSize = aicoreParams_.ubSize / ubSizePerNumber;
        inputsTensorUbSize = Ops::Base::FloorAlign(inputsTensorUbSize * sizePerElem, blockSize) / sizePerElem;
    }
    foreachSoloTilingData_.set_inputsTensorUbSize(inputsTensorUbSize);
    return ge::GRAPH_SUCCESS;
}

uint64_t ForeachSignRegbaseTiling::GetTilingKey() const
{
    switch (dataType_) {
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF;
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT;
        case ge::DT_INT32:
            return TILING_KEY_INT;
        case ge::DT_BF16:
            return TILING_KEY_BF16;
        case ge::DT_INT8:
            return TILING_KEY_INT8;
        case ge::DT_INT64:
            return TILING_KEY_INT64;
        default:
            return 0;
    }
}

REGISTER_TILING_DATA_CLASS(ForeachSign_10001, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachSign_10002, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachSign_10003, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachSign_10004, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachSign_10007, ForeachSoloTilingDataRegbase)
REGISTER_TILING_DATA_CLASS(ForeachSign_10010, ForeachSoloTilingDataRegbase)

} // namespace optiling
