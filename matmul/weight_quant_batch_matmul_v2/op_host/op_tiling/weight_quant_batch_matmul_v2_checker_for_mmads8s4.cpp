/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_checker_for_mmads8s4.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_checker_for_mmads8s4.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"

namespace {
constexpr size_t MM_SHAPE_LEN_ND = 2UL;
constexpr size_t MIN_SHAPE_LEN_ND = 2UL;
constexpr size_t MAX_SHAPE_LEN_ND = 6UL;
constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr uint64_t MIN_GROUP_SIZE = 32UL;
constexpr uint64_t MAX_INT32 = 2147483647UL;
constexpr uint32_t ANTIQUANT_OFFSET_INDEX = 3;
constexpr uint32_t QUANT_SCALE_INDEX = 4;
constexpr uint32_t QUANT_OFFSET_INDEX = 5;
constexpr uint32_t BIAS_INDEX = 6;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_BATCH_DIM_INDEX = 3;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr const char* DEFAULT_OP_NAME = "WeightQuantBatchMatmulV2";
} // namespace

namespace optiling {
ge::graphStatus WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckContext()
{
    // check Raw TilingData
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    // check the Required input and output desc
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    const char* nodeName = context_->GetNodeName();
    inputParams_.opName = nodeName != nullptr && nodeName[0] != '\0' ? nodeName : DEFAULT_OP_NAME;
    // OP_LOG_FULL
    OPS_LOG_I(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    return ge::GRAPH_SUCCESS;
}

/*
The function is check the dtype limit:
    1. A16W8 (weight ND int8): x fp16, weight int8, scale uint64/int64, y fp16/int8
    2. A16W4 pergroup (weight FRACTAL_NZ int4): x fp16, weight int4, scale/offset fp16, y fp16/int8
    3. Bias dtype should be int32/fp16
    4. INT8 output requires uint64 quant_scale; quant_offset is encoded into quant_scale by fusion passes
*/
bool WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckDtype()
{
    size_t idx = 0;
    inputParams_.aDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.antiQuantScaleDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_FLOAT16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "x", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                        "the dtype of x must be FLOAT16"),
                    return false);

    if (inputParams_.bDtype == ge::DT_INT4) {
        // A16W4 pergroup 场景（weight FRACTAL_NZ）：scale 为 fp16，输出量化由 quant_scale 承载
        OP_TILING_CHECK(inputParams_.antiQuantScaleDtype != ge::DT_FLOAT16,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "antiquantScale",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.antiQuantScaleDtype).c_str(),
                            "the dtype of antiquantScale must be FLOAT16 when weight dtype is INT4"),
                        return false);
        OP_TILING_CHECK(
            inputParams_.cDtype != ge::DT_FLOAT16 && inputParams_.cDtype != ge::DT_INT8,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "y",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                                  "the dtype of y must be FLOAT16 or INT8 when weight dtype is INT4"),
            return false);
        if (inputParams_.cDtype == ge::DT_INT8) {
            auto quantScaleDesc = context_->GetOptionalInputDesc(QUANT_SCALE_INDEX);
            OP_TILING_CHECK(
                quantScaleDesc == nullptr || quantScaleDesc->GetDataType() != ge::DT_UINT64,
                VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "INT8 output requires UINT64 quant_scale"),
                return false);
        }
        auto antiQuantOffsetDesc = context_->GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX);
        if (antiQuantOffsetDesc != nullptr) {
            auto antiQuantOffsetDtype = antiQuantOffsetDesc->GetDataType();
            OP_TILING_CHECK(antiQuantOffsetDtype != inputParams_.antiQuantScaleDtype,
                            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                                inputParams_.opName, "antiquantOffset",
                                ge::TypeUtils::DataTypeToSerialString(antiQuantOffsetDtype).c_str(),
                                "the dtype of antiquantOffset must be the same as antiquantScale"),
                            return false);
        }
    } else {
        OP_TILING_CHECK(
            inputParams_.bDtype != ge::DT_INT8,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "weight",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                                                  "the dtype of weight must be INT8 or INT4"),
            return false);

        OP_TILING_CHECK(
            inputParams_.antiQuantScaleDtype != ge::DT_UINT64 && inputParams_.antiQuantScaleDtype != ge::DT_INT64,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "antiquantScale",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.antiQuantScaleDtype).c_str(),
                "the dtype of antiquantScale must be UINT64 or INT64"),
            return false);
    }

    OP_TILING_CHECK(inputParams_.cDtype != ge::DT_FLOAT16 && inputParams_.cDtype != ge::DT_INT8,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "y", ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                        "the dtype of y must be FLOAT16 or INT8"),
                    return false);

    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    if (biasDesc != nullptr) {
        inputParams_.biasDtype = biasDesc->GetDataType();
        OP_TILING_CHECK(
            inputParams_.biasDtype != ge::DT_FLOAT16 && inputParams_.biasDtype != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                                                  "the dtype of bias must be FLOAT16 or INT32"),
            return false);
    }
    return true;
}

/*
The function is check the attr limit:
    1. A16W8 (weight ND int8): Group size should be 0
    2. A16W4 pergroup (weight FRACTAL_NZ int4): Group size should be (0, kSize) and aligned to 32
    3. innerPrecise only support 0 or 1
*/
bool WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckAttr()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto transposeX = attrs->GetAttrPointer<bool>(idx++);
    auto transposeWeight = attrs->GetAttrPointer<bool>(idx++);
    const int64_t* groupSizePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    idx++; // 跳过dtype属性
    const int64_t* innerPrecisePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        innerPrecisePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    inputParams_.transA = transposeX != nullptr && *transposeX;
    inputParams_.transB = transposeWeight != nullptr && *transposeWeight;

    if (groupSizePtr != nullptr) {
        inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);
    }
    if (inputParams_.bDtype == ge::DT_INT4) {
        inputParams_.bFormat = GetInputStorageFormat(context_, 1);
        const bool unsupportedTrans = (inputParams_.transA || inputParams_.transB) &&
                                      (inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ || inputParams_.groupSize == 0);
        OP_TILING_CHECK(unsupportedTrans,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "transposeX/transposeWeight", "true",
                            "only INT4 FRACTAL_NZ per-group mode supports transposed x and weight"),
                        return false);
        OP_TILING_CHECK(inputParams_.groupSize == 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSize",
                                                              std::to_string(inputParams_.groupSize).c_str(),
                                                              "the value of groupSize must be greater than 0 for INT4"),
                        return false);
    } else {
        OP_TILING_CHECK(inputParams_.groupSize != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSize",
                                                              std::to_string(inputParams_.groupSize).c_str(),
                                                              "the value of groupSize must be 0 for INT8"),
                        return false);
    }
    if (innerPrecisePtr != nullptr) {
        OP_TILING_CHECK((*innerPrecisePtr != 0) && (*innerPrecisePtr != 1),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "innerPrecise",
                                                              std::to_string(*innerPrecisePtr).c_str(),
                                                              "the value of innerPrecise must be 0 or 1"),
                        return false);
        inputParams_.innerPrecise = static_cast<uint64_t>(*innerPrecisePtr);
    }
    return true;
}

/*
The function is check the shape limit:
    1. the input x shape dims should be [2, 6](ND), weight shape dims should be [2, 6](ND)
    2. the k of x and weight must be same
    3. antiquant shape must be:
        per_channel: not trans_n: (1, n) or (n); trans_b: (n, 1) or (n)
        per_tensor: (1,)
    4. INT4 INT8 output supports per-tensor/per-channel quant scale; other quant inputs must be empty
*/
bool WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckShape()
{
    size_t idx = 0;
    auto xShape = context_->GetInputShape(idx++);
    auto weightShape = context_->GetInputShape(idx++);
    auto antiQuantScaleShape = context_->GetInputShape(idx++);
    auto antiQuantOffsetShape = context_->GetOptionalInputShape(idx++);
    auto quantScaleShape = context_->GetOptionalInputShape(idx++);
    auto quantOffsetShape = context_->GetOptionalInputShape(idx++);
    auto biasShape = context_->GetOptionalInputShape(idx++);
    auto outputShape = context_->GetOutputShape(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, antiQuantScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    // not yet support empty tensor for input
    OP_TILING_CHECK(xShape->GetStorageShape().GetShapeSize() == 0 ||
                        weightShape->GetStorageShape().GetShapeSize() == 0 ||
                        antiQuantScaleShape->GetStorageShape().GetShapeSize() == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "x, weight, antiquantScale", "empty",
                                                          "empty tensor is not supported"),
                    return false);
    // check x, weight
    inputParams_.aFormat = GetInputStorageFormat(context_, 0);
    inputParams_.bFormat = GetInputStorageFormat(context_, 1);
    OP_TILING_CHECK(!CheckInputShape(xShape, weightShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check input x and weight shape failed"),
                    return false);
    // check batch
    OP_TILING_CHECK(!CheckBatchInfo(xShape->GetOriginShape(), weightShape->GetOriginShape()),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "x, weight",
                                                          (Ops::Base::ToString(xShape->GetOriginShape()) + ", " +
                                                           Ops::Base::ToString(weightShape->GetOriginShape()))
                                                              .c_str(),
                                                          "the batch dims of x and weight must be equal or 1"),
                    return false);
    // check antiquant scale, antiquant offset
    OP_TILING_CHECK(!CheckAntiQuantShape(antiQuantScaleShape, antiQuantOffsetShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check antiquant shape failed"), return false);
    // check quant scale, quant offset
    const bool isInt4Quant = inputParams_.bDtype == ge::DT_INT4 && inputParams_.cDtype == ge::DT_INT8;
    if (isInt4Quant) {
        OP_TILING_CHECK(
            quantScaleShape == nullptr || quantScaleShape->GetStorageShape().GetDimNum() != 1,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "INT8 output requires a one-dimensional quant scale"),
            return false);
        const auto quantScaleSize = quantScaleShape->GetStorageShape().GetShapeSize();
        OP_TILING_CHECK(quantScaleSize != 1 && quantScaleSize != static_cast<int64_t>(inputParams_.nSize),
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Quant scale size should be 1 or n size"),
                        return false);
        inputParams_.quantType = quantScaleSize == 1 ? QuantType::PER_TENSOR : QuantType::PER_CHANNEL;
    } else {
        OP_TILING_CHECK(quantScaleShape != nullptr,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Quant scale should be nullptr"),
                        return false);
    }
    OP_TILING_CHECK(quantOffsetShape != nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Quant offset should be nullptr"),
                    return false);
    // check bias
    if (biasShape != nullptr) {
        OP_TILING_CHECK(
            biasShape->GetStorageShape().GetDimNum() != 1,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "bias",
                                                  std::to_string(biasShape->GetStorageShape().GetDimNum()).c_str(),
                                                  "the dim num of bias must be 1"),
            return false);
        OP_TILING_CHECK(static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(0)) != inputParams_.nSize,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "bias", std::to_string(biasShape->GetStorageShape().GetDim(0)).c_str(),
                            ("the size of bias must be equal to the n size of weight, actual n size is " +
                             std::to_string(inputParams_.nSize))
                                .c_str()),
                        return false);
    }
    // check dim value
    OP_TILING_CHECK(
        inputParams_.groupSize >= inputParams_.kSize || inputParams_.groupSize % MIN_GROUP_SIZE != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            inputParams_.opName, "groupSize", std::to_string(inputParams_.groupSize).c_str(),
            ("groupSize must be less than kSize " + std::to_string(inputParams_.kSize) + " and aligned to 32").c_str()),
        return false);
    return true;
}

bool WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckInputShape(const gert::StorageShape* xShape,
                                                               const gert::StorageShape* weightShape)
{
    OP_TILING_CHECK(inputParams_.aFormat != ge::Format::FORMAT_ND && inputParams_.aFormat != ge::Format::FORMAT_NCHW,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Input x format should be ND , actual format is %s",
                                          ge::TypeUtils::FormatToSerialString(inputParams_.aFormat).c_str()),
                    return false);

    // A16W4 pergroup 场景 weight 为 FRACTAL_NZ（storage 4D [N/16, K/16, 16, 16]，origin 仍为 (K, N)）
    if (inputParams_.bDtype == ge::DT_INT4) {
        OP_TILING_CHECK(inputParams_.bFormat != ge::Format::FORMAT_FRACTAL_NZ,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "Input weight format should be FRACTAL_NZ when dtype is INT4, "
                                              "actual format is %s",
                                              ge::TypeUtils::FormatToSerialString(inputParams_.bFormat).c_str()),
                        return false);
    } else {
        OP_TILING_CHECK(
            inputParams_.bFormat != ge::Format::FORMAT_ND && inputParams_.bFormat != ge::Format::FORMAT_NCHW,
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "Input weight format should be ND , actual format is %s",
                                  ge::TypeUtils::FormatToSerialString(inputParams_.bFormat).c_str()),
            return false);
    }

    size_t xOriDimNum = xShape->GetOriginShape().GetDimNum();
    size_t weightOriDimNum = weightShape->GetOriginShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t weightDimNum = weightShape->GetStorageShape().GetDimNum();

    OP_TILING_CHECK(
        xOriDimNum < MIN_SHAPE_LEN_ND || xOriDimNum > MAX_SHAPE_LEN_ND,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "x", std::to_string(xOriDimNum).c_str(),
                                              "the dim num of the origin shape of x must be in range [2, 6]"),
        return false);

    OP_TILING_CHECK(
        weightOriDimNum < MIN_SHAPE_LEN_ND || weightOriDimNum > MAX_SHAPE_LEN_ND,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "weight", std::to_string(weightOriDimNum).c_str(),
                                              "the dim num of the origin shape of weight must be in range [2, 6]"),
        return false);

    OP_TILING_CHECK(
        xDimNum < MIN_SHAPE_LEN_ND || xDimNum > MAX_SHAPE_LEN_ND,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "x", std::to_string(xDimNum).c_str(),
                                              "the dim num of the storage shape of x must be in range [2, 6]"),
        return false);

    OP_TILING_CHECK(
        weightDimNum < MIN_SHAPE_LEN_ND || weightDimNum > MAX_SHAPE_LEN_ND,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "weight", std::to_string(weightDimNum).c_str(),
                                              "the dim num of the storage shape of weight must be in range [2, 6]"),
        return false);

    auto xShapeLen = xShape->GetOriginShape().GetDimNum();
    auto weightShapeLen = weightShape->GetOriginShape().GetDimNum();
    uint64_t weightLastDim = weightShape->GetOriginShape().GetDim(weightShapeLen - LAST_FIRST_DIM_INDEX);
    uint64_t weightFirstDim = weightShape->GetOriginShape().GetDim(weightShapeLen - LAST_SECOND_DIM_INDEX);
    uint64_t xFirstDim = xShape->GetOriginShape().GetDim(xShapeLen - LAST_SECOND_DIM_INDEX);
    uint64_t xLastDim = xShape->GetOriginShape().GetDim(xShapeLen - LAST_FIRST_DIM_INDEX);
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? xLastDim : xFirstDim);
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? xFirstDim : xLastDim);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? weightFirstDim : weightLastDim);

    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? weightLastDim : weightFirstDim);
    OP_TILING_CHECK(inputParams_.kSize != kBSize,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        inputParams_.opName, "x, weight",
                        (std::to_string(inputParams_.kSize) + ", " + std::to_string(kBSize)).c_str(),
                        "the k dim of x and weight must be equal"),
                    return false);
    return true;
}

bool WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckBatchInfo(const gert::Shape& xShape,
                                                              const gert::Shape& weightShape) const
{
    auto x1DimNum = xShape.GetDimNum();
    auto x2DimNum = weightShape.GetDimNum();
    auto outDimNum = std::max(x1DimNum, x2DimNum);
    const gert::Shape& shapeLong = x1DimNum > x2DimNum ? xShape : weightShape;
    const gert::Shape& shapeShort = x1DimNum > x2DimNum ? weightShape : xShape;
    size_t validOffset = outDimNum - std::min(x1DimNum, x2DimNum);
    for (size_t i = 0; i < outDimNum - LAST_SECOND_DIM_INDEX; i++) {
        auto shortDim = i < validOffset ? 1 : shapeShort.GetDim(i - validOffset);
        auto longDim = shapeLong.GetDim(i);
        if (shortDim > 1 && longDim > 1 && shortDim != longDim) {
            return false;
        }
    }
    return true;
}

bool WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckAntiQuantShape(const gert::StorageShape* antiQuantScaleShape,
                                                                   const gert::StorageShape* antiQuantOffsetShape)
{
    size_t antiQuantScaleDimNum = antiQuantScaleShape->GetStorageShape().GetDimNum();
    size_t antiQuantScaleShapeSize = static_cast<size_t>(antiQuantScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(antiQuantScaleDimNum > MM_SHAPE_LEN_ND,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        inputParams_.opName, "antiquantScale", std::to_string(antiQuantScaleDimNum).c_str(),
                        "the dim num of antiquantScale can not be greater than 2"),
                    return false);
    if (antiQuantScaleShapeSize != 1) {
        if (antiQuantScaleDimNum == MM_SHAPE_LEN_ND) {
            gert::Shape expectShape;
            uint64_t kNum = inputParams_.groupSize > 0 ? ops::CeilDiv(inputParams_.kSize, inputParams_.groupSize) : 1;
            if (inputParams_.transB) {
                expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
                expectShape.AppendDim(static_cast<int64_t>(kNum));
            } else {
                expectShape.AppendDim(static_cast<int64_t>(kNum));
                expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
            }
            OP_TILING_CHECK(expectShape != antiQuantScaleShape->GetStorageShape(),
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                inputParams_.opName, "antiquantScale",
                                Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str(),
                                ("the shape of antiquantScale must be " + Ops::Base::ToString(expectShape)).c_str()),
                            return false);
            inputParams_.antiQuantType = inputParams_.groupSize > 0 ? QuantType::PER_GROUP : QuantType::PER_CHANNEL;
        } else {
            OP_TILING_CHECK(antiQuantScaleShapeSize != inputParams_.nSize,
                            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                                inputParams_.opName, "antiquantScale", std::to_string(antiQuantScaleShapeSize).c_str(),
                                ("the size of antiquantScale must be equal to the n size of weight in perchannel "
                                 "scenario, actual n size is " +
                                 std::to_string(inputParams_.nSize))
                                    .c_str()),
                            return false);
            inputParams_.antiQuantType = QuantType::PER_CHANNEL;
        }
    } else {
        inputParams_.antiQuantType = QuantType::PER_TENSOR;
    }

    if (antiQuantOffsetShape != nullptr) {
        OP_TILING_CHECK(inputParams_.bDtype != ge::DT_INT4 || inputParams_.antiQuantType != QuantType::PER_GROUP,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams_.opName, "Antiquant offset only supports INT4 FRACTAL_NZ per-group mode"),
                        return false);
        OP_TILING_CHECK(antiQuantScaleShape->GetStorageShape() != antiQuantOffsetShape->GetStorageShape(),
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams_.opName, "Antiquant scale %s and offset %s should have the same shape",
                            Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str(),
                            Ops::Base::ToString(antiQuantOffsetShape->GetStorageShape()).c_str()),
                        return false);
        inputParams_.hasAntiQuantOffset = true;
    }
    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2Checker4MmadS8S4::Check()
{
    auto ret = CheckContext();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // check the input and output dtype: x, weight, antiquant_scale, antiquant_offset, y.
    OP_TILING_CHECK(!CheckDtype(), VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check input dtype failed"),
                    return ge::GRAPH_FAILED);
    // check attrs: transA, transB, group_size, dtype, innerPrecise
    OP_TILING_CHECK(!CheckAttr(), VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check attr failed"),
                    return ge::GRAPH_FAILED);
    // check the input and output shape: all
    OP_TILING_CHECK(!CheckShape(), VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check shape failed"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
