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
 * \file fused_weight_quant_mat_mul_checker.cpp
 * \brief dtype/shape 约束（A16W8 PerChannel + relu，与底座能力对齐）：
 *        1. x 仅 FP16，weight 仅 INT8，antiquant_scale 仅 UINT64（FP32 为 A16W4 通路，未放开）
 *        2. y ∈ {FP16, INT8}——**合并编码契约**：y INT8 时
 *           antiquant_scale 的 uint64 编码为"反量化 × 输出量化"合并 scale（s_w*s_q，由融合
 *           pass 常量折叠生成，与底座 wqbmm_v2 的 y=INT8 契约一致）；y FP16 时为纯反量化 scale
 *        3. bias 仅 FP16/INT32
 *        4. antiquant 仅 PerChannel（scale shape (n)/(1,n)/(n,1)，group_size == 0）
 *        5. antiquant_offset/quant_scale/quant_offset 必须为空（quant 输入为后续独立通路
 *           的预留槽位，当前不支持）
 *        6. x/weight 仅 ND 格式
 *        7. fused_op_type 仅 "relu"（输出量化由 y dtype 承载，不占 fused_op_type）
 */

#include <cstring>
#include "fused_weight_quant_mat_mul_checker.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "matmul/fused_weight_quant_mat_mul/common/fused_weight_quant_mat_mul_constants.h"

namespace {
constexpr size_t MM_SHAPE_LEN_ND = 2UL;
constexpr size_t MIN_SHAPE_LEN_ND = 2UL;
constexpr size_t MAX_SHAPE_LEN_ND = 6UL;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
} // namespace

namespace optiling {
namespace fused_weight_quant_mat_mul {

ge::graphStatus FusedWeightQuantMatMulChecker::CheckContext()
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
    inputParams_.opName = nodeName != nullptr && nodeName[0] != '\0' ? nodeName : "FusedWeightQuantMatMul";
    OPS_LOG_I(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    return ge::GRAPH_SUCCESS;
}

bool FusedWeightQuantMatMulChecker::CheckDtype()
{
    size_t idx = 0;
    inputParams_.aDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.antiQuantScaleDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_FLOAT16,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input x dtype must be FLOAT16, but is [%s]",
                                                    ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),
                    return false);

    OP_TILING_CHECK(inputParams_.bDtype != ge::DT_INT8,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input weight dtype must be INT8, but is [%s]",
                                                    ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                    return false);

    OP_TILING_CHECK(inputParams_.antiQuantScaleDtype != ge::DT_UINT64,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        inputParams_.opName, "Input antiquant scale dtype must be UINT64, but is [%s]",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.antiQuantScaleDtype).c_str()),
                    return false);

    OP_TILING_CHECK(
        inputParams_.cDtype != ge::DT_FLOAT16 && inputParams_.cDtype != ge::DT_INT8,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Output y dtype must be FLOAT16/INT8, but is [%s]",
                                        ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str()),
        return false);
    // 合并编码契约：y INT8 时 antiquant_scale 为"反量化 × 输出量化"合并编码（pass 折叠生成），
    // 仅由融合 pass 产生；checker 不区分编码来源（与底座 y=INT8 行为一致）

    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    if (biasDesc != nullptr) {
        inputParams_.biasDtype = biasDesc->GetDataType();
        OP_TILING_CHECK(
            inputParams_.biasDtype != ge::DT_FLOAT16 && inputParams_.biasDtype != ge::DT_INT32,
            VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Input bias dtype must be FLOAT16/INT32, but is [%s]",
                                            ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str()),
            return false);
    }
    return true;
}

bool FusedWeightQuantMatMulChecker::CheckAttr()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto transposeX = attrs->GetAttrPointer<bool>(TRANSPOSE_X_ATTR_INDEX);
    auto transposeWeight = attrs->GetAttrPointer<bool>(TRANSPOSE_WEIGHT_ATTR_INDEX);
    const int64_t* groupSizePtr = nullptr;
    if (attrs->GetAttrNum() > ANTIQUANT_GROUP_SIZE_ATTR_INDEX) {
        groupSizePtr = attrs->GetAttrPointer<int64_t>(ANTIQUANT_GROUP_SIZE_ATTR_INDEX);
    }
    const char* fusedOpType = nullptr;
    if (attrs->GetAttrNum() > FUSED_OP_TYPE_ATTR_INDEX) {
        fusedOpType = attrs->GetAttrPointer<char>(FUSED_OP_TYPE_ATTR_INDEX);
    }
    inputParams_.transA = transposeX != nullptr && *transposeX;
    inputParams_.transB = transposeWeight != nullptr && *transposeWeight;

    if (groupSizePtr != nullptr) {
        OP_TILING_CHECK(*groupSizePtr != 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams_.opName, "Group size should be 0 (per-channel), but is [%ld]", *groupSizePtr),
                        return false);
        inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);
    }
    OP_TILING_CHECK(
        fusedOpType == nullptr || strcmp(fusedOpType, ops::fused_weight_quant_mat_mul::FUSED_OP_TYPE_RELU) != 0,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "fused_op_type only supports relu, but is [%s]",
                                        fusedOpType == nullptr ? "null" : fusedOpType),
        return false);
    return true;
}

bool FusedWeightQuantMatMulChecker::CheckInputShape(const gert::StorageShape* xShape,
                                                    const gert::StorageShape* weightShape)
{
    inputParams_.aFormat = GetInputStorageFormat(context_, 0);
    inputParams_.bFormat = GetInputStorageFormat(context_, 1);
    OP_TILING_CHECK(inputParams_.aFormat != ge::Format::FORMAT_ND && inputParams_.aFormat != ge::Format::FORMAT_NCHW,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Input x format should be ND, actual format is %s",
                                          ge::TypeUtils::FormatToSerialString(inputParams_.aFormat).c_str()),
                    return false);

    OP_TILING_CHECK(inputParams_.bFormat != ge::Format::FORMAT_ND && inputParams_.bFormat != ge::Format::FORMAT_NCHW,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Input weight format should be ND, actual format is %s",
                                          ge::TypeUtils::FormatToSerialString(inputParams_.bFormat).c_str()),
                    return false);

    size_t xOriDimNum = xShape->GetOriginShape().GetDimNum();
    size_t weightOriDimNum = weightShape->GetOriginShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t weightDimNum = weightShape->GetStorageShape().GetDimNum();

    OP_TILING_CHECK(
        xOriDimNum < MIN_SHAPE_LEN_ND || xOriDimNum > MAX_SHAPE_LEN_ND,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "OriginalShape of x should be in [%zu, %zu], but is [%zu]",
                                        MIN_SHAPE_LEN_ND, MAX_SHAPE_LEN_ND, xOriDimNum),
        return false);

    OP_TILING_CHECK(weightOriDimNum < MIN_SHAPE_LEN_ND || weightOriDimNum > MAX_SHAPE_LEN_ND,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                                                    "OriginalShape of weight should be in [%zu, %zu], but is [%zu]",
                                                    MIN_SHAPE_LEN_ND, MAX_SHAPE_LEN_ND, weightOriDimNum),
                    return false);

    OP_TILING_CHECK(
        xDimNum < MIN_SHAPE_LEN_ND || xDimNum > MAX_SHAPE_LEN_ND,
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "StorageShape of x should be in [%zu, %zu], but is [%zu]",
                                        MIN_SHAPE_LEN_ND, MAX_SHAPE_LEN_ND, xDimNum),
        return false);

    OP_TILING_CHECK(weightDimNum < MIN_SHAPE_LEN_ND || weightDimNum > MAX_SHAPE_LEN_ND,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                                                    "StorageShape of weight should be in [%zu, %zu], but is [%zu]",
                                                    MIN_SHAPE_LEN_ND, MAX_SHAPE_LEN_ND, weightDimNum),
                    return false);

    uint64_t weightLastDim = weightShape->GetOriginShape().GetDim(weightOriDimNum - LAST_FIRST_DIM_INDEX);
    uint64_t weightFirstDim = weightShape->GetOriginShape().GetDim(weightOriDimNum - LAST_SECOND_DIM_INDEX);
    uint64_t xFirstDim = xShape->GetOriginShape().GetDim(xOriDimNum - LAST_SECOND_DIM_INDEX);
    uint64_t xLastDim = xShape->GetOriginShape().GetDim(xOriDimNum - LAST_FIRST_DIM_INDEX);
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? xLastDim : xFirstDim);
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? xFirstDim : xLastDim);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? weightFirstDim : weightLastDim);

    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? weightLastDim : weightFirstDim);
    OP_TILING_CHECK(inputParams_.kSize != kBSize,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                                                    "K dim of x and weight must equal, but they are [%lu] and [%lu]",
                                                    inputParams_.kSize, kBSize),
                    return false);
    return true;
}

bool FusedWeightQuantMatMulChecker::CheckBatchInfo(const gert::Shape& xShape, const gert::Shape& weightShape) const
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

bool FusedWeightQuantMatMulChecker::CheckAntiQuantShape(const gert::StorageShape* antiQuantScaleShape,
                                                        const gert::StorageShape* antiQuantOffsetShape)
{
    size_t antiQuantScaleDimNum = antiQuantScaleShape->GetStorageShape().GetDimNum();
    size_t antiQuantScaleShapeSize = static_cast<size_t>(antiQuantScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(antiQuantScaleDimNum > MM_SHAPE_LEN_ND,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName,
                                                    "antiquant scale shape dim should not be more than 2, but is [%zu]",
                                                    antiQuantScaleDimNum),
                    return false);
    inputParams_.antiQuantType = QuantType::PER_CHANNEL;
    if (antiQuantScaleDimNum == MM_SHAPE_LEN_ND) {
        gert::Shape expectShape;
        if (inputParams_.transB) {
            expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
            expectShape.AppendDim(1);
        } else {
            expectShape.AppendDim(1);
            expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
        }
        const bool shapeMatch = expectShape == antiQuantScaleShape->GetStorageShape();
        OP_TILING_CHECK(!shapeMatch,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams_.opName, "Antiquant scale must have per-channel shape %s, but shape is %s",
                            Ops::Base::ToString(expectShape).c_str(),
                            Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str()),
                        return false);
    } else {
        // 一期仅 PerChannel：scale shape (1,) 为 PerTensor，拒绝
        OP_TILING_CHECK(antiQuantScaleShapeSize != inputParams_.nSize,
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            inputParams_.opName, "antiquant scale should be per-channel with size n[%lu], but is [%zu]",
                            inputParams_.nSize, antiQuantScaleShapeSize),
                        return false);
    }

    OP_TILING_CHECK(antiQuantOffsetShape != nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Antiquant offset should be nullptr"),
                    return false);
    return true;
}

bool FusedWeightQuantMatMulChecker::CheckShape()
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
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Not yet support empty tensor"), return false);
    OP_TILING_CHECK(!CheckInputShape(xShape, weightShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check input x and weight shape failed"),
                    return false);
    OP_TILING_CHECK(
        !CheckBatchInfo(xShape->GetOriginShape(), weightShape->GetOriginShape()),
        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Batch dimension should be equal when their value not 1."),
        return false);
    OP_TILING_CHECK(!CheckAntiQuantShape(antiQuantScaleShape, antiQuantOffsetShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check antiquant shape failed"), return false);
    // quant_scale/quant_offset 必须为空（y INT8 的输出量化由 pass 常量折叠进 antiquant_scale
    // 合并编码承载，不走独立 quant 输入；独立通路为后续阶段预留）
    OP_TILING_CHECK(quantScaleShape != nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Quant scale should be nullptr"),
                    return false);
    OP_TILING_CHECK(quantOffsetShape != nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Quant offset should be nullptr"),
                    return false);
    if (biasShape != nullptr) {
        OP_TILING_CHECK(biasShape->GetStorageShape().GetDimNum() != 1,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Bias dimension should equal to 1"),
                        return false);
        OP_TILING_CHECK(static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(0)) != inputParams_.nSize,
                        VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Bias size should be equal to nsize"),
                        return false);
    }
    return true;
}

ge::graphStatus FusedWeightQuantMatMulChecker::Check()
{
    auto ret = CheckContext();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    OP_TILING_CHECK(!CheckDtype(), VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check input dtype failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckAttr(), VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check attr failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckShape(), VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check shape failed"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace fused_weight_quant_mat_mul
} // namespace optiling
