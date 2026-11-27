/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file normalize_batch_graph_infer.cpp
 * \brief Runtime 2.0 graph declaration verification for NormalizeBatch.
 */

#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "graph/utils/type_utils.h" // ge::TypeUtils::DataTypeToSerialString
#include "normalize_batch_proto.h"

namespace {
constexpr uint64_t EXPECTED_RANK = 3U; // (n, c, d)，与 op_host 侧 EXPECTED_RANK 基线对齐

// 形状字符串化（"(" + 维度 + ")"），供错误码 incorrectShapes 字段
std::string ShapeDimsToString(const ge::Shape& shape)
{
    std::string dims = "(";
    for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
        if (i != 0U) {
            dims += ", ";
        }
        dims += std::to_string(shape.GetDim(i));
    }
    return dims + ")";
}

bool DeclaredShapeCompatible(const ge::Shape& declared, const ge::Shape& input)
{
    if (declared.GetDimNum() == 0U || declared.GetDims() == ge::UNKNOWN_RANK || input.GetDims() == ge::UNKNOWN_RANK) {
        return true;
    }
    if (declared.GetDimNum() != input.GetDimNum()) {
        return false;
    }
    for (size_t index = 0U; index < input.GetDimNum(); ++index) {
        if (input.GetDim(index) >= 0 && declared.GetDim(index) >= 0 && input.GetDim(index) != declared.GetDim(index)) {
            return false;
        }
    }
    return true;
}
} // namespace

namespace ge {
// Validate graph declarations before inference replaces the output descriptors.
IMPLEMT_VERIFIER(NormalizeBatch, VerifyNormalizeBatch)
{
    const auto inputX = op.GetInputDesc("input_x");
    const auto seqLen = op.GetInputDesc("seq_len");
    const auto outputY = op.GetOutputDesc("output_y");
    const auto xShape = inputX.GetShape();

    if (xShape.GetDims() != UNKNOWN_RANK && xShape.GetDimNum() != EXPECTED_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            op.GetName().c_str(), "input_x", std::to_string(xShape.GetDimNum()).c_str(), "rank must be 3 (n, c, d)");
        return GRAPH_FAILED;
    }
    const auto seqShape = seqLen.GetShape();
    if (seqShape.GetDims() != UNKNOWN_RANK && seqShape.GetDimNum() != 1U) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(op.GetName().c_str(), "seq_len",
                                                 std::to_string(seqShape.GetDimNum()).c_str(), "rank must be 1");
        return GRAPH_FAILED;
    }
    if (seqLen.GetDataType() != DT_UNDEFINED && seqLen.GetDataType() != DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(op.GetName().c_str(), "seq_len",
                                              ge::TypeUtils::DataTypeToSerialString(seqLen.GetDataType()),
                                              "dtype must be int32");
        return GRAPH_FAILED;
    }
    if (!DeclaredShapeCompatible(outputY.GetShape(), xShape)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            op.GetName().c_str(), "input_x, output_y",
            ("input_x " + ShapeDimsToString(xShape) + ", declared output_y " + ShapeDimsToString(outputY.GetShape()))
                .c_str(),
            "declared output_y shape must be compatible with input_x shape");
        return GRAPH_FAILED;
    }
    if (outputY.GetDataType() != DT_UNDEFINED && outputY.GetDataType() != inputX.GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            op.GetName().c_str(), "input_x, output_y",
            ("input_x " + ge::TypeUtils::DataTypeToSerialString(inputX.GetDataType()) + ", declared output_y " +
             ge::TypeUtils::DataTypeToSerialString(outputY.GetDataType()))
                .c_str(),
            "declared output_y dtype must match input_x dtype");
        return GRAPH_FAILED;
    }
    if ((inputX.GetFormat() != FORMAT_ND && inputX.GetFormat() != FORMAT_RESERVED) ||
        (outputY.GetFormat() != FORMAT_ND && outputY.GetFormat() != FORMAT_RESERVED)) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(op.GetName().c_str(), "input_x, output_y",
                                               (std::to_string(static_cast<int64_t>(inputX.GetFormat())) + "," +
                                                std::to_string(static_cast<int64_t>(outputY.GetFormat())))
                                                   .c_str(),
                                               "both must use ND format");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

VERIFY_FUNC_REG(NormalizeBatch, VerifyNormalizeBatch);

IMPLEMT_COMMON_INFERFUNC(NormalizeBatchInferShape)
{
    TensorDesc tensordescOutput = op.GetOutputDescByName("output_y");
    TensorDesc tensordescInput = op.GetInputDescByName("input_x");
    auto inputShape = tensordescInput.GetShape();
    if (inputShape.GetDims() != UNKNOWN_RANK && inputShape.GetDimNum() != EXPECTED_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(op.GetName().c_str(), "input_x",
                                                 std::to_string(inputShape.GetDimNum()).c_str(),
                                                 "rank must be 3 (n, c, d)");
        return GRAPH_FAILED;
    }
    tensordescOutput.SetShape(inputShape);
    tensordescOutput.SetDataType(tensordescInput.GetDataType());
    (void)op.UpdateOutputDesc("output_y", tensordescOutput);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(NormalizeBatch, NormalizeBatchInferShape);
} // namespace ge
