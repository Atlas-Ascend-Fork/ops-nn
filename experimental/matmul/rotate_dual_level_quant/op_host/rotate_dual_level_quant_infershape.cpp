/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rotate_dual_level_quant_common.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

namespace ops {
namespace {
namespace RDLQ = ::RotateDualLevelQuantNS;

std::vector<int64_t> Dimensions(const gert::Shape& shape)
{
    std::vector<int64_t> result;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        result.push_back(shape.GetDim(i));
    }
    return result;
}

void SetShape(gert::Shape& output, const std::vector<int64_t>& dims)
{
    output.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        output.SetDim(i, dims[i]);
    }
}

ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetComputeNodeInputNum() < 2 || context->GetComputeNodeInputNum() > 4 ||
        context->GetComputeNodeOutputNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    const auto* x = context->GetInputShape(0);
    const auto* rotation = context->GetInputShape(1);
    auto* y = context->GetOutputShape(0);
    auto* level0 = context->GetOutputShape(1);
    auto* level1 = context->GetOutputShape(2);
    if (x == nullptr || rotation == nullptr || y == nullptr || level0 == nullptr || level1 == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* attrs = context->GetAttrs();
    const auto* level0BlockSize = attrs != nullptr && attrs->GetAttrNum() > 7 ? attrs->GetAttrPointer<int64_t>(7) :
                                                                                nullptr;
    const auto* level1BlockSize = attrs != nullptr && attrs->GetAttrNum() > 8 ? attrs->GetAttrPointer<int64_t>(8) :
                                                                                nullptr;
    if (attrs != nullptr && ((attrs->GetAttrNum() > 7 && level0BlockSize == nullptr) ||
                             (attrs->GetAttrNum() > 8 && level1BlockSize == nullptr))) {
        OP_LOGE(context->GetNodeName(), "Block-size attributes must be integers.");
        return ge::GRAPH_FAILED;
    }
    std::string reason;
    if (!RDLQ::ValidateBlockSizes(level0BlockSize == nullptr ? RDLQ::LEVEL0_BLOCK_SIZE : *level0BlockSize,
                                  level1BlockSize == nullptr ? RDLQ::LEVEL1_BLOCK_SIZE : *level1BlockSize, reason)) {
        OP_LOGE(context->GetNodeName(), "%s", reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (Ops::Base::IsUnknownRank(*x)) {
        Ops::Base::SetUnknownRank(*y);
        Ops::Base::SetUnknownRank(*level0);
        Ops::Base::SetUnknownRank(*level1);
        return ge::GRAPH_SUCCESS;
    }
    const auto* alpha = context->GetOptionalInputShape(2);
    if (alpha != nullptr && !Ops::Base::IsUnknownRank(*alpha)) {
        const auto dims = Dimensions(*alpha);
        if (dims.size() != 1 || (dims.front() != -1 && dims.front() != 1)) {
            OP_LOGE(context->GetNodeName(), "alpha must have shape [1].");
            return ge::GRAPH_FAILED;
        }
    }
    const auto* groups = context->GetOptionalInputShape(3);
    if (groups != nullptr && x->GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "Grouped x must be two-dimensional.");
        return ge::GRAPH_FAILED;
    }
    // Unknown rotation/group rank cannot determine their final rank, but output
    // shapes still depend only on x. Use unknown dimensions for that metadata
    // while keeping known x dimensions on the same validation path.
    const auto rotationDims = Ops::Base::IsUnknownRank(*rotation) ?
                                  (groups == nullptr ? std::vector<int64_t>{-1, -1} :
                                                       std::vector<int64_t>{-1, -1, -1}) :
                                  Dimensions(*rotation);
    const auto groupDims = groups == nullptr ?
                               std::vector<int64_t>{} :
                               (Ops::Base::IsUnknownRank(*groups) ? std::vector<int64_t>{-1} : Dimensions(*groups));
    RDLQ::ShapeInfo info;
    if (!RDLQ::ValidateShapes(Dimensions(*x), rotationDims, groups == nullptr ? nullptr : &groupDims, true, info,
                              reason)) {
        OP_LOGE(context->GetNodeName(), "%s", reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (attrs != nullptr && attrs->GetAttrNum() > 6) {
        const auto* type = attrs->GetAttrPointer<int64_t>(6);
        if (type == nullptr || (*type != 0 && *type != 1)) {
            OP_LOGE(context->GetNodeName(), "group_list_type must be 0 or 1.");
            return ge::GRAPH_FAILED;
        }
    }
    *y = *x;
    SetShape(*level0, info.level0Scale);
    SetShape(*level1, info.level1Scale);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* dtype = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (dtype == nullptr ||
        (*dtype != ge::DT_FLOAT4_E2M1 && *dtype != ge::DT_FLOAT8_E4M3FN && *dtype != ge::DT_FLOAT8_E5M2)) {
        OP_LOGE(context->GetNodeName(), "y_dtype must be FP4_E2M1, FP8_E4M3FN or FP8_E5M2.");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, static_cast<ge::DataType>(*dtype));
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT8_E8M0);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(RotateDualLevelQuant).InferShape(InferShape).InferDataType(InferDataType);
} // namespace ops
