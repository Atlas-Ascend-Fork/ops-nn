/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "centralization_proto.h"
#include "error_util.h"
#include "register/op_impl_registry.h"

namespace ge {
namespace {
bool ShapesCompatible(const std::vector<int64_t>& x, const std::vector<int64_t>& y)
{
    if (y.empty() || x == UNKNOWN_RANK || y == UNKNOWN_RANK) {
        return true;
    }
    return x.size() == y.size() && std::equal(x.begin(), x.end(), y.begin(), [](int64_t a, int64_t b) {
               return a == UNKNOWN_DIM || b == UNKNOWN_DIM || a == b;
           });
}

graphStatus CheckFormat(const std::string& node, const char* port, Format format)
{
    if (format != FORMAT_ND && format != FORMAT_RESERVED) {
        OP_LOGE_FOR_INVALID_FORMAT(node.c_str(), port, Ops::Base::ToString(format), "ND");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
} // namespace

IMPLEMT_VERIFIER(Centralization, VerifyCentralization)
{
    const auto node = op.GetName();
    const auto x = op.GetInputDesc("x");
    const auto y = op.GetOutputDesc("y");
    const auto xType = x.GetDataType();
    if (xType != DT_UNDEFINED && xType != DT_FLOAT && xType != DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPE(node.c_str(), "x", Ops::Base::ToString(xType), "FLOAT or FLOAT16");
        return GRAPH_FAILED;
    }
    if (CheckFormat(node, "x", x.GetFormat()) != GRAPH_SUCCESS ||
        CheckFormat(node, "y", y.GetFormat()) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    const auto xShape = x.GetShape();
    const auto yShape = y.GetShape();
    if (!ShapesCompatible(xShape.GetDims(), yShape.GetDims())) {
        OP_LOGE_FOR_INVALID_SHAPE(node.c_str(), "y", Shape2String(yShape), Shape2String(xShape));
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(Centralization, VerifyCentralization);
} // namespace ge

namespace ops {

static ge::graphStatus InferDataTypeForCentralization(gert::InferDataTypeContext* context)
{
    const ge::DataType xDtype = context->GetInputDataType(0);
    return context->SetOutputDataType(0, xDtype);
}

IMPL_OP(Centralization).InferDataType(InferDataTypeForCentralization);

} // namespace ops
