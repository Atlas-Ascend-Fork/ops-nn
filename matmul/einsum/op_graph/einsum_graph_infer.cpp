/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file einsum_graph_infer.cpp
 * \brief
 */
#include "error_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t EINSUM_X_IDX = 0;
constexpr size_t EINSUM_Y_IDX = 0;
} // namespace

// 对齐canndev RT1.0的EinsumInferShape：y跟随输入x0的dtype；同时校验所有输入dtype一致
// （einsum语义要求全部输入与输出同dtype，TF 2.x einsum同样如此；原型声明的TensorType允许集
// 仅为各自独立的三种类型，不表达跨张量约束，此处显式拦截混合dtype构图）
ge::graphStatus InferDataTypeEinsum(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeEinsum");
    const size_t inputNum = context->GetComputeNodeInputNum();
    if (inputNum == 0) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x", "the input dtype must not be null");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType xDtype = context->GetInputDataType(EINSUM_X_IDX);
    for (size_t i = 1; i < inputNum; i++) {
        const ge::DataType curDtype = context->GetInputDataType(i);
        if (curDtype != xDtype) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(curDtype).c_str(),
                "the dtypes of all input tensors must be the same for the Einstein summation, expected "
                "the dtype of the first input");
            return ge::GRAPH_FAILED;
        }
    }
    context->SetOutputDataType(EINSUM_Y_IDX, xDtype);
    const ge::DataType yDtype = context->GetOutputDataType(EINSUM_Y_IDX);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeEinsum, x dtype:%s, y dtype:%s",
            ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(yDtype).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Einsum).InferDataType(InferDataTypeEinsum);
} // namespace ops
