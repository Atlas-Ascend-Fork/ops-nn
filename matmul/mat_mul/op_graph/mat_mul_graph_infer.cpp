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
 * \file mat_mul_graph_infer.cpp
 * \brief
 */
#include "error_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t MATMUL_X1_IDX = 0;
constexpr size_t MATMUL_X2_IDX = 1;
constexpr size_t MATMUL_BIAS_IDX = 2;
constexpr size_t MATMUL_Y_IDX = 0;

bool IsMatMulInputDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_INT32 || dtype == ge::DT_BF16 ||
           dtype == ge::DT_HIFLOAT8 || dtype == ge::DT_INT8;
}

bool IsMatMulBiasDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_INT32 || dtype == ge::DT_BF16;
}
} // namespace

ge::graphStatus InferDataTypeMatMul(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMatMul");
    const ge::DataType x1Dtype = context->GetInputDataType(MATMUL_X1_IDX);
    const ge::DataType x2Dtype = context->GetInputDataType(MATMUL_X2_IDX);
    const ge::DataType biasDtype = context->GetOptionalInputDataType(MATMUL_BIAS_IDX);
    OP_CHECK_IF(!IsMatMulInputDtype(x1Dtype) || !IsMatMulInputDtype(x2Dtype),
                CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                                      "x1/x2 dtype only support float32, float16, int32, bfloat16, hifloat8"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(x1Dtype != x2Dtype,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "x1 dtype should be same as x2 dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        biasDtype != ge::DT_UNDEFINED && !IsMatMulBiasDtype(biasDtype),
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "bias dtype only support float32, float16, int32, bfloat16"),
        return ge::GRAPH_FAILED);
    // int8输入走整数累加,y为int32(对齐950组合表);其余跟随x1
    // 对齐canndev RT1行为:直接按推导结果设置输出dtype,不校验节点已声明的y
    const ge::DataType yDtype = (x1Dtype == ge::DT_INT8) ? ge::DT_INT32 : x1Dtype;
    context->SetOutputDataType(MATMUL_Y_IDX, yDtype);
    OP_LOGD(
        context->GetNodeName(), "End to do InferDataTypeMatMul, x1 dtype:%s, x2 dtype:%s, bias dtype:%s, y dtype:%s",
        ge::TypeUtils::DataTypeToSerialString(x1Dtype).c_str(), ge::TypeUtils::DataTypeToSerialString(x2Dtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(yDtype).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MatMul).InferDataType(InferDataTypeMatMul);
} // namespace ops
