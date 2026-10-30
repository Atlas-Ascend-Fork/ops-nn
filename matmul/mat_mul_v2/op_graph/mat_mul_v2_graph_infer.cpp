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
 * \file mat_mul_v2_graph_infer.cpp
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

bool IsMatMulV2InputDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_INT32 || dtype == ge::DT_INT8 ||
           dtype == ge::DT_INT4 || dtype == ge::DT_BF16 || dtype == ge::DT_HIFLOAT8;
}

bool IsMatMulV2BiasDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_INT32 || dtype == ge::DT_BF16;
}

// x1/x2为int8/int4时输出为int32(整数累加),其余场景输出与x1一致
ge::DataType GetMatMulV2BaseOutDtype(ge::DataType x1Dtype)
{
    return (x1Dtype == ge::DT_INT8 || x1Dtype == ge::DT_INT4) ? ge::DT_INT32 : x1Dtype;
}
} // namespace

ge::graphStatus InferDataTypeMatMulV2(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMatMulV2");
    const ge::DataType x1Dtype = context->GetInputDataType(MATMUL_X1_IDX);
    const ge::DataType x2Dtype = context->GetInputDataType(MATMUL_X2_IDX);
    const ge::DataType biasDtype = context->GetOptionalInputDataType(MATMUL_BIAS_IDX);
    OP_CHECK_IF(
        !IsMatMulV2InputDtype(x1Dtype) || !IsMatMulV2InputDtype(x2Dtype),
        CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                              "x1/x2 dtype only support float32, float16, int32, int8, int4, bfloat16, hifloat8"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(x1Dtype != x2Dtype,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "x1 dtype should be same as x2 dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        biasDtype != ge::DT_UNDEFINED && !IsMatMulV2BiasDtype(biasDtype),
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "bias dtype only support float32, float16, int32, bfloat16"),
        return ge::GRAPH_FAILED);
    const ge::DataType yDtype = GetMatMulV2BaseOutDtype(x1Dtype);
    context->SetOutputDataType(MATMUL_Y_IDX, yDtype);
    OP_LOGD(
        context->GetNodeName(), "End to do InferDataTypeMatMulV2, x1 dtype:%s, x2 dtype:%s, bias dtype:%s, y dtype:%s",
        ge::TypeUtils::DataTypeToSerialString(x1Dtype).c_str(), ge::TypeUtils::DataTypeToSerialString(x2Dtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(yDtype).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MatMulV2).InferDataType(InferDataTypeMatMulV2);
} // namespace ops
