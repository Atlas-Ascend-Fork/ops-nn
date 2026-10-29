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
 * \file batch_mat_mul_v3_graph_infer.cpp
 * \brief
 */
#include "error_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t BATCH_MATMUL_X1_IDX = 0;
constexpr size_t BATCH_MATMUL_X2_IDX = 1;
constexpr size_t BATCH_MATMUL_BIAS_IDX = 2;
constexpr size_t BATCH_MATMUL_Y_IDX = 0;

bool IsBatchMatMulV3InputDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_BF16;
}
} // namespace

ge::graphStatus InferDataTypeBatchMatMulV3(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeBatchMatMulV3");
    const ge::DataType x1Dtype = context->GetInputDataType(BATCH_MATMUL_X1_IDX);
    const ge::DataType x2Dtype = context->GetInputDataType(BATCH_MATMUL_X2_IDX);
    const ge::DataType biasDtype = context->GetOptionalInputDataType(BATCH_MATMUL_BIAS_IDX);
    OP_CHECK_IF(!IsBatchMatMulV3InputDtype(x1Dtype) || !IsBatchMatMulV3InputDtype(x2Dtype),
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "x1/x2 dtype only support float16, float32, bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(x1Dtype != x2Dtype,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "x1 dtype should be same as x2 dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(biasDtype != ge::DT_UNDEFINED && biasDtype != x1Dtype && biasDtype != ge::DT_FLOAT,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "bias dtype should be same as x1 or float32"),
                return ge::GRAPH_FAILED);
    // y未显式声明时跟随x1;显式声明y仅允许两种场景:与x1一致,或fp16/bf16输入声明fp32输出(16进32出)
    ge::DataType yDtype = x1Dtype;
    const ge::DataType declaredYDtype = context->GetOutputDataType(BATCH_MATMUL_Y_IDX);
    if (declaredYDtype != ge::DT_UNDEFINED && declaredYDtype < ge::DT_MAX) {
        const bool is16Cast32 = (x1Dtype == ge::DT_FLOAT16 || x1Dtype == ge::DT_BF16) && declaredYDtype == ge::DT_FLOAT;
        OP_CHECK_IF(declaredYDtype != x1Dtype && !is16Cast32,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                                          "y dtype should be same as x1, or float32 when x1 is float16/bfloat16"),
                    return ge::GRAPH_FAILED);
        yDtype = declaredYDtype;
    }
    context->SetOutputDataType(BATCH_MATMUL_Y_IDX, yDtype);
    OP_LOGD(context->GetNodeName(),
            "End to do InferDataTypeBatchMatMulV3, x1 dtype:%s, x2 dtype:%s, bias dtype:%s, y dtype:%s",
            ge::TypeUtils::DataTypeToSerialString(x1Dtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(x2Dtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(yDtype).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(BatchMatMulV3).InferDataType(InferDataTypeBatchMatMulV3);
} // namespace ops
