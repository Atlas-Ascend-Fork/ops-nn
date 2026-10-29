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
 * \file transpose_batch_mat_mul_graph_infer.cpp
 * \brief
 */
#include "error_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t kTransposeBatchMatMulScaleIdx = 3;

bool IsTransposeBatchMatMulInputDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_BF16;
}
} // namespace

ge::graphStatus InferDataTypeForTransposeBatchMatMul(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeTransposeBatchMatMul");
    const ge::DataType x1Dtype = context->GetInputDataType(0);
    const ge::DataType x2Dtype = context->GetInputDataType(1);
    OP_CHECK_IF(!IsTransposeBatchMatMulInputDtype(x1Dtype) || !IsTransposeBatchMatMulInputDtype(x2Dtype),
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "input dtype not support"), return ge::GRAPH_FAILED);
    // def中x1/x2各组合始终成对同dtype
    OP_CHECK_IF(x1Dtype != x2Dtype,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "x1 dtype should be same as x2 dtype"),
                return ge::GRAPH_FAILED);

    // scale仅允许int64/uint64(def声明),存在时y固定为int8(量化输出)
    const ge::DataType scaleDtype = context->GetOptionalInputDataType(kTransposeBatchMatMulScaleIdx);
    ge::DataType yDtype = x1Dtype;
    if (scaleDtype != ge::DT_UNDEFINED) {
        OP_CHECK_IF(scaleDtype != ge::DT_INT64 && scaleDtype != ge::DT_UINT64,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "scale dtype only support int64/uint64"),
                    return ge::GRAPH_FAILED);
        yDtype = ge::DT_INT8;
    }
    context->SetOutputDataType(0, yDtype);
    OP_LOGD(context->GetNodeName(),
            "End to do InferDataTypeTransposeBatchMatMul, x1 dtype:%s, x2 dtype:%s, scale dtype:%s, y dtype:%s",
            ge::TypeUtils::DataTypeToSerialString(x1Dtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(x2Dtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(scaleDtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(yDtype).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(TransposeBatchMatMul).InferDataType(InferDataTypeForTransposeBatchMatMul);
} // namespace ops
