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
 * \file fused_mat_mul_graph_infer.cpp
 * \brief
 */
#include <cstring>

#include "error_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t kMatMulX1Idx = 0;
constexpr size_t kMatMulX2Idx = 1;
constexpr size_t kBatchMatMulBiasIdx = 2; // input_bias index is different with op_type
constexpr size_t kFusedMatMulX3Idx = 3;
constexpr size_t kOutputIdx = 0;

bool IsFusedMatMulInputDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_BF16;
}

// 模式判定与tiling的NpuArchFusedOpSupport/dtype组合表对齐
bool IsQuantOpType(const char* fused_op_type)
{
    return strcmp(fused_op_type, "quant") == 0 || strcmp(fused_op_type, "relu_quant") == 0;
}

bool IsAddMulOpType(const char* fused_op_type)
{
    return strcmp(fused_op_type, "add") == 0 || strcmp(fused_op_type, "mul") == 0;
}

bool IsScaleAddOpType(const char* fused_op_type) { return strcmp(fused_op_type, "scale_add") == 0; }
} // namespace
ge::graphStatus InferDataTypeForFusedMatMul(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeFusedMatMul");
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "attrs is null"),
                return ge::GRAPH_FAILED);
    const char* fusedOpType = attrs->GetAttrPointer<char>(kFusedMatMulX3Idx);
    OP_CHECK_IF(fusedOpType == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "fused_op_type is null"),
                return ge::GRAPH_FAILED);
    const bool isQuant = IsQuantOpType(fusedOpType);
    const bool is16Cast32 = strcmp(fusedOpType, "16cast32") == 0;
    const bool isScaleAdd = IsScaleAddOpType(fusedOpType);
    // add/mul/scale_add要求x3; ""/relu/gelu_erf/gelu_tanh不支持x3(对齐tiling dtype组合表)
    const bool isX3Required = IsAddMulOpType(fusedOpType) || isScaleAdd;
    const ge::DataType aDtype = context->GetInputDataType(kMatMulX1Idx);
    const ge::DataType bDtype = context->GetInputDataType(kMatMulX2Idx);
    const ge::DataType biasDtype = context->GetOptionalInputDataType(kBatchMatMulBiasIdx);
    const ge::DataType cDtype = context->GetOptionalInputDataType(kFusedMatMulX3Idx);

    OP_CHECK_IF(!IsFusedMatMulInputDtype(aDtype) || !IsFusedMatMulInputDtype(bDtype),
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "input dtype not support"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(aDtype != bDtype,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "a data type should be same as b data type"),
                return ge::GRAPH_FAILED);

    // bias: quant仅fp16; scale_add不支持bias; 其余模式与x1同dtype或fp32
    if (biasDtype != ge::DT_UNDEFINED) {
        const bool biasOk = isQuant ? biasDtype == ge::DT_FLOAT16 :
                                      (!isScaleAdd && (biasDtype == aDtype || biasDtype == ge::DT_FLOAT));
        OP_CHECK_IF(!biasOk, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "bias input dtype not support"),
                    return ge::GRAPH_FAILED);
    }

    ge::DataType yDtype = aDtype;
    if (isQuant) {
        OP_CHECK_IF(aDtype != ge::DT_FLOAT16 || cDtype != ge::DT_UINT64,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "quant only supports float16 x1/x2 and uint64 x3"),
                    return ge::GRAPH_FAILED);
        yDtype = ge::DT_INT8;
    } else if (is16Cast32) {
        // 16进32出: x1/x2必须为fp16/bf16(不允许fp32),不支持x3, y的数据类型固定为fp32
        OP_CHECK_IF(aDtype == ge::DT_FLOAT,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "16cast32 only supports float16/bfloat16 x1/x2"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(cDtype != ge::DT_UNDEFINED,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "16cast32 does not support x3"),
                    return ge::GRAPH_FAILED);
        yDtype = ge::DT_FLOAT;
    } else {
        OP_CHECK_IF(isX3Required && cDtype == ge::DT_UNDEFINED,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "x3 is required for add/mul/scale_add"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!isX3Required && cDtype != ge::DT_UNDEFINED,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "x3 is only supported for add/mul/scale_add"),
                    return ge::GRAPH_FAILED);
        if (cDtype != ge::DT_UNDEFINED) {
            OP_CHECK_IF(aDtype != cDtype,
                        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "c data type should be same as x1 data type"),
                        return ge::GRAPH_FAILED);
        }
    }
    context->SetOutputDataType(kOutputIdx, yDtype);
    OP_LOGD(
        context->GetNodeName(),
        "End to do InferDataTypeFusedMatMul, x1 dtype:%s, x2 dtype:%s, bias dtype:%s, x3 dtype:%s, y dtype:%s, "
        "fused_op_type:%s",
        ge::TypeUtils::DataTypeToSerialString(aDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(bDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(cDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(), fusedOpType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(FusedMatMul).InferDataType(InferDataTypeForFusedMatMul);
} // namespace ops
