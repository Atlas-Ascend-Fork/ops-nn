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
 * \file gemm_v3_graph_infer.cpp
 * \brief
 */
#include "error_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
bool IsGemmV3InputDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_BF16;
}
} // namespace

ge::graphStatus InferDataTypeForGemmV3(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeGemmV3");
    const ge::DataType aDtype = context->GetInputDataType(0);
    const ge::DataType bDtype = context->GetInputDataType(1);
    const ge::DataType cDtype = context->GetOptionalInputDataType(2);
    OP_CHECK_IF(!IsGemmV3InputDtype(aDtype) || !IsGemmV3InputDtype(bDtype),
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "gemmv3 input dtype not support"),
                return ge::GRAPH_FAILED);
    // y按c推导: c为fp16/bf16时y跟随c(16进16出), c为fp32时y为fp32(覆盖16进32出);
    // c未实例化时放行,由infershape给出"c必填"的准确报错,此时y退化为跟随a
    ge::DataType yDtype = aDtype;
    if (cDtype != ge::DT_UNDEFINED) {
        OP_CHECK_IF(
            cDtype != ge::DT_FLOAT && cDtype != ge::DT_FLOAT16 && cDtype != ge::DT_BF16,
            CUBE_INNER_ERR_REPORT(context->GetNodeName(), "gemmv3 input c dtype only support float16/bfloat16/float32"),
            return ge::GRAPH_FAILED);
        yDtype = cDtype;
    }
    context->SetOutputDataType(0, yDtype);
    OP_LOGD(
        context->GetNodeName(), "End to do InferDataTypeGemmV3, a dtype:%s, b dtype:%s, c dtype:%s, y dtype:%s",
        ge::TypeUtils::DataTypeToSerialString(aDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(bDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(cDtype).c_str(), ge::TypeUtils::DataTypeToSerialString(yDtype).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(GemmV3).InferDataType(InferDataTypeForGemmV3);
} // namespace ops
