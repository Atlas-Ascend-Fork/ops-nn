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
 * \file apply_adagrad_d_graph_infer.cpp
 * \brief ApplyAdagradD 算子输出数据类型推导
 *
 * 输出 dtype 推导规则：var/accum 为 ref 输出，分别跟随同名输入
 *
 * 注册方式：使用通用 IMPL_OP(ApplyAdagradD) 宏注册 InferDataType，
 * 与 op_host/*_infershape.cpp 中的 IMPL_OP_INFERSHAPE 注册并列存在、互不冲突。
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_VAR = 0;
static constexpr int64_t IDX_ACCUM = 1;

ge::graphStatus InferDataTypeApplyAdagradD(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeApplyAdagradD");
    context->SetOutputDataType(IDX_VAR, context->GetInputDataType(IDX_VAR));
    context->SetOutputDataType(IDX_ACCUM, context->GetInputDataType(IDX_ACCUM));
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeApplyAdagradD");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(ApplyAdagradD).InferDataType(InferDataTypeApplyAdagradD);
} // namespace ops
