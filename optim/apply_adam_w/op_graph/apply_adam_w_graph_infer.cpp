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
 * \file apply_adam_w_graph_infer.cpp
 * \brief ApplyAdamW 算子输出数据类型推导
 *
 * 输出 dtype 推导规则：var / m / v 为 ref 输出，分别跟随同名输入
 *
 * 注册方式：使用通用 IMPL_OP(ApplyAdamW) 宏注册 InferDataType，
 * 与 op_host/apply_adam_w_infershape.cpp 中的 IMPL_OP_INFERSHAPE 注册并列存在、互不冲突。
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_VAR = 0;
static constexpr int64_t IDX_M = 1;
static constexpr int64_t IDX_V = 2;

ge::graphStatus InferDataTypeApplyAdamW(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeApplyAdamW");
    context->SetOutputDataType(IDX_VAR, context->GetInputDataType(IDX_VAR));
    context->SetOutputDataType(IDX_M, context->GetInputDataType(IDX_M));
    context->SetOutputDataType(IDX_V, context->GetInputDataType(IDX_V));
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeApplyAdamW");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(ApplyAdamW).InferDataType(InferDataTypeApplyAdamW);
} // namespace ops
