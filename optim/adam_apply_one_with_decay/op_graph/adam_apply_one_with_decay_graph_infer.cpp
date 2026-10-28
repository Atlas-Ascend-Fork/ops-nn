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
 * \file adam_apply_one_with_decay_graph_infer.cpp
 * \brief AdamApplyOneWithDecay 算子输出数据类型推导
 *
 * 输出 dtype 推导规则：三个输出（output0=m 一阶矩 / output1=v 二阶矩 / output2=var 参数）
 * 均与 input0（一阶矩 m）同类型（公式中梯度为 input4，各张量输入同 dtype 约束）
 *
 * 注册方式：使用通用 IMPL_OP(AdamApplyOneWithDecay) 宏注册 InferDataType，
 * 与 op_host/*_infershape.cpp 中的 IMPL_OP_INFERSHAPE 注册并列存在、互不冲突。
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_INPUT0 = 0;
static constexpr int64_t IDX_OUTPUT0 = 0;
static constexpr int64_t IDX_OUTPUT1 = 1;
static constexpr int64_t IDX_OUTPUT2 = 2;

ge::graphStatus InferDataTypeAdamApplyOneWithDecay(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeAdamApplyOneWithDecay");
    context->SetOutputDataType(IDX_OUTPUT0, context->GetInputDataType(IDX_INPUT0));
    context->SetOutputDataType(IDX_OUTPUT1, context->GetInputDataType(IDX_INPUT0));
    context->SetOutputDataType(IDX_OUTPUT2, context->GetInputDataType(IDX_INPUT0));
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeAdamApplyOneWithDecay");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AdamApplyOneWithDecay).InferDataType(InferDataTypeAdamApplyOneWithDecay);
} // namespace ops
