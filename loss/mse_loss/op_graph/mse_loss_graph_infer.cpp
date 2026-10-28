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
 * \file mse_loss_graph_infer.cpp
 * \brief MseLoss 算子输出数据类型推导
 *
 * 输出 dtype 推导规则：损失输出与 predict 输入同类型
 *
 * 注册方式：使用通用 IMPL_OP(MseLoss) 宏注册 InferDataType，
 * 与 op_host/*_infershape.cpp 中的 IMPL_OP_INFERSHAPE 注册并列存在、互不冲突。
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_PREDICT = 0;
static constexpr int64_t IDX_Y = 0;

ge::graphStatus InferDataTypeMseLoss(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMseLoss");
    context->SetOutputDataType(IDX_Y, context->GetInputDataType(IDX_PREDICT));
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMseLoss");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MseLoss).InferDataType(InferDataTypeMseLoss);
} // namespace ops
