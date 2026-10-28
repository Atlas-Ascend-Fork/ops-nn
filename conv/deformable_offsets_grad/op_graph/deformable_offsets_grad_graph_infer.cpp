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
 * \file deformable_offsets_grad_graph_infer.cpp
 * \brief DeformableOffsetsGrad 算子输出数据类型推导
 *
 * 输出 dtype 推导规则：grad_x 跟随 x 输入，grad_offsets 跟随 offsets 输入
 *
 * 注册方式：使用通用 IMPL_OP(DeformableOffsetsGrad) 宏注册 InferDataType，
 * 与 op_host/deformable_offsets_grad_infershape.cpp 中的 IMPL_OP_INFERSHAPE 注册并列存在、互不冲突。
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_X = 1;
static constexpr int64_t IDX_OFFSETS = 2;
static constexpr int64_t IDX_GRAD_X = 0;
static constexpr int64_t IDX_GRAD_OFFSETS = 1;

ge::graphStatus InferDataTypeDeformableOffsetsGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeDeformableOffsetsGrad");
    context->SetOutputDataType(IDX_GRAD_X, context->GetInputDataType(IDX_X));
    context->SetOutputDataType(IDX_GRAD_OFFSETS, context->GetInputDataType(IDX_OFFSETS));
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeDeformableOffsetsGrad");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(DeformableOffsetsGrad).InferDataType(InferDataTypeDeformableOffsetsGrad);
} // namespace ops
