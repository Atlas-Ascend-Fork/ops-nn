/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr int64_t IDX_0 = 0;

ge::graphStatus InferDataType4SigmoidCrossEntropyWithLogitsV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4SigmoidCrossEntropyWithLogitsV2");
    const ge::DataType predictDtype = context->GetInputDataType(IDX_0);
    const ge::DataType outputDtype = context->GetOutputDataType(IDX_0);
    // 输出 dtype 须与 tiling 校验一致：等于 predict 或为 DT_FLOAT，否则归一化为 predict 的 dtype。
    if (outputDtype != predictDtype && outputDtype != ge::DT_FLOAT) {
        if (context->SetOutputDataType(IDX_0, predictDtype) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context->GetNodeName(), "Set output data type failed for SigmoidCrossEntropyWithLogitsV2");
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4SigmoidCrossEntropyWithLogitsV2");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SigmoidCrossEntropyWithLogitsV2).InferDataType(InferDataType4SigmoidCrossEntropyWithLogitsV2);
} // namespace ops
