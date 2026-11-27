/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// lp_norm_reduce_v2/op_graph/lp_norm_reduce_v2_graph_infer.cpp
// =============================================================================
//
// ROLE: Graph-level data type inference for the LpNormReduceV2 operator.
//   本文件为 GEIR 图模式数据类型推导实现（InferDataType，按 nn 仓拆分口径从
//   op_host/infershape 迁出，注册链与 V1 LpNormReduce 同构）：
//   - 注册链 IMPL_OP(LpNormReduceV2).InferDataType(InferDataTypeForLpNormReduceV2)
//     （签名：gert::InferDataTypeContext* → ge::graphStatus）；
//   - 推导规则：y.dtype = x.dtype（same_as_first_input，无跨 dtype 组合）；
//   - 非法 dtype 校验：输入须为 DT_FLOAT16 / DT_FLOAT / DT_BF16，
//     越界报 dtype_not_supported（DESIGN §8）。
// =============================================================================

#include <string>

#include "register/op_impl_registry.h"                // IMPL_OP macro for operator registration
#include "exe_graph/runtime/infer_datatype_context.h" // InferDataTypeContext
#include "graph/utils/type_utils.h"                   // ge::TypeUtils::DataTypeToSerialString
#include "op_common/log/log.h"                        // OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON

using namespace ge;

namespace ops {

// ---------------------------------------------------------------------------
// InferDataTypeForLpNormReduceV2(context) — data type inference callback
//
// y.dtype = x.dtype（唯一输入决定唯一输出，无 dtype promotion；fp16 输入必 fp16 输出、
// fp32 输入必 fp32 输出、bf16 输入必 bf16 输出）。输入 dtype 不在 950 交付的三档
// （DT_FLOAT16 / DT_FLOAT / DT_BF16）内 → 推导失败。
// （非 static：tests/ut/op_graph 的 graph_infer_pass 用例经原型声明直测本函数）
// ---------------------------------------------------------------------------
ge::graphStatus InferDataTypeForLpNormReduceV2(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType xDtype = context->GetInputDataType(0);
    OP_CHECK_IF(xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT && xDtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(xDtype),
                                                      "only DT_FLOAT16/DT_FLOAT/DT_BF16 are supported on ascend950"),
                return ge::GRAPH_FAILED);
    return context->SetOutputDataType(0, xDtype);
}

// IMPL_OP(LpNormReduceV2).InferDataType(func):
//   Registers InferDataTypeForLpNormReduceV2 as the type inference function
//   for the LpNormReduceV2 operator type.
IMPL_OP(LpNormReduceV2).InferDataType(InferDataTypeForLpNormReduceV2);
} // namespace ops
