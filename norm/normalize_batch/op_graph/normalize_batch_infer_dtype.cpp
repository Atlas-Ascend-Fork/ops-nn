/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file normalize_batch_infer_dtype.cpp
 * \brief Runtime 2.0 graph data type inference for NormalizeBatch.
 */

#include "register/op_impl_registry.h" // IMPL_OP macro for operator registration

using namespace ge;

namespace ops {

// InferDataTypeForNormalizeBatch(context) — data type inference callback
//
// output_y.dtype = input_x.dtype 透传（无 promotion）：
//   dtype 白名单（DT_FLOAT）由 def 侧 REG_OP DTYPE_INPUT_X 把关，
//   seq_len 的 DT_INT32 与声明一致性由 graph verifier（VerifyNormalizeBatch）承载，
//   本回调只负责 gert 轨的输出 dtype 推导。
static ge::graphStatus InferDataTypeForNormalizeBatch(gert::InferDataTypeContext* context)
{
    const ge::DataType inputDtype = context->GetInputDataType(0);
    if (context->SetOutputDataType(0, inputDtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(NormalizeBatch).InferDataType(InferDataTypeForNormalizeBatch);
} // namespace ops
