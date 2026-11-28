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
 * \file transpose_quant_batch_mat_mul_graph_infer.cpp
 * \brief
 */
#include <string>

#include "error_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t ATTR_INDEX_DST_TYPE = 0;
constexpr int64_t Y_INDEX = 0;
} // namespace

ge::graphStatus InferDataTypeForTransposeQuantBatchMatMul(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        CUBE_INNER_ERR_REPORT("TransposeQuantBatchMatMul", "context is null");
        return ge::GRAPH_FAILED;
    }
    const char* nameOp = context->GetNodeName();
    if (nameOp == nullptr) {
        CUBE_INNER_ERR_REPORT("TransposeQuantBatchMatMul", "node name is null");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(nameOp, "TransposeQuantBatchMatMulInferDataType begin");
    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(nameOp, "attrs is nullptr");
        return ge::GRAPH_FAILED;
    }
    const int64_t* dtype = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DST_TYPE);
    if (dtype == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nameOp, "dtype");
        return ge::GRAPH_FAILED;
    }
    const int64_t dstDtype = *dtype;
    if ((dstDtype != static_cast<int64_t>(ge::DT_FLOAT16)) && (dstDtype != static_cast<int64_t>(ge::DT_BF16)) &&
        (dstDtype != static_cast<int64_t>(ge::DT_HIFLOAT8))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nameOp, "dtype", std::to_string(dstDtype).c_str(),
                                              "The attr dtype must be FLOAT16(1), BF16(27) or HIFLOAT8(34)");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(Y_INDEX, static_cast<ge::DataType>(dstDtype));
    OP_LOGD(nameOp, "TransposeQuantBatchMatMulInferDataType end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(TransposeQuantBatchMatMul).InferDataType(InferDataTypeForTransposeQuantBatchMatMul);
} // namespace ops
