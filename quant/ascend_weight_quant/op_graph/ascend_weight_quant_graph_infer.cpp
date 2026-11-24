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
 * \file ascend_weight_quant_graph_infer.cpp
 * \brief AscendWeightQuant InferDataType
 */

#include "exe_graph/runtime/infer_datatype_context.h"
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t OUTPUT_Y_IDX = 0;
constexpr size_t ATTR_DST_TYPE_IDX = 0;
constexpr int64_t DEFAULT_DST_TYPE = ge::DT_INT8;
const char* const DST_TYPE_UNSUPPORTED = "dst_type can only support DT_INT8 or DT_INT4.";
} // namespace

static ge::graphStatus CheckDstType(const char* nodeName, const int64_t* dstTypePtr)
{
    const int64_t dstType = (dstTypePtr != nullptr) ? *dstTypePtr : DEFAULT_DST_TYPE;
    if (dstType != ge::DT_INT8 && dstType != ge::DT_INT4) {
        OP_LOGE(nodeName, "%s", DST_TYPE_UNSUPPORTED);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForAscendWeightQuant(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("AscendWeightQuant", "%s is nullptr!", "context"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter dtype infer. ");

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int64_t* dstTypePtr = (attrs != nullptr) ? attrs->GetAttrPointer<int64_t>(ATTR_DST_TYPE_IDX) : nullptr;
    if (CheckDstType(context->GetNodeName(), dstTypePtr) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const int64_t dstType = (dstTypePtr != nullptr) ? *dstTypePtr : DEFAULT_DST_TYPE;
    OP_LOGD(context->GetNodeName(), "Infer dst_type[%d].", static_cast<int32_t>(dstType));
    return context->SetOutputDataType(OUTPUT_Y_IDX, static_cast<ge::DataType>(dstType));
}

IMPL_OP(AscendWeightQuant).InferDataType(InferDataTypeForAscendWeightQuant);
} // namespace ops
