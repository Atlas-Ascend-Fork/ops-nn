/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_apply_rms_prop_graph_infer.cpp
 * \brief InplaceApplyRMSProp graph InferDataType implementation.
 *
 * The dtype gate independently rejects the documented contract violations that are
 * decidable from the input dtypes alone: the {FLOAT, FLOAT16, BFLOAT16} whitelist and
 * cross-input dtype consistency. This is the operator's single InferDataType
 * registration; opt_host registers InferShape only.
 */

#include <string>
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

namespace {
constexpr size_t kInputCount = 8;
const char* const kInputNames[kInputCount] = {"var", "ms", "mom", "lr", "rho", "momentum", "epsilon", "grad"};

bool IsSupportedDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}
} // namespace

static ge::graphStatus InferDataType4InplaceApplyRMSProp(gert::InferDataTypeContext* context)
{
    const ge::DataType varDtype = context->GetInputDataType(0);
    for (size_t i = 0; i < kInputCount; ++i) {
        const ge::DataType dtype = context->GetInputDataType(i);
        if (dtype == ge::DT_UNDEFINED) {
            // Not resolved at this point; the Tiling gate re-checks the concrete dtype.
            continue;
        }
        if (!IsSupportedDtype(dtype)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), kInputNames[i],
                                                  Ops::Base::ToString(dtype).c_str(),
                                                  "dtype must be float, float16, or bfloat16");
            return ge::GRAPH_FAILED;
        }
        if (varDtype != ge::DT_UNDEFINED && dtype != varDtype) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context->GetNodeName(), (std::string(kInputNames[i]) + " and var").c_str(),
                (Ops::Base::ToString(dtype) + " and " + Ops::Base::ToString(varDtype)).c_str(),
                "all inputs must have the same dtype");
            return ge::GRAPH_FAILED;
        }
    }

    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(1));
    context->SetOutputDataType(2, context->GetInputDataType(2));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(InplaceApplyRMSProp).InferDataType(InferDataType4InplaceApplyRMSProp);

} // namespace ops
