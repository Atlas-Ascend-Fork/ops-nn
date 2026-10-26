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
 * \file avg_pool_update_infershape.cpp
 * \brief Infershape implementation for avg_pool_update operator
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShapeAvgPoolUpdate(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeAvgPoolUpdate");

    const gert::Shape* x1Shape = context->GetInputShape(IDX_0);
    OP_CHECK_IF(
        x1Shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "x1",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_IF(yShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "yShape is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);

    // 输出 y shape 直接复制 x1 shape（正确处理动态 shape: rank 未知 -2 / dim 未知 -1）
    *yShape = *x1Shape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeAvgPoolUpdate");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AvgPoolUpdate).InferShape(InferShapeAvgPoolUpdate);
} // namespace ops
