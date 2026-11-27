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
 * \file lp_norm_reduce_v2_infershape.cpp
 * \brief LpNormReduceV2 算子形状推导实现（InferDataType 见 op_graph/graph_infer，按 nn 仓拆分口径）
 *
 * 对齐 canndev InferShape4LpNorm（空 axes 展开、负索引归一化、keepdim 置 1/删除），
 * 并按 REQUIREMENTS §9 增强校验：rank ≤ 8、axes 越界/重复/数量 > rank 报错（attribute_value_out_of_range）。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

#include <string>
#include <vector>

using namespace ge;

namespace ops {

constexpr int64_t MAX_RANK = 8;

static ge::graphStatus InferShape4LpNormReduceV2(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* axes = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(1); // 属性索引 1：axes
    OP_CHECK_NULL_WITH_CONTEXT(context, axes);
    const bool* keepdim = attrs->GetAttrPointer<bool>(2); // 属性索引 2：keepdim
    OP_CHECK_NULL_WITH_CONTEXT(context, keepdim);

    const int64_t dimNum = xShape->GetDimNum();
    // rank(x) ≤ 8 校验（shape_mismatch，DESIGN §8）
    OP_CHECK_IF(dimNum > MAX_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(dimNum).c_str(),
                                                         "rank of x must be in [0, 8]"),
                return ge::GRAPH_FAILED);

    // axes 归一化：空列表 → 全维度；负索引 → a + rank(x)（对齐 canndev InferShape4LpNorm）
    std::vector<int64_t> newAxes;
    if (axes->GetSize() == 0UL) {
        for (int64_t i = 0; i < dimNum; i++) {
            newAxes.push_back(i);
        }
    } else {
        OP_CHECK_IF(static_cast<int64_t>(axes->GetSize()) > dimNum,
                    OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON(context->GetNodeName(), "axes",
                                                             std::to_string(axes->GetSize()).c_str(),
                                                             "axes size must not exceed rank of x"),
                    return ge::GRAPH_FAILED);
        for (size_t i = 0; i < axes->GetSize(); i++) {
            const int64_t tmpAxes = (axes->GetData())[i];
            const int64_t realAxes = (tmpAxes < 0) ? (tmpAxes + dimNum) : tmpAxes;
            OP_CHECK_IF(realAxes < 0 || realAxes >= dimNum,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "axes", std::to_string(tmpAxes).c_str(),
                            "axis is out of range [-rank(x), rank(x)) after negative-index normalization"),
                        return ge::GRAPH_FAILED);
            for (size_t j = 0; j < newAxes.size(); j++) {
                OP_CHECK_IF(newAxes[j] == realAxes,
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axes",
                                                                  std::to_string(realAxes).c_str(),
                                                                  "duplicate elements in axes are not allowed"),
                            return ge::GRAPH_FAILED);
            }
            newAxes.push_back(realAxes);
        }
    }

    // 输出 shape：i ∈ newAxes ? (keepdim ? 1 : 删除) : x.shape[i]
    std::vector<int64_t> yVec;
    for (int64_t i = 0; i < dimNum; i++) {
        bool isReduce = false;
        for (size_t j = 0; j < newAxes.size(); j++) {
            if (newAxes[j] == i) {
                isReduce = true;
                break;
            }
        }
        if (isReduce) {
            if (*keepdim) {
                yVec.push_back(1);
            }
        } else {
            yVec.push_back(xShape->GetDim(i));
        }
    }
    // 全维度归约 + keepdim=false → 输出 shape (1,) 而非 0 维标量：
    // GE RunGraph 对 0 维输出张量 GetSize()=0（不回拷数据），调用方按 1 元素读
    // 会读到未初始化值；标量语义以 1 元素张量承载（对齐 V1 标量链路与
    // REQUIREMENTS v1.1 的 GE 兼容口径，TTK/ST 数据比对按 flat 元素一致）。
    if (yVec.empty()) {
        yVec.push_back(1);
    }
    yShape->SetDimNum(static_cast<size_t>(yVec.size()));
    for (size_t i = 0; i < yVec.size(); i++) {
        yShape->SetDim(static_cast<int64_t>(i), yVec[i]);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LpNormReduceV2).InferShape(InferShape4LpNormReduceV2);

} // namespace ops
