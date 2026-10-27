/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <set>
#include <string>
#include <vector>

#include "register/op_impl_registry.h"             // IMPL_OP_INFERSHAPE macro
#include "exe_graph/runtime/infer_shape_context.h" // InferShapeContext, gert::Shape
#include "exe_graph/runtime/runtime_attrs.h"       // gert::RuntimeAttrs
#include "util/shape_util.h"
#include "op_common/log/log.h" // OP_CHECK_* / OP_LOGE_FOR_INVALID_* macros

using namespace ge;

namespace ops {

// 属性下标（op_graph/lp_norm_reduce_proto.h REG_OP ATTR 声明序，与
// op_host/arch35/lp_norm_reduce_tiling_arch35.cpp 常量一致）
constexpr size_t ATTR_P_IDX = 0;        // attr p（Int，默认 2）
constexpr size_t ATTR_AXES_IDX = 1;     // attr axes（ListInt，默认 {} = 全归约）
constexpr size_t ATTR_KEEP_DIM_IDX = 2; // attr keepdim（Bool，默认 false）

// p 值域哨兵（spec.yaml attributes.p：int_in_range [0, 2147483647]；
// -inf 以整型哨兵 -2147483648 表达，canndev CONST_INF 口径）
constexpr int64_t INF_POS_SENTINEL = 2147483647;
constexpr int64_t INF_NEG_SENTINEL = -2147483648LL;
constexpr int64_t DEFAULT_P = 2;

// ---------------------------------------------------------------------------
// InferShape4LpNormReduce(context) — shape inference function
//
// 推导主体与 canndev InferShape4LpNorm 同构（axes 空列表 → 全部维度归约、
// 负轴 +rank 归一、keepdim 决定归约维保留 1 还是删除），本工程按
// design/InferShapeDtype.md 前置 rank / p 值域 / axes 域与去重 / ±inf 空维
// 四项校验（canndev 原生推导静默忽略越界轴，本工程按 spec 语义显式拒绝）。
// ---------------------------------------------------------------------------
static ge::graphStatus InferShape4LpNormReduce(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin InferShape4LpNormReduce.");
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // ── 校验 rank(x) ∈ [0, 8]（批 b-5 对齐 canndev/changwei：rank-0 标量放行，
    //    空 axes = 单元素全归约 → y 同为标量；rank-0 + 非空 axes 由 axes 值域拒绝）──
    const int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    OP_CHECK_IF(rank > 8,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(rank),
                                                         "rank of x must be in [0, 8]"),
                return ge::GRAPH_FAILED);

    // ── 校验 p 值域（批 b-1 对齐 canndev 公共契约：p >= 0 完整非负 int64）──
    const int64_t* attrP = attrs->GetAttrPointer<int64_t>(ATTR_P_IDX);
    const int64_t p = (attrP == nullptr) ? DEFAULT_P : (*attrP);
    OP_CHECK_IF(
        p < 0 && p != INF_NEG_SENTINEL,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "p", std::to_string(p),
                                              "p must be non-negative (full int64) or -2147483648 (-inf sentinel); "
                                              "2147483647 is reserved as the +inf sentinel"),
        return ge::GRAPH_FAILED);

    // Unknown rank is a marker, not a scalar or a one-dimensional tensor.
    // Validate shape-independent attributes first; defer axes until rank is known.
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    // ── attr keepdim（idx 2，默认 false）──
    const bool* attrKeepDim = attrs->GetAttrPointer<bool>(ATTR_KEEP_DIM_IDX);
    const bool keepdim = (attrKeepDim == nullptr) ? false : (*attrKeepDim);

    // ── axes 归一校验：空列表 → 全部维度；负轴 +rank；越界拒绝；
    //    重复轴按首次出现去重（批 b-2，对齐 canndev/changwei 语义，set 天然去重）──
    const gert::TypedContinuousVector<int64_t>* axesVec = attrs->GetListInt(ATTR_AXES_IDX);
    const int64_t axesNum = (axesVec == nullptr) ? 0 : static_cast<int64_t>(axesVec->GetSize());
    std::set<int64_t> reduceSet; // 归一后的归约轴集合（首次出现去重 + 输出构造复用）
    if (axesNum == 0) {
        for (int64_t i = 0; i < rank; i++) {
            reduceSet.insert(i);
        }
    } else {
        const int64_t* data = axesVec->GetData();
        OP_CHECK_IF(data == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "axes",
                                                             "axes ListInt attr data ptr is null"),
                    return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < axesNum; i++) {
            const int64_t v = data[i];
            OP_CHECK_IF(
                v < -rank || v >= rank,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axes", std::to_string(v),
                                                      "axes[" + std::to_string(i) + "] out of range [-" +
                                                          std::to_string(rank) + ", " + std::to_string(rank) + ")"),
                return ge::GRAPH_FAILED);
            const int64_t norm = (v < 0) ? (v + rank) : v;
            (void)reduceSet.insert(norm); // 重复轴静默去重（首次出现生效）
        }
    }

    // ── 校验 p=±inf 哨兵且归约轴长度为 0 → 拒绝（空集 max/min 无定义，
    //    canndev 约束 2）。动态编译期未知维（-1）跳过，交运行时 tiling 拒绝 ──
    if (p == INF_POS_SENTINEL || p == INF_NEG_SENTINEL) {
        for (const int64_t axis : reduceSet) {
            if (xShape->GetDim(static_cast<size_t>(axis)) == 0) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "p", std::to_string(p),
                    "p=+-inf sentinel cannot reduce over the whole (empty) tensor or an empty dim "
                    "(canndev constraint, spec.yaml shape_constraints)");
                return ge::GRAPH_FAILED;
            }
        }
    }

    // ── 输出 shape 构造（canndev InferShape4LpNorm 同构）──
    // keepdim=true：归约维以 size=1 保留，非归约维保留原值（含动态 -1 未知维），
    //              rank(y) = rank(x)；
    // keepdim=false：归约维从输出删除，rank(y) = rank(x) − 归约轴个数
    //              （axes 为空 → 标量 [] 输出）。
    std::vector<int64_t> yVec;
    yVec.reserve(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; i++) {
        if (reduceSet.find(i) != reduceSet.end()) {
            if (keepdim) {
                yVec.push_back(1);
            }
        } else {
            yVec.push_back(xShape->GetDim(static_cast<size_t>(i)));
        }
    }
    yShape->SetDimNum(yVec.size());
    for (size_t i = 0; i < yVec.size(); i++) {
        yShape->SetDim(i, yVec[i]);
    }
    return ge::GRAPH_SUCCESS;
}

// IMPL_OP_INFERSHAPE(LpNormReduce).InferShape(func):
//   Registers InferShape4LpNormReduce as the shape inference function
//   for the LpNormReduce operator type（注册链原样保留）。
IMPL_OP_INFERSHAPE(LpNormReduce).InferShape(InferShape4LpNormReduce);

} // namespace ops
