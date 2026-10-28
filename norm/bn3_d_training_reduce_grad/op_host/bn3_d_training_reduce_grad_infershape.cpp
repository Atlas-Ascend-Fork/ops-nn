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
 * \file bn3_d_training_reduce_grad_infershape.cpp
 * \brief BN3DTrainingReduceGrad 的 shape 推导（InferShape）
 */

#include "register/op_impl_registry.h"             // IMPL_OP_INFERSHAPE macro
#include "exe_graph/runtime/infer_shape_context.h" // InferShapeContext, gert::Shape
#include "graph/types.h"                           // ge::UNKNOWN_DIM_NUM
#include "op_common/log/log.h"                     // OP_CHECK_NULL_WITH_CONTEXT macro

#include <string>

using namespace ge;

namespace ops {
namespace {

constexpr size_t kGradsIndex = 0U;
constexpr size_t kXIndex = 1U;
constexpr size_t kParamBegin = 2U; // diff_scale / diff_offset / scale / batch_mean / batch_variance
constexpr size_t kParamNum = 5U;

// 参数张量名，顺序与输入一致：diff_scale / diff_offset / scale / batch_mean / batch_variance。
const std::string kParamNames[kParamNum] = {"diff_scale", "diff_offset", "scale", "batch_mean", "batch_variance"};
constexpr size_t kRank5 = 5U;
constexpr size_t kChannelAxisNchw = 1U;  // NCDHW 通道轴 = dim1
constexpr size_t kChannelAxisNdhwc = 4U; // NDHWC 通道轴 = dim4

// V2 gert Shape 中 UNKNOWN_RANK 表示为 dim_num == 1 且 dim0 == UNKNOWN_DIM_NUM。
bool IsUnknownRank(const gert::Shape* shape)
{
    return shape->GetDimNum() == 1U && shape->GetDim(0) == ge::UNKNOWN_DIM_NUM;
}

// 维度值为负（-1 动态维 / -2 未知秩标记）视为未知，不参与数值比较。
bool IsKnownDim(const int64_t dim) { return dim >= 0; }

// 形状中任一维为 0 → 空 tensor。
bool HasZeroDim(const gert::Shape* shape)
{
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        if (shape->GetDim(i) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

static ge::graphStatus InferShape4BN3DTrainingReduceGrad(gert::InferShapeContext* context)
{
    // GetInputShape(0/1): grads 与 x 的 shape。
    const gert::Shape* grads_shape = context->GetInputShape(kGradsIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, grads_shape);
    const gert::Shape* x_shape = context->GetInputShape(kXIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    // GetOutputShape(0): output y's shape descriptor.
    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    // y.shape 依赖未知 rank 输入时传播 UNKNOWN_RANK（动态签名场景）。
    if (IsUnknownRank(grads_shape)) {
        *output_shape = *grads_shape;
        return ge::GRAPH_SUCCESS;
    }

    // 校验 1: rank 校验（grads / x 必须 5D）。
    if (grads_shape->GetDimNum() != kRank5 || (!IsUnknownRank(x_shape) && x_shape->GetDimNum() != kRank5)) {
        return ge::GRAPH_FAILED;
    }
    // 参数张量必须 1D（未知秩参数跳过，运行时由 tiling 复核）。
    for (size_t i = 0; i < kParamNum; ++i) {
        const gert::Shape* param_shape = context->GetInputShape(kParamBegin + i);
        OP_CHECK_NULL_WITH_CONTEXT(context, param_shape);
        if (IsUnknownRank(param_shape)) {
            continue;
        }
        if (param_shape->GetDimNum() != 1U) {
            return ge::GRAPH_FAILED;
        }
    }

    // 校验 2: 空 tensor 校验（任一维为 0 → null_input）。
    if (HasZeroDim(grads_shape) || (!IsUnknownRank(x_shape) && HasZeroDim(x_shape))) {
        return ge::GRAPH_FAILED;
    }

    // 校验 3: grads 与 x 同 shape（未知维仅跳过比较）。
    if (!IsUnknownRank(x_shape)) {
        if (x_shape->GetDimNum() != grads_shape->GetDimNum()) {
            return ge::GRAPH_FAILED;
        }
        for (size_t i = 0; i < grads_shape->GetDimNum(); ++i) {
            if (IsKnownDim(grads_shape->GetDim(i)) && IsKnownDim(x_shape->GetDim(i)) &&
                grads_shape->GetDim(i) != x_shape->GetDim(i)) {
                return ge::GRAPH_FAILED;
            }
        }
    }

    // 校验 4: 参数长度 = C。通道轴由 grads 的 origin format 决定：
    //   NCDHW → dim1；NDHWC → dim4（与 tiling 的 format 判定保持一致）。
    //   format 可靠（GE 图执行时恒为上述两者之一），且 tiling 已强制
    //   “format must be NCDHW/NDHWC”，故 infershape 只需按单轴判定，无需
    //   “dim1 || dim4” 的或兜底；未知 format 直接拒绝，tiling 侧仍保留
    //   format 校验作为第二道防线。
    const gert::CompileTimeTensorDesc* grads_desc = context->GetInputDesc(kGradsIndex);
    const ge::Format grads_format = (grads_desc == nullptr) ?
                                        ge::FORMAT_ND :
                                        static_cast<ge::Format>(ge::GetPrimaryFormat(grads_desc->GetOriginFormat()));
    int64_t channel = -1;
    if (grads_format == ge::FORMAT_NCDHW) {
        channel = grads_shape->GetDim(kChannelAxisNchw);
    } else if (grads_format == ge::FORMAT_NDHWC) {
        channel = grads_shape->GetDim(kChannelAxisNdhwc);
    } else {
        // 未知/ND format：infershape 直接拒绝。描述缺失时 format 已默认 ND，日志如实反映。
        const std::string actualFormat = Ops::Base::ToString(grads_format);
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "grads", actualFormat.c_str(), "NCDHW or NDHWC");
        return ge::GRAPH_FAILED;
    }
    if (IsKnownDim(channel)) {
        for (size_t i = 0; i < kParamNum; ++i) {
            const gert::Shape* param_shape = context->GetInputShape(kParamBegin + i);
            OP_CHECK_NULL_WITH_CONTEXT(context, param_shape);
            if (IsUnknownRank(param_shape)) {
                continue;
            }
            const int64_t param_len = param_shape->GetDim(0);
            if (!IsKnownDim(param_len)) {
                continue;
            }
            if (param_len != channel) {
                const std::string actualShape = std::to_string(param_len);
                const std::string reason = "The 0th axis of input " + kParamNames[i] +
                                           " must be equal to the C axis of input grads";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), kParamNames[i].c_str(),
                                                       actualShape.c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }

    // 输出推导: y.shape = grads.shape（5D 逐维复制）。
    *output_shape = *grads_shape;

    return ge::GRAPH_SUCCESS;
}

// IMPL_OP_INFERSHAPE(BN3DTrainingReduceGrad).InferShape(func):
//   Registers InferShape4BN3DTrainingReduceGrad as the shape inference function
//   for the BN3DTrainingReduceGrad operator type.
//   To create FooBar: change BN3DTrainingReduceGrad → FooBar.
IMPL_OP_INFERSHAPE(BN3DTrainingReduceGrad).InferShape(InferShape4BN3DTrainingReduceGrad);

} // namespace ops
