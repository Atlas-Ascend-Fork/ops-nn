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
 * \file fake_quant_with_min_max_vars_per_channel_infershape.cpp
 * \brief FakeQuantWithMinMaxVarsPerChannel 形状/dtype 推导
 *
 * y.shape = x.shape
 * y.dtype = x.dtype (fp32)
 *
 * 约束校验（README「约束说明」）：x rank∈[1,8]；min/max rank==1 且长度==x 末维 C 且两者等长；
 * num_bits∈[2,16]；输入 dtype 仅 fp32。与同族 fake_quant_with_min_max_vars_per_channel_gradient
 * 的 InferShape 惯例一致：未知 rank/未知维延后到 tiling 用具体 shape 复核。
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {

static constexpr size_t kMaxRank = 8U;
static constexpr int64_t kNumBitsMin = 2;
static constexpr int64_t kNumBitsMax = 16;

static ge::graphStatus InferShape4FakeQuantWithMinMaxVarsPerChannel(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* minShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, minShape);
    const gert::Shape* maxShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // x 已知 rank（含非法的 rank 0 标量）时必须落在 [1, kMaxRank]
    const bool xUnknownRank = Ops::Base::IsUnknownRank(*xShape);
    const size_t xRank = xShape->GetDimNum();
    if (!xUnknownRank && (xRank < 1U || xRank > kMaxRank)) {
        OP_LOGE(context, "FakeQuantPC: x rank %zu out of range [1, %zu]", xRank, kMaxRank);
        return ge::GRAPH_FAILED;
    }
    // min/max 是否 1-D 与 x 的 rank 是否未知无关，可判定即必须校验
    const bool minUnknown = Ops::Base::IsUnknownRank(*minShape);
    const bool maxUnknown = Ops::Base::IsUnknownRank(*maxShape);
    if ((!minUnknown && minShape->GetDimNum() != 1U) || (!maxUnknown && maxShape->GetDimNum() != 1U)) {
        OP_LOGE(context, "FakeQuantPC: min/max must be 1-D, got min rank %zu max rank %zu", minShape->GetDimNum(),
                maxShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    // P1-2(fix7)：min/max 等长校验只要求 min/max 长度可判定（!minUnknown && !maxUnknown），
    // 不依赖 x 秩/末维是否已知——x 为未知秩(-2)而 min/max 已知不等长时也须在 Host 拒绝
    if (!minUnknown && !maxUnknown) {
        const int64_t minLen = minShape->GetDim(0);
        const int64_t maxLen = maxShape->GetDim(0);
        if (minLen >= 0 && maxLen >= 0 && minLen != maxLen) {
            OP_LOGE(context, "FakeQuantPC: min length (%ld) != max length (%ld)", minLen, maxLen);
            return ge::GRAPH_FAILED;
        }
        // 末维已知时校验 min/max 长度等于 C
        if (!xUnknownRank) {
            const int64_t channel = xShape->GetDim(static_cast<int64_t>(xRank) - 1);
            if (channel >= 0 && minLen >= 0 && maxLen >= 0 && (minLen != channel || maxLen != channel)) {
                OP_LOGE(context, "FakeQuantPC: min/max length (%ld/%ld) must equal channel C (%ld)", minLen, maxLen,
                        channel);
                return ge::GRAPH_FAILED;
            }
        }
    }

    // num_bits 为可选属性，默认 8；注册类型为 Int(int64)
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() >= 1) {
        const int64_t* numBits = attrs->GetAttrPointer<int64_t>(0);
        if (numBits != nullptr && (*numBits < kNumBitsMin || *numBits > kNumBitsMax)) {
            OP_LOGE(context, "FakeQuantPC: num_bits %ld out of range [%ld, %ld]", *numBits, kNumBitsMin, kNumBitsMax);
            return ge::GRAPH_FAILED;
        }
    }

    *yShape = *xShape;
    OP_LOGI(context->GetNodeName(), "FakeQuantWithMinMaxVarsPerChannel output shape: %s.",
            Ops::Base::ToString(*yShape).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4FakeQuantWithMinMaxVarsPerChannel(gert::InferDataTypeContext* context)
{
    // 输入仅支持 FLOAT32（README 约束说明，与 op_host def 注册一致）。
    // DT_UNDEFINED 表示图构建期 dtype 尚未确定，留给后续阶段，不在此判失败。
    for (size_t index = 0U; index < 3U; ++index) {
        const ge::DataType dtype = context->GetInputDataType(index);
        if (dtype != ge::DT_UNDEFINED && dtype != ge::DT_FLOAT) {
            OP_LOGE(context, "FakeQuantPC: input %zu dtype %d is not supported, only float32", index,
                    static_cast<int>(dtype));
            return ge::GRAPH_FAILED;
        }
    }
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FakeQuantWithMinMaxVarsPerChannel)
    .InferShape(InferShape4FakeQuantWithMinMaxVarsPerChannel)
    .InferDataType(InferDataType4FakeQuantWithMinMaxVarsPerChannel);

} // namespace ops
