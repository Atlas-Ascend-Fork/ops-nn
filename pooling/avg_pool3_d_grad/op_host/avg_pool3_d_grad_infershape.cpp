/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace {
constexpr size_t IDX_0 = 0;
constexpr size_t IDX_1 = 1;
constexpr size_t kConv3dDimSizeLimit = 5;
constexpr int32_t UNKNOWN_SHAPE_DIM = -1;

inline bool IsConstTensor(const gert::Tensor* input_tensor)
{
    if (input_tensor != nullptr) {
        if (input_tensor->GetAddr() == nullptr) {
            // empty tensor
            return input_tensor->GetShapeSize() == 0;
        }
        return true;
    }
    return false;
}

constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

/*
 * @brief: check whether the output shape is unknown rank shape
 * @param [out] output_shape: the output shape ptr
 * @return bool
 */
inline bool IsUnknownShape(const gert::Shape* check_shape)
{
    for (size_t i = 0; i < check_shape->GetDimNum(); i++) {
        if (check_shape->GetDim(i) == UNKNOWN_DIM_VALUE_) {
            return true;
        }
    }
    return false;
}

using ge::GRAPH_FAILED;
using ge::graphStatus;
using gert::InferShapeContext;
} // namespace

namespace ops {

static graphStatus InferShapeForAvg(InferShapeContext* context, size_t const_tensor_idx, const char* const_tensor_name)
{
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed.", "context"), return GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Enter avgpool3dgrad shape infer.");
    const auto op_name = context->GetNodeName();
    auto y_shape = context->GetOutputShape(0);
    OP_CHECK_IF(y_shape == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed.", "y shape"), return GRAPH_FAILED);

    auto const_tensor = context->GetInputTensor(const_tensor_idx);
    OP_CHECK_IF(const_tensor == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null %s tensor.", const_tensor_name),
                return GRAPH_FAILED);
    const gert::Shape& const_tensor_ori_shape = const_tensor->GetOriginShape();

    size_t IDX = const_tensor_idx == IDX_0 ? IDX_1 : IDX_0;
    auto first_input_shape = context->GetInputShape(IDX);
    OP_CHECK_IF(first_input_shape == nullptr, CUBE_INNER_ERR_REPORT(op_name, "get null input tensor."),
                return GRAPH_FAILED);

    // aligned with canndev AvgPool3DGradVerify: orig_input_shape desc rank must be 1 or 5
    const size_t origRank = const_tensor_ori_shape.GetDimNum();
    OP_CHECK_IF(origRank != 1 && origRank != kConv3dDimSizeLimit,
                CUBE_INNER_ERR_REPORT(op_name, "%s dim num %zu invalid.", const_tensor_name, origRank),
                return GRAPH_FAILED);
    // aligned with canndev Verify: grads rank must be 5 or unknown rank
    const bool gradsIsUnknownRank = Ops::Base::IsUnknownRank(*first_input_shape);
    OP_CHECK_IF(first_input_shape->GetDimNum() != kConv3dDimSizeLimit && !gradsIsUnknownRank,
                CUBE_INNER_ERR_REPORT(op_name, "grads dim num %zu invalid.", first_input_shape->GetDimNum()),
                return GRAPH_FAILED);

    const bool origDyn = IsUnknownShape(&const_tensor_ori_shape) || Ops::Base::IsUnknownRank(const_tensor_ori_shape);
    if (!IsConstTensor(const_tensor) || origDyn) {
        // aligned with canndev dynamic branch: unknown-rank grads is intercepted when the output
        // can not be determined (is_dynamic); a fully-known rank-5 desc takes the desc dims without it
        const bool skipGradsCheck = (origRank == kConv3dDimSizeLimit) && !origDyn;
        OP_CHECK_IF(
            !skipGradsCheck && gradsIsUnknownRank,
            CUBE_INNER_ERR_REPORT(op_name, "grads is unknown rank while %s can not be resolved.", const_tensor_name),
            return GRAPH_FAILED);
        y_shape->SetDimNum(kConv3dDimSizeLimit);
        if (origRank == kConv3dDimSizeLimit) {
            // aligned with canndev: rank-5 desc, output takes the desc dims
            for (size_t idx = 0; idx < kConv3dDimSizeLimit; ++idx) {
                y_shape->SetDim(idx, const_tensor_ori_shape.GetDim(idx));
            }
        } else {
            for (size_t idx = 0; idx < kConv3dDimSizeLimit; ++idx) {
                y_shape->SetDim(idx, UNKNOWN_SHAPE_DIM);
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    // aligned with canndev: const data must carry exactly 5 values
    size_t const_tensor_dim_num = static_cast<size_t>(const_tensor_ori_shape.GetShapeSize());
    OP_CHECK_IF(const_tensor_dim_num != kConv3dDimSizeLimit,
                CUBE_INNER_ERR_REPORT(op_name, "%s dim num %zu invalid.", const_tensor_name, const_tensor_dim_num),
                return GRAPH_FAILED);
    y_shape->SetDimNum(kConv3dDimSizeLimit);

    auto datatype = const_tensor->GetDataType();
    if (datatype == ge::DT_INT32) {
        auto tensor_data = const_tensor->GetData<int32_t>();
        for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
            y_shape->SetDim(idx, tensor_data[idx]);
        }
    } else if (datatype == ge::DT_INT64) {
        auto tensor_data = const_tensor->GetData<int64_t>();
        for (size_t idx = 0; idx < const_tensor_dim_num; ++idx) {
            y_shape->SetDim(idx, tensor_data[idx]);
        }
    } else {
        CUBE_INNER_ERR_REPORT(op_name, "tensor %s not support dtype %s.", const_tensor_name,
                              ge::TypeUtils::DataTypeToAscendString(datatype).GetString());
        return GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "y_shape: %s.", Shape2String(*y_shape).c_str());
    OP_LOGD(context->GetNodeName(), "End avgpool3dgrad shape infer.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferShapeForAvgPool3DGrad(InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter avgpool3dgrad shape infer.");
    return InferShapeForAvg(context, 0, "orig_input_shape");
}

IMPL_OP_INFERSHAPE(AvgPool3DGrad)
    .InferShape(InferShapeForAvgPool3DGrad)
    .InputsDataDependency({0})
    .PrivateAttr("padding", "");
} // namespace ops
