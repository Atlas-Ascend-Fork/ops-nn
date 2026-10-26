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
 * \file batch_norm3_d_grad_graph_infer.cpp
 * \brief BatchNorm3DGrad graph infer registration.
 */

#include <vector>

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ge {
IMPLEMT_COMMON_INFERFUNC(BatchNorm3DGradLegacyV1InferShape)
{
    const TensorDesc dyDesc = op.GetInputDescByName("y_backprop");
    const TensorDesc scaleDesc = op.GetInputDescByName("scale");

    // x_backprop <- y_backprop（shape + dtype）
    TensorDesc dxDesc = op.GetOutputDescByName("x_backprop");
    dxDesc.SetShape(dyDesc.GetShape());
    dxDesc.SetDataType(dyDesc.GetDataType());
    if (op.UpdateOutputDesc("x_backprop", dxDesc) != GRAPH_SUCCESS) {
        OP_LOGE("BatchNorm3DGrad", "Update output desc x_backprop failed.");
        return GRAPH_FAILED;
    }

    // scale_backprop / offset_backprop <- scale（shape + dtype）
    for (const char* outName : {"scale_backprop", "offset_backprop"}) {
        TensorDesc outDesc = op.GetOutputDescByName(outName);
        outDesc.SetShape(scaleDesc.GetShape());
        outDesc.SetDataType(scaleDesc.GetDataType());
        if (op.UpdateOutputDesc(outName, outDesc) != GRAPH_SUCCESS) {
            OP_LOGE("BatchNorm3DGrad", "Update output desc %s failed.", outName);
            return GRAPH_FAILED;
        }
    }

    // reserve_space_4 / reserve_space_5 <- {0}
    for (const char* outName : {"reserve_space_4", "reserve_space_5"}) {
        TensorDesc outDesc = op.GetOutputDescByName(outName);
        outDesc.SetShape(Shape(std::vector<int64_t>{0}));
        outDesc.SetDataType(scaleDesc.GetDataType());
        if (op.UpdateOutputDesc(outName, outDesc) != GRAPH_SUCCESS) {
            OP_LOGE("BatchNorm3DGrad", "Update output desc %s failed.", outName);
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

// 静态 GEIR 图仍会走内置算子对应的 V1 注册表（nn_batch_norm_ops.cc 的
// BatchNorm3DGradInferShape 只接受 NDHWC/NCDHW，拒绝 tiling 实际支持的
// NCHW/NHWC 4D 用例）。本包先于 libopsproto.so 加载，因此在此注册与
// runtime2.0 语义一致的推导实现：不做 data_format 字符串校验，形状照抄输入，
// 占位输出固定为 {0}。
COMMON_INFER_FUNC_REG(BatchNorm3DGrad, BatchNorm3DGradLegacyV1InferShape);
} // namespace ge
