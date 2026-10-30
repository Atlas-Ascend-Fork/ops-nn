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
 * \file fused_weight_quant_mat_mul_onnx_plugin.cpp
 * \brief FusedWeightQuantMatMul 的 ONNX 算子插件注册（NPUFusedWeightQuantMatMul 导入）。
 *        属性布局与 proto 严格对齐：transpose_x/transpose_weight(Bool)、
 *        antiquant_group_size/dtype(Int)、fused_op_type(String, 默认 "relu")；
 *        不含 inner_precise（融合算子固定高精度模式）。
 */

#include "plugin_util.h"
#include "register/register.h"
#include "graph/operator.h"
#include "nlohmann/json.hpp"
#include "matmul/fused_weight_quant_mat_mul/common/fused_weight_quant_mat_mul_constants.h"

namespace domi {
using Json = nlohmann::json;

static Status ParseParamsFusedWeightQuantMatMul(const ge::Operator& opSrc, ge::Operator& opDst)
{
    int64_t antiquantGroupSize = 0;
    int64_t dtype = -1;
    std::string fusedOpType = ops::fused_weight_quant_mat_mul::FUSED_OP_TYPE_RELU;
    ge::AscendString attrsString;
    if (opSrc.GetAttr("attribute", attrsString) == ge::GRAPH_SUCCESS) {
        try {
            Json attrs = Json::parse(attrsString.GetString());
            if (attrs.contains("attribute") && attrs["attribute"].is_array()) {
                for (Json& attr : attrs["attribute"]) {
                    std::string attrName = attr.value("name", "");
                    if (attrName == "antiquant_group_size" && attr.contains("i")) {
                        antiquantGroupSize = attr["i"].get<int64_t>();
                        continue;
                    }
                    if (attrName == "dtype" && attr.contains("i")) {
                        dtype = attr["i"].get<int64_t>();
                        continue;
                    }
                    if (attrName == "fused_op_type" && attr.contains("s")) {
                        fusedOpType = attr["s"].get<std::string>();
                        continue;
                    }
                }
            }
        } catch (const nlohmann::json::exception& e) {
            OP_LOGE(GetOpName(opDst).c_str(), "JSON parse error: %s", e.what());
            return FAILED;
        } catch (...) {
            OP_LOGE(GetOpName(opDst).c_str(), "get unknown exception, please check compile info json.");
            return FAILED;
        }
    }
    // onnx doesn't have transpose attr
    opDst.SetAttr("transpose_x", false);
    opDst.SetAttr("transpose_weight", false);
    opDst.SetAttr("antiquant_group_size", antiquantGroupSize);
    opDst.SetAttr("dtype", dtype);
    // fused_op_type 缺省 "relu"（当前唯一支持的激活；非法值由 tiling checker 拒绝）
    opDst.SetAttr("fused_op_type", fusedOpType);
    return SUCCESS;
}

REGISTER_CUSTOM_OP("FusedWeightQuantMatMul")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::FusedWeightQuantMatMul"),
                   ge::AscendString("npu::1::NPUFusedWeightQuantMatMul"),
                   ge::AscendString("ai.onnx::11::NPUFusedWeightQuantMatMul"),
                   ge::AscendString("ai.onnx::12::NPUFusedWeightQuantMatMul"),
                   ge::AscendString("ai.onnx::13::NPUFusedWeightQuantMatMul"),
                   ge::AscendString("ai.onnx::14::NPUFusedWeightQuantMatMul"),
                   ge::AscendString("ai.onnx::15::NPUFusedWeightQuantMatMul"),
                   ge::AscendString("ai.onnx::16::NPUFusedWeightQuantMatMul"),
                   ge::AscendString("ai.onnx::17::NPUFusedWeightQuantMatMul"),
                   ge::AscendString("ai.onnx::18::NPUFusedWeightQuantMatMul"),
                   ge::AscendString("ai.onnx::19::NPUFusedWeightQuantMatMul")})
    .ParseParamsByOperatorFn(ParseParamsFusedWeightQuantMatMul)
    .ImplyType(ImplyType::TVM);
} // namespace domi
