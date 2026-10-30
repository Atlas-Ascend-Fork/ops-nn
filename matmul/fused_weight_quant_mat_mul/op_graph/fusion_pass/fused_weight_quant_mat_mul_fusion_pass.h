/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NN_FUSED_WEIGHT_QUANT_MAT_MUL_FUSION_PASS_H
#define NN_FUSED_WEIGHT_QUANT_MAT_MUL_FUSION_PASS_H

#include "ge/fusion/pass/fusion_base_pass.h"

namespace ops {

// WQBMMV2 + Relu → FusedWeightQuantMatMul（仅支持 mmad s8s4 的平台）
class __attribute__((visibility("default"))) FusedWeightQuantMatMulFusionPass : public ge::fusion::FusionBasePass {
public:
    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext) override;
};

} // namespace ops

#endif // NN_FUSED_WEIGHT_QUANT_MAT_MUL_FUSION_PASS_H
