/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CONV2DT_QUANT_PROCESS_FUSION_PASS_H_
#define CONV2DT_QUANT_PROCESS_FUSION_PASS_H_

#include "fusion_pass/bias_optimize_quant_rollback_base.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace ops {
class __attribute__((visibility("default"))) Conv2DTQuantProcessFusionPass : public ge::fusion::FusionBasePass,
                                                                             public BiasOptimizeQuantRollbackBase {
public:
    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext) override;

protected:
    uint32_t GetBiasInputIndex() const override { return 3U; }
    uint32_t GetInputSizeContainsBias() const override { return 4U; }
    uint32_t GetWeightInputIndex() const override { return 2U; }
    uint32_t GetBiasOptimizationModeIndex() const override
    {
        return static_cast<uint32_t>(HostOpMode::BIAS_OPTIMIZATION_MODE_CONVDT);
    }
    uint32_t GetWeightRollBackModeIndex() const override
    {
        return static_cast<uint32_t>(HostOpMode::WEIGHT_ROLL_BACK_MODE_CONVDT);
    }
    uint32_t GetBiasRollBackModeIndex() const override
    {
        return static_cast<uint32_t>(HostOpMode::BIAS_ROLL_BACK_MODE_CONVDT);
    }
    ge::Status GetQuantProcessMode(const ge::GNodePtr& quantNode, const ge::GNodePtr& cubeNode,
                                   QuantProcessMode& quantProcessMode) override;
    void SetCinCoutReverse(const ge::GNodePtr& node) override;
    bool CheckSoc() const;
};
} // namespace ops

#endif // CONV2DT_QUANT_PROCESS_FUSION_PASS_H_
