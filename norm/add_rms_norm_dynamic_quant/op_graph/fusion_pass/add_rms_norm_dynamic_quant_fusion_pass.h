/** Copyright (c) 2026 Huawei Technologies Co., Ltd. See LICENSE for licensing details. */
#ifndef OPS_NN_ADD_RMS_NORM_DYNAMIC_QUANT_FUSION_PASS_H
#define OPS_NN_ADD_RMS_NORM_DYNAMIC_QUANT_FUSION_PASS_H
#include "ge/fusion/pass/fusion_base_pass.h"
namespace ops {
class __attribute__((visibility("default"))) AddRmsNormDynamicQuantFusionPass : public ge::fusion::FusionBasePass {
public:
    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& context) override;
};
} // namespace ops
#endif
