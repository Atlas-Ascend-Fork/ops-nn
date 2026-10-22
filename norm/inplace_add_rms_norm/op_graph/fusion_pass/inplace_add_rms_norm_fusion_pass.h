/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NORM_INPLACE_ADD_RMS_NORM_FUSION_PASS_H_
#define OPS_NORM_INPLACE_ADD_RMS_NORM_FUSION_PASS_H_

#include "ge/fusion/pass/fusion_base_pass.h"

namespace ops {
class __attribute__((visibility("default"))) InplaceAddRmsNormFusionPass : public ge::fusion::FusionBasePass {
public:
    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& context) override;
};
} // namespace ops

#endif // OPS_NORM_INPLACE_ADD_RMS_NORM_FUSION_PASS_H_
