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
 * \file apply_adagrad_to_inplace_fusion_pass.h
 * \brief ApplyAdagrad -> InplaceApplyAdagrad GEIR graph rewrite pass.
 */

#ifndef OPTIM_APPLY_ADAGRAD_TO_INPLACE_FUSION_PASS_H
#define OPTIM_APPLY_ADAGRAD_TO_INPLACE_FUSION_PASS_H

#include <memory>
#include <vector>

#include "ge/fusion/pass/decompose_pass.h"

namespace ops {

class __attribute__((visibility("default"))) ApplyAdagradToInplaceFusionPass : public ge::fusion::DecomposePass {
public:
    explicit ApplyAdagradToInplaceFusionPass(const std::vector<ge::AscendString>& opTypes) : DecomposePass(opTypes) {}

protected:
    bool MeetRequirements(const ge::GNode& matchedNode) override;

    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& matchedNode) override;
};

} // namespace ops

#endif // OPTIM_APPLY_ADAGRAD_TO_INPLACE_FUSION_PASS_H
