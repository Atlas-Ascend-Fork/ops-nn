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
 * \file biasadd_conv_fusion_pass.h
 * \brief conv + biasadd/add --> conv with bias
 */
#ifndef BIASADD_CONV_FUSION_PASS_H
#define BIASADD_CONV_FUSION_PASS_H

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace BiasaddConvFusion {

const std::string FUSION_NAME = "AABiasaddConvFusion";
const std::string INTRINSIC_FIX_PIPE_L0C2OUT = "Intrinsic_fix_pipe_l0c2out";
const std::string RESHAPE_SINGLE_OUTPUT_PATTERN = "AABiasaddConvFusion_Conv3D_Add_R_single_output";
const std::string RESHAPE_DUAL_OUTPUT_PATTERN = "AABiasaddConvFusion_Conv3D_Add_R_dual_output";
const std::string REVERSED_ADD_PATTERN_SUFFIX = "_Rev";

constexpr int32_t CONV_CAPTURE = 0;
constexpr int32_t ADD_CAPTURE = 1;
constexpr int32_t RESHAPE_CAPTURE = 2;
constexpr int32_t PATTERN_SHAPE_INPUT_INDEX = 3;

constexpr size_t NHWC_CHANNEL_AXIS = 3U;
constexpr size_t HWCN_CHANNEL_AXIS = 2U;
constexpr size_t NDHWC_CHANNEL_AXIS = 4U;
constexpr int64_t MIN_ONE_DIM_COUNT = 4;
constexpr int64_t UNKNOWN_DIM_SENTINEL = -2;

const std::set<ge::DataType> supportedBiasDtypes = {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT,
                                                    ge::DataType::DT_BF16, ge::DataType::DT_INT32};

const std::vector<std::string> convTypes = {"Conv2D", "DepthwiseConv2D", "Conv3D"};
const std::vector<std::string> addTypes = {"Add", "BiasAdd"};

} // namespace BiasaddConvFusion

class __attribute__((visibility("default"))) AABiasaddConvFusion : public ge::fusion::PatternFusionPass {
protected:
    std::vector<ge::fusion::PatternUniqPtr> Patterns() override;
    bool MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;
    std::unique_ptr<ge::Graph> Replacement(const std::unique_ptr<ge::fusion::MatchResult>& matchResult) override;
};

} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // BIASADD_CONV_FUSION_PASS_H
