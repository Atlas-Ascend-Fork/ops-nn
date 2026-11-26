/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gemmv3_tiling_strategy.h
 * \brief
 */
#pragma once

#include <map>
#include <vector>
#include <cstdint>

#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "op_common/op_host/util/op_const_def.h"

namespace optiling {
namespace gemmv3 {
namespace strategy {
constexpr int32_t BASIC_ASWT = 4; // 与 matmul_v3_advanced::strategy::BASIC_ASWT 保持一致

// A5/A6: GemmV3 走 Blaze(TENSOR_LEVEL) ASWT
const static std::map<NpuArch, std::vector<int32_t>> Gemmv3PrioritiesMap = {
    {Ops::Base::DAV_3510, {strategy::BASIC_ASWT}},
    {::Ops::Base::DAV_9201, {strategy::BASIC_ASWT}},
    {::Ops::Base::DAV_9202, {strategy::BASIC_ASWT}}};

inline std::vector<int32_t> GetGemmV3Priorities(NpuArch npuArch)
{
    std::vector<int32_t> priorities = {};
    if (Gemmv3PrioritiesMap.find(npuArch) != Gemmv3PrioritiesMap.end()) {
        priorities = Gemmv3PrioritiesMap.at(npuArch);
    }
    return priorities;
};
} // namespace strategy
} // namespace gemmv3
} // namespace optiling
