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
 * \file conv2d_v2_base_tiling_aswt.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV2D_V2_BASE_TILING_ASWT_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV2D_V2_BASE_TILING_ASWT_H

#include "conv2d_v2_base_tiling.h"

namespace optiling {
namespace conv_ops_tiling {

constexpr static uint8_t TAIL_REGIONS = 3;
constexpr static uint8_t TAIL_REGION0 = 0;
constexpr static uint8_t TAIL_REGION1 = 1;
constexpr static uint8_t TAIL_REGION2 = 2;
constexpr static uint8_t MAX_DEPTH_RECURSION = 100;

struct tailRegion {
    uint64_t standardBatch = 0;
    uint64_t oriM = 0;
    uint64_t oriN = 0;
    uint64_t tailBatch = 0;
    uint64_t tailOriN = 0;
};

} // namespace conv_ops_tiling
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CONV2D_V2_BASE_TILING_ASWT_H
