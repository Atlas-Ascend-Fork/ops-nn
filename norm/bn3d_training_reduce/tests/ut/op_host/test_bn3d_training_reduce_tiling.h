/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_BN3D_TRAINING_REDUCE_H_
#define OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_BN3D_TRAINING_REDUCE_H_

#include <cstdint>
#include "../../../op_kernel/arch35/bn3d_training_reduce_tiling_data.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {

// 与 op_host/bn3d_training_reduce_tiling.h 的 BN3DTrainingReduceCompileInfo 布局一致。
// tiling_parse_func 把解析结果写入此结构；测试侧独立声明以避免与 op_host 头重复包含冲突
// （对齐 in_training_reduce_v2 的 test_in_training_reduce_v2_tiling.h 做法）。
struct BN3DTrainingReduceCompileInfo {
    uint64_t coreNum;      // AIV 核数
    uint64_t ubSize;       // UB 空间
    uint32_t vectorLength; // 向量寄存器字节宽度
    uint64_t ubBlockSize;  // 32B，UB 的字节对齐单位
};

using ::BN3DTrainingReduceDenseChannelTilingData;

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TEST_TILING_RUNTIME_BN3D_TRAINING_REDUCE_H_
