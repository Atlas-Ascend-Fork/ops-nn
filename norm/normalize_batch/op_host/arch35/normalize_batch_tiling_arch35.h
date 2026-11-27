/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Tiling compile-time info header for arch35
//   (ascend950 / ascend960dt / ascend960pr / ascend350).
//   NormalizeBatchCompileInfo 为 TilingParse 载体（空结构，范式同 l2_normalize）：
//   平台参数（AIV 核数 / UB 大小 / block / VL）每次经 TilingFunc 内
//   GetPlatformInfo() 现取，不挂 CompileInfo 缓存。

#ifndef OPS_NN_NORMALIZE_BATCH_TILING_ARCH35_H
#define OPS_NN_NORMALIZE_BATCH_TILING_ARCH35_H

#include "../../op_kernel/arch35/normalize_batch_tiling_struct.h"

namespace optiling {

// NormalizeBatchCompileInfo — TilingParse 载体（空结构）
struct NormalizeBatchCompileInfo {};

} // namespace optiling

#endif // OPS_NN_NORMALIZE_BATCH_TILING_ARCH35_H
