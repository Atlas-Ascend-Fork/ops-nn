/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "tiling/platform/platform_ascendc.h"

namespace optiling::confidence_head_fused {

struct ConfidenceHeadFusedCompileInfo {
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::RESERVED_VERSION;
    uint64_t aicNum = 0;
    uint64_t aivNum = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
    uint64_t ubSize = 0;
};

inline void GetConfidenceHeadFusedPlatformInfo(platform_ascendc::PlatformAscendC& platform,
                                               ConfidenceHeadFusedCompileInfo& info)
{
    info.socVersion = platform.GetSocVersion();
    info.aicNum = platform.GetCoreNumAic();
    info.aivNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, info.l1Size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, info.l0aSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, info.l0bSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, info.l0cSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);
}

} // namespace optiling::confidence_head_fused
