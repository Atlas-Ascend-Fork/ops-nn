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
 * \file repeat_interleave_tiling_arch35.cpp
 * \brief
 */
#include "repeat_interleave_tiling_arch35.h"
#include "repeat_interleave_tiling_normal.h"
#include "repeat_interleave_tiling_repeat.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"
#include "../../../../matmul/common/op_host/op_tiling/tiling_cache.h"
#include "op_api/runtime2_util_nn.h"
#include "../../../../matmul/common/op_host/op_tiling/hash.h"

using Ops::NN::Optiling::TilingRegistry;

namespace optiling {
constexpr size_t ATTR_AXIS_IDX = 0;
constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t OUTPUT_INDEX_Y = 0;
constexpr size_t INPUT_INDEX_REPEATS = 1;
constexpr int64_t ELEMENT_SIZE = 32;
constexpr size_t SUPPORT_DATA_MOVE_PAD = 1;
constexpr size_t SUPPORT_VECTOR_DUP = 11;

ge::graphStatus RepeatInterleaveTilingForAscendC(gert::TilingContext* context)
{
    OP_CHECK_IF((context == nullptr), OP_LOGE_WITHOUT_REPORT("RepeatInterleave", "Tiling context is null"),
                return ge::GRAPH_FAILED);
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareRepeatInterleaveForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareRepeatInterleaveForAscendC entering.");
    auto compileInfo = GetCompileInfoPtr<RepeatInterleaveCompileInfo>(context);
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Missing compile information"),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Missing platform information"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNumAscendc = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->coreNumAscendc <= 0) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Invalid platform coreNum: %ld",
                               static_cast<int64_t>(compileInfo->coreNumAscendc));
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSizeAscendc = static_cast<int64_t>(ubSize);
    if (compileInfo->ubSizeAscendc <= 0) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Invalid platform ubSize: %ld",
                               static_cast<int64_t>(compileInfo->ubSizeAscendc));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(RepeatInterleave)
    .Tiling(RepeatInterleaveTilingForAscendC)
    .TilingParse<RepeatInterleaveCompileInfo>(TilingPrepareRepeatInterleaveForAscendC);
} // namespace optiling
