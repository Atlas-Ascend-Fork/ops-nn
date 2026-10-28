/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/confidence_head_fused_tiling.h"
#include "arch35/confidence_head_fused_compile_info.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
using confidence_head_fused::ConfidenceHeadFusedCompileInfo;
using confidence_head_fused::ConfidenceHeadFusedTiling;

REGISTER_TILING_TEMPLATE("ConfidenceHeadFused", ConfidenceHeadFusedTiling, 0);

static ge::graphStatus ConfidenceHeadFusedTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE("ConfidenceHeadFused", "context is null."), return ge::GRAPH_FAILED);
    return ConfidenceHeadFusedTiling(context).DoTiling();
}

static ge::graphStatus TilingPrepareForConfidenceHeadFused(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE("ConfidenceHeadFused", "context is null."), return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<ConfidenceHeadFusedCompileInfo>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    confidence_head_fused::GetConfidenceHeadFusedPlatformInfo(platform, *compileInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ConfidenceHeadFused)
    .Tiling(ConfidenceHeadFusedTilingFunc)
    .TilingParse<ConfidenceHeadFusedCompileInfo>(TilingPrepareForConfidenceHeadFused);

} // namespace optiling
