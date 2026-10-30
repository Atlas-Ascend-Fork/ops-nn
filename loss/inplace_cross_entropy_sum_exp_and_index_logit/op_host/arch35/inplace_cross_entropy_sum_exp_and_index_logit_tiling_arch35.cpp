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
 * \file inplace_cross_entropy_sum_exp_and_index_logit_tiling_arch35.cpp
 * \brief A5 (ascend950) inplace tiling wrapper — reuses base tiling class with isInplace=true
 */

#include "inplace_cross_entropy_sum_exp_and_index_logit_tiling_arch35.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

ge::graphStatus TilingPrepareForInplaceCrossEntropySumExpAndIndexLogit(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForInplaceCrossEntropySumExpAndIndexLogit start.");
    auto compileInfo = context->GetCompiledInfo<CrossEntropySumExpAndIndexLogitCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "TilingPrepareForInplaceCrossEntropySumExpAndIndexLogit end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingInplaceCrossEntropySumExpAndIndexLogit(gert::TilingContext* context)
{
    OP_LOGD(context, "InplaceCrossEntropySumExpAndIndexLogit tiling entry (isInplace=true).");
    CrossEntropySumExpAndIndexLogitTiling tilingObject(context, true);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

IMPL_OP_OPTILING(InplaceCrossEntropySumExpAndIndexLogit)
    .Tiling(TilingInplaceCrossEntropySumExpAndIndexLogit)
    .TilingParse<CrossEntropySumExpAndIndexLogitCompileInfo>(TilingPrepareForInplaceCrossEntropySumExpAndIndexLogit);

} // namespace optiling
