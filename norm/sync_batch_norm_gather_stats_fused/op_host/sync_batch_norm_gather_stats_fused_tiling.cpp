/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sync_batch_norm_gather_stats_fused_tiling.cpp
 * \brief
 */

#include <iostream>
#include <string>
#include "op_host/tiling_util.h"
#include "sync_batch_norm_gather_stats_fused_tiling.h"

namespace optiling {
using namespace Ops::NN::OpTiling;
// Tiling阶段框架契约类失败(框架对象为空/平台信息异常/策略失败)统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "SyncBatchNormGatherStatsFused",
                                               reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

static ge::graphStatus Tiling4SyncBatchNormGatherStatsFused(gert::TilingContext* context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4SyncBatchNormGatherStatsFused(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4SyncBatchNormGatherStatsFused enter.");

    auto compileInfo = context->GetCompiledInfo<SyncBatchNormGatherStatsFusedCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr,
                ReportTilingStageError(
                    context->GetNodeName(),
                    "GetCompiledInfo: compile info is null; check the SoC-specific OPP package and its loading logs"),
                return ge::GRAPH_FAILED);

    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    compileInfo->vlFp32 = Ops::Base::GetVRegSize(context) / FLOAT_SIZE;

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        ReportTilingStageError(
            context->GetNodeName(),
            "GetPlatformInfo returned null; check the TilingParseContext construction and the platform registration"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                ReportTilingStageError(context->GetNodeName(), "GetCoreNumAiv returned " +
                                                                   std::to_string(compileInfo->coreNum) +
                                                                   "; check the platform info and SoC"),
                return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0),
                ReportTilingStageError(context->GetNodeName(), "GetCoreMemSize(UB) returned " +
                                                                   std::to_string(compileInfo->ubSizePlatForm) +
                                                                   "; check the platform info and SoC"),
                return ge::GRAPH_FAILED);
    compileInfo->isRegBase = IsRegbaseSocVersion(context) ? true : false;

    OP_LOGD(context->GetNodeName(),
            "TilingPrepare4SyncBatchNormGatherStatsFused exit, coreNum: %u, blockSize: %ld, ubSize: %ld, vlFp32: %ld.",
            compileInfo->coreNum, compileInfo->blockSize, compileInfo->ubSizePlatForm, compileInfo->vlFp32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SyncBatchNormGatherStatsFused)
    .Tiling(Tiling4SyncBatchNormGatherStatsFused)
    .TilingParse<SyncBatchNormGatherStatsFusedCompileInfo>(TilingPrepare4SyncBatchNormGatherStatsFused);
} // namespace optiling
