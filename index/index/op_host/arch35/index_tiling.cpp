/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_tiling.cpp
 * \brief Index operator tiling entry and registration
 */

#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "index_tiling.h"

using namespace ge;
using Ops::NN::Optiling::TilingRegistry;

namespace optiling {

uint64_t IndexTilingCommon::ParamsDtypeImprove(uint64_t lastDimSize, uint64_t dataTypeBytes)
{
    uint64_t lastAxisByte = lastDimSize * dataTypeBytes;
    if ((dataTypeBytes < DTYPE_SIZE_B128) && ((lastAxisByte % DTYPE_SIZE_B128) == 0)) {
        OP_LOGD(context_->GetNodeName(), "ParamsDtypeImprove lastAxisByte %lu, improve to DTYPE_SIZE_B128",
                lastAxisByte);
        return DTYPE_SIZE_B128 / dataTypeBytes;
    }

    if ((dataTypeBytes < DTYPE_SIZE_B64) && ((lastAxisByte % DTYPE_SIZE_B64) == 0)) {
        OP_LOGD(context_->GetNodeName(), "ParamsDtypeImprove lastAxisByte %lu, improve to DTYPE_SIZE_B64",
                lastAxisByte);
        return DTYPE_SIZE_B64 / dataTypeBytes;
    }

    if ((dataTypeBytes < DTYPE_SIZE_B32) && ((lastAxisByte % DTYPE_SIZE_B32) == 0)) {
        OP_LOGD(context_->GetNodeName(), "ParamsDtypeImprove lastAxisByte %lu, improve to DTYPE_SIZE_B32",
                lastAxisByte);
        return DTYPE_SIZE_B32 / dataTypeBytes;
    }

    if ((dataTypeBytes < DTYPE_SIZE_B16) && ((lastAxisByte % DTYPE_SIZE_B16) == 0)) {
        OP_LOGD(context_->GetNodeName(), "ParamsDtypeImprove lastAxisByte %lu, improve to DTYPE_SIZE_B16",
                lastAxisByte);
        return DTYPE_SIZE_B16 / dataTypeBytes;
    }
    return 0;
}

uint64_t IndexTilingCommon::ImproveLastAxisDtype(uint64_t* inputShapes, uint32_t inputDimNum, uint64_t& inputLength,
                                                 uint64_t& outputLength)
{
    uint64_t dataTypeBytes = GenXDtype();
    if (dataTypeBytes == 0UL) {
        auto firstInput = context_->GetInputDesc(0);
        if (firstInput != nullptr) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), "x", Ops::Base::ToString(firstInput->GetDataType()).c_str(),
                "x dtype not supported, must be in [DT_INT64, DT_INT32, DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, "
                "DT_UINT8, DT_BOOL, DT_COMPLEX64]");
        }
        return 0;
    }
    uint64_t factor = ParamsDtypeImprove(inputShapes[inputDimNum - 1], dataTypeBytes);
    if (factor) {
        inputLength /= factor;
        outputLength /= factor;
        inputShapes[inputDimNum - 1] /= factor;
    }
    return factor ? factor : 1;
}

static ge::graphStatus Tiling4Index(gert::TilingContext* context)
{
    OP_LOGD("Index", "Tiling4Index rt2.0 is running.");
    auto compile_info = static_cast<const IndexCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    OP_LOGD(context->GetNodeName(), "Tiling4Index dsl compile_info is Null, running Simt tiling.");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareIndexForAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start init IndexSimtTiling.");
    auto ci = context->GetCompiledInfo<IndexCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (ci->core_num <= 0),
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "core_num", std::to_string(ci->core_num).c_str(), "> 0"),
        return ge::GRAPH_FAILED);
    uint64_t indexUbSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, indexUbSize);
    ci->ubSize = static_cast<int64_t>(indexUbSize);
    OP_CHECK_IF((ci->ubSize <= 0),
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "ubSize", std::to_string(ci->ubSize).c_str(), "> 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4Index(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<IndexCompileInfo>();
    OP_CHECK_IF(compile_info == nullptr,
                OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(context->GetNodeName(), "tiling", "compile_info", "null",
                                                       "compile info cannot be null"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "AscendC TilingPrepare4Index Simt Mode success!");
    auto ret = TilingPrepareIndexForAscendC(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Index)
    .Tiling(Tiling4Index)
    .TilingInputsDataDependency({1})
    .TilingParse<IndexCompileInfo>(TilingPrepare4Index);
} // namespace optiling
