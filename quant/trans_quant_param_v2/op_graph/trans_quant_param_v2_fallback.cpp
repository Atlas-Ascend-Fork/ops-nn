/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_fallback.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
// Tiling阶段框架契约类失败(框架对象为空/平台信息异常/策略失败)统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "TransQuantParamV2", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

using namespace ge;
using namespace gert;
constexpr size_t INPUTSCALE_INDEX = 0;
constexpr size_t INPUTOFFSET_INDEX = 1;
constexpr size_t OUTPUT_INDEX = 0;

static graphStatus TransQuantParamV2ExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_CHECK_IF(
        host_api_ctx == nullptr,
        ReportTilingStageError("TransQuantParamV2",
                               "fallback: OpExecuteContext is null; the host fallback framework contract is broken"),
        return GRAPH_FAILED);
    auto scale = host_api_ctx->GetInputTensor(INPUTSCALE_INDEX);
    OP_CHECK_IF(scale == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback trans_quant_param_v2", "scale",
                                                         "the input tensor of scale is missing in the fallback execute "
                                                         "context; check that the caller provides all required inputs"),
                return GRAPH_FAILED);
    auto offset = host_api_ctx->GetOptionalInputTensor(INPUTOFFSET_INDEX);
    auto output = host_api_ctx->GetOutputTensor(OUTPUT_INDEX);
    OP_CHECK_IF(
        output == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback trans_quant_param_v2", "y",
                                                 "the output tensor of y is missing in the fallback execute context; "
                                                 "check that the caller provides all required outputs"),
        return GRAPH_FAILED);
    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        ReportTilingStageError("TransQuantParamV2",
                               "fallback: GetAttrs() returned null; check the fallback execute context construction"),
        return GRAPH_FAILED);
    int64_t roundMode = (attrs->GetInt(0) == nullptr) ? 0 : (*(attrs->GetInt(0)));
    // execute opapi
    auto apiRet = CANN_OPS_OPB_SYN_EXEC_ACLNN(host_api_ctx, aclnnTransQuantParamV3, scale, offset, roundMode, output);
    OP_CHECK_IF(apiRet != GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT("aclnnfallback trans_quant_param_v2", "apiRet failed:%d", apiRet),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(TransQuantParamV2).OpExecuteFunc(TransQuantParamV2ExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
