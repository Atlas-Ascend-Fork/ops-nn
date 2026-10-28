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
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "AscendQuantV2", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

using namespace ge;
using namespace gert;
static const size_t INPUT_X_INDEX = 0;
static const size_t SCALE_INDEX = 1;
static const size_t OFFSET_INDEX = 2;
static const size_t ATTR_SQRT_MODE_INDEX = 0;
static const size_t ATTR_ROUND_MODE_INDEX = 1;
static const size_t ATTR_DST_DATATYPE_INDEX = 2;
static const size_t ATTR_AXIS_INDEX = 3;

static graphStatus AscendQuantHostExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_CHECK_IF(
        host_api_ctx == nullptr,
        ReportTilingStageError("AscendQuantV2",
                               "fallback: OpExecuteContext is null; the host fallback framework contract is broken"),
        return GRAPH_FAILED);

    auto inputX = host_api_ctx->GetInputTensor(INPUT_X_INDEX);
    OP_CHECK_IF(inputX == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback ascend_quant_v2", "x",
                                                         "the input tensor of x is missing in the fallback execute "
                                                         "context; check that the caller provides all required inputs"),
                return GRAPH_FAILED);

    auto scale = host_api_ctx->GetInputTensor(SCALE_INDEX);
    OP_CHECK_IF(scale == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback ascend_quant_v2", "scale",
                                                         "the input tensor of scale is missing in the fallback execute "
                                                         "context; check that the caller provides all required inputs"),
                return GRAPH_FAILED);

    auto offset = host_api_ctx->GetOptionalInputTensor(OFFSET_INDEX);

    auto output = host_api_ctx->GetOutputTensor(0);
    OP_CHECK_IF(
        output == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback ascend_quant_v2", "y",
                                                 "the output tensor of y is missing in the fallback execute context; "
                                                 "check that the caller provides all required outputs"),
        return GRAPH_FAILED);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        ReportTilingStageError("AscendQuantV2",
                               "fallback: GetAttrs() returned null; check the fallback execute context construction"),
        return GRAPH_FAILED);

    const bool* sqrtMode = attrs->GetBool(ATTR_SQRT_MODE_INDEX);
    OP_CHECK_IF(sqrtMode == nullptr,
                ReportTilingStageError("AscendQuantV2", "fallback: the required attr sqrt_mode is missing in the "
                                                        "fallback execute context; set the sqrt_mode attribute"),
                return GRAPH_FAILED);
    const char* roundMode = attrs->GetStr(ATTR_ROUND_MODE_INDEX);
    OP_CHECK_IF(roundMode == nullptr,
                ReportTilingStageError("AscendQuantV2", "fallback: the required attr round_mode is missing in the "
                                                        "fallback execute context; set the round_mode attribute"),
                return GRAPH_FAILED);
    const int64_t* dstDtype = attrs->GetInt(ATTR_DST_DATATYPE_INDEX);
    OP_CHECK_IF(dstDtype == nullptr,
                ReportTilingStageError("AscendQuantV2", "fallback: the required attr dst_dtype is missing in the "
                                                        "fallback execute context; set the dst_dtype attribute"),
                return GRAPH_FAILED);
    const int64_t* axis = attrs->GetInt(ATTR_AXIS_INDEX);
    OP_CHECK_IF(
        axis == nullptr,
        ReportTilingStageError(
            "AscendQuantV2",
            "fallback: the required attr axis is missing in the fallback execute context; set the axis attribute"),
        return GRAPH_FAILED);

    OP_LOGD("aclnnfallback ascend_quant_v2", "fallback begin");
    int32_t dstDtypeVal = static_cast<int32_t>(*dstDtype);
    int32_t axisVal = static_cast<int32_t>(*axis);
    auto api_ret = CANN_OPS_OPB_SYN_EXEC_ACLNN(host_api_ctx, aclnnAscendQuantV3, inputX, scale, offset, *sqrtMode,
                                               roundMode, dstDtypeVal, axisVal, output);

    OP_CHECK_IF(api_ret != GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT("aclnnfallback ascend_quant_v2", "api_ret failed:%u", api_ret),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(AscendQuantV2).OpExecuteFunc(AscendQuantHostExecuteFunc);
} // namespace fallback

#ifdef __cplusplus
}
#endif
