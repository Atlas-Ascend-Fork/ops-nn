/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_mat_mul_fallback.cpp
 * \brief FusedMatMul is ACLNNTYPE aclnn_exclude, so the build does not emit an execute-func
 * registration. RT2 aclnn lowering requires one. Register the two-stage funcs here and run
 * aclnnFusedMatmul, the same API the host op uses.
 */

#include "op_graph/fallback_common_2stages_nn.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;

constexpr size_t kFusedMatmulInputX1 = 0;
constexpr size_t kFusedMatmulInputX2 = 1;
constexpr size_t kFusedMatmulInputBias = 2;
constexpr size_t kFusedMatmulInputX3 = 3;
constexpr size_t kFusedMatmulOutput = 0;
constexpr size_t kTransposeX1AttrIdx = 0;
constexpr size_t kTransposeX2AttrIdx = 1;
constexpr size_t kEnableHf32AttrIdx = 2;
constexpr size_t kFusedOpTypeAttrIdx = 3;

static graphStatus FusedMatmulExecuteFunc(OpExecutePrepareContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    auto x1_ge = host_api_ctx->GetInputTensor(kFusedMatmulInputX1);
    OP_CHECK_IF(x1_ge == nullptr, OP_LOGE("aclnnfallback", "x1 is null"), return GRAPH_FAILED);
    auto x2_ge = host_api_ctx->GetInputTensor(kFusedMatmulInputX2);
    OP_CHECK_IF(x2_ge == nullptr, OP_LOGE("aclnnfallback", "x2 is null"), return GRAPH_FAILED);
    auto bias_ge = host_api_ctx->GetOptionalInputTensor(kFusedMatmulInputBias);
    auto x3_ge = host_api_ctx->GetOptionalInputTensor(kFusedMatmulInputX3);
    auto y_ge = host_api_ctx->GetOutputTensor(kFusedMatmulOutput);
    OP_CHECK_IF(y_ge == nullptr, OP_LOGE("aclnnfallback", "y is null"), return GRAPH_FAILED);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE("aclnnfallback", "attrs is null"), return GRAPH_FAILED);
    auto transX1Ptr = attrs->GetAttrPointer<bool>(kTransposeX1AttrIdx);
    OP_CHECK_IF(transX1Ptr == nullptr, OP_LOGE("aclnnfallback", "transpose_x1 attr is null"), return GRAPH_FAILED);
    auto transX2Ptr = attrs->GetAttrPointer<bool>(kTransposeX2AttrIdx);
    OP_CHECK_IF(transX2Ptr == nullptr, OP_LOGE("aclnnfallback", "transpose_x2 attr is null"), return GRAPH_FAILED);

    auto x1_acl = ConvertMmType(x1_ge, *transX1Ptr);
    OP_CHECK_IF(x1_acl == nullptr, OP_LOGE("aclnnfallback", "x1_acl is null"), return GRAPH_FAILED);
    auto x2_acl = ConvertMmType(x2_ge, *transX2Ptr);
    OP_CHECK_IF(x2_acl == nullptr, OP_LOGE("aclnnfallback", "x2_acl is null"), return GRAPH_FAILED);
    auto bias_acl = ConvertType(bias_ge);
    auto x3_acl = ConvertType(x3_ge);

    // enable_hf32 is the node's own signal. Otherwise follow the shared matmul precision mode.
    int8_t cubeMathType = GetMathType(host_api_ctx);
    auto enableHf32Ptr = attrs->GetAttrPointer<bool>(kEnableHf32AttrIdx);
    if (enableHf32Ptr != nullptr && *enableHf32Ptr) {
        cubeMathType = USE_HF32;
    }
    // IR default of fused_op_type is "". A missing attr is that default, not an error.
    const char* fusedOpType = attrs->GetStr(kFusedOpTypeAttrIdx);
    if (fusedOpType == nullptr) {
        fusedOpType = "";
    }

    auto api_ret = CANN_OPS_OPB_ASYN_EXEC_ACLNN(host_api_ctx, aclnnFusedMatmul, x1_acl, x2_acl, bias_acl, x3_acl,
                                                fusedOpType, cubeMathType, y_ge);
    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE("aclnnfallback", "aclnnFusedMatmul api_ret failed:%d", api_ret),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

IMPL_OP(FusedMatMul).Op2StageExecuteFuncs(FusedMatmulExecuteFunc, ExecuteOpLaunch);

} // namespace fallback

#ifdef __cplusplus
}
#endif
