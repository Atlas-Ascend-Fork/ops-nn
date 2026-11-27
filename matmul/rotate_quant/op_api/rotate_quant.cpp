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
 * \file rotate_quant.cpp
 * \brief RotateQuant Level 0 Operator implementation
 */

#include "rotate_quant.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(RotateQuant);

static std::tuple<aclTensor*, aclTensor*> RotateQuantAICore(const aclTensor* x, const aclTensor* rotation,
                                                            const aclTensor* alpha, const aclTensor* groupList,
                                                            int64_t axis, const char* roundMode, int64_t scaleAlg,
                                                            float dstTypeMax, bool trans, int64_t dstType,
                                                            int64_t groupListType, aclTensor* yOut, aclTensor* scaleOut,
                                                            aclOpExecutor* executor)
{
    L0_DFX(RotateQuantAICore, x, rotation, alpha, groupList, axis, roundMode, scaleAlg, dstTypeMax, trans, dstType,
           groupListType, yOut, scaleOut);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        RotateQuant, OP_INPUT(x, rotation, alpha, groupList), OP_OUTPUT(yOut, scaleOut),
        OP_ATTR(dstType, axis, roundMode, scaleAlg, dstTypeMax, trans, groupListType));
    if (retAicore != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID,
                               "Failed to add RotateQuant to the AICore launch list, error code: %d.",
                               static_cast<int>(retAicore));
        return std::tuple(nullptr, nullptr);
    }
    return std::tie(yOut, scaleOut);
}

const std::tuple<aclTensor*, aclTensor*> RotateQuant(const aclTensor* x, const aclTensor* rotation,
                                                     const aclTensor* alpha, const aclTensor* groupList, int64_t axis,
                                                     const char* roundMode, int64_t scaleAlg, float dstTypeMax,
                                                     bool trans, int64_t dstType, int64_t groupListType,
                                                     aclTensor* yOut, aclTensor* scaleOut, aclOpExecutor* executor)
{
    // Only the 910B int path (scale=FLOAT) needs to reallocate a 1D scale; use the positive-form check aligned with
    // cann_out: the DAV_9201/DAV_9202 hif4 path (scale=HIFLOAT4_SCALE) must not enter this branch, otherwise the
    // scale dtype is changed to FLOAT and its shape collapses to 1D (the old mainline negative-form check
    // "!=DT_FLOAT8_E8M0" misjudges hif4; on-device symptom: hifloat4_scale becomes float before entering workspace).
    if (scaleOut->GetDataType() == DataType::DT_FLOAT) {
        auto yAlloc = executor->AllocTensor(x->GetViewShape(), op::DataType(dstType), op::Format::FORMAT_ND);
        int64_t M = x->GetViewShape().GetDim(0);
        auto scaleAlloc = executor->AllocTensor(op::Shape({M}), op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
        if (yAlloc == nullptr || scaleAlloc == nullptr) {
            OP_LOGE_WITHOUT_REPORT(
                ACLNN_ERR_INNER_NULLPTR,
                "Failed to allocate y or scale tensor in RotateQuant; AllocTensor returned nullptr.");
            return std::tuple(nullptr, nullptr);
        }
        yOut = yAlloc;
        scaleOut = scaleAlloc;
    }
    return RotateQuantAICore(x, rotation, alpha, groupList, axis, roundMode, scaleAlg, dstTypeMax, trans, dstType,
                             groupListType, yOut, scaleOut, executor);
}

} // namespace l0op
