/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INPLACE_APPLY_POWER_SIGN_HOST_GUARD_H_
#define INPLACE_APPLY_POWER_SIGN_HOST_GUARD_H_

#include "exe_graph/runtime/extended_kernel_context.h"
#include "log/log.h"

namespace ops {

constexpr const char* POWER_SIGN_INPUT_NAMES[] = {"var", "m", "lr", "logbase", "sign_decay", "beta", "grad"};

inline ge::graphStatus ReportPowerSignFailure(const char* opName, const std::string& reason,
                                              const char* errorCode = "EZ1014")
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG(errorCode, keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    return ge::GRAPH_FAILED;
}

// Check node instances before a fixed input index can address kernel extension data.
// This validates the callback context, not connectivity in the source GE graph.
inline bool HasInplaceApplyPowerSignRequiredIo(const gert::ExtendedKernelContext* context,
                                               const char* errorCode = "EZ1014")
{
    if (context == nullptr) {
        ReportPowerSignFailure("InplaceApplyPowerSign", "Callback context is nullptr", errorCode);
        return false;
    }
    const auto* nodeInfo = context->GetComputeNodeInfo();
    constexpr size_t REQUIRED_INPUTS = 7;
    constexpr size_t REQUIRED_OUTPUTS = 2;
    if (nodeInfo == nullptr) {
        ReportPowerSignFailure(context->GetNodeName(), "ComputeNodeInfo is nullptr", errorCode);
        return false;
    }
    if (nodeInfo->GetInputsNum() != REQUIRED_INPUTS || nodeInfo->GetOutputsNum() != REQUIRED_OUTPUTS) {
        ReportPowerSignFailure(
            context->GetNodeName(),
            "Callback I/O slot count is invalid: inputs=" + std::to_string(nodeInfo->GetInputsNum()) +
                ", outputs=" + std::to_string(nodeInfo->GetOutputsNum()) + "; expected seven inputs and two outputs",
            errorCode);
        return false;
    }
    if (nodeInfo->GetIrInputsNum() != REQUIRED_INPUTS) {
        ReportPowerSignFailure(context->GetNodeName(), "IR input metadata must contain seven entries", errorCode);
        return false;
    }
    for (size_t i = 0; i < REQUIRED_INPUTS; ++i) {
        const auto* instance = nodeInfo->GetInputInstanceInfo(i);
        if (instance == nullptr) {
            ReportPowerSignFailure(context->GetNodeName(),
                                   "Input instance metadata is nullptr for " + std::string(POWER_SIGN_INPUT_NAMES[i]),
                                   errorCode);
            return false;
        }
        OP_CHECK_IF(instance->GetInstanceNum() == 0,
                    OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), POWER_SIGN_INPUT_NAMES[i]), return false);
        if (instance->GetInstanceNum() != 1) {
            ReportPowerSignFailure(context->GetNodeName(),
                                   "Invalid required-input instance metadata for " +
                                       std::string(POWER_SIGN_INPUT_NAMES[i]) +
                                       ": count=" + std::to_string(instance->GetInstanceNum()) + "; expected one",
                                   errorCode);
            return false;
        }
    }
    return true;
}

} // namespace ops

#endif // INPLACE_APPLY_POWER_SIGN_HOST_GUARD_H_
