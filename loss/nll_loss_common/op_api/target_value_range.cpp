/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "target_value_range.h"
#include <vector>
#include "acl/acl_rt.h"
#include "opdev/op_log.h"

using namespace op;

bool CheckTargetValueRange(const aclTensor* target, int64_t nClasses, int64_t ignoreIndex)
{
    const int64_t targetNumel = target->GetViewShape().GetShapeSize();
    if (targetNumel == 0 || nClasses <= 0) {
        return true;
    }

    // 仅覆盖连续、零偏移、地址可解析的 target；否则跳过校验（行为与修复前一致）
    if (!op::IsContiguous(target) || target->GetViewOffset() != 0 || target->GetData() == nullptr) {
        OP_LOGD("Skip target value range check: target is not contiguous or address unresolved.");
        return true;
    }

    const op::DataType dtype = target->GetDataType();
    const size_t elemSize = op::TypeSize(dtype);
    std::vector<uint8_t> hostBuf(static_cast<size_t>(targetNumel) * elemSize);
    auto ret = aclrtMemcpy(hostBuf.data(), hostBuf.size(), target->GetData(), hostBuf.size(),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    // 回读失败（如 UT 等无 device 数据环境）时降级为跳过校验，不影响算子正常流程
    if (ret != ACL_SUCCESS) {
        OP_LOGW("Copy target from device to host failed, ret=%d, skip target value range check.", ret);
        return true;
    }
    return ValidateTargetValues(hostBuf.data(), targetNumel, dtype, nClasses, ignoreIndex);
}

// 逐元素校验 host buffer 中的 target 值域 [0, nClasses)，ignoreIndex 豁免。
// 供 aclnnNLLLoss/aclnnNLLLoss2d/aclnnNLLLossBackward 的一段值域校验复用，
// 也供 UT 直接单测（UT 环境无 device 数据，无法走 aclrtMemcpy 入口）。
bool ValidateTargetValues(const void* hostBuf, int64_t targetNumel, op::DataType dtype, int64_t nClasses,
                          int64_t ignoreIndex)
{
    if (hostBuf == nullptr || targetNumel <= 0 || nClasses <= 0) {
        return true;
    }
    for (int64_t i = 0; i < targetNumel; ++i) {
        int64_t idx = 0;
        switch (dtype) {
            case op::DataType::DT_INT64:
                idx = static_cast<const int64_t*>(hostBuf)[i];
                break;
            case op::DataType::DT_INT32:
                idx = static_cast<const int32_t*>(hostBuf)[i];
                break;
            case op::DataType::DT_UINT8:
                idx = static_cast<const uint8_t*>(hostBuf)[i];
                break;
            default:
                return true;
        }
        if (idx == ignoreIndex) {
            continue;
        }
        if (idx < 0 || idx >= nClasses) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Target %ld is out of bounds.", idx);
            return false;
        }
    }
    return true;
}
