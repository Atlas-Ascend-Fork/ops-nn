/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rotate_dual_level_quant.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(RotateDualLevelQuant);

std::tuple<aclTensor*, aclTensor*, aclTensor*> RotateDualLevelQuant(
    const aclTensor* x, const aclTensor* rotation, const aclTensor* alpha, const aclTensor* groupList, int64_t axis,
    const char* roundMode, int64_t scaleAlg, float dstTypeMax, bool trans, int64_t dstType, int64_t groupListType,
    int64_t level0BlockSize, int64_t level1BlockSize, aclTensor* yOut, aclTensor* level0ScaleOut,
    aclTensor* level1ScaleOut, aclOpExecutor* executor)
{
    L0_DFX(RotateDualLevelQuant, x, rotation, alpha, groupList, axis, roundMode, scaleAlg, dstTypeMax, trans, dstType,
           groupListType, level0BlockSize, level1BlockSize, yOut, level0ScaleOut, level1ScaleOut);
    auto ret = INFER_SHAPE(RotateDualLevelQuant, OP_INPUT(x, rotation, alpha, groupList),
                           OP_OUTPUT(yOut, level0ScaleOut, level1ScaleOut),
                           OP_ATTR(dstType, axis, roundMode, scaleAlg, dstTypeMax, trans, groupListType,
                                   level0BlockSize, level1BlockSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "RotateDualLevelQuant shape inference failed.");
        return {nullptr, nullptr, nullptr};
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(RotateDualLevelQuant, OP_INPUT(x, rotation, alpha, groupList),
                                      OP_OUTPUT(yOut, level0ScaleOut, level1ScaleOut),
                                      OP_ATTR(dstType, axis, roundMode, scaleAlg, dstTypeMax, trans, groupListType,
                                              level0BlockSize, level1BlockSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID, "RotateDualLevelQuant launcher registration failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {yOut, level0ScaleOut, level1ScaleOut};
}
} // namespace l0op
