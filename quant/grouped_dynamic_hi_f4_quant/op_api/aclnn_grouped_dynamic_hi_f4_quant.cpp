/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_grouped_dynamic_hi_f4_quant.h"
#include "grouped_dynamic_hi_f4_quant.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int64_t DT_HIFLOAT4 = 42;
static constexpr int64_t DT_HIFLOAT4_SCALE = 43;
static constexpr uint64_t NUM_ZERO = 0;

static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                         op::DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor* x, const aclTensor* groupIndex, const aclTensor* y,
                                const aclTensor* scale)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(groupIndex, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(scale, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* groupIndex, const aclTensor* y, const aclTensor* scale)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST, return false);
    if (groupIndex->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "group_index data type must be INT32.");
        return false;
    }
    if (static_cast<int64_t>(y->GetDataType()) != DT_HIFLOAT4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out(y) data type must be HIFLOAT4.");
        return false;
    }
    if (static_cast<int64_t>(scale->GetDataType()) != DT_HIFLOAT4_SCALE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out(scale) data type must be HIFLOAT4_SCALE.");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_SHAPE_NOT_EQUAL(y, x, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* groupIndex, const aclTensor* y,
                               const aclTensor* scale)
{
    CHECK_RET(CheckNotNull(x, groupIndex, y, scale), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, groupIndex, y, scale), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, y), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedDynamicHiF4QuantGetWorkspaceSize(const aclTensor* x, const aclTensor* group_index,
                                                         int64_t scale_alg, aclTensor* y, aclTensor* scale,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGroupedDynamicHiF4Quant, DFX_IN(x, group_index, scale_alg), DFX_OUT(y, scale));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, group_index, y, scale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (group_index->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "group_index does not support empty values.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x->IsEmpty()) {
        *workspaceSize = NUM_ZERO;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto groupIndexContiguous = l0op::Contiguous(group_index, uniqueExecutor.get());
    CHECK_RET(groupIndexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::GroupedDynamicHiF4Quant(xContiguous, groupIndexContiguous, scale_alg, y, scale,
                                                uniqueExecutor.get());
    const aclTensor* yOut = result[0];
    const aclTensor* scaleOut = result[1];
    CHECK_RET(yOut != nullptr && scaleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyY = l0op::ViewCopy(yOut, y, uniqueExecutor.get());
    CHECK_RET(viewCopyY != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyScale = l0op::ViewCopy(scaleOut, scale, uniqueExecutor.get());
    CHECK_RET(viewCopyScale != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedDynamicHiF4Quant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedDynamicHiF4Quant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
