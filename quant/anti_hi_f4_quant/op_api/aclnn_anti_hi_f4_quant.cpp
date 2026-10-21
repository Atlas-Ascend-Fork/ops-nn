/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_anti_hi_f4_quant.h"
#include "anti_hi_f4_quant.h"
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

static constexpr int64_t DT_NUM_FLOAT16 = 1;
static constexpr int64_t DT_NUM_BFLOAT16 = 27;
static constexpr int64_t DT_HIFLOAT4 = 42;
static constexpr int64_t DT_HIFLOAT4_SCALE = 43;
static constexpr uint64_t NUM_ZERO = 0;

static const std::initializer_list<op::DataType> Y_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                         op::DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor* x, const aclTensor* scale, const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckAttrValid(int64_t dstType)
{
    if (dstType != DT_NUM_FLOAT16 && dstType != DT_NUM_BFLOAT16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dstType must be 1(FLOAT16) or 27(BF16), but got %ld.", dstType);
        return false;
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* scale, const aclTensor* y, int64_t dstType)
{
    if (static_cast<int64_t>(x->GetDataType()) != DT_HIFLOAT4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x dtype must be HIFLOAT4.");
        return false;
    }
    if (static_cast<int64_t>(scale->GetDataType()) != DT_HIFLOAT4_SCALE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale dtype must be HIFLOAT4_SCALE.");
        return false;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(y, Y_DTYPE_SUPPORT_LIST, return false);
    if (static_cast<int64_t>(y->GetDataType()) != dstType) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out(y) data type must be the same as dstType.");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_SHAPE_NOT_EQUAL(y, x, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* scale, int64_t dstType, const aclTensor* y)
{
    CHECK_RET(CheckNotNull(x, scale, y), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckAttrValid(dstType), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(x, scale, y, dstType), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, y), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAntiHiF4QuantGetWorkspaceSize(const aclTensor* x, const aclTensor* scale, int64_t dstType,
                                               aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAntiHiF4Quant, DFX_IN(x, scale, dstType), DFX_OUT(y));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, scale, dstType, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty() || scale->IsEmpty()) {
        *workspaceSize = NUM_ZERO;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    if (!IsContiguous(x)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When the data type of x is HIFLOAT4, it must be contiguous");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsContiguous(scale)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When the data type of scale is HIFLOAT4_SCALE, it must be contiguous");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto scaleContiguous = l0op::Contiguous(scale, uniqueExecutor.get());
    CHECK_RET(scaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto antiHiF4QuantResult = l0op::AntiHiF4Quant(xContiguous, scaleContiguous, dstType, uniqueExecutor.get());
    CHECK_RET(antiHiF4QuantResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(antiHiF4QuantResult, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAntiHiF4Quant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAntiHiF4Quant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
