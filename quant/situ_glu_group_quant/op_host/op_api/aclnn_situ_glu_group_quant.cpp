/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_situ_glu_group_quant.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"
#include "aclnnInner_situ_glu_group_quant.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t MIN_DIM_NUM = 1;
constexpr int64_t SPLIT_NUM = 2;
// MXFP8 group size; y_scale last axis interleave factor
constexpr int64_t MX_BLOCK_SIZE = 32;
constexpr int64_t SCALE_ALIGN_NUM = 2;
constexpr int64_t SCALE_GROUP_SIZE = MX_BLOCK_SIZE * SCALE_ALIGN_NUM;
// whole-row chunk cap (tiling fail-fasts beyond it)
static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                         op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> GROUP_INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> Y_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> MXSCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT8_E8M0};

static bool CheckNotNull(const aclTensor* x, const aclTensor* weight, const aclTensor* groupIndex,
                         const aclTensor* yOut, const aclTensor* yScaleOut)
{
    // groupIndex / weight are OPTIONAL: nullptr = whole-tensor mode / no scaling
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(yOut, return false);
    OP_CHECK_NULL(yScaleOut, return false);
    return true;
}

static bool CheckScalarParams(double beta)
{
    if (beta <= 0.0) {
        const std::string reason = (("beta must be greater than 0, but got " + std::to_string(beta) + "."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ0026",
                                  (std::vector<const char*>{"param_name", "op_name", "incorrect_value", "reason"}),
                                  (std::vector<const char*>{"beta", "aclnnSituGluGroupQuant",
                                                            (std::to_string(beta)).c_str(), reason.c_str()}));
        return false;
    }
    // linear_beta: <= 0 disables the up-branch tanh soft-clamp (up
    // passthrough, TPL dimension); > 0 enables it — any value is legal.
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* weight, const aclTensor* groupIndex,
                            const aclTensor* yOut, const aclTensor* yScaleOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST, return false);
    if (groupIndex != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(groupIndex, GROUP_INDEX_DTYPE_SUPPORT_LIST, return false);
    }
    if (weight != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(weight, op::DataType::DT_FLOAT, return false);
    }
    // y is always e4m3fn and y_scale is always e8m0 (fixed dtypes)
    OP_CHECK_DTYPE_NOT_SUPPORT(yOut, Y_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(yScaleOut, MXSCALE_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* weight, const aclTensor* groupIndex, const aclTensor* yOut,
                        const aclTensor* yScaleOut)
{
    // torch_npu对3/4/5维连续Tensor会推导NCL/NCHW/NCDHW格式标签，其连续存储与ND字节布局一致，
    // 属于等效线性格式，均需放行；分形等非线性存储格式仍拦截
    const auto isLinearFormat = [](op::Format format) {
        return format == op::Format::FORMAT_ND || format == op::Format::FORMAT_NCL ||
               format == op::Format::FORMAT_NCHW || format == op::Format::FORMAT_NCDHW;
    };
    if (!isLinearFormat(x->GetStorageFormat()) || !isLinearFormat(yOut->GetStorageFormat()) ||
        !isLinearFormat(yScaleOut->GetStorageFormat()) ||
        (groupIndex != nullptr && !isLinearFormat(groupIndex->GetStorageFormat())) ||
        (weight != nullptr && !isLinearFormat(weight->GetStorageFormat()))) {
        const std::string reason = ((
            "x, weight, groupIndex, yOut and yScaleOut only support linear formats (ND/NCL/NCHW/NCDHW), but got x: " +
            std::string(op::ToString(x->GetStorageFormat()).GetString()) +
            ", yOut: " + std::string(op::ToString(yOut->GetStorageFormat()).GetString()) +
            ", yScaleOut: " + std::string(op::ToString(yScaleOut->GetStorageFormat()).GetString()) +
            (weight != nullptr ? ", weight: " + std::string(op::ToString(weight->GetStorageFormat()).GetString()) :
                                 "") +
            (groupIndex != nullptr ?
                 ", groupIndex: " + std::string(op::ToString(groupIndex->GetStorageFormat()).GetString()) :
                 "") +
            "."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        const std::string paramNames = std::string("x,yOut,yScaleOut") + (weight != nullptr ? ",weight" : "") +
                                       (groupIndex != nullptr ? ",groupIndex" : "");
        const std::string incorrectFormats = std::string(op::ToString(x->GetStorageFormat()).GetString()) + "," +
                                             op::ToString(yOut->GetStorageFormat()).GetString() + "," +
                                             op::ToString(yScaleOut->GetStorageFormat()).GetString() +
                                             (weight != nullptr ?
                                                  "," + std::string(
                                                            op::ToString(weight->GetStorageFormat()).GetString()) :
                                                  "") +
                                             (groupIndex != nullptr ?
                                                  "," + std::string(
                                                            op::ToString(groupIndex->GetStorageFormat()).GetString()) :
                                                  "");
        REPORT_PREDEFINED_ERR_MSG("EZ0018",
                                  (std::vector<const char*>{"param_names", "op_name", "incorrect_formats", "reason"}),
                                  (std::vector<const char*>{paramNames.c_str(), "aclnnSituGluGroupQuant",
                                                            incorrectFormats.c_str(), reason.c_str()}));
        return false;
    }
    return true;
}

static bool HasUnknownDim(const op::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) < 0) {
            return true;
        }
    }
    return false;
}

static bool CheckShape(const aclTensor* x, const aclTensor* groupIndex, const aclTensor* yOut,
                       const aclTensor* yScaleOut)
{
    // recipes caliber: no empty-tensor rejection (a zero-size leading dim
    // degenerates to blockDim/validRows of 0, exactly like the recipes op);
    // rank >= 1 without the 8-dim cap
    OP_CHECK_MIN_DIM(x, MIN_DIM_NUM, return false);

    // recipes caliber: last dim must be positive and even
    const op::Shape& xShape = x->GetViewShape();
    const int64_t lastDim = xShape.GetDim(xShape.GetDimNum() - 1);
    if (lastDim <= 0 || lastDim % SPLIT_NUM != 0) {
        const std::string reason = (("The last dimension of x must be positive and divisible by 2, but got " +
                                     std::to_string(lastDim) + "."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ0015",
                                  (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "reason"}),
                                  (std::vector<const char*>{"x", "aclnnSituGluGroupQuant",
                                                            (std::to_string(lastDim)).c_str(), reason.c_str()}));
        return false;
    }
    const int64_t epr = lastDim / SPLIT_NUM;
    // recipes caliber: no EPR upper bound (the whole-row chunk cap is an
    // upstream-caller assumption of the recipes op, not an enforced limit)

    op::Shape expectedYShape;
    for (size_t i = 0; i < xShape.GetDimNum() - 1; i++) {
        expectedYShape.AppendDim(xShape.GetDim(i));
    }
    expectedYShape.AppendDim(epr);

    op::Shape expectedYScaleShape = expectedYShape;
    const int64_t scaleNum = (epr + SCALE_GROUP_SIZE - 1) / SCALE_GROUP_SIZE;
    expectedYScaleShape.SetDim(expectedYScaleShape.GetDimNum() - 1, scaleNum);
    expectedYScaleShape.AppendDim(SCALE_ALIGN_NUM);

    // 含未知维(-1/-2)的动态shape无法精确比较，交由执行阶段推导校验
    if (!HasUnknownDim(xShape)) {
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(yOut, expectedYShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(yScaleOut, expectedYScaleShape, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* weight, const aclTensor* groupIndex, double beta,
                               double linearBeta, int64_t axis, int64_t quantMode, int64_t dstType, double dstTypeMax,
                               const char* roundModeOptional, int64_t groupListType, const aclTensor* yOut,
                               const aclTensor* yScaleOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, weight, groupIndex, yOut, yScaleOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查host侧标量参数：beta/linearBeta
    CHECK_RET(CheckScalarParams(beta), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查扩展属性参数（situ_mx_quant caliber，当前均为单一支持值；
    //    完整语义校验在host tiling ParseAttrs 中做，这里仅做空指针/符号防护）
    if (roundModeOptional == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_NULLPTR, "roundModeOptional must not be null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (dstTypeMax < 0.0) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "dstTypeMax must be >= 0, but got %f.", dstTypeMax);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (groupListType != 0 && groupListType != 1) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "Invalid groupListType: %ld, only 0(cumsum), 1(count).",
                               groupListType);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (groupListType != 1) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID,
                               "groupListType=0 (cumsum mode) is reserved and not supported yet, pass 1 instead.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    (void)axis;
    (void)quantMode;
    (void)dstType;
    // linear_beta: <= 0 disables the up-branch soft-clamp (up passthrough);
    // > 0 enables it. No sign/range rejection — the split is a TPL dimension.
    (void)linearBeta;

    // 4. 检查输入/输出的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x, weight, groupIndex, yOut, yScaleOut), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查format是否符合要求
    CHECK_RET(CheckFormat(x, weight, groupIndex, yOut, yScaleOut), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查shape/空tensor是否符合约束
    CHECK_RET(CheckShape(x, groupIndex, yOut, yScaleOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnSituGluGroupQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* weightOptional,
                                                   const aclTensor* groupIndexOptional, double beta, double linearBeta,
                                                   bool activateLeft, int64_t axis, int64_t quantMode, int64_t dstType,
                                                   double dstTypeMax, const char* roundModeOptional,
                                                   int64_t groupListType, const aclTensor* yOut,
                                                   const aclTensor* yScaleOut, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    auto ret = CheckParams(x, weightOptional, groupIndexOptional, beta, linearBeta, axis, quantMode, dstType,
                           dstTypeMax, roundModeOptional, groupListType, yOut, yScaleOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return aclnnInnerSituGluGroupQuantGetWorkspaceSize(
        x, weightOptional, groupIndexOptional, beta, linearBeta, activateLeft, axis, quantMode, dstType, dstTypeMax,
        const_cast<char*>(roundModeOptional), groupListType, yOut, yScaleOut, workspaceSize, executor);
}

aclnnStatus aclnnSituGluGroupQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerSituGluGroupQuant(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
