/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file aclnn_unique_dim.cpp
 * \brief
 */

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def_nn.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "log/log.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_util.h"
#include "opdev/make_op_executor.h"
#include "unique_with_counts_ext2.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_unique_dim.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr const char* kOpName = "aclnnUniqueDim";

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8,  op::DataType::DT_INT8,  op::DataType::DT_UINT16,  op::DataType::DT_INT16,
    op::DataType::DT_UINT32, op::DataType::DT_INT32, op::DataType::DT_UINT64,  op::DataType::DT_INT64,
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8,  op::DataType::DT_INT8,  op::DataType::DT_UINT16,  op::DataType::DT_INT16,
    op::DataType::DT_UINT32, op::DataType::DT_INT32, op::DataType::DT_UINT64,  op::DataType::DT_INT64,
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_BOOL};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 || Ops::NN::AclnnUtil::IsRegbase()) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}
static bool CheckNotNull(const aclTensor* self, aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut)
{
    if (self == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "self");
        return false;
    }
    if (valueOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "valueOut");
        return false;
    }
    if (inverseOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "inverseOut");
        return false;
    }
    if (countsOut == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "countsOut");
        return false;
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* valueOut, const aclTensor* inverseOut,
                            const aclTensor* countsOut)
{
    auto DTYPE_SUPPORT_LIST = GetDtypeSupportList();
    if (!CheckType(self->GetDataType(), DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            kOpName, "self", op::ToString(self->GetDataType()).GetString(),
            (std::string("self dtype must be one of ") + op::ToString(DTYPE_SUPPORT_LIST).GetString()).c_str());
        return false;
    }
    if (inverseOut->GetDataType() != op::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "inverseOut", op::ToString(inverseOut->GetDataType()).GetString(), "int64");
        return false;
    }
    if (countsOut->GetDataType() != op::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, "countsOut", op::ToString(countsOut->GetDataType()).GetString(), "int64");
        return false;
    }
    if (valueOut->GetDataType() != self->GetDataType()) {
        const std::string dtypes = std::string(op::ToString(self->GetDataType()).GetString()) + ", " +
                                   op::ToString(valueOut->GetDataType()).GetString();
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(kOpName, "self, valueOut", dtypes.c_str(),
                                               "valueOut dtype must be the same as self dtype");
        return false;
    }
    return true;
}

static bool CheckShapeValid(const aclTensor* self)
{
    const auto dimNum = self->GetViewShape().GetDimNum();
    if (dimNum > MAX_SUPPORT_DIMS_NUMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "self", std::to_string(dimNum).c_str(),
                                                 "self rank must not exceed 8");
        return false;
    }
    return true;
}

static bool CheckDimValue(const aclTensor* self, int64_t dim)
{
    int64_t dimSize = self->GetViewShape().GetDimNum();
    int64_t dimMin = std::min(-1 * dimSize, dimSize - 1);
    int64_t dimMax = std::max(-1 * dimSize, dimSize - 1);
    if (dim > dimMax || dim < dimMin) {
        const std::string expected = "[" + std::to_string(dimMin) + ", " + std::to_string(dimMax) + "]";
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "dim", std::to_string(dim).c_str(), expected.c_str());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
                               int64_t dim)
{
    // 1. 检查输入输出是否为nullptr
    if (!CheckNotNull(self, valueOut, inverseOut, countsOut)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    // 2. 检查数据类型
    if (!CheckDtypeValid(self, valueOut, inverseOut, countsOut)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 3. 检查数据Shape
    if (!CheckShapeValid(self)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 4. 检查参数dim
    if (!CheckDimValue(self, dim)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUniqueDimGetWorkspaceSize(const aclTensor* self, bool sorted, bool returnInverse, int64_t dim,
                                           aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
                                           uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (workspaceSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    L2_DFX_PHASE_1(aclnnUniqueDim, DFX_IN(self, sorted, returnInverse, dim), DFX_OUT(valueOut, inverseOut, countsOut));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_CREATE_EXECUTOR, "Create op executor failed.");
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }

    // 固定写法，参数检查
    auto ret = CheckParams(self, valueOut, inverseOut, countsOut, dim);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    // 空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    if (selfContiguous == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "Contiguous of self failed; propagate lower-layer error.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto valueViewShape = valueOut->GetViewShape();
    valueOut->SetStorageShape(valueViewShape);
    valueOut->SetOriginalShape(valueViewShape);

    auto countsViewShape = countsOut->GetViewShape();
    countsOut->SetStorageShape(countsViewShape);
    countsOut->SetOriginalShape(countsViewShape);

    // 调用UniqueDim算子
    auto opRet = l0op::UniqueDim(selfContiguous, sorted, returnInverse, dim, valueOut, inverseOut, countsOut,
                                 uniqueExecutor.get());
    if (opRet != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(opRet, "UniqueDim failed, ret=%d; propagate lower-layer error.", opRet);
        return opRet;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUniqueDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUniqueDim);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
