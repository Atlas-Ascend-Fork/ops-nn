/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file linear_index.cpp
 * \brief
 */
#include "linear_index.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"

using namespace op;

static const std::initializer_list<op::DataType> AICORE_910B_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32,
                                                                                   op::DataType::DT_INT64};

namespace l0op {
OP_TYPE_REGISTER(LinearIndex);

const aclTensor* LinearIndex(const aclTensor* indices, const aclTensor* var, int64_t axis, bool combine,
                             aclOpExecutor* executor)
{
    L0_DFX(LinearIndex, indices, var);

    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[LinearIndex] only support ASCEND910B and ASCEND910_93");
    }

    if (!CheckType(indices->GetDataType(), AICORE_910B_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[LinearIndex] indices must be in int32,int64");
    }

    op::Shape outShape = indices->GetViewShape();
    if (combine) {
        int64_t combineShape = 1;
        for (size_t i = 0; i < outShape.GetDimNum(); ++i) {
            combineShape *= outShape.GetDim(i);
        }
        outShape.SetDimNum(1);
        outShape.SetDim(0, combineShape);
    }
    auto index = executor->AllocTensor(outShape, op::DataType::DT_INT32, indices->GetViewFormat());
    OP_CHECK(index != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LinearIndex: aclOpExecutor::AllocTensor returned nullptr for index."),
             return nullptr);

    FVector<int64_t> shapeVector;
    op::Shape varShape = var->GetViewShape();
    for (size_t i = 0; i < varShape.GetDimNum(); ++i) {
        shapeVector.emplace_back(varShape.GetDim(i));
    }
    aclIntArray* shapeArray = executor->AllocIntArray(shapeVector.data(), shapeVector.size());
    OP_CHECK(
        shapeArray != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LinearIndex: aclOpExecutor::AllocIntArray returned nullptr for shapeArray."),
        return nullptr);
    auto shapeTensor = executor->ConvertToTensor(shapeArray, ToOpDataType(ACL_INT32));
    OP_CHECK(shapeTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "LinearIndex: aclOpExecutor::ConvertToTensor returned nullptr for shapeTensor."),
             return nullptr);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LinearIndex, OP_INPUT(indices, shapeTensor), OP_OUTPUT(index),
                                           OP_ATTR(axis, combine));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR,
                                    "LinearIndexAiCore ADD_TO_LAUNCHER_LIST_AICORE failed. ret: %d.", ret),
             return nullptr);
    return index;
}
} // namespace l0op
