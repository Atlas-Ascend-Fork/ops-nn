/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "op_api/aclnn_util.h"
#include "softmax_v2_op.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(SoftmaxV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* self) { return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST); }

// AICORE算子kernel
static const aclTensor* SoftmaxAiCore(const aclTensor* self, aclIntArray* dimList, aclTensor* out, bool half_to_float,
                                      aclOpExecutor* executor)
{
    L0_DFX(SoftmaxAiCore);
    ADD_TO_LAUNCHER_LIST_AICORE(SoftmaxV2, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(dimList, half_to_float));
    return out;
}

// AICPU算子kernel
static const aclTensor* SoftmaxAiCpu(const aclTensor* self, aclIntArray* dimList, aclTensor* out,
                                     aclOpExecutor* executor)
{
    L0_DFX(SoftmaxAiCpu);
    static internal::AicpuTaskSpace space("SoftmaxV2", ge::DEPEND_IN_SHAPE, false);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(SoftmaxV2, OP_ATTR_NAMES({"axes"}), OP_INPUT(self), OP_OUTPUT(out),
                                          OP_ATTR(dimList));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

// SoftmaxV2公共实现：按指定的输出dtype构造输出并派发kernel
static const aclTensor* SoftmaxV2Impl(const aclTensor* self, int64_t dim, bool half_to_float, op::DataType outDtype,
                                      aclOpExecutor* executor)
{
    // 输出shape和self shape一致
    auto opOut = executor->AllocTensor(self->GetStorageShape(), outDtype);
    FVector<int64_t> newdim{dim};
    auto dimList = executor->AllocIntArray(newdim.data(), 1);

    if (IsAiCoreSupport(self)) {
        return SoftmaxAiCore(self, dimList, opOut, half_to_float, executor);
    } else {
        return SoftmaxAiCpu(self, dimList, opOut, executor);
    }
}

const aclTensor* SoftmaxV2(const aclTensor* self, int64_t dim, aclOpExecutor* executor)
{
    // 原接口保持既有行为：按self dtype构造输出，half_to_float恒为false
    return SoftmaxV2Impl(self, dim, false, self->GetDataType(), executor);
}

const aclTensor* SoftmaxV2(const aclTensor* self, int64_t dim, const aclTensor* out, aclOpExecutor* executor)
{
    // regbase场景输入为FP16且输出为FP32时，置half_to_float并按FP32构造输出，走FP32精度路径，其余场景保持不变
    bool half_to_float = false;
    auto outDtype = self->GetDataType();
    if (Ops::NN::AclnnUtil::IsRegbase() && self->GetDataType() == op::DataType::DT_FLOAT16 &&
        out->GetDataType() == op::DataType::DT_FLOAT) {
        half_to_float = true;
        outDtype = op::DataType::DT_FLOAT;
    }
    return SoftmaxV2Impl(self, dim, half_to_float, outDtype, executor);
}

} // namespace l0op
