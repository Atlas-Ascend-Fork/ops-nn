/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_rotate_dual_level_quant.h"
#include "rotate_dual_level_quant.h"
#include "../op_host/rotate_dual_level_quant_common.h"
#include <vector>
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "op_common/log/log.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
namespace RDLQ = RotateDualLevelQuantNS;
constexpr const char* API_NAME = "aclnnRotateDualLevelQuantGetWorkspaceSize";

std::vector<int64_t> Dimensions(const op::Shape& shape)
{
    std::vector<int64_t> dims;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        dims.push_back(shape.GetDim(i));
    }
    return dims;
}

RDLQ::TargetType Target(op::DataType dtype)
{
    switch (dtype) {
        case op::DataType::DT_FLOAT4_E2M1:
            return RDLQ::TargetType::FP4_E2M1;
        case op::DataType::DT_FLOAT8_E4M3FN:
            return RDLQ::TargetType::FP8_E4M3FN;
        case op::DataType::DT_FLOAT8_E5M2:
            return RDLQ::TargetType::FP8_E5M2;
        default:
            return RDLQ::TargetType::UNSUPPORTED;
    }
}

bool Invalid(const char* argument, const std::string& reason)
{
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, argument, reason.c_str());
    return false;
}

bool CheckTensorTypes(const aclTensor* x, const aclTensor* rotation, const aclTensor* alpha, const aclTensor* groupList,
                      const aclTensor* y, const aclTensor* level0, const aclTensor* level1)
{
    for (const auto* tensor : {x, rotation, alpha, groupList, y, level0, level1}) {
        if (tensor != nullptr && tensor->GetStorageFormat() != op::Format::FORMAT_ND) {
            return Invalid("tensor.format", "all input and output tensors must use ND format");
        }
    }
    if ((x->GetDataType() != op::DataType::DT_FLOAT16 && x->GetDataType() != op::DataType::DT_BF16) ||
        rotation->GetDataType() != x->GetDataType()) {
        return Invalid("x, rotation", "x and rotation must have matching FLOAT16 or BFLOAT16 types");
    }
    if (Target(y->GetDataType()) == RDLQ::TargetType::UNSUPPORTED || level0->GetDataType() != op::DataType::DT_FLOAT ||
        level1->GetDataType() != op::DataType::DT_FLOAT8_E8M0) {
        return Invalid("outputs", "y must be FP4/FP8, level0_scale FLOAT32 and level1_scale FLOAT8_E8M0");
    }
    if (alpha != nullptr && (alpha->GetDataType() != op::DataType::DT_BF16 ||
                             Dimensions(alpha->GetViewShape()) != std::vector<int64_t>{1})) {
        return Invalid("alpha", "alpha must be a BFLOAT16 tensor with shape [1]");
    }
    if (groupList != nullptr && groupList->GetDataType() != op::DataType::DT_INT64) {
        return Invalid("group_list", "group_list must be INT64");
    }
    return true;
}

bool CheckContract(const aclTensor* x, const aclTensor* rotation, const aclTensor* alpha, const aclTensor* groupList,
                   const aclTensor* y, const aclTensor* level0, const aclTensor* level1, const RDLQ::Attributes& attr,
                   RDLQ::ShapeInfo& info)
{
    if (!CheckTensorTypes(x, rotation, alpha, groupList, y, level0, level1)) {
        return false;
    }
    const auto xShape = Dimensions(x->GetViewShape());
    const auto rotShape = Dimensions(rotation->GetViewShape());
    const auto groupShape = groupList == nullptr ? std::vector<int64_t>{} : Dimensions(groupList->GetViewShape());
    std::string reason;
    if (!RDLQ::ValidateShapes(xShape, rotShape, groupList == nullptr ? nullptr : &groupShape, false, info, reason) ||
        !RDLQ::ValidateAttributes(attr, xShape.size(), Target(y->GetDataType()), reason)) {
        return Invalid("shape/attributes", reason);
    }
    if (Dimensions(y->GetViewShape()) != xShape || Dimensions(level0->GetViewShape()) != info.level0Scale ||
        Dimensions(level1->GetViewShape()) != info.level1Scale) {
        return Invalid("outputs.shape", "expected y=x.shape, level0_scale=[...,ceil(N/512)], "
                                        "level1_scale=[...,ceil(N/64),2]");
    }
    return true;
}

bool MakeContiguous(const aclTensor*& tensor, aclOpExecutor* executor)
{
    if (tensor != nullptr) {
        tensor = l0op::Contiguous(tensor, executor);
        return tensor != nullptr;
    }
    return true;
}

aclTensor* PrepareOutput(aclTensor* output, bool preserveInitialValues, aclOpExecutor* executor)
{
    if (op::IsContiguous(output)) {
        return output;
    }
    // Grouped scheduling can leave any suffix (or all rows) untouched. Materialize
    // initial values before launch so full ViewCopy cannot destroy uncovered rows.
    if (preserveInitialValues) {
        return const_cast<aclTensor*>(l0op::Contiguous(output, executor));
    }
    return executor->AllocTensor(output->GetViewShape(), output->GetDataType(), op::Format::FORMAT_ND);
}
} // namespace

extern "C" aclnnStatus aclnnRotateDualLevelQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* rotation, const aclTensor* alpha, const aclTensor* groupList,
    int64_t groupListType, int64_t axis, char* roundMode, int64_t scaleAlg, double dstTypeMax, bool trans,
    int64_t level0BlockSize, int64_t level1BlockSize, aclTensor* yOut, aclTensor* level0ScaleOut,
    aclTensor* level1ScaleOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRotateDualLevelQuant,
                   DFX_IN(x, rotation, alpha, groupList, groupListType, axis, roundMode, scaleAlg, dstTypeMax, trans,
                          level0BlockSize, level1BlockSize),
                   DFX_OUT(yOut, level0ScaleOut, level1ScaleOut));
    if (workspaceSize == nullptr || executor == nullptr) {
        Invalid("workspaceSize, executor", "workspaceSize and executor must not be null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *workspaceSize = 0;
    *executor = nullptr;
    if (x == nullptr || rotation == nullptr || yOut == nullptr || level0ScaleOut == nullptr ||
        level1ScaleOut == nullptr) {
        Invalid("required tensors", "x, rotation and all three output tensors must not be null");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND950) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "RotateDualLevelQuant supports Ascend950 (A5) only.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    const RDLQ::Attributes attr{axis,
                                roundMode == nullptr ? "rint" : roundMode,
                                scaleAlg,
                                dstTypeMax,
                                trans,
                                groupListType,
                                level0BlockSize,
                                level1BlockSize};
    RDLQ::ShapeInfo info;
    if (!CheckContract(x, rotation, alpha, groupList, yOut, level0ScaleOut, level1ScaleOut, attr, info)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (info.empty) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    CHECK_RET(MakeContiguous(x, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(rotation, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(alpha, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguous(groupList, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);
    auto* y = PrepareOutput(yOut, groupList != nullptr, uniqueExecutor.get());
    auto* level0 = PrepareOutput(level0ScaleOut, groupList != nullptr, uniqueExecutor.get());
    auto* level1 = PrepareOutput(level1ScaleOut, groupList != nullptr, uniqueExecutor.get());
    CHECK_RET(y != nullptr && level0 != nullptr && level1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const auto result = l0op::RotateDualLevelQuant(
        x, rotation, alpha, groupList, axis, attr.roundMode.c_str(), scaleAlg, static_cast<float>(dstTypeMax), trans,
        static_cast<int64_t>(yOut->GetDataType()), groupListType, level0BlockSize, level1BlockSize, y, level0, level1,
        uniqueExecutor.get());
    CHECK_RET(std::get<0>(result) != nullptr && std::get<1>(result) != nullptr && std::get<2>(result) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    if (y != yOut) {
        CHECK_RET(l0op::ViewCopy(y, yOut, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (level0 != level0ScaleOut) {
        CHECK_RET(l0op::ViewCopy(level0, level0ScaleOut, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (level1 != level1ScaleOut) {
        CHECK_RET(l0op::ViewCopy(level1, level1ScaleOut, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnRotateDualLevelQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRotateDualLevelQuant);
    if (executor == nullptr || (workspaceSize != 0 && workspace == nullptr)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
