/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "ada_layer_norm.h"
#include "aclnn_ada_layer_norm.h"
#include "log/log.h"

using namespace op;

namespace {
// aclnn op_api层空指针失败出口统一上报EZ1011(nnopbase注册码, Arglist={functionName,paraName})。
// op_api编译单元带-DOP_LOG_LIBOPAPI_ONLY, log/log.h内OP_LOGE_WITHOUT_REPORT(opName版)不可见,
// 日志走始终可见的OP_LOGE_LIBOPAPI_REPORT。
void ReportNullPointerError(const char* functionName, const char* paraName)
{
    std::string funcName = functionName;
    std::string paramName = paraName;
    std::string msg = funcName + " failed because " + paramName + " cannot be a NULL pointer.";
    OP_LOGE_LIBOPAPI_REPORT(funcName.c_str(), "%s", msg.c_str());
    const std::vector<const char*> msgKey = {"functionName", "paraName"};
    const std::vector<const char*> msgValue = {funcName.c_str(), paramName.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1011", msgKey, msgValue);
}
} // namespace
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static constexpr int32_t MIN_X_DIM = 2;
static constexpr int32_t MAX_X_DIM = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_BF16};

// EZ码细化：dtype 不在支持列表由泛化 EZ1001 细化为 EZ0020，dtype 不一致由泛化 EZ1001 细化为 EZ0021，
// 校验条件保持不变
static bool CheckDtypeInList(const aclTensor* tensor, const char* tensorName,
                             const std::initializer_list<op::DataType>& supportList)
{
    if (!CheckType(tensor->GetDataType(), supportList)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnAdaLayerNorm", tensorName, op::ToString(tensor->GetDataType()).GetString(),
            ("dtype should be in support list " + std::string(op::ToString(supportList).GetString())).c_str());
        return false;
    }
    return true;
}

static bool CheckDtypeSamePair(const aclTensor* tensor1, const aclTensor* tensor2, const char* name1, const char* name2)
{
    if (tensor1->GetDataType() != tensor2->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "aclnnAdaLayerNorm", (std::string(name1) + "," + name2).c_str(),
            (std::string(op::ToString(tensor1->GetDataType()).GetString()) + ", " +
             op::ToString(tensor2->GetDataType()).GetString())
                .c_str(),
            (std::string("dtype of ") + name2 + " should be the same as " + name1).c_str());
        return false;
    }
    return true;
}

static bool CheckNotNull(l0op::AdaLayerNormInputTensor& inputTensor, l0op::AdaLayerNormOutputTensor& outputTensor)
{
    OP_CHECK(inputTensor.x != nullptr, ReportNullPointerError(__func__, "inputTensor.x"), return false);
    OP_CHECK(inputTensor.scale != nullptr, ReportNullPointerError(__func__, "inputTensor.scale"), return false);
    OP_CHECK(inputTensor.shift != nullptr, ReportNullPointerError(__func__, "inputTensor.shift"), return false);
    OP_CHECK(outputTensor.out != nullptr, ReportNullPointerError(__func__, "outputTensor.out"), return false);
    return true;
}

static bool CheckDtypeValid(l0op::AdaLayerNormInputTensor& inputTensor, l0op::AdaLayerNormOutputTensor& outputTensor)
{
    if (!CheckDtypeInList(inputTensor.x, "x", DTYPE_SUPPORT_LIST)) {
        return false;
    }
    if (!CheckDtypeSamePair(inputTensor.x, inputTensor.scale, "x", "scale")) {
        return false;
    }
    if (!CheckDtypeSamePair(inputTensor.x, inputTensor.shift, "x", "shift")) {
        return false;
    }
    if (inputTensor.weightOptional != nullptr && inputTensor.biasOptional != nullptr) {
        // weight和bias都不为空时，数据类型保持一致
        if (!CheckDtypeSamePair(inputTensor.biasOptional, inputTensor.weightOptional, "biasOptional",
                                "weightOptional")) {
            return false;
        }
    }
    if (inputTensor.weightOptional != nullptr) {
        if (!outputTensor.useV2 || inputTensor.weightOptional->GetDataType() != op::DataType::DT_FLOAT) {
            if (!CheckDtypeSamePair(inputTensor.x, inputTensor.weightOptional, "x", "weightOptional")) {
                return false;
            }
        }
    }
    if (inputTensor.biasOptional != nullptr) {
        if (!outputTensor.useV2 || inputTensor.biasOptional->GetDataType() != op::DataType::DT_FLOAT) {
            if (!CheckDtypeSamePair(inputTensor.x, inputTensor.biasOptional, "x", "biasOptional")) {
                return false;
            }
        }
    }
    if (!CheckDtypeSamePair(inputTensor.x, outputTensor.out, "x", "out")) {
        return false;
    }
    if (outputTensor.meanOutOptional != nullptr) {
        if (!CheckDtypeSamePair(inputTensor.x, outputTensor.meanOutOptional, "x", "meanOutOptional")) {
            return false;
        }
    }
    if (outputTensor.rstdOutOptional != nullptr) {
        if (!CheckDtypeSamePair(inputTensor.x, outputTensor.rstdOutOptional, "x", "rstdOutOptional")) {
            return false;
        }
    }
    return true;
}

static bool CheckShape(l0op::AdaLayerNormInputTensor& inputTensor, l0op::AdaLayerNormOutputTensor& outputTensor)
{
    int64_t xDim = inputTensor.x->GetViewShape().GetDimNum();
    // EZ码细化：rank 越界由 EZ0011 细化为 EZ0012，校验条件保持不变
    OP_CHECK(xDim >= MIN_X_DIM && xDim <= MAX_X_DIM,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnAdaLayerNorm", "x", (std::to_string(xDim) + "D").c_str(),
                                                      "the number of dims of x should be between 2 and 8"),
             return false);

    op::Shape xShape = inputTensor.x->GetViewShape();
    int64_t B = 1;
    int64_t S = xShape[xDim - MIN_X_DIM];
    int64_t H = xShape[xDim - 1];
    op::Shape expectShape1;
    op::Shape expectShape2;
    for (int64_t i = 0; i < xDim - MIN_X_DIM; i++) {
        expectShape1.AppendDim(xShape[i]);
        expectShape2.AppendDim(xShape[i]);
        B *= xShape[i];
    }
    expectShape1.AppendDim(1);
    expectShape1.AppendDim(H);
    expectShape2.AppendDim(H);
    OP_CHECK(B > 0 && S > 0 && H > 0,
             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnAdaLayerNorm", "x",
                                                   op::ToString(inputTensor.x->GetViewShape()).GetString(),
                                                   "x sizes should be greater than 0"),
             return false);

    op::Shape scaleShape = inputTensor.scale->GetViewShape();
    op::Shape shiftShape = inputTensor.shift->GetViewShape();
    // 上报通道补齐：scale/shift shape 关系校验失败此前静默返回，补齐 EZ0010 上报，校验条件保持不变
    if (scaleShape != expectShape1 && scaleShape != expectShape2) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "aclnnAdaLayerNorm", "x,scale",
            (std::string(op::ToString(inputTensor.x->GetViewShape()).GetString()) + ", " +
             op::ToString(scaleShape).GetString())
                .c_str(),
            "shape of scale should be [B..., H] or [B..., 1, H] matching the leading dims of x");
        return false;
    }
    if (shiftShape != expectShape1 && shiftShape != expectShape2) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "aclnnAdaLayerNorm", "x,shift",
            (std::string(op::ToString(inputTensor.x->GetViewShape()).GetString()) + ", " +
             op::ToString(shiftShape).GetString())
                .c_str(),
            "shape of shift should be [B..., H] or [B..., 1, H] matching the leading dims of x");
        return false;
    }
    if (inputTensor.weightOptional != nullptr) {
        op::Shape weightExpectShape = {H};
        OP_CHECK(inputTensor.weightOptional->GetViewShape() == weightExpectShape,
                 OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                     "aclnnAdaLayerNorm", "inputTensor.weightOptional",
                     op::ToString(inputTensor.weightOptional->GetViewShape()).GetString(),
                     "shape of inputTensor.weightOptional should be the same as weightExpectShape"),
                 return false);
    }
    if (inputTensor.biasOptional != nullptr) {
        op::Shape biasExpectShape = {H};
        OP_CHECK(inputTensor.biasOptional->GetViewShape() == biasExpectShape,
                 OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                     "aclnnAdaLayerNorm", "inputTensor.biasOptional",
                     op::ToString(inputTensor.biasOptional->GetViewShape()).GetString(),
                     "shape of inputTensor.biasOptional should be the same as biasExpectShape"),
                 return false);
    }
    // EZ码细化：out/meanOutOptional/rstdOutOptional 与 x 的 shape 关系由 EZ0009（单参）细化为 EZ0010（多参关系），
    // 校验条件保持不变
    OP_CHECK(
        outputTensor.out->GetViewShape() == xShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnAdaLayerNorm", "x,out",
                                               (std::string(op::ToString(inputTensor.x->GetViewShape()).GetString()) +
                                                ", " + op::ToString(outputTensor.out->GetViewShape()).GetString())
                                                   .c_str(),
                                               "shape of out should be the same as x"),
        return false);
    xShape[xDim - 1] = 1;
    if (outputTensor.meanOutOptional != nullptr) {
        OP_CHECK(outputTensor.meanOutOptional->GetViewShape() == xShape,
                 OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                     "aclnnAdaLayerNorm", "x,meanOutOptional",
                     (std::string(op::ToString(inputTensor.x->GetViewShape()).GetString()) + ", " +
                      op::ToString(outputTensor.meanOutOptional->GetViewShape()).GetString())
                         .c_str(),
                     "shape of meanOutOptional should be the same as x except that the last dim should be 1"),
                 return false);
    }
    if (outputTensor.rstdOutOptional != nullptr) {
        OP_CHECK(outputTensor.rstdOutOptional->GetViewShape() == xShape,
                 OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                     "aclnnAdaLayerNorm", "x,rstdOutOptional",
                     (std::string(op::ToString(inputTensor.x->GetViewShape()).GetString()) + ", " +
                      op::ToString(outputTensor.rstdOutOptional->GetViewShape()).GetString())
                         .c_str(),
                     "shape of rstdOutOptional should be the same as x except that the last dim should be 1"),
                 return false);
    }
    return true;
}

static aclnnStatus CheckParams(l0op::AdaLayerNormInputTensor& inputTensor, l0op::AdaLayerNormOutputTensor& outputTensor)
{
    // 1. 检查参数是否为空指针
    CHECK_COND(CheckNotNull(inputTensor, outputTensor), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_COND(CheckDtypeValid(inputTensor, outputTensor), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

    // 3. 检查Shape是否支持
    CHECK_COND(CheckShape(inputTensor, outputTensor), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");

    return ACLNN_SUCCESS;
}

static aclnnStatus AdaLayerNormCalculate(l0op::AdaLayerNormInputTensor& inputTensor,
                                         l0op::AdaLayerNormOutputTensor& outputTensor, double epsilon,
                                         aclOpExecutor* executor)
{
    // 如果非连续，需要转连续
    inputTensor.x = l0op::Contiguous(inputTensor.x, executor);
    inputTensor.scale = l0op::Contiguous(inputTensor.scale, executor);
    inputTensor.shift = l0op::Contiguous(inputTensor.shift, executor);
    CHECK_RET(inputTensor.x != nullptr && inputTensor.scale != nullptr && inputTensor.shift != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    if (inputTensor.weightOptional != nullptr) {
        inputTensor.weightOptional = l0op::Contiguous(inputTensor.weightOptional, executor);
        CHECK_RET(inputTensor.weightOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (inputTensor.biasOptional != nullptr) {
        inputTensor.biasOptional = l0op::Contiguous(inputTensor.biasOptional, executor);
        CHECK_RET(inputTensor.biasOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* resultTensor = l0op::AdaLayerNorm(inputTensor, static_cast<float>(epsilon), executor);
    CHECK_RET(resultTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outResult = l0op::ViewCopy(resultTensor, outputTensor.out, executor);
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
}; // namespace

aclnnStatus aclnnAdaLayerNormGetWorkspaceSize(const aclTensor* x, const aclTensor* scale, const aclTensor* shift,
                                              const aclTensor* weightOptional, const aclTensor* biasOptional,
                                              double epsilon, aclTensor* out, uint64_t* workspaceSize,
                                              aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAdaLayerNorm, DFX_IN(x, scale, shift, weightOptional, biasOptional, epsilon), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    l0op::AdaLayerNormInputTensor inputTensor = {x, scale, shift, weightOptional, biasOptional};
    l0op::AdaLayerNormOutputTensor outputTensor = {out, nullptr, nullptr, false};
    auto ret = CheckParams(inputTensor, outputTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = AdaLayerNormCalculate(inputTensor, outputTensor, epsilon, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaLayerNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAdaLayerNorm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
