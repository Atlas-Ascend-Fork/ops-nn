/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "op_api/aclnn_util.h"
#include "ada_layer_norm_quant.h"
#include "aclnn_ada_layer_norm_quant.h"
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
struct AdaLayerNormQuantInputTensor {
    const aclTensor* x;
    const aclTensor* scale;
    const aclTensor* shift;
    const aclTensor* weightOptional;
    const aclTensor* biasOptional;
    const aclTensor* smoothScalesOptional;
};

struct AdaLayerNormQuantOutputTensor {
    aclTensor* out;
    aclTensor* quantScale;
    aclTensor* quantOffsetOptional;
};

static constexpr int32_t MIN_X_DIM = 2;
static constexpr int32_t MAX_X_DIM = 8;

static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                         op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT8};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_INT8, op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN};

static const std::initializer_list<op::DataType> SCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

// EZ码细化：dtype 不在支持列表由泛化 EZ1001 细化为 EZ0020，多参 dtype 不一致由泛化 EZ1001 细化为 EZ0021，
// 校验条件保持不变
static bool CheckDtypeInList(const aclTensor* tensor, const char* tensorName,
                             const std::initializer_list<op::DataType>& supportList)
{
    if (!CheckType(tensor->GetDataType(), supportList)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnAdaLayerNormQuant", tensorName, op::ToString(tensor->GetDataType()).GetString(),
            ("dtype should be in support list " + std::string(op::ToString(supportList).GetString())).c_str());
        return false;
    }
    return true;
}

static bool CheckDtypeSamePair(const aclTensor* tensor1, const aclTensor* tensor2, const char* name1, const char* name2)
{
    if (tensor1->GetDataType() != tensor2->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "aclnnAdaLayerNormQuant", (std::string(name1) + "," + name2).c_str(),
            (std::string(op::ToString(tensor1->GetDataType()).GetString()) + ", " +
             op::ToString(tensor2->GetDataType()).GetString())
                .c_str(),
            (std::string("dtype of ") + name2 + " should be the same as " + name1).c_str());
        return false;
    }
    return true;
}

static bool CheckNotNull(AdaLayerNormQuantInputTensor& inputTensor, AdaLayerNormQuantOutputTensor& outputTensor)
{
    OP_CHECK(inputTensor.x != nullptr, ReportNullPointerError(__func__, "inputTensor.x"), return false);
    OP_CHECK(inputTensor.scale != nullptr, ReportNullPointerError(__func__, "inputTensor.scale"), return false);
    OP_CHECK(inputTensor.shift != nullptr, ReportNullPointerError(__func__, "inputTensor.shift"), return false);
    OP_CHECK(outputTensor.out != nullptr, ReportNullPointerError(__func__, "outputTensor.out"), return false);
    OP_CHECK(outputTensor.quantScale != nullptr, ReportNullPointerError(__func__, "outputTensor.quantScale"),
             return false);
    return true;
}

static bool CheckDtypeValid(AdaLayerNormQuantInputTensor& inputTensor, AdaLayerNormQuantOutputTensor& outputTensor)
{
    if (!CheckDtypeInList(inputTensor.x, "x", X_DTYPE_SUPPORT_LIST)) {
        return false;
    }
    if (!CheckDtypeSamePair(inputTensor.x, inputTensor.scale, "x", "scale")) {
        return false;
    }
    if (!CheckDtypeSamePair(inputTensor.x, inputTensor.shift, "x", "shift")) {
        return false;
    }
    if (inputTensor.weightOptional != nullptr) {
        if (!CheckDtypeSamePair(inputTensor.x, inputTensor.weightOptional, "x", "weightOptional")) {
            return false;
        }
    }
    if (inputTensor.biasOptional != nullptr) {
        if (!CheckDtypeSamePair(inputTensor.x, inputTensor.biasOptional, "x", "biasOptional")) {
            return false;
        }
    }
    if (inputTensor.smoothScalesOptional != nullptr) {
        if (!CheckDtypeSamePair(inputTensor.x, inputTensor.smoothScalesOptional, "x", "smoothScalesOptional")) {
            return false;
        }
    }
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        if (!CheckDtypeInList(outputTensor.out, "out", OUT_DTYPE_SUPPORT_LIST_REGBASE)) {
            return false;
        }
    } else {
        if (!CheckDtypeInList(outputTensor.out, "out", OUT_DTYPE_SUPPORT_LIST)) {
            return false;
        }
    }
    if (!CheckDtypeInList(outputTensor.quantScale, "quantScale", SCALE_DTYPE_SUPPORT_LIST)) {
        return false;
    }
    if (outputTensor.quantOffsetOptional != nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnAdaLayerNormQuant", "quantOffsetOptional",
                                                 "quantOffsetOptional should be nullptr");
        return false;
    }
    return true;
}

static bool CheckShape(AdaLayerNormQuantInputTensor& inputTensor, AdaLayerNormQuantOutputTensor& outputTensor)
{
    int64_t xDim = inputTensor.x->GetViewShape().GetDimNum();
    // EZ码细化：rank 越界由 EZ0011（带正确 dim）细化为 EZ0012（带 reason），诊断取值与其他算子统一为 ND 形式，
    // 校验条件保持不变
    OP_CHECK(
        xDim >= MIN_X_DIM && xDim <= MAX_X_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnAdaLayerNormQuant", "x", (std::to_string(xDim) + "D").c_str(),
                                                 "the number of dims of x should be between 2 and 8"),
        return false);

    op::Shape xShape = inputTensor.x->GetViewShape();
    op::Shape expectShape1;
    op::Shape expectShape2;
    op::Shape scaleExpectShape;
    int64_t B = 1;
    int64_t S = xShape[xDim - MIN_X_DIM];
    int64_t H = xShape[xDim - 1];
    for (int64_t i = 0; i < xDim - MIN_X_DIM; i++) {
        expectShape1.AppendDim(xShape[i]);
        expectShape2.AppendDim(xShape[i]);
        scaleExpectShape.AppendDim(xShape[i]);
        B *= xShape[i];
    }
    expectShape1.AppendDim(1);
    expectShape1.AppendDim(H);
    expectShape2.AppendDim(H);
    scaleExpectShape.AppendDim(S);
    OP_CHECK(B > 0 && S > 0 && H > 0,
             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnAdaLayerNormQuant", "x",
                                                   op::ToString(inputTensor.x->GetViewShape()).GetString(),
                                                   "x sizes should be greater than 0"),
             return false);

    op::Shape scaleShape = inputTensor.scale->GetViewShape();
    op::Shape shiftShape = inputTensor.shift->GetViewShape();
    // EZ码细化：scale/shift 与 x 的 shape 关系缺失上报通道，补齐为 EZ0010，校验条件保持不变
    if (scaleShape != expectShape1 && scaleShape != expectShape2) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "aclnnAdaLayerNormQuant", "x,scale",
            (std::string(op::ToString(xShape).GetString()) + ", " + op::ToString(scaleShape).GetString()).c_str(),
            "shape of scale should match x, with the last two dims being [1, H] or [H]");
        return false;
    }
    if (shiftShape != expectShape1 && shiftShape != expectShape2) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "aclnnAdaLayerNormQuant", "x,shift",
            (std::string(op::ToString(xShape).GetString()) + ", " + op::ToString(shiftShape).GetString()).c_str(),
            "shape of shift should match x, with the last two dims being [1, H] or [H]");
        return false;
    }
    op::Shape expectShape = {H};
    if (inputTensor.weightOptional != nullptr) {
        OP_CHECK(inputTensor.weightOptional->GetViewShape() == expectShape,
                 OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                     "aclnnAdaLayerNormQuant", "inputTensor.weightOptional",
                     op::ToString(inputTensor.weightOptional->GetViewShape()).GetString(),
                     "shape of inputTensor.weightOptional should be the same as expectShape"),
                 return false);
    }
    if (inputTensor.biasOptional != nullptr) {
        OP_CHECK(inputTensor.biasOptional->GetViewShape() == expectShape,
                 OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                     "aclnnAdaLayerNormQuant", "inputTensor.biasOptional",
                     op::ToString(inputTensor.biasOptional->GetViewShape()).GetString(),
                     "shape of inputTensor.biasOptional should be the same as expectShape"),
                 return false);
    }
    if (inputTensor.smoothScalesOptional != nullptr) {
        OP_CHECK(inputTensor.smoothScalesOptional->GetViewShape() == expectShape,
                 OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                     "aclnnAdaLayerNormQuant", "inputTensor.smoothScalesOptional",
                     op::ToString(inputTensor.smoothScalesOptional->GetViewShape()).GetString(),
                     "shape of inputTensor.smoothScalesOptional should be the same as expectShape"),
                 return false);
    }
    // EZ码细化：out/quantScale 与 x 的 shape 关系由 EZ0009（单参）细化为 EZ0010（多参关系），校验条件保持不变
    OP_CHECK(outputTensor.out->GetViewShape() == xShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnAdaLayerNormQuant", "x,out",
                                                    (std::string(op::ToString(xShape).GetString()) + ", " +
                                                     op::ToString(outputTensor.out->GetViewShape()).GetString())
                                                        .c_str(),
                                                    "shape of out should be the same as x"),
             return false);
    OP_CHECK(outputTensor.quantScale->GetViewShape() == scaleExpectShape,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "aclnnAdaLayerNormQuant", "quantScale",
                 (std::string(op::ToString(scaleExpectShape).GetString()) + ", " +
                  op::ToString(outputTensor.quantScale->GetViewShape()).GetString())
                     .c_str(),
                 "shape of quantScale should match the leading dims of x; the first reported shape is the expected "
                 "shape derived from x, and the second is the actual shape of quantScale"),
             return false);
    return true;
}

static bool CheckAttr(const char* quantMode)
{
    std::string mode = "dynamic";
    if (quantMode != nullptr) {
        mode = std::string(quantMode);
    }
    if (mode != "dynamic") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnAdaLayerNormQuant", "quantMode", mode.c_str(),
                                              "quantMode only supports dynamic");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(AdaLayerNormQuantInputTensor& inputTensor, AdaLayerNormQuantOutputTensor& outputTensor,
                               const char* quantMode)
{
    // 1. 检查参数是否为空指针
    CHECK_COND(CheckNotNull(inputTensor, outputTensor), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_COND(CheckDtypeValid(inputTensor, outputTensor), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

    // 3. 检查Shape是否支持
    CHECK_COND(CheckShape(inputTensor, outputTensor), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");

    // 4. 检查Attr是否合法
    CHECK_COND(CheckAttr(quantMode), ACLNN_ERR_PARAM_INVALID, "CheckAttr failed!");

    return ACLNN_SUCCESS;
}
}; // namespace

aclnnStatus aclnnAdaLayerNormQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* scale, const aclTensor* shift,
                                                   const aclTensor* weightOptional, const aclTensor* biasOptional,
                                                   const aclTensor* smoothScalesOptional, double epsilon,
                                                   const char* quantMode, aclTensor* out, aclTensor* quantScale,
                                                   aclTensor* quantOffsetOptional, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAdaLayerNormQuant,
                   DFX_IN(x, scale, shift, weightOptional, biasOptional, smoothScalesOptional, epsilon, quantMode),
                   DFX_OUT(out, quantScale, quantOffsetOptional));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    AdaLayerNormQuantInputTensor inputTensor = {x, scale, shift, weightOptional, biasOptional, smoothScalesOptional};
    AdaLayerNormQuantOutputTensor outputTensor = {out, quantScale, quantOffsetOptional};
    auto ret = CheckParams(inputTensor, outputTensor, quantMode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor直接返回
    if (x->IsEmpty() || scale->IsEmpty() || shift->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 如果非连续，需要转连续
    x = l0op::Contiguous(x, uniqueExecutor.get());
    scale = l0op::Contiguous(scale, uniqueExecutor.get());
    shift = l0op::Contiguous(shift, uniqueExecutor.get());
    CHECK_RET(x != nullptr && scale != nullptr && shift != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (weightOptional != nullptr) {
        weightOptional = l0op::Contiguous(weightOptional, uniqueExecutor.get());
        CHECK_RET(weightOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (biasOptional != nullptr) {
        biasOptional = l0op::Contiguous(biasOptional, uniqueExecutor.get());
        CHECK_RET(biasOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (smoothScalesOptional != nullptr) {
        smoothScalesOptional = l0op::Contiguous(smoothScalesOptional, uniqueExecutor.get());
        CHECK_RET(smoothScalesOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    int32_t dstType = out->GetDataType();
    std::tuple<aclTensor*, aclTensor*> result = l0op::AdaLayerNormQuant(
        x, scale, shift, weightOptional, biasOptional, smoothScalesOptional, static_cast<float>(epsilon), quantMode,
        dstType, uniqueExecutor.get());
    const aclTensor* resultTensor = std::get<0>(result);
    const aclTensor* quantScaleTensor = std::get<1>(result);
    CHECK_RET(resultTensor != nullptr && quantScaleTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outResult = l0op::ViewCopy(resultTensor, out, uniqueExecutor.get());
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto scaleResult = l0op::ViewCopy(quantScaleTensor, quantScale, uniqueExecutor.get());
    CHECK_RET(scaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaLayerNormQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAdaLayerNormQuant);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
