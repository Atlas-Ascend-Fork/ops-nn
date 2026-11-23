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
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def_nn.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/small_vector.h"
#include "opdev/platform.h"
#include "instance_norm_v3.h"
#include "aclnn_instance_norm.h"
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

constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr int IDX_2 = 2;
constexpr int IDX_3 = 3;

constexpr int DIM_NUM = 4;
constexpr int DIM_NUM_GAMMA = 1;

static const std::initializer_list<op::DataType> ASCEND310P_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                  op::DataType::DT_FLOAT16};

// EZ码细化：dtype 不在支持列表由泛化 EZ1001 细化为 EZ0020，dtype 不一致由泛化 EZ1001 细化为 EZ0021，
// rank 超上限由泛化 EZ1001 细化为 EZ0012，校验条件保持不变
static bool CheckDtypeInList(const aclTensor* tensor, const char* tensorName,
                             const std::initializer_list<op::DataType>& supportList)
{
    if (!CheckType(tensor->GetDataType(), supportList)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnInstanceNorm", tensorName, op::ToString(tensor->GetDataType()).GetString(),
            ("dtype should be in support list " + std::string(op::ToString(supportList).GetString())).c_str());
        return false;
    }
    return true;
}

static bool CheckDtypeSamePair(const aclTensor* tensor1, const aclTensor* tensor2, const char* name1, const char* name2)
{
    if (tensor1->GetDataType() != tensor2->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "aclnnInstanceNorm", (std::string(name1) + "," + name2).c_str(),
            (std::string(op::ToString(tensor1->GetDataType()).GetString()) + ", " +
             op::ToString(tensor2->GetDataType()).GetString())
                .c_str(),
            (std::string("dtype of ") + name2 + " should be the same as " + name1).c_str());
        return false;
    }
    return true;
}

static bool CheckDimNumMax(const aclTensor* tensor, const char* tensorName, size_t maxDim)
{
    if (tensor->GetViewShape().GetDimNum() > maxDim) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnInstanceNorm", tensorName, (std::to_string(tensor->GetViewShape().GetDimNum()) + "D").c_str(),
            ("the number of dims cannot exceed " + std::to_string(maxDim)).c_str());
        return false;
    }
    return true;
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, const aclTensor* y,
                         const aclTensor* mean, const aclTensor* variance, const char* dataFormat)
{
    OP_CHECK(x != nullptr, ReportNullPointerError(__func__, "x"), return false);
    OP_CHECK(gamma != nullptr, ReportNullPointerError(__func__, "gamma"), return false);
    OP_CHECK(beta != nullptr, ReportNullPointerError(__func__, "beta"), return false);
    OP_CHECK(y != nullptr, ReportNullPointerError(__func__, "y"), return false);
    OP_CHECK(mean != nullptr, ReportNullPointerError(__func__, "mean"), return false);
    OP_CHECK(variance != nullptr, ReportNullPointerError(__func__, "variance"), return false);
    // 上报通道补齐：dataFormat 空指针此前静默返回，按本文件空指针统一出口补齐 EZ1011 上报，校验条件保持不变
    OP_CHECK(dataFormat != nullptr, ReportNullPointerError(__func__, "dataFormat"), return false);
    return true;
}

static bool CheckParamValid(const char* dataFormat)
{
    // 检查参数 dataFormat 是否合法
    OP_LOGD("CheckParamValid: dataFormat: %s", dataFormat);
    OP_LOGD("CheckParamValid: nhwc: %d, nchw: %d", strncmp(dataFormat, "NHWC", strlen("NHWC")),
            strncmp(dataFormat, "NCHW", strlen("NCHW")));
    bool isValidDataFormat = strncmp(dataFormat, "NHWC", strlen("NHWC")) == 0 ||
                             strncmp(dataFormat, "NCHW", strlen("NCHW")) == 0;
    // 上报通道补齐：dataFormat 值域校验失败此前静默返回，补齐 EZ0026 上报，校验条件保持不变
    if (!isValidDataFormat) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnInstanceNorm", "dataFormat", dataFormat,
                                              "dataFormat should be NCHW or NHWC");
    }
    return isValidDataFormat;
}

static aclnnStatus CheckPlatform()
{
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P) {
        // SoC不支持统一上报EZ3002(已注册, Arglist=optype,opskernel,reason), 返回值保持不变
        const std::vector<const char*> ez3002Key = {"optype", "opskernel", "reason"};
        const std::vector<const char*> ez3002Value = {"aclnnInstanceNorm", "AiCore", "only Ascend310P is supported"};
        OP_LOGE_WITHOUT_REPORT("EZ3002", "%s", "aclnnInstanceNorm only supports Ascend310p");
        REPORT_PREDEFINED_ERR_MSG("EZ3002", ez3002Key, ez3002Value);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, const aclTensor* y,
                            const aclTensor* mean, const aclTensor* variance)
{
    // 检查 x 的数据类型是否在 InstanceNorm 算子的支持列表内
    if (!CheckDtypeInList(x, "x", ASCEND310P_DTYPE_SUPPORT_LIST)) {
        return false;
    }
    if (!CheckDtypeSamePair(x, gamma, "x", "gamma")) {
        return false;
    }
    if (!CheckDtypeSamePair(x, beta, "x", "beta")) {
        return false;
    }
    if (!CheckDtypeSamePair(x, y, "x", "y")) {
        return false;
    }
    if (!CheckDtypeSamePair(x, mean, "x", "mean")) {
        return false;
    }
    if (!CheckDtypeSamePair(x, variance, "x", "variance")) {
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* x)
{
    if (!CheckDimNumMax(x, "x", MAX_SUPPORT_DIMS_NUMS)) {
        return false;
    }
    return true;
}

static aclnnStatus CheckInputAndWeightShape(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta,
                                            bool needTranspose)
{
    int64_t xDimNums = x->GetViewShape().GetDimNum();
    int64_t gammaDimNums = gamma->GetViewShape().GetDimNum();
    int64_t betaDimNums = beta->GetViewShape().GetDimNum();
    OP_CHECK((xDimNums == DIM_NUM) && (gammaDimNums == DIM_NUM_GAMMA) && (betaDimNums == DIM_NUM_GAMMA),
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                 "aclnnInstanceNorm", "x/gamma/beta",
                 (std::to_string(xDimNums) + ", " + std::to_string(gammaDimNums) + ", " + std::to_string(betaDimNums))
                     .c_str(),
                 "x dim should be 4, gamma dim should be 1, beta dim should be 1"),
             return ACLNN_ERR_PARAM_INVALID);
    int64_t gammaDimValue = gamma->GetViewShape().GetDim(IDX_0);
    int64_t xCAxisValue = (needTranspose) ? x->GetViewShape().GetDim(IDX_3) : x->GetViewShape().GetDim(IDX_1);
    OP_LOGD("aclnnInstanceNormGetWorkspaceSize xCAxisValue: %ld, gammaDimValue: %ld", xCAxisValue, gammaDimValue);
    OP_CHECK(
        xCAxisValue == gammaDimValue,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnInstanceNorm", "gamma", std::to_string(gammaDimValue).c_str(),
                                                 "gamma.shape[0] must be equal to the C axis of x"),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta,
                               const char* dataFormat, const aclTensor* y, const aclTensor* mean,
                               const aclTensor* variance)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, gamma, beta, y, mean, variance, dataFormat), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查参数 dataFormat 是否合法
    CHECK_RET(CheckParamValid(dataFormat), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查 x, gamma, beta, y, mean, variance 的数据类型是否合法
    CHECK_RET(CheckDtypeValid(x, gamma, beta, y, mean, variance), ACLNN_ERR_PARAM_INVALID);

    // 4. 查输入tensor的shape是否为异常
    CHECK_RET(CheckShape(x), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ComputeInstanceNormOutputs(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta,
                                              const char* dataFormat, double eps, bool needTranspose,
                                              aclTensor** yComputeOut, aclTensor** meanComputeOut,
                                              aclTensor** varianceComputeOut, aclOpExecutor* executor)
{
    auto xContiguous = l0op::Contiguous(x, executor);
    auto gammaContiguous = l0op::Contiguous(gamma, executor);
    auto betaContiguous = l0op::Contiguous(beta, executor);
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gammaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(betaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (needTranspose) {
        std::vector<int64_t> permNHWC2NCHW = {IDX_0, IDX_3, IDX_1, IDX_2};
        std::vector<int64_t> permNCHW2NHWC = {IDX_0, IDX_2, IDX_3, IDX_1};
        aclIntArray* axesNHWC2NCHW = executor->AllocIntArray(permNHWC2NCHW.data(), DIM_NUM);
        aclIntArray* axesNCHW2NHWC = executor->AllocIntArray(permNCHW2NHWC.data(), DIM_NUM);
        CHECK_RET((axesNCHW2NHWC != nullptr && axesNHWC2NCHW != nullptr), ACLNN_ERR_INNER_NULLPTR);

        auto xTranspose = l0op::Transpose(xContiguous, axesNHWC2NCHW, executor);
        CHECK_RET(xTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const char* realDataFormat = "NCHW";
        auto instanceNormOut = l0op::InstanceNormV3(xTranspose, gammaContiguous, betaContiguous, realDataFormat, eps,
                                                    executor);
        *yComputeOut = std::get<IDX_0>(instanceNormOut);
        *meanComputeOut = std::get<IDX_1>(instanceNormOut);
        *varianceComputeOut = std::get<IDX_2>(instanceNormOut);
        CHECK_RET(*yComputeOut != nullptr && *meanComputeOut != nullptr && *varianceComputeOut != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
        auto yTransposed = l0op::Transpose(*yComputeOut, axesNCHW2NHWC, executor);
        CHECK_RET(yTransposed != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto yTransposedView = executor->CreateView(yTransposed, yTransposed->GetViewShape(),
                                                    yTransposed->GetViewOffset());
        CHECK_RET(yTransposedView != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *yComputeOut = yTransposedView;
    } else {
        auto instanceNormOut = l0op::InstanceNormV3(xContiguous, gammaContiguous, betaContiguous, dataFormat, eps,
                                                    executor);
        *yComputeOut = std::get<IDX_0>(instanceNormOut);
        *meanComputeOut = std::get<IDX_1>(instanceNormOut);
        *varianceComputeOut = std::get<IDX_2>(instanceNormOut);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CopyInstanceNormResults(const aclTensor* yComputeOut, const aclTensor* meanOut,
                                           const aclTensor* varianceOut, aclTensor* y, aclTensor* mean,
                                           aclTensor* variance, aclOpExecutor* executor)
{
    CHECK_RET(yComputeOut != nullptr && meanOut != nullptr && varianceOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyYResult = l0op::ViewCopy(yComputeOut, y, executor);
    CHECK_RET(viewCopyYResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyMeanResult = l0op::ViewCopy(meanOut, mean, executor);
    CHECK_RET(viewCopyMeanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyVarResult = l0op::ViewCopy(varianceOut, variance, executor);
    CHECK_RET(viewCopyVarResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInstanceNormGetWorkspaceSize(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta,
                                              const char* dataFormat, double eps, aclTensor* y, aclTensor* mean,
                                              aclTensor* variance, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("Enter aclnnInstanceNormGetWorkspaceSize");

    L2_DFX_PHASE_1(aclnnInstanceNorm, DFX_IN(x, gamma, beta, dataFormat, eps), DFX_OUT(y, mean, variance));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, gamma, beta, dataFormat, y, mean, variance);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckPlatform();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    bool needTranspose = (strncmp(dataFormat, "NHWC", strlen("NHWC")) == 0);
    OP_LOGD("aclnnInstanceNormGetWorkspaceSize needTranspose: %d", needTranspose);
    ret = CheckInputAndWeightShape(x, gamma, beta, needTranspose);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    bool hasEmptyTensor = x->IsEmpty() || gamma->IsEmpty() || beta->IsEmpty() || y->IsEmpty() || mean->IsEmpty() ||
                          variance->IsEmpty();
    if (hasEmptyTensor) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    aclTensor *yComputeOut = nullptr, *meanOut = nullptr, *varianceOut = nullptr;
    ret = ComputeInstanceNormOutputs(x, gamma, beta, dataFormat, eps, needTranspose, &yComputeOut, &meanOut,
                                     &varianceOut, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (needTranspose) {
        int64_t nhwcReduceShapeValue[DIM_NUM] = {(meanOut)->GetViewShape().GetDim(0), 1, 1,
                                                 (meanOut)->GetViewShape().GetDim(1)};
        aclIntArray* nhwcReduceShape = uniqueExecutor.get()->AllocIntArray(nhwcReduceShapeValue, DIM_NUM);
        CHECK_RET(nhwcReduceShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanReshaped = l0op::Reshape(meanOut, nhwcReduceShape, uniqueExecutor.get());
        CHECK_RET(meanReshaped != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanReshapedView = uniqueExecutor.get()->CreateView(meanReshaped, meanReshaped->GetViewShape(),
                                                                 meanReshaped->GetViewOffset());
        CHECK_RET(meanReshapedView != nullptr, ACLNN_ERR_INNER_NULLPTR);
        meanOut = meanReshapedView;
        auto varianceReshaped = l0op::Reshape(varianceOut, nhwcReduceShape, uniqueExecutor.get());
        CHECK_RET(varianceReshaped != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto varianceReshapedView = uniqueExecutor.get()->CreateView(varianceReshaped, varianceReshaped->GetViewShape(),
                                                                     varianceReshaped->GetViewOffset());
        CHECK_RET(varianceReshapedView != nullptr, ACLNN_ERR_INNER_NULLPTR);
        varianceOut = varianceReshapedView;
    }
    ret = CopyInstanceNormResults(yComputeOut, meanOut, varianceOut, y, mean, variance, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("Finish aclnnInstanceNormGetWorkspaceSize");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInstanceNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInstanceNorm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
