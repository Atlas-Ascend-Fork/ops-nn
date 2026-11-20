/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_layer_norm_quant.h"
#include "layer_norm_quant.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#ifndef OP_LOG_LIBOPAPI_ONLY
#define OP_LOG_LIBOPAPI_ONLY
#endif
#include "log/log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t NORMALIZE_LEN = 3;
constexpr size_t MAX_DIM_LEN = 8;
constexpr size_t GAMMA_STATIC_DIM = 2;
constexpr size_t GRAD_INPUT_INDEX = 0;
constexpr size_t GRAD_WEIGHT_INDEX = 1;
constexpr size_t GRAD_BIAS_INDEX = 2;
constexpr size_t LEAST_NORMALIZED_SHAPE_LEN = 1;
constexpr size_t GRAD_OUT_NUM = 3;
constexpr size_t X_OUT_NUM = 2;
constexpr size_t BETA_GAMMAX_NUM = 2;
constexpr int64_t MIN_GRAD_REDUCE_AXIS = 1024;
constexpr int64_t MAX_GRAD_REDUCE_AXIS = 4096;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_REGBASE_LIST = {DataType::DT_FLOAT16, DataType::DT_FLOAT,
                                                                           DataType::DT_BF16};
static const std::initializer_list<DataType> DTYPE_SUPPORT_910B_LIST = {DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<DataType> DTYPE_SUPPORT_310P_LIST = {DataType::DT_FLOAT16};

// EZ码细化：dtype 不在支持列表由泛化上报细化为 EZ0020，多参 dtype 不一致由泛化上报细化为 EZ0021，
// dtype 与期望值不符由泛化上报细化为 EZ0020，rank 超上限由泛化上报细化为 EZ0012，
// shape 关系不匹配由泛化上报细化为 EZ0010，shape 值不符由泛化上报细化为 EZ0009，
// 维数关系不符由泛化上报细化为 EZ0013，参数值非法由泛化上报细化为 EZ0023，校验条件均保持不变
static bool CheckDtypeInList(const aclTensor* tensor, const char* tensorName,
                             const std::initializer_list<DataType>& supportList)
{
    if (!CheckType(tensor->GetDataType(), supportList)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnLayerNormQuant", tensorName, op::ToString(tensor->GetDataType()).GetString(),
            ("dtype should be in support list " + std::string(op::ToString(supportList).GetString())).c_str());
        return false;
    }
    return true;
}

static bool CheckDtypeSamePair(const aclTensor* tensor1, const aclTensor* tensor2, const char* name1, const char* name2)
{
    if (tensor1->GetDataType() != tensor2->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "aclnnLayerNormQuant", (std::string(name1) + "," + name2).c_str(),
            (std::string(op::ToString(tensor1->GetDataType()).GetString()) + ", " +
             op::ToString(tensor2->GetDataType()).GetString())
                .c_str(),
            (std::string("dtype of ") + name2 + " should be the same as " + name1).c_str());
        return false;
    }
    return true;
}

static bool CheckDtypeMatchValue(const aclTensor* tensor, DataType expectedDtype, const char* tensorName)
{
    if (tensor->GetDataType() != expectedDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnLayerNormQuant", tensorName, op::ToString(tensor->GetDataType()).GetString(),
            (std::string("dtype of ") + tensorName + " must be " + op::ToString(expectedDtype).GetString()).c_str());
        return false;
    }
    return true;
}

static bool CheckDimNumMax(const aclTensor* tensor, const char* tensorName, size_t maxDim)
{
    if (tensor->GetViewShape().GetDimNum() > maxDim) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnLayerNormQuant", tensorName, (std::to_string(tensor->GetViewShape().GetDimNum()) + "D").c_str(),
            ("the number of dims cannot exceed " + std::to_string(maxDim)).c_str());
        return false;
    }
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();

    if (Ops::NN::AclnnUtil::IsRegbase(curArch)) {
        return DTYPE_SUPPORT_REGBASE_LIST;
    } else if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
        return DTYPE_SUPPORT_910B_LIST;
    } else if (socVersion == SocVersion::ASCEND310P) {
        return DTYPE_SUPPORT_310P_LIST;
    } else {
        return DTYPE_SUPPORT_910B_LIST;
    }
}

static inline bool CheckPlatform()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(
        Ops::NN::AclnnUtil::IsRegbase(curArch),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Support for npuArch %u is not implemented.", static_cast<uint32_t>(curArch)),
        return false);
    return true;
}

inline static bool CheckNotNull(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta,
                                const aclTensor* scale, const aclTensor* zeroPointsOptional, int quantMode,
                                aclTensor* res, aclTensor* scaleOut)
{
    // 校验输入是否为空指针
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(gamma, return false);
    OP_CHECK_NULL(beta, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(res, return false);
    OP_CHECK_NULL(scaleOut, return false);
    if (quantMode == 0) {
        OP_CHECK_NULL(zeroPointsOptional, return false);
    }
    return true;
}

static inline bool CheckOptInputDtype(const aclTensor* tensorPtr, op::DataType dtype)
{
    if (tensorPtr == nullptr) {
        return true;
    }
    if (!CheckDtypeMatchValue(tensorPtr, dtype, "zeroPointsOptional")) {
        return false;
    }
    return true;
}

static bool CheckDtype(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, const aclTensor* scale,
                       const aclTensor* zeroPointsOptional, const aclTensor* res, const aclTensor* scaleOut)
{
    // 检查 x 的数据类型是否在 AddLayerNormQuant 算子的支持列表内
    if (!CheckDtypeInList(x, "x", GetDtypeSupportList())) {
        return false;
    }
    if (!CheckDtypeSamePair(x, gamma, "x", "gamma")) {
        return false;
    }
    if (!CheckDtypeSamePair(x, beta, "x", "beta")) {
        return false;
    }
    if (!CheckDtypeSamePair(x, scale, "x", "scale")) {
        return false;
    }

    CHECK_RET(CheckOptInputDtype(zeroPointsOptional, op::DataType::DT_INT8), false);

    if (!CheckDtypeMatchValue(res, op::DataType::DT_INT8, "res")) {
        return false;
    }
    if (!CheckDtypeMatchValue(scaleOut, op::DataType::DT_FLOAT, "scaleOut")) {
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta, const aclTensor* scale,
                       const aclTensor* zeroPointsOptional, int quantMode, const aclTensor* res,
                       const aclTensor* scaleOut)
{
    if (!CheckDimNumMax(x, "x", MAX_DIM_LEN)) {
        return false;
    }
    if (!CheckDimNumMax(gamma, "gamma", GAMMA_STATIC_DIM)) {
        return false;
    }

    size_t xDimNums = x->GetViewShape().GetDimNum();
    size_t gammaDimNums = gamma->GetViewShape().GetDimNum();

    // 当前只支持quantMode = 0
    OP_CHECK(
        (quantMode == 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnLayerNormQuant", "quantMode", std::to_string(quantMode).c_str(),
                                              "Only support static mode quant currently, quantMode should be 0"),
        return false);

    OP_CHECK((x->GetViewShape().GetDim(xDimNums - 1) == gamma->GetViewShape().GetDim(gammaDimNums - 1)),
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnLayerNormQuant", "gamma and x",
                                                    (std::string(op::ToString(gamma->GetViewShape()).GetString()) +
                                                     ", " + op::ToString(x->GetViewShape()).GetString())
                                                        .c_str(),
                                                    "the last dim of gamma should be equal to the last dim of x"),
             return false);
    OP_CHECK((gamma->GetViewShape() == beta->GetViewShape()),
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnLayerNormQuant", "gamma and beta",
                                                    (std::string(op::ToString(gamma->GetViewShape()).GetString()) +
                                                     ", " + op::ToString(beta->GetViewShape()).GetString())
                                                        .c_str(),
                                                    "shape of beta should be the same as shape of gamma"),
             return false);
    // quantMode为0时gamma支持2维且第一维为1，避免仅读取第一行导致其余行被静默丢弃
    if (!gamma->IsEmpty() && gammaDimNums == GAMMA_STATIC_DIM) {
        OP_CHECK((gamma->GetViewShape().GetDim(0) == 1),
                 OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnLayerNormQuant", "gamma",
                                                       op::ToString(gamma->GetViewShape()).GetString(),
                                                       "First dim of gamma should be 1 when gamma is 2d"),
                 return false);
    }

    size_t scaleDimNums = scale->GetViewShape().GetDimNum();
    OP_CHECK((scaleDimNums == 1 && scale->GetViewShape().GetDim(0) == 1),
             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnLayerNormQuant", "scale",
                                                   op::ToString(scale->GetViewShape()).GetString(),
                                                   "Shape of scale should be equal to [1]"),
             return false);
    // zeroPointsOptional的shape需要与scale保持一致（即[1]），避免仅读取首个元素导致其余元素被静默丢弃
    if (!zeroPointsOptional->IsEmpty()) {
        OP_CHECK(
            (zeroPointsOptional->GetViewShape() == scale->GetViewShape()),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnLayerNormQuant", "scale and zeroPointsOptional",
                                                   (std::string(op::ToString(scale->GetViewShape()).GetString()) +
                                                    ", " + op::ToString(zeroPointsOptional->GetViewShape()).GetString())
                                                       .c_str(),
                                                   "Shape of zeroPointsOptional should be same as scale"),
            return false);
    }

    OP_CHECK((x->GetViewShape() == res->GetViewShape()),
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnLayerNormQuant", "x and res",
                                                    (std::string(op::ToString(x->GetViewShape()).GetString()) + ", " +
                                                     op::ToString(res->GetViewShape()).GetString())
                                                        .c_str(),
                                                    "shape of res should be the same as shape of x"),
             return false);

    // scaleOut的shape为x的shape剔除最后一维
    size_t scaleOutDimNums = scaleOut->GetViewShape().GetDimNum();
    OP_CHECK((scaleOutDimNums + 1 == xDimNums),
             OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                 "aclnnLayerNormQuant", "x and scaleOut",
                 (std::to_string(xDimNums) + "D and " + std::to_string(scaleOutDimNums) + "D").c_str(),
                 "the dim num of scaleOut should be the dim num of x minus 1"),
             return false);
    for (size_t i = 0; i < scaleOutDimNums; i++) {
        OP_CHECK((scaleOut->GetViewShape().GetDim(i) == x->GetViewShape().GetDim(i)),
                 OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnLayerNormQuant", "x and scaleOut",
                                                        (std::string(op::ToString(x->GetViewShape()).GetString()) +
                                                         ", " + op::ToString(scaleOut->GetViewShape()).GetString())
                                                            .c_str(),
                                                        "Shape of scaleOut should be x shape without last dim"),
                 return false);
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta,
                               const aclTensor* scale, const aclTensor* zeroPointsOptional, int quantMode,
                               aclTensor* res, aclTensor* scaleOut)
{
    // 1. 检查数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtype(x, gamma, beta, scale, zeroPointsOptional, res, scaleOut), ACLNN_ERR_PARAM_INVALID);
    // 2. 检查入参间的shape关系
    CHECK_RET(CheckShape(x, gamma, beta, scale, zeroPointsOptional, quantMode, res, scaleOut), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLayerNormQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta,
                                                const aclTensor* scale, const aclTensor* zeroPointsOptional,
                                                int quantMode, double epsilon, aclTensor* res, aclTensor* scaleOut,
                                                uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGI("aclnnLayerNormQuantGetWorkspaceSize start.");
    L2_DFX_PHASE_1(aclnnLayerNormQuant, DFX_IN(x, gamma, beta, scale, zeroPointsOptional, quantMode, epsilon),
                   DFX_OUT(res, scaleOut));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 提前检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, gamma, beta, scale, zeroPointsOptional, quantMode, res, scaleOut),
              ACLNN_ERR_PARAM_NULLPTR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, gamma, beta, scale, zeroPointsOptional, quantMode, res, scaleOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor场景处理
    if (x->IsEmpty()) {
        OP_LOGW("Got empty tensor in aclnnLayerNormQuant!");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的tensor
    auto xCont = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gammaCont = l0op::Contiguous(gamma, uniqueExecutor.get());
    CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto betaCont = l0op::Contiguous(beta, uniqueExecutor.get());
    CHECK_RET(betaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto scaleCont = l0op::Contiguous(scale, uniqueExecutor.get());
    CHECK_RET(scaleCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto zeroPointsOptionalCont = l0op::Contiguous(zeroPointsOptional, uniqueExecutor.get());
    CHECK_RET(zeroPointsOptionalCont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // LayerNormQuant returns two tensors: y and the optional scaleOut
    constexpr size_t kLayerNormQuantResultNum = 2U;
    std::array<aclTensor*, kLayerNormQuantResultNum> result = l0op::LayerNormQuant(
        xCont, gammaCont, betaCont, scaleCont, zeroPointsOptionalCont, quantMode, epsilon, uniqueExecutor.get());
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResResult = l0op::ViewCopy(result[0], res, uniqueExecutor.get());
    CHECK_RET(viewCopyResResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (quantMode == 1) {
        CHECK_RET(result[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyScaleResult = l0op::ViewCopy(result[1], scaleOut, uniqueExecutor.get());
        CHECK_RET(viewCopyScaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLayerNormQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLayerNormQuant);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
