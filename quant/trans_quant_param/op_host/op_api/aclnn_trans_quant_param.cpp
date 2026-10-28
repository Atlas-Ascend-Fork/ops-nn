/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_trans_quant_param.h"

#include <vector>

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "quant_util.h"
#include "log/log.h"

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
static const size_t ALIGN_NUM = 16;

static aclnnStatus CheckQuantParam(const float* offsetArray, uint64_t offsetSize, const float* scaleArray,
                                   uint64_t scaleSize)
{
    if (offsetArray != nullptr) {
        if (offsetSize < 1) {
            OP_LOGE_FOR_INVALID_LISTSIZE("aclnnTransQuantParam", "offsetSize", std::to_string(offsetSize).c_str(),
                                         "at least 1");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (offsetSize != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnTransQuantParam", "offsetSize",
                                                  std::to_string(offsetSize).c_str(),
                                                  "when offsetArray is nullptr, offsetSize must be 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (scaleArray != nullptr) {
        if (scaleSize < 1) {
            OP_LOGE_FOR_INVALID_LISTSIZE("aclnnTransQuantParam", "scaleSize", std::to_string(scaleSize).c_str(),
                                         "at least 1");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        ReportNullPointerError(__func__, "scaleArray");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (scaleSize == 0 && offsetSize == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnTransQuantParam", "scaleSize/offsetSize", "0",
                                              "offsetSize and scaleSize must not both be 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
}; // namespace

aclnnStatus aclnnTransQuantParam(const float* scaleArray, uint64_t scaleSize, const float* offsetArray,
                                 uint64_t offsetSize, uint64_t** quantParam, uint64_t* quantParamSize)
{
    CHECK_RET(CheckQuantParam(offsetArray, offsetSize, scaleArray, scaleSize) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(quantParam != nullptr, ReportNullPointerError(__func__, "quantParam"), return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(quantParamSize != nullptr, ReportNullPointerError(__func__, "quantParamSize"),
             return ACLNN_ERR_PARAM_NULLPTR);

    // align
    *quantParamSize = (scaleSize > offsetSize) ? scaleSize : offsetSize;
    if (*quantParamSize != 1) {
        auto remainder = *quantParamSize % ALIGN_NUM;
        if (remainder != 0) {
            *quantParamSize = *quantParamSize + ALIGN_NUM - remainder;
        }
    }
    *quantParam = (uint64_t*)calloc(*quantParamSize, sizeof(uint64_t));

    if (*quantParam == nullptr) {
        // calloc失败属内部资源错误, 不复用EZ1011入参空指针语义; 上报EZ1009(执行失败)并保持原返回值
        const std::vector<const char*> ez1009Key = {"opName", "reason"};
        const std::vector<const char*> ez1009Value = {"aclnnTransQuantParam",
                                                      "memory allocation for quantParam failed (out of memory)"};
        REPORT_PREDEFINED_ERR_MSG("EZ1009", ez1009Key, ez1009Value);
        *quantParamSize = 0;
        return ACLNN_ERR_INNER_NULLPTR;
    }

    if (scaleArray != nullptr) {
        TransQuantScaleToM1(scaleArray, scaleSize, quantParam);
    }

    if (offsetArray != nullptr) {
        TransQuantOffsetToToM1(offsetArray, offsetSize, quantParam);
    }

    if (*quantParam == nullptr) {
        // calloc失败属内部资源错误, 不复用EZ1011入参空指针语义; 上报EZ1009(执行失败)并保持原返回值
        const std::vector<const char*> ez1009Key = {"opName", "reason"};
        const std::vector<const char*> ez1009Value = {"aclnnTransQuantParam",
                                                      "memory allocation for quantParam failed (out of memory)"};
        REPORT_PREDEFINED_ERR_MSG("EZ1009", ez1009Key, ez1009Value);
        *quantParamSize = 0;
        return ACLNN_ERR_INNER_NULLPTR;
    }

    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
