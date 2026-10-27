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
 * \file foreach_pow_scalar_and_tensor_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "../../foreach_utils/op_host/common_dtype.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/storage_shape.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4ForeachPowScalarAndTensor(gert::InferShapeContext* context)
{
    uint32_t outputNum = context->GetComputeNodeOutputNum();
    const auto inputInfo = context->GetIrInputInstanceInfo(1);
    if (inputInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::string errMsg = optiling::ConcatString("num of dynamic input1 ", inputInfo->GetInstanceNum(),
                                                "not equal num of dynamic output0 ", outputNum);
    OP_CHECK_IF(inputInfo->GetInstanceNum() != outputNum, ([&]() {
                    const char* e9NodeName = context->GetNodeName();
                    const std::string e9Op = (e9NodeName == nullptr ? "ForeachPowScalarAndTensor" : e9NodeName);
                    const std::string e9Parameter = "x";
                    const std::string e9Size = std::to_string(inputInfo->GetInstanceNum());
                    const std::string e9Reason = (std::string(errMsg.c_str()));
                    OP_LOGE_WITHOUT_REPORT(e9Op.c_str(), "Parameter %s has invalid list size %s: %s.",
                                           e9Parameter.c_str(), e9Size.c_str(), e9Reason.c_str());
                    const std::vector<const char*> e9Keys = {"param_name", "op_name", "incorrect_size", "reason"};
                    const std::vector<const char*> e9Values = {e9Parameter.c_str(), e9Op.c_str(), e9Size.c_str(),
                                                               e9Reason.c_str()};
                    REPORT_PREDEFINED_ERR_MSG("EZ0038", e9Keys, e9Values);
                }()),
                return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputInfo->GetInstanceNum(); i++) {
        auto xShape = context->GetDynamicInputShape(1, i);
        auto yShape = context->GetOutputShape(i);
        if ((xShape == nullptr) || (yShape == nullptr)) {
            return ge::GRAPH_FAILED;
        }
        *yShape = *xShape;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ForeachPowScalarAndTensor(gert::InferDataTypeContext* context)
{
    uint32_t outputNumber = context->GetComputeNodeOutputNum();
    const auto inputInfomation = context->GetIrInputInstanceInfo(1);
    if (inputInfomation == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::string errMsg = optiling::ConcatString("num of dynamic input1 ", inputInfomation->GetInstanceNum(),
                                                "not equal num of dynamic output0 ", outputNumber);
    OP_CHECK_IF(inputInfomation->GetInstanceNum() != outputNumber, ([&]() {
                    const char* e9NodeName = context->GetNodeName();
                    const std::string e9Op = (e9NodeName == nullptr ? "ForeachPowScalarAndTensor" : e9NodeName);
                    const std::string e9Parameter = "x";
                    const std::string e9Size = std::to_string(inputInfomation->GetInstanceNum());
                    const std::string e9Reason = (std::string(errMsg.c_str()));
                    OP_LOGE_WITHOUT_REPORT(e9Op.c_str(), "Parameter %s has invalid list size %s: %s.",
                                           e9Parameter.c_str(), e9Size.c_str(), e9Reason.c_str());
                    const std::vector<const char*> e9Keys = {"param_name", "op_name", "incorrect_size", "reason"};
                    const std::vector<const char*> e9Values = {e9Parameter.c_str(), e9Op.c_str(), e9Size.c_str(),
                                                               e9Reason.c_str()};
                    REPORT_PREDEFINED_ERR_MSG("EZ0038", e9Keys, e9Values);
                }()),
                return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputInfomation->GetInstanceNum(); i++) {
        auto xDtype = context->GetDynamicInputDataType(1, i);
        context->SetOutputDataType(i, xDtype);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ForeachPowScalarAndTensor)
    .InferShape(InferShape4ForeachPowScalarAndTensor)
    .InferDataType(InferDataType4ForeachPowScalarAndTensor);
} // namespace ops
