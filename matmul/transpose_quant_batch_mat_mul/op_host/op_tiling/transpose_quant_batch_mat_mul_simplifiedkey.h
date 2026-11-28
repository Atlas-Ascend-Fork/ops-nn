/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file transpose_quant_batch_mat_mul_simplifiedkey.h
 * \brief
 */

#ifndef __OP_HOST_TRANSPOSE_QUANT_BATCH_MATMUL_SIMPLIFIEDKEY_H__
#define __OP_HOST_TRANSPOSE_QUANT_BATCH_MATMUL_SIMPLIFIEDKEY_H__

#include "exe_graph/runtime/tiling_context.h"
#include "error_util.h"

namespace optiling {
namespace transpose_quant_batch_mat_mul_advanced {
inline ge::graphStatus GenSimplifiedKey(gert::TilingContext* context, ge::char_t* simplifiedKey)
{
    static const size_t maxLenSimplifiedKey = 256;
    static const int32_t input0Index = 0;
    static const int32_t input1Index = 1;
    static const int32_t biasIndex = 2;
    static const int32_t x1ScaleIndex = 3;
    static const int32_t x2ScaleIndex = 4;
    if (context == nullptr) {
        CUBE_INNER_ERR_REPORT("TransposeQuantBatchMatMul", "context is null in GenSimplifiedKey");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context->GetNodeName(), "Enter genSimplifiedKey.");
    OP_TILING_CHECK(simplifiedKey == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "simplifiedKey is null"),
                    return ge::GRAPH_FAILED);

    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(input0Index));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(input1Index));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));

    auto input0Format = context->GetInputDesc(input0Index)->GetStorageFormat();
    auto input1Format = context->GetInputDesc(input1Index)->GetStorageFormat();
    auto outputFormat = context->GetOutputDesc(0)->GetStorageFormat();
    auto input0DataType = context->GetInputDesc(input0Index)->GetDataType();
    auto input1DataType = context->GetInputDesc(input1Index)->GetDataType();
    auto outputDataType = context->GetOutputDesc(0)->GetDataType();
    auto x1ScaleDataType = ge::DT_FLOAT;
    auto x2ScaleDataType = ge::DT_FLOAT;
    auto biasDataType = ge::DT_FLOAT;
    bool isHIFP8 = input0DataType == ge::DT_HIFLOAT8 && input1DataType == ge::DT_HIFLOAT8;
    if (context->GetOptionalInputDesc(biasIndex) != nullptr) {
        biasDataType = context->GetOptionalInputDesc(biasIndex)->GetDataType();
    }
    if (context->GetOptionalInputDesc(x1ScaleIndex) != nullptr) {
        x1ScaleDataType = context->GetOptionalInputDesc(x1ScaleIndex)->GetDataType();
    } else if (isHIFP8) {
        x1ScaleDataType = ge::DT_UINT64;
    }
    if (context->GetOptionalInputDesc(x2ScaleIndex) != nullptr) {
        x2ScaleDataType = context->GetOptionalInputDesc(x2ScaleIndex)->GetDataType();
    }
    std::string simpleKeyTemp = "";
    errno_t err = strcat_s(simplifiedKey, maxLenSimplifiedKey, "diy,");
    if (err != 0) {
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "strcat_s prefix failed with error code %d.", err);
        return ge::GRAPH_FAILED;
    }
    simpleKeyTemp.append(std::to_string(input0Format))
        .append("/")
        .append(std::to_string(input1Format))
        .append("/")
        .append(std::to_string(ge::FORMAT_ND))
        .append("/")
        .append(std::to_string(ge::FORMAT_ND))
        .append("/")
        .append(std::to_string(ge::FORMAT_ND))
        .append("/")
        .append(std::to_string(outputFormat))
        .append("/")
        .append(std::to_string(input0DataType))
        .append("/")
        .append(std::to_string(input1DataType))
        .append("/")
        .append(std::to_string(biasDataType))
        .append("/")
        .append(std::to_string(x1ScaleDataType))
        .append("/")
        .append(std::to_string(x2ScaleDataType))
        .append("/")
        .append(std::to_string(outputDataType));
    err = strcat_s(simplifiedKey, maxLenSimplifiedKey, simpleKeyTemp.c_str());
    if (err != 0) {
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "strcat_s failed with error code %d.", err);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace transpose_quant_batch_mat_mul_advanced
} // namespace optiling
#endif // __OP_HOST_TRANSPOSE_QUANT_BATCH_MATMUL_SIMPLIFIEDKEY_H__
