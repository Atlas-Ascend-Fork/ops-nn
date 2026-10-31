/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>

#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForINInferV2(gert::InferDataTypeContext* context);
}

namespace {
void CheckDataTypes(const std::vector<uint32_t>& inputInstances, const std::vector<uint32_t>& outputInstances,
                    const std::vector<ge::DataType>& inputs, const std::vector<ge::DataType>& expected,
                    ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("INInferV2").OpName("in_infer_v2_dtype");
    // IOInstanceNum also sets the IR counts and is mutually exclusive with IONum.
    builder.IOInstanceNum(inputInstances, outputInstances);
    for (size_t i = 0; i < inputs.size(); ++i) {
        builder.InputTensorDesc(i, inputs[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    size_t inputStart = 0;
    for (size_t i = 0; i < inputInstances.size(); ++i) {
        const auto* instance = context->GetIrInputInstanceInfo(i);
        ASSERT_NE(instance, nullptr);
        ASSERT_EQ(instance->GetInstanceStart(), inputStart);
        ASSERT_EQ(instance->GetInstanceNum(), inputInstances[i]);
        inputStart += inputInstances[i];
    }
    ASSERT_EQ(inputStart, inputs.size());
    size_t outputStart = 0;
    for (size_t i = 0; i < outputInstances.size(); ++i) {
        const auto* instance = context->GetIrOutputInstanceInfo(i);
        ASSERT_NE(instance, nullptr);
        ASSERT_EQ(instance->GetInstanceStart(), outputStart);
        ASSERT_EQ(instance->GetInstanceNum(), outputInstances[i]);
        outputStart += outputInstances[i];
    }
    ASSERT_EQ(outputStart, expected.size());
    ASSERT_EQ(ops::InferDataTypeForINInferV2(context), status);
    if (status == ge::GRAPH_SUCCESS) {
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(context->GetOutputDataType(i), expected[i]);
        }
    }
}
} // namespace

TEST(INInferV2GraphInfer, CopiesThreeDeclaredDtypes)
{
    for (auto type : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        CheckDataTypes({1, 1, 1, 1, 1}, {1, 1, 1}, {type, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
                       {type, ge::DT_FLOAT, ge::DT_FLOAT});
    }
}

TEST(INInferV2GraphInfer, MapsOptionalInputsWithoutAffine)
{
    CheckDataTypes({1, 0, 0, 1, 1}, {1, 1, 1}, {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT},
                   {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT});
}

TEST(INInferV2GraphInfer, HandlesEveryOptionalOutputCombination)
{
    for (bool meanPresent : {false, true}) {
        for (bool variancePresent : {false, true}) {
            std::vector<uint32_t> instances = {1, meanPresent ? 1U : 0U, variancePresent ? 1U : 0U};
            while (instances.size() > 1 && instances.back() == 0) {
                instances.pop_back();
            }
            std::vector<ge::DataType> expected = {ge::DT_FLOAT16};
            if (meanPresent) {
                expected.push_back(ge::DT_FLOAT);
            }
            if (variancePresent) {
                expected.push_back(ge::DT_FLOAT);
            }
            CheckDataTypes({1, 0, 0, 1, 1}, instances, {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT}, expected);
        }
    }
}

TEST(INInferV2GraphInfer, MapsVarianceWhenMeanOutputIsAbsent)
{
    // Distinct synthetic dtypes expose wrong IR-to-instance mapping. This does
    // not expand the public FP32-only statistic-input contract.
    CheckDataTypes({1, 0, 0, 1, 1}, {1, 0, 1}, {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT},
                   {ge::DT_FLOAT, ge::DT_FLOAT});
}

TEST(INInferV2GraphInfer, RejectsMissingRequiredOutput)
{
    CheckDataTypes({1, 0, 0, 1, 1}, {0, 1, 1}, {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
                   {ge::DT_UNDEFINED, ge::DT_UNDEFINED}, ge::GRAPH_FAILED);
}

TEST(INInferV2GraphInfer, RejectsMultipleStatisticOutputInstances)
{
    CheckDataTypes({1, 0, 0, 1, 1}, {1, 2, 0}, {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
                   {ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_UNDEFINED}, ge::GRAPH_FAILED);
}

TEST(INInferV2GraphInfer, RejectsNullContext) { EXPECT_EQ(ops::InferDataTypeForINInferV2(nullptr), ge::GRAPH_FAILED); }
