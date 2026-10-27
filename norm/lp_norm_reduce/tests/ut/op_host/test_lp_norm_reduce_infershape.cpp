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
#include <string>
#include <vector>

#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/lp_norm_reduce_proto.h"

namespace {
void CheckShape(const std::vector<int64_t>& input, int64_t p, const std::vector<int64_t>& axes, bool keepdim,
                ge::graphStatus expectedStatus, const std::vector<int64_t>& expectedShape = {})
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("LpNormReduce");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    auto xShape = CreateShape(input);
    gert::Shape yShape;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("LpNormReduce")
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"p", Ops::NN::AnyValue::CreateFrom<int64_t>(p)},
                                  {"axes", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(axes)},
                                  {"keepdim", Ops::NN::AnyValue::CreateFrom<bool>(keepdim)}})
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(impl->infer_shape(context), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(*context->GetOutputShape(0), CreateShape(expectedShape));
    }
}

TEST(LpNormReduceInferShape, PreservesUnknownRankWithDeferredAxisCheck)
{
    CheckShape({-2}, 2, {1}, false, ge::GRAPH_SUCCESS, {-2});
}

TEST(LpNormReduceInferShape, RejectsInvalidPBeforeUnknownRankReturn)
{
    CheckShape({-2}, -1, {1}, false, ge::GRAPH_FAILED);
}

TEST(LpNormReduceInferShape, NormalizesDuplicateNegativeAxes)
{
    CheckShape({2, 3, 4}, 2, {-1, -1}, true, ge::GRAPH_SUCCESS, {2, 3, 1});
}

TEST(LpNormReduceInferShape, RejectsAxisOutsideKnownRank) { CheckShape({2, 3}, 2, {2}, false, ge::GRAPH_FAILED); }

TEST(LpNormReduceInferShape, DistinguishesEmptyReductionFromScalar)
{
    CheckShape({1, 0, 1}, 2, {1}, false, ge::GRAPH_SUCCESS, {1, 1});
    CheckShape({}, 2, {}, false, ge::GRAPH_SUCCESS, {});
}

struct ShapeCase {
    const char* name;
    std::vector<int64_t> input;
    int64_t p;
    std::vector<int64_t> axes;
    bool keepdim;
    ge::graphStatus expectedStatus;
    std::vector<int64_t> expectedShape;
};

class LpNormReduceInferShapeCases : public testing::TestWithParam<ShapeCase> {};

TEST_P(LpNormReduceInferShapeCases, MatchesExpectedShape)
{
    const auto& param = GetParam();
    CheckShape(param.input, param.p, param.axes, param.keepdim, param.expectedStatus, param.expectedShape);
}

// Exercise the former standalone test scenarios through the registered callback.
// The scalar and positive out-of-range axis cases are already covered above.
const ShapeCase SHAPE_CASES[] = {
    {"ReduceAxis1", {2, 32}, 2, {1}, false, ge::GRAPH_SUCCESS, {2}},
    {"ReduceAxis1Keepdim", {2, 32}, 2, {1}, true, ge::GRAPH_SUCCESS, {2, 1}},
    {"ReduceAll", {3, 5}, 2, {}, false, ge::GRAPH_SUCCESS, {}},
    {"ReduceAllKeepdim", {3, 5}, 2, {}, true, ge::GRAPH_SUCCESS, {1, 1}},
    {"NegativeAxis", {2, 32}, 1, {-1}, false, ge::GRAPH_SUCCESS, {2}},
    {"DuplicateAxis", {3, 5}, 2, {0, 0}, true, ge::GRAPH_SUCCESS, {1, 5}},
    {"MiddleAxisKeepdim", {2, 3, 4}, 2, {1}, true, ge::GRAPH_SUCCESS, {2, 1, 4}},
    {"MiddleAxis", {2, 3, 4}, 2, {1}, false, ge::GRAPH_SUCCESS, {2, 4}},
    {"MultipleAxes", {2, 3, 4, 5}, 2, {1, 3}, false, ge::GRAPH_SUCCESS, {2, 4}},
    {"NegativeAxisOutOfRange", {2, 3}, 2, {-3}, false, ge::GRAPH_FAILED, {}},
    {"UnknownRank", {-2}, 2, {}, false, ge::GRAPH_SUCCESS, {-2}},
    {"ScalarWithAxis", {}, 2, {0}, false, ge::GRAPH_FAILED, {}},
    {"DynamicNonReductionDim", {-1, 16}, 2, {1}, false, ge::GRAPH_SUCCESS, {-1}},
    {"PositiveInfinity", {2, 32}, 2147483647, {1}, false, ge::GRAPH_SUCCESS, {2}},
    {"NegativeInfinityEmptyReduction", {2, 0}, -2147483648LL, {1}, false, ge::GRAPH_FAILED, {}},
    {"NegativeP", {2, 32}, -5, {1}, false, ge::GRAPH_FAILED, {}},
    {"RankAboveLimit", {1, 1, 1, 1, 1, 1, 1, 1, 1}, 2, {}, false, ge::GRAPH_FAILED, {}},
    {"RankAtLimit", {2, 2, 2, 2, 2, 2, 2, 2}, 2, {0}, false, ge::GRAPH_SUCCESS, {2, 2, 2, 2, 2, 2, 2}},
};

INSTANTIATE_TEST_SUITE_P(ShapeScenarios, LpNormReduceInferShapeCases, testing::ValuesIn(SHAPE_CASES),
                         [](const testing::TestParamInfo<ShapeCase>& info) { return std::string(info.param.name); });
} // namespace
