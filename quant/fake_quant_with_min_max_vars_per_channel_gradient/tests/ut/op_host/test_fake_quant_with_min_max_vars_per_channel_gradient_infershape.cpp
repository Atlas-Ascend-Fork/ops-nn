/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_fake_quant_with_min_max_vars_per_channel_gradient_infershape.cpp
 * \brief FakeQuantWithMinMaxVarsPerChannelGradient InferShape 单元测试
 *
 * 第一段（本分支用例）——形状透传规则：
 *   backprops_wrt_x.shape   = x.shape（即 gradients.shape）；
 *   backprops_wrt_min.shape = min.shape；
 *   backprops_wrt_max.shape = max.shape（-1 动态维 / -2 unknown rank 原样透传）。
 *
 * 第二段（吸收自 PR #12549）——文档化契约的通过/拒绝路径与 InferDataType：
 *   ok: rank1/2/4；reject: gradients!=x、min/max 长度!=C、rank>8；输出 dtype 全 float。
 */

#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "infershape_case_executor.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"

#include "../../../op_graph/fake_quant_with_min_max_vars_per_channel_gradient_proto.h"

class FakeQuantWithMinMaxVarsPerChannelGradientInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FakeQuantWithMinMaxVarsPerChannelGradientInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FakeQuantWithMinMaxVarsPerChannelGradientInfershape TearDown" << std::endl;
    }
};

// 2D：bpx 跟随 x（[b, d]），bpmin/bpmax 跟随 min/max（[d]）
TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientInfershape, infershape_2d)
{
    gert::InfershapeContextPara para("FakeQuantWithMinMaxVarsPerChannelGradient",
                                     {
                                         {{{32, 128}, {32, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{32, 128}, {32, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 128}, {128}, {128}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 4D：bpx 跟随 x（[b, h, w, d]）
TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientInfershape, infershape_4d)
{
    gert::InfershapeContextPara para("FakeQuantWithMinMaxVarsPerChannelGradient",
                                     {
                                         {{{2, 3, 4, 16}, {2, 3, 4, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2, 3, 4, 16}, {2, 3, 4, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{16}, {16}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4, 16}, {16}, {16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态维：-1 原样透传
TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientInfershape, infershape_dynamic)
{
    gert::InfershapeContextPara para("FakeQuantWithMinMaxVarsPerChannelGradient",
                                     {
                                         {{{1, -1, -1, 64}, {1, -1, -1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1, -1, -1, 64}, {1, -1, -1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, -1, -1, 64}, {64}, {64}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态维：x 末维未知（-1）时同样透传（E89999 类场景的 desc 形态）
TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientInfershape, infershape_dynamic_last_dim_unknown)
{
    gert::InfershapeContextPara para("FakeQuantWithMinMaxVarsPerChannelGradient",
                                     {
                                         {{{3, -1, -1}, {3, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{3, -1, -1}, {3, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, -1, -1}, {-1}, {-1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// unknown rank：-2 原样透传
TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientInfershape, infershape_unknown_rank)
{
    gert::InfershapeContextPara para("FakeQuantWithMinMaxVarsPerChannelGradient",
                                     {
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}, {-2}, {-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ==================== 以下用例吸收自 PR #12549 ====================

class FakeQuantWithMinMaxVarsPerChannelGradientInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FakeQuantWithMinMaxVarsPerChannelGradient infershape SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "FakeQuantWithMinMaxVarsPerChannelGradient infershape TearDown" << std::endl;
    }
};

TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientInferShapeTest, infer_shape_ok_rank2)
{
    ge::op::FakeQuantWithMinMaxVarsPerChannelGradient op;
    op.UpdateInputDesc("gradients", create_desc({8, 16}, ge::DT_FLOAT));
    op.UpdateInputDesc("x", create_desc({8, 16}, ge::DT_FLOAT));
    op.UpdateInputDesc("min", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("max", create_desc({16}, ge::DT_FLOAT));

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    EXPECT_EQ(op.GetOutputDescByName("backprops_wrt_x").GetShape().GetDims(), std::vector<int64_t>({8, 16}));
    EXPECT_EQ(op.GetOutputDescByName("backprops_wrt_min").GetShape().GetDims(), std::vector<int64_t>({16}));
    EXPECT_EQ(op.GetOutputDescByName("backprops_wrt_max").GetShape().GetDims(), std::vector<int64_t>({16}));
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientInferShapeTest, infer_shape_ok_rank4)
{
    ge::op::FakeQuantWithMinMaxVarsPerChannelGradient op;
    op.UpdateInputDesc("gradients", create_desc({2, 3, 4, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("x", create_desc({2, 3, 4, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("min", create_desc({5}, ge::DT_FLOAT));
    op.UpdateInputDesc("max", create_desc({5}, ge::DT_FLOAT));

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDescByName("backprops_wrt_x").GetShape().GetDims(), std::vector<int64_t>({2, 3, 4, 5}));
    EXPECT_EQ(op.GetOutputDescByName("backprops_wrt_min").GetShape().GetDims(), std::vector<int64_t>({5}));
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientInferShapeTest, infer_shape_ok_rank1_min_c)
{
    ge::op::FakeQuantWithMinMaxVarsPerChannelGradient op;
    op.UpdateInputDesc("gradients", create_desc({1}, ge::DT_FLOAT));
    op.UpdateInputDesc("x", create_desc({1}, ge::DT_FLOAT));
    op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT));
    op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT));

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDescByName("backprops_wrt_x").GetShape().GetDims(), std::vector<int64_t>({1}));
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientInferShapeTest, infer_shape_reject_gradients_shape_mismatch)
{
    ge::op::FakeQuantWithMinMaxVarsPerChannelGradient op;
    op.UpdateInputDesc("gradients", create_desc({8, 15}, ge::DT_FLOAT)); // != x
    op.UpdateInputDesc("x", create_desc({8, 16}, ge::DT_FLOAT));
    op.UpdateInputDesc("min", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("max", create_desc({16}, ge::DT_FLOAT));

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientInferShapeTest, infer_shape_reject_min_length_mismatch)
{
    ge::op::FakeQuantWithMinMaxVarsPerChannelGradient op;
    op.UpdateInputDesc("gradients", create_desc({8, 16}, ge::DT_FLOAT));
    op.UpdateInputDesc("x", create_desc({8, 16}, ge::DT_FLOAT));
    op.UpdateInputDesc("min", create_desc({8}, ge::DT_FLOAT)); // != C(16)
    op.UpdateInputDesc("max", create_desc({16}, ge::DT_FLOAT));

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientInferShapeTest, infer_shape_reject_rank_over_8)
{
    ge::op::FakeQuantWithMinMaxVarsPerChannelGradient op;
    op.UpdateInputDesc("gradients", create_desc({1, 1, 1, 1, 1, 1, 1, 1, 4}, ge::DT_FLOAT));
    op.UpdateInputDesc("x", create_desc({1, 1, 1, 1, 1, 1, 1, 1, 4}, ge::DT_FLOAT));
    op.UpdateInputDesc("min", create_desc({4}, ge::DT_FLOAT));
    op.UpdateInputDesc("max", create_desc({4}, ge::DT_FLOAT));

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientInferShapeTest, infer_dtype_float)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxVarsPerChannelGradient"), nullptr);
    auto infer_dtype_func = gert::OpImplRegistry::GetInstance()
                                .GetOpImpl("FakeQuantWithMinMaxVarsPerChannelGradient")
                                ->infer_datatype;
    ASSERT_NE(infer_dtype_func, nullptr);

    ge::DataType g_ref = ge::DT_FLOAT;
    ge::DataType x_ref = ge::DT_FLOAT;
    ge::DataType mn_ref = ge::DT_FLOAT;
    ge::DataType mx_ref = ge::DT_FLOAT;
    ge::DataType bpx_ref = ge::DT_FLOAT;
    ge::DataType bpmn_ref = ge::DT_FLOAT;
    ge::DataType bpmx_ref = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(4, 3)
                              .IrInstanceNum({1, 1, 1, 1})
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&g_ref, &x_ref, &mn_ref, &mx_ref})
                              .OutputDataTypes({&bpx_ref, &bpmn_ref, &bpmx_ref})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(infer_dtype_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
}
