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
 * \file test_global_lp_pool_infershape.cpp
 * \brief GlobalLpPool InferShape UT: 4D/5D, dynamic dim (-1), unknown rank (-2) and invalid rank
 */
#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ge;

namespace {
const char* OP_TYPE = "GlobalLpPool";

std::vector<std::pair<std::string, Ops::NN::AnyValue>> MakeAttrs(float p)
{
    return {{"p", Ops::NN::AnyValue::CreateFrom<float>(p)}};
}

std::vector<int64_t> DimsOf(const gert::Shape* shape)
{
    std::vector<int64_t> dims;
    if (shape == nullptr) {
        return dims;
    }
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        dims.push_back(shape->GetDim(i));
    }
    return dims;
}
} // namespace

class GlobalLpPoolInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(GlobalLpPoolInferShapeTest, infershape_4d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x = {2, 4, 8, 8};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x})
                      .NodeAttrs(MakeAttrs(2.0f))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DimsOf(context->GetOutputShape(0)), (std::vector<int64_t>{2, 4, 1, 1}));
}

TEST_F(GlobalLpPoolInferShapeTest, infershape_5d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x = {1, 8, 2, 2, 3};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x})
                      .NodeAttrs(MakeAttrs(2.0f))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DimsOf(context->GetOutputShape(0)), (std::vector<int64_t>{1, 8, 1, 1, 1}));
}

TEST_F(GlobalLpPoolInferShapeTest, infershape_dynamic_dim)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x = {-1, -1, 7, 7};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x})
                      .NodeAttrs(MakeAttrs(2.0f))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DimsOf(context->GetOutputShape(0)), (std::vector<int64_t>{-1, -1, 1, 1}));
}

TEST_F(GlobalLpPoolInferShapeTest, infershape_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x = {-2};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x})
                      .NodeAttrs(MakeAttrs(2.0f))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DimsOf(context->GetOutputShape(0)), (std::vector<int64_t>{-2}));
}

TEST_F(GlobalLpPoolInferShapeTest, infershape_invalid_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x = {1, 2, 3};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x})
                      .NodeAttrs(MakeAttrs(2.0f))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

TEST_F(GlobalLpPoolInferShapeTest, infershape_pseudo_unknown_rank_rejected)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    // {-2, 3} has dimNum 2 — not the canonical unknown-rank desc {-2}; must
    // fall through to the rank check instead of the passthrough branch.
    gert::Shape x = {-2, 3};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x})
                      .NodeAttrs(MakeAttrs(2.0f))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

TEST_F(GlobalLpPoolInferShapeTest, infershape_6d_rejected)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE)->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x = {1, 2, 3, 4, 5, 6};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x})
                      .NodeAttrs(MakeAttrs(2.0f))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}
