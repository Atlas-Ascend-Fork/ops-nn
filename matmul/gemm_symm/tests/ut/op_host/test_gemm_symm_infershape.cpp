/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "log/log.h"

namespace {

std::vector<std::pair<std::string, Ops::NN::AnyValue>> Attrs(bool transA, bool transB)
{
    return {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0F)},
            {"beta", Ops::NN::AnyValue::CreateFrom<float>(1.0F)},
            {"transpose_a", Ops::NN::AnyValue::CreateFrom<bool>(transA)},
            {"transpose_b", Ops::NN::AnyValue::CreateFrom<bool>(transB)},
            {"symm_type", Ops::NN::AnyValue::CreateFrom<std::string>("full")},
            {"op_impl_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0x1)}};
}

class GemmSymmInferShapeTest : public testing::Test {};

TEST_F(GemmSymmInferShapeTest, square_2d)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmSymm")->infer_shape;
    gert::StorageShape aShape = {{32, 16}, {32, 16}};
    gert::StorageShape bShape = {{16, 32}, {16, 32}};
    gert::StorageShape cShape = {{32, 32}, {32, 32}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&aShape, &bShape, &cShape})
                      .OutputShapes({&yShape})
                      .NodeAttrs(Attrs(false, false))
                      .Build();
    EXPECT_EQ(inferShape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[32, 32]");
}

TEST_F(GemmSymmInferShapeTest, transpose_and_batch)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmSymm")->infer_shape;
    gert::StorageShape aShape = {{2, 16, 32}, {2, 16, 32}};
    gert::StorageShape bShape = {{2, 32, 16}, {2, 32, 16}};
    gert::StorageShape cShape = {{2, 32, 32}, {2, 32, 32}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&aShape, &bShape, &cShape})
                      .OutputShapes({&yShape})
                      .NodeAttrs(Attrs(true, true))
                      .Build();
    EXPECT_EQ(inferShape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[2, 32, 32]");
}

TEST_F(GemmSymmInferShapeTest, reject_mismatched_b)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmSymm")->infer_shape;
    gert::StorageShape aShape = {{32, 16}, {32, 16}};
    gert::StorageShape bShape = {{16, 31}, {16, 31}};
    gert::StorageShape cShape = {{32, 32}, {32, 32}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&aShape, &bShape, &cShape})
                      .OutputShapes({&yShape})
                      .NodeAttrs(Attrs(false, false))
                      .Build();
    EXPECT_EQ(inferShape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(GemmSymmInferShapeTest, dtype_follows_a)
{
    auto inferDtype = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmSymm")->infer_datatype;
    ASSERT_NE(inferDtype, nullptr);
    ge::DataType aDtype = ge::DT_BF16;
    ge::DataType bDtype = ge::DT_BF16;
    ge::DataType yDtype = ge::DT_FLOAT;
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(3)
                      .NodeIoNum(3, 1)
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&aDtype, &bDtype, nullptr})
                      .OutputDataTypes({&yDtype})
                      .NodeAttrs(Attrs(false, false))
                      .Build();
    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDtype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

TEST_F(GemmSymmInferShapeTest, dtype_follows_a_fp16)
{
    auto inferDtype = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmSymm")->infer_datatype;
    ASSERT_NE(inferDtype, nullptr);
    ge::DataType aDtype = ge::DT_FLOAT16;
    ge::DataType bDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_BF16;
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(3)
                      .NodeIoNum(3, 1)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&aDtype, &bDtype, nullptr})
                      .OutputDataTypes({&yDtype})
                      .NodeAttrs(Attrs(false, false))
                      .Build();
    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDtype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

} // namespace
