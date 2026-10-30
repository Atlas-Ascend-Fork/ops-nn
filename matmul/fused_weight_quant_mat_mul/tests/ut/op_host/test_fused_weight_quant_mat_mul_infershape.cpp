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

#include "log/log.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace {
class TestFusedWeightQuantMatMulInferShape : public testing::Test {};

// 常规 2D：输出 shape 与底座 WQBMMV2 一致（relu 不改 shape/dtype）
TEST_F(TestFusedWeightQuantMatMulInferShape, InferShapeBasic)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedWeightQuantMatMul")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{96, 11264}, {96, 11264}};
    gert::StorageShape weightShape = {{11264, 1664}, {11264, 1664}};
    gert::StorageShape antiQuantScale = {{1664}, {1664}};
    gert::StorageShape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &weightShape, &antiQuantScale, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&yShape})
                      .NodeAttrs({{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"antiquant_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"fused_op_type", Ops::NN::AnyValue::CreateFrom<std::string>("relu")}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[96, 1664]");
}

// transpose_weight=true：y 的 n 取 weight 倒数第二维
TEST_F(TestFusedWeightQuantMatMulInferShape, InferShapeTransposeWeight)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedWeightQuantMatMul")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{1, 96, 1408}, {1, 96, 1408}};
    gert::StorageShape weightShape = {{1664, 1408}, {1664, 1408}};
    gert::StorageShape antiQuantScale = {{1664}, {1664}};
    gert::StorageShape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &weightShape, &antiQuantScale, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&yShape})
                      .NodeAttrs({{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"antiquant_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"fused_op_type", Ops::NN::AnyValue::CreateFrom<std::string>("relu")}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, 96, 1664]");
}

// 未知维度数：输出 [-2]
TEST_F(TestFusedWeightQuantMatMulInferShape, UnknownDimNum)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedWeightQuantMatMul")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {-2};
    gert::Shape weightShape = {-2};
    gert::Shape antiQuantScale = {-2};
    gert::Shape expectYShape = {-2};
    gert::Shape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &weightShape, &antiQuantScale, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&yShape})
                      .NodeAttrs({{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"antiquant_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"fused_op_type", Ops::NN::AnyValue::CreateFrom<std::string>("relu")}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectYShape));
}

// K 不匹配：失败
TEST_F(TestFusedWeightQuantMatMulInferShape, KDimMismatch)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedWeightQuantMatMul")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{64, 128}, {64, 128}};
    gert::StorageShape weightShape = {{256, 64}, {256, 64}};
    gert::StorageShape antiQuantScale = {{64}, {64}};
    gert::StorageShape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &weightShape, &antiQuantScale, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&yShape})
                      .NodeAttrs({{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"antiquant_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"fused_op_type", Ops::NN::AnyValue::CreateFrom<std::string>("relu")}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 非法 fused_op_type：拒绝
TEST_F(TestFusedWeightQuantMatMulInferShape, InvalidFusedOpType)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedWeightQuantMatMul")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{64, 128}, {64, 128}};
    gert::StorageShape weightShape = {{128, 64}, {128, 64}};
    gert::StorageShape antiQuantScale = {{64}, {64}};
    gert::StorageShape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &weightShape, &antiQuantScale, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&yShape})
                      .NodeAttrs({{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"antiquant_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"fused_op_type", Ops::NN::AnyValue::CreateFrom<std::string>("gelu")}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TestFusedWeightQuantMatMulInferShape, RejectsSevenDimensionalInput)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedWeightQuantMatMul")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{1, 1, 1, 1, 1, 64, 128}, {1, 1, 1, 1, 1, 64, 128}};
    gert::StorageShape weightShape = {{128, 64}, {128, 64}};
    gert::StorageShape antiQuantScale = {{64}, {64}};
    gert::StorageShape yShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &weightShape, &antiQuantScale, nullptr, nullptr, nullptr, nullptr})
                      .OutputShapes({&yShape})
                      .NodeAttrs({{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"antiquant_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"fused_op_type", Ops::NN::AnyValue::CreateFrom<std::string>("relu")}})
                      .Build();
    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

static ge::graphStatus RunInferDataType(int64_t dtype, ge::DataType& outputDtype)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedWeightQuantMatMul")->infer_datatype;
    EXPECT_NE(inferDataTypeFunc, nullptr);
    if (inferDataTypeFunc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(7, 1)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"antiquant_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)}})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    auto status = inferDataTypeFunc(context);
    outputDtype = context->GetOutputDataType(0);
    return status;
}

TEST_F(TestFusedWeightQuantMatMulInferShape, InferDataTypeDefaultsToInput)
{
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(-1, outputDtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputDtype, ge::DT_FLOAT16);
}

TEST_F(TestFusedWeightQuantMatMulInferShape, InferDataTypeSupportsInt8)
{
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(static_cast<int64_t>(ge::DT_INT8), outputDtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputDtype, ge::DT_INT8);
}

TEST_F(TestFusedWeightQuantMatMulInferShape, InferDataTypeRejectsInvalidType)
{
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    EXPECT_EQ(RunInferDataType(static_cast<int64_t>(ge::DT_BF16), outputDtype), ge::GRAPH_FAILED);
}
} // namespace
