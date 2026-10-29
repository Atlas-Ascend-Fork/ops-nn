/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "array_ops.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/gather_v2_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/expand_dims_type.h"
#include "exe_graph/runtime/tensor.h"
#include "log/log.h"
#include "platform/platform_info.h"

class GatherV2InfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GatherV2InfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GatherV2InfershapeTest TearDown" << std::endl; }
};

TEST_F(GatherV2InfershapeTest, gather_v2_infershape_test_1)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherV2")->infer_shape;

    gert::Shape input_x_shape = {6, 7, 2};
    gert::Shape input_indices_shape = {9, 10, 2};
    gert::Shape input_axes_shape = {1};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_x_shape, &input_indices_shape, &input_axes_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"attr_batch_dims", Ops::NN::AnyValue::CreateFrom<int64_t>(10)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 常规场景：axis=0，输出为 indices.shape + x.shape[1:]
TEST_F(GatherV2InfershapeTest, gather_v2_infershape_test_2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherV2")->infer_shape;

    gert::Shape input_x_shape = {4, 3};
    gert::Shape input_indices_shape = {2};
    gert::Shape input_axes_shape = {1};
    gert::Shape output_shape = {};
    int32_t axisData = 0;

    gert::StorageShape xStorageShape = {{4, 3}, {4, 3}};
    gert::StorageShape indicesStorageShape = {{2}, {2}};
    gert::StorageShape axesStorageShape = {{1}, {1}};
    gert::Tensor xTensor(xStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor indicesTensor(indicesStorageShape,
                               gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                               gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Tensor axesTensor(axesStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                            gert::TensorPlacement::kOnHost, ge::DT_INT32, &axisData);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_x_shape, &input_indices_shape, &input_axes_shape})
                      .OutputShapes({&output_shape})
                      .InputTensors({&xTensor, &indicesTensor, &axesTensor})
                      .NodeAttrs({{"attr_batch_dims", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedShape = {2, 3};
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedShape));
}

// 仅axis输入为未知rank（-2）场景：输出rank可推导（x_rank+indices_rank-1），各维置未知-1
TEST_F(GatherV2InfershapeTest, gather_v2_infershape_axis_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherV2")->infer_shape;

    gert::Shape input_x_shape = {4, 3};
    gert::Shape input_indices_shape = {2};
    gert::Shape input_axes_shape = {-2};
    gert::Shape output_shape = {};

    gert::StorageShape xStorageShape = {{4, 3}, {4, 3}};
    gert::StorageShape indicesStorageShape = {{2}, {2}};
    gert::StorageShape axesStorageShape = {{-2}, {-2}};
    gert::Tensor xTensor(xStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor indicesTensor(indicesStorageShape,
                               gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                               gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Tensor axesTensor(axesStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                            gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_x_shape, &input_indices_shape, &input_axes_shape})
                      .OutputShapes({&output_shape})
                      .InputTensors({&xTensor, &indicesTensor, &axesTensor})
                      .NodeAttrs({{"attr_batch_dims", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedShape = {-1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedShape));
}

// axis数据未就绪场景：输出rank可推导（x_rank+indices_rank-1），各维置未知-1
TEST_F(GatherV2InfershapeTest, gather_v2_infershape_axis_data_not_ready)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherV2")->infer_shape;

    gert::Shape input_x_shape = {4, 3};
    gert::Shape input_indices_shape = {2};
    gert::Shape input_axes_shape = {1};
    gert::Shape output_shape = {};

    gert::StorageShape xStorageShape = {{4, 3}, {4, 3}};
    gert::StorageShape indicesStorageShape = {{2}, {2}};
    gert::StorageShape axesStorageShape = {{1}, {1}};
    gert::Tensor xTensor(xStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor indicesTensor(indicesStorageShape,
                               gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                               gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Tensor axesTensor(axesStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                            gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_x_shape, &input_indices_shape, &input_axes_shape})
                      .OutputShapes({&output_shape})
                      .InputTensors({&xTensor, &indicesTensor, &axesTensor})
                      .NodeAttrs({{"attr_batch_dims", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedShape = {-1, -1};
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedShape));
}

// 常规场景：axis=1，输出为 x.shape[:1] + indices.shape
TEST_F(GatherV2InfershapeTest, gather_v2_infershape_test_3)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherV2")->infer_shape;

    gert::Shape input_x_shape = {4, 3};
    gert::Shape input_indices_shape = {2};
    gert::Shape input_axes_shape = {1};
    gert::Shape output_shape = {};
    int32_t axisData = 1;

    gert::StorageShape xStorageShape = {{4, 3}, {4, 3}};
    gert::StorageShape indicesStorageShape = {{2}, {2}};
    gert::StorageShape axesStorageShape = {{1}, {1}};
    gert::Tensor xTensor(xStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor indicesTensor(indicesStorageShape,
                               gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                               gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Tensor axesTensor(axesStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                            gert::TensorPlacement::kOnHost, ge::DT_INT32, &axisData);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_x_shape, &input_indices_shape, &input_axes_shape})
                      .OutputShapes({&output_shape})
                      .InputTensors({&xTensor, &indicesTensor, &axesTensor})
                      .NodeAttrs({{"attr_batch_dims", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedShape = {4, 2};
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedShape));
}

// x为未知rank（-2）场景：输出置为未知rank，不依赖axis数据
TEST_F(GatherV2InfershapeTest, gather_v2_infershape_x_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherV2")->infer_shape;

    gert::Shape input_x_shape = {-2};
    gert::Shape input_indices_shape = {2};
    gert::Shape input_axes_shape = {1};
    gert::Shape output_shape = {};
    int32_t axisData = 0;

    gert::StorageShape xStorageShape = {{-2}, {}};
    gert::StorageShape indicesStorageShape = {{2}, {2}};
    gert::StorageShape axesStorageShape = {{1}, {1}};
    gert::Tensor xTensor(xStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor indicesTensor(indicesStorageShape,
                               gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                               gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Tensor axesTensor(axesStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                            gert::TensorPlacement::kOnHost, ge::DT_INT32, &axisData);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_x_shape, &input_indices_shape, &input_axes_shape})
                      .OutputShapes({&output_shape})
                      .InputTensors({&xTensor, &indicesTensor, &axesTensor})
                      .NodeAttrs({{"attr_batch_dims", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedShape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedShape));
}

// indices为未知rank（-2）场景：输出置为未知rank
TEST_F(GatherV2InfershapeTest, gather_v2_infershape_indices_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherV2")->infer_shape;

    gert::Shape input_x_shape = {4, 3};
    gert::Shape input_indices_shape = {-2};
    gert::Shape input_axes_shape = {1};
    gert::Shape output_shape = {};
    int32_t axisData = 0;

    gert::StorageShape xStorageShape = {{4, 3}, {4, 3}};
    gert::StorageShape indicesStorageShape = {{-2}, {}};
    gert::StorageShape axesStorageShape = {{1}, {1}};
    gert::Tensor xTensor(xStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                         gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor indicesTensor(indicesStorageShape,
                               gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                               gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Tensor axesTensor(axesStorageShape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                            gert::TensorPlacement::kOnHost, ge::DT_INT32, &axisData);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_x_shape, &input_indices_shape, &input_axes_shape})
                      .OutputShapes({&output_shape})
                      .InputTensors({&xTensor, &indicesTensor, &axesTensor})
                      .NodeAttrs({{"attr_batch_dims", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedShape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedShape));
}
