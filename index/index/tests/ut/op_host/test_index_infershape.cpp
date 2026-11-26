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
 * \file test_index_infershape.cpp
 * \brief
 */

#include "ut_op_util.h"
#include "infershape_test_util.h"
#include <iostream>
#include <map>
#include <string>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "base/err_mgr.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/index_proto.h"
#include "infershape_case_executor.h"

class IndexTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Index SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Index TearDown" << std::endl; }
};

TEST_F(IndexTest, index_infershape_success_01)
{
    gert::StorageShape x_shape = {{1000}, {1000}};
    gert::StorageShape indexed_sizes_shape = {{1}, {1}};
    gert::StorageShape indexed_strides_shape = {{1}, {1}};
    gert::StorageShape indices_shape = {{1}, {1}};
    gert::Shape y_shape = {{1}};
    gert::Shape expected_output_shape = {1};

    ge::DataType dtype = ge::DT_FLOAT;

    vector<int64_t> indexed_sizes_data{1};
    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(indexed_sizes_shape.GetStorageShape().GetDimNum(), ge::DT_INT64,
                                                       total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(indexed_sizes_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(indexed_sizes_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), indexed_sizes_data.data(),
                   indexed_sizes_data.size() * sizeof(int64_t));

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Index")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x_shape, tensor, &indexed_strides_shape, &indices_shape})
                      .OutputShapes({&y_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expected_output_shape));
}

TEST_F(IndexTest, index_infershape_success_02)
{
    vector<int64_t> indexed_sizes_data{1, 0};
    gert::InfershapeContextPara::TensorDescription x_shape({{24572, 7500}, {24572, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_sizes_shape({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                       indexed_sizes_data.data());
    gert::InfershapeContextPara::TensorDescription indexed_strides_shape({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_01({{24572}, {24572}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_02({{24572}, {24572}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Index", {x_shape, indexed_sizes_shape, indexed_strides_shape, indices_shape_01, indices_shape_02}, {y},
        {1, 1, 1, 2}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {24572, 7500},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IndexTest, index_infershape_success_03)
{
    vector<int64_t> indexed_sizes_data{0, 1};
    gert::InfershapeContextPara::TensorDescription x_shape({{26, 51866}, {26, 51866}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_sizes_shape({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                       indexed_sizes_data.data());
    gert::InfershapeContextPara::TensorDescription indexed_strides_shape({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_01({{51866}, {51866}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_02({{51866}, {51866}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Index", {x_shape, indexed_sizes_shape, indexed_strides_shape, indices_shape_01, indices_shape_02}, {y},
        {1, 1, 1, 2}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {26, 51866},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IndexTest, index_infershape_compile_time_sizes_no_data)
{
    // 编译期 indexed_sizes 无 const 数据（GetData 为 nullptr），
    // 应返回成功并将输出置为 unknown rank，等待执行期数据就绪后重新推导
    gert::InfershapeContextPara::TensorDescription x_shape({{24572, 7500}, {24572, 7500}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_sizes_shape({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_strides_shape({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_01({{24572}, {24572}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Index", {x_shape, indexed_sizes_shape, indexed_strides_shape, indices_shape_01}, {y}, {1, 1, 1, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IndexTest, index_infershape_no_indexed_dim)
{
    // indexed_sizes 全 0（无索引维），输出与 x 相同
    vector<int64_t> indexed_sizes_data{0, 0};
    gert::InfershapeContextPara::TensorDescription x_shape({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_sizes_shape({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                       indexed_sizes_data.data());
    gert::InfershapeContextPara::TensorDescription indexed_strides_shape({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("Index", {x_shape, indexed_sizes_shape, indexed_strides_shape},
                                                      {y}, {1, 1, 1, 0}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {10, 20},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IndexTest, index_infershape_dynamic_shape_01)
{
    vector<int64_t> indexed_sizes_data{1, 0, 0};
    gert::InfershapeContextPara::TensorDescription x_shape({{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_sizes_shape({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                       indexed_sizes_data.data());
    gert::InfershapeContextPara::TensorDescription indexed_strides_shape({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_01({{-1, -1}, {-1, -1}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Index", {x_shape, indexed_sizes_shape, indexed_strides_shape, indices_shape_01}, {y}, {1, 1, 1, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IndexTest, index_infershape_dynamic_shape_02)
{
    vector<int64_t> indexed_sizes_data{0};
    gert::InfershapeContextPara::TensorDescription x_shape({{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_sizes_shape({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                       indexed_sizes_data.data());
    gert::InfershapeContextPara::TensorDescription indexed_strides_shape({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_01({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Index", {x_shape, indexed_sizes_shape, indexed_strides_shape, indices_shape_01}, {y}, {1, 1, 1, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IndexTest, index_infershape_dynamic_shape_03)
{
    vector<int64_t> indexed_sizes_data{1, 0, 0, 0};
    gert::InfershapeContextPara::TensorDescription x_shape({{8, 2, 4, 16}, {8, 2, 4, 16}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_sizes_shape({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                       indexed_sizes_data.data());
    gert::InfershapeContextPara::TensorDescription indexed_strides_shape({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_01({{-2}, {-2}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Index", {x_shape, indexed_sizes_shape, indexed_strides_shape, indices_shape_01}, {y}, {1, 1, 1, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IndexTest, index_infershape_dynamic_shape_04)
{
    vector<int64_t> indexed_sizes_data{1, 0, 0, 0};
    gert::InfershapeContextPara::TensorDescription x_shape({{-1, 2, 4, -1}, {-1, 2, 4, -1}}, ge::DT_FLOAT,
                                                           ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_sizes_shape({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                       indexed_sizes_data.data());
    gert::InfershapeContextPara::TensorDescription indexed_strides_shape({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_01({{-1, 2, 4, -1}, {-1, 2, 4, -1}}, ge::DT_INT64,
                                                                    ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Index", {x_shape, indexed_sizes_shape, indexed_strides_shape, indices_shape_01}, {y}, {1, 1, 1, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 2, 4, -1, 2, 4, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IndexTest, index_infershape_dynamic_shape_05)
{
    vector<int64_t> indexed_sizes_data{1, 0, 0, 0};
    gert::InfershapeContextPara::TensorDescription x_shape({{-1, 2, 4, -1}, {-1, 2, 4, -1}}, ge::DT_FLOAT,
                                                           ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_sizes_shape({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                       indexed_sizes_data.data());
    gert::InfershapeContextPara::TensorDescription indexed_strides_shape({{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_01({{-2}, {-2}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Index", {x_shape, indexed_sizes_shape, indexed_strides_shape, indices_shape_01}, {y}, {1, 1, 1, 1}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IndexTest, index_infershape_dynamic_shape_06)
{
    vector<int64_t> indexed_sizes_data{1, 1, 0};
    gert::InfershapeContextPara::TensorDescription x_shape({{10, 20, 30}, {10, 20, 30}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_sizes_shape({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                       indexed_sizes_data.data());
    gert::InfershapeContextPara::TensorDescription indexed_strides_shape({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_01({{-1, 5}, {-1, 5}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_02({{3, -1}, {3, -1}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Index", {x_shape, indexed_sizes_shape, indexed_strides_shape, indices_shape_01, indices_shape_02}, {y},
        {1, 1, 1, 2}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 5, 30},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IndexTest, index_infershape_broadcast_01)
{
    vector<int64_t> indexed_sizes_data{1, 1, 0};
    gert::InfershapeContextPara::TensorDescription x_shape({{10, 20, 30}, {10, 20, 30}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_sizes_shape({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                       indexed_sizes_data.data());
    gert::InfershapeContextPara::TensorDescription indexed_strides_shape({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_01({{3, 5}, {3, 5}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_02({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Index", {x_shape, indexed_sizes_shape, indexed_strides_shape, indices_shape_01, indices_shape_02}, {y},
        {1, 1, 1, 2}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 5, 30},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IndexTest, index_infershape_broadcast_02)
{
    vector<int64_t> indexed_sizes_data{1, 1, 0};
    gert::InfershapeContextPara::TensorDescription x_shape({{10, 20, 30}, {10, 20, 30}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_sizes_shape({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                       indexed_sizes_data.data());
    gert::InfershapeContextPara::TensorDescription indexed_strides_shape({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_01({{3, 1}, {3, 1}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_02({{1, 5}, {1, 5}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Index", {x_shape, indexed_sizes_shape, indexed_strides_shape, indices_shape_01, indices_shape_02}, {y},
        {1, 1, 1, 2}, {1});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 5, 30},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(IndexTest, index_infershape_broadcast_03)
{
    vector<int64_t> indexed_sizes_data{1, 1};
    gert::InfershapeContextPara::TensorDescription x_shape({{10, 20}, {10, 20}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indexed_sizes_shape({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true,
                                                                       indexed_sizes_data.data());
    gert::InfershapeContextPara::TensorDescription indexed_strides_shape({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_01({{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription indices_shape_02({{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription y({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara(
        "Index", {x_shape, indexed_sizes_shape, indexed_strides_shape, indices_shape_01, indices_shape_02}, {y},
        {1, 1, 1, 2}, {1});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(IndexTest, index_infershape_indexed_sizes_exceeds_x_rank_reports_ez0015)
{
    // 错误码整改负向用例：indexedSizes 元素个数超过 x 的秩时，InferShape4Index 以正式宏
    // OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON 首报 EZ0015，字段精确且唯一。
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();

    gert::StorageShape x_shape = {{1000}, {1000}};
    gert::StorageShape indexed_strides_shape = {{2}, {2}};
    gert::StorageShape indices_shape = {{1}, {1}};
    gert::Shape y_shape = {{1}};

    // x 秩为 1，indexedSizes 携带 2 个元素（shape size = 2 > 1）触发 EZ0015
    vector<int64_t> indexed_sizes_data{1, 1};
    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(indexed_sizes_data.size(), ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(static_cast<int64_t>(indexed_sizes_data.size()));
    tensor->MutableOriginShape().AppendDim(static_cast<int64_t>(indexed_sizes_data.size()));
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), indexed_sizes_data.data(),
                   indexed_sizes_data.size() * sizeof(int64_t));

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Index")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x_shape, tensor, &indexed_strides_shape, &indices_shape})
                      .OutputShapes({&y_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(std::string(errors.front().error_id.get()), "EZ0015");
    std::map<std::string, std::string> actual;
    for (size_t j = 0; j < errors.front().args_key.size(); ++j) {
        EXPECT_TRUE(actual.emplace(errors.front().args_key[j].get(), errors.front().args_value[j].get()).second);
    }
    const std::map<std::string, std::string> expected = {{"param_name", "indexedSizes"},
                                                         {"incorrect_size", "2"},
                                                         {"reason", "the element num must not be larger than the "
                                                                    "rank of x 1"}};
    for (const auto& entry : expected) {
        EXPECT_EQ(actual.at(entry.first), entry.second);
    }
    EXPECT_FALSE(actual.at("op_name").empty());
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}
