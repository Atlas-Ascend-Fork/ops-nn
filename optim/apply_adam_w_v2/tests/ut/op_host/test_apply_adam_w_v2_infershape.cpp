/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h> // NOLINT
#include <array>
#include <iostream>
#include <vector>
#include "infershape_test_util.h"         // NOLINT
#include "infer_datatype_context_faker.h" // NOLINT
#include "ut_op_common.h"
#include "../../../op_graph/apply_adam_w_v2_proto.h"

class ApplyAdamWV2 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyAdamWV2 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyAdamWV2 TearDown" << std::endl; }
};

TEST_F(ApplyAdamWV2, ApplyAdamWV2_infershape_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWV2"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape varShape = {{96, 256}, {96, 256}};
    gert::StorageShape mShape = {{96, 256}, {96, 256}};
    gert::StorageShape vShape = {{96, 256}, {96, 256}};
    gert::StorageShape gradShape = {{96, 256}, {96, 256}};
    gert::StorageShape stepShape = {{1}, {1}};
    gert::StorageShape max_grad_normShape = {{96, 256}, {96, 256}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &mShape, &vShape, &gradShape, &stepShape, &max_grad_normShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &max_grad_normShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// SEQ23: grad shape 与 var 不一致时应返回 GRAPH_FAILED
TEST_F(ApplyAdamWV2, infershape_reject_grad_shape_mismatch)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    gert::StorageShape varShape = {{96, 256}, {96, 256}};
    gert::StorageShape mShape = {{96, 256}, {96, 256}};
    gert::StorageShape vShape = {{96, 256}, {96, 256}};
    gert::StorageShape gradShape = {{97, 256}, {97, 256}};
    gert::StorageShape stepShape = {{1}, {1}};
    gert::StorageShape mgnShape = {{96, 256}, {96, 256}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &mShape, &vShape, &gradShape, &stepShape, &mgnShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &mgnShape})
                      .Build();
    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape(context), ge::GRAPH_FAILED);
}

// SEQ23: step 非单元素时应返回 GRAPH_FAILED
TEST_F(ApplyAdamWV2, infershape_reject_step_not_scalar)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    gert::StorageShape varShape = {{96, 256}, {96, 256}};
    gert::StorageShape mShape = {{96, 256}, {96, 256}};
    gert::StorageShape vShape = {{96, 256}, {96, 256}};
    gert::StorageShape gradShape = {{96, 256}, {96, 256}};
    gert::StorageShape stepShape = {{2}, {2}};
    gert::StorageShape mgnShape = {{96, 256}, {96, 256}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &mShape, &vShape, &gradShape, &stepShape, &mgnShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &mgnShape})
                      .Build();
    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape(context), ge::GRAPH_FAILED);
}

// SEQ23: max_grad_norm(实例化时) shape 与 var 不一致时应返回 GRAPH_FAILED
TEST_F(ApplyAdamWV2, infershape_reject_mgn_shape_mismatch)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    gert::StorageShape varShape = {{96, 256}, {96, 256}};
    gert::StorageShape mShape = {{96, 256}, {96, 256}};
    gert::StorageShape vShape = {{96, 256}, {96, 256}};
    gert::StorageShape gradShape = {{96, 256}, {96, 256}};
    gert::StorageShape stepShape = {{1}, {1}};
    gert::StorageShape mgnShape = {{97, 256}, {97, 256}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &mShape, &vShape, &gradShape, &stepShape, &mgnShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &mgnShape})
                      .Build();
    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape(context), ge::GRAPH_FAILED);
}

// SEQ23: 必选输入 shape 为空（GetInputShape 返回空指针）时应返回 GRAPH_FAILED
TEST_F(ApplyAdamWV2, infershape_reject_null_input_shape)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    gert::StorageShape varShape = {{96, 256}, {96, 256}};
    gert::StorageShape vShape = {{96, 256}, {96, 256}};
    gert::StorageShape gradShape = {{96, 256}, {96, 256}};
    gert::StorageShape stepShape = {{1}, {1}};
    gert::StorageShape mgnShape = {{96, 256}, {96, 256}};

    // 输入1(m) 不提供 shape：faker 对 nullptr 槽位不创建 tensor，GetInputShape(1) 返回空指针
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, nullptr, &vShape, &gradShape, &stepShape, &mgnShape})
                      .OutputShapes({&varShape, &vShape, &gradShape, &mgnShape})
                      .Build();
    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(opImpl->infer_shape(context), ge::GRAPH_FAILED);
}

// SEQ23: m/v/grad 与 var dtype 一致时 InferDataType 应返回 GRAPH_SUCCESS
TEST_F(ApplyAdamWV2, inferdatatype_pass_consistent_dtype)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWV2")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    ge::DataType varDtype = ge::DT_FLOAT;
    ge::DataType stepDtype = ge::DT_INT64;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .NodeInputTd(0, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, stepDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&varDtype, &varDtype, &varDtype, &varDtype, &stepDtype, &varDtype})
                      .OutputDataTypes({&outputDtype, &outputDtype, &outputDtype, &outputDtype})
                      .Build();

    gert::InferDataTypeContext* context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
}

// SEQ23: m 或 v 的 dtype 与 var 不一致时 InferDataType 应返回 GRAPH_FAILED
static void RunInferDataTypeMismatchCase(size_t mismatchInputIdx)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamWV2")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    std::array<ge::DataType, 6> inputDtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                               ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT};
    inputDtypes[mismatchInputIdx] = ge::DT_FLOAT16;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .NodeInputTd(0, inputDtypes[0], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtypes[1], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, inputDtypes[2], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, inputDtypes[3], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, inputDtypes[4], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, inputDtypes[5], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&inputDtypes[0], &inputDtypes[1], &inputDtypes[2], &inputDtypes[3],
                                       &inputDtypes[4], &inputDtypes[5]})
                      .OutputDataTypes({&outputDtype, &outputDtype, &outputDtype, &outputDtype})
                      .Build();

    gert::InferDataTypeContext* context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferDataTypeFunc(context), ge::GRAPH_FAILED);
}

TEST_F(ApplyAdamWV2, inferdatatype_reject_dtype_mismatch)
{
    RunInferDataTypeMismatchCase(1); // m 与 var dtype 不一致
    RunInferDataTypeMismatchCase(2); // v 与 var dtype 不一致
}
