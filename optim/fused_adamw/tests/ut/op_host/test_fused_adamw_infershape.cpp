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
#include <array>
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/storage_format.h"
#include "ut_op_util.h"
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"

#include "gtest/gtest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/kernel_context.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"

using namespace ge;
using namespace ut_util;
using namespace std;

class FusedAdamwInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FusedAdamwInferShape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "FusedAdamwInferShape TearDown" << std::endl; }
};

namespace {
std::vector<int64_t> ToVectorForFused(const gert::Shape& shape)
{
    size_t shape_size = shape.GetDimNum();
    std::vector<int64_t> shape_vec(shape_size, 0);

    for (size_t i = 0; i < shape_size; i++) {
        shape_vec[i] = shape.GetDim(i);
    }
    return shape_vec;
}
} // namespace

TEST_F(FusedAdamwInferShape, test_fused_adamw_infershape_same_shape)
{
    gert::StorageShape paramsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape gradsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape expAvgsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape expAvgSqsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape max_exp_avg_sqsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape stepShape = {{1}, {1}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    gert::StorageShape foundInfShape = {{1}, {1}};
    std::string opType("FusedAdamw");
    auto infershape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType)->infer_shape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(8, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&paramsShape, &gradsShape, &expAvgsShape, &expAvgSqsShape, &max_exp_avg_sqsShape,
                                    &stepShape, &gradScaleShape, &foundInfShape})
                      .OutputShapes({&paramsShape, &gradsShape, &expAvgsShape, &expAvgSqsShape, &max_exp_avg_sqsShape})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infershape_func(context), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedOutputShape1 = {3, 4, 5};
    for (int i = 0; i < 4; i++) {
        auto tmpOutShape1 = context->GetOutputShape(i);
        EXPECT_EQ(ToVectorForFused(*tmpOutShape1), expectedOutputShape1);
    }
}

// 负向场景覆盖说明：computeNodeInfo 判空（GetComputeNodeInfo）、paramsInstanceInfo 判空
// （GetInputInstanceInfo(INPUT_PARAMS_INDEX=0)）与 4 个输出实例信息判空
// （GetIrOutputInstanceInfo(OUTPUT_PARAMS/EXP_AVGS/EXP_AVG_SQS/MAX_EXP_AVG_SQS_INDEX)）检查点
// 不可构造：InferShapeContextFaker::Build 必然构造 ComputeNodeInfo；使
// GetInputInstanceInfo(0)/GetIrOutputInstanceInfo(i) 为空需要 IR 原型数为 0，
// 此时 faker 构建空 holder（GetContext 返回空指针），用例自身段错误；
// 上述检查点属 EZ0037 同上报路由，由场景3/4 的 shape 判空用例代表覆盖。
// InferDataTypeForFusedAdamW 中同型检查点（computeNodeInfo/paramsInstanceInfo/4 个输出实例信息）
// 因同样限制不单独构造，本文件仅驱动 infer_shape 入口

// 负向场景3：grads 输入实例个数为 0，GetDynamicInputShape(INPUT_GRADS_INDEX, 0) 返回空指针，
// 整改后应上报 EZ0037（param_name=gradsShape）首错并返回失败
TEST_F(FusedAdamwInferShape, fused_adamw_infershape_null_input_shape_003)
{
    gert::StorageShape paramsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape expAvgsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape expAvgSqsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape maxExpAvgSqsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape stepShape = {{1}, {1}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    gert::StorageShape foundInfShape = {{1}, {1}};
    std::string opType("FusedAdamw");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto infershape_func = opImpl->infer_shape;

    // grads 的输入实例个数置 0，其余输入保持 1 个实例
    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1, 0, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1})
                      .InputShapes({&paramsShape, nullptr, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape,
                                    &stepShape, &gradScaleShape, &foundInfShape})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(context->GetDynamicInputShape(1, 0), nullptr);
    EXPECT_EQ(infershape_func(context), ge::GRAPH_FAILED);
}

// 负向场景4：5 个动态输入各 2 个实例而 4 个输出原型各 1 个实例，i=1 时
// GetOutputShape(OUTPUT_MAX_EXP_AVG_SQS_INDEX 的 InstanceStart + 1) 越界返回空指针，
// 整改后应上报 EZ0037（param_name=maxExpAvgSqsRefShape）首错并返回失败
TEST_F(FusedAdamwInferShape, fused_adamw_infershape_null_output_shape_004)
{
    gert::StorageShape tensorShape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape scalarShape = {{1}, {1}};
    // 前 5 个输入原型各 2 个实例（共 10 个），后 3 个标量输入各 1 个实例
    std::vector<gert::StorageShape> inputShapes(10, tensorShape);
    inputShapes.push_back(scalarShape);
    inputShapes.push_back(scalarShape);
    inputShapes.push_back(scalarShape);
    std::vector<gert::StorageShape*> inputShapeRefs;
    for (auto& shape : inputShapes) {
        inputShapeRefs.push_back(&shape);
    }
    std::string opType("FusedAdamw");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto infershape_func = opImpl->infer_shape;

    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({2, 2, 2, 2, 2, 1, 1, 1}, {1, 1, 1, 1})
                      .InputShapes(inputShapeRefs)
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    // 输出仅 4 个实例槽位，i=1 时 max_exp_avg_sqs 的输出 shape 下标为 4，越界为空
    ASSERT_EQ(context->GetOutputShape(4), nullptr);
    EXPECT_EQ(infershape_func(context), ge::GRAPH_FAILED);
}

// 负向场景5：params 输入实例个数为 0（IrInstanceNum 首项为 0），
// 对应整改检查点 OP_LOGE_FOR_INVALID_TENSORNUM（inputNum == 0 分支，EZ0022 首报）
TEST_F(FusedAdamwInferShape, fused_adamw_infershape_params_instance_num_zero_005)
{
    gert::StorageShape gradsShape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape expAvgsShape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape expAvgSqsShape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape maxExpAvgSqsShape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape stepShape = {{1}, {1}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    gert::StorageShape foundInfShape = {{1}, {1}};
    std::string opType("FusedAdamw");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto infershape_func = opImpl->infer_shape;

    // params 实例个数置 0（形状槽位传 nullptr 占位），其余输入保持 1 个实例
    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({0, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1})
                      .InputShapes({nullptr, &gradsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape,
                                    &stepShape, &gradScaleShape, &foundInfShape})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(context->GetDynamicInputShape(0, 0), nullptr);
    EXPECT_EQ(infershape_func(context), ge::GRAPH_FAILED);
}

// 负向场景6：grads 与 params 等 5 个动态输入 shape 不一致，
// 对应整改检查点 OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON（shape 不匹配分支）
TEST_F(FusedAdamwInferShape, fused_adamw_infershape_shape_mismatch_006)
{
    gert::StorageShape paramsShape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape gradsShape = {{3, 4, 6}, {3, 4, 6}};
    gert::StorageShape expAvgsShape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape expAvgSqsShape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape maxExpAvgSqsShape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape stepShape = {{1}, {1}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    gert::StorageShape foundInfShape = {{1}, {1}};
    std::string opType("FusedAdamw");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto infershape_func = opImpl->infer_shape;

    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1})
                      .InputShapes({&paramsShape, &gradsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape,
                                    &stepShape, &gradScaleShape, &foundInfShape})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infershape_func(context), ge::GRAPH_FAILED);
}

// 场景7：InferDataTypeForFusedAdamw 正向用例，驱动 infer_datatype 入口
// （源文件中该函数此前无用例覆盖，4 个输出 dtype 均取对应动态输入的 dtype）
TEST_F(FusedAdamwInferShape, fused_adamw_infershape_infer_datatype_007)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FusedAdamw"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedAdamw")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_ref = ge::DT_FLOAT;
    ge::DataType output_ref = ge::DT_FLOAT;
    std::vector<void*> input_refs(8, &input_ref);
    std::vector<void*> output_refs(4, &output_ref);
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(8, 4)
                              .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1})
                              .InputDataTypes(input_refs)
                              .OutputDataTypes(output_refs)
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);

    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(context->GetOutputDataType(i), input_ref);
    }
}
