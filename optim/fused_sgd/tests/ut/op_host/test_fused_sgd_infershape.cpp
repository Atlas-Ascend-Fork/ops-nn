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

class FusedSgdInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FusedSgdInferShape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "FusedSgdInferShape TearDown" << std::endl; }
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

TEST_F(FusedSgdInferShape, test_fused_sgd_infershape_same_shape)
{
    gert::StorageShape paramsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape gradsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape momentumShape = {{3, 4, 5}, {3, 6}};
    std::string opType("FusedSgd");
    auto infershape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType)->infer_shape;

    gert::StorageShape paramsRefShape;
    gert::StorageShape gradsRefShape;
    gert::StorageShape momentumRefShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&paramsShape, &gradsShape, &momentumShape})
                      .OutputShapes({&paramsShape, &gradsShape, &momentumShape})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infershape_func(context), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedOutputShape1 = {3, 4, 5};
    for (int i = 0; i < 3; i++) {
        auto tmpOutShape1 = context->GetOutputShape(i);
        EXPECT_EQ(ToVectorForFused(*tmpOutShape1), expectedOutputShape1);
    }
}

// 负向场景覆盖说明：computeNodeInfo 判空（GetComputeNodeInfo）与 paramsInstanceInfo 判空
// （GetInputInstanceInfo(INPUT_PARAMS_INDEX=0)）检查点不可构造：InferShapeContextFaker::Build
// 必然构造 ComputeNodeInfo；使 GetInputInstanceInfo(0) 为空需要 IR 原型数为 0，
// 此时 faker 构建空 holder（GetContext 返回空指针），用例自身段错误；
// 上述检查点属 EZ0037 同上报路由，由场景3/4 的 shape 判空用例代表覆盖。
// paramsShape 判空（GetDynamicInputShape(INPUT_PARAMS_INDEX, i)）与 gradsShape 同路由，
// 由场景3 代表覆盖；momentumShape 判空（momentumBufferListInput 非空分支内
// GetDynamicInputShape(INPUT_MOMENTUM_BUFFER_INDEX, i)）与 paramsRefShape/gradsRefShape 判空
// （GetOutputShape）同路由，由场景4 代表覆盖，场景4 已进入 momentum 非空分支
// （momentum 实例 0 维度均非 0）；3 个 GetIrOutputInstanceInfo 在源码中未判空
// （非本次整改点），不构造用例。InferDataTypeForFusedSgd 中同型检查点
// （computeNodeInfo/paramsInstanceInfo）因同样限制不单独构造，本文件仅驱动 infer_shape 入口

// 负向场景3：grads 输入实例个数为 0，GetDynamicInputShape(INPUT_GRADS_INDEX, 0) 返回空指针，
// 整改后应上报 EZ0037（param_name=gradsShape）首错并返回失败
TEST_F(FusedSgdInferShape, fused_sgd_infershape_null_input_shape_003)
{
    gert::StorageShape paramsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape momentumShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    std::string opType("FusedSgd");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto infershape_func = opImpl->infer_shape;

    // grads 的输入实例个数置 0，其余输入保持 1 个实例
    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1, 0, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({&paramsShape, nullptr, &momentumShape, &gradScaleShape})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(context->GetDynamicInputShape(1, 0), nullptr);
    EXPECT_EQ(infershape_func(context), ge::GRAPH_FAILED);
}

// 负向场景4：3 个动态输入各 2 个实例而 3 个输出原型各 1 个实例，i=1 时
// GetOutputShape(OUTPUT_MOMENTUM_BUFFER_INDEX 的 InstanceStart + 1) 越界返回空指针，
// 整改后应上报 EZ0037（param_name=momentumRefShape）首错并返回失败
TEST_F(FusedSgdInferShape, fused_sgd_infershape_null_output_shape_004)
{
    gert::StorageShape tensorShape = {{3, 4, 5}, {3, 4, 5}};
    gert::StorageShape scalarShape = {{1}, {1}};
    // 前 3 个输入原型各 2 个实例（共 6 个），后 2 个标量输入各 1 个实例
    std::vector<gert::StorageShape> inputShapes(6, tensorShape);
    inputShapes.push_back(scalarShape);
    inputShapes.push_back(scalarShape);
    std::vector<gert::StorageShape*> inputShapeRefs;
    for (auto& shape : inputShapes) {
        inputShapeRefs.push_back(&shape);
    }
    std::string opType("FusedSgd");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto infershape_func = opImpl->infer_shape;

    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({2, 2, 2, 1, 1}, {1, 1, 1})
                      .InputShapes(inputShapeRefs)
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    // 输出仅 3 个实例槽位，i=1 时 momentum_buffer_list 的输出 shape 下标为 3，越界为空
    ASSERT_EQ(context->GetOutputShape(3), nullptr);
    EXPECT_EQ(infershape_func(context), ge::GRAPH_FAILED);
}

// 负向场景5：必选输入 params 的实例个数置 0（IrInstanceNum 第 1 个输入实例数为 0），
// GetInputInstanceInfo(INPUT_PARAMS_INDEX)->GetInstanceNum() 为 0，触发 inputNum==0 检查点，
// 整改后应上报 EZ0022（param_name=params，期望 tensorNum > 0）首错并返回失败
TEST_F(FusedSgdInferShape, fused_sgd_infershape_input_num_zero_005)
{
    gert::StorageShape gradsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape momentumShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    std::string opType("FusedSgd");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto infershape_func = opImpl->infer_shape;

    // params 的输入实例个数置 0（InputShapes 第 1 位 nullptr 占位），其余输入保持 1 个实例
    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({0, 1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({nullptr, &gradsShape, &momentumShape, &gradScaleShape})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context->GetComputeNodeInfo(), nullptr);
    ASSERT_EQ(context->GetComputeNodeInfo()->GetInputInstanceInfo(0)->GetInstanceNum(), 0u);
    EXPECT_EQ(infershape_func(context), ge::GRAPH_FAILED);
}

// 负向场景6：momentum_buffer_list 非空（无 0 维），params 与 momentum_buffer_list shape 不一致，
// 触发 shape 不匹配检查点的 momentum 非空分支，整改后应上报 EZ0016/EZ0017 首错并返回失败
TEST_F(FusedSgdInferShape, fused_sgd_infershape_shape_mismatch_momentum_006)
{
    gert::StorageShape paramsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape gradsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape momentumShape = {{3, 4, 6}, {3, 6}};
    std::string opType("FusedSgd");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto infershape_func = opImpl->infer_shape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({&paramsShape, &gradsShape, &momentumShape})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infershape_func(context), ge::GRAPH_FAILED);
}

// 负向场景7：momentum_buffer_list 含 0 维（被判定为空 tensorlist 置空），params 与 grads shape 不一致，
// 触发 shape 不匹配检查点的 momentum 空分支（同时覆盖 0 维检测逻辑），整改后应上报首错并返回失败
TEST_F(FusedSgdInferShape, fused_sgd_infershape_shape_mismatch_no_momentum_007)
{
    gert::StorageShape paramsShape = {{3, 4, 5}, {3, 6}};
    gert::StorageShape gradsShape = {{3, 4, 6}, {3, 6}};
    gert::StorageShape momentumShape = {{0, 4, 5}, {3, 6}};
    std::string opType("FusedSgd");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto infershape_func = opImpl->infer_shape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({&paramsShape, &gradsShape, &momentumShape})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infershape_func(context), ge::GRAPH_FAILED);
}

// 场景8：InferDataTypeForFusedSgd 正向用例，驱动 infer_datatype 入口
// （源文件中该函数此前无用例覆盖），3 个输出 dtype 均取对应动态输入的 dtype
TEST_F(FusedSgdInferShape, fused_sgd_infershape_infer_datatype_008)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FusedSgd"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedSgd")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_ref = ge::DT_FLOAT;
    ge::DataType output_ref = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(5, 3)
                              .IrInstanceNum({1, 1, 1, 1, 1}, {1, 1, 1})
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref, &input_ref})
                              .OutputDataTypes({&output_ref, &output_ref, &output_ref})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputDataType(0), input_ref);
    EXPECT_EQ(context->GetOutputDataType(1), input_ref);
    EXPECT_EQ(context->GetOutputDataType(2), input_ref);
}
