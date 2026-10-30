/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/apply_proximal_gradient_descent_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class ApplyProximalGradientDescentProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyProximalGradientDescent Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyProximalGradientDescent Proto Test TearDown" << std::endl; }
};

TEST_F(ApplyProximalGradientDescentProtoTest, infershape_1d_fp32)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyProximalGradientDescent")->infer_shape;

    gert::Shape var_shape = {128};
    gert::Shape alpha_shape = {1};
    gert::Shape l1_shape = {1};
    gert::Shape l2_shape = {1};
    gert::Shape delta_shape = {128};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&var_shape, &alpha_shape, &l1_shape, &l2_shape, &delta_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ApplyProximalGradientDescentProtoTest, infershape_2d_fp16)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyProximalGradientDescent")->infer_shape;

    gert::Shape var_shape = {8, 16};
    gert::Shape alpha_shape = {1};
    gert::Shape l1_shape = {1};
    gert::Shape l2_shape = {1};
    gert::Shape delta_shape = {8, 16};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&var_shape, &alpha_shape, &l1_shape, &l2_shape, &delta_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ApplyProximalGradientDescentProtoTest, infershape_3d_fp32)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyProximalGradientDescent")->infer_shape;

    gert::Shape var_shape = {2, 3, 4};
    gert::Shape alpha_shape = {1};
    gert::Shape l1_shape = {1};
    gert::Shape l2_shape = {1};
    gert::Shape delta_shape = {2, 3, 4};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&var_shape, &alpha_shape, &l1_shape, &l2_shape, &delta_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST(ApplyProximalGradientDescentErrorCodes, InferDtypeReportsOneSpecificError)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyProximalGradientDescent");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    for (int changed = -2; changed < 5; ++changed) {
        SCOPED_TRACE(changed);
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpName("ApplyProximalGradientDescent")
            .OpType("ApplyProximalGradientDescent")
            .IOInstanceNum(std::vector<uint32_t>(5, 1), std::vector<uint32_t>(1, 1));
        for (int i = 0; i < 5; ++i) {
            auto dtype = changed == -1 ? ge::DT_INT32 : (changed == i ? ge::DT_FLOAT16 : ge::DT_FLOAT);
            builder.InputTensorDesc(i, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
        }
        for (size_t i = 0; i < 1; ++i) {
            builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
        }
        auto holder = builder.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        manager.GetRawErrorMessages();
        const auto status = impl->infer_datatype(context);
        const auto errors = manager.GetRawErrorMessages();
        if (changed == -2) {
            EXPECT_EQ(status, ge::GRAPH_SUCCESS);
            EXPECT_TRUE(errors.empty());
            EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
        } else {
            EXPECT_EQ(status, ge::GRAPH_FAILED);
            ASSERT_EQ(errors.size(), 1U);
            EXPECT_EQ(errors[0].error_id, "EZ0019");
        }
    }
    manager.GetRawErrorMessages();
    EXPECT_EQ(impl->infer_datatype(nullptr), ge::GRAPH_FAILED);
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1014");
}

TEST(ApplyProximalGradientDescentErrorCodes, InferShapeReportsOneSpecificError)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyProximalGradientDescent");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    const char* codes[] = {"EZ0010", "EZ0009", "EZ0004", "EZ1014"};
    for (int scenario = 0; scenario < 4; ++scenario) {
        SCOPED_TRACE(scenario);
        gert::Shape tensor{16}, scalar{1}, different{8}, invalidScalar{2}, output, outputM;
        auto holder = gert::InferShapeContextFaker()
                          .SetOpType("ApplyProximalGradientDescent")
                          .NodeIoNum(5, 1)
                          .IrInstanceNum(std::vector<uint32_t>(5, 1))
                          .InputShapes({&tensor, scenario == 1 ? &invalidScalar : &scalar, &scalar, &scalar,
                                        scenario == 0 ? &different : &tensor})
                          .OutputShapes({&output})
                          .Build();
        auto* context = holder.GetContext<gert::InferShapeContext>();
        ASSERT_NE(context, nullptr);
        auto* kernelContext = holder.GetContext<gert::KernelContext>();
        if (scenario == 2) {
            kernelContext->MutableInput(0)->Set(nullptr, nullptr);
            ASSERT_EQ(context->GetInputShape(0), nullptr);
        }
        if (scenario == 3) {
            kernelContext->GetOutput(0)->Set(nullptr, nullptr);
            ASSERT_EQ(context->GetOutputShape(0), nullptr);
        }
        manager.GetRawErrorMessages();
        EXPECT_EQ(impl->infer_shape(context), ge::GRAPH_FAILED);
        const auto errors = manager.GetRawErrorMessages();
        ASSERT_EQ(errors.size(), 1U);
        EXPECT_EQ(errors[0].error_id, codes[scenario]);
    }
    manager.GetRawErrorMessages();
    EXPECT_EQ(impl->infer_shape(nullptr), ge::GRAPH_FAILED);
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1014");
}
