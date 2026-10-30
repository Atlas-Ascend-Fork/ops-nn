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
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h" // NOLINT
#include <iostream>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"
#include "../../../op_graph/apply_ftrl_proto.h"

class ApplyFtrl : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyFtrl SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyFtrl TearDown" << std::endl; }
};

TEST_F(ApplyFtrl, ApplyFtrl_infershape_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFtrl"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFtrl")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape varShape = {{96, 256}, {96, 256}};
    gert::StorageShape accumShape = {{96, 256}, {96, 256}};
    gert::StorageShape linearShape = {{96, 256}, {96, 256}};
    gert::StorageShape gradShape = {{96, 256}, {96, 256}};
    gert::StorageShape lrShape = {{1}, {1}};
    gert::StorageShape l1Shape = {{1}, {1}};
    gert::StorageShape l2Shape = {{1}, {1}};
    gert::StorageShape lrPowerShape = {{1}, {1}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(8, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &accumShape, &linearShape, &gradShape, &lrShape, &l1Shape, &l2Shape,
                                    &lrPowerShape})
                      .OutputShapes({&varShape, &accumShape, &linearShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(ApplyFtrl, missing_lr_reports_input_once)
{
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFtrl");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    gert::StorageShape tensorShape = {{2, 3}, {2, 3}};
    gert::StorageShape scalarShape = {{1}, {1}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ApplyFtrl")
                      .NodeIoNum(8, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&tensorShape, &tensorShape, &tensorShape, &tensorShape, nullptr, &scalarShape,
                                    &scalarShape, &scalarShape})
                      .OutputShapes({&tensorShape})
                      .Build();
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    auto context = holder.GetContext<gert::InferShapeContext>();
    // The builder materializes default shapes for nullptr entries; clear the actual input slot.
    auto kernelContext = holder.GetContext<gert::KernelContext>();
    ASSERT_NE(kernelContext->MutableInput(4), nullptr);
    kernelContext->MutableInput(4)->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetInputShape(4), nullptr);
    EXPECT_EQ(impl->infer_shape(context), ge::GRAPH_FAILED);
    auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0004");
    EXPECT_NE(errors[0].error_message.find("ApplyFtrl"), std::string::npos);
    EXPECT_NE(errors[0].error_message.find("lr"), std::string::npos);
    // InferShape-only UTs can use the SDK's older EZ0004 template.
    const auto& message = errors[0].error_message;
    EXPECT_TRUE(message == "Parameter lr of ApplyFtrl is required, but it is empty." ||
                message == "OP[ApplyFtrl] get [lr] failed.")
        << message;
}

TEST_F(ApplyFtrl, missing_output_storage_reports_infershape_failure_once)
{
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFtrl");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    gert::StorageShape tensorShape = {{2, 3}, {2, 3}};
    gert::StorageShape scalarShape = {{1}, {1}};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ApplyFtrl")
                      .NodeIoNum(8, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&tensorShape, &tensorShape, &tensorShape, &tensorShape, &scalarShape, &scalarShape,
                                    &scalarShape, &scalarShape})
                      .OutputShapes({&tensorShape})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    auto kernelContext = holder.GetContext<gert::KernelContext>();
    ASSERT_NE(kernelContext->GetOutput(0), nullptr);
    kernelContext->GetOutput(0)->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetOutputShape(0), nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    EXPECT_EQ(impl->infer_shape(context), ge::GRAPH_FAILED);
    auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1014");
    EXPECT_NE(errors[0].error_message.find("ApplyFtrl"), std::string::npos);
    EXPECT_NE(errors[0].error_message.find("output var shape storage is nullptr"), std::string::npos);
    EXPECT_NE(errors[0].error_message.find("expected output-slot count=1"), std::string::npos);
}

TEST_F(ApplyFtrl, null_context_reports_infershape_failure_once)
{
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFtrl");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    EXPECT_EQ(impl->infer_shape(nullptr), ge::GRAPH_FAILED);
    auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1014");
    EXPECT_NE(errors[0].error_message.find("ApplyFtrl"), std::string::npos);
    EXPECT_NE(errors[0].error_message.find("InferShapeContext is nullptr"), std::string::npos);
}

TEST(ApplyFtrlErrorCodes, InferDtypeReportsOneSpecificError)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFtrl");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    for (int changed = -2; changed < 8; ++changed) {
        SCOPED_TRACE(changed);
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpName("ApplyFtrl")
            .OpType("ApplyFtrl")
            .IOInstanceNum(std::vector<uint32_t>(8, 1), std::vector<uint32_t>(1, 1));
        for (int i = 0; i < 8; ++i) {
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
