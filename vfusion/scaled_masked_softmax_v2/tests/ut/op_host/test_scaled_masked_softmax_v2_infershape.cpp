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
 * \file test_scaled_masked_softmax_v2_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "log/log.h"

class ScaledMaskedSoftmaxV2Proto : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScaledMaskedSoftmaxV2 Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ScaledMaskedSoftmaxV2 Proto Test TearDown" << std::endl; }
};

TEST_F(ScaledMaskedSoftmaxV2Proto, inferShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScaledMaskedSoftmaxV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{1, 32, 32, 128}, {1, 32, 32, 128}};
    gert::StorageShape maskShape = {{1, 32, 32, 128}, {1, 32, 32, 128}};
    gert::StorageShape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &maskShape})
                      .OutputShapes({&yShape})
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, 32, 32, 128]");
}

TEST_F(ScaledMaskedSoftmaxV2Proto, inferDataType)
{
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScaledMaskedSoftmaxV2")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);
    ge::DataType xDtype = ge::DT_BF16;
    ge::DataType maskDtype = ge::DT_BOOL;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputDataTypes({&xDtype, &maskDtype})
                      .OutputDataTypes({&yDtype})
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    ASSERT_EQ(inferDataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(holder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_BF16);
}

TEST_F(ScaledMaskedSoftmaxV2Proto, infer_shape_reports_missing_mask)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ScaledMaskedSoftmaxV2");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    gert::StorageShape xShape = {{1, 32, 32, 128}, {1, 32, 32, 128}};
    gert::StorageShape yShape;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ScaledMaskedSoftmaxV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 0}, {1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    auto& manager = ErrorManager::GetInstance();
    (void)manager.GetRawErrorMessages();
    EXPECT_EQ(impl->infer_shape(context), ge::GRAPH_FAILED);
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0037");
    EXPECT_NE(errors[0].error_message.find("mask"), std::string::npos) << errors[0].error_message;
}
