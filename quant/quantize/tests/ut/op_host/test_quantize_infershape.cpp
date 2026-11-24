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
 * \file test_quantize_infershape.cpp
 * \brief
 */
#include <iostream>
#include <gtest/gtest.h>
#include <optional>
#include <vector>
#include "array_ops.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

class QuantizeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "quantize test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "quantize test TearDown" << std::endl; }
};

TEST_F(QuantizeTest, quantize_test_case_1)
{
    //------------------------
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Quantize")->infer_shape;
    gert::StorageShape Shape = {{3, 4, 5, 6}, {3, 4, 5, 6}};
    gert::StorageShape scalesShape = {{}, {}};
    gert::StorageShape zeroPointShape = {{1}, {1}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape, &scalesShape, &zeroPointShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {3, 4, 5, 6};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

namespace {
ge::graphStatus RunInferDataType(const std::optional<std::string>& dtypeAttr, ge::DataType& outDtype)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("Quantize");
    if (impl == nullptr || impl->infer_datatype == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType scalesDtype = ge::DT_FLOAT;
    ge::DataType zeroPointDtype = ge::DT_INT8;
    ge::DataType yDtype = ge::DT_INT8;
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs;
    if (dtypeAttr.has_value()) {
        attrs.emplace_back("dtype", Ops::NN::AnyValue::CreateFrom<std::string>(*dtypeAttr));
    }
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(3)
                      .NodeIoNum(3, 1)
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, scalesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, zeroPointDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&xDtype, &scalesDtype, &zeroPointDtype})
                      .OutputDataTypes({&yDtype})
                      .NodeAttrs(attrs)
                      .Build();
    auto* ctx = holder.GetContext<gert::InferDataTypeContext>();
    const ge::graphStatus ret = impl->infer_datatype(ctx);
    if (ret == ge::GRAPH_SUCCESS) {
        outDtype = ctx->GetOutputDataType(0);
    }
    return ret;
}
} // namespace

TEST_F(QuantizeTest, quantize_infer_datatype_qint32)
{
    // 修复项：torch.qint32 应映射 DT_INT32（原误写 DT_UINT32，且 def 输出 dtype 列表仅含 DT_INT32）
    ge::DataType dt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType("torch.qint32", dt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dt, ge::DT_INT32);
}

TEST_F(QuantizeTest, quantize_infer_datatype_qint8)
{
    ge::DataType dt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType("torch.qint8", dt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dt, ge::DT_INT8);
}

TEST_F(QuantizeTest, quantize_infer_datatype_quint8)
{
    ge::DataType dt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType("torch.quint8", dt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dt, ge::DT_UINT8);
}

TEST_F(QuantizeTest, quantize_infer_datatype_unknown_rejected)
{
    // 对齐 rt1.0：未知 dtype 值报错拒绝，不做静默零值插入
    ge::DataType dt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType("torch.invalid", dt), ge::GRAPH_FAILED);
}

TEST_F(QuantizeTest, quantize_infer_datatype_attr_missing_default)
{
    // nullptr 防护看护：不设置 dtype 属性，GetAttrPointer 返回 nullptr，走默认 torch.qint8 → DT_INT8
    ge::DataType dt = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(std::nullopt, dt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dt, ge::DT_INT8);
}
