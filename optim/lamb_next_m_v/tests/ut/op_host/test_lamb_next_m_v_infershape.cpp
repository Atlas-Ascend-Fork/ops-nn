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
#include <iostream>
#include <cstdint>
#include <string>
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/lamb_next_m_v_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "base/err_mgr.h"
#include "platform/platform_info.h"

using namespace ge;

namespace {
// ErrorManager 上下文隔离方式与 rnn/gru_grad/tests/ut/op_host/test_gru_grad_tiling.cpp 一致：
// 用独立 work_stream_id 记录，避免共享 Host UT 进程内的用例串扰。
class ScopedErrorContext {
public:
    ScopedErrorContext() : savedContext_(error_message::GetErrMgrContext())
    {
        auto context = savedContext_;
        context.work_stream_id = reinterpret_cast<std::uintptr_t>(this);
        error_message::SetErrMgrContext(context);
        (void)error_message::GetErrMgrRawErrorMessages();
    }

    ~ScopedErrorContext()
    {
        (void)error_message::GetErrMgrRawErrorMessages();
        error_message::SetErrMgrContext(savedContext_);
    }

    ScopedErrorContext(const ScopedErrorContext&) = delete;
    ScopedErrorContext& operator=(const ScopedErrorContext&) = delete;

private:
    error_message::ErrorManagerContext savedContext_;
};

struct CapturedError {
    size_t count = 0;
    std::string code;
    std::vector<std::string> keys;
    std::vector<std::string> values;
};

// 共用广播工具 (infershape_broadcast_util) 内部会用 OP_LOGE 逐层上报框架内部码 EZ9999，
// 这些不是本算子的上报入口；统计「首错唯一」时排除内部码，只保留预定义错误码。
bool IsFrameworkInternalError(const std::string& code) { return code == (std::string("EZ") + "9999"); }

CapturedError CaptureError()
{
    CapturedError captured;
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    for (const auto& error : errors) {
        const std::string code = error.error_id == nullptr ? "" : error.error_id.get();
        if (IsFrameworkInternalError(code)) {
            continue;
        }
        ++captured.count;
        captured.code = code;
        captured.keys.clear();
        captured.values.clear();
        for (const auto& key : error.args_key) {
            captured.keys.emplace_back(key == nullptr ? "" : key.get());
        }
        for (const auto& value : error.args_value) {
            captured.values.emplace_back(value == nullptr ? "" : value.get());
        }
    }
    return captured;
}

// 按 Arglist 的 key 取实际错误值；缺失时返回可读标记，便于断言失败时定位。
std::string ErrorValue(const CapturedError& error, const std::string& key)
{
    for (size_t i = 0; i < error.keys.size() && i < error.values.size(); ++i) {
        if (error.keys[i] == key) {
            return error.values[i];
        }
    }
    return "<missing:" + key + ">";
}
} // namespace

class LambNextMVProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(LambNextMVProtoTest, lamb_next_mv_case_2d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMV")->infer_shape;
    gert::Shape inShape = {-1, -1};
    gert::Shape outShape = {};
    gert::Shape expShape = {-1, -1};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(13, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape,
                                    &inShape, &inShape, &inShape, &inShape, &inShape})
                      .OutputShapes({&outShape, &outShape, &outShape, &outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto od0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*od0), Ops::Base::ToString(expShape));
    auto od1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*od1), Ops::Base::ToString(expShape));
    auto od2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    ASSERT_EQ(Ops::Base::ToString(*od2), Ops::Base::ToString(expShape));
    auto od3 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(3);
    ASSERT_EQ(Ops::Base::ToString(*od3), Ops::Base::ToString(expShape));
}

TEST_F(LambNextMVProtoTest, lamb_next_mv_case_fp16_4d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMV")->infer_shape;
    gert::Shape inShape = {-1, -1, -1, -1};
    gert::Shape outShape = {};
    gert::Shape expShape = {-1, -1, -1, -1};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(13, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape,
                                    &inShape, &inShape, &inShape, &inShape, &inShape})
                      .OutputShapes({&outShape, &outShape, &outShape, &outShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto od0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*od0), Ops::Base::ToString(expShape));
    auto od1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*od1), Ops::Base::ToString(expShape));
    auto od2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    ASSERT_EQ(Ops::Base::ToString(*od2), Ops::Base::ToString(expShape));
    auto od3 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(3);
    ASSERT_EQ(Ops::Base::ToString(*od3), Ops::Base::ToString(expShape));
}

TEST_F(LambNextMVProtoTest, lambnextmv_infer_datatype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMV"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMV")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);
    ge::DataType in0 = ge::DT_FLOAT;
    ge::DataType in1 = ge::DT_FLOAT;
    ge::DataType in2 = ge::DT_FLOAT;
    ge::DataType in3 = ge::DT_FLOAT;
    ge::DataType in4 = ge::DT_FLOAT;
    ge::DataType in5 = ge::DT_FLOAT;
    ge::DataType in6 = ge::DT_FLOAT;
    ge::DataType in7 = ge::DT_FLOAT;
    ge::DataType in8 = ge::DT_FLOAT;
    ge::DataType in9 = ge::DT_FLOAT;
    ge::DataType in10 = ge::DT_FLOAT;
    ge::DataType in11 = ge::DT_FLOAT;
    ge::DataType in12 = ge::DT_FLOAT;
    ge::DataType expOut0 = ge::DT_FLOAT;
    ge::DataType expOut1 = ge::DT_FLOAT;
    ge::DataType expOut2 = ge::DT_FLOAT;
    ge::DataType expOut3 = ge::DT_FLOAT;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(13, 4)
                             .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes(
                                 {&in0, &in1, &in2, &in3, &in4, &in5, &in6, &in7, &in8, &in9, &in10, &in11, &in12})
                             .OutputDataTypes({&expOut0, &expOut1, &expOut2, &expOut3})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT);
}

// InferShape4LambNextMV: 输入 rank 超出 0D-8D 上限必须被拒绝。
TEST_F(LambNextMVProtoTest, lamb_next_mv_infershape_reject_rank_over_max)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMV"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMV")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape inShape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    gert::Shape outShape = {};
    ScopedErrorContext errorContext;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(13, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape,
                                    &inShape, &inShape, &inShape, &inShape, &inShape})
                      .OutputShapes({&outShape, &outShape, &outShape, &outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_NE(holder.GetContext<gert::InferShapeContext>(), nullptr);
    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0012");
    EXPECT_EQ(ErrorValue(error, "param_name"), "input0");
    EXPECT_EQ(ErrorValue(error, "incorrect_dim"), "9D");
}

// InferShape4LambNextMV: 输入形状不可广播必须被拒绝。
TEST_F(LambNextMVProtoTest, lamb_next_mv_infershape_reject_broadcast_failure)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMV"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMV")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape in0Shape = {2, 3};
    gert::Shape in1Shape = {4, 5};
    gert::Shape inOtherShape = {1};
    gert::Shape outShape = {};
    ScopedErrorContext errorContext;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(13, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&in0Shape, &in1Shape, &inOtherShape, &inOtherShape, &inOtherShape, &inOtherShape,
                                    &inOtherShape, &inOtherShape, &inOtherShape, &inOtherShape, &inOtherShape,
                                    &inOtherShape, &inOtherShape})
                      .OutputShapes({&outShape, &outShape, &outShape, &outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_NE(holder.GetContext<gert::InferShapeContext>(), nullptr);
    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0010");
    EXPECT_EQ(ErrorValue(error, "param_names"), "all inputs");
}

// InferDataType4LambNextMV: 输入 dtype 非 float16/float 必须被拒绝。
TEST_F(LambNextMVProtoTest, lamb_next_mv_inferdatatype_reject_invalid_dtype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMV"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMV")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);
    ScopedErrorContext errorContext;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(13, 4)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0020");
    EXPECT_EQ(ErrorValue(error, "param_name"), "input_mul3");
}

// InferDataType4LambNextMV: 输入 dtype 不一致必须被拒绝。
TEST_F(LambNextMVProtoTest, lamb_next_mv_inferdatatype_reject_inconsistent_dtype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMV"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMV")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);
    ScopedErrorContext errorContext;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(13, 4)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0021");
    EXPECT_EQ(ErrorValue(error, "param_names"), "input1,input0");
}
