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
#include <vector>
#include <map>
#include "base/err_mgr.h"
#include <gtest/gtest.h>
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"

using namespace std;
using namespace ge;
using namespace gert;

static void CheckRawError(const std::string& code, const std::map<std::string, std::string>& expected)
{
    auto messages = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(messages.size(), 1U);
    const auto& message = messages.front();
    ASSERT_NE(message.error_id, nullptr);
    EXPECT_EQ(std::string(message.error_id.get()), code);
    ASSERT_EQ(message.args_key.size(), expected.size() + 1U);
    ASSERT_EQ(message.args_key.size(), message.args_value.size());
    std::map<std::string, std::string> arguments;
    for (size_t i = 0; i < message.args_key.size(); ++i) {
        ASSERT_NE(message.args_key[i], nullptr);
        ASSERT_NE(message.args_value[i], nullptr);
        EXPECT_TRUE(arguments.emplace(message.args_key[i].get(), message.args_value[i].get()).second);
    }
    EXPECT_FALSE(arguments.at("op_name").empty());
    for (const auto& entry : expected) {
        EXPECT_EQ(arguments.at(entry.first), entry.second);
    }
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

TEST(GlobalLpPoolInferShapeReporting, rank_and_normal_shapes)
{
    const auto* impl = OpImplRegistry::GetInstance().GetOpImpl("GlobalLpPool");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    for (int64_t rank : {3, 4, 5}) {
        ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
        (void)error_message::GetErrMgrRawErrorMessages();
        gert::StorageShape xShape;
        xShape.MutableStorageShape().SetDimNum(rank);
        xShape.MutableOriginShape().SetDimNum(rank);
        for (int64_t i = 0; i < rank; ++i) {
            xShape.MutableStorageShape().SetDim(i, i + 2);
            xShape.MutableOriginShape().SetDim(i, i + 2);
        }
        gert::StorageShape yShape;
        auto holder = gert::InferShapeContextFaker()
                          .SetOpType("GlobalLpPool")
                          .NodeIoNum(1, 1)
                          .IrInstanceNum({1})
                          .InputShapes({&xShape})
                          .OutputShapes({&yShape})
                          .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                          .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                          .Build();
        auto* context = holder.GetContext<gert::InferShapeContext>();
        ASSERT_NE(context, nullptr);
        if (rank == 3) {
            EXPECT_EQ(impl->infer_shape(context), ge::GRAPH_FAILED);
            CheckRawError("EZ0011",
                          {{"param_name", "x"}, {"incorrect_dim", "3"}, {"correct_dim", "4 (NCHW) or 5 (NCD0D1D2)"}});
        } else {
            EXPECT_EQ(impl->infer_shape(context), ge::GRAPH_SUCCESS);
            EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
            const auto* output = context->GetOutputShape(0);
            ASSERT_NE(output, nullptr);
            EXPECT_EQ(output->GetDimNum(), static_cast<size_t>(rank));
            for (int64_t i = 0; i < rank; ++i) {
                EXPECT_EQ(output->GetDim(i), i < 2 ? i + 2 : 1);
            }
        }
    }
}
