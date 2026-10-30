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

#include "base/context_builder/op_infer_shape_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "exe_graph/runtime/runtime_tensor.h"

namespace {
void CheckRegisteredShape(ge::Format format, const gert::Shape& inputShape, const gert::Shape& expected,
                          ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    // The graph UT runner loads the current source-built Host library into
    // this V2 registry. Keep the tensor alive for the context's borrowed input.
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto* impl = registry->GetOpImpl("BN3DTrainingReduce");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);

    gert::StorageShape storage;
    storage.MutableOriginShape() = inputShape;
    storage.MutableStorageShape() = inputShape;
    gert::Tensor input(storage, gert::StorageFormat(format, format, {}), ge::DT_FLOAT16);
    gert::OpInferShapeContextBuilder builder;
    builder.OpType("BN3DTrainingReduce").OpName("bn3d_training_reduce_shape");
    builder.IONum(1, 2).InputTensors({&input});
    builder.OutputTensorDesc(0, ge::DT_FLOAT, format, format);
    builder.OutputTensorDesc(1, ge::DT_FLOAT, format, format);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetOutputShape(0), nullptr);
    ASSERT_NE(context->GetOutputShape(1), nullptr);
    *context->GetOutputShape(0) = gert::Shape({91, 92});
    *context->GetOutputShape(1) = gert::Shape({93});
    ASSERT_EQ(impl->infer_shape(context), status);
    if (status == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(*context->GetOutputShape(0), expected);
        EXPECT_EQ(*context->GetOutputShape(1), expected);
    }
}
} // namespace

TEST(BN3DTrainingReduceGraphInferTest, RegisteredNcdhwUsesChannelAxisOne)
{
    CheckRegisteredShape(ge::FORMAT_NCDHW, {2, 3, 4, 4, 8}, {3});
}

TEST(BN3DTrainingReduceGraphInferTest, RegisteredNdhwcUsesLastChannelAxis)
{
    CheckRegisteredShape(ge::FORMAT_NDHWC, {2, 3, 4, 4, 8}, {8});
}

TEST(BN3DTrainingReduceGraphInferTest, RegisteredNdhwcRejectsRankFour)
{
    CheckRegisteredShape(ge::FORMAT_NDHWC, {2, 3, 4, 8}, {}, ge::GRAPH_FAILED);
}
