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
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"

TEST(IndexByTensorInferDataType, RegisteredGraphCallbackPreservesInputDtype)
{
    // Isolate the graph registry so an installed legacy registration cannot hide a missing callback.
    gert::OpImplSpaceRegistryV2 registry;
    gert::OppSoDesc graphSo({ge::AscendString(INDEX_BY_TENSOR_GRAPH_REGISTRY_SO)}, "index_by_tensor_graph_ut");
    ASSERT_EQ(registry.AddSoToRegistry(graphSo), ge::GRAPH_SUCCESS);
    const auto* impl = registry.GetOpImpl("IndexByTensor");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    for (auto dtype :
         {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32, ge::DT_INT64, ge::DT_BOOL, ge::DT_COMPLEX64}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("IndexByTensor").OpName("index_by_tensor_dtype").IONum(1, 1);
        builder.InputTensorDesc(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
        builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
        auto holder = builder.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
        ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), dtype);
    }
}
