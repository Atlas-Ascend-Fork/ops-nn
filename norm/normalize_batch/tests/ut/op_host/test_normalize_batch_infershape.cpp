/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// NormalizeBatch gert 风格 InferShape UT（op_host/infershape.cpp 的
// IMPL_OP_INFERSHAPE 注册轨，经 gert::OpImplRegistry->infer_shape 触发）。
// 覆盖：输出恒等拷贝、rank=3 门禁、静态负维拒绝、seq_len rank/(n,) 一致性、
// 空 d 维（0）透传。

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace {
constexpr const char* OP_TYPE = "NormalizeBatch";

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

// 返回执行状态；outputDims 出参回传推理后的输出 shape
ge::graphStatus RunInferShape(const std::vector<int64_t>& xDims, const std::vector<int64_t>& seqDims,
                              std::vector<int64_t>& outputDims)
{
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    EXPECT_NE(opImpl, nullptr);
    EXPECT_NE(opImpl->infer_shape, nullptr);
    gert::StorageShape xShape = MakeShape(xDims);
    gert::StorageShape seqShape = MakeShape(seqDims);
    gert::StorageShape yShape;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1}, {1})
                      .InputShapes({&xShape, &seqShape})
                      .OutputShapes({&yShape})
                      .Build();
    const ge::graphStatus status = opImpl->infer_shape(holder.GetContext<gert::InferShapeContext>());
    if (status == ge::GRAPH_SUCCESS) {
        const auto* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
        EXPECT_NE(output, nullptr);
        outputDims.clear();
        for (size_t i = 0; i < output->GetDimNum(); ++i) {
            outputDims.push_back(output->GetDim(i));
        }
    }
    return status;
}
} // namespace

TEST(NormalizeBatchInferShape, CopiesInputShape)
{
    std::vector<int64_t> outputDims;
    ASSERT_EQ(RunInferShape({2, 3, 4}, {2}, outputDims), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputDims, (std::vector<int64_t>{2, 3, 4}));
}

TEST(NormalizeBatchInferShape, RejectsWrongRank)
{
    std::vector<int64_t> outputDims;
    EXPECT_EQ(RunInferShape({2, 3}, {2}, outputDims), ge::GRAPH_FAILED);       // rank=2
    EXPECT_EQ(RunInferShape({2, 3, 4, 5}, {2}, outputDims), ge::GRAPH_FAILED); // rank=4
}

TEST(NormalizeBatchInferShape, RejectsNegativeDim)
{
    // 静态负维（动态 -1/-2 需框架展开后再进 infershape）
    std::vector<int64_t> outputDims;
    EXPECT_EQ(RunInferShape({-1, 3, 4}, {-1}, outputDims), ge::GRAPH_FAILED);
}

TEST(NormalizeBatchInferShape, RejectsSeqLenMismatch)
{
    std::vector<int64_t> outputDims;
    EXPECT_EQ(RunInferShape({2, 3, 4}, {2, 3}, outputDims), ge::GRAPH_FAILED); // seq_len rank=2
    EXPECT_EQ(RunInferShape({2, 3, 4}, {3}, outputDims), ge::GRAPH_FAILED);    // seq_len[0] != n
}

TEST(NormalizeBatchInferShape, AcceptsEmptyLastDim)
{
    std::vector<int64_t> outputDims;
    ASSERT_EQ(RunInferShape({2, 3, 0}, {2}, outputDims), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputDims, (std::vector<int64_t>{2, 3, 0})); // 空 d 维原样透传
}
