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
#include "infershape_case_executor.h"
#include "infershape_test_util.h"

namespace {
void CheckShape(const std::vector<int64_t>& x, const std::vector<std::vector<int64_t>>& indices,
                const std::vector<int64_t>& mask, const std::vector<int64_t>& expected,
                ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexByTensor");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    using Desc = gert::InfershapeContextPara::TensorDescription;
    std::vector<Desc> inputs{{CreateStorageShape(x), ge::DT_FLOAT, ge::FORMAT_ND}};
    for (const auto& shape : indices) {
        inputs.emplace_back(CreateStorageShape(shape), ge::DT_INT64, ge::FORMAT_ND);
    }
    std::vector<gert::InfershapeContextPara::OpAttr> attrs{
        {"indices_mask", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(mask)}};
    gert::InfershapeContextPara para("IndexByTensor", inputs,
                                     {Desc(CreateStorageShape({}), ge::DT_FLOAT, ge::FORMAT_ND)}, attrs,
                                     {1, static_cast<uint32_t>(indices.size())}, {1});
    ExecuteTestCase(para, status,
                    status == ge::GRAPH_SUCCESS ? std::vector<std::vector<int64_t>>{expected} :
                                                  std::vector<std::vector<int64_t>>{});
}
} // namespace

TEST(IndexByTensorInferShape, RegisteredInNativeHost)
{
    // This registry contains the native host library, without legacy inference.
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("IndexByTensor");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
}

TEST(IndexByTensorInferShape, ConsecutiveAxesWithImplicitTrailingAxis)
{
    CheckShape({2, 3, 4, 5}, {{3, 2}, {3, 2}}, {0, 1, 1}, {2, 3, 2, 5});
}

TEST(IndexByTensorInferShape, SeparatedAxesWithBroadcast)
{
    CheckShape({3, 4, 5}, {{2, 1}, {1, 3}}, {1, 0, 1}, {2, 3, 4});
}

TEST(IndexByTensorInferShape, UnknownRankAndUnknownDimension)
{
    CheckShape({-2}, {{2}}, {1}, {-2});
    CheckShape({3, 4}, {{-2}}, {1}, {-2});
    CheckShape({3, 4}, {{-1}}, {1}, {-1, 4});
}

TEST(IndexByTensorInferShape, NoIndicesPreservesShape)
{
    CheckShape({2, 3}, {}, {}, {2, 3});
    CheckShape({}, {}, {}, {});
}

TEST(IndexByTensorInferShape, ScalarIndex) { CheckShape({3, 4}, {{}}, {1}, {4}); }

TEST(IndexByTensorInferShape, ScalarAndVectorOnSeparatedAxes) { CheckShape({3, 4, 5}, {{}, {2}}, {1, 0, 1}, {2, 4}); }

TEST(IndexByTensorInferShape, OutputRankAtLimit)
{
    std::vector<int64_t> indexShape(gert::Shape::kMaxDimNum - 1, 1);
    auto expected = indexShape;
    expected.push_back(3);
    CheckShape({-1, 3}, {indexShape}, {1}, expected);
}

TEST(IndexByTensorInferShape, RejectsOutputRankAboveLimit)
{
    std::vector<int64_t> indexShape(gert::Shape::kMaxDimNum, 1);
    CheckShape({-1, 3}, {indexShape}, {1}, {}, ge::GRAPH_FAILED);
    CheckShape({2, 3, 4}, {indexShape, {}}, {1, 0, 1}, {}, ge::GRAPH_FAILED);
}

TEST(IndexByTensorInferShape, RejectsMismatchedMask) { CheckShape({3, 4}, {{2}}, {1, 1}, {}, ge::GRAPH_FAILED); }

TEST(IndexByTensorInferShape, RejectsIncompatibleIndices)
{
    CheckShape({3, 4}, {{2}, {3}}, {1, 1}, {}, ge::GRAPH_FAILED);
}
