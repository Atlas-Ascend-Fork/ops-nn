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
#include <vector>
#include "infershape_test_util.h"
#include "ut_op_common.h"

namespace {
using Dims = std::vector<int64_t>;
struct ShapeCase {
    const char* name;
    Dims x, rot, groups, level0, level1;
    ge::graphStatus status = ge::GRAPH_SUCCESS;
    int64_t block0 = 512, block1 = 32;
};

class RotateDualLevelQuantInferShape : public testing::TestWithParam<ShapeCase> {};

TEST_P(RotateDualLevelQuantInferShape, ThreeOutputShapes)
{
    const auto& c = GetParam();
    const bool grouped = !c.groups.empty();
    auto x = CreateStorageShape(c.x);
    auto rot = CreateStorageShape(c.rot);
    auto groups = CreateStorageShape(c.groups);
    gert::StorageShape y, level0, level1;
    auto faker = gert::InferShapeContextFaker();
    faker.IrInstanceNum({1, 1, 0, grouped ? 1U : 0U}, {1, 1, 1});
    if (grouped) {
        faker.InputShapes({&x, &rot, &groups});
        faker.NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    } else {
        faker.InputShapes({&x, &rot});
    }
    auto holder = faker.OutputShapes({&y, &level0, &level1})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT4_E2M1)},
                                  {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                  {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0)},
                                  {"trans", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"level0_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(c.block0)},
                                  {"level1_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(c.block1)}})
                      .Build();
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateDualLevelQuant");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(impl->infer_shape(context), c.status);
    if (c.status != ge::GRAPH_SUCCESS) {
        return;
    }
    const std::vector<Dims> expected{c.x, c.level0, c.level1};
    for (size_t output = 0; output < expected.size(); ++output) {
        const auto* shape = context->GetOutputShape(output);
        ASSERT_NE(shape, nullptr);
        ASSERT_EQ(shape->GetDimNum(), expected[output].size());
        for (size_t dim = 0; dim < expected[output].size(); ++dim) {
            EXPECT_EQ(shape->GetDim(dim), expected[output][dim]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    A5, RotateDualLevelQuantInferShape,
    testing::Values(ShapeCase{"rank1", {512}, {128, 128}, {}, {1}, {8, 2}},
                    ShapeCase{"tail512", {4, 576}, {64, 64}, {}, {4, 2}, {4, 9, 2}},
                    ShapeCase{"odd_mx_padding", {4, 544}, {17, 32, 32}, {}, {4, 2}, {4, 9, 2}},
                    ShapeCase{"rank3", {2, 3, 1024}, {256, 256}, {}, {2, 3, 2}, {2, 3, 16, 2}},
                    ShapeCase{
                        "rank7", {1, 2, 1, 1, 1, 3, 32}, {32, 32}, {}, {1, 2, 1, 1, 1, 3, 1}, {1, 2, 1, 1, 1, 3, 1, 2}},
                    ShapeCase{"empty_m", {0, 544}, {32, 32}, {}, {0, 2}, {0, 9, 2}},
                    ShapeCase{"empty_n", {4, 0}, {0, 64, 64}, {}, {4, 0}, {4, 0, 2}},
                    ShapeCase{"dynamic_n", {4, -1}, {64, 64}, {}, {4, -1}, {4, -1, 2}},
                    ShapeCase{"unknown_rank", {-2}, {64, 64}, {}, {-2}, {-2}},
                    ShapeCase{"grouped", {7, 544}, {3, 32, 32}, {3}, {7, 2}, {7, 9, 2}},
                    ShapeCase{"grouped_per_block", {7, 576}, {3, 9, 64, 64}, {3}, {7, 2}, {7, 9, 2}},
                    ShapeCase{"bad_k", {4, 512}, {16, 16}, {}, {}, {}, ge::GRAPH_FAILED},
                    ShapeCase{"bad_n", {4, 544}, {64, 64}, {}, {}, {}, ge::GRAPH_FAILED},
                    ShapeCase{"bad_group_count", {7, 544}, {3, 32, 32}, {4}, {}, {}, ge::GRAPH_FAILED},
                    ShapeCase{"bad_group_rank", {2, 7, 544}, {3, 32, 32}, {3}, {}, {}, ge::GRAPH_FAILED},
                    ShapeCase{"bad_block0", {4, 512}, {32, 32}, {}, {}, {}, ge::GRAPH_FAILED, 256, 32},
                    ShapeCase{"bad_block1", {4, 512}, {32, 32}, {}, {}, {}, ge::GRAPH_FAILED, 512, 64}),
    [](const testing::TestParamInfo<ShapeCase>& info) { return info.param.name; });

TEST(RotateDualLevelQuantInferDataType, ThreeDtypesAndUnsupportedTarget)
{
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateDualLevelQuant");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    for (auto target : {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_INT8}) {
        ge::DataType x = ge::DT_BF16, rot = ge::DT_BF16;
        ge::DataType y = ge::DT_UNDEFINED, s0 = ge::DT_UNDEFINED, s1 = ge::DT_UNDEFINED;
        auto holder = gert::InferDataTypeContextFaker()
                          .IrInputNum(2)
                          .NodeIoNum(2, 3)
                          .IrInstanceNum({1, 1})
                          .NodeInputTd(0, x, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, rot, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, target, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(target)}})
                          .InputDataTypes({&x, &rot})
                          .OutputDataTypes({&y, &s0, &s1})
                          .Build();
        auto* context = holder.GetContext<gert::InferDataTypeContext>();
        const auto expected = target == ge::DT_INT8 ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
        ASSERT_EQ(impl->infer_datatype(context), expected);
        if (expected == ge::GRAPH_SUCCESS) {
            EXPECT_EQ(context->GetOutputDataType(0), target);
            EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
            EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT8_E8M0);
        }
    }
}
} // namespace
