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
#include <array>
#include <string>
#include <vector>
#include "ut_op_util.h"
#include "infer_shape_context_faker.h"

namespace {
struct QuantFailure {
    const char* name;
    const char* mode;
    std::array<uint32_t, 4> optional;
    bool invalidRank;
};

class AddLayerNormQuantReporting : public testing::TestWithParam<QuantFailure> {};

TEST_P(AddLayerNormQuantReporting, RejectsInvalidInput)
{
    const auto& param = GetParam();
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    gert::StorageShape x = {{2, 8}, {2, 8}};
    gert::StorageShape gamma = {{8}, {8}};
    if (param.invalidRank) {
        gamma = gert::StorageShape({{1, 2, 8}, {1, 2, 8}});
    }
    gert::StorageShape y, y2, sum, scale, scale2;
    std::vector<gert::StorageShape*> inputs = {&x, &x, &gamma, &gamma, &gamma};
    std::vector<uint32_t> instances = {1, 1, 1, 1, 1};
    for (auto present : param.optional) {
        instances.push_back(present);
        if (present != 0) {
            inputs.push_back(&gamma);
        }
    }
    auto faker = gert::InferShapeContextFaker();
    faker.SetOpType("AddLayerNormQuant")
        .NodeIoNum(9, 5)
        .IrInstanceNum(instances, {1, 1, 1, 1, 1})
        .InputShapes(inputs)
        .OutputShapes({&y, &y2, &sum, &scale, &scale2});
    for (size_t i = 0; i < inputs.size(); ++i) {
        faker.NodeInputTd(i, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int i = 0; i < 5; ++i) {
        faker.NodeOutputTd(i, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (param.mode != nullptr) {
        faker.NodeAttrs({{"quant_mode", Ops::NN::AnyValue::CreateFrom<std::string>(param.mode)},
                         {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01f)},
                         {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}});
    } else {
        faker.NodeAttrs({});
    }
    auto holder = faker.Build();
    EXPECT_EQ(impl->infer_shape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

INSTANTIATE_TEST_SUITE_P(ErrorPaths, AddLayerNormQuantReporting,
                         testing::Values(QuantFailure{"ZeroPoint1", "dynamic", {0, 0, 1, 0}, false},
                                         QuantFailure{"ZeroPoint2", "dynamic", {0, 0, 0, 1}, false},
                                         QuantFailure{"Scale2Only", "dynamic", {0, 1, 0, 0}, false},
                                         QuantFailure{"ReductionRank", "dynamic", {0, 0, 0, 0}, true},
                                         QuantFailure{"StaticScale1", "static", {0, 0, 0, 0}, false},
                                         QuantFailure{"MissingQuantMode", nullptr, {0, 0, 0, 0}, false}),
                         [](const testing::TestParamInfo<QuantFailure>& info) { return info.param.name; });

} // namespace
