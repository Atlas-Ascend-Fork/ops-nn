/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_einsum_infershape.cpp
 * \brief Einsum InferShape UT：equation解析/省略号广播/未知维传播/非法方程拒绝
 */

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

#include "exe_graph/runtime/storage_shape.h"
#include "infer_shape_context_faker.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "op_impl_registry.h"

using namespace ge;

namespace {
// 动态输入x实例数与方程项数一致（Einsum最多2个输入）
struct EinsumCase {
    std::string equation;
    std::vector<std::vector<int64_t>> inputDims;
    std::vector<int64_t> expected;
    bool expectOk;
};

gert::KernelRunContextHolder BuildEinsumInferContext(const std::string& equation,
                                                     const std::vector<std::vector<int64_t>>& inputDims)
{
    std::vector<gert::StorageShape> inputs;
    inputs.reserve(inputDims.size());
    for (const auto& dims : inputDims) {
        gert::StorageShape shape({}, {});
        for (const auto dim : dims) {
            (void)shape.MutableOriginShape().AppendDim(dim);
            (void)shape.MutableStorageShape().AppendDim(dim);
        }
        inputs.push_back(shape);
    }
    std::vector<gert::StorageShape*> inputPtrs;
    for (auto& shape : inputs) {
        inputPtrs.push_back(&shape);
    }
    return gert::InferShapeContextFaker()
        .SetOpType("Einsum")
        .IrInstanceNum({static_cast<uint32_t>(inputDims.size())}, {1})
        .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
        .InputShapes(inputPtrs)
        .NodeAttrs({{"equation", Ops::NN::AnyValue::CreateFrom<std::string>(equation)},
                    {"N", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(inputDims.size()))}})
        .Build();
}

std::string Shape2String(const gert::Shape& shape)
{
    std::string result = "[";
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(shape.GetDim(i));
    }
    result += "]";
    return result;
}

std::string Expected2String(const std::vector<int64_t>& dims)
{
    std::string result = "[";
    for (size_t i = 0; i < dims.size(); i++) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(dims[i]);
    }
    result += "]";
    return result;
}

void RunEinsumCase(const EinsumCase& testCase)
{
    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("Einsum");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto holder = BuildEinsumInferContext(testCase.equation, testCase.inputDims);
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    if (testCase.expectOk) {
        ASSERT_EQ(opImpl->infer_shape(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(Shape2String(*context->GetOutputShape(0)), Expected2String(testCase.expected));
    } else {
        EXPECT_NE(opImpl->infer_shape(context), ge::GRAPH_SUCCESS);
    }
}
} // namespace

class EinsumInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Einsum InferShape Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Einsum InferShape Test TearDown" << std::endl; }
};

TEST_F(EinsumInferShapeTest, matmul_pattern) { RunEinsumCase({"ab,bc->ac", {{2, 3}, {3, 4}}, {2, 4}, true}); }

TEST_F(EinsumInferShapeTest, batch_matmul_pattern)
{
    RunEinsumCase({"abc,cd->abd", {{2, 16, 32}, {32, 48}}, {2, 16, 48}, true});
}

TEST_F(EinsumInferShapeTest, outer_product) { RunEinsumCase({"a,b->ab", {{3}, {4}}, {3, 4}, true}); }

TEST_F(EinsumInferShapeTest, ellipsis_prefix)
{
    RunEinsumCase({"...ab,bc->...ac", {{5, 6, 2, 3}, {3, 4}}, {5, 6, 2, 4}, true});
}

TEST_F(EinsumInferShapeTest, ellipsis_single_input_no_self_broadcast)
{
    // 单输入省略号维为1时不自我广播（对齐canndev原版行为）
    RunEinsumCase({"...ab,bc->...ac", {{1, 2, 3}, {3, 4}}, {1, 2, 4}, true});
}

TEST_F(EinsumInferShapeTest, ellipsis_broadcast_across_inputs)
{
    // 省略号维跨输入广播：1与2广播取2
    RunEinsumCase({"...ab,...ab->...ab", {{5, 1, 2, 3}, {5, 2, 2, 3}}, {5, 2, 2, 3}, true});
}

TEST_F(EinsumInferShapeTest, ellipsis_dim_count_mismatch_rejected)
{
    // 各输入省略号维数不一致（2维vs 1维，且两个ellSize均>0）必须被拒绝：曾因插入后begin()自比对漏判导致越界读
    RunEinsumCase({"...ab,...ab->...ab", {{5, 1, 2, 3}, {7, 2, 3}}, {}, false});
    // 反向插入序（长省略号在后）同样拒绝
    RunEinsumCase({"...ab,...ab->...ab", {{7, 2, 3}, {5, 1, 2, 3}}, {}, false});
    // 省略号匹配0维（ellSize=0）合法，不视为不一致
    RunEinsumCase({"...ab,...ab->...ab", {{5, 1, 2, 3}, {2, 3}}, {5, 1, 2, 3}, true});
    // 评审场景：字典序最小者为较长向量时（[1,2] < [5]），插入前检查必须拒绝（曾因插入后
    // begin()指向新元素自身而漏判，后续按begin()->size()遍历对短向量越界读）
    RunEinsumCase({"a...bc,...de->ad", {{2, 1, 5, 3, 4}, {1, 2, 6, 7}}, {}, false});
    // 同场景反向插入序
    RunEinsumCase({"...de,a...bc->ad", {{1, 2, 6, 7}, {2, 1, 5, 3, 4}}, {}, false});
}

TEST_F(EinsumInferShapeTest, ellipsis_dim_conflict_rejected)
{
    // 省略号维不匹配（3 vs 4，非广播关系）被拒绝
    RunEinsumCase({"...ab,...ab->...ab", {{5, 3, 2, 3}, {5, 4, 2, 3}}, {}, false});
}

TEST_F(EinsumInferShapeTest, unknown_dim_resolved_from_other_input)
{
    // 未知维-1可从另一输入解析出已知值
    RunEinsumCase({"ab,bc->ac", {{2, -1}, {-1, 4}}, {2, 4}, true});
}

TEST_F(EinsumInferShapeTest, unknown_dim_all_unknown)
{
    RunEinsumCase({"ab,bc->ac", {{-1, -1}, {-1, -1}}, {-1, -1}, true});
}

TEST_F(EinsumInferShapeTest, unknown_dim_ellipsis_broadcast)
{
    // 省略号维-1与任何已知维（含1）广播：结果保持-1传播到输出，由运行时校验广播合法性
    RunEinsumCase({"...ab,...ab->...ab", {{5, -1, 2, 3}, {5, 2, 2, 3}}, {5, -1, 2, 3}, true});
    RunEinsumCase({"...ab,...ab->...ab", {{5, -1, 2, 3}, {5, 1, 2, 3}}, {5, -1, 2, 3}, true});
}

TEST_F(EinsumInferShapeTest, unknown_dim_partial)
{
    RunEinsumCase({"abc,cd->abd", {{-1, 16, 32}, {32, 48}}, {-1, 16, 48}, true});
}

TEST_F(EinsumInferShapeTest, duplicated_label_rejected)
{
    RunEinsumCase({"aab,bc->aac", {{2, 2, 3}, {3, 4}}, {}, false});
}

TEST_F(EinsumInferShapeTest, output_new_label_rejected) { RunEinsumCase({"ab,bd->ac", {{2, 3}, {3, 4}}, {}, false}); }

TEST_F(EinsumInferShapeTest, missing_arrow_rejected) { RunEinsumCase({"ab,bc", {{2, 3}, {3, 4}}, {}, false}); }

TEST_F(EinsumInferShapeTest, label_shape_conflict_rejected)
{
    RunEinsumCase({"ab,bc->ac", {{2, 3}, {4, 5}}, {}, false});
}

TEST_F(EinsumInferShapeTest, input_count_mismatch_rejected)
{
    RunEinsumCase({"ab,bc->ac", {{2, 3}, {3, 4}, {3, 4}}, {}, false});
}

TEST_F(EinsumInferShapeTest, whitespace_tolerated)
{
    RunEinsumCase({"a b , b c - > a c", {{2, 3}, {3, 4}}, {2, 4}, true});
}
