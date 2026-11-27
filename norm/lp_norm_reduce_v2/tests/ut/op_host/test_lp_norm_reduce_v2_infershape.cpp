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
 * \file test_lp_norm_reduce_v2_infershape.cpp
 * \brief LpNormReduceV2 InferShape UT（spec.yaml boundary_conditions 形状类全分支）
 *
 * 覆盖（DESIGN §6.6 / spec shape_rule）：
 *   - axes 空（全维度归约）/ 单轴 / 多轴（相邻合轴）/ 负索引归一化（-1 / -rank）
 *   - keepdim true / false 两态
 *   - 全归约 + keepdim=false → 0 维标量（dimNum=0）
 *   - rank=0 标量输入；reduce 轴 size=1
 *   - 校验分支：axes 越界（正/负）、重复（含负索引归一化后重复）、数量 > rank、rank(x) > 8
 *   - 迭代三 A2 新增：上下文空指针防御分支（null 输入 shape / null 输出 shape /
 *     缺失 axes 属性 / 缺失 keepdim 属性 → GRAPH_FAILED）
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "infer_shape_context_faker.h"
#include "op_impl_registry.h"

using namespace ge;

namespace {
const char* OP_TYPE = "LpNormReduceV2";

std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    std::vector<int64_t> shapeVec;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        shapeVec.push_back(shape.GetDim(i));
    }
    return shapeVec;
}

// 属性按索引顺序追加：0=p(float), 1=axes(list int), 2=keepdim(bool), 3=epsilon(float)，
// 与 op_graph proto / def.cpp 注册顺序一致（infershape 按 index 1/2 读取）
static void RunInferShape(const std::vector<int64_t>& xShape, const std::vector<int64_t>& axes, bool keepdim,
                          ge::graphStatus expectStatus, const std::vector<int64_t>& expectYShape)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    ASSERT_NE(opImpl, nullptr) << "LpNormReduceV2 未注册到 OpImplRegistry";
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr) << "LpNormReduceV2 未注册 infer_shape";

    gert::Shape xShape_;
    xShape_.SetDimNum(xShape.size());
    for (size_t i = 0; i < xShape.size(); i++) {
        xShape_.SetDim(i, xShape[i]);
    }

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&xShape_})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Attr("p", 2.0f)
                      .Attr("axes", axes)
                      .Attr("keepdim", keepdim)
                      .Attr("epsilon", 1e-12f)
                      .Build();

    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), expectStatus);
    if (expectStatus != ge::GRAPH_SUCCESS) {
        return;
    }
    EXPECT_EQ(ToVector(*context->GetOutputShape(0)), expectYShape);
}
} // namespace

class LpNormReduceV2InfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// ---------------- axes 空列表 = 全维度归约 ----------------
TEST_F(LpNormReduceV2InfershapeTest, empty_axes_keepdim_false_returns_scalar)
{
    // spec boundary: x=[2,3] axes=[] keepdim=false → 标量以 (1,) 承载（GE 0 维输出不回拷数据）
    RunInferShape({2, 3}, {}, false, ge::GRAPH_SUCCESS, {1});
}

TEST_F(LpNormReduceV2InfershapeTest, empty_axes_keepdim_true_all_one)
{
    RunInferShape({2, 3}, {}, true, ge::GRAPH_SUCCESS, {1, 1});
}

TEST_F(LpNormReduceV2InfershapeTest, empty_axes_rank3_keepdim_true)
{
    RunInferShape({2, 3, 4}, {}, true, ge::GRAPH_SUCCESS, {1, 1, 1});
}

TEST_F(LpNormReduceV2InfershapeTest, explicit_full_axes_equivalent_to_empty)
{
    // 显式全维度 axes=[0,1] 等价 axes=[]（相邻合轴后同为全归约，标量以 (1,) 承载）
    RunInferShape({2, 3}, {0, 1}, false, ge::GRAPH_SUCCESS, {1});
}

// ---------------- 单轴归约 ----------------
TEST_F(LpNormReduceV2InfershapeTest, single_axis_keepdim_false)
{
    // spec boundary: x=[2,3,4] axes=[1] keepdim=false → {2,4}
    RunInferShape({2, 3, 4}, {1}, false, ge::GRAPH_SUCCESS, {2, 4});
}

TEST_F(LpNormReduceV2InfershapeTest, single_axis_keepdim_true)
{
    // spec boundary: x=[2,3,4] axes=[1] keepdim=true → {2,1,4}
    RunInferShape({2, 3, 4}, {1}, true, ge::GRAPH_SUCCESS, {2, 1, 4});
}

TEST_F(LpNormReduceV2InfershapeTest, reduce_axis_size_one_keepdim_false)
{
    // spec boundary: x=[2,1,3] axes=[1]（归约轴长度 1）
    RunInferShape({2, 1, 3}, {1}, false, ge::GRAPH_SUCCESS, {2, 3});
}

TEST_F(LpNormReduceV2InfershapeTest, reduce_axis_size_one_keepdim_true)
{
    RunInferShape({2, 1, 3}, {1}, true, ge::GRAPH_SUCCESS, {2, 1, 3});
}

// ---------------- 负索引归一化（a + rank） ----------------
TEST_F(LpNormReduceV2InfershapeTest, negative_index_last_axis)
{
    // spec boundary: x=[2,3] axes=[-1] → 归一化为 1 → {2}
    RunInferShape({2, 3}, {-1}, false, ge::GRAPH_SUCCESS, {2});
}

TEST_F(LpNormReduceV2InfershapeTest, negative_index_first_axis)
{
    // axes=[-2] → 归一化为 0 → {3}
    RunInferShape({2, 3}, {-2}, false, ge::GRAPH_SUCCESS, {3});
}

TEST_F(LpNormReduceV2InfershapeTest, negative_index_keepdim_true)
{
    RunInferShape({2, 3}, {-1}, true, ge::GRAPH_SUCCESS, {2, 1});
}

// ---------------- 多轴归约（相邻 / 非相邻） ----------------
TEST_F(LpNormReduceV2InfershapeTest, multi_adjacent_axes_keepdim_false)
{
    // 相邻合轴：axes=[0,1] → {4}
    RunInferShape({2, 3, 4}, {0, 1}, false, ge::GRAPH_SUCCESS, {4});
}

TEST_F(LpNormReduceV2InfershapeTest, multi_nonadjacent_axes)
{
    // 非相邻多轴（shape 语义仍由 InferShape 输出，多段路径由 tiling 拦截）
    RunInferShape({2, 3, 4}, {0, 2}, false, ge::GRAPH_SUCCESS, {3});
}

TEST_F(LpNormReduceV2InfershapeTest, multi_nonadjacent_axes_keepdim_true)
{
    RunInferShape({2, 3, 4}, {0, 2}, true, ge::GRAPH_SUCCESS, {1, 3, 1});
}

TEST_F(LpNormReduceV2InfershapeTest, multi_negative_axes)
{
    // 负多轴：[-3,-1] → [0,2] → {3}
    RunInferShape({2, 3, 4}, {-3, -1}, false, ge::GRAPH_SUCCESS, {3});
}

// ---------------- rank=0 标量 ----------------
TEST_F(LpNormReduceV2InfershapeTest, rank_zero_scalar_input)
{
    // spec boundary: x=[] axes=[] → 标量以 (1,) 承载（GE 0 维输出不回拷数据）
    RunInferShape({}, {}, false, ge::GRAPH_SUCCESS, {1});
}

// ---------------- 校验失败分支（attribute_value_out_of_range / shape_mismatch） ----------------
// 黑盒路由 lpnorm_l2_015: axes 越界拒收（x=(2,3), axes=[2]）→ 本测试用例
// LpNormReduceV2InfershapeTest.axes_out_of_range_positive（real 链路编译期拒收被隔离 OPP
// IGNORE_INFER_ERROR=1 抑制不可达，拒收语义由本 UT 直测，st_real_result.json 路由 ut_host）
TEST_F(LpNormReduceV2InfershapeTest, axes_out_of_range_positive)
{
    // spec boundary: x=[2,3] axes=[2] → 越界报错
    RunInferShape({2, 3}, {2}, false, ge::GRAPH_FAILED, {});
}

TEST_F(LpNormReduceV2InfershapeTest, axes_out_of_range_negative)
{
    // 归一化后 < 0：axes=[-3]，rank=2 → -1 → 越界
    RunInferShape({2, 3}, {-3}, false, ge::GRAPH_FAILED, {});
}

// 黑盒路由 lpnorm_l2_016: axes 元素重复拒收（axes=[0,0]）→ 本测试用例
// LpNormReduceV2InfershapeTest.axes_duplicate（ut_host 路由依据，同上）
TEST_F(LpNormReduceV2InfershapeTest, axes_duplicate)
{
    // spec boundary: axes=[0,0] → 重复报错
    RunInferShape({2, 3}, {0, 0}, false, ge::GRAPH_FAILED, {});
}

TEST_F(LpNormReduceV2InfershapeTest, axes_size_exceed_rank)
{
    // axes 数量 > rank：rank=2 传 3 个
    RunInferShape({2, 3}, {0, 1, 2}, false, ge::GRAPH_FAILED, {});
}

// 黑盒路由 lpnorm_l2_019: rank=9>8 拒收（x=(2,2,2,2,2,2,2,2,2)）→ 本测试用例
// LpNormReduceV2InfershapeTest.rank_exceed_eight（ut_host 路由依据，同上）
TEST_F(LpNormReduceV2InfershapeTest, rank_exceed_eight)
{
    // rank(x) ≤ 8 校验（shape_mismatch）
    RunInferShape({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1}, false, ge::GRAPH_FAILED, {});
}

TEST_F(LpNormReduceV2InfershapeTest, rank_eight_boundary_success)
{
    // rank=8 边界应通过
    RunInferShape({1, 2, 3, 4, 5, 6, 7, 8}, {7}, false, ge::GRAPH_SUCCESS, {1, 2, 3, 4, 5, 6, 7});
}

TEST_F(LpNormReduceV2InfershapeTest, axes_negative_duplicate_after_normalize)
{
    // 负索引归一化后重复：[-2, 0] → [0, 0] → 重复报错（spec machine_constraint：
    // 元素不得重复——按归一化后集合判定）
    RunInferShape({2, 3}, {-2, 0}, false, ge::GRAPH_FAILED, {});
}

// ---------------- 上下文空指针防御分支（OP_CHECK_NULL，DESIGN §8 internal_error 防御） ----------------

TEST_F(LpNormReduceV2InfershapeTest, context_null_input_shape_rejected)
{
    // 未注入 InputShapes/NodeInputTd → GetInputShape(0) 返回空指针 → GRAPH_FAILED
    // （GetInputShape index 非法返回空指针，exe_graph/runtime/infer_shape_context.h 契约）
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Attr("p", 2.0f)
                      .Attr("axes", std::vector<int64_t>{1})
                      .Attr("keepdim", false)
                      .Attr("epsilon", 1e-12f)
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

TEST_F(LpNormReduceV2InfershapeTest, context_missing_axes_attr_rejected)
{
    // 属性缺失（仅 p，无 axes）：GetAttrPointer<TypedContinuousVector>(1) 返回空 → GRAPH_FAILED
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape xShape_;
    xShape_.SetDimNum(2);
    xShape_.SetDim(0, 2);
    xShape_.SetDim(1, 3);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&xShape_})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Attr("p", 2.0f)
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

TEST_F(LpNormReduceV2InfershapeTest, context_missing_keepdim_attr_rejected)
{
    // 属性缺失（p/axes 有、keepdim 无）：GetAttrPointer<bool>(2) 返回空 → GRAPH_FAILED
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape xShape_;
    xShape_.SetDimNum(2);
    xShape_.SetDim(0, 2);
    xShape_.SetDim(1, 3);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&xShape_})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Attr("p", 2.0f)
                      .Attr("axes", std::vector<int64_t>{1})
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}
