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

#include "../../../framework/fused_weight_quant_mat_mul_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "FusedWeightQuantMatMul"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    if (!attrs.empty()) {
        op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    }
    return op_src;
}
} // namespace

TEST(OnnxFusedWeightQuantMatMulPluginTest, ParseAllAttrs)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"antiquant_group_size","type":7,"i":0},{"name":"dtype","type":7,"i":-1},)"
        R"({"name":"fused_op_type","type":8,"s":"relu"}]})");
    ge::Operator op_dst = CreateOperator("fwqm");
    EXPECT_EQ(domi::ParseParamsFusedWeightQuantMatMul(op_src, op_dst), domi::SUCCESS);

    int64_t group_size = -1;
    int64_t dtype = 0;
    bool transpose_x = true;
    bool transpose_weight = true;
    ge::AscendString fusedOpType;
    EXPECT_EQ(op_dst.GetAttr("antiquant_group_size", group_size), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("dtype", dtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("transpose_x", transpose_x), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("transpose_weight", transpose_weight), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("fused_op_type", fusedOpType), ge::GRAPH_SUCCESS);
    EXPECT_EQ(group_size, 0);
    EXPECT_EQ(dtype, -1);
    EXPECT_FALSE(transpose_x);
    EXPECT_FALSE(transpose_weight);
    EXPECT_STREQ(fusedOpType.GetString(), "relu");
}

TEST(OnnxFusedWeightQuantMatMulPluginTest, KeepsDefaultWithoutAttributes)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":{}})");
    ge::Operator op_dst = CreateOperator("fwqm");
    EXPECT_EQ(domi::ParseParamsFusedWeightQuantMatMul(op_src, op_dst), domi::SUCCESS);

    int64_t group_size = -1;
    int64_t dtype = 0;
    ge::AscendString fusedOpType;
    EXPECT_EQ(op_dst.GetAttr("antiquant_group_size", group_size), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("dtype", dtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("fused_op_type", fusedOpType), ge::GRAPH_SUCCESS);
    EXPECT_EQ(group_size, 0);
    EXPECT_EQ(dtype, -1);
    EXPECT_STREQ(fusedOpType.GetString(), "relu");
}

TEST(OnnxFusedWeightQuantMatMulPluginTest, NoAttributeAttrKeepsDefault)
{
    // onnx 节点无 attribute 属性时（op_src 不携带 "attribute"），全部保持默认值
    ge::Operator op_src = CreateSourceOperator("");
    ge::Operator op_dst = CreateOperator("fwqm");
    EXPECT_EQ(domi::ParseParamsFusedWeightQuantMatMul(op_src, op_dst), domi::SUCCESS);

    ge::AscendString fusedOpType;
    EXPECT_EQ(op_dst.GetAttr("fused_op_type", fusedOpType), ge::GRAPH_SUCCESS);
    EXPECT_STREQ(fusedOpType.GetString(), "relu");
}

TEST(OnnxFusedWeightQuantMatMulPluginTest, MalformedJsonReturnsFailed)
{
    ge::Operator op_src = CreateSourceOperator(R"(not a json)");
    ge::Operator op_dst = CreateOperator("fwqm");
    EXPECT_EQ(domi::ParseParamsFusedWeightQuantMatMul(op_src, op_dst), domi::FAILED);
}

TEST(OnnxFusedWeightQuantMatMulPluginTest, PreservesInt64GroupSize)
{
    ge::Operator opSrc = CreateSourceOperator(
        R"({"attribute":[{"name":"antiquant_group_size","type":7,"i":4294967296}]})");
    ge::Operator opDst = CreateOperator("fwqm");
    ASSERT_EQ(domi::ParseParamsFusedWeightQuantMatMul(opSrc, opDst), domi::SUCCESS);

    int64_t groupSize = 0;
    ASSERT_EQ(opDst.GetAttr("antiquant_group_size", groupSize), ge::GRAPH_SUCCESS);
    EXPECT_EQ(groupSize, 4294967296LL);
}
