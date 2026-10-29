/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "cube_utils/cube_fp16_t.h"
#include "../../../op_graph/fusion_pass/gelu_v2_onnx_fusion_pass.h"
#include "register/register_custom_pass.h"

// UT stub for rtGetSocSpec: the pass gates on NpuArch via this API and the UT must control the
// arch per test case. The weak definition is effective because the UT executable does not link
// the runtime lib, feeding the arch from the ASCEND_NPU_ARCH env variable (same convention as
// the opbase runtime_stub.cpp). It lives at global scope so the extern "C" definition keeps
// external linkage and satisfies the pass TU's reference.
extern "C" {
typedef int32_t rtError_t;
static const int32_t RT_ERROR_NONE = 0;
rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
}

__attribute__((weak)) rtError_t rtGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen)
{
    const char* arch = std::getenv("ASCEND_NPU_ARCH");
    if (arch == nullptr || std::strcmp(label, "version") != 0 || std::strcmp(key, "NpuArch") != 0) {
        return static_cast<rtError_t>(1);
    }
    (void)std::snprintf(val, maxLen, "%s", arch);
    return RT_ERROR_NONE;
}

namespace {
// erf: 3 topologies times 8 input-order variants = 24; tanh: full formula with kappa/beta/one
// const positions and the outer mul orders = 32 (the 0.5 const keeps the first-binding family),
// see Patterns()
constexpr size_t kExpectedPatternCount = 56;
constexpr int32_t kArch950 = 3510;
constexpr int32_t kArch910_93 = 2201; // real NpuArch of the Ascend910_93 family (platform_config ini)

struct GeluGraphParam {
    bool useTanh = false;          // full tanh formula or erf
    bool useTrueDiv = false;       // erf only: div = Mul(inputX, cDiv) or RealDiv(inputX, cDiv)
    float divConst = 1.41421F;     // erf div const
    float kappaConst = 0.044715F;  // tanh kappa cubic coefficient
    float betaConst = 0.79788456F; // tanh beta factor, sqrt(2 / pi)
    bool kappaConstFirst = true;   // tanh: Mul(kappa, cubeMul) or Mul(cubeMul, kappa)
    bool betaConstFirst = true;    // tanh: Mul(beta, innerAdd) or Mul(innerAdd, beta)
    bool fp16Const = false;        // const dtype fp16 or fp32
    bool mul1ConstFamily = false;  // 0.5 const on mul1 (family 2) or mul0 (family 1/3)
    bool addConstFirst = false;    // Add(const, activation) or Add(activation, const)
    bool mul0First = false;        // family 1/3: Mul(const, inputX); family 2: Mul(add, inputX)
    bool mul1First = false;        // family 1/3: Mul(add, mul0); family 2: Mul(const, mul0)
    ge::DataType xDtype = ge::DT_FLOAT;
    std::vector<int64_t> dims = {1, 512, 3072};
};

// convert a float value to fp16 bits with the shared cube_utils fp16_t
uint16_t FloatToFp16Bits(const float value)
{
    ops::fp16_t fp16Value = value;
    return fp16Value.val;
}

void SetOutDesc(const ge::es::EsTensorHolder& holder, ge::DataType dtype, const std::vector<int64_t>& dims)
{
    ge::TensorDesc desc;
    holder.GetProducer()->GetOutputDesc(0, desc);
    desc.SetDataType(dtype);
    desc.SetShape(ge::Shape(dims));
    holder.GetProducer()->UpdateOutputDesc(0, desc);
}

void SetInDesc(ge::GNode& node, int32_t idx, ge::DataType dtype, const std::vector<int64_t>& dims)
{
    ge::TensorDesc desc;
    node.GetInputDesc(idx, desc);
    desc.SetDataType(dtype);
    desc.SetShape(ge::Shape(dims));
    node.UpdateInputDesc(idx, desc);
}

ge::es::EsTensorHolder MakeScalarConst(ge::es::EsGraphBuilder& builder, const float value, const bool fp16)
{
    if (fp16) {
        return builder.CreateConst(std::vector<uint16_t>{FloatToFp16Bits(value)}, std::vector<int64_t>{1},
                                   ge::DT_FLOAT16);
    }
    return builder.CreateConst(std::vector<float>{value}, std::vector<int64_t>{1});
}

// RealDiv/Mul/Erf/Tanh have no ES API here, build source nodes with ge::es::CompliantNodeBuilder
ge::es::EsTensorHolder BuildBinaryNode(ge::es::EsGraphBuilder& builder, const ge::es::EsTensorHolder& in0,
                                       const ge::es::EsTensorHolder& in1, const char* opType)
{
    auto* cBuilder = builder.GetCGraphBuilder();
    auto* graph = cBuilder->GetGraph();
    ge::GNode node = ge::es::CompliantNodeBuilder(graph)
                         .OpType(opType)
                         .Name(cBuilder->GenerateNodeName(opType).GetString())
                         .IrDefInputs({{"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *in0.GetProducer(), in0.GetProducerOutIndex(), node, 0);
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *in1.GetProducer(), in1.GetProducerOutIndex(), node, 1);
    return ge::es::EsTensorHolder(cBuilder->GetTensorHolderFromNode(node, 0));
}

ge::es::EsTensorHolder BuildUnaryNode(ge::es::EsGraphBuilder& builder, const ge::es::EsTensorHolder& in0,
                                      const char* opType)
{
    auto* cBuilder = builder.GetCGraphBuilder();
    auto* graph = cBuilder->GetGraph();
    ge::GNode node = ge::es::CompliantNodeBuilder(graph)
                         .OpType(opType)
                         .Name(cBuilder->GenerateNodeName(opType).GetString())
                         .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *in0.GetProducer(), in0.GetProducerOutIndex(), node, 0);
    return ge::es::EsTensorHolder(cBuilder->GetTensorHolderFromNode(node, 0));
}

void SetElementwiseDescs(const ge::es::EsTensorHolder& node, ge::DataType dtype, const std::vector<int64_t>& dims,
                         const ge::DataType constDtype, const bool secondInputIsConst)
{
    SetInDesc(*node.GetProducer(), 0, dtype, dims);
    SetInDesc(*node.GetProducer(), 1, secondInputIsConst ? constDtype : dtype,
              secondInputIsConst ? std::vector<int64_t>{1} : dims);
    SetOutDesc(node, dtype, dims);
}

void SetUnaryDescs(const ge::es::EsTensorHolder& node, ge::DataType dtype, const std::vector<int64_t>& dims)
{
    SetInDesc(*node.GetProducer(), 0, dtype, dims);
    SetOutDesc(node, dtype, dims);
}

// build the 5-node erf-shaped source graph, all tensor descs are set manually for the pattern
// matcher; actOpType allows reusing the shape for the legacy simplified tanh graph in
// rejection tests
std::shared_ptr<ge::Graph> BuildErfShapedGraph(const GeluGraphParam& param, const char* actOpType)
{
    const ge::DataType constDtype = param.fp16Const ? ge::DT_FLOAT16 : ge::DT_FLOAT;
    ge::es::EsGraphBuilder builder("gelu_v2_fusion_test");
    auto inputX = builder.CreateInput(0, "x", param.xDtype, ge::FORMAT_ND, param.dims);
    SetOutDesc(inputX, param.xDtype, param.dims);

    auto divConst = MakeScalarConst(builder, param.divConst, param.fp16Const);
    auto addConst = MakeScalarConst(builder, 1.0F, param.fp16Const);
    auto mulConst = MakeScalarConst(builder, 0.5F, param.fp16Const);

    const char* divOpType = param.useTrueDiv ? "Mul" : "RealDiv";
    auto div = BuildBinaryNode(builder, inputX, divConst, divOpType);
    auto activation = BuildUnaryNode(builder, div, actOpType);
    auto add = param.addConstFirst ? BuildBinaryNode(builder, addConst, activation, "Add") :
                                     BuildBinaryNode(builder, activation, addConst, "Add");
    ge::es::EsTensorHolder mul0;
    ge::es::EsTensorHolder mul1;
    if (param.mul1ConstFamily) {
        mul0 = param.mul0First ? BuildBinaryNode(builder, add, inputX, "Mul") :
                                 BuildBinaryNode(builder, inputX, add, "Mul");
        mul1 = param.mul1First ? BuildBinaryNode(builder, mulConst, mul0, "Mul") :
                                 BuildBinaryNode(builder, mul0, mulConst, "Mul");
    } else {
        mul0 = param.mul0First ? BuildBinaryNode(builder, mulConst, inputX, "Mul") :
                                 BuildBinaryNode(builder, inputX, mulConst, "Mul");
        mul1 = param.mul1First ? BuildBinaryNode(builder, add, mul0, "Mul") :
                                 BuildBinaryNode(builder, mul0, add, "Mul");
    }

    // set descs manually, the op_graph UT environment does not invoke infershape
    SetElementwiseDescs(div, param.xDtype, param.dims, constDtype, true);
    SetUnaryDescs(activation, param.xDtype, param.dims);
    SetElementwiseDescs(add, param.xDtype, param.dims, constDtype, true);
    if (param.mul1ConstFamily) {
        SetElementwiseDescs(mul0, param.xDtype, param.dims, constDtype, false);
        SetElementwiseDescs(mul1, param.xDtype, param.dims, constDtype, true);
    } else {
        SetElementwiseDescs(mul0, param.xDtype, param.dims, constDtype, true);
        SetElementwiseDescs(mul1, param.xDtype, param.dims, constDtype, false);
    }
    return builder.BuildAndReset({mul1});
}

// build the full tanh gelu source graph: 0.5*inputX*(1+tanh(beta*(inputX+kappa*inputX^3)))
std::shared_ptr<ge::Graph> BuildGeluTanhFullGraph(const GeluGraphParam& param)
{
    const ge::DataType constDtype = param.fp16Const ? ge::DT_FLOAT16 : ge::DT_FLOAT;
    ge::es::EsGraphBuilder builder("gelu_v2_tanh_full_test");
    auto inputX = builder.CreateInput(0, "x", param.xDtype, ge::FORMAT_ND, param.dims);
    SetOutDesc(inputX, param.xDtype, param.dims);

    auto kappaConst = MakeScalarConst(builder, param.kappaConst, param.fp16Const);
    auto betaConst = MakeScalarConst(builder, param.betaConst, param.fp16Const);
    auto addConst = MakeScalarConst(builder, 1.0F, param.fp16Const);
    auto mulConst = MakeScalarConst(builder, 0.5F, param.fp16Const);

    auto squareMul = BuildBinaryNode(builder, inputX, inputX, "Mul");
    auto cubeMul = BuildBinaryNode(builder, squareMul, inputX, "Mul");
    auto kappaMul = param.kappaConstFirst ? BuildBinaryNode(builder, kappaConst, cubeMul, "Mul") :
                                            BuildBinaryNode(builder, cubeMul, kappaConst, "Mul");
    auto innerAdd = BuildBinaryNode(builder, inputX, kappaMul, "Add");
    auto betaMul = param.betaConstFirst ? BuildBinaryNode(builder, betaConst, innerAdd, "Mul") :
                                          BuildBinaryNode(builder, innerAdd, betaConst, "Mul");
    auto activation = BuildUnaryNode(builder, betaMul, "Tanh");
    auto add = param.addConstFirst ? BuildBinaryNode(builder, addConst, activation, "Add") :
                                     BuildBinaryNode(builder, activation, addConst, "Add");
    ge::es::EsTensorHolder mul0;
    ge::es::EsTensorHolder mul1;
    if (param.mul1ConstFamily) {
        mul0 = param.mul0First ? BuildBinaryNode(builder, add, inputX, "Mul") :
                                 BuildBinaryNode(builder, inputX, add, "Mul");
        mul1 = param.mul1First ? BuildBinaryNode(builder, mulConst, mul0, "Mul") :
                                 BuildBinaryNode(builder, mul0, mulConst, "Mul");
    } else {
        mul0 = param.mul0First ? BuildBinaryNode(builder, mulConst, inputX, "Mul") :
                                 BuildBinaryNode(builder, inputX, mulConst, "Mul");
        mul1 = param.mul1First ? BuildBinaryNode(builder, add, mul0, "Mul") :
                                 BuildBinaryNode(builder, mul0, add, "Mul");
    }

    SetElementwiseDescs(squareMul, param.xDtype, param.dims, constDtype, false);
    SetElementwiseDescs(cubeMul, param.xDtype, param.dims, constDtype, false);
    SetElementwiseDescs(kappaMul, param.xDtype, param.dims, constDtype, true);
    SetElementwiseDescs(innerAdd, param.xDtype, param.dims, constDtype, false);
    SetElementwiseDescs(betaMul, param.xDtype, param.dims, constDtype, true);
    SetUnaryDescs(activation, param.xDtype, param.dims);
    SetElementwiseDescs(add, param.xDtype, param.dims, constDtype, true);
    if (param.mul1ConstFamily) {
        SetElementwiseDescs(mul0, param.xDtype, param.dims, constDtype, false);
        SetElementwiseDescs(mul1, param.xDtype, param.dims, constDtype, true);
    } else {
        SetElementwiseDescs(mul0, param.xDtype, param.dims, constDtype, true);
        SetElementwiseDescs(mul1, param.xDtype, param.dims, constDtype, false);
    }
    return builder.BuildAndReset({mul1});
}

// the legacy simplified tanh formula 0.5*inputX*(1+tanh(inputX*sqrt(2/pi))) without the cubic term:
// no longer matched by any pattern, kept to lock the deletion decision
std::shared_ptr<ge::Graph> BuildLegacySimplifiedTanhGraph()
{
    GeluGraphParam param;
    param.useTrueDiv = true;
    param.divConst = 0.79788456F; // sqrt(2 / pi)
    return BuildErfShapedGraph(param, "Tanh");
}

std::shared_ptr<ge::Graph> BuildGeluGraph(const GeluGraphParam& param)
{
    if (param.useTanh) {
        return BuildGeluTanhFullGraph(param);
    }
    return BuildErfShapedGraph(param, "Erf");
}

// build a gelu graph where div = Mul(const, inputX), the const sits on div input 0
std::shared_ptr<ge::Graph> BuildTrueDivConstFirstGraph(const float divConst)
{
    GeluGraphParam param;
    param.useTrueDiv = true;
    ge::es::EsGraphBuilder builder("gelu_v2_truediv_const_first_test");
    auto inputX = builder.CreateInput(0, "x", ge::DT_FLOAT, ge::FORMAT_ND, param.dims);
    SetOutDesc(inputX, ge::DT_FLOAT, param.dims);
    auto divConstNode = builder.CreateConst(std::vector<float>{divConst}, std::vector<int64_t>{1});
    auto addConst = builder.CreateConst(std::vector<float>{1.0F}, std::vector<int64_t>{1});
    auto mulConst = builder.CreateConst(std::vector<float>{0.5F}, std::vector<int64_t>{1});
    auto div = BuildBinaryNode(builder, divConstNode, inputX, "Mul");
    auto erf0 = BuildUnaryNode(builder, div, "Erf");
    auto add = BuildBinaryNode(builder, addConst, erf0, "Add");
    auto mul0 = BuildBinaryNode(builder, mulConst, inputX, "Mul");
    auto mul1 = BuildBinaryNode(builder, mul0, add, "Mul");
    SetElementwiseDescs(div, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    SetInDesc(*div.GetProducer(), 0, ge::DT_FLOAT, {1});
    SetInDesc(*div.GetProducer(), 1, ge::DT_FLOAT, param.dims);
    SetUnaryDescs(erf0, ge::DT_FLOAT, param.dims);
    SetElementwiseDescs(add, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul0, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul1, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    return builder.BuildAndReset({mul1});
}

// build a gelu graph with an invalid add const value
std::shared_ptr<ge::Graph> BuildInvalidAddConstGraph()
{
    GeluGraphParam param;
    ge::es::EsGraphBuilder builder("gelu_v2_add_const_invalid_test");
    auto inputX = builder.CreateInput(0, "x", ge::DT_FLOAT, ge::FORMAT_ND, param.dims);
    SetOutDesc(inputX, ge::DT_FLOAT, param.dims);
    auto divConst = builder.CreateConst(std::vector<float>{1.41421F}, std::vector<int64_t>{1});
    auto addConst = builder.CreateConst(std::vector<float>{2.0F}, std::vector<int64_t>{1});
    auto mulConst = builder.CreateConst(std::vector<float>{0.5F}, std::vector<int64_t>{1});
    auto div = BuildBinaryNode(builder, inputX, divConst, "RealDiv");
    auto erf0 = BuildUnaryNode(builder, div, "Erf");
    auto add = BuildBinaryNode(builder, erf0, addConst, "Add");
    auto mul0 = BuildBinaryNode(builder, inputX, mulConst, "Mul");
    auto mul1 = BuildBinaryNode(builder, mul0, add, "Mul");
    SetElementwiseDescs(div, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetUnaryDescs(erf0, ge::DT_FLOAT, param.dims);
    SetElementwiseDescs(add, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul0, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul1, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    return builder.BuildAndReset({mul1});
}

// build a gelu graph where the div const is not a scalar (dims {2})
std::shared_ptr<ge::Graph> BuildNonScalarDivConstGraph()
{
    GeluGraphParam param;
    ge::es::EsGraphBuilder builder("gelu_v2_non_scalar_const_test");
    auto inputX = builder.CreateInput(0, "x", ge::DT_FLOAT, ge::FORMAT_ND, param.dims);
    SetOutDesc(inputX, ge::DT_FLOAT, param.dims);
    auto divConst = builder.CreateConst(std::vector<float>{1.41421F, 1.41421F}, std::vector<int64_t>{2});
    auto addConst = builder.CreateConst(std::vector<float>{1.0F}, std::vector<int64_t>{1});
    auto mulConst = builder.CreateConst(std::vector<float>{0.5F}, std::vector<int64_t>{1});
    auto div = BuildBinaryNode(builder, inputX, divConst, "RealDiv");
    auto erf0 = BuildUnaryNode(builder, div, "Erf");
    auto add = BuildBinaryNode(builder, erf0, addConst, "Add");
    auto mul0 = BuildBinaryNode(builder, inputX, mulConst, "Mul");
    auto mul1 = BuildBinaryNode(builder, mul0, add, "Mul");
    SetInDesc(*div.GetProducer(), 0, ge::DT_FLOAT, param.dims);
    SetInDesc(*div.GetProducer(), 1, ge::DT_FLOAT, {2});
    SetOutDesc(div, ge::DT_FLOAT, param.dims);
    SetUnaryDescs(erf0, ge::DT_FLOAT, param.dims);
    SetElementwiseDescs(add, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul0, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul1, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    return builder.BuildAndReset({mul1});
}

// build a gelu graph where the boundary input itself is a Const node holding xValue, and the
// 0.5 const slot on mul0 holds mulConstValue. mul0 = Mul(mulConst, inputX) so inputX sits on input x2,
// the exact input the legacy const-read logic reads first
std::shared_ptr<ge::Graph> BuildBoundaryConstXGraph(const float xValue, const float mulConstValue)
{
    GeluGraphParam param;
    param.mul0First = true;
    ge::es::EsGraphBuilder builder("gelu_v2_boundary_const_x_test");
    auto inputX = builder.CreateConst(std::vector<float>{xValue}, std::vector<int64_t>{1});
    auto divConst = builder.CreateConst(std::vector<float>{1.41421F}, std::vector<int64_t>{1});
    auto addConst = builder.CreateConst(std::vector<float>{1.0F}, std::vector<int64_t>{1});
    auto mulConst = builder.CreateConst(std::vector<float>{mulConstValue}, std::vector<int64_t>{1});
    auto div = BuildBinaryNode(builder, inputX, divConst, "RealDiv");
    auto erf0 = BuildUnaryNode(builder, div, "Erf");
    auto add = BuildBinaryNode(builder, erf0, addConst, "Add");
    auto mul0 = BuildBinaryNode(builder, mulConst, inputX, "Mul");
    auto mul1 = BuildBinaryNode(builder, mul0, add, "Mul");
    SetElementwiseDescs(div, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetUnaryDescs(erf0, ge::DT_FLOAT, param.dims);
    SetElementwiseDescs(add, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul0, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul1, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    return builder.BuildAndReset({mul1});
}

// build a gelu graph where the Erf output is also consumed by an Abs node outside the pattern:
// removing the matched subgraph would break the extra consumer, so the fusion must be skipped
std::shared_ptr<ge::Graph> BuildMultiConsumerGraph()
{
    GeluGraphParam param;
    ge::es::EsGraphBuilder builder("gelu_v2_multi_consumer_test");
    auto inputX = builder.CreateInput(0, "x", ge::DT_FLOAT, ge::FORMAT_ND, param.dims);
    SetOutDesc(inputX, ge::DT_FLOAT, param.dims);
    auto divConst = builder.CreateConst(std::vector<float>{1.41421F}, std::vector<int64_t>{1});
    auto addConst = builder.CreateConst(std::vector<float>{1.0F}, std::vector<int64_t>{1});
    auto mulConst = builder.CreateConst(std::vector<float>{0.5F}, std::vector<int64_t>{1});
    auto div = BuildBinaryNode(builder, inputX, divConst, "RealDiv");
    auto erf0 = BuildUnaryNode(builder, div, "Erf");
    auto add = BuildBinaryNode(builder, erf0, addConst, "Add");
    auto mul0 = BuildBinaryNode(builder, inputX, mulConst, "Mul");
    auto mul1 = BuildBinaryNode(builder, mul0, add, "Mul");
    auto sink = BuildUnaryNode(builder, erf0, "Abs");
    SetElementwiseDescs(div, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetUnaryDescs(erf0, ge::DT_FLOAT, param.dims);
    SetElementwiseDescs(add, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul0, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul1, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    SetUnaryDescs(sink, ge::DT_FLOAT, param.dims);
    return builder.BuildAndReset({mul1, sink});
}

// build a full tanh gelu graph where the cubeMul output is also consumed by an Abs node outside
// the pattern, exercising the tanh branch of the 8-node single-consumer chain
std::shared_ptr<ge::Graph> BuildTanhMultiConsumerGraph()
{
    GeluGraphParam param;
    ge::es::EsGraphBuilder builder("gelu_v2_tanh_multi_consumer_test");
    auto inputX = builder.CreateInput(0, "x", ge::DT_FLOAT, ge::FORMAT_ND, param.dims);
    SetOutDesc(inputX, ge::DT_FLOAT, param.dims);
    auto kappaConst = builder.CreateConst(std::vector<float>{0.044715F}, std::vector<int64_t>{1});
    auto betaConst = builder.CreateConst(std::vector<float>{0.79788456F}, std::vector<int64_t>{1});
    auto addConst = builder.CreateConst(std::vector<float>{1.0F}, std::vector<int64_t>{1});
    auto mulConst = builder.CreateConst(std::vector<float>{0.5F}, std::vector<int64_t>{1});
    auto squareMul = BuildBinaryNode(builder, inputX, inputX, "Mul");
    auto cubeMul = BuildBinaryNode(builder, squareMul, inputX, "Mul");
    auto kappaMul = BuildBinaryNode(builder, kappaConst, cubeMul, "Mul");
    auto innerAdd = BuildBinaryNode(builder, inputX, kappaMul, "Add");
    auto betaMul = BuildBinaryNode(builder, betaConst, innerAdd, "Mul");
    auto act = BuildUnaryNode(builder, betaMul, "Tanh");
    auto add = BuildBinaryNode(builder, addConst, act, "Add");
    auto mul0 = BuildBinaryNode(builder, inputX, mulConst, "Mul");
    auto mul1 = BuildBinaryNode(builder, mul0, add, "Mul");
    auto sink = BuildUnaryNode(builder, cubeMul, "Abs");
    SetElementwiseDescs(squareMul, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    SetElementwiseDescs(cubeMul, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    SetElementwiseDescs(kappaMul, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(innerAdd, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    SetElementwiseDescs(betaMul, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetUnaryDescs(act, ge::DT_FLOAT, param.dims);
    SetElementwiseDescs(add, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul0, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul1, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    SetUnaryDescs(sink, ge::DT_FLOAT, param.dims);
    return builder.BuildAndReset({mul1, sink});
}

// build a full tanh gelu graph where the boundary input itself is a Const node holding xValue,
// and the 0.5 const slot on mul0 holds mulConstValue. mul0 = Mul(mulConst, inputX) so inputX sits
// on input x2, the exact input the const-read logic reads first: the tanh boundary exclusion via
// the squareMul carrier must keep x's value from being mistaken for the 0.5 const
std::shared_ptr<ge::Graph> BuildTanhBoundaryConstXGraph(const float xValue, const float mulConstValue)
{
    GeluGraphParam param;
    ge::es::EsGraphBuilder builder("gelu_v2_tanh_boundary_const_x_test");
    auto inputX = builder.CreateConst(std::vector<float>{xValue}, std::vector<int64_t>{1});
    auto kappaConst = builder.CreateConst(std::vector<float>{0.044715F}, std::vector<int64_t>{1});
    auto betaConst = builder.CreateConst(std::vector<float>{0.79788456F}, std::vector<int64_t>{1});
    auto addConst = builder.CreateConst(std::vector<float>{1.0F}, std::vector<int64_t>{1});
    auto mulConst = builder.CreateConst(std::vector<float>{mulConstValue}, std::vector<int64_t>{1});
    auto squareMul = BuildBinaryNode(builder, inputX, inputX, "Mul");
    auto cubeMul = BuildBinaryNode(builder, squareMul, inputX, "Mul");
    auto kappaMul = BuildBinaryNode(builder, kappaConst, cubeMul, "Mul");
    auto innerAdd = BuildBinaryNode(builder, inputX, kappaMul, "Add");
    auto betaMul = BuildBinaryNode(builder, betaConst, innerAdd, "Mul");
    auto act = BuildUnaryNode(builder, betaMul, "Tanh");
    auto add = BuildBinaryNode(builder, addConst, act, "Add");
    auto mul0 = BuildBinaryNode(builder, mulConst, inputX, "Mul");
    auto mul1 = BuildBinaryNode(builder, mul0, add, "Mul");
    SetElementwiseDescs(squareMul, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    SetElementwiseDescs(cubeMul, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    SetElementwiseDescs(kappaMul, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(innerAdd, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    SetElementwiseDescs(betaMul, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetUnaryDescs(act, ge::DT_FLOAT, param.dims);
    SetElementwiseDescs(add, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul0, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul1, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    return builder.BuildAndReset({mul1});
}

// build a gelu graph where the 0.5 const on mul0 is not a scalar (dims {2}): neither mul0 nor
// mul1 provides a parseable const, the "cannot get 0.5 const" branch must reject the fusion
std::shared_ptr<ge::Graph> BuildNonScalarMulConstGraph()
{
    GeluGraphParam param;
    ge::es::EsGraphBuilder builder("gelu_v2_non_scalar_mul_const_test");
    auto inputX = builder.CreateInput(0, "x", ge::DT_FLOAT, ge::FORMAT_ND, param.dims);
    SetOutDesc(inputX, ge::DT_FLOAT, param.dims);
    auto divConst = builder.CreateConst(std::vector<float>{1.41421F}, std::vector<int64_t>{1});
    auto addConst = builder.CreateConst(std::vector<float>{1.0F}, std::vector<int64_t>{1});
    auto mulConst = builder.CreateConst(std::vector<float>{0.5F, 0.5F}, std::vector<int64_t>{2});
    auto div = BuildBinaryNode(builder, inputX, divConst, "RealDiv");
    auto erf0 = BuildUnaryNode(builder, div, "Erf");
    auto add = BuildBinaryNode(builder, erf0, addConst, "Add");
    auto mul0 = BuildBinaryNode(builder, inputX, mulConst, "Mul");
    auto mul1 = BuildBinaryNode(builder, mul0, add, "Mul");
    SetElementwiseDescs(div, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetUnaryDescs(erf0, ge::DT_FLOAT, param.dims);
    SetElementwiseDescs(add, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetInDesc(*mul0.GetProducer(), 0, ge::DT_FLOAT, param.dims);
    SetInDesc(*mul0.GetProducer(), 1, ge::DT_FLOAT, {2});
    SetOutDesc(mul0, ge::DT_FLOAT, param.dims);
    SetElementwiseDescs(mul1, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    return builder.BuildAndReset({mul1});
}

// build an erf graph whose div const is a bf16 scalar: the const dtype whitelist only accepts
// fp32/fp16, so the fusion must be skipped even though the value is the nearest of sqrt(2)
std::shared_ptr<ge::Graph> BuildBf16ConstGraph()
{
    GeluGraphParam param;
    ge::es::EsGraphBuilder builder("gelu_v2_bf16_const_test");
    auto inputX = builder.CreateInput(0, "x", ge::DT_FLOAT, ge::FORMAT_ND, param.dims);
    SetOutDesc(inputX, ge::DT_FLOAT, param.dims);
    auto divConst = builder.CreateConst(std::vector<uint16_t>{0x3FB5}, std::vector<int64_t>{1},
                                        ge::DT_BF16); // bf16 bits of sqrt(2)
    auto addConst = builder.CreateConst(std::vector<float>{1.0F}, std::vector<int64_t>{1});
    auto mulConst = builder.CreateConst(std::vector<float>{0.5F}, std::vector<int64_t>{1});
    auto div = BuildBinaryNode(builder, inputX, divConst, "RealDiv");
    auto erf0 = BuildUnaryNode(builder, div, "Erf");
    auto add = BuildBinaryNode(builder, erf0, addConst, "Add");
    auto mul0 = BuildBinaryNode(builder, inputX, mulConst, "Mul");
    auto mul1 = BuildBinaryNode(builder, mul0, add, "Mul");
    SetInDesc(*div.GetProducer(), 0, ge::DT_FLOAT, param.dims);
    SetInDesc(*div.GetProducer(), 1, ge::DT_BF16, {1});
    SetOutDesc(div, ge::DT_FLOAT, param.dims);
    SetUnaryDescs(erf0, ge::DT_FLOAT, param.dims);
    SetElementwiseDescs(add, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul0, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul1, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    return builder.BuildAndReset({mul1});
}

// build an erf graph where the add const is not a scalar (dims {2}): the shared scalar
// validation in ParseScalarValue must reject the fusion
std::shared_ptr<ge::Graph> BuildNonScalarAddConstGraph()
{
    GeluGraphParam param;
    ge::es::EsGraphBuilder builder("gelu_v2_non_scalar_add_const_test");
    auto inputX = builder.CreateInput(0, "x", ge::DT_FLOAT, ge::FORMAT_ND, param.dims);
    SetOutDesc(inputX, ge::DT_FLOAT, param.dims);
    auto divConst = builder.CreateConst(std::vector<float>{1.41421F}, std::vector<int64_t>{1});
    auto addConst = builder.CreateConst(std::vector<float>{1.0F, 1.0F}, std::vector<int64_t>{2});
    auto mulConst = builder.CreateConst(std::vector<float>{0.5F}, std::vector<int64_t>{1});
    auto div = BuildBinaryNode(builder, inputX, divConst, "RealDiv");
    auto erf0 = BuildUnaryNode(builder, div, "Erf");
    auto add = BuildBinaryNode(builder, erf0, addConst, "Add");
    auto mul0 = BuildBinaryNode(builder, inputX, mulConst, "Mul");
    auto mul1 = BuildBinaryNode(builder, mul0, add, "Mul");
    SetElementwiseDescs(div, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetUnaryDescs(erf0, ge::DT_FLOAT, param.dims);
    SetInDesc(*add.GetProducer(), 0, ge::DT_FLOAT, param.dims);
    SetInDesc(*add.GetProducer(), 1, ge::DT_FLOAT, {2});
    SetOutDesc(add, ge::DT_FLOAT, param.dims);
    SetElementwiseDescs(mul0, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul1, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    return builder.BuildAndReset({mul1});
}

static void SetNpuArch(const char* arch) { (void)setenv("ASCEND_NPU_ARCH", arch, 1); }

static void SetPlatform(const std::string& soc, const int32_t npuArch)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = soc;
    optiCompilationInfo.soc_version = soc;
    fe::PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    SetNpuArch(std::to_string(npuArch).c_str());
}
} // namespace

class GeluV2ONNXFusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950", kArch950); }

    // verify the graph contains exactly one GeluV2 node with the expected approximate attr,
    // input/output desc and input mapping, and no residual source ops
    void ExpectGeluV2Fused(const std::shared_ptr<ge::Graph>& graph, const std::string& expectedApproximate,
                           ge::DataType expectedDtype, const std::vector<int64_t>& expectedDims)
    {
        int32_t geluCount = 0;
        int32_t residualCount = 0;
        for (auto node : graph->GetAllNodes()) {
            ge::AscendString type;
            node.GetType(type);
            if ((type == "RealDiv") || (type == "Erf") || (type == "Tanh") || (type == "Add") || (type == "Mul")) {
                residualCount++;
            }
            if (type == "GeluV2") {
                geluCount++;
                // the ES api omits attrs equal to the IR default, a missing attr means "none"
                ge::AscendString approximate;
                std::string approximateStr = "none";
                if (node.GetAttr(ge::AscendString("approximate"), approximate) == ge::GRAPH_SUCCESS) {
                    approximateStr = std::string(approximate.GetString());
                }
                EXPECT_EQ(approximateStr, expectedApproximate);
                ge::TensorDesc inputDesc;
                node.GetInputDesc(0, inputDesc);
                EXPECT_EQ(inputDesc.GetDataType(), expectedDtype);
                EXPECT_EQ(inputDesc.GetShape().GetDims(), expectedDims);
                ge::TensorDesc outputDesc;
                node.GetOutputDesc(0, outputDesc);
                EXPECT_EQ(outputDesc.GetDataType(), expectedDtype);
                EXPECT_EQ(outputDesc.GetShape().GetDims(), expectedDims);
                // verify the input comes from the source inputX node
                auto src = node.GetInDataNodesAndPortIndexs(0);
                ASSERT_NE(src.first, nullptr);
                ge::AscendString srcName;
                src.first->GetName(srcName);
                EXPECT_NE(std::string(srcName.GetString()).find("x"), std::string::npos);
            }
        }
        EXPECT_EQ(geluCount, 1);
        EXPECT_EQ(residualCount, 0);
    }
};

TEST_F(GeluV2ONNXFusionPassTest, patternTest)
{
    ops::GeluV2ONNXFusionPass pass;
    std::vector<ge::fusion::PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_EQ(patterns.size(), kExpectedPatternCount);
}

TEST_F(GeluV2ONNXFusionPassTest, erfRealDivMul0ConstFp32Success)
{
    GeluGraphParam param;
    param.divConst = 1.41421F;
    param.dims = {1, 512, 3072};
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
    ExpectGeluV2Fused(graph, "none", ge::DT_FLOAT, param.dims);
}

TEST_F(GeluV2ONNXFusionPassTest, erfRealDivMul1ConstFp32Success)
{
    GeluGraphParam param;
    param.mul1ConstFamily = true;
    param.divConst = 1.41421F;
    param.dims = {1, 3, 224, 224};
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
    ExpectGeluV2Fused(graph, "none", ge::DT_FLOAT, param.dims);
}

TEST_F(GeluV2ONNXFusionPassTest, erfRealDivMul0ConstFp16XSuccess)
{
    GeluGraphParam param;
    param.xDtype = ge::DT_FLOAT16;
    param.divConst = 1.41421F;
    param.dims = {2, 32, 128};
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
    ExpectGeluV2Fused(graph, "none", ge::DT_FLOAT16, param.dims);
}

TEST_F(GeluV2ONNXFusionPassTest, erfRealDivInvalidDivConstNotChanged)
{
    GeluGraphParam param;
    param.divConst = 1.41456F; // tolerance is 1e-5 for fp32, this must not fuse
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, erfNanDivConstNotChanged)
{
    // NaN compares false against every tolerance, the scalar parser must reject it before the
    // fusion silently changes the numerics (the source graph would output NaN, not gelu(x))
    GeluGraphParam param;
    param.divConst = std::numeric_limits<float>::quiet_NaN();
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, erfTrueDivFp16ConstOrderSwappedSuccess)
{
    GeluGraphParam param;
    param.xDtype = ge::DT_FLOAT16;
    param.useTrueDiv = true;
    param.fp16Const = true;
    param.divConst = 0.70703125F; // fp16 nearest value of 1/sqrt(2)
    param.addConstFirst = true;   // Add(const, erf)
    param.mul0First = true;       // Mul(const, inputX)
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
    ExpectGeluV2Fused(graph, "none", ge::DT_FLOAT16, param.dims);
}

TEST_F(GeluV2ONNXFusionPassTest, erfTrueDivFp32Success)
{
    GeluGraphParam param;
    param.useTrueDiv = true;
    param.divConst = 0.70710677F;
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
    ExpectGeluV2Fused(graph, "none", ge::DT_FLOAT, param.dims);
}

TEST_F(GeluV2ONNXFusionPassTest, erfTrueDivConstAtDivInput0NotChanged)
{
    // div = Mul(const, inputX), the legacy pass does not support const on div input 0
    std::shared_ptr<ge::Graph> graph = BuildTrueDivConstFirstGraph(0.70710677F);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, tanhFullCanonicalSuccess)
{
    // the exact PyTorch onnx export shape: const-first kappa/beta/one, 0.5 on mul0, reading orders
    GeluGraphParam param;
    param.useTanh = true;
    param.addConstFirst = true;
    param.dims = {1, 512, 3072};
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
    ExpectGeluV2Fused(graph, "tanh", ge::DT_FLOAT, param.dims);
}

TEST_F(GeluV2ONNXFusionPassTest, tanhMul1ConstFamilyNotChanged)
{
    // the 0.5 const binds last (family 2): no known onnx exporter produces this form for the
    // tanh formula, the family is dropped from the pattern set to keep the count at 32
    GeluGraphParam param;
    param.useTanh = true;
    param.mul1ConstFamily = true;
    param.addConstFirst = true;
    param.mul1First = true;
    param.dims = {1, 3, 224, 224};
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, tanhFullConstSecondOrderSwappedSuccess)
{
    // kappa/beta consts on the second input and the outer muls swapped
    GeluGraphParam param;
    param.useTanh = true;
    param.kappaConstFirst = false;
    param.betaConstFirst = false;
    param.mul0First = true;
    param.mul1First = true;
    param.dims = {4, 64};
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
    ExpectGeluV2Fused(graph, "tanh", ge::DT_FLOAT, param.dims);
}

TEST_F(GeluV2ONNXFusionPassTest, tanhFullInvalidBetaNotChanged)
{
    // a beta of sqrt(2) belongs to the erf formula, the tanh formula must reject it
    GeluGraphParam param;
    param.useTanh = true;
    param.betaConst = 1.41421356F;
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, tanhFullInvalidKappaNotChanged)
{
    GeluGraphParam param;
    param.useTanh = true;
    param.kappaConst = 0.1F;
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, simplifiedTanhNotChanged)
{
    // the legacy simplified tanh formula (no cubic term) is no longer matched: the kernel
    // computes the full approximation, fusing the simplified shape would change the numerics
    std::shared_ptr<ge::Graph> graph = BuildLegacySimplifiedTanhGraph();
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, erfWithTanhConstNotChanged)
{
    // erf graph with the tanh div const must not fuse
    GeluGraphParam param;
    param.divConst = 1.25331413731550F;
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, addConstNotOneNotChanged)
{
    std::shared_ptr<ge::Graph> graph = BuildInvalidAddConstGraph();
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, nonScalarDivConstNotChanged)
{
    std::shared_ptr<ge::Graph> graph = BuildNonScalarDivConstGraph();
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, boundaryConstXWithInvalidMulConstNotChanged)
{
    // inputX is a Const(0.5) node but the real mul const is 2.0: the graph computes 2.0*inputX*(1+erf(inputX/sqrt(2)))
    // instead of gelu(inputX), the const value on inputX must not be mistaken for the 0.5 const
    std::shared_ptr<ge::Graph> graph = BuildBoundaryConstXGraph(0.5F, 2.0F);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, boundaryConstXValidMulConstFused)
{
    // inputX is a Const(0.5) node and the real mul const is 0.5: the graph still computes gelu(inputX),
    // the boundary const must not be rejected by the inputX-exclusion logic.
    // no ExpectGeluV2Fused here: the gelu input producer is the boundary Const node whose
    // auto-generated name does not contain "x", unlike the named input in the other tests
    std::shared_ptr<ge::Graph> graph = BuildBoundaryConstXGraph(0.5F, 0.5F);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
}

TEST_F(GeluV2ONNXFusionPassTest, unsupportedDtypeNotChanged)
{
    GeluGraphParam param;
    param.xDtype = ge::DT_INT32;
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, unsupportedPlatformNotChanged)
{
    // the Ascend910_93 family reports NpuArch 2201, the arch guard must reject it; SetUp
    // restores the Ascend950 platform for the next test
    SetPlatform("Ascend910_93", kArch910_93);
    GeluGraphParam param;
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, archQueryFailedNotChanged)
{
    // rtGetSocSpec fails (env unset in the stub): IsDav3510Arch falls back to false with a
    // warning, the fusion must be skipped
    (void)unsetenv("ASCEND_NPU_ARCH");
    GeluGraphParam param;
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, bf16InputSuccess)
{
    GeluGraphParam param;
    param.xDtype = ge::DT_BF16;
    param.divConst = 1.41421F;
    param.dims = {2, 32, 128};
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
    ExpectGeluV2Fused(graph, "none", ge::DT_BF16, param.dims);
}

TEST_F(GeluV2ONNXFusionPassTest, multiConsumerIntermediateNotChanged)
{
    // the Erf output is also consumed by an Abs node outside the pattern, removing the matched
    // subgraph would break that consumer, so the fusion must be skipped
    std::shared_ptr<ge::Graph> graph = BuildMultiConsumerGraph();
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, tanhMultiConsumerIntermediateNotChanged)
{
    // the cubeMul output is also consumed by an Abs node outside the pattern, exercising the
    // tanh branch of the 8-node single-consumer chain
    std::shared_ptr<ge::Graph> graph = BuildTanhMultiConsumerGraph();
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, tanhBoundaryConstXWithInvalidMulConstNotChanged)
{
    // the boundary input is a Const(0.5) node but the real mul const is 2.0: the graph computes
    // 2.0*x*(1+tanh(...)) instead of gelu(x), the const value on inputX must not be mistaken
    // for the 0.5 const via the squareMul carrier path
    std::shared_ptr<ge::Graph> graph = BuildTanhBoundaryConstXGraph(0.5F, 2.0F);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, tanhBoundaryConstXValidMulConstFused)
{
    // the boundary input is a Const(0.5) node and the real mul const is 0.5: the graph still
    // computes gelu(x), the boundary const must not be rejected by the x-exclusion logic.
    // no ExpectGeluV2Fused here: the gelu input producer is the boundary Const node whose
    // auto-generated name does not contain "x", unlike the named input in the other tests
    std::shared_ptr<ge::Graph> graph = BuildTanhBoundaryConstXGraph(0.5F, 0.5F);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
}

TEST_F(GeluV2ONNXFusionPassTest, trueDivScalarXNotChanged)
{
    // legacy restriction: for the TrueDiv(Mul) div node, the boundary input must not be a scalar
    GeluGraphParam param;
    param.useTrueDiv = true;
    param.divConst = 0.70710677F;
    param.dims = {1};
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, realDivScalarXFused)
{
    // the scalar-inputX restriction is TrueDiv-only: RealDiv with a scalar inputX still fuses
    GeluGraphParam param;
    param.divConst = 1.41421F;
    param.dims = {1};
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
    ExpectGeluV2Fused(graph, "none", ge::DT_FLOAT, param.dims);
}

TEST_F(GeluV2ONNXFusionPassTest, nonScalarMulConstNotChanged)
{
    // the 0.5 const on mul0 is not a scalar, neither mul0 nor mul1 provides a parseable const
    std::shared_ptr<ge::Graph> graph = BuildNonScalarMulConstGraph();
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, unsupportedConstDtypeNotChanged)
{
    // the div const is a bf16 scalar: the const dtype whitelist only accepts fp32/fp16, the
    // fusion must be skipped even though the bf16 value is the nearest of sqrt(2)
    std::shared_ptr<ge::Graph> graph = BuildBf16ConstGraph();
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, nonScalarAddConstNotChanged)
{
    // the add const is not a scalar (dims {2}), the shared scalar validation must reject it
    std::shared_ptr<ge::Graph> graph = BuildNonScalarAddConstGraph();
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
}

TEST_F(GeluV2ONNXFusionPassTest, erfRealDivFp16ConstSuccess)
{
    // fp16 consts are quantized to the nearest fp16 value, covered by the fp16 tolerance
    GeluGraphParam param;
    param.fp16Const = true;
    param.divConst = 1.4140625F; // fp16 nearest value of sqrt(2)
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
    ExpectGeluV2Fused(graph, "none", ge::DT_FLOAT, param.dims);
}

TEST_F(GeluV2ONNXFusionPassTest, tanhFullFp16ConstSuccess)
{
    // fp16 consts are quantized to the nearest fp16 value, covered by the kappa/beta tolerances
    GeluGraphParam param;
    param.useTanh = true;
    param.fp16Const = true;
    param.kappaConst = 0.044708251953125F; // fp16 nearest value of 0.044715
    param.betaConst = 0.7978515625F;       // fp16 nearest value of sqrt(2 / pi)
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
    ExpectGeluV2Fused(graph, "tanh", ge::DT_FLOAT, param.dims);
}

TEST_F(GeluV2ONNXFusionPassTest, tanhFullFp16ConstSecondOrderSwappedSuccess)
{
    GeluGraphParam param;
    param.useTanh = true;
    param.fp16Const = true;
    param.kappaConstFirst = false;
    param.betaConstFirst = false;
    param.mul0First = true;
    param.kappaConst = 0.044708251953125F; // fp16 nearest value of 0.044715
    param.betaConst = 0.7978515625F;       // fp16 nearest value of sqrt(2 / pi)
    std::shared_ptr<ge::Graph> graph = BuildGeluGraph(param);
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::SUCCESS);
    ExpectGeluV2Fused(graph, "tanh", ge::DT_FLOAT, param.dims);
}

TEST_F(GeluV2ONNXFusionPassTest, erfSharedConstNotChanged)
{
    // the 0.5 const also feeds an Abs node outside the pattern: the matcher enforces exact
    // output fan-out on every pattern-internal node (consts included), so the shared const
    // must reject the fusion and the external consumer keeps its valid const input
    GeluGraphParam param;
    ge::es::EsGraphBuilder builder("gelu_v2_shared_const_test");
    auto inputX = builder.CreateInput(0, "x", ge::DT_FLOAT, ge::FORMAT_ND, param.dims);
    auto divConst = builder.CreateConst(std::vector<float>{1.41421F}, std::vector<int64_t>{1});
    auto addConst = builder.CreateConst(std::vector<float>{1.0F}, std::vector<int64_t>{1});
    auto mulConst = builder.CreateConst(std::vector<float>{0.5F}, std::vector<int64_t>{1});
    auto div = BuildBinaryNode(builder, inputX, divConst, "RealDiv");
    auto erf0 = BuildUnaryNode(builder, div, "Erf");
    auto add = BuildBinaryNode(builder, erf0, addConst, "Add");
    auto mul0 = BuildBinaryNode(builder, inputX, mulConst, "Mul");
    auto mul1 = BuildBinaryNode(builder, mul0, add, "Mul");
    auto sink = BuildUnaryNode(builder, mulConst, "Abs");
    SetElementwiseDescs(div, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetUnaryDescs(erf0, ge::DT_FLOAT, param.dims);
    SetElementwiseDescs(add, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul0, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, true);
    SetElementwiseDescs(mul1, ge::DT_FLOAT, param.dims, ge::DT_FLOAT, false);
    SetUnaryDescs(sink, ge::DT_FLOAT, param.dims);
    std::shared_ptr<ge::Graph> graph = builder.BuildAndReset({mul1, sink});
    ge::CustomPassContext passContext;
    ops::GeluV2ONNXFusionPass pass;
    EXPECT_EQ(pass.Run(graph, passContext), ge::GRAPH_NOT_CHANGED);
    int32_t geluCount = 0;
    int32_t absWithConstInput = 0;
    for (auto node : graph->GetAllNodes()) {
        ge::AscendString type;
        node.GetType(type);
        if (type == "GeluV2") {
            geluCount++;
        }
        if (type == "Abs") {
            auto src = node.GetInDataNodesAndPortIndexs(0);
            ASSERT_NE(src.first, nullptr);
            ge::AscendString srcType;
            src.first->GetType(srcType);
            if (srcType == "Const") {
                absWithConstInput++;
            }
        }
    }
    EXPECT_EQ(geluCount, 0);
    EXPECT_EQ(absWithConstInput, 1);
}
