/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <dlfcn.h>

#include <gtest/gtest.h>
#include "../../../op_graph/fusion_pass/rms_norm_fusion_pass.h"
#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/ge_api.h"
#include "ge/ge_utils.h"
#include "platform/platform_info.h"
#include "version/ge-compiler_version.h"

// CANN 8.5 declares an inline context destructor with an incomplete Impl.
// That SDK is exercised through GE's own callback in integration tests.
#if defined(GE_COMPILER_VERSION_NUM) && GE_COMPILER_VERSION_NUM >= 90000000U
namespace {
using Dims = std::vector<int64_t>;
using Tensor = ge::es::EsTensorHolder;

struct Case {
    std::string name = "canonical";
    int square = 0;
    int mean = 0;
    int epsilon = 0;
    int root = 0;
    int restore = 0;
    int casts = 0;
    int swaps = 0;
    int tail = 1;
    int repeat = 0;
    int gammaMode = 1;
    int gammaSource = 0;
    int restoreDepth = 1;
    int outputCasts = 0;
    bool realDiv = false;
    bool axesInt32 = false;
    bool reverseMean = false;
    bool reverseNormalize = false;
    bool typedConstants = false;
    bool device = false;
    bool wrongRestore = false;
    std::string coverage;
    bool restoreAtRoot = false;
    bool splitRestore = false;
    bool inferSource = false;
    bool keep = true;
    bool noopWithEmptyAxes = true;
    bool side = false;
    bool exposeRstd = false;
    bool exposedReciprocal = false;
    bool duplicateRstd = false;
    bool beta = false;
    bool betaAdds = false;
    bool residual = false;
    bool residualSwap = false;
    bool residualAddV2 = false;
    bool residualAlsoBeta = false;
    bool differentX = false;
    bool wrongOutput = false;
    bool control = false;
    // 1/2: external incoming/outgoing root-Cast controls; 3: duplicate data dependency.
    // 4/5: nonredundant gamma -> square / square -> gamma controls.
    int rootControl = 0;
    bool rootStream = false;
    bool sharedConstant = false;
    bool sharedCast = false;
    bool downcastScale = false;
    bool wrongNormalizedType = false;
    bool valid = true;
    bool targetSupported = true;
    bool dynamicEpsilon = false;
    // 0: absent; 1: true; 2: false; 3: wrong attribute type.
    int lowPrecision = 0;
    float eps = 1e-5F;
    float exponent = 2.0F;
    float rootExponent = -0.5F;
    float divisor = 64.0F;
    ge::DataType factorType = ge::DT_UNDEFINED;
    float rootNumerator = 1.0F;
    float exposedNumerator = 1.0F;
    Dims dims{2, 64};
    Dims gamma{64};
    Dims axes{-1};
    Dims epsilonShape{};
    Dims runtimeDims{};
    Dims runtimeGamma{};
    Dims outputDims{};
    Dims residualShape{};
    ge::DataType dtype = ge::DT_FLOAT;
    // Independent JD boundaries; undefined preserves the legacy fixture layout.
    ge::DataType gammaDtype = ge::DT_UNDEFINED;
    ge::DataType normalizedCastType = ge::DT_UNDEFINED;
    // A low root Cast can feed a Pow/Mul chain that promotes to FP32.
    ge::DataType rootCastType = ge::DT_UNDEFINED;
};

ge::DataType GammaDtype(const Case& c)
{
    return c.gammaDtype != ge::DT_UNDEFINED ? c.gammaDtype : (c.casts == 3 ? ge::DT_FLOAT : c.dtype);
}

ge::DataType NormalizedCastDtype(const Case& c)
{
    return c.wrongNormalizedType                    ? ge::DT_BF16 :
           c.normalizedCastType != ge::DT_UNDEFINED ? c.normalizedCastType :
                                                      c.dtype;
}

ge::DataType GammaMulDtype(const Case& c)
{
    if (c.gammaMode != 1) {
        return c.casts == 3 ? ge::DT_FLOAT : c.dtype;
    }
    const auto normalized = (c.casts == 1 || c.casts == 2) && !c.downcastScale ? NormalizedCastDtype(c) :
                            c.downcastScale || c.casts == 0                    ? c.dtype :
                                                                                 ge::DT_FLOAT;
    return normalized == GammaDtype(c) ? normalized : ge::DT_FLOAT;
}

ge::DataType OutputDtype(const Case& c)
{
    if (c.outputCasts != 0) {
        return c.outputCasts % 2 == 1 ? ge::DT_FLOAT : c.dtype;
    }
    return c.casts == 3 ? c.dtype : GammaMulDtype(c);
}

// Test constants use the same storage dtype as the arithmetic graph. Explicit
// IEEE conversion keeps T1 fixtures valid without an implicit promotion to FP32.
uint16_t Encode16(float value, ge::DataType dtype)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if (dtype == ge::DT_BF16) {
        return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
    }
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const int exponent = static_cast<int>((bits >> 23) & 255U) - 127;
    if (exponent > 15) {
        return sign | 0x7c00U | ((bits & 0x7fffffU) != 0 && exponent == 128 ? 0x0200U : 0U);
    }
    if (exponent < -25) {
        return sign;
    }
    const double magnitude = std::abs(static_cast<double>(value));
    const int step = std::max(exponent - 10, -24);
    const auto rounded = static_cast<uint32_t>(std::nearbyint(std::ldexp(magnitude, -step)));
    if (exponent < -14) {
        return sign | static_cast<uint16_t>(rounded);
    }
    return sign | static_cast<uint16_t>(((exponent + 14) << 10) + rounded);
}

std::string Type(const ge::GNode& node)
{
    ge::AscendString type;
    node.GetType(type);
    return type.GetString() == nullptr ? "" : type.GetString();
}

void Describe(const Tensor& tensor, const Dims& dims, ge::DataType dtype)
{
    auto* node = tensor.GetProducer();
    if (node == nullptr) {
        throw std::runtime_error("null fixture node");
    }
    ge::TensorDesc desc(ge::Shape(dims), ge::FORMAT_ND, dtype);
    desc.SetOriginShape(ge::Shape(dims));
    desc.SetOriginFormat(ge::FORMAT_ND);
    if ((Type(*node) == "Data" && node->UpdateInputDesc(0, desc) != ge::GRAPH_SUCCESS) ||
        node->UpdateOutputDesc(tensor.GetProducerOutIndex(), desc) != ge::GRAPH_SUCCESS) {
        throw std::runtime_error("fixture descriptor update failed");
    }
}

Tensor Scalar(ge::es::EsGraphBuilder& builder, float value, ge::DataType dtype, const Dims& dims = {})
{
    auto tensor = builder.CreateConst(std::vector<float>{value}, dims);
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        const uint16_t bits = Encode16(value, dtype);
        ge::Tensor constant(ge::TensorDesc(ge::Shape(dims), ge::FORMAT_ND, dtype),
                            reinterpret_cast<const uint8_t*>(&bits), sizeof(bits));
        if (tensor.GetProducer()->SetAttr("value", constant) != ge::GRAPH_SUCCESS) {
            throw std::runtime_error("constant dtype update failed");
        }
        Describe(tensor, dims, dtype);
    }
    return tensor;
}

Tensor Divide(const Tensor& lhs, const Tensor& rhs, bool realDiv)
{
    return realDiv ? ge::es::RealDiv(lhs, rhs) : ge::es::Div(lhs, rhs);
}

Tensor Restore(ge::es::EsGraphBuilder& builder, const Tensor& input, int kind, const Dims& keep)
{
    switch (kind) {
        case 1:
            return ge::es::Reshape(input, builder.CreateVector(keep));
        case 2:
            return ge::es::Unsqueeze(input, {-1});
        case 3:
            return ge::es::ExpandDims(input, builder.CreateScalar(int64_t{-1}));
        case 4:
            return ge::es::UnsqueezeV2(input, {-1});
        default:
            return input;
    }
}

ge::GraphPtr Build(const Case& c)
{
    ge::es::EsGraphBuilder builder(c.name.c_str());
    auto x = builder.CreateInput(0, "x", c.dtype, ge::FORMAT_ND, c.dims);
    auto gamma = builder.CreateInput(1, "gamma", GammaDtype(c), ge::FORMAT_ND, c.gamma);
    Describe(x, c.dims, c.dtype);
    Describe(gamma, c.gamma, GammaDtype(c));
    const auto computeType = c.casts != 0 ? ge::DT_FLOAT : c.dtype;
    const auto rootType = c.rootCastType == ge::DT_UNDEFINED ? computeType : c.rootCastType;
    auto computeX = c.casts != 0 ? ge::es::Cast(x, rootType) : x;
    const auto constantType = c.typedConstants ? computeType : ge::DT_FLOAT;
    Describe(computeX, c.dims, rootType);
    auto squareX = c.casts == 2 ? ge::es::Cast(x, rootType) : computeX;
    Describe(squareX, c.dims, rootType);
    if (c.differentX) {
        squareX = builder.CreateInput(2, "other_x", rootType, ge::FORMAT_ND, c.dims);
    }
    auto square = c.square == 0 ? ge::es::Square(squareX) :
                  c.square == 1 ? ge::es::Mul(squareX, squareX) :
                                  ge::es::Pow(squareX, Scalar(builder, c.exponent, constantType));
    Describe(square, c.dims, computeType);
    Dims keep = c.dims;
    Dims reductionAxes = c.axes;
    if (reductionAxes.empty() && !c.noopWithEmptyAxes) {
        for (size_t i = 0; i < c.dims.size(); ++i) {
            reductionAxes.push_back(static_cast<int64_t>(i));
        }
    }
    bool legalAxes = !reductionAxes.empty();
    for (auto& axis : reductionAxes) {
        if (axis < 0) {
            axis += static_cast<int64_t>(c.dims.size());
        }
        legalAxes = legalAxes && axis >= 0 && axis < static_cast<int64_t>(c.dims.size());
    }
    if (legalAxes) {
        for (auto axis : reductionAxes) {
            keep[axis] = 1;
        }
    } else {
        for (int i = 0; i < c.tail; ++i) {
            keep[keep.size() - 1 - i] = 1;
        }
    }
    Dims reduced;
    for (size_t i = 0; i < c.dims.size(); ++i) {
        const bool isReduced = legalAxes ?
                                   std::find(reductionAxes.begin(), reductionAxes.end(), i) != reductionAxes.end() :
                                   i >= c.dims.size() - c.tail;
        if (!isReduced) {
            reduced.push_back(c.dims[i]);
        }
    }
    const bool keepdims = c.restore == 0 && c.keep;
    Dims statisticShape = keepdims ? keep : reduced;
    const auto axes = c.axesInt32 ? builder.CreateVector(std::vector<int32_t>(c.axes.begin(), c.axes.end())) :
                                    builder.CreateVector(c.axes.empty() ? Dims{-1} : c.axes);
    if (c.axes.empty()) {
        ge::Tensor empty(ge::TensorDesc(ge::Shape({0}), ge::FORMAT_ND, ge::DT_INT64));
        axes.GetProducer()->SetAttr("value", empty);
        Describe(axes, {0}, ge::DT_INT64);
    }
    Tensor mean = c.mean == 0 ? ge::es::ReduceMean(square, axes, keepdims) : ge::es::ReduceSum(square, axes, keepdims);
    bool noop = c.noopWithEmptyAxes;
    if (mean.GetProducer()->SetAttr("noop_with_empty_axes", noop) != ge::GRAPH_SUCCESS) {
        return nullptr;
    }
    Describe(mean, statisticShape, computeType);
    if (c.mean != 0) {
        const auto factorType = c.factorType == ge::DT_UNDEFINED ? constantType : c.factorType;
        auto factor = Scalar(builder, c.mean == 1 ? 1.0F / c.divisor : c.divisor, factorType);
        mean = c.mean == 1 ? ((c.swaps & 1) ? ge::es::Mul(factor, mean) : ge::es::Mul(mean, factor)) :
                             (c.reverseMean ? Divide(factor, mean, c.realDiv) : Divide(mean, factor, c.realDiv));
        Describe(mean, statisticShape, computeType);
    }
    if (c.restore != 0 && !c.restoreAtRoot) {
        const int steps = c.splitRestore ? 1 : c.restoreDepth;
        for (int i = 0; i < steps; ++i) {
            mean = Restore(builder, mean, c.restore, keep);
            // An insertion restores one axis per node; only Reshape restores all
            // axes at once. Keep descriptors faithful to the actual primitives.
            if (c.restore == 1) {
                statisticShape = keep;
            } else {
                statisticShape.push_back(1);
            }
            if (c.wrongRestore) {
                statisticShape = {1, c.dims.front()};
            }
            Describe(mean, statisticShape, computeType);
        }
    }
    auto epsilon = c.dynamicEpsilon ? builder.CreateInput(2, "epsilon", ge::DT_FLOAT, ge::FORMAT_ND, {}) :
                                      Scalar(builder, c.eps, constantType, c.epsilonShape);
    const auto left = (c.swaps & 2) ? epsilon : mean;
    const auto right = (c.swaps & 2) ? mean : epsilon;
    auto stable = c.epsilon == 0 ? ge::es::Add(left, right) :
                  c.epsilon == 1 ? ge::es::AddV2(left, right) :
                                   ge::es::Adds(mean, c.eps);
    // Preserve small, nonzero E3 values even when the ES builder omits float
    // attributes near the default. The input graph must match the test case.
    float requestedEpsilon = c.eps;
    if (c.epsilon == 2 && (stable.GetProducer() == nullptr ||
                           stable.GetProducer()->SetAttr("value", requestedEpsilon) != ge::GRAPH_SUCCESS)) {
        return nullptr;
    }
    Describe(stable, statisticShape, computeType);
    Tensor scale;
    switch (c.root) {
        case 0:
            scale = ge::es::Rsqrt(stable);
            break;
        case 1: {
            auto rms = ge::es::Sqrt(stable);
            Describe(rms, statisticShape, computeType);
            scale = ge::es::Reciprocal(rms);
            break;
        }
        case 2: {
            auto rms = ge::es::Sqrt(stable);
            Describe(rms, statisticShape, computeType);
            scale = Divide(Scalar(builder, c.rootNumerator, constantType), rms, c.realDiv);
            break;
        }
        case 3:
            scale = ge::es::Pow(stable, Scalar(builder, c.rootExponent, constantType));
            break;
        case 4:
            scale = ge::es::Sqrt(stable);
            break;
        default:
            scale = ge::es::Pow(stable, Scalar(builder, -c.rootExponent, constantType));
            break;
    }
    Describe(scale, statisticShape, computeType);
    if (c.restore != 0 && (c.restoreAtRoot || c.splitRestore)) {
        const int steps = c.splitRestore ? c.restoreDepth - 1 : c.restoreDepth;
        for (int i = 0; i < steps; ++i) {
            scale = Restore(builder, scale, c.restore, keep);
            if (c.restore == 1) {
                statisticShape = keep;
            } else {
                statisticShape.push_back(1);
            }
            if (c.wrongRestore) {
                statisticShape = {1, c.dims.front()};
            }
            Describe(scale, statisticShape, computeType);
        }
    }
    if (c.downcastScale) {
        scale = ge::es::Cast(scale, c.dtype);
        Describe(scale, statisticShape, c.dtype);
    }
    auto normalizedX = c.downcastScale ? x : computeX;
    const auto normalizedDtype = c.downcastScale ? c.dtype : computeType;
    if (c.gammaMode == 2) {
        normalizedX = ge::es::Mul(normalizedX, gamma);
        Describe(normalizedX, c.dims, normalizedDtype);
    }
    auto normalized = c.root >= 4   ? (c.reverseNormalize ? Divide(scale, normalizedX, c.realDiv) :
                                                            Divide(normalizedX, scale, c.realDiv)) :
                      (c.swaps & 4) ? ge::es::Mul(scale, normalizedX) :
                                      ge::es::Mul(normalizedX, scale);
    Describe(normalized, c.dims, normalizedDtype);
    if ((c.casts == 1 || c.casts == 2) && !c.downcastScale) {
        const auto normalizedType = NormalizedCastDtype(c);
        normalized = ge::es::Cast(normalized, normalizedType);
        Describe(normalized, c.dims, normalizedType);
    }
    const auto originalGamma = gamma;
    if (c.gammaSource != 0) {
        gamma = c.gammaSource == 1 ? scale : c.gammaSource == 2 ? normalized : computeX;
    }
    auto y = c.gammaMode == 1 ? ((c.swaps & 8) ? ge::es::Mul(gamma, normalized) : ge::es::Mul(normalized, gamma)) :
                                normalized;
    if (c.gammaMode == 3) {
        auto weightedScale = ge::es::Mul(scale, gamma);
        Describe(weightedScale, c.dims, computeType);
        y = ge::es::Mul(computeX, weightedScale);
    }
    Describe(y, c.outputDims.empty() ? c.dims : c.outputDims, GammaMulDtype(c));
    if (c.lowPrecision != 0) {
        ge::graphStatus status;
        if (c.lowPrecision == 3) {
            int64_t enabled = 1;
            status = y.GetProducer()->SetAttr("_rms_norm_fusion_low_precision", enabled);
        } else {
            bool enabled = c.lowPrecision == 1;
            status = y.GetProducer()->SetAttr("_rms_norm_fusion_low_precision", enabled);
        }
        if (status != ge::GRAPH_SUCCESS) {
            throw std::runtime_error("fixture low-precision attribute update failed");
        }
    }
    if (c.wrongOutput) {
        Describe(y, {1, 64}, c.dtype);
    }
    if (c.casts == 3) {
        y = ge::es::Cast(y, c.dtype);
        Describe(y, c.dims, c.dtype);
    }
    for (int i = 0; i < c.outputCasts; ++i) {
        const auto dtype = i % 2 == 0 ? ge::DT_FLOAT : c.dtype;
        y = ge::es::Cast(y, dtype);
        Describe(y, c.dims, dtype);
    }
    const auto normOutput = y;
    const auto outputDtype = OutputDtype(c);
    if (c.residual) {
        const auto shape = c.residualShape.empty() ? c.dims : c.residualShape;
        auto residual = builder.CreateInput(2, "residual", outputDtype, ge::FORMAT_ND, shape);
        Describe(residual, shape, outputDtype);
        const auto left = c.residualSwap ? residual : y;
        const auto right = c.residualSwap ? y : residual;
        y = c.residualAddV2 ? ge::es::AddV2(left, right) : ge::es::Add(left, right);
        Describe(y, c.dims, outputDtype);
    }
    if (c.beta) {
        y = c.betaAdds ? ge::es::Adds(y, 1.0F) : ge::es::Add(y, Scalar(builder, 1.0F, outputDtype));
        Describe(y, c.dims, outputDtype);
    }
    std::vector<Tensor> outputs{y};
    if (c.gammaSource != 0) {
        outputs.push_back(originalGamma);
    }
    if (c.residualAlsoBeta) {
        auto beta = ge::es::Add(normOutput, Scalar(builder, 1.0F, outputDtype));
        Describe(beta, c.dims, outputDtype);
        outputs.push_back(beta);
    }
    if (c.repeat != 0) {
        auto input = c.repeat == 2 ? y : x;
        auto q = ge::es::Square(input);
        Describe(q, c.dims, c.dtype);
        auto m = ge::es::ReduceMean(q, builder.CreateVector(Dims{-1}), true);
        Describe(m, keep, c.dtype);
        auto s = ge::es::Add(m, builder.CreateScalar(c.eps));
        Describe(s, keep, c.dtype);
        auto r = ge::es::Rsqrt(s);
        Describe(r, keep, c.dtype);
        auto n = ge::es::Mul(input, r);
        Describe(n, c.dims, c.dtype);
        auto y2 = ge::es::Mul(n, gamma);
        Describe(y2, c.dims, c.dtype);
        if (c.repeat == 2) {
            outputs = {y2};
        } else {
            outputs.push_back(y2);
        }
    }
    if (c.side) {
        // Shared statistics used by another operation remain outside the scope.
        auto side = ge::es::Add(scale, Scalar(builder, 0.0F, computeType));
        Describe(side, statisticShape, computeType);
        outputs.push_back(side);
    }
    if (c.exposeRstd) {
        auto rstd = scale;
        if (c.root >= 4) {
            rstd = c.exposedReciprocal ? ge::es::Reciprocal(scale) :
                                         Divide(Scalar(builder, c.exposedNumerator, computeType), scale, c.realDiv);
        }
        Describe(rstd, statisticShape, computeType);
        outputs.push_back(rstd);
        if (c.duplicateRstd) {
            outputs.push_back(rstd);
        }
    }
    if (c.sharedCast) {
        outputs.push_back(computeX);
    }
    if (c.sharedConstant) {
        outputs.push_back(epsilon);
    }
    auto squareNode = *square.GetProducer();
    auto meanNode = *mean.GetProducer();
    auto rootCastNode = *computeX.GetProducer();
    auto gammaNode = *originalGamma.GetProducer();
    if (c.rootStream) {
        ge::AscendString label("retained_root_stream");
        if (rootCastNode.SetAttr("_user_stream_label", label) != ge::GRAPH_SUCCESS) {
            return nullptr;
        }
    }
    ge::GraphPtr graph = builder.BuildAndReset(outputs);
    if (!graph) {
        return nullptr;
    }
    for (auto node : graph->GetAllNodes()) {
        for (int i = 0; i < static_cast<int>(node.GetInputsSize()); ++i) {
            auto source = node.GetInDataNodesAndPortIndexs(i);
            if (source.first) {
                ge::TensorDesc desc;
                if (source.first->GetOutputDesc(source.second, desc) != ge::GRAPH_SUCCESS ||
                    node.UpdateInputDesc(i, desc) != ge::GRAPH_SUCCESS) {
                    return nullptr;
                }
            }
        }
    }
    if (c.control && graph->AddControlEdge(squareNode, meanNode) != ge::GRAPH_SUCCESS) {
        return nullptr;
    }
    if (c.rootControl != 0) {
        auto from = c.rootControl == 1 || c.rootControl == 4 ? gammaNode :
                    c.rootControl == 5                       ? squareNode :
                                                               rootCastNode;
        auto to = c.rootControl == 1 ? rootCastNode : c.rootControl == 2 || c.rootControl == 5 ? gammaNode : squareNode;
        if (graph->AddControlEdge(from, to) != ge::GRAPH_SUCCESS) {
            return nullptr;
        }
    }
    return graph;
}

// The UT image need not contain the target's RmsNorm ops-info. Stub only the
// capability query so matching, infer-shape and subgraph replacement use real GE.
class TestRmsNormFusionPass : public ops::RmsNormFusionPass {
public:
    explicit TestRmsNormFusionPass(bool supported = true) : supported_(supported) {}

    bool TargetChecked() const { return targetChecked_; }

protected:
    bool CheckOpSupported(const ge::GNode& node) const override
    {
        EXPECT_EQ(Type(node), "RmsNorm");
        targetChecked_ = true;
        return supported_;
    }

private:
    bool supported_;
    mutable bool targetChecked_ = false;
};

ge::Status RunPass(ops::RmsNormFusionPass& pass, ge::GraphPtr& graph)
{
    ge::CustomPassContext context;
    context.SetPassName("RmsNormFusionPass");
    // Exercise the framework entry and its reporting path, including reruns.
    return pass.Run(graph, context);
}

std::string Snapshot(const ge::Graph& graph)
{
    std::ostringstream stream;
    for (const auto& node : graph.GetAllNodes()) {
        ge::AscendString name;
        node.GetName(name);
        stream << name.GetString() << ':' << Type(node) << ';';
        for (size_t i = 0; i < node.GetInputsSize(); ++i) {
            const auto source = node.GetInDataNodesAndPortIndexs(i);
            if (source.first != nullptr) {
                source.first->GetName(name);
                stream << i << '<' << name.GetString() << ':' << source.second << ';';
            }
            ge::TensorDesc desc;
            node.GetInputDesc(i, desc);
            for (auto dim : desc.GetShape().GetDims()) {
                stream << dim << ',';
            }
            stream << desc.GetDataType() << ':' << desc.GetFormat() << ';';
        }
        for (size_t i = 0; i < node.GetOutputsSize(); ++i) {
            ge::TensorDesc desc;
            node.GetOutputDesc(i, desc);
            for (auto dim : desc.GetShape().GetDims()) {
                stream << dim << ',';
            }
            stream << desc.GetDataType() << ':' << desc.GetFormat() << ';';
        }
        for (const auto& control : node.GetInControlNodes()) {
            control->GetName(name);
            stream << '^' << name.GetString() << ';';
        }
    }
    return stream.str();
}
std::set<std::pair<std::string, std::string>> ControlEdges(const ge::Graph& graph)
{
    std::set<std::pair<std::string, std::string>> edges;
    for (const auto& node : graph.GetAllNodes()) {
        ge::AscendString target;
        node.GetName(target);
        for (const auto& source : node.GetInControlNodes()) {
            ge::AscendString name;
            source->GetName(name);
            edges.emplace(name.GetString(), target.GetString());
        }
    }
    return edges;
}

std::map<std::string, std::string> StreamLabels(const ge::Graph& graph)
{
    std::map<std::string, std::string> labels;
    for (auto node : graph.GetAllNodes()) {
        if (!node.HasAttr("_user_stream_label")) {
            continue;
        }
        ge::AscendString name;
        ge::AscendString label;
        node.GetName(name);
        if (node.GetAttr("_user_stream_label", label) == ge::GRAPH_SUCCESS) {
            labels.emplace(name.GetString(), label.GetString());
        }
    }
    return labels;
}

class RmsNormFusionPassTest : public testing::TestWithParam<Case> {
protected:
    static void SetUpTestSuite()
    {
        // Exercise the real GE infer-shape registry and graph rewriter.
        // The repository generates a subset ES library. GEInitialize also loads
        // installed passes, which require the full SDK's ES symbol set.
        const char* cann = std::getenv("ASCEND_HOME_PATH");
        ASSERT_NE(cann, nullptr);
        for (const char* library : {"libes_math.so", "libes_nn.so"}) {
            const std::string path = std::string(cann) + "/lib64/" + library;
            // Older SDKs ship only es_math; missing builders come from this SDK's gen_esb.
            if (!std::ifstream(path).good() && std::string(library) == "libes_nn.so") {
                continue;
            }
            ASSERT_NE(dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL), nullptr) << dlerror();
        }
        const char* soc = std::getenv("RMSNORM_TEST_SOC");
        const char* target = soc == nullptr ? "Ascend950PR_9579" : soc;
        const std::map<ge::AscendString, ge::AscendString> options{{"ge.socVersion", target}};
        auto& manager = fe::PlatformInfoManager::Instance();
        fe::PlatFormInfos platform;
        restoreOptions = manager.GetPlatformInfoWithOutSocVersion(platform, savedOptions) == ge::GRAPH_SUCCESS;
        ASSERT_EQ(ge::GEInitialize(options), ge::SUCCESS);

        // Other op_graph suites also select a platform on this process-wide manager.
        // Set the target used by CheckNpuArch as well as GE's compilation target.
        fe::OptionalInfos selected;
        ASSERT_TRUE(selected.Init());
        selected.SetSocVersion(target);
        manager.SetOptionalCompilationInfo(selected);
        ASSERT_EQ(manager.GetPlatformInfoWithOutSocVersion(platform, selected), ge::GRAPH_SUCCESS) << target;
        std::string arch;
        ASSERT_TRUE(platform.GetPlatformRes("version", "NpuArch", arch)) << target;
        if (soc == nullptr) {
            ASSERT_EQ(arch, "3510") << target;
        }
    }

    static void TearDownTestSuite()
    {
        EXPECT_EQ(ge::GEFinalize(), ge::SUCCESS);
        if (restoreOptions) {
            fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(savedOptions);
        }
    }

private:
    static inline fe::OptionalInfos savedOptions;
    static inline bool restoreOptions = false;
};

bool ExpectedPlatformEnabled()
{
    const char* expected = std::getenv("RMSNORM_TEST_EXPECT_PLATFORM_ENABLED");
    return expected == nullptr || std::string(expected) == "1";
}

void CheckInterleavedCallbacks()
{
    Case a;
    a.name = "base_a";
    Case b;
    b.name = "base_b";
    b.dims = {3, 32};
    b.gamma = {32};
    b.eps = 0.125F;
    auto graphA = Build(a);
    auto graphB = Build(b);
    ASSERT_NE(graphA, nullptr);
    ASSERT_NE(graphB, nullptr);
    TestRmsNormFusionPass pass;

    for (int index : {0, 1, 0}) {
        auto& graph = index == 0 ? graphA : graphB;
        const auto status = RunPass(pass, graph);
        EXPECT_TRUE(status == ge::SUCCESS || status == ge::GRAPH_NOT_CHANGED);
        size_t count = 0;
        for (const auto& node : graph->GetAllNodes()) {
            if (Type(node) != "RmsNorm") {
                continue;
            }
            ++count;
            float epsilon = -1;
            ASSERT_EQ(node.GetAttr("epsilon", epsilon), ge::GRAPH_SUCCESS);
            EXPECT_EQ(epsilon, index == 0 ? a.eps : b.eps);
            ge::TensorDesc desc;
            ASSERT_EQ(node.GetOutputDesc(0, desc), ge::GRAPH_SUCCESS);
            EXPECT_EQ(desc.GetShape().GetDims(), index == 0 ? a.dims : b.dims);
        }
        EXPECT_EQ(count, ExpectedPlatformEnabled() ? 1U : 0U);
    }
}

void CheckExecutionAttributes(const std::string& name, ge::DataType dtype)
{
    Case c;
    c.dtype = dtype;
    c.lowPrecision = 1;
    c.typedConstants = true;
    c.eps = 0.125F;
    auto graph = Build(c);
    ASSERT_NE(graph, nullptr);
    int index = 0;
    for (auto node : graph->GetAllNodes()) {
        const auto type = Type(node);
        if (type == "Data" || type == "NetOutput" || type == "Const" || type == "Constant") {
            continue;
        }
        if (name == "stream_partial" && index++ != 0) {
            continue;
        }
        if (name == "stream_wrong_type") {
            int64_t invalid = 1;
            ASSERT_EQ(node.SetAttr("_user_stream_label", invalid), ge::GRAPH_SUCCESS);
        } else {
            const char* label = name == "stream_conflict" && index++ % 2 ? "stream_b" : "stream_a";
            ge::AscendString value(label);
            ASSERT_EQ(node.SetAttr("_user_stream_label", value), ge::GRAPH_SUCCESS);
        }
    }
    const auto before = Snapshot(*graph);
    TestRmsNormFusionPass pass;
    const bool valid = name == "stream_same" && dtype == ge::DT_FLOAT && ExpectedPlatformEnabled();
    ASSERT_EQ(RunPass(pass, graph), valid ? ge::SUCCESS : ge::GRAPH_NOT_CHANGED);
    if (!valid) {
        EXPECT_EQ(Snapshot(*graph), before);
    } else {
        size_t norms = 0;
        size_t arithmetic = 0;
        for (const auto& node : graph->GetAllNodes()) {
            const auto type = Type(node);
            if (type != "RmsNorm" && type != "Cast" && type != "Mul") {
                continue;
            }
            ge::AscendString label;
            ASSERT_EQ(node.GetAttr("_user_stream_label", label), ge::GRAPH_SUCCESS);
            EXPECT_STREQ(label.GetString(), "stream_a");
            norms += type == "RmsNorm";
            ++arithmetic;
        }
        EXPECT_EQ(norms, 1U);
        EXPECT_EQ(arithmetic, 1U);
    }
    EXPECT_EQ(RunPass(pass, graph), ge::GRAPH_NOT_CHANGED);
}

void CheckKeepdimsAttributes(const Case& c)
{
    auto graph = Build(c);
    ASSERT_NE(graph, nullptr);
    bool changed = false;
    for (auto node : graph->GetAllNodes()) {
        const auto type = Type(node);
        if (type != "ReduceMean" && type != "ReduceSum" && type != "ReduceMeanD" && type != "ReduceSumD") {
            continue;
        }
        if (c.name.find("wrong_type") != std::string::npos) {
            int64_t value = 1;
            ASSERT_EQ(node.SetAttr("keep_dims", value), ge::GRAPH_SUCCESS);
        } else {
            bool value = false;
            ASSERT_EQ(node.SetAttr("keep_dims", value), ge::GRAPH_SUCCESS);
        }
        changed = true;
    }
    ASSERT_TRUE(changed);
    // Keep the original descriptors: shape alone must not bypass the attribute.
    const auto before = Snapshot(*graph);
    TestRmsNormFusionPass pass;
    EXPECT_EQ(RunPass(pass, graph), ge::GRAPH_NOT_CHANGED);
    EXPECT_EQ(Snapshot(*graph), before);
}

void CheckZeroMeanFactor(float epsilon)
{
    // 1 / 2^25 rounds to zero in FP16. Merely checking FP16 X is insufficient:
    // this valid mixed-dtype graph no longer computes a nonzero mean for X != 0.
    constexpr int64_t width = int64_t{1} << 25;
    ge::es::EsGraphBuilder builder("zero_mean_factor");
    auto x = builder.CreateInput(0, "x", ge::DT_FLOAT16, ge::FORMAT_ND, Dims{2, width});
    auto gamma = builder.CreateInput(1, "gamma", ge::DT_FLOAT16, ge::FORMAT_ND, Dims{width});
    auto compute = ge::es::Cast(x, ge::DT_FLOAT);
    auto sum = ge::es::ReduceSum(ge::es::Square(compute), builder.CreateVector(Dims{-1}), true);
    auto mean = ge::es::Mul(sum, Scalar(builder, 0.0F, ge::DT_FLOAT16));
    auto stable = ge::es::Add(mean, builder.CreateScalar(epsilon));
    auto normalized = ge::es::RealDiv(compute, ge::es::Sqrt(stable));
    auto y = ge::es::Mul(ge::es::Cast(normalized, ge::DT_FLOAT16), gamma);
    ge::GraphPtr graph = builder.BuildAndReset({y});
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(ge::GeUtils::InferShape(*graph, {ge::Shape({2, width}), ge::Shape({width})}), ge::SUCCESS);
    const auto before = Snapshot(*graph);
    TestRmsNormFusionPass pass;
    EXPECT_EQ(RunPass(pass, graph), ge::GRAPH_NOT_CHANGED);
    EXPECT_EQ(Snapshot(*graph), before);
}

void UseAttributeAxes(ge::Graph& graph, const Case& c)
{
    // Use the same node builder as the LayerNorm UT for unsupported IR forms.
    // Keep a valid graph so rejection exercises the operator-type guard.
    for (auto node : graph.GetAllNodes()) {
        if (Type(node) != (c.mean == 0 ? "ReduceMean" : "ReduceSum")) {
            continue;
        }
        auto source = node.GetInDataNodesAndPortIndexs(0);
        ASSERT_NE(source.first, nullptr);
        auto replacement = ge::es::CompliantNodeBuilder(&graph)
                               .OpType(c.mean == 0 ? "ReduceMeanD" : "ReduceSumD")
                               .Name("attribute_axes_reduction")
                               .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                               .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                               .Build();
        auto axes = c.axes;
        bool keepdims = true;
        ASSERT_EQ(replacement.SetAttr("axes", axes), ge::GRAPH_SUCCESS);
        ASSERT_EQ(replacement.SetAttr("keep_dims", keepdims), ge::GRAPH_SUCCESS);
        ge::TensorDesc desc;
        ASSERT_EQ(node.GetOutputDesc(0, desc), ge::GRAPH_SUCCESS);
        ASSERT_EQ(replacement.UpdateOutputDesc(0, desc), ge::GRAPH_SUCCESS);
        ASSERT_EQ(ge::es::AddEdgeAndUpdatePeerDesc(graph, *source.first, source.second, replacement, 0),
                  ge::GRAPH_SUCCESS);
        for (auto consumer : node.GetOutDataNodesAndPortIndexs(0)) {
            ASSERT_NE(consumer.first, nullptr);
            ASSERT_EQ(graph.RemoveEdge(node, 0, *consumer.first, consumer.second), ge::GRAPH_SUCCESS);
            ASSERT_EQ(ge::es::AddEdgeAndUpdatePeerDesc(graph, replacement, 0, *consumer.first, consumer.second),
                      ge::GRAPH_SUCCESS);
        }
        ASSERT_EQ(graph.RemoveNode(node), ge::GRAPH_SUCCESS);
        return;
    }
    FAIL() << "fixture has no reduction";
}

void CheckScheduledCandidates(const Case& c)
{
    ge::es::EsGraphBuilder builder(c.name.c_str());
    auto x = builder.CreateInput(0, "x", ge::DT_FLOAT, ge::FORMAT_ND, {2, 64});
    auto gamma = builder.CreateInput(1, "gamma", ge::DT_FLOAT, ge::FORMAT_ND, {64});
    auto axes = builder.CreateVector(Dims{-1});
    auto factor = builder.CreateScalar(1.0F / 64.0F);
    auto epsilon = builder.CreateScalar(1e-5F);
    std::vector<Tensor> outputs;
    if (c.coverage == "prefix") {
        // These share X/gamma with every valid candidate, but cannot match.
        // Replacing another candidate must not invalidate their opaque inputs.
        for (int i = 0; i < 128; ++i) {
            outputs.push_back(ge::es::Mul(x, gamma));
        }
    }
    auto previous = x;
    size_t expectedNorms = 0;
    for (int i = 0; i < c.repeat; ++i) {
        auto input = c.coverage == "chain" ? previous : x;
        // Several Mul anchors belong to one candidate. Some are removed before
        // the scheduler reaches them, including in the dual-output boundary.
        auto square = ge::es::Mul(input, input);
        auto mean = ge::es::Mul(ge::es::ReduceSum(square, axes, true), factor);
        auto rstd = ge::es::Rsqrt(ge::es::Add(mean, epsilon));
        auto normalized = ge::es::Mul(input, rstd);
        auto y = ge::es::Mul(normalized, gamma);
        previous = y;
        if (c.coverage == "rejected" || (c.coverage == "mixed" && i % 2 == 0)) {
            outputs.push_back(ge::es::Add(y, builder.CreateScalar(0.125F)));
        } else if (c.coverage == "shared" && i % 2 == 0) {
            // Shared normalization cannot be consumed by either replacement.
            outputs.push_back(y);
            outputs.push_back(ge::es::Mul(normalized, builder.CreateScalar(2.0F)));
        } else {
            ++expectedNorms;
            outputs.push_back(y);
            if (c.coverage == "dual") {
                outputs.push_back(rstd);
            } else if (c.coverage == "duplicate") {
                outputs.push_back(y);
            }
        }
    }
    ge::GraphPtr graph = builder.BuildAndReset(outputs);
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(ge::GeUtils::InferShape(*graph, {ge::Shape({2, 64}), ge::Shape({64})}), ge::SUCCESS);
    std::vector<ge::TensorDesc> outputDescs;
    for (const auto& node : graph->GetDirectNode()) {
        if (Type(node) == "NetOutput") {
            for (size_t i = 0; i < node.GetInputsSize(); ++i) {
                ge::TensorDesc desc;
                ASSERT_EQ(node.GetInputDesc(i, desc), ge::GRAPH_SUCCESS);
                outputDescs.push_back(desc);
            }
        }
    }
    ASSERT_EQ(outputDescs.size(), outputs.size());
    const auto before = Snapshot(*graph);
    expectedNorms = ExpectedPlatformEnabled() ? expectedNorms : 0U;
    TestRmsNormFusionPass pass;
    EXPECT_EQ(RunPass(pass, graph), expectedNorms == 0 ? ge::GRAPH_NOT_CHANGED : ge::SUCCESS);
    size_t norms = 0;
    for (const auto& node : graph->GetDirectNode()) {
        norms += Type(node) == "RmsNorm" ? 1U : 0U;
        if (Type(node) == "NetOutput") {
            ASSERT_EQ(node.GetInputsSize(), outputDescs.size());
            for (size_t i = 0; i < outputDescs.size(); ++i) {
                ge::TensorDesc desc;
                ASSERT_EQ(node.GetInputDesc(i, desc), ge::GRAPH_SUCCESS);
                EXPECT_EQ(desc.GetShape().GetDims(), outputDescs[i].GetShape().GetDims());
                EXPECT_EQ(desc.GetDataType(), outputDescs[i].GetDataType());
                EXPECT_EQ(desc.GetFormat(), outputDescs[i].GetFormat());
                EXPECT_NE(node.GetInDataNodesAndPortIndexs(i).first, nullptr);
            }
        }
    }
    EXPECT_EQ(norms, expectedNorms);
    const auto after = Snapshot(*graph);
    if (expectedNorms == 0) {
        EXPECT_EQ(after, before);
    }
    EXPECT_EQ(RunPass(pass, graph), ge::GRAPH_NOT_CHANGED);
    EXPECT_EQ(Snapshot(*graph), after);
}

TEST_P(RmsNormFusionPassTest, Contract)
{
    const auto& c = GetParam();
    if (c.name.rfind("schedule_", 0) == 0) {
        CheckScheduledCandidates(c);
        return;
    }
    if (c.name.rfind("keepdims_attr_", 0) == 0) {
        CheckKeepdimsAttributes(c);
        return;
    }
    if (c.name.rfind("stream_", 0) == 0) {
        CheckExecutionAttributes(c.name, c.dtype);
        return;
    }
    if (c.name == "null_graph") {
        ge::GraphPtr graph;
        TestRmsNormFusionPass pass;
        EXPECT_EQ(RunPass(pass, graph), ge::FAILED);
        return;
    }
    if (c.name == "interleaved_callbacks") {
        CheckInterleavedCallbacks();
        return;
    }
    if (c.name == "zero_mean_factor") {
        CheckZeroMeanFactor(0.0F);
        CheckZeroMeanFactor(1e-5F);
        return;
    }
    auto graph = Build(c);
    ASSERT_NE(graph, nullptr);
    if (c.name.rfind("excluded_reduce_d_", 0) == 0) {
        ASSERT_NO_FATAL_FAILURE(UseAttributeAxes(*graph, c));
    }
    if (c.inferSource) {
        // These regression graphs must be valid according to the real SDK,
        // rather than passing solely because the fixture populated descriptors.
        std::vector<ge::Shape> inputs{ge::Shape(c.dims), ge::Shape(c.gamma)};
        if (c.residual) {
            inputs.emplace_back(c.residualShape.empty() ? c.dims : c.residualShape);
        }
        ASSERT_EQ(ge::GeUtils::InferShape(*graph, inputs), ge::GRAPH_SUCCESS);
    }
    const auto originalCount = graph->GetAllNodes().size();
    const auto originalGraph = Snapshot(*graph);
    const auto originalControls = ControlEdges(*graph);
    const auto originalStreams = StreamLabels(*graph);
    std::vector<ge::TensorDesc> originalOutputs;
    for (const auto& node : graph->GetAllNodes()) {
        if (Type(node) != "NetOutput") {
            continue;
        }
        for (size_t i = 0; i < node.GetInputsSize(); ++i) {
            ge::TensorDesc desc;
            ASSERT_EQ(node.GetInputDesc(i, desc), ge::GRAPH_SUCCESS);
            originalOutputs.push_back(desc);
        }
    }

    bool sourceReducesAll = true;
    if (c.inferSource && c.mean != 0 && c.axes.empty() && !c.noopWithEmptyAxes) {
        for (const auto& node : graph->GetAllNodes()) {
            if (Type(node) != "ReduceSum") {
                continue;
            }
            ge::TensorDesc desc;
            ASSERT_EQ(node.GetOutputDesc(0, desc), ge::GRAPH_SUCCESS);
            sourceReducesAll = desc.GetShape().GetDims() == Dims(c.dims.size(), 1);
            // Older SDKs infer empty-axis ReduceSum as a no-op. Such a graph
            // must remain unchanged; it is not a valid all-axis RMSNorm chain.
            ASSERT_TRUE(sourceReducesAll || desc.GetShape().GetDims() == c.dims);
        }
    }
    const bool expectedFusion = c.valid && sourceReducesAll && ExpectedPlatformEnabled();
    TestRmsNormFusionPass pass(c.targetSupported);
    ASSERT_EQ(RunPass(pass, graph), expectedFusion ? ge::SUCCESS : ge::GRAPH_NOT_CHANGED);
    if (!c.targetSupported && ExpectedPlatformEnabled()) {
        EXPECT_TRUE(pass.TargetChecked());
    }
    size_t norms = 0;
    for (const auto& node : graph->GetAllNodes()) {
        if (Type(node) != "RmsNorm") {
            continue;
        }
        ++norms;
        float epsilon = -1;
        ASSERT_EQ(node.GetAttr("epsilon", epsilon), ge::GRAPH_SUCCESS);
        EXPECT_EQ(epsilon, c.eps);
        ge::TensorDesc y;
        ASSERT_EQ(node.GetOutputDesc(0, y), ge::GRAPH_SUCCESS);
        EXPECT_EQ(y.GetShape().GetDims(), c.dims);
        EXPECT_EQ(y.GetDataType(), ge::DT_FLOAT);
        EXPECT_EQ(y.GetFormat(), ge::FORMAT_ND);
        size_t begin = c.axes.empty() ? 0 : c.dims.size();
        for (auto axis : c.axes) {
            begin = std::min(begin, static_cast<size_t>(axis < 0 ? axis + c.dims.size() : axis));
        }
        if (std::find(c.dims.begin() + begin, c.dims.end(), -1) != c.dims.end()) {
            // Runtime gamma must be derived from X, never from external gamma:
            // its shape defines the normalized axes of the native RmsNorm.
            const auto gamma = node.GetInDataNodesAndPortIndexs(1);
            ASSERT_NE(gamma.first, nullptr);
            ASSERT_EQ(Type(*gamma.first), "Fill");
            const auto slice = gamma.first->GetInDataNodesAndPortIndexs(0);
            ASSERT_NE(slice.first, nullptr);
            ASSERT_EQ(Type(*slice.first), "Slice");
            const auto shape = slice.first->GetInDataNodesAndPortIndexs(0);
            ASSERT_NE(shape.first, nullptr);
            ASSERT_EQ(Type(*shape.first), "Shape");
            const auto shapeInput = shape.first->GetInDataNodesAndPortIndexs(0);
            const auto normInput = node.GetInDataNodesAndPortIndexs(0);
            ASSERT_NE(shapeInput.first, nullptr);
            ASSERT_NE(normInput.first, nullptr);
            ge::AscendString shapeSource;
            ge::AscendString normSource;
            ASSERT_EQ(shapeInput.first->GetName(shapeSource), ge::GRAPH_SUCCESS);
            ASSERT_EQ(normInput.first->GetName(normSource), ge::GRAPH_SUCCESS);
            EXPECT_EQ(shapeSource, normSource);
            EXPECT_EQ(shapeInput.second, normInput.second);
        }
    }
    EXPECT_EQ(norms, expectedFusion ? (c.repeat == 0 ? 1U : 2U) : 0U);
    size_t rstdOutputBegin = originalOutputs.size();
    size_t rstdOutputEnd = originalOutputs.size();
    if (c.exposeRstd) {
        // The builder appends rstd before any shared Cast or constant outputs.
        rstdOutputEnd -= (c.sharedCast ? 1U : 0U) + (c.sharedConstant ? 1U : 0U);
        const size_t rstdOutputs = c.duplicateRstd ? 2U : 1U;
        ASSERT_GE(rstdOutputEnd, rstdOutputs);
        rstdOutputBegin = rstdOutputEnd - rstdOutputs;
    }
    size_t outputIndex = 0;
    for (const auto& node : graph->GetAllNodes()) {
        if (Type(node) != "NetOutput") {
            continue;
        }
        for (size_t i = 0; i < node.GetInputsSize(); ++i) {
            ASSERT_LT(outputIndex, originalOutputs.size());
            ge::TensorDesc desc;
            ASSERT_EQ(node.GetInputDesc(i, desc), ge::GRAPH_SUCCESS);
            EXPECT_EQ(desc.GetDataType(), originalOutputs[outputIndex].GetDataType());
            EXPECT_EQ(desc.GetShape().GetDims(), originalOutputs[outputIndex].GetShape().GetDims());
            EXPECT_EQ(desc.GetFormat(), originalOutputs[outputIndex].GetFormat());
            if (expectedFusion && outputIndex >= rstdOutputBegin && outputIndex < rstdOutputEnd) {
                const auto producer = node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
                ASSERT_NE(producer.first, nullptr);
                EXPECT_EQ(Type(*producer.first), "RmsNorm");
                EXPECT_EQ(producer.second, 1);
            }
            ++outputIndex;
        }
    }
    EXPECT_EQ(outputIndex, originalOutputs.size());
    if (c.exposeRstd && c.control && expectedFusion) {
        // Both ends of this redundant internal control edge were replaced.
        EXPECT_TRUE(ControlEdges(*graph).empty());
    } else {
        EXPECT_EQ(ControlEdges(*graph), originalControls);
    }
    if (c.rootStream) {
        ASSERT_EQ(originalStreams.size(), 1U);
        EXPECT_EQ(StreamLabels(*graph), originalStreams);
    }
    if (!expectedFusion) {
        EXPECT_EQ(graph->GetAllNodes().size(), originalCount);
        EXPECT_EQ(Snapshot(*graph), originalGraph);
    }
    // A second invocation must not duplicate a successful fusion.
    EXPECT_EQ(RunPass(pass, graph), ge::GRAPH_NOT_CHANGED);
}

std::vector<Case> Cases()
{
    std::vector<Case> cases;
    auto add = [&](const char* name, bool valid = true) -> Case& {
        auto& c = cases.emplace_back();
        c.name = name;
        c.valid = valid;
        return c;
    };
    add("rsqrt");
    auto& reciprocal = add("sqrt_reciprocal");
    reciprocal.square = 1;
    reciprocal.mean = 1;
    reciprocal.factorType = ge::DT_FLOAT16;
    reciprocal.root = 1;
    reciprocal.swaps = 15;
    reciprocal.dims = {2, 4, 16};
    reciprocal.axes = {-2, -1};
    reciprocal.gamma = {4, 16};
    reciprocal.tail = 2;

    auto& oneOver = add("one_over_sqrt_dual_output");
    oneOver.root = 2;
    oneOver.square = 2;
    oneOver.mean = 2;
    oneOver.factorType = ge::DT_BF16;
    oneOver.realDiv = true;
    oneOver.exposeRstd = true;
    oneOver.duplicateRstd = true;

    auto& inversePower = add("inverse_power_fp16");
    inversePower.root = 3;
    inversePower.dtype = ge::DT_FLOAT16;
    inversePower.casts = 1;

    auto& divide = add("divide_sqrt_bf16_dual_output");
    divide.root = 4;
    divide.dtype = ge::DT_BF16;
    divide.casts = 2;
    divide.exposeRstd = true;
    divide.control = true;

    auto& power = add("divide_power_fp32_gamma");
    power.root = 5;
    power.dtype = ge::DT_FLOAT16;
    power.casts = 3;
    power.eps = 0;

    add("dynamic_prefix").dims = {-1, 64};
    auto& dynamic = add("dynamic_tail");
    dynamic.dims = {2, -1};
    dynamic.gamma = {-1};
    dynamic.root = 5;
    add("reshape_gamma").gamma = {1, 64};
    add("keepdims_attr_false", false);
    add("non_tail_axis", false).axes = {0};
    add("runtime_epsilon", false).dynamicEpsilon = true;
    add("all_low_precision", false).dtype = ge::DT_FLOAT16;
    auto& bias = add("scalar_bias", false);
    bias.beta = true;
    bias.betaAdds = true;
    auto& residual = add("residual_consumer", false);
    residual.residual = true;
    residual.root = 4;
    residual.dtype = ge::DT_FLOAT16;
    residual.casts = 1;
    add("shared_statistics", false).side = true;
    auto& factor = add("rounded_half_reciprocal", false);
    factor.mean = 1;
    factor.dims = {2, 7};
    factor.gamma = {7};
    factor.divisor = 7;
    factor.factorType = ge::DT_FLOAT16;
    auto& denominator = add("inexact_fp32_divisor", false);
    denominator.mean = 2;
    denominator.dims = {2, 16777217};
    denominator.gamma = {16777217};
    denominator.divisor = 16777216.0F;
    add("zero_mean_factor", false);
    add("subnormal_epsilon", false).eps = std::numeric_limits<float>::denorm_min();
    add("wrong_gamma", false).gamma = {32};
    auto& exponent = add("inexact_square", false);
    exponent.square = 2;
    exponent.exponent = 1.9999F;
    add("unsupported_target", false).targetSupported = false;
    add("null_graph", false);
    add("interleaved_callbacks");
    auto& chain = add("schedule_chain");
    chain.coverage = "chain";
    chain.repeat = 3;
    add("stream_same");
    add("stream_conflict", false);
    add("excluded_reduce_d_mean", false);
    auto& gamma = add("dependent_gamma", false);
    gamma.casts = 1;
    gamma.square = 2;
    gamma.root = 2;
    gamma.gammaSource = 1;
    gamma.realDiv = true;
    gamma.inferSource = true;
    return cases;
}
INSTANTIATE_TEST_SUITE_P(Variants, RmsNormFusionPassTest, testing::ValuesIn(Cases()),
                         [](const testing::TestParamInfo<Case>& info) { return info.param.name; });
} // namespace

#else
TEST(RmsNormFusionPassTest, RequiresConstructiblePassContext)
{
    GTEST_SKIP() << "CANN 8.5 context construction requires the GE framework callback";
}
#endif
