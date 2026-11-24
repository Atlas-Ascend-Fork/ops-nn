/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rms_norm_fusion_pass.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "error_util.h"
#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/ge_utils.h"
#include "norm/common/op_graph/fusion_report_utils.h"
#include "op_graph/arch_utils.h"

namespace ops {
namespace {
constexpr const char* kPassName = "RmsNormFusionPass";
constexpr size_t kMaxRank = 8;
// A5, MC62 (1952), and A6 DT/PR, using the shared compilation-target check.
const std::set<std::string> kSupportedNpuArchs = {"3510", "5102", "9201", "9202"};
using Edge = std::pair<ge::GNodePtr, int32_t>;
using Dims = std::vector<int64_t>;
using ge::fusion::NodeIo;

std::string Name(const ge::GNode& node)
{
    ge::AscendString value;
    return node.GetName(value) == ge::GRAPH_SUCCESS && value.GetString() != nullptr ? value.GetString() : "";
}

std::string Type(const ge::GNode& node)
{
    ge::AscendString value;
    return node.GetType(value) == ge::GRAPH_SUCCESS && value.GetString() != nullptr ? value.GetString() : "";
}

bool Is(const Edge& edge, const char* type, size_t inputs)
{
    return edge.first != nullptr && edge.second == 0 && Type(*edge.first) == type &&
           edge.first->GetInputsSize() == inputs && edge.first->GetOutputsSize() == 1;
}

Edge Input(const Edge& edge, int32_t index = 0)
{
    return edge.first == nullptr ? Edge{} : edge.first->GetInDataNodesAndPortIndexs(index);
}

bool Same(const Edge& lhs, const Edge& rhs)
{
    return lhs.first != nullptr && rhs.first != nullptr && lhs.second == rhs.second && !Name(*lhs.first).empty() &&
           Name(*lhs.first) == Name(*rhs.first);
}

bool Describe(const Edge& edge, ge::TensorDesc& desc)
{
    return edge.first != nullptr && edge.second >= 0 &&
           edge.first->GetOutputDesc(edge.second, desc) == ge::GRAPH_SUCCESS;
}

bool Half(ge::DataType type) { return type == ge::DT_FLOAT16 || type == ge::DT_BF16; }

bool Floating(ge::DataType type) { return type == ge::DT_FLOAT || Half(type); }

struct Match {
    Edge x;
    Edge computeX;
    Edge gamma;
    Edge output;
    Edge reduction;
    Edge scale;
    Edge rstdOutput;
    std::map<std::string, ge::GNode> nodes;
    std::vector<Edge> moments;
    Dims axes;
    Dims gammaShape;
    ge::DataType computeType = ge::DT_UNDEFINED;
    ge::DataType normalizedType = ge::DT_UNDEFINED;
    float epsilon = 0.0F;
    double factor = 0.0;
    bool promoteInput = false;
    bool dynamicReduction = false;
    bool sum = false;
    bool divide = false;
    bool downcast = false;
    bool retainGamma = false;
    bool normalizeDivide = false;
    bool normalizeRealDiv = false;

    bool Add(const Edge& edge)
    {
        if (edge.first == nullptr || edge.second != 0 || Name(*edge.first).empty()) {
            return false;
        }
        nodes.emplace(Name(*edge.first), *edge.first);
        return true;
    }
};

template <typename T>
T Load(const uint8_t* data)
{
    T result;
    std::memcpy(&result, data, sizeof(result));
    return result;
}

double Decode16(uint16_t bits, ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        const uint32_t wide = static_cast<uint32_t>(bits) << 16;
        return Load<float>(reinterpret_cast<const uint8_t*>(&wide));
    }
    const int exponent = (bits >> 10) & 31;
    const int mantissa = bits & 1023;
    const double magnitude = exponent == 31 ? std::numeric_limits<double>::infinity() :
                             exponent == 0  ? std::ldexp(static_cast<double>(mantissa), -24) :
                                              std::ldexp(static_cast<double>(1024 + mantissa), exponent - 25);
    return (bits & 32768) != 0 ? -magnitude : magnitude;
}

bool Constant(const Edge& edge, ge::Tensor& tensor, Match& match)
{
    return (Is(edge, "Const", 0) || Is(edge, "Constant", 0)) &&
           edge.first->GetAttr("value", tensor) == ge::GRAPH_SUCCESS && match.Add(edge);
}

bool Scalar(const Edge& edge, double& value, Match& match)
{
    ge::Tensor tensor;
    if (!Constant(edge, tensor, match)) {
        return false;
    }
    const auto desc = tensor.GetTensorDesc();
    const auto dims = desc.GetShape().GetDims();
    if (tensor.GetData() == nullptr || desc.GetFormat() != ge::FORMAT_ND ||
        !std::all_of(dims.begin(), dims.end(), [](int64_t dim) { return dim == 1; })) {
        return false;
    }
    const auto dtype = desc.GetDataType();
    if (dtype == ge::DT_FLOAT && tensor.GetSize() == sizeof(float)) {
        value = Load<float>(tensor.GetData());
    } else if (Half(dtype) && tensor.GetSize() == sizeof(uint16_t)) {
        value = Decode16(Load<uint16_t>(tensor.GetData()), dtype);
    } else {
        return false;
    }
    return std::isfinite(value);
}

bool Exact(const Edge& edge, double expected, Match& match)
{
    double value = 0;
    return Scalar(edge, value, match) && value == expected;
}

bool Integers(const Edge& edge, Dims& values, Match& match)
{
    ge::Tensor tensor;
    if (!Constant(edge, tensor, match)) {
        return false;
    }
    const auto desc = tensor.GetTensorDesc();
    const auto shape = desc.GetShape().GetDims();
    const auto dtype = desc.GetDataType();
    if (shape.size() > 1 || (dtype != ge::DT_INT32 && dtype != ge::DT_INT64)) {
        return false;
    }
    const int64_t count = shape.empty() ? 1 : shape[0];
    const size_t width = dtype == ge::DT_INT32 ? sizeof(int32_t) : sizeof(int64_t);
    if (count < 0 || (count != 0 && tensor.GetData() == nullptr) || tensor.GetSize() % width != 0 ||
        static_cast<uint64_t>(count) != tensor.GetSize() / width) {
        return false;
    }
    values.clear();
    for (int64_t i = 0; i < count; ++i) {
        const auto* p = tensor.GetData() + static_cast<size_t>(i) * width;
        values.push_back(dtype == ge::DT_INT32 ? Load<int32_t>(p) : Load<int64_t>(p));
    }
    return true;
}

bool Root(Edge edge, bool inverse, Edge& stable, Match& match)
{
    match.moments.push_back(edge);
    if ((inverse && Is(edge, "Rsqrt", 1)) || (!inverse && Is(edge, "Sqrt", 1))) {
        stable = Input(edge);
        return match.Add(edge);
    }
    if (Is(edge, "Pow", 2) && Exact(Input(edge, 1), inverse ? -0.5 : 0.5, match)) {
        stable = Input(edge);
        return match.Add(edge);
    }
    if (inverse && Is(edge, "Reciprocal", 1)) {
        return match.Add(edge) && Root(Input(edge), false, stable, match);
    }
    if (inverse && (Is(edge, "Div", 2) || Is(edge, "RealDiv", 2)) && Exact(Input(edge), 1.0, match)) {
        return match.Add(edge) && Root(Input(edge, 1), false, stable, match);
    }
    return false;
}

bool Epsilon(Edge edge, Edge& mean, Match& match)
{
    match.moments.push_back(edge);
    // E1 only: the actual GE node must be Add with one constant input.
    // Keep both input orders; AddV2 and attribute-based Adds are out of scope.
    if (!Is(edge, "Add", 2)) {
        return false;
    }
    for (int32_t port = 0; port < 2; ++port) {
        auto candidate = match;
        double epsilon = 0;
        if (Scalar(Input(edge, port), epsilon, candidate) && epsilon >= 0 &&
            static_cast<double>(static_cast<float>(epsilon)) == epsilon) {
            candidate.epsilon = static_cast<float>(epsilon);
            mean = Input(edge, 1 - port);
            candidate.Add(edge);
            match = std::move(candidate);
            return true;
        }
    }
    return false;
}

bool Mean(Edge edge, Edge& square, Match& match)
{
    match.moments.push_back(edge);
    const bool div = Is(edge, "Div", 2) || Is(edge, "RealDiv", 2);
    if (div || Is(edge, "Mul", 2)) {
        bool found = false;
        for (int32_t p = 1; p >= (div ? 1 : 0); --p) {
            auto candidate = match;
            if (Scalar(Input(edge, p), candidate.factor, candidate)) {
                candidate.sum = true;
                candidate.divide = div;
                candidate.Add(edge);
                edge = Input(edge, 1 - p);
                match = std::move(candidate);
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
        match.moments.push_back(edge);
    }
    if (!Is(edge, match.sum ? "ReduceSum" : "ReduceMean", 2)) {
        return false;
    }
    // The supported contract keeps every reduced dimension. Neither direct
    // keep_dims=false broadcasting nor a dimension-restoration chain is in scope.
    bool keepdims = false;
    if (edge.first->GetAttr("keep_dims", keepdims) != ge::GRAPH_SUCCESS || !keepdims) {
        OPS_LOG_D(kPassName, "skip: keep_dims must be true");
        return false;
    }
    if (!Integers(Input(edge, 1), match.axes, match)) {
        return false;
    }
    if (match.axes.empty()) {
        // An empty axis tensor is not unconditionally a full reduction.
        bool noop = true;
        if (edge.first->GetAttr("noop_with_empty_axes", noop) != ge::GRAPH_SUCCESS || noop) {
            return false;
        }
    }
    match.reduction = edge;
    square = Input(edge);
    return match.Add(edge);
}

bool Square(const Edge& edge, Edge& x, Match& match)
{
    if (Is(edge, "Square", 1) || (Is(edge, "Mul", 2) && Same(Input(edge), Input(edge, 1))) ||
        (Is(edge, "Pow", 2) && Exact(Input(edge, 1), 2.0, match))) {
        x = Input(edge);
        return match.Add(edge);
    }
    return false;
}

bool Lift(Edge& edge, Match& match)
{
    if (!Is(edge, "Cast", 1)) {
        return true;
    }
    ge::TensorDesc src;
    ge::TensorDesc dst;
    // The replacement consumes this Cast's output, so its conversion remains
    // upstream. Non-exact conversions must share the same Cast in both paths.
    if (!Describe(Input(edge), src) || !Describe(edge, dst) ||
        (!Floating(src.GetDataType()) && !Same(edge, match.computeX)) ||
        (dst.GetDataType() != ge::DT_FLOAT && !(match.promoteInput && Half(dst.GetDataType()))) ||
        src.GetShape().GetDims() != dst.GetShape().GetDims() || src.GetFormat() != ge::FORMAT_ND ||
        dst.GetFormat() != ge::FORMAT_ND) {
        return false;
    }
    match.Add(edge);
    edge = Input(edge);
    return true;
}

bool Product(const Dims& dims, int64_t& size)
{
    size = 1;
    for (int64_t dim : dims) {
        if (dim <= 0 || size > std::numeric_limits<int64_t>::max() / dim) {
            return false;
        }
        size *= dim;
    }
    return true;
}

bool SameDesc(const ge::TensorDesc& a, const ge::TensorDesc& b)
{
    return a.GetShape().GetDims() == b.GetShape().GetDims() && a.GetDataType() == b.GetDataType() &&
           a.GetFormat() == b.GetFormat();
}

bool Shapes(Match& match)
{
    ge::TensorDesc x;
    ge::TensorDesc gamma;
    ge::TensorDesc y;
    ge::TensorDesc scale;
    if (!Describe(match.x, x) || !Describe(match.gamma, gamma) || !Describe(match.output, y) ||
        !Describe(match.scale, scale) || x.GetFormat() != ge::FORMAT_ND || gamma.GetFormat() != ge::FORMAT_ND ||
        y.GetFormat() != ge::FORMAT_ND) {
        return false;
    }
    const auto dims = x.GetShape().GetDims();
    if (dims.empty() || dims.size() > kMaxRank ||
        std::any_of(dims.begin(), dims.end(), [](int64_t d) { return d == 0 || d < -1; })) {
        return false;
    }
    if (match.axes.empty()) {
        // Mean has proved noop_with_empty_axes=false for this case.
        for (size_t i = 0; i < dims.size(); ++i) {
            match.axes.push_back(static_cast<int64_t>(i));
        }
    }
    for (auto& axis : match.axes) {
        if (axis < -static_cast<int64_t>(dims.size()) || axis >= static_cast<int64_t>(dims.size())) {
            return false;
        }
        if (axis < 0) {
            axis += static_cast<int64_t>(dims.size());
        }
    }
    std::sort(match.axes.begin(), match.axes.end());
    match.axes.erase(std::unique(match.axes.begin(), match.axes.end()), match.axes.end());
    const size_t begin = static_cast<size_t>(match.axes.front());
    for (size_t i = begin; i < dims.size(); ++i) {
        // Reducing a singleton axis does not add elements or change their order.
        if (!std::binary_search(match.axes.begin(), match.axes.end(), static_cast<int64_t>(i)) && dims[i] != 1) {
            return false;
        }
    }
    match.gammaShape = Dims(dims.begin() + begin, dims.end());
    int64_t count = 0;
    match.dynamicReduction = std::find(match.gammaShape.begin(), match.gammaShape.end(), -1) != match.gammaShape.end();
    // A1 derives its divisor from the runtime shape. A2 still needs a statically
    // proven N, and static products must retain their overflow check.
    if ((match.dynamicReduction && match.sum) || (!match.dynamicReduction && !Product(match.gammaShape, count))) {
        return false;
    }
    if (match.sum) {
        if (std::find(dims.begin(), dims.end(), -1) != dims.end()) {
            return false;
        }
        // RmsNorm computes its coefficient in FP32. Compare the actual source
        // constant without rounding the expected coefficient to its storage dtype.
        const float divisor = static_cast<float>(count);
        if (match.factor != (match.divide ? divisor : 1.0F / divisor)) {
            return false;
        }
        // Division must use the exact element count, even above FP32 integer precision.
        if (match.divide && static_cast<uint64_t>(divisor) != static_cast<uint64_t>(count)) {
            return false;
        }
    }
    const auto gammaDims = gamma.GetShape().GetDims();
    if (std::any_of(gammaDims.begin(), gammaDims.end(), [](int64_t d) { return d < -1 || d == 0; })) {
        return false;
    }
    // Older Mul inference may accept incompatible static broadcast dimensions.
    for (size_t i = 0; i < std::min(dims.size(), gammaDims.size()); ++i) {
        const auto xDim = dims[dims.size() - 1 - i];
        const auto gammaDim = gammaDims[gammaDims.size() - 1 - i];
        if (xDim > 1 && gammaDim > 1 && xDim != gammaDim) {
            return false;
        }
    }
    Dims padded(gammaDims.size() <= dims.size() ? dims.size() - gammaDims.size() : 0, 1);
    padded.insert(padded.end(), gammaDims.begin(), gammaDims.end());
    bool nativeGamma = padded.size() == dims.size();
    for (size_t i = 0; i < dims.size(); ++i) {
        nativeGamma = nativeGamma && padded[i] == (i < begin ? 1 : dims[i]);
    }
    Dims statisticShape = dims;
    std::fill(statisticShape.begin() + begin, statisticShape.end(), 1);
    ge::TensorDesc reductionDesc;
    if (!Describe(match.reduction, reductionDesc) || reductionDesc.GetShape().GetDims() != statisticShape ||
        scale.GetShape().GetDims() != statisticShape) {
        return false;
    }
    for (const auto& moment : match.moments) {
        ge::TensorDesc desc;
        if (!Describe(moment, desc) || desc.GetDataType() != match.computeType ||
            desc.GetShape().GetDims() != statisticShape) {
            return false;
        }
    }
    if (!Floating(match.normalizedType) || !Floating(gamma.GetDataType())) {
        return false;
    }
    // Mul promotes differing FP32/FP16/BF16 inputs to FP32. The original X,
    // normalized Cast target and gamma are independent dtype boundaries.
    const auto outputType = match.normalizedType == gamma.GetDataType() ? match.normalizedType : ge::DT_FLOAT;
    // General/dynamic gamma broadcasting belongs to the retained Mul, not to
    // RmsNorm's axis contract. Its inferred output is checked before rewriting.
    match.retainGamma = match.dynamicReduction || !nativeGamma || match.normalizedType != ge::DT_FLOAT ||
                        gamma.GetDataType() != ge::DT_FLOAT;
    return y.GetDataType() == outputType;
}

bool ResidualAdd(const Edge& output, const Edge& consumer, size_t prefixRank)
{
    const auto& add = *consumer.first;
    if (add.GetInputsSize() != 2 || (consumer.second != 0 && consumer.second != 1)) {
        return false;
    }
    ge::TensorDesc y;
    ge::TensorDesc residual;
    if (!Describe(output, y) || !Describe(add.GetInDataNodesAndPortIndexs(1 - consumer.second), residual) ||
        !SameDesc(y, residual)) {
        return false;
    }
    const auto dims = y.GetShape().GetDims();
    if (prefixRank > dims.size() || std::any_of(dims.begin(), dims.end(), [](int64_t d) { return d <= 0; })) {
        return false;
    }
    // A nontrivial prefix distinguishes a full elementwise residual from a
    // channel-only beta. Rank-1/all-singleton-prefix cases remain ambiguous.
    return std::any_of(dims.begin(), dims.begin() + prefixRank, [](int64_t d) { return d > 1; });
}

bool OutputConsumers(const Match& match)
{
    ge::TensorDesc x;
    if (!Describe(match.x, x)) {
        return false;
    }
    const size_t prefixRank = x.GetShape().GetDims().size() - match.gammaShape.size();
    std::vector<Edge> pending{match.output};
    std::set<std::string> visited;
    while (!pending.empty()) {
        const auto output = pending.back();
        pending.pop_back();
        if (!visited.insert(Name(*output.first)).second) {
            continue;
        }
        for (const auto& consumer : output.first->GetOutDataNodesAndPortIndexs(output.second)) {
            if (consumer.first == nullptr) {
                return false;
            }
            const auto type = Type(*consumer.first);
            // Output-side Adds is a scalar beta, also when reached through
            // Casts. Epsilon Adds is inside the statistical chain, not here.
            if (type == "Adds") {
                return false;
            }
            if (type == "Add" || type == "AddV2") {
                if (Half(match.computeType)) {
                    OPS_LOG_D(kPassName, "skip: residual with half-precision statistics: %s",
                              Name(*consumer.first).c_str());
                    return false;
                }
                if (!ResidualAdd(output, consumer, prefixRank)) {
                    OPS_LOG_D(kPassName, "skip: Add consumer is not a full elementwise residual: %s",
                              Name(*consumer.first).c_str());
                    return false;
                }
                // Keep these layout checks for diagnosing the precision risk.
                if (match.normalizeDivide) {
                    OPS_LOG_D(kPassName, "residual with division normalization: %s", Name(*match.output.first).c_str());
                }
                if (match.downcast) {
                    OPS_LOG_D(kPassName, "residual after low-precision Cast: %s", Name(*match.output.first).c_str());
                }
                // RmsNorm may round its statistics differently from the source
                // chain. An arbitrary residual can cancel y and amplify even
                // one FP32 ulp; R3 also has a counterexample without either of
                // the above layouts. A low Cast does not prove equal rounding.
                // Preserve the original computation until equivalence can be
                // proved; shape/dtype equality alone is not a precision proof.
                OPS_LOG_D(kPassName, "skip: unproven precision under residual cancellation: %s",
                          Name(*match.output.first).c_str());
                return false;
            }
            if (type == "Cast") {
                pending.push_back({consumer.first, 0});
            }
        }
    }
    return true;
}

bool RedundantControl(const ge::GNode& source, const ge::GNode& target, const Match& match)
{
    if (match.nodes.count(Name(source)) == 0 || match.nodes.count(Name(target)) == 0 || Type(source) == "Const" ||
        Type(source) == "Constant" || Type(target) == "Const" || Type(target) == "Constant") {
        return false;
    }
    // A direct data edge already requires source to complete before target.
    // Only accept a control edge that adds exactly the same dependency.
    for (size_t i = 0; i < target.GetInputsSize(); ++i) {
        const auto input = target.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        if (input.first != nullptr && Name(*input.first) == Name(source)) {
            return true;
        }
    }
    return false;
}

bool ExposedRstd(Match& match, const Edge& stable)
{
    for (const auto& consumer : match.scale.first->GetOutDataNodesAndPortIndexs(match.scale.second)) {
        if (consumer.first == nullptr) {
            return false;
        }
        if (match.nodes.count(Name(*consumer.first)) != 0) {
            continue;
        }
        if (!match.normalizeDivide) {
            if (Type(*consumer.first) != "NetOutput") {
                return false;
            }
            match.rstdOutput = match.scale;
            continue;
        }
        // R5/R6 normalize by RMS; an exported RSTD must be its reciprocal.
        // Prove the same epsilon-adjusted statistic, including the numerator 1.
        const Edge rstd{consumer.first, 0};
        Edge source;
        auto candidate = match;
        if ((match.rstdOutput.first != nullptr && !Same(match.rstdOutput, rstd)) ||
            !Root(rstd, true, source, candidate) || !Same(source, stable)) {
            return false;
        }
        const auto outputs = rstd.first->GetOutDataNodesAndPortIndexs(0);
        if (outputs.empty() || !std::all_of(outputs.begin(), outputs.end(), [](const Edge& output) {
                return output.first != nullptr && Type(*output.first) == "NetOutput";
            })) {
            return false;
        }
        candidate.rstdOutput = rstd;
        match = std::move(candidate);
    }
    return true;
}

bool Closed(const Match& match)
{
    if (!OutputConsumers(match)) {
        return false;
    }
    // General broadcasting must not capture an internal statistic or the
    // normalized result as an external gamma input. Constants and the retained
    // root Cast are valid shared inputs, outside the replaced normalization.
    const auto gammaType = Type(*match.gamma.first);
    if (match.nodes.count(Name(*match.gamma.first)) != 0 && gammaType != "Const" && gammaType != "Constant" &&
        !Same(match.gamma, match.computeX)) {
        return false;
    }
    for (const auto& item : match.nodes) {
        const auto& node = item.second;
        const bool retainedCast = item.first == Name(*match.computeX.first) && Is(match.computeX, "Cast", 1);
        // Candidate accepts only built-in arithmetic/shape primitives with fixed
        // arity. Function and control-flow nodes are outside those IR contracts.
        const auto safeControls = [&](const std::vector<ge::GNodePtr>& controls, bool incoming) {
            return std::all_of(controls.begin(), controls.end(), [&](const ge::GNodePtr& control) {
                if (control == nullptr) {
                    return false;
                }
                if (retainedCast && match.nodes.count(Name(*control)) == 0) {
                    return true;
                }
                return incoming ? RedundantControl(*control, node, match) : RedundantControl(node, *control, match);
            });
        };
        const auto inControls = node.GetInControlNodes();
        const auto outControls = node.GetOutControlNodes();
        // Retain external controls on the root Cast. Internal controls may be
        // absorbed only when an existing direct data edge proves redundancy.
        if (!safeControls(inControls, true) || !safeControls(outControls, false)) {
            OPS_LOG_D(kPassName, "skip: control dependency %s", item.first.c_str());
            return false;
        }
        const bool constant = Type(node) == "Const" || Type(node) == "Constant";
        if (constant) {
            // Statistical constants are outside the pattern boundary and may be shared.
        } else {
            ge::TensorDesc desc;
            if (node.GetOutputDesc(0, desc) != ge::GRAPH_SUCCESS || desc.GetFormat() != ge::FORMAT_ND ||
                !Floating(desc.GetDataType()) ||
                (Type(node) != "Cast" && Type(node) != "Mul" && desc.GetDataType() != match.computeType)) {
                OPS_LOG_D(kPassName, "skip: node dtype %s", item.first.c_str());
                return false;
            }
        }
        for (int32_t i = 0; i < static_cast<int32_t>(node.GetInputsSize()); ++i) {
            const auto source = node.GetInDataNodesAndPortIndexs(i);
            ge::TensorDesc src;
            ge::TensorDesc dst;
            if (!Describe(source, src) || node.GetInputDesc(i, dst) != ge::GRAPH_SUCCESS || !SameDesc(src, dst) ||
                (!Same(source, match.x) && !Same(source, match.gamma) && match.nodes.count(Name(*source.first)) == 0)) {
                OPS_LOG_D(kPassName, "skip: input descriptor %s:%d", item.first.c_str(), i);
                return false;
            }
        }
        if (item.first == Name(*match.output.first)) {
            continue;
        }
        if (constant) {
            continue;
        }
        // The root Cast is already an input boundary of the replacement.
        // Its other consumers remain connected to the original conversion.
        if (retainedCast) {
            continue;
        }
        for (const auto& consumer : node.GetOutDataNodesAndPortIndexs(0)) {
            if (consumer.first == nullptr || match.nodes.count(Name(*consumer.first)) == 0) {
                if (match.rstdOutput.first != nullptr && item.first == Name(*match.rstdOutput.first) &&
                    consumer.first != nullptr && Type(*consumer.first) == "NetOutput") {
                    continue;
                }
                OPS_LOG_D(kPassName, "skip: side consumer %s", item.first.c_str());
                return false;
            }
        }
    }
    return true;
}

// Preserve explicit execution-stream constraints when replacing the subgraph.
bool ExecutionCompatible(const Match& match)
{
    std::string label;
    bool initialized = false;
    for (const auto& item : match.nodes) {
        auto node = item.second;
        if (Type(node) == "Const" || Type(node) == "Constant" ||
            (item.first == Name(*match.computeX.first) && Is(match.computeX, "Cast", 1))) {
            continue;
        }
        ge::AscendString value;
        std::string current;
        if (node.HasAttr("_user_stream_label")) {
            if (node.GetAttr("_user_stream_label", value) != ge::GRAPH_SUCCESS || value.GetString() == nullptr) {
                return false;
            }
            current = value.GetString();
        }
        if (initialized && label != current) {
            return false;
        }
        initialized = true;
        label = current;
    }
    return true;
}

bool Candidate(const Edge& output, int32_t gammaPort, int32_t scalePort, Match& match)
{
    match.output = output;
    match.gamma = Input(output, gammaPort);
    Edge normalized = Input(output, 1 - gammaPort);
    match.Add(output);
    if (Is(normalized, "Cast", 1)) {
        ge::TensorDesc in;
        ge::TensorDesc out;
        if (!Describe(Input(normalized), in) || !Describe(normalized, out) || in.GetDataType() != ge::DT_FLOAT ||
            !Floating(out.GetDataType()) || in.GetShape().GetDims() != out.GetShape().GetDims()) {
            return false;
        }
        match.downcast = Half(out.GetDataType());
        match.normalizedType = out.GetDataType();
        match.Add(normalized);
        normalized = Input(normalized);
    }
    const bool divide = Is(normalized, "Div", 2) || Is(normalized, "RealDiv", 2);
    match.normalizeDivide = divide;
    match.normalizeRealDiv = Is(normalized, "RealDiv", 2);
    if ((!divide && !Is(normalized, "Mul", 2)) || (divide && scalePort != 1)) {
        return false;
    }
    ge::TensorDesc compute;
    if (!Describe(normalized, compute) || !Floating(compute.GetDataType())) {
        return false;
    }
    // T1 rounds the square, reduction and epsilon addition in the low dtype.
    // RmsNorm accumulates statistics in FP32; casting its rstd afterwards cannot
    // reproduce FP16 underflow/overflow or those rounding points. An opt-in
    // attribute cannot prove precision equivalence, so preserve this graph.
    if (Half(compute.GetDataType())) {
        return false;
    }
    match.computeType = compute.GetDataType();
    if (match.normalizedType == ge::DT_UNDEFINED) {
        match.normalizedType = match.computeType;
    }
    match.scale = Input(normalized, scalePort);
    Edge x = Input(normalized, 1 - scalePort);
    match.computeX = x;
    Edge stable;
    Edge mean;
    Edge square;
    Edge squareX;
    if (!match.Add(normalized) || !Root(match.scale, !divide, stable, match) || !Epsilon(stable, mean, match) ||
        !Mean(mean, square, match) || !Square(square, squareX, match)) {
        return false;
    }
    ge::TensorDesc xDesc;
    ge::TensorDesc squareDesc;
    ge::TensorDesc squareXDesc;
    if (!Describe(x, xDesc) || !Describe(square, squareDesc) || !Describe(squareX, squareXDesc) ||
        !SameDesc(xDesc, squareXDesc)) {
        return false;
    }
    ge::TensorDesc exponentDesc;
    // Preserve the shared low Cast, then make the Pow/Mul Promote-to-FP32
    // contract explicit in the replacement. Do not merge distinct rounding
    // paths or accept a low-precision square (T1).
    match.promoteInput = !divide && Half(xDesc.GetDataType()) && Same(x, squareX) && Is(square, "Pow", 2) &&
                         Describe(Input(square, 1), exponentDesc) && exponentDesc.GetDataType() == ge::DT_FLOAT &&
                         squareDesc.GetDataType() == match.computeType &&
                         squareDesc.GetShape().GetDims() == xDesc.GetShape().GetDims() &&
                         squareDesc.GetFormat() == xDesc.GetFormat();
    if ((!match.promoteInput && (!SameDesc(xDesc, squareDesc) || xDesc.GetDataType() != match.computeType)) ||
        !Lift(x, match) || !Lift(squareX, match) || !Same(x, squareX)) {
        return false;
    }
    match.x = x;
    if (!ExposedRstd(match, stable) || !Shapes(match)) {
        OPS_LOG_D(kPassName, "skip: axes, gamma or precision shape contract");
        return false;
    }
    // Zero is a supported epsilon attribute, including negative zero. Keep the
    // separate positive-subnormal guard; do not impose a positive lower bound
    // on the zero-epsilon path based on the input dtype or dynamic tail.
    if (match.epsilon > 0.0F && match.epsilon < std::numeric_limits<float>::min()) {
        OPS_LOG_D(kPassName, "skip: unsupported positive subnormal epsilon");
        return false;
    }
    if (!Closed(match) || !ExecutionCompatible(match)) {
        OPS_LOG_D(kPassName, "skip: control, shared statistics or descriptor contract");
        return false;
    }
    return true;
}

bool Inside(const Match& match, const ge::GNode& node)
{
    return match.nodes.count(Name(node)) != 0 && Type(node) != "Const" && Type(node) != "Constant" &&
           !(match.computeX.first != nullptr && Name(node) == Name(*match.computeX.first) && Type(node) == "Cast");
}

bool StatisticsBoundary(const Match& match, int32_t gammaPort, ge::fusion::SubgraphBoundary& boundary,
                        std::vector<Edge>& sources)
{
    // Export y and rstd from one replacement. Include the statistics so the
    // original reduction is removed, rather than computed a second time.
    sources = {match.computeX, match.gamma};
    std::vector<std::vector<NodeIo>> ports(sources.size());
    for (const auto& item : match.nodes) {
        if (!Inside(match, item.second)) {
            continue;
        }
        for (int32_t index = 0; index < static_cast<int32_t>(item.second.GetInputsSize()); ++index) {
            const auto source = item.second.GetInDataNodesAndPortIndexs(index);
            if (source.first == nullptr) {
                return false;
            }
            if (Inside(match, *source.first)) {
                continue;
            }
            size_t input = 0;
            if (item.first == Name(*match.output.first) && index == gammaPort) {
                input = 1;
            } else {
                while (input < sources.size() && !Same(sources[input], source)) {
                    ++input;
                }
                if (input == sources.size()) {
                    sources.push_back(source);
                    ports.emplace_back();
                }
            }
            ports[input].push_back({item.second, index});
        }
    }
    for (size_t index = 0; index < ports.size(); ++index) {
        if (ports[index].empty() || boundary.AddInput(index, ge::fusion::SubgraphInput(ports[index])) != ge::SUCCESS) {
            return false;
        }
    }
    return boundary.AddOutput(0, ge::fusion::SubgraphOutput({*match.output.first, 0})) == ge::SUCCESS &&
           boundary.AddOutput(1, ge::fusion::SubgraphOutput({*match.rstdOutput.first, 0})) == ge::SUCCESS;
}

bool Boundary(const Match& match, int32_t gammaPort, int32_t scalePort, ge::fusion::SubgraphBoundary& boundary,
              std::vector<Edge>& sources)
{
    if (match.rstdOutput.first != nullptr) {
        return StatisticsBoundary(match, gammaPort, boundary, sources);
    }
    Edge normalized = Input(match.output, 1 - gammaPort);
    // Candidate has already validated either a low Cast or an FP32 identity.
    if (Is(normalized, "Cast", 1)) {
        normalized = Input(normalized);
    }
    const std::vector<NodeIo> ports{
        {*normalized.first, 1 - scalePort}, {*match.output.first, gammaPort}, {*normalized.first, scalePort}};
    sources = {match.computeX, match.gamma, match.scale};
    for (size_t index = 0; index < ports.size(); ++index) {
        if (!Same(ports[index].node.GetInDataNodesAndPortIndexs(ports[index].index), sources[index]) ||
            boundary.AddInput(index, ge::fusion::SubgraphInput({ports[index]})) != ge::SUCCESS) {
            return false;
        }
    }
    return boundary.AddOutput(0, ge::fusion::SubgraphOutput({*match.output.first, 0})) == ge::SUCCESS;
}

std::unique_ptr<ge::Graph> MakeReplacement(const Match& match, const std::vector<Edge>& sources, ge::GNode& normNode)
{
    ge::es::EsGraphBuilder builder(kPassName);
    std::vector<ge::es::EsTensorHolder> inputs;
    std::vector<ge::TensorDesc> descs(sources.size());
    std::vector<ge::Shape> inputShapes;
    for (size_t index = 0; index < descs.size(); ++index) {
        if (!Describe(sources[index], descs[index])) {
            return nullptr;
        }
        inputShapes.push_back(descs[index].GetShape());
        const std::string inputName = "input" + std::to_string(index);
        inputs.push_back(builder.CreateInput(index, inputName.c_str(), descs[index].GetDataType(),
                                             descs[index].GetFormat(), descs[index].GetShape().GetDims()));
        auto* inputNode = inputs.back().GetProducer();
        if (inputNode == nullptr || inputNode->UpdateInputDesc(0, descs[index]) != ge::GRAPH_SUCCESS ||
            inputNode->UpdateOutputDesc(0, descs[index]) != ge::GRAPH_SUCCESS) {
            return nullptr;
        }
    }
    auto gamma = inputs[1];
    if (!match.retainGamma && descs[1].GetShape().GetDims() != match.gammaShape) {
        gamma = ge::es::Reshape(gamma, builder.CreateVector(match.gammaShape));
    }
    if (match.retainGamma) {
        // Fuse normalization in FP32, then retain the source Cast and gamma Mul.
        // A shape-sized Fill avoids allocating a large weight vector in the pass.
        // Slice the runtime X shape, not gamma's shape: only the reduction axes
        // determine RmsNorm's statistics. Keep the original gamma Mul outside.
        const auto gammaDims = match.dynamicReduction ?
                                   ge::es::Slice(
                                       ge::es::Shape(inputs[0], ge::DT_INT64),
                                       builder.CreateVector(Dims{match.axes.front()}),
                                       builder.CreateVector(Dims{static_cast<int64_t>(match.gammaShape.size())})) :
                                   builder.CreateVector(match.gammaShape);
        gamma = ge::es::Fill(gammaDims, builder.CreateScalar(1.0F));
    }
    auto computeX = inputs[0];
    if (match.promoteInput) {
        // Exact low-dtype identity separates the lossy root Cast from promotion.
        const auto one = ge::es::Cast(builder.CreateScalar(1.0F), descs[0].GetDataType());
        computeX = ge::es::Cast(ge::es::Mul(inputs[0], one), ge::DT_FLOAT);
    }
    const auto norm = ge::es::RmsNorm(computeX, gamma, match.epsilon);
    if (norm.y.GetProducer() == nullptr) {
        return nullptr;
    }
    normNode = *norm.y.GetProducer();
    // Some ES builders omit float attributes close to their IR default. Epsilon
    // is part of the computation, so write the proven value explicitly rather
    // than allowing InferShape to restore a different default value.
    float requestedEpsilon = match.epsilon;
    if (normNode.SetAttr("epsilon", requestedEpsilon) != ge::GRAPH_SUCCESS) {
        return nullptr;
    }
    auto y = norm.y;
    if (match.retainGamma) {
        // Division and multiplication differ on subnormal inputs even before
        // the low Cast. Keep the source division for R5/R6 in these layouts.
        if (match.normalizeDivide) {
            const auto rms = ge::es::Reciprocal(norm.rstd);
            y = match.normalizeRealDiv ? ge::es::RealDiv(inputs[0], rms) : ge::es::Div(inputs[0], rms);
        }
        if (match.downcast) {
            y = ge::es::Cast(y, match.normalizedType);
            if (descs[1].GetDataType() != match.normalizedType) {
                // Keep the original low rounding before gamma's type promotion.
                const auto one = ge::es::Cast(builder.CreateScalar(1.0F), match.normalizedType);
                y = ge::es::Mul(y, one);
            }
        }
        // Keep the original gamma shape, dtype and ordered input ports.
        y = Same(Input(match.output), match.gamma) ? ge::es::Mul(inputs[1], y) : ge::es::Mul(y, inputs[1]);
    }
    if (y.GetProducer() == nullptr) {
        return nullptr;
    }
    const auto outputNode = *y.GetProducer();
    std::vector<ge::es::EsTensorHolder> outputs{y};
    if (match.rstdOutput.first != nullptr) {
        outputs.push_back(norm.rstd);
    }
    auto replacement = builder.BuildAndReset(outputs);
    if (replacement == nullptr) {
        return nullptr;
    }
    const auto infer = ge::GeUtils::InferShape(*replacement, inputShapes);
    if (infer != ge::SUCCESS) {
        OPS_LOG_D(kPassName, "skip: replacement InferShape status=%u", infer);
        return nullptr;
    }
    float epsilon = 0.0F;
    if (normNode.GetAttr("epsilon", epsilon) != ge::GRAPH_SUCCESS || epsilon != match.epsilon) {
        OPS_LOG_D(kPassName, "skip: replacement epsilon differs");
        return nullptr;
    }
    ge::TensorDesc expected;
    ge::TensorDesc actual;
    if (!Describe(match.output, expected) || outputNode.GetOutputDesc(0, actual) != ge::GRAPH_SUCCESS ||
        !SameDesc(expected, actual)) {
        OPS_LOG_D(kPassName, "skip: replacement output descriptor differs");
        return nullptr;
    }
    if (match.rstdOutput.first != nullptr &&
        (!Describe(match.rstdOutput, expected) || normNode.GetOutputDesc(1, actual) != ge::GRAPH_SUCCESS ||
         !SameDesc(expected, actual))) {
        OPS_LOG_D(kPassName, "skip: replacement rstd descriptor differs");
        return nullptr;
    }
    return replacement;
}

// Only internal nodes and inspected output Casts can make a rejected candidate
// eligible after another replacement. Opaque X/gamma inputs and constants do
// not depend on their consumer lists, so sharing them must not trigger retries.
void WatchMatch(const Match& match, std::set<std::string>& watched)
{
    // Before X is proved, failures depend only on types, attributes and input
    // edges. Replacing a private RMSNorm chain cannot create a missing formula;
    // only the later consumer/closure checks need dependency tracking.
    if (match.x.first == nullptr) {
        return;
    }
    for (const auto& item : match.nodes) {
        if (Inside(match, item.second)) {
            watched.insert(item.first);
        }
    }
    for (const auto& edge : match.moments) {
        if (edge.first != nullptr) {
            watched.insert(Name(*edge.first));
        }
    }
    if (match.scale.first != nullptr) {
        watched.insert(Name(*match.scale.first));
    }
    std::vector<Edge> pending{match.output};
    std::set<std::string> visited;
    while (!pending.empty()) {
        const auto edge = pending.back();
        pending.pop_back();
        if (edge.first == nullptr || !visited.insert(Name(*edge.first)).second) {
            continue;
        }
        watched.insert(Name(*edge.first));
        for (const auto& consumer : edge.first->GetOutDataNodesAndPortIndexs(edge.second)) {
            if (consumer.first != nullptr && Type(*consumer.first) == "Cast") {
                pending.push_back({consumer.first, 0});
            }
        }
    }
}

// Capture names before replacing: the removed nodes' handles may expire. The
// adjacent nodes remain alive, but their input or consumer lists can change.
void ChangedNodes(const Match& match, std::set<std::string>& removed, std::set<std::string>& affected)
{
    for (const auto& item : match.nodes) {
        if (!Inside(match, item.second)) {
            continue;
        }
        removed.insert(item.first);
        affected.insert(item.first);
        const auto& node = item.second;
        for (size_t i = 0; i < node.GetInputsSize(); ++i) {
            const auto input = node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
            if (input.first != nullptr) {
                affected.insert(Name(*input.first));
            }
        }
        for (const auto& output : node.GetOutDataNodesAndPortIndexs(0)) {
            if (output.first != nullptr) {
                affected.insert(Name(*output.first));
            }
        }
        for (const auto& control : node.GetInControlNodes()) {
            if (control != nullptr) {
                affected.insert(Name(*control));
            }
        }
        for (const auto& control : node.GetOutControlNodes()) {
            if (control != nullptr) {
                affected.insert(Name(*control));
            }
        }
    }
}
} // namespace

bool RmsNormFusionPass::CheckOpSupported(const ge::GNode& node) const
{
    bool supported = false;
    ge::AscendString reason;
    const auto check = ge::GeUtils::CheckNodeSupportOnAicore(node, supported, reason);
    if (check != ge::SUCCESS || !supported) {
        OPS_LOG_D(kPassName, "skip: target checker status=%u, reason=%s", check,
                  reason.GetString() != nullptr ? reason.GetString() : "unavailable");
        return false;
    }
    return true;
}

ge::Status RmsNormFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& context)
{
    if (graph == nullptr) {
        return ge::GRAPH_NOT_CHANGED;
    }
    if (!ops::CheckNpuArch(kPassName, kSupportedNpuArchs)) {
        return ge::GRAPH_NOT_CHANGED;
    }
    // Keep only candidate anchors, never a Match or a replacement boundary.
    // GetDirectNode builds a graph-wide snapshot, so call it only once.
    std::vector<std::pair<std::string, ge::GNode>> candidates;
    std::set<size_t> pending;
    for (const auto& node : graph->GetDirectNode()) {
        if (Type(node) == "Mul") {
            pending.insert(candidates.size());
            candidates.emplace_back(Name(node), node);
        }
    }
    std::map<std::string, std::set<size_t>> waiting;
    std::vector<std::set<std::string>> watched(candidates.size());
    std::set<std::string> removed;
    bool changed = false;
    while (!pending.empty()) {
        const auto index = *pending.begin();
        pending.erase(pending.begin());
        for (const auto& name : watched[index]) {
            auto found = waiting.find(name);
            if (found != waiting.end()) {
                found->second.erase(index);
                if (found->second.empty()) {
                    waiting.erase(found);
                }
            }
        }
        watched[index].clear();
        if (removed.count(candidates[index].first) != 0) {
            continue;
        }
        const Edge output{std::make_shared<ge::GNode>(candidates[index].second), 0};
        if (!Is(output, "Mul", 2)) {
            continue;
        }
        bool replaced = false;
        for (int32_t gammaPort = 0; gammaPort < 2 && !replaced; ++gammaPort) {
            for (int32_t scalePort = 0; scalePort < 2 && !replaced; ++scalePort) {
                Match match;
                ge::fusion::SubgraphBoundary boundary;
                std::vector<Edge> sources;
                if (!Candidate(output, gammaPort, scalePort, match) ||
                    !Boundary(match, gammaPort, scalePort, boundary, sources)) {
                    WatchMatch(match, watched[index]);
                    continue;
                }
                ge::GNode normNode;
                auto replacement = MakeReplacement(match, sources, normNode);
                if (replacement == nullptr || !CheckOpSupported(normNode)) {
                    WatchMatch(match, watched[index]);
                    continue;
                }
                std::set<std::string> affected;
                ChangedNodes(match, removed, affected);
                const auto status = ops::fusion::ReplaceSubgraphWithReport(boundary, *replacement, context);
                if (status != ge::SUCCESS) {
                    return status;
                }
                replaced = true;
                changed = true;
                for (const auto& name : affected) {
                    const auto found = waiting.find(name);
                    if (found != waiting.end()) {
                        for (const auto candidate : found->second) {
                            if (removed.count(candidates[candidate].first) == 0) {
                                pending.insert(candidate);
                            }
                        }
                    }
                }
            }
        }
        if (!replaced) {
            for (const auto& name : watched[index]) {
                waiting[name].insert(index);
            }
        }
    }
    return changed ? ge::SUCCESS : ge::GRAPH_NOT_CHANGED;
}

REG_FUSION_PASS(RmsNormFusionPass).Stage(ge::CustomPassStage::kAfterInferShape);
} // namespace ops
