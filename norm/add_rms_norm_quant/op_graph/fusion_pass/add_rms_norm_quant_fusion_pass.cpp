/** Copyright (c) 2026 Huawei Technologies Co., Ltd. See LICENSE for licensing details. */
#include "add_rms_norm_quant_fusion_pass.h"
#include <functional>
#include <map>
#include <numeric>
#include <memory>
#include <string>
#include <vector>
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "arch_utils.h"
#include "common/inc/error_util.h"
#include "ge/ge_utils.h"
#include <set>

namespace ops {
namespace {
using ge::GNode;
using ge::TensorDesc;
using ge::es::EsGraphBuilder;
using ge::es::EsTensorHolder;
using namespace ge::fusion;

const std::string kPass = "AddRmsNormQuantFusionPass";
const std::set<std::string> kSupportedNpuArchs = {"3510", "9201", "9202"};

constexpr int64_t kLastDimAlignment = 16;
constexpr float kDefaultEpsilon = 1e-5f;
constexpr size_t kZeroPointsBit = 0U;
constexpr size_t kReshapeBit = 1U;
constexpr size_t kPatternBitNum = 2U;
constexpr size_t kPatternConfigNum = 1U << kPatternBitNum;

bool IsBitSet(size_t config, size_t bit) { return ((config >> bit) & 1U) != 0U; }

std::string Name(const GNode& node)
{
    ge::AscendString value;
    return node.GetName(value) == ge::GRAPH_SUCCESS && value.GetString() ? value.GetString() : "";
}
TensorDesc InputDesc(const GNode& node, int port)
{
    TensorDesc desc;
    node.GetInputDesc(port, desc);
    return desc;
}
TensorDesc OutputDesc(const GNode& node, int port)
{
    TensorDesc desc;
    node.GetOutputDesc(port, desc);
    return desc;
}
bool SetDesc(const EsTensorHolder& tensor, const TensorDesc& desc)
{
    auto* node = tensor.GetProducer();
    const auto port = tensor.GetProducerOutIndex();
    if (node->UpdateOutputDesc(port, desc) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(kPass.c_str(), "Cannot update output descriptor: node=%s, port=%d.", Name(*node).c_str(),
                  static_cast<int>(port));
        return false;
    }
    return true;
}
bool SetQuantInputDescs(const ge::es::AddRmsNormQuantOutput& fused)
{
    auto* node = fused.y1.GetProducer();
    if (node == nullptr) {
        OPS_LOG_E(kPass.c_str(), "Cannot synchronize input descriptors: producer is null.");
        return false;
    }
    // Visit all input slots, including optional inputs.
    constexpr int kQuantInputCount = 8;
    for (int port = 0; port < kQuantInputCount; ++port) {
        const auto source = node->GetInDataNodesAndPortIndexs(port);
        if (source.first == nullptr) {
            continue;
        }
        TensorDesc desc;
        if (source.first->GetOutputDesc(source.second, desc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kPass.c_str(), "Cannot read source descriptor: node=%s, output=%d, target=%s, input=%d.",
                      Name(*source.first).c_str(), source.second, Name(*node).c_str(), port);
            return false;
        }
        if (node->UpdateInputDesc(port, desc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kPass.c_str(), "Cannot update input descriptor: node=%s, port=%d.", Name(*node).c_str(), port);
            return false;
        }
    }
    return true;
}
PatternUniqPtr MakePattern(size_t config)
{
    const bool has_reshape = IsBitSet(config, kReshapeBit);
    const bool has_zero_points = IsBitSet(config, kZeroPointsBit);
    EsGraphBuilder builder(kPass.c_str());
    int index = 0;
    auto input = [&]() { return builder.CreateInput(index++); };
    auto x1 = input(), x2 = input(), gamma = input();
    auto norm = ge::es::AddRmsNorm(x1, x2, gamma);
    auto y = has_reshape ? ge::es::Reshape(norm.y, input()) : norm.y;
    auto scales = input();
    auto zero_points = has_zero_points ? input() : EsTensorHolder(nullptr);
    auto quant = ge::es::Quantize(y, scales, zero_points, "torch.qint8", -1);
    NodeIo norm_capture{*norm.y.GetProducer(), 0}, quant_capture{*quant.GetProducer(), 0};
    auto graph = builder.BuildAndReset({quant, norm.x});
    if (!graph) {
        OPS_LOG_E(kPass.c_str(), "Cannot build pattern: config=%zu.", config);
        return nullptr;
    }
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor(norm_capture);
    pattern->CaptureTensor(quant_capture);
    return pattern;
}
bool Capture(const MatchResult& match, GNode& norm, GNode& quant)
{
    NodeIo io;
    if (match.GetCapturedTensor(0, io) != ge::SUCCESS) {
        OPS_LOG_E(kPass.c_str(), "Cannot capture AddRmsNorm: capture index=0.");
        return false;
    }
    norm = io.node;
    if (match.GetCapturedTensor(1, io) != ge::SUCCESS) {
        OPS_LOG_E(kPass.c_str(), "Cannot capture Quantize: capture index=1.");
        return false;
    }
    quant = io.node;
    return true;
}
int64_t Destination(const GNode& quant)
{
    ge::AscendString dtype("torch.qint8");
    quant.GetAttr("dtype", dtype);
    static const std::map<std::string, int64_t> types{{"torch.qint8", ge::DT_INT8},
                                                      {"torch.hifloat8", ge::DT_HIFLOAT8},
                                                      {"torch.float8_e4m3fn", ge::DT_FLOAT8_E4M3FN},
                                                      {"torch.float8_e5m2", ge::DT_FLOAT8_E5M2}};
    auto found = types.find(dtype.GetString() ? dtype.GetString() : "");
    return found == types.end() ? ge::DT_UNDEFINED : found->second;
}
int64_t Elements(const TensorDesc& desc)
{
    auto dims = desc.GetShape().GetDims();
    return std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());
}
bool CheckNorm(const GNode& norm)
{
    const auto x1 = InputDesc(norm, 0);
    const auto dims = x1.GetShape().GetDims();
    const auto dtype = x1.GetDataType();
    if ((dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) || dims.empty() || dims.back() % kLastDimAlignment != 0) {
        OPS_LOG_D(kPass.c_str(), "guard_rejected : x1 dtype or alignment.");
        return false;
    }
    if (norm.GetOutDataNodesAndPortIndexs(0).size() != 1) {
        OPS_LOG_D(kPass.c_str(), "guard_rejected : y must have one consumer connection.");
        return false;
    }
    return true;
}
bool CheckQuant(const GNode& norm, const GNode& quant)
{
    if (Destination(quant) == ge::DT_UNDEFINED) {
        OPS_LOG_D(kPass.c_str(), "guard_rejected : unsupported quantization destination.");
        return false;
    }
    const auto gamma_size = Elements(InputDesc(norm, 2));
    if (Elements(InputDesc(quant, 1)) != gamma_size) {
        OPS_LOG_D(kPass.c_str(), "guard_rejected : scales size differs from gamma.");
        return false;
    }
    if (quant.GetInDataNodesAndPortIndexs(2).first != nullptr) {
        TensorDesc zero_points;
        if (quant.GetInputDesc(2, zero_points) != ge::GRAPH_SUCCESS) {
            OPS_LOG_D(kPass.c_str(), "guard_rejected : cannot read zero_points descriptor.");
            return false;
        }
        if (!zero_points.GetShape().GetDims().empty() && Elements(zero_points) != gamma_size) {
            OPS_LOG_D(kPass.c_str(), "guard_rejected : zero_points size differs from gamma.");
            return false;
        }
    }
    return true;
}
} // namespace
std::vector<PatternUniqPtr> AddRmsNormQuantFusionPass::Patterns()
{
    if (!ops::CheckNpuArch(kPass, kSupportedNpuArchs)) {
        return {};
    }
    std::vector<PatternUniqPtr> patterns;
    for (size_t config = 0; config < kPatternConfigNum; ++config) {
        patterns.push_back(MakePattern(config));
    }
    return patterns;
}
bool AddRmsNormQuantFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match)
{
    GNode norm, quant;
    if (!match) {
        OPS_LOG_E(kPass.c_str(), "Cannot check requirements: match is null.");
        return false;
    }
    if (!Capture(*match, norm, quant)) {
        return false;
    }
    if (!norm.GetOutDataNodesAndPortIndexs(1).empty()) {
        OPS_LOG_D(kPass.c_str(), "guard_rejected : rstd has consumers.");
        return false;
    }
    if (!CheckNorm(norm) || !CheckQuant(norm, quant)) {
        return false;
    }
    OPS_LOG_D(kPass.c_str(), "guard_passed : %s.", Name(norm).c_str());
    return true;
}
ge::fusion::GraphUniqPtr AddRmsNormQuantFusionPass::Replacement(const std::unique_ptr<MatchResult>& match)
{
    GNode norm, quant;
    if (!match) {
        OPS_LOG_E(kPass.c_str(), "Cannot build replacement: match is null.");
        return nullptr;
    }
    if (!Capture(*match, norm, quant)) {
        return nullptr;
    }
    auto boundary = match->ToSubgraphBoundary();
    if (!boundary) {
        OPS_LOG_E(kPass.c_str(), "Cannot obtain subgraph boundary: node=%s.", Name(norm).c_str());
        return nullptr;
    }
    EsGraphBuilder builder(kPass.c_str());
    std::vector<SubgraphInput> boundary_inputs;
    if (boundary->GetAllInputs(boundary_inputs) != ge::SUCCESS) {
        OPS_LOG_E(kPass.c_str(), "Cannot enumerate boundary inputs: node=%s.", Name(norm).c_str());
        return nullptr;
    }
    auto inputs = builder.CreateInputs(boundary_inputs.size());
    std::vector<ge::Shape> input_shapes;
    input_shapes.reserve(boundary_inputs.size());
    for (size_t i = 0; i < boundary_inputs.size(); ++i) {
        const auto consumers = boundary_inputs[i].GetAllInputs();
        if (consumers.empty()) {
            OPS_LOG_E(kPass.c_str(), "Cannot initialize boundary input %zu: no consumer.", i);
            return nullptr;
        }
        const auto desc = InputDesc(consumers[0].node, consumers[0].index);
        if (!SetDesc(inputs[i], desc)) {
            return nullptr;
        }
        if (inputs[i].GetProducer()->UpdateInputDesc(0, desc) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(kPass.c_str(), "Cannot update boundary Data input descriptor: index=%zu.", i);
            return nullptr;
        }
        input_shapes.emplace_back(desc.GetShape());
    }
    auto input = [&](const GNode& node, int port) -> EsTensorHolder {
        for (size_t i = 0; i < boundary_inputs.size(); ++i) {
            for (const auto& consumer : boundary_inputs[i].GetAllInputs()) {
                if (consumer.index == port && Name(consumer.node) == Name(node)) {
                    return inputs[i];
                }
            }
        }
        return nullptr;
    };
    float epsilon = kDefaultEpsilon;
    int64_t axis = -1;
    norm.GetAttr("epsilon", epsilon);
    quant.GetAttr("axis", axis);
    const auto dst = Destination(quant);
    auto fused = ge::es::AddRmsNormQuant(input(norm, 0), input(norm, 1), input(norm, 2), input(quant, 1), nullptr,
                                         input(quant, 2), nullptr, nullptr, axis, epsilon, true, dst);
    if (!SetQuantInputDescs(fused)) {
        OPS_LOG_E(kPass.c_str(), "replacement_rejected : cannot set input descriptors.");
        return nullptr;
    }
    auto y2_desc = OutputDesc(*fused.y2.GetProducer(), fused.y2.GetProducerOutIndex());
    y2_desc.SetDataType(static_cast<ge::DataType>(dst));
    if (!SetDesc(fused.y2, y2_desc)) {
        return nullptr;
    }
    if (!SetDesc(fused.y1, OutputDesc(quant, 0)) || !SetDesc(fused.x, OutputDesc(norm, 2))) {
        return nullptr;
    }
    OPS_LOG_D(kPass.c_str(), "replacement : %s.", Name(norm).c_str());
    const std::vector<EsTensorHolder> outputs{fused.y1, fused.x};
    auto graph = builder.BuildAndReset(outputs);
    if (!graph) {
        OPS_LOG_E(kPass.c_str(), "Cannot build replacement graph.");
        return nullptr;
    }
    const auto status = ge::GeUtils::InferShape(*graph, input_shapes);
    if (status != ge::SUCCESS) {
        OPS_LOG_E(kPass.c_str(), "Cannot infer replacement graph, status=%u.", static_cast<unsigned int>(status));
        return nullptr;
    }
    return graph;
}
REG_FUSION_PASS(AddRmsNormQuantFusionPass).Stage(ge::CustomPassStage::kAfterBuiltinFusionPass);
} // namespace ops
