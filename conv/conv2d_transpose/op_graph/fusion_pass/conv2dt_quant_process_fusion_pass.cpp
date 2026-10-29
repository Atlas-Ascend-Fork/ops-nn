/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "conv2dt_quant_process_fusion_pass.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "conv/common/op_host/conv_npu_arch_resolver.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

namespace ops {
namespace {
constexpr const char* kPassName = "Conv2DTQuantProcessFusionPass";
constexpr const char* kConv2DT = "Conv2DTranspose";
constexpr const char* kCinCoutReverse = "quant_cin_cout_reverse";
constexpr int32_t kCubeXInputPort = 1;
constexpr int32_t kDequantXInputPort = 0;
constexpr int32_t kOutputPort = 0;
const std::map<std::string, NpuArch> kSupportSocList = {{"Ascend950", NpuArch::DAV_3510},
                                                        {"Ascend350", NpuArch::DAV_3510}};

bool IsFuseSoc()
{
    fe::PlatFormInfos platformInfo;
    fe::OptionalInfos optionalInfo;
    return fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) ==
               ge::GRAPH_SUCCESS &&
           conv_arch::IsCubeVectorFuseSoc(platformInfo);
}

bool HasType(const ge::GNodePtr& node, const ge::AscendString& type)
{
    ge::AscendString nodeType;
    return node != nullptr && node->GetType(nodeType) == ge::GRAPH_SUCCESS && nodeType == type;
}

ge::GNodePtr GetDequantConsumer(const ge::GNode& cube)
{
    for (const auto& consumer : cube.GetOutDataNodesAndPortIndexs(kOutputPort)) {
        if (consumer.second == kDequantXInputPort && HasType(consumer.first, ASCEND_DEQUANT)) {
            return consumer.first;
        }
    }
    return nullptr;
}
} // namespace

ge::Status Conv2DTQuantProcessFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    (void)passContext;
    OPS_LOG_I(kPassName, "[OPS_NN_CONV2DT_QUANT_PROCESS] pass invoked.");
    if (graph == nullptr || !graph->IsValid()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    if (!CheckSoc()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    bool changed = false;
    for (auto cube : graph->GetAllNodes()) {
        auto cubeNode = std::make_shared<ge::GNode>(cube);
        if (!HasType(cubeNode, kConv2DT)) {
            continue;
        }
        const auto quantInput = cube.GetInDataNodesAndPortIndexs(kCubeXInputPort);
        if (quantInput.second != kOutputPort || !HasType(quantInput.first, ASCEND_QUANT)) {
            continue;
        }
        const auto dequantNode = GetDequantConsumer(cube);
        if (dequantNode == nullptr) {
            continue;
        }
        std::vector<ge::GNodePtr> fusionNodes;
        std::vector<ge::GNodePtr> nodesToRemove;
        if (OptimizeMainProc(graph, quantInput.first, dequantNode, cubeNode, fusionNodes, nodesToRemove) !=
            ge::SUCCESS) {
            OPS_LOG_E(kPassName, "Quant process failed for Conv2DTranspose.");
            return ge::FAILED;
        }
        for (const auto& node : nodesToRemove) {
            if (node == nullptr || graph->RemoveNode(*node) != ge::GRAPH_SUCCESS) {
                OPS_LOG_E(kPassName, "Remove quant process node failed for Conv2DTranspose.");
                return ge::FAILED;
            }
        }
        if (!fusionNodes.empty()) {
            OPS_LOG_I(kPassName, "[OPS_NN_CONV2DT_QUANT_PROCESS] optimized Conv2DTranspose quant chain.");
        }
        changed = changed || !fusionNodes.empty();
    }
    return changed ? ge::SUCCESS : ge::GRAPH_NOT_CHANGED;
}

ge::Status Conv2DTQuantProcessFusionPass::GetQuantProcessMode(const ge::GNodePtr& quantNode,
                                                              const ge::GNodePtr& cubeNode,
                                                              QuantProcessMode& quantProcessMode)
{
    (void)quantNode;
    (void)cubeNode;
    quantProcessMode = QuantProcessMode::BIAS_OPTIMIZE;
    return ge::SUCCESS;
}

void Conv2DTQuantProcessFusionPass::SetCinCoutReverse(const ge::GNodePtr& node)
{
    bool reverse = true;
    if (node == nullptr || node->SetAttr(kCinCoutReverse, reverse) != ge::GRAPH_SUCCESS) {
        OPS_LOG_W(kPassName, "Set quant_cin_cout_reverse failed.");
    }
}

bool Conv2DTQuantProcessFusionPass::CheckSoc() const
{
    NpuArch npuArch = NpuArch::DAV_RESV;
    if (Ops::NN::Conv::ConvFusionUtils::ConvFusionUtilsPass::CheckSocList(kSupportSocList, npuArch)) {
        return true;
    }

    if (IsFuseSoc()) {
        return true;
    }
    OPS_LOG_D(kPassName, "Current SoC does not support this pass.");
    return false;
}

#if GE_COMPILER_VERSION_NUM >= 90300000U
REG_FUSION_PASS(Conv2DTQuantProcessFusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);
#endif
} // namespace ops
