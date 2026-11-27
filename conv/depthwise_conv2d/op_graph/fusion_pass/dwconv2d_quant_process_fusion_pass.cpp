/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dwconv2d_quant_process_fusion_pass.h"

#include <algorithm>
#include <limits>
#include <map>
#include <vector>

#include "acl/acl_platform.h"

#include "conv/common/op_graph/fusion_pass/tensor_desc_origin_dtype_api.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace ge;

namespace {
constexpr size_t LAST_AXIS_INDEX = 3;

const std::string FUSION_NAME = "DWConv2DQuantProcessFusionPass";
const ge::AscendString PAD = "Pad";
const std::string DN2NZ_INTRINSIC = "Intrinsic_data_move_out2l1_dn2nz";

// 各 NpuArch 算子信息库的 (x/filter)/bias/offset_w 支持 dtype 白名单与 x/filter 支持 format 白名单
struct DepthwiseQuantWhitelist {
    std::vector<ge::DataType> supportedDtypes;
    std::vector<ge::DataType> biasDtypes;
    std::vector<ge::DataType> offsetWDtypes;
    std::vector<ge::Format> fmapFormats;
    std::vector<ge::Format> filterFormats;
};

// dtype、format白名单源自：静态算子信息库（opdef.cpp、ops-info.ini) + 动态查询 op_select_format
const DepthwiseQuantWhitelist* GetDepthwiseWhitelist(const NpuArch arch)
{
    static const std::map<NpuArch, DepthwiseQuantWhitelist> whitelistTable = {
        {NpuArch::DAV_3510,
         {{ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_HIFLOAT8, ge::DT_INT8},
          {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32},
          {ge::DT_INT8},
          {},
          {}}},
        {NpuArch::DAV_9201,
         {{ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_HIFLOAT8, ge::DT_INT8},
          {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32},
          {ge::DT_INT8},
          {},
          {}}},
        {NpuArch::DAV_9202,
         {{ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_HIFLOAT8, ge::DT_INT8},
          {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32},
          {ge::DT_INT8},
          {},
          {}}},
        {NpuArch::DAV_2002,
         {{ge::DT_FLOAT16, ge::DT_INT8},
          {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32},
          {ge::DT_FLOAT16, ge::DT_INT8},
          {ge::FORMAT_NC1HWC0},
          {ge::FORMAT_FRACTAL_Z}}},
        {NpuArch::DAV_1001,
         {{ge::DT_FLOAT16, ge::DT_INT8},
          {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32},
          {ge::DT_FLOAT16, ge::DT_INT8},
          {ge::FORMAT_NC1HWC0},
          {ge::FORMAT_FRACTAL_Z}}},
        {NpuArch::DAV_3004,
         {{ge::DT_FLOAT16, ge::DT_INT8},
          {ge::DT_FLOAT16, ge::DT_INT32},
          {ge::DT_FLOAT16, ge::DT_INT8},
          {ge::FORMAT_NC1HWC0},
          {ge::FORMAT_FRACTAL_Z}}},
        {NpuArch::DAV_3102,
         {{ge::DT_FLOAT16, ge::DT_INT8},
          {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32},
          {ge::DT_FLOAT16, ge::DT_INT8},
          {ge::FORMAT_NC1HWC0},
          {ge::FORMAT_FRACTAL_Z}}},
        {NpuArch::DAV_2201,
         {{ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT8},
          {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32},
          {ge::DT_INT8},
          {ge::FORMAT_NC1HWC0},
          {ge::FORMAT_FRACTAL_Z}}},
        {NpuArch::DAV_3002,
         {{ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT8},
          {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32},
          {ge::DT_INT8},
          {ge::FORMAT_NC1HWC0},
          {ge::FORMAT_FRACTAL_Z}}},
        // DAV_5102信息库无 DepthwiseConv2D section：空表项使 dtype 校验恒拒绝以镜像该行为
        {NpuArch::DAV_5102, {}},
    };
    const auto iter = whitelistTable.find(arch);
    return iter == whitelistTable.end() ? nullptr : &iter->second;
}

NpuArch GetCurNpuArch()
{
    char npuArchStr[kNpuArchStrLen] = {};
    if (aclplatformGetDeviceInfo(ACL_PLATFORM_NPU_ARCH, npuArchStr, sizeof(npuArchStr)) != ACL_SUCCESS) {
        OP_LOGW(FUSION_NAME, "Get npu arch failed.");
        return NpuArch::DAV_RESV;
    }
    return static_cast<NpuArch>(std::atoi(npuArchStr));
}

// 量化链决策相关输入（x、filter）的 dtype 白名单校验，不在支持列表（如 950 家族不含 DT_INT4）时须走
// 量化回退。format 维度不设白名单：各 SoC 信息库中 DepthwiseConv2D 的 format 为动态查询（950 家族
// legacy json）或 NC1HWC0/FRACTAL_Z（静态 SoC，非 ND 布局老校验直接放行，且 FRACTAL_Z 可转 ND），
// 老 CheckFormatSupported 对该算子 format 维度实际全放行，保持一致。
// 表外 SoC（动态格式查询/无映射遗留 SoC）不校验，沿用历史行为（待 GE 能力查询接口长期解决）
// 老 FE CheckFormatSupported 等价（sub_ops_store.cc DTYPE_FORMAT_MODE）：信息库 format 列表为空
// 直接放行（950 家族 legacy json 无静态列表）；origin format 非 ND 直接放行（量化图 x/filter origin
// 为 NC1HWC0/FRACTAL_Z/NCHW 等布局，不做列表校验）；origin format 为 ND 时，算子信息库 format 列表
// 须含 ND 可转格式（ND/FRACTAL_NZ/ND_RNN_BIAS/FRACTAL_ZN_RNN/FRACTAL_Z）
bool IsOriginFormatSupported(const ge::TensorDesc& inputDesc, const std::vector<ge::Format>& supportedFormats)
{
    if (supportedFormats.empty()) {
        return true;
    }
    const ge::Format originFormat = inputDesc.GetOriginFormat();
    if (originFormat != ge::FORMAT_ND) {
        return true;
    }
    static const std::vector<ge::Format> ndTransableFormats = {
        ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND_RNN_BIAS, ge::FORMAT_FRACTAL_ZN_RNN, ge::FORMAT_FRACTAL_Z};
    const auto transIter = std::find_first_of(supportedFormats.begin(), supportedFormats.end(),
                                              ndTransableFormats.begin(), ndTransableFormats.end());
    return transIter != supportedFormats.end();
}

bool CheckQuantInputSupported(const ge::GNode& node, const uint32_t weightInputIndex)
{
    const NpuArch arch = GetCurNpuArch();
    const DepthwiseQuantWhitelist* whitelist = GetDepthwiseWhitelist(arch);
    if (whitelist == nullptr) {
        OP_LOGW(FUSION_NAME, "No DepthwiseConv2D quant whitelist for NpuArch[%u], skip dtype check.",
                static_cast<uint32_t>(arch));
        return true;
    }
    const int32_t biasIndex = static_cast<int32_t>(weightInputIndex) + 1;
    const int32_t offsetWIndex = biasIndex + 1;
    const int32_t inputIndexes[] = {0, static_cast<int32_t>(weightInputIndex), biasIndex, offsetWIndex};
    const std::vector<ge::DataType>* inputDtypes[] = {&whitelist->supportedDtypes, &whitelist->supportedDtypes,
                                                      &whitelist->biasDtypes, &whitelist->offsetWDtypes};
    const std::vector<ge::Format>* inputFormats[] = {&whitelist->fmapFormats, &whitelist->filterFormats, nullptr,
                                                     nullptr};
    for (size_t i = 0; i < sizeof(inputIndexes) / sizeof(inputIndexes[0]); ++i) {
        ge::TensorDesc inputDesc;
        if (node.GetInputDesc(inputIndexes[i], inputDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        const ge::DataType dtype = inputDesc.GetDataType();
        if (std::find(inputDtypes[i]->begin(), inputDtypes[i]->end(), dtype) == inputDtypes[i]->end()) {
            OP_LOGD(FUSION_NAME, "Node input[%d] dtype[%s] is not in DepthwiseConv2D supported dtypes.",
                    inputIndexes[i], ge::TypeUtils::DataTypeToSerialString(dtype).c_str());
            return false;
        }
        if (inputFormats[i] != nullptr && !IsOriginFormatSupported(inputDesc, *inputFormats[i])) {
            OP_LOGD(FUSION_NAME, "Node input[%d] origin format is ND without a transferable format.", inputIndexes[i]);
            return false;
        }
    }
    return true;
}

bool IsNodeType(const ge::GNodePtr& node, const ge::AscendString& expectType)
{
    if (node == nullptr) {
        return false;
    }
    ge::AscendString type;
    FUSION_PASS_CHECK_NOLOG(node->GetType(type) != GRAPH_SUCCESS, return false);
    return type == expectType;
}

bool PadShapeTo4Dim(ge::Format filterFormat, const std::vector<int64_t>& filterDims, std::vector<int64_t>& filterDims4D)
{
    if (filterDims.empty()) {
        OP_LOGE(FUSION_NAME, "Filter dims is empty, can not pad shape.");
        return false;
    }
    size_t sizeOfFilter = filterDims.size();
    for (size_t i = 0; i <= LAST_AXIS_INDEX; ++i) {
        if (i < sizeOfFilter) {
            filterDims4D.emplace_back(filterDims.at(i));
        } else if (filterFormat == ge::FORMAT_NCHW) {
            filterDims4D.emplace_back(1);
        } else if (filterFormat == ge::FORMAT_HWCN) {
            filterDims4D.insert(filterDims4D.cbegin(), 1);
        } else if (filterFormat == ge::FORMAT_NHWC) {
            filterDims4D.insert(filterDims4D.cbegin() + 1, 1);
        } else if (filterFormat == ge::FORMAT_ND) {
            filterDims4D.emplace_back(0);
        } else {
            OP_LOGE(FUSION_NAME, "Format [%s] can not pad shape.", GeFormatToString(filterFormat).c_str());
            return false;
        }
    }
    return true;
}

bool CheckInt64MulOverflow(int64_t left, int64_t right)
{
    if (left == 0 || right == 0) {
        return false;
    }
    if (left > 0) {
        return right > 0 ? left > std::numeric_limits<int64_t>::max() / right :
                           right < std::numeric_limits<int64_t>::min() / left;
    }
    return right > 0 ? left < std::numeric_limits<int64_t>::min() / right :
                       right < std::numeric_limits<int64_t>::max() / left;
}
} // namespace

ge::Status DWConv2DQuantProcessFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    if (graph == nullptr || !graph->IsValid()) {
        return GRAPH_NOT_CHANGED;
    }
    passContext.SetPassName(FUSION_NAME.c_str());

    std::vector<GNode> cubeNodes;
    if (!ConvFusionUtilsPass::GetMatchedNodes(graph, cubeNodes, {DEPTHWISE_CONV2D})) {
        return FAILED;
    }
    if (cubeNodes.empty()) {
        return GRAPH_NOT_CHANGED;
    }

    int32_t effectTimes = 0;
    for (auto& cubeNode : cubeNodes) {
        InitMember();

        if (!CheckQuantChain(cubeNode)) {
            continue;
        }
        FUSION_PASS_CHECK_NOLOG(!ConvFusionUtilsPass::GetConvDescInfo(cubeNode, convDescInfo), return FAILED);

        std::vector<GNode> nodesBeforeFuse;
        if (quantNode_ != nullptr) {
            nodesBeforeFuse.emplace_back(*quantNode_);
        }
        nodesBeforeFuse.emplace_back(cubeNode);
        if (dequantNode_ != nullptr) {
            nodesBeforeFuse.emplace_back(*dequantNode_);
        }
        if (padNode_ != nullptr) {
            nodesBeforeFuse.emplace_back(*padNode_);
        }
        FUSION_PASS_CHECK_NOLOG(!ReportMatchNodes(nodesBeforeFuse, passContext), return FAILED);
        if (!CanFuseNodes(nodesBeforeFuse)) {
            continue;
        }

        ge::GNodePtr cubeNodePtr = std::make_shared<ge::GNode>(cubeNode);
        std::vector<ge::GNodePtr> fusionNodes;
        std::vector<ge::GNodePtr> nodesToRemove;
        auto status = OptimizeMainProc(graph, quantNode_, dequantNode_, cubeNodePtr, fusionNodes, nodesToRemove);
        if (status == FAILED) {
            return FAILED;
        }
        if (fusionNodes.empty()) {
            continue;
        }

        for (auto& nodeToRemove : nodesToRemove) {
            FUSION_PASS_CHECK(graph->RemoveNode(*nodeToRemove) != GRAPH_SUCCESS,
                              OP_LOGE(FUSION_NAME, "Remove node failed."), return FAILED);
        }
        std::vector<GNode> nodesForReport;
        nodesForReport.emplace_back(cubeNode);
        if (padNode_ != nullptr) {
            nodesForReport.emplace_back(*padNode_);
        }
        std::vector<GNode> nodesAfterFuse;
        for (auto& fusedNode : fusionNodes) {
            nodesAfterFuse.emplace_back(*fusedNode);
        }
        FUSION_PASS_CHECK_NOLOG(!ReportFuseNodes(nodesForReport, nodesAfterFuse, passContext), return FAILED);
        PrintGraphStructure();
        effectTimes++;
    }
    return effectTimes != 0 ? SUCCESS : GRAPH_NOT_CHANGED;
}

void DWConv2DQuantProcessFusionPass::InitMember()
{
    npuArch = NpuArch::DAV_RESV;
    convDescInfo = ConvDescInfo();
    quantNode_ = nullptr;
    dequantNode_ = nullptr;
    padNode_ = nullptr;
}

bool DWConv2DQuantProcessFusionPass::CheckQuantChain(const ge::GNode& cubeNode)
{
    padNode_ = nullptr;
    auto in0Pair = cubeNode.GetInDataNodesAndPortIndexs(INPUT_FMAP_INDEX);
    ge::GNodePtr in0Node = in0Pair.first;
    if (IsNodeType(in0Node, ASCEND_QUANT)) {
        quantNode_ = in0Node;
    } else if (IsNodeType(in0Node, PAD)) {
        auto quantPair = in0Node->GetInDataNodesAndPortIndexs(INPUT_FMAP_INDEX);
        FUSION_PASS_CHECK(!IsNodeType(quantPair.first, ASCEND_QUANT),
                          OP_LOGD(FUSION_NAME, "Pad input producer is not AscendQuant, no fusion."), return false);
        quantNode_ = quantPair.first;
        padNode_ = in0Node;
    } else {
        OP_LOGD(FUSION_NAME, "DepthwiseConv2D input producer is not AscendQuant or Pad, no fusion.");
        return false;
    }

    dequantNode_ = nullptr;
    for (auto& consumerPair : cubeNode.GetOutDataNodesAndPortIndexs(OUTPUT_INDEX)) {
        if (consumerPair.first == nullptr || consumerPair.second != INPUT_FMAP_INDEX) {
            continue;
        }
        if (IsNodeType(consumerPair.first, ASCEND_DEQUANT)) {
            dequantNode_ = consumerPair.first;
            break;
        }
    }
    FUSION_PASS_CHECK(dequantNode_ == nullptr,
                      OP_LOGD(FUSION_NAME, "DepthwiseConv2D output consumer is not AscendDequant, no fusion."),
                      return false);
    return true;
}

ge::Status DWConv2DQuantProcessFusionPass::GetCoValue(const ge::GNodePtr& cubeNode, int64_t& co)
{
    ge::TensorDesc filterDesc;
    FUSION_PASS_CHECK(cubeNode->GetInputDesc(static_cast<int32_t>(GetWeightInputIndex()), filterDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Get filter desc failed."), return FAILED);

    ge::Format filterFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(static_cast<int32_t>(filterDesc.GetFormat())));
    if (filterFormat == FORMAT_ND) {
        return SUCCESS;
    }
    int64_t groups = 1;
    (void)cubeNode->GetAttr(GROUPS, groups);
    std::vector<int64_t> filterDims4D;
    FUSION_PASS_CHECK_NOLOG(!PadShapeTo4Dim(filterFormat, filterDesc.GetShape().GetDims(), filterDims4D),
                            return FAILED);
    FUSION_PASS_CHECK(filterDims4D.empty(),
                      OP_LOGE(FUSION_NAME, "Node[%s] filter dims4d is empty.", convDescInfo.nodeNameStr.c_str()),
                      return FAILED);
    int32_t indexCo = ops::GetAxisIndexByFormat(filterFormat, "N");
    if (indexCo < 0) {
        OP_LOGE(FUSION_NAME, "Node[%s] index_co is negative, check filter format.", convDescInfo.nodeNameStr.c_str());
        return FAILED;
    }
    if (indexCo >= static_cast<int32_t>(filterDims4D.size())) {
        OP_LOGE(FUSION_NAME, "Node[%s] index_co is larger than the size of filter dims.",
                convDescInfo.nodeNameStr.c_str());
        return FAILED;
    }
    FUSION_PASS_CHECK(CheckInt64MulOverflow(filterDims4D[indexCo], groups),
                      OP_LOGE(FUSION_NAME, "Node[%s] int64 mul overflow.", convDescInfo.nodeNameStr.c_str()),
                      return FAILED);
    co = filterDims4D[indexCo] * groups;
    return SUCCESS;
}

ge::Status DWConv2DQuantProcessFusionPass::SetDataTypeOfNodes(const ge::GNodePtr& cubeNode)
{
    if (SetCubeNodeDataType(cubeNode, ge::DT_FLOAT, ge::DT_FLOAT16) != SUCCESS) {
        OP_LOGE(FUSION_NAME, "Set cube node[%s] data type failed.", convDescInfo.nodeNameStr.c_str());
        return FAILED;
    }

    if (padNode_ != nullptr && SetPadNodeDataType(*padNode_) != SUCCESS) {
        OP_LOGE(FUSION_NAME, "Set pad node data type failed.");
        return FAILED;
    }
    return SUCCESS;
}

ge::GNodePtr DWConv2DQuantProcessFusionPass::GetCubeNodeInputNode(const ge::GNodePtr& cubeNode) const
{
    return padNode_ != nullptr ? padNode_ : cubeNode;
}

ge::Status DWConv2DQuantProcessFusionPass::SetPadNodeDataType(ge::GNode& padNode) const
{
    ge::TensorDesc padInDesc;
    FUSION_PASS_CHECK(padNode.GetInputDesc(INPUT_FMAP_INDEX, padInDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Get pad input desc failed."), return FAILED);
    padInDesc.SetDataType(ge::DT_FLOAT16);
    SetOriginDataType(padInDesc, ge::DT_FLOAT16);
    FUSION_PASS_CHECK(padNode.UpdateInputDesc(INPUT_FMAP_INDEX, padInDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Fail to update input desc of pad node."), return FAILED);

    ge::TensorDesc padOutDesc;
    FUSION_PASS_CHECK(padNode.GetOutputDesc(OUTPUT_INDEX, padOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Get pad output desc failed."), return FAILED);
    padOutDesc.SetDataType(ge::DT_FLOAT16);
    SetOriginDataType(padOutDesc, ge::DT_FLOAT16);
    FUSION_PASS_CHECK(padNode.UpdateOutputDesc(OUTPUT_INDEX, padOutDesc) != GRAPH_SUCCESS,
                      OP_LOGE(FUSION_NAME, "Fail to update output desc of pad node."), return FAILED);

    OP_LOGD(FUSION_NAME, "Data type of Pad has been updated to float16 during quant rollback.");
    return SUCCESS;
}

bool DWConv2DQuantProcessFusionPass::MeetRequirements(const ge::GNode& convNode)
{
    (void)convNode;
    return true;
}

bool DWConv2DQuantProcessFusionPass::CheckOpSupported(const ge::GNode& node) const
{
    // dtype 白名单校验（按 NpuArch 取信息库支持列表，如 950 家族不含 DT_INT4），须走量化回退
    if (!CheckQuantInputSupported(node, GetWeightInputIndex())) {
        return false;
    }
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != 0U) {
        OP_LOGD(FUSION_NAME, "Get platform info failed, keep cube node supported.");
        return true;
    }
    if (platformInfo.ai_core_intrinsic_dtype_map.find(DN2NZ_INTRINSIC) !=
        platformInfo.ai_core_intrinsic_dtype_map.end()) {
        OP_LOGD(FUSION_NAME, "Soc has dn2nz intrinsic without legacy depthwise impl, quant rollback.");
        return false;
    }
    return true;
}

std::set<ge::AscendString> DWConv2DQuantProcessFusionPass::GetNodeTypes() const { return {DEPTHWISE_CONV2D}; }

void DWConv2DQuantProcessFusionPass::PrintGraphStructure() const
{
    OP_LOGI(FUSION_NAME, "%s %s done.", convDescInfo.nodeNameStr.c_str(), FUSION_NAME.c_str());
}

#if GE_COMPILER_VERSION_NUM >= 90300000U
REG_FUSION_PASS(DWConv2DQuantProcessFusionPass).Stage(CustomPassStage::kCompatibleInherited);
#endif
} // namespace Ops
