/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_adagrad_to_inplace_fusion_pass.cpp
 * \brief ApplyAdagrad -> InplaceApplyAdagrad GEIR graph rewrite pass.
 *
 * ApplyAdagrad is a prototype-only shell op: it carries no op_host/op_kernel
 * implementation. On platforms where InplaceApplyAdagrad is implemented the
 * GEIR graph node is rewritten to InplaceApplyAdagrad, so the actual tiling
 * and AICore kernel of InplaceApplyAdagrad are reused.
 *
 * rewrite rule:
 *    var  accum  lr  grad                var  accum  lr  grad
 *      \    |    |   /                     \    |    |   /
 *       \   |    |  /                        \   |    |  /
 *        ApplyAdagrad            -->     InplaceApplyAdagrad
 *             |                                 |        \
 *             v                                 v         (inplace update of accum)
 *            var(out)                        var(out)
 */

#include "apply_adagrad_to_inplace_fusion_pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "common/inc/error_util.h"
#include "es_nn_ops.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"

using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;

namespace ops {
namespace {

constexpr char kPassName[] = "ApplyAdagradToInplaceFusionPass";
constexpr char kSourceOpType[] = "ApplyAdagrad";
constexpr int64_t kInputNum = 4;
constexpr int64_t kVarIdx = 0;
constexpr int64_t kAccumIdx = 1;
constexpr int64_t kLrIdx = 2;
constexpr int64_t kGradIdx = 3;
constexpr int64_t kOutputVarIdx = 0;
constexpr int64_t kInplaceOutputAccumIdx = 1;
const std::vector<DataType> kSupportedDtypes = {DT_BF16, DT_FLOAT16, DT_FLOAT};

bool IsTargetPlatform()
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    OP_LOGE_IF(PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
               false, kPassName, "Get platform info failed.");
    const std::string soc = platformInfo.str_info.short_soc_version;
    // 仅 A5（Ascend950 系列）生效，不适配 960；其余平台不做改写，保持原有行为。
    constexpr char kA5SocPrefix[] = "Ascend950";
    return soc.compare(0, sizeof(kA5SocPrefix) - 1, kA5SocPrefix) == 0;
}

bool IsSupportedDtype(DataType dtype)
{
    return std::find(kSupportedDtypes.begin(), kSupportedDtypes.end(), dtype) != kSupportedDtypes.end();
}

} // namespace

bool ApplyAdagradToInplaceFusionPass::MeetRequirements(const GNode& matchedNode)
{
    OPS_LOG_D(kPassName, "Enter MeetRequirements for %s", kPassName);
    if (!IsTargetPlatform()) {
        return false;
    }
    OP_LOGE_IF(matchedNode.GetInputsSize() != static_cast<size_t>(kInputNum), false, kPassName,
               "ApplyAdagrad input num is %zu, expected %ld.", matchedNode.GetInputsSize(), kInputNum);
    TensorDesc varDesc;
    OP_LOGE_IF(matchedNode.GetInputDesc(kVarIdx, varDesc) != SUCCESS, false, kPassName, "Get var desc failed.");
    OP_LOGE_IF(!IsSupportedDtype(varDesc.GetDataType()), false, kPassName,
               "Var dtype %d is not supported by InplaceApplyAdagrad.", varDesc.GetDataType());
    return true;
}

GraphUniqPtr ApplyAdagradToInplaceFusionPass::Replacement(const GNode& matchedNode)
{
    OPS_LOG_D(kPassName, "Enter Replacement for %s", kPassName);
    TensorDesc inputDescs[kInputNum];
    for (int64_t i = 0; i < kInputNum; ++i) {
        OP_LOGE_IF(matchedNode.GetInputDesc(i, inputDescs[i]) != SUCCESS, nullptr, kPassName,
                   "Get input %ld desc failed.", i);
    }
    TensorDesc varOutDesc;
    OP_LOGE_IF(matchedNode.GetOutputDesc(kOutputVarIdx, varOutDesc) != SUCCESS, nullptr, kPassName,
               "Get output var desc failed.");

    bool updateSlots = true;
    bool useLocking = false;
    (void)matchedNode.GetAttr("update_slots", updateSlots);
    (void)matchedNode.GetAttr("use_locking", useLocking);

    auto replaceGraphBuilder = es::EsGraphBuilder("InplaceApplyAdagrad");
    const char* inputNames[kInputNum] = {"var", "accum", "lr", "grad"};
    es::EsTensorHolder replaceInputs[kInputNum];
    for (int64_t i = 0; i < kInputNum; ++i) {
        replaceInputs[i] = replaceGraphBuilder.CreateInput(i, inputNames[i], inputDescs[i].GetDataType(),
                                                           inputDescs[i].GetFormat(),
                                                           inputDescs[i].GetShape().GetDims());
        OP_LOGE_IF(replaceInputs[i].GetProducer()->UpdateOutputDesc(0, inputDescs[i]) != GRAPH_SUCCESS, nullptr,
                   kPassName, "Update input %ld desc failed.", i);
    }

    auto inplaceOut = es::InplaceApplyAdagrad(replaceInputs[kVarIdx], replaceInputs[kAccumIdx], replaceInputs[kLrIdx],
                                              replaceInputs[kGradIdx], updateSlots, useLocking);
    GNode* inplaceNode = inplaceOut.ref_var.GetProducer();
    OP_LOGE_IF(inplaceNode == nullptr, nullptr, kPassName, "Get InplaceApplyAdagrad producer failed.");
    for (int64_t i = 0; i < kInputNum; ++i) {
        OP_LOGE_IF(inplaceNode->UpdateInputDesc(i, inputDescs[i]) != GRAPH_SUCCESS, nullptr, kPassName,
                   "Update InplaceApplyAdagrad input %ld desc failed.", i);
    }
    OP_LOGE_IF(inplaceNode->UpdateOutputDesc(kOutputVarIdx, varOutDesc) != GRAPH_SUCCESS, nullptr, kPassName,
               "Update InplaceApplyAdagrad output var desc failed.");
    OP_LOGE_IF(inplaceNode->UpdateOutputDesc(kInplaceOutputAccumIdx, inputDescs[kAccumIdx]) != GRAPH_SUCCESS, nullptr,
               kPassName, "Update InplaceApplyAdagrad output accum desc failed.");

    return replaceGraphBuilder.BuildAndReset({inplaceOut.ref_var});
}

#if GE_COMPILER_VERSION_NUM >= 90000000U
REG_DECOMPOSE_PASS(ApplyAdagradToInplaceFusionPass, {kSourceOpType}).Stage(CustomPassStage::kAfterInferShape);
#endif

} // namespace ops
