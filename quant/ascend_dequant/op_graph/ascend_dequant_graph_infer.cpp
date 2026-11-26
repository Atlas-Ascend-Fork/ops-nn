/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// quant/ascend_dequant/op_graph/ascend_dequant_graph_infer.cpp
// =============================================================================
//
// ROLE: Graph-level InferDataType / InferFormat for AscendDequant.
//   GE 在图构建/编译期（早于 tiling）需要节点的输出 dtype 与各张量 format，
//   这两个钩子必须注册在 op_graph 侧（随 op_graph/*.cpp 编进 op_proto 图插件库），
//   op_host 侧只保留 InferShape / InferShapeRange。
//
//   语义与老版 canndev/ops/built-in/op_proto/quantize_ops.cc 对齐：
//     InferDataType: y.dtype = attr `dtype`（0 -> float32，1 -> float16），
//                    与老版一致不做值域校验；属性缺失时沿用 x 的 dtype。
//     InferFormat  : y 的 origin/storage format 跟随 x；deq_scale 固定 NCHW。
//     FormatPass   : REGISTER_CUSTOM_PASS 在 kAfterInferShape 复刻老版 SetQuant3DFormat
//                    （需要图邻居/属性，InferFormat 上下文拿不到），并在 3D 场景把
//                    x/y/deq_scale 统一成 NCDHW/NDHWC。
//
// =============================================================================

#include <cstdint>
#include <vector>

#include "exe_graph/runtime/infer_datatype_context.h" // InferDataTypeContext
#include "graph/infer_format_context.h"               // InferFormatContext, StorageFormat
#include "log/log.h"                                  // OP_LOGD / OP_LOGE / OP_CHECK_NULL_WITH_CONTEXT
#include "register/op_impl_registry.h"                // IMPL_OP macro
#include "register/register_custom_pass.h"            // REGISTER_CUSTOM_PASS, CustomPassStage

namespace ops {
namespace {

constexpr size_t X_IDX = 0;
constexpr size_t SCALE_IDX = 1;
constexpr size_t Y_IDX = 0;
constexpr size_t ATTR_IDX_DTYPE = 2; // sqrt_mode, relu_flag, dtype

// 老版 canndev quantize_ops.cc 的 SetQuant3DFormat 常量
constexpr size_t QUANT_3D_ORI_DIM = 5;
constexpr char CONV3D_TYPE[] = "Conv3D";
constexpr char TF_TAG_ATTR[] = "tf_tag";

void CollectInDataNodes(const ge::GNode& node, std::vector<ge::GNodePtr>& inNodes)
{
    const size_t inputsSize = node.GetInputsSize();
    for (size_t i = 0; i < inputsSize; ++i) {
        const auto inNode = node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        if (inNode.first != nullptr) {
            inNodes.push_back(inNode.first);
        }
    }
}

// 老版 GetConv3DFlag：任一输入节点是 Conv3D 即命中，取其输出 format 的 primary format
bool GetConv3DFlag(const std::vector<ge::GNodePtr>& nodes, ge::Format& convFormat)
{
    for (const auto& nodePtr : nodes) {
        if (nodePtr == nullptr) {
            continue;
        }
        ge::AscendString type;
        if (nodePtr->GetType(type) != ge::GRAPH_SUCCESS) {
            continue;
        }
        if (type == ge::AscendString(CONV3D_TYPE)) {
            ge::TensorDesc convOutDesc;
            if (nodePtr->GetOutputDesc(0, convOutDesc) != ge::GRAPH_SUCCESS) {
                continue;
            }
            convFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(convOutDesc.GetFormat()));
            return true;
        }
    }
    return false;
}

// 老版 SetQuant3DFormat(op, DUAL_INPUT)：
//   rank(x)==5 且输入侧（或 tf_tag 非空时再往上一层）存在 Conv3D 时，
//   把 x / y / deq_scale 统一设为 Conv3D 的 NCDHW/NDHWC；无法判定时按 tf_tag 取默认值。
ge::Status ApplyQuant3DFormat(ge::GNode& node, ge::TensorDesc& xDesc, ge::TensorDesc& yDesc, ge::TensorDesc& scaleDesc)
{
    if (xDesc.GetShape().GetDimNum() != QUANT_3D_ORI_DIM) {
        return ge::GRAPH_SUCCESS;
    }
    ge::AscendString tfTag;
    const bool hasTfTag = (node.GetAttr(ge::AscendString(TF_TAG_ATTR), tfTag) == ge::GRAPH_SUCCESS) &&
                          (tfTag.GetLength() != 0U);

    std::vector<ge::GNodePtr> inNodes;
    CollectInDataNodes(node, inNodes);
    if (inNodes.empty()) {
        return ge::GRAPH_SUCCESS;
    }
    ge::Format convFormat = ge::FORMAT_ND;
    bool conv3dFlag = GetConv3DFlag(inNodes, convFormat);
    if (!conv3dFlag && hasTfTag) {
        // conv3d + biasadd + dequant/requant：再往前一层找 Conv3D
        std::vector<ge::GNodePtr> preNodes;
        CollectInDataNodes(*inNodes[0], preNodes);
        conv3dFlag = !preNodes.empty() && GetConv3DFlag(preNodes, convFormat);
    }
    if (!conv3dFlag) {
        return ge::GRAPH_SUCCESS;
    }
    const ge::Format format3d = ((convFormat == ge::FORMAT_NCDHW) || (convFormat == ge::FORMAT_NDHWC)) ?
                                    convFormat :
                                    (hasTfTag ? ge::FORMAT_NDHWC : ge::FORMAT_NCDHW);
    xDesc.SetOriginFormat(format3d);
    xDesc.SetFormat(format3d);
    yDesc.SetOriginFormat(format3d);
    yDesc.SetFormat(format3d);
    scaleDesc.SetOriginFormat(format3d);
    scaleDesc.SetFormat(format3d);
    ge::AscendString nodeName;
    (void)node.GetName(nodeName);
    OP_LOGD(nodeName.GetString(), "[AscendDequant][FormatPass] 3D quant: convFormat=%d -> format3d=%d",
            static_cast<int>(convFormat), static_cast<int>(format3d));
    return ge::GRAPH_SUCCESS;
}

} // namespace

// ---------------------------------------------------------------------------
// InferDataTypeForAscendDequant(context) — y dtype follows attr `dtype`
// ---------------------------------------------------------------------------
ge::graphStatus InferDataTypeForAscendDequant(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("AscendDequant", "InferDataType context is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const int64_t* dtypePtr = (attrs == nullptr) ? nullptr : attrs->GetInt(ATTR_IDX_DTYPE);
    if (dtypePtr != nullptr) {
        // 老版：op.GetAttr("dtype", type) 成功即按属性设置，无值域校验
        OP_LOGD(context->GetNodeName(), "[AscendDequant][InferDataType] attr dtype=%ld -> y dtype=%d", *dtypePtr,
                static_cast<int>(*dtypePtr));
        context->SetOutputDataType(Y_IDX, static_cast<ge::DataType>(*dtypePtr));
    } else {
        // 老版：属性缺失时保留 OneInOneOutDynamicInfer 写入的输入 dtype
        const ge::DataType xDtype = context->GetInputDataType(X_IDX);
        OP_LOGD(context->GetNodeName(), "[AscendDequant][InferDataType] attr dtype absent -> y dtype = x dtype %d",
                static_cast<int>(xDtype));
        context->SetOutputDataType(Y_IDX, xDtype);
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// InferFormatForAscendDequant(context) — 复刻老版 canndev quantize_ops.cc 的格式推导
//   ① AscendDequantInferFormat: y 的 origin/storage format 跟随 x
//   ② AscendDequantInferShape: deq_scale 的 origin/storage format 固定为 NCHW
// ---------------------------------------------------------------------------
ge::graphStatus InferFormatForAscendDequant(gert::InferFormatContext* context)
{
    if (context == nullptr) {
        OP_LOGE("AscendDequant", "InferFormat context is nullptr");
        return ge::GRAPH_FAILED;
    }

    // ① y 跟随 x 的格式
    gert::StorageFormat* xFormat = context->GetInputFormat(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xFormat);
    gert::StorageFormat* yFormat = context->GetOutputFormat(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yFormat);
    yFormat->SetOriginFormat(xFormat->GetOriginFormat());
    yFormat->SetStorageFormat(xFormat->GetStorageFormat());

    // ② deq_scale 固定 NCHW
    gert::StorageFormat* scaleFormat = context->GetInputFormat(SCALE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleFormat);
    scaleFormat->SetOriginFormat(ge::FORMAT_NCHW);
    scaleFormat->SetStorageFormat(ge::FORMAT_NCHW);

    OP_LOGD(context->GetNodeName(),
            "[AscendDequant][InferFormat] x(origin=%d,storage=%d) -> y(origin=%d,storage=%d), deq_scale -> NCHW(%d)",
            static_cast<int>(xFormat->GetOriginFormat()), static_cast<int>(xFormat->GetStorageFormat()),
            static_cast<int>(yFormat->GetOriginFormat()), static_cast<int>(yFormat->GetStorageFormat()),
            static_cast<int>(scaleFormat->GetOriginFormat()));

    // 3D 量化场景（老版 SetQuant3DFormat）需要图邻居，由 AscendDequantFormatPass 处理。
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// AscendDequantFormatPass — 复刻老版 InferShape 里的 deq_scale=NCHW / SetQuant3DFormat
//   以及 InferFormat 的 y 跟随 x（需要图邻居，故用 custom pass 实现）
// ---------------------------------------------------------------------------
ge::Status AscendDequantFormatPass(ge::GraphPtr& graph, ge::CustomPassContext&)
{
    if (graph == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (auto& node : graph->GetAllNodes()) {
        ge::AscendString type;
        if (node.GetType(type) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (type != ge::AscendString("AscendDequant")) {
            continue;
        }
        ge::TensorDesc xDesc;
        ge::TensorDesc scaleDesc;
        ge::TensorDesc yDesc;
        if ((node.GetInputDesc(static_cast<int32_t>(X_IDX), xDesc) != ge::GRAPH_SUCCESS) ||
            (node.GetInputDesc(static_cast<int32_t>(SCALE_IDX), scaleDesc) != ge::GRAPH_SUCCESS) ||
            (node.GetOutputDesc(static_cast<int32_t>(Y_IDX), yDesc) != ge::GRAPH_SUCCESS)) {
            ge::AscendString nodeName;
            (void)node.GetName(nodeName);
            OP_LOGE(nodeName.GetString(), "[AscendDequant][FormatPass] get tensor desc failed.");
            return ge::GRAPH_FAILED;
        }
        // ① deq_scale 固定 NCHW（老版 AscendDequantInferShape）
        scaleDesc.SetOriginFormat(ge::FORMAT_NCHW);
        scaleDesc.SetFormat(ge::FORMAT_NCHW);
        // ② 3D 量化场景（老版 SetQuant3DFormat）
        if (ApplyQuant3DFormat(node, xDesc, yDesc, scaleDesc) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        // ③ y 跟随 x（老版 AscendDequantInferFormat）
        yDesc.SetOriginFormat(xDesc.GetOriginFormat());
        yDesc.SetFormat(xDesc.GetFormat());

        if ((node.UpdateInputDesc(static_cast<int32_t>(X_IDX), xDesc) != ge::GRAPH_SUCCESS) ||
            (node.UpdateInputDesc(static_cast<int32_t>(SCALE_IDX), scaleDesc) != ge::GRAPH_SUCCESS) ||
            (node.UpdateOutputDesc(static_cast<int32_t>(Y_IDX), yDesc) != ge::GRAPH_SUCCESS)) {
            ge::AscendString nodeName;
            (void)node.GetName(nodeName);
            OP_LOGE(nodeName.GetString(), "[AscendDequant][FormatPass] update tensor desc failed.");
            return ge::GRAPH_FAILED;
        }
        ge::AscendString nodeName;
        (void)node.GetName(nodeName);
        OP_LOGD(nodeName.GetString(),
                "[AscendDequant][FormatPass] x(origin=%d,storage=%d) y(origin=%d,storage=%d) deq_scale(origin=%d,"
                "storage=%d)",
                static_cast<int>(xDesc.GetOriginFormat()), static_cast<int>(xDesc.GetFormat()),
                static_cast<int>(yDesc.GetOriginFormat()), static_cast<int>(yDesc.GetFormat()),
                static_cast<int>(scaleDesc.GetOriginFormat()), static_cast<int>(scaleDesc.GetFormat()));
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(AscendDequant).InferDataType(InferDataTypeForAscendDequant).InferFormat(InferFormatForAscendDequant);

// 图编译期 format 修补（含需要图邻居的 3D 场景），在 shape 推导后、图融合前执行。
REGISTER_CUSTOM_PASS("AscendDequantFormatPass")
    .CustomPassFn(AscendDequantFormatPass)
    .Stage(ge::CustomPassStage::kAfterInferShape);
} // namespace ops
