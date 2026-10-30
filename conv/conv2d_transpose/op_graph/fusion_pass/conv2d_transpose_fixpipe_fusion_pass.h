/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv2d_transpose_fixpipe_fusion_pass.h
 * \brief conv2dtranspose to extendconvtranspose fusion pass
 * (conv2dtranspose -> postcube ===> extendConvTranspose, 2D input expand to 3D, 3D output squeeze to 2D,
 * A16W8 weight rearranged by the RearrangeA16W8WeightFzOp host node before ExtendConvTranspose)
 */
#ifndef CONV2D_TRANSPOSE_FIXPIPE_FUSION_PASS_H
#define CONV2D_TRANSPOSE_FIXPIPE_FUSION_PASS_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../../conv/common/op_graph/fusion_pass/conv_backprop_fusion_base_pass.h"
#include "cube_utils/cube_utils.h"
#include "ge/fusion/subgraph_boundary.h"

namespace ops {
namespace Conv2DTransposeFixpipeFusion {

const std::string FUSION_NAME = "Conv2DTransposeFixpipeFusionPass";
const ge::AscendString CONV2D_TRANSPOSE = "Conv2DTranspose";
const ge::AscendString POST_CUBE_OP = "FixPipe";
const ge::AscendString FUSION_OP_LIST = "fusion_op_list";
const ge::AscendString kAscendDequantOp = "AscendDequant";
const ge::AscendString kAscendRequantOp = "AscendRequant";
const ge::AscendString kAscendQuantOp = "AscendQuant";
const ge::AscendString kReluOp = "Relu";
const ge::AscendString SET_M1_DEQUANT_OP = "SetM1Dequant";
// output transpose fusion (migrated from the original fixpipe pass): the single Transpose
// successor of the output path with a 4D NHWC output is removed, the ExtendConvTranspose output
// becomes NDHWC and the Squeeze output takes the removed Transpose 4D desc
const ge::AscendString TRANSPOSE_OP = "Transpose";
constexpr int64_t TRANSPOSE_OUTPUT_DIM = 4;

// A16W8 weight rearrange host op: rearranges the int8 weight into the layout required by the AIC
// of ExtendConvTranspose, it has no registered op_proto and is rebuilt inside the replacement graph
const ge::AscendString REARRANGE_A16W8_WEIGHT_FZ_OP = "RearrangeA16W8WeightFzOp";
// RearrangeA16W8WeightFzOp attr names and values: conv2dtranspose consumes the weight in reverse
// order, so the reverse mode is required (keep consistent with the original fixpipe pass)
constexpr char const* kAttrRearrangeGroups = "groups";
constexpr char const* kAttrReverseMode = "attrReverseMode";
constexpr int64_t REARRANGE_REVERSE_MODE = 1;
// A16W8 (fp16/fp32 x + int8 filter, both group and non-group scenarios) always rearranges the
// weight for the AIC (migrated from the original fixpipe pass)
// fixed shift value derived from the original filter dtype when the attr is absent/zero
constexpr int64_t FIXED_SHIFT_VALUE_FP16 = 42;
constexpr int64_t FIXED_SHIFT_VALUE_INT8 = 13;
// RearrangeA16W8WeightFzOp input "x" is at index 0
constexpr int32_t REARRANGE_INPUT_X_INDEX = 0;

// postcube fusion node white list, head node type included
const std::vector<ge::AscendString> SUPPORTED_NODE_TYPES = {CONV2D_TRANSPOSE, kAscendDequantOp, kAscendRequantOp,
                                                            kAscendQuantOp, kReluOp};

// ExtendConvTranspose supports int8 x + int8 filter (quant scenario) and float16 x + int8 filter
// (A16W8 scenario, the weight is rearranged before entering the op), y supports float16/int8;
// the other dtype combos are converted to Conv3DTranspose by Conv2DTransposeToV2FusionPass
constexpr ge::DataType SUPPORT_FILTER_DTYPE = ge::DT_INT8;
// PostCube output descriptors may still carry the pre-conversion dtype here.
// Run() converts FLOAT/INT32 to the FLOAT16/INT8 outputs of ExtendConvTranspose.
const std::vector<ge::DataType> SUPPORT_OUTPUT_DTYPES = {ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT, ge::DT_INT32};

// only NCHW input expands to NCDHW consistently, NHWC and the other formats are not supported
constexpr ge::Format SUPPORT_FORMAT_2D = ge::FORMAT_NCHW;

// Conv2DTranspose input index: input_size(0), x(1), filter(2), bias(3), offset_w(4)
constexpr int32_t INPUT_SIZE_INDEX = 0;
constexpr int32_t X_INDEX = 1;
constexpr int32_t FILTER_INDEX = 2;
constexpr int32_t BIAS_INDEX = 3;
// ExtendConvTranspose input index: input_size(0), x(1), filter(2), bias(3), scale(4);
// ExtendConvTransposeV2 additionally contains scale1(5) (migrated from the updated fixpipe pass)
constexpr int32_t SCALE_INDEX = 4;
constexpr int32_t SCALE1_INDEX = 5;
constexpr size_t CONV2D_TRANSPOSE_INPUT_NUM = 3;
// PostCube(FixPipe) input index: quant_scale_0 is at index 2
constexpr int32_t POST_CUBE_INPUT_QUANT_SCALE_0_INDEX = 2;
constexpr int32_t POST_CUBE_INPUT_QUANT_SCALE_1_INDEX = 5;
// AscendDequant/AscendRequant scale input (deq_scale/req_scale) is at index 1
constexpr int32_t QUANT_OP_SCALE_INPUT_INDEX = 1;
// SetM1Dequant input "x1" is at index 0
constexpr int32_t SET_M1_INPUT_INDEX = 0;
constexpr int32_t DUAL_OUTPUT_NUM = 2;
// 3D attr length after expanding the 2D attrs: strides/dilations/output_padding are 5, pads are 6
constexpr size_t PADS_3D_SIZE = 6;
constexpr int64_t DEFAULT_FUSION_MODE = 0;
constexpr int64_t RELU_FUSION_MODE = 1;
constexpr int64_t RELU1_FUSION_MODE = 2;
constexpr int64_t DEFAULT_Y_QUANT_MODE = 0;

// postcube fusion result: kSkipNode means the graph has already been changed by PostCubeUtils,
// the conv2dtranspose to extendconvtranspose conversion must be skipped to keep the graph consistent
enum class PostCubeFusionResult : std::uint8_t { kNoPostCube, kFused, kSkipNode };

// output case of the postcube fusion (migrated from the updated fixpipe pass): the dual output
// cases create ExtendConvTransposeV2 and are enabled only on the cube vector fuse soc with
// exactly two outputs of the conv2dtranspose
enum class OutputCase : std::uint8_t {
    SINGLE,          // single FixPipe branch, no passthrough consumer
    DUAL_FIXPIPE,    // both outputs are FixPipe branches
    FIXPIPE_CONV2DT, // FixPipe branch first + passthrough consumer second
    CONV2DT_FIXPIPE  // passthrough consumer first + FixPipe branch second
};
} // namespace Conv2DTransposeFixpipeFusion

class __attribute__((visibility("default"))) Conv2DTransposeFixpipeFusionPass : public ConvBackpropFusionBasePass {
public:
    explicit Conv2DTransposeFixpipeFusionPass(const std::vector<ge::AscendString>& opTypes)
        : ConvBackpropFusionBasePass(opTypes)
    {}

    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext) override;

protected:
    void InitMember() override;
    bool MeetRequirements(const ge::GNode& convTransposeNode) override;
    bool GetNodeDesc(const ge::GNode& node) override;
    bool GetNodeAttrs(const ge::GNode& node) override;
    ge::AscendString GetNodeType() const override;
    ge::fusion::GraphUniqPtr Replacement(const ge::GNode& convTransposeNode) override;

private:
    // postcube fusion: create PostCube node for the post nodes of conv2dtranspose
    Conv2DTransposeFixpipeFusion::PostCubeFusionResult PostCubeFusionImpl(ge::GraphPtr& graph,
                                                                          const ge::GNode& convTransposeNode,
                                                                          ge::CustomPassContext& passContext);
    void SelectPostCubePassByWhiteList(std::vector<ops::PostCubePassInfo>& matchLists) const;

    // all skip decisions of the postcube fusion are made before CreatePostCubeNode mutates the
    // graph: the call deletes the folded post op chain and the graph cannot be restored, a skip
    // after it would leave the engine-less FixPipe node in the graph and the graph compile fails
    // with "cannot find engine"; every post-creation check (GetPostCubeNodes, ReadAndCheck-
    // PostCubeFusionOps) is derived from the selected passes and the original graph here
    // instead, an empty match list adopts a pre-existing FixPipe and creates nothing, so it
    // always passes
    bool CheckPostCubePassesPrepared(const ge::GNode& convTransposeNode,
                                     const std::vector<ops::PostCubePassInfo>& matchLists) const;

    Conv2DTransposeFixpipeFusion::PostCubeFusionResult GetPostCubeNodes(const ge::GNode& convTransposeNode);
    bool ReadAndCheckPostCubeFusionOps(const ge::GNode& postCubeNode, std::vector<ge::AscendString>& fusionOps) const;
    bool HasScaleNode() const;
    bool HasReluNode() const;

    // dual output support (migrated from the updated fixpipe pass): collect the FixPipe branches
    // and the passthrough consumers of the conv2dtranspose output, decide the output case (only
    // the cube vector fuse soc with exactly two outputs enables the dual output)
    void CollectOutputNodes(const ge::GNode& convTransposeNode);
    void SetOutputCase();
    bool IsCubeVectorFusePlatform() const;
    bool IsScaleEnable(const std::vector<ge::AscendString>& fusionOps) const;
    bool IsReluEnable(const std::vector<ge::AscendString>& fusionOps) const;

    // trace SetM1Dequant node which produces postcube quant_scale_0 (dequant param -> M1 scale),
    // nullptr if quant_scale_0 is not produced by SetM1Dequant
    ge::GNodePtr GetScaleSetM1Node(const ge::GNode& postCubeNode) const;

    // detect the removable Transpose successor of the output path (single successor + Transpose
    // type + 4D NHWC output, migrated from the original fixpipe pass), the result is stored in
    // hasFusedTranspose/transposeOutputDesc
    void CheckOutputTranspose(const ge::GNode& outputNode);

    // A16W8 weight trans scenario: fp16 x + int8 filter + groups == 1, the int8 weight must be
    // rearranged by the RearrangeA16W8WeightFzOp host node before entering ExtendConvTranspose
    // (the weight trans functionality migrated from the original fixpipe pass)
    bool NeedRearrangeWeightNode() const;

    // build the RearrangeA16W8WeightFzOp host node between the filter input (the Unsqueeze node
    // or the directly connected fractal filter) and ExtendConvTranspose inside the replacement
    // graph, the rearranged weight is returned as the filter input of ExtendConvTranspose
    bool CreateRearrangeWeightNode(ge::es::EsGraphBuilder& builder, ge::es::EsTensorHolder& filterInput,
                                   const ge::TensorDesc& filterInputDesc, const std::string& nodeNamePrefix,
                                   ge::es::EsTensorHolder& rearrangeOutput);

    // Capture the INT8 branch's zero point before PostCube removes its original nodes.
    // The generated FixPipe name identifies the branch even if dual outputs are reordered.
    void CaptureRequantOffsets(const std::string& nodeName, const std::vector<ops::PostCubePassInfo>& matchLists);

    // resolve the requant zero point for the rebuilt SetM1Dequant: prefer the original
    // quantized output offset, then the FixPipe input and host op attrs, finally the existing
    // symmetric int8 fallback; encode the result as a 9-bit M1 field
    void ResolveRequantOffset();
    float ResolveRequantZeroPoint(const ge::GNode& setM1Node, const ge::GNode& postCubeNode) const;

    // replacement graph input index of scale, must be consistent with the boundary index and
    // keep continuous from 0 when the optional bias input is absent
    int64_t GetScaleGraphInputIndex() const;

    // subgraph boundary: conv2dtranspose(+postcube) inputs/outputs to replacement inputs/outputs
    std::unique_ptr<ge::fusion::SubgraphBoundary> ConstructBoundary(const ge::GNode& convTransposeNode);

    // update input/output tensor desc of ExtendConvTranspose node
    bool UpdateExtendConvTransposeDesc(ge::GNode& extendConvTransposeNode, const ge::TensorDesc& unsqueezeXOutDesc,
                                       const ge::TensorDesc& unsqueezeFilterOutDesc, const ge::TensorDesc& output3DDesc,
                                       const ge::TensorDesc& output3DDesc1);

    // rebuild the SetM1Dequant host node inside the replacement graph (the original one has no
    // registered op_proto and is pruned during subgraph replace): the attrs are copied from the
    // original node, the int8 path forces the dequantouttype and sets the 9-bit encoded requant
    // zero point as the float offset attr
    bool BuildSetM1DequantNode(ge::es::EsGraphBuilder& builder, ge::es::EsTensorHolder& iScale,
                               const ge::GNode& setM1Node, const std::string& setM1Name, float requantOffsetValue,
                               const ge::TensorDesc& pathOutputDesc, ge::es::EsTensorHolder& scaleOutput);

    // replacement graph input index of scale1 (the dual output second scale), must be consistent
    // with the boundary index and keep continuous from 0 when the optional bias input is absent
    int64_t GetScale1GraphInputIndex() const;

    ge::GNodePtr postCubeNode = nullptr;
    std::vector<ge::AscendString> postCubeFusionOps = {};

    // dual output support (migrated from the updated fixpipe pass): all FixPipe branches, their
    // fusion op lists, the passthrough consumers (node + input index) and the output case;
    // postCubeNode/postCubeFusionOps above always alias the first FixPipe branch
    std::vector<ge::GNodePtr> allPostCubeNodes = {};
    std::vector<std::vector<ge::AscendString>> allPostCubeFusionOps = {};
    std::vector<std::pair<ge::GNodePtr, int32_t>> allOtherNodes = {};
    Conv2DTransposeFixpipeFusion::OutputCase outputCase = Conv2DTransposeFixpipeFusion::OutputCase::SINGLE;
    // the first consumer of the conv2dtranspose output (decides the output order of the mixed
    // dual output cases) and the conv2dtranspose original output desc (the passthrough path desc)
    ge::GNodePtr firstOutputNode = nullptr;
    ge::TensorDesc convTransposeOutputDesc;
    // the second FixPipe branch carries a quant scale op
    bool hasScale1 = false;

    // output transpose fusion (migrated from the original fixpipe pass): the removable Transpose
    // node of the output path, its 4D NHWC output desc and the detection result
    ge::GNodePtr transposeNode = nullptr;
    ge::TensorDesc transposeOutputDesc;
    bool hasFusedTranspose = false;

    bool hasPostCube = false;
    bool hasScale = false;
    bool hasBias = false;

    // A16W8 weight trans scenario flag (resolved by NeedRearrangeWeightNode after the node attrs
    // are read), the RearrangeA16W8WeightFzOp host node is rebuilt in the replacement graph
    bool hasRearrangeWeight = false;

    // 9-bit encoded requant zero point ((zp & 0x1FF)) for the rebuilt SetM1Dequant offset
    // attr (resolved by ResolveRequantOffset before the replacement changes the graph), the
    // second value is for the scale1 path of the dual output
    float requantOffset = 0.0f;
    float requantOffset1 = 0.0f;
    std::map<std::string, int32_t> quantOffsetsByPostCubeName;

    ge::TensorDesc biasDesc;
    // the y1 output desc of the dual output cases (from the back FixPipe branch or the
    // conv2dtranspose original output of the mixed case, migrated from the updated fixpipe pass)
    ge::TensorDesc outputDesc1;
    std::vector<int64_t> outputPadding = {};
    int64_t offsetX = 0;
    int64_t fusionMode = Conv2DTransposeFixpipeFusion::DEFAULT_FUSION_MODE;
};

} // namespace ops

#endif // CONV2D_TRANSPOSE_FIXPIPE_FUSION_PASS_H
