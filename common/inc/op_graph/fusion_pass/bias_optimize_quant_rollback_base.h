/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef COMMON_OP_GRAPH_FUSION_PASS_BIAS_OPTIMIZE_QUANT_ROLLBACK_BASE_H_
#define COMMON_OP_GRAPH_FUSION_PASS_BIAS_OPTIMIZE_QUANT_ROLLBACK_BASE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ge/ge_api_error_codes.h"
#include "graph/graph.h"
#include "graph/gnode.h"
#include "graph/tensor.h"

namespace ops {
static constexpr const char* const PATTERN_QUANT = "ascendquant";
static constexpr const char* const PATTERN_CUBE = "cube";
static constexpr const char* const PATTERN_DEQUANT = "ascenddequant";
static constexpr const char* const PATTERN_PAD = "pad";
static constexpr const char* const ASCEND_QUANT = "AscendQuant";
static constexpr const char* const ASCEND_DEQUANT = "AscendDequant";

static constexpr uint32_t BIAS_INPUT_INDEX = 2U;
static constexpr uint32_t WEIGHT_INPUT_INDEX = 1U;
static constexpr uint32_t INPUT_SIZE_CONTAINS_BIAS = 3U;
static constexpr uint32_t DEQUANT_SCALE_INPUT_INDEX = 1U;
static constexpr uint32_t NCHW_DIMENSION_NUM = 4U;

enum class HostOpMode {
    WEIGHT_ROLL_BACK_MODE = 0,
    BIAS_ROLL_BACK_MODE = 1,
    BIAS_OPTIMIZATION_MODE = 2,
    BIAS_OPTIMIZATION_MODE_CONVDT = 3,
    WEIGHT_ROLL_BACK_MODE_CONVDT = 4,
    BIAS_ROLL_BACK_MODE_CONVDT = 5,
    MODE_BOTTOM = 6
};

enum class QuantProcessMode { BIAS_OPTIMIZE = 0, QUANT_ROLLBACK, QUANT_UNDEFINED };

struct PatternNodes {
    ge::GNodePtr cubeNode;
    ge::GNodePtr dequantNode;
    ge::GNodePtr quantNode;
    ge::GNodePtr weightConstNode;
    ge::GNodePtr ascendWeightQuantNode;
};

int32_t GetAxisIndexByFormat(const ge::Format& format, const std::string& axis);

class BiasOptimizeQuantRollbackBase {
public:
    BiasOptimizeQuantRollbackBase() = default;
    virtual ~BiasOptimizeQuantRollbackBase() = default;
    BiasOptimizeQuantRollbackBase(const BiasOptimizeQuantRollbackBase&) = delete;
    BiasOptimizeQuantRollbackBase& operator=(const BiasOptimizeQuantRollbackBase&) = delete;
    ge::Status OptimizeMainProc(ge::GraphPtr& graph, const ge::GNodePtr& quantNode, const ge::GNodePtr& dequantNode,
                                const ge::GNodePtr& cubeNode, std::vector<ge::GNodePtr>& fusionNodes,
                                std::vector<ge::GNodePtr>& nodesToRemove);

protected:
    virtual uint32_t GetBiasInputIndex() const { return BIAS_INPUT_INDEX; }
    virtual uint32_t GetInputSizeContainsBias() const { return INPUT_SIZE_CONTAINS_BIAS; }
    virtual uint32_t GetWeightInputIndex() const { return WEIGHT_INPUT_INDEX; }
    virtual uint32_t GetBiasOptimizationModeIndex() const
    {
        return static_cast<uint32_t>(HostOpMode::BIAS_OPTIMIZATION_MODE);
    }
    virtual uint32_t GetWeightRollBackModeIndex() const
    {
        return static_cast<uint32_t>(HostOpMode::WEIGHT_ROLL_BACK_MODE);
    }
    virtual uint32_t GetBiasRollBackModeIndex() const { return static_cast<uint32_t>(HostOpMode::BIAS_ROLL_BACK_MODE); }

    virtual bool CheckOpSupported(const ge::GNode& node) const;

    virtual ge::Status GetCoValue(const ge::GNodePtr& cubeNode, int64_t& co);

    virtual void SetCinCoutReverse(const ge::GNodePtr& node);

    ge::Status SetBiasName(const std::string& biasName);

    bool IsCubeNodeSupported(const ge::GNodePtr& cubeNode) const;

    bool IsCubeSupportDtype(const ge::GNodePtr& cubeNode, const ge::DataType& dtype) const;

    static bool IsWeightTranspose(const ge::GNodePtr& cubeNode);

    static void SetNodeOutputDtype(const ge::GNodePtr& node, const ge::DataType& checkDtype,
                                   const ge::DataType& targetDtype);

    virtual ge::Status GetQuantProcessMode(const ge::GNodePtr& quantNode, const ge::GNodePtr& cubeNode,
                                           QuantProcessMode& quantProcessMode);

    ge::Status JudgeDeqscaleShape(const ge::GNodePtr& dequantNode) const;

    virtual ge::Status RemoveEnter(ge::Graph& graph, const ge::GNodePtr& node) const;

    ge::Status SetQuantParameters(const ge::GNodePtr& cubeNode, const ge::GNodePtr& quantNode) const;

    virtual ge::Status ProcessBiasInput(ge::Graph& graph, const ge::GNodePtr& cubeNode);

    ge::Status CreateBiasInput(ge::Graph& graph, const ge::GNodePtr& cubeNode, const int64_t& co);

    ge::Status CreateBiasInput(ge::Graph& graph, const ge::GNodePtr& cubeNode, const ge::Shape& biasShape);

    bool IsCubeHasBiasInput(const ge::GNodePtr& cubeNode) const;

    virtual bool IsCubeNeedBiasInput(const ge::GNodePtr& cubeNode);

    virtual ge::Status BiasOptimize(ge::Graph& graph, const ge::GNodePtr& cubeNode, const ge::GNodePtr& dequantNode,
                                    const ge::GNodePtr& quantNode, std::vector<ge::GNodePtr>& fusionNodes);

    ge::Status SetCubeNodeDataType(const ge::GNodePtr& cubeNode, const ge::DataType& dataType,
                                   const ge::DataType& targetDataType) const;

    virtual ge::Status SetDataTypeOfNodes(const ge::GNodePtr& cubeNode);

    using PeerOutAnchors = std::vector<std::pair<ge::GNodePtr, int32_t>>;
    ge::Status RemoveInputEdgeAndSingleConstInput(ge::Graph& graph, const ge::GNodePtr& node,
                                                  PeerOutAnchors& peerOutAnchorsOfNode) const;

    ge::Status LinkOutputEdgeWithoutControl(ge::Graph& graph, const ge::GNodePtr& oldNode, const ge::GNodePtr& newNode,
                                            const std::string& cubeName,
                                            const PeerOutAnchors& peerOutAnchorsOfOldNode) const;

    ge::Status CreateNewHostCpuOp(const std::string& opType, const PatternNodes& patternNode, ge::Graph& graph,
                                  const uint32_t mode, std::vector<ge::GNodePtr>& fusNodes);

    ge::Status LinkHostOpEdge(ge::Graph& graph, const PatternNodes& patternNode, const ge::GNodePtr& host_node,
                              const uint32_t& anchorIndex, const size_t& index) const;

    ge::Status DoFusion(ge::Graph& graph, const ge::GNodePtr& cubeNode, const ge::GNodePtr& quantNode,
                        const ge::GNodePtr& dequantNode, std::vector<ge::GNodePtr>& fusNodes);

    ge::Status SetGroupInfoForAscendWeightQuantNodes(const ge::GNodePtr& cubeNode) const;

    virtual ge::GNodePtr GetCubeNodeInputNode(const ge::GNodePtr& cubeNode) const;

    ge::Status GetWeightNodesOfCubeNode(const ge::GNodePtr& cubeNode, ge::GNodePtr& weightConstNode,
                                        ge::GNodePtr& ascendWeightQuantNode) const;

    ge::Status WeightRollbackDeleteIsolateNodes(ge::Graph& graph, const ge::GNodePtr& ascendWeightQuantNode) const;

    ge::Status SetInt4OrInt8Flag(const ge::GNodePtr& ascendWeightQuantNode,
                                 const ge::GNodePtr& quantHostCpuOpNode) const;

    ge::Status ChangeQuantNodeEdge(ge::Graph& graph, const ge::GNodePtr& cubeNode, const ge::GNodePtr& quantNode) const;

    virtual ge::Status ChangeDequantNodeEdge(ge::Graph& graph, const ge::GNodePtr& cubeNode,
                                             const ge::GNodePtr& dequantNode);

    virtual ge::Status QuantRollback(ge::Graph& graph, const ge::GNodePtr& cubeNode, const ge::GNodePtr& dequantNode,
                                     const ge::GNodePtr& quantNode, std::vector<ge::GNodePtr>& fusionNodes);

private:
    ge::Status SetHostOpNodeDescs(const ge::GNodePtr& host_node, const PatternNodes& patternNode,
                                  const std::vector<uint32_t>& anchorIndexlist,
                                  const std::vector<std::string>& tensorDescNameList, ge::TensorDesc& outputDesc) const;
    ge::Status LinkHostOpEdges(ge::Graph& graph, const PatternNodes& patternNode, const ge::GNodePtr& host_node,
                               const std::vector<uint32_t>& anchorIndexlist) const;
    ge::Status SetHostOpNodeAttrs(const std::string& opType, const PatternNodes& patternNode,
                                  const ge::GNodePtr& host_node);
    ge::Tensor CreateBiasTensor(const ge::Shape& biasShape) const;
    ge::GNodePtr CreateBiasNode(ge::Graph& graph, const ge::Tensor& biasTensor, const ge::GNodePtr& cubeNode) const;
    ge::Status UpdateBiasOutputDesc(const ge::GNodePtr& cubeNode, const ge::Shape& shape, const ge::Format& format,
                                    const uint32_t index) const;
    ge::Status UpdateCubeInputDesc(const ge::GNodePtr& cubeNode, const ge::Shape& shape, const ge::Format& format,
                                   const uint32_t index) const;
    std::string bias_name_ = "bias";
    mutable std::vector<ge::GNodePtr> nodesToRemove_;
};
} // namespace ops
#endif // COMMON_OP_GRAPH_FUSION_PASS_BIAS_OPTIMIZE_QUANT_ROLLBACK_BASE_H_
