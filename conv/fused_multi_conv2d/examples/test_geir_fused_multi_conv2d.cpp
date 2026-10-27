/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include "array_ops.h"
#include "ge_api.h"
#include "../../conv2d/op_graph/conv2d_proto.h"
#include "../../../quant/ascend_requant/op_graph/ascend_requant_proto.h"

namespace {
// 创建常量节点；name 为节点名，description 为张量描述，values 为常量数据。
template <typename T>
ge::op::Const MakeConst(const char* name, const ge::TensorDesc& description, const std::vector<T>& values)
{
    ge::Tensor tensor(description, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T));
    auto constant = ge::op::Const(name).set_attr_value(tensor);
    constant.update_output_desc_y(description);
    return constant;
}

// 添加一层卷积及量化；inputOperator 为上游节点，layerIndex 用于区分两层节点名称。
ge::Operator AppendConvRequantLayer(ge::Operator& inputOperator, int layerIndex)
{
    const ge::TensorDesc fmapDesc(ge::Shape({1, 32, 4, 4}), ge::FORMAT_NCHW, ge::DT_INT8);
    const ge::TensorDesc accumDesc(ge::Shape({1, 32, 4, 4}), ge::FORMAT_NCHW, ge::DT_INT32);
    const ge::TensorDesc filterDesc(ge::Shape({32, 32, 1, 1}), ge::FORMAT_NCHW, ge::DT_INT8);
    const ge::TensorDesc scaleDesc(ge::Shape({32}), ge::FORMAT_ND, ge::DT_UINT64);
    const auto suffix = std::to_string(layerIndex);
    auto filter = MakeConst(("filter" + suffix).c_str(), filterDesc, std::vector<int8_t>(32 * 32, 1));
    auto scale = MakeConst(("scale" + suffix).c_str(), scaleDesc, std::vector<uint64_t>(32, 0x3e000000ULL));
    auto conv = ge::op::Conv2D(("conv" + suffix).c_str())
                    .set_input_x(inputOperator)
                    .set_input_filter(filter)
                    .set_attr_strides({1, 1, 1, 1})
                    .set_attr_pads({0, 0, 0, 0})
                    .set_attr_dilations({1, 1, 1, 1})
                    .set_attr_groups(1)
                    .set_attr_data_format("NCHW");
    conv.update_input_desc_x(fmapDesc);
    conv.update_input_desc_filter(filterDesc);
    conv.update_output_desc_y(accumDesc);
    auto requant = ge::op::AscendRequant(("requant" + suffix).c_str())
                       .set_input_x(conv)
                       .set_input_req_scale(scale)
                       .set_attr_relu_flag(true);
    requant.update_input_desc_x(accumDesc);
    requant.update_input_desc_req_scale(scaleDesc);
    requant.update_output_desc_y(fmapDesc);
    return requant;
}

// 构造两层连续卷积图；inputDesc 为图输入的形状、布局和类型。
ge::Graph BuildExampleGraph(const ge::TensorDesc& inputDesc)
{
    auto inputNode = ge::op::Data("x").set_attr_index(0);
    inputNode.update_input_desc_x(inputDesc);
    inputNode.update_output_desc_y(inputDesc);
    ge::Operator current = inputNode;
    for (int layerIndex = 1; layerIndex <= 2; ++layerIndex) {
        current = AppendConvRequantLayer(current, layerIndex);
    }
    ge::Graph graph("fused_multi_conv2d_example");
    graph.SetInputs({inputNode}).SetOutputs({current});
    return graph;
}

// 逐元素核对量化结果；outputs 为运行输出，expectedBytes 为单个输出的预期字节数。
ge::Status CheckExampleOutput(const std::vector<ge::Tensor>& outputs, size_t expectedBytes)
{
    if (outputs.size() != 1 || outputs[0].GetTensorDesc().GetDataType() != ge::DT_INT8 ||
        outputs[0].GetSize() != expectedBytes) {
        return ge::FAILED;
    }
    const auto* actual = reinterpret_cast<const int8_t*>(outputs[0].GetData());
    for (size_t position = 0; position < expectedBytes; ++position) {
        const int8_t expected = position % 16 < 8 ? 0 : 16;
        if (actual[position] != expected) {
            std::cerr << "output[" << position << "]=" << static_cast<int>(actual[position]) << ", expected "
                      << static_cast<int>(expected) << std::endl;
            return ge::FAILED;
        }
    }
    std::cout << "FusedMultiConv2d example passed" << std::endl;
    return ge::SUCCESS;
}

// 开启两阶段融合，执行示例图并核对输出。
ge::Status RunExample()
{
    const char* fusionFile = "fused_multi_conv2d_fusion.json";
    std::ofstream fusionSwitch(fusionFile);
    fusionSwitch << R"({"Switch":{"GraphFusion":{
        "Conv2DFixPipeToExtendConv2DFusionPass":"on",
        "ExtendConv2DPairToFusedMultiConv2DFusionPass":"on"}}})";
    fusionSwitch.close();
    if (!fusionSwitch) {
        return ge::FAILED;
    }
    const ge::TensorDesc inputDesc(ge::Shape({1, 32, 4, 4}), ge::FORMAT_NCHW, ge::DT_INT8);
    auto graph = BuildExampleGraph(inputDesc);
    std::vector<int8_t> inputValues(32 * 4 * 4);
    for (size_t position = 0; position < inputValues.size(); ++position) {
        inputValues[position] = position % 16 < 8 ? -1 : 1;
    }
    ge::Tensor input(inputDesc, reinterpret_cast<const uint8_t*>(inputValues.data()), inputValues.size());
    ge::Session session(std::map<ge::AscendString, ge::AscendString>{{"ge.fusionSwitchFile", fusionFile}});
    auto status = session.AddGraph(0, graph);
    if (status != ge::SUCCESS) {
        return status;
    }
    std::vector<ge::Tensor> outputs;
    status = session.RunGraph(0, {input}, outputs);
    return status == ge::SUCCESS ? CheckExampleOutput(outputs, inputValues.size()) : status;
}
} // namespace

// 初始化图引擎、运行示例并释放资源，以退出码报告结果。
int main()
{
    const auto status = ge::GEInitialize(
        std::map<ge::AscendString, ge::AscendString>{{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}});
    if (status != ge::SUCCESS) {
        return 1;
    }
    const auto result = RunExample();
    const auto finalized = ge::GEFinalize();
    return result == ge::SUCCESS && finalized == ge::SUCCESS ? 0 : 1;
}
