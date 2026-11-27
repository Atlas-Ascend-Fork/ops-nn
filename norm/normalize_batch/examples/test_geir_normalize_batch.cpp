/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_normalize_batch.cpp
 * \brief GE graph construction sample for NormalizeBatch (arch35 ND format).
 */
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <string>
#include <vector>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "graph/operator_reg.h"
#include "../op_graph/normalize_batch_proto.h"

// 旧版 CANN 的 graph.h 与算子域存在 Data 定义重叠，测试内自带精简 Data 定义规避冲突
#ifndef OPS_PROTO_DEF_DATA
#define OPS_PROTO_DEF_DATA
namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
} // namespace ge
#endif // OPS_PROTO_DEF_DATA

#define FAILED -1
#define SUCCESS 0
using namespace ge;
using std::string;
using std::vector;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

string GetTime()
{
    const time_t current = time(nullptr);
    if (current == static_cast<time_t>(-1)) {
        return "unknown-time";
    }
    std::tm localTime = {};
    if (localtime_r(&current, &localTime) == nullptr) {
        return "unknown-time";
    }
    char buffer[32] = {};
    if (strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &localTime) == 0) {
        return "unknown-time";
    }
    return string(buffer);
}

// 每行填 0..d-1（per_feature 统计可由闭式导出，输出校验用同公式重算）
int32_t GenRowData(const vector<int64_t>& shapes, Tensor& tensor, TensorDesc& desc)
{
    desc.SetRealDimCnt(shapes.size());
    int64_t size = 1;
    for (auto d : shapes) {
        size *= d;
    }
    const int64_t rowLen = shapes.back();
    vector<float> v(static_cast<size_t>(size));
    for (int64_t i = 0; i < size; ++i) {
        v[static_cast<size_t>(i)] = static_cast<float>(i % rowLen);
    }
    desc.SetShape(ge::Shape(shapes));
    tensor.SetTensorDesc(desc);
    if (tensor.SetData(reinterpret_cast<const uint8_t*>(v.data()), size * sizeof(float)) != ge::GRAPH_SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

int32_t GenSeqData(int64_t n, int64_t len, Tensor& tensor, TensorDesc& desc)
{
    vector<int32_t> v(static_cast<size_t>(n), static_cast<int32_t>(len));
    desc.SetShape(ge::Shape({n}));
    tensor.SetTensorDesc(desc);
    if (tensor.SetData(reinterpret_cast<const uint8_t*>(v.data()), n * sizeof(int32_t)) != ge::GRAPH_SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}

int CreateGraph(std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
                Graph& graph)
{
    // per_feature 全 seq（L=d=8）：mean=3.5，var·N/(N-1)=6，y=(x-3.5)/(sqrt(6)+eps)
    const std::vector<int64_t> shape = {2, 3, 8};
    constexpr int64_t seqLen = 8;

    auto dataX = op::Data("ph_x").set_attr_index(0);
    TensorDesc xDesc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetOriginFormat(FORMAT_ND);
    xDesc.SetOriginShape(ge::Shape(shape));
    Tensor xTensor;
    if (GenRowData(shape, xTensor, xDesc) != SUCCESS) {
        return FAILED;
    }
    dataX.update_input_desc_x(xDesc);
    dataX.update_output_desc_y(xDesc);
    input.push_back(xTensor);
    graph.AddOp(dataX);
    inputs.push_back(dataX);

    auto dataSeq = op::Data("ph_seq").set_attr_index(1);
    TensorDesc seqDesc(ge::Shape({shape[0]}), FORMAT_ND, DT_INT32);
    seqDesc.SetPlacement(ge::kPlacementHost);
    seqDesc.SetOriginFormat(FORMAT_ND);
    seqDesc.SetOriginShape(ge::Shape({shape[0]}));
    Tensor seqTensor;
    if (GenSeqData(shape[0], seqLen, seqTensor, seqDesc) != SUCCESS) {
        return FAILED;
    }
    dataSeq.update_input_desc_x(seqDesc);
    dataSeq.update_output_desc_y(seqDesc);
    input.push_back(seqTensor);
    graph.AddOp(dataSeq);
    inputs.push_back(dataSeq);

    auto opNode = op::NormalizeBatch("normalize_batch_op");
    opNode.set_input_input_x(dataX);
    opNode.set_input_seq_len(dataSeq);
    opNode.set_attr_normalize_type("per_feature");
    opNode.set_attr_epsilon(1e-5f);
    TensorDesc yDesc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    yDesc.SetOriginFormat(FORMAT_ND);
    yDesc.SetOriginShape(ge::Shape(shape));
    opNode.update_output_desc_output_y(yDesc);
    outputs.push_back(opNode);
    return SUCCESS;
}

bool CheckOutput(const std::vector<ge::Tensor>& output)
{
    constexpr size_t elementCount = 2 * 3 * 8;
    if (output.size() != 1 || output[0].GetTensorDesc().GetShape().GetDims() != std::vector<int64_t>{2, 3, 8} ||
        output[0].GetTensorDesc().GetDataType() != DT_FLOAT || output[0].GetSize() != elementCount * sizeof(float) ||
        output[0].GetData() == nullptr) {
        LOG_PRINT("FAIL - unexpected output metadata or buffer\n");
        return false;
    }
    // 闭式期望：L=d=8 全行 [0..7] → mean=3.5，无偏 std=sqrt(6)，
    // y = (x - mean) / (std + eps)，eps 加在 sqrt 之后（对齐 canndev TIK 口径）
    const float mean = 3.5f;
    const float stdDev = std::sqrt(6.0f);
    for (size_t index = 0; index < elementCount; ++index) {
        float value = 0.0f;
        std::memcpy(&value, output[0].GetData() + index * sizeof(float), sizeof(float));
        const float x = static_cast<float>(index % 8U);
        const float want = (x - mean) / (stdDev + 1e-5f);
        if (!std::isfinite(value) || std::fabs(value - want) > 1e-4f) {
            LOG_PRINT("FAIL - output[%zu]=%.9g, expected %.9g\n", index, static_cast<double>(value),
                      static_cast<double>(want));
            return false;
        }
    }
    return true;
}

int main()
{
    Graph graph("normalize_batch_ge_irrun_test");
    std::vector<ge::Tensor> input;
    std::map<AscendString, AscendString> gopt = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(gopt) != SUCCESS) {
        LOG_PRINT("%s - FAIL - GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }
    std::vector<Operator> inputs{}, outputs{};
    if (CreateGraph(input, inputs, outputs, graph) != SUCCESS) {
        LOG_PRINT("%s - FAIL - CreateGraph failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);
    std::map<AscendString, AscendString> bopt = {};
    uint32_t gid = 0;
    std::map<AscendString, AscendString> gropt = {};
    std::vector<ge::Tensor> output;
    int32_t result = SUCCESS;
    {
        // session 在 GEFinalize 前析构
        Session session(bopt);
        if (session.AddGraph(gid, graph, gropt) != SUCCESS) {
            LOG_PRINT("%s - FAIL - [XIR]: AddGraph failed\n", GetTime().c_str());
            result = FAILED;
        } else if (session.RunGraph(gid, input, output) != SUCCESS) {
            LOG_PRINT("%s - FAIL - [XIR]: RunGraph failed\n", GetTime().c_str());
            result = FAILED;
        } else if (!CheckOutput(output)) {
            result = FAILED;
        } else {
            LOG_PRINT("%s - PASS - [XIR]: Session RunGraph succeeded, outputs=%zu\n", GetTime().c_str(), output.size());
            for (size_t i = 0; i < output.size(); i++) {
                LOG_PRINT("  output[%zu] dim=%zu\n", i, output[i].GetTensorDesc().GetShape().GetDimNum());
            }
        }
    }
    ge::GEFinalize();
    return result;
}
