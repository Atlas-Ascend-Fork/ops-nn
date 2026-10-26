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
 * \file test_geir_ascend_requant.cpp
 * \brief GE IR graph-mode invocation example for AscendRequant.
 *
 * AscendRequant has no aclnn API; this sample builds the op through the GE IR
 * (op_graph/ascend_requant_proto.h), runs the graph and verifies the result:
 *   y = SaturateInt8(RoundToNearestEven(x * scale)), with relu_flag applied on
 *   top when enable. `scale` is decoded from the UINT64 DeqScale tensor
 *   req_scale, which is broadcast to x by the kernel: here x is [4, 8] and
 *   req_scale is [8] (per-channel scale over the last dim).
 * With x = -100 and req_scale = 2, every element is expected to saturate to
 * INT8 min: y = SaturateInt8(-200) = -128.
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"
#include "../op_graph/ascend_requant_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

int32_t GenConstInt32Data(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, int32_t value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    uint32_t data_len = size * sizeof(int32_t);
    int32_t* pData = new (std::nothrow) int32_t[size];
    if (pData == nullptr) {
        printf("%s - ERROR - [XIR]: Allocate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        pData[i] = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t GenConstUint64Data(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, uint64_t value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    uint32_t data_len = size * sizeof(uint64_t);
    uint64_t* pData = new (std::nothrow) uint64_t[size];
    if (pData == nullptr) {
        printf("%s - ERROR - [XIR]: Allocate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        pData[i] = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        printf("%s - ERROR - [XIR]: Open output file %s failed\n", GetTime().c_str(), bin_file.c_str());
        return FAILED;
    }
    size_t written = fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    if (written != data_size) {
        printf("%s - ERROR - [XIR]: Write output file %s failed\n", GetTime().c_str(), bin_file.c_str());
        return FAILED;
    }
    return SUCCESS;
}

int CreateOppInGraph(std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
                     Graph& graph)
{
    Status ret = SUCCESS;
    auto requantOp = op::AscendRequant("ascendRequant1");

    // --- Input x: INT32 accumulator, shape [4, 8] ---
    std::vector<int64_t> xShape = {4, 8};
    auto xPlaceholder = op::Data("placeholder_x").set_attr_index(0);
    TensorDesc xDesc = TensorDesc(ge::Shape(xShape), FORMAT_ND, DT_INT32);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetFormat(FORMAT_ND);
    Tensor xTensor;
    ret = GenConstInt32Data(xShape, xTensor, xDesc, -100);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    xPlaceholder.update_input_desc_x(xDesc);
    xPlaceholder.update_output_desc_y(xDesc);
    input.push_back(xTensor);
    graph.AddOp(xPlaceholder);
    inputs.push_back(xPlaceholder);
    requantOp.set_input_x(xPlaceholder);

    // --- Input req_scale: UINT64 DeqScale, shape [8] broadcasts over the last dim of x ---
    std::vector<int64_t> reqScaleShape = {8};
    auto scalePlaceholder = op::Data("placeholder_req_scale").set_attr_index(1);
    TensorDesc scaleDesc = TensorDesc(ge::Shape(reqScaleShape), FORMAT_ND, DT_UINT64);
    scaleDesc.SetPlacement(ge::kPlacementHost);
    scaleDesc.SetFormat(FORMAT_ND);
    Tensor scaleTensor;
    ret = GenConstUint64Data(reqScaleShape, scaleTensor, scaleDesc, 2);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    scalePlaceholder.update_input_desc_x(scaleDesc);
    scalePlaceholder.update_output_desc_y(scaleDesc);
    input.push_back(scaleTensor);
    graph.AddOp(scalePlaceholder);
    inputs.push_back(scalePlaceholder);
    requantOp.set_input_req_scale(scalePlaceholder);

    // --- Optional attr relu_flag: false keeps negative values ---
    requantOp.set_attr_relu_flag(false);

    // --- Output y: INT8, same shape as x ---
    TensorDesc yDesc = TensorDesc(ge::Shape(xShape), FORMAT_ND, DT_INT8);
    requantOp.update_output_desc_y(yDesc);

    outputs.push_back(requantOp);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_ascend_requant";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    if (output.empty()) {
        printf("%s - ERROR - [XIR]: Graph output is empty\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    int32_t verifyResult = SUCCESS;
    for (size_t i = 0; i < output.size(); i++) {
        const int8_t* yData = reinterpret_cast<const int8_t*>(output[i].GetData());
        int64_t yShapeSize = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = static_cast<uint32_t>(yShapeSize) * sizeof(int8_t);
        for (int64_t j = 0; j < yShapeSize; j++) {
            // SaturateInt8(RoundToNearestEven(-100 * 2)) = -128
            if (yData[j] != static_cast<int8_t>(-128)) {
                printf("%s - ERROR - [XIR]: output[%zu][%lld] = %d, expect -128\n", GetTime().c_str(), i,
                       static_cast<long long>(j), static_cast<int32_t>(yData[j]));
                verifyResult = FAILED;
                break;
            }
        }
        string output_file = "./tc_ge_irrun_ascend_requant_npu_output_" + std::to_string(i) + ".bin";
        if (WriteDataToFile(output_file, data_size, output[i].GetData()) != SUCCESS) {
            verifyResult = FAILED;
        }
    }
    if (verifyResult != SUCCESS) {
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GE IR pathway verification PASSED\n", GetTime().c_str());

    delete session;

    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
