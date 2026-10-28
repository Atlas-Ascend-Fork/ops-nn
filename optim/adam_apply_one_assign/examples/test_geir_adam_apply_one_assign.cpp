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
 * \file test_geir_adam_apply_one_assign.cpp
 * \brief
 */
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <cstring>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/adam_apply_one_assign_proto.h"

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

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16)
        return 2;
    if (dt == ge::DT_INT32 || dt == ge::DT_UINT32)
        return 4;
    if (dt == ge::DT_INT64 || dt == ge::DT_UINT64)
        return 8;
    if (dt == ge::DT_INT8 || dt == ge::DT_UINT8)
        return 1;
    return 4;
}

static int32_t GenOnesDataFp32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++)
        size *= shapes[i];
    uint32_t data_len = size * 4;
    float* pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i)
        *(pData + i) = value;
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto op1 = op::AdamApplyOneAssign("adam1");
    std::vector<int64_t> shape = {4};

    // Define 10 inputs as Data (placeholder) nodes
    auto mkInput = [&](int idx, const char* name, float val) {
        auto data = op::Data("input" + std::to_string(idx));
        TensorDesc desc(ge::Shape(shape), FORMAT_ND, inDtype);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor;
        GenOnesDataFp32(shape, tensor, desc, val);
        data.update_input_desc_x(desc);
        input.push_back(tensor);
        graph.AddOp(data);

        // set_input_input0 ~ set_input_add2_y
        if (idx == 0)
            op1.set_input_input0(data);
        else if (idx == 1)
            op1.set_input_input1(data);
        else if (idx == 2)
            op1.set_input_input2(data);
        else if (idx == 3)
            op1.set_input_input3(data);
        else if (idx == 4)
            op1.set_input_input4(data);
        else if (idx == 5)
            op1.set_input_mul0_x(data);
        else if (idx == 6)
            op1.set_input_mul1_x(data);
        else if (idx == 7)
            op1.set_input_mul2_x(data);
        else if (idx == 8)
            op1.set_input_mul3_x(data);
        else if (idx == 9)
            op1.set_input_add2_y(data);
        inputs.push_back(data);
    };

    // input0=2, input1=3, input2=4, input3=10, input4=0.1, mul0_x=0.9, mul1_x=0.999, mul2_x=0.999, mul3_x=0.999,
    // add2_y=1e-8
    float vals[10] = {2.0f, 3.0f, 4.0f, 10.0f, 0.1f, 0.9f, 0.999f, 0.999f, 0.999f, 1e-8f};
    for (int i = 0; i < 10; i++)
        mkInput(i, nullptr, vals[i]);

    // Define 3 outputs
    TensorDesc out0Desc(ge::Shape(shape), FORMAT_ND, inDtype);
    op1.update_output_desc_input1(out0Desc);
    TensorDesc out1Desc(ge::Shape(shape), FORMAT_ND, inDtype);
    op1.update_output_desc_input2(out1Desc);
    TensorDesc out2Desc(ge::Shape(shape), FORMAT_ND, inDtype);
    op1.update_output_desc_input3(out2Desc);

    outputs.push_back(op1);
    return SUCCESS;
}

static bool VerifyOutputs(const std::vector<ge::Tensor>& output, const float expected[3])
{
    if (output.size() != 3) {
        printf("Expected 3 outputs, got %zu\n", output.size());
        return false;
    }
    bool pass = true;
    for (size_t i = 0; i < 3; ++i) {
        // This example constructs three FP32 outputs with shape [4].
        if (output[i].GetTensorDesc().GetDataType() != DT_FLOAT || output[i].GetSize() != 4 * sizeof(float) ||
            output[i].GetData() == nullptr) {
            printf("input%zu_out has invalid dtype, data length or data pointer\n", i + 1);
            pass = false;
            continue;
        }
        for (size_t j = 0; j < 4; ++j) {
            float actual;
            std::memcpy(&actual, output[i].GetData() + j * sizeof(actual), sizeof(actual));
            // All elements in this fixed example have the same finite CPU expectation.
            if (!std::isfinite(actual) || !std::isfinite(expected[i]) || std::abs(actual - expected[i]) > 1e-4f) {
                printf("input%zu_out[%zu] mismatch: NPU=%f CPU=%f\n", i + 1, j, actual, expected[i]);
                pass = false;
            }
        }
    }
    return pass;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_adam";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [GEIR]: Start GEInitialize\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    DataType inDtype = DT_FLOAT;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: CreateOppInGraph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [GEIR]: Creating session\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [GEIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    printf("%s - INFO - [GEIR]: AddGraph done, running...\n", GetTime().c_str());

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [GEIR]: RunGraph failed\n", GetTime().c_str());
        ge::AscendString err = ge::GEGetErrorMsgV2();
        printf("GE Error: %s\n", err.GetString());
        ge::AscendString warn = ge::GEGetWarningMsgV2();
        printf("GE Warning: %s\n", warn.GetString());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [GEIR]: RunGraph success\n", GetTime().c_str());

    // CPU verification (same logic as aclnn test)
    float input0 = 2.0f, input1 = 3.0f, input2 = 4.0f, input3 = 10.0f, input4 = 0.1f, mul0_x = 0.9f, mul1_x = 0.999f,
          mul2_x = 0.999f, mul3_x = 0.999f, add2_y = 1e-8f;
    float cpu_o0 = input1 * mul2_x + input0 * input0 * mul3_x;
    float cpu_o1 = mul0_x * input2 + input0 * mul1_x;
    float cpu_o2 = input3 - cpu_o1 / (std::sqrt(cpu_o0) + add2_y) * input4;
    printf("CPU expected: input1_out=%f input2_out=%f input3_out=%f\n", cpu_o0, cpu_o1, cpu_o2);

    const float expected[3] = {cpu_o0, cpu_o1, cpu_o2};
    const bool pass = VerifyOutputs(output, expected);

    // Print only after validating output count, data pointers and byte lengths.
    if (pass) {
        for (size_t i = 0; i < output.size(); ++i) {
            printf("output[%zu] (size=4): ", i);
            for (size_t j = 0; j < 4; ++j) {
                float value;
                std::memcpy(&value, output[i].GetData() + j * sizeof(float), sizeof(value));
                printf("%f ", value);
            }
            printf("\n");
        }
    }

    ge::GEFinalize();
    if (pass) {
        printf("\n=== GEIR RESULT: PASS ===\n");
        return SUCCESS;
    } else {
        printf("\n=== GEIR RESULT: FAIL ===\n");
        return FAILED;
    }
}
