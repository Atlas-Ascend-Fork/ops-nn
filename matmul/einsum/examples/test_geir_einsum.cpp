/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_einsum.cpp
 * \brief Einsum GE IR example (fp16 dynamic inputs, fp16 output).
 */

#include <cstdio>
#include <memory>
#include <ctime>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/einsum_proto.h"

#define GEIR_FAILED -1
#define GEIR_SUCCESS 0

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
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        return fourByte;
    } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        return twoByte;
    }
    return 1;
}

uint16_t FloatToHalfBits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

uint16_t FloatToBFloat16Bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    uint32_t mant = (bits >> 16) & 0x007F;
    return static_cast<uint16_t>(sign | mant); // bf16: 截取fp32高16位(符号+8位指数+7位尾数)
}

// 按dtype编码数据:float32直接写float值,float16/bfloat16写位模式(布局不同,分别转换)
int32_t GenOnesData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    const size_t data_len = size * GetDataTypeSize(data_type);
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr) {
        return GEIR_FAILED;
    }
    if (data_type == ge::DT_FLOAT) {
        float* pFloat = reinterpret_cast<float*>(pData);
        for (size_t i = 0; i < size; ++i) {
            pFloat[i] = value;
        }
    } else if (data_type == ge::DT_FLOAT16 || data_type == ge::DT_BF16) {
        const uint16_t bits = (data_type == ge::DT_FLOAT16) ? FloatToHalfBits(value) : FloatToBFloat16Bits(value);
        uint16_t* pUint16 = reinterpret_cast<uint16_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            pUint16[i] = bits;
        }
    } else if (data_type == ge::DT_INT32) {
        int32_t* pInt32 = reinterpret_cast<int32_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            pInt32[i] = static_cast<int32_t>(value);
        }
    } else {
        delete[] pData;
        return GEIR_FAILED;
    }
    input_tensor.SetData(pData, data_len); // session接管pData内存,run后统一释放
    input_tensor.SetTensorDesc(input_tensor_desc);
    input_tensor.SetPlacement(ge::kPlacementHost);
    return GEIR_SUCCESS;
}

int32_t CompareWithGolden(uint8_t* output_data, int64_t outSize, DataType data_type, float golden)
{
    float absTol = 1e-3;
    float relTol = 1e-3;
    if (output_data == nullptr) {
        printf("[XIR]: compare failed, output data is nullptr\n");
        return GEIR_FAILED;
    }
    float maxDiff = 0.0f;
    float maxRel = 0.0f;
    int64_t badCount = 0;
    for (int64_t i = 0; i < outSize; i++) {
        float outputValue = 0.0f;
        if (data_type == ge::DT_FLOAT) {
            outputValue = reinterpret_cast<float*>(output_data)[i];
        } else if (data_type == ge::DT_FLOAT16 || data_type == ge::DT_BF16) {
            const uint16_t bits = reinterpret_cast<uint16_t*>(output_data)[i];
            uint32_t fBits;
            if (data_type == ge::DT_BF16) {
                fBits = static_cast<uint32_t>(bits) << 16; // bf16: 高16位即fp32布局
            } else {
                const uint32_t sign = static_cast<uint32_t>((bits >> 15) & 0x1) << 31;
                const int32_t exp = (bits >> 10) & 0x1F;
                const uint32_t mant = bits & 0x3FF;
                if (exp == 0) {
                    fBits = sign;
                } else if (exp == 31) {
                    fBits = sign | 0x7F800000 | (mant << 13);
                } else {
                    fBits = sign | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
                }
            }
            memcpy(&outputValue, &fBits, sizeof(outputValue));
        } else if (data_type == ge::DT_INT32) {
            outputValue = static_cast<float>(reinterpret_cast<int32_t*>(output_data)[i]);
        } else {
            printf("[XIR]: compare failed, unsupported dtype %d\n", static_cast<int>(data_type));
            return GEIR_FAILED;
        }
        float diff = fabs(outputValue - golden);
        float rel = (fabs(golden) > 0) ? diff / fabs(golden) : diff;
        maxDiff = std::max(maxDiff, diff);
        maxRel = std::max(maxRel, rel);
        if (diff > absTol + relTol * fabs(static_cast<double>(golden))) {
            ++badCount;
        }
    }
    printf("[XIR]: golden compare (expect %g): max abs diff = %g, max rel diff = %g, mismatch = %ld / %ld -> %s\n",
           static_cast<double>(golden), maxDiff, maxRel, badCount, outSize, badCount == 0 ? "PASS" : "FAIL");
    return badCount == 0 ? GEIR_SUCCESS : GEIR_FAILED;
}

#define ADD_EINSUM_INPUT(inputIndex, inputDtype, inputShape)                                                           \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                      \
    auto placeholder##inputIndex = op::Data("placeholder" + std::to_string(inputIndex))                                \
                                       .set_attr_index((inputIndex) - 1);                                              \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND,      \
                                                           inputDtype);                                                \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                                   \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                               \
    placeholder##inputIndex##_desc.SetOriginFormat(FORMAT_ND);                                                         \
    Tensor tensor_placeholder##inputIndex;                                                                             \
    ret = GenOnesData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc, \
                      inputDtype, 2.0f);                                                                               \
    if (ret != GEIR_SUCCESS) {                                                                                         \
        printf("%s - ERROR - [XIR]: Generate inputTensors data failed\n", GetTime().c_str());                          \
        return GEIR_FAILED;                                                                                            \
    }                                                                                                                  \
    inputTensors.push_back(tensor_placeholder##inputIndex);                                                            \
    graph.AddOp(placeholder##inputIndex);                                                                              \
    tmpOp.set_dynamic_input_x((inputIndex) - 1, placeholder##inputIndex);                                              \
    tmpOp.update_dynamic_input_desc_x((inputIndex) - 1, placeholder##inputIndex##_desc);                               \
    inputOps.push_back(placeholder##inputIndex);

int32_t CreateEinsumInGraph(std::vector<ge::Tensor>& inputTensors, std::vector<Operator>& inputOps,
                            std::vector<Operator>& outputOps, Graph& graph)
{
    Status ret = GEIR_SUCCESS;
    auto tmpOp = op::Einsum("einsum");
    // pattern 005: abc,cd->abd, einsum_fusion_pass分解为BatchMatMul(无转置)
    const std::string equation = "abc,cd->abd";
    constexpr uint32_t inputNum = 2;
    std::vector<int64_t> x1Shape = {2, 16, 32};
    std::vector<int64_t> x2Shape = {32, 48};
    std::vector<int64_t> yShape = {2, 16, 48};
    tmpOp.set_attr_equation(equation);
    tmpOp.set_attr_N(inputNum);
    tmpOp.create_dynamic_input_x(inputNum);
    ADD_EINSUM_INPUT(1, ge::DT_FLOAT16, x1Shape);
    ADD_EINSUM_INPUT(2, ge::DT_FLOAT16, x2Shape);
    // y跟随x为fp16;期望y = sum(x1 * x2)按c轴求和 = 2.0 * 2.0 * 32 = 128.0
    TensorDesc y_desc = TensorDesc(ge::Shape(yShape), FORMAT_ND, ge::DT_FLOAT16);
    tmpOp.update_output_desc_y(y_desc);
    outputOps.push_back(tmpOp);
    if (!inputOps.empty() && !outputOps.empty()) {
        graph.SetInputs(inputOps).SetOutputs(outputOps);
    }
    return GEIR_SUCCESS;
}

int32_t RunEinsumExample(Graph& graph);

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    const char* graph_name = "tc_ge_irrun_test_einsum";
    Graph graph(graph_name);

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != GEIR_SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return GEIR_FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    // GEInitialize成功后所有退出路径必须GEFinalize,统一收口在此
    const int32_t runRet = RunEinsumExample(graph);

    printf("%s - INFO - [XIR]: Start to finalize ge\n", GetTime().c_str());
    if (ge::GEFinalize() != GEIR_SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
        return GEIR_FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ge success\n", GetTime().c_str());
    return runRet;
}

// 在GEInitialize与GEFinalize之间执行:建图、会话、运行与golden比对;内部任何失败直接返回,GEFinalize由main统一收口
int32_t RunEinsumExample(Graph& graph)
{
    Status ret = GEIR_SUCCESS;
    std::vector<Operator> inputOps{};
    std::vector<ge::Tensor> inputTensors;
    std::vector<Operator> outputOps{};
    ret = CreateEinsumInGraph(inputTensors, inputOps, outputOps, graph);
    if (ret != GEIR_SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir graph failed\n", GetTime().c_str());
        return GEIR_FAILED;
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    std::unique_ptr<ge::Session> session = std::make_unique<Session>(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return GEIR_FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != GEIR_SUCCESS) {
        printf("%s - ERROR - [XIR]: Session add ir compute graph failed\n", GetTime().c_str());
        return GEIR_FAILED;
    }
    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./einsum_dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, inputTensors, output);
    if (ret != GEIR_SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        return GEIR_FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int32_t compareRet = GEIR_SUCCESS;
    int output_num = static_cast<int>(output.size());
    for (int i = 0; i < output_num; i++) {
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        if (CompareWithGolden(output_data_i, output_shape, output[i].GetTensorDesc().GetDataType(), 128.0f) !=
            GEIR_SUCCESS) {
            compareRet = GEIR_FAILED;
        }
    }
    return compareRet;
}
