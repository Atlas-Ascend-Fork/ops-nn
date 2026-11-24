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
 * \file test_geir_dua_quantize_add_layer_norm.cpp
 * \brief
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
#include "securec.h"

#include "../op_graph/dua_quantize_add_layer_norm_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;
#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape)                                                 \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                            \
                      placeholder##intputIndex##_desc, intputDtype, 2);                                             \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);                                  \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                                 \
    input.push_back(tensor_placeholder##intputIndex);                                                               \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    add1.set_input_##intputName(placeholder##intputIndex);                                                          \
    inputs.push_back(placeholder##intputIndex)

#define ADD_INPUT_ATTR(attrName, attrValue) add1.set_attr_##attrName(attrValue)

#define ADD_CONST_INPUT(intputIndex, intputName, intputDtype, inputShape)                                           \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Const("placeholder" + intputIndex);                                         \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                            \
                      placeholder##intputIndex##_desc, intputDtype, 2);                                             \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.SetAttr("value", tensor_placeholder##intputIndex);                                     \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                                 \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    add1.set_input_##intputName(placeholder##intputIndex);                                                          \
    add1.update_input_desc_##intputName(placeholder##intputIndex##_desc);                                           \
    inputs.push_back(placeholder##intputIndex)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                        \
    TensorDesc outputName##outputIndex##_desc_ = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    add1.update_output_desc_##outputName(outputName##outputIndex##_desc)

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
    uint32_t dilation = 1;
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        dilation = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_BF16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_UINT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_UINT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_INT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_UINT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_INT8) {
        dilation = oneByte;
    }
    return dilation;
}

// fp32 → bf16/fp16 位型转换的移位与宽度常量。
namespace {
constexpr uint32_t BF16_DROP_BITS = 16;                        // bf16 = fp32 舍去低 16 位
constexpr uint32_t BF16_ROUND_BIAS = 0x7FFFU;                  // 就近偶舍入偏置（2^15 - 1）
constexpr uint32_t FP16_SIGN_SHIFT = 16;                       // fp32 符号位(bit31) → fp16 符号位(bit15)
constexpr uint32_t FP16_EXP_OFFSET = 10;                       // fp16 指数字段起始位
constexpr uint32_t FP16_MANT_DROP_BITS = 13;                   // fp32 尾数 23 位 → fp16 尾数 10 位
constexpr uint32_t NARROW_DTYPE_ELEM_BYTES = sizeof(uint16_t); // fp16/bf16 元素宽度（字节）
} // namespace

// float → bf16 位型：bf16 即 fp32 的高 16 位（就近取整）。
uint16_t FloatToBf16Bits(float v)
{
    uint32_t bits = 0;
    if (memcpy_s(&bits, sizeof(bits), &v, sizeof(v)) != EOK) {
        printf("%s - ERROR - [XIR]: memcpy_s failed for float bits\n", GetTime().c_str());
        return 0;
    }
    uint32_t rounded = bits + BF16_ROUND_BIAS + ((bits >> BF16_DROP_BITS) & 1U); // round-to-nearest-even
    return static_cast<uint16_t>(rounded >> BF16_DROP_BITS);
}

// float → fp16 位型。本样例只用整数值（2.0f、0.0f），
// 故仅处理规格化数与上下溢，不生成次正规数。
uint16_t FloatToFp16Bits(float v)
{
    uint32_t bits = 0;
    if (memcpy_s(&bits, sizeof(bits), &v, sizeof(v)) != EOK) {
        printf("%s - ERROR - [XIR]: memcpy_s failed for float bits\n", GetTime().c_str());
        return 0;
    }
    const uint32_t sign = (bits >> FP16_SIGN_SHIFT) & 0x8000U;
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    const uint32_t mant = bits & 0x7FFFFFU;
    if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00U); // 上溢 → inf
    }
    if (exp <= 0) {
        return static_cast<uint16_t>(sign); // 下溢 → ±0
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << FP16_EXP_OFFSET) |
                                 (mant >> FP16_MANT_DROP_BITS));
}

// 按张量**声明的 dtype** 填充常量数据：缓冲区元素宽度必须与 dtype 一致，
// 否则 GE 只会取到错误的一半字节（如按 fp32 写入却以 bf16 解读 → 全 0）。
int32_t GenOnesData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    if (size == 0) {
        printf("%s - ERROR - [XIR]: empty shape, nothing to generate\n", GetTime().c_str());
        return FAILED;
    }

    uint32_t elemBytes = GetDataTypeSize(data_type);
    uint32_t data_len = static_cast<uint32_t>(size) * elemBytes;
    if (data_type == ge::DT_FLOAT) {
        float* pData = new (std::nothrow) float[size];
        if (pData == nullptr) {
            return FAILED;
        }
        for (size_t i = 0; i < size; ++i) {
            pData[i] = value;
        }
        input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
        return SUCCESS;
    }
    if (elemBytes != NARROW_DTYPE_ELEM_BYTES) {
        printf("%s - ERROR - [XIR]: unsupported dtype for data generation: %d\n", GetTime().c_str(),
               static_cast<int32_t>(data_type));
        return FAILED;
    }
    uint16_t* pData = new (std::nothrow) uint16_t[size];
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        pData[i] = (data_type == ge::DT_BF16) ? FloatToBf16Bits(value) : FloatToFp16Bits(value);
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(const string& path, size_t size, const uint8_t* data)
{
    if (data == nullptr || size == 0) {
        printf("%s - ERROR - [XIR]: invalid data buffer or size for %s\n", GetTime().c_str(), path.c_str());
        return FAILED;
    }
    FILE* fp = fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        printf("%s - ERROR - [XIR]: cannot open %s for export\n", GetTime().c_str(), path.c_str());
        return FAILED;
    }
    const size_t written = fwrite(data, 1, size, fp);
    const int closeStatus = fclose(fp);
    if (written != size || closeStatus != 0) {
        printf("%s - ERROR - [XIR]: cannot export %s\n", GetTime().c_str(), path.c_str());
        return FAILED;
    }
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto add1 = op::DuaQuantizeAddLayerNorm("add1");
    std::vector<int64_t> x1Shape = {4, 1, 8};
    std::vector<int64_t> x2Shape = {4, 1, 8};
    std::vector<int64_t> gammaShape = {8};
    std::vector<int64_t> betaShape = {8};
    std::vector<int64_t> biasShape = {8};
    std::vector<int64_t> scales1Shape = {8};
    std::vector<int64_t> scales2Shape = {8};
    std::vector<int64_t> points1Shape = {8};
    std::vector<int64_t> points2Shape = {8};
    ADD_INPUT(1, x1, inDtype, x1Shape);
    ADD_INPUT(2, x2, inDtype, x2Shape);
    ADD_INPUT(3, gamma, inDtype, gammaShape);
    ADD_INPUT(4, beta, inDtype, betaShape);
    ADD_INPUT(5, bias, inDtype, biasShape);
    ADD_INPUT(6, scales1, inDtype, scales1Shape);
    ADD_INPUT(7, scales2, inDtype, scales2Shape);
    ADD_INPUT(8, zero_points1, inDtype, points1Shape);
    ADD_INPUT(9, zero_points2, inDtype, points2Shape);
    ADD_INPUT_ATTR(dtype, DT_INT32);

    outputs.push_back(add1);
    // 添加完毕
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
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

    DataType inDtype = DT_BF16;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {

    };
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
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

    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(input_file, data_size, input_data_i) != SUCCESS) {
            delete session;
            GEFinalize();
            return FAILED;
        }
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(output_file, data_size, output_data_i) != SUCCESS) {
            delete session;
            GEFinalize();
            return FAILED;
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
