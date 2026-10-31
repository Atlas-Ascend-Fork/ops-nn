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
 * \file test_geir_quant_batch_matmul_v3.cpp
 * \brief INT8 pertoken-perchannel quant matmul with a host-side golden check.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

#include "graph.h"
#include "tensor.h"
#include "ge_api.h"
#include "array_ops.h"
#include "../op_graph/quant_batch_matmul_v3_proto.h"

namespace {
constexpr int64_t M = 16;
constexpr int64_t K = 32;
constexpr int64_t N = 64;

template <typename T>
ge::op::Data AddInput(const char* name, int64_t index, const std::vector<int64_t>& shape, ge::DataType dtype,
                      const std::vector<T>& values, std::vector<ge::Tensor>& tensors)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    auto data = ge::op::Data(name).set_attr_index(index);
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    tensors.emplace_back(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T));
    return data;
}

float DecodeFloat16(uint16_t bits)
{
    const int exponent = (bits >> 10) & 0x1F;
    const int fraction = bits & 0x3FF;
    if (exponent == 31) {
        return NAN;
    }
    const float value = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24) :
                                        std::ldexp(1.0f + fraction / 1024.0f, exponent - 15);
    return (bits & 0x8000) ? -value : value;
}

int CheckOutput(const std::vector<ge::Tensor>& outputs, const std::vector<int8_t>& x1, const std::vector<int8_t>& x2,
                const std::vector<float>& scale, const std::vector<float>& pertokenScale)
{
    if (outputs.size() != 1 || outputs[0].GetSize() != M * N * sizeof(uint16_t) ||
        outputs[0].GetTensorDesc().GetDataType() != ge::DT_FLOAT16 ||
        outputs[0].GetTensorDesc().GetShape().GetDims() != std::vector<int64_t>({M, N}) ||
        outputs[0].GetData() == nullptr) {
        std::printf("[XIR]: Unexpected output tensor\n");
        return 1;
    }
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            int32_t accumulator = 0;
            for (int64_t k = 0; k < K; ++k) {
                // x2 is stored as (N, K) and transpose_x2 is true.
                accumulator += static_cast<int32_t>(x1[m * K + k]) * x2[n * K + k];
            }
            const float expected = accumulator * scale[n] * pertokenScale[m];
            uint16_t bits = 0;
            std::memcpy(&bits, outputs[0].GetData() + (m * N + n) * sizeof(bits), sizeof(bits));
            const float actual = DecodeFloat16(bits);
            if (!std::isfinite(actual) || std::fabs(actual - expected) > 1e-3f + 1e-3f * std::fabs(expected)) {
                std::printf("[XIR]: Mismatch at (%lld, %lld): actual=%g, expected=%g\n", static_cast<long long>(m),
                            static_cast<long long>(n), actual, expected);
                return 1;
            }
        }
    }
    std::printf("[XIR]: Golden comparison PASS\n");
    return 0;
}

int RunExample()
{
    std::vector<int8_t> x1(M * K);
    std::vector<int8_t> x2(N * K);
    std::vector<float> scale(N);
    std::vector<float> pertokenScale(M);
    for (int64_t i = 0; i < M * K; ++i) {
        x1[i] = static_cast<int8_t>(i % 7 - 3);
    }
    for (int64_t i = 0; i < N * K; ++i) {
        x2[i] = static_cast<int8_t>(i % 5 - 2);
    }
    for (int64_t n = 0; n < N; ++n) {
        scale[n] = (n % 4 + 1) / 8.0f;
    }
    for (int64_t m = 0; m < M; ++m) {
        pertokenScale[m] = (m % 3 + 1) / 4.0f;
    }

    std::vector<ge::Tensor> inputs;
    auto x1Data = AddInput("x1", 0, {M, K}, ge::DT_INT8, x1, inputs);
    auto x2Data = AddInput("x2", 1, {N, K}, ge::DT_INT8, x2, inputs);
    auto scaleData = AddInput("scale", 2, {N}, ge::DT_FLOAT, scale, inputs);
    auto pertokenData = AddInput("pertoken_scale", 3, {M}, ge::DT_FLOAT, pertokenScale, inputs);
    auto matmul = ge::op::QuantBatchMatmulV3("quant_batch_matmul_v3");
    matmul.set_input_x1(x1Data)
        .set_input_x2(x2Data)
        .set_input_scale(scaleData)
        .set_input_pertoken_scale(pertokenData)
        .set_attr_dtype(static_cast<int64_t>(ge::DT_FLOAT16))
        .set_attr_transpose_x1(false)
        .set_attr_transpose_x2(true)
        .set_attr_group_size(0);
    matmul.update_output_desc_y(ge::TensorDesc(ge::Shape({M, N}), ge::FORMAT_ND, ge::DT_FLOAT16));

    ge::Graph graph("quant_batch_matmul_v3_example");
    std::vector<ge::Operator> graphInputs = {x1Data, x2Data, scaleData, pertokenData};
    std::vector<ge::Operator> graphOutputs = {matmul};
    graph.SetInputs(graphInputs).SetOutputs(graphOutputs);
    std::map<ge::AscendString, ge::AscendString> options;
    ge::Session session(options);
    if (session.AddGraph(0, graph, options) != ge::SUCCESS) {
        std::printf("[XIR]: AddGraph failed\n");
        return 1;
    }
    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(0, inputs, outputs) != ge::SUCCESS) {
        std::printf("[XIR]: RunGraph failed\n");
        return 1;
    }
    return CheckOutput(outputs, x1, x2, scale, pertokenScale);
}
} // namespace

int main()
{
    std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(options) != ge::SUCCESS) {
        std::printf("[XIR]: GEInitialize failed\n");
        return 1;
    }
    // Destroy the session before finalizing GE.
    const int result = RunExample();
    if (ge::GEFinalize() != ge::SUCCESS) {
        std::printf("[XIR]: GEFinalize failed\n");
        return 1;
    }
    return result;
}
