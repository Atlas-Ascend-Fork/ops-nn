/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_situ_glu_group_quant.cpp
 * \brief Graph-mode (C++ GE API) end-to-end test for the standalone
 *        SituGluGroupQuant port: build the op into a ge::Graph via the
 *        op_graph proto (ge::op::SituGluGroupQuant), run it through a
 *        ge::Session and compare y/y_scale against the recipes-semantics CPU
 *        reference (same caliber as the eager example). Two cases: all rows
 *        valid, and zero-token expert + padding rows (validRows only).
 */

#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "../../op_graph/situ_glu_group_quant_proto.h"

#define FAILED -1
#define SUCCESS 0

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

static bool g_precision_failed = false;

using namespace ge;

#define LOG_PRINT(message, ...) printf(message, ##__VA_ARGS__)

// ==================== dtype helpers ====================
uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        return 4;
    }
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        return 2;
    }
    if (dt == ge::DT_INT64 || dt == ge::DT_UINT64) {
        return 8;
    }
    // FP8 family (E4M3FN/E5M2/E8M0) and INT8
    return 1;
}

// ==================== BF16 / FP8 / E8M0 conversion helpers (recipes caliber) ====================
uint16_t FloatToBf16(float val)
{
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float Fp8E4M3ToFloat(uint8_t val)
{
    int32_t sign = (val >> 7) & 1;
    int32_t exp = (val >> 3) & 0xF;
    int32_t mant = val & 0x7;
    float result;
    if (exp == 0) {
        result = (mant == 0) ? 0.0f : (mant / 8.0f) * std::ldexp(1.0f, 1 - 7);
    } else if (exp == 0xF && mant == 0x7) {
        return sign ? -INFINITY : INFINITY;
    } else {
        result = (1.0f + mant / 8.0f) * std::ldexp(1.0f, exp - 7);
    }
    return sign ? -result : result;
}

uint8_t FloatToFp8E4M3(float val)
{
    if (std::isnan(val))
        return 0x7F;
    if (val == 0.0f)
        return std::signbit(val) ? 0x80 : 0x00;
    bool sign = val < 0;
    float absVal = std::fabs(val);
    if (absVal >= 448.0f) {
        return sign ? 0xFE : 0x7E;
    }
    int exp;
    float mant = std::frexp(absVal, &exp);
    exp = exp + 6;
    if (exp <= 0) {
        int32_t mantBits = static_cast<int32_t>(std::nearbyint(absVal * 8.0f * 64.0f));
        if (mantBits <= 0)
            return sign ? 0x80 : 0x00;
        if (mantBits > 7)
            mantBits = 7;
        return static_cast<uint8_t>((sign ? 0x80 : 0) | mantBits);
    }
    if (exp > 15) {
        return sign ? 0xFE : 0x7E;
    }
    int32_t mantBits = static_cast<int32_t>(std::nearbyint((mant * 2.0f - 1.0f) * 8.0f));
    if (mantBits < 0)
        mantBits = 0;
    if (mantBits > 7) {
        mantBits = 0;
        exp++;
        if (exp > 15) {
            return sign ? 0xFE : 0x7E;
        }
    }
    if (exp == 15 && mantBits >= 7) {
        mantBits = 6;
    }
    return static_cast<uint8_t>((sign ? 0x80 : 0) | (exp << 3) | mantBits);
}

// ==================== CPU reference (recipes SituGluGroupQuant semantics) ====================
void CpuSituGluGroupQuant(const std::vector<uint16_t>& xBf16, int64_t totalRows, int64_t colNum, float beta,
                          float linearBeta, const std::vector<int64_t>& groupIndex, std::vector<uint8_t>& yFp8,
                          std::vector<uint8_t>& y_scaleE8M0)
{
    const int64_t epr = colNum / 2;
    const int64_t blockSize = 32;
    const int64_t groupCount = (epr + blockSize - 1) / blockSize;
    const int64_t scaleColNum = (groupCount + 1) / 2 * 2;
    const int32_t emax = 8;

    int64_t validRows = 0;
    for (auto t : groupIndex) {
        validRows += t;
    }
    validRows = std::min(validRows, totalRows);

    // Two-path tanh (tanh.h reference, mirrors VFTanhTwoPathVf; aligned with
    // test_aclnn example). Horner uses fmaf() to mirror the kernel
    // FusedMulDstAdd single-rounding order.
    struct TwoPathTanh {
        float operator()(float x) const
        {
            const float c1 = -0.333327681f, c2 = 0.133152977f;
            const float c3 = -0.0523039624f, c4 = 0.0157396831f;
            float x2 = x * x;
            float p = x2 * c4;
            p = p + c3;
            p = fmaf(x2, p, c2);
            p = fmaf(x2, p, c1);
            p = p * x2;
            p = fmaf(x, p, x);
            float naive = 2.0f / (1.0f + std::exp(-2.0f * x)) - 1.0f;
            return std::fabs(x) >= 0.6f ? naive : p;
        }
    } twoPathTanh;
    auto SituBranch = [beta, &twoPathTanh](float g) {
        float x = g / beta; // plain division: subnormal beta -> +-inf, sigmoid branch consumes
        float tanhResult = twoPathTanh(x);
        float sigmoidResult = 1.0f / (1.0f + std::exp(-g));
        if (std::fabs(sigmoidResult) < 1.17549435e-38f) {
            sigmoidResult = 0.0f; // hardware FTZ: flush subnormal sigmoid results
        }
        return (tanhResult * sigmoidResult) * beta;
    };
    auto UpBranch = [linearBeta, &twoPathTanh](float u) { return twoPathTanh(u / linearBeta) * linearBeta; };

    for (int64_t r = 0; r < validRows; r++) {
        std::vector<float> act(epr);
        for (int64_t c = 0; c < epr; c++) {
            float gate = Bf16ToFloat(xBf16[r * colNum + c]);
            float up = Bf16ToFloat(xBf16[r * colNum + c + epr]);
            float gateBranch = SituBranch(gate);
            float upBranch = UpBranch(up);
            gateBranch = Bf16ToFloat(FloatToBf16(gateBranch));
            upBranch = Bf16ToFloat(FloatToBf16(upBranch));
            act[c] = Bf16ToFloat(FloatToBf16(gateBranch * upBranch));
        }
        for (int64_t g = 0; g < groupCount; g++) {
            int64_t start = g * blockSize;
            int64_t end = std::min(start + blockSize, epr);
            float maxVal = 0.0f;
            for (int64_t c = start; c < end; c++) {
                maxVal = std::max(maxVal, std::fabs(act[c]));
            }
            uint8_t e8m0;
            float recipScale;
            if (maxVal <= 0.0f) {
                e8m0 = 0;
                recipScale = 0.0f;
            } else if (std::isinf(maxVal) || std::isnan(maxVal)) {
                e8m0 = 0xFF;
                recipScale = 0.0f;
            } else {
                int32_t sharedExp = static_cast<int32_t>(std::floor(std::log2(maxVal))) - emax;
                if (sharedExp + 127 <= 0) {
                    e8m0 = 0;
                    recipScale = 0.0f;
                } else if (sharedExp == 0) {
                    e8m0 = static_cast<uint8_t>(sharedExp + 127);
                    recipScale = 0.0f;
                } else {
                    e8m0 = static_cast<uint8_t>(sharedExp + 127);
                    recipScale = std::ldexp(1.0f, -sharedExp);
                }
            }
            y_scaleE8M0[r * scaleColNum + g] = e8m0;
            for (int64_t c = start; c < end; c++) {
                float prod = Bf16ToFloat(FloatToBf16(act[c] * recipScale));
                yFp8[r * epr + c] = FloatToFp8E4M3(prod);
            }
        }
    }
}

// ==================== graph case runner ====================
// Builds Data(x) + Data(group_index) -> SituGluGroupQuant -> {y, y_scale},
// runs it and compares against the CPU reference within validRows.
int RunGraphCase(ge::Session* session, uint32_t graphId, int64_t totalRows, int64_t epr,
                 const std::vector<int64_t>& groupIndex, float beta, float linearBeta, const char* desc)
{
    LOG_PRINT("\n--- %s ---\n", desc);
    const int64_t colNum = epr * 2;
    const int64_t groupCount = (epr + 31) / 32;
    const int64_t scaleColNum = (groupCount + 1) / 2 * 2;
    Status ret;

    Graph graph("situ_glu_group_quant_case");
    auto opNode = ge::op::SituGluGroupQuant("situ_glu_group_quant");

    // input x: bf16 host tensor with pseudo-random data in [-2, 2]
    std::vector<int64_t> xShape = {totalRows, colNum};
    TensorDesc xDesc(ge::Shape(xShape), FORMAT_ND, DT_BF16);
    xDesc.SetRealDimCnt(xShape.size());
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetFormat(FORMAT_ND);
    std::vector<uint16_t> xHost(totalRows * colNum);
    uint32_t seed = 42;
    for (auto& v : xHost) {
        seed = seed * 1664525 + 1013904223;
        float f = static_cast<float>((seed >> 8) % 4000) / 1000.0f - 2.0f;
        v = FloatToBf16(f);
    }
    Tensor xTensor(xDesc, reinterpret_cast<uint8_t*>(xHost.data()), xHost.size() * sizeof(uint16_t));

    // input group_index: int64 host tensor
    std::vector<int64_t> tokensShape = {static_cast<int64_t>(groupIndex.size())};
    TensorDesc tokensDesc(ge::Shape(tokensShape), FORMAT_ND, DT_INT64);
    tokensDesc.SetRealDimCnt(tokensShape.size());
    tokensDesc.SetPlacement(ge::kPlacementHost);
    tokensDesc.SetFormat(FORMAT_ND);
    Tensor tokensTensor(tokensDesc, reinterpret_cast<uint8_t*>(const_cast<int64_t*>(groupIndex.data())),
                        groupIndex.size() * sizeof(int64_t));

    auto dataX = op::Data("data_x");
    dataX.update_input_desc_x(xDesc);
    auto dataTokens = op::Data("data_group_index");
    dataTokens.update_input_desc_x(tokensDesc);

    opNode.set_input_x(dataX);
    opNode.set_input_group_index(dataTokens);
    opNode.set_attr_beta(beta);
    opNode.set_attr_linear_beta(linearBeta);
    opNode.set_attr_activate_left(true); // keep recipes direction in the geir example
    // group_list_type defaults to 0 (cumsum, reserved) and would be rejected by
    // the host tiling; the example runs the supported count mode explicitly.
    opNode.set_attr_group_list_type(1);

    TensorDesc yDesc(ge::Shape({totalRows, epr}), FORMAT_ND, DT_FLOAT8_E4M3FN);
    opNode.update_output_desc_y(yDesc);
    TensorDesc scaleDesc(ge::Shape({totalRows, scaleColNum / 2, 2}), FORMAT_ND, DT_FLOAT8_E8M0);
    opNode.update_output_desc_y_scale(scaleDesc);

    graph.AddOp(dataX);
    graph.AddOp(dataTokens);
    graph.AddOp(opNode);
    std::vector<Operator> graphInputs = {dataX, dataTokens};
    std::vector<Operator> graphOutputs = {opNode};
    graph.SetInputs(graphInputs).SetOutputs(graphOutputs);

    std::map<AscendString, AscendString> graphOptions = {};
    ret = session->AddGraph(graphId, graph, graphOptions);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("AddGraph failed: %u\n", ret); return FAILED);

    std::vector<ge::Tensor> inputs = {xTensor, tokensTensor};
    std::vector<ge::Tensor> outputs;
    ret = session->RunGraph(graphId, inputs, outputs);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("RunGraph failed: %u\n", ret); return FAILED);
    CHECK_RET(outputs.size() == 2, LOG_PRINT("unexpected output count %zu\n", outputs.size()); return FAILED);

    // CPU reference
    std::vector<uint8_t> yRef(totalRows * epr, 0);
    std::vector<uint8_t> scaleRef(totalRows * scaleColNum, 0);
    CpuSituGluGroupQuant(xHost, totalRows, colNum, beta, linearBeta, groupIndex, yRef, scaleRef);

    int64_t validRows = 0;
    for (auto t : groupIndex) {
        validRows += t;
    }
    validRows = std::min(validRows, totalRows);

    const uint8_t* yGot = outputs[0].GetData();
    const uint8_t* scaleGot = outputs[1].GetData();
    int64_t yMismatch = 0;
    int64_t scaleMismatch = 0;
    for (int64_t r = 0; r < validRows; r++) {
        for (int64_t c = 0; c < epr; c++) {
            uint8_t got = yGot[r * epr + c];
            uint8_t exp = yRef[r * epr + c];
            bool sameValue = (got == exp) || (Fp8E4M3ToFloat(got) == Fp8E4M3ToFloat(exp));
            bool withinUlp = (got / 0x80 == exp / 0x80) &&
                             (std::abs(static_cast<int>(got % 0x80) - static_cast<int>(exp % 0x80)) <= 1);
            if (!sameValue && !withinUlp) {
                if (yMismatch < 5) {
                    LOG_PRINT("  y mismatch at [%ld,%ld]: got 0x%02x expect 0x%02x\n", r, c, got, exp);
                }
                yMismatch++;
            }
        }
        for (int64_t c = 0; c < scaleColNum; c++) {
            if (scaleGot[r * scaleColNum + c] != scaleRef[r * scaleColNum + c]) {
                if (scaleMismatch < 5) {
                    LOG_PRINT("  scale mismatch at [%ld,%ld]: got 0x%02x expect 0x%02x\n", r, c,
                              scaleGot[r * scaleColNum + c], scaleRef[r * scaleColNum + c]);
                }
                scaleMismatch++;
            }
        }
    }
    LOG_PRINT("  validRows=%ld/%ld | y mismatch=%ld | scale mismatch=%ld\n", validRows, totalRows, yMismatch,
              scaleMismatch);
    bool passed = (yMismatch == 0 && scaleMismatch == 0);
    LOG_PRINT("  -> %s\n", passed ? "PASS" : "FAIL");
    if (!passed) {
        g_precision_failed = true;
    }
    return passed ? SUCCESS : FAILED;
}

int main()
{
    LOG_PRINT("=== SituGluGroupQuant (standalone port) Graph-mode (GE API) Test ===\n");
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        LOG_PRINT("GEInitialize failed: %u\n", ret);
        return FAILED;
    }

    std::map<AscendString, AscendString> sessionOptions = {};
    ge::Session* session = new ge::Session(sessionOptions);
    if (session == nullptr) {
        ge::GEFinalize();
        return FAILED;
    }

    int failed = 0;
    // Case 1: 2 experts x 4 rows, all rows valid, EPR=128 (safe range)
    failed += (RunGraphCase(session, 0, 8, 128, {4, 4}, 1.0f, 2.0f, "GR1: 2 experts x4 rows, EPR=128") != SUCCESS);
    // Case 2: zero-token expert + padding rows (5/8 valid), hp=true
    failed += (RunGraphCase(session, 1, 8, 128, {2, 0, 3}, 0.5f, 1.5f, "GR2: zero-token expert + padding rows") !=
               SUCCESS);

    delete session;
    ge::GEFinalize();

    LOG_PRINT("\n=== Summary: %d/2 graph cases PASSED ===\n", 2 - failed);
    return (failed == 0) ? SUCCESS : FAILED;
}
