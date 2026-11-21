/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_geir_situ_glu_group_grad.cpp
 * @brief GE IR graph-mode call example - SituGluGroupGrad (optional weight / group_index, required grad_weight)
 *
 * Case "full":  bs=5, dimH=16, weight + cumsum group_index{1,2,3} (realBs=3) + grad_weight, fp32.
 * Case "base":  bs=2, dimH=16, no weight / group_index (regression baseline), empty grad_weight [0], fp32.
 * Outputs are compared against a host-side golden; padding rows must stay MAGIC (untouched).
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"
#include "acl/acl.h"

#include "../op_graph/situ_glu_group_grad_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr float MAGIC = -768.0f;
constexpr double TOL = 1e-4;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

Tensor MakeTensor(const std::vector<int64_t>& shape, DataType dtype, const void* data, size_t len)
{
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    return Tensor(desc, reinterpret_cast<uint8_t*>(const_cast<void*>(data)), len);
}

// golden (double), mirrors the verified backward formulas
void Golden(int64_t bs, int64_t dimH, double beta, double linearBeta, bool activateLeft, bool withWeight,
            bool withGradW, const std::vector<float>& gradY, const std::vector<float>& x,
            const std::vector<float>& weight, int64_t realBs, std::vector<double>& gradX, std::vector<double>& gradW)
{
    int64_t dim2H = dimH * 2;
    gradX.assign(bs * dim2H, MAGIC);
    gradW.assign(bs, MAGIC);
    for (int64_t n = 0; n < realBs; ++n) {
        double w = withWeight ? static_cast<double>(weight[n]) : 1.0;
        double wsum = 0.0;
        for (int64_t h = 0; h < dimH; ++h) {
            int64_t gateIdx = n * dim2H + (activateLeft ? h : dimH + h);
            int64_t upIdx = n * dim2H + (activateLeft ? dimH + h : h);
            double g = x[gateIdx];
            double u = x[upIdx];
            double gy = gradY[n * dimH + h];
            double t = std::tanh(g / beta);
            double s = 1.0 / (1.0 + std::exp(-g));
            double a = beta * t * s;
            double upPrime = u;
            double upGrad = 1.0;
            if (linearBeta > 0.0) {
                double tu = std::tanh(u / linearBeta);
                upPrime = linearBeta * tu;
                upGrad = 1.0 - tu * tu;
            }
            wsum += gy * a * upPrime;
            gradX[gateIdx] = gy * upPrime * s * ((1.0 - t * t) + beta * t * (1.0 - s)) * w;
            gradX[upIdx] = gy * a * upGrad * w;
        }
        if (withGradW) {
            gradW[n] = wsum;
        }
    }
}

int CheckOutputs(const char* name, const std::vector<ge::Tensor>& output, int64_t bs, int64_t dimH, bool withGradW,
                 const std::vector<double>& expectX, const std::vector<double>& expectW)
{
    int64_t dim2H = dimH * 2;
    // grad_weight is a required output: it is materialized (empty [0] when weight is absent).
    if (output.size() < 1 || (withGradW && output.size() != 2U)) {
        printf("  [FAIL] %s: output num %zu, expect %s\n", name, output.size(), withGradW ? "2" : ">=1");
        return FAILED;
    }
    if (!withGradW && output.size() == 2U) {
        // no weight -> grad_weight must be an empty tensor
        const int64_t gradWElemNum = static_cast<int64_t>(output[1].GetTensorDesc().GetShape().GetShapeSize());
        if (gradWElemNum != 0) {
            printf("  [FAIL] %s: grad_weight should be empty, got %ld elements\n", name, gradWElemNum);
            return FAILED;
        }
    }
    const float* gradXData = reinterpret_cast<const float*>(output[0].GetData());
    int64_t errCnt = 0;
    for (int64_t i = 0; i < bs * dim2H; ++i) {
        if (expectX[i] == static_cast<double>(MAGIC)) {
            // GE allocates zero-initialized output buffers; a "not written" padding row therefore
            // reads 0. The authoritative not-written proof (magic prefill) is in the aclnn example.
            if (gradXData[i] != 0.0f) {
                printf("  [FAIL] %s: padding grad_x[%ld] written: %f\n", name, i, gradXData[i]);
                errCnt++;
            }
        } else if (std::fabs(gradXData[i] - expectX[i]) > TOL + TOL * std::fabs(expectX[i])) {
            printf("  [FAIL] %s: grad_x[%ld] = %f, expect %f\n", name, i, gradXData[i], expectX[i]);
            errCnt++;
        }
    }
    if (withGradW) {
        const float* gradWData = reinterpret_cast<const float*>(output[1].GetData());
        for (int64_t n = 0; n < bs; ++n) {
            if (expectW[n] == static_cast<double>(MAGIC)) {
                if (gradWData[n] != 0.0f) {
                    printf("  [FAIL] %s: padding grad_weight[%ld] written: %f\n", name, n, gradWData[n]);
                    errCnt++;
                }
            } else if (std::fabs(gradWData[n] - expectW[n]) > TOL + TOL * std::fabs(expectW[n])) {
                printf("  [FAIL] %s: grad_weight[%ld] = %f, expect %f\n", name, n, gradWData[n], expectW[n]);
                errCnt++;
            }
        }
    }
    if (errCnt == 0) {
        printf("  [PASS] %s\n", name);
    }
    return errCnt == 0 ? SUCCESS : FAILED;
}

int BuildAndRun(Graph& graph, const std::vector<ge::Tensor>& inputs, const std::vector<Operator>& inputOps,
                const std::vector<Operator>& outputOps, std::vector<ge::Tensor>& outputs)
{
    Status ret = SUCCESS;
    auto g = graph;
    if (!inputOps.empty() && !outputOps.empty()) {
        g.SetInputs(inputOps).SetOutputs(outputOps);
    }
    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        return FAILED;
    }
    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, g, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed, ret=%u\n", GetTime().c_str(), ret);
        delete session;
        return FAILED;
    }
    ret = session->RunGraph(graph_id, inputs, outputs);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed, ret=%u\n", GetTime().c_str(), ret);
        delete session;
        return FAILED;
    }
    delete session;
    return SUCCESS;
}

// full case: weight + cumsum group_index{1,2,3} + grad_weight, bs=5, dimH=16
int RunFullCase()
{
    const int64_t bs = 5;
    const int64_t dimH = 16;
    const int64_t dim2H = dimH * 2;
    const double beta = 1.3;
    const double linearBeta = 0.8;
    // group_list_type 0 (cumsum): the last element is the total row count, so realBs = 3.
    std::vector<int64_t> groupIndex = {1, 2, 3};
    int64_t realBs = 3;

    std::vector<float> gradY(bs * dimH);
    std::vector<float> x(bs * dim2H);
    std::vector<float> weight(bs);
    for (size_t i = 0; i < gradY.size(); ++i) {
        gradY[i] = std::fmod(static_cast<float>(i) * 0.37f + 0.11f, 4.0f) - 2.0f;
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::fmod(static_cast<float>(i) * 0.53f + 0.29f, 4.0f) - 2.0f;
    }
    for (int64_t i = 0; i < bs; ++i) {
        weight[i] = 0.5f + 0.25f * static_cast<float>(i % 4);
    }
    std::vector<float> gradXInit(bs * dim2H, MAGIC);
    std::vector<float> gradWInit(bs, MAGIC);

    Graph graph("situ_glu_group_grad_geir_full");
    auto op = op::SituGluGroupGrad("situ_glu_group_grad_full");

    auto dataGradY = op::Data("data_grad_y").set_attr_index(0);
    dataGradY.update_input_desc_x(TensorDesc(ge::Shape({bs, dimH}), FORMAT_ND, DT_FLOAT));
    graph.AddOp(dataGradY);
    op.set_input_grad_y(dataGradY);

    auto dataX = op::Data("data_x").set_attr_index(1);
    dataX.update_input_desc_x(TensorDesc(ge::Shape({bs, dim2H}), FORMAT_ND, DT_FLOAT));
    graph.AddOp(dataX);
    op.set_input_x(dataX);

    auto dataWeight = op::Data("data_weight").set_attr_index(2);
    dataWeight.update_input_desc_x(TensorDesc(ge::Shape({bs}), FORMAT_ND, DT_FLOAT));
    graph.AddOp(dataWeight);
    op.set_input_weight(dataWeight);

    auto dataGroupIndex = op::Data("data_group_index").set_attr_index(3);
    dataGroupIndex.update_input_desc_x(
        TensorDesc(ge::Shape({static_cast<int64_t>(groupIndex.size())}), FORMAT_ND, DT_INT64));
    graph.AddOp(dataGroupIndex);
    op.set_input_group_index(dataGroupIndex);

    op.update_output_desc_grad_x(TensorDesc(ge::Shape({bs, dim2H}), FORMAT_ND, DT_FLOAT));
    op.update_output_desc_grad_weight(TensorDesc(ge::Shape({bs}), FORMAT_ND, DT_FLOAT));

    op.set_attr_dim(-1);
    op.set_attr_beta(static_cast<float>(beta));
    op.set_attr_linear_beta(static_cast<float>(linearBeta));
    op.set_attr_activate_left(true);
    op.set_attr_group_list_type(0); // 0: cumsum group_index, 1: count group_index

    std::vector<ge::Tensor> inputs = {
        MakeTensor({bs, dimH}, DT_FLOAT, gradY.data(), gradY.size() * sizeof(float)),
        MakeTensor({bs, dim2H}, DT_FLOAT, x.data(), x.size() * sizeof(float)),
        MakeTensor({bs}, DT_FLOAT, weight.data(), weight.size() * sizeof(float)),
        MakeTensor({static_cast<int64_t>(groupIndex.size())}, DT_INT64, groupIndex.data(),
                   groupIndex.size() * sizeof(int64_t)),
    };
    std::vector<Operator> inputOps = {dataGradY, dataX, dataWeight, dataGroupIndex};
    std::vector<Operator> outputOps = {op};

    std::vector<ge::Tensor> outputs;
    if (BuildAndRun(graph, inputs, inputOps, outputOps, outputs) != SUCCESS) {
        return FAILED;
    }

    std::vector<double> expectX;
    std::vector<double> expectW;
    Golden(bs, dimH, beta, linearBeta, true, true, true, gradY, x, weight, realBs, expectX, expectW);
    return CheckOutputs("geir_full(weight+group_index+grad_weight)", outputs, bs, dimH, true, expectX, expectW);
}

// baseline case: no optional inputs / outputs, bs=2, dimH=16
int RunBaseCase()
{
    const int64_t bs = 2;
    const int64_t dimH = 16;
    const int64_t dim2H = dimH * 2;
    const double beta = 4.0;
    const double linearBeta = 25.0;

    std::vector<float> gradY(bs * dimH);
    std::vector<float> x(bs * dim2H);
    std::vector<float> weight(bs, 1.0f);
    for (size_t i = 0; i < gradY.size(); ++i) {
        gradY[i] = std::fmod(static_cast<float>(i) * 0.37f + 0.11f, 4.0f) - 2.0f;
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::fmod(static_cast<float>(i) * 0.53f + 0.29f, 4.0f) - 2.0f;
    }

    Graph graph("situ_glu_group_grad_geir_base");
    auto op = op::SituGluGroupGrad("situ_glu_group_grad_base");

    auto dataGradY = op::Data("data_grad_y").set_attr_index(0);
    dataGradY.update_input_desc_x(TensorDesc(ge::Shape({bs, dimH}), FORMAT_ND, DT_FLOAT));
    graph.AddOp(dataGradY);
    op.set_input_grad_y(dataGradY);

    auto dataX = op::Data("data_x").set_attr_index(1);
    dataX.update_input_desc_x(TensorDesc(ge::Shape({bs, dim2H}), FORMAT_ND, DT_FLOAT));
    graph.AddOp(dataX);
    op.set_input_x(dataX);

    op.update_output_desc_grad_x(TensorDesc(ge::Shape({bs, dim2H}), FORMAT_ND, DT_FLOAT));
    // no weight -> grad_weight is a required empty output
    op.update_output_desc_grad_weight(TensorDesc(ge::Shape({0}), FORMAT_ND, DT_FLOAT));

    op.set_attr_dim(-1);
    op.set_attr_beta(static_cast<float>(beta));
    op.set_attr_linear_beta(static_cast<float>(linearBeta));
    op.set_attr_activate_left(true);

    std::vector<ge::Tensor> inputs = {
        MakeTensor({bs, dimH}, DT_FLOAT, gradY.data(), gradY.size() * sizeof(float)),
        MakeTensor({bs, dim2H}, DT_FLOAT, x.data(), x.size() * sizeof(float)),
    };
    std::vector<Operator> inputOps = {dataGradY, dataX};
    std::vector<Operator> outputOps = {op};

    std::vector<ge::Tensor> outputs;
    if (BuildAndRun(graph, inputs, inputOps, outputOps, outputs) != SUCCESS) {
        return FAILED;
    }

    std::vector<double> expectX;
    std::vector<double> expectW;
    Golden(bs, dimH, beta, linearBeta, true, false, false, gradY, x, weight, bs, expectX, expectW);
    return CheckOutputs("geir_base(regression)", outputs, bs, dimH, false, expectX, expectW);
}
} // namespace

bool CheckHardwareSupport()
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        printf("Warning: Cannot get SOC name, skip hardware check\n");
        return true;
    }
    printf("Current SOC: %s\n", socName);
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }
    printf("Warning: SituGluGroupGrad only supports Ascend950, current SOC '%s' is not supported. Skip test.\n",
           socName);
    return false;
}

int main()
{
    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed, ret=%u\n", GetTime().c_str(), ret);
        return FAILED;
    }

    if (!CheckHardwareSupport()) {
        printf("\n=== GEIR Test SKIPPED (hardware not supported) ===\n");
        ge::GEFinalize();
        return SUCCESS;
    }

    int failed = 0;
    failed += RunBaseCase() != SUCCESS ? 1 : 0;
    failed += RunFullCase() != SUCCESS ? 1 : 0;
    printf(failed == 0 ? "ALL GEIR CASES PASS\n" : "%d GEIR CASE(S) FAILED\n", failed);

    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    return failed == 0 ? SUCCESS : FAILED;
}
