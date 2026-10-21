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
#include <ctime>
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

#include "../op_graph/sparse_slice_proto.h"

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr uint32_t kDeviceId = 0;

std::string GetTime()
{
    time_t now;
    time(&now);
    char buf[64] = {0};
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S,000", localtime(&now));
    return buf;
}

template <typename T>
ge::Tensor BuildTensor(const std::vector<int64_t>& shape, ge::DataType dtype, const std::vector<T>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T));
}

ge::op::Data CreateDataOp(const std::string& name, int32_t index, const ge::Tensor& tensor)
{
    auto data = ge::op::Data(name.c_str()).set_attr_index(index);
    ge::TensorDesc desc = tensor.GetTensorDesc();
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    return data;
}

int CreateGraph(std::vector<ge::Tensor>& inputs, std::vector<ge::Operator>& graphInputs,
                std::vector<ge::Operator>& graphOutputs, ge::Graph& graph)
{
    // SparseTensor of dense shape [3, 3] with the 9 non-zero coordinates filled in row-major order.
    ge::Tensor indicesTensor = BuildTensor<int64_t>({9, 2}, ge::DT_INT64,
                                                    {0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2, 2, 0, 2, 1, 2, 2});
    ge::Tensor valuesTensor = BuildTensor<int64_t>({9}, ge::DT_INT64, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    ge::Tensor shapeTensor = BuildTensor<int64_t>({2}, ge::DT_INT64, {3, 3});
    // Slice starts at [0, 1] with size [2, 2]: kept coordinates are [0, 1] and [1, 0] of the sub-tensor.
    ge::Tensor startTensor = BuildTensor<int64_t>({2}, ge::DT_INT64, {0, 1});
    ge::Tensor sizeTensor = BuildTensor<int64_t>({2}, ge::DT_INT64, {2, 2});

    auto indices = CreateDataOp("indices", 0, indicesTensor);
    auto values = CreateDataOp("values", 1, valuesTensor);
    auto shape = CreateDataOp("shape", 2, shapeTensor);
    auto start = CreateDataOp("start", 3, startTensor);
    auto size = CreateDataOp("size", 4, sizeTensor);
    graph.AddOp(indices);
    graph.AddOp(values);
    graph.AddOp(shape);
    graph.AddOp(start);
    graph.AddOp(size);

    auto op = ge::op::SparseSlice("sparse_slice");
    op.set_input_indices(indices);
    op.set_input_values(values);
    op.set_input_shape(shape);
    op.set_input_start(start);
    op.set_input_size(size);
    op.update_output_desc_y_indices(ge::TensorDesc(ge::Shape({4, 2}), ge::FORMAT_ND, ge::DT_INT64));
    op.update_output_desc_y_values(ge::TensorDesc(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT64));
    op.update_output_desc_y_shape(ge::TensorDesc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT64));

    inputs = {indicesTensor, valuesTensor, shapeTensor, startTensor, sizeTensor};
    graphInputs = {indices, values, shape, start, size};
    graphOutputs = {op};
    return kSuccess;
}

template <typename T>
bool CheckArray(const T* data, const std::vector<T>& expected)
{
    for (size_t i = 0; i < expected.size(); ++i) {
        if (data[i] != expected[i]) {
            std::cerr << "Unexpected output at index " << i << ": " << data[i] << ", expected " << expected[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

int CheckOutput(const std::vector<ge::Tensor>& outputs)
{
    if (outputs.size() != 3) {
        std::cerr << "Unexpected output count: " << outputs.size() << std::endl;
        return kFailed;
    }

    const auto* yIndices = reinterpret_cast<const int64_t*>(outputs[0].GetData());
    const auto* yValues = reinterpret_cast<const int64_t*>(outputs[1].GetData());
    const auto* yShape = reinterpret_cast<const int64_t*>(outputs[2].GetData());
    if (!CheckArray<int64_t>(yIndices, {0, 0, 0, 1, 1, 0, 1, 1})) {
        return kFailed;
    }
    if (!CheckArray<int64_t>(yValues, {2, 3, 5, 6})) {
        return kFailed;
    }
    if (!CheckArray<int64_t>(yShape, {2, 2})) {
        return kFailed;
    }
    return kSuccess;
}
} // namespace

int main()
{
    std::cout << GetTime() << " - INFO - Start SparseSlice GEIR example" << std::endl;
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    ge::Graph graph("sparse_slice_graph");
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Operator> graphInputs;
    std::vector<ge::Operator> graphOutputs;
    if (CreateGraph(inputs, graphInputs, graphOutputs, graph) != kSuccess) {
        ge::GEFinalize();
        return kFailed;
    }
    graph.SetInputs(graphInputs).SetOutputs(graphOutputs);

    std::map<ge::AscendString, ge::AscendString> sessionOptions;
    ge::Session session(sessionOptions);
    if (session.AddGraph(0, graph) != ge::SUCCESS) {
        std::cerr << "AddGraph failed" << std::endl;
        ge::GEFinalize();
        return kFailed;
    }

    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(0, inputs, outputs) != ge::SUCCESS) {
        std::cerr << "RunGraph failed" << std::endl;
        ge::GEFinalize();
        return kFailed;
    }

    const int ret = CheckOutput(outputs);
    ge::GEFinalize();
    if (ret != kSuccess) {
        return kFailed;
    }
    std::cout << GetTime() << " - INFO - SparseSlice GEIR example success" << std::endl;
    return kSuccess;
}
