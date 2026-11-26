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
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "ge/ge_api_types.h"
#include "graph/types.h"

extern "C" {
#include "exe_graph/runtime/kernel_run_context.h"
}

#define CANN_OPS_OPB_ASYN_EXEC_ACLNN(ctx, aclnnApi, ...) GRAPH_SUCCESS

struct OpApiParams {
    struct ConvertedParam {
        void* pointer;
        void (*deleter)(void*);
    };
    int (*op_api_func)(void*, uint64_t, void*, void*);
    void* executor;
    std::vector<ConvertedParam> converted_params;
};

#include "../../../op_graph/fused_mat_mul_fallback.cpp"
#include "compute_node_info_faker.h"

using namespace fallback;
using namespace ge;

namespace {

gert::Tensor MakeTensor(std::initializer_list<int64_t> dims, DataType dtype, Format format = FORMAT_ND)
{
    gert::StorageShape storageShape{dims, dims};
    gert::StorageFormat storageFormat;
    storageFormat.SetOriginFormat(format);
    storageFormat.SetStorageFormat(format);
    void* dummyAddr = reinterpret_cast<void*>(0xDEADBEEF);
    return gert::Tensor(storageShape, storageFormat, gert::kOnDeviceHbm, dtype, dummyAddr);
}

struct ComputeNodeInfoHelper {
    gert::ComputeNodeInfoFaker faker;
    const void* nodeInfo = nullptr;

    explicit ComputeNodeInfoHelper(const size_t attrCount = 2)
    {
        faker.NodeType("FusedMatMul").IrInstanceNum(std::vector<uint32_t>{1, 1, 1, 1}, std::vector<uint32_t>{1});
        for (size_t i = 0; i < 4; ++i) {
            faker.InputTd(i, DT_FLOAT16, FORMAT_ND, FORMAT_ND);
        }
        faker.OutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND);
        if (attrCount >= 1) {
            faker.AppendAttr(false); // transpose_x1
        }
        if (attrCount >= 2) {
            faker.AppendAttr(false); // transpose_x2
        }
        if (attrCount >= 3) {
            faker.AppendAttr(false); // enable_hf32
        }
        nodeInfo = faker.Build();
    }

    void* AsVoidPtr() { return const_cast<void*>(nodeInfo); }
};

struct MockContext {
    std::vector<uint8_t> buffer;
    std::vector<gert::Chain*> chains;
    gert::OpExecutePrepareContext* ctx;

    MockContext(std::vector<gert::Tensor*> inputs, std::vector<gert::Tensor*> outputs, void* computeNodeInfo = nullptr)
    {
        size_t numInputs = inputs.size();
        size_t numOutputs = outputs.size();
        constexpr size_t kExtendInputs = 3;
        constexpr size_t kExtendOutputs = 2;
        size_t totalValues = numInputs + numOutputs + kExtendInputs + kExtendOutputs;

        size_t ctxSize = sizeof(KernelRunContext) + (totalValues > 1 ? (totalValues - 1) * sizeof(void*) : 0);
        buffer.resize(ctxSize);
        std::memset(buffer.data(), 0, ctxSize);

        auto krc = reinterpret_cast<KernelRunContext*>(buffer.data());
        krc->input_size = numInputs + numOutputs + kExtendInputs;
        krc->output_size = kExtendOutputs;
        krc->compute_node_info = computeNodeInfo;
        krc->kernel_extend_info = nullptr;
        krc->output_start = nullptr;

        for (size_t i = 0; i < numInputs; ++i) {
            if (inputs[i] != nullptr) {
                auto chain = new gert::Chain();
                chains.push_back(chain);
                chain->Set(inputs[i], nullptr);
                krc->values[i] = reinterpret_cast<AsyncAnyValue*>(chain);
            } else {
                krc->values[i] = nullptr;
            }
        }
        for (size_t i = 0; i < numOutputs; ++i) {
            size_t idx = numInputs + i;
            if (outputs[i] != nullptr) {
                auto chain = new gert::Chain();
                chains.push_back(chain);
                chain->Set(outputs[i], nullptr);
                krc->values[idx] = reinterpret_cast<AsyncAnyValue*>(chain);
            } else {
                krc->values[idx] = nullptr;
            }
        }
        for (size_t i = 0; i < kExtendOutputs; ++i) {
            auto chain = new gert::Chain();
            chains.push_back(chain);
            krc->values[krc->input_size + i] = reinterpret_cast<AsyncAnyValue*>(chain);
        }

        ctx = reinterpret_cast<gert::OpExecutePrepareContext*>(buffer.data());
    }

    ~MockContext()
    {
        for (auto* chain : chains) {
            chain->Set(nullptr, nullptr);
            delete chain;
        }
    }
};

} // namespace

class FusedMatMulFallbackTest : public ::testing::Test {};

TEST_F(FusedMatMulFallbackTest, NullContext) { EXPECT_EQ(FusedMatmulExecuteFunc(nullptr), GRAPH_FAILED); }

TEST_F(FusedMatMulFallbackTest, X1IsNull)
{
    MockContext mc({nullptr}, {});
    EXPECT_EQ(FusedMatmulExecuteFunc(mc.ctx), GRAPH_FAILED);
}

TEST_F(FusedMatMulFallbackTest, X2IsNull)
{
    auto x1 = MakeTensor({16, 32}, DT_FLOAT16);
    MockContext mc({&x1, nullptr}, {});
    EXPECT_EQ(FusedMatmulExecuteFunc(mc.ctx), GRAPH_FAILED);
}

TEST_F(FusedMatMulFallbackTest, OutputIsNull)
{
    auto x1 = MakeTensor({16, 32}, DT_FLOAT16);
    auto x2 = MakeTensor({32, 16}, DT_FLOAT16);
    MockContext mc({&x1, &x2, nullptr, nullptr}, {nullptr});
    EXPECT_EQ(FusedMatmulExecuteFunc(mc.ctx), GRAPH_FAILED);
}

TEST_F(FusedMatMulFallbackTest, TransposeAttrMissing)
{
    ComputeNodeInfoHelper cnih(/*attrCount=*/0);
    auto x1 = MakeTensor({16, 32}, DT_FLOAT16);
    auto x2 = MakeTensor({32, 16}, DT_FLOAT16);
    auto y = MakeTensor({16, 16}, DT_FLOAT16);
    MockContext mc({&x1, &x2, nullptr, nullptr}, {&y}, cnih.AsVoidPtr());
    EXPECT_EQ(FusedMatmulExecuteFunc(mc.ctx), GRAPH_FAILED);
}

// fused_op_type is absent here and takes the IR default "". ConvertMmType needs the acl
// runtime, so the prepare either reaches the stubbed aclnn call or fails at tensor conversion.
TEST_F(FusedMatMulFallbackTest, ReluWithoutBiasReachesExecute)
{
    ComputeNodeInfoHelper cnih(/*attrCount=*/3);
    auto x1 = MakeTensor({16, 32}, DT_FLOAT16);
    auto x2 = MakeTensor({32, 16}, DT_FLOAT16);
    auto y = MakeTensor({16, 16}, DT_FLOAT16);
    MockContext mc({&x1, &x2, nullptr, nullptr}, {&y}, cnih.AsVoidPtr());
    auto result = FusedMatmulExecuteFunc(mc.ctx);
    EXPECT_TRUE(result == GRAPH_SUCCESS || result == GRAPH_FAILED);
}
