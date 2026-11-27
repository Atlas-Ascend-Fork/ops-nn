/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include "compute_node_info_faker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

namespace {

struct TransducerCompileStub {
    int64_t coreNum = 32;
    uint64_t ubSize = 196608;
};

void InitTransducerPlatform(fe::PlatFormInfos& platformInfo)
{
    const std::string hwJson = R"({
        "hardware_info": {
            "BT_SIZE": 0, "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": false,
            "Intrinsic_data_move_l12ub": true,
            "Intrinsic_data_move_l0c2ub": true,
            "Intrinsic_data_move_out2l1_nd2nz": false,
            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
            "CORE_NUM": 32
        }
    })";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    ut_util::GetPlatFormInfos(hwJson.c_str(), socInfos, aicoreSpec, intrinsics);
    platformInfo.Init();
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
}

class OptimizedTransducerTilingUt : public testing::Test {};

TEST_F(OptimizedTransducerTilingUt, NormalFp32AndFp16TilingSuccess)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("OptimizedTransducer");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    ASSERT_NE(impl->tiling_parse, nullptr);

    fe::PlatFormInfos platformInfo;
    InitTransducerPlatform(platformInfo);
    TransducerCompileStub compileInfo{};

    auto parseHolder = gert::KernelRunContextFaker()
                           .KernelIONum(2, 1)
                           .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platformInfo)})
                           .Outputs({&compileInfo})
                           .Build();
    EXPECT_EQ(impl->tiling_parse(parseHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape logitsShape = {{4, 8, 6, 16}, {4, 8, 6, 16}};
    gert::StorageShape targetsShape = {{4, 5}, {4, 5}};
    gert::StorageShape lenShape = {{4}, {4}};
    gert::StorageShape lossShape = {{4}, {4}};

    for (ge::DataType dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        auto tilingBuf = gert::TilingData::CreateCap(4096);
        auto wsHolder = gert::ContinuousVector::Create<size_t>(16);
        auto* wsVec = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

        auto ctxHolder = gert::TilingContextFaker()
                             .SetOpType("OptimizedTransducer")
                             .NodeIoNum(4, 2)
                             .IrInstanceNum({1, 1, 1, 1})
                             .InputShapes({&logitsShape, &targetsShape, &lenShape, &lenShape})
                             .OutputShapes({&lossShape, &logitsShape})
                             .CompileInfo(&compileInfo)
                             .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                             .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                         {"clamp", Ops::NN::AnyValue::CreateFrom<float>(0.0F)},
                                         {"fused_log_softmax", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                         {"requires_grad", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                             .TilingData(tilingBuf.get())
                             .Workspace(wsVec)
                             .Build();
        EXPECT_EQ(impl->tiling(ctxHolder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
    }
}

TEST_F(OptimizedTransducerTilingUt, RejectsNullAttrsMissingAttrNullShapeAndNullDesc)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("OptimizedTransducer");
    ASSERT_NE(impl, nullptr);

    fe::PlatFormInfos platformInfo;
    InitTransducerPlatform(platformInfo);
    TransducerCompileStub compileInfo{};

    gert::StorageShape logitsShape = {{2, 4, 4, 8}, {2, 4, 4, 8}};
    gert::StorageShape targetsShape = {{2, 3}, {2, 3}};
    gert::StorageShape lenShape = {{2}, {2}};
    gert::StorageShape lossShape = {{2}, {2}};

    auto makeHolder = [&](const std::vector<std::pair<std::string, Ops::NN::AnyValue>>& attrs, bool nullLogitsShape) {
        auto tilingBuf = gert::TilingData::CreateCap(4096);
        auto wsHolder = gert::ContinuousVector::Create<size_t>(16);
        auto* wsVec = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
        std::vector<gert::StorageShape*> inShapes = {nullLogitsShape ? nullptr : &logitsShape, &targetsShape, &lenShape,
                                                     &lenShape};
        auto holder = gert::TilingContextFaker()
                          .SetOpType("OptimizedTransducer")
                          .NodeIoNum(4, 2)
                          .IrInstanceNum({1, 1, 1, 1})
                          .InputShapes(inShapes)
                          .OutputShapes({&lossShape, &logitsShape})
                          .CompileInfo(&compileInfo)
                          .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                          .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs(attrs)
                          .TilingData(tilingBuf.get())
                          .Workspace(wsVec)
                          .Build();
        return std::make_tuple(std::move(holder), std::move(tilingBuf), std::move(wsHolder));
    };

    // 1. 属性列表为空与逐项缺失，覆盖四个属性指针判空分支
    {
        auto [h0, t0, w0] = makeHolder({}, false);
        EXPECT_EQ(impl->tiling(h0.GetContext<gert::TilingContext>()), ge::GRAPH_FAILED);
    }
    {
        auto [h1, t1, w1] = makeHolder({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}}, false);
        EXPECT_EQ(impl->tiling(h1.GetContext<gert::TilingContext>()), ge::GRAPH_FAILED);
    }
    {
        auto [h2, t2, w2] = makeHolder({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                        {"clamp", Ops::NN::AnyValue::CreateFrom<float>(0.0F)}},
                                       false);
        EXPECT_EQ(impl->tiling(h2.GetContext<gert::TilingContext>()), ge::GRAPH_FAILED);
    }
    {
        auto [h3, t3, w3] = makeHolder({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                        {"clamp", Ops::NN::AnyValue::CreateFrom<float>(0.0F)},
                                        {"fused_log_softmax", Ops::NN::AnyValue::CreateFrom<bool>(true)}},
                                       false);
        EXPECT_EQ(impl->tiling(h3.GetContext<gert::TilingContext>()), ge::GRAPH_FAILED);
    }

    const std::vector<std::pair<std::string, Ops::NN::AnyValue>> fullAttrs = {
        {"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
        {"clamp", Ops::NN::AnyValue::CreateFrom<float>(0.0F)},
        {"fused_log_softmax", Ops::NN::AnyValue::CreateFrom<bool>(true)},
        {"requires_grad", Ops::NN::AnyValue::CreateFrom<bool>(true)}};

    // 2. logitsShape 为空指针
    {
        auto [hShape, tS, wS] = makeHolder(fullAttrs, true);
        EXPECT_EQ(impl->tiling(hShape.GetContext<gert::TilingContext>()), ge::GRAPH_FAILED);
    }

    // 3. logitsDesc 为空指针（保持 IO 总槽位数 6 不变以确保 PlatformInfo 槽位对齐，同时令 inputs_num 为 0）
    {
        auto [hDesc, tD, wD] = makeHolder(fullAttrs, false);
        auto* ctx = hDesc.GetContext<gert::TilingContext>();
        gert::OpKernelContextBuilder nodeBuilder;
        auto nodeHolder = nodeBuilder.OpType("OptimizedTransducer")
                              .OpName("OptimizedTransducer")
                              .IOInstanceNum({}, {1, 1, 1, 1, 1, 1})
                              .AppendAttr(static_cast<int64_t>(0))
                              .AppendAttr(static_cast<float>(0.0F))
                              .AppendAttr(true)
                              .AppendAttr(true)
                              .Build();
        ctx->GetContext()->compute_node_info = nodeHolder.GetContext()->GetContext()->compute_node_info;
        EXPECT_EQ(impl->tiling(ctx), ge::GRAPH_FAILED);

        // 4. ComputeNodeInfo 置空使 GetAttrs() 返回 nullptr
        ctx->GetContext()->compute_node_info = nullptr;
        EXPECT_EQ(impl->tiling(ctx), ge::GRAPH_FAILED);
    }
}

} // namespace
