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
#include <cstring>
#include "kernel_run_context_facker.h"
#include "platform/platform_info.h"
#include "test_cube_util.h"
#include "../../../op_host/fused_multi_conv2d_attrs.h"
#include "../../../op_kernel/arch35/fused_multi_conv2d_geometry.h"

namespace {
struct TilingBlob {
#define DECLARE_FIELD(member, name) uint32_t name;
    FC2D_FLAT_FIELDS(DECLARE_FIELD)
#undef DECLARE_FIELD
    uint32_t blockDim;                 // 实际参与计算的核数
    uint32_t fixedShift1, fixedShift2; // 每层定点定标参数
    uint32_t hasBias1, hasBias2;       // 每层偏置使能
    uint32_t relu1, relu2;             // 每层 ReLU 使能
};
static_assert(sizeof(TilingBlob) == 78 * sizeof(uint32_t));

struct TilingInvocation {
    ge::DataType xType = ge::DT_FLOAT16;               // 输入数据类型
    ge::DataType weightType = ge::DT_FLOAT16;          // 权重数据类型
    ge::DataType yType = ge::DT_FLOAT16;               // 输出数据类型
    std::vector<int64_t> strides{1, 1, 1, 1};          // 两层 H/W 步长
    std::vector<int64_t> pads{0, 0, 0, 0, 0, 0, 0, 0}; // 两层四边补边
    std::vector<int64_t> shifts{0, 0};                 // 两层定点定标参数
    std::vector<int64_t> groups{1, 1};                 // 两层分组数
    int64_t inputChannels = 32;                        // 输入通道数
    int64_t intermediateChannels = 32;                 // 第一层输出通道数
    int64_t outputChannels = 2;                        // 最终输出通道数
    bool cubeVectorFuse = true;                        // Cube/Vector 融合能力
};

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED; // tiling 返回状态
    uint32_t blockDim = 0;                     // 实际核数
    size_t dataSize = 0;                       // 序列化数据字节数
    TilingBlob data{};                         // 序列化的 tiling 数据
};

// 创建存储形状；storage 为物理维度，origin 为逻辑维度，省略时沿用 storage。
gert::StorageShape MakeShape(const std::vector<int64_t>& storage, const std::vector<int64_t>& origin = {})
{
    gert::StorageShape shape;
    for (int64_t dim : storage) {
        shape.MutableStorageShape().AppendDim(dim);
    }
    for (int64_t dim : origin.empty() ? storage : origin) {
        shape.MutableOriginShape().AppendDim(dim);
    }
    return shape;
}

// 构造节点属性；invocation 提供本次测试的类型、几何和定标配置。
std::vector<std::pair<std::string, Ops::NN::AnyValue>> MakeTilingAttributes(const TilingInvocation& invocation)
{
    using Ops::NN::AnyValue;
    return {{"bias_mask", AnyValue::CreateFrom<std::vector<bool>>({true, false})},
            {"dtype", AnyValue::CreateFrom<int64_t>(invocation.yType)},
            {"strides", AnyValue::CreateFrom<std::vector<int64_t>>(invocation.strides)},
            {"pads", AnyValue::CreateFrom<std::vector<int64_t>>(invocation.pads)},
            {"dilations", AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1, 1, 1, 1})},
            {"groups", AnyValue::CreateFrom<std::vector<int64_t>>(invocation.groups)},
            {"data_format", AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
            {"offset_x", AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
            {"round_mode", AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
            {"pad_mode", AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
            {"enable_hf32", AnyValue::CreateFrom<std::vector<bool>>({false, false})},
            {"enable_relu", AnyValue::CreateFrom<std::vector<int64_t>>({1, 0})},
            {"ascendc_op_para_size", AnyValue::CreateFrom<int64_t>(2 * 1024 * 1024)},
            {FusedMultiConv2dAttrs::FIXED_SHIFT_ATTR, AnyValue::CreateFrom<std::vector<int64_t>>(invocation.shifts)}};
}

// 执行一次 host tiling 并读取结果；invocation 为测试输入及平台配置。
TilingResult RunTiling(const TilingInvocation& invocation)
{
    const bool weightIsFp16 = invocation.weightType == ge::DT_FLOAT16;
    const int64_t weightChannelBlock = weightIsFp16 ? 16 : 32;
    auto inputShape = MakeShape({1, invocation.inputChannels, 20, 30});
    auto firstFilterShape = MakeShape({invocation.inputChannels / weightChannelBlock * 3,
                                       invocation.intermediateChannels / 16, 16, weightChannelBlock},
                                      {invocation.intermediateChannels, invocation.inputChannels, 1, 3});
    auto secondFilterShape = MakeShape(
        {invocation.intermediateChannels / weightChannelBlock * 3, 1, 16, weightChannelBlock},
        {2, invocation.intermediateChannels, 3, 1});
    auto firstAuxiliaryShape = MakeShape({invocation.intermediateChannels});
    auto secondAuxiliaryShape = MakeShape({2});
    auto outputShape = MakeShape({1, invocation.outputChannels, 18, 28});
    const auto attributes = MakeTilingAttributes(invocation);
    const char* compileInfoText = R"({"hardware_info": {
        "BT_SIZE": 4096, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 1048576,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "FB_SIZE": 4096, "L0C_SIZE": 262144,
        "CORE_NUM": 8, "cube_core_cnt": 8, "vector_core_cnt": 8,
        "core_type_list": "CubeCore,VectorCore"}})";
    std::map<std::string, std::string> socInfo, coreSpec, intrinsics;
    GetPlatFormInfos(compileInfoText, socInfo, coreSpec, intrinsics);
    socInfo["cube_vector_combine"] = invocation.cubeVectorFuse ? "fuse" : "split";
    fe::PlatFormInfos platform;
    platform.Init();
    int compileInfo = 0;
    auto tilingBuffer = gert::TilingData::CreateCap(4096);
    auto workspace = gert::ContinuousVector::Create<size_t>(1);
    gert::TilingContextFaker faker;
    faker.SetOpType("FusedMultiConv2d")
        .NodeIoNum(5, 1)
        .IrInstanceNum({1, 2, 2, 2, 2})
        .InputShapes({&inputShape, &firstFilterShape, &secondFilterShape, &firstAuxiliaryShape, &secondAuxiliaryShape,
                      &firstAuxiliaryShape, &secondAuxiliaryShape, &firstAuxiliaryShape, &secondAuxiliaryShape})
        .OutputShapes({&outputShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platform))
        .NodeAttrs(attributes)
        .NodeInputTd(0, invocation.xType, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeInputTd(1, invocation.weightType, ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_Z)
        .NodeInputTd(2, weightIsFp16 ? ge::DT_FLOAT16 : ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, invocation.yType, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .TilingData(tilingBuffer.get())
        .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()));
    auto holder = faker.Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfo);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", coreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    const auto* opImplementation = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedMultiConv2d");
    EXPECT_NE(opImplementation, nullptr);
    TilingResult result;
    if (opImplementation == nullptr || opImplementation->tiling == nullptr) {
        return result;
    }
    result.status = opImplementation->tiling(context);
    if (result.status == ge::GRAPH_SUCCESS) {
        result.blockDim = context->GetBlockDim();
        result.dataSize = context->GetRawTilingData()->GetDataSize();
        EXPECT_EQ(result.dataSize, sizeof(TilingBlob));
        if (result.dataSize == sizeof(TilingBlob)) {
            std::memcpy(&result.data, context->GetRawTilingData()->GetData(), sizeof(TilingBlob));
        }
        EXPECT_EQ(context->GetTilingKey(), 0U);
        EXPECT_EQ(context->GetWorkspaceSizes(1)[0], 16U * 1024 * 1024);
    }
    return result;
}
} // namespace

// 验证四种数据通路的 tiling 字段与固定 ABI。
TEST(FusedMultiConv2dTiling, FourDataTypePaths)
{
    for (const auto& types : std::vector<std::vector<ge::DataType>>{{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
                                                                    {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8},
                                                                    {ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16},
                                                                    {ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT16}}) {
        TilingInvocation invocation;
        invocation.xType = types[0];
        invocation.weightType = types[1];
        invocation.yType = types[2];
        const auto result = RunTiling(invocation);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        EXPECT_GE(result.blockDim, 1U);
        EXPECT_LE(result.blockDim, 8U);
        EXPECT_EQ(result.data.blockDim, result.blockDim);
        EXPECT_EQ(result.data.elemBytes, types[0] == ge::DT_FLOAT16 ? 2U : 1U);
        EXPECT_EQ(result.data.weightElemBytes, types[1] == ge::DT_FLOAT16 ? 2U : 1U);
        EXPECT_EQ(result.data.midElemBytes, types[1] == ge::DT_FLOAT16 ? 2U : 1U);
        EXPECT_EQ(result.data.kh1, 1U);
        EXPECT_EQ(result.data.kw1, 3U);
        EXPECT_EQ(result.data.kh2, 3U);
        EXPECT_EQ(result.data.kw2, 1U);
        EXPECT_EQ(result.data.cout2, 2U);
        EXPECT_EQ(result.data.ho2, 18U);
        EXPECT_EQ(result.data.wo2, 28U);
        EXPECT_EQ(result.data.fixedShift1, 0U);
        EXPECT_EQ(result.data.fixedShift2, 0U);
        EXPECT_EQ(result.data.hasBias1, 1U);
        EXPECT_EQ(result.data.hasBias2, 0U);
        EXPECT_EQ(result.data.relu1, 1U);
        EXPECT_EQ(result.data.relu2, 0U);
    }
}

// 验证每层私有定标参数及其取值范围。
TEST(FusedMultiConv2dTiling, PrivateFixedShift)
{
    TilingInvocation invocation;
    invocation.shifts = {42, 37};
    auto result = RunTiling(invocation);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.data.fixedShift1, 42U);
    EXPECT_EQ(result.data.fixedShift2, 37U);
    for (const auto& shifts : std::vector<std::vector<int64_t>>{{0}, {-1, 0}, {0, 59}}) {
        invocation.shifts = shifts;
        EXPECT_EQ(RunTiling(invocation).status, ge::GRAPH_FAILED);
    }
}

// 验证不等步长、非对称补边与分组配置被拒绝。
TEST(FusedMultiConv2dTiling, InvalidGeometryAndGroups)
{
    TilingInvocation invocation;
    invocation.strides = {1, 2, 1, 1};
    EXPECT_EQ(RunTiling(invocation).status, ge::GRAPH_FAILED);
    invocation.strides = {1, 1, 1, 1};
    invocation.pads = {1, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(RunTiling(invocation).status, ge::GRAPH_FAILED);
    invocation.pads.assign(8, 0);
    invocation.groups = {1, 2};
    EXPECT_EQ(RunTiling(invocation).status, ge::GRAPH_FAILED);
}

// 验证输出不匹配和不支持的平台能力被拒绝。
TEST(FusedMultiConv2dTiling, InvalidOutputAndPlatform)
{
    TilingInvocation invocation;
    invocation.outputChannels = 16;
    EXPECT_EQ(RunTiling(invocation).status, ge::GRAPH_FAILED);
    invocation.outputChannels = 2;
    EXPECT_EQ(RunTiling(invocation).status, ge::GRAPH_SUCCESS);
    invocation.cubeVectorFuse = false;
    EXPECT_EQ(RunTiling(invocation).status, ge::GRAPH_FAILED);
}

// 验证 K 分块兼顾 L0B ping/pong，并保留可整套常驻的权重。
TEST(FusedMultiConv2dTiling, KTileAndL0bReuse)
{
    TilingInvocation invocation;
    invocation.xType = ge::DT_INT8;
    invocation.weightType = ge::DT_INT8;
    invocation.yType = ge::DT_INT8;
    invocation.inputChannels = 512;
    invocation.intermediateChannels = 512;
    auto result = RunTiling(invocation);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.data.tileK1, 64U);
    EXPECT_EQ(result.data.mMax1, 128U);
    EXPECT_EQ(result.data.l0bPP1, 1U);

    invocation.xType = ge::DT_FLOAT16;
    invocation.yType = ge::DT_FLOAT16;
    invocation.inputChannels = 64;
    invocation.intermediateChannels = 256;
    result = RunTiling(invocation);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.data.tileK1, 96U);
    EXPECT_EQ(result.data.l0bChunks1, 1U);
    EXPECT_EQ(result.data.l0bPP1, 0U);
}
