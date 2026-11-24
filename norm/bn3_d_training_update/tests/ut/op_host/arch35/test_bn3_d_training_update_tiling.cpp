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
 * \file test_bn3_d_training_update_tiling.cpp
 * \brief BN3DTrainingUpdate arch35 tiling UT。
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_host/arch35/bn3_d_training_update_tiling.h"
#include "../../../../op_kernel/arch35/bn3_d_training_update_tiling_data.h"
#include "../../../../op_kernel/arch35/bn3_d_training_update_tiling_key.h"

namespace {

using Bn3dTrainingUpdateOps::BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES;
using Bn3dTrainingUpdateOps::BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES;
using Bn3dTrainingUpdateOps::BN3D_TRAINING_UPDATE_FLAT_TILING_DATA_BYTES;
using Bn3dTrainingUpdateOps::Bn3dTrainingUpdateChannelFirstTilingData;
using Bn3dTrainingUpdateOps::Bn3dTrainingUpdateChannelLastTilingData;
using Bn3dTrainingUpdateOps::Bn3dTrainingUpdateFlatTilingData;

constexpr uint64_t kCoreNum = 64U;
constexpr uint64_t kUbSize = 245760U;
constexpr uint64_t kTilingDataCap = 4096U;
constexpr int64_t kVlFp32 = 256 / 4; // Ops::Base::GetVRegSize() / sizeof(float)
constexpr int64_t kBlockBytes = 32;

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

struct TilingRun {
    ge::graphStatus status = ge::GRAPH_FAILED;
    int64_t key = -1;
    size_t dataSize = 0;
    std::vector<uint8_t> data;
    size_t blockNum = 0;
};

// x 的 7 输入 / 5 输出 + 2 attr 上下文;statLen 为六路 (C,) 统计量的长度。
TilingRun RunTiling(const std::vector<int64_t>& xDims, ge::Format xFormat, ge::DataType xDtype, int64_t statLen,
                    float factor = 0.1f, float epsilon = 1.0e-5f, ge::Format statFormat = ge::FORMAT_ND,
                    std::vector<int64_t> statDims = {}, std::vector<int64_t> outStatDims = {})
{
    const gert::StorageShape xShape = MakeShape(xDims);
    const std::vector<int64_t> statDimsEff = statDims.empty() ? std::vector<int64_t>{statLen} : statDims;
    const gert::StorageShape statShape = MakeShape(statDimsEff);
    const gert::StorageShape outStatShape = MakeShape(outStatDims.empty() ? statDimsEff : outStatDims);
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.emplace_back(xShape, xDtype, xFormat, false, nullptr, xFormat);
    for (int64_t i = 0; i < 6; ++i) {
        inputs.emplace_back(statShape, ge::DT_FLOAT, statFormat, false, nullptr, statFormat);
    }
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.emplace_back(xShape, xDtype, xFormat, false, nullptr, xFormat);
    for (int64_t i = 0; i < 4; ++i) {
        outputs.emplace_back(outStatShape, ge::DT_FLOAT, statFormat, false, nullptr, statFormat);
    }
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    attrs.emplace_back("factor", Ops::NN::AnyValue::CreateFrom<float>(factor));
    attrs.emplace_back("epsilon", Ops::NN::AnyValue::CreateFrom<float>(epsilon));
    optiling::Bn3dTrainingUpdateCompileInfo compileInfo{static_cast<int64_t>(kCoreNum), static_cast<int64_t>(kUbSize),
                                                        256, kBlockBytes};
    gert::TilingContextPara para("BN3DTrainingUpdate", inputs, outputs, attrs, &compileInfo, kCoreNum, kUbSize,
                                 kTilingDataCap);
    TilingInfo info;
    TilingRun run;
    const bool executed = ExecuteTiling(para, info);
    run.status = executed ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
    run.key = info.tilingKey;
    run.dataSize = info.tilingDataSize;
    run.blockNum = info.blockNum;
    if (info.tilingData != nullptr && info.tilingDataSize > 0) {
        run.data.assign(info.tilingData.get(), info.tilingData.get() + info.tilingDataSize);
    }
    return run;
}

const Bn3dTrainingUpdateChannelFirstTilingData* AsCf(const TilingRun& run)
{
    return reinterpret_cast<const Bn3dTrainingUpdateChannelFirstTilingData*>(run.data.data());
}

const Bn3dTrainingUpdateChannelLastTilingData* AsCl(const TilingRun& run)
{
    return reinterpret_cast<const Bn3dTrainingUpdateChannelLastTilingData*>(run.data.data());
}

const Bn3dTrainingUpdateFlatTilingData* AsFlat(const TilingRun& run)
{
    return reinterpret_cast<const Bn3dTrainingUpdateFlatTilingData*>(run.data.data());
}

int64_t ElemBytesOf(ge::DataType dtype) { return (dtype == ge::DT_FLOAT) ? 4 : 2; }

optiling::Bn3dTrainingUpdatePlatform Platform()
{
    return optiling::Bn3dTrainingUpdatePlatform{static_cast<int64_t>(kCoreNum), static_cast<int64_t>(kUbSize), kVlFp32,
                                                kBlockBytes};
}

} // namespace

class BN3DTrainingUpdateTilingTest : public testing::Test {};

// ============================ TilingKey 映射 + ABI ============================

TEST_F(BN3DTrainingUpdateTilingTest, channel_first_nchw_fp16)
{
    const TilingRun run = RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 3);
    EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_FIRST));
    EXPECT_EQ(run.dataSize, BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES);
    ASSERT_EQ(run.data.size(), BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES);
    const auto* td = AsCf(run);
    EXPECT_EQ(td->c, 3);
    EXPECT_EQ(td->spatial, 20); // 4 * 5
    // 单元总数恒等于 cWindows·N·spatialTiles(cWindows = ceil(C/K),N = 2 来自本用例的 shape)。
    ASSERT_GE(td->channelWindow, 1);
    const int64_t cWindows = (td->c + td->channelWindow - 1) / td->channelWindow;
    EXPECT_EQ(td->baseTilesPerCore * static_cast<int64_t>(run.blockNum) + td->extraCoreCount,
              cWindows * 2 * td->spatialTiles);
    EXPECT_GE(td->spatialTile, 1);
    EXPECT_GE(run.blockNum, 1U);
}

TEST_F(BN3DTrainingUpdateTilingTest, channel_first_ncdhw_bf16)
{
    const TilingRun run = RunTiling({2, 3, 4, 5, 6}, ge::FORMAT_NCDHW, ge::DT_BF16, 3);
    EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_FIRST));
    EXPECT_EQ(run.dataSize, BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES);
    const auto* td = AsCf(run);
    EXPECT_EQ(td->spatial, 120); // 4 * 5 * 6
}

TEST_F(BN3DTrainingUpdateTilingTest, channel_last_nhwc_fp32)
{
    const TilingRun run = RunTiling({2, 4, 5, 3}, ge::FORMAT_NHWC, ge::DT_FLOAT, 3);
    EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_LAST));
    EXPECT_EQ(run.dataSize, BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES);
    ASSERT_EQ(run.data.size(), BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES);
    const auto* td = AsCl(run);
    EXPECT_EQ(td->c, 3);
    EXPECT_EQ(td->num, 40); // 2 * 4 * 5
    const int64_t channelBlocks = (td->c + td->channelBlock - 1) / td->channelBlock;
    EXPECT_EQ(td->baseTilesPerCore * static_cast<int64_t>(run.blockNum) + td->extraCoreCount,
              channelBlocks * td->tilesPerBlock);
    EXPECT_GE(td->coeffCap, td->channelBlock);
    EXPECT_GE(td->rowsPerTile, 1);
}

TEST_F(BN3DTrainingUpdateTilingTest, channel_last_ndhwc_fp16)
{
    const TilingRun run = RunTiling({2, 4, 5, 6, 3}, ge::FORMAT_NDHWC, ge::DT_FLOAT16, 3);
    EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_LAST));
    EXPECT_EQ(run.dataSize, BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES);
    const auto* td = AsCl(run);
    EXPECT_EQ(td->num, 240); // 2 * 4 * 5 * 6
}

// BN3D-1 定向回归:channel-last 且 C == 6144(邻域 6143/6145/8192/12288)必须能出合法 tiling。
TEST_F(BN3DTrainingUpdateTilingTest, channel_last_regression_bn3d_1_neighbourhood)
{
    const std::vector<int64_t> channels = {6143, 6144, 6145, 8192, 12288};
    for (const int64_t c : channels) {
        const TilingRun run = RunTiling({1, 1, 1, 1, c}, ge::FORMAT_NDHWC, ge::DT_FLOAT16, c);
        ASSERT_EQ(run.status, ge::GRAPH_SUCCESS) << "C=" << c;
        ASSERT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_LAST)) << "C=" << c;
        const auto* td = AsCl(run);
        EXPECT_EQ(td->c, c);
        EXPECT_GE(td->channelBlock, 1);
        EXPECT_GE(td->rowsPerTile, 1);
    }
}

// ============================ UB 反解不变量 ============================

TEST_F(BN3DTrainingUpdateTilingTest, channel_first_ub_ledger_within_ub)
{
    const TilingRun run = RunTiling({2, 3, 64, 64}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3);
    ASSERT_EQ(run.status, ge::GRAPH_SUCCESS);
    const auto* td = AsCf(run);
    const auto plat = Platform();
    EXPECT_LE(optiling::CfKernelBytes(td->spatialTile, td->channelWindow, 4, plat), plat.ubSize);
    EXPECT_LE(optiling::StatPhaseBytes(plat) + optiling::kPipeMetaReserveBytes, plat.ubSize);
    // 切分和恒等于单元总数(每核段 + 尾部分摊);shape {2,3,64,64} ⇒ 总数 = cWindows·N·spatialTiles。
    const int64_t cw = (td->c + td->channelWindow - 1) / td->channelWindow;
    EXPECT_EQ(td->baseTilesPerCore * static_cast<int64_t>(run.blockNum) + td->extraCoreCount,
              cw * 2 * td->spatialTiles);
    EXPECT_LE(td->channelWindow, td->c); // K <= C
}

TEST_F(BN3DTrainingUpdateTilingTest, channel_last_ub_ledger_within_ub)
{
    const TilingRun run = RunTiling({2, 64, 64, 100}, ge::FORMAT_NHWC, ge::DT_FLOAT, 100);
    ASSERT_EQ(run.status, ge::GRAPH_SUCCESS);
    const auto* td = AsCl(run);
    const auto plat = Platform();
    EXPECT_LE(optiling::ClKernelBytes(td->rowsPerTile, td->channelBlock, td->coeffCap, 4, plat), plat.ubSize);
    EXPECT_LE(optiling::StatPhaseBytes(plat) + optiling::kPipeMetaReserveBytes, plat.ubSize);
    EXPECT_EQ(td->statUnitCount, (td->c + td->statChunk - 1) / td->statChunk);
    EXPECT_EQ(td->baseStatPerCore * static_cast<int64_t>(run.blockNum) + td->extraStatCount, td->statUnitCount);
    // VF 循环计数器(uint16_t)上界不变量:一个瓦片的元素数必须装得下。
    EXPECT_LE(td->rowsPerTile * td->channelBlock, optiling::kMaxTileElementsPerVf);
}

// 块长 = "UB 上界" 与 "单元数目标" 取小 —— 小形状不得再让单元数塌到 1。
TEST_F(BN3DTrainingUpdateTilingTest, f1_units_target_reached_on_small_shapes)
{
    const int64_t unitsTarget = static_cast<int64_t>(kCoreNum) * optiling::kUnitsPerCoreTarget;
    // tiny_total 形态:NDHWC (1,1,64,64,5) —— 旧实现 blockDim 塌到 1;修正后单元数应 >= unitsTarget。
    const TilingRun cl = RunTiling({1, 1, 64, 64, 5}, ge::FORMAT_NDHWC, ge::DT_BF16, 5);
    ASSERT_EQ(cl.status, ge::GRAPH_SUCCESS);
    const auto* tdc = AsCl(cl);
    const int64_t channelBlocks = (tdc->c + tdc->channelBlock - 1) / tdc->channelBlock;
    EXPECT_GE(channelBlocks * tdc->tilesPerBlock, unitsTarget) << "channel-last 小形状单元数未达标";
    EXPECT_EQ(tdc->baseTilesPerCore * static_cast<int64_t>(cl.blockNum) + tdc->extraCoreCount,
              channelBlocks * tdc->tilesPerBlock);
    // 几何精确钉住(单元数目标生效后的实测值):C=5 ⇒ Cb=5、channelBlocks=1;rows 由单元数目标压到 16
    EXPECT_EQ(tdc->channelBlock, 5);
    EXPECT_EQ(tdc->rowsPerTile, 16);
    EXPECT_EQ(tdc->tilesPerBlock, 256);
    // channel-first 小 S 侧同式:NCDHW (8,1024,1,2,160) 的 S=320 缩到 S=2 的极端
    const TilingRun cf = RunTiling({2, 1024, 1, 2}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 1024);
    ASSERT_EQ(cf.status, ge::GRAPH_SUCCESS);
    const auto* tdf = AsCf(cf);
    const int64_t cw = (tdf->c + tdf->channelWindow - 1) / tdf->channelWindow;
    EXPECT_GE(cw * 2 * tdf->spatialTiles, unitsTarget) << "channel-first 小 S 单元数未达标";
    EXPECT_LE(optiling::CfKernelBytes(tdf->spatialTile, tdf->channelWindow, 2, Platform()), Platform().ubSize);
    // 几何精确钉住:C=1024、N=2、S=2 ⇒ K_ub 被 kParamChunkLen 钳到 512,但 Kcap_units =
    // floor(C·N·spatialTiles / unitsTarget) = floor(2048/256) = 8 更小 ⇒ K = 8(取小,非阈值路由);
    // cWindows = 128 ⇒ 单元数 = 128·2·1 = 256 = unitsTarget ✓
    EXPECT_EQ(tdf->channelWindow, 8);
    EXPECT_EQ(tdf->spatialTile, 2);
    EXPECT_EQ(cw * 2 * tdf->spatialTiles, unitsTarget);
}

// CF 单元放大为 K 个连续通道平面 —— 大 C + 小 S 的形状必须有 K > 1。
TEST_F(BN3DTrainingUpdateTilingTest, f2_channel_window_enlarged_for_small_spatial)
{
    // xpf_00070 形态:NCDHW (4, 32768, 1, 4, 4) ⇒ S = 16
    const TilingRun run = RunTiling({4, 32768, 1, 4, 4}, ge::FORMAT_NCDHW, ge::DT_BF16, 32768);
    ASSERT_EQ(run.status, ge::GRAPH_SUCCESS);
    const auto* td = AsCf(run);
    EXPECT_GT(td->channelWindow, 1); // 单元不再被钉在"1 个通道平面"
    EXPECT_LE(td->channelWindow, 32768);
    const int64_t cw = (td->c + td->channelWindow - 1) / td->channelWindow;
    EXPECT_EQ(td->baseTilesPerCore * static_cast<int64_t>(run.blockNum) + td->extraCoreCount,
              cw * 4 * td->spatialTiles);
    EXPECT_LE(optiling::CfKernelBytes(td->spatialTile, td->channelWindow, 2, Platform()), Platform().ubSize);
    // 几何精确钉住(xpf_00070 形态):K 达折叠分块上界 512、Ss=16(=S)、单元数 = 64·4·1 = 256
    EXPECT_EQ(td->channelWindow, 512);
    EXPECT_EQ(td->spatialTile, 16);
    EXPECT_EQ(cw * 4 * td->spatialTiles, 256);
}

// ============================ 边界:C == 0 支持(空张量)/ num == 0 支持(空 batch)============================

// C == 0(通道数为 0)⇒ **不拒收**:x 与所有 (C,) 统计量均为空张量,产出"全空"tiling(无 X / 相 S 单元),
// kernel 空跑即正确。对齐旧版语义与 A2/A3 支持面,DFX 有 empt_c0_* 覆盖。
TEST_F(BN3DTrainingUpdateTilingTest, zero_channel_supported)
{
    // channel-first:NCHW (2,0,4,5),统计量 (0,) ⇒ C = 0。
    const TilingRun cf = RunTiling({2, 0, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0);
    EXPECT_EQ(cf.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(cf.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_FIRST));
    ASSERT_EQ(cf.data.size(), BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES);
    EXPECT_EQ(AsCf(cf)->c, 0);
    EXPECT_EQ(AsCf(cf)->statUnitCount, 0);
    EXPECT_EQ(AsCf(cf)->baseTilesPerCore, 0);
    EXPECT_EQ(AsCf(cf)->extraCoreCount, 0);

    // channel-last:NHWC (2,4,5,0),统计量 (0,) ⇒ C = 0。
    const TilingRun cl = RunTiling({2, 4, 5, 0}, ge::FORMAT_NHWC, ge::DT_FLOAT, 0);
    EXPECT_EQ(cl.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(cl.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_LAST));
    ASSERT_EQ(cl.data.size(), BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES);
    EXPECT_EQ(AsCl(cl)->c, 0);
    EXPECT_EQ(AsCl(cl)->statUnitCount, 0);
    EXPECT_EQ(AsCl(cl)->baseTilesPerCore, 0);
    EXPECT_EQ(AsCl(cl)->extraCoreCount, 0);
}

// num == 0(C >= 1、某非通道轴为 0)⇒ 零统计契约,**不报错**(与 C == 0 同为空张量,但 C >= 1)。
TEST_F(BN3DTrainingUpdateTilingTest, empty_batch_supported_channel_first)
{
    const TilingRun run = RunTiling({0, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3);
    EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_FIRST));
    const auto* td = AsCf(run);
    EXPECT_EQ(td->baseTilesPerCore * static_cast<int64_t>(run.blockNum) + td->extraCoreCount, 0);
    EXPECT_EQ(td->numRec, 0.0f);
    EXPECT_EQ(td->besselScaler, 0.0f);
    EXPECT_GE(td->statUnitCount, 1); // 相 S 单元照常覆盖 [0, C)
}

TEST_F(BN3DTrainingUpdateTilingTest, empty_batch_supported_channel_last)
{
    const TilingRun run = RunTiling({2, 4, 0, 3}, ge::FORMAT_NHWC, ge::DT_FLOAT, 3);
    EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_LAST));
    const auto* td = AsCl(run);
    EXPECT_EQ(td->num, 0);
    EXPECT_EQ(td->baseTilesPerCore * static_cast<int64_t>(run.blockNum) + td->extraCoreCount, 0);
    EXPECT_EQ(td->numRec, 0.0f);
    EXPECT_GE(td->statUnitCount, 1);
}

// num == 1:Bessel scaler 置 0(仅影响 running variance 的 EMA)。
TEST_F(BN3DTrainingUpdateTilingTest, single_sample_batch_bessel_zero)
{
    const TilingRun run = RunTiling({1, 3, 1, 1, 1}, ge::FORMAT_NCDHW, ge::DT_FLOAT, 3);
    ASSERT_EQ(run.status, ge::GRAPH_SUCCESS);
    const auto* td = AsCf(run);
    EXPECT_EQ(td->numRec, 1.0f);
    EXPECT_EQ(td->besselScaler, 0.0f);
}

TEST_F(BN3DTrainingUpdateTilingTest, bessel_scaler_applied_for_num_gt_one)
{
    const TilingRun run = RunTiling({2, 3, 2, 2}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3);
    ASSERT_EQ(run.status, ge::GRAPH_SUCCESS);
    const auto* td = AsCf(run);
    EXPECT_FLOAT_EQ(td->numRec, 1.0f / 8.0f);
    EXPECT_FLOAT_EQ(td->besselScaler, 8.0f / 7.0f);
}

// ============================ 负向:非法输入干净拒绝 ============================

TEST_F(BN3DTrainingUpdateTilingTest, negative_rank3_rejected)
{
    EXPECT_EQ(RunTiling({2, 3, 4}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3).status, ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, negative_rank5_with_4d_format_rejected)
{
    EXPECT_EQ(RunTiling({2, 3, 4, 5, 6}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3).status, ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, negative_unsupported_format_rejected)
{
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT, 3).status, ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, negative_dtype_rejected)
{
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_INT32, 3).status, ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, negative_channel_mismatch_rejected)
{
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 7).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NHWC, ge::DT_FLOAT, 7).status, ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, stat_format_label_irrelevant_when_rank1)
{
    const ge::Format labels[] = {ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_NCDHW};
    for (const ge::Format label : labels) {
        const TilingRun run = RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, label);
        EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_FIRST));
        ASSERT_EQ(run.data.size(), BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES);
        EXPECT_EQ(AsCf(run)->c, 3);
    }
}

TEST_F(BN3DTrainingUpdateTilingTest, stat_layout_rank_gt1_rejected_by_rank)
{
    // 对照(同 label NC1HWC0、仅 rank 不同):rank-1 ⇒ 通过;下一行 rank-5 ⇒ 拒 ⇒ 否决权在 rank。
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_NC1HWC0).status,
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(
        RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_NC1HWC0, {1, 1, 4, 5, 16})
            .status,
        ge::GRAPH_FAILED);
    EXPECT_EQ(
        RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_NCDHW, {1, 3, 4, 5, 6})
            .status,
        ge::GRAPH_FAILED);
    EXPECT_EQ(
        RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_FRACTAL_NZ, {1, 3, 16, 32})
            .status,
        ge::GRAPH_FAILED);
    // 同 rank 换 label:ND 标签 + rank-5 一样被拒 ⇒ 否决权在 rank,不在 format。
    EXPECT_EQ(
        RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_ND, {1, 3, 4, 5, 6}).status,
        ge::GRAPH_FAILED);
}

// 统计量非 rank-1(带布局的二维/更高维张量)仍然被拒绝——"统计量必须纯一维"的意图未丢。
TEST_F(BN3DTrainingUpdateTilingTest, stat_rank_two_rejected)
{
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_NCHW, {1, 3}).status,
              ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_ND, {3, 1}).status,
              ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, stat_output_rank_gt1_rejected_by_shape_equality)
{
    // 对照:输出 (3,) 与输入同形 ⇒ 通过。
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3).status, ge::GRAPH_SUCCESS);
    // 输出声明成 rank-2 (1,3):与输入 (3,) 形状不等(rank 即不等)⇒ 被 shape 相等校验拒。
    EXPECT_EQ(
        RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_ND, {}, {1, 3}).status,
        ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, stat_output_shape_mismatch_rejected)
{
    // 对照:输出 (3,) 与输入同形 ⇒ 通过。
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3).status, ge::GRAPH_SUCCESS);
    // 输出 rank-1 但长度 (4,) != C = 3 → 仅"输出必须等于输入 (C,) 形状"能拒它。
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_ND, {}, {4}).status,
              ge::GRAPH_FAILED);
}

// ============================ 周期展平(key 4000)============================

namespace {
struct FlatGateCase {
    std::vector<int64_t> dims;
    ge::Format format;
    ge::DataType dtype;
    int64_t statLen;
    int64_t expectKey;
    int64_t expectPeriod;
    int64_t expectUnroll;
    const char* label;
};
} // namespace

// 门限真值表(设计 §10.3 v2 tier-1):
//   进 4000 ⇔ C >= 1 && num > 0 && total = num·C >= 4096
//          && ( channelLast ? C <= 2 : (C <= 4 && S < VL) )
//          && unroll <= 4 && UB 可解,  其中 period = (C == 1) ? 1 : C·S
// 覆盖:issue 7 条(v1)形态、§10.4 的 4 条新吸收形态、以及必须留在老路径的对照
// (channel-last C >= 3、channel-first S >= VL 含 C=1 大 S、极小形状、空 tensor)。
TEST_F(BN3DTrainingUpdateTilingTest, flat_gate_truth_table)
{
    const FlatGateCase cases[] = {
        // ---- 走 4000:channel-last 的 C = 1/2(通道紧邻,P = C)----
        {{6, 2, 47590, 4, 1}, ge::FORMAT_NDHWC, ge::DT_FLOAT, 1, 4000, 1, 1, "L1_011 C=1 fp32"},
        {{3, 1720023, 4, 1}, ge::FORMAT_NHWC, ge::DT_FLOAT16, 1, 4000, 1, 1, "L1_080 C=1 fp16"},
        {{1, 2, 11738724, 1}, ge::FORMAT_NHWC, ge::DT_FLOAT16, 1, 4000, 1, 1, "L1_296 C=1 fp16"},
        {{7872, 2, 2, 284, 2}, ge::FORMAT_NDHWC, ge::DT_FLOAT16, 2, 4000, 2, 1, "L1_089 C=2 fp16"},
        {{2, 169, 2, 5705, 2}, ge::FORMAT_NDHWC, ge::DT_FLOAT, 2, 4000, 2, 1, "L1_333 C=2 fp32"},
        {{1, 1, 271643, 2, 2}, ge::FORMAT_NDHWC, ge::DT_BF16, 2, 4000, 2, 1, "L1_096 C=2 bf16"},
        {{1608871, 2, 2, 2}, ge::FORMAT_NHWC, ge::DT_FLOAT16, 2, 4000, 2, 1, "L1_302 C=2 fp16"},
        // ---- 走 4000:channel-first 的 C = 2 小游程(S < VL,P = C·S)----
        {{1272457, 2, 2, 2}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 2, 4000, 8, 1, "L1_038 C=2 S=4 m=1"},
        {{382737, 2, 2, 3, 2}, ge::FORMAT_NCDHW, ge::DT_FLOAT16, 2, 4000, 24, 3, "L1_378 C=2 S=12 m=3"},
        {{200000, 2, 2, 24}, ge::FORMAT_NCHW, ge::DT_FLOAT, 2, 4000, 96, 3, "CF C=2 S=48 m=3"},
        // ---- 走 4000:v2 tier-1 新吸收的 4 条(S < VL、C <= 4)----
        // §10.6 形态 1:C == 1 ⇒ 周期退化为 1,与 S 无关(S=9 非 2 的幂)
        {{515998, 1, 3, 3}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 1, 4000, 1, 1, "L1_520 C=1 S=9 P=1 m=1"},
        // §10.6 形态 2:C = 3/4 且 m = 3(现网此前只验证过 C=2, P=24, m=3)
        {{1790118, 3, 2, 2}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 3, 4000, 12, 3, "L1_505 C=3 S=4 P=12 m=3"},
        {{1314137, 4, 2, 2, 2}, ge::FORMAT_NCDHW, ge::DT_FLOAT16, 4, 4000, 32, 1, "L1_525 C=4 S=8 P=32 m=1"},
        {{508680, 4, 2, 6}, ge::FORMAT_NCHW, ge::DT_BF16, 4, 4000, 48, 3, "rangebound_180 C=4 S=12 P=48 m=3"},
        // C == 1 的 channel-first 也走"周期退化"路径(不再受 S 的整除性影响)
        {{200000, 1, 1, 3}, ge::FORMAT_NCHW, ge::DT_FLOAT, 1, 4000, 1, 1, "CF C=1 S=3 P=1 m=1"},
        // ---- 边界(显式钉住三条判据的临界值)----
        // S 判据是严格小于 VL:S = VL-1 吸收、S = VL 拒收(两侧都要钉)
        {{1000, 1, 7, 9}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 1, 4000, 1, 1, "边界 S=63(=VL-1)吸收"},
        {{1000, 1, 8, 8}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 1, 2000, 0, 0, "边界 S=64(=VL)拒收"},
        // 规模下限是 total >= 4096:4095 拒收、4096 吸收
        {{1, 4095, 1, 1}, ge::FORMAT_NHWC, ge::DT_FLOAT16, 1, 3000, 0, 0, "边界 total=4095 拒收"},
        {{1, 4096, 1, 1}, ge::FORMAT_NHWC, ge::DT_FLOAT16, 1, 4000, 1, 1, "边界 total=4096 吸收"},
        // ---- 留在 3000:channel-last C >= 3(tier-1 结构性隔离)----
        {{2, 4, 5, 3}, ge::FORMAT_NHWC, ge::DT_FLOAT, 3, 3000, 0, 0, "CL C=3"},
        {{3, 622, 855, 3, 3}, ge::FORMAT_NDHWC, ge::DT_FLOAT16, 3, 3000, 0, 0, "L1_228 C=3"},
        {{6144, 2, 2, 3}, ge::FORMAT_NHWC, ge::DT_FLOAT16, 3, 3000, 0, 0, "CL C=3 大 num"},
        // ---- 留在 2000:channel-first 的 S >= VL(旧路径已能打满向量,C=1/2 大 S 反而更快)----
        {{2, 2, 2, 2, 1100599}, ge::FORMAT_NCDHW, ge::DT_FLOAT, 2, 2000, 0, 0, "L1_338 S=4.4M"},
        {{3, 2, 2, 2, 7598411}, ge::FORMAT_NCDHW, ge::DT_FLOAT, 2, 2000, 0, 0, "L1_356 S=30.4M"},
        {{200000, 2, 2, 48}, ge::FORMAT_NCHW, ge::DT_FLOAT, 2, 2000, 0, 0, "CF C=2 S=96 >= VL"},
        {{200000, 2, 1, 128}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 2, 2000, 0, 0, "CF C=2 S=128 >= VL"},
        {{14, 1, 19, 6, 11}, ge::FORMAT_NCDHW, ge::DT_FLOAT16, 1, 2000, 0, 0, "L1_071 C=1 S=1254 >= VL"},
        // ---- 留在 2000:C >= 5 的 channel-first(S < VL 也不行)----
        {{100000, 4, 10, 10}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 4, 2000, 0, 0, "CF C=4 S=100 >= VL"},
        {{200000, 5, 2, 4}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 5, 2000, 0, 0, "CF C=5 超上界"},
        // ---- 留在老路径:total < 4096 / C == 0 / 空 tensor ----
        {{1, 1, 1, 1}, ge::FORMAT_NHWC, ge::DT_FLOAT, 1, 3000, 0, 0, "total=1 极小"},
        {{2, 0, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0, 2000, 0, 0, "C=0 CF"},
        {{2, 4, 5, 0}, ge::FORMAT_NHWC, ge::DT_FLOAT, 0, 3000, 0, 0, "C=0 CL"},
        {{0, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 2000, 0, 0, "num=0 CF"},
        {{2, 4, 0, 3}, ge::FORMAT_NHWC, ge::DT_FLOAT, 3, 3000, 0, 0, "num=0 CL"},
    };
    for (const auto& c : cases) {
        const TilingRun run = RunTiling(c.dims, c.format, c.dtype, c.statLen);
        ASSERT_EQ(run.status, ge::GRAPH_SUCCESS) << c.label;
        EXPECT_EQ(run.key, c.expectKey) << c.label;
        if (c.expectKey != static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_FLAT)) {
            continue;
        }
        EXPECT_EQ(run.dataSize, BN3D_TRAINING_UPDATE_FLAT_TILING_DATA_BYTES) << c.label;
        const auto* td = AsFlat(run);
        EXPECT_EQ(td->period, c.expectPeriod) << c.label;
        EXPECT_EQ(td->unroll, c.expectUnroll) << c.label;
        EXPECT_EQ(td->laneBlock, td->unroll * kVlFp32) << c.label;
        // 周期必须整除展开块(保瓦片内相位为 0),展开倍数受上界约束。
        EXPECT_EQ(td->laneBlock % td->period, 0) << c.label;
        EXPECT_LE(td->unroll, optiling::kFlatMaxUnroll) << c.label;
    }
}

// 瓦片切分不变量:UB 记账不超、laneBlock 对齐、切分和恒等于瓦片总数、VF 计数器上界。
TEST_F(BN3DTrainingUpdateTilingTest, flat_tile_split_invariants)
{
    const FlatGateCase cases[] = {
        {{6, 2, 47590, 4, 1}, ge::FORMAT_NDHWC, ge::DT_FLOAT, 1, 4000, 1, 1, "L1_011"},
        {{7872, 2, 2, 284, 2}, ge::FORMAT_NDHWC, ge::DT_FLOAT16, 2, 4000, 2, 1, "L1_089"},
        {{1272457, 2, 2, 2}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 2, 4000, 8, 1, "L1_038"},
        {{382737, 2, 2, 3, 2}, ge::FORMAT_NCDHW, ge::DT_FLOAT16, 2, 4000, 24, 3, "L1_378"},
        {{1, 2, 11738724, 1}, ge::FORMAT_NHWC, ge::DT_FLOAT16, 1, 4000, 1, 1, "L1_296"},
        {{2, 169, 2, 5705, 2}, ge::FORMAT_NDHWC, ge::DT_FLOAT, 2, 4000, 2, 1, "L1_333"},
        {{1024, 2, 2, 2}, ge::FORMAT_NCHW, ge::DT_FLOAT, 2, 4000, 8, 1, "CF 小 total=8192"},
        // C == 1 的 channel-first:周期退化为 1(与 S 无关)—— 相位断言不能写成 C·R,本行负责盯住它
        {{515998, 1, 3, 3}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 1, 4000, 1, 1, "L1_520 C=1 CF S=9"},
    };
    const auto plat = Platform();
    for (const auto& c : cases) {
        const TilingRun run = RunTiling(c.dims, c.format, c.dtype, c.statLen);
        ASSERT_EQ(run.status, ge::GRAPH_SUCCESS) << c.label;
        ASSERT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_FLAT)) << c.label;
        const auto* td = AsFlat(run);
        const int64_t elemBytes = ElemBytesOf(c.dtype);
        // ① UB 记账:整 kernel 占用不超 UB
        EXPECT_LE(optiling::FlatKernelBytes(td->tileElems, elemBytes, td->laneBlock, plat), plat.ubSize) << c.label;
        // ② 相位:P | L | tileElems(单一瓦片时 tileElems == total,相位恒为 0)。
        //    C == 1 时周期退化为 1(通道恒为 0,与 S 无关)⇒ 不能用 C·R 表达。
        EXPECT_EQ(td->period, td->c == 1 ? 1 : td->c * td->channelRun) << c.label;
        EXPECT_EQ(td->laneBlock % td->period, 0) << c.label;
        EXPECT_EQ(td->laneBlock, td->unroll * kVlFp32) << c.label;
        if (td->tileCount > 1) {
            EXPECT_EQ(td->tileElems % td->laneBlock, 0) << c.label;
        }
        // ③ 切分:瓦片总数与每核段之和恒等
        EXPECT_EQ(td->tileCount, (td->total + td->tileElems - 1) / td->tileElems) << c.label;
        EXPECT_EQ(td->baseTilesPerCore * static_cast<int64_t>(run.blockNum) + td->extraCoreCount, td->tileCount)
            << c.label;
        EXPECT_GE(td->tileCount, 1) << c.label;
        // ④ VF 循环计数器(uint16_t)上界
        EXPECT_LE(td->tileElems, optiling::kMaxTileElementsPerVf) << c.label;
        // ⑤ total / numRec 自洽
        EXPECT_GE(td->total, optiling::kFlatMinElements) << c.label;
        EXPECT_GT(td->numRec, 0.0f) << c.label;
    }
}

// 4000 的 ABI:结构体必须与宿主 TILING_DATA_DEF 序列化布局逐字节一致。
TEST_F(BN3DTrainingUpdateTilingTest, flat_tiling_data_abi_size)
{
    EXPECT_EQ(sizeof(Bn3dTrainingUpdateFlatTilingData), 136U);
    EXPECT_EQ(BN3D_TRAINING_UPDATE_FLAT_TILING_DATA_BYTES, 136U);
    const TilingRun run = RunTiling({6, 2, 47590, 4, 1}, ge::FORMAT_NDHWC, ge::DT_FLOAT, 1);
    ASSERT_EQ(run.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(run.dataSize, BN3D_TRAINING_UPDATE_FLAT_TILING_DATA_BYTES);
}

// 周期块解算:SolveFlatTileElems 的输出必须是 laneBlock 的整数倍(除非退化为单一瓦片),
// 且调用方传入的 laneBlock 由 (C, R, VL) 唯一确定。
TEST_F(BN3DTrainingUpdateTilingTest, flat_pattern_period_isolation)
{
    const auto plat = Platform();
    // channel-last C=1/2:周期 <= VL ⇒ m = 1
    EXPECT_EQ(optiling::SolveFlatTileElems(1000000, 4, 64, plat) % 64, 0);
    // 极小 total(仍 >= kFlatMinElements)不得切出 0 长度瓦片,且瓦片数 >= 1
    const int64_t tiny = optiling::SolveFlatTileElems(optiling::kFlatMinElements, 4, 64, plat);
    EXPECT_GT(tiny, 0);
    EXPECT_LE(tiny, optiling::kFlatMinElements);
}

// 4000 的两条拒收分支:**门限成立后不可达**,这里把"不可达"依赖的前提逐条钉死,
// 使后续改动(改门限 / 改常量 / 改 TilingData 注册)一旦破坏前提就当场失败,而不是静默失效。
//   ① TilingFlat() 的 `tileElems_ < 1`:只在 total <= 0 或 laneBlock <= 0 时发生,
//      而 FlatPathGate() 已保证 num_ > 0 && total >= kFlatMinElements && laneBlock = m·VL >= VL;
//      并且门限内的任何 (total, laneBlock) 都解出 >= laneBlock 的瓦片。
//   ② FillFlat() 的容量分支:调用方给出的 tiling 容量(kTilingDataCap)远大于 ABI 136B。
//   ③ FillFlat() 的 SetTilingKey 分支:4000 已 REGISTER_TILING_DATA_CLASS,key 恒为 4000。
TEST_F(BN3DTrainingUpdateTilingTest, flat_reject_branches_unreachable_after_gate)
{
    const auto plat = Platform();

    // ---- ① TilingFlat 的 tileElems_ < 1 ----
    EXPECT_EQ(optiling::SolveFlatTileElems(0, 4, 64, plat), 0);      // total == 0:门限已排除
    EXPECT_EQ(optiling::SolveFlatTileElems(-1, 4, 64, plat), 0);     // total < 0:门限已排除
    EXPECT_EQ(optiling::SolveFlatTileElems(1000000, 4, 0, plat), 0); // L == 0:门限已保证 L >= VL
    for (const int64_t total : {static_cast<int64_t>(optiling::kFlatMinElements), int64_t{1} << 20, int64_t{1} << 25}) {
        for (const int64_t laneBlock : {int64_t{64}, int64_t{192}, int64_t{256}}) {
            const int64_t te = optiling::SolveFlatTileElems(total, 4, laneBlock, plat);
            EXPECT_GE(te, laneBlock) << "total=" << total << " L=" << laneBlock;
        }
    }

    const FlatGateCase flats[] = {
        {{6, 2, 47590, 4, 1}, ge::FORMAT_NDHWC, ge::DT_FLOAT, 1, 4000, 1, 1, "L1_011 m=1"},
        {{382737, 2, 2, 3, 2}, ge::FORMAT_NCDHW, ge::DT_FLOAT16, 2, 4000, 24, 3, "L1_378 m=3"},
        {{200000, 4, 2, 24}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 4, 4000, 96, 3, "CF C=4 S=48 m=3"},
        {{515998, 1, 3, 3}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 1, 4000, 1, 1, "L1_520 C=1 P=1"},
    };
    for (const auto& c : flats) {
        const TilingRun run = RunTiling(c.dims, c.format, c.dtype, c.statLen);
        ASSERT_EQ(run.status, ge::GRAPH_SUCCESS) << c.label;
        ASSERT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_FLAT)) << c.label;
        const auto* td = AsFlat(run);
        const int64_t elemBytes = ElemBytesOf(c.dtype);
        // 门限的第 4 条(一个周期块的瓦片必须装得进 UB)⇒ TilingFlat 的 UB 反解恒成功
        EXPECT_LE(optiling::FlatKernelBytes(td->laneBlock, elemBytes, td->laneBlock, plat), plat.ubSize) << c.label;
        EXPECT_GE(td->tileElems, td->laneBlock) << c.label;
        // ---- ② FillFlat 的容量分支 ----
        EXPECT_GE(static_cast<size_t>(kTilingDataCap), BN3D_TRAINING_UPDATE_FLAT_TILING_DATA_BYTES) << c.label;
        // ---- ③ FillFlat 的 SetTilingKey 分支(dataSize 未被截断 + key 确为 4000)----
        EXPECT_EQ(run.dataSize, BN3D_TRAINING_UPDATE_FLAT_TILING_DATA_BYTES) << c.label;
        EXPECT_EQ(run.key, 4000) << c.label;
    }
}
