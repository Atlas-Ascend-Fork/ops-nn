/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv2d_v2_ascendc_stridewrite_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "log/log.h"
#include "array_ops.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base_utils.h"
#include "conv/conv2d_v2/op_host/op_tiling/conv2d_v2_tiling.h"
#include "../../../../conv2d_v2/op_host/op_tiling/arch35/conv2d_v2_base_tiling.h"

using namespace std;
using namespace ge;
using namespace ut_util;

namespace {
constexpr uint64_t MEM_SIZE_4K = 4096;
constexpr int64_t HW = 16;
constexpr int64_t ORG_COUT = 16;

int32_t RunStrideWriteTilingCase(const string& format, const vector<int64_t>& totalShape, uint64_t& outTotalShapeCout,
                                 uint64_t& outDstStride)
{
    optiling::conv_ops_tiling::Conv2dTilingCache::GetInstance().SetSocVersion(
        NpuArch::DAV_RESV, optiling::conv_ops_tiling::ConvTilingParseInfo());
    const int64_t batch = 1;
    const int64_t cin = 16;
    const int64_t coutCh = ORG_COUT;
    ge::Format fmapFormat = ge::FORMAT_NCHW;
    ge::Format weightFormat = ge::FORMAT_NCHW;
    ge::Format outputFormat = ge::FORMAT_NCHW;
    gert::StorageShape featuremap = {{batch, cin, HW, HW}, {batch, cin, HW, HW}};
    gert::StorageShape weight = {{coutCh, cin, 1, 1}, {coutCh, cin, 1, 1}};
    gert::StorageShape bias;
    gert::StorageShape output = {{batch, coutCh, HW, HW}, {batch, coutCh, HW, HW}};
    if (format == "NHWC") {
        fmapFormat = ge::FORMAT_NHWC;
        weightFormat = ge::FORMAT_HWCN;
        outputFormat = ge::FORMAT_NHWC;
        featuremap = {{batch, HW, HW, cin}, {batch, HW, HW, cin}};
        weight = {{1, 1, cin, coutCh}, {1, 1, cin, coutCh}};
        output = {{batch, HW, HW, coutCh}, {batch, HW, HW, coutCh}};
    }
    std::vector<void*> inputShapeRef = {&featuremap, &weight, nullptr, nullptr};
    std::vector<void*> outputShapeRef = {&output};
    std::vector<int64_t> stridesVec = (format == "NHWC") ? vector<int64_t>{1, 1, 1, 1} : vector<int64_t>{1, 1, 1, 1};
    std::vector<int64_t> padsVec = {0, 0, 0, 0};
    std::vector<int64_t> dilationsVec = {1, 1, 1, 1};

    string compileInfoString = R"({"hardware_info":
        {"BT_SIZE": 4096, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
        "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 253952,
        "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "FB_SIZE": 4096,
        "BT_SIZE": 4096, "L0C_SIZE": 262144, "CORE_NUM": 32, "cube_core_cnt": 32, "vector_core_cnt": 64,
        "core_type_list": "CubeCore,VectorCore"}})";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);
    socInfos["cube_vector_combine"] = "fuse";
    std::map<std::string, std::string> socVersionInfos = {
        {"SoC_version", "MC62CM12AA"}, {"Short_SoC_version", "MC62CM12AA"}, {"NpuArch", "5102"}};
    aicoreSpec.insert({"fb0_size", "4096"});
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::conv_ops_tiling::ConvTilingParseInfo compileInfo;

    auto tilingDataPtr = gert::TilingData::CreateCap(MEM_SIZE_4K);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(MEM_SIZE_4K);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> nodeAttrs = {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stridesVec)},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padsVec)},
        {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilationsVec)},
        {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
        {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(format)},
        {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
        {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
        {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
        {"_reserved_slot", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
        {"fixed_shift_value", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}};
    if (!totalShape.empty()) {
        nodeAttrs.emplace_back("total_shape", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(totalShape));
    }

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv2DV2")
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes(inputShapeRef)
                      .OutputShapes(outputShapeRef)
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, fmapFormat, fmapFormat)
                      .NodeInputTd(1, ge::DT_FLOAT16, weightFormat, weightFormat)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, outputFormat, outputFormat)
                      .NodeAttrs(nodeAttrs)
                      .TilingData(tilingDataPtr.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext->GetPlatformInfo() == nullptr) {
        return -1;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DV2");
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return -1;
    }
    if (opImpl->tiling(tilingContext) != ge::GRAPH_SUCCESS) {
        return 1;
    }
    auto rawTilingData = tilingContext->GetRawTilingData();
    const uint8_t* rawBuf = reinterpret_cast<const uint8_t*>(rawTilingData->GetData());
    size_t rawSize = rawTilingData->GetDataSize();
    if (rawSize < 16) {
        return -1;
    }
    auto readTailU64 = [&rawBuf, &rawSize](const size_t idxFromEnd) -> uint64_t {
        size_t off = rawSize - 8 - idxFromEnd * 8;
        uint64_t v = 0;
        for (int k = 7; k >= 0; k--) {
            v = (v << 8) | rawBuf[off + k];
        }
        return v;
    };
    outDstStride = readTailU64(0);
    outTotalShapeCout = readTailU64(1);
    return 0;
}
} // namespace

class Conv2dV2StrideWriteTiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(Conv2dV2StrideWriteTiling, stridewrite_enable_nhwc)
{
    uint64_t totalShapeCout = 0;
    uint64_t dstStride = 0;
    ASSERT_EQ(RunStrideWriteTilingCase("NHWC", {1, HW, HW, 32}, totalShapeCout, dstStride), 0);
    EXPECT_EQ(totalShapeCout, 32);
    EXPECT_EQ(dstStride, 32);
}

TEST_F(Conv2dV2StrideWriteTiling, stridewrite_enable_nchw)
{
    uint64_t totalShapeCout = 0;
    uint64_t dstStride = 0;
    ASSERT_EQ(RunStrideWriteTilingCase("NCHW", {1, 32, HW, HW}, totalShapeCout, dstStride), 0);
    EXPECT_EQ(totalShapeCout, 32);
    EXPECT_EQ(dstStride, 32 * HW * HW);
}

TEST_F(Conv2dV2StrideWriteTiling, stridewrite_reject_hw_mismatch)
{
    uint64_t totalShapeCout = 0;
    uint64_t dstStride = 0;
    ASSERT_EQ(RunStrideWriteTilingCase("NCHW", {1, 32, HW + 1, HW}, totalShapeCout, dstStride), 1);
}

TEST_F(Conv2dV2StrideWriteTiling, stridewrite_reject_c_not_larger)
{
    uint64_t totalShapeCout = 0;
    uint64_t dstStride = 0;
    ASSERT_EQ(RunStrideWriteTilingCase("NCHW", {1, 8, HW, HW}, totalShapeCout, dstStride), 1);
}

TEST_F(Conv2dV2StrideWriteTiling, stridewrite_no_attr_not_enabled)
{
    uint64_t totalShapeCout = 0;
    uint64_t dstStride = 0;
    ASSERT_EQ(RunStrideWriteTilingCase("NCHW", {}, totalShapeCout, dstStride), 0);
    EXPECT_EQ(totalShapeCout, 0);
    EXPECT_EQ(dstStride, 0);
}

TEST_F(Conv2dV2StrideWriteTiling, stridewrite_not_4d_not_enabled)
{
    uint64_t totalShapeCout = 0;
    uint64_t dstStride = 0;
    ASSERT_EQ(RunStrideWriteTilingCase("NCHW", {1, 32, HW}, totalShapeCout, dstStride), 0);
    EXPECT_EQ(totalShapeCout, 0);
    EXPECT_EQ(dstStride, 0);
}

TEST_F(Conv2dV2StrideWriteTiling, stridewrite_reject_not_aligned)
{
    uint64_t totalShapeCout = 0;
    uint64_t dstStride = 0;
    ASSERT_EQ(RunStrideWriteTilingCase("NCHW", {1, 33, HW, HW}, totalShapeCout, dstStride), 1);
}
