/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_unsorted_segment_sum_tiling.cpp
 * \brief arch35 (Ascend950) tiling UT for unsorted_segment_sum error-code remediation
 */

#include <iostream>
#include <map>
#include <string>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "base/err_mgr.h"
#include "../../../../op_host/arch35/unsorted_segment_sum_tiling_arch35.h"

namespace optiling {
// CalcTiling4Float is compiled into the op_host library but not exported by the
// tiling header; declare the EleByte enum and the function so this UT can
// drive the converted EZ0026 site directly (identical enumerator values keep
// the ODR and the mangled symbol name consistent with the library). The
// enclosing SegmentTIKTiling chain is compiled but not registered in the
// template registry, so the registered tiling entry cannot reach it.
enum EleByte { FP16_BYTE = 2, FP32_BYTE = 4, INT32_BYTE = 4, INT64_BYTE = 8, INT8_BYTE = 1, UINT8_BYTE = 1 };
ge::graphStatus CalcTiling4Float(gert::TilingContext* context, const int32_t e_size, const int32_t num_segments,
                                 const int32_t input_size, const int32_t ids_size,
                                 const int32_t output_ub_ele_num_one_row, const EleByte input_ele_byte,
                                 const EleByte ids_ele_byte);
} // namespace optiling

using namespace ut_util;

class UnsortedSegmentSumTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UnsortedSegmentSumTilingArch35 SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "UnsortedSegmentSumTilingArch35 TearDown" << std::endl; }
};

namespace {

gert::TilingContext* BuildSumTilingContext(gert::KernelRunContextHolder& holder,
                                           optiling::TilingPrepareForUnsortedSegmentSumCompileInfo& compileInfo)
{
    compileInfo.core_num = 64;
    compileInfo.ub_size = 196608;
    compileInfo.ub_tensor_num = 2;
    compileInfo.impl_mode = 0;
    compileInfo.max_thread = 512;

    static gert::StorageShape xShape = {{4, 4}, {4, 4}};
    static gert::StorageShape yShape = {{4, 4}, {4, 4}};
    static auto tilingDataCap = gert::TilingData::CreateCap(8192);
    static fe::PlatFormInfos platformInfo;
    static std::map<std::string, std::string> socInfos;
    static std::map<std::string, std::string> aicoreSpec;
    static std::map<std::string, std::string> intrinsics;
    if (socInfos.empty()) {
        GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", socInfos, aicoreSpec, intrinsics);
        platformInfo.Init();
    }
    auto workspace = gert::ContinuousVector::Create<size_t>(4096);
    gert::TilingContextFaker faker;
    faker.SetOpType("UnsortedSegmentSum")
        .NodeIoNum(1, 1)
        .IrInstanceNum({1}, {1})
        .InputShapes({&xShape})
        .OutputShapes({&yShape})
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .TilingData(tilingDataCap.get())
        .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()));
    holder = faker.Build();
    return holder.GetContext<gert::TilingContext>();
}

void ExpectSingleRawError(const char* expectedCode, const std::map<std::string, std::string>& expectedMap)
{
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(std::string(errors.front().error_id.get()), expectedCode);
    std::map<std::string, std::string> actual;
    for (size_t j = 0; j < errors.front().args_key.size(); ++j) {
        EXPECT_TRUE(actual.emplace(errors.front().args_key[j].get(), errors.front().args_value[j].get()).second);
    }
    EXPECT_EQ(actual, expectedMap);
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

} // namespace

// ---------------------------------------------------------------------------
// Error-code remediation negative test: GetTilingMode has no branch for a
// non-positive num_segments, so a num_segments of 0 makes it return false and
// the CalcTiling4Float wrapper reports a single formal EZ0026 first error with
// the offending value instead of the previous generic EZ9999.
// ---------------------------------------------------------------------------
TEST_F(UnsortedSegmentSumTilingArch35, tiling_mode_failure_reports_single_formal_error)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();

    optiling::TilingPrepareForUnsortedSegmentSumCompileInfo compileInfo;
    gert::KernelRunContextHolder holder;
    auto* ctx = BuildSumTilingContext(holder, compileInfo);
    ASSERT_NE(ctx, nullptr);

    EXPECT_EQ(optiling::CalcTiling4Float(ctx, /* e_size */ 4, /* num_segments */ 0, /* input_size */ 16,
                                         /* ids_size */ 4, /* output_ub_ele_num_one_row */ 8, optiling::FP32_BYTE,
                                         optiling::INT32_BYTE),
              ge::GRAPH_FAILED);
    ExpectSingleRawError("EZ0026", {{"param_name", "num_segments"},
                                    {"op_name", "UnsortedSegmentSum"},
                                    {"incorrect_value", "0"},
                                    {"reason", "GetTilingMode failed: num_segments must be a positive integer, and "
                                               "x must contain at least one element when num_segments is greater "
                                               "than 1 (x contains 16 elements)"}});
}
