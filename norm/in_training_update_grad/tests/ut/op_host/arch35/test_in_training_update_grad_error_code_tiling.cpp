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
 * \file test_in_training_update_grad_error_code_tiling.cpp
 * \brief INTrainingUpdateGrad arch35 tiling 错误码 UT（EZ0009）。
 *
 *        覆盖 GetShapeAttrsInfo 的内核包络门（C0 lane 上界 / uint32 乘积），断言：
 *        - tiling 返回 GRAPH_FAILED；
 *        - 首错唯一（count == 1）；
 *        - 错误码精确等于 EZ0009，param_name=dy，reason 命中具体乘积项。
 *
 *        仅 arch35（ascend950）收录：本目录依赖 cmake/ut.cmake 的 target_dir glob 裁剪，
 *        ascend910b（arch22）不编译本文件，避免无 tiling 注册时的空指针。
 */

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "base/err_mgr.h"
#include "../../../../op_host/arch35/in_training_update_grad_tiling_arch35.h"

using namespace std;
using namespace ge;

namespace {
constexpr const char* kOpType = "INTrainingUpdateGrad";

// ErrorManager 上下文隔离：用独立 work_stream_id 记录，避免共享 Host UT 进程内的用例串扰。
class ScopedErrorContext {
public:
    ScopedErrorContext() : savedContext_(error_message::GetErrMgrContext())
    {
        auto context = savedContext_;
        context.work_stream_id = reinterpret_cast<std::uintptr_t>(this);
        error_message::SetErrMgrContext(context);
        (void)error_message::GetErrMgrRawErrorMessages();
    }

    ~ScopedErrorContext()
    {
        (void)error_message::GetErrMgrRawErrorMessages();
        error_message::SetErrMgrContext(savedContext_);
    }

    ScopedErrorContext(const ScopedErrorContext&) = delete;
    ScopedErrorContext& operator=(const ScopedErrorContext&) = delete;

private:
    error_message::ErrorManagerContext savedContext_;
};

struct CapturedError {
    size_t count = 0;
    std::string code;
    std::vector<std::string> keys;
    std::vector<std::string> values;
};

// GetErrMgrRawErrorMessages() 读取即消费：每条用例只调用一次。
CapturedError CaptureError()
{
    CapturedError captured;
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    captured.count = errors.size();
    if (errors.empty()) {
        return captured;
    }
    const auto& error = errors.back();
    captured.code = error.error_id == nullptr ? "" : error.error_id.get();
    for (const auto& key : error.args_key) {
        captured.keys.emplace_back(key == nullptr ? "" : key.get());
    }
    for (const auto& value : error.args_value) {
        captured.values.emplace_back(value == nullptr ? "" : value.get());
    }
    return captured;
}

// 按 Arglist 的 key 取实际错误值；缺失时返回可读标记，便于断言失败时定位。
std::string ErrorValue(const CapturedError& error, const std::string& key)
{
    for (size_t i = 0; i < error.keys.size() && i < error.values.size(); ++i) {
        if (error.keys[i] == key) {
            return error.values[i];
        }
    }
    return "<missing:" + key + ">";
}

bool Contains(const std::string& text, const std::string& sub) { return text.find(sub) != std::string::npos; }

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (auto d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

// 四输入(dy/x/variance/mean) 两输出(res_gamma/res_beta)，format 固定 NDC1HWC0。
// 平台参数与现有 tiling UT 一致：coreNum=64 / ubSize=245760 / vectorLength=256 / ubBlockSize=32。
ge::graphStatus RunTiling(const std::vector<int64_t>& dyDims, const std::vector<int64_t>& paramDims)
{
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                              "Intrinsic_fix_pipe_l0c2out": false,
                              "Intrinsic_data_move_l12ub": true,
                              "Intrinsic_data_move_l0c2ub": true,
                              "Intrinsic_data_move_out2l1_nd2nz": false,
                              "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                              "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                              "CORE_NUM": 64}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // tiling 从 CompileInfo 取平台参数，按 Ascend950 规格填充。
    optiling::InTrainingUpdateGradCompileInfo compile_info;
    compile_info.coreNum = 64U;
    compile_info.ubSize = 245760U;
    compile_info.vectorLength = 256U; // VRegSize，单位字节
    compile_info.ubBlockSize = 32U;

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    if (op_impl == nullptr || op_impl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tiling_func = op_impl->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    if (param == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    gert::StorageShape dy = MakeShape(dyDims);
    gert::StorageShape x = MakeShape(dyDims);
    gert::StorageShape var = MakeShape(paramDims);
    gert::StorageShape mean = MakeShape(paramDims);
    gert::StorageShape resGamma = MakeShape(paramDims);
    gert::StorageShape resBeta = MakeShape(paramDims);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(kOpType) // 走 TilingRegistry 模板注册表，必须设置 op type 才能命中模板
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&dy, &x, &var, &mean})
                      .OutputShapes({&resGamma, &resBeta})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context == nullptr || tiling_context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    return tiling_func(tiling_context);
}
} // namespace

class INTrainingUpdateGradErrorCodeTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "INTrainingUpdateGrad ErrorCode Tiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "INTrainingUpdateGrad ErrorCode Tiling TearDown" << std::endl; }
};

// 1. C0=128 > 64 lane 上界：tiling 在包络门拒绝，EZ0009 / param_name=dy
TEST_F(INTrainingUpdateGradErrorCodeTiling, tiling_reject_c0_over_64_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunTiling({2, 4, 2, 8, 8, 128}, {2, 1, 2, 1, 1, 128}), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dy");
    EXPECT_EQ(ErrorValue(error, "incorrect_shape"), "[2, 4, 2, 8, 8, 128]");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "must not exceed 64"));
}

// 2. H*W 超 uint32（65536*65536）：包络乘积门拒绝，EZ0009 / param_name=dy / reason 含 H*W
TEST_F(INTrainingUpdateGradErrorCodeTiling, tiling_reject_hw_product_overflow_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunTiling({1, 1, 1, 65536, 65536, 16}, {1, 1, 1, 1, 1, 16}), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dy");
    EXPECT_EQ(ErrorValue(error, "incorrect_shape"), "[1, 1, 1, 65536, 65536, 16]");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "H*W"));
}
