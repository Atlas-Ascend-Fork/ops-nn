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
#include <vector>

#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "op_host/tiling_templates_registry.h"
#include "../../../op_host/op_tiling/fused_weight_quant_mat_mul_tiling_common.h"
#include "../../../op_host/op_tiling/arch35/fused_weight_quant_mat_mul_asw_tiling.h"
#include "../../../op_host/op_tiling/arch35/fused_weight_quant_mat_mul_iterbatch_tiling.h"
#include "../../../op_kernel/arch35/fused_weight_quant_mat_mul_tilingkey.h"
#include "test_cube_util.h"

using namespace std;
using namespace ge;
using namespace ut_util;
using namespace optiling;

namespace {
constexpr size_t kTilingDataSize = 8192;
constexpr size_t kWorkspaceSize = 4096;
constexpr uint64_t kFusedOpTypeShift = 36UL;
constexpr uint64_t kFusedOpTypeMask = 0x3UL;
constexpr uint64_t kIsBiasInt32Shift = 34UL;
constexpr uint64_t kIsBiasInt32Mask = 0x1UL;
constexpr int64_t kUnsupportedGroupSize = 128;

// 与底座 UT 相同的平台信息（Intrinsic_mmad 含 s8s4 → supportMmadS8S4 = true）
const std::string kS8S4CompileInfo = R"({
     "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true,
                       "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l12bt": true,
                       "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                       "Intrinsic_fix_pipe_pre_conv_cast": true, "Intrinsic_mmad": true,
                       "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 1048576,
                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 16,
                       "cube_core_cnt": 16, "vector_core_cnt": 16, "core_type_list": "CubeCore,VectorCore",
                       "lut_type": "MTE2_QTABLE"}
                      })";
// 无 s8s4 特征的平台：TilingFunc 应直接报不支持
const std::string kNonS8S4CompileInfo = R"({
     "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true,
                       "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l12bt": true,
                       "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                       "Intrinsic_fix_pipe_pre_conv_cast": true,
                       "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 1048576,
                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 16,
                       "cube_core_cnt": 16, "vector_core_cnt": 16, "core_type_list": "CubeCore,VectorCore"}
                      })";

struct FusedTilingParam {
    std::string caseName;
    int64_t m = 64;
    int64_t k = 64;
    int64_t n = 64;
    bool hasBias = false;
    ge::DataType biasDtype = ge::DT_FLOAT16;
    bool transA = false;
    bool transB = false;
    // 非法场景参数
    bool expectSuccess = true;
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType weightDtype = ge::DT_INT8;
    ge::DataType scaleDtype = ge::DT_UINT64;
    ge::DataType yDtype = ge::DT_FLOAT16;
    int64_t groupSize = 0;
    bool perChannelScale = true; // false → scale shape (1,) per-tensor
    vector<int64_t> scaleShape;
    bool hasAntiQuantOffset = false;
    bool hasQuantScale = false;
    std::string fusedOpType = "relu";
    std::string compileInfo = kS8S4CompileInfo;
};

uint64_t ExpectedAswKey(bool transA, bool transB, bool hasBias, bool isBiasInt32 = false)
{
    // 与 kernel 侧 ASCENDC_TPL_SEL 一致的编码：MMAD_S8S4 + FIXPIPE_ANTIQUANT + ASW + PerChannel +
    // 无量化输入（quantType 取默认 PER_TENSOR）+ 无 offset + bias{有无 × dtype} + weight ND + relu
    return GET_TPL_TILING_KEY(FWQM_SOC_SUPPORT_MMAD_S8S4, FWQM_DEFAULT, FWQM_DEFAULT, FWQM_ALGO_FIXPIPE_ANTIQUANT,
                              FWQM_SUB_ALGO_ASW, FWQM_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2, 0, transA, transB,
                              FWQM_ANTIQUANT_TYPE_PER_CHANNEL, FWQM_QUANT_TYPE_PER_TENSOR, false, hasBias, isBiasInt32,
                              false, FWQM_FUSED_OPTYPE_RELU);
}

struct FusedTilingResult {
    ge::graphStatus ret = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0UL;
};

// 注意：TilingContextFaker 为栈上对象，出函数即销毁；
// tilingKey 等信息须在函数内读取，不得外泄 TilingContext 指针
// directAsw = true 时跳过 TilingFunc 分发，直接以 FusedWeightQuantMatMulASWTiling 驱动 DoTiling
FusedTilingResult RunFusedTiling(const FusedTilingParam& p, bool directAsw = false)
{
    gert::StorageShape xShape;
    gert::StorageShape weightShape;
    gert::StorageShape antiQuantScaleShape;
    gert::StorageShape antiQuantOffsetShape;
    gert::StorageShape quantScaleShape;
    gert::StorageShape biasShape;
    gert::StorageShape outputShape({p.m, p.n}, {p.m, p.n});

    if (p.transA) {
        xShape.MutableStorageShape() = gert::Shape({p.k, p.m});
        xShape.MutableOriginShape() = gert::Shape({p.k, p.m});
    } else {
        xShape.MutableStorageShape() = gert::Shape({p.m, p.k});
        xShape.MutableOriginShape() = gert::Shape({p.m, p.k});
    }
    if (p.transB) {
        weightShape.MutableStorageShape() = gert::Shape({p.n, p.k});
        weightShape.MutableOriginShape() = gert::Shape({p.n, p.k});
    } else {
        weightShape.MutableStorageShape() = gert::Shape({p.k, p.n});
        weightShape.MutableOriginShape() = gert::Shape({p.k, p.n});
    }
    if (!p.scaleShape.empty()) {
        gert::Shape scaleShape;
        for (auto dim : p.scaleShape) {
            scaleShape.AppendDim(dim);
        }
        antiQuantScaleShape.MutableStorageShape() = scaleShape;
        antiQuantScaleShape.MutableOriginShape() = scaleShape;
    } else if (p.perChannelScale) {
        antiQuantScaleShape.MutableStorageShape() = gert::Shape({p.n});
        antiQuantScaleShape.MutableOriginShape() = gert::Shape({p.n});
    } else {
        antiQuantScaleShape.MutableStorageShape() = gert::Shape({1});
        antiQuantScaleShape.MutableOriginShape() = gert::Shape({1});
    }
    antiQuantOffsetShape.MutableStorageShape() = gert::Shape({p.n});
    quantScaleShape.MutableStorageShape() = gert::Shape({1});
    biasShape.MutableStorageShape() = gert::Shape({p.n});

    map<std::string, std::string> socInfos;
    map<std::string, std::string> aicoreSpec;
    map<std::string, std::string> intrinsics;
    GetPlatFormInfos(p.compileInfo.c_str(), socInfos, aicoreSpec, intrinsics);
    aicoreSpec["cube_freq"] = "1800";

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    WeightQuantBatchMatmulV2CompileInfo compileInfo;
    std::string compileInfoStr = p.compileInfo;

    auto kernelHold = gert::KernelRunContextFaker()
                          .KernelIONum(2, 1)
                          .Inputs({compileInfoStr.data(), reinterpret_cast<void*>(&platformInfo)})
                          .Outputs({&compileInfo})
                          .Build();

    FusedTilingResult result;
    std::string opType("FusedWeightQuantMatMul");
    if (gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()) == nullptr) {
        return result;
    }
    auto rawTilingData = gert::TilingData::CreateCap(kTilingDataSize);
    if (rawTilingData == nullptr) {
        return result;
    }
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(kWorkspaceSize);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &weightShape, &antiQuantScaleShape,
                                    p.hasAntiQuantOffset ? &antiQuantOffsetShape : nullptr,
                                    p.hasQuantScale ? &quantScaleShape : nullptr, nullptr,
                                    p.hasBias ? &biasShape : nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(&platformInfo)
                      .NodeInputTd(0, p.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, p.weightDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, p.scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, p.biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, p.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(p.transA)},
                                  {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(p.transB)},
                                  {"antiquant_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(p.groupSize)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"fused_op_type", Ops::NN::AnyValue::CreateFrom<std::string>(p.fusedOpType)}})
                      .TilingData(rawTilingData.get())
                      .Workspace(workspace)
                      .SetOpType(opType)
                      .Build();

    map<std::string, std::string> socVersionInfos;
    socVersionInfos.insert(make_pair("Short_SoC_version", "RESERVED_VERSION"));

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    if (tilingParseFunc == nullptr ||
        tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return result;
    }
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    if (tilingFunc == nullptr) {
        return result;
    }
    if (directAsw) {
        auto aswTiling = optiling::fused_weight_quant_mat_mul::FusedWeightQuantMatMulASWTiling(tilingContext);
        result.ret = aswTiling.DoTiling();
    } else {
        result.ret = tilingFunc(tilingContext);
    }
    result.tilingKey = tilingContext->GetTilingKey();
    return result;
}
} // namespace

class TestFusedWeightQuantMatMulTiling : public testing::TestWithParam<FusedTilingParam> {};

TEST_P(TestFusedWeightQuantMatMulTiling, generalTest)
{
    auto param = GetParam();
    auto result = RunFusedTiling(param);
    if (!param.expectSuccess) {
        ASSERT_EQ(result.ret, ge::GRAPH_FAILED);
        return;
    }
    ASSERT_EQ(result.ret, ge::GRAPH_SUCCESS);
    auto tilingKey = result.tilingKey;
    // FUSED_OPTYPE 位（第 36 位，2bit 字段）编码 RELU(1)
    ASSERT_EQ((tilingKey >> kFusedOpTypeShift) & kFusedOpTypeMask, static_cast<uint64_t>(FWQM_FUSED_OPTYPE_RELU))
        << "tiling key: " << tilingKey;
    // bias dtype 位（第 34 位，IS_BIAS_INT32）：bias 为 INT32 时置 1（单 binary 双 dtype 由该位承载）
    bool isBiasInt32 = param.hasBias && param.biasDtype == ge::DT_INT32;
    ASSERT_EQ((tilingKey >> kIsBiasInt32Shift) & kIsBiasInt32Mask, static_cast<uint64_t>(isBiasInt32))
        << "tiling key: " << tilingKey;
    if (!param.transA && !param.transB) {
        // 2D 无 batch 场景由 ASW 模板承接，校验 key 与 ASCENDC_TPL_SEL 声明一致（host/kernel 编码一致性）
        ASSERT_EQ(tilingKey, ExpectedAswKey(param.transA, param.transB, param.hasBias, isBiasInt32))
            << "tiling key: " << tilingKey;
    }
}

// 正向：A16W8 PerChannel × trans × bias{fp16,int32,无} × y{fp16,int8}
static vector<FusedTilingParam> GenPositiveCases()
{
    vector<FusedTilingParam> cases;
    cases.push_back({"basic_64_64_64_no_bias", 64, 64, 64, false});
    cases.push_back({"basic_64_64_64_bias_fp16", 64, 64, 64, true});
    cases.push_back({"basic_64_64_64_bias_int32", 64, 64, 64, true, ge::DT_INT32});
    cases.push_back({"trans_a_64_64_64", 64, 64, 64, false, ge::DT_FLOAT16, true, false});
    cases.push_back({"trans_b_64_64_64", 64, 64, 64, false, ge::DT_FLOAT16, false, true});
    cases.push_back({"trans_ab_bias_128_128_128", 128, 128, 128, true, ge::DT_FLOAT16, true, true});
    cases.push_back({"llm_shape_3072_2048_4096", 3072, 2048, 4096, false});
    cases.push_back({"llm_shape_bias_3072_2048_4096", 3072, 2048, 4096, true, ge::DT_FLOAT16});
    // y INT8（输出量化，合并编码契约）：key 值域与 y FP16 相同（y dtype 不参与 tiling key 编码，
    // VREQ8/REQ8 由 def dtype 组合层的 DTYPE_Y 承载）
    FusedTilingParam yInt8Basic;
    yInt8Basic.caseName = "y_int8_output_basic";
    yInt8Basic.yDtype = ge::DT_INT8;
    cases.push_back(yInt8Basic);
    FusedTilingParam yInt8Bias;
    yInt8Bias.caseName = "y_int8_output_bias_int32";
    yInt8Bias.hasBias = true;
    yInt8Bias.biasDtype = ge::DT_INT32;
    yInt8Bias.yDtype = ge::DT_INT8;
    cases.push_back(yInt8Bias);
    FusedTilingParam scaleRow;
    scaleRow.caseName = "per_channel_scale_1_n";
    scaleRow.scaleShape = {1, 64};
    cases.push_back(scaleRow);
    FusedTilingParam scaleColumn;
    scaleColumn.caseName = "per_channel_scale_n_1_trans_b";
    scaleColumn.transB = true;
    scaleColumn.scaleShape = {64, 1};
    cases.push_back(scaleColumn);
    return cases;
}

// 非法：checker 收紧项全部拒绝
static vector<FusedTilingParam> GenNegativeCases()
{
    vector<FusedTilingParam> cases;
    FusedTilingParam base; // 默认合法
    base.expectSuccess = false;

    FusedTilingParam bf16 = base;
    bf16.caseName = "neg_x_bf16";
    bf16.xDtype = ge::DT_BF16;
    cases.push_back(bf16);

    FusedTilingParam w4 = base;
    w4.caseName = "neg_weight_int4";
    w4.weightDtype = ge::DT_INT4;
    cases.push_back(w4);

    FusedTilingParam fp32Scale = base;
    fp32Scale.caseName = "neg_scale_fp32";
    fp32Scale.scaleDtype = ge::DT_FLOAT;
    cases.push_back(fp32Scale);

    FusedTilingParam yBf16 = base;
    yBf16.caseName = "neg_y_bf16";
    yBf16.yDtype = ge::DT_BF16;
    cases.push_back(yBf16);

    FusedTilingParam biasFp32 = base;
    biasFp32.caseName = "neg_bias_fp32";
    biasFp32.hasBias = true;
    biasFp32.biasDtype = ge::DT_FLOAT;
    cases.push_back(biasFp32);

    FusedTilingParam perGroup = base;
    perGroup.caseName = "neg_group_size_128";
    perGroup.groupSize = kUnsupportedGroupSize;
    cases.push_back(perGroup);

    FusedTilingParam perTensor = base;
    perTensor.caseName = "neg_per_tensor_scale";
    perTensor.perChannelScale = false;
    cases.push_back(perTensor);

    FusedTilingParam invalidScaleLayout = base;
    invalidScaleLayout.caseName = "neg_per_channel_scale_invalid_2d_layout";
    invalidScaleLayout.scaleShape = {2, 32};
    cases.push_back(invalidScaleLayout);

    FusedTilingParam offset = base;
    offset.caseName = "neg_antiquant_offset";
    offset.hasAntiQuantOffset = true;
    cases.push_back(offset);

    FusedTilingParam quant = base;
    quant.caseName = "neg_quant_scale";
    quant.hasQuantScale = true;
    cases.push_back(quant);

    FusedTilingParam gelu = base;
    gelu.caseName = "neg_fused_op_type_gelu";
    gelu.fusedOpType = "gelu";
    cases.push_back(gelu);

    FusedTilingParam nonS8S4 = base;
    nonS8S4.caseName = "neg_platform_without_s8s4";
    nonS8S4.compileInfo = kNonS8S4CompileInfo;
    cases.push_back(nonS8S4);
    return cases;
}

INSTANTIATE_TEST_CASE_P(FusedWeightQuantMatMulPositive, TestFusedWeightQuantMatMulTiling,
                        testing::ValuesIn(GenPositiveCases()));
INSTANTIATE_TEST_CASE_P(FusedWeightQuantMatMulNegative, TestFusedWeightQuantMatMulTiling,
                        testing::ValuesIn(GenNegativeCases()));

// 直接驱动 ASW tiling 子类，覆盖 GetFusedShapeAttrsInfo（融合属性解析）与 GetTilingKey：
// 在 faker 存活作用域内构造子类并 DoTiling，校验其 GetTilingKey 编码
TEST(TestFusedWeightQuantMatMulTilingDirect, DirectDoTiling)
{
    // 通用框架路径（TilingFunc -> 模板注册表）已由 generalTest 覆盖；
    // 此处复用同一 faker 流程，构造后直接以 ASW 子类 DoTiling 并断言 key
    FusedTilingParam p;
    p.m = 64;
    p.k = 64;
    p.n = 64;
    auto result = RunFusedTiling(p, true);
    ASSERT_EQ(result.ret, ge::GRAPH_SUCCESS);
    ASSERT_EQ(result.tilingKey, ExpectedAswKey(false, false, false)) << "tiling key: " << result.tilingKey;
}
