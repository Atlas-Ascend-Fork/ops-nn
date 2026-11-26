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
 * \file legacy_fusion_pass.cc
 * \brief 把桩头编入 libopgraph_nn.so，并在满足条件时登记 canndev 遗留（日落）fe 融合 pass 名字。
 */

// 仅当编译 soc 为 ascend950 时，当前 cpp 中的打桩注册内容才编入
#if defined(OP_GRAPH_NN_ASCEND950)

#include "version/ge-compiler_version.h"
#include "fe_fusion_pass_registry_stub.h"

// 编译期保护：构建环境 ge_compiler >= 9.0.0 才编入；宏未定义的旧版本头按不满足处理、静默不编入
// （与 cube_common.cc 既有写法一致；运行时检查在 STUB_REG_PASS 宏内）
#if defined(GE_COMPILER_VERSION_NUM) && GE_COMPILER_VERSION_NUM >= 90000000

// 登记 canndev 旧融合 pass 名字，名字/类型/attr 照抄 canndev 的 REG_PASS/REGISTER_PASS
// attr：FORBIDDEN_CLOSE=0x01、SINGLE_SCENE_OPEN=0x04；3 参 REGISTER_PASS 默认 0
// 只登记 nn 仓实现 Stage 为 kCompatibleInherited 且 canndev 有同名注册的 pass；下列两类不登记：
//   1. canndev 有同名注册但 nn 实现 Stage 为 kAfterInferShape(7 个)/kAfterBuiltinFusionPass(12 个)，
//      不在 kCompatibleInherited 表里，登记后 fe 查不到实现会报 "pattern fusion is nullptr"
//   2. canndev 无同名注册的 9 个 nn 新增 pass（如 SoftmaxGradExtFusionPass，canndev 注册名为
//      SoftmaxGradExtFusion），无 fe 表名字需要接管
// 门控对齐原则（逐 pass 与 nn 实现的 REG_FUSION_PASS 注册条件一致）：
//   - 编译期门槛 = 实现的 #if GE_COMPILER_VERSION_NUM 编译守卫，按 9.0/9.1/9.2/9.3 分块编入；
//     某块未编入时该组实现同样未注册，名字不会出现"在表无实现"的错配
//   - 运行期门槛（STUB_REG_PASS 第 4 参）= 实现的运行期 Stage 判定阈值：实现为
//     IsTargetVersion()/IsGeVersionSupported()/Get*Stage() 判定型的（不足时降级 kAfterInferShape），
//     桩取同一阈值；实现为固定 kCompatibleInherited 的，桩取 90000000
//   - 构建形态门槛：单算子选择性构建（build.sh --ops=，CMake 变量 ASCEND_OP_NAME 有效）时本文件
//     不参与编译（common/CMakeLists.txt 门控 add_subdirectory）——pattern 实现随算子模块被 --ops 裁剪，
//     名字与实现分离会让 fe 图优化报 E29999 pattern fusion is nullptr（issue #9971）；
//     单算子包的内置 pass 注册表由全量部署的基础环境提供
//   - MatmulQuantToFixpipeFusionPass 实现额外依赖 OriginDataType 公共 API（同版本号的不同 CANN 包
//     API 可用性可能不同），桩单独复刻弱符号探测
// 白名单说明（对照 opbase 仓 scripts/fusion_config/fusion_pass/config/support_fusion_pass.json，
// 实际生效以部署环境 OPP 白名单为准）：67 个中 64 个在 3510(ascend950) 名单内、正常生效；
// 其余 3 个（仅 5102 名单的 Conv2DTransposeWeightTransFusionPass，及名单暂未收录的
// QuantBatchMatmulV3TransposeLimitFusionPass、WeightQuantBatchMatmulV2TransposeNZFusionPass）
// fe 会 force close 跳过，待白名单侧补录后生效，故保留登记

// ===== 编译门槛 9.0 组（22 个）：实现无编译守卫或守卫为 90000000 =====
STUB_REG_PASS("AInplaceAddFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("AInplaceSubFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("AInplaceUpdateFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("BatchMatMul2MulFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("Conv2DTransposeWeightTransFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("EinsumPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("GatherToGatherV2FusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("GemmToMatmulFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("IndexByTensorStaticFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("MatmulReshapeTransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, 0, 90000000);
STUB_REG_PASS("MatmulToGemmOpFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("MatMulToMatmulV3FusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("MatMulUnsqueezeSqueezeFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("MaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("MaxPoolGradWithArgmaxV3FusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("MaxPoolWithArgmaxV3FusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("QuantBatchMatmulV4ToV3FusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("TensorScatterAddFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("TensorScatterUpdateFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("UniqueSortFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("WhereFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("ZZMatMulTOQBMMV3FusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, 0x04, 90000000);

// ===== REG_DECOMPOSE_PASS（分解改写类，11 个）：V1→V2 算子承接改写，实现无守卫或守卫 90000000，
// 固定 kCompatibleInherited（运行门槛 9.0），与融合 pass 同体系注册到 kCompat 分区、同等待遇桥接 =====
STUB_REG_PASS("ADeformableConv2dPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("Conv2DBackpropFilterToV3FusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("Conv2DBackpropInputToV2FusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("Conv2DTransposeToV2FusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("DepthwiseDfFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("DepthwiseDwMulFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("Conv3DBackpropFilterToV2FusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("Conv3DBackpropInputToV2FusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("Conv3DTransposeToV2FusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("Conv3dToConv3dV2FusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("DepthwiseToConv2dFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);

#if GE_COMPILER_VERSION_NUM >= 90100000
// ===== 编译门槛 9.1 组（23 个）：实现编译守卫为 90100000 =====
// 固定 kCompatibleInherited（运行期不降级，运行门槛 9.0）
STUB_REG_PASS("AConv2dMulFusion", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("ADepthwiseFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("ASplitConv2dConcatPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("Conv2DFixPipeToExtendConv2DFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("Conv2DSqueezeBiasaddFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("Conv2dToConv2dV2FusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("Conv3DDequantToQuantConv3DFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("PadConv2dFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("PadDepthwiseConv2dFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("QuantUnsqueezeConv2DFusionPass", fe::BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS, 0x01, 90000000);
// 运行判定型（实现 IsTargetVersion()/IsGeVersionSupported() 阈值 90100000，不足降级 kAfterInferShape）
STUB_REG_PASS("AntiQuantMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90100000);
STUB_REG_PASS("BatchMatMul2TransposeBatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90100000);
STUB_REG_PASS("BatchMatMulTransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90100000);
STUB_REG_PASS("FusedMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90100000);
STUB_REG_PASS("MatMulBiasAddOpenFusionPass", fe::BUILT_IN_GRAPH_PASS, 0, 90100000);
STUB_REG_PASS("MatMulReshapeBiasAddFusionPass", fe::BUILT_IN_GRAPH_PASS, 0, 90100000);
STUB_REG_PASS("QuantBatchMatmulInplaceAddTransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90100000);
STUB_REG_PASS("QuantBatchMatmulV3TransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90100000);
STUB_REG_PASS("QuantBatchMatmulV3TransposeLimitFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90100000);
STUB_REG_PASS("QuantBatchMatmulV4TransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90100000);
STUB_REG_PASS("WeightQuantBatchMatmulV2TransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x04, 90100000);
STUB_REG_PASS("WeightQuantBatchMatmulV2TransposeNZFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90100000);
STUB_REG_PASS("Conv2DTransposeFixpipeFusionPass", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
#endif // GE_COMPILER_VERSION_NUM >= 90100000

#if GE_COMPILER_VERSION_NUM >= 90200000
// ===== 编译门槛 9.2 组（1 个）：实现编译守卫 90200000，运行期版本阈值 90200000 且依赖
// OriginDataType 公共 API（同版本号的不同 CANN 包 API 可用性可能不同），桩复刻同一弱符号探测 =====
static ::fe::FusionPassRegistrar* stub_reg_matmul_quant_to_fixpipe __attribute__((
    unused)) = (::fe::StubIsGeCompilerVersionAtLeast(90200000) && ::fe::StubFixpipePublicApiAvailable()) ?
                   new ::fe::FusionPassRegistrar(
                       fe::BUILT_IN_GRAPH_PASS, "MatmulQuantToFixpipeFusionPass",
                       []() -> ::fe::GraphPass* { return nullptr; }, 0x01) :
                   nullptr;
#endif // GE_COMPILER_VERSION_NUM >= 90200000

#if GE_COMPILER_VERSION_NUM >= 90300000
// ===== 编译门槛 9.3 组（10 个）：实现编译守卫 90300000，固定 kCompatibleInherited（运行门槛 9.0）=====
// fixpipe 3 个：canndev 注册类型 FixPipeAbilityProcessPass/FIXPIPEAPREQUANTFUSIONPASS/FIXPIPEARELUQUANTFusionPass
// 为 GetGraphFusionPassType()（运行时按 npu_arch，3510 不在 SecondStage 集合 → BUILT_IN_GRAPH_PASS，
// 桩在 ascend950 门控下照抄 A5 取值）
// attr NEED_SORT=0x02|SINGLE_SCENE_OPEN|FORBIDDEN_CLOSE=0x07（位值与 GE graph_fusion_pass_base.h 一致）
STUB_REG_PASS("AABiasaddConvFusion", fe::BUILT_IN_GRAPH_PASS, 0x01, 90000000);
STUB_REG_PASS("BatchMatmulV2QuantProcessFusionPass", fe::BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("Conv2DQuantProcessFusionPass", fe::BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("Conv2DTQuantProcessFusionPass", fe::BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("Conv3DQuantProcessFusionPass", fe::BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("DWConv2DQuantProcessFusionPass", fe::BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("FixPipeAbilityProcessPass", fe::BUILT_IN_GRAPH_PASS, 0x07, 90000000);
STUB_REG_PASS("FIXPIPEAPREQUANTFUSIONPASS", fe::BUILT_IN_GRAPH_PASS, 0x04, 90000000);
STUB_REG_PASS("FIXPIPEARELUQUANTFUSIONPASS", fe::BUILT_IN_GRAPH_PASS, 0, 90000000);
STUB_REG_PASS("MatmulV2QuantProcessFusionPass", fe::BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS, 0x04, 90000000);
#endif // GE_COMPILER_VERSION_NUM >= 90300000

#endif // defined(GE_COMPILER_VERSION_NUM) && GE_COMPILER_VERSION_NUM >= 90000000
#endif // OP_GRAPH_NN_ASCEND950
