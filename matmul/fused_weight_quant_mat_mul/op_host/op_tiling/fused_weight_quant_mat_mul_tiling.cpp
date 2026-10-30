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
 * \file fused_weight_quant_mat_mul_tiling.cpp
 * \brief FusedWeightQuantMatMul tiling 主入口 + TilingParse + 模板注册。
 *        主 TilingFunc 对齐底座分发（supportMmadS8S4 时 checker + {ITERBATCH, ASW} 模板），
 *        checker 换用融合版（一期收紧 + 属性布局适配），对齐 fused_quant_mat_mul 的薄封装模式。
 */

#include <cstring>
#include "fused_weight_quant_mat_mul_tiling_common.h"
#include "fused_weight_quant_mat_mul_checker.h"
#include "op_tiling/weight_quant_batch_matmul_v2_tiling.h"
#include "register/op_impl_registry.h"
#include "op_cache_tiling.h"
#include "platform/platform_infos_def.h"
#include "../../op_kernel/arch35/fused_weight_quant_mat_mul_tilingkey.h"
#include "matmul/fused_weight_quant_mat_mul/common/fused_weight_quant_mat_mul_constants.h"

using Ops::NN::TilingPrepareForOpCache;
using Ops::NN::Optiling::TilingRegistry;

namespace optiling {

// compile info 直接别名（薄封装模式，对齐 fused_quant_mat_mul）
using FusedWeightQuantMatMulCompileInfo = WeightQuantBatchMatmulV2CompileInfo;

namespace fused_weight_quant_mat_mul {

ge::graphStatus GetFusedShapeAttrsInfo(gert::TilingContext* context,
                                       std::unique_ptr<WeightQuantBatchMatmulInfo>& matmulInfoPtr, const char*& opName,
                                       uint64_t& fusedOpType, uint32_t& shiftValue)
{
    try {
        matmulInfoPtr = std::make_unique<WeightQuantBatchMatmulInfo>();
    } catch (const std::bad_alloc& e) {
        OP_LOGE(context->GetNodeName(), "failed to instantiate matmul info");
        return ge::GRAPH_FAILED;
    }

    // 属性解析按本算子布局：transpose_x(0)/transpose_weight(1)/antiquant_group_size(2)/dtype(3)/
    // fused_op_type(4)。底座 GetAttrs 在索引 4 读 int64 的 inner_precise，与本算子 String 属性布局
    // 不兼容（GetAttrPointer 为裸 reinterpret_cast），不可复用；inner_precise 不存在，保持默认 0
    // （融合算子固定高精度模式）。
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto transposeX = attrs->GetAttrPointer<bool>(TRANSPOSE_X_ATTR_INDEX);
    auto transposeWeight = attrs->GetAttrPointer<bool>(TRANSPOSE_WEIGHT_ATTR_INDEX);
    const int64_t* groupSizePtr = nullptr;
    if (attrs->GetAttrNum() > ANTIQUANT_GROUP_SIZE_ATTR_INDEX) {
        groupSizePtr = attrs->GetAttrPointer<int64_t>(ANTIQUANT_GROUP_SIZE_ATTR_INDEX);
    }
    if (groupSizePtr != nullptr) {
        matmulInfoPtr->groupSize = static_cast<uint64_t>(*groupSizePtr);
    }
    const char* fusedOpTypeStr = nullptr;
    if (attrs->GetAttrNum() > FUSED_OP_TYPE_ATTR_INDEX) {
        fusedOpTypeStr = attrs->GetAttrPointer<char>(FUSED_OP_TYPE_ATTR_INDEX);
    }
    matmulInfoPtr->transA = transposeX != nullptr && *transposeX;
    matmulInfoPtr->transB = transposeWeight != nullptr && *transposeWeight;
    // 一期仅 relu；checker 已校验，此处未知值按 NONE 处理并由 GetTilingKey 编码拒绝实例化
    fusedOpType = (fusedOpTypeStr != nullptr &&
                   strcmp(fusedOpTypeStr, ops::fused_weight_quant_mat_mul::FUSED_OP_TYPE_RELU) == 0) ?
                      static_cast<uint64_t>(FusedOpType::RELU) :
                      static_cast<uint64_t>(FusedOpType::NONE);
    // shiftValue 固定默认值（底座 ORI_SHIFT_VALUE=13，未注册 shift_value 私有属性）
    shiftValue = ORI_SHIFT_VALUE;

    // dtype/inputs 解析复用底座自由函数（输入索引 0-6 与底座严格对齐，直接可用）
    GetDtype(*matmulInfoPtr, context);
    GetInputs(*matmulInfoPtr, context);
    GetBatchInfo(*matmulInfoPtr, context);
    opName = context->GetNodeName();
    if (opName == nullptr || opName[0] == '\0') {
        opName = "FusedWeightQuantMatMul";
    }

    // n == 1 时 per-channel scale (n,) = (1,) 与 per-tensor 标量形状不可区分
    // 底座 GetInputs 按 shapeSize == 1 判为 PER_TENSOR，对于 FusedWeightQuantMatmul，判定为 per-channel
    if (matmulInfoPtr->antiQuantType == QuantType::PER_TENSOR && matmulInfoPtr->nSize == 1UL) {
        matmulInfoPtr->antiQuantType = QuantType::PER_CHANNEL;
    }

    OP_LOGD(opName,
            "fused input params: MKN[%lu, %lu, %lu], transA[%s], transB[%s], bias[%s], "
            "group size[%lu], fusedOpType[%lu]",
            matmulInfoPtr->mSize, matmulInfoPtr->kSize, matmulInfoPtr->nSize, matmulInfoPtr->transA ? "true" : "false",
            matmulInfoPtr->transB ? "true" : "false", matmulInfoPtr->hasBias ? "true" : "false",
            matmulInfoPtr->groupSize, fusedOpType);
    return ge::GRAPH_SUCCESS;
}

} // namespace fused_weight_quant_mat_mul

static ge::graphStatus FusedWeightQuantMatMulTilingFunc(gert::TilingContext* context)
{
    platform_ascendc::SocVersion socVersion;
    bool supportMmadS8S4 = false;
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "FusedWeightQuantMatMul", "tilingContext is null");
    auto compileInfoPtr = reinterpret_cast<const FusedWeightQuantMatMulCompileInfo*>(context->GetCompileInfo());
    if (compileInfoPtr == nullptr) {
        auto platformInfoPtr = context->GetPlatformInfo();
        OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "platformInfoPtr is null");
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        socVersion = ascendcPlatform.GetSocVersion();
        std::string mmad;
        bool res = platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", mmad);
        supportMmadS8S4 = res && mmad.find("s8s4") != std::string::npos;
    } else {
        socVersion = compileInfoPtr->socVersion;
        supportMmadS8S4 = compileInfoPtr->supportMmadS8S4;
    }
    // 算子仅在 s8s4 平台存在二进制
    OP_TILING_CHECK(!supportMmadS8S4,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                                                    "FusedWeightQuantMatMul only support platform with mmad s8s4"),
                    return ge::GRAPH_FAILED);
    fused_weight_quant_mat_mul::FusedWeightQuantMatMulChecker checker(context);
    OP_TILING_CHECK(checker.Check() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "FusedWeightQuantMatMul para is illegal"),
                    return ge::GRAPH_FAILED);
    std::vector<int32_t> registerList = {FWQM_ITERBATCH_PRIORITY, FWQM_ASW_PRIORITY};
    return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
}

static ge::graphStatus TilingParseForFusedWeightQuantMatMul(gert::TilingParseContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "FusedWeightQuantMatMul", "tilingParseContext is null");
    auto platformInfoPtr = context->GetPlatformInfo();
    // 在tilingParse获取不到GetPlatformInfo时，返回成功。会在之后的InitCompileInfo阶段设置compileInfo信息。
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_SUCCESS, context->GetNodeName(), "platformInfoPtr is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<FusedWeightQuantMatMulCompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "compileInfoPtr is null");

    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    OP_LOGE_IF(compileInfoPtr->aicNum == 0, ge::GRAPH_FAILED, context->GetNodeName(), "aicNum is 0");
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    compileInfoPtr->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    std::string mmad;
    bool res = platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", mmad);
    compileInfoPtr->supportMmadS8S4 = res && mmad.find("s8s4") != std::string::npos;

    TilingPrepareForOpCache(context);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedWeightQuantMatMul)
    .Tiling(FusedWeightQuantMatMulTilingFunc)
    .TilingParse<FusedWeightQuantMatMulCompileInfo>(TilingParseForFusedWeightQuantMatMul); // 向框架注册入口函数
} // namespace optiling
