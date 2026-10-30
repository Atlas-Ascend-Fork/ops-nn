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
 * \file fused_weight_quant_mat_mul_iterbatch_tiling.cpp
 * \brief FusedWeightQuantMatMul IterBatch tiling 实现。
 *        SUB_ALGORITHM 与底座一致按 enableBatchOut 取 ITERATE_BATCH / ITERATE_BATCH_NO_BATCH_OUT。
 */

#include "fused_weight_quant_mat_mul_iterbatch_tiling.h"
#include "matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling.h"
#include "matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "../../../op_kernel/arch35/fused_weight_quant_mat_mul_tilingkey.h"

using namespace platform_ascendc;

namespace optiling {
namespace fused_weight_quant_mat_mul {

namespace {
constexpr uint64_t CUBE_BLOCK = 16UL;
} // namespace

ge::graphStatus FusedWeightQuantMatMulIterbatchTiling::GetShapeAttrsInfo()
{
    auto ret = GetFusedShapeAttrsInfo(context_, matmulInfoPtr_, opName_, fusedOpType_, shiftValue_);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ConfigureReuseScenarios();
    return ge::GRAPH_SUCCESS;
}

uint64_t FusedWeightQuantMatMulIterbatchTiling::GetTilingKey() const
{
    constexpr uint64_t socVersionType = FWQM_SOC_SUPPORT_MMAD_S8S4;
    constexpr uint64_t subSocVersionType = FWQM_DEFAULT;
    constexpr uint64_t antiquantScenario = FWQM_DEFAULT;
    constexpr uint64_t algorithm = FWQM_ALGO_FIXPIPE_ANTIQUANT;
    // 开启NBatchOut需满足如下条件（与底座一致）：
    // 1. baseM >= singleCoreM && baseN >= singleCoreN
    // 2. N/16向上取整应为偶数
    bool enableBatchOut = basicTiling_.baseM >= basicTiling_.singleCoreM &&
                          basicTiling_.baseN >= basicTiling_.singleCoreN &&
                          static_cast<uint64_t>(ops::CeilDiv(matmulInfoPtr_->nSize, CUBE_BLOCK)) % 2 == 0;
    uint64_t subAlgorithm = enableBatchOut ?
                                static_cast<uint64_t>(OptimizationAlgorithmSubCategory::ITERATE_BATCH) :
                                static_cast<uint64_t>(OptimizationAlgorithmSubCategory::ITERATE_BATCH_NO_BATCH_OUT);
    constexpr uint64_t templateCustom = static_cast<uint64_t>(Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_2);
    constexpr uint64_t apiConstexpr = 0UL;
    bool transA = matmulInfoPtr_->transA;
    bool transB = matmulInfoPtr_->transB;
    uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
    uint64_t quantType = static_cast<uint64_t>(matmulInfoPtr_->quantType);
    bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    bool hasBias = matmulInfoPtr_->hasBias;
    bool isBiasInt32 = matmulInfoPtr_->biasDtype == ge::DT_INT32 &&
                       matmulInfoPtr_->hasBias; // bias dtype 由 tiling key 位 34 承载
    constexpr bool isWeightNz = false;
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        socVersionType, subSocVersionType, antiquantScenario, algorithm, subAlgorithm, templateCustom, apiConstexpr,
        transA, transB, antiquantType, quantType, hasAntiquantOffset, hasBias, isBiasInt32, isWeightNz, fusedOpType_);
    return tilingKey;
}

REGISTER_TILING_TEMPLATE("FusedWeightQuantMatMul", FusedWeightQuantMatMulIterbatchTiling, FWQM_ITERBATCH_PRIORITY);

} // namespace fused_weight_quant_mat_mul
} // namespace optiling
