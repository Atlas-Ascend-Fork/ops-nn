/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fused_weight_quant_mat_mul.cpp
 * \brief FusedWeightQuantMatMul kernel 入口（分发壳，仿照 fused_quant_mat_mul arch35 模式）。
 *        __CUBE_S8S4_S4S4__ 路径：按模板参数分发到底座 wqbmm_v2 的
 *        ASW（手写 CustomMmad kernel，relu 由 FixpipeOut 的 fixpipeParams.reluEn 随路生效）与
 *        ITERBATCH（Matmul API，relu 由 enableRelu 的 MatmulConfig 随路生效）kernel，
 *        tiling data/输入布局与底座完全一致，融合算子仅多注入 FusedOpType::RELU。
 */

#include "../../inc/macro.h"

#define ENABLE_L2_CACHE
// 注意：kernel 编译基于复制树 build/tbe/ascendc/<op>/（op_kernel 内容平铺），相对路径按该布局编写
#include "../../weight_quant_batch_matmul_v2/weight_quant_batch_matmul_v2_constant.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "adv_api/matmul/tiling.h"
#include "../../weight_quant_batch_matmul_v2/arch35/weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#include "../../weight_quant_batch_matmul_v2/arch35/weight_quant_batch_matmul_v2_adaptive_sliding_window.h"
#include "../../weight_quant_batch_matmul_v2/arch35/weight_quant_batch_matmul_v2_iterbatch.h"
#include "fused_weight_quant_mat_mul_tilingkey.h"

using namespace WeightQuantBatchMatmulV2;
using namespace WeightQuantBatchMatmulV2::Arch35;

// relu 使能的 MatmulConfig：fixpipe 随路激活（qbmm_cube_on_the_fly.h 同款机制，
// T1.2 评估结论：ITERBATCH 模板参数已含 const MatmulConfig&，由本分发壳直接传 relu cfg，底座零改动）。
// 注意：GetMMConfig 带 __aicore__ 标注，不能在 lambda（host 上下文）内调用，
// 须先在 namespace 作用域构造基础 cfg（与底座 weight_quant_batch_matmul_v2.cpp 同款），lambda 仅改字段。
constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
constexpr MatmulBatchParams batchParams{false, BatchMode::BATCH_LESS_THAN_L1, false, BatchOutMode::MULTI_BATCH};
constexpr MatmulBatchParams batchParamsNoBatchOut{false, BatchMode::BATCH_LESS_THAN_L1, false,
                                                  BatchOutMode::SINGLE_BATCH};
constexpr MatmulConfig MM_CFG_BASE_MULTI_BATCH = GetMMConfig<configMode>(batchParams);
constexpr MatmulConfig MM_CFG_BASE_MULTI_BATCH_NO_BATCH_OUT = GetMMConfig<configMode>(batchParamsNoBatchOut);
constexpr MatmulConfig MM_CFG_RELU_MULTI_BATCH = [] {
    auto cfg = MM_CFG_BASE_MULTI_BATCH;
    cfg.enableRelu = true;
    return cfg;
}();
constexpr MatmulConfig MM_CFG_RELU_MULTI_BATCH_NO_BATCH_OUT = [] {
    auto cfg = MM_CFG_BASE_MULTI_BATCH_NO_BATCH_OUT;
    cfg.enableRelu = true;
    return cfg;
}();

// ASW kernel（SUB_ALGO_ASW）：tiling/输入布局与底座一致，仅追加 FusedOpType::RELU 模板参数。
// bias dtype 由 tiling key 位 34（IS_BIAS_INT32）在编译期条件分派（对齐底座
// weight_quant_batch_matmul_v2.cpp:155 的 conditional_t 模式）——binary 组合去重不含
// optional 输入（bias）的 dtype，DTYPE_BIAS 宏不可靠，单 binary 双 dtype 由此承载
#define INVOKE_FUSED_WEIGHT_QUANT_BMM_OP_ASW_IMPL(templateClass, ...)                                                \
    do {                                                                                                             \
        GET_TILING_DATA_WITH_STRUCT(wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWCustomTilingDataParams, tilingDataIn, \
                                    tiling);                                                                         \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DtypeBias, DTYPE_Y, __VA_ARGS__> op;                                    \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace,             \
                &tilingDataIn);                                                                                      \
        op.Process();                                                                                                \
    } while (0)

// ITERBATCH kernel：tiling/输入布局与底座一致，mmCfg 换用 enableRelu 的 cfg；
// bias dtype 同样由 IS_BIAS_INT32 位条件分派
#define INVOKE_FUSED_WEIGHT_QUANT_BMM_ITERBATCH_IMPL(templateClass, ...)                                       \
    do {                                                                                                       \
        GET_TILING_DATA_WITH_STRUCT(wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingDataParams, tilingDataIn, \
                                    tiling);                                                                   \
        AscendC::TPipe tPipe;                                                                                  \
        templateClass<DTYPE_X, DTYPE_WEIGHT, DtypeBias, DTYPE_Y, __VA_ARGS__> op;                              \
        op.Init(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace,       \
                &tilingDataIn, &tPipe);                                                                        \
        op.Process();                                                                                          \
    } while (0)

template <int SOC_VERSION_TYPE, int SUB_SOC_VERSION_TYPE, int ANTIQUANT_SCENARIO, int ALGORITHM, int SUB_ALGORITHM,
          int TEMPLATE_CUSTOM, int API_CONSTEXPR, bool TRANS_A, bool TRANS_B, int ANTIQUANT_TYPE, int QUANT_TYPE,
          bool HAS_ANTIQUANT_OFFSET, bool HAS_BIAS, bool IS_BIAS_INT32, bool IS_WEIGHT_NZ, int FUSED_OPTYPE>
__global__ __aicore__ void fused_weight_quant_mat_mul(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale,
                                                      GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
                                                      GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingDataParams);
#if (defined(FORMAT_WEIGHT) && (FORMAT_WEIGHT != FORMAT_FRACTAL_NZ))
    // SEL 已限定：MMAD_S8S4 + FIXPIPE_ANTIQUANT + PER_CHANNEL + 无量化输入（quantType 取底座默认 PER_TENSOR）+
    // 无 antiquant_offset + weight ND + relu；此处按 SUB_ALGORITHM 分发，antiQuantType/quantType 取模板参数。
    // bias dtype：IS_BIAS_INT32 位条件分派（fp16 / int32），DTYPE_BIAS 宏不参与实例化
    using DtypeBias = AscendC::Std::conditional_t<IS_BIAS_INT32, int32_t, DTYPE_X>;
    constexpr QuantType antiQuantType = static_cast<QuantType>(ANTIQUANT_TYPE);
    constexpr QuantType quantType = static_cast<QuantType>(QUANT_TYPE);
    if constexpr (FUSED_OPTYPE == FWQM_FUSED_OPTYPE_RELU && SOC_VERSION_TYPE == FWQM_SOC_SUPPORT_MMAD_S8S4 &&
                  SUB_ALGORITHM == FWQM_SUB_ALGO_ASW) {
        // ASW（手写 CustomMmad kernel）：relu 由 FixpipeOut 的 fixpipeParams.reluEn 随路生效
        INVOKE_FUSED_WEIGHT_QUANT_BMM_OP_ASW_IMPL(WeightQuantBatchMatmulV2ASWKernel, TRANS_A, TRANS_B, antiQuantType,
                                                  false, quantType, FusedOpType::RELU);
    } else if constexpr (FUSED_OPTYPE == FWQM_FUSED_OPTYPE_RELU && SOC_VERSION_TYPE == FWQM_SOC_SUPPORT_MMAD_S8S4 &&
                         SUB_ALGORITHM == FWQM_SUB_ALGO_ITERATE_BATCH) {
        INVOKE_FUSED_WEIGHT_QUANT_BMM_ITERBATCH_IMPL(WeightQuantBatchMatmulV2IterBatchKernel, TRANS_A, TRANS_B,
                                                     antiQuantType, false, quantType, MM_CFG_RELU_MULTI_BATCH);
    } else if constexpr (FUSED_OPTYPE == FWQM_FUSED_OPTYPE_RELU && SOC_VERSION_TYPE == FWQM_SOC_SUPPORT_MMAD_S8S4 &&
                         SUB_ALGORITHM == FWQM_SUB_ALGO_ITERATE_BATCH_NO_BATCH_OUT) {
        INVOKE_FUSED_WEIGHT_QUANT_BMM_ITERBATCH_IMPL(WeightQuantBatchMatmulV2IterBatchKernel, TRANS_A, TRANS_B,
                                                     antiQuantType, false, quantType,
                                                     MM_CFG_RELU_MULTI_BATCH_NO_BATCH_OUT);
    }
#endif
}
