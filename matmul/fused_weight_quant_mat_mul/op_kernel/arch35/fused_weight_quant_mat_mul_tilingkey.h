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
 * \file fused_weight_quant_mat_mul_tilingkey.h
 * \brief FusedWeightQuantMatMul kernel 侧 tiling key 声明。
 *        模板参数 = 底座 15 个 + FUSED_OPTYPE；FWQM_* 宏取值与底座
 *        weight_quant_batch_matmul_v2_arch35_tiling_key.h 的 WQBMMV2_* 宏显式对齐
 *        （避免 include 底座 tiling key 造成同名 TPL 声明冲突）。
 *        取值域（由 checker/融合 pass 保证）：
 *        ANTIQUANT_TYPE 仅 PER_CHANNEL、QUANT_TYPE 沿用底座无量化输入时的 PER_TENSOR 默认值、
 *        HAS_ANTIQUANT_OFFSET=0、IS_WEIGHT_NZ=0、FUSED_OPTYPE 仅 RELU。
 *        位 34（底座 IS_BIAS_FP32 位置）语义重定义为 IS_BIAS_INT32：bias dtype ∈ {FP16, INT32}
 *        由 tiling key 运行时选择——binary 生成时 optional 输入（bias）的
 *        dtype 不参与组合去重（ascendc_bin_param_build.py 按 required 输入去重），单 binary
 *        无法区分 bias dtype，须由 tiling key 位承载（对齐底座 IS_BIAS_FP32 机制，
 *        底座 conditional_t 模式见 weight_quant_batch_matmul_v2.cpp:155）。
 */

#ifndef OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_TILING_KEY_H
#define OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// ===== 以下宏值对齐底座 WQBMMV2_*（weight_quant_batch_matmul_v2_arch35_tiling_key.h）=====
// 平台大类
#define FWQM_SOC_RESERVED 0
#define FWQM_SOC_SUPPORT_L0C_TO_OUT 1
#define FWQM_SOC_SUPPORT_L1_TO_BT_BF16 2
#define FWQM_SOC_SUPPORT_MMAD_S8S4 3

#define FWQM_DEFAULT 0

// 算法大类
#define FWQM_ALGO_VECTOR_ANTIQUANT 0
#define FWQM_ALGO_MULTI_SCALE_DEQUANT 1
#define FWQM_ALGO_FIXPIPE_ANTIQUANT 2

// 算法小类
#define FWQM_SUB_ALGO_VDEFAULT 0
#define FWQM_SUB_ALGO_SPLIT_K 1
#define FWQM_SUB_ALGO_N_FIRST_TAIL_RESPLIT 2
#define FWQM_SUB_ALGO_N_FIRST_BASIC_BLOCK 3
#define FWQM_SUB_ALGO_ASW 4
#define FWQM_SUB_ALGO_ANTI_REG_SCSC 5
#define FWQM_SUB_ALGO_ITERATE_BATCH 6
#define FWQM_SUB_ALGO_CUSTOM_BACKWARD_COMPATIBLE 7
#define FWQM_SUB_ALGO_ITERATE_BATCH_NO_BATCH_OUT 8

// 模板自定义
#define FWQM_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 0

// 反量化类型
#define FWQM_ANTIQUANT_TYPE_NONE 0
#define FWQM_ANTIQUANT_TYPE_PER_TENSOR 1
#define FWQM_ANTIQUANT_TYPE_PER_CHANNEL 2
#define FWQM_ANTIQUANT_TYPE_PER_GROUP 3

// 量化类型（一期无输出量化；底座 tiling 在无量化输入时 quantType 取默认值 PER_TENSOR）
#define FWQM_QUANT_TYPE_NONE 0
#define FWQM_QUANT_TYPE_PER_TENSOR 1
#define FWQM_QUANT_TYPE_PER_CHANNEL 2

// 融合 epilogue 类型，RELU 对齐 kernel 侧 WeightQuantBatchMatmulV2::Arch35::FusedOpType 枚举值
#define FWQM_FUSED_OPTYPE_NONE 0
#define FWQM_FUSED_OPTYPE_RELU 1

// tiling 模板注册优先级（对齐底座 ITERBATCH_PRIORITY/ASW_PRIORITY）
constexpr int32_t FWQM_ITERBATCH_PRIORITY = 9;
constexpr int32_t FWQM_ASW_PRIORITY = 11;

ASCENDC_TPL_ARGS_DECL(
    FusedWeightQuantMatMul,
    ASCENDC_TPL_UINT_DECL( // 平台大类
        SOC_VERSION_TYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, FWQM_SOC_RESERVED, FWQM_SOC_SUPPORT_L0C_TO_OUT,
        FWQM_SOC_SUPPORT_L1_TO_BT_BF16, FWQM_SOC_SUPPORT_MMAD_S8S4),
    ASCENDC_TPL_UINT_DECL( // 平台小类
        SUB_SOC_VERSION_TYPE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, FWQM_DEFAULT),
    ASCENDC_TPL_UINT_DECL( // 伪量化场景
        ANTIQUANT_SCENARIO, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, FWQM_DEFAULT),
    ASCENDC_TPL_UINT_DECL( // 算法大类
        ALGORITHM, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, FWQM_ALGO_VECTOR_ANTIQUANT, FWQM_ALGO_MULTI_SCALE_DEQUANT,
        FWQM_ALGO_FIXPIPE_ANTIQUANT),
    ASCENDC_TPL_UINT_DECL( // 算法小类
        SUB_ALGORITHM, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, FWQM_SUB_ALGO_VDEFAULT, FWQM_SUB_ALGO_SPLIT_K,
        FWQM_SUB_ALGO_N_FIRST_TAIL_RESPLIT, FWQM_SUB_ALGO_N_FIRST_BASIC_BLOCK, FWQM_SUB_ALGO_ASW,
        FWQM_SUB_ALGO_ANTI_REG_SCSC, FWQM_SUB_ALGO_ITERATE_BATCH, FWQM_SUB_ALGO_CUSTOM_BACKWARD_COMPATIBLE,
        FWQM_SUB_ALGO_ITERATE_BATCH_NO_BATCH_OUT),
    ASCENDC_TPL_UINT_DECL( // 模板自定义
        TEMPLATE_CUSTOM, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, FWQM_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2),
    ASCENDC_TPL_UINT_DECL( // api常量保留
        API_CONSTEXPR, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, FWQM_DEFAULT),
    ASCENDC_TPL_BOOL_DECL( // 转置场景
        TRANS_A, 0, 1),
    ASCENDC_TPL_BOOL_DECL( // 转置场景
        TRANS_B, 0, 1),
    ASCENDC_TPL_UINT_DECL( // 反量化类型
        ANTIQUANT_TYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, FWQM_ANTIQUANT_TYPE_NONE, FWQM_ANTIQUANT_TYPE_PER_TENSOR,
        FWQM_ANTIQUANT_TYPE_PER_CHANNEL, FWQM_ANTIQUANT_TYPE_PER_GROUP),
    ASCENDC_TPL_UINT_DECL( // 量化类型
        QUANT_TYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, FWQM_QUANT_TYPE_NONE, FWQM_QUANT_TYPE_PER_TENSOR,
        FWQM_QUANT_TYPE_PER_CHANNEL),
    ASCENDC_TPL_BOOL_DECL( // 可选输入
        HAS_ANTIQUANT_OFFSET, 0, 1),
    ASCENDC_TPL_BOOL_DECL( // 可选输入
        HAS_BIAS, 0, 1),
    ASCENDC_TPL_BOOL_DECL( // bias dtype 为 INT32（位 34，底座 IS_BIAS_FP32 位置语义重定义）
        IS_BIAS_INT32, 0, 1),
    ASCENDC_TPL_BOOL_DECL( // weight的格式
        IS_WEIGHT_NZ, 0, 1),
    ASCENDC_TPL_UINT_DECL( // 融合epilogue类型
        FUSED_OPTYPE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, FWQM_FUSED_OPTYPE_NONE, FWQM_FUSED_OPTYPE_RELU), );

// 实例化范围：MMAD_S8S4 + FIXPIPE_ANTIQUANT + ASW/ITERBATCH(+NO_BATCH_OUT) +
// PerChannel + 无量化输入（quantType 取底座默认 PER_TENSOR）+ 无 antiquant_offset +
// bias 有无 × bias dtype{fp16,int32}（tiling key 位 34 承载，kernel 侧 conditional_t 分派）+
// weight ND + relu
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),
                         ASCENDC_TPL_UINT_SEL(SOC_VERSION_TYPE, ASCENDC_TPL_UI_LIST, FWQM_SOC_SUPPORT_MMAD_S8S4),
                         ASCENDC_TPL_UINT_SEL(SUB_SOC_VERSION_TYPE, ASCENDC_TPL_UI_LIST, FWQM_DEFAULT),
                         ASCENDC_TPL_UINT_SEL(ANTIQUANT_SCENARIO, ASCENDC_TPL_UI_LIST, FWQM_DEFAULT),
                         ASCENDC_TPL_UINT_SEL(ALGORITHM, ASCENDC_TPL_UI_LIST, FWQM_ALGO_FIXPIPE_ANTIQUANT),
                         ASCENDC_TPL_UINT_SEL(SUB_ALGORITHM, ASCENDC_TPL_UI_LIST, FWQM_SUB_ALGO_ASW,
                                              FWQM_SUB_ALGO_ITERATE_BATCH, FWQM_SUB_ALGO_ITERATE_BATCH_NO_BATCH_OUT),
                         ASCENDC_TPL_UINT_SEL(TEMPLATE_CUSTOM, ASCENDC_TPL_UI_LIST,
                                              FWQM_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2),
                         ASCENDC_TPL_UINT_SEL(API_CONSTEXPR, ASCENDC_TPL_UI_LIST, FWQM_DEFAULT),
                         ASCENDC_TPL_BOOL_SEL(TRANS_A, 0, 1), ASCENDC_TPL_BOOL_SEL(TRANS_B, 0, 1),
                         ASCENDC_TPL_UINT_SEL(ANTIQUANT_TYPE, ASCENDC_TPL_UI_LIST, FWQM_ANTIQUANT_TYPE_PER_CHANNEL),
                         ASCENDC_TPL_UINT_SEL(QUANT_TYPE, ASCENDC_TPL_UI_LIST, FWQM_QUANT_TYPE_PER_TENSOR),
                         ASCENDC_TPL_BOOL_SEL(HAS_ANTIQUANT_OFFSET, 0), ASCENDC_TPL_BOOL_SEL(HAS_BIAS, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(IS_BIAS_INT32, 0, 1), ASCENDC_TPL_BOOL_SEL(IS_WEIGHT_NZ, 0),
                         ASCENDC_TPL_UINT_SEL(FUSED_OPTYPE, ASCENDC_TPL_UI_LIST, FWQM_FUSED_OPTYPE_RELU)), );

#endif // OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_TILING_KEY_H
