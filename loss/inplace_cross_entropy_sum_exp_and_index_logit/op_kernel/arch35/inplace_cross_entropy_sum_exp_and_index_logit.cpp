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
 * \file inplace_cross_entropy_sum_exp_and_index_logit.cpp
 * \brief A5 (ascend950) inplace kernel entry — reuses base kernel class with IS_INPLACE=true
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#if __has_include( \
    "../../cross_entropy_sum_exp_and_index_logit/op_kernel/arch35/cross_entropy_sum_exp_and_index_logit_struct.h")
#include "../../cross_entropy_sum_exp_and_index_logit/op_kernel/arch35/cross_entropy_sum_exp_and_index_logit_struct.h"
#include "../../cross_entropy_sum_exp_and_index_logit/op_kernel/arch35/cross_entropy_sum_exp_and_index_logit_tiling_key.h"
#include "../../cross_entropy_sum_exp_and_index_logit/op_kernel/arch35/cross_entropy_sum_exp_and_index_logit.h"
#else
#include "../cross_entropy_sum_exp_and_index_logit/arch35/cross_entropy_sum_exp_and_index_logit_struct.h"
#include "../cross_entropy_sum_exp_and_index_logit/arch35/cross_entropy_sum_exp_and_index_logit_tiling_key.h"
#include "../cross_entropy_sum_exp_and_index_logit/arch35/cross_entropy_sum_exp_and_index_logit.h"
#endif

using namespace AscendC;
using namespace CrossEntropySumExpAndIndexLogit;

template <uint32_t MODE>
__global__ __aicore__ void inplace_cross_entropy_sum_exp_and_index_logit(
    GM_ADDR vocab_parallel_logits, GM_ADDR target, GM_ADDR global_logits_max, GM_ADDR predicted_logits,
    GM_ADDR sum_exp_logits, GM_ADDR vocab_parallel_logits_out, GM_ADDR target_offset, GM_ADDR target_mask,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (MODE == CE_SCH_MODE_DEFAULT) {
        GET_TILING_DATA_WITH_STRUCT(CrossEntropySumExpAndIndexLogitRegBaseTilingData, tilingData, tiling);
        // 原地版：IS_INPLACE=true，编译期裁剪出 stride 路径 + BF16 窄化写路径
        // vocab_parallel_logits_out 即 expLogits 位置（与 vocab_parallel_logits 同址）
        KernelCrossEntropyRegbase<DTYPE_VOCAB_PARALLEL_LOGITS, true> op;
        op.Init(vocab_parallel_logits, target, global_logits_max, predicted_logits, sum_exp_logits,
                vocab_parallel_logits_out, target_offset, target_mask, &tilingData);
        op.Process();
    }
}
