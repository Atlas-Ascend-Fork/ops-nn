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
 * \file confidence_head_fused_kernel.h
 * \brief Kernel entry wrapper for ConfidenceHeadFused.
 */

#ifndef CONFIDENCE_HEAD_FUSED_KERNEL_H
#define CONFIDENCE_HEAD_FUSED_KERNEL_H

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"
#include "confidence_head_fused_tiling_data.h"
#include "confidence_head_fused_kernel_tiling_key.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_confidence_head_fused.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/epilogue/block/block_epilogue_confidence_head_fused.h"
#include "blaze/gemm/kernel/kernel_confidence_head_fused_mix.h"

namespace CHF {

template <int8_t TilingKey, class HiddenType, class WeightType, class OutType>
__aicore__ inline void ConfidenceHeadFusedKernel(GM_ADDR hidden, GM_ADDR weight, GM_ADDR markovEmb, GM_ADDR temperature,
                                                 GM_ADDR bias, GM_ADDR out, GM_ADDR workspace,
                                                 const ConfidenceHeadFusedTilingData* tiling)
{
    static_assert(TilingKey == CONFIDENCE_HEAD_FUSED_NO_MARKOV || TilingKey == CONFIDENCE_HEAD_FUSED_WITH_MARKOV,
                  "Unsupported ConfidenceHeadFused tiling key.");
    constexpr bool HAS_MARKOV = TilingKey == CONFIDENCE_HEAD_FUSED_WITH_MARKOV;
    (void)workspace;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::dn_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using DispatchPolicy = Blaze::Gemm::ConfidenceHeadFusedPolicy<Blaze::Gemm::A_FULL_LOAD_MODE, false, HAS_MARKOV>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, Blaze::Gemm::A_FULL_LOAD_MODE>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, WeightType, LayoutA, HiddenType, LayoutB, float,
                                                    LayoutC, float, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueConfidenceHeadFused<float, OutType>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelParams = typename MatmulKernel::Params;

    const ProblemShape problemShape{1L, static_cast<int64_t>(tiling->params.n),
                                    static_cast<int64_t>(tiling->params.kEmbed),
                                    static_cast<int64_t>(tiling->params.kMarkov)};
    KernelParams params = {
        problemShape,
        weight,
        hidden,
        markovEmb,
        {tiling->params.kEmbed, HAS_MARKOV ? tiling->params.kMarkov : 0U, tiling->cubeTiling.baseN,
         tiling->cubeTiling.baseK, tiling->cubeTiling.kL1, tiling->cubeTiling.l1BufferNum,
         static_cast<uint16_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT), tiling->params.hasBias != 0 ? bias : nullptr},
        {out, temperature, bias, tiling->params.lowLimit, tiling->params.highLimit, tiling->params.biasN,
         tiling->params.hasBias != 0, tiling->params.hasLowLimit != 0, tiling->params.hasHighLimit != 0, problemShape,
         tiling->cubeTiling.baseN},
        {1U, tiling->cubeTiling.baseN, tiling->cubeTiling.kL1, 1U, tiling->cubeTiling.baseN, tiling->cubeTiling.baseK,
         1U, 1U}};

    MatmulKernel kernel;
    kernel(params);
}

} // namespace CHF

#endif // CONFIDENCE_HEAD_FUSED_KERNEL_H
