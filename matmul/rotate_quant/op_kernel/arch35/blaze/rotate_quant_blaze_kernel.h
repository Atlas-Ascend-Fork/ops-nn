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
 * \file rotate_quant_blaze_kernel.h
 * \brief
 */

#pragma once

#include "../rotate_quant_tiling_data.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_grouped_matmul.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/epilogue/block/block_epilogue_rotate_quant.h"
#include "tensor_api/tensor.h"

namespace RotateQuantNS {

template <bool EnableGrouped, uint64_t L0COutModel, class X_TYPE, class Y_TYPE, class SCALE_TYPE>
__aicore__ inline void RotateQuantBlazeKernelImpl(GM_ADDR xGM, GM_ADDR rotGM, GM_ADDR alphaGM, GM_ADDR groupListGM,
                                                  GM_ADDR yGM, GM_ADDR scaleGM,
                                                  const RotateQuantAptOpt::RotateQuantAptTilingData& tilingData)
{
    static_assert(L0COutModel == Blaze::Gemm::ND_ALIG_1V1_FIXPIPE || L0COutModel == Blaze::Gemm::ND_ALIG_1V2_FIXPIPE,
                  "RotateQuant requires a single- or dual-AIV Fixpipe output mode");
    using TbmmProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
    using GroupedProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = AscendC::Std::conditional_t<EnableGrouped, GroupedProblemShape, TbmmProblemShape>;
    using Layout = asc::te::nd_ext_layout_ptn;
    constexpr bool BATCH_INTERLEAVED = true;
    using TbmmScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<TbmmProblemShape, 0, false, true,
                                                                        BATCH_INTERLEAVED>;
    using BlockScheduler = AscendC::Std::conditional_t<EnableGrouped, Blaze::Gemm::Block::BlockSchedulerGmmNoQuant,
                                                       TbmmScheduler>;
    using KernelSchedule = AscendC::Std::conditional_t<EnableGrouped, Blaze::Gemm::KernelGroupedMmadNoQuant,
                                                       Blaze::Gemm::KernelMmadMultiBlockTBMM>;
    constexpr uint64_t NON_CONTIGUOUS_TYPE = EnableGrouped ?
                                                 0 :
                                                 static_cast<uint64_t>(
                                                     Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1);
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<L0COutModel, Blaze::Gemm::OP_TYPE_QUANT,
                                                                    KernelSchedule, NON_CONTIGUOUS_TYPE>;
    // Both paths retain FP32 Cube accumulation followed by a BF16 intermediate result.
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, X_TYPE, Layout, X_TYPE, Layout, bfloat16_t, Layout,
                                                    X_TYPE, Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueRotateQuant<bfloat16_t, Y_TYPE, SCALE_TYPE>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    static_assert(MatmulKernel::ENABLE_QUANT, "RotateQuant requires the Quant epilogue execution path");

    using Params = typename MatmulKernel::Params;
    constexpr bool SPLIT_M = L0COutModel == Blaze::Gemm::ND_ALIG_1V2_FIXPIPE;

    bfloat16_t alpha{};
    bool needClamp = false;
    if (tilingData.hasAlpha == 1) {
        auto alphaLayout = asc::te::make_layout(asc::te::make_shape(1L), asc::te::make_shape(1L));
        auto gmAlpha = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ bfloat16_t*>(alphaGM)), alphaLayout);
        alpha = gmAlpha[0];
        const float alphaValue = AscendC::Cast(alpha);
        needClamp = alphaValue > 0.0f && alphaValue < 1.0f;
    }

    const typename BlockEpilogue::Params epilogueParams{yGM,
                                                        scaleGM,
                                                        {tilingData.M, tilingData.N, tilingData.K, tilingData.B},
                                                        alpha,
                                                        needClamp,
                                                        tilingData.axis,
                                                        tilingData.roundMode,
                                                        tilingData.scaleAlg,
                                                        tilingData.dstTypeMax,
                                                        tilingData.invDstTypeMax,
                                                        tilingData.ubDB,
                                                        SPLIT_M};

    MatmulKernel kernel;
    if constexpr (EnableGrouped) {
        // GMM keeps the original shape and resolves each group through its scheduler.
        const int64_t initialM = tilingData.B == 1 ? static_cast<int64_t>(tilingData.M) * tilingData.nKRatio :
                                                     tilingData.M;
        Params params{
            {tilingData.M, tilingData.N, tilingData.K, tilingData.B}, // problemShape
            {xGM, rotGM, nullptr, nullptr, groupListGM, nullptr, tilingData.K, tilingData.mL1, tilingData.K,
             tilingData.K, tilingData.baseM, tilingData.K, tilingData.baseK, 2, 2, static_cast<uint64_t>(SPLIT_M),
             static_cast<uint8_t>(tilingData.ubDB), 0, 0}, // mmParams
            epilogueParams,
            {static_cast<int32_t>(tilingData.mL1), static_cast<int32_t>(tilingData.K), 1, 1, 1, 1, 0,
             tilingData.groupNum, initialM, true, true, true, false, false, static_cast<uint32_t>(sizeof(X_TYPE)),
             static_cast<uint32_t>(tilingData.groupListType), BlockScheduler::MatrixView::BatchTransposedA,
             SPLIT_M},                                                                              // schedulerParams
            {tilingData.groupNum, 0, static_cast<uint32_t>(tilingData.groupListType), 1, 1, 1, 0}}; // gmmParams
        kernel(params);
    } else {
        // Shared rotation folds feature slices into M; independent rotations use batch.
        const int64_t effectiveM = tilingData.B == 1 ?
                                       static_cast<int64_t>(tilingData.M) * (tilingData.N / tilingData.K) :
                                       tilingData.M;
        Params params{{effectiveM, tilingData.K, tilingData.K, tilingData.B, 1}, // problemShape
                      {xGM, rotGM, nullptr, nullptr, nullptr, nullptr, tilingData.K, tilingData.mL1, tilingData.K,
                       tilingData.K, tilingData.baseM, tilingData.K, tilingData.baseK, 2, 2,
                       static_cast<uint64_t>(SPLIT_M), static_cast<uint8_t>(tilingData.ubDB), 0, 0}, // mmadParams
                      epilogueParams,
                      // Preserve the Host tile boundaries without additional tail splitting.
                      {tilingData.mL1, tilingData.K, tilingData.K, 0, 0, 0, 1, 1, 1, 1}}; // schParams
        kernel(params);
    }
}

template <class X_TYPE, class Y_TYPE, class SCALE_TYPE, bool EnableGrouped = false>
__aicore__ inline void RotateQuantBlazeKernel(GM_ADDR xGM, GM_ADDR rotGM, GM_ADDR alphaGM, GM_ADDR groupListGM,
                                              GM_ADDR yGM, GM_ADDR scaleGM,
                                              const RotateQuantAptOpt::RotateQuantAptTilingData& tilingData)
{
    // Both features follow the Host ratio and the entry's MIX launch profile.
    if (tilingData.aivNumRatio == 1) {
        RotateQuantBlazeKernelImpl<EnableGrouped, Blaze::Gemm::ND_ALIG_1V1_FIXPIPE, X_TYPE, Y_TYPE, SCALE_TYPE>(
            xGM, rotGM, alphaGM, groupListGM, yGM, scaleGM, tilingData);
    } else if (tilingData.aivNumRatio == 2) {
        RotateQuantBlazeKernelImpl<EnableGrouped, Blaze::Gemm::ND_ALIG_1V2_FIXPIPE, X_TYPE, Y_TYPE, SCALE_TYPE>(
            xGM, rotGM, alphaGM, groupListGM, yGM, scaleGM, tilingData);
    }
}

} // namespace RotateQuantNS
