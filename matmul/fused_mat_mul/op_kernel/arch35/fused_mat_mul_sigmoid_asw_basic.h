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
 * \file fused_mat_mul_sigmoid_asw_basic.h
 * \brief Blaze assembly for the BF16/FP16 FusedMatMul sigmoid template on DAV_3510.
 */

#pragma once

#include "blaze/epilogue/block/block_epilogue_matmul_with_sigmoid.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "fused_mat_mul_tiling_data.h"
#include "fused_mat_mul_tiling_key_public.h"

namespace FusedMatMulAdvanced {

template <typename ElementType, typename BiasType, bool TransA, bool TransB>
__aicore__ inline void FusedMatMulSigmoidAswBasicKernel(GM_ADDR x1Gm, GM_ADDR x2Gm, GM_ADDR biasGm, GM_ADDR yGm,
                                                        GM_ADDR workspaceGm,
                                                        const MatMulV3BasicTilingData& matmulTiling)
{
    using AccType = float;
    static constexpr bool splitM = AscendC::IsSameType<AccType, float>::value;
    // The BF16/FP16 sigmoid output reuses UB space after the FP32 accumulator.
    static constexpr uint8_t singleUbBuffer = 1U;
    using LayoutA = AscendC::Std::conditional_t<TransA, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutB = AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<Blaze::Gemm::ND_ALIG_1V2_FIXPIPE, F_OPTYPE_SIGMOID,
                                                                    Blaze::Gemm::KernelMmadWithEpilogue>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, Blaze::Gemm::NONE_FULL_LOAD_MODE,
                                                                         false, true>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ElementType, LayoutA, ElementType, LayoutB, AccType,
                                                    LayoutC, BiasType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueMatmulWithSigmoid<DispatchPolicy, ElementType>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelParams = typename MatmulKernel::Params;

    KernelParams params = {
        {matmulTiling.m, matmulTiling.n, matmulTiling.k, 1},
        {x1Gm, x2Gm, nullptr, biasGm, nullptr, workspaceGm, matmulTiling.k, matmulTiling.mL1, matmulTiling.nL1,
         matmulTiling.kL1, matmulTiling.baseM, matmulTiling.baseN, matmulTiling.baseK, matmulTiling.l1BufferNum,
         matmulTiling.l0cDB, splitM, singleUbBuffer},
        {yGm},
        {matmulTiling.mL1, matmulTiling.nL1, matmulTiling.kL1, matmulTiling.baseM, matmulTiling.baseN,
         matmulTiling.baseK, matmulTiling.mTailCnt, matmulTiling.nTailCnt, matmulTiling.mBaseTailSplitCnt,
         matmulTiling.nBaseTailSplitCnt, matmulTiling.mTailMain, matmulTiling.nTailMain, matmulTiling.mmadParam,
         static_cast<uint32_t>(matmulTiling.l2CacheDisable), matmulTiling.sliceM, matmulTiling.srcNdStride,
         matmulTiling.innerBatch}};

    params.epilogueParams.problemShape = params.problemShape;
    params.epilogueParams.splitM = params.mmadParams.splitM != 0;

    MatmulKernel kernel;
    kernel(params);
}

} // namespace FusedMatMulAdvanced
