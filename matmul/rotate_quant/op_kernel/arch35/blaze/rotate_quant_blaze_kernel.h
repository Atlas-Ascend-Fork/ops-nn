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

#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_rotate_quant.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/epilogue/block/block_epilogue_rotate_quant.h"
#include "tensor_api/tensor.h"
#include "adv_api/matmul/matmul.h"

namespace RotateQuantNS {

struct RowMajor {};

template <class X_TYPE, class Y_TYPE, class SCALE_TYPE, class C_LAYOUT, bool EnableGrouped = false>
__aicore__ inline void RotateQuantBlazeKernel(GM_ADDR xGM, GM_ADDR rotGM, GM_ADDR alphaGM, GM_ADDR yGM, GM_ADDR scaleGM,
                                              GM_ADDR workspaceGM,
                                              const RotateQuantAptOpt::RotateQuantAptTilingData& tilingData,
                                              GM_ADDR groupListGM = nullptr)
{
    using AType = X_TYPE;     // x: fp16/bf16
    using BType = X_TYPE;     // rot: 与 x 同 dtype
    using CType = bfloat16_t; // Fixpipe(F322BF16) 输出，epilogue 输入
    using OutType = Y_TYPE;   // fp4x2_e2m1_t / fp8_e4m3fn_t / fp8_e5m2_t
    using ScaleType = SCALE_TYPE;

    using LayoutA = C_LAYOUT;
    using LayoutB = C_LAYOUT;
    using LayoutC = C_LAYOUT;

    using AMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, false>;
    using BMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, false>;
    using CMatmulType = AscendC::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, CType>;
    using BiasMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, ScaleType>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerRotateQuant<
        AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>, EnableGrouped>;
    using DispatchPolicy = Blaze::Gemm::BlockRotateQuant<Blaze::Gemm::KernelRotateQuant>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AMatmulType, LayoutA, BMatmulType, LayoutB,
                                                    CMatmulType, LayoutC, BiasMatmulType, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueRotateQuant<CType, OutType, ScaleType>;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;

    Params params = {
        {tilingData.M, tilingData.N, tilingData.K, tilingData.B}, // shape
        {xGM,
         rotGM,
         alphaGM,
         {tilingData.M, tilingData.N, tilingData.K, tilingData.B},
         {tilingData.mL1, tilingData.nL1, tilingData.kL1, 1},
         {tilingData.baseM, tilingData.baseN, tilingData.baseK, 1},
         tilingData.B}, // MMAD params
        {yGM,
         scaleGM,
         {tilingData.M, tilingData.N, tilingData.K, tilingData.B},
         {},    // alpha is read from GM by the kernel.
         false, // needClamp is determined by the kernel.
         tilingData.axis,
         tilingData.roundMode,
         tilingData.scaleAlg,
         tilingData.dstTypeMax,
         tilingData.invDstTypeMax}, // epilogue params
        {tilingData.mL1, tilingData.nL1, tilingData.kL1, tilingData.baseM, tilingData.baseN, tilingData.baseK,
         tilingData.nKRatio, tilingData.tailML1, tilingData.dstTypeMax, tilingData.invDstTypeMax, tilingData.axis,
         tilingData.roundMode, tilingData.scaleAlg, tilingData.hasAlpha, tilingData.trans}}; // scheduler params
    if constexpr (EnableGrouped) {
        params.groupParams = {groupListGM, tilingData.groupNum, tilingData.groupListType};
    }

    MatmulKernel mm;
    mm(params);
}
} // namespace RotateQuantNS
