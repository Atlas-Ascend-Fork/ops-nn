/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "kernel_operator.h"
#include "../rotate_dual_level_quant_tiling_data.h"

#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/epilogue/block/block_epilogue_rotate_quant.h"
#include "tensor_api/tensor.h"

namespace RotateDualLevelQuantNS {

template <class XType, class YType, class Level0ScaleType, class Level1ScaleType, bool EnableGrouped = false>
__aicore__ inline void RotateDualLevelQuantBlazeKernel(GM_ADDR xGM, GM_ADDR rotationGM, GM_ADDR alphaGM,
                                                       GM_ADDR groupListGM, GM_ADDR yGM, GM_ADDR level0ScaleGM,
                                                       GM_ADDR level1ScaleGM,
                                                       const RotateDualLevelQuantTilingData& data)
{
    static_assert(AscendC::IsSameType<XType, half>::value || AscendC::IsSameType<XType, bfloat16_t>::value,
                  "RotateDualLevelQuant requires matching FP16/BF16 x and rotation");
    static_assert(AscendC::IsSameType<YType, fp4x2_e2m1_t>::value || AscendC::IsSameType<YType, fp8_e4m3fn_t>::value ||
                      AscendC::IsSameType<YType, fp8_e5m2_t>::value,
                  "RotateDualLevelQuant requires MX FP4/FP8 output");
    static_assert(AscendC::IsSameType<Level0ScaleType, float>::value, "level0_scale must be FP32");
    static_assert(sizeof(Level1ScaleType) == 1, "level1_scale must use one-byte E8M0 storage");

    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using Layout = asc::te::nd_ext_layout_ptn;
    using SchedulerShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = AscendC::Std::conditional_t<EnableGrouped, Blaze::Gemm::Block::BlockSchedulerGmmNoQuant,
                                                       Blaze::Gemm::Block::BlockSchedulerMatmulBasic<SchedulerShape>>;
    // The common Block produces K-wide Split-M results and owns the Fixpipe slots.
    // The epilogue collects each slice before returning its input credit.
    using Policy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<
        Blaze::Gemm::ND_ALIG_1V2_FIXPIPE, Blaze::Gemm::OP_TYPE_QUANT, Blaze::Gemm::KernelMmadRotateDualLevelQuant, 0>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<Policy, XType, Layout, XType, Layout, bfloat16_t, Layout, XType,
                                                    Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueRotateQuant<
        bfloat16_t, YType, Level1ScaleType, Blaze::Epilogue::Fusion::DefaultFusion<YType, bfloat16_t>, true>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    static_assert(MatmulKernel::ENABLE_QUANT, "RotateDualLevelQuant requires the quantization epilogue execution path");
    static_assert(MX_WORKSPACE_BYTES >= BlockEpilogue::NORMALIZED_WORKSPACE_BYTES,
                  "NN must reserve the complete normalized MX workspace");
    static_assert(MAX_ROWS_PER_AIV <= BlockEpilogue::MAX_NORMALIZED_ROWS,
                  "NN local-row capacity exceeds the normalized MX workspace");

    // Schedule complete level0 blocks; BlockMmad produces rotK-wide slices of each block.
    typename BlockScheduler::Params schedulerParams;
    if constexpr (EnableGrouped) {
        schedulerParams = {
            static_cast<int32_t>(data.mL1),
            static_cast<int32_t>(data.level0BlockSize), // baseM, baseN
            1,
            1,
            1,
            1, // mTailCnt, nTailCnt, mTailAlign, nTailAlign
            0,
            data.groupNum,
            data.M, // groupType (split M), groupNum, initialM
            true,
            true,
            true, // singleX, singleWeight, singleY
            false,
            false, // transB, weightNz
            sizeof(XType),
            static_cast<uint32_t>(data.groupListType), // weightElementSize, groupListType
        };
    } else {
        schedulerParams = {
            data.mL1,  data.level0BlockSize,
            data.rotK, // mL1, nL1, kL1
            data.mL1,  data.level0BlockSize,
            data.rotK,    // baseM, baseN, baseK
            1,         1, // mTailCnt, nTailCnt
        };
    }

    bfloat16_t alpha{0.0f};
    bool needClamp = false;
    if (data.hasAlpha != 0) {
        auto layout = asc::te::make_layout(asc::te::make_shape(1L), asc::te::make_stride(1L));
        auto alphaTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ bfloat16_t*>(alphaGM)), layout);
        alpha = alphaTensor[0];
        const float value = AscendC::Cast(alpha);
        needClamp = value > 0.0f && value < 1.0f;
    }

    const ProblemShape problemShape{data.M, data.N, data.rotK, data.rotationMatricesPerGroup};
    const uint32_t pitchGran = data.ubPitchGran == 0 ? static_cast<uint32_t>(asc::te::c0_element<bfloat16_t>) :
                                                       data.ubPitchGran;
    const uint32_t inputPitch = Blaze::Gemm::CeilAlign<uint32_t>(data.baseN, pitchGran);
    using Params = typename MatmulKernel::Params;
    const Params params = {
        problemShape,
        {
            // BlockMmad
            xGM,
            rotationGM,
            nullptr,
            nullptr,
            nullptr,
            nullptr, // A, B, C, bias, group list, workspace GM addresses
            data.rotK,
            data.mL1,
            data.rotK,
            data.rotK, // oriK, mL1, nL1, kL1
            data.baseM,
            data.baseN,
            data.baseK, // mL0, nL0, kL0
            ROTATION_BUFFER_COUNT,
            ROTATION_BUFFER_COUNT, // l1Stages, l0cStages
            data.splitM,
            static_cast<uint8_t>(data.ubDB),
            0,
            data.ubPitchGran, // splitM, ubDB, rowStride, ubPitchGran
        },
        schedulerParams,
        EnableGrouped ? groupListGM : nullptr, // groupListGmAddr
        {
            // BlockEpilogue
            {
                // MX (level1)
                yGM,
                level1ScaleGM,
                problemShape,
                bfloat16_t{0.0f},
                false, // alpha, needClamp: clipping is applied at level0
                data.axis,
                data.roundMode,
                data.scaleAlg,
                data.dstTypeMax,
                data.invDstTypeMax,
                data.ubDB,
                data.splitM != 0,
            },
            {
                // Level0
                level0ScaleGM,
                data.level0ScalesPerRow,
                data.maxRowsPerAiv,
                data.rotK,
                data.fixpipeReceiveBytes,
                inputPitch, // workspaceBase in bytes, inputPitch in BF16 elements
                data.collectUbOffset,
                data.level0ScaleUbOffset,
                alpha,
                needClamp,
            },
        },
    };
    MatmulKernel kernel;
    kernel(params);
}
} // namespace RotateDualLevelQuantNS
