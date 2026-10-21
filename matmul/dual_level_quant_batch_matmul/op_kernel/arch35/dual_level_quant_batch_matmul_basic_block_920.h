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
 * \file dual_level_quant_batch_matmul_basic_block_920.h
 * \brief
 */
#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_BASIC_BLOCK_920_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_BASIC_BLOCK_920_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "dual_level_quant_batch_matmul_block.h"
#include "dual_level_quant_batch_matmul_cube_compute.h"
#include "dual_level_quant_batch_matmul_cube_compute_tools.h"
#include "dual_level_quant_batch_matmul_vec_compute_920.h"
#include "tool_arch35.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::LocalTensor;
using AscendC::TPosition;

namespace DualLevelQuantBatchMatmul::Arch35 {

LOCAL_TEMPLATE_CLASS_PARAMS
class DualLevelQuantMatmulBasicBlock920 {
public:
    __aicore__ inline DualLevelQuantMatmulBasicBlock920() = default;

    __aicore__ inline void Init(__gm__ x1Type* x1, __gm__ x2Type* x2, __gm__ x1Level0ScaleType* x1Level0Scale,
                                __gm__ x1Level1ScaleType* x1Level1Scale, __gm__ x2Level0ScaleType* x2Level0Scale,
                                __gm__ x2Level1ScaleType* x2Level1Scale, __gm__ biasType* bias, __gm__ yType* y,
                                const DualLevelQuantBatchMatmulBasicTilingData* tilingData);

    __aicore__ inline void SetAivToAic();

    __aicore__ inline void WaitAivToAic();

    __aicore__ inline void ComputeBasicBlock(DualLevelQbmmBasicBlockOffsetParams& offsetParams,
                                             L0CopyAndCalcParams& l0Params);

    __aicore__ inline void ComputeAicBasicBlock(DualLevelQbmmBasicBlockOffsetParams& offsetParams,
                                                L0CopyAndCalcParams& l0Params, FixL0CToDstParams& fixpParams,
                                                uint64_t kGmOffset);

    __aicore__ inline void ComputeAivBasicBlock(DualLevelQbmmBasicBlockOffsetParams& offsetParams,
                                                L0CopyAndCalcParams& l0Params, uint64_t kGmOffset);

    __aicore__ inline void End();

protected:
    static constexpr uint64_t AIV_SYNC_FLAG_DIST = 16;
    static constexpr uint64_t AIV_TO_AIC_ID = 2;
    static constexpr uint64_t AIC_TO_AIV_ID = 3;

    static constexpr uint64_t CV_LOOP_NUM = 2;
    static constexpr uint64_t C_TMP_FP32_UB_OFFSET_920 = 128;
    static constexpr uint64_t X2_LEVEL0_SCALE_K_UB_SIZE_920 = 16 * 512;

    static constexpr uint64_t FIXP_DST_4BANK_STRIDE = 256L; // fixp搬出N方向跨4bank stride固定256元素

    DualLevelQuantBatchMatmulVectorCompute920<x1Level0ScaleType, x2Level0ScaleType, biasType, yType, hasBias>
        vecCompute_;
    DualLevelQuantBatchMatmulCubeCompute<x1Type, x2Type, x1Level1ScaleType, x2Level1ScaleType, float, false>
        cubeCompute_;

    uint64_t cvLoopIdx_ = 0;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock920<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    __gm__ x1Type* x1, __gm__ x2Type* x2, __gm__ x1Level0ScaleType* x1Level0Scale,
    __gm__ x1Level1ScaleType* x1Level1Scale, __gm__ x2Level0ScaleType* x2Level0Scale,
    __gm__ x2Level1ScaleType* x2Level1Scale, __gm__ biasType* bias, __gm__ yType* y,
    const DualLevelQuantBatchMatmulBasicTilingData* tilingData)
{
    if ASCEND_IS_AIC {
        cubeCompute_.Init(x1, x2, x1Level1Scale, x2Level1Scale);
    } else {
        vecCompute_.Init(x1Level0Scale, x2Level0Scale, bias, y);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock920<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeBasicBlock(
    DualLevelQbmmBasicBlockOffsetParams& offsetParams, L0CopyAndCalcParams& l0Params)
{
    FixL0CToDstParams fixpParams;
    fixpParams.outNSize = FIXP_DST_4BANK_STRIDE;

    offsetParams.mUbSize = l0Params.mIterSize;
    offsetParams.x2Level0ScaleKUbSize = X2_LEVEL0_SCALE_K_UB_SIZE_920;

    for (uint64_t kGmOffset = 0; kGmOffset < offsetParams.kSize; kGmOffset += offsetParams.l0BaseK) {
        offsetParams.kGmOffset = kGmOffset;

        uint64_t kGmRealSize = kGmOffset + offsetParams.l0BaseK > offsetParams.kSize ? offsetParams.kSize - kGmOffset :
                                                                                       offsetParams.l0BaseK;
        l0Params.kL1Size = kGmRealSize;
        l0Params.kL0Size = kGmRealSize;

        if ASCEND_IS_AIC {
            ComputeAicBasicBlock(offsetParams, l0Params, fixpParams, kGmOffset);
        } else {
            ComputeAivBasicBlock(offsetParams, l0Params, kGmOffset);
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock920<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeAicBasicBlock(
    DualLevelQbmmBasicBlockOffsetParams& offsetParams, L0CopyAndCalcParams& l0Params, FixL0CToDstParams& fixpParams,
    uint64_t kGmOffset)
{
    cubeCompute_.WaitMte1ToMte2(offsetParams);
    cubeCompute_.CopyGmToL1(offsetParams, l0Params);
    // 先循环 m 方向，避免 fixpipe 写 UB 时与 VF 计算读 UB 的 bank 冲突
    for (uint64_t mL1Offset = 0; mL1Offset < l0Params.mL1Size; mL1Offset += l0Params.mIterSize) {
        l0Params.mL0Size = mL1Offset == 0 ? l0Params.mIterSize : l0Params.mL1Size - l0Params.mIterSize;
        for (uint64_t nL1Offset = 0; nL1Offset < l0Params.nL1Size; nL1Offset += l0Params.nIterSize) {
            l0Params.nL0Size = nL1Offset == 0 ? l0Params.nIterSize : l0Params.nL1Size - l0Params.nIterSize;
            CrossCoreWaitFlag<SyncMode::SYNC_MODE4, PIPE_FIX>(AIV_TO_AIC_ID);

            fixpParams.mL0Size = l0Params.mL0Size;
            fixpParams.nL0Size = l0Params.nL0Size;
            cubeCompute_.LaunchMatmul(mL1Offset, nL1Offset, 0, kGmOffset % offsetParams.level1ScaleKL1Size,
                                      vecCompute_.GetCTmpFp32Ub()[cvLoopIdx_ % CV_LOOP_NUM * C_TMP_FP32_UB_OFFSET_920],
                                      l0Params, fixpParams);
            CrossCoreSetFlag<SyncMode::SYNC_MODE4, PIPE_FIX>(AIC_TO_AIV_ID);
            cvLoopIdx_++;
        }
    }
    cubeCompute_.SetMte1ToMte2(offsetParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock920<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeAivBasicBlock(
    DualLevelQbmmBasicBlockOffsetParams& offsetParams, L0CopyAndCalcParams& l0Params, uint64_t kGmOffset)
{
    if (kGmOffset % offsetParams.x1Level0ScaleKUbSize == 0) {
        vecCompute_.WaitVToMte2ForX1();
        vecCompute_.CopyX1Level0ScaleToUb(offsetParams, l0Params);
    }
    if (kGmOffset % offsetParams.x2Level0ScaleKUbSize == 0) {
        vecCompute_.WaitVToMte2ForX2();
        vecCompute_.CopyX2Level0ScaleToUb(offsetParams, l0Params);
    }
    if (kGmOffset == 0) {
        vecCompute_.WaitVToMte2ForBias();
        vecCompute_.CopyBiasToUb(offsetParams, l0Params);
    }
    vecCompute_.SetAndWaitMte2ToV();
    for (uint64_t mL1Offset = 0; mL1Offset < l0Params.mL1Size; mL1Offset += l0Params.mIterSize) {
        l0Params.mL0Size = mL1Offset == 0 ? l0Params.mIterSize : l0Params.mL1Size - l0Params.mIterSize;
        uint64_t mIndex = mL1Offset / l0Params.mIterSize;
        // 循环内轴为 n 方向，n 方向 acc buffer 位于 UB 不同 bank，与 add buffer 交错，可缓解 UB bank 冲突
        for (uint64_t nL1Offset = 0; nL1Offset < l0Params.nL1Size; nL1Offset += l0Params.nIterSize) {
            l0Params.nL0Size = nL1Offset == 0 ? l0Params.nIterSize : l0Params.nL1Size - l0Params.nIterSize;
            uint64_t nIndex = nL1Offset / l0Params.nIterSize;
            if (kGmOffset == 0) {
                vecCompute_.WaitMte3ToV(mIndex, nIndex);
                vecCompute_.InitUb(mIndex, nIndex, nL1Offset);
            }
            CrossCoreWaitFlag<SyncMode::SYNC_MODE4, PIPE_V>(AIC_TO_AIV_ID);
            vecCompute_.AntiQuantCompute(nIndex, cvLoopIdx_, mL1Offset, nL1Offset, offsetParams, l0Params);
            CrossCoreSetFlag<SyncMode::SYNC_MODE4, PIPE_V>(AIV_TO_AIC_ID);
            cvLoopIdx_++;
            if (kGmOffset + offsetParams.l0BaseK >= offsetParams.kSize) {
                vecCompute_.CopyUbToGm(nIndex, mL1Offset, nL1Offset, offsetParams, l0Params);
                vecCompute_.SetMte3ToV(mIndex, nIndex);
            }
        }
    }

    if ((kGmOffset + offsetParams.l0BaseK) % offsetParams.x1Level0ScaleKUbSize == 0 ||
        kGmOffset + offsetParams.l0BaseK >= offsetParams.kSize) {
        vecCompute_.SetVToMte2ForX1();
    }
    if ((kGmOffset + offsetParams.l0BaseK) % offsetParams.x2Level0ScaleKUbSize == 0 ||
        kGmOffset + offsetParams.l0BaseK >= offsetParams.kSize) {
        vecCompute_.SetVToMte2ForX2();
    }
    if (kGmOffset == 0) {
        vecCompute_.SetVToMte2ForBias();
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock920<LOCAL_TEMPLATE_FUNC_PARAMS>::SetAivToAic()
{
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SyncMode::SYNC_MODE4, PIPE_V>(AIV_TO_AIC_ID);
        CrossCoreSetFlag<SyncMode::SYNC_MODE4, PIPE_V>(AIV_TO_AIC_ID);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock920<LOCAL_TEMPLATE_FUNC_PARAMS>::WaitAivToAic()
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SyncMode::SYNC_MODE4, PIPE_FIX>(AIV_TO_AIC_ID);
        CrossCoreWaitFlag<SyncMode::SYNC_MODE4, PIPE_FIX>(AIV_TO_AIC_ID);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void DualLevelQuantMatmulBasicBlock920<LOCAL_TEMPLATE_FUNC_PARAMS>::End()
{
    if ASCEND_IS_AIC {
        cubeCompute_.EndSync();
    } else {
        vecCompute_.WaitVectorFlag();
    }
}

} // namespace DualLevelQuantBatchMatmul::Arch35
#endif // DUAL_LEVEL_QUANT_BATCH_MATMUL_BASIC_BLOCK_920_H
