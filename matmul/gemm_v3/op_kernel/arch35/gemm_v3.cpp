/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gemm_v3.cpp
 * \brief
 */
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
#include "../../mat_mul_v3/arch35/mat_mul_tiling_data.h"
#include "gemm_v3_tiling_key.h"
#include "../../mat_mul_v3/arch35/mat_mul_pingpong_basic.h"
#endif

using namespace AscendC;

#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif
#ifndef DTYPE_A
#define DTYPE_A half
#endif
#ifndef DTYPE_B
#define DTYPE_B half
#endif
#ifndef DTYPE_C
#define DTYPE_C half
#endif
#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

template <int8_t API_LEVEL, int8_t A_TRANS, int8_t B_TRANS, int8_t BATCH_MODEL, int8_t MODEL, int8_t FULL_LOAD,
          int8_t L0C2OUT_MODEL>
__global__ __aicore__ void gemm_v3(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR yGM, GM_ADDR workspaceGM,
                                   GM_ADDR tilingGM)
{
    constexpr bool aTran = (A_TRANS == 1);
    constexpr bool bTran = (B_TRANS == 1);
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    REGISTER_TILING_DEFAULT(MatMulV3TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // GemmV3复用matmulV3 kernel和tilingKey，暂只支持aswt模板
    if constexpr (API_LEVEL == MAT_MUL_TENSOR_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD && MODEL == MAT_MUL_BASIC &&
                  L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
        // A5/A6 Blaze(TENSOR_LEVEL)：C 与 Y 原地累加，靠 INPLACE_ADD 的 fixpipe atomic add 实现 y = C + A@B
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, basicTilingData, tilingGM);
        using layoutA = AscendC::Std::conditional_t<aTran, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
        using layoutB = AscendC::Std::conditional_t<bTran, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
        using layoutC = asc::te::nd_ext_layout_ptn;
        MatmulV3Advanced::MatMulBasicKernel<DTYPE_A, DTYPE_B, DTYPE_C, DTYPE_BIAS, layoutA, layoutB, layoutC, 0, 0, 0,
                                            Blaze::Gemm::MatmulOutputMode::INPLACE_ADD>(aGM, bGM, nullptr, yGM, nullptr,
                                                                                        basicTilingData);
    }
#endif
}
