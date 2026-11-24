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
 * \file rotate_quant.cpp
 * \brief Unified kernel entry point for RotateQuant operator
 */

#include "kernel_operator.h"
#include "rotate_quant_tiling_data.h"
#include "blaze/rotate_quant_blaze_kernel.h"
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
#include "rotate_quant_basic_cmct.h"
#endif

using namespace AscendC;
using namespace RotateQuantNS;
using RotateQuantAptOpt::RotateQuantAptTilingData;

namespace {
template <bool EnableGrouped, class XType, class YType, class ScaleType>
__aicore__ inline void DispatchRotateQuant(GM_ADDR x, GM_ADDR rot, GM_ADDR alpha, GM_ADDR groupList, GM_ADDR y,
                                           GM_ADDR scale, GM_ADDR workspace, const RotateQuantAptTilingData& tilingData)
{
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    if constexpr (AscendC::Std::is_same_v<YType, hifloat4x2_t>) {
        if constexpr (!EnableGrouped) {
            RotateQuantCmctKernel<XType, YType, ScaleType, Cmct::Gemm::layout::RowMajor>(x, rot, alpha, y, scale,
                                                                                         workspace, tilingData);
        }
    } else
#endif
    {
        RotateQuantBlazeKernel<XType, YType, ScaleType, EnableGrouped>(x, rot, alpha, groupList, y, scale, tilingData);
    }
}
} // namespace

extern "C" __global__ __aicore__ void rotate_quant(GM_ADDR x, GM_ADDR rot, GM_ADDR alpha, GM_ADDR groupList, GM_ADDR y,
                                                   GM_ADDR scale, GM_ADDR workSpace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(RotateQuantAptTilingData);
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1); // DAV_9201/DAV_9202(AIC:AIV=1:1)
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    GET_TILING_DATA_WITH_STRUCT(RotateQuantAptTilingData, tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        DispatchRotateQuant<false, DTYPE_X, DTYPE_Y, DTYPE_SCALE>(x, rot, alpha, groupList, y, scale, workSpace,
                                                                  tilingData);
        return;
    }
    if (TILING_KEY_IS(1)) {
        DispatchRotateQuant<true, DTYPE_X, DTYPE_Y, DTYPE_SCALE>(x, rot, alpha, groupList, y, scale, workSpace,
                                                                 tilingData);
    }
}
