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

// Match FlatQuant: DAV_9201/DAV_9202 use CMCT; DAV_3510 products use Blaze.
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
#include "rotate_quant_basic_cmct.h"
#else
#define ROTATE_QUANT_BLAZE_KERNEL 1
#include "blaze/rotate_quant_blaze_kernel.h"
#endif

using namespace AscendC;
using namespace RotateQuantNS;
using RotateQuantAptOpt::RotateQuantAptTilingData;
#if !defined(ROTATE_QUANT_BLAZE_KERNEL)
using namespace Cmct;
using namespace Cmct::Gemm;
#endif

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

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    // Preserve the upstream CMCT workspace convention on DAV_9201/DAV_9202.
    if (TILING_KEY_IS(0)) {
        RotateQuantCmctKernel<DTYPE_X, DTYPE_Y, DTYPE_SCALE, layout::RowMajor>(x, rot, alpha, y, scale, workSpace,
                                                                               tilingData);
    }
#else
    __gm__ uint8_t* userWorkspace = GetUserWorkspace(workSpace);
    if (userWorkspace == nullptr) {
        return;
    }
    if (TILING_KEY_IS(0)) {
        RotateQuantBlazeKernel<DTYPE_X, DTYPE_Y, DTYPE_SCALE, RotateQuantNS::RowMajor, false>(
            x, rot, alpha, y, scale, userWorkspace, tilingData);
    } else if (TILING_KEY_IS(1)) {
        RotateQuantBlazeKernel<DTYPE_X, DTYPE_Y, DTYPE_SCALE, RotateQuantNS::RowMajor, true>(
            x, rot, alpha, y, scale, userWorkspace, tilingData, groupList);
    }
#endif
}
