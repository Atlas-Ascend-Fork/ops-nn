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
 * \file swiglu_hi_f4_quant.cpp
 * \brief swiglu_hi_f4_quant kernel enter
 */

#include "kernel_operator.h"
#include "arch92/swiglu_hi_f4_quant_regbase.h"
#include "arch92/swiglu_hi_f4_quant_tiling_data.h"

#define TILING_KEY_HIF4_GROUP_NONE_SWIGLU_NEG1_QUANT_NEG1 1000
#define TILING_KEY_HIF4_GROUP_NONE_SWIGLU_NEG1_QUANT_NEG2 1001
#define TILING_KEY_HIF4_GROUP_NONE_SWIGLU_NEG2_QUANT_NEG1 1010
#define TILING_KEY_HIF4_GROUP_NONE_SWIGLU_NEG2_QUANT_NEG2 1011
#define TILING_KEY_HIF4_GROUP_INT32_SWIGLU_NEG1_QUANT_NEG1 1100
#define TILING_KEY_HIF4_GROUP_INT32_SWIGLU_NEG1_QUANT_NEG2 1101
#define TILING_KEY_HIF4_GROUP_INT32_SWIGLU_NEG2_QUANT_NEG1 1110
#define TILING_KEY_HIF4_GROUP_INT32_SWIGLU_NEG2_QUANT_NEG2 1111
#define TILING_KEY_HIF4_GROUP_INT64_SWIGLU_NEG1_QUANT_NEG1 1200
#define TILING_KEY_HIF4_GROUP_INT64_SWIGLU_NEG1_QUANT_NEG2 1201
#define TILING_KEY_HIF4_GROUP_INT64_SWIGLU_NEG2_QUANT_NEG1 1210
#define TILING_KEY_HIF4_GROUP_INT64_SWIGLU_NEG2_QUANT_NEG2 1211

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace AscendC;

template <typename ST, typename DT, typename GT, bool hasGroupIndex>
__aicore__ inline void RunSwigluHiF4Quant(int64_t scaleAlg, GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale,
                                          const SwigluHiF4QuantTilingData* tilingData, TPipe* pipe)
{
    if (scaleAlg == 1) {
        SwigluHiF4Quant::SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, 1> op(pipe);
        op.Init(x, group_index, y, scale, tilingData);
        op.Process();
    } else {
        SwigluHiF4Quant::SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, 0> op(pipe);
        op.Init(x, group_index, y, scale, tilingData);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void swiglu_hi_f4_quant(GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SwigluHiF4QuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwigluHiF4QuantTilingData, tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
#if (__NPU_ARCH__ == 3101)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif
    TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_HIF4_GROUP_NONE_SWIGLU_NEG1_QUANT_NEG1)) {
        RunSwigluHiF4Quant<DTYPE_X, DTYPE_Y, int32_t, false>(tilingData.scaleAlg, x, group_index, y, scale, &tilingData,
                                                             &pipe);
    } else if (TILING_KEY_IS(TILING_KEY_HIF4_GROUP_INT32_SWIGLU_NEG1_QUANT_NEG1)) {
        RunSwigluHiF4Quant<DTYPE_X, DTYPE_Y, int32_t, true>(tilingData.scaleAlg, x, group_index, y, scale, &tilingData,
                                                            &pipe);
    } else if (TILING_KEY_IS(TILING_KEY_HIF4_GROUP_INT64_SWIGLU_NEG1_QUANT_NEG1)) {
        RunSwigluHiF4Quant<DTYPE_X, DTYPE_Y, int64_t, true>(tilingData.scaleAlg, x, group_index, y, scale, &tilingData,
                                                            &pipe);
    }
#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
    return;
}
