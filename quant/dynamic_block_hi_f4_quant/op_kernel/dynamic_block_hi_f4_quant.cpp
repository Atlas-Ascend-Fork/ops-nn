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
 * \file dynamic_block_hi_f4_quant.cpp
 * \brief Kernel entry for DynamicBlockHiF4Quant
 */

#include "arch35/dynamic_block_hi_f4_quant_struct.h"
#include "arch35/dynamic_block_hi_f4_quant_tilingdata.h"
#include "arch35/dynamic_block_hi_f4_quant_kernel.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60 // 控制饱和模式

using namespace DynamicBlockHiF4Quant;
using namespace DynamicBlockHiF4QuantOp;

namespace {
// round_mode 模板参数 -> AscendC::RoundMode 映射
// 本算子 round_mode 仅支持 "rint" 与 "trunc" 两种模式
template <uint64_t roundMode>
struct RoundModeMapper {
    static constexpr AscendC::RoundMode value = []() {
        if constexpr (roundMode == TPL_RINT) {
            return AscendC::RoundMode::CAST_RINT;
        } else if constexpr (roundMode == TPL_TRUNC) {
            return AscendC::RoundMode::CAST_TRUNC;
        } else {
            return AscendC::RoundMode::CAST_RINT;
        }
    }();
};
} // namespace

template <uint64_t roundMode>
__global__ __aicore__ void dynamic_block_hi_f4_quant(GM_ADDR x, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(DynamicBlockHiF4QuantTilingData);

#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    GET_TILING_DATA_WITH_STRUCT(DynamicBlockHiF4QuantTilingData, tilingData, tiling);
    TPipe pipe;
    constexpr AscendC::RoundMode ascendcRoundMode = RoundModeMapper<roundMode>::value;
    DynamicBlockHiF4QuantKernel<DTYPE_X, DTYPE_Y, ascendcRoundMode> op(&tilingData, &pipe);
    op.Init(x, y, scale1, scale2);
    op.Process();

#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
