/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_fma_tiling.h
 * \brief fp32高精度模式vector FMA矩阵乘tiling模板
 */
#pragma once

#include "matmul_v3_base_tiling_advanced.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3FmaTiling : public MatMulV3BaseTiling {
public:
    MatMulV3FmaTiling(gert::TilingContext* context, MatMulTilingCfg& cfg) : MatMulV3BaseTiling(context, cfg) {};

    ~MatMulV3FmaTiling() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetTilingData(TilingResult& tiling) const override;

    ge::graphStatus GetTilingDataProcess(MatMulFmaTilingData& tilingData) const override;

private:
    // 计算体64宽子块单元(regbase 256B, fp32一次64个数); baseN四档均为其倍数
    static constexpr uint64_t FMA_SUB_UNIT = 64;
    // baseN=64档MR(m方向寄存器分块行数): (MR=8, NS=1), 寄存器17个
    static constexpr uint64_t FMA_MR_NARROW = 8;
    // baseN>=128档MR: (MR=4, NS=baseN/64), 寄存器(NS=4时)24个低于regbase干净预算25
    static constexpr uint64_t FMA_MR_WIDE = 4;
    // baseN档位上限(初选档与轮数计算基准, N超此值为多n-tile)
    static constexpr uint64_t FMA_BASE_N_MAX = 256;
    static constexpr uint64_t FMA_BASE_K = FMA_SUB_UNIT;
    // 满波tile行数baseM的对齐网格与下限(尾波tile行数baseMTail无对齐约束, 仅按慢行代价微调)
    static constexpr uint64_t FMA_BASE_M_ALIGN = 8;
    // UB预算(3510的TOTAL_UB_SIZE=248KB)
    static constexpr uint64_t FMA_UB_BUDGET = 248 * 1024;
    // 大B降档判定阈值(B工作集K*N*4B, 近似L2容量边界): B超此值时重读打GM, 降档以省B重读流量;
    // B驻L2时重读被L2吸收, 降档只白付A重读不触发
    static constexpr uint64_t FMA_SPREAD_B_LIMIT = 16UL * 1024UL * 1024UL;
    // transB staging行距(元素数, 与kernel的FMA_BSTRIDE一致)
    static constexpr uint64_t FMA_BSTRIDE = 72;
};
} // namespace matmul_v3_advanced
} // namespace optiling
