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
 * \file full_mn_kernel_block_ctx.h
 * \brief BlockCtx: per-block partition result carrier (no loop variables).
 */

#ifndef FULL_MN_KERNEL_BLOCK_CTX_H
#define FULL_MN_KERNEL_BLOCK_CTX_H

#include "full_mn_kernel_common.h"

namespace full_mn {

template <typename FmapT, typename WeightT, typename OutputT, typename BiasT>
struct BlockCtx {
    // ===== Type aliases (compile-time) =====
    using FmapType = FmapT;
    using WeightType = WeightT;
    using OutputType = OutputT;
    using BiasType = BiasT;

    // dtype-dependent constants
    // GK0: FP16/BF16=16, FP32/HF32=8 (C0_SIZE=32, sizeof(half/bf16)=2, sizeof(float)=4)
    static constexpr uint32_t GK0 = static_cast<uint32_t>(C0_SIZE) / static_cast<uint32_t>(sizeof(WeightT));
    using L0cType = typename conv::GetDstType<FmapT>::Type;

    // ===== Block-partition indices =====
    uint32_t batchIdx = 0; // batch dimension partition index

    // ===== M direction partition result =====
    uint32_t mIdxStart = 0; // this core's M start offset in totalM
    uint32_t actualM = 0;   // actual M points (may be < singleCoreM for tail core)

    // ===== N direction partition result =====
    uint32_t actualCo = 0;  // actual Co channels (may be < singleCoreCo for tail core)
    uint32_t nIdxStart = 0; // this core's N start offset in cout
    uint32_t n1PerCore = 0; // N1 = CeilDiv(actualCo, GN0)
};

} // namespace full_mn

#endif // FULL_MN_KERNEL_BLOCK_CTX_H
