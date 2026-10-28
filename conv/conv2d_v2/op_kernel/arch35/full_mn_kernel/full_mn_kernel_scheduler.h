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
 * \file full_mn_kernel_scheduler.h
 * \brief BlockScheduler: stateless block-partition algorithm + fmap H range calculation.
 */

#ifndef FULL_MN_KERNEL_SCHEDULER_H
#define FULL_MN_KERNEL_SCHEDULER_H

#include "full_mn_kernel_common.h"
#include "full_mn_kernel_block_ctx.h"

namespace full_mn {

template <typename BlockCtx, typename TilingData>
struct BlockScheduler {
    __aicore__ inline BlockCtx operator()(const TilingData& tiling) const
    {
        BlockCtx bc;
        uint32_t blockIdx = GetBlockIdx();

        // Decompose: batch → N → M (M_MODE + MITER_FIRST)
        uint32_t nxm = tiling.nDim * tiling.hoDim;
        bc.batchIdx = blockIdx / nxm * tiling.singleCoreBatch;
        uint32_t rem = blockIdx % nxm;
        uint32_t nIdx = rem / tiling.hoDim;
        uint32_t mIdx = rem % tiling.hoDim;

        // M direction
        uint32_t totalM = static_cast<uint32_t>(tiling.hout * tiling.wout);
        bc.mIdxStart = mIdx * static_cast<uint32_t>(tiling.singleCoreHo);
        bc.actualM = 0;
        if (bc.actualM >= totalM) {
            bc.actualM = 0;
        } else if (bc.mIdxStart + static_cast<uint32_t>(tiling.singleCoreHo) > totalM) {
            bc.actualM = totalM - bc.mIdxStart;
        } else {
            bc.actualM = static_cast<uint32_t>(tiling.singleCoreHo);
        }

        // N direction (tail core handling)
        uint32_t coutStart = nIdx * tiling.singleCoreCo;
        bc.actualCo = 0;
        if (coutStart >= tiling.cout) {
            bc.actualCo = 0;
        } else if (coutStart + tiling.singleCoreCo > tiling.cout) {
            bc.actualCo = tiling.cout - coutStart;
        } else {
            bc.actualCo = tiling.singleCoreCo;
        }
        bc.nIdxStart = coutStart;
        bc.n1PerCore = static_cast<uint32_t>(CeilDiv(bc.actualCo, N0));

        return bc;
    }
};

} // namespace full_mn

#endif // FULL_MN_KERNEL_SCHEDULER_H
