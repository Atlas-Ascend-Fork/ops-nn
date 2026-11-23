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
 * \file block_scheduler_fma.h
 * \brief
 */

#pragma once

#include "cmct/block/block_scheduler_utils.h"
#include "cmct/block/block_scheduler_policy.h"
#include "mat_mul_tiling_data.h"

namespace Cmct {
namespace Gemm {
namespace Block {
// TilingDataT_: tilingdata类型(FMA为MatMulFmaTilingData, 调度器只读usedCoreNum/baseM/baseN/baseK)
template <class ProblemShape_, class L1TileShape_, class L0TileShape_, class TilingDataT_>
class BlockSchedulerVectorBuiltIn {
public:
    int64_t usedCoreNum_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t baseM_{0};
    int64_t tailM_{0};
    int64_t baseN_{0};
    int64_t tailN_{0};
    int64_t baseK_{0};
    int64_t tailK_{0};
    int64_t loopK_{0};
    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    // 尾波均摊: 满波tile数/尾波tile行数(见MatMulFmaTilingData)
    int64_t mTileFull_{0};
    int64_t baseMTail_{0};

    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        const TilingDataT_* tilingData;
        // 尾波均摊(FMA专用, 非FMA调用方不传保持默认即uniform行为)
        uint32_t mTileFull = 0;
        uint32_t baseMTail = 0;
    };

public:
    __aicore__ inline BlockSchedulerVectorBuiltIn(const ProblemShape& shape, const Params& params)
    {
        usedCoreNum_ = params.tilingData->usedCoreNum;
        baseM_ = params.tilingData->baseM;
        baseN_ = params.tilingData->baseN;
        baseK_ = params.tilingData->baseK;
        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.k;
        // 尾波均摊: 满波tile[0, mTileFull)行数baseM, 其后尾波tile行数baseMTail(末tile为余量tailM);
        // mTileFull=0且baseMTail=baseM(或baseMTail=0)时为原uniform切分, tile序即波序(满波tile在前)
        mTileFull_ = static_cast<int64_t>(params.mTileFull);
        baseMTail_ = params.baseMTail > 0 ? static_cast<int64_t>(params.baseMTail) : baseM_;
        int64_t spreadRows = m_ - mTileFull_ * baseM_;
        if (mTileFull_ == 0 || spreadRows < 0) {
            // 防御: 非法输入退化为uniform
            mTileFull_ = 0;
            baseMTail_ = baseM_;
            spreadRows = m_;
        }
        const int64_t spreadTileNum = spreadRows > 0 ? CeilDiv(spreadRows, baseMTail_) : 0;
        mTileNum_ = mTileFull_ + spreadTileNum;
        tailM_ = spreadRows > 0 ? (spreadRows - (spreadTileNum - 1) * baseMTail_) : baseM_;
        tailN_ = n_ % baseN_ == 0 ? baseN_ : n_ % baseN_;
        tailK_ = k_ % baseK_ == 0 ? baseK_ : k_ % baseK_;
        loopK_ = CeilDiv(k_, baseK_);
        nTileNum_ = CeilDiv(n_, baseN_);
    }

    __aicore__ inline int64_t GetRealBlockNum() { return usedCoreNum_; }

    __aicore__ inline int64_t GetTileNum() { return mTileNum_ * nTileNum_; }

    __aicore__ inline TupleShape GetBlockInfo() { return {baseM_, baseN_, mTileNum_, nTileNum_}; }

    __aicore__ inline TupleShape GetTailInfo() { return {tailM_, tailN_, tailK_, loopK_}; }

    // tileIdxM的实际行数: 满波tile为baseM, 尾波tile为baseMTail(末tile为余量tailM)
    __aicore__ inline int64_t TileRows(int64_t tileIdxM)
    {
        if (tileIdxM < mTileFull_) {
            return baseM_;
        }
        return (tileIdxM == mTileNum_ - 1) ? tailM_ : baseMTail_;
    }

    // tileIdxM的行偏移(GM侧A/C的m向基准): 满波/尾波单独累计, 非均匀步进
    __aicore__ inline int64_t TileRowOffset(int64_t tileIdxM)
    {
        if (tileIdxM < mTileFull_) {
            return tileIdxM * baseM_;
        }
        return mTileFull_ * baseM_ + (tileIdxM - mTileFull_) * baseMTail_;
    }
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Cmct::Gemm::BuiltInFmaScheduler, TransA_,
                              TransB_> {
    using SchedulerOp = BlockSchedulerVectorBuiltIn<ProblemShape_, L1TileShape_, L0TileShape_, MatMulFmaTilingData>;
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
