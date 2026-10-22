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
 * \file block_scheduler_qbmm_streamk.h
 * \brief
 */

#ifndef BLOCK_SCHEDULER_QBMM_STREAMK_H
#define BLOCK_SCHEDULER_QBMM_STREAMK_H

#include "./block_scheduler_utils.h"
#include "block_scheduler_policy.h"
#include "../utils/common_utils.h"

/*!
 * \file qbmm_block_scheduler.h
 * \brief
 */

namespace Cmct {
namespace Gemm {
namespace Block {

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool Transbatch_>
class BlockSchedulerQBMMStreamK {
public:
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t batch_{0};

    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t baseK_{0};

    int64_t mL1_{0};
    int64_t nL1_{0};
    int64_t kL1_{0};

    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    int64_t singleBatchCnt_{0};
    int64_t totalCnt_{0};

    int64_t blockIdx_{0};
    int64_t blockNum_{0};
    int64_t startBlockIdx_{0};
    int64_t endBlockIdx_{0};
    int64_t roundIdx_{0};
    int64_t round_{0};
    int64_t mTailTile_{0};
    int64_t nTailTile_{0};
    int64_t totalTailTile_{0};

    int64_t usedCoreNum_{0};
    int64_t batchIdx_{0};
    int64_t tileNum_{0};
    int64_t skKSingleCore_{0};

    uint8_t l1BuferNum_{0};
    uint8_t l0cDbatch_{1};
    uint8_t ubDbatch_{1};

    int64_t skKTileNum_{0};
    int64_t mTileIdx_{1};
    int64_t nTileIdx_{1};
    int64_t kTileIdx_{1};
    int64_t curKTileNum_{1};
    int64_t totalMNTileNumInDP_{0};

    // streamK 模板中不对尾块切分
    struct Params {
        int64_t usedCoreNum;
        int64_t baseM;
        int64_t baseN;
        int64_t baseK;

        int64_t skSingleCoreK;
        int64_t kL1;
    };

    const int64_t WINDOW_LEN = 4;

public:
    __aicore__ inline BlockSchedulerQBMMStreamK(const ProblemShape& shape, const Params& params)
    {
        usedCoreNum_ = params.usedCoreNum;

        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.k;
        batch_ = shape.b;

        baseM_ = static_cast<int64_t>(params.baseM);
        baseN_ = static_cast<int64_t>(params.baseN);

        // 来自tiling data
        skKSingleCore_ = params.skSingleCoreK;
        baseK_ = params.baseK;
        mL1_ = params.baseM;
        nL1_ = params.baseN;
        kL1_ = params.kL1; // stepk*baseK,

        mTileNum_ = Cmct::Gemm::CeilDiv(m_, baseM_);
        nTileNum_ = Cmct::Gemm::CeilDiv(n_, baseN_);
        skKTileNum_ = Cmct::Gemm::CeilDiv(k_, skKSingleCore_);

        int64_t tailMNTileNum = (mTileNum_ * nTileNum_) %
                                usedCoreNum_; // tail mCnt * nCnt num of SK, 尾块用streamk 切分
        tileNum_ = (mTileNum_ * nTileNum_ - tailMNTileNum) + tailMNTileNum * skKTileNum_; // 总共的块数是 dp
        totalMNTileNumInDP_ = mTileNum_ * nTileNum_ - tailMNTileNum;                      // 用于切K 的m/n 块数

        singleBatchCnt_ = mTileNum_ * nTileNum_;
        totalCnt_ = batch_ * singleBatchCnt_;

        blockIdx_ = AscendC::GetBlockIdx();
        blockNum_ = AscendC::GetBlockNum();

        roundIdx_ = 0;
        round_ = Cmct::Gemm::CeilDiv(singleBatchCnt_, blockNum_);
    }

    __aicore__ inline uint64_t GetL1BuferNum_() { return static_cast<uint64_t>(l1BuferNum_); }

    __aicore__ inline bool GetL0cDB() { return l0cDbatch_ > 1; }

    __aicore__ inline bool GetUbDB() { return ubDbatch_ > 1; }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTileL0Shape() { return {baseM_, baseN_, baseK_, 1}; }

    __aicore__ inline BlockCoord GetSingleCoreCoord(int64_t tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        return {mTileIdx_, nTileIdx_, kTileIdx_, 0};
    }

    __aicore__ inline BlockShape GetSingleCoreShape(int64_t tileIdx)
    {
        // 单核上分到的m n k 的大小
        UpdateMNTileIdx(tileIdx);
        int64_t tailL1M = m_ - (mTileNum_ - 1) * mL1_;
        int64_t tailL1N = n_ - (nTileNum_ - 1) * nL1_;
        int64_t tailSingleCoreK = k_ - (curKTileNum_ - 1) * skKSingleCore_;
        int64_t blkM = (mTileIdx_ == (mTileNum_ - 1)) ? tailL1M : mL1_;
        int64_t blkN = (nTileIdx_ == (nTileNum_ - 1)) ? tailL1N : nL1_;
        int64_t blkK = (kTileIdx_ == (curKTileNum_ - 1)) ? tailSingleCoreK : skKSingleCore_;
        return {blkM, blkN, blkK, 0};
    }

    __aicore__ inline void UpdateMNTileIdx(int64_t tileIdx)
    {
        // judge now in dp loop (kTileNum = 1) or in sk loop
        curKTileNum_ = CheckIsSkScene(tileIdx) ? skKTileNum_ : 1;
        int64_t mnIdxInCurLoop = 0;
        if (CheckIsSkScene(tileIdx)) { // SK scene
            kTileIdx_ = (tileIdx % usedCoreNum_) % curKTileNum_;
            mnIdxInCurLoop = (tileIdx % usedCoreNum_) / curKTileNum_ + totalMNTileNumInDP_;
        } else { // DP scene
            kTileIdx_ = 0;
            mnIdxInCurLoop = tileIdx / curKTileNum_;
        }

        int64_t mainWindow = AscendC::Std::min(WINDOW_LEN, mTileNum_);
        int64_t mainRow = mTileNum_ / mainWindow - 1UL;
        int64_t tailWindow = mTileNum_ - mainRow * mainWindow;

        int64_t rowIdx = mnIdxInCurLoop / nTileNum_ / mainWindow;
        if (rowIdx < mainRow) {
            mTileIdx_ = rowIdx * mainWindow + mnIdxInCurLoop % mainWindow;
            nTileIdx_ = (mnIdxInCurLoop / mainWindow) % nTileNum_;
        } else {
            rowIdx = mainRow;
            int64_t tailIndex = mnIdxInCurLoop - mainRow * mainWindow * nTileNum_;
            mTileIdx_ = mainRow * mainWindow + tailIndex % tailWindow;
            nTileIdx_ = (tailIndex / tailWindow) % nTileNum_;
        }
        // mod 2 means even row, need reverse scan
        if (rowIdx % 2 != 0UL) {
            nTileIdx_ = nTileNum_ - 1UL - nTileIdx_;
        }
    }

    __aicore__ inline int64_t GetTileNum()
    {
        if (batch_ > 1) {
            return totalCnt_;
        } else {
            int64_t finalRoundTileNum = singleBatchCnt_ % blockNum_;
            return singleBatchCnt_ - finalRoundTileNum + finalRoundTileNum * totalTailTile_;
        }
    }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetMNKTileNum()
    {
        return {mTileNum_, nTileNum_, skKTileNum_, 1};
    }

    __aicore__ inline int64_t GetTotalTileNum() { return tileNum_ * batch_; }

    __aicore__ inline int64_t GetCurKSingleCore(int64_t tileIdx)
    {
        return (CheckIsSkScene(tileIdx) ? skKSingleCore_ : k_);
    }

    __aicore__ inline bool CheckIsSkScene(int64_t tileIdx)
    {
        return (CeilDiv((tileIdx + 1), usedCoreNum_) == CeilDiv(tileNum_, usedCoreNum_)); // true is sk, false is dp
    }

    __aicore__ inline int64_t GetBlockNum(int64_t blockNum)
    {
        int64_t tilingBlockNum = 0;
        if (tileNum_ * batch_ < blockNum) {
            tilingBlockNum = tileNum_ * batch_;
        } else {
            tilingBlockNum = blockNum;
        }
        return tilingBlockNum;
    }

    __aicore__ inline void IncrementRoundIdx() { roundIdx_++; }

    __aicore__ inline void UpdateNextBatchBlockRoundParams()
    {
        startBlockIdx_ = (endBlockIdx_ + 1) % blockNum_;
        endBlockIdx_ = (singleBatchCnt_ + startBlockIdx_ - 1) % blockNum_;

        roundIdx_ = 0;
        round_ = Cmct::Gemm::CeilDiv(singleBatchCnt_, blockNum_);
        if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        }

        batchIdx_++;
    }
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool Transbatch_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Cmct::Gemm::BuiltInStreamKSchedulerQbmm,
                              TransA_, Transbatch_> {
    using SchedulerOp = BlockSchedulerQBMMStreamK<ProblemShape_, L1TileShape_, L0TileShape_, TransA_, Transbatch_>;
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif
