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
 * \file kernel_qbmm_pergroup.h
 * \brief Coupled MC62 per-group Kernel composed with CMCT block and scheduler interfaces.
 */
#pragma once
#include "../utils/tuple_utils.h"
#include "../block/block_scheduler_qbmm.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {

template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
class QuantMmBatchX2PerGroup {
public:
    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<
        ProblemShape, typename BlockMmad::L1TileShape, typename BlockMmad::L0TileShape, BlockScheduler,
        BlockMmad::transA, BlockMmad::transB, typename BlockMmad::AType>::SchedulerOp;
    using BlockMmadOp = BlockMmad;
    using TileParams = typename BlockMmad::TileParams;
    using GroupResultParams = typename BlockMmad::GroupResultParams;
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;

    struct GroupParams {
        uint64_t groupSizeK{0};
        bool pairReduction{false};
    };

    struct BatchParams {
        static constexpr uint32_t DIM_COUNT = 4U;
        uint64_t a[DIM_COUNT], b[DIM_COUNT], c[DIM_COUNT];
        uint32_t order[DIM_COUNT];
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        GroupParams groupParams;
        BatchParams batchParams;
    };

    __aicore__ inline static bool CanUsePairAccumulation(uint64_t k, uint64_t groupSizeK, uint8_t l0cBufferCount)
    {
        return l0cBufferCount == BlockMmad::L0C_BUFFER_COUNT && groupSizeK != 0UL && k > groupSizeK;
    }

    __aicore__ inline void operator()(const Params& params, AscendC::TPipe* pipe) { Run(params, pipe); }

private:
    __aicore__ inline void Init(const Params& params)
    {
        params_ = &params;
        const uint64_t k = static_cast<uint64_t>(params.problemShape.k);
        groupCount_ = Cmct::Gemm::CeilDiv(k, params.groupParams.groupSizeK);
        mCount_ = Cmct::Gemm::CeilDiv(static_cast<uint64_t>(params.problemShape.m), params.mmadParams.baseM);
        nCount_ = Cmct::Gemm::CeilDiv(static_cast<uint64_t>(params.problemShape.n), params.mmadParams.baseN);
    }

    __aicore__ inline void Run(const Params& params, AscendC::TPipe* pipe)
    {
        if (params.problemShape.m <= 0 || params.problemShape.n <= 0 || params.problemShape.b <= 0 ||
            params.mmadParams.baseM == 0U || params.mmadParams.baseN == 0U || params.groupParams.groupSizeK == 0UL) {
            return;
        }
        Init(params);
        mmadOp_.Init(params.mmadParams, pipe);
        epilogue_.Init(params.epilogueParams, pipe);
        Process(epilogue_);
    }

    __aicore__ inline void InitSync()
    {
        for (uint8_t slot = 0U; slot < GroupResultParams::UB_RING_DEPTH; ++slot) {
            AscendC::TQueSync<PIPE_V, PIPE_FIX> credit;
            credit.SetFlag(static_cast<uint16_t>(GroupResultParams::VEC2CUBE_FLAG + slot));
        }
    }

    __aicore__ inline void EndSync()
    {
        for (uint8_t slot = 0U; slot < GroupResultParams::UB_RING_DEPTH; ++slot) {
            AscendC::TQueSync<PIPE_V, PIPE_FIX> credit;
            credit.WaitFlag(static_cast<uint16_t>(GroupResultParams::VEC2CUBE_FLAG + slot));
        }
    }

    __aicore__ inline void Process(BlockEpilogue& aiv)
    {
        // Idle cores also seed and drain both credits, without waiting for data.
        InitSync();
        const uint64_t blockNum = AscendC::GetBlockNum();
        uint64_t partitions = 1UL;
        uint64_t workCount = mCount_ * nCount_;
        uint64_t batches = static_cast<uint64_t>(params_->problemShape.b);
        ProblemShape schedulerShape = params_->problemShape;
        if constexpr (BlockMmad::LOAD_MODE == B_FULL_LOAD_MODE) {
            partitions = Cmct::Gemm::Min<uint64_t>(mCount_, Cmct::Gemm::CeilDiv(blockNum, nCount_));
            workCount = nCount_ * partitions;
            batches = 1UL;
        } else {
            if (params_->mmadParams.broadcastReuse) {
                partitions = Cmct::Gemm::Min<uint64_t>(batches, Cmct::Gemm::CeilDiv(blockNum, workCount));
                workCount *= partitions;
                batches = 1UL;
            }
        }
        if (BlockMmad::LOAD_MODE != NONE_FULL_LOAD_MODE || params_->mmadParams.broadcastReuse) {
            // Flatten resident work; A uses a fixed M panel, B/broadcast own contiguous ranges.
            schedulerShape = {1, static_cast<int64_t>(workCount), 1, 1};
        }
        BlockSchedulerOp scheduler(schedulerShape, params_->schParams);
        for (uint64_t batch = 0UL; batch < batches; ++batch) {
            typename BlockSchedulerOp::BlockCoord coord{};
            while (scheduler.GetTileIdx(coord)) {
                ProcessTileRange(aiv, coord, batch, partitions);
            }
            if (batch + 1UL < batches) {
                scheduler.UpdateNextBatchBlockRoundParams();
            }
        }
        EndSync();
        // Return ring credits before draining output and releasing events.
        aiv.End();
        mmadOp_.End();
    }

    __aicore__ inline void ProcessTileRange(BlockEpilogue& aiv, const typename BlockSchedulerOp::BlockCoord& coord,
                                            uint64_t batch, uint64_t partitions)
    {
        uint64_t mIndex = static_cast<uint64_t>(Get<MNK_M>(coord));
        uint64_t nIndex = static_cast<uint64_t>(Get<MNK_N>(coord));
        uint64_t partition = 0UL, extent = 1UL;
        if constexpr (BlockMmad::LOAD_MODE == B_FULL_LOAD_MODE) {
            partition = nIndex % partitions;
            nIndex /= partitions;
            extent = mCount_;
        } else if constexpr (BlockMmad::LOAD_MODE == A_FULL_LOAD_MODE) {
            // mCount_ divides blockNum: each core keeps the same A panel across N rounds.
            mIndex = nIndex % mCount_;
            nIndex /= mCount_;
        } else if (params_->mmadParams.broadcastReuse) {
            partition = nIndex % partitions;
            nIndex /= partitions;
            extent = static_cast<uint64_t>(params_->problemShape.b);
            mIndex = nIndex / nCount_;
            nIndex %= nCount_;
        }
        const uint64_t count = extent / partitions;
        const uint64_t extra = extent % partitions;
        const uint64_t begin = partition * count + Cmct::Gemm::Min<uint64_t>(partition, extra);
        const uint64_t end = begin + count + (partition < extra ? 1UL : 0UL);
        for (uint64_t item = begin; item < end; ++item) {
            if constexpr (BlockMmad::LOAD_MODE == B_FULL_LOAD_MODE) {
                mIndex = item;
            } else if (params_->mmadParams.broadcastReuse) {
                batch = item;
            }
            const TileParams tile = GetTileParams(batch, mIndex, nIndex);
            ProcessSingleTile(aiv, tile);
        }
    }

    __aicore__ inline TileParams GetTileParams(uint64_t batch, uint64_t mIndex, uint64_t nIndex) const
    {
        const auto& p = params_->batchParams;
        uint64_t coords[BatchParams::DIM_COUNT]{};
        for (int32_t i = static_cast<int32_t>(BatchParams::DIM_COUNT) - 1; i >= 0; --i) {
            const uint32_t axis = p.order[i];
            coords[axis] = batch % p.c[axis];
            batch /= p.c[axis];
        }
        TileParams tile{};
        for (uint32_t i = 0U; i < BatchParams::DIM_COUNT; ++i) {
            tile.batchA = tile.batchA * p.a[i] + (p.a[i] == 1UL ? 0UL : coords[i]);
            tile.batchB = tile.batchB * p.b[i] + (p.b[i] == 1UL ? 0UL : coords[i]);
            tile.batchC = tile.batchC * p.c[i] + coords[i];
        }
        tile.mOffset = mIndex * params_->mmadParams.baseM;
        tile.nOffset = nIndex * params_->mmadParams.baseN;
        tile.actualM = static_cast<uint32_t>(Cmct::Gemm::Min<uint64_t>(
            params_->mmadParams.baseM, static_cast<uint64_t>(params_->problemShape.m) - tile.mOffset));
        tile.actualN = static_cast<uint32_t>(Cmct::Gemm::Min<uint64_t>(
            params_->mmadParams.baseN, static_cast<uint64_t>(params_->problemShape.n) - tile.nOffset));
        return tile;
    }

    __aicore__ inline void ProcessSingleTile(BlockEpilogue& aiv, const TileParams& tile)
    {
        if (groupCount_ == 0UL) {
            return;
        }
        ProcessTileGroups(aiv, tile, params_->groupParams.groupSizeK);
        aiv.AddBiasAndCopyOut(tile);
    }

    __aicore__ inline uint32_t GetGroupKSize(uint64_t group, uint64_t groupSizeK) const
    {
        const uint64_t kOffset = group * groupSizeK;
        return static_cast<uint32_t>(
            Cmct::Gemm::Min<uint64_t>(groupSizeK, static_cast<uint64_t>(params_->problemShape.k) - kOffset));
    }

    __aicore__ inline void ProcessTileGroups(BlockEpilogue& aiv, const TileParams& tile, uint64_t groupSizeK)
    {
        const bool pairReduction = params_->groupParams.pairReduction;
        const uint64_t groupsPerBatch = pairReduction ? GroupResultParams::GROUPS_PER_BATCH : 1UL;
        const uint64_t resultSlots = pairReduction ? GroupResultParams::PAIR_RESULT_SLOTS : 1UL;
        uint64_t produceGroup = 0UL;
        const uint64_t primeGroups = Cmct::Gemm::Min<uint64_t>(groupCount_, resultSlots);
        for (; produceGroup < primeGroups; ++produceGroup) {
            RunGroupMmad(aiv, tile, produceGroup, groupSizeK);
        }
        for (uint64_t firstGroup = 0UL; firstGroup < groupCount_; firstGroup += groupsPerBatch) {
            const bool hasSecond = groupsPerBatch > 1UL && firstGroup + 1UL < groupCount_;
            aiv.ProcessGroupResults(firstGroup, hasSecond, tile.actualM, tile.actualN);
            const uint64_t refillEnd = Cmct::Gemm::Min<uint64_t>(groupCount_, produceGroup + groupsPerBatch);
            // FIX waits for Vector's credit before overwriting the refilled batch.
            for (; produceGroup < refillEnd; ++produceGroup) {
                RunGroupMmad(aiv, tile, produceGroup, groupSizeK);
            }
        }
    }

    __aicore__ inline void RunGroupMmad(BlockEpilogue& aiv, const TileParams& tile, uint64_t group, uint64_t groupSizeK)
    {
        // VDEQF16 remains per group even when Vector reduces a pair.
        const uint32_t groupK = GetGroupKSize(group, groupSizeK);
        mmadOp_(tile, group, groupK);
        mmadOp_.FixGroupResult(tile, group, aiv.GetGroupResultTensor(group));
    }

    const Params* params_{nullptr};
    BlockMmad mmadOp_;
    BlockEpilogue epilogue_;
    uint64_t groupCount_{0};
    uint64_t mCount_{0}, nCount_{0};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct
