/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file unique_consecutive_multi_core_kernel.h
 * \brief
 */

#ifndef UNIQUE_CONSECUTIVE_MULTI_CORE_KERNEL_H
#define UNIQUE_CONSECUTIVE_MULTI_CORE_KERNEL_H

#include "unique_consecutive_helper.h"

constexpr int64_t MAGIC_GM_PAGE_SIZE = 128;
constexpr int32_t BUFFER_NUM_MULTI = 1;

template <typename T, typename T1, typename DTYPE_COUNT, bool COUNT_OUT, bool ISINT64>
class UniqueConsecutiveMutilCoreKerenl {
public:
    __aicore__ inline UniqueConsecutiveMutilCoreKerenl(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR count, GM_ADDR shape_out, GM_ADDR workspace,
                                const UniqueConsecutiveTilingData* tilingData)
    {
        coreIdx_ = GetBlockIdx();
        isFinalCore_ = (coreIdx_ == GetBlockNum() - 1);

        coreTileLength_ = (isFinalCore_) ? tilingData->tileLengthTailCore : tilingData->tileLengthPerCore;
        totalSize_ = tilingData->totalSize;
        tileLengthPerCore_ = tilingData->tileLengthPerCore;
        idxQueueSize_ = tilingData->idxQueueSize;
        adjUbTileLength_ = tilingData->adjUbTileLength;
        ubTileLength_ = adjUbTileLength_ - 1;

        xGm_.SetGlobalBuffer((__gm__ T*)(x) + tilingData->tileLengthPerCore * coreIdx_);
        yGm_.SetGlobalBuffer((__gm__ T*)(y));

        if (isFinalCore_) {
            shapeGm_.SetGlobalBuffer((__gm__ uint64_t*)shape_out);
        }

        pipe_->InitBuffer(xQueue_, BUFFER_NUM_MULTI, tilingData->valueQueueSize);
        pipe_->InitBuffer(yQueue_, BUFFER_NUM_MULTI, tilingData->valueQueueSize);

        countGm_.SetGlobalBuffer((__gm__ int32_t*)(count));
        if constexpr (COUNT_OUT) {
            pipe_->InitBuffer(countQueue_, BUFFER_NUM_MULTI, tilingData->countQueueSize);
            pipe_->InitBuffer(idxQueue_, BUFFER_NUM_MULTI, tilingData->idxQueueSize);
        }

        pipe_->InitBuffer(collectingCntBuf_, tilingData->collectingCntBufSize);
        pipe_->InitBuffer(offsetCntBuf_, tilingData->offsetCntBufSize);
        pipe_->InitBuffer(prevIdxBuf_, tilingData->prevIdxBufSize);
        pipe_->InitBuffer(shapeBuf_, tilingData->shapeBufSize);

        coreCollectWorkspaceGm_.SetGlobalBuffer((__gm__ int64_t*)(workspace));
        idxGm_.SetGlobalBuffer((__gm__ int32_t*)(idx) + tilingData->tileLengthPerCore * coreIdx_);
    }

    __aicore__ inline void Process()
    {
        if (isFinalCore_) {
            LocalTensor<uint64_t> shapeTensor = shapeBuf_.Get<uint64_t>();
            Duplicate(shapeTensor, (uint64_t)1, SHAPE_LEN);
        }

        int64_t coreCollectNums = 0;
        int64_t firstUniqueIdx = (coreIdx_ == 0) ? 0 : -1;
        ProcessCollecting(coreCollectNums, firstUniqueIdx);

        PipeBarrier<PIPE_ALL>();
        SyncAll();

        if (coreCollectNums == 0 && !isFinalCore_) {
            return;
        }

        int64_t yOffset = 0;
        int64_t nextUniqueIdx = totalSize_;
        FindOffset(yOffset, nextUniqueIdx);

        if (coreCollectNums > 0) {
            ProcessMerging(yOffset, nextUniqueIdx, coreCollectNums);
        }

        if (isFinalCore_) {
            coreCollectNums += yOffset;
            CopyOutShape(coreCollectNums, coreCollectNums);
        }
    }

    __aicore__ inline void ProcessMerging(int64_t copyOutOffset, int64_t nextUniqueIdx, int64_t coreCollectNums)
    {
        int64_t ubLoops = CEIL_DIV(coreTileLength_, ubTileLength_) - 1;
        int64_t ubTailLength = coreTileLength_ - ubTileLength_ * ubLoops;

        int64_t offsetXGm = 0;
        int64_t gatherCnt = 0;
        int64_t prevBoundary = -1;
        int64_t lastHeadIdx = (coreIdx_ == 0) ? 0 : -1;
        int64_t remainingCandidates = coreCollectNums - ((coreIdx_ == 0) ? 1 : 0);

        for (int64_t i = 0; i <= ubLoops; ++i) {
            bool addFirst = (coreIdx_ == 0 && i == 0);
            int64_t sourceStart = 0;
            int64_t actualLen = (i == ubLoops) ? ubTailLength : ubTileLength_;
            int64_t windowLen = CopyInWindow(offsetXGm, actualLen, !addFirst, sourceStart);
            if (addFirst) {
                CollectUniqueAndCount<true>(sourceStart, windowLen, gatherCnt, prevBoundary, lastHeadIdx, copyOutOffset,
                                            nextUniqueIdx, remainingCandidates);
            } else {
                CollectUniqueAndCount<false>(sourceStart, windowLen, gatherCnt, prevBoundary, lastHeadIdx,
                                             copyOutOffset, nextUniqueIdx, remainingCandidates);
            }
            offsetXGm += ubTileLength_;
            copyOutOffset += gatherCnt;
        }
        if constexpr (COUNT_OUT) {
            if (coreIdx_ == 0 && coreCollectNums == 1) {
                CopyOutLastCount(copyOutOffset - 1, nextUniqueIdx - lastHeadIdx);
            }
        }
    }

    __aicore__ inline void ProcessCollecting(int64_t& coreCollectNums, int64_t& firstUniqueIdx)
    {
        int64_t ubLoops = CEIL_DIV(coreTileLength_, ubTileLength_) - 1;
        int64_t ubTailLength = coreTileLength_ - ubTileLength_ * ubLoops;

        int64_t offsetXGm = 0;
        int64_t gatherCnt = 0;

        for (int64_t i = 0; i <= ubLoops; ++i) {
            bool addFirst = (coreIdx_ == 0 && i == 0);
            int64_t sourceStart = 0;
            int64_t actualLen = (i == ubLoops) ? ubTailLength : ubTileLength_;
            int64_t windowLen = CopyInWindow(offsetXGm, actualLen, !addFirst, sourceStart);
            if (addFirst) {
                CollectNumsAndFirstIdx<true>(sourceStart, windowLen, gatherCnt, firstUniqueIdx);
            } else {
                CollectNumsAndFirstIdx<false>(sourceStart, windowLen, gatherCnt, firstUniqueIdx);
            }
            offsetXGm += ubTileLength_;
            coreCollectNums += gatherCnt;
        }

        CopyOutCnt2Workspace(coreCollectNums, firstUniqueIdx);
    }

    template <bool ADD_FIRST>
    __aicore__ inline void CollectNumsAndFirstIdx(int64_t sourceStart, int64_t nums, int64_t& gatherCnt,
                                                  int64_t& firstUniqueIdx)
    {
        LocalTensor<T> xLocal = xQueue_.template DeQue<T>();
        LocalTensor<int32_t> outTensor = yQueue_.template AllocTensor<int32_t>();
        uint64_t reduceCntValue = -1;
        CountAdjacentNe<T, T1, ADD_FIRST>(outTensor, xLocal, nums, reduceCntValue);
        gatherCnt = reduceCntValue;
        yQueue_.FreeTensor(outTensor);

        if constexpr (COUNT_OUT) {
            if (gatherCnt > 0) {
                if constexpr (ISINT64) {
                    LocalTensor<int32_t> idxLocal = idxQueue_.template AllocTensor<int32_t>();
                    uint64_t candidateCount = -1;
                    uint64_t alignPosition = idxQueueSize_ / MIDPOINT_DIVIDER / sizeof(int32_t);
                    CollectPostUniqueIdx<T, T1>(idxLocal, xLocal, 0, nums, nums, candidateCount, alignPosition);
                    if (candidateCount > 0) {
                        LocalTensor<int64_t> idxLocalInt64 = idxLocal.template ReinterpretCast<int64_t>();
                        CastAndAddsOffsets(idxLocalInt64, idxLocal, candidateCount, alignPosition, sourceStart);
                        SimpleNativePipeSync<HardEvent::V_S>();
                        if (firstUniqueIdx < 0) {
                            firstUniqueIdx = idxLocalInt64.GetValue(0) + 1;
                        }
                    }
                    idxQueue_.FreeTensor(idxLocal);
                } else {
                    LocalTensor<int32_t> idxLocal = idxQueue_.template AllocTensor<int32_t>();
                    uint64_t candidateCount = -1;
                    CollectPostUniqueIdx<T, T1>(idxLocal, xLocal, static_cast<int32_t>(sourceStart), totalSize_, nums,
                                                candidateCount, START_POSITION);
                    if (candidateCount > 0) {
                        SimpleNativePipeSync<HardEvent::V_S>();
                        if (firstUniqueIdx < 0) {
                            firstUniqueIdx = static_cast<int64_t>(idxLocal.GetValue(0)) + 1;
                        }
                    }
                    idxQueue_.FreeTensor(idxLocal);
                }
            }
        }
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyInX(int64_t offset, int64_t copyLen)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>((copyLen) * sizeof(T)), 0, 0, 0};

        LocalTensor<T> xLocal = xQueue_.template AllocTensor<T>();
        DataCopyPad(xLocal, xGm_[offset], dataCopyParams, padParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline int64_t CopyInWindow(int64_t offset, int64_t actualLen, bool hasPrev, int64_t& sourceStart)
    {
        int64_t sourceOffset = offset - (hasPrev ? 1 : 0);
        int64_t windowLen = actualLen + (hasPrev ? 1 : 0);
        sourceStart = coreIdx_ * tileLengthPerCore_ + sourceOffset;
        CopyInX(sourceOffset, windowLen);
        return windowLen;
    }

    __aicore__ inline void CopyOutCnt2Workspace(int64_t coreCollectNums, int64_t firstUniqueIdx)
    {
        LocalTensor<int64_t> countLocal = collectingCntBuf_.Get<int64_t>();
        countLocal.SetValue(0, coreCollectNums);
        int64_t copyNums = 1;
        if constexpr (COUNT_OUT) {
            countLocal.SetValue(1, firstUniqueIdx);
            copyNums = 2;
        }
        SimpleNativePipeSync<HardEvent::S_MTE3>();
        Copy2GmEx<int64_t>(coreCollectWorkspaceGm_[CORE_COLLECT_NUM_OFFSET * coreIdx_], countLocal, 1, copyNums, 0, 0);
    }

    __aicore__ inline void CopyInCounts(LocalTensor<int64_t>& countLocal)
    {
        DataCopyPadExtParams<int64_t> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = GetBlockNum();
        dataCopyParams.blockLen = sizeof(int64_t);
        dataCopyParams.srcStride = MAGIC_GM_PAGE_SIZE - sizeof(int64_t);
        dataCopyParams.dstStride = 0;

        DataCopyPad<int64_t, PaddingMode::Compact>(countLocal, coreCollectWorkspaceGm_, dataCopyParams, padParams);
    }

    __aicore__ inline void FindOffset(int64_t& yOffset, int64_t& nextUniqueIdx)
    {
        LocalTensor<int64_t> countLocal = collectingCntBuf_.Get<int64_t>();
        LocalTensor<int64_t> offsetLocal = offsetCntBuf_.Get<int64_t>();
        CopyInCounts(countLocal);
        SimpleNativePipeSync<HardEvent::MTE2_V>();
        ReduceSum(offsetLocal, countLocal, offsetLocal, coreIdx_);
        SimpleNativePipeSync<HardEvent::V_S>();
        yOffset = offsetLocal.GetValue(0);
        if constexpr (COUNT_OUT) {
            nextUniqueIdx = totalSize_;
            for (int64_t i = coreIdx_ + 1; i < GetBlockNum(); ++i) {
                int64_t coreCount = countLocal.GetValue(i);
                if (coreCount != 0) {
                    CopyInCoreFirst(i, nextUniqueIdx);
                    break;
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInCoreFirst(int64_t nextCoreIdx, int64_t& nextUniqueIdx)
    {
        LocalTensor<int64_t> nextIdxLocal = prevIdxBuf_.Get<int64_t>();
        DataCopyPadExtParams<int64_t> padParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sizeof(int64_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(nextIdxLocal,
                    coreCollectWorkspaceGm_[CORE_COLLECT_NUM_OFFSET * nextCoreIdx + CORE_FIRST_UNIQUE_IDX_OFFSET],
                    dataCopyParams, padParams);
        SimpleNativePipeSync<HardEvent::MTE2_S>();
        nextUniqueIdx = nextIdxLocal.GetValue(0);
    }

    template <typename COUNT_TYPE>
    __aicore__ inline void ProcessCount(LocalTensor<COUNT_TYPE>& idxLocal, uint64_t candidateCount,
                                        int64_t copyOutOffset, int64_t& prevBoundary, int64_t& lastHeadIdx,
                                        int64_t nextUniqueIdx, int64_t& remainingCandidates)
    {
        if (candidateCount == 0) {
            return;
        }

        remainingCandidates -= candidateCount;
        bool writeLast = (remainingCandidates == 0);
        COUNT_TYPE lastBoundary = idxLocal.GetValue(candidateCount - 1);
        lastHeadIdx = static_cast<int64_t>(lastBoundary) + 1;
        if (coreIdx_ == 0 || prevBoundary >= 0) {
            COUNT_TYPE firstValue = idxLocal.GetValue(0) - static_cast<COUNT_TYPE>(prevBoundary);
            LocalTensor<COUNT_TYPE> outCount = countQueue_.template AllocTensor<COUNT_TYPE>();
            PostAdjDiff<COUNT_TYPE>(outCount, idxLocal, firstValue, candidateCount, START_POSITION);
            int64_t copyNums = candidateCount;
            if (writeLast) {
                outCount.SetValue(copyNums, static_cast<COUNT_TYPE>(nextUniqueIdx - lastHeadIdx));
                copyNums += 1;
            }
            countQueue_.EnQue(outCount);
            CopyOutCount((copyOutOffset == 0) ? 0 : copyOutOffset - 1, copyNums);
        } else {
            if (candidateCount > 1 || writeLast) {
                LocalTensor<COUNT_TYPE> outCount = countQueue_.template AllocTensor<COUNT_TYPE>();
                if (candidateCount > 1) {
                    COUNT_TYPE firstValue = idxLocal.GetValue(1) - idxLocal.GetValue(0);
                    PostAdjDiff<COUNT_TYPE, true>(outCount, idxLocal, firstValue, candidateCount, START_POSITION);
                }
                int64_t copyNums = candidateCount - 1;
                if (writeLast) {
                    outCount.SetValue(copyNums, static_cast<COUNT_TYPE>(nextUniqueIdx - lastHeadIdx));
                    copyNums += 1;
                }
                countQueue_.EnQue(outCount);
                CopyOutCount(copyOutOffset, copyNums);
            }
        }
        prevBoundary = static_cast<int64_t>(lastBoundary);
    }

    template <bool ADD_FIRST>
    __aicore__ inline void CollectUniqueAndCount(int64_t sourceStart, int64_t nums, int64_t& gatherCnt,
                                                 int64_t& prevBoundary, int64_t& lastHeadIdx, int64_t copyOutOffset,
                                                 int64_t nextUniqueIdx, int64_t& remainingCandidates)
    {
        LocalTensor<T> xLocal = xQueue_.template DeQue<T>();
        LocalTensor<T> outTensor = yQueue_.template AllocTensor<T>();
        uint64_t reduceCntValue = -1;
        CollectPostUniqueValue<T, T1, ADD_FIRST>(outTensor, xLocal, nums, reduceCntValue);
        gatherCnt = reduceCntValue;
        SimpleNativePipeSync<HardEvent::S_MTE3>();
        Copy2GmEx<T>(yGm_[copyOutOffset], outTensor, 1, gatherCnt, 0, 0);
        yQueue_.FreeTensor(outTensor);

        if constexpr (COUNT_OUT) {
            if (gatherCnt > 0) {
                if constexpr (sizeof(DTYPE_COUNT) == sizeof(int64_t)) {
                    LocalTensor<int32_t> idxLocal = idxQueue_.template AllocTensor<int32_t>();
                    uint64_t candidateCount = -1;
                    uint64_t alignPosition = idxQueueSize_ / MIDPOINT_DIVIDER / sizeof(int32_t);
                    CollectPostUniqueIdx<T, T1>(idxLocal, xLocal, 0, nums, nums, candidateCount, alignPosition);
                    if (candidateCount > 0) {
                        LocalTensor<int64_t> idxLocalInt64 = idxLocal.template ReinterpretCast<int64_t>();
                        CastAndAddsOffsets(idxLocalInt64, idxLocal, candidateCount, alignPosition, sourceStart);
                        SimpleNativePipeSync<HardEvent::V_S>();
                        ProcessCount<int64_t>(idxLocalInt64, candidateCount, copyOutOffset, prevBoundary, lastHeadIdx,
                                              nextUniqueIdx, remainingCandidates);
                    }
                    idxQueue_.FreeTensor(idxLocal);
                } else {
                    LocalTensor<int32_t> idxLocal = idxQueue_.template AllocTensor<int32_t>();
                    uint64_t candidateCount = -1;
                    CollectPostUniqueIdx<T, T1>(idxLocal, xLocal, static_cast<int32_t>(sourceStart), totalSize_, nums,
                                                candidateCount, START_POSITION);
                    if (candidateCount > 0) {
                        SimpleNativePipeSync<HardEvent::V_S>();
                        ProcessCount<int32_t>(idxLocal, candidateCount, copyOutOffset, prevBoundary, lastHeadIdx,
                                              nextUniqueIdx, remainingCandidates);
                    }
                    idxQueue_.FreeTensor(idxLocal);
                }
            }
        }
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutLastCount(int64_t offset, int64_t countValue)
    {
        LocalTensor<DTYPE_COUNT> outCount = countQueue_.template AllocTensor<DTYPE_COUNT>();
        outCount.SetValue(0, static_cast<DTYPE_COUNT>(countValue));
        countQueue_.EnQue(outCount);
        CopyOutCount(offset, 1);
    }

    __aicore__ inline void CopyOutCount(int64_t offset, int64_t copyLen)
    {
        if constexpr (sizeof(DTYPE_COUNT) == sizeof(int64_t)) {
            LocalTensor<int32_t> outCount = countQueue_.template DeQue<int32_t>();
            Copy2GmEx<int32_t>(countGm_[offset * DOUBLE_OFFSET], outCount, 1, copyLen * DOUBLE_OFFSET, 0, 0);
            countQueue_.FreeTensor(outCount);
        } else {
            LocalTensor<int32_t> outCount = countQueue_.template DeQue<int32_t>();
            Copy2GmEx<int32_t>(countGm_[offset], outCount, 1, copyLen, 0, 0);
            countQueue_.FreeTensor(outCount);
        }
    }

    __aicore__ inline void CopyOutShape(uint64_t dimNumValue, uint64_t dimNumIdx)
    {
        LocalTensor<uint64_t> shapeTensor = shapeBuf_.Get<uint64_t>();

        shapeTensor.SetValue(SHAPE0_SIZE_IDX, UINT64_SHAPE_DIM_ONE);
        shapeTensor.SetValue(SHAPE0_DIM0_IDX, dimNumValue);

        shapeTensor.SetValue(SHAPE1_SIZE_IDX, 1);
        shapeTensor.SetValue(SHAPE1_DIM0_IDX, 1);

        shapeTensor.SetValue(SHAPE2_SIZE_IDX, 1);
        if constexpr (COUNT_OUT) {
            shapeTensor.SetValue(SHAPE2_DIM0_IDX, dimNumIdx);
        } else {
            shapeTensor.SetValue(SHAPE2_DIM0_IDX, 1);
        }

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = SHAPE_LEN * sizeof(uint64_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        SimpleNativePipeSync<HardEvent::S_MTE3>();
        DataCopyPad(shapeGm_, shapeTensor, dataCopyParams);
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM_MULTI> xQueue_;

    TQue<QuePosition::VECOUT, BUFFER_NUM_MULTI> yQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_MULTI> countQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_MULTI> idxQueue_;

    TBuf<TPosition::VECCALC> collectingCntBuf_;
    TBuf<TPosition::VECCALC> offsetCntBuf_;
    TBuf<TPosition::VECCALC> prevIdxBuf_;
    TBuf<TPosition::VECCALC> shapeBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<int32_t> countGm_;
    GlobalTensor<int32_t> idxGm_;
    GlobalTensor<uint64_t> shapeGm_;

    GlobalTensor<int64_t> coreCollectWorkspaceGm_;

    int64_t ubTileLength_;
    int64_t adjUbTileLength_;
    int64_t coreIdx_;
    bool isFinalCore_;
    int64_t coreTileLength_;
    int64_t totalSize_;

    int64_t tileLengthPerCore_;
    int64_t idxQueueSize_;

    TPipe* pipe_ = nullptr;
};
#endif // UNIQUE_CONSECUTIVE_MULTI_CORE_KERNEL_H
