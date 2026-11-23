/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_GENERIC_GROUP_VECTOR_IMPL_H_
#define OPS_NORM_CENTRALIZATION_GENERIC_GROUP_VECTOR_IMPL_H_

__aicore__ inline void ProcessGroupVector(uint32_t core, uint32_t cores)
{
    const int64_t lastKeepDim = tiling_->keepDims[tiling_->keepRank - 1];
    const int64_t tile = Min(kLargeChunkElements, lastKeepDim);
    const int64_t blocksPerKeep = (lastKeepDim + tile - 1) / tile;
    const int64_t prefixCount = tiling_->groupCount / lastKeepDim;
    const int64_t blockCount = prefixCount * blocksPerKeep;
    const int64_t blocksPerCore = (blockCount + static_cast<int64_t>(cores) - 1) / static_cast<int64_t>(cores);
    const int64_t begin = static_cast<int64_t>(core) * blocksPerCore;
    const int64_t end = Min(begin + blocksPerCore, blockCount);
    for (int64_t block = begin; block < end; ++block) {
        const int64_t prefix = block / blocksPerKeep;
        const int64_t within = block - prefix * blocksPerKeep;
        const int64_t groupStart = prefix * lastKeepDim + within * tile;
        const int64_t validGroups = Min(tile, lastKeepDim - within * tile);
        const int64_t base = GroupBase(groupStart);
        LocalTensor<float> sum = sumBuf_.Get<float>();
        const int64_t corrOffset = tile;
        ZeroSumTileAt(sum, 0, validGroups);
        ZeroSumTileAt(sum, corrOffset, validGroups);
        PipeBarrier<PIPE_V>();

        for (int64_t r = 0; r < tiling_->reduceCount; ++r) {
            CopyInContiguous(base + ReduceOffset(r), validGroups);
            LocalTensor<T> input = inQueue_.DeQue<T>();
            AccumulateContiguousNeumaierAt(input, sum, 0, corrOffset, validGroups);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(input);
        }
        FinalizeMeanHiLoAt(sum, 0, corrOffset, validGroups);
        PipeBarrier<PIPE_V>();

        for (int64_t r = 0; r < tiling_->reduceCount; ++r) {
            const int64_t offset = base + ReduceOffset(r);
            CopyInContiguous(offset, validGroups);
            LocalTensor<T> input = inQueue_.DeQue<T>();
            LocalTensor<T> output = outQueue_.AllocTensor<T>();
            CentralizeContiguousHiLoAt(input, output, sum, 0, corrOffset, validGroups);
            PipeBarrier<PIPE_V>();
            outQueue_.EnQue(output);
            inQueue_.FreeTensor(input);
            CopyOutContiguous(offset, validGroups);
        }
    }
}

#endif
