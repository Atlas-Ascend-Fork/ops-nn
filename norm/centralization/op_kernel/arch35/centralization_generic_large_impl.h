/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_GENERIC_LARGE_IMPL_H_
#define OPS_NORM_CENTRALIZATION_GENERIC_LARGE_IMPL_H_

__aicore__ inline void ProcessLarge(uint32_t core, uint32_t cores)
{
    if (tiling_->largePath == kInterleavedColumnsPath) {
        ProcessInterleavedColumns(core, cores);
        return;
    }
    if (tiling_->largePath == kStridedSuffixPath) {
        for (int64_t task = core; task < tiling_->groupCount * tiling_->coresPerRow; task += cores) {
            ProcessStridedSuffix(task);
        }
        return;
    }
    if (tiling_->largePlane != 0) {
        ProcessLargePlane(core, cores);
        return;
    }
    if (tiling_->rowParallel != 0) {
        ProcessLargeRowParallel(core, cores);
        return;
    }
    ProcessLargeMultiCoreReduce(core, cores);
}

__aicore__ inline void ProcessLargeRowParallel(uint32_t core, uint32_t cores)
{
    // 先按 group 并行算出 mean 存 workspace，再统一按 32B block 并行写输出。
    // 不能逐元素写：非连续/带 stride 的 reduce axes 下，相邻元素常属于不同 group，
    // 不同核并发写同一 cache line 会丢写（表现为偶发、非确定的错误元素）。
    for (int64_t group = core; group < tiling_->groupCount; group += cores) {
        const int64_t base = GroupBase(group);
        const MeanHiLo m = CalculateMeanFromBase(base);
        StoreWorkspacePair(WorkspaceSlot(group, 0), m.hi, m.lo);
    }
    SyncAll();
    ProcessOutputBlocks(core, cores, true);
}

// Boundary indices range from zero to irregularReduceChunks, inclusive; the last boundary is reduceSegmentCount.
__aicore__ inline int64_t SuffixSegmentBoundary(int64_t part) const
{
    const int64_t segments = tiling_->reduceSegmentCount;
    const int64_t parts = tiling_->irregularReduceChunks;
    return segments / parts * part + Min(part, segments % parts);
}

__aicore__ inline void ProcessLargeMultiCoreReduce(uint32_t core, uint32_t cores)
{
    const uint32_t coresPerRow = static_cast<uint32_t>(tiling_->coresPerRow);
    const int64_t taskCount = tiling_->groupCount * static_cast<int64_t>(coresPerRow);
    for (int64_t task = static_cast<int64_t>(core); task < taskCount; task += static_cast<int64_t>(cores)) {
        const int64_t group = task / static_cast<int64_t>(coresPerRow);
        const uint32_t localCore = static_cast<uint32_t>(task % static_cast<int64_t>(coresPerRow));
        const int64_t base = GroupBase(group);
        LargeCompensatedSum partial{0.0f, 0.0f};
        const int64_t lastReduceDim = tiling_->reduceDims[tiling_->reduceRank - 1];
        if (tiling_->irregularReduceChunks > 1) {
            AccumulateIrregularReduceSuffixTile(base, 0, 1, &partial, SuffixSegmentBoundary(localCore),
                                                SuffixSegmentBoundary(localCore + 1));
        } else if (lastReduceDim > 1 && tiling_->reduceStrides[tiling_->reduceRank - 1] == 1) {
            const int64_t outerCount = tiling_->reduceCount / lastReduceDim;
            for (int64_t outer = static_cast<int64_t>(localCore); outer < outerCount;
                 outer += static_cast<int64_t>(coresPerRow)) {
                const int64_t offset = base + ReduceOffset(outer * lastReduceDim);
                for (int64_t col = 0; col < lastReduceDim; col += kLargeChunkElements) {
                    const int64_t validElements = Min(kLargeChunkElements, lastReduceDim - col);
                    CopyInContiguous(offset + col, validElements);
                    LocalTensor<T> input = inQueue_.DeQue<T>();
                    AddLargeCompensated(partial, ReduceSegmentToScalar(input, validElements));
                    inQueue_.FreeTensor(input);
                }
            }
        } else {
            constexpr int64_t kBatch = 8;
            float batchValues[kBatch] = {};
            const int64_t chunk = (tiling_->reduceCount + static_cast<int64_t>(coresPerRow) - 1) /
                                  static_cast<int64_t>(coresPerRow);
            const int64_t begin = static_cast<int64_t>(localCore) * chunk;
            const int64_t end = Min(begin + chunk, tiling_->reduceCount);
            int64_t reduceCoords[8] = {};
            int64_t beginValue = begin;
            for (int64_t i = 0; i < tiling_->reduceRank; ++i) {
                reduceCoords[i] = beginValue / tiling_->reduceIndexStrides[i];
                beginValue %= tiling_->reduceIndexStrides[i];
            }
            int64_t currentOffset = ReduceOffset(begin);
            for (int64_t batchStart = begin; batchStart < end; batchStart += kBatch) {
                const int64_t batchCount = Min(kBatch, end - batchStart);
                int64_t batchOffsets[kBatch] = {};
                for (int64_t i = 0; i < batchCount; ++i) {
                    batchOffsets[i] = currentOffset;
                    AdvanceReduceOffset(currentOffset, reduceCoords, tiling_->reduceRank);
                }
                LoadInputBatch(base, batchOffsets, batchCount, batchValues);
                for (int64_t i = 0; i < batchCount; ++i) {
                    AddLargeCompensated(partial, batchValues[i]);
                }
            }
        }
        StoreWorkspacePair(WorkspaceSlot(static_cast<int64_t>(group), static_cast<int64_t>(localCore)), partial.sum,
                           partial.correction);
    }
    SyncAll();
    for (int64_t task = static_cast<int64_t>(core); task < taskCount; task += static_cast<int64_t>(cores)) {
        const int64_t group = task / static_cast<int64_t>(coresPerRow);
        const uint32_t localCore = static_cast<uint32_t>(task % static_cast<int64_t>(coresPerRow));
        if (localCore != 0) {
            continue;
        }
        LargeCompensatedSum total{0.0f, 0.0f};
        for (uint32_t c = 0; c < coresPerRow; ++c) {
            float partialSum = 0.0f;
            float partialCorrection = 0.0f;
            LoadWorkspacePair(WorkspaceSlot(group, static_cast<int64_t>(c)), partialSum, partialCorrection);
            AddLargeCompensated(total, partialSum);
            AddLargeCompensated(total, partialCorrection);
        }
        const MeanHiLo m = FinalizeMeanHiLo(total, tiling_->reduceCount);
        StoreWorkspacePair(WorkspaceSlot(group, 0), m.hi, m.lo);
    }
    SyncAll();
    ProcessOutputBlocks(core, GetBlockNum(), true);
}

__aicore__ inline void LoadInputBatch(int64_t base, const int64_t* offsets, int64_t count, float* values)
{
    constexpr int64_t kBlockBytes = 32;
    constexpr int64_t kBlockElements = kBlockBytes / sizeof(T);
    int64_t lanes[8] = {};
    int64_t slots[8] = {};
    int64_t blockOffsets[8] = {};
    int64_t slotCount = 0;
    LocalTensor<T> input = inQueue_.AllocTensor<T>();
    for (int64_t i = 0; i < count; ++i) {
        const int64_t offset = base + offsets[i];
        const int64_t alignedOffset = offset / kBlockElements * kBlockElements;
        lanes[i] = offset - alignedOffset;
        int64_t slot = 0;
        while (slot < slotCount && blockOffsets[slot] != alignedOffset) {
            ++slot;
        }
        if (slot == slotCount) {
            blockOffsets[slotCount] = alignedOffset;
            const int64_t validElements = (tiling_->totalCount - alignedOffset < kBlockElements) ?
                                              tiling_->totalCount - alignedOffset :
                                              kBlockElements;
            LocalTensor<T> dst = input[slotCount * kBlockElements];
            if (validElements == kBlockElements) {
                DataCopy(dst, xGm_[static_cast<uint64_t>(alignedOffset)], static_cast<uint32_t>(kBlockElements));
            } else {
                DataCopyExtParams copy{1, static_cast<uint32_t>(validElements * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(kBlockElements - validElements),
                                            static_cast<T>(0)};
                DataCopyPad(dst, xGm_[static_cast<uint64_t>(alignedOffset)], copy, pad);
            }
            ++slotCount;
        }
        slots[i] = slot;
    }
    inQueue_.EnQue(input);
    input = inQueue_.DeQue<T>();
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    for (int64_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>(input.GetValue(slots[i] * kBlockElements + lanes[i]));
    }
    inQueue_.FreeTensor(input);
}

__aicore__ inline float LoadInput(int64_t offset)
{
    constexpr int64_t kBlockBytes = 32;
    constexpr int64_t kBlockElements = kBlockBytes / sizeof(T);
    const int64_t alignedOffset = offset / kBlockElements * kBlockElements;
    const int64_t lane = offset - alignedOffset;
    const int64_t validElements = (tiling_->totalCount - alignedOffset < kBlockElements) ?
                                      tiling_->totalCount - alignedOffset :
                                      kBlockElements;
    LocalTensor<T> input = inQueue_.AllocTensor<T>();
    if (validElements == kBlockElements) {
        DataCopy(input, xGm_[static_cast<uint64_t>(alignedOffset)], static_cast<uint32_t>(kBlockElements));
    } else {
        DataCopyExtParams copy{1, static_cast<uint32_t>(validElements * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(kBlockElements - validElements), static_cast<T>(0)};
        DataCopyPad(input, xGm_[static_cast<uint64_t>(alignedOffset)], copy, pad);
    }
    inQueue_.EnQue(input);
    input = inQueue_.DeQue<T>();
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    const float value = static_cast<float>(input.GetValue(lane));
    inQueue_.FreeTensor(input);
    return value;
}

struct LargeCompensatedSum {
    float sum;
    float correction;
};

struct MeanHiLo {
    float hi;
    float lo;
};

// Kahan/Neumaier 补偿求和：超大 reduce 时抑制 fp32 累加的线性误差增长
// （对齐 bn3d_training_reduce / large_trailing 的补偿归约）。
__aicore__ inline void AddLargeCompensated(LargeCompensatedSum& acc, float value) const
{
    const float next = acc.sum + value;
    const float absSum = acc.sum < 0.0f ? -acc.sum : acc.sum;
    const float absValue = value < 0.0f ? -value : value;
    // inf/NaN 保护：非有限值不做 Neumaier 修正，避免 inf-inf=NaN
    constexpr float kMaxFinite = 3.4028235e38f;
    if (!(absSum <= kMaxFinite && absValue <= kMaxFinite)) {
        acc.sum = next;
        return;
    }
    if (absSum >= absValue) {
        acc.correction += (acc.sum - next) + value;
    } else {
        acc.correction += (value - next) + acc.sum;
    }
    acc.sum = next;
}

// 双 FP32 均值：sum + correction ≈ Σx（Neumaier）。
//   mean  = sum / N
//   r     = sum - mean*N（精确，Dekker） // 除法本身的舍入残差
//   meanLow = (r + correction) / N      // 除法残差 + 求和补偿，一起保留
__aicore__ inline MeanHiLo FinalizeMeanHiLo(const LargeCompensatedSum& acc, int64_t count) const
{
    const float n = static_cast<float>(count);
    MeanHiLo out;
    out.hi = acc.sum / n;
    // mean 非有限（组内含 inf/NaN）时除法残差会产生 NaN；置 0 保持 y = x - mean
    constexpr float kMaxFinite = 3.4028235e38f;
    if (!(out.hi <= kMaxFinite && out.hi >= -kMaxFinite)) {
        out.lo = 0.0f;
        return out;
    }
    const float r = MeanProductResidual(out.hi, n, acc.sum);
    out.lo = (r + acc.correction) / n;
    return out;
}

__aicore__ inline MeanHiLo CalculateMeanFromBase(int64_t base)
{
    const uint16_t numSlots = static_cast<uint16_t>(
        (tiling_->numAccSlots > 0 && tiling_->numAccSlots <= kMaxAccSlots) ? tiling_->numAccSlots : 1);
    LargeCompensatedSum slots[kMaxAccSlots];
    for (uint16_t i = 0; i < kMaxAccSlots; ++i) {
        slots[i] = LargeCompensatedSum{0.0f, 0.0f};
    }
    uint16_t slot = 0;
    for (int64_t r = 0; r < tiling_->reduceCount; ++r) {
        AddLargeCompensated(slots[slot], LoadInput(base + ReduceOffset(r)));
        slot = static_cast<uint16_t>((slot + 1 == numSlots) ? 0 : (slot + 1));
    }
    for (uint16_t active = numSlots; active > 1; active = static_cast<uint16_t>(active / 2)) {
        const uint16_t half = static_cast<uint16_t>(active / 2);
        for (uint16_t g = 0; g < half; ++g) {
            AddLargeCompensated(slots[g], slots[g + half].sum);
            AddLargeCompensated(slots[g], slots[g + half].correction);
        }
    }
    return FinalizeMeanHiLo(slots[0], tiling_->reduceCount);
}

__aicore__ inline MeanHiLo CalculateMean(int64_t group) { return CalculateMeanFromBase(GroupBase(group)); }

__aicore__ inline int64_t GroupFromOffset(int64_t offset) const
{
    int64_t strides[8] = {};
    int64_t stride = 1;
    const int64_t rank = tiling_->keepRank + tiling_->reduceRank;
    for (int64_t i = rank - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= tiling_->dims[i];
    }

    int64_t group = 0;
    int64_t value = offset;
    int64_t keepIndex = 0;
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t coord = value / strides[i];
        value %= strides[i];
        if (tiling_->reduceMask[i] == 0) {
            group += coord * tiling_->keepIndexStrides[keepIndex++];
        }
    }
    return group;
}

__aicore__ inline int64_t BuildMeanCache(int64_t blockOffset, int64_t validElements, uint32_t /*cores*/,
                                         bool useWorkspace, const int64_t* laneGroups, int64_t* cachedGroups,
                                         float* cachedMeans, float* cachedMeansLow)
{
    constexpr int64_t kBlockElements = 32 / sizeof(T);
    int64_t cachedCount = 0;
    for (int64_t lane = 0; lane < validElements; ++lane) {
        const int64_t group = laneGroups[lane];
        bool found = false;
        for (int64_t i = 0; i < cachedCount; ++i) {
            if (cachedGroups[i] == group) {
                found = true;
                break;
            }
        }
        if (!found && cachedCount < kBlockElements) {
            cachedGroups[cachedCount] = group;
            if (localMeanTable_) {
                const LocalTensor<float> table = sumBuf_.Get<float>();
                cachedMeans[cachedCount] = table.GetValue(group * 2);
                cachedMeansLow[cachedCount] = table.GetValue(group * 2 + 1);
            } else if (tiling_->meanCache != 0) {
                LoadWorkspaceMeanCompact(group, cachedMeans[cachedCount], cachedMeansLow[cachedCount]);
            } else if (useWorkspace) {
                // workspace 的 slot0 保存均值 hi/lo，两步相减保留低位
                LoadWorkspacePair(WorkspaceSlot(group, 0), cachedMeans[cachedCount], cachedMeansLow[cachedCount]);
            } else {
                const MeanHiLo m = CalculateMean(group);
                cachedMeans[cachedCount] = m.hi;
                cachedMeansLow[cachedCount] = m.lo;
            }
            ++cachedCount;
        }
    }
    return cachedCount;
}

__aicore__ inline void ProcessBlock(int64_t blockOffset, int64_t validElements, uint32_t cores, bool useWorkspace)
{
    constexpr int64_t kBlockElements = 32 / sizeof(T);
    int64_t laneGroups[kBlockElements] = {};
    int64_t cachedGroups[kBlockElements] = {};
    float cachedMeans[kBlockElements] = {};
    float cachedMeansLow[kBlockElements] = {};
    for (int64_t lane = 0; lane < validElements; ++lane) {
        laneGroups[lane] = GroupFromOffset(blockOffset + lane);
    }
    int64_t cachedCount = 0;
    if (!localMeanTable_) {
        cachedCount = BuildMeanCache(blockOffset, validElements, cores, useWorkspace, laneGroups, cachedGroups,
                                     cachedMeans, cachedMeansLow);
    }

    LocalTensor<T> input = inQueue_.AllocTensor<T>();
    DataCopyExtParams copy{1, static_cast<uint32_t>(validElements * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(kBlockElements - validElements), static_cast<T>(0)};
    DataCopyPad(input, xGm_[static_cast<uint64_t>(blockOffset)], copy, pad);
    inQueue_.EnQue(input);
    input = inQueue_.DeQue<T>();
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    LocalTensor<T> output = outQueue_.AllocTensor<T>();
    const LocalTensor<float> table = sumBuf_.Get<float>();
    for (int64_t lane = 0; lane < validElements; ++lane) {
        const int64_t group = laneGroups[lane];
        float mean = 0.0f;
        float meanLow = 0.0f;
        if (localMeanTable_) {
            mean = table.GetValue(group * 2);
            meanLow = table.GetValue(group * 2 + 1);
        } else {
            for (int64_t i = 0; i < cachedCount; ++i) {
                if (cachedGroups[i] == group) {
                    mean = cachedMeans[i];
                    meanLow = cachedMeansLow[i];
                    break;
                }
            }
        }
        // 两步相减，避免均值低阶在单次减法中被丢弃
        output.SetValue(lane, static_cast<T>((static_cast<float>(input.GetValue(lane)) - mean) - meanLow));
    }
    outQueue_.EnQue(output);
    output = outQueue_.DeQue<T>();
    DataCopyExtParams copyOut{1, static_cast<uint32_t>(validElements * sizeof(T)), 0, 0, 0};
    SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
    DataCopyPad(yGm_[static_cast<uint64_t>(blockOffset)], output, copyOut);
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    outQueue_.FreeTensor(output);
    inQueue_.FreeTensor(input);
}

__aicore__ inline void StoreWorkspaceMeanCompact(int64_t group, float hi, float lo)
{
    LocalTensor<float> output = outQueue_.AllocTensor<float>();
    output.SetValue(0, hi);
    output.SetValue(1, lo);
    outQueue_.EnQue(output);
    output = outQueue_.DeQue<float>();
    SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
    DataCopyExtParams copy{1, static_cast<uint32_t>(2 * sizeof(float)), 0, 0, 0};
    DataCopyPad(workspaceGm_[static_cast<uint64_t>(group * 2)], output, copy);
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    outQueue_.FreeTensor(output);
}

__aicore__ inline void LoadWorkspaceMeanCompact(int64_t group, float& hi, float& lo)
{
    LocalTensor<float> input = inQueue_.AllocTensor<float>();
    DataCopyExtParams copy{1, static_cast<uint32_t>(2 * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(input, workspaceGm_[static_cast<uint64_t>(group * 2)], copy, pad);
    inQueue_.EnQue(input);
    input = inQueue_.DeQue<float>();
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    hi = input.GetValue(0);
    lo = input.GetValue(1);
    inQueue_.FreeTensor(input);
}

__aicore__ inline void ComputeMeanCache(uint32_t core, uint32_t cores)
{
    for (int64_t group = core; group < tiling_->groupCount; group += cores) {
        const MeanHiLo m = CalculateMean(group);
        StoreWorkspaceMeanCompact(group, m.hi, m.lo);
    }
}

__aicore__ inline void PreloadMeanTable()
{
    LocalTensor<float> table = sumBuf_.Get<float>();
    for (int64_t group = 0; group < tiling_->groupCount; ++group) {
        float hi = 0.0f;
        float lo = 0.0f;
        LoadWorkspacePair(WorkspaceSlot(group, 0), hi, lo);
        table.SetValue(group * 2, hi);
        table.SetValue(group * 2 + 1, lo);
    }
}

__aicore__ inline void ProcessLargeOutputTile(int64_t offset, int64_t validElements)
{
    if (tiling_->irregularReduceChunks > 1) {
        // Each core owns complete output blocks, even when two groups share a 32-byte boundary.
        const int64_t suffix = tiling_->irregularReduceSuffixElements;
        const LocalTensor<float> table = sumBuf_.Get<float>();
        for (int64_t pos = 0; pos < validElements;) {
            const int64_t start = offset + pos;
            const int64_t group = GroupFromOffset(start);
            const int64_t run = Min(validElements - pos, suffix - start % suffix);
            WriteCenteredSuffix(start, run, table.GetValue(group * 2), table.GetValue(group * 2 + 1));
            pos += run;
        }
        return;
    }
    LocalTensor<T> input = inQueue_.AllocTensor<T>();
    DataCopyExtParams copy{1, static_cast<uint32_t>(validElements * sizeof(T)), 0, 0, 0};
    const int64_t alignedElements = AlignToBlock(validElements);
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(alignedElements - validElements), static_cast<T>(0)};
    DataCopyPad(input, xGm_[static_cast<uint64_t>(offset)], copy, pad);
    inQueue_.EnQue(input);
    input = inQueue_.DeQue<T>();
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    LocalTensor<T> output = outQueue_.AllocTensor<T>();
    const LocalTensor<float> table = sumBuf_.Get<float>();
    for (int64_t lane = 0; lane < validElements; ++lane) {
        const int64_t group = GroupFromOffset(offset + lane);
        const float mean = table.GetValue(group * 2);
        const float meanLow = table.GetValue(group * 2 + 1);
        output.SetValue(lane, static_cast<T>((static_cast<float>(input.GetValue(lane)) - mean) - meanLow));
    }
    outQueue_.EnQue(output);
    output = outQueue_.DeQue<T>();
    DataCopyExtParams copyOut{1, static_cast<uint32_t>(validElements * sizeof(T)), 0, 0, 0};
    SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
    DataCopyPad(yGm_[static_cast<uint64_t>(offset)], output, copyOut);
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    outQueue_.FreeTensor(output);
    inQueue_.FreeTensor(input);
}

__aicore__ inline void ProcessOutputBlocks(uint32_t core, uint32_t cores, bool useWorkspace)
{
    constexpr int64_t kBlockElements = 32 / sizeof(T);
    const int64_t meanTableBytes = tiling_->groupCount * static_cast<int64_t>(2 * sizeof(float));
    if (useWorkspace && tiling_->largePath != 0 && meanTableBytes > 0 &&
        meanTableBytes <= static_cast<int64_t>(tiling_->genericSumBufferBytes)) {
        PreloadMeanTable();
        localMeanTable_ = true;
    }
    if (localMeanTable_) {
        int64_t tileBytes = kOutputTileBytes;
        tileBytes = Min(tileBytes, static_cast<int64_t>(tiling_->genericInputBufferBytes));
        tileBytes = Min(tileBytes, static_cast<int64_t>(tiling_->genericOutputBufferBytes));
        int64_t tileElements = tileBytes / static_cast<int64_t>(sizeof(T));
        if (tileElements < 1) {
            tileElements = 1;
        }
        const int64_t tileCount = (tiling_->totalCount + tileElements - 1) / tileElements;
        const int64_t tilesPerCore = (tileCount + static_cast<int64_t>(cores) - 1) / static_cast<int64_t>(cores);
        const int64_t tileBegin = static_cast<int64_t>(core) * tilesPerCore;
        const int64_t tileEnd = Min(tileBegin + tilesPerCore, tileCount);
        for (int64_t tile = tileBegin; tile < tileEnd; ++tile) {
            const int64_t offset = tile * tileElements;
            const int64_t validElements = Min(tileElements, tiling_->totalCount - offset);
            ProcessLargeOutputTile(offset, validElements);
        }
        return;
    }
    const int64_t blockCount = (tiling_->totalCount + kBlockElements - 1) / kBlockElements;
    const int64_t blocksPerCore = (blockCount + static_cast<int64_t>(cores) - 1) / static_cast<int64_t>(cores);
    const int64_t begin = static_cast<int64_t>(core) * blocksPerCore;
    const int64_t end = Min(begin + blocksPerCore, blockCount);
    for (int64_t block = begin; block < end; ++block) {
        const int64_t blockOffset = block * kBlockElements;
        const int64_t validElements = (tiling_->totalCount - blockOffset < kBlockElements) ?
                                          tiling_->totalCount - blockOffset :
                                          kBlockElements;
        ProcessBlock(blockOffset, validElements, cores, useWorkspace);
    }
}

__aicore__ inline void LoadWorkspacePair(int64_t offset, float& first, float& second, uint32_t pair = 0)
{
    constexpr int64_t kWorkspaceElements = 32 / sizeof(float);
    LocalTensor<float> input = inQueue_.AllocTensor<float>();
    DataCopy(input, workspaceGm_[static_cast<uint64_t>(offset * kWorkspaceElements)],
             static_cast<uint32_t>(kWorkspaceElements));
    inQueue_.EnQue(input);
    input = inQueue_.DeQue<float>();
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    first = input.GetValue(2 * pair);
    second = input.GetValue(2 * pair + 1);
    inQueue_.FreeTensor(input);
}

__aicore__ inline void StoreWorkspacePair(int64_t offset, float first, float second, float scaledFirst = 0.0f,
                                          float scaledSecond = 0.0f)
{
    constexpr int64_t kWorkspaceElements = 32 / sizeof(float);
    LocalTensor<float> output = outQueue_.AllocTensor<float>();
    output.SetValue(0, first);
    output.SetValue(1, second);
    output.SetValue(2, scaledFirst);
    output.SetValue(3, scaledSecond);
    for (int64_t i = 4; i < kWorkspaceElements; ++i) {
        output.SetValue(i, 0.0f);
    }
    outQueue_.EnQue(output);
    output = outQueue_.DeQue<float>();
    SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
    DataCopy(workspaceGm_[static_cast<uint64_t>(offset * kWorkspaceElements)], output,
             static_cast<uint32_t>(kWorkspaceElements));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    outQueue_.FreeTensor(output);
}

// ─────────────────────────────────────────────────────────────────────────
// Plane 模式按物理连续行切片。单段归约布局为 [K_outer, R, K_inner]；
// 对首维归约且组号周期重复的多段布局，物理行含整数个周期，合并阶段按逻辑组折叠各列。
// 三阶段（SyncAll 分隔）：
//   1. partial 归约：task=(outer, localCore)，localCore 按 reduce 行切片，
//      行批搬运 + 列向补偿累加，partial 按 (outer, localCore) 落 workspace；
//   2. 合并：按 outer 轮转折叠 coresPerRow 份 partial，得 (outer,i) 的
//      hi/lo 均值，按 outer 分块落 workspace；
//   3. 输出：task=(outer, rowChunk) 全核轮转，均值按 outer 缓存复用，
//      行批中心化后整批写回（替代逐元素 32B 标量 gather/写）。
// workspace 先存所有 partial 块，步长为 PlanePartialStride(inner) 个 float；
// 随后每个 outer 存一块均值，高、低两部分相隔 rowStride 个 float。
// ─────────────────────────────────────────────────────────────────────────
__aicore__ inline int64_t PlaneAlignedFloats(int64_t floats) const
{
    return (floats * static_cast<int64_t>(sizeof(float)) + 31) / 32 * (32 / static_cast<int64_t>(sizeof(float)));
}

__aicore__ inline int64_t PlanePartialStride(int64_t inner) const { return PlaneAlignedFloats(2 * inner); }

// Reusing the same lane pattern for reduction and output requires each vector to start at column zero.
// Of the supported widths up to four, only 1, 2 and 4 divide the 64-lane vector.
__aicore__ inline bool PlaneSubBlock(int64_t inner) const { return inner == 1 || inner == 2 || inner == 4; }

// 子块归约：slice 的 flat 范围 [begin*inner, end*inner) 按 chunk 连续搬运，
// 每列经 pattern 分离后 Reg Reduce，chunk 末标量 Neumaier 折叠。
// sumBuf 布局：pattern [0, inner*V_LENGTH)；scratch 每列占 8 个 float，自 inner*V_LENGTH 起。
__aicore__ inline void ReduceSubBlockSlice(int64_t planeBase, int64_t begin, int64_t end, int64_t inner,
                                           LargeCompensatedSum* accs)
{
    LocalTensor<float> sum = sumBuf_.Get<float>();
    const int64_t patternBase = 0;
    // scratch 每列占 8 float（32B）：StoreRegForDtype 的目标地址必须 32B 对齐
    const int64_t scratchBase = inner * V_LENGTH;
    for (int64_t i = 0; i < inner; ++i) {
        for (int64_t lane = 0; lane < V_LENGTH; ++lane) {
            sum.SetValue(patternBase + i * V_LENGTH + lane, (lane % inner) == i ? 1.0f : 0.0f);
        }
        accs[i] = LargeCompensatedSum{0.0f, 0.0f};
    }
    SetFlag<HardEvent::S_V>(eventSToV_);
    WaitFlag<HardEvent::S_V>(eventSToV_);
    const int64_t bufferElements = static_cast<int64_t>(tiling_->genericInputBufferBytes / sizeof(T));
    const int64_t chunkElems = bufferElements > 0 ? bufferElements : 1;
    for (int64_t off = begin * inner; off < end * inner; off += chunkElems) {
        const int64_t valid = Min(chunkElems, end * inner - off);
        CopyInContiguous(planeBase + off, valid);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
        __local_mem__ float* sumAddr = (__local_mem__ float*)sum.GetPhyAddr();
        const uint16_t loops = static_cast<uint16_t>((valid + V_LENGTH - 1) / V_LENGTH);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> patReg;
            RegTensor<float> selected;
            RegTensor<float> zero;
            RegTensor<float> partialReg;
            RegTensor<float> colReg;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            Duplicate(zero, 0.0f, pregFull);
            for (uint16_t i = 0; i < static_cast<uint16_t>(inner); ++i) {
                Duplicate(colReg, static_cast<float>(0.0f), pregOne);
                LoadRegForDtype<float>(sumAddr, patReg, pregFull, static_cast<uint32_t>(patternBase + i * V_LENGTH));
                MaskReg column;
                Compares<float, CMPMODE::EQ>(column, patReg, 1.0f, pregFull);
                for (uint16_t loop = 0; loop < loops; ++loop) {
                    const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
                    const uint32_t remaining = static_cast<uint32_t>(valid) - offset;
                    uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
                    MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
                    LoadRegForDtype<T>(inputAddr, xReg, pregLoop, offset);
                    Select(selected, xReg, zero, column);
                    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(partialReg, selected, pregLoop);
                    Add(colReg, colReg, partialReg, pregOne);
                }
                StoreRegForDtype<float>(sumAddr, colReg, pregOne, static_cast<uint32_t>(scratchBase + i * 8));
            }
        }
        SetFlag<HardEvent::V_S>(eventVToS_);
        WaitFlag<HardEvent::V_S>(eventVToS_);
        for (uint16_t i = 0; i < static_cast<uint16_t>(inner); ++i) {
            AddLargeCompensated(accs[i], sum.GetValue(scratchBase + static_cast<int64_t>(i) * 8));
        }
        inQueue_.FreeTensor(input);
    }
}

// 子块输出：把 (hi,lo) 按 lane%inner 展开成 64-lane 向量（S 域构造 + S_V 同步）
__aicore__ inline void BuildSubBlockMeanVec(LocalTensor<float>& sum, LocalTensor<float>& meanTensor, int64_t inner,
                                            int64_t rowStride)
{
    for (int64_t lane = 0; lane < V_LENGTH; ++lane) {
        const int64_t col = lane % inner;
        sum.SetValue(lane, meanTensor.GetValue(col));
        sum.SetValue(V_LENGTH + lane, meanTensor.GetValue(rowStride + col));
    }
    SetFlag<HardEvent::S_V>(eventSToV_);
    WaitFlag<HardEvent::S_V>(eventSToV_);
}

// 子块中心化：flat 块两次 Sub（hi/lo 向量在 sumBuf [0..V_LENGTH) 和 [V_LENGTH..2*V_LENGTH)）
__aicore__ inline void CentralizeSubBlockChunk(LocalTensor<T>& input, LocalTensor<T>& output,
                                               LocalTensor<float>& meanVec, int64_t valid)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
    __local_mem__ T* outputAddr = (__local_mem__ T*)output.GetPhyAddr();
    __local_mem__ float* meanAddr = (__local_mem__ float*)meanVec.GetPhyAddr();
    const uint16_t loops = static_cast<uint16_t>((valid + V_LENGTH - 1) / V_LENGTH);
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> hiReg;
        RegTensor<float> loReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        LoadRegForDtype<float>(meanAddr, hiReg, pregFull, 0);
        LoadRegForDtype<float>(meanAddr, loReg, pregFull, V_LENGTH);
        for (uint16_t loop = 0; loop < loops; ++loop) {
            const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
            const uint32_t remaining = static_cast<uint32_t>(valid) - offset;
            uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
            MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
            LoadRegForDtype<T>(inputAddr, xReg, pregLoop, offset);
            Sub(xReg, xReg, hiReg, pregLoop);
            Sub(xReg, xReg, loReg, pregLoop);
            StoreRegForDtype<T>(outputAddr, xReg, pregLoop, offset);
        }
    }
}

__aicore__ inline int64_t PlaneMeanFloatOffset(int64_t o, int64_t rowStride, int64_t outer, int64_t inner,
                                               int64_t coresPerRow) const
{
    // partial 区共 outer*coresPerRow*PlanePartialStride(inner) 个 float；
    // mean 块按 outer 分块，块内 [hi @0, lo @rowStride]（lo 起点保持 32B 对齐，VEC LoadReg 要求）
    return outer * coresPerRow * PlanePartialStride(inner) + o * 2 * rowStride;
}

// partial 块 {sum[0..inner), corr[0..inner)}（步进 PlanePartialStride）：S 域 SetValue 组装 + MTE3 落 workspace
__aicore__ inline void StorePlanePartials(int64_t o, int64_t localCore, LocalTensor<float>& sum, int64_t corrOffset,
                                          int64_t inner, int64_t coresPerRow)
{
    const int64_t floatOffset = (o * coresPerRow + localCore) * PlanePartialStride(inner);
    const int64_t alignedFloats = PlanePartialStride(inner);
    LocalTensor<float> output = outQueue_.AllocTensor<float>();
    for (int64_t i = 0; i < alignedFloats; ++i) {
        output.SetValue(i, 0.0f);
    }
    for (int64_t i = 0; i < inner; ++i) {
        output.SetValue(i, sum.GetValue(i));
        output.SetValue(inner + i, sum.GetValue(corrOffset + i));
    }
    outQueue_.EnQue(output);
    output = outQueue_.DeQue<float>();
    SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
    DataCopy(workspaceGm_[static_cast<uint64_t>(floatOffset)], output, static_cast<uint32_t>(alignedFloats));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    outQueue_.FreeTensor(output);
}

// If a partial or its merge overflows, reread this logical group with binary scaling.
// Batch the innermost reduction axis into DMA rows without disturbing the other groups in UB.
__aicore__ inline MeanHiLo RecomputePlaneMean(int64_t group)
{
    const int64_t base = GroupBase(group);
    const int64_t count = tiling_->reduceCount;
    const int64_t axis = tiling_->reduceRank - 1;
    const int64_t width = tiling_->reduceDims[axis];
    const int64_t stride = tiling_->reduceStrides[axis];
    const int64_t pitch = AlignToBlock(1);
    const int64_t capacity = tiling_->genericInputBufferBytes / (pitch * sizeof(T));
    const int64_t maxRows = stride <= UINT32_MAX / sizeof(T) ? Min(capacity, 4095) : 1;
    const float restore = ReductionRestoreFactor(count);
    const float scale = 1.0f / restore;
    LargeCompensatedSum acc{};
    for (int64_t r = 0; r < count;) {
        const int64_t rows = Min(maxRows, width - r % width);
        CopyInRowsPadded(base + ReduceOffset(r), rows, 1, pitch, rows > 1 ? stride : 1);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
        WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
        for (int64_t i = 0; i < rows; ++i) {
            AddLargeCompensated(acc, static_cast<float>(input.GetValue(i * pitch)) * scale);
        }
        inQueue_.FreeTensor(input);
        r += rows;
    }
    MeanHiLo mean = FinalizeMeanHiLo(acc, count);
    mean.hi *= restore;
    mean.lo *= restore;
    return mean;
}

// 合并 plane o 的 coresPerRow 份 partial，写 hi/lo 均值块到 workspace。
// 高位从块首写入，低位从 OptimalRowStride(inner) 个 float 处写入。
__aicore__ inline void MergePlaneMeans(int64_t o, int64_t inner, int64_t coresPerRow)
{
    LocalTensor<float> sum = sumBuf_.Get<float>();
    const int64_t rowStride = OptimalRowStride(inner);
    const int64_t corrOffset = rowStride;
    const int64_t partialFloats = PlanePartialStride(inner);
    const int64_t meanFloats = PlaneAlignedFloats(2 * rowStride);
    for (int64_t i = 0; i < inner; ++i) {
        sum.SetValue(i, 0.0f);
        sum.SetValue(corrOffset + i, 0.0f);
    }
    for (int64_t c = 0; c < coresPerRow; ++c) {
        const int64_t floatOffset = (o * coresPerRow + c) * partialFloats;
        LocalTensor<float> input = inQueue_.AllocTensor<float>();
        DataCopy(input, workspaceGm_[static_cast<uint64_t>(floatOffset)], static_cast<uint32_t>(partialFloats));
        inQueue_.EnQue(input);
        input = inQueue_.DeQue<float>();
        SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
        WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
        for (int64_t i = 0; i < inner; ++i) {
            LargeCompensatedSum acc{sum.GetValue(i), sum.GetValue(corrOffset + i)};
            AddLargeCompensated(acc, input.GetValue(i));
            AddLargeCompensated(acc, input.GetValue(inner + i));
            sum.SetValue(i, acc.sum);
            sum.SetValue(corrOffset + i, acc.correction);
        }
        inQueue_.FreeTensor(input);
    }
    LocalTensor<float> output = outQueue_.AllocTensor<float>();
    for (int64_t i = 0; i < meanFloats; ++i) {
        output.SetValue(i, 0.0f);
    }
    if (tiling_->reduceRank > 1) {
        // Fold physical lanes belonging to the same logical group, then broadcast back to the row layout.
        for (int64_t i = 0; i < inner; ++i) {
            const int64_t group = GroupFromOffset(i);
            LargeCompensatedSum acc{output.GetValue(group), output.GetValue(rowStride + group)};
            AddLargeCompensated(acc, sum.GetValue(i));
            AddLargeCompensated(acc, sum.GetValue(corrOffset + i));
            output.SetValue(group, acc.sum);
            output.SetValue(rowStride + group, acc.correction);
        }
        for (int64_t i = 0; i < inner; ++i) {
            const int64_t group = GroupFromOffset(i);
            sum.SetValue(i, output.GetValue(group));
            sum.SetValue(corrOffset + i, output.GetValue(rowStride + group));
        }
    }
    for (int64_t i = 0; i < inner; ++i) {
        const LargeCompensatedSum acc{sum.GetValue(i), sum.GetValue(corrOffset + i)};
        const MeanHiLo m = IsFinitePair(acc.sum, acc.correction) ?
                               FinalizeMeanHiLo(acc, tiling_->reduceCount) :
                               RecomputePlaneMean(
                                   GroupFromOffset(o * (tiling_->totalCount / tiling_->contiguousOuterCount) + i));
        output.SetValue(i, m.hi);
        output.SetValue(rowStride + i, m.lo);
    }
    outQueue_.EnQue(output);
    output = outQueue_.DeQue<float>();
    SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
    DataCopy(workspaceGm_[static_cast<uint64_t>(
                 PlaneMeanFloatOffset(o, rowStride, tiling_->contiguousOuterCount, inner, coresPerRow))],
             output, static_cast<uint32_t>(meanFloats));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    outQueue_.FreeTensor(output);
}

__aicore__ inline void ProcessLargePlane(uint32_t core, uint32_t cores)
{
    const int64_t outer = tiling_->contiguousOuterCount;
    const int64_t inner = tiling_->contiguousInnerCount;
    const int64_t rc = tiling_->totalCount / outer / inner;
    const int64_t coresPerRow = tiling_->coresPerRow;
    const int64_t rowStride = OptimalRowStride(inner);
    const int64_t bufferElements = static_cast<int64_t>(tiling_->genericInputBufferBytes / sizeof(T));
    constexpr int64_t kMaxAbiRows = 65535;
    const int64_t rowBudget = bufferElements / rowStride > 0 ? bufferElements / rowStride : 1;
    const int64_t rowsPerBatch = Min(rowBudget, Min(kMaxAbiRows, rc));
    const int64_t rowsPerCore = (rc + coresPerRow - 1) / coresPerRow;
    const int64_t nCores = static_cast<int64_t>(cores);

    // ---- 阶段 1：partial 归约（task = (outer, localCore)）----
    const int64_t planeTasks = outer * coresPerRow;
    const bool subBlock = PlaneSubBlock(inner);
    for (int64_t task = static_cast<int64_t>(core); task < planeTasks; task += nCores) {
        const int64_t o = task / coresPerRow;
        const int64_t localCore = task - o * coresPerRow;
        const int64_t begin = localCore * rowsPerCore;
        const int64_t end = Min(begin + rowsPerCore, rc);
        if (begin >= end) {
            continue;
        }
        LocalTensor<float> sum = sumBuf_.Get<float>();
        const int64_t corrOffset = rowStride;
        const int64_t planeBase = o * rc * inner;
        if (subBlock) {
            // 子块域：flat 连续块 + pattern 列分离（pattern 与行批互斥使用 sumBuf）
            LargeCompensatedSum accs[16];
            ReduceSubBlockSlice(planeBase, begin, end, inner, accs);
            for (int64_t i = 0; i < inner; ++i) {
                sum.SetValue(i, accs[i].sum);
                sum.SetValue(corrOffset + i, accs[i].correction);
            }
            StorePlanePartials(o, localCore, sum, corrOffset, inner, coresPerRow);
            continue;
        }
        ZeroSumTileAt(sum, 0, inner);
        ZeroSumTileAt(sum, corrOffset, inner);
        PipeBarrier<PIPE_V>();
        uint16_t slot = 0;
        for (int64_t rOff = begin; rOff < end; rOff += rowsPerBatch) {
            const int64_t rows = Min(rowsPerBatch, end - rOff);
            CopyInRowsPadded(planeBase + rOff * inner, rows, inner, rowStride);
            LocalTensor<T> input = inQueue_.DeQue<T>();
            AccumulateContiguousRows(input, sum, 1, rowStride, corrOffset, rows, inner, rowStride, slot);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(input);
        }
        // partial 前的 VEC store 需 V→S 可见
        SetFlag<HardEvent::V_S>(eventVToS_);
        WaitFlag<HardEvent::V_S>(eventVToS_);
        StorePlanePartials(o, localCore, sum, corrOffset, inner, coresPerRow);
    }
    SyncAll();

    // ---- 阶段 2：合并均值（按 outer 轮转）----
    for (int64_t o = static_cast<int64_t>(core); o < outer; o += nCores) {
        MergePlaneMeans(o, inner, coresPerRow);
    }
    SyncAll();

    // ---- 阶段 3：输出（task = (outer, rowChunk)，均值按 outer 缓存）----
    const int64_t outChunkElems = subBlock ? bufferElements : rowsPerBatch;
    const int64_t outTotal = subBlock ? rc * inner : rc;
    const int64_t chunksPerPlane = (outTotal + outChunkElems - 1) / outChunkElems;
    LocalTensor<float> meanTensor;
    int64_t cachedO = -1;
    const int64_t outTasks = outer * chunksPerPlane;
    for (int64_t task = static_cast<int64_t>(core); task < outTasks; task += nCores) {
        const int64_t o = task / chunksPerPlane;
        const int64_t chunk = task - o * chunksPerPlane;
        if (o != cachedO) {
            if (cachedO >= 0) {
                inQueue_.FreeTensor(meanTensor);
            }
            meanTensor = inQueue_.AllocTensor<float>();
            DataCopy(meanTensor,
                     workspaceGm_[static_cast<uint64_t>(
                         PlaneMeanFloatOffset(o, rowStride, tiling_->contiguousOuterCount, inner, coresPerRow))],
                     static_cast<uint32_t>(PlaneAlignedFloats(2 * rowStride)));
            inQueue_.EnQue(meanTensor);
            meanTensor = inQueue_.DeQue<float>();
            cachedO = o;
            if (subBlock) {
                // 标量读 meanTensor 前补 MTE2→S 同步（队列只保证 VEC 可见）
                SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
                WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
                LocalTensor<float> sum = sumBuf_.Get<float>();
                BuildSubBlockMeanVec(sum, meanTensor, inner, rowStride);
            }
        }
        const int64_t chunkBegin = chunk * outChunkElems;
        const int64_t planeBase = o * rc * inner;
        if (subBlock) {
            const int64_t valid = Min(outChunkElems, rc * inner - chunkBegin);
            CopyInContiguous(planeBase + chunkBegin, valid);
            LocalTensor<T> input = inQueue_.DeQue<T>();
            LocalTensor<T> output = outQueue_.AllocTensor<T>();
            LocalTensor<float> sum = sumBuf_.Get<float>();
            CentralizeSubBlockChunk(input, output, sum, valid);
            PipeBarrier<PIPE_V>();
            outQueue_.EnQue(output);
            inQueue_.FreeTensor(input);
            CopyOutContiguous(planeBase + chunkBegin, valid);
            continue;
        }
        const int64_t rows = Min(rowsPerBatch, rc - chunkBegin);
        CopyInRowsPadded(planeBase + chunkBegin * inner, rows, inner, rowStride);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        LocalTensor<T> output = outQueue_.AllocTensor<T>();
        CentralizeContiguousRows(input, output, meanTensor, 0, rowStride, rows, inner, rowStride);
        PipeBarrier<PIPE_V>();
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
        CopyOutRowsPadded(planeBase + chunkBegin * inner, rows, inner, rowStride);
    }
    if (cachedO >= 0) {
        inQueue_.FreeTensor(meanTensor);
    }
}

#endif
