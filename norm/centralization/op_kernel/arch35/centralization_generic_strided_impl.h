/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_GENERIC_STRIDED_IMPL_H_
#define OPS_NORM_CENTRALIZATION_GENERIC_STRIDED_IMPL_H_

struct StridedSuffixTile {
    int64_t offset;
    int64_t rows;
    int64_t width;
    int64_t ubStride;
    int64_t gmStride;
};

// Keep coordinates are fixed by base. The selected reduction axis supplies the DMA rows;
// a kept innermost axis gives a one-element suffix for a single output column.
__aicore__ inline StridedSuffixTile GetStridedSuffixTile(int64_t base, int64_t pos, int64_t remaining, int64_t capacity,
                                                         int64_t axis) const
{
    const int64_t suffix = tiling_->irregularReduceSuffixElements;
    const int64_t column = pos % suffix;
    const int64_t row = pos / suffix;
    const int64_t axisDim = tiling_->reduceDims[axis];
    const int64_t axisRow = row % axisDim;
    int64_t outer = row / axisDim;
    int64_t offset = base + column + axisRow * tiling_->reduceStrides[axis];
    for (int64_t i = tiling_->reduceRank - 2; i >= 0; --i) {
        if (i != axis) {
            offset += (outer % tiling_->reduceDims[i]) * tiling_->reduceStrides[i];
            outer /= tiling_->reduceDims[i];
        }
    }
    const int64_t width = Min(capacity, Min(suffix - column, remaining));
    const int64_t ubStride = AlignToBlock(width);
    constexpr int64_t kMaxDmaRows = 4095;
    const int64_t rows = (width == suffix && tiling_->reduceStrides[axis] <= UINT32_MAX / sizeof(T)) ?
                             Min(kMaxDmaRows, Min(axisDim - axisRow, Min(remaining / width, capacity / ubStride))) :
                             1;
    return {offset, rows, width, ubStride, rows > 1 ? tiling_->reduceStrides[axis] : width};
}

__aicore__ inline void AccumulateStridedTile(LocalTensor<T>& input, uint32_t count, float scale,
                                             LargeCompensatedSum& sum, int64_t scratchOffset)
{
    LocalTensor<float> scratch = outQueue_.AllocTensor<float>();
    LocalTensor<float> values = scratch;
    if constexpr (IsSameType<T, float>::value) {
        values = input.template ReinterpretCast<float>();
        if (scale != 1.0f) {
            Muls(scratch, values, scale, count);
            PipeBarrier<PIPE_V>();
            values = scratch;
        }
    } else {
        Cast(values, input, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<float> partial = sumBuf_.Get<float>()[scratchOffset];
    ReduceTileWithResidual(values, partial, count);
    SetFlag<HardEvent::V_S>(eventVToS_);
    WaitFlag<HardEvent::V_S>(eventVToS_);
    const float high = partial.GetValue(0);
    AddLargeCompensated(sum, high);
    if (high <= 0x1.fffffep127f && high >= -0x1.fffffep127f) {
        AddLargeCompensated(sum, partial.GetValue(kResidualOffset));
    }
    outQueue_.FreeTensor(scratch);
}

template <bool Write>
__aicore__ inline LargeCompensatedSum WalkStridedSuffix(int64_t base, int64_t begin, int64_t end, MeanHiLo mean,
                                                        int64_t axis, float scale = 1.0f, int64_t scratchOffset = 0)
{
    const int64_t capacity = Min(tiling_->genericInputBufferBytes, tiling_->genericOutputBufferBytes) /
                             (V_LENGTH * sizeof(float)) * V_LENGTH;
    LargeCompensatedSum sum{};
    while (begin < end) {
        const StridedSuffixTile tile = GetStridedSuffixTile(base, begin, end - begin, capacity, axis);
        CopyInRowsPadded(tile.offset, tile.rows, tile.width, tile.ubStride, tile.gmStride);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        // Multi-row DMA zero-fills each UB row up to the block boundary.
        const uint32_t count = static_cast<uint32_t>(tile.rows == 1 ? tile.width : tile.rows * tile.ubStride);
        if constexpr (Write) {
            LocalTensor<T> output = outQueue_.AllocTensor<T>();
            CentralizeSegmentHiLo(input, output, count, mean.hi, mean.lo);
            outQueue_.EnQue(output);
            CopyOutRowsPadded(tile.offset, tile.rows, tile.width, tile.ubStride, tile.gmStride);
        } else {
            AccumulateStridedTile(input, count, scale, sum, scratchOffset);
        }
        inQueue_.FreeTensor(input);
        begin += tile.rows * tile.width;
    }
    return sum;
}

__aicore__ inline bool IsFinitePair(float high, float low) const
{
    constexpr float maxFinite = 0x1.fffffep127f;
    return high <= maxFinite && high >= -maxFinite && low <= maxFinite && low >= -maxFinite;
}

__aicore__ inline float ReductionRestoreFactor(int64_t count) const
{
    float restore = 2.0f;
    for (int64_t n = count - 1; n > 0; n >>= 1) {
        restore *= 2.0f;
    }
    return restore;
}

__aicore__ inline LargeCompensatedSum MergeStridedPartials(int64_t group, int64_t parts, uint32_t pair)
{
    LargeCompensatedSum sum{};
    for (int64_t i = 0; i < parts; ++i) {
        float high;
        float low;
        LoadWorkspacePair(group * parts + i, high, low, pair);
        AddLargeCompensated(sum, high);
        if (high <= 0x1.fffffep127f && high >= -0x1.fffffep127f) {
            AddLargeCompensated(sum, low);
        }
    }
    return sum;
}

__aicore__ inline void ProcessStridedSuffix(int64_t core)
{
    const int64_t parts = tiling_->coresPerRow;
    const int64_t group = core / parts;
    const int64_t part = core % parts;
    const int64_t base = GroupBase(group);
    const int64_t axis = tiling_->stridedReduceAxis;
    const int64_t count = tiling_->reduceCount;
    const int64_t suffix = tiling_->irregularReduceSuffixElements;
    const int64_t unit = suffix <= kMaxBatchedSuffixElements ? suffix : 32 / sizeof(T);
    const int64_t units = (count - 1) / unit + 1;
    const int64_t begin = (units / parts * part + Min(part, units % parts)) * unit;
    const int64_t end = part + 1 == parts ? count : (units / parts * (part + 1) + Min(part + 1, units % parts)) * unit;
    LargeCompensatedSum sum = WalkStridedSuffix<false>(base, begin, end, {}, axis);
    LargeCompensatedSum scaled{};
    float restore = 1.0f;
    if constexpr (IsSameType<T, float>::value) {
        // The spare workspace pair stores the scaled partial sum and its correction.
        // A nonfinite unscaled partial is recomputed from scaled inputs.
        restore = ReductionRestoreFactor(count);
        const float scale = 1.0f / restore;
        scaled = IsFinitePair(sum.sum, sum.correction) ? LargeCompensatedSum{sum.sum * scale, sum.correction * scale} :
                                                         WalkStridedSuffix<false>(base, begin, end, {}, axis, scale);
    }
    if (parts > 1) {
        StoreWorkspacePair(core, sum.sum, sum.correction, scaled.sum, scaled.correction);
        SyncAll();
        sum = MergeStridedPartials(group, parts, 0);
    }
    float meanScale = 1.0f;
    if constexpr (IsSameType<T, float>::value) {
        if (!IsFinitePair(sum.sum, sum.correction)) {
            sum = parts > 1 ? MergeStridedPartials(group, parts, 1) : scaled;
            meanScale = restore;
        }
    }
    MeanHiLo mean = FinalizeMeanHiLo(sum, count);
    mean.hi *= meanScale;
    mean.lo *= meanScale;
    WalkStridedSuffix<true>(base, begin, end, mean, axis);
}

#endif
