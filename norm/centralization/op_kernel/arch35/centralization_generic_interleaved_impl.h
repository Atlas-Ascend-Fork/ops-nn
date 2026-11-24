/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_GENERIC_INTERLEAVED_IMPL_H_
#define OPS_NORM_CENTRALIZATION_GENERIC_INTERLEAVED_IMPL_H_

struct InterleavedTile {
    StridedSuffixTile dma;
    int64_t periods;
    int64_t localRows;
    int64_t pitch;
};

// A physical row contains groups * suffix contiguous elements. Several complete outer segments
// can share a DMA when their padding is also a whole number of physical rows.
__aicore__ inline InterleavedTile GetInterleavedTile(int64_t base, int64_t position, int64_t remaining,
                                                     int64_t capacity, int64_t period) const
{
    const int64_t axis = tiling_->stridedReduceAxis;
    const int64_t span = tiling_->reduceDims[axis];
    const int64_t row = position % span;
    int64_t outer = position / span;
    int64_t offset = base + row * period;
    for (int64_t i = axis - 1; i >= 0; --i) {
        offset += (outer % tiling_->reduceDims[i]) * tiling_->reduceStrides[i];
        outer /= tiling_->reduceDims[i];
    }
    if (period == 3) {
        const int64_t pitch = AlignToBlock(period);
        const int64_t rows = Min(4095, Min(capacity / pitch, Min(span - row, remaining)));
        return {{offset, rows, period, pitch, period}, rows, rows, pitch};
    }
    const int64_t rows = Min(capacity / period, Min(span - row, remaining));
    const int64_t width = rows * period;
    const int64_t pitch = AlignToBlock(width);
    if (axis > 0 && rows == span && pitch % period == 0 && tiling_->reduceStrides[axis - 1] <= UINT32_MAX / sizeof(T)) {
        const int64_t nextDim = tiling_->reduceDims[axis - 1];
        const int64_t repeats = Min(4095,
                                    Min(nextDim - position / span % nextDim, Min(capacity / pitch, remaining / span)));
        if (repeats > 1) {
            return {{offset, repeats, width, pitch, tiling_->reduceStrides[axis - 1]},
                    repeats * span,
                    repeats * (pitch / period),
                    period};
        }
    }
    return {{offset, 1, width, pitch, width}, rows, rows, period};
}

template <bool Write>
__aicore__ inline void CopyInterleavedColumn(LocalTensor<float>& values, LocalTensor<float>& result, uint32_t count,
                                             uint32_t pitch, uint32_t group, uint32_t suffix, MeanHiLo mean)
{
    using namespace AscendC::Reg;
    auto* src = (__local_mem__ float*)values.GetPhyAddr();
    auto* dst = (__local_mem__ float*)result.GetPhyAddr();
    const uint16_t loops = static_cast<uint16_t>((count + V_LENGTH - 1) / V_LENGTH);
    const uint32_t step = V_LENGTH / suffix * pitch;
    const int16_t shift = suffix == 4 ? 2 : 1;
    __VEC_SCOPE__
    {
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        RegTensor<uint32_t> lane, row, col, offsets;
        Arange(reinterpret_cast<RegTensor<int32_t>&>(lane), 0);
        if (suffix == 1) {
            Muls(offsets, lane, pitch, full);
        } else {
            ShiftRights(row, lane, shift, full);
            RegTensor<uint32_t> mask;
            Duplicate(mask, suffix - 1, full);
            And(col, lane, mask, full);
            Muls(offsets, row, pitch, full);
            Add(offsets, offsets, col, full);
        }
        Adds(offsets, offsets, group * suffix, full);
        RegTensor<float> negativeHi, low, value;
        if constexpr (Write) {
            Duplicate(negativeHi, -mean.hi, full);
            Duplicate(low, mean.lo, full);
        }
        uint32_t remaining = count;
        for (uint16_t i = 0; i < loops; ++i) {
            MaskReg active = UpdateMask<float>(remaining);
            AscendC::Reg::Gather(value, src + i * step, offsets, active);
            if constexpr (Write) {
                SubtractMeanWithResidual(value, negativeHi, low, active);
                AscendC::Reg::Scatter(dst + i * step, value, offsets, active);
            } else {
                StoreAlign(dst + i * V_LENGTH, value, active);
            }
        }
    }
}

template <bool Write>
__aicore__ inline void InterleavedPass(int64_t base, int64_t pos, int64_t end, int64_t groups,
                                       LargeCompensatedSum* sums, const MeanHiLo* means)
{
    const int64_t suffix = tiling_->irregularReduceSuffixElements;
    const int64_t bytes = Min(tiling_->genericSumBufferBytes,
                              Min(tiling_->genericInputBufferBytes, tiling_->genericOutputBufferBytes));
    const int64_t capacity = (bytes / (sizeof(float) * V_LENGTH) - 2) * V_LENGTH;
    while (pos < end) {
        const auto tile = GetInterleavedTile(base, pos, end - pos, capacity, groups * suffix);
        const auto dma = tile.dma;
        CopyInRowsPadded(dma.offset, dma.rows, dma.width, dma.ubStride, dma.gmStride);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        LocalTensor<T> output = outQueue_.AllocTensor<T>();
        auto values = input.template ReinterpretCast<float>();
        auto centered = output.template ReinterpretCast<float>();
        auto scratch = sumBuf_.Get<float>();
        auto gathered = scratch[V_LENGTH];
        const uint32_t stored = static_cast<uint32_t>(dma.rows == 1 ? dma.width : dma.rows * dma.ubStride);
        if constexpr (!IsSameType<T, float>::value) {
            values = Write ? scratch : centered;
            Cast(values, input, RoundMode::CAST_NONE, stored);
            PipeBarrier<PIPE_V>();
            if constexpr (Write) {
                centered = input.template ReinterpretCast<float>();
            }
        }
        const uint32_t count = static_cast<uint32_t>(tile.localRows * suffix);
        for (int64_t g = 0; g < groups; ++g) {
            if constexpr (Write) {
                CopyInterleavedColumn<true>(values, centered, count, tile.pitch, g, suffix, means[g]);
            } else {
                CopyInterleavedColumn<false>(values, gathered, count, tile.pitch, g, suffix, {});
                PipeBarrier<PIPE_V>();
                ReduceTileWithResidual(gathered, scratch, count);
                SetFlag<HardEvent::V_S>(eventVToS_);
                WaitFlag<HardEvent::V_S>(eventVToS_);
                const float high = scratch.GetValue(0);
                AddLargeCompensated(sums[g], high);
                if (high <= 0x1.fffffep127f && high >= -0x1.fffffep127f) {
                    AddLargeCompensated(sums[g], scratch.GetValue(kResidualOffset));
                }
            }
        }
        if constexpr (Write) {
            if constexpr (!IsSameType<T, float>::value) {
                PipeBarrier<PIPE_V>();
                Cast(output, centered, RoundMode::CAST_RINT, stored);
            }
            outQueue_.EnQue(output);
            CopyOutRowsPadded(dma.offset, dma.rows, dma.width, dma.ubStride, dma.gmStride);
        } else {
            outQueue_.FreeTensor(output);
        }
        inQueue_.FreeTensor(input);
        pos += tile.periods;
    }
}

__aicore__ inline void MergeInterleavedPartials(int64_t task, int64_t groups, LargeCompensatedSum* sums)
{
    const int64_t parts = tiling_->coresPerRow;
    const uint32_t stride = static_cast<uint32_t>((groups + 7) / 8 * 8);
    LocalTensor<float> output = outQueue_.AllocTensor<float>();
    for (int64_t g = 0; g < groups; ++g) {
        output.SetValue(g, sums[g].sum);
        output.SetValue(stride + g, sums[g].correction);
    }
    outQueue_.EnQue(output);
    output = outQueue_.DeQue<float>();
    SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
    DataCopy(workspaceGm_[task * 2 * stride], output, 2 * stride);
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    outQueue_.FreeTensor(output);
    SyncAll();

    LocalTensor<float> partials = inQueue_.AllocTensor<float>();
    DataCopy(partials, workspaceGm_[task / parts * parts * 2 * stride], static_cast<uint32_t>(parts * 2 * stride));
    inQueue_.EnQue(partials);
    partials = inQueue_.DeQue<float>();
    auto merged = sumBuf_.Get<float>();
    auto* src = (__local_mem__ float*)partials.GetPhyAddr();
    auto* dst = (__local_mem__ float*)merged.GetPhyAddr();
    const uint16_t chunks = static_cast<uint16_t>(parts);
    __VEC_SCOPE__
    {
        using namespace AscendC::Reg;
        uint32_t activeCount = static_cast<uint32_t>(groups);
        MaskReg active = UpdateMask<float>(activeCount);
        RegTensor<float> hi, lo, value;
        Duplicate(hi, 0.0f, active);
        Duplicate(lo, 0.0f, active);
        for (uint16_t c = 0; c < chunks; ++c) {
            LoadAlign(value, src + c * stride * 2);
            AddWithResidual(hi, lo, value, active);
            LoadAlign(value, src + c * stride * 2 + stride);
            AddWithResidual(hi, lo, value, active);
        }
        StoreAlign(dst, hi, active);
        StoreAlign(dst + V_LENGTH, lo, active);
    }
    SetFlag<HardEvent::V_S>(eventVToS_);
    WaitFlag<HardEvent::V_S>(eventVToS_);
    for (int64_t g = 0; g < groups; ++g) {
        sums[g] = {merged.GetValue(g), merged.GetValue(V_LENGTH + g)};
    }
    inQueue_.FreeTensor(partials);
}

__aicore__ inline void ProcessInterleavedColumns(uint32_t core, uint32_t cores)
{
    const int64_t groups = tiling_->keepDims[tiling_->keepRank - 1];
    const int64_t prefixCount = tiling_->groupCount / groups;
    const int64_t parts = tiling_->coresPerRow;
    const int64_t rows = tiling_->reduceCount / tiling_->irregularReduceSuffixElements;
    for (int64_t task = core; task < prefixCount * parts; task += cores) {
        const int64_t first = task / parts * groups;
        const int64_t base = GroupBase(first);
        const int64_t part = task % parts;
        const int64_t begin = rows / parts * part + Min(part, rows % parts);
        const int64_t end = begin + rows / parts + (part < rows % parts ? 1 : 0);
        LargeCompensatedSum sums[kMaxInterleavedGroups]{};
        MeanHiLo means[kMaxInterleavedGroups];
        InterleavedPass<false>(base, begin, end, groups, sums, means);
        if (parts > 1) {
            MergeInterleavedPartials(task, groups, sums);
        }
        for (int64_t g = 0; g < groups; ++g) {
            float restore = 1.0f;
            if (!IsFinitePair(sums[g].sum, sums[g].correction)) {
                if constexpr (IsSameType<T, float>::value) {
                    restore = ReductionRestoreFactor(tiling_->reduceCount);
                }
                sums[g] = WalkStridedSuffix<false>(GroupBase(first + g), 0, tiling_->reduceCount, {},
                                                   tiling_->stridedReduceAxis, 1.0f / restore);
            }
            means[g] = FinalizeMeanHiLo(sums[g], tiling_->reduceCount);
            means[g].hi *= restore;
            means[g].lo *= restore;
        }
        InterleavedPass<true>(base, begin, end, groups, sums, means);
    }
}

#endif
