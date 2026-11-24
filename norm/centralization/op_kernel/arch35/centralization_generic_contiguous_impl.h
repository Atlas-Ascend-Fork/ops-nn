/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_GENERIC_CONTIGUOUS_IMPL_H_
#define OPS_NORM_CENTRALIZATION_GENERIC_CONTIGUOUS_IMPL_H_

__aicore__ inline void ProcessSmallInnerOneTask(int64_t outer)
{
    const int64_t blockElements = tiling_->smallContiguousBlockElements;
    const int64_t base = ContiguousBase(outer, 0);
    CopyInContiguous(base, blockElements);
    LocalTensor<T> input = inQueue_.DeQue<T>();
    LocalTensor<T> output = outQueue_.AllocTensor<T>();
    const float sum = ReduceSegmentToScalar(input, blockElements);
    const float mean = sum / static_cast<float>(tiling_->reduceCount);
    CentralizeSegmentScalar(input, output, blockElements, mean);
    PipeBarrier<PIPE_V>();
    outQueue_.EnQue(output);
    inQueue_.FreeTensor(input);
    CopyOutContiguous(base, blockElements);
}

__aicore__ inline void ProcessSmallWholeBlockTask(int64_t outer)
{
    const int64_t meanElements = tiling_->smallContiguousMeanElements;
    const int64_t rowElements = tiling_->contiguousInnerCount;
    const int64_t rowStride = AlignToVector(rowElements);
    const int64_t base = ContiguousBase(outer, 0);
    CopyInSmallPadded(base, rowElements, rowStride);
    LocalTensor<T> input = inQueue_.DeQue<T>();
    LocalTensor<T> output = outQueue_.AllocTensor<T>();
    LocalTensor<float> sum = sumBuf_.Get<float>();
    const uint16_t numSlots = static_cast<uint16_t>(tiling_->numAccSlots > 0 ? tiling_->numAccSlots : 1);
    const int64_t sumStride = AlignToVector(tiling_->smallContiguousMeanElements);
    for (uint16_t slot = 0; slot < numSlots; ++slot) {
        ZeroSumTileAt(sum, static_cast<int64_t>(slot) * sumStride, meanElements);
    }
    PipeBarrier<PIPE_V>();
    uint16_t slot = 0;
    for (int64_t r = 0; r < tiling_->reduceCount; ++r) {
        const int64_t rowOffset = r * rowStride;
        AccumulateContiguousAt(input, sum, static_cast<int64_t>(slot) * sumStride, rowElements, rowOffset);
        slot = static_cast<uint16_t>((slot + 1 == numSlots) ? 0 : (slot + 1));
    }
    FoldAccumulatorSlots(sum, numSlots, sumStride, meanElements);
    PipeBarrier<PIPE_V>();
    ScaleSumToMeanAt(sum, 0, meanElements);
    PipeBarrier<PIPE_V>();
    for (int64_t r = 0; r < tiling_->reduceCount; ++r) {
        const int64_t rowOffset = r * rowStride;
        CentralizeContiguousAt(input, output, sum, 0, rowElements, rowOffset, rowOffset);
    }
    PipeBarrier<PIPE_V>();
    outQueue_.EnQue(output);
    inQueue_.FreeTensor(input);
    CopyOutSmallPadded(base, rowElements, rowStride);
}

__aicore__ inline void ProcessSmallWholeBlockBatchTask(int64_t outerStart, int64_t outerCount, bool packed)
{
    const int64_t reduceCount = tiling_->reduceCount;
    const int64_t innerCount = tiling_->contiguousInnerCount;
    const int64_t blockElements = reduceCount * innerCount;
    const int64_t totalElements = outerCount * blockElements;
    const int64_t base = ContiguousBase(outerStart, 0);
    CopyInContiguous(base, totalElements);
    LocalTensor<T> input = inQueue_.DeQue<T>();
    LocalTensor<T> output = outQueue_.AllocTensor<T>();
    if (packed) {
        LocalTensor<float> values = input.template ReinterpretCast<float>();
        LocalTensor<float> centered = output.template ReinterpretCast<float>();
        if constexpr (IsSameType<T, half>::value) {
            // Once converted, the input buffer can hold the FP32 result for the final cast.
            values = sumBuf_.Get<float>();
            centered = input.template ReinterpretCast<float>();
            Cast(values, input, RoundMode::CAST_NONE, static_cast<uint32_t>(totalElements));
            PipeBarrier<PIPE_V>();
        }
        for (uint32_t col = 0; col < static_cast<uint32_t>(innerCount); ++col) {
            CenterPackedRows(values, centered, static_cast<uint32_t>(outerCount), static_cast<uint16_t>(reduceCount),
                             static_cast<uint32_t>(innerCount), col);
        }
        if constexpr (IsSameType<T, half>::value) {
            PipeBarrier<PIPE_V>();
            Cast(output, centered, RoundMode::CAST_RINT, static_cast<uint32_t>(totalElements));
        }
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
        CopyOutContiguous(base, totalElements);
        return;
    }
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    for (int64_t outer = 0; outer < outerCount; ++outer) {
        const int64_t outerBase = outer * blockElements;
        for (int64_t inner = 0; inner < innerCount; ++inner) {
            LargeCompensatedSum sum{0.0f, 0.0f};
            for (int64_t r = 0; r < reduceCount; ++r) {
                AddLargeCompensated(sum, static_cast<float>(input.GetValue(outerBase + r * innerCount + inner)));
            }
            const MeanHiLo mean = FinalizeMeanHiLo(sum, reduceCount);
            for (int64_t r = 0; r < reduceCount; ++r) {
                const float value = static_cast<float>(input.GetValue(outerBase + r * innerCount + inner));
                output.SetValue(outerBase + r * innerCount + inner, static_cast<T>((value - mean.hi) - mean.lo));
            }
        }
    }
    SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
    outQueue_.EnQue(output);
    inQueue_.FreeTensor(input);
    CopyOutContiguous(base, totalElements);
}

__aicore__ inline void ProcessSmallContiguous(uint32_t core, uint32_t cores)
{
    if (tiling_->smallContiguousMode == 1) {
        const int64_t taskCount = tiling_->contiguousOuterCount;
        for (int64_t task = core; task < taskCount; task += cores) {
            ProcessSmallInnerOneTask(task);
        }
        return;
    }
    const int64_t blockElements = tiling_->smallContiguousBlockElements;
    const int64_t bufferElements = blockElements > 0 ?
                                       static_cast<int64_t>(tiling_->genericInputBufferBytes / sizeof(T)) :
                                       0;
    constexpr int64_t kBatchTargetBytes = 16 * 1024;
    const int64_t targetElements = kBatchTargetBytes / static_cast<int64_t>(sizeof(T));
    int64_t maxBatch = blockElements > 0 ? bufferElements / blockElements : 0;
    if (maxBatch < 1) {
        maxBatch = 1;
    }
    int64_t targetBatch = blockElements > 0 ? targetElements / blockElements : 1;
    if (targetBatch < 1) {
        targetBatch = 1;
    } else if (targetBatch > maxBatch) {
        targetBatch = maxBatch;
    }
    const int64_t blockRows = 32 / sizeof(T);
    const int64_t width = tiling_->reduceCount;
    const bool packedWidth = (IsSameType<T, float>::value && (width <= 4 || width == 6)) ||
                             (IsSameType<T, half>::value && width <= 8);
    if constexpr (IsSameType<T, half>::value) {
        if (packedWidth && blockElements <= 2 * V_LENGTH) {
            // Both the converted input and the centered tile occupy FP32 storage.
            const int64_t floatBatch = Min(tiling_->genericInputBufferBytes, tiling_->genericSumBufferBytes) /
                                       static_cast<int64_t>(sizeof(float)) / blockElements;
            const int64_t outerPerCore = (tiling_->contiguousOuterCount + cores - 1) / cores;
            targetBatch = Min(targetBatch, Min(floatBatch, AlignToBlock(outerPerCore)));
            targetBatch = targetBatch > 0 ? targetBatch : 1;
        }
    }
    const bool packed = packedWidth && blockElements <= 2 * V_LENGTH && tiling_->contiguousOuterCount >= V_LENGTH &&
                        targetBatch >= blockRows;
    const int64_t outerBatch = packed ? targetBatch / blockRows * blockRows : targetBatch;
    const int64_t taskCount = (tiling_->contiguousOuterCount + outerBatch - 1) / outerBatch;
    for (int64_t task = core; task < taskCount; task += cores) {
        const int64_t outerStart = task * outerBatch;
        const int64_t outerCount = Min(outerBatch, tiling_->contiguousOuterCount - outerStart);
        ProcessSmallWholeBlockBatchTask(outerStart, outerCount, packed);
    }
}

__aicore__ inline void AccumulateRowsSmallInner(LocalTensor<T>& input, LocalTensor<float>& sum, int64_t validRows,
                                                int64_t innerCount, int64_t rowStride)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
    __local_mem__ float* sumAddr = (__local_mem__ float*)sum.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> sumReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        uint32_t activeCount = static_cast<uint32_t>(innerCount);
        const uint16_t rowLoops = static_cast<uint16_t>(validRows);
        MaskReg preg = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
        LoadRegForDtype<float>(sumAddr, sumReg, preg, 0);
        for (uint16_t r = 0; r < rowLoops; ++r) {
            LoadRegForDtype<T>(inputAddr, xReg, preg, static_cast<uint32_t>(r * rowStride));
            Add(sumReg, sumReg, xReg, preg);
        }
        StoreRegForDtype<float>(sumAddr, sumReg, preg, 0);
    }
}

__aicore__ inline void CentralizeRowsSmallInner(LocalTensor<T>& input, LocalTensor<T>& output, LocalTensor<float>& mean,
                                                int64_t validRows, int64_t innerCount, int64_t rowStride)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
    __local_mem__ T* outputAddr = (__local_mem__ T*)output.GetPhyAddr();
    __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> meanReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        uint32_t activeCount = static_cast<uint32_t>(innerCount);
        const uint16_t rowLoops = static_cast<uint16_t>(validRows);
        MaskReg preg = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
        LoadRegForDtype<float>(meanAddr, meanReg, preg, 0);
        for (uint16_t r = 0; r < rowLoops; ++r) {
            const uint32_t rowOffset = static_cast<uint32_t>(r * rowStride);
            LoadRegForDtype<T>(inputAddr, xReg, preg, rowOffset);
            Sub(xReg, xReg, meanReg, preg);
            StoreRegForDtype<T>(outputAddr, xReg, preg, rowOffset);
        }
    }
}

// colOffset 指向本批的起始列。每个向量通道沿行序累加一列。
__aicore__ inline void AccumulateRowsSmallInnerKahan(LocalTensor<T>& input, LocalTensor<float>& sum, int64_t sumOffset,
                                                     int64_t corrOffset, int64_t validRows, int64_t innerCount,
                                                     int64_t rowStride, int64_t colOffset = 0)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
    __local_mem__ float* sumAddr = (__local_mem__ float*)sum.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> sumReg;
        RegTensor<float> corrReg;
        RegTensor<float> tReg;
        RegTensor<float> c1Reg;
        RegTensor<float> c2Reg;
        RegTensor<float> addReg;
        RegTensor<float> absSum;
        RegTensor<float> absX;
        MaskReg dom;
        MaskReg fin;
        RegTensor<float> zeroReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        uint32_t activeCount = static_cast<uint32_t>(innerCount);
        const uint16_t rowLoops = static_cast<uint16_t>(validRows);
        MaskReg preg = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
        Duplicate(zeroReg, 0.0f, pregFull);
        LoadRegForDtype<float>(sumAddr, sumReg, preg, static_cast<uint32_t>(sumOffset));
        LoadRegForDtype<float>(sumAddr, corrReg, preg, static_cast<uint32_t>(corrOffset));
        for (uint16_t r = 0; r < rowLoops; ++r) {
            LoadRegForDtype<T>(inputAddr, xReg, preg, static_cast<uint32_t>(colOffset + r * rowStride));
            Add(tReg, sumReg, xReg, preg);
            Abs(absSum, sumReg, preg);
            Abs(absX, xReg, preg);
            Compare<float, CMPMODE::GE>(dom, absSum, absX, preg);
            Sub(c1Reg, sumReg, tReg, preg);
            Add(c1Reg, c1Reg, xReg, preg);
            Sub(c2Reg, xReg, tReg, preg);
            Add(c2Reg, c2Reg, sumReg, preg);
            Select(addReg, c1Reg, c2Reg, dom);
            Abs(absSum, tReg, preg);
            Compares<float, CMPMODE::LE>(fin, absSum, 3.4028235e38f, preg);
            Select(addReg, addReg, zeroReg, fin);
            Add(corrReg, corrReg, addReg, preg);
            sumReg = tReg;
        }
        StoreRegForDtype<float>(sumAddr, sumReg, preg, static_cast<uint32_t>(sumOffset));
        StoreRegForDtype<float>(sumAddr, corrReg, preg, static_cast<uint32_t>(corrOffset));
    }
}

// colOffset 同上：宽 tile 中心化按 64-lane 列块轮转
__aicore__ inline void CentralizeRowsSmallInnerHiLo(LocalTensor<T>& input, LocalTensor<T>& output,
                                                    LocalTensor<float>& mean, int64_t meanOffset, int64_t meanLowOffset,
                                                    int64_t validRows, int64_t innerCount, int64_t rowStride,
                                                    int64_t colOffset = 0)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
    __local_mem__ T* outputAddr = (__local_mem__ T*)output.GetPhyAddr();
    __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> hiReg;
        RegTensor<float> loReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        uint32_t activeCount = static_cast<uint32_t>(innerCount);
        const uint16_t rowLoops = static_cast<uint16_t>(validRows);
        MaskReg preg = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
        LoadRegForDtype<float>(meanAddr, hiReg, preg, static_cast<uint32_t>(meanOffset));
        LoadRegForDtype<float>(meanAddr, loReg, preg, static_cast<uint32_t>(meanLowOffset));
        Muls(hiReg, hiReg, -1.0f, preg);
        for (uint16_t r = 0; r < rowLoops; ++r) {
            const uint32_t rowOffset = static_cast<uint32_t>(colOffset + r * rowStride);
            LoadRegForDtype<T>(inputAddr, xReg, preg, rowOffset);
            SubtractMeanWithResidual(xReg, hiReg, loReg, preg);
            StoreRegForDtype<T>(outputAddr, xReg, preg, rowOffset);
        }
    }
}

__aicore__ inline void ProcessLargeContiguousSmallInnerTask(int64_t outer)
{
    const int64_t innerCount = tiling_->contiguousInnerCount;
    const int64_t reduceTile = tiling_->largeContiguousReduceTile;
    const int64_t rowStride = OptimalRowStride(innerCount);
    const int64_t planeBase = outer * tiling_->reduceCount * innerCount;
    LocalTensor<float> sum = sumBuf_.Get<float>();
    const int64_t corrOffset = AlignToVector(innerCount);
    ZeroSumTile(sum, innerCount);
    ZeroSumTileAt(sum, corrOffset, innerCount);
    PipeBarrier<PIPE_V>();

    for (int64_t reduceOffset = 0; reduceOffset < tiling_->reduceCount; reduceOffset += reduceTile) {
        const int64_t validRows = Min(reduceTile, tiling_->reduceCount - reduceOffset);
        CopyInRowsPadded(planeBase + reduceOffset * innerCount, validRows, innerCount, rowStride);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        AccumulateRowsSmallInnerKahan(input, sum, 0, corrOffset, validRows, innerCount, rowStride);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(input);
    }

    FinalizeMeanHiLoAt(sum, 0, corrOffset, innerCount);
    PipeBarrier<PIPE_V>();

    for (int64_t reduceOffset = 0; reduceOffset < tiling_->reduceCount; reduceOffset += reduceTile) {
        const int64_t validRows = Min(reduceTile, tiling_->reduceCount - reduceOffset);
        const int64_t gmOffset = planeBase + reduceOffset * innerCount;
        CopyInRowsPadded(gmOffset, validRows, innerCount, rowStride);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        LocalTensor<T> output = outQueue_.AllocTensor<T>();
        CentralizeRowsSmallInnerHiLo(input, output, sum, 0, corrOffset, validRows, innerCount, rowStride);
        PipeBarrier<PIPE_V>();
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
        CopyOutRowsPadded(gmOffset, validRows, innerCount, rowStride);
    }
}

__aicore__ inline void ProcessLargeContiguousSmallInner(uint32_t core, uint32_t cores)
{
    const int64_t taskCount = tiling_->contiguousOuterCount;
    for (int64_t task = core; task < taskCount; task += cores) {
        ProcessLargeContiguousSmallInnerTask(task);
    }
}

// 行批累加：一次队列周期处理多行。
//   validElements <= V_LENGTH：寄存器驻留 Neumaier（复用 lcsi 的行级实现，sum/corr 全程在寄存器，
//     仅批首 load / 批尾 store），槽位按批轮转——批内为补偿链，批间靠槽位折叠缩短链长；
//   宽列按 V_LENGTH 分块，所有列块处理完后再轮转当前批次的累加槽。
__aicore__ inline void AccumulateContiguousRows(LocalTensor<T>& input, LocalTensor<float>& sum, uint16_t numSlots,
                                                int64_t sumStride, int64_t corrBase, int64_t rows,
                                                int64_t validElements, int64_t rowStride, uint16_t& slot)
{
    const int64_t sumOffset = static_cast<int64_t>(slot) * sumStride;
    if (validElements <= V_LENGTH) {
        AccumulateRowsSmallInnerKahan(input, sum, sumOffset, corrBase + sumOffset, rows, validElements, rowStride);
        slot = static_cast<uint16_t>((slot + 1 == numSlots) ? 0 : (slot + 1));
        return;
    }
    const int64_t colChunks = (validElements + V_LENGTH - 1) / V_LENGTH;
    for (int64_t cc = 0; cc < colChunks; ++cc) {
        const int64_t chunkValid = Min(V_LENGTH, validElements - cc * V_LENGTH);
        AccumulateRowsSmallInnerKahan(input, sum, sumOffset + cc * V_LENGTH, corrBase + sumOffset + cc * V_LENGTH, rows,
                                      chunkValid, rowStride, cc * V_LENGTH);
    }
    slot = static_cast<uint16_t>((slot + 1 == numSlots) ? 0 : (slot + 1));
}

// 行批中心化：窄列寄存器驻留（复用 lcsi 实现），宽列按 64-lane 列块轮转。
__aicore__ inline void CentralizeContiguousRows(LocalTensor<T>& input, LocalTensor<T>& output, LocalTensor<float>& mean,
                                                int64_t meanOffset, int64_t meanLowOffset, int64_t rows,
                                                int64_t validElements, int64_t rowStride)
{
    if (validElements <= V_LENGTH) {
        CentralizeRowsSmallInnerHiLo(input, output, mean, meanOffset, meanLowOffset, rows, validElements, rowStride);
        return;
    }
    const int64_t colChunks = (validElements + V_LENGTH - 1) / V_LENGTH;
    for (int64_t cc = 0; cc < colChunks; ++cc) {
        const int64_t chunkValid = Min(V_LENGTH, validElements - cc * V_LENGTH);
        CentralizeRowsSmallInnerHiLo(input, output, mean, meanOffset + cc * V_LENGTH, meanLowOffset + cc * V_LENGTH,
                                     rows, chunkValid, rowStride, cc * V_LENGTH);
    }
}

__aicore__ inline void ProcessContiguousTask(int64_t outer, int64_t innerOffset, int64_t validElements)
{
    LocalTensor<float> sum = sumBuf_.Get<float>();
    const uint16_t numSlots = static_cast<uint16_t>(tiling_->numAccSlots > 0 ? tiling_->numAccSlots : 1);
    const int64_t sumStride = kLargeChunkElements;
    const int64_t corrBase = static_cast<int64_t>(numSlots) * sumStride;
    for (uint16_t slot = 0; slot < numSlots; ++slot) {
        ZeroSumTileAt(sum, static_cast<int64_t>(slot) * sumStride, validElements);
        ZeroSumTileAt(sum, corrBase + static_cast<int64_t>(slot) * sumStride, validElements);
    }
    PipeBarrier<PIPE_V>();

    const int64_t base = ContiguousBase(outer, innerOffset);
    const int64_t innerCount = tiling_->contiguousInnerCount;
    const int64_t rowStride = OptimalRowStride(validElements);
    const int64_t bufferElements = static_cast<int64_t>(tiling_->genericInputBufferBytes / sizeof(T));
    // UB 能放下一个对齐行时使用行批搬运。GM 行间空隙以字节计，UB 行间空隙以 32 字节块计。
    const bool batchRows = bufferElements >= rowStride;
    constexpr int64_t kMaxAbiRows = 65535;
    const int64_t rowsPerBatch = batchRows ? Min(Min(bufferElements / rowStride, kMaxAbiRows), tiling_->reduceCount) :
                                             1;

    uint16_t slot = 0;
    for (int64_t reduceOffset = 0; reduceOffset < tiling_->reduceCount; reduceOffset += rowsPerBatch) {
        const int64_t rows = Min(rowsPerBatch, tiling_->reduceCount - reduceOffset);
        const int64_t gmOffset = base + reduceOffset * innerCount;
        if (batchRows) {
            CopyInRowsPadded(gmOffset, rows, validElements, rowStride, innerCount);
        } else {
            CopyInContiguous(gmOffset, validElements);
        }
        LocalTensor<T> input = inQueue_.DeQue<T>();
        if (batchRows) {
            AccumulateContiguousRows(input, sum, numSlots, sumStride, corrBase, rows, validElements, rowStride, slot);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(input);
        } else {
            AccumulateContiguousNeumaierAt(input, sum, static_cast<int64_t>(slot) * sumStride,
                                           corrBase + static_cast<int64_t>(slot) * sumStride, validElements);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(input);
            slot = static_cast<uint16_t>((slot + 1 == numSlots) ? 0 : (slot + 1));
        }
    }

    FoldAccumulatorSlotsNeumaier(sum, numSlots, sumStride, corrBase, validElements);
    PipeBarrier<PIPE_V>();
    FinalizeMeanHiLoAt(sum, 0, corrBase, validElements);
    PipeBarrier<PIPE_V>();

    for (int64_t reduceOffset = 0; reduceOffset < tiling_->reduceCount; reduceOffset += rowsPerBatch) {
        const int64_t rows = Min(rowsPerBatch, tiling_->reduceCount - reduceOffset);
        const int64_t gmOffset = base + reduceOffset * innerCount;
        if (batchRows) {
            CopyInRowsPadded(gmOffset, rows, validElements, rowStride, innerCount);
        } else {
            CopyInContiguous(gmOffset, validElements);
        }
        LocalTensor<T> input = inQueue_.DeQue<T>();
        LocalTensor<T> output = outQueue_.AllocTensor<T>();
        if (batchRows) {
            CentralizeContiguousRows(input, output, sum, 0, corrBase, rows, validElements, rowStride);
        } else {
            CentralizeContiguousHiLoAt(input, output, sum, 0, corrBase, validElements);
        }
        PipeBarrier<PIPE_V>();
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
        if (batchRows) {
            CopyOutRowsPadded(gmOffset, rows, validElements, rowStride, innerCount);
        } else {
            CopyOutContiguous(gmOffset, validElements);
        }
    }
}

__aicore__ inline void ProcessContiguous(uint32_t core, uint32_t cores)
{
    const int64_t tileElements = tiling_->contiguousInnerTileElements > 0 ?
                                     Min(tiling_->contiguousInnerTileElements, kLargeChunkElements) :
                                     kLargeChunkElements;
    const int64_t innerTileCount = tiling_->contiguousInnerTileCount;
    const int64_t taskCount = tiling_->contiguousOuterCount * innerTileCount;
    for (int64_t task = core; task < taskCount; task += cores) {
        const int64_t outer = task / innerTileCount;
        const int64_t innerTile = task - outer * innerTileCount;
        const int64_t innerOffset = innerTile * tileElements;
        const int64_t validElements = Min(tileElements, tiling_->contiguousInnerCount - innerOffset);
        ProcessContiguousTask(outer, innerOffset, validElements);
    }
}

#endif
