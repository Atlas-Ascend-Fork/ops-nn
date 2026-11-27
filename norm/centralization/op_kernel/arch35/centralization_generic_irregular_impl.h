/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_GENERIC_IRREGULAR_IMPL_H_
#define OPS_NORM_CENTRALIZATION_GENERIC_IRREGULAR_IMPL_H_

__aicore__ inline void ZeroIrregularVectorSums(LocalTensor<float>& sum, int64_t validKeepElements, int64_t sumStride,
                                               int64_t corrBase, int64_t validElements)
{
    for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
        ZeroSumTileAt(sum, keepLocal * sumStride, validElements);
        ZeroSumTileAt(sum, corrBase + keepLocal * sumStride, validElements);
    }
}

// 向量模式行批：末条目 R_last 的 stride == innerCount（归一化布局恒成立），
// 固定外层坐标 + keepLocal 后，R_last 段 × inner 列是 GM 连续块。
// 行 = R_last 段（GM 行步进 innerCount），列 = inner tile（validElements）。
// 按归约段编号升序遍历，在每段内按行号升序累加。
// segBegin/segEnd 限定 outer reduce 段范围（reduce 分片三阶段用；默认全量）。
__aicore__ inline void AccumulateIrregularVectorTile(int64_t keepBase, int64_t keepStride, int64_t innerOffset,
                                                     int64_t validKeepElements, int64_t sumStride, int64_t corrBase,
                                                     int64_t validElements, LocalTensor<float>& sum,
                                                     int64_t segBegin = 0, int64_t segEnd = -1)
{
    const int64_t innerCount = tiling_->irregularVectorInnerCount;
    const int64_t lastDim = tiling_->reduceDims[tiling_->reduceRank - 1];
    const int64_t rowStride = OptimalRowStride(validElements);
    const int64_t bufferElements = static_cast<int64_t>(tiling_->genericInputBufferBytes / sizeof(T));
    constexpr int64_t kMaxAbiRows = 65535;
    const int64_t rowBudget = bufferElements / rowStride > 0 ? bufferElements / rowStride : 1;
    const int64_t rowsPerBatch = Min(rowBudget, kMaxAbiRows);
    const int64_t segmentEnd = segEnd < 0 ? tiling_->reduceSegmentCount : segEnd;
    int64_t reduceCoords[8] = {};
    // Vector tiles keep the last reduction dimension in the coordinate decoder. Each segment starts at row zero.
    int64_t currentReduceOffset = IrregularReduceOuterOffset(segBegin * lastDim, reduceCoords);
    for (int64_t r = segBegin; r < segmentEnd; ++r) {
        const int64_t reduceOffset = tiling_->useReduceOffsetTable != 0 ? tiling_->reduceSegmentOffsets[r] :
                                                                          currentReduceOffset;
        for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
            const int64_t sumOffset = keepLocal * sumStride;
            const int64_t corrOffset = corrBase + keepLocal * sumStride;
            const int64_t offset = keepBase + keepLocal * keepStride + reduceOffset + innerOffset;
            for (int64_t rowOff = 0; rowOff < lastDim; rowOff += rowsPerBatch) {
                const int64_t rows = Min(rowsPerBatch, lastDim - rowOff);
                CopyInRowsPadded(offset + rowOff * innerCount, rows, validElements, rowStride, innerCount);
                LocalTensor<T> input = inQueue_.DeQue<T>();
                if (validElements <= V_LENGTH) {
                    AccumulateRowsSmallInnerKahan(input, sum, sumOffset, corrOffset, rows, validElements, rowStride);
                    PipeBarrier<PIPE_V>();
                } else {
                    const int64_t colChunks = (validElements + V_LENGTH - 1) / V_LENGTH;
                    for (int64_t cc = 0; cc < colChunks; ++cc) {
                        const int64_t chunkValid = Min(V_LENGTH, validElements - cc * V_LENGTH);
                        AccumulateRowsSmallInnerKahan(input, sum, sumOffset + cc * V_LENGTH, corrOffset + cc * V_LENGTH,
                                                      rows, chunkValid, rowStride, cc * V_LENGTH);
                    }
                    PipeBarrier<PIPE_V>();
                }
                inQueue_.FreeTensor(input);
            }
        }
        if (tiling_->useReduceOffsetTable == 0) {
            AdvanceReduceOffset(currentReduceOffset, reduceCoords, tiling_->reduceRank - 1);
        }
    }
}

__aicore__ inline void FinalizeIrregularVectorSums(LocalTensor<float>& sum, int64_t validKeepElements,
                                                   int64_t sumStride, int64_t corrBase, int64_t validElements)
{
    for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
        const int64_t sumOffset = keepLocal * sumStride;
        FinalizeMeanHiLoAt(sum, sumOffset, corrBase + sumOffset, validElements);
    }
}

__aicore__ inline void WriteIrregularVectorTile(int64_t keepBase, int64_t keepStride, int64_t innerOffset,
                                                int64_t validKeepElements, int64_t sumStride, int64_t corrBase,
                                                int64_t validElements, LocalTensor<float>& sum, int64_t segBegin = 0,
                                                int64_t segEnd = -1)
{
    const int64_t innerCount = tiling_->irregularVectorInnerCount;
    const int64_t lastDim = tiling_->reduceDims[tiling_->reduceRank - 1];
    const int64_t rowStride = OptimalRowStride(validElements);
    const int64_t bufferElements = static_cast<int64_t>(tiling_->genericInputBufferBytes / sizeof(T));
    constexpr int64_t kMaxAbiRows = 65535;
    const int64_t rowBudget = bufferElements / rowStride > 0 ? bufferElements / rowStride : 1;
    const int64_t rowsPerBatch = Min(rowBudget, kMaxAbiRows);
    const int64_t segmentEnd = segEnd < 0 ? tiling_->reduceSegmentCount : segEnd;
    int64_t reduceCoordsSecond[8] = {};
    int64_t currentReduceOffset = IrregularReduceOuterOffset(segBegin * lastDim, reduceCoordsSecond);
    for (int64_t r = segBegin; r < segmentEnd; ++r) {
        const int64_t reduceOffset = tiling_->useReduceOffsetTable != 0 ? tiling_->reduceSegmentOffsets[r] :
                                                                          currentReduceOffset;
        for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
            const int64_t meanOffset = keepLocal * sumStride;
            const int64_t meanLowOffset = corrBase + keepLocal * sumStride;
            const int64_t offset = keepBase + keepLocal * keepStride + reduceOffset + innerOffset;
            for (int64_t rowOff = 0; rowOff < lastDim; rowOff += rowsPerBatch) {
                const int64_t rows = Min(rowsPerBatch, lastDim - rowOff);
                CopyInRowsPadded(offset + rowOff * innerCount, rows, validElements, rowStride, innerCount);
                LocalTensor<T> input = inQueue_.DeQue<T>();
                LocalTensor<T> output = outQueue_.AllocTensor<T>();
                if (validElements <= V_LENGTH) {
                    CentralizeRowsSmallInnerHiLo(input, output, sum, meanOffset, meanLowOffset, rows, validElements,
                                                 rowStride);
                } else {
                    // 宽 tile：64-lane 列块轮转寄存器驻留
                    const int64_t colChunks = (validElements + V_LENGTH - 1) / V_LENGTH;
                    for (int64_t cc = 0; cc < colChunks; ++cc) {
                        const int64_t chunkValid = Min(V_LENGTH, validElements - cc * V_LENGTH);
                        CentralizeRowsSmallInnerHiLo(input, output, sum, meanOffset + cc * V_LENGTH,
                                                     meanLowOffset + cc * V_LENGTH, rows, chunkValid, rowStride,
                                                     cc * V_LENGTH);
                    }
                }
                PipeBarrier<PIPE_V>();
                outQueue_.EnQue(output);
                inQueue_.FreeTensor(input);
                CopyOutRowsPadded(offset + rowOff * innerCount, rows, validElements, rowStride, innerCount);
            }
        }
        if (tiling_->useReduceOffsetTable == 0) {
            AdvanceReduceOffset(currentReduceOffset, reduceCoordsSecond, tiling_->reduceRank - 1);
        }
    }
}

__aicore__ inline void ProcessIrregularVectorTileTask(int64_t keepTileStart, int64_t validKeepElements,
                                                      int64_t innerOffset, int64_t validElements)
{
    LocalTensor<float> sum = sumBuf_.Get<float>();
    const int64_t sumStride = AlignToVector(validElements);
    const int64_t corrBase = validKeepElements * sumStride;
    ZeroIrregularVectorSums(sum, validKeepElements, sumStride, corrBase, validElements);
    PipeBarrier<PIPE_V>();

    const int64_t keepBase = IrregularKeepOuterBase(keepTileStart);
    const int64_t keepStride = tiling_->irregularKeepOuterRank > 0 ?
                                   tiling_->keepStrides[tiling_->irregularKeepOuterRank - 1] :
                                   0;
    AccumulateIrregularVectorTile(keepBase, keepStride, innerOffset, validKeepElements, sumStride, corrBase,
                                  validElements, sum);
    FinalizeIrregularVectorSums(sum, validKeepElements, sumStride, corrBase, validElements);
    PipeBarrier<PIPE_V>();
    WriteIrregularVectorTile(keepBase, keepStride, innerOffset, validKeepElements, sumStride, corrBase, validElements,
                             sum);
}

// ── reduce 分片三阶段（irregularReduceChunks > 1）：对齐 Plane 的 partial → 合并 → 输出。──
// task = (irregularTask, chunk)，chunk 按 outer reduce 段切；块布局 = 任务内 sum 区镜像：
//   [keepLocal*sumStride + col] sum/hi，[corrBase + keepLocal*sumStride + col] corr/lo；
// workspace float 偏移：partial [(task*chunks + c)*blockStride]，
//   mean [taskCount*chunks*blockStride + task*blockStride]。
__aicore__ inline int64_t IrregularPartialFloatOffset(int64_t task, int64_t chunk, int64_t chunks,
                                                      int64_t blockStride) const
{
    return (task * chunks + chunk) * blockStride;
}

__aicore__ inline int64_t IrregularMeanFloatOffset(int64_t task, int64_t taskCount, int64_t chunks,
                                                   int64_t blockStride) const
{
    return taskCount * chunks * blockStride + task * blockStride;
}

// partial 块落 workspace：sum 区 [0, blockFloats) 本身就是连续的 [sum|corr] 布局
// （corrBase == validKeep*sumStride），V_MTE3 同步后由 MTE3 从 UB 直写 GM，
// 免去标量 SetValue 镜像（块可达 512 float，标量搬运是三阶段的主要杂项开销）。
__aicore__ inline void StoreIrregularPartial(int64_t task, int64_t chunk, int64_t chunks, int64_t blockStride,
                                             LocalTensor<float>& sum, int64_t blockFloats)
{
    SetFlag<HardEvent::V_MTE3>(eventVToMte3_);
    WaitFlag<HardEvent::V_MTE3>(eventVToMte3_);
    DataCopy(workspaceGm_[static_cast<uint64_t>(IrregularPartialFloatOffset(task, chunk, chunks, blockStride))], sum,
             static_cast<uint32_t>(blockFloats));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
}

// merge 折叠用的 float 版 Neumaier 向量累加：partial 块与 acc 均为 float
// （AccumulateContiguousNeumaierAt 的输入类型随算子 dtype 走，fp16 算子下不适用）。
__aicore__ inline void AccumulateFloatNeumaierAt(LocalTensor<float>& input, LocalTensor<float>& sum, int64_t sumOffset,
                                                 int64_t corrOffset, int64_t validElements, int64_t inputOffset = 0)
{
    __local_mem__ float* inputAddr = (__local_mem__ float*)input.GetPhyAddr();
    __local_mem__ float* sumAddr = (__local_mem__ float*)sum.GetPhyAddr();
    const uint16_t loops = static_cast<uint16_t>((validElements + V_LENGTH - 1) / V_LENGTH);
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
        Duplicate(zeroReg, 0.0f, pregFull);
        for (uint16_t loop = 0; loop < loops; ++loop) {
            const uint32_t offset = static_cast<uint32_t>(loop) * V_LENGTH;
            const uint32_t remaining = static_cast<uint32_t>(validElements) - offset;
            uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
            MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
            LoadRegForDtype<float>(inputAddr, xReg, pregLoop, static_cast<uint32_t>(inputOffset) + offset);
            LoadRegForDtype<float>(sumAddr, sumReg, pregLoop, static_cast<uint32_t>(sumOffset) + offset);
            LoadRegForDtype<float>(sumAddr, corrReg, pregLoop, static_cast<uint32_t>(corrOffset) + offset);
            Add(tReg, sumReg, xReg, pregLoop);
            Abs(absSum, sumReg, pregLoop);
            Abs(absX, xReg, pregLoop);
            Compare<float, CMPMODE::GE>(dom, absSum, absX, pregLoop);
            Sub(c1Reg, sumReg, tReg, pregLoop);
            Add(c1Reg, c1Reg, xReg, pregLoop);
            Sub(c2Reg, xReg, tReg, pregLoop);
            Add(c2Reg, c2Reg, sumReg, pregLoop);
            Select(addReg, c1Reg, c2Reg, dom);
            Abs(absSum, tReg, pregLoop);
            Compares<float, CMPMODE::LE>(fin, absSum, 3.4028235e38f, pregLoop);
            Select(addReg, addReg, zeroReg, fin);
            Add(corrReg, corrReg, addReg, pregLoop);
            StoreRegForDtype<float>(sumAddr, tReg, pregLoop, static_cast<uint32_t>(sumOffset) + offset);
            StoreRegForDtype<float>(sumAddr, corrReg, pregLoop, static_cast<uint32_t>(corrOffset) + offset);
        }
    }
}

// A chunk can overflow before cancellation with a different chunk. Recompute only affected
// FP32 columns with a bounded power-of-two scale, leaving other columns at their original precision.
__aicore__ inline void RecoverIrregularMeans(LocalTensor<float>& sums, int64_t keepStart, int64_t innerOffset,
                                             int64_t keeps, int64_t stride, int64_t columns)
{
    const int64_t corr = keeps * stride;
    SetFlag<HardEvent::V_S>(eventVToS_);
    WaitFlag<HardEvent::V_S>(eventVToS_);
    for (int64_t k = 0; k < keeps; ++k) {
        for (int64_t col = 0; col < columns; ++col) {
            const int64_t pos = k * stride + col;
            if (IsFinitePair(sums.GetValue(pos), sums.GetValue(corr + pos))) {
                continue;
            }
            const int64_t base = IrregularKeepOuterBase(keepStart + k) + innerOffset + col;
            const float restore = ReductionRestoreFactor(tiling_->reduceCount);
            const auto total = WalkStridedSuffix<false>(base, 0, tiling_->reduceCount, {}, tiling_->reduceRank - 1,
                                                        1.0f / restore, 2 * corr);
            const MeanHiLo mean = FinalizeMeanHiLo(total, tiling_->reduceCount);
            sums.SetValue(pos, mean.hi * restore);
            sums.SetValue(corr + pos, mean.lo * restore);
        }
    }
    SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
}

// 合并 chunks 份 partial → hi/lo 均值（向量 Neumaier 折叠 + 向量 Finalize），落 workspace
__aicore__ inline void MergeIrregularMeans(int64_t task, int64_t taskCount, int64_t chunks, int64_t blockStride,
                                           int64_t validKeepElements, int64_t sumStride, int64_t corrBase,
                                           int64_t validElements, int64_t keepStart, int64_t innerOffset)
{
    const int64_t blockFloats = 2 * validKeepElements * sumStride;
    LocalTensor<float> sum = sumBuf_.Get<float>();
    ZeroSumTileAt(sum, 0, blockFloats);
    PipeBarrier<PIPE_V>();
    for (int64_t c = 0; c < chunks; ++c) {
        LocalTensor<float> input = inQueue_.AllocTensor<float>();
        DataCopy(input, workspaceGm_[static_cast<uint64_t>(IrregularPartialFloatOffset(task, c, chunks, blockStride))],
                 static_cast<uint32_t>(blockFloats));
        inQueue_.EnQue(input);
        input = inQueue_.DeQue<float>();
        // 向量折叠：acc += p.sum（Neumaier），acc += p.corr（Neumaier）——
        // 与 7020 标量折叠 AddLargeCompensated(total, sum); AddLargeCompensated(total, corr) 同语义
        for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
            const int64_t sumOffset = keepLocal * sumStride;
            const int64_t corrOffset = corrBase + keepLocal * sumStride;
            AccumulateFloatNeumaierAt(input, sum, sumOffset, corrOffset, validElements, sumOffset);
            AccumulateFloatNeumaierAt(input, sum, sumOffset, corrOffset, validElements, corrOffset);
        }
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(input);
    }
    for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
        FinalizeMeanHiLoAt(sum, keepLocal * sumStride, corrBase + keepLocal * sumStride, validElements);
    }
    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<T, float>::value) {
        RecoverIrregularMeans(sum, keepStart, innerOffset, validKeepElements, sumStride, validElements);
    }
    // Finalize 后 [0, blockFloats) 即连续 [hi|lo] 均值块，V_MTE3 直写 GM
    SetFlag<HardEvent::V_MTE3>(eventVToMte3_);
    WaitFlag<HardEvent::V_MTE3>(eventVToMte3_);
    DataCopy(workspaceGm_[static_cast<uint64_t>(IrregularMeanFloatOffset(task, taskCount, chunks, blockStride))], sum,
             static_cast<uint32_t>(blockFloats));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
}

__aicore__ inline void ProcessIrregularVectorParallel(uint32_t core, uint32_t cores)
{
    const int64_t tileElements = tiling_->irregularVectorTileElements > 0 ?
                                     Min(tiling_->irregularVectorTileElements, kLargeChunkElements) :
                                     kLargeChunkElements;
    const int64_t innerTileCount = tiling_->irregularVectorTileCount;
    const int64_t keepTileCountPerPrefix = tiling_->irregularKeepTileCountPerPrefix;
    const int64_t taskCount = tiling_->irregularVectorTaskCount;
    const int64_t chunks = tiling_->irregularReduceChunks;
    const int64_t blockStride = tiling_->irregularPartialStride;
    const int64_t totalTasks = taskCount * chunks;
    const int64_t nCores = static_cast<int64_t>(cores);

    // ---- 阶段 1：partial 归约（task = (irregularTask, chunk)）----
    for (int64_t t = static_cast<int64_t>(core); t < totalTasks; t += nCores) {
        const int64_t task = t / chunks;
        const int64_t chunk = t - task * chunks;
        const int64_t segBegin = SuffixSegmentBoundary(chunk);
        const int64_t segEnd = SuffixSegmentBoundary(chunk + 1);
        if (segBegin >= segEnd) {
            continue;
        }
        const int64_t keepTileTask = task / innerTileCount;
        const int64_t innerTile = task - keepTileTask * innerTileCount;
        const int64_t keepPrefix = keepTileTask / keepTileCountPerPrefix;
        const int64_t keepTile = keepTileTask - keepPrefix * keepTileCountPerPrefix;
        const int64_t keepTileStart = keepPrefix * tiling_->irregularKeepInnerCount +
                                      keepTile * tiling_->irregularKeepTileElements;
        const int64_t validKeepElements = Min(
            tiling_->irregularKeepTileElements,
            tiling_->irregularKeepInnerCount - keepTile * tiling_->irregularKeepTileElements);
        const int64_t innerOffset = innerTile * tileElements;
        const int64_t validElements = Min(tileElements, tiling_->irregularVectorInnerCount - innerOffset);
        const int64_t sumStride = AlignToVector(validElements);
        const int64_t corrBase = validKeepElements * sumStride;
        LocalTensor<float> sum = sumBuf_.Get<float>();
        ZeroIrregularVectorSums(sum, validKeepElements, sumStride, corrBase, validElements);
        PipeBarrier<PIPE_V>();
        const int64_t keepBase = IrregularKeepOuterBase(keepTileStart);
        const int64_t keepStride = tiling_->irregularKeepOuterRank > 0 ?
                                       tiling_->keepStrides[tiling_->irregularKeepOuterRank - 1] :
                                       0;
        AccumulateIrregularVectorTile(keepBase, keepStride, innerOffset, validKeepElements, sumStride, corrBase,
                                      validElements, sum, segBegin, segEnd);
        StoreIrregularPartial(task, chunk, chunks, blockStride, sum, 2 * validKeepElements * sumStride);
    }
    SyncAll();

    // ---- 阶段 2：合并均值（按 irregularTask 轮转）----
    for (int64_t task = static_cast<int64_t>(core); task < taskCount; task += nCores) {
        const int64_t keepTileTask = task / innerTileCount;
        const int64_t innerTile = task - keepTileTask * innerTileCount;
        const int64_t keepPrefix = keepTileTask / keepTileCountPerPrefix;
        const int64_t keepTile = keepTileTask - keepPrefix * keepTileCountPerPrefix;
        const int64_t keepTileStart = keepPrefix * tiling_->irregularKeepInnerCount +
                                      keepTile * tiling_->irregularKeepTileElements;
        const int64_t validKeepElements = Min(
            tiling_->irregularKeepTileElements,
            tiling_->irregularKeepInnerCount - keepTile * tiling_->irregularKeepTileElements);
        const int64_t innerOffset = innerTile * tileElements;
        const int64_t validElements = Min(tileElements, tiling_->irregularVectorInnerCount - innerOffset);
        const int64_t sumStride = AlignToVector(validElements);
        const int64_t corrBase = validKeepElements * sumStride;
        MergeIrregularMeans(task, taskCount, chunks, blockStride, validKeepElements, sumStride, corrBase, validElements,
                            keepTileStart, innerOffset);
    }
    SyncAll();

    // ---- 阶段 3：输出（task = (irregularTask, chunk)，均值块按 task 缓存）----
    LocalTensor<float> meanTensor;
    int64_t cachedTask = -1;
    for (int64_t t = static_cast<int64_t>(core); t < totalTasks; t += nCores) {
        const int64_t task = t / chunks;
        const int64_t chunk = t - task * chunks;
        const int64_t segBegin = SuffixSegmentBoundary(chunk);
        const int64_t segEnd = SuffixSegmentBoundary(chunk + 1);
        if (segBegin >= segEnd) {
            continue;
        }
        const int64_t keepTileTask = task / innerTileCount;
        const int64_t innerTile = task - keepTileTask * innerTileCount;
        const int64_t keepPrefix = keepTileTask / keepTileCountPerPrefix;
        const int64_t keepTile = keepTileTask - keepPrefix * keepTileCountPerPrefix;
        const int64_t keepTileStart = keepPrefix * tiling_->irregularKeepInnerCount +
                                      keepTile * tiling_->irregularKeepTileElements;
        const int64_t validKeepElements = Min(
            tiling_->irregularKeepTileElements,
            tiling_->irregularKeepInnerCount - keepTile * tiling_->irregularKeepTileElements);
        const int64_t innerOffset = innerTile * tileElements;
        const int64_t validElements = Min(tileElements, tiling_->irregularVectorInnerCount - innerOffset);
        const int64_t sumStride = AlignToVector(validElements);
        const int64_t corrBase = validKeepElements * sumStride;
        if (task != cachedTask) {
            if (cachedTask >= 0) {
                inQueue_.FreeTensor(meanTensor);
            }
            meanTensor = inQueue_.AllocTensor<float>();
            DataCopy(
                meanTensor,
                workspaceGm_[static_cast<uint64_t>(IrregularMeanFloatOffset(task, taskCount, chunks, blockStride))],
                static_cast<uint32_t>(2 * validKeepElements * sumStride));
            inQueue_.EnQue(meanTensor);
            meanTensor = inQueue_.DeQue<float>();
            cachedTask = task;
        }
        const int64_t keepBase = IrregularKeepOuterBase(keepTileStart);
        const int64_t keepStride = tiling_->irregularKeepOuterRank > 0 ?
                                       tiling_->keepStrides[tiling_->irregularKeepOuterRank - 1] :
                                       0;
        WriteIrregularVectorTile(keepBase, keepStride, innerOffset, validKeepElements, sumStride, corrBase,
                                 validElements, meanTensor, segBegin, segEnd);
    }
    if (cachedTask >= 0) {
        inQueue_.FreeTensor(meanTensor);
    }
}

__aicore__ inline void ProcessIrregularVector(uint32_t core, uint32_t cores)
{
    if (tiling_->irregularVectorizable == 2) {
        ProcessIrregularReduceSuffix(core, cores);
        return;
    }
    if (tiling_->irregularReduceChunks > 1) {
        ProcessIrregularVectorParallel(core, cores);
        return;
    }
    const int64_t tileElements = tiling_->irregularVectorTileElements > 0 ?
                                     Min(tiling_->irregularVectorTileElements, kLargeChunkElements) :
                                     kLargeChunkElements;
    const int64_t innerTileCount = tiling_->irregularVectorTileCount;
    const int64_t keepTileCountPerPrefix = tiling_->irregularKeepTileCountPerPrefix;
    const int64_t taskCount = tiling_->irregularVectorTaskCount;
    for (int64_t task = core; task < taskCount; task += cores) {
        const int64_t keepTileTask = task / innerTileCount;
        const int64_t innerTile = task - keepTileTask * innerTileCount;
        const int64_t keepPrefix = keepTileTask / keepTileCountPerPrefix;
        const int64_t keepTile = keepTileTask - keepPrefix * keepTileCountPerPrefix;
        const int64_t keepTileStart = keepPrefix * tiling_->irregularKeepInnerCount +
                                      keepTile * tiling_->irregularKeepTileElements;
        const int64_t validKeepElements = Min(
            tiling_->irregularKeepTileElements,
            tiling_->irregularKeepInnerCount - keepTile * tiling_->irregularKeepTileElements);
        const int64_t innerOffset = innerTile * tileElements;
        const int64_t validElements = Min(tileElements, tiling_->irregularVectorInnerCount - innerOffset);
        ProcessIrregularVectorTileTask(keepTileStart, validKeepElements, innerOffset, validElements);
    }
}

__aicore__ inline void ProcessIrregularReduceSuffixTask(int64_t keepOuter)
{
    float sum = 0.0f;
    const int64_t keepBase = IrregularKeepOuterBase(keepOuter);
    for (int64_t reduceOuter = 0; reduceOuter < tiling_->irregularReduceOuterCount; ++reduceOuter) {
        const int64_t segmentBase = keepBase + IrregularReduceOuterOffset(reduceOuter);
        for (int64_t segmentOffset = 0; segmentOffset < tiling_->irregularReduceSuffixElements;
             segmentOffset += kLargeChunkElements) {
            const int64_t validElements = Min(kLargeChunkElements,
                                              tiling_->irregularReduceSuffixElements - segmentOffset);
            CopyInContiguous(segmentBase + segmentOffset, validElements);
            LocalTensor<T> input = inQueue_.DeQue<T>();
            sum += ReduceSegmentToScalar(input, validElements);
            inQueue_.FreeTensor(input);
        }
    }
    const float mean = sum / static_cast<float>(tiling_->reduceCount);

    for (int64_t reduceOuter = 0; reduceOuter < tiling_->irregularReduceOuterCount; ++reduceOuter) {
        const int64_t segmentBase = keepBase + IrregularReduceOuterOffset(reduceOuter);
        for (int64_t segmentOffset = 0; segmentOffset < tiling_->irregularReduceSuffixElements;
             segmentOffset += kLargeChunkElements) {
            const int64_t validElements = Min(kLargeChunkElements,
                                              tiling_->irregularReduceSuffixElements - segmentOffset);
            CopyInContiguous(segmentBase + segmentOffset, validElements);
            LocalTensor<T> input = inQueue_.DeQue<T>();
            LocalTensor<T> output = outQueue_.AllocTensor<T>();
            CentralizeSegmentScalar(input, output, validElements, mean);
            PipeBarrier<PIPE_V>();
            outQueue_.EnQue(output);
            inQueue_.FreeTensor(input);
            CopyOutContiguous(segmentBase + segmentOffset, validElements);
        }
    }
}

__aicore__ inline void AccumulateIrregularReduceSuffixTile(int64_t keepBase, int64_t keepStride,
                                                           int64_t validKeepElements, LargeCompensatedSum* sums,
                                                           int64_t begin, int64_t end)
{
    int64_t reduceCoords[8] = {};
    int64_t currentReduceOffset = IrregularReduceOuterOffset(begin, reduceCoords);
    for (int64_t reduceOuter = begin; reduceOuter < end; ++reduceOuter) {
        const int64_t reduceOffset = tiling_->useReduceOffsetTable != 0 ? tiling_->reduceSegmentOffsets[reduceOuter] :
                                                                          currentReduceOffset;
        for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
            const int64_t segmentBase = keepBase + keepLocal * keepStride + reduceOffset;
            LargeCompensatedSum segment{};
            for (int64_t segmentOffset = 0; segmentOffset < tiling_->irregularReduceSuffixElements;
                 segmentOffset += kLargeChunkElements) {
                const int64_t validElements = Min(kLargeChunkElements,
                                                  tiling_->irregularReduceSuffixElements - segmentOffset);
                CopyInContiguous(segmentBase + segmentOffset, validElements);
                LocalTensor<T> input = inQueue_.DeQue<T>();
                if constexpr (IsSameType<T, float>::value) {
                    LocalTensor<float> values = input.template ReinterpretCast<float>();
                    LocalTensor<float> partial = sumBuf_.Get<float>();
                    ReduceTileWithResidual(values, partial, static_cast<uint32_t>(validElements));
                    SetFlag<HardEvent::V_S>(eventVToS_);
                    WaitFlag<HardEvent::V_S>(eventVToS_);
                    const float high = partial.GetValue(0);
                    AddLargeCompensated(segment, high);
                    if (high <= 3.4028235e38f && high >= -3.4028235e38f) {
                        AddLargeCompensated(segment, partial.GetValue(kResidualOffset));
                    }
                } else {
                    AddLargeCompensated(sums[keepLocal], ReduceSegmentToScalar(input, validElements));
                }
                inQueue_.FreeTensor(input);
            }
            if constexpr (IsSameType<T, float>::value) {
                const float high = segment.sum + segment.correction;
                AddLargeCompensated(sums[keepLocal], high);
                if (high <= 3.4028235e38f && high >= -3.4028235e38f) {
                    const float recovered = high - segment.sum;
                    const float low = (segment.sum - (high - recovered)) + (segment.correction - recovered);
                    AddLargeCompensated(sums[keepLocal], low);
                }
            }
        }
        if (tiling_->useReduceOffsetTable == 0) {
            AdvanceReduceOffset(currentReduceOffset, reduceCoords, tiling_->irregularReduceOuterRank);
        }
    }
}

__aicore__ inline void FinalizeIrregularReduceSuffixSums(const LargeCompensatedSum* sums, MeanHiLo* means,
                                                         int64_t validKeepElements)
{
    for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
        means[keepLocal] = FinalizeMeanHiLo(sums[keepLocal], tiling_->reduceCount);
    }
}

__aicore__ inline void WriteCenteredSuffix(int64_t segmentBase, int64_t length, float hi, float lo)
{
    for (int64_t segmentOffset = 0; segmentOffset < length; segmentOffset += kLargeChunkElements) {
        const int64_t validElements = Min(kLargeChunkElements, length - segmentOffset);
        CopyInContiguous(segmentBase + segmentOffset, validElements);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        LocalTensor<T> output = outQueue_.AllocTensor<T>();
        CentralizeSegmentHiLo(input, output, validElements, hi, lo);
        PipeBarrier<PIPE_V>();
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
        CopyOutContiguous(segmentBase + segmentOffset, validElements);
    }
}

__aicore__ inline void WriteIrregularReduceSuffixTile(int64_t keepBase, int64_t keepStride, int64_t validKeepElements,
                                                      const MeanHiLo* means)
{
    int64_t reduceCoordsSecond[8] = {};
    int64_t currentReduceOffset = 0;
    for (int64_t reduceOuter = 0; reduceOuter < tiling_->reduceSegmentCount; ++reduceOuter) {
        const int64_t reduceOffset = tiling_->useReduceOffsetTable != 0 ? tiling_->reduceSegmentOffsets[reduceOuter] :
                                                                          currentReduceOffset;
        for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
            const int64_t segmentBase = keepBase + keepLocal * keepStride + reduceOffset;
            WriteCenteredSuffix(segmentBase, tiling_->irregularReduceSuffixElements, means[keepLocal].hi,
                                means[keepLocal].lo);
        }
        if (tiling_->useReduceOffsetTable == 0) {
            AdvanceReduceOffset(currentReduceOffset, reduceCoordsSecond, tiling_->irregularReduceOuterRank);
        }
    }
}

__aicore__ inline void ProcessIrregularReduceSuffixTileTask(int64_t keepTileStart, int64_t validKeepElements)
{
    LargeCompensatedSum sums[kMaxIrregularScalarKeepTile] = {};
    MeanHiLo means[kMaxIrregularScalarKeepTile] = {};
    const int64_t keepBase = IrregularKeepOuterBase(keepTileStart);
    const int64_t keepStride = tiling_->irregularKeepOuterRank > 0 ?
                                   tiling_->keepStrides[tiling_->irregularKeepOuterRank - 1] :
                                   0;
    AccumulateIrregularReduceSuffixTile(keepBase, keepStride, validKeepElements, sums, 0, tiling_->reduceSegmentCount);
    FinalizeIrregularReduceSuffixSums(sums, means, validKeepElements);
    WriteIrregularReduceSuffixTile(keepBase, keepStride, validKeepElements, means);
}

static constexpr int64_t kPackedSuffixAccElements = (IsSameType<T, float>::value ? 3 : 2) * V_LENGTH;

struct PackedSuffixLayout {
    int64_t suffix;
    int64_t unitsPerPrefix;
    int64_t groupsPerUnit;
    int64_t segmentsPerUnit;
    int64_t planeRun;
    int64_t planeStride;
    int64_t unitsPerTile;
    int16_t groupShift;
    uint32_t minMagnitude;
    uint32_t maxMagnitude;
};

__aicore__ inline bool GetPackedSuffixLayout(PackedSuffixLayout& layout) const
{
    const int64_t keepRank = tiling_->irregularKeepOuterRank;
    const int64_t outerRank = tiling_->irregularReduceOuterRank;
    const int64_t suffix = tiling_->irregularReduceSuffixElements;
    const int64_t maxSuffix = IsSameType<T, half>::value && tiling_->irregularKeepInnerCount >= V_LENGTH ?
                                  2 * V_LENGTH :
                                  V_LENGTH;
    if (keepRank == 0 || outerRank == 0 || suffix > maxSuffix ||
        tiling_->genericSumBufferBytes < kPackedSuffixAccElements * sizeof(float)) {
        return false;
    }
    int64_t planeAxis = outerRank - 1;
    const int64_t innerKeep = tiling_->keepDims[keepRank - 1];
    layout = {suffix, innerKeep, 1, 1, 0, 0, 0, 0, 0, 0};
    if constexpr (IsSameType<T, float>::value) {
        uint32_t bits = 0;
        for (int64_t remaining = tiling_->reduceCount - 1; remaining > 0; remaining >>= 1) {
            ++bits;
        }
        // Allow headroom for the sum and for a one-ULP residual divided by the group size.
        layout.minMagnitude = (25U + bits) << 23;  // 2^(-102 + bits)
        layout.maxMagnitude = (253U - bits) << 23; // 2^(126 - bits)
    }
    // Fold a small interleaved tail into one unit to batch across the preceding keep axis.
    if (keepRank >= 2 && outerRank >= 2 && (innerKeep == 2 || innerKeep == 4) &&
        tiling_->reduceDims[planeAxis] <= V_LENGTH / (innerKeep * suffix)) {
        layout.groupsPerUnit = innerKeep;
        layout.segmentsPerUnit = tiling_->reduceDims[planeAxis];
        layout.unitsPerPrefix = tiling_->keepDims[keepRank - 2];
        layout.groupShift = innerKeep == 2 ? 1 : 2;
        --planeAxis;
    }
    layout.planeRun = tiling_->reduceDims[planeAxis];
    layout.planeStride = tiling_->reduceStrides[planeAxis];
    if (layout.planeStride > UINT32_MAX / sizeof(T)) {
        return false;
    }
    const int64_t unitElements = layout.groupsPerUnit * layout.segmentsPerUnit * suffix;
    const int64_t bufferElements = Min(tiling_->genericInputBufferBytes, tiling_->genericOutputBufferBytes) /
                                   sizeof(float);
    const int64_t alignedCapacity = bufferElements / (32 / sizeof(T)) * (32 / sizeof(T));
    layout.unitsPerTile = Min(V_LENGTH / layout.groupsPerUnit, alignedCapacity / unitElements);
    const int64_t targetUnits = (tiling_->groupCount / layout.groupsPerUnit + GetBlockNum() - 1) / GetBlockNum();
    layout.unitsPerTile = Min(layout.unitsPerTile, targetUnits);
    return layout.unitsPerTile > 0;
}

// Each lane carries one independent group through all of the DMA rows.
template <bool Write>
__aicore__ inline void ComputePackedSuffixRows(LocalTensor<float>& values, LocalTensor<float>& centered,
                                               LocalTensor<float>& sums, const PackedSuffixLayout& layout,
                                               uint32_t groups, uint16_t rows, uint32_t rowStride)
{
    using namespace AscendC::Reg;
    auto* src = (__local_mem__ float*)values.GetPhyAddr();
    auto* dst = (__local_mem__ float*)centered.GetPhyAddr();
    auto* acc = (__local_mem__ float*)sums.GetPhyAddr();
    const uint16_t width = static_cast<uint16_t>(layout.suffix);
    const uint16_t segments = static_cast<uint16_t>(layout.segmentsPerUnit);
    const uint32_t step = layout.groupsPerUnit * width;
    const uint32_t gap = (segments - 1) * step;
    const int16_t shift = layout.groupShift;
    const uint32_t minMagnitude = layout.minMagnitude;
    const uint32_t maxMagnitude = layout.maxMagnitude;
    __VEC_SCOPE__
    {
        uint32_t valid = groups;
        MaskReg active = UpdateMask<float>(valid);
        RegTensor<uint32_t> lane, offsets, extra;
        Arange(reinterpret_cast<RegTensor<int32_t>&>(lane), 0);
        Muls(offsets, lane, static_cast<uint32_t>(width), active);
        if (segments > 1) {
            ShiftRights(extra, lane, shift, active);
            Muls(extra, extra, gap, active);
            Add(offsets, offsets, extra, active);
        }
        RegTensor<float> high, low, value;
        RegTensor<uint32_t> flags, ones, signMask;
        if constexpr (!Write && IsSameType<T, float>::value) {
            LoadAlign(flags, (__local_mem__ uint32_t*)(acc + 2 * V_LENGTH));
            Duplicate(ones, 1U, active);
            Duplicate(signMask, 0x7fffffffU, active);
        }
        LoadAlign(high, acc);
        LoadAlign(low, acc + V_LENGTH);
        if constexpr (Write) {
            Muls(high, high, -1.0f, active);
        }
        for (uint16_t row = 0; row < rows; ++row) {
            for (uint16_t segment = 0; segment < segments; ++segment) {
                const uint32_t base = row * rowStride + segment * step;
                for (uint16_t col = 0; col < width; ++col) {
                    RegTensor<uint32_t> index;
                    Adds(index, offsets, static_cast<uint32_t>(col), active);
                    AscendC::Reg::Gather(value, src + base, index, active);
                    if constexpr (Write) {
                        SubtractMeanWithResidual(value, high, low, active);
                        AscendC::Reg::Scatter(dst + base, value, index, active);
                    } else {
                        if constexpr (IsSameType<T, float>::value) {
                            RegTensor<uint32_t> magnitude;
                            And(magnitude, reinterpret_cast<RegTensor<uint32_t>&>(value), signMask, active);
                            MaskReg small, large, nonzero;
                            Compares<uint32_t, CMPMODE::LT>(small, magnitude, minMagnitude, active);
                            Compares<uint32_t, CMPMODE::GT>(nonzero, magnitude, 0U, active);
                            Compares<uint32_t, CMPMODE::GE>(large, magnitude, maxMagnitude, active);
                            And(small, small, nonzero, active);
                            Or(large, large, small, active);
                            Select(flags, ones, flags, large);
                        }
                        AddWithResidual(high, low, value, active);
                    }
                }
            }
        }
        if constexpr (!Write) {
            if constexpr (IsSameType<T, float>::value) {
                Reduce<ReduceType::MAX>(flags, flags, active);
                MaskReg first = CreateMask<uint32_t, MaskPattern::VL1>();
                StoreAlign((__local_mem__ uint32_t*)(acc + 2 * V_LENGTH), flags, first);
            }
            StoreAlign(acc, high, active);
            StoreAlign(acc + V_LENGTH, low, active);
        }
    }
}

template <bool Write>
__aicore__ inline void RunPackedSuffixPass(const PackedSuffixLayout& layout, int64_t keepBase, int64_t groups,
                                           LocalTensor<float>& sums)
{
    const int64_t planes = tiling_->reduceSegmentCount / layout.segmentsPerUnit;
    const int64_t elements = groups * layout.segmentsPerUnit * layout.suffix;
    const int64_t stride = AlignToBlock(elements);
    const int64_t capacity = Min(tiling_->genericInputBufferBytes, tiling_->genericOutputBufferBytes) / sizeof(float);
    const int64_t batch = Min(4095, capacity / stride);
    for (int64_t plane = 0; plane < planes;) {
        const int64_t rows = Min(batch, layout.planeRun - plane % layout.planeRun);
        const int64_t segment = plane * layout.segmentsPerUnit;
        const int64_t offset = keepBase + (tiling_->useReduceOffsetTable ? tiling_->reduceSegmentOffsets[segment] :
                                                                           IrregularReduceOuterOffset(segment));
        CopyInRowsPadded(offset, rows, elements, stride, layout.planeStride);
        auto input = inQueue_.DeQue<T>();
        auto output = outQueue_.AllocTensor<T>();
        auto values = input.template ReinterpretCast<float>();
        auto centered = output.template ReinterpretCast<float>();
        const uint32_t count = static_cast<uint32_t>(rows * stride);
        if constexpr (!IsSameType<T, float>::value) {
            values = output.template ReinterpretCast<float>();
            centered = input.template ReinterpretCast<float>();
            Cast(values, input, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        }
        ComputePackedSuffixRows<Write>(values, centered, sums, layout, groups, rows, stride);
        PipeBarrier<PIPE_V>();
        if constexpr (Write) {
            if constexpr (!IsSameType<T, float>::value) {
                Cast(output, centered, RoundMode::CAST_RINT, count);
                PipeBarrier<PIPE_V>();
            }
            outQueue_.EnQue(output);
            CopyOutRowsPadded(offset, rows, elements, stride, layout.planeStride);
        } else {
            outQueue_.FreeTensor(output);
        }
        inQueue_.FreeTensor(input);
        plane += rows;
    }
}

__aicore__ inline void ProcessPackedSuffix(uint32_t core, uint32_t cores, const PackedSuffixLayout& layout)
{
    const int64_t tiles = (layout.unitsPerPrefix + layout.unitsPerTile - 1) / layout.unitsPerTile;
    const int64_t prefixGroups = layout.unitsPerPrefix * layout.groupsPerUnit;
    const int64_t tasks = tiling_->groupCount / prefixGroups * tiles;
    auto sums = sumBuf_.Get<float>();
    for (int64_t task = core; task < tasks; task += cores) {
        const int64_t start = task % tiles * layout.unitsPerTile;
        const int64_t groups = Min(layout.unitsPerTile, layout.unitsPerPrefix - start) * layout.groupsPerUnit;
        const int64_t firstGroup = task / tiles * prefixGroups + start * layout.groupsPerUnit;
        const int64_t base = IrregularKeepOuterBase(firstGroup);
        ZeroSumTile(sums, kPackedSuffixAccElements);
        PipeBarrier<PIPE_V>();
        RunPackedSuffixPass<false>(layout, base, groups, sums);
        if constexpr (IsSameType<T, float>::value) {
            SetFlag<HardEvent::V_S>(eventVToS_);
            WaitFlag<HardEvent::V_S>(eventVToS_);
            if (sums.template ReinterpretCast<uint32_t>().GetValue(2 * V_LENGTH) != 0) {
                // Preserve segment-first summation for extreme values before writing any output.
                const int64_t inner = tiling_->irregularKeepInnerCount;
                for (int64_t group = firstGroup; group < firstGroup + groups;) {
                    const int64_t count = Min(kMaxIrregularScalarKeepTile,
                                              Min(firstGroup + groups - group, inner - group % inner));
                    ProcessIrregularReduceSuffixTileTask(group, count);
                    group += count;
                }
                continue;
            }
        }
        FinalizeMeanHiLoAt(sums, 0, V_LENGTH, groups);
        PipeBarrier<PIPE_V>();
        RunPackedSuffixPass<true>(layout, base, groups, sums);
    }
}

__aicore__ inline void ProcessIrregularReduceSuffix(uint32_t core, uint32_t cores)
{
    PackedSuffixLayout layout{};
    if (GetPackedSuffixLayout(layout)) {
        ProcessPackedSuffix(core, cores, layout);
        return;
    }
    const int64_t keepTileCountPerPrefix = tiling_->irregularKeepTileCountPerPrefix;
    const int64_t taskCount = tiling_->irregularKeepTileTaskCount;
    for (int64_t task = core; task < taskCount; task += cores) {
        const int64_t keepPrefix = task / keepTileCountPerPrefix;
        const int64_t keepTile = task - keepPrefix * keepTileCountPerPrefix;
        const int64_t keepTileStart = keepPrefix * tiling_->irregularKeepInnerCount +
                                      keepTile * tiling_->irregularKeepTileElements;
        const int64_t validKeepElements = Min(
            tiling_->irregularKeepTileElements,
            tiling_->irregularKeepInnerCount - keepTile * tiling_->irregularKeepTileElements);
        ProcessIrregularReduceSuffixTileTask(keepTileStart, validKeepElements);
    }
}

#endif
