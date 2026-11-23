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
// 累加顺序与旧逐段实现一致（外层坐标升序、段内 c_last 升序），结果逐位相同。
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
    int64_t currentReduceOffset = 0;
    for (int64_t r = 0; r < segmentEnd; ++r) {
        const int64_t reduceOffset = tiling_->useReduceOffsetTable != 0 ? tiling_->reduceSegmentOffsets[r] :
                                                                          currentReduceOffset;
        if (r >= segBegin) {
            for (int64_t keepLocal = 0; keepLocal < validKeepElements; ++keepLocal) {
                const int64_t sumOffset = keepLocal * sumStride;
                const int64_t corrOffset = corrBase + keepLocal * sumStride;
                const int64_t offset = keepBase + keepLocal * keepStride + reduceOffset + innerOffset;
                for (int64_t rowOff = 0; rowOff < lastDim; rowOff += rowsPerBatch) {
                    const int64_t rows = Min(rowsPerBatch, lastDim - rowOff);
                    CopyInRowsPadded(offset + rowOff * innerCount, rows, validElements, rowStride, innerCount);
                    LocalTensor<T> input = inQueue_.DeQue<T>();
                    if (validElements <= V_LENGTH) {
                        AccumulateRowsSmallInnerKahan(input, sum, sumOffset, corrOffset, rows, validElements,
                                                      rowStride);
                        PipeBarrier<PIPE_V>();
                    } else {
                        // 宽 tile：64-lane 列块轮转寄存器驻留（每列行序不变，结果与逐行路径逐位相同）
                        const int64_t colChunks = (validElements + V_LENGTH - 1) / V_LENGTH;
                        for (int64_t cc = 0; cc < colChunks; ++cc) {
                            const int64_t chunkValid = Min(V_LENGTH, validElements - cc * V_LENGTH);
                            AccumulateRowsSmallInnerKahan(input, sum, sumOffset + cc * V_LENGTH,
                                                          corrOffset + cc * V_LENGTH, rows, chunkValid, rowStride,
                                                          cc * V_LENGTH);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                    inQueue_.FreeTensor(input);
                }
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
    int64_t currentReduceOffset = 0;
    for (int64_t r = 0; r < segmentEnd; ++r) {
        const int64_t reduceOffset = tiling_->useReduceOffsetTable != 0 ? tiling_->reduceSegmentOffsets[r] :
                                                                          currentReduceOffset;
        if (r >= segBegin) {
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

// 合并 chunks 份 partial → hi/lo 均值（向量 Neumaier 折叠 + 向量 Finalize），落 workspace
__aicore__ inline void MergeIrregularMeans(int64_t task, int64_t taskCount, int64_t chunks, int64_t blockStride,
                                           int64_t validKeepElements, int64_t sumStride, int64_t corrBase,
                                           int64_t validElements)
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
    const int64_t segments = tiling_->reduceSegmentCount;
    const int64_t segPerChunk = (segments + chunks - 1) / chunks;
    const int64_t totalTasks = taskCount * chunks;
    const int64_t nCores = static_cast<int64_t>(cores);

    // ---- 阶段 1：partial 归约（task = (irregularTask, chunk)）----
    for (int64_t t = static_cast<int64_t>(core); t < totalTasks; t += nCores) {
        const int64_t task = t / chunks;
        const int64_t chunk = t - task * chunks;
        const int64_t segBegin = chunk * segPerChunk;
        const int64_t segEnd = Min(segments, segBegin + segPerChunk);
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
        MergeIrregularMeans(task, taskCount, chunks, blockStride, validKeepElements, sumStride, corrBase,
                            validElements);
    }
    SyncAll();

    // ---- 阶段 3：输出（task = (irregularTask, chunk)，均值块按 task 缓存）----
    LocalTensor<float> meanTensor;
    int64_t cachedTask = -1;
    for (int64_t t = static_cast<int64_t>(core); t < totalTasks; t += nCores) {
        const int64_t task = t / chunks;
        const int64_t chunk = t - task * chunks;
        const int64_t segBegin = chunk * segPerChunk;
        const int64_t segEnd = Min(segments, segBegin + segPerChunk);
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
                                                           int64_t validKeepElements, LargeCompensatedSum* sums)
{
    int64_t reduceCoords[8] = {};
    int64_t currentReduceOffset = 0;
    for (int64_t reduceOuter = 0; reduceOuter < tiling_->reduceSegmentCount; ++reduceOuter) {
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
            for (int64_t segmentOffset = 0; segmentOffset < tiling_->irregularReduceSuffixElements;
                 segmentOffset += kLargeChunkElements) {
                const int64_t validElements = Min(kLargeChunkElements,
                                                  tiling_->irregularReduceSuffixElements - segmentOffset);
                CopyInContiguous(segmentBase + segmentOffset, validElements);
                LocalTensor<T> input = inQueue_.DeQue<T>();
                LocalTensor<T> output = outQueue_.AllocTensor<T>();
                CentralizeSegmentHiLo(input, output, validElements, means[keepLocal].hi, means[keepLocal].lo);
                PipeBarrier<PIPE_V>();
                outQueue_.EnQue(output);
                inQueue_.FreeTensor(input);
                CopyOutContiguous(segmentBase + segmentOffset, validElements);
            }
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
    AccumulateIrregularReduceSuffixTile(keepBase, keepStride, validKeepElements, sums);
    FinalizeIrregularReduceSuffixSums(sums, means, validKeepElements);
    WriteIrregularReduceSuffixTile(keepBase, keepStride, validKeepElements, means);
}

__aicore__ inline void ProcessIrregularReduceSuffix(uint32_t core, uint32_t cores)
{
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
