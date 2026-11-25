/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#pragma once
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
namespace ChunkLossGradArch22Opt {
using namespace AscendC;
constexpr uint16_t LOGITS_READY = 8;
constexpr uint16_t G_READY = 9;
constexpr uint16_t PARTIAL_READY = 10;
constexpr uint16_t SLOT_RELEASED = 11;
constexpr uint16_t HIDDEN_READY = 12;
constexpr uint16_t WEIGHT_READY = 13;
constexpr uint32_t VOCAB_TILE = 128;
constexpr uint32_t HIDDEN_ALIGNMENT = 128;
constexpr uint32_t MAX_LANE_ROWS = 256;
constexpr uint32_t BATCH_ROWS = 16;
constexpr uint32_t BATCH_COUNT = MAX_LANE_ROWS / BATCH_ROWS;
constexpr uint32_t STATE_BYTES = MAX_LANE_ROWS * (sizeof(float) * 2 + sizeof(int64_t));
constexpr uint32_t SCALAR_BYTES = 64;
constexpr uint32_t BROADCAST_FLOATS = MAX_LANE_ROWS * 8;
// The one-window path needs no accumulator queue. Its budget can hold both
// fully broadcast coefficient arrays for the complete invocation.
constexpr uint32_t VECTOR_ELEMENTS = BATCH_ROWS * VOCAB_TILE;
constexpr uint32_t OPT_UB_BYTES = 2 * VECTOR_ELEMENTS * (sizeof(float) + sizeof(half)) + STATE_BYTES + SCALAR_BYTES +
                                  BROADCAST_FLOATS * sizeof(float) * 2;
static_assert(OPT_UB_BYTES <= 4096 * (sizeof(float) * 2 + sizeof(half)) + 8192,
              "Vector buffers must fit the UB bound checked by host tiling");
// Pure-AIC MDL backend; TPipe initializes its synchronization state in the constructor.
constexpr auto MM_CONFIG = GetMDLConfig(false, false, 0, false, false, false, false, true, true, false, false, false);
template <class InputType>
using LogitsMatmul = matmul::MatmulImpl<matmul::MatmulType<TPosition::GM, CubeFormat::ND, InputType>,
                                        matmul::MatmulType<TPosition::GM, CubeFormat::ND, InputType, true>,
                                        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
                                        matmul::MatmulType<TPosition::GM, CubeFormat::ND, InputType>, MM_CONFIG>;
template <class InputType>
using WeightMatmul = matmul::MatmulImpl<matmul::MatmulType<TPosition::GM, CubeFormat::ND, InputType, true>,
                                        matmul::MatmulType<TPosition::GM, CubeFormat::ND, InputType>,
                                        matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
                                        matmul::MatmulType<TPosition::GM, CubeFormat::ND, InputType>, MM_CONFIG>;

template <class InputType, class Tiling>
class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR hidden, GM_ADDR weight, GM_ADDR labels, GM_ADDR maximum, GM_ADDR sum,
                                GM_ADDR grad, GM_ADDR output, GM_ADDR workspace, const Tiling& tiling)
    {
        tilingData = tiling;
        group = GetBlockIdx();
        lane = 0;
        if ASCEND_IS_AIV {
            group = GetBlockIdx() / 2;
            lane = GetBlockIdx() % 2;
        }
        hiddenGm.SetGlobalBuffer((__gm__ InputType*)hidden);
        weightGm.SetGlobalBuffer((__gm__ InputType*)weight);
        labelsGm.SetGlobalBuffer((__gm__ int64_t*)labels);
        softmaxMaxGm.SetGlobalBuffer((__gm__ float*)maximum);
        softmaxSumGm.SetGlobalBuffer((__gm__ float*)sum);
        upstream.SetGlobalBuffer((__gm__ float*)grad);
        gradWeightGm.SetGlobalBuffer((__gm__ InputType*)output);
        auto base = GetUserWorkspace(workspace) + uint64_t(group) * tilingData.coreWorkspaceBytes;
        logits.SetGlobalBuffer((__gm__ float*)base);
        logitsGradGm.SetGlobalBuffer((__gm__ InputType*)(base + tilingData.logitsBytes));
        partial.SetGlobalBuffer((__gm__ float*)(base + tilingData.logitsBytes + tilingData.logitsGradBytes));
        paddedTokens = (tilingData.numTokens + 127) / 128 * 128;
        hiddenPitch = (tilingData.hiddenDim + HIDDEN_ALIGNMENT - 1) / HIDDEN_ALIGNMENT * HIDDEN_ALIGNMENT;
        if (paddedTokens != tilingData.numTokens || hiddenPitch != tilingData.hiddenDim) {
            const uint64_t hiddenOffset = tilingData.logitsBytes + tilingData.logitsGradBytes + tilingData.partialBytes;
            hiddenGm.SetGlobalBuffer((__gm__ InputType*)(base + hiddenOffset));
            sourceHiddenGm.SetGlobalBuffer((__gm__ InputType*)hidden);
            sourceHiddenPacked.SetGlobalBuffer((__gm__ float*)hidden);
            paddedHiddenPacked.SetGlobalBuffer((__gm__ float*)(base + hiddenOffset));
        }
        if (hiddenPitch != tilingData.hiddenDim || tilingData.vocabSize % VOCAB_TILE != 0) {
            const uint64_t weightBytes = uint64_t(VOCAB_TILE) * hiddenPitch * sizeof(InputType);
            packedWeightGm.SetGlobalBuffer((__gm__ InputType*)(base + tilingData.coreWorkspaceBytes - weightBytes));
        }
        if ASCEND_IS_AIC {
            logitsMatmul.Init(&tilingData.logitsTiling, &pipe);
            gradWeightMatmul.Init(&tilingData.weightTiling, &pipe);
        }
        if ASCEND_IS_AIV {
            pipe.InitBuffer(inputBuf, 2 * VECTOR_ELEMENTS * sizeof(float));
            pipe.InitBuffer(outputBuf, 2 * VECTOR_ELEMENTS * sizeof(InputType));
            pipe.InitBuffer(stateBuf, STATE_BYTES + SCALAR_BYTES);
            pipe.InitBuffer(broadcastBuf, BROADCAST_FLOATS * sizeof(float) * 2);
            InitEvents();
            CacheVectorState();
        }
    }
    __aicore__ inline void Process()
    {
        if (paddedTokens != tilingData.numTokens || hiddenPitch != tilingData.hiddenDim) {
            if ASCEND_IS_AIV {
                if (hiddenPitch == tilingData.hiddenDim) {
                    PadHidden();
                } else {
                    PackRows(sourceHiddenGm, hiddenGm, laneBegin, laneBegin, validRows, laneRows);
                }
                CrossCoreSetFlag<2, PIPE_MTE3>(HIDDEN_READY);
            }
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag(HIDDEN_READY);
            }
        }
        for (uint64_t vocabOffset = uint64_t(group) * tilingData.vocabTile; vocabOffset < tilingData.vocabSize;
             vocabOffset += uint64_t(tilingData.coreNum) * tilingData.vocabTile) {
            uint32_t vocabCount = tilingData.vocabSize - vocabOffset < tilingData.vocabTile ?
                                      tilingData.vocabSize - vocabOffset :
                                      tilingData.vocabTile;
            const uint32_t tokenCount = paddedTokens;
            const bool packWeight = hiddenPitch != tilingData.hiddenDim || vocabCount != VOCAB_TILE;
            if (packWeight) {
                if ASCEND_IS_AIV {
                    const uint32_t weightBegin = lane * (VOCAB_TILE / 2);
                    const uint32_t remaining = vocabCount > weightBegin ? vocabCount - weightBegin : 0;
                    const uint32_t weightRows = remaining < VOCAB_TILE / 2 ? remaining : VOCAB_TILE / 2;
                    PackRows(weightGm, packedWeightGm, vocabOffset + weightBegin, weightBegin, weightRows,
                             VOCAB_TILE / 2);
                    CrossCoreSetFlag<2, PIPE_MTE3>(WEIGHT_READY);
                }
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag(WEIGHT_READY);
                }
            }
            if ASCEND_IS_AIC {
                ComputeLogits(vocabOffset, tokenCount, packWeight);
                CrossCoreSetFlag<2, PIPE_FIX>(LOGITS_READY);
                CrossCoreWaitFlag(G_READY);
                ComputeWeight(tokenCount);
                CrossCoreSetFlag<2, PIPE_FIX>(PARTIAL_READY);
                CrossCoreWaitFlag(SLOT_RELEASED);
            }
            if ASCEND_IS_AIV {
                CrossCoreWaitFlag(LOGITS_READY);
                MakeGBatched(vocabOffset, vocabCount);
                CrossCoreSetFlag<2, PIPE_MTE3>(G_READY);
                CrossCoreWaitFlag(PARTIAL_READY);
                if (hiddenPitch == tilingData.hiddenDim) {
                    ReduceWeight(vocabOffset, vocabCount);
                } else {
                    ReduceWeightCropped(vocabOffset, vocabCount);
                }
                CrossCoreSetFlag<2, PIPE_MTE3>(SLOT_RELEASED);
            }
        }
        if ASCEND_IS_AIV {
            DrainEvents();
        }
        if ASCEND_IS_AIC {
            // End releases queue events; each object stays live across all token/vocab windows.
            logitsMatmul.End();
            gradWeightMatmul.End();
        }
    }

private:
    __aicore__ inline void PackRows(GlobalTensor<InputType>& source, GlobalTensor<InputType>& destination,
                                    uint64_t sourceRow, uint32_t destinationRow, uint32_t validCount, uint32_t rowCount)
    {
        // Borrow both input slots while Cube waits for the packed operand.
        WaitFlag<HardEvent::V_MTE2>(inputFree[0]);
        WaitFlag<HardEvent::V_MTE2>(inputFree[1]);
        auto values = inputBuf.Get<float>();
        auto low = inputBuf.Get<InputType>();
        const uint32_t rowsPerCopy = (2 * VECTOR_ELEMENTS * sizeof(float)) / (hiddenPitch * sizeof(InputType));
        const uint32_t nativePitch = (tilingData.hiddenDim + 15) / 16 * 16;
        const auto ready = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_MTE3>());
        const auto free = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
        for (uint32_t offset = 0; offset < rowCount; offset += rowsPerCopy) {
            const uint32_t rows = rowCount - offset < rowsPerCopy ? rowCount - offset : rowsPerCopy;
            const uint32_t remaining = validCount > offset ? validCount - offset : 0;
            const uint32_t valid = remaining < rows ? remaining : rows;
            // Clear inter-row pitch gaps and fully padded rows before the DMA.
            Duplicate(values, 0.0f, rows * hiddenPitch / 2);
            SetFlag<HardEvent::V_MTE2>(inputFree[0]);
            WaitFlag<HardEvent::V_MTE2>(inputFree[0]);
            if (valid != 0) {
                DataCopyExtParams read{uint16_t(valid), uint32_t(tilingData.hiddenDim * sizeof(InputType)), 0,
                                       (hiddenPitch - nativePitch) / 16, 0};
                DataCopyPad(low, source[(sourceRow + offset) * tilingData.hiddenDim], read,
                            DataCopyPadExtParams<InputType>{true, 0, uint8_t(nativePitch - tilingData.hiddenDim),
                                                            InputType(0)});
            }
            SetFlag<HardEvent::MTE2_MTE3>(ready);
            WaitFlag<HardEvent::MTE2_MTE3>(ready);
            DataCopyExtParams write{1, uint32_t(rows * hiddenPitch * sizeof(InputType)), 0, 0, 0};
            DataCopyPad(destination[uint64_t(destinationRow + offset) * hiddenPitch], low, write);
            SetFlag<HardEvent::MTE3_V>(free);
            WaitFlag<HardEvent::MTE3_V>(free);
        }
        pipe.ReleaseEventID<HardEvent::MTE2_MTE3>(ready);
        pipe.ReleaseEventID<HardEvent::MTE3_V>(free);
        SetFlag<HardEvent::V_MTE2>(inputFree[0]);
        SetFlag<HardEvent::V_MTE2>(inputFree[1]);
    }

    __aicore__ inline void PadHidden()
    {
        // Copy raw pairs of FP16/BF16 values, preserving every input bit.
        // Each AIV owns disjoint complete rows in this group's private workspace.
        const uint64_t rowPairs = tilingData.hiddenDim / 2;
        const uint64_t elements = uint64_t(laneRows) * rowPairs;
        const uint64_t validElements = uint64_t(validRows) * rowPairs;
        const uint64_t begin = uint64_t(laneBegin) * rowPairs;
        const auto ready = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_MTE3>());
        const auto free = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
        for (uint64_t offset = 0; offset < elements; offset += VECTOR_ELEMENTS) {
            const uint32_t count = elements - offset < VECTOR_ELEMENTS ? elements - offset : VECTOR_ELEMENTS;
            const uint64_t remaining = validElements > offset ? validElements - offset : 0;
            const uint32_t valid = remaining < count ? remaining : count;
            WaitFlag<HardEvent::V_MTE2>(inputFree[0]);
            auto values = Input(0);
            if (valid < count) {
                Duplicate(values, 0.0f, count);
                SetFlag<HardEvent::V_MTE2>(inputFree[0]);
                WaitFlag<HardEvent::V_MTE2>(inputFree[0]);
            }
            if (valid != 0) {
                DataCopyExtParams read{1, uint32_t(valid * sizeof(float)), 0, 0, 0};
                DataCopyPad(values, sourceHiddenPacked[begin + offset], read,
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
            }
            SetFlag<HardEvent::MTE2_MTE3>(ready);
            WaitFlag<HardEvent::MTE2_MTE3>(ready);
            DataCopyExtParams write{1, uint32_t(count * sizeof(float)), 0, 0, 0};
            DataCopyPad(paddedHiddenPacked[begin + offset], values, write);
            // UB is reused by both Vector zero-fill and MTE2. Order MTE3 -> V -> MTE2.
            SetFlag<HardEvent::MTE3_V>(free);
            WaitFlag<HardEvent::MTE3_V>(free);
            SetFlag<HardEvent::V_MTE2>(inputFree[0]);
        }
        pipe.ReleaseEventID<HardEvent::MTE2_MTE3>(ready);
        pipe.ReleaseEventID<HardEvent::MTE3_V>(free);
    }

    __aicore__ inline void InitEvents()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            inputReady[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
            inputFree[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
            outputReady[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
            outputFree[slot] = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
            SetFlag<HardEvent::V_MTE2>(inputFree[slot]);
            SetFlag<HardEvent::MTE3_V>(outputFree[slot]);
        }
    }

    __aicore__ inline void DrainEvents()
    {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            WaitFlag<HardEvent::V_MTE2>(inputFree[slot]);
            WaitFlag<HardEvent::MTE3_V>(outputFree[slot]);
            pipe.ReleaseEventID<HardEvent::MTE2_V>(inputReady[slot]);
            pipe.ReleaseEventID<HardEvent::V_MTE2>(inputFree[slot]);
            pipe.ReleaseEventID<HardEvent::V_MTE3>(outputReady[slot]);
            pipe.ReleaseEventID<HardEvent::MTE3_V>(outputFree[slot]);
        }
    }

    __aicore__ inline LocalTensor<float> Input(uint32_t slot)
    {
        return inputBuf.GetWithOffset<float>(VECTOR_ELEMENTS, slot * VECTOR_ELEMENTS * sizeof(float));
    }

    __aicore__ inline void Prefetch(uint32_t slot, GlobalTensor<float> source, uint32_t count)
    {
        WaitFlag<HardEvent::V_MTE2>(inputFree[slot]);
        auto value = Input(slot);
        if (count < VECTOR_ELEMENTS) {
            Duplicate(value, 0.0f, VECTOR_ELEMENTS);
            SetFlag<HardEvent::V_MTE2>(inputFree[slot]);
            WaitFlag<HardEvent::V_MTE2>(inputFree[slot]);
        }
        if (count != 0) {
            DataCopyExtParams copy{1, uint32_t(count * sizeof(float)), 0, 0, 0};
            DataCopyPad(value, source, copy, DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(inputReady[slot]);
    }

    __aicore__ inline void Store(uint32_t slot, LocalTensor<float>& value, GlobalTensor<InputType> destination,
                                 uint32_t count)
    {
        WaitFlag<HardEvent::MTE3_V>(outputFree[slot]);
        auto low = outputBuf.GetWithOffset<InputType>(VECTOR_ELEMENTS, slot * VECTOR_ELEMENTS * sizeof(InputType));
        Cast(low, value, RoundMode::CAST_RINT, count);
        SetFlag<HardEvent::V_MTE2>(inputFree[slot]);
        SetFlag<HardEvent::V_MTE3>(outputReady[slot]);
        WaitFlag<HardEvent::V_MTE3>(outputReady[slot]);
        DataCopyExtParams copy{1, uint32_t(count * sizeof(InputType)), 0, 0, 0};
        DataCopyPad(destination, low, copy);
        SetFlag<HardEvent::MTE3_V>(outputFree[slot]);
    }

    __aicore__ inline void ComputeLogits(uint64_t vocabOffset, uint32_t tokenCount, bool packed)
    {
        logitsMatmul.SetOrgShape(tokenCount, VOCAB_TILE, hiddenPitch);
        logitsMatmul.SetSingleShape(tokenCount, VOCAB_TILE, hiddenPitch);
        logitsMatmul.SetTensorA(hiddenGm, false);
        if (packed) {
            logitsMatmul.SetTensorB(packedWeightGm, true);
        } else {
            logitsMatmul.SetTensorB(weightGm[vocabOffset * tilingData.hiddenDim], true);
        }
        logitsMatmul.IterateAll(logits);
        // Drain MDL's outstanding L1 transfers before the next GEMM reuses L1.
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void ComputeWeight(uint32_t tokenCount)
    {
        gradWeightMatmul.SetOrgShape(VOCAB_TILE, hiddenPitch, tokenCount);
        gradWeightMatmul.SetSingleShape(VOCAB_TILE, hiddenPitch, tokenCount);
        gradWeightMatmul.SetTensorA(logitsGradGm, true);
        gradWeightMatmul.SetTensorB(hiddenGm, false);
        gradWeightMatmul.IterateAll(partial);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void CacheVectorState()
    {
        // Give each lane complete 16-row batches; padded rows produce zero G.
        const uint32_t numTokens = static_cast<uint32_t>(tilingData.numTokens);
        laneRows = paddedTokens / 2;
        laneBegin = lane * laneRows;
        validRows = numTokens > laneBegin ? (numTokens - laneBegin < laneRows ? numTokens - laneBegin : laneRows) : 0;
        auto maximum = stateBuf.GetWithOffset<float>(MAX_LANE_ROWS, 0);
        auto inverseSum = stateBuf.GetWithOffset<float>(MAX_LANE_ROWS, MAX_LANE_ROWS * sizeof(float));
        auto labels = stateBuf.GetWithOffset<int64_t>(MAX_LANE_ROWS, MAX_LANE_ROWS * sizeof(float) * 2);
        auto grad = stateBuf.GetWithOffset<float>(8, STATE_BYTES + 32);
        DataCopyExtParams floats{1, uint32_t(validRows * sizeof(float)), 0, 0, 0};
        DataCopyExtParams integers{1, uint32_t(validRows * sizeof(int64_t)), 0, 0, 0};
        DataCopyExtParams scalar{1, sizeof(float), 0, 0, 0};
        DataCopyPad(maximum, softmaxMaxGm[laneBegin], floats, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(inverseSum, softmaxSumGm[laneBegin], floats, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(labels, labelsGm[laneBegin], integers, DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        DataCopyPad(grad, upstream, scalar, DataCopyPadExtParams<float>{false, 0, 0, 0});
        auto ms = pipe.FetchEventID<HardEvent::MTE2_S>();
        SetFlag<HardEvent::MTE2_S>(ms);
        WaitFlag<HardEvent::MTE2_S>(ms);
        normalizationScale = tilingData.alpha;
        outputScale = grad.GetValue(0);
        for (uint32_t batch = 0; batch < BATCH_COUNT; ++batch) {
            targetHeads[batch] = -1;
            ignoredRows[batch] = 0;
        }
        uint32_t targets = 0;
        for (uint32_t row = 0; row < laneRows; ++row) {
            if (row >= validRows) {
                maximum.SetValue(row, 0.0f);
                inverseSum.SetValue(row, 1.0f);
                ignoredRows[row / BATCH_ROWS] |= 1U << (row % BATCH_ROWS);
                continue;
            }
            // Setup only: preserve the original scalar FP32 reciprocal.
            inverseSum.SetValue(row, 1.0f / inverseSum.GetValue(row));
            const int64_t label = labels.GetValue(row);
            if (label == tilingData.ignoreIndex) {
                ignoredRows[row / BATCH_ROWS] |= 1U << (row % BATCH_ROWS);
            } else if (label >= 0 && label < static_cast<int64_t>(tilingData.vocabSize)) {
                const uint32_t tile = static_cast<uint32_t>(label) / VOCAB_TILE;
                if (tile % tilingData.coreNum == group) {
                    const uint32_t batch = row / BATCH_ROWS;
                    targetTiles[targets] = tile;
                    targetOffsets[targets] = (row % BATCH_ROWS) * VOCAB_TILE +
                                             static_cast<uint32_t>(label) % VOCAB_TILE;
                    targetNext[targets] = targetHeads[batch];
                    targetHeads[batch] = static_cast<int32_t>(targets++);
                }
            }
        }
        auto sv = pipe.FetchEventID<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(sv);
        WaitFlag<HardEvent::S_V>(sv);
        auto brcbMax = broadcastBuf.GetWithOffset<float>(BROADCAST_FLOATS, 0);
        auto brcbInverse = broadcastBuf.GetWithOffset<float>(BROADCAST_FLOATS, BROADCAST_FLOATS * sizeof(float));
        Brcb(brcbMax, maximum, uint8_t(laneRows / 8), {1, 8});
        Brcb(brcbInverse, inverseSum, uint8_t(laneRows / 8), {1, 8});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CorrectTargets(LocalTensor<float>& values, uint32_t batch, uint32_t tile)
    {
        int32_t target = targetHeads[batch];
        while (target >= 0 && targetTiles[target] != tile) {
            target = targetNext[target];
        }
        if (target < 0) {
            return;
        }
        auto vs = pipe.FetchEventID<HardEvent::V_S>();
        SetFlag<HardEvent::V_S>(vs);
        WaitFlag<HardEvent::V_S>(vs);
        while (target >= 0) {
            if (targetTiles[target] == tile) {
                const uint32_t offset = targetOffsets[target];
                values.SetValue(offset, values.GetValue(offset) - 1.0f);
            }
            target = targetNext[target];
        }
        auto sv = pipe.FetchEventID<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(sv);
        WaitFlag<HardEvent::S_V>(sv);
    }

    __aicore__ inline void ClearVocabPadding(LocalTensor<float>& values, uint32_t vocabCount)
    {
        // Bit masks keep every Vector destination aligned to a complete repeat.
        for (uint32_t block = vocabCount / 64; block < VOCAB_TILE / 64; ++block) {
            const uint32_t valid = vocabCount > block * 64 ? vocabCount - block * 64 : 0;
            uint64_t mask[2] = {~uint64_t(0) << valid, 0};
            Duplicate(values[block * 64], 0.0f, mask, uint8_t(BATCH_ROWS), 1, VOCAB_TILE / 8);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MakeGBatched(uint64_t vocabOffset, uint32_t vocabCount)
    {
        auto brcbMax = broadcastBuf.GetWithOffset<float>(BROADCAST_FLOATS, 0);
        auto brcbInverse = broadcastBuf.GetWithOffset<float>(BROADCAST_FLOATS, BROADCAST_FLOATS * sizeof(float));
        // FP32 block=8 elements; each repeat advances one 128-column row,
        // and src1 advances one block containing eight copies of its scalar.
        const BinaryRepeatParams broadcast{1, 1, 0, VOCAB_TILE / 8, VOCAB_TILE / 8, 1};
        const uint32_t firstRows = validRows < BATCH_ROWS ? validRows : BATCH_ROWS;
        Prefetch(0, logits[uint64_t(laneBegin) * VOCAB_TILE], firstRows * VOCAB_TILE);
        for (uint32_t first = 0; first < laneRows; first += BATCH_ROWS) {
            const uint32_t slot = (first / BATCH_ROWS) & 1;
            const uint32_t count = VECTOR_ELEMENTS;
            WaitFlag<HardEvent::MTE2_V>(inputReady[slot]);
            if (first + BATCH_ROWS < laneRows) {
                const uint32_t remaining = validRows > first + BATCH_ROWS ? validRows - first - BATCH_ROWS : 0;
                const uint32_t nextRows = remaining < BATCH_ROWS ? remaining : BATCH_ROWS;
                Prefetch(slot ^ 1, logits[uint64_t(laneBegin + first + BATCH_ROWS) * VOCAB_TILE],
                         nextRows * VOCAB_TILE);
            }
            auto values = Input(slot);
            Sub(values, values, brcbMax[first * 8], uint64_t(64), uint8_t(BATCH_ROWS), broadcast);
            Sub(values[64], values[64], brcbMax[first * 8], uint64_t(64), uint8_t(BATCH_ROWS), broadcast);
            PipeBarrier<PIPE_V>();
            if (vocabCount != VOCAB_TILE) {
                // Avoid exponentiating padding against an unrelated real-row max.
                ClearVocabPadding(values, vocabCount);
            }
            Exp(values, values, count);
            PipeBarrier<PIPE_V>();
            Mul(values, values, brcbInverse[first * 8], uint64_t(64), uint8_t(BATCH_ROWS), broadcast);
            Mul(values[64], values[64], brcbInverse[first * 8], uint64_t(64), uint8_t(BATCH_ROWS), broadcast);
            CorrectTargets(values, first / BATCH_ROWS, static_cast<uint32_t>(vocabOffset / VOCAB_TILE));
            PipeBarrier<PIPE_V>();
            Muls(values, values, normalizationScale, count);
            PipeBarrier<PIPE_V>();
            if (vocabCount != VOCAB_TILE) {
                ClearVocabPadding(values, vocabCount);
            }
            const uint32_t ignored = ignoredRows[first / BATCH_ROWS];
            if (ignored != 0) {
                for (uint32_t row = 0; row < BATCH_ROWS; ++row) {
                    if (ignored & (1U << row)) {
                        Duplicate(values[row * VOCAB_TILE], 0.0f, VOCAB_TILE);
                    }
                }
                PipeBarrier<PIPE_V>();
            }
            Store(slot, values, logitsGradGm[uint64_t(laneBegin + first) * VOCAB_TILE], count);
        }
    }

    __aicore__ inline void ReduceWeight(uint64_t vocabOffset, uint32_t vocabCount)
    {
        const uint32_t rowsPerLane = (vocabCount + 1) / 2;
        const uint32_t firstRow = lane * rowsPerLane;
        const uint32_t endRow = vocabCount < firstRow + rowsPerLane ? vocabCount : firstRow + rowsPerLane;
        const uint64_t begin = uint64_t(firstRow) * tilingData.hiddenDim;
        const uint64_t end = uint64_t(endRow) * tilingData.hiddenDim;
        if (begin >= end) {
            return;
        }
        const uint32_t firstCount = end - begin < VECTOR_ELEMENTS ? end - begin : VECTOR_ELEMENTS;
        Prefetch(0, partial[begin], firstCount);
        uint32_t iteration = 0;
        for (uint64_t offset = begin; offset < end; offset += VECTOR_ELEMENTS, ++iteration) {
            const uint32_t slot = iteration & 1;
            const uint32_t count = end - offset < VECTOR_ELEMENTS ? end - offset : VECTOR_ELEMENTS;
            WaitFlag<HardEvent::MTE2_V>(inputReady[slot]);
            if (offset + VECTOR_ELEMENTS < end) {
                const uint64_t next = offset + VECTOR_ELEMENTS;
                const uint32_t nextCount = end - next < VECTOR_ELEMENTS ? end - next : VECTOR_ELEMENTS;
                Prefetch(slot ^ 1, partial[next], nextCount);
            }
            auto value = Input(slot);
            Muls(value, value, outputScale, count);
            PipeBarrier<PIPE_V>();
            Store(slot, value, gradWeightGm[vocabOffset * tilingData.hiddenDim + offset], count);
        }
    }

    __aicore__ inline void PrefetchCropped(uint32_t slot, uint32_t row, uint32_t column, uint32_t rows,
                                           uint32_t columns, uint32_t pitch)
    {
        WaitFlag<HardEvent::V_MTE2>(inputFree[slot]);
        auto value = Input(slot);
        if (columns != pitch) {
            // GM float rows align to eight elements, while the half output
            // needs sixteen. Initialize the extra skipped UB block as well.
            Duplicate(value, 0.0f, rows * pitch);
            SetFlag<HardEvent::V_MTE2>(inputFree[slot]);
            WaitFlag<HardEvent::V_MTE2>(inputFree[slot]);
        }
        const uint32_t nativePitch = (columns + 7) / 8 * 8;
        DataCopyExtParams read{uint16_t(rows), uint32_t(columns * sizeof(float)),
                               uint32_t((hiddenPitch - columns) * sizeof(float)), (pitch - nativePitch) / 8, 0};
        DataCopyPad(value, partial[uint64_t(row) * hiddenPitch + column], read,
                    DataCopyPadExtParams<float>{true, 0, uint8_t(nativePitch - columns), 0.0f});
        SetFlag<HardEvent::MTE2_V>(inputReady[slot]);
    }

    __aicore__ inline void StoreCropped(uint32_t slot, LocalTensor<float>& value, uint64_t row, uint32_t column,
                                        uint32_t rows, uint32_t columns, uint32_t pitch)
    {
        WaitFlag<HardEvent::MTE3_V>(outputFree[slot]);
        auto low = outputBuf.GetWithOffset<InputType>(VECTOR_ELEMENTS, slot * VECTOR_ELEMENTS * sizeof(InputType));
        Cast(low, value, RoundMode::CAST_RINT, rows * pitch);
        SetFlag<HardEvent::V_MTE2>(inputFree[slot]);
        SetFlag<HardEvent::V_MTE3>(outputReady[slot]);
        WaitFlag<HardEvent::V_MTE3>(outputReady[slot]);
        DataCopyExtParams write{uint16_t(rows), uint32_t(columns * sizeof(InputType)), 0,
                                uint32_t((tilingData.hiddenDim - columns) * sizeof(InputType)), 0};
        DataCopyPad(gradWeightGm[row * tilingData.hiddenDim + column], low, write);
        SetFlag<HardEvent::MTE3_V>(outputFree[slot]);
    }

    __aicore__ inline void ReduceWeightCropped(uint64_t vocabOffset, uint32_t vocabCount)
    {
        // An aligned row boundary gives the two AIVs disjoint GM data blocks,
        // even when the logical H is odd. Empty lanes keep their free events.
        const uint32_t split = ((vocabCount + 1) / 2 + 15) / 16 * 16;
        const uint32_t begin = lane == 0 ? 0 : split;
        const uint32_t end = lane == 0 && vocabCount > split ? split : vocabCount;
        if (begin >= end) {
            return;
        }
        constexpr uint32_t COLUMNS = VECTOR_ELEMENTS / BATCH_ROWS;
        const uint32_t firstRows = end - begin < BATCH_ROWS ? end - begin : BATCH_ROWS;
        const uint32_t firstColumns = tilingData.hiddenDim < COLUMNS ? tilingData.hiddenDim : COLUMNS;
        PrefetchCropped(0, begin, 0, firstRows, firstColumns, (firstColumns + 15) / 16 * 16);
        uint32_t iteration = 0;
        for (uint32_t row = begin; row < end; row += BATCH_ROWS) {
            const uint32_t rows = end - row < BATCH_ROWS ? end - row : BATCH_ROWS;
            for (uint32_t column = 0; column < tilingData.hiddenDim; column += COLUMNS, ++iteration) {
                const uint32_t slot = iteration & 1;
                const uint32_t columns = tilingData.hiddenDim - column < COLUMNS ? tilingData.hiddenDim - column :
                                                                                   COLUMNS;
                const uint32_t pitch = (columns + 15) / 16 * 16;
                WaitFlag<HardEvent::MTE2_V>(inputReady[slot]);
                uint32_t nextRow = row;
                uint32_t nextColumn = column + COLUMNS;
                if (nextColumn >= tilingData.hiddenDim) {
                    nextRow += BATCH_ROWS;
                    nextColumn = 0;
                }
                if (nextRow < end) {
                    const uint32_t nextRows = end - nextRow < BATCH_ROWS ? end - nextRow : BATCH_ROWS;
                    const uint32_t nextColumns = tilingData.hiddenDim - nextColumn < COLUMNS ?
                                                     tilingData.hiddenDim - nextColumn :
                                                     COLUMNS;
                    PrefetchCropped(slot ^ 1, nextRow, nextColumn, nextRows, nextColumns, (nextColumns + 15) / 16 * 16);
                }
                auto value = Input(slot);
                Muls(value, value, outputScale, rows * pitch);
                PipeBarrier<PIPE_V>();
                StoreCropped(slot, value, vocabOffset + row, column, rows, columns, pitch);
            }
        }
    }

    Tiling tilingData;
    uint32_t group, lane;
    uint32_t laneBegin, laneRows, validRows, paddedTokens;
    uint32_t hiddenPitch;
    float normalizationScale, outputScale;
    int32_t targetHeads[BATCH_COUNT];
    uint32_t ignoredRows[BATCH_COUNT];
    int32_t targetNext[MAX_LANE_ROWS];
    uint32_t targetTiles[MAX_LANE_ROWS], targetOffsets[MAX_LANE_ROWS];
    TPipe pipe;
    LogitsMatmul<InputType> logitsMatmul;
    WeightMatmul<InputType> gradWeightMatmul;
    GlobalTensor<InputType> hiddenGm, weightGm, logitsGradGm, gradWeightGm;
    GlobalTensor<InputType> sourceHiddenGm, packedWeightGm;
    GlobalTensor<int64_t> labelsGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, upstream, logits, partial;
    GlobalTensor<float> sourceHiddenPacked, paddedHiddenPacked;
    event_t inputReady[2], inputFree[2], outputReady[2], outputFree[2];
    TBuf<TPosition::VECCALC> inputBuf, outputBuf;
    TBuf<TPosition::VECCALC> stateBuf, broadcastBuf;
};
} // namespace ChunkLossGradArch22Opt
