/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef APPLY_TOP_K_TOP_P_OPT_H_KERNEL
#define APPLY_TOP_K_TOP_P_OPT_H_KERNEL

#include "kernel_operator.h"

using namespace AscendC;
namespace ApplyTopKTopPOptOp {
constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
constexpr uint16_t BF16_NEG_INF = 0xFF80;
constexpr int32_t FLOAT32_NEG_INF = 0xFF800000;
constexpr int32_t FLOAT32_INF = 0x7F800000;

constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t RESERVED_UB = 1024;
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t SOFTMAX_UB_NUM = 2;
constexpr uint32_t CMP_ALIGN_BYTES = 256;
constexpr uint32_t SCATTER_PART_LENGTH = 1024;
constexpr uint32_t COPY_CHUNK_BYTES = 16 * 1024;

template <typename inputT, typename calT, typename outputT>
class ApplyTopKTopPOpt {
public:
    __aicore__ inline ApplyTopKTopPOpt(){};
    __aicore__ inline void InitTilingData(const ApplyTopKTopPWithSortedTilingData& __restrict tilingData,
                                          GM_ADDR sorted_value, GM_ADDR sorted_indices, GM_ADDR p, GM_ADDR k,
                                          GM_ADDR logits, GM_ADDR out, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void ProcessTopKTopPOpt();
    __aicore__ inline void ProcessTopKOpt();
    __aicore__ inline void ProcessTopPOpt();

private:
    __aicore__ inline void GetMaxValue(int64_t baseGmIdx);
    __aicore__ inline void ComputeSoftmaxSum(int64_t sortedBase, float kthVal, uint32_t kthStartIdx);
    __aicore__ inline void WriteSoftmaxToGm(int64_t sortedBase, int64_t sfGmBase, uint32_t kthStartIdx,
                                            uint32_t dataStart, uint32_t dataLen);
    __aicore__ inline void CumsumKoggleStone(int64_t sfGmBase, uint32_t dataLen);
    __aicore__ inline void FindFirstIndex(int64_t sfGmBase, uint32_t dataLen, float cumsumBase, uint32_t& firstIdx);
    __aicore__ inline void ProcessCumsumAndFind(int64_t sortedBase, int64_t batchGmBase, uint32_t kthStartIdx,
                                                uint32_t& firstIdx);
    __aicore__ inline void CompareSelectOnLogits(int64_t batchGmBase, inputT thresholdVal, bool useGE);
    __aicore__ inline void ScatterBoundary(int64_t batchGmBase, int64_t sortedBase, uint32_t firstIdx,
                                           inputT thresholdVal);
    __aicore__ inline void CopyOutLast(int64_t batchGmBase, int64_t sortedBase);
    __aicore__ inline void FillNegInfAndScatterTopK(int64_t batchGmBase, int64_t sortedBase, uint32_t kthStartIdx);
    __aicore__ inline uint32_t FindKthStartIdx(int64_t sortedBase, int32_t kValue);
    __aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y) { return y == 0 ? x : (x + y - 1) / y; }

    // 310P has no usable DataCopyPad (it writes zeros at runtime), so all copies go through plain
    // DataCopy, which requires both sides to be 32B aligned. The Kogge-Stone prefix sum walks the
    // GM buffer at offsets 1/2/4/... and at dataTail, so the address itself has to be inspected:
    // aligned runs use DataCopy, misaligned ones fall back to scalar access.
    template <typename T>
    __aicore__ inline uint32_t MisalignElems(const GlobalTensor<T>& t)
    {
        return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(t.GetPhyAddr()) % BLOCK_BYTES) / sizeof(T);
    }

    template <typename T>
    __aicore__ inline void SafeCopyIn(LocalTensor<T> dst, const GlobalTensor<T>& src, uint32_t count)
    {
#if __CCE_AICORE__ != 200
        // Other architectures have a working DataCopyPad, which handles arbitrary lengths and
        // offsets without the alignment workarounds below.
        DataCopyPad(dst, src, {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
#else
        constexpr uint32_t elemsPerBlock = BLOCK_BYTES / sizeof(T);
        constexpr uint32_t maxElems = COPY_CHUNK_BYTES / sizeof(T);
        uint32_t aligned = AscendC::AlignUp(count, elemsPerBlock);
        if (MisalignElems(src) == 0) {
            for (uint32_t off = 0; off < aligned; off += maxElems) {
                uint32_t len = (aligned - off) < maxElems ? (aligned - off) : maxElems;
                DataCopy(dst[off], src[off], len);
            }
            return;
        }
        PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < count; i++) {
            dst.SetValue(i, src.GetValue(i));
        }
        PipeBarrier<PIPE_ALL>();
#endif
    }

    // Partial 32B block write: read the whole block, overlay `len` elements, write it back. Both
    // DataCopy operands must be 32B aligned, so the read starts at the enclosing block boundary,
    // which is `back` elements before `dst`. Going through MTE2/MTE3 keeps every GM access
    // vectorised: scalar SetValue writes would land in the per-core data cache, and a
    // clean+invalidate of a line shared with another core would overwrite that core's output.
    template <typename T>
    __aicore__ inline void RmwBlock(GlobalTensor<T> dst, uint32_t back, LocalTensor<T> src, uint32_t srcOff,
                                    uint32_t dstOff, uint32_t len)
    {
        constexpr uint32_t elemsPerBlock = BLOCK_BYTES / sizeof(T);
        LocalTensor<T> stage = rmwStage.template ReinterpretCast<T>();
        GlobalTensor<T> block;
        uintptr_t blockAddr = reinterpret_cast<uintptr_t>(dst.GetPhyAddr()) - back * sizeof(T);
        block.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(blockAddr), elemsPerBlock);
        MTE3ToMTE2Sync();
        DataCopy(stage, block, elemsPerBlock);
        MTE2ToSSync();
        for (uint32_t i = 0; i < len; i++) {
            stage.SetValue(dstOff + i, src.GetValue(srcOff + i));
        }
        SToMTE3Sync();
        DataCopy(block, stage, elemsPerBlock);
        MTE3ToMTE2Sync();
    }

    // DataCopy requires both sides to be 32B aligned. The leading and trailing partial blocks are
    // written with a 32B read-modify-write; only whole blocks in between go through DataCopy.
    template <typename T>
    __aicore__ inline void SafeCopyOut(GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count)
    {
#if __CCE_AICORE__ != 200
        // Other architectures have a working DataCopyPad, which handles arbitrary lengths and
        // offsets without the alignment workarounds below.
        DataCopyPad(dst, src, {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0});
#else
        constexpr uint32_t elemsPerBlock = BLOCK_BYTES / sizeof(T);
        constexpr uint32_t maxElems = COPY_CHUNK_BYTES / sizeof(T);
        // MisalignElems returns the offset inside the current block, so the distance to the next
        // boundary is elemsPerBlock - head, not head itself.
        uint32_t head = MisalignElems(dst);
        if (head != 0) {
            // After the leading partial block the body would have to read from src[elemsPerBlock -
            // head], which is not a multiple of the block size, so the UB operand would be
            // misaligned. Tiling keeps vocabSize block aligned, so this path is not reached in
            // practice; write block by block to stay correct anyway.
            uint32_t done = 0;
            while (done < count) {
                uint32_t dstOff = (done == 0) ? head : 0;
                uint32_t room = elemsPerBlock - dstOff;
                uint32_t len = room < (count - done) ? room : (count - done);
                RmwBlock(dst[done], dstOff, src, done, dstOff, len);
                done += len;
            }
            return;
        }
        uint32_t body = count / elemsPerBlock * elemsPerBlock;
        for (uint32_t off = 0; off < body; off += maxElems) {
            uint32_t len = (body - off) < maxElems ? (body - off) : maxElems;
            DataCopy(dst[off], src[off], len);
        }
        if (body < count) {
            RmwBlock(dst[body], 0, src, body, 0, count - body);
        }
#endif
    }

    __aicore__ inline void SToMTE3Sync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(id);
        WaitFlag<HardEvent::S_MTE3>(id);
    }
    __aicore__ inline void VToMTE3Sync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(id);
        WaitFlag<HardEvent::V_MTE3>(id);
    }
    __aicore__ inline void MTE2ToVSync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(id);
        WaitFlag<HardEvent::MTE2_V>(id);
    }
    __aicore__ inline void VToMTE2Sync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(id);
        WaitFlag<HardEvent::V_MTE2>(id);
    }
    __aicore__ inline void MTE2ToSSync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(id);
        WaitFlag<HardEvent::MTE2_S>(id);
    }
    __aicore__ inline void MTE3ToSSync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(id);
        WaitFlag<HardEvent::MTE3_S>(id);
    }
    __aicore__ inline void MTE3ToMTE2Sync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(id);
        WaitFlag<HardEvent::MTE3_MTE2>(id);
    }
    __aicore__ inline void VToSSync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(id);
        WaitFlag<HardEvent::V_S>(id);
    }
    __aicore__ inline void SToVSync()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(id);
        WaitFlag<HardEvent::S_V>(id);
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> calBuf_;

    uint32_t batchSize_ = 0;
    uint32_t vocabSize_ = 0;
    uint32_t batchPerCore_ = 0;
    uint32_t tailBatch_ = 0;
    uint32_t blockNum_ = 0;
    uint32_t calUbSize_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t loopBatch_ = 0;
    uint32_t batchOffset_ = 0;

    uint32_t softmaxLength = 1;
    uint32_t lineSfLoopTimes = 1;
    uint32_t softmaxLengthTail = 1;
    uint32_t cmpSelTileLength = 0;
    uint32_t scatterLength = 1;

    GlobalTensor<inputT> mGmSortedValue_;
    GlobalTensor<int32_t> mGmSortedIndices_;
    GlobalTensor<inputT> mGmP_;
    GlobalTensor<int32_t> mGmK_;
    GlobalTensor<inputT> mGmLogits_;
    GlobalTensor<outputT> mGmOut_;
    GlobalTensor<float> softMaxGm;

    LocalTensor<uint8_t> totalUb;
    LocalTensor<float> sfLocalFp32;
    LocalTensor<inputT> sfLocalInput;
    LocalTensor<float> sfResLocal;
    LocalTensor<float> reduceLocal;

    LocalTensor<inputT> cmpLogitsLocal;
    LocalTensor<inputT> cmpNegInfLocal;
    LocalTensor<float> cmpFp32Local;
    LocalTensor<uint8_t> cmpMaskLocal;

    LocalTensor<inputT> scatterValLocal;
    LocalTensor<int32_t> scatterIdxLocal;
    LocalTensor<inputT> scatterAlignedLocal;
    LocalTensor<float> cumsumLocal;
    LocalTensor<uint8_t> rmwStage;

    float negMaxValue = 0;
    float pValue = 0;
    float kthValue_ = 0;
    float reduceSumValue = 0;
    float reduceSumValueInvert = 0;
};

// --- InitTilingData ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::InitTilingData(
    const ApplyTopKTopPWithSortedTilingData& __restrict tilingData, GM_ADDR sorted_value, GM_ADDR sorted_indices,
    GM_ADDR p, GM_ADDR k, GM_ADDR logits, GM_ADDR out, GM_ADDR workspace)
{
    batchSize_ = tilingData.batchSize;
    vocabSize_ = tilingData.vocabSize;
    batchPerCore_ = tilingData.batchPerCore;
    tailBatch_ = tilingData.tailBatch;
    blockNum_ = tilingData.blockNum;
    calUbSize_ = static_cast<uint32_t>(tilingData.calUbSize);
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ < tailBatch_) {
        loopBatch_ = batchPerCore_ + 1;
        batchOffset_ = blockIdx_ * loopBatch_;
    } else {
        loopBatch_ = batchPerCore_;
        batchOffset_ = blockIdx_ * batchPerCore_ + tailBatch_;
    }
    uint32_t maxSfLen = (calUbSize_ - RESERVED_UB) / SOFTMAX_UB_NUM / FLOAT_BYTES;
    softmaxLength = maxSfLen < vocabSize_ ? maxSfLen : vocabSize_;
    // Keep the tile a whole number of 32B blocks so that every prefix-sum chunk and every softmax
    // write is block aligned instead of relying on the UB size happening to be a multiple of 32B.
    constexpr uint32_t sfBlockElems = BLOCK_BYTES / FLOAT_BYTES;
    if (softmaxLength >= sfBlockElems) {
        softmaxLength = softmaxLength / sfBlockElems * sfBlockElems;
    }
    lineSfLoopTimes = CeilDiv(vocabSize_, softmaxLength);
    softmaxLengthTail = vocabSize_ - (lineSfLoopTimes - 1) * softmaxLength;

    mGmSortedValue_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT*>(sorted_value));
    mGmSortedIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sorted_indices));
    mGmP_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT*>(p));
    mGmK_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(k));
    mGmLogits_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT*>(logits));
    mGmOut_.SetGlobalBuffer(reinterpret_cast<__gm__ outputT*>(out));
    softMaxGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out),
                              static_cast<uint64_t>(batchSize_) * vocabSize_ * sizeof(outputT) / sizeof(float));
}

// --- InitBuffer ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::InitBuffer(TPipe* inputPipe)
{
    pipe_ = inputPipe;
    pipe_->InitBuffer(calBuf_, calUbSize_);
    totalUb = calBuf_.Get<uint8_t>();

    uint32_t sfLenAligned = CeilDiv(softmaxLength, BLOCK_BYTES / sizeof(inputT)) * (BLOCK_BYTES / sizeof(inputT));
    sfLocalFp32 = totalUb.ReinterpretCast<float>();
    sfLocalInput = totalUb[sfLenAligned * sizeof(inputT)].ReinterpretCast<inputT>();
    sfResLocal = totalUb[sfLenAligned * sizeof(float)].ReinterpretCast<float>();
    reduceLocal = totalUb[sfLenAligned * sizeof(float) * 2].ReinterpretCast<float>();

    constexpr uint32_t DATA_PER_BLOCK = BLOCK_BYTES / sizeof(inputT);
    scatterLength = (calUbSize_ - RESERVED_UB - BLOCK_BYTES) /
                    (SOFTMAX_UB_NUM * FLOAT_BYTES + sizeof(inputT) + BLOCK_BYTES) / SCATTER_PART_LENGTH *
                    SCATTER_PART_LENGTH;
    scatterValLocal = totalUb[0].ReinterpretCast<inputT>();
    scatterIdxLocal = totalUb[scatterLength * sizeof(inputT)].ReinterpretCast<int32_t>();
    cumsumLocal = totalUb[scatterLength * (FLOAT_BYTES + sizeof(inputT))].ReinterpretCast<float>();
    uint32_t alignedOff = scatterLength * (SOFTMAX_UB_NUM * FLOAT_BYTES + sizeof(inputT));
    scatterAlignedLocal = totalUb[alignedOff].ReinterpretCast<inputT>();
    // cmpSelTileLength reserves one trailing block, so the last block is free for the RMW stage.
    rmwStage = totalUb[calUbSize_ - BLOCK_BYTES];

    constexpr uint32_t CMP_ALIGN_ELEMS = CMP_ALIGN_BYTES / sizeof(inputT);
    uint32_t cmpAvail = calUbSize_ - RESERVED_UB - BLOCK_BYTES;
    if constexpr (IsSameType<inputT, float>::value || IsSameType<inputT, half>::value) {
        cmpSelTileLength = cmpAvail / (2 * sizeof(inputT) + 1) / CMP_ALIGN_ELEMS * CMP_ALIGN_ELEMS;
        cmpLogitsLocal = totalUb[0].ReinterpretCast<inputT>();
        cmpNegInfLocal = totalUb[cmpSelTileLength * sizeof(inputT)].ReinterpretCast<inputT>();
        cmpMaskLocal = totalUb[cmpSelTileLength * 2 * sizeof(inputT)];
    } else {
        cmpSelTileLength = cmpAvail / (2 * sizeof(inputT) + sizeof(float) + 1) / CMP_ALIGN_ELEMS * CMP_ALIGN_ELEMS;
        cmpLogitsLocal = totalUb[0].ReinterpretCast<inputT>();
        cmpNegInfLocal = totalUb[cmpSelTileLength * sizeof(inputT)].ReinterpretCast<inputT>();
        cmpFp32Local = totalUb[cmpSelTileLength * 2 * sizeof(inputT)].ReinterpretCast<float>();
        cmpMaskLocal = totalUb[cmpSelTileLength * (2 * sizeof(inputT) + sizeof(float))];
    }
}

// --- GetMaxValue: get negated max from sorted values for softmax numerical stability ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::GetMaxValue(int64_t baseGmIdx)
{
    int64_t idx = baseGmIdx + vocabSize_ - 1;
    if constexpr (IsSameType<inputT, float>::value) {
        negMaxValue = -mGmSortedValue_[idx].GetValue(0);
    } else if constexpr (IsSameType<inputT, half>::value) {
        negMaxValue = -static_cast<float>(mGmSortedValue_[idx].GetValue(0));
    } else {
        negMaxValue = -ToFloat(mGmSortedValue_[idx].GetValue(0));
    }
}

// --- ComputeSoftmaxSum: tile-by-tile exp sum with TopK filtering ---
// For partial tile: only load valid data (from kthStartIdx); UB is pre-cleared by prior Duplicate.
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::ComputeSoftmaxSum(int64_t sortedBase, float kthVal,
                                                                                  uint32_t kthStartIdx)
{
    reduceSumValue = 0;
    uint32_t loopDataNum = softmaxLength;
    for (uint32_t loopInner = 0; loopInner < lineSfLoopTimes; loopInner++) {
        uint32_t tileStart = loopInner * softmaxLength;
        int64_t gmIdx = sortedBase + tileStart;
        if (loopInner == lineSfLoopTimes - 1)
            loopDataNum = softmaxLengthTail;
        uint32_t tileEnd = tileStart + loopDataNum;

        if (tileEnd <= kthStartIdx) {
            continue;
        }

        if (tileStart >= kthStartIdx) {
            if constexpr (!IsSameType<inputT, float>::value) {
                SafeCopyIn(sfLocalInput, mGmSortedValue_[gmIdx], loopDataNum);
                MTE2ToVSync();
                Cast(sfLocalFp32, sfLocalInput, RoundMode::CAST_NONE, loopDataNum);
                PipeBarrier<PIPE_V>();
            } else {
                SafeCopyIn(sfLocalFp32, mGmSortedValue_[gmIdx], loopDataNum);
                MTE2ToVSync();
            }
        } else {
            uint32_t localOffset = kthStartIdx - tileStart;
            const uint32_t partialTileDataNum = loopDataNum;
            if constexpr (!IsSameType<inputT, float>::value) {
                SafeCopyIn(sfLocalInput, mGmSortedValue_[gmIdx], partialTileDataNum);
                MTE2ToVSync();
                Cast(sfLocalFp32, sfLocalInput, RoundMode::CAST_NONE, partialTileDataNum);
                PipeBarrier<PIPE_V>();
            } else {
                SafeCopyIn(sfLocalFp32, mGmSortedValue_[gmIdx], partialTileDataNum);
                MTE2ToVSync();
            }
            Duplicate(sfLocalFp32, -1.0e30f, localOffset);
            PipeBarrier<PIPE_V>();
        }

        Adds(sfResLocal, sfLocalFp32, negMaxValue, loopDataNum);
        PipeBarrier<PIPE_V>();
        Exp(sfResLocal, sfResLocal, loopDataNum);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceLocal, sfResLocal, reduceLocal, loopDataNum);
        VToSSync();
        reduceSumValue += reduceLocal.GetValue(0);
        SToVSync();
    }
    reduceSumValueInvert = (reduceSumValue > 0.0f) ? (1.0f / reduceSumValue) : 0.0f;
}

// --- WriteSoftmaxToGm: compute normalized probabilities for [dataStart, dataStart+dataLen), write to softMaxGm ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::WriteSoftmaxToGm(int64_t sortedBase, int64_t sfGmBase,
                                                                                 uint32_t kthStartIdx,
                                                                                 uint32_t dataStart, uint32_t dataLen)
{
    uint32_t dataEnd = dataStart + dataLen;
    uint32_t sfTileLoopTimes = CeilDiv(dataLen, softmaxLength);
    uint32_t sfTileTail = dataLen - (sfTileLoopTimes - 1) * softmaxLength;
    uint32_t loopDataNum = softmaxLength;
    for (uint32_t loopInner = 0; loopInner < sfTileLoopTimes; loopInner++) {
        uint32_t tileStart = dataStart + loopInner * softmaxLength;
        int64_t sortedGmIdx = sortedBase + tileStart;
        int64_t sfGmIdx = sfGmBase + loopInner * softmaxLength;
        if (loopInner == sfTileLoopTimes - 1)
            loopDataNum = sfTileTail;
        uint32_t tileEnd = tileStart + loopDataNum;

        if (tileEnd <= kthStartIdx) {
            Duplicate(sfResLocal, 0.0f, loopDataNum);
            VToMTE3Sync();
            SafeCopyOut(softMaxGm[sfGmIdx], sfResLocal, loopDataNum);
            MTE3ToMTE2Sync();
            continue;
        }

        if constexpr (!IsSameType<inputT, float>::value) {
            SafeCopyIn(sfLocalInput, mGmSortedValue_[sortedGmIdx], loopDataNum);
            MTE2ToVSync();
            Cast(sfLocalFp32, sfLocalInput, RoundMode::CAST_NONE, loopDataNum);
            PipeBarrier<PIPE_V>();
        } else {
            SafeCopyIn(sfLocalFp32, mGmSortedValue_[sortedGmIdx], loopDataNum);
            MTE2ToVSync();
        }

        if (tileStart < kthStartIdx) {
            uint32_t localOffset = kthStartIdx - tileStart;
            Duplicate(sfLocalFp32, -1.0e30f, localOffset);
            PipeBarrier<PIPE_V>();
        }

        Adds(sfResLocal, sfLocalFp32, negMaxValue, loopDataNum);
        PipeBarrier<PIPE_V>();
        Exp(sfResLocal, sfResLocal, loopDataNum);
        PipeBarrier<PIPE_V>();
        Muls(sfResLocal, sfResLocal, reduceSumValueInvert, loopDataNum);
        VToMTE3Sync();
        SafeCopyOut(softMaxGm[sfGmIdx], sfResLocal, loopDataNum);
        MTE3ToMTE2Sync();
    }
}

// --- CumsumKoggleStone: chunked prefix sum, done inside UB ---
// The original implementation ran the Kogge-Stone passes directly on GM at offsets 1/2/4/...,
// which are not 32B aligned. On 310P that forces scalar access plus cache flushes for every pass.
// Here each chunk is scanned in UB (where element offsets are free) and only aligned chunks are
// written back, with the running total carried across chunks.
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::CumsumKoggleStone(int64_t sfGmBase, uint32_t dataLen)
{
    constexpr uint32_t elemsPerBlock = BLOCK_BYTES / sizeof(float);
    // The chunk must not exceed dataLen, otherwise the vector loop below never runs and the whole
    // prefix sum degrades to a serial scalar chain. That is the common case for fp16/bf16 on medium
    // vocabularies, where dataLen is halfLen and can be far below softmaxLength.
    uint32_t maxChunk = softmaxLength < dataLen ? softmaxLength : dataLen;
    uint32_t chunkLen = maxChunk / elemsPerBlock * elemsPerBlock;
    if (chunkLen == 0) {
        chunkLen = elemsPerBlock;
    }

    float running = 0.0f;
    uint32_t done = 0;
    for (; done + chunkLen <= dataLen; done += chunkLen) {
        SafeCopyIn(sfLocalFp32, softMaxGm[sfGmBase + done], chunkLen);
        MTE2ToVSync();

        LocalTensor<float> src = sfLocalFp32;
        LocalTensor<float> dst = sfResLocal;
        for (uint32_t offset = 1; offset < chunkLen; offset <<= 1) {
            Adds(dst, src, 0.0f, offset);
            PipeBarrier<PIPE_V>();
            Add(dst[offset], src[offset], src, chunkLen - offset);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> tmp = src;
            src = dst;
            dst = tmp;
        }
        if (running != 0.0f) {
            Adds(src, src, running, chunkLen);
            PipeBarrier<PIPE_V>();
        }
        VToMTE3Sync();
        SafeCopyOut(softMaxGm[sfGmBase + done], src, chunkLen);
        MTE3ToMTE2Sync();

        VToSSync();
        running = src.GetValue(chunkLen - 1);
        SToVSync();
    }

    // Tail shorter than one chunk: scan it in UB and write it back through SafeCopyOut. Scalar
    // stores to GM would sit in the per-core data cache, and flushing a line that is shared with
    // the neighbouring batch row would overwrite that row's output.
    if (done < dataLen) {
        uint32_t tail = dataLen - done;
        SafeCopyIn(sfLocalFp32, softMaxGm[sfGmBase + done], tail);
        MTE2ToSSync();
        for (uint32_t i = 0; i < tail; i++) {
            running += sfLocalFp32.GetValue(i);
            sfLocalFp32.SetValue(i, running);
        }
        SToMTE3Sync();
        SafeCopyOut(softMaxGm[sfGmBase + done], sfLocalFp32, tail);
        MTE3ToMTE2Sync();
    }
}

// --- FindFirstIndex: read cumsum from softMaxGm to find threshold crossing ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::FindFirstIndex(int64_t sfGmBase, uint32_t dataLen,
                                                                               float cumsumBase, uint32_t& firstIdx)
{
    firstIdx = dataLen;
    bool found = false;
    float threshold = pValue - cumsumBase;
    uint32_t searchTimes = CeilDiv(dataLen, scatterLength);
    uint32_t searchTail = dataLen - (searchTimes - 1) * scatterLength;
    for (uint32_t sIdx = 0; sIdx < searchTimes && !found; sIdx++) {
        uint32_t curLen = (sIdx == searchTimes - 1) ? searchTail : scatterLength;
        int64_t gmOff = sfGmBase + sIdx * scatterLength;
        SafeCopyIn(cumsumLocal, softMaxGm[gmOff], curLen);
        MTE2ToSSync();
        if (cumsumLocal.GetValue(curLen - 1) <= threshold) {
            continue;
        }
        uint32_t scatterLoop = CeilDiv(curLen, SCATTER_PART_LENGTH);
        uint32_t scatterNumsTail = curLen - (scatterLoop - 1) * SCATTER_PART_LENGTH;
        for (uint32_t si = 0; si < scatterLoop && !found; si++) {
            uint32_t curNums = (si == scatterLoop - 1) ? scatterNumsTail : SCATTER_PART_LENGTH;
            if (cumsumLocal.GetValue(si * SCATTER_PART_LENGTH + curNums - 1) <= threshold) {
                continue;
            }
            for (uint32_t i = 0; i < curNums; i++) {
                uint32_t off = si * SCATTER_PART_LENGTH + i;
                if (cumsumLocal.GetValue(off) > threshold) {
                    firstIdx = sIdx * scatterLength + off;
                    found = true;
                    break;
                }
            }
        }
    }
}

// --- ProcessCumsumAndFind: write softmax, cumsum, and find threshold index using output space ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::ProcessCumsumAndFind(int64_t sortedBase,
                                                                                     int64_t batchGmBase,
                                                                                     uint32_t kthStartIdx,
                                                                                     uint32_t& firstIdx)
{
    if constexpr (IsSameType<outputT, float>::value) {
        int64_t sfGmBase = batchGmBase;
        WriteSoftmaxToGm(sortedBase, sfGmBase, kthStartIdx, 0, vocabSize_);
        CumsumKoggleStone(sfGmBase, vocabSize_);
        FindFirstIndex(sfGmBase, vocabSize_, 0.0f, firstIdx);
    } else {
        int64_t sfGmBase = (batchGmBase + 1) / 2;
        uint32_t evenLen = vocabSize_ & ~1u;
        uint32_t halfLen = evenLen / 2;

        float lastElemProb = 0.0f;
        if (vocabSize_ & 1u) {
            int64_t lastIdx = sortedBase + vocabSize_ - 1;
            float lastVal;
            if constexpr (IsSameType<inputT, half>::value) {
                lastVal = static_cast<float>(mGmSortedValue_[lastIdx].GetValue(0));
            } else {
                lastVal = ToFloat(mGmSortedValue_[lastIdx].GetValue(0));
            }
            sfLocalFp32.SetValue(0, lastVal + negMaxValue);
            Exp(sfResLocal, sfLocalFp32, 1);
            VToSSync();
            lastElemProb = sfResLocal.GetValue(0) * reduceSumValueInvert;
            SToVSync();
        }

        WriteSoftmaxToGm(sortedBase, sfGmBase, kthStartIdx, 0, halfLen);
        CumsumKoggleStone(sfGmBase, halfLen);

        uint32_t localIdx = halfLen;
        FindFirstIndex(sfGmBase, halfLen, 0.0f, localIdx);
        if (localIdx < halfLen) {
            firstIdx = localIdx;
            return;
        }

        SafeCopyIn(cumsumLocal, softMaxGm[sfGmBase + halfLen - 1], 1);
        MTE2ToSSync();
        float firstHalfCumsum = cumsumLocal.GetValue(0);

        WriteSoftmaxToGm(sortedBase, sfGmBase, kthStartIdx, halfLen, halfLen);
        CumsumKoggleStone(sfGmBase, halfLen);

        localIdx = halfLen;
        FindFirstIndex(sfGmBase, halfLen, firstHalfCumsum, localIdx);
        if (localIdx < halfLen) {
            firstIdx = halfLen + localIdx;
            return;
        }

        if (vocabSize_ & 1u) {
            SafeCopyIn(cumsumLocal, softMaxGm[sfGmBase + halfLen - 1], 1);
            MTE2ToSSync();
            float secondHalfCumsum = cumsumLocal.GetValue(0);
            float totalEvenCumsum = firstHalfCumsum + secondHalfCumsum;
            if (totalEvenCumsum + lastElemProb > pValue) {
                firstIdx = vocabSize_ - 1;
            } else {
                firstIdx = vocabSize_;
            }
        } else {
            firstIdx = vocabSize_;
        }
    }
}

// --- CompareSelectOnLogits: vectorized compare + select on original logits ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::CompareSelectOnLogits(int64_t batchGmBase,
                                                                                      inputT thresholdVal, bool useGE)
{
    constexpr uint32_t CMP_ALIGN_ELEMS = CMP_ALIGN_BYTES / sizeof(inputT);
    uint32_t tileTimes = CeilDiv(vocabSize_, cmpSelTileLength);
    uint32_t tileTail = vocabSize_ - (tileTimes - 1) * cmpSelTileLength;

    for (uint32_t tIdx = 0; tIdx < tileTimes; tIdx++) {
        uint32_t curLen = (tIdx == tileTimes - 1) ? tileTail : cmpSelTileLength;
        uint32_t curLenAligned = CeilDiv(curLen, CMP_ALIGN_ELEMS) * CMP_ALIGN_ELEMS;
        int64_t gmOff = batchGmBase + static_cast<int64_t>(tIdx) * cmpSelTileLength;

        SafeCopyIn(cmpLogitsLocal, mGmLogits_[gmOff], curLen);
        MTE2ToVSync();

        if constexpr (IsSameType<inputT, float>::value) {
            Duplicate(cmpNegInfLocal.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, curLenAligned);
            PipeBarrier<PIPE_V>();
            CMPMODE cmpMode = useGE ? CMPMODE::GE : CMPMODE::GT;
            Compares(cmpMaskLocal, cmpLogitsLocal, thresholdVal, cmpMode, curLenAligned);
            PipeBarrier<PIPE_V>();
            Select(cmpLogitsLocal, cmpMaskLocal, cmpLogitsLocal, cmpNegInfLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   curLenAligned);
        } else if constexpr (IsSameType<inputT, half>::value) {
            Duplicate(cmpNegInfLocal.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, curLenAligned);
            PipeBarrier<PIPE_V>();
            CMPMODE cmpMode = useGE ? CMPMODE::GE : CMPMODE::GT;
            Compares(cmpMaskLocal, cmpLogitsLocal, thresholdVal, cmpMode, curLenAligned);
            PipeBarrier<PIPE_V>();
            Select(cmpLogitsLocal, cmpMaskLocal, cmpLogitsLocal, cmpNegInfLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   curLenAligned);
        } else {
            Duplicate(cmpNegInfLocal.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, curLenAligned);
            PipeBarrier<PIPE_V>();
            Cast(cmpFp32Local, cmpLogitsLocal, RoundMode::CAST_NONE, curLenAligned);
            PipeBarrier<PIPE_V>();
            float threshFp32 = ToFloat(thresholdVal);
            CMPMODE cmpMode = useGE ? CMPMODE::GE : CMPMODE::GT;
            Compares(cmpMaskLocal, cmpFp32Local, threshFp32, cmpMode, curLenAligned);
            PipeBarrier<PIPE_V>();
            Select(cmpLogitsLocal.template ReinterpretCast<half>(), cmpMaskLocal,
                   cmpLogitsLocal.template ReinterpretCast<half>(), cmpNegInfLocal.template ReinterpretCast<half>(),
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, curLenAligned);
        }
        VToMTE3Sync();
        SafeCopyOut(mGmOut_[gmOff], cmpLogitsLocal, curLen);
        MTE3ToMTE2Sync();
    }
}

// --- ScatterBoundary: scatter elements == threshold at sorted positions >= firstIndex ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::ScatterBoundary(int64_t batchGmBase, int64_t sortedBase,
                                                                                uint32_t firstIdx, inputT thresholdVal)
{
    constexpr uint32_t DATA_PER_BLOCK = BLOCK_BYTES / sizeof(inputT);
    constexpr uint32_t BRCB_ELEM_PER_REP = 8;
    constexpr uint32_t BRCB_SRC_ALIGN = BLOCK_BYTES / sizeof(inputT);
    constexpr uint32_t BRCB_ALIGN_REPEATS = BRCB_SRC_ALIGN / BRCB_ELEM_PER_REP;
    constexpr uint32_t MAX_BRCB_REPEAT = 255;
    constexpr uint32_t ALIGNED_MAX_REPEAT = (MAX_BRCB_REPEAT / BRCB_ALIGN_REPEATS) * BRCB_ALIGN_REPEATS;

    float threshFloat;
    if constexpr (IsSameType<inputT, float>::value) {
        threshFloat = thresholdVal;
    } else if constexpr (IsSameType<inputT, half>::value) {
        threshFloat = static_cast<float>(thresholdVal);
    } else {
        threshFloat = ToFloat(thresholdVal);
    }

    uint32_t equalCount = 0;
    for (uint32_t ei = firstIdx; ei < vocabSize_; ei++) {
        float val;
        if constexpr (IsSameType<inputT, float>::value) {
            val = mGmSortedValue_[sortedBase + ei].GetValue(0);
        } else if constexpr (IsSameType<inputT, half>::value) {
            val = static_cast<float>(mGmSortedValue_[sortedBase + ei].GetValue(0));
        } else {
            val = ToFloat(mGmSortedValue_[sortedBase + ei].GetValue(0));
        }
        if (val != threshFloat)
            break;
        equalCount++;
    }
    if (equalCount == 0)
        return;

    uint32_t copyTimes = CeilDiv(equalCount, scatterLength);
    uint32_t copyTail = equalCount - (copyTimes - 1) * scatterLength;
    for (uint32_t ci = 0; ci < copyTimes; ci++) {
        uint32_t curLen = (ci == copyTimes - 1) ? copyTail : scatterLength;
        int64_t gmOff = sortedBase + firstIdx + ci * scatterLength;
        SafeCopyIn(scatterIdxLocal, mGmSortedIndices_[gmOff], curLen);
        SafeCopyIn(scatterValLocal, mGmSortedValue_[gmOff], curLen);
        MTE2ToVSync();
        MTE2ToSSync();

        uint32_t totalRepeat = CeilDiv(curLen, BRCB_ELEM_PER_REP);
        uint32_t brcbDone = 0;
        while (brcbDone < totalRepeat) {
            uint32_t remaining = totalRepeat - brcbDone;
            uint32_t curRepeat = remaining > ALIGNED_MAX_REPEAT ?
                                     ALIGNED_MAX_REPEAT :
                                     (remaining + BRCB_ALIGN_REPEATS - 1) / BRCB_ALIGN_REPEATS * BRCB_ALIGN_REPEATS;
            Brcb(scatterAlignedLocal[brcbDone * BRCB_ELEM_PER_REP * DATA_PER_BLOCK],
                 scatterValLocal[brcbDone * BRCB_ELEM_PER_REP], curRepeat, {1, 8});
            brcbDone += curRepeat;
        }
        VToMTE3Sync();

        for (uint32_t idx = 0; idx < curLen; idx++) {
            int32_t lineIndex = scatterIdxLocal.GetValue(idx);
            SafeCopyOut(mGmOut_[batchGmBase + lineIndex], scatterAlignedLocal[idx * DATA_PER_BLOCK], 1);
        }
        MTE3ToSSync();
    }
}

// --- CopyOutLast: unconditionally output the last sorted element ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::CopyOutLast(int64_t batchGmBase, int64_t sortedBase)
{
    constexpr uint32_t DATA_PER_BLOCK = BLOCK_BYTES / sizeof(inputT);
    int64_t lastOff = sortedBase + vocabSize_ - 1;
    SafeCopyIn(scatterAlignedLocal, mGmSortedValue_[lastOff], 1);
    int32_t lineIndex = mGmSortedIndices_.GetValue(lastOff);
    MTE2ToSSync();
    SToMTE3Sync();
    SafeCopyOut(mGmOut_[batchGmBase + lineIndex], scatterAlignedLocal.template ReinterpretCast<outputT>(), 1);
    MTE3ToSSync();
}

// --- FindKthStartIdx: find actual start index by searching backward for boundary ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline uint32_t ApplyTopKTopPOpt<inputT, calT, outputT>::FindKthStartIdx(int64_t sortedBase, int32_t kValue)
{
    int64_t kthIdx = sortedBase + vocabSize_ - kValue;
    float kthFloat;
    if constexpr (IsSameType<inputT, float>::value) {
        kthFloat = mGmSortedValue_[kthIdx].GetValue(0);
    } else if constexpr (IsSameType<inputT, half>::value) {
        kthFloat = static_cast<float>(mGmSortedValue_[kthIdx].GetValue(0));
    } else {
        kthFloat = ToFloat(mGmSortedValue_[kthIdx].GetValue(0));
    }

    uint32_t kthStartIdx = vocabSize_ - static_cast<uint32_t>(kValue);
    while (kthStartIdx > 0) {
        uint32_t chunkLen = (kthStartIdx > scatterLength) ? scatterLength : kthStartIdx;
        uint32_t chunkStart = kthStartIdx - chunkLen;
        SafeCopyIn(scatterValLocal, mGmSortedValue_[sortedBase + chunkStart], chunkLen);
        MTE2ToSSync();
        bool foundBoundary = false;
        for (int32_t i = static_cast<int32_t>(chunkLen) - 1; i >= 0; i--) {
            float prevVal;
            if constexpr (IsSameType<inputT, float>::value) {
                prevVal = scatterValLocal.GetValue(i);
            } else if constexpr (IsSameType<inputT, half>::value) {
                prevVal = static_cast<float>(scatterValLocal.GetValue(i));
            } else {
                prevVal = ToFloat(scatterValLocal.GetValue(i));
            }
            if (prevVal < kthFloat) {
                kthStartIdx = chunkStart + static_cast<uint32_t>(i) + 1;
                foundBoundary = true;
                break;
            }
        }
        if (foundBoundary)
            break;
        kthStartIdx = chunkStart;
    }
    return kthStartIdx;
}

// --- FillNegInfAndScatterTopK: fill output with -inf, then Brcb scatter elements from kthStartIdx ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::FillNegInfAndScatterTopK(int64_t batchGmBase,
                                                                                         int64_t sortedBase,
                                                                                         uint32_t kthStartIdx)
{
    constexpr uint32_t DATA_PER_BLOCK = BLOCK_BYTES / sizeof(inputT);
    constexpr uint32_t BRCB_ELEM_PER_REP = 8;
    constexpr uint32_t BRCB_SRC_ALIGN = BLOCK_BYTES / sizeof(inputT);
    constexpr uint32_t BRCB_ALIGN_REPEATS = BRCB_SRC_ALIGN / BRCB_ELEM_PER_REP;
    constexpr uint32_t MAX_BRCB_REPEAT = 255;
    constexpr uint32_t ALIGNED_MAX_REPEAT = (MAX_BRCB_REPEAT / BRCB_ALIGN_REPEATS) * BRCB_ALIGN_REPEATS;

    uint32_t tileTimes = CeilDiv(vocabSize_, cmpSelTileLength);
    uint32_t tileTail = vocabSize_ - (tileTimes - 1) * cmpSelTileLength;
    for (uint32_t tIdx = 0; tIdx < tileTimes; tIdx++) {
        uint32_t curLen = (tIdx == tileTimes - 1) ? tileTail : cmpSelTileLength;
        int64_t gmOff = batchGmBase + static_cast<int64_t>(tIdx) * cmpSelTileLength;
        if constexpr (IsSameType<inputT, float>::value) {
            Duplicate(cmpLogitsLocal.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, curLen);
        } else if constexpr (IsSameType<inputT, half>::value) {
            Duplicate(cmpLogitsLocal.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, curLen);
        } else {
            Duplicate(cmpLogitsLocal.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, curLen);
        }
        VToMTE3Sync();
        SafeCopyOut(mGmOut_[gmOff], cmpLogitsLocal, curLen);
        MTE3ToMTE2Sync();
    }

    uint32_t topKCount = vocabSize_ - kthStartIdx;
    int64_t topKStart = sortedBase + kthStartIdx;

    uint32_t copyTimes = CeilDiv(topKCount, scatterLength);
    uint32_t copyTail = topKCount - (copyTimes - 1) * scatterLength;
    for (uint32_t ci = 0; ci < copyTimes; ci++) {
        uint32_t curLen = (ci == copyTimes - 1) ? copyTail : scatterLength;
        int64_t gmOff = topKStart + ci * scatterLength;
        SafeCopyIn(scatterValLocal, mGmSortedValue_[gmOff], curLen);
        SafeCopyIn(scatterIdxLocal, mGmSortedIndices_[gmOff], curLen);
        MTE2ToVSync();
        MTE2ToSSync();

        uint32_t totalRepeat = CeilDiv(curLen, BRCB_ELEM_PER_REP);
        uint32_t brcbDone = 0;
        while (brcbDone < totalRepeat) {
            uint32_t remaining = totalRepeat - brcbDone;
            uint32_t curRepeat = remaining > ALIGNED_MAX_REPEAT ?
                                     ALIGNED_MAX_REPEAT :
                                     (remaining + BRCB_ALIGN_REPEATS - 1) / BRCB_ALIGN_REPEATS * BRCB_ALIGN_REPEATS;
            Brcb(scatterAlignedLocal[brcbDone * BRCB_ELEM_PER_REP * DATA_PER_BLOCK],
                 scatterValLocal[brcbDone * BRCB_ELEM_PER_REP], curRepeat, {1, 8});
            brcbDone += curRepeat;
        }
        VToMTE3Sync();

        for (uint32_t idx = 0; idx < curLen; idx++) {
            int32_t lineIndex = scatterIdxLocal.GetValue(idx);
            SafeCopyOut(mGmOut_[batchGmBase + lineIndex], scatterAlignedLocal[idx * DATA_PER_BLOCK], 1);
        }
        MTE3ToSSync();
    }
}

// --- ProcessTopKTopPOpt: main entry for optimized TopKTopP ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::ProcessTopKTopPOpt()
{
    for (uint32_t loopBatch = 0; loopBatch < loopBatch_; loopBatch++) {
        int64_t sortedBase = (batchOffset_ + loopBatch) * vocabSize_;
        int64_t batchGmBase = sortedBase;

        int32_t kValue = mGmK_.GetValue(batchOffset_ + loopBatch);

        if constexpr (IsSameType<inputT, float>::value) {
            pValue = 1.0f - mGmP_[batchOffset_ + loopBatch].GetValue(0);
        } else if constexpr (IsSameType<inputT, half>::value) {
            pValue = 1.0f - static_cast<float>(mGmP_[batchOffset_ + loopBatch].GetValue(0));
        } else {
            pValue = 1.0f - ToFloat(mGmP_[batchOffset_ + loopBatch].GetValue(0));
        }

        GetMaxValue(sortedBase);

        if (*reinterpret_cast<int32_t*>(&negMaxValue) == FLOAT32_NEG_INF ||
            *reinterpret_cast<int32_t*>(&negMaxValue) == FLOAT32_INF) {
            if (kValue <= 0 || static_cast<uint32_t>(kValue) >= vocabSize_) {
                uint32_t tileTimes = CeilDiv(vocabSize_, cmpSelTileLength);
                uint32_t tileTail = vocabSize_ - (tileTimes - 1) * cmpSelTileLength;
                for (uint32_t tIdx = 0; tIdx < tileTimes; tIdx++) {
                    uint32_t curLen = (tIdx == tileTimes - 1) ? tileTail : cmpSelTileLength;
                    int64_t gmOff = batchGmBase + static_cast<int64_t>(tIdx) * cmpSelTileLength;
                    SafeCopyIn(cmpLogitsLocal, mGmLogits_[gmOff], curLen);
                    MTE2ToVSync();
                    VToMTE3Sync();
                    SafeCopyOut(mGmOut_[gmOff], cmpLogitsLocal, curLen);
                    MTE3ToMTE2Sync();
                }
            } else {
                uint32_t kthStart = FindKthStartIdx(sortedBase, kValue);
                uint32_t actualTopKCount = vocabSize_ - kthStart;
                if (actualTopKCount <= scatterLength) {
                    FillNegInfAndScatterTopK(batchGmBase, sortedBase, kthStart);
                } else {
                    int64_t kthIdx = sortedBase + kthStart;
                    inputT kthValRaw = mGmSortedValue_[kthIdx].GetValue(0);
                    CompareSelectOnLogits(batchGmBase, kthValRaw, true);
                    CopyOutLast(batchGmBase, sortedBase);
                }
            }
            continue;
        }

        uint32_t kthStartIdx = 0;
        float kthVal = 0;
        if (kValue > 0 && static_cast<uint32_t>(kValue) < vocabSize_) {
            kthStartIdx = FindKthStartIdx(sortedBase, kValue);
            int64_t kthIdx = sortedBase + kthStartIdx;
            if constexpr (IsSameType<inputT, float>::value) {
                kthVal = mGmSortedValue_[kthIdx].GetValue(0);
            } else if constexpr (IsSameType<inputT, half>::value) {
                kthVal = static_cast<float>(mGmSortedValue_[kthIdx].GetValue(0));
            } else {
                kthVal = ToFloat(mGmSortedValue_[kthIdx].GetValue(0));
            }
            kthValue_ = kthVal;
        }

        if (kValue > 0 && (vocabSize_ - kthStartIdx) <= scatterLength && (vocabSize_ - kthStartIdx) <= softmaxLength) {
            constexpr uint32_t DATA_PER_BLOCK = BLOCK_BYTES / sizeof(inputT);
            constexpr uint32_t BRCB_ELEM_PER_REP = 8;
            constexpr uint32_t BRCB_SRC_ALIGN = BLOCK_BYTES / sizeof(inputT);
            constexpr uint32_t BRCB_ALIGN_REPEATS = BRCB_SRC_ALIGN / BRCB_ELEM_PER_REP;
            constexpr uint32_t MAX_BRCB_REPEAT = 255;
            constexpr uint32_t ALIGNED_MAX_REPEAT = (MAX_BRCB_REPEAT / BRCB_ALIGN_REPEATS) * BRCB_ALIGN_REPEATS;

            uint32_t topKActual = vocabSize_ - kthStartIdx;
            int64_t topKStart = sortedBase + kthStartIdx;

            if constexpr (!IsSameType<inputT, float>::value) {
                SafeCopyIn(sfLocalInput, mGmSortedValue_[topKStart], topKActual);
                MTE2ToVSync();
                Cast(sfLocalFp32, sfLocalInput, RoundMode::CAST_NONE, topKActual);
                PipeBarrier<PIPE_V>();
            } else {
                SafeCopyIn(sfLocalFp32, mGmSortedValue_[topKStart], topKActual);
                MTE2ToVSync();
            }
            Adds(sfResLocal, sfLocalFp32, negMaxValue, topKActual);
            PipeBarrier<PIPE_V>();
            Exp(sfResLocal, sfResLocal, topKActual);
            PipeBarrier<PIPE_V>();
            ReduceSum(reduceLocal, sfResLocal, reduceLocal, topKActual);
            VToSSync();
            float localReduceSum = reduceLocal.GetValue(0);
            float localReduceSumInv = (localReduceSum > 0.0f) ? (1.0f / localReduceSum) : 0.0f;
            SToVSync();

            Muls(sfResLocal, sfResLocal, localReduceSumInv, topKActual);
            VToSSync();

            float cumVal = 0.0f;
            uint32_t firstScatterIdx = topKActual - 1;
            for (uint32_t ci = 0; ci < topKActual; ci++) {
                cumVal += sfResLocal.GetValue(ci);
                if (cumVal > pValue) {
                    firstScatterIdx = ci;
                    break;
                }
            }

            uint32_t scatterCount = topKActual - firstScatterIdx;
            int64_t scatterGmStart = topKStart + firstScatterIdx;

            uint32_t outputTileTimes = CeilDiv(vocabSize_, cmpSelTileLength);
            uint32_t outputTileTail = vocabSize_ - (outputTileTimes - 1) * cmpSelTileLength;
            for (uint32_t outputTileIdx = 0; outputTileIdx < outputTileTimes; outputTileIdx++) {
                uint32_t outputTileLen = (outputTileIdx == outputTileTimes - 1) ? outputTileTail : cmpSelTileLength;
                int64_t outputGmOffset = batchGmBase + static_cast<int64_t>(outputTileIdx) * cmpSelTileLength;
                if constexpr (IsSameType<inputT, float>::value) {
                    Duplicate(cmpLogitsLocal.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, outputTileLen);
                } else if constexpr (IsSameType<inputT, half>::value) {
                    Duplicate(cmpLogitsLocal.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, outputTileLen);
                } else {
                    Duplicate(cmpLogitsLocal.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, outputTileLen);
                }
                VToMTE3Sync();
                SafeCopyOut(mGmOut_[outputGmOffset], cmpLogitsLocal, outputTileLen);
                MTE3ToMTE2Sync();
            }

            uint32_t scCopyTimes = CeilDiv(scatterCount, scatterLength);
            uint32_t scCopyTail = scatterCount - (scCopyTimes - 1) * scatterLength;
            for (uint32_t sci = 0; sci < scCopyTimes; sci++) {
                uint32_t curScLen = (sci == scCopyTimes - 1) ? scCopyTail : scatterLength;
                int64_t scGmOff = scatterGmStart + sci * scatterLength;

                SafeCopyIn(scatterValLocal, mGmSortedValue_[scGmOff], curScLen);
                SafeCopyIn(scatterIdxLocal, mGmSortedIndices_[scGmOff], curScLen);
                MTE2ToVSync();
                MTE2ToSSync();

                uint32_t totalRepeat = CeilDiv(curScLen, BRCB_ELEM_PER_REP);
                uint32_t brcbDone = 0;
                while (brcbDone < totalRepeat) {
                    uint32_t remaining = totalRepeat - brcbDone;
                    uint32_t curRepeat = remaining > ALIGNED_MAX_REPEAT ? ALIGNED_MAX_REPEAT :
                                                                          (remaining + BRCB_ALIGN_REPEATS - 1) /
                                                                              BRCB_ALIGN_REPEATS * BRCB_ALIGN_REPEATS;
                    Brcb(scatterAlignedLocal[brcbDone * BRCB_ELEM_PER_REP * DATA_PER_BLOCK],
                         scatterValLocal[brcbDone * BRCB_ELEM_PER_REP], curRepeat, {1, 8});
                    brcbDone += curRepeat;
                }
                VToMTE3Sync();

                for (uint32_t idx = 0; idx < curScLen; idx++) {
                    int32_t lineIndex = scatterIdxLocal.GetValue(idx);
                    SafeCopyOut(mGmOut_[batchGmBase + lineIndex], scatterAlignedLocal[idx * DATA_PER_BLOCK], 1);
                }
                MTE3ToSSync();
            }
            continue;
        }

        ComputeSoftmaxSum(sortedBase, kthVal, kthStartIdx);

        uint32_t firstIdx = vocabSize_;
        ProcessCumsumAndFind(sortedBase, batchGmBase, kthStartIdx, firstIdx);
        if (firstIdx >= vocabSize_)
            firstIdx = vocabSize_ - 1;

        inputT thresholdVal = mGmSortedValue_[sortedBase + firstIdx].GetValue(0);

        // Always use GE compare (keeps all elements >= threshold)
        CompareSelectOnLogits(batchGmBase, thresholdVal, true);

        // Exclude elements == threshold at sorted positions BEFORE firstIdx
        // These were incorrectly kept by GE, need to write -inf back
        float threshFloat;
        if constexpr (IsSameType<inputT, float>::value) {
            threshFloat = thresholdVal;
        } else if constexpr (IsSameType<inputT, half>::value) {
            threshFloat = static_cast<float>(thresholdVal);
        } else {
            threshFloat = ToFloat(thresholdVal);
        }

        uint32_t excludeCount = 0;
        if (firstIdx > 0) {
            uint32_t searchPos = firstIdx;
            bool exDone = false;
            while (searchPos > 0 && !exDone) {
                uint32_t chunkLen = (searchPos > scatterLength) ? scatterLength : searchPos;
                uint32_t chunkStart = searchPos - chunkLen;
                SafeCopyIn(scatterValLocal, mGmSortedValue_[sortedBase + chunkStart], chunkLen);
                MTE2ToSSync();
                for (int32_t i = static_cast<int32_t>(chunkLen) - 1; i >= 0; i--) {
                    float val;
                    if constexpr (IsSameType<inputT, float>::value) {
                        val = scatterValLocal.GetValue(i);
                    } else if constexpr (IsSameType<inputT, half>::value) {
                        val = static_cast<float>(scatterValLocal.GetValue(i));
                    } else {
                        val = ToFloat(scatterValLocal.GetValue(i));
                    }
                    if (val != threshFloat) {
                        exDone = true;
                        break;
                    }
                    excludeCount++;
                }
                searchPos = chunkStart;
            }
        }

        if (excludeCount > 0) {
            uint32_t excludeStart = firstIdx - excludeCount;
            uint32_t writeTimes = CeilDiv(excludeCount, scatterLength);
            uint32_t writeTail = excludeCount - (writeTimes - 1) * scatterLength;
            for (uint32_t wi = 0; wi < writeTimes; wi++) {
                uint32_t curLen = (wi == writeTimes - 1) ? writeTail : scatterLength;
                int64_t idxGmOff = sortedBase + excludeStart + wi * scatterLength;
                SafeCopyIn(scatterIdxLocal, mGmSortedIndices_[idxGmOff], curLen);
                MTE2ToSSync();
                if constexpr (IsSameType<inputT, float>::value) {
                    scatterAlignedLocal.template ReinterpretCast<int32_t>().SetValue(0, FLOAT32_NEG_INF);
                } else if constexpr (IsSameType<inputT, half>::value) {
                    scatterAlignedLocal.template ReinterpretCast<uint16_t>().SetValue(0, FLOAT16_NEG_INF);
                } else {
                    scatterAlignedLocal.template ReinterpretCast<uint16_t>().SetValue(0, BF16_NEG_INF);
                }
                SToMTE3Sync();
                for (uint32_t ei = 0; ei < curLen; ei++) {
                    int32_t lineIndex = scatterIdxLocal.GetValue(ei);
                    SafeCopyOut(mGmOut_[batchGmBase + lineIndex],
                                scatterAlignedLocal.template ReinterpretCast<outputT>(), 1);
                }
                MTE3ToSSync();
            }
        }
        CopyOutLast(batchGmBase, sortedBase);
    }
}

// --- ProcessTopKOpt: main entry for optimized TopK-only ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::ProcessTopKOpt()
{
    for (uint32_t loopBatch = 0; loopBatch < loopBatch_; loopBatch++) {
        int64_t sortedBase = (batchOffset_ + loopBatch) * vocabSize_;
        int64_t batchGmBase = sortedBase;

        int32_t kValue = mGmK_.GetValue(batchOffset_ + loopBatch);

        if (kValue <= 0 || static_cast<uint32_t>(kValue) >= vocabSize_) {
            uint32_t logitsCopyTileTimes = CeilDiv(vocabSize_, cmpSelTileLength);
            uint32_t logitsCopyTileTail = vocabSize_ - (logitsCopyTileTimes - 1) * cmpSelTileLength;
            for (uint32_t logitsCopyTileIdx = 0; logitsCopyTileIdx < logitsCopyTileTimes; logitsCopyTileIdx++) {
                uint32_t logitsCopyLen = (logitsCopyTileIdx == logitsCopyTileTimes - 1) ? logitsCopyTileTail :
                                                                                          cmpSelTileLength;
                int64_t logitsCopyGmOffset = batchGmBase + static_cast<int64_t>(logitsCopyTileIdx) * cmpSelTileLength;
                SafeCopyIn(cmpLogitsLocal, mGmLogits_[logitsCopyGmOffset], logitsCopyLen);
                MTE2ToVSync();
                VToMTE3Sync();
                SafeCopyOut(mGmOut_[logitsCopyGmOffset], cmpLogitsLocal, logitsCopyLen);
                MTE3ToMTE2Sync();
            }
            continue;
        }

        uint32_t kthStart = FindKthStartIdx(sortedBase, kValue);
        uint32_t actualTopKCount = vocabSize_ - kthStart;

        if (actualTopKCount <= scatterLength) {
            FillNegInfAndScatterTopK(batchGmBase, sortedBase, kthStart);
            continue;
        }
        int64_t kthIdx = sortedBase + kthStart;
        inputT kthValRaw = mGmSortedValue_[kthIdx].GetValue(0);

        CompareSelectOnLogits(batchGmBase, kthValRaw, true);
        CopyOutLast(batchGmBase, sortedBase);
    }
}

// --- ProcessTopPOpt: main entry for optimized TopP-only (with logits) ---
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPOpt<inputT, calT, outputT>::ProcessTopPOpt()
{
    for (uint32_t loopBatch = 0; loopBatch < loopBatch_; loopBatch++) {
        int64_t sortedBase = (batchOffset_ + loopBatch) * vocabSize_;
        int64_t batchGmBase = sortedBase;
        const auto pGmIndex = batchOffset_ + loopBatch;

        if constexpr (IsSameType<inputT, float>::value) {
            pValue = 1.0f - mGmP_[pGmIndex].GetValue(0);
        } else if constexpr (IsSameType<inputT, half>::value) {
            pValue = 1.0f - static_cast<float>(mGmP_[pGmIndex].GetValue(0));
        } else {
            pValue = 1.0f - ToFloat(mGmP_[pGmIndex].GetValue(0));
        }

        GetMaxValue(sortedBase);

        if (*reinterpret_cast<int32_t*>(&negMaxValue) == FLOAT32_NEG_INF ||
            *reinterpret_cast<int32_t*>(&negMaxValue) == FLOAT32_INF) {
            uint32_t topPLogitsCopyTileTimes = CeilDiv(vocabSize_, cmpSelTileLength);
            uint32_t topPLogitsCopyTileTail = vocabSize_ - (topPLogitsCopyTileTimes - 1) * cmpSelTileLength;
            for (uint32_t topPLogitsCopyTileIdx = 0; topPLogitsCopyTileIdx < topPLogitsCopyTileTimes;
                 topPLogitsCopyTileIdx++) {
                uint32_t topPLogitsCopyLen = (topPLogitsCopyTileIdx == topPLogitsCopyTileTimes - 1) ?
                                                 topPLogitsCopyTileTail :
                                                 cmpSelTileLength;
                int64_t topPLogitsCopyGmOffset = batchGmBase +
                                                 static_cast<int64_t>(topPLogitsCopyTileIdx) * cmpSelTileLength;
                SafeCopyIn(cmpLogitsLocal, mGmLogits_[topPLogitsCopyGmOffset], topPLogitsCopyLen);
                MTE2ToVSync();
                VToMTE3Sync();
                SafeCopyOut(mGmOut_[topPLogitsCopyGmOffset], cmpLogitsLocal, topPLogitsCopyLen);
                MTE3ToMTE2Sync();
            }
            continue;
        }

        ComputeSoftmaxSum(sortedBase, 0, 0);

        uint32_t firstIdx = vocabSize_;
        ProcessCumsumAndFind(sortedBase, batchGmBase, 0, firstIdx);
        if (firstIdx >= vocabSize_)
            firstIdx = vocabSize_ - 1;

        inputT thresholdVal = mGmSortedValue_[sortedBase + firstIdx].GetValue(0);
        CompareSelectOnLogits(batchGmBase, thresholdVal, true);

        float threshFloat;
        if constexpr (IsSameType<inputT, float>::value) {
            threshFloat = thresholdVal;
        } else if constexpr (IsSameType<inputT, half>::value) {
            threshFloat = static_cast<float>(thresholdVal);
        } else {
            threshFloat = ToFloat(thresholdVal);
        }

        uint32_t excludeCount = 0;
        if (firstIdx > 0) {
            uint32_t searchPos = firstIdx;
            bool exDone = false;
            while (searchPos > 0 && !exDone) {
                uint32_t chunkLen = (searchPos > scatterLength) ? scatterLength : searchPos;
                uint32_t chunkStart = searchPos - chunkLen;
                SafeCopyIn(scatterValLocal, mGmSortedValue_[sortedBase + chunkStart], chunkLen);
                MTE2ToSSync();
                for (int32_t i = static_cast<int32_t>(chunkLen) - 1; i >= 0; i--) {
                    float val;
                    if constexpr (IsSameType<inputT, float>::value) {
                        val = scatterValLocal.GetValue(i);
                    } else if constexpr (IsSameType<inputT, half>::value) {
                        val = static_cast<float>(scatterValLocal.GetValue(i));
                    } else {
                        val = ToFloat(scatterValLocal.GetValue(i));
                    }
                    if (val != threshFloat) {
                        exDone = true;
                        break;
                    }
                    excludeCount++;
                }
                searchPos = chunkStart;
            }
        }

        if (excludeCount > 0) {
            uint32_t excludeStart = firstIdx - excludeCount;
            uint32_t writeTimes = CeilDiv(excludeCount, scatterLength);
            uint32_t writeTail = excludeCount - (writeTimes - 1) * scatterLength;
            for (uint32_t wi = 0; wi < writeTimes; wi++) {
                uint32_t curLen = (wi == writeTimes - 1) ? writeTail : scatterLength;
                int64_t idxGmOff = sortedBase + excludeStart + wi * scatterLength;
                SafeCopyIn(scatterIdxLocal, mGmSortedIndices_[idxGmOff], curLen);
                MTE2ToSSync();
                if constexpr (IsSameType<inputT, float>::value) {
                    scatterAlignedLocal.template ReinterpretCast<int32_t>().SetValue(0, FLOAT32_NEG_INF);
                } else if constexpr (IsSameType<inputT, half>::value) {
                    scatterAlignedLocal.template ReinterpretCast<uint16_t>().SetValue(0, FLOAT16_NEG_INF);
                } else {
                    scatterAlignedLocal.template ReinterpretCast<uint16_t>().SetValue(0, BF16_NEG_INF);
                }
                SToMTE3Sync();
                for (uint32_t ei = 0; ei < curLen; ei++) {
                    int32_t lineIndex = scatterIdxLocal.GetValue(ei);
                    SafeCopyOut(mGmOut_[batchGmBase + lineIndex],
                                scatterAlignedLocal.template ReinterpretCast<outputT>(), 1);
                }
                MTE3ToSSync();
            }
        }
        CopyOutLast(batchGmBase, sortedBase);
    }
}

} // namespace ApplyTopKTopPOptOp
#endif
