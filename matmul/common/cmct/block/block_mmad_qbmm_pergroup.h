/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file block_mmad_qbmm_pergroup.h
 * \brief Shared per-group Cube pipeline with INT8/INT4 weight policies.
 */
#pragma once

#include "../utils/common_utils.h"
#include "../policy/dispatch_policy.h"
#include "../utils/quant_batch_matmul_constant.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <class WeightPolicy, bool TransB, uint64_t LoadMode>
class BlockMmadX2PerGroup {
public:
    static constexpr uint64_t LOAD_MODE = LoadMode;
    static constexpr uint8_t L0C_BUFFER_COUNT = 2U;
    static constexpr uint8_t FULL_LOAD_BUFFER_COUNT = 4U;
    static constexpr uint64_t FULL_LOAD_SCALE_COPY_BYTES = 32UL * 1024UL;
    static constexpr uint64_t MTE2_BASEK_COUNT = 4UL;
    using AType = int8_t;
    using CType = int32_t;
    using DispatchPolicy = MatmulWithScale<>;
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    static constexpr bool transA = false;
    static constexpr bool transB = TransB;
    using BType = typename WeightPolicy::BType;
    using WeightTraits = WeightPolicy;

    struct TileParams {
        uint64_t batchA;
        uint64_t batchB;
        uint64_t batchC;
        uint64_t mOffset;
        uint64_t nOffset;
        uint32_t actualM;
        uint32_t actualN;
    };

    struct GroupResultParams {
        static constexpr uint8_t UB_RING_DEPTH = 2U;
        static constexpr uint8_t GROUPS_PER_BATCH = 2U;
        static constexpr uint8_t PAIR_RESULT_SLOTS = UB_RING_DEPTH * GROUPS_PER_BATCH;
        // Scale uses IDs 0/1; IDs 6/7 are reserved by the SDK.
        static constexpr uint16_t VEC2CUBE_FLAG = 2U;
        static constexpr uint16_t CUBE2VEC_FLAG = 4U;
        static_assert(UB_RING_DEPTH > 0U && VEC2CUBE_FLAG >= 2U && VEC2CUBE_FLAG + UB_RING_DEPTH <= CUBE2VEC_FLAG &&
                          CUBE2VEC_FLAG + UB_RING_DEPTH <= 6U,
                      "fusion ring events must avoid scale IDs 0/1 and SDK IDs 6/7");

        __aicore__ inline static uint8_t GetCreditSlot(uint64_t group, bool pairReduction)
        {
            const uint64_t batch = pairReduction ? group / GROUPS_PER_BATCH : group;
            return static_cast<uint8_t>(batch % UB_RING_DEPTH);
        }
    };

    struct L1Params {
        uint32_t kAL1{0}, kBL1{0};
        uint64_t groupSizeK{0};
    };

    struct L1Layout {
        uint64_t alignedM, alignedN;
        uint64_t aElements, bElements;
        uint64_t bOffset, scaleOffset, bufferStride, scaleStride;
        uint64_t scaleRowsPerBuffer;
    };

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR scaleGmAddr{nullptr};
        uint64_t m{0}, n{0}, k{0};
        uint32_t baseM{0}, baseN{0}, baseK{0};
        uint8_t dbL0C{1};
        bool broadcastReuse{false};
        bool pairReduction{false};
        L1Params l1Params;
        L1Layout l1Layout{};
    };

    __aicore__ inline static uint64_t GetFullLoadScaleRows(uint64_t dataBytes, uint64_t rowBytes, uint64_t groupCount,
                                                           uint64_t l1Bytes)
    {
        const uint64_t availableBytes = l1Bytes - Cmct::Gemm::Min<uint64_t>(dataBytes, l1Bytes);
        const uint64_t targetRows = Cmct::Gemm::Max<uint64_t>(1UL, FULL_LOAD_SCALE_COPY_BYTES / rowBytes);
        return Cmct::Gemm::Min<uint64_t>(groupCount,
                                         Cmct::Gemm::Min<uint64_t>(targetRows, availableBytes / (2UL * rowBytes)));
    }

    __aicore__ inline static L1Layout CalculateL1Layout(const Params& params, uint64_t groupCount,
                                                        bool broadcastReuse = false)
    {
        constexpr uint64_t cubeBlock = 16UL;
        constexpr uint64_t dataBlock = 32UL;
        const bool streaming = LoadMode == NONE_FULL_LOAD_MODE && !broadcastReuse;
        const bool bResident = LoadMode == B_FULL_LOAD_MODE;
        L1Layout result{};
        result.alignedM = Cmct::Gemm::Align(params.baseM, cubeBlock);
        result.alignedN = Cmct::Gemm::Align(params.baseN, cubeBlock);
        result.aElements = result.alignedM *
                           Cmct::Gemm::Align((streaming || bResident) ? params.l1Params.kAL1 : params.k,
                                             bResident ? dataBlock : AscendC::AuxGetC0Size<int8_t>());
        result.bElements = Cmct::Gemm::Align(params.baseN, transB ? cubeBlock : WeightPolicy::C0) *
                           Cmct::Gemm::Align(
                               (streaming || LoadMode == A_FULL_LOAD_MODE) ? params.l1Params.kBL1 : params.k,
                               WeightPolicy::C0);
        const uint64_t aSlotBytes = Cmct::Gemm::Align(result.aElements * sizeof(int8_t), dataBlock);
        if constexpr (LoadMode == A_FULL_LOAD_MODE) {
            // One full-K A panel, four B windows, and two independently protected scale batches.
            result.bOffset = aSlotBytes;
            result.bufferStride = Cmct::Gemm::Align(
                Cmct::Gemm::CeilDiv(result.bElements, WeightPolicy::ELEMENTS_PER_BYTE), dataBlock);
            result.scaleOffset = result.bOffset + result.bufferStride * FULL_LOAD_BUFFER_COUNT;
            const uint64_t scaleRowBytes = result.alignedN * sizeof(uint64_t);
            result.scaleRowsPerBuffer = GetFullLoadScaleRows(result.scaleOffset, scaleRowBytes, groupCount,
                                                             AscendC::TOTAL_L1_SIZE);
            result.scaleStride = result.scaleRowsPerBuffer * scaleRowBytes;
            return result;
        }
        if constexpr (LoadMode == B_FULL_LOAD_MODE) {
            // Four A windows, one full-K B panel, and two independently protected scale batches.
            result.bOffset = aSlotBytes * FULL_LOAD_BUFFER_COUNT;
            result.scaleOffset = Cmct::Gemm::Align(
                result.bOffset + Cmct::Gemm::CeilDiv(result.bElements, WeightPolicy::ELEMENTS_PER_BYTE), dataBlock);
            result.bufferStride = aSlotBytes;
            const uint64_t scaleRowBytes = result.alignedN * sizeof(uint64_t);
            result.scaleRowsPerBuffer = GetFullLoadScaleRows(result.scaleOffset, scaleRowBytes, groupCount,
                                                             AscendC::TOTAL_L1_SIZE);
            result.scaleStride = result.scaleRowsPerBuffer * scaleRowBytes;
            return result;
        }
        result.bOffset = aSlotBytes;
        result.scaleOffset = Cmct::Gemm::Align(
            result.bOffset + Cmct::Gemm::CeilDiv(result.bElements, WeightPolicy::ELEMENTS_PER_BYTE), dataBlock);
        const uint64_t scaleRows = streaming ? 1UL : groupCount;
        result.bufferStride = Cmct::Gemm::Align(result.scaleOffset + scaleRows * result.alignedN * sizeof(uint64_t),
                                                dataBlock);
        return result;
    }

    __aicore__ inline BlockMmadX2PerGroup() = default;
    __aicore__ inline ~BlockMmadX2PerGroup() = default;

    __aicore__ inline void Init(const Params& params, AscendC::TPipe* pipe)
    {
        params_ = &params;
        pipe_ = pipe;
        l0PingPong_ = 0U;
        isBFullLoad_ = false;
        isAFullLoad_ = false;
        residentInputLoaded_ = false;
        scaleBufferPhase_ = 0U;
        broadcastReuse_ = false;
        broadcastACached_ = false;
        broadcastBCached_ = false;
        broadcastACacheHit_ = false;
        broadcastBCacheHit_ = false;
        InitEvents();
        AscendC::SetMMLayoutTransform(true);
        InitGlobalTensor(params);
        InitParams();
        InitL1Buffer();
        InitSegmentEvents();
        l0cBufferCount_ = params_->dbL0C == L0C_BUFFER_COUNT ? L0C_BUFFER_COUNT : 1U;
    }

    // MMAD-only entry. The kernel schedules FixGroupResult separately.
    __aicore__ inline void operator()(const TileParams& tile, uint64_t groupIndex, uint32_t groupK)
    {
        if (isAFullLoad_ || isBFullLoad_) {
            IterateFullLoad(tile, groupIndex, groupK);
        } else {
            IterateGroupL1(tile, groupIndex, groupK);
        }
    }

    __aicore__ inline void FixGroupResult(const TileParams& tile, uint64_t groupIndex,
                                          const AscendC::LocalTensor<half>& groupOutput)
    {
        const uint8_t cBuffer = static_cast<uint8_t>(groupIndex % l0cBufferCount_);
        const bool residentScales = broadcastReuse_;
        uint8_t scaleBuffer = residentScales ? RESIDENT_BUFFER_ID : static_cast<uint8_t>(groupIndex & 1UL);
        uint64_t scaleRow = residentScales ? groupIndex : 0UL;
        bool releaseScale = !residentScales || groupIndex + 1UL == groupCount_;
        if constexpr (LoadMode == A_FULL_LOAD_MODE || LoadMode == B_FULL_LOAD_MODE) {
            scaleBuffer = static_cast<uint8_t>((scaleBufferPhase_ + groupIndex / scaleRowsPerBuffer_) & 1UL);
            scaleRow = groupIndex % scaleRowsPerBuffer_;
            releaseScale = scaleRow + 1UL == scaleRowsPerBuffer_ || groupIndex + 1UL == groupCount_;
        }
        CopyGroupResult(tile, groupIndex, scaleBuffer, cBuffer, groupOutput, scaleRow, releaseScale);
        if constexpr (LoadMode == A_FULL_LOAD_MODE || LoadMode == B_FULL_LOAD_MODE) {
            if (groupIndex + 1UL == groupCount_) {
                // Keep scale ping/pong continuous across tiles, including one-batch tiles.
                scaleBufferPhase_ = static_cast<uint8_t>(scaleBuffer ^ 1U);
            }
        }
        if (broadcastReuse_ && groupIndex + 1UL == groupCount_) {
            // Cache identities become valid after the final FIX consumes the resident scales.
            broadcastACached_ = true;
            broadcastBCached_ = true;
            cachedBatchA_ = tile.batchA;
            cachedMOffset_ = tile.mOffset;
            cachedActualM_ = tile.actualM;
            cachedBatchB_ = tile.batchB;
            cachedNOffset_ = tile.nOffset;
            cachedActualN_ = tile.actualN;
        }
    }

    __aicore__ inline void End()
    {
        for (uint8_t i = 0; i < INPUT_BUFFER_COUNT; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(inputEvent_[i]);
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(inputEvent_[i]);
        }
        for (uint8_t i = 0; i < DOUBLE_BUFFER; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Event_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cEvent_[i]);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(scaleEvent_[i]);
            pipe_->ReleaseEventID<AscendC::HardEvent::M_MTE1>(l0Event_[i]);
            pipe_->ReleaseEventID<AscendC::HardEvent::FIX_M>(l0cEvent_[i]);
            pipe_->ReleaseEventID<AscendC::HardEvent::FIX_MTE2>(scaleEvent_[i]);
        }
        if (broadcastReuse_) {
            for (uint8_t i = 0; i < DOUBLE_BUFFER; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(segmentEvent_[i]);
                pipe_->ReleaseEventID<AscendC::HardEvent::MTE1_MTE2>(segmentEvent_[i]);
            }
        }
        AscendC::SetMMLayoutTransform(false);
    }

private:
    static constexpr AscendC::FixpipeConfig ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

    __aicore__ inline void InitEvents()
    {
        for (uint8_t i = 0; i < INPUT_BUFFER_COUNT; ++i) {
            // FetchEventID does not reserve IDs; each ping/pong slot needs its own event.
            inputEvent_[i] = pipe_->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(inputEvent_[i]);
        }
        for (uint8_t i = 0; i < DOUBLE_BUFFER; ++i) {
            l0Event_[i] = pipe_->AllocEventID<AscendC::HardEvent::M_MTE1>();
            l0cEvent_[i] = pipe_->AllocEventID<AscendC::HardEvent::FIX_M>();
            scaleEvent_[i] = pipe_->AllocEventID<AscendC::HardEvent::FIX_MTE2>();
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Event_[i]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cEvent_[i]);
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(scaleEvent_[i]);
        }
    }

    __aicore__ inline void InitGlobalTensor(const Params& params)
    {
        x1_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(params.aGmAddr));
        scale_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(params.scaleGmAddr));
        x2_.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(params.bGmAddr));
    }

    __aicore__ inline void InitParams()
    {
        const auto& mm = *params_;
        // Iterate over the common K span; the L1 layout reserves each operand's full window.
        kL1_ = Cmct::Gemm::Max<uint64_t>(static_cast<uint64_t>(mm.baseK),
                                         Cmct::Gemm::Min<uint64_t>(static_cast<uint64_t>(mm.l1Params.kAL1),
                                                                   static_cast<uint64_t>(mm.l1Params.kBL1)));
        groupCount_ = Cmct::Gemm::CeilDiv(mm.k, params_->l1Params.groupSizeK);
        broadcastSegmentK_ = static_cast<uint64_t>(mm.baseK) * MTE2_BASEK_COUNT;
    }

    __aicore__ inline void InitL1Buffer()
    {
        const auto& layout = params_->l1Layout;
        alignedL1M_ = layout.alignedM;
        alignedN_ = layout.alignedN;
        bL1ByteOffset_ = layout.bOffset;
        scaleL1ByteOffset_ = layout.scaleOffset;
        l1BufferBytes_ = layout.bufferStride;
        scaleL1BufferBytes_ = layout.scaleStride;
        scaleRowsPerBuffer_ = layout.scaleRowsPerBuffer;
        isAFullLoad_ = LoadMode == A_FULL_LOAD_MODE;
        isBFullLoad_ = LoadMode == B_FULL_LOAD_MODE;
        broadcastReuse_ = params_->broadcastReuse;
    }

    __aicore__ inline void InitSegmentEvents()
    {
        if (broadcastReuse_) {
            for (uint8_t i = 0; i < DOUBLE_BUFFER; ++i) {
                segmentEvent_[i] = pipe_->AllocEventID<AscendC::HardEvent::MTE1_MTE2>();
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(segmentEvent_[i]);
            }
        }
    }

    __aicore__ inline void PrefetchInputWindow(const TileParams& tile, uint8_t bufferId)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(inputEvent_[bufferId]);
        const uint32_t currentK = static_cast<uint32_t>(
            Cmct::Gemm::Min<uint64_t>(kL1_, params_->k - nextInputWindowK_));
        if constexpr (LoadMode == A_FULL_LOAD_MODE) {
            CopyX2Nz(GetB1(bufferId), tile, nextInputWindowK_, currentK);
        } else {
            CopyX1NdToNz(tile, nextInputWindowK_, currentK, bufferId);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(inputEvent_[bufferId]);
        nextInputWindowK_ += currentK;
    }

    __aicore__ inline void IterateFullLoad(const TileParams& tile, uint64_t groupIndex, uint32_t groupK)
    {
        if (groupIndex == 0UL) {
            // Full-load scheduling keeps the resident M or N panel fixed on each core.
            if (!residentInputLoaded_) {
                if constexpr (LoadMode == A_FULL_LOAD_MODE) {
                    CopyX1NdToNz(tile, 0UL, params_->k, RESIDENT_BUFFER_ID);
                } else {
                    CopyX2Nz(GetB1(RESIDENT_BUFFER_ID), tile, 0UL, params_->k, 0UL, params_->k);
                }
                residentInputLoaded_ = true;
            }
            nextInputWindowK_ = 0UL;
            inputWindowIndex_ = 0UL;
            for (uint8_t slot = 0U; slot < FULL_LOAD_BUFFER_COUNT && nextInputWindowK_ < params_->k; ++slot) {
                PrefetchInputWindow(tile, slot);
            }
        }
        if (groupIndex % scaleRowsPerBuffer_ == 0UL) {
            const uint8_t scaleBuffer = static_cast<uint8_t>((scaleBufferPhase_ + groupIndex / scaleRowsPerBuffer_) &
                                                             1UL);
            const uint64_t rows = Cmct::Gemm::Min<uint64_t>(scaleRowsPerBuffer_, groupCount_ - groupIndex);
            CopyScales<false>(tile, groupIndex, scaleBuffer, rows);
        }
        const uint8_t cBufferId = static_cast<uint8_t>(groupIndex % l0cBufferCount_);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cEvent_[cBufferId]);
        const uint64_t groupKOffset = groupIndex * params_->l1Params.groupSizeK;
        uint64_t processedK = 0UL;
        while (processedK < groupK) {
            const uint8_t bufferId = static_cast<uint8_t>(inputWindowIndex_ % FULL_LOAD_BUFFER_COUNT);
            const uint32_t windowK = static_cast<uint32_t>(
                Cmct::Gemm::Min<uint64_t>(kL1_, static_cast<uint64_t>(groupK) - processedK));
            WaitInputsReady(bufferId);
            for (uint64_t offset = 0UL; offset < windowK;) {
                const uint32_t currentK = static_cast<uint32_t>(
                    Cmct::Gemm::Min<uint64_t>(params_->baseK, static_cast<uint64_t>(windowK) - offset));
                const bool lastK = processedK + offset + currentK == groupK;
                LoadL0AndMmad(tile, currentK, processedK + offset == 0UL, lastK, bufferId, offset, windowK, cBufferId,
                              groupKOffset + processedK + offset);
                offset += currentK;
            }
            // MTE1 has consumed this window before MTE2 can refill it four windows ahead.
            ReleaseInputs(bufferId);
            if (nextInputWindowK_ < params_->k) {
                PrefetchInputWindow(tile, bufferId);
            }
            ++inputWindowIndex_;
            processedK += windowK;
        }
    }

    __aicore__ inline void IterateGroupL1(const TileParams& tile, uint64_t groupIndex, uint32_t groupK)
    {
        // Broadcast caching retains its existing layout within the non-full-load mode.
        if (broadcastReuse_) {
            IterateBroadcastReuse(tile, groupIndex, groupK);
            return;
        }
        const uint64_t groupKOffset = groupIndex * params_->l1Params.groupSizeK;
        const uint8_t groupBufferId = static_cast<uint8_t>(groupIndex & 1UL);
        const uint8_t cBufferId = static_cast<uint8_t>(groupIndex % l0cBufferCount_);
        // When a full group fits in L1, the previous group prefetches it; group zero starts here.
        const bool enableCrossGroupPrefetch = static_cast<uint64_t>(params_->l1Params.groupSizeK) <= kL1_;
        CopyScales<false>(tile, groupIndex, groupBufferId);
        uint64_t l1KOffset = 0;
        uint32_t currentL1K = static_cast<uint32_t>(Cmct::Gemm::Min<uint64_t>(kL1_, static_cast<uint64_t>(groupK)));
        uint8_t l1BufferId = enableCrossGroupPrefetch ? groupBufferId : 0U;
        if (!enableCrossGroupPrefetch || groupIndex == 0UL) {
            BeginCopyInputs(tile, groupKOffset, currentL1K, l1BufferId);
        }

        // Start GM->L1 before the L0C wait to overlap MTE2 with the preceding FixPipe.
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cEvent_[cBufferId]);
        bool firstK = true;
        while (l1KOffset < groupK) {
            const uint64_t nextL1KOffset = l1KOffset + currentL1K;
            const uint8_t nextL1BufferId = static_cast<uint8_t>(l1BufferId ^ 1U);
            uint32_t nextL1K = 0;
            if (nextL1KOffset < groupK) {
                nextL1K = static_cast<uint32_t>(
                    Cmct::Gemm::Min<uint64_t>(kL1_, static_cast<uint64_t>(groupK) - nextL1KOffset));
                BeginCopyInputs(tile, groupKOffset + nextL1KOffset, nextL1K, nextL1BufferId);
            } else if (enableCrossGroupPrefetch) {
                const uint64_t nextGroupKOffset = groupKOffset + params_->l1Params.groupSizeK;
                if (nextGroupKOffset < params_->k) {
                    const uint32_t nextGroupK = static_cast<uint32_t>(
                        Cmct::Gemm::Min<uint64_t>(static_cast<uint64_t>(params_->l1Params.groupSizeK),
                                                  static_cast<uint64_t>(params_->k) - nextGroupKOffset));
                    BeginCopyInputs(tile, nextGroupKOffset, nextGroupK, nextL1BufferId);
                }
            }

            WaitInputsReady(l1BufferId);
            uint64_t processedL1K = 0;
            while (processedL1K < currentL1K) {
                const uint32_t currentK = static_cast<uint32_t>(Cmct::Gemm::Min<uint64_t>(
                    static_cast<uint64_t>(params_->baseK), static_cast<uint64_t>(currentL1K) - processedL1K));
                const bool lastK = l1KOffset + processedL1K + currentK == groupK;
                LoadL0AndMmad(tile, currentK, firstK, lastK, l1BufferId, processedL1K, currentL1K, cBufferId,
                              groupKOffset + l1KOffset + processedL1K);
                firstK = false;
                processedL1K += currentK;
            }
            ReleaseInputs(l1BufferId);
            l1KOffset = nextL1KOffset;
            currentL1K = nextL1K;
            l1BufferId = nextL1BufferId;
        }
    }

    __aicore__ inline void IterateBroadcastReuse(const TileParams& tile, uint64_t groupIndex, uint32_t groupK)
    {
        const auto& mm = *params_;
        if (groupIndex == 0UL) {
            broadcastBufferId_ = RESIDENT_BUFFER_ID;
            broadcastACacheHit_ = broadcastACached_ && cachedBatchA_ == tile.batchA && cachedMOffset_ == tile.mOffset &&
                                  cachedActualM_ == tile.actualM;
            broadcastBCacheHit_ = broadcastBCached_ && cachedBatchB_ == tile.batchB && cachedNOffset_ == tile.nOffset &&
                                  cachedActualN_ == tile.actualN;
            // inputEvent_ protects the full L1 window; segmentEvent_ protects readiness-phase reuse.
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(inputEvent_[broadcastBufferId_]);
            broadcastReadyK_ = 0UL;
            broadcastSegmentIndex_ = 0UL;
            const uint32_t firstK = static_cast<uint32_t>(
                Cmct::Gemm::Min<uint64_t>(broadcastSegmentK_, static_cast<uint64_t>(mm.k)));
            // Publish A/B readiness before the scale DMA so Cube can start consuming the segment.
            BeginCopyBroadcastSegment(tile, 0UL, firstK, 0U);
            if (broadcastBCacheHit_) {
                // Even without DMA, consume the FIX release credit exactly once.
                AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(scaleEvent_[broadcastBufferId_]);
            } else {
                CopyScales<true>(tile, 0UL, broadcastBufferId_);
            }
        }

        const uint8_t cBufferId = static_cast<uint8_t>(groupIndex % l0cBufferCount_);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cEvent_[cBufferId]);
        const uint64_t groupKOffset = groupIndex * params_->l1Params.groupSizeK;
        uint64_t processedK = 0;
        while (processedK < groupK) {
            const uint32_t currentK = static_cast<uint32_t>(
                Cmct::Gemm::Min<uint64_t>(static_cast<uint64_t>(mm.baseK), static_cast<uint64_t>(groupK) - processedK));
            const bool lastK = processedK + currentK == groupK;
            WaitBroadcastInputsReady(tile, groupKOffset + processedK + currentK);
            // Both the K offset and the non-transposed B source stride refer
            // to the complete resident window, not the current quant group.
            LoadL0AndMmad(tile, currentK, processedK == 0UL, lastK, broadcastBufferId_, groupKOffset + processedK, mm.k,
                          cBufferId);
            processedK += currentK;
        }
        const bool lastGroup = groupIndex + 1UL == groupCount_;
        if (lastGroup) {
            ReleaseInputs(broadcastBufferId_);
        }
    }

    __aicore__ inline void BeginCopyBroadcastSegment(const TileParams& tile, uint64_t kOffset, uint32_t currentK,
                                                     uint8_t readyPhase)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(segmentEvent_[readyPhase]);
        if (!broadcastACacheHit_) {
            CopyX1NdToNz(tile, kOffset, currentK, broadcastBufferId_, kOffset);
        }
        if (!broadcastBCacheHit_) {
            CopyX2Nz(GetB1(broadcastBufferId_), tile, kOffset, currentK, kOffset, params_->k);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(segmentEvent_[readyPhase]);
    }

    __aicore__ inline void WaitBroadcastInputsReady(const TileParams& tile, uint64_t requiredKEnd)
    {
        if (requiredKEnd <= broadcastReadyK_) {
            return;
        }
        const uint64_t k = params_->k;
        const uint64_t currentEnd = Cmct::Gemm::Min<uint64_t>(broadcastReadyK_ + broadcastSegmentK_, k);
        const uint8_t phase = static_cast<uint8_t>(broadcastSegmentIndex_ & 1UL);
        if (currentEnd < k) {
            const uint32_t nextK = static_cast<uint32_t>(Cmct::Gemm::Min<uint64_t>(broadcastSegmentK_, k - currentEnd));
            // Prefetch before waiting so MTE2 overlaps consumption of the current segment.
            BeginCopyBroadcastSegment(tile, currentEnd, nextK, static_cast<uint8_t>(phase ^ 1U));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(segmentEvent_[phase]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(segmentEvent_[phase]);
        broadcastReadyK_ = currentEnd;
        ++broadcastSegmentIndex_;
        // G is a multiple of baseK; therefore one MMAD cannot straddle two
        // segments. A segment may span groups, and remains ready across them.
    }

    __aicore__ inline AscendC::LocalTensor<int8_t> GetA1(uint8_t bufferId)
    {
        if (isAFullLoad_) {
            return l1_.template ReinterpretCast<int8_t>();
        }
        return l1_.template ReinterpretCast<int8_t>()[bufferId * l1BufferBytes_];
    }

    __aicore__ inline AscendC::LocalTensor<BType> GetB1(uint8_t bufferId)
    {
        bufferId = isBFullLoad_ ? RESIDENT_BUFFER_ID : bufferId;
        return l1_.template ReinterpretCast<BType>()[(bufferId * l1BufferBytes_ + bL1ByteOffset_) *
                                                     WeightPolicy::ELEMENTS_PER_BYTE];
    }

    __aicore__ inline AscendC::LocalTensor<uint64_t> GetScaleL1(uint8_t bufferId)
    {
        if (isAFullLoad_ || isBFullLoad_) {
            return l1_.template ReinterpretCast<uint64_t>()[(scaleL1ByteOffset_ + bufferId * scaleL1BufferBytes_) /
                                                            sizeof(uint64_t)];
        }
        return l1_
            .template ReinterpretCast<uint64_t>()[(bufferId * l1BufferBytes_ + scaleL1ByteOffset_) / sizeof(uint64_t)];
    }

    __aicore__ inline void BeginCopyInputs(const TileParams& tile, uint64_t kOffset, uint32_t currentK,
                                           uint8_t bufferId)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(inputEvent_[bufferId]);
        if (!broadcastACacheHit_) {
            CopyX1NdToNz(tile, kOffset, currentK, bufferId);
        }
        if (!broadcastBCacheHit_) {
            CopyX2Nz(GetB1(bufferId), tile, kOffset, currentK);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(inputEvent_[bufferId]);
    }

    __aicore__ inline void WaitInputsReady(uint8_t bufferId)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(inputEvent_[bufferId]);
    }

    __aicore__ inline void ReleaseInputs(uint8_t bufferId)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(inputEvent_[bufferId]);
    }

    __aicore__ inline static uint64_t WeightBytes(uint64_t elementCount)
    {
        return Cmct::Gemm::CeilDiv(elementCount, WeightPolicy::ELEMENTS_PER_BYTE);
    }

    // DataCopyPad blockLen/srcStride use bytes; dstStride uses 32-byte blocks.
    __aicore__ inline void CopyX2Nz(const AscendC::LocalTensor<BType>& dst, const TileParams& tile, uint64_t kOffset,
                                    uint32_t currentK, uint64_t l1KOffset = 0UL, uint64_t l1K = 0UL) const
    {
        constexpr uint64_t c0 = WeightPolicy::C0;
        static_assert(c0 == AscendC::AuxGetC0Size<BType>(), "Weight policy must match SDK NZ C0");
        const uint64_t kAlign16 = Cmct::Gemm::Align(params_->k, BLOCK_CUBE);
        const uint64_t nAlign16 = Cmct::Gemm::Align(params_->n, BLOCK_CUBE);
        const uint64_t batchElements = transB ? Cmct::Gemm::Align(params_->k, c0) * nAlign16 :
                                                Cmct::Gemm::Align(params_->n, c0) * kAlign16;
        const uint64_t batchBytes = WeightBytes(batchElements);
        uint64_t elementOffset = 0;
        uint64_t dstElementOffset = 0;
        AscendC::DataCopyExtParams params{};
        AscendC::DataCopyPadExtParams<uint8_t> padParams{};
        uint64_t srcStrideBytes = 0UL;

        if constexpr (transB) {
            // Logical [N, K], physical NZ (k1, nAlign16, k0=c0).
            const uint64_t copyN = Cmct::Gemm::Align(tile.actualN, BLOCK_CUBE);
            params.blockCount = static_cast<uint32_t>(Cmct::Gemm::CeilDiv(currentK, c0));
            params.blockLen = static_cast<uint32_t>(WeightBytes(copyN * c0));
            srcStrideBytes = WeightBytes((nAlign16 - copyN) * c0);
            params.dstStride = 0;
            elementOffset = (kOffset / c0) * nAlign16 * c0 + tile.nOffset * c0;
            dstElementOffset = (l1KOffset / c0) * copyN * c0;
        } else {
            // Logical [K, N], physical NZ (n1, kAlign16, n0=c0).
            const uint64_t copyK = Cmct::Gemm::Align(currentK, BLOCK_CUBE);
            // A transposed load consumes C0/16 K fractals: 2 for INT8, 4 for INT4.
            const uint64_t l1KPad = Cmct::Gemm::Align(l1K == 0UL ? static_cast<uint64_t>(currentK) : l1K,
                                                      WeightPolicy::TRANSPOSE_STEP * BLOCK_CUBE);
            params.blockCount = static_cast<uint32_t>(Cmct::Gemm::Align(tile.actualN, c0) / c0);
            params.blockLen = static_cast<uint32_t>(WeightBytes(copyK * c0));
            srcStrideBytes = WeightBytes((kAlign16 - copyK) * c0);
            params.dstStride = static_cast<uint32_t>(WeightBytes((l1KPad - copyK) * c0) / DATA_BLOCK_BYTES);
            elementOffset = (tile.nOffset / c0) * kAlign16 * c0 + kOffset * c0;
            // Resident segments retain the full-K n1 slab pitch to avoid overwriting earlier K.
            dstElementOffset = l1KOffset * c0;
        }

        auto srcBytes = x2_.template ReinterpretCast<uint8_t>();
        auto dstBytes = dst.template ReinterpretCast<uint8_t>();
        const uint64_t dstOffset = WeightBytes(dstElementOffset);
        const uint64_t srcOffset = tile.batchB * batchBytes + WeightBytes(elementOffset);
        if (srcStrideBytes <= UINT32_MAX) {
            params.srcStride = static_cast<uint32_t>(srcStrideBytes);
            AscendC::DataCopyPad(dstBytes[dstOffset], srcBytes[srcOffset], params, padParams);
            return;
        }
        // DataCopyExtParams cannot represent a wider GM gap. Issue one NZ
        // block per DMA with 64-bit tensor offsets, keeping the same L1 pitch.
        const uint32_t blockCount = params.blockCount;
        const uint64_t srcPitch = params.blockLen + srcStrideBytes;
        const uint64_t dstPitch = params.blockLen + static_cast<uint64_t>(params.dstStride) * DATA_BLOCK_BYTES;
        params.blockCount = 1U;
        params.srcStride = 0U;
        for (uint32_t block = 0; block < blockCount; ++block) {
            AscendC::DataCopyPad(dstBytes[dstOffset + block * dstPitch], srcBytes[srcOffset + block * srcPitch], params,
                                 padParams);
        }
    }

    __aicore__ inline void CopyX1NdToNz(const TileParams& tile, uint64_t kOffset, uint32_t currentK, uint8_t bufferId,
                                        uint64_t l1KOffset = 0UL)
    {
        const auto& mm = *params_;
        const uint64_t gmOffset = tile.batchA * static_cast<uint64_t>(mm.m) * mm.k + tile.mOffset * mm.k + kOffset;
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = tile.actualM;
        params.dValue = currentK;
        params.srcNdMatrixStride = 1;
        params.srcDValue = mm.k;
        params.dstNzC0Stride = alignedL1M_;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 1;
        // ND2NZ uses the full-window M pitch for every baseK-aligned K segment.
        AscendC::DataCopy(GetA1(bufferId)[l1KOffset * alignedL1M_], x1_[gmOffset], params);
    }

    template <bool AllGroups>
    __aicore__ inline void CopyScales(const TileParams& tile, uint64_t groupIndex, uint8_t bufferId,
                                      uint64_t groupRows = 1UL)
    {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(scaleEvent_[bufferId]);
        const auto& mm = *params_;
        const uint64_t gmOffset = (tile.batchB * groupCount_ + groupIndex) * mm.n + tile.nOffset;
        const uint64_t rowCount = AllGroups ? groupCount_ : groupRows;
        const uint64_t fixpipeN = Cmct::Gemm::Align(tile.actualN, BLOCK_CUBE);
        const uint64_t rowStride = (AllGroups || rowCount > 1UL) ? alignedN_ : fixpipeN;
        if (fixpipeN != tile.actualN) {
            // UB NZ2ND writes complete 16-half channel groups. Clear the
            // expanded channels in every copied row at the selected L1 pitch.
            auto scaleWords = GetScaleL1(bufferId).template ReinterpretCast<uint16_t>();
            AscendC::InitConstValueParams<uint16_t> init;
            init.repeatTimes = 1;
            init.blockNum = static_cast<uint16_t>(rowCount * rowStride * sizeof(uint64_t) / DATA_BLOCK_BYTES);
            init.dstGap = 0;
            init.initValue = 0;
            AscendC::InitConstValue(scaleWords, init);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        // DMA pads to one 32-byte block; the remaining FixPipe channels are zeroed above.
        const uint64_t scalePerBlock = DATA_BLOCK_BYTES / sizeof(uint64_t);
        const uint64_t paddedN = Cmct::Gemm::Align(tile.actualN, scalePerBlock);
        AscendC::DataCopyExtParams params{};
        params.blockCount = static_cast<uint16_t>(rowCount);
        params.blockLen = static_cast<uint32_t>(tile.actualN * sizeof(uint64_t));
        if (AllGroups || rowCount > 1UL) {
            params.srcStride = static_cast<uint32_t>((mm.n - tile.actualN) * sizeof(uint64_t));
            params.dstStride = static_cast<uint32_t>((rowStride - paddedN) * sizeof(uint64_t) / DATA_BLOCK_BYTES);
        }
        AscendC::DataCopyPadExtParams<uint64_t> padParams{true, 0, static_cast<uint8_t>(paddedN - tile.actualN), 0UL};
        AscendC::DataCopyPad(GetScaleL1(bufferId), scale_[gmOffset], params, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(scaleEvent_[bufferId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(scaleEvent_[bufferId]);
    }

    __aicore__ inline void LoadL0AndMmad(const TileParams& tile, uint32_t currentK, bool firstK, bool lastK,
                                         uint8_t l1BufferId, uint64_t l1KOffset, uint32_t currentL1K, uint8_t cBufferId,
                                         uint64_t residentKOffset = 0UL)
    {
        // Every issued MMAD alternates L0 slots continuously across groups.
        const uint8_t l0BufferId = l0PingPong_;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0Event_[l0BufferId]);
        LoadA2(tile.actualM, currentK, l1BufferId, l0BufferId, isAFullLoad_ ? residentKOffset : l1KOffset);
        LoadB2(tile.actualN, currentK, l1BufferId, l0BufferId, isBFullLoad_ ? residentKOffset : l1KOffset,
               isBFullLoad_ ? params_->k : currentL1K);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0Event_[l0BufferId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0Event_[l0BufferId]);

        AscendC::MmadParams params;
        params.m = tile.actualM;
        params.n = tile.actualN;
        params.k = currentK;
        params.disableGemv = true;
        // Double-buffered L0C is protected by FIX_M/M_FIX events, not UnitFlag.
        // For a single buffer, ONLY the final MMAD of the entire group changes
        // its blocks to readable; an intermediate value of 2 leaves them writable.
        params.unitFlag = l0cBufferCount_ == DOUBLE_BUFFER ?
                              0U :
                              (lastK ? Cmct::Gemm::QuantBatchMatmul::FINAL_ACCUMULATION :
                                       Cmct::Gemm::QuantBatchMatmul::NON_FINAL_ACCUMULATION);
        params.cmatrixInitVal = firstK;
        params.cmatrixSource = false;
        AscendC::Mmad(GetC1(cBufferId), GetA2(l0BufferId), GetB2(l0BufferId), params);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0Event_[l0BufferId]);
        l0PingPong_ ^= 1U;
    }

    __aicore__ inline AscendC::LocalTensor<int8_t> GetA2(uint8_t bufferId)
    {
        return aL0_[bufferId * (AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER)];
    }

    __aicore__ inline AscendC::LocalTensor<BType> GetB2(uint8_t bufferId)
    {
        // LocalTensor offsets count logical elements; each L0B slot occupies half the bytes.
        return bL0_[bufferId * (AscendC::TOTAL_L0B_SIZE / DOUBLE_BUFFER) * WeightPolicy::ELEMENTS_PER_BYTE];
    }

    __aicore__ inline AscendC::LocalTensor<int32_t> GetC1(uint8_t bufferId)
    {
        return cL0_[bufferId * (AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER / sizeof(int32_t))];
    }

    __aicore__ inline void LoadA2(uint32_t actualM, uint32_t currentK, uint8_t l1BufferId, uint8_t l0BufferId,
                                  uint64_t l1KOffset)
    {
        constexpr uint64_t aC0 = static_cast<uint64_t>(AscendC::AuxGetC0Size<int8_t>());
        AscendC::LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = Cmct::Gemm::CeilDiv(l1KOffset, aC0);
        params.mStep = Cmct::Gemm::CeilDiv(actualM, BLOCK_CUBE);
        params.kStep = Cmct::Gemm::CeilDiv(currentK, aC0);
        params.srcStride = Cmct::Gemm::CeilDiv(alignedL1M_, BLOCK_CUBE);
        params.dstStride = params.mStep;
        params.ifTranspose = false;
        AscendC::LoadData(GetA2(l0BufferId), GetA1(l1BufferId), params);
    }

    __aicore__ inline void LoadB2(uint32_t actualN, uint32_t currentK, uint8_t l1BufferId, uint8_t l0BufferId,
                                  uint64_t l1KOffset, uint32_t currentL1K)
    {
        constexpr uint64_t bC0 = static_cast<uint64_t>(AscendC::AuxGetC0Size<BType>());
        const uint64_t n1 = Cmct::Gemm::CeilDiv(actualN, BLOCK_CUBE);
        AscendC::LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        if constexpr (transB) {
            params.kStartPosition = Cmct::Gemm::CeilDiv(l1KOffset, bC0);
            params.mStep = n1;
            params.kStep = Cmct::Gemm::CeilDiv(currentK, bC0);
            params.srcStride = params.mStep;
            params.dstStride = params.mStep;
            params.ifTranspose = false;
        } else {
            const uint64_t kBlocks = Cmct::Gemm::CeilDiv(currentK, BLOCK_CUBE);
            const bool directLoad = n1 % WeightPolicy::TRANSPOSE_STEP == 0UL;
            const uint64_t l1MStart = Cmct::Gemm::CeilDiv(l1KOffset, BLOCK_CUBE);
            params.mStartPosition = l1MStart;
            // Transpose steps follow the weight C0, including K tails.
            params.mStep = directLoad ? Cmct::Gemm::Align(kBlocks, WeightPolicy::TRANSPOSE_STEP) :
                                        WeightPolicy::TRANSPOSE_STEP;
            params.kStep = Cmct::Gemm::CeilDiv(actualN, bC0);
            params.srcStride = Cmct::Gemm::CeilDiv(
                Cmct::Gemm::Align(currentL1K, WeightPolicy::TRANSPOSE_STEP * BLOCK_CUBE), BLOCK_CUBE);
            params.dstStride = n1;
            params.ifTranspose = true;
            AscendC::LoadData(GetB2(l0BufferId), GetB1(l1BufferId), params);
            if (!directLoad) {
                AscendC::PipeBarrier<PIPE_MTE1>();
                const uint64_t loadTimes = Cmct::Gemm::CeilDiv(kBlocks, WeightPolicy::TRANSPOSE_STEP);
                for (uint64_t i = 1; i < loadTimes; ++i) {
                    params.mStartPosition = l1MStart + WeightPolicy::TRANSPOSE_STEP * i;
                    const uint64_t l0Offset = i * n1 * BLOCK_CUBE * bC0;
                    AscendC::LoadData(GetB2(l0BufferId)[l0Offset], GetB1(l1BufferId), params);
                    AscendC::PipeBarrier<PIPE_MTE1>();
                }
            }
            return;
        }
        AscendC::LoadData(GetB2(l0BufferId), GetB1(l1BufferId), params);
    }

    __aicore__ inline void AcquireGroupResult(uint64_t groupIndex)
    {
        if (!params_->pairReduction || groupIndex % GroupResultParams::GROUPS_PER_BATCH == 0UL) {
            const uint8_t slot = GroupResultParams::GetCreditSlot(groupIndex, params_->pairReduction);
            AscendC::TQueSync<PIPE_V, PIPE_FIX> credit;
            credit.WaitFlag(static_cast<uint16_t>(GroupResultParams::VEC2CUBE_FLAG + slot));
        }
    }

    __aicore__ inline void CommitGroupResult(uint64_t groupIndex)
    {
        if (!params_->pairReduction || (groupIndex + 1UL) % GroupResultParams::GROUPS_PER_BATCH == 0UL ||
            groupIndex + 1UL == groupCount_) {
            const uint8_t slot = GroupResultParams::GetCreditSlot(groupIndex, params_->pairReduction);
            AscendC::TQueSync<PIPE_FIX, PIPE_V> ready;
            ready.SetFlag(static_cast<uint16_t>(GroupResultParams::CUBE2VEC_FLAG + slot));
        }
    }

    __aicore__ inline void CopyGroupResult(const TileParams& tile, uint64_t groupIndex, uint8_t groupBufferId,
                                           uint8_t cBufferId, const AscendC::LocalTensor<half>& groupOutput,
                                           uint64_t scaleRow = 0UL, bool releaseScale = true)
    {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cEvent_[cBufferId]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cEvent_[cBufferId]);
        AcquireGroupResult(groupIndex);
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = Cmct::Gemm::Align(tile.actualN, BLOCK_CUBE);
        params.dstStride = params.nSize;
        // MC62 is 1:1 coupled: the full M tile is written to one UB slot.
        params.dualDstCtl = 0;
        params.subBlockId = false;
        params.mSize = tile.actualM;
        params.srcStride = Cmct::Gemm::Align(tile.actualM, BLOCK_CUBE);
        params.quantPre = QuantMode_t::VDEQF16;
        params.unitFlag = l0cBufferCount_ == DOUBLE_BUFFER ? 0U : Cmct::Gemm::QuantBatchMatmul::FINAL_ACCUMULATION;
        params.params = {1, 1, 1};
        AscendC::Fixpipe<half, int32_t, ROW_MAJOR_UB>(groupOutput, GetC1(cBufferId),
                                                      GetScaleL1(groupBufferId)[scaleRow * alignedN_], params);
        CommitGroupResult(groupIndex);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cEvent_[cBufferId]);
        if (releaseScale) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(scaleEvent_[groupBufferId]);
        }
    }

private:
    static constexpr uint8_t RESIDENT_BUFFER_ID = 0U;
    static constexpr uint8_t DOUBLE_BUFFER = 2;
    static constexpr uint8_t INPUT_BUFFER_COUNT = LoadMode == NONE_FULL_LOAD_MODE ? DOUBLE_BUFFER :
                                                                                    FULL_LOAD_BUFFER_COUNT;
    static constexpr uint64_t BLOCK_CUBE = 16UL;
    static constexpr uint64_t DATA_BLOCK_BYTES = 32UL;

    const Params* params_{nullptr};
    AscendC::TPipe* pipe_{nullptr};
    AscendC::GlobalTensor<int8_t> x1_;
    AscendC::GlobalTensor<BType> x2_;
    AscendC::GlobalTensor<uint64_t> scale_;
    AscendC::LocalTensor<uint8_t> l1_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<int8_t> aL0_{AscendC::TPosition::A2, 0, AscendC::TOTAL_L0A_SIZE};
    AscendC::LocalTensor<BType> bL0_{AscendC::TPosition::B2, 0,
                                     AscendC::TOTAL_L0B_SIZE* WeightPolicy::ELEMENTS_PER_BYTE};
    AscendC::LocalTensor<int32_t> cL0_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE / sizeof(int32_t)};
    AscendC::TEventID inputEvent_[INPUT_BUFFER_COUNT]{};
    AscendC::TEventID segmentEvent_[DOUBLE_BUFFER]{};
    AscendC::TEventID l0Event_[DOUBLE_BUFFER]{};
    AscendC::TEventID l0cEvent_[DOUBLE_BUFFER]{};
    AscendC::TEventID scaleEvent_[DOUBLE_BUFFER]{};
    uint64_t alignedL1M_{0};
    uint64_t alignedN_{0};
    uint64_t kL1_{0};
    uint64_t bL1ByteOffset_{0};
    uint64_t scaleL1ByteOffset_{0};
    uint64_t l1BufferBytes_{0};
    uint64_t scaleL1BufferBytes_{0};
    uint64_t scaleRowsPerBuffer_{0};
    uint8_t scaleBufferPhase_{0};
    uint64_t groupCount_{0};
    uint64_t broadcastSegmentK_{0};
    uint64_t broadcastReadyK_{0};
    uint64_t broadcastSegmentIndex_{0};
    uint64_t nextInputWindowK_{0}, inputWindowIndex_{0};
    bool isAFullLoad_{false};
    bool residentInputLoaded_{false};
    bool isBFullLoad_{false};
    bool broadcastReuse_{false};
    bool broadcastACached_{false}, broadcastBCached_{false};
    bool broadcastACacheHit_{false}, broadcastBCacheHit_{false};
    uint64_t cachedBatchA_{0}, cachedMOffset_{0};
    uint64_t cachedBatchB_{0}, cachedNOffset_{0};
    uint32_t cachedActualM_{0}, cachedActualN_{0};
    uint8_t broadcastBufferId_{1};
    uint8_t l0PingPong_{0};
    uint8_t l0cBufferCount_{1};
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
