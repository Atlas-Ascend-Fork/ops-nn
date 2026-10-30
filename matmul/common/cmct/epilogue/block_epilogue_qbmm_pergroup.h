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
 * \file block_epilogue_qbmm_pergroup.h
 * \brief Direct-UB group consumption, FP32 reduction and FP16 bias/output for X2 per-group.
 */
#pragma once

#include "../utils/common_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <class GroupResultParams>
class BlockEpilogueX2PerGroup {
public:
    struct Params {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint64_t m{0}, n{0};
        uint32_t baseM{0}, baseN{0};
        bool pairReduction{false};
    };

    __aicore__ inline void Init(const Params& params, AscendC::TPipe* pipe)
    {
        params_ = &params;
        pipe_ = pipe;
        pairReduction_ = params.pairReduction;
        InitEvents();
        ringSlotElements_ = static_cast<uint64_t>(params_->baseM) * params_->baseN;
        InitGlobalTensor(params);
        InitUbBuffer();
        outputInFlight_ = false;
    }

    __aicore__ inline AscendC::LocalTensor<half> GetGroupResultTensor(uint64_t group)
    {
        return groupResult_[GetResultSlot(group)];
    }

    __aicore__ inline void ProcessGroupResults(uint64_t firstGroup, bool hasSecond, uint32_t actualM, uint32_t actualN)
    {
        if (!pairReduction_) {
            const uint8_t slot = GetResultSlot(firstGroup);
            WaitGroupResult(firstGroup);
            WaitPendingAccumulation();
            const uint32_t count = actualM * GetUbRowStride(actualN);
            if (firstGroup == 0UL) {
                AscendC::Cast(accumulation_, groupResult_[slot], AscendC::RoundMode::CAST_NONE, count);
            } else {
                AscendC::Cast(castTmp_, groupResult_[slot], AscendC::RoundMode::CAST_NONE, count);
            }
            ReleaseGroupResult(firstGroup);
            if (firstGroup != 0UL) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accumulation_, accumulation_, castTmp_, count);
            }
            accumulationPending_ = true;
            return;
        }
        WaitGroupResult(firstGroup);
        WaitPendingAccumulation();
        const uint8_t firstSlot = GetResultSlot(firstGroup);
        const uint32_t count = actualM * GetUbRowStride(actualN);
        if (firstGroup == 0UL) {
            if (hasSecond) {
                AccumulateGroups<true, true>(firstSlot, count);
            } else {
                AccumulateGroups<true, false>(firstSlot, count);
            }
        } else if (hasSecond) {
            AccumulateGroups<false, true>(firstSlot, count);
        } else {
            AccumulateGroups<false, false>(firstSlot, count);
        }
        ReleaseGroupResult(firstGroup);
        accumulationPending_ = true;
    }

    template <class TileParams>
    __aicore__ inline void AddBiasAndCopyOut(const TileParams& tile)
    {
        WaitPendingAccumulation();
        WaitPreviousStore();
        if (hasBias_) {
            CopyInBias(tile);
        }
        AddBiasCastAndCopyOut(tile);
    }

    __aicore__ inline void End()
    {
        WaitPendingAccumulation();
        WaitPreviousStore();
        for (uint8_t slot = 0U; slot < GroupResultParams::UB_RING_DEPTH; ++slot) {
            pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(inputReadyEvent_[slot]);
            pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(slotReadDoneEvent_[slot]);
        }
    }

private:
    __aicore__ inline uint8_t GetResultSlot(uint64_t group) const
    {
        const uint8_t slots = pairReduction_ ? GroupResultParams::PAIR_RESULT_SLOTS : GroupResultParams::UB_RING_DEPTH;
        return static_cast<uint8_t>(group % slots);
    }

    __aicore__ inline void WaitGroupResult(uint64_t group)
    {
        const uint8_t slot = GroupResultParams::GetCreditSlot(group, pairReduction_);
        {
            AscendC::TQueSync<PIPE_FIX, PIPE_V> ready;
            ready.WaitFlag(static_cast<uint16_t>(GroupResultParams::CUBE2VEC_FLAG + slot));
        }
    }

    __aicore__ inline void ReleaseGroupResult(uint64_t group)
    {
        const uint8_t slot = GroupResultParams::GetCreditSlot(group, pairReduction_);
        AscendC::TQueSync<PIPE_V, PIPE_FIX> credit;
        credit.SetFlag(static_cast<uint16_t>(GroupResultParams::VEC2CUBE_FLAG + slot));
    }

    __aicore__ inline void WaitBiasBufferAvailable()
    {
        // Wait for Vector before reusing the FP16 bias buffer.
        const auto biasReadDone = pipe_->FetchEventID<AscendC::HardEvent::V_MTE2>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(biasReadDone);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(biasReadDone);
    }

    __aicore__ inline void WaitBiasReady()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReadyEvent_[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReadyEvent_[0]);
    }

    template <bool isFirstPair, bool hasSecond>
    static __simd_vf__ inline void AccumulateGroupsVf(__ubuf__ float* dst, __ubuf__ half* first, __ubuf__ half* second,
                                                      uint32_t count, uint16_t repeats, uint32_t lanes)
    {
        static constexpr AscendC::Reg::CastTrait halfToFloat = {
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN};
        AscendC::Reg::RegTensor<half> input;
        AscendC::Reg::RegTensor<float> converted;
        AscendC::Reg::RegTensor<float> acc;
        for (uint16_t i = 0; i < repeats; ++i) {
            auto mask = AscendC::Reg::UpdateMask<float>(count);
            const uint32_t offset = static_cast<uint32_t>(i) * lanes;
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(input, first + offset);
            if constexpr (isFirstPair) {
                // Initialize from group zero, not +0: preserve signed zero.
                AscendC::Reg::Cast<float, half, halfToFloat>(acc, input, mask);
            } else {
                AscendC::Reg::Cast<float, half, halfToFloat>(converted, input, mask);
                AscendC::Reg::LoadAlign(acc, dst + offset);
                AscendC::Reg::Add(acc, acc, converted, mask);
            }
            if constexpr (hasSecond) {
                AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(input, second + offset);
                AscendC::Reg::Cast<float, half, halfToFloat>(converted, input, mask);
                // Two ordered FP32 adds, never acc + (first + second).
                AscendC::Reg::Add(acc, acc, converted, mask);
            }
            AscendC::Reg::StoreAlign(dst + offset, acc, mask);
        }
    }

    template <bool isFirstPair, bool hasSecond>
    __aicore__ inline void AccumulateGroups(uint8_t firstSlot, uint32_t count)
    {
        const uint32_t lanes = AscendC::GetVecLen() / sizeof(float);
        const uint16_t repeats = static_cast<uint16_t>(
            Cmct::Gemm::CeilDiv(static_cast<uint64_t>(count), static_cast<uint64_t>(lanes)));
        auto dst = reinterpret_cast<__ubuf__ float*>(accumulation_.GetPhyAddr());
        auto first = reinterpret_cast<__ubuf__ half*>(groupResult_[firstSlot].GetPhyAddr());
        auto second = reinterpret_cast<__ubuf__ half*>(groupResult_[firstSlot + 1U].GetPhyAddr());
        // The final full-register read stays inside baseM*baseN capacity;
        // only actualM*aligned(actualN) elements are written under the mask.
        AscendC::VF_CALL<AccumulateGroupsVf<isFirstPair, hasSecond>>(dst, first, second, count, repeats, lanes);
    }

    __aicore__ inline void InitEvents()
    {
        for (uint8_t slot = 0U; slot < GroupResultParams::UB_RING_DEPTH; ++slot) {
            inputReadyEvent_[slot] = pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>();
            slotReadDoneEvent_[slot] = pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>();
        }
        outputReadyEvent_ = pipe_->FetchEventID<AscendC::HardEvent::V_MTE3>();
        outputReleasedEvent_ = pipe_->FetchEventID<AscendC::HardEvent::MTE3_V>();
    }

    __aicore__ inline void InitGlobalTensor(const Params& params)
    {
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(params.outGmAddr));
        hasBias_ = params.biasGmAddr != nullptr;
        if (hasBias_) {
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(params.biasGmAddr));
        }
    }

    __aicore__ inline void InitUbBuffer()
    {
        const auto& mm = *params_;
        const uint64_t halfBytes = ringSlotElements_ * sizeof(half);
        const uint64_t floatBytes = ringSlotElements_ * sizeof(float);
        const uint8_t resultSlots = pairReduction_ ? GroupResultParams::PAIR_RESULT_SLOTS :
                                                     GroupResultParams::UB_RING_DEPTH;
        const uint64_t groupBytes = halfBytes * resultSlots;
        for (uint8_t slot = 0; slot < resultSlots; ++slot) {
            groupResult_[slot] = AscendC::LocalTensor<half>(AscendC::TPosition::VECCALC, slot * halfBytes,
                                                            ringSlotElements_);
        }
        // Both modes use 14 bytes per tile element, plus optional FP32/FP16 bias buffers.
        accumulation_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, groupBytes, ringSlotElements_);
        uint64_t outputOffset = groupBytes + floatBytes;
        if (!pairReduction_) {
            castTmp_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, outputOffset, ringSlotElements_);
            outputOffset += floatBytes;
        }
        output_ = AscendC::LocalTensor<half>(AscendC::TPosition::VECCALC, outputOffset, ringSlotElements_);
        if (hasBias_) {
            biasU_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, outputOffset + halfBytes, mm.baseN);
            biasHalf_ = AscendC::LocalTensor<half>(
                AscendC::TPosition::VECCALC, outputOffset + halfBytes + static_cast<uint64_t>(mm.baseN) * sizeof(float),
                mm.baseN);
        }
    }

    __aicore__ inline void WaitPendingAccumulation()
    {
        if (accumulationPending_) {
            AscendC::PipeBarrier<PIPE_V>();
            accumulationPending_ = false;
        }
    }

    __aicore__ inline static uint32_t GetUbRowStride(uint32_t actualN)
    {
        constexpr uint32_t halfPerBlock = 32U / sizeof(half);
        return static_cast<uint32_t>(Cmct::Gemm::Align(actualN, halfPerBlock));
    }

    template <class TileParams>
    __aicore__ inline void CopyInBias(const TileParams& tile)
    {
        WaitBiasBufferAvailable();
        const uint64_t gmOffset = tile.nOffset;
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(tile.actualN * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> padParams;
        AscendC::DataCopyPad(biasHalf_, biasGlobal_[gmOffset], params, padParams);
        WaitBiasReady();
        // Bias is in the output domain, with no group scale.
        AscendC::Cast(biasU_, biasHalf_, AscendC::RoundMode::CAST_NONE, tile.actualN);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddBias(const AscendC::LocalTensor<float>& accumulation, uint32_t actualM, uint32_t actualN)
    {
        constexpr uint32_t elemsPerRepeat = 256U / sizeof(float);
        const uint8_t rowStride = static_cast<uint8_t>(GetUbRowStride(actualN) * sizeof(float) / 32U);
        // Split Vector repeats because a tile can have 256 rows but the repeat field is uint8.
        constexpr uint32_t maxRepeat = 255U;
        const uint32_t ubRowStride = GetUbRowStride(actualN);
        for (uint32_t row = 0; row < actualM; row += maxRepeat) {
            const uint8_t repeats = static_cast<uint8_t>(Cmct::Gemm::Min<uint64_t>(actualM - row, maxRepeat));
            const uint64_t rowOffset = static_cast<uint64_t>(row) * ubRowStride;
            for (uint32_t col = 0; col < actualN; col += elemsPerRepeat) {
                const uint32_t width = static_cast<uint32_t>(Cmct::Gemm::Min<uint64_t>(actualN - col, elemsPerRepeat));
                AscendC::Add<float>(accumulation[rowOffset + col], accumulation[rowOffset + col], biasU_[col],
                                    static_cast<uint64_t>(width), repeats,
                                    AscendC::BinaryRepeatParams(1, 1, 1, rowStride, rowStride, 0));
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <class TileParams>
    __aicore__ inline void AddBiasCastAndCopyOut(const TileParams& tile)
    {
        const auto& mm = *params_;
        const uint64_t yBase = tile.batchC * static_cast<uint64_t>(mm.m) * mm.n + tile.mOffset * mm.n + tile.nOffset;

        if (hasBias_) {
            AddBias(accumulation_, tile.actualM, tile.actualN);
        }
        // A separate FP16 output buffer lets the next tile reuse group slots during MTE3.
        AscendC::LocalTensor<half> output = output_;
        const uint32_t count = tile.actualM * GetUbRowStride(tile.actualN);
        AscendC::Cast(output, accumulation_, AscendC::RoundMode::CAST_RINT, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputReadyEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputReadyEvent_);
        AscendC::DataCopyExtParams params{static_cast<uint16_t>(tile.actualM),
                                          static_cast<uint32_t>(tile.actualN * sizeof(half)), 0,
                                          static_cast<uint32_t>((mm.n - tile.actualN) * sizeof(half)), 0};
        AscendC::DataCopyPad(yGlobal_[yBase], output, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputReleasedEvent_);
        outputInFlight_ = true;
    }

    __aicore__ inline void WaitPreviousStore()
    {
        if (outputInFlight_) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputReleasedEvent_);
            outputInFlight_ = false;
        }
    }

private:
    const Params* params_{nullptr};
    AscendC::TPipe* pipe_{nullptr};
    AscendC::GlobalTensor<half> yGlobal_;
    AscendC::GlobalTensor<half> biasGlobal_;
    AscendC::LocalTensor<half> groupResult_[GroupResultParams::PAIR_RESULT_SLOTS];
    AscendC::LocalTensor<half> output_;
    AscendC::LocalTensor<half> biasHalf_;
    AscendC::LocalTensor<float> accumulation_;
    AscendC::LocalTensor<float> castTmp_;
    AscendC::LocalTensor<float> biasU_;
    bool hasBias_{false};
    bool pairReduction_{false};
    bool outputInFlight_{false};
    bool accumulationPending_{false};
    uint64_t ringSlotElements_{0};
    AscendC::TEventID inputReadyEvent_[GroupResultParams::UB_RING_DEPTH]{};
    AscendC::TEventID slotReadDoneEvent_[GroupResultParams::UB_RING_DEPTH]{};
    AscendC::TEventID outputReadyEvent_{};
    AscendC::TEventID outputReleasedEvent_{};
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
