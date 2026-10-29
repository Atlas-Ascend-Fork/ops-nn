/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_bp_input_sub_func_polyphase_transpose_phased.h
 * \brief PHASED 任务、phase 和 tap 分段调度。
 */

#ifndef CONV_BP_INPUT_SUB_FUNC_POLYPHASE_TRANSPOSE_PHASED_H
#define CONV_BP_INPUT_SUB_FUNC_POLYPHASE_TRANSPOSE_PHASED_H

// PHASED 公共任务、phase 与 tap 分段调度。

template <PolyphaseRole role>
__aicore__ inline void ProcessFullTapPhase(const PhasedTaskDescriptor& work, uint32_t phase, uint32_t outputBank,
                                           PhasedCoreState& state) const
{
    const uint32_t phaseElements = tiling_->baseN * static_cast<uint32_t>(tiling_->kValueSegment);
    const auto& geometry = work.geometry;
    const uint32_t bBank = state.filterPhaseOrdinal++ & 1U;
    const uint32_t bOffset = memoryLayout_.a1Bytes + bBank * memoryLayout_.b1BankBytes;
    if constexpr (role == PolyphaseRole::CUBE) {
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_MTE1>(POLYPHASE_AIV1_FLAG_BASE + bBank);
        LocalTensor<T> a1(TPosition::A1, 0, static_cast<uint32_t>(memoryLayout_.a1Bytes / sizeof(T)));
        LocalTensor<T> b1(TPosition::B1, bOffset, phaseElements);
        const uint32_t stripeAlign = Ops::Base::CeilAlign<uint32_t>(static_cast<uint32_t>(tiling_->singleCoreM),
                                                                    POLYPHASE_C0);
        for (uint32_t inner = 0; inner < geometry.curJ; inner += tiling_->baseM) {
            const uint32_t curM = Std::min(tiling_->baseM, geometry.curJ - inner);
            ComputePhaseTileToUb(geometry, inner, curM, stripeAlign, outputBank, phase, inner, a1, b1);
        }
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE1>(POLYPHASE_AIV1_FLAG_BASE + bBank);
    } else {
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(bBank);
        PackPhaseFilter(geometry, phase, bOffset);
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(bBank);
    }
}

template <PhasedStrategy strategy>
__aicore__ inline void StorePhasedOutput(const PhasedTaskDescriptor& work, uint32_t bank, PhasedCoreState& state) const
{
    const auto& geometry = work.geometry;
    if constexpr (strategy == PhasedStrategy::ONE_PHASE) {
        const uint32_t mAlign = Ops::Base::CeilAlign<uint32_t>(static_cast<uint32_t>(tiling_->singleCoreM),
                                                               POLYPHASE_C0);
        TransformAndStore(geometry, 0, geometry.curJ, mAlign, bank, tiling_->polyphaseDense != 0U);
    } else if constexpr (strategy == PhasedStrategy::DENSE_SEGMENTED) {
        const auto output = DenseSegmentedOutput(geometry, work.phaseGroup);
        DenseSegmentedStore(geometry, output, bank);
    } else if constexpr (strategy == PhasedStrategy::PARALLEL_DEPTHWISE) {
        const auto output = SegmentedParallelOutput(geometry, work.phaseGroup);
        InitDepthwiseOutputIndex<true>(output.allZero, state.outputInitialized);
        SegmentedParallelStore(geometry, output, bank);
    } else {
        const auto output = SegmentedOutput(geometry);
        InitDepthwiseOutputIndex<false>(output.allZero, state.outputInitialized);
        SegmentedStore(geometry, output, bank);
    }
}

template <PolyphaseRole role, PhasedStrategy strategy>
__aicore__ inline void RunPhasedTask(const PhasedTaskDescriptor& work, PhasedCoreState& state) const
{
    const uint32_t workBank = static_cast<uint32_t>(work.round & 1U);
    // ONE_PHASE 独占整条输入；事件顺序与原两层转发一致。
    if constexpr (role == PolyphaseRole::CUBE && strategy == PhasedStrategy::ONE_PHASE) {
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(0));
        LocalTensor<T> a1(TPosition::A1, 0, static_cast<uint32_t>(memoryLayout_.a1Bytes / sizeof(T)));
        LoadA1(work.geometry, a1);
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
    }
    if constexpr (role == PolyphaseRole::FILTER) {
        ForEachActivePhase<role, strategy>(work, 0U, state);
    } else if constexpr (role == PolyphaseRole::CUBE) {
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(workBank);
        if constexpr (strategy == PhasedStrategy::ONE_PHASE) {
            state.filterPhaseOrdinal = static_cast<uint32_t>(work.round * tiling_->stepKb);
        }
        ForEachActivePhase<PolyphaseRole::CUBE, strategy>(work, workBank, state);
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(workBank);
    } else {
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_V>(workBank);
        StorePhasedOutput<strategy>(work, workBank, state);
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(workBank);
    }
    if constexpr (role == PolyphaseRole::CUBE && strategy == PhasedStrategy::ONE_PHASE) {
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(0));
    }
}

template <PolyphaseRole role, PhasedStrategy strategy>
__aicore__ inline void ForEachTapSegment(const PolyphaseWorkGeometry& geometry, const SegmentedPhaseRange& range,
                                         PhasedCoreState& state, SegmentedParallelA1Cache& cache) const
{
    const uint32_t cap = tiling_->stepKa;
    bool firstK = true;
    // dense/depthwise 的生产者与消费者共用唯一有界分段遍历。
    for (uint64_t first = range.first; first < range.end; first += cap) {
        const PhaseSegment segment{
            first, Std::min(cap, static_cast<uint32_t>(range.end - first)),
            static_cast<uint32_t>(state.filterSegmentOrdinal++ & 1U),
            static_cast<int64_t>(geometry.uStart) + range.inputShift - static_cast<int64_t>(first + cap - 1U)};
        if constexpr (role == PolyphaseRole::CUBE) {
            if constexpr (strategy == PhasedStrategy::DENSE_SEGMENTED) {
                DenseSegment(geometry, segment, state.denseKTileOrdinal, firstK);
            } else {
                DepthwiseSegment<strategy == PhasedStrategy::PARALLEL_DEPTHWISE>(geometry, range, segment, cache);
            }
        } else {
            CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(segment.bank);
            if constexpr (strategy == PhasedStrategy::DENSE_SEGMENTED) {
                DenseSegmentedPackB(geometry, range, first, segment.bank);
            } else {
                if (!state.filterInitialized) {
                    LocalTensor<uint16_t> index(TPosition::VECIN,
                                                POLYPHASE_FILTER_PACK_BUFFERS * memoryLayout_.phaseB1Bytes,
                                                (Ops::Base::GetVRegSize() / sizeof(T)));
                    SetFilterGatherIndex(index, tiling_->stepKa, tiling_->baseK);
                    NdDmaDci();
                    state.filterInitialized = true;
                }
                SegmentedPackFilter(geometry, range, first, segment.length, segment.bank);
            }
            CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(segment.bank);
        }
    }
}

template <PhasedStrategy strategy>
__aicore__ inline void FinishCappedPhase(uint32_t localPhase, uint32_t outputBank) const
{
    const uint32_t m = tiling_->baseM;
    LocalTensor<L0cT> c0(TPosition::CO1, 0, m * POLYPHASE_C0);
    SetFlag<HardEvent::M_FIX>(static_cast<event_t>(0));
    WaitFlag<HardEvent::M_FIX>(static_cast<event_t>(0));
    if constexpr (strategy == PhasedStrategy::DENSE_SEGMENTED) {
        LocalTensor<float> phaseOut(
            TPosition::VECIN,
            outputBank * memoryLayout_.outputBankPitchBytes + localPhase * POLYPHASE_C0 * m * sizeof(float),
            POLYPHASE_C0 * m);
        DataMove::L0cToUb(phaseOut, c0, m, POLYPHASE_C0, m, m, QuantMode_t::NoQuant);
    } else {
        LocalTensor<T> phaseOut(
            TPosition::VECIN,
            outputBank * memoryLayout_.outputBankPitchBytes + localPhase * POLYPHASE_C0 * m * sizeof(T),
            POLYPHASE_C0 * m);
        DataMove::L0cToUb(phaseOut, c0, m, POLYPHASE_C0, m, m,
                          (std::is_same<T, half>::value ? QuantMode_t::F322F16 : QuantMode_t::F322BF16));
    }
    SetFlag<HardEvent::FIX_M>(static_cast<event_t>(0));
}

template <PhasedStrategy strategy>
__aicore__ inline SegmentedPhaseIterator ActivePhases(const PhasedTaskDescriptor& work, uint32_t& phaseBase) const
{
    phaseBase = 0U;
    if constexpr (strategy == PhasedStrategy::SERIAL_DEPTHWISE) {
        const auto output = SegmentedOutput(work.geometry);
        return output.allZero ? SegmentedPhaseIterator{0U, 0U, 0U, 0U} : SegmentedPhases(output.phases);
    } else if constexpr (strategy == PhasedStrategy::PARALLEL_DEPTHWISE) {
        const auto output = SegmentedParallelOutput(work.geometry, work.phaseGroup);
        phaseBase = output.begin;
        return output.allZero ? SegmentedPhaseIterator{0U, 0U, 0U, 0U} : SegmentedParallelPhases(output);
    } else if constexpr (strategy == PhasedStrategy::DENSE_SEGMENTED) {
        const auto output = DenseSegmentedOutput(work.geometry, work.phaseGroup);
        phaseBase = output.begin;
        return {output.begin, output.begin + output.width, 0U, 0U};
    } else {
        return {0U, tiling_->stepKb, 0U, 0U};
    }
}

template <PolyphaseRole role, PhasedStrategy strategy>
__aicore__ inline void ForEachActivePhase(const PhasedTaskDescriptor& work, uint32_t outputBank,
                                          PhasedCoreState& state) const
{
    uint32_t phaseBase;
    auto iterator = ActivePhases<strategy>(work, phaseBase);
    // 并行 depthwise 的 A1 缓存仅在本输出任务内复用；
    // 全 tap 路径不使用该缓存。
    SegmentedParallelA1Cache cache;
    for (uint32_t phase; SegmentedNextPhase(iterator, phase);) {
        if constexpr (strategy != PhasedStrategy::ONE_PHASE) {
            const auto range = SegmentedRange(work.geometry, phase);
            if (range.end > range.first) {
                if constexpr (role == PolyphaseRole::CUBE) {
                    WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(0));
                }
                ForEachTapSegment<role, strategy>(work.geometry, range, state, cache);
                if constexpr (role == PolyphaseRole::CUBE) {
                    FinishCappedPhase<strategy>(phase - phaseBase, outputBank);
                }
            }
        } else {
            // 全 tap 路径一次计算完整 phase。
            ProcessFullTapPhase<role>(work, phase, outputBank, state);
        }
    }
}

template <PhasedStrategy strategy>
__aicore__ inline uint32_t GetPhaseGroupCount() const
{
    if constexpr (strategy == PhasedStrategy::PARALLEL_DEPTHWISE || strategy == PhasedStrategy::DENSE_SEGMENTED) {
        return tiling_->dDim;
    }
    return 1U;
}

template <PhasedStrategy strategy>
__aicore__ inline PhasedTaskDescriptor DecodePhasedTask(uint64_t task, uint64_t round) const
{
    PhasedTaskDescriptor work;
    const uint32_t phaseGroups = GetPhaseGroupCount<strategy>();
    work.phaseGroup = static_cast<uint32_t>(task % phaseGroups);
    work.round = round;
    const uint64_t basic = task / phaseGroups;
    auto& geometry = work.geometry;
    if constexpr (strategy == PhasedStrategy::ONE_PHASE) {
        DecodeGeometry(basic, geometry);
    } else {
        const uint64_t perBatch = static_cast<uint64_t>(tiling_->nDim) * tiling_->mDim;
        const uint64_t remainder = basic % perBatch;
        geometry.batchIdx = static_cast<uint32_t>(basic / perBatch);
        geometry.nStart = static_cast<uint32_t>(remainder / tiling_->mDim) * POLYPHASE_C0;
        geometry.curN = Std::min(POLYPHASE_C0, tiling_->cin - geometry.nStart);
        geometry.uStart = remainder % tiling_->mDim * tiling_->baseM;
        const uint64_t j = (static_cast<uint64_t>(tiling_->wi) + tiling_->stepKb - 1U) / tiling_->stepKb;
        geometry.curJ = Std::min(tiling_->baseM, static_cast<uint32_t>(j - geometry.uStart));
        geometry.localA1W = tiling_->baseM + tiling_->stepKa - 1U;
    }
    return work;
}

template <PolyphaseRole role, PhasedStrategy strategy>
__aicore__ inline void InitPhasedSync() const
{
    const uint32_t phaseElements = tiling_->baseN * static_cast<uint32_t>(tiling_->kValueSegment);
    if constexpr (role == PolyphaseRole::CUBE) {
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(0));
        if constexpr (strategy != PhasedStrategy::ONE_PHASE) {
            SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
            SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
        }
        SetFlag<HardEvent::FIX_M>(static_cast<event_t>(0));
        SeedAicProtocol();
    } else if constexpr (role == PolyphaseRole::OUTPUT) {
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(0);
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(1);
    } else if constexpr (strategy == PhasedStrategy::ONE_PHASE) {
        if (tiling_->polyphaseDense == 0U) {
            LocalTensor<uint16_t> index(TPosition::VECIN, POLYPHASE_FILTER_PACK_BUFFERS * phaseElements * sizeof(T),
                                        (Ops::Base::GetVRegSize() / sizeof(T)));
            SetFilterGatherIndex(index, tiling_->stepKa, static_cast<uint32_t>(tiling_->kValueSegment));
        }
        NdDmaDci();
    } else if constexpr (strategy == PhasedStrategy::DENSE_SEGMENTED) {
        NdDmaDci();
    }
}

template <PolyphaseRole role, PhasedStrategy strategy>
__aicore__ inline void DrainPhasedSync() const
{
    if constexpr (role == PolyphaseRole::CUBE) {
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(0));
        if constexpr (strategy != PhasedStrategy::ONE_PHASE) {
            WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
            WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
        }
        WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(0));
        DrainAicProtocol();
    } else if constexpr (role == PolyphaseRole::FILTER) {
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(0);
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(1);
    }
}

template <PolyphaseRole role, PhasedStrategy strategy>
__aicore__ inline void ProcessPhased()
{
    if constexpr (strategy == PhasedStrategy::DENSE_SEGMENTED) {
        DenseSegmentedPrepareMemory();
    } else if constexpr (strategy == PhasedStrategy::PARALLEL_DEPTHWISE) {
        PrepareDepthwiseSegmentedMemory<true>();
    } else if constexpr (strategy == PhasedStrategy::SERIAL_DEPTHWISE) {
        PrepareDepthwiseSegmentedMemory<false>();
    } else {
        PrepareMemoryLayout(tiling_->polyphaseDense != 0U);
    }
    const uint64_t block = GetAicBlockIdx();
    const uint64_t tasks = static_cast<uint64_t>(tiling_->batch) * tiling_->nDim * tiling_->mDim *
                           GetPhaseGroupCount<strategy>();
    if (tasks == 0U || tiling_->coreNum == 0U || tiling_->coreNum > tasks || block >= tiling_->coreNum) {
        return;
    }
    const uint64_t rounds = 1U + (tasks - 1U - block) / tiling_->coreNum;
    PhasedCoreState state;
    InitPhasedSync<role, strategy>();
    // 任务依次按 batch、通道 tile、空间 tile、phase group 排列。
    uint64_t task = block;
    for (uint64_t round = 0U; round < rounds; ++round, task += tiling_->coreNum) {
        const auto work = DecodePhasedTask<strategy>(task, round);
        RunPhasedTask<role, strategy>(work, state);
    }
    DrainPhasedSync<role, strategy>();
}

template <PolyphaseRole role>
__aicore__ inline void DispatchPhased()
{
    switch (static_cast<PhasedStrategy>(tiling_->polyphaseStrategy)) {
        case PhasedStrategy::DENSE_SEGMENTED:
            ProcessPhased<role, PhasedStrategy::DENSE_SEGMENTED>();
            break;
        case PhasedStrategy::PARALLEL_DEPTHWISE:
            ProcessPhased<role, PhasedStrategy::PARALLEL_DEPTHWISE>();
            break;
        case PhasedStrategy::SERIAL_DEPTHWISE:
            ProcessPhased<role, PhasedStrategy::SERIAL_DEPTHWISE>();
            break;
        case PhasedStrategy::ONE_PHASE:
            ProcessPhased<role, PhasedStrategy::ONE_PHASE>();
            break;
    }
}

#endif // CONV_BP_INPUT_SUB_FUNC_POLYPHASE_TRANSPOSE_PHASED_H
