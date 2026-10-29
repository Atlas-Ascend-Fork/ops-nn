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
 * \file conv_bp_input_sub_func_polyphase_direct_dense.h
 * \brief DIRECT_DENSE 几何、计算与任务调度。
 */

#ifndef CONV_BP_INPUT_SUB_FUNC_POLYPHASE_DIRECT_DENSE_H
#define CONV_BP_INPUT_SUB_FUNC_POLYPHASE_DIRECT_DENSE_H

struct DirectDenseWork {
    uint32_t batchIdx;
    uint32_t groupStart;
    uint32_t liveGroups;
    uint32_t channelStart;
    uint32_t channelCount;
    uint32_t outputStart;
    uint32_t outputWidth;
    uint32_t inputStart;
    uint32_t inputWidth;
};

__aicore__ inline uint64_t DirectDenseTasks() const
{
    uint64_t tasks = static_cast<uint64_t>(tiling_->mDim) * tiling_->nDim;
    if (!tasks || !tiling_->groupDim || !tiling_->batch || tasks > UINT32_MAX / tiling_->groupDim) {
        return 0U;
    }
    tasks *= tiling_->groupDim;
    return tasks > UINT32_MAX / tiling_->batch ? 0U : tasks * tiling_->batch;
}

__aicore__ inline DirectDenseWork DirectDenseGeometry(uint64_t task) const
{
    DirectDenseWork work;
    uint64_t index = task / tiling_->mDim;
    work.channelStart = index % tiling_->nDim * tiling_->singleCoreCin;
    index /= tiling_->nDim;
    work.groupStart = index % tiling_->groupDim * tiling_->enlarge;
    work.liveGroups = Std::min(tiling_->enlarge, tiling_->oriGroup - work.groupStart);
    work.batchIdx = index / tiling_->groupDim;
    work.channelCount = Std::min(tiling_->singleCoreCin,
                                 work.liveGroups * (tiling_->cin / tiling_->oriGroup) - work.channelStart);
    work.outputStart = task % tiling_->mDim * tiling_->singleCoreM;
    work.outputWidth = Std::min(static_cast<uint32_t>(tiling_->singleCoreM), tiling_->wi - work.outputStart);
    int64_t lo = CeilDivSigned(static_cast<int64_t>(work.outputStart) + tiling_->padLeft -
                                   static_cast<int64_t>(tiling_->wk - 1) * tiling_->dilationW,
                               tiling_->strideW);
    int64_t hi = FloorDivSigned(static_cast<int64_t>(work.outputStart) + work.outputWidth - 1 + tiling_->padLeft,
                                tiling_->strideW);
    lo = lo < 0 ? 0 : lo;
    hi = hi >= tiling_->wo ? tiling_->wo - 1 : hi;
    work.inputStart = lo < tiling_->wo ? lo : tiling_->wo;
    work.inputWidth = hi >= lo ? hi - lo + 1 : 0;
    return work;
}
// VF 的 ZERO 转换将值放在偶数 16 位槽；float 掩码保护
// 奇数槽，index >> 1 将有效槽映射到输出元素。

__aicore__ inline void DirectDenseEpilogue(const DirectDenseWork& work, uint32_t cBase, uint32_t matmulOffset,
                                           uint32_t yOffset, uint32_t biasOffset)
{
    LocalTensor<T> output(TPosition::VECIN, yOffset,
                          tiling_->singleCoreCin * Ops::Base::CeilAlign<uint32_t>(tiling_->singleCoreM, POLYPHASE_C0));
    LocalTensor<float> matmulOut(TPosition::VECIN, matmulOffset, tiling_->baseM * tiling_->baseN);
    LocalTensor<float> bias(
        TPosition::VECIN, biasOffset,
        Ops::Base::CeilAlign<uint32_t>(tiling_->singleCoreCin, (Ops::Base::GetUbBlockSize() / sizeof(float))));
    Duplicate(output, T(0), output.GetSize());
    PipeBarrier<PIPE_V>();
    for (uint32_t phase = 0; phase < Std::min(work.outputWidth, tiling_->strideW); ++phase) {
        uint32_t first = tiling_->wk;
        uint32_t step = 1;
        uint32_t taps = 0;
        uint32_t a = tiling_->strideW;
        uint32_t b = tiling_->dilationW;
        while (b) {
            uint32_t r = a % b;
            a = b;
            b = r;
        }
        step = tiling_->strideW / a;
        for (uint32_t k = 0; k < Std::min(step, tiling_->wk); ++k) {
            if ((static_cast<uint64_t>(k) * tiling_->dilationW) % tiling_->strideW ==
                (static_cast<uint64_t>(work.outputStart) + phase + tiling_->padLeft) % tiling_->strideW) {
                first = k;
                break;
            }
        }
        if (first < tiling_->wk) {
            taps = (tiling_->wk - 1 - first) / step + 1;
        }
        int64_t input = taps ? (static_cast<int64_t>(work.outputStart) + phase + tiling_->padLeft -
                                static_cast<int64_t>(first) * tiling_->dilationW) /
                                       tiling_->strideW -
                                   work.inputStart :
                               0;
        Convolution3DBackpropFunc::PolyphaseDirectDensePhase(
            reinterpret_cast<__ubuf__ float*>(matmulOut.GetPhyAddr()) +
                static_cast<uint32_t>(cBase * tiling_->wk * tiling_->baseM),
            reinterpret_cast<__ubuf__ float*>(bias.GetPhyAddr()), reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()),
            work.channelCount, tiling_->wk, tiling_->baseM,
            Ops::Base::CeilAlign<uint32_t>(tiling_->singleCoreM, POLYPHASE_C0),
            (work.outputWidth - 1 - phase) / tiling_->strideW + 1, phase, tiling_->strideW, first, step, taps, input,
            tiling_->dilationW / a, work.inputWidth, HasBias());
    }
}
__aicore__ inline void DirectDenseCopyA(const DirectDenseWork& work, uint32_t start, uint32_t span, uint32_t offset)
{
    const uint32_t ci = tiling_->cout / tiling_->oriGroup;
    const uint32_t channels = work.liveGroups * ci;
    LocalTensor<T> a1(TPosition::A1, offset, tiling_->baseM * span);
    if (work.inputWidth != tiling_->baseM || start >= channels || Std::min(span, channels - start) != span) {
        DataMove::ZeroL1(a1);
        PipeBarrier<PIPE_MTE2>();
    }
    if (work.inputWidth && start < channels) {
        const uint64_t channel = static_cast<uint64_t>(work.batchIdx) * tiling_->cout +
                                 static_cast<uint64_t>(work.groupStart) * ci + start;
        DataMove::GmToL1Matrix(a1, dedyGm_[channel * tiling_->wo + work.inputStart], work.inputWidth,
                               Std::min(span, channels - start), tiling_->wo, tiling_->baseM);
    }
}
__aicore__ inline void DirectDenseCopyB(const DirectDenseWork& work, uint32_t start, uint32_t span, uint32_t offset)
{
    const uint32_t ci = tiling_->cout / tiling_->oriGroup;
    const uint32_t co = tiling_->cin / tiling_->oriGroup;
    LocalTensor<T> b1(TPosition::B1, offset, tiling_->baseN * span);
    if (work.channelCount * tiling_->wk != tiling_->baseN || start >= ci || Std::min(span, ci - start) != span) {
        DataMove::ZeroL1(b1);
        PipeBarrier<PIPE_MTE2>();
    }
    if (start < ci) {
        const uint64_t gm = ((static_cast<uint64_t>(work.groupStart) * ci + start) * co + work.channelStart) *
                            tiling_->wk;
        DataMove::GmToL1Matrix(b1, filterGm_[gm], work.channelCount * tiling_->wk, Std::min(span, ci - start),
                               static_cast<uint64_t>(co) * tiling_->wk, tiling_->baseN);
    }
}
// 仅 E>1 使用全驻留块对角 B；E=1 仍使用上方 CopyB。
__aicore__ inline void CopyDirectDenseGroupFilter(const DirectDenseWork& work, LocalTensor<T>& dn) const
{
    const uint32_t ci = tiling_->cout / tiling_->oriGroup;
    const uint32_t co = tiling_->cin / tiling_->oriGroup;
    const uint64_t offset = static_cast<uint64_t>(work.groupStart) * ci * co * tiling_->wk;
    NdDmaDci();
    DataMove::GmToUb<POLYPHASE_NDDMA_3D>(
        dn, filterGm_[offset],
        {{co * tiling_->wk, 1U, 1U},
         {ci, co * tiling_->wk, tiling_->baseN},
         {work.liveGroups, static_cast<uint64_t>(ci) * co * tiling_->wk, ci * tiling_->baseN + co * tiling_->wk}});
}

__aicore__ inline void DirectDensePrepareGroupB(const DirectDenseWork& work, uint32_t biasOffset, uint32_t& cachedGroup,
                                                uint32_t& cachedChannel)
{
    if (tiling_->enlarge == 1U || !work.inputWidth ||
        (cachedGroup == work.groupStart && cachedChannel == work.channelStart)) {
        return;
    }
    CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_V>(POLYPHASE_GROUP_PACK_REQUEST);
    const uint32_t k = tiling_->kValueSegment;
    const uint32_t n = tiling_->baseN;
    const uint32_t packBytes = Ops::Base::CeilAlign<uint32_t>(sizeof(T) * k * n, Ops::Base::GetUbBlockSize());
    const uint32_t packOffset = biasOffset + Ops::Base::CeilAlign<uint32_t>(sizeof(float) * tiling_->singleCoreCin,
                                                                            Ops::Base::GetUbBlockSize());
    LocalTensor<T> dn(TPosition::VECIN, packOffset, k * n);
    LocalTensor<T> nz(TPosition::VECIN, packOffset + packBytes, k * n);
    LocalTensor<uint16_t> index(TPosition::VECIN, packOffset + POLYPHASE_FILTER_PACK_BUFFERS * packBytes,
                                (Ops::Base::GetVRegSize() / sizeof(T)));
    Duplicate(dn, T(0), k * n);
    SetFlag<HardEvent::V_MTE2>(POLYPHASE_FILTER_EVENT);
    WaitFlag<HardEvent::V_MTE2>(POLYPHASE_FILTER_EVENT);
    CopyDirectDenseGroupFilter(work, dn);
    SetFlag<HardEvent::MTE2_V>(POLYPHASE_FILTER_EVENT);
    WaitFlag<HardEvent::MTE2_V>(POLYPHASE_FILTER_EVENT);
    SetFlag<HardEvent::V_S>(POLYPHASE_FILTER_EVENT);
    WaitFlag<HardEvent::V_S>(POLYPHASE_FILTER_EVENT);
    for (uint32_t i = 0U; i < POLYPHASE_C0; ++i) {
        index.SetValue(i, static_cast<uint16_t>(i * n));
    }
    SetFlag<HardEvent::S_V>(POLYPHASE_FILTER_EVENT);
    WaitFlag<HardEvent::S_V>(POLYPHASE_FILTER_EVENT);
    auto idx = reinterpret_cast<__ubuf__ uint16_t*>(index.GetPhyAddr());
    Convolution3DBackpropFunc::ExpandGatherIdxByStride<uint16_t>(
        idx, (Ops::Base::GetVRegSize() / sizeof(T) / POLYPHASE_C0) - 1U, POLYPHASE_C0, POLYPHASE_C0, POLYPHASE_C0, 1U);
    PipeBarrier<PIPE_V>();
    Convolution3DBackpropFunc::GatherDn2Nz4Group<T, uint16_t>(
        idx, reinterpret_cast<__ubuf__ T*>(dn.GetPhyAddr()), reinterpret_cast<__ubuf__ T*>(nz.GetPhyAddr()),
        k / POLYPHASE_C0, 1U, n / (Ops::Base::GetVRegSize() / sizeof(T) / POLYPHASE_C0), POLYPHASE_C0 * n,
        (Ops::Base::GetVRegSize() / sizeof(T) / POLYPHASE_C0), POLYPHASE_C0 * n, POLYPHASE_C0 * n,
        (Ops::Base::GetVRegSize() / sizeof(T)));
    SetFlag<HardEvent::V_MTE3>(POLYPHASE_FILTER_EVENT);
    WaitFlag<HardEvent::V_MTE3>(POLYPHASE_FILTER_EVENT);
    LocalTensor<T> b(TPosition::B1, sizeof(T) * tiling_->baseM * k, k * n);
    DataMove::UbToL1Blocks(b, nz, k * n);
    CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(POLYPHASE_GROUP_PACK_READY);
    cachedGroup = work.groupStart;
    cachedChannel = work.channelStart;
}
__aicore__ inline void DirectDenseLoadL0(uint32_t aOffset, uint32_t bOffset, uint32_t aK, uint32_t bK, uint32_t bank)
{
    const uint32_t m = tiling_->baseM;
    const uint32_t n = tiling_->baseN;
    const uint32_t k = tiling_->baseK;
    LocalTensor<T> a1(TPosition::A1, aOffset, 1U);
    LocalTensor<T> b1(TPosition::B1, bOffset, 1U);
    LocalTensor<T> a2(TPosition::A2, bank * m * k * sizeof(T), m * k);
    LocalTensor<T> b2(TPosition::B2, bank * n * k * sizeof(T), n * k);
    DataMove::L1ToL0A(a2, a1, {m, tiling_->singleCoreCout, 1U, 0U, m, aK, k, m / POLYPHASE_C0});
    DataMove::L1ToL0B(b2, b1, n / POLYPHASE_C0, n / POLYPHASE_C0, bK / POLYPHASE_C0, k / POLYPHASE_C0);
}

__aicore__ inline void DirectDenseBias(const DirectDenseWork& work, uint32_t offset)
{
    if (!HasBias()) {
        return;
    }
    SetFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::V_MTE2>(1);
    LocalTensor<float> bias(
        TPosition::VECIN, offset,
        Ops::Base::CeilAlign<uint32_t>(tiling_->singleCoreCin, (Ops::Base::GetUbBlockSize() / sizeof(float))));
    const uint64_t offsetGm = static_cast<uint64_t>(work.groupStart) * (tiling_->cin / tiling_->oriGroup) +
                              work.channelStart;
    DataMove::GmToUbVector(bias, biasGm_[offsetGm], work.channelCount, 0U);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
}
__aicore__ inline void DirectDenseStore(const DirectDenseWork& work, uint32_t offset)
{
    const uint32_t pitch = Ops::Base::CeilAlign<uint32_t>(tiling_->singleCoreM, POLYPHASE_C0);
    LocalTensor<T> y(TPosition::VECIN, offset, tiling_->singleCoreCin * pitch);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    const uint64_t channel = static_cast<uint64_t>(work.batchIdx) * tiling_->cin +
                             static_cast<uint64_t>(work.groupStart) * (tiling_->cin / tiling_->oriGroup) +
                             work.channelStart;
    DataMove::UbToGm(yGm_[channel * tiling_->wi + work.outputStart], y, work.channelCount, work.outputWidth,
                     (pitch - Ops::Base::CeilAlign<uint32_t>(work.outputWidth, POLYPHASE_C0)) / POLYPHASE_C0,
                     static_cast<uint64_t>(tiling_->wi - work.outputWidth) * sizeof(T));
}

struct DirectDenseL1Layout {
    uint32_t m;
    uint32_t n;
    uint32_t k0;
    uint32_t ka;
    uint32_t kb;
    bool aFullLoad;
    bool bFullLoad;
    uint32_t aBytes;
    uint32_t bBytes;
    uint32_t bBase;
    uint32_t chunk;
    uint32_t chunks;
};

__aicore__ inline DirectDenseL1Layout GetDirectDenseL1Layout() const
{
    const uint32_t m = tiling_->baseM;
    const uint32_t n = tiling_->baseN;
    const uint32_t k0 = tiling_->baseK;
    const uint32_t ka = tiling_->singleCoreCout;
    const uint32_t kb = tiling_->stepKb * k0;
    const bool aFullLoad = tiling_->enableFullLoad & POLYPHASE_A_FULL;
    const bool bFullLoad = tiling_->enableFullLoad & POLYPHASE_B_FULL;
    const uint32_t aBytes = m * ka * sizeof(T);
    const uint32_t bBytes = n * kb * sizeof(T);
    const uint32_t bBase = aBytes * (aFullLoad ? 1 : POLYPHASE_DOUBLE_BUFFER);
    const uint32_t chunk = aFullLoad ? (bFullLoad ? tiling_->kValueSegment : kb) : ka;
    const uint32_t chunks = (tiling_->kValueSegment + chunk - 1) / chunk;
    return {m, n, k0, ka, kb, aFullLoad, bFullLoad, aBytes, bBytes, bBase, chunk, chunks};
}

__aicore__ inline void PrepareDirectDenseTask(const DirectDenseWork& work, const DirectDenseL1Layout& layout,
                                              uint32_t& cachedBGroup, uint32_t& cachedBChannel)
{
    const bool bMiss = cachedBGroup != work.groupStart || cachedBChannel != work.channelStart;
    if (tiling_->enlarge > 1U && bMiss) {
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE1>(POLYPHASE_GROUP_PACK_REQUEST);
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_MTE1>(POLYPHASE_GROUP_PACK_READY);
        cachedBGroup = work.groupStart;
        cachedBChannel = work.channelStart;
    }
    if (layout.aFullLoad) {
        DirectDenseCopyA(work, 0, layout.ka, 0);
    }
    if (tiling_->enlarge == 1U && layout.bFullLoad && bMiss) {
        DirectDenseCopyB(work, 0, layout.kb, layout.bBase);
        cachedBGroup = work.groupStart;
        cachedBChannel = work.channelStart;
    }
    for (uint32_t c = 0; c < Std::min(layout.chunks, POLYPHASE_DOUBLE_BUFFER); ++c) {
        if (!layout.aFullLoad) {
            DirectDenseCopyA(work, c * layout.chunk, layout.ka, c * layout.aBytes);
        }
        if (!layout.bFullLoad) {
            DirectDenseCopyB(work, c * layout.chunk, layout.kb, layout.bBase + c * layout.bBytes);
        }
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(c));
    }
}

__aicore__ inline void ComputeDirectDenseKTile(const DirectDenseL1Layout& layout, uint32_t bank, uint32_t begin,
                                               uint32_t kk, uint32_t iteration, LocalTensor<float>& acc)
{
    const event_t lane = static_cast<event_t>(iteration & 1);
    WaitFlag<HardEvent::M_MTE1>(lane);
    DirectDenseLoadL0(layout.aFullLoad ? 0 : bank * layout.aBytes,
                      layout.bBase + (layout.bFullLoad ? 0 : bank * layout.bBytes), layout.aFullLoad ? kk : kk - begin,
                      layout.bFullLoad ? kk : kk - begin, lane);
    SetFlag<HardEvent::MTE1_M>(lane);
    WaitFlag<HardEvent::MTE1_M>(lane);
    LocalTensor<T> a2(TPosition::A2, static_cast<uint32_t>(lane) * layout.m * layout.k0 * sizeof(T),
                      layout.m * layout.k0);
    LocalTensor<T> b2(TPosition::B2, static_cast<uint32_t>(lane) * layout.n * layout.k0 * sizeof(T),
                      layout.n * layout.k0);
    ComputeMatmul(acc, a2, b2, layout.m, layout.n, layout.k0, iteration == 0);
    SetFlag<HardEvent::M_MTE1>(lane);
}

__aicore__ inline void ComputeDirectDenseChunk(const DirectDenseWork& work, const DirectDenseL1Layout& layout,
                                               uint32_t c, uint32_t& iteration, LocalTensor<float>& acc)
{
    const uint32_t bank = c & 1;
    const uint32_t begin = c * layout.chunk;
    const uint32_t end = Std::min(begin + layout.chunk, static_cast<uint32_t>(tiling_->kValueSegment));
    WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(bank));
    for (uint32_t kk = begin; kk < end; kk += layout.k0, ++iteration) {
        ComputeDirectDenseKTile(layout, bank, begin, kk, iteration, acc);
    }
    if (c + POLYPHASE_DOUBLE_BUFFER < layout.chunks) {
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(bank));
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(bank));
        if (!layout.aFullLoad) {
            DirectDenseCopyA(work, (c + POLYPHASE_DOUBLE_BUFFER) * layout.chunk, layout.ka, bank * layout.aBytes);
        }
        if (!layout.bFullLoad) {
            DirectDenseCopyB(work, (c + POLYPHASE_DOUBLE_BUFFER) * layout.chunk, layout.kb,
                             layout.bBase + bank * layout.bBytes);
        }
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(bank));
    }
}

template <bool dualAiv>
__aicore__ inline void StoreDirectDenseAccumulator(const DirectDenseL1Layout& layout, uint32_t outBank,
                                                   LocalTensor<float>& acc)
{
    const uint32_t matmulBytes = layout.m * layout.n * sizeof(float);
    SetFlag<HardEvent::M_FIX>(0);
    WaitFlag<HardEvent::M_FIX>(0);
    LocalTensor<float> matmulOut(TPosition::VECIN, outBank * matmulBytes, layout.m * layout.n);
    DataMove::L0cToUb(matmulOut, acc, layout.m, layout.n, layout.m, layout.m, QuantMode_t::NoQuant);
    if constexpr (dualAiv) {
        DataMove::L0cToUb<float, true>(matmulOut, acc, layout.m, layout.n, layout.m, layout.m, QuantMode_t::NoQuant);
    }
    SetFlag<HardEvent::FIX_M>(0);
    WaitFlag<HardEvent::FIX_M>(0);
}

template <bool dualAiv>
__aicore__ inline void DirectCubeTask(const DirectDenseWork& work, uint32_t ordinal, uint32_t& cachedBGroup,
                                      uint32_t& cachedBChannel)
{
    const auto layout = GetDirectDenseL1Layout();
    const uint32_t outBank = ordinal & 1;
    CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(outBank);
    if constexpr (dualAiv) {
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(POLYPHASE_AIV1_FLAG_BASE + outBank);
    }
    WaitFlag<HardEvent::MTE1_MTE2>(POLYPHASE_DIRECT_L1_EVENT);
    if (work.inputWidth) {
        PrepareDirectDenseTask(work, layout, cachedBGroup, cachedBChannel);
        uint32_t iteration = 0;
        LocalTensor<float> acc(TPosition::CO1, 0, layout.m * layout.n);
        for (uint32_t c = 0; c < layout.chunks; ++c) {
            ComputeDirectDenseChunk(work, layout, c, iteration, acc);
        }
        StoreDirectDenseAccumulator<dualAiv>(layout, outBank, acc);
    }
    SetFlag<HardEvent::MTE1_MTE2>(POLYPHASE_DIRECT_L1_EVENT);
    CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(outBank);
    if constexpr (dualAiv) {
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(POLYPHASE_AIV1_FLAG_BASE + outBank);
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(POLYPHASE_OUTPUT_STORE_DONE);
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(POLYPHASE_AIV1_FLAG_BASE + POLYPHASE_OUTPUT_STORE_DONE);
    }
}

template <PolyphaseRole role, bool dualAiv>
__aicore__ inline void DirectVectorTask(DirectDenseWork work, uint32_t ordinal, uint32_t& cachedBGroup,
                                        uint32_t& cachedBChannel)
{
    const uint32_t matmulBytes = tiling_->baseM * tiling_->baseN * sizeof(float);
    const uint32_t yBytes = Ops::Base::CeilAlign<uint32_t>(
        tiling_->singleCoreCin * Ops::Base::CeilAlign<uint32_t>(tiling_->singleCoreM, POLYPHASE_C0) * sizeof(T),
        Ops::Base::GetUbBlockSize());
    const uint32_t biasOffset = POLYPHASE_DOUBLE_BUFFER * matmulBytes + POLYPHASE_DOUBLE_BUFFER * yBytes;
    const uint32_t bank = ordinal & 1;
    if constexpr (role == PolyphaseRole::OUTPUT) {
        DirectDensePrepareGroupB(work, biasOffset, cachedBGroup, cachedBChannel);
    }
    uint32_t cBase = 0U;
    if constexpr (dualAiv) {
        const uint32_t half = Ops::Base::CeilDiv(work.channelCount, static_cast<uint32_t>(GetTaskRatio()));
        cBase = role == PolyphaseRole::FILTER ? half : 0U;
        work.channelCount = role == PolyphaseRole::FILTER ? work.channelCount - half : half;
        work.channelStart += cBase;
    }
    CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_V>(bank);
    if (!dualAiv || work.channelCount) {
        DirectDenseBias(work, biasOffset);
        if (!work.inputWidth) {
            LocalTensor<float> matmulOut(TPosition::VECIN, bank * matmulBytes, tiling_->baseM * tiling_->baseN);
            Duplicate(matmulOut, 0.0f, matmulOut.GetSize());
            PipeBarrier<PIPE_V>();
        }
        DirectDenseEpilogue(work, cBase, bank * matmulBytes, POLYPHASE_DOUBLE_BUFFER * matmulBytes + bank * yBytes,
                            biasOffset);
    }
    if constexpr (dualAiv && role == PolyphaseRole::FILTER) {
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(POLYPHASE_OUTPUT_STORE_DONE);
    }
    if (!dualAiv || work.channelCount) {
        DirectDenseStore(work, POLYPHASE_DOUBLE_BUFFER * matmulBytes + bank * yBytes);
    }
    CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(bank);
    if constexpr (dualAiv && role == PolyphaseRole::OUTPUT) {
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(POLYPHASE_OUTPUT_STORE_DONE);
    }
}

template <PolyphaseRole role>
__aicore__ inline void InitDirectDenseSync() const
{
    if constexpr (role == PolyphaseRole::CUBE) {
        SetFlag<HardEvent::MTE1_MTE2>(POLYPHASE_DIRECT_L1_EVENT);
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_MTE1>(1);
    } else {
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(0);
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(1);
    }
}

template <PolyphaseRole role, bool dualAiv>
__aicore__ inline void DrainDirectDenseSync() const
{
    if constexpr (role == PolyphaseRole::CUBE) {
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(1);
        WaitFlag<HardEvent::MTE1_MTE2>(POLYPHASE_DIRECT_L1_EVENT);
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(0);
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(1);
        if constexpr (dualAiv) {
            CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(POLYPHASE_AIV1_FLAG_BASE);
            CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(POLYPHASE_AIV1_FLAG_BASE + 1U);
        }
    }
}

template <PolyphaseRole role, bool dualAiv>
__aicore__ inline void ProcessDirectDense()
{
    const uint32_t block = GetAicBlockIdx();
    const uint64_t tasks = DirectDenseTasks();
    if (block >= tiling_->coreNum || block >= tasks) {
        return;
    }
    if constexpr (!dualAiv && role == PolyphaseRole::FILTER) {
        return;
    }
    uint32_t cachedBGroup = tiling_->oriGroup;
    uint32_t cachedBChannel = tiling_->cin;
    InitDirectDenseSync<role>();
    uint32_t ordinal = 0;
    // DIRECT 各角色共用唯一任务遍历。
    for (uint64_t task = block; task < tasks; task += tiling_->coreNum, ++ordinal) {
        const auto work = DirectDenseGeometry(task);
        if constexpr (role == PolyphaseRole::CUBE) {
            DirectCubeTask<dualAiv>(work, ordinal, cachedBGroup, cachedBChannel);
        } else {
            DirectVectorTask<role, dualAiv>(work, ordinal, cachedBGroup, cachedBChannel);
        }
    }
    DrainDirectDenseSync<role, dualAiv>();
}

#endif // CONV_BP_INPUT_SUB_FUNC_POLYPHASE_DIRECT_DENSE_H
