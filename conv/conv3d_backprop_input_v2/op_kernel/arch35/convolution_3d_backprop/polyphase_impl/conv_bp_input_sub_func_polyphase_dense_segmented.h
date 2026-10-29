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
 * \file conv_bp_input_sub_func_polyphase_dense_segmented.h
 * \brief PHASED dense 分段计算与输出。
 */

#ifndef CONV_BP_INPUT_SUB_FUNC_POLYPHASE_DENSE_SEGMENTED_H
#define CONV_BP_INPUT_SUB_FUNC_POLYPHASE_DENSE_SEGMENTED_H

struct DenseSegmentedOutputGeometry {
    uint32_t begin = 0U;
    uint32_t width = 0U;
    uint32_t fullRows = 0U;
    uint32_t tailWidth = 0U;
};

__aicore__ inline void DenseSegmentedPrepareMemory()
{
    const uint64_t m = tiling_->baseM;
    const uint64_t phases = tiling_->singleCoreGroup * POLYPHASE_C0 * m;
    memoryLayout_.a1Bytes = Ops::Base::CeilAlign<uint64_t>(
        sizeof(T) * tiling_->singleCoreCout * (m + tiling_->stepKa - 1U), Ops::Base::GetUbBlockSize());
    memoryLayout_.phaseB1Bytes = (POLYPHASE_C0 * sizeof(T)) * tiling_->singleCoreCout * tiling_->stepKa;
    memoryLayout_.b1BankBytes = memoryLayout_.phaseB1Bytes;
    memoryLayout_.phaseOutputOffsetBytes = phases * sizeof(float);
    memoryLayout_.outputOffsetBytes = phases * (sizeof(float) + sizeof(T));
    memoryLayout_.biasOffsetBytes = memoryLayout_.outputOffsetBytes + (POLYPHASE_C0 * sizeof(T)) * m * POLYPHASE_C0;
    memoryLayout_.outputBankPitchBytes = Ops::Base::CeilAlign<uint64_t>(
        memoryLayout_.biasOffsetBytes + (POLYPHASE_C0 * sizeof(float)), Ops::Base::GetUbBlockSize());
    memoryLayout_.indexOffsetBytes = POLYPHASE_DOUBLE_BUFFER * memoryLayout_.outputBankPitchBytes;
}

__aicore__ inline DenseSegmentedOutputGeometry DenseSegmentedOutput(const PolyphaseWorkGeometry& geometry,
                                                                    uint32_t group) const
{
    DenseSegmentedOutputGeometry output;
    output.begin = group * tiling_->singleCoreGroup;
    output.width = Std::min(tiling_->singleCoreGroup, tiling_->stepKb - output.begin);
    output.width = Std::min(output.width, tiling_->wi - output.begin);
    const uint64_t first = geometry.uStart * tiling_->stepKb + output.begin;
    if (first >= tiling_->wi) {
        return output;
    }
    const uint64_t remaining = static_cast<uint64_t>(tiling_->wi) - first;
    if (remaining >= output.width) {
        output.fullRows = Std::min(geometry.curJ,
                                   static_cast<uint32_t>(1U + (remaining - output.width) / tiling_->stepKb));
    }
    if (output.fullRows < geometry.curJ && remaining > static_cast<uint64_t>(output.fullRows) * tiling_->stepKb) {
        output.tailWidth = Std::min(
            output.width, static_cast<uint32_t>(remaining - static_cast<uint64_t>(output.fullRows) * tiling_->stepKb));
    }
    return output;
}

__aicore__ inline void DenseSegmentedLoadA1(const PolyphaseWorkGeometry& geometry, LocalTensor<T>& a1) const
{
    const uint64_t offset = static_cast<uint64_t>(geometry.batchIdx) * tiling_->cout * tiling_->wo;
    DataMove::GmToL1Window(a1, dedyGm_[offset], geometry.xStart, geometry.localA1W, tiling_->wo, tiling_->cout);
}

__aicore__ inline bool DenseSegmentedBulkPackEligible(const PolyphaseWorkGeometry& geometry,
                                                      const SegmentedPhaseRange& range, uint64_t first,
                                                      uint32_t& liveTap) const
{
    // 唯一调用来自 first < range.end 的分段循环，tap 范围非负。
    const uint64_t span = range.end - first;
    const uint64_t cap = tiling_->stepKa;
    liveTap = static_cast<uint32_t>(span < cap ? span : cap);
    const uint64_t bytes = memoryLayout_.phaseB1Bytes;
    const uint64_t limit = 1ULL << POLYPHASE_NDDMA_GM_STRIDE_BITS;
    const uint64_t stride = tiling_->stepKb;
    const uint64_t kernel = tiling_->wk;
    const uint64_t channelStride = static_cast<uint64_t>(tiling_->cin) * kernel;
    if (liveTap < POLYPHASE_C0 || tiling_->polyphaseBulkFilter == 0U ||
        bytes / sizeof(T) > POLYPHASE_GATHER_INDEX_LIMIT || stride >= limit || kernel >= limit ||
        channelStride >= limit) {
        return false;
    }
    const uint64_t sourceSpan = (static_cast<uint64_t>(tiling_->cout) - 1U) * channelStride +
                                (static_cast<uint64_t>(geometry.curN) - 1U) * kernel +
                                (static_cast<uint64_t>(liveTap) - 1U) * stride + 1U;
    return sizeof(T) * sourceSpan <= limit;
}

__simd_vf__ inline static void DenseSegmentedReversePackVF(__ubuf__ T* raw, __ubuf__ T* nz, uint32_t channels,
                                                           uint32_t cap)
{
    constexpr uint32_t c0Bits = 4U;
    Reg::RegTensor<uint32_t> lane;
    Reg::RegTensor<uint32_t> n;
    Reg::RegTensor<uint32_t> c0;
    Reg::RegTensor<uint32_t> product;
    Reg::RegTensor<uint32_t> index;
    Reg::RegTensor<uint16_t> packed;
    Reg::RegTensor<uint16_t> indices;
    Reg::RegTensor<uint16_t> zero;
    Reg::RegTensor<T> values;
    auto all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    auto all16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    uint32_t count = (Ops::Base::GetVRegSize() / sizeof(float));
    auto half = Reg::UpdateMask<uint16_t>(count);
    Reg::Duplicate(zero, static_cast<uint16_t>(0), all16);
    for (uint32_t c1 = 0U; c1 < channels / POLYPHASE_C0; ++c1) {
        for (uint32_t h = 0U; h < cap; ++h) {
            for (uint32_t offset = 0U; offset < (POLYPHASE_C0 * POLYPHASE_C0);
                 offset += (Ops::Base::GetVRegSize() / sizeof(float))) {
                Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), static_cast<int32_t>(offset));
                Reg::ShiftRights(n, lane, static_cast<int16_t>(c0Bits), all);
                Reg::Muls(product, n, POLYPHASE_C0, all);
                Reg::Sub(c0, lane, product, all);
                Reg::Muls(index, c0, POLYPHASE_C0, all);
                Reg::Adds(index, index, c1 * (POLYPHASE_C0 * POLYPHASE_C0), all);
                Reg::Add(index, index, n, all);
                Reg::Muls(index, index, cap, all);
                Reg::Adds(index, index, cap - 1U - h, all);
                // 选择该打包分支前，host 已检查 UB 和 Gather 范围。
                // 所有无效 Gather lane 均显式指向 raw 的第零个元素。
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(packed, index);
                Reg::Select(indices, packed, zero, half);
                Reg::Gather(values, raw, indices, half);
                Reg::StoreAlign(nz + (c1 * cap + h) * (POLYPHASE_C0 * POLYPHASE_C0) + offset, values, half);
            }
        }
    }
}

__aicore__ inline void DenseSegmentedPackBBulk(const PolyphaseWorkGeometry& geometry, const SegmentedPhaseRange& range,
                                               uint64_t first, uint32_t bank, uint32_t liveTap) const
{
    const uint32_t cap = tiling_->stepKa;
    const uint32_t elements = static_cast<uint32_t>(memoryLayout_.phaseB1Bytes / sizeof(T));
    LocalTensor<T> nz(TPosition::VECIN, 0, elements);
    LocalTensor<T> raw(TPosition::VECIN, memoryLayout_.phaseB1Bytes, elements);
    Duplicate<T>(raw, static_cast<T>(0), elements);
    const event_t event = static_cast<event_t>(0);
    SetFlag<HardEvent::V_MTE2>(event);
    WaitFlag<HardEvent::V_MTE2>(event);
    const uint64_t gm = static_cast<uint64_t>(geometry.nStart) * tiling_->wk + range.k0 + first * tiling_->stepKb;
    DataMove::GmToUb<POLYPHASE_NDDMA_3D>(
        raw, filterGm_[gm],
        {{liveTap, tiling_->stepKb, 1U},
         {geometry.curN, tiling_->wk, cap},
         {tiling_->cout, static_cast<uint64_t>(tiling_->cin) * tiling_->wk, POLYPHASE_C0 * cap}});
    SetFlag<HardEvent::MTE2_V>(event);
    WaitFlag<HardEvent::MTE2_V>(event);
    DenseSegmentedReversePackVF(reinterpret_cast<__ubuf__ T*>(raw.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ T*>(nz.GetPhyAddr()), tiling_->singleCoreCout, cap);
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    LocalTensor<T> b1(TPosition::B1, memoryLayout_.a1Bytes + bank * memoryLayout_.phaseB1Bytes, elements);
    DataMove::UbToL1(b1, nz, elements);
    SetFlag<HardEvent::MTE3_V>(event);
    WaitFlag<HardEvent::MTE3_V>(event);
}

__aicore__ inline void CopyDenseSegmentTaps(const PolyphaseWorkGeometry& geometry, const SegmentedPhaseRange& range,
                                            uint64_t first, LocalTensor<T>& nz) const
{
    const uint32_t cap = tiling_->stepKa;
    const uint32_t full = tiling_->cout / POLYPHASE_C0;
    const uint32_t tail = tiling_->cout % POLYPHASE_C0;
    const uint64_t channelStride = static_cast<uint64_t>(tiling_->cin) * tiling_->wk;
    for (uint32_t h = 0U; h < cap; ++h) {
        const uint64_t tap = first + cap - 1U - h;
        if (tap >= range.end) {
            continue;
        }
        const uint64_t k = range.k0 + tap * tiling_->stepKb;
        const uint64_t gm = static_cast<uint64_t>(geometry.nStart) * tiling_->wk + k;
        if (full != 0U) {
            DataMove::GmToUb<POLYPHASE_NDDMA_3D>(
                nz[h * (POLYPHASE_C0 * POLYPHASE_C0)], filterGm_[gm],
                {{POLYPHASE_C0, channelStride, 1U},
                 {geometry.curN, tiling_->wk, POLYPHASE_C0},
                 {full, POLYPHASE_C0 * channelStride, cap * (POLYPHASE_C0 * POLYPHASE_C0)}});
        }
        if (tail != 0U) {
            DataMove::GmToUb<POLYPHASE_NDDMA_3D>(
                nz[(full * cap + h) * (POLYPHASE_C0 * POLYPHASE_C0)],
                filterGm_[gm + static_cast<uint64_t>(full) * POLYPHASE_C0 * channelStride],
                {{tail, channelStride, 1U},
                 {geometry.curN, tiling_->wk, POLYPHASE_C0},
                 {1U, POLYPHASE_C0 * channelStride, cap * (POLYPHASE_C0 * POLYPHASE_C0)}});
        }
    }
}

__aicore__ inline void DenseSegmentedPackB(const PolyphaseWorkGeometry& geometry, const SegmentedPhaseRange& range,
                                           uint64_t first, uint32_t bank) const
{
    uint32_t liveTap = 0U;
    if (DenseSegmentedBulkPackEligible(geometry, range, first, liveTap)) {
        DenseSegmentedPackBBulk(geometry, range, first, bank, liveTap);
        return;
    }
    const uint32_t elements = static_cast<uint32_t>(memoryLayout_.phaseB1Bytes / sizeof(T));
    LocalTensor<T> nz(TPosition::VECIN, 0, elements);
    Duplicate<T>(nz, static_cast<T>(0), elements);
    const event_t event = static_cast<event_t>(0);
    SetFlag<HardEvent::V_MTE2>(event);
    WaitFlag<HardEvent::V_MTE2>(event);
    CopyDenseSegmentTaps(geometry, range, first, nz);
    SetFlag<HardEvent::MTE2_MTE3>(event);
    WaitFlag<HardEvent::MTE2_MTE3>(event);
    LocalTensor<T> b1(TPosition::B1, memoryLayout_.a1Bytes + bank * memoryLayout_.phaseB1Bytes, elements);
    DataMove::UbToL1(b1, nz, elements);
    SetFlag<HardEvent::MTE3_V>(event);
    WaitFlag<HardEvent::MTE3_V>(event);
}

__aicore__ inline void DenseSegmentedLoadL0(const PolyphaseWorkGeometry& geometry, uint32_t channel,
                                            uint32_t curChannels, const LocalTensor<T>& a1, const LocalTensor<T>& b1,
                                            LocalTensor<T>& a2, LocalTensor<T>& b2) const
{
    DataMove::L1ToL0A(a2, a1,
                      {geometry.localA1W, tiling_->singleCoreCout, tiling_->stepKa, 0U, tiling_->baseM,
                       channel * tiling_->stepKa, curChannels * tiling_->stepKa, tiling_->baseM / POLYPHASE_C0});
    DataMove::L1ToL0B(b2, b1, 1U, 1U, channel * tiling_->stepKa / POLYPHASE_C0,
                      curChannels * tiling_->stepKa / POLYPHASE_C0);
}

// PHASED 调度器负责 tap 迭代，此处仅计算一个 dense 分段。
__aicore__ inline void DenseSegment(PolyphaseWorkGeometry geometry, const PhaseSegment& segment, uint64_t& kOrdinal,
                                    bool& firstK) const
{
    const uint32_t m = tiling_->baseM;
    const uint32_t cap = tiling_->stepKa;
    const uint32_t channelTile = tiling_->baseK / cap;
    const uint32_t a2Elements = m * tiling_->baseK;
    const uint32_t b2Elements = POLYPHASE_C0 * tiling_->baseK;
    LocalTensor<T> a1(TPosition::A1, 0, memoryLayout_.a1Bytes / sizeof(T));
    LocalTensor<float> c0(TPosition::CO1, 0, m * POLYPHASE_C0);
    const uint32_t bBank = segment.bank;
    geometry.xStart = segment.xStart;
    WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(0));
    DenseSegmentedLoadA1(geometry, a1);
    SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
    CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_MTE1>(POLYPHASE_AIV1_FLAG_BASE + bBank);
    LocalTensor<T> b1(TPosition::B1, memoryLayout_.a1Bytes + bBank * memoryLayout_.phaseB1Bytes,
                      memoryLayout_.phaseB1Bytes / sizeof(T));
    for (uint32_t channel = 0U; channel < tiling_->singleCoreCout; channel += channelTile) {
        const uint32_t cur = Std::min(channelTile, tiling_->singleCoreCout - channel);
        const uint32_t bank = static_cast<uint32_t>(kOrdinal++ & 1U);
        const event_t event = static_cast<event_t>(bank);
        WaitFlag<HardEvent::M_MTE1>(event);
        LocalTensor<T> a2(TPosition::A2, bank * a2Elements * sizeof(T), a2Elements);
        LocalTensor<T> b2(TPosition::B2, bank * b2Elements * sizeof(T), b2Elements);
        DenseSegmentedLoadL0(geometry, channel, cur, a1, b1, a2, b2);
        SetFlag<HardEvent::MTE1_M>(event);
        WaitFlag<HardEvent::MTE1_M>(event);
        ComputeMatmul(c0, a2, b2, static_cast<uint16_t>(m), POLYPHASE_C0, static_cast<uint16_t>(cur * cap), firstK);
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(event);
        firstK = false;
    }
    SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(0));
    CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE1>(POLYPHASE_AIV1_FLAG_BASE + bBank);
}

__simd_vf__ inline static void DenseSegmentedOutputIndex(__ubuf__ uint16_t* index, uint32_t m, uint32_t width)
{
    constexpr uint32_t c0Bits = 4U;
    Reg::RegTensor<uint32_t> lane;
    Reg::RegTensor<uint32_t> u;
    Reg::RegTensor<uint32_t> phase;
    Reg::RegTensor<uint32_t> product;
    Reg::RegTensor<uint32_t> result;
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint16_t> packed;
    auto all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    uint32_t count = (Ops::Base::GetVRegSize() / sizeof(float));
    auto half = Reg::UpdateMask<uint16_t>(count);
    Reg::Duplicate(zero, 0U, all);
    for (uint32_t offset = 0U; offset < m * POLYPHASE_C0; offset += (Ops::Base::GetVRegSize() / sizeof(float))) {
        Reg::Arange<int32_t>(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane), static_cast<int32_t>(offset));
        Reg::ShiftRights(u, lane, static_cast<int16_t>(c0Bits), all);
        Reg::Muls(product, u, POLYPHASE_C0, all);
        Reg::Sub(phase, lane, product, all);
        Reg::Muls(result, phase, POLYPHASE_C0 * m, all);
        Reg::Add(result, result, u, all);
        Reg::MaskReg valid;
        Reg::Compares<uint32_t, CMPMODE::LT>(valid, phase, width, all);
        Reg::Select(result, result, zero, valid);
        // host 约束 phase/M 范围；padding lane 在收窄前置为索引零。
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(packed, result);
        Reg::StoreAlign(index + offset, packed, half);
    }
}

__simd_vf__ inline static void DenseSegmentedGather(__ubuf__ uint16_t* index, __ubuf__ T* source,
                                                    __ubuf__ T* destination, uint32_t channels, uint32_t m)
{
    Reg::RegTensor<uint16_t> indices;
    Reg::RegTensor<T> values;
    auto all = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    for (uint32_t offset = 0U; offset < m * POLYPHASE_C0; offset += (Ops::Base::GetVRegSize() / sizeof(T))) {
        Reg::LoadAlign(indices, index + offset);
        for (uint32_t c = 0U; c < channels; ++c) {
            Reg::Gather(values, source + c * m, indices, all);
            Reg::StoreAlign(destination + c * m * POLYPHASE_C0 + offset, values, all);
        }
    }
}

__aicore__ inline void DenseSegmentedLoadBias(const PolyphaseWorkGeometry& geometry, uint32_t bank) const
{
    LocalTensor<float> bias(TPosition::VECIN, bank * memoryLayout_.outputBankPitchBytes + memoryLayout_.biasOffsetBytes,
                            POLYPHASE_C0);
    Duplicate<float>(bias, 0.0F, POLYPHASE_C0);
    SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(0));
    WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(0));
    const uint8_t rightPadding = static_cast<uint8_t>(((Ops::Base::GetUbBlockSize() / sizeof(float)) -
                                                       geometry.curN % (Ops::Base::GetUbBlockSize() / sizeof(float))) %
                                                      (Ops::Base::GetUbBlockSize() / sizeof(float)));
    DataMove::GmToUbVector(bias, biasGm_[geometry.nStart], geometry.curN, rightPadding);
    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
}

__aicore__ inline void DenseSegmentedStore(const PolyphaseWorkGeometry& geometry,
                                           const DenseSegmentedOutputGeometry& output, uint32_t bank) const
{
    if (output.fullRows == 0U && output.tailWidth == 0U) {
        return;
    }
    const uint32_t m = tiling_->baseM;
    const uint32_t phases = tiling_->singleCoreGroup;
    const uint32_t row = m * POLYPHASE_C0;
    const uint32_t offset = bank * memoryLayout_.outputBankPitchBytes;
    LocalTensor<float> fp32(TPosition::VECIN, offset, phases * POLYPHASE_C0 * m);
    LocalTensor<T> phaseOutput(TPosition::VECIN, offset + memoryLayout_.phaseOutputOffsetBytes,
                               phases * POLYPHASE_C0 * m);
    LocalTensor<float> bias(TPosition::VECIN, offset + memoryLayout_.biasOffsetBytes, POLYPHASE_C0);
    LocalTensor<T> dst(TPosition::VECIN, offset + memoryLayout_.outputOffsetBytes, POLYPHASE_C0 * row);
    LocalTensor<uint16_t> index(TPosition::VECIN, memoryLayout_.indexOffsetBytes, row);
    for (uint32_t p = 0U; p < phases; ++p) {
        const auto range = SegmentedRange(geometry, output.begin + p);
        if (p >= output.width || range.first >= range.end) {
            Duplicate<float>(fp32[p * POLYPHASE_C0 * m], 0.0F, POLYPHASE_C0 * m);
        }
    }
    DenseSegmentedLoadBias(geometry, bank);
    PipeBarrier<PIPE_V>();
    Convolution3DBackpropFunc::PolyphaseAddBiasCast(
        reinterpret_cast<__ubuf__ float*>(fp32.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(bias.GetPhyAddr()),
        reinterpret_cast<__ubuf__ T*>(phaseOutput.GetPhyAddr()), static_cast<uint16_t>(phases),
        static_cast<uint16_t>(geometry.curN), POLYPHASE_C0, static_cast<uint16_t>(m));
    DenseSegmentedOutputIndex(reinterpret_cast<__ubuf__ uint16_t*>(index.GetPhyAddr()), m, output.width);
    PipeBarrier<PIPE_V>();
    DenseSegmentedGather(reinterpret_cast<__ubuf__ uint16_t*>(index.GetPhyAddr()),
                         reinterpret_cast<__ubuf__ T*>(phaseOutput.GetPhyAddr()),
                         reinterpret_cast<__ubuf__ T*>(dst.GetPhyAddr()), geometry.curN, m);
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    StoreSegmentedRows(geometry, output, dst, POLYPHASE_C0, row, false);
    SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
}

#endif // CONV_BP_INPUT_SUB_FUNC_POLYPHASE_DENSE_SEGMENTED_H
