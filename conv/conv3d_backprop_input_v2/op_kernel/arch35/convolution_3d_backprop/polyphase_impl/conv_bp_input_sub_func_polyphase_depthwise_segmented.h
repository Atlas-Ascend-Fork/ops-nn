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
 * \file conv_bp_input_sub_func_polyphase_depthwise_segmented.h
 * \brief PHASED depthwise 串行和并行分段计算。
 */

#ifndef CONV_BP_INPUT_SUB_FUNC_POLYPHASE_DEPTHWISE_SEGMENTED_H
#define CONV_BP_INPUT_SUB_FUNC_POLYPHASE_DEPTHWISE_SEGMENTED_H

struct SegmentedPhaseRange {
    uint64_t first = 0U;
    uint64_t end = 0U;
    uint32_t k0 = 0U;
    int64_t inputShift = 0;
};

struct SegmentedOutputGeometry {
    uint64_t firstW;
    uint32_t validW;
    uint32_t phases;
    bool allZero;
};

struct SegmentedPhaseIterator {
    uint32_t current;
    uint32_t end;
    uint32_t next;
    uint32_t nextEnd;
};

__aicore__ inline SegmentedOutputGeometry SegmentedOutput(const PolyphaseWorkGeometry& geometry) const
{
    SegmentedOutputGeometry output;
    const uint64_t stride = tiling_->stepKb;
    output.firstW = geometry.uStart * stride;
    const uint64_t extent = stride * geometry.curJ;
    const uint64_t remaining = static_cast<uint64_t>(tiling_->wi) - output.firstW;
    // host 的 stride/M 约束与任务几何保证有效行宽可安全收窄。
    output.validW = static_cast<uint32_t>(extent < remaining ? extent : remaining);
    output.phases = Std::min(tiling_->stepKb, output.validW);
    const uint64_t left = output.firstW + static_cast<uint64_t>(tiling_->padLeft);
    const uint64_t right = left + output.validW - 1U;
    const uint64_t lastTap = static_cast<uint64_t>(tiling_->wk) - 1U;
    const uint64_t firstInput = left <= lastTap ? 0U : (left - lastTap + stride - 1U) / stride;
    const uint64_t lastInput = right / stride < static_cast<uint64_t>(tiling_->wo) - 1U ?
                                   right / stride :
                                   static_cast<uint64_t>(tiling_->wo) - 1U;
    output.allZero = firstInput > lastInput;
    return output;
}

__aicore__ inline SegmentedPhaseIterator SegmentedPhases(uint32_t phases) const
{
    const uint32_t stride = tiling_->stepKb;
    const uint32_t length = Std::min(tiling_->wk, stride);
    if (length == stride) {
        return {0U, phases, 0U, 0U};
    }
    const uint32_t residue = static_cast<uint32_t>(static_cast<uint64_t>(tiling_->padLeft) % stride);
    const uint32_t first = (stride - residue) % stride;
    const uint32_t end = first + length;
    if (end > stride) {
        return {0U, Std::min(end - stride, phases), Std::min(first, phases), phases};
    }
    return {Std::min(first, phases), Std::min(end, phases), 0U, 0U};
}

__aicore__ inline static bool SegmentedNextPhase(SegmentedPhaseIterator& iterator, uint32_t& phase)
{
    if (iterator.current >= iterator.end) {
        iterator.current = iterator.next;
        iterator.end = iterator.nextEnd;
        iterator.next = iterator.nextEnd;
    }
    if (iterator.current >= iterator.end) {
        return false;
    }
    phase = iterator.current++;
    return true;
}

__simd_vf__ inline static void SegmentedGatherOutput(__ubuf__ uint16_t* index, __ubuf__ T* src, __ubuf__ T* dst,
                                                     uint16_t channels, uint16_t m, uint16_t row, uint16_t validW)
{
    Reg::RegTensor<uint16_t> indices;
    Reg::RegTensor<uint16_t> zeroIndex;
    Reg::RegTensor<T> values;
    Reg::RegTensor<T> zero;
    auto all = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(zeroIndex, static_cast<uint16_t>(0), all);
    Reg::Duplicate(zero, T(0), all);
    for (uint16_t c = 0U; c < channels; ++c) {
        uint32_t remaining = validW;
        for (uint32_t offset = 0U; offset < row; offset += (Ops::Base::GetVRegSize() / sizeof(T))) {
            auto valid = Reg::UpdateMask<T>(remaining);
            Reg::LoadAlign(indices, index + offset);
            // 无效 lane 也指向已初始化 phase0，不依赖掩码禁止访问。
            Reg::Select(indices, indices, zeroIndex, valid);
            Reg::Gather(values, src + static_cast<uint32_t>(c) * m, indices, valid);
            Reg::Select(values, values, zero, valid);
            Reg::StoreAlign(dst + static_cast<uint32_t>(c) * row + offset, values, all);
        }
    }
}

template <bool phaseParallel>
__aicore__ inline void PrepareDepthwiseSegmentedMemory()
{
    const uint64_t m = tiling_->baseM;
    const uint64_t cap = tiling_->stepKa;
    const uint64_t phaseWidth = phaseParallel ? tiling_->singleCoreGroup : tiling_->stepKb;
    const uint64_t phasePitch = phaseParallel ? Ops::Base::CeilAlign<uint64_t>(phaseWidth, POLYPHASE_C0) : phaseWidth;
    const uint64_t row = Ops::Base::CeilAlign<uint64_t>(m * phasePitch, (Ops::Base::GetVRegSize() / sizeof(T)));
    memoryLayout_.a1Bytes = (POLYPHASE_C0 * sizeof(T)) * (m + cap - 1U);
    memoryLayout_.phaseB1Bytes = (POLYPHASE_C0 * POLYPHASE_C0 * sizeof(T)) * cap;
    memoryLayout_.b1BankBytes = memoryLayout_.phaseB1Bytes;
    memoryLayout_.outputOffsetBytes = (POLYPHASE_C0 * sizeof(T)) * phaseWidth * m;
    memoryLayout_.indexOffsetBytes = memoryLayout_.outputOffsetBytes + (POLYPHASE_C0 * sizeof(T)) * row;
    memoryLayout_.outputBankPitchBytes = memoryLayout_.indexOffsetBytes + sizeof(T) * row;
}

__aicore__ inline SegmentedPhaseRange SegmentedRange(const PolyphaseWorkGeometry& geometry, uint32_t phase) const
{
    SegmentedPhaseRange range;
    const uint32_t stride = tiling_->stepKb;
    const uint64_t residue = phase + static_cast<uint64_t>(tiling_->padLeft) % stride;
    range.k0 = static_cast<uint32_t>(residue % stride);
    range.inputShift = static_cast<int64_t>(tiling_->padLeft) / stride + residue / stride;
    if (phase >= tiling_->wi || range.k0 >= tiling_->wk) {
        return range;
    }
    const uint64_t jr = 1U + (static_cast<uint64_t>(tiling_->wi) - 1U - phase) / stride;
    if (geometry.uStart >= jr) {
        return range;
    }
    const uint64_t mr = Std::min(tiling_->baseM, static_cast<uint32_t>(jr - geometry.uStart));
    const uint64_t q = 1U + (static_cast<uint64_t>(tiling_->wk) - 1U - range.k0) / stride;
    const int64_t lo = static_cast<int64_t>(geometry.uStart) + range.inputShift -
                       (static_cast<int64_t>(tiling_->wo) - 1);
    const int64_t hi = static_cast<int64_t>(geometry.uStart + mr) + range.inputShift;
    range.first = lo > 0 ? static_cast<uint64_t>(lo) : 0U;
    range.end = static_cast<uint64_t>(hi) < q ? static_cast<uint64_t>(hi) : q;
    return range;
}

__aicore__ inline void SegmentedPackFilter(const PolyphaseWorkGeometry& geometry, const SegmentedPhaseRange& range,
                                           uint64_t first, uint32_t length, uint32_t bank) const
{
    const uint64_t firstTap = range.k0 + first * tiling_->stepKb;
    const uint32_t bytes = (POLYPHASE_C0 * POLYPHASE_C0) * tiling_->stepKa * sizeof(T);
    // 按完整容量反向打包，短分段占据 [cap-length, cap)。
    PackDepthwiseFilter(geometry, firstTap, length, memoryLayout_.a1Bytes + bank * bytes);
}

__aicore__ inline void SegmentedStore(const PolyphaseWorkGeometry& geometry, const SegmentedOutputGeometry& output,
                                      uint32_t bank) const
{
    const uint32_t m = tiling_->baseM;
    const uint32_t stride = tiling_->stepKb;
    const uint32_t row = static_cast<uint32_t>(
        Ops::Base::CeilAlign<uint64_t>(stride * m, (Ops::Base::GetVRegSize() / sizeof(T))));
    const uint32_t offset = bank * memoryLayout_.outputBankPitchBytes;
    LocalTensor<T> phaseOut(TPosition::VECIN, offset, stride * POLYPHASE_C0 * m);
    LocalTensor<T> dst(TPosition::VECIN, offset + memoryLayout_.outputOffsetBytes, POLYPHASE_C0 * row);
    LocalTensor<uint16_t> index(TPosition::VECIN, memoryLayout_.indexOffsetBytes, row);
    const uint32_t rowPitch = static_cast<uint32_t>(
        Ops::Base::CeilAlign<uint64_t>(output.validW, (Ops::Base::GetVRegSize() / sizeof(T))));
    if (output.allZero) {
        // 所有有效通道共用只读对齐零行，不读取 phase 或索引。
        Duplicate<T>(dst, static_cast<T>(0),
                     static_cast<uint32_t>(Ops::Base::CeilAlign<uint64_t>(output.validW, POLYPHASE_C0)));
    } else {
        // output-ready 确保 Fixpipe 写入可见，再初始化空 phase。
        ClearDepthwiseZeroRuns(geometry, phaseOut, SegmentedPhases(output.phases), 0U, output.phases);
        PipeBarrier<PIPE_V>();
        SegmentedGatherOutput(reinterpret_cast<__ubuf__ uint16_t*>(index.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ T*>(phaseOut.GetPhyAddr()),
                              reinterpret_cast<__ubuf__ T*>(dst.GetPhyAddr()), static_cast<uint16_t>(geometry.curN),
                              static_cast<uint16_t>(m), static_cast<uint16_t>(rowPitch),
                              static_cast<uint16_t>(output.validW));
    }
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    for (uint32_t c = 0U; c < geometry.curN; ++c) {
        const uint64_t gm = (static_cast<uint64_t>(geometry.batchIdx) * tiling_->cin + geometry.nStart + c) *
                                tiling_->wi +
                            output.firstW;
        DataMove::UbToGm(yGm_[gm], dst[output.allZero ? 0U : c * rowPitch], 1U, output.validW, 0U, 0U);
    }
    SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
}

// 上方为串行 depthwise 路径，下方为并行 phase 的专用几何和输出。
// 两者共用分段调度，保持原策略值和同步协议。

__aicore__ inline void ClearDepthwiseZeroRuns(const PolyphaseWorkGeometry& geometry, const LocalTensor<T>& phaseOut,
                                              SegmentedPhaseIterator iterator, uint32_t begin, uint32_t end) const
{
    const uint32_t plane = POLYPHASE_C0 * tiling_->baseM;
    uint32_t zeroStart = begin;
    for (uint32_t phase; SegmentedNextPhase(iterator, phase);) {
        const auto range = SegmentedRange(geometry, phase);
        if (range.end > range.first) {
            if (phase > zeroStart) {
                Duplicate<T>(phaseOut[(zeroStart - begin) * plane], static_cast<T>(0), (phase - zeroStart) * plane);
            }
            zeroStart = phase + 1U;
        }
    }
    if (zeroStart < end) {
        Duplicate<T>(phaseOut[(zeroStart - begin) * plane], static_cast<T>(0), (end - zeroStart) * plane);
    }
}

struct SegmentedParallelOutputGeometry {
    uint32_t begin;
    uint32_t end;
    uint32_t width;
    uint32_t phases;
    uint32_t fullRows;
    uint32_t tailWidth;
    bool allZero;
};

struct SegmentedParallelA1Cache {
    bool valid = false;
    int64_t xStart = 0;
};

__aicore__ inline uint32_t SegmentedParallelBoundary(uint32_t group) const
{
    const uint32_t stride = tiling_->stepKb;
    const uint32_t groups = tiling_->dDim;
    if (group == 0U) {
        return 0U;
    }
    if (group == groups) {
        return stride;
    }
    const uint32_t active = Std::min(tiling_->wk, stride);
    const uint32_t ordinal = (group * active + groups - 1U) / groups;
    if (active == stride) {
        return ordinal;
    }
    const uint32_t first = (stride - static_cast<uint64_t>(tiling_->padLeft) % stride) % stride;
    const uint32_t wrapped = first + active > stride ? first + active - stride : 0U;
    return ordinal < wrapped ? ordinal : first + ordinal - wrapped;
}

__aicore__ inline SegmentedPhaseIterator SegmentedParallelPhases(const SegmentedParallelOutputGeometry& output) const
{
    auto iterator = SegmentedPhases(output.phases);
    iterator.current = iterator.current < output.begin ? output.begin : iterator.current;
    iterator.end = Std::min(iterator.end, output.end);
    iterator.next = iterator.next < output.begin ? output.begin : iterator.next;
    iterator.nextEnd = Std::min(iterator.nextEnd, output.end);
    return iterator;
}

__aicore__ inline SegmentedParallelOutputGeometry SegmentedParallelOutput(const PolyphaseWorkGeometry& geometry,
                                                                          uint32_t group) const
{
    SegmentedParallelOutputGeometry output;
    output.begin = SegmentedParallelBoundary(group);
    output.end = SegmentedParallelBoundary(group + 1U);
    output.width = output.end - output.begin;
    const uint64_t firstRow = geometry.uStart * tiling_->stepKb;
    output.phases = Std::min(output.end, static_cast<uint32_t>(static_cast<uint64_t>(tiling_->wi) - firstRow));
    output.fullRows = 0U;
    output.tailWidth = 0U;
    output.allZero = true;
    const uint64_t first = firstRow + output.begin;
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
    auto iterator = SegmentedParallelPhases(output);
    for (uint32_t phase; SegmentedNextPhase(iterator, phase);) {
        const auto range = SegmentedRange(geometry, phase);
        if (range.end > range.first) {
            output.allZero = false;
            break;
        }
    }
    return output;
}

__simd_vf__ inline static void SegmentedParallelGatherOutput(__ubuf__ uint16_t* index, __ubuf__ T* src, __ubuf__ T* dst,
                                                             uint16_t channels, uint16_t m, uint16_t q, uint16_t row,
                                                             uint16_t curJ, uint16_t width, uint16_t stride,
                                                             uint16_t validSpan)
{
    Reg::RegTensor<uint16_t> lane;
    Reg::RegTensor<uint16_t> u;
    Reg::RegTensor<uint16_t> v;
    Reg::RegTensor<uint16_t> product;
    Reg::RegTensor<uint16_t> globalOffset;
    Reg::RegTensor<uint16_t> divisor;
    Reg::RegTensor<uint16_t> indices;
    Reg::RegTensor<uint16_t> zeroIndex;
    Reg::RegTensor<T> values;
    Reg::RegTensor<T> zero;
    auto all = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(divisor, q, all);
    Reg::Duplicate(zeroIndex, static_cast<uint16_t>(0), all);
    Reg::Duplicate(zero, T(0), all);
    for (uint32_t offset = 0U; offset < row; offset += (Ops::Base::GetVRegSize() / sizeof(T))) {
        Reg::Arange<int16_t>(reinterpret_cast<Reg::RegTensor<int16_t>&>(lane), static_cast<int16_t>(offset));
        Reg::Div(u, lane, divisor, all);
        Reg::Muls(product, u, q, all);
        Reg::Sub(v, lane, product, all);
        Reg::Muls(globalOffset, u, stride, all);
        Reg::Add(globalOffset, globalOffset, v, all);
        Reg::MaskReg validU;
        Reg::MaskReg validV;
        Reg::MaskReg validW;
        Reg::MaskReg valid;
        Reg::Compares<uint16_t, CMPMODE::LT>(validU, u, curJ, all);
        Reg::Compares<uint16_t, CMPMODE::LT>(validV, v, width, all);
        Reg::Compares<uint16_t, CMPMODE::LT>(validW, globalOffset, validSpan, all);
        Reg::And(valid, validU, validV, all);
        Reg::And(valid, valid, validW, all);
        Reg::LoadAlign(indices, index + offset);
        // 无效 lane 也指向已初始化 phase0，不读取 padding phase。
        Reg::Select(indices, indices, zeroIndex, valid);
        for (uint16_t c = 0U; c < channels; ++c) {
            Reg::Gather(values, src + static_cast<uint32_t>(c) * m, indices, valid);
            Reg::Select(values, values, zero, valid);
            Reg::StoreAlign(dst + static_cast<uint32_t>(c) * row + offset, values, all);
        }
    }
}

__aicore__ inline void SegmentedParallelStore(const PolyphaseWorkGeometry& geometry,
                                              const SegmentedParallelOutputGeometry& output, uint32_t bank) const
{
    if (output.fullRows == 0U && output.tailWidth == 0U) {
        return;
    }
    const uint32_t m = tiling_->baseM;
    const uint32_t stride = tiling_->stepKb;
    const uint32_t q = static_cast<uint32_t>(Ops::Base::CeilAlign<uint64_t>(tiling_->singleCoreGroup, POLYPHASE_C0));
    const uint32_t row = static_cast<uint32_t>(
        Ops::Base::CeilAlign<uint64_t>(m * q, (Ops::Base::GetVRegSize() / sizeof(T))));
    const uint32_t offset = bank * memoryLayout_.outputBankPitchBytes;
    LocalTensor<T> phaseOut(TPosition::VECIN, offset, tiling_->singleCoreGroup * POLYPHASE_C0 * m);
    LocalTensor<T> dst(TPosition::VECIN, offset + memoryLayout_.outputOffsetBytes, POLYPHASE_C0 * row);
    LocalTensor<uint16_t> index(TPosition::VECIN, memoryLayout_.indexOffsetBytes, row);
    if (output.allZero) {
        // 每个有效通道复用只读零行序列，不执行 Gather。
        Duplicate<T>(dst, static_cast<T>(0), row);
    } else {
        ClearDepthwiseZeroRuns(geometry, phaseOut, SegmentedParallelPhases(output), output.begin, output.end);
        PipeBarrier<PIPE_V>();
        const uint64_t first = geometry.uStart * stride + output.begin;
        const uint32_t validSpan = Std::min(geometry.curJ * stride,
                                            static_cast<uint32_t>(static_cast<uint64_t>(tiling_->wi) - first));
        SegmentedParallelGatherOutput(
            reinterpret_cast<__ubuf__ uint16_t*>(index.GetPhyAddr()),
            reinterpret_cast<__ubuf__ T*>(phaseOut.GetPhyAddr()), reinterpret_cast<__ubuf__ T*>(dst.GetPhyAddr()),
            static_cast<uint16_t>(geometry.curN), static_cast<uint16_t>(m), static_cast<uint16_t>(q),
            static_cast<uint16_t>(row), static_cast<uint16_t>(geometry.curJ), static_cast<uint16_t>(output.width),
            static_cast<uint16_t>(stride), static_cast<uint16_t>(validSpan));
    }
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    StoreSegmentedRows(geometry, output, dst, q, row, output.allZero);
    SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
}

// PHASED 调度器负责 tap 迭代，此处仅计算一个 depthwise 分段。
template <bool parallel>
__aicore__ inline void DepthwiseSegment(PolyphaseWorkGeometry geometry, const SegmentedPhaseRange& range,
                                        const PhaseSegment& segment, SegmentedParallelA1Cache& cache) const
{
    const uint32_t phaseElements = tiling_->baseN * static_cast<uint32_t>(tiling_->kValueSegment);
    const uint32_t m = tiling_->baseM;
    const uint32_t cap = tiling_->stepKa;
    const uint32_t a2Elements = m * tiling_->baseK;
    const uint32_t b2Elements = POLYPHASE_C0 * tiling_->baseK;
    LocalTensor<T> a1(TPosition::A1, 0, static_cast<uint32_t>(memoryLayout_.a1Bytes / sizeof(T)));
    LocalTensor<L0cT> c0(TPosition::CO1, 0, m * POLYPHASE_C0);
    const uint64_t first = segment.first;
    const uint32_t bank = segment.bank;
    const event_t event = static_cast<event_t>(bank);
    geometry.xStart = segment.xStart;
    WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(0));
    if (!parallel || !cache.valid || cache.xStart != geometry.xStart) {
        LoadA1(geometry, a1);
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
        if constexpr (parallel) {
            cache.xStart = geometry.xStart;
            cache.valid = true;
        }
    }
    CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_MTE1>(POLYPHASE_AIV1_FLAG_BASE + bank);
    WaitFlag<HardEvent::M_MTE1>(event);
    LocalTensor<T> a2(TPosition::A2, bank * a2Elements * sizeof(T), a2Elements);
    LocalTensor<T> b2(TPosition::B2, bank * b2Elements * sizeof(T), b2Elements);
    LocalTensor<T> b1(TPosition::B1, memoryLayout_.a1Bytes + bank * memoryLayout_.phaseB1Bytes, phaseElements);
    LoadA1ToA2<false>(geometry, 0U, m, 0U, tiling_->baseK, 0U, a1, a2);
    LoadB1ToB2<false>(0U, tiling_->baseK, geometry.curN, 0U, b1, b2);
    // 命中缓存时，每个有效分段仍获取并归还 A1 所有权。
    SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(0));
    CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE1>(POLYPHASE_AIV1_FLAG_BASE + bank);
    SetFlag<HardEvent::MTE1_M>(event);
    WaitFlag<HardEvent::MTE1_M>(event);
    ComputeMatmul(c0, a2, b2, static_cast<uint16_t>(m), POLYPHASE_C0, static_cast<uint16_t>(tiling_->baseK),
                  first == range.first);
    PipeBarrier<PIPE_M>();
    SetFlag<HardEvent::M_MTE1>(event);
}

template <bool parallel>
__aicore__ inline void InitDepthwiseOutputIndex(bool allZero, bool& initialized) const
{
    if (allZero || initialized) {
        return;
    }
    const uint32_t m = tiling_->baseM;
    const uint32_t width = parallel ? tiling_->singleCoreGroup : tiling_->stepKb;
    // 并行 phase 按 C0 补齐；串行保持完整 stride，不能合并两种 VF。
    const uint32_t pitch = parallel ? static_cast<uint32_t>(Ops::Base::CeilAlign<uint64_t>(width, POLYPHASE_C0)) :
                                      width;
    const uint32_t row = static_cast<uint32_t>(
        Ops::Base::CeilAlign<uint64_t>(m * pitch, (Ops::Base::GetVRegSize() / sizeof(T))));
    LocalTensor<uint16_t> index(TPosition::VECIN, memoryLayout_.indexOffsetBytes, row);
    if constexpr (parallel) {
        Convolution3DBackpropFunc::SegmentedParallelOutputIndex(
            reinterpret_cast<__ubuf__ uint16_t*>(index.GetPhyAddr()), static_cast<uint16_t>(m),
            static_cast<uint16_t>(pitch), static_cast<uint16_t>(width), static_cast<uint16_t>(row));
    } else {
        Convolution3DBackpropFunc::SegmentedOutputIndex(reinterpret_cast<__ubuf__ uint16_t*>(index.GetPhyAddr()),
                                                        static_cast<uint16_t>(m), static_cast<uint16_t>(width),
                                                        static_cast<uint16_t>(row));
    }
    PipeBarrier<PIPE_V>();
    initialized = true;
}

#endif // CONV_BP_INPUT_SUB_FUNC_POLYPHASE_DEPTHWISE_SEGMENTED_H
