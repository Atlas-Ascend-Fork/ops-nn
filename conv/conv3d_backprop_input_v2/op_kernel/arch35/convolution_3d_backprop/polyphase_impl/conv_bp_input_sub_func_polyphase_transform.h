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
 * \file conv_bp_input_sub_func_polyphase_transform.h
 * \brief PHASED 滤波器打包、结果转换与共用分段输出。
 */

#ifndef CONV_BP_INPUT_SUB_FUNC_POLYPHASE_TRANSFORM_H
#define CONV_BP_INPUT_SUB_FUNC_POLYPHASE_TRANSFORM_H

// PHASED GM 到 L1 与滤波器打包
__aicore__ inline void LoadA1(const PolyphaseWorkGeometry& geometry, LocalTensor<T>& a1) const
{
    const uint32_t channelStart = tiling_->polyphaseDense != 0U ? 0U : geometry.nStart;
    const uint32_t channels = tiling_->polyphaseDense != 0U ? tiling_->cout : geometry.curN;
    const uint64_t offset = (static_cast<uint64_t>(geometry.batchIdx) * tiling_->cout + channelStart) * tiling_->wo;
    DataMove::GmToL1Window(a1, dedyGm_[offset], geometry.xStart, geometry.localA1W, tiling_->wo, channels);
}

__aicore__ inline void SetFilterGatherIndex(LocalTensor<uint16_t>& index, uint32_t q, uint32_t kCompact) const
{
    // B1 fractal 的 N lane 位于 K lane 外层。
    // 每个向量从同一对齐 raw 基址 Gather 完整 C0 列。
    for (uint32_t n = 0; n < (Ops::Base::GetVRegSize() / sizeof(T) / POLYPHASE_C0); ++n) {
        for (uint32_t cout = 0; cout < POLYPHASE_C0; ++cout) {
            index.SetValue(n * POLYPHASE_C0 + cout, static_cast<uint16_t>(n * kCompact + cout * q));
        }
    }
    const event_t event = static_cast<event_t>(0);
    SetFlag<HardEvent::S_V>(event);
    WaitFlag<HardEvent::S_V>(event);
}

// PHASED dense 的 NZ 直接搬运与 bias 读取。
__aicore__ inline void CopyDenseFilterTap(const PolyphaseWorkGeometry& geometry, uint32_t originalTap,
                                          uint32_t hDstOffset, LocalTensor<T>& nz) const
{
    const uint32_t nPitch = tiling_->baseN / POLYPHASE_C0;
    const uint32_t nBlocks = geometry.curN / POLYPHASE_C0;
    if (nBlocks != 0U) {
        const uint64_t offset = static_cast<uint64_t>(geometry.nStart) * tiling_->wk + originalTap;
        DataMove::GmToUb<POLYPHASE_NDDMA_4D>(
            nz[hDstOffset], filterGm_[offset],
            {{POLYPHASE_C0, static_cast<uint64_t>(tiling_->cin) * tiling_->wk, 1U},
             {POLYPHASE_C0, tiling_->wk, POLYPHASE_C0},
             {nBlocks, static_cast<uint64_t>(POLYPHASE_C0) * tiling_->wk, (POLYPHASE_C0 * POLYPHASE_C0)},
             {tiling_->cout / POLYPHASE_C0, static_cast<uint64_t>(POLYPHASE_C0) * tiling_->cin * tiling_->wk,
              nPitch * (POLYPHASE_C0 * POLYPHASE_C0)}});
    }
}

__aicore__ inline void CopyDenseFilterTail(const PolyphaseWorkGeometry& geometry, uint32_t originalTap,
                                           uint32_t hDstOffset, LocalTensor<T>& nz) const
{
    const uint32_t nPitch = tiling_->baseN / POLYPHASE_C0;
    const uint32_t nBlocks = geometry.curN / POLYPHASE_C0;
    const uint32_t nTail = geometry.curN % POLYPHASE_C0;
    if (nTail != 0U) {
        const uint64_t offset = static_cast<uint64_t>(geometry.nStart + nBlocks * POLYPHASE_C0) * tiling_->wk +
                                originalTap;
        DataMove::GmToUb<POLYPHASE_NDDMA_3D>(
            nz[hDstOffset + nBlocks * (POLYPHASE_C0 * POLYPHASE_C0)], filterGm_[offset],
            {{POLYPHASE_C0, static_cast<uint64_t>(tiling_->cin) * tiling_->wk, 1U},
             {nTail, tiling_->wk, POLYPHASE_C0},
             {tiling_->cout / POLYPHASE_C0, static_cast<uint64_t>(POLYPHASE_C0) * tiling_->cin * tiling_->wk,
              nPitch * (POLYPHASE_C0 * POLYPHASE_C0)}});
    }
}

__aicore__ inline void PackDenseFilterPhaseDirectNz(const PolyphaseWorkGeometry& geometry, uint32_t phase,
                                                    uint32_t bOffsetBytes) const
{
    const uint32_t phaseElements = tiling_->baseN * static_cast<uint32_t>(tiling_->kValueSegment);
    const uint32_t q = tiling_->stepKa;
    const uint32_t nFracPitch = tiling_->baseN / POLYPHASE_C0;
    const uint32_t coutBlocks = tiling_->cout / POLYPHASE_C0;
    LocalTensor<T> nz(TPosition::VECIN, 0, phaseElements);
    Duplicate<T>(nz, static_cast<T>(0), phaseElements);
    const event_t event = static_cast<event_t>(0);
    SetFlag<HardEvent::V_MTE2>(event);
    WaitFlag<HardEvent::V_MTE2>(event);
    for (uint32_t h = 0U; h < q; ++h) {
        const uint32_t originalTap = phase + tiling_->stepKb * (q - 1U - h);
        const uint32_t hDstOffset = h * coutBlocks * nFracPitch * POLYPHASE_C0 * POLYPHASE_C0;
        CopyDenseFilterTap(geometry, originalTap, hDstOffset, nz);
        CopyDenseFilterTail(geometry, originalTap, hDstOffset, nz);
    }
    SetFlag<HardEvent::MTE2_MTE3>(event);
    WaitFlag<HardEvent::MTE2_MTE3>(event);
    LocalTensor<T> b1(TPosition::B1, bOffsetBytes, phaseElements);
    DataMove::UbToL1(b1, nz, phaseElements);
    SetFlag<HardEvent::MTE3_V>(event);
    WaitFlag<HardEvent::MTE3_V>(event);
}

__aicore__ inline void PackPhaseFilter(const PolyphaseWorkGeometry& geometry, uint32_t phase,
                                       uint32_t bOffsetBytes) const
{
    if (tiling_->polyphaseDense != 0U) {
        PackDenseFilterPhaseDirectNz(geometry, phase, bOffsetBytes);
    } else {
        PackDepthwiseFilter(geometry, phase, tiling_->stepKa, bOffsetBytes);
    }
}

__aicore__ inline void LoadBiasToBank(const PolyphaseWorkGeometry& geometry, uint32_t outputBank, uint32_t mAlign) const
{
    const uint32_t bankBase = outputBank * GetOutputBankBytes(mAlign);
    const uint32_t biasOffset = bankBase + static_cast<uint32_t>(memoryLayout_.biasOffsetBytes);
    const uint32_t biasElements = static_cast<uint32_t>(
        (memoryLayout_.indexOffsetBytes - memoryLayout_.biasOffsetBytes) / sizeof(float));
    LocalTensor<float> bias(TPosition::VECIN, biasOffset, biasElements);
    Duplicate<float>(bias, 0.0F, biasElements);
    event_t event = static_cast<event_t>(0);
    SetFlag<HardEvent::V_MTE2>(event);
    WaitFlag<HardEvent::V_MTE2>(event);
    const uint8_t rightPadding = static_cast<uint8_t>(((Ops::Base::GetUbBlockSize() / sizeof(float)) -
                                                       geometry.curN % (Ops::Base::GetUbBlockSize() / sizeof(float))) %
                                                      (Ops::Base::GetUbBlockSize() / sizeof(float)));
    DataMove::GmToUbVector(bias, biasGm_[geometry.nStart], geometry.curN, rightPadding);
    event = static_cast<event_t>(0);
    SetFlag<HardEvent::MTE2_V>(event);
    WaitFlag<HardEvent::MTE2_V>(event);
}
// PHASED L1 到 L0
template <bool tapwise>
__aicore__ inline void LoadA1ToA2(const PolyphaseWorkGeometry& geometry, uint32_t innerStart, uint32_t mAlign,
                                  uint32_t channelOffset, uint32_t curK, uint32_t tap, const LocalTensor<T>& a1,
                                  LocalTensor<T>& a2) const
{
    const uint32_t channels = tiling_->polyphaseDense != 0U ? tiling_->cout : POLYPHASE_C0;
    DataMove::L1ToL0A(
        a2, a1,
        {geometry.localA1W, channels, tapwise ? 1U : tiling_->stepKa, innerStart + (tapwise ? tap : 0U), mAlign,
         channelOffset, curK, static_cast<uint32_t>(Ops::Base::CeilDiv<uint64_t>(mAlign, POLYPHASE_C0))});
}

template <bool tapwise>
__aicore__ inline void LoadB1ToB2(uint32_t channelOffset, uint32_t curK, uint32_t curN, uint32_t tap,
                                  const LocalTensor<T>& b1, LocalTensor<T>& b2) const
{
    const uint32_t channels = tiling_->polyphaseDense != 0U ? tiling_->cout : POLYPHASE_C0;
    const uint32_t step = tapwise ? 1U : tiling_->stepKa;
    const uint32_t firstBlock = channelOffset / (step * POLYPHASE_C0);
    const uint32_t blocks = curK / (step * POLYPHASE_C0);
    const uint32_t nPitch = tiling_->baseN / POLYPHASE_C0;
    const uint32_t nBlocks = (curN + POLYPHASE_C0 - 1U) / POLYPHASE_C0;
    const uint32_t tapOffset = tapwise ? tap * (channels / POLYPHASE_C0) * (POLYPHASE_C0 * POLYPHASE_C0) : 0U;
    for (uint32_t block = 0U; block < blocks; ++block) {
        DataMove::L1ToL0B(b2[block * step * nBlocks * (POLYPHASE_C0 * POLYPHASE_C0)], b1[tapOffset], nPitch, nBlocks,
                          (firstBlock + block) * step, step);
    }
}

__aicore__ inline void PackDepthwiseFilter(const PolyphaseWorkGeometry& geometry, uint64_t firstTap, uint32_t taps,
                                           uint32_t bOffsetBytes) const
{
    const uint32_t channels = tiling_->polyphaseDense != 0U ? tiling_->cout : POLYPHASE_C0;
    const uint32_t rawElements = tiling_->baseN * static_cast<uint32_t>(tiling_->kValueSegment);
    const uint32_t rawBytes = rawElements * sizeof(T);
    LocalTensor<T> raw(TPosition::VECIN, 0, rawElements);
    LocalTensor<T> nz(TPosition::VECIN, rawBytes, rawElements);
    LocalTensor<uint16_t> index(TPosition::VECIN, POLYPHASE_FILTER_PACK_BUFFERS * rawBytes,
                                (Ops::Base::GetVRegSize() / sizeof(T)));
    Duplicate<T>(raw, static_cast<T>(0), rawElements);
    event_t event = static_cast<event_t>(0);
    SetFlag<HardEvent::V_MTE2>(event);
    WaitFlag<HardEvent::V_MTE2>(event);
    const uint64_t srcOffset = static_cast<uint64_t>(geometry.nStart) * tiling_->wk + firstTap;
    DataMove::GmToUb<POLYPHASE_NDDMA_2D>(
        raw, filterGm_[srcOffset],
        {{taps, tiling_->stepKb, 1U}, {geometry.curN, tiling_->wk, (POLYPHASE_C0 + 1U) * tiling_->stepKa}});
    event = static_cast<event_t>(0);
    SetFlag<HardEvent::MTE2_V>(event);
    WaitFlag<HardEvent::MTE2_V>(event);
    Convolution3DBackpropFunc::PolyphaseFilterDn2Nz(
        reinterpret_cast<__ubuf__ uint16_t*>(index.GetPhyAddr()), reinterpret_cast<__ubuf__ T*>(raw.GetPhyAddr()),
        reinterpret_cast<__ubuf__ T*>(nz.GetPhyAddr()), static_cast<uint16_t>(channels / POLYPHASE_C0),
        static_cast<uint16_t>(tiling_->stepKa), static_cast<uint16_t>(tiling_->kValueSegment));
    event = static_cast<event_t>(0);
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    LocalTensor<T> b1(TPosition::B1, bOffsetBytes, rawElements);
    DataMove::UbToL1(b1, nz, rawElements);
    // 等待 MTE3 完成后，下个 phase 才可覆盖共用 raw/NZ UB。
    event = static_cast<event_t>(0);
    SetFlag<HardEvent::MTE3_V>(event);
    WaitFlag<HardEvent::MTE3_V>(event);
}

// PHASED L0C 转换与输出
__aicore__ inline uint32_t GetOutputRowAlign(uint32_t mAlign) const
{
    // stride=2 用 Interleave 生成紧凑行，逐行 Gather 需要向量对齐。
    return tiling_->stepKb == POLYPHASE_INTERLEAVE_STRIDE ?
               tiling_->stepKb * mAlign :
               Ops::Base::CeilAlign<uint32_t>(tiling_->stepKb * mAlign, (Ops::Base::GetVRegSize() / sizeof(T)));
}

__aicore__ inline uint32_t GetOutputBankBytes(uint32_t mAlign) const
{
    if (tiling_->polyphaseDense != 0U) {
        return static_cast<uint32_t>(memoryLayout_.outputBankPitchBytes);
    }
    const uint32_t outputRowAlign = GetOutputRowAlign(mAlign);
    const uint32_t indexElements = tiling_->stepKb == POLYPHASE_INTERLEAVE_STRIDE ? 0U : outputRowAlign;
    return (tiling_->stepKb * tiling_->baseN * mAlign + tiling_->baseN * outputRowAlign + indexElements) * sizeof(T);
}

__aicore__ inline void SetOutputGatherIndex(LocalTensor<uint16_t>& index, uint32_t mAlign,
                                            uint32_t outputRowAlign) const
{
    const uint32_t slotElements = tiling_->baseN * mAlign;
    const uint32_t valid = tiling_->stepKb * mAlign;
    for (uint32_t w = 0; w < outputRowAlign; ++w) {
        const uint32_t src = w < valid ? (w % tiling_->stepKb) * slotElements + w / tiling_->stepKb : 0U;
        index.SetValue(w, static_cast<uint16_t>(src));
    }
    const event_t event = static_cast<event_t>(0);
    SetFlag<HardEvent::S_V>(event);
    WaitFlag<HardEvent::S_V>(event);
}

__aicore__ inline void CastFullTapOutput(const PolyphaseWorkGeometry& geometry, uint32_t outputBank, uint32_t mAlign,
                                         uint32_t bankBase, bool loadBias) const
{
    const uint32_t slotElements = tiling_->baseN * mAlign;
    LocalTensor<float> source(TPosition::VECIN, bankBase, tiling_->stepKb * slotElements);
    LocalTensor<float> bias(TPosition::VECIN, bankBase + static_cast<uint32_t>(memoryLayout_.biasOffsetBytes),
                            tiling_->baseN);
    LocalTensor<T> phase(TPosition::VECIN, bankBase + static_cast<uint32_t>(memoryLayout_.phaseOutputOffsetBytes),
                         tiling_->stepKb * slotElements);
    if (tiling_->isBiasFullLoad != 0U) {
        if (loadBias) {
            LoadBiasToBank(geometry, outputBank, mAlign);
        }
        Convolution3DBackpropFunc::PolyphaseAddBiasCast(
            reinterpret_cast<__ubuf__ float*>(source.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(bias.GetPhyAddr()), reinterpret_cast<__ubuf__ T*>(phase.GetPhyAddr()),
            static_cast<uint16_t>(tiling_->stepKb), static_cast<uint16_t>(geometry.curN),
            static_cast<uint16_t>(tiling_->baseN), static_cast<uint16_t>(mAlign));
    } else {
        Convolution3DBackpropFunc::PolyphaseCastNoBias(
            reinterpret_cast<__ubuf__ float*>(source.GetPhyAddr()), reinterpret_cast<__ubuf__ T*>(phase.GetPhyAddr()),
            static_cast<uint16_t>(tiling_->stepKb), static_cast<uint16_t>(geometry.curN),
            static_cast<uint16_t>(tiling_->baseN), static_cast<uint16_t>(mAlign));
    }
}

template <uint32_t stride>
__aicore__ inline void GatherFullTapOutput(const PolyphaseWorkGeometry& geometry, uint32_t mAlign, uint32_t row,
                                           const LocalTensor<uint16_t>& index, const LocalTensor<T>& src,
                                           LocalTensor<T>& dst) const
{
    Convolution3DBackpropFunc::GatherPhaseMajorToW<stride>(
        reinterpret_cast<__ubuf__ uint16_t*>(index.GetPhyAddr()), reinterpret_cast<__ubuf__ T*>(src.GetPhyAddr()),
        reinterpret_cast<__ubuf__ T*>(dst.GetPhyAddr()), static_cast<uint16_t>(geometry.curN),
        static_cast<uint16_t>(mAlign), static_cast<uint16_t>(row));
}

__aicore__ inline bool TransformFullTapOutput(const PolyphaseWorkGeometry& geometry, uint32_t mAlign, uint32_t bankBase,
                                              uint32_t dstOffsetBytes, const LocalTensor<T>& src,
                                              LocalTensor<T>& dst) const
{
    const uint32_t slotElements = tiling_->baseN * mAlign;
    const uint32_t outputRowAlign = GetOutputRowAlign(mAlign);
    const bool dense = tiling_->polyphaseDense != 0U;
    if (tiling_->stepKb == POLYPHASE_INTERLEAVE_STRIDE) {
        const uint32_t vfLen = (Ops::Base::GetVRegSize() / sizeof(T));
        const uint16_t repeats = static_cast<uint16_t>(Ops::Base::CeilDiv<uint64_t>(slotElements, vfLen));
        Convolution3DBackpropFunc::InterleaveKernelSplitNormal<T>(
            reinterpret_cast<__ubuf__ T*>(src.GetPhyAddr()),
            reinterpret_cast<__ubuf__ T*>(src[slotElements].GetPhyAddr()),
            reinterpret_cast<__ubuf__ T*>(dst.GetPhyAddr()), reinterpret_cast<__ubuf__ T*>(dst[vfLen].GetPhyAddr()),
            vfLen, vfLen * POLYPHASE_INTERLEAVE_STRIDE, repeats);
    } else {
        const uint32_t indexOffsetBytes = dense ? bankBase + static_cast<uint32_t>(memoryLayout_.indexOffsetBytes) :
                                                  dstOffsetBytes + tiling_->baseN * outputRowAlign * sizeof(T);
        LocalTensor<uint16_t> index(TPosition::VECIN, indexOffsetBytes, outputRowAlign);
        SetOutputGatherIndex(index, mAlign, outputRowAlign);
        switch (tiling_->stepKb) {
            case POLYPHASE_GATHER_STRIDE3:
                GatherFullTapOutput<POLYPHASE_GATHER_STRIDE3>(geometry, mAlign, outputRowAlign, index, src, dst);
                break;
            case POLYPHASE_GATHER_STRIDE4:
                GatherFullTapOutput<POLYPHASE_GATHER_STRIDE4>(geometry, mAlign, outputRowAlign, index, src, dst);
                break;
            case POLYPHASE_GATHER_STRIDE5:
                GatherFullTapOutput<POLYPHASE_GATHER_STRIDE5>(geometry, mAlign, outputRowAlign, index, src, dst);
                break;
            default:
                return false;
        }
    }
    return true;
}

__aicore__ inline void StoreFullTapOutput(const PolyphaseWorkGeometry& geometry, uint32_t innerStart, uint32_t curM,
                                          uint32_t outputRowAlign, const LocalTensor<T>& dst) const
{
    const uint32_t validW = tiling_->stepKb * curM;
    const uint64_t outputPosition = tiling_->stepKb * (geometry.uStart + innerStart);
    const uint64_t offset = (static_cast<uint64_t>(geometry.batchIdx) * tiling_->cin + geometry.nStart) * tiling_->wi +
                            outputPosition;
    if (tiling_->polyphaseDense != 0U) {
        for (uint32_t n = 0U; n < geometry.curN; ++n) {
            DataMove::UbToGm(yGm_[offset + static_cast<uint64_t>(n) * tiling_->wi], dst[n * outputRowAlign], 1U, validW,
                             0U, 0U);
        }
        return;
    }
    DataMove::UbToGm(yGm_[offset], dst, geometry.curN, validW,
                     (outputRowAlign - validW) * sizeof(T) / Ops::Base::GetUbBlockSize(),
                     static_cast<uint64_t>(tiling_->wi - validW) * sizeof(T));
}

__aicore__ inline void TransformAndStore(const PolyphaseWorkGeometry& geometry, uint32_t innerStart, uint32_t curM,
                                         uint32_t mAlign, uint32_t outputBank, bool loadBias) const
{
    const uint32_t slotElements = tiling_->baseN * mAlign;
    const uint32_t outputRowAlign = GetOutputRowAlign(mAlign);
    const bool dense = tiling_->polyphaseDense != 0U;
    const uint32_t bankBase = outputBank * GetOutputBankBytes(mAlign);
    if (dense) {
        CastFullTapOutput(geometry, outputBank, mAlign, bankBase, loadBias);
    }
    const uint32_t srcOffsetBytes = bankBase +
                                    (dense ? static_cast<uint32_t>(memoryLayout_.phaseOutputOffsetBytes) : 0U);
    const uint32_t dstOffsetBytes = dense ? bankBase + static_cast<uint32_t>(memoryLayout_.outputOffsetBytes) :
                                            bankBase + tiling_->stepKb * slotElements * sizeof(T);
    LocalTensor<T> dst(TPosition::VECIN, dstOffsetBytes, tiling_->baseN * outputRowAlign);
    LocalTensor<T> src(TPosition::VECIN, srcOffsetBytes, tiling_->stepKb * slotElements);
    if (!TransformFullTapOutput(geometry, mAlign, bankBase, dstOffsetBytes, src, dst)) {
        return;
    }
    const event_t event = static_cast<event_t>(0);
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    StoreFullTapOutput(geometry, innerStart, curM, outputRowAlign, dst);
}

template <class OutputGeometry>
__aicore__ inline void StoreSegmentedRows(const PolyphaseWorkGeometry& geometry, const OutputGeometry& output,
                                          const LocalTensor<T>& dst, uint32_t pitch, uint32_t row, bool allZero) const
{
    const uint32_t stride = tiling_->stepKb;
    for (uint32_t c = 0U; c < geometry.curN; ++c) {
        const uint64_t gm = (static_cast<uint64_t>(geometry.batchIdx) * tiling_->cin + geometry.nStart + c) *
                                tiling_->wi +
                            geometry.uStart * stride + output.begin;
        const uint32_t ub = allZero ? 0U : c * row;
        if (output.fullRows != 0U) {
            const uint32_t gap = (pitch -
                                  static_cast<uint32_t>(Ops::Base::CeilAlign<uint64_t>(output.width, POLYPHASE_C0))) /
                                 POLYPHASE_C0;
            DataMove::UbToGm(yGm_[gm], dst[ub], output.fullRows, output.width, gap,
                             static_cast<uint64_t>(stride - output.width) * sizeof(T));
        }
        if (output.tailWidth != 0U) {
            DataMove::UbToGm(yGm_[gm + static_cast<uint64_t>(output.fullRows) * stride],
                             dst[ub + output.fullRows * pitch], 1U, output.tailWidth, 0U, 0U);
        }
    }
}

#endif // CONV_BP_INPUT_SUB_FUNC_POLYPHASE_TRANSFORM_H
