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
 * \file conv_bp_input_sub_func_polyphase_pipeline.h
 * \brief PHASED 全 tap 计算与 DX 调度。
 */

#ifndef CONV_BP_INPUT_SUB_FUNC_POLYPHASE_PIPELINE_H
#define CONV_BP_INPUT_SUB_FUNC_POLYPHASE_PIPELINE_H

// PHASED Cube 计算
__aicore__ inline void ComputePhaseTileToUb(const PolyphaseWorkGeometry& geometry, uint32_t innerStart, uint32_t curM,
                                            uint32_t outputPitch, uint32_t outputBank, uint32_t phase,
                                            uint32_t outputInnerOffset, const LocalTensor<T>& a1,
                                            const LocalTensor<T>& b1) const
{
    const uint32_t slotElements = tiling_->baseN * outputPitch;
    const uint32_t bankBytes = GetOutputBankBytes(outputPitch);
    if (tiling_->polyphaseDense != 0U) {
        const uint32_t offsetBytes = outputBank * bankBytes +
                                     (phase * slotElements + outputInnerOffset) * sizeof(float);
        LocalTensor<float> phaseOut(TPosition::VECIN, offsetBytes, slotElements);
        ComputeDensePhaseTile(geometry, innerStart, curM, outputPitch, a1, b1, phaseOut);
        return;
    }

    const uint32_t phaseBytes = slotElements * sizeof(T);
    const uint32_t offsetBytes = outputBank * bankBytes + phase * phaseBytes + outputInnerOffset * sizeof(T);
    LocalTensor<T> phaseOut(TPosition::VECIN, offsetBytes, slotElements);
    ComputePhaseTile(geometry, innerStart, curM, outputPitch, a1, b1, phaseOut);
}

__aicore__ inline void ComputeDepthwiseKTile(const PolyphaseWorkGeometry& geometry, uint32_t innerStart,
                                             uint32_t mAlign, uint32_t kOffset, uint32_t curK, uint32_t bank,
                                             bool firstK, const LocalTensor<T>& a1, const LocalTensor<T>& b1,
                                             LocalTensor<L0cT>& c0) const
{
    const uint32_t a2BankElements = tiling_->baseM * tiling_->baseK;
    const uint32_t b2BankElements = tiling_->baseN * tiling_->baseK;
    const event_t event = static_cast<event_t>(bank);
    LocalTensor<T> a2(TPosition::A2, bank * a2BankElements * sizeof(T), a2BankElements);
    LocalTensor<T> b2(TPosition::B2, bank * b2BankElements * sizeof(T), b2BankElements);
    WaitFlag<HardEvent::M_MTE1>(event);
    LoadA1ToA2<false>(geometry, innerStart, mAlign, kOffset, curK, 0U, a1, a2);
    LoadB1ToB2<false>(kOffset, curK, geometry.curN, 0U, b1, b2);
    SetFlag<HardEvent::MTE1_M>(event);
    WaitFlag<HardEvent::MTE1_M>(event);
    ComputeMatmul(c0, a2, b2, static_cast<uint16_t>(mAlign),
                  static_cast<uint16_t>(Ops::Base::CeilDiv<uint64_t>(geometry.curN, POLYPHASE_C0) * POLYPHASE_C0),
                  static_cast<uint16_t>(curK), firstK);
    // MMAD 消费完 L0A/L0B 后再发 M_MTE1，避免第三个 K tile
    // 覆盖仍在读取的 bank，造成不稳定 phase 误差。
    PipeBarrier<PIPE_M>();
    SetFlag<HardEvent::M_MTE1>(event);
}

__aicore__ inline void ComputePhaseTile(const PolyphaseWorkGeometry& geometry, uint32_t innerStart, uint32_t curM,
                                        uint32_t outputStride, const LocalTensor<T>& a1, const LocalTensor<T>& b1,
                                        LocalTensor<T>& phaseOut) const
{
    const uint32_t mAlign = Ops::Base::CeilAlign<uint32_t>(curM, POLYPHASE_C0);
    const uint32_t kCompact = static_cast<uint32_t>(tiling_->kValueSegment);
    LocalTensor<L0cT> c0(TPosition::CO1, 0, tiling_->baseM * tiling_->baseN);
    // 双缓冲防止下一个 K tile 覆盖当前 MMAD 输入。
    // 每个 A2/B2 bank 使用独立事件，沿用 SmallShape 的配对方式。
    SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
    SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));

    WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(0));
    bool firstK = true;
    uint32_t localKOrdinal = 0U;
    for (uint32_t kOffset = 0; kOffset < kCompact; kOffset += tiling_->baseK) {
        const uint32_t curK = Std::min(tiling_->baseK, kCompact - kOffset);
        const uint32_t bank = localKOrdinal++ & 1U;

        ComputeDepthwiseKTile(geometry, innerStart, mAlign, kOffset, curK, bank, firstK, a1, b1, c0);
        firstK = false;
    }
    WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
    WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));

    SetFlag<HardEvent::M_FIX>(static_cast<event_t>(0));
    WaitFlag<HardEvent::M_FIX>(static_cast<event_t>(0));

    DataMove::L0cToUb(phaseOut, c0, mAlign, POLYPHASE_C0, mAlign, outputStride,
                      (std::is_same<T, half>::value ? QuantMode_t::F322F16 : QuantMode_t::F322BF16));
    SetFlag<HardEvent::FIX_M>(static_cast<event_t>(0));
}

__aicore__ inline void ComputeDensePhaseTile(const PolyphaseWorkGeometry& geometry, uint32_t innerStart, uint32_t curM,
                                             uint32_t outputStride, const LocalTensor<T>& a1, const LocalTensor<T>& b1,
                                             LocalTensor<float>& phaseOut) const
{
    const uint32_t mAlign = Ops::Base::CeilAlign<uint32_t>(curM, POLYPHASE_C0);
    const uint32_t channels = tiling_->cout; // 唯一调用已限定 dense，无需再次判断。
    const uint32_t channelTile = tiling_->baseK / tiling_->stepKa;
    const uint32_t a2BankElements = tiling_->baseM * channelTile;
    const uint32_t b2BankElements = tiling_->baseN * channelTile;
    LocalTensor<L0cT> c0(TPosition::CO1, 0, tiling_->baseM * tiling_->baseN);
    SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
    SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
    WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(0));
    bool firstK = true;
    uint32_t ordinal = 0U;
    // B1 反序保存 tap，反向遍历保持原物理 tap 累加顺序。
    for (uint32_t tapDown = tiling_->stepKa; tapDown > 0U; --tapDown) {
        const uint32_t tap = tapDown - 1U;
        for (uint32_t channel = 0U; channel < channels; channel += channelTile, ++ordinal) {
            const uint32_t curK = Std::min(channelTile, channels - channel);
            const uint32_t bank = ordinal & 1U;
            const event_t event = static_cast<event_t>(bank);
            LocalTensor<T> a2(TPosition::A2, bank * a2BankElements * sizeof(T), a2BankElements);
            LocalTensor<T> b2(TPosition::B2, bank * b2BankElements * sizeof(T), b2BankElements);
            WaitFlag<HardEvent::M_MTE1>(event);
            LoadA1ToA2<true>(geometry, innerStart, mAlign, channel, curK, tap, a1, a2);
            LoadB1ToB2<true>(channel, curK, geometry.curN, tap, b1, b2);
            SetFlag<HardEvent::MTE1_M>(event);
            WaitFlag<HardEvent::MTE1_M>(event);
            ComputeMatmul(c0, a2, b2, static_cast<uint16_t>(mAlign), static_cast<uint16_t>(POLYPHASE_C0),
                          static_cast<uint16_t>(curK), firstK);
            PipeBarrier<PIPE_M>();
            firstK = false;
            SetFlag<HardEvent::M_MTE1>(event);
        }
    }
    WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
    WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
    SetFlag<HardEvent::M_FIX>(static_cast<event_t>(0));
    WaitFlag<HardEvent::M_FIX>(static_cast<event_t>(0));
    DataMove::L0cToUb(phaseOut, c0, mAlign, POLYPHASE_C0, mAlign, outputStride, QuantMode_t::NoQuant);
    SetFlag<HardEvent::FIX_M>(static_cast<event_t>(0));
}

// DX 保持 ALL_PHASES 驻留；ONE_PHASE 共用 Transpose 任务遍历。
__aicore__ inline void ProcessDxAllPhases()
{
    PrepareMemoryLayout(tiling_->polyphaseDense != 0U);
    const uint64_t blockIdx = GetAicBlockIdx();
    if (blockIdx >= tiling_->coreNum) {
        return;
    }
    if (GetRounds(blockIdx) == 0U) {
        return;
    }
    if ASCEND_IS_AIC_SCALAR {
        ProcessDxCube(blockIdx);
    }
    if ASCEND_IS_AIV_SCALAR {
        if (GetSubBlockIdx() == 0U) {
            ProcessDxOutput(blockIdx);
        } else {
            ProcessDxFilter(blockIdx);
        }
    }
}

__aicore__ inline void SeedAicProtocol() const
{
    // AIC 的 POLYPHASE_AIV1_FLAG_BASE + bank 释放对应 AIV1 滤波器 bank。
    CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE1>(POLYPHASE_AIV1_FLAG_BASE);
    CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE1>(POLYPHASE_AIV1_FLAG_BASE + 1U);
}

__aicore__ inline void DrainAicProtocol() const
{
    // 等待 AIV0 释放两个输出 bank。
    CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(0);
    CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(1);
}

__aicore__ inline void ProcessDxCube(uint64_t blockIdx) const
{
    const uint32_t phaseElements = tiling_->baseN * static_cast<uint32_t>(tiling_->kValueSegment);
    const uint64_t rounds = GetRounds(blockIdx);
    uint64_t basic = blockIdx;
    uint32_t outputTileOrdinal = 0;
    SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(0));
    SetFlag<HardEvent::FIX_M>(static_cast<event_t>(0));
    SeedAicProtocol();
    for (uint64_t round = 0; round < rounds; ++round, basic += tiling_->coreNum) {
        PolyphaseWorkGeometry geometry;
        DecodeGeometry(basic, geometry);
        const uint32_t workBank = static_cast<uint32_t>(round & 1U);
        WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(0));
        LocalTensor<T> a1(TPosition::A1, 0, static_cast<uint32_t>(memoryLayout_.a1Bytes / sizeof(T)));
        LoadA1(geometry, a1);
        SetFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<event_t>(0));
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_MTE1>(POLYPHASE_AIV1_FLAG_BASE + workBank);
        for (uint32_t inner = 0; inner < geometry.curJ; inner += tiling_->baseM) {
            const uint32_t curM = Std::min(tiling_->baseM, geometry.curJ - inner);
            const uint32_t outputBank = outputTileOrdinal & 1U;
            CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(outputBank);
            for (uint32_t phase = 0; phase < tiling_->stepKb; ++phase) {
                const uint32_t bOffset = memoryLayout_.a1Bytes + workBank * memoryLayout_.b1BankBytes +
                                         phase * phaseElements * sizeof(T);
                LocalTensor<T> b1(TPosition::B1, bOffset, phaseElements);
                ComputePhaseTileToUb(geometry, inner, curM,
                                     Ops::Base::CeilAlign<uint32_t>(tiling_->baseM, POLYPHASE_C0), outputBank, phase,
                                     0U, a1, b1);
            }
            CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_FIX>(outputBank);
            ++outputTileOrdinal;
        }
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE1>(POLYPHASE_AIV1_FLAG_BASE + workBank);
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(0));
    }
    WaitFlag<HardEvent::MTE1_MTE2>(static_cast<event_t>(0));
    WaitFlag<HardEvent::FIX_M>(static_cast<event_t>(0));
    DrainAicProtocol();
}

__aicore__ inline void ProcessDxFilter(uint64_t blockIdx) const
{
    const uint32_t phaseElements = tiling_->baseN * static_cast<uint32_t>(tiling_->kValueSegment);
    const uint64_t rounds = GetRounds(blockIdx);
    uint64_t basic = blockIdx;
    if (tiling_->polyphaseDense == 0U) {
        LocalTensor<uint16_t> index(TPosition::VECIN, POLYPHASE_FILTER_PACK_BUFFERS * phaseElements * sizeof(T),
                                    (Ops::Base::GetVRegSize() / sizeof(T)));
        SetFilterGatherIndex(index, tiling_->stepKa, static_cast<uint32_t>(tiling_->kValueSegment));
    }
    NdDmaDci();
    for (uint64_t round = 0; round < rounds; ++round, basic += tiling_->coreNum) {
        PolyphaseWorkGeometry geometry;
        DecodeGeometry(basic, geometry);
        const uint32_t bank = static_cast<uint32_t>(round & 1U);
        CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(bank);
        for (uint32_t phase = 0; phase < tiling_->stepKb; ++phase) {
            const uint32_t bOffset = memoryLayout_.a1Bytes + bank * memoryLayout_.b1BankBytes +
                                     phase * phaseElements * sizeof(T);
            PackPhaseFilter(geometry, phase, bOffset);
        }
        CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(bank);
    }
    // kernel 退出前，AIC 必须消费两个 bank。
    CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(0);
    CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(1);
}

__aicore__ inline void ProcessDxOutput(uint64_t blockIdx) const
{
    const uint64_t rounds = GetRounds(blockIdx);
    uint64_t basic = blockIdx;
    uint32_t outputTileOrdinal = 0;
    CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(0);
    CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(1);
    for (uint64_t round = 0; round < rounds; ++round, basic += tiling_->coreNum) {
        PolyphaseWorkGeometry geometry;
        DecodeGeometry(basic, geometry);
        const uint32_t mAlign = Ops::Base::CeilAlign<uint32_t>(tiling_->baseM, POLYPHASE_C0);
        bool biasReady[POLYPHASE_DOUBLE_BUFFER] = {false, false};
        for (uint32_t inner = 0; inner < geometry.curJ; inner += tiling_->baseM) {
            const uint32_t curM = Std::min(tiling_->baseM, geometry.curJ - inner);
            const uint32_t bank = outputTileOrdinal & 1U;
            CrossCoreWaitFlag<POLYPHASE_SYNC_MODE, PIPE_V>(bank);
            TransformAndStore(geometry, inner, curM, mAlign, bank, !biasReady[bank]);
            biasReady[bank] = true;
            CrossCoreSetFlag<POLYPHASE_SYNC_MODE, PIPE_MTE3>(bank);
            ++outputTileOrdinal;
        }
    }
    // AIC drain 消费最后两个释放信号，无需额外屏障。
}

#endif // CONV_BP_INPUT_SUB_FUNC_POLYPHASE_PIPELINE_H
