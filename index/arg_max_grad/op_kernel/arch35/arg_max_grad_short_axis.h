/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ARG_MAX_GRAD_SHORT_AXIS_H
#define ARG_MAX_GRAD_SHORT_AXIS_H

#include <type_traits>

#include "kernel_operator.h"
#include "arg_max_grad_tiling_data.h"

namespace ArgMaxGrad {
using namespace AscendC;

template <typename T, bool LastAxis = true>
class ArgMaxGradShortAxis {
private:
    using RawT = std::conditional_t<sizeof(T) == 1, uint8_t, std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>;
    static_assert(sizeof(T) == sizeof(RawT), "Short-axis scatter requires a 1-, 2- or 4-byte payload");
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t VF_ROWS = 64;
    static constexpr int16_t DIV_SHIFT = 16;

public:
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y,
                                const ArgMaxGradArch35TilingData* tiling, TPipe* pipe)
    {
        dimSize_ = static_cast<int32_t>(tiling->dimSize);
        inner_ = LastAxis ? 1 : static_cast<int32_t>(tiling->inner);
        const int64_t rowsPerCore = tiling->elemsPerCore / (dimSize_ * inner_);
        firstRow_ = static_cast<int64_t>(GetBlockIdx()) * rowsPerCore;
        const int64_t coreEnd = firstRow_ + rowsPerCore;
        endRow_ = coreEnd < tiling->outer ? coreEnd : tiling->outer;
        tileRows_ = static_cast<uint32_t>(tiling->rowsPerChunk);
        varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ RawT*>(var), tiling->totalElems);
        const int64_t inputCount = tiling->outer * inner_;
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices), inputCount);
        updatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ RawT*>(updates), inputCount);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ RawT*>(y), tiling->totalElems);
        pipe->InitBuffer(varQueue_, BUFFER_NUM, static_cast<uint32_t>(tiling->tBufBytes));
        pipe->InitBuffer(indicesQueue_, BUFFER_NUM, static_cast<uint32_t>(tiling->idxBufBytes));
        pipe->InitBuffer(updatesQueue_, BUFFER_NUM, static_cast<uint32_t>(tiling->updBufBytes));
    }

    __aicore__ inline void Process()
    {
        const auto inputsReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        const auto outputReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        const auto tailReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        const int32_t inner = LastAxis ? 1 : inner_;
        const int32_t reciprocal = LastAxis ? 0 : ((1 << DIV_SHIFT) + inner - 1) / inner;
        for (int64_t base = firstRow_; base < endRow_;) {
            const int64_t remaining = endRow_ - base;
            const uint32_t rows = remaining < tileRows_ ? static_cast<uint32_t>(remaining) : tileRows_;
            const int64_t inputBase = base * inner;
            const int64_t varBase = inputBase * dimSize_;
            const uint32_t inputs = rows * inner;
            const uint32_t varBytes = inputs * dimSize_ * sizeof(RawT);
            const uint32_t idxBytes = inputs * sizeof(int32_t);
            const uint32_t updBytes = inputs * sizeof(RawT);
            LocalTensor<RawT> values = varQueue_.AllocTensor<RawT>();
            LocalTensor<int32_t> indices = indicesQueue_.AllocTensor<int32_t>();
            LocalTensor<RawT> updates = updatesQueue_.AllocTensor<RawT>();

            // A tail still performs one full input register load. Its unused lanes must be defined before Pack.
            const uint32_t tail = inputs % VF_ROWS;
            if (tail != 0) {
                Duplicate(indices[inputs - tail], int32_t{0}, VF_ROWS);
                auto updateWords = updates.template ReinterpretCast<uint32_t>();
                const uint32_t tailWord = (inputs - tail) * sizeof(RawT) / sizeof(uint32_t);
                const uint32_t vectorWords = VF_ROWS * sizeof(RawT) / sizeof(uint32_t);
                Duplicate(updateWords[tailWord], uint32_t{0}, vectorWords);
                SetFlag<HardEvent::V_MTE2>(tailReady);
                WaitFlag<HardEvent::V_MTE2>(tailReady);
            }

            const DataCopyExtParams varCopy{1, varBytes, 0, 0, 0};
            DataCopyPad(values, varGm_[varBase], varCopy, DataCopyPadExtParams<RawT>{false, 0, 0, 0});
            DataCopyPad(indices, indicesGm_[inputBase], DataCopyExtParams{1, idxBytes, 0, 0, 0},
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            DataCopyPad(updates, updatesGm_[inputBase], DataCopyExtParams{1, updBytes, 0, 0, 0},
                        DataCopyPadExtParams<RawT>{false, 0, 0, 0});
            indicesQueue_.EnQue(indices);
            updatesQueue_.EnQue(updates);
            indices = indicesQueue_.DeQue<int32_t>();
            updates = updatesQueue_.DeQue<RawT>();
            SetFlag<HardEvent::MTE2_V>(inputsReady);
            WaitFlag<HardEvent::MTE2_V>(inputsReady);
            asc_vf_call<ScatterRows>(reinterpret_cast<__ubuf__ RawT*>(values.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ int32_t*>(indices.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ RawT*>(updates.GetPhyAddr()), inputs, dimSize_, inner,
                                     reciprocal);
            SetFlag<HardEvent::V_MTE3>(outputReady);
            WaitFlag<HardEvent::V_MTE3>(outputReady);
            indicesQueue_.FreeTensor(indices);
            updatesQueue_.FreeTensor(updates);

            varQueue_.EnQue(values);
            values = varQueue_.DeQue<RawT>();
            DataCopyPad(yGm_[varBase], values, varCopy);
            varQueue_.FreeTensor(values);
            base += rows;
        }
    }

private:
    __simd_vf__ static inline void ScatterRows(__ubuf__ RawT* values, __ubuf__ int32_t* indices, __ubuf__ RawT* updates,
                                               uint32_t inputs, int32_t dimSize, int32_t inner, int32_t reciprocal)
    {
        auto all = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<int32_t> rowOffsets, rowIndex, zero;
        Reg::Arange(rowIndex, 0);
        if constexpr (LastAxis) {
            Reg::Muls(rowOffsets, rowIndex, dimSize, all);
        }
        Reg::Duplicate(zero, int32_t{0}, all);
        for (uint32_t row = 0; row < inputs; row += VF_ROWS) {
            uint32_t remaining = inputs - row;
            auto active = Reg::UpdateMask<int32_t>(remaining);
            Reg::RegTensor<int32_t> index, safeIndex, offsets;
            Reg::MaskReg geZero, ltDim, valid;
            Reg::LoadAlign(index, indices + row);
            Reg::Compares<int32_t, CMPMODE::GE>(geZero, index, int32_t{0}, active);
            Reg::Compares<int32_t, CMPMODE::LT>(ltDim, index, dimSize, active);
            Reg::MaskAnd(valid, geZero, ltDim, active);
            Reg::Select(safeIndex, index, zero, valid);

            auto* dst = values;
            if constexpr (LastAxis) {
                Reg::Add(offsets, rowOffsets, safeIndex, all);
                dst += row * dimSize;
            } else {
                Reg::RegTensor<int32_t> inputSlot, quotient, group, previousGroup;
                Reg::MaskReg correction;
                // With at most 32768 inputs, ceiling reciprocal division overestimates by at most one.
                Reg::Adds(inputSlot, rowIndex, static_cast<int32_t>(row), all);
                Reg::Muls(quotient, inputSlot, reciprocal, all);
                Reg::ShiftRights(quotient, quotient, DIV_SHIFT, all);
                Reg::Muls(group, quotient, inner, all);
                Reg::Compare<int32_t, CMPMODE::GT>(correction, group, inputSlot, all);
                Reg::Adds(previousGroup, group, -inner, all);
                Reg::Select(group, previousGroup, group, correction);
                // D==2: output offset is i + floor(i / inner) * inner + index * inner.
                Reg::Muls(safeIndex, safeIndex, inner, all);
                Reg::Add(offsets, inputSlot, group, all);
                Reg::Add(offsets, offsets, safeIndex, all);
            }
            auto& offsets32 = reinterpret_cast<Reg::RegTensor<uint32_t>&>(offsets);
            Reg::RegTensor<RawT> data;
            if constexpr (sizeof(RawT) == sizeof(uint32_t)) {
                Reg::LoadAlign(data, updates + row);
                Reg::Scatter(dst, data, offsets32, valid);
            } else {
                // Load exactly 64 updates into low portions of B32 lanes, then rearrange the original bits.
                if constexpr (sizeof(RawT) == sizeof(uint8_t)) {
                    Reg::LoadAlign<RawT, Reg::LoadDist::DIST_UNPACK4_B8>(data, updates + row);
                } else {
                    Reg::LoadAlign<RawT, Reg::LoadDist::DIST_UNPACK_B16>(data, updates + row);
                }
                Reg::RegTensor<uint16_t> offsets16, data16;
                Reg::MaskReg valid16;
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(offsets16, offsets32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                    data16, reinterpret_cast<Reg::RegTensor<uint32_t>&>(data));
                Reg::Pack(valid16, valid);
                // For B8, Scatter consumes the low byte (even byte lane) of each packed B16 slot.
                Reg::Scatter(dst, reinterpret_cast<Reg::RegTensor<RawT>&>(data16), offsets16, valid16);
            }
        }
    }

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> varQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> indicesQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> updatesQueue_;
    GlobalTensor<RawT> varGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<RawT> updatesGm_;
    GlobalTensor<RawT> yGm_;
    int64_t firstRow_ = 0;
    int64_t endRow_ = 0;
    uint32_t tileRows_ = 0;
    int32_t dimSize_ = 0;
    int32_t inner_ = 1;
};
} // namespace ArgMaxGrad

#endif // ARG_MAX_GRAD_SHORT_AXIS_H
