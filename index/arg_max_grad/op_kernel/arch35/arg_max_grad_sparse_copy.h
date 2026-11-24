/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ARG_MAX_GRAD_SPARSE_COPY_H
#define ARG_MAX_GRAD_SPARSE_COPY_H

#include "kernel_operator.h"
#include "arg_max_grad_tiling_data.h"

namespace ArgMaxGrad {
using namespace AscendC;

template <typename T>
class ArgMaxGradSparseCopy {
public:
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y,
                                const ArgMaxGradArch35TilingData* tiling, TPipe* pipe)
    {
        static_assert(sizeof(T) == sizeof(RawT), "Sparse copy requires a 16-bit payload");
        dimSize_ = tiling->dimSize;
        inner_ = tiling->inner;
        first_ = static_cast<int64_t>(GetBlockIdx()) * tiling->elemsPerCore;
        const int64_t coreEnd = first_ + tiling->elemsPerCore;
        end_ = coreEnd < tiling->totalElems ? coreEnd : tiling->totalElems;
        tileElems_ = tiling->tBufBytes / sizeof(RawT);

        const int64_t inputCount = tiling->outer * inner_;
        varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ RawT*>(var), tiling->totalElems);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ RawT*>(y), tiling->totalElems);
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices), inputCount);
        updatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ RawT*>(updates), inputCount);
        pipe->InitBuffer(copyQueue_, BUFFER_NUM, static_cast<uint32_t>(tiling->tBufBytes));
        pipe->InitBuffer(indicesBuf_, static_cast<uint32_t>(tiling->idxBufBytes));
        pipe->InitBuffer(updatesBuf_, static_cast<uint32_t>(tiling->updBufBytes));
    }

    __aicore__ inline void Process()
    {
        if (first_ >= end_) {
            return;
        }
        LocalTensor<int32_t> offsets = indicesBuf_.Get<int32_t>();
        LocalTensor<RawT> values = updatesBuf_.Get<RawT>();
        const event_t copyReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        const event_t updatesReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        const int32_t owned = CollectUpdates(offsets, values, copyReady);
        const DataCopyPadExtParams<RawT> pad{false, 0, 0, 0};
        int32_t next = 0;
        for (int64_t base = first_; base < end_;) {
            const int64_t remaining = end_ - base;
            const int64_t count = remaining < tileElems_ ? remaining : tileElems_;
            const int64_t tileStart = base - first_;
            const int64_t tileEnd = tileStart + count;
            const DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(RawT)), 0, 0, 0};
            LocalTensor<RawT> tile = copyQueue_.AllocTensor<RawT>();
            DataCopyPad(tile, varGm_[base], copy, pad);
            if (next < owned && offsets.GetValue(next) < tileEnd) {
                SetFlag<HardEvent::MTE2_S>(copyReady);
                WaitFlag<HardEvent::MTE2_S>(copyReady);
                do {
                    const uint32_t relative = static_cast<uint32_t>(offsets.GetValue(next) - tileStart);
                    tile.SetValue(relative, values.GetValue(next));
                    ++next;
                } while (next < owned && offsets.GetValue(next) < tileEnd);
                SetFlag<HardEvent::S_MTE3>(updatesReady);
                WaitFlag<HardEvent::S_MTE3>(updatesReady);
            }
            copyQueue_.EnQue(tile);
            tile = copyQueue_.DeQue<RawT>();
            DataCopyPad(yGm_[base], tile, copy);
            copyQueue_.FreeTensor(tile);
            base += count;
        }
    }

private:
    // Narrow integer loads and stores preserve the FP16 payload, including NaNs and signed zero.
    using RawT = uint16_t;
    static constexpr uint32_t BUFFER_NUM = 2;

    __aicore__ inline int32_t CollectUpdates(LocalTensor<int32_t>& offsets, LocalTensor<RawT>& values, event_t ready)
    {
        const int64_t outerWidth = dimSize_ * inner_;
        const int64_t firstOuter = first_ / outerWidth;
        const int64_t endOuter = (end_ - 1) / outerWidth + 1;
        const int64_t inputStart = firstOuter * inner_;
        const int64_t inputCount = (endOuter - firstOuter) * inner_;
        const DataCopyExtParams indexCopy{1, static_cast<uint32_t>(inputCount * sizeof(int32_t)), 0, 0, 0};
        const DataCopyExtParams updateCopy{1, static_cast<uint32_t>(inputCount * sizeof(RawT)), 0, 0, 0};
        DataCopyPad(offsets, indicesGm_[inputStart], indexCopy, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        DataCopyPad(values, updatesGm_[inputStart], updateCopy, DataCopyPadExtParams<RawT>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(ready);
        WaitFlag<HardEvent::MTE2_S>(ready);

        int32_t owned = 0;
        int64_t input = 0;
        for (int64_t outer = firstOuter; outer < endOuter; ++outer) {
            const int64_t outerBase = outer * outerWidth;
            for (int64_t col = 0; col < inner_; ++col, ++input) {
                const int32_t index = offsets.GetValue(input);
                if (index < 0 || index >= dimSize_) {
                    continue;
                }
                const int64_t position = outerBase + static_cast<int64_t>(index) * inner_ + col;
                if (position < first_ || position >= end_) {
                    continue;
                }
                // Host bounds the core-relative offset to int32. The sorted prefix uses only consumed slots.
                const int32_t offset = static_cast<int32_t>(position - first_);
                const RawT value = values.GetValue(input);
                int32_t slot = owned;
                while (slot > 0 && offsets.GetValue(slot - 1) > offset) {
                    offsets.SetValue(slot, offsets.GetValue(slot - 1));
                    values.SetValue(slot, values.GetValue(slot - 1));
                    --slot;
                }
                offsets.SetValue(slot, offset);
                values.SetValue(slot, value);
                ++owned;
            }
        }
        return owned;
    }

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> copyQueue_;
    TBuf<QuePosition::VECCALC> indicesBuf_;
    TBuf<QuePosition::VECCALC> updatesBuf_;
    GlobalTensor<RawT> varGm_;
    GlobalTensor<int32_t> indicesGm_;
    GlobalTensor<RawT> updatesGm_;
    GlobalTensor<RawT> yGm_;
    int64_t dimSize_ = 0;
    int64_t inner_ = 0;
    int64_t first_ = 0;
    int64_t end_ = 0;
    int64_t tileElems_ = 0;
};
} // namespace ArgMaxGrad

#endif // ARG_MAX_GRAD_SPARSE_COPY_H
