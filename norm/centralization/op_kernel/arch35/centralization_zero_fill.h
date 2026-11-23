/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_ZERO_FILL_H_
#define OPS_NORM_CENTRALIZATION_ZERO_FILL_H_
#include "kernel_operator.h"
#include "centralization_tiling_data.h"

namespace Centralization {
using namespace AscendC;

// Key 9000 writes zeros without reading x. The host uses Fast for singleton reductions
// so that nonfinite inputs retain their x - x semantics.
template <typename T>
class ZeroFill {
public:
    __aicore__ inline ZeroFill(TPipe* pipe, const CentralizationTilingData* tiling) : pipe_(pipe), tiling_(tiling) {}
    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* workspace)
    {
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        const uint32_t core = GetBlockIdx();
        const uint32_t cores = GetBlockNum();
        // 输出条带：按 32B 块均分，各核起点对齐，最后一段按实际元素数写回
        constexpr int64_t kBlockElems = 32 / sizeof(T);
        const int64_t totalBlocks = (tiling_->totalCount + kBlockElems - 1) / kBlockElems;
        const int64_t blocksPerCore = (totalBlocks + cores - 1) / cores;
        begin_ = static_cast<int64_t>(core) * blocksPerCore * kBlockElems;
        end_ = Min(begin_ + blocksPerCore * kBlockElems, tiling_->totalCount);
        // UB 缓冲 ~128KB（chunk 按 32B 块对齐）
        const int64_t ubElems = 131072 / sizeof(T);
        chunkElems_ = ubElems / kBlockElems * kBlockElems;
        eventVToMte3_ = pipe_->FetchEventID(HardEvent::V_MTE3);
        eventMte3ToMte2_ = pipe_->FetchEventID(HardEvent::MTE3_MTE2);
        pipe_->InitBuffer(zeroBuf_, 131072);
    }

    __aicore__ inline void Process()
    {
        if (begin_ >= end_) {
            return;
        }
        LocalTensor<T> zeros = zeroBuf_.Get<T>();
        const int64_t bufferElems = 131072 / sizeof(T);
        Duplicate(zeros, static_cast<T>(0), static_cast<int32_t>(bufferElems));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3_);
        for (int64_t offset = begin_; offset < end_; offset += bufferElems) {
            const int64_t valid = Min(bufferElems, end_ - offset);
            DataCopyExtParams copy{1, static_cast<uint32_t>(valid * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[offset], zeros, copy);
        }
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    }

private:
    static constexpr int64_t kZeroBufBytes = 131072;
    __aicore__ inline int64_t Min(int64_t a, int64_t b) const { return a < b ? a : b; }

    TPipe* pipe_;
    const CentralizationTilingData* tiling_;
    GlobalTensor<T> yGm_;
    TBuf<TPosition::VECCALC> zeroBuf_;
    int64_t begin_ = 0;
    int64_t end_ = 0;
    int64_t chunkElems_ = 1;
    TEventID eventVToMte3_{};
    TEventID eventMte3ToMte2_{};
};
} // namespace Centralization
#endif
