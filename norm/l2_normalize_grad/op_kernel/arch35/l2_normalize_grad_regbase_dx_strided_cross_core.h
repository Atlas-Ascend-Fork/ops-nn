/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been technically reviewed for functional accuracy.
 */

/*!
 * \file l2_normalize_grad_regbase_dx_strided_cross_core.h
 * \brief Share a few long strided reductions across D shards (7060).
 */
#ifndef L2_NORMALIZE_GRAD_REGBASE_DX_STRIDED_CROSS_CORE_H
#define L2_NORMALIZE_GRAD_REGBASE_DX_STRIDED_CROSS_CORE_H
#include "l2_normalize_grad_regbase_dx_strided_split.h"

namespace L2NormalizeGrad {
template <typename T_X>
class RegbaseDxStridedCrossCore : public RegbaseDxStridedSplit<T_X> {
    using Base = RegbaseDxStridedSplit<T_X>;

public:
    __aicore__ inline RegbaseDxStridedCrossCore(TPipe* pipe, const L2NormalizeGradTilingData* data)
        : Base(pipe, data), data_(data)
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace)
    {
        Base::Init(x, y, dy, dx);
        const int64_t group = GetBlockIdx() / data_->coresPerGroup;
        partial_.SetGlobalBuffer((__gm__ float*)workspace + group * 2 * data_->coresPerGroup * data_->slotStride);
    }
    __aicore__ inline void Process()
    {
        this->ZeroSums();
        int64_t slot = 0;
        for (int64_t d = 0; d < this->D_; d += this->dFactor_) {
            this->AccumChunk(0, d, Min(this->dFactor_, this->D_ - d), 0, this->inner_, slot);
            if (++slot == this->chunkSlots_) {
                this->FlushSlots(slot, this->inner_);
                slot = 0;
            }
        }
        if (slot > 0) {
            this->FlushSlots(slot, this->inner_);
        }
        MergePartials();
        for (int64_t d = 0; d < this->D_; d += this->dFactor_) {
            this->WriteChunk(0, d, Min(this->dFactor_, this->D_ - d), 0, this->inner_);
        }
    }

private:
    __aicore__ inline void MergePartials()
    {
        // One aligned slot per core and per statistic. Each core reads all
        // slots and performs the same tree reduction, without atomic updates.
        const uint32_t stride = static_cast<uint32_t>(data_->slotStride);
        const uint32_t total = static_cast<uint32_t>(data_->coresPerGroup) * stride;
        const uint32_t offset = (GetBlockIdx() % data_->coresPerGroup) * stride;
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(partial_[offset], this->sumSqBuf_.template Get<float>(), stride);
        DataCopy(partial_[total + offset], this->sumSBuf_.template Get<float>(), stride);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        SyncAll();
        LocalTensor<float> sq = this->slotSqBuf_.template Get<float>();
        LocalTensor<float> s = this->slotSBuf_.template Get<float>();
        DataCopy(sq, partial_, total);
        DataCopy(s, partial_[total], total);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        uint32_t shape[2] = {static_cast<uint32_t>(data_->coresPerGroup), stride};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(this->sumSqBuf_.template Get<float>(), sq, shape,
                                                                      false);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(this->sumSBuf_.template Get<float>(), s, shape,
                                                                      false);
    }
    const L2NormalizeGradTilingData* data_;
    GlobalTensor<float> partial_;
};
} // namespace L2NormalizeGrad
#endif
