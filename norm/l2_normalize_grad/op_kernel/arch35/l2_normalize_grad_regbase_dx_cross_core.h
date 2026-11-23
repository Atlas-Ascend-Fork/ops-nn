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
 * \file l2_normalize_grad_regbase_dx_cross_core.h
 * \brief L2NormalizeGrad cross-core D reduction kernel (TilingKey 7040).
 */
#ifndef L2_NORMALIZE_GRAD_REGBASE_DX_CROSS_CORE_H
#define L2_NORMALIZE_GRAD_REGBASE_DX_CROSS_CORE_H

#include "l2_normalize_grad_regbase_dx_split_d_base.h"

namespace L2NormalizeGrad {

template <typename T_X>
class RegbaseDxCrossCore : public RegbaseDxSplitDBase<T_X> {
    using Base = RegbaseDxSplitDBase<T_X>;
    using Base::accumBufS_;
    using Base::cols_;
    using Base::CopyIn2D;
    using Base::coreIdx_;
    using Base::dxGm_;
    using Base::dyGm_;
    using Base::eps_;
    using Base::FormerProcess;
    using Base::InitSplitDBuffers;
    using Base::inQueueDy_;
    using Base::inQueueX_;
    using Base::inQueueY_;
    using Base::LatterProcess;
    using Base::outQueueDx_;
    using Base::tiling_;
    using Base::tmpSumSBuf_;
    using Base::tmpSumSqBuf_;
    using Base::totalSHi_;
    using Base::totalSLo_;
    using Base::ubFactorD_;
    using Base::usedCoreNum_;
    using Base::xGm_;
    using Base::yGm_;

public:
    __aicore__ inline RegbaseDxCrossCore(TPipe* pipe, const L2NormalizeGradTilingData* tilingData)
        : Base(pipe, tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace)
    {
        coreIdx_ = GetBlockIdx();
        usedCoreNum_ = tiling_->usedCoreNum;
        eps_ = tiling_->eps;
        coresPerGroup_ = tiling_->coresPerGroup;
        shardId_ = coreIdx_ % coresPerGroup_;
        const int64_t group = coreIdx_ / coresPerGroup_;
        const int64_t shardOffset = static_cast<int64_t>(shardId_) * tiling_->coreDimLen;
        const int64_t offset = group * tiling_->dimLen + shardOffset;
        cols_ = Min(tiling_->coreDimLen, tiling_->dimLen - shardOffset);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + offset);
        yGm_.SetGlobalBuffer((__gm__ T_X*)y + offset);
        dyGm_.SetGlobalBuffer((__gm__ T_X*)dy + offset);
        dxGm_.SetGlobalBuffer((__gm__ T_X*)dx + offset);
        partialGm_.SetGlobalBuffer((__gm__ float*)workspace + group * coresPerGroup_ * 8);
        if constexpr (IsSameType<T_X, half>::value) {
            if (tiling_->orderedDotBlocks != 0) {
                orderedDotGm_.SetGlobalBuffer((__gm__ float*)workspace + coresPerGroup_ * PARTIAL_SLOT_FLOATS);
                orderedYGm_.SetGlobalBuffer((__gm__ T_X*)y);
                orderedDyGm_.SetGlobalBuffer((__gm__ T_X*)dy);
            }
        }
        InitSplitDBuffers();
    }

    __aicore__ inline void Process()
    {
        if constexpr (IsSameType<T_X, half>::value) {
            if (tiling_->orderedDotBlocks != 0) {
                this->template FormerProcess<false>(0);
                PublishOrderedDot();
                MergePartials();
                LoadOrderedDot();
                LatterProcess(0);
                return;
            }
        }
        FormerProcess(0);
        MergePartials();
        this->template LatterProcess<IsSameType<T_X, float>::value>(0);
    }

private:
    static constexpr uint32_t PARTIAL_SLOT_FLOATS = 8;

    __aicore__ inline void AccumulateDotGroup(RegTensor<float>& product, RegTensor<float>& first,
                                              RegTensor<float>& second, RegTensor<float>& third,
                                              RegTensor<float>& fourth, uint32_t group, MaskReg& lanes)
    {
        Add(first, first, product, lanes);
        if (group % ORDERED_DOT_GROUP == 0) {
            Add(second, second, first, lanes);
            Duplicate(first, 0.0F, lanes);
            if (group % (ORDERED_DOT_GROUP * ORDERED_DOT_GROUP) == 0) {
                Add(third, third, second, lanes);
                Duplicate(second, 0.0F, lanes);
                if (group == ORDERED_DOT_BLOCK / ORDERED_DOT_WIDTH) {
                    Add(fourth, fourth, third, lanes);
                    Duplicate(third, 0.0F, lanes);
                }
            }
        }
    }

    __aicore__ inline void AccumulateDotTile(int64_t start, uint32_t count, uint32_t groupsBefore,
                                             LocalTensor<float>& sums)
    {
        CopyIn2D(inQueueY_, orderedYGm_, start, 1, count, 0);
        CopyIn2D(inQueueDy_, orderedDyGm_, start, 1, count, 0);
        auto y = inQueueY_.template DeQue<float>();
        auto dy = inQueueDy_.template DeQue<float>();
        auto yAddr = (__ubuf__ T_X*)y.GetPhyAddr();
        auto dyAddr = (__ubuf__ T_X*)dy.GetPhyAddr();
        auto sumAddr = (__ubuf__ float*)sums.GetPhyAddr();
        const uint16_t repeats = static_cast<uint16_t>(DivCeil(count, V_LENGTH));
        __VEC_SCOPE__
        {
            LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            RegTensor<float> first, second, third, fourth, yReg, dyReg, product, upper;
            RegTensor<int32_t> upperIndex;
            uint32_t width = ORDERED_DOT_WIDTH;
            MaskReg lanes = UpdateMask<float>(width);
            Arange(upperIndex, static_cast<int32_t>(ORDERED_DOT_WIDTH));
            DataCopy(first, sumAddr);
            DataCopy(second, sumAddr + ORDERED_DOT_WIDTH);
            DataCopy(third, sumAddr + 2 * ORDERED_DOT_WIDTH);
            DataCopy(fourth, sumAddr + 3 * ORDERED_DOT_WIDTH);
            uint32_t remaining = count;
            for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                const uint32_t offset = repeat * V_LENGTH;
                MaskReg valid = UpdateMask<float>(remaining);
                LoadAndCast(yReg, yAddr, valid, offset);
                LoadAndCast(dyReg, dyAddr, valid, offset);
                Mul(product, yReg, dyReg, valid);
                uint32_t group = groupsBefore + offset / ORDERED_DOT_WIDTH + 1;
                AccumulateDotGroup(product, first, second, third, fourth, group, lanes);
                if (offset + ORDERED_DOT_WIDTH < count) {
                    Gather<float, uint32_t>(upper, product, (RegTensor<uint32_t>&)upperIndex);
                    AccumulateDotGroup(upper, first, second, third, fourth, group + 1, lanes);
                }
            }
            DataCopy(sumAddr, first, lanes);
            DataCopy(sumAddr + ORDERED_DOT_WIDTH, second, lanes);
            DataCopy(sumAddr + 2 * ORDERED_DOT_WIDTH, third, lanes);
            DataCopy(sumAddr + 3 * ORDERED_DOT_WIDTH, fourth, lanes);
        }
        inQueueY_.FreeTensor(y);
        inQueueDy_.FreeTensor(dy);
    }

    __aicore__ inline void StoreTailProducts(int64_t offset, uint32_t count, LocalTensor<float>& tail)
    {
        if (count == 0) {
            AscendC::Duplicate(tail, 0.0F, ORDERED_DOT_WIDTH);
            return;
        }
        CopyIn2D(inQueueY_, orderedYGm_, offset, 1, count, 0);
        CopyIn2D(inQueueDy_, orderedDyGm_, offset, 1, count, 0);
        auto y = inQueueY_.template DeQue<float>();
        auto dy = inQueueDy_.template DeQue<float>();
        auto yAddr = (__ubuf__ T_X*)y.GetPhyAddr();
        auto dyAddr = (__ubuf__ T_X*)dy.GetPhyAddr();
        auto tailAddr = (__ubuf__ float*)tail.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> yReg, dyReg, product;
            MaskReg valid = UpdateMask<float>(count);
            LoadAndCast(yReg, yAddr, valid, 0);
            LoadAndCast(dyReg, dyAddr, valid, 0);
            Mul(product, yReg, dyReg, valid);
            uint32_t width = ORDERED_DOT_WIDTH;
            MaskReg lanes = UpdateMask<float>(width);
            DataCopy(tailAddr, product, lanes);
        }
        inQueueY_.FreeTensor(y);
        inQueueDy_.FreeTensor(dy);
    }

    // Workspace order is defined by the reference's logical blocks, independent of the core shards.
    __aicore__ inline void PublishOrderedDot()
    {
        const int64_t blocks = tiling_->orderedDotBlocks;
        auto sums = accumBufS_.template Get<float>();
        for (int64_t block = shardId_; block <= blocks; block += coresPerGroup_) {
            const bool isTail = block == blocks;
            const int64_t start = block * ORDERED_DOT_BLOCK;
            const int64_t length = isTail ? tiling_->dimLen - start : ORDERED_DOT_BLOCK;
            const int64_t completeGroups = length / ORDERED_DOT_WIDTH * ORDERED_DOT_WIDTH;
            AscendC::Duplicate(sums, 0.0F, ORDERED_DOT_TAIL_SLOTS * ORDERED_DOT_WIDTH);
            for (int64_t offset = 0; offset < completeGroups; offset += ubFactorD_) {
                const uint32_t count = static_cast<uint32_t>(Min(ubFactorD_, completeGroups - offset));
                AccumulateDotTile(start + offset, count, offset / ORDERED_DOT_WIDTH, sums);
            }
            auto last = sums[3 * ORDERED_DOT_WIDTH];
            if (isTail) {
                StoreTailProducts(start + completeGroups, length - completeGroups, last);
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            if (isTail) {
                DataCopy(orderedDotGm_[block * ORDERED_DOT_WIDTH], sums, ORDERED_DOT_TAIL_SLOTS * ORDERED_DOT_WIDTH);
            } else {
                DataCopy(orderedDotGm_[block * ORDERED_DOT_WIDTH], last, ORDERED_DOT_WIDTH);
            }
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void LoadOrderedDot()
    {
        const uint16_t blocks = static_cast<uint16_t>(tiling_->orderedDotBlocks);
        auto partials = inQueueX_.template AllocTensor<float>();
        DataCopy(partials, orderedDotGm_, (blocks + ORDERED_DOT_TAIL_SLOTS) * ORDERED_DOT_WIDTH);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        auto addr = (__ubuf__ float*)partials.GetPhyAddr();
        auto sums = accumBufS_.template Get<float>();
        auto sumAddr = (__ubuf__ float*)sums.GetPhyAddr();
        __VEC_SCOPE__
        {
            uint32_t width = ORDERED_DOT_WIDTH;
            MaskReg lanes = UpdateMask<float>(width);
            RegTensor<float> complete, part, total;
            Duplicate(complete, 0.0F, lanes);
            for (uint16_t block = 0; block < blocks; ++block) {
                DataCopy(part, addr + block * ORDERED_DOT_WIDTH);
                Add(complete, complete, part, lanes);
            }
            auto tailAddr = addr + blocks * ORDERED_DOT_WIDTH;
            DataCopy(total, tailAddr);
            DataCopy(part, tailAddr + ORDERED_DOT_WIDTH);
            Add(total, total, part, lanes);
            DataCopy(part, tailAddr + 2 * ORDERED_DOT_WIDTH);
            Add(total, total, part, lanes);
            Add(total, total, complete, lanes);
            DataCopy(sumAddr, total, lanes);
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const auto tail = partials[(blocks + 3) * ORDERED_DOT_WIDTH];
        const uint32_t tailCount = tiling_->dimLen % ORDERED_DOT_WIDTH;
        const uint32_t vectorTail = tailCount / ORDERED_DOT_LANES * ORDERED_DOT_LANES;
        float dot = 0.0F;
        for (uint32_t i = vectorTail; i < tailCount; ++i) {
            dot += tail.GetValue(i);
        }
        for (uint32_t lane = 0; lane < ORDERED_DOT_LANES; ++lane) {
            float value = sums.GetValue(lane);
            for (uint32_t i = lane; i < vectorTail; i += ORDERED_DOT_LANES) {
                value += tail.GetValue(i);
            }
            for (uint32_t i = ORDERED_DOT_LANES + lane; i < ORDERED_DOT_WIDTH; i += ORDERED_DOT_LANES) {
                value += sums.GetValue(i);
            }
            dot += value;
        }
        inQueueX_.FreeTensor(partials);
        tmpSumSBuf_.template Get<float>().SetValue(0, dot);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }

    // Each core owns a 32B workspace slot. Publish before the global barrier, then
    // combine in a fixed core order on every core; no atomics or rounded s partials.
    __aicore__ inline void MergePartials()
    {
        constexpr uint32_t slotFloats = PARTIAL_SLOT_FLOATS;
        LocalTensor<float> out = outQueueDx_.template AllocTensor<float>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t i = 0; i < slotFloats; ++i) {
            out.SetValue(i, 0.0f);
        }
        out.SetValue(0, tmpSumSqBuf_.template Get<float>().GetValue(0));
        out.SetValue(1, totalSHi_);
        out.SetValue(2, totalSLo_);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(partialGm_[shardId_ * slotFloats], out, slotFloats);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        outQueueDx_.FreeTensor(out);
        SyncAll();

        LocalTensor<float> partial = inQueueX_.template AllocTensor<float>();
        DataCopy(partial, partialGm_, coresPerGroup_ * slotFloats);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        float sqHi = 0.0f;
        float sqLo = 0.0f;
        float sHi = 0.0f;
        float sLo = 0.0f;
        for (uint32_t i = 0; i < coresPerGroup_; ++i) {
            const uint32_t off = i * slotFloats;
            DfAccumScalar(sqHi, sqLo, partial.GetValue(off), 0.0f);
            DfAccumScalar(sHi, sLo, partial.GetValue(off + 1), partial.GetValue(off + 2));
        }
        inQueueX_.FreeTensor(partial);
        tmpSumSqBuf_.template Get<float>().SetValue(0, sqHi + sqLo);
        tmpSumSBuf_.template Get<float>().SetValue(0, sHi + sLo);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }

    uint32_t coresPerGroup_ = 0;
    uint32_t shardId_ = 0;
    GlobalTensor<float> partialGm_;
    GlobalTensor<float> orderedDotGm_;
    GlobalTensor<T_X> orderedYGm_;
    GlobalTensor<T_X> orderedDyGm_;
};
} // namespace L2NormalizeGrad
#endif // L2_NORMALIZE_GRAD_REGBASE_DX_CROSS_CORE_H
