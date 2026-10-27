/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef NORM_BATCH_NORM3D_COMPENSATED_H
#define NORM_BATCH_NORM3D_COMPENSATED_H

#include "batch_norm3d_base.h"

namespace BatchNorm3DOps {
namespace CompensatedBN {
using namespace AscendC::Reg;
constexpr uint32_t WIDTH = 64;
constexpr uint32_t SLOTS = 20;

// Reduce rows while preserving each channel's high/low expansion. The
// power-of-two row pitch also permits a register Gather to broadcast results.
__aicore__ inline void ReduceRows(RegTensor<float>& hi, RegTensor<float>& lo, uint16_t channels, bool maximum)
{
    RegTensor<int32_t> index, rhsIndex, channelIndex, bits;
    RegTensor<float> rhsHi, rhsLo, resultHi, resultLo;
    auto full = CreateMask<float, MaskPattern::ALL>();
    Arange(index, int32_t(0));
    for (uint16_t active = WIDTH / 2; active >= channels; active /= 2) {
        uint32_t count = active;
        auto mask = UpdateMask<float>(count);
        Adds(rhsIndex, index, static_cast<int32_t>(active), full);
        Gather<float, uint32_t>(rhsHi, hi, (RegTensor<uint32_t>&)rhsIndex);
        if (maximum) {
            Max(hi, hi, rhsHi, mask);
        } else {
            Gather<float, uint32_t>(rhsLo, lo, (RegTensor<uint32_t>&)rhsIndex);
            AddPair(hi, lo, hi, lo, rhsHi, rhsLo, mask);
        }
    }
    Duplicate(bits, static_cast<int32_t>(channels - 1), full);
    And(channelIndex, index, bits, full);
    Gather<float, uint32_t>(resultHi, hi, (RegTensor<uint32_t>&)channelIndex);
    Gather<float, uint32_t>(resultLo, lo, (RegTensor<uint32_t>&)channelIndex);
    Adds(hi, resultHi, 0.0f, full);
    Adds(lo, resultLo, 0.0f, full);
}
} // namespace CompensatedBN

template <typename T>
class BatchNorm3DCompensated {
    static constexpr uint32_t WIDTH = CompensatedBN::WIDTH;
    static constexpr uint32_t ROWS = 256;
    static constexpr uint32_t SPATIAL_TILE = 4096;
    static constexpr uint32_t BUFFER_FLOATS = ROWS * WIDTH + WIDTH;

public:
    __aicore__ inline BatchNorm3DCompensated(const BatchNorm3DCompensatedTilingData* data, TPipe* pipe)
        : data_(data), pipe_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR mean, GM_ADDR variance, GM_ADDR y,
                                GM_ADDR batchMean, GM_ADDR batchVariance, GM_ADDR reserveMean, GM_ADDR reserveRstd,
                                GM_ADDR workspace)
    {
        x_.SetGlobalBuffer((__gm__ T*)x);
        scale_.SetGlobalBuffer((__gm__ float*)scale);
        offset_.SetGlobalBuffer((__gm__ float*)offset);
        mean_.SetGlobalBuffer((__gm__ float*)mean);
        variance_.SetGlobalBuffer((__gm__ float*)variance);
        y_.SetGlobalBuffer((__gm__ T*)y);
        outputs_[0].SetGlobalBuffer((__gm__ float*)batchMean);
        outputs_[1].SetGlobalBuffer((__gm__ float*)batchVariance);
        outputs_[2].SetGlobalBuffer((__gm__ float*)reserveMean);
        outputs_[3].SetGlobalBuffer((__gm__ float*)reserveRstd);
        workspace_.SetGlobalBuffer((__gm__ float*)workspace);
        pipe_->InitBuffer(input_, 1, BUFFER_FLOATS * sizeof(float));
        pipe_->InitBuffer(output_, 1, BUFFER_FLOATS * sizeof(float));
        pipe_->InitBuffer(stats_, CompensatedBN::SLOTS * WIDTH * sizeof(float));
        channelLast_ = data_->r0 == 1 && data_->a > 1;
    }

    __aicore__ inline void Process()
    {
        const int64_t channelTile = channelLast_ ? WIDTH : 1;
        const int64_t reduceBlock = GetBlockIdx() % data_->reduceBlocks;
        const int64_t totalR = data_->r1 * data_->r0;
        const int64_t perBlock = (totalR + data_->reduceBlocks - 1) / data_->reduceBlocks;
        const int64_t firstR = reduceBlock * perBlock;
        const int64_t lastR = totalR < firstR + perBlock ? totalR : firstR + perBlock;
        for (int64_t c = GetBlockIdx() / data_->reduceBlocks * channelTile; c < data_->a;
             c += data_->blockNum / data_->reduceBlocks * channelTile) {
            channels_ = static_cast<uint32_t>((channelTile < data_->a - c ? channelTile : data_->a - c));
            stride_ = channelLast_ ? BLOCK_SIZE / sizeof(T) : WIDTH;
            while (stride_ < channels_) {
                stride_ *= 2;
            }
            ResetAndLoad(c);
            for (uint32_t stage = 0; stage < 4; ++stage) {
                const int64_t tile = channelLast_ ? ROWS : SPATIAL_TILE;
                for (int64_t r = firstR; r < lastR;) {
                    int64_t available = lastR - r;
                    if (!channelLast_ && data_->a > 1) {
                        const int64_t rowRemaining = data_->r0 - r % data_->r0;
                        available = available < rowRemaining ? available : rowRemaining;
                    }
                    const uint32_t count = static_cast<uint32_t>(tile < available ? tile : available);
                    const int64_t index = channelLast_ ? r * data_->a + c :
                                                         (r / data_->r0 * data_->a + c) * data_->r0 + r % data_->r0;
                    ProcessTile(index, count, stage);
                    r += count;
                }
                if (stage < 3) {
                    MergeCores(stage);
                    FinalizeStage(stage);
                }
            }
            if (reduceBlock == 0) {
                CopyStatistics(c);
            }
        }
    }

private:
    __aicore__ inline void MergeCores(uint32_t stage)
    {
        using namespace AscendC::Reg;
        using namespace CompensatedBN;
        if (data_->reduceBlocks == 1) {
            return;
        }
        auto stats = stats_.Get<float>();
        const uint32_t slot = stage == 0 ? 4 : stage == 2 ? 2 : 0;
        auto event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        DataCopy(workspace_[GetBlockIdx() * 2 * WIDTH], stats[slot * WIDTH], 2 * WIDTH);
        SyncAll();
        auto partial = input_.AllocTensor<float>();
        const int64_t group = GetBlockIdx() / data_->reduceBlocks;
        DataCopy(partial, workspace_[group * data_->reduceBlocks * 2 * WIDTH],
                 static_cast<uint32_t>(data_->reduceBlocks * 2 * WIDTH));
        input_.EnQue(partial);
        partial = input_.DeQue<float>();
        auto p = (__ubuf__ float*)partial.GetPhyAddr();
        auto s = (__ubuf__ float*)stats.GetPhyAddr();
        const uint16_t blocks = data_->reduceBlocks;
        __VEC_SCOPE__
        {
            auto mask = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> hi, lo, ph, pl;
            Duplicate(hi, 0.0f, mask);
            Duplicate(lo, 0.0f, mask);
            for (uint16_t b = 0; b < blocks; ++b) {
                LoadAlign(ph, p + b * 2 * WIDTH);
                if (stage == 0) {
                    Max(hi, hi, ph, mask);
                } else {
                    LoadAlign(pl, p + b * 2 * WIDTH + WIDTH);
                    AddPair(hi, lo, hi, lo, ph, pl, mask);
                }
            }
            StoreAlign(s + slot * WIDTH, hi, mask);
            if (stage > 0) {
                StoreAlign(s + (slot + 1) * WIDTH, lo, mask);
            }
        }
        input_.FreeTensor(partial);
        // Every reader must finish before another stage reuses the workspace.
        SyncAll();
    }

    __aicore__ inline void ResetAndLoad(int64_t c)
    {
        auto stats = stats_.Get<float>();
        auto ptr = (__ubuf__ float*)stats.GetPhyAddr();
        __VEC_SCOPE__
        {
            auto mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::RegTensor<float> zero;
            AscendC::Reg::Duplicate(zero, 0.0f, mask);
            for (uint16_t i = 0; i < static_cast<uint16_t>(CompensatedBN::SLOTS); ++i) {
                AscendC::Reg::StoreAlign(ptr + i * WIDTH, zero, mask);
            }
        }
        auto resetEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(resetEvent);
        WaitFlag<HardEvent::V_MTE2>(resetEvent);
        DataCopyExtParams copy{1, channels_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(stats[9 * WIDTH], scale_[c], copy, pad);
        DataCopyPad(stats[10 * WIDTH], offset_[c], copy, pad);
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(stats[17 * WIDTH], x_[c * data_->r0], copy, pad);
        }
        if (data_->useRunningMeanVar) {
            DataCopyPad(stats[11 * WIDTH], mean_[c], copy, pad);
            DataCopyPad(stats[12 * WIDTH], variance_[c], copy, pad);
        }
        auto event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
        if constexpr (!IsSameType<T, float>::value) {
            auto reference = input_.AllocTensor<T>();
            DataCopyExtParams refCopy{1, channels_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> refPad{false, 0, 0, 0};
            DataCopyPad(reference, x_[c * data_->r0], refCopy, refPad);
            input_.EnQue(reference);
            reference = input_.DeQue<T>();
            auto refPtr = (__ubuf__ T*)reference.GetPhyAddr();
            uint32_t length = channels_;
            __VEC_SCOPE__
            {
                auto mask = UpdateMask<float>(length);
                RegTensor<float> value;
                LoadTensorForDtypeT<T>(value, refPtr, mask, 0);
                AscendC::Reg::StoreAlign(ptr + 17 * WIDTH, value, mask);
            }
            input_.FreeTensor(reference);
        }
    }

    __aicore__ inline void ProcessTile(int64_t index, uint32_t count, uint32_t stage)
    {
        constexpr uint32_t alignElements = BLOCK_SIZE / sizeof(T);
        auto input = input_.AllocTensor<T>();
        DataCopyExtParams copy{
            static_cast<uint16_t>(channelLast_ ? count : 1),
            (channelLast_ ? channels_ : count) * static_cast<uint32_t>(sizeof(T)),
            static_cast<uint32_t>(channelLast_ ? (data_->a - channels_) * sizeof(T) : 0),
            channelLast_ ? (stride_ - (channels_ + alignElements - 1) / alignElements * alignElements) / alignElements :
                           0,
            0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(input, x_[index], copy, pad);
        input_.EnQue(input);
        input = input_.DeQue<T>();
        auto output = output_.AllocTensor<T>();
        ComputeTile((__ubuf__ T*)input.GetPhyAddr(), (__ubuf__ T*)output.GetPhyAddr(), count, stage);
        input_.FreeTensor(input);
        if (stage == 3) {
            output_.EnQue(output);
            output = output_.DeQue<T>();
            const uint32_t ubGap = copy.dstStride;
            copy.dstStride = copy.srcStride;
            copy.srcStride = ubGap;
            DataCopyPad(y_[index], output, copy);
        }
        output_.FreeTensor(output);
    }

    __aicore__ inline void ComputeTile(__ubuf__ T* input, __ubuf__ T* output, uint32_t count, uint32_t stage)
    {
        using namespace AscendC::Reg;
        using namespace CompensatedBN;
        auto s = (__ubuf__ float*)stats_.Get<float>().GetPhyAddr();
        const uint32_t elements = channelLast_ ? count * stride_ : count;
        const uint16_t loops = (elements + WIDTH - 1) / WIDTH;
        const uint32_t stride = stride_;
        const uint32_t width = channels_;
        const bool last = channelLast_;
        __VEC_SCOPE__
        {
            auto full = CreateMask<float, MaskPattern::ALL>();
            RegTensor<int32_t> columns, columnMask;
            MaskReg validColumns;
            Arange(columns, int32_t(0));
            Duplicate(columnMask, static_cast<int32_t>(stride - 1), full);
            And(columns, columns, columnMask, full);
            CompareScalar<int32_t, CMPMODE::LT>(validColumns, columns, static_cast<int32_t>(width), full);
            RegTensor<float> hi, lo, x, xl, tmp, th, tl, zero, down, meanHi, meanLo, rstdHi, rstdLo, gamma, beta,
                reference;
            Duplicate(zero, 0.0f, full);
            LoadAlign(hi, s + (stage == 0 ? 4 : stage == 2 ? 2 : 0) * WIDTH);
            LoadAlign(lo, s + (stage == 2 ? 3 : 1) * WIDTH);
            LoadAlign(down, s + 5 * WIDTH);
            LoadAlign(meanHi, s);
            LoadAlign(meanLo, s + WIDTH);
            LoadAlign(rstdHi, s + 7 * WIDTH);
            LoadAlign(rstdLo, s + 8 * WIDTH);
            LoadAlign(gamma, s + 9 * WIDTH);
            LoadAlign(beta, s + 10 * WIDTH);
            LoadAlign(reference, s + 17 * WIDTH);
            FiniteResidual(reference, full);
            Mul(reference, reference, down, full);
            Muls(reference, reference, -1.0f, full);
            for (uint16_t i = 0; i < loops; ++i) {
                uint32_t remaining = elements - i * WIDTH;
                auto mask = UpdateMask<float>(remaining);
                if (last) {
                    And(mask, mask, validColumns, full);
                }
                LoadTensorForDtypeT<T>(x, input, mask, i * WIDTH);
                if (stage == 0) {
                    Abs(tmp, x, mask);
                    Max(hi, hi, tmp, full);
                } else {
                    Mul(x, x, down, mask);
                    if (stage == 1) {
                        // Center the sum on an input value. A constant channel
                        // then has an exactly zero sum and no reciprocal error.
                        TwoSum(x, xl, x, reference, mask);
                        AddPair(hi, lo, hi, lo, x, xl, full);
                    } else {
                        Muls(tmp, meanHi, -1.0f, mask);
                        TwoSum(x, xl, x, tmp, mask);
                        Sub(xl, xl, meanLo, mask);
                        TwoSum(x, xl, x, xl, mask);
                        if (stage == 2) {
                            MulPair(th, tl, x, xl, x, xl, mask);
                            AddPair(hi, lo, hi, lo, th, tl, full);
                        } else {
                            MulPair(th, tl, x, xl, rstdHi, rstdLo, mask);
                            Normalize(th, tl, gamma, beta, mask);
                            StoreTensorForDtypeT<T>(output, th, mask, i * WIDTH);
                        }
                    }
                }
            }
            if (stage < 3) {
                StoreAlign(s + (stage == 0 ? 4 : stage == 2 ? 2 : 0) * WIDTH, hi, full);
                if (stage > 0) {
                    StoreAlign(s + (stage == 2 ? 3 : 1) * WIDTH, lo, full);
                }
            }
        }
    }

    __aicore__ inline void Normalize(AscendC::Reg::RegTensor<float>& hi, AscendC::Reg::RegTensor<float>& lo,
                                     AscendC::Reg::RegTensor<float>& gamma, AscendC::Reg::RegTensor<float>& beta,
                                     AscendC::Reg::MaskReg& mask)
    {
        using namespace AscendC::Reg;
        using namespace CompensatedBN;
        RegTensor<float> absolute, temp, down, up, gs, bs, zero, ordinary, result;
        MaskReg large, finiteGamma;
        Abs(absolute, gamma, mask);
        Abs(temp, beta, mask);
        Max(absolute, absolute, temp, mask);
        CompareScalar<float, CMPMODE::GT>(large, absolute, 0x1p32f, mask);
        Duplicate(down, 1.0f, mask);
        Duplicate(up, 0x1p-64f, mask);
        Select(down, up, down, large);
        Duplicate(up, 1.0f, mask);
        Duplicate(temp, 0x1p64f, mask);
        Select(up, temp, up, large);
        Duplicate(zero, 0.0f, mask);
        Add(ordinary, hi, lo, mask);
        Mul(ordinary, ordinary, gamma, mask);
        Add(ordinary, ordinary, beta, mask);
        Mul(gs, gamma, down, mask);
        Mul(bs, beta, down, mask);
        MulPair(hi, lo, hi, lo, gs, zero, mask);
        AddPair(hi, lo, hi, lo, bs, zero, mask);
        Add(result, hi, lo, mask);
        Mul(result, result, up, mask);
        Abs(absolute, gamma, mask);
        CompareScalar<float, CMPMODE::LE>(finiteGamma, absolute, 0x1.fffffep127f, mask);
        Select(hi, result, ordinary, finiteGamma);
    }

    __aicore__ inline void FinalizeStage(uint32_t stage)
    {
        using namespace AscendC::Reg;
        using namespace CompensatedBN;
        auto s = (__ubuf__ float*)stats_.Get<float>().GetPhyAddr();
        const uint16_t period = channelLast_ ? stride_ : 1;
        const float inverse = data_->invCount;
        const float inverseLow = data_->invCountLow;
        const float epsilon = data_->epsilon;
        __VEC_SCOPE__
        {
            auto mask = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> hi, lo, rh, rl, zero, temp, down, up;
            Duplicate(zero, 0.0f, mask);
            LoadAlign(hi, s + (stage == 0 ? 4 : stage == 2 ? 2 : 0) * WIDTH);
            LoadAlign(lo, s + (stage == 2 ? 3 : 1) * WIDTH);
            ReduceRows(hi, lo, period, stage == 0);
            if (stage == 0) {
                MaskReg large;
                CompareScalar<float, CMPMODE::GT>(large, hi, 0x1p32f, mask);
                // For large channels, place the largest magnitude in [2^16,
                // 2^17). Both scale factors remain normal binary32 values;
                // smaller large inputs must not use the FLT_MAX scaling.
                RegTensor<uint32_t> exponent, encodedDown, encodedUp, shift;
                ShiftRights(exponent, (RegTensor<uint32_t>&)hi, int16_t(23), mask);
                Duplicate(shift, uint32_t(16), mask);
                Sub(encodedUp, exponent, shift, mask);
                ShiftLefts(encodedUp, encodedUp, int16_t(23), mask);
                Duplicate(encodedDown, uint32_t(270), mask);
                Sub(encodedDown, encodedDown, exponent, mask);
                ShiftLefts(encodedDown, encodedDown, int16_t(23), mask);
                Duplicate(down, 1.0f, mask);
                Select(down, (RegTensor<float>&)encodedDown, down, large);
                Duplicate(up, 1.0f, mask);
                Select(up, (RegTensor<float>&)encodedUp, up, large);
                StoreAlign(s + 5 * WIDTH, down, mask);
                StoreAlign(s + 6 * WIDTH, up, mask);
            } else {
                Duplicate(rh, inverse, mask);
                Duplicate(rl, inverseLow, mask);
                MulPair(hi, lo, hi, lo, rh, rl, mask);
                if (stage == 1) {
                    LoadAlign(temp, s + 17 * WIDTH);
                    FiniteResidual(temp, mask);
                    LoadAlign(down, s + 5 * WIDTH);
                    Mul(temp, temp, down, mask);
                    AddPair(hi, lo, hi, lo, temp, zero, mask);
                }
                StoreAlign(s + (stage == 1 ? 0 : 2) * WIDTH, hi, mask);
                StoreAlign(s + (stage == 1 ? 1 : 3) * WIDTH, lo, mask);
                if (stage == 2) {
                    MaskReg constant;
                    CompareScalar<float, CMPMODE::EQ>(constant, hi, 0.0f, mask);
                    LoadAlign(down, s + 5 * WIDTH);
                    Duplicate(temp, epsilon, mask);
                    Mul(temp, temp, down, mask);
                    Mul(temp, temp, down, mask);
                    Duplicate(rh, epsilon, mask);
                    Select(temp, rh, temp, constant);
                    AddPair(hi, lo, hi, lo, temp, zero, mask);
                    ComputeInverseRoot(rh, rl, hi, lo, mask);
                    StoreAlign(s + 7 * WIDTH, rh, mask);
                    StoreAlign(s + 8 * WIDTH, rl, mask);
                }
            }
        }
        if (stage == 2) {
            FinalizeStatistics();
        } else if (stage == 0) {
            // Broadcast channel parameters over the rows packed in a vector.
            __VEC_SCOPE__
            {
                auto mask = CreateMask<float, MaskPattern::ALL>();
                RegTensor<float> value, repeated;
                RegTensor<int32_t> index, bits;
                Arange(index, int32_t(0));
                Duplicate(bits, static_cast<int32_t>(period - 1), mask);
                And(index, index, bits, mask);
                for (uint16_t slot = 9; slot <= 12; ++slot) {
                    LoadAlign(value, s + slot * WIDTH);
                    Gather<float, uint32_t>(repeated, value, (RegTensor<uint32_t>&)index);
                    StoreAlign(s + slot * WIDTH, repeated, mask);
                }
                LoadAlign(value, s + 17 * WIDTH);
                Gather<float, uint32_t>(repeated, value, (RegTensor<uint32_t>&)index);
                StoreAlign(s + 17 * WIDTH, repeated, mask);
            }
        }
    }

    __aicore__ inline void FinalizeStatistics()
    {
        using namespace AscendC::Reg;
        auto s = (__ubuf__ float*)stats_.Get<float>().GetPhyAddr();
        const float correction = data_->varianceCorrection;
        const float momentum = data_->momentum;
        const bool running = data_->useRunningMeanVar;
        __VEC_SCOPE__
        {
            auto mask = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> mean, variance, rstd, low, up, down, old;
            LoadAlign(up, s + 6 * WIDTH);
            LoadAlign(down, s + 5 * WIDTH);
            LoadAlign(mean, s);
            LoadAlign(low, s + WIDTH);
            Add(mean, mean, low, mask);
            Mul(mean, mean, up, mask);
            StoreAlign(s + 15 * WIDTH, mean, mask);
            LoadAlign(variance, s + 2 * WIDTH);
            MaskReg constant;
            CompareScalar<float, CMPMODE::EQ>(constant, variance, 0.0f, mask);
            Duplicate(old, 1.0f, mask);
            Select(down, old, down, constant);
            LoadAlign(low, s + 3 * WIDTH);
            Add(variance, variance, low, mask);
            Muls(variance, variance, correction, mask);
            Mul(variance, variance, up, mask);
            Mul(variance, variance, up, mask);
            LoadAlign(rstd, s + 7 * WIDTH);
            LoadAlign(low, s + 8 * WIDTH);
            Add(rstd, rstd, low, mask);
            Mul(rstd, rstd, down, mask);
            StoreAlign(s + 16 * WIDTH, rstd, mask);
            Muls(mean, mean, momentum, mask);
            Muls(variance, variance, momentum, mask);
            if (running) {
                LoadAlign(old, s + 11 * WIDTH);
                Axpy(mean, old, 1.0f - momentum, mask);
                LoadAlign(old, s + 12 * WIDTH);
                Axpy(variance, old, 1.0f - momentum, mask);
            }
            StoreAlign(s + 13 * WIDTH, mean, mask);
            StoreAlign(s + 14 * WIDTH, variance, mask);
        }
    }

    __aicore__ inline void CopyStatistics(int64_t c)
    {
        auto stats = stats_.Get<float>();
        auto event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        DataCopyExtParams copy{1, channels_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        for (uint32_t i = 0; i < 4; ++i) {
            DataCopyPad(outputs_[i][c], stats[(13 + i) * WIDTH], copy);
        }
        event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event);
        WaitFlag<HardEvent::MTE3_V>(event);
    }

    const BatchNorm3DCompensatedTilingData* data_;
    TPipe* pipe_;
    bool channelLast_ = false;
    uint32_t channels_ = 0;
    uint32_t stride_ = 0;
    GlobalTensor<T> x_, y_;
    GlobalTensor<float> scale_, offset_, mean_, variance_, workspace_, outputs_[4];
    TQue<QuePosition::VECIN, 1> input_;
    TQue<QuePosition::VECOUT, 1> output_;
    TBuf<TPosition::VECCALC> stats_;
};
} // namespace BatchNorm3DOps
#endif // NORM_BATCH_NORM3D_COMPENSATED_H
