/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_FAST_H_
#define OPS_NORM_CENTRALIZATION_FAST_H_
#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "centralization_compensated_sum.h"
#include "centralization_tiling_data.h"

namespace Centralization {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;
using AscendC::Reg::Reduce;
using NormCommon::V_LENGTH;
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

template <typename T>
class Fast {
public:
    __aicore__ inline Fast(TPipe* pipe, const CentralizationTilingData* tiling) : pipe_(pipe), tiling_(tiling) {}
    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        rows_ = tiling_->groupCount;
        cols_ = tiling_->reduceCount;
        const int64_t alignedReduce = (tiling_->ubFactor > 0) ? tiling_->ubFactor : tiling_->alignedReduce;
        copyAlignedCols_ = AlignToBlock(cols_);
        alignedCols_ = UseNarrowVector() ? copyAlignedCols_ : AlignToVLength(alignedReduce);
        batchRows_ = (tiling_->rowsPerLoop > 1) ? tiling_->rowsPerLoop : tiling_->rowsPerBatch;
        bufNum_ = (tiling_->bufNum > 0) ? static_cast<uint32_t>(tiling_->bufNum) : 2U;
        const uint32_t core = GetBlockIdx();
        start_ = core * tiling_->blockFactor;
        end_ = (rows_ < start_ + tiling_->blockFactor) ? rows_ : start_ + tiling_->blockFactor;
        const uint32_t rowBytes = static_cast<uint32_t>(cols_ * sizeof(T));
        const uint32_t batchBytes = static_cast<uint32_t>(batchRows_ * alignedCols_ * sizeof(T));
        pipe_->InitBuffer(inQueue_, bufNum_, batchBytes);
        pipe_->InitBuffer(outQueue_, bufNum_, batchBytes);
        eventMte2ToS_ = pipe_->FetchEventID(HardEvent::MTE2_S);
        eventSToMte3_ = pipe_->FetchEventID(HardEvent::S_MTE3);
        rowBytes_ = rowBytes;
    }

    __aicore__ inline void Process()
    {
        if (start_ >= end_) {
            return;
        }
        const int64_t batchCount = (end_ - start_ + batchRows_ - 1) / batchRows_;
        for (int64_t stage = 0; stage < batchCount + 2; ++stage) {
            if (stage < batchCount) {
                const int64_t batchStart = GetBatchStart(stage);
                CopyIn(batchStart, GetBatchRows(batchStart));
            }
            if (stage >= 1 && (stage - 1) < batchCount) {
                const int64_t batchStart = GetBatchStart(stage - 1);
                Compute(GetBatchRows(batchStart));
            }
            if (stage >= 2 && (stage - 2) < batchCount) {
                const int64_t batchStart = GetBatchStart(stage - 2);
                CopyOut(batchStart, GetBatchRows(batchStart));
            }
        }
    }

private:
    // Small reductions use scalar compensation except for the FP32 widths handled by UseNarrowVector().
    static constexpr int64_t kScalarReduceLimit = 128;

    __aicore__ inline int64_t GetBatchStart(int64_t batch) const { return start_ + batch * batchRows_; }

    __aicore__ inline int64_t GetBatchRows(int64_t batchStart) const
    {
        const int64_t remaining = end_ - batchStart;
        return batchRows_ < remaining ? batchRows_ : remaining;
    }

    __aicore__ inline void CopyIn(int64_t batchStart, int64_t rowCount)
    {
        LocalTensor<T> input = inQueue_.AllocTensor<T>();
        const uint32_t dstStride = static_cast<uint32_t>((alignedCols_ - copyAlignedCols_) * sizeof(T) / 32);
        DataCopyExtParams copy{static_cast<uint16_t>(rowCount), rowBytes_, 0, dstStride, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(copyAlignedCols_ - cols_), static_cast<T>(0)};
        DataCopyPad(input, xGm_[batchStart * cols_], copy, pad);
        inQueue_.EnQue(input);
    }

    __aicore__ inline void Compute(int64_t rowCount)
    {
        LocalTensor<T> input = inQueue_.DeQue<T>();
        LocalTensor<T> output = outQueue_.AllocTensor<T>();
        ComputeBatch(input, output, rowCount);
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
    }

    __aicore__ inline void CopyOut(int64_t batchStart, int64_t rowCount)
    {
        LocalTensor<T> output = outQueue_.DeQue<T>();
        const uint32_t srcStride = static_cast<uint32_t>((alignedCols_ - copyAlignedCols_) * sizeof(T) / 32);
        DataCopyExtParams copy{static_cast<uint16_t>(rowCount), rowBytes_, srcStride, 0, 0};
        DataCopyPad(yGm_[batchStart * cols_], output, copy);
        outQueue_.FreeTensor(output);
    }

    __aicore__ inline void AddNeumaier(float& sum, float& corr, float value) const
    {
        const float next = sum + value;
        const float absSum = sum < 0.0f ? -sum : sum;
        const float absValue = value < 0.0f ? -value : value;
        constexpr float kMaxFinite = 3.4028235e38f;
        if (!(absSum <= kMaxFinite && absValue <= kMaxFinite)) {
            sum = next;
            return;
        }
        if (absSum >= absValue) {
            corr += (sum - next) + value;
        } else {
            corr += (value - next) + sum;
        }
        sum = next;
    }

    __aicore__ inline void FinalizeMeanHiLo(float sum, float corr, float n, float& hi, float& lo) const
    {
        hi = sum / n;
        constexpr float kMaxFinite = 3.4028235e38f;
        if (!(hi <= kMaxFinite && hi >= -kMaxFinite)) {
            lo = 0.0f;
            return;
        }
        const float r = MeanProductResidual(hi, n, sum);
        lo = (r + corr) / n;
    }

    // Scale both the inputs and the divisor so that the mean is computed in the original units.
    __aicore__ inline void RecomputeScaledMean(LocalTensor<T>& input, int64_t base, int64_t count, float& meanHi,
                                               float& meanLo) const
    {
        uint32_t power = 1;
        while (power < count) {
            power <<= 1;
        }
        const float restore = static_cast<float>(2 * power);
        const float scale = 1.0f / restore;
        float sum = 0.0f;
        float corr = 0.0f;
        for (int64_t col = 0; col < count; ++col) {
            AddNeumaier(sum, corr, static_cast<float>(input.GetValue(base + col)) * scale);
        }
        FinalizeMeanHiLo(sum, corr, static_cast<float>(count) * scale, meanHi, meanLo);
    }

    __aicore__ inline void ComputeBatchScalar(LocalTensor<T>& input, LocalTensor<T>& output, int64_t rowCount)
    {
        SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
        WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
        const int64_t rowStride = alignedCols_;
        const int64_t count = cols_;
        for (int64_t row = 0; row < rowCount; ++row) {
            float sum = 0.0f;
            float corr = 0.0f;
            const int64_t base = row * rowStride;
            for (int64_t col = 0; col < count; ++col) {
                AddNeumaier(sum, corr, static_cast<float>(input.GetValue(base + col)));
            }
            float meanHi = 0.0f;
            float meanLo = 0.0f;
            constexpr float kMaxFinite = 0x1.fffffep127f;
            if (IsSameType<T, float>::value && !(sum <= kMaxFinite && sum >= -kMaxFinite)) {
                RecomputeScaledMean(input, base, count, meanHi, meanLo);
            } else {
                FinalizeMeanHiLo(sum, corr, static_cast<float>(count), meanHi, meanLo);
            }
            for (int64_t col = 0; col < count; ++col) {
                const float value = static_cast<float>(input.GetValue(base + col));
                output.SetValue(base + col, static_cast<T>((value - meanHi) - meanLo));
            }
        }
        SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
        WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
    }

    __aicore__ inline bool UseNarrowVector() const { return IsSameType<T, float>::value && (cols_ == 4 || cols_ == 6); }

    // One lane per row. Gather a column at a time so each TwoSum correction
    // is carried through the entire reduction, then scatter the centered values.
    __aicore__ inline void ComputeBatchNarrowFloat(LocalTensor<T>& input, LocalTensor<T>& output, uint32_t rowCount)
    {
        __local_mem__ float* inputAddr = (__local_mem__ float*)input.GetPhyAddr();
        __local_mem__ float* outputAddr = (__local_mem__ float*)output.GetPhyAddr();
        const uint32_t stride = static_cast<uint32_t>(alignedCols_);
        const uint16_t cols = static_cast<uint16_t>(cols_);
        const uint16_t blocks = static_cast<uint16_t>((rowCount + V_LENGTH - 1) / V_LENGTH);
        // Keep headroom for the sum and the residual calculation even at FLT_MAX.
        const float restoreFactor = cols == 4 ? 8.0f : 16.0f;
        const float scaleFactor = 1.0f / restoreFactor;
        __VEC_SCOPE__
        {
            RegTensor<uint32_t> rowOffsets;
            RegTensor<uint32_t> offsets;
            RegTensor<float> count;
            RegTensor<float> zero;
            MaskReg full = CreateMask<float, MaskPattern::ALL>();
            AscendC::Reg::Arange(reinterpret_cast<RegTensor<int32_t>&>(rowOffsets), 0);
            Muls(rowOffsets, rowOffsets, stride, full);
            Duplicate(count, static_cast<float>(cols), full);
            Duplicate(zero, 0.0f, full);
            uint32_t remaining = rowCount;
            for (uint16_t block = 0; block < blocks; ++block) {
                MaskReg active = UpdateMask<float>(remaining);
                const uint32_t base = block * V_LENGTH * stride;
                RegTensor<float> sum;
                RegTensor<float> correction;
                RegTensor<float> value;
                RegTensor<float> meanHi;
                RegTensor<float> meanLo;
                RegTensor<float> scale;
                RegTensor<float> divisor;
                RegTensor<float> one;
                // Probe in the same column order as the compensated sum, one lane per row.
                Duplicate(sum, 0.0f, active);
                for (uint16_t col = 0; col < cols; ++col) {
                    Adds(offsets, rowOffsets, static_cast<uint32_t>(col), active);
                    AscendC::Reg::Gather(value, inputAddr + base, offsets, active);
                    Add(sum, sum, value, active);
                }
                Abs(value, sum, active);
                MaskReg unscaled;
                Compares<float, CMPMODE::LE>(unscaled, value, 0x1.fffffep127f, active);
                Duplicate(one, 1.0f, active);
                Duplicate(scale, scaleFactor, active);
                Select(scale, one, scale, unscaled);
                Mul(divisor, count, scale, active);
                Duplicate(sum, 0.0f, active);
                Duplicate(correction, 0.0f, active);
                for (uint16_t col = 0; col < cols; ++col) {
                    Adds(offsets, rowOffsets, static_cast<uint32_t>(col), active);
                    AscendC::Reg::Gather(value, inputAddr + base, offsets, active);
                    Mul(value, value, scale, active);
                    AddWithResidual(sum, correction, value, active);
                }
                Div(meanHi, sum, divisor, active);
                Muls(meanLo, meanHi, -1.0f, active);
                MulDstAdd<float, AscendC::Reg::MaskMergeMode::ZEROING>(meanLo, divisor, sum, active);
                Add(meanLo, meanLo, correction, active);
                Div(meanLo, meanLo, divisor, active);
                Abs(value, meanHi, active);
                MaskReg finite;
                Compares<float, CMPMODE::LE>(finite, value, 3.4028235e38f, active);
                Select(meanLo, meanLo, zero, finite);
                for (uint16_t col = 0; col < cols; ++col) {
                    Adds(offsets, rowOffsets, static_cast<uint32_t>(col), active);
                    AscendC::Reg::Gather(value, inputAddr + base, offsets, active);
                    Sub(value, value, meanHi, active);
                    Sub(value, value, meanLo, active);
                    AscendC::Reg::Scatter(outputAddr + base, value, offsets, active);
                }
            }
        }
    }

    __aicore__ inline void ComputeBatch(LocalTensor<T>& input, LocalTensor<T>& output, int64_t rowCount)
    {
        if (UseNarrowVector()) {
            ComputeBatchNarrowFloat(input, output, static_cast<uint32_t>(rowCount));
            return;
        }
        if (cols_ > 1 && cols_ <= kScalarReduceLimit) {
            ComputeBatchScalar(input, output, rowCount);
            return;
        }
        __local_mem__ T* inputAddr = (__local_mem__ T*)input.GetPhyAddr();
        __local_mem__ T* outputAddr = (__local_mem__ T*)output.GetPhyAddr();
        const uint32_t alignedCount = static_cast<uint32_t>(alignedCols_);
        const uint32_t validCount = static_cast<uint32_t>(cols_);
        const uint16_t colLoops = static_cast<uint16_t>((validCount + V_LENGTH - 1) / V_LENGTH);
        const uint32_t validRows = static_cast<uint32_t>(rowCount);

        __VEC_SCOPE__
        {
            RegTensor<float> sum;
            RegTensor<float> corr;
            RegTensor<float> meanHigh;
            RegTensor<float> meanLow;
            RegTensor<float> meanHiDup;
            RegTensor<float> meanLoDup;
            RegTensor<float> nReg;
            RegTensor<float> zero;
            RegTensor<float> negMean;
            RegTensor<float> tmp;
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            Duplicate(nReg, static_cast<float>(cols_), pregOne);
            Duplicate(zero, 0.0f, pregOne);

            for (uint16_t row = 0; row < static_cast<uint16_t>(validRows); ++row) {
                const uint32_t base = row * alignedCount;
                ReduceRow(inputAddr, sum, corr, base, colLoops, validCount);
                Div(meanHigh, sum, nReg, pregOne);
                Muls(negMean, meanHigh, -1.0f, pregOne);
                MulDstAdd<float, AscendC::Reg::MaskMergeMode::ZEROING>(negMean, nReg, sum, pregOne);
                Add(tmp, negMean, corr, pregOne);
                Div(meanLow, tmp, nReg, pregOne);
                Abs(tmp, meanHigh, pregOne);
                MaskReg finite;
                Compares<float, CMPMODE::LE>(finite, tmp, 3.4028235e38f, pregOne);
                Select(meanLow, meanLow, zero, finite);
                Duplicate(meanHiDup, meanHigh, pregFull);
                Duplicate(meanLoDup, meanLow, pregFull);
                CentralizeRow(inputAddr, outputAddr, meanHiDup, meanLoDup, base, colLoops, validCount);
            }
        }
    }

    __aicore__ inline void ReduceRow(__local_mem__ T* inputAddr, RegTensor<float>& sum, RegTensor<float>& corr,
                                     uint32_t base, uint16_t colLoops, uint32_t validCount)
    {
        RegTensor<float> xReg;
        RegTensor<float> partialReg;
        RegTensor<float> nextReg;
        RegTensor<float> c1Reg;
        RegTensor<float> c2Reg;
        RegTensor<float> addReg;
        RegTensor<float> absSum;
        RegTensor<float> absPart;
        RegTensor<float> zeroReg;
        MaskReg dom;
        MaskReg fin;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        Duplicate(zeroReg, static_cast<float>(0.0f), pregFull);
        Duplicate(sum, static_cast<float>(0.0f), pregOne);
        Duplicate(corr, static_cast<float>(0.0f), pregOne);
        for (uint16_t loop = 0; loop < colLoops; ++loop) {
            const uint32_t offset = base + static_cast<uint32_t>(loop) * V_LENGTH;
            const uint32_t remaining = validCount - static_cast<uint32_t>(loop) * V_LENGTH;
            uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
            MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
            LoadRegForDtype(inputAddr, xReg, pregLoop, offset);
            AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(partialReg, xReg, pregLoop);
            Add(nextReg, sum, partialReg, pregOne);
            Abs(absSum, sum, pregOne);
            Abs(absPart, partialReg, pregOne);
            Compare<float, CMPMODE::GE>(dom, absSum, absPart, pregOne);
            Sub(c1Reg, sum, nextReg, pregOne);
            Add(c1Reg, c1Reg, partialReg, pregOne);
            Sub(c2Reg, partialReg, nextReg, pregOne);
            Add(c2Reg, c2Reg, sum, pregOne);
            Select(addReg, c1Reg, c2Reg, dom);
            Abs(absSum, nextReg, pregOne);
            Compares<float, CMPMODE::LE>(fin, absSum, 3.4028235e38f, pregOne);
            Select(addReg, addReg, zeroReg, fin);
            Add(corr, corr, addReg, pregOne);
            sum = nextReg;
        }
    }

    __aicore__ inline void CentralizeRow(__local_mem__ T* inputAddr, __local_mem__ T* outputAddr,
                                         RegTensor<float>& meanHiDup, RegTensor<float>& meanLoDup, uint32_t base,
                                         uint16_t colLoops, uint32_t validCount)
    {
        RegTensor<float> xReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t loop = 0; loop < colLoops; ++loop) {
            const uint32_t offset = base + static_cast<uint32_t>(loop) * V_LENGTH;
            const uint32_t remaining = validCount - static_cast<uint32_t>(loop) * V_LENGTH;
            uint32_t activeCount = remaining < V_LENGTH ? remaining : V_LENGTH;
            MaskReg pregLoop = activeCount == V_LENGTH ? pregFull : UpdateMask<float>(activeCount);
            LoadRegForDtype(inputAddr, xReg, pregLoop, offset);
            Sub(xReg, xReg, meanHiDup, pregLoop);
            Sub(xReg, xReg, meanLoDup, pregLoop);
            StoreRegForDtype(outputAddr, xReg, pregLoop, offset);
        }
    }

    __aicore__ inline int64_t AlignToVLength(int64_t value) const
    {
        return (value + V_LENGTH - 1) / V_LENGTH * V_LENGTH;
    }

    __aicore__ inline int64_t AlignToBlock(int64_t value) const
    {
        constexpr int64_t blockElements = 32 / sizeof(T);
        return (value + blockElements - 1) / blockElements * blockElements;
    }

    TPipe* pipe_;
    const CentralizationTilingData* tiling_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    TQue<QuePosition::VECIN, 2> inQueue_;
    TQue<QuePosition::VECOUT, 2> outQueue_;
    int64_t rows_ = 0;
    int64_t cols_ = 0;
    int64_t alignedCols_ = 0;
    int64_t copyAlignedCols_ = 0;
    int64_t batchRows_ = 1;
    uint32_t bufNum_ = 2;
    uint32_t rowBytes_ = 0;
    TEventID eventMte2ToS_{};
    TEventID eventSToMte3_{};
    int64_t start_ = 0;
    int64_t end_ = 0;
};
} // namespace Centralization
#endif
