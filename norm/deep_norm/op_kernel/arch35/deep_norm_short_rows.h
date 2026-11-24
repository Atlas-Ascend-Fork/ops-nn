/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEEP_NORM_ARCH35_SHORT_ROWS_H
#define DEEP_NORM_ARCH35_SHORT_ROWS_H

#include "deep_norm.h"

namespace NsDeepNorm {

// affineBuffer 为 gamma 与 beta 拼接，各 DEEP_NORM_VL_FP32 个 fp32。
constexpr uint32_t DEEP_NORM_AFFINE_TENSOR_COUNT = 2;

template <typename T>
__aicore__ inline void DeepNormShortPackAffine(__ubuf__ T* gamma, __ubuf__ T* beta, __ubuf__ float* affine,
                                               uint32_t columns)
{
    __VEC_SCOPE__
    {
        MaskReg mask = UpdateMask<float>(columns);
        RegTensor<float> value;
        NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gamma, value, mask, 0);
        Reg::DataCopy(affine, value, mask);
        NormCommon::NormCommonRegbase::LoadRegForDtype<T>(beta, value, mask, 0);
        Reg::DataCopy(affine + DEEP_NORM_VL_FP32, value, mask);
    }
}

template <typename T>
__aicore__ inline void DeepNormShortBuildH(__ubuf__ T* x, __ubuf__ T* gx, __ubuf__ float* h, uint32_t elements,
                                           float alpha)
{
    __VEC_SCOPE__
    {
        const uint16_t loops = (elements + DEEP_NORM_VL_FP32 - 1) / DEEP_NORM_VL_FP32;
        for (uint16_t loop = 0; loop < loops; ++loop) {
            const uint32_t offset = loop * DEEP_NORM_VL_FP32;
            MaskReg mask = UpdateMask<float>(elements);
            RegTensor<float> value, residual;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(x, value, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gx, residual, mask, offset);
            Reg::Muls(value, value, alpha, mask);
            Reg::Add(value, value, residual, mask);
            Reg::DataCopy(h + offset, value, mask);
        }
    }
}

template <typename T>
__aicore__ inline void DeepNormUnitRows(__ubuf__ T* x, __ubuf__ T* gx, __ubuf__ float* affine, __ubuf__ float* mean,
                                        __ubuf__ float* rstd, __ubuf__ T* y, uint32_t rows, float alpha, float epsilon)
{
    __VEC_SCOPE__
    {
        RegTensor<float> gamma, beta;
        Reg::DataCopy<float, LoadDist::DIST_BRC_B32>(gamma, affine);
        Reg::DataCopy<float, LoadDist::DIST_BRC_B32>(beta, affine + DEEP_NORM_VL_FP32);
        const uint16_t loops = (rows + DEEP_NORM_VL_FP32 - 1) / DEEP_NORM_VL_FP32;
        for (uint16_t loop = 0; loop < loops; ++loop) {
            const uint32_t offset = loop * DEEP_NORM_VL_FP32;
            MaskReg mask = UpdateMask<float>(rows);
            RegTensor<float> h, residual, centered, variance, inverse;
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(x, h, mask, offset);
            NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gx, residual, mask, offset);
            Reg::Muls(h, h, alpha, mask);
            Reg::Add(h, h, residual, mask);
            Reg::DataCopy(mean + offset, h, mask);
            // The subtraction preserves the original behavior for infinite or NaN H.
            Reg::Sub(centered, h, h, mask);
            Reg::Mul(variance, centered, centered, mask);
            NormCommon::ComputeRstdNewtonRaphsonReg(variance, inverse, mask, epsilon);
            Reg::DataCopy(rstd + offset, inverse, mask);
            Reg::Mul(centered, centered, inverse, mask);
            Reg::Mul(centered, centered, gamma, mask);
            Reg::Add(centered, centered, beta, mask);
            NormCommon::NormCommonRegbase::StoreRegForDtype<T>(y, centered, mask, offset);
        }
    }
}

__aicore__ inline void DeepNormAddMeanTerm(RegTensor<float>& sum, RegTensor<float>& correction, RegTensor<float>& value,
                                           MaskReg& mask)
{
    constexpr float maxFinite = 3.402823466e+38f;
    RegTensor<float> next, part, error, remainder, zero;
    MaskReg finite;
    Reg::Add(next, sum, value, mask);
    Reg::Sub(part, next, sum, mask);
    Reg::Sub(error, next, part, mask);
    Reg::Sub(error, sum, error, mask);
    Reg::Sub(remainder, value, part, mask);
    Reg::Add(error, error, remainder, mask);
    // Error-free addition uses subtraction internally. Preserve the rounded
    // sum's nonfinite result without accumulating an artificial Inf - Inf.
    Reg::Abs(part, next, mask);
    Reg::Compares<float, CMPMODE::LE>(finite, part, maxFinite, mask);
    Reg::Duplicate(zero, 0.0f, mask);
    Reg::Select(error, error, zero, finite);
    Reg::Add(correction, correction, error, mask);
    Reg::Copy(sum, next, mask);
}

template <bool Square>
__aicore__ inline void DeepNormShortSum(RegTensor<float>& sum, RegTensor<float>& mean, __ubuf__ float* h,
                                        RegTensor<uint32_t>& rows, uint32_t columns, MaskReg& mask)
{
    RegTensor<float> even, odd;
    Reg::Duplicate(even, 0.0f, mask);
    Reg::Duplicate(odd, 0.0f, mask);
    for (uint16_t column = 0; column < columns; ++column) {
        RegTensor<uint32_t> index;
        RegTensor<float> value;
        Reg::Adds(index, rows, static_cast<uint32_t>(column), mask);
        Reg::Gather(value, h, index, mask);
        if constexpr (Square) {
            Reg::Sub(value, value, mean, mask);
            Reg::Mul(value, value, value, mask);
            if ((column & 1U) == 0) {
                Reg::Add(even, even, value, mask);
            } else {
                Reg::Add(odd, odd, value, mask);
            }
        } else {
            DeepNormAddMeanTerm(even, odd, value, mask);
        }
    }
    Reg::Add(sum, even, odd, mask);
}

__aicore__ inline void DeepNormShortNormalize(__ubuf__ float* h, __ubuf__ float* affine, __ubuf__ float* mean,
                                              __ubuf__ float* rstd, __ubuf__ float* y, uint32_t rows, uint32_t columns,
                                              float average, float epsilon)
{
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> rowIndices;
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        Reg::Arange(reinterpret_cast<RegTensor<int32_t>&>(rowIndices), 0);
        Reg::Muls(rowIndices, rowIndices, columns, full);
        const uint16_t loops = (rows + DEEP_NORM_VL_FP32 - 1) / DEEP_NORM_VL_FP32;
        // Each vector lane handles a separate row, with one batched store per statistics output.
        for (uint16_t loop = 0; loop < loops; ++loop) {
            const uint32_t rowOffset = loop * DEEP_NORM_VL_FP32;
            const uint32_t elementOffset = rowOffset * columns;
            MaskReg mask = UpdateMask<float>(rows);
            RegTensor<float> rowMean, variance, inverse;
            DeepNormShortSum<false>(rowMean, rowMean, h + elementOffset, rowIndices, columns, mask);
            Reg::Muls(rowMean, rowMean, average, mask);
            Reg::DataCopy(mean + rowOffset, rowMean, mask);
            DeepNormShortSum<true>(variance, rowMean, h + elementOffset, rowIndices, columns, mask);
            Reg::Muls(variance, variance, average, mask);
            NormCommon::ComputeRstdNewtonRaphsonReg(variance, inverse, mask, epsilon);
            Reg::DataCopy(rstd + rowOffset, inverse, mask);
            for (uint16_t column = 0; column < columns; ++column) {
                RegTensor<uint32_t> index;
                RegTensor<float> value, gamma, beta;
                Reg::Adds(index, rowIndices, static_cast<uint32_t>(column), mask);
                Reg::Gather(value, h + elementOffset, index, mask);
                Reg::DataCopy<float, LoadDist::DIST_BRC_B32>(gamma, affine + column);
                Reg::DataCopy<float, LoadDist::DIST_BRC_B32>(beta, affine + DEEP_NORM_VL_FP32 + column);
                Reg::Sub(value, value, rowMean, mask);
                Reg::Mul(value, value, inverse, mask);
                Reg::Mul(value, value, gamma, mask);
                Reg::Add(value, value, beta, mask);
                Reg::Scatter(y + elementOffset, value, index, mask);
            }
        }
    }
}

template <typename T>
__aicore__ inline void DeepNormShortCast(__ubuf__ float* src, __ubuf__ T* y, uint32_t elements)
{
    __VEC_SCOPE__
    {
        const uint16_t loops = (elements + DEEP_NORM_VL_FP32 - 1) / DEEP_NORM_VL_FP32;
        for (uint16_t loop = 0; loop < loops; ++loop) {
            const uint32_t offset = loop * DEEP_NORM_VL_FP32;
            MaskReg mask = UpdateMask<float>(elements);
            RegTensor<float> value;
            Reg::LoadAlign(value, src + offset);
            NormCommon::NormCommonRegbase::StoreRegForDtype<T>(y, value, mask, offset);
        }
    }
}

template <typename T>
class DeepNormShortRows {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gx, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd,
                                GM_ADDR y, const DeepNormTilingData* data)
    {
        data_ = data;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        gx_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gx));
        gamma_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma));
        beta_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(beta));
        mean_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mean));
        rstd_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rstd));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        const uint32_t elements = data_->tileLength * data_->numCol;
        pipe_.InitBuffer(xQueue_, 1, elements * sizeof(T));
        pipe_.InitBuffer(gxQueue_, 1, elements * sizeof(T));
        pipe_.InitBuffer(yQueue_, 1, elements * sizeof(T));
        pipe_.InitBuffer(meanQueue_, 1, data_->tileLength * sizeof(float));
        pipe_.InitBuffer(rstdQueue_, 1, data_->tileLength * sizeof(float));
        pipe_.InitBuffer(gammaQueue_, 1, DEEP_NORM_VL_FP32 * sizeof(T));
        pipe_.InitBuffer(betaQueue_, 1, DEEP_NORM_VL_FP32 * sizeof(T));
        pipe_.InitBuffer(affineBuffer_, DEEP_NORM_AFFINE_TENSOR_COUNT * DEEP_NORM_VL_FP32 * sizeof(float));
        if (data_->numCol > 1) {
            pipe_.InitBuffer(hBuffer_, elements * sizeof(float));
            if constexpr (!IsSameType<T, float>::value) {
                pipe_.InitBuffer(affineOutputBuffer_, elements * sizeof(float));
            }
        }
    }

    __aicore__ inline void Process()
    {
        const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * data_->rowPerCore;
        if (begin >= data_->numRow) {
            return;
        }
        const uint64_t remaining = data_->numRow - begin;
        const uint64_t end = begin + (remaining < data_->rowPerCore ? remaining : data_->rowPerCore);
        LoadAffine();
        for (uint64_t row = begin; row < end;) {
            const uint64_t rowsLeft = end - row;
            const uint32_t rows = rowsLeft < data_->tileLength ? rowsLeft : data_->tileLength;
            ProcessTile(row, rows);
            row += rows;
        }
    }

private:
    __aicore__ inline void LoadAffine()
    {
        LocalTensor<T> gamma = gammaQueue_.template AllocTensor<T>();
        LocalTensor<T> beta = betaQueue_.template AllocTensor<T>();
        const DataCopyExtParams copy{1, data_->numCol * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> padding{false, 0, 0, 0};
        DataCopyPad(gamma, gamma_, copy, padding);
        DataCopyPad(beta, beta_, copy, padding);
        gammaQueue_.EnQue(gamma);
        betaQueue_.EnQue(beta);
        gamma = gammaQueue_.template DeQue<T>();
        beta = betaQueue_.template DeQue<T>();
        auto affine = affineBuffer_.Get<float>();
        DeepNormShortPackAffine<T>(reinterpret_cast<__ubuf__ T*>(gamma.GetPhyAddr()),
                                   reinterpret_cast<__ubuf__ T*>(beta.GetPhyAddr()),
                                   reinterpret_cast<__ubuf__ float*>(affine.GetPhyAddr()), data_->numCol);
        PipeBarrier<PIPE_V>();
        gammaQueue_.FreeTensor(gamma);
        betaQueue_.FreeTensor(beta);
    }

    __aicore__ inline void ProcessTile(uint64_t row, uint32_t rows)
    {
        const uint32_t columns = data_->numCol;
        const uint32_t elements = rows * columns;
        const uint64_t offset = row * columns;
        const DataCopyExtParams dataCopy{1, elements * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        const DataCopyExtParams statCopy{1, rows * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<T> padding{false, 0, 0, 0};
        LocalTensor<T> x = xQueue_.template AllocTensor<T>();
        LocalTensor<T> gx = gxQueue_.template AllocTensor<T>();
        DataCopyPad(x, x_[offset], dataCopy, padding);
        DataCopyPad(gx, gx_[offset], dataCopy, padding);
        xQueue_.EnQue(x);
        gxQueue_.EnQue(gx);
        x = xQueue_.template DeQue<T>();
        gx = gxQueue_.template DeQue<T>();
        LocalTensor<T> y = yQueue_.template AllocTensor<T>();
        LocalTensor<float> mean = meanQueue_.template AllocTensor<float>();
        LocalTensor<float> rstd = rstdQueue_.template AllocTensor<float>();
        auto* xAddr = reinterpret_cast<__ubuf__ T*>(x.GetPhyAddr());
        auto* gxAddr = reinterpret_cast<__ubuf__ T*>(gx.GetPhyAddr());
        auto* yAddr = reinterpret_cast<__ubuf__ T*>(y.GetPhyAddr());
        auto* meanAddr = reinterpret_cast<__ubuf__ float*>(mean.GetPhyAddr());
        auto* rstdAddr = reinterpret_cast<__ubuf__ float*>(rstd.GetPhyAddr());
        auto* affineAddr = reinterpret_cast<__ubuf__ float*>(affineBuffer_.Get<float>().GetPhyAddr());
        if (columns == 1) {
            DeepNormUnitRows<T>(xAddr, gxAddr, affineAddr, meanAddr, rstdAddr, yAddr, rows, data_->alpha, data_->eps);
        } else {
            auto* h = reinterpret_cast<__ubuf__ float*>(hBuffer_.Get<float>().GetPhyAddr());
            DeepNormShortBuildH<T>(xAddr, gxAddr, h, elements, data_->alpha);
            PipeBarrier<PIPE_V>();
            __ubuf__ float* output;
            if constexpr (IsSameType<T, float>::value) {
                output = yAddr;
            } else {
                output = reinterpret_cast<__ubuf__ float*>(affineOutputBuffer_.Get<float>().GetPhyAddr());
            }
            DeepNormShortNormalize(h, affineAddr, meanAddr, rstdAddr, output, rows, columns, data_->avgFactor,
                                   data_->eps);
            if constexpr (!IsSameType<T, float>::value) {
                PipeBarrier<PIPE_V>();
                DeepNormShortCast<T>(output, yAddr, elements);
            }
        }
        xQueue_.FreeTensor(x);
        gxQueue_.FreeTensor(gx);
        yQueue_.EnQue(y);
        meanQueue_.EnQue(mean);
        rstdQueue_.EnQue(rstd);
        y = yQueue_.template DeQue<T>();
        mean = meanQueue_.template DeQue<float>();
        rstd = rstdQueue_.template DeQue<float>();
        DataCopyPad(y_[offset], y, dataCopy);
        DataCopyPad(mean_[row], mean, statCopy);
        DataCopyPad(rstd_[row], rstd, statCopy);
        yQueue_.FreeTensor(y);
        meanQueue_.FreeTensor(mean);
        rstdQueue_.FreeTensor(rstd);
    }

    const DeepNormTilingData* data_ = nullptr;
    TPipe pipe_;
    GlobalTensor<T> x_, gx_, gamma_, beta_, y_;
    GlobalTensor<float> mean_, rstd_;
    TQue<QuePosition::VECIN, 1> xQueue_, gxQueue_, gammaQueue_, betaQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_, meanQueue_, rstdQueue_;
    TBuf<QuePosition::VECCALC> affineBuffer_, hBuffer_, affineOutputBuffer_;
};

} // namespace NsDeepNorm

#endif // DEEP_NORM_ARCH35_SHORT_ROWS_H
