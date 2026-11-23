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
 * \file l2_normalize_grad_regbase_dx_elementwise.h
 * \brief Length-one reduction, processed as contiguous elementwise tiles (7050).
 */
#ifndef L2_NORMALIZE_GRAD_REGBASE_DX_ELEMENTWISE_H
#define L2_NORMALIZE_GRAD_REGBASE_DX_ELEMENTWISE_H
#include "l2_normalize_grad_regbase_base.h"

namespace L2NormalizeGrad {
constexpr float MIN_NORMAL_FP32 = 1.175494351e-38F;

template <typename T_X>
class RegbaseDxElementwise : public RegbaseDxBase<T_X> {
    using Base = RegbaseDxBase<T_X>;

public:
    __aicore__ inline RegbaseDxElementwise(TPipe* pipe, const L2NormalizeGradTilingData* data) : Base(pipe, data) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR dy, GM_ADDR dx)
    {
        if (!this->InitCommon(x, y, dy, dx, 1)) {
            return;
        }
        this->InitQueues(qx_, qy_, qdy_, qdx_);
    }
    __aicore__ inline void Process()
    {
        if (this->coreIdx_ >= this->usedCoreNum_) {
            return;
        }
        const int64_t count = Min(this->blockFactor_,
                                  this->tiling_->outer - static_cast<int64_t>(this->coreIdx_) * this->blockFactor_);
        for (int64_t start = 0; start < count; start += this->tiling_->ubFactorElems) {
            const int64_t n = Min(this->tiling_->ubFactorElems, count - start);
            this->CopyIn2D(qx_, this->xGm_, start, 1, n, 0);
            this->CopyIn2D(qy_, this->yGm_, start, 1, n, 0);
            this->CopyIn2D(qdy_, this->dyGm_, start, 1, n, 0);
            LocalTensor<T_X> x = qx_.template DeQue<T_X>();
            LocalTensor<T_X> y = qy_.template DeQue<T_X>();
            LocalTensor<T_X> dy = qdy_.template DeQue<T_X>();
            LocalTensor<T_X> dx = qdx_.template AllocTensor<T_X>();
            __local_mem__ T_X* xp = (__ubuf__ T_X*)x.GetPhyAddr();
            __local_mem__ T_X* yp = (__ubuf__ T_X*)y.GetPhyAddr();
            __local_mem__ T_X* dyp = (__ubuf__ T_X*)dy.GetPhyAddr();
            __local_mem__ T_X* dxp = (__ubuf__ T_X*)dx.GetPhyAddr();
            const uint16_t repeats = static_cast<uint16_t>(DivCeil(n, static_cast<int64_t>(V_LENGTH)));
            const float eps = this->eps_;
            __VEC_SCOPE__
            {
                RegTensor<float> xr, yr, dyr, sq, s, norm, ys, sub, result;
                uint32_t remaining = static_cast<uint32_t>(n);
                for (uint16_t r = 0; r < repeats; ++r) {
                    MaskReg mask = UpdateMask<float>(remaining);
                    const uint32_t off = r * V_LENGTH;
                    LoadAndCast(xr, xp, mask, off);
                    LoadAndCast(yr, yp, mask, off);
                    LoadAndCast(dyr, dyp, mask, off);
                    Mul(sq, xr, xr, mask);
                    Mul(s, yr, dyr, mask);
                    if constexpr (IsSameType<T_X, float>::value) {
                        // For a normal finite square, the correctly rounded sqrt is |x|.
                        // Preserve the square/sqrt behavior on underflow, overflow and NaN.
                        RegTensor<float> absX, sqrtSq;
                        MaskReg normalSq, finiteSq, needsSqrt;
                        Compares<float, CMPMODE::GE>(normalSq, sq, MIN_NORMAL_FP32, mask);
                        Compares<float, CMPMODE::LE>(finiteSq, sq, MAX_FINITE_FP32, mask);
                        MaskAnd(normalSq, normalSq, finiteSq, mask);
                        MaskNot(needsSqrt, normalSq, mask);
                        Sqrt(sqrtSq, sq, needsSqrt);
                        Abs(absX, xr, mask);
                        Select(norm, absX, sqrtSq, normalSq);
                    } else {
                        Sqrt(norm, sq, mask);
                    }
                    Maxs(norm, norm, eps, mask);
                    Mul(ys, yr, s, mask);
                    Sub(sub, dyr, ys, mask);
                    if constexpr (IsSameType<T_X, float>::value) {
                        // Refine the native quotient to avoid a 1-ULP division error.
                        Div<float, &FP32_PRECISE_DIV_MODE>(result, sub, norm, mask);
                    } else {
                        Div(result, sub, norm, mask);
                    }
                    StoreDx<T_X>(dxp, off, result, mask);
                }
            }
            qx_.FreeTensor(x);
            qy_.FreeTensor(y);
            qdy_.FreeTensor(dy);
            qdx_.EnQue(dx);
            this->CopyOut2D(qdx_, start, 1, n, 0);
        }
    }

private:
    TQue<QuePosition::VECIN, DEPTH_TWO> qx_, qy_, qdy_;
    TQue<QuePosition::VECOUT, DEPTH_TWO> qdx_;
};
} // namespace L2NormalizeGrad
#endif
