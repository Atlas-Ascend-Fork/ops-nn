/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swi_glu_grad_base.h
 * \brief
 */

#ifndef SWI_GLU_GRAD_BASE_H
#define SWI_GLU_GRAD_BASE_H
#include "kernel_operator.h"
#include "../inc/platform.h"

namespace SwiGluGrad {
using namespace AscendC;

const int32_t DOUBLE_BUFFER = 2;
const int32_t BUFFER_NUMBER = 5;
const uint32_t SPLIT_HALF = 2;

static constexpr uint32_t ONE_BLK_SIZE = platform::GetUbBlockSize();
constexpr uint32_t ONE_VL_SIZE = platform::GetVRegSize();

__aicore__ inline uint32_t RoundDownToVL(uint32_t x) { return x / ONE_VL_SIZE * ONE_VL_SIZE; }

static constexpr Reg::CastTrait castTraitB162B32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                    Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

static constexpr Reg::CastTrait castTraitB322B16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                    Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T>
class SwiGluGradBaseKernel {
public:
    __aicore__ inline SwiGluGradBaseKernel() = default;
    __aicore__ inline bool InitBase(const GluBaseTilingData& tilingData);

protected:
    __aicore__ inline void LoadOneTensor(const __ubuf__ void* input, Reg::RegTensor<float>& dst, Reg::MaskReg& preg,
                                         uint32_t offset);
    __aicore__ inline void StoreOneTensor(const __ubuf__ void* output, Reg::RegTensor<float>& src, Reg::MaskReg& preg,
                                          uint32_t offset);
    __aicore__ inline void Compute(LocalTensor<T>& xATensor, LocalTensor<T>& xBTensor, LocalTensor<T>& gradTensor,
                                   LocalTensor<T>& outATensor, LocalTensor<T>& outBTensor, int64_t dataCount);
    __aicore__ inline void ComputeInterleaved(__ubuf__ T* xAddr, __ubuf__ T* gradAddr, __ubuf__ T* outAddr,
                                              uint32_t rowLen, uint32_t colLen, uint32_t colTotal);

protected:
    int64_t rowTotalA_{0};
    int64_t colTotalA_{0};
    int64_t colTotal_{0};
    int64_t rowBase_{0};
    int64_t colBase_{0};
    int64_t rowTail_{0};
    int64_t colTail_{0};
    uint64_t ubSize_{0};
    uint32_t rowTileNum_{0};
    uint32_t colTileNum_{0};
    uint32_t usedCoreNum_{0};

    int64_t rowProcess_{0};
    int64_t colProcess_{0};
    int64_t partAStart_{0};
    int64_t partBStart_{0};
    int64_t gradStart_{0};
    int32_t blockIdx_{0};

    float beta_ = -1.0;
    static constexpr uint16_t vfFp32Block_ = platform::GetVRegSize() / sizeof(float);
    static constexpr uint16_t vfBlock_ = platform::GetVRegSize() / sizeof(T);
};

template <typename T>
__aicore__ inline bool SwiGluGradBaseKernel<T>::InitBase(const GluBaseTilingData& tilingData)
{
    blockIdx_ = GetBlockIdx();
    usedCoreNum_ = tilingData.usedCoreNum;
    if (blockIdx_ >= usedCoreNum_) {
        return false;
    }

    rowTotalA_ = tilingData.rowTotal;
    colTotalA_ = tilingData.colTotal;
    rowBase_ = tilingData.rowBase;
    colBase_ = tilingData.colBase;
    rowTail_ = tilingData.rowTail;
    colTail_ = tilingData.colTail;
    rowTileNum_ = tilingData.rowTileNum;
    colTileNum_ = tilingData.colTileNum;
    ubSize_ = tilingData.ubSize;

    colTotal_ = colTotalA_ * SPLIT_HALF;

    colProcess_ = (blockIdx_ + 1) % colTileNum_ != 0 ? colBase_ : colTail_;
    rowProcess_ = (blockIdx_ + 1) / colTileNum_ < (rowTileNum_ - 1) ? rowBase_ : rowTail_;

    int64_t colOffset = blockIdx_ % colTileNum_;
    int64_t rowOffset = blockIdx_ / colTileNum_;

    partAStart_ = rowBase_ * rowOffset * colTotal_ + colBase_ * colOffset;
    partBStart_ = partAStart_ + colTotalA_;
    gradStart_ = rowBase_ * rowOffset * colTotalA_ + colBase_ * colOffset;

    return true;
}

template <typename T>
__aicore__ inline void SwiGluGradBaseKernel<T>::LoadOneTensor(const __ubuf__ void* input, Reg::RegTensor<float>& dst,
                                                              Reg::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(xFp16, (__ubuf__ half*)(input) + offset);
        Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(xBf16, (__ubuf__ bfloat16_t*)(input) + offset);
        Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
    } else {
        Reg::LoadAlign(dst, (__ubuf__ float*)(input) + offset);
    }
}

template <typename T>
__aicore__ inline void SwiGluGradBaseKernel<T>::StoreOneTensor(const __ubuf__ void* output, Reg::RegTensor<float>& src,
                                                               Reg::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        Reg::RegTensor<half> xFp16;
        Cast<half, float, castTraitB322B16>(xFp16, src, preg);
        Reg::StoreAlign<half, Reg::StoreDist::DIST_PACK_B32>((__ubuf__ half*)(output) + offset, xFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        Reg::RegTensor<bfloat16_t> xBf16;
        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, src, preg);
        Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>((__ubuf__ bfloat16_t*)(output) + offset, xBf16,
                                                                   preg);
    } else {
        Reg::StoreAlign((__ubuf__ float*)(output) + offset, src, preg);
    }
}

template <typename T>
__aicore__ inline void SwiGluGradBaseKernel<T>::Compute(LocalTensor<T>& xATensor, LocalTensor<T>& xBTensor,
                                                        LocalTensor<T>& gradTensor, LocalTensor<T>& outATensor,
                                                        LocalTensor<T>& outBTensor, int64_t dataCount)
{
    __ubuf__ T* ubSrcAddrA = (__ubuf__ T*)xATensor.GetPhyAddr();
    __ubuf__ T* ubSrcAddrB = (__ubuf__ T*)xBTensor.GetPhyAddr();
    __ubuf__ T* ubGradAddr = (__ubuf__ T*)gradTensor.GetPhyAddr();
    __ubuf__ T* ubDstAddrA = (__ubuf__ T*)outATensor.GetPhyAddr();
    __ubuf__ T* ubDstAddrB = (__ubuf__ T*)outBTensor.GetPhyAddr();

    uint16_t repeatTimes = (dataCount + vfFp32Block_ - 1) / vfFp32Block_;

    float beta = beta_;
    uint32_t calcCount = dataCount;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> srcA;
        Reg::RegTensor<float> srcB;
        Reg::RegTensor<float> grad;
        Reg::RegTensor<float> dstA;
        Reg::RegTensor<float> dstB;
        Reg::RegTensor<float> sigReg;
        Reg::RegTensor<float> tmpReg;
        Reg::RegTensor<float> oneReg;

        Reg::MaskReg preg = Reg::UpdateMask<float>(calcCount);
        Duplicate(oneReg, -beta, preg);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            uint32_t offset = i * vfFp32Block_;
            LoadOneTensor(ubSrcAddrA, srcA, preg, offset);
            LoadOneTensor(ubSrcAddrB, srcB, preg, offset);
            LoadOneTensor(ubGradAddr, grad, preg, offset);

            Muls(tmpReg, srcA, beta, preg);
            Exp(tmpReg, tmpReg, preg);
            Adds(tmpReg, tmpReg, -beta, preg);
            Div(sigReg, oneReg, tmpReg, preg);

            Muls(dstA, sigReg, beta, preg);
            Adds(dstA, dstA, -beta, preg);
            Mul(dstA, dstA, srcA, preg);
            Adds(dstA, dstA, -beta, preg);
            Mul(tmpReg, grad, sigReg, preg);
            Mul(dstA, dstA, tmpReg, preg);
            Mul(dstA, dstA, srcB, preg);

            Mul(dstB, tmpReg, srcA, preg);

            StoreOneTensor(ubDstAddrA, dstA, preg, offset);
            StoreOneTensor(ubDstAddrB, dstB, preg, offset);
        }
    }
}

template <typename T>
__aicore__ inline void SwiGluGradBaseKernel<T>::ComputeInterleaved(__ubuf__ T* xAddr, __ubuf__ T* gradAddr,
                                                                   __ubuf__ T* outAddr, uint32_t rowLen,
                                                                   uint32_t colLen, uint32_t colTotal)
{
    uint16_t mainBlocks = colLen / vfFp32Block_;
    uint32_t tailCount = colLen - mainBlocks * vfFp32Block_;
    float beta = beta_;
    uint16_t tailRows = tailCount > 0 ? rowLen : 0;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> srcA;
        Reg::RegTensor<float> srcB;
        Reg::RegTensor<float> grad;
        Reg::RegTensor<float> dstA;
        Reg::RegTensor<float> dstB;
        Reg::RegTensor<float> sigReg;
        Reg::RegTensor<float> tmpReg;
        Reg::RegTensor<float> oneReg;

        Reg::MaskReg pAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg pTail = Reg::UpdateMask<float>(tailCount);
        Duplicate(oneReg, -beta, pAll);

        for (uint16_t row = 0; row < rowLen; ++row) {
            uint32_t rowOffsetX = row * colTotal;
            uint32_t rowOffsetG = row * colLen;
            for (uint16_t b = 0; b < mainBlocks; ++b) {
                uint32_t off = b * vfFp32Block_;
                LoadOneTensor(xAddr, srcA, pAll, rowOffsetX + off);
                LoadOneTensor(xAddr, srcB, pAll, rowOffsetX + colLen + off);
                LoadOneTensor(gradAddr, grad, pAll, rowOffsetG + off);

                Muls(tmpReg, srcA, beta, pAll);
                Exp(tmpReg, tmpReg, pAll);
                Adds(tmpReg, tmpReg, -beta, pAll);
                Div(sigReg, oneReg, tmpReg, pAll);

                Muls(dstA, sigReg, beta, pAll);
                Adds(dstA, dstA, -beta, pAll);
                Mul(dstA, dstA, srcA, pAll);
                Adds(dstA, dstA, -beta, pAll);
                Mul(tmpReg, grad, sigReg, pAll);
                Mul(dstA, dstA, tmpReg, pAll);
                Mul(dstA, dstA, srcB, pAll);

                Mul(dstB, tmpReg, srcA, pAll);

                StoreOneTensor(outAddr, dstA, pAll, rowOffsetX + off);
                StoreOneTensor(outAddr, dstB, pAll, rowOffsetX + colLen + off);
            }
        }

        for (uint16_t row = 0; row < tailRows; ++row) {
            uint32_t rowOffsetX = row * colTotal;
            uint32_t rowOffsetG = row * colLen;
            uint32_t off = mainBlocks * vfFp32Block_;
            LoadOneTensor(xAddr, srcA, pTail, rowOffsetX + off);
            LoadOneTensor(xAddr, srcB, pTail, rowOffsetX + colLen + off);
            LoadOneTensor(gradAddr, grad, pTail, rowOffsetG + off);

            Muls(tmpReg, srcA, beta, pTail);
            Exp(tmpReg, tmpReg, pTail);
            Adds(tmpReg, tmpReg, -beta, pTail);
            Div(sigReg, oneReg, tmpReg, pTail);

            Muls(dstA, sigReg, beta, pTail);
            Adds(dstA, dstA, -beta, pTail);
            Mul(dstA, dstA, srcA, pTail);
            Adds(dstA, dstA, -beta, pTail);
            Mul(tmpReg, grad, sigReg, pTail);
            Mul(dstA, dstA, tmpReg, pTail);
            Mul(dstA, dstA, srcB, pTail);

            Mul(dstB, tmpReg, srcA, pTail);

            StoreOneTensor(outAddr, dstA, pTail, rowOffsetX + off);
            StoreOneTensor(outAddr, dstB, pTail, rowOffsetX + colLen + off);
        }
    }
}

} // namespace SwiGluGrad

#endif
