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
 * \file dynamic_hi_f4_quant_not_tail_axis.h
 * \brief
 */
#ifndef DYNAMIC_HIF4_QUANT_NOT_TAIL_AXIS_HIFLOAT4_H
#define DYNAMIC_HIF4_QUANT_NOT_TAIL_AXIS_HIFLOAT4_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "dynamic_hi_f4_quant_struct.h"

namespace DynamicHiF4Quant {
using namespace AscendC;

template <typename ST, typename DT, uint32_t useQueNum>
class DynamicHiF4QuantNotTailAxis {
public:
    __aicore__ inline DynamicHiF4QuantNotTailAxis(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale,
                                const DynamicHiF4QuantTilingData* __restrict tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitParams();
    __aicore__ inline void ProcessOneLoop(int64_t calcCol, int64_t calcRow, int64_t xUbOffset, int64_t scale2Offset,
                                          int64_t innerLoopMaxper4, int64_t innerLoopMaxper8, int64_t regLoop);
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t yOffset, int64_t scale2OutOffset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void ComputeAll(int64_t blockCount, int64_t dataLen, int64_t innerLoopMaxper4,
                                      int64_t innerLoopMaxper8, int64_t regLoop);
    __aicore__ inline void ComputeMaxPer4(uint16_t innerLoopMaxper4, __ubuf__ ST* xAddr,
                                          __ubuf__ uint16_t* maxPer4BufAddr);
    __aicore__ inline void ComputeMaxPer8(uint16_t innerLoopMaxper8, __ubuf__ uint16_t* maxPer4BufReadAddr,
                                          __ubuf__ uint16_t* maxPer8BufAddr);
    __aicore__ inline void ComputeE6M2(uint16_t regLoop, __ubuf__ uint16_t* maxPer8BufReadAddr,
                                       __ubuf__ uint16_t* e8BufAddr);
    __aicore__ inline void ComputeTmpScale(uint16_t regLoop, uint16_t innerLoopMaxper4, uint16_t innerLoopMaxper8,
                                           __ubuf__ uint16_t* maxPer4BufRead2Addr,
                                           __ubuf__ uint16_t* maxPer8BufRead2Addr, __ubuf__ uint16_t* e8BufRead1Addr,
                                           __ubuf__ uint8_t* e1p8BufAddr, __ubuf__ uint8_t* e1p16BufAddr,
                                           __ubuf__ uint16_t* tmpScaleAddr);
    __aicore__ inline void ComputeScale(uint16_t regLoop, __ubuf__ uint8_t* e8BufRead2Addr,
                                        __ubuf__ uint8_t* e1p8BufReadAddr, __ubuf__ uint8_t* e1p16BufReadAddr,
                                        __ubuf__ uint8_t* scaleAddr1, __ubuf__ uint16_t* scaleAddr2);
    __aicore__ inline void ComputeOut(uint16_t innerLoopMaxper4, __ubuf__ ST* xAddr2,
                                      __ubuf__ uint16_t* tmpScaleReadAddr, __ubuf__ int32_t* yAddr);

private:
protected:
    const DynamicHiF4QuantTilingData* __restrict tilingData_{nullptr};
    TPipe* pipe_{nullptr};

    TQue<QuePosition::VECIN, DB_BUFFER> inQueueX_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outScaleFrontQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outScaleBackQueue_;

    TBuf<TPosition::VECCALC> bufE8_;
    TBuf<TPosition::VECCALC> buf1E8_;
    TBuf<TPosition::VECCALC> buf1E16_;
    TBuf<TPosition::VECCALC> bufTmpScale_;
    TBuf<TPosition::VECCALC> bufMaxPer4_;
    TBuf<TPosition::VECCALC> bufMaxPer8_;

    GlobalTensor<ST> xGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint16_t> scaleGm_;

    int64_t blockIdx_ = 0;
    int64_t blockOffset_ = 0;
    int64_t loopPerCore_ = 0;
    int64_t ubRowLen_ = 0;
    int64_t ubRowLenTail_ = 0;
    int64_t ubRowCount_ = 0;
    int64_t ubRowCountTail_ = 0;
    int64_t dimNeg2ScaleNum_ = 0;
    int64_t dimNeg1ScaleNum_ = 0;
    int64_t blockCountPerPage_ = 0;
    // 由 scale_alg 决定的 Eb/Ec 比较阈值
    // scale_alg=0(默认): Eb阈值4, Ec阈值2; scale_alg=1: Eb阈值3.75, Ec阈值1.875
    bfloat16_t e8Thresh_ = static_cast<bfloat16_t>(4.0F);
    bfloat16_t e16Thresh_ = static_cast<bfloat16_t>(2.0F);
    int64_t blockSize_ = 64;

    int64_t inBuffer_ = 0;
    int64_t outBuffer_ = 0;
    int64_t scaleFrontBuffer_ = 0;
    int64_t scaleBackBuffer_ = 0;
    int64_t bufE8Buffer_ = 0;
    int64_t buf1E8Buffer_ = 0;
    int64_t buf1E16Buffer_ = 0;
    int64_t maxPer4Buffer_ = 0;
    int64_t maxPer8Buffer_ = 0;
    int64_t tmpScaleBuffer_ = 0;

    static constexpr int64_t UBBlockSize_ = 32;
    static constexpr uint16_t MAXPER8_REDUCE_LOOP = 7;
    static constexpr uint16_t E6M2_BLOCK_PER_GROUP = 8;
    static constexpr uint16_t MAXPER4_SUBGROUP_PER_MAXPER8 = 2;
    constexpr static MicroAPI::CastTrait castTrait_ = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
};

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ProcessOneLoop(
    int64_t calcCol, int64_t calcRow, int64_t xUbOffset, int64_t scale2Offset, int64_t innerLoopMaxper4,
    int64_t innerLoopMaxper8, int64_t regLoop)
{
    CopyIn(xUbOffset, calcRow, calcCol);
    ComputeAll(calcRow, calcCol, innerLoopMaxper4, innerLoopMaxper8, regLoop);
    CopyOut(xUbOffset, scale2Offset, calcRow, calcCol);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::CopyIn(int64_t offset, int64_t blockCount,
                                                                              int64_t dataLen)
{
    // 第一行第一块到第二行的第一块，间隔长度
    int64_t rightPadding = Ops::Base::CeilAlign(static_cast<int64_t>(dataLen * sizeof(ST)), UBBlockSize_) / sizeof(ST) -
                           dataLen;
    DataCopyExtParams copyInParams = {static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen * sizeof(ST)),
                                      static_cast<uint32_t>((tilingData_->dimNeg1 - dataLen) * sizeof(ST)),
                                      static_cast<uint32_t>((ubRowLen_ - dataLen) * sizeof(ST) / UBBlockSize_),
                                      static_cast<uint32_t>(0)};
    DataCopyPadExtParams<ST> padParams{true, 0, static_cast<uint8_t>(rightPadding), 0};

    LocalTensor<ST> xLocal = inQueueX_.template AllocTensor<ST>();
    DataCopyPad(xLocal, xGm_[offset], copyInParams, padParams);
    inQueueX_.template EnQue(xLocal);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::CopyOut(int64_t yOffset, int64_t scale2OutOffset,
                                                                               int64_t blockCount, int64_t dataLen)
{
    uint16_t outBurst = 0;
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    int64_t YOffset = yOffset;

    int64_t scaleExtraOffset = tilingData_->dimNeg1;
    uint32_t scaleSrcStride = Ops::Base::CeilDiv((ubRowLen_ - dataLen), UBBlockSize_) * sizeof(uint16_t);
    uint32_t scaleDstStride = DIGIT_TWO * tilingData_->dimNeg1 - dataLen;

    outBurst = blockCount;
    outBlockLen = dataLen / DIGIT_TWO * sizeof(uint8_t);
    srcStride = ((ubRowLen_ - dataLen) / DIGIT_TWO * sizeof(uint8_t) / UBBlockSize_);
    dstStride = (tilingData_->dimNeg1 - dataLen) / DIGIT_TWO * sizeof(uint8_t);
    YOffset = yOffset / DIGIT_TWO;

    DataCopyExtParams yCopyOutParams = {static_cast<uint16_t>(outBurst), static_cast<uint32_t>(outBlockLen),
                                        static_cast<uint32_t>(srcStride), static_cast<uint32_t>(dstStride),
                                        static_cast<uint32_t>(0)};

    DataCopyExtParams scale2CopyOutParams = {
        static_cast<uint16_t>(Ops::Base::CeilDiv(blockCount, blockSize_)),
        static_cast<uint32_t>(dataLen * DIGIT_TWO * sizeof(uint8_t)), static_cast<uint32_t>(scaleSrcStride),
        static_cast<uint32_t>((DIGIT_TWO * tilingData_->dimNeg1 - dataLen) * sizeof(uint16_t)),
        static_cast<uint32_t>(0)};

    auto yLocalOut = outQueue_.template DeQue<uint8_t>();
    DataCopyPad(yGm_[YOffset], yLocalOut, yCopyOutParams);
    outQueue_.FreeTensor(yLocalOut);

    auto scaleLocalFront = outScaleFrontQueue_.template DeQue<uint16_t>();
    DataCopyPad(scaleGm_[scale2OutOffset], scaleLocalFront, scale2CopyOutParams);
    outScaleFrontQueue_.FreeTensor(scaleLocalFront);

    auto scaleLocalBack = outScaleBackQueue_.template DeQue<uint16_t>();
    DataCopyPad(scaleGm_[scale2OutOffset + scaleExtraOffset], scaleLocalBack, scale2CopyOutParams);
    outScaleBackQueue_.FreeTensor(scaleLocalBack);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ComputeAll(int64_t calcCol, int64_t calcRow,
                                                                                  int64_t innerLoopMaxper4,
                                                                                  int64_t innerLoopMaxper8,
                                                                                  int64_t regLoop)
{
    LocalTensor<ST> x = inQueueX_.template DeQue<ST>();
    LocalTensor<uint8_t> y = outQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint16_t> scalefront = outScaleFrontQueue_.template AllocTensor<uint16_t>();
    LocalTensor<uint16_t> scaleback = outScaleBackQueue_.template AllocTensor<uint16_t>();

    LocalTensor<uint16_t> e8buf = bufE8_.Get<uint16_t>();
    LocalTensor<uint8_t> e1p8buf = buf1E8_.Get<uint8_t>();
    LocalTensor<uint8_t> e1p16buf = buf1E16_.Get<uint8_t>();
    LocalTensor<uint16_t> tmpscale = bufTmpScale_.Get<uint16_t>();
    LocalTensor<uint16_t> maxper4 = bufMaxPer4_.Get<uint16_t>();
    LocalTensor<uint16_t> maxper8 = bufMaxPer8_.Get<uint16_t>();

    auto xAddr = (__ubuf__ ST*)x.GetPhyAddr();
    auto xAddr2 = (__ubuf__ ST*)x.GetPhyAddr();
    auto yAddr = (__ubuf__ int32_t*)y.GetPhyAddr();
    auto scaleAddr1 = (__ubuf__ uint8_t*)scalefront.GetPhyAddr();
    auto scaleAddr2 = (__ubuf__ uint16_t*)scaleback.GetPhyAddr();

    auto e8BufAddr = (__ubuf__ uint16_t*)e8buf.GetPhyAddr();
    auto e8BufAddr2 = (__ubuf__ uint8_t*)e8buf.GetPhyAddr();
    auto e1p8BufAddr = (__ubuf__ uint8_t*)e1p8buf.GetPhyAddr();
    auto e1p16BufAddr = (__ubuf__ uint8_t*)e1p16buf.GetPhyAddr();
    auto tmpScaleAddr = (__ubuf__ uint16_t*)tmpscale.GetPhyAddr();
    auto maxPer4BufAddr = (__ubuf__ uint16_t*)maxper4.GetPhyAddr();
    auto maxPer8BufAddr = (__ubuf__ uint16_t*)maxper8.GetPhyAddr();

    auto e8BufRead1Addr = e8BufAddr;
    auto e8BufRead2Addr = e8BufAddr2;
    auto e1p8BufReadAddr = e1p8BufAddr;
    auto e1p16BufReadAddr = e1p16BufAddr;
    auto tmpScaleReadAddr = tmpScaleAddr;
    auto maxPer4BufReadAddr = maxPer4BufAddr;
    auto maxPer4BufRead2Addr = maxPer4BufAddr;
    auto maxPer8BufReadAddr = maxPer8BufAddr;
    auto maxPer8BufRead2Addr = maxPer8BufAddr;

    ComputeMaxPer4(static_cast<uint16_t>(innerLoopMaxper4), xAddr, maxPer4BufAddr);
    ComputeMaxPer8(static_cast<uint16_t>(innerLoopMaxper8), maxPer4BufReadAddr, maxPer8BufAddr);
    ComputeE6M2(static_cast<uint16_t>(regLoop), maxPer8BufReadAddr, e8BufAddr);
    ComputeTmpScale(static_cast<uint16_t>(regLoop), static_cast<uint16_t>(innerLoopMaxper4),
                    static_cast<uint16_t>(innerLoopMaxper8), maxPer4BufRead2Addr, maxPer8BufRead2Addr, e8BufRead1Addr,
                    e1p8BufAddr, e1p16BufAddr, tmpScaleAddr);
    ComputeScale(static_cast<uint16_t>(regLoop), e8BufRead2Addr, e1p8BufReadAddr, e1p16BufReadAddr, scaleAddr1,
                 scaleAddr2);
    ComputeOut(static_cast<uint16_t>(innerLoopMaxper4), xAddr2, tmpScaleReadAddr, yAddr);

    outQueue_.template EnQue(y);
    outScaleFrontQueue_.template EnQue(scalefront);
    outScaleBackQueue_.template EnQue(scaleback);
    inQueueX_.template FreeTensor(x);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ComputeMaxPer4(uint16_t innerLoopMaxper4,
                                                                                      __ubuf__ ST* xAddr,
                                                                                      __ubuf__ uint16_t* maxPer4BufAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<ST> vregIn1;
        MicroAPI::RegTensor<ST> vregIn2;
        MicroAPI::RegTensor<ST> vregIn3;
        MicroAPI::RegTensor<ST> vregIn4;
        MicroAPI::RegTensor<bfloat16_t> vregOutbf16;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        AscendC::MicroAPI::Duplicate(expMaskBF16, 0x7fff);
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < innerLoopMaxper4; loopIdx++) {
            MicroAPI::DataCopy<ST, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(vregIn1,
                                                                                                           xAddr, 128);
            MicroAPI::DataCopy<ST, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(vregIn2,
                                                                                                           xAddr, 128);
            MicroAPI::DataCopy<ST, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(vregIn3,
                                                                                                           xAddr, 128);
            MicroAPI::DataCopy<ST, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(vregIn4,
                                                                                                           xAddr, 128);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1, expMaskBF16, maskAll);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2, expMaskBF16, maskAll);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3, expMaskBF16, maskAll);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4, expMaskBF16, maskAll);
            MicroAPI::Max(vregIn1, vregIn1, vregIn2, maskAll);
            MicroAPI::Max(vregIn3, vregIn3, vregIn4, maskAll);
            MicroAPI::Max(vregIn1, vregIn1, vregIn3, maskAll);

            if constexpr (IsSameType<ST, half>::value) {
                AscendC::MicroAPI::Cast<bfloat16_t, ST, castTraitHalf2Bf16>(vregOutbf16, vregIn1, maskAll);
                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maxPer4BufAddr, (MicroAPI::RegTensor<uint16_t>&)vregOutbf16, 128, maskAll);
            } else {
                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maxPer4BufAddr, (MicroAPI::RegTensor<uint16_t>&)vregIn1, 128, maskAll);
            }
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ComputeMaxPer8(
    uint16_t innerLoopMaxper8, __ubuf__ uint16_t* maxPer4BufReadAddr, __ubuf__ uint16_t* maxPer8BufAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> vregIn1;
        MicroAPI::RegTensor<uint16_t> vregIn2;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        for (uint16_t loopIdx = 0; loopIdx < innerLoopMaxper8; loopIdx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                vregIn1, maxPer4BufReadAddr, 128);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                vregIn2, maxPer4BufReadAddr, 128);
            MicroAPI::Max<ST>((MicroAPI::RegTensor<ST>&)vregIn1, (MicroAPI::RegTensor<ST>&)vregIn1,
                              (MicroAPI::RegTensor<ST>&)vregIn2, maskAll);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxPer8BufAddr, (MicroAPI::RegTensor<uint16_t>&)vregIn1, 128, maskAll);
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ComputeE6M2(
    uint16_t regLoop, __ubuf__ uint16_t* maxPer8BufReadAddr, __ubuf__ uint16_t* e8BufAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> vregMaxper8;
        MicroAPI::RegTensor<uint16_t> vregGround;
        MicroAPI::RegTensor<uint16_t> vreg0E6m2;
        MicroAPI::RegTensor<uint16_t> vregNumscale;
        MicroAPI::Duplicate(vregNumscale, 0x3e12);

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitE6M2 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            // MicroAPI::Duplicate(vregGround, 0x0000);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<uint16_t>&)vregGround, maxPer8BufReadAddr, 128);
            for (uint16_t loop2Idx = 0; loop2Idx < MAXPER8_REDUCE_LOOP; loop2Idx++) {
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    (MicroAPI::RegTensor<uint16_t>&)vregMaxper8, maxPer8BufReadAddr, 128);
                MicroAPI::Max<bfloat16_t>((MicroAPI::RegTensor<bfloat16_t>&)vregGround,
                                          (MicroAPI::RegTensor<bfloat16_t>&)vregGround,
                                          (MicroAPI::RegTensor<bfloat16_t>&)vregMaxper8, maskAll);
            }
            MicroAPI::Mul((MicroAPI::RegTensor<bfloat16_t>&)vregGround, (MicroAPI::RegTensor<bfloat16_t>&)vregGround,
                          (MicroAPI::RegTensor<bfloat16_t>&)vregNumscale, maskAll);
            MicroAPI::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>(
                (MicroAPI::RegTensor<fp8_e6m2_t>&)vreg0E6m2, (MicroAPI::RegTensor<bfloat16_t>&)vregGround, maskAll);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                e8BufAddr, (MicroAPI::RegTensor<uint16_t>&)vreg0E6m2, 128, maskAll);
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ComputeTmpScale(
    uint16_t regLoop, uint16_t innerLoopMaxper4, uint16_t innerLoopMaxper8, __ubuf__ uint16_t* maxPer4BufRead2Addr,
    __ubuf__ uint16_t* maxPer8BufRead2Addr, __ubuf__ uint16_t* e8BufRead1Addr, __ubuf__ uint8_t* e1p8BufAddr,
    __ubuf__ uint8_t* e1p16BufAddr, __ubuf__ uint16_t* tmpScaleAddr)
{
    __VEC_SCOPE__
    {
        static constexpr AscendC::MicroAPI::CastTrait castTraitReciprocalCast = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        MicroAPI::RegTensor<bfloat16_t> vregMaxper4;
        MicroAPI::RegTensor<bfloat16_t> vregMaxper8;
        MicroAPI::RegTensor<bfloat16_t> vregMaxper64;
        MicroAPI::RegTensor<bfloat16_t> vb16_Ea_rec_bf16;
        MicroAPI::RegTensor<bfloat16_t> vregTmpScale;

        float number1 = 1.0;
        MicroAPI::RegTensor<bfloat16_t> vregNum1;
        MicroAPI::Duplicate(vregNum1, number1);
        float numberhalf = 0.5;
        MicroAPI::RegTensor<bfloat16_t> vregNumhalf;
        MicroAPI::Duplicate(vregNumhalf, static_cast<bfloat16_t>(numberhalf));

        MicroAPI::MaskReg vb1E8Comp;
        MicroAPI::MaskReg vb1E8CompTmp;
        MicroAPI::MaskReg vb1E16Comp;
        MicroAPI::MaskReg vb1E16CompTmp;
        MicroAPI::UnalignReg ureg1;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<uint16_t>&)vregMaxper64, e8BufRead1Addr, 128);
            MicroAPI::ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitReciprocalCast>(
                vb16_Ea_rec_bf16, (MicroAPI::RegTensor<fp8_e6m2_t>&)vregMaxper64, maskAll);
            for (uint16_t loop2Idx = 0; loop2Idx < E6M2_BLOCK_PER_GROUP; loop2Idx++) {
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    (MicroAPI::RegTensor<uint16_t>&)vregMaxper8, maxPer8BufRead2Addr, 128);
                MicroAPI::Mul(vregMaxper8, vregMaxper8, vb16_Ea_rec_bf16, maskAll);
                MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vregMaxper8, e8Thresh_, maskAll);
                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                    e1p8BufAddr, vb1E8Comp, 32);
                MicroAPI::Select(vregMaxper8, vregNumhalf, vregNum1, vb1E8Comp);
                for (uint16_t loop3Idx = 0; loop3Idx < MAXPER4_SUBGROUP_PER_MAXPER8; loop3Idx++) {
                    MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                       MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<uint16_t>&)vregMaxper4,
                                                                      maxPer4BufRead2Addr, 128);
                    MicroAPI::Mul(vregMaxper4, vregMaxper4, vb16_Ea_rec_bf16, maskAll);
                    MicroAPI::Mul(vregMaxper4, vregMaxper4, vregMaxper8, maskAll);
                    MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vregMaxper4, e16Thresh_, maskAll);
                    MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                        e1p16BufAddr, vb1E16Comp, 32);
                    MicroAPI::Select(vregMaxper4, vregNumhalf, vregNum1, vb1E16Comp);
                    MicroAPI::Mul(vregMaxper4, vregMaxper4, vregMaxper8, maskAll);
                    MicroAPI::Mul(vregTmpScale, vregMaxper4, vb16_Ea_rec_bf16, maskAll);
                    AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        tmpScaleAddr, (MicroAPI::RegTensor<uint16_t>&)vregTmpScale, 128, maskAll);
                }
            }
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ComputeScale(
    uint16_t regLoop, __ubuf__ uint8_t* e8BufRead2Addr, __ubuf__ uint8_t* e1p8BufReadAddr,
    __ubuf__ uint8_t* e1p16BufReadAddr, __ubuf__ uint8_t* scaleAddr1, __ubuf__ uint16_t* scaleAddr2)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint8_t> vregBuild1;
        MicroAPI::RegTensor<uint16_t> vregBuild2;
        MicroAPI::RegTensor<uint8_t> vregGround1;
        MicroAPI::RegTensor<uint16_t> vregGround2;
        MicroAPI::RegTensor<uint8_t> vregE8;

        MicroAPI::RegTensor<uint8_t> vreg8Flex1, vreg8Flex2, vreg8Flex3, vreg8Flex4, vreg8Flex5, vreg8Flex6, vreg8Flex7,
            vreg8Flex8;
        MicroAPI::RegTensor<uint16_t> vreg16Flex1, vreg16Flex2, vreg16Flex3, vreg16Flex4, vreg16Flex5, vreg16Flex6,
            vreg16Flex7, vreg16Flex8, vreg16Flex9, vreg16Flex10, vreg16Flex11, vreg16Flex12, vreg16Flex13, vreg16Flex14,
            vreg16Flex15, vreg16Flex16;
        MicroAPI::Duplicate(vreg8Flex1, 0x01);
        MicroAPI::Duplicate(vreg8Flex2, 0x02);
        MicroAPI::Duplicate(vreg8Flex3, 0x04);
        MicroAPI::Duplicate(vreg8Flex4, 0x08);
        MicroAPI::Duplicate(vreg8Flex5, 0x10);
        MicroAPI::Duplicate(vreg8Flex6, 0x20);
        MicroAPI::Duplicate(vreg8Flex7, 0x40);
        MicroAPI::Duplicate(vreg8Flex8, 0x80);
        MicroAPI::Duplicate(vreg16Flex1, 0x0001);
        MicroAPI::Duplicate(vreg16Flex2, 0x0002);
        MicroAPI::Duplicate(vreg16Flex3, 0x0004);
        MicroAPI::Duplicate(vreg16Flex4, 0x0008);
        MicroAPI::Duplicate(vreg16Flex5, 0x0010);
        MicroAPI::Duplicate(vreg16Flex6, 0x0020);
        MicroAPI::Duplicate(vreg16Flex7, 0x0040);
        MicroAPI::Duplicate(vreg16Flex8, 0x0080);
        MicroAPI::Duplicate(vreg16Flex9, 0x0100);
        MicroAPI::Duplicate(vreg16Flex10, 0x0200);
        MicroAPI::Duplicate(vreg16Flex11, 0x0400);
        MicroAPI::Duplicate(vreg16Flex12, 0x0800);
        MicroAPI::Duplicate(vreg16Flex13, 0x1000);
        MicroAPI::Duplicate(vreg16Flex14, 0x2000);
        MicroAPI::Duplicate(vreg16Flex15, 0x4000);
        MicroAPI::Duplicate(vreg16Flex16, 0x8000);

        MicroAPI::Duplicate(vregGround1, 0x00);
        MicroAPI::Duplicate(vregGround2, 0x0000);
        MicroAPI::MaskReg maskof2e18;
        MicroAPI::MaskReg maskof2e18Tmp;
        MicroAPI::MaskReg maskof2e116;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();

        for (uint16_t loopIdx = 0; loopIdx < (regLoop + 1) / 2; loopIdx++) {
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DS_B8>(
                vregE8, e8BufRead2Addr, 512);

            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18, e1p8BufReadAddr + loopIdx * 512);
            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18Tmp,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 256);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Add(vregBuild1, vregGround1, vreg8Flex1, maskof2e18);

            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 32);
            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18Tmp,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 256 + 32);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Add<uint8_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild1, vregBuild1, vreg8Flex2, maskof2e18);

            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 64);
            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18Tmp,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 256 + 64);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Add<uint8_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild1, vregBuild1, vreg8Flex3, maskof2e18);

            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 96);
            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18Tmp,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 256 + 96);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Add<uint8_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild1, vregBuild1, vreg8Flex4, maskof2e18);

            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 128);
            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18Tmp,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 256 + 128);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Add<uint8_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild1, vregBuild1, vreg8Flex5, maskof2e18);

            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 160);
            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18Tmp,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 256 + 160);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Add<uint8_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild1, vregBuild1, vreg8Flex6, maskof2e18);

            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 192);
            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18Tmp,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 256 + 192);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Add<uint8_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild1, vregBuild1, vreg8Flex7, maskof2e18);

            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 224);
            MicroAPI::DataCopy<uint8_t, MicroAPI::MaskDist::DIST_NORM>(maskof2e18Tmp,
                                                                       e1p8BufReadAddr + loopIdx * 512 + 256 + 224);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Add<uint8_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild1, vregBuild1, vreg8Flex8, maskof2e18);

            MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_INTLV_B8>(
                scaleAddr1 + loopIdx * 512, (MicroAPI::RegTensor<uint8_t>&)vregE8,
                (MicroAPI::RegTensor<uint8_t>&)vregBuild1, maskAll);
        }

        for (uint16_t loop3Idx = 0; loop3Idx < regLoop; loop3Idx++) {
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add(vregBuild2, vregGround2, vreg16Flex1, maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex2, maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex3, maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex4, maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex5, maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex6, maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex7, maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex8, maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex9, maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex10,
                                                                      maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex11,
                                                                      maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex12,
                                                                      maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex13,
                                                                      maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex14,
                                                                      maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex15,
                                                                      maskof2e116);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, e1p16BufReadAddr, 32);
            MicroAPI::Add<uint16_t, MicroAPI::MaskMergeMode::MERGING>(vregBuild2, vregBuild2, vreg16Flex16,
                                                                      maskof2e116);

            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                scaleAddr2, (MicroAPI::RegTensor<uint16_t>&)vregBuild2, 128, maskAll);
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::ComputeOut(uint16_t innerLoopMaxper4,
                                                                                  __ubuf__ ST* xAddr2,
                                                                                  __ubuf__ uint16_t* tmpScaleReadAddr,
                                                                                  __ubuf__ int32_t* yAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<ST> vregOut;
        MicroAPI::RegTensor<uint16_t> vregScaleTmp;
        MicroAPI::RegTensor<fp4x2_e1m2_t> vbOut;
        MicroAPI::RegTensor<bfloat16_t> vregOutbf16;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < innerLoopMaxper4; loopIdx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                vregScaleTmp, tmpScaleReadAddr, 128);
            for (uint16_t loop2Idx = 0; loop2Idx < 4; loop2Idx++) { // 4 elements per micro-block
                if constexpr (IsSameType<ST, half>::value) {
                    MicroAPI::DataCopy<ST, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                        vregOut, xAddr2, 128);
                    AscendC::MicroAPI::Cast<bfloat16_t, ST, castTraitHalf2Bf16>(
                        (MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16, vregOut, maskAll);
                } else {
                    MicroAPI::DataCopy<ST, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                        vregOutbf16, xAddr2, 128);
                }
                MicroAPI::Mul((MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16,
                              (MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16,
                              (MicroAPI::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
                MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(
                    vbOut, (MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16, maskAll);
                MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                   MicroAPI::StoreDist::DIST_PACK4_B32>(yAddr, (MicroAPI::RegTensor<int32_t>&)vbOut, 16,
                                                                        maskAll);
            }
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    int64_t headCoreNum = tilingData_->headCoreNum;
    if (blockIdx_ < headCoreNum) {
        loopPerCore_ = tilingData_->blockPerHeadCore;
        // 切分基本块个数偏移
        blockOffset_ = blockIdx_ * loopPerCore_;
    } else {
        loopPerCore_ = tilingData_->blockPerTailCore;
        blockOffset_ = headCoreNum * tilingData_->blockPerHeadCore + (blockIdx_ - headCoreNum) * loopPerCore_;
    }

    // 一次vf计算的行长度，如果是tail后续处理,128
    ubRowLen_ = tilingData_->blockW;
    ubRowLenTail_ = tilingData_->dimNeg1Tail;

    // 一次UB计算的行数，如果是tail后续处理
    ubRowCount_ = tilingData_->splitBlockH;
    ubRowCountTail_ = tilingData_->dimNeg2Tail;

    // 一个batch总共多少个切分基本块
    blockCountPerPage_ = tilingData_->blockCountPerBatch;

    // 一个batch的-2轴scale行数
    dimNeg2ScaleNum_ = tilingData_->scale2RowCountPerBatch;

    // 一个batch的-1轴scale列数
    dimNeg1ScaleNum_ = tilingData_->scale1ColCountPerBatch;

    // 根据 scale_alg 选择 Eb/Ec 比较阈值
    e8Thresh_ = (tilingData_->scaleAlg == 1) ? static_cast<bfloat16_t>(3.75F) : static_cast<bfloat16_t>(4.0F);
    e16Thresh_ = (tilingData_->scaleAlg == 1) ? static_cast<bfloat16_t>(1.875F) : static_cast<bfloat16_t>(2.0F);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicHiF4QuantTilingData* __restrict tilingData)
{
    tilingData_ = tilingData;
    InitParams();
    blockIdx_ = GetBlockIdx();

    xGm_.SetGlobalBuffer((__gm__ ST*)(x));
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y));
    scaleGm_.SetGlobalBuffer((__gm__ uint16_t*)(scale));

    inBuffer_ = ubRowLen_ * ubRowCount_ * sizeof(ST);
    outBuffer_ = ubRowLen_ * ubRowCount_ * sizeof(DT);
    scaleFrontBuffer_ = ubRowLen_ * static_cast<int64_t>(((ubRowCount_ + 127) / 128) * 128) / blockSize_ *
                        sizeof(uint16_t);
    scaleBackBuffer_ = ubRowLen_ * ubRowCount_ / blockSize_ * sizeof(uint16_t);
    bufE8Buffer_ = ubRowLen_ * ubRowCount_ / blockSize_ * sizeof(ST);
    buf1E8Buffer_ = ubRowLen_ * ubRowCount_ / blockSize_ * sizeof(ST);
    buf1E16Buffer_ = ubRowLen_ * ubRowCount_ / blockSize_ * sizeof(ST) * 2;
    maxPer4Buffer_ = ubRowLen_ * ubRowCount_ / 4 * sizeof(ST);
    maxPer8Buffer_ = ubRowLen_ * ubRowCount_ / 8 * sizeof(ST);
    tmpScaleBuffer_ = ubRowLen_ * ubRowCount_ / 4 * sizeof(ST);

    pipe_->InitBuffer(inQueueX_, useQueNum, inBuffer_);
    pipe_->InitBuffer(outQueue_, useQueNum, outBuffer_);
    pipe_->InitBuffer(outScaleFrontQueue_, useQueNum, scaleFrontBuffer_);
    pipe_->InitBuffer(outScaleBackQueue_, useQueNum, scaleBackBuffer_);

    pipe_->InitBuffer(bufE8_, bufE8Buffer_);
    pipe_->InitBuffer(buf1E8_, buf1E8Buffer_);
    pipe_->InitBuffer(buf1E16_, buf1E16Buffer_);
    pipe_->InitBuffer(bufMaxPer4_, maxPer4Buffer_);
    pipe_->InitBuffer(bufMaxPer8_, maxPer8Buffer_);
    pipe_->InitBuffer(bufTmpScale_, tmpScaleBuffer_);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantNotTailAxis<ST, DT, useQueNum>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    for (int64_t i = 0; i < loopPerCore_; i++) {
        // 由本次ub计算的block块数推导列数
        int64_t calcCol = ((blockOffset_ + i) % tilingData_->dimNeg1BlockNum == tilingData_->dimNeg1BlockNum - 1) ?
                              ubRowLenTail_ :
                              ubRowLen_;
        // 本次ub计算的行数
        int64_t calcRow = ((blockOffset_ + i) / tilingData_->dimNeg1BlockNum % tilingData_->dimNeg2SplitBlockNum) ==
                                  (tilingData_->dimNeg2SplitBlockNum - 1) ?
                              ubRowCountTail_ :
                              ubRowCount_;
        // 单batch偏移+单行偏移+单列偏移
        int64_t xUbOffset = (blockOffset_ + i) / blockCountPerPage_ * tilingData_->dimNeg1 * tilingData_->dimNeg2 +
                            (blockOffset_ + i) % blockCountPerPage_ / tilingData_->dimNeg1BlockNum * ubRowCount_ *
                                tilingData_->dimNeg1 +
                            (blockOffset_ + i) % blockCountPerPage_ % tilingData_->dimNeg1BlockNum * ubRowLen_;
        // -2轴偏移
        int64_t scale2Offset = (blockOffset_ + i) / blockCountPerPage_ * dimNeg2ScaleNum_ * tilingData_->dimNeg1 * 2 +
                               (blockOffset_ + i) % blockCountPerPage_ / tilingData_->dimNeg1BlockNum *
                                   tilingData_->splitBlockH / blockSize_ * tilingData_->dimNeg1 * 2 +
                               (blockOffset_ + i) % blockCountPerPage_ % tilingData_->dimNeg1BlockNum * ubRowLen_;

        int64_t innerLoopMaxper4 = ((blockOffset_ + i) / tilingData_->dimNeg1BlockNum %
                                    tilingData_->dimNeg2SplitBlockNum) == (tilingData_->dimNeg2SplitBlockNum - 1) ?
                                       ubRowCountTail_ / 4 :
                                       ubRowCount_ / 4;
        int64_t innerLoopMaxper8 = ((blockOffset_ + i) / tilingData_->dimNeg1BlockNum %
                                    tilingData_->dimNeg2SplitBlockNum) == (tilingData_->dimNeg2SplitBlockNum - 1) ?
                                       ubRowCountTail_ / 8 :
                                       ubRowCount_ / 8;
        int64_t regLoop = ((blockOffset_ + i) / tilingData_->dimNeg1BlockNum % tilingData_->dimNeg2SplitBlockNum) ==
                                  (tilingData_->dimNeg2SplitBlockNum - 1) ?
                              ubRowCountTail_ / blockSize_ :
                              ubRowCount_ / blockSize_;
        ProcessOneLoop(calcCol, calcRow, xUbOffset, scale2Offset, innerLoopMaxper4, innerLoopMaxper8, regLoop);
    }
}
} // namespace DynamicHiF4Quant
#endif // DYNAMIC_HIF4_QUANT_NOT_TAIL_AXIS_HIFLOAT4_H
