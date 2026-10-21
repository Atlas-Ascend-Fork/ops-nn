/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file anti_hi_f4_quant_not_tail.h
 * \brief
 */

#ifndef OPS_NN_ANTI_HI_F4_QUANT_NOT_TAIL_H
#define OPS_NN_ANTI_HI_F4_QUANT_NOT_TAIL_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "anti_hi_f4_quant_tilingdata.h"
#include "anti_hi_f4_quant_common.h"

namespace AntiHiF4Quant {
using namespace AscendC;

template <typename xDtype, typename yDtype>
class AntiHiF4QuantNotTail {
public:
    __aicore__ inline AntiHiF4QuantNotTail(const AntiHiF4QuantNotTailTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitData();
    __aicore__ inline void CopyIn(int64_t blockCount, int64_t blockLen, int64_t xGmOffset, int64_t scaleGmOffset);
    __aicore__ inline void Compute(int64_t blockCount, int64_t blockLen);
    __aicore__ inline void CopyOut(int64_t blockCount, int64_t blockLen, int64_t xGmOffset);
    __aicore__ inline void ComputeData(uint16_t loopNum, __ubuf__ fp4x2_e1m2_t* xAddr, __ubuf__ yDtype* yAddr,
                                       __ubuf__ uint16_t* scaleFrontAddr, __ubuf__ uint16_t* scaleBackAddr);
    const AntiHiF4QuantNotTailTilingData* tilingData_{};
    TPipe* pipe_;

    TQue<QuePosition::VECIN, DB_BUFFER> xInQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> scaleFrontOutQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> scaleBackOutQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> yOutQueue_;

    GlobalTensor<uint8_t> xGm_;
    GlobalTensor<yDtype> yGm_;
    GlobalTensor<uint16_t> scaleGm_;

private:
    int64_t coreIdx_ = 0;
    int64_t loopPerCore_ = 0;
    int64_t blockOffset_ = 0;
    int64_t xDimNeg1_ = 0;
    int64_t xDimNeg2_ = 0;
    int64_t xDimNeg3_ = 0;
    int64_t splitBlockColSize_ = 0;
    int64_t splitBlockRowSize_ = 0;
    int64_t tailBlockRowSize_ = 0;
    int64_t tailBlockColSize_ = 0;
    int64_t splitBlockColNum_ = 0;
    int64_t splitBlockRowNum_ = 0;
    int64_t ubBlockNum_ = 0;
    int64_t taskNumPerBatch_ = 0;
};
template <typename xDtype, typename yDtype>
__aicore__ inline void AntiHiF4QuantNotTail<xDtype, yDtype>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR y)
{
#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    coreIdx_ = GetBlockIdx();
    if (coreIdx_ > tilingData_->usedCoreNum) {
        return;
    }

    xGm_.SetGlobalBuffer((__gm__ uint8_t*)(x));
    yGm_.SetGlobalBuffer((__gm__ yDtype*)(y));
    scaleGm_.SetGlobalBuffer((__gm__ uint16_t*)(scale));
    InitData();

    int64_t bufferSize = splitBlockRowSize_ * splitBlockColSize_;
    int64_t xBufferSize = bufferSize / X_NUM_OF_ONE_BYTE;
    int64_t yBufferSize = bufferSize * BYTE_OF_OUTPUT_Y;
    int64_t scaleFrontBufferSize = ubBlockNum_ * splitBlockRowSize_ * NUM_TWO;
    int64_t scaleBackBufferSize = scaleFrontBufferSize;

    pipe_->InitBuffer(xInQueue_, DB_BUFFER, xBufferSize);
    pipe_->InitBuffer(yOutQueue_, DB_BUFFER, yBufferSize);
    pipe_->InitBuffer(scaleFrontOutQueue_, DB_BUFFER, scaleFrontBufferSize);
    pipe_->InitBuffer(scaleBackOutQueue_, DB_BUFFER, scaleBackBufferSize);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void AntiHiF4QuantNotTail<xDtype, yDtype>::InitData()
{
    int64_t headCoreNum = tilingData_->headCoreNum;
    if (coreIdx_ < headCoreNum) {
        loopPerCore_ = tilingData_->headCoreLoopNum;
        blockOffset_ = coreIdx_ * loopPerCore_;
    } else {
        loopPerCore_ = tilingData_->tailCoreLoopNum;
        blockOffset_ = headCoreNum * tilingData_->headCoreLoopNum + (coreIdx_ - headCoreNum) * loopPerCore_;
    }
    xDimNeg1_ = tilingData_->xDimNeg1;
    xDimNeg2_ = tilingData_->xDimNeg2;
    xDimNeg3_ = tilingData_->xDimNeg3;
    splitBlockColSize_ = tilingData_->splitBlockColSize;
    splitBlockColNum_ = tilingData_->splitBlockColNum;
    splitBlockRowSize_ = tilingData_->splitBlockRowSize;
    splitBlockRowNum_ = tilingData_->splitBlockRowNum;
    tailBlockRowSize_ = tilingData_->tailBlockRowSize;
    tailBlockColSize_ = tilingData_->tailBlockColSize;
    ubBlockNum_ = tilingData_->ubBlockNum;
    taskNumPerBatch_ = tilingData_->taskNumPerBatch;
}

template <typename xDtype, typename yDtype>
__aicore__ inline void AntiHiF4QuantNotTail<xDtype, yDtype>::Process()
{
    int64_t curBlockOffset = blockOffset_;
    for (int64_t i = 0; i < loopPerCore_; i++) {
        int64_t splitBlockBatchIdx = curBlockOffset / taskNumPerBatch_;
        int64_t splitBlockColIdx = curBlockOffset % taskNumPerBatch_ / splitBlockRowNum_;
        int64_t splitBlockRowIdx = curBlockOffset % taskNumPerBatch_ % splitBlockRowNum_;

        int64_t xGmOffset = (splitBlockBatchIdx * xDimNeg2_ + splitBlockColIdx * splitBlockColSize_) * xDimNeg1_ +
                            splitBlockRowIdx * splitBlockRowSize_;
        int64_t scaleGmOffset = (splitBlockBatchIdx * xDimNeg2_ / BLOCK_SIZE + splitBlockColIdx * ubBlockNum_) *
                                    xDimNeg1_ * NUM_TWO +
                                splitBlockRowIdx * splitBlockRowSize_;

        int64_t curBlockRowSize = (splitBlockRowIdx == splitBlockRowNum_ - 1) ? tailBlockRowSize_ : splitBlockRowSize_;
        int64_t curBlockColSize = (splitBlockColIdx == splitBlockColNum_ - 1) ? tailBlockColSize_ : splitBlockColSize_;

        CopyIn(curBlockColSize, curBlockRowSize, xGmOffset, scaleGmOffset);
        Compute(curBlockColSize, curBlockRowSize);
        CopyOut(curBlockColSize, curBlockRowSize, xGmOffset);

        curBlockOffset++;
    }
}

template <typename xDtype, typename yDtype>
__aicore__ inline void AntiHiF4QuantNotTail<xDtype, yDtype>::CopyIn(int64_t blockCount, int64_t blockLen,
                                                                    int64_t xGmOffset, int64_t scaleGmOffset)
{
    DataCopyExtParams copyInXParams = {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(blockLen / X_NUM_OF_ONE_BYTE),
        static_cast<uint32_t>((xDimNeg1_ - blockLen) / X_NUM_OF_ONE_BYTE),
        static_cast<uint32_t>((splitBlockRowSize_ - blockLen) / X_NUM_OF_ONE_BYTE / UB_BLOCK_SIZE),
        static_cast<uint32_t>(0)};
    DataCopyPadExtParams<uint8_t> xPadParams{false, 0, 0, 0};

    LocalTensor<uint8_t> xLocal = xInQueue_.template AllocTensor<uint8_t>();
    DataCopyPad(xLocal, xGm_[xGmOffset / 2], copyInXParams, xPadParams);
    xInQueue_.template EnQue(xLocal);

    DataCopyExtParams copyInScaleParams = {
        static_cast<uint16_t>(blockCount / BLOCK_SIZE), static_cast<uint32_t>(blockLen * NUM_TWO),
        static_cast<uint32_t>((NUM_TWO * xDimNeg1_ - blockLen) * NUM_TWO),
        static_cast<uint32_t>((splitBlockRowSize_ - blockLen) * NUM_TWO / UB_BLOCK_SIZE), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<uint16_t> scalePadParams{false, 0, 0, 0};

    LocalTensor<uint16_t> scaleFrontLocal = scaleFrontOutQueue_.template AllocTensor<uint16_t>();
    DataCopyPad(scaleFrontLocal, scaleGm_[scaleGmOffset], copyInScaleParams, scalePadParams);
    scaleFrontOutQueue_.template EnQue(scaleFrontLocal);

    LocalTensor<uint16_t> scaleBackLocal = scaleBackOutQueue_.template AllocTensor<uint16_t>();
    DataCopyPad(scaleBackLocal, scaleGm_[scaleGmOffset + xDimNeg1_], copyInScaleParams, scalePadParams);
    scaleBackOutQueue_.template EnQue(scaleBackLocal);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void AntiHiF4QuantNotTail<xDtype, yDtype>::CopyOut(int64_t blockCount, int64_t blockLen,
                                                                     int64_t xGmOffset)
{
    int64_t ySrcStride = (splitBlockRowSize_ - blockLen) * BYTE_OF_OUTPUT_Y / UB_BLOCK_SIZE;
    int64_t yDstStride = (xDimNeg1_ - blockLen) * BYTE_OF_OUTPUT_Y;

    DataCopyExtParams yCopyOutParams = {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(blockLen * BYTE_OF_OUTPUT_Y),
        static_cast<uint32_t>(ySrcStride), static_cast<uint32_t>(yDstStride), static_cast<uint32_t>(0)};

    auto yLocal = yOutQueue_.template DeQue<yDtype>();
    DataCopyPad(yGm_[xGmOffset], yLocal, yCopyOutParams);
    yOutQueue_.template FreeTensor(yLocal);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void AntiHiF4QuantNotTail<xDtype, yDtype>::Compute(int64_t blockCount, int64_t blockLen)
{
    LocalTensor<uint8_t> x = xInQueue_.template DeQue<uint8_t>();
    LocalTensor<yDtype> y = yOutQueue_.template AllocTensor<yDtype>();
    LocalTensor<uint16_t> scaleFront = scaleFrontOutQueue_.template DeQue<uint16_t>();
    LocalTensor<uint16_t> scaleBack = scaleBackOutQueue_.template DeQue<uint16_t>();

    auto xAddr = (__ubuf__ fp4x2_e1m2_t*)x.GetPhyAddr();
    auto yAddr = (__ubuf__ yDtype*)y.GetPhyAddr();
    auto scaleFrontAddr = (__ubuf__ uint16_t*)scaleFront.GetPhyAddr();
    auto scaleBackAddr = (__ubuf__ uint16_t*)scaleBack.GetPhyAddr();

    uint16_t loopNum = static_cast<uint16_t>(blockCount / BLOCK_SIZE);

    ComputeData(loopNum, xAddr, yAddr, scaleFrontAddr, scaleBackAddr);

    xInQueue_.template FreeTensor(x);
    scaleFrontOutQueue_.template FreeTensor(scaleFront);
    scaleBackOutQueue_.template FreeTensor(scaleBack);
    yOutQueue_.template EnQue(y);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void AntiHiF4QuantNotTail<xDtype, yDtype>::ComputeData(uint16_t loopNum, __ubuf__ fp4x2_e1m2_t* xAddr,
                                                                         __ubuf__ yDtype* yAddr,
                                                                         __ubuf__ uint16_t* scaleFrontAddr,
                                                                         __ubuf__ uint16_t* scaleBackAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> scaleFront;
        MicroAPI::RegTensor<bfloat16_t> EaReg;
        MicroAPI::RegTensor<uint16_t> EbReg;
        MicroAPI::RegTensor<uint16_t> EcReg;
        MicroAPI::RegTensor<uint16_t> EbBitReg;
        MicroAPI::RegTensor<uint16_t> EcBitReg;
        MicroAPI::RegTensor<uint16_t> powReg;
        MicroAPI::RegTensor<bfloat16_t> scaleValueReg;
        MicroAPI::RegTensor<fp4x2_e1m2_t> xReg;
        MicroAPI::RegTensor<bfloat16_t> xBf16Reg;
        MicroAPI::RegTensor<bfloat16_t> yReg;

        MicroAPI::RegTensor<uint16_t> EaMaskReg;
        MicroAPI::RegTensor<uint16_t> tmpMaskReg;

        MicroAPI::Duplicate(EaMaskReg, 0x00ff);
        MicroAPI::Duplicate(tmpMaskReg, 0x0001);

        MicroAPI::MaskReg maskAll16 = MicroAPI::CreateMask<uint16_t>();
        MicroAPI::MaskReg maskAll8 = MicroAPI::CreateMask<uint8_t>();

        for (uint16_t loop1Idx = 0; loop1Idx < loopNum; loop1Idx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(scaleFront, scaleFrontAddr, 128);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(EcReg, scaleBackAddr, 128);

            MicroAPI::Cast<bfloat16_t, fp8_e6m2_t, castTraitE6M2ToBF16>(
                EaReg, (MicroAPI::RegTensor<fp8_e6m2_t>&)scaleFront, maskAll8);
            MicroAPI::ShiftRights(EbReg, scaleFront, static_cast<int16_t>(8), maskAll16);

            for (uint16_t loop2Idx = 0; loop2Idx < static_cast<uint16_t>(8); loop2Idx++) {
                MicroAPI::And(EbBitReg, EbReg, tmpMaskReg, maskAll16);
                for (uint16_t loop3Idx = 0; loop3Idx < static_cast<uint16_t>(2); loop3Idx++) {
                    MicroAPI::And(EcBitReg, EcReg, tmpMaskReg, maskAll16);
                    MicroAPI::Add(powReg, EbBitReg, EcBitReg, maskAll16);
                    MicroAPI::Adds(powReg, powReg, BF16_BIAS, maskAll16);
                    MicroAPI::ShiftLefts(powReg, powReg, BF16_SHAFT_LEFT_NUM, maskAll16);
                    MicroAPI::Mul(scaleValueReg, EaReg, (MicroAPI::RegTensor<bfloat16_t>&)(powReg), maskAll16);
                    for (uint16_t loop4Idx = 0; loop4Idx < static_cast<uint16_t>(4); loop4Idx++) {
                        MicroAPI::DataCopy<fp4x2_e1m2_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                           MicroAPI::LoadDist::DIST_UNPACK4_B8>(xReg, xAddr, 64);

                        MicroAPI::Cast<bfloat16_t, fp4x2_e1m2_t, castTraitB4ToB16>(xBf16Reg, xReg, maskAll16);
                        MicroAPI::Mul(yReg, xBf16Reg, scaleValueReg, maskAll16);
                        if constexpr (IsSameType<yDtype, half>::value) {
                            MicroAPI::Cast<yDtype, bfloat16_t, castTraitBF16ToFP16>((MicroAPI::RegTensor<yDtype>&)yReg,
                                                                                    yReg, maskAll16);
                        }

                        MicroAPI::DataCopy<yDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            yAddr, (MicroAPI::RegTensor<yDtype>&)yReg, 128, maskAll16);
                    }
                    MicroAPI::ShiftRights(EcReg, EcReg, SHIFT_RIGHT_NUM, maskAll16);
                }
                MicroAPI::ShiftRights(EbReg, EbReg, SHIFT_RIGHT_NUM, maskAll16);
            }
        }
    }
}

} // namespace AntiHiF4Quant
#endif // OPS_NN_ANTI_HI_F4_QUANT_NOT_TAIL_H
