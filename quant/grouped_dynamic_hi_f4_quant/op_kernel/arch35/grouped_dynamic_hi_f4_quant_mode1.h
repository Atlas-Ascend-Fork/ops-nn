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
 * \file grouped_dynamic_hi_f4_quant_model1.h
 * \brief
 */

#ifndef OPS_NN_GROUPED_DYNAMIC_HIF4_QUANT_MODE_1_H
#define OPS_NN_GROUPED_DYNAMIC_HIF4_QUANT_MODE_1_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "grouped_dynamic_hi_f4_quant_tilingdata.h"
#include "grouped_dynamic_hi_f4_quant_base.h"

namespace GroupedDynamicHiF4Quant {
using namespace AscendC;

template <typename xDtype, typename yDtype>
class GroupedDynamicHiF4QuantMode1 : public GroupedDynamicHiF4QuantBase<xDtype, yDtype> {
public:
    __aicore__ inline GroupedDynamicHiF4QuantMode1(const GroupedDynamicHiF4QuantTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitData();
    __aicore__ inline void SplitGroup(int64_t groupColSize, int64_t groupRowSize, int64_t groupCoreNum);
    __aicore__ inline void CopyIn(int64_t blockCount, int64_t blockLen, int64_t xGmOffset);
    __aicore__ inline void Compute(int64_t blockCount, int64_t blockLen);
    __aicore__ inline void CopyOut(int64_t blockCount, int64_t blockLen, int64_t xGmOffset, int64_t scaleGmOffset);
    const GroupedDynamicHiF4QuantTilingData* tilingData_{};
    TPipe* pipe_;

    TQue<QuePosition::VECIN, DB_BUFFER> xInQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> yOutQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> scaleOutFrontQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> scaleOutBackQueue_;

    TBuf<TPosition::VECCALC> tmpScaleBuf_;
    TBuf<TPosition::VECCALC> maxPer4Buf_;
    TBuf<TPosition::VECCALC> maxPer8Buf_;
    TBuf<TPosition::VECCALC> eaBuf_;
    TBuf<TPosition::VECCALC> ebBuf_;
    TBuf<TPosition::VECCALC> ecBuf_;

    GlobalTensor<xDtype> xGm_;
    GlobalTensor<int32_t> groupIndexGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint16_t> scaleGm_;

private:
    // 当前计算核索引
    int64_t coreIdx_ = 0;
    // 当前计算的group的索引
    int64_t curGroupIdx_ = 0;
    // 当前计算的group所需的核数
    int64_t curGroupCoreNum_ = 0;
    // 当前核是计算当前group的第几个，用于计算地址偏移
    int64_t curGroupCoreIdx_ = 0;
    // 当前group-2轴方向所切块数
    int64_t groupBlockColNum_ = 0;
    // 当前group-2轴方向所切块大小
    int64_t groupBlockColSize_ = 0;
    int64_t loopPerCore_ = 0;
    int64_t blockOffset_ = 0;
};
template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantMode1<xDtype, yDtype>::Init(GM_ADDR x, GM_ADDR group_index, GM_ADDR y,
                                                                          GM_ADDR scale)
{
#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    coreIdx_ = GetBlockIdx();
    if (coreIdx_ > tilingData_->usedCoreNum) {
        return;
    }

    xGm_.SetGlobalBuffer((__gm__ xDtype*)(x));
    groupIndexGm_.SetGlobalBuffer((__gm__ int32_t*)(group_index));
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y));
    scaleGm_.SetGlobalBuffer((__gm__ uint16_t*)(scale));
    InitData();

    int64_t bufferSize = this->blockRowSize_ * this->blockColSize_;
    int64_t xBufferSize = bufferSize * BYTE_OF_INPUT_X;
    int64_t yBufferSize = bufferSize / Y_NUM_OF_ONE_BYTE;
    int64_t scaleFrontBufferSize = this->ubMaxBlockNum_ * this->blockRowSize_ * BYTE_OF_INPUT_X;
    int64_t scaleBackBufferSize = scaleFrontBufferSize;
    int64_t tmpScaleBufferSize = bufferSize / NUM_FOUR * BYTE_OF_INPUT_X;
    int64_t maxPer4BufSize = tmpScaleBufferSize;
    int64_t maxPer8BufSize = bufferSize / NUM_EIGHT * BYTE_OF_INPUT_X;
    int64_t eaBufSize = this->ubMaxBlockNum_ * this->blockRowSize_ * BYTE_OF_INPUT_X;
    int64_t ebBufSize = eaBufSize;
    int64_t ecBufSize = eaBufSize * NUM_TWO;

    pipe_->InitBuffer(xInQueue_, DB_BUFFER, xBufferSize);
    pipe_->InitBuffer(yOutQueue_, DB_BUFFER, yBufferSize);
    pipe_->InitBuffer(scaleOutFrontQueue_, DB_BUFFER, scaleFrontBufferSize);
    pipe_->InitBuffer(scaleOutBackQueue_, DB_BUFFER, scaleBackBufferSize);
    pipe_->InitBuffer(tmpScaleBuf_, tmpScaleBufferSize);
    pipe_->InitBuffer(maxPer4Buf_, maxPer4BufSize);
    pipe_->InitBuffer(maxPer8Buf_, maxPer8BufSize);
    pipe_->InitBuffer(eaBuf_, eaBufSize);
    pipe_->InitBuffer(ebBuf_, ebBufSize);
    pipe_->InitBuffer(ecBuf_, ecBufSize);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantMode1<xDtype, yDtype>::InitData()
{
    this->BaseInitData(tilingData_);
    // 计算头group的核的数量
    int64_t headCoreNum = tilingData_->headCoreNum;
    // 头group数量
    int64_t headGroupNum = tilingData_->headGroupNum;
    // 头group需要核数
    int64_t headGroupCoreNum = tilingData_->headGroupCoreNum;
    // 尾group需要核数
    int64_t tailGroupCoreNum = tilingData_->tailGroupCoreNum;
    if (coreIdx_ < headCoreNum) {
        curGroupIdx_ = coreIdx_ / headGroupCoreNum;
        curGroupCoreNum_ = headGroupCoreNum;
        curGroupCoreIdx_ = coreIdx_ - curGroupIdx_ * headGroupCoreNum;
    } else {
        curGroupIdx_ = headGroupNum + (coreIdx_ - headCoreNum) / tailGroupCoreNum;
        curGroupCoreNum_ = tailGroupCoreNum;
        curGroupCoreIdx_ = coreIdx_ - headCoreNum - (curGroupIdx_ - headGroupNum) * tailGroupCoreNum;
    }
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantMode1<xDtype, yDtype>::Process()
{
    int64_t curGroupStart = 0;
    if (curGroupIdx_ > 0) {
        curGroupStart = groupIndexGm_.GetValue(curGroupIdx_ - 1);
    }
    int64_t curGroupEnd = groupIndexGm_.GetValue(curGroupIdx_);
    assert((0 <= curGroupEnd && curGroupEnd <= this->colSize_), "groupIndex %lld = %lld out of range[0 %lld]!\n",
           curGroupIdx_, curGroupEnd, this->colSize_);
    assert((curGroupStart <= curGroupEnd), "groupIndex %lld = %lld less than previous curGroupIdx = %lld!\n",
           curGroupIdx_, curGroupEnd, curGroupStart);
    int64_t curGroupSize = curGroupEnd - curGroupStart;
    // 当前group使用curGroupCoreNum_个核进行计算，对group进行切分
    SplitGroup(curGroupSize, this->rowSize_, curGroupCoreNum_);
    int64_t xGmOffset = curGroupStart * this->rowSize_;
    int64_t scaleGmOffset = (curGroupStart / BLOCK_SIZE + curGroupIdx_) * this->rowSize_ * NUM_TWO;
    for (int64_t loop1Num = 0; loop1Num < loopPerCore_; loop1Num++) {
        // 当前计算的block的索引
        int64_t blockRowIdx = blockOffset_ % this->blockRowNum_;
        int64_t blockColIdx = blockOffset_ / this->blockRowNum_;
        // 当前计算block的大小
        int64_t curBlockRowSize = (blockRowIdx == this->blockRowNum_ - 1) ? this->tailBlockRowSize_ :
                                                                            this->blockRowSize_;
        int64_t curBlockColSize = (blockColIdx == groupBlockColNum_ - 1) ?
                                      (curGroupSize - (groupBlockColNum_ - 1) * groupBlockColSize_) :
                                      groupBlockColSize_;
        // 当前计算block的偏移
        int64_t xOffset = xGmOffset + blockColIdx * groupBlockColSize_ * this->rowSize_ +
                          blockRowIdx * this->blockRowSize_;
        int64_t scaleOffset = scaleGmOffset + blockColIdx * NUM_TWO * this->rowSize_ * NUM_TWO +
                              blockRowIdx * this->blockRowSize_;
        CopyIn(curBlockColSize, curBlockRowSize, xOffset);
        Compute(curBlockColSize, curBlockRowSize);
        CopyOut(curBlockColSize, curBlockRowSize, xOffset, scaleOffset);
        blockOffset_++;
    }
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantMode1<xDtype, yDtype>::SplitGroup(int64_t groupColSize,
                                                                                int64_t groupRowSize,
                                                                                int64_t groupCoreNum)
{
    // 对当前group进行切块，切块大小暂定为ub能装下最大块的一半（或者采用动态调整的方式，待实现）
    groupBlockColSize_ = BLOCK_SIZE * NUM_TWO;
    groupBlockColNum_ = ops::CeilDiv(groupColSize, groupBlockColSize_);
    int64_t groupBlockNum = groupBlockColNum_ * this->blockRowNum_;
    // 采用均匀分的方式，保证核之间负载均衡
    int64_t groupHeadBlockNum = groupBlockNum % groupCoreNum;
    int64_t groupHeadBlockLoop = ops::CeilDiv(groupBlockNum, groupCoreNum);
    int64_t groupTailBlockLoop = groupBlockNum / groupCoreNum;
    if (curGroupCoreIdx_ < groupHeadBlockNum) {
        loopPerCore_ = groupHeadBlockLoop;
        blockOffset_ = curGroupCoreIdx_ * groupHeadBlockLoop;
    } else {
        loopPerCore_ = groupTailBlockLoop;
        blockOffset_ = groupHeadBlockNum * groupHeadBlockLoop +
                       (curGroupCoreIdx_ - groupHeadBlockNum) * groupTailBlockLoop;
    }
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantMode1<xDtype, yDtype>::CopyIn(int64_t blockCount, int64_t blockLen,
                                                                            int64_t xGmOffset)
{
    DataCopyExtParams copyInParams = {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(blockLen * BYTE_OF_INPUT_X),
        static_cast<uint32_t>((this->rowSize_ - blockLen) * BYTE_OF_INPUT_X),
        static_cast<uint32_t>((this->blockRowSize_ - blockLen) * BYTE_OF_INPUT_X / UB_BLOCK_SIZE),
        static_cast<uint32_t>(0)};
    DataCopyPadExtParams<xDtype> padParams{false, 0, 0, 0};

    LocalTensor<xDtype> xLocal = xInQueue_.template AllocTensor<xDtype>();
    DataCopyPad(xLocal, xGm_[xGmOffset], copyInParams, padParams);
    xInQueue_.template EnQue(xLocal);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantMode1<xDtype, yDtype>::CopyOut(int64_t blockCount, int64_t blockLen,
                                                                             int64_t xGmOffset, int64_t scaleGmOffset)
{
    int64_t ySrcStride = (this->blockRowSize_ - blockLen) / NUM_TWO / UB_BLOCK_SIZE;
    int64_t yDstStride = (this->rowSize_ - blockLen) / NUM_TWO;
    int64_t scaleBlockCount = ops::CeilDiv(blockCount, BLOCK_SIZE);
    int64_t scaleSrcStride = (this->blockRowSize_ - blockLen) * NUM_TWO / UB_BLOCK_SIZE;
    int64_t scaleDstStride = (NUM_TWO * this->rowSize_ - blockLen) * NUM_TWO;

    DataCopyExtParams yCopyOutParams = {static_cast<uint16_t>(blockCount), static_cast<uint32_t>(blockLen / NUM_TWO),
                                        static_cast<uint32_t>(ySrcStride), static_cast<uint32_t>(yDstStride),
                                        static_cast<uint32_t>(0)};
    DataCopyExtParams scaleCopyOutParams = {
        static_cast<uint16_t>(scaleBlockCount), static_cast<uint32_t>(blockLen * NUM_TWO),
        static_cast<uint32_t>(scaleSrcStride), static_cast<uint32_t>(scaleDstStride), static_cast<uint32_t>(0)};

    auto yLocal = yOutQueue_.template DeQue<uint8_t>();
    DataCopyPad(yGm_[xGmOffset / 2], yLocal, yCopyOutParams);
    yOutQueue_.template FreeTensor(yLocal);

    auto scaleFrontLocal = scaleOutFrontQueue_.template DeQue<uint16_t>();
    DataCopyPad(scaleGm_[scaleGmOffset], scaleFrontLocal, scaleCopyOutParams);
    scaleOutFrontQueue_.template FreeTensor(scaleFrontLocal);

    auto scaleBackLocal = scaleOutBackQueue_.template DeQue<uint16_t>();
    DataCopyPad(scaleGm_[scaleGmOffset + this->rowSize_], scaleBackLocal, scaleCopyOutParams);
    scaleOutBackQueue_.template FreeTensor(scaleBackLocal);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantMode1<xDtype, yDtype>::Compute(int64_t blockCount, int64_t blockLen)
{
    LocalTensor<xDtype> x = xInQueue_.template DeQue<xDtype>();
    LocalTensor<uint8_t> y = yOutQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint16_t> scaleFront = scaleOutFrontQueue_.template AllocTensor<uint16_t>();
    LocalTensor<uint16_t> scaleBack = scaleOutBackQueue_.template AllocTensor<uint16_t>();

    LocalTensor<uint16_t> ea = eaBuf_.Get<uint16_t>();
    LocalTensor<uint8_t> eb = ebBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> ec = ecBuf_.Get<uint8_t>();
    LocalTensor<uint16_t> tmpScale = tmpScaleBuf_.Get<uint16_t>();
    LocalTensor<uint16_t> maxPer4 = maxPer4Buf_.Get<uint16_t>();
    LocalTensor<uint16_t> maxPer8 = maxPer8Buf_.Get<uint16_t>();

    auto xAddr = (__ubuf__ xDtype*)x.GetPhyAddr();
    auto yAddr = (__ubuf__ uint32_t*)y.GetPhyAddr();
    auto scaleFrontAddr = (__ubuf__ uint16_t*)scaleFront.GetPhyAddr();
    auto scaleBackAddr = (__ubuf__ uint16_t*)scaleBack.GetPhyAddr();
    auto eaAddr8 = (__ubuf__ uint8_t*)ea.GetPhyAddr();
    auto eaAddr16 = (__ubuf__ uint16_t*)ea.GetPhyAddr();
    auto ebAddr = (__ubuf__ uint8_t*)eb.GetPhyAddr();
    auto ecAddr = (__ubuf__ uint8_t*)ec.GetPhyAddr();
    auto tmpScaleAddr = (__ubuf__ uint16_t*)tmpScale.GetPhyAddr();
    auto maxPer4Addr = (__ubuf__ uint16_t*)maxPer4.GetPhyAddr();
    auto maxPer8Addr = (__ubuf__ uint16_t*)maxPer8.GetPhyAddr();

    int64_t maxPer4Num = ops::CeilDiv(blockCount, NUM_FOUR);
    uint16_t maxPer4LoopNum = static_cast<uint16_t>(maxPer4Num - 1);
    uint16_t tailMaxper4 = static_cast<uint16_t>(blockCount - maxPer4LoopNum * NUM_FOUR);

    int64_t maxPer8Num = ops::CeilDiv(maxPer4Num, NUM_TWO);
    uint16_t maxPer8LoopNum = static_cast<uint16_t>(maxPer8Num - 1);
    uint16_t tailMaxper8 = static_cast<uint16_t>(maxPer4Num - maxPer8LoopNum * NUM_TWO);

    int64_t maxPer64Num = ops::CeilDiv(maxPer8Num, NUM_EIGHT);
    uint16_t maxPer64LoopNum = static_cast<uint16_t>(maxPer64Num - 1);
    uint16_t tail8Maxper64 = static_cast<uint16_t>(maxPer8Num - maxPer64LoopNum * NUM_EIGHT);
    uint16_t tail4Maxper64 = static_cast<uint16_t>(maxPer4Num - maxPer64LoopNum * NUM_SIXTENN);

    this->ComputeMaxPer4(maxPer4LoopNum, tailMaxper4, xAddr, maxPer4Addr);
    this->ComputeMaxPer8(maxPer8LoopNum, tailMaxper8, maxPer4Addr, maxPer8Addr);
    this->ComputeE6M2(maxPer64LoopNum, tail8Maxper64, maxPer8Addr, eaAddr16);
    this->ComputeTmpScale(maxPer64LoopNum, tail8Maxper64 - 1, tailMaxper8, maxPer4Addr, maxPer8Addr, eaAddr16, ebAddr,
                          ecAddr, tmpScaleAddr);
    this->ComputeScale(maxPer64LoopNum, tail8Maxper64, tail4Maxper64, eaAddr8, ebAddr, ecAddr, scaleFrontAddr,
                       scaleBackAddr);
    this->ComputeOut(maxPer4LoopNum, tailMaxper4, xAddr, tmpScaleAddr, yAddr);

    xInQueue_.template FreeTensor(x);
    yOutQueue_.template EnQue(y);
    scaleOutFrontQueue_.template EnQue(scaleFront);
    scaleOutBackQueue_.template EnQue(scaleBack);
}

} // namespace GroupedDynamicHiF4Quant
#endif // OPS_NN_GROUPED_DYNAMIC_HIF4_QUANT_MODE_1_H
