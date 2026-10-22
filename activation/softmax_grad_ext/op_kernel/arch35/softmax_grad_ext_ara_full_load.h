/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_grad_ext_ara_full_load.h
 * \brief
 */

#ifndef SOFTMAX_GRAD_EXT_ARA_FULL_LOAD_H
#define SOFTMAX_GRAD_EXT_ARA_FULL_LOAD_H

#include "kernel_tiling/kernel_tiling.h"

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "softmax_grad_ext_base.h"

namespace SoftmaxGradExt {
using namespace AscendC;
using namespace AscendC::Reg;

using AscendC::Reg::LoadAlign;
using AscendC::Reg::LoadDist;
using AscendC::Reg::MaskMergeMode;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreAlign;
using AscendC::Reg::StoreDist;

constexpr int64_t SCALE_COEF_TWO = 2;
constexpr int64_t SCALE_COEF_FOUR = 4;
constexpr int64_t SCALE_COEF_EIGHT = 8;

constexpr int64_t BIN_ADD_PADDING = 1;

constexpr uint16_t ROW_ZERO = 0;
constexpr uint16_t ROW_ONE = 1;
constexpr uint16_t ROW_TWO = 2;
constexpr uint16_t ROW_THREE = 3;
constexpr uint16_t ROW_FOUR = 4;
constexpr uint16_t ROW_FIVE = 5;
constexpr uint16_t ROW_SIX = 6;
constexpr uint16_t ROW_SEVEN = 7;

constexpr uint32_t ROW_TWO_OFFSET = 2;
constexpr uint32_t ROW_THREE_OFFSET = 3;
constexpr uint32_t ROW_FOUR_OFFSET = 4;
constexpr uint32_t ROW_FIVE_OFFSET = 5;
constexpr uint32_t ROW_SIX_OFFSET = 6;
constexpr uint32_t ROW_SEVEN_OFFSET = 7;

// ARA 全载模板：[A1, R, A0] 布局，R 在行、A0 在列，单 tile 装下 [R, tileA0Len]
// 语义：y = (grad * x1 - x1 * sum_R(grad * x1)) * x2
template <typename T>
class SoftmaxGradExtARA {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();

public:
    __aicore__ inline SoftmaxGradExtARA(){};

    __aicore__ inline SoftmaxGradExtARA(const SoftmaxGradExtARATilingData* tilingDataIn) { tilingData_ = tilingDataIn; }

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, TPipe* pipeIn)
    {
        pipe_ = pipeIn;

        gradGm_.SetGlobalBuffer((__gm__ T*)grad);
        x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm_.SetGlobalBuffer((__gm__ T*)x2);
        yGm_.SetGlobalBuffer((__gm__ T*)y);

        int64_t tileShapeLen = tilingData_->tileA0Len * tilingData_->totalRLen;
        pipe_->InitBuffer(gradQueue_, BUFFER_NUM, tileShapeLen * sizeof(T));
        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, tileShapeLen * sizeof(T));
        pipe_->InitBuffer(x2Queue_, BUFFER_NUM, tileShapeLen * sizeof(T));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, tileShapeLen * sizeof(T));

        pipe_->InitBuffer(tmpBuf_, tilingData_->tileA0Len * (tilingData_->totalRLen + BIN_ADD_PADDING) * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, tilingData_->tileA0Len * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilingData_->tilesPerCore;
        int64_t endIdx = beginIdx + tilingData_->tilesPerCore;
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curA0Idx = curIdx % tilingData_->a0Outer;
            int64_t curA1Idx = curIdx / tilingData_->a0Outer;

            uint32_t curTileA0Len = curA0Idx == (tilingData_->a0Outer - 1) ? tilingData_->tileA0Tail :
                                                                             tilingData_->tileA0Len;

            int64_t xOffset =
                // a1 offset
                curA1Idx * tilingData_->totalRLen * tilingData_->totalA0Len +
                // a0 offset
                curA0Idx * tilingData_->tileA0Len;

            CopyIn(xOffset, tilingData_->totalRLen, curTileA0Len);
            Compute(tilingData_->totalRLen, curTileA0Len);
            CopyOutY(xOffset, tilingData_->totalRLen, curTileA0Len);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t xGmOffset, int64_t curTileRLen, uint32_t curTileA0Len)
    {
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileRLen;
        copyInParams.blockLen = curTileA0Len * sizeof(T);
        copyInParams.srcStride = (tilingData_->totalA0Len - curTileA0Len) * sizeof(T);
        copyInParams.dstStride = (tilingData_->tileA0Len - curTileA0Len) * sizeof(T) / BLOCK_SIZE;

        LocalTensor<T> gradLocal = gradQueue_.AllocTensor<T>();
        DataCopyPad<T, PaddingMode::Normal>(gradLocal, gradGm_[xGmOffset], copyInParams, dataCopyPadExtParams);
        gradQueue_.EnQue(gradLocal);

        LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        DataCopyPad<T, PaddingMode::Normal>(x1Local, x1Gm_[xGmOffset], copyInParams, dataCopyPadExtParams);
        x1Queue_.EnQue(x1Local);

        if (tilingData_->x2IsScalar == 1) {
            // x2 单点输入：搬 1 点后 Duplicate 广播到整 tile
            LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
            DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T)),
                                         static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
            DataCopyPadExtParams<T> padParams{true, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                              static_cast<T>(0.0)};
            DataCopyPad(x2Local, x2Gm_[0], copyParams, padParams);
            x2Queue_.EnQue(x2Local);
            x2Local = x2Queue_.DeQue<T>();
            Duplicate<T>(x2Local, x2Local, tilingData_->tileA0Len * curTileRLen);
            x2Queue_.EnQue(x2Local);
        } else {
            LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
            DataCopyPad<T, PaddingMode::Normal>(x2Local, x2Gm_[xGmOffset], copyInParams, dataCopyPadExtParams);
            x2Queue_.EnQue(x2Local);
        }
    }

    __aicore__ inline void Compute(int64_t curTileRLen, uint32_t curTileA0Len)
    {
        LocalTensor<T> grad = gradQueue_.DeQue<T>();
        LocalTensor<T> x1 = x1Queue_.DeQue<T>();
        LocalTensor<T> x2 = x2Queue_.DeQue<T>();
        __ubuf__ T* gradLocal = (__ubuf__ T*)grad.GetPhyAddr();
        __ubuf__ T* x1Local = (__ubuf__ T*)x1.GetPhyAddr();
        __ubuf__ T* x2Local = (__ubuf__ T*)x2.GetPhyAddr();

        LocalTensor<float> tmpTensor = tmpBuf_.Get<float>();
        __ubuf__ float* tmpLocal = (__ubuf__ float*)tmpTensor.GetPhyAddr();

        uint16_t loopA0Num = ops::CeilDiv(curTileA0Len, VL_FP32);

        // Pass1: tmp = grad * x1 (fp32)
        VFMulVector(tmpLocal, gradLocal, x1Local, curTileRLen, curTileA0Len, loopA0Num);

        LocalTensor<T> y = yQueue_.AllocTensor<T>();
        __ubuf__ T* yLocal = (__ubuf__ T*)y.GetPhyAddr();

        LocalTensor<float> reduceTensor = reduceBuf_.Get<float>();
        __ubuf__ float* reduceLocal = (__ubuf__ float*)reduceTensor.GetPhyAddr();

        // Pass2: 跨 R 行折叠求 S（R > 8 时 y 缓冲复用作折叠 scratch）
        VFReduceSum(reduceLocal, tmpLocal, (__ubuf__ float*)yLocal, curTileRLen, curTileA0Len);

        // Pass3: y = (grad * x1 - x1 * S) * x2
        VFCalculateOutput(yLocal, gradLocal, x1Local, x2Local, reduceLocal, curTileRLen, curTileA0Len, loopA0Num);

        yQueue_.EnQue(y);
        gradQueue_.FreeTensor<T>(grad);
        x1Queue_.FreeTensor<T>(x1);
        x2Queue_.FreeTensor<T>(x2);
    }

    __aicore__ inline void VFMulVector(__ubuf__ float* tmpLocal, __ubuf__ T* gradLocal, __ubuf__ T* x1Local,
                                       uint16_t curTileRLen, uint32_t curTileA0Len, uint16_t loopA0Num)
    {
        uint32_t tileA0Len = tilingData_->tileA0Len;
        __VEC_SCOPE__
        {
            RegTensor<float> gradReg;
            RegTensor<float> x1Reg;
            RegTensor<float> mulReg;

            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;

            for (uint16_t k = 0; k < loopA0Num; k++) {
                pregMask = UpdateMask<float>(sreg);
                for (uint16_t i = 0; i < curTileRLen; i++) {
                    uint32_t offset = i * tileA0Len + k * VL_FP32;
                    LoadTensorForDtypeTIn(gradLocal, gradReg, pregMask, offset);
                    LoadTensorForDtypeTIn(x1Local, x1Reg, pregMask, offset);
                    Mul(mulReg, gradReg, x1Reg, pregMask);
                    StoreAlign(((__ubuf__ float*)tmpLocal) + offset, mulReg, pregMask);
                }
            }
        }
    }

    __aicore__ inline void VFReduceSum(__ubuf__ float* reduceLocal, __ubuf__ float* tmpLocal, __ubuf__ float* yInUb,
                                       uint16_t curTileRLen, uint16_t curTileA0Len)
    {
        if (tilingData_->totalRLen <= SCALE_COEF_TWO) {
            SumRLessThan2(tmpLocal, reduceLocal, curTileA0Len);
        } else if (tilingData_->totalRLen <= SCALE_COEF_FOUR) {
            SumRLessThan4(tmpLocal, reduceLocal, curTileA0Len);
        } else if (tilingData_->totalRLen <= SCALE_COEF_EIGHT) {
            SumRLessThan8(tmpLocal, reduceLocal, curTileA0Len);
        } else {
            SumRMoreThan8(tmpLocal, yInUb, reduceLocal, curTileA0Len);
        }
    }

    __aicore__ inline void SumRLessThan2(__ubuf__ float* tmpLocal, __ubuf__ float* reduceLocal, uint32_t curTileA0Len)
    {
        uint32_t rStride = tilingData_->tileA0Len;
        uint16_t rLoopCount = tilingData_->totalRLen;

        uint16_t aLoopCount = ops::CeilDiv(curTileA0Len, VL_FP32);

        __VEC_SCOPE__
        {
            RegTensor<float> xld;
            RegTensor<float> sum;
            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                Duplicate(sum, 0.0, pregLoop);
                for (uint16_t i = 0; i < rLoopCount; i++) {
                    LoadAlign(xld, ((__ubuf__ float*)tmpLocal + i * rStride + k * VL_FP32));
                    Add(sum, sum, xld, pregLoop);
                }
                StoreAlign(((__ubuf__ float*)reduceLocal + k * VL_FP32), sum, pregLoop);
            }
        }
    }

    __aicore__ inline void SumRLessThan4(__ubuf__ float* tmpLocal, __ubuf__ float* reduceLocal, uint32_t curTileA0Len)
    {
        uint32_t remainderOffset = SCALE_COEF_TWO * tilingData_->tileA0Len;
        uint32_t aLength = tilingData_->tileA0Len;
        uint32_t validNumInXUb = tilingData_->totalRLen * tilingData_->tileA0Len;

        uint16_t remainderTailCount = tilingData_->totalRLen - SCALE_COEF_TWO;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;

        uint16_t aLoopCount = ops::CeilDiv(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> zero;
            Duplicate(zero, 0.0, pregMain);

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                StoreAlign(((__ubuf__ float*)tmpLocal + validNumInXUb + aLoopOffset), zero, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                TwoRowAddWithTail(x1, tmpLocal, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset,
                                  aLength + aLoopOffset, remainderTailOffset1 + aLoopOffset, rem, nextRow, remNextRow);
                StoreAlign(((__ubuf__ float*)reduceLocal + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void TwoRowAddWithTail(RegTensor<float>& dst, __ubuf__ float* input, MaskReg& preg,
                                             uint32_t offset1, uint32_t offset2, uint32_t offset3, uint32_t offset4,
                                             RegTensor<float>& rem, RegTensor<float>& nextRow,
                                             RegTensor<float>& remNextRow)
    {
        LoadAlign(dst, ((__ubuf__ float*)(input) + (offset1)));
        LoadAlign(rem, ((__ubuf__ float*)(input) + (offset2)));
        Add(dst, dst, rem, preg);
        LoadAlign(nextRow, ((__ubuf__ float*)(input) + (offset3)));
        LoadAlign(remNextRow, ((__ubuf__ float*)(input) + (offset4)));
        Add(nextRow, nextRow, remNextRow, preg);
        Add(dst, dst, nextRow, preg);
    }

    __aicore__ inline void SumRLessThan8(__ubuf__ float* tmpLocal, __ubuf__ float* reduceLocal, uint32_t curTileA0Len)
    {
        uint32_t remainderOffset = SCALE_COEF_FOUR * tilingData_->tileA0Len;
        uint32_t aLength = tilingData_->tileA0Len;
        uint32_t validNumInXUb = tilingData_->totalRLen * tilingData_->tileA0Len;

        uint16_t remainderTailCount = tilingData_->totalRLen - SCALE_COEF_FOUR;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;
        uint32_t remainderTailOffset2 = (ROW_TWO > remainderTailCount) ? validNumInXUb :
                                                                         remainderOffset + ROW_TWO_OFFSET * aLength;
        uint32_t remainderTailOffset3 = (ROW_THREE > remainderTailCount) ? validNumInXUb :
                                                                           remainderOffset + ROW_THREE_OFFSET * aLength;

        uint16_t aLoopCount = ops::CeilDiv(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> zero;
            Duplicate(zero, 0.0, pregMain);

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                StoreAlign(((__ubuf__ float*)tmpLocal + validNumInXUb + aLoopOffset), zero, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                TwoRowAddWithTail(x1, tmpLocal, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset,
                                  aLength + aLoopOffset, remainderTailOffset1 + aLoopOffset, rem, nextRow, remNextRow);
                TwoRowAddWithTail(x2, tmpLocal, pregLoop, ROW_TWO_OFFSET * aLength + aLoopOffset,
                                  remainderTailOffset2 + aLoopOffset, ROW_THREE_OFFSET * aLength + aLoopOffset,
                                  remainderTailOffset3 + aLoopOffset, rem, nextRow, remNextRow);
                Add(x1, x1, x2, pregLoop);
                StoreAlign(((__ubuf__ float*)reduceLocal + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void SumRMoreThan8(__ubuf__ float* xInUb, __ubuf__ float* yInUb, __ubuf__ float* reduceLocal,
                                         uint32_t curTileA0Len)
    {
        uint16_t remainderLoopCount = tilingData_->remainderLoopCount;
        uint16_t remainderLoopCountTmp = remainderLoopCount - 1;
        uint16_t quotientLoopCount = tilingData_->quotientLoopCount;

        uint32_t remainderOffset = tilingData_->remainderOffset;
        uint32_t baseLineOffset = tilingData_->baseLineOffset;
        uint32_t aLength = tilingData_->tileA0Len;

        uint16_t binaryAddKLoop = tilingData_->binaryAddK;
        uint16_t binaryAddInnerLoop = tilingData_->binaryAddInnerLoop;
        uint16_t binaryAddLastLoop = tilingData_->binaryAddLast;

        uint32_t validNumInXUb = tilingData_->validNumInXUb;

        uint16_t remainderTailCount = tilingData_->remainderTailCount;
        uint32_t quotientTailOffset = tilingData_->quotientTailOffset;
        uint32_t remainderTailOffset = tilingData_->remainderTailOffset;
        uint32_t remainderTailOffset0 = tilingData_->remainderTailOffset0;
        uint32_t remainderTailOffset1 = tilingData_->remainderTailOffset1;
        uint32_t remainderTailOffset2 = tilingData_->remainderTailOffset2;
        uint32_t remainderTailOffset3 = tilingData_->remainderTailOffset3;
        uint32_t remainderTailOffset4 = tilingData_->remainderTailOffset4;
        uint32_t remainderTailOffset5 = tilingData_->remainderTailOffset5;
        uint32_t remainderTailOffset6 = tilingData_->remainderTailOffset6;
        uint32_t remainderTailOffset7 = tilingData_->remainderTailOffset7;

        uint16_t aLoopCount = ops::CeilDiv(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;

            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> zero;
            Duplicate(zero, 0.0, pregMain);

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                StoreAlign(((__ubuf__ float*)xInUb + validNumInXUb + aLoopOffset), zero, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 前半部分与后半部分中，都为8行的部分
                for (uint16_t i = 0; i < remainderLoopCountTmp; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    TwoRowAddWithTail(x1, xInUb, pregLoop, quotOffset, remOffset, quotOffset + aLength,
                                      remOffset + aLength, rem, nextRow, remNextRow);
                    TwoRowAddWithTail(x2, xInUb, pregLoop, quotOffset + ROW_TWO_OFFSET * aLength,
                                      remOffset + ROW_TWO_OFFSET * aLength, quotOffset + ROW_THREE_OFFSET * aLength,
                                      remOffset + ROW_THREE_OFFSET * aLength, rem, nextRow, remNextRow);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddWithTail(x3, xInUb, pregLoop, quotOffset + ROW_FOUR_OFFSET * aLength,
                                      remOffset + ROW_FOUR_OFFSET * aLength, quotOffset + ROW_FIVE_OFFSET * aLength,
                                      remOffset + ROW_FIVE_OFFSET * aLength, rem, nextRow, remNextRow);
                    TwoRowAddWithTail(x4, xInUb, pregLoop, quotOffset + ROW_SIX_OFFSET * aLength,
                                      remOffset + ROW_SIX_OFFSET * aLength, quotOffset + ROW_SEVEN_OFFSET * aLength,
                                      remOffset + ROW_SEVEN_OFFSET * aLength, rem, nextRow, remNextRow);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    StoreAlign(((__ubuf__ float*)yInUb + i * aLength + aLoopOffset), x1, pregLoop);
                }
                // 前半部分为8行，后半部分可能不足8行
                {
                    TwoRowAddWithTail(x1, xInUb, pregLoop, quotientTailOffset + aLoopOffset,
                                      remainderTailOffset0 + aLoopOffset, quotientTailOffset + aLength + aLoopOffset,
                                      remainderTailOffset1 + aLoopOffset, rem, nextRow, remNextRow);
                    TwoRowAddWithTail(x2, xInUb, pregLoop, quotientTailOffset + ROW_TWO_OFFSET * aLength + aLoopOffset,
                                      remainderTailOffset2 + aLoopOffset,
                                      quotientTailOffset + ROW_THREE_OFFSET * aLength + aLoopOffset,
                                      remainderTailOffset3 + aLoopOffset, rem, nextRow, remNextRow);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddWithTail(x3, xInUb, pregLoop, quotientTailOffset + ROW_FOUR_OFFSET * aLength + aLoopOffset,
                                      remainderTailOffset4 + aLoopOffset,
                                      quotientTailOffset + ROW_FIVE_OFFSET * aLength + aLoopOffset,
                                      remainderTailOffset5 + aLoopOffset, rem, nextRow, remNextRow);
                    TwoRowAddWithTail(x4, xInUb, pregLoop, quotientTailOffset + ROW_SIX_OFFSET * aLength + aLoopOffset,
                                      remainderTailOffset6 + aLoopOffset,
                                      quotientTailOffset + ROW_SEVEN_OFFSET * aLength + aLoopOffset,
                                      remainderTailOffset7 + aLoopOffset, rem, nextRow, remNextRow);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    StoreAlign(((__ubuf__ float*)yInUb + (remainderLoopCount - 1) * aLength + aLoopOffset), x1,
                               pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理8行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    TwoRowAdd(x1, xInUb, pregLoop, baseOffset, baseOffset + aLength, nextRow);
                    TwoRowAdd(x2, xInUb, pregLoop, baseOffset + ROW_TWO_OFFSET * aLength,
                              baseOffset + ROW_THREE_OFFSET * aLength, nextRow);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAdd(x3, xInUb, pregLoop, baseOffset + ROW_FOUR_OFFSET * aLength,
                              baseOffset + ROW_FIVE_OFFSET * aLength, nextRow);
                    TwoRowAdd(x4, xInUb, pregLoop, baseOffset + ROW_SIX_OFFSET * aLength,
                              baseOffset + ROW_SEVEN_OFFSET * aLength, nextRow);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    StoreAlign(((__ubuf__ float*)yInUb + (remainderLoopCount + i) * aLength + aLoopOffset), x1,
                               pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                BinaryAddVF((__ubuf__ float*)yInUb, aLength, aLoopOffset, binaryAddKLoop, binaryAddInnerLoop,
                            binaryAddLastLoop, pregLoop, x1, x2, x3, x4);
                LoadAlign(x1, ((__ubuf__ float*)yInUb + aLoopOffset));
                StoreAlign(((__ubuf__ float*)reduceLocal + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void TwoRowAdd(RegTensor<float>& dst, __ubuf__ float* input, MaskReg& preg, uint32_t offset1,
                                     uint32_t offset2, RegTensor<float>& nextRow)
    {
        LoadAlign(dst, ((__ubuf__ float*)(input) + (offset1)));
        LoadAlign(nextRow, ((__ubuf__ float*)(input) + (offset2)));
        Add(dst, dst, nextRow, preg);
    }

    __aicore__ inline void BinaryAddVF(__ubuf__ float* binaryAddTmpAddr, uint32_t rLoopStride, uint32_t offset,
                                       uint16_t binaryAddKLoop, uint16_t binaryAddInnerLoop, uint16_t binaryAddLastLoop,
                                       MaskReg& pregLoop, RegTensor<float>& x1, RegTensor<float>& x2,
                                       RegTensor<float>& x3, RegTensor<float>& x4)
    {
        uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
        for (uint16_t i = 0; i < binaryAddKLoop; i++) {
            curBinaryAddInnerLoop = curBinaryAddInnerLoop / ROW_FOUR_OFFSET;
            for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
                LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET) * rLoopStride + offset));
                LoadAlign(x2, ((__ubuf__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + 1) * rLoopStride + offset));
                Add(x1, x1, x2, pregLoop);
                LoadAlign(x3, ((__ubuf__ float*)binaryAddTmpAddr +
                               (j * ROW_FOUR_OFFSET + ROW_TWO_OFFSET) * rLoopStride + offset));
                LoadAlign(x4, ((__ubuf__ float*)binaryAddTmpAddr +
                               (j * ROW_FOUR_OFFSET + ROW_THREE_OFFSET) * rLoopStride + offset));
                Add(x3, x3, x4, pregLoop);
                Add(x1, x1, x3, pregLoop);
                StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + j * rLoopStride + offset), x1, pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        for (uint16_t i = 0; i < binaryAddLastLoop; i++) {
            LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + offset));
            LoadAlign(x2, ((__ubuf__ float*)binaryAddTmpAddr + rLoopStride + offset));
            Add(x1, x1, x2, pregLoop);
            StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + offset), x1, pregLoop);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }

    __aicore__ inline void VFCalculateOutput(__ubuf__ T* yLocal, __ubuf__ T* gradLocal, __ubuf__ T* x1Local,
                                             __ubuf__ T* x2Local, __ubuf__ float* reduceLocal, uint16_t curTileRLen,
                                             uint32_t curTileA0Len, uint16_t loopA0Num)
    {
        uint32_t tileA0Len = tilingData_->tileA0Len;
        __VEC_SCOPE__
        {
            RegTensor<float> gradReg;
            RegTensor<float> x1Reg;
            RegTensor<float> x2Reg;
            RegTensor<float> yReg;
            RegTensor<float> sumReg;

            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;

            for (uint16_t k = 0; k < loopA0Num; k++) {
                pregMask = UpdateMask<float>(sreg);
                LoadAlign<float, LoadDist::DIST_NORM>(sumReg, (__ubuf__ float*)reduceLocal + k * VL_FP32);

                for (uint16_t i = 0; i < curTileRLen; i++) {
                    uint32_t offset = i * tileA0Len + k * VL_FP32;

                    LoadTensorForDtypeTIn(gradLocal, gradReg, pregMask, offset);
                    LoadTensorForDtypeTIn(x1Local, x1Reg, pregMask, offset);
                    LoadTensorForDtypeTIn(x2Local, x2Reg, pregMask, offset);

                    // y = (grad * x1 - x1 * S) * x2
                    Mul(yReg, gradReg, x1Reg, pregMask);
                    Mul(x1Reg, x1Reg, sumReg, pregMask);
                    Sub(yReg, yReg, x1Reg, pregMask);
                    Mul(yReg, yReg, x2Reg, pregMask);

                    StoreTensorForDtypeTOut(yLocal, yReg, pregMask, offset);
                }
            }
        }
    }

    __aicore__ inline void LoadTensorForDtypeTIn(__ubuf__ T* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T, float>::value) {
            LoadAlign<float, LoadDist::DIST_NORM>(dst, (__ubuf__ float*)src + offset);
        } else { // fp16、bf16
            RegTensor<T> xFp16;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__ubuf__ T*)src + offset));
            Cast<float, T, castTraitFp16ToFp32>(dst, xFp16, preg);
        }
    }

    __aicore__ inline void StoreTensorForDtypeTOut(__ubuf__ T* dst, RegTensor<float>& src, MaskReg& preg,
                                                   uint32_t offset)
    {
        if constexpr (IsSameType<T, float>::value) {
            StoreAlign<T, StoreDist::DIST_NORM>(dst + offset, src, preg);
        } else {
            RegTensor<T> xFp16;
            Cast<T, float, castTraitFp32ToFp16>(xFp16, src, preg);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
        }
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, int64_t curTileRLen, uint32_t curTileA0Len)
    {
        LocalTensor<T> y = yQueue_.DeQue<T>();
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = curTileRLen;
        copyOutParams.blockLen = curTileA0Len * sizeof(T);
        copyOutParams.srcStride = (tilingData_->tileA0Len - curTileA0Len) * sizeof(T) / BLOCK_SIZE;
        copyOutParams.dstStride = (tilingData_->totalA0Len - curTileA0Len) * sizeof(T);
        DataCopyPad<T, PaddingMode::Normal>(yGm_[yGmOffset], y, copyOutParams);
        yQueue_.FreeTensor(y);
    }

private:
    const SoftmaxGradExtARATilingData* tilingData_;

    TPipe* pipe_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> gradQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> x1Queue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> x2Queue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> yQueue_;

    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;

    GlobalTensor<T> gradGm_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> yGm_;
};
} // namespace SoftmaxGradExt

#endif
