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
 * \file swiglu_hi_f4_quant_regbase.h
 * \brief SwiGLU activation + HiFloat4 quantization kernel (regbase, axis=-1)
 */

#ifndef SWIGLU_HI_F4_QUANT_REGBASE_H_
#define SWIGLU_HI_F4_QUANT_REGBASE_H_

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "swiglu_hi_f4_quant_tiling_data.h"

namespace SwigluHiF4Quant {
using namespace AscendC;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::PostLiteral;
using AscendC::MicroAPI::RegLayout;
using AscendC::MicroAPI::SatMode;
using AscendC::MicroAPI::StoreDist;

constexpr int64_t DQ_HIFP4_GROUP_ELE_NUM = 64;
constexpr int64_t DQ_HIFP4_GROUP_BYTE_NUM = 32;
constexpr int64_t DQ_HIFP4_SCALE_ALIGN = 256;
constexpr int64_t DQ_HIFP4_MAXPER4_ALIGN = 128;
constexpr int64_t DQ_HIFP4_MAXPER8_ALIGN = 128;
constexpr int64_t DQ_HIFP4_SCALETMP_ALIGN = 256;
constexpr int64_t DQ_HIFP4_NUM_IN_ONE_BYTE = 2;
constexpr int64_t DQ_HIFP4_BLOCK_BYTE = 32;
constexpr int64_t DQ_HIFP4_BF16_BLOCK_NUM = 16;
constexpr int64_t DQ_HIFP4_BRC_BLOCK_NUM = 8;
constexpr int64_t DQ_HIFP4_CAST_COPYOUT_PACK4_NUM = 4;
constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DIGIT_EIGHT = 8;
constexpr int64_t QUANT_ONCE_NUM = 512;
constexpr int64_t X_ONCE_NUM = 1024;
constexpr int64_t SCALE_ONCE_NUM = 8; // 512个数对应的scale是8

constexpr int64_t HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM = 8;
constexpr int64_t HIFP4_MAXPER8_REG_ONELOOP = 16;
constexpr int64_t HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM = 256; // scale 输出的计算，一次reg循环计算256个group的scale
constexpr int64_t HIFP4_OUTPUT_REG_ONELOOP = 2;

template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using TrueType = IntegralConstant<bool, true>;
using FalseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public FalseType {};
template <typename Tp>
struct IsSame<Tp, Tp> : public TrueType {};

static constexpr AscendC::MicroAPI::CastTrait CAST_BF16_FP16_TO_FP32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FP16_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

// SCALE_ALG: 0=mse_first(有偏): Eb/Ec阈值为4/2; 1=unbiased(无偏): Eb/Ec阈值为3.75/1.875(RNE中点)
template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG = 0>
class SwigluHiF4QuantNeg1Neg1 {
public:
    __aicore__ inline SwigluHiF4QuantNeg1Neg1(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale,
                                const SwigluHiF4QuantTilingData* __restrict tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitGroupIndex();
    __aicore__ inline void CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t dim0OnceSize, int64_t dim1OnceSize);
    __aicore__ inline void ComputeAll(int64_t currentBlockNum);
    __aicore__ inline void ComputeSwigluV1(uint16_t regLoopForSwiglu, __ubuf__ ST* x1Addr, __ubuf__ ST* x2Addr,
                                           __ubuf__ bfloat16_t* yUb);
    __aicore__ inline void ComputeSwigluV2(uint16_t regLoopForSwiglu, __ubuf__ ST* x1Addr, __ubuf__ ST* x2Addr,
                                           __ubuf__ bfloat16_t* yUb);
    __aicore__ inline void ComputeMaxPer4(uint16_t regLoopForMaxPer4, __ubuf__ float* swigluAddr2,
                                          __ubuf__ uint16_t* maxPer4BufAddr);
    __aicore__ inline void ComputeMaxPer8(uint16_t regLoopForMaxPer8, __ubuf__ uint16_t* maxPer4BufReadAddr,
                                          __ubuf__ uint16_t* e8BufAddr, __ubuf__ uint16_t* maxPer8BufAddr);
    __aicore__ inline void ComputeTmpScale(uint16_t regLoopForMaxPer4, __ubuf__ uint16_t* maxPer4BufRead2Addr,
                                           __ubuf__ uint16_t* maxPer8BufReadAddr, __ubuf__ uint16_t* e8BufRead1Addr,
                                           __ubuf__ uint32_t* e1p8BufAddr, __ubuf__ uint8_t* e1p16BufAddr,
                                           __ubuf__ uint16_t* tmpScaleAddr);
    __aicore__ inline void ComputeScale(uint16_t regLoopForScale, __ubuf__ uint8_t* e8BufRead2Addr,
                                        __ubuf__ uint8_t* e1p8BufReadAddr, __ubuf__ uint8_t* e1p16BufReadAddr,
                                        __ubuf__ uint16_t* scaleAddr);
    __aicore__ inline void ComputeOut(uint16_t regLoopForOut, __ubuf__ bfloat16_t* swigluAddr,
                                      __ubuf__ uint16_t* tmpScaleReadAddr, __ubuf__ int32_t* yAddr);
    __aicore__ inline void CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t dim0OnceSize,
                                   int64_t dim1OnceSize);

private:
    const SwigluHiF4QuantTilingData* __restrict tilingData_{nullptr};
    TPipe* pipe_{nullptr};
    int64_t blockIdx_{0};

    TQue<QuePosition::VECIN, DB_BUFFER> inQueueX_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outScaleQueue_;

    TBuf<TPosition::VECCALC> reduceSumBuffer_;
    TBuf<TPosition::VECCALC> bufE8_;
    TBuf<TPosition::VECCALC> buf1E8_;
    TBuf<TPosition::VECCALC> buf1E16_;
    TBuf<TPosition::VECCALC> bufTmpScale_;
    TBuf<TPosition::VECCALC> bufMaxPer4_;
    TBuf<TPosition::VECCALC> bufMaxPer8_;
    TBuf<TPosition::VECCALC> bufSwiglu_;

    GlobalTensor<ST> xGm_;
    GlobalTensor<GT> groupIndexGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<float> scaleGm_;

    int64_t sysRegLen_ = 256;
    int64_t gmCopyInRegF32_ = 0;
    int64_t gmCopyInRegStep_ = 0;
    int64_t gmCopyOutRegStep_ = 0;
    int64_t gmCopyOutScaleRegStep_ = 0;
    int64_t gmCopyInRegStep2_ = 0;
    int64_t gmCopyTmpE1_ = 0;
    int64_t gmCopyTmpE8_ = 0;
    int64_t gmCopyTmpE16_ = 0;
    int64_t gmCopyTmpBufE8Step_ = 0;
    int64_t gmCopyTmpBufOtherStep_ = 0;

    int64_t bufE8Buffer_ = 0;
    int64_t buf1E8Buffer_ = 0;
    int64_t buf1E16Buffer_ = 0;
    int64_t maxPer4Buffer_ = 0;
    int64_t maxPer8Buffer_ = 0;
    int64_t tmpScaleBuffer_ = 0;

    int64_t HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM = 8;
    int64_t HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM = 256;
    int64_t blockSize_ = 64;
    int64_t swigluMode_ = 0;   // swigluMode 0 是左右激活, 1是奇偶激活
    int64_t activateLeft_ = 0; // 0 表示false，右半部分激活, 1表示true，左半部分激活
    float clampLimit_ = 0.0f;  // swiglu 上限值
    float gluBias_ = 0.0f;     // 变体glu的bias
    float gluAlpha_ = 0.0f;    // 变体glu的alpha

    int64_t realCoreNum_ = 0;    // 运行的总核数
    int64_t dim0Size_ = 0;       // 合轴为2维, 和轴后0维度大小  和轴后的shape是 M,2N,  dim0Size_ = M
    int64_t dim1Size_ = 0;       // 和轴后1维度大小 dim1Size_ = 2N
    int64_t factorDim0Size_ = 0; // 和轴后0维基本块大小 基本块大小是按照1* 256个数来的
    int64_t factorDim1Size_ = 0; // 和轴后1维基本块大小  注意基本块是按照N计算的,不是2N,也就是可以放多少个256个数
    int64_t blockFactor_ = 0; // 对dim0分核，每个核处理多少行,例如行数172，核数realCoreNum_=64，blockFactor_ = 172/ 64
    int64_t tailBlock_ = 0;      // 取余的结果，例如核数64,行数是172,那么blockFactor = 172/ 64
                                 // =2,前44行每个核处理3行，后面20个核每个核处理2行,tailBlock_=20
    int64_t loopTimesBDim0_ = 0; // 前面的核0维循环次数，也就是tailBlock_个核循环次数
    int64_t tailBDim0_ = 0; // 前面的核0维循环尾块大小，也就是tailBlock_个核循环最后一次处理多少行
    int64_t loopTimesTDim0_ = 0; // 后面的核0维循环次数，也就是大于tailBlock_的核循环次数
    int64_t tailTDim0_ = 0; // 后面的核0维循环尾块，也就是大于tailBlock_的核循环最后一次处理多少数
    int64_t loopTimesDim1_ = 0; // 1维循环次数
    int64_t tailDim1_ = 0; // 1维最后一次循环的尾块大小，肯定小于256 注意这个也是swiglu后的一般的个数 这里也是代表一半
    int64_t groupIndexNum_ = 0; // groupIndex的shape
    int64_t halfInput_ = 0;
    int64_t quantBlockNumPerUB_ = 0; // UB内一次能保存的最大quant block个数

    int64_t blockOffset_ = 0; // 核间offset
    int64_t blockFactorNum_ = 0;
    int64_t loopTimesDim0_ = 0;
    int64_t tailDim0_ = 0;
    int64_t outputScaleRowBytes_ = 0;

    uint32_t oneBlockUb_ = Ops::Base::GetUbBlockSize();
    constexpr static MicroAPI::CastTrait castTrait_ = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                       RoundMode::CAST_RINT};
};

template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG>
__aicore__ inline void SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, SCALE_ALG>::Init(
    GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale, const SwigluHiF4QuantTilingData* __restrict tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer((__gm__ ST*)x);
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale);
    if constexpr (hasGroupIndex) {
        groupIndexGm_.SetGlobalBuffer((__gm__ GT*)group_index);
    }
    // 开始get tilingData
    dim0Size_ = tilingData_->inputDimNeg2;
    dim1Size_ = tilingData_->inputDimNeg1;
    halfInput_ = tilingData_->outputYDimNeg1;
    realCoreNum_ = tilingData_->usedCoreNum;
    factorDim0Size_ = tilingData_->maxBasicNumUbDimNeg2;
    factorDim1Size_ = tilingData_->maxBasicNumUbDimNeg1;
    int32_t factorSize = factorDim0Size_ * factorDim1Size_;
    quantBlockNumPerUB_ = factorSize * 8;

    outputScaleRowBytes_ = halfInput_ / DQ_HIFP4_GROUP_ELE_NUM * 4;

    pipe_->InitBuffer(inQueueX_, DB_BUFFER, quantBlockNumPerUB_ * DQ_HIFP4_GROUP_ELE_NUM * 2 * sizeof(ST));
    pipe_->InitBuffer(outQueue_, DB_BUFFER, quantBlockNumPerUB_ * DQ_HIFP4_GROUP_BYTE_NUM);
    int64_t alignTo256Ub = Ops::Base::CeilAlign(static_cast<int64_t>(quantBlockNumPerUB_),
                                                static_cast<int64_t>(DQ_HIFP4_SCALE_ALIGN));
    pipe_->InitBuffer(outScaleQueue_, DB_BUFFER, alignTo256Ub * sizeof(float));

    bufE8Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(quantBlockNumPerUB_),
                                        static_cast<int64_t>(DQ_HIFP4_BLOCK_BYTE));
    buf1E8Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(quantBlockNumPerUB_),
                                         static_cast<int64_t>(DQ_HIFP4_BLOCK_BYTE));
    buf1E16Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(quantBlockNumPerUB_),
                                          static_cast<int64_t>(DQ_HIFP4_BLOCK_BYTE));
    maxPer4Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(quantBlockNumPerUB_),
                                          static_cast<int64_t>(DQ_HIFP4_MAXPER4_ALIGN));
    maxPer8Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(quantBlockNumPerUB_),
                                          static_cast<int64_t>(DQ_HIFP4_MAXPER8_ALIGN));
    tmpScaleBuffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(quantBlockNumPerUB_),
                                           static_cast<int64_t>(DQ_HIFP4_SCALETMP_ALIGN));

    pipe_->InitBuffer(bufE8_, bufE8Buffer_ * sizeof(int16_t));
    pipe_->InitBuffer(buf1E8_, buf1E8Buffer_ * sizeof(int32_t));
    pipe_->InitBuffer(buf1E16_, buf1E16Buffer_ * sizeof(int16_t));
    pipe_->InitBuffer(bufMaxPer4_, maxPer4Buffer_ * 16 * sizeof(int16_t));
    pipe_->InitBuffer(bufMaxPer8_, maxPer8Buffer_ * 8 * sizeof(int16_t));
    pipe_->InitBuffer(bufTmpScale_, tmpScaleBuffer_ * 32 * sizeof(int16_t));
    // SwiGLU buffer: quantBlockNumPerUB_ groups of 64 bfloat16 elements
    pipe_->InitBuffer(bufSwiglu_, quantBlockNumPerUB_ * DQ_HIFP4_GROUP_ELE_NUM * sizeof(bfloat16_t));

    swigluMode_ = tilingData_->swigluMode;
    activateLeft_ = tilingData_->activateLeft;
    clampLimit_ = tilingData_->clampLimit;
    gluBias_ = tilingData_->gluBias;
    gluAlpha_ = tilingData_->gluAlpha;

    gmCopyInRegF32_ = sysRegLen_ / sizeof(float);
    gmCopyInRegStep_ = sysRegLen_ / sizeof(float) * 2;
    gmCopyOutRegStep_ = sysRegLen_ / sizeof(int32_t) / DQ_HIFP4_CAST_COPYOUT_PACK4_NUM;
    gmCopyOutScaleRegStep_ = sysRegLen_ / sizeof(float);
    gmCopyInRegStep2_ = sysRegLen_ / sizeof(ST);
    gmCopyTmpE1_ = sysRegLen_ * DIGIT_FOUR / blockSize_ / DIGIT_TWO;
    gmCopyTmpE8_ = sysRegLen_ * DIGIT_FOUR / DIGIT_EIGHT;
    gmCopyTmpE16_ = sysRegLen_ * DIGIT_FOUR / DIGIT_FOUR;
    gmCopyTmpBufE8Step_ = sysRegLen_ * DIGIT_TWO;
    gmCopyTmpBufOtherStep_ = sysRegLen_;
}

template <typename GT>
__aicore__ inline void ReduceAllVf(LocalTensor<GT>& reduceSumUb, LocalTensor<GT>& groupIndexUb, int64_t groupIndexNum_)
{
    uint32_t vfTidx = Ops::Base::GetVRegSize() / sizeof(GT);
    uint16_t times = groupIndexNum_ / vfTidx;
    uint32_t tailNum = groupIndexNum_ % vfTidx;
    uint16_t tailTimes = tailNum != 0 ? 1 : 0;
    auto dstAddr = (__ubuf__ GT*)reduceSumUb.GetPhyAddr();
    auto srcAddr = (__ubuf__ GT*)groupIndexUb.GetPhyAddr();
    auto srcAddr1 = (__ubuf__ GT*)groupIndexUb[times * vfTidx].GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<GT> addReg;
        AscendC::MicroAPI::RegTensor<GT> reduceSumReg;
        AscendC::MicroAPI::RegTensor<GT> reduceSumTReg;
        AscendC::MicroAPI::RegTensor<GT> srcReg;
        AscendC::MicroAPI::Duplicate(addReg, 0);
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<GT, MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < times; i++) {
            AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<GT>(i, vfTidx);
            AscendC::MicroAPI::DataCopy(srcReg, srcAddr, srcIdxOffset);
            AscendC::MicroAPI::Add(addReg, addReg, srcReg, mask);
        }
        AscendC::MicroAPI::ReduceSum(reduceSumReg, addReg, mask);
        for (uint16_t j = 0; j < tailTimes; j++) {
            AscendC::MicroAPI::MaskReg maskT = AscendC::MicroAPI::UpdateMask<GT>(tailNum);
            AscendC::MicroAPI::DataCopy(srcReg, srcAddr1);
            AscendC::MicroAPI::ReduceSum(reduceSumTReg, srcReg, maskT);
            AscendC::MicroAPI::Add(reduceSumReg, reduceSumTReg, reduceSumReg, maskT);
        }
        AscendC::MicroAPI::MaskReg maskOne = AscendC::MicroAPI::CreateMask<GT, MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::DataCopy(dstAddr, reduceSumReg, maskOne);
    }
}

template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG>
__aicore__ inline void SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, SCALE_ALG>::InitGroupIndex()
{
    groupIndexNum_ = tilingData_->groupIndexNum;
    // groupindex存在，核内计算tiling, 先allreduce来计算总行数
    LocalTensor<GT> reduceSumUb = reduceSumBuffer_.Get<GT>();
    LocalTensor<GT> groupIndexUb = inQueueX_.AllocTensor<GT>();
    DataCopyExtParams copyInParam = {1, 0, 0, 0, 0};
    DataCopyPadExtParams<GT> copyPadParams = {false, 0, 0, 0};
    copyInParam.blockLen = groupIndexNum_ * sizeof(GT);
    DataCopyPad(groupIndexUb, groupIndexGm_[0], copyInParam, copyPadParams);
    inQueueX_.EnQue(groupIndexUb);
    groupIndexUb = inQueueX_.DeQue<GT>();
    ReduceAllVf<GT>(reduceSumUb, groupIndexUb, groupIndexNum_);
    inQueueX_.FreeTensor(groupIndexUb);
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);
    int64_t allNum = static_cast<int64_t>(reduceSumUb.GetValue(0));
    // 开始计算分核
    blockFactor_ = allNum / realCoreNum_;
    tailBlock_ = allNum % realCoreNum_;
    int64_t bBlockNum = blockFactor_ + 1;
    int64_t tBlockNum = blockFactor_;
    if (allNum < realCoreNum_) {
        realCoreNum_ = allNum;
        blockFactor_ = 1;
        tailBlock_ = 0;
    }
    if (tailBlock_ == 0) {
        bBlockNum = blockFactor_;
        tBlockNum = blockFactor_;
    }
    loopTimesBDim0_ = (bBlockNum + factorDim0Size_ - 1) /
                      factorDim0Size_; // 前面的核0维循环次数，也就是tailBlock_个核循环次数
    tailBDim0_ = bBlockNum - (loopTimesBDim0_ - 1) *
                                 factorDim0Size_; // 前面的核0维循环尾块大小，也就是tailBlock_个核循环最后一次处理多少行
    loopTimesTDim0_ = (tBlockNum + factorDim0Size_ - 1) /
                      factorDim0Size_; // 后面的核0维循环次数，也就是大于tailBlock_的核循环次数
    tailTDim0_ = tBlockNum - (loopTimesTDim0_ - 1) *
                                 factorDim0Size_; // 后面的核0维循环尾块，也就是大于tailBlock_的核循环最后一次处理多少数
}

template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG>
__aicore__ inline void SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, SCALE_ALG>::CopyIn(int64_t dim0LoopIdx,
                                                                                             int64_t dim1LoopIdx,
                                                                                             int64_t dim0OnceSize,
                                                                                             int64_t dim1OnceSize)
{
    LocalTensor<ST> xLocal = inQueueX_.AllocTensor<ST>();
    DataCopyExtParams copyInParam = {0, 0, 0, 0, 0};
    DataCopyPadExtParams<ST> copyPadParams = {false, 0, 0, 0};
    copyInParam.blockCount = dim0OnceSize;
    copyInParam.blockLen = dim1OnceSize * sizeof(ST);
    if (swigluMode_ == 0) {
        int64_t offset = (blockOffset_ + dim0LoopIdx * factorDim0Size_) * dim1Size_ +
                         dim1LoopIdx * factorDim1Size_ * QUANT_ONCE_NUM;
        copyInParam.srcStride = (dim1Size_ - dim1OnceSize) * sizeof(ST);
        DataCopyPad(xLocal, xGm_[offset], copyInParam, copyPadParams);
        DataCopyPad(xLocal[factorDim0Size_ * factorDim1Size_ * QUANT_ONCE_NUM], xGm_[offset + halfInput_], copyInParam,
                    copyPadParams);
    } else {
        int64_t offset = (blockOffset_ + dim0LoopIdx * factorDim0Size_) * dim1Size_ +
                         dim1LoopIdx * factorDim1Size_ * X_ONCE_NUM;
        copyInParam.blockCount = dim0OnceSize;
        copyInParam.blockLen = dim1OnceSize * DIGIT_TWO * sizeof(ST);
        copyInParam.srcStride = (dim1Size_ - dim1OnceSize * DIGIT_TWO) * sizeof(ST);
        DataCopyPad(xLocal, xGm_[offset], copyInParam, copyPadParams);
    }
    inQueueX_.EnQue(xLocal);
}

template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG>
__aicore__ inline void SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, SCALE_ALG>::ComputeAll(
    int64_t currentBlockNum)
{
    LocalTensor<ST> x = inQueueX_.template DeQue<ST>();
    LocalTensor<uint8_t> y = outQueue_.template AllocTensor<uint8_t>();
    LocalTensor<float> scale = outScaleQueue_.template AllocTensor<float>();

    LocalTensor<uint16_t> e8buf = bufE8_.Get<uint16_t>();
    LocalTensor<uint8_t> e1p8buf = buf1E8_.Get<uint8_t>();
    LocalTensor<uint8_t> e1p16buf = buf1E16_.Get<uint8_t>();
    LocalTensor<uint16_t> tmpscale = bufTmpScale_.Get<uint16_t>();
    LocalTensor<uint16_t> maxper4 = bufMaxPer4_.Get<uint16_t>();
    LocalTensor<uint16_t> maxper8 = bufMaxPer8_.Get<uint16_t>();
    LocalTensor<bfloat16_t> swiglubuf = bufSwiglu_.Get<bfloat16_t>();

    auto x1Addr = (__ubuf__ ST*)x.GetPhyAddr();
    auto x2Addr = (__ubuf__ ST*)x[quantBlockNumPerUB_ * DQ_HIFP4_GROUP_ELE_NUM].GetPhyAddr();
    if (swigluMode_ == 0 && activateLeft_ == 0) {
        x1Addr = (__ubuf__ ST*)x[quantBlockNumPerUB_ * DQ_HIFP4_GROUP_ELE_NUM].GetPhyAddr();
        x2Addr = (__ubuf__ ST*)x.GetPhyAddr();
    }
    if (swigluMode_ == 1) {
        x1Addr = (__ubuf__ ST*)x.GetPhyAddr();
        x2Addr = (__ubuf__ ST*)x[DQ_HIFP4_GROUP_ELE_NUM].GetPhyAddr();
    }

    auto yAddr = (__ubuf__ int32_t*)y.GetPhyAddr();
    auto scaleAddr = (__ubuf__ uint16_t*)scale.GetPhyAddr();

    auto e8BufAddr = (__ubuf__ uint16_t*)e8buf.GetPhyAddr();
    auto e8BufAddr2 = (__ubuf__ uint8_t*)e8buf.GetPhyAddr();
    auto e1p8BufAddr = (__ubuf__ uint32_t*)e1p8buf.GetPhyAddr();
    auto e1p16BufAddr = (__ubuf__ uint8_t*)e1p16buf.GetPhyAddr();
    auto tmpScaleAddr = (__ubuf__ uint16_t*)tmpscale.GetPhyAddr();
    auto maxPer4BufAddr = (__ubuf__ uint16_t*)maxper4.GetPhyAddr();
    auto maxPer8BufAddr = (__ubuf__ uint16_t*)maxper8.GetPhyAddr();

    auto e8BufRead1Addr = e8BufAddr;
    auto e8BufRead2Addr = e8BufAddr2;
    auto e1p8BufReadAddr = (__ubuf__ uint8_t*)e1p8buf.GetPhyAddr();
    auto e1p16BufReadAddr = e1p16BufAddr;
    auto tmpScaleReadAddr = tmpScaleAddr;
    auto maxPer4BufReadAddr = maxPer4BufAddr;
    auto maxPer4BufRead2Addr = maxPer4BufAddr;
    auto maxPer8BufReadAddr = maxPer8BufAddr;

    auto swigluAddr = (__ubuf__ bfloat16_t*)swiglubuf.GetPhyAddr();
    auto swigluAddr2 = (__ubuf__ float*)swiglubuf.GetPhyAddr();

    int64_t regLoopForSwiglu = Ops::Base::CeilDiv(currentBlockNum, static_cast<int64_t>(1));
    int64_t regLoopForMaxPer4 = Ops::Base::CeilDiv(currentBlockNum, HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM);
    int64_t regLoopForMaxPer8 = Ops::Base::CeilDiv(currentBlockNum, HIFP4_MAXPER8_REG_ONELOOP);
    int64_t regLoopForScale = Ops::Base::CeilDiv(currentBlockNum, HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM);
    int64_t regLoopForOut = Ops::Base::CeilDiv(currentBlockNum, HIFP4_OUTPUT_REG_ONELOOP);
    // SwiGLU activation
    if (tilingData_->swigluMode == 0) {
        ComputeSwigluV1(static_cast<uint16_t>(regLoopForSwiglu), x1Addr, x2Addr, swigluAddr);
    } else {
        ComputeSwigluV2(static_cast<uint16_t>(regLoopForSwiglu), x1Addr, x2Addr, swigluAddr);
    }
    // HiF4 quantization
    ComputeMaxPer4(static_cast<uint16_t>(regLoopForMaxPer4), swigluAddr2, maxPer4BufAddr);
    ComputeMaxPer8(static_cast<uint16_t>(regLoopForMaxPer8), maxPer4BufReadAddr, e8BufAddr, maxPer8BufAddr);
    ComputeTmpScale(static_cast<uint16_t>(regLoopForMaxPer4), maxPer4BufRead2Addr, maxPer8BufReadAddr, e8BufRead1Addr,
                    e1p8BufAddr, e1p16BufAddr, tmpScaleAddr);
    ComputeScale(static_cast<uint16_t>(regLoopForScale), e8BufRead2Addr, e1p8BufReadAddr, e1p16BufReadAddr, scaleAddr);
    ComputeOut(static_cast<uint16_t>(regLoopForOut), swigluAddr, tmpScaleReadAddr, yAddr);

    outQueue_.EnQue(y);
    outScaleQueue_.EnQue(scale);
    inQueueX_.FreeTensor(x);
}

template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG>
__aicore__ inline void SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, SCALE_ALG>::ComputeSwigluV1(
    uint16_t regLoopForSwiglu, __ubuf__ ST* x1Addr, __ubuf__ ST* x2Addr, __ubuf__ bfloat16_t* yUb)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<ST> vregX1;
        MicroAPI::RegTensor<ST> vregX2;
        MicroAPI::RegTensor<float> vregX1F;
        MicroAPI::RegTensor<float> vregX2F;
        MicroAPI::RegTensor<float> negReg;
        MicroAPI::RegTensor<float> expReg;
        MicroAPI::RegTensor<float> addsReg;
        MicroAPI::RegTensor<float> sigmoidReg;
        MicroAPI::RegTensor<float> outFReg;
        MicroAPI::RegTensor<bfloat16_t> outTReg;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        float scalarOne = 1.0f;
        float negScalarOne = -1.0f;
        for (uint16_t loopIdx = 0; loopIdx < regLoopForSwiglu; loopIdx++) {
            // AscendC::MicroAPI::AddrReg srcIdxOffset =
            //         AscendC::MicroAPI::CreateAddrReg<ST>(loopIdx, 64);
            auto tmpX1Addr = x1Addr + loopIdx * 64;
            auto tmpX2Addr = x2Addr + loopIdx * 64;
            MicroAPI::DataCopy<ST, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vregX1, tmpX1Addr, 64);
            MicroAPI::DataCopy<ST, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                vregX2, tmpX2Addr, 64);
            MicroAPI::Cast<float, ST, CAST_BF16_FP16_TO_FP32>(vregX1F, vregX1, maskAll);
            MicroAPI::Cast<float, ST, CAST_BF16_FP16_TO_FP32>(vregX2F, vregX2, maskAll);
            MicroAPI::Muls(negReg, vregX1F, negScalarOne, maskAll);
            MicroAPI::Exp(expReg, negReg, maskAll);
            MicroAPI::Adds(addsReg, expReg, scalarOne, maskAll);
            MicroAPI::Div(sigmoidReg, vregX1F, addsReg, maskAll);
            MicroAPI::Mul(outFReg, sigmoidReg, vregX2F, maskAll);
            MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, maskAll);
            AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(loopIdx, 64);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(yUb, outTReg, outOffset, maskAll);
        }
    }
    return;
}

template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG>
__aicore__ inline void SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, SCALE_ALG>::ComputeSwigluV2(
    uint16_t regLoopForSwiglu, __ubuf__ ST* x1Addr, __ubuf__ ST* x2Addr, __ubuf__ bfloat16_t* yUb)
{
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<ST> vregX1;
        AscendC::MicroAPI::RegTensor<ST> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> vregX1DeF;
        AscendC::MicroAPI::RegTensor<float> vregX2DeF;
        AscendC::MicroAPI::RegTensor<float> minsReg;
        AscendC::MicroAPI::RegTensor<float> mulsReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<bfloat16_t> outTReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        float scalarOne = 1.0f;
        float negScalarOne = -1.0f;
        float clampLimit = clampLimit_;
        float negClampLimit = -clampLimit_;
        float negAplha = -gluAlpha_;
        float gluBias = gluBias_;
        for (uint16_t loopIdx = 0; loopIdx < regLoopForSwiglu; loopIdx++) {
            AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<ST>(loopIdx, 128);
            AscendC::MicroAPI::DataCopy<ST, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, x1Addr, srcIdxOffset);
            AscendC::MicroAPI::DataCopy<ST, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, x2Addr, srcIdxOffset);
            AscendC::MicroAPI::Cast<float, ST, CAST_BF16_FP16_TO_FP32>(vregX1F, vregX1, mask); // 64
            AscendC::MicroAPI::Cast<float, ST, CAST_BF16_FP16_TO_FP32>(vregX2F, vregX2, mask); // // 64

            AscendC::MicroAPI::DeInterleave(vregX1DeF, vregX2DeF, vregX1F, vregX2F);
            AscendC::MicroAPI::Mins(minsReg, vregX1DeF, clampLimit, mask);
            AscendC::MicroAPI::Muls(mulsReg, minsReg, negAplha, mask);
            AscendC::MicroAPI::Exp(expReg, mulsReg, mask);
            AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);
            AscendC::MicroAPI::Div(sigmoidReg, minsReg, addsReg, mask);

            AscendC::MicroAPI::Mins(vregX2DeF, vregX2DeF, clampLimit, mask);
            AscendC::MicroAPI::Maxs(vregX2DeF, vregX2DeF, negClampLimit, mask);
            AscendC::MicroAPI::Adds(vregX2DeF, vregX2DeF, gluBias, mask);

            AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2DeF, mask);

            AscendC::MicroAPI::Cast<bfloat16_t, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
            AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(loopIdx, 64);
            DataCopy<bfloat16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(yUb, outTReg, outOffset, mask);
        }
    }
    return;
}

template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG>
__aicore__ inline void SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, SCALE_ALG>::ComputeMaxPer4(
    uint16_t regLoopForMaxPer4, __ubuf__ float* swigluAddr2, __ubuf__ uint16_t* maxPer4BufAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<bfloat16_t> vregIn1;
        MicroAPI::RegTensor<bfloat16_t> vregIn2;
        MicroAPI::RegTensor<bfloat16_t> vregIn3;
        MicroAPI::RegTensor<bfloat16_t> vregIn4;
        MicroAPI::RegTensor<bfloat16_t> vregMax1;
        MicroAPI::RegTensor<bfloat16_t> vregMax2;
        MicroAPI::RegTensor<bfloat16_t> vregOut;
        MicroAPI::RegTensor<bfloat16_t> vregOutbf16;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        AscendC::MicroAPI::Duplicate(expMaskBF16, 0x7fff);
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < regLoopForMaxPer4; loopIdx++) {
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B32>(
                (MicroAPI::RegTensor<float>&)vregIn1, (MicroAPI::RegTensor<float>&)vregIn2, swigluAddr2,
                gmCopyInRegStep_);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1, expMaskBF16, maskAll);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2, expMaskBF16, maskAll);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B32>(
                (MicroAPI::RegTensor<float>&)vregIn3, (MicroAPI::RegTensor<float>&)vregIn4, swigluAddr2,
                gmCopyInRegStep_);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3, expMaskBF16, maskAll);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4, expMaskBF16, maskAll);
            MicroAPI::Max(vregMax1, vregIn1, vregIn2, maskAll);
            MicroAPI::Max(vregMax2, vregIn3, vregIn4, maskAll);
            MicroAPI::DeInterleave(vregMax1, vregMax2, vregMax1, vregMax2);
            MicroAPI::Max(vregOut, vregMax1, vregMax2, maskAll);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxPer4BufAddr, (MicroAPI::RegTensor<uint16_t>&)vregOut, gmCopyInRegStep2_, maskAll);
        }
    }
    return;
}

template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG>
__aicore__ inline void SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, SCALE_ALG>::ComputeMaxPer8(
    uint16_t regLoopForMaxPer8, __ubuf__ uint16_t* maxPer4BufReadAddr, __ubuf__ uint16_t* e8BufAddr,
    __ubuf__ uint16_t* maxPer8BufAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<bfloat16_t> vreg0Maxper4;
        MicroAPI::RegTensor<bfloat16_t> vreg1Maxper4;
        MicroAPI::RegTensor<bfloat16_t> vreg0Maxper8;
        MicroAPI::RegTensor<bfloat16_t> vreg0Maxper64;
        MicroAPI::RegTensor<bfloat16_t> vreg1Maxper64;
        MicroAPI::RegTensor<uint16_t> vreg0E6m2;
        MicroAPI::RegTensor<uint16_t> vreg1E6m2;
        MicroAPI::RegTensor<uint16_t> vregNumscale;
        MicroAPI::Duplicate(vregNumscale, 0x3e12);
        MicroAPI::UnalignReg ureg1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitE6M2 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < regLoopForMaxPer8; loopIdx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(
                (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, (MicroAPI::RegTensor<uint16_t>&)vreg1Maxper4,
                maxPer4BufReadAddr, gmCopyInRegStep2_ * DIGIT_TWO);
            MicroAPI::Max(vreg0Maxper8, vreg0Maxper4, vreg1Maxper4, maskAll);
            MicroAPI::Interleave(vreg0Maxper4, vreg1Maxper4, vreg0Maxper4, vreg1Maxper4);
            MicroAPI::ReduceMaxWithDataBlock<uint16_t>((MicroAPI::RegTensor<uint16_t>&)vreg0Maxper64,
                                                       (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, maskAll);
            MicroAPI::ReduceMaxWithDataBlock<uint16_t>((MicroAPI::RegTensor<uint16_t>&)vreg1Maxper64,
                                                       (MicroAPI::RegTensor<uint16_t>&)vreg1Maxper4, maskAll);
            MicroAPI::Mul(vreg0Maxper64, vreg0Maxper64, (MicroAPI::RegTensor<bfloat16_t>&)vregNumscale, maskAll);
            MicroAPI::Mul(vreg1Maxper64, vreg1Maxper64, (MicroAPI::RegTensor<bfloat16_t>&)vregNumscale, maskAll);
            MicroAPI::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((MicroAPI::RegTensor<fp8_e6m2_t>&)vreg0E6m2,
                                                                  vreg0Maxper64, maskAll);
            MicroAPI::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((MicroAPI::RegTensor<fp8_e6m2_t>&)vreg1E6m2,
                                                                  vreg1Maxper64, maskAll);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxPer8BufAddr, (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper8, gmCopyInRegStep2_, maskAll);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                e8BufAddr, vreg0E6m2, ureg1, 8);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                e8BufAddr, vreg1E6m2, ureg1, 8);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(e8BufAddr, ureg1, 0);
    }
    return;
}

template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG>
__aicore__ inline void SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, SCALE_ALG>::ComputeTmpScale(
    uint16_t regLoopForMaxPer4, __ubuf__ uint16_t* maxPer4BufRead2Addr, __ubuf__ uint16_t* maxPer8BufReadAddr,
    __ubuf__ uint16_t* e8BufRead1Addr, __ubuf__ uint32_t* e1p8BufAddr, __ubuf__ uint8_t* e1p16BufAddr,
    __ubuf__ uint16_t* tmpScaleAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<bfloat16_t> vreg0Maxper4;
        MicroAPI::RegTensor<bfloat16_t> vreg0Maxper8;
        MicroAPI::RegTensor<bfloat16_t> vreg0Maxper64;
        MicroAPI::RegTensor<bfloat16_t> vb16_Ea_rec_bf16;
        MicroAPI::RegTensor<bfloat16_t> vreg0E6m2;
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
        static constexpr AscendC::MicroAPI::CastTrait castTraitReciprocalCast = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < regLoopForMaxPer4; loopIdx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
                (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper64, e8BufRead1Addr, 8);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_US_B16>(
                (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper8, maxPer8BufReadAddr, 64);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, maxPer4BufRead2Addr, 128);
            MicroAPI::ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitReciprocalCast>(
                (MicroAPI::RegTensor<bfloat16_t>&)vb16_Ea_rec_bf16, (MicroAPI::RegTensor<fp8_e6m2_t>&)vreg0Maxper64,
                maskAll);
            MicroAPI::Mul(vreg0Maxper8, vreg0Maxper8, vb16_Ea_rec_bf16, maskAll);
            if constexpr (SCALE_ALG == 1) {
                // unbiased: RNE中点3.75, [3.5,3.75)->3.5
                MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vreg0Maxper8, 3.75, maskAll);
            } else {
                // mse_first: 阈值4, [3.5,4)截断到3.5
                MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vreg0Maxper8, 4, maskAll);
            }
            MicroAPI::Select(vreg0Maxper8, vregNumhalf, vregNum1, vb1E8Comp);
            MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);
            MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
            if constexpr (SCALE_ALG == 1) {
                // unbiased: RNE中点1.875, [1.75,1.875)->1.75, [1.875,2)->2
                MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vreg0Maxper4, 1.875, maskAll);
            } else {
                // mse_first: 阈值2, [1.75,2)截断到1.75
                MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vreg0Maxper4, 2, maskAll);
            }
            MicroAPI::Select(vreg0Maxper4, vregNumhalf, vregNum1, vb1E16Comp);
            MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
            MicroAPI::Mul(vregTmpScale, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);
            MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                tmpScaleAddr + loopIdx * gmCopyTmpBufOtherStep_, (MicroAPI::RegTensor<uint16_t>&)vregTmpScale,
                (MicroAPI::RegTensor<uint16_t>&)vregTmpScale, maskAll);
            MicroAPI::DataCopyUnAlign<uint32_t>(e1p8BufAddr, vb1E8Comp, ureg1);
            MicroAPI::DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE, MaskDist::DIST_PACK>(e1p16BufAddr, vb1E16Comp,
                                                                                            16);
        }
        MicroAPI::DataCopyUnAlignPost<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(e1p8BufAddr, ureg1, 0);
    }
    return;
}

template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG>
__aicore__ inline void SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, SCALE_ALG>::ComputeScale(
    uint16_t regLoopForScale, __ubuf__ uint8_t* e8BufRead2Addr, __ubuf__ uint8_t* e1p8BufReadAddr,
    __ubuf__ uint8_t* e1p16BufReadAddr, __ubuf__ uint16_t* scaleAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint8_t> vregE8;
        MicroAPI::RegTensor<uint8_t> vreg1E8;
        MicroAPI::RegTensor<uint8_t> vreg1E16First;
        MicroAPI::RegTensor<uint8_t> vreg1E16Second;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        for (uint16_t loopIdx = 0; loopIdx < regLoopForScale; loopIdx++) {
            MicroAPI::DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DS_B8>(vregE8, e8BufRead2Addr,
                                                                                             gmCopyTmpBufE8Step_);
            MicroAPI::DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(vreg1E8, e1p8BufReadAddr,
                                                                                            gmCopyTmpBufOtherStep_);
            MicroAPI::DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(
                vreg1E16First, e1p16BufReadAddr, gmCopyTmpBufOtherStep_);
            MicroAPI::DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(
                vreg1E16Second, e1p16BufReadAddr, gmCopyTmpBufOtherStep_);
            MicroAPI::Interleave<uint8_t>(vregE8, vreg1E8, vregE8, vreg1E8);
            MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                scaleAddr + loopIdx * gmCopyTmpBufE8Step_, (MicroAPI::RegTensor<uint16_t>&)vregE8,
                (MicroAPI::RegTensor<uint16_t>&)vreg1E16First, maskAll);
            MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                scaleAddr + loopIdx * gmCopyTmpBufE8Step_ + gmCopyTmpBufOtherStep_,
                (MicroAPI::RegTensor<uint16_t>&)vreg1E8, (MicroAPI::RegTensor<uint16_t>&)vreg1E16Second, maskAll);
        }
    }
    return;
}

template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG>
__aicore__ inline void SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, SCALE_ALG>::ComputeOut(
    uint16_t regLoopForOut, __ubuf__ bfloat16_t* swigluAddr, __ubuf__ uint16_t* tmpScaleReadAddr,
    __ubuf__ int32_t* yAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<bfloat16_t> vregOut;
        MicroAPI::RegTensor<uint16_t> vregScaleTmp;
        MicroAPI::RegTensor<fp4x2_e1m2_t> vbOut;
        MicroAPI::RegTensor<bfloat16_t> vregOutbf16;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < regLoopForOut; loopIdx++) {
            MicroAPI::DataCopy<bfloat16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(vregOutbf16, swigluAddr,
                                                                                               gmCopyInRegStep2_);

            MicroAPI::DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_US_B16>(
                vregScaleTmp, tmpScaleReadAddr, gmCopyOutScaleRegStep_);
            MicroAPI::Mul((MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16, (MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16,
                          (MicroAPI::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
            MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut, vregOutbf16, maskAll);
            MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                yAddr, (MicroAPI::RegTensor<int32_t>&)vbOut, gmCopyOutRegStep_, maskAll);
        }
    }
    return;
}

template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG>
__aicore__ inline void SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, SCALE_ALG>::CopyOut(int64_t dim0LoopIdx,
                                                                                              int64_t dim1LoopIdx,
                                                                                              int64_t dim0OnceSize,
                                                                                              int64_t dim1OnceSize)
{
    LocalTensor<float> scaleLocal = outScaleQueue_.DeQue<float>();
    LocalTensor<uint8_t> yLocal = outQueue_.DeQue<uint8_t>();
    DataCopyExtParams copyOutParamData = {0, 0, 0, 0, 0};
    copyOutParamData.blockCount = dim0OnceSize;
    copyOutParamData.blockLen = dim1OnceSize / DIGIT_TWO;
    copyOutParamData.srcStride = 0; // ub需要block对齐，stride单位是block
    copyOutParamData.dstStride = dim1Size_ / DIGIT_FOUR - copyOutParamData.blockLen;
    int64_t offset = (blockOffset_ + dim0LoopIdx * factorDim0Size_) * dim1Size_ / DIGIT_FOUR +
                     dim1LoopIdx * factorDim1Size_ * QUANT_ONCE_NUM / DIGIT_TWO;

    DataCopyPad(yGm_[offset], yLocal, copyOutParamData);

    DataCopyExtParams copyOutParamScale = {0, 0, 0, 0, 0};
    uint32_t realScaleNum = dim1OnceSize / DQ_HIFP4_GROUP_ELE_NUM; // dim1OnceSize个y对应的scale个数
    copyOutParamScale.blockCount = dim0OnceSize;
    copyOutParamScale.blockLen = realScaleNum * 4;
    copyOutParamScale.srcStride = 0;
    copyOutParamScale.dstStride = outputScaleRowBytes_ - copyOutParamScale.blockLen;
    int64_t offsetScale = (blockOffset_ + dim0LoopIdx * factorDim0Size_) * outputScaleRowBytes_ / 4 +
                          dim1LoopIdx * factorDim1Size_ * SCALE_ONCE_NUM;
    DataCopyPad<float, PaddingMode::Compact>(scaleGm_[offsetScale], scaleLocal, copyOutParamScale);
    outQueue_.FreeTensor(yLocal);
    outScaleQueue_.FreeTensor(scaleLocal);
}

template <typename ST, typename DT, typename GT, bool hasGroupIndex, int64_t SCALE_ALG>
__aicore__ inline void SwigluHiF4QuantNeg1Neg1<ST, DT, GT, hasGroupIndex, SCALE_ALG>::Process()
{
    // InitGroupIndex放在Process中
    if constexpr (hasGroupIndex) {
        pipe_->InitBuffer(reduceSumBuffer_, oneBlockUb_);
        InitGroupIndex();
    } else {
        blockFactor_ = tilingData_->tailCoreBasicNumDimNeg2;
        tailBlock_ = tilingData_->frontCoreNum;
        loopTimesBDim0_ = tilingData_->frontCoreLoopTimes;
        tailBDim0_ = tilingData_->frontCoreLastLoopBasicNum;
        loopTimesTDim0_ = tilingData_->tailCoreLoopTimes;
        tailTDim0_ = tilingData_->tailCoreLastLoopBasicNum;
    }
    loopTimesDim1_ = tilingData_->ubLoopPerRow;
    tailDim1_ = tilingData_->ubTailPerRow; // 注意这个是一半的数
    if (blockIdx_ < tailBlock_) {
        blockFactorNum_ = blockFactor_ + 1;
        blockOffset_ = blockIdx_ * blockFactorNum_;
        loopTimesDim0_ = loopTimesBDim0_;
        tailDim0_ = tailBDim0_;
    } else {
        blockFactorNum_ = blockFactor_;
        blockOffset_ = tailBlock_ * (blockFactor_ + 1) + (blockIdx_ - tailBlock_) * blockFactorNum_;
        loopTimesDim0_ = loopTimesTDim0_;
        tailDim0_ = tailTDim0_;
    }
    // 以上
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    int64_t dim1Size = factorDim1Size_ * QUANT_ONCE_NUM;
    for (int64_t dim0LoopIdx = 0; dim0LoopIdx < loopTimesDim0_; dim0LoopIdx++) {
        int64_t dim0Size = dim0LoopIdx == loopTimesDim0_ - 1 ? tailDim0_ : factorDim0Size_;
        for (int64_t dim1LoopIdx = 0; dim1LoopIdx < loopTimesDim1_; dim1LoopIdx++) {
            int64_t dim1SizeNow = dim1LoopIdx == loopTimesDim1_ - 1 ? tailDim1_ : dim1Size;
            int64_t currentBlockNum = dim0Size * dim1SizeNow / 64;
            CopyIn(dim0LoopIdx, dim1LoopIdx, dim0Size, dim1SizeNow);
            ComputeAll(currentBlockNum);
            CopyOut(dim0LoopIdx, dim1LoopIdx, dim0Size, dim1SizeNow);
        }
    }
}

} // namespace SwigluHiF4Quant

#endif // SWIGLU_HI_F4_QUANT_REGBASE_H_
