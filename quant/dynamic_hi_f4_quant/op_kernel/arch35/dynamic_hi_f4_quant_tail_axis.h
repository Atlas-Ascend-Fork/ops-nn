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
 * \file dynamic_hi_f4_quant_tail_axis.h
 * \brief
 */
#ifndef DYNAMIC_HIF4_QUANT_TAIL_AXIS_HIFLOAT4_H
#define DYNAMIC_HIF4_QUANT_TAIL_AXIS_HIFLOAT4_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "dynamic_hi_f4_quant_struct.h"

namespace DynamicHiF4Quant {
using namespace AscendC;

constexpr int64_t DQ_HIFP4_GROUP_ELE_NUM = 64;
constexpr int64_t DQ_HIFP4_GROUP_BYTE_NUM = 32; // 64 * 0.5 byte
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
constexpr int64_t BLOCK_SIZE = 64;
template <typename ST, typename DT, uint32_t useQueNum>
class DynamicHiF4QuantTailAxis {
public:
    __aicore__ inline DynamicHiF4QuantTailAxis(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale,
                                const DynamicHiF4QuantTilingData* __restrict tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyIn(const int64_t& gmOffset);
    __aicore__ inline void ComputeAll(int64_t regLoop, int64_t maxPer8RegLoop, int64_t scaleRegLoop,
                                      int64_t outRegLoop);
    __aicore__ inline void ComputeMaxPer4(uint16_t regLoop, __ubuf__ float* xAddr1, __ubuf__ uint16_t* maxPer4BufAddr);
    __aicore__ inline void ComputeMaxPer8(uint16_t maxPer8RegLoop, __ubuf__ uint16_t* maxPer4BufReadAddr,
                                          __ubuf__ uint16_t* e8BufAddr, __ubuf__ uint16_t* maxPer8BufAddr);
    __aicore__ inline void ComputeTmpScale(uint16_t regLoop, __ubuf__ uint16_t* maxPer4BufRead2Addr,
                                           __ubuf__ uint16_t* maxPer8BufReadAddr, __ubuf__ uint16_t* e8BufRead1Addr,
                                           __ubuf__ uint32_t* e1p8BufAddr, __ubuf__ uint8_t* e1p16BufAddr,
                                           __ubuf__ uint16_t* tmpScaleAddr);
    __aicore__ inline void ComputeScale(uint16_t scaleRegLoop, __ubuf__ uint8_t* e8BufRead2Addr,
                                        __ubuf__ uint8_t* e1p8BufReadAddr, __ubuf__ uint8_t* e1p16BufReadAddr,
                                        __ubuf__ uint16_t* scaleAddr);
    __aicore__ inline void ComputeOut(uint16_t outRegLoop, __ubuf__ ST* xAddr2, __ubuf__ uint16_t* tmpScaleReadAddr,
                                      __ubuf__ int32_t* yAddr);
    __aicore__ inline void CopyOut(const int64_t& gmOffset, const int64_t& scaleGmOffset);

private:
protected:
    const DynamicHiF4QuantTilingData* __restrict tilingData_{nullptr};
    TPipe* pipe_{nullptr};
    int64_t blockIdx_{0};
    // 由 scale_alg 决定的 Eb/Ec 比较阈值
    // scale_alg=0(默认): Eb阈值4, Ec阈值2; scale_alg=1: Eb阈值3.75, Ec阈值1.875
    bfloat16_t e8Thresh_ = static_cast<bfloat16_t>(4.0F);
    bfloat16_t e16Thresh_ = static_cast<bfloat16_t>(2.0F);

    TQue<QuePosition::VECIN, DB_BUFFER> inQueueX_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outScaleQueue_;

    TBuf<TPosition::VECCALC> bufE8_;
    TBuf<TPosition::VECCALC> buf1E8_;
    TBuf<TPosition::VECCALC> buf1E16_;
    TBuf<TPosition::VECCALC> bufTmpScale_;
    TBuf<TPosition::VECCALC> bufMaxPer4_;
    TBuf<TPosition::VECCALC> bufMaxPer8_;

    GlobalTensor<ST> xGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<float> scaleGm_;

    DataCopyExtParams dataCopyInParams_;
    DataCopyPadExtParams<ST> padParams_;
    DataCopyExtParams dataCopyOutParams_;
    DataCopyExtParams dataCopyOutScaleParams_;

    int64_t sysRegLen_ = 256;
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

    static constexpr int64_t HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM = 8;
    static constexpr int64_t HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM = 256;
    int64_t blockSize_ = 0;
    constexpr static MicroAPI::CastTrait castTrait_ = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
};

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicHiF4QuantTilingData* __restrict tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    blockSize_ = BLOCK_SIZE;

    xGm_.SetGlobalBuffer((__gm__ ST*)x + blockIdx_ * tilingData_->groupNum * DQ_HIFP4_GROUP_ELE_NUM);
    pipe_->InitBuffer(inQueueX_, useQueNum, tilingData_->alignGroupNum * sizeof(ST) * DQ_HIFP4_GROUP_ELE_NUM);

    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + blockIdx_ * tilingData_->groupNum * DQ_HIFP4_GROUP_BYTE_NUM);
    pipe_->InitBuffer(outQueue_, useQueNum, tilingData_->alignGroupNum * DQ_HIFP4_GROUP_BYTE_NUM);

    scaleGm_.SetGlobalBuffer((__gm__ float*)scale + blockIdx_ * tilingData_->groupNum);
    int64_t alignTo256UbFormer = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                                      static_cast<int64_t>(DQ_HIFP4_SCALE_ALIGN));
    pipe_->InitBuffer(outScaleQueue_, useQueNum, alignTo256UbFormer * sizeof(float));

    bufE8Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                        static_cast<int64_t>(DQ_HIFP4_BLOCK_BYTE));
    buf1E8Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                         static_cast<int64_t>(DQ_HIFP4_BLOCK_BYTE));
    buf1E16Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                          static_cast<int64_t>(DQ_HIFP4_BLOCK_BYTE));
    maxPer4Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                          static_cast<int64_t>(DQ_HIFP4_MAXPER4_ALIGN));
    maxPer8Buffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                          static_cast<int64_t>(DQ_HIFP4_MAXPER8_ALIGN));
    tmpScaleBuffer_ = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData_->alignGroupNum),
                                           static_cast<int64_t>(DQ_HIFP4_SCALETMP_ALIGN));

    pipe_->InitBuffer(bufE8_, bufE8Buffer_ * sizeof(int16_t));
    pipe_->InitBuffer(buf1E8_, buf1E8Buffer_ * sizeof(int32_t));
    pipe_->InitBuffer(buf1E16_, buf1E16Buffer_ * sizeof(int16_t));
    pipe_->InitBuffer(bufMaxPer4_, maxPer4Buffer_ * 16 * sizeof(int16_t));
    pipe_->InitBuffer(bufMaxPer8_, maxPer8Buffer_ * 8 * sizeof(int16_t));
    pipe_->InitBuffer(bufTmpScale_, tmpScaleBuffer_ * 32 * sizeof(int16_t));

    dataCopyInParams_.blockCount = 1;
    dataCopyInParams_.blockLen = 0;
    dataCopyInParams_.srcStride = 0;
    dataCopyInParams_.dstStride = 0;

    dataCopyOutParams_.blockCount = 1;
    dataCopyOutParams_.blockLen = 0;
    dataCopyOutParams_.srcStride = 0;
    dataCopyOutParams_.dstStride = 0;

    dataCopyOutScaleParams_.blockCount = 1;
    dataCopyOutScaleParams_.blockLen = 0;
    dataCopyOutScaleParams_.srcStride = 0;
    dataCopyOutScaleParams_.dstStride = 0;

    padParams_.isPad = false;
    padParams_.leftPadding = 0;
    padParams_.rightPadding = 0;

    gmCopyInRegStep_ = sysRegLen_ / sizeof(float) * 2; // 按照类型float，双搬入  //128
    gmCopyOutRegStep_ = sysRegLen_ / sizeof(int32_t) /
                        DQ_HIFP4_CAST_COPYOUT_PACK4_NUM; // 类型 uint32  DQ_HIFP4_CAST_COPYOUT_PACK4_NUM是4   //16
    gmCopyOutScaleRegStep_ = sysRegLen_ / sizeof(float); // 类型 fp32//64
    gmCopyInRegStep2_ = sysRegLen_ / sizeof(ST);         // 按照类型float//128
    gmCopyTmpE1_ = sysRegLen_ * DIGIT_FOUR / blockSize_ / DIGIT_TWO;
    gmCopyTmpE8_ = sysRegLen_ * DIGIT_FOUR / DIGIT_EIGHT;
    gmCopyTmpE16_ = sysRegLen_ * DIGIT_FOUR / DIGIT_FOUR;
    gmCopyTmpBufE8Step_ = sysRegLen_ * DIGIT_TWO; // 512
    gmCopyTmpBufOtherStep_ = sysRegLen_;          // 256

    // 根据 scale_alg 选择 Eb/Ec 比较阈值
    e8Thresh_ = (tilingData_->scaleAlg == 1) ? static_cast<bfloat16_t>(3.75F) : static_cast<bfloat16_t>(4.0F);
    e16Thresh_ = (tilingData_->scaleAlg == 1) ? static_cast<bfloat16_t>(1.875F) : static_cast<bfloat16_t>(2.0F);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::CopyIn(const int64_t& gmOffset)
{
    auto xLocalIn = inQueueX_.template AllocTensor<ST>();
    DataCopyPad(xLocalIn, xGm_[gmOffset], dataCopyInParams_, padParams_);
    inQueueX_.EnQue(xLocalIn);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::ComputeAll(int64_t regLoop, int64_t maxPer8RegLoop,
                                                                               int64_t scaleRegLoop, int64_t outRegLoop)
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

    auto xAddr1 = (__ubuf__ float*)x.GetPhyAddr();
    auto xAddr2 = (__ubuf__ ST*)x.GetPhyAddr();
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

    ComputeMaxPer4(static_cast<uint16_t>(regLoop), xAddr1, maxPer4BufAddr);
    ComputeMaxPer8(static_cast<uint16_t>(maxPer8RegLoop), maxPer4BufReadAddr, e8BufAddr, maxPer8BufAddr);
    ComputeTmpScale(static_cast<uint16_t>(regLoop), maxPer4BufRead2Addr, maxPer8BufReadAddr, e8BufRead1Addr,
                    e1p8BufAddr, e1p16BufAddr, tmpScaleAddr);
    ComputeScale(static_cast<uint16_t>(scaleRegLoop), e8BufRead2Addr, e1p8BufReadAddr, e1p16BufReadAddr, scaleAddr);
    ComputeOut(static_cast<uint16_t>(outRegLoop), xAddr2, tmpScaleReadAddr, yAddr);

    outQueue_.template EnQue(y);
    outScaleQueue_.template EnQue(scale);
    inQueueX_.template FreeTensor(x);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::ComputeMaxPer4(uint16_t regLoop,
                                                                                   __ubuf__ float* xAddr1,
                                                                                   __ubuf__ uint16_t* maxPer4BufAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<ST> vregIn1;
        MicroAPI::RegTensor<ST> vregIn2;
        MicroAPI::RegTensor<ST> vregIn3;
        MicroAPI::RegTensor<ST> vregIn4;
        MicroAPI::RegTensor<ST> vregMax1;
        MicroAPI::RegTensor<ST> vregMax2;
        MicroAPI::RegTensor<ST> vregOut;
        MicroAPI::RegTensor<bfloat16_t> vregOutbf16;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16;
        AscendC::MicroAPI::Duplicate(expMaskBF16, 0x7fff);
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
            AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                (MicroAPI::RegTensor<float>&)vregIn1, (MicroAPI::RegTensor<float>&)vregIn2, xAddr1, gmCopyInRegStep_);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn1, expMaskBF16, maskAll);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn2, expMaskBF16, maskAll);
            MicroAPI::DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                (MicroAPI::RegTensor<float>&)vregIn3, (MicroAPI::RegTensor<float>&)vregIn4, xAddr1, gmCopyInRegStep_);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn3, expMaskBF16, maskAll);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4,
                                   (AscendC::MicroAPI::RegTensor<uint16_t>&)vregIn4, expMaskBF16, maskAll);
            MicroAPI::Max(vregMax1, vregIn1, vregIn2, maskAll);
            MicroAPI::Max(vregMax2, vregIn3, vregIn4, maskAll);
            MicroAPI::DeInterleave(vregMax1, vregMax2, vregMax1, vregMax2);
            MicroAPI::Max(vregOut, vregMax1, vregMax2, maskAll);
            if constexpr (IsSameType<ST, half>::value) {
                AscendC::MicroAPI::Cast<bfloat16_t, ST, castTraitHalf2Bf16>(vregOutbf16, vregOut, maskAll);
                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maxPer4BufAddr, (MicroAPI::RegTensor<uint16_t>&)vregOutbf16, gmCopyInRegStep2_, maskAll);
            } else {
                AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maxPer4BufAddr, (MicroAPI::RegTensor<uint16_t>&)vregOut, gmCopyInRegStep2_, maskAll);
            }
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::ComputeMaxPer8(
    uint16_t maxPer8RegLoop, __ubuf__ uint16_t* maxPer4BufReadAddr, __ubuf__ uint16_t* e8BufAddr,
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
        MicroAPI::RegTensor<uint16_t> vregBf16E6M2ScaleBias;
        MicroAPI::Duplicate(vregBf16E6M2ScaleBias, 0x3e12);
        MicroAPI::UnalignReg ureg1;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        static constexpr AscendC::MicroAPI::CastTrait castTraitE6M2 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < maxPer8RegLoop; loopIdx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DINTLV_B16>(
                (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, (MicroAPI::RegTensor<uint16_t>&)vreg1Maxper4,
                maxPer4BufReadAddr, gmCopyInRegStep2_ * DIGIT_TWO);
            MicroAPI::Max(vreg0Maxper8, vreg0Maxper4, vreg1Maxper4, maskAll);
            MicroAPI::Interleave(vreg0Maxper4, vreg1Maxper4, vreg0Maxper4, vreg1Maxper4);

            MicroAPI::ReduceMaxWithDataBlock<uint16_t>((MicroAPI::RegTensor<uint16_t>&)vreg0Maxper64,
                                                       (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, maskAll);
            MicroAPI::ReduceMaxWithDataBlock<uint16_t>((MicroAPI::RegTensor<uint16_t>&)vreg1Maxper64,
                                                       (MicroAPI::RegTensor<uint16_t>&)vreg1Maxper4, maskAll);

            MicroAPI::Mul(vreg0Maxper64, vreg0Maxper64, (MicroAPI::RegTensor<bfloat16_t>&)vregBf16E6M2ScaleBias,
                          maskAll);
            MicroAPI::Mul(vreg1Maxper64, vreg1Maxper64, (MicroAPI::RegTensor<bfloat16_t>&)vregBf16E6M2ScaleBias,
                          maskAll);

            MicroAPI::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((MicroAPI::RegTensor<fp8_e6m2_t>&)vreg0E6m2,
                                                                  vreg0Maxper64, maskAll); // 只有前8位才是有效值
            MicroAPI::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((MicroAPI::RegTensor<fp8_e6m2_t>&)vreg1E6m2,
                                                                  vreg1Maxper64, maskAll); // 只有前8位才是有效值

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

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::ComputeTmpScale(
    uint16_t regLoop, __ubuf__ uint16_t* maxPer4BufRead2Addr, __ubuf__ uint16_t* maxPer8BufReadAddr,
    __ubuf__ uint16_t* e8BufRead1Addr, __ubuf__ uint32_t* e1p8BufAddr, __ubuf__ uint8_t* e1p16BufAddr,
    __ubuf__ uint16_t* tmpScaleAddr)
{
    __VEC_SCOPE__
    {
        static constexpr AscendC::MicroAPI::CastTrait castTraitReciprocalCast = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

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
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_E2B_B16>(
                (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper64, e8BufRead1Addr, 8);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_US_B16>(
                (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper8, maxPer8BufReadAddr, 64);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<uint16_t>&)vreg0Maxper4, maxPer4BufRead2Addr, 128);

            MicroAPI::ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitReciprocalCast>(
                (MicroAPI::RegTensor<bfloat16_t>&)vb16_Ea_rec_bf16, (MicroAPI::RegTensor<fp8_e6m2_t>&)vreg0Maxper64,
                maskAll);
            MicroAPI::Mul(vreg0Maxper8, vreg0Maxper8, vb16_Ea_rec_bf16, maskAll);
            MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vreg0Maxper8, e8Thresh_, maskAll);
            MicroAPI::Select(vreg0Maxper8, vregNumhalf, vregNum1, vb1E8Comp);
            MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);
            MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
            MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vreg0Maxper4, e16Thresh_, maskAll);
            MicroAPI::Select(vreg0Maxper4, vregNumhalf, vregNum1, vb1E16Comp);
            MicroAPI::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
            MicroAPI::Mul(vregTmpScale, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);

            MicroAPI::DataCopy<uint16_t, MicroAPI::StoreDist::DIST_INTLV_B16>(
                tmpScaleAddr + loopIdx * gmCopyTmpBufOtherStep_, (MicroAPI::RegTensor<uint16_t>&)vregTmpScale,
                (MicroAPI::RegTensor<uint16_t>&)vregTmpScale, maskAll);

            MicroAPI::DataCopyUnAlign<uint32_t>(e1p8BufAddr, vb1E8Comp, ureg1);

            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_PACK>(
                e1p16BufAddr, vb1E16Comp, 16);
        }
        MicroAPI::DataCopyUnAlignPost<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(e1p8BufAddr, ureg1, 0);
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::ComputeScale(uint16_t scaleRegLoop,
                                                                                 __ubuf__ uint8_t* e8BufRead2Addr,
                                                                                 __ubuf__ uint8_t* e1p8BufReadAddr,
                                                                                 __ubuf__ uint8_t* e1p16BufReadAddr,
                                                                                 __ubuf__ uint16_t* scaleAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint8_t> vregE8;
        MicroAPI::RegTensor<uint8_t> vreg1E8;
        MicroAPI::RegTensor<uint8_t> vreg1E16First;
        MicroAPI::RegTensor<uint8_t> vreg1E16Second;

        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        for (uint16_t loopIdx = 0; loopIdx < scaleRegLoop; loopIdx++) {
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DS_B8>(
                vregE8, e8BufRead2Addr, gmCopyTmpBufE8Step_);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                vreg1E8, e1p8BufReadAddr, gmCopyTmpBufOtherStep_);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                vreg1E16First, e1p16BufReadAddr, gmCopyTmpBufOtherStep_);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
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

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::ComputeOut(uint16_t outRegLoop, __ubuf__ ST* xAddr2,
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
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < outRegLoop; loopIdx++) {
            if constexpr (IsSameType<ST, half>::value) {
                MicroAPI::DataCopy<ST, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    vregOut, xAddr2, gmCopyInRegStep2_);
                AscendC::MicroAPI::Cast<bfloat16_t, ST, castTraitHalf2Bf16>(
                    (MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16, vregOut, maskAll);
            } else {
                MicroAPI::DataCopy<ST, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    vregOutbf16, xAddr2, gmCopyInRegStep2_);
            }
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_US_B16>(
                vregScaleTmp, tmpScaleReadAddr, gmCopyOutScaleRegStep_);
            MicroAPI::Mul((MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16, (MicroAPI::RegTensor<bfloat16_t>&)vregOutbf16,
                          (MicroAPI::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
            MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut, vregOutbf16, maskAll);
            MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                yAddr, (MicroAPI::RegTensor<int32_t>&)vbOut, gmCopyOutRegStep_, maskAll);
        }
    }
    return;
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::CopyOut(const int64_t& gmOffset,
                                                                            const int64_t& scaleGmOffset)
{
    auto yLocalOut = outQueue_.template DeQue<uint8_t>();
    DataCopyPad(yGm_[gmOffset], yLocalOut, dataCopyOutParams_);
    outQueue_.FreeTensor(yLocalOut);

    auto scaleLocalOut = outScaleQueue_.template DeQue<float>();
    DataCopyPad(scaleGm_[scaleGmOffset], scaleLocalOut, dataCopyOutScaleParams_);
    outScaleQueue_.FreeTensor(scaleLocalOut);
}

template <typename ST, typename DT, uint32_t useQueNum>
__aicore__ inline void DynamicHiF4QuantTailAxis<ST, DT, useQueNum>::Process()
{
    bool isLastBlockFlag = (blockIdx_ == tilingData_->usedCoreNum - 1);
    int64_t loopNum = isLastBlockFlag ? tilingData_->rowPerTailCore : tilingData_->rowPerHeadCore;
    int64_t tailNum = isLastBlockFlag ? tilingData_->multiRowNumTailCore : tilingData_->multiRowNumHeadCore;
    int64_t tailRegLoop = isLastBlockFlag ? tilingData_->innerLoopTail : tilingData_->innerLoopTimes;
    int64_t tailMaxPer8RegLoop = isLastBlockFlag ? tilingData_->max8LoopTail : tilingData_->max8LoopTimes;
    int64_t tailScaleRegLoop = isLastBlockFlag ? tilingData_->scaleLoopTail : tilingData_->scaleLoopTimes;
    int64_t tailOutRegLoop = isLastBlockFlag ? tilingData_->outLoopTail : tilingData_->outLoopTimes;

    int64_t xRealFormer = tilingData_->alignGroupNum * DQ_HIFP4_GROUP_ELE_NUM; // 多少个x
    int64_t yRealFormer = tilingData_->alignGroupNum *
                          DQ_HIFP4_GROUP_BYTE_NUM; // 按照uint8 搬出 多少个unit8（一个unin8包含两个hif4）
    int64_t scaleRealFormer = tilingData_->alignGroupNum; // 多少个scale

    dataCopyInParams_.blockLen = xRealFormer * sizeof(ST);
    dataCopyOutParams_.blockLen = yRealFormer;
    dataCopyOutScaleParams_.blockLen = scaleRealFormer * sizeof(float);
    int64_t xGmOffset = 0;
    int64_t yGmOffset = 0;
    int64_t scaleGmOffset = 0;
    for (int64_t i = 0; i < loopNum - 1; ++i) {
        CopyIn(xGmOffset);
        ComputeAll(tilingData_->innerLoopEle, tilingData_->max8LoopEle, tilingData_->scaleLoopEle,
                   tilingData_->outLoopEle);
        CopyOut(yGmOffset, scaleGmOffset);
        xGmOffset += xRealFormer;
        yGmOffset += yRealFormer;
        scaleGmOffset += scaleRealFormer;
    }
    // 尾块部分
    dataCopyInParams_.blockLen = tailNum * DQ_HIFP4_GROUP_ELE_NUM * sizeof(ST);
    dataCopyOutParams_.blockLen = tailNum * DQ_HIFP4_GROUP_BYTE_NUM; // 按照uint8 搬入;
    dataCopyOutScaleParams_.blockLen = tailNum * sizeof(float);
    CopyIn(xGmOffset);
    ComputeAll(tailRegLoop, tailMaxPer8RegLoop, tailScaleRegLoop, tailOutRegLoop);
    CopyOut(yGmOffset, scaleGmOffset);
}

} // namespace DynamicHiF4Quant
#endif
