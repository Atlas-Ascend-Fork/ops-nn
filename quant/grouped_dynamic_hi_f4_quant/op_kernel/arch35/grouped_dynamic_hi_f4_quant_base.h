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
 * \file grouped_dynamic_hi_f4_quant_base.h
 * \brief
 */

#ifndef OPS_NN_GROUPED_DYNAMIC_HIF4_QUANT_BASE_H
#define OPS_NN_GROUPED_DYNAMIC_HIF4_QUANT_BASE_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "grouped_dynamic_hi_f4_quant_tilingdata.h"

namespace GroupedDynamicHiF4Quant {
using namespace AscendC;
constexpr int64_t BLOCK_SIZE = 64;
constexpr int64_t BYTE_OF_INPUT_X = 2;
constexpr int64_t BYTE_OF_INPUT_GROUP_INDEX = 4;
constexpr int64_t BYTE_OF_OUTPUT_SCALE = 4;
constexpr int64_t Y_NUM_OF_ONE_BYTE = 2;
constexpr int64_t DB_BUFFER = 2;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t NUM_EIGHT = 8;
constexpr int64_t NUM_SIXTENN = 16;
constexpr int64_t UB_BLOCK_SIZE = 32;
constexpr int16_t SHIFT_INDEX = 1;

template <typename xDtype, typename yDtype>
class GroupedDynamicHiF4QuantBase {
public:
    __aicore__ inline GroupedDynamicHiF4QuantBase(){};

protected:
    __aicore__ inline void BaseInitData(const GroupedDynamicHiF4QuantTilingData* tilingData);
    // 计算每4个数最大值
    __aicore__ inline void ComputeMaxPer4(uint16_t maxPer4LoopNum, uint16_t tailMaxper4, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint16_t* maxPer4Addr);
    // 计算每8个数最大值
    __aicore__ inline void ComputeMaxPer8(uint16_t maxPer8LoopNum, uint16_t tailMaxper8, __ubuf__ uint16_t* maxPer4Addr,
                                          __ubuf__ uint16_t* maxPer8Addr);
    // 计算每64个数最大值，并求出E6M2
    __aicore__ inline void ComputeE6M2(uint16_t maxPer64LoopNum, uint16_t tail8Maxper64, __ubuf__ uint16_t* maxPer8Addr,
                                       __ubuf__ uint16_t* eaAddr);
    // 计算scale，方便后续量化
    __aicore__ inline void ComputeTmpScale(uint16_t maxPer64LoopNum, uint16_t tail8Maxper64, uint16_t tailMaxper8,
                                           __ubuf__ uint16_t* maxPer4Addr, __ubuf__ uint16_t* maxPer8Addr,
                                           __ubuf__ uint16_t* eaAddr, __ubuf__ uint8_t* ebAddr,
                                           __ubuf__ uint8_t* ecAddr, __ubuf__ uint16_t* tmpScaleAddr);
    // 拼接ea,eb,ec，输出scale
    __aicore__ inline void ComputeScale(uint16_t maxPer64LoopNum, uint16_t tail8Maxper64, uint16_t tail4Maxper64,
                                        __ubuf__ uint8_t* eaAddr, __ubuf__ uint8_t* ebAddr, __ubuf__ uint8_t* ecAddr,
                                        __ubuf__ uint16_t* scaleFrontAddr, __ubuf__ uint16_t* scaleBackAddr);
    // 计算y
    __aicore__ inline void ComputeOut(uint16_t maxPer4LoopNum, uint16_t tailMaxper4, __ubuf__ xDtype* xAddr,
                                      __ubuf__ uint16_t* tmpScaleAddr, __ubuf__ uint32_t* yAddr);

protected:
    int64_t rowSize_ = 0;
    int64_t colSize_ = 0;
    int64_t blockRowSize_ = 0;
    int64_t tailBlockRowSize_ = 0;
    int64_t blockColSize_ = 0;
    int64_t blockRowNum_ = 0;
    // ub中装下最多的block数
    int64_t ubMaxBlockNum_ = 0;
    int64_t groupNum_ = 0;
    // 由 scale_alg 决定的 Eb/Ec 比较阈值
    // scale_alg=0(默认): Eb 阈值4, Ec 阈值2; scale_alg=1: Eb 阈值3.75, Ec 阈值1.875
    bfloat16_t e8Thresh_ = static_cast<bfloat16_t>(4.0F);
    bfloat16_t e16Thresh_ = static_cast<bfloat16_t>(2.0F);
};

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantBase<xDtype, yDtype>::BaseInitData(
    const GroupedDynamicHiF4QuantTilingData* tilingData)
{
    colSize_ = tilingData->colSize;
    rowSize_ = tilingData->rowSize;
    blockRowNum_ = tilingData->blockRowNum;
    ubMaxBlockNum_ = tilingData->ubMaxblockNum;
    blockRowSize_ = tilingData->blockRowSize;
    tailBlockRowSize_ = tilingData->tailBlockRowSize;
    blockColSize_ = tilingData->blockColSize;
    groupNum_ = tilingData->groupNum;
    e8Thresh_ = (tilingData->scaleAlg == 1) ? static_cast<bfloat16_t>(3.75F) : static_cast<bfloat16_t>(4.0F);
    e16Thresh_ = (tilingData->scaleAlg == 1) ? static_cast<bfloat16_t>(1.875F) : static_cast<bfloat16_t>(2.0F);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantBase<xDtype, yDtype>::ComputeMaxPer4(uint16_t maxPer4LoopNum,
                                                                                   uint16_t tailMaxper4,
                                                                                   __ubuf__ xDtype* xAddr,
                                                                                   __ubuf__ uint16_t* maxPer4Addr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> vregIn;
        MicroAPI::RegTensor<uint16_t> vregMax;
        MicroAPI::RegTensor<uint16_t> vregIn1;
        MicroAPI::RegTensor<uint16_t> vregIn2;
        MicroAPI::RegTensor<uint16_t> vregIn3;
        MicroAPI::RegTensor<uint16_t> vregIn4;
        MicroAPI::RegTensor<bfloat16_t> vregOutbf16;
        MicroAPI::RegTensor<uint16_t> absMask;
        MicroAPI::Duplicate(absMask, 0x7fff);
        MicroAPI::Duplicate(vregMax, 0);
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint16_t>();
        static constexpr MicroAPI::CastTrait castTraitHalf2Bf16 = {
            MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < maxPer4LoopNum; loopIdx++) {
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE>((MicroAPI::RegTensor<xDtype>&)vregIn1,
                                                                                xAddr, 128);
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE>((MicroAPI::RegTensor<xDtype>&)vregIn2,
                                                                                xAddr, 128);
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE>((MicroAPI::RegTensor<xDtype>&)vregIn3,
                                                                                xAddr, 128);
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE>((MicroAPI::RegTensor<xDtype>&)vregIn4,
                                                                                xAddr, 128);
            MicroAPI::And(vregIn1, vregIn1, absMask, maskAll);
            MicroAPI::And(vregIn2, vregIn2, absMask, maskAll);
            MicroAPI::And(vregIn3, vregIn3, absMask, maskAll);
            MicroAPI::And(vregIn4, vregIn4, absMask, maskAll);
            MicroAPI::Max(vregIn1, vregIn1, vregIn2, maskAll);
            MicroAPI::Max(vregIn1, vregIn1, vregIn3, maskAll);
            MicroAPI::Max(vregIn1, vregIn1, vregIn4, maskAll);

            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(vregOutbf16,
                                                                       (MicroAPI::RegTensor<xDtype>&)vregIn1, maskAll);
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    maxPer4Addr, (MicroAPI::RegTensor<uint16_t>&)vregOutbf16, 128, maskAll);
            } else {
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(maxPer4Addr, vregIn1, 128,
                                                                                      maskAll);
            }
        }
        // 尾块处理
        for (uint16_t loopIdx = 0; loopIdx < tailMaxper4; loopIdx++) {
            MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE>((MicroAPI::RegTensor<xDtype>&)vregIn,
                                                                                xAddr, 128);
            MicroAPI::And(vregIn, vregIn, absMask, maskAll);
            MicroAPI::Max(vregMax, vregMax, vregIn, maskAll);
        }
        if constexpr (IsSameType<xDtype, half>::value) {
            MicroAPI::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(vregOutbf16, (MicroAPI::RegTensor<xDtype>&)vregMax,
                                                                   maskAll);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxPer4Addr, (MicroAPI::RegTensor<uint16_t>&)vregOutbf16, 128, maskAll);
        } else {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(maxPer4Addr, vregMax, 128, maskAll);
        }
    }
    return;
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantBase<xDtype, yDtype>::ComputeMaxPer8(uint16_t maxPer8LoopNum,
                                                                                   uint16_t tailMaxper8,
                                                                                   __ubuf__ uint16_t* maxPer4Addr,
                                                                                   __ubuf__ uint16_t* maxPer8Addr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> vregIn;
        MicroAPI::RegTensor<uint16_t> vregMax;
        MicroAPI::RegTensor<uint16_t> vregIn1;
        MicroAPI::RegTensor<uint16_t> vregIn2;
        MicroAPI::Duplicate(vregMax, 0);
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint16_t>();
        for (uint16_t loopIdx = 0; loopIdx < maxPer8LoopNum; loopIdx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregIn1, maxPer4Addr, 128);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregIn2, maxPer4Addr, 128);
            MicroAPI::Max(vregIn1, vregIn1, vregIn2, maskAll);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(maxPer8Addr, vregIn1, 128, maskAll);
        }
        // 尾块处理
        for (uint16_t loopIdx = 0; loopIdx < tailMaxper8; loopIdx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregIn, maxPer4Addr, 128);
            MicroAPI::Max(vregMax, vregMax, vregIn, maskAll);
        }
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(maxPer8Addr, vregMax, 128, maskAll);
    }
    return;
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantBase<xDtype, yDtype>::ComputeE6M2(uint16_t maxPer64LoopNum,
                                                                                uint16_t tail8Maxper64,
                                                                                __ubuf__ uint16_t* maxPer8Addr,
                                                                                __ubuf__ uint16_t* eaAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint16_t> vregMaxper8;
        MicroAPI::RegTensor<bfloat16_t> vregGround;
        MicroAPI::RegTensor<uint16_t> vreg0E6m2;
        MicroAPI::RegTensor<bfloat16_t> vregNumscale;
        // 1/7 = 0x3e12
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t>&)vregNumscale, 0x3e12);
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        static constexpr MicroAPI::CastTrait castTraitE6M2 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t loop1Idx = 0; loop1Idx < maxPer64LoopNum; loop1Idx++) {
            MicroAPI::Duplicate(vregGround, 0);
            for (uint16_t loop2Idx = 0; loop2Idx < 8; loop2Idx++) {
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregMaxper8, maxPer8Addr, 128);
                MicroAPI::Max(vregGround, vregGround, (MicroAPI::RegTensor<bfloat16_t>&)vregMaxper8, maskAll);
            }
            MicroAPI::Mul(vregGround, vregGround, vregNumscale, maskAll);
            MicroAPI::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((MicroAPI::RegTensor<fp8_e6m2_t>&)vreg0E6m2,
                                                                  vregGround, maskAll);
            // E6M2 作为 ea 部分，搬出
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(eaAddr, vreg0E6m2, 128, maskAll);
        }
        // 尾块处理
        MicroAPI::Duplicate(vregGround, 0x0000);
        for (uint16_t loop2Idx = 0; loop2Idx < tail8Maxper64; loop2Idx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregMaxper8, maxPer8Addr, 128);
            MicroAPI::Max(vregGround, vregGround, (MicroAPI::RegTensor<bfloat16_t>&)vregMaxper8, maskAll);
        }
        MicroAPI::Mul(vregGround, vregGround, vregNumscale, maskAll);
        MicroAPI::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((MicroAPI::RegTensor<fp8_e6m2_t>&)vreg0E6m2, vregGround,
                                                              maskAll);
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(eaAddr, vreg0E6m2, 128, maskAll);
    }
    return;
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantBase<xDtype, yDtype>::ComputeTmpScale(
    uint16_t maxPer64LoopNum, uint16_t tail8Maxper64, uint16_t tailMaxper8, __ubuf__ uint16_t* maxPer4Addr,
    __ubuf__ uint16_t* maxPer8Addr, __ubuf__ uint16_t* eaAddr, __ubuf__ uint8_t* ebAddr, __ubuf__ uint8_t* ecAddr,
    __ubuf__ uint16_t* tmpScaleAddr)
{
    __VEC_SCOPE__
    {
        static constexpr AscendC::MicroAPI::CastTrait castTraitReciprocalCast = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        MicroAPI::RegTensor<bfloat16_t> vregMaxper4;
        MicroAPI::RegTensor<bfloat16_t> vregMaxper8;
        MicroAPI::RegTensor<uint16_t> vregMaxper64;
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
        for (uint16_t loop1Idx = 0; loop1Idx < maxPer64LoopNum; loop1Idx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregMaxper64, eaAddr, 128);
            MicroAPI::ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitReciprocalCast>(
                vb16_Ea_rec_bf16, (MicroAPI::RegTensor<fp8_e6m2_t>&)vregMaxper64, maskAll);
            for (uint16_t loop2Idx = 0; loop2Idx < 8; loop2Idx++) {
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    (MicroAPI::RegTensor<uint16_t>&)vregMaxper8, maxPer8Addr, 128);
                MicroAPI::Mul(vregMaxper8, vregMaxper8, vb16_Ea_rec_bf16, maskAll);
                // 每8个数最大值 乘 1/E6M2 大于等于4 => mask，作为输出scale的 eb 部分
                // mask寄存器256bit，一个bit位表示一个字节，两字节类型的mask有效位隔1个bit存放，如10 10 10
                // 10，下方ec相同
                MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vregMaxper8, e8Thresh_, maskAll);
                // 搬出 eb 部分
                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ebAddr, vb1E8Comp, 32);
                MicroAPI::Select(vregMaxper8, vregNumhalf, vregNum1, vb1E8Comp);
                for (uint16_t loop3Idx = 0; loop3Idx < 2; loop3Idx++) {
                    MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        (MicroAPI::RegTensor<uint16_t>&)vregMaxper4, maxPer4Addr, 128);
                    MicroAPI::Mul(vregMaxper4, vregMaxper4, vb16_Ea_rec_bf16, maskAll);
                    MicroAPI::Mul(vregMaxper4, vregMaxper4, vregMaxper8, maskAll);
                    // 每4个数最大值 乘 每8个数最大值 乘 1/E6M2 大于等于2 => mask，作为输出scale的 ec 部分
                    MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vregMaxper4, e16Thresh_, maskAll);
                    // 搬出 ec 部分
                    MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ecAddr, vb1E16Comp, 32);
                    MicroAPI::Select(vregMaxper4, vregNumhalf, vregNum1, vb1E16Comp);
                    MicroAPI::Mul(vregMaxper4, vregMaxper4, vregMaxper8, maskAll);
                    MicroAPI::Mul(vregTmpScale, vregMaxper4, vb16_Ea_rec_bf16, maskAll);
                    // tmpScale作为后续量化系数搬出
                    MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                        tmpScaleAddr, (MicroAPI::RegTensor<uint16_t>&)vregTmpScale, 128, maskAll);
                }
            }
        }
        // 尾块处理，64行是一整块，以8行为一个单位，最后一个尾块还有tail8Maxper64个
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregMaxper64, eaAddr, 128);
        MicroAPI::ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitReciprocalCast>(
            vb16_Ea_rec_bf16, (MicroAPI::RegTensor<fp8_e6m2_t>&)vregMaxper64, maskAll);
        for (uint16_t loop4Idx = 0; loop4Idx < tail8Maxper64; loop4Idx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                (MicroAPI::RegTensor<uint16_t>&)vregMaxper8, maxPer8Addr, 128);
            MicroAPI::Mul(vregMaxper8, vregMaxper8, vb16_Ea_rec_bf16, maskAll);
            MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vregMaxper8, e8Thresh_, maskAll);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ebAddr, vb1E8Comp, 32);
            MicroAPI::Select(vregMaxper8, vregNumhalf, vregNum1, vb1E8Comp);
            for (uint16_t loop5Idx = 0; loop5Idx < 2; loop5Idx++) {
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                    (MicroAPI::RegTensor<uint16_t>&)vregMaxper4, maxPer4Addr, 128);
                MicroAPI::Mul(vregMaxper4, vregMaxper4, vb16_Ea_rec_bf16, maskAll);
                MicroAPI::Mul(vregMaxper4, vregMaxper4, vregMaxper8, maskAll);
                MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vregMaxper4, e16Thresh_, maskAll);
                MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ecAddr, vb1E16Comp, 32);
                MicroAPI::Select(vregMaxper4, vregNumhalf, vregNum1, vb1E16Comp);
                MicroAPI::Mul(vregMaxper4, vregMaxper4, vregMaxper8, maskAll);
                MicroAPI::Mul(vregTmpScale, vregMaxper4, vb16_Ea_rec_bf16, maskAll);
                MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    tmpScaleAddr, (MicroAPI::RegTensor<uint16_t>&)vregTmpScale, 128, maskAll);
            }
        }
        // 尾块的尾块处理，8行是一整块，以4行为一个单位，最后一个尾块还有tailMaxper8个
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            (MicroAPI::RegTensor<uint16_t>&)vregMaxper8, maxPer8Addr, 128);
        MicroAPI::Mul(vregMaxper8, vregMaxper8, vb16_Ea_rec_bf16, maskAll);
        MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vregMaxper8, e8Thresh_, maskAll);
        MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ebAddr, vb1E8Comp, 32);
        MicroAPI::Select(vregMaxper8, vregNumhalf, vregNum1, vb1E8Comp);
        for (uint16_t loop6Idx = 0; loop6Idx < tailMaxper8; loop6Idx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<uint16_t>&)vregMaxper4, maxPer4Addr, 128);
            MicroAPI::Mul(vregMaxper4, vregMaxper4, vb16_Ea_rec_bf16, maskAll);
            MicroAPI::Mul(vregMaxper4, vregMaxper4, vregMaxper8, maskAll);
            MicroAPI::CompareScalar<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vregMaxper4, e16Thresh_, maskAll);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ecAddr, vb1E16Comp, 32);
            MicroAPI::Select(vregMaxper4, vregNumhalf, vregNum1, vb1E16Comp);
            MicroAPI::Mul(vregMaxper4, vregMaxper4, vregMaxper8, maskAll);
            MicroAPI::Mul(vregTmpScale, vregMaxper4, vb16_Ea_rec_bf16, maskAll);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                tmpScaleAddr, (MicroAPI::RegTensor<uint16_t>&)vregTmpScale, 128, maskAll);
        }
    }
    return;
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantBase<xDtype, yDtype>::ComputeScale(
    uint16_t maxPer64LoopNum, uint16_t tail8Maxper64, uint16_t tail4Maxper64, __ubuf__ uint8_t* eaAddr,
    __ubuf__ uint8_t* ebAddr, __ubuf__ uint8_t* ecAddr, __ubuf__ uint16_t* scaleFrontAddr,
    __ubuf__ uint16_t* scaleBackAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<uint8_t> vregGround1;
        MicroAPI::RegTensor<uint8_t> vregZero8;
        MicroAPI::Duplicate(vregZero8, 0x00);
        MicroAPI::RegTensor<uint16_t> vregGround2;
        MicroAPI::RegTensor<uint16_t> vregZero16;
        MicroAPI::Duplicate(vregZero16, 0x0000);
        MicroAPI::RegTensor<uint8_t> vregFlex1;
        MicroAPI::RegTensor<uint16_t> vregFlex2;
        MicroAPI::RegTensor<uint8_t> vregTmp8;
        MicroAPI::RegTensor<uint16_t> vregTmp16;
        MicroAPI::RegTensor<uint8_t> vregE8;

        MicroAPI::MaskReg maskof2e18;
        MicroAPI::MaskReg maskof2e18Tmp;
        MicroAPI::MaskReg maskof2e116;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();

        for (uint16_t loop1Idx = 0; loop1Idx < maxPer64LoopNum; loop1Idx++) {
            // 搬入e6m2，作为ea，与eb进行拼接；之前由bfloat16 cast到e6m2，有效位为隔字节存放，采用下采样搬入
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DS_B8>(
                vregE8, eaAddr, 256);
            MicroAPI::Duplicate(vregGround1, 0x00);

            // 获取eb：按照bit级别 将8行的-2轴排列变为连续存放（-1轴排列）
            MicroAPI::Duplicate(vregFlex1, 0x01);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e18, ebAddr, 32);
            // mask是隔一位起作用，deinterleave取出真实有效位
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Select(vregTmp8, vregFlex1, vregZero8, maskof2e18);
            MicroAPI::Add(vregGround1, vregGround1, vregTmp8, maskAll);

            MicroAPI::Duplicate(vregFlex1, 0x02);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e18, ebAddr, 32);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Select(vregTmp8, vregFlex1, vregZero8, maskof2e18);
            MicroAPI::Add(vregGround1, vregGround1, vregTmp8, maskAll);

            MicroAPI::Duplicate(vregFlex1, 0x04);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e18, ebAddr, 32);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Select(vregTmp8, vregFlex1, vregZero8, maskof2e18);
            MicroAPI::Add(vregGround1, vregGround1, vregTmp8, maskAll);

            MicroAPI::Duplicate(vregFlex1, 0x08);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e18, ebAddr, 32);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Select(vregTmp8, vregFlex1, vregZero8, maskof2e18);
            MicroAPI::Add(vregGround1, vregGround1, vregTmp8, maskAll);

            MicroAPI::Duplicate(vregFlex1, 0x10);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e18, ebAddr, 32);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Select(vregTmp8, vregFlex1, vregZero8, maskof2e18);
            MicroAPI::Add(vregGround1, vregGround1, vregTmp8, maskAll);

            MicroAPI::Duplicate(vregFlex1, 0x20);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e18, ebAddr, 32);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Select(vregTmp8, vregFlex1, vregZero8, maskof2e18);
            MicroAPI::Add(vregGround1, vregGround1, vregTmp8, maskAll);

            MicroAPI::Duplicate(vregFlex1, 0x40);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e18, ebAddr, 32);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Select(vregTmp8, vregFlex1, vregZero8, maskof2e18);
            MicroAPI::Add(vregGround1, vregGround1, vregTmp8, maskAll);

            MicroAPI::Duplicate(vregFlex1, 0x80);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e18, ebAddr, 32);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Select(vregTmp8, vregFlex1, vregZero8, maskof2e18);
            MicroAPI::Add(vregGround1, vregGround1, vregTmp8, maskAll);

            // ea与eb拼接
            Interleave<uint8_t>(vregE8, vregGround1, vregE8, vregGround1);
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                scaleFrontAddr, (MicroAPI::RegTensor<uint16_t>&)vregE8, 128, maskAll);
        }
        // 尾块处理，由于不是整块处理，行数不能确定，因此采用循环移位的方式处理
        MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_DS_B8>(
            vregE8, eaAddr, 256);
        MicroAPI::Duplicate(vregGround1, 0x00);
        MicroAPI::Duplicate(vregFlex1, 0x01);
        for (uint16_t loop2Idx = 0; loop2Idx < tail8Maxper64; loop2Idx++) {
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e18, ebAddr, 32);
            MicroAPI::MaskDeInterleave<uint8_t>(maskof2e18, maskof2e18Tmp, maskof2e18, maskof2e18Tmp);
            MicroAPI::Select(vregTmp8, vregFlex1, vregZero8, maskof2e18);
            MicroAPI::Add(vregGround1, vregGround1, vregTmp8, maskAll);
            MicroAPI::ShiftLefts(vregFlex1, vregFlex1, SHIFT_INDEX, maskAll);
        }
        Interleave<uint8_t>(vregE8, vregGround1, vregE8, vregGround1);
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            scaleFrontAddr, (MicroAPI::RegTensor<uint16_t>&)vregE8, 128, maskAll);

        for (uint16_t loop3Idx = 0; loop3Idx < maxPer64LoopNum; loop3Idx++) {
            MicroAPI::Duplicate(vregGround2, 0x0000);

            // 获取ec: 按照bit级别，将16行的 -2轴排列 变为连续存放（-1轴排列）
            MicroAPI::Duplicate(vregFlex2, 0x0001);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x0002);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x0004);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x0008);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x0010);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x0020);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x0040);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x0080);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x0100);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x0200);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x0400);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x0800);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x1000);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x2000);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x4000);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::Duplicate(vregFlex2, 0x8000);
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);

            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                scaleBackAddr, (MicroAPI::RegTensor<uint16_t>&)vregGround2, 128, maskAll);
        }
        // 尾块处理，由于不是整块处理，行数不能确定，因此采用循环移位的方式处理
        MicroAPI::Duplicate(vregGround2, 0x0000);
        MicroAPI::Duplicate(vregFlex2, 0x0001);
        for (uint16_t loop4Idx = 0; loop4Idx < tail4Maxper64; loop4Idx++) {
            MicroAPI::DataCopy<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_NORM>(
                maskof2e116, ecAddr, 32);
            MicroAPI::Select(vregTmp16, vregFlex2, vregZero16, maskof2e116);
            MicroAPI::Add(vregGround2, vregGround2, vregTmp16, maskAll);
            MicroAPI::ShiftLefts(vregFlex2, vregFlex2, SHIFT_INDEX, maskAll);
        }
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            scaleBackAddr, (MicroAPI::RegTensor<uint16_t>&)vregGround2, 128, maskAll);
    }
    return;
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantBase<xDtype, yDtype>::ComputeOut(uint16_t maxPer4LoopNum,
                                                                               uint16_t tailMaxper4,
                                                                               __ubuf__ xDtype* xAddr,
                                                                               __ubuf__ uint16_t* tmpScaleAddr,
                                                                               __ubuf__ uint32_t* yAddr)
{
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> vregOut;
        MicroAPI::RegTensor<uint16_t> vregScaleTmp;
        MicroAPI::RegTensor<fp4x2_e1m2_t> vbOut;
        MicroAPI::RegTensor<bfloat16_t> vregOutbf16;
        MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t>();
        constexpr static MicroAPI::CastTrait castTraitBfloatToE2M1 = {
            MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            RoundMode::CAST_RINT};
        static constexpr MicroAPI::CastTrait castTraitHalf2Bf16 = {
            MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING,
            RoundMode::CAST_RINT};
        for (uint16_t loop1Idx = 0; loop1Idx < maxPer4LoopNum; loop1Idx++) {
            MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregScaleTmp, tmpScaleAddr, 128);
            for (uint16_t loop2Idx = 0; loop2Idx < 4; loop2Idx++) {
                if constexpr (IsSameType<xDtype, half>::value) {
                    MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregOut, xAddr, 128);
                    MicroAPI::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(vregOutbf16, vregOut, maskAll);
                } else {
                    MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregOutbf16, xAddr, 128);
                }
                MicroAPI::Mul(vregOutbf16, vregOutbf16, (MicroAPI::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
                MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, castTraitBfloatToE2M1>(vbOut, vregOutbf16, maskAll);
                MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                   MicroAPI::StoreDist::DIST_PACK4_B32>(yAddr, (MicroAPI::RegTensor<uint32_t>&)vbOut,
                                                                        16, maskAll);
            }
        }
        // 尾块处理
        MicroAPI::DataCopy<uint16_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregScaleTmp, tmpScaleAddr, 128);
        for (uint16_t loop3Idx = 0; loop3Idx < tailMaxper4; loop3Idx++) {
            if constexpr (IsSameType<xDtype, half>::value) {
                MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregOut, xAddr, 128);
                MicroAPI::Cast<bfloat16_t, xDtype, castTraitHalf2Bf16>(vregOutbf16, vregOut, maskAll);
            } else {
                MicroAPI::DataCopy<xDtype, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregOutbf16, xAddr, 128);
            }
            MicroAPI::Mul(vregOutbf16, vregOutbf16, (MicroAPI::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
            MicroAPI::Cast<fp4x2_e1m2_t, bfloat16_t, castTraitBfloatToE2M1>(vbOut, vregOutbf16, maskAll);
            MicroAPI::DataCopy<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::StoreDist::DIST_PACK4_B32>(
                yAddr, (MicroAPI::RegTensor<uint32_t>&)vbOut, 16, maskAll);
        }
    }
    return;
}

} // namespace GroupedDynamicHiF4Quant
#endif // OPS_NN_GROUPED_DYNAMIC_HIF4_QUANT_BASE_H
