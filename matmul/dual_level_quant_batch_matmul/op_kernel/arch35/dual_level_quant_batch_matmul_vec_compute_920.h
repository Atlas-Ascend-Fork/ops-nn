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
 * \file dual_level_quant_batch_matmul_vec_compute_920.h
 * \brief
 */
#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_VEC_COMPUTE_920_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_VEC_COMPUTE_920_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "dual_level_quant_batch_matmul_vf.h"
#include "op_kernel/math_util.h"
#include "tool_arch35.h"

using AscendC::GetSubBlockIdx;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;

namespace DualLevelQuantBatchMatmul::Arch35 {
#define DLQBMM_VEC_COMPUTE_920_CLASS \
    DualLevelQuantBatchMatmulVectorCompute920<x1Level0ScaleType, x2Level0ScaleType, biasType, yType, hasBias>

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
class DualLevelQuantBatchMatmulVectorCompute920 {
public:
    __aicore__ inline DualLevelQuantBatchMatmulVectorCompute920(){};

    __aicore__ inline void Init(__gm__ x1Level0ScaleType* x1Level0Scale, __gm__ x2Level0ScaleType* x2Level0Scale,
                                __gm__ biasType* bias, __gm__ yType* y);
    __aicore__ inline void CopyX1Level0ScaleToUb(const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam,
                                                 const L0CopyAndCalcParams& l0Params);
    __aicore__ inline void CopyX2Level0ScaleToUb(const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam,
                                                 const L0CopyAndCalcParams& l0Params);
    __aicore__ inline void CopyBiasToUb(const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam,
                                        const L0CopyAndCalcParams& l0Params);
    __aicore__ inline void InitUb(uint64_t mIndex, uint64_t nIndex, uint64_t nL1Offset);
    __aicore__ inline void AntiQuantCompute(uint64_t nIndex, uint64_t cvLoopIdx, uint64_t mL1Offset, uint64_t nL1Offset,
                                            const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam,
                                            const L0CopyAndCalcParams& l0Params);
    __aicore__ inline void CopyUbToGm(uint64_t nIndex, uint64_t mL1Offset, uint64_t nL1Offset,
                                      const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam,
                                      const L0CopyAndCalcParams& l0Params);

    __aicore__ inline LocalTensor<float>& GetCTmpFp32Ub();

    __aicore__ inline void WaitVectorFlag();
    __aicore__ inline void SetMte3ToV(uint64_t mIndex, uint64_t nIndex);
    __aicore__ inline void WaitMte3ToV(uint64_t mIndex, uint64_t nIndex);
    __aicore__ inline void SetAndWaitMte2ToV();
    __aicore__ inline void WaitVToMte2ForX1();
    __aicore__ inline void SetVToMte2ForX1();
    __aicore__ inline void WaitVToMte2ForX2();
    __aicore__ inline void SetVToMte2ForX2();
    __aicore__ inline void WaitVToMte2ForBias();
    __aicore__ inline void SetVToMte2ForBias();

private:
    static constexpr uint64_t CV_LOOP_NUM = 2;
    static constexpr uint64_t SINGLE_REG_PROCESS_SIZE = 64;

    static constexpr uint64_t EVENT_V_MTE2_ID_X1 = 2;
    static constexpr uint64_t EVENT_V_MTE2_ID_X2 = 0;
    static constexpr uint64_t EVENT_V_MTE2_ID_BIAS = 4;
    static constexpr uint64_t EVENT_MTE3_V_ID = 0;
    static constexpr uint64_t EVENT_V_MTE3_ID = 0;
    static constexpr uint64_t EVENT_MTE2_V_ID = 0;

    static constexpr uint64_t BIAS_B32_OFFSET = 256;
    static constexpr uint64_t B_SCALE_B32_OFFSET = 256;
    static constexpr uint64_t X1_SCALE_B32_OFFSET = 160 * 32;
    static constexpr uint64_t X2_SCALE_B32_OFFSET = 256 * 16;
    static constexpr uint64_t C_B32_OFFSET = 128 * 256;
    static constexpr uint64_t C_B32_SIZE_1 = 128 * 128;
    static constexpr uint64_t C_B32_SIZE_2 = 32 * 128;
    static constexpr uint64_t B32_2BANK_OFFSET = 128;

    static constexpr uint64_t X1_LEVEL0_SCALE_B32_STORE_INNER_SIZE = 32; // x1Level0ScaleUB上存放数据的内轴最大长度
    static constexpr uint64_t X2_LEVEL0_SCALE_B32_STORE_K_SIZE = 16; // x2Level0ScaleUB上存放数据的K方向(外轴)最大长度
    uint64_t mte2BiasBufIdx_ = 0;
    uint64_t mte2X1Level0ScaleBufIdx_ = 0;
    uint64_t mte2X2Level0ScaleBufIdx_ = 0;

    GlobalTensor<float> x1Level0ScaleB32Gm_;
    GlobalTensor<float> x2Level0ScaleB32Gm_;
    GlobalTensor<float> biasB32Gm_;
    GlobalTensor<yType> yGm_;

    LocalTensor<float> accUb_{TPosition::VECOUT, 0, 160 * 256};                     // shape(160, 256), 160KB
    LocalTensor<float> addB32Ub_{TPosition::VECIN, 160 * 1024, 128 * 256};          // shape(128, 256), 128KB
    LocalTensor<float> biasB32Ub_{TPosition::VECIN, 288 * 1024, 2 * 256};           // shape(256,) * 2 , 2KB
    LocalTensor<float> x1Level0ScaleB32Ub_{TPosition::VECIN, 290 * 1024, 40 * 256}; // shape(160,32) * 2 , 40KB
    LocalTensor<float> x2Level0ScaleB32Ub_{TPosition::VECIN, 330 * 1024, 32 * 256}; // shape(16,256) * 2 , 32KB
};

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::Init(__gm__ x1Level0ScaleType* x1Level0Scale,
                                                          __gm__ x2Level0ScaleType* x2Level0Scale,
                                                          __gm__ biasType* bias, __gm__ yType* y)
{
    x1Level0ScaleB32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1Level0Scale));
    x2Level0ScaleB32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2Level0Scale));
    biasB32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y));

    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        SetFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X1 + i);
        SetFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X2 + i);
        SetFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_BIAS + i);
        for (uint64_t j = 0; j < DOUBLE_BUFFER_NUM; j++) {
            SetFlag<HardEvent::MTE3_V>(EVENT_MTE3_V_ID + i * DOUBLE_BUFFER_NUM + j);
        }
    }
}

/**
 * 数据搬运逻辑：
 * - mL1Size: 当前core在M方向处理的总数据量（典型值256）
 */
DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::CopyX1Level0ScaleToUb(
    const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam, const L0CopyAndCalcParams& l0Params)
{
    // 计算K方向需要搬运的group数量
    uint64_t x1Level0ScaleGroupNum = basicBlockParam.kGmOffset + basicBlockParam.x1Level0ScaleKUbSize >
                                             basicBlockParam.kSize ?
                                         Ops::Base::CeilDiv(basicBlockParam.kSize - basicBlockParam.kGmOffset,
                                                            basicBlockParam.level0GroupSize) :
                                         Ops::Base::CeilDiv(basicBlockParam.x1Level0ScaleKUbSize,
                                                            basicBlockParam.level0GroupSize);

    DataCopyPad2D<float>(x1Level0ScaleB32Ub_[mte2X1Level0ScaleBufIdx_ % DOUBLE_BUFFER_NUM * X1_SCALE_B32_OFFSET],
                         x1Level0ScaleB32Gm_[basicBlockParam.mGmOffset * basicBlockParam.level0ScaleGmGroupNum +
                                             basicBlockParam.kGmOffset / basicBlockParam.level0GroupSize],
                         l0Params.mL1Size, x1Level0ScaleGroupNum, X1_LEVEL0_SCALE_B32_STORE_INNER_SIZE,
                         basicBlockParam.level0ScaleGmGroupNum);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::CopyX2Level0ScaleToUb(
    const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam, const L0CopyAndCalcParams& l0Params)
{
    uint64_t x2KUbReal = DualLevelQuantBatchMatmul::Arch35::Min<uint64_t>(
        basicBlockParam.l0BaseK * X2_LEVEL0_SCALE_B32_STORE_K_SIZE, basicBlockParam.kSize - basicBlockParam.kGmOffset);
    uint64_t ubX2ScaleKBlockLen = Ops::Base::CeilDiv(x2KUbReal, basicBlockParam.level0GroupSize);
    DataCopyPad2D<float>(
        x2Level0ScaleB32Ub_[mte2X2Level0ScaleBufIdx_ % DOUBLE_BUFFER_NUM * X2_SCALE_B32_OFFSET],
        x2Level0ScaleB32Gm_[Ops::Base::CeilDiv(basicBlockParam.kGmOffset, basicBlockParam.level0GroupSize) *
                                basicBlockParam.nSize +
                            basicBlockParam.nGmOffset],
        ubX2ScaleKBlockLen, l0Params.nL1Size, B_SCALE_B32_OFFSET, basicBlockParam.nSize);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::CopyBiasToUb(
    const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam, const L0CopyAndCalcParams& l0Params)
{
    if constexpr (!hasBias) {
        return;
    }
    // BIAS 的 N 轴的搬运量
    DataCopyPad2D<float>(biasB32Ub_[mte2BiasBufIdx_ % DOUBLE_BUFFER_NUM * BIAS_B32_OFFSET],
                         biasB32Gm_[basicBlockParam.nGmOffset], 1, l0Params.nL1Size, BIAS_B32_OFFSET, l0Params.nL1Size);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::InitUb(uint64_t mIndex, uint64_t nIndex, uint64_t nL1Offset)
{
    uint64_t accUbSize = (mIndex == 0) ? C_B32_SIZE_1 : C_B32_SIZE_2;
    if constexpr (hasBias) {
        // mIndex, nIndex用于确定accUb_数据块的索引，以基本块（160, 256）为例
        //     左侧         右侧
        // 上  (128,128)   (128,128)
        // 下  (32,128)    (32,128)
        InitUbToBias<true>(
            accUbSize / B32_2BANK_OFFSET,
            (__ubuf__ float*)biasB32Ub_.GetPhyAddr(mte2BiasBufIdx_ % DOUBLE_BUFFER_NUM * BIAS_B32_OFFSET + nL1Offset),
            (__ubuf__ float*)accUb_.GetPhyAddr(mIndex * C_B32_OFFSET + nIndex * B32_2BANK_OFFSET));
    } else {
        // 没有bias全填充成0
        InitUbToZero920(accUbSize / B32_2BANK_OFFSET,
                        (__ubuf__ int32_t*)accUb_.template ReinterpretCast<int32_t>().GetPhyAddr(
                            mIndex * C_B32_OFFSET + nIndex * B32_2BANK_OFFSET));
    }
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::AntiQuantCompute(
    uint64_t nIndex, uint64_t cvLoopIdx, uint64_t mL1Offset, uint64_t nL1Offset,
    const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam, const L0CopyAndCalcParams& l0Params)
{
    uint64_t x1GroupNumIdx = basicBlockParam.kGmOffset / basicBlockParam.level0GroupSize %
                             X1_LEVEL0_SCALE_B32_STORE_INNER_SIZE;
    uint64_t x2GroupNumIdx = basicBlockParam.kGmOffset / basicBlockParam.level0GroupSize %
                             X2_LEVEL0_SCALE_B32_STORE_K_SIZE;

    __ubuf__ float* addFp32Addr = (__ubuf__ float*)addB32Ub_.GetPhyAddr(cvLoopIdx % CV_LOOP_NUM * B32_2BANK_OFFSET);
    __ubuf__ float* accAddr = (__ubuf__ float*)accUb_.GetPhyAddr(nIndex * B32_2BANK_OFFSET +
                                                                 mL1Offset * VEC_4BANK_ELEM_B32_920);
    __ubuf__ float* x1Level0ScaleAddr = (__ubuf__ float*)x1Level0ScaleB32Ub_.GetPhyAddr(
        mte2X1Level0ScaleBufIdx_ % DOUBLE_BUFFER_NUM * X1_SCALE_B32_OFFSET +
        mL1Offset * X1_LEVEL0_SCALE_B32_STORE_INNER_SIZE + x1GroupNumIdx);
    __ubuf__ float* x2Level0ScaleAddr = (__ubuf__ float*)x2Level0ScaleB32Ub_.GetPhyAddr(
        mte2X2Level0ScaleBufIdx_ % DOUBLE_BUFFER_NUM * X2_SCALE_B32_OFFSET + x2GroupNumIdx * B_SCALE_B32_OFFSET +
        nL1Offset);
    if (basicBlockParam.kGmOffset + basicBlockParam.l0BaseK < basicBlockParam.kSize) {
        if (l0Params.nL0Size <= SINGLE_REG_PROCESS_SIZE) {
            MulAdd<float, true>(addFp32Addr, accAddr, x1Level0ScaleAddr, x2Level0ScaleAddr, l0Params.mL0Size,
                                X1_LEVEL0_SCALE_B32_STORE_INNER_SIZE);
        } else {
            MulDoubleAdd<float, true>(addFp32Addr, accAddr, x1Level0ScaleAddr, x2Level0ScaleAddr, l0Params.mL0Size,
                                      X1_LEVEL0_SCALE_B32_STORE_INNER_SIZE);
        }
    } else {
        if (l0Params.nL0Size <= SINGLE_REG_PROCESS_SIZE) {
            MulAdd<yType, true>(addFp32Addr, accAddr, x1Level0ScaleAddr, x2Level0ScaleAddr, l0Params.mL0Size,
                                X1_LEVEL0_SCALE_B32_STORE_INNER_SIZE);
        } else {
            MulDoubleAdd<yType, true>(addFp32Addr, accAddr, x1Level0ScaleAddr, x2Level0ScaleAddr, l0Params.mL0Size,
                                      X1_LEVEL0_SCALE_B32_STORE_INNER_SIZE);
        }
    }
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::CopyUbToGm(
    uint64_t nIndex, uint64_t mL1Offset, uint64_t nL1Offset, const DualLevelQbmmBasicBlockOffsetParams& basicBlockParam,
    const L0CopyAndCalcParams& l0Params)
{
    SetFlag<HardEvent::V_MTE3>(EVENT_V_MTE3_ID);
    WaitFlag<HardEvent::V_MTE3>(EVENT_V_MTE3_ID);

    uint64_t realML0Size = mL1Offset + l0Params.mL0Size > l0Params.mL1Size ? l0Params.mL1Size - mL1Offset :
                                                                             l0Params.mL0Size;
    uint64_t realNL0Size = nL1Offset + l0Params.nL0Size > l0Params.nL1Size ? l0Params.nL1Size - nL1Offset :
                                                                             l0Params.nL0Size;

    if (realML0Size <= 0 || realNL0Size <= 0) {
        return;
    }

    // yGm_ M轴方向的偏移：GM上的偏移 + 基本块内的偏移量mL1Offset
    // yGm_ N轴方向的偏移：GM上的偏移 + 基本块内的偏移量nL1Offset
    // accUb_ 的偏移,以(160,256)基本快为例：nIndex索引确定左侧还是右侧块，mL1Offset确定M方向的偏移量
    //  左侧           右侧
    //  (128,128)    (128,128)
    //  (32,128)     (32,128)
    DataCopyPad2D(
        yGm_[(basicBlockParam.mGmOffset + mL1Offset) * basicBlockParam.nSize + basicBlockParam.nGmOffset + nL1Offset],
        accUb_[nIndex * B32_2BANK_OFFSET + mL1Offset * VEC_4BANK_ELEM_B32_920].template ReinterpretCast<yType>(),
        realML0Size, realNL0Size, sizeof(float) / sizeof(yType) * VEC_4BANK_ELEM_B32_920, basicBlockParam.nSize);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline LocalTensor<float>& DLQBMM_VEC_COMPUTE_920_CLASS::GetCTmpFp32Ub() { return addB32Ub_; }

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::SetMte3ToV(uint64_t mIndex, uint64_t nIndex)
{
    SetFlag<HardEvent::MTE3_V>(EVENT_MTE3_V_ID + mIndex * DOUBLE_BUFFER_NUM + nIndex);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::WaitMte3ToV(uint64_t mIndex, uint64_t nIndex)
{
    WaitFlag<HardEvent::MTE3_V>(EVENT_MTE3_V_ID + mIndex * DOUBLE_BUFFER_NUM + nIndex);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::WaitVToMte2ForX1()
{
    WaitFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X1);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::SetVToMte2ForX1()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X1);
    mte2X1Level0ScaleBufIdx_++;
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::WaitVToMte2ForX2()
{
    WaitFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X2 + mte2X2Level0ScaleBufIdx_ % DOUBLE_BUFFER_NUM);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::SetVToMte2ForX2()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X2 + mte2X2Level0ScaleBufIdx_ % DOUBLE_BUFFER_NUM);
    mte2X2Level0ScaleBufIdx_++;
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::WaitVToMte2ForBias()
{
    WaitFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_BIAS + mte2BiasBufIdx_ % DOUBLE_BUFFER_NUM);
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::SetVToMte2ForBias()
{
    SetFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_BIAS + mte2BiasBufIdx_ % DOUBLE_BUFFER_NUM);
    mte2BiasBufIdx_++;
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::WaitVectorFlag()
{
    for (uint64_t i = 0; i < DOUBLE_BUFFER_NUM; i++) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X1 + i);
        WaitFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_X2 + i);
        WaitFlag<HardEvent::V_MTE2>(EVENT_V_MTE2_ID_BIAS + i);
        for (uint64_t j = 0; j < DOUBLE_BUFFER_NUM; j++) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_MTE3_V_ID + i * DOUBLE_BUFFER_NUM + j);
        }
    }
}

DLQBMM_VEC_COMPUTE_TEMPLATE_PARAM
__aicore__ inline void DLQBMM_VEC_COMPUTE_920_CLASS::SetAndWaitMte2ToV()
{
    SetFlag<HardEvent::MTE2_V>(EVENT_MTE2_V_ID);
    WaitFlag<HardEvent::MTE2_V>(EVENT_MTE2_V_ID);
}

} // namespace DualLevelQuantBatchMatmul::Arch35

#endif // DUAL_LEVEL_QUANT_BATCH_MATMUL_VEC_COMPUTE_920_H
