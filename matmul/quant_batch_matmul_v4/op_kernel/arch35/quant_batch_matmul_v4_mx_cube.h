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
 * \file quant_batch_matmul_v4_mx_cube.h
 * \brief
 */

#pragma once

#include "lib/matmul_intf.h"
#include "matmul_custom_impl.h"
#include "quant_batch_matmul_v4_constant.h"
#include "tool.h"

using AscendC::BLOCK_CUBE;
using AscendC::DataCopy;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::DataCopyParams;
using AscendC::Dn2NzParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::QuePosition;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::WaitFlag;
using matmul::MatmulTypeWithScale;

#define MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS                                                               \
    template <typename inputX1Type, typename inputX2Type, typename biasType, typename yType, bool bTrans, \
              typename inputX1ScaleType, typename inputX2ScaleType>

#define MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS \
    inputX1Type, inputX2Type, biasType, yType, bTrans, inputX1ScaleType, inputX2ScaleType

/* MxA8W4纯Cube模板 */
namespace QuantBatchMatmulV4 {
struct ScaleDnToNzParam {
    int64_t row;
    int64_t col;
    int64_t height;
    int64_t width;
    int64_t gScaleCol;
};
template <typename T>
__aicore__ inline constexpr bool IsHif4Scale()
{
    return AscendC::IsSameType<T, AscendC::hif4_scale>::value;
}

template <typename T>
__aicore__ inline constexpr int32_t GetScaleGroupSize()
{
    if constexpr (IsHif4Scale<T>()) {
        return GROUP_SIZE_HIF4;
    } else {
        return GROUP_SIZE_32;
    }
}

/* MxFp8和MxFp4的scale类型为float8e8m0 */
template <typename SCALE_TYPE>
__aicore__ inline void CopyMxScaleDn2Nz(const LocalTensor<SCALE_TYPE>& dst, const GlobalTensor<SCALE_TYPE>& src,
                                        const ScaleDnToNzParam& inputParam)
{
    Dn2NzParams dn2NzParams;
    dn2NzParams.dnNum = 1;
    dn2NzParams.dValue = inputParam.height;
    dn2NzParams.nValue = CeilDiv<uint64_t>(inputParam.width,
                                           SCALE_COPY_GROUP_SIZE); // 将两个float8e8m0合成一个half类型，一起搬运
    dn2NzParams.srcDnMatrixStride = SCALE_COPY_DEFAULT_STRIDE;
    dn2NzParams.srcDValue = inputParam.gScaleCol / SCALE_COPY_GROUP_SIZE;
    dn2NzParams.dstNzC0Stride = CeilDiv<uint64_t>(inputParam.col, SCALE_COPY_GROUP_SIZE);
    dn2NzParams.dstNzNStride = SCALE_COPY_DEFAULT_NS_STRIDE;
    dn2NzParams.dstNzMatrixStride = SCALE_COPY_DEFAULT_STRIDE;
    GlobalTensor<half> bf16ScaleGlobal;
    bf16ScaleGlobal.SetGlobalBuffer(((__gm__ half*)(src.GetPhyAddr())), inputParam.height * inputParam.width / 2);

    auto bf16ScaleLocal = dst.template ReinterpretCast<half>();
    DataCopy(bf16ScaleLocal, bf16ScaleGlobal, dn2NzParams, true);
}

/* Hif4的scale类型为B32 */
template <typename SCALE_TYPE>
__aicore__ inline void CopyHif4ScaleDn2Nz(const LocalTensor<SCALE_TYPE>& dst, const GlobalTensor<SCALE_TYPE>& src,
                                          const ScaleDnToNzParam& inputParam)
{
    // hif4的scale类型为b32，使用b16(half)类型dn2nz到L1
    Dn2NzParams dn2NzParams;
    dn2NzParams.dnNum = 1;
    dn2NzParams.dValue = inputParam.height;
    dn2NzParams.nValue = inputParam.width * SCALE_COPY_GROUP_SIZE;
    dn2NzParams.srcDnMatrixStride = SCALE_COPY_DEFAULT_STRIDE;
    dn2NzParams.srcDValue = inputParam.gScaleCol * SCALE_COPY_GROUP_SIZE;
    dn2NzParams.dstNzC0Stride = inputParam.col * SCALE_COPY_GROUP_SIZE;
    dn2NzParams.dstNzNStride = SCALE_COPY_DEFAULT_NS_STRIDE;
    dn2NzParams.dstNzMatrixStride = SCALE_COPY_DEFAULT_STRIDE;

    GlobalTensor<half> bf16ScaleGlobal;
    bf16ScaleGlobal.SetGlobalBuffer(((__gm__ half*)(src.GetPhyAddr())),
                                    inputParam.height * inputParam.width * SCALE_COPY_GROUP_SIZE);
    auto bf16ScaleLocal = dst.template ReinterpretCast<half>();
    DataCopy(bf16ScaleLocal, bf16ScaleGlobal, dn2NzParams, true);
}

template <typename SCALE_TYPE>
__aicore__ inline void CopyScaleDn2Nz(const LocalTensor<SCALE_TYPE>& dst, const GlobalTensor<SCALE_TYPE>& src,
                                      const ScaleDnToNzParam& inputParam)
{
    if constexpr (IsHif4Scale<SCALE_TYPE>()) {
        CopyHif4ScaleDn2Nz(dst, src, inputParam);
    } else {
        CopyMxScaleDn2Nz(dst, src, inputParam);
    }
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
class QuantBatchMatmulV4MxCubeKernel {
public:
    static constexpr uint64_t L1_BUFFER_SIZE = 512 * 1024;
    static constexpr uint64_t L1_BUFFER_HALF_SIZE = 256 * 1024;
    static constexpr uint64_t QUADRUPLE_BUFFER = 4;
    static constexpr uint64_t DOUBLE_BUFFER = 2;
    static constexpr uint8_t MAX_AL1_BUF_NUM = 4;
    static constexpr uint8_t MAX_BL1_BUF_NUM = 4;

    __aicore__ inline QuantBatchMatmulV4MxCubeKernel(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y,
                                GM_ADDR workspace, const qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams* tilingData,
                                TPipe* tPipe);
    __aicore__ inline void InitInputs(GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR bias, GM_ADDR y,
                                      GM_ADDR workspace);
    __aicore__ inline void InitL1Params();
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitL1Buffs();

    __aicore__ inline void GetAL1(int64_t kFactorIdx, AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM]);
    __aicore__ inline void GetBL1(int64_t kFactorIdx);
    __aicore__ inline void CopyX2Gm2L1();
    __aicore__ inline void CopyND2NZ(int aL1Idx);
    __aicore__ inline void CopyScaleB2L1();
    __aicore__ inline void CopyBias2L1();
    __aicore__ inline void SetBias();

    __aicore__ inline void IterateMatmul(int64_t kFactorIdx);
    __aicore__ inline bool IterMatmulOut();
    __aicore__ inline void UpdateIdxAndStep();
    __aicore__ inline void Process();
    __aicore__ inline void InitSync(AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM]);
    __aicore__ inline void EndSync(AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM]);
    __aicore__ inline void PostProcess(int32_t kFactorIdx, AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM]);

    using x1ScaleType = inputX1ScaleType;
    using x2ScaleType = typename AscendC::Conditional<IsSameType<inputX2ScaleType, float>::value, AscendC::hif4_scale,
                                                      inputX2ScaleType>::type;
    static constexpr int32_t X2_SCALE_GROUP_SIZE = GetScaleGroupSize<x2ScaleType>();

    using xType = inputX1Type;
    using wType = typename AscendC::Conditional<IsSameType<inputX2ScaleType, float>::value, hifloat4x2_t,
                                                inputX2Type>::type;
    using inputWType = MatmulTypeWithScale<TPosition::TSCM, TPosition::TSCM, CubeFormat::NZ, wType, bTrans,
                                           TPosition::GM>;
    MatmulCustomImpl<xType, wType, biasType, yType, x1ScaleType, x2ScaleType, false, bTrans, inputWType> mmObj_;

protected:
    TPipe* pipe_;
    const qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams* tiling_;

    int32_t curBlockIdx_;
    int32_t nDimIdx_;
    int32_t mDimIdx_;

    int64_t mAL1Size_;
    int64_t kAL1Size_;
    int64_t nBL1Size_;
    int64_t kBL1Size_;
    int64_t aL1DataSize_;
    int64_t bL1DataSize_;

    int64_t kSingleCoreIterNum_;

    int64_t mSingleCoreSize_;
    int64_t kSingleCoreSize_;
    int64_t nSingleCoreSize_;

    int64_t tailM_;
    int64_t tailN_;

    int64_t tailL1M_;
    int64_t tailL1N_;

    int64_t mBlockOffset_;
    int64_t nBlockOffset_;

    int64_t kAL1Offset_;
    int64_t kBL1Offset_;
    int64_t kAL1Len_;
    int64_t kBL1Len_;
    int64_t nBL1Offset_;
    int64_t aGmOffset_;
    int64_t bL1Offset_;
    int64_t aL1Offset_;
    int64_t scaleAL1DataSize_;
    int64_t scaleBL1DataSize_;
    int64_t scaleAGmOffset_;
    int64_t scaleBGmOffset_;

    int64_t curML0Idx_;
    int64_t curNL0Idx_;
    int64_t curML1Idx_;
    int64_t curNL1Idx_;
    int64_t curStepM_;
    int64_t curStepN_;
    int64_t mIter_;
    int64_t nIter_;
    int64_t baseUseM_;
    int64_t baseUseN_;
    int64_t mAL1Len_;
    int64_t nBL1Len_;

    int64_t kAl1Factor_;
    int64_t kBl1Factor_;

    uint8_t curAL1BufIdx_ = 0;
    uint64_t curBL1BufIdx_ = 0;
    bool isFirstIter_ = true;
    int8_t biasPingPong_ = 1;
    int8_t biasIdx_ = 0;

    int8_t scaleAIdx_ = 0;
    int8_t scaleBIdx_ = 0;
    int8_t scaleAFactor_ = 1;
    int8_t scaleBFactor_ = 1;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<uint8_t> wGlobalU8_;
    GlobalTensor<x1ScaleType> x1ScaleGlobal_;
    GlobalTensor<x2ScaleType> x2ScaleGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<yType> yGlobal_;

    LocalTensor<xType> aL1LocalBuf0_;
    LocalTensor<xType> aL1LocalBuf1_;
    LocalTensor<xType> aL1LocalBuf2_;
    LocalTensor<xType> aL1LocalBuf3_;
    LocalTensor<wType> bL1LocalBuf0_;
    LocalTensor<wType> bL1LocalBuf1_;
    LocalTensor<wType> bL1LocalBuf2_;
    LocalTensor<wType> bL1LocalBuf3_;
    LocalTensor<x1ScaleType> scaleAL1Buf0_;
    LocalTensor<x1ScaleType> scaleAL1Buf1_;
    LocalTensor<x1ScaleType> scaleAL1Buf2_;
    LocalTensor<x1ScaleType> scaleAL1Buf3_;
    LocalTensor<x2ScaleType> scaleBL1Buf0_;
    LocalTensor<x2ScaleType> scaleBL1Buf1_;
    LocalTensor<x2ScaleType> scaleBL1Buf2_;
    LocalTensor<x2ScaleType> scaleBL1Buf3_;
    LocalTensor<biasType> biasL1Buf0_;
    LocalTensor<biasType> biasL1Buf1_;
    TBuf<QuePosition::TSCM> l1Tbuf_;
    AscendC::TEventID eventIdsScaleAMte1ToMte2_[DOUBLE_BUFFER];
    AscendC::TEventID eventIdsScaleBMte1ToMte2_[DOUBLE_BUFFER];
};

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y, GM_ADDR workspace,
    const qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams* tilingData, TPipe* tPipe)
{
    if ASCEND_IS_AIV {
        return;
    }

    tiling_ = tilingData;
    pipe_ = tPipe;

    curBlockIdx_ = GetBlockIdx();

    nDimIdx_ = curBlockIdx_ / tiling_->cubeNumBlocksM;
    mDimIdx_ = curBlockIdx_ % tiling_->cubeNumBlocksM;

    InitTilingData();

    InitInputs(x1, x2, x1Scale, x2Scale, bias, y, workspace);

    InitL1Buffs();

    mmObj_.Init(&tiling_->matmulTiling, tPipe);
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::InitInputs(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace)
{
    // 申请 Global Buffer
    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x1));
    wGlobalU8_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x2));
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y));
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType*>(bias));
    x1ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ x1ScaleType*>(x1Scale));
    x2ScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ x2ScaleType*>(x2Scale));
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::InitL1Params()
{
    kAL1Size_ = tiling_->matmulTiling.stepKa * tiling_->matmulTiling.baseK;
    kBL1Size_ = tiling_->matmulTiling.stepKb * tiling_->matmulTiling.baseK;
    int64_t maxKL1 = Max(kAL1Size_, kBL1Size_);
    kAl1Factor_ = CeilDiv(maxKL1, kBL1Size_);
    kBl1Factor_ = CeilDiv(maxKL1, kAL1Size_);
    scaleAFactor_ = tiling_->matmulTiling.mxTypePara & 0xff;
    scaleBFactor_ = (tiling_->matmulTiling.mxTypePara >> 8) & 0xff; // 高8位表示scaleBFactor_

    mAL1Size_ = tiling_->matmulTiling.stepM * tiling_->matmulTiling.baseM;
    nBL1Size_ = tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseN;
    aL1DataSize_ = mAL1Size_ * kAL1Size_;
    bL1DataSize_ = nBL1Size_ * kBL1Size_;
    scaleAL1DataSize_ = aL1DataSize_ * scaleAFactor_ / GROUP_SIZE_32;
    scaleBL1DataSize_ = bL1DataSize_ * scaleBFactor_ / X2_SCALE_GROUP_SIZE;

    uint64_t singleM = CeilAlign<uint64_t>(CeilDiv<uint64_t>(tiling_->mSize, tiling_->cubeNumBlocksM), BLOCK_CUBE);
    // weightNz转置场景 shape为(k1, n1, n0, k0) n0为固定值16
    uint64_t singleN = CeilAlign<uint64_t>(CeilDiv<uint64_t>(tiling_->nSize, tiling_->cubeNumBlocksN), BLOCK_CUBE);

    kSingleCoreIterNum_ = CeilDiv<uint64_t>(tiling_->kSize, Min(kAL1Size_, kBL1Size_));

    int64_t mTailCoreSize_ = tiling_->mSize - (tiling_->cubeNumBlocksM - 1) * singleM; // 尾核 m 大小
    int64_t nTailCoreSize_ = tiling_->nSize - (tiling_->cubeNumBlocksN - 1) * singleN; // 尾核 n 大小

    mSingleCoreSize_ = mDimIdx_ == tiling_->cubeNumBlocksM - 1 ? mTailCoreSize_ : singleM; // 单核内 m 方向大小
    nSingleCoreSize_ = nDimIdx_ == tiling_->cubeNumBlocksN - 1 ? nTailCoreSize_ : singleN; // 单核内 n 方向大小

    tailL1M_ = mSingleCoreSize_ % mAL1Size_; // 单核内l1上m方向尾块
    if (tailL1M_ == 0) {
        tailL1M_ = mAL1Size_;
    }

    tailL1N_ = nSingleCoreSize_ % nBL1Size_; // 单核内l1上n方向尾块
    if (tailL1N_ == 0) {
        tailL1N_ = nBL1Size_;
    }

    mBlockOffset_ = mDimIdx_ * singleM; // m 方向核间偏移
    nBlockOffset_ = nDimIdx_ * singleN; // n 方向核间偏移
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::InitTilingData()
{
    InitL1Params();

    mIter_ = CeilDiv<int64_t>(mSingleCoreSize_, tiling_->matmulTiling.baseM); // 单核内一共有basem块数量
    nIter_ = CeilDiv<int64_t>(nSingleCoreSize_, tiling_->matmulTiling.baseN); // 单核内一共有basen块数量
    biasPingPong_ = Min<int64_t>(nIter_, DOUBLE_BUFFER);

    tailM_ = mSingleCoreSize_ % tiling_->matmulTiling.baseM; // 单核内l0上m方向尾块
    if (tailM_ == 0) {
        tailM_ = tiling_->matmulTiling.baseM;
    }

    tailN_ = nSingleCoreSize_ % tiling_->matmulTiling.baseN; // 单核内l0上n方向尾块
    if (tailN_ == 0) {
        tailN_ = tiling_->matmulTiling.baseN;
    }
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::InitL1Buffs()
{
    pipe_->InitBuffer(l1Tbuf_, L1_BUFFER_SIZE);

    uint64_t l1BufOffsetPart1 = 0;
    uint64_t l1BufOffsetPart2 = L1_BUFFER_HALF_SIZE;
    uint64_t bL1DataBytes = bL1DataSize_ >> 1; // BL1 是 FP4 类型

    bL1LocalBuf0_ = l1Tbuf_.template GetWithOffset<wType>(bL1DataSize_, l1BufOffsetPart1);
    l1BufOffsetPart1 += bL1DataBytes;
    if (tiling_->BL1Pingpong == DOUBLE_BUFFER) {
        bL1LocalBuf1_ = l1Tbuf_.template GetWithOffset<wType>(bL1DataSize_, l1BufOffsetPart2);
        l1BufOffsetPart2 += bL1DataBytes;
    }

    aL1LocalBuf0_ = l1Tbuf_.template GetWithOffset<xType>(aL1DataSize_, l1BufOffsetPart1);
    l1BufOffsetPart1 += aL1DataSize_ * sizeof(xType);
    if (tiling_->AL1Pingpong == DOUBLE_BUFFER) {
        aL1LocalBuf1_ = l1Tbuf_.template GetWithOffset<xType>(aL1DataSize_, l1BufOffsetPart2);
        l1BufOffsetPart2 += aL1DataSize_ * sizeof(xType);
    }

    scaleBL1Buf0_ = l1Tbuf_.template GetWithOffset<x2ScaleType>(scaleBL1DataSize_, l1BufOffsetPart1);
    l1BufOffsetPart1 += scaleBL1DataSize_ * sizeof(x2ScaleType);
    scaleBL1Buf1_ = l1Tbuf_.template GetWithOffset<x2ScaleType>(scaleBL1DataSize_, l1BufOffsetPart2);
    l1BufOffsetPart2 += scaleBL1DataSize_ * sizeof(x2ScaleType);

    scaleAL1Buf0_ = l1Tbuf_.template GetWithOffset<x1ScaleType>(scaleAL1DataSize_, l1BufOffsetPart1);
    l1BufOffsetPart1 += scaleAL1DataSize_ * sizeof(x1ScaleType);
    scaleAL1Buf1_ = l1Tbuf_.template GetWithOffset<x1ScaleType>(scaleAL1DataSize_, l1BufOffsetPart2);
    l1BufOffsetPart2 += scaleAL1DataSize_ * sizeof(x1ScaleType);

    if (tiling_->matmulTiling.isBias) {
        biasL1Buf0_ = l1Tbuf_.template GetWithOffset<biasType>(nBL1Size_, l1BufOffsetPart1);
        l1BufOffsetPart1 += nBL1Size_ * sizeof(biasType);
        if (biasPingPong_ > 1) {
            biasL1Buf1_ = l1Tbuf_.template GetWithOffset<biasType>(nBL1Size_, l1BufOffsetPart2);
            l1BufOffsetPart2 += nBL1Size_ * sizeof(biasType);
        }
    }
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::CopyND2NZ(int aL1Idx)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;

    // 以下参数是A不转置场景的设置
    nd2nzParams.nValue = mAL1Len_;
    nd2nzParams.dValue = kAL1Len_;
    nd2nzParams.srcDValue = tiling_->matmulTiling.Ka;

    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = CeilAlign<int64_t>(nd2nzParams.nValue, BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    DataCopy(aL1Idx == IDX_0 ? aL1LocalBuf0_ : aL1LocalBuf1_, xGlobal_[aGmOffset_], nd2nzParams, true);
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::GetAL1(
    int64_t kFactorIdx, AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM])
{
    kAL1Offset_ = kFactorIdx / kAl1Factor_ * kAL1Size_;
    kAL1Len_ = Min<int64_t>(tiling_->kSize - kAL1Offset_, kAL1Size_);
    if (kFactorIdx % kAl1Factor_ != 0) {
        return;
    }

    // 等待目标buffer可用
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[curAL1BufIdx_]);
    aGmOffset_ = mBlockOffset_ + curML0Idx_ * tiling_->matmulTiling.baseM;

    // A不转置，设置GmOffset
    aGmOffset_ *= tiling_->matmulTiling.Ka;
    aGmOffset_ += (kFactorIdx / kAl1Factor_) * kAL1Size_;

    if (kFactorIdx % scaleAFactor_ == 0) {
        scaleAGmOffset_ = mBlockOffset_ + curML0Idx_ * tiling_->matmulTiling.baseM;
        scaleAGmOffset_ *= CeilDiv<int64_t>(tiling_->kAlign, GROUP_SIZE_32);
        scaleAGmOffset_ += CeilDiv<int64_t>(kAL1Size_ * scaleAFactor_ * (kFactorIdx / scaleAFactor_), GROUP_SIZE_32);

        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[scaleAIdx_]);
        int64_t scaleKAL1Size = kAL1Size_ * scaleAFactor_;
        if (kAL1Offset_ + scaleKAL1Size > tiling_->kSize) {
            scaleKAL1Size = tiling_->kSize - kAL1Offset_;
        }

        ScaleDnToNzParam x1ScaleDnToNzParam = {0, kAL1Size_ * scaleAFactor_ / GROUP_SIZE_32, mAL1Len_,
                                               CeilDiv<int64_t>(scaleKAL1Size, GROUP_SIZE_32),
                                               CeilDiv<int64_t>(tiling_->kAlign, GROUP_SIZE_32)};
        if (scaleAIdx_ == 0) {
            CopyScaleDn2Nz<x1ScaleType>(scaleAL1Buf0_, x1ScaleGlobal_[scaleAGmOffset_], x1ScaleDnToNzParam);
        } else {
            CopyScaleDn2Nz<x1ScaleType>(scaleAL1Buf1_, x1ScaleGlobal_[scaleAGmOffset_], x1ScaleDnToNzParam);
        }
    }
    // CopyND2NZ 函数内部包含 SetFlag<MTE2_MTE1> / WaitFlag<MTE2_MTE1> 等同步指令
    CopyND2NZ(curAL1BufIdx_);
}

/*
 * 该函数作用为通过 IterMatmulOut 每次移动一个 baseM 或 baseN，并循环遍历 KL1，
 * 计算好 AL1 和 BL1 的搬运时刻和 index 后，调用 compute 进行计算
 */
MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }

    uint64_t usedCoreNum = tiling_->cubeNumBlocksM * tiling_->cubeNumBlocksN;
    if ((curBlockIdx_) >= usedCoreNum) {
        return;
    }
    AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM];
    InitSync(eventIdsMte1ToMte2);

    while (IterMatmulOut()) {
        for (int32_t kFactorIdx = 0; kFactorIdx < kSingleCoreIterNum_; kFactorIdx++) {
            GetAL1(kFactorIdx, eventIdsMte1ToMte2);
            GetBL1(kFactorIdx);
            IterateMatmul(kFactorIdx);
            PostProcess(kFactorIdx, eventIdsMte1ToMte2);
        }

        uint64_t outOffset = (mBlockOffset_ + curML0Idx_ * tiling_->matmulTiling.baseM) * tiling_->nSize +
                             nBlockOffset_ + curNL0Idx_ * tiling_->matmulTiling.baseN;
#ifndef __CCE_KT_TEST__
        mmObj_.GetTensorC(yGlobal_[outOffset]);
#endif
    }

    mmObj_.End();

    EndSync(eventIdsMte1ToMte2);
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::GetBL1(int64_t kFactorIdx)
{
    kBL1Offset_ = kFactorIdx / kBl1Factor_ * kBL1Size_;
    kBL1Len_ = Min<int64_t>(tiling_->kSize - kBL1Offset_, kBL1Size_);
    nBL1Offset_ = nBlockOffset_ + curNL1Idx_ * tiling_->matmulTiling.baseN;
    if (kFactorIdx % kBl1Factor_ != 0) {
        return;
    }
    // A和B使用同一组反向同步MTE1_MTE2
    CopyX2Gm2L1();
    SetFlag<HardEvent::MTE2_MTE1>(curBL1BufIdx_);
    WaitFlag<HardEvent::MTE2_MTE1>(curBL1BufIdx_);
    if (tiling_->matmulTiling.isBias && kFactorIdx == 0) {
        CopyBias2L1();
    }

    if (kFactorIdx % scaleBFactor_ == 0) {
        scaleBGmOffset_ = nBL1Offset_ * CeilDiv<int64_t>(tiling_->kAlign, X2_SCALE_GROUP_SIZE) +
                          kBL1Size_ * scaleBFactor_ * (kFactorIdx / scaleBFactor_) / X2_SCALE_GROUP_SIZE;
        CopyScaleB2L1();
    }
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::CopyX2Gm2L1()
{
    int64_t nSizePadded = CeilAlign<int64_t>(tiling_->nSize, BLOCK_CUBE);
    int64_t x2GmOffset = (nBL1Offset_ * ONE_BLK_ELEM_B4 + kBL1Offset_ * nSizePadded) / 2;

    DataCopyExtParams dataCopyParams;
    DataCopyPadExtParams<uint8_t> padParams;

    dataCopyParams.blockCount = CeilDiv(kBL1Len_, ONE_BLK_ELEM_B4);
    dataCopyParams.blockLen = CeilAlign<int64_t>(nBL1Len_, BLOCK_CUBE) * ONE_BLK_SIZE;
    dataCopyParams.srcStride = (nSizePadded - CeilAlign<int64_t>(nBL1Len_, BLOCK_CUBE)) * ONE_BLK_SIZE;
    dataCopyParams.dstStride = 0;

    if (curBL1BufIdx_ == IDX_0) {
        DataCopyPad(bL1LocalBuf0_.template ReinterpretCast<uint8_t>(), wGlobalU8_[x2GmOffset], dataCopyParams,
                    padParams, true);
    } else {
        DataCopyPad(bL1LocalBuf1_.template ReinterpretCast<uint8_t>(), wGlobalU8_[x2GmOffset], dataCopyParams,
                    padParams, true);
    }
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::SetBias()
{
    if (biasIdx_ == IDX_0) {
        mmObj_.SetBias(biasL1Buf0_);
    } else {
        mmObj_.SetBias(biasL1Buf1_);
    }
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::IterateMatmul(
    int64_t kFactorIdx)
{
    // A不转置B转置
    mmObj_.SetOrgShape(CeilAlign<int64_t>(mAL1Len_, BLOCK_CUBE), CeilAlign<int64_t>(nBL1Len_, BLOCK_CUBE), kAL1Len_,
                       kBL1Len_, tiling_->nSize);

    mmObj_.SetTensorA(curAL1BufIdx_ == IDX_0 ? aL1LocalBuf0_ : aL1LocalBuf1_);

    mmObj_.SetTensorB(curBL1BufIdx_ == IDX_0 ? bL1LocalBuf0_ : bL1LocalBuf1_);

    if constexpr (IsSameType<x1ScaleType, AscendC::fp8_e8m0_t>::value) {
        if (scaleAIdx_ == IDX_0) {
            mmObj_.SetTensorScaleA(
                scaleAL1Buf0_[(kAL1Offset_ - (scaleAFactor_ * kAL1Size_) * (kFactorIdx / scaleAFactor_)) / 2]);
        } else {
            mmObj_.SetTensorScaleA(
                scaleAL1Buf1_[(kAL1Offset_ - (scaleAFactor_ * kAL1Size_) * (kFactorIdx / scaleAFactor_)) / 2]);
        }

        // B矩阵转置场景：bScaleL1Offset = ((kL1Offset * kFactorIdx / scaleBFactor) / group_size) * n0_size
        if (scaleBIdx_ == IDX_0) {
            mmObj_.SetTensorScaleB(
                scaleBL1Buf0_[(kBL1Offset_ - (scaleBFactor_ * kBL1Size_) * (kFactorIdx / scaleBFactor_)) /
                              X2_SCALE_GROUP_SIZE * BLOCK_CUBE]);
        } else {
            mmObj_.SetTensorScaleB(
                scaleBL1Buf1_[(kBL1Offset_ - (scaleBFactor_ * kBL1Size_) * (kFactorIdx / scaleBFactor_)) /
                              X2_SCALE_GROUP_SIZE * BLOCK_CUBE]);
        }
    }
    if (tiling_->matmulTiling.isBias) {
        SetBias();
    }
    mmObj_.SetTail(baseUseM_, baseUseN_, Min(kAL1Len_, kBL1Len_));
    mmObj_.Iterate(kFactorIdx != 0);
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline bool QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::IterMatmulOut()
{
    if (unlikely(isFirstIter_)) {
        curML0Idx_ = 0;
        curNL0Idx_ = 0;
        curML1Idx_ = 0;
        curNL1Idx_ = 0;
        isFirstIter_ = false;
        // 考虑尾核
        curStepM_ = (mIter_ - curML0Idx_) > tiling_->matmulTiling.stepM ? tiling_->matmulTiling.stepM :
                                                                          (mIter_ - curML1Idx_);
        curStepN_ = (nIter_ - curNL0Idx_) > tiling_->matmulTiling.stepN ? tiling_->matmulTiling.stepN :
                                                                          (nIter_ - curNL1Idx_);
    } else if (likely(tiling_->matmulTiling.iterateOrder == static_cast<int>(IterateOrder::ORDER_N))) {
        if (++curML0Idx_ >= curML1Idx_ + curStepM_) {
            curML0Idx_ = curML1Idx_;
            if (++curNL0Idx_ >= curNL1Idx_ + curStepN_) {
                curML1Idx_ += curStepM_;
                if (curNL0Idx_ >= nIter_ && curML1Idx_ >= mIter_) {
                    return false;
                }
                if (curML1Idx_ >= mIter_) {
                    curML1Idx_ = 0;
                    curNL1Idx_ += curStepN_;
                }
                UpdateIdxAndStep();
            }
        }
    } else {
        if (++curNL0Idx_ >= curNL1Idx_ + curStepN_) {
            curNL0Idx_ = curNL1Idx_;
            if (++curML0Idx_ >= curML1Idx_ + curStepM_) {
                curNL1Idx_ += curStepN_;
                if (curML0Idx_ >= mIter_ && curNL1Idx_ >= nIter_) {
                    return false;
                }
                if (curNL1Idx_ >= nIter_) {
                    curNL1Idx_ = 0;
                    curML1Idx_ += curStepM_;
                }
                UpdateIdxAndStep();
            }
        }
    }

    baseUseM_ = (curML0Idx_ + 1 == mIter_) ? tailM_ : tiling_->matmulTiling.baseM;
    baseUseN_ = (curNL0Idx_ + 1 == nIter_) ? tailN_ : tiling_->matmulTiling.baseN;
    mAL1Len_ = (curML1Idx_ + curStepM_ >= mIter_) ? tailL1M_ : mAL1Size_;
    nBL1Len_ = (curNL1Idx_ + curStepN_ >= nIter_) ? tailL1N_ : nBL1Size_;
    return true;
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::UpdateIdxAndStep()
{
    curML0Idx_ = curML1Idx_;
    curNL0Idx_ = curNL1Idx_;
    curStepM_ = (mIter_ - curML0Idx_) > tiling_->matmulTiling.stepM ? tiling_->matmulTiling.stepM :
                                                                      (mIter_ - curML1Idx_);
    curStepN_ = (nIter_ - curNL0Idx_) > tiling_->matmulTiling.stepN ? tiling_->matmulTiling.stepN :
                                                                      (nIter_ - curNL1Idx_);
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::CopyBias2L1()
{
    // 拷贝长度向上对齐到32字节
    int64_t biasCopyLen = CeilAlign<int64_t>(baseUseN_ * sizeof(biasType), 32) / sizeof(biasType);
    if (biasIdx_ == IDX_0) {
        AscendC::DataCopy(biasL1Buf0_, biasGlobal_[curNL0Idx_ * tiling_->matmulTiling.baseN + nBlockOffset_],
                          biasCopyLen, true);
    } else {
        AscendC::DataCopy(biasL1Buf1_, biasGlobal_[curNL0Idx_ * tiling_->matmulTiling.baseN + nBlockOffset_],
                          biasCopyLen, true);
    }
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::CopyScaleB2L1()
{
    int64_t scaleKBL1Size = kBL1Size_ * scaleBFactor_;
    if (kBL1Offset_ + scaleKBL1Size > tiling_->kSize) {
        scaleKBL1Size = tiling_->kSize - kBL1Offset_;
    }

    ScaleDnToNzParam x2ScaleDnToNzParam = {
        0, CeilDiv<int64_t>(kBL1Size_ * scaleBFactor_, X2_SCALE_GROUP_SIZE), CeilAlign<int64_t>(nBL1Len_, BLOCK_CUBE),
        CeilDiv<int64_t>(scaleKBL1Size, X2_SCALE_GROUP_SIZE), CeilDiv<int64_t>(tiling_->kAlign, X2_SCALE_GROUP_SIZE)};

    if (scaleBIdx_ == IDX_0) {
        CopyScaleDn2Nz<x2ScaleType>(scaleBL1Buf0_, x2ScaleGlobal_[scaleBGmOffset_], x2ScaleDnToNzParam);
    } else {
        CopyScaleDn2Nz<x2ScaleType>(scaleBL1Buf1_, x2ScaleGlobal_[scaleBGmOffset_], x2ScaleDnToNzParam);
    }

    SetFlag<HardEvent::MTE2_MTE1>(scaleBIdx_);
    WaitFlag<HardEvent::MTE2_MTE1>(scaleBIdx_);
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::PostProcess(
    int32_t kFactorIdx, AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM])
{
    if ((kFactorIdx + 1) % kBl1Factor_ == 0 || (kFactorIdx + 1) == kSingleCoreIterNum_) {
        curBL1BufIdx_ = (curBL1BufIdx_ + 1) % tiling_->BL1Pingpong;
    }
    // AL1 k非全载时，每次跨kAl1Factor_前一拍做同步，考虑mk合轴后开db时kAL1可能存在尾块，因此在最后一次k循环也要加同步
    // AL1 k全载时，仅当完成一次n方向iter后做同步
    if ((kFactorIdx + 1) % kAl1Factor_ == 0 || (kFactorIdx + 1) == kSingleCoreIterNum_) {
        SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[curAL1BufIdx_]);
        curAL1BufIdx_ = (curAL1BufIdx_ + 1) % tiling_->AL1Pingpong;
    }

    if (tiling_->matmulTiling.isBias && kFactorIdx == kSingleCoreIterNum_ - 1) {
        biasIdx_ = (biasIdx_ + 1) % biasPingPong_;
    }
    if ((kFactorIdx + 1) % scaleAFactor_ == 0 || (kFactorIdx + 1) == kSingleCoreIterNum_) {
        SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[scaleAIdx_]);
        scaleAIdx_ = (scaleAIdx_ + 1) % DOUBLE_BUFFER;
    }
    if ((kFactorIdx + 1) % scaleBFactor_ == 0 || (kFactorIdx + 1) == kSingleCoreIterNum_) {
        scaleBIdx_ = (scaleBIdx_ + 1) % DOUBLE_BUFFER;
    }
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::InitSync(
    AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM])
{
    for (int32_t idx = 0; idx < tiling_->AL1Pingpong; idx++) {
        eventIdsMte1ToMte2[idx] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
        SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[idx]);
    }

    eventIdsScaleAMte1ToMte2_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
    eventIdsScaleAMte1ToMte2_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
    eventIdsScaleBMte1ToMte2_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
    eventIdsScaleBMte1ToMte2_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
    SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[0]);
    SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[1]);
    SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[0]);
    SetFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[1]);
}

MX_CUBE_LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void QuantBatchMatmulV4MxCubeKernel<MX_CUBE_LOCAL_TEMPLATE_FUNC_PARAMS>::EndSync(
    AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM])
{
    for (int8_t idx = 0; idx < tiling_->AL1Pingpong; idx++) {
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[idx]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2[idx]);
    }

    WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[0]);
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[1]);
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[0]);
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsScaleAMte1ToMte2_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsScaleBMte1ToMte2_[1]);
}
} // namespace QuantBatchMatmulV4
