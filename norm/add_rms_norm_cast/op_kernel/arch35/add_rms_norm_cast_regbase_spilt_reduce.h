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
 * \file add_rms_norm_cast_regbase_split_reduce.h
 * \brief AddRmsNormCast regbase split reduce dim template.
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_REGBASE_SPLIT_REDUCE_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_REGBASE_SPLIT_REDUCE_H
#include "add_rms_norm_cast_regbase_common.h"

namespace AddRmsNormCast {
template <typename T_X>
class KernelAddRmsNormCastRegBaseSpiltReduce {
    static constexpr uint32_t BUFFER_NUM = 2;

public:
    __aicore__ inline KernelAddRmsNormCastRegBaseSpiltReduce(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y1, GM_ADDR y2, GM_ADDR rstd, GM_ADDR x,
                                const AddRmsNormCastRegbaseTilingData* tilingData)
    {
        numM_ = tilingData->numM;
        numN_ = tilingData->numN;
        baseN_ = tilingData->baseN;
        baseM_ = tilingData->baseM;
        baseNDtypeAlign_ = tilingData->baseNDtypeAlign;
        baseNReduceAlign_ = tilingData->baseNReduceAlign;
        powerSplit_ = tilingData->powerSplit;
        powerLoop_ = tilingData->powerLoop;
        mPerCore_ = tilingData->mPerCore;
        mLastCore_ = tilingData->mLastCore;
        epsilon_ = tilingData->epsilon;
        avgFactor_ = tilingData->avgFactor;

        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();

        CalBlockTail();
        InitBuffer(x1, x2, gamma, y1, y2, rstd, x);
    }

    __aicore__ inline void CalBlockTail()
    {
        mCore_ = blockIdx_ == (blockNum_ - 1) ? mLastCore_ : mPerCore_;
        nCnt_ = CeilDiv(numN_, baseN_);
        tailN_ = numN_ - (nCnt_ - 1) * baseN_;

        tailNDtypeAlign_ = CeilAlign(tailN_, AddRmsNormCast::BLOCK_SIZE / sizeof(T_X));
        baseNB8Align_ = CeilAlign(baseN_, AddRmsNormCast::B8_BLOCK_NUM);
        uint64_t reduceSumBufLen = baseNReduceAlign_ / (2 * V_LENGTH);
        reduceSumBufAlign_ = CeilAlign(reduceSumBufLen, AddRmsNormCast::B32_BLOCK_NUM);
    }

    __aicore__ inline void InitBuffer(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y1, GM_ADDR y2, GM_ADDR rstd,
                                      GM_ADDR x)
    {
        // Init GM
        uint64_t gmOffset = blockIdx_ * mPerCore_ * numN_;
        uint64_t gmLen = mCore_ * numN_;
        uint64_t rstdGmOffset = blockIdx_ * mPerCore_;
        x1Gm_.SetGlobalBuffer((__gm__ T_X*)x1 + gmOffset, gmLen);
        x2Gm_.SetGlobalBuffer((__gm__ T_X*)x2 + gmOffset, gmLen);
        y1Gm_.SetGlobalBuffer((__gm__ float*)y1 + gmOffset, gmLen);
        y2Gm_.SetGlobalBuffer((__gm__ T_X*)y2 + gmOffset, gmLen);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, gmLen);
        gammaGm_.SetGlobalBuffer((__gm__ T_X*)gamma, numN_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + rstdGmOffset, mCore_);

        // Init Buffer
        uint64_t ubFactorRstd = AddRmsNormCast::B32_BLOCK_NUM;
        pipe_->InitBuffer(inQueueX1_, BUFFER_NUM, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueX2_, BUFFER_NUM, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueY1_, BUFFER_NUM, baseNDtypeAlign_ * sizeof(float));
        pipe_->InitBuffer(outQueueY2_, BUFFER_NUM, baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueX_, BUFFER_NUM, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueGamma_, BUFFER_NUM, baseNDtypeAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueRstd_, 1, ubFactorRstd * sizeof(float));
        pipe_->InitBuffer(reduceSumBuf_, reduceSumBufAlign_ * sizeof(float));
        pipe_->InitBuffer(level1Buf_, RmsNorm::ONCE_VECTOR_SIZE * sizeof(float));
        pipe_->InitBuffer(level2Buf_, RmsNorm::ONCE_VECTOR_SIZE * sizeof(float));
        pipe_->InitBuffer(level3Buf_, RmsNorm::ONCE_VECTOR_SIZE * sizeof(float));
        pipe_->InitBuffer(tempBuf_, V_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint64_t powerMain = powerSplit_ * powerLoop_;
        uint64_t powerTail = numN_ - powerMain;
        uint64_t powerTailLoop = powerTail / powerSplit_;
        uint64_t powerTailTail = powerTail % powerSplit_;
        uint64_t masterLoop = powerTailTail != 0 ? 1 : 0;
        masterLoop = powerLoop_ - powerTailLoop - masterLoop;

        for (uint64_t mIdx = 0; mIdx < mCore_; mIdx++) {
            uint64_t xGmOffset = mIdx * numN_;

            // Reduce before writing x, which may share GM with either input.
            LocalTensor<float> rstdLocal = outQueueRstd_.AllocTensor<float>();
            SubProcessRstd<true>(rstdLocal, 0, xGmOffset, masterLoop, powerTailLoop, powerTailTail, avgFactor_,
                                 epsilon_);
            outQueueRstd_.EnQue(rstdLocal);
            rstdLocal = outQueueRstd_.DeQue<float>();
            CopyOutRstd(rstdLocal, mIdx, 1);
            // Recompute and copy out x/y from each original input block.
            SubProcessY(rstdLocal, mIdx);
            outQueueRstd_.FreeTensor(rstdLocal);
        }
    }

private:
    template <typename T>
    __aicore__ inline void CopyInX(TQue<QuePosition::VECIN, 1>& inQueueX, GlobalTensor<T>& srcGm, uint64_t gmOffset,
                                   uint32_t blockLen, uint32_t left = 0, uint32_t right = 0)
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{
            true,                        // isPad
            static_cast<uint8_t>(left),  // leftPadding
            static_cast<uint8_t>(right), // rightPadding
            static_cast<T>(0.0)          // paddingValue
        };
        RmsNorm::DataCopyImpl<T>(xLocal, srcGm[gmOffset], 1, blockLen, 0, 0, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGamma(uint64_t gmOffset, uint32_t blockLen)
    {
        LocalTensor<T_X> gammaLocal = inQueueGamma_.AllocTensor<T_X>();
        RmsNorm::DataCopyImpl<T_X>(gammaLocal, gammaGm_[gmOffset], 1, blockLen);
        inQueueGamma_.EnQue(gammaLocal);
    }

    template <typename T>
    __aicore__ inline void CopyOutXY(GlobalTensor<T>& yGm, TQue<QuePosition::VECOUT, 1>& outQueueY, uint64_t gmOffset,
                                     uint64_t blockLen)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        RmsNorm::DataCopyImpl<T>(yGm[gmOffset], yLocal, 1, blockLen);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(LocalTensor<float>& rstdLocal, uint64_t gmOffset, uint32_t num)
    {
        RmsNorm::DataCopyImpl<float>(rstdGm_[gmOffset], rstdLocal, 1, num, 0, 0);
    }

    /**
     * @brief Reduce the original x1/x2 block to dst[dstOffset] without writing x.
     */
    __aicore__ inline void ComputeFormerHandle(LocalTensor<float>& dstLocal, uint64_t srcGmOffset, uint64_t dstOffset,
                                               uint64_t count, uint64_t powerSplit)
    {
        uint32_t calCount = CeilAlign((uint64_t)(count * sizeof(T_X)), ALIGN_32_FACTOR) / sizeof(T_X);
        CopyInX(inQueueX1_, x1Gm_, srcGmOffset, count, 0, calCount - count);
        CopyInX(inQueueX2_, x2Gm_, srcGmOffset, count, 0, calCount - count);
        LocalTensor<T_X> x1Local = inQueueX1_.DeQue<T_X>();
        LocalTensor<T_X> x2Local = inQueueX2_.DeQue<T_X>();

        LocalTensor<float> workLocal = reduceSumBuf_.Get<float>();
        uint32_t calNum = CeilAlign((uint64_t)(count * sizeof(T_X)), ALIGN_512_FACTOR) / sizeof(T_X);
        uint64_t dupLen = calNum - calCount;
        if (dupLen > 0) {
            Duplicate(x1Local[calCount], (T_X)0.0, dupLen);
            Duplicate(x2Local[calCount], (T_X)0.0, dupLen);
        }

        NormCommon::ReduceSumRstd<T_X, false, false, false>(dstLocal, x1Local, workLocal, x1Local, x2Local, workLocal,
                                                            dstOffset, calNum, powerSplit);
        inQueueX1_.FreeTensor(x1Local);
        inQueueX2_.FreeTensor(x2Local);
    }

    template <bool IS_RSTD>
    __aicore__ inline void SubProcessRstd(LocalTensor<float>& dstLocal, uint64_t position, uint64_t curRow,
                                          uint64_t masterLoop, uint64_t tailLoop, uint64_t tail, float avgFactor = 1.0f,
                                          float epsilon = 0.0f)
    {
        uint64_t offset{curRow};
        uint32_t level1Idx{0};
        uint32_t level2Idx{0};
        uint32_t level3Idx{0};
        LocalTensor<float> level1Local = level1Buf_.Get<float>();
        LocalTensor<float> level2Local = level2Buf_.Get<float>();
        LocalTensor<float> level3Local = level3Buf_.Get<float>();
        LocalTensor<float> tempLocal = tempBuf_.Get<float>();
        Duplicate(level1Local, (float)0.0, RmsNorm::ONCE_VECTOR_SIZE);
        Duplicate(level2Local, (float)0.0, RmsNorm::ONCE_VECTOR_SIZE);
        Duplicate(level3Local, (float)0.0, RmsNorm::ONCE_VECTOR_SIZE);
        // Stage1: Cal TailLoop
        // Use TailBlock as MainBlock and reduce 2 nearby Mainblock twice.
        for (uint32_t repeat = 0; repeat < tailLoop; repeat++) {
            // Tail
            ComputeFormerHandle(tempLocal, offset, 0, powerSplit_, powerSplit_);
            offset += powerSplit_;
            // Main
            ComputeFormerHandle(tempLocal, offset, 1, powerSplit_, powerSplit_);
            offset += powerSplit_;

            RmsNorm::ComputeSum(level1Local, tempLocal, level1Idx, SUM_COUNT);
            level1Idx += 1;
            RmsNorm::ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1Idx, level2Idx, level3Idx);
        }
        // Stage2: Cal TailTail
        // Use TailBlock as MainBlock and reduce 2 nearby Mainblock twice.
        uint64_t powerSplitHalf = powerSplit_ / CONST_FACTOR_2;
        if (tail > 0 && tail <= powerSplitHalf) {
            // Tail
            ComputeFormerHandle(tempLocal, offset, 0, powerSplitHalf + tail, powerSplitHalf);
            offset += powerSplitHalf + tail;
            // Main
            ComputeFormerHandle(tempLocal, offset, 1, powerSplitHalf, powerSplitHalf);
            offset += powerSplitHalf;
        } else if (tail > powerSplitHalf) {
            // Half Main & Half Tail
            ComputeFormerHandle(tempLocal, offset, 0, powerSplit_, powerSplit_);
            offset += powerSplit_;
            // Other Tail
            ComputeFormerHandle(tempLocal, offset, 1, tail, powerSplitHalf);
            offset += tail;
        }
        if (tail > 0) {
            RmsNorm::ComputeSum(level1Local, tempLocal, level1Idx, SUM_COUNT);
            level1Idx += 1;
            RmsNorm::ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1Idx, level2Idx, level3Idx);
        }
        // Stage3: Cal MasterLoop
        for (uint32_t repeat = 0; repeat < masterLoop; repeat++) {
            ComputeFormerHandle(level1Local, offset, level1Idx, powerSplit_, powerSplit_);
            offset += powerSplit_;
            level1Idx += 1;
            RmsNorm::ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1Idx, level2Idx, level3Idx);
        }
        ComputeMultiLevelRstd<IS_RSTD>(dstLocal, position, level1Local, level2Local, level3Local, level1Idx, level2Idx,
                                       avgFactor, epsilon);
    }

    // Consume the recomputed fp32 sum in registers; no block or whole-row sum cache is needed.
    __aicore__ inline void ComputeRecomputeY(LocalTensor<T_X>& xLocal, LocalTensor<float>& y1Local,
                                             LocalTensor<T_X>& y2Local, LocalTensor<T_X>& x1Local,
                                             LocalTensor<T_X>& x2Local, LocalTensor<T_X>& gammaLocal,
                                             LocalTensor<float>& rstdLocal, uint32_t count)
    {
        __ubuf__ T_X* x1Addr = (__ubuf__ T_X*)x1Local.GetPhyAddr();
        __ubuf__ T_X* x2Addr = (__ubuf__ T_X*)x2Local.GetPhyAddr();
        __ubuf__ T_X* gammaAddr = (__ubuf__ T_X*)gammaLocal.GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __ubuf__ float* y1Addr = (__ubuf__ float*)y1Local.GetPhyAddr();
        __ubuf__ T_X* y2Addr = (__ubuf__ T_X*)y2Local.GetPhyAddr();
        __ubuf__ T_X* xAddr = (__ubuf__ T_X*)xLocal.GetPhyAddr();
        uint16_t repeats = CeilDivision(count, V_LENGTH);
        __VEC_SCOPE__
        {
            uint32_t remaining = count;
            RegTensor<T_X> x1Reg, x2Reg, y2Reg;
            RegTensor<float> sumReg, rstdReg, gammaReg, normalizedReg, y1Reg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr);
            for (uint16_t i = 0; i < repeats; ++i) {
                MaskReg mask = UpdateMask<float>(remaining);
                uint32_t offset = static_cast<uint32_t>(i) * V_LENGTH;
                if constexpr (IsSameType<T_X, float>::value) {
                    LoadAlign(x1Reg, x1Addr + offset);
                    LoadAlign(x2Reg, x2Addr + offset);
                } else {
                    LoadAlign<T_X, LoadDist::DIST_UNPACK_B16>(x1Reg, x1Addr + offset);
                    LoadAlign<T_X, LoadDist::DIST_UNPACK_B16>(x2Reg, x2Addr + offset);
                }
                NormCommon::CastAddVF(sumReg, x1Reg, x2Reg, mask);
                if constexpr (IsSameType<T_X, float>::value) {
                    StoreAlign(xAddr + offset, sumReg, mask);
                } else {
                    Cast<T_X, float, castTraitB322B16>(y2Reg, sumReg, mask);
                    StoreAlign<T_X, StoreDist::DIST_PACK_B32>(xAddr + offset, y2Reg, mask);
                }
                NormCommon::LoadCastRegVF(gammaReg, gammaAddr, i, mask);
                Mul(normalizedReg, sumReg, rstdReg, mask);
                Mul(y1Reg, normalizedReg, gammaReg, mask);
                StoreAlign(y1Addr + offset, y1Reg, mask);
                if constexpr (IsSameType<T_X, float>::value) {
                    StoreAlign(y2Addr + offset, y1Reg, mask);
                } else {
                    Cast<T_X, float, castTraitB322B16>(y2Reg, y1Reg, mask);
                    StoreAlign<T_X, StoreDist::DIST_PACK_B32>(y2Addr + offset, y2Reg, mask);
                }
            }
        }
    }

    __aicore__ inline void SubProcessY(LocalTensor<float>& rstdLocal, uint64_t mIdx)
    {
        for (uint64_t nIdx = 0; nIdx < nCnt_; nIdx++) {
            uint64_t realN = (nIdx == nCnt_ - 1) ? tailN_ : baseN_;
            CopyInGamma(nIdx * baseN_, realN);
            LocalTensor<T_X> gammaLocal = inQueueGamma_.DeQue<T_X>();

            uint64_t gmOffsetXY = mIdx * numN_ + nIdx * baseN_;
            // Load both original inputs before x can overwrite either GM block.
            CopyInX(inQueueX1_, x1Gm_, gmOffsetXY, realN);
            CopyInX(inQueueX2_, x2Gm_, gmOffsetXY, realN);
            LocalTensor<T_X> x1Local = inQueueX1_.DeQue<T_X>();
            LocalTensor<T_X> x2Local = inQueueX2_.DeQue<T_X>();

            LocalTensor<float> y1Local = outQueueY1_.AllocTensor<float>();
            LocalTensor<T_X> y2Local = outQueueY2_.AllocTensor<T_X>();
            LocalTensor<T_X> xLocal = outQueueX_.AllocTensor<T_X>();
            ComputeRecomputeY(xLocal, y1Local, y2Local, x1Local, x2Local, gammaLocal, rstdLocal, realN);

            inQueueX1_.FreeTensor(x1Local);
            inQueueX2_.FreeTensor(x2Local);

            inQueueGamma_.FreeTensor(gammaLocal);
            outQueueY1_.EnQue<float>(y1Local);
            outQueueY2_.EnQue<T_X>(y2Local);
            outQueueX_.EnQue<T_X>(xLocal);
            CopyOutXY(y1Gm_, outQueueY1_, gmOffsetXY, realN);
            CopyOutXY(y2Gm_, outQueueY2_, gmOffsetXY, realN);
            CopyOutXY(xGm_, outQueueX_, gmOffsetXY, realN);
        }
    }

private:
    TPipe* pipe_ = nullptr;
    // GM Buffer
    GlobalTensor<T_X> x1Gm_;
    GlobalTensor<T_X> x2Gm_;
    GlobalTensor<T_X> gammaGm_;
    GlobalTensor<float> y1Gm_;
    GlobalTensor<T_X> y2Gm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<T_X> xGm_;
    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX1_;
    TQue<QuePosition::VECIN, 1> inQueueX2_;
    TQue<QuePosition::VECIN, 1> inQueueGamma_;
    TQue<QuePosition::VECOUT, 1> outQueueY1_;
    TQue<QuePosition::VECOUT, 1> outQueueY2_;
    TQue<QuePosition::VECOUT, 1> outQueueRstd_;
    TQue<QuePosition::VECOUT, 1> outQueueX_;
    TBuf<TPosition::VECCALC> reduceSumBuf_;
    TBuf<TPosition::VECCALC> level1Buf_;
    TBuf<TPosition::VECCALC> level2Buf_;
    TBuf<TPosition::VECCALC> level3Buf_;
    TBuf<TPosition::VECCALC> tempBuf_;
    // Tiling data
    uint64_t numM_{0};
    uint64_t numN_{0};
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t baseNDtypeAlign_{0};
    uint64_t tailNDtypeAlign_{0};
    uint64_t baseNReduceAlign_{0};
    uint64_t reduceSumBufAlign_{0};
    uint64_t powerSplit_{0};
    uint64_t powerLoop_{0};
    uint64_t mPerCore_{0};
    uint64_t mLastCore_{0};
    float epsilon_{0};
    float avgFactor_{0};
    // Platform
    int64_t blockIdx_{0};
    int64_t blockNum_{0};
    // Cal params
    uint64_t mCore_;
    uint64_t nCnt_;
    uint64_t tailN_;
    uint64_t baseNB8Align_;
    // Other
    uint32_t vfLength_{0};
};
} // namespace AddRmsNormCast
#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_REGBASE_SPLIT_REDUCE_H
