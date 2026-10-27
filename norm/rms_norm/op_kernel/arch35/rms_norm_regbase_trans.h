/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMS_NORM_REGBASE_TRANS_H
#define RMS_NORM_REGBASE_TRANS_H
#include "rms_norm_regbase_common.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace RmsNorm {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

constexpr int32_t TRANS_COPY_DIM = 2; // NdDma 的二维 [A,R] 布局。
constexpr int32_t TRANS_B16_PER_B32 = sizeof(float) / sizeof(uint16_t);
// TransDataTo5HD 的 src/dst 地址表长度由接口规定；FP32 每个元素占两个 B16 地址槽。
constexpr int32_t TRANS_ADDR_LIST_SIZE = NCHW_CONV_ADDR_LIST_SIZE;
constexpr int32_t TRANS_FP32_ADDR_GROUP_SIZE = TRANS_ADDR_LIST_SIZE / TRANS_B16_PER_B32;

/**
 * 小 R 转置模板：将 [A,R] 搬成 UB [R,A] 后沿 A 向量化，计算完成再转回；
 * 每核可循环多个 A tile，性能准入由 Host 决定。
 */
template <typename T, typename G>
class KernelRmsNormRegBaseTrans {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
    static constexpr uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);

public:
    __aicore__ inline explicit KernelRmsNormRegBaseTrans(TPipe* pipe) { pPipe = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,
                                const RMSNormRegbaseTransTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        numRow = tiling->numRow;
        numCol = tiling->numCol;
        rAligned = tiling->rAligned;
        rTileBase = tiling->rTileBase;
        tileALen = tiling->tileALen;
        epsilon = tiling->epsilon;
        avgFactor = tiling->avgFactor;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        gammaGm.SetGlobalBuffer((__gm__ G*)gamma);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd);

        // 二维 buffer 按最大 A tile 分配，输入与转置输出各使用两个物理槽位。
        uint64_t xShapeLen = tileALen * rAligned;
        pPipe->InitBuffer(xQueue_, BUFFER_NUM, xShapeLen * sizeof(T));
        pPipe->InitBuffer(yQueue_, BUFFER_NUM, xShapeLen * sizeof(T));
        // gamma 每核只搬入一次，在输出计算 VF 内转换为 FP32。
        pPipe->InitBuffer(gammaQueue_, 1, rAligned * sizeof(G));
        // 保留第一遍转换后的 FP32 x，第二遍直接复用。
        pPipe->InitBuffer(xFp32Buf_, xShapeLen * sizeof(float));
        // rstdQueue_ 管理跨 tile 的 V→MTE3 生命周期。
        pPipe->InitBuffer(rstdQueue_, BUFFER_NUM, tileALen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t usedCoreNum = GetBlockNum();
        // 商余法把连续 A 行均衡分核。
        int64_t baseRowsPerCore = static_cast<int64_t>(numRow) / usedCoreNum;
        int64_t extraRowCoreNum = static_cast<int64_t>(numRow) % usedCoreNum;
        int64_t beginRow = blockIdx * baseRowsPerCore + (blockIdx < extraRowCoreNum ? blockIdx : extraRowCoreNum);
        int64_t coreRowNum = baseRowsPerCore + (blockIdx < extraRowCoreNum ? 1 : 0);
        int64_t endRow = beginRow + coreRowNum;

        // gamma 在本核所有 A tile 间复用。
        CopyInGamma();
        LocalTensor<G> gammaLocal = gammaQueue_.DeQue<G>();

        // 首 tile 预取，后续迭代复用双缓冲队列。
        uint32_t firstTileALen = static_cast<uint32_t>(
            coreRowNum < static_cast<int64_t>(tileALen) ? coreRowNum : static_cast<int64_t>(tileALen));
        int64_t firstXOffset = beginRow * numCol;
        // NDDMA 首次访问 GM 前每核刷新一次 DataCache，后续 tile 无需重复刷新。
        NdDmaDci();
        CopyInX(firstXOffset, firstTileALen);

        uint64_t coreTileCount = static_cast<uint64_t>(coreRowNum) / tileALen +
                                 static_cast<uint64_t>(static_cast<uint64_t>(coreRowNum) % tileALen != 0);
        for (uint64_t tileIdx = 0; tileIdx < coreTileCount; ++tileIdx) {
            int64_t curRow = beginRow + static_cast<int64_t>(tileIdx * tileALen);
            uint32_t curTileALen = static_cast<uint32_t>(
                endRow - curRow < static_cast<int64_t>(tileALen) ? endRow - curRow : static_cast<int64_t>(tileALen));
            int64_t nextRow = curRow + curTileALen;
            bool hasNextTile = nextRow < endRow;
            int64_t nextRowNum = endRow - nextRow;
            uint32_t nextTileALen = static_cast<uint32_t>(
                nextRowNum < static_cast<int64_t>(tileALen) ? nextRowNum : static_cast<int64_t>(tileALen));
            int64_t nextXOffset = nextRow * numCol;
            int64_t xOffset = curRow * numCol;
            Compute(gammaLocal, curTileALen, hasNextTile, nextXOffset, nextTileALen);
            CopyOutY(xOffset, curTileALen);
            CopyOutRstd(curRow, curTileALen);
        }
        gammaQueue_.FreeTensor(gammaLocal);
    }

private:
    __aicore__ inline void CopyInGamma()
    {
        // DataCopyPad 仅搬真实 R，UB 对齐区不参与计算。
        LocalTensor<G> gammaLocal = gammaQueue_.AllocTensor<G>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = numCol * sizeof(G);
        DataCopyPadExtParams<G> padParams;
        padParams.isPad = false;
        DataCopyPad(gammaLocal, gammaGm, params, padParams);
        gammaQueue_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyInX(int64_t xGmOffset, uint32_t curTileALen)
    {
        // NdDma 将 GM [A,R] 搬为 UB [R,A]。
        static constexpr NdDmaConfig config = {false};
        NdDmaLoopInfo<TRANS_COPY_DIM> copyLoopInfo;
        copyLoopInfo.loopSrcStride[0] = 1;
        copyLoopInfo.loopSrcStride[1] = numCol;
        copyLoopInfo.loopDstStride[0] = tileALen;
        copyLoopInfo.loopDstStride[1] = 1;
        copyLoopInfo.loopSize[0] = numCol;
        copyLoopInfo.loopSize[1] = curTileALen;
        NdDmaParams<T, TRANS_COPY_DIM> params = {copyLoopInfo, 0};

        LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
        DataCopy<T, TRANS_COPY_DIM, config>(xLocal, xGm[xGmOffset], params);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(const LocalTensor<G>& gammaLocal, uint32_t curTileALen, bool hasNextTile,
                                   int64_t nextXOffset, uint32_t nextTileALen)
    {
        // 第一遍保存 FP32 x 并计算 rstd，第二遍复用该中间结果生成 y。
        LocalTensor<T> xLocal = xQueue_.DeQue<T>();
        LocalTensor<float> xFp32Local = xFp32Buf_.Get<float>();
        LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
        LocalTensor<float> rstdLocal = rstdQueue_.AllocTensor<float>();

        __ubuf__ T* xInUb = (__ubuf__ T*)xLocal.GetPhyAddr();
        __ubuf__ float* xFp32InUb = (__ubuf__ float*)xFp32Local.GetPhyAddr();
        __ubuf__ T* yInUb = (__ubuf__ T*)yLocal.GetPhyAddr();
        __ubuf__ float* rstdTmp = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __ubuf__ G* gammaInUb = (__ubuf__ G*)gammaLocal.GetPhyAddr();

        uint32_t sreg = curTileALen;
        uint16_t aChunks = (sreg + VL_FP32 - 1) / VL_FP32;
        uint16_t numRows = static_cast<uint16_t>(numCol);
        // UpdateMask 会递减 maskCount，必须跨 A chunk 持久保存。
        uint32_t maskCount = sreg;

        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> sqReg;
            RegTensor<float> sumReg;
            RegTensor<float> rstdReg;
            MaskReg pregLoop;
            for (uint16_t a = 0; a < aChunks; ++a) {
                uint32_t aOff = a * VL_FP32;
                pregLoop = UpdateMask<float>(maskCount);
                Duplicate(sumReg, float(0.0), pregLoop);
                for (uint16_t r = 0; r < numRows; ++r) {
                    uint32_t rowOff = r * tileALen + aOff;
                    LoadRegForDtype<T>(xInUb, xReg, pregLoop, rowOff);
                    StoreAlign<float, StoreDist::DIST_NORM_B32>(xFp32InUb + rowOff, xReg, pregLoop);
                    Mul(sqReg, xReg, xReg, pregLoop);
                    Add(sumReg, sumReg, sqReg, pregLoop);
                }
                Muls(sumReg, sumReg, avgFactor, pregLoop);
                NormCommon::ComputeRstdNewtonRaphsonReg<true>(sumReg, rstdReg, pregLoop, epsilon);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(rstdTmp + aOff, rstdReg, pregLoop);
            }
        }

        xQueue_.FreeTensor(xLocal);
        // 输入槽位释放后预取下一 tile。
        if (hasNextTile) {
            CopyInX(nextXOffset, nextTileALen);
        }

        uint32_t outputMaskCount = sreg;
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> gammaReg;
            RegTensor<float> yReg;
            RegTensor<float> rstdReg;
            MaskReg pregLoop;
            for (uint16_t a = 0; a < aChunks; ++a) {
                uint32_t aOff = a * VL_FP32;
                pregLoop = UpdateMask<float>(outputMaskCount);
                LoadAlign(rstdReg, rstdTmp + aOff);
                for (uint16_t r = 0; r < numRows; ++r) {
                    uint32_t rowOff = r * tileALen + aOff;
                    LoadAlign(xReg, xFp32InUb + rowOff);
                    LoadRegForDtype<G, LoadDist::DIST_BRC_B32, LoadDist::DIST_BRC_B16>(gammaInUb, gammaReg, pregLoop,
                                                                                       r);
                    Mul(yReg, xReg, rstdReg, pregLoop);
                    Mul(yReg, yReg, gammaReg, pregLoop);
                    StoreRegForDtype<T>(yInUb, yReg, pregLoop, rowOff);
                }
            }
        }

        yQueue_.EnQue<T>(yLocal);
        rstdQueue_.EnQue<float>(rstdLocal);
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, uint32_t curTileALen)
    {
        LocalTensor<T> yLocal = yQueue_.DeQue<T>();
        // 将 y 从 [R,A] 转回 [A,R]，只搬真实 A/R。
        LocalTensor<T> yTrans = yQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            CalcTransposeFp32(yLocal, yTrans, curTileALen);
        } else {
            CalcTransposeFp16(yLocal, yTrans, curTileALen);
        }
        yQueue_.FreeTensor(yLocal);
        yQueue_.EnQue<T>(yTrans);
        LocalTensor<T> yTransOut = yQueue_.DeQue<T>();
        DataCopyExtParams params;
        params.blockCount = curTileALen;
        params.blockLen = numCol * sizeof(T);
        uint64_t srcAlignLen = (numCol * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        params.srcStride = (rAligned * sizeof(T) - srcAlignLen) / BLOCK_SIZE;
        params.dstStride = 0;
        DataCopyPad(yGm[yGmOffset], yTransOut, params);
        yQueue_.FreeTensor(yTransOut);
    }

    // FP32 每个元素占两个 B16 地址槽。
    __aicore__ inline void CalcTransposeFp32(const LocalTensor<T>& src, LocalTensor<T>& dst, uint32_t curTileALen)
    {
        int rRepeatTimes = ops::CeilDiv(static_cast<int64_t>(rAligned),
                                        static_cast<int64_t>(TRANS_FP32_ADDR_GROUP_SIZE));
        for (int i = 0; i < rRepeatTimes; i++) {
            TransDataTo5HDParams params;
            LocalTensor<T> srcLocalList[TRANS_ADDR_LIST_SIZE];
            LocalTensor<T> dstLocalList[TRANS_ADDR_LIST_SIZE];

            uint32_t aRepeatTimes = ops::CeilDiv(curTileALen, uint32_t(TRANS_ADDR_LIST_SIZE));
            params.repeatTimes = aRepeatTimes;
            params.srcRepStride = aRepeatTimes == 1 ? 0 : TRANS_B16_PER_B32;
            params.dstRepStride = aRepeatTimes == 1 ? 0 : TRANS_ADDR_LIST_SIZE * rRepeatTimes;
            for (int j = 0; j < TRANS_FP32_ADDR_GROUP_SIZE; j++) {
                uint32_t offset = TRANS_FP32_ADDR_GROUP_SIZE * tileALen * i + tileALen * j;
                srcLocalList[j] = src[offset];
                srcLocalList[j + TRANS_FP32_ADDR_GROUP_SIZE] = src[offset + TRANS_FP32_ADDR_GROUP_SIZE];
            }
            for (int j = 0; j < TRANS_FP32_ADDR_GROUP_SIZE; j++) {
                uint32_t offset = TRANS_FP32_ADDR_GROUP_SIZE * i + TRANS_FP32_ADDR_GROUP_SIZE * rRepeatTimes * j;
                dstLocalList[j * TRANS_B16_PER_B32] = dst[offset];
                dstLocalList[j * TRANS_B16_PER_B32 +
                             1] = dst[offset + TRANS_FP32_ADDR_GROUP_SIZE * TRANS_FP32_ADDR_GROUP_SIZE * rRepeatTimes];
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, params);
        }
    }

    __aicore__ inline void CalcTransposeFp16(const LocalTensor<T>& src, LocalTensor<T>& dst, uint32_t curTileALen)
    {
        // FP16/BF16 每个元素占一个 B16 地址槽。
        int rRepeatTimes = ops::CeilDiv(static_cast<int64_t>(rAligned), static_cast<int64_t>(rTileBase));
        uint32_t aRepeatTimes = ops::CeilDiv(curTileALen, static_cast<uint32_t>(rTileBase));
        for (int i = 0; i < rRepeatTimes; i++) {
            TransDataTo5HDParams params;
            params.repeatTimes = aRepeatTimes;
            params.srcRepStride = aRepeatTimes == 1 ? 0 : 1;
            params.dstRepStride = aRepeatTimes == 1 ? 0 : (rTileBase * rRepeatTimes);
            LocalTensor<T> srcLocalList[TRANS_ADDR_LIST_SIZE];
            LocalTensor<T> dstLocalList[TRANS_ADDR_LIST_SIZE];
            for (int j = 0; j < TRANS_ADDR_LIST_SIZE; j++) {
                uint32_t offset = rTileBase * tileALen * i + tileALen * j;
                srcLocalList[j] = src[offset];
            }
            for (int j = 0; j < TRANS_ADDR_LIST_SIZE; j++) {
                uint32_t offset = rTileBase * i + rAligned * j;
                dstLocalList[j] = dst[offset];
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, params);
        }
    }

    __aicore__ inline void CopyOutRstd(int64_t rstdOffset, uint32_t curTileALen)
    {
        // rstd 已按 A 连续，直接搬真实行数。
        LocalTensor<float> rstdLocal = rstdQueue_.DeQue<float>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = curTileALen * sizeof(float);
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPad(rstdGm[rstdOffset], rstdLocal, params);
        rstdQueue_.FreeTensor(rstdLocal);
    }

private:
    TPipe* pPipe;

    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> gammaQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TQue<QuePosition::VECOUT, 1> rstdQueue_;
    TBuf<TPosition::VECCALC> xFp32Buf_;

    GlobalTensor<T> xGm;
    GlobalTensor<G> gammaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> rstdGm;

    uint64_t numRow;
    uint64_t numCol;
    uint64_t rAligned;
    uint64_t rTileBase;
    uint64_t tileALen;
    float epsilon;
    float avgFactor;
};
} // namespace RmsNorm
#endif // RMS_NORM_REGBASE_TRANS_H
