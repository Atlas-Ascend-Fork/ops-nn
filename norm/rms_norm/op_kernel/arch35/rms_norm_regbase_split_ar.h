/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMS_NORM_REGBASE_SPLIT_AR_H
#define RMS_NORM_REGBASE_SPLIT_AR_H
#include <limits>
#include "rms_norm_regbase_common.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace RmsNorm {
using namespace AscendC;

// 每层 cache 占一个 UB block，层数覆盖 uint64_t 计数范围。
constexpr int64_t AR_RECOMPUTE_SUM_LEN = platform::GetUbBlockSize() / sizeof(float);
constexpr int64_t AR_VECTOR_LEN = platform::GetVRegSize() / sizeof(float);
constexpr uint16_t AR_CACHE_LEVEL_COUNT = static_cast<uint16_t>(std::numeric_limits<uint64_t>::digits);
// 非 2 次幂 tile 数补齐到下一个 2 次幂的倍增因子。
constexpr uint64_t SPLIT_AR_NEXT_POW2_FACTOR = 2;

// 商余式向上取整避免 value + divisor - 1 在大 R 下溢出整数范围。
__aicore__ inline uint64_t SplitARCeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

// 返回严格小于 value 的最大 2 的幂。
__aicore__ inline uint64_t GetReduceFoldPoint(uint64_t value)
{
    if (value <= 1) {
        return 0;
    }
    uint64_t reduced = value - 1;
    uint64_t power = std::numeric_limits<uint64_t>::digits - 1 - AscendC::ScalarCountLeadingZero(reduced);
    return static_cast<uint64_t>(1) << power;
}

// 返回二进制计数器插入第 idx+1 个 partial 时的进位层。
__aicore__ inline uint16_t GetCacheId(uint64_t idx)
{
    return static_cast<uint16_t>(AscendC::ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1);
}

// 在线合并被进位覆盖的低层 partial。
__aicore__ inline void UpdateCache(__ubuf__ float* cachePtr, __ubuf__ float* srcPtr, uint16_t cacheId)
{
    __VEC_SCOPE__
    {
        RegTensor<float> aReg;
        RegTensor<float> bReg;
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        LoadAlign<float, LoadDist::DIST_BRC_B32>(aReg, srcPtr);
        for (uint16_t j = 0; j < cacheId; ++j) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(bReg, cachePtr + j * AR_RECOMPUTE_SUM_LEN);
            AscendC::MicroAPI::Add(aReg, aReg, bReg, pregOne);
        }
        StoreAlign(cachePtr + cacheId * AR_RECOMPUTE_SUM_LEN, aReg, pregOne);
    }
}

__aicore__ inline void CopyPartial(__ubuf__ float* dstPtr, __ubuf__ float* srcPtr)
{
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg;
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        LoadAlign<float, LoadDist::DIST_BRC_B32>(srcReg, srcPtr);
        StoreAlign(dstPtr, srcReg, pregOne);
    }
}

__aicore__ inline void ClearPartialVector(__ubuf__ float* dstPtr, uint32_t count)
{
    uint16_t loopCount = static_cast<uint16_t>(SplitARCeilDiv(count, AR_VECTOR_LEN));
    uint32_t maskCount = count;
    __VEC_SCOPE__
    {
        RegTensor<float> zeroReg;
        MaskReg pregLoop;
        for (uint16_t i = 0; i < loopCount; ++i) {
            pregLoop = UpdateMask<float>(maskCount);
            Duplicate(zeroReg, 0.0f, pregLoop);
            StoreAlign(dstPtr + i * AR_VECTOR_LEN, zeroReg, pregLoop);
        }
    }
}

// srcPtr 按 block 对齐，dstPtr 允许只偏移一个 float。
__aicore__ inline void StorePartialElement(__ubuf__ float* dstPtr, __ubuf__ float* srcPtr)
{
    __VEC_SCOPE__
    {
        RegTensor<float> srcReg;
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        LoadAlign<float, LoadDist::DIST_BRC_B32>(srcReg, srcPtr);
        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr, srcReg, pregOne);
    }
}

/**
 * SplitAR 每核固定一个 R 分块并遍历全部 A 行。Pass1 写 [核,A] partial，SyncAll 后
 * 每核归并得到全部 rstd；Pass2 按 R tile 复用 gamma。ubFactor 循环支持任意大 R。
 */
template <typename T, typename G>
class KernelRmsNormRegBaseSplitAR {
    static constexpr uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
    static constexpr uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);
    static constexpr uint32_t BLK_FP32 = BLOCK_SIZE / sizeof(float);

public:
    __aicore__ inline explicit KernelRmsNormRegBaseSplitAR(TPipe* pipe) { pPipe = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR userWorkspace,
                                const RMSNormRegbaseSplitARTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        numRow = tiling->numRow;
        numCol = tiling->numCol;
        rBlockFactor = tiling->rBlockFactor;
        rBlockNum = tiling->rBlockNum;
        tailR = tiling->tailR;
        ubFactor = tiling->ubFactor;
        numRowAlign = tiling->numRowAlign;
        epsilon = tiling->epsilon;
        avgFactor = tiling->avgFactor;
        rBlockIdx = GetBlockIdx();
        curRStart = rBlockIdx * rBlockFactor;
        curRLen = (rBlockIdx == rBlockNum - 1) ? tailR : rBlockFactor;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        gammaGm.SetGlobalBuffer((__gm__ G*)gamma + curRStart, curRLen);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd, numRow);
        wsGm.SetGlobalBuffer((__gm__ float*)userWorkspace);

        // 输入、gamma 和输出使用双缓冲，重叠相邻 tile 的搬运与计算。
        pPipe->InitBuffer(xQueue, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
        pPipe->InitBuffer(gammaQueue, DOUBLE_BUFFER_NUM, ubFactor * sizeof(G));
        pPipe->InitBuffer(yQueue, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
        pPipe->InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        // ReduceSum<RA> 的源起址及可写尾部均按 VL 对齐。
        uint64_t combElementCount = SplitARCeilDiv(rBlockNum * numRowAlign, VL_FP32) * VL_FP32;
        pPipe->InitBuffer(combQueue, 1, combElementCount * sizeof(float));
        pPipe->InitBuffer(partQueue, 1, numRowAlign * sizeof(float));
        pPipe->InitBuffer(cacheBuf, AR_RECOMPUTE_SUM_LEN * AR_CACHE_LEVEL_COUNT * sizeof(float));
        pPipe->InitBuffer(workBuf, (ubFactor + VL_FP32) * sizeof(float));
        pPipe->InitBuffer(rstdQueue, 1, numRowAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        DoPass1PartialReduce();
        LocalTensor<float> part = partQueue.DeQue<float>();
        WritePartial(part);
        // 跨核 workspace 写后读同步。
        SyncAll<true>();
        ComputeCrossCoreRstd(part);
        LocalTensor<float> rstdLocal = rstdQueue.DeQue<float>();
        if (rBlockIdx == 0) {
            CopyOutRstd(rstdLocal);
        }
        DoPass2Output(rstdLocal);
        rstdQueue.FreeTensor(rstdLocal);
        partQueue.FreeTensor(part);
    }

private:
    __aicore__ inline void DoPass1PartialReduce()
    {
        LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
        LocalTensor<float> part = partQueue.AllocTensor<float>();
        LocalTensor<float> workLocal = workBuf.Get<float>();
        LocalTensor<float> cache = cacheBuf.Get<float>();
        __ubuf__ float* cachePtr = (__ubuf__ float*)cache.GetPhyAddr();
        __ubuf__ float* partPtr = (__ubuf__ float*)part.GetPhyAddr();
        __ubuf__ float* tilePartPtr = (__ubuf__ float*)workLocal.GetPhyAddr() + ubFactor;
        LocalTensor<float> tilePart = workLocal[ubFactor];
        ClearPartialVector(partPtr, static_cast<uint32_t>(numRowAlign));

        uint64_t tileCount = SplitARCeilDiv(curRLen, ubFactor);
        uint64_t paddedTileCount = tileCount;
        if ((tileCount & (tileCount - 1)) != 0) {
            paddedTileCount = GetReduceFoldPoint(tileCount) * SPLIT_AR_NEXT_POW2_FACTOR;
        }

        // Pass1 先遍历 A，再遍历当前行的 R tile。
        for (uint64_t row = 0; row < numRow; ++row) {
            for (uint64_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
                uint64_t tileOffset = tileIdx * ubFactor;
                uint32_t curTileLen = static_cast<uint32_t>(curRLen - tileOffset < ubFactor ? curRLen - tileOffset :
                                                                                              ubFactor);
                uint64_t foldPoint = curTileLen == ubFactor ? ubFactor / 2 : GetReduceFoldPoint(curTileLen);
                ProcessPass1Tile(row, tileOffset, curTileLen, foldPoint, xFp32, tilePart, workLocal);
                UpdateCache(cachePtr, tilePartPtr, GetCacheId(tileIdx));
            }
            // 追加零 partial 到下一次幂，直接形成唯一根节点，避免额外的 cache 扫描归并。
            if (paddedTileCount != tileCount) {
                ClearPartialVector(tilePartPtr, static_cast<uint32_t>(AR_RECOMPUTE_SUM_LEN));
                for (uint64_t tileIdx = tileCount; tileIdx < paddedTileCount; ++tileIdx) {
                    UpdateCache(cachePtr, tilePartPtr, GetCacheId(tileIdx));
                }
            }
            uint64_t resultCacheOffset = GetCacheId(paddedTileCount - 1) * AR_RECOMPUTE_SUM_LEN;
            CopyPartial(tilePartPtr, cachePtr + resultCacheOffset);
            StorePartialElement(partPtr + row, tilePartPtr);
        }
        partQueue.EnQue(part);
    }

    __aicore__ inline void ProcessPass1Tile(uint64_t row, uint64_t tileOffset, uint32_t curTileLen, uint64_t foldPoint,
                                            LocalTensor<float>& xFp32, LocalTensor<float>& part,
                                            LocalTensor<float>& workLocal)
    {
        CopyInX(row, tileOffset, curTileLen);
        ConvertInput(xFp32, curTileLen);
        NormCommon::NormCommonRegbase::CalculateSquareReduceSum<float>(xFp32, part, workLocal, 1,
                                                                       static_cast<uint32_t>(ubFactor), curTileLen,
                                                                       static_cast<uint32_t>(foldPoint), BLK_FP32);
    }

    __aicore__ inline void WritePartial(const LocalTensor<float>& part)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(numRowAlign * sizeof(float)), 0, 0, 0};
        DataCopyPad(wsGm[GetBlockIdx() * numRowAlign], part, params);
    }

    __aicore__ inline void CopyInX(uint64_t row, uint64_t tileOffset, uint32_t curTileLen)
    {
        LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(curTileLen * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        uint64_t gmOffset = row * numCol + curRStart + tileOffset;
        DataCopyPad(xLocal, xGm[gmOffset], params, pad);
        xQueue.EnQue(xLocal);
    }

    __aicore__ inline void ConvertInput(LocalTensor<float>& xFp32, uint32_t curTileLen)
    {
        LocalTensor<T> xLocal = xQueue.DeQue<T>();
        __ubuf__ T* xPtr = (__ubuf__ T*)xLocal.GetPhyAddr();
        __ubuf__ float* xFp32Ptr = (__ubuf__ float*)xFp32.GetPhyAddr();
        uint16_t loopCount = static_cast<uint16_t>(SplitARCeilDiv(curTileLen, VL_FP32));
        uint32_t maskCount = curTileLen;
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * VL_FP32;
                pregLoop = UpdateMask<float>(maskCount);
                LoadRegForDtype<T>(xPtr, xReg, pregLoop, offset);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(xFp32Ptr + offset, xReg, pregLoop);
            }
        }
        xQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeRstdFromSum(LocalTensor<float>& sumLocal)
    {
        LocalTensor<float> rstdLocal = rstdQueue.AllocTensor<float>();
        NormCommon::ComputeRstdNewtonRaphson<true, true>(sumLocal, rstdLocal, static_cast<uint32_t>(numRow), epsilon,
                                                         avgFactor, VL_FP32);
        rstdQueue.EnQue(rstdLocal);
    }

    __aicore__ inline void CopyOutRstd(const LocalTensor<float>& rstdLocal)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(numRow * sizeof(float)), 0, 0, 0};
        DataCopyPad(rstdGm, rstdLocal, params);
    }

    __aicore__ inline void ComputeCrossCoreRstd(LocalTensor<float>& part)
    {
        // 每核读回 [rBlockNum,numRowAlign]，沿核维归并全部 A 行。
        LocalTensor<float> comb = combQueue.AllocTensor<float>();
        DataCopyExtParams cpr{1, static_cast<uint32_t>(rBlockNum * numRowAlign * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padr{false, 0, 0, 0};
        DataCopyPad(comb, wsGm, cpr, padr);

        combQueue.EnQue(comb);
        comb = combQueue.DeQue<float>();

        uint32_t srcShape[2] = {static_cast<uint32_t>(rBlockNum), static_cast<uint32_t>(numRowAlign)};
        ReduceSum<float, Pattern::Reduce::RA, true>(part, comb, srcShape, false);
        combQueue.FreeTensor(comb);
        ComputeRstdFromSum(part);
    }

    __aicore__ inline void DoPass2Output(const LocalTensor<float>& rstdLocal)
    {
        __ubuf__ float* rstdPtr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        uint64_t tileCount = SplitARCeilDiv(curRLen, ubFactor);
        // Pass2 先遍历 R tile，再遍历 A，以跨行复用 gamma。
        for (uint64_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
            uint64_t tileOffset = tileIdx * ubFactor;
            uint32_t curTileLen = static_cast<uint32_t>(curRLen - tileOffset < ubFactor ? curRLen - tileOffset :
                                                                                          ubFactor);
            CopyInGamma(tileOffset, curTileLen);
            LocalTensor<G> gammaLocal = gammaQueue.DeQue<G>();
            for (uint64_t row = 0; row < numRow; ++row) {
                CopyInX(row, tileOffset, curTileLen);
                LocalTensor<T> xLocal = xQueue.DeQue<T>();
                LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
                ComputeOutput(xLocal, gammaLocal, yLocal, rstdPtr + row, curTileLen);
                xQueue.FreeTensor(xLocal);
                yQueue.EnQue(yLocal);
                CopyOutY(row, tileOffset, curTileLen);
            }
            gammaQueue.FreeTensor(gammaLocal);
        }
    }

    __aicore__ inline void CopyInGamma(uint64_t tileOffset, uint32_t curTileLen)
    {
        LocalTensor<G> gammaLocal = gammaQueue.AllocTensor<G>();
        DataCopyExtParams params{1, static_cast<uint32_t>(curTileLen * sizeof(G)), 0, 0, 0};
        DataCopyPadExtParams<G> pad{false, 0, 0, 0};
        DataCopyPad(gammaLocal, gammaGm[tileOffset], params, pad);
        gammaQueue.EnQue(gammaLocal);
    }

    __aicore__ inline void ComputeOutput(const LocalTensor<T>& xLocal, const LocalTensor<G>& gammaLocal,
                                         LocalTensor<T>& yLocal, __ubuf__ float* rstdPtr, uint32_t curTileLen)
    {
        __ubuf__ T* xPtr = (__ubuf__ T*)xLocal.GetPhyAddr();
        __ubuf__ G* gammaPtr = (__ubuf__ G*)gammaLocal.GetPhyAddr();
        __ubuf__ T* yPtr = (__ubuf__ T*)yLocal.GetPhyAddr();
        uint16_t loopCount = static_cast<uint16_t>((curTileLen + VL_FP32 - 1) / VL_FP32);
        // 将本核 rstd 广播后，以 FP32 计算并按 T 写回。
        uint32_t maskCount = curTileLen;

        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> gammaReg;
            RegTensor<float> yReg;
            RegTensor<float> rstdReg;
            MaskReg pregLoop;
            AddrReg xAddr;
            AddrReg gammaAddr;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr);
            for (uint16_t i = 0; i < loopCount; ++i) {
                xAddr = CreateAddrReg<T>(i, VL_FP32);
                gammaAddr = CreateAddrReg<G>(i, VL_FP32);
                pregLoop = UpdateMask<float>(maskCount);
                if constexpr (IsSameType<T, float>::value) {
                    LoadAlign<float, LoadDist::DIST_NORM>(xReg, xPtr, xAddr);
                } else {
                    RegTensor<T> xIn;
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(xIn, xPtr, xAddr);
                    Cast<float, T, castTraitB162B32>(xReg, xIn, pregLoop);
                }
                if constexpr (IsSameType<G, float>::value) {
                    LoadAlign<float, LoadDist::DIST_NORM>(gammaReg, gammaPtr, gammaAddr);
                } else {
                    RegTensor<G> gammaIn;
                    LoadAlign<G, LoadDist::DIST_UNPACK_B16>(gammaIn, gammaPtr, gammaAddr);
                    Cast<float, G, castTraitB162B32>(gammaReg, gammaIn, pregLoop);
                }
                Mul(yReg, xReg, rstdReg, pregLoop);
                Mul(yReg, yReg, gammaReg, pregLoop);
                if constexpr (IsSameType<T, float>::value) {
                    StoreAlign<float, StoreDist::DIST_NORM>(yPtr, yReg, xAddr, pregLoop);
                } else {
                    RegTensor<T> yOut;
                    Cast<T, float, castTraitB322B16>(yOut, yReg, pregLoop);
                    StoreAlign<T, StoreDist::DIST_PACK_B32>(yPtr, yOut, xAddr, pregLoop);
                }
            }
        }
    }

    __aicore__ inline void CopyOutY(uint64_t row, uint64_t tileOffset, uint32_t curTileLen)
    {
        // 仅搬出当前 tile 的真实长度。
        LocalTensor<T> yLocal = yQueue.DeQue<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(curTileLen * sizeof(T)), 0, 0, 0};
        uint64_t gmOffset = row * numCol + curRStart + tileOffset;
        DataCopyPad(yGm[gmOffset], yLocal, params);
        yQueue.FreeTensor(yLocal);
    }

private:
    TPipe* pPipe;
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> gammaQueue;
    TQue<QuePosition::VECOUT, 1> yQueue;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> workBuf;
    TQue<QuePosition::VECOUT, 1> partQueue;
    TQue<QuePosition::VECIN, 1> combQueue;
    TBuf<TPosition::VECCALC> cacheBuf;
    TQue<QuePosition::VECOUT, 1> rstdQueue;

    GlobalTensor<T> xGm;
    GlobalTensor<G> gammaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> wsGm;

    uint64_t numRow;
    uint64_t numCol;
    uint64_t rBlockFactor;
    uint64_t rBlockNum;
    uint64_t tailR;
    uint64_t ubFactor;
    uint64_t numRowAlign;
    float epsilon;
    float avgFactor;

    uint64_t rBlockIdx;
    uint64_t curRStart;
    uint64_t curRLen;
};
} // namespace RmsNorm
#endif // RMS_NORM_REGBASE_SPLIT_AR_H
