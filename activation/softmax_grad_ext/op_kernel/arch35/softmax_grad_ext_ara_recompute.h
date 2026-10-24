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
 * \file softmax_grad_ext_ara_recompute.h
 * \brief
 */

#ifndef SOFTMAX_GRAD_EXT_ARA_RECOMPUTE_H
#define SOFTMAX_GRAD_EXT_ARA_RECOMPUTE_H

#include "kernel_tiling/kernel_tiling.h"

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
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

// ARA 重计算模板：R 过大单 tile 装不下时按 binAddRFactor 分段求 S、逐段重读输出
// 语义：y = (grad * x1 - x1 * sum_R(grad * x1)) * x2
// 段内折叠：每 SCRATCH_ROW_NUM(8) 行 scratch 承载段内 ROWS_PER_SCRATCH(16) 个输入行的寄存器累加，
// 再经 NlastDichotomyAdd 二分折叠为 1 行；跨段按 binAdd* 二分累加结构进 cacheBuffer。
template <typename T>
class SoftmaxGradExtARARecompute : public SoftmaxGradExtBase {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BUFFER_DEPTH = 1;
    static constexpr int32_t MC_PARTIAL_BATCH = 8; // mc 部分和分批读取个数
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
    // VF 循环归纳变量要求 uint16_t
    static constexpr uint16_t SCRATCH_ROW_NUM = static_cast<uint16_t>(CONST_EIGHT);              // 8 行折叠 scratch
    static constexpr uint16_t ROWS_PER_SCRATCH = static_cast<uint16_t>(CONST_EIGHT * CONST_TWO); // 每行承载16输入行
    // 段长折叠容量耦合约束：scratch 总承载行数须与 tiling 侧默认段长一致（人工同步易漏，编译期钉死）
    static_assert(static_cast<int64_t>(SCRATCH_ROW_NUM) * ROWS_PER_SCRATCH == 128,
                  "must equal DEFAULT_BIN_ADD_R_FACTOR(128) in softmax_grad_ext_tiling_ara_recompute_arch35.cpp");

public:
    __aicore__ inline SoftmaxGradExtARARecompute(){};

    __aicore__ inline SoftmaxGradExtARARecompute(const SoftmaxGradExtARARecomputeTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, TPipe* pipeIn)
    {
        pipe_ = pipeIn;

        gradGm_.SetGlobalBuffer((__gm__ T*)grad);
        x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm_.SetGlobalBuffer((__gm__ T*)x2);
        yGm_.SetGlobalBuffer((__gm__ T*)y);

        // LoadAlign 整寄存器读恒读满一个向量宽度（fp32 64 float=256B；fp16/bf16 经 DIST_UNPACK_B16
        // 按偏移步进 64 元素推断为 64 half=128B，按 256B 兜底覆盖最大读宽）：行宽不足向量宽度时
        // 非 flat 路径末行读取会越过段 buffer 末尾（数值被 mask 丢弃但属越界），统一补一个向量
        // 寄存器宽度的 guard
        constexpr int64_t VEC_REG_BYTES = 256;
        int64_t segShapeLen = tilingData_->tileA0Len * tilingData_->binAddRFactor +
                              VEC_REG_BYTES / static_cast<int64_t>(sizeof(T));
        pipe_->InitBuffer(gradQueue_, BUFFER_NUM, segShapeLen * sizeof(T));
        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, segShapeLen * sizeof(T));
        pipe_->InitBuffer(x2Queue_, BUFFER_NUM, segShapeLen * sizeof(T));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, segShapeLen * sizeof(T));

        // xSumBuf/reduceSumTempBuffer 的 LoadAlign 与 NlastDichotomyAdd 同样按 64 float 整读，
        // MulFoldToScratch 中 g*tileA0Len+k*VL_FP32 偏移的读可越过 max(tileA0Len, VL_FP32)，
        // 各追加 VL_FP32 个 float 作越界 guard
        int64_t xSumLen = tilingData_->tileA0Len > VL_FP32 ? tilingData_->tileA0Len : VL_FP32;
        xSumLen += VL_FP32;
        pipe_->InitBuffer(xSumBuf_, xSumLen * sizeof(float));
        pipe_->InitBuffer(cacheBuffer_, tilingData_->binAddCacheBufferCount * tilingData_->tileA0Len * sizeof(float));
        pipe_->InitBuffer(tempBuffer_, tilingData_->tileA0Len * sizeof(float));
        int64_t reduceSumLen = CONST_EIGHT * tilingData_->tileA0Len;
        reduceSumLen = reduceSumLen < static_cast<int64_t>(VL_FP32) ? static_cast<int64_t>(VL_FP32) : reduceSumLen;
        reduceSumLen += VL_FP32;
        pipe_->InitBuffer(reduceSumTempBuffer_, reduceSumLen * sizeof(float));
        if (tilingData_->mcEnabled == 1) {
            // 部分和 overlay 于输出张量头部（规避 TTK 下 workspace 仅 32B 占位的问题）
            partialGm_.SetGlobalBuffer((__gm__ float*)y);
            pipe_->InitBuffer(partialOutQueue_, BUFFER_NUM, VL_FP32 * sizeof(float));
            pipe_->InitBuffer(partialInQueue_, BUFFER_NUM, MC_PARTIAL_BATCH * VL_FP32 * sizeof(float));
            // xSumBuf 按 VL_FP32 全 64 清零：a0<64 时 P1 仅累加 [0,a0)，脏 lane 保持 0 才不致写入 overlay
            LocalTensor<float> xSumTensor = xSumBuf_.Get<float>();
            Duplicate(xSumTensor, static_cast<float>(0.0), VL_FP32);
        }

        tempTensor_ = tempBuffer_.Get<float>();
        cacheTensor_ = cacheBuffer_.Get<float>();
        reduceSumTempTensor_ = reduceSumTempBuffer_.Get<float>();
    }

    __aicore__ inline void Process()
    {
        if (tilingData_->mcEnabled == 1) {
            ProcessMulticore();
            return;
        }

        int64_t blockIdx = GetBlockIdx();
        int64_t used = tilingData_->usedCoreNums;
        if (blockIdx >= used) {
            return;
        }
        // 余数分配：前 rem 核各多分 1 个 tile，保证 totalTiles 全部分配（零同步）
        int64_t base = tilingData_->totalTiles / used;
        int64_t rem = tilingData_->totalTiles % used;
        int64_t beginIdx = blockIdx * base + (blockIdx < rem ? blockIdx : rem);
        int64_t endIdx = beginIdx + base + (blockIdx < rem ? 1 : 0);
        endIdx = endIdx > tilingData_->totalTiles ? tilingData_->totalTiles : endIdx;

        // pattern is [A1, R, A0]
        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curA0Idx = curIdx % tilingData_->tileA0Outer;
            int64_t curA1Idx = curIdx / tilingData_->tileA0Outer;

            uint32_t curTileA0Len = curA0Idx == (tilingData_->tileA0Outer - 1) ? tilingData_->tileA0Tail :
                                                                                 tilingData_->tileA0Len;

            int64_t xOffset =
                // a1 offset
                curA1Idx * tilingData_->totalRLen * tilingData_->totalA0Len +
                // a0 offset
                curA0Idx * tilingData_->tileA0Len;

            uint16_t loopA0Num = ops::CeilDiv(curTileA0Len, VL_FP32);

            // Pass1: 逐段求 S
            CalcReduceSum(curTileA0Len, xOffset, loopA0Num);

            // Pass2: 逐段重读 grad/x1/x2 输出（尾段非整 quad，走逐行路径）
            bool flatVec = IsFlatVecEnabled(curTileA0Len, tilingData_->binAddRFactor);
            if (flatVec) {
                BuildTiledSum();
            }
            for (int64_t idx = 0; idx < tilingData_->binAddRTotalLoop; idx++) {
                int64_t curTileRLen = idx == (tilingData_->binAddRTotalLoop - 1) ? tilingData_->binAddRTail :
                                                                                   tilingData_->binAddRFactor;

                int64_t offset = xOffset + idx * tilingData_->binAddRFactor * tilingData_->totalA0Len;
                CopyInOutput(offset, curTileRLen, curTileA0Len);
                if (flatVec && curTileRLen == tilingData_->binAddRFactor) {
                    CalcOutputFlat(curTileRLen, curTileA0Len);
                } else {
                    CalcOutput(curTileRLen, curTileA0Len, loopA0Num);
                }
                CopyOutY(offset, curTileRLen, curTileA0Len);
            }
        }
    }

private:
    // mc（R 切分多核）路径：P1 各核按 (a1, R 分块) 求部分和写入输出张量头部 overlay，
    // SyncAll 后各核汇总 S，再次 SyncAll 后 P2 全宽连续读写；a0 整除 64 lane（16/32/64）时启用，
    // a0<64 的段累加走 AccumulateSegmentFlat（tiled lane j=列 j%a0），
    // 部分和汇总由 LoadFinalSumMc 标量折叠（S[c] = Σ_g xSum[c + g*a0]）完成
    __aicore__ inline void ProcessMulticore()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t chunks = tilingData_->mcChunksPerA1;
        int64_t a1Idx = blockIdx / chunks;
        int64_t cIdx = blockIdx - a1Idx * chunks;
        int64_t totalR = tilingData_->totalRLen;
        int64_t a0 = tilingData_->totalA0Len;

        int64_t rBegin = cIdx * totalR / chunks;
        int64_t rEnd = (cIdx + 1) * totalR / chunks;

        CalcPartialSumMc(a1Idx, cIdx, rBegin, rEnd - rBegin);
        AscendC::SyncAll();
        LoadFinalSumMc(a1Idx);
        AscendC::SyncAll();

        // P2：全宽连续拷贝 + flat 输出（尾段 <64 lane 残差在 CalcOutputFlat 内掩码处理）
        int64_t xOffset = a1Idx * totalR * a0;
        int64_t seg = rBegin;
        while (seg < rEnd) {
            int64_t rows = rEnd - seg;
            if (rows > tilingData_->binAddRFactor) {
                rows = tilingData_->binAddRFactor;
            }
            int64_t offset = xOffset + seg * a0;
            CopyInOutput(offset, rows, a0);
            CalcOutputFlat(rows, a0);
            CopyOutY(offset, rows, a0);
            seg += rows;
        }
    }

    // mc P1：当前核 R 分块内按段累加 grad*x1，部分和 [64] 写入 overlay（a0<64 时脏 lane 为 0）
    __aicore__ inline void CalcPartialSumMc(int64_t a1Idx, int64_t cIdx, int64_t rBegin, int64_t rowCount)
    {
        int64_t a0 = tilingData_->totalA0Len;
        int64_t xOffset = a1Idx * tilingData_->totalRLen * a0;
        LocalTensor<float> xSumTensor = xSumBuf_.Get<float>();
        __ubuf__ float* xSumLocal = (__ubuf__ float*)xSumTensor.GetPhyAddr();
        Duplicate(xSumTensor, static_cast<float>(0.0), VL_FP32);

        int64_t segEnd = rBegin + rowCount;
        int64_t seg = rBegin;
        while (seg < segEnd) {
            int64_t rows = segEnd - seg;
            if (rows > tilingData_->binAddRFactor) {
                rows = tilingData_->binAddRFactor;
            }
            CopyInSum(xOffset + seg * a0, rows, a0);
            LocalTensor<T> grad = gradQueue_.DeQue<T>();
            LocalTensor<T> x1 = x1Queue_.DeQue<T>();
            // a0|64 时按 64 lane 整 quad 累加进 tiled 部分和（lane j = 列 j%a0），尾部残差掩码处理
            AccumulateSegmentFlat((__ubuf__ T*)grad.GetPhyAddr(), (__ubuf__ T*)x1.GetPhyAddr(), rows, a0, xSumLocal);
            gradQueue_.FreeTensor(grad);
            x1Queue_.FreeTensor(x1);
            seg += rows;
        }

        LocalTensor<float> partial = partialOutQueue_.AllocTensor<float>();
        __ubuf__ float* partialLocal = (__ubuf__ float*)partial.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> accReg;
            MaskReg pFull;
            pFull = CreateMask<float, MaskPattern::ALL>();
            LoadAlign<float, LoadDist::DIST_NORM>(accReg, xSumLocal);
            StoreAlign(partialLocal, accReg, pFull);
        }
        partialOutQueue_.EnQue(partial);
        LocalTensor<float> partialOut = partialOutQueue_.DeQue<float>();
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(VL_FP32 * sizeof(float)), 0, 0, 0};
        DataCopyPad<float, PaddingMode::Normal>(partialGm_[(a1Idx * tilingData_->mcChunksPerA1 + cIdx) * VL_FP32],
                                                partialOut, copyParams);
        partialOutQueue_.FreeTensor(partialOut);
    }

    // mc P1 段内 flat 累加：quads 全 64 lane 累入寄存器后并入 xSum[64]
    __aicore__ inline void AccumulateSegmentFlat(__ubuf__ T* gradLocal, __ubuf__ T* x1Local, int64_t curTileRLen,
                                                 uint32_t curTileA0Len, __ubuf__ float* xSumLocal)
    {
        uint32_t vlLen = static_cast<uint32_t>(VL_FP32);
        uint32_t total = static_cast<uint32_t>(curTileRLen) * curTileA0Len;
        uint16_t fullQuads = static_cast<uint16_t>(total / vlLen);
        __VEC_SCOPE__
        {
            RegTensor<float> accReg;
            RegTensor<float> sumReg;
            RegTensor<float> gradReg;
            RegTensor<float> x1Reg;
            RegTensor<float> mulReg;
            MaskReg pFull;
            pFull = CreateMask<float, MaskPattern::ALL>();

            Duplicate(accReg, 0.0, pFull);
            for (uint16_t q = 0; q < fullQuads; q++) {
                uint32_t offset = static_cast<uint32_t>(q) * vlLen;
                LoadTensorForDtypeTIn(gradLocal, gradReg, pFull, offset);
                LoadTensorForDtypeTIn(x1Local, x1Reg, pFull, offset);
                Mul(mulReg, gradReg, x1Reg, pFull);
                Add(accReg, accReg, mulReg, pFull);
            }
            LoadAlign<float, LoadDist::DIST_NORM>(sumReg, xSumLocal);
            Add(sumReg, sumReg, accReg, pFull);
            StoreAlign(xSumLocal, sumReg, pFull);
        }
        // 段尾 <64 lane 残差：偏移 64 对齐的掩码累加（独立函数，避免 VF 循环后 UpdateMask）
        uint32_t tailElems = total % vlLen;
        if (tailElems != 0U) {
            AccumulateSegmentFlatTail(gradLocal, x1Local, xSumLocal, static_cast<uint32_t>(fullQuads) * vlLen,
                                      tailElems);
        }
    }

    // mc P1 尾段（<64 lane 残差）：掩码累加进 tiled 部分和（tailOffset 恒为 64 lane 对齐）
    __aicore__ inline void AccumulateSegmentFlatTail(__ubuf__ T* gradLocal, __ubuf__ T* x1Local,
                                                     __ubuf__ float* xSumLocal, uint32_t tailOffset, uint32_t tailElems)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> sumReg;
            RegTensor<float> gradReg;
            RegTensor<float> x1Reg;
            RegTensor<float> mulReg;
            MaskReg pTail;
            pTail = UpdateMask<float>(tailElems);
            LoadAlign<float, LoadDist::DIST_NORM>(sumReg, xSumLocal);
            LoadTensorForDtypeTIn(gradLocal, gradReg, pTail, tailOffset);
            LoadTensorForDtypeTIn(x1Local, x1Reg, pTail, tailOffset);
            Mul(mulReg, gradReg, x1Reg, pTail);
            Add(sumReg, sumReg, mulReg, pTail);
            StoreAlign(xSumLocal, sumReg, pTail);
        }
    }

    // mc P2 前置：读取本 a1 全部部分和并汇总为 S（分批 [64] 向量），随后平铺供 flat 输出使用
    __aicore__ inline void LoadFinalSumMc(int64_t a1Idx)
    {
        int64_t chunks = tilingData_->mcChunksPerA1;
        int64_t a0 = tilingData_->totalA0Len;
        uint32_t a0Len = static_cast<uint32_t>(a0);
        uint32_t vlLen = static_cast<uint32_t>(VL_FP32);
        LocalTensor<float> xSumTensor = xSumBuf_.Get<float>();
        __ubuf__ float* xSumLocal = (__ubuf__ float*)xSumTensor.GetPhyAddr();
        Duplicate(xSumTensor, static_cast<float>(0.0), VL_FP32);

        int64_t base = a1Idx * chunks * VL_FP32;
        int64_t done = 0;
        while (done < chunks) {
            int64_t batch = chunks - done;
            if (batch > MC_PARTIAL_BATCH) {
                batch = MC_PARTIAL_BATCH;
            }
            uint16_t batchCnt = static_cast<uint16_t>(batch);
            LocalTensor<float> in = partialInQueue_.AllocTensor<float>();
            DataCopyPadExtParams<float> padParams = {false, 0, 0, 0.0};
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(batch * VL_FP32 * sizeof(float)), 0, 0, 0};
            DataCopyPad<float, PaddingMode::Normal>(in, partialGm_[base + done * VL_FP32], copyParams, padParams);
            partialInQueue_.EnQue(in);
            in = partialInQueue_.DeQue<float>();
            __VEC_SCOPE__
            {
                RegTensor<float> accReg;
                RegTensor<float> inReg;
                MaskReg pFull;
                // 部分和槽位固定 64 float：a0<64 时为 tiled 部分和（lane j = 列 j%a0），整向量累加
                pFull = CreateMask<float, MaskPattern::ALL>();
                LoadAlign<float, LoadDist::DIST_NORM>(accReg, xSumLocal);
                for (uint16_t i = 0; i < batchCnt; i++) {
                    LoadAlign<float, LoadDist::DIST_NORM>(
                        inReg, (__ubuf__ float*)in.GetPhyAddr() + static_cast<uint32_t>(i) * vlLen);
                    Add(accReg, accReg, inReg, pFull);
                }
                StoreAlign(xSumLocal, accReg, pFull);
            }
            partialInQueue_.FreeTensor(in);
            done += batch;
        }
        // a0<64：折叠 tiled 64 lane 部分和为 S[0..a0)（S[c] = Σ_g xSum[c + g*a0]；每 chunk 一次，标量开销可忽略）
        if (a0Len < VL_FP32) {
            AscendC::PipeBarrier<PIPE_ALL>();
            uint32_t groups = VL_FP32 / a0Len;
            for (uint32_t c = 0; c < a0Len; c++) {
                float s = 0.0f;
                for (uint32_t g = 0; g < groups; g++) {
                    s += xSumTensor.GetValue(c + g * a0Len);
                }
                xSumTensor.SetValue(c, s);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        if (a0Len >= static_cast<uint32_t>(CONST_EIGHT)) {
            BuildTiledSum();
        } else {
            // a0<8：DataCopy 的 count/地址须 8 float 对齐，改用标量平铺 tiled[j] = S[j%a0]
            LocalTensor<float> tiledTensor = reduceSumTempBuffer_.Get<float>();
            uint32_t groups = VL_FP32 / a0Len;
            for (uint32_t g = 0; g < groups; g++) {
                for (uint32_t c = 0; c < a0Len; c++) {
                    tiledTensor.SetValue(g * a0Len + c, xSumTensor.GetValue(c));
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void CalcReduceSum(uint32_t curTileA0Len, int64_t xOffsetStart, uint16_t loopA0Num)
    {
        int64_t xSrcStride = tilingData_->totalA0Len;

        for (int64_t basicBlockIdx = 0; basicBlockIdx < tilingData_->binAddBasicBlockLoop; basicBlockIdx++) {
            ProcessMainBlock(tilingData_->binAddRFactor, curTileA0Len,
                             xOffsetStart + basicBlockIdx * xSrcStride * tilingData_->binAddRFactor, loopA0Num);
            if (basicBlockIdx < tilingData_->binAddMainFoldCount) {
                ProcessFoldBlock(tilingData_->binAddRFactor, curTileA0Len,
                                 xOffsetStart + (basicBlockIdx + tilingData_->binAddBasicBlockLoop) * xSrcStride *
                                                    tilingData_->binAddRFactor,
                                 loopA0Num);
            } else if (basicBlockIdx == tilingData_->binAddMainFoldCount && tilingData_->binAddRTail > 0 &&
                       tilingData_->binAddRTail != tilingData_->binAddRFactor) {
                ProcessFoldBlock(tilingData_->binAddRTail, curTileA0Len,
                                 xOffsetStart + (basicBlockIdx + tilingData_->binAddBasicBlockLoop) * xSrcStride *
                                                    tilingData_->binAddRFactor,
                                 loopA0Num);
            }
            ProcessSummation(basicBlockIdx, curTileA0Len);
        }

        // R 很小（单段以内），不需要做跨段二分累加
        if (tilingData_->binAddBasicBlockLoop == 0) {
            ProcessMainBlock(tilingData_->binAddRTail, curTileA0Len, xOffsetStart, loopA0Num);
            ProcessSummation(0, curTileA0Len);
        }
        LocalTensor<float> xSumTensor = xSumBuf_.Get<float>();
        DataCopy(xSumTensor, cacheTensor_[tilingData_->binAddResultCacheID * tilingData_->tileA0Len],
                 tilingData_->tileA0Len);
    }

    // 段内逐行 mul 折叠进 8 行 scratch（main 段覆盖写，fold 段累加）
    __aicore__ inline void ProcessMainBlock(const int64_t curTileRLen, const uint32_t curTileA0Len,
                                            const int64_t gmOffset, uint16_t loopA0Num)
    {
        CopyInSum(gmOffset, curTileRLen, curTileA0Len);
        LocalTensor<T> gradMain = gradQueue_.DeQue<T>();
        LocalTensor<T> x1Main = x1Queue_.DeQue<T>();
        __ubuf__ T* gradLocal = (__ubuf__ T*)gradMain.GetPhyAddr();
        __ubuf__ T* x1Local = (__ubuf__ T*)x1Main.GetPhyAddr();

        MulFoldToScratch(gradLocal, x1Local, curTileRLen, curTileA0Len, loopA0Num, false);

        gradQueue_.FreeTensor(gradMain);
        x1Queue_.FreeTensor(x1Main);
    }

    // fold 段逐行 mul 累加进 scratch 已有值
    __aicore__ inline void ProcessFoldBlock(const int64_t curTileRLen, const uint32_t curTileA0Len,
                                            const int64_t gmOffset, uint16_t loopA0Num)
    {
        CopyInSum(gmOffset, curTileRLen, curTileA0Len);
        LocalTensor<T> gradFold = gradQueue_.DeQue<T>();
        LocalTensor<T> x1Fold = x1Queue_.DeQue<T>();
        __ubuf__ T* gradLocal = (__ubuf__ T*)gradFold.GetPhyAddr();
        __ubuf__ T* x1Local = (__ubuf__ T*)x1Fold.GetPhyAddr();

        MulFoldToScratch(gradLocal, x1Local, curTileRLen, curTileA0Len, loopA0Num, true);

        gradQueue_.FreeTensor(gradFold);
        x1Queue_.FreeTensor(x1Fold);
    }

    __aicore__ inline void ProcessSummation(const int64_t basicBlockIdx, uint32_t curTileA0Len)
    {
        int64_t cacheID = GetCacheID(basicBlockIdx);
        FoldScratchToTemp(tempTensor_, curTileA0Len);
        UpdateCache(cacheTensor_, tempTensor_, cacheID, tilingData_->tileA0Len, tilingData_->tileA0Len);
    }

    // 将 [curTileRLen, tileA0Len] 的 grad*x1 折叠进 8 行 scratch：
    // scratch 行 g 承载输入行 [g*16, g*16+16)，越界组写 0（覆盖尾段 stale 行）；
    // accumulate=true 时在 scratch 已有值上累加（fold 段）
    __aicore__ inline void MulFoldToScratch(__ubuf__ T* gradLocal, __ubuf__ T* x1Local, int64_t curTileRLen,
                                            uint32_t curTileA0Len, uint16_t loopA0Num, bool accumulate)
    {
        if (IsFlatVecEnabled(curTileA0Len, curTileRLen)) {
            MulFoldToScratchFlat(gradLocal, x1Local, curTileRLen, curTileA0Len, accumulate);
            return;
        }

        uint32_t tileA0Len = tilingData_->tileA0Len;
        __ubuf__ float* scratchLocal = (__ubuf__ float*)reduceSumTempTensor_.GetPhyAddr();

        __VEC_SCOPE__
        {
            RegTensor<float> gradReg;
            RegTensor<float> x1Reg;
            RegTensor<float> mulReg;
            RegTensor<float> accReg;

            MaskReg pregMask;

            for (uint16_t g = 0; g < SCRATCH_ROW_NUM; g++) {
                // 每行 scratch 承载 ROWS_PER_SCRATCH 个输入行；越界组行数为 0（写 0 覆盖尾段 stale 行）
                uint16_t rowStart = static_cast<uint16_t>(g * ROWS_PER_SCRATCH);
                uint16_t rowCnt = static_cast<uint16_t>(curTileRLen);
                rowCnt = static_cast<uint16_t>(rowCnt > rowStart ? rowCnt - rowStart : 0);
                rowCnt = rowCnt > ROWS_PER_SCRATCH ? ROWS_PER_SCRATCH : rowCnt;
                for (uint16_t k = 0; k < loopA0Num; k++) {
                    uint32_t sreg = curTileA0Len;
                    pregMask = UpdateMask<float>(sreg);
                    uint32_t scratchOffset = g * tileA0Len + k * VL_FP32;
                    if (accumulate) {
                        LoadAlign<float, LoadDist::DIST_NORM>(accReg, scratchLocal + scratchOffset);
                    } else {
                        Duplicate(accReg, 0.0, pregMask);
                    }
                    for (uint16_t i = 0; i < rowCnt; i++) {
                        uint32_t offset = (rowStart + i) * tileA0Len + k * VL_FP32;
                        LoadTensorForDtypeTIn(gradLocal, gradReg, pregMask, offset);
                        LoadTensorForDtypeTIn(x1Local, x1Reg, pregMask, offset);
                        Mul(mulReg, gradReg, x1Reg, pregMask);
                        Add(accReg, accReg, mulReg, pregMask);
                    }
                    StoreAlign(scratchLocal + scratchOffset, accReg, pregMask);
                }
            }
        }
    }

    // flat 满车道路径条件：行宽整除 64 lane 向量寄存器、段内行紧凑、且段元素数对齐向量宽度；
    // 尾块行宽不等或段非整 quad（如 R 尾段）时回退逐行路径。
    // 注意：VF 循环所在函数内若存在循环后的 UpdateMask，编译器会将错误掩码关联进硬件循环，
    // 因此 flat 路径只处理整 quad 段，函数体内不出现任何 UpdateMask
    __aicore__ inline bool IsFlatVecEnabled(uint32_t curTileA0Len, int64_t curTileRLen)
    {
        uint32_t tileA0Len = static_cast<uint32_t>(tilingData_->tileA0Len);
        return tileA0Len == curTileA0Len && tileA0Len >= static_cast<uint32_t>(CONST_EIGHT) &&
               tileA0Len <= static_cast<uint32_t>(VL_FP32) && (VL_FP32 % static_cast<int64_t>(tileA0Len)) == 0 &&
               (curTileRLen * static_cast<int64_t>(tileA0Len)) % static_cast<int64_t>(VL_FP32) == 0;
    }

    // flat 满车道折叠：一个 64 lane 向量覆盖 VL_FP32/tileA0Len 个整行，寄存器累加求乘积和后整向量落
    // scratch——组 g 恰为 scratch 行 g，组间折叠复用下游 FoldScratchToTemp(8->1)；行 groups~7 清零。
    // 注：VF 硬件循环地址只支持向量宽度步长，故清零按连续区间向量宽度步进，禁用行步长循环
    __aicore__ inline void MulFoldToScratchFlat(__ubuf__ T* gradLocal, __ubuf__ T* x1Local, int64_t curTileRLen,
                                                uint32_t curTileA0Len, bool accumulate)
    {
        uint32_t tileA0Len = static_cast<uint32_t>(tilingData_->tileA0Len);
        uint32_t vlLen = static_cast<uint32_t>(VL_FP32);
        uint32_t total = static_cast<uint32_t>(curTileRLen) * tileA0Len;
        uint16_t fullQuads = static_cast<uint16_t>(total / vlLen);
        uint16_t groups = static_cast<uint16_t>(vlLen / tileA0Len);
        __ubuf__ float* scratchLocal = (__ubuf__ float*)reduceSumTempTensor_.GetPhyAddr();

        __VEC_SCOPE__
        {
            RegTensor<float> accReg;
            RegTensor<float> gradReg;
            RegTensor<float> x1Reg;
            RegTensor<float> mulReg;
            RegTensor<float> oldRowReg;
            RegTensor<float> zeroReg;
            MaskReg pFull;
            pFull = CreateMask<float, MaskPattern::ALL>();

            if (accumulate) {
                LoadAlign<float, LoadDist::DIST_NORM>(oldRowReg, scratchLocal);
            }
            Duplicate(accReg, 0.0, pFull);
            for (uint16_t q = 0; q < fullQuads; q++) {
                uint32_t offset = static_cast<uint32_t>(q) * vlLen;
                LoadTensorForDtypeTIn(gradLocal, gradReg, pFull, offset);
                LoadTensorForDtypeTIn(x1Local, x1Reg, pFull, offset);
                Mul(mulReg, gradReg, x1Reg, pFull);
                Add(accReg, accReg, mulReg, pFull);
            }
            // fold 段旧值按车道位置并入（行 g = 旧组 g + 新组 g，8->1 折叠求和后即旧总和+新总和）
            if (accumulate) {
                Add(accReg, accReg, oldRowReg, pFull);
            }
            StoreAlign(scratchLocal, accReg, pFull);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            // 行 groups~7 清零：连续区间 [vlLen, SCRATCH_ROW_NUM*tileA0Len)，按向量宽度步进
            uint32_t zeroCnt = (static_cast<uint32_t>(SCRATCH_ROW_NUM) - static_cast<uint32_t>(groups)) * tileA0Len;
            if (zeroCnt != 0U) {
                uint16_t zeroFull = static_cast<uint16_t>(zeroCnt / vlLen);
                Duplicate(zeroReg, 0.0, pFull);
                for (uint16_t z = 0; z < zeroFull; z++) {
                    StoreAlign(scratchLocal + vlLen + static_cast<uint32_t>(z) * vlLen, zeroReg, pFull);
                }
            }
        }
    }

    // P2 前将 xSum 平铺为 [tileA0Len] * groups 的 64 lane 模式（复用 P1 scratch 缓冲，两阶段分时互斥）
    __aicore__ inline void BuildTiledSum()
    {
        uint32_t tileA0Len = static_cast<uint32_t>(tilingData_->tileA0Len);
        uint32_t groups = static_cast<uint32_t>(static_cast<uint32_t>(VL_FP32) / tileA0Len);
        LocalTensor<float> xSumTensor = xSumBuf_.Get<float>();
        LocalTensor<float> tiledTensor = reduceSumTempBuffer_.Get<float>();
        for (uint32_t g = 0; g < groups; g++) {
            DataCopy(tiledTensor[g * tileA0Len], xSumTensor, tileA0Len);
        }
    }

    // 8 行 scratch 二分折叠为 1 行 temp（复用 base.h NlastDichotomyAdd）
    __aicore__ inline void FoldScratchToTemp(const LocalTensor<float>& dstTensor, uint32_t curTileA0Len)
    {
        uint16_t loopTimes = ops::CeilDiv(curTileA0Len, VL_FP32);
        uint32_t stride = tilingData_->tileA0Len;
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* srcA = (__ubuf__ float*)reduceSumTempTensor_.GetPhyAddr();
            __ubuf__ float* srcB = srcA + stride;
            uint32_t count = curTileA0Len;
            RegTensor<float> accReg;
            MaskReg pMask;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                __ubuf__ float* curSrcA = srcA + i * VL_FP32;
                __ubuf__ float* curSrcB = srcB + i * VL_FP32;
                pMask = UpdateMask<float>(count);
                NlastDichotomyAdd<CONST_EIGHT>::LoadAndAccumulate(accReg, curSrcA, curSrcB, pMask, stride);
                StoreAlign((__ubuf__ float*)dst + i * VL_FP32, accReg, pMask);
            }
        }
    }

    // Pass1 段搬入：仅 grad / x1
    __aicore__ inline void CopyInSum(int64_t xGmOffset, int64_t curTileRLen, uint32_t curTileA0Len)
    {
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        if (tilingData_->tileA0Len == tilingData_->totalA0Len) {
            // 不切 A0：GM 行连续，整段单 block 搬运（兼容 a0*sizeof(T)<32B 的逐行小块）；
            // blockLen 单位 Byte，超 32B 由 MTE 内部拆分
            copyInParams.blockCount = 1;
            copyInParams.blockLen = static_cast<uint32_t>(curTileRLen * curTileA0Len * sizeof(T));
        } else {
            copyInParams.blockCount = curTileRLen;
            copyInParams.blockLen = curTileA0Len * sizeof(T);
        }
        copyInParams.srcStride = (tilingData_->totalA0Len - curTileA0Len) * sizeof(T);
        copyInParams.dstStride = (tilingData_->tileA0Len - curTileA0Len) * sizeof(T) / BLOCK_SIZE;

        LocalTensor<T> gradLocal = gradQueue_.AllocTensor<T>();
        DataCopyPad<T, PaddingMode::Normal>(gradLocal, gradGm_[xGmOffset], copyInParams, dataCopyPadExtParams);
        gradQueue_.EnQue(gradLocal);

        LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        DataCopyPad<T, PaddingMode::Normal>(x1Local, x1Gm_[xGmOffset], copyInParams, dataCopyPadExtParams);
        x1Queue_.EnQue(x1Local);
    }

    // Pass2 段搬入：grad / x1 / x2（x2 scalar 时 1 点 Duplicate 到段 buffer）
    __aicore__ inline void CopyInOutput(int64_t xGmOffset, int64_t curTileRLen, uint32_t curTileA0Len)
    {
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        if (tilingData_->tileA0Len == tilingData_->totalA0Len) {
            copyInParams.blockCount = 1;
            copyInParams.blockLen = static_cast<uint32_t>(curTileRLen * curTileA0Len * sizeof(T));
        } else {
            copyInParams.blockCount = curTileRLen;
            copyInParams.blockLen = curTileA0Len * sizeof(T);
        }
        copyInParams.srcStride = (tilingData_->totalA0Len - curTileA0Len) * sizeof(T);
        copyInParams.dstStride = (tilingData_->tileA0Len - curTileA0Len) * sizeof(T) / BLOCK_SIZE;

        LocalTensor<T> gradLocal = gradQueue_.AllocTensor<T>();
        DataCopyPad<T, PaddingMode::Normal>(gradLocal, gradGm_[xGmOffset], copyInParams, dataCopyPadExtParams);
        gradQueue_.EnQue(gradLocal);

        LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        DataCopyPad<T, PaddingMode::Normal>(x1Local, x1Gm_[xGmOffset], copyInParams, dataCopyPadExtParams);
        x1Queue_.EnQue(x1Local);

        if (tilingData_->x2IsScalar == 1) {
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

    __aicore__ inline void CalcOutput(int64_t curTileRLen, uint32_t curTileA0Len, uint16_t loopA0Num)
    {
        if (IsFlatVecEnabled(curTileA0Len, curTileRLen)) {
            CalcOutputFlat(curTileRLen, curTileA0Len);
            return;
        }

        LocalTensor<T> grad = gradQueue_.DeQue<T>();
        LocalTensor<T> x1 = x1Queue_.DeQue<T>();
        LocalTensor<T> x2 = x2Queue_.DeQue<T>();
        LocalTensor<T> y = yQueue_.AllocTensor<T>();

        LocalTensor<float> xSumTensor = xSumBuf_.Get<float>();
        __ubuf__ float* xSumLocal = (__ubuf__ float*)xSumTensor.GetPhyAddr();
        __ubuf__ T* gradLocal = (__ubuf__ T*)grad.GetPhyAddr();
        __ubuf__ T* x1Local = (__ubuf__ T*)x1.GetPhyAddr();
        __ubuf__ T* x2Local = (__ubuf__ T*)x2.GetPhyAddr();
        __ubuf__ T* yLocal = (__ubuf__ T*)y.GetPhyAddr();

        uint32_t tileA0Len = tilingData_->tileA0Len;
        uint16_t curTileRLenVl = static_cast<uint16_t>(curTileRLen);
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
                LoadAlign<float, LoadDist::DIST_NORM>(sumReg, (__ubuf__ float*)xSumLocal + k * VL_FP32);
                for (uint16_t i = 0; i < curTileRLenVl; i++) {
                    uint32_t xOffset = i * tileA0Len + k * VL_FP32;

                    LoadTensorForDtypeTIn(gradLocal, gradReg, pregMask, xOffset);
                    LoadTensorForDtypeTIn(x1Local, x1Reg, pregMask, xOffset);
                    LoadTensorForDtypeTIn(x2Local, x2Reg, pregMask, xOffset);

                    // y = (grad * x1 - x1 * S) * x2
                    Mul(yReg, gradReg, x1Reg, pregMask);
                    Mul(x1Reg, x1Reg, sumReg, pregMask);
                    Sub(yReg, yReg, x1Reg, pregMask);
                    Mul(yReg, yReg, x2Reg, pregMask);

                    StoreTensorForDtypeTOut(yLocal, yReg, pregMask, xOffset);
                }
            }
        }

        yQueue_.EnQue(y);
        gradQueue_.FreeTensor<T>(grad);
        x1Queue_.FreeTensor<T>(x1);
        x2Queue_.FreeTensor<T>(x2);
    }

    // flat 满车道输出：y = (grad * x1 - x1 * S) * x2，S 平铺模式段内常驻寄存器
    __aicore__ inline void CalcOutputFlat(int64_t curTileRLen, uint32_t curTileA0Len)
    {
        LocalTensor<T> grad = gradQueue_.DeQue<T>();
        LocalTensor<T> x1 = x1Queue_.DeQue<T>();
        LocalTensor<T> x2 = x2Queue_.DeQue<T>();
        LocalTensor<T> y = yQueue_.AllocTensor<T>();

        __ubuf__ float* tiledSumLocal = (__ubuf__ float*)reduceSumTempTensor_.GetPhyAddr();
        __ubuf__ T* gradLocal = (__ubuf__ T*)grad.GetPhyAddr();
        __ubuf__ T* x1Local = (__ubuf__ T*)x1.GetPhyAddr();
        __ubuf__ T* x2Local = (__ubuf__ T*)x2.GetPhyAddr();
        __ubuf__ T* yLocal = (__ubuf__ T*)y.GetPhyAddr();

        uint32_t tileA0Len = static_cast<uint32_t>(tilingData_->tileA0Len);
        uint32_t vlLen = static_cast<uint32_t>(VL_FP32);
        uint32_t total = static_cast<uint32_t>(curTileRLen) * tileA0Len;
        uint16_t fullQuads = static_cast<uint16_t>(total / vlLen);

        __VEC_SCOPE__
        {
            RegTensor<float> gradReg;
            RegTensor<float> x1Reg;
            RegTensor<float> x2Reg;
            RegTensor<float> yReg;
            RegTensor<float> tReg;
            RegTensor<float> tiledReg;
            MaskReg pFull;
            pFull = CreateMask<float, MaskPattern::ALL>();

            LoadAlign<float, LoadDist::DIST_NORM>(tiledReg, tiledSumLocal);
            for (uint16_t q = 0; q < fullQuads; q++) {
                uint32_t offset = static_cast<uint32_t>(q) * vlLen;
                LoadTensorForDtypeTIn(gradLocal, gradReg, pFull, offset);
                LoadTensorForDtypeTIn(x1Local, x1Reg, pFull, offset);
                LoadTensorForDtypeTIn(x2Local, x2Reg, pFull, offset);
                Mul(yReg, gradReg, x1Reg, pFull);
                Mul(tReg, x1Reg, tiledReg, pFull);
                Sub(yReg, yReg, tReg, pFull);
                Mul(yReg, yReg, x2Reg, pFull);
                StoreTensorForDtypeTOut(yLocal, yReg, pFull, offset);
            }
        }
        // 段尾 <64 lane 残差：偏移 64 对齐的掩码输出（独立函数，避免 VF 循环后 UpdateMask）
        uint32_t tailElems = total % vlLen;
        if (tailElems != 0U) {
            CalcOutputFlatTail(gradLocal, x1Local, x2Local, yLocal, tiledSumLocal,
                               static_cast<uint32_t>(fullQuads) * vlLen, tailElems);
        }

        yQueue_.EnQue(y);
        gradQueue_.FreeTensor<T>(grad);
        x1Queue_.FreeTensor<T>(x1);
        x2Queue_.FreeTensor<T>(x2);
    }

    // flat 输出尾段（<64 lane 残差）：掩码计算 y（tailOffset 恒为 64 lane 对齐）
    __aicore__ inline void CalcOutputFlatTail(__ubuf__ T* gradLocal, __ubuf__ T* x1Local, __ubuf__ T* x2Local,
                                              __ubuf__ T* yLocal, __ubuf__ float* tiledSumLocal, uint32_t tailOffset,
                                              uint32_t tailElems)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> gradReg;
            RegTensor<float> x1Reg;
            RegTensor<float> x2Reg;
            RegTensor<float> yReg;
            RegTensor<float> tReg;
            RegTensor<float> tiledReg;
            MaskReg pTail;
            pTail = UpdateMask<float>(tailElems);
            LoadAlign<float, LoadDist::DIST_NORM>(tiledReg, tiledSumLocal);
            LoadTensorForDtypeTIn(gradLocal, gradReg, pTail, tailOffset);
            LoadTensorForDtypeTIn(x1Local, x1Reg, pTail, tailOffset);
            LoadTensorForDtypeTIn(x2Local, x2Reg, pTail, tailOffset);
            Mul(yReg, gradReg, x1Reg, pTail);
            Mul(tReg, x1Reg, tiledReg, pTail);
            Sub(yReg, yReg, tReg, pTail);
            Mul(yReg, yReg, x2Reg, pTail);
            StoreTensorForDtypeTOut(yLocal, yReg, pTail, tailOffset);
        }
    }

    __aicore__ inline void LoadTensorForDtypeTIn(__ubuf__ T* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T, float>::value) {
            LoadAlign<float, LoadDist::DIST_NORM>(dst, (__ubuf__ float*)src + offset);
        } else { // fp16、bf16：DIST_UNPACK_B16 完成 b16->b32 车道排布，偏移恒为 64 lane 对齐
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
        if (tilingData_->tileA0Len == tilingData_->totalA0Len) {
            copyOutParams.blockCount = 1;
            copyOutParams.blockLen = static_cast<uint32_t>(curTileRLen * curTileA0Len * sizeof(T));
        } else {
            copyOutParams.blockCount = curTileRLen;
            copyOutParams.blockLen = curTileA0Len * sizeof(T);
        }
        copyOutParams.srcStride = (tilingData_->tileA0Len - curTileA0Len) * sizeof(T) / BLOCK_SIZE;
        copyOutParams.dstStride = (tilingData_->totalA0Len - curTileA0Len) * sizeof(T);
        DataCopyPad<T, PaddingMode::Normal>(yGm_[yGmOffset], y, copyOutParams);
        yQueue_.FreeTensor(y);
    }

private:
    const SoftmaxGradExtARARecomputeTilingData* tilingData_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> gradQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> x1Queue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> x2Queue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> yQueue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> partialInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> partialOutQueue_;

    TBuf<TPosition::VECCALC> xSumBuf_;
    TBuf<TPosition::VECCALC> cacheBuffer_;
    TBuf<TPosition::VECCALC> tempBuffer_;
    TBuf<TPosition::VECCALC> reduceSumTempBuffer_;

    LocalTensor<float> tempTensor_;
    LocalTensor<float> cacheTensor_;
    LocalTensor<float> reduceSumTempTensor_;

    GlobalTensor<T> gradGm_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> partialGm_; // mc 部分和 overlay（输出张量头部）
};
} // namespace SoftmaxGradExt

#endif
