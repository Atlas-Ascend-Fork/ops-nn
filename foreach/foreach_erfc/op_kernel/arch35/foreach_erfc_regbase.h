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
 * \file foreach_erfc_regbase.h
 * \brief Flat segmented RegBase/VF kernel for foreach_erfc (arch35 / Ascend950).
 *
 * 每个核处理 tiling 给出的一个连续扁平元素区间 [tensorStart@startOffset, tensorEnd@endOffset]。
 * stage 把跨 tensor 边界的段打包进同一个 UB tile（每段 32B 上取整对齐），每个 stage 只发射
 * S 条 DataCopyPad(GM->UB)、1 次 MTE2->V 同步、1 次 Erfc、1 次 V->MTE3 同步、S 条
 * DataCopyPad(UB->GM)。段循环内没有任何 EnQue/DeQue/SetFlag/WaitFlag/PipeBarrier。
 */

#ifndef OPS_NN_FOREACH_ERFC_REGBASE_H_
#define OPS_NN_FOREACH_ERFC_REGBASE_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../../foreach_utils/arch35/foreach_flat_tiling_data.h"

namespace ForeachErfc {
using namespace AscendC;

constexpr uint32_t ERFC_UB_ALIGN_BYTES = 32;
constexpr int32_t ERFC_QUEUE_DEPTH = 2;

/*!
 * \brief 扁平段打包 kernel：一个核一个连续区间，stage 内跨 tensor 打包。
 */
template <typename T, typename Tiling>
class ForeachErfcRegbase {
public:
    __aicore__ inline ForeachErfcRegbase() = default;

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR outputs, const Tiling* tilingData, TPipe* tPipe)
    {
        blockIdx_ = GetBlockIdx();
        inDesc_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(inputs));
        outDesc_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(outputs));

        tensorDataCountList_ = const_cast<int64_t*>(tilingData->tensorDataCountList);
        tensorStart_ = tilingData->tensorStartList[blockIdx_];
        tensorEnd_ = tilingData->tensorEndList[blockIdx_];
        tensorStartOffset_ = tilingData->tensorStartOffsetList[blockIdx_];
        tensorEndOffset_ = tilingData->tensorEndOffsetList[blockIdx_];

        tileElements_ = tilingData->tileElems;
        tileBytes_ = static_cast<uint32_t>(tileElements_ * static_cast<uint32_t>(sizeof(T)));
        if (tileBytes_ < ERFC_UB_ALIGN_BYTES) {
            // tiling 至少给出一个 32B 槽；这里只做防御，不改变契约内的路径。
            tileBytes_ = ERFC_UB_ALIGN_BYTES;
            tileElements_ = ERFC_UB_ALIGN_BYTES / sizeof(T);
        }
        tPipe->InitBuffer(inQueue_, ERFC_QUEUE_DEPTH, tileBytes_);
        tPipe->InitBuffer(outQueue_, ERFC_QUEUE_DEPTH, tileBytes_);
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            // bf16 -> fp32 -> Erfc<float>(原地) -> bf16：两个 generation 的 f32 scratch（每元素 8B）。
            static constexpr uint32_t FLOAT_SCRATCH_BYTES_PER_ELEM = sizeof(float) * 2U;
            tPipe->InitBuffer(floatBuf_, tileElements_ * FLOAT_SCRATCH_BYTES_PER_ELEM);
        }
    }

    __aicore__ inline void Process()
    {
        if (tensorDataCountList_ == nullptr) {
            return;
        }
        // 本核区间为空（总量 0 / 空列表）：不发任何 DMA。
        if (tensorStart_ > tensorEnd_ || (tensorStart_ == tensorEnd_ && tensorStartOffset_ > tensorEndOffset_)) {
            return;
        }

        uint16_t cursorIdx = tensorStart_;
        int64_t cursorOff = tensorStartOffset_;
        while (cursorIdx < tensorEnd_ || (cursorIdx == tensorEnd_ && cursorOff <= tensorEndOffset_)) {
            LocalTensor<T> inLocal = inQueue_.template AllocTensor<T>();
            uint16_t fillIdx = cursorIdx;
            int64_t fillOff = cursorOff;
            uint32_t packedBytes = WalkSegments<false>(inLocal, fillIdx, fillOff);
            if (packedBytes == 0) {
                inQueue_.FreeTensor(inLocal);
                return;
            }
            inQueue_.EnQue(inLocal);

            LocalTensor<T> inPacked = inQueue_.template DeQue<T>();
            LocalTensor<T> outLocal = outQueue_.template AllocTensor<T>();
            const uint32_t calCount = (packedBytes + static_cast<uint32_t>(sizeof(T)) - 1U) /
                                      static_cast<uint32_t>(sizeof(T));
            Compute(inPacked, outLocal, calCount);
            inQueue_.FreeTensor(inPacked);
            outQueue_.template EnQue<T>(outLocal);

            LocalTensor<T> outPacked = outQueue_.template DeQue<T>();
            uint16_t drainIdx = cursorIdx;
            int64_t drainOff = cursorOff;
            WalkSegments<true>(outPacked, drainIdx, drainOff);
            outQueue_.FreeTensor(outPacked);

            cursorIdx = fillIdx;
            cursorOff = fillOff;
        }
    }

private:
    //! 把本 stage 的段依次搬入(kOut=false)/搬出(kOut=true) UB，返回打包字节数。
    //! 同一个游标推进规则被 fill 与 drain 复用，保证两遍生成的 slot 布局逐字节一致。
    template <bool kOut>
    __aicore__ inline uint32_t WalkSegments(const LocalTensor<T>& local, uint16_t& cursorIdx, int64_t& cursorOff)
    {
        uint32_t slotBytes = 0;
        while (cursorIdx <= tensorEnd_) {
            const int64_t tensorLast = (cursorIdx == tensorEnd_) ? tensorEndOffset_ :
                                                                   (tensorDataCountList_[cursorIdx] - 1);
            const int64_t remain = tensorLast - cursorOff + 1;
            if (remain <= 0) {
                // 空 tensor（numel==0）或本 tensor 已在本 stage 搬完：跳过，不发 0 字节 DMA。
                ++cursorIdx;
                cursorOff = 0;
                continue;
            }
            uint32_t capacity = tileBytes_ - slotBytes;
            if (capacity < static_cast<uint32_t>(sizeof(T))) {
                break;
            }
            const uint64_t remainBytes = static_cast<uint64_t>(remain) * sizeof(T);
            uint32_t bytes = (remainBytes > static_cast<uint64_t>(capacity)) ? capacity :
                                                                               static_cast<uint32_t>(remainBytes);
            bytes -= bytes % static_cast<uint32_t>(sizeof(T));
            if (bytes == 0) {
                break;
            }
            if constexpr (kOut) {
                CopyOutSegment(local, slotBytes, cursorIdx, cursorOff, bytes);
            } else {
                CopyInSegment(local, slotBytes, cursorIdx, cursorOff, bytes);
            }
            cursorOff += bytes / static_cast<uint32_t>(sizeof(T));
            slotBytes = AlignUpBytes(slotBytes + bytes);
            if (cursorOff > tensorLast) {
                ++cursorIdx;
                cursorOff = 0;
            }
            if (tileBytes_ - slotBytes < static_cast<uint32_t>(sizeof(T))) {
                break;
            }
        }
        return slotBytes;
    }

    __aicore__ inline void CopyInSegment(const LocalTensor<T>& local, uint32_t slotBytes, uint16_t tensorIdx,
                                         int64_t tensorOffset, uint32_t bytes)
    {
        GlobalTensor<T> src;
        src.SetGlobalBuffer(inDesc_.GetDataPtr<T>(tensorIdx) + tensorOffset);
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = bytes; // 实际字节：不为 32B 对齐多读 GM
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(local[slotBytes / static_cast<uint32_t>(sizeof(T))], src, copyParams, padParams);
    }

    __aicore__ inline void CopyOutSegment(const LocalTensor<T>& local, uint32_t slotBytes, uint16_t tensorIdx,
                                          int64_t tensorOffset, uint32_t bytes)
    {
        GlobalTensor<T> dst;
        dst.SetGlobalBuffer(outDesc_.GetDataPtr<T>(tensorIdx) + tensorOffset);
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = bytes; // 实际字节
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(dst, local[slotBytes / static_cast<uint32_t>(sizeof(T))], copyParams);
    }

    //! 每 stage 只有一次 Erfc（native），或 Cast/Erfc/Cast 各一次（bf16）。
    __aicore__ inline void Compute(const LocalTensor<T>& inPacked, const LocalTensor<T>& outPacked, uint32_t calCount)
    {
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            LocalTensor<float> floatAll = floatBuf_.Get<float>();
            LocalTensor<float> floatScratch = floatAll[generation_ * tileElements_];
            Cast(floatScratch, inPacked, RoundMode::CAST_NONE, calCount);
            Erfc<float, true>(floatScratch, floatScratch, calCount);
            Cast(outPacked, floatScratch, RoundMode::CAST_RINT, calCount);
        } else {
            // fp32 / fp16 由 Erfc intrinsic 原生支持（half 在 3510 实现内部提升到 fp32 计算后回写）。
            Erfc<T, false>(outPacked, inPacked, calCount);
        }
        generation_ ^= 1U;
    }

    __aicore__ static inline uint32_t AlignUpBytes(uint32_t bytes)
    {
        return (bytes + ERFC_UB_ALIGN_BYTES - 1U) & ~(ERFC_UB_ALIGN_BYTES - 1U);
    }

    TQue<QuePosition::VECIN, ERFC_QUEUE_DEPTH> inQueue_;
    TQue<QuePosition::VECOUT, ERFC_QUEUE_DEPTH> outQueue_;
    TBuf<TPosition::VECCALC> floatBuf_;
    ListTensorDesc inDesc_;
    ListTensorDesc outDesc_;

    int64_t* tensorDataCountList_ = nullptr;
    uint32_t blockIdx_ = 0;

    uint16_t tensorStart_ = 0;
    uint16_t tensorEnd_ = 0;
    int64_t tensorStartOffset_ = 0;
    int64_t tensorEndOffset_ = -1;

    uint32_t tileElements_ = 0;
    uint32_t tileBytes_ = 0;
    uint32_t generation_ = 0;
};

} // namespace ForeachErfc

#endif // OPS_NN_FOREACH_ERFC_REGBASE_H_
