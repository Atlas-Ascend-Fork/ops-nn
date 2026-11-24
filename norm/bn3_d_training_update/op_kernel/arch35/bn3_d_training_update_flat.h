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
 * \file bn3_d_training_update_flat.h
 * \brief 周期展平(flat-periodic)路径:x 视为**一维连续数组**,元素 j 的通道号只由 j 对周期 P 取模决定
 *   channel-last(NHWC / NDHWC):c(j) = j % C            => R = 1、P = C
 *   channel-first(NCHW / NCDHW):c(j) = (j % P) / S      => R = S、P = C·S
 * 于是 y[j] = (x[j] − bm[c(j)])·mult[c(j)] + off[c(j)] 与"行/平面"结构完全解耦:
 * 只要 P 整除瓦片长度(host 保证 laneBlock = m·VL 整除瓦片长度,且 P | laneBlock),
 * 瓦片内每个 VL 块的系数向量就是 pattern 里固定的一个块(或随块号按 m 轮转)。
 *
 * 主循环每 64 元素只有 5 条(fp32)/ 7 条(fp16/bf16)向量指令,lane 利用率 100%,
 * 且整瓦片一次连续 DMA(blockCount = 1)——同时修好"lane 塌陷"与"per-tile 固定开销"两个瓶颈。
 *
 * 与既有两条路径的算式**逐字相同**(非折叠式 Sub → Mul → Add、同一 FoldRegs 系数、fp16/bf16 单次
 * CAST_RINT 舍入),故逐元素结果位级一致。
 */

#ifndef BN3_D_TRAINING_UPDATE_FLAT_H
#define BN3_D_TRAINING_UPDATE_FLAT_H

#include <cstdint>
#include "kernel_operator.h"
#include "bn3_d_training_update_common.h"
#include "bn3_d_training_update_stat.h"
#include "bn3_d_training_update_tiling_data.h"

namespace Bn3dTrainingUpdateOps {

// VF 承载点 ⑤:周期展平的逐元素主循环。
//   unroll == 1:3 路系数块在整个瓦片内常驻寄存器(瓦片只读 1 次);
//   unroll > 1 :瓦片内以 laneBlock 为一个完整周期组,组内逐 m 个 VL 块取对应系数块。
//                **不用块号取模**:周期组内系数块号由内层归纳变量直接给出(嵌套循环),避免 VF 内取模。
// 满块 for + 尾块 0/1 次 for(VF 内无 if 分支的 return);尾块只靠 mask 屏蔽,
// 整 VL 读不越界由 XyTileBufferBytes 预留的 1 个 VL 余量保证。
template <typename T_X>
__simd_vf__ inline void VFFlatStream(__ubuf__ T_X* xAddr, __ubuf__ T_X* yAddr, __ubuf__ float* patMult,
                                     __ubuf__ float* patBm, __ubuf__ float* patOffset, uint32_t extent,
                                     uint32_t laneBlock, uint32_t unroll)
{
    __VEC_SCOPE__
    {
        RegTensor<float> multReg;
        RegTensor<float> bmReg;
        RegTensor<float> offReg;
        RegTensor<float> xReg;
        RegTensor<float> yReg;
        const uint16_t totalBlocks = static_cast<uint16_t>((extent + VL_FP32 - 1) / VL_FP32);
        if (unroll == 1U) {
            const uint16_t fullBlocks = static_cast<uint16_t>(extent / VL_FP32);
            LoadFp32Aligned(multReg, patMult, 0); // 周期长度 <= VL ⇒ 一个系数块覆盖整个瓦片
            LoadFp32Aligned(bmReg, patBm, 0);
            LoadFp32Aligned(offReg, patOffset, 0);
            for (uint16_t b = 0; b < fullBlocks; ++b) {
                const uint32_t offset = static_cast<uint32_t>(b) * VL_FP32;
                uint32_t maskCount = VL_FP32;
                MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(maskCount);
                LoadAlignedXToFp32<T_X>(xReg, xAddr, offset, mask);
                AscendC::Reg::Sub(yReg, xReg, bmReg, mask);   // (x − bm)
                AscendC::Reg::Mul(yReg, yReg, multReg, mask); // × multiplier
                AscendC::Reg::Add(yReg, yReg, offReg, mask);  // + offset
                StoreAlignedYFromFp32<T_X>(yAddr, yReg, offset, mask);
            }
            for (uint16_t b = fullBlocks; b < totalBlocks; ++b) { // 尾块 0 或 1 次,无 if
                const uint32_t offset = static_cast<uint32_t>(b) * VL_FP32;
                uint32_t maskCount = extent - offset;
                MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(maskCount);
                LoadAlignedXToFp32<T_X>(xReg, xAddr, offset, mask);
                AscendC::Reg::Sub(yReg, xReg, bmReg, mask);
                AscendC::Reg::Mul(yReg, yReg, multReg, mask);
                AscendC::Reg::Add(yReg, yReg, offReg, mask);
                StoreAlignedYFromFp32<T_X>(yAddr, yReg, offset, mask);
            }
        } else {
            // 周期组 = laneBlock 个元素(= m 个 VL 块),组内第 j 块对应 pattern 的第 j 块。
            // 组边界按 laneBlock 对齐 ⇒ 每组内的相位恒为 0,故尾组(不足一组)同样从第 0 块起取系数。
            const uint16_t groupBlocks = static_cast<uint16_t>(laneBlock / VL_FP32); // == unroll
            const uint32_t fullGroups = extent / laneBlock;
            const uint16_t fullGroupCount = static_cast<uint16_t>(fullGroups);
            const uint32_t tailElems = extent - fullGroups * laneBlock;
            const uint16_t tailBlocks = static_cast<uint16_t>((tailElems + VL_FP32 - 1) / VL_FP32);
            for (uint16_t g = 0; g < fullGroupCount; ++g) {
                const uint32_t groupBase = static_cast<uint32_t>(g) * laneBlock;
                for (uint16_t j = 0; j < groupBlocks; ++j) {
                    const uint32_t patBlkOff = static_cast<uint32_t>(j) * VL_FP32;
                    const uint32_t offset = groupBase + patBlkOff;
                    uint32_t maskCount = VL_FP32;
                    MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(maskCount);
                    LoadFp32Aligned(multReg, patMult, patBlkOff);
                    LoadFp32Aligned(bmReg, patBm, patBlkOff);
                    LoadFp32Aligned(offReg, patOffset, patBlkOff);
                    LoadAlignedXToFp32<T_X>(xReg, xAddr, offset, mask);
                    AscendC::Reg::Sub(yReg, xReg, bmReg, mask);
                    AscendC::Reg::Mul(yReg, yReg, multReg, mask);
                    AscendC::Reg::Add(yReg, yReg, offReg, mask);
                    StoreAlignedYFromFp32<T_X>(yAddr, yReg, offset, mask);
                }
            }
            for (uint16_t j = 0; j < tailBlocks; ++j) { // 尾组:0 到 groupBlocks-1 块,块尾靠 mask
                const uint32_t patBlkOff = static_cast<uint32_t>(j) * VL_FP32;
                const uint32_t offset = fullGroups * laneBlock + patBlkOff;
                uint32_t maskCount = extent - offset;
                MaskReg mask = AscendC::MicroAPI::UpdateMask<float>(maskCount);
                LoadFp32Aligned(multReg, patMult, patBlkOff);
                LoadFp32Aligned(bmReg, patBm, patBlkOff);
                LoadFp32Aligned(offReg, patOffset, patBlkOff);
                LoadAlignedXToFp32<T_X>(xReg, xAddr, offset, mask);
                AscendC::Reg::Sub(yReg, xReg, bmReg, mask);
                AscendC::Reg::Mul(yReg, yReg, multReg, mask);
                AscendC::Reg::Add(yReg, yReg, offReg, mask);
                StoreAlignedYFromFp32<T_X>(yAddr, yReg, offset, mask);
            }
        }
    }
}

template <typename T_X>
class Bn3dTrainingUpdateFlat {
public:
    __aicore__ inline Bn3dTrainingUpdateFlat() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, GM_ADDR scale, GM_ADDR offset, GM_ADDR mean,
                                GM_ADDR variance, GM_ADDR y, GM_ADDR meanOut, GM_ADDR varianceOut, GM_ADDR batchMean,
                                GM_ADDR batchVariance, const Bn3dTrainingUpdateFlatTilingData* tilingData, TPipe* pipe)
    {
        tilingData_ = tilingData;
        pipe_ = pipe;
        xGm_.SetGlobalBuffer((__gm__ T_X*)x);
        yGm_.SetGlobalBuffer((__gm__ T_X*)y);
        sumGm_.SetGlobalBuffer((__gm__ float*)sum, tilingData->c);
        squareSumGm_.SetGlobalBuffer((__gm__ float*)squareSum, tilingData->c);
        scaleGm_.SetGlobalBuffer((__gm__ float*)scale, tilingData->c);
        offsetGm_.SetGlobalBuffer((__gm__ float*)offset, tilingData->c);

        const int64_t vlFp32 = static_cast<int64_t>(VL_FP32);
        const int64_t elemBytes = static_cast<int64_t>(sizeof(T_X));
        chunkSlotBytes_ = ChunkSlotBytes(vlFp32, UB_BLOCK_BYTES);
        chunkSlotElems_ = chunkSlotBytes_ / static_cast<int64_t>(sizeof(float));
        pipe_->InitBuffer(xQueue_, kXyTileBufferNum,
                          XyTileBufferBytes(tilingData->tileElems, elemBytes, vlFp32, UB_BLOCK_BYTES));
        pipe_->InitBuffer(yQueue_, kXyTileBufferNum,
                          XyTileBufferBytes(tilingData->tileElems, elemBytes, vlFp32, UB_BLOCK_BYTES));
        pipe_->InitBuffer(foldInBuf_, kFoldInputSlots * chunkSlotBytes_);
        pipe_->InitBuffer(foldOutBuf_, kFoldOutputSlots * chunkSlotBytes_);
        patternElems_ = tilingData->laneBlock;
        pipe_->InitBuffer(patternBuf_, kFlatPatternSlots * patternElems_ * static_cast<int64_t>(sizeof(float)));

        Bn3dTrainingUpdateStatDesc desc;
        desc.channel = tilingData->c;
        desc.statUnitCount = tilingData->statUnitCount;
        desc.statChunk = tilingData->statChunk;
        desc.baseStatPerCore = tilingData->baseStatPerCore;
        desc.extraStatCount = tilingData->extraStatCount;
        desc.numRec = tilingData->numRec;
        desc.besselScaler = tilingData->besselScaler;
        desc.factor = tilingData->factor;
        desc.oneMinusFactor = tilingData->oneMinusFactor;
        desc.epsilon = tilingData->epsilon;
        stat_.Init(sum, squareSum, scale, offset, mean, variance, meanOut, varianceOut, batchMean, batchVariance, desc,
                   pipe_);
    }

    __aicore__ inline void Process()
    {
        stat_.Process(); // 相 S 先跑:唯一写 4 路 (C,) 输出的地方
        PhaseX();
    }

private:
    __aicore__ inline __ubuf__ float* ChunkSlot(int64_t slot)
    {
        return (__ubuf__ float*)foldInBuf_.Get<float>()[slot * chunkSlotElems_].GetPhyAddr();
    }

    __aicore__ inline __ubuf__ float* FoldOutSlot(int64_t slot)
    {
        return (__ubuf__ float*)foldOutBuf_.Get<float>()[slot * chunkSlotElems_].GetPhyAddr();
    }

    __aicore__ inline __ubuf__ float* PatternSlot(int64_t slot)
    {
        return (__ubuf__ float*)patternBuf_.Get<float>()[slot * patternElems_].GetPhyAddr();
    }

    __aicore__ inline void StageFoldInput(const GlobalTensor<float>& gm, int64_t slot, int64_t count)
    {
        LocalTensor<float> ub = foldInBuf_.Get<float>()[slot * chunkSlotElems_];
        DataCopyExtParams params{1, static_cast<uint32_t>(count * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(ub, gm[0], params, pad);
    }

    // 每核一次:4 路 (C,) 统计量一次折叠出 mult / bm / offset(C <= 2 ⇒ 单次 VFFoldChunk),
    // 再按 pattern[l] = coeff[(l % period) / channelRun] 展开成 laneBlock 长的周期向量。
    // 标量展开仅 laneBlock <= 256 次,≈1 μs 量级,每核一次(不是每瓦片一次)。
    __aicore__ inline void BuildPattern()
    {
        event_t waitV2Mte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(waitV2Mte2);
        WaitFlag<HardEvent::V_MTE2>(waitV2Mte2);
        StageFoldInput(sumGm_, kFoldInSum, tilingData_->c);
        StageFoldInput(squareSumGm_, kFoldInSquareSum, tilingData_->c);
        StageFoldInput(scaleGm_, kFoldInScale, tilingData_->c);
        StageFoldInput(offsetGm_, kFoldInOffset, tilingData_->c);
        event_t waitMte2V = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(waitMte2V);
        WaitFlag<HardEvent::MTE2_V>(waitMte2V);
        asc_vf_call<VFFoldChunk>(ChunkSlot(kFoldInSum), ChunkSlot(kFoldInSquareSum), ChunkSlot(kFoldInScale),
                                 ChunkSlot(kFoldInOffset), FoldOutSlot(kFlatFoldMult), FoldOutSlot(kFlatFoldBm),
                                 FoldOutSlot(kFlatFoldOffset), FoldOutSlot(kFlatFoldSaveVar), tilingData_->numRec,
                                 tilingData_->epsilon, static_cast<uint32_t>(tilingData_->c));

        // 折叠结果在 UB 内 → 标量展开(V→S → 标量读写 → S→V,用具体事件而非整栏 barrier)。
        event_t waitV2S = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(waitV2S);
        WaitFlag<HardEvent::V_S>(waitV2S);
        LocalTensor<float> foldOut = foldOutBuf_.Get<float>();
        LocalTensor<float> pattern = patternBuf_.Get<float>();
        const int64_t period = tilingData_->period;
        const int64_t channelRun = tilingData_->channelRun;
        const int64_t laneBlock = tilingData_->laneBlock;
        for (int64_t l = 0; l < laneBlock; ++l) {
            const int64_t k = (l % period) / channelRun;
            pattern.SetValue(l, foldOut.GetValue(kFlatFoldMult * chunkSlotElems_ + k));
            pattern.SetValue(laneBlock + l, foldOut.GetValue(kFlatFoldBm * chunkSlotElems_ + k));
            pattern.SetValue(2 * laneBlock + l, foldOut.GetValue(kFlatFoldOffset * chunkSlotElems_ + k));
        }
        event_t waitS2V = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(waitS2V);
        WaitFlag<HardEvent::S_V>(waitS2V);
    }

    // 整瓦片一次连续 DMA(blockCount = 1;等价于既有 CF 的 count == S 形态)。
    __aicore__ inline void CopyIn(int64_t gmBase, int64_t extent)
    {
        LocalTensor<T_X> x = xQueue_.AllocTensor<T_X>();
        DataCopyExtParams params{1, static_cast<uint32_t>(extent * static_cast<int64_t>(sizeof(T_X))), 0, 0, 0};
        DataCopyPadExtParams<T_X> pad{false, 0, 0, 0};
        DataCopyPad(x, xGm_[gmBase], params, pad);
        xQueue_.EnQue(x);
    }

    __aicore__ inline void Compute(int64_t extent)
    {
        LocalTensor<T_X> x = xQueue_.DeQue<T_X>();
        LocalTensor<T_X> y = yQueue_.AllocTensor<T_X>();
        asc_vf_call<VFFlatStream<T_X>>((__ubuf__ T_X*)x.GetPhyAddr(), (__ubuf__ T_X*)y.GetPhyAddr(),
                                       PatternSlot(kFlatPatMult), PatternSlot(kFlatPatBm), PatternSlot(kFlatPatOffset),
                                       static_cast<uint32_t>(extent), static_cast<uint32_t>(tilingData_->laneBlock),
                                       static_cast<uint32_t>(tilingData_->unroll));
        yQueue_.EnQue(y);
        xQueue_.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(int64_t gmBase, int64_t extent)
    {
        LocalTensor<T_X> y = yQueue_.DeQue<T_X>();
        DataCopyExtParams params{1, static_cast<uint32_t>(extent * static_cast<int64_t>(sizeof(T_X))), 0, 0, 0};
        DataCopyPad(yGm_[gmBase], y, params);
        yQueue_.FreeTensor(y);
    }

    __aicore__ inline void PhaseX()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t tileCount = tilingData_->baseTilesPerCore + (blockIdx < tilingData_->extraCoreCount ? 1 : 0);
        if (tileCount <= 0) {
            return; // total == 0:无 X 单元(host 门限保证 total >= 4096,此处仅防御)
        }
        int64_t tileId = blockIdx * tilingData_->baseTilesPerCore +
                         (blockIdx < tilingData_->extraCoreCount ? blockIdx : tilingData_->extraCoreCount);
        BuildPattern(); // 系数只依赖 (C, R),与瓦片无关 ⇒ 每核一次
        for (int64_t i = 0; i < tileCount; ++i, ++tileId) {
            const int64_t gmBase = tileId * tilingData_->tileElems;
            const int64_t remain = tilingData_->total - gmBase;
            const int64_t extent = remain < tilingData_->tileElems ? remain : tilingData_->tileElems;
            CopyIn(gmBase, extent);
            Compute(extent);
            CopyOut(gmBase, extent);
        }
    }

private:
    const Bn3dTrainingUpdateFlatTilingData* tilingData_ = nullptr;
    TPipe* pipe_ = nullptr;
    int64_t chunkSlotBytes_ = 0;
    int64_t chunkSlotElems_ = 0;
    int64_t patternElems_ = 0;
    Bn3dTrainingUpdateStat stat_;
    TBuf<TPosition::VECCALC> foldInBuf_;
    TBuf<TPosition::VECCALC> foldOutBuf_;
    TBuf<TPosition::VECCALC> patternBuf_;
    TQue<QuePosition::VECIN, kXyTileBufferNum> xQueue_;
    TQue<QuePosition::VECOUT, kXyTileBufferNum> yQueue_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_X> yGm_;
    GlobalTensor<float> sumGm_;
    GlobalTensor<float> squareSumGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;
};

} // namespace Bn3dTrainingUpdateOps

#endif // BN3_D_TRAINING_UPDATE_FLAT_H
