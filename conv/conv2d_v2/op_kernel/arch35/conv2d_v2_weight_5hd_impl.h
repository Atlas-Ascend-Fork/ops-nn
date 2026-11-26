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
 * \file conv2d_v2_weight_5hd_impl.h
 * \brief
 */

#ifndef CONV2D_V2_WEIGHT_5HD_IMPL_H
#define CONV2D_V2_WEIGHT_5HD_IMPL_H

#include "conv2d_v2_config.h"
#include "conv2d_v2_util.h"
#include "conv2d_v2_instr.h"

namespace Conv2dFunc {
using namespace AscendC;
using namespace conv;

template <typename WeightT>
struct Weight5HDCfg {
    uint64_t cin = 0;
    uint64_t coutTotal = 0;
    uint64_t hw = 0;
    uint64_t kBL1 = 0;
    uint64_t singleCoreCi = 0;
    uint32_t bUbNStep = 0;
    uint32_t bUbKStep = 0;
    uint64_t aivNum = 0;
    uint64_t blockDim = 0;
    uint64_t isRealCore = 1;
};

template <typename WeightT, typename TilingData>
__aicore__ inline Weight5HDCfg<WeightT> MakeWeight5HDCfg(const TilingData& td, uint64_t aivNum, uint64_t blockDim,
                                                         bool isRealCore)
{
    Weight5HDCfg<WeightT> cfg;
    cfg.cin = td.cin;
    cfg.coutTotal = td.orgCo;
    cfg.hw = static_cast<uint64_t>(td.kernelHxkernelW);
    cfg.kBL1 = td.kBL1;
    cfg.singleCoreCi = td.singleCoreCi;
    cfg.bUbNStep = td.bUbNStep;
    cfg.bUbKStep = td.bUbKStep;
    cfg.aivNum = aivNum;
    cfg.blockDim = blockDim;
    cfg.isRealCore = isRealCore ? 1UL : 0UL;
    return cfg;
}

template <typename WeightT>
class Weight5HDVecTools {
public:
    static constexpr uint32_t K0 = static_cast<uint32_t>(C0_SIZE) / static_cast<uint32_t>(sizeof(WeightT));
    static constexpr bool IS_FP32 = AscendC::IsSameType<WeightT, float>::value;
    static constexpr bool IS_HIF8 = AscendC::IsSameType<WeightT, hifloat8_t>::value;
    using Weight5HDT = typename Conditional<IS_HIF8, uint8_t, WeightT>::type;

    __aicore__ inline Weight5HDVecTools() {}

    static __aicore__ inline uint64_t CalcChunkNum(const Weight5HDCfg<WeightT>& cfg)
    {
        const uint64_t totalK = AlignB(cfg.singleCoreCi, static_cast<uint64_t>(K0)) * cfg.hw;
        return CeilDiv(totalK, cfg.kBL1);
    }

    __aicore__ inline void Init(const Weight5HDCfg<WeightT>& cfg, TPipe& pipe, TBuf<TPosition::VECIN>& ndUbBuf,
                                TBuf<TPosition::VECOUT>& nzUbBuf)
    {
        cfg_ = cfg;
        chunkNum_ = CalcChunkNum(cfg);
        kernelHw_ = cfg.hw;
        alignedHW_ = AlignB(cfg.hw, static_cast<uint64_t>(K0));
        tilesPerSlot_ = CeilDiv(static_cast<uint64_t>(cfg.bUbKStep), static_cast<uint64_t>(K0));
        ndSlotElems_ = static_cast<uint64_t>(cfg.bUbKStep) * alignedHW_;
        nzTileElems_ = alignedHW_ * K0;
        nzSlotElems_ = tilesPerSlot_ * nzTileElems_;

        aivIdx_ = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t coutTotal = cfg.coutTotal;
        nStep_ = CeilDiv(coutTotal, cfg.aivNum);
        nStart_ = aivIdx_ * nStep_;
        nCnt_ = (nStart_ >= coutTotal) ? 0 : (nStep_ < (coutTotal - nStart_) ? nStep_ : (coutTotal - nStart_));

        pipe.InitBuffer(ndUbBuf, static_cast<uint32_t>(cfg.bUbNStep * ndSlotElems_ * sizeof(WeightT)));
        pipe.InitBuffer(nzUbBuf, static_cast<uint32_t>(cfg.bUbNStep * nzSlotElems_ * sizeof(WeightT)));
        ndUbBuf_ = &ndUbBuf;
        nzUbBuf_ = &nzUbBuf;

        evVToMte2_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
        evMte2ToV_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        evVToMte3_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        evMte3ToMte2_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void BindGm(const GlobalTensor<WeightT>& srcGm, const GlobalTensor<WeightT>& dstGm)
    {
        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Weight5HDT*>(srcGm.GetPhyAddr(0)), srcGm.GetSize());
        dstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Weight5HDT*>(dstGm.GetPhyAddr(0)), dstGm.GetSize());
    }

    __aicore__ inline void RunVec()
    {
        const bool isRealCore = cfg_.isRealCore != 0U;
        for (uint64_t chunk = 0; chunk < chunkNum_; ++chunk) {
            if (chunk >= WEIGHT_5HD_EVENT_ID_NUM && isRealCore) {
                CrossCoreWaitFlag<WEIGHT_5HD_SYNC_MODE>(WEIGHT_5HD_SYNC_ID_BASE + (chunk & 1U));
            }
            if (nCnt_ > 0U) {
                ConvertChunk(chunk);
            }
            CrossCoreSetFlag<0, PIPE_MTE3>(WEIGHT_5HD_BARRIER_ID);
            CrossCoreWaitFlag<0, PIPE_MTE3>(WEIGHT_5HD_BARRIER_ID);
            if (isRealCore) {
                CrossCoreSetFlag<WEIGHT_5HD_SYNC_MODE, PIPE_MTE3>(WEIGHT_5HD_SYNC_ID_BASE + (chunk & 1U));
            }
        }
    }

private:
    __aicore__ inline void ConvertChunk(uint64_t chunk)
    {
        const uint64_t cinPerChunk = cfg_.kBL1 / kernelHw_;
        const uint64_t c1PerChunk = cinPerChunk / K0;
        const uint64_t alignCin = AlignB(cfg_.singleCoreCi, static_cast<uint64_t>(K0));
        const uint64_t chunkCin = cinPerChunk < (alignCin - chunk * cinPerChunk) ? cinPerChunk :
                                                                                   (alignCin - chunk * cinPerChunk);
        const uint64_t c1Base = chunk * c1PerChunk;
        const uint64_t ciLoops = CeilDiv(chunkCin, static_cast<uint64_t>(cfg_.bUbKStep));
        const uint64_t nLoops = CeilDiv(nCnt_, static_cast<uint64_t>(cfg_.bUbNStep));
        for (uint64_t nl = 0; nl < nLoops; nl++) {
            const uint64_t curN = (cfg_.bUbNStep < (nCnt_ - nl * cfg_.bUbNStep)) ? cfg_.bUbNStep :
                                                                                   (nCnt_ - nl * cfg_.bUbNStep);
            const uint64_t nBase = nStart_ + nl * cfg_.bUbNStep;
            for (uint64_t ci = 0; ci < ciLoops; ci++) {
                const uint64_t curCin = (cfg_.bUbKStep < (chunkCin - ci * cfg_.bUbKStep)) ?
                                            cfg_.bUbKStep :
                                            (chunkCin - ci * cfg_.bUbKStep);
                const uint64_t kLocal = (c1Base + ci * (static_cast<uint64_t>(cfg_.bUbKStep) / K0)) * kernelHw_ * K0;
                ConvertBatch(nBase, curN, kLocal, curCin);
            }
        }
    }

    __aicore__ inline void ConvertBatch(uint64_t nBase, uint64_t nCnt, uint64_t kLocal, uint64_t cinCnt)
    {
        LocalTensor<Weight5HDT> nd = ndUbBuf_->template Get<Weight5HDT>();
        LocalTensor<Weight5HDT> nz = nzUbBuf_->template Get<Weight5HDT>();
        uint64_t tileCnt = CeilDiv(cinCnt, static_cast<uint64_t>(K0));

        PipeBarrier<PIPE_V>();
        Duplicate(nd, static_cast<Weight5HDT>(0), static_cast<int32_t>(cfg_.bUbNStep * ndSlotElems_));
        SetFlag<HardEvent::V_MTE2>(evVToMte2_);
        WaitFlag<HardEvent::V_MTE2>(evVToMte2_);
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = static_cast<uint16_t>(cinCnt);
        copyInParams.blockLen = static_cast<uint32_t>(kernelHw_ * sizeof(Weight5HDT));
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPadExtParams<Weight5HDT> padParams{false, 0, 0, 0};
        for (uint64_t i = 0; i < nCnt; i++) {
            uint64_t srcBase = (nBase + i) * cfg_.cin * kernelHw_ + kLocal;
            DataCopyPad(nd[i * ndSlotElems_], srcGm_[srcBase], copyInParams, padParams);
        }

        SetFlag<HardEvent::MTE2_V>(evMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(evMte2ToV_);
        for (uint64_t i = 0; i < nCnt; i++) {
            TransTile(nd[i * ndSlotElems_], nz[i * nzSlotElems_], tileCnt);
        }

        SetFlag<HardEvent::V_MTE3>(evVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(evVToMte3_);
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = static_cast<uint16_t>(tileCnt);
        copyOutParams.blockLen = static_cast<uint32_t>(kernelHw_ * K0 * sizeof(Weight5HDT));
        copyOutParams.srcStride = (alignedHW_ - kernelHw_) * K0 * sizeof(Weight5HDT) / C0_SIZE;
        copyOutParams.dstStride = 0;
        for (uint64_t i = 0; i < nCnt; i++) {
            const uint64_t dstBase = (nBase + i) * CeilDiv(cfg_.cin, static_cast<uint64_t>(K0)) * kernelHw_ * K0 +
                                     kLocal;
            DataCopyPad(dstGm_[dstBase], nz[i * nzSlotElems_], copyOutParams);
        }
        SetFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(evMte3ToMte2_);
    }

    __aicore__ inline void TransTile(LocalTensor<Weight5HDT> nd, LocalTensor<Weight5HDT> nz, uint64_t tileCnt)
    {
        TransDataTo5HDParams transParams;
        transParams.srcHighHalf = false;
        transParams.dstHighHalf = false;
        transParams.repeatTimes = static_cast<uint16_t>(alignedHW_ / K0);
        const bool singleRepeat = (transParams.repeatTimes == 1U);
        transParams.srcRepStride = singleRepeat ? 0U : 1U;

        uint64_t srcList[WEIGHT_5HD_ADDR_LIST_SIZE];
        uint64_t dstList[WEIGHT_5HD_ADDR_LIST_SIZE];
        if constexpr (IS_HIF8) {
            transParams.dstRepStride = singleRepeat ? 0U : WEIGHT_5HD_DST_REP_STRIDE_32;
            for (uint64_t c1 = 0; c1 < tileCnt; c1++) {
                const uint64_t dstTile = c1 * nzTileElems_;
                for (uint64_t chHalf = 0; chHalf < 2; chHalf++) { // 2 = dst 高/低半区数，hif8 单次 trans 搬一个半区
                    for (uint64_t i = 0; i < WEIGHT_5HD_ADDR_LIST_SIZE; i++) {
                        srcList[i] = static_cast<uint64_t>(
                            nd[(c1 * K0 + chHalf * WEIGHT_5HD_ADDR_LIST_SIZE + i) * alignedHW_].GetPhyAddr());
                    }
                    for (uint64_t posHalf = 0; posHalf < 2; posHalf++) { // 2 = src 高/低半区数
                        transParams.srcHighHalf = (posHalf == 1);
                        transParams.dstHighHalf = (chHalf == 1);
                        for (uint64_t i = 0; i < WEIGHT_5HD_ADDR_LIST_SIZE; i++) {
                            dstList[i] = static_cast<uint64_t>(
                                nz[dstTile + (posHalf * WEIGHT_5HD_ADDR_LIST_SIZE + i) * K0].GetPhyAddr());
                        }
                        TransDataTo5HD<Weight5HDT>(dstList, srcList, transParams);
                    }
                }
            }
        } else if constexpr (IS_FP32) {
            transParams.dstRepStride = singleRepeat ? 0U : WEIGHT_5HD_DST_REP_STRIDE_8;
            const uint64_t c1Pairs = CeilDiv(tileCnt, 2); // 2 = fp32 单次 trans 打包相邻 2 个 c1（k0=8，2×8=16 通道）
            for (uint64_t p = 0; p < c1Pairs; p++) {
                const uint64_t c1A = p * 2; // 每对跨 2 个 c1
                const uint64_t c1B = c1A + 1;
                for (uint64_t j = 0; j < WEIGHT_5HD_ADDR_HALF_LIST_SIZE; j++) { // 16 项地址 = 8 对
                    dstList[2 * j] = static_cast<uint64_t>(
                        nz[c1A * nzTileElems_ + j * K0].GetPhyAddr()); // 偶数项取 c1A
                    dstList[2 * j + 1] = static_cast<uint64_t>(        // 奇数项取 c1B
                        nz[c1B * nzTileElems_ + j * K0].GetPhyAddr());
                }
                for (uint64_t i = 0; i < WEIGHT_5HD_ADDR_LIST_SIZE; i++) {
                    srcList[i] = static_cast<uint64_t>(
                        nd[(p * WEIGHT_5HD_ADDR_LIST_SIZE + i) * alignedHW_].GetPhyAddr());
                }
                TransDataTo5HD<Weight5HDT>(dstList, srcList, transParams);
            }
        } else {
            transParams.dstRepStride = singleRepeat ? 0U : WEIGHT_5HD_DST_REP_STRIDE_16;
            for (uint64_t c1 = 0; c1 < tileCnt; c1++) {
                const uint64_t dstTile = c1 * nzTileElems_;
                for (uint64_t i = 0; i < WEIGHT_5HD_ADDR_LIST_SIZE; i++) {
                    srcList[i] = static_cast<uint64_t>(nd[(c1 * K0 + i) * alignedHW_].GetPhyAddr());
                    dstList[i] = static_cast<uint64_t>(nz[dstTile + i * K0].GetPhyAddr());
                }
                TransDataTo5HD<Weight5HDT>(dstList, srcList, transParams);
            }
        }
    }

    Weight5HDCfg<WeightT> cfg_;
    TBuf<TPosition::VECIN>* ndUbBuf_ = nullptr;
    TBuf<TPosition::VECOUT>* nzUbBuf_ = nullptr;
    GlobalTensor<Weight5HDT> srcGm_;
    GlobalTensor<Weight5HDT> dstGm_;

    uint64_t aivIdx_ = 0;
    uint64_t nStep_ = 0;
    uint64_t nStart_ = 0;
    uint64_t nCnt_ = 0;
    uint64_t chunkNum_ = 0;
    uint64_t kernelHw_ = 0;
    uint64_t alignedHW_ = 0;
    uint64_t tilesPerSlot_ = 0;
    uint64_t ndSlotElems_ = 0;
    uint64_t nzTileElems_ = 0;
    uint64_t nzSlotElems_ = 0;

    event_t evVToMte2_ = static_cast<event_t>(0);    // V_MTE2
    event_t evMte2ToV_ = static_cast<event_t>(0);    // MTE2_V
    event_t evVToMte3_ = static_cast<event_t>(0);    // V_MTE3
    event_t evMte3ToMte2_ = static_cast<event_t>(0); // MTE3_MTE2
};

template <class Intf>
class Weight5HDProcessTools {
public:
    __aicore__ inline Weight5HDProcessTools() {}

    __aicore__ inline void Weight5HDAicInit(Intf* self)
    {
        chunkNum_ = MakeCfg(self).chunkNum_;
        loadedChunkCnt_ = 0;
    }

    __aicore__ inline void BL1SyncLoad(Intf* self, uint64_t kBL1Iter, uint64_t nBL1Iter)
    {
        self->ctx.bl1 = self->ctx.queueBL1.template AllocTensor<typename Intf::WeightT>();
        if (kBL1Iter >= loadedChunkCnt_) {
            CrossCoreWaitFlag<WEIGHT_5HD_SYNC_MODE, PIPE_MTE2>(WEIGHT_5HD_SYNC_ID_BASE + (kBL1Iter & 1));
            self->ctx.loadBL1Ins.LoadBL1From5HD(kBL1Iter, nBL1Iter);
            if (kBL1Iter + WEIGHT_5HD_EVENT_ID_NUM < chunkNum_) {
                CrossCoreSetFlag<WEIGHT_5HD_SYNC_MODE, PIPE_MTE2>(WEIGHT_5HD_SYNC_ID_BASE + (kBL1Iter & 1));
            }
            loadedChunkCnt_++;
        } else {
            self->ctx.loadBL1Ins.LoadBL1From5HD(kBL1Iter, nBL1Iter);
        }
        self->ctx.queueBL1.template EnQue(self->ctx.bl1);
    }

private:
    struct Weight5HDCfgWrapper : public Weight5HDCfg<typename Intf::WeightT> {
        uint64_t chunkNum_ = 0;
    };

    __aicore__ inline Weight5HDCfgWrapper MakeCfg(const Intf* self)
    {
        const auto* td = self->ctx.convTilingData;
        const uint64_t numBlocks = static_cast<uint64_t>(td->batchDim) * td->hoDim * td->woDim * td->nDim;
        Weight5HDCfgWrapper cfg;
        static_cast<Weight5HDCfg<typename Intf::WeightT>&>(cfg) = MakeWeight5HDCfg<typename Intf::WeightT>(
            *td, static_cast<uint64_t>(GetBlockNum()), numBlocks, static_cast<uint64_t>(GetBlockIdx()) < numBlocks);
        cfg.chunkNum_ = Weight5HDVecTools<typename Intf::WeightT>::CalcChunkNum(cfg);
        return cfg;
    }

    uint64_t chunkNum_ = 0;
    uint64_t loadedChunkCnt_ = 0;
};

template <typename WeightT>
__aicore__ inline void RunWeight5HDVecCore(const Weight5HDCfg<WeightT>& cfg, GM_ADDR filter, GM_ADDR workspace)
{
    TPipe pipe;
    TBuf<TPosition::VECIN> ndUbBuf;
    TBuf<TPosition::VECOUT> nzUbBuf;
    GlobalTensor<WeightT> srcGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightT*>(filter));
    GlobalTensor<WeightT> dstGm;
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightT*>(workspace));

    Weight5HDVecTools<WeightT> tools;
    tools.Init(cfg, pipe, ndUbBuf, nzUbBuf);
    tools.BindGm(srcGm, dstGm);
    tools.RunVec();
}

template <typename WeightT, typename TilingData>
__aicore__ inline void RunWeight5HDVec(const TilingData& td, GM_ADDR filter, GM_ADDR workspace, bool isRealCore)
{
    const uint64_t numBlocks = static_cast<uint64_t>(td.batchDim) * td.hoDim * td.woDim * td.nDim;
    const Weight5HDCfg<WeightT> cfg = MakeWeight5HDCfg<WeightT>(td, static_cast<uint64_t>(GetBlockNum()), numBlocks,
                                                                isRealCore);
    RunWeight5HDVecCore(cfg, filter, workspace);
}

template <typename WeightT, typename TilingData>
__aicore__ inline void RunWeight5HDVec(const TilingData& td, GM_ADDR filter, GM_ADDR workspace)
{
    Weight5HDCfg<WeightT> cfg;
    cfg.cin = td.cin;
    cfg.coutTotal = td.cout;
    cfg.hw = static_cast<uint64_t>(td.kh) * td.kw;
    cfg.kBL1 = td.kBL1;
    cfg.singleCoreCi = td.singleCoreCi;
    cfg.bUbNStep = td.bUbNStep;
    cfg.bUbKStep = td.bUbKStep;
    cfg.aivNum = static_cast<uint64_t>(GetBlockNum());
    cfg.blockDim = static_cast<uint64_t>(td.batchDim * td.nDim * td.hoDim);
    const uint64_t aivIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t nxm = static_cast<uint64_t>(td.nDim) * td.hoDim;
    const uint64_t nIdx = (aivIdx % nxm) / static_cast<uint64_t>(td.hoDim);
    const uint64_t mIdx = (aivIdx % nxm) % static_cast<uint64_t>(td.hoDim);
    const bool isRealCore = (nIdx * static_cast<uint64_t>(td.singleCoreCo) < static_cast<uint64_t>(td.cout)) &&
                            (mIdx * static_cast<uint64_t>(td.singleCoreHo) <
                             static_cast<uint64_t>(td.hout) * td.wout) &&
                            (aivIdx < cfg.blockDim);
    cfg.isRealCore = isRealCore ? 1UL : 0UL;
    RunWeight5HDVecCore(cfg, filter, workspace);
}

} // namespace Conv2dFunc

#endif // CONV2D_V2_WEIGHT_5HD_IMPL_H
