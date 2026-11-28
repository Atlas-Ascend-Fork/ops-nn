/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * WeightPartloadProcessor: weight not fully L1-resident (kBL1 < full K and/or nBL1 <
 * singleCoreCo), reloaded per (K block x N tile) slice during the K reduction; scope
 * mirrors the GetSmallKernelVal gate. FmapLoadMode: FULL = block window resident; SPLIT
 * = kAL1-sized chunks; FULL_TO_SPLIT = the CinSplit ladder. Weight ping-pong iff
 * pBufferFlag bit4, L0A/L0B per bit0/bit1. K runs = weight K block n FM chunk.
 */
#ifndef WEIGHT_PARTLOAD_PROCESSOR_H
#define WEIGHT_PARTLOAD_PROCESSOR_H

#include "../conv2d_small_kernel_common.h"
#include "../conv2d_small_kernel_block_ctx.h"
#include "../conv2d_small_kernel_tile_ops.h"

namespace small_conv {

enum class FmapLoadMode : uint8_t {
    FULL,
    SPLIT,
    FULL_TO_SPLIT,
};

// Event ids (per (src,dst) pipe pair): 0/1 M_MTE1 L0 ping-pong (0 = both-closed protocol);
// 1/2 FM buffers; 3/4 weight buffers; 5 bias L1 done / BT sync; 7 bias group token.
constexpr event_t EVT_WP_W0 = static_cast<event_t>(3);
constexpr event_t EVT_WP_W1 = static_cast<event_t>(4);
constexpr event_t EVT_WP_BT_SYNC = static_cast<event_t>(6); // M_MTE1: extra BT reuse sync
constexpr uint32_t WP_INVALID_IDX = ~0u;
constexpr uint32_t WP_L1_PB_BL1_BIT = 4;  // pBufferFlag bit: BL1 ping-pong reserved
constexpr uint32_t WP_L0A_PP_FLAG = 0x01; // pBufferFlag bit0: L0A ping-pong
constexpr uint32_t WP_L0B_PP_FLAG = 0x02; // pBufferFlag bit1: L0B ping-pong
constexpr uint32_t WP_GEN_SHIFT = 16;     // FM cache tag: (spatialGen << 16) | fmChunk

template <typename BlockCtx>
struct WpGroupCtx {
    GlobalTensor<typename BlockCtx::FmapT> fmapGm;
    GlobalTensor<typename BlockCtx::WeightT> filterGm;
    GlobalTensor<uint64_t> scale0Gm;
    GlobalTensor<uint64_t> scale1Gm;
    GlobalTensor<float> reluWeight0Gm;
    GlobalTensor<float> reluWeight1Gm;
    GM_ADDR y;
    GM_ADDR y1;
    uint32_t groupCoutOff;
    uint32_t n1Total; // per-group n1 row width of the NZ weight block
};

struct WpFmCache {
    uint32_t loaded[DOUBLE_BUFFER_NUM]; // (gen << WP_GEN_SHIFT) | fmChunk; INVALID = stale
    bool pend[DOUBLE_BUFFER_NUM];       // load issued, not yet consumed
    Load3DCtx ctx[DOUBLE_BUFFER_NUM];   // per-buffer Load3D config
    uint32_t ctxTag[DOUBLE_BUFFER_NUM]; // tag the ctx was built for
    uint32_t gen;                       // spatial-chunk generation
};

struct WpNextUnit {
    bool valid;
    bool isBatch;
    uint32_t nextBatchIdx;
    uint32_t curHi;
    uint32_t hiLoadOff;
    uint32_t curWi;
    uint32_t wiLoadOff;
};

template <typename BlockCtx, typename TilingData, typename Scheme, FmapLoadMode FmMode, bool OneBatch>
struct WeightPartloadProcessor {
    using FmapT = typename BlockCtx::FmapT;
    using WeightT = typename BlockCtx::WeightT;
    using BiasT = typename BlockCtx::BiasT;
    using OutputT = typename BlockCtx::OutputT;
    using Output1T = typename BlockCtx::Output1T;
    using L0cT = typename BlockCtx::L0cT;
    using GroupCtx = WpGroupCtx<BlockCtx>;
    static constexpr uint32_t GK0 = BlockCtx::GK0;
    static constexpr uint32_t GK0Fmap = BlockCtx::GK0Fmap;

    Scheme scheme_;
    ComputeL1LayoutTool<BlockCtx, TilingData> l1Layout_;
    SetupGmTool<BlockCtx, TilingData> setupGm_;
    SetupFmapGmBatchTool<BlockCtx, TilingData> setupFmapGmBatch_;
    LoadBiasScaleL1Tool<BlockCtx, TilingData> loadBiasScale_;
    LoadBiasToBTTool<BlockCtx, TilingData> loadBiasToBt_;
    SetupLoad3DChunkTool<BlockCtx, TilingData> setupLoad3dChunk_;
    LoadFmapChunkTool<BlockCtx, TilingData> loadFmapChunk_;
    LoadAL0Tool<BlockCtx> loadAl0_;
    InitMmadParamsTool<BlockCtx, TilingData> initMmadParams_;
    CopyOutTool<BlockCtx, TilingData> copyOut_;
    LoadWeightBlockTool<BlockCtx, TilingData> loadWeightBlock_;
    LoadBL0BlockTool<BlockCtx> loadBl0Block_;

    static __aicore__ inline uint32_t KernelHxW(const TilingData& tiling)
    {
        return tiling.kernelHxkernelW == 0 ? tiling.kh * tiling.kw : tiling.kernelHxkernelW;
    }

    // FTS ladder aligned to the weight k0 (A16W8: GK0 = 2 x GK0Fmap).
    static __aicore__ inline uint32_t CinL1(const BlockCtx& blockCtx, const TilingData& tiling)
    {
        if constexpr (FmMode == FmapLoadMode::FULL) {
            return static_cast<uint32_t>(AlignB(tiling.singleCoreCi, GK0Fmap));
        } else if constexpr (FmMode == FmapLoadMode::SPLIT) {
            uint32_t kStepRaw = tiling.kAL1 < tiling.kBL1 ? static_cast<uint32_t>(tiling.kAL1) :
                                                            static_cast<uint32_t>(tiling.kBL1);
            return static_cast<uint32_t>(AlignB(kStepRaw / KernelHxW(tiling), GK0Fmap));
        } else {
            uint32_t cin = blockCtx.cinAligned;
            if (tiling.kernelHxkernelW == 1) {
                if (cin >= CIN_L1_SPLIT_QUARTER * MIN_GK0_MULTIPLIER_PER_SPLIT_1X1 * GK0) {
                    return static_cast<uint32_t>(AlignB(cin / CIN_L1_SPLIT_QUARTER, GK0));
                }
                return static_cast<uint32_t>(AlignB(cin / CIN_L1_SPLIT_HALF, GK0));
            }
            if (cin >= CIN_L1_SPLIT_QUARTER * GK0) {
                return static_cast<uint32_t>(AlignB(cin / CIN_L1_SPLIT_QUARTER, GK0));
            }
            return static_cast<uint32_t>(AlignB(cin / CIN_L1_SPLIT_HALF, GK0));
        }
    }

    static __aicore__ inline uint32_t KStepFm(const BlockCtx& blockCtx, const TilingData& tiling)
    {
        return CinL1(blockCtx, tiling) * KernelHxW(tiling);
    }

    static __aicore__ inline uint32_t FmBufCnt(const TilingData& tiling)
    {
        if constexpr (FmMode == FmapLoadMode::FULL) {
            return 1;
        } else if constexpr (FmMode == FmapLoadMode::SPLIT) {
            return (tiling.pBufferFlag & AL1_DB_FLAG) != 0 ? DOUBLE_BUFFER_NUM : 1;
        } else {
            return DOUBLE_BUFFER_NUM;
        }
    }

    static __aicore__ inline uint32_t WtBufCnt(const TilingData& tiling)
    {
        return ((static_cast<uint32_t>(tiling.pBufferFlag) >> WP_L1_PB_BL1_BIT) & 1) != 0 ? DOUBLE_BUFFER_NUM : 1;
    }

    static __aicore__ inline uint32_t KBlkCnt(const BlockCtx& blockCtx, uint32_t kBL1Eff)
    {
        return static_cast<uint32_t>(CeilDiv(blockCtx.kTotalFmap, kBL1Eff));
    }

    static __aicore__ inline uint32_t NTileStep(const TilingData& tiling)
    {
        return tiling.nL0 < tiling.nBL1 ? static_cast<uint32_t>(tiling.nL0) : static_cast<uint32_t>(tiling.nBL1);
    }

    static __aicore__ inline uint32_t Bl1ElemPerBuf(const TilingData& tiling, uint32_t k1Blk)
    {
        uint32_t n1Tile = static_cast<uint32_t>(AlignB(NTileStep(tiling), GN0)) / GN0;
        return k1Blk * n1Tile * GN0 * GK0;
    }

    __aicore__ inline uint32_t Al1ElemPerBuf(const BlockCtx& blockCtx, const TilingData& tiling) const
    {
        FmapRange blockWindow = scheme_.ResolveBlockWindow(blockCtx, tiling);
        if constexpr (FmMode == FmapLoadMode::FULL) {
            return blockWindow.curHi * blockWindow.orgWin * CinL1(blockCtx, tiling);
        } else {
            // Worst-case real window height of an L0 chunk (middle chunk, unclipped);
            // edge chunks clip pads and are smaller.
            uint32_t maxHoRelEnd = scheme_.MaxChunkHoSpan(blockCtx, tiling);
            uint32_t maxHiL1 = maxHoRelEnd * tiling.strideH + tiling.dilationH * (tiling.kh - 1) + 1;
            uint32_t hinFull = static_cast<uint32_t>(tiling.hin) + tiling.padTop + tiling.padBottom;
            if (maxHiL1 > hinFull) {
                maxHiL1 = hinFull;
            }
            return maxHiL1 * blockWindow.orgWin * CinL1(blockCtx, tiling);
        }
    }

    // L1 budget: on overflow cap the per-buffer k1 rows and walk more weight
    // k-blocks; ping-pong buffer counts are tiling decisions and never reduced.
    __aicore__ inline uint32_t K1BlkCap(const BlockCtx& blockCtx, const TilingData& tiling) const
    {
        uint32_t k1Blk = static_cast<uint32_t>(tiling.kBL1) / GK0;
        L1Layout full = l1Layout_(tiling, FmBufCnt(tiling), Al1ElemPerBuf(blockCtx, tiling),
                                  Bl1ElemPerBuf(tiling, k1Blk), WtBufCnt(tiling));
        if (full.endBytes <= TOTAL_L1_SIZE) {
            return k1Blk;
        }
        uint32_t bl1Bytes = full.bl1ElemCount * sizeof(WeightT);
        uint32_t tailBytes = full.endBytes - full.bl1OffBytes - bl1Bytes;
        uint32_t n1Tile = static_cast<uint32_t>(AlignB(NTileStep(tiling), GN0)) / GN0;
        uint32_t rowBytes = n1Tile * GN0 * GK0 * sizeof(WeightT);
        uint32_t reserve = full.bl1OffBytes + tailBytes + ADDR_ALIGN_SIZE;
        if (reserve >= TOTAL_L1_SIZE) {
            return 1;
        }
        uint32_t k1Cap = (TOTAL_L1_SIZE - reserve) / (WtBufCnt(tiling) * rowBytes);
        return k1Cap < k1Blk ? (k1Cap > 0 ? k1Cap : 1) : k1Blk;
    }

    __aicore__ inline L1Layout MakeL1(const BlockCtx& blockCtx, const TilingData& tiling) const
    {
        return l1Layout_(tiling, FmBufCnt(tiling), Al1ElemPerBuf(blockCtx, tiling),
                         Bl1ElemPerBuf(tiling, K1BlkCap(blockCtx, tiling)), WtBufCnt(tiling));
    }

    static __aicore__ inline event_t FmEv(uint32_t buf) { return buf == 0 ? EVT_FMAP_BUF0 : EVT_FMAP_BUF1; }

    static __aicore__ inline event_t WEv(uint32_t buf) { return buf == 0 ? EVT_WP_W0 : EVT_WP_W1; }

    static __aicore__ inline uint32_t FmBufOf(const TilingData& tiling, uint32_t fmChunk)
    {
        return FmBufCnt(tiling) == 1 ? 0 : (fmChunk & 1);
    }

    static __aicore__ inline uint32_t FmTag(uint32_t gen, uint32_t fmChunk) { return (gen << WP_GEN_SHIFT) | fmChunk; }

    static __aicore__ inline void CalcFmChunk(const TilingData& tiling, uint32_t cinL1, uint32_t fmChunk,
                                              uint32_t& cinOff, uint32_t& curCinOri)
    {
        cinOff = fmChunk * cinL1;
        curCinOri = cinL1;
        if (cinOff + curCinOri > static_cast<uint32_t>(tiling.singleCoreCi)) {
            curCinOri = static_cast<uint32_t>(tiling.singleCoreCi) - cinOff;
        }
    }

    __aicore__ inline void IssueFmLoad(const BlockCtx& blockCtx, const TilingData& tiling, const GroupCtx& gm,
                                       const L1Layout& l1, uint32_t buf, uint32_t fmChunk, uint32_t curHi,
                                       uint32_t hiLoadOff, uint32_t curWi, uint32_t wiLoadOff) const
    {
        uint32_t cinOff;
        uint32_t curCinOri;
        CalcFmChunk(tiling, CinL1(blockCtx, tiling), fmChunk, cinOff, curCinOri);
        loadFmapChunk_(scheme_, tiling, gm.fmapGm, l1.al1BufBytes, buf, curHi, hiLoadOff, curWi, wiLoadOff, cinOff,
                       curCinOri);
        SetFlag<HardEvent::MTE2_MTE1>(FmEv(buf));
    }

    __aicore__ inline void LoadWeightBlockTo(const BlockCtx& blockCtx, const TilingData& tiling, const GroupCtx& gm,
                                             const L1Layout& l1, uint32_t kOff, uint32_t curK, uint32_t nOff,
                                             uint32_t n1Tile, uint32_t wbuf, uint32_t k1Blk) const
    {
        LocalTensor<WeightT> bl1(TPosition::B1, l1.bl1OffBytes + wbuf * Bl1ElemPerBuf(tiling, k1Blk) * sizeof(WeightT),
                                 Bl1ElemPerBuf(tiling, k1Blk));
        uint32_t n1Total = gm.n1Total;
        loadWeightBlock_(blockCtx, tiling, gm.filterGm, bl1, kOff, curK, nOff, n1Tile, n1Total);
    }

    __aicore__ inline void operator()(const BlockCtx& blockCtx, const TilingData& tiling, GM_ADDR x, GM_ADDR filter,
                                      GM_ADDR bias, GM_ADDR y, const ExtendParams* ep) const
    {
        if (!blockCtx.active || blockCtx.actualCo == 0 || blockCtx.singleCoreBatch == 0) {
            return;
        }
        L1Layout l1 = MakeL1(blockCtx, tiling);

        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
        SetFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF0); // all modes: FULL pairs release/wait per (group, batch) window
        if (FmBufCnt(tiling) == DOUBLE_BUFFER_NUM) {
            SetFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF1);
        }
        SetFlag<HardEvent::MTE1_MTE2>(EVT_WP_W0);
        if (WtBufCnt(tiling) == DOUBLE_BUFFER_NUM) {
            SetFlag<HardEvent::MTE1_MTE2>(EVT_WP_W1);
        }

        uint32_t l0pp = 0;
        bool btUsed = false;
        Process(blockCtx, tiling, l1, x, filter, bias, y, ep, l0pp, btUsed);

        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
        WaitFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF0);
        if (FmBufCnt(tiling) == DOUBLE_BUFFER_NUM) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF1);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVT_WP_W0);
        if (WtBufCnt(tiling) == DOUBLE_BUFFER_NUM) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVT_WP_W1);
        }
    }

private:
    __aicore__ inline void Process(const BlockCtx& blockCtx, const TilingData& tiling, const L1Layout& l1, GM_ADDR x,
                                   GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const ExtendParams* ep, uint32_t& l0pp,
                                   bool& btUsed) const
    {
        SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
        // Group scale/relu reload token: FIX_MTE2, because the L1 scale/relu
        // buffers are read by the Fixpipe (FIX queue), not by MTE1. The FIX
        // queue is empty here, so this arm fires immediately.
        SetFlag<HardEvent::FIX_MTE2>(EVT_GROUP_SCALE_DONE);
        for (uint32_t groupIter = 0; groupIter < scheme_.GroupIters(blockCtx); groupIter++) {
            GroupPlan groupPlan = scheme_.GroupAt(blockCtx, tiling, groupIter);
            if (!groupPlan.valid) {
                continue;
            }
            blockCtx.actualCo = groupPlan.actualCo;

            GroupCtx gm;
            gm.y = y;
            gm.y1 = (tiling.dualOutput != 0) ? ep->y1 : y;
            gm.groupCoutOff = groupPlan.groupCoutOff;
            gm.n1Total = static_cast<uint32_t>(AlignB(blockCtx.groupCoutStep, GN0)) / GN0;
            GlobalTensor<BiasT> biasGm;
            GlobalTensor<OutputT> outputGm;
            GlobalTensor<Output1T> output1Gm;
            setupGm_(blockCtx, tiling, x, filter, bias, y, ep, groupIter, groupPlan.groupCoutOff,
                     groupPlan.groupChanOff, scheme_.OutBaseOff(blockCtx, tiling), gm.fmapGm, gm.filterGm, biasGm,
                     outputGm, output1Gm, gm.scale0Gm, gm.scale1Gm, gm.reluWeight0Gm, gm.reluWeight1Gm);

            // Wait only when a reload happens, else the token unbalances (!hasBias).
            if (tiling.hasBias) {
                WaitFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
            }
            // Hold the scale/relu reload until the previous group's last
            // Fixpipe finished reading the L1 vector scale/relu buffers.
            WaitFlag<HardEvent::FIX_MTE2>(EVT_GROUP_SCALE_DONE);
            loadBiasScale_(blockCtx, tiling, biasGm, gm.scale0Gm, gm.scale1Gm, gm.reluWeight0Gm, gm.reluWeight1Gm, l1);
            SetFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
            bool biasL1Waited = false;

            WpFmCache fmCache = {
                {WP_INVALID_IDX, WP_INVALID_IDX}, {false, false}, {{}, {}}, {WP_INVALID_IDX, WP_INVALID_IDX}, 0};
            if constexpr (OneBatch) {
                ProcessChunks(blockCtx, tiling, l1, gm, x, groupPlan.groupChanOff, 0, fmCache, l0pp, btUsed,
                              biasL1Waited);
            } else {
                for (uint32_t batchIdx = 0; batchIdx < blockCtx.singleCoreBatch; batchIdx++) {
                    blockCtx.curBatchIdx = blockCtx.batchStart + batchIdx;
                    setupFmapGmBatch_(tiling, x, blockCtx.curBatchIdx, groupPlan.groupChanOff, gm.fmapGm);
                    ProcessChunks(blockCtx, tiling, l1, gm, x, groupPlan.groupChanOff, batchIdx, fmCache, l0pp, btUsed,
                                  biasL1Waited);
                }
            }
            if (tiling.hasBias) {
                SetFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
            }
            // Enqueued on the FIX queue behind this group's Fixpipes: releases
            // the next group's scale/relu reload once they all completed.
            SetFlag<HardEvent::FIX_MTE2>(EVT_GROUP_SCALE_DONE);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVT_GROUP_BIAS_DONE);
        WaitFlag<HardEvent::FIX_MTE2>(EVT_GROUP_SCALE_DONE);
    }

    // One inner batch: spatial-chunk loop. FULL mode: block window GM-loaded once per
    // (group, batch), resident; chunks select sub-windows via the block-window POSM.
    __aicore__ inline void ProcessChunks(const BlockCtx& blockCtx, const TilingData& tiling, const L1Layout& l1,
                                         GroupCtx& gm, GM_ADDR x, uint32_t groupChanOff, uint32_t batchIdx,
                                         WpFmCache& fmCache, uint32_t& l0pp, bool& btUsed, bool& biasL1Waited) const
    {
        if constexpr (FmMode == FmapLoadMode::FULL) {
            FmapRange blockWindow = scheme_.ResolveBlockWindow(blockCtx, tiling);
            WaitFlag<HardEvent::MTE1_MTE2>(FmEv(0));
            IssueFmLoad(blockCtx, tiling, gm, l1, 0, 0, blockWindow.curHi, blockWindow.hiStart, blockWindow.curWi,
                        blockWindow.wiStart);
            fmCache.loaded[0] = FmTag(fmCache.gen, 0);
            fmCache.pend[0] = true;
        }
        typename Scheme::ChunkIter chunkIter;
        SpatialChunk chunk;
        for (bool more = scheme_.ChunkFirst(blockCtx, tiling, chunkIter, chunk); more;
             more = scheme_.ChunkNext(blockCtx, tiling, chunkIter, chunk)) {
            WpNextUnit nextUnit = {false, false, 0, 0, 0, 0, 0};
            if constexpr (FmMode != FmapLoadMode::FULL) {
                if (FmBufCnt(tiling) == DOUBLE_BUFFER_NUM) {
                    ChunkPeek peek;
                    if (scheme_.PeekNextWindow(blockCtx, tiling, chunkIter, peek)) {
                        nextUnit = {true, false, 0, peek.curHi, peek.hiLoadOff, peek.curWi, peek.wiLoadOff};
                    } else if (!OneBatch && batchIdx + 1 < blockCtx.singleCoreBatch) {
                        ChunkPeek first;
                        scheme_.FirstChunkWindow(blockCtx, tiling, first);
                        nextUnit = {true,
                                    true,
                                    blockCtx.batchStart + batchIdx + 1,
                                    first.curHi,
                                    first.hiLoadOff,
                                    first.curWi,
                                    first.wiLoadOff};
                    }
                }
            }

            ProcessTiles(blockCtx, tiling, l1, gm, x, groupChanOff, chunk, nextUnit, fmCache, l0pp, btUsed,
                         biasL1Waited);
            fmCache.gen++;
        }
        if constexpr (FmMode == FmapLoadMode::FULL) {
            SetFlag<HardEvent::MTE1_MTE2>(FmEv(0));
            fmCache.loaded[0] = WP_INVALID_IDX;
        }
    }

    // N-block x N-tile x K-block loop. FM chunks are cached across the N blocks/tiles of
    // one spatial chunk; co-resident chunks stay resident until the last tile, else the
    // buffers recycle per walk. Weight cache is tile-local.
    __aicore__ inline void ProcessTiles(const BlockCtx& blockCtx, const TilingData& tiling, const L1Layout& l1,
                                        GroupCtx& gm, GM_ADDR x, uint32_t groupChanOff, const SpatialChunk& chunk,
                                        const WpNextUnit& nextUnit, WpFmCache& fmCache, uint32_t& l0pp, bool& btUsed,
                                        bool& biasL1Waited) const
    {
        uint32_t fullK = blockCtx.kTotalFmap;
        uint32_t k1BlkCap = K1BlkCap(blockCtx, tiling);
        uint32_t kBL1 = k1BlkCap * GK0;
        uint32_t kStep = KStepFm(blockCtx, tiling);
        uint32_t cinL1 = CinL1(blockCtx, tiling);
        uint32_t fmBufCnt = FmBufCnt(tiling);
        uint32_t wtBufCnt = WtBufCnt(tiling);
        uint32_t kBlkCnt = KBlkCnt(blockCtx, kBL1);
        uint32_t fmChunkCnt = static_cast<uint32_t>(CeilDiv(fullK, kStep));
        uint32_t nBlk = static_cast<uint32_t>(tiling.nBL1);
        uint32_t nTileStep = NTileStep(tiling);
        uint32_t nBlkCnt = static_cast<uint32_t>(CeilDiv(blockCtx.actualCo, nBlk));

        for (uint32_t nBlkIdx = 0; nBlkIdx < nBlkCnt; nBlkIdx++) {
            uint32_t nBlkStart = nBlkIdx * nBlk;
            uint32_t curNBlk = blockCtx.actualCo - nBlkStart;
            if (curNBlk > nBlk) {
                curNBlk = nBlk;
            }
            for (uint32_t nTileOff = 0; nTileOff < curNBlk; nTileOff += nTileStep) {
                uint32_t curN = curNBlk - nTileOff;
                if (curN > nTileStep) {
                    curN = nTileStep;
                }
                uint32_t nOff = nBlkStart + nTileOff;
                uint32_t n1Tile = static_cast<uint32_t>(AlignB(curN, GN0)) / GN0;
                bool lastTile = (nBlkIdx + 1 == nBlkCnt) && (nTileOff + nTileStep >= curNBlk);

                LocalTensor<L0cT> cl0(TPosition::CO1, 0, BlockCtx::L0C_ELEMS);
                MmadParams mmadParams;
                initMmadParams_(tiling, mmadParams, chunk.curMAlign, n1Tile * GN0);

                uint32_t wLoaded[DOUBLE_BUFFER_NUM] = {WP_INVALID_IDX, WP_INVALID_IDX};
                bool wPend[DOUBLE_BUFFER_NUM] = {false, false};

                for (uint32_t kBlk = 0; kBlk < kBlkCnt; kBlk++) {
                    uint32_t kBlkOff = kBlk * kBL1;
                    uint32_t curKBlk = fullK - kBlkOff;
                    if (curKBlk > kBL1) {
                        curKBlk = kBL1;
                    }
                    uint32_t wbuf = kBlk % wtBufCnt;
                    if (wLoaded[wbuf] != kBlk) {
                        WaitFlag<HardEvent::MTE1_MTE2>(WEv(wbuf));
                        LoadWeightBlockTo(blockCtx, tiling, gm, l1, kBlkOff, curKBlk, nOff, n1Tile, wbuf, k1BlkCap);
                        SetFlag<HardEvent::MTE2_MTE1>(WEv(wbuf));
                        wLoaded[wbuf] = kBlk;
                        wPend[wbuf] = true;
                    }
                    if (wtBufCnt == DOUBLE_BUFFER_NUM && kBlk + 1 < kBlkCnt && wLoaded[(kBlk + 1) & 1] != kBlk + 1) {
                        uint32_t nextOff = (kBlk + 1) * kBL1;
                        uint32_t nextK = fullK - nextOff;
                        if (nextK > kBL1) {
                            nextK = kBL1;
                        }
                        WaitFlag<HardEvent::MTE1_MTE2>(WEv((kBlk + 1) & 1));
                        LoadWeightBlockTo(blockCtx, tiling, gm, l1, nextOff, nextK, nOff, n1Tile, (kBlk + 1) & 1,
                                          k1BlkCap);
                        SetFlag<HardEvent::MTE2_MTE1>(WEv((kBlk + 1) & 1));
                        wLoaded[(kBlk + 1) & 1] = kBlk + 1;
                        wPend[(kBlk + 1) & 1] = true;
                    }

                    uint32_t kCur = kBlkOff;
                    uint32_t kEnd = kBlkOff + curKBlk;
                    while (kCur < kEnd) {
                        // K run = weight K block n FM chunk.
                        uint32_t fmChunk = kCur / kStep;
                        uint32_t chunkEnd = (fmChunk + 1) * kStep;
                        if (chunkEnd > fullK) {
                            chunkEnd = fullK;
                        }
                        uint32_t runEnd = chunkEnd < kEnd ? chunkEnd : kEnd;
                        uint32_t fbuf = FmBufOf(tiling, fmChunk);
                        uint32_t tag = FmTag(fmCache.gen, fmChunk);
                        if constexpr (FmMode != FmapLoadMode::FULL) {
                            if (fmCache.loaded[fbuf] != tag) {
                                WaitFlag<HardEvent::MTE1_MTE2>(FmEv(fbuf));
                                IssueFmLoad(blockCtx, tiling, gm, l1, fbuf, fmChunk, chunk.curHi, chunk.hiLoadOff,
                                            chunk.curWi, chunk.wiLoadOff);
                                fmCache.loaded[fbuf] = tag;
                                fmCache.pend[fbuf] = true;
                            }
                            if (fmBufCnt == DOUBLE_BUFFER_NUM && runEnd < fullK) {
                                uint32_t nextChunk = runEnd / kStep;
                                uint32_t nextBuf = FmBufOf(tiling, nextChunk);
                                uint32_t nextTag = FmTag(fmCache.gen, nextChunk);
                                if (fmCache.loaded[nextBuf] != nextTag) {
                                    WaitFlag<HardEvent::MTE1_MTE2>(FmEv(nextBuf));
                                    IssueFmLoad(blockCtx, tiling, gm, l1, nextBuf, nextChunk, chunk.curHi,
                                                chunk.hiLoadOff, chunk.curWi, chunk.wiLoadOff);
                                    fmCache.loaded[nextBuf] = nextTag;
                                    fmCache.pend[nextBuf] = true;
                                }
                            }
                        }
                        // Consume each load-done flag exactly once.
                        if (fmCache.pend[fbuf]) {
                            WaitFlag<HardEvent::MTE2_MTE1>(FmEv(fbuf));
                            fmCache.pend[fbuf] = false;
                        }
                        if (wPend[wbuf]) {
                            WaitFlag<HardEvent::MTE2_MTE1>(WEv(wbuf));
                            wPend[wbuf] = false;
                        }
                        if (fmCache.ctxTag[fbuf] != tag) {
                            uint32_t cinOff;
                            uint32_t curCinOri;
                            CalcFmChunk(tiling, cinL1, fmChunk, cinOff, curCinOri);
                            if constexpr (FmMode == FmapLoadMode::FULL) {
                                // Block window resident: chunk selects via block-window POSM.
                                FmapRange blockWindow = scheme_.ResolveBlockWindow(blockCtx, tiling);
                                uint8_t padList[PAD_LIST_NUM] = {static_cast<uint8_t>(blockWindow.padLeft),
                                                                 static_cast<uint8_t>(blockWindow.padRight),
                                                                 static_cast<uint8_t>(blockWindow.padTop),
                                                                 static_cast<uint8_t>(PAD_BOTTOM_VALUE)};
                                Load3DSetFMatrixCal(blockWindow.curHi, blockWindow.orgWin, padList);
                                Load3DSetPaddingCal(tiling.offsetx);
                                Load3DCtx ctx;
                                UnionDataXt xt;
                                xt.n = static_cast<uint64_t>(tiling.unionDataXt);
                                xt.bf.channelSize = AlignB(curCinOri, GK0Fmap);
                                ctx.xtBase = xt.n;
                                UnionDataXm xm;
                                xm.n = 0;
                                xm.bf.mExtension_ = chunk.curMAlign;
                                xm.bf.mStartPt_ = chunk.blockWinPosM;
                                ctx.xmTmp = xm.n;
                                SetLoadDataRepeatWithStride(LoadDataRepeatParamWithStride(
                                    0, 1, 0, static_cast<uint16_t>(chunk.curMAlign / GM0)));
                                fmCache.ctx[fbuf] = ctx;
                            } else {
                                fmCache.ctx[fbuf] = setupLoad3dChunk_(tiling, chunk, AlignB(curCinOri, GK0Fmap));
                            }
                            fmCache.ctxTag[fbuf] = tag;
                        }

                        LocalTensor<FmapT> al1(TPosition::A1, fbuf * l1.al1BufBytes, l1.al1ElemCount / fmBufCnt);
                        LocalTensor<WeightT> bl1(
                            TPosition::B1, l1.bl1OffBytes + wbuf * Bl1ElemPerBuf(tiling, k1BlkCap) * sizeof(WeightT),
                            Bl1ElemPerBuf(tiling, k1BlkCap));
                        // Release at the chunk end only when consumed for good: co-resident
                        // chunks are re-read by later N tiles; only the last tile may free.
                        bool releaseFm = (FmMode != FmapLoadMode::FULL) && (runEnd == chunkEnd) &&
                                         (lastTile || fmChunkCnt > fmBufCnt);
                        bool releaseW = (runEnd == kEnd);
                        bool isFirstRun = (kBlk == 0) && (kCur == kBlkOff);
                        bool isWalkEnd = (runEnd == fullK);
                        RunKRange(tiling, l1, al1, bl1, cl0, mmadParams, fmCache.ctx[fbuf], kCur - fmChunk * kStep,
                                  kCur - kBlkOff, runEnd - kCur, nOff, curN, n1Tile, isFirstRun, isWalkEnd, l0pp,
                                  btUsed, biasL1Waited);
                        // Safe: the run's last kl0 already waited MTE1_M (L0 loads done).
                        if (releaseFm) {
                            SetFlag<HardEvent::MTE1_MTE2>(FmEv(fbuf));
                            fmCache.loaded[fbuf] = WP_INVALID_IDX;
                        }
                        if (releaseW) {
                            SetFlag<HardEvent::MTE1_MTE2>(WEv(wbuf));
                            wLoaded[wbuf] = WP_INVALID_IDX;
                        }
                        if (isWalkEnd && lastTile && nextUnit.valid) {
                            uint32_t nextTag = FmTag(fmCache.gen + 1, 0);
                            if (fmCache.loaded[0] != nextTag) {
                                if (nextUnit.isBatch) {
                                    setupFmapGmBatch_(tiling, x, nextUnit.nextBatchIdx, groupChanOff, gm.fmapGm);
                                }
                                WaitFlag<HardEvent::MTE1_MTE2>(FmEv(0));
                                uint32_t cinOff;
                                uint32_t curCinOri;
                                CalcFmChunk(tiling, cinL1, 0, cinOff, curCinOri);
                                loadFmapChunk_(scheme_, tiling, gm.fmapGm, l1.al1BufBytes, 0, nextUnit.curHi,
                                               nextUnit.hiLoadOff, nextUnit.curWi, nextUnit.wiLoadOff, cinOff,
                                               curCinOri);
                                SetFlag<HardEvent::MTE2_MTE1>(FmEv(0));
                                if (nextUnit.isBatch) {
                                    setupFmapGmBatch_(tiling, x, blockCtx.curBatchIdx, groupChanOff, gm.fmapGm);
                                }
                                fmCache.loaded[0] = nextTag;
                                fmCache.pend[0] = true;
                            }
                        }
                        kCur = runEnd;
                    }
                }

                copyOut_.template DoCopyOut<OutputT, 0>(blockCtx, tiling, gm.y, cl0, l1, chunk.outOff, chunk.fpMSize,
                                                        chunk.curMAlign, curN, chunk.fpDnNum, chunk.fpDstDnStride, nOff,
                                                        gm.groupCoutOff, gm.scale0Gm, gm.scale1Gm, gm.reluWeight0Gm,
                                                        gm.reluWeight1Gm);
                if (tiling.dualOutput) {
                    copyOut_.template DoCopyOut<Output1T, 1>(blockCtx, tiling, gm.y1, cl0, l1, chunk.outOff,
                                                             chunk.fpMSize, chunk.curMAlign, curN, chunk.fpDnNum,
                                                             chunk.fpDstDnStride, nOff, gm.groupCoutOff, gm.scale0Gm,
                                                             gm.scale1Gm, gm.reluWeight0Gm, gm.reluWeight1Gm);
                }
            }
        }
    }

    // One K run: kl0-sized L0 tiles accumulating into cl0. relKOff/bRelKOff: run start
    // relative to the FM chunk (A) / weight block (B); they differ inside a weight block.
    __aicore__ inline void RunKRange(const TilingData& tiling, const L1Layout& l1, LocalTensor<FmapT>& al1,
                                     LocalTensor<WeightT>& bl1, LocalTensor<L0cT>& cl0, MmadParams& mmadParams,
                                     const Load3DCtx& ctx, uint32_t relKOff, uint32_t bRelKOff, uint32_t kLen,
                                     uint32_t nOff, uint32_t curN, uint32_t n1Tile, bool isFirstRun, bool isLastRun,
                                     uint32_t& l0pp, bool& btUsed, bool& biasL1Waited) const
    {
        uint32_t kL0 = static_cast<uint32_t>(tiling.kL0);
        uint32_t kl0Iters = static_cast<uint32_t>(CeilDiv(kLen, kL0));
        bool aPp = (tiling.pBufferFlag & WP_L0A_PP_FLAG) != 0;
        bool bPp = (tiling.pBufferFlag & WP_L0B_PP_FLAG) != 0;
        bool anyPp = aPp || bPp;
        for (uint32_t kl0Iter = 0; kl0Iter < kl0Iters; kl0Iter++) {
            uint32_t kOff = kl0Iter * kL0;
            uint32_t curKInner = (kOff + kL0 > kLen) ? (kLen - kOff) : kL0;
            uint32_t curKAL0 = curKInner;
            if (isLastRun && kl0Iter + 1 == kl0Iters) {
                curKAL0 = kLen - kOff;
                curKInner = static_cast<uint32_t>(AlignB(curKAL0, GK0));
            }

            uint32_t pp = l0pp & 1;
            event_t ev = anyPp ? static_cast<event_t>(pp) : static_cast<event_t>(L0_SYBC_DB_CLOSE);
            LocalTensor<FmapT> al0(TPosition::A2, aPp ? pp * AL0_BUF_BYTES : 0,
                                   aPp ? BlockCtx::AL0_BUF_ELEMS : static_cast<uint32_t>(ASC_L0A_SIZE) / sizeof(FmapT));
            LocalTensor<WeightT> bl0(
                TPosition::B2, bPp ? pp * BL0_BUF_BYTES : 0,
                bPp ? BlockCtx::BL0_BUF_ELEMS : static_cast<uint32_t>(ASC_L0B_SIZE) / sizeof(WeightT));

            WaitFlag<HardEvent::M_MTE1>(ev);
            if (aPp) {
                loadAl0_(al1, al0, relKOff + kOff, curKAL0, ctx);
            }
            if (bPp) {
                loadBl0Block_(bl0, bl1, bRelKOff + kOff, curKInner, n1Tile);
            }
            if (!aPp) {
                if (anyPp) {
                    event_t e = static_cast<event_t>(pp ^ 1);
                    WaitFlag<HardEvent::M_MTE1>(e);
                    SetFlag<HardEvent::M_MTE1>(e);
                }
                loadAl0_(al1, al0, relKOff + kOff, curKAL0, ctx);
            }
            if (!bPp) {
                if (anyPp) {
                    event_t e = static_cast<event_t>(pp ^ 1);
                    WaitFlag<HardEvent::M_MTE1>(e);
                    SetFlag<HardEvent::M_MTE1>(e);
                }
                loadBl0Block_(bl0, bl1, bRelKOff + kOff, curKInner, n1Tile);
            }
            if (tiling.hasBias) {
                if (!biasL1Waited) {
                    WaitFlag<HardEvent::MTE2_MTE1>(EVT_BIAS_DONE);
                    biasL1Waited = true;
                }
                if (isFirstRun && kl0Iter == 0) {
                    if (btUsed) {
                        SetFlag<HardEvent::M_MTE1>(EVT_WP_BT_SYNC);
                        WaitFlag<HardEvent::M_MTE1>(EVT_WP_BT_SYNC);
                    }
                    loadBiasToBt_(tiling, l1, nOff, curN);
                    btUsed = true;
                }
            }
            SetFlag<HardEvent::MTE1_M>(ev);
            WaitFlag<HardEvent::MTE1_M>(ev);

            mmadParams.k = curKAL0;
            mmadParams.unitFlag = (isLastRun && kl0Iter + 1 == kl0Iters) ? UNIT_FLAG_ENABLE_WITH_FLIP :
                                                                           UNIT_FLAG_ENABLE_ONLY;
            Mmad(cl0, al0, bl0, mmadParams);
            mmadParams.cmatrixInitVal = false;
            mmadParams.cmatrixSource = false;
            SetFlag<HardEvent::M_MTE1>(ev);
            l0pp++;
        }
    }
};

} // namespace small_conv

#endif // WEIGHT_PARTLOAD_PROCESSOR_H
