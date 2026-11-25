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
 * \file situ_glu_group_quant_base.h
 * \brief Fused sparse-SiTU activation + MXFP8(e4m3fn, group=32) quantization.
 *        Kernel body ported verbatim from cann-recipes-infer SituGluGroupQuant
 *        (standalone port into ops-nn; not the merged SituMxQuant op).
 *
 * Backbone: situ_and_mul_sparse (per-row pipeline, sparse validRows, register
 * SiTU activation VF). The activation result stays in UB (never spilled to GM)
 * and is quantized in place by the dynamic_mx_quant tail-axis e4m3 OCP
 * three-pass VF. Bit-identical (within validRows) to the two small ops chained
 * npu_situ_and_mul_sparse -> npu_dynamic_mx_quant(dst=e4m3fn).
 *
 * VF form: __simd_vf__ functions invoked via asc_vf_call (the arch35 built-in
 * convention, e.g. dawsn/acos). asc_vf_call carries the MTE/VF boundary sync, so
 * the activation VF's actBuf store is ordered before the quant VF's load without
 * a manual drain.
 *
 * Port adaptations (kept surgical, kernel body untouched):
 *   1. FLOAT_OVERFLOW_MODE_CTRL guarded (defined locally when absent);
 *   2. Reg::ReduceDataBlock<Reg::ReduceType::MAX> fully qualified (the unqualified
 *      spelling depends on a newer CANN header alias; see port doc).
 *
 * KNOWN LIMITATIONS (inherited verbatim from recipes; both points below are now
 * REJECTED by the host tiling (EPR must be a multiple of 32 in [128, 3072]),
 * kept here as the kernel-side rationale):
 *   1. groupCount_ = epr_ / QUANT_BLOCK truncates: when EPR is not a multiple of
 *      32, the tail group is not quantized while CopyOut still writes the whole
 *      row (tail bytes are uninitialized UB data).
 *   2. When groupCount < 4 (EPR < 128) the masked PACK_B16 scale store silently
 *      drops the sub-8B payload and y_scale stays uninitialized (onboard-found).
 *   The tiling enforces EPR <= 3072 (whole-row CopyIn buffer limit, onboard-found
 *   overrun) plus EPR % 32 == 0 and EPR >= 128 (owner decision).
 */

#ifndef SITU_GLU_GROUP_QUANT_BASE_H
#define SITU_GLU_GROUP_QUANT_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "situ_glu_group_quant_quant.h"

#ifndef FLOAT_OVERFLOW_MODE_CTRL
#define FLOAT_OVERFLOW_MODE_CTRL 60
#endif

namespace SituGluGroupQuant {
using namespace AscendC;

constexpr uint32_t VF_TILE_FLOATS = 64U; // one full fp32 vector register (256B / 4)
constexpr int64_t QUANT_BLOCK = 32;      // MXFP8 group_size

__aicore__ inline int64_t CeilDivI(int64_t a, int64_t b) { return (a + b - 1) / b; }
__aicore__ inline int64_t CeilAlignI(int64_t a, int64_t b) { return CeilDivI(a, b) * b; }

// ---------------------------------------------------------------------------
// SiTU activation VF (__simd_vf__; ported verbatim from situ_and_mul_sparse).
// 64-fp32 tile per iter. Output is T (bf16/fp16) written to `out`.
// Invoked via asc_vf_call.
// kLinearDisabled (TPL dimension): linear_beta <= 0 disables the up-branch
// tanh soft-clamp — upF stays the loaded value (passthrough) and the whole
// tanh chain is compile-time eliminated; per-element instruction sequence of
// the enabled path is unchanged.
// ---------------------------------------------------------------------------
template <typename T, bool kLinearDisabled>
__simd_vf__ inline void SituActivationVF(__ubuf__ T* gate, __ubuf__ T* up, __ubuf__ T* out, uint32_t elements,
                                         float neg2OverBeta, float negBeta, float neg2OverLinearBeta,
                                         float negLinearBeta, float beta, float linearBeta)
{
    static constexpr Reg::CastTrait kF32ToTCastTrait = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait kTToF32CastTrait = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_NONE};

    const uint16_t tileCount = static_cast<uint16_t>((elements + VF_TILE_FLOATS - 1U) / VF_TILE_FLOATS);
    uint32_t remaining = elements;
    Reg::RegTensor<float> gateF;
    Reg::RegTensor<float> upF;
    Reg::RegTensor<float> tmpF;
    Reg::RegTensor<float> branchF;
    Reg::RegTensor<T> packedF;
    Reg::MaskReg preg;
    for (uint16_t i = 0; i < tileCount; ++i) {
        preg = Reg::UpdateMask<float>(remaining);
        const uint32_t offset = static_cast<uint32_t>(i) * VF_TILE_FLOATS;
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(packedF, gate + offset);
        Reg::Cast<float, T, kTToF32CastTrait>(gateF, packedF, preg);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(packedF, up + offset);
        Reg::Cast<float, T, kTToF32CastTrait>(upF, packedF, preg);

        // gate subchain
        Reg::Muls(tmpF, gateF, neg2OverBeta, preg);
        Reg::Exp(tmpF, tmpF, preg);
        Reg::Adds(tmpF, tmpF, 1.0f, preg);
        Reg::Duplicate(branchF, beta, preg);
        Reg::Div(branchF, branchF, tmpF, preg);
        Reg::Muls(branchF, branchF, 2.0f, preg);
        Reg::Adds(branchF, branchF, negBeta, preg);
        Reg::Muls(tmpF, gateF, -1.0f, preg);
        Reg::Exp(tmpF, tmpF, preg);
        Reg::Adds(tmpF, tmpF, 1.0f, preg);
        Reg::Div(branchF, branchF, tmpF, preg);
        // up subchain
        if constexpr (!kLinearDisabled) {
            Reg::Muls(tmpF, upF, neg2OverLinearBeta, preg);
            Reg::Exp(tmpF, tmpF, preg);
            Reg::Adds(tmpF, tmpF, 1.0f, preg);
            Reg::Duplicate(upF, linearBeta, preg);
            Reg::Div(upF, upF, tmpF, preg);
            Reg::Muls(upF, upF, 2.0f, preg);
            Reg::Adds(upF, upF, negLinearBeta, preg);
        }
        // kLinearDisabled: upF keeps the loaded fp32 value (passthrough)
        Reg::Mul(branchF, branchF, upF, preg);

        // exit: fp32 -> T then packed store
        Reg::Cast<T, float, kF32ToTCastTrait>(packedF, branchF, preg);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(out + offset, packedF, preg);
    }
}

// Weighted batched activation VF (K-pro): same per-element chain as
// SituActivationVF plus a per-row Muls(w) in fp32 before the exit Cast. The
// row loop lives INSIDE the VF so one call covers the whole batch: wRow points
// at the batch's K contiguous fp32 scalars already staged in UB by CopyIn
// (weightQueue), and each row's scalar is read from UB (no GM scalar loads in
// the hot loop). Only used when EPR % VF_TILE_FLOATS == 0 (the host batching
// gate guarantees it for K>1; every tile is full-width so no tail masking is
// needed — per-tile instruction sequence is identical to the legacy VF, hence
// bit-identical per-element numerics).
// kLinearDisabled: same TPL caliber as SituActivationVF — up passthrough.
template <typename T, bool kLinearDisabled>
__simd_vf__ inline void SituActivationWeightRowVF(__ubuf__ T* gate, __ubuf__ T* up, __ubuf__ T* out, uint32_t elements,
                                                  uint32_t rowElements, __ubuf__ const float* wRow, float neg2OverBeta,
                                                  float negBeta, float neg2OverLinearBeta, float negLinearBeta,
                                                  float beta, float linearBeta)
{
    static constexpr Reg::CastTrait kF32ToTCastTrait = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait kTToF32CastTrait = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_NONE};

    // Unified weighted VF: per-row tail-masked tiles (64 + 64 + ... + masked
    // remainder) handle ANY EPR multiple of 32 — no epr % 64 gate, no legacy
    // fallback. Per-element instruction sequence is identical to the flat VF
    // (same ops, same remaining-mask tail semantics, same loads/stores), so
    // numerics are bit-identical; only the per-row wVal lookup is added.
    const uint32_t rowsEpr = rowElements > 0 ? rowElements : 1;
    const uint32_t tilesPerRow = (rowsEpr + VF_TILE_FLOATS - 1U) / VF_TILE_FLOATS;
    const int32_t rows = static_cast<int32_t>(elements / rowsEpr);
    Reg::RegTensor<float> gateF;
    Reg::RegTensor<float> upF;
    Reg::RegTensor<float> tmpF;
    Reg::RegTensor<float> branchF;
    Reg::RegTensor<T> packedF;
    Reg::MaskReg preg;
    for (int32_t r = 0; r < rows; ++r) {
        const float wVal = wRow[r];
        const uint32_t rowOffset = static_cast<uint32_t>(r) * rowsEpr;
        uint32_t remaining = rowsEpr;
        for (uint32_t t = 0; t < tilesPerRow; ++t) {
            // fresh per-tile mask count: UpdateMask takes a non-const ref and
            // passing the loop-carried bookkeeping variable directly proved
            // unreliable under this toolchain (onboard: tile >= 1 stores were
            // skipped); mirror the proven flat-VF pattern instead — a fresh
            // local each tile keeps the tail semantics with no carried state.
            uint32_t maskLanes = (remaining > VF_TILE_FLOATS) ? VF_TILE_FLOATS : remaining;
            preg = Reg::UpdateMask<float>(maskLanes);
            const uint32_t offset = rowOffset + t * VF_TILE_FLOATS;
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(packedF, gate + offset);
            Reg::Cast<float, T, kTToF32CastTrait>(gateF, packedF, preg);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(packedF, up + offset);
            Reg::Cast<float, T, kTToF32CastTrait>(upF, packedF, preg);

            // gate subchain (identical to SituActivationVF)
            Reg::Muls(tmpF, gateF, neg2OverBeta, preg);
            Reg::Exp(tmpF, tmpF, preg);
            Reg::Adds(tmpF, tmpF, 1.0f, preg);
            Reg::Duplicate(branchF, beta, preg);
            Reg::Div(branchF, branchF, tmpF, preg);
            Reg::Muls(branchF, branchF, 2.0f, preg);
            Reg::Adds(branchF, branchF, negBeta, preg);
            Reg::Muls(tmpF, gateF, -1.0f, preg);
            Reg::Exp(tmpF, tmpF, preg);
            Reg::Adds(tmpF, tmpF, 1.0f, preg);
            Reg::Div(branchF, branchF, tmpF, preg);

            // up subchain (identical to SituActivationVF)
            if constexpr (!kLinearDisabled) {
                Reg::Muls(tmpF, upF, neg2OverLinearBeta, preg);
                Reg::Exp(tmpF, tmpF, preg);
                Reg::Adds(tmpF, tmpF, 1.0f, preg);
                Reg::Duplicate(upF, linearBeta, preg);
                Reg::Div(upF, upF, tmpF, preg);
                Reg::Muls(upF, upF, 2.0f, preg);
                Reg::Adds(upF, upF, negLinearBeta, preg);
            }
            // kLinearDisabled: upF keeps the loaded fp32 value (passthrough)
            Reg::Mul(branchF, branchF, upF, preg);

            Reg::Muls(branchF, branchF, wVal, preg);

            // exit: fp32 -> T then packed store
            Reg::Cast<T, float, kF32ToTCastTrait>(packedF, branchF, preg);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(out + offset, packedF, preg);

            remaining = remaining > VF_TILE_FLOATS ? remaining - VF_TILE_FLOATS : 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Fused kernel class.
// ---------------------------------------------------------------------------
template <typename T, bool hasGroupIndex, bool hasWeight, bool kActivateLeft, bool kGroupListCount,
          bool kLinearDisabled>
class SituGluGroupQuantKernel {
    using U = fp8_e4m3fn_t;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR y_scale, GM_ADDR weight,
                                const SituGluGroupQuantTilingData* tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        epr_ = tiling_->outputElementsPerRow;
        chunk_ = tiling_->chunkElements;
        groupCount_ = epr_ / QUANT_BLOCK;
        scaleColNum_ = tiling_->scaleColNum;
        rowsPerBatch_ = tiling_->rowsPerBatch > 0 ? tiling_->rowsPerBatch : 1;

        xGm_.SetGlobalBuffer((__gm__ T*)x);
        // Optional inputs follow the swiglu_mx_quant group_index caliber:
        // presence is a TPL dimension, so the binding is compile-time isolated.
        // An absent-optional instance never binds the placeholder GM_ADDR that
        // opc's gen_placeholder mode passes for uninstantiated optional inputs.
        if constexpr (hasGroupIndex) {
            groupIndexGm_.SetGlobalBuffer((__gm__ int64_t*)groupIndex);
        }
        if constexpr (hasWeight) {
            weightGm_.SetGlobalBuffer((__gm__ float*)weight);
        }
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
        scaleGm_.SetGlobalBuffer((__gm__ uint8_t*)y_scale);

        // K-row batching: every batched buffer scales by rowsPerBatch_ (K=1
        // degenerates to the exact legacy per-row sizes). The gateup buffer is
        // deinterleaved layout [first halves K*chunk | second halves K*chunk].
        // K=1 gc%8!=0 overrun guard (see actBuf note below): gateupQueue's
        // activation VF reads 64-element tiles; with chunk%256!=0 the last tile
        // may read past the second half's slot end (e.g. EPR=1056: tile at 1024
        // reads 64 elems, 32 past the 1056-elem half). UpdateMask constrains
        // register compute, but whether it bounds the LoadAlign memory access
        // width is undocumented — pad the halves to the 256-element quantum too,
        // so the tail read lands in-bounds regardless of mask semantics.
        const int64_t K = rowsPerBatch_;
        const int64_t scaleBufBytes = CeilAlignI(groupCount_, grouped_situ::VF_LEN_16) * sizeof(uint16_t);
        const bool needPad = (K == 1) && (chunk_ % static_cast<int64_t>(grouped_situ::VF_LEN_16_DOUBLE) != 0);
        const int64_t paddedChunk = needPad ? CeilAlignI(chunk_, static_cast<int64_t>(grouped_situ::VF_LEN_16_DOUBLE)) :
                                              chunk_;
        pipe->InitBuffer(gateupQueue_, 2, 2 * K * paddedChunk * sizeof(T));
        pipe->InitBuffer(yQueue_, 2, K * paddedChunk * sizeof(uint8_t));
        pipe->InitBuffer(mxScaleQueue_, 2, K * CeilAlignI(scaleColNum_, grouped_situ::VF_LEN_16) * sizeof(uint8_t));
        // K=1 gc%8!=0 overrun guard: the maxexp/data VFs process a fixed
        // 256-element register per iteration. When K*chunk is not a multiple
        // of VF_LEN_16_DOUBLE, their last partial iteration reads/writes past
        // the buffer end into adjacent UB (onboard-verified: EPR=1088 gc=34,
        // Data VF y-write overrun from y slot 1 lands on mxScale slots and
        // corrupts odd-row scale to 0x70; recipes has the same overrun with a
        // milder 12.5% all-row corruption). For K==1 we pad both buffers to
        // the 256-element quantum; for K>1 the overrun lands inside the OTHER
        // DB2 slot (which is not the active one) and is harmless, so the
        // tiling perRowBase stays unchanged.
        // Pad both actBuf and yQueue to the 256-element VF quantum when K==1
        // and gc%8!=0: the maxexp VF's last partial iteration READS past
        // actBuf into maxExpBuf/recipBuf (recipes-inherited 12.5% corruption),
        // and the Data VF's y-WRITE past yQueue slot 1 lands on mxScale slots
        // (K-Pro specific: odd-row scale = 0x70). For K>1 the overrun lands
        // in the other DB2 slot and is harmless, so no padding / no perRowBase
        // change is needed. Onboard-verified: EPR=1088 gc=34.
        pipe->InitBuffer(actBuf_, K * paddedChunk * sizeof(T));
        pipe->InitBuffer(maxExpBuf_, K * scaleBufBytes);
        pipe->InitBuffer(recipScaleBuf_, K * scaleBufBytes);
        if constexpr (hasWeight) {
            // per-batch weight staging: K contiguous fp32 scalars per slot
            // (DB2, rides the same MTE2 pipeline as gateupQueue_)
            pipe->InitBuffer(weightQueue_, 2, static_cast<uint32_t>((K * sizeof(float) + 31) / 32 * 32));
        }

        // Assert saturation once for the e4m3 SAT cast in the quant pass (clamp to
        // 448 on overflow), matching dynamic_mx_quant / swiglu_group_quant which
        // set this at Init. The kernel entry saves/restores the original mode.
#if (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 9201)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    }

    __aicore__ inline void Process()
    {
        const int64_t totalRows = tiling_->totalElements / epr_;
        // Optional group_index (owner decision): absent -> whole-tensor mode,
        // ALL rows are processed (no tokens buffer exists in this instance);
        // present -> recipes grouped semantics (sum + clamp, padding skipped).
        int64_t validRows = totalRows;
        if constexpr (hasGroupIndex) {
            if constexpr (kGroupListCount) {
                validRows = LoadValidRows(totalRows);
            } else {
                // cumsum mode (group_list_type=0) is reserved: the host tiling
                // rejects it, so this instance is unreachable; process nothing.
                validRows = 0;
            }
        }
        const int64_t block = GetBlockIdx();
        const int64_t blockCount = GetBlockNum();
        // Blocked row partition: each core walks a contiguous row range in
        // K-row batches (batch copies need GM-contiguous rows). K=1 keeps the
        // legacy cycle (strided rows today == blocked ranges with K=1 after
        // this change only in assignment, not in numerics: rows are
        // independent).
        const int64_t rowsPerCore = CeilDivI(validRows, blockCount);
        const int64_t rowBegin = block * rowsPerCore;
        if (rowBegin >= validRows) {
            return;
        }
        const int64_t rowEnd = (rowBegin + rowsPerCore) < validRows ? (rowBegin + rowsPerCore) : validRows;
        for (int64_t row = rowBegin; row < rowEnd; row += rowsPerBatch_) {
            const int64_t kEff = (row + rowsPerBatch_ <= rowEnd) ? rowsPerBatch_ : (rowEnd - row);
            // per-token weight scalars are batch-staged into UB by CopyIn
            // (weightQueue_): one contiguous vector copy per batch replaces the
            // legacy per-row GM scalar load (latency-bound, un-pipelined).
            CopyIn(row, kEff);
            Compute(kEff);
            CopyOut(row, kEff);
        }
    }

private:
    __aicore__ inline int64_t LoadValidRows(int64_t totalRows)
    {
        int64_t validRows = 0;
        for (int64_t i = 0; i < tiling_->groupIndexCount; ++i) {
            validRows += groupIndexGm_.GetValue(i);
        }
        return validRows > totalRows ? totalRows : (validRows > 0 ? validRows : 0);
    }

    __aicore__ inline void CopyIn(int64_t row, int64_t kEff)
    {
        LocalTensor<T> both = gateupQueue_.AllocTensor<T>();
        const int64_t inputRowOffset = row * epr_ * 2;
        // K-row deinterleaved copy: first halves then second halves packed
        // contiguously (layout [first K*chunk | second K*chunk]); two strided
        // descriptors total for the whole batch, regardless of kEff (kEff==1
        // degenerates to one block per descriptor — the single-DataCopy legacy
        // path was unified away after onboard verification proved the
        // deinterleaved path correct for all kEff values). EPR%32==0 + host
        // K gate guarantee per-block 32B alignment (EPR >= 128, T = 2B).
        LocalTensor<T> first = both;
        LocalTensor<T> second = both[rowsPerBatch_ * chunk_];
        DataCopyExtParams halfParams;
        halfParams.blockCount = static_cast<uint32_t>(kEff);
        halfParams.blockLen = static_cast<uint32_t>(epr_ * sizeof(T));
        halfParams.srcStride = static_cast<uint32_t>(epr_ * sizeof(T)); // skip the other half
        halfParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad(first, xGm_[inputRowOffset], halfParams, padParams);
        DataCopyPad(second, xGm_[inputRowOffset + epr_], halfParams, padParams);
        if constexpr (hasWeight) {
            // K contiguous fp32 scalars (batch rows are GM-contiguous by the
            // blocked partition); DataCopyPad keeps any kEff (incl. 1) safe.
            LocalTensor<float> wUb = weightQueue_.AllocTensor<float>();
            DataCopyExtParams wParams = {1U, static_cast<uint32_t>(kEff * sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> wPad = {false, 0, 0, 0};
            DataCopyPad(wUb, weightGm_[row], wParams, wPad);
            weightQueue_.EnQue(wUb);
        }
        gateupQueue_.EnQue(both);
    }

    __aicore__ inline void Compute(int64_t kEff)
    {
        LocalTensor<T> both = gateupQueue_.DeQue<T>();
        LocalTensor<uint8_t> y = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint16_t> scale = mxScaleQueue_.AllocTensor<uint16_t>();
        LocalTensor<T> act = actBuf_.template Get<T>();
        LocalTensor<uint16_t> maxExp = maxExpBuf_.template Get<uint16_t>();
        LocalTensor<uint16_t> recip = recipScaleBuf_.template Get<uint16_t>();

        // activate_left (owner decision, mirrors situ_mx_quant normal path):
        //   true  -> gate = first half, up = second half (recipes original)
        //   false -> gate = second half, up = first half (situ normal, default)
        // TPL dimension (swiglu activateDimLast caliber): compile-time selected.
        __ubuf__ T* base = (__ubuf__ T*)both.GetPhyAddr();
        // Layout must match CopyIn: kEff==1 takes the legacy whole-row copy
        // (gate/up halves adjacent in the first chunk); kEff>1 takes the
        // Deinterleaved batch layout from CopyIn: [first K*chunk | second K*chunk].
        // kEff==1 with K==1 degenerates to the legacy adjacent layout (second
        // offset = 1*chunk_); kEff==1 with K>1 (partial batch tail) has a gap
        // between the halves — harmless, the VF reads via pointers.
        __ubuf__ T* first = base;
        __ubuf__ T* second = base + rowsPerBatch_ * chunk_;
        __ubuf__ T* gate = kActivateLeft ? first : second;
        __ubuf__ T* up = kActivateLeft ? second : first;
        __ubuf__ T* actAddr = (__ubuf__ T*)act.GetPhyAddr();
        __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)maxExp.GetPhyAddr();
        __ubuf__ uint16_t* recipAddr = (__ubuf__ uint16_t*)recip.GetPhyAddr();
        __ubuf__ uint16_t* scaleAddr = (__ubuf__ uint16_t*)scale.GetPhyAddr();
        __ubuf__ int8_t* yAddr = (__ubuf__ int8_t*)y.GetPhyAddr();

        // --- SiTU activation -> act (bf16/fp16), same round-to-input as situ ---
        const uint32_t count = static_cast<uint32_t>(kEff * epr_);
        const float neg2OverBeta = -2.0f / tiling_->beta;
        const float negBeta = -tiling_->beta;
        // Disabled instance: the up-chain coefficients are compile-time
        // eliminated (never reach the VF); zeroed here so -2/linearBeta never
        // produces inf when linear_beta <= 0.
        const float neg2OverLinearBeta = kLinearDisabled ? 0.0f : -2.0f / tiling_->linearBeta;
        const float negLinearBeta = kLinearDisabled ? 0.0f : -tiling_->linearBeta;
        const float linearBetaArg = kLinearDisabled ? 0.0f : tiling_->linearBeta;
        if constexpr (hasWeight) {
            // ONE unified weighted VF for every EPR: per-row tail-masked tiles
            // (64 + 64 + ... + masked remainder) — no epr % 64 gate, no legacy
            // fallback; per-row scalars read from the UB staging queue.
            LocalTensor<float> wUb = weightQueue_.DeQue<float>();
            __ubuf__ const float* wUbAddr = (__ubuf__ const float*)wUb.GetPhyAddr();
            asc_vf_call<SituActivationWeightRowVF<T, kLinearDisabled>>(
                gate, up, actAddr, count, static_cast<uint32_t>(epr_), wUbAddr, neg2OverBeta, negBeta,
                neg2OverLinearBeta, negLinearBeta, tiling_->beta, linearBetaArg);
            weightQueue_.FreeTensor(wUb);
        } else {
            // no-weight instance: unchanged flat VF (exact recipes 11-param ABI,
            // bit-identical codegen to the legacy per-row path)
            asc_vf_call<SituActivationVF<T, kLinearDisabled>>(gate, up, actAddr, count, neg2OverBeta, negBeta,
                                                              neg2OverLinearBeta, negLinearBeta, tiling_->beta,
                                                              linearBetaArg);
        }

        // --- MXFP8 e4m3 OCP quantization (asc_vf_call, ported from dynamic_mx_quant) ---
        // flat group arrays over the whole batch: K*gc contiguous groups, group
        // boundaries never straddle rows (EPR%32==0), per-group 32-element semantics
        // unchanged -> bit-identical quantization per element.
        const int64_t groups = groupCount_ * kEff;
        const uint16_t loopNum2VF = static_cast<uint16_t>(CeilDivI(groups, grouped_situ::ELEMENT_AFTER_REDUCE));
        const uint16_t loopNum1VF = static_cast<uint16_t>(CeilDivI(groups, grouped_situ::VF_LEN_16));
        asc_vf_call<grouped_situ::ComputeMaxExpOcpVF<T>>(actAddr, maxExpAddr, loopNum2VF);
        asc_vf_call<grouped_situ::ComputeScaleOcpVF>(maxExpAddr, scaleAddr, recipAddr, loopNum1VF,
                                                     static_cast<uint32_t>(groups));
        asc_vf_call<grouped_situ::ComputeDataVF<T, U>>(actAddr, recipAddr, yAddr, loopNum2VF);

        yQueue_.EnQue(y);
        mxScaleQueue_.EnQue(scale);
        gateupQueue_.FreeTensor(both);
    }

    __aicore__ inline void CopyOut(int64_t row, int64_t kEff)
    {
        LocalTensor<uint16_t> scale = mxScaleQueue_.DeQue<uint16_t>();
        LocalTensor<uint8_t> scaleU8 = scale.template ReinterpretCast<uint8_t>();
        if (scaleColNum_ == groupCount_) {
            // even gc: K consecutive rows' scales are contiguous in GM (host
            // K gate), one flat pad copy per batch
            DataCopyExtParams scaleParams = {1U, static_cast<uint32_t>(kEff * scaleColNum_ * sizeof(uint8_t)), 0U, 0U,
                                             0U};
            DataCopyPad(scaleGm_[row * scaleColNum_], scaleU8, scaleParams);
        } else {
            // odd gc (scaleColNum = gc+1): GM rows carry the interleave-2 tail
            // padding while the UB rows stay compact (VF emits the flat K*gc
            // byte stream) — copy per-row Compact blocks instead (swiglu_mx_quant
            // caliber): 1-byte granularity on both sides, srcStride = 0 keeps
            // the compact UB source, dstStride = per-row GM padding gap.
            // Executed once per batch, outside the vector hot loop.
            DataCopyExtParams scaleParams = {0, 0, 0, 0, 0};
            scaleParams.blockCount = static_cast<uint32_t>(kEff);
            scaleParams.blockLen = static_cast<uint32_t>(groupCount_);
            scaleParams.srcStride = 0U;
            scaleParams.dstStride = static_cast<uint32_t>(scaleColNum_ - groupCount_);
            DataCopyPad<uint8_t, PaddingMode::Compact>(scaleGm_[row * scaleColNum_], scaleU8, scaleParams);
        }
        mxScaleQueue_.FreeTensor(scale);

        LocalTensor<uint8_t> y = yQueue_.DeQue<uint8_t>();
        DataCopy(yGm_[row * epr_], y, kEff * epr_);
        yQueue_.FreeTensor(y);
    }

    const SituGluGroupQuantTilingData* tiling_;
    int64_t epr_ = 0;
    int64_t chunk_ = 0;
    int64_t groupCount_ = 0;
    int64_t scaleColNum_ = 0;
    int64_t rowsPerBatch_ = 1;
    GlobalTensor<T> xGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<float> weightGm_;

    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> scaleGm_;
    TQue<QuePosition::VECIN, 2> gateupQueue_;
    TQue<QuePosition::VECOUT, 2> yQueue_;
    TQue<QuePosition::VECOUT, 2> mxScaleQueue_;
    TBuf<QuePosition::VECCALC> actBuf_;
    TBuf<QuePosition::VECCALC> maxExpBuf_;
    TBuf<QuePosition::VECCALC> recipScaleBuf_;
    TQue<QuePosition::VECIN, 2> weightQueue_;
};

} // namespace SituGluGroupQuant
#endif // SITU_GLU_GROUP_QUANT_BASE_H
