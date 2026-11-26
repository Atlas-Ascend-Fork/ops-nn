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
 * \file turbo_quant_regbase.h
 * \brief ascend950 (arch35/regbase) kernel for TurboQuant.
 *
 * Byte-for-byte compatible with the ascend910b implementation in ../turbo_quant.h: same outputs
 * (y rows of packed nibbles, low nibble first in dim order, no row padding; the fp16 L2 norm in
 * the separate scale tensor), same NaN/Inf edge behaviour (it emerges from mirroring the
 * arithmetic: u = x * (1 / norm)).
 *
 * Differences from the 910b kernel, all register-level:
 *  - compute runs on vector registers via LoadAlign/StoreAlign instead of whole-tensor UB ops, so the
 *    u / sel / one / mask / tmp round-trip buffers disappear and the per-token UB cost drops to
 *    inQ + nib + the packed y row;
 *  - the nearest-centroid chain uses Reg::Compares<CMPMODE::GE> + Select accumulate;
 *  - all per-token FP math (the eps add, sqrt, the 1 / norm division) runs on vector instructions --
 *    VL1-masked on lane 0 or full-lane on broadcast operands; the regbase scalar unit does no FP
 *    math, so the 910b scalar sqrt / 1 / norm / fp16 conversion moves into the vector domain;
 *  - the int4 pack has no Cast<int4b_t> analogue on regbase: nibbles stay exact small integers in
 *    fp32, the two nibbles of a byte are collected with Reg::Gather (indices built in-register with
 *    Arange -> u32 reinterpret), combined arithmetically as byte = nib[2k] + 16 * nib[2k + 1], and
 *    narrowed through the half -> uint8 (RINT) cast chain before a DIST_PACK4_B32 store. uint8 holds
 *    the full [0, 255] byte exactly, so the written bits match the 910b hardware pack (which packs
 *    indices 8..15 as the same 4-bit patterns) without relying on NO_SAT out-of-range behavior;
 *  - the fp16 scale is narrowed with a vector Cast plus a DIST_FIRST_ELEMENT_B16 store into a
 *    64B-strided staging area behind the y rows, then copied out per token.
 */

#ifndef TURBO_QUANT_REGBASE_H
#define TURBO_QUANT_REGBASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace TurboQuantRegbase {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::LoadDist;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreAlign;
using AscendC::Reg::StoreDist;

constexpr uint32_t N_CENT = 16;
constexpr uint32_t ALIGN_BYTES = 64;
// Keep in sync with TQ_REGBASE_MAX_TOKENS_PER_BATCH in the tiling header.
constexpr uint32_t MAX_TOKENS_PER_BATCH = 24;
// Each token's L2 norm reduces into its own 64B-aligned slot: +0 holds the final norm.
constexpr uint32_t NORM_SLOT_FLOATS = 16;
// The fp16 scale stages into its own 64B-aligned slot per token (one DIST_FIRST_ELEMENT_B16 store
// per slot; 2-byte-strided packing would rely on untested narrow-store alignment).
constexpr uint32_t SCALE_SLOT_HALFS = ALIGN_BYTES / sizeof(half);
constexpr float NORM_EPS = 1e-16f;

constexpr uint32_t vlFp32 = GetVecLen() / sizeof(float);

constexpr AscendC::Reg::CastTrait castTraitF32ToF16Rint = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

// uint8 dst keeps every byte value in [0, 255] exactly in range: no reliance on what a NO_SAT
// signed narrowing does out of range (dav_3510 vconv narrowing matrix lists Tuple<uint8_t, half>).
constexpr AscendC::Reg::CastTrait castTraitF16ToU8Rint = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__aicore__ inline uint32_t AlignUpTo(uint32_t value, uint32_t align) { return (value + align - 1) / align * align; }

class KernelTurboQuantRegbase {
public:
    __aicore__ inline KernelTurboQuantRegbase(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR latent, GM_ADDR centroids, GM_ADDR yOut, GM_ADDR scaleOut, uint32_t numTokens,
                                uint32_t tokensPerCore, uint32_t headDim, uint32_t packedBytes, uint32_t tokensPerBatch)
    {
        numTokens_ = numTokens;
        headDim_ = headDim;
        packedBytes_ = packedBytes;
        batch_ = tokensPerBatch < 1 ? 1 : tokensPerBatch;
        if (batch_ > MAX_TOKENS_PER_BATCH) {
            batch_ = MAX_TOKENS_PER_BATCH;
        }

        uint32_t coreIdx = GetBlockIdx();
        tokStart_ = coreIdx * tokensPerCore;
        tokEnd_ = tokStart_ + tokensPerCore;
        if (tokEnd_ > numTokens_) {
            tokEnd_ = numTokens_;
        }

        latentGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(latent));
        centGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(centroids));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(yOut));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scaleOut));

        // Register-resident compute only needs the input, the nibble staging area and the output
        // (packed y rows + the 64B-strided fp16 scale staging) in UB.
        uint32_t batchElems = batch_ * headDim_;
        uint32_t fp32Bytes = AlignUpTo(batchElems * sizeof(float), ALIGN_BYTES);
        yBytes_ = AlignUpTo(batch_ * packedBytes_, ALIGN_BYTES);
        uint32_t scaleBytes = AlignUpTo(batch_ * SCALE_SLOT_HALFS * sizeof(half), ALIGN_BYTES);
        uint32_t normBytes = AlignUpTo(batch_ * NORM_SLOT_FLOATS * sizeof(float), ALIGN_BYTES);
        uint32_t centBytes = AlignUpTo(N_CENT * sizeof(float), ALIGN_BYTES);

        pipe_->InitBuffer(inQ_, 1, fp32Bytes);
        pipe_->InitBuffer(outQ_, 1, yBytes_ + scaleBytes);
        pipe_->InitBuffer(nibBuf_, fp32Bytes);
        pipe_->InitBuffer(normBuf_, normBytes);
        pipe_->InitBuffer(centBuf_, centBytes);

        LocalTensor<float> cent = centBuf_.Get<float>();
        DataCopy(cent, centGm_, N_CENT);
        // Precise MTE2 -> scalar sync for the one dependency that exists here: the scalar GetValue
        // reads below waiting on this 64B copy. No full-core barrier:
        // nothing else is in flight after InitBuffer, and the bnd_ midpoint math is scalar-domain
        // only -- the VF scopes in Compute consume it after Init returns, in issue order.
        event_t centReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(centReady);
        WaitFlag<HardEvent::MTE2_S>(centReady);
        for (uint32_t i = 0; i < N_CENT; ++i) {
            centVal_[i] = cent.GetValue(i);
        }
        for (uint32_t i = 0; i + 1 < N_CENT; ++i) {
            // 15 midpoint boundaries; counting how many a value exceeds yields the nearest-centroid index
            bnd_[i] = (centVal_[i] + centVal_[i + 1]) * 0.5f;
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t t = tokStart_; t < tokEnd_; t += batch_) {
            uint32_t count = tokEnd_ - t;
            if (count > batch_) {
                count = batch_;
            }
            CopyIn(t, count);
            Compute(count);
            CopyOut(t, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t t, uint32_t count)
    {
        LocalTensor<float> in = inQ_.AllocTensor<float>();
        DataCopy(in, latentGm_[static_cast<uint64_t>(t) * headDim_], count * headDim_);
        inQ_.EnQue(in);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<float> in = inQ_.DeQue<float>();
        LocalTensor<uint8_t> out = outQ_.AllocTensor<uint8_t>();
        LocalTensor<float> nib = nibBuf_.Get<float>();
        LocalTensor<float> norm = normBuf_.Get<float>();
        __ubuf__ float* inAddr = (__ubuf__ float*)in[0].GetPhyAddr();
        __ubuf__ float* nibAddr = (__ubuf__ float*)nib.GetPhyAddr();
        __ubuf__ float* normAddr = (__ubuf__ float*)norm.GetPhyAddr();
        __ubuf__ uint8_t* outAddr = (__ubuf__ uint8_t*)out[0].GetPhyAddr();
        __ubuf__ uint8_t* yAddr = outAddr;
        __ubuf__ half* scaleAddr = (__ubuf__ half*)(outAddr + yBytes_);

        const uint32_t chunksPerToken = headDim_ / vlFp32;
        const uint32_t byteChunks = packedBytes_ / vlFp32;
        // The ccec vector-loop pattern inside __VEC_SCOPE__ requires uint16_t induction vars and bounds.
        const uint16_t tokenLoop = static_cast<uint16_t>(count);
        const uint16_t chunkLoop = static_cast<uint16_t>(chunksPerToken);
        const uint16_t byteLoop = static_cast<uint16_t>(byteChunks);

        // Per-token L2 norm, fully in vector domain (the regbase scalar unit does no FP math):
        // the eps add and the sqrt run on lane 0 only.
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> sq;
            RegTensor<float> acc;
            RegTensor<float> normReg;
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();

            for (uint16_t i = 0; i < tokenLoop; i++) {
                Duplicate(acc, 0.0f, pregMain);
                for (uint16_t c = 0; c < chunkLoop; c++) {
                    LoadAlign(x, inAddr + i * headDim_ + c * vlFp32);
                    Mul(sq, x, x, pregMain);
                    Add(acc, acc, sq, pregMain);
                }
                ReduceSum(normReg, acc, pregMain);
                Adds(normReg, normReg, NORM_EPS, pregMerge);
                Sqrt(normReg, normReg, pregMerge);
                StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(normAddr + i * NORM_SLOT_FLOATS, normReg,
                                                                     pregMerge);
            }
            // The store->load fence must live INSIDE the __VEC_SCOPE__: a function-level (out-of-scope)
            // LocalMemBar miscompiles on this ccec build and faults the vector core on device.
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }

        // Nearest-centroid chain: nib[d] = #{boundaries <= u[d]} with u = x * broadcast(1 / norm).
        // 1 / norm is a full-lane vector Div on broadcast operands (lane-wise IEEE division, identical
        // to the 910b scalar reciprocal); the 15 boundary comparisons are unrolled with literal
        // indexes so the bnd_ loads stay loop-invariant and get hoisted out of the vector region.
        // MicroAPI Compares yields a mask, so counting mirrors the 910b three-step idiom
        // Select(sel = mask ? 1 : 0) + Add(nib, nib, sel); muxing straight into nib collapses the
        // count to a 0/1 indicator.
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> u;
            RegTensor<float> nibReg;
            RegTensor<float> one;
            RegTensor<float> zeroF;
            RegTensor<float> sel;
            RegTensor<float> normV;
            RegTensor<float> invNorm;
            MaskReg cmp;
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            Duplicate(one, 1.0f, pregMain);
            Duplicate(zeroF, 0.0f, pregMain);
#define TQ_COUNT_BOUNDARY(B)                                               \
    AscendC::Reg::Compares<float, CMPMODE::GE>(cmp, u, bnd_[B], pregMain); \
    Select(sel, one, zeroF, cmp);                                          \
    Add(nibReg, nibReg, sel, pregMain);
            for (uint16_t i = 0; i < tokenLoop; i++) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(normV, normAddr + i * NORM_SLOT_FLOATS);
                Div(invNorm, one, normV, pregMain);
                for (uint16_t c = 0; c < chunkLoop; c++) {
                    LoadAlign(x, inAddr + i * headDim_ + c * vlFp32);
                    Mul(u, x, invNorm, pregMain);
                    Duplicate(nibReg, 0.0f, pregMain);
                    TQ_COUNT_BOUNDARY(0)
                    TQ_COUNT_BOUNDARY(1)
                    TQ_COUNT_BOUNDARY(2)
                    TQ_COUNT_BOUNDARY(3)
                    TQ_COUNT_BOUNDARY(4)
                    TQ_COUNT_BOUNDARY(5)
                    TQ_COUNT_BOUNDARY(6)
                    TQ_COUNT_BOUNDARY(7)
                    TQ_COUNT_BOUNDARY(8)
                    TQ_COUNT_BOUNDARY(9)
                    TQ_COUNT_BOUNDARY(10)
                    TQ_COUNT_BOUNDARY(11)
                    TQ_COUNT_BOUNDARY(12)
                    TQ_COUNT_BOUNDARY(13)
                    TQ_COUNT_BOUNDARY(14)
#undef TQ_COUNT_BOUNDARY
                    StoreAlign(nibAddr + i * headDim_ + c * vlFp32, nibReg, pregMain);
                }
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }

        // int4 pack without Cast<int4b_t>: byte k = nib[2k] + 16 * nib[2k + 1], assembled from exact
        // small-integer fp32 lanes via two Gathers, then narrowed through half -> uint8 (RINT) to a
        // DIST_PACK4_B32 store. uint8 keeps [0, 255] exactly in range, so the packed byte pattern
        // never depends on NO_SAT out-of-range narrowing behavior. The low nibble holds the even
        // dim, matching the 910b layout (y rows are contiguous, no padding). Indices are generated
        // in-register with Arange and reinterpreted to u32 for the index math (Gather requires
        // u16/u32 indices per the dav_3510 static_assert matrix).
        __VEC_SCOPE__
        {
            RegTensor<float> evenNib;
            RegTensor<float> oddNib;
            RegTensor<float> byteF;
            RegTensor<half> byteH;
            RegTensor<uint8_t> byteU8;
            RegTensor<int32_t> arangeIdx;
            RegTensor<uint32_t> evenIdx;
            RegTensor<uint32_t> oddIdx;
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            // Gather indices must be u16/u32 (device constraint); the u32 mask keys the index math.
            MaskReg pregIdx = CreateMask<uint32_t, MaskPattern::ALL>();
            for (uint16_t i = 0; i < tokenLoop; i++) {
                for (uint16_t c = 0; c < byteLoop; c++) {
                    AscendC::Reg::Arange(arangeIdx, static_cast<int32_t>(c * vlFp32));
                    evenIdx = (RegTensor<uint32_t>&)arangeIdx;
                    AscendC::Reg::Muls(evenIdx, evenIdx, static_cast<uint32_t>(2), pregIdx);
                    AscendC::Reg::Adds(oddIdx, evenIdx, static_cast<uint32_t>(1), pregIdx);
                    AscendC::Reg::Gather(evenNib, nibAddr + i * headDim_, evenIdx, pregMain);
                    AscendC::Reg::Gather(oddNib, nibAddr + i * headDim_, oddIdx, pregMain);
                    Muls(oddNib, oddNib, 16.0f, pregMain);
                    Add(byteF, evenNib, oddNib, pregMain);
                    Cast<half, float, castTraitF32ToF16Rint>(byteH, byteF, pregMain);
                    Cast<uint8_t, half, castTraitF16ToU8Rint>(byteU8, byteH, pregMain);
                    StoreAlign<uint8_t, StoreDist::DIST_PACK4_B32>(
                        (__ubuf__ uint8_t*)(yAddr + i * packedBytes_ + c * vlFp32), byteU8, pregMain);
                }
            }
            // No LocalMemBar here: no VF load below consumes the packed bytes. Their consumer is
            // the CopyOut MTE copy, ordered by the outQ_.EnQue handoff -- the same mechanism the
            // fp16 scale scope below already relies on exclusively.
        }

        // fp16 scale from the final norm: vector Cast on lane 0 plus a DIST_FIRST_ELEMENT_B16
        // store into the per-token 64B slot behind the y rows.
        __VEC_SCOPE__
        {
            RegTensor<float> normF;
            RegTensor<half> normH;
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            for (uint16_t i = 0; i < tokenLoop; i++) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(normF, normAddr + i * NORM_SLOT_FLOATS);
                Cast<half, float, castTraitF32ToF16Rint>(normH, normF, pregMerge);
                StoreAlign<half, StoreDist::DIST_FIRST_ELEMENT_B16>(scaleAddr + i * SCALE_SLOT_HALFS, normH, pregMerge);
            }
        }

        inQ_.FreeTensor(in);
        outQ_.EnQue(out);
    }

    __aicore__ inline void CopyOut(uint32_t t, uint32_t count)
    {
        LocalTensor<uint8_t> out = outQ_.DeQue<uint8_t>();
        DataCopy(yGm_[static_cast<uint64_t>(t) * packedBytes_], out, count * packedBytes_);
        // The fp16 scales sit 64B-strided behind the y rows; each token copies its own 2 bytes out.
        // On this arch DataCopyPad takes blockLen in BYTES (CopyUbufToGmAlignV2: unitOfBytes is 1 for
        // the Pad variant, 32 for DataCopy), so one burst moves exactly one half -- it neither
        // reaches the next slot nor the neighbouring GM scales.
        LocalTensor<half> scale = out[yBytes_].ReinterpretCast<half>();
        for (uint32_t i = 0; i < count; ++i) {
            DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(half)), 0, 0};
            DataCopyPad(scaleGm_[t + i], scale[i * SCALE_SLOT_HALFS], copyParams);
        }
        outQ_.FreeTensor(out);
    }

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inQ_;
    TQue<QuePosition::VECOUT, 1> outQ_;
    TBuf<TPosition::VECCALC> nibBuf_;
    TBuf<TPosition::VECCALC> normBuf_;
    TBuf<TPosition::VECCALC> centBuf_;
    GlobalTensor<float> latentGm_;
    GlobalTensor<float> centGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<half> scaleGm_;
    uint32_t numTokens_ = 0;
    uint32_t headDim_ = 0;
    uint32_t yBytes_ = 0;
    uint32_t packedBytes_ = 0;
    uint32_t batch_ = 1;
    uint32_t tokStart_ = 0;
    uint32_t tokEnd_ = 0;
    float bnd_[N_CENT] = {};
    float centVal_[N_CENT] = {};
};

} // namespace TurboQuantRegbase
#endif // TURBO_QUANT_REGBASE_H
