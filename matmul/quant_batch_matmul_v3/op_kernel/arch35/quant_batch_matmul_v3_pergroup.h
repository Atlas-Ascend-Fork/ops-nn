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
 * \file quant_batch_matmul_v3_pergroup.h
 * \brief Operator tiling adapter and CMCT composition for MC62 X2 per-group.
 */
#pragma once
#include "quant_batch_matmul_v3_tiling_data.h"
#include "cmct/block/block_mmad_qbmm_pergroup.h"
#include "cmct/epilogue/block_epilogue_qbmm_pergroup.h"
#include "cmct/kernel/kernel_qbmm_pergroup.h"

namespace QuantBatchMatmulV3 {

// Count logical elements separately from physical bytes (int4b_t is not a byte-sized weight).
template <class T>
struct X2PerGroupWeightTraits;
template <>
struct X2PerGroupWeightTraits<int8_t> {
    using BType = int8_t;
    static constexpr uint64_t ELEMENTS_PER_BYTE = 1;
    static constexpr uint64_t C0 = 32;
    static constexpr uint64_t TRANSPOSE_STEP = 2;
};
template <>
struct X2PerGroupWeightTraits<AscendC::int4b_t> {
    using BType = AscendC::int4b_t;
    static constexpr uint64_t ELEMENTS_PER_BYTE = 2;
    static constexpr uint64_t C0 = 64;
    static constexpr uint64_t TRANSPOSE_STEP = 4;
};

// Select cache reuse from actual device resources and a complete-K resident window.
__aicore__ inline bool CanX2PerGroupBroadcastReuse(const DequantBmm::QuantBatchMatmulV3BasicAPITilingData& data,
                                                   uint64_t residentBytes)
{
    const auto& mm = data.matmulTiling;
    const auto& p = data.params;
    const uint64_t cores = AscendC::GetBlockNum();
    if (mm.m == 0U || mm.n == 0U || mm.k == 0U || mm.baseM == 0U || mm.baseN == 0U || p.groupSizeK == 0UL ||
        p.batchC <= 1UL || cores == 0UL || residentBytes > AscendC::TOTAL_L1_SIZE ||
        mm.n > UINT32_MAX / sizeof(uint64_t)) {
        return false;
    }
    const uint64_t batchA = p.batchA1 * p.batchA2 * p.batchA3 * p.batchA4;
    const uint64_t batchB = p.batchB1 * p.batchB2 * p.batchB3 * p.batchB4;
    if (batchA >= p.batchC && batchB >= p.batchC) {
        return false;
    }
    const uint64_t mCount = Cmct::Gemm::CeilDiv(static_cast<uint64_t>(mm.m), mm.baseM);
    const uint64_t nCount = Cmct::Gemm::CeilDiv(static_cast<uint64_t>(mm.n), mm.baseN);
    const uint64_t tiles = mCount * nCount;
    const uint64_t groups = Cmct::Gemm::Min<uint64_t>(p.batchC, Cmct::Gemm::CeilDiv(cores, tiles));
    return Cmct::Gemm::CeilDiv(p.batchC, groups) > 1UL;
}

template <class BatchParams>
__aicore__ inline void SetPerGroupBatchOrder(BatchParams& batch, uint64_t batchCount, uint64_t aPanelBytes,
                                             uint64_t bPanelBytes)
{
    if (batchCount == 0UL || aPanelBytes == 0UL || bPanelBytes == 0UL) {
        return;
    }
    const uint64_t bytesLimit = UINT64_MAX / batchCount;
    if (aPanelBytes > bytesLimit || bPanelBytes > bytesLimit - aPanelBytes) {
        return;
    }
    uint32_t order[BatchParams::DIM_COUNT]{};
    uint32_t pos = 0U;
    for (uint32_t i = 0U; i < BatchParams::DIM_COUNT; ++i) {
        if (batch.b[i] > 1UL) {
            order[pos++] = i;
        }
    }
    for (uint32_t i = 0U; i < BatchParams::DIM_COUNT; ++i) {
        if (batch.b[i] == 1UL) {
            order[pos++] = i;
        }
    }
    uint64_t oldA = 1UL, oldB = 1UL, newA = 1UL, newB = 1UL;
    bool oa = true, ob = true, na = true, nb = true;
    for (int32_t i = static_cast<int32_t>(BatchParams::DIM_COUNT) - 1; i >= 0; --i) {
        const uint32_t j = order[i];
        oa = oa && batch.a[i] == 1UL;
        ob = ob && batch.b[i] == 1UL;
        na = na && batch.a[j] == 1UL;
        nb = nb && batch.b[j] == 1UL;
        if (oa) {
            oldA *= batch.c[i];
        }
        if (ob) {
            oldB *= batch.c[i];
        }
        if (na) {
            newA *= batch.c[j];
        }
        if (nb) {
            newB *= batch.c[j];
        }
    }
    if (batchCount / newA * aPanelBytes + batchCount / newB * bPanelBytes <
        batchCount / oldA * aPanelBytes + batchCount / oldB * bPanelBytes) {
        for (uint32_t i = 0U; i < BatchParams::DIM_COUNT; ++i) {
            batch.order[i] = order[i];
        }
    }
}

// This bridge is the only operator-ABI conversion. Each CMCT component receives
// its own Params, following BlockMmadMx / QuantMmBatchPertile composition.
template <class Kernel>
__aicore__ inline typename Kernel::Params MakeX2PerGroupCmctTiling(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR bias, GM_ADDR y,
    const DequantBmm::QuantBatchMatmulV3BasicAPITilingData& data)
{
    using BlockMmad = typename Kernel::BlockMmadOp;
    using WeightTraits = typename BlockMmad::WeightTraits;
    using BatchParams = typename Kernel::BatchParams;
    using BType = typename BlockMmad::BType;
    typename Kernel::Params result{};
    const auto& mm = data.matmulTiling;
    const auto& p = data.params;
    const uint64_t groupCount = Cmct::Gemm::CeilDiv(static_cast<uint64_t>(mm.k), p.groupSizeK);
    const bool pairReduction = Kernel::CanUsePairAccumulation(mm.k, p.groupSizeK, mm.dbL0C);
    constexpr auto loadMode = BlockMmad::LOAD_MODE;

    result.problemShape = {static_cast<int64_t>(mm.m), static_cast<int64_t>(mm.n), static_cast<int64_t>(mm.k),
                           static_cast<int64_t>(p.batchC)};
    auto& compute = result.mmadParams;
    compute.aGmAddr = x1;
    compute.bGmAddr = x2;
    compute.scaleGmAddr = scale;
    compute.m = mm.m;
    compute.n = mm.n;
    compute.k = mm.k;
    compute.baseM = mm.baseM;
    compute.baseN = mm.baseN;
    compute.baseK = mm.baseK;
    compute.dbL0C = mm.dbL0C;
    compute.pairReduction = pairReduction;
    compute.broadcastReuse = false;
    compute.l1Params = {mm.kAL1, mm.kBL1, p.groupSizeK};
    compute.l1Layout = BlockMmad::CalculateL1Layout(compute, groupCount);
    if constexpr (loadMode == Cmct::Gemm::NONE_FULL_LOAD_MODE) {
        const auto resident = BlockMmad::CalculateL1Layout(compute, groupCount, true);
        compute.broadcastReuse = CanX2PerGroupBroadcastReuse(data, resident.bufferStride);
        if (compute.broadcastReuse) {
            compute.l1Layout = resident;
        }
    }

    result.epilogueParams = {y, bias, mm.m, mm.n, mm.baseM, mm.baseN, pairReduction};
    result.groupParams = {p.groupSizeK, pairReduction};
    const bool useWindow = loadMode == Cmct::Gemm::NONE_FULL_LOAD_MODE && !compute.broadcastReuse;
    result.schParams = {useWindow ? mm.baseM : 1U,
                        useWindow ? mm.baseN : 1U,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        static_cast<int64_t>(Cmct::Gemm::Block::WINDOW_LEN)};
    result.batchParams = {{p.batchA1, p.batchA2, p.batchA3, p.batchA4},
                          {p.batchB1, p.batchB2, p.batchB3, p.batchB4},
                          {p.batchC1, p.batchC2, p.batchC3, p.batchC4},
                          {}};
    for (uint32_t i = 0U; i < BatchParams::DIM_COUNT; ++i) {
        result.batchParams.order[i] = i;
    }
    if (compute.broadcastReuse && mm.m != 0U && mm.n != 0U && mm.baseM != 0U && mm.baseN != 0U && p.batchC != 0UL) {
        const uint64_t spatial = Cmct::Gemm::CeilDiv(static_cast<uint64_t>(mm.m), mm.baseM) *
                                 Cmct::Gemm::CeilDiv(static_cast<uint64_t>(mm.n), mm.baseN);
        const uint64_t coreCount = AscendC::GetBlockNum();
        if (Cmct::Gemm::Min<uint64_t>(p.batchC, Cmct::Gemm::CeilDiv(coreCount, spatial)) == 1UL) {
            const uint64_t aPanelBytes = compute.l1Layout.aElements * sizeof(int8_t);
            const uint64_t bPanelBytes = Cmct::Gemm::CeilDiv(compute.l1Layout.bElements,
                                                             WeightTraits::ELEMENTS_PER_BYTE) +
                                         groupCount * compute.l1Layout.alignedN * sizeof(uint64_t);
            SetPerGroupBatchOrder(result.batchParams, p.batchC, aPanelBytes, bPanelBytes);
        }
    }
    // Host leaves unused scaleKL1/nBufferNum/weightMustHitL2 at zero. This
    // route uses the explicit layout above and preserves its existing L2 policy.
    return result;
}

// The dispatch selects INT8 or INT4 weights; both use this CMCT composition.
template <class BType, bool TransB, uint64_t LoadMode>
__aicore__ inline void QbmmX2PerGroupCmctKernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR bias, GM_ADDR y,
                                                const DequantBmm::QuantBatchMatmulV3BasicAPITilingData& data,
                                                AscendC::TPipe* pipe)
{
    using BlockMmad = Cmct::Gemm::Block::BlockMmadX2PerGroup<X2PerGroupWeightTraits<BType>, TransB, LoadMode>;
    using BlockEpilogue = Cmct::Gemm::Block::BlockEpilogueX2PerGroup<typename BlockMmad::GroupResultParams>;
    using BlockScheduler = Cmct::Gemm::QuantBatchMatmulV3Scheduler<>;
    using Kernel = Cmct::Gemm::Kernel::QuantMmBatchX2PerGroup<Cmct::Gemm::MatmulShape, BlockMmad, BlockEpilogue,
                                                              BlockScheduler>;
    const auto params = MakeX2PerGroupCmctTiling<Kernel>(x1, x2, scale, bias, y, data);
    Kernel kernel;
    kernel(params, pipe);
}

} // namespace QuantBatchMatmulV3
