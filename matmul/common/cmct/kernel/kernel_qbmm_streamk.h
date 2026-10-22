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
 * \file kernel_qbmm_mx.h
 * \brief
 */

#ifndef KERNEL_QBMM_STREAMK_H
#define KERNEL_QBMM_STREAMK_H

#include "../block/block_scheduler_qbmm_streamk.h"
#include "../utils/common_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/fill_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/quant_batch_matmul_constant.h"
#include "../utils/tensor_utils.h"
#include "../utils/tuple_utils.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {

#define QBMM_STREAM_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QBMM_STREAMK_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

using namespace Cmct;
using namespace Cmct::Gemm;
using namespace AscendC;

namespace {
constexpr static uint16_t AIC_SYNC_AIV_MODE_4 = 4;
constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;
constexpr static uint16_t FLAG_ID_MAX = 16;

constexpr static uint64_t IDX_BLOCK_OFFSET_X1 = 0UL;
constexpr static uint64_t IDX_BLOCK_OFFSET_X2 = 1UL;
constexpr static uint64_t IDX_BLOCK_OFFSET_X1_SCALE = 2UL;
constexpr static uint64_t IDX_BLOCK_OFFSET_X2_SCALE = 3UL;
constexpr static uint64_t IDX_BLOCK_OFFSET_C = 4UL;
constexpr static uint64_t IDX_BLOCK_OFFSET_BIAS = 5UL;
} // namespace

QBMM_STREAM_KERNEL_CLASS_TEM_PARAMS
class KernelQBMMStreamk {
public:
    __aicore__ inline KernelQBMMStreamk() {}
    __aicore__ inline ~KernelQBMMStreamk() {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmad::L1TileShape,
                                                                    typename BlockMmad::L0TileShape, BlockScheduler,
                                                                    transA, transB>::SchedulerOp;

    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using ScaleType = typename BlockMmad::ScaleType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutB = typename BlockMmad::LayoutB;
    static constexpr CubeFormat FormatB = TagToFormat<LayoutB>::format;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using CoordClass = Coordinate<transA, transB, CubeFormat::ND, FormatB, CubeFormat::ND>;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;
    using EpilogueParams = typename BlockEpilogue::Params;
    using WorkspaceDtype = typename BlockEpilogue::WorkspaceType;

    struct QBMMTiling {
        uint32_t batchA1;
        uint32_t batchA2;
        uint32_t batchA3;
        uint32_t batchA4;
        uint32_t batchB1;
        uint32_t batchB2;
        uint32_t batchB3;
        uint32_t batchB4;
        uint32_t batchC1;
        uint32_t batchC2;
        uint32_t batchC3;
        uint32_t batchC4;
        uint32_t baseM = 0;
        uint32_t baseN = 0;
        uint32_t baseK = 0;
        uint32_t isBias;
        uint32_t dbL0C;
    };

    struct AddrParams {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
    };

    struct Params {
        ProblemShape problemShape;
        AddrParams addrParams;
        L1Params l1Params;
        EpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
    };

public:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params) { Run(params); }

private:
    __aicore__ inline void ProcessSingleBatch(const Params& params, BlockSchedulerOp& bs);
    __aicore__ inline TupleShape ToShapeTuple(const ProblemShape& problemShape)
    {
        return {problemShape.m, problemShape.n, problemShape.k};
    }
    __aicore__ inline void GetScalarScaleAiv(const Params& params, AscendC::GlobalTensor<ScaleType>& x1Scale,
                                             AscendC::GlobalTensor<ScaleType>& x2Scale);

private:
    BlockMmad mmadOp_;
    BlockEpilogue epilogueOp_;
    TupleShape problemShape_{};
    uint64_t mL1_{0UL};
    uint64_t nL1_{0UL};
    uint64_t kL1_{0UL};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<WorkspaceDtype> workspaceGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    AscendC::GlobalTensor<ScaleType> x1scaleGlobal_;
    AscendC::GlobalTensor<ScaleType> x2scaleGlobal_;
    uint64_t blockIdx_;
    uint64_t batchCOffset_{0};
    uint64_t batchAOffset_{0};
    uint64_t batchBOffset_{0};
    bool isBias_{false};

    float scalarScaleFp32_ = 1;
    uint64_t usedCoreNum_{0};
    const static uint64_t BLOCK_BASE_M = 256;
    const static uint64_t BLOCK_BASE_N = 256;
};

QBMM_STREAM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void KernelQBMMStreamk<QBMM_STREAMK_KERNEL_FUN_TEM_PARAMS>::Run(const Params& params)
{
    Init(params);
    BlockSchedulerOp bs(params.problemShape, params.schParams);
    problemShape_ = ToShapeTuple(params.problemShape);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    bool enableL0CPingPong = (params.qbmmParams.dbL0C > 1);
    enableL0CPingPong = false;
    mmadOp_.Init(problemShape_, l0TileShape, params.l1Params, isBias_, enableL0CPingPong);

    if (params.problemShape.b == 1) {
        ProcessSingleBatch(params, bs);
        return;
    }
}

QBMM_STREAM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void KernelQBMMStreamk<QBMM_STREAMK_KERNEL_FUN_TEM_PARAMS>::Init(const Params& params)
{
    usedCoreNum_ = params.schParams.usedCoreNum;
    mL1_ = params.schParams.baseM;
    nL1_ = params.schParams.baseN;
    kL1_ = params.schParams.kL1;

    cGlobal_.SetGlobalBuffer((__gm__ CType*)params.addrParams.cGmAddr);
    workspaceGlobal_.SetGlobalBuffer((__gm__ WorkspaceDtype*)params.addrParams.workspaceGmAddr);
    x1scaleGlobal_.SetGlobalBuffer((__gm__ ScaleType*)params.addrParams.x1ScaleGmAddr);
    x2scaleGlobal_.SetGlobalBuffer((__gm__ ScaleType*)params.addrParams.x2ScaleGmAddr);

    if ASCEND_IS_AIV {
        if (params.l1Params.x2QuantMode == static_cast<uint32_t>(QuantBatchMatmul::QuantMode::PERTENSOR_MODE)) {
            GetScalarScaleAiv(params, x1scaleGlobal_, x2scaleGlobal_);
        }
        return;
    }

    aGlobal_.SetGlobalBuffer((__gm__ AType*)params.addrParams.aGmAddr);
    bGlobal_.SetGlobalBuffer((__gm__ BType*)params.addrParams.bGmAddr);

    if (params.qbmmParams.isBias == 1) {
        isBias_ = true;
        biasGlobal_.SetGlobalBuffer((__gm__ BiasType*)params.addrParams.biasGmAddr);
    }
}

QBMM_STREAM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void KernelQBMMStreamk<QBMM_STREAMK_KERNEL_FUN_TEM_PARAMS>::GetScalarScaleAiv(
    const Params& params, AscendC::GlobalTensor<ScaleType>& x1Scale, AscendC::GlobalTensor<ScaleType>& x2Scale)
{
    if (params.l1Params.x1QuantMode == static_cast<uint32_t>(QuantBatchMatmul::QuantMode::MX_PERGROUP_MODE) ||
        params.l1Params.x1QuantMode == static_cast<uint32_t>(QuantBatchMatmul::QuantMode::HIF4_PEGROUP_MODE)) {
        return;
    }

    if (params.l1Params.x1QuantMode == static_cast<uint32_t>(QuantBatchMatmul::QuantMode::PERTENSOR_MODE) &&
        params.l1Params.x2QuantMode == static_cast<uint32_t>(QuantBatchMatmul::QuantMode::PERTENSOR_MODE)) {
        if constexpr (IsSameType<ScaleType, float>::value) {
            float deqScale = x1Scale.GetValue(0) * x2Scale.GetValue(0);
            uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&deqScale));
            scalarScaleFp32_ = *reinterpret_cast<float*>(uint32Scale & QuantBatchMatmul::DEQ_SCALE_MUL32);

            uint64_t testout = static_cast<uint64_t>(uint32Scale & QuantBatchMatmul::DEQ_SCALE_MUL32);
        }
    } else if (params.l1Params.x2QuantMode == static_cast<uint32_t>(QuantBatchMatmul::QuantMode::PERTENSOR_MODE)) {
        if constexpr (IsSameType<ScaleType, uint64_t>::value || IsSameType<ScaleType, int64_t>::value) {
            uint64_t scalarScale = x2Scale.GetValue(0);
            uint32_t validBit = static_cast<uint32_t>(scalarScale & QuantBatchMatmul::DEQ_SCALE_MUL);
            scalarScaleFp32_ = *reinterpret_cast<float*>(&validBit);
        } else if constexpr (IsSameType<ScaleType, bfloat16_t>::value) {
            AscendC::GlobalTensor<uint16_t> scaleUint16;
            scaleUint16.SetGlobalBuffer((__gm__ uint16_t*)x2Scale.GetPhyAddr());
            uint16_t uint16Scale = scaleUint16.GetValue(0);
            uint32_t uint32Scale = static_cast<uint32_t>(uint16Scale << QuantBatchMatmul::LEFT_SHIFT_16);
            scalarScaleFp32_ = *reinterpret_cast<float*>(&uint32Scale);
        } else if constexpr (IsSameType<ScaleType, float>::value) {
            float deqScale = x2Scale.GetValue(0);
            uint32_t uint32Scale = *(reinterpret_cast<uint32_t*>(&deqScale));
            scalarScaleFp32_ = *reinterpret_cast<float*>(uint32Scale & QuantBatchMatmul::DEQ_SCALE_MUL32);
        }
    }
}

QBMM_STREAM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void KernelQBMMStreamk<QBMM_STREAMK_KERNEL_FUN_TEM_PARAMS>::ProcessSingleBatch(const Params& params,
                                                                                                 BlockSchedulerOp& bs)
{
    int64_t mTileNum = Get<MNK_M>(bs.GetMNKTileNum());
    int64_t nTileNum = Get<MNK_N>(bs.GetMNKTileNum());
    int64_t skKTileNum = Get<MNK_K>(bs.GetMNKTileNum()); // it only used in sk
    int64_t tileNum = bs.GetTotalTileNum();

    if ASCEND_IS_AIC {
        CoordClass coord(params.problemShape.m, params.problemShape.n, params.problemShape.k, params.qbmmParams.baseM,
                         params.qbmmParams.baseN, params.qbmmParams.baseK);
        BlockCoord blockIdx;
        int64_t curBlockIdx = AscendC::GetBlockIdx();

        if (curBlockIdx >= bs.GetBlockNum(usedCoreNum_)) {
            CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
            CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
            return;
        }

        int64_t tailSKTotalTileNum = static_cast<int64_t>(((mTileNum * nTileNum) % usedCoreNum_) * skKTileNum);

        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += usedCoreNum_) {
            int64_t tmpTileIdx = tileIdx;

            if (!bs.CheckIsSkScene(0)) { // SK Preload in DP+SK
                if (tileIdx % usedCoreNum_ < tailSKTotalTileNum &&
                    (Cmct::Gemm::CeilDiv(static_cast<uint64_t>(tileIdx + 1), usedCoreNum_) ==
                     (Cmct::Gemm::CeilDiv(static_cast<uint64_t>(tileNum), usedCoreNum_) - 1))) {
                    tmpTileIdx = tileIdx + usedCoreNum_;
                } else if (tileIdx % usedCoreNum_ < tailSKTotalTileNum &&
                           (Cmct::Gemm::CeilDiv(static_cast<uint64_t>(tileIdx + 1), usedCoreNum_) ==
                            Cmct::Gemm::CeilDiv(static_cast<uint64_t>(tileNum), usedCoreNum_))) {
                    tmpTileIdx = tileIdx - usedCoreNum_;
                }
            }

            auto singleCoreShape = bs.GetSingleCoreShape(tmpTileIdx);
            using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
            BlockCoord singleCoreCoord = bs.GetSingleCoreCoord(tmpTileIdx);
            using TupleShapeType = AscendC::Shape<int64_t, int64_t, int64_t>;

            auto blockOffset = Cmct::Gemm::GetQuantStreamkOffset<BlockCoord, TupleShapeType, FormatB, BType>(
                singleCoreCoord,
                TupleShapeType(Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_)),
                BlockCoord(mL1_, nL1_, kL1_, 1), bs.GetCurKSingleCore(tmpTileIdx), transA, transB, isBias_,
                params.l1Params.x1QuantMode, params.l1Params.x2QuantMode);

            int64_t offsetA = Get<IDX_BLOCK_OFFSET_X1>(blockOffset);
            int64_t offsetB = Get<IDX_BLOCK_OFFSET_X2>(blockOffset);
            int64_t offsetX1scale = Get<IDX_BLOCK_OFFSET_X1_SCALE>(blockOffset);
            int64_t offsetX2scale = Get<IDX_BLOCK_OFFSET_X2_SCALE>(blockOffset);
            int64_t offsetC = Get<IDX_BLOCK_OFFSET_C>(blockOffset);
            int64_t offsetBias = Get<IDX_BLOCK_OFFSET_BIAS>(blockOffset);
            int64_t offsetWorkspace = (((tmpTileIdx % usedCoreNum_) / skKTileNum) * skKTileNum +
                                       Get<MNK_K>(singleCoreCoord)) *
                                      params.qbmmParams.baseM * params.qbmmParams.baseN;

            bool enableAddBias = false;
            if (isBias_ && Get<MNK_K>(singleCoreCoord) == 0) {
                enableAddBias = true;
            }

            mmadOp_(aGlobal_[offsetA], bGlobal_[offsetB], x1scaleGlobal_[offsetX1scale], x2scaleGlobal_[offsetX2scale],
                    biasGlobal_[offsetBias], cGlobal_[offsetC], workspaceGlobal_[offsetWorkspace], singleCoreShape,
                    enableAddBias, bs.CheckIsSkScene(tmpTileIdx));

            if (tmpTileIdx + usedCoreNum_ >= tileNum) {
                CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
                CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
            }
            bs.IncrementRoundIdx();
        }
        bs.UpdateNextBatchBlockRoundParams();
    }

    if ASCEND_IS_AIV {
        uint64_t lastLoopTotalCnt = (mTileNum * nTileNum % usedCoreNum_) * skKTileNum;
        uint64_t curBlockIdxInAiv = AscendC::GetBlockIdx();

        // 核id 小于 用于 sk 的核数才会继续做后处理
        if (curBlockIdxInAiv >= lastLoopTotalCnt * AscendC::GetTaskRation()) {
            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
            SyncAll();
            return;
        }

        // 做累加也不需要所有后处理的核都做
        CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
        SyncAll();
        BlockShape epilogueShape(Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_), 1);
        BlockCoord tileL1Shape(mL1_, nL1_, kL1_, 1);
        BlockCoord tileL0Shape(params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0);

        epilogueOp_.Init(params.epilogueParams, epilogueShape, tileL1Shape, bs.GetMNKTileNum(), usedCoreNum_,
                         bs.CheckIsSkScene(0), tileL0Shape);
        if (params.l1Params.x2QuantMode == static_cast<uint32_t>(QuantBatchMatmul::QuantMode::PERTENSOR_MODE)) {
            epilogueOp_.SetDeqScale(scalarScaleFp32_);
        }
        epilogueOp_();
    }
}

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct

#endif
