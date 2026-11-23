/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_BASIC_SYRK_WORKSPACE_TLA_HPP
#define CATLASS_GEMM_KERNEL_BASIC_SYRK_WORKSPACE_TLA_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm_tla.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/matrix.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include <cstdint>

#ifndef CATLASS_DEVICE_SIMD_VF
#define CATLASS_DEVICE_SIMD_VF __simd_vf__ static
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
inline constexpr AscendC::FixpipeConfig kGemmSymmRowUb = {AscendC::CO2Layout::ROW_MAJOR, true};
inline constexpr AscendC::FixpipeConfig kGemmSymmColUb = {AscendC::CO2Layout::COLUMN_MAJOR, true};
#endif

// v1.5.0 CopyL0CToUBTla has no sub-block id and no ColumnMajor UB store.
struct GemmSymmFixpipeUb {
    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dst, TensorSrc const& src, bool subBlockId, uint8_t unitFlag) const
    {
        using ElementDst = typename TensorDst::Element;
        using ElementSrc = typename TensorSrc::Element;
        constexpr auto quantPre = Catlass::Gemm::Tile::CopyL0CToDstQuantMode<
            Catlass::Arch::Ascend950, ElementSrc, ElementDst, Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT>::VALUE;
        auto dstOffset = dst.layout()(dst.coord());
        auto srcOffset = src.layout()(src.coord());
        if constexpr (tla::detail::isRowMajor<typename TensorDst::Layout>::value) {
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
            params.nSize = tla::get<1>(dst.originShape());
            params.mSize = tla::get<0>(dst.originShape());
            params.srcStride = tla::get<1, 1>(src.stride()) / tla::get<0, 0>(src.stride());
            params.dstStride = tla::get<0>(dst.stride());
            params.quantPre = quantPre;
            params.reluEn = false;
            params.unitFlag = unitFlag;
            params.dualDstCtl = 0;
            params.subBlockId = subBlockId;
            AscendC::Fixpipe<ElementDst, ElementSrc, kGemmSymmRowUb>(dst.data()[dstOffset], src.data()[srcOffset],
                                                                     params);
        } else {
            constexpr uint32_t elePerBlk = Catlass::BytesToBits(Catlass::BYTE_PER_BLK) /
                                           Catlass::SizeOfBits<ElementDst>::value;
            AscendC::FixpipeParamsC310<AscendC::CO2Layout::COLUMN_MAJOR> params;
            params.nSize = tla::get<1>(dst.originShape());
            params.mSize = RoundUp<elePerBlk>(tla::get<0>(dst.originShape()));
            params.srcStride = tla::get<1, 1>(src.stride()) / tla::get<0, 0>(src.stride());
            params.dstStride = tla::get<1>(dst.stride());
            params.params = AscendC::Nz2DnParams(1, 0, 0, 1);
            params.quantPre = quantPre;
            params.reluEn = false;
            params.unitFlag = unitFlag;
            params.dualDstCtl = 0;
            params.subBlockId = subBlockId;
            AscendC::Fixpipe<ElementDst, ElementSrc, kGemmSymmColUb>(dst.data()[dstOffset], src.data()[srcOffset],
                                                                     params);
        }
    }
};

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, uint32_t WS_STAGES_ = 2,
          bool kFixpipeBf16_ = false, bool kFixpipeFp32Ub_ = false>
class BasicSyrkWorkspaceTla {
public:
    using BlockMmad = BlockMmad_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;

    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementX = typename BlockMmad::ElementX;
    using LayoutTagX = typename BlockMmad::LayoutTagX;
    using LayoutX = typename BlockMmad::LayoutX;
    using ElementXt = typename BlockMmad::ElementXt;
    using LayoutTagXt = typename BlockMmad::LayoutTagXt;
    using LayoutXt = typename BlockMmad::LayoutXt;
    using ElementP = typename BlockMmad::ElementY;
    using LayoutTagP = typename BlockMmad::LayoutTagY;
    using LayoutP = typename BlockMmad::LayoutY;
    using LayoutTagPT = layout::ColumnMajor;
    using LayoutPT = detail::TagToLayout_t<ElementP, LayoutTagPT>;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using ElementY = typename BlockEpilogue::ElementY;
    using LayoutY = typename BlockEpilogue::LayoutY;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t WS_STAGES = WS_STAGES_;
    static constexpr bool kFixpipeBf16 = kFixpipeBf16_;
    static constexpr bool kFixpipeFp32Ub = kFixpipeFp32Ub_;
    static constexpr uint32_t SLOT_ELEMS = L1_TILE_M * L1_TILE_N;
    static constexpr uint32_t SLOT_BYTES = SLOT_ELEMS * sizeof(ElementP);
    static constexpr uint32_t STAGE_BYTES = SLOT_BYTES * 2;
    static constexpr uint32_t CORE_BYTES = STAGE_BYTES * WS_STAGES;

    static_assert(std::is_same_v<ElementP, float>, "Workspace P tiles must be float");
    static_assert(WS_STAGES >= 1 && WS_STAGES <= 2, "WS_STAGES must be 1 or 2");
    static_assert(!kFixpipeBf16 || std::is_same_v<ElementD, half> || std::is_same_v<ElementD, bfloat16_t>,
                  "UB fixpipe cast expects fp16 or bf16 output");
    static_assert(!kFixpipeBf16 || std::is_same_v<ElementY, ElementD>,
                  "UB fixpipe cast expects C and Y to share one 16-bit type");
    static_assert(!kFixpipeBf16 || !BlockMmad::MIRROR_DIAGONAL, "UB fixpipe path does not mirror a diagonal tile");
    static constexpr uint32_t UB_TILE_BYTES = L1_TILE_M * L1_TILE_N * sizeof(ElementD);
    // v1.5.0 keeps an 8KB spill reserve in UB_SIZE. These two tiles use the physical 256KB.
    static constexpr uint32_t kPhysicalUbBytes = 256 * 1024;
    static_assert(!kFixpipeBf16 || UB_TILE_BYTES * 2 <= kPhysicalUbBytes, "bf16 tile plus x3 exceeds UB");
    static_assert(!(kFixpipeBf16 && kFixpipeFp32Ub), "pick one UB fixpipe path");
    static constexpr uint32_t FP32_UB_BYTES = L1_TILE_M * L1_TILE_N * sizeof(ElementP);
    static_assert(!kFixpipeFp32Ub || FP32_UB_BYTES + UB_TILE_BYTES <= kPhysicalUbBytes,
                  "fp32 fixpipe tile plus x3 exceeds UB");

    using EpilogueParams = typename BlockEpilogue::Params;

    struct Params {
        uint32_t batchCount;
        GemmCoord problemShape;
        GM_ADDR ptrX;
        GM_ADDR ptrY;
        GM_ADDR ptrD;
        GM_ADDR ptrWorkspace;
        LayoutX layoutX;
        LayoutXt layoutXt;
        LayoutY layoutY;
        LayoutD layoutD;
        int64_t strideX;
        int64_t strideY;
        int64_t strideD;
        float alpha;
        float beta;
        GM_ADDR ptrXt{nullptr};
        int64_t strideXt{0};

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(uint32_t batchCount_, GemmCoord const& problemShape_, GM_ADDR ptrX_, GM_ADDR ptrY_, GM_ADDR ptrD_,
               GM_ADDR ptrWorkspace_, LayoutX layoutX_, LayoutXt layoutXt_, LayoutY layoutY_, LayoutD layoutD_,
               int64_t strideX_, int64_t strideY_, int64_t strideD_, float alpha_, float beta_,
               GM_ADDR ptrXt_ = nullptr, int64_t strideXt_ = 0)
            : batchCount(batchCount_),
              problemShape(problemShape_),
              ptrX(ptrX_),
              ptrY(ptrY_),
              ptrD(ptrD_),
              ptrWorkspace(ptrWorkspace_),
              layoutX(layoutX_),
              layoutXt(layoutXt_),
              layoutY(layoutY_),
              layoutD(layoutD_),
              strideX(strideX_),
              strideY(strideY_),
              strideD(strideD_),
              alpha(alpha_),
              beta(beta_),
              ptrXt(ptrXt_),
              strideXt(strideXt_)
        {}
    };

    struct Arguments {
        uint32_t batchCount;
        GemmCoord problemShape;
        uint8_t* ptrX;
        uint8_t* ptrY;
        uint8_t* ptrD;
        float alpha;
        float beta;
        uint32_t aicCoreNum;
        uint8_t* ptrXt{nullptr};
    };

    static bool CanImplement(const Arguments& args)
    {
        bool hasY = (args.beta == 0.0f) || (args.ptrY != nullptr);
        return args.problemShape.m() == args.problemShape.n() && args.aicCoreNum > 0 && hasY;
    }

    static size_t GetWorkspaceSize(const Arguments& args)
    {
        if constexpr (kFixpipeBf16 || kFixpipeFp32Ub) {
            return 0;
        }
        return static_cast<size_t>(args.aicCoreNum) * CORE_BYTES;
    }

    static Params ToUnderlyingArguments(const Arguments& args, uint8_t* workspace)
    {
        uint32_t m = args.problemShape.m();
        uint32_t k = args.problemShape.k();
        int64_t strideX = static_cast<int64_t>(m) * k;
        int64_t strideY = static_cast<int64_t>(m) * m;
        int64_t strideD = strideY;
        using LayoutTagY = layout::RowMajor;
        return Params{
            args.batchCount,
            args.problemShape,
            args.ptrX,
            args.ptrY,
            args.ptrD,
            workspace,
            tla::MakeLayout<ElementX, LayoutTagX>(m, k),
            tla::MakeLayout<ElementXt, LayoutTagXt>(k, m),
            tla::MakeLayout<ElementY, LayoutTagY>(m, m),
            tla::MakeLayout<ElementD, LayoutTagY>(m, m),
            strideX,
            strideY,
            strideD,
            args.alpha,
            args.beta,
            args.ptrXt,
            strideX,
        };
    }

    CATLASS_DEVICE
    BasicSyrkWorkspaceTla()
    {
        if ASCEND_IS_AIV {
            for (uint32_t i = 0; i < WS_STAGES; ++i) {
                AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + i);
            }
        }
    }

    CATLASS_DEVICE
    ~BasicSyrkWorkspaceTla()
    {
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < WS_STAGES; ++i) {
                AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_FIX>(AIV_SYNC_AIC_FLAG + i);
            }
        }
    }

    template <int32_t CoreType_ = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        BlockScheduler blockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N), AscendC::GetBlockNum());
        uint32_t coreLoops = params.batchCount * blockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementX> gmX;
        gmX.SetGlobalBuffer((__gm__ ElementX*)params.ptrX);
        AscendC::GlobalTensor<ElementXt> gmXt;
        GM_ADDR ptrXt = params.ptrXt != nullptr ? params.ptrXt : params.ptrX;
        int64_t strideXt = params.ptrXt != nullptr ? params.strideXt : params.strideX;
        gmXt.SetGlobalBuffer((__gm__ ElementXt*)ptrXt);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t wsStage = 0;
        LayoutP layoutSlotP = tla::MakeLayout<ElementP, LayoutTagP>(L1_TILE_M, L1_TILE_N);
        LayoutPT layoutSlotPT = tla::MakeLayout<ElementP, LayoutTagPT>(L1_TILE_M, L1_TILE_N);

        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            uint32_t batchIdx = blockScheduler.GetBatchIdx(loopIdx);
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = blockScheduler.GetTaskShape(loopIdx);
            MatrixCoord origin = blockScheduler.GetElementOffset(loopIdx);

            int64_t batchOffsetX = static_cast<int64_t>(batchIdx) * params.strideX;
            int64_t batchOffsetXt = static_cast<int64_t>(batchIdx) * strideXt;
            auto tensorX = tla::MakeTensor(gmX[batchOffsetX], params.layoutX, Arch::PositionGM{});
            auto tensorXt = tla::MakeTensor(gmXt[batchOffsetXt], params.layoutXt, Arch::PositionGM{});

            auto tileX = tla::GetTile(tensorX, tla::MakeCoord(origin.row(), 0),
                                      tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
            auto tileXt = tla::GetTile(tensorXt, tla::MakeCoord(0, origin.column()),
                                       tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));

            GM_ADDR stageBase = nullptr;
            if constexpr (!kFixpipeBf16) {
                stageBase = params.ptrWorkspace + coreIdx * CORE_BYTES + wsStage * STAGE_BYTES;
            }
            AscendC::GlobalTensor<ElementP> gmP;
            gmP.SetGlobalBuffer((__gm__ ElementP*)stageBase);
            AscendC::GlobalTensor<ElementP> gmPT;
            gmPT.SetGlobalBuffer((__gm__ ElementP*)(stageBase + SLOT_BYTES));
            auto tensorP = tla::MakeTensor(gmP, layoutSlotP, Arch::PositionGM{});
            auto tensorPT = tla::MakeTensor(gmPT, layoutSlotPT, Arch::PositionGM{});
            auto tileP = tla::GetTile(tensorP, tla::MakeCoord(0, 0),
                                      tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
            auto tilePT = tla::GetTile(tensorPT, tla::MakeCoord(0, 0),
                                       tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

            bool diagonalMirror = false;
            if constexpr (BlockMmad::MIRROR_DIAGONAL) {
                diagonalMirror = blockCoord.m() == blockCoord.n();
            }

            if constexpr (kFixpipeBf16 || kFixpipeFp32Ub) {
                if (params.alpha != 0.0f) {
                    if constexpr (kFixpipeFp32Ub && BlockMmad::L0C_STAGES > 1) {
                        blockMmad.Compute(tileX, tileXt, actualBlockShape);
                        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
                        auto ubP = resource.ubBuf.template GetBufferByByte<ElementP>(0);
                        Fp32UbFixpipe store{ubP, actualBlockShape.m(), actualBlockShape.n(),
                                            blockCoord.m() > blockCoord.n()};
                        blockMmad.CopyOut(tileP, tilePT, actualBlockShape, blockCoord, &store);
                    } else {
                        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
                        if constexpr (kFixpipeFp32Ub) {
                            auto ubP = resource.ubBuf.template GetBufferByByte<ElementP>(0);
                            Fp32UbFixpipe store{ubP, actualBlockShape.m(), actualBlockShape.n(),
                                                blockCoord.m() > blockCoord.n()};
                            blockMmad(tileX, tileXt, tileP, tilePT, actualBlockShape, blockCoord, &store);
                        } else {
                            auto ubP = resource.ubBuf.template GetBufferByByte<ElementD>(0);
                            Bf16UbFixpipe store{ubP, actualBlockShape.m(), actualBlockShape.n(),
                                                blockCoord.m() > blockCoord.n()};
                            blockMmad(tileX, tileXt, tileP, tilePT, actualBlockShape, blockCoord, &store);
                        }
                    }
                    AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
                }
                continue;
            }

            AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_FIX>(AIV_SYNC_AIC_FLAG + wsStage);

            if (diagonalMirror) {
                WaitDiagonalEntered();
                auto normal = resource.ubBuf.template GetBufferByByte<ElementP>(0);
                auto transposed = resource.ubBuf.template GetBufferByByte<ElementP>(DiagonalBandBytes());
                DiagonalFixpipe store{normal, transposed, actualBlockShape.m()};
                blockMmad(tileX, tileXt, tileP, tilePT, actualBlockShape, blockCoord, &store);
                WaitDiagonalVector();
            } else {
                blockMmad(tileX, tileXt, tileP, tilePT, actualBlockShape, blockCoord);
                AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_FIX>(AIC_SYNC_AIV_FLAG + wsStage);
            }
            wsStage = (wsStage + 1 < WS_STAGES) ? (wsStage + 1) : 0;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        BlockScheduler blockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N), AscendC::GetBlockNum());
        uint32_t coreLoops = params.batchCount * blockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        if constexpr (kFixpipeFp32Ub) {
            ConsumeFp32Ub(params, resource);
            return;
        }
        if constexpr (kFixpipeBf16) {
            ConsumeBf16Ub(params, resource);
            return;
        }
        EpilogueParams epiParams{params.alpha, params.beta, params.ptrY, params.ptrD, params.layoutY, params.layoutD};
        BlockEpilogue blockEpilogue(resource, epiParams);

        using StoreMode = typename BlockEpilogue::SymmStoreMode;
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t aicBlockIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicBlockNum = AscendC::GetBlockNum();
        uint32_t wsStage = 0;

        for (uint32_t loopIdx = aicBlockIdx; loopIdx < coreLoops; loopIdx += aicBlockNum) {
            uint32_t batchIdx = blockScheduler.GetBatchIdx(loopIdx);
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = blockScheduler.GetTaskShape(loopIdx);
            MatrixCoord origin = blockScheduler.GetElementOffset(loopIdx);
            uint32_t mActual = actualBlockShape.m();
            uint32_t nActual = actualBlockShape.n();
            uint32_t mStart = origin.row();
            uint32_t nStart = origin.column();
            bool dualWrite = blockCoord.m() > blockCoord.n();
            bool diagonalMirror = false;
            if constexpr (BlockMmad::MIRROR_DIAGONAL) {
                diagonalMirror = blockCoord.m() == blockCoord.n();
            }
            int64_t batchOffsetY = static_cast<int64_t>(batchIdx) * params.strideY;

            GM_ADDR stageBase = params.ptrWorkspace + aicBlockIdx * CORE_BYTES + wsStage * STAGE_BYTES;

            if (diagonalMirror) {
                ApplyDiagonalTile(params, resource, mStart, nStart, mActual, batchOffsetY);
            } else {
                AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE2>(AIC_SYNC_AIV_FLAG + wsStage);

                if (subBlockIdx == 0) {
                    AscendC::GlobalTensor<ElementP> gmP;
                    gmP.SetGlobalBuffer((__gm__ ElementP*)stageBase);
                    auto layoutP = tla::MakeLayout(tla::MakeShape(mActual, nActual),
                                                   tla::MakeStride(static_cast<int64_t>(L1_TILE_N), tla::Int<1>{}));
                    auto tensorP = tla::MakeTensor(gmP, layoutP, Arch::PositionGM{});
                    blockEpilogue(tensorP, mStart, nStart, mActual, nActual, batchOffsetY, StoreMode::FULL);
                } else if (dualWrite) {
                    AscendC::GlobalTensor<ElementP> gmPT;
                    gmPT.SetGlobalBuffer((__gm__ ElementP*)(stageBase + SLOT_BYTES));
                    auto layoutPT = tla::MakeLayout(tla::MakeShape(nActual, mActual),
                                                    tla::MakeStride(static_cast<int64_t>(L1_TILE_M), tla::Int<1>{}));
                    auto tensorPT = tla::MakeTensor(gmPT, layoutPT, Arch::PositionGM{});
                    blockEpilogue(tensorPT, nStart, mStart, nActual, mActual, batchOffsetY, StoreMode::FULL);
                }
            }

            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE3>(AIV_SYNC_AIC_FLAG + wsStage);
            wsStage = (wsStage + 1 < WS_STAGES) ? (wsStage + 1) : 0;
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    static constexpr uint16_t CROSS_CORE_SYNC_MODE_2 = 2;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 8;
    static constexpr uint16_t DIAG_SYNC_MODE = 4;
    static constexpr uint16_t DIAG_RELEASE = 4;
    static constexpr uint16_t DIAG_READY = 5;
    static constexpr uint16_t DIAG_ENTER = 7;
    static constexpr uint16_t DIAG_SECOND_AIV = 16;
    static constexpr uint32_t DIAG_BAND_ROWS = 64;

    static constexpr uint32_t kUbAlign = 32 / sizeof(ElementD);

    CATLASS_DEVICE static uint32_t AlignUb(uint32_t n) { return (n + kUbAlign - 1) / kUbAlign * kUbAlign; }

    struct Bf16UbFixpipe {
        AscendC::LocalTensor<ElementD> ub;
        uint32_t m;
        uint32_t n;
        bool dualWrite;

        template <class TensorL0C>
        CATLASS_DEVICE void operator()(TensorL0C const& source) const
        {
            uint32_t alignN = (n + kUbAlign - 1) / kUbAlign * kUbAlign;
            auto rowLayout = tla::MakeLayout<ElementD, layout::RowMajor>(m, alignN);
            auto rowDst = tla::MakeTensor(ub, rowLayout, Arch::PositionUB{});
            GemmSymmFixpipeUb{}(rowDst, source, false, static_cast<uint8_t>(0));
            if (dualWrite) {
                uint32_t alignM = (m + kUbAlign - 1) / kUbAlign * kUbAlign;
                auto colLayout = tla::MakeLayout<ElementD, layout::ColumnMajor>(alignM, n);
                auto colDst = tla::MakeTensor(ub, colLayout, Arch::PositionUB{});
                GemmSymmFixpipeUb{}(colDst, source, true, static_cast<uint8_t>(0));
            }
        }
    };

    template <class T, bool kScaleP, bool kScaleC>
    CATLASS_DEVICE_SIMD_VF void Bf16AxpbyVf(__ubuf__ T* p, __ubuf__ T* c, __ubuf__ T* d, uint32_t count, float alpha,
                                            float beta)
    {
        static constexpr AscendC::MicroAPI::CastTrait kUp = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait kDown = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr uint32_t kLanes = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        constexpr uint32_t kStep = kLanes * 2;
        uint16_t repeat = static_cast<uint16_t>(count / kStep);
        uint32_t done = static_cast<uint32_t>(repeat) * kStep;
        auto full = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < repeat; ++i) {
            uint32_t base = static_cast<uint32_t>(i) * kStep;
            AscendC::MicroAPI::RegTensor<T> pIn0;
            AscendC::MicroAPI::RegTensor<T> pIn1;
            AscendC::MicroAPI::RegTensor<T> cIn0;
            AscendC::MicroAPI::RegTensor<T> cIn1;
            AscendC::MicroAPI::RegTensor<float> p0;
            AscendC::MicroAPI::RegTensor<float> p1;
            AscendC::MicroAPI::RegTensor<float> c0;
            AscendC::MicroAPI::RegTensor<float> c1;
            AscendC::MicroAPI::RegTensor<T> d0;
            AscendC::MicroAPI::RegTensor<T> d1;
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(pIn0, p + base);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(pIn1, p + base + kLanes);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(cIn0, c + base);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(cIn1, c + base + kLanes);
            AscendC::MicroAPI::Cast<float, T, kUp>(p0, pIn0, full);
            AscendC::MicroAPI::Cast<float, T, kUp>(p1, pIn1, full);
            AscendC::MicroAPI::Cast<float, T, kUp>(c0, cIn0, full);
            AscendC::MicroAPI::Cast<float, T, kUp>(c1, cIn1, full);
            if constexpr (kScaleP) {
                AscendC::MicroAPI::Muls(p0, p0, alpha, full);
                AscendC::MicroAPI::Muls(p1, p1, alpha, full);
            }
            if constexpr (kScaleC) {
                AscendC::MicroAPI::Muls(c0, c0, beta, full);
                AscendC::MicroAPI::Muls(c1, c1, beta, full);
            }
            AscendC::MicroAPI::Add(p0, p0, c0, full);
            AscendC::MicroAPI::Add(p1, p1, c1, full);
            AscendC::MicroAPI::Cast<T, float, kDown>(d0, p0, full);
            AscendC::MicroAPI::Cast<T, float, kDown>(d1, p1, full);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(d + base, d0, full);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(d + base + kLanes, d1, full);
        }
        uint32_t remaining = count - done;
        uint16_t tail = static_cast<uint16_t>((remaining + kLanes - 1) / kLanes);
        for (uint16_t i = 0; i < tail; ++i) {
            auto mask = AscendC::MicroAPI::UpdateMask<float>(remaining);
            uint32_t base = done + static_cast<uint32_t>(i) * kLanes;
            AscendC::MicroAPI::RegTensor<T> pIn;
            AscendC::MicroAPI::RegTensor<T> cIn;
            AscendC::MicroAPI::RegTensor<float> pf;
            AscendC::MicroAPI::RegTensor<float> cf;
            AscendC::MicroAPI::RegTensor<T> stored;
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(pIn, p + base);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(cIn, c + base);
            AscendC::MicroAPI::Cast<float, T, kUp>(pf, pIn, mask);
            AscendC::MicroAPI::Cast<float, T, kUp>(cf, cIn, mask);
            if constexpr (kScaleP) {
                AscendC::MicroAPI::Muls(pf, pf, alpha, mask);
            }
            if constexpr (kScaleC) {
                AscendC::MicroAPI::Muls(cf, cf, beta, mask);
            }
            AscendC::MicroAPI::Add(pf, pf, cf, mask);
            AscendC::MicroAPI::Cast<T, float, kDown>(stored, pf, mask);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(d + base, stored, mask);
        }
    }

    template <class T>
    CATLASS_DEVICE_SIMD_VF void Bf16ScaleVf(__ubuf__ T* p, __ubuf__ T* d, uint32_t count, float alpha)
    {
        static constexpr AscendC::MicroAPI::CastTrait kUp = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait kDown = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr uint32_t kLanes = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        uint32_t remaining = count;
        uint16_t repeat = static_cast<uint16_t>((count + kLanes - 1) / kLanes);
        for (uint16_t i = 0; i < repeat; ++i) {
            auto mask = AscendC::MicroAPI::UpdateMask<float>(remaining);
            uint32_t base = static_cast<uint32_t>(i) * kLanes;
            AscendC::MicroAPI::RegTensor<T> in;
            AscendC::MicroAPI::RegTensor<float> widened;
            AscendC::MicroAPI::RegTensor<T> stored;
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(in, p + base);
            AscendC::MicroAPI::Cast<float, T, kUp>(widened, in, mask);
            AscendC::MicroAPI::Muls(widened, widened, alpha, mask);
            AscendC::MicroAPI::Cast<T, float, kDown>(stored, widened, mask);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(d + base, stored, mask);
        }
    }

    struct ResidualTile {
        bool active{false};
        uint32_t batchIdx{0};
        uint32_t gmRow{0};
        uint32_t gmCol{0};
        uint32_t rows{0};
        uint32_t cols{0};
    };

    template <class Scheduler>
    CATLASS_DEVICE ResidualTile DescribeResidualTile(Scheduler const& blockScheduler, uint32_t loopIdx,
                                                     uint32_t subBlockIdx)
    {
        ResidualTile tile;
        tile.batchIdx = blockScheduler.GetBatchIdx(loopIdx);
        GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
        GemmCoord actualBlockShape = blockScheduler.GetTaskShape(loopIdx);
        MatrixCoord origin = blockScheduler.GetElementOffset(loopIdx);
        tile.rows = subBlockIdx == 0 ? actualBlockShape.m() : actualBlockShape.n();
        tile.cols = subBlockIdx == 0 ? actualBlockShape.n() : actualBlockShape.m();
        tile.gmRow = subBlockIdx == 0 ? origin.row() : origin.column();
        tile.gmCol = subBlockIdx == 0 ? origin.column() : origin.row();
        tile.active = (subBlockIdx == 0 || blockCoord.m() > blockCoord.n()) && tile.rows > 0 && tile.cols > 0;
        return tile;
    }

    // A 2D DMA stride is in 32-byte blocks, so a padded row that is not a whole block is copied one row at a time.
    CATLASS_DEVICE void LoadResidualTile(Params const& params, AscendC::LocalTensor<ElementD> const& ubC,
                                         uint32_t batchIdx, uint32_t gmRow, uint32_t gmCol, uint32_t rows,
                                         uint32_t cols, uint32_t stride)
    {
        if (stride == cols) {
            LoadResidualRows(params, ubC, batchIdx, gmRow, gmCol, rows, cols);
        } else {
            for (uint32_t row = 0; row < rows; ++row) {
                LoadResidualRows(params, ubC[row * stride], batchIdx, gmRow + row, gmCol, 1, cols);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }

    CATLASS_DEVICE void LoadResidualRows(Params const& params, AscendC::LocalTensor<ElementD> const& ubC,
                                         uint32_t batchIdx, uint32_t gmRow, uint32_t gmCol, uint32_t rows,
                                         uint32_t cols)
    {
        int64_t batchOffset = static_cast<int64_t>(batchIdx) * params.strideY;
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer(reinterpret_cast<__gm__ ElementY*>(params.ptrY) + batchOffset);
        auto tensorC = tla::MakeTensor(gmY, params.layoutY, Arch::PositionGM{});
        auto tileC = tla::GetTile(tensorC, tla::MakeCoord(gmRow, gmCol), tla::MakeShape(rows, cols));
        auto layoutC = tla::MakeLayout<ElementD, layout::RowMajor>(rows, cols);
        auto tensorCUb = tla::MakeTensor(ubC, layoutC, Arch::PositionUB{});
        using CopyIn = Epilogue::Tile::CopyGm2UbTla<ArchTag, decltype(tileC), decltype(tensorCUb)>;
        CopyIn{}(tensorCUb, tileC);
    }

    CATLASS_DEVICE void StoreUbTile(Params const& params, AscendC::LocalTensor<ElementD> const& ub, uint32_t batchIdx,
                                    uint32_t gmRow, uint32_t gmCol, uint32_t rows, uint32_t cols, uint32_t stride)
    {
        if (stride == cols) {
            StoreUbRows(params, ub, batchIdx, gmRow, gmCol, rows, cols);
            return;
        }
        for (uint32_t row = 0; row < rows; ++row) {
            StoreUbRows(params, ub[row * stride], batchIdx, gmRow + row, gmCol, 1, cols);
        }
    }

    CATLASS_DEVICE void StoreUbRows(Params const& params, AscendC::LocalTensor<ElementD> const& ub, uint32_t batchIdx,
                                    uint32_t gmRow, uint32_t gmCol, uint32_t rows, uint32_t cols)
    {
        int64_t batchOffset = static_cast<int64_t>(batchIdx) * params.strideD;
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(params.ptrD) + batchOffset);
        auto tensorD = tla::MakeTensor(gmD, params.layoutD, Arch::PositionGM{});
        auto tileD = tla::GetTile(tensorD, tla::MakeCoord(gmRow, gmCol), tla::MakeShape(rows, cols));
        auto layoutD = tla::MakeLayout<ElementD, layout::RowMajor>(rows, cols);
        auto tensorDUb = tla::MakeTensor(ub, layoutD, Arch::PositionUB{});
        using CopyOut = Epilogue::Tile::CopyUb2GmTla<ArchTag, decltype(tensorDUb), decltype(tileD)>;
        CopyOut{}(tileD, tensorDUb);
    }

    // fp32 Fixpipe into UB stores the edge in 16-float units. Rounding only to 8
    // leaves widths such as 120, and that store scrambles the whole tile.
    CATLASS_DEVICE static uint32_t AlignFp32(uint32_t n) { return (n + 15u) / 16u * 16u; }

    struct Fp32UbFixpipe {
        AscendC::LocalTensor<ElementP> ub;
        uint32_t m;
        uint32_t n;
        bool dualWrite;

        template <class TensorL0C>
        CATLASS_DEVICE void operator()(TensorL0C const& source) const
        {
            uint32_t alignN = AlignFp32(n);
            auto rowLayout = tla::MakeLayout<ElementP, layout::RowMajor>(m, alignN);
            auto rowDst = tla::MakeTensor(ub, rowLayout, Arch::PositionUB{});
            GemmSymmFixpipeUb{}(rowDst, source, false, static_cast<uint8_t>(0));
            if (dualWrite) {
                uint32_t alignM = AlignFp32(m);
                auto colLayout = tla::MakeLayout<ElementP, layout::ColumnMajor>(alignM, n);
                auto colDst = tla::MakeTensor(ub, colLayout, Arch::PositionUB{});
                GemmSymmFixpipeUb{}(colDst, source, true, static_cast<uint8_t>(0));
            }
        }
    };

    template <class T, bool kScaleP, bool kScaleC>
    CATLASS_DEVICE_SIMD_VF void Fp32AxpbyVf(__ubuf__ float* p, __ubuf__ T* c, __ubuf__ T* d, uint32_t count,
                                            float alpha, float beta)
    {
        static constexpr AscendC::MicroAPI::CastTrait kUp = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        static constexpr AscendC::MicroAPI::CastTrait kDown = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr uint32_t kLanes = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        uint32_t remaining = count;
        uint16_t repeat = static_cast<uint16_t>((count + kLanes - 1) / kLanes);
        for (uint16_t i = 0; i < repeat; ++i) {
            auto mask = AscendC::MicroAPI::UpdateMask<float>(remaining);
            uint32_t base = static_cast<uint32_t>(i) * kLanes;
            AscendC::MicroAPI::RegTensor<float> pf;
            AscendC::MicroAPI::RegTensor<T> cIn;
            AscendC::MicroAPI::RegTensor<float> cf;
            AscendC::MicroAPI::RegTensor<T> stored;
            AscendC::MicroAPI::DataCopy(pf, p + base);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(cIn, c + base);
            AscendC::MicroAPI::Cast<float, T, kUp>(cf, cIn, mask);
            if constexpr (kScaleP) {
                AscendC::MicroAPI::Muls(pf, pf, alpha, mask);
            }
            if constexpr (kScaleC) {
                AscendC::MicroAPI::Muls(cf, cf, beta, mask);
            }
            AscendC::MicroAPI::Add(pf, pf, cf, mask);
            AscendC::MicroAPI::Cast<T, float, kDown>(stored, pf, mask);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(d + base, stored, mask);
        }
    }

    template <class T, bool kScaleP>
    CATLASS_DEVICE_SIMD_VF void Fp32CastVf(__ubuf__ float* p, __ubuf__ T* d, uint32_t count, float alpha)
    {
        static constexpr AscendC::MicroAPI::CastTrait kDown = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr uint32_t kLanes = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        uint32_t remaining = count;
        uint16_t repeat = static_cast<uint16_t>((count + kLanes - 1) / kLanes);
        for (uint16_t i = 0; i < repeat; ++i) {
            auto mask = AscendC::MicroAPI::UpdateMask<float>(remaining);
            uint32_t base = static_cast<uint32_t>(i) * kLanes;
            AscendC::MicroAPI::RegTensor<float> pf;
            AscendC::MicroAPI::RegTensor<T> stored;
            AscendC::MicroAPI::DataCopy(pf, p + base);
            if constexpr (kScaleP) {
                AscendC::MicroAPI::Muls(pf, pf, alpha, mask);
            }
            AscendC::MicroAPI::Cast<T, float, kDown>(stored, pf, mask);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(d + base, stored, mask);
        }
    }

    CATLASS_DEVICE void ApplyFp32Row(__ubuf__ float* p, __ubuf__ ElementD* c, uint32_t count, bool scaleP, bool scaleC,
                                     bool useC, float alpha, float beta)
    {
        if (useC) {
            if (scaleP && scaleC) {
                Fp32AxpbyVf<ElementD, true, true>(p, c, c, count, alpha, beta);
            } else if (scaleP) {
                Fp32AxpbyVf<ElementD, true, false>(p, c, c, count, alpha, beta);
            } else if (scaleC) {
                Fp32AxpbyVf<ElementD, false, true>(p, c, c, count, alpha, beta);
            } else {
                Fp32AxpbyVf<ElementD, false, false>(p, c, c, count, alpha, beta);
            }
        } else if (scaleP) {
            Fp32CastVf<ElementD, true>(p, c, count, alpha);
        } else {
            Fp32CastVf<ElementD, false>(p, c, count, alpha);
        }
    }

    CATLASS_DEVICE void ApplyFp32Rows(__ubuf__ float* p, __ubuf__ ElementD* c, uint32_t rows, uint32_t cols,
                                      uint32_t stride, bool scaleP, bool scaleC, bool useC, float alpha, float beta)
    {
        if (stride == cols) {
            ApplyFp32Row(p, c, rows * cols, scaleP, scaleC, useC, alpha, beta);
            return;
        }
        for (uint32_t row = 0; row < rows; ++row) {
            ApplyFp32Row(p + row * stride, c + row * stride, cols, scaleP, scaleC, useC, alpha, beta);
        }
    }

    CATLASS_DEVICE void ConsumeFp32Ub(Params const& params, Arch::Resource<ArchTag>& resource)
    {
        BlockScheduler blockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N), AscendC::GetBlockNum());
        uint32_t coreLoops = params.batchCount * blockScheduler.GetCoreLoops();
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t aicBlockIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicBlockNum = AscendC::GetBlockNum();

        auto ubP = resource.ubBuf.template GetBufferByByte<ElementP>(0);
        auto ubC = resource.ubBuf.template GetBufferByByte<ElementD>(FP32_UB_BYTES);
        const bool scaleP = params.alpha != 1.0f;
        const bool scaleC = params.beta != 1.0f;
        const bool useProduct = params.alpha != 0.0f;
        const bool useC = params.beta != 0.0f;
        // alpha == 0 has no cross-core flag, so this drains the previous tile's MTE3 before the UB is reused.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        for (uint32_t loopIdx = aicBlockIdx; loopIdx < coreLoops; loopIdx += aicBlockNum) {
            ResidualTile tile = DescribeResidualTile(blockScheduler, loopIdx, subBlockIdx);
            if (useProduct) {
                if (!useC && !scaleP) {
                    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_V>(AIC_SYNC_AIV_FLAG);
                } else if (!useC) {
                    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_V>(AIC_SYNC_AIV_FLAG);
                } else {
                    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE2>(AIC_SYNC_AIV_FLAG);
                }
            }
            if (tile.active) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                uint32_t stride = AlignFp32(tile.cols);
                auto pBase = reinterpret_cast<__ubuf__ float*>(ubP.GetPhyAddr());
                auto cBase = reinterpret_cast<__ubuf__ ElementD*>(ubC.GetPhyAddr());
                if (!useProduct) {
                    if (useC && !scaleC) {
                        LoadResidualTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols,
                                         stride);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                        StoreUbTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols, stride);
                    } else if (useC) {
                        LoadResidualTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols,
                                         stride);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                        if (stride == tile.cols) {
                            Bf16ScaleVf<ElementD>(cBase, cBase, tile.rows * tile.cols, params.beta);
                        } else {
                            for (uint32_t row = 0; row < tile.rows; ++row) {
                                Bf16ScaleVf<ElementD>(cBase + row * stride, cBase + row * stride, tile.cols,
                                                      params.beta);
                            }
                        }
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        StoreUbTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols, stride);
                    } else {
                        AscendC::Duplicate(ubC, static_cast<ElementD>(0), tile.rows * stride);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        StoreUbTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols, stride);
                    }
                } else {
                    if (useC) {
                        LoadResidualTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols,
                                         stride);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    }
                    ApplyFp32Rows(pBase, cBase, tile.rows, tile.cols, stride, scaleP, scaleC, useC, params.alpha,
                                  params.beta);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    StoreUbTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols, stride);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
            if (useProduct) {
                AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE void ConsumeBf16Ub(Params const& params, Arch::Resource<ArchTag>& resource)
    {
        BlockScheduler blockScheduler(params.problemShape, MakeCoord(L1_TILE_M, L1_TILE_N), AscendC::GetBlockNum());
        uint32_t coreLoops = params.batchCount * blockScheduler.GetCoreLoops();
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t aicBlockIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicBlockNum = AscendC::GetBlockNum();

        auto ubP = resource.ubBuf.template GetBufferByByte<ElementD>(0);
        auto ubC = resource.ubBuf.template GetBufferByByte<ElementD>(UB_TILE_BYTES);
        const bool scaleP = params.alpha != 1.0f;
        const bool scaleC = params.beta != 1.0f;
        const bool useProduct = params.alpha != 0.0f;
        const bool useC = params.beta != 0.0f;
        // alpha == 0 has no cross-core flag, so this drains the previous tile's MTE3 before the UB is reused.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        for (uint32_t loopIdx = aicBlockIdx; loopIdx < coreLoops; loopIdx += aicBlockNum) {
            ResidualTile tile = DescribeResidualTile(blockScheduler, loopIdx, subBlockIdx);
            if (useProduct) {
                // Wait on the pipe that reads the UB next.
                if (!useC && !scaleP) {
                    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
                } else if (!useC) {
                    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_V>(AIC_SYNC_AIV_FLAG);
                } else {
                    AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE2>(AIC_SYNC_AIV_FLAG);
                }
            }
            if (tile.active) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                uint32_t stride = AlignUb(tile.cols);
                uint32_t count = tile.rows * tile.cols;
                auto pBase = reinterpret_cast<__ubuf__ ElementD*>(ubP.GetPhyAddr());
                auto cBase = reinterpret_cast<__ubuf__ ElementD*>(ubC.GetPhyAddr());
                if (!useProduct) {
                    if (useC && !scaleC) {
                        LoadResidualTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols,
                                         stride);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                        StoreUbTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols, stride);
                    } else if (useC) {
                        LoadResidualTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols,
                                         stride);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                        if (stride == tile.cols) {
                            Bf16ScaleVf<ElementD>(cBase, cBase, count, params.beta);
                        } else {
                            for (uint32_t row = 0; row < tile.rows; ++row) {
                                Bf16ScaleVf<ElementD>(cBase + row * stride, cBase + row * stride, tile.cols,
                                                      params.beta);
                            }
                        }
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        StoreUbTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols, stride);
                    } else {
                        AscendC::Duplicate(ubC, static_cast<ElementD>(0), tile.rows * stride);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        StoreUbTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols, stride);
                    }
                } else if (!useC && !scaleP) {
                    StoreUbTile(params, ubP, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols, stride);
                } else if (!useC) {
                    if (stride == tile.cols) {
                        Bf16ScaleVf<ElementD>(pBase, cBase, count, params.alpha);
                    } else {
                        for (uint32_t row = 0; row < tile.rows; ++row) {
                            Bf16ScaleVf<ElementD>(pBase + row * stride, cBase + row * stride, tile.cols, params.alpha);
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    StoreUbTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols, stride);
                } else {
                    LoadResidualTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols, stride);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    if (stride == tile.cols) {
                        if (scaleP && scaleC) {
                            Bf16AxpbyVf<ElementD, true, true>(pBase, cBase, cBase, count, params.alpha, params.beta);
                        } else if (scaleP) {
                            Bf16AxpbyVf<ElementD, true, false>(pBase, cBase, cBase, count, params.alpha, params.beta);
                        } else if (scaleC) {
                            Bf16AxpbyVf<ElementD, false, true>(pBase, cBase, cBase, count, params.alpha, params.beta);
                        } else {
                            Bf16AxpbyVf<ElementD, false, false>(pBase, cBase, cBase, count, params.alpha, params.beta);
                        }
                    } else if (scaleP && scaleC) {
                        for (uint32_t row = 0; row < tile.rows; ++row) {
                            Bf16AxpbyVf<ElementD, true, true>(pBase + row * stride, cBase + row * stride,
                                                              cBase + row * stride, tile.cols, params.alpha,
                                                              params.beta);
                        }
                    } else if (scaleP) {
                        for (uint32_t row = 0; row < tile.rows; ++row) {
                            Bf16AxpbyVf<ElementD, true, false>(pBase + row * stride, cBase + row * stride,
                                                               cBase + row * stride, tile.cols, params.alpha,
                                                               params.beta);
                        }
                    } else if (scaleC) {
                        for (uint32_t row = 0; row < tile.rows; ++row) {
                            Bf16AxpbyVf<ElementD, false, true>(pBase + row * stride, cBase + row * stride,
                                                               cBase + row * stride, tile.cols, params.alpha,
                                                               params.beta);
                        }
                    } else {
                        for (uint32_t row = 0; row < tile.rows; ++row) {
                            Bf16AxpbyVf<ElementD, false, false>(pBase + row * stride, cBase + row * stride,
                                                                cBase + row * stride, tile.cols, params.alpha,
                                                                params.beta);
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    StoreUbTile(params, ubC, tile.batchIdx, tile.gmRow, tile.gmCol, tile.rows, tile.cols, stride);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
            if (useProduct) {
                AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE_2, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    static constexpr uint32_t DiagonalBandBytes() { return DIAG_BAND_ROWS * L1_TILE_N * sizeof(ElementP); }

    CATLASS_DEVICE static void WaitDiagonalEntered()
    {
        AscendC::CrossCoreWaitFlag<DIAG_SYNC_MODE, PIPE_FIX>(DIAG_ENTER);
        AscendC::CrossCoreWaitFlag<DIAG_SYNC_MODE, PIPE_FIX>(DIAG_ENTER + DIAG_SECOND_AIV);
    }

    CATLASS_DEVICE static void WaitDiagonalVector()
    {
        AscendC::CrossCoreWaitFlag<DIAG_SYNC_MODE, PIPE_FIX>(DIAG_RELEASE);
        AscendC::CrossCoreWaitFlag<DIAG_SYNC_MODE, PIPE_FIX>(DIAG_RELEASE + DIAG_SECOND_AIV);
    }

    struct DiagonalFixpipe {
        AscendC::LocalTensor<ElementP> normal;
        AscendC::LocalTensor<ElementP> transposed;
        uint32_t extent;

        template <class TensorL0C>
        CATLASS_DEVICE void operator()(TensorL0C const& source) const
        {
            // A previous zN fixpipe latches channel-split and the ND stride.
            set_channel_para(static_cast<uint64_t>(1) << 48);
            AscendC::SetFixpipeNz2ndFlag(1, 0, 0);
            uint32_t half = RoundUp(extent, 32u) / 2;
            uint32_t stride = RoundUp(extent, 32u / sizeof(ElementP));
            uint32_t bands = CeilDiv(half, DIAG_BAND_ROWS);
            for (uint32_t band = 0; band < bands; ++band) {
                WaitDiagonalVector();
                for (uint32_t sub = 0; sub < 2; ++sub) {
                    uint32_t first = sub * half + band * DIAG_BAND_ROWS;
                    if (first >= extent) {
                        continue;
                    }
                    uint32_t rows = min(DIAG_BAND_ROWS, min(half - band * DIAG_BAND_ROWS, extent - first));
                    auto normalLayout = tla::MakeLayout<ElementP, layout::RowMajor>(rows, stride);
                    auto normalDst = tla::MakeTensor(normal, normalLayout, Arch::PositionUB{});
                    auto normalSrc = tla::GetTile(source, tla::MakeCoord(first, 0), tla::MakeShape(rows, extent));
                    GemmSymmFixpipeUb{}(normalDst, normalSrc, static_cast<bool>(sub), 0);

                    auto transposeLayout = tla::MakeLayout<ElementP, layout::ColumnMajor>(stride, rows);
                    auto transposeDst = tla::MakeTensor(transposed, transposeLayout, Arch::PositionUB{});
                    auto transposeSrc = tla::GetTile(source, tla::MakeCoord(0, first), tla::MakeShape(extent, rows));
                    GemmSymmFixpipeUb{}(transposeDst, transposeSrc, static_cast<bool>(sub), 0);
                }
                AscendC::CrossCoreSetFlag<DIAG_SYNC_MODE, PIPE_FIX>(DIAG_READY);
                AscendC::CrossCoreSetFlag<DIAG_SYNC_MODE, PIPE_FIX>(DIAG_READY + DIAG_SECOND_AIV);
            }
        }
    };

    using DiagonalBits = uint32_t;
    using DiagonalIndex = int32_t;

    CATLASS_DEVICE_SIMD_VF void SelectDiagonalVf(__ubuf__ DiagonalBits* direct, __ubuf__ DiagonalBits* transposed,
                                                 uint16_t diagCol0, uint16_t rows, uint16_t extent, uint16_t stride)
    {
        constexpr uint16_t LANES = AscendC::VECTOR_REG_WIDTH / sizeof(DiagonalBits);
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t remaining = extent;
            for (uint16_t column = 0; column < extent; column += LANES) {
                AscendC::MicroAPI::RegTensor<DiagonalBits> fromP, fromPt, selected;
                AscendC::MicroAPI::RegTensor<DiagonalIndex> indices;
                auto valid = AscendC::MicroAPI::UpdateMask<DiagonalBits>(remaining);
                AscendC::MicroAPI::MaskReg upper;
                AscendC::MicroAPI::Arange<DiagonalIndex>(indices, static_cast<DiagonalIndex>(column));
                AscendC::MicroAPI::CompareScalar<DiagonalIndex, AscendC::CMPMODE::GT>(
                    upper, indices, static_cast<DiagonalIndex>(diagCol0 + row), valid);
                uint32_t offset = static_cast<uint32_t>(row) * stride + column;
                AscendC::MicroAPI::DataCopy(fromP, direct + offset);
                AscendC::MicroAPI::DataCopy(fromPt, transposed + offset);
                AscendC::MicroAPI::Select(selected, fromPt, fromP, upper);
                AscendC::MicroAPI::DataCopy(direct + offset, selected, valid);
            }
        }
    }

    CATLASS_DEVICE void ApplyDiagonalTile(Params const& params, Arch::Resource<ArchTag>& resource, uint32_t mStart,
                                          uint32_t nStart, uint32_t extent, int64_t batchOffsetY)
    {
        uint32_t sub = AscendC::GetSubBlockIdx();
        uint16_t release = DIAG_RELEASE + (sub == 0 ? 0 : DIAG_SECOND_AIV);
        uint16_t ready = DIAG_READY + (sub == 0 ? 0 : DIAG_SECOND_AIV);
        // Drain the previous MTE3 store before releasing this UB to the cube.
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::CrossCoreSetFlag<DIAG_SYNC_MODE, PIPE_MTE3>(DIAG_ENTER + (sub == 0 ? 0 : DIAG_SECOND_AIV));
        AscendC::CrossCoreSetFlag<DIAG_SYNC_MODE, PIPE_MTE3>(release);

        uint32_t half = RoundUp(extent, 32u) / 2;
        uint32_t stride = RoundUp(extent, 32u / sizeof(ElementP));
        uint32_t bands = CeilDiv(half, DIAG_BAND_ROWS);
        auto normal = resource.ubBuf.template GetBufferByByte<ElementP>(0);
        auto cFp32 = resource.ubBuf.template GetBufferByByte<ElementP>(DiagonalBandBytes());
        constexpr uint32_t kAux = 2 * DIAG_BAND_ROWS * L1_TILE_N * sizeof(ElementP);
        auto ubC = resource.ubBuf.template GetBufferByByte<ElementY>(kAux);
        auto ubD = resource.ubBuf.template GetBufferByByte<ElementD>(kAux +
                                                                     DIAG_BAND_ROWS * L1_TILE_N * sizeof(ElementY));

        AscendC::GlobalTensor<ElementY> gmC;
        AscendC::GlobalTensor<ElementD> gmD;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementY*>(params.ptrY) + batchOffsetY);
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(params.ptrD) + batchOffsetY);
        auto tensorC = tla::MakeTensor(gmC, params.layoutY, Arch::PositionGM{});
        auto tensorD = tla::MakeTensor(gmD, params.layoutD, Arch::PositionGM{});

        for (uint32_t band = 0; band < bands; ++band) {
            AscendC::CrossCoreWaitFlag<DIAG_SYNC_MODE, PIPE_V>(ready);
            uint32_t first = sub * half + band * DIAG_BAND_ROWS;
            if (first < extent) {
                uint32_t rows = min(DIAG_BAND_ROWS, min(half - band * DIAG_BAND_ROWS, extent - first));
                // Muls leaves the mask in counter mode, and the select then reads the wrong lanes.
                AscendC::SetMaskNorm();
                AscendC::ResetMask();
                SelectDiagonalVf(reinterpret_cast<__ubuf__ DiagonalBits*>(normal.GetPhyAddr()),
                                 reinterpret_cast<__ubuf__ DiagonalBits*>(cFp32.GetPhyAddr()),
                                 static_cast<uint16_t>(first), static_cast<uint16_t>(rows),
                                 static_cast<uint16_t>(extent), static_cast<uint16_t>(stride));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetMaskNorm();
                AscendC::ResetMask();

                uint32_t cAlign = RoundUp(extent, BYTE_PER_BLK / sizeof(ElementY));
                uint32_t dAlign = RoundUp(extent, BYTE_PER_BLK / sizeof(ElementD));
                if (params.beta != 0.0f) {
                    auto tileC = tla::GetTile(tensorC, tla::MakeCoord(mStart + first, nStart),
                                              tla::MakeShape(rows, extent));
                    auto layoutCUb = tla::MakeLayout<ElementY, layout::RowMajor>(rows, cAlign);
                    auto tensorCUb = tla::MakeTensor(ubC, layoutCUb, Arch::PositionUB{});
                    using CopyC = Epilogue::Tile::CopyGm2UbTla<ArchTag, decltype(tileC), decltype(tensorCUb)>;
                    CopyC{}(tensorCUb, tileC);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                }
                if (params.beta != 0.0f && stride == extent && cAlign == extent && dAlign == extent) {
                    Epilogue::Block::SyrkAxpbyFusedVf<ElementY, ElementD>(
                        reinterpret_cast<__ubuf__ float*>(normal.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ ElementY*>(ubC.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ ElementD*>(ubD.GetPhyAddr()), rows * extent, params.alpha,
                        params.beta);
                } else if (params.beta == 0.0f && stride == extent && dAlign == extent) {
                    Epilogue::Block::SyrkAxpbyNoCVf<ElementD>(reinterpret_cast<__ubuf__ float*>(normal.GetPhyAddr()),
                                                              reinterpret_cast<__ubuf__ ElementD*>(ubD.GetPhyAddr()),
                                                              rows * extent, params.alpha);
                } else {
                    for (uint32_t row = 0; row < rows; ++row) {
                        if (params.beta != 0.0f) {
                            Epilogue::Block::SyrkAxpbyFusedVf<ElementY, ElementD>(
                                reinterpret_cast<__ubuf__ float*>(normal.GetPhyAddr()) + row * stride,
                                reinterpret_cast<__ubuf__ ElementY*>(ubC.GetPhyAddr()) + row * cAlign,
                                reinterpret_cast<__ubuf__ ElementD*>(ubD.GetPhyAddr()) + row * dAlign, extent,
                                params.alpha, params.beta);
                        } else {
                            Epilogue::Block::SyrkAxpbyNoCVf<ElementD>(
                                reinterpret_cast<__ubuf__ float*>(normal.GetPhyAddr()) + row * stride,
                                reinterpret_cast<__ubuf__ ElementD*>(ubD.GetPhyAddr()) + row * dAlign, extent,
                                params.alpha);
                        }
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);

                auto tileD = tla::GetTile(tensorD, tla::MakeCoord(mStart + first, nStart),
                                          tla::MakeShape(rows, extent));
                auto layoutDUb = tla::MakeLayout<ElementD, layout::RowMajor>(rows, dAlign);
                auto tensorDUb = tla::MakeTensor(ubD, layoutDUb, Arch::PositionUB{});
                using CopyD = Epilogue::Tile::CopyUb2GmTla<ArchTag, decltype(tensorDUb), decltype(tileD)>;
                CopyD{}(tileD, tensorDUb);
            }
            AscendC::CrossCoreSetFlag<DIAG_SYNC_MODE, PIPE_MTE3>(release);
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_BASIC_SYRK_WORKSPACE_TLA_HPP
