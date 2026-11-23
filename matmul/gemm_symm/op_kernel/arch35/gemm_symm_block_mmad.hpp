/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_SYRK_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_SYRK_TLA_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/matrix.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#include <type_traits>

namespace Catlass::Gemm::Block {

// v1.5.0 CopyL0CToGmTla stores RowMajor and zN. The transposed workspace tile is ColumnMajor.
struct GemmSymmFixpipeGmCol {
    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const& dst, TensorSrc const& src) const
    {
        using ElementDst = typename TensorDst::Element;
        using ElementSrc = typename TensorSrc::Element;
        constexpr auto quantPre = Gemm::Tile::CopyL0CToDstQuantMode<Arch::Ascend950, ElementSrc, ElementDst,
                                                                    Gemm::Tile::ScaleGranularity::NO_QUANT>::VALUE;
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::COLUMN_MAJOR> params;
        params.nSize = tla::get<1>(dst.originShape());
        params.mSize = tla::get<0>(dst.originShape());
        params.srcStride = tla::get<1, 1>(src.stride()) / tla::get<0, 0>(src.stride());
        params.dstStride = tla::get<1>(dst.stride());
        params.params = AscendC::Nz2DnParams(1, 0, 0, 1);
        params.quantPre = quantPre;
        params.reluEn = false;
        params.unitFlag = 0;
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_COLUMN_MAJOR>(
            dst.data()[dst.layout()(dst.coord())], src.data()[src.layout()(src.coord())], params);
    }
};

template <class L1TileShape_, class L0TileShape_, class ElementX_, class ElementY_,
          class TileCopy_ = Gemm::Tile::PackedTileCopyTla<Arch::Ascend950, ElementX_, layout::RowMajor, ElementX_,
                                                          layout::ColumnMajor, ElementY_, layout::RowMajor>,
          bool MIRROR_DIAGONAL_ = false, uint32_t L0CStages_ = 1,
          class TileMmad_ = Gemm::Tile::TileMmadTla<Arch::Ascend950, ElementX_, typename TileCopy_::LayoutTagL1A>>
struct BlockMmadSyrkTla {
public:
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementX = ElementX_;
    using ElementXt = ElementX_;
    using ElementY = ElementY_;
    using TileCopy = TileCopy_;
    using TileMmad = TileMmad_;

    using LayoutTagX = typename TileCopy::LayoutTagA;
    using LayoutTagXt = typename TileCopy::LayoutTagB;
    using LayoutTagY = layout::RowMajor;

    static constexpr bool MIRROR_DIAGONAL = MIRROR_DIAGONAL_;

    static_assert((std::is_same_v<LayoutTagX, layout::RowMajor> || std::is_same_v<LayoutTagX, layout::ColumnMajor>) &&
                      (std::is_same_v<LayoutTagXt, layout::RowMajor> ||
                       std::is_same_v<LayoutTagXt, layout::ColumnMajor>),
                  "BlockMmadSyrkTla requires TileCopy LayoutTagA and LayoutTagB to be RowMajor or ColumnMajor");
    static_assert(std::is_same_v<typename TileCopy::LayoutTagC, LayoutTagY>,
                  "BlockMmadSyrkTla requires TileCopy LayoutTagC = RowMajor (Y)");

    using DispatchPolicy = Gemm::MmadPingpong<Arch::Ascend950, false>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static_assert(std::is_same_v<ArchTag, Arch::Ascend950>, "BlockMmadSyrkTla is Ascend950-only");
    static_assert(!DispatchPolicy::ENABLE_UNIT_FLAG, "BlockMmadSyrkTla cannot use unitFlag (dual-write)");
    static_assert(!DispatchPolicy::USE_HF32_MODE, "BlockMmadSyrkTla does not support HF32");
    static_assert(!DispatchPolicy::ENABLE_L1_RESIDENT, "BlockMmadSyrkTla does not support L1 resident");

    using LayoutX = typename TileCopy::LayoutA;
    using LayoutXt = typename TileCopy::LayoutB;
    using LayoutY = typename TileCopy::LayoutC;

    using ElementBias = void;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementX,
                                                                                 ElementXt>::ElementAccumulator;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
    using LayoutTagL1X = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1Xt = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static_assert(tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
                  "L1TileShape must be tla::tuple and static!");
    static_assert(tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
                  "L0TileShape must be tla::tuple and static!");

    static constexpr uint32_t L1X_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1XT_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = L0CStages_;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    static constexpr uint32_t L1X_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementX);
    static constexpr uint32_t L1XT_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementXt);
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementX);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementXt);
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    static_assert(L0C_STAGES == 1 || L0C_STAGES == 2, "BlockMmadSyrkTla supports one or two L0C buffers");
    static_assert(tla::detail::isRowMajor<LayoutY>::value,
                  "BlockMmadSyrkTla requires LayoutY = RowMajor for the nz2nd store");
    static_assert(L1X_TILE_SIZE * L1X_STAGES + L1XT_TILE_SIZE * L1XT_STAGES <= ArchTag::L1_SIZE,
                  "L1TileShape exceeding the L1 space!");
    static_assert(L0A_TILE_SIZE * L0A_STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * L0B_STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");
    static_assert(L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
                  "BlockMmadSyrkTla requires L1 and L0 tile M/N equal (no m/n L0 loop)");
    static_assert(L0_TILE_K <= L1_TILE_K, "L0TileShape::K cannot exceed L1TileShape::K");
    static_assert((L1X_STAGES + L1XT_STAGES) <= 8, "L1 Buffer overflow: Exceeds the supported range of EVENT(0~7)");
    static_assert((L0A_STAGES + L0B_STAGES) <= 8, "L0 Buffer overflow: Exceeds the supported range of EVENT_ID(0~7)");

    static constexpr auto L1X_LAYOUT = tla::MakeLayout<ElementX, LayoutTagL1X>(tla::Int<L1_TILE_M>{},
                                                                               tla::Int<L1_TILE_K>{});
    static constexpr auto L1XT_LAYOUT = tla::MakeLayout<ElementXt, LayoutTagL1Xt>(tla::Int<L1_TILE_K>{},
                                                                                  tla::Int<L1_TILE_N>{});

    CATLASS_DEVICE
    BlockMmadSyrkTla() {}

    CATLASS_DEVICE
    BlockMmadSyrkTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0)
    {
        if ASCEND_IS_AIC {
            AscendC::SetHF32Mode(false);

            uint32_t l1XOffset = l1BufAddrStart;
            uint32_t l1XtOffset = l1BufAddrStart + L1X_TILE_SIZE * L1X_STAGES;

            for (uint32_t i = 0; i < L1X_STAGES; i++) {
                l1XTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementX>(l1XOffset + L1X_TILE_SIZE * i);
                l1XEventList[i] = static_cast<int32_t>(i);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[i]);
            }
            for (uint32_t i = 0; i < L1XT_STAGES; i++) {
                l1XtTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementXt>(l1XtOffset + L1XT_TILE_SIZE * i);
                l1XtEventList[i] = static_cast<int32_t>(i + L1X_STAGES);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1XtEventList[i]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementX>(L0A_TILE_SIZE * i);
                l0AEventList[i] = static_cast<int32_t>(i);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementXt>(L0B_TILE_SIZE * i);
                l0BEventList[i] = static_cast<int32_t>(i + L0A_STAGES);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            for (uint32_t i = 0; i < L0C_STAGES; i++) {
                l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
                l0CEventList[i] = static_cast<int32_t>(i);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
            }
        }
    }

    CATLASS_DEVICE
    ~BlockMmadSyrkTla()
    {
        if ASCEND_IS_AIC {
            for (uint32_t i = 0; i < L1X_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[i]);
            }
            for (uint32_t i = 0; i < L1XT_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XtEventList[i]);
            }
            for (uint32_t i = 0; i < L0A_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            for (uint32_t i = 0; i < L0B_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            for (uint32_t i = 0; i < L0C_STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
            }
        }
    }

    template <class TensorX, class TensorXt, class TensorY, class TensorYT, class DiagonalStore = void>
    CATLASS_DEVICE void operator()(TensorX& tensorX, TensorXt& tensorXt, TensorY& tensorY, TensorYT& tensorYT,
                                   GemmCoord const& actualShape, GemmCoord const& blockCoord,
                                   DiagonalStore* diagonalStore = nullptr)
    {
        Compute(tensorX, tensorXt, actualShape);
        CopyOut(tensorY, tensorYT, actualShape, blockCoord, diagonalStore);
    }

    template <class TensorX, class TensorXt>
    CATLASS_DEVICE void Compute(TensorX& tensorX, TensorXt& tensorXt, GemmCoord const& actualShape)
    {
        using CopyGmToL1X = typename TileCopy::template CopyGmToL1A<TensorX>;
        using CopyGmToL1Xt = typename TileCopy::template CopyGmToL1B<TensorXt>;
        CopyGmToL1X copyGmToL1X;
        CopyGmToL1Xt copyGmToL1Xt;

        uint32_t mActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nActual = actualShape.n();

        uint32_t kL1Actual = min(kBlockActual, L1_TILE_K);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[l1XListId]);
        auto tensorL1X = tla::MakeTensor(l1XTensorList[l1XListId], L1X_LAYOUT, Arch::PositionL1{});
        auto tensorTileX = GetTile(tensorX, tla::MakeCoord(0, 0), tla::MakeShape(mActual, kL1Actual));
        copyGmToL1X(tensorL1X, tensorTileX);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1XEventList[l1XListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XtEventList[l1XtListId]);
        auto tensorL1Xt = tla::MakeTensor(l1XtTensorList[l1XtListId], L1XT_LAYOUT, Arch::PositionL1{});
        auto tensorTileXt = GetTile(tensorXt, tla::MakeCoord(0, 0), tla::MakeShape(kL1Actual, nActual));
        copyGmToL1Xt(tensorL1Xt, tensorTileXt);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1XtEventList[l1XtListId]);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
        auto layoutInL0C = tla::MakeLayoutL0C(mActual, nActual);
        auto tensorL0C = tla::MakeTensor(l0CTensorList[l0CListId], layoutInL0C, Arch::PositionL0C{});

        uint32_t kL1Loop = CeilDiv<L1_TILE_K>(kBlockActual);
        for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
            uint32_t l1XListIdNext = (l1XListId + 1 < L1X_STAGES) ? (l1XListId + 1) : 0;
            uint32_t l1XtListIdNext = (l1XtListId + 1 < L1XT_STAGES) ? (l1XtListId + 1) : 0;
            uint32_t kL1ActualNext{0};
            if (kL1Idx < kL1Loop - 1) {
                uint32_t kL1IdxNext = kL1Idx + 1;
                kL1ActualNext = (kL1IdxNext < kL1Loop - 1) ? L1_TILE_K : (kBlockActual - kL1IdxNext * L1_TILE_K);

                auto tensorL1XNext = tla::MakeTensor(l1XTensorList[l1XListIdNext], L1X_LAYOUT, Arch::PositionL1{});
                auto tensorL1XtNext = tla::MakeTensor(l1XtTensorList[l1XtListIdNext], L1XT_LAYOUT, Arch::PositionL1{});
                auto tensorTileXNext = GetTile(tensorX, tla::MakeCoord(0, kL1IdxNext * L1_TILE_K),
                                               tla::MakeShape(mActual, kL1ActualNext));
                auto tensorTileXtNext = GetTile(tensorXt, tla::MakeCoord(kL1IdxNext * L1_TILE_K, 0),
                                                tla::MakeShape(kL1ActualNext, nActual));

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[l1XListIdNext]);
                copyGmToL1X(tensorL1XNext, tensorTileXNext);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1XEventList[l1XListIdNext]);

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1XtEventList[l1XtListIdNext]);
                copyGmToL1Xt(tensorL1XtNext, tensorTileXtNext);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1XtEventList[l1XtListIdNext]);
            }

            tensorL1X = tla::MakeTensor(l1XTensorList[l1XListId], L1X_LAYOUT, Arch::PositionL1{});
            tensorL1Xt = tla::MakeTensor(l1XtTensorList[l1XtListId], L1XT_LAYOUT, Arch::PositionL1{});
            uint32_t kL0Loop = CeilDiv<L0_TILE_K>(kL1Actual);

            for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0_TILE_K : (kL1Actual - kL0Idx * L0_TILE_K);

                auto layoutAInL0 = tla::MakeLayout<ElementX, LayoutTagL0A>(mActual, kL0Actual);
                auto tensorL0A = tla::MakeTensor(l0ATensorList[l0AListId], layoutAInL0, Arch::PositionL0A{});
                auto tensorTileL1X = GetTile(tensorL1X, tla::MakeCoord(0, kL0Idx * L0_TILE_K),
                                             tla::MakeShape(mActual, kL0Actual));

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                if (kL0Idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1XEventList[l1XListId]);
                }
                copyL1ToL0A(tensorL0A, tensorTileL1X);
                if (kL0Idx == kL0Loop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1XEventList[l1XListId]);
                }

                auto layoutBInL0 = tla::MakeLayout<ElementXt, LayoutTagL0B>(kL0Actual, nActual);
                auto tensorL0B = tla::MakeTensor(l0BTensorList[l0BListId], layoutBInL0, Arch::PositionL0B{});
                auto tensorTileL1Xt = GetTile(tensorL1Xt, tla::MakeCoord(kL0Idx * L0_TILE_K, 0),
                                              tla::MakeShape(kL0Actual, nActual));

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                if (kL0Idx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1XtEventList[l1XtListId]);
                }
                copyL1ToL0B(tensorL0B, tensorTileL1Xt);
                if (kL0Idx == kL0Loop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1XtEventList[l1XtListId]);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[0]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[0]);

                bool initC = ((kL1Idx == 0) && (kL0Idx == 0));
                tileMmad(tensorL0C, tensorL0A, tensorL0B, mActual, nActual, kL0Actual, initC);

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
            }
            l1XListId = l1XListIdNext;
            l1XtListId = l1XtListIdNext;
            kL1Actual = kL1ActualNext;
        }

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
    }

    template <class TensorY, class TensorYT, class DiagonalStore = void>
    CATLASS_DEVICE void CopyOut(TensorY& tensorY, TensorYT& tensorYT, GemmCoord const& actualShape,
                                GemmCoord const& blockCoord, DiagonalStore* diagonalStore = nullptr)
    {
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorY>;
        CopyL0CToDst copyL0CToDst;
        uint32_t mActual = actualShape.m();
        uint32_t nActual = actualShape.n();
        auto layoutInL0C = tla::MakeLayoutL0C(mActual, nActual);
        auto tensorL0C = tla::MakeTensor(l0CTensorList[l0CListId], layoutInL0C, Arch::PositionL0C{});
        if constexpr (TensorY::position == AscendC::TPosition::VECCALC) {
            copyL0CToDst(tensorY, tensorL0C);
        } else {
            if constexpr (!std::is_void_v<DiagonalStore>) {
                (*diagonalStore)(tensorL0C);
            } else if (blockCoord.m() == blockCoord.n() && !MIRROR_DIAGONAL) {
                copyL0CToDst(tensorY, tensorL0C);
            } else {
                copyL0CToDst(tensorY, tensorL0C);
                if constexpr (tla::detail::isColumnMajor<typename TensorYT::Layout>::value) {
                    GemmSymmFixpipeGmCol{}(tensorYT, tensorL0C);
                } else {
                    using CopyL0CToTransposeDst = typename TileCopy::template CopyL0CToDst<TensorYT>;
                    CopyL0CToTransposeDst{}(tensorYT, tensorL0C);
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
        l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
    }

protected:
    AscendC::LocalTensor<ElementX> l1XTensorList[L1X_STAGES];
    AscendC::LocalTensor<ElementXt> l1XtTensorList[L1XT_STAGES];
    AscendC::LocalTensor<ElementX> l0ATensorList[L0A_STAGES];
    AscendC::LocalTensor<ElementXt> l0BTensorList[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];

    int32_t l1XEventList[L1X_STAGES];
    int32_t l1XtEventList[L1XT_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    int32_t l0CEventList[L0C_STAGES];

    uint32_t l1XListId{0};
    uint32_t l1XtListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    uint32_t l0CListId{0};

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_SYRK_TLA_HPP
