/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_TILE_COPY_HPP
#define CATLASS_GEMM_TILE_TILE_COPY_HPP

#include <type_traits>

#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/ascend950/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/ascend950/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/ascend950/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/ascend950/copy_l1_to_l0b.hpp"
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Tile {

template <
    /// Tag indicating architecture
    class ArchTag, class ElementA_, class LayoutTagA_, class ElementB_, class LayoutTagB_, class ElementC_,
    class LayoutTagC_, class ElementBias = void, bool ReluEnable_ = false,
    ScaleGranularity DEQUANT_GRANULARITY_ = ScaleGranularity::NO_QUANT, class L0CCopyMode = CopyToGM>
struct PackedTileCopyTla {
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using LayoutTagA = LayoutTagA_;
    using LayoutTagB = LayoutTagB_;
    using LayoutTagC = LayoutTagC_;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA,
                                                                                 ElementB>::ElementAccumulator;
    static constexpr bool ReluEnable = ReluEnable_;
    static constexpr ScaleGranularity DEQUANT_GRANULARITY = DEQUANT_GRANULARITY_;

    static constexpr bool HAS_BIAS = !std::is_void_v<ElementBias>;
    static constexpr bool HAS_QUANT_TENSOR = (DEQUANT_GRANULARITY == ScaleGranularity::PER_CHANNEL);

    using LayoutTagL1A = typename helper::L1ATypeSelector<Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
    using LayoutTagL1B = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
    using LayoutTagL0A = typename helper::L0ALayoutSelector<ArchTag>::Layout;
    using LayoutTagL0B = layout::nZ;
    using LayoutTagL0C = layout::L0C;

    using LayoutA = detail::TagToLayout_t<ElementA, LayoutTagA>;
    using LayoutB = detail::TagToLayout_t<ElementB, LayoutTagB>;
    using LayoutC = detail::TagToLayout_t<ElementC_, LayoutTagC>;

    using LayoutL1A = detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL0A = detail::TagToLayout_t<ElementA, LayoutTagL0A>;
    using LayoutL0B = detail::TagToLayout_t<ElementB, LayoutTagL0B>;
    using LayoutL0C = typename detail::LayoutL0C;

    using TensorL1AVectorLayout = tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0>,
                                              AscendC::TPosition::A1>;
    using TensorL1ALayout = tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0, tla::_0>,
                                        AscendC::TPosition::A1>;

    using TensorL1A = std::conditional_t<tla::detail::isVector<LayoutTagA>::value, TensorL1AVectorLayout,
                                         TensorL1ALayout>;
    using TensorL1B = tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, tla::Coord<tla::_0, tla::_0>,
                                  AscendC::TPosition::A1>;
    using TensorL0A = tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, tla::Coord<tla::_0, tla::_0>,
                                  AscendC::TPosition::A2>;
    using TensorL0B = tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, tla::Coord<tla::_0, tla::_0>,
                                  AscendC::TPosition::B2>;
    using TensorL0C = tla::Tensor<AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, tla::Coord<tla::_0, tla::_0>,
                                  AscendC::TPosition::CO1>;
    using TensorL1Bias = std::conditional_t<
        HAS_BIAS,
        tla::Tensor<AscendC::LocalTensor<ElementBias>, detail::TagToLayout_t<ElementBias, layout::VectorLayout>,
                    tla::Coord<tla::_0>, AscendC::TPosition::A1>,
        EmptyClass>;
    using TensorL0Bias = tla::Tensor<AscendC::LocalTensor<ElementAccumulator>,
                                     detail::TagToLayout_t<ElementAccumulator, layout::VectorLayout>,
                                     tla::Coord<tla::_0>, AscendC::TPosition::C2>;
    using TensorL1Quant = std::conditional_t<
        HAS_QUANT_TENSOR,
        tla::Tensor<AscendC::LocalTensor<uint64_t>, detail::TagToLayout_t<uint64_t, layout::VectorLayout>,
                    tla::Coord<tla::_0>, AscendC::TPosition::A1>,
        EmptyClass>;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    template <class TensorA>
    using CopyGmToL1A = Gemm::Tile::TileCopyTla<ArchTag, TensorA, TensorL1A>;

    template <class TensorB>
    using CopyGmToL1B = Gemm::Tile::TileCopyTla<ArchTag, TensorB, TensorL1B>;

    template <class TensorBias>
    using CopyGmToL1Bias = std::conditional_t<HAS_BIAS, Gemm::Tile::TileCopyTla<ArchTag, TensorBias, TensorL1Bias>,
                                              EmptyClass>;

    template <class TensorQuant>
    using CopyGmToL1Scale = std::conditional_t<
        HAS_QUANT_TENSOR, Gemm::Tile::TileCopyTla<ArchTag, TensorQuant, TensorL1Quant>, EmptyClass>;

    using CopyL1ToL0A = Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
    using CopyL1ToBT = std::conditional_t<HAS_BIAS, Gemm::Tile::TileCopyTla<ArchTag, TensorL1Bias, TensorL0Bias>,
                                          EmptyClass>;

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
    template <class TensorC>
    using CopyL0CToDst = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>;
#endif
};

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_TILE_COPY_HPP
