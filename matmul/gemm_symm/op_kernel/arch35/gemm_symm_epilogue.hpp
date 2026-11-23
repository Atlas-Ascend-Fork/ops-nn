/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SYRK_AXPBY_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SYRK_AXPBY_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub_tla.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm_tla.hpp"
#include "catlass/layout/matrix.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

#ifndef CATLASS_DEVICE_SIMD_VF
#define CATLASS_DEVICE_SIMD_VF __simd_vf__ static
#endif

namespace Catlass::Epilogue::Block {

template <class ElementY, class ElementD>
CATLASS_DEVICE_SIMD_VF void SyrkAxpbyFusedVf(__ubuf__ float* p, __ubuf__ ElementY* c, __ubuf__ ElementD* d,
                                             uint32_t count, float alpha, float beta)
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
        AscendC::MicroAPI::RegTensor<float> p0;
        AscendC::MicroAPI::RegTensor<float> p1;
        AscendC::MicroAPI::RegTensor<ElementY> c0;
        AscendC::MicroAPI::RegTensor<ElementY> c1;
        AscendC::MicroAPI::RegTensor<float> f0;
        AscendC::MicroAPI::RegTensor<float> f1;
        AscendC::MicroAPI::RegTensor<ElementD> d0;
        AscendC::MicroAPI::RegTensor<ElementD> d1;
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(p0, p + base);
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(p1, p + base + kLanes);
        AscendC::MicroAPI::DataCopy<ElementY, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(c0, c + base);
        AscendC::MicroAPI::DataCopy<ElementY, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(c1, c + base + kLanes);
        AscendC::MicroAPI::Muls(p0, p0, alpha, full);
        AscendC::MicroAPI::Muls(p1, p1, alpha, full);
        AscendC::MicroAPI::Cast<float, ElementY, kUp>(f0, c0, full);
        AscendC::MicroAPI::Cast<float, ElementY, kUp>(f1, c1, full);
        AscendC::MicroAPI::Muls(f0, f0, beta, full);
        AscendC::MicroAPI::Muls(f1, f1, beta, full);
        AscendC::MicroAPI::Add(p0, p0, f0, full);
        AscendC::MicroAPI::Add(p1, p1, f1, full);
        AscendC::MicroAPI::Cast<ElementD, float, kDown>(d0, p0, full);
        AscendC::MicroAPI::Cast<ElementD, float, kDown>(d1, p1, full);
        AscendC::MicroAPI::DataCopy<ElementD, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(d + base, d0, full);
        AscendC::MicroAPI::DataCopy<ElementD, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(d + base + kLanes, d1, full);
    }

    uint32_t remaining = count - done;
    uint16_t tail = static_cast<uint16_t>((remaining + kLanes - 1) / kLanes);
    for (uint16_t i = 0; i < tail; ++i) {
        auto mask = AscendC::MicroAPI::UpdateMask<float>(remaining);
        uint32_t base = done + static_cast<uint32_t>(i) * kLanes;
        AscendC::MicroAPI::RegTensor<float> product;
        AscendC::MicroAPI::RegTensor<ElementY> residualIn;
        AscendC::MicroAPI::RegTensor<float> residual;
        AscendC::MicroAPI::RegTensor<ElementD> stored;
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(product, p + base);
        AscendC::MicroAPI::DataCopy<ElementY, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(residualIn, c + base);
        AscendC::MicroAPI::Muls(product, product, alpha, mask);
        AscendC::MicroAPI::Cast<float, ElementY, kUp>(residual, residualIn, mask);
        AscendC::MicroAPI::Muls(residual, residual, beta, mask);
        AscendC::MicroAPI::Add(product, product, residual, mask);
        AscendC::MicroAPI::Cast<ElementD, float, kDown>(stored, product, mask);
        AscendC::MicroAPI::DataCopy<ElementD, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(d + base, stored, mask);
    }
}

template <class ElementD>
CATLASS_DEVICE_SIMD_VF void SyrkAxpbyNoCVf(__ubuf__ float* p, __ubuf__ ElementD* d, uint32_t count, float alpha)
{
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
        AscendC::MicroAPI::RegTensor<float> p0;
        AscendC::MicroAPI::RegTensor<float> p1;
        AscendC::MicroAPI::RegTensor<ElementD> d0;
        AscendC::MicroAPI::RegTensor<ElementD> d1;
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(p0, p + base);
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(p1, p + base + kLanes);
        AscendC::MicroAPI::Muls(p0, p0, alpha, full);
        AscendC::MicroAPI::Muls(p1, p1, alpha, full);
        AscendC::MicroAPI::Cast<ElementD, float, kDown>(d0, p0, full);
        AscendC::MicroAPI::Cast<ElementD, float, kDown>(d1, p1, full);
        AscendC::MicroAPI::DataCopy<ElementD, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(d + base, d0, full);
        AscendC::MicroAPI::DataCopy<ElementD, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(d + base + kLanes, d1, full);
    }

    uint32_t remaining = count - done;
    uint16_t tail = static_cast<uint16_t>((remaining + kLanes - 1) / kLanes);
    for (uint16_t i = 0; i < tail; ++i) {
        auto mask = AscendC::MicroAPI::UpdateMask<float>(remaining);
        uint32_t base = done + static_cast<uint32_t>(i) * kLanes;
        AscendC::MicroAPI::RegTensor<float> product;
        AscendC::MicroAPI::RegTensor<ElementD> stored;
        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(product, p + base);
        AscendC::MicroAPI::Muls(product, product, alpha, mask);
        AscendC::MicroAPI::Cast<ElementD, float, kDown>(stored, product, mask);
        AscendC::MicroAPI::DataCopy<ElementD, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(d + base, stored, mask);
    }
}

template <class ArchTag_, class ElementY_, class ElementD_ = ElementY_>
class BlockEpilogueSyrkAxpby {
public:
    using ArchTag = ArchTag_;
    using ElementY = ElementY_;
    using ElementD = ElementD_;
    using ElementCompute = float;

    static constexpr bool Y_NEEDS_CAST = !std::is_same_v<ElementY, ElementCompute>;
    static constexpr bool D_NEEDS_CAST = !std::is_same_v<ElementD, ElementCompute>;

    static constexpr uint32_t MAX_COLS = 256;
    static constexpr uint32_t COMPUTE_ROWS = 64;
    static constexpr uint32_t STAGES = 2;
    // A UB row always starts on a 32-byte boundary, so rows are padded in units of the
    // smallest element held here. Padding in float units instead would start a 16-bit row
    // halfway through those 32 bytes.
    static constexpr size_t MIN_ELEMENT_BYTES_YD = (sizeof(ElementY) < sizeof(ElementD)) ? sizeof(ElementY) :
                                                                                           sizeof(ElementD);
    static constexpr size_t MIN_ELEMENT_BYTES = (MIN_ELEMENT_BYTES_YD < sizeof(ElementCompute)) ?
                                                    MIN_ELEMENT_BYTES_YD :
                                                    sizeof(ElementCompute);
    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / MIN_ELEMENT_BYTES;
    static constexpr uint32_t ALIGN_COLS = RoundUp(MAX_COLS, ELE_NUM_PER_BLK);
    static constexpr uint32_t CHUNK_ELEMS = COMPUTE_ROWS * ALIGN_COLS;

    using LayoutY = detail::TagToLayout_t<ElementY, layout::RowMajor>;
    using LayoutD = detail::TagToLayout_t<ElementD, layout::RowMajor>;
    using LayoutP = detail::TagToLayout_t<ElementCompute, layout::RowMajor>;

    enum class SymmStoreMode : uint8_t {
        FULL = 0,
        LOWER = 1,
        UPPER = 2,
    };

    static constexpr uint32_t STAGE_BYTES = CHUNK_ELEMS * sizeof(ElementCompute) + CHUNK_ELEMS * sizeof(ElementD);
    static constexpr uint32_t UB_BYTES_USED = STAGES * STAGE_BYTES;
    static_assert(sizeof(ElementY) == sizeof(ElementD), "C and D must be the same size to share one UB buffer");
    static_assert(UB_BYTES_USED <= ArchTag::UB_SIZE, "BlockEpilogueSyrkAxpby exceeds UB");

    struct Params {
        ElementCompute alpha{1.0f};
        ElementCompute beta{0.0f};
        GM_ADDR ptrY{nullptr};
        GM_ADDR ptrD{nullptr};
        LayoutY layoutY{};
        LayoutD layoutD{};

        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(ElementCompute alpha_, ElementCompute beta_, GM_ADDR ptrY_, GM_ADDR ptrD_, LayoutY layoutY_,
               LayoutD layoutD_)
            : alpha(alpha_), beta(beta_), ptrY(ptrY_), ptrD(ptrD_), layoutY(layoutY_), layoutD(layoutD_)
        {}
    };

    CATLASS_DEVICE
    BlockEpilogueSyrkAxpby() = default;

    CATLASS_DEVICE
    BlockEpilogueSyrkAxpby(Arch::Resource<ArchTag>& resource, Params const& params) : params_(params)
    {
        uint32_t offset = 0;
        for (uint32_t stage = 0; stage < STAGES; ++stage) {
            ubPStage_[stage] = resource.ubBuf.template GetBufferByByte<ElementCompute>(offset);
            offset += CHUNK_ELEMS * sizeof(ElementCompute);
            ubYStage_[stage] = resource.ubBuf.template GetBufferByByte<ElementY>(offset);
            ubDStage_[stage] = resource.ubBuf.template GetBufferByByte<ElementD>(offset);
            offset += CHUNK_ELEMS * sizeof(ElementD);
        }
        ubP_ = ubPStage_[0];
        ubY_ = ubYStage_[0];
        ubD_ = ubDStage_[0];
        if constexpr (D_NEEDS_CAST) {
            ubSaved_ = ubDStage_[1];
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    CATLASS_DEVICE
    ~BlockEpilogueSyrkAxpby()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    template <class TensorP>
    CATLASS_DEVICE void operator()(TensorP const& tensorP, uint32_t gmRow, uint32_t gmCol, uint32_t rows, uint32_t cols,
                                   int64_t batchOffsetElems = 0, SymmStoreMode mode = SymmStoreMode::FULL)
    {
        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer(reinterpret_cast<__gm__ ElementY*>(params_.ptrY) + batchOffsetElems);
        auto tensorY = tla::MakeTensor(gmY, params_.layoutY, Arch::PositionGM{});
        (*this)(tensorP, tensorY, gmRow, gmCol, rows, cols, batchOffsetElems, mode);
    }

    template <class TensorP, class TensorC>
    CATLASS_DEVICE void operator()(TensorP const& tensorP, TensorC const& tensorC, uint32_t gmRow, uint32_t gmCol,
                                   uint32_t rows, uint32_t cols, int64_t batchOffsetElems, SymmStoreMode mode)
    {
        if (rows == 0 || cols == 0) {
            return;
        }

        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(params_.ptrD) + batchOffsetElems);
        auto tensorD = tla::MakeTensor(gmD, params_.layoutD, Arch::PositionGM{});

        if (mode != SymmStoreMode::FULL) {
            return;
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        if constexpr (Y_NEEDS_CAST && D_NEEDS_CAST) {
            uint32_t alignCols = RoundUp(cols, ELE_NUM_PER_BLK);
            uint32_t maxRowsPerChunk = CHUNK_ELEMS / alignCols;
            if (maxRowsPerChunk == 0) {
                maxRowsPerChunk = 1;
            }
            uint32_t chunkCount = (rows + maxRowsPerChunk - 1) / maxRowsPerChunk;
            IssueLoad(tensorP, tensorC, gmRow, gmCol, cols, alignCols, maxRowsPerChunk, rows, 0);
            for (uint32_t index = 0; index < chunkCount; ++index) {
                if (index + 1 < chunkCount) {
                    IssueLoad(tensorP, tensorC, gmRow, gmCol, cols, alignCols, maxRowsPerChunk, rows, index + 1);
                }
                Consume(tensorD, gmRow, gmCol, cols, alignCols, maxRowsPerChunk, rows, index);
            }
        }
    }

    template <class TensorP, class TensorC>
    CATLASS_DEVICE void IssueLoad(TensorP const& tensorP, TensorC const& tensorC, uint32_t gmRow, uint32_t gmCol,
                                  uint32_t cols, uint32_t alignCols, uint32_t maxRowsPerChunk, uint32_t rows,
                                  uint32_t index)
    {
        uint32_t stage = index % STAGES;
        auto event = stage == 0 ? EVENT_ID0 : EVENT_ID1;
        uint32_t r = index * maxRowsPerChunk;
        uint32_t chunkRows = (rows - r) < maxRowsPerChunk ? (rows - r) : maxRowsPerChunk;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        auto tileP = GetTile(tensorP, tla::MakeCoord(r, 0), tla::MakeShape(chunkRows, cols));
        auto layoutPUb = tla::MakeLayout(tla::MakeShape(chunkRows, cols),
                                         tla::MakeStride(static_cast<int64_t>(alignCols), tla::Int<1>{}));
        auto tensorPUb = tla::MakeTensor(ubPStage_[stage], layoutPUb, Arch::PositionUB{});
        using CopyGmToUbP = Tile::CopyGm2UbTla<ArchTag, decltype(tileP), decltype(tensorPUb)>;
        CopyGmToUbP{}(tensorPUb, tileP);
        if (params_.beta != static_cast<ElementCompute>(0)) {
            auto tileC = GetTile(tensorC, tla::MakeCoord(gmRow + r, gmCol), tla::MakeShape(chunkRows, cols));
            auto layoutYUb = tla::MakeLayout<ElementY, layout::RowMajor>(chunkRows, alignCols);
            auto tensorYUb = tla::MakeTensor(ubYStage_[stage], layoutYUb, Arch::PositionUB{});
            using CopyGmToUbY = Tile::CopyGm2UbTla<ArchTag, decltype(tileC), decltype(tensorYUb)>;
            CopyGmToUbY{}(tensorYUb, tileC);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    template <class TensorD>
    CATLASS_DEVICE void Consume(TensorD const& tensorD, uint32_t gmRow, uint32_t gmCol, uint32_t cols,
                                uint32_t alignCols, uint32_t maxRowsPerChunk, uint32_t rows, uint32_t index)
    {
        uint32_t stage = index % STAGES;
        auto event = stage == 0 ? EVENT_ID0 : EVENT_ID1;
        uint32_t r = index * maxRowsPerChunk;
        uint32_t chunkRows = (rows - r) < maxRowsPerChunk ? (rows - r) : maxRowsPerChunk;
        uint32_t chunkElems = chunkRows * alignCols;

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        if (params_.beta != static_cast<ElementCompute>(0)) {
            SyrkAxpbyFusedVf<ElementY, ElementD>(reinterpret_cast<__ubuf__ float*>(ubPStage_[stage].GetPhyAddr()),
                                                 reinterpret_cast<__ubuf__ ElementY*>(ubYStage_[stage].GetPhyAddr()),
                                                 reinterpret_cast<__ubuf__ ElementD*>(ubDStage_[stage].GetPhyAddr()),
                                                 chunkElems, params_.alpha, params_.beta);
        } else {
            SyrkAxpbyNoCVf<ElementD>(reinterpret_cast<__ubuf__ float*>(ubPStage_[stage].GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ ElementD*>(ubDStage_[stage].GetPhyAddr()), chunkElems,
                                     params_.alpha);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

        auto tileD = GetTile(tensorD, tla::MakeCoord(gmRow + r, gmCol), tla::MakeShape(chunkRows, cols));
        auto layoutDUb = tla::MakeLayout<ElementD, layout::RowMajor>(chunkRows, alignCols);
        auto tensorDUb = tla::MakeTensor(ubDStage_[stage], layoutDUb, Arch::PositionUB{});
        using CopyUbToGmD = Tile::CopyUb2GmTla<ArchTag, decltype(tensorDUb), decltype(tileD)>;
        CopyUbToGmD{}(tileD, tensorDUb);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }

    template <class TensorP, class TensorPT>
    CATLASS_DEVICE void StoreDiagonal(TensorP const& tensorP, TensorPT const& tensorPT, uint32_t gmRow, uint32_t gmCol,
                                      uint32_t rows, uint32_t cols, int64_t batchOffsetElems)
    {
        if (rows == 0 || cols == 0) {
            return;
        }

        AscendC::GlobalTensor<ElementY> gmY;
        gmY.SetGlobalBuffer(reinterpret_cast<__gm__ ElementY*>(params_.ptrY) + batchOffsetElems);
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(params_.ptrD) + batchOffsetElems);
        auto tensorC = tla::MakeTensor(gmY, params_.layoutY, Arch::PositionGM{});
        auto tensorD = tla::MakeTensor(gmD, params_.layoutD, Arch::PositionGM{});

        uint32_t alignCols = RoundUp(cols, ELE_NUM_PER_BLK);
        uint32_t maxRowsPerChunk = CHUNK_ELEMS / alignCols;
        if (maxRowsPerChunk == 0) {
            maxRowsPerChunk = 1;
        }

        for (uint32_t r = 0; r < rows;) {
            uint32_t chunkRows = (rows - r) < maxRowsPerChunk ? (rows - r) : maxRowsPerChunk;
            uint32_t chunkElems = chunkRows * alignCols;

            ComputeChunk(tensorP, tensorC, gmRow, gmCol, r, chunkRows, cols, alignCols);
            if constexpr (D_NEEDS_CAST) {
                AscendC::DataCopy(ubSaved_, ubD_, chunkElems);
            } else {
                AscendC::DataCopy(ubSaved_, ubP_, chunkElems);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

            ComputeChunk(tensorPT, tensorC, gmRow, gmCol, r, chunkRows, cols, alignCols);
            using Bits = std::conditional_t<sizeof(ElementD) == 2, uint16_t, uint32_t>;
            __ubuf__ Bits* mirrored;
            if constexpr (D_NEEDS_CAST) {
                mirrored = reinterpret_cast<__ubuf__ Bits*>(ubD_.GetPhyAddr());
            } else {
                mirrored = reinterpret_cast<__ubuf__ Bits*>(ubP_.GetPhyAddr());
            }
            SelectUpper(reinterpret_cast<__ubuf__ Bits*>(ubSaved_.GetPhyAddr()), mirrored, static_cast<uint16_t>(r),
                        static_cast<uint16_t>(chunkRows), static_cast<uint16_t>(cols),
                        static_cast<uint16_t>(alignCols));
            AscendC::PipeBarrier<PIPE_V>();

            auto tileD = GetTile(tensorD, tla::MakeCoord(gmRow + r, gmCol), tla::MakeShape(chunkRows, cols));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto layoutDUb = tla::MakeLayout<ElementD, layout::RowMajor>(chunkRows, alignCols);
            auto tensorDUb = tla::MakeTensor(ubSaved_, layoutDUb, Arch::PositionUB{});
            using CopyUbToGmD = Tile::CopyUb2GmTla<ArchTag, decltype(tensorDUb), decltype(tileD)>;
            CopyUbToGmD copyUbToGmD;
            copyUbToGmD(tileD, tensorDUb);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

            r += chunkRows;
        }
    }

    using Bits = std::conditional_t<sizeof(ElementD) == 2, uint16_t, uint32_t>;
    using Index = std::conditional_t<sizeof(ElementD) == 2, int16_t, int32_t>;

    CATLASS_DEVICE_SIMD_VF void SelectUpperVf(__ubuf__ Bits* direct, __ubuf__ Bits* transposed, uint16_t diagCol0,
                                              uint16_t rows, uint16_t cols, uint16_t stride)
    {
        constexpr uint16_t LANES = AscendC::VECTOR_REG_WIDTH / sizeof(Bits);
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t remaining = cols;
            for (uint16_t column = 0; column < cols; column += LANES) {
                AscendC::MicroAPI::RegTensor<Bits> fromP, fromPt, selected;
                AscendC::MicroAPI::RegTensor<Index> indices;
                auto valid = AscendC::MicroAPI::UpdateMask<Bits>(remaining);
                AscendC::MicroAPI::MaskReg upper;
                AscendC::MicroAPI::Arange<Index>(indices, static_cast<Index>(column));
                AscendC::MicroAPI::CompareScalar<Index, AscendC::CMPMODE::GT>(
                    upper, indices, static_cast<Index>(diagCol0 + row), valid);
                uint32_t offset = static_cast<uint32_t>(row) * stride + column;
                AscendC::MicroAPI::DataCopy(fromP, direct + offset);
                AscendC::MicroAPI::DataCopy(fromPt, transposed + offset);
                AscendC::MicroAPI::Select(selected, fromPt, fromP, upper);
                AscendC::MicroAPI::DataCopy(direct + offset, selected, valid);
            }
        }
    }

    CATLASS_DEVICE void SelectUpper(__ubuf__ Bits* direct, __ubuf__ Bits* transposed, uint16_t diagCol0, uint16_t rows,
                                    uint16_t cols, uint16_t stride)
    {
        SelectUpperVf(direct, transposed, diagCol0, rows, cols, stride);
    }

    template <class TensorSrc, class TensorC>
    CATLASS_DEVICE void ComputeChunk(TensorSrc const& tensorSrc, TensorC const& tensorC, uint32_t gmRow, uint32_t gmCol,
                                     uint32_t rowOffset, uint32_t chunkRows, uint32_t cols, uint32_t alignCols)
    {
        IssueLoad(tensorSrc, tensorC, gmRow, gmCol, cols, alignCols, chunkRows, rowOffset + chunkRows, 0);
        Consume(tla::MakeTensor(tensorC.data(), tensorC.layout(), Arch::PositionGM{}), gmRow, gmCol, cols, alignCols,
                chunkRows, rowOffset + chunkRows, 0);
    }

    Params params_;
    AscendC::LocalTensor<ElementCompute> ubPStage_[STAGES];
    AscendC::LocalTensor<ElementY> ubYStage_[STAGES];
    AscendC::LocalTensor<ElementD> ubDStage_[STAGES];
    AscendC::LocalTensor<ElementCompute> ubP_;
    AscendC::LocalTensor<ElementY> ubY_;
    AscendC::LocalTensor<ElementD> ubD_;
    AscendC::LocalTensor<ElementD> ubSaved_;
};

} // namespace Catlass::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SYRK_AXPBY_HPP
