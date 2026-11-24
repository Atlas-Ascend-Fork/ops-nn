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
 * \file cosine_embedding_loss.h
 * \brief CosineEmbeddingLoss arch35 (DAV_3510) RegBase kernel.
 *
 * Generic scalar path matching legacy canndev semantics:
 *
 *   - x1/x2 are broadcast to a common ND shape.
 *   - dimension 1 of that common shape is reduced.
 *   - target is broadcast with the reduced x shape.
 *   - target == 1 selects 1 - cos; target == -1 selects max(0, cos - margin);
 *     any other target value contributes 0.
 *
 * Contiguous two-dimensional inputs use a UB/VF path with fp32 feature reduction. Other
 * broadcast layouts use the scalar fallback. sum/mean split rows across vector cores and merge
 * one 32B-aligned partial per core through workspace.
 */
#ifndef OPS_LOSS_COSINE_EMBEDDING_LOSS_H_
#define OPS_LOSS_COSINE_EMBEDDING_LOSS_H_

#include <float.h>
#include <math.h>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cosine_embedding_loss_tiling_data.h"

namespace NsCosineEmbeddingLoss {
using namespace AscendC;

constexpr uint32_t CEL_VF_LENGTH = 64;
constexpr uint32_t CEL_UB_BLOCK_BYTES = 32;
constexpr int64_t CEL_OUTPUT_TILE_ROWS = 1024;
constexpr uint32_t CEL_STREAM_COLUMNS = 2 * CEL_VF_LENGTH;
constexpr uint32_t CEL_STREAM_BURST_FEATURES = 28;
constexpr uint32_t CEL_STREAM_TILE_FEATURES = 4 * CEL_STREAM_BURST_FEATURES;
constexpr uint32_t CEL_INT_STREAM_ACC_PLANES = 6;
constexpr uint32_t CEL_INT_STREAM_SCALAR_PITCH = CEL_UB_BLOCK_BYTES / sizeof(int32_t);

struct IntColumnStream {
    int64_t span = 0;
    int64_t targetRepeat = 1;
    bool broadcastX2 = false;
};

constexpr AscendC::Reg::CastTrait CEL_CAST_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

template <typename T>
__simd_vf__ inline void CountOutOfRangeVf(__ubuf__ T* src, __ubuf__ float* result, uint32_t count, uint16_t repeats,
                                          float lower, float upper)
{
    AscendC::Reg::RegTensor<float> values, zero, one, total, bad, reduced;
    AscendC::Reg::MaskReg all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(zero, 0.0f);
    AscendC::Reg::Duplicate(one, 1.0f);
    AscendC::Reg::Duplicate(total, 0.0f);
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeats; ++i) {
        const uint32_t offset = static_cast<uint32_t>(i) * CEL_VF_LENGTH;
        AscendC::Reg::MaskReg valid = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (IsSameType<T, float>::value) {
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(values, src + offset);
        } else {
            AscendC::Reg::RegTensor<T> raw;
            if constexpr (IsSameType<T, half>::value) {
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(raw, src + offset);
            } else {
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_NORM>(raw, src + offset);
            }
            AscendC::Reg::Cast<float, T, CEL_CAST_TO_FP32>(values, raw, valid);
        }
        AscendC::Reg::MaskReg lowerOk, upperOk;
        AscendC::Reg::Compares<float, AscendC::CMPMODE::GE>(lowerOk, values, lower, valid);
        AscendC::Reg::Compares<float, AscendC::CMPMODE::LE>(upperOk, values, upper, valid);
        AscendC::Reg::Select<float>(bad, zero, one, lowerOk);
        AscendC::Reg::Select<float>(bad, bad, one, upperOk);
        AscendC::Reg::Select<float>(bad, bad, zero, valid);
        AscendC::Reg::Add<float>(total, total, bad, all);
    }
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reduced, total, all);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(result, reduced, all);
}

template <typename X1T, typename X2T, bool BroadcastX2 = false>
__simd_vf__ inline void ComputeFeaturePartialVf(__ubuf__ X1T* x1, __ubuf__ X2T* x2, __ubuf__ float* partial,
                                                uint32_t count, uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<float> x1Reg;
    AscendC::Reg::RegTensor<float> x2Reg;
    AscendC::Reg::RegTensor<float> zeroReg;
    AscendC::Reg::RegTensor<float> tmpReg;
    AscendC::Reg::RegTensor<float> dotReg;
    AscendC::Reg::RegTensor<float> square1Reg;
    AscendC::Reg::RegTensor<float> square2Reg;
    AscendC::Reg::RegTensor<float> reducedDot;
    AscendC::Reg::RegTensor<float> reducedSquare1;
    AscendC::Reg::RegTensor<float> reducedSquare2;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg validMask;

    AscendC::Reg::Duplicate(zeroReg, 0.0f);
    AscendC::Reg::Duplicate(dotReg, 0.0f);
    AscendC::Reg::Duplicate(square1Reg, 0.0f);
    AscendC::Reg::Duplicate(square2Reg, 0.0f);

    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        const uint32_t offset = static_cast<uint32_t>(i) * CEL_VF_LENGTH;
        validMask = AscendC::Reg::UpdateMask<float>(remaining);

        if constexpr (IsSameType<X1T, float>::value) {
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(x1Reg, x1 + offset);
        } else if constexpr (IsSameType<X1T, half>::value) {
            AscendC::Reg::RegTensor<X1T> x1Raw;
            AscendC::Reg::DataCopy<X1T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(x1Raw, x1 + offset);
            AscendC::Reg::Cast<float, X1T, CEL_CAST_TO_FP32>(x1Reg, x1Raw, validMask);
        } else {
            AscendC::Reg::RegTensor<int32_t> x1Raw;
            AscendC::Reg::DataCopy<int32_t, AscendC::Reg::LoadDist::DIST_NORM>(
                x1Raw, reinterpret_cast<__ubuf__ int32_t*>(x1) + offset);
            AscendC::Reg::Cast<float, int32_t, CEL_CAST_TO_FP32>(x1Reg, x1Raw, validMask);
        }

        if constexpr (BroadcastX2) {
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(x2Reg, x2);
        } else if constexpr (IsSameType<X2T, float>::value) {
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(x2Reg, x2 + offset);
        } else if constexpr (IsSameType<X2T, half>::value) {
            AscendC::Reg::RegTensor<X2T> x2Raw;
            AscendC::Reg::DataCopy<X2T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(x2Raw, x2 + offset);
            AscendC::Reg::Cast<float, X2T, CEL_CAST_TO_FP32>(x2Reg, x2Raw, validMask);
        } else {
            AscendC::Reg::RegTensor<int32_t> x2Raw;
            AscendC::Reg::DataCopy<int32_t, AscendC::Reg::LoadDist::DIST_NORM>(
                x2Raw, reinterpret_cast<__ubuf__ int32_t*>(x2) + offset);
            AscendC::Reg::Cast<float, int32_t, CEL_CAST_TO_FP32>(x2Reg, x2Raw, validMask);
        }

        // Aligned loads may read the padded tail. Select removes those lanes before accumulation.
        AscendC::Reg::Select<float>(x1Reg, x1Reg, zeroReg, validMask);
        AscendC::Reg::Select<float>(x2Reg, x2Reg, zeroReg, validMask);
        AscendC::Reg::Mul<float>(tmpReg, x1Reg, x2Reg, allMask);
        AscendC::Reg::Add<float>(dotReg, dotReg, tmpReg, allMask);
        AscendC::Reg::Mul<float>(tmpReg, x1Reg, x1Reg, allMask);
        AscendC::Reg::Add<float>(square1Reg, square1Reg, tmpReg, allMask);
        AscendC::Reg::Mul<float>(tmpReg, x2Reg, x2Reg, allMask);
        AscendC::Reg::Add<float>(square2Reg, square2Reg, tmpReg, allMask);
    }

    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reducedDot, dotReg, allMask);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reducedSquare1, square1Reg, allMask);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reducedSquare2, square2Reg, allMask);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(partial, reducedDot, allMask);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(partial + 1, reducedSquare1,
                                                                                   allMask);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(partial + 2, reducedSquare2,
                                                                                   allMask);
}

__simd_callee__ inline void AddColumnTerm(AscendC::Reg::RegTensor<float>& total,
                                          AscendC::Reg::RegTensor<float>& correction,
                                          AscendC::Reg::RegTensor<float>& term, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<float> addend, next;
    AscendC::Reg::Sub<float>(addend, term, correction, mask);
    AscendC::Reg::Add<float>(next, total, addend, mask);
    AscendC::Reg::Sub<float>(correction, next, total, mask);
    AscendC::Reg::Sub<float>(correction, correction, addend, mask);
    AscendC::Reg::Copy<float>(total, next, mask);
}

enum class ColumnX2Layout { SCALAR, PAIRED, CONTIGUOUS, ADJACENT_PAIRS, PERIODIC, GROUPED };
constexpr uint32_t CEL_MAX_X2_COLUMN_PERIOD = CEL_VF_LENGTH;

struct X2ColumnGather {
    __ubuf__ int32_t* indices = nullptr;
    uint32_t width = 0;
    uint32_t phase = 0;
    uint32_t phaseStep = 0;
    uint32_t constantPitch = 0;
    uint32_t packedColumns = 1;
    uint32_t groupStep = 0;
};

template <typename T>
__simd_callee__ inline void LoadColumnVector(AscendC::Reg::RegTensor<float>& dst, __ubuf__ T* src,
                                             AscendC::Reg::MaskReg valid)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Reg::LoadAlign(dst, src);
    } else {
        AscendC::Reg::RegTensor<T> raw;
        if constexpr (IsSameType<T, half>::value) {
            AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(raw, src);
        } else {
            AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_NORM>(raw, src);
        }
        AscendC::Reg::Cast<float, T, CEL_CAST_TO_FP32>(dst, raw, valid);
    }
}

template <typename T>
__simd_vf__ inline void AccumulateLongFeatureVf(__ubuf__ T* x1, __ubuf__ float* x2, __ubuf__ float* state,
                                                uint32_t columns, uint32_t pitch, uint32_t features)
{
    uint32_t remaining = columns;
    const uint32_t vectors = (columns + CEL_VF_LENGTH - 1) / CEL_VF_LENGTH;
    for (uint32_t vf = 0; vf < vectors; ++vf) {
        const uint32_t col = vf * CEL_VF_LENGTH;
        AscendC::Reg::MaskReg valid = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> b, dot, square, dotComp, squareComp;
        AscendC::Reg::LoadAlign(b, x2 + col);
        AscendC::Reg::LoadAlign(dot, state + col);
        AscendC::Reg::LoadAlign(square, state + pitch + col);
        AscendC::Reg::LoadAlign(dotComp, state + 2 * pitch + col);
        AscendC::Reg::LoadAlign(squareComp, state + 3 * pitch + col);
        for (uint32_t feature = 0; feature < features; ++feature) {
            AscendC::Reg::RegTensor<float> a, term;
            LoadColumnVector(a, x1 + feature * pitch + col, valid);
            AscendC::Reg::Mul<float>(term, a, b, valid);
            AddColumnTerm(dot, dotComp, term, valid);
            AscendC::Reg::Mul<float>(term, a, a, valid);
            AddColumnTerm(square, squareComp, term, valid);
        }
        AscendC::Reg::StoreAlign(state + col, dot, valid);
        AscendC::Reg::StoreAlign(state + pitch + col, square, valid);
        AscendC::Reg::StoreAlign(state + 2 * pitch + col, dotComp, valid);
        AscendC::Reg::StoreAlign(state + 3 * pitch + col, squareComp, valid);
    }
}

template <bool BroadcastX2>
__simd_vf__ inline void AccumulateIntStreamVf(__ubuf__ int32_t* x1, __ubuf__ int32_t* x2, __ubuf__ float* state,
                                              uint32_t columns, uint32_t features)
{
    constexpr uint32_t pitch = CEL_STREAM_COLUMNS;
    uint32_t remaining = columns;
    const uint32_t vectors = (columns + CEL_VF_LENGTH - 1) / CEL_VF_LENGTH;
    for (uint32_t vf = 0; vf < vectors; ++vf) {
        const uint32_t col = vf * CEL_VF_LENGTH;
        AscendC::Reg::MaskReg valid = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<float> dot, square1, square2, dotComp, square1Comp, square2Comp;
        AscendC::Reg::LoadAlign(dot, state + col);
        AscendC::Reg::LoadAlign(square1, state + pitch + col);
        AscendC::Reg::LoadAlign(square2, state + 2 * pitch + col);
        AscendC::Reg::LoadAlign(dotComp, state + 3 * pitch + col);
        AscendC::Reg::LoadAlign(square1Comp, state + 4 * pitch + col);
        AscendC::Reg::LoadAlign(square2Comp, state + 5 * pitch + col);
        for (uint32_t feature = 0; feature < features; ++feature) {
            AscendC::Reg::RegTensor<float> a, b, term;
            LoadColumnVector(a, x1 + feature * pitch + col, valid);
            if constexpr (BroadcastX2) {
                AscendC::Reg::RegTensor<int32_t> scalar;
                AscendC::Reg::DataCopy<int32_t, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                    scalar, x2 + feature * CEL_INT_STREAM_SCALAR_PITCH);
                AscendC::Reg::Cast<float, int32_t, CEL_CAST_TO_FP32>(b, scalar, valid);
            } else {
                LoadColumnVector(b, x2 + feature * pitch + col, valid);
            }
            AscendC::Reg::Mul<float>(term, a, b, valid);
            AddColumnTerm(dot, dotComp, term, valid);
            AscendC::Reg::Mul<float>(term, a, a, valid);
            AddColumnTerm(square1, square1Comp, term, valid);
            AscendC::Reg::Mul<float>(term, b, b, valid);
            AddColumnTerm(square2, square2Comp, term, valid);
        }
        AscendC::Reg::StoreAlign(state + col, dot, valid);
        AscendC::Reg::StoreAlign(state + pitch + col, square1, valid);
        AscendC::Reg::StoreAlign(state + 2 * pitch + col, square2, valid);
        AscendC::Reg::StoreAlign(state + 3 * pitch + col, dotComp, valid);
        AscendC::Reg::StoreAlign(state + 4 * pitch + col, square1Comp, valid);
        AscendC::Reg::StoreAlign(state + 5 * pitch + col, square2Comp, valid);
    }
}

__simd_callee__ inline void ComputeRepeatedPairPositiveLoss(AscendC::Reg::RegTensor<float>& loss, __ubuf__ float* x1,
                                                            __ubuf__ float* x2, uint32_t pitch,
                                                            AscendC::Reg::RegTensor<float>& norm1,
                                                            AscendC::Reg::RegTensor<float>& norm2,
                                                            AscendC::Reg::RegTensor<float>& one, float eps,
                                                            AscendC::Reg::MaskReg& valid)
{
    AscendC::Reg::RegTensor<float> a, b0, b1, inv1, inv2;
    LoadColumnVector(a, x1, valid);
    LoadColumnVector(b0, x2, valid);
    LoadColumnVector(b1, x2 + pitch, valid);
    AscendC::Reg::Div<float>(inv1, one, norm1, valid);
    AscendC::Reg::Div<float>(inv2, one, norm2, valid);
    AscendC::Reg::Mul<float>(a, a, inv1, valid);
    AscendC::Reg::Mul<float>(b0, b0, inv2, valid);
    AscendC::Reg::Mul<float>(b1, b1, inv2, valid);
    AscendC::Reg::Sub<float>(b0, a, b0, valid);
    AscendC::Reg::Sub<float>(b1, a, b1, valid);
    AscendC::Reg::Mul<float>(b0, b0, b0, valid);
    AscendC::Reg::Mul<float>(b1, b1, b1, valid);
    AscendC::Reg::Mul<float>(inv1, inv1, inv1, valid);
    AscendC::Reg::Mul<float>(inv2, inv2, inv2, valid);
    AscendC::Reg::Add<float>(inv1, inv1, inv2, valid);
    AscendC::Reg::Muls<float>(inv1, inv1, eps, valid);
    AscendC::Reg::Add<float>(loss, b0, b1, valid);
    AscendC::Reg::Add<float>(loss, loss, inv1, valid);
    AscendC::Reg::Muls<float>(loss, loss, 0.5f, valid);
}

template <typename T, typename TargetT, bool DenseTarget>
__simd_vf__ inline void ComputeShortFeatureColumnsVf(__ubuf__ T* x1, __ubuf__ T* x2, __ubuf__ TargetT* target,
                                                     __ubuf__ float* constants, __ubuf__ float* output,
                                                     uint32_t columns, uint32_t x1Pitch, uint32_t pitch,
                                                     uint32_t features, ColumnX2Layout layout, bool negative,
                                                     bool reduce, float margin, float eps, const X2ColumnGather gather,
                                                     bool x2FeatureVaries)
{
    AscendC::Reg::MaskReg all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<float> zero, one, total, totalCorrection;
    AscendC::Reg::Duplicate(zero, 0.0f);
    AscendC::Reg::Duplicate(one, 1.0f);
    AscendC::Reg::Duplicate(total, 0.0f);
    if constexpr (IsSameType<T, int32_t>::value && DenseTarget) {
        AscendC::Reg::Duplicate(totalCorrection, 0.0f);
    }
    uint32_t remaining = columns;
    uint32_t phase = gather.phase;
    uint32_t groupBase = 0;
    const bool stablePositive = IsSameType<T, float>::value && DenseTarget && reduce && features == 2 && x1Pitch == 0 &&
                                layout == ColumnX2Layout::CONTIGUOUS;
    const bool stableIntNone = IsSameType<T, int32_t>::value && !DenseTarget && !reduce && !negative &&
                               !x2FeatureVaries && layout == ColumnX2Layout::CONTIGUOUS;
    const uint32_t vectors = (columns + CEL_VF_LENGTH - 1) / CEL_VF_LENGTH;
    for (uint32_t v = 0; v < vectors; ++v) {
        const uint32_t laneBase = v * CEL_VF_LENGTH;
        AscendC::Reg::MaskReg valid = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::RegTensor<int32_t> x2Indices;
        if (layout == ColumnX2Layout::PERIODIC || layout == ColumnX2Layout::GROUPED) {
            AscendC::Reg::LoadAlign(x2Indices, gather.indices + phase * CEL_VF_LENGTH);
            if (layout == ColumnX2Layout::GROUPED) {
                AscendC::Reg::Adds(x2Indices, x2Indices, static_cast<int32_t>(groupBase), valid);
                groupBase += gather.groupStep + (phase + gather.phaseStep >= gather.width ? gather.packedColumns : 0);
            }
            phase += gather.phaseStep;
            phase -= phase >= gather.width ? gather.width : 0;
        }
        AscendC::Reg::RegTensor<float> dot, aa, bb, dotComp, aaComp, bbComp;
        AscendC::Reg::Duplicate(dot, 0.0f);
        AscendC::Reg::Duplicate(aa, 0.0f);
        AscendC::Reg::Duplicate(bb, 0.0f);
        AscendC::Reg::Duplicate(dotComp, 0.0f);
        AscendC::Reg::Duplicate(aaComp, 0.0f);
        AscendC::Reg::Duplicate(bbComp, 0.0f);
        for (uint32_t feature = 0; feature < features; ++feature) {
            AscendC::Reg::RegTensor<float> a, b, term;
            LoadColumnVector(a, x1 + feature * x1Pitch + laneBase, valid);
            if (layout == ColumnX2Layout::CONTIGUOUS) {
                LoadColumnVector(b, x2 + feature * pitch + laneBase, valid);
            } else if (layout == ColumnX2Layout::ADJACENT_PAIRS) {
                AscendC::Reg::RegTensor<float> packed, unused;
                LoadColumnVector(packed, x2 + feature * pitch + laneBase / 2, valid);
                AscendC::Reg::Interleave(b, unused, packed, packed);
            } else if (layout == ColumnX2Layout::PAIRED) {
                AscendC::Reg::RegTensor<float> first, second, unused;
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(first, constants + feature * 2);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(second,
                                                                                    constants + feature * 2 + 1);
                AscendC::Reg::Interleave(b, unused, first, second);
            } else if (layout == ColumnX2Layout::PERIODIC) {
                auto& indices = reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(x2Indices);
                AscendC::Reg::Gather<float, float, uint32_t>(b, constants + feature * gather.constantPitch, indices,
                                                             valid);
            } else if (layout == ColumnX2Layout::GROUPED) {
                auto& indices = reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(x2Indices);
                if constexpr (IsSameType<T, float>::value) {
                    AscendC::Reg::Gather(b, x2 + feature * pitch, indices, valid);
                } else if constexpr (IsSameType<T, int32_t>::value) {
                    AscendC::Reg::RegTensor<int32_t> packed;
                    AscendC::Reg::Gather(packed, x2 + feature * pitch, indices, valid);
                    AscendC::Reg::Cast<float, int32_t, CEL_CAST_TO_FP32>(b, packed, valid);
                } else {
                    AscendC::Reg::Duplicate(b, 0.0f);
                }
            } else {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(b, constants + feature * 2);
            }
            AscendC::Reg::Mul<float>(term, a, b, valid);
            AddColumnTerm(dot, dotComp, term, valid);
            AscendC::Reg::Mul<float>(term, a, a, valid);
            AddColumnTerm(aa, aaComp, term, valid);
            AscendC::Reg::Mul<float>(term, b, b, valid);
            AddColumnTerm(bb, bbComp, term, valid);
        }
        AscendC::Reg::MaskReg collinear;
        AscendC::Reg::RegTensor<float> collinearLoss;
        if (stableIntNone) {
            // Integer Gram products below 2^24 are exact. For positively collinear rows
            // only the epsilon terms remain; their second-order contribution is below FP32 precision.
            AscendC::Reg::RegTensor<float> product, dotSquared;
            AscendC::Reg::MaskReg belowLimit, positive;
            AscendC::Reg::Mul<float>(product, aa, bb, valid);
            AscendC::Reg::Mul<float>(dotSquared, dot, dot, valid);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(collinear, product, dotSquared, valid);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(belowLimit, product, 0x1p24f, valid);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(positive, dot, 0.0f, valid);
            AscendC::Reg::MaskAnd(collinear, collinear, belowLimit, valid);
            AscendC::Reg::MaskAnd(collinear, collinear, positive, valid);
            AscendC::Reg::Add<float>(collinearLoss, aa, bb, collinear);
            AscendC::Reg::Muls<float>(collinearLoss, collinearLoss, 0.5f * eps, collinear);
            AscendC::Reg::Div<float>(collinearLoss, collinearLoss, product, collinear);
        }
        AscendC::Reg::Adds<float>(aa, aa, eps, valid);
        if (IsSameType<T, float>::value && !DenseTarget && reduce && negative && x2FeatureVaries &&
            layout == ColumnX2Layout::CONTIGUOUS) {
            // Preserve the X2 square-sum compensation when adding epsilon.
            AscendC::Reg::RegTensor<float> epsTerm;
            AscendC::Reg::Duplicate(epsTerm, eps);
            AddColumnTerm(bb, bbComp, epsTerm, valid);
        } else {
            AscendC::Reg::Adds<float>(bb, bb, eps, valid);
        }
        AscendC::Reg::Sqrt<float>(aa, aa, valid);
        AscendC::Reg::Sqrt<float>(bb, bb, valid);
        AscendC::Reg::RegTensor<float> positiveLoss;
        if constexpr (DenseTarget && IsSameType<T, float>::value) {
            if (stablePositive) {
                // Squared distance between normalized vectors avoids cancellation in 1 - cos.
                ComputeRepeatedPairPositiveLoss(positiveLoss, x1 + laneBase, x2 + laneBase, pitch, aa, bb, one, eps,
                                                valid);
            }
        }
        AscendC::Reg::Mul<float>(aa, aa, bb, valid);
        AscendC::Reg::Div<float>(dot, dot, aa, valid);
        AscendC::Reg::RegTensor<float> loss;
        if constexpr (DenseTarget) {
            AscendC::Reg::RegTensor<float> labels, negativeLoss;
            AscendC::Reg::MaskReg positiveLabel, negativeLabel, positiveLossMask, nan;
            LoadColumnVector(labels, target + laneBase, valid);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::EQ>(positiveLabel, labels, 1.0f, valid);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::EQ>(negativeLabel, labels, -1.0f, valid);
            if (!stablePositive) {
                AscendC::Reg::Sub<float>(positiveLoss, one, dot, valid);
            }
            AscendC::Reg::Adds<float>(negativeLoss, dot, -margin, valid);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(positiveLossMask, negativeLoss, 0.0f, valid);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nan, negativeLoss, negativeLoss, valid);
            AscendC::Reg::Select<float>(loss, negativeLoss, zero, positiveLossMask);
            AscendC::Reg::Select<float>(loss, negativeLoss, loss, nan);
            AscendC::Reg::Select<float>(loss, loss, zero, negativeLabel);
            AscendC::Reg::Select<float>(loss, positiveLoss, loss, positiveLabel);
        } else if (negative) {
            AscendC::Reg::MaskReg positive, nan;
            AscendC::Reg::Adds<float>(dot, dot, -margin, valid);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(positive, dot, 0.0f, valid);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nan, dot, dot, valid);
            AscendC::Reg::Select<float>(loss, dot, zero, positive);
            AscendC::Reg::Select<float>(loss, dot, loss, nan);
        } else {
            AscendC::Reg::Sub<float>(loss, one, dot, valid);
            if (stableIntNone) {
                AscendC::Reg::Select<float>(loss, collinearLoss, loss, collinear);
            }
        }
        if (reduce) {
            AscendC::Reg::Select<float>(loss, loss, zero, valid);
            if constexpr (IsSameType<T, int32_t>::value && DenseTarget) {
                // Preserve the low bits before the horizontal sum rounds the tile total.
                AddColumnTerm(total, totalCorrection, loss, all);
            } else {
                AscendC::Reg::Add<float>(total, total, loss, all);
            }
        } else {
            AscendC::Reg::StoreAlign(output + laneBase, loss, valid);
        }
    }
    if (reduce) {
        AscendC::Reg::RegTensor<float> reduced;
        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reduced, total, all);
        AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(output, reduced, all);
    }
}

template <typename TargetT>
__simd_vf__ inline void FinalizeLossVf(__ubuf__ float* stats, __ubuf__ TargetT* target, __ubuf__ float* loss,
                                       float margin, float eps)
{
    AscendC::Reg::RegTensor<float> dotReg;
    AscendC::Reg::RegTensor<float> square1Reg;
    AscendC::Reg::RegTensor<float> square2Reg;
    AscendC::Reg::RegTensor<float> targetReg;
    AscendC::Reg::RegTensor<float> denomReg;
    AscendC::Reg::RegTensor<float> cosReg;
    AscendC::Reg::RegTensor<float> zeroReg;
    AscendC::Reg::RegTensor<float> oneReg;
    AscendC::Reg::RegTensor<float> positiveLossReg;
    AscendC::Reg::RegTensor<float> negativeLossReg;
    AscendC::Reg::RegTensor<float> clampedNegativeReg;
    AscendC::Reg::RegTensor<float> positiveOutReg;
    AscendC::Reg::RegTensor<float> negativeOutReg;
    AscendC::Reg::RegTensor<float> outputReg;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg positiveMask;
    AscendC::Reg::MaskReg negativeMask;
    AscendC::Reg::MaskReg greaterThanZeroMask;
    AscendC::Reg::MaskReg nanMask;

    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(dotReg, stats);
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(square1Reg, stats + 1);
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(square2Reg, stats + 2);
    if constexpr (IsSameType<TargetT, float>::value) {
        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(targetReg, target);
    } else if constexpr (IsSameType<TargetT, half>::value) {
        AscendC::Reg::RegTensor<TargetT> targetRaw;
        AscendC::Reg::DataCopy<TargetT, AscendC::Reg::LoadDist::DIST_BRC_B16>(targetRaw, target);
        AscendC::Reg::Cast<float, TargetT, CEL_CAST_TO_FP32>(targetReg, targetRaw, allMask);
    } else {
        AscendC::Reg::RegTensor<int32_t> targetRaw;
        AscendC::Reg::DataCopy<int32_t, AscendC::Reg::LoadDist::DIST_BRC_B32>(
            targetRaw, reinterpret_cast<__ubuf__ int32_t*>(target));
        AscendC::Reg::Cast<float, int32_t, CEL_CAST_TO_FP32>(targetReg, targetRaw, allMask);
    }

    AscendC::Reg::Adds<float>(square1Reg, square1Reg, eps, allMask);
    AscendC::Reg::Adds<float>(square2Reg, square2Reg, eps, allMask);
    AscendC::Reg::Sqrt<float>(square1Reg, square1Reg, allMask);
    AscendC::Reg::Sqrt<float>(square2Reg, square2Reg, allMask);
    AscendC::Reg::Mul<float>(denomReg, square1Reg, square2Reg, allMask);
    AscendC::Reg::Div<float>(cosReg, dotReg, denomReg, allMask);

    AscendC::Reg::Duplicate(zeroReg, 0.0f);
    AscendC::Reg::Duplicate(oneReg, 1.0f);
    AscendC::Reg::Sub<float>(positiveLossReg, oneReg, cosReg, allMask);
    AscendC::Reg::Adds<float>(negativeLossReg, cosReg, -margin, allMask);

    AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(greaterThanZeroMask, negativeLossReg, 0.0f, allMask);
    AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nanMask, negativeLossReg, negativeLossReg, allMask);
    AscendC::Reg::Select<float>(clampedNegativeReg, negativeLossReg, zeroReg, greaterThanZeroMask);
    AscendC::Reg::Select<float>(clampedNegativeReg, negativeLossReg, clampedNegativeReg, nanMask);

    AscendC::Reg::Compares<float, AscendC::CMPMODE::EQ>(positiveMask, targetReg, 1.0f, allMask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::EQ>(negativeMask, targetReg, -1.0f, allMask);
    AscendC::Reg::Select<float>(positiveOutReg, positiveLossReg, zeroReg, positiveMask);
    AscendC::Reg::Select<float>(negativeOutReg, clampedNegativeReg, zeroReg, negativeMask);
    AscendC::Reg::Add<float>(outputReg, positiveOutReg, negativeOutReg, allMask);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(loss, outputReg, allMask);
}

template <typename X1T, typename X2T, typename TargetT>
class CosineEmbeddingLossKernel {
public:
    __aicore__ inline CosineEmbeddingLossKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR target, GM_ADDR y, GM_ADDR workspace,
                                const CosineEmbeddingLossTilingData* td, TPipe* pipe)
    {
        td_ = td;
        pipe_ = pipe;
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        rowBase_ = blockIdx_ * td_->rowsPerCore;
        if (blockIdx_ >= td_->usedCoreNum) {
            rows_ = 0;
        } else if (blockIdx_ == td_->usedCoreNum - 1) {
            rows_ = td_->tailRows;
        } else {
            rows_ = td_->rowsPerCore;
        }

        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ X1T*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ X2T*>(x2));
        tgtGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TargetT*>(target));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        if (td_->fastPath == COSINE_EMBEDDING_LOSS_LONG_FEATURE_PATH) {
            const int64_t pitch = LongFeaturePitch();
            const int64_t stats = pitch * COSINE_EMBEDDING_LOSS_LONG_FEATURE_ACC_PLANES;
            const int64_t reductionSlot = td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE ?
                                              0 :
                                              COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE;
            pipe_->InitBuffer(scalarBuf_, pitch * sizeof(float));
            pipe_->InitBuffer(columnX1Buf_, (td_->featureTile * pitch + CEL_VF_LENGTH) * sizeof(X1T));
            pipe_->InitBuffer(columnX2Buf_, (pitch + CEL_VF_LENGTH) * sizeof(float));
            pipe_->InitBuffer(featurePartialBuf_, (stats + CEL_VF_LENGTH) * sizeof(float));
            pipe_->InitBuffer(corePartialBuf_, td_->featureSplit * stats * sizeof(float));
            wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace),
                                  td_->usedCoreNum * (stats + reductionSlot));
            return;
        }
        const uint32_t reducedStageBytes = IsSameType<X1T, float>::value ? 2 * MAX_COLUMN_FEATURES * sizeof(float) :
                                                                           CEL_UB_BLOCK_BYTES;
        const uint32_t stagingBytes = UseGenericNone() ? static_cast<uint32_t>(CEL_OUTPUT_TILE_ROWS * sizeof(float)) :
                                                         reducedStageBytes;
        pipe_->InitBuffer(scalarBuf_, stagingBytes);
        PrepareShortFeatureColumns();
        PrepareRepeatedNone();
        intStream_ = FindIntColumnStream();
        PrepareRepeatedStreaming();

        if (CanStreamFeatureColumns() || streamRepeat_ > 1) {
            pipe_->InitBuffer(columnX1Buf_, CEL_STREAM_TILE_FEATURES * CEL_STREAM_COLUMNS * sizeof(X1T));
            pipe_->InitBuffer(columnX2Buf_, CEL_STREAM_COLUMNS * sizeof(float));
            pipe_->InitBuffer(featurePartialBuf_,
                              COSINE_EMBEDDING_LOSS_LONG_FEATURE_ACC_PLANES * CEL_STREAM_COLUMNS * sizeof(float));
        } else if (intStream_.span != 0) {
            const uint32_t x2Pitch = intStream_.broadcastX2 ? CEL_INT_STREAM_SCALAR_PITCH : CEL_STREAM_COLUMNS;
            pipe_->InitBuffer(columnX1Buf_, CEL_STREAM_TILE_FEATURES * CEL_STREAM_COLUMNS * sizeof(X1T));
            pipe_->InitBuffer(columnX2Buf_, CEL_STREAM_TILE_FEATURES * x2Pitch * sizeof(X2T));
            pipe_->InitBuffer(featurePartialBuf_, CEL_INT_STREAM_ACC_PLANES * CEL_STREAM_COLUMNS * sizeof(float));
        } else if (td_->fastPath == COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH) {
            pipe_->InitBuffer(x1Queue_, 1, td_->featureTile * static_cast<int64_t>(sizeof(X1T)));
            pipe_->InitBuffer(x2Queue_, 1, td_->featureTile * static_cast<int64_t>(sizeof(X2T)));
            pipe_->InitBuffer(targetQueue_, 1, CEL_UB_BLOCK_BYTES);
            pipe_->InitBuffer(featurePartialBuf_, CEL_UB_BLOCK_BYTES);
        } else if (CanVectorizeRepeatedRow()) {
            pipe_->InitBuffer(x1Queue_, 1, REPEATED_FEATURE_TILE * sizeof(float));
            pipe_->InitBuffer(x2Queue_, 1, CEL_UB_BLOCK_BYTES);
            pipe_->InitBuffer(featurePartialBuf_, CEL_UB_BLOCK_BYTES);
        }
        if (td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE || (UseGenericNone() && CanScanLargeTarget())) {
            const int64_t workspaceElements = td_->usedCoreNum * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE;
            wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), workspaceElements);
            if (CanScanZeroReduction() ||
                (columnCheckUniform_ && td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE) || streamRepeat_ > 1) {
                pipe_->InitBuffer(zeroScanBuf_, CEL_ZERO_SCAN_TILE * sizeof(float));
            }
            pipe_->InitBuffer(corePartialBuf_,
                              COSINE_EMBEDDING_LOSS_PARTIAL_BUF_ELEMS * static_cast<int64_t>(sizeof(float)));
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (IsSameType<X1T, float>::value || IsSameType<X1T, half>::value ||
                      IsSameType<X1T, int32_t>::value) {
            if (td_->fastPath == COSINE_EMBEDDING_LOSS_LONG_FEATURE_PATH) {
                ProcessLongFeatureColumns();
                return;
            }
        }
        if (td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE && TargetProducesOnlyZeros()) {
            // If target has no active label, its broadcast contributes only zeros.
            if (blockIdx_ == 0) {
                WriteScalarOutput(0, 0.0f);
            }
            return;
        }
        if (TryProcessZeroReduction()) {
            return;
        }
        if constexpr (IsSameType<X1T, X2T>::value) {
            if ((CanStreamFeatureColumns() || streamRepeat_ > 1) && TryProcessStreamingFeatureColumns()) {
                return;
            }
        }
        if constexpr (IsSameType<X1T, int32_t>::value && IsSameType<X2T, int32_t>::value) {
            if (intStream_.span != 0) {
                if (intStream_.broadcastX2) {
                    ProcessIntColumnStream<true>();
                } else {
                    ProcessIntColumnStream<false>();
                }
                return;
            }
        }
        if (TryProcessShortFeatureColumns()) {
            return;
        }
        if (TryProcessRepeatedNone()) {
            return;
        }
        if (UseGenericNone()) {
            ProcessGenericNone();
            return;
        }
        if (td_->fastPath == COSINE_EMBEDDING_LOSS_FEATURE_BROADCAST_REDUCTION_PATH) {
            if (TryProcessFeatureBroadcastReduction() || TryProcessConstReduction()) {
                return;
            }
        } else if (td_->fastPath == COSINE_EMBEDDING_LOSS_CONST_REDUCTION_PATH && TryProcessConstReduction()) {
            return;
        }
        if (TryProcessRepeatedTargetReduction()) {
            return;
        }

        float partial = 0.0f;
        float partialComp = 0.0f;
        const int64_t tileRows = td_->ubTileRows > 0 ? td_->ubTileRows : 1;
        // Split very large row counts into bounded tiles so the kernel does not rely on one giant loop.
        for (int64_t tileBase = 0; tileBase < rows_; tileBase += tileRows) {
            const int64_t currentRows = rows_ - tileBase > tileRows ? tileRows : rows_ - tileBase;
            for (int64_t row = 0; row < currentRows; ++row) {
                const int64_t outputIndex = rowBase_ + tileBase + row;
                const bool useFastPath = td_->fastPath == COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH;
                const float loss = useFastPath ? ComputeFastLoss(outputIndex) : ComputeGenericLoss(outputIndex);
                if (td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE) {
                    WriteScalarOutput(outputIndex, loss);
                } else {
                    // Kahan compensated summation: n can reach tens of millions and naive
                    // fp32 accumulation loses ~sqrt(n)*eps of relative accuracy.
                    const float addY = loss - partialComp;
                    const float addT = partial + addY;
                    partialComp = (addT - partial) - addY;
                    partial = addT;
                }
            }
        }
        if (td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE) {
            FinishReduction(partial);
        }
    }

private:
    template <typename T>
    __aicore__ inline float ReadAsFloat(GlobalTensor<T>& gm, int64_t offset) const
    {
        return static_cast<float>(gm.GetValue(static_cast<uint64_t>(offset)));
    }

    template <typename T>
    __aicore__ inline bool IsSameScalarValue(T lhs, T rhs) const
    {
        if constexpr (IsSameType<T, int32_t>::value) {
            return lhs == rhs;
        } else {
            return static_cast<float>(lhs) == static_cast<float>(rhs);
        }
    }

    template <typename T>
    __aicore__ inline bool IsTensorConstant(GlobalTensor<T>& gm, int64_t count, T reference) const
    {
        if (count <= 0 || td_->usedCoreNum <= 0) {
            return false;
        }
        const int64_t chunk = (count + td_->usedCoreNum - 1) / td_->usedCoreNum;
        const int64_t start = blockIdx_ * chunk;
        const int64_t end = start + chunk > count ? count : start + chunk;
        for (int64_t idx = start; idx < end; ++idx) {
            if (!IsSameScalarValue(gm.GetValue(static_cast<uint64_t>(idx)), reference)) {
                return false;
            }
        }
        return true;
    }

    template <typename T>
    __aicore__ inline bool IsTensorFilledWithFloatValue(GlobalTensor<T>& gm, int64_t count, float value)
    {
        if (count <= 0 || td_->usedCoreNum <= 0) {
            return false;
        }
        const int64_t perCore = count / td_->usedCoreNum;
        const int64_t extra = count % td_->usedCoreNum;
        const int64_t start = blockIdx_ * perCore + (blockIdx_ < extra ? blockIdx_ : extra);
        const int64_t end = start + perCore + (blockIdx_ < extra ? 1 : 0);
        for (int64_t idx = start; idx < end; ++idx) {
            if (ReadAsFloat(gm, idx) != value) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void WriteWorkspaceScalar(int64_t offset, float value)
    {
        LocalTensor<float> stage = scalarBuf_.Get<float>();
        stage.SetValue(0, value);
        event_t eventSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventSMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventSMTE3);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(wsGm_[offset], stage, copyParams);
        event_t eventMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMTE3S);
        WaitFlag<HardEvent::MTE3_S>(eventMTE3S);
    }

    __aicore__ inline bool ReduceWorkspaceBoolean(bool localValue)
    {
        WriteWorkspaceScalar(blockIdx_ * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE, localValue ? 1.0f : 0.0f);

        SyncAll();
        if (blockIdx_ == 0) {
            LocalTensor<float> flags = corePartialBuf_.Get<float>();
            const int64_t workspaceElements = td_->usedCoreNum * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE;
            DataCopyExtParams inputParams{
                1, static_cast<uint32_t>(workspaceElements * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
            DataCopyPadExtParams<float> inputPad{false, 0, 0, 0};
            DataCopyPad(flags, wsGm_, inputParams, inputPad);
            event_t eventMTE2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventMTE2S);
            WaitFlag<HardEvent::MTE2_S>(eventMTE2S);

            bool allTrue = true;
            for (int64_t core = 0; core < td_->usedCoreNum; ++core) {
                if (flags.GetValue(core * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE) != 1.0f) {
                    allTrue = false;
                    break;
                }
            }
            WriteWorkspaceScalar(0, allTrue ? 1.0f : 0.0f);
        }
        SyncAll();

        // Load the decision via MTE2 so no core can reuse a stale scalar DCache line.
        LocalTensor<float> decision = scalarBuf_.Get<float>();
        DataCopyExtParams decisionParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> decisionPad{false, 0, 0, 0};
        DataCopyPad(decision, wsGm_[0], decisionParams, decisionPad);
        const event_t decisionReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(decisionReady);
        WaitFlag<HardEvent::MTE2_S>(decisionReady);
        const bool allTrue = decision.GetValue(0) == 1.0f;
        // All cores consume the decision before a later phase can overwrite slot 0.
        SyncAll();
        return allTrue;
    }

    // Only scan original data when doing so is small relative to the broadcasted reduction.
    __aicore__ inline bool CanProveUnitFeatureZero() const
    {
        return td_->d == 1 && td_->eps >= 0.0f && td_->eps <= 1e-12f && td_->margin >= -0.99f && td_->margin < 0.0f;
    }

    __aicore__ inline bool CanScanZeroReduction() const
    {
        if (td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE ||
            td_->fastPath == COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH || td_->n <= 0 || td_->d <= 0 ||
            td_->usedCoreNum <= 0 || td_->x1Num <= 0 || td_->x2Num <= 0 || td_->targetNum <= 0) {
            return false;
        }
        if (CanProveUnitFeatureZero()) {
            return td_->n >= COSINE_EMBEDDING_LOSS_TARGET_SCAN_MIN_ROWS;
        }
        if (!(td_->margin >= 0.0f && td_->margin <= 1.0f)) {
            return false;
        }
        if constexpr (IsSameType<X1T, float>::value && IsSameType<X2T, float>::value) {
            const int64_t scanBudget = td_->n / CEL_ZERO_SCAN_MIN_REUSE;
            return td_->n >= COSINE_EMBEDDING_LOSS_CONST_REDUCTION_MIN_N && td_->x1Num <= scanBudget &&
                   td_->x2Num <= scanBudget && td_->targetNum <= scanBudget;
        }
        return td_->d > (COSINE_EMBEDDING_LOSS_CONST_REDUCTION_MIN_N - 1) / td_->n;
    }

    template <typename T>
    __aicore__ inline bool CheckBounds(GlobalTensor<T>& input, int64_t count, float lower, float upper)
    {
        const int64_t base = count / td_->usedCoreNum;
        const int64_t extra = count % td_->usedCoreNum;
        const int64_t start = blockIdx_ * base + (blockIdx_ < extra ? blockIdx_ : extra);
        const int64_t end = start + base + (blockIdx_ < extra ? 1 : 0);
        LocalTensor<T> tile = zeroScanBuf_.Get<T>();
        LocalTensor<float> verdict = scalarBuf_.Get<float>();
        auto* const tileAddr = reinterpret_cast<__ubuf__ T*>(tile.GetPhyAddr());
        auto* const verdictAddr = reinterpret_cast<__ubuf__ float*>(verdict.GetPhyAddr());
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        const event_t loaded = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        const event_t checked = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        const event_t consumed = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        for (int64_t offset = start; offset < end;) {
            const int64_t length = end - offset < CEL_ZERO_SCAN_TILE ? end - offset : CEL_ZERO_SCAN_TILE;
            DataCopyExtParams params{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
            DataCopyPad(tile, input[offset], params, pad);
            SetFlag<HardEvent::MTE2_V>(loaded);
            WaitFlag<HardEvent::MTE2_V>(loaded);
            const auto repeats = static_cast<uint16_t>((length + CEL_VF_LENGTH - 1) / CEL_VF_LENGTH);
            asc_vf_call<CountOutOfRangeVf<T>>(tileAddr, verdictAddr, static_cast<uint32_t>(length), repeats, lower,
                                              upper);
            SetFlag<HardEvent::V_S>(checked);
            WaitFlag<HardEvent::V_S>(checked);
            SetFlag<HardEvent::V_MTE2>(consumed);
            WaitFlag<HardEvent::V_MTE2>(consumed);
            if (verdict.GetValue(0) != 0.0f) {
                return false;
            }
            offset += length;
        }
        return true;
    }

    __aicore__ inline bool TryProcessZeroReduction()
    {
        if (!CanScanZeroReduction() || ReadAsFloat(tgtGm_, 0) != -1.0f) {
            return false;
        }
        const float a = ReadAsFloat(x1Gm_, 0);
        const float b = ReadAsFloat(x2Gm_, 0);
        const bool x1Nonnegative = (a > 0.0f && b <= 0.0f) || (a == 0.0f && b < 0.0f);
        const bool x1Nonpositive = (a < 0.0f && b >= 0.0f) || (a == 0.0f && b > 0.0f);
        if (!x1Nonnegative && !x1Nonpositive) {
            return false;
        }
        // Limits keep squared norms and their product after sqrt finite in float32.
        float limit = 1.0e12f / static_cast<float>(td_->d);
        if constexpr (IsSameType<X1T, half>::value) {
            limit = 65504.0f;
        } else if constexpr (IsSameType<X1T, int32_t>::value) {
            limit = 2147483648.0f;
        }
        // For d=1 and both magnitudes >= 1e-3, opposing signs give cos <= -0.999999
        // before rounding. The margin guard leaves room for FP32 rounding.
        const float lowerBound = CanProveUnitFeatureZero() ? 1e-3f : 0.0f;
        const bool localZero = CheckBounds(tgtGm_, td_->targetNum, -1.0f, -1.0f) &&
                               CheckBounds(x1Gm_, td_->x1Num, x1Nonnegative ? lowerBound : -limit,
                                           x1Nonnegative ? limit : -lowerBound) &&
                               CheckBounds(x2Gm_, td_->x2Num, x1Nonnegative ? -limit : lowerBound,
                                           x1Nonnegative ? -lowerBound : limit);
        if (!ReduceWorkspaceBoolean(localZero)) {
            return false;
        }
        if (blockIdx_ == 0) {
            WriteScalarOutput(0, 0.0f);
        }
        return true;
    }

    __aicore__ inline bool TryProcessConstReduction()
    {
        if (td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE || td_->x1Num <= 0 || td_->x2Num <= 0 ||
            td_->targetNum <= 0) {
            return false;
        }

        const X1T x1Reference = x1Gm_.GetValue(0);
        const X2T x2Reference = x2Gm_.GetValue(0);
        const TargetT targetReference = tgtGm_.GetValue(0);
        const bool localConstant = IsTensorConstant(x1Gm_, td_->x1Num, x1Reference) &&
                                   IsTensorConstant(x2Gm_, td_->x2Num, x2Reference) &&
                                   IsTensorConstant(tgtGm_, td_->targetNum, targetReference);
        if (!ReduceWorkspaceBoolean(localConstant)) {
            return false;
        }
        if (blockIdx_ == 0) {
            const float loss = ComputeGenericLoss(0);
            const float total = td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_MEAN ?
                                    loss :
                                    loss * static_cast<float>(td_->n);
            WriteScalarOutput(0, total);
        }
        return true;
    }

    __aicore__ inline bool BuildFeatureBroadcastPlan(int64_t sharedAxes[COSINE_EMBEDDING_LOSS_MAX_RANK],
                                                     int64_t& sharedRank,
                                                     int64_t x1OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK],
                                                     int64_t& x1OnlyRank,
                                                     int64_t x2OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK],
                                                     int64_t& x2OnlyRank, int64_t& sharedCount, int64_t& x1OnlyCount,
                                                     int64_t& x2OnlyCount, int64_t& neutralCount) const
    {
        if (td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE || td_->x1ReduceStride != 0 ||
            td_->outputRank == 0 || td_->d <= 0 || td_->usedCoreNum <= 0) {
            return false;
        }

        sharedRank = 0;
        x1OnlyRank = 0;
        x2OnlyRank = 0;
        sharedCount = 1;
        x1OnlyCount = 1;
        x2OnlyCount = 1;
        neutralCount = 1;
        for (uint32_t axis = 0; axis < td_->outputRank; ++axis) {
            const int64_t dim = td_->outputShape[axis];
            if (dim <= 0) {
                return false;
            }
            const bool x1Depends = td_->x1OutStrides[axis] != 0;
            const bool x2Depends = td_->x2OutStrides[axis] != 0;
            if (x1Depends && x2Depends) {
                sharedAxes[sharedRank++] = axis;
                sharedCount *= dim;
            } else if (x1Depends) {
                x1OnlyAxes[x1OnlyRank++] = axis;
                x1OnlyCount *= dim;
            } else if (x2Depends) {
                x2OnlyAxes[x2OnlyRank++] = axis;
                x2OnlyCount *= dim;
            } else {
                neutralCount *= dim;
            }
        }

        return x1OnlyCount > 0 && x2OnlyCount > 0 && x2OnlyCount <= COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS &&
               sharedCount > 0 && neutralCount > 0;
    }

    __aicore__ inline int64_t LinearOffsetForAxes(int64_t linear, const int64_t axes[COSINE_EMBEDDING_LOSS_MAX_RANK],
                                                  int64_t rank,
                                                  const int64_t strides[COSINE_EMBEDDING_LOSS_MAX_RANK]) const
    {
        int64_t offset = 0;
        for (int64_t i = rank - 1; i >= 0; --i) {
            const int64_t axis = axes[i];
            const int64_t dim = td_->outputShape[static_cast<uint32_t>(axis)];
            const int64_t coord = linear % dim;
            linear /= dim;
            offset += coord * strides[static_cast<uint32_t>(axis)];
        }
        return offset;
    }

    __aicore__ inline void InsertSorted(float values[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS], int64_t count,
                                        float value) const
    {
        int64_t pos = count;
        while (pos > 0 && values[pos - 1] > value) {
            values[pos] = values[pos - 1];
            --pos;
        }
        values[pos] = value;
    }

    __aicore__ inline int64_t LowerBound(const float values[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS], int64_t count,
                                         float value) const
    {
        int64_t left = 0;
        int64_t right = count;
        while (left < right) {
            const int64_t mid = (left + right) / 2;
            if (values[mid] < value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    __aicore__ inline int64_t UpperBound(const float values[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS], int64_t count,
                                         float value) const
    {
        int64_t left = 0;
        int64_t right = count;
        while (left < right) {
            const int64_t mid = (left + right) / 2;
            if (values[mid] <= value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    __aicore__ inline float SumSortedNegativeTargetLoss(float scale,
                                                        const float values[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS],
                                                        const float prefix[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS + 1],
                                                        int64_t count) const
    {
        if (td_->margin != td_->margin || scale != scale) {
            return td_->margin != td_->margin ? td_->margin : scale;
        }
        if (scale == 0.0f) {
            return 0.0f > td_->margin ? -td_->margin * static_cast<float>(count) : 0.0f;
        }

        const float threshold = td_->margin / scale;
        if (scale > 0.0f) {
            const int64_t first = UpperBound(values, count, threshold);
            const int64_t activeCount = count - first;
            const float activeSum = prefix[count] - prefix[first];
            return scale * activeSum - td_->margin * static_cast<float>(activeCount);
        }

        const int64_t firstInactive = LowerBound(values, count, threshold);
        const float activeSum = prefix[firstInactive];
        return scale * activeSum - td_->margin * static_cast<float>(firstInactive);
    }

    __aicore__ inline float NegativeTargetLossFromCos(float cos) const
    {
        const float neg = cos - td_->margin;
        return (neg != neg || neg > 0.0f) ? neg : 0.0f;
    }

    __aicore__ inline float ComputeFeatureBroadcastGroup(
        int64_t sharedIndex, const int64_t sharedAxes[COSINE_EMBEDDING_LOSS_MAX_RANK], int64_t sharedRank,
        const int64_t x1OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK], int64_t x1OnlyRank, int64_t x1Begin, int64_t x1End,
        const int64_t x2OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK], int64_t x2OnlyRank, int64_t x2OnlyCount)
    {
        float sortedStats[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS] = {};
        float prefix[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS + 1] = {};
        const int64_t sharedX1Base = LinearOffsetForAxes(sharedIndex, sharedAxes, sharedRank, td_->x1OutStrides);
        const int64_t sharedX2Base = LinearOffsetForAxes(sharedIndex, sharedAxes, sharedRank, td_->x2OutStrides);

        for (int64_t x2Index = 0; x2Index < x2OnlyCount; ++x2Index) {
            const int64_t x2Base = sharedX2Base +
                                   LinearOffsetForAxes(x2Index, x2OnlyAxes, x2OnlyRank, td_->x2OutStrides);
            float sum = 0.0f;
            float square = 0.0f;
            for (int64_t c = 0; c < td_->d; ++c) {
                const float b = ReadAsFloat(x2Gm_, x2Base + c * td_->x2ReduceStride);
                sum += b;
                square += b * b;
            }
            const float stat = sum / sqrt(square + td_->eps);
            if (stat != stat) {
                return stat;
            }
            InsertSorted(sortedStats, x2Index, stat);
        }

        prefix[0] = 0.0f;
        float prefixComp = 0.0f;
        for (int64_t i = 0; i < x2OnlyCount; ++i) {
            const float addend = sortedStats[i] - prefixComp;
            const float next = prefix[i] + addend;
            prefixComp = (next - prefix[i]) - addend;
            prefix[i + 1] = next;
        }

        float total = 0.0f;
        float totalComp = 0.0f;
        const float dFloat = static_cast<float>(td_->d);
        for (int64_t x1Index = x1Begin; x1Index < x1End; ++x1Index) {
            const int64_t x1Base = sharedX1Base +
                                   LinearOffsetForAxes(x1Index, x1OnlyAxes, x1OnlyRank, td_->x1OutStrides);
            const float a = ReadAsFloat(x1Gm_, x1Base);
            const float scale = a / sqrt(dFloat * a * a + td_->eps);
            const float groupLoss = SumSortedNegativeTargetLoss(scale, sortedStats, prefix, x2OnlyCount);
            const float groupY = groupLoss - totalComp;
            const float groupT = total + groupY;
            totalComp = (groupT - total) - groupY;
            total = groupT;
        }
        return total;
    }

    __aicore__ inline bool TryProcessFeatureBroadcastReduction()
    {
        if (td_->targetNum <= 0) {
            return false;
        }

        int64_t sharedAxes[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
        int64_t x1OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
        int64_t x2OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
        int64_t sharedRank = 0;
        int64_t x1OnlyRank = 0;
        int64_t x2OnlyRank = 0;
        int64_t sharedCount = 0;
        int64_t x1OnlyCount = 0;
        int64_t x2OnlyCount = 0;
        int64_t neutralCount = 0;
        if (!BuildFeatureBroadcastPlan(sharedAxes, sharedRank, x1OnlyAxes, x1OnlyRank, x2OnlyAxes, x2OnlyRank,
                                       sharedCount, x1OnlyCount, x2OnlyCount, neutralCount)) {
            return false;
        }

        const bool localTargetAllNegOne = IsTensorFilledWithFloatValue(tgtGm_, td_->targetNum, -1.0f);
        if (!ReduceWorkspaceBoolean(localTargetAllNegOne)) {
            return false;
        }

        float partial = 0.0f;
        float partialComp = 0.0f;
        // Partition all (shared group, x1) pairs. A single shared group can span many cores.
        const int64_t visits = sharedCount * x1OnlyCount;
        const int64_t perCore = visits / td_->usedCoreNum;
        const int64_t remainder = visits % td_->usedCoreNum;
        int64_t cursor = blockIdx_ * perCore + (blockIdx_ < remainder ? blockIdx_ : remainder);
        const int64_t end = cursor + perCore + (blockIdx_ < remainder ? 1 : 0);
        const float neutralScale = static_cast<float>(neutralCount);
        while (cursor < end) {
            const int64_t sharedIndex = cursor / x1OnlyCount;
            const int64_t x1Begin = cursor % x1OnlyCount;
            const int64_t remainingInGroup = x1OnlyCount - x1Begin;
            const int64_t count = end - cursor < remainingInGroup ? end - cursor : remainingInGroup;
            const float groupSum = ComputeFeatureBroadcastGroup(sharedIndex, sharedAxes, sharedRank, x1OnlyAxes,
                                                                x1OnlyRank, x1Begin, x1Begin + count, x2OnlyAxes,
                                                                x2OnlyRank, x2OnlyCount);
            cursor += count;
            const float exactAdd = groupSum * neutralScale;
            const float exactY = exactAdd - partialComp;
            const float exactT = partial + exactY;
            partialComp = (exactT - partial) - exactY;
            partial = exactT;
        }

        const bool splitCompensation = IsSameType<X1T, float>::value && IsSameType<X2T, float>::value && td_->d == 1;
        FinishReduction(partial, 0, -partialComp, splitCompensation);
        return true;
    }

    __aicore__ inline int64_t RepresentativeOutputIndex(int64_t uniqueIndex, uint32_t dataAxes) const
    {
        int64_t offset = 0;
        int64_t stride = 1;
        for (int64_t axis = static_cast<int64_t>(td_->outputRank) - 1; axis >= 0; --axis) {
            const int64_t dim = td_->outputShape[axis];
            if ((dataAxes & (1U << axis)) != 0) {
                offset += (uniqueIndex % dim) * stride;
                uniqueIndex /= dim;
            }
            stride *= dim;
        }
        return offset;
    }

    __aicore__ inline bool TryProcessRepeatedTargetReduction()
    {
        const bool wideFeature = td_->d >= 64;
        const int64_t minRepeat = wideFeature ? 4 : 8;
        const int64_t targetScanBudget = wideFeature ? td_->n : td_->n / minRepeat;
        constexpr int64_t minRows = 4096;
        if (td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE || td_->n < minRows || td_->targetNum <= 0 ||
            td_->targetNum > targetScanBudget) {
            return false;
        }
        int64_t repeat = 1;
        uint32_t dataAxes = 0;
        for (uint32_t axis = 0; axis < td_->outputRank; ++axis) {
            if (td_->x1OutStrides[axis] == 0 && td_->x2OutStrides[axis] == 0) {
                repeat *= td_->outputShape[axis];
            } else if (td_->outputShape[axis] > 1) {
                dataAxes |= 1U << axis;
            }
        }
        if (repeat < minRepeat) {
            return false;
        }
        const float label = ReadAsFloat(tgtGm_, 0);
        if (label != 1.0f && label != -1.0f) {
            return false;
        }
        const bool uniform = IsTensorFilledWithFloatValue(tgtGm_, td_->targetNum, label);
        if (!ReduceWorkspaceBoolean(uniform)) {
            return false;
        }

        // Axes introduced only by target repeat the same loss when all labels agree.
        const int64_t visits = td_->n / repeat;
        const int64_t perCore = visits / td_->usedCoreNum;
        const int64_t extra = visits % td_->usedCoreNum;
        const int64_t start = blockIdx_ * perCore + (blockIdx_ < extra ? blockIdx_ : extra);
        const int64_t end = start + perCore + (blockIdx_ < extra ? 1 : 0);
        float partial = 0.0f;
        float compensation = 0.0f;
        for (int64_t i = start; i < end; ++i) {
            const float loss = ComputeRepeatedLoss(RepresentativeOutputIndex(i, dataAxes), label);
            const float addend = loss - compensation;
            const float next = partial + addend;
            compensation = (next - partial) - addend;
            partial = next;
        }
        const bool deferRepeat = CanVectorizeRepeatedRow() && label == 1.0f;
        // Merge representative rows before rounding the broadcast scale.
        const float contribution = deferRepeat ? partial : partial * static_cast<float>(repeat);
        FinishReduction(contribution, 0, 0.0f, false, deferRepeat ? repeat : 1);
        return true;
    }

    static constexpr int64_t REPEATED_FEATURE_TILE = 32768;

    __aicore__ inline bool CanVectorizeRepeatedRow() const
    {
        return IsSameType<X1T, float>::value && IsSameType<X2T, float>::value &&
               td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE && td_->n >= 4096 && td_->x1ReduceStride == 1 &&
               td_->x2ReduceStride == 0 && td_->d >= REPEATED_FEATURE_TILE;
    }

    __aicore__ inline float ComputeRepeatedLoss(int64_t index, float label)
    {
        if constexpr (IsSameType<X1T, float>::value && IsSameType<X2T, float>::value) {
            if (CanVectorizeRepeatedRow()) {
                int64_t coords[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
                LinearToOutputCoords(index, coords);
                const int64_t x1Base = OffsetFromStrides(coords, td_->x1OutStrides);
                const int64_t x2Base = OffsetFromStrides(coords, td_->x2OutStrides);
                LocalTensor<float> x2Local = x2Queue_.template AllocTensor<float>();
                DataCopyPad(x2Local, x2Gm_[x2Base], DataCopyExtParams{1, sizeof(float), 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
                x2Queue_.EnQue(x2Local);
                x2Local = x2Queue_.template DeQue<float>();
                LocalTensor<float> tileStats = featurePartialBuf_.Get<float>();
                auto* const statsAddr = reinterpret_cast<__ubuf__ float*>(tileStats.GetPhyAddr());
                auto* const x2Addr = reinterpret_cast<__ubuf__ float*>(x2Local.GetPhyAddr());
                const event_t vectorReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                float sums[3] = {};
                float compensation[3] = {};
                for (int64_t offset = 0; offset < td_->d; offset += REPEATED_FEATURE_TILE) {
                    const int64_t left = td_->d - offset;
                    const auto count = static_cast<uint32_t>(left < REPEATED_FEATURE_TILE ? left :
                                                                                            REPEATED_FEATURE_TILE);
                    LocalTensor<float> x1Local = x1Queue_.template AllocTensor<float>();
                    DataCopyPad(x1Local, x1Gm_[x1Base + offset],
                                DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0},
                                DataCopyPadExtParams<float>{false, 0, 0, 0});
                    x1Queue_.EnQue(x1Local);
                    x1Local = x1Queue_.template DeQue<float>();
                    auto* const x1Addr = reinterpret_cast<__ubuf__ float*>(x1Local.GetPhyAddr());
                    const auto repeats = static_cast<uint16_t>((count + CEL_VF_LENGTH - 1) / CEL_VF_LENGTH);
                    asc_vf_call<ComputeFeaturePartialVf<float, float, true>>(x1Addr, x2Addr, statsAddr, count, repeats);
                    SetFlag<HardEvent::V_S>(vectorReady);
                    WaitFlag<HardEvent::V_S>(vectorReady);
                    for (int32_t i = 0; i < 3; ++i) {
                        const float addend = tileStats.GetValue(i) - compensation[i];
                        const float next = sums[i] + addend;
                        compensation[i] = (next - sums[i]) - addend;
                        sums[i] = next;
                    }
                    x1Queue_.FreeTensor(x1Local);
                }
                x2Queue_.FreeTensor(x2Local);
                const float denominator = sqrt(sums[1] + td_->eps) * sqrt(sums[2] + td_->eps);
                constexpr float maxFloat = 0x1.fffffep127f;
                if (denominator <= maxFloat && sums[0] >= -maxFloat && sums[0] <= maxFloat) {
                    const float cos = sums[0] / denominator;
                    return label == 1.0f ? 1.0f - cos : NegativeTargetLossFromCos(cos);
                }
            }
        }
        return ComputeGenericLoss(index);
    }

    __aicore__ inline void LinearToOutputCoords(int64_t index, int64_t coords[COSINE_EMBEDDING_LOSS_MAX_RANK]) const
    {
        for (int64_t axis = static_cast<int64_t>(td_->outputRank) - 1; axis >= 0; --axis) {
            const int64_t dim = td_->outputShape[static_cast<uint32_t>(axis)];
            coords[static_cast<uint32_t>(axis)] = index % dim;
            index /= dim;
        }
    }

    __aicore__ inline int64_t OffsetFromStrides(const int64_t coords[COSINE_EMBEDDING_LOSS_MAX_RANK],
                                                const int64_t strides[COSINE_EMBEDDING_LOSS_MAX_RANK]) const
    {
        int64_t offset = 0;
        for (uint32_t axis = 0; axis < td_->outputRank; ++axis) {
            offset += coords[axis] * strides[axis];
        }
        return offset;
    }

    __aicore__ inline bool UseGenericNone() const
    {
        return td_->fastPath == COSINE_EMBEDDING_LOSS_GENERIC_PATH &&
               td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE;
    }

    static constexpr int64_t NONE_OUTPUT_BLOCK = CEL_UB_BLOCK_BYTES / sizeof(float);
    static constexpr int64_t NONE_OUTPUT_HALF_BLOCK = NONE_OUTPUT_BLOCK / 2;

    __aicore__ inline void PrepareRepeatedNone()
    {
        constexpr int64_t maxTargetValues = 2048;
        constexpr int64_t minRepeat = 4;
        constexpr int64_t maxRepeat = 32;
        constexpr int64_t minFeatureTerms = 1 << 20;
        if (!UseGenericNone() || columnSpan_ != 0 || td_->targetNum <= 0 || td_->targetNum > maxTargetValues ||
            td_->n <= 0 || td_->usedCoreNum <= 0 || td_->d <= (minFeatureTerms - 1) / td_->n) {
            return;
        }
        uint32_t axis = 0;
        while (axis < td_->outputRank && td_->outputShape[axis] == 1) {
            ++axis;
        }
        if (axis == td_->outputRank || td_->x1OutStrides[axis] != 0 || td_->x2OutStrides[axis] != 0) {
            return;
        }
        const int64_t repeat = td_->outputShape[axis];
        if (repeat < minRepeat || repeat > maxRepeat || td_->targetNum % repeat != 0 ||
            td_->targetOutStrides[axis] != td_->targetNum / repeat) {
            return;
        }
        const int64_t compactRows = td_->n / repeat;
        const int64_t remainder = compactRows % NONE_OUTPUT_BLOCK;
        if (compactRows < td_->usedCoreNum * NONE_OUTPUT_HALF_BLOCK ||
            (remainder != 0 && remainder != NONE_OUTPUT_HALF_BLOCK)) {
            return;
        }
        repeatedNoneRows_ = compactRows;
        repeatedNoneTargetSlice_ = td_->targetNum / repeat;
        if (remainder != 0) {
            pipe_->InitBuffer(repeatedNoneShiftBuf_, CEL_OUTPUT_TILE_ROWS * sizeof(float));
        }
    }

    __aicore__ inline bool TryProcessRepeatedNone()
    {
        const int64_t compactRows = repeatedNoneRows_;
        if (compactRows == 0) {
            return false;
        }
        if (rows_ == 0) {
            return true;
        }
        // Each active core checks the same immutable target before choosing its output partition.
        for (int64_t i = 0; i < repeatedNoneTargetSlice_; ++i) {
            const TargetT first = tgtGm_.GetValue(i);
            for (int64_t j = i + repeatedNoneTargetSlice_; j < td_->targetNum; j += repeatedNoneTargetSlice_) {
                if (!IsSameScalarValue(tgtGm_.GetValue(j), first)) {
                    return false;
                }
            }
        }

        const int64_t compactPerCore = (compactRows - 1) / td_->usedCoreNum + 1;
        const int64_t alignedPerCore = (compactPerCore + NONE_OUTPUT_BLOCK - 1) / NONE_OUTPUT_BLOCK * NONE_OUTPUT_BLOCK;
        if (blockIdx_ > (compactRows - 1) / alignedPerCore) {
            return true;
        }
        const int64_t begin = blockIdx_ * alignedPerCore;
        const int64_t end = begin + (alignedPerCore < compactRows - begin ? alignedPerCore : compactRows - begin);
        const bool needsShift = compactRows % NONE_OUTPUT_BLOCK != 0;
        const int64_t halo = needsShift ? NONE_OUTPUT_HALF_BLOCK : 0;
        const int64_t tileRows = CEL_OUTPUT_TILE_ROWS - (needsShift ? NONE_OUTPUT_BLOCK : 0);
        LocalTensor<float> stage = scalarBuf_.Get<float>();
        LocalTensor<float> shifted = needsShift ? repeatedNoneShiftBuf_.Get<float>() : stage;
        const event_t outputReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        const event_t outputDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        for (int64_t base = begin; base < end;) {
            const int64_t count = end - base < tileRows ? end - base : tileRows;
            for (int64_t i = 0; i < count + halo; ++i) {
                const int64_t index = base + i;
                const float value = ComputeGenericLoss(index < compactRows ? index : index - compactRows);
                stage.SetValue(i, value);
                if (needsShift && i >= halo) {
                    shifted.SetValue(i - halo, value);
                }
            }

            SetFlag<HardEvent::S_MTE3>(outputReady);
            WaitFlag<HardEvent::S_MTE3>(outputReady);
            for (int64_t repeatBase = 0; repeatBase < td_->n; repeatBase += compactRows) {
                // Round both ends up so neighboring tiles and repeats share a 32B boundary.
                const int64_t outputBegin = repeatBase + base;
                const int64_t outputEnd = outputBegin + count;
                const int64_t shift = (NONE_OUTPUT_BLOCK - outputBegin % NONE_OUTPUT_BLOCK) % NONE_OUTPUT_BLOCK;
                const int64_t padding = (NONE_OUTPUT_BLOCK - outputEnd % NONE_OUTPUT_BLOCK) % NONE_OUTPUT_BLOCK;
                const int64_t tail = td_->n - outputEnd;
                const int64_t length = count + (padding < tail ? padding : tail) - shift;
                if (length > 0) {
                    DataCopyPad(yGm_[outputBegin + shift], shift == 0 ? stage : shifted,
                                DataCopyExtParams{1, static_cast<uint32_t>(length * sizeof(float)), 0, 0, 0});
                }
            }
            SetFlag<HardEvent::MTE3_S>(outputDone);
            WaitFlag<HardEvent::MTE3_S>(outputDone);
            base += count;
        }
        return true;
    }

    __aicore__ inline bool SmallTargetProducesOnlyZeros()
    {
        if (td_->targetNum <= 0 || td_->targetNum > COSINE_EMBEDDING_LOSS_MAX_SMALL_TARGET) {
            return false;
        }
        return TargetRangeProducesOnlyZeros(0, td_->targetNum);
    }

    __aicore__ inline bool TargetRangeProducesOnlyZeros(int64_t begin, int64_t end)
    {
        for (int64_t i = begin; i < end; ++i) {
            const float target = ReadAsFloat(tgtGm_, i);
            if (target == 1.0f || target == -1.0f) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool CanScanLargeTarget() const
    {
        const int64_t budget = td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE ?
                                   td_->n / COSINE_EMBEDDING_LOSS_TARGET_SCAN_MIN_REUSE :
                                   td_->n;
        return td_->targetNum > COSINE_EMBEDDING_LOSS_MAX_SMALL_TARGET &&
               td_->n >= COSINE_EMBEDDING_LOSS_TARGET_SCAN_MIN_ROWS && td_->targetNum <= budget;
    }

    __aicore__ inline bool TargetProducesOnlyZeros()
    {
        if (td_->targetNum <= COSINE_EMBEDDING_LOSS_MAX_SMALL_TARGET) {
            return SmallTargetProducesOnlyZeros();
        }
        if (!CanScanLargeTarget() || !TargetRangeProducesOnlyZeros(0, 1)) {
            return false;
        }
        const int64_t perCore = td_->targetNum / td_->usedCoreNum;
        const int64_t extra = td_->targetNum % td_->usedCoreNum;
        const int64_t begin = blockIdx_ * perCore + (blockIdx_ < extra ? blockIdx_ : extra);
        const int64_t end = begin + perCore + (blockIdx_ < extra ? 1 : 0);
        return ReduceWorkspaceBoolean(TargetRangeProducesOnlyZeros(begin, end));
    }

    static constexpr int64_t MAX_COLUMN_FEATURES = 17;
    static constexpr int64_t MAX_COLUMN_TARGET_VALUES = 32;

    struct ShortColumnPlan {
        int64_t span = 1;
        int64_t burstElements = 0;
        uint32_t burstGapBytes = 0;
        uint32_t x2Period = 0;
        uint32_t x2Group = 0;
        uint32_t x2Packed = 1;
        ColumnX2Layout x2Layout = ColumnX2Layout::SCALAR;
        bool denseTarget = false;
    };

    __aicore__ inline bool TryPlanGroupedRun(bool tiledTarget, int64_t& axis, ShortColumnPlan& plan) const
    {
        const int64_t packed = plan.span;
        int64_t span = packed;
        for (int64_t next = axis; next >= 0; --next) {
            const int64_t dim = td_->outputShape[next];
            if (dim == 1) {
                continue;
            }
            if (td_->x1OutStrides[next] != span ||
                (tiledTarget && td_->targetOutStrides[next] != (plan.denseTarget ? span : 0))) {
                return false;
            }
            const int64_t stride = td_->x2OutStrides[next];
            if (stride == packed && span > packed) {
                plan.x2Group = static_cast<uint32_t>(span);
                plan.x2Packed = static_cast<uint32_t>(packed);
                plan.x2Layout = ColumnX2Layout::GROUPED;
                plan.span = span * dim;
                axis = next;
                return true;
            }
            if (stride != 0 || span > CEL_VF_LENGTH / dim) {
                return false;
            }
            span *= dim;
        }
        return false;
    }

    __aicore__ inline ShortColumnPlan PlanShortColumns(bool tiledTarget) const
    {
        ShortColumnPlan plan;
        auto& span = plan.span;
        auto& layout = plan.x2Layout;
        plan.denseTarget = tiledTarget;
        for (int64_t axis = static_cast<int64_t>(td_->outputRank) - 1; axis >= 0; --axis) {
            const int64_t dim = td_->outputShape[axis];
            if (dim == 1) {
                continue;
            }
            const int64_t x1Stride = td_->x1OutStrides[axis];
            if (x1Stride != span) {
                // A 2-D copy packs short, block-aligned rows into a contiguous vector tile.
                if (!tiledTarget && td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE && span < CEL_VF_LENGTH &&
                    span * sizeof(X1T) % CEL_UB_BLOCK_BYTES == 0 && x1Stride > span &&
                    x1Stride - span <= static_cast<int64_t>(UINT32_MAX / sizeof(X1T)) && td_->x2OutStrides[axis] == 0 &&
                    (layout == ColumnX2Layout::SCALAR || layout == ColumnX2Layout::PAIRED)) {
                    plan.burstElements = span;
                    plan.burstGapBytes = static_cast<uint32_t>((x1Stride - span) * sizeof(X1T));
                    span *= dim;
                }
                break;
            }
            if (tiledTarget) {
                const int64_t targetStride = td_->targetOutStrides[axis];
                if (span == 1 && !UseGenericNone()) {
                    plan.denseTarget = targetStride != 0;
                }
                if (targetStride != (plan.denseTarget ? span : 0)) {
                    break;
                }
            }
            const int64_t x2Stride = td_->x2OutStrides[axis];
            constexpr bool groupedSupported = IsSameType<X2T, float>::value || IsSameType<X2T, int32_t>::value;
            if constexpr (groupedSupported) {
                if (layout == ColumnX2Layout::CONTIGUOUS && x2Stride == 0 &&
                    TryPlanGroupedRun(tiledTarget, axis, plan)) {
                    continue;
                }
            }
            if (span == 1 && x2Stride == 1) {
                layout = ColumnX2Layout::CONTIGUOUS;
            } else if (span == 2 && layout == ColumnX2Layout::CONTIGUOUS && x2Stride == 0) {
                layout = ColumnX2Layout::PAIRED;
            } else if (span >= 3 && span <= CEL_MAX_X2_COLUMN_PERIOD && layout == ColumnX2Layout::CONTIGUOUS &&
                       x2Stride == 0) {
                layout = ColumnX2Layout::PERIODIC;
                plan.x2Period = static_cast<uint32_t>(span);
            } else if (span == 2 && layout == ColumnX2Layout::SCALAR && x2Stride == 1) {
                layout = ColumnX2Layout::ADJACENT_PAIRS;
            } else if (groupedSupported && span > 2 && span <= CEL_VF_LENGTH && layout == ColumnX2Layout::SCALAR &&
                       x2Stride == 1) {
                plan.x2Group = static_cast<uint32_t>(span);
                layout = ColumnX2Layout::GROUPED;
            } else {
                const int64_t expected = layout == ColumnX2Layout::ADJACENT_PAIRS ? span / 2 :
                                         layout == ColumnX2Layout::GROUPED    ? span / plan.x2Group * plan.x2Packed :
                                         layout == ColumnX2Layout::CONTIGUOUS ? span :
                                                                                0;
                if (x2Stride != expected) {
                    break;
                }
            }
            span *= dim;
        }
        return plan;
    }

    __aicore__ inline void PrepareShortFeatureColumns()
    {
        if constexpr (!IsSameType<X1T, X2T>::value) {
            return;
        }
        const bool generic = td_->fastPath == COSINE_EMBEDDING_LOSS_GENERIC_PATH ||
                             td_->fastPath == COSINE_EMBEDDING_LOSS_CONST_REDUCTION_PATH ||
                             td_->fastPath == COSINE_EMBEDDING_LOSS_FEATURE_BROADCAST_REDUCTION_PATH;
        if (!generic || td_->targetNum <= 0 || td_->n < 4096 || td_->d < 1 || td_->d > MAX_COLUMN_FEATURES) {
            return;
        }
        const bool tiledTarget = td_->targetNum > MAX_COLUMN_TARGET_VALUES;
        ShortColumnPlan plan = PlanShortColumns(tiledTarget);
        const int64_t minParallelScanRows = td_->d >= 8 ? 32768 : 65536;
        const bool scanBudget = UseGenericNone() ? td_->targetNum <= 2048 :
                                                   (td_->n >= minParallelScanRows && td_->targetNum <= td_->n / 4);
        if (plan.span < CEL_VF_LENGTH && tiledTarget && scanBudget) {
            plan = PlanShortColumns(false);
            columnCheckUniform_ = plan.span >= CEL_VF_LENGTH;
        }
        if (plan.span < CEL_VF_LENGTH) {
            return;
        }
        columnSpan_ = plan.span;
        columnBurstElements_ = plan.burstElements;
        columnBurstGapBytes_ = plan.burstGapBytes;
        columnX2Layout_ = plan.x2Layout;
        columnX2Period_ = plan.x2Period;
        columnX2Group_ = plan.x2Group;
        columnX2Packed_ = plan.x2Packed;
        columnDenseTarget_ = plan.denseTarget;
        // Bound each input arena to the same 32 KiB used by the eight-feature tile.
        constexpr uint32_t maxInputTileBytes = 32 * 1024;
        const uint32_t maxPitch = maxInputTileBytes / (td_->d * sizeof(X1T));
        columnPitch_ = maxPitch < CEL_OUTPUT_TILE_ROWS ? maxPitch / CEL_VF_LENGTH * CEL_VF_LENGTH :
                                                         CEL_OUTPUT_TILE_ROWS;
        const uint32_t planeBytes = td_->d * columnPitch_ * sizeof(X1T);
        pipe_->InitBuffer(columnX1Buf_, td_->x1ReduceStride == 0 ? columnPitch_ * sizeof(X1T) : planeBytes);
        const bool loadX2Columns = plan.x2Layout == ColumnX2Layout::CONTIGUOUS ||
                                   plan.x2Layout == ColumnX2Layout::ADJACENT_PAIRS;
        uint32_t x2Bytes = loadX2Columns ? planeBytes : MAX_COLUMN_FEATURES * 2 * sizeof(float);
        columnX2Pitch_ = columnPitch_;
        if (columnX2Period_ != 0) {
            const uint32_t x2Planes = td_->x2ReduceStride == 0 ? 1 : td_->d;
            x2Bytes = x2Planes * CEL_MAX_X2_COLUMN_PERIOD * sizeof(float);
        } else if (columnX2Group_ != 0) {
            const uint32_t maxPacked = (columnPitch_ + 2 * (columnX2Group_ - 1)) / columnX2Group_ * columnX2Packed_;
            columnX2Pitch_ = (maxPacked + CEL_VF_LENGTH - 1) / CEL_VF_LENGTH * CEL_VF_LENGTH;
            const uint32_t x2Planes = td_->x2ReduceStride == 0 ? 1 : td_->d;
            x2Bytes = x2Planes * columnX2Pitch_ * sizeof(X2T);
        }
        if (columnX2Period_ != 0 || columnX2Group_ != 0) {
            const uint32_t phases = columnX2Period_ != 0 ? columnX2Period_ : columnX2Group_;
            pipe_->InitBuffer(columnX2IndicesBuf_, phases * CEL_VF_LENGTH * sizeof(int32_t));
        }
        pipe_->InitBuffer(columnX2Buf_, x2Bytes);
        if (plan.denseTarget) {
            pipe_->InitBuffer(columnTargetBuf_, columnPitch_ * sizeof(TargetT));
        }
    }

    __aicore__ inline bool TryProcessShortFeatureColumns()
    {
        if constexpr (IsSameType<X1T, X2T>::value) {
            if (columnSpan_ == 0) {
                return false;
            }
            // Preserve the existing early reduction for uniform negative labels.
            if (td_->fastPath == COSINE_EMBEDDING_LOSS_FEATURE_BROADCAST_REDUCTION_PATH &&
                ReadAsFloat(tgtGm_, 0) == -1.0f) {
                return false;
            }
            const bool denseTarget = columnDenseTarget_;
            const bool tiledTarget = td_->targetNum > MAX_COLUMN_TARGET_VALUES && !columnCheckUniform_;
            const bool reduce = td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE;
            float label = 0.0f;
            LocalTensor<TargetT> targetLocal;
            __ubuf__ TargetT* targetAddr = nullptr;
            if (denseTarget) {
                targetLocal = columnTargetBuf_.template Get<TargetT>();
                targetAddr = reinterpret_cast<__ubuf__ TargetT*>(targetLocal.GetPhyAddr());
            } else if (!tiledTarget) {
                label = ReadAsFloat(tgtGm_, 0);
                if (label != 1.0f && label != -1.0f) {
                    return false;
                }
                if (columnCheckUniform_ && reduce) {
                    const bool localUniform = CheckBounds(tgtGm_, td_->targetNum, label, label);
                    if (!ReduceWorkspaceBoolean(localUniform)) {
                        return false;
                    }
                } else {
                    // Prove the broadcast labels are uniform before processing any output tile.
                    for (int64_t i = 1; i < td_->targetNum; ++i) {
                        if (ReadAsFloat(tgtGm_, i) != label) {
                            return false;
                        }
                    }
                }
            }
            constexpr float safeFp32 = 1e12f;
            const bool repeatedX1 = td_->x1ReduceStride == 0;
            const uint32_t x1Pitch = repeatedX1 ? 0 : columnPitch_;
            const bool adjacentPairs = columnX2Layout_ == ColumnX2Layout::ADJACENT_PAIRS;
            const bool groupedX2 = columnX2Group_ != 0;
            const bool periodicX2 = columnX2Period_ != 0;
            const bool loadX2Columns = columnX2Layout_ == ColumnX2Layout::CONTIGUOUS || adjacentPairs || groupedX2;
            const bool repeatedX2 = groupedX2 && td_->x2ReduceStride == 0;
            const uint32_t x2Pitch = repeatedX2 ? 0 : columnX2Pitch_;
            const bool pairedX2 = columnX2Layout_ == ColumnX2Layout::PAIRED;
            X2ColumnGather gather;
            if (periodicX2 || groupedX2) {
                LocalTensor<int32_t> indices = columnX2IndicesBuf_.Get<int32_t>();
                gather.indices = reinterpret_cast<__ubuf__ int32_t*>(indices.GetPhyAddr());
                gather.width = periodicX2 ? columnX2Period_ : columnX2Group_;
                gather.phaseStep = CEL_VF_LENGTH % gather.width;
                gather.constantPitch = td_->x2ReduceStride == 0 ? 0 : CEL_MAX_X2_COLUMN_PERIOD;
                gather.packedColumns = columnX2Packed_;
                gather.groupStep = CEL_VF_LENGTH / gather.width * columnX2Packed_;
                // Gather indexes float elements. A full VF row keeps each phase table 32B-aligned.
                for (uint32_t phase = 0; phase < gather.width; ++phase) {
                    uint32_t index = phase;
                    for (uint32_t lane = 0; lane < CEL_VF_LENGTH; ++lane) {
                        const uint32_t offset = phase + lane;
                        const uint32_t packedIndex = offset / gather.width * columnX2Packed_ + offset % columnX2Packed_;
                        indices.SetValue(phase * CEL_VF_LENGTH + lane, periodicX2 ? index : packedIndex);
                        ++index;
                        index -= index == gather.width ? gather.width : 0;
                    }
                }
            }
            LocalTensor<X1T> x1Local = columnX1Buf_.template Get<X1T>();
            LocalTensor<X2T> x2Local = columnX2Buf_.template Get<X2T>();
            LocalTensor<float> constants = columnX2Buf_.Get<float>();
            LocalTensor<float> stage = scalarBuf_.Get<float>();
            auto* const x1Addr = reinterpret_cast<__ubuf__ X1T*>(x1Local.GetPhyAddr());
            auto* const x2Addr = reinterpret_cast<__ubuf__ X2T*>(x2Local.GetPhyAddr());
            auto* const constantAddr = reinterpret_cast<__ubuf__ float*>(constants.GetPhyAddr());
            auto* const stageAddr = reinterpret_cast<__ubuf__ float*>(stage.GetPhyAddr());
            const event_t inputReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            const event_t constantsReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            const event_t scalarReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            const event_t storeReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            const event_t scalarStoreReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            const event_t stored = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            float partial = 0.0f;
            float correction = 0.0f;
            int64_t begin = rowBase_;
            int64_t end = begin + rows_;
            if (adjacentPairs) {
                // Move a split through a pair to the following even boundary.
                begin += begin & 1;
                end += end & 1;
            }
            int64_t tileColumns = columnPitch_;
            int64_t repeat = 1;
            if (columnBurstElements_ > 0) {
                // Leading output axes that only come from a uniform target repeat the same loss.
                for (uint32_t axis = 0; axis < td_->outputRank; ++axis) {
                    if (td_->outputShape[axis] > 1 && (td_->x1OutStrides[axis] != 0 || td_->x2OutStrides[axis] != 0)) {
                        break;
                    }
                    repeat *= td_->outputShape[axis];
                }
                const int64_t bursts = td_->n / repeat / columnBurstElements_;
                const int64_t perCore = bursts / td_->usedCoreNum;
                const int64_t extra = bursts % td_->usedCoreNum;
                const int64_t firstBurst = blockIdx_ * perCore + (blockIdx_ < extra ? blockIdx_ : extra);
                begin = firstBurst * columnBurstElements_;
                end = begin + (perCore + (blockIdx_ < extra ? 1 : 0)) * columnBurstElements_;
                tileColumns = columnPitch_ / columnBurstElements_ * columnBurstElements_;
                const int64_t inputPlanes = repeatedX1 ? 1 : td_->d;
                Duplicate(x1Local, static_cast<X1T>(0), static_cast<int32_t>(inputPlanes * columnPitch_));
                const event_t cleared = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(cleared);
                WaitFlag<HardEvent::V_MTE2>(cleared);
            }
            for (int64_t base = begin; base < end;) {
                const int64_t toBoundary = columnSpan_ - base % columnSpan_;
                const int64_t available = end - base < toBoundary ? end - base : toBoundary;
                const auto count = static_cast<uint32_t>(available < tileColumns ? available : tileColumns);
                if (periodicX2 || groupedX2) {
                    gather.phase = static_cast<uint32_t>(base % gather.width);
                }
                const uint32_t x2Count = groupedX2 ? (gather.phase + count + columnX2Group_ - 1) / columnX2Group_ *
                                                         columnX2Packed_ :
                                         adjacentPairs ? count / 2 :
                                                         count;
                int64_t coords[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
                LinearToOutputCoords(base, coords);
                if (tiledTarget && !denseTarget) {
                    label = ReadAsFloat(tgtGm_, OffsetFromStrides(coords, td_->targetOutStrides));
                    if (label != 1.0f && label != -1.0f) {
                        base += count;
                        continue;
                    }
                }
                const int64_t x1Base = OffsetFromStrides(coords, td_->x1OutStrides);
                const int64_t x2Base = OffsetFromStrides(coords, td_->x2OutStrides) -
                                       (groupedX2 ? gather.phase % columnX2Packed_ : 0);
                DataCopyExtParams load{1, static_cast<uint32_t>(count * sizeof(X1T)), 0, 0, 0};
                if (columnBurstElements_ > 0) {
                    load.blockCount = static_cast<uint16_t>(count / columnBurstElements_);
                    load.blockLen = static_cast<uint32_t>(columnBurstElements_ * sizeof(X1T));
                    load.srcStride = columnBurstGapBytes_;
                }
                const DataCopyPadExtParams<X1T> padding{false, 0, 0, 0};
                bool safe = true;
                for (int64_t k = 0; k < td_->d; ++k) {
                    if (!repeatedX1 || k == 0) {
                        DataCopyPad(x1Local[k * columnPitch_], x1Gm_[x1Base + k * td_->x1ReduceStride], load, padding);
                    }
                    const int64_t offset = x2Base + k * td_->x2ReduceStride;
                    if (loadX2Columns) {
                        if (!repeatedX2 || k == 0) {
                            DataCopyPad(x2Local[k * x2Pitch], x2Gm_[offset],
                                        DataCopyExtParams{1, static_cast<uint32_t>(x2Count * sizeof(X2T)), 0, 0, 0},
                                        padding);
                        }
                    } else if (periodicX2) {
                        if (k != 0 && gather.constantPitch == 0) {
                            continue;
                        }
                        const int64_t periodBase = offset - gather.phase;
                        for (uint32_t j = 0; j < gather.width; ++j) {
                            const float value = ReadAsFloat(x2Gm_, periodBase + j);
                            constants.SetValue(k * gather.constantPitch + j, value);
                            if constexpr (IsSameType<X1T, float>::value) {
                                safe = safe && value >= -safeFp32 && value <= safeFp32;
                            }
                        }
                    } else {
                        const float first = ReadAsFloat(x2Gm_, offset);
                        const float second = pairedX2 ? ReadAsFloat(x2Gm_, offset + ((base & 1) == 0 ? 1 : -1)) : first;
                        constants.SetValue(k * 2, first);
                        constants.SetValue(k * 2 + 1, second);
                        if constexpr (IsSameType<X1T, float>::value) {
                            safe = safe && first >= -safeFp32 && first <= safeFp32 && second >= -safeFp32 &&
                                   second <= safeFp32;
                        }
                    }
                }
                if (denseTarget) {
                    const int64_t offset = OffsetFromStrides(coords, td_->targetOutStrides);
                    DataCopyPad(targetLocal, tgtGm_[offset],
                                DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(TargetT)), 0, 0, 0},
                                DataCopyPadExtParams<TargetT>{false, 0, 0, 0});
                }
                SetFlag<HardEvent::MTE2_V>(inputReady);
                WaitFlag<HardEvent::MTE2_V>(inputReady);
                if constexpr (IsSameType<X1T, float>::value) {
                    const auto repeats = static_cast<uint16_t>((count + CEL_VF_LENGTH - 1) / CEL_VF_LENGTH);
                    // Scan all FP32 source planes before choosing the bounded vector arithmetic.
                    for (int64_t k = 0; safe && k < td_->d; ++k) {
                        const int64_t input = k * columnPitch_;
                        if (!repeatedX1 || k == 0) {
                            asc_vf_call<CountOutOfRangeVf<float>>(x1Addr + input, stageAddr + k, count, repeats,
                                                                  -safeFp32, safeFp32);
                        }
                        if (loadX2Columns && (!repeatedX2 || k == 0)) {
                            const auto x2Repeats = static_cast<uint16_t>((x2Count + CEL_VF_LENGTH - 1) / CEL_VF_LENGTH);
                            asc_vf_call<CountOutOfRangeVf<float>>(x2Addr + k * x2Pitch,
                                                                  stageAddr + MAX_COLUMN_FEATURES + k, x2Count,
                                                                  x2Repeats, -safeFp32, safeFp32);
                        }
                    }
                    SetFlag<HardEvent::V_S>(scalarReady);
                    WaitFlag<HardEvent::V_S>(scalarReady);
                    for (int64_t k = 0; safe && k < td_->d; ++k) {
                        safe = stage.GetValue(repeatedX1 ? 0 : k) == 0.0f &&
                               (!loadX2Columns || stage.GetValue(MAX_COLUMN_FEATURES + (repeatedX2 ? 0 : k)) == 0.0f);
                    }
                }
                if (safe) {
                    SetFlag<HardEvent::S_V>(constantsReady);
                    WaitFlag<HardEvent::S_V>(constantsReady);
                    if (denseTarget) {
                        asc_vf_call<ComputeShortFeatureColumnsVf<X1T, TargetT, true>>(
                            x1Addr, x2Addr, targetAddr, constantAddr, stageAddr, count, x1Pitch, x2Pitch,
                            static_cast<uint32_t>(td_->d), columnX2Layout_, false, reduce, td_->margin, td_->eps,
                            gather, false);
                    } else {
                        asc_vf_call<ComputeShortFeatureColumnsVf<X1T, TargetT, false>>(
                            x1Addr, x2Addr, targetAddr, constantAddr, stageAddr, count, x1Pitch, x2Pitch,
                            static_cast<uint32_t>(td_->d), columnX2Layout_, label < 0.0f, reduce, td_->margin, td_->eps,
                            gather, td_->x2ReduceStride != 0);
                    }
                }
                if (reduce) {
                    if (safe) {
                        SetFlag<HardEvent::V_S>(scalarReady);
                        WaitFlag<HardEvent::V_S>(scalarReady);
                        AddScalarCompensated(stage.GetValue(0), partial, correction);
                    } else {
                        for (uint32_t lane = 0; lane < count; ++lane) {
                            AddScalarCompensated(ComputeGenericLoss(base + lane), partial, correction);
                        }
                    }
                } else {
                    if (safe) {
                        SetFlag<HardEvent::V_MTE3>(storeReady);
                        WaitFlag<HardEvent::V_MTE3>(storeReady);
                    } else {
                        for (uint32_t lane = 0; lane < count; ++lane) {
                            stage.SetValue(lane, ComputeGenericLoss(base + lane));
                        }
                        SetFlag<HardEvent::S_MTE3>(scalarStoreReady);
                        WaitFlag<HardEvent::S_MTE3>(scalarStoreReady);
                    }
                    DataCopyPad(yGm_[base], stage,
                                DataCopyExtParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0});
                    SetFlag<HardEvent::MTE3_S>(stored);
                    WaitFlag<HardEvent::MTE3_S>(stored);
                }
                base += count;
            }
            if (reduce) {
                FinishReduction(partial * static_cast<float>(repeat));
            }
            return true;
        }
        return false;
    }

    __aicore__ inline bool CanStreamFeatureColumns() const
    {
        if constexpr (!IsSameType<X1T, half>::value || !IsSameType<X2T, half>::value) {
            return false;
        }
        constexpr int64_t minFeatures = 256;
        constexpr int64_t maxFeatures = 4096;
        constexpr int64_t minRows = 4096;
        const int64_t span = td_->x1ReduceStride;
        if (td_->fastPath != COSINE_EMBEDDING_LOSS_GENERIC_PATH ||
            td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE || td_->outputRank < 2 || td_->n < minRows ||
            td_->d < minFeatures || td_->d > maxFeatures || td_->x2Num != 1 || td_->x2ReduceStride != 0 ||
            td_->targetNum != 1 || span < CEL_STREAM_COLUMNS || span > UINT32_MAX / sizeof(X1T) || td_->n % span != 0 ||
            td_->n / span != td_->outputShape[0] || td_->x1Num % td_->d != 0 || td_->x1Num / td_->d != td_->n ||
            td_->x1OutStrides[0] != td_->d * span) {
            return false;
        }

        // Within each outer group, every feature contains the same dense run of output columns.
        int64_t contiguous = 1;
        for (uint32_t axis = td_->outputRank - 1; axis > 0; --axis) {
            if (td_->outputShape[axis] > 1 && td_->x1OutStrides[axis] != contiguous) {
                return false;
            }
            contiguous *= td_->outputShape[axis];
        }
        return contiguous == span;
    }

    __aicore__ inline void PrepareRepeatedStreaming()
    {
        constexpr bool intInputs = IsSameType<X1T, int32_t>::value && IsSameType<X2T, int32_t>::value;
        constexpr bool floatInputs = IsSameType<X1T, float>::value && IsSameType<X2T, float>::value;
        if constexpr (intInputs || floatInputs) {
            const bool scalarX2 = td_->x2Num == 1;
            const bool generic = td_->fastPath == COSINE_EMBEDDING_LOSS_GENERIC_PATH ||
                                 (scalarX2 && td_->fastPath == COSINE_EMBEDDING_LOSS_CONST_REDUCTION_PATH);
            // With INT32 inputs the compensated FP32 norms remain finite over this feature range.
            const int64_t maxFeatures = intInputs ? 4096 : 255;
            const int64_t targetBudget = scalarX2 ? td_->n : td_->n / 4;
            if (!generic || td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE || td_->d < 64 ||
                td_->d > maxFeatures || td_->x1ReduceStride < CEL_VF_LENGTH ||
                td_->x1ReduceStride > UINT32_MAX / sizeof(X1T) || td_->x2ReduceStride != 0 ||
                (intInputs && !scalarX2) || td_->targetNum <= 0 || td_->targetNum > targetBudget) {
                return;
            }
            int64_t repeat = 1;
            for (uint32_t axis = 0; axis < td_->outputRank; ++axis) {
                if (td_->outputShape[axis] > 1 && (td_->x1OutStrides[axis] != 0 || td_->x2OutStrides[axis] != 0)) {
                    break;
                }
                repeat *= td_->outputShape[axis];
            }
            if (repeat < 4 || td_->n / repeat < 4096 || td_->x1Num / td_->d != td_->n / repeat) {
                return;
            }
            int64_t span = 1;
            for (int64_t axis = static_cast<int64_t>(td_->outputRank) - 1; axis >= 0; --axis) {
                const int64_t dim = td_->outputShape[axis];
                if (dim == 1) {
                    continue;
                }
                if (td_->x1OutStrides[axis] != span || td_->x2OutStrides[axis] != 0) {
                    break;
                }
                span *= dim;
            }
            if (span >= CEL_VF_LENGTH && span <= td_->x1ReduceStride && td_->x1ReduceStride % span == 0) {
                streamRepeat_ = repeat;
                streamSpan_ = span;
            }
        }
    }

    template <typename T>
    __aicore__ inline void LoadStreamInputTile(const LocalTensor<T>& local, GlobalTensor<T>& global, int64_t base,
                                               int64_t featureStride, uint32_t columns, uint32_t pitch,
                                               uint32_t features)
    {
        // GM pitch is exact bytes. UB pitch includes DataCopyPad's block alignment and a block gap.
        const uint32_t pitchBytes = pitch * sizeof(T);
        const uint32_t columnBytes = columns * sizeof(T);
        const uint32_t alignedBytes = (columnBytes + CEL_UB_BLOCK_BYTES - 1) / CEL_UB_BLOCK_BYTES * CEL_UB_BLOCK_BYTES;
        const uint32_t srcGap = static_cast<uint32_t>((featureStride - columns) * sizeof(T));
        const uint32_t dstGap = (pitchBytes - alignedBytes) / CEL_UB_BLOCK_BYTES;
        const DataCopyPadExtParams<T> noPad{false, 0, 0, 0};
        for (uint32_t first = 0; first < features; first += CEL_STREAM_BURST_FEATURES) {
            const uint32_t remaining = features - first;
            const uint16_t count = remaining < CEL_STREAM_BURST_FEATURES ? remaining : CEL_STREAM_BURST_FEATURES;
            const DataCopyExtParams load{count, columnBytes, srcGap, dstGap, 0};
            DataCopyPad(local[first * pitch], global[base + first * featureStride], load, noPad);
        }
    }

    __aicore__ inline void LoadStreamFeatureTile(const LocalTensor<X1T>& x1, int64_t base, int64_t span,
                                                 uint32_t columns, uint32_t features)
    {
        LoadStreamInputTile(x1, x1Gm_, base, span, columns, CEL_STREAM_COLUMNS, features);
    }

    __aicore__ inline bool TryProcessStreamingFeatureColumns()
    {
        constexpr float halfMax = 65504.0f;
        const float label = ReadAsFloat(tgtGm_, 0);
        if (label != -1.0f && label != 1.0f) {
            return false;
        }
        if constexpr (IsSameType<X1T, float>::value) {
            constexpr float bound = 1.0e12f;
            const bool localSafe = CheckBounds(tgtGm_, td_->targetNum, label, label) &&
                                   CheckBounds(x1Gm_, td_->x1Num, -bound, bound) &&
                                   CheckBounds(x2Gm_, td_->x2Num, -bound, bound);
            if (!ReduceWorkspaceBoolean(localSafe)) {
                return false;
            }
        } else if constexpr (IsSameType<X1T, int32_t>::value) {
            const bool localUniform = CheckBounds(tgtGm_, td_->targetNum, label, label);
            if (!ReduceWorkspaceBoolean(localUniform)) {
                return false;
            }
        } else {
            const float b = ReadAsFloat(x2Gm_, 0);
            if (!(b >= -halfMax && b <= halfMax)) {
                return false;
            }
        }

        LocalTensor<X1T> x1 = columnX1Buf_.template Get<X1T>();
        LocalTensor<float> x2 = columnX2Buf_.Get<float>();
        LocalTensor<float> state = featurePartialBuf_.Get<float>();
        auto* const x1Addr = reinterpret_cast<__ubuf__ X1T*>(x1.GetPhyAddr());
        auto* const x2Addr = reinterpret_cast<__ubuf__ float*>(x2.GetPhyAddr());
        auto* const stateAddr = reinterpret_cast<__ubuf__ float*>(state.GetPhyAddr());
        const event_t inputConsumed = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        const event_t inputReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        const event_t scalarReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        const event_t stateConsumed = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        Duplicate(x1, static_cast<X1T>(0), CEL_STREAM_TILE_FEATURES * CEL_STREAM_COLUMNS);
        SetFlag<HardEvent::V_MTE2>(inputConsumed);
        WaitFlag<HardEvent::V_MTE2>(inputConsumed);

        const float boundary = label == 1.0f ? 1.0f : td_->margin;
        float partial = 0.0f;
        float correction = 0.0f;
        const int64_t span = td_->x1ReduceStride;
        const int64_t segment = streamRepeat_ > 1 ? streamSpan_ : span;
        int64_t begin = rowBase_;
        int64_t end = begin + rows_;
        if (streamRepeat_ > 1) {
            const int64_t visits = td_->n / streamRepeat_;
            const int64_t perCore = visits / td_->usedCoreNum;
            const int64_t extra = visits % td_->usedCoreNum;
            begin = blockIdx_ * perCore + (blockIdx_ < extra ? blockIdx_ : extra);
            end = begin + perCore + (blockIdx_ < extra ? 1 : 0);
        }
        int64_t lastX2Offset = -1;
        float norm2 = 0.0f;
        for (int64_t output = begin; output < end;) {
            const int64_t inGroup = segment - output % segment;
            const int64_t available = end - output < inGroup ? end - output : inGroup;
            const uint32_t columns = available < CEL_STREAM_COLUMNS ? available : CEL_STREAM_COLUMNS;
            int64_t coords[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
            LinearToOutputCoords(output, coords);
            const int64_t base = OffsetFromStrides(coords, td_->x1OutStrides);
            const int64_t x2Offset = OffsetFromStrides(coords, td_->x2OutStrides);
            if (x2Offset != lastX2Offset) {
                const float b = ReadAsFloat(x2Gm_, x2Offset);
                float secondSquare = 0.0f;
                float secondCorrection = 0.0f;
                for (int64_t feature = 0; feature < td_->d; ++feature) {
                    AddScalarCompensated(b * b, secondSquare, secondCorrection);
                }
                norm2 = sqrt(secondSquare + td_->eps);
                Duplicate(x2, b, CEL_STREAM_COLUMNS);
                lastX2Offset = x2Offset;
            }
            Duplicate(state, 0.0f, COSINE_EMBEDDING_LOSS_LONG_FEATURE_ACC_PLANES * CEL_STREAM_COLUMNS);
            PipeBarrier<PIPE_V>();
            for (int64_t feature = 0; feature < td_->d; feature += CEL_STREAM_TILE_FEATURES) {
                const int64_t remaining = td_->d - feature;
                const uint32_t count = remaining < CEL_STREAM_TILE_FEATURES ? remaining : CEL_STREAM_TILE_FEATURES;
                LoadStreamFeatureTile(x1, base + feature * span, span, columns, count);
                SetFlag<HardEvent::MTE2_V>(inputReady);
                WaitFlag<HardEvent::MTE2_V>(inputReady);
                asc_vf_call<AccumulateLongFeatureVf<X1T>>(x1Addr, x2Addr, stateAddr, columns, CEL_STREAM_COLUMNS,
                                                          count);
                SetFlag<HardEvent::V_MTE2>(inputConsumed);
                WaitFlag<HardEvent::V_MTE2>(inputConsumed);
            }
            SetFlag<HardEvent::V_S>(scalarReady);
            WaitFlag<HardEvent::V_S>(scalarReady);
            for (uint32_t lane = 0; lane < columns; ++lane) {
                const float dot = state.GetValue(lane);
                const float square = state.GetValue(CEL_STREAM_COLUMNS + lane);
                const float cosine = dot / (sqrt(square + td_->eps) * norm2);
                const float distance = cosine - boundary;
                float loss;
                if (!(cosine >= -2.0f && cosine <= 2.0f) || (distance >= -1e-6f && distance <= 1e-6f)) {
                    loss = ComputeGenericLoss(output + lane);
                } else {
                    loss = label == 1.0f ? 1.0f - cosine : NegativeTargetLossFromCos(cosine);
                }
                AddScalarCompensated(loss, partial, correction);
            }
            // Scalar reads of the statistics must finish before the next vector clear reuses the buffer.
            SetFlag<HardEvent::S_V>(stateConsumed);
            WaitFlag<HardEvent::S_V>(stateConsumed);
            output += columns;
        }
        FinishReduction(partial * static_cast<float>(streamRepeat_));
        return true;
    }

    __aicore__ inline IntColumnStream FindIntColumnStream() const
    {
        if constexpr (!IsSameType<X1T, int32_t>::value || !IsSameType<X2T, int32_t>::value) {
            return {};
        }
        constexpr int64_t minFeatures = CEL_VF_LENGTH;
        constexpr int64_t maxFeatures = 4096;
        constexpr int64_t minRows = 4096;
        constexpr int64_t maxFeatureStride = UINT32_MAX / sizeof(int32_t);
        const bool reduce = td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE;
        if (td_->fastPath != COSINE_EMBEDDING_LOSS_GENERIC_PATH || columnSpan_ != 0 || repeatedNoneRows_ != 0 ||
            td_->n < minRows || td_->d < minFeatures || td_->d > maxFeatures || td_->targetNum <= 0 ||
            td_->x1ReduceStride <= 0 || td_->x1ReduceStride > maxFeatureStride || td_->x2ReduceStride <= 0 ||
            td_->x2ReduceStride > maxFeatureStride ||
            (!reduce && td_->usedCoreNum > 1 && td_->rowsPerCore % NONE_OUTPUT_BLOCK != 0)) {
            return {};
        }

        int64_t span = 1;
        int64_t targetRepeat = 1;
        bool broadcastX2 = false;
        for (int32_t axis = static_cast<int32_t>(td_->outputRank) - 1; axis >= 0; --axis) {
            const int64_t dim = td_->outputShape[axis];
            if (dim == 1) {
                continue;
            }
            if (span == 1) {
                broadcastX2 = td_->x2OutStrides[axis] == 0;
            }
            // The DMA must stay within contiguous feature rows of both source tensors.
            if (dim <= 0 || td_->x1OutStrides[axis] != span || td_->x2OutStrides[axis] != (broadcastX2 ? 0 : span) ||
                dim > td_->x1ReduceStride / span || (!broadcastX2 && dim > td_->x2ReduceStride / span)) {
                break;
            }
            const int64_t targetStride = td_->targetOutStrides[axis];
            if (span == targetRepeat && targetStride == 0) {
                targetRepeat *= dim;
            } else if (targetStride != span / targetRepeat) {
                break;
            }
            span *= dim;
        }
        if (span < CEL_VF_LENGTH || td_->n % span != 0) {
            return {};
        }
        return {span, targetRepeat, broadcastX2};
    }

    __aicore__ inline float FinishIntStreamLoss(const LocalTensor<float>& state, uint32_t lane, float label,
                                                int64_t output)
    {
        if (label != -1.0f && label != 1.0f) {
            return 0.0f;
        }
        const float dot = state.GetValue(lane);
        const float square1 = state.GetValue(CEL_STREAM_COLUMNS + lane);
        const float square2 = state.GetValue(2 * CEL_STREAM_COLUMNS + lane);
        const float cosine = dot / (sqrt(square1 + td_->eps) * sqrt(square2 + td_->eps));
        const float distance = cosine - (label == 1.0f ? 1.0f : td_->margin);
        if (!(cosine >= -2.0f && cosine <= 2.0f) || (distance >= -1e-6f && distance <= 1e-6f)) {
            return ComputeGenericLoss(output);
        }
        return label == 1.0f ? 1.0f - cosine : NegativeTargetLossFromCos(cosine);
    }

    template <bool BroadcastX2>
    __aicore__ inline void ProcessIntColumnStream()
    {
        constexpr uint32_t x2Pitch = BroadcastX2 ? CEL_INT_STREAM_SCALAR_PITCH : CEL_STREAM_COLUMNS;
        LocalTensor<X1T> x1 = columnX1Buf_.template Get<X1T>();
        LocalTensor<X2T> x2 = columnX2Buf_.template Get<X2T>();
        LocalTensor<float> state = featurePartialBuf_.Get<float>();
        LocalTensor<float> stage = scalarBuf_.Get<float>();
        auto* const x1Addr = reinterpret_cast<__ubuf__ int32_t*>(x1.GetPhyAddr());
        auto* const x2Addr = reinterpret_cast<__ubuf__ int32_t*>(x2.GetPhyAddr());
        auto* const stateAddr = reinterpret_cast<__ubuf__ float*>(state.GetPhyAddr());
        const event_t inputConsumed = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        const event_t inputReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        const event_t scalarReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        const event_t stateConsumed = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        Duplicate(x1, static_cast<X1T>(0), CEL_STREAM_TILE_FEATURES * CEL_STREAM_COLUMNS);
        Duplicate(x2, static_cast<X2T>(0), CEL_STREAM_TILE_FEATURES * x2Pitch);
        SetFlag<HardEvent::V_MTE2>(inputConsumed);
        WaitFlag<HardEvent::V_MTE2>(inputConsumed);

        const bool reduce = td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE;
        float partial = 0.0f;
        float correction = 0.0f;
        int64_t coords[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
        const int64_t end = rowBase_ + rows_;
        for (int64_t output = rowBase_; output < end;) {
            const int64_t column = output % intStream_.span;
            const int64_t inGroup = intStream_.span - column;
            const int64_t available = end - output < inGroup ? end - output : inGroup;
            const uint32_t columns = available < CEL_STREAM_COLUMNS ? available : CEL_STREAM_COLUMNS;
            LinearToOutputCoords(output, coords);
            const int64_t aBase = OffsetFromStrides(coords, td_->x1OutStrides);
            const int64_t bBase = OffsetFromStrides(coords, td_->x2OutStrides);
            int64_t tBase = OffsetFromStrides(coords, td_->targetOutStrides);
            Duplicate(state, 0.0f, CEL_INT_STREAM_ACC_PLANES * CEL_STREAM_COLUMNS);
            PipeBarrier<PIPE_V>();
            for (int64_t feature = 0; feature < td_->d; feature += CEL_STREAM_TILE_FEATURES) {
                const int64_t remaining = td_->d - feature;
                const uint32_t count = remaining < CEL_STREAM_TILE_FEATURES ? remaining : CEL_STREAM_TILE_FEATURES;
                LoadStreamInputTile(x1, x1Gm_, aBase + feature * td_->x1ReduceStride, td_->x1ReduceStride, columns,
                                    CEL_STREAM_COLUMNS, count);
                LoadStreamInputTile(x2, x2Gm_, bBase + feature * td_->x2ReduceStride, td_->x2ReduceStride,
                                    BroadcastX2 ? 1 : columns, x2Pitch, count);
                SetFlag<HardEvent::MTE2_V>(inputReady);
                WaitFlag<HardEvent::MTE2_V>(inputReady);
                asc_vf_call<AccumulateIntStreamVf<BroadcastX2>>(x1Addr, x2Addr, stateAddr, columns, count);
                SetFlag<HardEvent::V_MTE2>(inputConsumed);
                WaitFlag<HardEvent::V_MTE2>(inputConsumed);
            }
            SetFlag<HardEvent::V_S>(scalarReady);
            WaitFlag<HardEvent::V_S>(scalarReady);
            int64_t nextTarget = intStream_.targetRepeat - column % intStream_.targetRepeat;
            float label = ReadAsFloat(tgtGm_, tBase);
            for (uint32_t lane = 0; lane < columns; ++lane) {
                if (lane == nextTarget) {
                    label = ReadAsFloat(tgtGm_, ++tBase);
                    nextTarget += intStream_.targetRepeat;
                }
                const float loss = FinishIntStreamLoss(state, lane, label, output + lane);
                if (reduce) {
                    AddScalarCompensated(loss, partial, correction);
                } else {
                    stage.SetValue(lane, loss);
                }
            }
            SetFlag<HardEvent::S_V>(stateConsumed);
            WaitFlag<HardEvent::S_V>(stateConsumed);
            if (!reduce) {
                const event_t outputReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(outputReady);
                WaitFlag<HardEvent::S_MTE3>(outputReady);
                const uint32_t bytes = columns * sizeof(float);
                DataCopyPad(yGm_[output], stage, DataCopyExtParams{1, bytes, 0, 0, 0});
                const event_t stored = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
                SetFlag<HardEvent::MTE3_S>(stored);
                WaitFlag<HardEvent::MTE3_S>(stored);
            }
            output += columns;
        }
        if (reduce) {
            FinishReduction(partial);
        }
    }

    __aicore__ inline int64_t LongFeaturePitch() const
    {
        return td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE ? td_->x1ReduceStride :
                                                                        COSINE_EMBEDDING_LOSS_LONG_FEATURE_MEAN_PITCH;
    }

    __aicore__ inline void ProcessLongFeatureColumns()
    {
        const int64_t columns = td_->x1ReduceStride;
        const int64_t pitch = LongFeaturePitch();
        const int64_t stats = pitch * COSINE_EMBEDDING_LOSS_LONG_FEATURE_ACC_PLANES;
        const int64_t splits = td_->featureSplit;
        const int64_t group = blockIdx_ / splits;
        const int64_t part = blockIdx_ % splits;
        const int64_t outputBase = group * columns;

        LocalTensor<X1T> x1 = columnX1Buf_.template Get<X1T>();
        LocalTensor<float> x2 = columnX2Buf_.Get<float>();
        LocalTensor<float> state = featurePartialBuf_.Get<float>();
        auto* const x1Addr = reinterpret_cast<__ubuf__ X1T*>(x1.GetPhyAddr());
        auto* const x2Addr = reinterpret_cast<__ubuf__ float*>(x2.GetPhyAddr());
        auto* const stateAddr = reinterpret_cast<__ubuf__ float*>(state.GetPhyAddr());

        int64_t coords[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
        for (int64_t col = 0; col < columns; ++col) {
            LinearToOutputCoords(outputBase + col, coords);
            x2.SetValue(col, ReadAsFloat(x2Gm_, OffsetFromStrides(coords, td_->x2OutStrides)));
        }
        for (int64_t col = columns; col < pitch + CEL_VF_LENGTH; ++col) {
            x2.SetValue(col, 0.0f);
        }
        const event_t constantsReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(constantsReady);
        WaitFlag<HardEvent::S_V>(constantsReady);

        // Tail VF loads can read past the current DMA payload inside UB. Initialize that storage once.
        Duplicate(x1, static_cast<X1T>(0), static_cast<int32_t>(td_->featureTile * pitch + CEL_VF_LENGTH));
        Duplicate(state, 0.0f, static_cast<int32_t>(stats + CEL_VF_LENGTH));
        const event_t inputConsumed = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(inputConsumed);
        WaitFlag<HardEvent::V_MTE2>(inputConsumed);
        const event_t inputReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        const DataCopyPadExtParams<X1T> noPad{false, 0, 0, 0};
        const int64_t first = td_->d * part / splits;
        const int64_t end = td_->d * (part + 1) / splits;
        const int64_t groupBase = group * td_->d * columns;
        const uint32_t columnBytes = static_cast<uint32_t>(columns * sizeof(X1T));
        const uint32_t alignedColumnBytes = (columnBytes + CEL_UB_BLOCK_BYTES - 1) / CEL_UB_BLOCK_BYTES *
                                            CEL_UB_BLOCK_BYTES;
        const uint32_t ubGap = pitch == columns ? 0 : (pitch * sizeof(X1T) - alignedColumnBytes) / CEL_UB_BLOCK_BYTES;
        for (int64_t row = first; row < end;) {
            const int64_t count = end - row < td_->featureTile ? end - row : td_->featureTile;
            const DataCopyExtParams load = pitch == columns ?
                                               DataCopyExtParams{1, static_cast<uint32_t>(count * columnBytes), 0, 0,
                                                                 0} :
                                               DataCopyExtParams{static_cast<uint16_t>(count), columnBytes, 0, ubGap,
                                                                 0};
            DataCopyPad(x1, x1Gm_[groupBase + row * columns], load, noPad);
            SetFlag<HardEvent::MTE2_V>(inputReady);
            WaitFlag<HardEvent::MTE2_V>(inputReady);
            asc_vf_call<AccumulateLongFeatureVf<X1T>>(x1Addr, x2Addr, stateAddr, static_cast<uint32_t>(columns),
                                                      static_cast<uint32_t>(pitch), static_cast<uint32_t>(count));
            SetFlag<HardEvent::V_MTE2>(inputConsumed);
            WaitFlag<HardEvent::V_MTE2>(inputConsumed);
            row += count;
        }

        const event_t partialReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(partialReady);
        WaitFlag<HardEvent::V_MTE3>(partialReady);
        DataCopyPad(wsGm_[blockIdx_ * stats], state,
                    DataCopyExtParams{1, static_cast<uint32_t>(stats * sizeof(float)), 0, 0, 0});
        const event_t stored = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(stored);
        WaitFlag<HardEvent::MTE3_S>(stored);
        SyncAll();
        float partial = 0.0f;
        float correction = 0.0f;
        if (part == 0) {
            partial = MergeLongFeatureColumns(x2, outputBase, blockIdx_ * stats, correction);
        }
        if (td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE) {
            // Store reduction slots after every group's feature statistics to avoid cross-core aliasing.
            FinishReduction(partial, td_->usedCoreNum * stats, -correction);
        }
    }

    __aicore__ inline void AddScalarCompensated(float value, float& total, float& correction)
    {
        const float addend = value - correction;
        const float next = total + addend;
        correction = (next - total) - addend;
        total = next;
    }

    __aicore__ inline float MergeLongFeatureColumns(const LocalTensor<float>& x2, int64_t outputBase,
                                                    int64_t workspaceBase, float& correction)
    {
        const int64_t columns = td_->x1ReduceStride;
        const int64_t pitch = LongFeaturePitch();
        const int64_t stats = pitch * COSINE_EMBEDDING_LOSS_LONG_FEATURE_ACC_PLANES;
        const bool reduce = td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE;
        const int64_t physicalOutputs = td_->x1Num / td_->d;
        LocalTensor<float> partials = corePartialBuf_.Get<float>();
        LocalTensor<float> stage = scalarBuf_.Get<float>();
        const DataCopyExtParams load{1, static_cast<uint32_t>(td_->featureSplit * stats * sizeof(float)), 0, 0, 0};
        DataCopyPad(partials, wsGm_[workspaceBase], load, DataCopyPadExtParams<float>{false, 0, 0, 0});
        const event_t partialsReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(partialsReady);
        WaitFlag<HardEvent::MTE2_S>(partialsReady);

        // Keep both squared norms and the product of the two norms in the finite FP32 range.
        constexpr float safeSum = 1e36f;
        constexpr float cancellationLimit = 1e-6f;
        int64_t coords[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
        float partial = 0.0f;
        for (int64_t col = 0; col < columns; ++col) {
            float dot = 0.0f;
            float square = 0.0f;
            float dotComp = 0.0f;
            float squareComp = 0.0f;
            for (int64_t part = 0; part < td_->featureSplit; ++part) {
                const int64_t offset = part * stats + col;
                AddScalarCompensated(partials.GetValue(offset), dot, dotComp);
                AddScalarCompensated(-partials.GetValue(offset + 2 * pitch), dot, dotComp);
                AddScalarCompensated(partials.GetValue(offset + pitch), square, squareComp);
                AddScalarCompensated(-partials.GetValue(offset + 3 * pitch), square, squareComp);
            }
            const float b = x2.GetValue(col);
            const float square2 = (b * b) * static_cast<float>(td_->d);
            const bool safe = dot >= -safeSum && dot <= safeSum && square >= 0.0f && square <= safeSum &&
                              square2 >= 0.0f && square2 <= safeSum;
            const float cosine = safe ? dot / (sqrt(square + td_->eps) * sqrt(square2 + td_->eps)) : 0.0f;
            // Each target-only prefix maps back to this same physical column, but has its own label.
            for (int64_t outputIndex = outputBase + col; outputIndex < td_->n; outputIndex += physicalOutputs) {
                LinearToOutputCoords(outputIndex, coords);
                const float label = ReadAsFloat(tgtGm_, OffsetFromStrides(coords, td_->targetOutStrides));
                float loss = 0.0f;
                if (label == 1.0f || label == -1.0f) {
                    const float boundary = label == 1.0f ? 1.0f : td_->margin;
                    const float delta = cosine - boundary;
                    if (!safe || (delta >= -cancellationLimit && delta <= cancellationLimit)) {
                        loss = ComputeGenericLoss(outputIndex);
                    } else {
                        loss = label == 1.0f ? 1.0f - cosine : NegativeTargetLossFromCos(cosine);
                    }
                }
                if (reduce) {
                    AddScalarCompensated(loss, partial, correction);
                } else {
                    stage.SetValue(col, loss);
                }
            }
        }
        if (reduce) {
            return partial;
        }
        const event_t outputReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(outputReady);
        WaitFlag<HardEvent::S_MTE3>(outputReady);
        DataCopyPad(yGm_[outputBase], stage,
                    DataCopyExtParams{1, static_cast<uint32_t>(columns * sizeof(float)), 0, 0, 0});
        const event_t stored = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(stored);
        WaitFlag<HardEvent::MTE3_S>(stored);
        return 0.0f;
    }

    __aicore__ inline void ProcessGenericNone()
    {
        if (rows_ == 0) {
            return;
        }
        LocalTensor<float> stage = scalarBuf_.Get<float>();
        const bool zeroOnly = TargetProducesOnlyZeros();
        if (zeroOnly) {
            Duplicate(stage, 0.0f, static_cast<int32_t>(CEL_OUTPUT_TILE_ROWS));
            event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        }

        for (int64_t tileBase = 0; tileBase < rows_;) {
            const int64_t remaining = rows_ - tileBase;
            const int64_t count = remaining < CEL_OUTPUT_TILE_ROWS ? remaining : CEL_OUTPUT_TILE_ROWS;
            const int64_t outputBase = rowBase_ + tileBase;
            if (!zeroOnly) {
                for (int64_t row = 0; row < count; ++row) {
                    stage.SetValue(row, ComputeGenericLoss(outputBase + row));
                }
                event_t eventSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventSMTE3);
                WaitFlag<HardEvent::S_MTE3>(eventSMTE3);
            }
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
            DataCopyPad(yGm_[outputBase], stage, copyParams);
            event_t eventMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventMTE3S);
            WaitFlag<HardEvent::MTE3_S>(eventMTE3S);
            tileBase += count;
        }
    }

    struct FeatureStats {
        float dot;
        float square1;
        float square2;
    };

    template <bool Rescale = false>
    __aicore__ inline FeatureStats AccumulateGenericFeatures(int64_t x1Base, int64_t x2Base, float scale1 = 1.0f,
                                                             float scale2 = 1.0f)
    {
        float dot = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float dotComp = 0.0f;
        float s1Comp = 0.0f;
        float s2Comp = 0.0f;
        for (int64_t c = 0; c < td_->d; ++c) {
            float a = ReadAsFloat(x1Gm_, x1Base + c * td_->x1ReduceStride);
            float b = ReadAsFloat(x2Gm_, x2Base + c * td_->x2ReduceStride);
            if constexpr (Rescale) {
                a *= scale1;
                b *= scale2;
            }
            // Kahan compensated accumulation.
            const float dotAdd = a * b - dotComp;
            const float dotSum = dot + dotAdd;
            dotComp = (dotSum - dot) - dotAdd;
            dot = dotSum;
            const float s1Add = a * a - s1Comp;
            const float s1Sum = s1 + s1Add;
            s1Comp = (s1Sum - s1) - s1Add;
            s1 = s1Sum;
            const float s2Add = b * b - s2Comp;
            const float s2Sum = s2 + s2Add;
            s2Comp = (s2Sum - s2) - s2Add;
            s2 = s2Sum;
        }

        return {dot, s1, s2};
    }

    __aicore__ inline float ComputeGenericLoss(int64_t outputIndex)
    {
        int64_t coords[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
        LinearToOutputCoords(outputIndex, coords);
        const int64_t targetOffset = OffsetFromStrides(coords, td_->targetOutStrides);
        const float target = ReadAsFloat(tgtGm_, targetOffset);
        if (target != 1.0f && target != -1.0f) {
            return 0.0f;
        }

        const int64_t x1Base = OffsetFromStrides(coords, td_->x1OutStrides);
        const int64_t x2Base = OffsetFromStrides(coords, td_->x2OutStrides);
        auto stats = AccumulateGenericFeatures(x1Base, x2Base);
        const float norm1 = sqrt(stats.square1 + td_->eps);
        const float norm2 = sqrt(stats.square2 + td_->eps);
        float denominator = norm1 * norm2;
        if constexpr (IsSameType<X1T, float>::value && IsSameType<X2T, float>::value) {
            if (!(denominator <= FLT_MAX && stats.dot >= -FLT_MAX && stats.dot <= FLT_MAX)) {
                bool scaleX1 = !(stats.square1 <= FLT_MAX);
                bool scaleX2 = !(stats.square2 <= FLT_MAX);
                if (!scaleX1 && !scaleX2) {
                    scaleX1 = stats.square1 >= stats.square2;
                    scaleX2 = !scaleX1;
                }
                // 2^-100 keeps FP32 products and their sums in range even for very long rows.
                // The scaled epsilon rounds to zero. Keep epsilon on any unscaled input.
                constexpr float scale = 0x1p-100f;
                stats = AccumulateGenericFeatures<true>(x1Base, x2Base, scaleX1 ? scale : 1.0f, scaleX2 ? scale : 1.0f);
                const float eps1 = scaleX1 ? 0.0f : td_->eps;
                const float eps2 = scaleX2 ? 0.0f : td_->eps;
                denominator = sqrt(stats.square1 + eps1) * sqrt(stats.square2 + eps2);
            }
        }
        // 说明：1 - cos 在 cos -> 1 时存在 fp32 灾难性抵消。实测对比（025，3432 采样）：
        //   本式 1 - dot/S :           max err 2.47e-7 (mean 4.8e-8)   <= 最优
        //   |a/|a| - b/|b||^2 / 2 :    max err 2.14e-4 (mean 2.5e-6)
        //   (s1*s2-dot^2)/(...) :      max err 2.07e-4 (mean 2.4e-6)
        //   半角 sin^2/(1+cos) :       max err 1.95e-4 (mean 2.2e-6)
        // 直接式已是 fp32 下精度最好的，故保留；不能再靠算子侧提升。
        const float cos = stats.dot / denominator;
        if (target == 1.0f) {
            return 1.0f - cos;
        }
        return NegativeTargetLossFromCos(cos);
    }

    __aicore__ inline float ComputeFastLoss(int64_t outputIndex)
    {
        const int64_t x1Base = outputIndex * td_->x1OutStrides[0];
        const int64_t x2Base = outputIndex * td_->x2OutStrides[0];
        float dot = 0.0f;
        float square1 = 0.0f;
        float square2 = 0.0f;

        for (int64_t featureOffset = 0; featureOffset < td_->d; featureOffset += td_->featureTile) {
            const int64_t remaining = td_->d - featureOffset;
            const int64_t current = remaining > td_->featureTile ? td_->featureTile : remaining;
            LocalTensor<X1T> x1Local = x1Queue_.template AllocTensor<X1T>();
            LocalTensor<X2T> x2Local = x2Queue_.template AllocTensor<X2T>();
            DataCopyExtParams x1Params{1, static_cast<uint32_t>(current * static_cast<int64_t>(sizeof(X1T))), 0, 0, 0};
            DataCopyExtParams x2Params{1, static_cast<uint32_t>(current * static_cast<int64_t>(sizeof(X2T))), 0, 0, 0};
            DataCopyPadExtParams<X1T> x1Pad{false, 0, 0, 0};
            DataCopyPadExtParams<X2T> x2Pad{false, 0, 0, 0};
            DataCopyPad(x1Local, x1Gm_[x1Base + featureOffset], x1Params, x1Pad);
            DataCopyPad(x2Local, x2Gm_[x2Base + featureOffset], x2Params, x2Pad);
            x1Queue_.EnQue(x1Local);
            x2Queue_.EnQue(x2Local);

            x1Local = x1Queue_.template DeQue<X1T>();
            x2Local = x2Queue_.template DeQue<X2T>();
            LocalTensor<float> partialLocal = featurePartialBuf_.Get<float>();
            auto* x1Addr = reinterpret_cast<__ubuf__ X1T*>(x1Local.GetPhyAddr());
            auto* x2Addr = reinterpret_cast<__ubuf__ X2T*>(x2Local.GetPhyAddr());
            auto* partialAddr = reinterpret_cast<__ubuf__ float*>(partialLocal.GetPhyAddr());
            const uint16_t repeatTime = static_cast<uint16_t>((current + CEL_VF_LENGTH - 1) / CEL_VF_LENGTH);
            asc_vf_call<ComputeFeaturePartialVf<X1T, X2T>>(x1Addr, x2Addr, partialAddr, static_cast<uint32_t>(current),
                                                           repeatTime);

            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            dot += partialLocal.GetValue(0);
            square1 += partialLocal.GetValue(1);
            square2 += partialLocal.GetValue(2);
            x1Queue_.FreeTensor(x1Local);
            x2Queue_.FreeTensor(x2Local);
        }

        LocalTensor<float> statsLocal = featurePartialBuf_.Get<float>();
        statsLocal.SetValue(0, dot);
        statsLocal.SetValue(1, square1);
        statsLocal.SetValue(2, square2);

        LocalTensor<TargetT> targetLocal = targetQueue_.template AllocTensor<TargetT>();
        const int64_t targetOffset = outputIndex * td_->targetOutStrides[0];
        DataCopyExtParams targetParams{1, static_cast<uint32_t>(sizeof(TargetT)), 0, 0, 0};
        DataCopyPadExtParams<TargetT> targetPad{false, 0, 0, 0};
        DataCopyPad(targetLocal, tgtGm_[targetOffset], targetParams, targetPad);
        targetQueue_.EnQue(targetLocal);
        targetLocal = targetQueue_.template DeQue<TargetT>();

        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        LocalTensor<float> lossLocal = scalarBuf_.Get<float>();
        auto* statsAddr = reinterpret_cast<__ubuf__ float*>(statsLocal.GetPhyAddr());
        auto* targetAddr = reinterpret_cast<__ubuf__ TargetT*>(targetLocal.GetPhyAddr());
        auto* lossAddr = reinterpret_cast<__ubuf__ float*>(lossLocal.GetPhyAddr());
        asc_vf_call<FinalizeLossVf<TargetT>>(statsAddr, targetAddr, lossAddr, td_->margin, td_->eps);

        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        const float loss = lossLocal.GetValue(0);
        targetQueue_.FreeTensor(targetLocal);
        return loss;
    }

    __aicore__ inline void WriteScalarOutput(int64_t outputIndex, float value)
    {
        LocalTensor<float> stage = scalarBuf_.Get<float>();
        stage.SetValue(0, value);
        event_t eventSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventSMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventSMTE3);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(yGm_[outputIndex], stage, copyParams);
        event_t eventMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMTE3S);
        WaitFlag<HardEvent::MTE3_S>(eventMTE3S);
    }

    __aicore__ inline void FinishReduction(float partial, int64_t workspaceBase = 0, float lowPart = 0.0f,
                                           bool splitCompensation = false, int64_t repeat = 1)
    {
        LocalTensor<float> stage = scalarBuf_.Get<float>();
        stage.SetValue(0, partial);
        const bool mergeCompensation = workspaceBase != 0 || splitCompensation;
        if (mergeCompensation) {
            stage.SetValue(1, lowPart);
        }
        event_t eventSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventSMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventSMTE3);
        const uint32_t transferBytes = (mergeCompensation ? 2 : 1) * sizeof(float);
        DataCopyExtParams workspaceParams{1, transferBytes, 0, 0, 0};
        DataCopyPad(wsGm_[workspaceBase + blockIdx_ * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE], stage, workspaceParams);
        const event_t eventMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMTE3S);
        WaitFlag<HardEvent::MTE3_S>(eventMTE3S);

        SyncAll();
        if (blockIdx_ != 0) {
            return;
        }

        LocalTensor<float> partials = corePartialBuf_.Get<float>();
        const int64_t workspaceElements = td_->usedCoreNum * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE;
        DataCopyExtParams inputParams{1, static_cast<uint32_t>(workspaceElements * static_cast<int64_t>(sizeof(float))),
                                      0, 0, 0};
        DataCopyPadExtParams<float> inputPad{false, 0, 0, 0};
        DataCopyPad(partials, wsGm_[workspaceBase], inputParams, inputPad);
        event_t eventMTE2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMTE2S);
        WaitFlag<HardEvent::MTE2_S>(eventMTE2S);

        float total = 0.0f;
        float totalComp = 0.0f;
        float lowTotal = 0.0f;
        float lowComp = 0.0f;
        for (int64_t core = 0; core < td_->usedCoreNum; ++core) {
            const int64_t offset = core * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE;
            AddScalarCompensated(partials.GetValue(offset), total, totalComp);
            if (splitCompensation) {
                AddScalarCompensated(partials.GetValue(offset + 1), lowTotal, lowComp);
            } else if (mergeCompensation) {
                AddScalarCompensated(partials.GetValue(offset + 1), total, totalComp);
            }
        }
        if (splitCompensation) {
            // Combine the core residuals separately so the next large partial cannot round them away.
            total += lowTotal - (totalComp + lowComp);
        }
        if (td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_MEAN) {
            // Divide once so the mean does not inherit the rounding of an FP32 reciprocal.
            total /= static_cast<float>(td_->n / repeat);
        } else if (repeat != 1) {
            total *= static_cast<float>(repeat);
        }
        WriteScalarOutput(0, total);
    }

    const CosineEmbeddingLossTilingData* td_ = nullptr;
    TPipe* pipe_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t rowBase_ = 0;
    int64_t rows_ = 0;

    GlobalTensor<X1T> x1Gm_;
    GlobalTensor<X2T> x2Gm_;
    GlobalTensor<TargetT> tgtGm_;
    GlobalTensor<float> yGm_;
    GlobalTensor<float> wsGm_;

    TQue<QuePosition::VECIN, 1> x1Queue_;
    TQue<QuePosition::VECIN, 1> x2Queue_;
    TQue<QuePosition::VECIN, 1> targetQueue_;
    TBuf<QuePosition::VECCALC> featurePartialBuf_;
    TBuf<QuePosition::VECCALC> scalarBuf_;
    TBuf<QuePosition::VECCALC> zeroScanBuf_;
    TBuf<QuePosition::VECCALC> columnX1Buf_;
    TBuf<QuePosition::VECCALC> columnX2Buf_;
    TBuf<QuePosition::VECCALC> columnX2IndicesBuf_;
    TBuf<QuePosition::VECCALC> columnTargetBuf_;
    TBuf<QuePosition::VECCALC> repeatedNoneShiftBuf_;
    IntColumnStream intStream_;
    int64_t columnSpan_ = 0;
    uint32_t columnPitch_ = CEL_OUTPUT_TILE_ROWS;
    int64_t columnBurstElements_ = 0;
    uint32_t columnBurstGapBytes_ = 0;
    int64_t repeatedNoneRows_ = 0;
    int64_t repeatedNoneTargetSlice_ = 0;
    ColumnX2Layout columnX2Layout_ = ColumnX2Layout::SCALAR;
    uint32_t columnX2Period_ = 0;
    uint32_t columnX2Group_ = 0;
    uint32_t columnX2Packed_ = 1;
    uint32_t columnX2Pitch_ = 0;
    bool columnDenseTarget_ = false;
    bool columnCheckUniform_ = false;
    int64_t streamRepeat_ = 1;
    int64_t streamSpan_ = 0;
    static constexpr int64_t CEL_ZERO_SCAN_TILE = 8192;
    static constexpr int64_t CEL_ZERO_SCAN_MIN_REUSE = 16;
    TBuf<QuePosition::VECCALC> corePartialBuf_;
};

} // namespace NsCosineEmbeddingLoss

#endif // OPS_LOSS_COSINE_EMBEDDING_LOSS_H_
