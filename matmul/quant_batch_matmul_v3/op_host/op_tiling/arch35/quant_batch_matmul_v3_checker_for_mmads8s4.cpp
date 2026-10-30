/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v3_checker_for_mmads8s4.cpp
 * \brief
 */

#include "quant_batch_matmul_v3_checker_for_mmads8s4.h"
#include "quant_batch_matmul_v3_tiling_util.h"

#include "common/op_host/op_tiling/tiling_type_mm.h"
#include "error_util.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"

namespace {
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
// X2 scale stores [..., ceilDiv(K, G), N].
constexpr size_t X2_PERGROUP_SCALE_MIN_RANK = 2;
} // namespace

namespace optiling {

using Ops::NN::FormatString;

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckDtypesInRange() const
{
    static const std::vector<ge::DataType> legalInputX1Dtypes = {ge::DT_INT8, ge::DT_INT4};
    static const std::vector<ge::DataType> legalInputX2Dtypes = {ge::DT_INT8, ge::DT_INT4};
    static const std::vector<ge::DataType> legalOutputDtypes = {ge::DT_INT8, ge::DT_FLOAT16};
    // x1 supports INT8 and INT4.
    OP_TILING_CHECK(std::find(legalInputX1Dtypes.begin(), legalInputX1Dtypes.end(), inputParams_.aDtype) ==
                        legalInputX1Dtypes.end(),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "x1", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                        "the dtype of x1 must be INT8/INT4"),
                    return false);
    // x2 supports INT8, INT4.
    OP_TILING_CHECK(std::find(legalInputX2Dtypes.begin(), legalInputX2Dtypes.end(), inputParams_.bDtype) ==
                        legalInputX2Dtypes.end(),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "x2", ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                        "the dtype of x2 must be INT8/INT4"),
                    return false);
    // Output supports INT8 and FLOAT16.
    OP_TILING_CHECK(
        std::find(legalOutputDtypes.begin(), legalOutputDtypes.end(), inputParams_.cDtype) == legalOutputDtypes.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                              "the dtype of output must be INT8/FLOAT16"),
        return false);
    // offset supports FLOAT only.
    auto offsetDesc = context_->GetOptionalInputDesc(GetOffsetIdx());
    OP_TILING_CHECK(
        (offsetDesc && context_->GetOptionalInputShape(GetOffsetIdx()) != nullptr) &&
            offsetDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "offset",
                                              ge::TypeUtils::DataTypeToSerialString(offsetDesc->GetDataType()).c_str(),
                                              "the dtype of offset must be FLOAT"),
        return false);
    // scale supports UINT64 and INT64 only.
    OP_TILING_CHECK(
        !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "scale",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                                              "the dtype of scale must be UINT64/INT64"),
        return false);
    // Dtype checking precedes quantization-mode inference, so use the parsed group attribute here.
    // Common dtype checks above apply to both routes; only X2 optional dtypes differ.
    if (inputParams_.groupSizeK != 0UL && inputParams_.aDtype == ge::DT_INT8 &&
        (inputParams_.bDtype == ge::DT_INT8 || inputParams_.bDtype == ge::DT_INT4)) {
        OP_TILING_CHECK(
            (context_->GetOptionalInputDesc(GetBiasIdx()) != nullptr &&
             context_->GetOptionalInputShape(GetBiasIdx()) != nullptr) &&
                inputParams_.biasDtype != ge::DT_FLOAT16,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                                                  "for X2 Per-Group, bias must be FLOAT16"),
            return false);
    } else {
        // bias supports INT32 only.
        OP_TILING_CHECK(
            (context_->GetOptionalInputDesc(GetBiasIdx()) != nullptr &&
             context_->GetOptionalInputShape(GetBiasIdx()) != nullptr) &&
                inputParams_.biasDtype != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                                                  "the dtype of bias must be INT32"),
            return false);
    }
    // pertokenScale is not supported in the mmad S8S4 path.
    OP_TILING_CHECK((context_->GetOptionalInputDesc(GetPertokenIdx()) != nullptr &&
                     context_->GetOptionalInputShape(GetPertokenIdx()) != nullptr),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "pertokenScale", "not null",
                                                          "pertokenScale must be null"),
                    return false);
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckABDtypes() const
{
    // INT8 x1 accepts INT8 or INT4 x2.
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT8 &&
                        !(inputParams_.bDtype == ge::DT_INT8 || inputParams_.bDtype == ge::DT_INT4),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "x2", ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                        "when the dtype of x1 is INT8, the dtype of x2 must be INT8/INT4"),
                    return false);

    // INT4 x1 accepts INT4 x2.
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT4 && inputParams_.bDtype != ge::DT_INT4,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "x2", ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                        "when the dtype of x1 is INT4, the dtype of x2 must be INT4"),
                    return false);
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckDtype() const
{
    if (!CheckDtypesInRange()) {
        return false;
    }
    if (!CheckABDtypes()) {
        return false;
    }
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckBias(const gert::StorageShape* biasShape) const
{
    // Bias must be 1D in this path.
    if (biasShape != nullptr) {
        auto biasDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(biasDimNum != 1,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "bias",
                                                                 FormatString("%zuD", biasDimNum).c_str(),
                                                                 "the shape dim of bias must be 1"),
                        return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckOffset(const gert::StorageShape* offsetShape) const
{
    if (offsetShape != nullptr) {
        // offset is only allowed when the output dtype is INT8.
        OP_TILING_CHECK(
            inputParams_.cDtype != ge::DT_INT8,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "offset", "not null",
                                                  "when the dtype of output is not INT8, offset must be null"),
            return false);
        // offset must be 1D.
        OP_TILING_CHECK(
            offsetShape->GetStorageShape().GetDimNum() != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                inputParams_.opName, "offset", FormatString("%zuD", offsetShape->GetStorageShape().GetDimNum()).c_str(),
                "the shape dim of offset must be 1"),
            return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckShapeInRangeForOptionalInputs(const gert::StorageShape* biasShape,
                                                                            const gert::StorageShape* offsetShape) const
{
    return CheckBias(biasShape) && CheckOffset(offsetShape);
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckShapeInBoundary(const gert::Shape& shape, uint32_t shapeIdx) const
{
    int64_t mul = 1;
    int64_t mulBound = 1;
    const char* dimName = shapeIdx == GetX1Idx() ? "x1" : "x2";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t curDim = shape.GetDim(i);

        // Each x1/x2 dimension must stay within INT32 range.
        OP_TILING_CHECK(
            curDim <= 0 || curDim > static_cast<int64_t>(INT32_MAX),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                inputParams_.opName, FormatString("%s dimension %zu", dimName, i).c_str(),
                std::to_string(curDim).c_str(),
                FormatString("the dimension value of %s must be within the range of [1,%d]", dimName, INT32_MAX)
                    .c_str()),
            return false);
        // PG checks the bound before the shared signed multiplication; curDim is positive above.
        if (inputParams_.isX2PerGroup) {
            OP_TILING_CHECK(mul > INT64_MAX / curDim,
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                inputParams_.opName, dimName, Shape2String(shape).c_str(),
                                "the product of shape dimensions must be within the range of INT64_MAX"),
                            return false);
        }
        // The total element count must stay within INT64 range.
        mulBound = curDim * mul;
        OP_TILING_CHECK(mulBound / curDim != mul,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            inputParams_.opName, dimName, Shape2String(shape).c_str(),
                            "the product of shape dimensions must be within the range of INT64_MAX"),
                        return false);
        mul = mulBound;
    }
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::ExtraInputCheck() const
{
    // INT4 x1 does not support transA.
    OP_TILING_CHECK(
        inputParams_.aDtype == ge::DT_INT4 && inputParams_.transA,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "trans_a", inputParams_.transA ? "true" : "false",
                                              "when the dtype of x1 is INT4, trans_a must be false"),
        return false);

    // x1 must use ND format.
    auto x1Desc = context_->GetInputDesc(GetX1Idx());
    auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
    OP_TILING_CHECK(x1Format != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(inputParams_.opName, "x1",
                                               ge::TypeUtils::FormatToSerialString(x1Format).c_str(), "ND"),
                    return false);
    // INT4 x1 requires x2 to use FRACTAL_NZ.
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    auto x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT4 && x2Format != ge::Format::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                        inputParams_.opName, "x2", ge::TypeUtils::FormatToSerialString(x2Format).c_str(),
                        "when the dtype of x1 is INT4, the format of x2 must be FRACTAL_NZ"),
                    return false);
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckX2PerGroup(const gert::Shape& scaleShape,
                                                         const gert::Shape& x2Shape) const
{
    if (!inputParams_.isX2PerGroup) {
        return true;
    }
    // Shared checks: AnalyzeInputs validates input rank [2, 6] and A/B batch broadcasting;
    // CheckDimValue has checked K equality, bias length and offset shape before this call.
    // CheckShapeInBoundary checks positive INT32 dimensions and element counts afterwards.
    // Output rank, batch prefix and M/N come from the common InferShapeForBatchMatMul path.
    // PG-specific group encoding: [groupM, groupN, groupK] = [1, 1, G], G % 64 == 0,
    // with G in [64, 65472]. Validate G before using it as the scale-group divisor.
    OP_TILING_CHECK(
        inputParams_.groupSizeM != 1UL || inputParams_.groupSizeN != 1UL,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opName, "groupSizeM, groupSizeN, groupSizeK",
            FormatString("%lu, %lu, %lu", inputParams_.groupSizeM, inputParams_.groupSizeN, inputParams_.groupSizeK)
                .c_str(),
            "for X2 Per-Group, [groupSizeM, groupSizeN, groupSizeK] must be [1, 1, G], "
            "groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32"),
        return false);
    OP_TILING_CHECK(
        inputParams_.groupSizeK % X2PerGroupConfig::GROUP_ALIGN != 0UL ||
            inputParams_.groupSizeK < X2PerGroupConfig::MIN_GROUP_SIZE ||
            inputParams_.groupSizeK > X2PerGroupConfig::MAX_GROUP_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            inputParams_.opName, "groupSizeK", std::to_string(inputParams_.groupSizeK).c_str(),
            FormatString("groupSizeK must be a multiple of %lu and within [%lu, %lu]", X2PerGroupConfig::GROUP_ALIGN,
                         X2PerGroupConfig::MIN_GROUP_SIZE, X2PerGroupConfig::MAX_GROUP_SIZE)
                .c_str()),
        return false);
    // PG dtypes: A=INT8, B=INT8/INT4, scale=UINT64, output=FLOAT16.
    OP_TILING_CHECK(
        inputParams_.cDtype != ge::DT_FLOAT16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                              "for X2 Per-Group, the dtype of output must be FLOAT16"),
        return false);
    OP_TILING_CHECK(
        inputParams_.scaleDtype != ge::DT_UINT64,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "scale",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str(),
                                              "for X2 Per-Group, the dtype of scale must be UINT64"),
        return false);
    // Unlike ordinary S8S4, this PG path requires INT8 activations for both weight types.
    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_INT8 ||
                        (inputParams_.bDtype != ge::DT_INT8 && inputParams_.bDtype != ge::DT_INT4),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "x2", ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                        "for X2 Per-Group, x1 must be INT8 and x2 must be INT8/INT4"),
                    return false);
    // PG supports only nontransposed A and NZ weights; either B transpose is accepted.
    // ExtraInputCheck separately validates the common X1 ND-format requirement.
    OP_TILING_CHECK(inputParams_.transA,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "trans_a", "true",
                                                          "for X2 Per-Group, trans_a must be false"),
                    return false);
    auto x2PerGroupDesc = context_->GetInputDesc(GetX2Idx());
    auto x2PerGroupFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(x2PerGroupDesc->GetStorageFormat()));
    OP_TILING_CHECK(x2PerGroupFormat != ge::Format::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                        inputParams_.opName, "x2", ge::TypeUtils::FormatToSerialString(x2PerGroupFormat).c_str(),
                        "for X2 Per-Group, the format of x2 must be FRACTAL_NZ"),
                    return false);
    // PG has no offset or pertoken scale. Optional bias is FP16 [N], shared across batches;
    // its dtype, rank and N length are checked by the existing dtype/shape entry.
    OP_TILING_CHECK(inputParams_.isPertoken || context_->GetOptionalInputShape(GetOffsetIdx()) != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "offset/pertokenScale", "not null",
                                                          "for X2 Per-Group, offset and pertokenScale must be null"),
                    return false);
    // INT4 requires an even innermost logical X2 axis for two-elements-per-byte packing:
    // N for [K, N] (transB=false), K for [N, K] (transB=true).
    const uint64_t packedInner = inputParams_.transB ? inputParams_.kSize : inputParams_.nSize;
    OP_TILING_CHECK(inputParams_.bDtype == ge::DT_INT4 && (packedInner & 1UL) != 0UL,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, inputParams_.transB ? "k" : "n", std::to_string(packedInner).c_str(),
                        "for INT4 x2, its innermost logical axis must be even (K if transB, otherwise N)"),
                    return false);
    // PG scale must have the same rank as B and trailing dimensions [ceilDiv(K, G), N].
    // PerChannel's one-dimensional scale rule does not apply here; M/N tails remain supported.
    const auto scaleRank = scaleShape.GetDimNum();
    const auto x2Rank = x2Shape.GetDimNum();
    OP_TILING_CHECK(scaleRank < X2_PERGROUP_SCALE_MIN_RANK || scaleRank != x2Rank,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        inputParams_.opName, "scale", FormatString("%zuD", scaleRank).c_str(),
                        FormatString("for X2 Per-Group, the shape dim of scale must be at least 2 and the same "
                                     "as that of x2 (%zuD)",
                                     x2Rank)
                            .c_str()),
                    return false);
    const uint64_t groupCnt = (inputParams_.kSize + inputParams_.groupSizeK - 1UL) / inputParams_.groupSizeK;
    const auto scaleGroupCnt = static_cast<uint64_t>(scaleShape.GetDim(scaleRank - LAST_SECOND_DIM_INDEX));
    const auto scaleN = static_cast<uint64_t>(scaleShape.GetDim(scaleRank - LAST_FIRST_DIM_INDEX));
    OP_TILING_CHECK(
        scaleGroupCnt != groupCnt || scaleN != inputParams_.nSize,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opName, "scale last two dims", FormatString("%lu, %lu", scaleGroupCnt, scaleN).c_str(),
            FormatString("the last two dims of scale must be [ceilDiv(K, G), N] = [%lu, %lu], "
                         "where K=%lu, G=%lu",
                         groupCnt, inputParams_.nSize, inputParams_.kSize, inputParams_.groupSizeK)
                .c_str()),
        return false);
    // Scale batch dimensions must equal B's exactly; scale does not broadcast independently.
    // The shared B boundary check supplies positivity/range validation for equal scale batch axes.
    for (size_t i = 0UL; i + LAST_SECOND_DIM_INDEX < scaleRank; ++i) {
        OP_TILING_CHECK(scaleShape.GetDim(i) != x2Shape.GetDim(i),
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                            inputParams_.opName, "scale batch, x2 batch",
                            FormatString("%ld, %ld", scaleShape.GetDim(i), x2Shape.GetDim(i)).c_str(),
                            "for X2 Per-Group, the batch dims of scale must be the same as those of x2"),
                        return false);
    }
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckDimValue(const gert::Shape& scaleShape,
                                                       const gert::StorageShape* biasShape,
                                                       const gert::StorageShape* offsetShape,
                                                       const std::vector<int64_t>& dimValueOfMKN) const
{
    // dimValueOfMKN stores x1 inner, x1 outer, x2 inner, and x2 outer.
    auto x1Inner = dimValueOfMKN[0];
    auto x2Inner = dimValueOfMKN[2];
    auto x2Outer = dimValueOfMKN[3];
    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? x2Inner : x2Outer);
    OP_TILING_CHECK(inputParams_.kSize != kBSize,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        inputParams_.opName, "x1 K, x2 K", FormatString("%lu, %lu", inputParams_.kSize, kBSize).c_str(),
                        "the K dimension of x1 must be equal to the K dimension of x2"),
                    return false);
    // Bias shape must match N.
    OP_TILING_CHECK(
        biasShape != nullptr && static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(0)) != inputParams_.nSize,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opName, "biasDim, n",
            FormatString("%ld, %lu", biasShape->GetStorageShape().GetDim(0), inputParams_.nSize).c_str(),
            "dimension 0 of bias must be equal to n"),
        return false);
    // offset shape must be 1 or N.
    OP_TILING_CHECK(offsetShape != nullptr &&
                        !(offsetShape->GetStorageShape().GetDim(0) == 1 ||
                          static_cast<uint64_t>(offsetShape->GetStorageShape().GetDim(0)) == inputParams_.nSize),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        inputParams_.opName, "n, offsetDim",
                        FormatString("%lu, %ld", inputParams_.nSize, offsetShape->GetStorageShape().GetDim(0)).c_str(),
                        "dimension 0 of offset must be 1 or n"),
                    return false);
    // PG shares K/bias/offset validation above; its scale and mode rules differ below.
    if (inputParams_.isX2PerGroup) {
        return CheckX2PerGroup(scaleShape, context_->GetInputShape(GetX2Idx())->GetOriginShape());
    }
    // scale must be 1D.
    OP_TILING_CHECK(scaleShape.GetDimNum() != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "scale",
                                                             FormatString("%zuD", scaleShape.GetDimNum()).c_str(),
                                                             "the shape dim of scale must be 1"),
                    return false);
    // INT8 x1 supports PerTensor and PerChannel.
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT8 && !(inputParams_.isPerTensor || inputParams_.isPerChannel),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "x1", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                        "when the dtype of x1 is INT8, the quantization mode must be PerTensor/PerChannel"),
                    return false);
    // INT4 x1 supports PerChannel only.
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT4 && !inputParams_.isPerChannel,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "x1", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                        "when the dtype of x1 is INT4, the quantization mode must be PerChannel"),
                    return false);
    // INT4 x1 requires an even inner axis.
    OP_TILING_CHECK(inputParams_.aDtype == ge::DT_INT4 && (x1Inner < 0 || x1Inner % 2 != 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "x1 last axis", std::to_string(x1Inner).c_str(),
                        "when the dtype of x1 is INT4, the last axis of x1 must be a positive even number"),
                    return false);
    // INT4 x2 requires an even inner axis.
    OP_TILING_CHECK(inputParams_.bDtype == ge::DT_INT4 && (x2Inner < 0 || x2Inner % 2 != 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "x2 last axis", std::to_string(x2Inner).c_str(),
                        "when the dtype of x2 is INT4, the last axis of x2 must be a positive even number"),
                    return false);
    return true;
}

bool QuantBatchMatmulV3Checker4MmadS8S4::CheckShape(const std::vector<gert::Shape*>& mandtoryShape,
                                                    const gert::StorageShape* biasShape,
                                                    const gert::StorageShape* /* pertokenShape */,
                                                    const std::vector<int64_t>& dimValueOfMKN) const
{
    auto& x1Shape = *mandtoryShape[0];
    auto& x2Shape = *mandtoryShape[1];
    auto& scaleShape = *mandtoryShape[2];
    auto offsetShape = context_->GetOptionalInputShape(GetOffsetIdx());
    if (!CheckShapeInRangeForOptionalInputs(biasShape, offsetShape) ||
        !CheckDimValue(scaleShape, biasShape, offsetShape, dimValueOfMKN) ||
        !CheckShapeInBoundary(x1Shape, GetX1Idx()) || !CheckShapeInBoundary(x2Shape, GetX2Idx())) {
        return false;
    }
    if (!ExtraInputCheck()) {
        return false;
    }
    return true;
}
} // namespace optiling
