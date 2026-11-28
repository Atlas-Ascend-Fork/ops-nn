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
 * \file transpose_quant_batch_mat_mul_infershape.cpp
 * \brief
 */
#include <string>

#include "error_util.h"
#include "common/op_host/matmul_common_infershape.h"
#include "matmul/common/op_host/log_format_util.h"
#include "../../../common/inc/op_host/op_const_def_nn.h"
using namespace gert;
namespace {
#define CHECK(cond, log_func, return_expr) \
    do {                                   \
        if (cond) {                        \
            log_func;                      \
            return_expr;                   \
        }                                  \
    } while (0)
const size_t X1_SCALE_IDX = 3;
const size_t X2_SCALE_IDX = 4;
const size_t DIM_2 = 2;
const size_t DIM_1 = 1;
const size_t DIM_0 = 0;
const size_t VALID_BATCH_SPLIT_FACTOR = 1;
constexpr static int64_t UNKNOWN_DIM = static_cast<int64_t>(-1);
constexpr static int64_t N_DIM_NUM = 2;
constexpr static size_t PERM_DIM_NUM = 3;

static inline bool IsMicroScaling(const ge::DataType& dtypeX1Scale, const ge::DataType& dtypeX2Scale)
{
    return dtypeX1Scale == ge::DT_FLOAT8_E8M0 && dtypeX2Scale == ge::DT_FLOAT8_E8M0;
}

static inline bool IsHIFP8(const ge::DataType& dtypeX1, const ge::DataType& dtypeX2)
{
    return dtypeX1 == ge::DT_HIFLOAT8 && dtypeX2 == ge::DT_HIFLOAT8;
}

static inline bool IsHIFP4(const ge::DataType& dtypeX1, const ge::DataType& dtypeX2)
{
    return dtypeX1 == Ops::NN::DT_HIFLOAT4 && dtypeX2 == Ops::NN::DT_HIFLOAT4;
}

static bool CheckDtypeValid(const char* nameOp, const ge::DataType& dtypeX1, const ge::DataType& dtypeX2,
                            const ge::DataType& dtypeX1Scale, const ge::DataType& dtypeX2Scale)
{
    // A2/A3 INT8 W8A8: per-token and per-channel scales are both FP32.
    if (dtypeX1 == ge::DT_INT8 && dtypeX2 == ge::DT_INT8) {
        CHECK(dtypeX1Scale != ge::DT_FLOAT || dtypeX2Scale != ge::DT_FLOAT,
              OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                  nameOp, "x1Scale, x2Scale",
                  Ops::NN::FormatString("%s, %s", Ops::Base::ToString(dtypeX1Scale).c_str(),
                                        Ops::Base::ToString(dtypeX2Scale).c_str())
                      .c_str(),
                  Ops::NN::FormatString("The dtypes of %s must be %s in INT8 W8A8 mode", "x1Scale, x2Scale", "FLOAT")
                      .c_str()),
              return false);
        return true;
    }
    // MXFP8 / MXFP4
    if (IsMicroScaling(dtypeX1Scale, dtypeX2Scale)) {
        const bool isMXFP4 = (dtypeX1 == ge::DT_FLOAT4_E2M1 && dtypeX2 == ge::DT_FLOAT4_E2M1);
        const bool isMXFP8 = (dtypeX1 == ge::DT_FLOAT8_E4M3FN && dtypeX2 == ge::DT_FLOAT8_E4M3FN);
        CHECK(
            !isMXFP4 && !isMXFP8,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(nameOp, "x1, x2",
                                                   Ops::NN::FormatString("%s, %s", Ops::Base::ToString(dtypeX1).c_str(),
                                                                         Ops::Base::ToString(dtypeX2).c_str())
                                                       .c_str(),
                                                   Ops::NN::FormatString("The dtypes of %s must be %s or %s in MX mode",
                                                                         "x1, x2", "FLOAT8_E4M3FN", "FLOAT4_E2M1")
                                                       .c_str()),
            return false);
        // HIFP4 data dtype is already identified by IsHIFP4; the scale dtype is checked below.
    } else if (!IsHIFP4(dtypeX1, dtypeX2) && !IsHIFP8(dtypeX1, dtypeX2)) {
        // FP8
        CHECK((dtypeX1 != ge::DT_FLOAT8_E4M3FN && dtypeX1 != ge::DT_FLOAT8_E5M2) ||
                  (dtypeX2 != ge::DT_FLOAT8_E4M3FN && dtypeX2 != ge::DT_FLOAT8_E5M2),
              OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                  nameOp, "x1, x2",
                  Ops::NN::FormatString("%s, %s", Ops::Base::ToString(dtypeX1).c_str(),
                                        Ops::Base::ToString(dtypeX2).c_str())
                      .c_str(),
                  Ops::NN::FormatString("The dtypes of %s must be %s or %s", "x1, x2", "FLOAT8_E4M3FN", "FLOAT8_E5M2")
                      .c_str()),
              return false);
    }
    // MXFP8/FP8/HIFP4 scale dtype check
    if (IsHIFP4(dtypeX1, dtypeX2)) {
        CHECK(
            dtypeX1Scale != Ops::NN::DT_HIFLOAT4_SCALE || dtypeX2Scale != Ops::NN::DT_HIFLOAT4_SCALE,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                nameOp, "x1Scale, x2Scale",
                Ops::NN::FormatString("%s, %s", Ops::Base::ToString(dtypeX1Scale).c_str(),
                                      Ops::Base::ToString(dtypeX2Scale).c_str())
                    .c_str(),
                Ops::NN::FormatString("The dtypes of %s must be %s in HIFP4 mode", "x1Scale, x2Scale", "HIFLOAT4_SCALE")
                    .c_str()),
            return false);
    } else if (!IsHIFP8(dtypeX1, dtypeX2)) {
        CHECK((dtypeX1Scale != ge::DT_FLOAT && dtypeX1Scale != ge::DT_FLOAT8_E8M0) ||
                  (dtypeX2Scale != ge::DT_FLOAT && dtypeX2Scale != ge::DT_FLOAT8_E8M0),
              OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                  nameOp, "x1Scale, x2Scale",
                  Ops::NN::FormatString("%s, %s", Ops::Base::ToString(dtypeX1Scale).c_str(),
                                        Ops::Base::ToString(dtypeX2Scale).c_str())
                      .c_str(),
                  Ops::NN::FormatString("The dtypes of %s must be %s or %s", "x1Scale, x2Scale", "FLOAT", "FLOAT8_E8M0")
                      .c_str()),
              return false);
    } else {
        CHECK(dtypeX2Scale != ge::DT_UINT64,
              OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                  nameOp, "x2Scale", Ops::Base::ToString(dtypeX2Scale).c_str(),
                  Ops::NN::FormatString("The dtype of %s must be %s in HIFP8 mode", "x2Scale", "UINT64").c_str()),
              return false);
    }
    return true;
}

static bool CheckPerm(const char* nameOp, const TypedContinuousVector<int64_t>* permX1,
                      const TypedContinuousVector<int64_t>* permX2, const TypedContinuousVector<int64_t>* permY)
{
    CHECK(permX1 == nullptr, OP_LOGE_WITH_INVALID_INPUT(nameOp, "permX1"), return false);
    CHECK(permX2 == nullptr, OP_LOGE_WITH_INVALID_INPUT(nameOp, "permX2"), return false);
    CHECK(permY == nullptr, OP_LOGE_WITH_INVALID_INPUT(nameOp, "permY"), return false);

    if (permX1->GetSize() != PERM_DIM_NUM) {
        OP_LOGE_FOR_INVALID_LISTSIZE(nameOp, "permX1", std::to_string(permX1->GetSize()).c_str(),
                                     std::to_string(PERM_DIM_NUM).c_str());
        return false;
    }
    if (permX2->GetSize() != PERM_DIM_NUM) {
        OP_LOGE_FOR_INVALID_LISTSIZE(nameOp, "permX2", std::to_string(permX2->GetSize()).c_str(),
                                     std::to_string(PERM_DIM_NUM).c_str());
        return false;
    }
    if (permY->GetSize() != PERM_DIM_NUM) {
        OP_LOGE_FOR_INVALID_LISTSIZE(nameOp, "permY", std::to_string(permY->GetSize()).c_str(),
                                     std::to_string(PERM_DIM_NUM).c_str());
        return false;
    }
    const auto permX1Data = permX1->GetData();
    const auto permX2Data = permX2->GetData();
    const auto permYData = permY->GetData();
    CHECK(permX1Data == nullptr || permX2Data == nullptr || permYData == nullptr,
          OP_LOGE(nameOp, "perm list data is nullptr"), return false);

    const bool checkPermX1 = (permX1Data[0] == 1 && permX1Data[1] == 0 && permX1Data[2] == 2); // permX1 is [1,0,2]
    CHECK(!checkPermX1,
          OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
              nameOp, "permX1",
              Ops::NN::FormatString("[%ld, %ld, %ld]", permX1Data[0], permX1Data[1], permX1Data[2]).c_str(),
              Ops::NN::FormatString("The value of %s must be %s", "permX1", "[1,0,2]").c_str()),
          return false);

    const bool checkPermX2 = (permX2Data[0] == 0 && permX2Data[1] == 1 && permX2Data[2] == 2) ||
                             (permX2Data[0] == 0 && permX2Data[1] == 2 && permX2Data[2] == 1);
    CHECK(!checkPermX2,
          OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
              nameOp, "permX2",
              Ops::NN::FormatString("[%ld, %ld, %ld]", permX2Data[0], permX2Data[1], permX2Data[2]).c_str(),
              Ops::NN::FormatString("The value of %s must be %s", "permX2", "{[0,1,2], [0,2,1]}").c_str()),
          return false);

    const bool checkPermY = (permYData[0] == 1 && permYData[1] == 0 && permYData[2] == 2);
    CHECK(
        !checkPermY,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nameOp, "permY", Ops::NN::FormatString("[%ld, %ld, %ld]", permYData[0], permYData[1], permYData[2]).c_str(),
            Ops::NN::FormatString("The value of %s must be %s", "permY", "[1,0,2]").c_str()),
        return false);
    return true;
}

static bool TransposeShape(const Shape& src, const TypedContinuousVector<int64_t>& perm, Shape& dst)
{
    if (perm.GetSize() == 0) {
        dst = src;
        return true;
    }

    if (src.GetDimNum() != perm.GetSize() || dst.GetDimNum() != 0) {
        return false;
    }

    for (size_t idxDst = 0; idxDst < perm.GetSize(); ++idxDst) {
        dst.AppendDim(src.GetDim(*(perm.GetData() + idxDst)));
    }

    return true;
}

static ge::graphStatus SetShapeY(const char* nameOp, Shape& shapeY, const Shape& shapeX1Transposed,
                                 const Shape& shapeX2Transposed)
{
    constexpr int64_t expectedDim = 3; // the dim of y should be 3
    shapeY.SetDimNum(expectedDim);
    const int64_t mDim = shapeX1Transposed.GetDim(1);
    const int64_t batchDim = shapeX1Transposed.GetDim(0);
    const int64_t nDim = shapeX2Transposed.GetDim(2);
    // m, b, n
    shapeY.SetDim(0, mDim);
    shapeY.SetDim(1, batchDim);
    shapeY.SetDim(N_DIM_NUM, nDim);
    OP_LOGD(nameOp, "y_shape: %s", Ops::Base::ToString(shapeY).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForTransposeQuantBatchMatMul(InferShapeContext* context)
{
    CHECK(context == nullptr, OP_LOGE("TransposeQuantBatchMatMul", "context is null"), return ge::GRAPH_FAILED);

    auto shapeX1 = context->GetInputShape(0);
    auto shapeX2 = context->GetInputShape(1);
    auto shapeY = context->GetOutputShape(0);
    auto attrs = context->GetAttrs();
    const char* nameOp = context->GetNodeName();
    CHECK(nameOp == nullptr, OP_LOGE("TransposeQuantBatchMatMul", "node name is null"), return ge::GRAPH_FAILED);
    CHECK(shapeX1 == nullptr || shapeX2 == nullptr || shapeY == nullptr || attrs == nullptr,
          OP_LOGE(nameOp, "input/output shape or attrs is nullptr"), return ge::GRAPH_FAILED);

    const auto dtype = attrs->GetAttrPointer<int64_t>(0); // dtype index is 0
    CHECK(dtype == nullptr, OP_LOGE_WITH_INVALID_INPUT(nameOp, "dtype"), return ge::GRAPH_FAILED);
    auto tensorX1 = context->GetInputDesc(0);
    auto tensorX2 = context->GetInputDesc(1);
    CHECK(tensorX1 == nullptr || tensorX2 == nullptr, OP_LOGE(nameOp, "x1 or x2 desc is nullptr"),
          return ge::GRAPH_FAILED);
    const ge::DataType dtypeX1 = tensorX1->GetDataType();
    const ge::DataType dtypeX2 = tensorX2->GetDataType();
    const bool isHIFP8 = IsHIFP8(dtypeX1, dtypeX2);
    auto tensorX1Scale = context->GetOptionalInputDesc(X1_SCALE_IDX);
    auto tensorX2Scale = context->GetOptionalInputDesc(X2_SCALE_IDX);
    CHECK(!isHIFP8 && tensorX1Scale == nullptr, OP_LOGE_WITH_INVALID_INPUT(nameOp, "x1Scale"), return ge::GRAPH_FAILED);
    CHECK(tensorX2Scale == nullptr, OP_LOGE_WITH_INVALID_INPUT(nameOp, "x2Scale"), return ge::GRAPH_FAILED);

    const ge::DataType dtypeX1Scale = isHIFP8 ? ge::DT_UINT64 : tensorX1Scale->GetDataType();
    const ge::DataType dtypeX2Scale = tensorX2Scale->GetDataType();
    CHECK(!CheckDtypeValid(nameOp, dtypeX1, dtypeX2, dtypeX1Scale, dtypeX2Scale),
          OP_LOGE_WITHOUT_REPORT(nameOp, "check dtype failed"), return ge::GRAPH_FAILED);

    const auto permX1 = attrs->GetListInt(2);                        // permX1 index is 2
    const auto permX2 = attrs->GetListInt(3);                        // permX2 index is 3
    const auto permY = attrs->GetListInt(4);                         // permY index is 4
    const auto batchSplitFactor = attrs->GetAttrPointer<int32_t>(5); // batchSplitFactor index is 5
    CHECK(!CheckPerm(nameOp, permX1, permX2, permY), OP_LOGE_WITHOUT_REPORT(nameOp, "check perm failed"),
          return ge::GRAPH_FAILED);
    if (dtypeX1 == ge::DT_INT8 && dtypeX2 == ge::DT_INT8) {
        const auto permX2Data = permX2->GetData();
        const bool kcPerm = permX2Data[0] == 0 && permX2Data[1] == 1 && permX2Data[2] == 2;
        CHECK(!kcPerm,
              OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                  nameOp, "permX2",
                  Ops::NN::FormatString("[%ld, %ld, %ld]", permX2Data[0], permX2Data[1], permX2Data[2]).c_str(),
                  Ops::NN::FormatString("The value of %s must be %s in INT8 W8A8 mode", "permX2", "[0,1,2]").c_str()),
              return ge::GRAPH_FAILED);
    }
    Shape shapeX1Transposed;
    Shape shapeX2Transposed;
    CHECK(!TransposeShape(*shapeX1, *permX1, shapeX1Transposed), OP_LOGE(nameOp, "failed to transpose shape of x1"),
          return ge::GRAPH_FAILED);
    CHECK(!TransposeShape(*shapeX2, *permX2, shapeX2Transposed), OP_LOGE(nameOp, "failed to transpose shape of x2"),
          return ge::GRAPH_FAILED);

    // Check x1 and x2 shape
    CHECK((shapeX1Transposed.GetDimNum() != PERM_DIM_NUM) || (shapeX2Transposed.GetDimNum() != PERM_DIM_NUM),
          OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
              nameOp, "x1, x2",
              Ops::NN::FormatString("%zu, %zu", shapeX1Transposed.GetDimNum(), shapeX2Transposed.GetDimNum()).c_str(),
              Ops::NN::FormatString("The shape dims of %s must be %zu", "x1, x2", PERM_DIM_NUM).c_str()),
          return ge::GRAPH_FAILED);
    // Check x1 and x2 k-axis
    const int64_t kX1 = shapeX1Transposed.GetDim(DIM_2);
    const int64_t kX2 = shapeX2Transposed.GetDim(DIM_1);
    CHECK(kX1 != UNKNOWN_DIM && kX2 != UNKNOWN_DIM && kX1 != kX2,
          OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
              nameOp, "x1, x2",
              Ops::NN::FormatString("%s, %s", Ops::Base::ToString(shapeX1Transposed).c_str(),
                                    Ops::Base::ToString(shapeX2Transposed).c_str())
                  .c_str(),
              Ops::NN::FormatString("%s of %s must be equal", "K-axis", "x1, x2").c_str()),
          return ge::GRAPH_FAILED);
    // Check x1 and x2 batch-axis
    const int64_t batchX1 = shapeX1Transposed.GetDim(DIM_0);
    const int64_t batchX2 = shapeX2Transposed.GetDim(DIM_0);
    CHECK(batchX1 != UNKNOWN_DIM && batchX2 != UNKNOWN_DIM && batchX1 != batchX2,
          OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
              nameOp, "x1, x2",
              Ops::NN::FormatString("%s, %s", Ops::Base::ToString(shapeX1Transposed).c_str(),
                                    Ops::Base::ToString(shapeX2Transposed).c_str())
                  .c_str(),
              Ops::NN::FormatString("%s of %s must be equal", "Batch-axis", "x1, x2").c_str()),
          return ge::GRAPH_FAILED);

    CHECK(batchSplitFactor == nullptr, OP_LOGE_WITH_INVALID_INPUT(nameOp, "batchSplitFactor"), return ge::GRAPH_FAILED);
    CHECK(
        *batchSplitFactor != static_cast<int32_t>(VALID_BATCH_SPLIT_FACTOR),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nameOp, "batchSplitFactor", std::to_string(*batchSplitFactor).c_str(),
            Ops::NN::FormatString("The value of %s must be %zu", "batchSplitFactor", VALID_BATCH_SPLIT_FACTOR).c_str()),
        return ge::GRAPH_FAILED);

    // Set shapeY
    const ge::graphStatus ret = SetShapeY(nameOp, *shapeY, shapeX1Transposed, shapeX2Transposed);
    CHECK(ret != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(nameOp, "set shapeY failed"), return ge::GRAPH_FAILED);
    // no need to SetDataType in runtime
    return ge::GRAPH_SUCCESS;
}

} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(TransposeQuantBatchMatMul).InferShape(InferShapeForTransposeQuantBatchMatMul);
}
