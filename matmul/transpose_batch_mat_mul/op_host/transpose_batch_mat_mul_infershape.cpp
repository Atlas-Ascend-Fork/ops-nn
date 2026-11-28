/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_batch_mat_mul_infershape.cpp
 * \brief
 */
#include <string>

#include "error_util.h"
#include "common/op_host/matmul_common_infershape.h"
using namespace gert;
namespace {
#define CHECK(cond, log_func, return_expr) \
    do {                                   \
        if (cond) {                        \
            log_func;                      \
            return_expr;                   \
        }                                  \
    } while (0)
const size_t TRANSPOSE_BATCH_MAT_MUL_SCALE_IDX = 3;
constexpr static int64_t UNKNOWN_DIM = static_cast<int64_t>(-1);
constexpr static int64_t UNKNOWN_DIM_NUM = static_cast<int64_t>(-2);
constexpr static int64_t N_DIM_NUM = 2;

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

static bool CheckIsUnknownDimNum(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == UNKNOWN_DIM_NUM;
}

static inline bool SafeMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs == UNKNOWN_DIM || rhs == UNKNOWN_DIM) {
        result = UNKNOWN_DIM;
        return true;
    }
    return !__builtin_mul_overflow(lhs, rhs, &result);
}

static inline int64_t SafeDiv(int64_t lhs, int64_t rhs) { return (lhs == UNKNOWN_DIM) ? UNKNOWN_DIM : lhs / rhs; }

static ge::graphStatus CheckScaleForTransposeBatchMatMul(const InferShapeContext* context,
                                                         const Shape& shapeX2Transposed)
{
    const Shape* shapeScale = context->GetOptionalInputShape(TRANSPOSE_BATCH_MAT_MUL_SCALE_IDX);
    if (shapeScale == nullptr || CheckIsUnknownDimNum(*shapeScale)) {
        return ge::GRAPH_SUCCESS;
    }
    if (shapeScale->GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "scale",
                                                 std::to_string(shapeScale->GetDimNum()).c_str(),
                                                 "The ViewShape dim of scale must be 1");
        return ge::GRAPH_FAILED;
    }
    int64_t expectedScaleDim = 0;
    CHECK(!SafeMul(shapeX2Transposed.GetDim(0), shapeX2Transposed.GetDim(2), expectedScaleDim),
          OP_LOGE(context->GetNodeName(), "The dimension of n * b overflows int64."), return ge::GRAPH_FAILED);
    if (expectedScaleDim != UNKNOWN_DIM && shapeScale->GetDim(0) != expectedScaleDim) {
        const std::string reason = "The first dimension of scale must be equal to n * b (" +
                                   std::to_string(expectedScaleDim) + ")";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "scale", Ops::Base::ToString(*shapeScale).c_str(),
                                              reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetShapeY(Shape& shapeY, const Shape& shapeX1Transposed, const Shape& shapeX2Transposed,
                                 const TypedContinuousVector<int64_t>& permY, const int32_t batchSplitFactor,
                                 bool hasScaleShape, const char* nameOp)
{
    constexpr int EXPECTED_DIM = 3; // the dim of y should be 3
    shapeY.SetDimNum(EXPECTED_DIM);
    auto mDim = shapeX1Transposed.GetDim(1);
    auto batchDim = shapeX1Transposed.GetDim(0);
    auto nDim = shapeX2Transposed.GetDim(2);
    // permY=[0,1,2]: the output layout is batch-first (B,M,N); permY=[1,0,2]: batch-in-the-middle (M,B,N)
    auto permYData = permY.GetData();
    bool isBatchFirstY = (permYData[0] == 0 && permYData[1] == 1 && permYData[2] == 2);
    if (isBatchFirstY) {
        shapeY.SetDim(0, batchDim);
        shapeY.SetDim(1, mDim);
        shapeY.SetDim(N_DIM_NUM, nDim);
    } else {
        shapeY.SetDim(0, mDim);
        shapeY.SetDim(1, batchDim);
        shapeY.SetDim(N_DIM_NUM, nDim);
    }

    if (hasScaleShape) {
        shapeY.SetDim(0, mDim);
        shapeY.SetDim(1, 1);
        int64_t scaleBatchN = 0;
        CHECK(!SafeMul(batchDim, nDim, scaleBatchN),
              OP_LOGE(nameOp, "batch * n overflows int64, the shape is out of range."), return ge::GRAPH_FAILED);
        shapeY.SetDim(N_DIM_NUM, scaleBatchN);
    }

    if (batchSplitFactor > 1) {
        int64_t m = shapeY.GetDim(0);
        int64_t batchN = 0;
        CHECK(!SafeMul(shapeY.GetDim(1), shapeY.GetDim(2), batchN),
              OP_LOGE(nameOp, "batch * n overflows int64, the shape is out of range."), return ge::GRAPH_FAILED);
        shapeY.SetDim(0, batchSplitFactor);
        shapeY.SetDim(1, m);
        shapeY.SetDim(2, SafeDiv(batchN, batchSplitFactor)); // 2 is dim index
    }

    // The transposed result itself is not consumed, but the call validates that permY is a valid
    // permutation of the y rank, so it must be kept.
    Shape shapeYTransposed;
    CHECK(!TransposeShape(shapeY, permY, shapeYTransposed), OP_LOGE(nameOp, "failed to transpose the shape of y."),
          return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckPermForTransposeBatchMatMul(const char* nameOp,
                                                        const TypedContinuousVector<int64_t>& permX1,
                                                        const TypedContinuousVector<int64_t>& permX2,
                                                        const TypedContinuousVector<int64_t>& permY)
{
    constexpr size_t PERM_DIM_NUM = 3;
    CHECK(permX1.GetSize() != PERM_DIM_NUM || permX2.GetSize() != PERM_DIM_NUM || permY.GetSize() != PERM_DIM_NUM,
          OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON(nameOp, "permX1, permX2, permY",
                                                   std::to_string(permX1.GetSize()).c_str(),
                                                   std::to_string(PERM_DIM_NUM).c_str()),
          return ge::GRAPH_FAILED);

    const auto permX1Attr = permX1.GetData();
    auto checkPermX1 = (*permX1Attr == 1 && *(permX1Attr + 1) == 0 && *(permX1Attr + 2) == 2) ||
                       (*permX1Attr == 0 && *(permX1Attr + 1) == 1 && *(permX1Attr + 2) == 2) ||
                       (*permX1Attr == 0 && *(permX1Attr + 1) == 2 && *(permX1Attr + 2) == 1);
    CHECK(!checkPermX1,
          OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nameOp, "permX1", "invalid perm list",
                                                 "permX1 must be one of {1, 0, 2}, {0, 1, 2}, {0, 2, 1}"),
          return ge::GRAPH_FAILED);

    const auto permX2Attr = permX2.GetData();
    auto checkPermX2 = (*permX2Attr == 0 && *(permX2Attr + 1) == 1 && *(permX2Attr + 2) == 2) ||
                       (*permX2Attr == 0 && *(permX2Attr + 1) == 2 && *(permX2Attr + 2) == 1) ||
                       (*permX2Attr == 1 && *(permX2Attr + 1) == 0 && *(permX2Attr + 2) == 2) ||
                       (*permX2Attr == 1 && *(permX2Attr + 1) == 2 && *(permX2Attr + 2) == 0);
    CHECK(!checkPermX2,
          OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nameOp, "permX2", "invalid perm list",
                                                 "permX2 must be one of {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}"),
          return ge::GRAPH_FAILED);

    const auto permYAttr = permY.GetData();
    auto checkPermY = (*permYAttr == 1 && *(permYAttr + 1) == 0 && *(permYAttr + 2) == 2) ||
                      (*permYAttr == 0 && *(permYAttr + 1) == 1 && *(permYAttr + 2) == 2);
    CHECK(!checkPermY,
          OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nameOp, "permY", "invalid perm list",
                                                 "permY must be one of {1, 0, 2}, {0, 1, 2}"),
          return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForTransposeBatchMatMul(InferShapeContext* context)
{
    CHECK(context == nullptr, OP_LOGE("TransposeBatchMatMul", "context is null"), return ge::GRAPH_FAILED);

    auto shapeX1 = context->GetInputShape(0);
    auto shapeX2 = context->GetInputShape(1);
    auto shapeY = context->GetOutputShape(0);
    auto attrs = context->GetAttrs();
    auto nameOp = context->GetNodeName();
    CHECK(shapeX1 == nullptr || shapeX2 == nullptr || shapeY == nullptr || attrs == nullptr,
          OP_LOGE(nameOp, "shape or attrs is null"), return ge::GRAPH_FAILED);

    const auto permX1 = attrs->GetListInt(0);
    const auto permX2 = attrs->GetListInt(1);
    const auto permY = attrs->GetListInt(2);
    const auto batchSplitFactor = attrs->GetAttrPointer<int32_t>(4); // batchSplitFactor index is 4
    CHECK(permX1 == nullptr || permX2 == nullptr || permY == nullptr || batchSplitFactor == nullptr,
          OP_LOGE(nameOp, "perm or batchSplitFactor attr is null"), return ge::GRAPH_FAILED);

    CHECK(CheckPermForTransposeBatchMatMul(nameOp, *permX1, *permX2, *permY) != ge::GRAPH_SUCCESS,
          OP_LOGE_WITHOUT_REPORT(nameOp, "check perm failed"), return ge::GRAPH_FAILED);

    if (CheckIsUnknownDimNum(*shapeX1) || CheckIsUnknownDimNum(*shapeX2)) {
        shapeY->SetDimNum(1);
        shapeY->SetDim(0, UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }

    // The bmn output (permY = [0, 1, 2]) is mutually exclusive with batchSplitFactor.
    const auto permYData = permY->GetData();
    bool isBatchFirstY = (permYData[0] == 0 && permYData[1] == 1 && permYData[2] == 2);
    CHECK(isBatchFirstY && *batchSplitFactor > 1,
          OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nameOp, "permY, batchSplitFactor", "conflicting combination",
                                                 "permY = {0, 1, 2} is mutually exclusive with batchSplitFactor > 1"),
          return ge::GRAPH_FAILED);
    Shape shapeX1Transposed;
    Shape shapeX2Transposed;
    CHECK(!TransposeShape(*shapeX1, *permX1, shapeX1Transposed), OP_LOGE(nameOp, "failed to transpose shape of x1"),
          return ge::GRAPH_FAILED);
    CHECK(!TransposeShape(*shapeX2, *permX2, shapeX2Transposed), OP_LOGE(nameOp, "failed to transpose shape of x2"),
          return ge::GRAPH_FAILED);

    const int64_t kDimX1 = shapeX1Transposed.GetDim(2);
    const int64_t kDimX2 = shapeX2Transposed.GetDim(1);
    CHECK(kDimX1 != UNKNOWN_DIM && kDimX2 != UNKNOWN_DIM && kDimX1 != kDimX2,
          OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nameOp, "x1, x2", Ops::Base::ToString(shapeX1Transposed).c_str(),
                                                 "The k-axis of x1 and x2 must be equal"),
          return ge::GRAPH_FAILED);

    const int64_t batchDimX1 = shapeX1Transposed.GetDim(0);
    const int64_t batchDimX2 = shapeX2Transposed.GetDim(0);
    CHECK(batchDimX1 != UNKNOWN_DIM && batchDimX2 != UNKNOWN_DIM && batchDimX1 != batchDimX2,
          OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nameOp, "x1, x2", Ops::Base::ToString(shapeX1Transposed).c_str(),
                                                 "The batch-axis of x1 and x2 must be equal"),
          return ge::GRAPH_FAILED);

    CHECK(*batchSplitFactor <= 0 ||
              (batchDimX1 != UNKNOWN_DIM && (*batchSplitFactor > batchDimX1 || batchDimX1 % *batchSplitFactor != 0)),
          OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nameOp, "batchSplitFactor", std::to_string(*batchSplitFactor).c_str(),
                                                "batchSplitFactor must be positive and divide the batch size"),
          return ge::GRAPH_FAILED);

    auto* shapeScale = context->GetOptionalInputShape(TRANSPOSE_BATCH_MAT_MUL_SCALE_IDX);
    if (shapeScale != nullptr) {
        // The bmn output (permY = [0, 1, 2]) is mutually exclusive with scale.
        CHECK(isBatchFirstY,
              OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nameOp, "permY, scale", "conflicting combination",
                                                     "permY = {0, 1, 2} is mutually exclusive with scale"),
              return ge::GRAPH_FAILED);
        CHECK(
            *batchSplitFactor != 1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nameOp, "batchSplitFactor", std::to_string(*batchSplitFactor).c_str(),
                                                  "batchSplitFactor must be 1 when scale is not null"),
            return ge::GRAPH_FAILED);
        int64_t expectedScaleDim = 0;
        CHECK(!SafeMul(batchDimX1, shapeX2Transposed.GetDim(2), expectedScaleDim),
              OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nameOp, "scale", "int64 overflow",
                                                     "The dimension of n * b overflows int64"),
              return ge::GRAPH_FAILED);
        if (!CheckIsUnknownDimNum(*shapeScale) && expectedScaleDim != UNKNOWN_DIM) {
            if (shapeScale->GetDimNum() != 1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nameOp, "scale",
                                                         std::to_string(shapeScale->GetDimNum()).c_str(),
                                                         "The ViewShape dim of scale must be 1");
                return ge::GRAPH_FAILED;
            }
            if (shapeScale->GetDim(0) != expectedScaleDim) {
                const std::string reason = "The first dimension of scale must be equal to n * b (" +
                                           std::to_string(expectedScaleDim) + ")";
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nameOp, "scale", Ops::Base::ToString(*shapeScale).c_str(),
                                                      reason.c_str());
                return ge::GRAPH_FAILED;
            }
        }
        auto tensorX1 = context->GetInputDesc(0);
        auto tensorX2 = context->GetInputDesc(1);
        CHECK(tensorX1 == nullptr || tensorX2 == nullptr, OP_LOGE(nameOp, "x1 or x2 desc is null."),
              return ge::GRAPH_FAILED);
        ge::DataType dtypeX1 = tensorX1->GetDataType();
        ge::DataType dtypeX2 = tensorX2->GetDataType();
        CHECK(dtypeX1 != ge::DT_FLOAT16 || dtypeX2 != ge::DT_FLOAT16,
              OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(nameOp, "x1, x2", "unsupported dtype",
                                                     "float16 when the scale takes effect"),
              return ge::GRAPH_FAILED);
    }
    ge::graphStatus ret = SetShapeY(*shapeY, shapeX1Transposed, shapeX2Transposed, *permY, *batchSplitFactor,
                                    shapeScale != nullptr, nameOp);
    OP_LOGD(nameOp, "y_shape: %s", Ops::Base::ToString(*shapeY).c_str());
    CHECK(ret != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(nameOp, "set shape y failed"), return ge::GRAPH_FAILED);

    CHECK(CheckScaleForTransposeBatchMatMul(context, shapeX2Transposed) != ge::GRAPH_SUCCESS,
          OP_LOGE_WITHOUT_REPORT(nameOp, "scale shape check failed"), return ge::GRAPH_FAILED);
    // no need to SetDataType in runtime
    return ge::GRAPH_SUCCESS;
}

} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(TransposeBatchMatMul).InferShape(InferShapeForTransposeBatchMatMul);
}
