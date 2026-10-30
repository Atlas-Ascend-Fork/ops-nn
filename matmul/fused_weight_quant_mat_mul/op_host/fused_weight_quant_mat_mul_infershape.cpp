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
 * \file fused_weight_quant_mat_mul_infershape.cpp
 * \brief FusedWeightQuantMatMul shape 推导。
 *        输出 shape 与 WeightQuantBatchMatmulV2 完全一致（relu 不改 shape/dtype），
 *        底座 InferShapeForWeightQuantBatchMatmulV2 为文件内 static 实现，此处复制并追加
 *        fused_op_type 合法性校验（对齐 fused_quant_mat_mul_infershape.cpp 的做法），不侵入底座。
 */

#include <cstring>
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "matmul/fused_weight_quant_mat_mul/common/fused_weight_quant_mat_mul_constants.h"

namespace {
constexpr size_t MINIMUM_SHAPE_SIZE = 2UL;
constexpr size_t MAXNUM_SHAPE_SIZE = 6UL;
constexpr int64_t UNKNOWN_DIM_NUM = -2;
const char* NODE_NAME = "FusedWeightQuantMatMul";
// 属性索引与底座前 4 个属性一致：transpose_x(0)/transpose_weight(1)/antiquant_group_size(2)/dtype(3)
constexpr size_t TRANSPOSE_X_INDEX = 0UL;
constexpr size_t TRANSPOSE_WEIGHT_INDEX = 1UL;
constexpr size_t FUSED_OP_TYPE_INDEX = 4UL;

static bool BroadcastBatchDim(const int64_t dimA, const int64_t dimB, int64_t& dim)
{
    if (dimA > 1 && dimB > 1) {
        if (dimA != dimB) {
            return false;
        }
        dim = dimA;
        return true;
    }
    dim = std::max(dimA, dimB);
    return true;
}

bool CheckIsUnknownDimNum(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == UNKNOWN_DIM_NUM;
}

static ge::graphStatus SetBroadcastShape(const gert::Shape* xShape, const gert::Shape* weightShape, gert::Shape* yShape)
{
    size_t numDimA = xShape->GetDimNum();
    size_t numDimB = weightShape->GetDimNum();
    size_t outDimNum = std::max(numDimA, numDimB);
    int64_t broadcastDim = 0;
    if (numDimA == numDimB) {
        for (size_t i = 0; i < outDimNum - MINIMUM_SHAPE_SIZE; i++) {
            OP_CHECK_IF(
                !BroadcastBatchDim(xShape->GetDim(i), weightShape->GetDim(i), broadcastDim),
                OP_LOGE(NODE_NAME,
                        "batch of xShape and batch of wShape can't broadcast, x shape is %s, weight shape is %s",
                        Ops::Base::ToString(*xShape).c_str(), Ops::Base::ToString(*weightShape).c_str()),
                return ge::GRAPH_FAILED);
            yShape->SetDim(i, broadcastDim);
        }
    } else {
        auto longerShape = numDimA > numDimB ? xShape : weightShape;
        auto smallShape = numDimA < numDimB ? xShape : weightShape;
        uint64_t batchOffset = longerShape->GetDimNum() - smallShape->GetDimNum();
        for (size_t i = 0; i < batchOffset; i++) {
            yShape->SetDim(i, longerShape->GetDim(i));
        }
        for (size_t i = batchOffset; i < outDimNum - MINIMUM_SHAPE_SIZE; i++) {
            OP_CHECK_IF(
                !BroadcastBatchDim(longerShape->GetDim(i), smallShape->GetDim(i - batchOffset), broadcastDim),
                OP_LOGE(NODE_NAME,
                        "batch of xShape and batch of wShape can't broadcast, x shape is %s, weight shape is %s",
                        Ops::Base::ToString(*xShape).c_str(), Ops::Base::ToString(*weightShape).c_str()),
                return ge::GRAPH_FAILED);
            yShape->SetDim(i, broadcastDim);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ShapeCheckAndInfer(const gert::Shape* xShape, const gert::Shape* weightShape,
                                          gert::Shape* yShape, const bool transposeX, const bool transposeWeight)
{
    size_t numDimA = xShape->GetDimNum();
    size_t numDimB = weightShape->GetDimNum();
    OP_CHECK_IF(std::min(numDimA, numDimB) < MINIMUM_SHAPE_SIZE || std::max(numDimA, numDimB) > MAXNUM_SHAPE_SIZE,
                OP_LOGE(NODE_NAME, "The shape of x and weight must >= 2 and <=6, which are %s and %s",
                        Ops::Base::ToString(*xShape).c_str(), Ops::Base::ToString(*weightShape).c_str()),
                return ge::GRAPH_FAILED);
    size_t mIdx = transposeX ? 1UL : 2UL;
    size_t kaIdx = transposeX ? 2UL : 1UL;
    size_t kbIdx = transposeWeight ? 1UL : 2UL;
    size_t nIdx = transposeWeight ? 2UL : 1UL;
    int64_t xK = xShape->GetDim(numDimA - kaIdx);
    int64_t weightK = weightShape->GetDim(numDimB - kbIdx);
    int64_t weightN = weightShape->GetDim(numDimB - nIdx);
    OP_CHECK_IF(xK != weightK && xK > 0 && weightK > 0, OP_LOGE(NODE_NAME, "Ka[%ld] != Kb[%ld]", xK, weightK),
                return ge::GRAPH_FAILED);
    size_t outDimNum = std::max(numDimA, numDimB);
    yShape->SetDimNum(outDimNum);
    yShape->SetDim(outDimNum - 2, xShape->GetDim(numDimA - mIdx)); // 2:设置yShape m
    yShape->SetDim(outDimNum - 1, weightN);                        // 1:设置yShape n
    return SetBroadcastShape(xShape, weightShape, yShape);
}

static ge::graphStatus InferShapeForFusedWeightQuantMatMul(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return ge::GRAPH_FAILED);
    auto xShape = context->GetInputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto weightShape = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    auto yShape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* transposeX = attrs->GetAttrPointer<bool>(TRANSPOSE_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, transposeX);
    const bool* transposeWeight = attrs->GetAttrPointer<bool>(TRANSPOSE_WEIGHT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, transposeWeight);
    const char* fusedOpType = attrs->GetAttrPointer<char>(FUSED_OP_TYPE_INDEX);
    OP_CHECK_IF(fusedOpType == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "fused_op_type is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        strcmp(fusedOpType, ops::fused_weight_quant_mat_mul::FUSED_OP_TYPE_RELU) != 0,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "fused_op_type only supports relu, but is %s", fusedOpType),
        return ge::GRAPH_FAILED);
    if (CheckIsUnknownDimNum(*xShape) || CheckIsUnknownDimNum(*weightShape)) {
        yShape->SetDimNum(1);
        yShape->SetDim(0, UNKNOWN_DIM_NUM);
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(ShapeCheckAndInfer(xShape, weightShape, yShape, *transposeX, *transposeWeight) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The Shape Check failed "), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

constexpr size_t DTYPE_INDEX = 3UL;
constexpr size_t OUTPUT_INDEX = 0UL;

static ge::graphStatus InferDataTypeForFusedWeightQuantMatMul(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* dtype = attrs->GetAttrPointer<int64_t>(DTYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dtype);
    ge::DataType outputDtype = static_cast<ge::DataType>(*dtype);
    if (*dtype == -1 || outputDtype == ge::DT_UNDEFINED) {
        outputDtype = context->GetInputDataType(0);
    }
    OP_CHECK_IF(outputDtype != ge::DT_FLOAT16 && outputDtype != ge::DT_INT8,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "dtype only supports FLOAT16/INT8, but is %ld", *dtype),
                return ge::GRAPH_FAILED);
    return context->SetOutputDataType(OUTPUT_INDEX, outputDtype);
}

} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(FusedWeightQuantMatMul)
    .InferShape(InferShapeForFusedWeightQuantMatMul)
    .InferDataType(InferDataTypeForFusedWeightQuantMatMul);
}
