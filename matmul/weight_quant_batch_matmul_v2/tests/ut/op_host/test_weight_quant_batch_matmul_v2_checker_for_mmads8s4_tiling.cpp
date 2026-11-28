/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "graph/types.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "op_host/op_const_def_nn.h"
#include "../../../op_host/op_tiling/weight_quant_batch_matmul_v2_checker_for_mmads8s4.h"

namespace {
struct CheckerCase {
    ge::DataType weightDtype = ge::DT_INT4;
    ge::DataType scaleDtype = ge::DT_FLOAT16;
    ge::DataType offsetDtype = ge::DT_FLOAT16;
    ge::DataType outputDtype = ge::DT_FLOAT16;
    ge::DataType quantScaleDtype = ge::DT_UINT64;
    bool transB = false;
    bool hasOffset = true;
    int64_t quantScaleSize = 0;
    bool hasQuantOffset = false;
    std::vector<int64_t> offsetShapeOverride;
};

gert::Shape MakeShape(const std::vector<int64_t>& dims)
{
    gert::Shape shape;
    for (auto dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

ge::graphStatus RunChecker(const CheckerCase& c)
{
    constexpr int64_t m = 16;
    constexpr int64_t k = 128;
    constexpr int64_t n = 64;
    const bool isInt4 = c.weightDtype == ge::DT_INT4;
    const int64_t groupSize = isInt4 ? 64 : 0;
    const auto weightOrigin = c.transB ? std::vector<int64_t>{n, k} : std::vector<int64_t>{k, n};
    const auto weightStorage = isInt4 ? (c.transB ? std::vector<int64_t>{k / 16, n / 16, 16, 16} :
                                                    std::vector<int64_t>{n / 16, k / 16, 16, 16}) :
                                        weightOrigin;
    const auto scaleDims = isInt4 ? (c.transB ? std::vector<int64_t>{n, k / groupSize} :
                                                std::vector<int64_t>{k / groupSize, n}) :
                                    std::vector<int64_t>{1, n};
    const auto& offsetDims = c.offsetShapeOverride.empty() ? scaleDims : c.offsetShapeOverride;

    gert::StorageShape xShape;
    xShape.MutableOriginShape() = MakeShape({m, k});
    xShape.MutableStorageShape() = MakeShape({m, k});
    gert::StorageShape weightShape;
    weightShape.MutableOriginShape() = MakeShape(weightOrigin);
    weightShape.MutableStorageShape() = MakeShape(weightStorage);
    gert::StorageShape scaleShape;
    scaleShape.MutableOriginShape() = MakeShape(scaleDims);
    scaleShape.MutableStorageShape() = MakeShape(scaleDims);
    gert::StorageShape offsetShape;
    offsetShape.MutableOriginShape() = MakeShape(offsetDims);
    offsetShape.MutableStorageShape() = MakeShape(offsetDims);
    gert::StorageShape quantScaleShape;
    quantScaleShape.MutableOriginShape() = MakeShape({c.quantScaleSize});
    quantScaleShape.MutableStorageShape() = MakeShape({c.quantScaleSize});
    gert::StorageShape quantOffsetShape;
    quantOffsetShape.MutableOriginShape() = MakeShape({1});
    quantOffsetShape.MutableStorageShape() = MakeShape({1});
    gert::StorageShape outputShape;
    outputShape.MutableOriginShape() = MakeShape({m, n});
    outputShape.MutableStorageShape() = MakeShape({m, n});

    auto rawTilingData = gert::TilingData::CreateCap(4096);
    EXPECT_NE(rawTilingData, nullptr);
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    uint64_t dummyCompileInfo = 0U;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &weightShape, &scaleShape, c.hasOffset ? &offsetShape : nullptr,
                                    c.quantScaleSize > 0 ? &quantScaleShape : nullptr,
                                    c.hasQuantOffset ? &quantOffsetShape : nullptr, nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&dummyCompileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, c.weightDtype, ge::FORMAT_ND, isInt4 ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND)
                      .NodeInputTd(2, c.scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, c.offsetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, c.quantScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, c.outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(c.transB)},
                                  {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"inner_precise", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .TilingData(rawTilingData.get())
                      .SetOpType("WeightQuantBatchMatmulV2")
                      .Build();
    auto tilingContext = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tilingContext, nullptr);
    optiling::WeightQuantBatchMatmulV2Checker4MmadS8S4 checker(tilingContext);
    return checker.Check();
}
} // namespace

TEST(WeightQuantBatchMatmulV2Checker4MmadS8S4Test, AcceptsInt4PergroupOffset)
{
    CheckerCase nonTrans;
    EXPECT_EQ(RunChecker(nonTrans), ge::GRAPH_SUCCESS);
    CheckerCase trans;
    trans.transB = true;
    EXPECT_EQ(RunChecker(trans), ge::GRAPH_SUCCESS);
}

TEST(WeightQuantBatchMatmulV2Checker4MmadS8S4Test, AcceptsInt4PergroupWithoutOffset)
{
    CheckerCase c;
    c.hasOffset = false;
    EXPECT_EQ(RunChecker(c), ge::GRAPH_SUCCESS);
}

TEST(WeightQuantBatchMatmulV2Checker4MmadS8S4Test, RejectsMismatchedOffsetDtype)
{
    CheckerCase c;
    c.offsetDtype = ge::DT_BF16;
    EXPECT_EQ(RunChecker(c), ge::GRAPH_FAILED);
}

TEST(WeightQuantBatchMatmulV2Checker4MmadS8S4Test, RejectsMismatchedOffsetShape)
{
    CheckerCase c;
    c.transB = true;
    c.offsetShapeOverride = {64, 1};
    EXPECT_EQ(RunChecker(c), ge::GRAPH_FAILED);
}

TEST(WeightQuantBatchMatmulV2Checker4MmadS8S4Test, KeepsRejectingInt8Offset)
{
    CheckerCase c;
    c.weightDtype = ge::DT_INT8;
    c.scaleDtype = ge::DT_UINT64;
    c.offsetDtype = ge::DT_INT32;
    EXPECT_EQ(RunChecker(c), ge::GRAPH_FAILED);
}

TEST(WeightQuantBatchMatmulV2Checker4MmadS8S4Test, AcceptsInt4Int8PerTensorAndPerChannelQuant)
{
    CheckerCase perTensor;
    perTensor.outputDtype = ge::DT_INT8;
    perTensor.quantScaleSize = 1;
    EXPECT_EQ(RunChecker(perTensor), ge::GRAPH_SUCCESS);

    CheckerCase perChannel;
    perChannel.outputDtype = ge::DT_INT8;
    perChannel.quantScaleSize = 64;
    EXPECT_EQ(RunChecker(perChannel), ge::GRAPH_SUCCESS);
}

TEST(WeightQuantBatchMatmulV2Checker4MmadS8S4Test, RejectsInt4Int8WithoutQuantScale)
{
    CheckerCase c;
    c.outputDtype = ge::DT_INT8;
    EXPECT_EQ(RunChecker(c), ge::GRAPH_FAILED);
}

TEST(WeightQuantBatchMatmulV2Checker4MmadS8S4Test, RejectsInt4Int8WithInvalidQuantScale)
{
    CheckerCase wrongDtype;
    wrongDtype.outputDtype = ge::DT_INT8;
    wrongDtype.quantScaleSize = 1;
    wrongDtype.quantScaleDtype = ge::DT_FLOAT;
    EXPECT_EQ(RunChecker(wrongDtype), ge::GRAPH_FAILED);

    CheckerCase wrongSize;
    wrongSize.outputDtype = ge::DT_INT8;
    wrongSize.quantScaleSize = 2;
    EXPECT_EQ(RunChecker(wrongSize), ge::GRAPH_FAILED);
}

TEST(WeightQuantBatchMatmulV2Checker4MmadS8S4Test, RejectsSeparateQuantOffset)
{
    CheckerCase c;
    c.outputDtype = ge::DT_INT8;
    c.quantScaleSize = 1;
    c.hasQuantOffset = true;
    EXPECT_EQ(RunChecker(c), ge::GRAPH_FAILED);
}
