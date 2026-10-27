/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "index/unsorted_segment_common/op_kernel/arch35/unsorted_segment_prod_tiling_key.h"

namespace {
struct UnsortedSegmentCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    uint64_t maxThread = 0;
};

UnsortedSegmentCompileInfo g_compileInfo = {64, 253952, 2048};

gert::TilingContextPara MakeTilingContext(const gert::StorageShape& xShape, const gert::StorageShape& idsShape,
                                          const gert::StorageShape& outShape, ge::DataType xDtype,
                                          ge::DataType idsDtype, int32_t& numSegments)
{
    return gert::TilingContextPara("UnsortedSegmentProd",
                                   {
                                       {xShape, xDtype, ge::FORMAT_ND},
                                       {idsShape, idsDtype, ge::FORMAT_ND},
                                       {gert::StorageShape({1}, {1}), ge::DT_INT32, ge::FORMAT_ND, true, &numSegments},
                                   },
                                   {
                                       {outShape, xDtype, ge::FORMAT_ND},
                                   },
                                   &g_compileInfo);
}

void ExpectTilingKey(const gert::StorageShape& xShape, const gert::StorageShape& idsShape,
                     const gert::StorageShape& outShape, ge::DataType xDtype, ge::DataType idsDtype,
                     int32_t numSegments, uint64_t expectedKey)
{
    auto context = MakeTilingContext(xShape, idsShape, outShape, xDtype, idsDtype, numSegments);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(static_cast<uint64_t>(tilingInfo.tilingKey), expectedKey);
}
} // namespace

TEST(UnsortedSegmentProdTilingTest, out_full_load_ws_merge)
{
    int32_t numSegments = 8;
    ExpectTilingKey({{20000, 16}, {20000, 16}}, {{20000}, {20000}}, {{8, 16}, {8, 16}}, ge::DT_FLOAT, ge::DT_INT32,
                    numSegments, USS_TEMPLATE_OUT_FL_WS_MERGE);
}

TEST(UnsortedSegmentProdTilingTest, out_full_load_small_float32)
{
    int32_t numSegments = 8;
    ExpectTilingKey({{10000, 8}, {10000, 8}}, {{10000}, {10000}}, {{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::DT_INT32,
                    numSegments, USS_TEMPLATE_OUT_FL);
}

TEST(UnsortedSegmentProdTilingTest, input_partition_mid_inner)
{
    int32_t numSegments = 128;
    ExpectTilingKey({{4096, 64}, {4096, 64}}, {{4096}, {4096}}, {{128, 64}, {128, 64}}, ge::DT_FLOAT, ge::DT_INT32,
                    numSegments, USS_TEMPLATE_INPUT_PART);
}

TEST(UnsortedSegmentProdTilingTest, segment_sort_mid_inner)
{
    int32_t numSegments = 128;
    ExpectTilingKey({{2048, 256}, {2048, 256}}, {{2048}, {2048}}, {{128, 256}, {128, 256}}, ge::DT_FLOAT, ge::DT_INT32,
                    numSegments, USS_TEMPLATE_SEGMENT_SORT);
}

TEST(UnsortedSegmentProdTilingTest, simd_split_col_large_inner)
{
    int32_t numSegments = 8;
    ExpectTilingKey({{8, 8192}, {8, 8192}}, {{8}, {8}}, {{8, 8192}, {8, 8192}}, ge::DT_FLOAT, ge::DT_INT32, numSegments,
                    USS_TEMPLATE_SIMD_SPLIT_COL);
}

TEST(UnsortedSegmentProdTilingTest, sort_simt_int64_ratio_ge_5)
{
    int32_t numSegments = 16;
    ExpectTilingKey({{2048, 8}, {2048, 8}}, {{2048}, {2048}}, {{16, 8}, {16, 8}}, ge::DT_INT64, ge::DT_INT32,
                    numSegments, USS_TEMPLATE_SORT_SIMT);
}

TEST(UnsortedSegmentProdTilingTest, simt_large_ratio_large_inner)
{
    int32_t numSegments = 60;
    ExpectTilingKey({{30000, 512}, {30000, 512}}, {{30000}, {30000}}, {{60, 512}, {60, 512}}, ge::DT_FLOAT,
                    ge::DT_INT32, numSegments, USS_TEMPLATE_SIMT);
}
