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
#include "index/unsorted_segment_common/op_kernel/arch35/unsorted_segment_max_tiling_key.h"

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
    return gert::TilingContextPara("UnsortedSegmentMax",
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

TEST(UnsortedSegmentMaxTilingTest, simt_int64_small_inner)
{
    int32_t numSegments = 8;
    ExpectTilingKey({{16, 8}, {16, 8}}, {{16}, {16}}, {{8, 8}, {8, 8}}, ge::DT_INT64, ge::DT_INT32, numSegments,
                    USS_TEMPLATE_SIMT);
}

TEST(UnsortedSegmentMaxTilingTest, sort_simt_int64_ratio_ge_5)
{
    int32_t numSegments = 20;
    ExpectTilingKey({{100, 32}, {100, 32}}, {{100}, {100}}, {{20, 32}, {20, 32}}, ge::DT_INT64, ge::DT_INT32,
                    numSegments, USS_TEMPLATE_SORT_SIMT);
}

TEST(UnsortedSegmentMaxTilingTest, out_full_load_small_float32)
{
    int32_t numSegments = 5;
    ExpectTilingKey({{8, 16}, {8, 16}}, {{8}, {8}}, {{5, 16}, {5, 16}}, ge::DT_FLOAT, ge::DT_INT32, numSegments,
                    USS_TEMPLATE_OUT_FL);
}

TEST(UnsortedSegmentMaxTilingTest, simd_split_col_large_inner)
{
    int32_t numSegments = 8;
    ExpectTilingKey({{8, 8192}, {8, 8192}}, {{8}, {8}}, {{8, 8192}, {8, 8192}}, ge::DT_FLOAT, ge::DT_INT32, numSegments,
                    USS_TEMPLATE_SIMD_SPLIT_COL);
}

TEST(UnsortedSegmentMaxTilingTest, simd_non_sort_mid_inner)
{
    int32_t numSegments = 16;
    ExpectTilingKey({{32, 512}, {32, 512}}, {{32}, {32}}, {{16, 512}, {16, 512}}, ge::DT_FLOAT, ge::DT_INT32,
                    numSegments, USS_TEMPLATE_SIMD_NON_SORT);
}

TEST(UnsortedSegmentMaxTilingTest, simd_dyn_sort_int32_ids)
{
    int32_t numSegments = 20;
    ExpectTilingKey({{200, 512}, {200, 512}}, {{200}, {200}}, {{20, 512}, {20, 512}}, ge::DT_FLOAT, ge::DT_INT32,
                    numSegments, USS_TEMPLATE_SIMD_DYN_SORT | (USS_CAST_INT32_TO_UINT8 << 3));
}

TEST(UnsortedSegmentMaxTilingTest, simd_dyn_sort_int64_ids)
{
    int32_t numSegments = 20;
    ExpectTilingKey({{200, 512}, {200, 512}}, {{200}, {200}}, {{20, 512}, {20, 512}}, ge::DT_FLOAT, ge::DT_INT64,
                    numSegments, USS_TEMPLATE_SIMD_DYN_SORT | (USS_CAST_INT64_TO_UINT8 << 3));
}
