/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include <cstdlib>
#include <limits>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_sparse4to2quant_matmul_weight_nz.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_Sparse4to2QuantMatmul_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_Sparse4to2QuantMatmul_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_Sparse4to2QuantMatmul_test TearDown" << endl; }

    static void FillSparse42Weight(std::vector<int8_t>& weight)
    {
        // pattern 0b1100: first two zeros, last two non-zeros in every group of 4
        for (size_t i = 0; i + 3 < weight.size(); i += 4) {
            weight[i] = 0;
            weight[i + 1] = 0;
            weight[i + 2] = 1;
            weight[i + 3] = 1;
        }
    }

    static void CheckIndexDescriptor(const TensorDesc& indexDesc, aclnnStatus expected)
    {
        TensorDesc xDesc({64, 512}, ACL_INT8, ACL_FORMAT_ND);
        TensorDesc weightDesc({256, 256}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {8, 16, 16, 32});
        TensorDesc xScaleDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
        TensorDesc weightScaleDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
        TensorDesc outDesc({64, 256}, ACL_BF16, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnSparse4to2QuantMatmulWeightNz,
                            INPUT(xDesc, weightDesc, indexDesc, xScaleDesc, weightScaleDesc, nullptr), OUTPUT(outDesc));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), expected);
    }
};

TEST_F(l2_Sparse4to2QuantMatmul_test, ascend910B2_test_normal_case_01)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({64, 512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc sparse_weight_desc = TensorDesc({256, 256}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {8, 16, 16, 32});
    TensorDesc index_desc = TensorDesc({256, 64}, ACL_UINT8, ACL_FORMAT_ND, {}, 0, {8, 16, 16, 8});
    TensorDesc x_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc sparse_weight_scale_desc = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnSparse4to2QuantMatmulWeightNz,
        INPUT(x_desc, sparse_weight_desc, index_desc, x_scale_desc, sparse_weight_scale_desc, bias_desc),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, ascend910B2_test_normal_no_bias)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({64, 512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc sparse_weight_desc = TensorDesc({256, 256}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {8, 16, 16, 32});
    TensorDesc index_desc = TensorDesc({256, 64}, ACL_UINT8, ACL_FORMAT_ND, {}, 0, {8, 16, 16, 8});
    TensorDesc x_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc sparse_weight_scale_desc = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSparse4to2QuantMatmulWeightNz,
                        INPUT(x_desc, sparse_weight_desc, index_desc, x_scale_desc, sparse_weight_scale_desc, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, ascend910B2_invalid_dtype_x)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc sparse_weight_desc = TensorDesc({256, 256}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {8, 16, 16, 32});
    TensorDesc index_desc = TensorDesc({256, 64}, ACL_UINT8, ACL_FORMAT_ND, {}, 0, {8, 16, 16, 8});
    TensorDesc x_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc sparse_weight_scale_desc = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSparse4to2QuantMatmulWeightNz,
                        INPUT(x_desc, sparse_weight_desc, index_desc, x_scale_desc, sparse_weight_scale_desc, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, ascend910B2_invalid_empty_x)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({0, 512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc sparse_weight_desc = TensorDesc({256, 256}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {8, 16, 16, 32});
    TensorDesc index_desc = TensorDesc({256, 64}, ACL_UINT8, ACL_FORMAT_ND, {}, 0, {8, 16, 16, 8});
    TensorDesc x_scale_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc sparse_weight_scale_desc = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({0, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSparse4to2QuantMatmulWeightNz,
                        INPUT(x_desc, sparse_weight_desc, index_desc, x_scale_desc, sparse_weight_scale_desc, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, ascend910B2_invalid_bias_dtype)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({64, 512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc sparse_weight_desc = TensorDesc({256, 256}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {8, 16, 16, 32});
    TensorDesc index_desc = TensorDesc({256, 64}, ACL_UINT8, ACL_FORMAT_ND, {}, 0, {8, 16, 16, 8});
    TensorDesc x_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc sparse_weight_scale_desc = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnSparse4to2QuantMatmulWeightNz,
        INPUT(x_desc, sparse_weight_desc, index_desc, x_scale_desc, sparse_weight_scale_desc, bias_desc),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, ascend910B2_invalid_sparse_weight_nd)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({64, 512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc sparse_weight_desc = TensorDesc({256, 256}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({256, 64}, ACL_UINT8, ACL_FORMAT_ND, {}, 0, {8, 16, 16, 8});
    TensorDesc x_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc sparse_weight_scale_desc = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSparse4to2QuantMatmulWeightNz,
                        INPUT(x_desc, sparse_weight_desc, index_desc, x_scale_desc, sparse_weight_scale_desc, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, unsupported_soc)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    TensorDesc x_desc = TensorDesc({64, 512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc sparse_weight_desc = TensorDesc({256, 256}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {8, 16, 16, 32});
    TensorDesc index_desc = TensorDesc({256, 64}, ACL_UINT8, ACL_FORMAT_ND, {}, 0, {8, 16, 16, 8});
    TensorDesc x_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc sparse_weight_scale_desc = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSparse4to2QuantMatmulWeightNz,
                        INPUT(x_desc, sparse_weight_desc, index_desc, x_scale_desc, sparse_weight_scale_desc, nullptr),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, trans_sparse4to2_para_success)
{
    const int64_t n = 128;
    const int64_t k = 512;
    std::vector<int64_t> shapeVec = {n, k};
    std::vector<int8_t> weight(static_cast<size_t>(n * k), 0);
    FillSparse42Weight(weight);
    aclIntArray* shape = aclCreateIntArray(shapeVec.data(), shapeVec.size());
    int8_t* sparseWeight = nullptr;
    int64_t* sparseWeightDims = nullptr;
    uint64_t sparseWeightDimsNum = 0;
    uint8_t* index = nullptr;
    int64_t* indexDims = nullptr;
    uint64_t indexDimsNum = 0;

    EXPECT_EQ(aclnnTransSparse4to2Para(weight.data(), shape, &sparseWeight, &sparseWeightDims, &sparseWeightDimsNum,
                                       &index, &indexDims, &indexDimsNum),
              ACLNN_SUCCESS);
    EXPECT_EQ(sparseWeightDimsNum, 4u);
    EXPECT_EQ(sparseWeightDims[0], 8);
    EXPECT_EQ(sparseWeightDims[1], 8);
    EXPECT_EQ(sparseWeightDims[2], 16);
    EXPECT_EQ(sparseWeightDims[3], 32);
    EXPECT_EQ(indexDimsNum, 4u);
    EXPECT_EQ(indexDims[0], 8);
    EXPECT_EQ(indexDims[1], 8);
    EXPECT_EQ(indexDims[2], 16);
    EXPECT_EQ(indexDims[3], 8);
    free(sparseWeight);
    free(sparseWeightDims);
    free(index);
    free(indexDims);
    aclDestroyIntArray(shape);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, trans_sparse4to2_para_success_small)
{
    const int64_t n = 16;
    const int64_t k = 64;
    std::vector<int64_t> shapeVec = {n, k};
    std::vector<int8_t> weight(static_cast<size_t>(n * k), 0);
    FillSparse42Weight(weight);
    aclIntArray* shape = aclCreateIntArray(shapeVec.data(), shapeVec.size());
    int8_t* sparseWeight = nullptr;
    int64_t* sparseWeightDims = nullptr;
    uint64_t sparseWeightDimsNum = 0;
    uint8_t* index = nullptr;
    int64_t* indexDims = nullptr;
    uint64_t indexDimsNum = 0;

    EXPECT_EQ(aclnnTransSparse4to2Para(weight.data(), shape, &sparseWeight, &sparseWeightDims, &sparseWeightDimsNum,
                                       &index, &indexDims, &indexDimsNum),
              ACLNN_SUCCESS);
    // CeilDiv(CeilDiv(64,32),2)=1, CeilDiv(16,16)=1 → {1,1,16,32}
    EXPECT_EQ(sparseWeightDims[0], 1);
    EXPECT_EQ(sparseWeightDims[1], 1);
    free(sparseWeight);
    free(sparseWeightDims);
    free(index);
    free(indexDims);
    aclDestroyIntArray(shape);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, trans_sparse4to2_para_invalid_null_shape)
{
    int8_t w[8] = {0};
    int8_t* sw = nullptr;
    int64_t* swd = nullptr;
    uint64_t swdn = 0;
    uint8_t* idx = nullptr;
    int64_t* id = nullptr;
    uint64_t idn = 0;
    EXPECT_EQ(aclnnTransSparse4to2Para(w, nullptr, &sw, &swd, &swdn, &idx, &id, &idn), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, trans_sparse4to2_para_invalid_null_weight)
{
    std::vector<int64_t> shapeVec = {16, 64};
    aclIntArray* shape = aclCreateIntArray(shapeVec.data(), shapeVec.size());
    int8_t* sw = nullptr;
    int64_t* swd = nullptr;
    uint64_t swdn = 0;
    uint8_t* idx = nullptr;
    int64_t* id = nullptr;
    uint64_t idn = 0;
    EXPECT_EQ(aclnnTransSparse4to2Para(nullptr, shape, &sw, &swd, &swdn, &idx, &id, &idn), ACLNN_ERR_PARAM_NULLPTR);
    aclDestroyIntArray(shape);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, trans_sparse4to2_para_invalid_shape_dim)
{
    std::vector<int64_t> shapeVec = {16, 64, 1};
    aclIntArray* shape = aclCreateIntArray(shapeVec.data(), shapeVec.size());
    std::vector<int8_t> weight(16 * 64, 0);
    FillSparse42Weight(weight);
    int8_t* sw = nullptr;
    int64_t* swd = nullptr;
    uint64_t swdn = 0;
    uint8_t* idx = nullptr;
    int64_t* id = nullptr;
    uint64_t idn = 0;
    EXPECT_EQ(aclnnTransSparse4to2Para(weight.data(), shape, &sw, &swd, &swdn, &idx, &id, &idn),
              ACLNN_ERR_PARAM_INVALID);
    aclDestroyIntArray(shape);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, trans_sparse4to2_para_invalid_sparsity)
{
    const int64_t n = 16;
    const int64_t k = 64;
    std::vector<int64_t> shapeVec = {n, k};
    std::vector<int8_t> weight(static_cast<size_t>(n * k), 1); // dense → fails SPARSE_MAP
    aclIntArray* shape = aclCreateIntArray(shapeVec.data(), shapeVec.size());
    int8_t* sw = nullptr;
    int64_t* swd = nullptr;
    uint64_t swdn = 0;
    uint8_t* idx = nullptr;
    int64_t* id = nullptr;
    uint64_t idn = 0;
    EXPECT_EQ(aclnnTransSparse4to2Para(weight.data(), shape, &sw, &swd, &swdn, &idx, &id, &idn),
              ACLNN_ERR_PARAM_INVALID);
    aclDestroyIntArray(shape);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, trans_sparse4to2_para_invalid_nonpositive_shape)
{
    const std::vector<std::array<int64_t, 2>> invalidShapes = {{-1, 4},
                                                               {-15, 4},
                                                               {-16, 64},
                                                               {16, -1},
                                                               {16, -32},
                                                               {16, -64},
                                                               {0, 8},
                                                               {8, 0},
                                                               {std::numeric_limits<int64_t>::min(), 64},
                                                               {16, std::numeric_limits<int64_t>::min()}};
    for (const auto& dims : invalidShapes) {
        SCOPED_TRACE(::testing::PrintToString(dims));
        aclIntArray* shape = aclCreateIntArray(dims.data(), dims.size());
        ASSERT_NE(shape, nullptr);
        int8_t weight[64] = {0};
        int8_t* sparseWeight = nullptr;
        uint8_t* index = nullptr;
        int64_t* sparseWeightDims = nullptr;
        int64_t* indexDims = nullptr;
        uint64_t sparseWeightDimsNum = 0;
        uint64_t indexDimsNum = 0;
        EXPECT_EQ(aclnnTransSparse4to2Para(weight, shape, &sparseWeight, &sparseWeightDims, &sparseWeightDimsNum,
                                           &index, &indexDims, &indexDimsNum),
                  ACLNN_ERR_PARAM_INVALID);
        EXPECT_EQ(sparseWeight, nullptr);
        EXPECT_EQ(index, nullptr);
        free(sparseWeight);
        free(sparseWeightDims);
        free(index);
        free(indexDims);
        aclDestroyIntArray(shape);
    }
}

TEST_F(l2_Sparse4to2QuantMatmul_test, trans_sparse4to2_para_invalid_shape_overflow)
{
    const auto maxDim = std::numeric_limits<int64_t>::max();
    const std::vector<std::array<int64_t, 2>> invalidShapes = {{maxDim, 1}, {1, maxDim}, {1LL << 32, 1LL << 32}};
    for (const auto& dims : invalidShapes) {
        SCOPED_TRACE(::testing::PrintToString(dims));
        aclIntArray* shape = aclCreateIntArray(dims.data(), dims.size());
        ASSERT_NE(shape, nullptr);
        int8_t weight[8] = {0};
        int8_t* sparseWeight = nullptr;
        uint8_t* index = nullptr;
        int64_t* sparseWeightDims = nullptr;
        int64_t* indexDims = nullptr;
        uint64_t sparseWeightDimsNum = 0;
        uint64_t indexDimsNum = 0;
        EXPECT_EQ(aclnnTransSparse4to2Para(weight, shape, &sparseWeight, &sparseWeightDims, &sparseWeightDimsNum,
                                           &index, &indexDims, &indexDimsNum),
                  ACLNN_ERR_PARAM_INVALID);
        EXPECT_EQ(sparseWeight, nullptr);
        EXPECT_EQ(index, nullptr);
        aclDestroyIntArray(shape);
    }
}

TEST_F(l2_Sparse4to2QuantMatmul_test, trans_sparse4to2_para_invalid_null_outputs)
{
    const std::array<int64_t, 2> dims = {16, 64};
    aclIntArray* shape = aclCreateIntArray(dims.data(), dims.size());
    ASSERT_NE(shape, nullptr);
    int8_t weight[8] = {0};
    for (size_t nullArg = 0; nullArg < 6; ++nullArg) {
        SCOPED_TRACE(nullArg);
        int8_t* sparseWeight = nullptr;
        uint8_t* index = nullptr;
        int64_t* sparseWeightDims = nullptr;
        int64_t* indexDims = nullptr;
        uint64_t sparseWeightDimsNum = 0;
        uint64_t indexDimsNum = 0;
        EXPECT_EQ(aclnnTransSparse4to2Para(
                      weight, shape, nullArg == 0 ? nullptr : &sparseWeight, nullArg == 1 ? nullptr : &sparseWeightDims,
                      nullArg == 2 ? nullptr : &sparseWeightDimsNum, nullArg == 3 ? nullptr : &index,
                      nullArg == 4 ? nullptr : &indexDims, nullArg == 5 ? nullptr : &indexDimsNum),
                  ACLNN_ERR_PARAM_NULLPTR);
    }
    aclDestroyIntArray(shape);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, ascend910B2_invalid_index_view_shape)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    const std::vector<std::vector<int64_t>> invalidShapes = {{256, 32}, {256, 512}, {256, 0},    {256, 1},
                                                             {128, 64}, {64},       {1, 256, 64}};
    for (const auto& dims : invalidShapes) {
        SCOPED_TRACE(::testing::PrintToString(dims));
        CheckIndexDescriptor(TensorDesc(dims, ACL_UINT8, ACL_FORMAT_ND, {}, 0, {8, 16, 16, 8}),
                             ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_Sparse4to2QuantMatmul_test, ascend910B2_invalid_index_storage_shape)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    const std::vector<std::vector<int64_t>> invalidShapes = {
        {256, 64}, {8, 16, 16}, {8, 16, 16, 8, 1}, {7, 16, 16, 8}, {8, 15, 16, 8}, {8, 16, 15, 8}, {8, 16, 16, 7}};
    for (const auto& dims : invalidShapes) {
        SCOPED_TRACE(::testing::PrintToString(dims));
        CheckIndexDescriptor(TensorDesc({256, 64}, ACL_UINT8, ACL_FORMAT_ND, {}, 0, dims), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_Sparse4to2QuantMatmul_test, ascend910B2_invalid_noncontiguous_index)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    CheckIndexDescriptor(TensorDesc({256, 64}, ACL_UINT8, ACL_FORMAT_ND, {128, 2}, 0, {8, 16, 16, 8}),
                         ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, ascend910B2_valid_padded_index)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc xDesc({64, 65}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc weightDesc({17, 36}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 32});
    TensorDesc indexDesc({17, 9}, ACL_UINT8, ACL_FORMAT_ND, {}, 0, {2, 2, 16, 8});
    TensorDesc xScaleDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc weightScaleDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc outDesc({64, 17}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSparse4to2QuantMatmulWeightNz,
                        INPUT(xDesc, weightDesc, indexDesc, xScaleDesc, weightScaleDesc, nullptr), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, ascend910B2_invalid_null_scales)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc xDesc({64, 512}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc weightDesc({256, 256}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {8, 16, 16, 32});
    TensorDesc indexDesc({256, 64}, ACL_UINT8, ACL_FORMAT_ND, {}, 0, {8, 16, 16, 8});
    TensorDesc xScaleDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc weightScaleDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc outDesc({64, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto noXScale = OP_API_UT(aclnnSparse4to2QuantMatmulWeightNz,
                              INPUT(xDesc, weightDesc, indexDesc, nullptr, weightScaleDesc, nullptr), OUTPUT(outDesc));
    auto noWeightScale = OP_API_UT(aclnnSparse4to2QuantMatmulWeightNz,
                                   INPUT(xDesc, weightDesc, indexDesc, xScaleDesc, nullptr, nullptr), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(noXScale.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(noWeightScale.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_Sparse4to2QuantMatmul_test, ascend910B2_invalid_empty_weight)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    const std::vector<std::array<int64_t, 2>> emptyShapes = {{0, 512}, {256, 0}};
    for (const auto& dims : emptyShapes) {
        SCOPED_TRACE(::testing::PrintToString(dims));
        const int64_t n = dims[0];
        const int64_t k = dims[1];
        TensorDesc xDesc({64, k}, ACL_INT8, ACL_FORMAT_ND);
        TensorDesc weightDesc({n, k / 2}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {k / 64, n / 16, 16, 32});
        TensorDesc indexDesc({n, k / 8}, ACL_UINT8, ACL_FORMAT_ND, {}, 0, {k / 64, n / 16, 16, 8});
        TensorDesc xScaleDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
        TensorDesc weightScaleDesc({n}, ACL_FLOAT, ACL_FORMAT_ND);
        TensorDesc outDesc({64, n}, ACL_BF16, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnSparse4to2QuantMatmulWeightNz,
                            INPUT(xDesc, weightDesc, indexDesc, xScaleDesc, weightScaleDesc, nullptr), OUTPUT(outDesc));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    }
}
