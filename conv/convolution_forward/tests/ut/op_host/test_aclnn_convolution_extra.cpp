/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_convolution_extra.cpp
 * \brief aclnnConvolution 分支补测：conv1d(3维输入)的 pad 尺寸分支与特判路径、
 *        conv3d转conv2d(D==1)分支、conv3d转置的权重DHWCN转置、dtype/bias shape校验分支。
 *        与现有 test_aclnn_convolution.cpp 互补，后者只覆盖常规4D/5D组合。
 */

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "../../../op_host/op_api/aclnn_convolution.h"
#include "../../../op_host/op_api/convolution_util.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace {
constexpr int64_t kGroups1 = 1;
constexpr int8_t kCubeMathType = 0; // KEEP_DTYPE
} // namespace

class convolution_extra_test : public testing::Test {
public:
    void SetUp() {}

    void TearDown() {}
};

// conv1d基础路径：3维输入+1维pad(ViewConv1dPad1dAs4d)，bias 1维
TEST_F(convolution_extra_test, conv1d_pad1d_success)
{
    vector<int64_t> inp = {2, 16, 32};
    vector<int64_t> weight = {16, 16, 3};
    vector<int64_t> bias = {16};
    vector<int64_t> output = {2, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{2});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0});

    auto inpDesc = TensorDesc(inp, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_FLOAT16, ACL_FORMAT_NCL).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, stridesDesc, paddingDesc, dilationDesc, false,
                              outputPaddingDesc, kGroups1),
                        OUTPUT(outputDesc), kCubeMathType);
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// conv1d：2维pad(ViewConv1dPad2dAs4d)
TEST_F(convolution_extra_test, conv1d_pad2d_bias3d_success)
{
    vector<int64_t> inp = {2, 16, 32};
    vector<int64_t> weight = {16, 16, 3};
    vector<int64_t> bias = {16};
    vector<int64_t> output = {2, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{2});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0});

    auto inpDesc = TensorDesc(inp, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_FLOAT16, ACL_FORMAT_NCL).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, stridesDesc, paddingDesc, dilationDesc, false,
                              outputPaddingDesc, kGroups1),
                        OUTPUT(outputDesc), kCubeMathType);
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// conv1d：pad维度非法(3)走报错分支
TEST_F(convolution_extra_test, conv1d_invalid_pad_size_fail)
{
    vector<int64_t> inp = {2, 16, 32};
    vector<int64_t> weight = {16, 16, 3};
    vector<int64_t> bias = {16};
    vector<int64_t> output = {2, 16, 32};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{2});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0});

    auto inpDesc = TensorDesc(inp, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_FLOAT16, ACL_FORMAT_NCL).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, stridesDesc, paddingDesc, dilationDesc, false,
                              outputPaddingDesc, kGroups1),
                        OUTPUT(outputDesc), kCubeMathType);
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACL_SUCCESS);
}

// conv1d特判路径：大stride(stride[1]=64>63)且等于weight的第3维，pad为0，C=1
TEST_F(convolution_extra_test, conv1d_special_stride_case)
{
    vector<int64_t> inp = {1, 1, 256};
    vector<int64_t> weight = {1, 1, 64};
    vector<int64_t> output = {1, 1, 4};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{64});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{0});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0});

    auto inpDesc = TensorDesc(inp, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_FLOAT16, ACL_FORMAT_NCL).Precision(0.01, 0.01);
    auto ut = OP_API_UT(
        aclnnConvolution,
        INPUT(inpDesc, weightDesc, nullptr, stridesDesc, paddingDesc, dilationDesc, false, outputPaddingDesc, kGroups1),
        OUTPUT(outputDesc), kCubeMathType);
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// conv3d转conv2d分支：D==1且Kd==1且padD==0
TEST_F(convolution_extra_test, conv3d_to_conv2d_d1_case)
{
    vector<int64_t> inp = {1, 4, 1, 8, 8};
    vector<int64_t> weight = {8, 4, 1, 3, 3};
    vector<int64_t> bias = {8};
    vector<int64_t> output = {1, 8, 1, 6, 6};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, stridesDesc, paddingDesc, dilationDesc, false,
                              outputPaddingDesc, kGroups1),
                        OUTPUT(outputDesc), kCubeMathType);
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// conv3d转置路径：transposed=true，非退化kernel，走权重DHWCN转置等分支
TEST_F(convolution_extra_test, conv3d_transposed_case)
{
    vector<int64_t> inp = {1, 4, 6, 6, 6};
    vector<int64_t> weight = {4, 8, 2, 2, 2};
    vector<int64_t> output = {1, 8, 7, 7, 7};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(
        aclnnConvolution,
        INPUT(inpDesc, weightDesc, nullptr, stridesDesc, paddingDesc, dilationDesc, true, outputPaddingDesc, kGroups1),
        OUTPUT(outputDesc), kCubeMathType);
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype校验分支：x与weight dtype不一致
TEST_F(convolution_extra_test, dtype_mismatch_fail)
{
    vector<int64_t> inp = {2, 16, 32, 16};
    vector<int64_t> weight = {16, 16, 3, 3};
    vector<int64_t> output = {2, 16, 32, 8};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 2});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 2});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 2});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(
        aclnnConvolution,
        INPUT(inpDesc, weightDesc, nullptr, stridesDesc, paddingDesc, dilationDesc, false, outputPaddingDesc, kGroups1),
        OUTPUT(outputDesc), kCubeMathType);
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACL_SUCCESS);
}

// bias shape校验分支：bias尺寸不等于Cout
TEST_F(convolution_extra_test, bias_shape_mismatch_fail)
{
    vector<int64_t> inp = {2, 16, 32, 16};
    vector<int64_t> weight = {16, 16, 3, 3};
    vector<int64_t> bias = {99};
    vector<int64_t> output = {2, 16, 32, 8};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 2});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 2});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 2});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0});

    auto inpDesc = TensorDesc(inp, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, stridesDesc, paddingDesc, dilationDesc, false,
                              outputPaddingDesc, kGroups1),
                        OUTPUT(outputDesc), kCubeMathType);
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACL_SUCCESS);
}

// groups>1的分组卷积：走groups相关校验与L0分支
TEST_F(convolution_extra_test, grouped_conv2d_case)
{
    vector<int64_t> inp = {2, 16, 32, 32};
    vector<int64_t> weight = {16, 8, 3, 3};
    vector<int64_t> output = {2, 16, 32, 32};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0});

    auto inpDesc = TensorDesc(inp, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(
        aclnnConvolution,
        INPUT(inpDesc, weightDesc, nullptr, stridesDesc, paddingDesc, dilationDesc, false, outputPaddingDesc, 2),
        OUTPUT(outputDesc), kCubeMathType);
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(convolution_extra_test, conv2d_transpose_vec_fallback_selector)
{
    std::vector<aclDataType> dataTypes = {ACL_FLOAT16, ACL_BF16, ACL_FLOAT};
    for (auto dataType : dataTypes) {
        auto input = TensorDesc({1, 16, 1, 16}, dataType, ACL_FORMAT_NCHW).ToAclType();
        auto weight = TensorDesc({16, 16, 64, 64}, dataType, ACL_FORMAT_NCHW).ToAclType();
        auto bias = TensorDesc({16}, dataType, ACL_FORMAT_NCHW).ToAclType();
        auto output = TensorDesc({1, 16, 64, 79}, dataType, ACL_FORMAT_NCHW).ToAclType();
        EXPECT_TRUE(ConvolutionUtil::ShouldUseConvTranspose2DVecFallbackWithL1(input.get(), weight.get(), bias.get(),
                                                                               output.get(), 512 * 1024));
        EXPECT_FALSE(ConvolutionUtil::ShouldUseConvTranspose2DVecFallbackWithL1(input.get(), weight.get(), bias.get(),
                                                                                output.get(), 8 * 1024 * 1024));
    }

    auto input = TensorDesc({1, 16, 1, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW).ToAclType();
    auto weight = TensorDesc({16, 16, 64, 64}, ACL_FLOAT16, ACL_FORMAT_NCHW).ToAclType();
    auto output = TensorDesc({1, 16, 64, 79}, ACL_FLOAT16, ACL_FORMAT_NCHW).ToAclType();
    EXPECT_TRUE(ConvolutionUtil::ShouldUseConvTranspose2DVecFallbackWithL1(input.get(), weight.get(), nullptr,
                                                                           output.get(), 512 * 1024));
    EXPECT_FALSE(ConvolutionUtil::ShouldUseConvTranspose2DVecFallbackWithL1(nullptr, weight.get(), nullptr,
                                                                            output.get(), 512 * 1024));
}

TEST_F(convolution_extra_test, conv2d_transpose_vec_fallback_selector_rejects_invalid_input)
{
    auto input = TensorDesc({1, 16, 1, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW).ToAclType();
    auto weight = TensorDesc({16, 16, 64, 64}, ACL_FLOAT16, ACL_FORMAT_NHWC).ToAclType();
    auto output = TensorDesc({1, 16, 64, 79}, ACL_FLOAT16, ACL_FORMAT_NCHW).ToAclType();
    EXPECT_FALSE(ConvolutionUtil::ShouldUseConvTranspose2DVecFallbackWithL1(input.get(), weight.get(), nullptr,
                                                                            output.get(), 512 * 1024));

    auto input3d = TensorDesc({1, 16, 16}, ACL_FLOAT16, ACL_FORMAT_NCL).ToAclType();
    auto weight3d = TensorDesc({16, 16, 3}, ACL_FLOAT16, ACL_FORMAT_NCL).ToAclType();
    auto output2d = TensorDesc({1, 16, 18}, ACL_FLOAT16, ACL_FORMAT_NCL).ToAclType();
    EXPECT_FALSE(ConvolutionUtil::ShouldUseConvTranspose2DVecFallbackWithL1(input3d.get(), weight3d.get(), nullptr,
                                                                            output2d.get(), 512 * 1024));

    auto int8Input = TensorDesc({1, 16, 1, 16}, ACL_INT8, ACL_FORMAT_NCHW).ToAclType();
    auto int8Weight = TensorDesc({16, 16, 64, 64}, ACL_INT8, ACL_FORMAT_NCHW).ToAclType();
    auto int8Output = TensorDesc({1, 16, 64, 79}, ACL_INT8, ACL_FORMAT_NCHW).ToAclType();
    EXPECT_FALSE(ConvolutionUtil::ShouldUseConvTranspose2DVecFallbackWithL1(int8Input.get(), int8Weight.get(), nullptr,
                                                                            int8Output.get(), 512 * 1024));
}

TEST_F(convolution_extra_test, conv3d_transpose_vec_fallback_selector)
{
    std::vector<aclDataType> dataTypes = {ACL_FLOAT16, ACL_FLOAT};
    auto stride = IntArrayDesc(std::vector<int64_t>{1, 1, 1}).ToAclType();
    auto dilation = IntArrayDesc(std::vector<int64_t>{1, 1, 1}).ToAclType();
    for (auto dataType : dataTypes) {
        auto input = TensorDesc({1, 8, 96, 1, 1024}, dataType, ACL_FORMAT_NCDHW).ToAclType();
        auto weight = TensorDesc({8, 8, 5, 3, 1}, dataType, ACL_FORMAT_NCDHW).ToAclType();
        auto output = TensorDesc({1, 8, 100, 3, 1024}, dataType, ACL_FORMAT_NCDHW).ToAclType();
        EXPECT_TRUE(ConvolutionUtil::ShouldUseConvTranspose3DVecFallbackWithL1(
            input.get(), weight.get(), output.get(), stride.get(), dilation.get(), 512 * 1024));
        EXPECT_FALSE(ConvolutionUtil::ShouldUseConvTranspose3DVecFallbackWithL1(
            input.get(), weight.get(), output.get(), stride.get(), dilation.get(), 8 * 1024 * 1024));
    }
}

TEST_F(convolution_extra_test, conv3d_transpose_vec_fallback_selector_handles_limits)
{
    auto input = TensorDesc({1, 8, 96, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    auto weight = TensorDesc({8, 8, 5, 3, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    auto output = TensorDesc({1, 8, 100, 3, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    auto stride = IntArrayDesc(std::vector<int64_t>{1, 1, 1}).ToAclType();
    auto dilation = IntArrayDesc(std::vector<int64_t>{1, 1, 1}).ToAclType();
    EXPECT_TRUE(ConvolutionUtil::ShouldUseConvTranspose3DVecFallbackWithL1(input.get(), weight.get(), output.get(),
                                                                           stride.get(), dilation.get(), 1024));

    auto wideInput = TensorDesc({1, 8, 96, 1, 8192}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    auto wideOutput = TensorDesc({1, 8, 100, 3, 8192}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    EXPECT_FALSE(ConvolutionUtil::ShouldUseConvTranspose3DVecFallbackWithL1(
        wideInput.get(), weight.get(), wideOutput.get(), stride.get(), dilation.get(), 8 * 1024 * 1024));
}

TEST_F(convolution_extra_test, conv3d_transpose_vec_fallback_selector_rejects_invalid_shape)
{
    auto input = TensorDesc({1, 8, 96, 1, 64}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    auto weight = TensorDesc({8, 8, 5, 3, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    auto output = TensorDesc({1, 8, 100, 3, 64}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    auto shortStride = IntArrayDesc(std::vector<int64_t>{1, 1}).ToAclType();
    auto dilation = IntArrayDesc(std::vector<int64_t>{1, 1, 1}).ToAclType();
    EXPECT_FALSE(ConvolutionUtil::ShouldUseConvTranspose3DVecFallbackWithL1(
        input.get(), weight.get(), output.get(), shortStride.get(), dilation.get(), 512 * 1024));

    auto invalidInput = TensorDesc({1, 8, 0, 1, 64}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    auto invalidOutput = TensorDesc({1, 8, 0, 3, 64}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    auto stride = IntArrayDesc(std::vector<int64_t>{1, 1, 1}).ToAclType();
    EXPECT_FALSE(ConvolutionUtil::ShouldUseConvTranspose3DVecFallbackWithL1(
        invalidInput.get(), weight.get(), invalidOutput.get(), stride.get(), dilation.get(), 512 * 1024));

    auto int8Input = TensorDesc({1, 8, 96, 1, 64}, ACL_INT8, ACL_FORMAT_NCDHW).ToAclType();
    auto int8Weight = TensorDesc({8, 8, 5, 3, 1}, ACL_INT8, ACL_FORMAT_NCDHW).ToAclType();
    auto int8Output = TensorDesc({1, 8, 100, 3, 64}, ACL_INT8, ACL_FORMAT_NCDHW).ToAclType();
    EXPECT_FALSE(ConvolutionUtil::ShouldUseConvTranspose3DVecFallbackWithL1(
        int8Input.get(), int8Weight.get(), int8Output.get(), stride.get(), dilation.get(), 512 * 1024));

    auto nhwcWeight = TensorDesc({8, 8, 5, 3, 1}, ACL_FLOAT16, ACL_FORMAT_NDHWC).ToAclType();
    EXPECT_FALSE(ConvolutionUtil::ShouldUseConvTranspose3DVecFallbackWithL1(input.get(), nhwcWeight.get(), output.get(),
                                                                            stride.get(), dilation.get(), 512 * 1024));
}

TEST_F(convolution_extra_test, conv_transpose_vec_fallback_is_disabled_on_other_architectures)
{
    auto input2d = TensorDesc({1, 16, 1, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW).ToAclType();
    auto weight2d = TensorDesc({16, 16, 64, 64}, ACL_FLOAT16, ACL_FORMAT_NCHW).ToAclType();
    auto output2d = TensorDesc({1, 16, 64, 79}, ACL_FLOAT16, ACL_FORMAT_NCHW).ToAclType();
    auto input3d = TensorDesc({1, 8, 96, 1, 1024}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    auto weight3d = TensorDesc({8, 8, 5, 3, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    auto output3d = TensorDesc({1, 8, 100, 3, 1024}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    auto stride = IntArrayDesc(std::vector<int64_t>{1, 1, 1}).ToAclType();
    auto dilation = IntArrayDesc(std::vector<int64_t>{1, 1, 1}).ToAclType();

    std::vector<NpuArch> unsupportedArchs = {
        NpuArch::DAV_1001, NpuArch::DAV_1002, NpuArch::DAV_1003, NpuArch::DAV_1004, NpuArch::DAV_1999,
        NpuArch::DAV_2002, NpuArch::DAV_2003, NpuArch::DAV_2004, NpuArch::DAV_2006, NpuArch::DAV_2102,
        NpuArch::DAV_2103, NpuArch::DAV_2104, NpuArch::DAV_3002, NpuArch::DAV_3003, NpuArch::DAV_3004,
        NpuArch::DAV_3102, NpuArch::DAV_3103, NpuArch::DAV_3113, NpuArch::DAV_3502, NpuArch::DAV_3505,
        NpuArch::DAV_3510, NpuArch::DAV_3801, NpuArch::DAV_5101, NpuArch::DAV_5102, NpuArch::DAV_5161,
        NpuArch::DAV_5162, NpuArch::DAV_9201, NpuArch::DAV_9202, NpuArch::DAV_9301, NpuArch::DAV_RESV};
    for (auto arch : unsupportedArchs) {
        op::NpuArchManager platform(arch);
        EXPECT_FALSE(ConvolutionUtil::ShouldUseConvTranspose2DVecFallbackWithL1(input2d.get(), weight2d.get(), nullptr,
                                                                                output2d.get(), 512 * 1024));
        EXPECT_FALSE(ConvolutionUtil::ShouldUseConvTranspose3DVecFallbackWithL1(
            input3d.get(), weight3d.get(), output3d.get(), stride.get(), dilation.get(), 512 * 1024));
    }
}

TEST_F(convolution_extra_test, conv_transpose_vec_fallback_builds_5d_output_view)
{
    auto executor = CREATE_EXECUTOR();
    ASSERT_NE(executor.get(), nullptr);

    auto output4d = TensorDesc({1, 16, 64, 79}, ACL_FLOAT16, ACL_FORMAT_NCHW).ToAclType();
    auto outputView = ConvolutionUtil::GetConvTransposeVecFallbackOutputInfo(output4d.get(), executor.get());
    ASSERT_NE(outputView, nullptr);
    EXPECT_EQ(outputView->GetViewShape().GetDimNum(), 5U);
    EXPECT_EQ(outputView->GetViewFormat(), op::Format::FORMAT_NCDHW);

    auto output5d = TensorDesc({1, 8, 100, 3, 1024}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ToAclType();
    EXPECT_EQ(ConvolutionUtil::GetConvTransposeVecFallbackOutputInfo(output5d.get(), executor.get()), output5d.get());

    auto output3d = TensorDesc({1, 16, 64}, ACL_FLOAT16, ACL_FORMAT_NCL).ToAclType();
    EXPECT_EQ(ConvolutionUtil::GetConvTransposeVecFallbackOutputInfo(output3d.get(), executor.get()), nullptr);
}
