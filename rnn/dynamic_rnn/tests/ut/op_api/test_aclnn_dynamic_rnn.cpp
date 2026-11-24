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
 * \file test_aclnn_dynamic_rnn.cpp
 * \brief
 */
#include "../../../op_api/aclnn_lstm.h"

#include <vector>
#include <array>
#include "gtest/gtest.h"
#include <limits>

#include "opdev/op_log.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api/op_api_def_nn.h"

#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_dynamic_rnn_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "dynamic_rnn_test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "dynamic_rnn_test TearDown" << std::endl; }
};

// 输入params的list长度不合法拦截
TEST_F(l2_dynamic_rnn_test, ascend910B_test_dynamic_rnn_params_dim_invalid)
{
    int time_step = 1;
    int batch_size = 1;
    int hidden_size = 4;
    int input_size = hidden_size;
    int64_t numLayers = 1;
    bool isbias = true;
    bool batchFirst = false;
    bool bidirectionl = true;
    bool isTraining = false;
    bool has_biases = true;
    int64_t d_scale = bidirectionl == true ? 2 : 1;
    vector<int64_t> inputDim = {time_step, batch_size, input_size};
    vector<int64_t> wiDim = {4 * hidden_size, input_size};
    vector<int64_t> whDim = {4 * hidden_size, hidden_size};
    vector<int64_t> bDim = {4 * hidden_size};
    vector<int64_t> outDim = {time_step, batch_size, d_scale * hidden_size};
    vector<int64_t> hycyDim = {d_scale * numLayers, batch_size, hidden_size};
    vector<int64_t> outHDim = {time_step, batch_size, hidden_size};

    auto input = TensorDesc(inputDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto wi = TensorDesc(wiDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto wh = TensorDesc(whDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto bi = TensorDesc(bDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto bh = TensorDesc(bDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto params = TensorListDesc({wi, wh, bi, bh});

    auto output = TensorDesc(outDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto hy = TensorDesc(hycyDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto cy = TensorDesc(hycyDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outIForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outIBackward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputI = TensorListDesc({outIForward, outIBackward});

    auto outJForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outJBackward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputJ = TensorListDesc({outJForward, outJBackward});

    auto outFForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outFBackward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputF = TensorListDesc({outFForward, outFBackward});

    auto outOForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outOBackward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputO = TensorListDesc({outOForward, outOBackward});

    auto outHForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outHBackward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputH = TensorListDesc({outHForward, outHBackward});

    auto outCForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outCBackward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputC = TensorListDesc({outCForward, outCBackward});

    auto outTanhCForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outTanhCBackward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputTanhC = TensorListDesc({outTanhCForward, outTanhCBackward});

    auto hx_desc = nullptr;
    auto batchSizes = nullptr;

    auto ut = OP_API_UT(aclnnLSTM, // host api第二段接口名称
                        INPUT(input, params, hx_desc, batchSizes, has_biases, numLayers, 0.0, isTraining, bidirectionl,
                              batchFirst), // host api输入
                        OUTPUT(output, hy, cy, outputI, outputJ, outputF, outputO, outputH, outputC, outputTanhC));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

/* ascend950 planning. Planning only -- nothing here says whether a kernel executes. */
namespace {
struct Lstm950Case {
    int timeStep = 2;
    int batch = 1;
    int hidden = 8;
    int inputSize = 8;
    int64_t layers = 1;
    bool bidirectional = false;
    bool train = true;
    double dropout = 0.0;
    aclDataType dtype = ACL_FLOAT;
};

aclnnStatus PlanLstm950(const Lstm950Case& c)
{
    const int64_t directions = c.bidirectional ? 2 : 1;
    const int64_t legs = c.layers * directions;
    const int gates = 4 * c.hidden;

    auto input = TensorDesc({c.timeStep, c.batch, c.inputSize}, c.dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto bias = TensorDesc({gates}, c.dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto recurrent = TensorDesc({gates, c.hidden}, c.dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<TensorDesc> paramDescs;
    for (int64_t layer = 0; layer < c.layers; ++layer) {
        const int64_t fanIn = layer == 0 ? c.inputSize : c.hidden * directions;
        for (int64_t direction = 0; direction < directions; ++direction) {
            paramDescs.emplace_back(TensorDesc({gates, fanIn}, c.dtype, ACL_FORMAT_ND).ValueRange(-1, 1));
            paramDescs.emplace_back(recurrent);
            paramDescs.emplace_back(bias);
            paramDescs.emplace_back(bias);
        }
    }
    auto params = TensorListDesc(paramDescs);

    auto output = TensorDesc({c.timeStep, c.batch, directions * c.hidden}, c.dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto state = TensorDesc({legs, c.batch, c.hidden}, c.dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gate = TensorDesc({c.timeStep, c.batch, c.hidden}, c.dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gateList = TensorListDesc(static_cast<int>(legs), gate);

    auto ut = OP_API_UT(
        aclnnLSTM, INPUT(input, params, nullptr, nullptr, true, c.layers, c.dropout, c.train, c.bidirectional, false),
        OUTPUT(output, state, state, gateList, gateList, gateList, gateList, gateList, gateList, gateList));
    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}
} // namespace

TEST_F(l2_dynamic_rnn_test, ascend950_accepts_bidirectional)
{
    NpuArchManager arch(NpuArch::DAV_3510);
    Lstm950Case plan;
    plan.bidirectional = true;
    EXPECT_EQ(PlanLstm950(plan), ACL_SUCCESS);
}

TEST_F(l2_dynamic_rnn_test, ascend950_rejects_dropout_that_applies)
{
    NpuArchManager arch(NpuArch::DAV_3510);
    Lstm950Case plan;
    plan.layers = 2;
    plan.dropout = 0.5;
    EXPECT_EQ(PlanLstm950(plan), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_dynamic_rnn_test, ascend950_accepts_dropout_on_single_layer)
{
    NpuArchManager arch(NpuArch::DAV_3510);
    Lstm950Case plan;
    plan.dropout = 0.5;
    EXPECT_EQ(PlanLstm950(plan), ACL_SUCCESS);
}

TEST_F(l2_dynamic_rnn_test, ascend950_accepts_dropout_when_not_training)
{
    NpuArchManager arch(NpuArch::DAV_3510);
    Lstm950Case plan;
    plan.layers = 2;
    plan.dropout = 0.5;
    plan.train = false;
    EXPECT_EQ(PlanLstm950(plan), ACL_SUCCESS);
}

TEST_F(l2_dynamic_rnn_test, ascend950_rejects_dropout_that_applies_bidirectional)
{
    NpuArchManager arch(NpuArch::DAV_3510);
    Lstm950Case plan;
    plan.bidirectional = true;
    plan.layers = 2;
    plan.dropout = 0.5;
    EXPECT_EQ(PlanLstm950(plan), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_dynamic_rnn_test, ascend950_rejects_bf16_with_bidirectional)
{
    NpuArchManager arch(NpuArch::DAV_3510);
    Lstm950Case plan;
    plan.bidirectional = true;
    plan.dtype = ACL_BF16;
    EXPECT_EQ(PlanLstm950(plan), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_dynamic_rnn_test, ascend950_accepts_bf16_unidirectional)
{
    NpuArchManager arch(NpuArch::DAV_3510);
    Lstm950Case plan;
    plan.dtype = ACL_BF16;
    EXPECT_EQ(PlanLstm950(plan), ACL_SUCCESS);
}

// 输入params的shape不合法拦截
TEST_F(l2_dynamic_rnn_test, ascend910B_test_dynamic_rnn_params_shape_invalid)
{
    int time_step = 1;
    int batch_size = 1;
    int hidden_size = 4;
    int input_size = hidden_size;
    int64_t numLayers = 1;
    bool isbias = true;
    bool batchFirst = false;
    bool bidirectionl = false;
    bool isTraining = true;
    bool has_biases = true;
    int64_t d_scale = bidirectionl == true ? 2 : 1;
    vector<int64_t> inputDim = {time_step, batch_size, input_size};
    vector<int64_t> wiDim = {4 * hidden_size, input_size};
    vector<int64_t> whDim = {4 * hidden_size, hidden_size};
    vector<int64_t> bDim = {3 * hidden_size};
    vector<int64_t> outDim = {time_step, batch_size, d_scale * hidden_size};
    vector<int64_t> hycyDim = {d_scale * numLayers, batch_size, hidden_size};
    vector<int64_t> outHDim = {time_step, batch_size, hidden_size};

    auto input = TensorDesc(inputDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto wi = TensorDesc(wiDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto wh = TensorDesc(whDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto bi = TensorDesc(bDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto bh = TensorDesc(bDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto params = TensorListDesc({wi, wh, bi, bh});

    auto output = TensorDesc(outDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto hy = TensorDesc(hycyDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto cy = TensorDesc(hycyDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outIForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputI = TensorListDesc({outIForward});

    auto outJForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputJ = TensorListDesc({outJForward});

    auto outFForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputF = TensorListDesc({outFForward});

    auto outOForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputO = TensorListDesc({outOForward});

    auto outHForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputH = TensorListDesc({outHForward});

    auto outCForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputC = TensorListDesc({outCForward});

    auto outTanhCForward = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputTanhC = TensorListDesc({outTanhCForward});

    auto hx_desc = nullptr;
    auto batchSizes = nullptr;

    auto ut = OP_API_UT(aclnnLSTM, // host api第二段接口名称
                        INPUT(input, params, hx_desc, batchSizes, has_biases, numLayers, 0.0, isTraining, bidirectionl,
                              batchFirst), // host api输入
                        OUTPUT(output, hy, cy, outputI, outputJ, outputF, outputO, outputH, outputC, outputTanhC));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

namespace {
auto CreateLstmUtForNumLayers(int64_t numLayers, bool train, bool bidirectional, bool hasBias)
{
    int time_step = 1;
    int batch_size = 1;
    int hidden_size = 4;
    int input_size = hidden_size;
    int64_t d_scale = bidirectional ? 2 : 1;
    int64_t layerCnt = numLayers > 0 ? numLayers : 1;
    int64_t shapeLayerCnt = std::min<int64_t>(layerCnt, 2);
    int64_t outListCnt = std::min<int64_t>(d_scale * layerCnt, 8);
    vector<int64_t> inputDim = {time_step, batch_size, input_size};
    vector<int64_t> wiDim = {4 * hidden_size, input_size};
    vector<int64_t> whDim = {4 * hidden_size, hidden_size};
    vector<int64_t> bDim = {4 * hidden_size};
    vector<int64_t> outDim = {time_step, batch_size, d_scale * hidden_size};
    vector<int64_t> hycyDim = {d_scale * shapeLayerCnt, batch_size, hidden_size};
    vector<int64_t> outHDim = {time_step, batch_size, hidden_size};

    auto input = TensorDesc(inputDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto wi = TensorDesc(wiDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto wh = TensorDesc(whDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto bi = TensorDesc(bDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto bh = TensorDesc(bDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<TensorDesc> paramsVec = {wi, wh};
    if (hasBias) {
        paramsVec.push_back(bi);
        paramsVec.push_back(bh);
    }
    auto params = TensorListDesc(paramsVec);

    auto output = TensorDesc(outDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hy = TensorDesc(hycyDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto cy = TensorDesc(hycyDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outH = TensorDesc(outHDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputI = TensorListDesc(static_cast<int>(outListCnt), outH);
    auto outputJ = TensorListDesc(static_cast<int>(outListCnt), outH);
    auto outputF = TensorListDesc(static_cast<int>(outListCnt), outH);
    auto outputO = TensorListDesc(static_cast<int>(outListCnt), outH);
    auto outputH = TensorListDesc(static_cast<int>(outListCnt), outH);
    auto outputC = TensorListDesc(static_cast<int>(outListCnt), outH);
    auto outputTanhC = TensorListDesc(static_cast<int>(outListCnt), outH);

    auto ut = OP_API_UT(aclnnLSTM,
                        INPUT(input, params, (const aclTensorList*)nullptr, (const aclTensor*)nullptr, hasBias,
                              numLayers, 0.0, train, bidirectional, false),
                        OUTPUT(output, hy, cy, outputI, outputJ, outputF, outputO, outputH, outputC, outputTanhC));
    return ut;
}
} // namespace

TEST_F(l2_dynamic_rnn_test, ascend910B_test_dynamic_rnn_num_layers_negative)
{
    auto ut = CreateLstmUtForNumLayers(-1, false, false, true);
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_dynamic_rnn_test, ascend910B_test_dynamic_rnn_num_layers_zero)
{
    auto ut = CreateLstmUtForNumLayers(0, false, false, true);
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_dynamic_rnn_test, ascend910B_test_dynamic_rnn_num_layers_int64_min)
{
    auto ut = CreateLstmUtForNumLayers(std::numeric_limits<int64_t>::min(), false, false, true);
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_dynamic_rnn_test, ascend910B_test_dynamic_rnn_num_layers_int64_max)
{
    auto ut = CreateLstmUtForNumLayers(std::numeric_limits<int64_t>::max(), false, false, true);
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_dynamic_rnn_test, ascend910B_test_dynamic_rnn_num_layers_wrap_bypass)
{
    auto ut = CreateLstmUtForNumLayers(std::numeric_limits<int64_t>::min() + 1, false, false, true);
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_dynamic_rnn_test, ascend910B_test_dynamic_rnn_num_layers_train_negative)
{
    auto ut = CreateLstmUtForNumLayers(-1, true, false, true);
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_dynamic_rnn_test, ascend910B_test_dynamic_rnn_num_layers_bidirectional_negative)
{
    auto ut = CreateLstmUtForNumLayers(-1, false, true, true);
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_dynamic_rnn_test, ascend910B_test_dynamic_rnn_num_layers_negative_data_mode)
{
    int batch_size = 1;
    int hidden_size = 4;
    int input_size = hidden_size;
    int64_t numLayers = -1;
    vector<int64_t> inputDim = {batch_size, input_size};
    vector<int64_t> wiDim = {4 * hidden_size, input_size};
    vector<int64_t> whDim = {4 * hidden_size, hidden_size};
    vector<int64_t> outDim = {1, batch_size, hidden_size};
    vector<int64_t> hycyDim = {1, batch_size, hidden_size};

    auto input = TensorDesc(inputDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto wi = TensorDesc(wiDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto wh = TensorDesc(whDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto params = TensorListDesc({wi, wh});
    auto batchSizes = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).Value(std::vector<int64_t>{1});

    auto output = TensorDesc(outDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto hy = TensorDesc(hycyDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto cy = TensorDesc(hycyDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outH = TensorDesc(outDim, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputI = TensorListDesc(1, outH);
    auto outputJ = TensorListDesc(1, outH);
    auto outputF = TensorListDesc(1, outH);
    auto outputO = TensorListDesc(1, outH);
    auto outputH = TensorListDesc(1, outH);
    auto outputC = TensorListDesc(1, outH);
    auto outputTanhC = TensorListDesc(1, outH);

    auto ut = OP_API_UT(
        aclnnLSTM,
        INPUT(input, params, (const aclTensorList*)nullptr, batchSizes, false, numLayers, 0.0, false, false, false),
        OUTPUT(output, hy, cy, outputI, outputJ, outputF, outputO, outputH, outputC, outputTanhC));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}
