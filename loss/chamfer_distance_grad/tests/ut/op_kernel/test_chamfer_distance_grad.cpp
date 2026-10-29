/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "test_chamfer_distance_grad_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void chamfer_distance_grad(GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR idx1, GM_ADDR idx2,
                                                            GM_ADDR grad_dist1, GM_ADDR grad_dist2, GM_ADDR grad_xyz1,
                                                            GM_ADDR grad_xyz2, GM_ADDR workspace, GM_ADDR tiling_data);
class chamfer_distance_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "chamfer_distance_grad_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "chamfer_distance_grad_test TearDown\n" << endl; }
};

TEST_F(chamfer_distance_grad_test, test_case_fp32)
{
    uint32_t batch_size = 2;
    uint32_t num = 2;
    // inputs
    size_t xyz_size = batch_size * num * 2 * sizeof(float) * 2;
    size_t grad_size = batch_size * num * sizeof(float) * 2;
    size_t idx_size = batch_size * num * sizeof(int32_t) * 2;
    size_t grad_xyz_size = batch_size * num * sizeof(float) * 2 * 2;
    size_t tiling_data_size = sizeof(ChamferDistanceGradTilingDataTest) + 1024;

    uint8_t* xyz1 = (uint8_t*)AscendC::GmAlloc(xyz_size);
    uint8_t* xyz2 = (uint8_t*)AscendC::GmAlloc(xyz_size);
    uint8_t* grad_dist1 = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* grad_dist2 = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* idx1 = (uint8_t*)AscendC::GmAlloc(idx_size);
    uint8_t* idx2 = (uint8_t*)AscendC::GmAlloc(idx_size);
    uint8_t* grad_xyz1 = (uint8_t*)AscendC::GmAlloc(grad_xyz_size);
    uint8_t* grad_xyz2 = (uint8_t*)AscendC::GmAlloc(grad_xyz_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    ChamferDistanceGradTilingDataTest* tilingDatafromBin = reinterpret_cast<ChamferDistanceGradTilingDataTest*>(tiling);

    tilingDatafromBin->batch_size = 2;
    tilingDatafromBin->num = 2;
    tilingDatafromBin->ub_size = 195538;
    tilingDatafromBin->task_per_core = 1;
    tilingDatafromBin->core_used = 4;
    tilingDatafromBin->task_tail_core = 1;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(chamfer_distance_grad, blockDim, xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2, grad_xyz1, grad_xyz2,
                workspace, tiling);
    AscendC::GmFree(xyz1);
    AscendC::GmFree(xyz2);
    AscendC::GmFree(grad_dist1);
    AscendC::GmFree(grad_dist2);
    AscendC::GmFree(idx1);
    AscendC::GmFree(idx2);
    AscendC::GmFree(grad_xyz1);
    AscendC::GmFree(grad_xyz2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(chamfer_distance_grad_test, test_case_fp16)
{
    uint32_t batch_size = 32;
    uint32_t num = 10000;

    // inputs
    size_t xyz_size = batch_size * num * 2 * sizeof(half);
    size_t grad_size = batch_size * num * sizeof(half);
    size_t idx_size = batch_size * num * sizeof(int32_t);
    size_t grad_xyz_size = batch_size * num * sizeof(half) * 2;
    size_t tiling_data_size = sizeof(ChamferDistanceGradTilingDataTest);

    uint8_t* xyz1 = (uint8_t*)AscendC::GmAlloc(xyz_size);
    uint8_t* xyz2 = (uint8_t*)AscendC::GmAlloc(xyz_size);
    uint8_t* grad_dist1 = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* grad_dist2 = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* idx1 = (uint8_t*)AscendC::GmAlloc(idx_size);
    uint8_t* idx2 = (uint8_t*)AscendC::GmAlloc(idx_size);
    uint8_t* grad_xyz1 = (uint8_t*)AscendC::GmAlloc(grad_xyz_size);
    uint8_t* grad_xyz2 = (uint8_t*)AscendC::GmAlloc(grad_xyz_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;
    ChamferDistanceGradTilingDataTest* tilingDatafromBin = reinterpret_cast<ChamferDistanceGradTilingDataTest*>(tiling);

    tilingDatafromBin->batch_size = 2;
    tilingDatafromBin->num = 2;
    tilingDatafromBin->ub_size = 195538;
    tilingDatafromBin->task_per_core = 1;
    tilingDatafromBin->core_used = 48;
    tilingDatafromBin->task_tail_core = 1;

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(chamfer_distance_grad, blockDim, xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2, grad_xyz1, grad_xyz2,
                workspace, tiling);
    AscendC::GmFree(xyz1);
    AscendC::GmFree(xyz2);
    AscendC::GmFree(grad_dist1);
    AscendC::GmFree(grad_dist2);
    AscendC::GmFree(idx1);
    AscendC::GmFree(idx2);
    AscendC::GmFree(grad_xyz1);
    AscendC::GmFree(grad_xyz2);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

namespace {
// 与kernel内部公式一致(fp32): per_ub_size = ub_size / 6 / sizeof(float) / 32 * 32
// ub_size = 1536时 per_ub_size = 1536 / 6 / 4 / 32 * 32 = 64，可用小规模数据构造整除点场景
constexpr uint64_t kUtUbSize = 1536;

using IdxOverride = std::pair<uint64_t, int32_t>; // <覆盖位置, idx值>

// 确定性模式等价tiling(core_used=1，单core处理全部batch*num个任务)跑fp32 kernel并用闭式解校验:
// 基础输入 xyz1[i]=(i,0)、xyz2[i]=(i+1,0)、idx=i%num、gradDist=1 时，
// 期望 gradXyz1[i]=(-4,0)、gradXyz2[i]=(4,0)；被idxOverrides注入非法值的位置，
// kernel钳制(issue#7041)后仅丢失该侧交叉贡献(每路-2/+2)，且不越界访问GM
void RunChamferDistanceGradFp32AndCheck(uint64_t batchSize, uint64_t num,
                                        const std::vector<IdxOverride>& idx1Overrides = {},
                                        const std::vector<IdxOverride>& idx2Overrides = {})
{
    const uint64_t total = batchSize * num;
    const size_t xyz_size = total * 2 * sizeof(float);
    const size_t grad_size = total * sizeof(float);
    const size_t idx_size = total * sizeof(int32_t);
    const size_t tiling_data_size = sizeof(ChamferDistanceGradTilingDataTest);

    uint8_t* xyz1 = (uint8_t*)AscendC::GmAlloc(xyz_size);
    uint8_t* xyz2 = (uint8_t*)AscendC::GmAlloc(xyz_size);
    uint8_t* grad_dist1 = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* grad_dist2 = (uint8_t*)AscendC::GmAlloc(grad_size);
    uint8_t* idx1 = (uint8_t*)AscendC::GmAlloc(idx_size);
    uint8_t* idx2 = (uint8_t*)AscendC::GmAlloc(idx_size);
    uint8_t* grad_xyz1 = (uint8_t*)AscendC::GmAlloc(xyz_size);
    uint8_t* grad_xyz2 = (uint8_t*)AscendC::GmAlloc(xyz_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    auto freeAllGm = [&]() {
        // GmFree(nullptr)等价free(nullptr)，对未分配成功的指针释放是安全的
        AscendC::GmFree(xyz1);
        AscendC::GmFree(xyz2);
        AscendC::GmFree(grad_dist1);
        AscendC::GmFree(grad_dist2);
        AscendC::GmFree(idx1);
        AscendC::GmFree(idx2);
        AscendC::GmFree(grad_xyz1);
        AscendC::GmFree(grad_xyz2);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    };
    if (xyz1 == nullptr || xyz2 == nullptr || grad_dist1 == nullptr || grad_dist2 == nullptr || idx1 == nullptr ||
        idx2 == nullptr || grad_xyz1 == nullptr || grad_xyz2 == nullptr || workspace == nullptr || tiling == nullptr) {
        freeAllGm();
        FAIL() << "GmAlloc failed";
        return;
    }

    float* xyz1F = reinterpret_cast<float*>(xyz1);
    float* xyz2F = reinterpret_cast<float*>(xyz2);
    float* gradDist1F = reinterpret_cast<float*>(grad_dist1);
    float* gradDist2F = reinterpret_cast<float*>(grad_dist2);
    int32_t* idx1I = reinterpret_cast<int32_t*>(idx1);
    int32_t* idx2I = reinterpret_cast<int32_t*>(idx2);
    for (uint64_t i = 0; i < total; i++) {
        xyz1F[2 * i] = static_cast<float>(i);
        xyz1F[2 * i + 1] = 0.0f;
        xyz2F[2 * i] = static_cast<float>(i + 1);
        xyz2F[2 * i + 1] = 0.0f;
        gradDist1F[i] = 1.0f;
        gradDist2F[i] = 1.0f;
        idx1I[i] = static_cast<int32_t>(i % num);
        idx2I[i] = static_cast<int32_t>(i % num);
    }
    for (const auto& override : idx1Overrides) {
        idx1I[override.first] = override.second;
    }
    for (const auto& override : idx2Overrides) {
        idx2I[override.first] = override.second;
    }
    memset(grad_xyz1, 0, xyz_size);
    memset(grad_xyz2, 0, xyz_size);

    ChamferDistanceGradTilingDataTest* tilingData = reinterpret_cast<ChamferDistanceGradTilingDataTest*>(tiling);
    tilingData->batch_size = batchSize;
    tilingData->num = num;
    tilingData->ub_size = kUtUbSize;
    tilingData->task_per_core = total;
    tilingData->core_used = 1;
    tilingData->task_tail_core = total;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(chamfer_distance_grad, 1, xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2, grad_xyz1, grad_xyz2,
                workspace, tiling);

    // 判断某位置的idx是否为合法取值[0, num)：被覆盖为非法值的位置由kernel钳制丢弃该侧贡献
    auto isIdxLegal = [&](uint64_t pos, const std::vector<IdxOverride>& overrides) -> bool {
        for (const auto& override : overrides) {
            if (override.first == pos) {
                return override.second >= 0 && static_cast<uint64_t>(override.second) < num;
            }
        }
        return true;
    };
    const float* gradXyz1F = reinterpret_cast<const float*>(grad_xyz1);
    const float* gradXyz2F = reinterpret_cast<const float*>(grad_xyz2);
    for (uint64_t i = 0; i < total; i++) {
        bool idx1Legal = isIdxLegal(i, idx1Overrides);
        bool idx2Legal = isIdxLegal(i, idx2Overrides);
        float expectGradXyz1 = (idx1Legal ? -2.0f : 0.0f) + (idx2Legal ? -2.0f : 0.0f);
        float expectGradXyz2 = (idx1Legal ? 2.0f : 0.0f) + (idx2Legal ? 2.0f : 0.0f);
        EXPECT_FLOAT_EQ(gradXyz1F[2 * i], expectGradXyz1) << "gradXyz1 point index=" << i;
        EXPECT_FLOAT_EQ(gradXyz1F[2 * i + 1], 0.0f) << "gradXyz1 point index=" << i;
        EXPECT_FLOAT_EQ(gradXyz2F[2 * i], expectGradXyz2) << "gradXyz2 point index=" << i;
        EXPECT_FLOAT_EQ(gradXyz2F[2 * i + 1], 0.0f) << "gradXyz2 point index=" << i;
    }

    freeAllGm();
}
} // namespace

// issue#7040: 确定性模式下batch*num恰等于per_ub_size(此处2*32=64)时，整除点两路梯度全为0的回归测试
TEST_F(chamfer_distance_grad_test, test_case_fp32_task_num_equal_per_ub_size)
{
    RunChamferDistanceGradFp32AndCheck(2, 32);
}

// 整除点邻域回归: batch*num > per_ub_size(65=64+1)，覆盖整块+尾块路径
TEST_F(chamfer_distance_grad_test, test_case_fp32_task_num_greater_per_ub_size)
{
    RunChamferDistanceGradFp32AndCheck(1, 65);
}

// 整除点邻域回归: batch*num < per_ub_size(63<64)，覆盖纯尾块路径
TEST_F(chamfer_distance_grad_test, test_case_fp32_task_num_less_per_ub_size)
{
    RunChamferDistanceGradFp32AndCheck(1, 63);
}

// issue#7041: 非法idx(-1、==num、INT32_MAX)经kernel钳制后不越界、不崩溃，
// 非法点位仅丢失对应侧交叉贡献(每路-2/+2)，其余点位闭式解不变
TEST_F(chamfer_distance_grad_test, test_case_fp32_invalid_idx_clamped)
{
    const std::vector<IdxOverride> idx1Overrides = {{3, -1}, {5, 32}, {7, INT32_MAX}};
    const std::vector<IdxOverride> idx2Overrides = {{11, -1}};
    RunChamferDistanceGradFp32AndCheck(1, 32, idx1Overrides, idx2Overrides);
}
