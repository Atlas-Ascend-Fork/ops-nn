/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_situ_glu_group_grad.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {
constexpr float MAGIC = -768.0f; // padding-row sentinel (exactly representable in fp32/fp16/bf16)
constexpr double TOL = 1e-4;

uint16_t FloatToFp16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x7FFFFF;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {
        u = sign;
    } else if (exp == 31) {
        u = sign | 0x7F800000 | (mant << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

uint16_t FloatToBf16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint32_t roundingBias = 0x7FFF + ((u >> 16) & 1);
    u += roundingBias;
    return static_cast<uint16_t>(u >> 16);
}

float Bf16ToFloat(uint16_t b)
{
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(const int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

bool CheckHardwareSupport()
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        LOG_PRINT("Warning: Cannot get SOC name, skip hardware check\n");
        return true;
    }

    LOG_PRINT("Current SOC: %s\n", socName);
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }

    LOG_PRINT("Warning: SituGluGroupGrad only supports Ascend950, current SOC '%s' is not supported. Skip test.\n",
              socName);
    return false;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    const aclDataType dataType, aclTensor** tensor)
{
    // grad_weight may be an empty tensor (shape [0]) when weight is absent: aclrtMalloc(0) is not
    // portable, so always reserve one block and skip the copy when there is nothing to copy.
    int64_t elemCount = GetShapeSize(shape);
    auto size = elemCount * sizeof(T);
    auto allocSize = size > 0 ? size : static_cast<decltype(size)>(32);
    auto ret = aclrtMalloc(deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    if (size > 0) {
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct TestCase {
    const char* name = "";
    int64_t bs = 0;
    int64_t dimH = 0;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    bool withWeight = false;         // grad_weight equals weight shape when set, empty [0] otherwise
    std::vector<int64_t> groupIndex; // empty -> not passed
    int64_t groupListType = 0;       // 0: groupIndex is a cumsum list, 1: a count list
    aclDataType dtype = aclDataType::ACL_FLOAT;
    double tol = TOL;
};

float Quantize(aclDataType dtype, float v)
{
    if (dtype == aclDataType::ACL_FLOAT16) {
        return Fp16ToFloat(FloatToFp16(v));
    }
    if (dtype == aclDataType::ACL_BF16) {
        return Bf16ToFloat(FloatToBf16(v));
    }
    return v;
}

// CPU golden (double), mirrors the verified backward formulas:
//   t = tanh(g/beta), s = sigmoid(g), A = beta*t*s, U = alpha*tanh(u/alpha) or u
//   grad_gate = gy*U*s*[(1-t^2)+beta*t*(1-s)] * w ; grad_up = gy*A*U'(u) * w
//   grad_w[n] = sum_h gy*A*U  (unweighted; rows >= realBs not produced)
void Golden(const TestCase& tc, const std::vector<float>& gradY, const std::vector<float>& x,
            const std::vector<float>& weight, int64_t realBs, std::vector<double>& gradX, std::vector<double>& gradW)
{
    int64_t dim2H = tc.dimH * 2;
    gradX.assign(tc.bs * dim2H, MAGIC);
    gradW.assign(tc.bs, MAGIC);
    for (int64_t n = 0; n < realBs; ++n) {
        double w = tc.withWeight ? static_cast<double>(weight[n]) : 1.0;
        double wsum = 0.0;
        for (int64_t h = 0; h < tc.dimH; ++h) {
            int64_t gateIdx = n * dim2H + (tc.activateLeft ? h : tc.dimH + h);
            int64_t upIdx = n * dim2H + (tc.activateLeft ? tc.dimH + h : h);
            double g = x[gateIdx];
            double u = x[upIdx];
            double gy = gradY[n * tc.dimH + h];
            double t = std::tanh(g / tc.beta);
            double s = 1.0 / (1.0 + std::exp(-g));
            double a = tc.beta * t * s;
            double upPrime = u;
            double upGrad = 1.0;
            if (tc.linearBeta > 0.0) {
                double tu = std::tanh(u / tc.linearBeta);
                upPrime = tc.linearBeta * tu;
                upGrad = 1.0 - tu * tu;
            }
            wsum += gy * a * upPrime;
            gradX[gateIdx] = gy * upPrime * s * ((1.0 - t * t) + tc.beta * t * (1.0 - s)) * w;
            gradX[upIdx] = gy * a * upGrad * w;
        }
        if (tc.withWeight) {
            gradW[n] = wsum;
        }
    }
}

int RunCase(const TestCase& tc, aclrtStream stream)
{
    int64_t dim2H = tc.dimH * 2;
    int64_t realBs = tc.bs;
    if (!tc.groupIndex.empty()) {
        // cumsum mode (0): the last element is the total row count;
        // count mode (1): the elements sum to the total row count.
        int64_t groupRows = tc.groupIndex.back();
        if (tc.groupListType == 1) {
            groupRows = 0;
            for (int64_t g : tc.groupIndex) {
                groupRows += g;
            }
        }
        realBs = groupRows < 0 ? 0 : (groupRows < tc.bs ? groupRows : tc.bs);
    }

    // inputs quantized to the storage dtype so the golden sees exactly what the kernel sees
    std::vector<float> gradYHost(tc.bs * tc.dimH);
    std::vector<float> xHost(tc.bs * dim2H);
    std::vector<float> weightHost(tc.bs);
    for (size_t i = 0; i < gradYHost.size(); ++i) {
        gradYHost[i] = Quantize(tc.dtype, std::fmod(static_cast<float>(i) * 0.37f + 0.11f, 4.0f) - 2.0f);
    }
    for (size_t i = 0; i < xHost.size(); ++i) {
        xHost[i] = Quantize(tc.dtype, std::fmod(static_cast<float>(i) * 0.53f + 0.29f, 4.0f) - 2.0f);
    }
    for (int64_t i = 0; i < tc.bs; ++i) {
        weightHost[i] = 0.5f + 0.25f * static_cast<float>(i % 4);
    }
    std::vector<float> gradXHost(tc.bs * dim2H, MAGIC);
    std::vector<float> gradWHost(tc.bs, MAGIC);

    // storage buffers in the kernel dtype
    auto storeCount = [](const std::vector<float>& v) { return v.size(); };
    std::vector<uint16_t> gradYStore;
    std::vector<uint16_t> xStore;
    std::vector<uint16_t> gradXStore;
    if (tc.dtype != aclDataType::ACL_FLOAT) {
        gradYStore.resize(gradYHost.size());
        xStore.resize(xHost.size());
        gradXStore.resize(gradXHost.size());
        for (size_t i = 0; i < gradYHost.size(); ++i) {
            gradYStore[i] = tc.dtype == aclDataType::ACL_FLOAT16 ? FloatToFp16(gradYHost[i]) :
                                                                   FloatToBf16(gradYHost[i]);
        }
        for (size_t i = 0; i < xHost.size(); ++i) {
            xStore[i] = tc.dtype == aclDataType::ACL_FLOAT16 ? FloatToFp16(xHost[i]) : FloatToBf16(xHost[i]);
        }
        for (size_t i = 0; i < gradXStore.size(); ++i) {
            gradXStore[i] = tc.dtype == aclDataType::ACL_FLOAT16 ? FloatToFp16(gradXHost[i]) :
                                                                   FloatToBf16(gradXHost[i]);
        }
    }
    (void)storeCount;

    void* gradYDev = nullptr;
    void* xDev = nullptr;
    void* weightDev = nullptr;
    void* groupIndexDev = nullptr;
    void* gradXDev = nullptr;
    void* gradWDev = nullptr;
    aclTensor* gradY = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* groupIndex = nullptr;
    aclTensor* gradX = nullptr;
    aclTensor* gradW = nullptr;

    int ret = 0;
    if (tc.dtype == aclDataType::ACL_FLOAT) {
        ret = CreateAclTensor(gradYHost, {tc.bs, tc.dimH}, &gradYDev, tc.dtype, &gradY);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(xHost, {tc.bs, dim2H}, &xDev, tc.dtype, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradXHost, {tc.bs, dim2H}, &gradXDev, tc.dtype, &gradX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    } else {
        ret = CreateAclTensor(gradYStore, {tc.bs, tc.dimH}, &gradYDev, tc.dtype, &gradY);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(xStore, {tc.bs, dim2H}, &xDev, tc.dtype, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gradXStore, {tc.bs, dim2H}, &gradXDev, tc.dtype, &gradX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    if (tc.withWeight) {
        ret = CreateAclTensor(weightHost, {tc.bs}, &weightDev, aclDataType::ACL_FLOAT, &weight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    if (!tc.groupIndex.empty()) {
        ret = CreateAclTensor(tc.groupIndex, {static_cast<int64_t>(tc.groupIndex.size())}, &groupIndexDev,
                              aclDataType::ACL_INT64, &groupIndex);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    // grad_weight is a required output: shape [bs] when weight is provided, empty [0] otherwise.
    std::vector<float> gradWEmpty;
    if (tc.withWeight) {
        ret = CreateAclTensor(gradWHost, {tc.bs}, &gradWDev, aclDataType::ACL_FLOAT, &gradW);
    } else {
        ret = CreateAclTensor(gradWEmpty, {0}, &gradWDev, aclDataType::ACL_FLOAT, &gradW);
    }
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnSituGluGroupGradGetWorkspaceSize(gradY, x, weight, groupIndex, -1, tc.beta, tc.linearBeta,
                                                tc.activateLeft, tc.groupListType, gradX, gradW, &workspaceSize,
                                                &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituGluGroupGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnSituGluGroupGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituGluGroupGrad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    if (tc.dtype == aclDataType::ACL_FLOAT) {
        ret = aclrtMemcpy(gradXHost.data(), gradXHost.size() * sizeof(float), gradXDev,
                          gradXHost.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy grad_x failed. ERROR: %d\n", ret); return ret);
    } else {
        ret = aclrtMemcpy(gradXStore.data(), gradXStore.size() * sizeof(uint16_t), gradXDev,
                          gradXStore.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy grad_x failed. ERROR: %d\n", ret); return ret);
        for (size_t i = 0; i < gradXStore.size(); ++i) {
            gradXHost[i] = tc.dtype == aclDataType::ACL_FLOAT16 ? Fp16ToFloat(gradXStore[i]) :
                                                                  Bf16ToFloat(gradXStore[i]);
        }
    }
    if (tc.withWeight) {
        ret = aclrtMemcpy(gradWHost.data(), gradWHost.size() * sizeof(float), gradWDev,
                          gradWHost.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy grad_weight failed. ERROR: %d\n", ret); return ret);
    }

    std::vector<double> expectX;
    std::vector<double> expectW;
    Golden(tc, gradYHost, xHost, weightHost, realBs, expectX, expectW);
    int64_t errCnt = 0;
    double tol = tc.tol;
    for (size_t i = 0; i < gradXHost.size(); ++i) {
        if (expectX[i] == static_cast<double>(MAGIC)) {
            if (gradXHost[i] != MAGIC) {
                LOG_PRINT("  [FAIL] %s: padding grad_x[%zu] written: %f\n", tc.name, i, gradXHost[i]);
                errCnt++;
            }
        } else if (std::fabs(gradXHost[i] - expectX[i]) > tol + tol * std::fabs(expectX[i])) {
            LOG_PRINT("  [FAIL] %s: grad_x[%zu] = %f, expect %f\n", tc.name, i, gradXHost[i], expectX[i]);
            errCnt++;
        }
    }
    if (tc.withWeight) {
        for (int64_t n = 0; n < tc.bs; ++n) {
            if (expectW[n] == static_cast<double>(MAGIC)) {
                if (gradWHost[n] != MAGIC) {
                    LOG_PRINT("  [FAIL] %s: padding grad_weight[%ld] written: %f\n", tc.name, n, gradWHost[n]);
                    errCnt++;
                }
            } else if (std::fabs(gradWHost[n] - expectW[n]) > tol + tol * std::fabs(expectW[n])) {
                LOG_PRINT("  [FAIL] %s: grad_weight[%ld] = %f, expect %f\n", tc.name, n, gradWHost[n], expectW[n]);
                errCnt++;
            }
        }
    }

    aclDestroyTensor(gradY);
    aclDestroyTensor(x);
    if (weight != nullptr) {
        aclDestroyTensor(weight);
    }
    if (groupIndex != nullptr) {
        aclDestroyTensor(groupIndex);
    }
    aclDestroyTensor(gradX);
    if (gradW != nullptr) {
        aclDestroyTensor(gradW);
    }
    aclrtFree(gradYDev);
    aclrtFree(xDev);
    if (weightDev != nullptr) {
        aclrtFree(weightDev);
    }
    if (groupIndexDev != nullptr) {
        aclrtFree(groupIndexDev);
    }
    aclrtFree(gradXDev);
    if (gradWDev != nullptr) {
        aclrtFree(gradWDev);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (errCnt == 0) {
        LOG_PRINT("  [PASS] %s\n", tc.name);
    }
    return errCnt == 0 ? 0 : 1;
}
} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    if (!CheckHardwareSupport()) {
        LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 0;
    }

    std::vector<TestCase> cases = {
        // no weight: grad_weight must come back as an empty [0] tensor
        {"baseline_shorth", 2, 16, 4.0, 25.0, true, false, {}, 0, aclDataType::ACL_FLOAT, TOL},
        // weight + grad_weight
        {"weight_gradw_shorth", 4, 16, 1.3, 0.8, true, true, {}, 0, aclDataType::ACL_FLOAT, TOL},
        // weight + cumsum group_index (last=3 < bs=5) + grad_weight: padding rows must stay untouched
        {"weight_group_gradw", 5, 16, 1.3, 0.8, true, true, {1, 2, 3}, 0, aclDataType::ACL_FLOAT, TOL},
        // weight + count group_index (sum=3 < bs=5) + grad_weight
        {"weight_group_gradw_count", 5, 16, 1.3, 0.8, true, true, {2, 1, 0}, 1, aclDataType::ACL_FLOAT, TOL},
        // group_index only, cumsum last > bs -> clamped to bs (no weight -> empty grad_weight)
        {"group_clamp", 2, 16, 1.0, 0.0, true, false, {5, 7}, 0, aclDataType::ACL_FLOAT, TOL},
        // LongH path (dimH exceeds one UB tile), activate_left=false, up passthrough
        {"weight_gradw_longh", 2, 4096, 1.3, 0.0, false, true, {}, 0, aclDataType::ACL_FLOAT, TOL},
        // fp16 mixed path: weight + count group_index(sum=3 < bs=4) + grad_weight
        {"fp16_weight_group_gradw", 4, 16, 1.3, 25.0, true, true, {1, 2}, 1, aclDataType::ACL_FLOAT16, 1e-2},
        // bf16 mixed path, LongH, weight + grad_weight
        {"bf16_weight_gradw_longh", 2, 4096, 1.3, 0.0, false, true, {}, 0, aclDataType::ACL_BF16, 5e-2},
        // fp16 regression baseline: no optional params
        {"fp16_baseline", 2, 16, 4.0, 25.0, true, false, {}, 0, aclDataType::ACL_FLOAT16, 1e-2},
    };

    int failed = 0;
    for (const auto& tc : cases) {
        failed += RunCase(tc, stream);
    }
    LOG_PRINT(failed == 0 ? "ALL CASES PASS\n" : "%d CASE(S) FAILED\n", failed);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return failed == 0 ? 0 : 1;
}
