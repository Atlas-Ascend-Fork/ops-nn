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
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gemm_symm.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

float Fp16ToFloat(uint16_t h)
{
    int s = (h >> 15) & 0x1;
    int e = (h >> 10) & 0x1F;
    int f = h & 0x3FF;
    if (e == 0) {
        if (f == 0) {
            return s ? -0.0f : 0.0f;
        }
        float sig = f / 1024.0f;
        float result = sig * pow(2, -24);
        return s ? -result : result;
    } else if (e == 31) {
        return f == 0 ? (s ? -INFINITY : INFINITY) : NAN;
    }
    float result = (1.0f + f / 1024.0f) * pow(2, e - 15);
    return s ? -result : result;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

// Y = alpha * (A @ B) + beta * C. A and B are all 1s, so each product is k.
// A is (128, 64), B is (64, 128), C and Y are (128, 128). alpha = beta = 1 gives 64 + 2 = 66.
int RunGemmSymm(aclrtStream& stream, bool transposeA, bool transposeB)
{
    int ret = ACL_SUCCESS;
    const uint16_t fp16One = 0x3C00;
    const uint16_t fp16Two = 0x4000;
    std::vector<int64_t> aShape = transposeA ? std::vector<int64_t>{64, 128} : std::vector<int64_t>{128, 64};
    std::vector<int64_t> bShape = transposeB ? std::vector<int64_t>{128, 64} : std::vector<int64_t>{64, 128};
    std::vector<int64_t> squareShape = {128, 128};

    void* aDeviceAddr = nullptr;
    void* bDeviceAddr = nullptr;
    void* cDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* a = nullptr;
    aclTensor* b = nullptr;
    aclTensor* c = nullptr;
    aclTensor* y = nullptr;
    std::vector<uint16_t> aHostData(GetShapeSize(aShape), fp16One);
    std::vector<uint16_t> bHostData(GetShapeSize(bShape), fp16One);
    std::vector<uint16_t> cHostData(GetShapeSize(squareShape), fp16Two);
    std::vector<uint16_t> yHostData(GetShapeSize(squareShape), 0);
    ret = CreateAclTensor(aHostData, aShape, &aDeviceAddr, aclDataType::ACL_FLOAT16, &a);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> aTensorPtr(a, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> aDeviceAddrPtr(aDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(bHostData, bShape, &bDeviceAddr, aclDataType::ACL_FLOAT16, &b);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> bTensorPtr(b, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> bDeviceAddrPtr(bDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cHostData, squareShape, &cDeviceAddr, aclDataType::ACL_FLOAT16, &c);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> cTensorPtr(c, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> cDeviceAddrPtr(cDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, squareShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    float alphaValue = 1.0f;
    float betaValue = 1.0f;
    aclScalar* alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    aclScalar* beta = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
    std::unique_ptr<aclScalar, aclnnStatus (*)(const aclScalar*)> alphaPtr(alpha, aclDestroyScalar);
    std::unique_ptr<aclScalar, aclnnStatus (*)(const aclScalar*)> betaPtr(beta, aclDestroyScalar);
    CHECK_RET(alpha != nullptr && beta != nullptr, LOG_PRINT("aclCreateScalar failed\n");
              return ACL_ERROR_INTERNAL_ERROR);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnGemmSymmGetWorkspaceSize(a, b, c, y, alpha, beta, transposeA, transposeB, "full", 0, &workspaceSize,
                                        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmSymmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnGemmSymm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmSymm failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(squareShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    bool allMatch = true;
    for (int64_t i = 0; i < size; i++) {
        if (std::fabs(Fp16ToFloat(resultData[i]) - 66.0f) > 1e-3f) {
            LOG_PRINT("result[%ld] is: %f, expected 66.0\n", i, Fp16ToFloat(resultData[i]));
            allMatch = false;
        }
    }
    LOG_PRINT("GemmSymm transposeA=%d transposeB=%d result check: %s\n", transposeA, transposeB,
              allMatch ? "PASS" : "FAIL");
    return allMatch ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    ret = RunGemmSymm(stream, false, false);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunGemmSymm failed. ERROR: %d\n", ret); return ret);

    ret = RunGemmSymm(stream, true, true);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunGemmSymm transpose failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
