/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_selu_backward.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> shape = {4, 2};
    void* gradOutputDeviceAddr = nullptr;
    void* resultDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* result = nullptr;
    aclTensor* gradInput = nullptr;

    // SELU 常量
    const float SCALE = 1.0507009873554804f;
    const float ALPHA = 1.6732632423543772f;
    const float SCALE_ALPHA_PRODUCT = SCALE * ALPHA;

    // 构造输入: gradOutput = 全1, result = [-2, -1, 0, 1, 2, 3, -0.5, 0.5]
    std::vector<float> gradOutputHostData = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> resultHostData = {-2, -1, 0, 1, 2, 3, -0.5, 0.5};
    std::vector<float> gradInputHostData(8, 0);

    ret = CreateAclTensor(gradOutputHostData, shape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(resultHostData, shape, &resultDeviceAddr, aclDataType::ACL_FLOAT, &result);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradInputHostData, shape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnSeluBackwardGetWorkspaceSize(gradOutput, result, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSeluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnSeluBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSeluBackward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(shape);
    std::vector<float> gradInputData(size, 0);
    ret = aclrtMemcpy(gradInputData.data(), gradInputData.size() * sizeof(float), gradInputDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradInput failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("\n=== SeluGrad Results ===\n");
    LOG_PRINT("result >= 0: gradInput = SCALE * gradOutput = %.6f * grad\n", SCALE);
    LOG_PRINT("result <  0: gradInput = gradOutput * (result + SCALE_ALPHA) = grad * (result + %.6f)\n\n",
              SCALE_ALPHA_PRODUCT);

    int pass = 0, fail = 0;
    for (int64_t i = 0; i < size; i++) {
        float expected;
        if (resultHostData[i] < 0) {
            expected = gradOutputHostData[i] * (resultHostData[i] + SCALE_ALPHA_PRODUCT);
        } else {
            expected = SCALE * gradOutputHostData[i];
        }
        bool ok = std::fabs(gradInputData[i] - expected) < 0.01f;
        if (ok)
            pass++;
        else
            fail++;
        LOG_PRINT("  [%ld] result=%.2f grad=%.2f => NPU=%.6f  expected=%.6f  %s\n", i, resultHostData[i],
                  gradOutputHostData[i], gradInputData[i], expected, ok ? "PASS" : "FAIL");
    }
    LOG_PRINT("\nTotal: %d PASS, %d FAILED\n", pass, fail);

    aclDestroyTensor(gradOutput);
    aclDestroyTensor(result);
    aclDestroyTensor(gradInput);
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(resultDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return fail > 0 ? 1 : 0;
}
