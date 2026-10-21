/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dynamic_hi_f4_quant.h"

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // x shape: [2, 128], last dim is multiple of 64
    std::vector<int64_t> xShape = {2, 128};
    std::vector<int64_t> yShape = {2, 128};
    // scale shape: ceil(128/64)=2 on axis=-1, append dim 1 => [2, 2, 1]
    std::vector<int64_t> scaleShape = {2, 2, 1};

    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* scale = nullptr;

    // x: FP16, 2*128 elements, each 2 bytes
    int64_t xElementNum = GetShapeSize(xShape);
    std::vector<aclFloat16> xHostData(xElementNum, 0);
    for (int64_t i = 0; i < xElementNum; i++) {
        xHostData[i] = aclFloat16(i % 10 + 1);
    }

    // y: HIFLOAT4, each element 4-bit (0.5 byte), packed as uint8
    // aclrtMalloc size = elementCount * sizeof(element) = xElementNum * 0.5 = xElementNum / 2 bytes
    // Host side uses uint8_t vector, size = xElementNum / 2
    int64_t yElementNum = xElementNum;
    size_t yBytes = yElementNum / 2;
    std::vector<uint8_t> yHostData(yBytes, 0);

    // scale: HIFLOAT4_SCALE, 本质float32 (4 bytes)
    // scaleShape = [2, 2, 1], total elements = 4
    int64_t scaleElementNum = GetShapeSize(scaleShape);
    size_t scaleBytes = scaleElementNum * sizeof(float);
    std::vector<float> scaleHostData(scaleElementNum, 0.0f);

    size_t xBytes = xElementNum * sizeof(aclFloat16);
    ret = aclrtMalloc(&xDeviceAddr, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc x failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(xDeviceAddr, xBytes, xHostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMalloc(&yDeviceAddr, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc y failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(yDeviceAddr, yBytes, yHostData.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy y failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMalloc(&scaleDeviceAddr, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc scale failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(scaleDeviceAddr, scaleBytes, scaleHostData.data(), scaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy scale failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> xStrides = {128, 1};
    std::vector<int64_t> yStrides = {128, 1};
    std::vector<int64_t> scaleStrides = {2, 1, 1};

    x = aclCreateTensor(xShape.data(), xShape.size(), aclDataType::ACL_FLOAT16, xStrides.data(), 0,
                        aclFormat::ACL_FORMAT_ND, xShape.data(), xShape.size(), xDeviceAddr);
    CHECK_RET(x != nullptr, LOG_PRINT("aclCreateTensor x failed.\n"); return 1);

    y = aclCreateTensor(yShape.data(), yShape.size(), aclDataType::ACL_HIFLOAT4, yStrides.data(), 0,
                        aclFormat::ACL_FORMAT_ND, yShape.data(), yShape.size(), yDeviceAddr);
    CHECK_RET(y != nullptr, LOG_PRINT("aclCreateTensor y failed.\n"); return 1);

    scale = aclCreateTensor(scaleShape.data(), scaleShape.size(), aclDataType::ACL_HIFLOAT4_SCALE, scaleStrides.data(),
                            0, aclFormat::ACL_FORMAT_ND, scaleShape.data(), scaleShape.size(), scaleDeviceAddr);
    CHECK_RET(scale != nullptr, LOG_PRINT("aclCreateTensor scale failed.\n"); return 1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    int64_t axis = -1;
    int64_t scaleAlg = 0;

    ret = aclnnDynamicHiF4QuantGetWorkspaceSize(x, axis, scaleAlg, y, scale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicHiF4QuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnDynamicHiF4Quant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicHiF4Quant failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Copy out y: HIFLOAT4, yBytes = yElementNum / 2
    std::vector<uint8_t> yResultData(yBytes, 0);
    ret = aclrtMemcpy(yResultData.data(), yBytes, yDeviceAddr, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y result from device to host failed. ERROR: %d\n", ret); return ret);

    // Copy out scale: HIFLOAT4_SCALE, 本质float32
    std::vector<float> scaleResultData(scaleElementNum, 0.0f);
    ret = aclrtMemcpy(scaleResultData.data(), scaleBytes, scaleDeviceAddr, scaleBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale result from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("---- y result (HIFLOAT4 packed as uint8) ----\n");
    for (int64_t i = 0; i < std::min(static_cast<int64_t>(yBytes), int64_t(16)); i++) {
        LOG_PRINT("y[%ld] = 0x%02x\n", i, yResultData[i]);
    }
    LOG_PRINT("---- scale result (HIFLOAT4_SCALE, float32) ----\n");
    for (int64_t i = 0; i < scaleElementNum; i++) {
        LOG_PRINT("scale[%ld] = %f\n", i, scaleResultData[i]);
    }

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclDestroyTensor(scale);

    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
