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
#include "aclnnop/aclnn_dynamic_block_hi_f4_quant.h"

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
        strides[i] = strides[i + 1] * shape[i + 1];
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

    // x shape=[128, 128], bf16，-1轴和-2轴均为64的倍数
    std::vector<int64_t> xShape = {128, 128};
    int64_t xSize = GetShapeSize(xShape);
    std::vector<int16_t> xHostData(xSize, 1);
    void* xDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    CreateAclTensor(xHostData, xShape, &xDeviceAddr, ACL_BF16, &x);

    // y shape=[128, 128], 数据类型HIFLOAT4
    std::vector<int64_t> yShape = {128, 128};
    int64_t ySize = GetShapeSize(yShape);
    std::vector<uint8_t> yHostData(ySize, 0);
    void* yDeviceAddr = nullptr;
    aclTensor* y = nullptr;
    CreateAclTensor(yHostData, yShape, &yDeviceAddr, ACL_HIFLOAT4, &y);

    // scale1 shape=[128, 2, 1], HIFLOAT4_SCALE（-1轴方向）
    std::vector<int64_t> scale1Shape = {128, 2, 1};
    int64_t scale1Size = GetShapeSize(scale1Shape);
    std::vector<uint8_t> scale1HostData(scale1Size, 0);
    void* scale1DeviceAddr = nullptr;
    aclTensor* scale1 = nullptr;
    CreateAclTensor(scale1HostData, scale1Shape, &scale1DeviceAddr, ACL_HIFLOAT4_SCALE, &scale1);

    // scale2 shape=[2, 128, 1], HIFLOAT4_SCALE（-2轴方向）
    std::vector<int64_t> scale2Shape = {2, 128, 1};
    int64_t scale2Size = GetShapeSize(scale2Shape);
    std::vector<uint8_t> scale2HostData(scale2Size, 0);
    void* scale2DeviceAddr = nullptr;
    aclTensor* scale2 = nullptr;
    CreateAclTensor(scale2HostData, scale2Shape, &scale2DeviceAddr, ACL_HIFLOAT4_SCALE, &scale2);

    // 获取 workspace 大小
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    char* roundMode = const_cast<char*>("rint");
    int64_t scaleAlg = 0;
    ret = aclnnDynamicBlockHiF4QuantGetWorkspaceSize(x, roundMode, scaleAlg, y, scale1, scale2, &workspaceSize,
                                                     &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicBlockHiF4QuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 分配 workspace 并执行
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnDynamicBlockHiF4Quant(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicBlockHiF4Quant failed. ERROR: %d\n", ret); return ret);

    // 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 释放资源
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(scale1DeviceAddr);
    aclrtFree(scale2DeviceAddr);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclDestroyTensor(scale1);
    aclDestroyTensor(scale2);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
