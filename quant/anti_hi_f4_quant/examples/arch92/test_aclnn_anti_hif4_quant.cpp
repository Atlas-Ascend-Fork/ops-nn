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
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_anti_hi_f4_quant.h"

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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnAntiHiF4QuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // PER_CHANNEL模式示例: x shape [2, 128], scale shape [2, 2, 1]
    // x有128个HiFloat4元素(物理存储为128/2=64个UINT8字节，每行)，共2行
    // scale来自DynamicHiF4Quant(axis=-1)输出，shape=[x_shape..., ceil(x[-1]/64), 1]=[2, 2, 1]
    std::vector<int64_t> xShape = {64, 128};
    std::vector<int64_t> scaleShape = {64, 2, 1}; // ceil(128/64) = 2, appended 1
    std::vector<int64_t> yShape = {64, 128};
    void* xDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* y = nullptr;

    // x: HIFLOAT4, each element 4-bit (0.5 byte), packed as uint8
    // xElementNum = 2*128 = 256, xBytes = 256/2 = 128 bytes
    int64_t xElementNum = GetShapeSize(xShape);
    size_t xBytes = xElementNum / 2;
    std::vector<uint8_t> xHostData(xBytes, 0);

    // scale: HIFLOAT4_SCALE, 本质float32 (4 bytes/element)
    // scaleShape = {2, 2, 1}, total elements = 2*2*1 = 4
    int64_t scaleElementNum = GetShapeSize(scaleShape);
    size_t scaleBytes = scaleElementNum * sizeof(float);
    std::vector<float> scaleHostData(scaleElementNum, 0.0f);

    // y: FP16, 2*128 elements, each 2 bytes
    int64_t yElementNum = GetShapeSize(yShape);
    size_t yBytes = yElementNum * sizeof(uint16_t);
    std::vector<uint16_t> yHostData(yElementNum, 0);

    ret = aclrtMalloc(&xDeviceAddr, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc x failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(xDeviceAddr, xBytes, xHostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMalloc(&scaleDeviceAddr, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc scale failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(scaleDeviceAddr, scaleBytes, scaleHostData.data(), scaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy scale failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMalloc(&yDeviceAddr, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc y failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(yDeviceAddr, yBytes, yHostData.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy y failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> xStrides = {128, 1};
    std::vector<int64_t> scaleStrides = {2, 1, 1};
    std::vector<int64_t> yStrides = {128, 1};

    x = aclCreateTensor(xShape.data(), xShape.size(), aclDataType::ACL_HIFLOAT4, xStrides.data(), 0,
                        aclFormat::ACL_FORMAT_ND, xShape.data(), xShape.size(), xDeviceAddr);
    CHECK_RET(x != nullptr, LOG_PRINT("aclCreateTensor x failed.\n"); return 1);

    scale = aclCreateTensor(scaleShape.data(), scaleShape.size(), aclDataType::ACL_HIFLOAT4_SCALE, scaleStrides.data(),
                            0, aclFormat::ACL_FORMAT_ND, scaleShape.data(), scaleShape.size(), scaleDeviceAddr);
    CHECK_RET(scale != nullptr, LOG_PRINT("aclCreateTensor scale failed.\n"); return 1);

    y = aclCreateTensor(yShape.data(), yShape.size(), aclDataType::ACL_FLOAT16, yStrides.data(), 0,
                        aclFormat::ACL_FORMAT_ND, yShape.data(), yShape.size(), yDeviceAddr);
    CHECK_RET(y != nullptr, LOG_PRINT("aclCreateTensor y failed.\n"); return 1);

    int64_t dstType = 1; // 1=FLOAT16, 27=BFLOAT16

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnAntiHiF4Quant第一段接口
    ret = aclnnAntiHiF4QuantGetWorkspaceSize(x, scale, dstType, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAntiHiF4QuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnAntiHiF4Quant第二段接口
    ret = aclnnAntiHiF4Quant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAntiHiF4Quant failed. ERROR: %d\n", ret); return ret);

    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出的值，将device侧内存上的结果拷贝至host侧
    // Copy out y: FLOAT16
    std::vector<uint16_t> yResultData(yElementNum, 0);
    ret = aclrtMemcpy(yResultData.data(), yBytes, yDeviceAddr, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("---- y result (FLOAT16) ----\n");
    for (int64_t i = 0; i < yElementNum; i++) {
        LOG_PRINT("y[%ld] is: %d\n", i, yResultData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnAntiHiF4QuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAntiHiF4QuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
