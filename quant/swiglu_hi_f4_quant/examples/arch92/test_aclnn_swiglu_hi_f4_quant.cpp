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
#include "aclnnop/aclnn_swiglu_hi_f4_quant.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, size_t bytes, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
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
    // 1. device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    // x: shape [1, 128], bf16类型，最后一维128是64的两倍（满足SwiGLU切分要求）且能被128整除
    // yOut: shape [1, 64], HIFLOAT4类型
    // scaleOut: shape [1, 1, 1], HIFLOAT4_SCALE类型（ceil(64/64)=1，带尾部dim=1）
    std::vector<int64_t> xShape = {1, 128};
    std::vector<int64_t> yOutShape = {1, 64};
    std::vector<int64_t> scaleOutShape = {1, 1, 1};

    void* xDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* scaleOutDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* out = nullptr;
    aclTensor* scaleOut = nullptr;

    // 填充x数据（128个bf16值，简单递增）
    std::vector<uint16_t> xHostData(128, 0);
    for (int i = 0; i < 128; i++) {
        // bf16: 设置简单的浮点值（1.0的bf16表示为0x3f80，2.0为0x4000）
        // 这里用简单的整数值映射到bf16的指数部分做示意
        xHostData[i] = static_cast<uint16_t>(0x3f00 + i % 16);
    }
    // yOut数据（HIFLOAT4类型，每字节包含2个fp4元素，shape[1,64]共32字节）
    std::vector<uint8_t> outHostData(32, 0);
    // scaleOut数据（将HIFLOAT4_SCALE用FLOAT类型承载，shape[1,1,1]共4字节）
    std::vector<float> scaleOutHostData(1, 0.0f);

    int64_t activateDim = -1;
    bool activateLeft = true;
    int64_t swigluMode = 0;
    float clampLimit = 7.0f;
    float gluAlpha = 1.702f;
    float gluBias = 1.0f;
    int64_t axis = -1;
    int64_t scaleAlg = 0;

    // 创建x aclTensor (BF16)
    int64_t xElementNum = GetShapeSize(xShape);
    size_t xBytes = xElementNum * sizeof(uint16_t);
    ret = CreateAclTensor(xHostData, xShape, xBytes, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor (HIFLOAT4)
    int64_t yOutElementNum = GetShapeSize(yOutShape);
    size_t yBytes = yOutElementNum * sizeof(uint8_t) / 2;
    ret = CreateAclTensor(outHostData, yOutShape, yBytes, &outDeviceAddr, aclDataType::ACL_HIFLOAT4, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scaleOut aclTensor (FLOAT)
    int64_t scaleOutElementNum = GetShapeSize(scaleOutShape);
    size_t scaleBytes = scaleOutElementNum * sizeof(float);
    ret = CreateAclTensor(scaleOutHostData, scaleOutShape, scaleBytes, &scaleOutDeviceAddr,
                          aclDataType::ACL_HIFLOAT4_SCALE, &scaleOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnSwigluHiF4Quant第一段接口
    ret = aclnnSwigluHiF4QuantGetWorkspaceSize(x, nullptr, activateDim, activateLeft, swigluMode, clampLimit, gluAlpha,
                                               gluBias, axis, scaleAlg, out, scaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluHiF4QuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnSwigluHiF4Quant第二段接口
    ret = aclnnSwigluHiF4Quant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluHiF4Quant failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值
    auto yHostElementNum = GetShapeSize(yOutShape) / 2;
    std::vector<uint8_t> yResultData(yHostElementNum, 0);
    ret = aclrtMemcpy(yResultData.data(), yBytes, outDeviceAddr, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y result from device to host failed. ERROR: %d\n", ret); return ret);

    auto scaleHostElementNum = GetShapeSize(scaleOutShape);
    std::vector<float> scaleResultData(scaleHostElementNum, 0.0f);
    ret = aclrtMemcpy(scaleResultData.data(), scaleHostElementNum * sizeof(float), scaleOutDeviceAddr,
                      scaleHostElementNum * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale result from device to host failed. ERROR: %d\n", ret);
              return ret);

    for (int64_t i = 0; i < std::min(yHostElementNum, (int64_t)16); i++) {
        LOG_PRINT("y[%ld] is: 0x%02x (包含2个fp4的values)\n", i, yResultData[i]);
    }
    for (int64_t i = 0; i < scaleHostElementNum; i++) {
        LOG_PRINT("scale[%ld] is: %f\n", i, scaleResultData[i]);
    }

    // 6. 释放aclTensor
    aclDestroyTensor(x);
    aclDestroyTensor(out);
    aclDestroyTensor(scaleOut);
    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(scaleOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
