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
 * \file test_aclnn_gemmv3_accum.cpp
 * \brief GemmV3 C 原地累加示例：out = A@B + C。
 *
 * GemmV3 支持 alpha=beta=1 且输出为原地累加语义：kernel 把 A@B 原子累加到输出缓冲上，
 * 因此调用前需把 C 的值预置到 out（或令 out 与 C 共用同一块内存），结果 out = C + A@B。
 * 本目录为 arch35 示例：C/out 为 FP32，A/B 支持 FP16/BF16/FP32。
 */

#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/level2/aclnn_gemm.h"

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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int AclnnGemmv3AccumTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // M=N=K=2：A = [[1,2],[3,4]]，B = [[1,0],[0,1]]（单位阵），C = [[10,20],[30,40]]
    std::vector<int64_t> AShape = {2, 2};
    std::vector<int64_t> BShape = {2, 2};
    std::vector<int64_t> CShape = {2, 2};
    std::vector<int64_t> outShape = {2, 2};
    void* ADeviceAddr = nullptr;
    void* BDeviceAddr = nullptr;
    void* CDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* A = nullptr;
    aclTensor* B = nullptr;
    aclTensor* C = nullptr;
    aclTensor* out = nullptr;

    // A/B 用 BF16 承载（0x3F80=1.0, 0x4000=2.0, 0x4040=3.0, 0x4080=4.0）
    std::vector<uint16_t> AHostData = {0x3F80, 0x4000, 0x4040, 0x4080};
    std::vector<uint16_t> BHostData = {0x3F80, 0x0000, 0x0000, 0x3F80};
    // C 与 out 均为 FP32（arch35 上 GemmV3 的 C/out 仅支持 FP32）
    std::vector<float> CHostData = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> outHostData = CHostData; // 预置 C 到 out，kernel 在其上原子累加 A@B
    float alpha = 1.0f;
    float beta = 1.0f;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = 1;

    ret = CreateAclTensor(AHostData, AShape, &ADeviceAddr, aclDataType::ACL_BF16, &A);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> ATensorPtr(A, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> AdeviceAddrPtr(ADeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(BHostData, BShape, &BDeviceAddr, aclDataType::ACL_BF16, &B);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> BTensorPtr(B, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> BdeviceAddrPtr(BDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(CHostData, CShape, &CDeviceAddr, aclDataType::ACL_FLOAT, &C);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> CTensorPtr(C, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> CdeviceAddrPtr(CDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> executorAddrPtr(nullptr, aclrtFree);
    ret = aclnnGemmGetWorkspaceSize(A, B, C, alpha, beta, transA, transB, out, cubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
        executorAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnGemm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemm failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // 期望 out = C + A@B = [[11,22],[33,44]]
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = AclnnGemmv3AccumTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnGemmv3AccumTest failed. ERROR: %d\n", ret); return ret);
    Finalize(deviceId, stream);
    return 0;
}
