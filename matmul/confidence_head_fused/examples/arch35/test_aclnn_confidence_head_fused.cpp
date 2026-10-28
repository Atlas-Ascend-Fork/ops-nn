/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_confidence_head_fused.h"

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

struct DeviceTensor {
    void* data = nullptr;
    aclTensor* tensor = nullptr;

    ~DeviceTensor()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (data != nullptr) {
            aclrtFree(data);
        }
    }
};

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    DeviceTensor& tensor)
{
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(&tensor.data, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(tensor.data, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    tensor.tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                                    shape.data(), shape.size(), tensor.data);
    CHECK_RET(tensor.tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return 1);
    return ACL_SUCCESS;
}

int AclnnConfidenceHeadFusedTest(aclrtStream stream)
{
    // 1. 构造三维 FP16 特征和 Markov 特征，最后一维拼接后与 weight 做线性投影。
    // FP16 常数 1 和 0.5 分别使用真实的 16 位编码 0x3C00 和 0x3800。
    std::vector<int64_t> hiddenShape = {2, 3, 2};
    std::vector<int64_t> markovShape = {2, 3, 1};
    std::vector<int64_t> weightShape = {3};
    std::vector<int64_t> temperatureShape = {1};
    std::vector<int64_t> biasShape = {3};
    std::vector<int64_t> outShape = {2, 3};
    std::vector<uint16_t> hiddenHostData(12, 0x3C00);
    std::vector<uint16_t> markovHostData(6, 0x3C00);
    std::vector<uint16_t> weightHostData(3, 0x3800);
    std::vector<float> temperatureHostData = {2.0f};
    std::vector<float> biasHostData = {-1.0f, 0.0f, 1.0f};
    std::vector<float> outHostData(6, 0.0f);
    DeviceTensor hidden, weight, markov, temperature, bias, out;

    auto ret = CreateAclTensor(hiddenHostData, hiddenShape, ACL_FLOAT16, hidden);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, ACL_FLOAT16, weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(markovHostData, markovShape, ACL_FLOAT16, markov);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(temperatureHostData, temperatureShape, ACL_FLOAT, temperature);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, ACL_FLOAT, bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, ACL_FLOAT, out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 2. 构造 Scalar 截断边界；可选参数不使用时可以传入 nullptr。
    float clampLowValue = -60.0f;
    float clampHighValue = 60.0f;
    std::unique_ptr<aclScalar, decltype(&aclDestroyScalar)> clampLow(aclCreateScalar(&clampLowValue, ACL_FLOAT),
                                                                     aclDestroyScalar);
    std::unique_ptr<aclScalar, decltype(&aclDestroyScalar)> clampHigh(aclCreateScalar(&clampHighValue, ACL_FLOAT),
                                                                      aclDestroyScalar);
    CHECK_RET(clampLow != nullptr && clampHigh != nullptr, LOG_PRINT("aclCreateScalar failed.\n"); return 1);

    // 3. 调用第一段接口，获取执行器及 workspace 大小。
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnConfidenceHeadFusedGetWorkspaceSize(hidden.tensor, weight.tensor, markov.tensor, temperature.tensor,
                                                   bias.tensor, clampLow.get(), clampHigh.get(), out.tensor,
                                                   &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConfidenceHeadFusedGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    std::unique_ptr<void, decltype(&aclrtFree)> workspace(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspace.reset(workspaceAddr);
    }

    // 4. 调用第二段接口，并在释放设备资源前同步等待任务结束。
    ret = aclnnConfidenceHeadFused(workspaceAddr, workspaceSize, executor, stream);
    auto syncRet = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConfidenceHeadFused failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(syncRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", syncRet); return syncRet);

    // 5. 获取 FP32 输出。bias[N] 在 batch 间共享，两批预期均为 [0.5621765, 0.6791787, 0.7772999]。
    ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(float), out.data,
                      outHostData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (size_t i = 0; i < outHostData.size(); ++i) {
        LOG_PRINT("result[%zu] is: %.7f\n", i, outHostData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = AclnnConfidenceHeadFusedTest(stream);
    Finalize(deviceId, stream);
    return ret;
}
