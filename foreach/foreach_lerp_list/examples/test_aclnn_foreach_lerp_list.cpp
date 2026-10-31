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
#include <memory>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_lerp_list.h"

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

using StreamPtr = std::unique_ptr<std::remove_pointer<aclrtStream>::type, decltype(&aclrtDestroyStream)>;
using DeviceMemPtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
using TensorListPtr = std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)>;

void PrintOutResult(const std::vector<int64_t>& shape, const DeviceMemPtr& deviceAddr, const char* name)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceAddr.get(),
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("%s result[%ld] is: %f\n", name, i, resultData[i]);
    }
}

int Init(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet)
{
    // 固定写法，初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    initialized = true;
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    deviceSet = true;
    aclrtStream rawStream = nullptr;
    ret = aclrtCreateStream(&rawStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    stream.reset(rawStream);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    DeviceMemPtr& deviceAddr, TensorPtr& tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    void* rawDeviceAddr = nullptr;
    auto ret = aclrtMalloc(&rawDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    deviceAddr.reset(rawDeviceAddr);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(deviceAddr.get(), size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    aclTensor* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), deviceAddr.get());
    CHECK_RET(rawTensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    tensor.reset(rawTensor);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    bool initialized = false;
    bool deviceSet = false;
    std::shared_ptr<void> aclGuard(nullptr, [&](void*) {
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (initialized) {
            aclFinalize();
        }
    });
    StreamPtr stream(nullptr, &aclrtDestroyStream);
    auto ret = Init(deviceId, stream, initialized, deviceSet);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape1 = {2, 3};
    std::vector<int64_t> selfShape2 = {1, 3};
    std::vector<int64_t> otherShape1 = {2, 3};
    std::vector<int64_t> otherShape2 = {1, 3};
    std::vector<int64_t> weightShape1 = {2, 3};
    std::vector<int64_t> weightShape2 = {1, 3};
    std::vector<int64_t> outShape1 = {2, 3};
    std::vector<int64_t> outShape2 = {1, 3};
    std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> input2HostData = {7, 8, 9};
    std::vector<float> other1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> other2HostData = {7, 8, 9};
    std::vector<float> weight1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> weight2HostData = {7, 8, 9};
    std::vector<float> out1HostData(6, 0);
    std::vector<float> out2HostData(3, 0);

    DeviceMemPtr input1DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr input2DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr other1DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr other2DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr weight1DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr weight2DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr out1DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr out2DeviceAddr(nullptr, &aclrtFree);

    TensorPtr input1(nullptr, &aclDestroyTensor);
    TensorPtr input2(nullptr, &aclDestroyTensor);
    TensorPtr other1(nullptr, &aclDestroyTensor);
    TensorPtr other2(nullptr, &aclDestroyTensor);
    TensorPtr weight1(nullptr, &aclDestroyTensor);
    TensorPtr weight2(nullptr, &aclDestroyTensor);
    TensorPtr out1(nullptr, &aclDestroyTensor);
    TensorPtr out2(nullptr, &aclDestroyTensor);

    TensorListPtr tensorListInput1(nullptr, &aclDestroyTensorList);
    TensorListPtr tensorListInput2(nullptr, &aclDestroyTensorList);
    TensorListPtr tensorListWeight(nullptr, &aclDestroyTensorList);
    TensorListPtr tensorListOutput(nullptr, &aclDestroyTensorList);

    // 创建input1 aclTensor
    ret = CreateAclTensor(input1HostData, selfShape1, aclDataType::ACL_FLOAT, input1DeviceAddr, input1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create input1 failed. ERROR: %d\n", ret); return ret);
    // 创建input2 aclTensor
    ret = CreateAclTensor(input2HostData, selfShape2, aclDataType::ACL_FLOAT, input2DeviceAddr, input2);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create input2 failed. ERROR: %d\n", ret); return ret);
    // 创建other1 aclTensor
    ret = CreateAclTensor(other1HostData, otherShape1, aclDataType::ACL_FLOAT, other1DeviceAddr, other1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create other1 failed. ERROR: %d\n", ret); return ret);
    // 创建other2 aclTensor
    ret = CreateAclTensor(other2HostData, otherShape2, aclDataType::ACL_FLOAT, other2DeviceAddr, other2);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create other2 failed. ERROR: %d\n", ret); return ret);
    // 创建weight1 aclTensor
    ret = CreateAclTensor(weight1HostData, weightShape1, aclDataType::ACL_FLOAT, weight1DeviceAddr, weight1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create weight1 failed. ERROR: %d\n", ret); return ret);
    // 创建weight2 aclTensor
    ret = CreateAclTensor(weight2HostData, weightShape2, aclDataType::ACL_FLOAT, weight2DeviceAddr, weight2);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create weight2 failed. ERROR: %d\n", ret); return ret);
    // 创建out1 aclTensor
    ret = CreateAclTensor(out1HostData, outShape1, aclDataType::ACL_FLOAT, out1DeviceAddr, out1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create out1 failed. ERROR: %d\n", ret); return ret);
    // 创建out2 aclTensor
    ret = CreateAclTensor(out2HostData, outShape2, aclDataType::ACL_FLOAT, out2DeviceAddr, out2);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create out2 failed. ERROR: %d\n", ret); return ret);

    // 创建TensorList
    std::vector<aclTensor*> tempInput1{input1.get(), input2.get()};
    tensorListInput1.reset(aclCreateTensorList(tempInput1.data(), tempInput1.size()));
    CHECK_RET(tensorListInput1 != nullptr, LOG_PRINT("aclCreateTensorList failed.\n"); return ACL_ERROR_FAILURE);
    // aclCreateTensorList接管aclTensor的所有权，释放本地智能指针避免重复释放
    input1.release();
    input2.release();

    std::vector<aclTensor*> tempInput2{other1.get(), other2.get()};
    tensorListInput2.reset(aclCreateTensorList(tempInput2.data(), tempInput2.size()));
    CHECK_RET(tensorListInput2 != nullptr, LOG_PRINT("aclCreateTensorList failed.\n"); return ACL_ERROR_FAILURE);
    other1.release();
    other2.release();

    std::vector<aclTensor*> tempWeight{weight1.get(), weight2.get()};
    tensorListWeight.reset(aclCreateTensorList(tempWeight.data(), tempWeight.size()));
    CHECK_RET(tensorListWeight != nullptr, LOG_PRINT("aclCreateTensorList failed.\n"); return ACL_ERROR_FAILURE);
    weight1.release();
    weight2.release();

    std::vector<aclTensor*> tempOutput{out1.get(), out2.get()};
    tensorListOutput.reset(aclCreateTensorList(tempOutput.data(), tempOutput.size()));
    CHECK_RET(tensorListOutput != nullptr, LOG_PRINT("aclCreateTensorList failed.\n"); return ACL_ERROR_FAILURE);
    out1.release();
    out2.release();

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnForeachLerpList第一段接口
    ret = aclnnForeachLerpListGetWorkspaceSize(tensorListInput1.get(), tensorListInput2.get(), tensorListWeight.get(),
                                               tensorListOutput.get(), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachLerpListGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);
    if (workspaceSize > 0) {
        void* rawWorkspaceAddr = nullptr;
        ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddr.reset(rawWorkspaceAddr);
    }
    // 调用aclnnForeachLerpList第二段接口
    ret = aclnnForeachLerpList(workspaceAddr.get(), workspaceSize, executor, stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachLerpList failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    PrintOutResult(outShape1, out1DeviceAddr, "out1");
    PrintOutResult(outShape2, out2DeviceAddr, "out2");

    return 0;
}
