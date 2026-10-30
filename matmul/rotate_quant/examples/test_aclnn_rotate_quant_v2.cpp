/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_rotate_quant_v2.h"

static void Check(int code, const char* api)
{
    if (code != ACL_SUCCESS) {
        throw std::runtime_error(std::string(api) + " failed, code=" + std::to_string(code));
    }
}

struct Resources {
    int32_t deviceId = 0; // 根据实际设备选择 deviceId。
    bool aclReady = false;
    bool deviceReady = false;
    aclrtStream stream = nullptr;
    void* workspace = nullptr;
    std::vector<aclTensor*> tensors;
    std::vector<void*> buffers;

    ~Resources()
    {
        // 异常退出时也先等待已提交任务结束，再释放其引用的内存。
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        for (auto tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        for (auto buffer : buffers) {
            aclrtFree(buffer);
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceReady) {
            aclrtResetDevice(deviceId);
        }
        if (aclReady) {
            aclFinalize();
        }
    }
};

static aclTensor* CreateDeviceTensor(Resources& resources, const std::vector<int64_t>& shape, aclDataType dtype,
                                     const void* hostData, size_t bytes, void** deviceData)
{
    Check(aclrtMalloc(deviceData, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    resources.buffers.push_back(*deviceData);
    Check(aclrtMemcpy(*deviceData, bytes, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");

    // stride 按逻辑元素计算；Device 存储字节数由调用方按 dtype 显式提供。
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * shape[i - 1];
    }
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                        shape.data(), shape.size(), *deviceData);
    if (tensor == nullptr) {
        throw std::runtime_error("aclCreateTensor failed");
    }
    resources.tensors.push_back(tensor);
    return tensor;
}

int main()
{
    Resources resources;
    try {
        resources.tensors.reserve(4);
        resources.buffers.reserve(4);
        Check(aclInit(nullptr), "aclInit");
        resources.aclReady = true;
        Check(aclrtSetDevice(resources.deviceId), "aclrtSetDevice");
        resources.deviceReady = true;
        Check(aclrtCreateStream(&resources.stream), "aclrtCreateStream");

        const char* socName = aclrtGetSocName();
        if (socName == nullptr) {
            throw std::runtime_error("aclrtGetSocName failed");
        }
        std::string soc(socName);
        std::transform(soc.begin(), soc.end(), soc.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        const bool isInt = soc.rfind("ascend910b", 0) == 0 || soc.rfind("ascend910_93", 0) == 0;
        const bool isMx = soc.rfind("ascend950", 0) == 0 || soc.rfind("ascend350", 0) == 0 ||
                          soc.rfind("ascend960dt", 0) == 0 || soc.rfind("ascend960pr", 0) == 0;
        if (!isInt && !isMx) {
            throw std::runtime_error("Unsupported SOC for this example: " + std::string(socName));
        }
        std::printf("Current SOC: %s; mode=ordinary, y=%s, scale=%s\n", socName, isMx ? "FLOAT8_E4M3FN" : "INT8",
                    isMx ? "FLOAT8_E8M0" : "FLOAT");

        const int64_t M = 1024;
        const int64_t N = 256;
        const int64_t K = 64;
        const int64_t scaleSlots = 2 * ((N + 63) / 64);
        const int64_t groupListType = 0;

        std::vector<int64_t> xShape = {M, N};
        std::vector<int64_t> rotShape = {K, K};
        std::vector<int64_t> yShape = {M, N};
        std::vector<int64_t> scaleShape = isMx ? std::vector<int64_t>{M, (N + 63) / 64, 2} : std::vector<int64_t>{M};

        // uint16_t 存储 FP16 原始位：0x3800=0.5，0x3c00=1.0。使用二维单位旋转矩阵。
        std::vector<uint16_t> xHost(M * N, 0x3800);
        std::vector<uint16_t> rotHost(K * K, 0);
        for (int64_t j = 0; j < K; ++j) {
            rotHost[j * K + j] = 0x3c00;
        }

        // INT8 和 FP8 每元素均为一字节；FP8 分支仅打印原始编码。
        std::vector<int8_t> yHost(M * N, 0);
        std::vector<float> scaleFloatHost(isMx ? 0 : M, 0.0f);
        std::vector<uint8_t> scaleMxHost(isMx ? M * scaleSlots : 0, 0);
        void* scaleHostData = isMx ? static_cast<void*>(scaleMxHost.data()) : static_cast<void*>(scaleFloatHost.data());
        const size_t yBytes = yHost.size() * sizeof(int8_t);
        const size_t scaleBytes = isMx ? scaleMxHost.size() * sizeof(uint8_t) : scaleFloatHost.size() * sizeof(float);
        void* xDevice = nullptr;
        void* rotDevice = nullptr;
        void* yDevice = nullptr;
        void* scaleDevice = nullptr;

        aclTensor* x = CreateDeviceTensor(resources, xShape, ACL_FLOAT16, xHost.data(), xHost.size() * sizeof(uint16_t),
                                          &xDevice);
        aclTensor* rotation = CreateDeviceTensor(resources, rotShape, ACL_FLOAT16, rotHost.data(),
                                                 rotHost.size() * sizeof(uint16_t), &rotDevice);
        aclTensor* yOut = CreateDeviceTensor(resources, yShape, isMx ? ACL_FLOAT8_E4M3FN : ACL_INT8, yHost.data(),
                                             yBytes, &yDevice);
        aclTensor* scaleOut = CreateDeviceTensor(resources, scaleShape, isMx ? ACL_FLOAT8_E8M0 : ACL_FLOAT,
                                                 scaleHostData, scaleBytes, &scaleDevice);

        int64_t axis = -1;
        char roundMode[] = "rint";
        int64_t scaleAlg = 0;
        double dstTypeMax = 0.0;
        bool trans = false;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;

        // V2 兼容普通模式：所有平台均传入空分组列表，groupListType 显式采用默认语义值 0。
        Check(
            aclnnRotateQuantV2GetWorkspaceSize(x, rotation, nullptr, nullptr, groupListType, axis, roundMode, scaleAlg,
                                               dstTypeMax, trans, yOut, scaleOut, &workspaceSize, &executor),
            "aclnnRotateQuantV2GetWorkspaceSize");
        if (workspaceSize != 0) {
            Check(aclrtMalloc(&resources.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
        }
        Check(aclnnRotateQuantV2(resources.workspace, workspaceSize, executor, resources.stream), "aclnnRotateQuantV2");
        Check(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream");

        Check(aclrtMemcpy(yHost.data(), yBytes, yDevice, yBytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy Y D2H");
        Check(aclrtMemcpy(scaleHostData, scaleBytes, scaleDevice, scaleBytes, ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy scale D2H");

        if (isMx) {
            // FP8/E8M0 的原始编码不能直接解释为量化结果的整数数值。
            std::printf("Y first code=0x%02x, scale first code=0x%02x\n",
                        static_cast<unsigned>(static_cast<uint8_t>(yHost[0])), static_cast<unsigned>(scaleMxHost[0]));
        } else {
            std::printf("Y first value=%d, scale first value=%f\n", static_cast<int>(yHost[0]),
                        static_cast<double>(scaleFloatHost[0]));
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
