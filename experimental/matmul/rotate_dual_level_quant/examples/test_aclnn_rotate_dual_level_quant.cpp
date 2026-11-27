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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rotate_dual_level_quant.h"

namespace {
void Check(int status, const char* api)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(api) + " failed, status=" + std::to_string(status));
    }
}

struct Resources {
    int32_t deviceId = 0;
    bool aclReady = false;
    bool deviceReady = false;
    aclrtStream stream = nullptr;
    void* workspace = nullptr;
    std::vector<void*> buffers;
    std::vector<aclTensor*> tensors;

    ~Resources()
    {
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        for (auto* tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        for (void* buffer : buffers) {
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

aclTensor* CreateTensor(Resources& resources, const std::vector<int64_t>& shape, aclDataType dtype, const void* initial,
                        size_t bytes, void*& address)
{
    // Give empty tensors a valid backing address without copying nonexistent elements.
    Check(aclrtMalloc(&address, std::max<size_t>(bytes, 1), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    resources.buffers.push_back(address);
    if (bytes != 0) {
        Check(aclrtMemcpy(address, bytes, initial, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * shape[i - 1];
    }
    auto* tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                   shape.size(), address);
    if (tensor == nullptr) {
        throw std::runtime_error("aclCreateTensor failed");
    }
    resources.tensors.push_back(tensor);
    return tensor;
}
} // namespace

int main(int argc, char** argv)
{
    Resources resources;
    try {
        if (argc > 3) {
            throw std::runtime_error("Usage: example [fp4|e4m3|e5m2] [ordinary|count|cumsum|empty]");
        }
        const std::string type = argc > 1 ? argv[1] : "fp4";
        const std::string mode = argc > 2 ? argv[2] : "ordinary";
        if (type != "fp4" && type != "e4m3" && type != "e5m2") {
            throw std::runtime_error("Unknown output dtype");
        }
        if (mode != "ordinary" && mode != "count" && mode != "cumsum" && mode != "empty") {
            throw std::runtime_error("Unknown grouping mode");
        }
        resources.buffers.reserve(8);
        resources.tensors.reserve(8);
        Check(aclInit(nullptr), "aclInit");
        resources.aclReady = true;
        Check(aclrtSetDevice(resources.deviceId), "aclrtSetDevice");
        resources.deviceReady = true;
        Check(aclrtCreateStream(&resources.stream), "aclrtCreateStream");
        const char* name = aclrtGetSocName();
        std::string soc = name == nullptr ? "" : name;
        std::transform(soc.begin(), soc.end(), soc.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (soc.rfind("ascend950", 0) != 0) {
            throw std::runtime_error("This example requires Ascend950 (A5)");
        }

        const bool grouped = mode == "count" || mode == "cumsum";
        const int64_t m = mode == "empty" ? 0 : 4;
        const int64_t n = 576; // Demonstrates a 512-wide block plus a 64-element tail.
        const int64_t k = 64;
        const int64_t e = grouped ? 2 : 1;
        const int64_t level0BlockSize = 512;
        const int64_t level1BlockSize = 32;
        const int64_t level0Count = (n + level0BlockSize - 1) / level0BlockSize;
        const int64_t level1Pairs = (n + level1BlockSize * 2 - 1) / (level1BlockSize * 2);
        const auto dtype = type == "fp4" ? ACL_FLOAT4_E2M1 : (type == "e4m3" ? ACL_FLOAT8_E4M3FN : ACL_FLOAT8_E5M2);
        const size_t yBytes = static_cast<size_t>(m * n) / (type == "fp4" ? 2 : 1);
        std::vector<uint16_t> xData(static_cast<size_t>(m * n), 0x3c00); // FP16 1.0.
        std::vector<uint16_t> rotData(static_cast<size_t>(e * k * k), 0);
        for (int64_t group = 0; group < e; ++group) {
            for (int64_t i = 0; i < k; ++i) {
                rotData[static_cast<size_t>(group * k * k + i * k + i)] = group == 0 ? 0x3c00 : 0xbc00;
            }
        }
        std::vector<uint8_t> yData(yBytes, 0);
        std::vector<float> level0Data(static_cast<size_t>(m * level0Count), -1.0f);
        std::vector<uint8_t> level1Data(static_cast<size_t>(m * level1Pairs * 2), 0);
        // Groups cover rows [0,3); row 3 retains initial y/level0/level1 values.
        std::vector<int64_t> groupData = mode == "cumsum" ? std::vector<int64_t>{2, 3} : std::vector<int64_t>{2, 1};
        void* xAddress = nullptr;
        void* rotAddress = nullptr;
        void* yAddress = nullptr;
        void* level0Address = nullptr;
        void* level1Address = nullptr;
        void* groupsAddress = nullptr;
        auto* x = CreateTensor(resources, {m, n}, ACL_FLOAT16, xData.data(), xData.size() * sizeof(uint16_t), xAddress);
        const std::vector<int64_t> rotShape = grouped ? std::vector<int64_t>{e, k, k} : std::vector<int64_t>{k, k};
        auto* rotation = CreateTensor(resources, rotShape, ACL_FLOAT16, rotData.data(),
                                      rotData.size() * sizeof(uint16_t), rotAddress);
        auto* y = CreateTensor(resources, {m, n}, dtype, yData.data(), yBytes, yAddress);
        auto* level0 = CreateTensor(resources, {m, level0Count}, ACL_FLOAT, level0Data.data(),
                                    level0Data.size() * sizeof(float), level0Address);
        auto* level1 = CreateTensor(resources, {m, level1Pairs, 2}, ACL_FLOAT8_E8M0, level1Data.data(),
                                    level1Data.size(), level1Address);
        aclTensor* groups = grouped ? CreateTensor(resources, {e}, ACL_INT64, groupData.data(),
                                                   groupData.size() * sizeof(int64_t), groupsAddress) :
                                      nullptr;
        char roundMode[] = "rint";
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        const auto status = aclnnRotateDualLevelQuantGetWorkspaceSize(
            x, rotation, nullptr, groups, mode == "cumsum" ? 0 : 1, -1, roundMode, 0, 0.0, false, level0BlockSize,
            level1BlockSize, y, level0, level1, &workspaceSize, &executor);
        if (status != ACL_SUCCESS) {
            std::fprintf(stderr, "aclnnRotateDualLevelQuantGetWorkspaceSize failed, status=%d.\n",
                         static_cast<int>(status));
            return 1;
        }
        if (workspaceSize != 0) {
            Check(aclrtMalloc(&resources.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "workspace allocation");
        }
        Check(aclnnRotateDualLevelQuant(resources.workspace, workspaceSize, executor, resources.stream),
              "aclnnRotateDualLevelQuant");
        Check(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream");
        if (m == 0) {
            std::printf("Empty input: execution completed with workspaceSize=%llu.\n",
                        static_cast<unsigned long long>(workspaceSize));
            return 0;
        }
        Check(aclrtMemcpy(yData.data(), yBytes, yAddress, yBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy y");
        Check(aclrtMemcpy(level0Data.data(), level0Data.size() * sizeof(float), level0Address,
                          level0Data.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
              "copy level0");
        Check(aclrtMemcpy(level1Data.data(), level1Data.size(), level1Address, level1Data.size(),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "copy level1");
        for (size_t i = 0; i < std::min<size_t>(16, yData.size()); ++i) {
            std::printf("y byte[%zu]=0x%02x\n", i, static_cast<unsigned>(yData[i]));
        }
        for (size_t i = 0; i < level0Data.size(); ++i) {
            std::printf("level0[%zu]=%g\n", i, level0Data[i]);
        }
        for (size_t i = 0; i < std::min<size_t>(18, level1Data.size()); ++i) {
            const uint8_t code = level1Data[i];
            std::printf("level1[%zu]: code=%u, value=%g\n", i, static_cast<unsigned>(code),
                        code == 255 ? std::nan("") : std::ldexp(1.0, static_cast<int>(code) - 127));
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
