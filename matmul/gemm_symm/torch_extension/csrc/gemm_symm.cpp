/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>

#include <string>
#include <vector>

#include "aclnn_common.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace cann_ops_nn {
namespace matmul {
namespace {

aclTensor* CreateAclTensorND(const at::Tensor& tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(CreateTensor);
    TORCH_CHECK(aclCreateTensor != nullptr, "aclCreateTensor is not available");
    const aclDataType aclType = ConvertToAclDataType(tensor.scalar_type());
    const auto sizes = tensor.sizes();
    const auto strides = tensor.strides();
    std::vector<int64_t> storageDims{static_cast<int64_t>(tensor.storage().nbytes() / tensor.itemsize())};
    return aclCreateTensor(sizes.data(), sizes.size(), aclType, strides.data(),
                           ConvertToAclStorageOffset(tensor, aclType), ACL_FORMAT_ND, storageDims.data(),
                           storageDims.size(), const_cast<void*>(tensor.storage().data()));
}

aclScalar* CreateFloatScalar(float value)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(CreateScalar);
    TORCH_CHECK(aclCreateScalar != nullptr, "aclCreateScalar is not available");
    aclScalar* scalar = aclCreateScalar(&value, ACL_FLOAT);
    TORCH_CHECK(scalar != nullptr, "aclCreateScalar failed");
    return scalar;
}

} // namespace

at::Tensor gemm_symm(const at::Tensor& a, const at::Tensor& b, const c10::optional<at::Tensor>& c, double alpha,
                     double beta, bool transposeA, bool transposeB, const std::string& symmType, int64_t cubeMathType)
{
    TORCH_CHECK(a.device().type() == at::kPrivateUse1 && b.device() == a.device(), "a and b must be on the same NPU");
    const bool supported = a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16;
    TORCH_CHECK(supported && b.scalar_type() == a.scalar_type(), "a and b must both be float16 or both be bfloat16");
    TORCH_CHECK(symmType == "full", "symm_type only supports full");
    const int64_t m = transposeA ? a.size(-1) : a.size(-2);
    at::Tensor y = a.dim() == 3 ? at::empty({a.size(0), m, m}, a.options()) : at::empty({m, m}, a.options());

    const c10::OptionalDeviceGuard deviceGuard(a.device());
    static const auto getWorkspaceSizeAddr = GetOpApiFuncAddr("aclnnGemmSymmGetWorkspaceSize");
    static const auto opApiAddr = GetOpApiFuncAddr("aclnnGemmSymm");
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");
    TORCH_CHECK(getWorkspaceSizeAddr != nullptr && opApiAddr != nullptr,
                "aclnnGemmSymm is not installed. Build and install ops-nn first.");

    auto npuStream = c10_npu::getCurrentNPUStream();
    auto aclStream = npuStream.stream(false);
    if (c10_npu::check_enqueue_need_use(aclStream)) {
        aclrtUseStreamResInCurrentThread(aclStream);
    }

    aclTensor* aclA = CreateAclTensorND(a);
    aclTensor* aclB = CreateAclTensorND(b);
    aclTensor* aclC = nullptr;
    if (c.has_value() && c->defined()) {
        TORCH_CHECK(c->device() == a.device() && c->scalar_type() == a.scalar_type(),
                    "c must match a in dtype and device");
        aclC = CreateAclTensorND(*c);
    }
    aclTensor* aclY = CreateAclTensorND(y);
    aclScalar* aclAlpha = CreateFloatScalar(static_cast<float>(alpha));
    aclScalar* aclBeta = CreateFloatScalar(static_cast<float>(beta));

    auto releaseHost = [&]() {
        Release(aclA);
        Release(aclB);
        Release(aclC);
        Release(aclY);
        Release(aclAlpha);
        Release(aclBeta);
    };
    auto unInitMem = []() {
        auto unInit = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
        if (unInit != nullptr) {
            unInit(nullptr, false);
        }
    };

    auto initMem = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    if (initMem != nullptr) {
        initMem(nullptr, false);
    }
    ApplyDeterministicConfig();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    using GetWs = int (*)(aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclScalar*, aclScalar*, bool, bool,
                          const char*, int8_t, uint64_t*, aclOpExecutor**);
    auto getWs = reinterpret_cast<GetWs>(getWorkspaceSizeAddr);
    int status = getWs(aclA, aclB, aclC, aclY, aclAlpha, aclBeta, transposeA, transposeB, symmType.c_str(),
                       static_cast<int8_t>(cubeMathType), &workspaceSize, &executor);
    if (status != 0) {
        releaseHost();
        unInitMem();
        TORCH_CHECK(false, "aclnnGemmSymmGetWorkspaceSize failed, detail:", aclGetRecentErrMsg());
    }

    void* workspaceAddr = nullptr;
    at::Tensor workspaceTensor;
    if (workspaceSize != 0) {
        workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)},
                                    at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kByte));
        workspaceAddr = const_cast<void*>(workspaceTensor.storage().data());
    }
    // Bind the launch to the current stream so a later same-shape call cannot read the previous X.
    auto aclCall = [aclA, aclB, aclC, aclY, aclAlpha, aclBeta, workspaceTensor, workspaceAddr, workspaceSize, executor,
                    aclStream]() -> int {
        (void)workspaceTensor;
        if (c10_npu::check_enqueue_need_use(aclStream)) {
            aclrtUseStreamResInCurrentThread(aclStream);
        }
        using RunFn = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
        auto run = reinterpret_cast<RunFn>(opApiAddr);
        int apiStatus = run(workspaceAddr, workspaceSize, executor, aclStream);
        Release(aclA);
        Release(aclB);
        Release(aclC);
        Release(aclY);
        Release(aclAlpha);
        Release(aclBeta);
        auto releaseMem = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
        if (releaseMem != nullptr) {
            releaseMem(nullptr, false);
        }
        TORCH_CHECK(apiStatus == 0, "aclnnGemmSymm failed, detail:", aclGetRecentErrMsg());
        return apiStatus;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnGemmSymm");
    cmd.SetCustomHandler(aclCall);
    cmd.Run();
    c10_npu::NPUCachingAllocator::recordStream(a.storage().data_ptr(), npuStream);
    c10_npu::NPUCachingAllocator::recordStream(b.storage().data_ptr(), npuStream);
    if (c.has_value() && c->defined()) {
        c10_npu::NPUCachingAllocator::recordStream(c->storage().data_ptr(), npuStream);
    }
    c10_npu::NPUCachingAllocator::recordStream(y.storage().data_ptr(), npuStream);
    if (workspaceTensor.defined()) {
        c10_npu::NPUCachingAllocator::recordStream(workspaceTensor.storage().data_ptr(), npuStream);
    }
    unInitMem();
    return y;
}

} // namespace matmul
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm_symm", &cann_ops_nn::matmul::gemm_symm, "GemmSymm on NPU"); }
