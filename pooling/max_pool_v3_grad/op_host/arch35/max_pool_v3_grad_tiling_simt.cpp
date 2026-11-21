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
 * \file max_pool_v3_grad_tiling_simt.cpp
 * \brief SIMT tiling template for MaxPoolV3Grad.
 */

#include <algorithm>
#include <limits>

#include "op_host/tiling_templates_registry.h"
#include "utils/extern_math_util.h"
#include "max_pool_v3_grad_tiling_simt.h"

namespace optiling {

bool MaxPoolV3GradSimtTiling::IsCapable() { return true; }

ge::graphStatus MaxPoolV3GradSimtTiling::DoOpTiling()
{
    int32_t needCoreNum = 1;
    if (totalInputElements_ > 0) {
        const int64_t desiredCoreNum = Ops::Base::CeilDiv(totalInputElements_, PER_CORE_MIN_ELEMENTS);
        needCoreNum = static_cast<int32_t>(std::min(coreNum_, std::max(desiredCoreNum, static_cast<int64_t>(1))));
    }
    needCoreNum_ = needCoreNum;

    // int32 模板的循环变量为 uint32 且按 gridStride 递增，安全上界需再前移一个 gridStride 防回绕。
    constexpr int64_t MAX_UINT32 = 4294967295LL;
    constexpr int64_t SIMT_THREAD_DIM = 256; // 必须与 kernel 侧 MaxPoolGrad::THREAD_DIM 一致
    const int64_t gridStride = static_cast<int64_t>(needCoreNum) * SIMT_THREAD_DIM;
    const int64_t maxUint32LoopCount = MAX_UINT32 + 1 - gridStride;
    needInt64Indices_ = inputHW_ > MAX_INT32 || totalInputElements_ > maxUint32LoopCount ||
                        totalOutputElements_ > maxUint32LoopCount;

    auto tilingData = context_->GetTilingData<MaxPoolGradWithArgmaxSimtTilingCommonData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData);
    *tilingData = MaxPoolGradWithArgmaxSimtTilingCommonData{};

    tilingData->nDim = inputData.nX;
    tilingData->cDim = inputData.cX;
    tilingData->hInDim = inputData.hX;
    tilingData->wInDim = inputData.wX;
    tilingData->hOutDim = inputData.hGrad;
    tilingData->wOutDim = inputData.wGrad;
    tilingData->kSizeH = inputData.hKernel;
    tilingData->kSizeW = inputData.wKernel;
    tilingData->stridesH = inputData.hStride;
    tilingData->stridesW = inputData.wStride;
    tilingData->padH = inputData.hPad;
    tilingData->padW = inputData.wPad;
    tilingData->dilationH = inputData.hDilation;
    tilingData->dilationW = inputData.wDilation;
    tilingData->ceilMode = inputData.ceilMode ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPoolV3GradSimtTiling::GetTilingKey() const
{
    const uint32_t format = inputData.inputFormat == ge::Format::FORMAT_NHWC ? TPL_NHWC_FORMAT : TPL_NCHW_FORMAT;
    const uint32_t indicesDtype = needInt64Indices_ ? TPL_INT64 : TPL_INT32;
    return GET_TPL_TILING_KEY(TPL_SIMT_KERNEL, format, indicesDtype, TPL_NO_CHECK_RANGE);
}

ge::graphStatus MaxPoolV3GradSimtTiling::GetWorkspaceSize()
{
    const size_t indicesSize = needInt64Indices_ ? static_cast<size_t>(INT64_SIZE) : static_cast<size_t>(INT32_SIZE);
    size_t* workspaceSizes = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaceSizes);

    size_t argmaxWorkspaceSize = 0;
    if (ge::MulOverflow(static_cast<size_t>(totalOutputElements_), indicesSize, argmaxWorkspaceSize) ||
        ge::AddOverflow(argmaxWorkspaceSize, WS_SYS_SIZE, argmaxWorkspaceSize)) {
        OP_LOGE(context_->GetNodeName(), "The argmax workspace size computation overflowed.");
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = argmaxWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3GradSimtTiling::PostTiling()
{
    auto status = context_->SetBlockDim(needCoreNum_);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "SetBlockDim failed, blockDim=%d, rc=%d.", static_cast<int32_t>(needCoreNum_),
                static_cast<int32_t>(status));
        return status;
    }

    // kernel 的 Process() 中有 SyncAll()，必须 batch mode 保证 block 同时启动，否则跨 block 同步死锁。
    status = context_->SetScheduleMode(BATCH_MODE);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "SetScheduleMode failed, scheduleMode=%u, rc=%d.",
                static_cast<uint32_t>(BATCH_MODE), static_cast<int32_t>(status));
        return status;
    }

    const int64_t localMemorySize = ubSize_ - static_cast<int64_t>(DCACHE_SIZE);
    if (localMemorySize > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE(context_->GetNodeName(), "The local memory size exceeds uint32_t.");
        return ge::GRAPH_FAILED;
    }
    const auto setMemoryStatus = context_->SetLocalMemorySize(static_cast<uint32_t>(localMemorySize));
    if (setMemoryStatus != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "SetLocalMemorySize failed, rc=%d, localMemorySize=%lld.",
                static_cast<int32_t>(setMemoryStatus), static_cast<long long>(localMemorySize));
        return setMemoryStatus;
    }
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolV3Grad, MaxPoolV3GradSimtTiling, 100);

} // namespace optiling
