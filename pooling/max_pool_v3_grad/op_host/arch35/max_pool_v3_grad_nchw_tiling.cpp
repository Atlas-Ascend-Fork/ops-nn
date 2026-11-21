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
 * \file max_pool_v3_grad_nchw_tiling.cpp
 * \brief NCHW small-kernel tiling template for MaxPoolV3Grad.
 */

#include <algorithm>
#include <limits>
#include <string>

#include "error_util.h"
#include "op_host/tiling_templates_registry.h"
#include "max_pool_v3_grad_nchw_tiling.h"

namespace optiling {

namespace {
bool IsIdentityTiling(const MaxPoolGradWithArgmaxInputInfoCommon& input, int64_t isPad)
{
    return input.hKernel == 1 && input.wKernel == 1 && input.hStride == 1 && input.wStride == 1 && isPad == 0;
}
constexpr int64_t BIG_MERGE_BUF_ALIGN = 32;
// inputBytes <= 2 时走 uint16 gather 索引路径
constexpr int64_t UINT16_GATHER_DTYPE_MAX_BYTES = 2;
constexpr int64_t UINT16_GATHER_MAX_ELEMS = 65536;
constexpr int64_t MAX_UINT32_BYTES = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
} // namespace

void MaxPoolV3GradNCHWTilingHelper::DoBufferCalculate()
{
    // 统一tiling模板: 按ksize元素数分流大小核两套缓冲公式, 与kernel入口的运行时分发阈值一致。
    // 大核(ksize元素数达阈值): 前向按窗口重算argmax, 缓冲按argmax tile组织;
    // 小核: 整窗装载布局。
    if (inputData->hKernel * inputData->wKernel >= KSIZE_THRESHOLD) {
        const int64_t hArgmaxBigInner = std::min<int64_t>(
            inputData->hGrad, Ops::Base::CeilDiv(splitData.hOutputInner + inputData->hKernel - 1, inputData->hStride));
        const int64_t wArgmaxBigInner = std::min<int64_t>(
            inputData->wGrad, Ops::Base::CeilDiv(splitData.wOutputInner + inputData->wKernel - 1, inputData->wStride));

        const int64_t wArgmaxBigAligned = Ops::Base::CeilAlign(wArgmaxBigInner, baseData.maxDataNumInOneBlock);
        const int64_t wOutputBigAligned = Ops::Base::CeilAlign(splitData.wOutputInner, baseData.maxDataNumInOneBlock);

        const int64_t gradPlaneSizeHW = hArgmaxBigInner * wArgmaxBigAligned;
        const int64_t argmaxPlaneSizeHW = hArgmaxBigInner * wArgmaxBigInner;
        const int64_t outputPlaneSizeHW = splitData.hOutputInner * wOutputBigAligned;

        splitData.gradBufferSize = splitData.highAxisInner * gradPlaneSizeHW * baseData.inputBytes;
        splitData.argmaxBufferSize = splitData.highAxisInner * argmaxPlaneSizeHW * baseData.indexBytes;
        // kernel侧整型按原dtype位宽累加(等价int64累加+截断), float按float累加
        const bool isFpCompute = inputData->inputDtype == ge::DT_FLOAT || inputData->inputDtype == ge::DT_FLOAT16 ||
                                 inputData->inputDtype == ge::DT_BF16;
        const int64_t outputBytes = isFpCompute ? FLOAT32_SIZE : baseData.inputBytes;
        splitData.outputBufferSize = splitData.highAxisInner * outputPlaneSizeHW * outputBytes;

        const int64_t fullKernelBytes = inputData->hKernel * inputData->wKernel * baseData.inputBytes;
        const int64_t forwardInputAvailableBytes = std::max<int64_t>(
            0, baseData.availableUb - splitData.argmaxBufferSize - MaxPoolGradNCHWBackwardBaseNameSpace::HELP_BUFFER -
                   BIG_MERGE_BUF_ALIGN);
        // 前向 stage 按 inputBufferSize 双缓冲 + merge buffer 计; 单窗整载只有在
        // 双缓冲后仍能放进可用 UB 时才可用, 否则必须分块加载, 避免直接回退 SIMT。
        const int64_t stageAvailBytes = std::max<int64_t>(
            0, (forwardInputAvailableBytes - BIG_MERGE_BUF_ALIGN) / DOUBLE_BUFFER);
        const int64_t maxLoadCount = std::max<int64_t>(1, stageAvailBytes / baseData.inputBytes);

        if (fullKernelBytes <= stageAvailBytes) {
            splitData.inputBufferSize = fullKernelBytes;
        } else {
            splitData.inputBufferSize = maxLoadCount * baseData.inputBytes;
        }

        const int64_t forwardStageBufferSize = splitData.inputBufferSize * DOUBLE_BUFFER + BIG_MERGE_BUF_ALIGN;
        const int64_t backwardStageBufferSize = splitData.gradBufferSize * DOUBLE_BUFFER +
                                                splitData.outputBufferSize * DOUBLE_BUFFER;

        // 大核 Init 始终额外申请 help buffer，tiling 的 UB 账必须计入，否则可能超出实际 UB。
        splitData.totalBufferSize = splitData.argmaxBufferSize + MaxPoolGradNCHWBackwardBaseNameSpace::HELP_BUFFER +
                                    std::max<int64_t>(forwardStageBufferSize, backwardStageBufferSize);
        // Narrow integer scatter packs tile offsets into uint16_t. UB capacity
        // alone permits byte tiles whose offsets would wrap during that pack.
        if (!isFpCompute && baseData.inputBytes <= UINT16_GATHER_DTYPE_MAX_BYTES &&
            std::max(splitData.gradBufferSize, splitData.outputBufferSize) / baseData.inputBytes >
                UINT16_GATHER_MAX_ELEMS) {
            splitData.totalBufferSize = baseData.availableUb + 1;
        }
        return;
    }

    // splitData.hOutputInner/wOutputInner 是 out_grad（= orig_input）的 tile 尺寸；
    // hArgmaxInner/wArgmaxInner 是这个输入 tile 会影响到的池化输出（argmax）行/列数。
    // 因此 grad/argmax buffer 按 ceil((inputTile + kernelSpan - 1) / stride) 估算，
    // 而 input buffer 按池化输出行数反推的真实输入跨度估算。
    int64_t hArgmaxInner = Ops::Base::CeilDiv(splitData.hOutputInner + inputData->hKernel - 1, inputData->hStride);
    int64_t wArgmaxInner = Ops::Base::CeilDiv(splitData.wOutputInner + inputData->wKernel - 1, inputData->wStride);
    int64_t wArgmaxInnerAligned = Ops::Base::CeilAlign(wArgmaxInner, baseData.maxDataNumInOneBlock);
    int64_t wOutputInnerAligned = Ops::Base::CeilAlign(splitData.wOutputInner, baseData.maxDataNumInOneBlock);
    // input tile 的真实行/列跨度：kernel 侧 ForwardCopyIn 按 (hArgmaxActual-1)*stride+kernelSpan 搬入，
    // 用 hArgmaxInner 反推可避免 stride>1 时按 2*(kernel-1)*dilation 多预留 halo。
    int64_t inputInnerRows = (hArgmaxInner - KERNEL_OFFSET) * inputData->hStride +
                             (inputData->hKernel - KERNEL_OFFSET) * inputData->hDilation + 1;
    int64_t inputInnerCols = (wArgmaxInner - KERNEL_OFFSET) * inputData->wStride +
                             (inputData->wKernel - KERNEL_OFFSET) * inputData->wDilation + 1;
    int64_t wInputAligned = Ops::Base::CeilAlign(inputInnerCols, baseData.maxDataNumInOneBlock);

    int64_t argmaxPlaneSizeHW = hArgmaxInner * wArgmaxInnerAligned;
    int64_t outputPlaneSizeHW = splitData.hOutputInner * wOutputInnerAligned;

    // 注意：inputBufferSize 必须按输入行/列窗口计算，不能用 hArgmaxInner（已除 stride 的
    // 池化输出行数），否则 hStride/wStride>1 时会低估 kernel 的 calcbuffer 写入量。
    splitData.inputBufferSize = splitData.highAxisInner * inputInnerRows * wInputAligned * baseData.inputBytes;
    splitData.gradBufferSize = splitData.highAxisInner * argmaxPlaneSizeHW * baseData.inputBytes;
    splitData.argmaxBufferSize = splitData.highAxisInner * argmaxPlaneSizeHW *
                                 (inputData->isInt32Meet ? INT64_SIZE : INT32_SIZE);
    const bool isFpCompute = inputData->inputDtype == ge::DT_FLOAT || inputData->inputDtype == ge::DT_FLOAT16 ||
                             inputData->inputDtype == ge::DT_BF16;
    const int64_t outputBytes = isFpCompute ? FLOAT32_SIZE : baseData.inputBytes;
    splitData.outputBufferSize = splitData.highAxisInner * outputPlaneSizeHW * outputBytes;

    if (IsIdentityTiling(*inputData, baseData.isPad)) {
        // identity 路径只做 T1 搬运：grad 经 VECIN 队列后由 Vector 拷到 output 队列。
        splitData.outputBufferSize = splitData.highAxisInner * outputPlaneSizeHW * baseData.inputBytes;
        splitData.totalBufferSize = splitData.gradBufferSize + splitData.outputBufferSize * DOUBLE_BUFFER;
        return;
    }

    // kernel 侧 input/grad 队列为单缓冲，output 队列为双缓冲（与 Init 的 IO_BUFFER_NUM 对应）。
    int64_t tmpTotalBufferSize = splitData.inputBufferSize + splitData.gradBufferSize +
                                 splitData.outputBufferSize * DOUBLE_BUFFER;
    // 浮点路径的 base BackwardCompute 需要 help buffer，kernel 侧按
    // MaxPoolGradNCHWBackwardBaseNameSpace::HELP_BUFFER 分配，必须计入 UB 账。
    if (isFpCompute) {
        tmpTotalBufferSize += MaxPoolGradNCHWBackwardBaseNameSpace::HELP_BUFFER;
    }
    splitData.totalBufferSize = tmpTotalBufferSize + splitData.argmaxBufferSize;
    if (baseData.isPad == 1) {
        splitData.totalBufferSize += splitData.inputBufferSize;
    }

    // b8/b16 gather 使用 uint16 索引；限制单个 tile 的 calcbuffer 元素数，避免 Pack 截断。
    constexpr int64_t MAX_U16_GATHER_ELEMS = 65536;
    const int64_t gatherBufferSize = isFpCompute ? splitData.inputBufferSize :
                                                   std::max({splitData.inputBufferSize, splitData.gradBufferSize,
                                                             splitData.outputBufferSize});
    if (baseData.inputBytes <= 2 && gatherBufferSize / baseData.inputBytes > MAX_U16_GATHER_ELEMS) {
        splitData.totalBufferSize = baseData.availableUb + 1;
    }
}

bool MaxPoolV3GradNCHWTiling::IsCapable()
{
    if (inputData.inputFormat != ge::Format::FORMAT_NCHW || inputData.hDilation != 1 || inputData.wDilation != 1 ||
        inputData.isInt32Meet != 0) {
        return false;
    }
    // 统一模板承接全部ksize: 大小核实现由kernel入口按tiling key运行时分发
    if (inputData.nX <= 0 || inputData.cX <= 0 || inputData.hX <= 0 || inputData.wX <= 0 || inputData.hGrad <= 0 ||
        inputData.wGrad <= 0) {
        return false;
    }

    // kernel 的 loop1SrcStride 为输入平面 hX*wX*sizeof(T)，需在 32 位范围内可表达；超限回退 SIMT。
    const int64_t inputBytes = ge::GetSizeByDataType(inputData.inputDtype);
    if (inputBytes <= 0 || inputData.hX > MAX_UINT32_BYTES / inputBytes / inputData.wX) {
        return false;
    }

    hwInfo.coreNum = coreNum_;
    hwInfo.ubSize = ubSize_;
    commonTiling.InitializationVars(context_, &hwInfo);
    return commonTiling.CheckUBSize();
}

void MaxPoolV3GradNCHWTilingHelper::SetTilingData(gert::TilingContext* context, uint64_t key)
{
    // H/W 只有一个 tile 时 argmax 不会落到 tile 外，可关闭 kernel 侧越界过滤。
    if (splitData.isCheckRange == 1 && splitData.hOutputOuter == 1 && splitData.wOutputOuter == 1) {
        splitData.isCheckRange = 0;
    }
    MaxPoolGradNCHWTilingCommon::SetTilingData(context, key);
}

ge::graphStatus MaxPoolV3GradNCHWTiling::DoOpTiling()
{
    hwInfo.coreNum = coreNum_;
    hwInfo.ubSize = ubSize_;
    commonTiling.InitializationVars(context_, &hwInfo);
    return commonTiling.DoOpTiling(context_, GetTilingKey());
}

uint64_t MaxPoolV3GradNCHWTiling::GetTilingKey() const
{
    const uint64_t kernelMode = (inputData.hKernel * inputData.wKernel >= KSIZE_THRESHOLD) ? TPL_NCHW_BIG_KERNEL :
                                                                                             TPL_NCHW_SMALL_KERNEL;
    return GET_TPL_TILING_KEY(kernelMode, TPL_NCHW_FORMAT, TPL_INT32,
                              static_cast<uint64_t>(commonTiling.GetSplitData().isCheckRange));
}

ge::graphStatus MaxPoolV3GradNCHWTiling::GetWorkspaceSize()
{
    size_t* workspaceSizes = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaceSizes);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3GradNCHWTiling::PostTiling()
{
    // commonTiling 内部按传入的 key 写 tilingData；最终 key 依赖 tiling 结束后的 isCheckRange，
    // 这里用最终值回填，并显式设置 context key，避免 tilingData 与框架 key 不一致。
    const uint64_t finalKey = GetTilingKey();
    auto* tilingData = context_->GetTilingData<
        MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData>();
    if (tilingData != nullptr) {
        tilingData->tilingKey = static_cast<int64_t>(finalKey);
    }
    auto status = commonTiling.PostTiling(context_);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    const auto setKeyStatus = context_->SetTilingKey(finalKey);
    if (setKeyStatus != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "SetTilingKey failed, tilingKey=%llu, rc=%d.",
                static_cast<unsigned long long>(finalKey), static_cast<int32_t>(setKeyStatus));
        return setKeyStatus;
    }
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPoolV3Grad", MaxPoolV3GradNCHWTiling, 0);

} // namespace optiling
