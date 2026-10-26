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
 * \file fake_quant_with_min_max_vars_per_channel_gradient_empty.h
 * \brief FakeQuantWithMinMaxVarsPerChannelGradient 空 tensor 专用模板（MODE=FQPCG_TPL_MODE_EMPTY）
 *
 * 语义（与 TF 参考实现一致）：
 *   - 输入 numel == 0（x 形状含 0 轴）时：
 *     * backprops_wrt_x 为逐元素输出，恒为空张量，无需写出；
 *     * backprop_wrt_min / backprop_wrt_max 为对 0 行求和的 (channelNum,) 向量，恒为 0 ——
 *       本模板将这两段 GM 显式写 0（channelNum == 0 时二者亦为空张量，直接返回）。
 *   - 单核执行（host 置 blockDim=1），无需 workspace 与跨核同步。
 */
#ifndef FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_EMPTY_H_
#define FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_EMPTY_H_

#include "kernel_operator.h"
#include "fake_quant_with_min_max_vars_per_channel_gradient_tilingdata.h"

namespace FakeQuantWMMVPCG {
using namespace AscendC;

constexpr uint32_t EMPTY_ZERO_BUF_ELEMS = 8192; // 32KB @ fp32

template <typename T>
class EmptyKernel {
public:
    __aicore__ inline EmptyKernel() {}

    __aicore__ inline void Init(GM_ADDR bpMin, GM_ADDR bpMax,
                                const FakeQuantWithMinMaxVarsPerChannelGradientTilingData* tilingData);

    __aicore__ inline void Process();

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> zeroBuf_;
    GlobalTensor<T> bpMinGm_;
    GlobalTensor<T> bpMaxGm_;
    uint32_t channelNum_ = 0;
};

template <typename T>
__aicore__ inline void EmptyKernel<T>::Init(GM_ADDR bpMin, GM_ADDR bpMax,
                                            const FakeQuantWithMinMaxVarsPerChannelGradientTilingData* tilingData)
{
    channelNum_ = tilingData->channelNum;
    bpMinGm_.SetGlobalBuffer((__gm__ T*)bpMin);
    bpMaxGm_.SetGlobalBuffer((__gm__ T*)bpMax);
}

template <typename T>
__aicore__ inline void EmptyKernel<T>::Process()
{
    // channelNum == 0：三个输出均为空张量，无任何写出
    if (channelNum_ == 0) {
        return;
    }
    // 空 tensor 模板固定单核；双保险避免多核重复写
    if (GetBlockIdx() != 0) {
        return;
    }

    pipe_.InitBuffer(zeroBuf_, EMPTY_ZERO_BUF_ELEMS * sizeof(T));
    LocalTensor<T> zeroLocal = zeroBuf_.template Get<T>();
    AscendC::Duplicate(zeroLocal, static_cast<T>(0), EMPTY_ZERO_BUF_ELEMS);
    PipeBarrier<PIPE_ALL>();

    uint32_t offset = 0;
    while (offset < channelNum_) {
        uint32_t count = channelNum_ - offset;
        if (count > EMPTY_ZERO_BUF_ELEMS) {
            count = EMPTY_ZERO_BUF_ELEMS;
        }
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(bpMinGm_[offset], zeroLocal, cp);
        DataCopyPad(bpMaxGm_[offset], zeroLocal, cp);
        offset += count;
    }
    PipeBarrier<PIPE_ALL>();
}

} // namespace FakeQuantWMMVPCG

#endif // FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_EMPTY_H_
