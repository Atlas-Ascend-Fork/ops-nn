/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BATCH_NORM_GRAD_V3_REDUCE_EMPTY_H_
#define BATCH_NORM_GRAD_V3_REDUCE_EMPTY_H_

#include "batch_norm_grad_v3_common.h"

namespace BatchNormGradV3 {
template <typename WeightType>
__aicore__ inline void FillEmptyReductionGradients(GM_ADDR dweight, GM_ADDR dbias, int64_t channels,
                                                   int64_t channelsPerCore, int64_t tileChannels, int64_t outputMask,
                                                   AscendC::TPipe& pipe)
{
    using namespace AscendC;
    const bool writeWeight = (outputMask & OUTPUT_MASK_DGAMMA) != 0;
    const bool writeBias = (outputMask & OUTPUT_MASK_DBETA) != 0;
    if (!writeWeight && !writeBias) {
        return;
    }

    // Integer zero has the same representation as floating-point +0 for all supported output types.
    using ZeroType = typename std::conditional<sizeof(WeightType) == sizeof(int32_t), int32_t, int16_t>::type;
    GlobalTensor<ZeroType> dweightGm;
    GlobalTensor<ZeroType> dbiasGm;
    if (writeWeight) {
        dweightGm.SetGlobalBuffer(reinterpret_cast<__gm__ ZeroType*>(dweight), channels);
    }
    if (writeBias) {
        dbiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ ZeroType*>(dbias), channels);
    }

    TQue<QuePosition::VECOUT, 1> zeroQueue;
    pipe.InitBuffer(zeroQueue, 1, tileChannels * sizeof(ZeroType));
    auto zeros = zeroQueue.AllocTensor<ZeroType>();
    Duplicate(zeros, static_cast<ZeroType>(0), static_cast<int32_t>(tileChannels));
    zeroQueue.EnQue(zeros);
    zeros = zeroQueue.DeQue<ZeroType>();

    const int64_t begin = GetBlockIdx() * channelsPerCore;
    const int64_t remainingChannels = channels - begin;
    const int64_t coreChannels = remainingChannels < channelsPerCore ? remainingChannels : channelsPerCore;
    for (int64_t channel = 0; channel < coreChannels; channel += tileChannels) {
        const int64_t offset = begin + channel;
        const int64_t remaining = coreChannels - channel;
        const int64_t count = remaining < tileChannels ? remaining : tileChannels;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(ZeroType)), 0, 0, 0};
        if (writeWeight) {
            DataCopyPad(dweightGm[offset], zeros, copyParams);
        }
        if (writeBias) {
            DataCopyPad(dbiasGm[offset], zeros, copyParams);
        }
    }
    zeroQueue.FreeTensor(zeros);
}
} // namespace BatchNormGradV3

#endif // BATCH_NORM_GRAD_V3_REDUCE_EMPTY_H_
