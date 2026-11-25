/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_SCATTER_ELEMENTS_INPUT_COPY_H_
#define ASCENDC_SCATTER_ELEMENTS_INPUT_COPY_H_

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"

namespace ScatterElements {
using namespace AscendC;

__aicore__ inline void CopyScatterInput(TPipe& pipe, GM_ADDR data, GM_ADDR output, uint64_t dataBytes,
                                        bool scatterFollows)
{
    if (dataBytes == 0 || data == output) {
        return;
    }

    constexpr uint64_t COPY_ALIGN_BYTES = 512;
    constexpr uint32_t COPY_TILE_BYTES = 16 * 1024;
    const uint64_t blockBytes = ops::Aligned(ops::CeilDiv(dataBytes, static_cast<uint64_t>(GetBlockNum())),
                                             COPY_ALIGN_BYTES);
    const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * blockBytes;
    const uint64_t end = AscendC::Std::min(begin + blockBytes, dataBytes);
    if (begin < end) {
        GlobalTensor<uint8_t> inputGm;
        GlobalTensor<uint8_t> outputGm;
        inputGm.SetGlobalBuffer(data);
        outputGm.SetGlobalBuffer(output);
        TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> copyQueue;
        pipe.InitBuffer(copyQueue, 1, COPY_TILE_BYTES);
        const DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        for (uint64_t offset = begin; offset < end; offset += COPY_TILE_BYTES) {
            const uint32_t bytes = static_cast<uint32_t>(
                AscendC::Std::min(end - offset, static_cast<uint64_t>(COPY_TILE_BYTES)));
            const DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
            auto local = copyQueue.AllocTensor<uint8_t>();
            DataCopyPad(local, inputGm[offset], copyParams, padParams);
            copyQueue.EnQue(local);
            local = copyQueue.DeQue<uint8_t>();
            DataCopyPad(outputGm[offset], local, copyParams);
            copyQueue.FreeTensor(local);
        }
        const auto event = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(event);
        WaitFlag<HardEvent::MTE3_S>(event);
    }
    // All output tiles must be visible before any core scatters into them.
    if (scatterFollows) {
        SyncAll();
        pipe.Reset();
    }
}

} // namespace ScatterElements

#endif // ASCENDC_SCATTER_ELEMENTS_INPUT_COPY_H_
