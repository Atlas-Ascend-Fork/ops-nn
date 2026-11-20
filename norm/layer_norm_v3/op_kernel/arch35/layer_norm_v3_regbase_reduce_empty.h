/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LAYER_NORM_V3_REGBASE_REDUCE_EMPTY_H
#define LAYER_NORM_V3_REGBASE_REDUCE_EMPTY_H

#include "layer_norm_v3_common.h"

namespace LayerNormV3 {
using namespace AscendC;

template <typename M>
class LayerNormV3RegbaseReduceEmpty {
public:
    __aicore__ inline LayerNormV3RegbaseReduceEmpty(const LayerNormV3TilingDataRegBaseReduceEmpty* tiling)
    {
        td_ = tiling;
    }

    __aicore__ inline void Init(GM_ADDR mean, GM_ADDR lastout)
    {
        const uint64_t begin = GetBlockIdx() * td_->rowsPerCore;
        rows_ = td_->rows - begin < td_->rowsPerCore ? td_->rows - begin : td_->rowsPerCore;
        rowsPerTile_ = td_->rowsPerTile;
        meanGm_.SetGlobalBuffer((__gm__ M*)mean + begin, rows_);
        lastoutGm_.SetGlobalBuffer((__gm__ M*)lastout + begin, rows_);
        pipe_.InitBuffer(outQueue_, 1, NUM_TWO * rowsPerTile_ * sizeof(M));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<M> outLocalTensor = outQueue_.template AllocTensor<M>();
        if constexpr (IsSameType<M, float>::value) {
            Duplicate(outLocalTensor, 0.0f, rowsPerTile_);
            Duplicate(outLocalTensor[rowsPerTile_], NumericLimits<float>::QuietNaN(), rowsPerTile_);
        } else {
            const uint16_t quietNan = GetScalarBitcodeValue<M, uint16_t>(NumericLimits<M>::QuietNaN());
            LocalTensor<uint16_t> outputBits = outLocalTensor.template ReinterpretCast<uint16_t>();
            Duplicate(outputBits, uint16_t(0), rowsPerTile_);
            Duplicate(outputBits[rowsPerTile_], quietNan, rowsPerTile_);
        }
        outQueue_.EnQue(outLocalTensor);
        outLocalTensor = outQueue_.template DeQue<M>();
        for (uint64_t offset = 0; offset < rows_; offset += rowsPerTile_) {
            const uint32_t count = static_cast<uint32_t>(rows_ - offset < rowsPerTile_ ? rows_ - offset : rowsPerTile_);
            DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(count * sizeof(M)), 0, 0, 0};
            DataCopyPad(meanGm_[offset], outLocalTensor, dataCopyParams);
            DataCopyPad(lastoutGm_[offset], outLocalTensor[rowsPerTile_], dataCopyParams);
        }
        outQueue_.FreeTensor(outLocalTensor);
    }

private:
    TPipe pipe_;
    const LayerNormV3TilingDataRegBaseReduceEmpty* td_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    GlobalTensor<M> meanGm_, lastoutGm_;
    uint64_t rows_ = 0;
    uint32_t rowsPerTile_ = 0;
};
} // namespace LayerNormV3
#endif // LAYER_NORM_V3_REGBASE_REDUCE_EMPTY_H
