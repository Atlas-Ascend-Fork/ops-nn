/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_H_
#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GRAD_H_

#include "kernel_operator.h"
#include "sigmoid_cross_entropy_with_logits_grad_tiling_data.h"

namespace SigmoidCrossEntropyWithLogitsGradOp {

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class SigmoidBceGradKernel {
public:
    __aicore__ inline SigmoidBceGradKernel() {}

    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR target, GM_ADDR dout, GM_ADDR gradient,
                                const SigmoidCrossEntropyWithLogitsGradTilingData& tilingData)
    {
        totalLength_ = tilingData.totalLength;
        coreNum_ = tilingData.coreNum;
        elemsPerCore_ = tilingData.elemsPerCore;
        elemsPerTile_ = tilingData.elemsPerTile;

        predictGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(predict), totalLength_);
        targetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(target), totalLength_);
        doutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dout), totalLength_);
        gradientGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradient), totalLength_);

        int64_t coreId = AscendC::GetBlockIdx();
        coreOffset_ = coreId * elemsPerCore_;
        coreLength_ = (coreId + 1 == coreNum_) ? (totalLength_ - (coreNum_ - 1) * elemsPerCore_) : elemsPerCore_;
        loopNum_ = (coreLength_ + elemsPerTile_ - 1) / elemsPerTile_;

        uint32_t tileSize = static_cast<uint32_t>(elemsPerTile_);
        pipe_.InitBuffer(inQueuePredict_, BUFFER_NUM, tileSize * sizeof(T));
        pipe_.InitBuffer(inQueueTarget_, BUFFER_NUM, tileSize * sizeof(T));
        pipe_.InitBuffer(inQueueDout_, BUFFER_NUM, tileSize * sizeof(T));
        pipe_.InitBuffer(outQueue_, 1, tileSize * sizeof(T));
        pipe_.InitBuffer(tmpBufA_, tileSize * sizeof(float));
        pipe_.InitBuffer(tmpBufB_, tileSize * sizeof(float));
        pipe_.InitBuffer(tmpBufC_, tileSize * sizeof(float));
        pipe_.InitBuffer(tmpBufD_, tileSize * sizeof(float));
        pipe_.InitBuffer(tmpBufE_, tileSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (loopNum_ == 0) {
            return;
        }
        CopyIn(0, GetTileLength(0));
        for (int64_t i = 0; i < loopNum_; i++) {
            if (i + 1 < loopNum_) {
                CopyIn(i + 1, GetTileLength(i + 1));
            }
            Compute(GetTileLength(i));
            CopyOut(i, GetTileLength(i));
        }
    }

private:
    __aicore__ inline int32_t GetTileLength(int64_t tileIdx)
    {
        int64_t processed = tileIdx * elemsPerTile_;
        int64_t remaining = coreLength_ - processed;
        return static_cast<int32_t>(remaining < elemsPerTile_ ? remaining : elemsPerTile_);
    }

    __aicore__ inline void CopyIn(int64_t tileIdx, int32_t len)
    {
        uint64_t offset = coreOffset_ + tileIdx * elemsPerTile_;
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::LocalTensor<T> predictLocal = inQueuePredict_.AllocTensor<T>();
        AscendC::LocalTensor<T> targetLocal = inQueueTarget_.AllocTensor<T>();
        AscendC::LocalTensor<T> doutLocal = inQueueDout_.AllocTensor<T>();
        AscendC::DataCopyPad(predictLocal, predictGm_[offset], copyParams, padParams);
        AscendC::DataCopyPad(targetLocal, targetGm_[offset], copyParams, padParams);
        AscendC::DataCopyPad(doutLocal, doutGm_[offset], copyParams, padParams);
        inQueuePredict_.EnQue(predictLocal);
        inQueueTarget_.EnQue(targetLocal);
        inQueueDout_.EnQue(doutLocal);
    }

    __aicore__ inline void Compute(int32_t len)
    {
        AscendC::LocalTensor<T> predictT = inQueuePredict_.DeQue<T>();
        AscendC::LocalTensor<T> targetT = inQueueTarget_.DeQue<T>();
        AscendC::LocalTensor<T> doutT = inQueueDout_.DeQue<T>();
        AscendC::LocalTensor<T> outT = outQueue_.AllocTensor<T>();

        if constexpr (!std::is_same_v<T, float>) {
            AscendC::LocalTensor<float> predictF = tmpBufA_.Get<float>();
            AscendC::LocalTensor<float> targetF = tmpBufB_.Get<float>();
            AscendC::LocalTensor<float> doutF = tmpBufC_.Get<float>();
            AscendC::LocalTensor<float> outF = tmpBufD_.Get<float>();
            AscendC::Cast(predictF, predictT, AscendC::RoundMode::CAST_NONE, len);
            AscendC::Cast(targetF, targetT, AscendC::RoundMode::CAST_NONE, len);
            AscendC::Cast(doutF, doutT, AscendC::RoundMode::CAST_NONE, len);
            ComputeCore<float>(predictF, targetF, doutF, outF, len);
            AscendC::Cast(outT, outF, AscendC::RoundMode::CAST_RINT, len);
        } else {
            ComputeCore<T>(predictT, targetT, doutT, outT, len);
        }

        inQueuePredict_.FreeTensor(predictT);
        inQueueTarget_.FreeTensor(targetT);
        inQueueDout_.FreeTensor(doutT);
        outQueue_.EnQue(outT);
    }

    // dx = (sigmoid(predict) - target) * dout
    template <typename CT>
    __aicore__ inline void ComputeCore(AscendC::LocalTensor<CT>& predict, AscendC::LocalTensor<CT>& target,
                                       AscendC::LocalTensor<CT>& dout, AscendC::LocalTensor<CT>& out, int32_t len)
    {
        AscendC::LocalTensor<CT> sig = tmpBufE_.Get<CT>();
        Sigmoid<CT>(predict, sig, len);
        AscendC::Sub(out, sig, target, len);
        AscendC::Mul(out, out, dout, len);
    }

    template <typename CT>
    __aicore__ inline void Sigmoid(AscendC::LocalTensor<CT>& input, AscendC::LocalTensor<CT>& output, int32_t len)
    {
        AscendC::Muls(output, input, static_cast<CT>(-1.0f), len);
        AscendC::Exp(output, output, len);
        AscendC::Adds(output, output, static_cast<CT>(1.0f), len);
        AscendC::Reciprocal(output, output, len);
    }

    __aicore__ inline void CopyOut(int64_t tileIdx, int32_t len)
    {
        uint64_t offset = coreOffset_ + tileIdx * elemsPerTile_;
        AscendC::LocalTensor<T> outLocal = outQueue_.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(gradientGm_[offset], outLocal, copyParams);
        outQueue_.FreeTensor(outLocal);
    }

    int64_t totalLength_ = 0;
    int64_t coreNum_ = 0;
    int64_t elemsPerCore_ = 0;
    int64_t elemsPerTile_ = 0;
    int64_t loopNum_ = 0;
    int64_t coreOffset_ = 0;
    int64_t coreLength_ = 0;

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> predictGm_;
    AscendC::GlobalTensor<T> targetGm_;
    AscendC::GlobalTensor<T> doutGm_;
    AscendC::GlobalTensor<T> gradientGm_;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuePredict_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueTarget_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueDout_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufA_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufB_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufC_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufD_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufE_;
};

} // namespace SigmoidCrossEntropyWithLogitsGradOp

#endif
