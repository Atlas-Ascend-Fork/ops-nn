/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN_TRAINING_REDUCE_GRAD_SHORT_PLANE_H
#define BN_TRAINING_REDUCE_GRAD_SHORT_PLANE_H

#include "bn_training_reduce_grad.h"

namespace BNTrainingReduceGradOps {

template <typename T>
class BNTrainingReduceGradShortPlane {
public:
    __aicore__ static inline uint32_t GetIndexCount(const BNTrainingReduceGradTilingData& tiling)
    {
        if (tiling.innerCores != 1 || tiling.innerSize < 1 || tiling.innerSize > MAX_INNER_SIZE || tiling.numC < 1) {
            return 0;
        }
        if (IsPeriodic(tiling)) {
            const uint32_t period = tiling.numC * tiling.innerSize;
            // One full sequence of 64-element vectors with the same starting phase.
            uint32_t cycle = VL_FP32;
            while (cycle % period != 0) {
                cycle += VL_FP32;
            }
            return cycle;
        }
        return tiling.innerSize >= MIN_BATCHED_INNER_SIZE ? VL_FP32 * tiling.innerSize : 0;
    }

    __aicore__ static inline uint32_t GetTileSize(const BNTrainingReduceGradTilingData& tiling, uint32_t indexCount)
    {
        if (indexCount == 0) {
            return 0;
        }
        constexpr int64_t dataBytesPerElement = 3 * BUFFER_COUNT * sizeof(T);
        const int64_t available = tiling.ubTileSize * dataBytesPerElement -
                                  static_cast<int64_t>(indexCount * sizeof(uint32_t));
        if (available < static_cast<int64_t>(indexCount) * dataBytesPerElement) {
            return 0;
        }
        if (!IsPeriodic(tiling)) {
            return indexCount;
        }
        // Every full tile returns to the initial index phase. The cycle is also VL-aligned.
        return available / dataBytesPerElement / indexCount * indexCount;
    }

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR x, GM_ADDR diffScale, GM_ADDR diffOffset, GM_ADDR scale,
                                GM_ADDR batchMean, GM_ADDR batchVar, GM_ADDR y,
                                const BNTrainingReduceGradTilingData& tiling, TPipe& pipe, uint32_t indexCount,
                                uint32_t tileSize)
    {
        tiling_ = &tiling;
        indexCount_ = indexCount;
        tileSize_ = tileSize;

        const int64_t elements = tiling.units * tiling.innerSize;
        gradsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grads), elements);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), elements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), elements);
        diffScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diffScale), tiling.numC);
        diffOffsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diffOffset), tiling.numC);
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale), tiling.numC);
        batchMeanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batchMean), tiling.numC);
        batchVarGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batchVar), tiling.numC);

        pipe.InitBuffer(gradsQue_, BUFFER_COUNT, tileSize * sizeof(T));
        pipe.InitBuffer(xQue_, BUFFER_COUNT, tileSize * sizeof(T));
        pipe.InitBuffer(yQue_, BUFFER_COUNT, tileSize * sizeof(T));
        pipe.InitBuffer(diffScaleQue_, 1, VL_FP32 * sizeof(float));
        pipe.InitBuffer(diffOffsetQue_, 1, VL_FP32 * sizeof(float));
        pipe.InitBuffer(scaleQue_, 1, VL_FP32 * sizeof(float));
        pipe.InitBuffer(batchMeanQue_, 1, VL_FP32 * sizeof(float));
        pipe.InitBuffer(batchVarQue_, 1, VL_FP32 * sizeof(float));
        pipe.InitBuffer(multiplierBuf_, VL_FP32 * sizeof(float));
        pipe.InitBuffer(addendBuf_, VL_FP32 * sizeof(float));
        pipe.InitBuffer(mulScaleBuf_, VL_FP32 * sizeof(float));
        pipe.InitBuffer(meanBuf_, VL_FP32 * sizeof(float));
        pipe.InitBuffer(indexBuf_, indexCount * sizeof(uint32_t));
    }

    __aicore__ inline void Process()
    {
        if (IsPeriodic(*tiling_)) {
            ProcessPeriodic();
        } else {
            ProcessBatched();
        }
    }

private:
    static constexpr int64_t MAX_INNER_SIZE = 8;
    static constexpr int64_t MIN_BATCHED_INNER_SIZE = 3;
    static constexpr int64_t MAX_PERIOD = 32;
    static constexpr uint32_t BUFFER_COUNT = 2;

    __aicore__ static inline bool IsPeriodic(const BNTrainingReduceGradTilingData& tiling)
    {
        return tiling.numC <= MAX_PERIOD / tiling.innerSize;
    }

    __aicore__ inline void ProcessPeriodic()
    {
        const int64_t elements = tiling_->units * tiling_->innerSize;
        const int64_t vectorCount = elements / VL_FP32 + (elements % VL_FP32 != 0);
        const int64_t block = GetBlockIdx();
        const int64_t vectorsPerCore = vectorCount / tiling_->unitCores;
        const int64_t extra = vectorCount % tiling_->unitCores;
        const int64_t start = (block * vectorsPerCore + (block < extra ? block : extra)) * VL_FP32;
        const int64_t capacity = (vectorsPerCore + (block < extra)) * VL_FP32;
        const int64_t count = capacity < elements - start ? capacity : elements - start;
        if (count <= 0) {
            return;
        }

        const uint32_t period = tiling_->numC * tiling_->innerSize;
        BuildIndices(start % period, tiling_->numC);
        StageStats(0, tiling_->numC);
        for (int64_t offset = 0; offset < count; offset += tileSize_) {
            const uint32_t extent = count - offset < tileSize_ ? count - offset : tileSize_;
            ProcessTile(start + offset, extent);
        }
    }

    __aicore__ inline void ProcessBatched()
    {
        const int64_t block = GetBlockIdx();
        int64_t plane = block < tiling_->formerCoreNum ? block * tiling_->formerUnits :
                                                         tiling_->formerCoreNum * tiling_->formerUnits +
                                                             (block - tiling_->formerCoreNum) * tiling_->latterUnits;
        const int64_t planeCount = block < tiling_->formerCoreNum ? tiling_->formerUnits : tiling_->latterUnits;
        const int64_t end = plane + planeCount;
        const int64_t channels = tiling_->numC;
        const int64_t inner = tiling_->innerSize;
        BuildIndices(0, VL_FP32);
        int64_t channel = plane % channels;
        while (plane < end) {
            const int64_t remaining = end - plane < channels - channel ? end - plane : channels - channel;
            const uint32_t count = remaining < VL_FP32 ? remaining : VL_FP32;
            StageStats(channel, count);
            ProcessTile(plane * inner, count * inner);
            plane += count;
            channel += count;
            if (channel == channels) {
                channel = 0;
            }
        }
    }

    __aicore__ inline void BuildIndices(uint32_t startPhase, uint32_t channels)
    {
        const uint32_t inner = tiling_->innerSize;
        uint32_t channel = startPhase / inner;
        uint32_t left = inner - startPhase % inner;
        auto index = indexBuf_.Get<uint32_t>();
        for (uint32_t lane = 0; lane < indexCount_; ++lane) {
            index.SetValue(lane, channel);
            if (--left == 0) {
                left = inner;
                ++channel;
                if (channel == channels) {
                    channel = 0;
                }
            }
        }
        const auto event = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event);
        WaitFlag<HardEvent::S_V>(event);
    }

    __aicore__ inline void CopyStat(TQue<QuePosition::VECIN, 1>& queue, GlobalTensor<float>& gm, int64_t channel,
                                    const DataCopyExtParams& copy)
    {
        auto ub = queue.AllocTensor<float>();
        DataCopyPad(ub, gm[channel], copy, DataCopyPadExtParams<float>{false, 0, 0, 0});
        queue.EnQue(ub);
    }

    __aicore__ inline void StageStats(int64_t channel, uint32_t count)
    {
        const DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        CopyStat(diffScaleQue_, diffScaleGm_, channel, copy);
        CopyStat(diffOffsetQue_, diffOffsetGm_, channel, copy);
        CopyStat(scaleQue_, scaleGm_, channel, copy);
        CopyStat(batchMeanQue_, batchMeanGm_, channel, copy);
        CopyStat(batchVarQue_, batchVarGm_, channel, copy);

        auto diffScale = diffScaleQue_.DeQue<float>();
        auto diffOffset = diffOffsetQue_.DeQue<float>();
        auto scale = scaleQue_.DeQue<float>();
        auto mean = batchMeanQue_.DeQue<float>();
        auto variance = batchVarQue_.DeQue<float>();
        PrecomputeCoefficients(reinterpret_cast<__ubuf__ float*>(diffScale.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ float*>(diffOffset.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ float*>(scale.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ float*>(mean.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ float*>(variance.GetPhyAddr()), count);
        diffScaleQue_.FreeTensor(diffScale);
        diffOffsetQue_.FreeTensor(diffOffset);
        scaleQue_.FreeTensor(scale);
        batchMeanQue_.FreeTensor(mean);
        batchVarQue_.FreeTensor(variance);
    }

    __aicore__ inline void PrecomputeCoefficients(__ubuf__ float* diffScale, __ubuf__ float* diffOffset,
                                                  __ubuf__ float* scale, __ubuf__ float* mean, __ubuf__ float* variance,
                                                  uint32_t count)
    {
        auto multiplier = reinterpret_cast<__ubuf__ float*>(multiplierBuf_.Get<float>().GetPhyAddr());
        auto addend = reinterpret_cast<__ubuf__ float*>(addendBuf_.Get<float>().GetPhyAddr());
        auto mulScale = reinterpret_cast<__ubuf__ float*>(mulScaleBuf_.Get<float>().GetPhyAddr());
        auto meanOutput = reinterpret_cast<__ubuf__ float*>(meanBuf_.Get<float>().GetPhyAddr());
        const float epsilon = tiling_->epsilon;
        const float negNumRecip = tiling_->negNumRecip;
        __VEC_SCOPE__
        {
            RegTensor<float> diffScaleReg, diffOffsetReg, scaleReg, meanReg, varReg;
            RegTensor<float> multReg, addReg, mulScaleReg, tmpReg;
            MaskReg mask = UpdateMask<float>(count);
            DataCopy<float, LoadDist::DIST_NORM>(diffScaleReg, diffScale);
            DataCopy<float, LoadDist::DIST_NORM>(diffOffsetReg, diffOffset);
            DataCopy<float, LoadDist::DIST_NORM>(scaleReg, scale);
            DataCopy<float, LoadDist::DIST_NORM>(meanReg, mean);
            DataCopy<float, LoadDist::DIST_NORM>(varReg, variance);
            Adds(varReg, varReg, epsilon, mask);
            Sqrt(varReg, varReg, mask);
            Muls(tmpReg, diffScaleReg, negNumRecip, mask);
            Div(multReg, tmpReg, varReg, mask);
            Muls(addReg, diffOffsetReg, negNumRecip, mask);
            Div(mulScaleReg, scaleReg, varReg, mask);
            DataCopy<float, StoreDist::DIST_NORM>(multiplier, multReg, mask);
            DataCopy<float, StoreDist::DIST_NORM>(addend, addReg, mask);
            DataCopy<float, StoreDist::DIST_NORM>(mulScale, mulScaleReg, mask);
            DataCopy<float, StoreDist::DIST_NORM>(meanOutput, meanReg, mask);
        }
    }

    __aicore__ inline void ProcessTile(int64_t start, uint32_t extent)
    {
        const DataCopyExtParams copy{1, extent * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        auto gradsUb = gradsQue_.AllocTensor<T>();
        auto xUb = xQue_.AllocTensor<T>();
        DataCopyPad(gradsUb, gradsGm_[start], copy, pad);
        DataCopyPad(xUb, xGm_[start], copy, pad);
        gradsQue_.EnQue(gradsUb);
        xQue_.EnQue(xUb);

        gradsUb = gradsQue_.DeQue<T>();
        xUb = xQue_.DeQue<T>();
        auto yUb = yQue_.AllocTensor<T>();
        ComputeTile(reinterpret_cast<__ubuf__ T*>(gradsUb.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ T*>(xUb.GetPhyAddr()), reinterpret_cast<__ubuf__ T*>(yUb.GetPhyAddr()),
                    extent);
        gradsQue_.FreeTensor(gradsUb);
        xQue_.FreeTensor(xUb);

        yQue_.EnQue(yUb);
        yUb = yQue_.DeQue<T>();
        DataCopyPad(yGm_[start], yUb, copy);
        yQue_.FreeTensor(yUb);
    }

    __aicore__ inline void ComputeTile(__ubuf__ T* grads, __ubuf__ T* x, __ubuf__ T* y, uint32_t extent)
    {
        auto multiplier = reinterpret_cast<__ubuf__ float*>(multiplierBuf_.Get<float>().GetPhyAddr());
        auto addend = reinterpret_cast<__ubuf__ float*>(addendBuf_.Get<float>().GetPhyAddr());
        auto mulScale = reinterpret_cast<__ubuf__ float*>(mulScaleBuf_.Get<float>().GetPhyAddr());
        auto mean = reinterpret_cast<__ubuf__ float*>(meanBuf_.Get<float>().GetPhyAddr());
        auto indices = reinterpret_cast<__ubuf__ uint32_t*>(indexBuf_.Get<uint32_t>().GetPhyAddr());
        const uint32_t indexCount = indexCount_;
        const uint16_t loops = (extent + VL_FP32 - 1) / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> multReg, addReg, scaleReg, meanReg;
            DataCopy<float, LoadDist::DIST_NORM>(multReg, multiplier);
            DataCopy<float, LoadDist::DIST_NORM>(addReg, addend);
            DataCopy<float, LoadDist::DIST_NORM>(scaleReg, mulScale);
            DataCopy<float, LoadDist::DIST_NORM>(meanReg, mean);
            uint32_t indexOffset = 0;
            for (uint16_t loop = 0; loop < loops; ++loop) {
                RegTensor<uint32_t> indexReg;
                RegTensor<float> vecMult, vecAdd, vecScale, vecMean, gradReg, xReg, yReg;
                MaskReg mask = UpdateMask<float>(extent);
                DataCopy<uint32_t, LoadDist::DIST_NORM>(indexReg, indices + indexOffset);
                AscendC::Reg::Gather(vecMult, multReg, indexReg);
                AscendC::Reg::Gather(vecAdd, addReg, indexReg);
                AscendC::Reg::Gather(vecScale, scaleReg, indexReg);
                AscendC::Reg::Gather(vecMean, meanReg, indexReg);
                const uint32_t offset = loop * VL_FP32;
                LoadToFp32(grads, gradReg, mask, offset);
                LoadToFp32(x, xReg, mask, offset);
                Sub(xReg, xReg, vecMean, mask);
                Mul(yReg, xReg, vecMult, mask);
                Add(yReg, gradReg, yReg, mask);
                Add(yReg, yReg, vecAdd, mask);
                Mul(yReg, yReg, vecScale, mask);
                StoreYFromFp32(y, yReg, mask, offset);
                indexOffset += VL_FP32;
                if (indexOffset == indexCount) {
                    indexOffset = 0;
                }
            }
        }
    }

    const BNTrainingReduceGradTilingData* tiling_;
    uint32_t indexCount_;
    uint32_t tileSize_;
    GlobalTensor<T> gradsGm_, xGm_, yGm_;
    GlobalTensor<float> diffScaleGm_, diffOffsetGm_, scaleGm_, batchMeanGm_, batchVarGm_;
    TQue<QuePosition::VECIN, BUFFER_COUNT> gradsQue_, xQue_;
    TQue<QuePosition::VECOUT, BUFFER_COUNT> yQue_;
    TQue<QuePosition::VECIN, 1> diffScaleQue_, diffOffsetQue_, scaleQue_, batchMeanQue_, batchVarQue_;
    TBuf<> multiplierBuf_, addendBuf_, mulScaleBuf_, meanBuf_, indexBuf_;
};

} // namespace BNTrainingReduceGradOps

#endif // BN_TRAINING_REDUCE_GRAD_SHORT_PLANE_H
