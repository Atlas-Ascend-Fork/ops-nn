/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEEP_NORM_SPLIT_ROW_ARCH35_H
#define DEEP_NORM_SPLIT_ROW_ARCH35_H

#include "deep_norm_partial_load.h"

namespace NsDeepNorm {

constexpr SyncAllConfig DEEP_NORM_SPLIT_SYNC = {PIPE_MTE3, PIPE_MTE2};

// Split only at power-of-two groups of the original partial-load tiles. A full
// group exports its tree root, while the final short group exports its occupied
// cache levels. Combining these nodes reconstructs the original FP32 add tree.
template <typename T>
class DeepNormSplitRow : protected DeepNormPartialLoad<T> {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gx, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd,
                                GM_ADDR y, GM_ADDR userWorkspace, const DeepNormTilingData* tilingData)
    {
        rows_ = tilingData->numRow;
        rowAlign_ = (rows_ + DEEP_NORM_BLK_B32 - 1) / DEEP_NORM_BLK_B32 * DEEP_NORM_BLK_B32;
        scalarCapacity_ = (rows_ + DEEP_NORM_VL_FP32 - 1) / DEEP_NORM_VL_FP32 * DEEP_NORM_VL_FP32;
        tilesPerCore_ = tilingData->splitTilesPerCore;
        tileCount_ = tilingData->numCol / tilingData->tileLength +
                     static_cast<uint32_t>(tilingData->numCol % tilingData->tileLength != 0);
        fullGroups_ = tileCount_ / tilesPerCore_;
        tailTiles_ = tileCount_ % tilesPerCore_;
        uint32_t groupSize = tilesPerCore_;
        while (groupSize > 1) {
            ++groupLevel_;
            groupSize >>= 1;
        }
        firstTile_ = static_cast<uint32_t>(GetBlockIdx()) * tilesPerCore_;
        localTiles_ = DeepNormMin(tilesPerCore_, tileCount_ - firstTile_);
        isTailGroup_ = localTiles_ != tilesPerCore_;
        phaseElements_ = (fullGroups_ + (tailTiles_ == 0 ? 0 : groupLevel_)) * rowAlign_;
        partElements_ = (isTailGroup_ ? groupLevel_ : 1) * rowAlign_;
        uint32_t partCapacity = (groupLevel_ == 0 ? 1 : groupLevel_) * rowAlign_;
        uint32_t combCapacity = (phaseElements_ + DEEP_NORM_VL_FP32 - 1) / DEEP_NORM_VL_FP32 * DEEP_NORM_VL_FP32;

        // ComputeRstdNewtonRaphson loads a whole register before masking stores.
        this->InitShared(x, gx, beta, gamma, mean, rstd, y, tilingData, scalarCapacity_);
        this->pipe_.InitBuffer(partQueue_, 1, partCapacity * sizeof(float));
        this->pipe_.InitBuffer(combQueue_, 1, combCapacity * sizeof(float));
        workspaceGm_.SetGlobalBuffer((__gm__ float*)userWorkspace);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> mean = this->meanQue_.template AllocTensor<float>();
        ReducePhase<false>(0, mean);
        MergePhase(0, mean);
        for (uint32_t row = 0; row < rows_; ++row) {
            LocalTensor<float> rowMean = mean[row];
            this->ScaleScalar(rowMean, this->avgFactor_);
        }
        this->meanQue_.EnQue(mean);
        mean = this->meanQue_.template DeQue<float>();
        if (GetBlockIdx() == 0) {
            CopyRowScalars(this->meanGm_, mean);
        }

        LocalTensor<float> rstd = this->rstdQue_.template AllocTensor<float>();
        ReducePhase<true>(1, mean);
        MergePhase(1, rstd);
        __VEC_SCOPE__ { LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>(); }
        NormCommon::ComputeRstdNewtonRaphson<true, true>(rstd, rstd, rows_, this->eps_, this->avgFactor_,
                                                         DEEP_NORM_VL_FP32);
        this->rstdQue_.EnQue(rstd);
        rstd = this->rstdQue_.template DeQue<float>();
        if (GetBlockIdx() == 0) {
            CopyRowScalars(this->rstdGm_, rstd);
        }
        OutputRows(mean, rstd);
        this->meanQue_.FreeTensor(mean);
        this->rstdQue_.FreeTensor(rstd);
    }

private:
    template <bool SQUARE>
    __aicore__ inline void ReducePhase(uint32_t phase, LocalTensor<float>& mean)
    {
        LocalTensor<float> part = partQueue_.template AllocTensor<float>();
        // Padding and unused levels are also copied back after the barrier.
        Duplicate(part, 0.0f, partElements_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> calc = this->calcBuf_.template Get<float>();
        LocalTensor<float> tileSum = this->tileSumBuf_.template Get<float>();
        for (uint32_t row = 0; row < rows_; ++row) {
            LocalTensor<float> rowMean = mean[row];
            for (uint32_t tile = 0; tile < localTiles_; ++tile) {
                uint64_t col = static_cast<uint64_t>(firstTile_ + tile) * this->tileLength_;
                uint32_t count = static_cast<uint32_t>(
                    DeepNormMin(static_cast<uint64_t>(this->tileLength_), this->numCol_ - col));
                uint64_t offset = static_cast<uint64_t>(row) * this->numCol_ + col;
                LocalTensor<T> x = this->CopyInTensor(this->xQue_, this->xGm_, offset, count);
                LocalTensor<T> gx = this->CopyInTensor(this->gxQue_, this->gxGm_, offset, count);
                if constexpr (SQUARE) {
                    this->ComputeCenteredSquare(x, gx, rowMean, calc, count);
                } else {
                    this->ComputeH(x, gx, calc, count);
                }
                NormCommon::NormCommonRegbase::CalculateReduceSum(calc, tileSum, this->reduceTmpBuf_, count,
                                                                  this->GetPowerSplit(count));
                this->AccumulateBinary(tileSum, tile);
                this->xQue_.FreeTensor(x);
                this->gxQue_.FreeTensor(gx);
            }
            ExportCache(part, row);
            // ExportCache reads the cache that the next row will overwrite.
            PipeBarrier<PIPE_V>();
        }
        partQueue_.EnQue(part);
        part = partQueue_.template DeQue<float>();
        uint64_t offset = static_cast<uint64_t>(phase) * phaseElements_ +
                          static_cast<uint64_t>(isTailGroup_ ? fullGroups_ : GetBlockIdx()) * rowAlign_;
        DataCopyExtParams params{1, partElements_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(workspaceGm_[offset], part, params);
        // All blocks execute this once per phase, including the final short group.
        // Separate sum/variance regions prevent an early writer from replacing
        // sums while another block is still reading them.
        SyncAll<true, DEEP_NORM_SPLIT_SYNC>();
        partQueue_.FreeTensor(part);
    }

    __aicore__ inline void ExportCache(LocalTensor<float>& part, uint32_t row)
    {
        LocalTensor<float> cache = this->binarySumBuf_.template Get<float>();
        __local_mem__ float* cachePtr = (__local_mem__ float*)cache.GetPhyAddr();
        __local_mem__ float* partPtr = (__local_mem__ float*)part.GetPhyAddr();
        uint32_t rowAlign = rowAlign_;
        uint32_t tailCount = tailTiles_;
        uint16_t groupLevel = groupLevel_;
        bool tailGroup = isTailGroup_;
        __VEC_SCOPE__
        {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            RegTensor<float> value;
            MaskReg mask = CreateMask<float, MaskPattern::VL1>();
            if (!tailGroup) {
                // Export the raw root, without the extra +0 from FinalizeBinary.
                DataCopy<float, LoadDist::DIST_BRC_B32>(value, cachePtr + groupLevel * DEEP_NORM_BLK_B32);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(partPtr + row, value, mask);
            } else {
                for (uint16_t level = 0; level < groupLevel; ++level) {
                    if ((tailCount & (1U << level)) != 0U) {
                        DataCopy<float, LoadDist::DIST_BRC_B32>(value, cachePtr + level * DEEP_NORM_BLK_B32);
                        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(partPtr + level * rowAlign + row, value,
                                                                           mask);
                    }
                }
            }
        }
    }

    __aicore__ inline void MergePhase(uint32_t phase, LocalTensor<float>& result)
    {
        LocalTensor<float> comb = combQueue_.template AllocTensor<float>();
        DataCopyExtParams params{1, phaseElements_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(comb, workspaceGm_[static_cast<uint64_t>(phase) * phaseElements_], params, pad);
        combQueue_.EnQue(comb);
        comb = combQueue_.template DeQue<float>();
        Duplicate(result, 0.0f, scalarCapacity_);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < rows_; ++row) {
            for (uint32_t group = 0; group < fullGroups_; ++group) {
                LocalTensor<float> root = comb[group * rowAlign_ + row];
                this->AccumulateBinary(root, group);
            }
            FinalizeSplit(comb, result, row);
            PipeBarrier<PIPE_V>();
        }
        combQueue_.FreeTensor(comb);
    }

    __aicore__ inline void FinalizeSplit(LocalTensor<float>& comb, LocalTensor<float>& dst, uint32_t row)
    {
        LocalTensor<float> cache = this->binarySumBuf_.template Get<float>();
        __local_mem__ float* cachePtr = (__local_mem__ float*)cache.GetPhyAddr();
        __local_mem__ float* combPtr = (__local_mem__ float*)comb.GetPhyAddr();
        __local_mem__ float* dstPtr = (__local_mem__ float*)dst.GetPhyAddr();
        uint32_t rowAlign = rowAlign_;
        uint32_t tileCount = tileCount_;
        uint32_t fullGroups = fullGroups_;
        uint16_t groupLevel = groupLevel_;
        __VEC_SCOPE__
        {
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            RegTensor<float> value;
            RegTensor<float> result;
            MaskReg mask = CreateMask<float, MaskPattern::VL1>();
            Duplicate(result, 0.0f, mask);
            for (uint16_t level = 0; level < DEEP_NORM_BINARY_LEVELS; ++level) {
                if ((tileCount & (1U << level)) != 0U) {
                    if (level < groupLevel) {
                        DataCopy<float, LoadDist::DIST_BRC_B32>(value, combPtr + (fullGroups + level) * rowAlign + row);
                    } else {
                        DataCopy<float, LoadDist::DIST_BRC_B32>(value,
                                                                cachePtr + (level - groupLevel) * DEEP_NORM_BLK_B32);
                    }
                    // Preserve both grouping and operand order from FinalizeBinary.
                    Add(result, value, result, mask);
                }
            }
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + row, result, mask);
        }
    }

    __aicore__ inline void OutputRows(LocalTensor<float>& mean, LocalTensor<float>& rstd)
    {
        // Each block owns the same column interval in every row. Reuse the
        // affine parameters for all rows before advancing to the next tile.
        for (uint32_t tile = 0; tile < localTiles_; ++tile) {
            uint64_t col = static_cast<uint64_t>(firstTile_ + tile) * this->tileLength_;
            uint32_t count = static_cast<uint32_t>(
                DeepNormMin(static_cast<uint64_t>(this->tileLength_), this->numCol_ - col));
            LocalTensor<T> gamma = this->CopyInTensor(this->gammaQue_, this->gammaGm_, col, count);
            LocalTensor<T> beta = this->CopyInTensor(this->betaQue_, this->betaGm_, col, count);
            for (uint32_t row = 0; row < rows_; ++row) {
                uint64_t offset = static_cast<uint64_t>(row) * this->numCol_ + col;
                LocalTensor<T> x = this->CopyInTensor(this->xQue_, this->xGm_, offset, count);
                LocalTensor<T> gx = this->CopyInTensor(this->gxQue_, this->gxGm_, offset, count);
                LocalTensor<T> y = this->yQue_.template AllocTensor<T>();
                LocalTensor<float> rowMean = mean[row];
                LocalTensor<float> rowRstd = rstd[row];
                this->ComputeY(x, gx, gamma, beta, rowMean, rowRstd, y, count);
                this->yQue_.EnQue(y);
                this->xQue_.FreeTensor(x);
                this->gxQue_.FreeTensor(gx);
                y = this->yQue_.template DeQue<T>();
                this->CopyOutTensor(this->yGm_, y, offset, count);
                this->yQue_.FreeTensor(y);
            }
            this->gammaQue_.FreeTensor(gamma);
            this->betaQue_.FreeTensor(beta);
        }
    }

    __aicore__ inline void CopyRowScalars(GlobalTensor<float>& gm, LocalTensor<float>& values)
    {
        DataCopyExtParams params{1, rows_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(gm, values, params);
    }

    TQue<QuePosition::VECOUT, 1> partQueue_;
    TQue<QuePosition::VECIN, 1> combQueue_;
    GlobalTensor<float> workspaceGm_;
    uint32_t rows_ = 0;
    uint32_t rowAlign_ = 0;
    uint32_t scalarCapacity_ = 0;
    uint32_t tileCount_ = 0;
    uint32_t tilesPerCore_ = 0;
    uint32_t fullGroups_ = 0;
    uint32_t tailTiles_ = 0;
    uint32_t firstTile_ = 0;
    uint32_t localTiles_ = 0;
    uint32_t phaseElements_ = 0;
    uint32_t partElements_ = 0;
    uint16_t groupLevel_ = 0;
    bool isTailGroup_ = false;
};

} // namespace NsDeepNorm

#endif // DEEP_NORM_SPLIT_ROW_ARCH35_H
