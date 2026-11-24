/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Internal implementation section of deep_norm_grad.h. Include only from deep_norm_grad.h. */

template <typename T>
__aicore__ inline void DeepNormGrad<T>::AccumulateKahan(LocalTensor<float>& sum, LocalTensor<float>& compensation,
                                                        LocalTensor<float>& value, uint32_t count)
{
    __local_mem__ float* sumPtr = (__local_mem__ float*)sum.GetPhyAddr();
    __local_mem__ float* compensationPtr = (__local_mem__ float*)compensation.GetPhyAddr();
    __local_mem__ float* valuePtr = (__local_mem__ float*)value.GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    uint32_t remaining = count;
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> sumReg;
        RegTensor<float> compensationReg;
        RegTensor<float> valueReg;
        RegTensor<float> adjustedReg;
        RegTensor<float> newSumReg;
        RegTensor<float> deltaReg;
        RegTensor<float> guardAbsReg;
        RegTensor<float> guardZeroReg;
        MaskReg mask;
        MaskReg guardCmpMask;
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = Min(remaining, VL_FP32);
            remaining -= valid;
            mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            DataCopy(sumReg, sumPtr + offset);
            DataCopy(compensationReg, compensationPtr + offset);
            DataCopy(valueReg, valuePtr + offset);
            Duplicate(guardZeroReg, 0.0F, mask);
            Sub(adjustedReg, valueReg, compensationReg, mask);
            Add(newSumReg, sumReg, adjustedReg, mask);
            Sub(deltaReg, newSumReg, sumReg, mask);
            Sub(compensationReg, deltaReg, adjustedReg, mask);
            KahanGuardComp(compensationReg, guardAbsReg, guardZeroReg, guardCmpMask, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(sumPtr + offset, newSumReg, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(compensationPtr + offset, compensationReg, mask);
        }
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::MergeDbetaPartial(LocalTensor<float>& sum, LocalTensor<float>& compensation,
                                                          LocalTensor<float>& partial, LocalTensor<float>& residual,
                                                          uint32_t count)
{
    auto sumPtr = (__local_mem__ float*)sum.GetPhyAddr();
    auto compPtr = (__local_mem__ float*)compensation.GetPhyAddr();
    auto partialPtr = (__local_mem__ float*)partial.GetPhyAddr();
    auto residualPtr = (__local_mem__ float*)residual.GetPhyAddr();
    uint16_t loops = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    uint32_t remaining = count;
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> total, correction, value, low, next, error, absSum, absValue, zero;
        MaskReg mask, cmp, guard;
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t valid = Min(remaining, VL_FP32);
            remaining -= valid;
            mask = UpdateMask<float>(valid);
            uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
            DataCopy(total, sumPtr + offset);
            DataCopy(correction, compPtr + offset);
            DataCopy(value, partialPtr + offset);
            DataCopy(low, residualPtr + offset);
            Duplicate(zero, 0.0f, mask);
            NeumaierAdd(total, correction, value, next, error, absSum, absValue, zero, cmp, guard, mask);
            Add(correction, correction, low, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(sumPtr + offset, total, mask);
            DataCopy<float, StoreDist::DIST_NORM_B32>(compPtr + offset, correction, mask);
        }
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ProcessBackwardRow(uint64_t row)
{
    LocalTensor<float> mean = CopyInScalar(meanQueue_, meanGm_, row);
    LocalTensor<float> rstd = CopyInScalar(rstdQueue_, rstdGm_, row);
    LocalTensor<float> sumTmp = sumTmpBuf_.Get<float>();
    LocalTensor<float> sumTmpNorm = sumTmpNormBuf_.Get<float>();
    LocalTensor<float> tileSums = tileSumBuf_.Get<float>();
    const bool multiTile = tiling_->numCols > tileLength_;
    // A single tile can reduce directly into the row sums.
    LocalTensor<float> tileTmp = multiTile ? tileSums : sumTmp;
    LocalTensor<float> tileTmpNorm = multiTile ? tileSums[SCALAR_BLOCK_ELEMS] : sumTmpNorm;
    LocalTensor<float> calc0 = calc0Buf_.Get<float>();
    LocalTensor<float> calc1 = calc1Buf_.Get<float>();
    // Each sum owns a UB block. Its second float holds the cross-tile rounding correction.
    LocalTensor<float> tmpComp = sumTmp[1];
    LocalTensor<float> normComp = sumTmpNorm[1];
    if (multiTile) {
        InitScalar(sumTmp);
        InitScalar(sumTmpNorm);
        InitScalar(tmpComp);
        InitScalar(normComp);
    }

    uint64_t rowOffset = row * tiling_->numCols;
    for (uint64_t col = 0; col < tiling_->numCols; col += tileLength_) {
        uint32_t count = static_cast<uint32_t>(Min(static_cast<uint64_t>(tileLength_), tiling_->numCols - col));
        LocalTensor<T> dy = CopyInTensor(dyQueue_, dyGm_, rowOffset + col, count);
        LocalTensor<T> x = CopyInTensor(xQueue_, xGm_, rowOffset + col, count);
        LocalTensor<T> gx = CopyInTensor(gxQueue_, gxGm_, rowOffset + col, count);
        LocalTensor<T> gamma = CopyInTensor(gammaQueue_, gammaGm_, col, count);
        ComputeFirstPass(dy, x, gx, gamma, mean, rstd, count);
        NormCommon::NormCommonRegbase::CalculateReduceSum(calc0, tileTmp, reduceTmpBuf_, count, GetPowerSplit(count));
        NormCommon::NormCommonRegbase::CalculateReduceSum(calc1, tileTmpNorm, reduceTmpBuf_, count,
                                                          GetPowerSplit(count));
        if (multiTile) {
            AccumulateTileSum(sumTmp, tileTmp);
            AccumulateTileSum(sumTmpNorm, tileTmpNorm);
        }
        dyQueue_.FreeTensor(dy);
        xQueue_.FreeTensor(x);
        gxQueue_.FreeTensor(gx);
        gammaQueue_.FreeTensor(gamma);
    }
    if (multiTile) {
        AccumulateScalar(sumTmp, tmpComp);
        AccumulateScalar(sumTmpNorm, normComp);
    }
    ScaleScalar(sumTmp, -invCols_);
    ScaleScalar(sumTmpNorm, -invCols_);

    for (uint64_t col = 0; col < tiling_->numCols; col += tileLength_) {
        uint32_t count = static_cast<uint32_t>(Min(static_cast<uint64_t>(tileLength_), tiling_->numCols - col));
        LocalTensor<T> dy = CopyInTensor(dyQueue_, dyGm_, rowOffset + col, count);
        LocalTensor<T> x = CopyInTensor(xQueue_, xGm_, rowOffset + col, count);
        LocalTensor<T> gx = CopyInTensor(gxQueue_, gxGm_, rowOffset + col, count);
        LocalTensor<T> gamma = CopyInTensor(gammaQueue_, gammaGm_, col, count);
        ComputeSecondPass(dy, x, gx, gamma, mean, rstd, sumTmp, sumTmpNorm, count);
        dyQueue_.FreeTensor(dy);
        xQueue_.FreeTensor(x);
        gxQueue_.FreeTensor(gx);
        gammaQueue_.FreeTensor(gamma);
        CopyOutTensor(dxQueue_, dxGm_, rowOffset + col, count);
        CopyOutTensor(dgxQueue_, dgxGm_, rowOffset + col, count);
    }
    meanQueue_.FreeTensor(mean);
    rstdQueue_.FreeTensor(rstd);
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ProcessBackward()
{
    uint32_t core = GetBlockIdx();
    if (core >= tiling_->backwardBlockDim) {
        return;
    }
    uint64_t rowBegin = static_cast<uint64_t>(core) * tiling_->rowsPerCore;
    if (rowBegin >= tiling_->numRows) {
        return;
    }
    uint64_t rowEnd = rowBegin + Min(tiling_->rowsPerCore, tiling_->numRows - rowBegin);
    for (uint64_t row = rowBegin; row < rowEnd; ++row) {
        ProcessBackwardRow(row);
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ProcessGammaBeta()
{
    uint32_t core = GetBlockIdx();
    if (core >= tiling_->gammaBetaBlockDim) {
        return;
    }
    uint64_t colBegin = static_cast<uint64_t>(core) * tiling_->colsPerCore;
    if (colBegin >= tiling_->numCols) {
        return;
    }
    uint64_t colEnd = colBegin + Min(tiling_->colsPerCore, tiling_->numCols - colBegin);
    for (uint64_t col = colBegin; col < colEnd; col += tileLength_) {
        uint32_t count = static_cast<uint32_t>(Min(static_cast<uint64_t>(tileLength_), colEnd - col));
        LocalTensor<float> dbeta = dbetaQueue_.template AllocTensor<float>();
        LocalTensor<float> dgamma = dgammaQueue_.template AllocTensor<float>();
        // 本通路 ProcessBackward 已执行完毕，calc0/calc1 空闲，复用为 dgamma/dbeta 补偿项，
        // 不额外占用 UB（tiling 的 FP32_BUFFER_COUNT 已覆盖这两个 buffer）
        LocalTensor<float> dgammaComp = calc0Buf_.Get<float>();
        LocalTensor<float> dbetaComp = calc1Buf_.Get<float>();
        __local_mem__ float* dgammaCompPtr = (__local_mem__ float*)dgammaComp.GetPhyAddr();
        __local_mem__ float* dbetaCompPtr = (__local_mem__ float*)dbetaComp.GetPhyAddr();
        Duplicate(dbeta, 0.0f, tileLengthAlign_);
        Duplicate(dgamma, 0.0f, tileLengthAlign_);
        Duplicate(dgammaComp, 0.0f, tileLengthAlign_);
        Duplicate(dbetaComp, 0.0f, tileLengthAlign_);
        for (uint64_t row = 0; row < tiling_->numRows; ++row) {
            uint64_t offset = row * tiling_->numCols + col;
            LocalTensor<float> mean = CopyInScalar(meanQueue_, meanGm_, row);
            LocalTensor<float> rstd = CopyInScalar(rstdQueue_, rstdGm_, row);
            LocalTensor<T> dy = CopyInTensor(dyQueue_, dyGm_, offset, count);
            LocalTensor<T> x = CopyInTensor(xQueue_, xGm_, offset, count);
            LocalTensor<T> gx = CopyInTensor(gxQueue_, gxGm_, offset, count);
            ComputeGammaBeta(dy, x, gx, mean, rstd, dgamma, dbeta, dgammaComp, dbetaComp, count);
            meanQueue_.FreeTensor(mean);
            rstdQueue_.FreeTensor(rstd);
            dyQueue_.FreeTensor(dy);
            xQueue_.FreeTensor(x);
            gxQueue_.FreeTensor(gx);
        }
        // 补偿项折叠回累加和：结果与 fp64 行归约（golden）位级一致
        FoldCompensation((__local_mem__ float*)dgamma.GetPhyAddr(), dgammaCompPtr, count);
        FoldCompensation((__local_mem__ float*)dbeta.GetPhyAddr(), dbetaCompPtr, count);
        dbetaQueue_.EnQue(dbeta);
        dgammaQueue_.EnQue(dgamma);
        CopyOutFloat(dbetaQueue_, dbetaGm_, col, count);
        CopyOutFloat(dgammaQueue_, dgammaGm_, col, count);
    }
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ProcessSmallD()
{
    uint32_t core = GetBlockIdx();
    uint64_t rowBegin = static_cast<uint64_t>(core) * tiling_->rowsPerCore;
    uint64_t rowEnd = rowBegin + Min(tiling_->rowsPerCore, tiling_->numRows - rowBegin);
    uint32_t count = static_cast<uint32_t>(tiling_->numCols);

    LocalTensor<T> gamma = CopyInTensor(gammaQueue_, gammaGm_, 0, count);
    LocalTensor<float> dgamma = dgammaQueue_.template AllocTensor<float>();
    LocalTensor<float> dbeta = dbetaQueue_.template AllocTensor<float>();
    LocalTensor<float> dgammaComp = dgammaCompBuf_.Get<float>();
    LocalTensor<float> dbetaComp = dbetaCompBuf_.Get<float>();
    Duplicate(dgamma, 0.0f, tiling_->smallColsAlign);
    Duplicate(dbeta, 0.0f, tiling_->smallColsAlign);
    Duplicate(dgammaComp, 0.0f, tiling_->smallColsAlign);
    Duplicate(dbetaComp, 0.0f, tiling_->smallColsAlign);

    for (uint64_t row = rowBegin; row < rowEnd; row += tiling_->smallRowsPerTile) {
        uint32_t rows = static_cast<uint32_t>(
            Min(static_cast<uint64_t>(tiling_->smallRowsPerTile), static_cast<uint64_t>(rowEnd - row)));
        uint64_t tensorOffset = row * tiling_->numCols;
        LocalTensor<T> dy = CopyInTensorBatch(dyQueue_, dyGm_, tensorOffset, rows);
        LocalTensor<T> x = CopyInTensorBatch(xQueue_, xGm_, tensorOffset, rows);
        LocalTensor<T> gx = CopyInTensorBatch(gxQueue_, gxGm_, tensorOffset, rows);
        LocalTensor<float> mean = CopyInScalars(meanQueue_, meanGm_, row, rows);
        LocalTensor<float> rstd = CopyInScalars(rstdQueue_, rstdGm_, row, rows);
        LocalTensor<T> dx = dxQueue_.template AllocTensor<T>();
        LocalTensor<T> dgx = dgxQueue_.template AllocTensor<T>();

        if (count == 1) {
            ComputeTinyDBatch<true>(dy, x, gx, gamma, mean, rstd, dx, dgx, dgamma, dbeta, dgammaComp, dbetaComp, rows);
        } else if (count <= VL_FP32) {
            ComputeTinyDBatch<false>(dy, x, gx, gamma, mean, rstd, dx, dgx, dgamma, dbeta, dgammaComp, dbetaComp, rows);
        } else {
            for (uint32_t localRow = 0; localRow < rows; ++localRow) {
                uint32_t localOffset = localRow * tiling_->smallRowStride;
                LocalTensor<T> dyRow = dy[localOffset];
                LocalTensor<T> xRow = x[localOffset];
                LocalTensor<T> gxRow = gx[localOffset];
                LocalTensor<T> dxRow = dx[localOffset];
                LocalTensor<T> dgxRow = dgx[localOffset];
                LocalTensor<float> meanRow = mean[localRow];
                LocalTensor<float> rstdRow = rstd[localRow];
                LocalTensor<float> sumTmp = sumTmpBuf_.Get<float>();
                LocalTensor<float> sumTmpNorm = sumTmpNormBuf_.Get<float>();
                LocalTensor<float> tileSums = tileSumBuf_.Get<float>();
                LocalTensor<float> tileTmp = tileSums;
                LocalTensor<float> tileTmpNorm = tileSums[SCALAR_BLOCK_ELEMS];
                LocalTensor<float> calc0 = calc0Buf_.Get<float>();
                LocalTensor<float> calc1 = calc1Buf_.Get<float>();
                InitScalar(sumTmp);
                InitScalar(sumTmpNorm);
                ComputeFirstPass(dyRow, xRow, gxRow, gamma, meanRow, rstdRow, count);
                NormCommon::NormCommonRegbase::CalculateReduceSum(calc0, tileTmp, reduceTmpBuf_, count,
                                                                  GetPowerSplit(count));
                NormCommon::NormCommonRegbase::CalculateReduceSum(calc1, tileTmpNorm, reduceTmpBuf_, count,
                                                                  GetPowerSplit(count));
                AccumulateScalar(sumTmp, tileTmp);
                AccumulateScalar(sumTmpNorm, tileTmpNorm);
                ScaleScalar(sumTmp, -invCols_);
                ScaleScalar(sumTmpNorm, -invCols_);
                ComputeSmallDSecondPass(dyRow, xRow, gxRow, gamma, meanRow, rstdRow, sumTmp, sumTmpNorm, dxRow, dgxRow,
                                        dgamma, dbeta, dgammaComp, dbetaComp, count);
            }
        }

        dxQueue_.EnQue(dx);
        dgxQueue_.EnQue(dgx);
        dyQueue_.FreeTensor(dy);
        xQueue_.FreeTensor(x);
        gxQueue_.FreeTensor(gx);
        meanQueue_.FreeTensor(mean);
        rstdQueue_.FreeTensor(rstd);
        CopyOutTensorBatch(dxQueue_, dxGm_, tensorOffset, rows);
        CopyOutTensorBatch(dgxQueue_, dgxGm_, tensorOffset, rows);
    }
    gammaQueue_.FreeTensor(gamma);

    uint64_t workspaceBase = static_cast<uint64_t>(core) * 3 * tiling_->smallColsAlign;
    dgammaQueue_.EnQue(dgamma);
    dbetaQueue_.EnQue(dbeta);
    CopyOutFloat(dgammaQueue_, workspaceGm_, workspaceBase, count);
    CopyOutFloat(dbetaQueue_, workspaceGm_, workspaceBase + tiling_->smallColsAlign, count);
    // Kahan stores rounded_sum - exact_sum. Preserve its negation for the cross-core merge.
    LocalTensor<float> residual = dbetaQueue_.template AllocTensor<float>();
    Muls(residual, dbetaComp, -1.0f, count);
    dbetaQueue_.EnQue(residual);
    CopyOutFloat(dbetaQueue_, workspaceGm_, workspaceBase + 2 * tiling_->smallColsAlign, count);
    ReduceSmallDPartials();
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::ReduceSmallDPartials()
{
    SyncAll();
    if (GetBlockIdx() != 0) {
        return;
    }

    uint32_t count = static_cast<uint32_t>(tiling_->numCols);
    LocalTensor<float> dgamma = dgammaQueue_.template AllocTensor<float>();
    LocalTensor<float> dbeta = dbetaQueue_.template AllocTensor<float>();
    LocalTensor<float> dgammaComp = calc0Buf_.Get<float>();
    LocalTensor<float> dbetaComp = calc1Buf_.Get<float>();
    Duplicate(dgamma, 0.0f, tiling_->smallColsAlign);
    Duplicate(dbeta, 0.0f, tiling_->smallColsAlign);
    Duplicate(dgammaComp, 0.0f, tiling_->smallColsAlign);
    Duplicate(dbetaComp, 0.0f, tiling_->smallColsAlign);

    for (uint32_t core = 0; core < tiling_->gammaBetaBlockDim; ++core) {
        uint64_t workspaceBase = static_cast<uint64_t>(core) * 3 * tiling_->smallColsAlign;
        LocalTensor<float> dgammaPartial = CopyInFloatTensor(dyQueue_, workspaceGm_, workspaceBase, count);
        LocalTensor<float> dbetaPartial = CopyInFloatTensor(xQueue_, workspaceGm_,
                                                            workspaceBase + tiling_->smallColsAlign, count);
        AccumulateKahan(dgamma, dgammaComp, dgammaPartial, count);
        LocalTensor<float> residual = CopyInFloatTensor(gxQueue_, workspaceGm_,
                                                        workspaceBase + 2 * tiling_->smallColsAlign, count);
        MergeDbetaPartial(dbeta, dbetaComp, dbetaPartial, residual, count);
        gxQueue_.FreeTensor(residual);
        dyQueue_.FreeTensor(dgammaPartial);
        xQueue_.FreeTensor(dbetaPartial);
    }

    FoldCompensation((__local_mem__ float*)dbeta.GetPhyAddr(), (__local_mem__ float*)dbetaComp.GetPhyAddr(), count);
    dgammaQueue_.EnQue(dgamma);
    dbetaQueue_.EnQue(dbeta);
    CopyOutFloat(dgammaQueue_, dgammaGm_, 0, count);
    CopyOutFloat(dbetaQueue_, dbetaGm_, 0, count);
}

template <typename T>
__aicore__ inline void DeepNormGrad<T>::Process()
{
    if (tiling_->gammaBetaRowSplit != 0) {
        ProcessSmallD();
    } else {
        ProcessBackward();
        ProcessGammaBeta();
    }
}
