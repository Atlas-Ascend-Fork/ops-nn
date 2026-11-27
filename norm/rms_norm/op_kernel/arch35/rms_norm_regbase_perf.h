/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rms_norm_regbase_perf.h
 * \brief RmsNorm regbase perf schedule
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_PERF_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_PERF_H
#include "rms_norm_regbase_common.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace RmsNorm {
using namespace AscendC;
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

// Cached ComputeY keeps row-major traversal until the tile holds more than
// this many rows per column vector; above it, column-outer traversal reuses
// the gamma registers across rows.
constexpr uint16_t COL_OUTER_ROW_PER_VECTOR_THRESHOLD = 4U;

template <typename DX, typename DG, bool IS_GEMMA, bool SAVE_X_FP32 = false>
class KernelRmsNormRegBasePerf {
public:
    __aicore__ inline KernelRmsNormRegBasePerf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,
                                const RMSNormArch35TilingData* tiling)
    {
        // Tilingdata init
        curBlockIdx = GetBlockIdx();
        coreNum = GetBlockNum();
        numRow = tiling->num_row;
        numCol = tiling->num_col;
        numColAlign = tiling->num_col_align;
        ubFactor = tiling->ub_factor;
        blockFactor = tiling->block_factor;
        blockFactorTail = tiling->last_block_factor;

        epsilon = tiling->epsilon;
        avgFactor = tiling->avg_factor;
        colFlodFactor = tiling->col_flod_factor;

        uint64_t rstdBlockCount = blockFactor;
        uint64_t xBlockCount = rstdBlockCount * numCol;
        uint64_t curRstdBlockCount = rstdBlockCount;
        uint64_t curXBlockCount = xBlockCount;
        curBlockFactor = blockFactor;
        if (curBlockIdx == (coreNum - 1)) {
            curBlockFactor = blockFactorTail;
            curRstdBlockCount = blockFactorTail;
            curXBlockCount = curRstdBlockCount * numCol;
        }

        curBlockLoops = (curBlockFactor + ubFactor - 1) / ubFactor;
        curUbTails = curBlockFactor - (curBlockLoops - 1) * ubFactor;

        // init gm buffer
        xGm.SetGlobalBuffer((__gm__ DX*)x + curBlockIdx * xBlockCount, curXBlockCount);
        gammaGm.SetGlobalBuffer((__gm__ DG*)gamma, numCol);
        yGm.SetGlobalBuffer((__gm__ DX*)y + curBlockIdx * xBlockCount, curXBlockCount);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + curBlockIdx * rstdBlockCount, curRstdBlockCount);

        // init que buffer
        pipe.InitBuffer(inQueueGamma, 1, numColAlign * sizeof(DG));
        pipe.InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, ubFactor * numColAlign * sizeof(DX));
        pipe.InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, ubFactor * numColAlign * sizeof(DX));
        uint64_t rstdUbSizeAlign = (ubFactor + blockSizeB32 - 1) / blockSizeB32 * blockSizeB32 *
                                   sizeof(float); // vector length align
        pipe.InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rstdUbSizeAlign);
        // init tmp buffer
        uint64_t firstVcaddResult = ubFactor *
                                    (((colFlodFactor + VectorLenB32 - 1) / VectorLenB32 + blockSizeB32 - 1) /
                                     blockSizeB32 * blockSizeB32) *
                                    sizeof(float);
        pipe.InitBuffer(xTmpBuf, firstVcaddResult);
        if constexpr (SAVE_X_FP32) {
            // ComputeY reads whole FP32 vectors without a mask, so the last row can
            // read past the final aligned column (up to one vector tail). Round the
            // cache up to the actual last-read end instead of relying on the buffer
            // allocated right after this one; the extra tail stays inside the
            // RETAINED_SIZE_1K reserve that the host tiling already deducts.
            const uint64_t colLoops = (numColAlign + VectorLenB32 - 1) / VectorLenB32;
            const uint64_t xFp32Elements = (ubFactor - 1) * numColAlign + colLoops * VectorLenB32;
            pipe.InitBuffer(xFp32Buf, xFp32Elements * sizeof(float));
        }
        // branch 4 need 512B tmpBuffer
        pipe.InitBuffer(xReduceTmpBuf, rstdUbSizeAlign);
    }

    __aicore__ inline void Process()
    {
        // gamma copy params
        DataCopyPadExtParams<DG> dataCopyPadExtParamsGamma;
        dataCopyPadExtParamsGamma.isPad = false;
        dataCopyPadExtParamsGamma.leftPadding = 0;
        dataCopyPadExtParamsGamma.rightPadding = 0;
        dataCopyPadExtParamsGamma.paddingValue = 0;
        DataCopyExtParams copyInParamsGamma;
        copyInParamsGamma.blockCount = 1;
        copyInParamsGamma.blockLen = numCol * sizeof(DG);
        copyInParamsGamma.srcStride = 0;
        copyInParamsGamma.dstStride = 0;

        LocalTensor<DG> gammaLocal = inQueueGamma.AllocTensor<DG>();
        DataCopyPad(gammaLocal, gammaGm, copyInParamsGamma, dataCopyPadExtParamsGamma);

        inQueueGamma.EnQue(gammaLocal);
        inQueueGamma.DeQue<DG>();

        for (uint64_t i = 0; i < curBlockLoops; i++) {
            curUbFactor = (i == (curBlockLoops - 1)) ? curUbTails : ubFactor;

            DataCopyPadExtParams<DX> dataCopyPadExtParamsX;
            dataCopyPadExtParamsX.isPad = false;
            dataCopyPadExtParamsX.leftPadding = 0;
            dataCopyPadExtParamsX.rightPadding = 0;
            dataCopyPadExtParamsX.paddingValue = 0;
            DataCopyExtParams copyInParamsX;
            copyInParamsX.blockCount = curUbFactor;
            copyInParamsX.blockLen = numCol * sizeof(DX);
            copyInParamsX.srcStride = 0;
            copyInParamsX.dstStride = (numColAlign - numCol) * sizeof(DX) / blockSize;
            LocalTensor<DX> xLocal = inQueueX.AllocTensor<DX>();
            DataCopyPad(xLocal, xGm[i * ubFactor * numCol], copyInParamsX, dataCopyPadExtParamsX);
            inQueueX.EnQue(xLocal);
            inQueueX.DeQue<DX>();
            LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
            LocalTensor<float> xTmpLocal = xTmpBuf.Get<float>();
            LocalTensor<float> xReduceTmpLocal = xReduceTmpBuf.Get<float>();
            LocalTensor<float> xFp32Local;
            if constexpr (SAVE_X_FP32) {
                xFp32Local = xFp32Buf.Get<float>();
                ComputeSquareReduceSum(xLocal, xTmpLocal, xReduceTmpLocal, xFp32Local, curUbFactor);
            } else {
                NormCommon::NormCommonRegbase::CalculateSquareReduceSum<DX>(
                    xLocal, xReduceTmpLocal, xTmpLocal, static_cast<uint16_t>(curUbFactor),
                    static_cast<uint32_t>(numColAlign), static_cast<uint32_t>(numCol),
                    static_cast<uint32_t>(colFlodFactor), static_cast<uint32_t>(blockSizeB32),
                    static_cast<uint32_t>(numColAlign));
            }
            NormCommon::ComputeRstdNewtonRaphson<false, true>(
                xReduceTmpLocal, rstdLocal, static_cast<uint32_t>(curUbFactor), epsilon, avgFactor, VectorLenB32);

            outQueueRstd.EnQue(rstdLocal);
            outQueueRstd.DeQue<float>();

            DataCopyExtParams copyInParamsRstd;
            copyInParamsRstd.blockCount = 1;
            copyInParamsRstd.blockLen = curUbFactor * sizeof(float);
            copyInParamsRstd.srcStride = 0;
            copyInParamsRstd.dstStride = 0;
            DataCopyPad(rstdGm[i * ubFactor], rstdLocal, copyInParamsRstd);

            LocalTensor<DX> yLocal = outQueueY.AllocTensor<DX>();
            ComputeY(xLocal, gammaLocal, rstdLocal, yLocal, xFp32Local, curUbFactor);
            inQueueX.FreeTensor(xLocal);
            outQueueRstd.FreeTensor(rstdLocal);
            outQueueY.EnQue(yLocal);
            outQueueY.DeQue<DX>();

            DataCopyExtParams copyInParamsY;
            copyInParamsY.blockCount = curUbFactor;
            copyInParamsY.blockLen = numCol * sizeof(DX);
            copyInParamsY.srcStride = (numColAlign - numCol) * sizeof(DX) / blockSize;
            copyInParamsY.dstStride = 0;
            DataCopyPad(yGm[i * ubFactor * numCol], yLocal, copyInParamsY);
            outQueueY.FreeTensor(yLocal);
        }
        inQueueGamma.FreeTensor(gammaLocal);
    }

private:
    __aicore__ inline void ComputeSquareReduceSum(LocalTensor<DX> xLocal, LocalTensor<float> xTmpLocal,
                                                  LocalTensor<float> xReduceTmpLocal, LocalTensor<float> xFp32Local,
                                                  uint64_t curUbFactor)
    {
        __local_mem__ DX* xLocalAddr = (__local_mem__ DX*)xLocal.GetPhyAddr();
        __local_mem__ float* xTmpLocalUbAddr = (__local_mem__ float*)xTmpLocal.GetPhyAddr();
        __local_mem__ float* xReduceTmpLocalUbAddr = (__local_mem__ float*)xReduceTmpLocal.GetPhyAddr();
        __local_mem__ float* xFp32LocalUbAddr = (__local_mem__ float*)xFp32Local.GetPhyAddr();

        if (numColAlign <= VectorLenB32) {
            CalculateSquareReduceSumRLessThanVL(xLocalAddr, xReduceTmpLocalUbAddr, xFp32LocalUbAddr, curUbFactor,
                                                numCol, numColAlign);
        } else if (numColAlign <= (VectorLenB32 + VectorLenB32)) {
            CalculateSquareReduceSumRLessThanTwoVL(xLocalAddr, xReduceTmpLocalUbAddr, xFp32LocalUbAddr, curUbFactor,
                                                   numCol, numColAlign);
        } else if (numColAlign <= VectorLenB32 * VectorLenB32 * NUM_TWO) {
            CalculateSquareReduceSumRCommon<NUM_ONE>(xLocalAddr, xTmpLocalUbAddr, xReduceTmpLocalUbAddr,
                                                     xFp32LocalUbAddr, curUbFactor, numCol, numColAlign, colFlodFactor);
        } else {
            CalculateSquareReduceSumRCommon<NUM_TWO>(xLocalAddr, xTmpLocalUbAddr, xReduceTmpLocalUbAddr,
                                                     xFp32LocalUbAddr, curUbFactor, numCol, numColAlign, colFlodFactor);
        }
    }

    __aicore__ inline void CalculateSquareReduceSumRLessThanVL(__local_mem__ DX* xLocalAddr,
                                                               __local_mem__ float* xReduceTmpLocalUbAddr,
                                                               __local_mem__ float* xFp32LocalUbAddr,
                                                               uint64_t curUbFactor, uint64_t numCol,
                                                               uint64_t numColAlign)
    {
        uint32_t colNum = static_cast<uint32_t>(numCol);
        uint16_t curAloops = static_cast<uint16_t>(curUbFactor);
        uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> squareReg;
            RegTensor<float> sumReg;

            // rstd cal
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            uint32_t colCountSreg = colNum;
            MaskReg pregLoop = UpdateMask<float>(colCountSreg);
            for (uint16_t i = 0; i < curAloops; i++) {
                LoadTensorForDtypeTIn(xLocalAddr, xReg, pregLoop, (i * colNumAlign));
                DataCopy(xFp32LocalUbAddr + i * colNumAlign, xReg, pregLoop);
                Mul(squareReg, xReg, xReg, pregLoop);
                ReduceSum(sumReg, squareReg, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSumRLessThanTwoVL(__local_mem__ DX* xLocalAddr,
                                                                  __local_mem__ float* xReduceTmpLocalUbAddr,
                                                                  __local_mem__ float* xFp32LocalUbAddr,
                                                                  uint64_t curUbFactor, uint64_t numCol,
                                                                  uint64_t numColAlign)
    {
        uint32_t colNum = static_cast<uint32_t>(numCol);
        uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
        uint16_t curAloops = static_cast<uint16_t>(curUbFactor);

        __VEC_SCOPE__
        {
            RegTensor<float> xReg;
            RegTensor<float> xTailReg;
            RegTensor<float> squareReg;
            RegTensor<float> squareTailReg;
            RegTensor<float> addReg;
            RegTensor<float> sumReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            // Tail mask counts the remaining FP32 lanes after the first full
            // vector; loads convert X to FP32 lanes, so re-verify this width
            // only when the supported X dtype set changes.
            uint32_t colTailSreg = colNum - VectorLenB32;
            MaskReg pregTail = UpdateMask<float>(colTailSreg);
            for (uint16_t i = 0; i < curAloops; i++) {
                LoadTensorForDtypeTIn(xLocalAddr, xReg, pregFull, (i * colNumAlign));
                LoadTensorForDtypeTIn(xLocalAddr + VectorLenB32, xTailReg, pregTail, (i * colNumAlign));
                DataCopy(xFp32LocalUbAddr + i * colNumAlign, xReg, pregFull);
                DataCopy(xFp32LocalUbAddr + i * colNumAlign + VectorLenB32, xTailReg, pregTail);
                Mul(squareReg, xReg, xReg, pregFull);
                Mul(squareTailReg, xTailReg, xTailReg, pregTail);
                // Zero the inactive tail lanes explicitly before the full-mask add,
                // matching reduce_common_regbase_part2.h instead of relying on
                // implicit masked-Mul behavior.
                ShiftLefts((RegTensor<uint32_t>&)squareTailReg, (RegTensor<uint32_t>&)squareTailReg,
                           static_cast<int16_t>(0), pregTail);
                Add(addReg, squareReg, squareTailReg, pregFull);
                ReduceSum(sumReg, addReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
            }
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateSquareReduceSumRCommon(__local_mem__ DX* xLocalAddr,
                                                           __local_mem__ float* xTmpLocalUbAddr,
                                                           __local_mem__ float* xReduceTmpLocalUbAddr,
                                                           __local_mem__ float* xFp32LocalUbAddr, uint64_t curUbFactor,
                                                           uint64_t numCol, uint64_t numColAlign,
                                                           uint64_t colFlodFactor)
    {
        uint32_t colNum = static_cast<uint32_t>(numCol);
        uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
        uint32_t colFlodNum = static_cast<uint32_t>(colFlodFactor);
        uint16_t curAloops = static_cast<uint16_t>(curUbFactor);

        // first flod
        uint32_t firstFlodTial = static_cast<uint32_t>(colNum - colFlodFactor);
        uint16_t firstFlodAddLoops = static_cast<uint16_t>((firstFlodTial + VectorLenB32 - 1) / VectorLenB32);
        uint16_t firstFlodWithOutAddLoops = static_cast<uint16_t>((colFlodNum + VectorLenB32 - 1) / VectorLenB32) -
                                            firstFlodAddLoops;

        // first vcadd
        uint32_t firstVcaddNum = static_cast<uint32_t>((colFlodFactor + VectorLenB32 - 1) / VectorLenB32);
        uint32_t firstVcaddNumCeilAlign = static_cast<uint32_t>((firstVcaddNum + blockSizeB32 - 1) / blockSizeB32 *
                                                                blockSizeB32);

        // second flod
        // rstd cal

        __VEC_SCOPE__
        {
            RegTensor<float> xReg1;
            RegTensor<float> xReg2;
            RegTensor<float> squareReg1;
            RegTensor<float> squareReg2;
            RegTensor<float> addReg;
            RegTensor<float> sumReg;

            RegTensor<float> xReg3;
            RegTensor<float> squareReg3;
            RegTensor<float> sumReg3;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t i = 0; i < curAloops; i++) {
                uint32_t sregfirstFlodTial = firstFlodTial;
                for (uint16_t j = 0; j < firstFlodAddLoops; j++) {
                    pregLoop = UpdateMask<float>(sregfirstFlodTial);
                    LoadTensorForDtypeTIn(xLocalAddr, xReg1, pregFull, (i * colNumAlign + j * VectorLenB32));
                    DataCopy(xFp32LocalUbAddr + (i * colNumAlign + j * VectorLenB32), xReg1, pregFull);
                    Mul(squareReg1, xReg1, xReg1, pregFull);
                    LoadTensorForDtypeTIn(xLocalAddr + colFlodNum, xReg2, pregFull,
                                          (i * colNumAlign + j * VectorLenB32));
                    DataCopy(xFp32LocalUbAddr + (i * colNumAlign + colFlodNum + j * VectorLenB32), xReg2, pregLoop);
                    Mul(squareReg2, xReg2, xReg2, pregLoop);
                    // Zero the inactive fold-tail lanes explicitly before the full-mask
                    // add, matching reduce_common_regbase_part2.h.
                    ShiftLefts((RegTensor<uint32_t>&)squareReg2, (RegTensor<uint32_t>&)squareReg2,
                               static_cast<int16_t>(0), pregLoop);
                    Add(addReg, squareReg1, squareReg2, pregFull);
                    ReduceSum(sumReg, addReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + j), sumReg, pregOne);
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(firstFlodWithOutAddLoops); j++) {
                    LoadTensorForDtypeTIn(xLocalAddr + firstFlodAddLoops * VectorLenB32, xReg3, pregFull,
                                          (i * colNumAlign + j * VectorLenB32));
                    DataCopy(xFp32LocalUbAddr + (i * colNumAlign + firstFlodAddLoops * VectorLenB32 + j * VectorLenB32),
                             xReg3, pregFull);
                    Mul(squareReg3, xReg3, xReg3, pregFull);
                    ReduceSum(sumReg3, squareReg3, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + firstFlodAddLoops + j),
                        sumReg3, pregOne);
                }
            }

            // if need a add to last repeat
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                uint32_t sregSecondReduce = firstVcaddNum;
                MaskReg pregLast = UpdateMask<float>(sregSecondReduce);
                for (uint16_t i = 0; i < curAloops; i++) {
                    DataCopy(xReg1, xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign));
                    ReduceSum(sumReg, xReg1, pregLast);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                uint32_t sregSecondReduce = firstVcaddNum - VectorLenB32;
                MaskReg pregLast = UpdateMask<float>(sregSecondReduce);
                RegTensor<float> shiftLeft;
                for (uint16_t i = 0; i < curAloops; i++) {
                    DataCopy(xReg1, xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign));
                    DataCopy(xReg2, xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + VectorLenB32));
                    ShiftLefts((RegTensor<uint32_t>&)shiftLeft, (RegTensor<uint32_t>&)xReg2, static_cast<int16_t>(0),
                               pregLast);
                    Add(addReg, xReg1, shiftLeft, pregFull);
                    ReduceSum(sumReg, addReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
                }
            }
        }
    }

    template <typename T_IN>
    __aicore__ inline void LoadTensorForDtypeTIn(__local_mem__ T_IN* src, RegTensor<float>& dst, MaskReg& preg,
                                                 uint32_t offset)
    {
        if constexpr (IsSameType<T_IN, float>::value) {
            DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
        } else {
            RegTensor<T_IN> xIn;
            DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
            Cast<float, T_IN, castTraitB162B32>(dst, xIn, preg);
        }
    }

    __aicore__ inline void ComputeY(LocalTensor<DX> xLocal, LocalTensor<DG> gammaLocal, LocalTensor<float> rstdLocal,
                                    LocalTensor<DX> yLocal, LocalTensor<float> xFp32Local, uint64_t curUbFactor)
    {
        __ubuf__ DX* xLocalAddr = (__ubuf__ DX*)xLocal.GetPhyAddr();
        __ubuf__ float* xFp32LocalUbAddr = nullptr;
        if constexpr (SAVE_X_FP32) {
            xFp32LocalUbAddr = (__ubuf__ float*)xFp32Local.GetPhyAddr();
        }
        __ubuf__ DG* gammaLocalUbAddr = (__ubuf__ DG*)gammaLocal.GetPhyAddr();
        __ubuf__ float* rstdLocalUbAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __ubuf__ DX* yLocalUbAddr = (__ubuf__ DX*)yLocal.GetPhyAddr();

        uint16_t curAloops = static_cast<uint16_t>(curUbFactor);
        uint16_t colLoops = static_cast<uint16_t>((numColAlign + VectorLenB32 - 1) / VectorLenB32);
        uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
        if constexpr (SAVE_X_FP32) {
            // Prefer row-major traversal unless this tile has more than four rows per column vector.
            if (curAloops > colLoops * COL_OUTER_ROW_PER_VECTOR_THRESHOLD) {
                if (numCol % VectorLenB32 == 0) {
                    ComputeYColOuter<true>(xFp32LocalUbAddr, gammaLocalUbAddr, rstdLocalUbAddr, yLocalUbAddr, curAloops,
                                           colLoops, colNumAlign);
                } else {
                    ComputeYColOuter<false>(xFp32LocalUbAddr, gammaLocalUbAddr, rstdLocalUbAddr, yLocalUbAddr,
                                            curAloops, colLoops, colNumAlign);
                }
                return;
            }
        }
        ComputeYRowOuter(xLocalAddr, xFp32LocalUbAddr, gammaLocalUbAddr, rstdLocalUbAddr, yLocalUbAddr, curAloops,
                         colLoops, colNumAlign);
    }

    __aicore__ inline void ComputeYRowOuter(__ubuf__ DX* xLocalAddr, __ubuf__ float* xFp32LocalUbAddr,
                                            __ubuf__ DG* gammaLocalUbAddr, __ubuf__ float* rstdLocalUbAddr,
                                            __ubuf__ DX* yLocalUbAddr, uint16_t curAloops, uint16_t colLoops,
                                            uint32_t colNumAlign)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> RstdReg;
            RegTensor<float> gammaReg;

            RegTensor<float> xReg;
            RegTensor<float> mul1Reg;
            RegTensor<float> mul2Reg;

            for (uint16_t i = 0; i < curAloops; i++) {
                uint32_t sregElewiseNum = numCol;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(RstdReg, rstdLocalUbAddr + i);
                for (uint16_t j = 0; j < colLoops; j++) {
                    MaskReg pregCurLoop = UpdateMask<float>(sregElewiseNum);
                    if constexpr (SAVE_X_FP32) {
                        AscendC::Reg::DataCopy<float, LoadDist::DIST_NORM>(
                            xReg, xFp32LocalUbAddr + (i * colNumAlign + j * VectorLenB32));
                    } else {
                        LoadRegForDtype(xLocalAddr, xReg, pregCurLoop, (i * colNumAlign + j * VectorLenB32));
                    }
                    Mul(mul1Reg, xReg, RstdReg, pregCurLoop);
                    LoadRegForDtype(gammaLocalUbAddr, gammaReg, pregCurLoop, (j * VectorLenB32));
                    if constexpr (IS_GEMMA) {
                        RegTensor<float> gammaAddReg;
                        Adds(gammaAddReg, gammaReg, 1.0f, pregCurLoop);
                        Mul(mul2Reg, mul1Reg, gammaAddReg, pregCurLoop);
                    } else {
                        Mul(mul2Reg, mul1Reg, gammaReg, pregCurLoop);
                    }
                    StoreRegForDtype(yLocalUbAddr, mul2Reg, pregCurLoop, (i * colNumAlign + j * VectorLenB32));
                }
            }
        }
    }

    template <bool FULL_COL>
    __aicore__ inline void ComputeYColOuter(__ubuf__ float* xFp32LocalUbAddr, __ubuf__ DG* gammaLocalUbAddr,
                                            __ubuf__ float* rstdLocalUbAddr, __ubuf__ DX* yLocalUbAddr,
                                            uint16_t curAloops, uint16_t colLoops, uint32_t colNumAlign)
    {
        uint32_t colNum = static_cast<uint32_t>(numCol);
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> rstdReg;
            AscendC::Reg::RegTensor<float> gammaReg;
            AscendC::Reg::RegTensor<float> xReg;
            AscendC::Reg::RegTensor<float> mul1Reg;
            AscendC::Reg::RegTensor<float> mul2Reg;
            AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            uint32_t remaining = colNum;
            for (uint16_t j = 0; j < colLoops; j++) {
                if constexpr (!FULL_COL) {
                    mask = AscendC::Reg::UpdateMask<float>(remaining);
                }
                // Keep gamma in registers across rows; no additional UB cache is needed.
                LoadRegForDtype(gammaLocalUbAddr, gammaReg, mask, j * VectorLenB32);
                if constexpr (IS_GEMMA) {
                    AscendC::Reg::Adds(gammaReg, gammaReg, 1.0f, mask);
                }
                for (uint16_t i = 0; i < curAloops; i++) {
                    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstdReg, rstdLocalUbAddr + i);
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(
                        xReg, xFp32LocalUbAddr + static_cast<uint32_t>(i) * colNumAlign + j * VectorLenB32);
                    AscendC::Reg::Mul(mul1Reg, xReg, rstdReg, mask);
                    AscendC::Reg::Mul(mul2Reg, mul1Reg, gammaReg, mask);
                    StoreRegForDtype(yLocalUbAddr, mul2Reg, mask,
                                     static_cast<uint32_t>(i) * colNumAlign + j * VectorLenB32);
                }
            }
        }
    }

private:
    TPipe pipe;
    // QUE
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueRstd;
    TBuf<TPosition::VECCALC> xTmpBuf;
    TBuf<TPosition::VECCALC> xReduceTmpBuf;
    TBuf<TPosition::VECCALC> xFp32Buf;

    // GM tensor
    GlobalTensor<DX> xGm;
    GlobalTensor<DG> gammaGm;
    GlobalTensor<DX> yGm;
    GlobalTensor<float> rstdGm;

    uint64_t curBlockIdx;
    uint64_t coreNum;
    uint64_t numRow;
    uint64_t numCol;
    uint64_t numColAlign;
    uint64_t blockFactor;
    uint64_t ubFactor;
    uint64_t colFlodFactor;
    uint64_t blockFactorTail;
    float epsilon;
    float avgFactor;
    uint64_t curBlockFactor;
    uint64_t curUbFactor;
    uint64_t curBlockLoops;
    uint64_t curUbTails;
    uint32_t blockSize = GetUbBlockSize();
    uint32_t blockSizeB32 = GetUbBlockSize() / sizeof(float);
    static constexpr uint32_t VectorLenB32 = GetVRegSize() / sizeof(float);
    static constexpr float RMS_POS_INF = 3.40282366920938E+38;
    static constexpr float RMS_ZERO = 0.0f;
    static constexpr int32_t NUM_ONE = 1;
    static constexpr int32_t NUM_TWO = 2;
};
} // namespace RmsNorm
#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_PERF_H
