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
 * \file pool3d_grad_ksize_one_kernel.h
 * \brief MaxPool3DGrad ksize=1 (kD=kH=kW=1) 专用 kernel (NCDHW), 复用 small kernel 轴切块.
 *        stride==1: 扁平拷贝; stride>1: strided scatter (W/H·W/D·H·W/NC·D·H·W 四分发).
 *        无需 orig_x / orig_y / argmax buffer.
 */

#ifndef POOL3D_GRAD_KSIZE_ONE_KERNEL_H_
#define POOL3D_GRAD_KSIZE_ONE_KERNEL_H_

#include "kernel_operator.h"
#include "pool3d_grad_struct_common.h"
#include "../pool_3d_common/arch35/pool_3d_grad_kernel_base.h"
#include "pool_utils/arch35/index/pool_grad_nchw_scatter_index.h"

namespace Pool3DGradKsizeOneNameSpace {
using namespace AscendC;
using namespace Pool3DGradNameSpace;
using namespace Pool3DGradCommon;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_BYTES = 32; // UB/DataBlock 32 字节, 架构不变量

// 散落索引类型: 硬件强制 (2 字节数据→uint16, 4 字节数据→uint32)
template <typename T>
using IndexU = typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;

// 仅 Arange 使用 (Arange 只吃有符号): uint32→int32, uint16→int16
template <typename U>
struct IndexArith {
    using type = typename std::conditional<
        std::is_same<U, uint32_t>::value, int32_t,
        typename std::conditional<std::is_same<U, uint16_t>::value, int16_t, U>::type>::type;
};

// 整数除法封装: fp32(U=uint32) 走 magic 除法, fp16(U=uint16) 走原生 Div, 全程无符号无重解释
template <typename T, typename U>
__simd_callee__ inline void DivIndex(Reg::RegTensor<U>& res, Reg::RegTensor<U>& src, U divisor, uint32_t magic,
                                     int16_t shift, Reg::MaskReg& mask)
{
    if constexpr (sizeof(T) == 4) {
        Reg::RegTensor<U> magicReg;
        Reg::Duplicate(magicReg, magic);
        PoolUtils::Index::FastDivImplVF(res, src, magicReg, shift, mask);
    } else {
        Reg::RegTensor<U> divisorReg;
        Reg::Duplicate(divisorReg, divisor);
        Reg::Div(res, src, divisorReg, mask);
    }
}

template <typename T>
class Pool3DGradKsizeOne : public Pool3DGradNcdhwKernelBase {
public:
    __aicore__ inline Pool3DGradKsizeOne(TPipe& pipe, const Pool3DGradNCDHWTilingData& tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void CopyInGrad();
    __aicore__ inline void Backward();
    __aicore__ inline void CopyOut();
    __aicore__ inline void ProcessIdentity();
    __aicore__ inline void ProcessZeroBlock();
    __aicore__ inline void ScatterRowWiseW(__ubuf__ T* yAddr, __ubuf__ T* gradAddr, uint32_t vecLenElems);

    __simd_vf__ inline void ScatterGradRow(__ubuf__ T* yRowAddr, __ubuf__ T* gradRowAddr, IndexU<T> wStride,
                                           uint32_t count);
    __simd_vf__ inline void ScatterGradTileHw(__ubuf__ T* yBaseAddr, __ubuf__ T* gradAddr, IndexU<T> startFlat,
                                              uint32_t totalElems, IndexU<T> wGrad, IndexU<T> hStride,
                                              IndexU<T> wStride, IndexU<T> gradPitch, IndexU<T> yRowBufElems,
                                              uint32_t magicW, int16_t shiftW);
    __simd_vf__ inline void ScatterGradTile(__ubuf__ T* yBaseAddr, __ubuf__ T* gradAddr, IndexU<T> startFlat,
                                            uint32_t totalElems, IndexU<T> hGrad, IndexU<T> wGrad, IndexU<T> dStride,
                                            IndexU<T> hStride, IndexU<T> wStride, IndexU<T> hOut, IndexU<T> gradPitch,
                                            IndexU<T> yRowBufElems, uint32_t magicHW, int16_t shiftHW, uint32_t magicW,
                                            int16_t shiftW);
    __simd_vf__ inline void ScatterGradTileNc(__ubuf__ T* yBaseAddr, __ubuf__ T* gradAddr, IndexU<T> startFlat,
                                              uint32_t totalElems, IndexU<T> dGrad, IndexU<T> hGrad, IndexU<T> wGrad,
                                              IndexU<T> dStride, IndexU<T> hStride, IndexU<T> wStride, IndexU<T> dOut,
                                              IndexU<T> hOut, IndexU<T> gradPitch, IndexU<T> yRowBufElems,
                                              uint32_t magicDHW, int16_t shiftDHW, uint32_t magicHW, int16_t shiftHW,
                                              uint32_t magicW, int16_t shiftW);

    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();

    TPipe& pipe_;
    const Pool3DGradNCDHWTilingData& tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQue_;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> yGm_;

    // ScalarCompute 计算的 strided grad 映射 (grad = y 的 strided 子集)
    int64_t dGradStart_ = 0;
    int64_t hGradStart_ = 0;
    int64_t wGradStart_ = 0;
    int64_t dGradActual_ = 0;
    int64_t hGradActual_ = 0;
    int64_t wGradActual_ = 0;
    int64_t wOutputAlignedBlock_ = 0;
    int64_t wGradAlignedBlock_ = 0;
    int64_t gradGmOffset_ = 0;
    int64_t yGmOffset_ = 0;
    int64_t phaseOffset_ = 0; // 散落相位偏移: 块起点相对首个 strided 位置的元素偏移
};

template <typename T>
__aicore__ inline void Pool3DGradKsizeOne<T>::Init(GM_ADDR grads, GM_ADDR y)
{
    ParseTilingData(tilingData_);
    blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
    curCoreProcessNum_ = (blockIdx_ + 1 == usedCoreNum_) ? tailCoreProcessNum_ : normalCoreProcessNum_;

    gradGm_.SetGlobalBuffer((__gm__ T*)grads);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    pipe_.InitBuffer(gradQue_, BUFFER_NUM, static_cast<uint32_t>(gradBufferSize_));
    pipe_.InitBuffer(yQue_, BUFFER_NUM, static_cast<uint32_t>(outputBufferSize_));
}

template <typename T>
__aicore__ inline void Pool3DGradKsizeOne<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    if (strideD_ == 1 && strideH_ == 1 && strideW_ == 1) {
        ProcessIdentity();
        return;
    }
    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        if (dGradActual_ <= 0 || hGradActual_ <= 0 || wGradActual_ <= 0) {
            ProcessZeroBlock();
            continue;
        }
        CopyInGrad();
        Backward();
        CopyOut();
    }
}

template <typename T>
__aicore__ inline void Pool3DGradKsizeOne<T>::ProcessIdentity()
{
    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; loopNum++) {
        ScalarCompute(loopNum);
        CopyInGrad();
        // identity: y == grad, 但需经一次 V 指令建立 MTE2->V->MTE3 同步链(纯 MTE2 直连 MTE3 会读脏数据)
        LocalTensor<T> gradLocal = gradQue_.DeQue<T>();
        LocalTensor<T> yLocal = yQue_.AllocTensor<T>();
        uint32_t yCount = static_cast<uint32_t>(highAxisActual_ * dOutputActual_ * hOutputActual_ *
                                                wOutputAlignedBlock_);
        DataCopy(yLocal, gradLocal, yCount);
        yQue_.EnQue(yLocal);
        gradQue_.FreeTensor(gradLocal);
        CopyOut();
    }
}

template <typename T>
__aicore__ inline void Pool3DGradKsizeOne<T>::ProcessZeroBlock()
{
    // 本 tile 无任何 grad 落点(非 stride 对齐切分下 *GradActual_ 可为 0): y 全 0 直接写回
    uint32_t yCount = static_cast<uint32_t>(highAxisActual_ * dOutputActual_ * hOutputActual_ * wOutputAlignedBlock_);
    LocalTensor<T> yLocal = yQue_.AllocTensor<T>();
    Duplicate(yLocal, T(0), yCount);
    yQue_.EnQue(yLocal);
    CopyOut();
}

template <typename T>
__aicore__ inline void Pool3DGradKsizeOne<T>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIdx = static_cast<int64_t>(blockIdx_) * normalCoreProcessNum_ + loopNum;

    int64_t dOuter = dOutputOuter_;
    int64_t hOuter = hOutputOuter_;
    int64_t wOuter = wOutputOuter_;

    highAxisIndex_ = baseBlockIdx / (dOuter * hOuter * wOuter);
    highAxisActual_ = (highAxisIndex_ == highAxisOuter_ - 1) ? highAxisTail_ : highAxisInner_;
    int64_t remDHW = baseBlockIdx - (dOuter * hOuter * wOuter) * (baseBlockIdx / (dOuter * hOuter * wOuter));
    dAxisIndex_ = remDHW / (hOuter * wOuter);
    dOutputActual_ = (dAxisIndex_ == dOuter - 1) ? dOutputTail_ : dOutputInner_;
    int64_t remHW = remDHW - (hOuter * wOuter) * (remDHW / (hOuter * wOuter));
    hAxisIndex_ = remHW / wOuter;
    hOutputActual_ = (hAxisIndex_ == hOuter - 1) ? hOutputTail_ : hOutputInner_;
    wAxisIndex_ = remHW - wOuter * (remHW / wOuter);
    wOutputActual_ = (wAxisIndex_ == wOuter - 1) ? wOutputTail_ : wOutputInner_;

    // kernel=1: 输出轴上的 strided 位置映射为连续的 grad 子块
    int64_t dStart = dAxisIndex_ * dOutputInner_;
    int64_t hStart = hAxisIndex_ * hOutputInner_;
    int64_t wStart = wAxisIndex_ * wOutputInner_;
    // 块起点可能非 stride 对齐(非对齐切分路径), grad 子块起点用 ceil(否则 floor 会少一格),
    // 子块个数 = ceil(end/stride) - ceil(start/stride) (而非 ceil(span/stride), 否则非对齐时多算)
    dGradStart_ = (dStart + strideD_ - 1) / strideD_;
    hGradStart_ = (hStart + strideH_ - 1) / strideH_;
    wGradStart_ = (wStart + strideW_ - 1) / strideW_;
    dGradActual_ = (dStart + dOutputActual_ + strideD_ - 1) / strideD_ - dGradStart_;
    hGradActual_ = (hStart + hOutputActual_ + strideH_ - 1) / strideH_ - hGradStart_;
    wGradActual_ = (wStart + wOutputActual_ + strideW_ - 1) / strideW_ - wGradStart_;

    int64_t blockElems = BLOCK_BYTES / sizeof(T);
    wOutputAlignedBlock_ = ((wOutputActual_ + blockElems - 1) / blockElems) * blockElems;
    // grad 行 stride 与 y 一致按 32B 块对齐(与 small kernel 一致); 行尾读取用 masked gather 防越界
    wGradAlignedBlock_ = ((wGradActual_ + blockElems - 1) / blockElems) * blockElems;

    // 散落相位移: 块起点相对首个 strided 位置补零间隙, 折成 y 块内元素偏移加到 scatter 基址
    int64_t phaseD = dGradStart_ * strideD_ - dStart;
    int64_t phaseH = hGradStart_ * strideH_ - hStart;
    int64_t phaseW = wGradStart_ * strideW_ - wStart;
    phaseOffset_ = phaseD * hOutputActual_ * wOutputAlignedBlock_ + phaseH * wOutputAlignedBlock_ + phaseW;

    // grad 为 NCDHW [nc, dGrad, hGrad, wGrad] 连续: nc 基址 = highAxisIndex_ * highAxisInner_(每 tile nc 数)
    gradGmOffset_ = highAxisIndex_ * highAxisInner_ * (dArgmax_ * hArgmax_ * wArgmax_) +
                    dGradStart_ * (hArgmax_ * wArgmax_) + hGradStart_ * wArgmax_ + wGradStart_;
    // y 为 NCDHW [nc, dX, hX, wX] 连续, 本块写回 [dStart.., hStart.., wStart..] 的稠密子块
    yGmOffset_ = highAxisIndex_ * highAxisInner_ * (dOutput_ * hOutput_ * wOutput_) + dStart * (hOutput_ * wOutput_) +
                 hStart * wOutput_ + wStart;
}

template <typename T>
__aicore__ inline void Pool3DGradKsizeOne<T>::CopyInGrad()
{
    LocalTensor<T> gradLocal = gradQue_.AllocTensor<T>();

    // 与小核一致: loop1=D / loop2=NC + 内层 H 行 x W 的 DataCopyPad, 行 pitch 32B 块对齐
    LoopModeParams loopParams;
    loopParams.loop1Size = static_cast<uint32_t>(dGradActual_);
    loopParams.loop2Size = static_cast<uint32_t>(highAxisActual_);
    loopParams.loop1SrcStride = static_cast<uint64_t>(hArgmax_) * wArgmax_ * sizeof(T);
    loopParams.loop2SrcStride = static_cast<uint64_t>(dArgmax_) * hArgmax_ * wArgmax_ * sizeof(T);
    loopParams.loop1DstStride = static_cast<uint64_t>(hGradActual_) * wGradAlignedBlock_ * sizeof(T);
    loopParams.loop2DstStride = static_cast<uint64_t>(dGradActual_) * hGradActual_ * wGradAlignedBlock_ * sizeof(T);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(hGradActual_);
    copyParams.blockLen = static_cast<uint32_t>(wGradActual_ * sizeof(T));
    copyParams.srcStride = static_cast<uint32_t>((wArgmax_ - wGradActual_) * sizeof(T));
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPad(gradLocal, gradGm_[gradGmOffset_], copyParams, padParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    gradQue_.EnQue(gradLocal);
}

template <typename T>
__aicore__ inline void Pool3DGradKsizeOne<T>::Backward()
{
    uint32_t vecLenElems = static_cast<uint32_t>(V_REG_SIZE / sizeof(T));
    uint32_t yCount = static_cast<uint32_t>(highAxisActual_ * dOutputActual_ * hOutputActual_ * wOutputAlignedBlock_);

    LocalTensor<T> yLocal = yQue_.AllocTensor<T>();
    Duplicate(yLocal, T(0), yCount);
    LocalTensor<T> gradLocal = gradQue_.DeQue<T>();
    __ubuf__ T* yAddr = (__ubuf__ T*)yLocal.GetPhyAddr() + phaseOffset_;
    __ubuf__ T* gradAddr = (__ubuf__ T*)gradLocal.GetPhyAddr();

    uint32_t magicW = 0;
    uint32_t shiftW = 0;
    uint32_t magicHW = 0;
    uint32_t shiftHW = 0;
    uint32_t magicDHW = 0;
    uint32_t shiftDHW = 0;
    if constexpr (sizeof(T) == 4) {
        GetUintDivMagicAndShift<uint32_t>(magicW, shiftW, static_cast<uint32_t>(wGradActual_));
        GetUintDivMagicAndShift<uint32_t>(magicHW, shiftHW, static_cast<uint32_t>(hGradActual_ * wGradActual_));
        GetUintDivMagicAndShift<uint32_t>(magicDHW, shiftDHW,
                                          static_cast<uint32_t>(dGradActual_ * hGradActual_ * wGradActual_));
    }

    // 块维度/stride 收敛为无符号索引类型 (int64 → IndexU, 每值一次)
    IndexU<T> dGradU = static_cast<IndexU<T>>(dGradActual_);
    IndexU<T> hGradU = static_cast<IndexU<T>>(hGradActual_);
    IndexU<T> wGradU = static_cast<IndexU<T>>(wGradActual_);
    IndexU<T> dOutU = static_cast<IndexU<T>>(dOutputActual_);
    IndexU<T> hOutU = static_cast<IndexU<T>>(hOutputActual_);
    IndexU<T> wOutAlignedU = static_cast<IndexU<T>>(wOutputAlignedBlock_);
    IndexU<T> dStrideU = static_cast<IndexU<T>>(strideD_);
    IndexU<T> hStrideU = static_cast<IndexU<T>>(strideH_);
    IndexU<T> wStrideU = static_cast<IndexU<T>>(strideW_);
    IndexU<T> wGradPitchU = static_cast<IndexU<T>>(wGradAlignedBlock_);

    if (wGradActual_ >= vecLenElems) {
        // 单行: 行内 W 按向量切块, 循环所有行
        ScatterRowWiseW(yAddr, gradAddr, vecLenElems);
    } else if (hGradActual_ * wGradActual_ >= vecLenElems) {
        // H·W: 每个 (nc,d) 平面内把多行塞满一个向量 (源索引用 padded pitch)
        int64_t hwPlane = hGradActual_ * wGradActual_;
        int64_t ncD = highAxisActual_ * dGradActual_;
        for (int64_t nd = 0; nd < ncD; ++nd) {
            int64_t nc = nd / dGradActual_;
            int64_t d = nd - nc * dGradActual_;
            __ubuf__ T* yPlane = yAddr + (nc * dOutputActual_ + d * strideD_) * hOutputActual_ * wOutputAlignedBlock_;
            __ubuf__ T* gradPlane = gradAddr + (nc * dGradActual_ + d) * hGradActual_ * wGradAlignedBlock_;
            for (int64_t off = 0; off < hwPlane; off += vecLenElems) {
                uint32_t subChunk = (hwPlane - off < vecLenElems) ? static_cast<uint32_t>(hwPlane - off) : vecLenElems;
                ScatterGradTileHw(yPlane, gradPlane, static_cast<IndexU<T>>(off), subChunk, wGradU, hStrideU, wStrideU,
                                  wGradPitchU, wOutAlignedU, magicW, static_cast<int16_t>(shiftW));
            }
        }
    } else if (dGradActual_ * hGradActual_ * wGradActual_ >= vecLenElems) {
        // D·H·W: 逐 nc 平面打包 (源索引用 padded pitch)
        int64_t dhwPlane = dGradActual_ * hGradActual_ * wGradActual_;
        for (int64_t nc = 0; nc < highAxisActual_; ++nc) {
            __ubuf__ T* yPlane = yAddr + nc * dOutputActual_ * hOutputActual_ * wOutputAlignedBlock_;
            __ubuf__ T* gradPlane = gradAddr + nc * dGradActual_ * hGradActual_ * wGradAlignedBlock_;
            for (int64_t off = 0; off < dhwPlane; off += vecLenElems) {
                uint32_t subChunk = (dhwPlane - off < vecLenElems) ? static_cast<uint32_t>(dhwPlane - off) :
                                                                     vecLenElems;
                ScatterGradTile(yPlane, gradPlane, static_cast<IndexU<T>>(off), subChunk, hGradU, wGradU, dStrideU,
                                hStrideU, wStrideU, hOutU, wGradPitchU, wOutAlignedU, magicHW,
                                static_cast<int16_t>(shiftHW), magicW, static_cast<int16_t>(shiftW));
            }
        }
    } else {
        // NC·D·H·W (源索引用 padded pitch)
        int64_t gradTotal = highAxisActual_ * dGradActual_ * hGradActual_ * wGradActual_;
        for (int64_t off = 0; off < gradTotal; off += vecLenElems) {
            uint32_t subChunk = (gradTotal - off < vecLenElems) ? static_cast<uint32_t>(gradTotal - off) : vecLenElems;
            ScatterGradTileNc(yAddr, gradAddr, static_cast<IndexU<T>>(off), subChunk, dGradU, hGradU, wGradU, dStrideU,
                              hStrideU, wStrideU, dOutU, hOutU, wGradPitchU, wOutAlignedU, magicDHW,
                              static_cast<int16_t>(shiftDHW), magicHW, static_cast<int16_t>(shiftHW), magicW,
                              static_cast<int16_t>(shiftW));
        }
    }

    gradQue_.FreeTensor(gradLocal);
    yQue_.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void Pool3DGradKsizeOne<T>::CopyOut()
{
    LocalTensor<T> yLocal = yQue_.DeQue<T>();
    LoopModeParams loopParams;
    loopParams.loop1Size = static_cast<uint32_t>(dOutputActual_);
    loopParams.loop2Size = static_cast<uint32_t>(highAxisActual_);
    loopParams.loop1SrcStride = static_cast<uint64_t>(hOutputActual_) * wOutputAlignedBlock_ * sizeof(T);
    loopParams.loop2SrcStride = static_cast<uint64_t>(dOutputActual_) * hOutputActual_ * wOutputAlignedBlock_ *
                                sizeof(T);
    loopParams.loop1DstStride = static_cast<uint64_t>(hOutput_) * wOutput_ * sizeof(T);
    loopParams.loop2DstStride = static_cast<uint64_t>(dOutput_) * hOutput_ * wOutput_ * sizeof(T);
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

    DataCopyExtParams copyParam;
    copyParam.blockCount = static_cast<uint16_t>(hOutputActual_);
    copyParam.blockLen = static_cast<uint32_t>(wOutputActual_ * sizeof(T));
    copyParam.srcStride = 0;
    copyParam.dstStride = static_cast<uint32_t>((wOutput_ - wOutputActual_) * sizeof(T));
    copyParam.rsv = 0;
    DataCopyPad(yGm_[yGmOffset_], yLocal, copyParam);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    yQue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Pool3DGradKsizeOne<T>::ScatterRowWiseW(__ubuf__ T* yAddr, __ubuf__ T* gradAddr,
                                                              uint32_t vecLenElems)
{
    int64_t gradRows = highAxisActual_ * dGradActual_ * hGradActual_;
    int64_t dH = dGradActual_ * hGradActual_;
    IndexU<T> wStride = static_cast<IndexU<T>>(strideW_);
    for (int64_t row = 0; row < gradRows; ++row) {
        int64_t nc = row / dH;
        int64_t rem = row - nc * dH;
        int64_t d = rem / hGradActual_;
        int64_t h = rem - d * hGradActual_;
        __ubuf__ T* yRowAddr = yAddr + ((nc * dOutputActual_ + d * strideD_) * hOutputActual_ + h * strideH_) *
                                           wOutputAlignedBlock_;
        __ubuf__ T* gradRowAddr = gradAddr + row * wGradAlignedBlock_;

        int64_t loopW = wGradActual_ / vecLenElems;
        int64_t tailW = wGradActual_ - loopW * vecLenElems;
        for (int64_t i = 0; i < loopW; ++i) {
            ScatterGradRow(yRowAddr + i * vecLenElems * wStride, gradRowAddr + i * vecLenElems, wStride, vecLenElems);
        }
        if (tailW > 0) {
            ScatterGradRow(yRowAddr + loopW * vecLenElems * wStride, gradRowAddr + loopW * vecLenElems, wStride,
                           static_cast<uint32_t>(tailW));
        }
    }
}

template <typename T>
__simd_vf__ inline void Pool3DGradKsizeOne<T>::ScatterGradRow(__ubuf__ T* yRowAddr, __ubuf__ T* gradRowAddr,
                                                              IndexU<T> wStride, uint32_t count)
{
    using ArithT = typename IndexArith<IndexU<T>>::type;

    Reg::RegTensor<IndexU<T>> indexReg;
    Reg::RegTensor<IndexU<T>> srcIndexReg;
    Reg::RegTensor<T> gradReg;
    Reg::MaskReg maskAll = Reg::CreateMask<IndexU<T>, Reg::MaskPattern::ALL>();
    // 生成确定性目标索引 [0, wStride, 2*wStride, ...]
    Reg::Arange((Reg::RegTensor<ArithT>&)indexReg, 0);
    Reg::Muls(indexReg, indexReg, wStride, maskAll);

    Reg::MaskReg mask = Reg::UpdateMask<IndexU<T>>(count);
    // 源连续索引 [0, 1, 2, ...], masked gather 只读 count 个, 避免 32B 行 pitch 下尾块整向量读越界
    Reg::Arange((Reg::RegTensor<ArithT>&)srcIndexReg, 0);
    Reg::DataCopyGather(gradReg, gradRowAddr, srcIndexReg, mask);
    Reg::Scatter(yRowAddr, gradReg, indexReg, mask);
}

template <typename T>
__simd_vf__ inline void Pool3DGradKsizeOne<T>::ScatterGradTileHw(__ubuf__ T* yBaseAddr, __ubuf__ T* gradAddr,
                                                                 IndexU<T> startFlat, uint32_t totalElems,
                                                                 IndexU<T> wGrad, IndexU<T> hStride, IndexU<T> wStride,
                                                                 IndexU<T> gradPitch, IndexU<T> yRowBufElems,
                                                                 uint32_t magicW, int16_t shiftW)
{
    using ArithT = typename IndexArith<IndexU<T>>::type;

    Reg::RegTensor<IndexU<T>> indexReg;
    Reg::RegTensor<IndexU<T>> srcIndexReg;
    Reg::RegTensor<T> gradReg;
    Reg::MaskReg maskAll = Reg::CreateMask<IndexU<T>, Reg::MaskPattern::ALL>();

    // 2D 目标索引: index[(h,w)] = h*hStride*yRowBufElems + w*wStride
    Reg::RegTensor<IndexU<T>> hReg;
    Reg::RegTensor<IndexU<T>> wReg;
    Reg::RegTensor<IndexU<T>> tmpReg;
    Reg::Arange((Reg::RegTensor<ArithT>&)indexReg, 0);
    Reg::Adds(indexReg, indexReg, startFlat, maskAll);
    DivIndex<T, IndexU<T>>(hReg, indexReg, wGrad, magicW, shiftW, maskAll);
    Reg::Muls(wReg, hReg, wGrad, maskAll);
    Reg::Sub(wReg, indexReg, wReg, maskAll);

    // 源索引(含行尾 padding): h*gradPitch + w
    Reg::Muls(tmpReg, hReg, gradPitch, maskAll);
    Reg::Add(srcIndexReg, tmpReg, wReg, maskAll);

    Reg::Muls(hReg, hReg, static_cast<IndexU<T>>(hStride * yRowBufElems), maskAll);
    Reg::Muls(wReg, wReg, wStride, maskAll);
    Reg::Add(indexReg, hReg, wReg, maskAll);

    Reg::MaskReg mask = Reg::UpdateMask<IndexU<T>>(totalElems);
    Reg::DataCopyGather(gradReg, gradAddr, srcIndexReg, mask);
    Reg::Scatter(yBaseAddr, gradReg, indexReg, mask);
}

template <typename T>
__simd_vf__ inline void Pool3DGradKsizeOne<T>::ScatterGradTile(
    __ubuf__ T* yBaseAddr, __ubuf__ T* gradAddr, IndexU<T> startFlat, uint32_t totalElems, IndexU<T> hGrad,
    IndexU<T> wGrad, IndexU<T> dStride, IndexU<T> hStride, IndexU<T> wStride, IndexU<T> hOut, IndexU<T> gradPitch,
    IndexU<T> yRowBufElems, uint32_t magicHW, int16_t shiftHW, uint32_t magicW, int16_t shiftW)
{
    using ArithT = typename IndexArith<IndexU<T>>::type;

    Reg::RegTensor<IndexU<T>> indexReg;
    Reg::RegTensor<IndexU<T>> srcIndexReg;
    Reg::RegTensor<T> gradReg;
    Reg::MaskReg maskAll = Reg::CreateMask<IndexU<T>, Reg::MaskPattern::ALL>();

    // 3D 目标索引: index[(d,h,w)] = d*dStride*hOut*yRowBufElems + h*hStride*yRowBufElems + w*wStride
    Reg::RegTensor<IndexU<T>> dReg;
    Reg::RegTensor<IndexU<T>> hReg;
    Reg::RegTensor<IndexU<T>> wReg;
    Reg::RegTensor<IndexU<T>> tmpReg;
    Reg::Arange((Reg::RegTensor<ArithT>&)indexReg, 0);
    Reg::Adds(indexReg, indexReg, startFlat, maskAll);
    DivIndex<T, IndexU<T>>(dReg, indexReg, static_cast<IndexU<T>>(hGrad * wGrad), magicHW, shiftHW, maskAll);
    Reg::Muls(tmpReg, dReg, static_cast<IndexU<T>>(hGrad * wGrad), maskAll);
    Reg::Sub(tmpReg, indexReg, tmpReg, maskAll);
    DivIndex<T, IndexU<T>>(hReg, tmpReg, wGrad, magicW, shiftW, maskAll);
    Reg::Muls(wReg, hReg, wGrad, maskAll);
    Reg::Sub(wReg, tmpReg, wReg, maskAll);

    // 源索引(含行尾 padding): d*(hGrad*gradPitch) + h*gradPitch + w
    Reg::Muls(tmpReg, dReg, static_cast<IndexU<T>>(static_cast<int64_t>(hGrad) * gradPitch), maskAll);
    Reg::Muls(srcIndexReg, hReg, gradPitch, maskAll);
    Reg::Add(srcIndexReg, srcIndexReg, tmpReg, maskAll);
    Reg::Add(srcIndexReg, srcIndexReg, wReg, maskAll);

    Reg::Muls(dReg, dReg, static_cast<IndexU<T>>(dStride * hOut * yRowBufElems), maskAll);
    Reg::Muls(hReg, hReg, static_cast<IndexU<T>>(hStride * yRowBufElems), maskAll);
    Reg::Muls(wReg, wReg, wStride, maskAll);
    Reg::Add(indexReg, dReg, hReg, maskAll);
    Reg::Add(indexReg, indexReg, wReg, maskAll);

    Reg::MaskReg mask = Reg::UpdateMask<IndexU<T>>(totalElems);
    Reg::DataCopyGather(gradReg, gradAddr, srcIndexReg, mask);
    Reg::Scatter(yBaseAddr, gradReg, indexReg, mask);
}

template <typename T>
__simd_vf__ inline void Pool3DGradKsizeOne<T>::ScatterGradTileNc(
    __ubuf__ T* yBaseAddr, __ubuf__ T* gradAddr, IndexU<T> startFlat, uint32_t totalElems, IndexU<T> dGrad,
    IndexU<T> hGrad, IndexU<T> wGrad, IndexU<T> dStride, IndexU<T> hStride, IndexU<T> wStride, IndexU<T> dOut,
    IndexU<T> hOut, IndexU<T> gradPitch, IndexU<T> yRowBufElems, uint32_t magicDHW, int16_t shiftDHW, uint32_t magicHW,
    int16_t shiftHW, uint32_t magicW, int16_t shiftW)
{
    using ArithT = typename IndexArith<IndexU<T>>::type;

    Reg::RegTensor<IndexU<T>> indexReg;
    Reg::RegTensor<IndexU<T>> srcIndexReg;
    Reg::RegTensor<T> gradReg;
    Reg::MaskReg maskAll = Reg::CreateMask<IndexU<T>, Reg::MaskPattern::ALL>();

    // 4D 目标索引: index[(nc,d,h,w)] = nc*dOut*hOut*yRowBufElems + d*dStride*hOut*yRowBufElems
    // + h*hStride*yRowBufElems + w*wStride
    Reg::RegTensor<IndexU<T>> ncReg;
    Reg::RegTensor<IndexU<T>> dReg;
    Reg::RegTensor<IndexU<T>> hReg;
    Reg::RegTensor<IndexU<T>> wReg;
    Reg::RegTensor<IndexU<T>> tmpReg;
    Reg::RegTensor<IndexU<T>> dhwRemReg;
    Reg::Arange((Reg::RegTensor<ArithT>&)indexReg, 0);
    Reg::Adds(indexReg, indexReg, startFlat, maskAll);
    DivIndex<T, IndexU<T>>(ncReg, indexReg, static_cast<IndexU<T>>(dGrad * hGrad * wGrad), magicDHW, shiftDHW, maskAll);
    Reg::Muls(tmpReg, ncReg, static_cast<IndexU<T>>(dGrad * hGrad * wGrad), maskAll);
    Reg::Sub(tmpReg, indexReg, tmpReg, maskAll);
    DivIndex<T, IndexU<T>>(dReg, tmpReg, static_cast<IndexU<T>>(hGrad * wGrad), magicHW, shiftHW, maskAll);
    Reg::Muls(dhwRemReg, dReg, static_cast<IndexU<T>>(hGrad * wGrad), maskAll);
    Reg::Sub(tmpReg, tmpReg, dhwRemReg, maskAll);
    DivIndex<T, IndexU<T>>(hReg, tmpReg, wGrad, magicW, shiftW, maskAll);
    Reg::Muls(wReg, hReg, wGrad, maskAll);
    Reg::Sub(wReg, tmpReg, wReg, maskAll);

    // 源索引(含 padding): nc*(dGrad*hGrad*gradPitch) + d*(hGrad*gradPitch) + h*gradPitch + w
    Reg::Muls(dhwRemReg, dReg, static_cast<IndexU<T>>(static_cast<int64_t>(hGrad) * gradPitch), maskAll);
    Reg::Muls(srcIndexReg, hReg, gradPitch, maskAll);
    Reg::Add(srcIndexReg, srcIndexReg, dhwRemReg, maskAll);
    Reg::Add(srcIndexReg, srcIndexReg, wReg, maskAll);
    Reg::Muls(tmpReg, ncReg, static_cast<IndexU<T>>(static_cast<int64_t>(dGrad) * hGrad * gradPitch), maskAll);
    Reg::Add(srcIndexReg, srcIndexReg, tmpReg, maskAll);

    Reg::Muls(ncReg, ncReg, static_cast<IndexU<T>>(dOut * hOut * yRowBufElems), maskAll);
    Reg::Muls(dReg, dReg, static_cast<IndexU<T>>(dStride * hOut * yRowBufElems), maskAll);
    Reg::Muls(hReg, hReg, static_cast<IndexU<T>>(hStride * yRowBufElems), maskAll);
    Reg::Muls(wReg, wReg, wStride, maskAll);
    Reg::Add(indexReg, ncReg, dReg, maskAll);
    Reg::Add(indexReg, indexReg, hReg, maskAll);
    Reg::Add(indexReg, indexReg, wReg, maskAll);

    Reg::MaskReg mask = Reg::UpdateMask<IndexU<T>>(totalElems);
    Reg::DataCopyGather(gradReg, gradAddr, srcIndexReg, mask);
    Reg::Scatter(yBaseAddr, gradReg, indexReg, mask);
}

} // namespace Pool3DGradKsizeOneNameSpace

#endif // POOL3D_GRAD_KSIZE_ONE_KERNEL_H_
