/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"

using namespace AscendC;

#define PIPE_FIX (pipe_t)10

// 查询硬件事件方向；sourcePipe/destinationPipe 分别为产生事件和消费事件的流水。
__aicore__ constexpr HardEvent GetHardEventByPipe(pipe_t sourcePipe, pipe_t destinationPipe)
{
    if (sourcePipe == PIPE_MTE2) {
        if (destinationPipe == PIPE_MTE1) {
            return HardEvent::MTE2_MTE1;
        } else if (destinationPipe == PIPE_V) {
            return HardEvent::MTE2_V;
        } else if (destinationPipe == PIPE_MTE3) {
            return HardEvent::MTE2_MTE3;
        }
    } else if (sourcePipe == PIPE_MTE1) {
        if (destinationPipe == PIPE_MTE2) {
            return HardEvent::MTE1_MTE2;
        } else if (destinationPipe == PIPE_M) {
            return HardEvent::MTE1_M;
        } else if (destinationPipe == PIPE_FIX) {
            return HardEvent::MTE1_FIX;
        }
    } else if (sourcePipe == PIPE_M) {
        if (destinationPipe == PIPE_MTE1) {
            return HardEvent::M_MTE1;
        } else if (destinationPipe == PIPE_FIX) {
            return HardEvent::M_FIX;
        }
    } else if (sourcePipe == PIPE_FIX) {
        if (destinationPipe == PIPE_M) {
            return HardEvent::FIX_M;
        } else if (destinationPipe == PIPE_MTE1) {
            return HardEvent::FIX_MTE1;
        }
    } else if (sourcePipe == PIPE_V) {
        if (destinationPipe == PIPE_MTE2) {
            return HardEvent::V_MTE2;
        } else if (destinationPipe == PIPE_MTE3) {
            return HardEvent::V_MTE3;
        }
    } else if (sourcePipe == PIPE_MTE3) {
        if (destinationPipe == PIPE_V) {
            return HardEvent::MTE3_V;
        } else if (destinationPipe == PIPE_MTE2) {
            return HardEvent::MTE3_MTE2;
        }
    }
    return HardEvent::MTE3_MTE2;
}

// 返回较小值；left/right 为待比较的两个值。
template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 left, T2 right)
{
    return (left < right) ? left : right;
}

// 建立片上张量视图；byteAddress 为起始字节地址，elementCount 为元素数，pos/T 为位置和元素类型。
template <TPosition pos, typename T>
__aicore__ inline LocalTensor<T> TensorAt(int byteAddress, int elementCount)
{
    return LocalTensor<T>(pos, (uint32_t)byteAddress, (uint32_t)elementCount);
}

template <typename T, TPosition pos>
class DBuff {
public:
    // 默认构造缓存视图，地址由 InitAt 设置；无参数。
    __aicore__ inline DBuff() {}

    // 建立相邻 ping/pong 视图；byteAddress 为 ping 字节地址，slotElements 为单槽元素数。
    __aicore__ inline void InitAt(int byteAddress, int slotElements)
    {
        ping = LocalTensor<T>(pos, (uint32_t)byteAddress, (uint32_t)slotElements);
        pong = LocalTensor<T>(pos, (uint32_t)(byteAddress + slotElements * (int)sizeof(T)), (uint32_t)slotElements);
    }

    // 按交替序号取槽；sequence 为消费或生产序号，偶数取 ping，奇数取 pong。
    __aicore__ inline LocalTensor<T> get(int sequence)
    {
        if (sequence % 2 == 0) {
            return ping;
        } else {
            return pong;
        }
    }

private:
    LocalTensor<T> ping; // ping 物理槽视图。
    LocalTensor<T> pong; // pong 物理槽视图。
};

template <pipe_t sourcePipe, pipe_t destinationPipe, bool preset, int ID>
class SEvent {
public:
    // 分配1个事件并按 preset 预置许可；sourcePipe/destinationPipe 为模板流水方向，无运行参数。
    __aicore__ inline SEvent()
    {
#if FC2D_FIXED_EVID
        pingEvent = (event_t)ID;
#else
        pingEvent = (event_t)GetTPipePtr()->AllocEventID<GetHardEventByPipe(sourcePipe, destinationPipe)>();
#endif
        if constexpr (preset) {
            setall();
        }
    }
    // 析构时回收尚未消费的预置许可；无参数。
    __aicore__ inline ~SEvent()
    {
        if constexpr (preset) {
            release();
        }
    }
    // 等候并消费下一个就绪许可；无参数。
    __aicore__ inline void wait() { wait_flag(sourcePipe, destinationPipe, pingEvent); }
    // 发布下一个就绪许可；无参数。
    __aicore__ inline void set() { set_flag(sourcePipe, destinationPipe, pingEvent); }
    // 为所有事件预置许可；无参数。
    __aicore__ inline void setall() { set(); }
    // 消费尚未回收的事件许可；无参数。
    __aicore__ inline void release() { wait(); }

private:
    event_t pingEvent; // 单事件编号。
};

template <pipe_t sourcePipe, pipe_t destinationPipe, bool preset, int ID0, int ID1>
class DEvent {
public:
    // 分配2个事件并按 preset 预置许可；sourcePipe/destinationPipe 为模板流水方向，无运行参数。
    __aicore__ inline DEvent()
    {
#if FC2D_FIXED_EVID
        pingEvent = (event_t)ID0;
        pongEvent = (event_t)ID1;
#else
        pingEvent = (event_t)GetTPipePtr()->AllocEventID<GetHardEventByPipe(sourcePipe, destinationPipe)>();
        pongEvent = (event_t)GetTPipePtr()->AllocEventID<GetHardEventByPipe(sourcePipe, destinationPipe)>();
#endif
        if constexpr (preset) {
            setall();
        }
    }
    // 析构时回收尚未消费的预置许可；无参数。
    __aicore__ inline ~DEvent()
    {
        if constexpr (preset) {
            release();
        }
    }
    // 等候并消费下一个就绪许可；无参数。
    __aicore__ inline void wait()
    {
        if (waitSequence % 2 == 0) {
            wait_flag(sourcePipe, destinationPipe, pingEvent);
        } else {
            wait_flag(sourcePipe, destinationPipe, pongEvent);
        }
        waitSequence++;
    }
    // 发布下一个就绪许可；无参数。
    __aicore__ inline void set()
    {
        if (setSequence % 2 == 0) {
            set_flag(sourcePipe, destinationPipe, pingEvent);
        } else {
            set_flag(sourcePipe, destinationPipe, pongEvent);
        }
        setSequence++;
    }
    // 为所有事件预置许可；无参数。
    __aicore__ inline void setall()
    {
        set();
        set();
    }
    // 消费尚未回收的事件许可；无参数。
    __aicore__ inline void release()
    {
        for (int pendingSequence = waitSequence; pendingSequence < setSequence; ++pendingSequence) {
            wait();
        }
    }

private:
    event_t pingEvent;    // ping 事件编号。
    event_t pongEvent;    // pong 事件编号。
    int waitSequence = 0; // 已消费许可的交替序号。
    int setSequence = 0;  // 已发布许可的交替序号。
};

// 搬入 L1；destination/source 为目标/源，burstCount 为段数，burstBlocks 为每段 32 字节块数，
// sourceStride/destinationStride 为段间跳过的 32 字节块数。
template <typename T>
__aicore__ inline void GM2L1(LocalTensor<T> destination, GlobalTensor<T> source, int burstCount, int burstBlocks,
                             int sourceStride, int destinationStride)
{
    constexpr int MAX_BLOCKS = 2047;
    if (burstCount == 1 && burstBlocks > MAX_BLOCKS) {
        int copiedBlocks = 0;
        while (copiedBlocks < burstBlocks) {
            const int segmentBlocks = (burstBlocks - copiedBlocks) > MAX_BLOCKS ? MAX_BLOCKS :
                                                                                  (burstBlocks - copiedBlocks);
            DataCopyParams segmentParams;
            segmentParams.blockCount = 1;
            segmentParams.blockLen = (uint16_t)segmentBlocks;
            segmentParams.srcStride = 0;
            segmentParams.dstStride = 0;

            const int elementsPerBlock = 32 / (int)sizeof(T);
            DataCopy(destination[copiedBlocks * elementsPerBlock], source[copiedBlocks * elementsPerBlock],
                     segmentParams);
            copiedBlocks += segmentBlocks;
        }
        return;
    }
    DataCopyParams copyParams;
    copyParams.blockCount = burstCount;
    copyParams.blockLen = burstBlocks;
    copyParams.srcStride = sourceStride;
    copyParams.dstStride = destinationStride;
    DataCopy(destination, source, copyParams);
}

// 按字节长度搬入 L1；destination/source 为目标/源，burstCount/burstBytes 为段数/段长，
// sourceStrideBytes 为源字节间距，destinationStride 为目标 32 字节块间距；当前不启用补零。
template <typename T>
__aicore__ inline void GM2L1PAD(LocalTensor<T> destination, GlobalTensor<T> source, int burstCount, int burstBytes,
                                int sourceStrideBytes, int destinationStride)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = burstCount;
    copyParams.blockLen = burstBytes;
    copyParams.srcStride = sourceStrideBytes;
    copyParams.dstStride = destinationStride;

    DataCopyPadExtParams<T> paddingParams;
    paddingParams.isPad = false;
    paddingParams.leftPadding = 0;
    paddingParams.rightPadding = 0;
    DataCopyPad<T, PaddingMode::Normal>(destination, source, copyParams, paddingParams);
}

template <typename T>
struct Load3DCtx {
    Load3DBitModeParam bitParams; // Load3D 硬件编码参数。
};

// 配置窗口展开；context 为配置输出，inputHeight/inputWidth/channels 为输入窗口，
// kernelHeight/kernelWidth 为卷积核，padLeft/padRight/padTop/padBottom 为补边，
// strideH/strideW、dilationH/dilationW 为步长/膨胀。
template <typename T>
__aicore__ inline void LOAD3D_BEGIN(Load3DCtx<T>& context, int inputHeight, int inputWidth, int channels,
                                    int kernelHeight, int kernelWidth, int padLeft, int padRight, int padTop,
                                    int padBottom, int strideH, int strideW, int dilationH, int dilationW)
{
    uint8_t padding[4] = {(uint8_t)padLeft, (uint8_t)padRight, (uint8_t)padTop, (uint8_t)padBottom};
    SetFmatrix((uint16_t)inputHeight, (uint16_t)inputWidth, padding, FmatrixMode::FMATRIX_LEFT);
    SetLoadDataPaddingValue<T>((T)0);

    LoadData3DParamsV2<T> loadParams;
    loadParams.padList[0] = (uint8_t)padLeft;
    loadParams.padList[1] = (uint8_t)padRight;
    loadParams.padList[2] = (uint8_t)padTop;
    loadParams.padList[3] = (uint8_t)padBottom;
    loadParams.l1H = (uint16_t)inputHeight;
    loadParams.l1W = (uint16_t)inputWidth;
    loadParams.channelSize = (uint16_t)channels;

    loadParams.kExtension = (uint16_t)(32 / sizeof(T));
    loadParams.mExtension = 0;
    loadParams.kStartPt = 0;
    loadParams.mStartPt = 0;
    loadParams.strideW = (uint8_t)strideW;
    loadParams.strideH = (uint8_t)strideH;
    loadParams.filterW = (uint8_t)kernelWidth;
    loadParams.filterH = (uint8_t)kernelHeight;
    loadParams.dilationFilterW = (uint8_t)dilationW;
    loadParams.dilationFilterH = (uint8_t)dilationH;
    loadParams.enTranspose = false;
    loadParams.enSmallK = false;
    loadParams.padValue = (T)0;
    loadParams.filterSizeW = false;
    loadParams.filterSizeH = false;
    loadParams.fMatrixCtrl = false;
    context.bitParams = Load3DBitModeParam(loadParams);
}

// 搬入一个卷积矩阵块；context 为窗口配置，destination/source 为 L0A/L1，
// kStart/mStart 为展开矩阵起点，kLength/mLength 为有效 K/M 长度。
template <typename T>
__aicore__ inline void LOAD3D_RUN(Load3DCtx<T>& context, LocalTensor<T> destination, LocalTensor<T> source, int kStart,
                                  int mStart, int kLength, int mLength)
{
    context.bitParams.SetMExtension((uint16_t)mLength);
    context.bitParams.SetKExtension((uint16_t)kLength);
    context.bitParams.SetKStartPt((uint16_t)kStart);
    context.bitParams.SetMStartPt((uint16_t)mStart);
    // 整块展开连续 K 轴；相邻 C0 块按对齐后的 M 长度写入 L0A。
#if defined(ASC_DEVKIT_VERSION_NUM) && ASC_DEVKIT_VERSION_NUM >= 90000000
    LoadDataRepeatParamWithStride repeatParams = {0, 1, 0, (uint16_t)((mLength + 15) / 16)};
    SetLoadDataRepeatWithStride(repeatParams);
#else
    LoadDataRepeatParam repeatParams = {0, 1, 0, (uint16_t)((mLength + 15) / 16)};
    SetLoadDataRepeat(repeatParams);
#endif
    LoadData<TPosition::A2, TPosition::A1, T>(destination, source, context.bitParams);
}

// 搬入 FRACTAL_Z 权重；destination/source 为 L0B/L1，destinationChannels/destinationK 为目标 N/K，
// sourceChannels/sourceK 为源 N/K，其中源 K 由连续布局隐含，无需单独配置。
template <typename T>
__aicore__ inline void L0NZ2NZ(LocalTensor<T> destination, LocalTensor<T> source, int destinationChannels,
                               int destinationK, int sourceChannels, int sourceK)
{
    LoadData2DParamsV2 loadParams;
    loadParams.mStep = (destinationChannels + 15) / 16;
    loadParams.kStep = (destinationK + 32 / sizeof(T) - 1) / (32 / sizeof(T));
    loadParams.srcStride = (sourceChannels + 15) / 16;
    loadParams.dstStride = (destinationChannels + 15) / 16;
    LoadData(destination, source, loadParams);
}

// 搬入偏置；destination/source 为 BT/L1，channelCount 为通道数，fixShiftVal 为偏置定标。
template <typename TDst, typename TSrc>
__aicore__ inline void L1_TO_BT(LocalTensor<TDst> destination, LocalTensor<TSrc> source, int channelCount,
                                uint8_t fixShiftVal = 0)
{
    constexpr int sourceLengthFactor = (sizeof(TDst) == sizeof(TSrc)) ? 2 : 1;
    uint16_t burstBlocks = (uint16_t)(((channelCount * sourceLengthFactor * 2) + 31) / 32);
    if (sizeof(TDst) == sizeof(TSrc)) {
        burstBlocks = (uint16_t)(((burstBlocks + 1) / 2) * 2);
    }
    DataCopyParams btParams{(uint16_t)1, burstBlocks, (uint16_t)0, (uint16_t)0};

    btParams.fixShiftVal = fixShiftVal;
    DataCopy(destination, source, btParams);
}

// 逐通道重量化到 L1 NZ；destination/source 为目标/L0C，positions/outputChannels 为有效 M/N，
// accumulatorPositions 为 L0C 的 M 跨距，destinationPositions 为目标 M 容量，destinationNzStride 为目标 C0
// 块空间跨度（0 时按 M 对齐）， scaleTable 为逐通道定标表，reluEn 控制普通 ReLU，unitFlag 为指令标记。
template <typename T, typename T2>
__aicore__ inline void L0C2L1_VREQ8(LocalTensor<T> destination, LocalTensor<T2> source,
                                    LocalTensor<uint64_t> scaleTable, int positions, int outputChannels,
                                    int destinationPositions, int accumulatorPositions, bool reluEn = false,
                                    uint8_t unitFlag = 0, int destinationNzStride = 0)
{
    FixpipeParamsArch3510<CO2Layout::NZ> fixParams;
    fixParams.nSize = (uint16_t)outputChannels;
    fixParams.mSize = (uint16_t)positions;
    fixParams.srcStride = (uint16_t)((accumulatorPositions + 15) / 16 * 16);

    fixParams.dstStride = (uint32_t)((destinationNzStride != 0 ? destinationNzStride :
                                                                 (destinationPositions + 15) / 16 * 16) *
                                     (32 / (int)sizeof(T)));
    fixParams.quantPre = QuantMode_t::VREQ8;
    fixParams.preReluMode = ReluMode::NO_RELU;
    fixParams.preClipReluMode = ClipReluMode::NOCLIP_RELU;
    fixParams.reluEn = reluEn;
    fixParams.unitFlag = unitFlag;
    fixParams.isChannelSplit = false;
    Fixpipe<T, T2, CFG_NZ>(destination, source, scaleTable, fixParams);
}

// 逐通道重量化到 GM NCHW；destination/source 为目标/L0C，positions/outputChannels 为有效 M/N，
// accumulatorPositions 为 L0C 的 M 跨距，outputPlane 为目标输出通道跨度，
// scaleTable 为逐通道定标表，reluEn 控制普通 ReLU，unitFlag 为指令标记。
template <typename T, typename T2>
__aicore__ inline void L0C2GM_NZ2DN_VREQ8(GlobalTensor<T> destination, LocalTensor<T2> source,
                                          LocalTensor<uint64_t> scaleTable, int positions, int outputChannels,
                                          int outputPlane, int accumulatorPositions, bool reluEn = false,
                                          uint8_t unitFlag = 0)
{
    FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> fixParams;
    fixParams.nSize = (uint16_t)outputChannels;
    fixParams.mSize = (uint16_t)positions;
    fixParams.srcStride = (uint16_t)((accumulatorPositions + 15) / 16 * 16);
    fixParams.dstStride = (uint32_t)outputPlane;
    fixParams.quantPre = QuantMode_t::VREQ8;
    fixParams.preReluMode = ReluMode::NO_RELU;
    fixParams.preClipReluMode = ClipReluMode::NOCLIP_RELU;
    fixParams.reluEn = reluEn;
    fixParams.unitFlag = unitFlag;
    fixParams.isChannelSplit = false;
    fixParams.params.dnNum = 1;
    fixParams.params.srcNzMatrixStride = 0;
    fixParams.params.dstDnMatrixStride = 0;
    fixParams.params.srcNzC0Stride = 1;
    Fixpipe<T, T2, CFG_COLUMN_MAJOR>(destination, source, scaleTable, fixParams);
}

// 逐通道反量化到 GM FLOAT16；destination/source 为目标/L0C，positions/outputChannels 为有效 M/N，
// accumulatorPositions 为 L0C 的 M 跨距，outputPlane 为目标输出通道跨度，
// scaleTable 为逐通道定标表，reluEn 控制普通 ReLU，unitFlag 为指令标记。
template <typename T, typename T2>
__aicore__ inline void L0C2GM_NZ2DN_VDEQF16(GlobalTensor<T> destination, LocalTensor<T2> source,
                                            LocalTensor<uint64_t> scaleTable, int positions, int outputChannels,
                                            int outputPlane, int accumulatorPositions, bool reluEn = false,
                                            uint8_t unitFlag = 0)
{
    FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> fixParams;
    fixParams.nSize = (uint16_t)outputChannels;
    fixParams.mSize = (uint16_t)positions;
    fixParams.srcStride = (uint16_t)((accumulatorPositions + 15) / 16 * 16);
    fixParams.dstStride = (uint32_t)outputPlane;
    fixParams.quantPre = QuantMode_t::VDEQF16;
    fixParams.preReluMode = ReluMode::NO_RELU;
    fixParams.preClipReluMode = ClipReluMode::NOCLIP_RELU;
    fixParams.reluEn = reluEn;
    fixParams.unitFlag = unitFlag;
    fixParams.isChannelSplit = false;

    fixParams.params.dnNum = 1;
    fixParams.params.srcNzMatrixStride = 0;
    fixParams.params.dstDnMatrixStride = 0;
    fixParams.params.srcNzC0Stride = 1;
    Fixpipe<T, T2, CFG_COLUMN_MAJOR>(destination, source, scaleTable, fixParams);
}

// 构造矩阵乘参数；positions/reductionLength/outputChannels 为 M/K/N，
// cmatrixInitVal 控制清零，cmatrixSource 控制 BT 初值，fixShiftVal 为定标，unitFlag 为指令标记。
__aicore__ inline MmadParams MakeMmadParams(uint16_t positions, uint16_t reductionLength, uint16_t outputChannels,
                                            bool cmatrixInitVal, bool cmatrixSource, uint8_t fixShiftVal,
                                            uint8_t unitFlag)
{
    MmadParams matrixParams;
    matrixParams.m = (uint16_t)((positions + 15) / 16 * 16);
    matrixParams.n = outputChannels;
    matrixParams.k = reductionLength;
    matrixParams.cmatrixInitVal = cmatrixInitVal;
    matrixParams.cmatrixSource = cmatrixSource;
    matrixParams.unitFlag = unitFlag;
    matrixParams.fixShiftVal = fixShiftVal;
    return matrixParams;
}

// 执行矩阵乘；destination/input/weight 为 L0C/L0A/L0B，positions/reductionLength/outputChannels 为 M/K/N，
// cmatrixInitVal 控制首次清零，fixShiftVal 为定标，unitFlag 为指令标记。
template <typename T1, typename T2, typename T3>
__aicore__ inline void MMAD(LocalTensor<T1> destination, LocalTensor<T2> input, LocalTensor<T3> weight,
                            uint16_t positions, uint16_t reductionLength, uint16_t outputChannels, bool cmatrixInitVal,
                            uint8_t fixShiftVal = 0, uint8_t unitFlag = 0)
{
    MmadParams matrixParams = MakeMmadParams(positions, reductionLength, outputChannels, cmatrixInitVal, false,
                                             fixShiftVal, unitFlag);
    Mmad(destination, input, weight, matrixParams);
}

// 执行带偏置的首次矩阵乘；destination/input/weight/biasTable 为 L0C/L0A/L0B/BT，
// positions/reductionLength/outputChannels 为 M/K/N，fixShiftVal 为定标，unitFlag 为指令标记。
template <typename T1, typename T2, typename T3, typename TB>
__aicore__ inline void MMAD_BIAS(LocalTensor<T1> destination, LocalTensor<T2> input, LocalTensor<T3> weight,
                                 LocalTensor<TB> biasTable, uint16_t positions, uint16_t reductionLength,
                                 uint16_t outputChannels, uint8_t fixShiftVal = 0, uint8_t unitFlag = 0)
{
    MmadParams matrixParams = MakeMmadParams(positions, reductionLength, outputChannels, false, true, fixShiftVal,
                                             unitFlag);
    Mmad(destination, input, weight, biasTable, matrixParams);
}

constexpr uint64_t FIXED_POINT_ONE = 1065353216ULL;

constexpr FixpipeConfig CFG_NZ_FIXED = {CO2Layout::NZ, false, true};
constexpr FixpipeConfig CFG_COLUMN_MAJOR_FIXED = {CO2Layout::COLUMN_MAJOR, false, true};

// 将 NCHW 搬入 NC1HWC0；destination/source 为 L1/GM，channels/positions 为通道数/空间位置数，
// sourcePlane 为源通道跨距，destinationChannelBlockStride 为目标相邻 C0 块的空间跨度（0 时取 positions）。
template <typename T>
__aicore__ inline void GM2L1_DN2NZ(LocalTensor<T> destination, GlobalTensor<T> source, int channels, int positions,
                                   int sourcePlane, int destinationChannelBlockStride = 0)
{
    Dn2NzParams copyParams;
    copyParams.dnNum = 1;
    copyParams.nValue = (uint16_t)positions;
    copyParams.dValue = (uint32_t)channels;
    copyParams.srcDnMatrixStride = 0;
    copyParams.srcDValue = (uint64_t)sourcePlane;
    copyParams.dstNzC0Stride = (uint16_t)(destinationChannelBlockStride != 0 ? destinationChannelBlockStride :
                                                                               positions);
    copyParams.dstNzNStride = 1;
    copyParams.dstNzMatrixStride = 0;

    DataCopy(destination, source, copyParams);
}

// 按定点定标转换到 L1 FLOAT16 NZ；destination/source 为目标/L0C，positions/outputChannels 为有效 M/N，
// accumulatorPositions 为 L0C 的 M 跨距，destinationPositions 为目标 M 容量，destinationNzStride 为目标 C0
// 块空间跨度（0 时按 M 对齐）， fixShiftVal 为定点定标，reluEn 控制普通 ReLU，unitFlag 为指令标记。
template <typename T, typename T2>
__aicore__ inline void L0C2L1_DEQF16(LocalTensor<T> destination, LocalTensor<T2> source, int positions,
                                     int outputChannels, int destinationPositions, int accumulatorPositions,
                                     uint8_t fixShiftVal, bool reluEn = false, uint8_t unitFlag = 0,
                                     int destinationNzStride = 0)
{
    FixpipeParamsArch3510<CO2Layout::NZ> fixParams;
    fixParams.nSize = (uint16_t)outputChannels;
    fixParams.mSize = (uint16_t)positions;
    fixParams.srcStride = (uint16_t)((accumulatorPositions + 15) / 16 * 16);

    fixParams.dstStride = (uint32_t)((destinationNzStride != 0 ? destinationNzStride :
                                                                 (destinationPositions + 15) / 16 * 16) *
                                     (32 / (int)sizeof(T)));
    fixParams.quantPre = QuantMode_t::DEQF16;
    fixParams.deqScalar = FIXED_POINT_ONE;
    fixParams.fixShiftVal = fixShiftVal;
    fixParams.preReluMode = ReluMode::NO_RELU;
    fixParams.preClipReluMode = ClipReluMode::NOCLIP_RELU;
    fixParams.reluEn = reluEn;
    fixParams.unitFlag = unitFlag;
    fixParams.isChannelSplit = false;
    Fixpipe<T, T2, CFG_NZ_FIXED>(destination, source, fixParams);
}

// 按定点定标转换到 GM FLOAT16 NCHW；destination/source 为目标/L0C，positions/outputChannels 为有效 M/N，
// accumulatorPositions 为 L0C 的 M 跨距，outputPlane 为目标输出通道跨度，
// fixShiftVal 为定点定标，reluEn 控制普通 ReLU，unitFlag 为指令标记。
template <typename T, typename T2>
__aicore__ inline void L0C2GM_NZ2DN_DEQF16(GlobalTensor<T> destination, LocalTensor<T2> source, int positions,
                                           int outputChannels, int outputPlane, int accumulatorPositions,
                                           uint8_t fixShiftVal, bool reluEn = false, uint8_t unitFlag = 0)
{
    FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> fixParams;
    fixParams.nSize = (uint16_t)outputChannels;
    fixParams.mSize = (uint16_t)positions;
    fixParams.srcStride = (uint16_t)((accumulatorPositions + 15) / 16 * 16);
    fixParams.dstStride = (uint32_t)outputPlane;
    fixParams.quantPre = QuantMode_t::DEQF16;
    fixParams.deqScalar = FIXED_POINT_ONE;
    fixParams.fixShiftVal = fixShiftVal;
    fixParams.preReluMode = ReluMode::NO_RELU;
    fixParams.preClipReluMode = ClipReluMode::NOCLIP_RELU;
    fixParams.reluEn = reluEn;
    fixParams.unitFlag = unitFlag;
    fixParams.isChannelSplit = false;
    fixParams.params.dnNum = 1;
    fixParams.params.srcNzMatrixStride = 0;
    fixParams.params.dstDnMatrixStride = 0;
    fixParams.params.srcNzC0Stride = 1;
    Fixpipe<T, T2, CFG_COLUMN_MAJOR_FIXED>(destination, source, fixParams);
}
