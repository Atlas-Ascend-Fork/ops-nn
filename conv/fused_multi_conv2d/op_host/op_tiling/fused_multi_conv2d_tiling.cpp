/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <vector>
#include <limits>
#include "../fused_multi_conv2d_attrs.h"
#include "../fused_multi_conv2d_tiling.h"

#include "../../op_kernel/arch35/fused_multi_conv2d_geometry.h"

#include "op_common/log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "../../../common/op_host/conv_npu_arch_resolver.h"

namespace FusedMultiConv2dShape {

#define FC2D_GEOM_REV "geom13"

constexpr int L0B_BYTES = 64 * 1024;
constexpr int L0B_SLOT_BYTES = L0B_BYTES / 2;
constexpr int L0C_BYTES = 256 * 1024;
constexpr int L0C_ELEMS = L0C_BYTES / 4;
constexpr int L1_SEG_ALIGN = 512;
constexpr int MTE1_COST_FACTOR = 190;
constexpr int MTE2_COST_FACTOR = 120;
constexpr int MTE2_ROW_COST = 8;
constexpr int FIX_L1_COST_FACTOR = 290;
constexpr int FIX_GM_COST_FACTOR = 105;
constexpr int FIX_GM_REQUEST_COST_X2 = 19;
constexpr int SCALAR_TILE_COST = 420;
constexpr int SCALAR_BASE_COST = 4000;

// 根据输入宽度选择默认偏置宽度；elemBytes 为输入元素字节数。
constexpr int BiasElemBytes(int elemBytes) { return elemBytes == 2 ? 2 : 4; }

struct ConvTilingInputs {
    int batchSize;                    // 批次数
    int inputChannels;                // 输入通道数
    int inputHeight;                  // 输入高度
    int inputWidth;                   // 输入宽度
    int outputChannels1;              // 第一层输出通道数
    int outputChannels2;              // 第二层输出通道数
    int kernelHeight1;                // 第一层卷积核高
    int kernelWidth1;                 // 第一层卷积核宽
    int stride1;                      // 第一层高宽公共步长
    int stride2;                      // 第二层高宽公共步长
    int paddingHeight1;               // 第一层上下对称补边
    int paddingWidth1;                // 第一层左右对称补边
    int paddingHeight2;               // 第二层上下对称补边
    int paddingWidth2;                // 第二层左右对称补边
    int inputElementBytes;            // 输入元素字节数
    int hasDequantScale2 = 0;         // 第二层是否使用反量化表
    int weightElementBytes = 0;       // 权重元素字节数，零时同输入
    int intermediateElementBytes = 0; // 中间元素字节数，零时同输入
    int biasElementBytes = 0;         // 偏置元素字节数，零时按输入推导
    int hasQuantScale1 = 0;           // 第一层是否使用量化表
    int hasQuantScale2 = 0;           // 第二层是否使用量化表
    int kernelHeight2 = 0;            // 第二层卷积核高，零时同第一层
    int kernelWidth2 = 0;             // 第二层卷积核宽，零时同第一层
};

// 读取权重元素宽度；inputs 为卷积配置，未指定时沿用输入宽度。
constexpr int WeightElemBytes(const ConvTilingInputs& inputs)
{
    return inputs.weightElementBytes == 0 ? inputs.inputElementBytes : inputs.weightElementBytes;
}
// 读取中间元素宽度；inputs 为卷积配置，未指定时沿用输入宽度。
constexpr int MidElemBytes(const ConvTilingInputs& inputs)
{
    return inputs.intermediateElementBytes == 0 ? inputs.inputElementBytes : inputs.intermediateElementBytes;
}
// 读取偏置元素宽度；inputs 为卷积配置，未指定时按输入类型推导。
constexpr int ParamBiasElemBytes(const ConvTilingInputs& inputs)
{
    return inputs.biasElementBytes == 0 ? BiasElemBytes(inputs.inputElementBytes) : inputs.biasElementBytes;
}

// 读取第二层卷积核高；inputs 为卷积配置，零值时沿用第一层。
constexpr int Kh2(const ConvTilingInputs& inputs)
{
    return inputs.kernelHeight2 == 0 ? inputs.kernelHeight1 : inputs.kernelHeight2;
}
// 读取第二层卷积核宽；inputs 为卷积配置，零值时沿用第一层。
constexpr int Kw2(const ConvTilingInputs& inputs)
{
    return inputs.kernelWidth2 == 0 ? inputs.kernelWidth1 : inputs.kernelWidth2;
}

enum Reject {
    RJ_OK = 0,
    RJ_ELEM_BYTES,
    RJ_SHAPE_POSITIVE,
    RJ_CHANNEL_C0,
    RJ_OUT_EMPTY,
    RJ_HB_DIVIDE,
    RJ_MIDROWS,
    RJ_TILE_K,
    RJ_M_TILES,
    RJ_L0C,
    RJ_L0B,
    RJ_BT,
    RJ_L1,
    RJ_U16,
    RJ_NO_HB,
    RJ_WSEG,
};

// 返回分块失败原因；reason 为几何校验返回码。
inline const char* RejectText(int reason)
{
    switch (reason) {
        case RJ_OK:
            return "ok";
        case RJ_ELEM_BYTES:
            return "element width must be 1 (int8) or 2 (fp16)";
        case RJ_SHAPE_POSITIVE:
            return "dimensions must be positive";
        case RJ_CHANNEL_C0:
            return "input and intermediate channels must satisfy C0 alignment";
        case RJ_OUT_EMPTY:
            return "convolution output is empty";
        case RJ_HB_DIVIDE:
            return "band height must divide output height";
        case RJ_MIDROWS:
            return "band has no valid intermediate rows";
        case RJ_TILE_K:
            return "no valid K tile fits L0A";
        case RJ_M_TILES:
            return "M tile count exceeds MAX_M_TILES";
        case RJ_L0C:
            return "M tile exceeds L0C capacity";
        case RJ_L0B:
            return "weight segments exceed L0B capacity";
        case RJ_BT:
            return "bias channels exceed BT slot capacity";
        case RJ_L1:
            return "L1 allocation exceeds available capacity";
        case RJ_U16:
            return "instruction extent exceeds the 16-bit limit";
        case RJ_NO_HB:
            return "no band height satisfies all constraints";
        case RJ_WSEG:
            return "column segment count must be within output width";
        default:
            return "unknown rejection";
    }
}

struct ConvTilingGeometry {
    ConvTilingInputs inputs;        // 两层卷积输入配置
    int inputC0;                    // 输入通道打包粒度
    int weightC0;                   // 权重通道打包粒度
    int intermediateC0;             // 中间通道打包粒度
    int inputWeightGroups;          // 输入通道的权重分组数
    int intermediateWeightGroups;   // 中间通道的权重分组数
    int outputHeight1;              // 第一层输出高
    int outputWidth1;               // 第一层输出宽
    int outputHeight2;              // 第二层输出高
    int outputWidth2;               // 第二层输出宽
    int reductionLength1;           // 第一层完整归约长度
    int reductionLength2;           // 第二层完整归约长度
    int bandHeight;                 // 输出行带高
    int heightBandCount;            // 每图高度方向行带数
    int segmentWidth;               // 输出列段宽度上限
    int widthSegmentCount;          // 每行带列段数
    int intermediateWidthMax;       // 中间窗口宽度上限
    int inputWindowWidthMax;        // 输入窗口宽度上限
    int totalBandTasks;             // 全部批次的行带列段任务数
    int intermediateRows;           // 第二层所需中间行跨度
    int inputWindowRows;            // 输入窗口有效行数上限
    int intermediatePositionsMax;   // 中间窗口位置数上限
    int tileK1;                     // 第一层单次归约长度
    int tileK2;                     // 第二层单次归约长度
    int tileMMax1;                  // 第一层单次 M 上限
    int tileMMax2;                  // 第二层单次 M 上限
    int weightSegmentCount1;        // 第一层权重段数
    int weightSegmentK1;            // 第一层单段归约长度
    int kTilesPerWeightSegment1;    // 第一层每权重段 K 分块数
    int weightSegmentCount2;        // 第二层权重段数
    int weightSegmentK2;            // 第二层单段归约长度
    int kTilesPerWeightSegment2;    // 第二层每权重段 K 分块数
    int l0bPingPong1;               // 第一层 L0B ping/pong 开关
    int l0bPingPong2;               // 第二层 L0B ping/pong 开关
    int l0bSlotElements1;           // 第一层 L0B 槽跨度，单位元素
    int l0bSlotElements2;           // 第二层 L0B 槽跨度，单位元素
    int l0bElements;                // 共享 L0B 元素容量
    int l0cElements;                // 共享 L0C 累加元素容量
    int l0cSlotStride;              // 第一层 L0C 槽跨度，单位元素
    int weightRing1;                // 第一层权重循环搬运开关
    int weightRing2;                // 第二层权重循环搬运开关
    int inputBufferElements;        // L1 输入缓冲元素数
    int intermediateBufferElements; // L1 中间缓冲元素数
    int weightBufferElements1;      // L1 第一层权重元素数
    int weightBufferElements2;      // L1 第二层权重元素数
    int inputBufferAddress;         // L1 输入缓冲字节地址
    int intermediateBufferAddress;  // L1 中间缓冲字节地址
    int weightBufferAddress1;       // L1 第一层权重字节地址
    int weightBufferAddress2;       // L1 第二层权重字节地址
    int biasBufferAddress1;         // L1 第一层偏置字节地址
    int biasBufferAddress2;         // L1 第二层偏置字节地址
    int biasBufferBytes1;           // 第一层偏置分配字节数
    int biasBufferBytes2;           // 第二层偏置分配字节数
    int dequantBufferAddress2;      // L1 第二层反量化表字节地址
    int dequantBufferBytes2;        // 第二层反量化表分配字节数
    int quantBufferAddress1;        // L1 第一层量化表字节地址
    int quantBufferBytes1;          // 第一层量化表分配字节数
    int quantBufferAddress2;        // L1 第二层量化表字节地址
    int quantBufferBytes2;          // 第二层量化表分配字节数
    int l1UsedBytes;                // L1 分配总字节数
    int inputImageElements;         // 每图输入元素数
    int outputImageElements;        // 每图输出元素数
    int filterStorageK1;            // 第一层 FRACTAL_Z 外层 K 维
    int filterStorageN1;            // 第一层 FRACTAL_Z 外层 N 维
    int filterStorageK2;            // 第二层 FRACTAL_Z 外层 K 维
    int filterStorageN2;            // 第二层 FRACTAL_Z 外层 N 维
};

struct ComputeTileChoice {
    int tileK;        // 单次归约长度
    int maxPositions; // 单次 M 上限
    int tileCount;    // M 分块数
    bool valid;       // 是否找到合法分块
};

// 选择 K 步长及 M 上限；reductionLength/kAlignment 为归约长度/粒度，inputBytes/weightBytes 为元素宽度，
// outputChannels 为输出通道数，positions 为待划分的位置数。
inline ComputeTileChoice PickTileK(int reductionLength, int kAlignment, int inputBytes, int outputChannels,
                                   int positions, int weightBytes)
{
    ComputeTileChoice best;
    best.tileK = 0;
    best.maxPositions = 0;
    best.tileCount = 0;
    best.valid = false;
    bool bestPingPong = false;
    if (reductionLength <= 0 || kAlignment <= 0 || outputChannels <= 0 || positions <= 0 || weightBytes <= 0) {
        return best;
    }

    const int accumulatorLimit = L0C_ELEMS / Align(outputChannels, MMAD_M0);

    const int weightKLimit = L0B_BYTES / (Align(outputChannels, MMAD_M0) * weightBytes);
    const int pingPongKLimit = L0B_SLOT_BYTES / (Align(outputChannels, MMAD_M0) * weightBytes);
    for (int tileK = kAlignment; tileK <= reductionLength; tileK += kAlignment) {
        if ((reductionLength % tileK) != 0) {
            continue;
        }
        if (tileK > weightKLimit) {
            continue;
        }

        const int operandLimit = L0A_SLOT_BYTES / (tileK * inputBytes);

        int maxPositions = MinI(operandLimit, accumulatorLimit) / MMAD_M0 * MMAD_M0;
        if (maxPositions <= 0) {
            continue;
        }
        if (maxPositions > positions) {
            maxPositions = positions;
        }
        const int tileCount = CeilDiv(positions, maxPositions);
        if (tileCount > MAX_M_TILES) {
            continue;
        }

        // M 分块数相同时，非整套常驻权重优先保留 L0B ping/pong，再选择较大的 K。
        const bool pingPong = reductionLength > weightKLimit && tileK <= pingPongKLimit;
        if (!best.valid || tileCount < best.tileCount ||
            (tileCount == best.tileCount &&
             ((pingPong && !bestPingPong) || (pingPong == bestPingPong && tileK > best.tileK)))) {
            best.tileK = tileK;
            best.maxPositions = maxPositions;
            best.tileCount = tileCount;
            best.valid = true;
            bestPingPong = pingPong;
        }
    }
    return best;
}

// 选择整分 K 的权重段；outputChannels/reductionLength 为通道/归约长度，weightBytes 为元素宽度，
// tileK 为归约步长，budgetBytes 为容量，segmentCount/segmentK 接收段数/段长。
inline bool PickL0bChunksIn(int outputChannels, int reductionLength, int weightBytes, int tileK, int budgetBytes,
                            int& segmentCount, int& segmentK)
{
    const int alignedChannels = Align(outputChannels, MMAD_M0);
    for (int count = 1; count <= reductionLength / tileK; ++count) {
        if ((reductionLength % count) != 0) {
            continue;
        }
        const int candidateK = reductionLength / count;
        if ((candidateK % tileK) != 0) {
            continue;
        }
        if (alignedChannels * candidateK * weightBytes <= budgetBytes) {
            segmentCount = count;
            segmentK = candidateK;
            return true;
        }
    }
    return false;
}

// 优先选择 L0B ping/pong；outputChannels/reductionLength 为通道/归约长度，weightBytes/tileK 为元素宽度/步长，
// segmentCount/segmentK/pingPong 接收段数/段长/缓冲开关。
inline bool PickL0bChunks(int outputChannels, int reductionLength, int weightBytes, int tileK, int& segmentCount,
                          int& segmentK, int& pingPong)
{
    // 整套权重能放入 L0B 时只搬入一次，无需为 ping/pong 拆段。
    if ((long long)Align(outputChannels, MMAD_M0) * reductionLength * weightBytes <= L0B_BYTES) {
        segmentCount = 1;
        segmentK = reductionLength;
        pingPong = 0;
        return true;
    }
    if (PickL0bChunksIn(outputChannels, reductionLength, weightBytes, tileK, L0B_SLOT_BYTES, segmentCount, segmentK)) {
        pingPong = 1;
        return true;
    }
    pingPong = 0;
    return PickL0bChunksIn(outputChannels, reductionLength, weightBytes, tileK, L0B_BYTES, segmentCount, segmentK);
}

// 写入 kernel 共用参数；geometry 为推导结果，chunksPerCore 为每核任务上限，flat 接收扁平字段。
inline void Flatten(const ConvTilingGeometry& geometry, int chunksPerCore, FlatGeom& flat)
{
    flat.n = geometry.inputs.batchSize;
    flat.ci = geometry.inputs.inputChannels;
    flat.hi = geometry.inputs.inputHeight;
    flat.wi = geometry.inputs.inputWidth;
    flat.cout1 = geometry.inputs.outputChannels1;
    flat.cout2 = geometry.inputs.outputChannels2;
    flat.kh1 = geometry.inputs.kernelHeight1;
    flat.kw1 = geometry.inputs.kernelWidth1;
    flat.kh2 = Kh2(geometry.inputs);
    flat.kw2 = Kw2(geometry.inputs);
    flat.stride1 = geometry.inputs.stride1;
    flat.stride2 = geometry.inputs.stride2;
    flat.padH1 = geometry.inputs.paddingHeight1;
    flat.padW1 = geometry.inputs.paddingWidth1;
    flat.padH2 = geometry.inputs.paddingHeight2;
    flat.padW2 = geometry.inputs.paddingWidth2;
    flat.elemBytes = geometry.inputs.inputElementBytes;
    flat.weightElemBytes = WeightElemBytes(geometry.inputs);
    flat.midElemBytes = MidElemBytes(geometry.inputs);
    flat.biasElemBytes = ParamBiasElemBytes(geometry.inputs);

    flat.c0 = geometry.inputC0;
    flat.weightC0 = geometry.weightC0;
    flat.midC0 = geometry.intermediateC0;
    flat.ho1 = geometry.outputHeight1;
    flat.wo1 = geometry.outputWidth1;
    flat.ho2 = geometry.outputHeight2;
    flat.wo2 = geometry.outputWidth2;
    flat.k1 = geometry.reductionLength1;
    flat.k2 = geometry.reductionLength2;

    flat.hb = geometry.bandHeight;
    flat.nchunk = geometry.heightBandCount;
    flat.wseg = geometry.segmentWidth;
    flat.nwseg = geometry.widthSegmentCount;
    flat.chunkTotal = geometry.totalBandTasks;
    flat.chunksPerCore = chunksPerCore;

    flat.midRows = geometry.intermediateRows;
    flat.mMax1 = geometry.tileMMax1;
    flat.mMax2 = geometry.tileMMax2;
    flat.tileK1 = geometry.tileK1;
    flat.tileK2 = geometry.tileK2;
    flat.l0bChunks1 = geometry.weightSegmentCount1;
    flat.l0bChunkK1 = geometry.weightSegmentK1;
    flat.tilesPerChunk1 = geometry.kTilesPerWeightSegment1;
    flat.l0bChunks2 = geometry.weightSegmentCount2;
    flat.l0bChunkK2 = geometry.weightSegmentK2;
    flat.tilesPerChunk2 = geometry.kTilesPerWeightSegment2;
    flat.l0bPP1 = geometry.l0bPingPong1;
    flat.l0bPP2 = geometry.l0bPingPong2;
    flat.l0bSlot1Elems = geometry.l0bSlotElements1;
    flat.l0bSlot2Elems = geometry.l0bSlotElements2;
    flat.l0bElems = geometry.l0bElements;
    flat.l0cElems = geometry.l0cElements;
    flat.l0cStride = geometry.l0cSlotStride;
    flat.w1Stage = geometry.weightRing1;
    flat.w2Stage = geometry.weightRing2;

    flat.fmAddr = geometry.inputBufferAddress;
    flat.fmElems = geometry.inputBufferElements;
    flat.midAddr = geometry.intermediateBufferAddress;
    flat.midElems = geometry.intermediateBufferElements;
    flat.w1Addr = geometry.weightBufferAddress1;
    flat.w1Elems = geometry.weightBufferElements1;
    flat.w2Addr = geometry.weightBufferAddress2;
    flat.w2Elems = geometry.weightBufferElements2;
    flat.b1Addr = geometry.biasBufferAddress1;
    flat.b2Addr = geometry.biasBufferAddress2;
    flat.dq2Addr = geometry.dequantBufferAddress2;
    flat.q1Addr = geometry.quantBufferAddress1;
    flat.q2Addr = geometry.quantBufferAddress2;

    flat.xPlane = geometry.inputImageElements;
    flat.yPlane = geometry.outputImageElements;
    flat.yPlaneM = geometry.outputHeight2 * geometry.outputWidth2;
}

// 分配 L1 字节地址；inputs 为通道配置，geometry 为缓冲大小及输出地址，
// xBytes/midBytes/weightBytes/biasBytes 分别为输入/中间/权重/偏置元素宽度。
inline void LayOutL1(const ConvTilingInputs& inputs, ConvTilingGeometry& geometry, int xBytes, int midBytes,
                     int weightBytes, int biasBytes)
{
    geometry.inputBufferAddress = 0;
    geometry.intermediateBufferAddress = Align(geometry.inputBufferAddress + geometry.inputBufferElements * xBytes,
                                               L1_SEG_ALIGN);
    geometry.weightBufferAddress1 = Align(
        geometry.intermediateBufferAddress + geometry.intermediateBufferElements * midBytes, L1_SEG_ALIGN);
    geometry.weightBufferAddress2 = Align(geometry.weightBufferAddress1 + geometry.weightBufferElements1 * weightBytes,
                                          L1_SEG_ALIGN);
    geometry.biasBufferAddress1 = Align(geometry.weightBufferAddress2 + geometry.weightBufferElements2 * weightBytes,
                                        L1_SEG_ALIGN);
    geometry.biasBufferBytes1 = Align(inputs.outputChannels1 * biasBytes, L1_SEG_ALIGN);
    geometry.biasBufferAddress2 = geometry.biasBufferAddress1 + geometry.biasBufferBytes1;
    geometry.biasBufferBytes2 = Align(inputs.outputChannels2 * biasBytes, L1_SEG_ALIGN);

    geometry.dequantBufferAddress2 = geometry.biasBufferAddress2 + geometry.biasBufferBytes2;
    geometry.dequantBufferBytes2 = (inputs.hasDequantScale2 != 0) ?
                                       Align(inputs.outputChannels2 * (int)sizeof(long long), L1_SEG_ALIGN) :
                                       0;
    geometry.quantBufferAddress1 = geometry.dequantBufferAddress2 + geometry.dequantBufferBytes2;
    geometry.quantBufferBytes1 = (inputs.hasQuantScale1 != 0) ?
                                     Align(inputs.outputChannels1 * (int)sizeof(long long), L1_SEG_ALIGN) :
                                     0;
    geometry.quantBufferAddress2 = geometry.quantBufferAddress1 + geometry.quantBufferBytes1;
    geometry.quantBufferBytes2 = (inputs.hasQuantScale2 != 0) ?
                                     Align(inputs.outputChannels2 * (int)sizeof(long long), L1_SEG_ALIGN) :
                                     0;
    geometry.l1UsedBytes = geometry.quantBufferAddress2 + geometry.quantBufferBytes2;
}

// 校验分块推导的基础输入；inputs 为两层卷积配置，hb 为候选行带高。
inline int ValidateDeriveInputs(const ConvTilingInputs& inputs, int hb)
{
    const int xBytes = inputs.inputElementBytes;
    const int weightBytes = WeightElemBytes(inputs);
    const int midBytes = MidElemBytes(inputs);
    const int biasBytes = ParamBiasElemBytes(inputs);
    if ((xBytes != 1 && xBytes != 2) || (weightBytes != 1 && weightBytes != 2) || (midBytes != 1 && midBytes != 2) ||
        (biasBytes != 2 && biasBytes != 4)) {
        return RJ_ELEM_BYTES;
    }
    if (inputs.batchSize <= 0 || inputs.inputChannels <= 0 || inputs.inputHeight <= 0 || inputs.inputWidth <= 0 ||
        inputs.outputChannels1 <= 0 || inputs.outputChannels2 <= 0 || inputs.kernelHeight1 <= 0 ||
        inputs.kernelWidth1 <= 0 || Kh2(inputs) <= 0 || Kw2(inputs) <= 0 || inputs.stride1 <= 0 ||
        inputs.stride2 <= 0 || inputs.paddingHeight1 < 0 || inputs.paddingWidth1 < 0 || inputs.paddingHeight2 < 0 ||
        inputs.paddingWidth2 < 0 || hb <= 0) {
        return RJ_SHAPE_POSITIVE;
    }
    return RJ_OK;
}

// 推导通道分组、两层输出和归约长度；inputs 为输入配置，geometry 接收几何结果。
inline int DeriveConvolutionGeometry(const ConvTilingInputs& inputs, ConvTilingGeometry& geometry)
{
    geometry.inputs = inputs;
    geometry.inputC0 = 32 / inputs.inputElementBytes;
    geometry.weightC0 = 32 / WeightElemBytes(inputs);
    geometry.intermediateC0 = 32 / MidElemBytes(inputs);
    if ((inputs.inputChannels % geometry.inputC0) != 0 || (inputs.inputChannels % geometry.weightC0) != 0 ||
        (inputs.outputChannels1 % geometry.intermediateC0) != 0 || (inputs.outputChannels1 % geometry.weightC0) != 0) {
        return RJ_CHANNEL_C0;
    }
    geometry.inputWeightGroups = inputs.inputChannels / geometry.weightC0;
    geometry.intermediateWeightGroups = inputs.outputChannels1 / geometry.weightC0;
    geometry.outputHeight1 = (inputs.inputHeight + 2 * inputs.paddingHeight1 - inputs.kernelHeight1) / inputs.stride1 +
                             1;
    geometry.outputWidth1 = (inputs.inputWidth + 2 * inputs.paddingWidth1 - inputs.kernelWidth1) / inputs.stride1 + 1;
    if (geometry.outputHeight1 <= 0 || geometry.outputWidth1 <= 0) {
        return RJ_OUT_EMPTY;
    }
    geometry.outputHeight2 = (geometry.outputHeight1 + 2 * inputs.paddingHeight2 - Kh2(inputs)) / inputs.stride2 + 1;
    geometry.outputWidth2 = (geometry.outputWidth1 + 2 * inputs.paddingWidth2 - Kw2(inputs)) / inputs.stride2 + 1;
    if (geometry.outputHeight2 <= 0 || geometry.outputWidth2 <= 0) {
        return RJ_OUT_EMPTY;
    }
    geometry.reductionLength1 = inputs.inputChannels * inputs.kernelHeight1 * inputs.kernelWidth1;
    geometry.reductionLength2 = inputs.outputChannels1 * Kh2(inputs) * Kw2(inputs);
    geometry.filterStorageK1 = geometry.inputWeightGroups * inputs.kernelHeight1 * inputs.kernelWidth1;
    geometry.filterStorageN1 = CeilDiv(inputs.outputChannels1, MMAD_M0);
    geometry.filterStorageK2 = geometry.intermediateWeightGroups * Kh2(inputs) * Kw2(inputs);
    geometry.filterStorageN2 = CeilDiv(inputs.outputChannels2, MMAD_M0);
    return RJ_OK;
}

// 反推输出行带所需的中间及输入窗口；hb/nwseg 为候选划分，geometry 为已知卷积几何。
inline int DeriveWindowGeometry(const ConvTilingInputs& inputs, int hb, int nwseg, ConvTilingGeometry& geometry)
{
    if ((geometry.outputHeight2 % hb) != 0) {
        return RJ_HB_DIVIDE;
    }
    if (nwseg <= 0 || nwseg > geometry.outputWidth2) {
        return RJ_WSEG;
    }
    geometry.bandHeight = hb;
    geometry.heightBandCount = geometry.outputHeight2 / hb;
    geometry.segmentWidth = CeilDiv(geometry.outputWidth2, nwseg);
    geometry.widthSegmentCount = CeilDiv(geometry.outputWidth2, geometry.segmentWidth);
    geometry.totalBandTasks = inputs.batchSize * geometry.heightBandCount * geometry.widthSegmentCount;
    // 连续输出的覆盖范围为步长乘间隔数，再加本层卷积核跨度。
    geometry.intermediateRows = inputs.stride2 * (hb - 1) + Kh2(inputs);
    const int midRowsCap = MinI(geometry.intermediateRows, geometry.outputHeight1);
    geometry.intermediateWidthMax = MinI(geometry.outputWidth1,
                                         (geometry.segmentWidth - 1) * inputs.stride2 + Kw2(inputs));
    geometry.inputWindowWidthMax = MinI(inputs.inputWidth,
                                        (geometry.intermediateWidthMax - 1) * inputs.stride1 + inputs.kernelWidth1);
    geometry.intermediatePositionsMax = midRowsCap * geometry.intermediateWidthMax;
    geometry.inputWindowRows = MinI(inputs.stride1 * (midRowsCap - 1) + inputs.kernelHeight1, inputs.inputHeight);
    if (geometry.intermediatePositionsMax > 65535 ||
        (long long)geometry.inputWindowRows * geometry.inputWindowWidthMax > 65535) {
        return RJ_U16;
    }
    return RJ_OK;
}

// 选择两层的 K 步长和 M 上限；inputs/geometry 为输入及窗口几何，mRunMax 为第二层单次 M 范围。
inline int SelectComputeTiles(const ConvTilingInputs& inputs, int mRunMax, ConvTilingGeometry& geometry)
{
    const int weightBytes = WeightElemBytes(inputs);
    const int kBlock1 = MaxI(geometry.inputC0, geometry.weightC0);
    const int kBlock2 = MaxI(geometry.intermediateC0, geometry.weightC0);
    if (Align(inputs.outputChannels1, MMAD_M0) * kBlock1 * weightBytes > L0B_BYTES ||
        Align(inputs.outputChannels2, MMAD_M0) * kBlock2 * weightBytes > L0B_BYTES) {
        return RJ_L0B;
    }
    const ComputeTileChoice tileChoice1 = PickTileK(geometry.reductionLength1, kBlock1, inputs.inputElementBytes,
                                                    inputs.outputChannels1, geometry.intermediatePositionsMax,
                                                    weightBytes);
    const ComputeTileChoice tileChoice2 = PickTileK(geometry.reductionLength2, kBlock2, MidElemBytes(inputs),
                                                    inputs.outputChannels2, mRunMax, weightBytes);
    if (!tileChoice1.valid || !tileChoice2.valid) {
        return RJ_TILE_K;
    }
    geometry.tileK1 = tileChoice1.tileK;
    geometry.tileMMax1 = tileChoice1.maxPositions;
    geometry.tileK2 = tileChoice2.tileK;
    geometry.tileMMax2 = tileChoice2.maxPositions;
    return RJ_OK;
}

// 配置权重段和 L0B 复用；inputs/geometry 为卷积配置及分块结果。
inline int SelectWeightSegments(const ConvTilingInputs& inputs, ConvTilingGeometry& geometry)
{
    const int weightBytes = WeightElemBytes(inputs);
    if (!PickL0bChunks(inputs.outputChannels1, geometry.reductionLength1, weightBytes, geometry.tileK1,
                       geometry.weightSegmentCount1, geometry.weightSegmentK1, geometry.l0bPingPong1) ||
        !PickL0bChunks(inputs.outputChannels2, geometry.reductionLength2, weightBytes, geometry.tileK2,
                       geometry.weightSegmentCount2, geometry.weightSegmentK2, geometry.l0bPingPong2)) {
        return RJ_L0B;
    }
    if (geometry.weightSegmentCount1 == 1) {
        geometry.l0bPingPong1 = 0;
    }
    if (geometry.weightSegmentCount2 == 1) {
        geometry.l0bPingPong2 = 0;
    }
    geometry.kTilesPerWeightSegment1 = geometry.weightSegmentK1 / geometry.tileK1;
    geometry.kTilesPerWeightSegment2 = geometry.weightSegmentK2 / geometry.tileK2;
    return RJ_OK;
}

// 检查边界行带的有效 M 划分；inputs/geometry 为卷积配置及候选几何，mRunMax 为第二层 M 范围。
inline int ValidateBandTiles(const ConvTilingInputs& inputs, int mRunMax, const ConvTilingGeometry& geometry)
{
    for (int bandIndex = 0; bandIndex < geometry.heightBandCount; ++bandIndex) {
        const int rawMidRow = inputs.stride2 * geometry.bandHeight * bandIndex - inputs.paddingHeight2;
        const int midRowBegin = MaxI(0, rawMidRow);
        const int topPadding2 = midRowBegin - rawMidRow;
        const int validMidRows = MinI(geometry.intermediateRows - topPadding2, geometry.outputHeight1 - midRowBegin);
        if (validMidRows <= 0) {
            return RJ_MIDROWS;
        }
        if (MakeTileSpec(validMidRows * geometry.intermediateWidthMax, geometry.tileMMax1, 1).n == 0) {
            return RJ_M_TILES;
        }
    }
    return MakeTileSpec(mRunMax, geometry.tileMMax2, 1).n == 0 ? RJ_M_TILES : RJ_OK;
}

// 检查 L0A/L0B 占用并记录缓冲跨度；inputs 为元素宽度配置，geometry 接收容量结果。
inline int ConfigureOperandBuffers(const ConvTilingInputs& inputs, ConvTilingGeometry& geometry)
{
    const int l0a1 = Align(geometry.tileMMax1, MMAD_M0) * geometry.tileK1 * inputs.inputElementBytes;
    const int l0a2 = Align(geometry.tileMMax2, MMAD_M0) * geometry.tileK2 * MidElemBytes(inputs);
    if (l0a1 > L0A_SLOT_BYTES || l0a2 > L0A_SLOT_BYTES) {
        return RJ_TILE_K;
    }
    const int weightBytes = WeightElemBytes(inputs);
    const int l0b1 = Align(inputs.outputChannels1, MMAD_M0) * geometry.weightSegmentK1 * weightBytes;
    const int l0b2 = Align(inputs.outputChannels2, MMAD_M0) * geometry.weightSegmentK2 * weightBytes;
    const int usedBytes1 = l0b1 * (geometry.l0bPingPong1 != 0 ? 2 : 1);
    const int usedBytes2 = l0b2 * (geometry.l0bPingPong2 != 0 ? 2 : 1);
    const int l0bMax = MaxI(usedBytes1, usedBytes2);
    if (l0bMax > L0B_BYTES) {
        return RJ_L0B;
    }
    geometry.l0bElements = l0bMax / weightBytes;
    geometry.l0bSlotElements1 = (geometry.l0bPingPong1 != 0) ? l0b1 / weightBytes : 0;
    geometry.l0bSlotElements2 = (geometry.l0bPingPong2 != 0) ? l0b2 / weightBytes : 0;
    return RJ_OK;
}

// 配置两层共用的 L0C 和固定分层 BT；inputs/geometry 为通道及 M 分块配置。
inline int ConfigureAccumulatorBuffers(const ConvTilingInputs& inputs, ConvTilingGeometry& geometry)
{
    const int l0c1 = Align(geometry.tileMMax1, MMAD_M0) * Align(inputs.outputChannels1, MMAD_M0);
    const int l0c2 = Align(geometry.tileMMax2, MMAD_M0) * Align(inputs.outputChannels2, MMAD_M0);
    if (MaxI(l0c1, l0c2) > L0C_ELEMS) {
        return RJ_L0C;
    }
    // 第一层能容纳两份累加结果时启用 ping/pong，第二层仍复用 L0C 起始地址。
    geometry.l0cSlotStride = (2 * l0c1 <= L0C_ELEMS) ? l0c1 : 0;
    geometry.l0cElements = MaxI(geometry.l0cSlotStride + l0c1, l0c2);
    return inputs.outputChannels1 > BT_SLOT_ELEMS || inputs.outputChannels2 > BT_SLOT_ELEMS ? RJ_BT : RJ_OK;
}

// 按驻留优先级选择 L1 权重布局；inputs/geometry 为配置及容量，l1Budget 为可用字节数。
inline int SelectL1WeightLayout(const ConvTilingInputs& inputs, int l1Budget, ConvTilingGeometry& geometry)
{
    geometry.inputBufferElements = geometry.inputWindowRows * geometry.inputWindowWidthMax * inputs.inputChannels;
    geometry.intermediateBufferElements = geometry.intermediatePositionsMax * inputs.outputChannels1;
    const int w1Full = Align(inputs.outputChannels1, MMAD_M0) * geometry.reductionLength1;
    const int w2Full = Align(inputs.outputChannels2, MMAD_M0) * geometry.reductionLength2;
    const int w1Ring = 2 * Align(inputs.outputChannels1, MMAD_M0) * geometry.weightSegmentK1;
    const int w2Ring = 2 * Align(inputs.outputChannels2, MMAD_M0) * geometry.weightSegmentK2;
    const bool w1CanStage = (geometry.weightSegmentCount1 > 1) && (w1Ring < w1Full);
    const bool w2CanStage = (geometry.weightSegmentCount2 > 1) && (w2Ring < w2Full);
    const bool w2Bigger = (w2Full - w2Ring) >= (w1Full - w1Ring);
    const int modes[4] = {0, w2Bigger ? 2 : 1, w2Bigger ? 1 : 2, 3};
    const int budget = (l1Budget > 0) ? l1Budget : 0x7fffffff;
    // 优先两套权重常驻；容量不足时先循环搬运更能节省空间的一层。
    for (int modeIndex = 0; modeIndex < 4; ++modeIndex) {
        const int mode = modes[modeIndex];
        const int stage1 = (mode & 1) != 0 ? 1 : 0;
        const int stage2 = (mode & 2) != 0 ? 1 : 0;
        if ((stage1 != 0 && !w1CanStage) || (stage2 != 0 && !w2CanStage)) {
            continue;
        }
        geometry.weightRing1 = stage1;
        geometry.weightRing2 = stage2;
        geometry.weightBufferElements1 = (stage1 != 0) ? w1Ring : w1Full;
        geometry.weightBufferElements2 = (stage2 != 0) ? w2Ring : w2Full;
        LayOutL1(inputs, geometry, inputs.inputElementBytes, MidElemBytes(inputs), WeightElemBytes(inputs),
                 ParamBiasElemBytes(inputs));
        if (geometry.l1UsedBytes <= budget) {
            return RJ_OK;
        }
    }
    return RJ_L1;
}

// 推导一组完整候选；inputs 为卷积配置，hb/nwseg 为划分，l1Budget 为预算，geometry 接收结果。
inline int DeriveWith(const ConvTilingInputs& inputs, int hb, int nwseg, int l1Budget, ConvTilingGeometry& geometry)
{
    int status = ValidateDeriveInputs(inputs, hb);
    if (status != RJ_OK) {
        return status;
    }
    status = DeriveConvolutionGeometry(inputs, geometry);
    if (status != RJ_OK) {
        return status;
    }
    status = DeriveWindowGeometry(inputs, hb, nwseg, geometry);
    if (status != RJ_OK) {
        return status;
    }
    const int mRunMax = (geometry.widthSegmentCount == 1) ? (hb * geometry.outputWidth2) : geometry.segmentWidth;
    if ((status = SelectComputeTiles(inputs, mRunMax, geometry)) != RJ_OK ||
        (status = SelectWeightSegments(inputs, geometry)) != RJ_OK ||
        (status = ValidateBandTiles(inputs, mRunMax, geometry)) != RJ_OK ||
        (status = ConfigureOperandBuffers(inputs, geometry)) != RJ_OK ||
        (status = ConfigureAccumulatorBuffers(inputs, geometry)) != RJ_OK ||
        (status = SelectL1WeightLayout(inputs, l1Budget, geometry)) != RJ_OK) {
        return status;
    }
    geometry.inputImageElements = inputs.inputChannels * inputs.inputHeight * inputs.inputWidth;
    geometry.outputImageElements = inputs.outputChannels2 * geometry.outputHeight2 * geometry.outputWidth2;
    return RJ_OK;
}

struct BandCost {
    long long cube = 0;   // 矩阵计算代价
    long long mte1 = 0;   // L1 到 L0 搬运代价
    long long mte2 = 0;   // GM 到 L1 搬运代价
    long long fix = 0;    // 累加结果输出代价
    long long scalar = 0; // 分块调度代价
};

// 估算一个行带的矩阵计算代价；geometry 为候选几何，band 为裁剪后的执行窗口。
inline long long EstimateBandCubeCost(const ConvTilingGeometry& geometry, const BandLite& band)
{
    const int cout1A = Align(geometry.inputs.outputChannels1, MMAD_M0);
    const int cout2A = Align(geometry.inputs.outputChannels2, MMAD_M0);
    const int cubeKFactor1 = (geometry.inputs.inputElementBytes == 2) ? 16 : 32;
    const int cubeKFactor2 = (MidElemBytes(geometry.inputs) == 2) ? 16 : 32;
    long long cost = 0;
    for (int tileIndex = 0; tileIndex < band.conv1Tile.n; ++tileIndex) {
        cost += (long long)(Align(TileRows(band.conv1Tile, tileIndex), MMAD_M0) / MMAD_M0) * (cout1A / MMAD_M0) *
                (geometry.reductionLength1 / cubeKFactor1);
    }
    for (int runIndex = 0; runIndex < band.nRun; ++runIndex) {
        for (int tileIndex = 0; tileIndex < band.conv2Tile.n; ++tileIndex) {
            cost += (long long)(Align(TileRows(band.conv2Tile, tileIndex), MMAD_M0) / MMAD_M0) * (cout2A / MMAD_M0) *
                    (geometry.reductionLength2 / cubeKFactor2);
        }
    }
    return cost;
}

// 估算操作数搬运代价；geometry 为候选几何，band 为本行带的有效 M 范围。
inline long long EstimateBandMte1Cost(const ConvTilingGeometry& geometry, const BandLite& band)
{
    const int cout1A = Align(geometry.inputs.outputChannels1, MMAD_M0);
    const int cout2A = Align(geometry.inputs.outputChannels2, MMAD_M0);
    const int weightBytes = WeightElemBytes(geometry.inputs);
    long long operandBytes = (long long)band.conv1MPerBand * geometry.reductionLength1 *
                                 geometry.inputs.inputElementBytes +
                             (long long)band.nRun * band.mRun * geometry.reductionLength2 *
                                 MidElemBytes(geometry.inputs);
    operandBytes += (geometry.weightSegmentCount1 == 1) ?
                        (long long)cout1A * geometry.reductionLength1 * weightBytes :
                        (long long)band.conv1Tile.n * cout1A * geometry.reductionLength1 * weightBytes;
    operandBytes += (geometry.weightSegmentCount2 == 1) ?
                        (long long)cout2A * geometry.reductionLength2 * weightBytes :
                        (long long)band.nRun * band.conv2Tile.n * cout2A * geometry.reductionLength2 * weightBytes;
    return operandBytes / MTE1_COST_FACTOR;
}

// 估算输入及循环权重搬运代价；geometry 为驻留配置，band 为有效输入窗口。
inline long long EstimateBandMte2Cost(const ConvTilingGeometry& geometry, const BandLite& band)
{
    const long long inputBytes = (long long)band.xRows * band.wIn * geometry.inputs.inputChannels *
                                 geometry.inputs.inputElementBytes;
    long long cost = inputBytes / MTE2_COST_FACTOR;
    if (band.wIn != geometry.inputs.inputWidth) {
        cost += (long long)band.xRows * geometry.inputs.inputChannels * MTE2_ROW_COST;
    }
    long long ringWeightBytes = 0;
    if (geometry.weightRing1 != 0) {
        ringWeightBytes += (long long)band.conv1Tile.n * Align(geometry.inputs.outputChannels1, MMAD_M0) *
                           geometry.reductionLength1 * WeightElemBytes(geometry.inputs);
    }
    if (geometry.weightRing2 != 0) {
        ringWeightBytes += (long long)band.nRun * band.conv2Tile.n * Align(geometry.inputs.outputChannels2, MMAD_M0) *
                           geometry.reductionLength2 * WeightElemBytes(geometry.inputs);
    }
    return cost + ringWeightBytes / MTE2_COST_FACTOR;
}

// 估算中间与最终结果写回代价；geometry 为数据通路配置，band 为本行带分块。
inline long long EstimateBandFixCost(const ConvTilingGeometry& geometry, const BandLite& band)
{
    const int cout1A = Align(geometry.inputs.outputChannels1, MMAD_M0);
    const int cout2A = Align(geometry.inputs.outputChannels2, MMAD_M0);
    const int outputBytes = geometry.inputs.hasDequantScale2 ? 2 : MidElemBytes(geometry.inputs);
    long long cost = (long long)band.conv1MPerBand * cout1A * 4 / FIX_L1_COST_FACTOR;
    const long long outputRequests = (long long)band.nRun * band.conv2Tile.n * cout2A;
    const long long outputSize = (long long)band.nRun * band.mRun * cout2A * outputBytes;
    cost += outputRequests * FIX_GM_REQUEST_COST_X2 / 2 + outputSize / FIX_GM_COST_FACTOR;
    return cost;
}

// 估算最忙核心的候选代价；geometry 为候选几何，chunksPerCore 为每核最多处理的行带数。
inline long long CoreCostCycles(const ConvTilingGeometry& geometry, int chunksPerCore)
{
    FlatGeom flat;
    Flatten(geometry, chunksPerCore, flat);
    const int bands[3] = {0, geometry.heightBandCount / 2, geometry.heightBandCount - 1};
    const int segments[3] = {0, geometry.widthSegmentCount / 2, geometry.widthSegmentCount - 1};
    BandCost worst;
    // 对首、中、尾窗口分别取各流水代价上界，避免边界裁剪低估最忙核心。
    for (int bandIndex : bands) {
        for (int segmentIndex : segments) {
            if (bandIndex < 0 || segmentIndex < 0) {
                continue;
            }
            BandLite band;
            MakeBandLite(flat, bandIndex, segmentIndex, band);
            const long long cube = EstimateBandCubeCost(geometry, band);
            const long long mte1 = EstimateBandMte1Cost(geometry, band);
            const long long mte2 = EstimateBandMte2Cost(geometry, band);
            const long long fix = EstimateBandFixCost(geometry, band);
            const long long scalar = (long long)(band.conv1Tile.n + band.nRun * band.conv2Tile.n) * SCALAR_TILE_COST;
            worst.cube = cube > worst.cube ? cube : worst.cube;
            worst.mte1 = mte1 > worst.mte1 ? mte1 : worst.mte1;
            worst.mte2 = mte2 > worst.mte2 ? mte2 : worst.mte2;
            worst.fix = fix > worst.fix ? fix : worst.fix;
            worst.scalar = scalar > worst.scalar ? scalar : worst.scalar;
        }
    }
    long long residentWeightBytes = 0;
    if (geometry.weightRing1 == 0) {
        residentWeightBytes += (long long)Align(geometry.inputs.outputChannels1, MMAD_M0) * geometry.reductionLength1 *
                               WeightElemBytes(geometry.inputs);
    }
    if (geometry.weightRing2 == 0) {
        residentWeightBytes += (long long)Align(geometry.inputs.outputChannels2, MMAD_M0) * geometry.reductionLength2 *
                               WeightElemBytes(geometry.inputs);
    }
    return (long long)chunksPerCore * (worst.cube + worst.mte1 + worst.mte2 + worst.fix + worst.scalar) +
           residentWeightBytes / MTE2_COST_FACTOR + SCALAR_BASE_COST;
}

struct RowBandChoice {
    int height = 0;             // 最佳完整行带高
    long long cost = 0;         // 行覆盖代价
    int stagedLayers = 3;       // 循环搬运权重的层数
    int firstReject = RJ_NO_HB; // 首个有效候选失败原因
};

struct SearchCostBounds {
    long long fullWidth[4] = {-1, -1, -1, -1};  // 各驻留等级的完整行带最小代价
    long long splitWidth[4] = {-1, -1, -1, -1}; // 各驻留等级的切列最小代价
};

struct SearchPreference {
    int stagedLayers = -1;   // 允许的权重循环层数
    bool allowSplit = false; // 是否允许切列
    long long cutoff = 0;    // 候选代价上界
};

struct SplitBandChoice {
    int height = 0;        // 最佳行带高
    int segments = 1;      // 最佳列段数
    int chunksPerCore = 0; // 每核最多行带数
};

// 校验搜索输入并得到最终输出高宽；inputs 为卷积配置，outputHeight/outputWidth 接收尺寸。
inline int GetSearchOutputShape(const ConvTilingInputs& inputs, int& outputHeight, int& outputWidth)
{
    if (inputs.inputElementBytes != 1 && inputs.inputElementBytes != 2) {
        return RJ_ELEM_BYTES;
    }
    if (inputs.batchSize <= 0 || inputs.inputChannels <= 0 || inputs.inputHeight <= 0 || inputs.inputWidth <= 0 ||
        inputs.outputChannels1 <= 0 || inputs.outputChannels2 <= 0 || inputs.kernelHeight1 <= 0 ||
        inputs.kernelWidth1 <= 0 || Kh2(inputs) <= 0 || Kw2(inputs) <= 0 || inputs.stride1 <= 0 ||
        inputs.stride2 <= 0 || inputs.paddingHeight1 < 0 || inputs.paddingWidth1 < 0 || inputs.paddingHeight2 < 0 ||
        inputs.paddingWidth2 < 0) {
        return RJ_SHAPE_POSITIVE;
    }
    const int midHeight = (inputs.inputHeight + 2 * inputs.paddingHeight1 - inputs.kernelHeight1) / inputs.stride1 + 1;
    if (midHeight <= 0) {
        return RJ_OUT_EMPTY;
    }
    outputHeight = (midHeight + 2 * inputs.paddingHeight2 - Kh2(inputs)) / inputs.stride2 + 1;
    if (outputHeight <= 0) {
        return RJ_OUT_EMPTY;
    }
    const int midWidth = (inputs.inputWidth + 2 * inputs.paddingWidth1 - inputs.kernelWidth1) / inputs.stride1 + 1;
    if (midWidth <= 0) {
        return RJ_OUT_EMPTY;
    }
    outputWidth = (midWidth + 2 * inputs.paddingWidth2 - Kw2(inputs)) / inputs.stride2 + 1;
    return outputWidth <= 0 ? RJ_OUT_EMPTY : RJ_OK;
}

// 搜索完整行带；inputs 为输入，outputHeight/aicNum/l1Budget 为搜索约束，返回最优行带和失败原因。
inline RowBandChoice SearchFullWidthBands(const ConvTilingInputs& inputs, int outputHeight, int aicNum, int l1Budget)
{
    RowBandChoice best;
    ConvTilingGeometry probe;
    for (int hb = 1; hb <= outputHeight; ++hb) {
        if ((outputHeight % hb) != 0) {
            continue;
        }
        const int status = DeriveWith(inputs, hb, 1, l1Budget, probe);
        if (status != RJ_OK) {
            if (best.firstReject == RJ_NO_HB && status != RJ_HB_DIVIDE) {
                best.firstReject = status;
            }
            continue;
        }
        const long long rounds = CeilDiv(probe.totalBandTasks, aicNum);
        const long long cost = rounds * (long long)probe.intermediateRows;
        const int stagedLayers = probe.weightRing1 + probe.weightRing2;
        if (best.height == 0 || stagedLayers < best.stagedLayers ||
            (stagedLayers == best.stagedLayers && (cost < best.cost || (cost == best.cost && hb > best.height)))) {
            best.height = hb;
            best.cost = cost;
            best.stagedLayers = stagedLayers;
        }
    }
    return best;
}

// 汇总两种列布局的最小代价；inputs 为卷积配置，outputHeight/maxSegments 为搜索范围，
// aicNum/l1Budget 为资源，firstReject 记录首个失败。
inline SearchCostBounds CollectSearchCosts(const ConvTilingInputs& inputs, int outputHeight, int maxSegments,
                                           int aicNum, int l1Budget, int& firstReject)
{
    SearchCostBounds bounds;
    ConvTilingGeometry probe;
    for (int hb = 1; hb <= outputHeight; ++hb) {
        if ((outputHeight % hb) != 0) {
            continue;
        }
        for (int nw = 1; nw <= maxSegments; ++nw) {
            const int status = DeriveWith(inputs, hb, nw, l1Budget, probe);
            if (status != RJ_OK) {
                if (firstReject == RJ_NO_HB && status != RJ_HB_DIVIDE && status != RJ_WSEG) {
                    firstReject = status;
                }
                continue;
            }
            const int activeCores = MinI(aicNum, probe.totalBandTasks);
            const int chunksPerCore = CeilDiv(probe.totalBandTasks, activeCores);
            const long long cost = CoreCostCycles(probe, chunksPerCore);
            const int stagedLayers = probe.weightRing1 + probe.weightRing2;
            long long& minimum = nw == 1 ? bounds.fullWidth[stagedLayers] : bounds.splitWidth[stagedLayers];
            if (minimum < 0 || cost < minimum) {
                minimum = cost;
            }
        }
    }
    return bounds;
}

// 确定可接受的代价区间；bounds 为各驻留等级的最优代价，返回切列许可及阈值。
inline SearchPreference ChooseSearchPreference(const SearchCostBounds& bounds)
{
    SearchPreference preference;
    for (int stage = 0; stage < 4; ++stage) {
        if (bounds.fullWidth[stage] >= 0 || bounds.splitWidth[stage] >= 0) {
            preference.stagedLayers = stage;
            break;
        }
    }
    if (preference.stagedLayers < 0) {
        return preference;
    }
    const long long fullBest = bounds.fullWidth[preference.stagedLayers];
    const long long splitBest = bounds.splitWidth[preference.stagedLayers];
    // 切列至少降低 20% 代价才启用；随后允许 2% 区间内按任务均衡和较少切列择优。
    preference.allowSplit = splitBest >= 0 && (fullBest < 0 || splitBest * 5 <= fullBest * 4);
    long long minimumCost = fullBest >= 0 ? fullBest : -1;
    if (preference.allowSplit && (minimumCost < 0 || splitBest < minimumCost)) {
        minimumCost = splitBest;
    }
    preference.cutoff = minimumCost + minimumCost / 50;
    return preference;
}

// 在代价区间内择优；inputs 为卷积配置，outputHeight/maxSegments 为搜索范围，
// aicNum/l1Budget 为资源，preference 为代价及驻留筛选规则。
inline SplitBandChoice SearchSplitBands(const ConvTilingInputs& inputs, int outputHeight, int maxSegments, int aicNum,
                                        int l1Budget, const SearchPreference& preference)
{
    SplitBandChoice best;
    ConvTilingGeometry probe;
    for (int hb = 1; hb <= outputHeight; ++hb) {
        if ((outputHeight % hb) != 0) {
            continue;
        }
        for (int nw = 1; nw <= maxSegments; ++nw) {
            if (DeriveWith(inputs, hb, nw, l1Budget, probe) != RJ_OK ||
                probe.weightRing1 + probe.weightRing2 != preference.stagedLayers ||
                (nw > 1 && !preference.allowSplit)) {
                continue;
            }
            const int chunksPerCore = CeilDiv(probe.totalBandTasks, MinI(aicNum, probe.totalBandTasks));
            if (CoreCostCycles(probe, chunksPerCore) > preference.cutoff) {
                continue;
            }
            if (best.height == 0 || chunksPerCore < best.chunksPerCore ||
                (chunksPerCore == best.chunksPerCore &&
                 (nw < best.segments || (nw == best.segments && hb > best.height)))) {
                best.height = hb;
                best.segments = nw;
                best.chunksPerCore = chunksPerCore;
            }
        }
    }
    return best;
}

// 搜索行带高和列段数；inputs 为输入，aicNum/l1Budget 为资源，geometry 接收最终可执行几何。
inline int PickHb(const ConvTilingInputs& inputs, int aicNum, int l1Budget, ConvTilingGeometry& geometry)
{
    if (aicNum <= 0) {
        aicNum = 1;
    }
    int outputHeight = 0, outputWidth = 0;
    const int status = GetSearchOutputShape(inputs, outputHeight, outputWidth);
    if (status != RJ_OK) {
        return status;
    }
    RowBandChoice rowChoice = SearchFullWidthBands(inputs, outputHeight, aicNum, l1Budget);
    if (rowChoice.height != 0) {
        const long long fullWidthTasks = (long long)inputs.batchSize * (outputHeight / rowChoice.height);
        if (fullWidthTasks >= (long long)aicNum && fullWidthTasks % (long long)aicNum == 0) {
            return DeriveWith(inputs, rowChoice.height, 1, l1Budget, geometry);
        }
    }
    const int maxSegments = MinI(aicNum, outputWidth);
    const auto bounds = CollectSearchCosts(inputs, outputHeight, maxSegments, aicNum, l1Budget, rowChoice.firstReject);
    const auto preference = ChooseSearchPreference(bounds);
    const auto splitChoice = preference.stagedLayers >= 0 ?
                                 SearchSplitBands(inputs, outputHeight, maxSegments, aicNum, l1Budget, preference) :
                                 SplitBandChoice{};
    if (splitChoice.height != 0) {
        return DeriveWith(inputs, splitChoice.height, splitChoice.segments, l1Budget, geometry);
    }
    return rowChoice.height == 0 ? rowChoice.firstReject : DeriveWith(inputs, rowChoice.height, 1, l1Budget, geometry);
}

} // namespace FusedMultiConv2dShape

namespace {
namespace S = FusedMultiConv2dShape;

using namespace FusedMultiConv2dAttrs;

constexpr int64_t RELU_NO = 0;
constexpr int64_t RELU_NORMAL = 1;
constexpr int64_t RELU_SCALAR = 2;
constexpr int64_t RELU_VECTOR = 3;

constexpr size_t IR_X = 0;
constexpr size_t IR_FILTERS = 1;
constexpr size_t IR_BIAS = 2;
constexpr size_t IR_SCALE = 3;
constexpr size_t IR_RELU_WEIGHT = 4;

constexpr size_t NUM_CONVS = 2;

// 检查动态列表恰好包含两层；name 为输入名，got 为实例数，nodeName 用于日志。
inline bool ExpectedInstanceNum(const char* name, size_t got, const char* nodeName)
{
    if (got == NUM_CONVS) {
        return true;
    }
    OP_LOGE(nodeName, "fused_multi_conv2d: %s requires exactly %zu dynamic instances, got %zu", name, NUM_CONVS, got);
    return false;
}

// 读取整型属性；attrs 为列表，attrIndex 为索引，defaultValue 为缺省值。
inline int64_t ReadIntAttr(const gert::RuntimeAttrs* attrs, size_t attrIndex, int64_t defaultValue)
{
    if (attrs == nullptr || attrs->GetAttrNum() <= attrIndex) {
        return defaultValue;
    }
    const int64_t* value = attrs->GetInt(attrIndex);
    return (value == nullptr) ? defaultValue : *value;
}

enum class AttributeValueKind { kInt, kListInt, kListBool };
// 检查必填属性；attrs 为属性列表，nodeName 用于日志。
inline bool RequiredAttrsPresent(const gert::RuntimeAttrs* attrs, const char* nodeName)
{
    struct RequiredAttributeSpec {
        size_t idx;              // 属性索引
        const char* name;        // 属性名称
        AttributeValueKind kind; // 属性数据类型
    };
    static const RequiredAttributeSpec REQUIRED_ATTRIBUTES[] = {

        {ATTR_BIAS_MASK, "bias_mask", AttributeValueKind::kListBool},
    };
    for (const RequiredAttributeSpec& requirement : REQUIRED_ATTRIBUTES) {
        bool missing = (attrs == nullptr) || (attrs->GetAttrNum() <= requirement.idx);
        if (!missing) {
            switch (requirement.kind) {
                case AttributeValueKind::kInt:
                    missing = (attrs->GetInt(requirement.idx) == nullptr);
                    break;
                case AttributeValueKind::kListInt:
                    missing = (attrs->GetListInt(requirement.idx) == nullptr);
                    break;
                case AttributeValueKind::kListBool:
                    missing = (attrs->GetAttrPointer<gert::ContinuousVector>(requirement.idx) == nullptr);
                    break;
                default:
                    missing = true;
                    break;
            }
        }
        if (missing) {
            OP_LOGE(nodeName, "fused_multi_conv2d: required attribute %s is missing (count=%zu index=%zu)",
                    requirement.name, attrs == nullptr ? static_cast<size_t>(0) : attrs->GetAttrNum(), requirement.idx);
            return false;
        }
    }
    return true;
}

// 校验公开、编译器和私有槽位布局；attrs 为属性列表，nodeName 用于日志。
inline bool CheckPrivateAttrsLayout(const gert::RuntimeAttrs* attrs, const char* nodeName)
{
    const size_t count = attrs == nullptr ? 0 : attrs->GetAttrNum();

    if (count >= RUNTIME_ATTR_COUNT) {
        return true;
    }
    OP_LOGE(nodeName,
            "fused_multi_conv2d: invalid RuntimeAttrs layout (%zu attrs): expected %zu public + "
            "1 compiler-injected ascendc_op_para_size + 1 private fixed_shift_value (at least %zu total)",
            count, static_cast<size_t>(PUBLIC_ATTR_COUNT), static_cast<size_t>(RUNTIME_ATTR_COUNT));
    return false;
}

// 读取每层整型属性；attrs/attrIndex/name 为列表/索引/名称，defaultValue 为缺省值，
// nodeName 用于日志，out 接收两层结果。
inline bool ReadPerConvInts(const gert::RuntimeAttrs* attrs, size_t attrIndex, const char* name, int64_t defaultValue,
                            const char* nodeName, int64_t out[NUM_CONVS])
{
    for (size_t layer = 0; layer < NUM_CONVS; ++layer) {
        out[layer] = defaultValue;
    }
    if (attrs == nullptr || attrs->GetAttrNum() <= attrIndex) {
        return true;
    }

    const auto* attrValues = attrs->GetListInt(attrIndex);
    if (attrValues == nullptr) {
        return true;
    }
    if (attrValues->GetSize() != NUM_CONVS) {
        OP_LOGE(nodeName, "fused_multi_conv2d: %s requires exactly %zu entries, got %zu", name, NUM_CONVS,
                static_cast<size_t>(attrValues->GetSize()));
        return false;
    }
    const int64_t* values = attrValues->GetData();
    for (size_t layer = 0; layer < NUM_CONVS; ++layer) {
        out[layer] = values[layer];
    }
    return true;
}

// 读取必填的每层布尔属性；attrs/attrIndex/name 为列表/索引/名称，nodeName 用于日志，out 接收两层结果。
inline bool ReadPerConvBools(const gert::RuntimeAttrs* attrs, size_t attrIndex, const char* name, const char* nodeName,
                             bool out[NUM_CONVS])
{
    for (size_t layer = 0; layer < NUM_CONVS; ++layer) {
        out[layer] = false;
    }
    if (attrs == nullptr || attrs->GetAttrNum() <= attrIndex) {
        OP_LOGE(nodeName, "fused_multi_conv2d: required attribute %s is missing", name);
        return false;
    }
    const auto* attrValues = attrs->GetAttrPointer<gert::ContinuousVector>(attrIndex);
    if (attrValues == nullptr) {
        OP_LOGE(nodeName, "fused_multi_conv2d: %s must be ListBool with one entry per convolution", name);
        return false;
    }
    if (attrValues->GetSize() != NUM_CONVS) {
        OP_LOGE(nodeName, "fused_multi_conv2d: %s requires exactly %zu entries, got %zu", name, NUM_CONVS,
                static_cast<size_t>(attrValues->GetSize()));
        return false;
    }
    const auto* values = static_cast<const bool*>(attrValues->GetData());
    for (size_t layer = 0; layer < NUM_CONVS; ++layer) {
        out[layer] = values[layer];
    }
    return true;
}

// 校验保留的 HF32 开关；attrs 为属性列表，nodeName 用于日志。
inline bool CheckHf32Attribute(const gert::RuntimeAttrs* attrs, const char* nodeName)
{
    if (attrs->GetAttrNum() > ATTR_ENABLE_HF32) {
        const auto* attrValues = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_ENABLE_HF32);
        if (attrValues != nullptr) {
            if (attrValues->GetSize() != NUM_CONVS) {
                OP_LOGE(nodeName, "fused_multi_conv2d: enable_hf32 requires two entries");
                return false;
            }
            const bool* values = static_cast<const bool*>(attrValues->GetData());
            if (values[0] || values[1]) {
                OP_LOGE(nodeName, "fused_multi_conv2d: enable_hf32 only supports [false,false]");
                return false;
            }
        }
    }
    return true;
}

// 校验保留属性仅取默认值；attrs 为属性列表，nodeName 用于日志。
inline bool CheckReservedAttrs(const gert::RuntimeAttrs* attrs, const char* nodeName)
{
    if (attrs == nullptr) {
        return true;
    }
    struct DefaultListAttributeSpec {
        size_t idx;       // 属性索引
        const char* name; // 属性名称
        size_t count;     // 列表元素数
        int64_t value;    // 允许的默认值
    };
    static const DefaultListAttributeSpec DEFAULT_LIST_ATTRIBUTES[] = {
        {ATTR_DILATIONS, "dilations", 8, 1},           {ATTR_GROUPS, "groups", NUM_CONVS, 1},
        {ATTR_OFFSET_X, "offset_x", NUM_CONVS, 0},     {ATTR_DATA_FORMAT, "data_format", NUM_CONVS, 0},
        {ATTR_ROUND_MODE, "round_mode", NUM_CONVS, 0}, {ATTR_PAD_MODE, "pad_mode", NUM_CONVS, 0},
    };
    for (const auto& requirement : DEFAULT_LIST_ATTRIBUTES) {
        if (attrs->GetAttrNum() <= requirement.idx) {
            continue;
        }
        const auto* values = attrs->GetListInt(requirement.idx);
        if (values == nullptr) {
            continue;
        }
        if (values->GetSize() != requirement.count) {
            OP_LOGE(nodeName, "fused_multi_conv2d: %s requires %zu entries", requirement.name, requirement.count);
            return false;
        }
        for (size_t index = 0; index < requirement.count; ++index) {
            if (values->GetData()[index] != requirement.value) {
                if (MappedEnumAttrIndex(requirement.name) != PUBLIC_ATTR_COUNT) {
                    const char* mappedName = MappedEnumValueName(requirement.idx, values->GetData()[index]);
                    OP_LOGE(nodeName, "fused_multi_conv2d: %s[%zu]=%ld (%s) is %s; only 0 (%s) is supported",
                            requirement.name, index, values->GetData()[index],
                            mappedName == nullptr ? "unknown" : mappedName,
                            mappedName == nullptr ? "an invalid enum code" : "not implemented",
                            MappedEnumValueName(requirement.idx, 0));
                } else {
                    OP_LOGE(nodeName, "fused_multi_conv2d: %s[%zu] must be %ld", requirement.name, index,
                            requirement.value);
                }
                return false;
            }
        }
    }
    return CheckHf32Attribute(attrs, nodeName);
}

// 读取指定存储维度，索引越界返回 -1；shape 为存储形状，index 为维度索引。
int64_t GetStorageDim(const gert::StorageShape* shape, size_t index)
{
    return index < shape->GetStorageShape().GetDimNum() ? shape->GetStorageShape().GetDim(index) : -1;
}
// 读取存储形状维数；shape 为张量存储形状。
size_t GetStorageRank(const gert::StorageShape* shape) { return shape->GetStorageShape().GetDimNum(); }

// 读取每层对称补边；attrs 为属性列表，nodeName 用于日志，padHeight1/padWidth1/padHeight2/padWidth2 接收上下/左右补边。
bool ReadPads(const gert::RuntimeAttrs* attrs, const char* nodeName, int& padHeight1, int& padWidth1, int& padHeight2,
              int& padWidth2)
{
    padHeight1 = padWidth1 = padHeight2 = padWidth2 = 0;
    if (attrs == nullptr || attrs->GetAttrNum() <= ATTR_PADS) {
        return true;
    }
    const auto* attrValues = attrs->GetListInt(ATTR_PADS);
    if (attrValues == nullptr) {
        return true;
    }
    if (attrValues->GetSize() != 8) {
        OP_LOGE(nodeName, "fused_multi_conv2d: pads requires eight entries [T1,B1,L1,R1,T2,B2,L2,R2]");
        return false;
    }
    const int64_t* padValues = attrValues->GetData();
    for (size_t index = 0; index < 8; ++index) {
        if (padValues[index] < 0 || padValues[index] > std::numeric_limits<int>::max()) {
            OP_LOGE(nodeName, "fused_multi_conv2d: pads[%zu] must be a nonnegative int", index);
            return false;
        }
    }
    for (size_t index = 0; index < 8; index += 2) {
        if (padValues[index] != padValues[index + 1]) {
            OP_LOGE(nodeName, "fused_multi_conv2d: pads must be symmetric: entries %zu and %zu differ", index,
                    index + 1);
            return false;
        }
    }
    padHeight1 = static_cast<int>(padValues[0]);
    padWidth1 = static_cast<int>(padValues[2]);
    padHeight2 = static_cast<int>(padValues[4]);
    padWidth2 = static_cast<int>(padValues[6]);
    return true;
}

struct LogicalFilter {
    int64_t outputChannels = 0; // 输出通道数
    int64_t inputChannels = 0;  // 输入通道数
    int64_t kernelHeight = 0;   // 卷积核高
    int64_t kernelWidth = 0;    // 卷积核宽
};

// 按原始格式读取逻辑权重维度；shape/desc 为原始形状/描述，nodeName 用于日志，layer 为层号，out 接收标准化维度。
bool ReadLogicalFilter(const gert::Shape* shape, const gert::CompileTimeTensorDesc* desc, const char* nodeName,
                       size_t layer, LogicalFilter& out)
{
    if (shape == nullptr || desc == nullptr || shape->GetDimNum() != 4) {
        OP_LOGE(nodeName, "fused_multi_conv2d: filter%zu requires a four-dimensional logical origin shape", layer);
        return false;
    }
    size_t outputAxis = 0, inputAxis = 1, heightAxis = 2, widthAxis = 3;
    switch (static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetOriginFormat()))) {
        case ge::FORMAT_NCHW:
            break;
        case ge::FORMAT_NHWC:
            inputAxis = 3;
            heightAxis = 1;
            widthAxis = 2;
            break;
        case ge::FORMAT_HWCN:
            outputAxis = 3;
            inputAxis = 2;
            heightAxis = 0;
            widthAxis = 1;
            break;
        default:
            OP_LOGE(nodeName, "fused_multi_conv2d: filter%zu origin format must be NCHW, NHWC or HWCN, got %d", layer,
                    static_cast<int>(desc->GetOriginFormat()));
            return false;
    }
    out = {shape->GetDim(outputAxis), shape->GetDim(inputAxis), shape->GetDim(heightAxis), shape->GetDim(widthAxis)};
    for (int64_t extent : {out.outputChannels, out.inputChannels, out.kernelHeight, out.kernelWidth}) {
        if (extent <= 0 || extent > std::numeric_limits<int>::max()) {
            OP_LOGE(nodeName, "fused_multi_conv2d: filter%zu requires positive static logical dimensions in int range",
                    layer);
            return false;
        }
    }
    return true;
}

// 读取 [H1,W1,H2,W2] 并检查每层高宽相等；attrs 为列表，nodeName 用于日志，stride1/stride2 接收两层步长。
bool ReadStrides(const gert::RuntimeAttrs* attrs, const char* nodeName, int& stride1, int& stride2)
{
    stride1 = stride2 = 1;
    if (attrs == nullptr || attrs->GetAttrNum() <= ATTR_STRIDES) {
        return true;
    }
    const auto* attrValues = attrs->GetListInt(ATTR_STRIDES);
    if (attrValues == nullptr) {
        return true;
    }
    if (attrValues->GetSize() != 4) {
        OP_LOGE(nodeName, "fused_multi_conv2d: strides requires four entries [H1,W1,H2,W2]");
        return false;
    }
    const int64_t* strideValues = attrValues->GetData();
    for (size_t index = 0; index < 4; ++index) {
        if (strideValues[index] <= 0 || strideValues[index] > std::numeric_limits<int>::max()) {
            OP_LOGE(nodeName, "fused_multi_conv2d: strides[%zu] must be a positive int", index);
            return false;
        }
    }
    if (strideValues[0] != strideValues[1] || strideValues[2] != strideValues[3]) {
        OP_LOGE(nodeName, "fused_multi_conv2d: each convolution requires equal height and width strides");
        return false;
    }
    stride1 = static_cast<int>(strideValues[0]);
    stride2 = static_cast<int>(strideValues[2]);
    return true;
}

// 检查单轴卷积输出；input/kernel/stride/pad 为输入长度/卷积核/步长/对称补边，nodeName 用于日志，output 接收长度。
bool CheckedConvExtent(int64_t input, int kernel, int stride, int pad, const char* nodeName, int64_t& output)
{
    const int64_t padded = input + 2LL * pad;
    if (input <= 0 || kernel <= 0 || stride <= 0 || pad < 0 || padded < kernel ||
        padded > std::numeric_limits<int>::max()) {
        OP_LOGE(nodeName, "fused_multi_conv2d: padded convolution extent must contain its kernel and fit int");
        return false;
    }
    output = (padded - kernel) / stride + 1;
    return output > 0 && output <= std::numeric_limits<int>::max();
}

} // namespace

namespace optiling {

struct ConvTilingPlatformInfo {
    bool isCubeVectorFuse = false; // Cube/Vector 融合架构标志
    uint32_t aicNum = 0;           // 可用矩阵核心数
    int l1Budget = 0;              // 每核 L1 字节预算
};

struct ConvTensorMetadata {
    const gert::CompileTimeTensorDesc* xDesc = nullptr;  // 输入描述
    const gert::CompileTimeTensorDesc* f1Desc = nullptr; // 第一层权重描述
    const gert::CompileTimeTensorDesc* f2Desc = nullptr; // 第二层权重描述
    const gert::CompileTimeTensorDesc* yDesc = nullptr;  // 输出描述
    const gert::StorageShape* xShape = nullptr;          // 输入存储形状
    const gert::StorageShape* f1Shape = nullptr;         // 第一层权重存储形状
    const gert::StorageShape* f2Shape = nullptr;         // 第二层权重存储形状
    const gert::StorageShape* yShape = nullptr;          // 输出存储形状
    const gert::StorageShape* b1Shape = nullptr;         // 第一层偏置存储形状
    const gert::StorageShape* b2Shape = nullptr;         // 第二层偏置存储形状
    const gert::StorageShape* q1Shape = nullptr;         // 第一层量化表形状
    const gert::StorageShape* q2Shape = nullptr;         // 第二层量化表形状
    const gert::StorageShape* dq2Shape = nullptr;        // 第二层反量化表形状
    bool allFp16 = false;                                // 全半精度路径
    bool allInt8 = false;                                // 全整型路径
    bool a16w8 = false;                                  // 半精度输入与整型权重路径
    bool hasQuantScale1 = false;                         // 第一层使用量化表
    bool hasQuantScale2 = false;                         // 第二层使用量化表
    bool hasDeqScale2 = false;                           // 第二层使用反量化表
    bool hasBias1 = false;                               // 第一层启用偏置
    bool hasBias2 = false;                               // 第二层启用偏置
};

struct ConvExecutionAttributes {
    int64_t fixedShift1 = 0; // 第一层定标值
    int64_t fixedShift2 = 0; // 第二层定标值
    bool relu1 = false;      // 第一层启用普通 ReLU
    bool relu2 = false;      // 第二层启用普通 ReLU
};

// 读取芯片能力和片上资源；context 为运行时上下文，platformInfo 接收平台信息。
static bool ReadPlatformInfo(gert::TilingContext* context, ConvTilingPlatformInfo& platformInfo)
{
    auto* platformResources = context->GetPlatformInfo();
    if (platformResources == nullptr) {
        return false;
    }
    platformInfo.isCubeVectorFuse = conv_arch::IsCubeVectorFuseSoc(*platformResources);
    if (!platformInfo.isCubeVectorFuse) {
        return false;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformResources);
    platformInfo.aicNum = platform.GetCoreNumAic();
    if (platformInfo.aicNum == 0) {
        return false;
    }
    uint64_t l1Size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    platformInfo.l1Budget = l1Size == 0 ? 1024 * 1024 : static_cast<int>(l1Size);
    return true;
}

// 校验各动态列表的实例数；context 为输入描述所在的运行时上下文。
static bool ValidateTensorListInstances(gert::TilingContext* context)
{
    const auto* filtersInfo = context->GetIrInputInstanceInfo(IR_FILTERS);
    if (filtersInfo == nullptr || filtersInfo->GetInstanceNum() != NUM_CONVS) {
        OP_LOGE(context->GetNodeName(), "fused_multi_conv2d: filters requires exactly two instances, got %zu",
                filtersInfo == nullptr ? static_cast<size_t>(0) : filtersInfo->GetInstanceNum());
        return false;
    }
    const auto* biasInfo = context->GetIrInputInstanceInfo(IR_BIAS);
    const auto* scaleInfo = context->GetIrInputInstanceInfo(IR_SCALE);
    const auto* reluWeightInfo = context->GetIrInputInstanceInfo(IR_RELU_WEIGHT);
    const size_t biasCount = biasInfo == nullptr ? 0U : biasInfo->GetInstanceNum();
    const size_t scaleCount = scaleInfo == nullptr ? 0U : scaleInfo->GetInstanceNum();
    const size_t reluWeightCount = reluWeightInfo == nullptr ? 0U : reluWeightInfo->GetInstanceNum();
    return ExpectedInstanceNum("bias", biasCount, context->GetNodeName()) &&
           ExpectedInstanceNum("scale", scaleCount, context->GetNodeName()) &&
           ExpectedInstanceNum("relu_weight", reluWeightCount, context->GetNodeName());
}

// 收集必需张量的描述和存储形状；context 为上下文，tensors 接收读取结果。
static bool ReadTensorMetadata(gert::TilingContext* context, ConvTensorMetadata& tensors)
{
    tensors.xDesc = context->GetRequiredInputDesc(IR_X);
    tensors.f1Desc = context->GetDynamicInputDesc(IR_FILTERS, 0);
    tensors.f2Desc = context->GetDynamicInputDesc(IR_FILTERS, 1);
    tensors.yDesc = context->GetOutputDesc(0);
    tensors.xShape = context->GetRequiredInputShape(IR_X);
    tensors.f1Shape = context->GetDynamicInputShape(IR_FILTERS, 0);
    tensors.f2Shape = context->GetDynamicInputShape(IR_FILTERS, 1);
    tensors.yShape = context->GetOutputShape(0);
    return tensors.xDesc != nullptr && tensors.f1Desc != nullptr && tensors.f2Desc != nullptr &&
           tensors.yDesc != nullptr && tensors.xShape != nullptr && tensors.f1Shape != nullptr &&
           tensors.f2Shape != nullptr && tensors.yShape != nullptr;
}

// 选择受支持的数据通路及定标表；context 用于日志，tensors 提供类型并接收通路标志。
static bool ConfigureDataPath(gert::TilingContext* context, ConvTensorMetadata& tensors)
{
    const ge::DataType inputType = tensors.xDesc->GetDataType();
    const ge::DataType weightType1 = tensors.f1Desc->GetDataType();
    const ge::DataType weightType2 = tensors.f2Desc->GetDataType();
    const ge::DataType outputType = tensors.yDesc->GetDataType();
    tensors.allFp16 = inputType == ge::DT_FLOAT16 && weightType1 == ge::DT_FLOAT16 && weightType2 == ge::DT_FLOAT16 &&
                      outputType == ge::DT_FLOAT16;
    tensors.allInt8 = inputType == ge::DT_INT8 && weightType1 == ge::DT_INT8 && weightType2 == ge::DT_INT8 &&
                      outputType == ge::DT_INT8;
    const bool int8ToFp16 = inputType == ge::DT_INT8 && weightType1 == ge::DT_INT8 && weightType2 == ge::DT_INT8 &&
                            outputType == ge::DT_FLOAT16;
    tensors.a16w8 = inputType == ge::DT_FLOAT16 && weightType1 == ge::DT_INT8 && weightType2 == ge::DT_INT8 &&
                    outputType == ge::DT_FLOAT16;
    if (!tensors.allFp16 && !tensors.allInt8 && !int8ToFp16 && !tensors.a16w8) {
        OP_LOGE(context->GetNodeName(),
                "fused_multi_conv2d: unsupported dtype combination x=%d filter1=%d filter2=%d y=%d",
                static_cast<int>(inputType), static_cast<int>(weightType1), static_cast<int>(weightType2),
                static_cast<int>(outputType));
        return false;
    }
    tensors.hasQuantScale1 = !tensors.allFp16;
    tensors.hasQuantScale2 = tensors.allInt8;
    tensors.hasDeqScale2 = int8ToFp16 || tensors.a16w8;
    return true;
}

// 读取启用的偏置和定标表；context 提供输入及 bias_mask，tensors 接收形状并校验偏置类型。
static bool ReadOptionalTensors(gert::TilingContext* context, ConvTensorMetadata& tensors)
{
    bool biasFlags[NUM_CONVS];
    if (!ReadPerConvBools(context->GetAttrs(), ATTR_BIAS_MASK, "bias_mask", context->GetNodeName(), biasFlags)) {
        return false;
    }
    tensors.hasBias1 = biasFlags[0];
    tensors.hasBias2 = biasFlags[1];
    tensors.b1Shape = tensors.hasBias1 ? context->GetDynamicInputShape(IR_BIAS, 0) : nullptr;
    tensors.b2Shape = tensors.hasBias2 ? context->GetDynamicInputShape(IR_BIAS, 1) : nullptr;
    tensors.q1Shape = tensors.hasQuantScale1 ? context->GetDynamicInputShape(IR_SCALE, 0) : nullptr;
    tensors.q2Shape = tensors.hasQuantScale2 ? context->GetDynamicInputShape(IR_SCALE, 1) : nullptr;
    tensors.dq2Shape = tensors.hasDeqScale2 ? context->GetDynamicInputShape(IR_SCALE, 1) : nullptr;
    const ge::DataType biasType = tensors.allFp16 ? ge::DT_FLOAT16 : ge::DT_INT32;
    const auto* biasDesc1 = tensors.hasBias1 ? context->GetDynamicInputDesc(IR_BIAS, 0) : nullptr;
    const auto* biasDesc2 = tensors.hasBias2 ? context->GetDynamicInputDesc(IR_BIAS, 1) : nullptr;
    if ((biasDesc1 != nullptr && biasDesc1->GetDataType() != biasType) ||
        (biasDesc2 != nullptr && biasDesc2->GetDataType() != biasType)) {
        OP_LOGE(context->GetNodeName(), "fused_multi_conv2d: bias dtype must be %d for this path",
                static_cast<int>(biasType));
        return false;
    }
    return true;
}

// 读取并校验两层定标值；attrs 为属性列表，a16w8 为通路标志，nodeName 用于日志，out 接收值。
static bool ReadFixedShift(const gert::RuntimeAttrs* attrs, bool a16w8, const char* nodeName,
                           ConvExecutionAttributes& out)
{
    int64_t fixedShift[NUM_CONVS];
    if (!ReadPerConvInts(attrs, ATTR_FIXED_SHIFT, FIXED_SHIFT_ATTR, DEFAULT_FIXED_SHIFT, nodeName, fixedShift)) {
        return false;
    }
    const int64_t firstLimit = a16w8 ? 29 : S::FIX_SHIFT_LEN_A16W16;
    if (fixedShift[0] < 0 || fixedShift[0] > firstLimit || fixedShift[1] < 0 ||
        fixedShift[1] > S::FIX_SHIFT_LEN_A16W16) {
        OP_LOGE(nodeName, "fused_multi_conv2d: fixed_shift_value requires first entry in [0,%ld], second in [0,58]",
                firstLimit);
        return false;
    }
    out.fixedShift1 = fixedShift[0];
    out.fixedShift2 = fixedShift[1];
    return true;
}

// 读取两层普通 ReLU 开关；attrs 为属性列表，nodeName 用于日志，out 接收激活配置。
static bool ReadReluModes(const gert::RuntimeAttrs* attrs, const char* nodeName, ConvExecutionAttributes& out)
{
    int64_t reluModes[NUM_CONVS];
    if (!ReadPerConvInts(attrs, ATTR_ENABLE_RELU, "enable_relu", RELU_NORMAL, nodeName, reluModes)) {
        return false;
    }
    for (size_t layer = 0; layer < NUM_CONVS; ++layer) {
        if (reluModes[layer] == RELU_SCALAR || reluModes[layer] == RELU_VECTOR) {
            OP_LOGE(nodeName, "fused_multi_conv2d: relu[%zu]=%ld is unsupported; only 0 and 1 are supported", layer,
                    static_cast<long>(reluModes[layer]));
            return false;
        }
        if (reluModes[layer] != RELU_NO && reluModes[layer] != RELU_NORMAL) {
            OP_LOGE(nodeName, "fused_multi_conv2d: relu[%zu]=%ld is invalid", layer,
                    static_cast<long>(reluModes[layer]));
            return false;
        }
    }
    out.relu1 = reluModes[0] != RELU_NO;
    out.relu2 = reluModes[1] != RELU_NO;
    return true;
}

// 校验属性布局并读取执行开关；context 为上下文，tensors 提供通路，out 接收结果。
static bool ReadExecutionAttributes(gert::TilingContext* context, const ConvTensorMetadata& tensors,
                                    ConvExecutionAttributes& out)
{
    const auto* attrs = context->GetAttrs();
    const char* nodeName = context->GetNodeName();
    return attrs != nullptr && RequiredAttrsPresent(attrs, nodeName) && CheckReservedAttrs(attrs, nodeName) &&
           CheckPrivateAttrsLayout(attrs, nodeName) && ReadFixedShift(attrs, tensors.a16w8, nodeName, out) &&
           ReadReluModes(attrs, nodeName, out);
}

// 校验物理格式及形状维数；context 用于日志，tensors 为必需张量描述。
static bool ValidateStorageFormats(gert::TilingContext* context, const ConvTensorMetadata& tensors)
{
    if (ge::GetPrimaryFormat(tensors.xDesc->GetStorageFormat()) != ge::FORMAT_NCHW ||
        ge::GetPrimaryFormat(tensors.yDesc->GetStorageFormat()) != ge::FORMAT_NCHW ||
        ge::GetPrimaryFormat(tensors.f1Desc->GetStorageFormat()) != ge::FORMAT_FRACTAL_Z ||
        ge::GetPrimaryFormat(tensors.f2Desc->GetStorageFormat()) != ge::FORMAT_FRACTAL_Z) {
        OP_LOGE(context->GetNodeName(), "fused_multi_conv2d: storage formats must be NCHW/FRACTAL_Z/FRACTAL_Z/NCHW");
        return false;
    }
    if (GetStorageRank(tensors.xShape) != 4 || GetStorageRank(tensors.f1Shape) != 4 ||
        GetStorageRank(tensors.f2Shape) != 4 || GetStorageRank(tensors.yShape) != 4) {
        OP_LOGE(context->GetNodeName(), "fused_multi_conv2d: x, filter1, filter2 and y must have four dimensions");
        return false;
    }
    for (size_t axis = 0; axis < 4; ++axis) {
        const int64_t extent = GetStorageDim(tensors.xShape, axis);
        if (extent <= 0 || extent > std::numeric_limits<int>::max()) {
            OP_LOGE(context->GetNodeName(), "fused_multi_conv2d: x requires positive static dimensions in int range");
            return false;
        }
    }
    return true;
}

// 汇总类型、形状和卷积核信息；tensors/filter1/filter2 为张量信息，inputs 接收分块输入。
static void FillConvolutionInputs(const ConvTensorMetadata& tensors, const LogicalFilter& filter1,
                                  const LogicalFilter& filter2, S::ConvTilingInputs& inputs)
{
    inputs.batchSize = static_cast<int>(GetStorageDim(tensors.xShape, 0));
    inputs.inputChannels = static_cast<int>(GetStorageDim(tensors.xShape, 1));
    inputs.inputHeight = static_cast<int>(GetStorageDim(tensors.xShape, 2));
    inputs.inputWidth = static_cast<int>(GetStorageDim(tensors.xShape, 3));
    inputs.outputChannels1 = static_cast<int>(filter1.outputChannels);
    inputs.outputChannels2 = static_cast<int>(filter2.outputChannels);
    inputs.kernelHeight1 = static_cast<int>(filter1.kernelHeight);
    inputs.kernelWidth1 = static_cast<int>(filter1.kernelWidth);
    inputs.kernelHeight2 = static_cast<int>(filter2.kernelHeight);
    inputs.kernelWidth2 = static_cast<int>(filter2.kernelWidth);
    inputs.inputElementBytes = tensors.xDesc->GetDataType() == ge::DT_FLOAT16 ? 2 : 1;
    inputs.weightElementBytes = tensors.f1Desc->GetDataType() == ge::DT_FLOAT16 ? 2 : 1;
    inputs.intermediateElementBytes = tensors.allFp16 ? 2 : 1;
    inputs.biasElementBytes = tensors.allFp16 ? 2 : 4;
    inputs.hasDequantScale2 = tensors.hasDeqScale2 ? 1 : 0;
    inputs.hasQuantScale1 = tensors.hasQuantScale1 ? 1 : 0;
    inputs.hasQuantScale2 = tensors.hasQuantScale2 ? 1 : 0;
}

// 构造分块输入并校验两层通道连接；context/tensors 为运行时输入，inputs 接收完整卷积配置。
static bool ReadConvolutionInputs(gert::TilingContext* context, const ConvTensorMetadata& tensors,
                                  S::ConvTilingInputs& inputs)
{
    LogicalFilter filter1, filter2;
    if (!ReadLogicalFilter(&tensors.f1Shape->GetOriginShape(), tensors.f1Desc, context->GetNodeName(), 1, filter1) ||
        !ReadLogicalFilter(&tensors.f2Shape->GetOriginShape(), tensors.f2Desc, context->GetNodeName(), 2, filter2) ||
        !ValidateStorageFormats(context, tensors)) {
        return false;
    }
    if (filter1.inputChannels != GetStorageDim(tensors.xShape, 1) || filter2.inputChannels != filter1.outputChannels) {
        OP_LOGE(context->GetNodeName(),
                "fused_multi_conv2d: filter logical Cin must match x channels and preceding Cout");
        return false;
    }
    FillConvolutionInputs(tensors, filter1, filter2, inputs);
    return ReadStrides(context->GetAttrs(), context->GetNodeName(), inputs.stride1, inputs.stride2) &&
           ReadPads(context->GetAttrs(), context->GetNodeName(), inputs.paddingHeight1, inputs.paddingWidth1,
                    inputs.paddingHeight2, inputs.paddingWidth2);
}

// 用宽位中间值校验两层输出尺寸；inputs 为卷积配置，nodeName 用于日志。
static bool ValidateConvolutionExtents(const S::ConvTilingInputs& inputs, const char* nodeName)
{
    int64_t ho1 = 0, wo1 = 0, ho2 = 0, wo2 = 0;
    return CheckedConvExtent(inputs.inputHeight, inputs.kernelHeight1, inputs.stride1, inputs.paddingHeight1, nodeName,
                             ho1) &&
           CheckedConvExtent(inputs.inputWidth, inputs.kernelWidth1, inputs.stride1, inputs.paddingWidth1, nodeName,
                             wo1) &&
           CheckedConvExtent(ho1, inputs.kernelHeight2, inputs.stride2, inputs.paddingHeight2, nodeName, ho2) &&
           CheckedConvExtent(wo1, inputs.kernelWidth2, inputs.stride2, inputs.paddingWidth2, nodeName, wo2);
}

// 搜索可执行分块；inputs 为卷积配置，platformInfo 为资源，nodeName 用于日志，geometry 接收几何。
static bool SelectConvolutionGeometry(const S::ConvTilingInputs& inputs, const ConvTilingPlatformInfo& platformInfo,
                                      const char* nodeName, S::ConvTilingGeometry& geometry)
{
    const int status = S::PickHb(inputs, static_cast<int>(platformInfo.aicNum), platformInfo.l1Budget, geometry);
    if (status == S::RJ_OK) {
        return true;
    }
    OP_LOGE(nodeName,
            "fused_multi_conv2d[%s]: unsupported geometry: %s; x=[%d,%d,%d,%d] cout=%d/%d "
            "kernel=%dx%d/%dx%d stride=%d/%d pad=%d,%d/%d,%d elemBytes=%d L1=%d",
            FC2D_GEOM_REV, S::RejectText(status), inputs.batchSize, inputs.inputChannels, inputs.inputHeight,
            inputs.inputWidth, inputs.outputChannels1, inputs.outputChannels2, inputs.kernelHeight1,
            inputs.kernelWidth1, S::Kh2(inputs), S::Kw2(inputs), inputs.stride1, inputs.stride2, inputs.paddingHeight1,
            inputs.paddingWidth1, inputs.paddingHeight2, inputs.paddingWidth2, inputs.inputElementBytes,
            platformInfo.l1Budget);
    return false;
}

// 核对存储形状和推导结果；tensors 为实际张量，inputs/geometry 为逻辑及分块几何，nodeName 用于日志。
static bool ValidateDerivedTensorShapes(const ConvTensorMetadata& tensors, const S::ConvTilingInputs& inputs,
                                        const S::ConvTilingGeometry& geometry, const char* nodeName)
{
    const int64_t weightC0 = 32 / inputs.weightElementBytes;
    const bool matches = inputs.batchSize >= 1 && inputs.inputChannels >= 1 && inputs.inputHeight >= 1 &&
                         inputs.inputWidth >= 1 && GetStorageDim(tensors.f1Shape, 0) == geometry.filterStorageK1 &&
                         GetStorageDim(tensors.f1Shape, 1) == geometry.filterStorageN1 &&
                         GetStorageDim(tensors.f1Shape, 2) == 16 && GetStorageDim(tensors.f1Shape, 3) == weightC0 &&
                         inputs.outputChannels2 >= 1 && GetStorageDim(tensors.f2Shape, 0) == geometry.filterStorageK2 &&
                         GetStorageDim(tensors.f2Shape, 1) == geometry.filterStorageN2 &&
                         GetStorageDim(tensors.f2Shape, 2) == 16 && GetStorageDim(tensors.f2Shape, 3) == weightC0 &&
                         (!tensors.hasBias1 || GetStorageDim(tensors.b1Shape, 0) == inputs.outputChannels1) &&
                         (!tensors.hasBias2 || GetStorageDim(tensors.b2Shape, 0) == inputs.outputChannels2) &&
                         (!tensors.hasDeqScale2 || GetStorageDim(tensors.dq2Shape, 0) == inputs.outputChannels2) &&
                         (!tensors.hasQuantScale1 || GetStorageDim(tensors.q1Shape, 0) == inputs.outputChannels1) &&
                         (!tensors.hasQuantScale2 || GetStorageDim(tensors.q2Shape, 0) == inputs.outputChannels2) &&
                         GetStorageDim(tensors.yShape, 0) == inputs.batchSize &&
                         GetStorageDim(tensors.yShape, 1) == inputs.outputChannels2 &&
                         GetStorageDim(tensors.yShape, 2) == geometry.outputHeight2 &&
                         GetStorageDim(tensors.yShape, 3) == geometry.outputWidth2;
    if (!matches) {
        OP_LOGE(nodeName,
                "fused_multi_conv2d: shape mismatch; expected filter1=[%d,%d,16,%ld] filter2=[%d,%d,16,%ld] "
                "bias=[%d]/[%d] y=[%d,%d,%d,%d]",
                geometry.filterStorageK1, geometry.filterStorageN1, static_cast<long>(weightC0),
                geometry.filterStorageK2, geometry.filterStorageN2, static_cast<long>(weightC0), inputs.outputChannels1,
                inputs.outputChannels2, inputs.batchSize, inputs.outputChannels2, geometry.outputHeight2,
                geometry.outputWidth2);
    }
    return matches;
}

// 分配核心任务并序列化；context 为输出上下文，geometry 为分块，platformInfo 为平台资源，
// attrs 为执行属性，tensors 为偏置开关与类型信息。
static ge::graphStatus SaveTilingData(gert::TilingContext* context, const S::ConvTilingGeometry& geometry,
                                      const ConvTilingPlatformInfo& platformInfo, const ConvExecutionAttributes& attrs,
                                      const ConvTensorMetadata& tensors)
{
    const uint32_t total = static_cast<uint32_t>(geometry.totalBandTasks);
    uint32_t blockDim = platformInfo.aicNum < total ? platformInfo.aicNum : total;
    const uint32_t chunksPerCore = (total + blockDim - 1) / blockDim;
    blockDim = (total + chunksPerCore - 1) / chunksPerCore;
    S::FlatGeom flat;
    S::Flatten(geometry, static_cast<int>(chunksPerCore), flat);
    FusedMultiConv2dTilingData tilingData;
#define FC2D_SET_FIELD(member, name) tilingData.set_##name(static_cast<uint32_t>(flat.member));
    FC2D_FLAT_FIELDS(FC2D_SET_FIELD)
#undef FC2D_SET_FIELD
    tilingData.set_blockDim(blockDim);
    tilingData.set_fixedShift1(static_cast<uint32_t>(attrs.fixedShift1));
    tilingData.set_fixedShift2(static_cast<uint32_t>(attrs.fixedShift2));
    tilingData.set_hasBias1(tensors.hasBias1 ? 1U : 0U);
    tilingData.set_hasBias2(tensors.hasBias2 ? 1U : 0U);
    tilingData.set_relu1(attrs.relu1 ? 1U : 0U);
    tilingData.set_relu2(attrs.relu2 ? 1U : 0U);
    OP_LOGI(context->GetNodeName(),
            "fused_multi_conv2d tiling[%s]: hb=%d nchunk=%d wseg=%d nwseg=%d chunkTotal=%u blockDim=%u/%u "
            "mid=%dx%d inputWindow=%dx%d tileK=%d/%d maxM=%d/%d L0BSegments=%d/%d "
            "weightStage=%d/%d L0CStride=%d L1=%d/%d",
            FC2D_GEOM_REV, geometry.bandHeight, geometry.heightBandCount, geometry.segmentWidth,
            geometry.widthSegmentCount, total, blockDim, platformInfo.aicNum, geometry.intermediateRows,
            geometry.intermediateWidthMax, geometry.inputWindowRows, geometry.inputWindowWidthMax, geometry.tileK1,
            geometry.tileK2, geometry.tileMMax1, geometry.tileMMax2, geometry.weightSegmentCount1,
            geometry.weightSegmentCount2, geometry.weightRing1, geometry.weightRing2, geometry.l0cSlotStride,
            geometry.l1UsedBytes, platformInfo.l1Budget);
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = static_cast<size_t>(16 * 1024 * 1024);
    return ge::GRAPH_SUCCESS;
}

// 校验输入并生成分块；context 提供运行时张量、属性和 tiling 输出空间。
static ge::graphStatus TilingForFusedMultiConv2d(gert::TilingContext* context)
{
    ConvTilingPlatformInfo platformInfo;
    ConvTensorMetadata tensors;
    ConvExecutionAttributes executionAttrs;
    S::ConvTilingInputs inputs;
    S::ConvTilingGeometry geometry;
    if (!ReadPlatformInfo(context, platformInfo) || !ValidateTensorListInstances(context) ||
        !ReadTensorMetadata(context, tensors) || !ConfigureDataPath(context, tensors) ||
        !ReadOptionalTensors(context, tensors) || !ReadExecutionAttributes(context, tensors, executionAttrs) ||
        !ReadConvolutionInputs(context, tensors, inputs) ||
        !ValidateConvolutionExtents(inputs, context->GetNodeName()) ||
        !SelectConvolutionGeometry(inputs, platformInfo, context->GetNodeName(), geometry) ||
        !ValidateDerivedTensorShapes(tensors, inputs, geometry, context->GetNodeName())) {
        return ge::GRAPH_FAILED;
    }
    return SaveTilingData(context, geometry, platformInfo, executionAttrs, tensors);
}

// 完成编译期准备；context 为解析上下文，当前无额外解析数据。
static ge::graphStatus TilingPrepareForFusedMultiConv2d(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedMultiConv2d).Tiling(TilingForFusedMultiConv2d).TilingParse<int>(TilingPrepareForFusedMultiConv2d);
} // namespace optiling

namespace ops {

struct ConvLogicalShapeInfo {
    const gert::Shape* inputShape = nullptr;   // NCHW 输入逻辑形状
    const gert::Shape* filterShape1 = nullptr; // 第一层权重逻辑形状
    const gert::Shape* filterShape2 = nullptr; // 第二层权重逻辑形状
    gert::Shape* outputShape = nullptr;        // 待填写的输出形状
    LogicalFilter filter1;                     // 第一层标准化逻辑维度
    LogicalFilter filter2;                     // 第二层标准化逻辑维度
};

// 读取形状推导的必需张量；context 为推导上下文，shapes 接收逻辑形状及权重维度。
static bool ReadLogicalShapeInfo(gert::InferShapeContext* context, ConvLogicalShapeInfo& shapes)
{
    const auto* filtersInfo = context->GetIrInputInstanceInfo(IR_FILTERS);
    const size_t filtersCount = filtersInfo == nullptr ? 0U : filtersInfo->GetInstanceNum();
    shapes.inputShape = context->GetRequiredInputShape(IR_X);
    shapes.filterShape1 = context->GetDynamicInputShape(IR_FILTERS, 0);
    shapes.filterShape2 = context->GetDynamicInputShape(IR_FILTERS, 1);
    shapes.outputShape = context->GetOutputShape(0);
    if (filtersCount != NUM_CONVS) {
        OP_LOGE(context->GetNodeName(), "infershape: filters requires exactly two instances, got %zu", filtersCount);
        return false;
    }
    if (shapes.inputShape == nullptr || shapes.filterShape1 == nullptr || shapes.filterShape2 == nullptr ||
        shapes.outputShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "infershape: required tensor shape is missing");
        return false;
    }
    const auto* inputDesc = context->GetRequiredInputDesc(IR_X);
    const auto* filterDesc1 = context->GetDynamicInputDesc(IR_FILTERS, 0);
    const auto* filterDesc2 = context->GetDynamicInputDesc(IR_FILTERS, 1);
    return inputDesc != nullptr && ge::GetPrimaryFormat(inputDesc->GetOriginFormat()) == ge::FORMAT_NCHW &&
           shapes.inputShape->GetDimNum() == 4 &&
           ReadLogicalFilter(shapes.filterShape1, filterDesc1, context->GetNodeName(), 1, shapes.filter1) &&
           ReadLogicalFilter(shapes.filterShape2, filterDesc2, context->GetNodeName(), 2, shapes.filter2);
}

// 检查逻辑输入尺寸和层间通道；shapes 为已读取的逻辑形状，nodeName 用于日志。
static bool ValidateLogicalShapeInfo(const ConvLogicalShapeInfo& shapes, const char* nodeName)
{
    for (size_t axis = 0; axis < 4; ++axis) {
        const int64_t extent = shapes.inputShape->GetDim(axis);
        if (extent <= 0 || extent > std::numeric_limits<int>::max()) {
            OP_LOGE(nodeName, "infershape: x requires positive static dimensions in int range");
            return false;
        }
    }
    if (shapes.filter1.inputChannels != shapes.inputShape->GetDim(1) ||
        shapes.filter2.inputChannels != shapes.filter1.outputChannels) {
        OP_LOGE(nodeName, "infershape: logical filter Cin does not match preceding tensor channels");
        return false;
    }
    return true;
}

// 构造形状计算配置；context 提供属性，shapes 提供逻辑维度，inputs 接收卷积配置。
static bool ReadInferenceConvolutionInputs(gert::InferShapeContext* context, const ConvLogicalShapeInfo& shapes,
                                           S::ConvTilingInputs& inputs)
{
    inputs.batchSize = static_cast<int>(shapes.inputShape->GetDim(0));
    inputs.inputChannels = static_cast<int>(shapes.inputShape->GetDim(1));
    inputs.inputHeight = static_cast<int>(shapes.inputShape->GetDim(2));
    inputs.inputWidth = static_cast<int>(shapes.inputShape->GetDim(3));
    inputs.outputChannels1 = static_cast<int>(shapes.filter1.outputChannels);
    inputs.outputChannels2 = static_cast<int>(shapes.filter2.outputChannels);
    inputs.kernelHeight1 = static_cast<int>(shapes.filter1.kernelHeight);
    inputs.kernelWidth1 = static_cast<int>(shapes.filter1.kernelWidth);
    inputs.kernelHeight2 = static_cast<int>(shapes.filter2.kernelHeight);
    inputs.kernelWidth2 = static_cast<int>(shapes.filter2.kernelWidth);
    const auto* attrs = context->GetAttrs();
    if (!ReadStrides(attrs, context->GetNodeName(), inputs.stride1, inputs.stride2) ||
        !ReadPads(attrs, context->GetNodeName(), inputs.paddingHeight1, inputs.paddingWidth1, inputs.paddingHeight2,
                  inputs.paddingWidth2)) {
        return false;
    }
    if (inputs.kernelHeight1 <= 0 || inputs.kernelWidth1 <= 0 || inputs.kernelHeight2 <= 0 ||
        inputs.kernelWidth2 <= 0 || inputs.stride1 <= 0 || inputs.stride2 <= 0 || inputs.paddingHeight1 < 0 ||
        inputs.paddingWidth1 < 0 || inputs.paddingHeight2 < 0 || inputs.paddingWidth2 < 0) {
        OP_LOGE(context->GetNodeName(), "infershape: invalid kernel, stride or padding attributes");
        return false;
    }
    return true;
}

// 逐层计算输出并写入 NCHW 形状；inputs 为逻辑卷积配置，outputShape 为输出，nodeName 用于日志。
static bool WriteInferredOutputShape(const S::ConvTilingInputs& inputs, gert::Shape& outputShape, const char* nodeName)
{
    int64_t ho1 = 0, wo1 = 0, ho2 = 0, wo2 = 0;
    if (!CheckedConvExtent(inputs.inputHeight, inputs.kernelHeight1, inputs.stride1, inputs.paddingHeight1, nodeName,
                           ho1) ||
        !CheckedConvExtent(inputs.inputWidth, inputs.kernelWidth1, inputs.stride1, inputs.paddingWidth1, nodeName,
                           wo1) ||
        !CheckedConvExtent(ho1, inputs.kernelHeight2, inputs.stride2, inputs.paddingHeight2, nodeName, ho2) ||
        !CheckedConvExtent(wo1, inputs.kernelWidth2, inputs.stride2, inputs.paddingWidth2, nodeName, wo2)) {
        return false;
    }
    outputShape.SetDimNum(4);
    outputShape.SetDim(0, inputs.batchSize);
    outputShape.SetDim(1, inputs.outputChannels2);
    outputShape.SetDim(2, ho2);
    outputShape.SetDim(3, wo2);
    OP_LOGD(nodeName, "infershape: y=[%d,%d,%ld,%ld]", inputs.batchSize, inputs.outputChannels2, ho2, wo2);
    return true;
}

// 根据两层卷积的逻辑形状推导输出；context 提供输入、属性及可写输出形状。
static ge::graphStatus InferShapeForFusedMultiConv2d(gert::InferShapeContext* context)
{
    ConvLogicalShapeInfo shapes;
    S::ConvTilingInputs inputs;
    if (!ReadLogicalShapeInfo(context, shapes) || !ValidateLogicalShapeInfo(shapes, context->GetNodeName()) ||
        !ReadInferenceConvolutionInputs(context, shapes, inputs) ||
        !WriteInferredOutputShape(inputs, *shapes.outputShape, context->GetNodeName())) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 依据 dtype 属性或输入类型确定输出类型；context 为类型推导上下文。
static ge::graphStatus InferDataTypeForFusedMultiConv2d(gert::InferDataTypeContext* context)
{
    const auto* attrs = context->GetAttrs();
    int64_t outputType = ReadIntAttr(attrs, ATTR_DTYPE, -1);
    if (outputType == -1) {
        outputType = static_cast<int64_t>(context->GetInputDataType(0));
    }
    if (outputType != static_cast<int64_t>(ge::DT_FLOAT16) && outputType != static_cast<int64_t>(ge::DT_INT8)) {
        OP_LOGE(context->GetNodeName(), "fused_multi_conv2d: dtype must be DT_FLOAT16(%d) or DT_INT8(%d), got %lld",
                static_cast<int>(ge::DT_FLOAT16), static_cast<int>(ge::DT_INT8), static_cast<long long>(outputType));
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, static_cast<ge::DataType>(outputType));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedMultiConv2d)
    .InferShape(InferShapeForFusedMultiConv2d)
    .InferDataType(InferDataTypeForFusedMultiConv2d)

    .PrivateAttr(FusedMultiConv2dAttrs::FIXED_SHIFT_ATTR,
                 std::vector<int64_t>{FusedMultiConv2dAttrs::DEFAULT_FIXED_SHIFT,
                                      FusedMultiConv2dAttrs::DEFAULT_FIXED_SHIFT});
} // namespace ops
