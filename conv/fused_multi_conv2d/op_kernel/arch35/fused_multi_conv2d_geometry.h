/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_MULTI_CONV2D_GEOMETRY_H
#define FUSED_MULTI_CONV2D_GEOMETRY_H

#if defined(__NPU_ARCH__)
#define FC2D_GEOM_FN __aicore__ inline
#define FC2D_GEOM_CE __aicore__ constexpr
#else
#define FC2D_GEOM_FN inline
#define FC2D_GEOM_CE constexpr
#endif

namespace FusedMultiConv2dShape {

constexpr int L0A_BYTES = 64 * 1024;
constexpr int L0A_SLOT_BYTES = L0A_BYTES / 2;
constexpr int BT_BYTES = 4 * 1024;
constexpr int BT_SLOT_ELEMS = 512;
constexpr int BT_SLOT1_BYTE_OFF = 2048;
constexpr int MMAD_M0 = 16;
constexpr int MAX_M_TILES = 64;
constexpr int FIX_SHIFT_LEN_A16W16 = 58;

// 返回较小值；left/right 为比较值。
FC2D_GEOM_CE int MinI(int left, int right) { return left < right ? left : right; }
// 返回较大值；left/right 为比较值。
FC2D_GEOM_CE int MaxI(int left, int right) { return left > right ? left : right; }
// 向上取整除法；value 为被除数，divisor 为除数。
FC2D_GEOM_CE int CeilDiv(int value, int divisor) { return (value + divisor - 1) / divisor; }
// 向上对齐；value 为原值，alignment 为对齐粒度。
FC2D_GEOM_CE int Align(int value, int alignment) { return (value + alignment - 1) / alignment * alignment; }

struct TileSpec {
    int n;     // M 块总数。
    int base;  // 每块基础粒度数。
    int extra; // 额外分配一个粒度的前部块数。
    int gran;  // 每个粒度包含的空间位置数。
};

// 均衡划分 M 轴；total 为位置总数，maxRows 为单块位置上限，gran 为块长粒度。
FC2D_GEOM_CE TileSpec MakeTileSpec(int total, int maxRows, int gran)
{
    TileSpec tiles = {0, 0, 0, gran};
    if (total <= 0 || gran <= 0 || maxRows < gran || (total % gran) != 0) {
        return tiles;
    }
    const int totalUnits = total / gran;
    const int maxTileUnits = maxRows / gran;
    const int tileCount = CeilDiv(totalUnits, maxTileUnits);
    if (tileCount > MAX_M_TILES) {
        return tiles;
    }
    tiles.n = tileCount;
    tiles.base = totalUnits / tileCount;
    tiles.extra = totalUnits % tileCount;
    return tiles;
}

// 返回指定块的长度；tiles 为均衡划分，index 为块索引。
FC2D_GEOM_CE int TileRows(const TileSpec& tiles, int index)
{
    return (tiles.base + (index < tiles.extra ? 1 : 0)) * tiles.gran;
}
// 返回指定块的起点；tiles 为均衡划分，index 为块索引。
FC2D_GEOM_CE int TileRow0(const TileSpec& tiles, int index)
{
    return (tiles.base * index + MinI(index, tiles.extra)) * tiles.gran;
}

struct FmLoadSpec {
    int n;      // 输入通道搬入批次数。
    int firstC; // 首批搬入的通道数。
};

// 选择输入分批搬入；inputChannels/channelBlock 为通道数/分形，firstTileK 为首个 K 块长度，
// firstWeightSegments 为权重段数，columnSegmentCount 为输出列段数，firstMTileCount 为第一层 M 块数。
FC2D_GEOM_CE FmLoadSpec MakeFmLoadSpec(int inputChannels, int channelBlock, int firstTileK, int firstWeightSegments,
                                       int columnSegmentCount, int firstMTileCount)
{
    FmLoadSpec loadSpec = {1, inputChannels};
    if (columnSegmentCount > 1 && firstWeightSegments == 1 && firstMTileCount == 1 && firstTileK >= channelBlock &&
        firstTileK < inputChannels && (firstTileK % channelBlock) == 0) {
        loadSpec.n = 2;
        loadSpec.firstC = firstTileK;
    }
    return loadSpec;
}

struct FlatGeom {
    int n;               // 批次数。
    int ci;              // 输入通道数。
    int hi;              // 输入高度。
    int wi;              // 输入宽度。
    int cout1;           // 第一层输出通道数。
    int cout2;           // 第二层输出通道数。
    int kh1;             // 第一层卷积核高度。
    int kw1;             // 第一层卷积核宽度。
    int kh2;             // 第二层卷积核高度。
    int kw2;             // 第二层卷积核宽度。
    int stride1;         // 第一层空间步长。
    int stride2;         // 第二层空间步长。
    int padH1;           // 第一层上下对称补边。
    int padW1;           // 第一层左右对称补边。
    int padH2;           // 第二层上下对称补边。
    int padW2;           // 第二层左右对称补边。
    int elemBytes;       // 输入元素字节数。
    int weightElemBytes; // 权重元素字节数。
    int midElemBytes;    // 中间结果元素字节数。
    int biasElemBytes;   // 偏置元素字节数。
    int c0;              // 输入通道分形大小。
    int weightC0;        // 权重通道分形大小。
    int midC0;           // 中间结果通道分形大小。
    int ho1;             // 第一层输出高度。
    int wo1;             // 第一层输出宽度。
    int ho2;             // 第二层输出高度。
    int wo2;             // 第二层输出宽度。
    int k1;              // 第一层完整归约 K 长度。
    int k2;              // 第二层完整归约 K 长度。
    int hb;              // 每个输出行带的行数。
    int nchunk;          // 每个批次的行带数。
    int chunkTotal;      // 全部输出块数。
    int chunksPerCore;   // 每核分配的连续块数。
    int wseg;            // 每个输出列段的宽度。
    int nwseg;           // 每个行带的列段数。
    int midRows;         // 最大中间窗口行数（含补边）。
    int mMax1;           // 第一层 M 块位置数上限。
    int mMax2;           // 第二层 M 块位置数上限。
    int tileK1;          // 第一层单次矩阵乘 K 长度。
    int tileK2;          // 第二层单次矩阵乘 K 长度。
    int l0bChunks1;      // 第一层 L0B 权重段数。
    int l0bChunkK1;      // 第一层每个 L0B 段 K 长度。
    int tilesPerChunk1;  // 第一层每段矩阵乘次数。
    int l0bChunks2;      // 第二层 L0B 权重段数。
    int l0bChunkK2;      // 第二层每个 L0B 段 K 长度。
    int tilesPerChunk2;  // 第二层每段矩阵乘次数。
    int l0bPP1;          // 第一层是否交替使用 L0B 槽。
    int l0bPP2;          // 第二层是否交替使用 L0B 槽。
    int l0bSlot1Elems;   // 第一层 L0B 单槽元素数。
    int l0bSlot2Elems;   // 第二层 L0B 单槽元素数。
    int l0bElems;        // L0B 总元素容量。
    int l0cElems;        // L0C 总累加元素容量。
    int l0cStride;       // 第一层 L0C 交替槽元素跨距（0 为单槽）。
    int w1Stage;         // 第一层权重 L1 策略（0 完整区，1 环形区）。
    int w2Stage;         // 第二层权重 L1 策略（0 完整区，1 环形区）。
    int fmAddr;          // 输入 L1 字节地址。
    int fmElems;         // 输入 L1 元素容量。
    int midAddr;         // 中间结果 L1 字节地址。
    int midElems;        // 中间结果 L1 元素容量。
    int w1Addr;          // 第一层权重 L1 字节地址。
    int w1Elems;         // 第一层权重 L1 元素容量。
    int w2Addr;          // 第二层权重 L1 字节地址。
    int w2Elems;         // 第二层权重 L1 元素容量。
    int b1Addr;          // 第一层偏置 L1 字节地址。
    int b2Addr;          // 第二层偏置 L1 字节地址。
    int dq2Addr;         // 第二层反量化表 L1 字节地址。
    int q1Addr;          // 第一层重量化表 L1 字节地址。
    int q2Addr;          // 第二层重量化表 L1 字节地址。
    int xPlane;          // 输入单个批次元素跨度。
    int yPlane;          // 输出单个批次元素跨度。
    int yPlaneM;         // 输出单通道空间元素跨度。
};

#define FC2D_FLAT_FIELDS(X)             \
    X(n, batch)                         \
    X(ci, cin)                          \
    X(hi, hin)                          \
    X(wi, win)                          \
    X(cout1, cout1)                     \
    X(cout2, cout2)                     \
    X(kh1, kh1)                         \
    X(kw1, kw1)                         \
    X(kh2, kh2)                         \
    X(kw2, kw2)                         \
    X(stride1, stride1)                 \
    X(stride2, stride2)                 \
    X(padH1, padH1)                     \
    X(padW1, padW1)                     \
    X(padH2, padH2)                     \
    X(padW2, padW2)                     \
    X(elemBytes, elemBytes)             \
    X(weightElemBytes, weightElemBytes) \
    X(midElemBytes, midElemBytes)       \
    X(biasElemBytes, biasElemBytes)     \
    X(c0, c0)                           \
    X(weightC0, weightC0)               \
    X(midC0, midC0)                     \
    X(ho1, ho1)                         \
    X(wo1, wo1)                         \
    X(ho2, ho2)                         \
    X(wo2, wo2)                         \
    X(k1, k1)                           \
    X(k2, k2)                           \
    X(hb, hb)                           \
    X(nchunk, nchunk)                   \
    X(wseg, wseg)                       \
    X(nwseg, nwseg)                     \
    X(chunkTotal, chunkTotal)           \
    X(chunksPerCore, chunksPerCore)     \
    X(midRows, midRows)                 \
    X(mMax1, mMax1)                     \
    X(mMax2, mMax2)                     \
    X(tileK1, tileK1)                   \
    X(tileK2, tileK2)                   \
    X(l0bChunks1, l0bChunks1)           \
    X(l0bChunkK1, l0bChunkK1)           \
    X(tilesPerChunk1, tilesPerChunk1)   \
    X(l0bChunks2, l0bChunks2)           \
    X(l0bChunkK2, l0bChunkK2)           \
    X(tilesPerChunk2, tilesPerChunk2)   \
    X(l0bPP1, l0bPP1)                   \
    X(l0bPP2, l0bPP2)                   \
    X(l0bSlot1Elems, l0bSlot1Elems)     \
    X(l0bSlot2Elems, l0bSlot2Elems)     \
    X(l0bElems, l0bElems)               \
    X(l0cElems, l0cElems)               \
    X(l0cStride, l0cStride)             \
    X(w1Stage, w1Stage)                 \
    X(w2Stage, w2Stage)                 \
    X(fmAddr, fmAddr)                   \
    X(fmElems, fmElems)                 \
    X(midAddr, midAddr)                 \
    X(midElems, midElems)               \
    X(w1Addr, w1Addr)                   \
    X(w1Elems, w1Elems)                 \
    X(w2Addr, w2Addr)                   \
    X(w2Elems, w2Elems)                 \
    X(b1Addr, b1Addr)                   \
    X(b2Addr, b2Addr)                   \
    X(dq2Addr, dq2Addr)                 \
    X(q1Addr, q1Addr)                   \
    X(q2Addr, q2Addr)                   \
    X(xPlane, xPlane)                   \
    X(yPlane, yPlane)                   \
    X(yPlaneM, yPlaneM)

struct BandLite {
    int aMid;           // 有效中间窗口起始行。
    int midReal;        // 有效中间窗口行数。
    int padT2;          // 第二层窗口上补边。
    int padB2;          // 第二层窗口下补边。
    int xRow0;          // 原始输入窗口起始行。
    int xRows;          // 原始输入有效行数。
    int padT1;          // 第一层窗口上补边。
    int padB1;          // 第一层窗口下补边。
    int outRow0;        // 输出行带起始行。
    int outCol0;        // 输出列段起始列。
    int outCols;        // 输出列段有效宽度。
    int midCol0;        // 有效中间窗口起始列。
    int wSub;           // 有效中间窗口宽度。
    int padL2;          // 第二层窗口左补边。
    int padR2;          // 第二层窗口右补边。
    int xCol0;          // 原始输入窗口起始列。
    int wIn;            // 原始输入有效宽度。
    int padL1;          // 第一层窗口左补边。
    int padR1;          // 第一层窗口右补边。
    int conv1MPerBand;  // 当前第一层展开矩阵 M 长度。
    int rowsPerRun;     // 第二层单次连续行组行数。
    int mRun;           // 第二层单次行组 M 长度。
    int nRun;           // 第二层连续行组数。
    TileSpec conv1Tile; // 第一层 M 块均衡划分。
    TileSpec conv2Tile; // 第二层 M 块均衡划分。
};

// 反推当前输入窗口及补边；geometry 为整体几何，rowBand/columnSegment 为输出块坐标，window 为计算结果。
FC2D_GEOM_FN void MakeBandLite(const FlatGeom& geometry, int rowBand, int columnSegment, BandLite& window)
{
    window.outRow0 = rowBand * geometry.hb;

    // 卷积反向窗口为 stride×首位置-pad，到 stride×末位置-pad+kernel；越界部分交给 Load3D 补零。
    const int rawMidRow = geometry.stride2 * geometry.hb * rowBand - geometry.padH2;
    window.aMid = MaxI(0, rawMidRow);
    window.padT2 = window.aMid - rawMidRow;
    const int requiredMidRows = geometry.midRows - window.padT2;
    window.midReal = MinI(requiredMidRows, geometry.ho1 - window.aMid);
    window.padB2 = geometry.midRows - window.padT2 - window.midReal;

    const int rawInputRow = geometry.stride1 * window.aMid - geometry.padH1;
    const int requiredInputRows = geometry.stride1 * (window.midReal - 1) + geometry.kh1;
    window.xRow0 = MaxI(0, rawInputRow);
    window.padT1 = window.xRow0 - rawInputRow;
    const int inputRowEnd = MinI(geometry.hi, rawInputRow + requiredInputRows);
    window.xRows = inputRowEnd - window.xRow0;
    window.padB1 = rawInputRow + requiredInputRows - inputRowEnd;

    window.outCol0 = columnSegment * geometry.wseg;
    window.outCols = MinI(geometry.wseg, geometry.wo2 - window.outCol0);

    const int rawMidColumnBegin = window.outCol0 * geometry.stride2 - geometry.padW2;
    const int rawMidColumnEnd = (window.outCol0 + window.outCols - 1) * geometry.stride2 - geometry.padW2 +
                                geometry.kw2;
    window.midCol0 = MaxI(0, rawMidColumnBegin);
    const int midColumnEnd = MinI(geometry.wo1, rawMidColumnEnd);
    window.padL2 = window.midCol0 - rawMidColumnBegin;
    window.padR2 = rawMidColumnEnd - midColumnEnd;
    window.wSub = midColumnEnd - window.midCol0;

    const int rawInputColumnBegin = window.midCol0 * geometry.stride1 - geometry.padW1;
    const int rawInputColumnEnd = (midColumnEnd - 1) * geometry.stride1 - geometry.padW1 + geometry.kw1;
    window.xCol0 = MaxI(0, rawInputColumnBegin);
    const int inputColumnEnd = MinI(geometry.wi, rawInputColumnEnd);
    window.padL1 = window.xCol0 - rawInputColumnBegin;
    window.padR1 = rawInputColumnEnd - inputColumnEnd;
    window.wIn = inputColumnEnd - window.xCol0;

    window.conv1MPerBand = window.midReal * window.wSub;
    window.conv1Tile = MakeTileSpec(window.conv1MPerBand, geometry.mMax1, 1);

    // 完整输出宽度可连续跨行；切列后按单行执行，避免把行间空隙误当作连续 M。
    window.rowsPerRun = (window.outCols == geometry.wo2) ? geometry.hb : 1;
    window.mRun = window.rowsPerRun * window.outCols;
    window.nRun = geometry.hb / window.rowsPerRun;
    window.conv2Tile = MakeTileSpec(window.mRun, geometry.mMax2, 1);
}

} // namespace FusedMultiConv2dShape

#endif
