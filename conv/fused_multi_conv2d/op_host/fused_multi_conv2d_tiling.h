/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_MULTI_CONV2D_TILING_H
#define FUSED_MULTI_CONV2D_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FusedMultiConv2dTilingData)

TILING_DATA_FIELD_DEF(uint32_t, batch); // 输入批次数
TILING_DATA_FIELD_DEF(uint32_t, cin);   // 输入通道数
TILING_DATA_FIELD_DEF(uint32_t, hin);   // 输入高度
TILING_DATA_FIELD_DEF(uint32_t, win);   // 输入宽度
TILING_DATA_FIELD_DEF(uint32_t, cout1); // 第一层输出通道数
TILING_DATA_FIELD_DEF(uint32_t, cout2); // 第二层输出通道数

TILING_DATA_FIELD_DEF(uint32_t, kh1);     // 第一层卷积核高度
TILING_DATA_FIELD_DEF(uint32_t, kw1);     // 第一层卷积核宽度
TILING_DATA_FIELD_DEF(uint32_t, kh2);     // 第二层卷积核高度
TILING_DATA_FIELD_DEF(uint32_t, kw2);     // 第二层卷积核宽度
TILING_DATA_FIELD_DEF(uint32_t, stride1); // 第一层空间步长
TILING_DATA_FIELD_DEF(uint32_t, stride2); // 第二层空间步长
TILING_DATA_FIELD_DEF(uint32_t, padH1);   // 第一层上下补边
TILING_DATA_FIELD_DEF(uint32_t, padW1);   // 第一层左右补边
TILING_DATA_FIELD_DEF(uint32_t, padH2);   // 第二层上下补边
TILING_DATA_FIELD_DEF(uint32_t, padW2);   // 第二层左右补边

TILING_DATA_FIELD_DEF(uint32_t, elemBytes);       // 输入元素字节数
TILING_DATA_FIELD_DEF(uint32_t, weightElemBytes); // 权重元素字节数
TILING_DATA_FIELD_DEF(uint32_t, midElemBytes);    // 中间元素字节数
TILING_DATA_FIELD_DEF(uint32_t, biasElemBytes);   // 偏置元素字节数

TILING_DATA_FIELD_DEF(uint32_t, c0);       // 输入通道块大小
TILING_DATA_FIELD_DEF(uint32_t, weightC0); // 权重通道块大小
TILING_DATA_FIELD_DEF(uint32_t, midC0);    // 中间通道块大小
TILING_DATA_FIELD_DEF(uint32_t, ho1);      // 第一层输出高度
TILING_DATA_FIELD_DEF(uint32_t, wo1);      // 第一层输出宽度
TILING_DATA_FIELD_DEF(uint32_t, ho2);      // 最终输出高度
TILING_DATA_FIELD_DEF(uint32_t, wo2);      // 最终输出宽度
TILING_DATA_FIELD_DEF(uint32_t, k1);       // 第一层归约长度
TILING_DATA_FIELD_DEF(uint32_t, k2);       // 第二层归约长度

TILING_DATA_FIELD_DEF(uint32_t, hb);     // 每个输出行带的行数
TILING_DATA_FIELD_DEF(uint32_t, nchunk); // 每张图的输出行带数

TILING_DATA_FIELD_DEF(uint32_t, wseg);          // 输出列段最大宽度
TILING_DATA_FIELD_DEF(uint32_t, nwseg);         // 每个行带的列段数
TILING_DATA_FIELD_DEF(uint32_t, chunkTotal);    // 全部批次的任务数
TILING_DATA_FIELD_DEF(uint32_t, chunksPerCore); // 每个计算核的任务上限
TILING_DATA_FIELD_DEF(uint32_t, midRows);       // 一个行带所需中间行数

TILING_DATA_FIELD_DEF(uint32_t, mMax1); // 第一层单次计算的 M 上限

TILING_DATA_FIELD_DEF(uint32_t, mMax2);          // 第二层单次计算的 M 上限
TILING_DATA_FIELD_DEF(uint32_t, tileK1);         // 第一层单次计算的 K 长度
TILING_DATA_FIELD_DEF(uint32_t, tileK2);         // 第二层单次计算的 K 长度
TILING_DATA_FIELD_DEF(uint32_t, l0bChunks1);     // 第一层权重的 L0B 段数
TILING_DATA_FIELD_DEF(uint32_t, l0bChunkK1);     // 第一层每段权重的 K 长度
TILING_DATA_FIELD_DEF(uint32_t, tilesPerChunk1); // 第一层每个权重段内的计算次数
TILING_DATA_FIELD_DEF(uint32_t, l0bChunks2);     // 第二层权重的 L0B 段数
TILING_DATA_FIELD_DEF(uint32_t, l0bChunkK2);     // 第二层每段权重的 K 长度
TILING_DATA_FIELD_DEF(uint32_t, tilesPerChunk2); // 第二层每个权重段内的计算次数
TILING_DATA_FIELD_DEF(uint32_t, l0bPP1);         // 第一层 L0B 双缓冲标志
TILING_DATA_FIELD_DEF(uint32_t, l0bPP2);         // 第二层 L0B 双缓冲标志
TILING_DATA_FIELD_DEF(uint32_t, l0bSlot1Elems);  // 第一层 L0B 缓冲槽元素间距
TILING_DATA_FIELD_DEF(uint32_t, l0bSlot2Elems);  // 第二层 L0B 缓冲槽元素间距
TILING_DATA_FIELD_DEF(uint32_t, l0bElems);       // L0B 视图元素数
TILING_DATA_FIELD_DEF(uint32_t, l0cElems);       // L0C 视图元素数

TILING_DATA_FIELD_DEF(uint32_t, l0cStride); // 第一层 L0C 缓冲槽元素间距

TILING_DATA_FIELD_DEF(uint32_t, w1Stage); // 第一层权重使用 L1 循环缓存的标志
TILING_DATA_FIELD_DEF(uint32_t, w2Stage); // 第二层权重使用 L1 循环缓存的标志

TILING_DATA_FIELD_DEF(uint32_t, fmAddr);   // L1 输入区域字节地址
TILING_DATA_FIELD_DEF(uint32_t, fmElems);  // L1 输入视图元素数
TILING_DATA_FIELD_DEF(uint32_t, midAddr);  // L1 中间区域字节地址
TILING_DATA_FIELD_DEF(uint32_t, midElems); // L1 中间视图元素数
TILING_DATA_FIELD_DEF(uint32_t, w1Addr);   // L1 第一层权重字节地址
TILING_DATA_FIELD_DEF(uint32_t, w1Elems);  // L1 第一层权重视图元素数
TILING_DATA_FIELD_DEF(uint32_t, w2Addr);   // L1 第二层权重字节地址
TILING_DATA_FIELD_DEF(uint32_t, w2Elems);  // L1 第二层权重视图元素数
TILING_DATA_FIELD_DEF(uint32_t, b1Addr);   // L1 第一层偏置字节地址
TILING_DATA_FIELD_DEF(uint32_t, b2Addr);   // L1 第二层偏置字节地址

TILING_DATA_FIELD_DEF(uint32_t, dq2Addr); // L1 第二层反量化表字节地址

TILING_DATA_FIELD_DEF(uint32_t, q1Addr); // L1 第一层量化表字节地址

TILING_DATA_FIELD_DEF(uint32_t, q2Addr); // L1 第二层量化表字节地址

TILING_DATA_FIELD_DEF(uint32_t, xPlane);  // 输入单批次元素数
TILING_DATA_FIELD_DEF(uint32_t, yPlane);  // 输出单批次元素数
TILING_DATA_FIELD_DEF(uint32_t, yPlaneM); // 输出单通道空间位置数

TILING_DATA_FIELD_DEF(uint32_t, blockDim); // 参与计算的核数

TILING_DATA_FIELD_DEF(uint32_t, fixedShift1); // 第一层定点定标参数
TILING_DATA_FIELD_DEF(uint32_t, fixedShift2); // 第二层定点定标参数

TILING_DATA_FIELD_DEF(uint32_t, hasBias1); // 第一层偏置使能
TILING_DATA_FIELD_DEF(uint32_t, hasBias2); // 第二层偏置使能
TILING_DATA_FIELD_DEF(uint32_t, relu1);    // 第一层 ReLU 使能
TILING_DATA_FIELD_DEF(uint32_t, relu2);    // 第二层 ReLU 使能
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedMultiConv2d, FusedMultiConv2dTilingData)
} // namespace optiling

#endif
