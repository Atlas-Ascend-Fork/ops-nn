/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_MULTI_CONV2D_IMPL_H
#define FUSED_MULTI_CONV2D_IMPL_H

#include <cstdint>

#include "fused_multi_conv2d_tensorutils.h"
#include "fused_multi_conv2d_geometry.h"

namespace FusedMultiConv2d {

namespace S = FusedMultiConv2dShape;

#ifndef DTYPE_X
#define DTYPE_X half
#endif

#ifndef DTYPE_FILTER1
#ifdef ORIG_DTYPE_FILTERS
template <int Code>
struct FilterTypeOf;
template <>
struct FilterTypeOf<DT_FLOAT16> {
    using T = half;
};
template <>
struct FilterTypeOf<DT_INT8> {
    using T = int8_t;
};
template <int Code>
using FilterTypeOfT = typename FilterTypeOf<Code>::T;
#define DTYPE_FILTER1 FilterTypeOfT<ORIG_DTYPE_FILTERS>
#else

#define DTYPE_FILTER1 DTYPE_X
#endif
#endif
#ifndef DTYPE_FILTER2
#define DTYPE_FILTER2 DTYPE_FILTER1
#endif
#ifndef DTYPE_Y
#define DTYPE_Y DTYPE_X
#endif

namespace geom_detail {
template <typename A, typename B>
struct SameT {
    static constexpr bool value = false; // 两个类型是否相同。
};
template <typename A>
struct SameT<A, A> {
    static constexpr bool value = true; // 两个类型是否相同。
};
template <bool C, typename T, typename F>
struct Choose {
    using type = T;
};
template <typename T, typename F>
struct Choose<false, T, F> {
    using type = F;
};
} // namespace geom_detail

template <typename X_, typename W_, typename Mid_, typename Y_>
struct Cfg {
    using XT = X_;
    using FmT = X_;
    using WtT = W_;
    using MidT = Mid_;
    using OutT = Y_;
    using AccT = int32_t;

    static constexpr bool INPUT_IS_FP16 = geom_detail::SameT<X_, half>::value;  // 输入是否为半精度。
    static constexpr bool WEIGHT_IS_FP16 = geom_detail::SameT<W_, half>::value; // 权重是否为半精度。
    static constexpr bool MID_IS_FP16 = geom_detail::SameT<Mid_, half>::value;  // 中间结果是否为半精度。

    static constexpr bool BIAS_IS_FP16 = INPUT_IS_FP16 && WEIGHT_IS_FP16 && MID_IS_FP16; // 偏置是否使用半精度。

    using BiasT = typename geom_detail::Choose<BIAS_IS_FP16, half, int32_t>::type;

    static constexpr int ELEM_BYTES = (int)sizeof(X_);                         // 输入元素字节数。
    static constexpr int WEIGHT_ELEM_BYTES = (int)sizeof(W_);                  // 权重元素字节数。
    static constexpr int MID_ELEM_BYTES = (int)sizeof(Mid_);                   // 中间元素字节数。
    static constexpr int BIAS_ELEM_BYTES = (int)sizeof(BiasT);                 // 偏置元素字节数。
    static constexpr int L0A1_SLOT_ELEMS = S::L0A_SLOT_BYTES / ELEM_BYTES;     // 第一层单个 L0A 槽元素数。
    static constexpr int L0A2_SLOT_ELEMS = S::L0A_SLOT_BYTES / MID_ELEM_BYTES; // 第二层单个 L0A 槽元素数。

    static constexpr bool OUT_IS_FP16 = geom_detail::SameT<Y_, half>::value; // 输出是否为半精度。
    static constexpr bool NEEDS_Q1_TABLE = !MID_IS_FP16;                     // 第一层是否读取重量化表。
    static constexpr bool NEEDS_Q2_TABLE = !OUT_IS_FP16;                     // 第二层是否读取重量化表。

    static constexpr bool NEEDS_DEQ_TABLE = OUT_IS_FP16 && !MID_IS_FP16; // 第二层是否读取反量化表。
    static constexpr bool CONV1_USES_FIX_SHIFT = INPUT_IS_FP16;          // 第一层是否使用矩阵乘定标。
    static constexpr bool CONV2_USES_FIX_SHIFT = MID_IS_FP16;            // 第二层是否使用矩阵乘定标。
};

using DefaultCfg = Cfg<DTYPE_X, DTYPE_FILTER1, DTYPE_FILTER2, DTYPE_Y>;

using XT = DefaultCfg::XT;
using FmT = DefaultCfg::FmT;
using WtT = DefaultCfg::WtT;
using MidT = DefaultCfg::MidT;
using OutT = DefaultCfg::OutT;
using AccT = DefaultCfg::AccT;

using S::FIX_SHIFT_LEN_A16W16;

constexpr int L0A_BYTES = S::L0A_BYTES;
constexpr int BT_BYTES = S::BT_BYTES;

constexpr int L0A_SLOT_BYTES = S::L0A_SLOT_BYTES;
constexpr int L0A_SLOT_ELEMS = DefaultCfg::L0A1_SLOT_ELEMS;
constexpr int L0A_S0_ADDR = 0;
constexpr int L0A_S1_ADDR = L0A_S0_ADDR + L0A_SLOT_BYTES;
constexpr int L0B_ADDR = 0;
constexpr int L0C_ADDR = 0;

constexpr int BT_SLOT_ELEMS = S::BT_SLOT_ELEMS;
constexpr int BT_S0_ADDR = 0;
constexpr int BT_S1_ADDR = BT_S0_ADDR + S::BT_SLOT1_BYTE_OFF;

static_assert(L0A_S1_ADDR + L0A_SLOT_BYTES <= L0A_BYTES, "L0A slots exceed capacity");
static_assert(BT_S1_ADDR + BT_SLOT_ELEMS * 4 <= BT_BYTES, "BT slots exceed capacity");

#ifndef FC2D_FIXED_EVID
#define FC2D_FIXED_EVID 0
#endif

constexpr int EVID_L1FREE0 = 0, EVID_L1FREE1 = 1, EVID_MTE2LAUNCH = 2, EVID_WFREE_A = 3, EVID_WFREE_B = 4;
constexpr int EVID_L1READY0 = 0, EVID_L1READY1_A = 1, EVID_L1READY1_B = 2, EVID_L1READYW2 = 3, EVID_WREADY_A = 4,
              EVID_WREADY_B = 5;

constexpr int EVID_L0FREE0 = 0, EVID_L0BFREE = 1, EVID_L0FREE1_A = 2, EVID_L0FREE1_B = 3, EVID_L0BFREE_B = 4;
constexpr int EVID_L0READY0 = 0, EVID_L0BREADY = 1, EVID_L0READY1_A = 2, EVID_L0READY1_B = 3, EVID_L0BREADY_B = 4;

constexpr int EVID_FIXFREE = 0, EVID_FIXFREE_B = 1, EVID_FIXREADY = 0, EVID_FIXREADY_B = 1;
constexpr int EVID_MIDFREE = 0, EVID_MIDREADY = 0;

constexpr int EVID_MAX_PER_KIND = 8;

static_assert(EVID_L1FREE0 != EVID_L1FREE1 && EVID_L1FREE0 != EVID_MTE2LAUNCH && EVID_L1FREE1 != EVID_MTE2LAUNCH &&
                  EVID_WFREE_A != EVID_WFREE_B && EVID_WFREE_A > EVID_MTE2LAUNCH && EVID_WFREE_B > EVID_MTE2LAUNCH,
              "Duplicate MTE1_MTE2 event IDs");
static_assert(EVID_L1READY0 != EVID_L1READY1_A && EVID_L1READY0 != EVID_L1READY1_B && EVID_L1READY0 != EVID_L1READYW2 &&
                  EVID_L1READY1_A != EVID_L1READY1_B && EVID_L1READY1_A != EVID_L1READYW2 &&
                  EVID_L1READY1_B != EVID_L1READYW2 && EVID_WREADY_A != EVID_WREADY_B &&
                  EVID_WREADY_A > EVID_L1READYW2 && EVID_WREADY_B > EVID_L1READYW2,
              "Duplicate MTE2_MTE1 event IDs");
static_assert(EVID_L0BFREE != EVID_L0FREE1_A && EVID_L0BFREE != EVID_L0FREE1_B && EVID_L0FREE1_A != EVID_L0FREE1_B &&
                  EVID_L0BFREE_B != EVID_L0BFREE && EVID_L0BFREE_B != EVID_L0FREE1_A &&
                  EVID_L0BFREE_B != EVID_L0FREE1_B && EVID_L0BFREE_B < EVID_MAX_PER_KIND,
              "Duplicate M_MTE1 event IDs");
static_assert(EVID_L0BREADY != EVID_L0READY1_A && EVID_L0BREADY != EVID_L0READY1_B &&
                  EVID_L0READY1_A != EVID_L0READY1_B && EVID_L0BREADY_B != EVID_L0BREADY &&
                  EVID_L0BREADY_B != EVID_L0READY1_A && EVID_L0BREADY_B != EVID_L0READY1_B &&
                  EVID_L0BREADY_B < EVID_MAX_PER_KIND,
              "Duplicate MTE1_M event IDs");
static_assert(EVID_FIXFREE != EVID_FIXFREE_B, "Duplicate FIX_M event IDs");
static_assert(EVID_FIXREADY != EVID_FIXREADY_B, "Duplicate M_FIX event IDs");
static_assert(EVID_FIXFREE_B < EVID_MAX_PER_KIND && EVID_FIXREADY_B < EVID_MAX_PER_KIND,
              "Fixpipe event IDs exceed available count");

static_assert(EVID_L1FREE1 < EVID_MAX_PER_KIND && EVID_L1READYW2 < EVID_MAX_PER_KIND &&
                  EVID_L0FREE1_B < EVID_MAX_PER_KIND && EVID_L0READY1_B < EVID_MAX_PER_KIND &&
                  EVID_WFREE_B < EVID_MAX_PER_KIND && EVID_WREADY_B < EVID_MAX_PER_KIND,
              "Event IDs exceed available count");

// 搬入输入窗口；inputL1/inputGlobal 为目标/源，batchOffset 为批次起点，currentBand 为窗口，
// inputHeight/inputWidth 为原图尺寸，channelBlock 为通道分形，channelBegin/channelCount 为通道范围。
template <typename T>
__aicore__ inline void LoadFmChannels(LocalTensor<T> inputL1, GlobalTensor<T> inputGlobal, int batchOffset,
                                      const S::BandLite& currentBand, int inputHeight, int inputWidth, int channelBlock,
                                      int channelBegin, int channelCount)
{
    if (channelCount <= 0) {
        return;
    }
    const int windowPositions = currentBand.xRows * currentBand.wIn;
    const int destinationChannelOffset = channelBegin / channelBlock * windowPositions * channelBlock;
    const int sourceChannelOffset = batchOffset + channelBegin * inputHeight * inputWidth;
    if (currentBand.wIn == inputWidth) {
        GM2L1_DN2NZ(inputL1[destinationChannelOffset],
                    inputGlobal[sourceChannelOffset + currentBand.xRow0 * inputWidth], channelCount,
                    currentBand.xRows * inputWidth, inputHeight * inputWidth, windowPositions);
        return;
    }
    for (int rowIndex = 0; rowIndex < currentBand.xRows; ++rowIndex) {
        GM2L1_DN2NZ(inputL1[destinationChannelOffset + rowIndex * currentBand.wIn * channelBlock],
                    inputGlobal[sourceChannelOffset + (currentBand.xRow0 + rowIndex) * inputWidth + currentBand.xCol0],
                    channelCount, currentBand.wIn, inputHeight * inputWidth, windowPositions);
    }
}

// 推进到下一输出块；rowBandCount/columnSegmentCount 为行带/列段数，batchIndex/rowBandIndex/columnSegmentIndex
// 为原地更新的批次/行带/列段索引。
__aicore__ inline void AdvanceChunk(int rowBandCount, int columnSegmentCount, int& batchIndex, int& rowBandIndex,
                                    int& columnSegmentIndex)
{
    if (++columnSegmentIndex == columnSegmentCount) {
        columnSegmentIndex = 0;
        if (++rowBandIndex == rowBandCount) {
            rowBandIndex = 0;
            ++batchIndex;
        }
    }
}

struct KernelArgs {
    S::FlatGeom geometry; // 主机下发的几何与缓存布局。
    uint8_t fixedShift1;  // 第一层定点定标。
    uint8_t fixedShift2;  // 第二层定点定标。
    bool hasBias1;        // 第一层是否加载偏置。
    bool hasBias2;        // 第二层是否加载偏置。
    bool relu1;           // 第一层是否执行普通 ReLU。
    bool relu2;           // 第二层是否执行普通 ReLU。
};

// 保存同一核跨分块复用的张量、事件和流水计数；C 选择编译期数据通路。
template <typename C>
class KernelExecutor {
    using XT = typename C::XT;
    using FmT = typename C::FmT;
    using WtT = typename C::WtT;
    using MidT = typename C::MidT;
    using OutT = typename C::OutT;
    using AccT = typename C::AccT;
    using BiasT = typename C::BiasT;
    static constexpr bool MID_IS_FP16 = C::MID_IS_FP16;       // 中间结果是否为半精度。
    static constexpr int WEIGHT_BYTES = C::WEIGHT_ELEM_BYTES; // 权重元素字节数。

    struct MatrixTile {
        int tileIndex;                 // 当前 M 块索引。
        int weightSlotParity;          // 当前 M 块首个权重段的 L0B 槽奇偶。
        int positionStart;             // 当前 M 块在展开平面的起点。
        int positionCount;             // 当前 M 块的有效位置数。
        LocalTensor<AccT> accumulator; // 当前 M 块的 L0C 视图。
    };
    TPipe pipeline;              // 当前核流水事件分配器，早于事件成员构造。
    const KernelArgs& arguments; // 主机下发的执行参数。
    const S::FlatGeom& geometry; // 参数中的几何与缓存布局视图。

public:
    // 建立张量视图；inputAddress 为输入，weight1Address/weight2Address 为权重，bias1Address/bias2Address 为偏置，
    // dequant2Address/quant1Address/quant2Address 为定标表，outputAddress 为输出，args 为执行参数。
    __aicore__ inline KernelExecutor(GM_ADDR inputAddress, GM_ADDR weight1Address, GM_ADDR bias1Address,
                                     GM_ADDR weight2Address, GM_ADDR bias2Address, GM_ADDR dequant2Address,
                                     GM_ADDR quant1Address, GM_ADDR quant2Address, GM_ADDR outputAddress,
                                     const KernelArgs& args)
        : arguments(args), geometry(args.geometry)
    {
        InitGeometry();
        InitGlobalTensors(inputAddress, weight1Address, bias1Address, weight2Address, bias2Address, dequant2Address,
                          quant1Address, quant2Address, outputAddress);
        InitLocalTensors();
    }

    // 执行本核连续输出块，并在退出前归还预置事件许可；无参数。
    __aicore__ inline void Run()
    {
        Bootstrap();
        InitTaskRange();
        for (taskIndex = taskBegin; taskIndex < taskEnd; ++taskIndex) {
            if (!BeginBand()) {
                break;
            }
            PrepareFirstLayer();
            ComputeFirstLayer();
            PrefetchNextInput();
            PrepareSecondLayer();
            ComputeSecondLayer();
            FinishBand();
        }
        if (accumulatorSlotCount == 1) {
            accumulatorFree.set();
        }
        initialL1Free.set();
        biasTableFree.set();
    }

private:
    // 绑定 GM 地址；inputAddress 为输入，weight1Address/weight2Address 为权重，bias1Address/bias2Address 为偏置，
    // dequant2Address/quant1Address/quant2Address 为定标表，outputAddress 为输出。
    __aicore__ inline void InitGlobalTensors(GM_ADDR inputAddress, GM_ADDR weight1Address, GM_ADDR bias1Address,
                                             GM_ADDR weight2Address, GM_ADDR bias2Address, GM_ADDR dequant2Address,
                                             GM_ADDR quant1Address, GM_ADDR quant2Address, GM_ADDR outputAddress)
    {
        inputGlobal.SetGlobalBuffer((__gm__ XT*)inputAddress);
        weight1Global.SetGlobalBuffer((__gm__ WtT*)weight1Address);
        bias1Global.SetGlobalBuffer((__gm__ BiasT*)bias1Address);
        weight2Global.SetGlobalBuffer((__gm__ WtT*)weight2Address);
        bias2Global.SetGlobalBuffer((__gm__ BiasT*)bias2Address);

        dequant2Global.SetGlobalBuffer((__gm__ uint64_t*)dequant2Address);
        quant1Global.SetGlobalBuffer((__gm__ uint64_t*)quant1Address);
        quant2Global.SetGlobalBuffer((__gm__ uint64_t*)quant2Address);
        outputGlobal.SetGlobalBuffer((__gm__ OutT*)outputAddress);
    }

    // 保存对齐尺寸和定标参数；无参数。
    __aicore__ inline void InitGeometry()
    {
        alignedCout1 = S::Align(geometry.cout1, S::MMAD_M0);
        alignedCout2 = S::Align(geometry.cout2, S::MMAD_M0);
        weight1SegmentElements = geometry.l0bChunkK1 * alignedCout1;
        landFirstWeightsInPieces = (geometry.w1Stage == 0) && (geometry.l0bChunks1 > 1);
        fixShiftIn1 = C::CONV1_USES_FIX_SHIFT ? arguments.fixedShift1 : 0;
        fixShiftIn2 = C::CONV2_USES_FIX_SHIFT ? arguments.fixedShift2 : 0;

        fixShiftOut1 = C::MID_IS_FP16 ? (uint8_t)(FIX_SHIFT_LEN_A16W16 - (int)arguments.fixedShift1) : 0;
        fixShiftOut2 = C::CONV2_USES_FIX_SHIFT ? (uint8_t)(FIX_SHIFT_LEN_A16W16 - (int)arguments.fixedShift2) : 0;
        biasShift1 = C::BIAS_IS_FP16 ? fixShiftOut1 : 0;
        biasShift2 = C::BIAS_IS_FP16 ? fixShiftOut2 : 0;
    }

    // 按主机分配的字节地址建立 L1/L0/BT 视图；两层 L0A 共用 ping/pong 地址，无参数。
    __aicore__ inline void InitLocalTensors()
    {
        inputL1 = TensorAt<TPosition::A1, FmT>(geometry.fmAddr, geometry.fmElems);
        intermediateL1 = TensorAt<TPosition::A1, MidT>(geometry.midAddr, geometry.midElems);
        weight1L1 = TensorAt<TPosition::A1, WtT>(geometry.w1Addr, geometry.w1Elems);
        weight2L1 = TensorAt<TPosition::A1, WtT>(geometry.w2Addr, geometry.w2Elems);

        constexpr int SEG_TENSOR_ALIGN = 64;
        const int bias1Elements = S::Align(geometry.cout1 * (int)sizeof(BiasT), SEG_TENSOR_ALIGN) / (int)sizeof(BiasT);
        const int bias2Elements = S::Align(geometry.cout2 * (int)sizeof(BiasT), SEG_TENSOR_ALIGN) / (int)sizeof(BiasT);
        const int dequant2Elements = S::Align(geometry.cout2 * (int)sizeof(uint64_t), SEG_TENSOR_ALIGN) /
                                     (int)sizeof(uint64_t);
        bias1L1 = TensorAt<TPosition::A1, BiasT>(geometry.b1Addr, bias1Elements);
        bias2L1 = TensorAt<TPosition::A1, BiasT>(geometry.b2Addr, bias2Elements);

        dequant2L1 = TensorAt<TPosition::A1, uint64_t>(geometry.dq2Addr, dequant2Elements);
        const int quant1Elements = S::Align(geometry.cout1 * (int)sizeof(uint64_t), SEG_TENSOR_ALIGN) /
                                   (int)sizeof(uint64_t);
        quant1L1 = TensorAt<TPosition::A1, uint64_t>(geometry.q1Addr, quant1Elements);
        const int quant2Elements = S::Align(geometry.cout2 * (int)sizeof(uint64_t), SEG_TENSOR_ALIGN) /
                                   (int)sizeof(uint64_t);
        quant2L1 = TensorAt<TPosition::A1, uint64_t>(geometry.q2Addr, quant2Elements);

        inputL0A.InitAt(L0A_S0_ADDR, C::L0A1_SLOT_ELEMS);
        intermediateL0A.InitAt(L0A_S0_ADDR, C::L0A2_SLOT_ELEMS);
        weightL0B = TensorAt<TPosition::B2, WtT>(L0B_ADDR, geometry.l0bElems);
        accumulatorL0C = TensorAt<TPosition::CO1, AccT>(L0C_ADDR, geometry.l0cElems);

        biasTable.InitAt(BT_S0_ADDR, BT_SLOT_ELEMS);
    }

    // 预载常驻权重、偏置与定标表，并将两层偏置写入各自 BT 区域；无参数。
    __aicore__ inline void Bootstrap()
    {
        initialL1Free.wait();

        if (geometry.w1Stage == 0 && !landFirstWeightsInPieces) {
            GM2L1(weight1L1, weight1Global, 1, geometry.w1Elems * WEIGHT_BYTES / 32, 0, 0);
        }

        if (arguments.hasBias1) {
            GM2L1(bias1L1, bias1Global, 1, geometry.cout1 * (int)sizeof(BiasT) / 32, 0, 0);
        }

        if (arguments.hasBias2) {
            GM2L1PAD(bias2L1, bias2Global, 1, geometry.cout2 * (int)sizeof(BiasT), 0, 0);
        }

        if constexpr (C::NEEDS_DEQ_TABLE) {
            GM2L1PAD(dequant2L1, dequant2Global, 1, geometry.cout2 * (int)sizeof(uint64_t), 0, 0);
        }
        if constexpr (C::NEEDS_Q1_TABLE) {
            GM2L1PAD(quant1L1, quant1Global, 1, geometry.cout1 * (int)sizeof(uint64_t), 0, 0);
        }
        if constexpr (C::NEEDS_Q2_TABLE) {
            GM2L1PAD(quant2L1, quant2Global, 1, geometry.cout2 * (int)sizeof(uint64_t), 0, 0);
        }
        initialL1Ready.set();
        initialL1Ready.wait();

        biasTableFree.wait();

        if (arguments.hasBias1) {
            L1_TO_BT(biasTable.get(0), bias1L1, geometry.cout1, biasShift1);
        }
        if (arguments.hasBias2) {
            L1_TO_BT(biasTable.get(1), bias2L1, geometry.cout2, biasShift2);
        }
        biasTableReady.set();
        biasTableReady.wait();

        PipeBarrier<PIPE_ALL>();
    }

    // 解码本核起始分块，并使 L0C 事件许可数与物理缓存槽数一致；无参数。
    __aicore__ inline void InitTaskRange()
    {
        taskBegin = geometry.chunksPerCore * (int)GetBlockIdx();
        taskEnd = Min(geometry.chunkTotal, geometry.chunksPerCore * ((int)GetBlockIdx() + 1));

        const int tasksPerBatch = geometry.nchunk * geometry.nwseg;
        batchIndex = (tasksPerBatch > 0) ? (taskBegin / tasksPerBatch) : 0;
        int batchTaskOffset = taskBegin - batchIndex * tasksPerBatch;
        rowBandIndex = batchTaskOffset / geometry.nwseg;
        columnSegmentIndex = batchTaskOffset - rowBandIndex * geometry.nwseg;

        accumulatorSlotCount = (geometry.l0cStride == 0) ? 1 : 2;
        if (accumulatorSlotCount == 1) {
            accumulatorFree.wait();
            ++accumulatorSlotSequence;
        }
    }

    // 计算当前窗口并等候输入就绪；首块仅按需要搬入首批通道，无参数。
    __aicore__ inline bool BeginBand()
    {
        if (taskIndex > taskBegin) {
            PipeBarrier<PIPE_ALL>();
        }

        S::MakeBandLite(geometry, rowBandIndex, columnSegmentIndex, currentBand);
        if (currentBand.conv1Tile.n == 0) {
            return false;
        }
        inputBatchOffset = batchIndex * geometry.xPlane;
        outputBatchOffset = batchIndex * geometry.yPlane;

        intermediatePositions = currentBand.conv1MPerBand;

        splitFirstInputLoad = false;
        inputLoadSpec = {1, geometry.ci};
        if (taskIndex == taskBegin) {
            inputL1Free.wait();
            inputLoadSpec = S::MakeFmLoadSpec(geometry.ci, geometry.c0, geometry.tileK1, geometry.l0bChunks1,
                                              geometry.nwseg, currentBand.conv1Tile.n);
            splitFirstInputLoad = inputLoadSpec.n == 2;
            LoadFmChannels(inputL1, inputGlobal, inputBatchOffset, currentBand, geometry.hi, geometry.wi, geometry.c0,
                           0, inputLoadSpec.firstC);
            inputL1Ready.set();
        }

        inputL1Ready.wait();

        return true;
    }

    // 逐段补齐路径先搬两段到独立 L1 地址，后续段随消费继续补齐；无参数。
    __aicore__ inline void PrimeFirstWeights()
    {
        if (landFirstWeightsInPieces && weight1LandingSent == 0) {
            while (weight1LandingSent < 2) {
                weightSegmentFree.wait();
                GM2L1(weight1L1[weight1LandingSent * weight1SegmentElements],
                      weight1Global[weight1LandingSent * weight1SegmentElements], 1,
                      weight1SegmentElements * WEIGHT_BYTES / 32, 0, 0);
                weightSegmentReady.set();
                ++weight1LandingSent;
            }
        }
    }

    // 准备第一层 Load3D 和中间结果写区；多次矩阵乘时将第二层整套权重延后搬入，无参数。
    __aicore__ inline void PrepareFirstLayer()
    {
        PrimeFirstWeights();
        inputChannelsPending = false;

        deferSecondWeightLoad = geometry.w2Stage == 0 && taskIndex == taskBegin &&
                                currentBand.conv1Tile.n * geometry.l0bChunks1 * geometry.tilesPerChunk1 > 1;
        if (geometry.w2Stage == 0 && taskIndex == taskBegin && !deferSecondWeightLoad) {
            GM2L1(weight2L1, weight2Global, 1, geometry.w2Elems * WEIGHT_BYTES / 32, 0, 0);
            weight2L1Ready.set();
        }

        intermediateFree.wait();

        LOAD3D_BEGIN<FmT>(load3dCtx1, currentBand.xRows, currentBand.wIn, geometry.ci, geometry.kh1, geometry.kw1,
                          currentBand.padL1, currentBand.padR1, currentBand.padT1, currentBand.padB1, geometry.stride1,
                          geometry.stride1, 1, 1);
        PrimeFirstWeightStream();
    }

    // 环形路径预填两段权重；一段即可容纳全权重时在当前层开始时搬入 L0B 一次，无参数。
    __aicore__ inline void PrimeFirstWeightStream()
    {
        weight1StreamTotal = (geometry.w1Stage != 0) ? currentBand.conv1Tile.n * geometry.l0bChunks1 : 0;
        weight1StreamSent = 0;
        weight1StreamConsumed = 0;
        while (weight1StreamSent < 2 && weight1StreamSent < weight1StreamTotal) {
            weightSegmentFree.wait();
            GM2L1(weight1L1[(weight1StreamSent & 1) * weight1SegmentElements],
                  weight1Global[(weight1StreamSent % geometry.l0bChunks1) * weight1SegmentElements], 1,
                  weight1SegmentElements * WEIGHT_BYTES / 32, 0, 0);
            weightSegmentReady.set();
            ++weight1StreamSent;
        }

        weight1Resident = (geometry.l0bChunks1 == 1);
        if (weight1Resident) {
            weightL0BFree.wait();
            L0NZ2NZ(weightL0B, weight1L1, alignedCout1, geometry.k1, alignedCout1, geometry.k1);
            weightL0BReady.set();
            weightL0BReady.wait();
        }

        firstLoad3D = true;
    }

    // 逐个 M 块累计第一层全部 K 段，并立即将完成块写入 L1；无参数。
    __aicore__ inline void ComputeFirstLayer()
    {
        int segmentBase = 0;
        for (int tileIdx = 0; tileIdx < currentBand.conv1Tile.n; ++tileIdx) {
            MatrixTile tile;
            tile.tileIndex = tileIdx;
            tile.weightSlotParity = segmentBase & 1;
            segmentBase += geometry.l0bChunks1;
            tile.positionStart = S::TileRow0(currentBand.conv1Tile, tileIdx);
            tile.positionCount = S::TileRows(currentBand.conv1Tile, tileIdx);
            tile.accumulator = accumulatorL0C[(accumulatorSlotSequence & 1) * geometry.l0cStride];
            ComputeFirstTile(tile);
            StoreIntermediate(tile);
        }
        if (weight1Resident) {
            weightL0BFree.set();
        }
    }

    // 累计一个 M 块；tile 为块的位置、长度、L0B 起始槽和 L0C 视图。
    __aicore__ inline void ComputeFirstTile(const MatrixTile& tile)
    {
        const int weightLoadAhead = geometry.l0bPP1 != 0 ? 1 : 0;
        const bool singleWeightSlot = geometry.l0bPP1 == 0 && !weight1Resident;
        if (singleWeightSlot) {
            weightL0BFree.wait();
        }
        // 双槽路径先装下一段再计算上一段，尾部额外迭代排空计算。
        for (int loadSegmentIndex = 0; loadSegmentIndex < geometry.l0bChunks1 + weightLoadAhead; ++loadSegmentIndex) {
            if (!weight1Resident && loadSegmentIndex < geometry.l0bChunks1) {
                LoadFirstWeightSegment(loadSegmentIndex, tile.weightSlotParity, weightLoadAhead);
            }
            const int weightSegmentIndex = loadSegmentIndex - weightLoadAhead;
            if (weightSegmentIndex < 0) {
                continue;
            }
            if (weightLoadAhead != 0 && !weight1Resident) {
                weightL0BReady.wait();
            }
            for (int kTileIdx = 0; kTileIdx < geometry.tilesPerChunk1; ++kTileIdx) {
                ComputeFirstKTile(tile, weightSegmentIndex, kTileIdx);
            }
            if (!weight1Resident) {
                weightL0BFree.set();
            }
        }
        if (singleWeightSlot) {
            weightL0BFree.set();
        }
    }

    // 读取已落入完整 L1 区的权重；loadSegmentIndex 为 K 段号，weightSlotParity 为当前 M 块的 L0B 首槽。
    __aicore__ inline void LoadFirstLandedSegment(int loadSegmentIndex, int weightSlotParity)
    {
        const bool landing = landFirstWeightsInPieces && (weight1LandingConsumed < geometry.l0bChunks1);
        if (landing) {
            weightSegmentReady.wait();
            ++weight1LandingConsumed;
        }
        L0NZ2NZ(weightL0B[((loadSegmentIndex + weightSlotParity) & 1) * geometry.l0bSlot1Elems],
                weight1L1[loadSegmentIndex * weight1SegmentElements], alignedCout1, geometry.l0bChunkK1, alignedCout1,
                geometry.l0bChunkK1);
        if (landing) {
            weightSegmentFree.set();
            if (weight1LandingSent < geometry.l0bChunks1) {
                weightSegmentFree.wait();
                GM2L1(weight1L1[weight1LandingSent * weight1SegmentElements],
                      weight1Global[weight1LandingSent * weight1SegmentElements], 1,
                      weight1SegmentElements * WEIGHT_BYTES / 32, 0, 0);
                weightSegmentReady.set();
                ++weight1LandingSent;
            }
        }
    }

    // 搬入下一 L0B 权重段并补充环形 L1；loadSegmentIndex 为段号，weightSlotParity 为首槽，weightLoadAhead
    // 表示是否提前装载。
    __aicore__ inline void LoadFirstWeightSegment(int loadSegmentIndex, int weightSlotParity, int weightLoadAhead)
    {
        weightL0BFree.wait();
        if (geometry.w1Stage != 0) {
            weightSegmentReady.wait();
            L0NZ2NZ(weightL0B[((loadSegmentIndex + weightSlotParity) & 1) * geometry.l0bSlot1Elems],
                    weight1L1[(weight1StreamConsumed & 1) * weight1SegmentElements], alignedCout1, geometry.l0bChunkK1,
                    alignedCout1, geometry.l0bChunkK1);
            weightSegmentFree.set();
            ++weight1StreamConsumed;
        } else {
            LoadFirstLandedSegment(loadSegmentIndex, weightSlotParity);
        }
        weightL0BReady.set();
        if (weightLoadAhead == 0) {
            weightL0BReady.wait();
        }

        if (geometry.w1Stage != 0 && weight1StreamSent < weight1StreamTotal) {
            weightSegmentFree.wait();
            GM2L1(weight1L1[(weight1StreamSent & 1) * weight1SegmentElements],
                  weight1Global[(weight1StreamSent % geometry.l0bChunks1) * weight1SegmentElements], 1,
                  weight1SegmentElements * WEIGHT_BYTES / 32, 0, 0);
            weightSegmentReady.set();
            ++weight1StreamSent;
        }
    }

    // 在首个 Load3D 完成后启动剩余输入通道及第二层整套权重搬入；firstOperation 表示首个 K 块。
    __aicore__ inline void LaunchDeferredTransfers(bool firstOperation)
    {
        if (firstOperation && (deferSecondWeightLoad || splitFirstInputLoad)) {
            deferredLoadReady.set();
            deferredLoadReady.wait();

            if (splitFirstInputLoad) {
                LoadFmChannels(inputL1, inputGlobal, inputBatchOffset, currentBand, geometry.hi, geometry.wi,
                               geometry.c0, inputLoadSpec.firstC, geometry.ci - inputLoadSpec.firstC);
                inputL1Ready.set();
                inputChannelsPending = true;
            }
            if (deferSecondWeightLoad) {
                GM2L1(weight2L1, weight2Global, 1, geometry.w2Elems * WEIGHT_BYTES / 32, 0, 0);
                weight2L1Ready.set();
            }
        }
    }

    // 搬入并计算一个 K 块；tile 指定 M 块，weightSegmentIndex 为权重段，kTileIdx 为段内 K 块。
    __aicore__ inline void ComputeFirstKTile(const MatrixTile& tile, int weightSegmentIndex, int kTileIdx)
    {
        const int weightSlotOffset = ((weightSegmentIndex + tile.weightSlotParity) & 1) * geometry.l0bSlot1Elems;
        const int segmentKOffset = kTileIdx * geometry.tileK1;
        // 段号给出完整 K 轴起点，段内偏移同时定位 L0A 窗口与 L0B 权重。
        const int globalKOffset = weightSegmentIndex * geometry.l0bChunkK1 + segmentKOffset;
        const bool firstOperation = tile.tileIndex == 0 && weightSegmentIndex == 0 && kTileIdx == 0;

        if (inputChannelsPending && !firstOperation) {
            inputL1Ready.wait();
            inputChannelsPending = false;
        }

        if (!firstLoad3D) {
            inputL0AFree.set();
        }
        firstLoad3D = false;
        inputL0AFree.wait();
        const int inputSlot = inputSlotSequence++;

        LOAD3D_RUN<FmT>(load3dCtx1, inputL0A.get(inputSlot), inputL1, globalKOffset, tile.positionStart,
                        geometry.tileK1, tile.positionCount);
        inputL0AReady.set();

        LaunchDeferredTransfers(firstOperation);
        inputL0AReady.wait();
        // 仅首个 K 块清零或注入一次偏置，后续块保持同一 L0C 累加。
        if (weightSegmentIndex == 0 && kTileIdx == 0) {
            accumulatorFree.wait();
            ++accumulatorSlotSequence;
            if (arguments.hasBias1) {
                MMAD_BIAS(tile.accumulator, inputL0A.get(inputSlot),
                          weightL0B[weightSlotOffset + segmentKOffset * alignedCout1], biasTable.get(0),
                          tile.positionCount, geometry.tileK1, alignedCout1, fixShiftIn1);
            } else {
                MMAD(tile.accumulator, inputL0A.get(inputSlot),
                     weightL0B[weightSlotOffset + segmentKOffset * alignedCout1], tile.positionCount, geometry.tileK1,
                     alignedCout1, true, fixShiftIn1);
            }
        } else {
            MMAD(tile.accumulator, inputL0A.get(inputSlot), weightL0B[weightSlotOffset + segmentKOffset * alignedCout1],
                 tile.positionCount, geometry.tileK1, alignedCout1, false, fixShiftIn1);
        }
    }

    // 将第一层完成块转为中间类型；tile 为累加块，NZ 通道跨距始终使用整个中间平面。
    __aicore__ inline void StoreIntermediate(const MatrixTile& tile)
    {
        accumulatorReady.set();
        accumulatorReady.wait();
        if constexpr (C::NEEDS_Q1_TABLE) {
            L0C2L1_VREQ8(intermediateL1[tile.positionStart * geometry.midC0], tile.accumulator, quant1L1,
                         tile.positionCount, geometry.cout1, intermediatePositions, tile.positionCount, arguments.relu1,
                         0, intermediatePositions);
        } else if constexpr (MID_IS_FP16) {
            L0C2L1_DEQF16(intermediateL1[tile.positionStart * geometry.midC0], tile.accumulator, tile.positionCount,
                          geometry.cout1, intermediatePositions, tile.positionCount, fixShiftOut1, arguments.relu1, 0,
                          intermediatePositions);
        }
        accumulatorFree.set();
    }

    // 第一层已不再读 fmap 时复用其单块 L1 区，预取下一输出块输入；无参数。
    __aicore__ inline void PrefetchNextInput()
    {
        inputL1Free.set();

        if (taskIndex + 1 < taskEnd) {
            int nextBatch = batchIndex;
            int nextRowBand = rowBandIndex;
            int nextColumnSegment = columnSegmentIndex;
            AdvanceChunk(geometry.nchunk, geometry.nwseg, nextBatch, nextRowBand, nextColumnSegment);
            S::BandLite nextBand;
            S::MakeBandLite(geometry, nextRowBand, nextColumnSegment, nextBand);

            if (nextBand.conv1Tile.n != 0) {
                inputL1Free.wait();
                LoadFmChannels(inputL1, inputGlobal, nextBatch * geometry.xPlane, nextBand, geometry.hi, geometry.wi,
                               geometry.c0, 0, geometry.ci);
                inputL1Ready.set();
            }
        }
    }

    // 等候中间结果和权重，准备第二层窗口及权重流水；无参数。
    __aicore__ inline void PrepareSecondLayer()
    {
        intermediateReady.set();
        intermediateReady.wait();

        if (geometry.w2Stage == 0 && taskIndex == taskBegin) {
            weight2L1Ready.wait();
        }

        LOAD3D_BEGIN<MidT>(load3dCtx2, currentBand.midReal, currentBand.wSub, geometry.cout1, geometry.kh2,
                           geometry.kw2, currentBand.padL2, currentBand.padR2, currentBand.padT2, currentBand.padB2,
                           geometry.stride2, geometry.stride2, 1, 1);

        weight2SegmentElements = geometry.l0bChunkK2 * alignedCout2;
        weight2StreamTotal = (geometry.w2Stage != 0) ?
                                 currentBand.nRun * currentBand.conv2Tile.n * geometry.l0bChunks2 :
                                 0;
        weight2StreamSent = 0;
        weight2StreamConsumed = 0;
        while (weight2StreamSent < 2 && weight2StreamSent < weight2StreamTotal) {
            weightSegmentFree.wait();
            GM2L1(weight2L1[(weight2StreamSent & 1) * weight2SegmentElements],
                  weight2Global[(weight2StreamSent % geometry.l0bChunks2) * weight2SegmentElements], 1,
                  weight2SegmentElements * WEIGHT_BYTES / 32, 0, 0);
            weightSegmentReady.set();
            ++weight2StreamSent;
        }

        weight2Resident = (geometry.l0bChunks2 == 1);
        if (weight2Resident) {
            weightL0BFree.wait();
            L0NZ2NZ(weightL0B, weight2L1, alignedCout2, geometry.l0bChunkK2, alignedCout2, geometry.l0bChunkK2);
            weightL0BReady.set();
            weightL0BReady.wait();
        }
    }

    // 按 runIndex 和 M 块执行第二层，每块全部 K 累加完成后写回 GM；无参数。
    __aicore__ inline void ComputeSecondLayer()
    {
        int segmentBase = 0;
        for (int runIndex = 0; runIndex < currentBand.nRun; ++runIndex) {
            for (int tileIdx = 0; tileIdx < currentBand.conv2Tile.n; ++tileIdx) {
                MatrixTile tile;
                tile.tileIndex = tileIdx;
                tile.weightSlotParity = segmentBase & 1;
                segmentBase += geometry.l0bChunks2;
                tile.positionStart = S::TileRow0(currentBand.conv2Tile, tileIdx);
                tile.positionCount = S::TileRows(currentBand.conv2Tile, tileIdx);
                tile.accumulator = accumulatorL0C;
                ComputeSecondTile(tile, runIndex);
                StoreOutput(tile, runIndex);
            }
        }
    }

    // 累计第二层一个 M 块；tile 指定块信息，runIndex 指定当前连续输出行组。
    __aicore__ inline void ComputeSecondTile(const MatrixTile& tile, int runIndex)
    {
        const int weightLoadAhead = geometry.l0bPP2 != 0 ? 1 : 0;
        const bool singleWeightSlot = geometry.l0bPP2 == 0 && !weight2Resident;
        if (singleWeightSlot) {
            weightL0BFree.wait();
        }
        for (int loadSegmentIndex = 0; loadSegmentIndex < geometry.l0bChunks2 + weightLoadAhead; ++loadSegmentIndex) {
            if (!weight2Resident && loadSegmentIndex < geometry.l0bChunks2) {
                LoadSecondWeightSegment(loadSegmentIndex, tile.weightSlotParity, weightLoadAhead);
            }
            const int weightSegmentIndex = loadSegmentIndex - weightLoadAhead;
            if (weightSegmentIndex < 0) {
                continue;
            }
            if (weightLoadAhead != 0 && !weight2Resident) {
                weightL0BReady.wait();
            }
            for (int kTileIdx = 0; kTileIdx < geometry.tilesPerChunk2; ++kTileIdx) {
                ComputeSecondKTile(tile, runIndex, weightSegmentIndex, kTileIdx);
            }
            if (!weight2Resident) {
                weightL0BFree.set();
            }
        }
        if (singleWeightSlot) {
            weightL0BFree.set();
        }
    }

    // 装入第二层 L0B 段并补充环形 L1；loadSegmentIndex 为 K 段号，weightSlotParity 为首槽，weightLoadAhead
    // 为装载提前量。
    __aicore__ inline void LoadSecondWeightSegment(int loadSegmentIndex, int weightSlotParity, int weightLoadAhead)
    {
        weightL0BFree.wait();
        if (geometry.w2Stage != 0) {
            weightSegmentReady.wait();
            L0NZ2NZ(weightL0B[((loadSegmentIndex + weightSlotParity) & 1) * geometry.l0bSlot2Elems],
                    weight2L1[(weight2StreamConsumed & 1) * weight2SegmentElements], alignedCout2, geometry.l0bChunkK2,
                    alignedCout2, geometry.l0bChunkK2);
            weightSegmentFree.set();
            ++weight2StreamConsumed;
        } else {
            L0NZ2NZ(weightL0B[((loadSegmentIndex + weightSlotParity) & 1) * geometry.l0bSlot2Elems],
                    weight2L1[loadSegmentIndex * weight2SegmentElements], alignedCout2, geometry.l0bChunkK2,
                    alignedCout2, geometry.l0bChunkK2);
        }
        weightL0BReady.set();
        if (weightLoadAhead == 0) {
            weightL0BReady.wait();
        }
        if (geometry.w2Stage != 0 && weight2StreamSent < weight2StreamTotal) {
            weightSegmentFree.wait();
            GM2L1(weight2L1[(weight2StreamSent & 1) * weight2SegmentElements],
                  weight2Global[(weight2StreamSent % geometry.l0bChunks2) * weight2SegmentElements], 1,
                  weight2SegmentElements * WEIGHT_BYTES / 32, 0, 0);
            weightSegmentReady.set();
            ++weight2StreamSent;
        }
    }

    // 搬入并累计第二层 K 块；tile 为 M 块，runIndex 为行组，weightSegmentIndex/kTileIdx 为权重段及段内块。
    __aicore__ inline void ComputeSecondKTile(const MatrixTile& tile, int runIndex, int weightSegmentIndex,
                                              int kTileIdx)
    {
        const int weightSlotOffset = ((weightSegmentIndex + tile.weightSlotParity) & 1) * geometry.l0bSlot2Elems;
        const int segmentKOffset = kTileIdx * geometry.tileK2;
        const int globalKOffset = weightSegmentIndex * geometry.l0bChunkK2 + segmentKOffset;
        if (!firstLoad3D) {
            inputL0AFree.set();
        }
        firstLoad3D = false;
        inputL0AFree.wait();
        const int inputSlot = inputSlotSequence++;

        LOAD3D_RUN<MidT>(load3dCtx2, intermediateL0A.get(inputSlot), intermediateL1, globalKOffset,
                         runIndex * currentBand.mRun + tile.positionStart, geometry.tileK2, tile.positionCount);
        inputL0AReady.set();
        inputL0AReady.wait();
        // 第二层使用 L0C 基址，首次累加前回收第一层所有物理槽。
        if (weightSegmentIndex == 0 && kTileIdx == 0) {
            for (int slotIndex = 0; slotIndex < accumulatorSlotCount; ++slotIndex) {
                accumulatorFree.wait();
                ++accumulatorSlotSequence;
            }
            if (arguments.hasBias2) {
                MMAD_BIAS(accumulatorL0C, intermediateL0A.get(inputSlot),
                          weightL0B[weightSlotOffset + segmentKOffset * alignedCout2], biasTable.get(1),
                          tile.positionCount, geometry.tileK2, alignedCout2, fixShiftIn2);
            } else {
                MMAD(accumulatorL0C, intermediateL0A.get(inputSlot),
                     weightL0B[weightSlotOffset + segmentKOffset * alignedCout2], tile.positionCount, geometry.tileK2,
                     alignedCout2, true, fixShiftIn2);
            }
        } else {
            MMAD(accumulatorL0C, intermediateL0A.get(inputSlot),
                 weightL0B[weightSlotOffset + segmentKOffset * alignedCout2], tile.positionCount, geometry.tileK2,
                 alignedCout2, false, fixShiftIn2);
        }
    }

    // 转换并写回第二层结果；tile 为 M 块，runIndex 为行组，输出通道跨距为完整 H×W。
    __aicore__ inline void StoreOutput(const MatrixTile& tile, int runIndex)
    {
        accumulatorReady.set();
        accumulatorReady.wait();

        // run 起始行和块内 M 偏移共同定位输出平面，通道方向保留整幅图跨度。
        const int outputOffset = outputBatchOffset +
                                 (currentBand.outRow0 + runIndex * currentBand.rowsPerRun) * geometry.wo2 +
                                 currentBand.outCol0 + tile.positionStart;

        if constexpr (C::NEEDS_DEQ_TABLE) {
            L0C2GM_NZ2DN_VDEQF16(outputGlobal[outputOffset], accumulatorL0C, dequant2L1, tile.positionCount,
                                 geometry.cout2, geometry.yPlaneM, tile.positionCount, arguments.relu2);
        } else if constexpr (MID_IS_FP16) {
            L0C2GM_NZ2DN_DEQF16(outputGlobal[outputOffset], accumulatorL0C, tile.positionCount, geometry.cout2,
                                geometry.yPlaneM, tile.positionCount, fixShiftOut2, arguments.relu2);
        } else {
            L0C2GM_NZ2DN_VREQ8(outputGlobal[outputOffset], accumulatorL0C, quant2L1, tile.positionCount, geometry.cout2,
                               geometry.yPlaneM, tile.positionCount, arguments.relu2);
        }
        for (int slotIndex = 0; slotIndex < accumulatorSlotCount; ++slotIndex) {
            accumulatorFree.set();
        }
    }

    // 归还第二层权重、中间结果与 L0A 使用许可，推进分块坐标；无参数。
    __aicore__ inline void FinishBand()
    {
        if (weight2Resident) {
            weightL0BFree.set();
        }

        intermediateFree.set();

        inputL0AFree.set();

        AdvanceChunk(geometry.nchunk, geometry.nwseg, batchIndex, rowBandIndex, columnSegmentIndex);
    }

    SEvent<PIPE_MTE1, PIPE_MTE2, true, EVID_L1FREE0> initialL1Free;         // 初始 L1 装载区许可。
    SEvent<PIPE_MTE2, PIPE_MTE1, false, EVID_L1READY0> initialL1Ready;      // 初始 L1 数据就绪。
    SEvent<PIPE_MTE1, PIPE_MTE2, true, EVID_L1FREE1> inputL1Free;           // 输入窗口复用许可。
    SEvent<PIPE_MTE1, PIPE_MTE2, false, EVID_MTE2LAUNCH> deferredLoadReady; // 首个 Load3D 后启动搬运。
    DEvent<PIPE_MTE2, PIPE_MTE1, false, EVID_L1READY1_A, EVID_L1READY1_B> inputL1Ready; // 输入通道批次就绪。
    SEvent<PIPE_MTE2, PIPE_MTE1, false, EVID_L1READYW2> weight2L1Ready; // 第二层整套权重就绪。
    DEvent<PIPE_MTE2, PIPE_MTE1, false, EVID_WREADY_A, EVID_WREADY_B> weightSegmentReady; // 环形权重段就绪。
    DEvent<PIPE_MTE1, PIPE_MTE2, true, EVID_WFREE_A, EVID_WFREE_B> weightSegmentFree; // 环形权重段复用许可。
    SEvent<PIPE_M, PIPE_MTE1, true, EVID_L0FREE0> biasTableFree;                      // 偏置 BT 区许可。
    SEvent<PIPE_MTE1, PIPE_M, false, EVID_L0READY0> biasTableReady;                   // 偏置 BT 数据就绪。
    DEvent<PIPE_M, PIPE_MTE1, true, EVID_L0FREE1_A, EVID_L0FREE1_B> inputL0AFree;     // L0A ping/pong 复用许可。
    DEvent<PIPE_MTE1, PIPE_M, false, EVID_L0READY1_A, EVID_L0READY1_B> inputL0AReady; // L0A ping/pong 数据就绪。
    DEvent<PIPE_M, PIPE_MTE1, true, EVID_L0BFREE, EVID_L0BFREE_B> weightL0BFree;      // L0B ping/pong 复用许可。
    DEvent<PIPE_MTE1, PIPE_M, false, EVID_L0BREADY, EVID_L0BREADY_B> weightL0BReady;  // L0B ping/pong 数据就绪。
    DEvent<PIPE_FIX, PIPE_M, true, EVID_FIXFREE, EVID_FIXFREE_B> accumulatorFree;     // L0C ping/pong 复用许可。
    DEvent<PIPE_M, PIPE_FIX, false, EVID_FIXREADY, EVID_FIXREADY_B> accumulatorReady; // L0C ping/pong 累加完成。
    SEvent<PIPE_MTE1, PIPE_FIX, true, EVID_MIDFREE> intermediateFree;                 // 中间结果区可写。
    SEvent<PIPE_FIX, PIPE_MTE1, false, EVID_MIDREADY> intermediateReady;              // 中间结果区可读。

    GlobalTensor<XT> inputGlobal;               // 输入 GM 张量。
    GlobalTensor<WtT> weight1Global;            // 第一层权重 GM 张量。
    GlobalTensor<BiasT> bias1Global;            // 第一层偏置 GM 张量。
    GlobalTensor<WtT> weight2Global;            // 第二层权重 GM 张量。
    GlobalTensor<BiasT> bias2Global;            // 第二层偏置 GM 张量。
    GlobalTensor<uint64_t> dequant2Global;      // 第二层反量化表。
    GlobalTensor<uint64_t> quant1Global;        // 第一层重量化表。
    GlobalTensor<uint64_t> quant2Global;        // 第二层重量化表。
    GlobalTensor<OutT> outputGlobal;            // 输出 GM 张量。
    LocalTensor<FmT> inputL1;                   // 当前或预取输入窗口。
    LocalTensor<MidT> intermediateL1;           // 第一层输出的 L1 NZ 平面。
    LocalTensor<WtT> weight1L1;                 // 第一层权重 L1 区。
    LocalTensor<WtT> weight2L1;                 // 第二层权重 L1 区。
    LocalTensor<BiasT> bias1L1;                 // 第一层偏置 L1 区。
    LocalTensor<BiasT> bias2L1;                 // 第二层偏置 L1 区。
    LocalTensor<uint64_t> dequant2L1;           // 第二层反量化表 L1 区。
    LocalTensor<uint64_t> quant1L1;             // 第一层重量化表 L1 区。
    LocalTensor<uint64_t> quant2L1;             // 第二层重量化表 L1 区。
    DBuff<FmT, TPosition::A2> inputL0A;         // 第一层 L0A 类型视图。
    DBuff<MidT, TPosition::A2> intermediateL0A; // 第二层 L0A 类型视图，与第一层共址。
    LocalTensor<WtT> weightL0B;                 // 两层共用权重计算缓存。
    LocalTensor<AccT> accumulatorL0C;           // 两层共用累加缓存。
    DBuff<int32_t, TPosition::C2> biasTable;    // 两层分别使用的 BT 偏置区。
    Load3DCtx<FmT> load3dCtx1;                  // 第一层窗口展开配置。
    Load3DCtx<MidT> load3dCtx2;                 // 第二层窗口展开配置。
    S::BandLite currentBand;                    // 当前输出块的输入窗口。
    S::FmLoadSpec inputLoadSpec;                // 首块通道分批搬入方式。
    int alignedCout1 = 0;                       // 第一层对齐输出通道数。
    int alignedCout2 = 0;                       // 第二层对齐输出通道数。
    int weight1SegmentElements = 0;             // 第一层每个权重段元素数。
    uint8_t fixShiftIn1 = 0;                    // 第一层矩阵乘定标。
    uint8_t fixShiftIn2 = 0;                    // 第二层矩阵乘定标。
    uint8_t fixShiftOut1 = 0;                   // 第一层结果定标。
    uint8_t fixShiftOut2 = 0;                   // 第二层结果定标。
    uint8_t biasShift1 = 0;                     // 第一层偏置定标。
    uint8_t biasShift2 = 0;                     // 第二层偏置定标。
    int taskBegin = 0;                          // 本核起始块编号。
    int taskEnd = 0;                            // 本核结束块编号。
    int taskIndex = 0;                          // 当前块编号。
    int batchIndex = 0;                         // 当前批次编号。
    int rowBandIndex = 0;                       // 当前行带编号。
    int columnSegmentIndex = 0;                 // 当前列段编号。
    int inputBatchOffset = 0;                   // 当前批次输入偏移。
    int outputBatchOffset = 0;                  // 当前批次输出偏移。
    int intermediatePositions = 0;              // 当前中间平面位置数。
    int inputSlotSequence = 0;                  // 两层共用的 L0A 交替计数。
    int accumulatorSlotSequence = 0;            // L0C 交替计数。
    int accumulatorSlotCount = 0;               // L0C 实际槽数。
    int weight1LandingSent = 0;                 // 完整 L1 区已搬入第一层段数。
    int weight1LandingConsumed = 0;             // 完整 L1 区已首次消费段数。
    int weight1StreamTotal = 0;                 // 第一层环形装载总段数。
    int weight1StreamSent = 0;                  // 第一层环形已搬入段数。
    int weight1StreamConsumed = 0;              // 第一层环形已消费段数。
    int weight2SegmentElements = 0;             // 第二层每段元素数。
    int weight2StreamTotal = 0;                 // 第二层环形装载总段数。
    int weight2StreamSent = 0;                  // 第二层环形已搬入段数。
    int weight2StreamConsumed = 0;              // 第二层环形已消费段数。
    bool landFirstWeightsInPieces = false;      // 第一层是否逐段补齐完整 L1 区。
    bool splitFirstInputLoad = false;           // 首块是否分批搬入输入通道。
    bool inputChannelsPending = false;          // 剩余输入通道是否尚未等候就绪。
    bool deferSecondWeightLoad = false;         // 第二层整套权重是否延迟搬入。
    bool weight1Resident = false;               // 第一层整套权重是否可留在 L0B。
    bool weight2Resident = false;               // 第二层整套权重是否可留在 L0B。
    bool firstLoad3D = false;                   // 当前块是否尚未发起首个 Load3D。
};

// 执行两层融合卷积；x/w1/b1/w2/b2 为输入与偏置，dq2/q1/q2 为定标表，y 为输出，args 为执行参数。
template <typename C>
__aicore__ inline void FusedMultiConv2dKernel(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR dq2,
                                              GM_ADDR q1, GM_ADDR q2, GM_ADDR y, const KernelArgs& args)
{
    KernelExecutor<C> executor(x, w1, b1, w2, b2, dq2, q1, q2, y, args);
    executor.Run();
}

} // namespace FusedMultiConv2d

#endif
