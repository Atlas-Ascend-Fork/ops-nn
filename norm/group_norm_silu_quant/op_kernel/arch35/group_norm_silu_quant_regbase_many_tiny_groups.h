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
 * \file group_norm_silu_quant_regbase_many_tiny_groups.h
 * \brief GroupNormSiluQuant arch35 调度: many-tiny-groups(tilingKey 1150)。
 *        场景(host 守卫保证): shapeD==1(每组正好 1 通道) + hwNum 小(<=32) + shapeC(=numGroups)>4096 +
 *        N*numGroups>=核数。此时每个 group 实例 = hwNum 个连续元素(elemNum==hwNum), 组数极多而单组 reduce 极小。
 *        慢路(1130)逐组 ReduceSum + 逐组 VF normalize, per-group 固定开销主导。本模板把"组"批量化:
 *        一次载入一批 B 个连续 group(B*hwNum 元素), 用一条分段归约(WholeReduceSum, mask=hwNum, repeat=B)
 *        算出 B 个组的 sum/sumsq; 逐批向量化算 mean/rstd/仿射系数; 用 Brcb 广播把 per-group 标量铺到 B*hwNum,
 *        一次性向量化做 normalize+silu+quant; 逐组连续落盘 int8。与其它 tilingKey 完全独立。
 *
 *        UB 组内布局: 每组占 padElems = RoundUp(hwNum,32) 个元素(hwNum<=32 => padElems 恒 =32), 载入时用
 *        DataCopyPad 的 dstStride 强制每组行 =padElems 元素。fp32/int8 共用同一 element index(contiguous Cast),
 *        故 fp32 行 =32*4B、int8 行 =32*1B 均 32B 对齐 —— 满足 fp32 分段归约(srcRepStride=padElems/8=4)与 int8
 *        逐组落盘(每组行 32B 对齐, DataCopyPad 以 32B 块读)两个约束。组尾 [hwNum,padElems) 是 gap(垃圾,
 *        归约用 mask=hwNum 屏蔽, 输出只落 hwNum, 无副作用)。
 */

#ifndef GROUP_NORM_SILU_QUANT_REGBASE_MANY_TINY_GROUPS_H_
#define GROUP_NORM_SILU_QUANT_REGBASE_MANY_TINY_GROUPS_H_

#include "group_norm_silu_quant_regbase_base.h"

namespace GroupNormSiluQuant {
using namespace AscendC;

template <typename T1, typename T2>
class GroupNormSiluQuantManyTinyGroups {
public:
    // 每批处理的组数 B。取 64: (1) <=255 满足 WholeReduceSum repeatTime 上限; (2) 8 的倍数, Brcb / 32B 对齐友好;
    // (3) B*MAX_PAD = 64*32 = 2048 个 fp32 一片 UB tile, double-buffer 后 UB 占用宽裕。
    static constexpr int64_t GROUPS_PER_TILE = 64;
    // hwNum<=32 => padElems<=AlignUp(32*4,32)/4=32(fp32) 或 AlignUp(32*2,32)/2=32(bf16/half), 故 MAX_PAD=32。
    static constexpr int64_t MAX_PAD = 32;
    static constexpr int64_t TILE_CAP = GROUPS_PER_TILE * MAX_PAD; // 2048: 每 tile fp32 元素容量
    static constexpr uint64_t U64_BIT_LEN = 64;                    // 取严格大于 n 的最小 2 的幂用

    __aicore__ inline GroupNormSiluQuantManyTinyGroups(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale, GM_ADDR silu, GM_ADDR mean,
                                GM_ADDR rstd, GM_ADDR workspace, const GroupNormSiluQuantRegbaseTilingData* tilingData)
    {
        (void)workspace; // 本调度单遍完成, 不用 workspace(无跨核归约)
        tiling = tilingData;
        blockIdx = GetBlockIdx();
        shapeC = tiling->shapeC;
        shapeD = tiling->shapeD;
        numGroups = tiling->numGroups;
        hwNum = tiling->hwNum;
        shapeQuantScale = tiling->shapeQuantScale;
        perChannelScale = (shapeQuantScale != 1);
        activateSilu = (tiling->activateSilu != 0);
        eps = tiling->epsilon;
        realCoreNum = tiling->realCoreNum;
        numPerCore = tiling->numPerCore;
        numLastCore = tiling->numLastCore;

        // shapeD==1 => elemNum==hwNum, 每组 hwNum 个元素。1/N = 1/hwNum(与 golden numRec 逐位一致: 同一整数的 fp32
        // 倒数)。
        invCnt = 1.0f / static_cast<float>(hwNum);
        // 统一组内行跨距 padElems(元素数): 取 RoundUp(hwNum,32)。hwNum<=32(host 守卫)=> padElems 恒为 32。
        // 选 32 的倍数的关键: (1) fp32 归约 srcRepStride=padElems/8 为整数; (2) int8 落盘每组行按 32B 对齐
        //   (CopySilu2Gm 的 DataCopyPad 以 32B 块粒度从 UB 读, 非 32B 行会跨组错位/越界, 见 base.h 注释)。
        // fp32 与 int8 用同一 element index(contiguous Cast 保持下标)=> fp32 行 =padElems*4B, int8 行 =padElems*1B, 均
        // 32B 对齐。
        padElems = ((hwNum + (BLOCK_SIZE - 1)) / BLOCK_SIZE) * BLOCK_SIZE;            // = 32
        srcRepStride = static_cast<int32_t>(padElems / (BLOCK_SIZE / sizeof(float))); // padElems*4/32 = padElems/8 = 4
        // 载入时把 T1 每组行强制对齐到 padElems 个元素: DataCopyPad 的 dstStride 令
        //   actDstStride = AlignUp(dstStride*32 + hwNum*sizeof(T1), 32) = dstStride*32 + baseBytes 恰 =
        //   padElems*sizeof(T1)。
        int64_t baseBytes = ((hwNum * static_cast<int64_t>(sizeof(T1)) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        loadDstStride = static_cast<uint32_t>((padElems * static_cast<int64_t>(sizeof(T1)) - baseBytes) / BLOCK_SIZE);

        InitGmTensors(x, gamma, beta, quantScale, silu, mean, rstd);
        InitUbBuffers();
    }

    __aicore__ inline void InitGmTensors(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale, GM_ADDR silu,
                                         GM_ADDR mean, GM_ADDR rstd)
    {
        xGm.SetGlobalBuffer((__gm__ T1*)x);
        quantScaleGm.SetGlobalBuffer((__gm__ float*)quantScale);
        siluGm.SetGlobalBuffer((__gm__ int8_t*)silu);
        if (beta != nullptr) {
            hasBeta = true;
            betaGm.SetGlobalBuffer((__gm__ T2*)beta);
        }
        if (gamma != nullptr) {
            hasGamma = true;
            gammaGm.SetGlobalBuffer((__gm__ T2*)gamma);
        }
        // mean/rstd 输出 dtype 跟随 split_reduce 同约定: T2==fp32 走 float 输出, 否则走 T1(half/bf16)输出。
        if constexpr (sizeof(T2) == sizeof(float)) {
            rstdT2Gm.SetGlobalBuffer((__gm__ T2*)rstd);
            meanT2Gm.SetGlobalBuffer((__gm__ T2*)mean);
        } else {
            rstdGm.SetGlobalBuffer((__gm__ T1*)rstd);
            meanGm.SetGlobalBuffer((__gm__ T1*)mean);
        }
    }

    __aicore__ inline void InitUbBuffers()
    {
        constexpr int64_t B = GROUPS_PER_TILE;
        // WholeReduceSum 内部每 repeat 载满一个 VL 向量寄存器(而非恰好 hwNum), 末组会向后多读到 VL 边界;
        // 故归约的 src(xf/sq)与 dst(sum/sqSum)各多留 1 个 VL_FP32 余量, 避免越过 buffer 末尾读/写。
        pipe.InitBuffer(inQue, 2, TILE_CAP * sizeof(T1));
        pipe.InitBuffer(outQue, 2, TILE_CAP * sizeof(int8_t));
        pipe.InitBuffer(xfBuf, (TILE_CAP + VL_FP32) * sizeof(float)); // x(fp32); 归约后原地做 normalize
        pipe.InitBuffer(sqBuf, (TILE_CAP + VL_FP32) * sizeof(float)); // x^2 / sigmoid 暂存 / int16 量化中转(复用)
        pipe.InitBuffer(aExpBuf,
                        TILE_CAP * sizeof(float)); // 广播后的 A(=rstd*gamma) / quantScale / fp16 量化中转(复用)
        pipe.InitBuffer(bExpBuf, TILE_CAP * sizeof(float)); // 广播后的 Bb(=beta-mean*A)
        pipe.InitBuffer(brcbBuf, B * 8 * sizeof(float));    // Brcb 输出([B][8] 块)
        // per-group 标量(B 宽 fp32); 归约 dst 多留 VL_FP32 余量
        pipe.InitBuffer(sumBuf, (B + VL_FP32) * sizeof(float));
        pipe.InitBuffer(sqSumBuf, (B + VL_FP32) * sizeof(float));
        pipe.InitBuffer(meanBuf, B * sizeof(float));
        pipe.InitBuffer(rstdBuf, B * sizeof(float));
        pipe.InitBuffer(aBuf, B * sizeof(float));
        pipe.InitBuffer(bBuf, B * sizeof(float));
        pipe.InitBuffer(tmpBuf, B * sizeof(float));
        pipe.InitBuffer(oneBuf, B * sizeof(float));
        pipe.InitBuffer(gammaBuf, B * sizeof(float));
        pipe.InitBuffer(betaBuf, B * sizeof(float));
        pipe.InitBuffer(qsRawBuf, B * sizeof(float));
        pipe.InitBuffer(gRawBuf, B * sizeof(T2));
        pipe.InitBuffer(bRawBuf, B * sizeof(T2));
        pipe.InitBuffer(meanOutBuf, B * sizeof(T1));
        pipe.InitBuffer(rstdOutBuf, B * sizeof(T1));
    }

    __aicore__ inline void Process()
    {
        if (blockIdx >= realCoreNum) {
            return;
        }
        // 跨 pipe 同步事件(FetchEventID: run-once, 无需 Release)。不用 6/7, 不用 PIPE_S barrier。
        evM2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        evV2M3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        evM3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        evV2M2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        evM2V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

        LocalTensor<float> one = oneBuf.Get<float>();
        Duplicate(one, 1.0f, static_cast<int32_t>(GROUPS_PER_TILE));
        PipeBarrier<PIPE_V>();

        int64_t count = (blockIdx == realCoreNum - 1) ? numLastCore : numPerCore;
        int64_t gStart = blockIdx * numPerCore;
        int64_t t = 0;
        while (t < count) {
            int64_t gTile = gStart + t;
            int64_t curB = (count - t) < GROUPS_PER_TILE ? (count - t) : GROUPS_PER_TILE;
            int64_t cBase = gTile % numGroups;
            // 关键: 把 tile 截断到不跨越 numGroups 边界(channel = g_global % numGroups)。这样一个 tile 内
            // gamma/beta/quantScale 的 channel 恒连续 [cBase, cBase+curB), 永不回绕, 载入/对齐都简单正确。
            if (cBase + curB > numGroups) {
                curB = numGroups - cBase;
            }
            ProcessTile(gTile, curB, cBase);
            t += curB;
        }
    }

private:
    // 一批 curB 个连续组(channel 连续 [cBase, cBase+curB), 无回绕)。
    __aicore__ inline void ProcessTile(int64_t gTile, int64_t curB, int64_t cBase)
    {
        const int32_t n = static_cast<int32_t>(curB * padElems); // 本 tile 参与向量计算的元素数(含 gap)
        CopyInTile(gTile, curB, n);
        ComputeMeanRstdTile(curB, n);
        float quantScaleScalar = 1.0f;
        LoadAffineParams(curB, cBase, quantScaleScalar);
        FoldAffineTile(curB);
        NormalizeSiluQuantStoreTile(gTile, curB, n, quantScaleScalar);
        // ---- 写 mean/rstd(本 tile curB 个, 输出 shape [N,numGroups] flat, 偏移 gTile) ----
        LocalTensor<float> meanT = meanBuf.Get<float>();
        LocalTensor<float> rstdT = rstdBuf.Get<float>();
        WriteMeanRstd(meanT, rstdT, gTile, curB);
    }

    // CopyIn: 一次 DataCopyPad 把 curB 组连续搬入, 用 dstStride 把每组对齐到 padElems(=32)元素行。
    __aicore__ inline void CopyInTile(int64_t gTile, int64_t curB, int32_t n)
    {
        LocalTensor<float> xf = xfBuf.Get<float>();
        LocalTensor<T1> xin = inQue.AllocTensor<T1>();
        // 先把整块 xin 清零: 每组尾部 gap [hwNum,padElems) 是未初始化 UB(可能 NaN/Inf), 归约 mask 未必能屏蔽其
        // 传播到 sumsq -> rstd 单组偶发偏差(偶现)。清零后 gap=0, sq gap=0, 归约干净。Duplicate(V) -> DataCopyPad(MTE2)
        // 需 V->MTE2 同步(否则 DataCopyPad 覆写与 Duplicate 竞争)。
        Duplicate(xin, static_cast<T1>(0), static_cast<int32_t>(curB * padElems));
        SetFlag<HardEvent::V_MTE2>(evV2M2);
        WaitFlag<HardEvent::V_MTE2>(evV2M2);
        // dstStride=loadDstStride 令每组在 UB 中占 padElems(=32)个元素(32B 对齐行), 组尾 [hwNum,padElems) 为 gap。
        DataCopyExtParams cp{static_cast<uint16_t>(curB), static_cast<uint32_t>(hwNum * sizeof(T1)), 0,
                             static_cast<int64_t>(loadDstStride), 0};
        DataCopyPadExtParams<T1> pad{false, 0, 0, 0};
        DataCopyPad(xin, xGm[gTile * hwNum], cp, pad); // group 实例 g_global 在 x 上偏移 g_global*hwNum(连续)
        inQue.EnQue(xin);
        xin = inQue.DeQue<T1>();
        Cast(xf, xin, RoundMode::CAST_NONE, n);
        inQue.FreeTensor(xin);
        PipeBarrier<PIPE_V>();
    }

    // 逐组走公共统计量层。本 key 的"段 = 组", 故退化为: 一段 -> 均值 -> 一段 eps -> 收口。
    // 组内元素连续、行距 padElems; 组循环属于调度, 归本 key。
    // 旧实现自写 mean = Σx/hwNum(无 2 的幂缩放、无补偿求和) + 两遍法 var, 与其它 key 各一套;
    // 现统一到公共接口, 数学与 1100/1110/1140 完全一致(四档探针依据见 base.h 公共统计量层注释)。
    __aicore__ inline void ComputeMeanRstdTile(int64_t curB, int32_t n)
    {
        LocalTensor<float> xf = xfBuf.Get<float>();
        LocalTensor<float> meanT = meanBuf.Get<float>();
        LocalTensor<float> rstdT = rstdBuf.Get<float>();
        LocalTensor<float> accTensor = sqBuf.Get<float>(); // 归约阶段复用为公共层累加器
        // 归约过读 slack 清零(末组会向后多读到 VL 边界)
        Duplicate(xf[n], 0.0f, static_cast<int32_t>(VL_FP32));
        PipeBarrier<PIPE_V>();
        __local_mem__ float* accBuf = (__local_mem__ float*)accTensor.GetPhyAddr();
        __local_mem__ float* tmpMeanLocal = accBuf;
        __local_mem__ float* tmpVarLocal = accBuf + VL_FP32;
        __local_mem__ float* statScratch = accBuf + 2 * VL_FP32;
        __local_mem__ float* xfLocal = (__local_mem__ float*)xf.GetPhyAddr();
        __local_mem__ float* meanLocal = (__local_mem__ float*)meanT.GetPhyAddr();
        __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdT.GetPhyAddr();
        for (int64_t g = 0; g < curB; g++) {
            __local_mem__ float* grpPtr = xfLocal + g * padElems;
            float mean = 0.0f;
            float m2 = 0.0f;
            float count = 0.0f;
            float sumHi = 0.0f;
            float sumLo = 0.0f;
            int32_t sk = SumScaleExp(static_cast<uint32_t>(hwNum));
            WelfordSegmentStat<float>(grpPtr, tmpMeanLocal, tmpVarLocal, statScratch, static_cast<uint32_t>(hwNum),
                                      mean, m2, count, Pow2f(-sk), sumHi, sumLo);
            WelfordGroupFinalize(meanLocal + g, rstdLocal + g, statScratch, mean, m2, count, eps, sumHi, sumLo,
                                 Pow2f(sk));
        }
        PipeBarrier<PIPE_ALL>(); // 后续对 meanT/rstdT 做矢量广播
    }

    // 载 gamma/beta/quantScale(channel 连续段), cast fp32; per-tensor 取 quantScaleScalar。
    __aicore__ inline void LoadAffineParams(int64_t curB, int64_t cBase, float& quantScaleScalar)
    {
        const int32_t cb = static_cast<int32_t>(curB);
        LocalTensor<float> gammaT = gammaBuf.Get<float>();
        LocalTensor<float> betaT = betaBuf.Get<float>();
        LocalTensor<float> qsRaw = qsRawBuf.Get<float>();
        if (hasGamma) {
            LocalTensor<T2> gRaw = gRawBuf.Get<T2>();
            DataCopyExtParams cpg{1, static_cast<uint32_t>(curB * sizeof(T2)), 0, 0, 0};
            DataCopyPadExtParams<T2> padg{false, 0, 0, 0};
            DataCopyPad(gRaw, gammaGm[cBase], cpg, padg);
            SetFlag<HardEvent::MTE2_V>(evM2V);
            WaitFlag<HardEvent::MTE2_V>(evM2V);
            Cast(gammaT, gRaw, RoundMode::CAST_NONE, cb);
            PipeBarrier<PIPE_V>();
        }
        if (hasBeta) {
            LocalTensor<T2> bRaw = bRawBuf.Get<T2>();
            DataCopyExtParams cpb{1, static_cast<uint32_t>(curB * sizeof(T2)), 0, 0, 0};
            DataCopyPadExtParams<T2> padb{false, 0, 0, 0};
            DataCopyPad(bRaw, betaGm[cBase], cpb, padb);
            SetFlag<HardEvent::MTE2_V>(evM2V);
            WaitFlag<HardEvent::MTE2_V>(evM2V);
            Cast(betaT, bRaw, RoundMode::CAST_NONE, cb);
            PipeBarrier<PIPE_V>();
        }
        if (perChannelScale) {
            // per-channel: quantScale 本身就是 fp32, 按 channel 连续段载入
            DataCopyExtParams cpq{1, static_cast<uint32_t>(curB * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padq{false, 0, 0, 0};
            DataCopyPad(qsRaw, quantScaleGm[cBase], cpq, padq);
            SetFlag<HardEvent::MTE2_V>(evM2V);
            WaitFlag<HardEvent::MTE2_V>(evM2V);
        } else {
            // per-tensor: 单标量广播, 直接取倒数(标量除法; 量化 scale 精度不敏感, 与 b16/split_reduce 同)
            quantScaleScalar = quantScaleGm.GetValue(0);
        }
    }

    // 折叠仿射系数(B 宽): pre = x*A + Bb, A=rstd*gamma, Bb=beta - mean*A。
    __aicore__ inline void FoldAffineTile(int64_t curB)
    {
        const int32_t cb = static_cast<int32_t>(curB);
        LocalTensor<float> rstdT = rstdBuf.Get<float>();
        LocalTensor<float> gammaT = gammaBuf.Get<float>();
        LocalTensor<float> betaT = betaBuf.Get<float>();
        LocalTensor<float> meanT = meanBuf.Get<float>();
        LocalTensor<float> tmp = tmpBuf.Get<float>();
        LocalTensor<float> aVec = aBuf.Get<float>();
        LocalTensor<float> bVec = bBuf.Get<float>();
        if (hasGamma) {
            Mul(aVec, rstdT, gammaT, cb);
        } else {
            Adds(aVec, rstdT, 0.0f, cb); // A = rstd
        }
        PipeBarrier<PIPE_V>();
        Mul(tmp, meanT, aVec, cb); // mean*A
        PipeBarrier<PIPE_V>();
        if (hasBeta) {
            Sub(bVec, betaT, tmp, cb); // Bb = beta - mean*A
        } else {
            Muls(bVec, tmp, -1.0f, cb); // Bb = -mean*A
        }
        PipeBarrier<PIPE_V>();
    }

    // 广播 A/Bb 一次性向量化 normalize; silu; 量化 y/quantScale -> clamp 到 [-128,127]。
    // 归一化+silu+量化+落盘: 走公共 C 层入口(向量版, regbase VF), 不再用 Level-2 API 自写 ——
    // silu 与量化收尾因此与其它 key 完全一致。A/B 已由 FoldAffineTile 折好(A=rstd*gamma、
    // B=beta-mean*rstd*gamma), quantScale 按 per-channel/per-tensor 广播到逐元素。
    __aicore__ inline void NormalizeSiluQuantStoreTile(int64_t gTile, int64_t curB, int32_t n, float quantScaleScalar)
    {
        LocalTensor<float> xf = xfBuf.Get<float>();
        LocalTensor<float> qsRaw = qsRawBuf.Get<float>();
        LocalTensor<float> aVec = aBuf.Get<float>();
        LocalTensor<float> bVec = bBuf.Get<float>();
        LocalTensor<float> aExp = aExpBuf.Get<float>();
        LocalTensor<float> bExp = bExpBuf.Get<float>();
        LocalTensor<float> qsExp = sqBuf.Get<float>(); // 归约阶段已用完, 此处复用为 quantScale 广播缓冲
        BroadcastGroups(aExp, aVec, curB);
        BroadcastGroups(bExp, bVec, curB);
        if (perChannelScale) {
            BroadcastGroups(qsExp, qsRaw, curB);
        } else {
            Duplicate(qsExp, quantScaleScalar, n);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<int8_t> yo = outQue.AllocTensor<int8_t>();
        VFNormalizeSiluQuantVec((__local_mem__ float*)xf.GetPhyAddr(), (__local_mem__ int8_t*)yo.GetPhyAddr(),
                                (__local_mem__ float*)aExp.GetPhyAddr(), (__local_mem__ float*)bExp.GetPhyAddr(),
                                (__local_mem__ float*)qsExp.GetPhyAddr(), static_cast<uint32_t>(n), activateSilu);
        outQue.EnQue(yo);
        yo = outQue.DeQue<int8_t>();
        // 逐组连续落盘: UB 组行按 padElems 对齐, GM 上 hwNum 连续; padElems!=hwNum 时逐行拷贝
        CopySilu2Gm<int8_t>(siluGm[gTile * hwNum], yo, static_cast<uint16_t>(curB), static_cast<uint32_t>(hwNum),
                            static_cast<uint32_t>(padElems));
        outQue.FreeTensor(yo);
    }

    // 把 per-group 标量 src[0..curB) 广播到 dst[g*padElems + j] = src[g], j in [0,padElems)。
    // Brcb 把每个标量铺成一个 32B 块(8 lane), 再按 padElems/8 个子块用 UB->UB DataCopy(PIPE_V)复制到位。
    __aicore__ inline void BroadcastGroups(const LocalTensor<float>& dst, const LocalTensor<float>& src, int64_t curB)
    {
        LocalTensor<float> brc = brcbBuf.Get<float>();
        uint8_t rep = static_cast<uint8_t>((curB + 7) / 8);
        Brcb(brc, src, rep, {1, 8}); // brc[g*8 + l] = src[g]  (PIPE_V 写 brc)
        // V->MTE2: 保证 Brcb 写完 brc, 再让 UB->UB DataCopy(MTE2)读; 否则 DataCopy 读到未写完的 brc(偶现整组标量错)。
        SetFlag<HardEvent::V_MTE2>(evV2M2);
        WaitFlag<HardEvent::V_MTE2>(evV2M2);
        uint16_t m = static_cast<uint16_t>(padElems / 8); // 每组占 m 个 32B 块
        for (uint16_t b = 0; b < m; b++) {
            // 每次拷 curB 个块(每块 1*32B=8 fp32): dst 块位置 g*m + b(dstStride=m-1), src 块位置 g(srcStride=0)。
            DataCopy(dst[b * 8], brc, DataCopyParams{static_cast<uint16_t>(curB), 1, 0, static_cast<uint16_t>(m - 1)});
        }
        // MTE2->V: 保证广播 DataCopy 写完 dst(aExp/bExp), 再让消费的 Mul(PIPE_V)读; 否则 Mul 读到未写完的广播值。
        SetFlag<HardEvent::MTE2_V>(evM2V2);
        WaitFlag<HardEvent::MTE2_V>(evM2V2);
    }

    // 复用现有 kernel 的 mean/rstd 输出 helper: T2==fp32 -> CopyMeanAndRstd2Gm<float>; 否则 ProcessMeanAndRstd<T1>。
    // meanT/rstdT 为向量算出的 fp32 结果(在 UB), 输出前需保证 V 完成。
    __aicore__ inline void WriteMeanRstd(const LocalTensor<float>& meanT, const LocalTensor<float>& rstdT,
                                         int64_t gTile, int64_t curB)
    {
        if constexpr (sizeof(T2) == sizeof(float)) {
            // float 输出: DataCopyPad(MTE3) 直接读 UB fp32, 需 V->MTE3
            SetFlag<HardEvent::V_MTE3>(evV2M3);
            WaitFlag<HardEvent::V_MTE3>(evV2M3);
            CopyMeanAndRstd2Gm<float>(meanT2Gm[gTile], rstdT2Gm[gTile], meanT, rstdT, 1, static_cast<uint32_t>(curB));
            // 串行化这次 mean/rstd 落盘: 下一 tile 会用 V 覆写同一 meanBuf/rstdBuf, 需等本次 MTE3 读完(MTE3->V)。
            SetFlag<HardEvent::MTE3_V>(evM3V);
            WaitFlag<HardEvent::MTE3_V>(evM3V);
        } else {
            // T1(half/bf16)输出: ProcessMeanAndRstd 内部 Reg 读 UB(V) 再窄化 + V->MTE3, 前置 V 屏障即可
            PipeBarrier<PIPE_V>();
            // ProcessMeanAndRstd 形参为非 const 引用, 用具名 LocalTensor(不可传 Get<>() 临时量)
            LocalTensor<float> meanLocal = meanBuf.Get<float>();
            LocalTensor<float> rstdLocal = rstdBuf.Get<float>();
            LocalTensor<T1> meanO = meanOutBuf.Get<T1>();
            LocalTensor<T1> rstdO = rstdOutBuf.Get<T1>();
            ProcessMeanAndRstd<T1>(meanLocal, meanO, meanGm, rstdLocal, rstdO, rstdGm, static_cast<uint64_t>(gTile),
                                   static_cast<uint32_t>(curB));
            // 串行化: 下一 tile 会 V 覆写 meanOutBuf/rstdOutBuf, 需等本次内部 MTE3 落盘读完(MTE3->V)。
            SetFlag<HardEvent::MTE3_V>(evM3V);
            WaitFlag<HardEvent::MTE3_V>(evM3V);
        }
    }

    TPipe pipe;
    TQue<TPosition::VECIN, 2> inQue;
    TQue<TPosition::VECOUT, 2> outQue;
    TBuf<> xfBuf;
    TBuf<> sqBuf;
    TBuf<> aExpBuf;
    TBuf<> bExpBuf;
    TBuf<> brcbBuf;
    TBuf<> sumBuf;
    TBuf<> sqSumBuf;
    TBuf<> meanBuf;
    TBuf<> rstdBuf;
    TBuf<> aBuf;
    TBuf<> bBuf;
    TBuf<> tmpBuf;
    TBuf<> oneBuf;
    TBuf<> gammaBuf;
    TBuf<> betaBuf;
    TBuf<> qsRawBuf;
    TBuf<> gRawBuf;
    TBuf<> bRawBuf;
    TBuf<> meanOutBuf;
    TBuf<> rstdOutBuf;

    GlobalTensor<T1> xGm;
    GlobalTensor<T1> meanGm;
    GlobalTensor<T2> gammaGm;
    GlobalTensor<T1> rstdGm;
    GlobalTensor<T2> betaGm;
    GlobalTensor<T2> meanT2Gm;
    GlobalTensor<float> quantScaleGm;
    GlobalTensor<T2> rstdT2Gm;
    GlobalTensor<int8_t> siluGm;

    const GroupNormSiluQuantRegbaseTilingData* tiling;
    int64_t blockIdx;
    int64_t numGroups;
    int64_t hwNum;
    int64_t shapeC;
    int64_t shapeD;
    int64_t shapeQuantScale;
    int64_t realCoreNum;
    int64_t numPerCore;
    int64_t numLastCore;
    int64_t padElems;
    int32_t srcRepStride;
    uint32_t loadDstStride;
    float invCnt;
    float eps;
    bool perChannelScale{false};
    bool activateSilu{true};
    bool hasGamma{false};
    bool hasBeta{false};
    event_t evM2V;
    event_t evV2M3;
    event_t evM3V;
    event_t evV2M2; // Brcb(V) -> 广播 DataCopy(MTE2) 的 V->MTE2 同步
    event_t evM2V2; // 广播 DataCopy(MTE2) -> 消费 Mul(V) 的 MTE2->V 同步
};

} // namespace GroupNormSiluQuant

#endif // GROUP_NORM_SILU_QUANT_REGBASE_MANY_TINY_GROUPS_H_

/* many_tiny_groups (tilingKey 1150) 设计: 每 tile 处理 B(GROUPS_PER_TILE)=64 个小组;
 * 每组行按 padElems = RoundUp(hwNum,32) 布局(hwNum<=32 => padElems=32), WholeReduceSum 以 mask=hwNum 屏蔽组尾 gap。
 */
