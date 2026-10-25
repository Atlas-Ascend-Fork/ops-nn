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
 * \file group_norm_silu_quant_regbase_split_reduce.h
 * \brief GroupNormSiluQuant arch35 调度: split-reduce(tilingKey 1140)。
 *        N*num_groups < 核数 且单组 reduce 很大(如 G=1)时,按 shapeD 通道把单个 group 切到 coresPerGroup 个核,
 *        两阶段跨核归约:Pass1 各核对自己那段通道求 sum/sumsq 写 workspace;SyncAll;各核读回本组全部 partial
 *        合成全局 mean/rstd;Pass2 各核对自己通道段做 normalize+silu+quant 落盘。与其它 tilingKey 完全独立。
 */

#ifndef GROUP_NORM_SILU_QUANT_REGBASE_SPLIT_REDUCE_H_
#define GROUP_NORM_SILU_QUANT_REGBASE_SPLIT_REDUCE_H_

#include "group_norm_silu_quant_regbase_base.h"

namespace GroupNormSiluQuant {
using namespace AscendC;

template <typename T1, typename T2>
class GroupNormSiluQuantSplitReduce {
public:
    static constexpr int64_t TILE = 4096; // 每次搬运/计算的元素数
    // 均值必须由【缩放和】相加后一次还原, 不能由各 tile/各核的均值 combine 得到 —— 合并引入的
    // 舍入之后再没有机会被 eps 精化, 恒定输入档会让归一化输出 z=(x-mean)*rstd 从 0 跳到 O(1),
    // 量化后大面积饱和。host MaybeSetSplitReduce 的 bytes 须按 8 float/核 同步维护。
    //
    // 每核槽位 = 8 float(32B), 实际载荷只有前 4 个 {mean, M2, count, 预留}。槽位按 32B 开是
    // 全仓范式(add_rms_norm 的 numRowAlign / ada_layer_norm_grad 的 colAlignV / batch_norm_grad
    // 的 aDim 同形): GM 写回按 32B 块, 若每核只占 16B 则 blockIdx=2i 与 2i+1 落在同一个 32B 块
    // 内, 两核的部分状态会互相刷掉。槽位上半 4 float 不被读(合并只取前 3 个)。
    static constexpr int64_t WS_FLOATS_PER_CORE = 8; // 每核槽位(stride), 32B 独占
    static constexpr int64_t WS_PART_FLOATS = 4;     // 槽位内真实载荷 {mean, M2, count, 预留}
    static constexpr uint64_t U64_BIT_LEN = 64;      // 取二次幂用

    __aicore__ inline GroupNormSiluQuantSplitReduce(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale, GM_ADDR silu, GM_ADDR mean,
                                GM_ADDR rstd, GM_ADDR workspace, const GroupNormSiluQuantRegbaseTilingData* tilingData)
    {
        tiling = tilingData;
        blockIdx = GetBlockIdx();
        numGroups = tiling->numGroups;
        hwNum = tiling->hwNum;
        shapeC = tiling->shapeC;
        shapeD = tiling->shapeD;
        shapeQuantScale = tiling->shapeQuantScale;
        perChannelScale = (shapeQuantScale != 1);
        eps = tiling->epsilon;
        activateSilu = (tiling->activateSilu != 0);
        coresPerGroup = tiling->coresPerGroup;

        groupGlobalIdx = blockIdx / coresPerGroup;
        subIdx = blockIdx % coresPerGroup;
        int64_t grp = groupGlobalIdx % numGroups;
        int64_t nIdx = groupGlobalIdx / numGroups;

        chanPerCore = CeilDiv(shapeD, coresPerGroup);
        chanStart = subIdx * chanPerCore;
        if (chanStart >= shapeD) {
            myChan = 0;
        } else {
            myChan = (chanStart + chanPerCore <= shapeD) ? chanPerCore : (shapeD - chanStart);
        }
        myElem = myChan * hwNum;

        int64_t chanBaseC = grp * shapeD + chanStart;
        xBase = nIdx * shapeC * hwNum + chanBaseC * hwNum;
        groupXBase = nIdx * shapeC * hwNum + (grp * shapeD) * hwNum; // 组内首元素(各核同值),方差平移原点
        gammaBase = chanBaseC;
        qsBase = perChannelScale ? chanBaseC : 0;

        InitGmTensors(x, gamma, beta, quantScale, silu, mean, rstd, workspace);
        InitUbBuffers();
    }

    __aicore__ inline void InitGmTensors(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale, GM_ADDR silu,
                                         GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace)
    {
        xGm.SetGlobalBuffer((__gm__ T1*)x);
        if (gamma != nullptr) {
            hasGamma = true;
            gammaGm.SetGlobalBuffer((__gm__ T2*)gamma);
        }
        if (beta != nullptr) {
            hasBeta = true;
            betaGm.SetGlobalBuffer((__gm__ T2*)beta);
        }
        quantScaleGm.SetGlobalBuffer((__gm__ float*)quantScale);
        siluGm.SetGlobalBuffer((__gm__ int8_t*)silu);
        if constexpr (sizeof(T2) == sizeof(float)) {
            meanT2Gm.SetGlobalBuffer((__gm__ T2*)mean);
            rstdT2Gm.SetGlobalBuffer((__gm__ T2*)rstd);
        } else {
            meanGm.SetGlobalBuffer((__gm__ T1*)mean);
            rstdGm.SetGlobalBuffer((__gm__ T1*)rstd);
        }
        wsGm.SetGlobalBuffer((__gm__ float*)workspace);
    }

    __aicore__ inline void InitUbBuffers()
    {
        int64_t chanAlign = (chanPerCore < 8 ? 8 : chanPerCore);
        pipe.InitBuffer(inQue, 2, TILE * sizeof(T1));
        pipe.InitBuffer(outQue, 2, TILE * sizeof(int8_t));
        pipe.InitBuffer(fbuf, TILE * sizeof(float));
        pipe.InitBuffer(rbuf, TILE * sizeof(float));
        pipe.InitBuffer(wbuf, TILE * sizeof(float));
        pipe.InitBuffer(reduf, 32 * sizeof(float)); // acc: [8]=sumsq [16]=ReduceSum 暂存, 需 >=24 floats
        pipe.InitBuffer(partBuf, BLOCK_SIZE);
        pipe.InitBuffer(combBuf, RoundUp<float>(coresPerGroup * WS_FLOATS_PER_CORE) * sizeof(float) + BLOCK_SIZE);
        // gamma/beta 的通道常驻缓冲(每通道 sizeof(T2)+4 字节)只在对应输入存在时才按通道数开:
        // 无 gamma/beta 时按 chanAlign 开是纯浪费, 通道多时会把 UB 撑爆。
        // 缺省给一个最小块, 使 gF/bF 的 Get<>() 仍合法
        // ——其值不会被读: DoPass2Normalize 在 hasGamma/hasBeta 为假时取常量 1.0f/0.0f。
        // 本核通道数受 host MaybeSetSplitReduce 的 UB 预算校验约束, 二者必须同时维护。
        pipe.InitBuffer(gRawBuf, hasGamma ? chanAlign * sizeof(T2) : BLOCK_SIZE);
        pipe.InitBuffer(bRawBuf, hasBeta ? chanAlign * sizeof(T2) : BLOCK_SIZE);
        pipe.InitBuffer(gFBuf, hasGamma ? chanAlign * sizeof(float) : BLOCK_SIZE);
        pipe.InitBuffer(bFBuf, hasBeta ? chanAlign * sizeof(float) : BLOCK_SIZE);
        // WriteMeanRstd 专用(ProcessMeanAndRstd idiom):Reg 读满 VL_FP32,故 mean/rstd 各留 VL_FP32
        pipe.InitBuffer(mrFBuf, 2 * VL_FP32 * sizeof(float));
        pipe.InitBuffer(mrOBuf, 2 * VL_FP32 * sizeof(T1));
    }

    __aicore__ inline void Process()
    {
        // 用 FetchEventID(run-once,无需 Release)做 Scalar<->MTE/V 跨 pipe 同步
        event_t evS2M3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        event_t evM2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t evS2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t evV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t evM2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

        DoPass1PartialReduce(evS2M3);
        SyncAll();
        float mean = 0.0f;
        float rstd = 0.0f;
        float m2 = 0.0f;
        float count = 0.0f;
        ComputeGlobalMeanAndM2(mean, m2, count, evM2S);
        FinalizeGlobal(mean, rstd, m2, count);

        if (myChan == 0) {
            return;
        }

        PreloadGammaBeta(evM2V, evV2S);
        DoPass2Normalize(mean, rstd);
        // pass2 全部完成后再写 mean/rstd(隔离 MTE3/时序干扰), 全屏障后由 subIdx==0 单核写
        PipeBarrier<PIPE_ALL>();
        if (subIdx == 0) {
            WriteMeanRstd(mean, rstd, evS2V);
        }
    }

    // 取组内首元素作为方差的平移原点。两遍式 var = Σx²/N - mean² 在 |mean| 远大于 σ 时灾难性相消
    // (如 x 恒为 -65504: Σx²/N 与 mean² 均为 4.29e9, 该量级 ULP≈512 ≫ 真方差 0, 舍入后可为负 ⇒ sqrt 得 nan)。
    // 改用平移式 Σ(x-c)²/N - (mean-c)²: c≈mean 时两项同阶且小, 无相消; x 恒定时精确为 0。
    __aicore__ inline float LoadShiftOrigin()
    {
        LocalTensor<float> cvt = combBuf.Get<float>();
        DataCopyExtParams cp{1, (uint32_t)sizeof(T1), 0, 0, 0};
        DataCopyPadExtParams<T1> pad{false, 0, 0, 0};
        if constexpr (sizeof(T1) == sizeof(float)) {
            DataCopyPad(cvt, xGm[groupXBase], cp, pad);
            event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(ev);
            WaitFlag<HardEvent::MTE2_S>(ev);
        } else {
            LocalTensor<T1> raw = partBuf.Get<T1>();
            DataCopyPad(raw, xGm[groupXBase], cp, pad);
            event_t evM2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(evM2V);
            WaitFlag<HardEvent::MTE2_V>(evM2V);
            Cast(cvt, raw, RoundMode::CAST_NONE, 1); // 向量转换,避免标量 bf16 转换
            event_t evV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(evV2S);
            WaitFlag<HardEvent::V_S>(evV2S);
        }
        return cvt.GetValue(0);
    }

    // Pass1: 本核 myElem 个连续元素 -> 一个 Welford 部分状态 {mean, M2, count}, 写 workspace[blockIdx*4]。
    // 统计量全部走公共接口(WelfordSegmentStat + WelfordCombinePartial), 本函数只负责搬运与切分。
    __aicore__ inline void DoPass1PartialReduce(event_t evS2M3)
    {
        LocalTensor<float> accTensor = rbuf.Get<float>(); // 公共统计量层累加器, 需 WELFORD_ACC_FLOATS 个 float
        __local_mem__ float* accBuf = (__local_mem__ float*)accTensor.GetPhyAddr();
        __local_mem__ float* tmpMeanLocal = accBuf;
        __local_mem__ float* tmpVarLocal = accBuf + VL_FP32;
        __local_mem__ float* statScratch = accBuf + 2 * VL_FP32;
        float mean = 0.0f;
        float m2 = 0.0f;
        float count = 0.0f;
        // TILE 批间走成对合并(见 base.h WelfordPairwiseAcc): 批数 = myElem/TILE 可达数十。
        WelfordPairwiseAcc acc;
        WelfordAccInit(acc);
        for (int64_t off = 0; off < myElem; off += TILE) {
            int64_t cur = (myElem - off) < TILE ? (myElem - off) : TILE;
            LocalTensor<T1> xin = inQue.AllocTensor<T1>();
            DataCopyExtParams cp{1, (uint32_t)(cur * sizeof(T1)), 0, 0, 0};
            // 搬入即零填尾块(全仓范式: group_norm_silu base.h:1608-1624 / rms_norm
            // split_d.h:121-130): 不补零时 xin[cur, 32B块) 是上轮残留, 一旦下游有任何一处
            // 按对齐长度整轮读满(LoadInputData 的 fp32 分支即无掩码整 VL 读), 残留就进和。
            DataCopyPadExtParams<T1> pad{true, 0, (uint8_t)(RoundUp<T1>(cur) - cur), 0};
            DataCopyPad(xin, xGm[xBase + off], cp, pad);
            inQue.EnQue(xin);
            xin = inQue.DeQue<T1>();
            LocalTensor<float> xf = fbuf.Get<float>();
            Cast(xf, xin, RoundMode::CAST_NONE, cur); // 向量转换, 避免标量 bf16 转换
            inQue.FreeTensor(xin);
            // Cast 只写 xf[0,cur), 而 fbuf 是跨 tile 复用的暂存槽: xf[cur, VL对齐) 是上一轮
            // 残留。搬入侧的 rightPadding 只能补到 32B 块(deep_norm.h:157-164 已记该边界),
            // 而公共 WelfordSegmentStat 的首折半按整 VL 读满(reduce_common_regbase_part2.h:
            // 110-124 用 pregFull), 故 slack 必须再补到 VL —— 全仓两段式补零范式
            // (rms_norm split_d.h:121-130 先 rightPadding 补 32B、再 Duplicate 补 VL)。
            // 起址必须 32B 对齐(矢量写 UB 的硬约束, 用 xf[cur] 这种任意偏移会 NO_OUTPUT):
            // 从 RoundUp<float>(cur) 起 —— 搬入侧 rightPadding 已把 [cur, RoundUp) 补 0,
            // 这里只接着补 [RoundUp, VL对齐), 两段正好衔接(范式同 rms_norm split_d.h:121-130,
            // 其 Duplicate 起址 calCount 本身就是对齐值)。
            int64_t curBlk = RoundUp<float>(cur);
            int64_t curVl = CeilDiv(cur, static_cast<int64_t>(VL_FP32)) * static_cast<int64_t>(VL_FP32);
            if (curVl > curBlk) {
                Duplicate(xf[curBlk], 0.0f, static_cast<int32_t>(curVl - curBlk));
            }
            PipeBarrier<PIPE_V>();
            // 本 tile 的部分统计量 -> 跨 tile 走公共 Welford 合并
            float tMean = 0.0f;
            float tM2 = 0.0f;
            float tCnt = 0.0f;
            float tHi = 0.0f;
            float tLo = 0.0f;
            WelfordSegmentStat<float>((__local_mem__ float*)xf.GetPhyAddr(), tmpMeanLocal, tmpVarLocal, statScratch,
                                      static_cast<uint32_t>(cur), tMean, tM2, tCnt, 0.0f, tHi, tLo);
            WelfordAccPush(acc, tMean, tM2, tCnt, tHi, tLo);
        }
        float accHi = 0.0f;
        float accLo = 0.0f;
        WelfordAccFinalize(acc, mean, m2, count, accHi, accLo);
        (void)accHi;
        (void)accLo;
        // partial -> workspace[blockIdx*WS_FLOATS_PER_CORE] = {mean, M2, count, 预留}
        LocalTensor<float> part = partBuf.Get<float>();
        part.SetValue(0, mean);
        part.SetValue(1, m2);
        part.SetValue(2, count);
        part.SetValue(3, 0.0f);
        SetFlag<HardEvent::S_MTE3>(evS2M3);
        WaitFlag<HardEvent::S_MTE3>(evS2M3);
        DataCopyExtParams cpp{1, (uint32_t)(WS_PART_FLOATS * sizeof(float)), 0, 0, 0};
        DataCopyPad(wsGm[blockIdx * WS_FLOATS_PER_CORE], part, cpp);
    }

    // 收口: 走公共接口出 mean/rstd(无精化), 再取回标量
    __aicore__ inline void FinalizeGlobal(float& mean, float& rstd, float m2, float count)
    {
        LocalTensor<float> accTensor = rbuf.Get<float>();
        __local_mem__ float* accBuf = (__local_mem__ float*)accTensor.GetPhyAddr();
        __local_mem__ float* statScratch = accBuf + 2 * VL_FP32;
        __local_mem__ float* meanSlot = accBuf + WELFORD_ACC_FLOATS;
        __local_mem__ float* rstdSlot = accBuf + WELFORD_ACC_FLOATS + 8;
        /* 跨核 partial 经 GM workspace 传递(布局属 host 侧, 本轮不改), 故这里不走补偿和:
           pw 传 0 表示退回 Welford 均值, 行为与改动前完全一致。 */
        WelfordGroupFinalize(meanSlot, rstdSlot, statScratch, mean, m2, count, eps, 0.0f, 0.0f, 0.0f);
        PipeBarrier<PIPE_ALL>(); // 矢量写 UB -> 标量读 UB
        mean = meanSlot[0];
        rstd = rstdSlot[0];
    }

    // combine: 读回本组 coresPerGroup 个部分统计量, 走 Welford 合并出全局 {mean, M2, count}。
    __aicore__ inline void ComputeGlobalMeanAndM2(float& mean, float& m2, float& count, event_t evM2S)
    {
        int64_t groupCoreBase = groupGlobalIdx * coresPerGroup;
        LocalTensor<float> comb = combBuf.Get<float>();
        DataCopyExtParams cpr{1, (uint32_t)(coresPerGroup * WS_FLOATS_PER_CORE * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padr{false, 0, 0, 0};
        DataCopyPad(comb, wsGm[groupCoreBase * WS_FLOATS_PER_CORE], cpr, padr);
        SetFlag<HardEvent::MTE2_S>(evM2S);
        WaitFlag<HardEvent::MTE2_S>(evM2S);
        /* 跨核 partial 走**两两树形**合并, 不用顺序累加。
           1140 比其它 key 多"跨核"这一层: coresPerGroup 可达数十(实测 L1_542 为 28),
           顺序累加的舍入线性累积, 实测该档 mean 距 fp64 真值 **-1.82 ULP**;
           改树形后降到 **+0.18 ULP**(与 golden 的 0.1 ULP 持平), 误差降一个数量级。
           放大倍数大的档(|gamma|/|quant_scale| 可达 1e5~1e9)会把 1 ULP 放大成约 0.5 个
           量化档, 故这 2 ULP 直接决定 out0 是否超差(实测 L1_542: 改前 q=107, golden=109)。
           注: 核内 tile 合并改树形**无收益**(实测 mean 一位不变), 瓶颈只在跨核这一层。
           就地两两归约, 结果写回 comb 的前半段, 不额外占 UB。 */
        int64_t cnt = coresPerGroup;
        while (cnt > 1) {
            int64_t half = cnt / 2;
            for (int64_t i = 0; i < half; i++) {
                float am = comb.GetValue((2 * i) * WS_FLOATS_PER_CORE);
                float av = comb.GetValue((2 * i) * WS_FLOATS_PER_CORE + 1);
                float ac = comb.GetValue((2 * i) * WS_FLOATS_PER_CORE + 2);
                float pHi = 0.0f;
                float pLo = 0.0f;
                WelfordCombinePartial(am, av, ac, comb.GetValue((2 * i + 1) * WS_FLOATS_PER_CORE),
                                      comb.GetValue((2 * i + 1) * WS_FLOATS_PER_CORE + 1),
                                      comb.GetValue((2 * i + 1) * WS_FLOATS_PER_CORE + 2), pHi, pLo, 0.0f, 0.0f);
                comb.SetValue(i * WS_FLOATS_PER_CORE, am);
                comb.SetValue(i * WS_FLOATS_PER_CORE + 1, av);
                comb.SetValue(i * WS_FLOATS_PER_CORE + 2, ac);
            }
            if (cnt % 2 != 0) { // 奇数时最后一个原样进入下一轮
                for (int64_t k = 0; k < 3; k++) {
                    comb.SetValue(half * WS_FLOATS_PER_CORE + k, comb.GetValue((cnt - 1) * WS_FLOATS_PER_CORE + k));
                }
                half++;
            }
            cnt = half;
        }
        mean = comb.GetValue(0);
        m2 = comb.GetValue(1);
        count = comb.GetValue(2);
    }

    // 预加载本核通道段的 gamma/beta 到 float UB(避免标量 bf16 转换)。
    __aicore__ inline void PreloadGammaBeta(event_t evM2V, event_t evV2S)
    {
        LocalTensor<float> gF = gFBuf.Get<float>();
        LocalTensor<float> bF = bFBuf.Get<float>();
        if (hasGamma) {
            LocalTensor<T2> gRaw = gRawBuf.Get<T2>();
            DataCopyExtParams cpg{1, (uint32_t)(myChan * sizeof(T2)), 0, 0, 0};
            DataCopyPadExtParams<T2> padg{true, 0, (uint8_t)(RoundUp<T2>(myChan) - myChan), 0};
            DataCopyPad(gRaw, gammaGm[gammaBase], cpg, padg);
            SetFlag<HardEvent::MTE2_V>(evM2V);
            WaitFlag<HardEvent::MTE2_V>(evM2V);
            Cast(gF, gRaw, RoundMode::CAST_NONE, myChan);
            PipeBarrier<PIPE_V>();
        }
        if (hasBeta) {
            LocalTensor<T2> bRaw = bRawBuf.Get<T2>();
            DataCopyExtParams cpb{1, (uint32_t)(myChan * sizeof(T2)), 0, 0, 0};
            DataCopyPadExtParams<T2> padb{true, 0, (uint8_t)(RoundUp<T2>(myChan) - myChan), 0};
            DataCopyPad(bRaw, betaGm[gammaBase], cpb, padb);
            SetFlag<HardEvent::MTE2_V>(evM2V);
            WaitFlag<HardEvent::MTE2_V>(evM2V);
            Cast(bF, bRaw, RoundMode::CAST_NONE, myChan);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(evV2S);
        WaitFlag<HardEvent::V_S>(evV2S);
    }

    // Pass2: 逐通道 normalize + silu + quant, 逐块落盘 int8。
    __aicore__ inline void DoPass2Normalize(float mean, float rstd)
    {
        LocalTensor<float> gF = gFBuf.Get<float>();
        LocalTensor<float> bF = bFBuf.Get<float>();
        for (int64_t ci = 0; ci < myChan; ci++) {
            float gVal = hasGamma ? gF.GetValue(ci) : 1.0f;
            float bVal = hasBeta ? bF.GetValue(ci) : 0.0f;
            float qsVal = perChannelScale ? quantScaleGm.GetValue(qsBase + ci) : quantScaleGm.GetValue(0);

            // rstd 与 gamma **不可预先融合**: scl = rstd*gVal 在 |gamma| 近 FLT_MAX 时溢出成
            // ±inf, 随后 (x-mean)*inf 在 x 恒定档(x-mean==0)得 0*inf = **NaN**, 该通道整条
            // 输出报废(L1_267: gamma 范围 ±3.38e38、C=2, 故恰好一个通道全错 = 精度 50.0%)。
            // 公式本身是 (x-mean)*rstd*gamma+beta 的分步乘, 1100/1110/1130 与 golden 均分步,
            // 只有本 key 融合过 —— 回归分步: 0*gamma = 0, 不产生 NaN。
            int64_t chOff = xBase + ci * hwNum;
            NormalizeChannelTiles(chOff, mean, rstd, gVal, bVal, qsVal);
        }
    }

    // 单通道: 逐 TILE 块 normalize + silu + quant, 落盘 int8。
    // 逐通道逐 TILE 走公共 C 层入口(regbase VF), 不再用 Level-2 API 自写 ——
    // silu 与量化收尾因此与 1100/1110/1130 完全一致(silu 用 x/(1+exp(-x)) 而非 Sigmoid API;
    // 量化 Div 后走 fp32->fp16(CAST_ODD)->int8, CAST_ODD 消除双重舍入)。
    // 公共入口按 (x-mean)*rstd*gamma+beta 组织, rstd 与 gamma 分步下发(不可预乘, 见调用处)。
    __aicore__ inline void NormalizeChannelTiles(int64_t chOff, float mean, float rstd, float gVal, float bVal,
                                                 float qsVal)
    {
        for (int64_t off = 0; off < hwNum; off += TILE) {
            int64_t cur = (hwNum - off) < TILE ? (hwNum - off) : TILE;
            LocalTensor<T1> xin = inQue.AllocTensor<T1>();
            DataCopyExtParams cp{1, (uint32_t)(cur * sizeof(T1)), 0, 0, 0};
            DataCopyPadExtParams<T1> pad{true, 0, (uint8_t)(RoundUp<T1>(cur) - cur), 0};
            DataCopyPad(xin, xGm[chOff + off], cp, pad);
            inQue.EnQue(xin);
            xin = inQue.DeQue<T1>();
            LocalTensor<int8_t> yo = outQue.AllocTensor<int8_t>();
            VFNormalizeSiluQuantScalar<T1>((__local_mem__ T1*)xin.GetPhyAddr(), (__local_mem__ int8_t*)yo.GetPhyAddr(),
                                           static_cast<uint32_t>(cur), mean, rstd, gVal, bVal, qsVal, activateSilu);
            inQue.FreeTensor(xin);
            outQue.EnQue(yo);
            yo = outQue.DeQue<int8_t>();
            DataCopyExtParams cpo{1, (uint32_t)(cur * sizeof(int8_t)), 0, 0, 0};
            DataCopyPad(siluGm[chOff + off], yo, cpo);
            outQue.FreeTensor(yo);
        }
    }

private:
    // mean/rstd 各用完全独立的对齐缓冲(mean->gRawBuf, rstd->bRawBuf, 此刻均空闲),各自从对齐 [0] 写,
    // 杜绝 buffer 复用 race 与 DataCopyPad 源非对齐(UB 源须 32B 对齐)。
    // 复用现有 kernel 的 ProcessMeanAndRstd idiom(稳定):mean/rstd 各占 VL_FP32,SetValue 后 S->V,由 ProcessMeanAndRstd
    // 内部 Reg 窄化 + DIST_PACK_B32 掩码存,再 CopyMeanAndRstd2Gm(blockCount=1,copyLen=1)写 GM。
    __aicore__ inline void WriteMeanRstd(float mean, float rstd, event_t evS2V)
    {
        LocalTensor<float> meanT = mrFBuf.Get<float>();
        LocalTensor<float> rstdT = mrFBuf.Get<float>()[VL_FP32];
        meanT.SetValue(0, mean);
        rstdT.SetValue(0, rstd);
        if constexpr (sizeof(T2) == sizeof(float)) {
            // fp32 输出:SetValue(S) 后直接 DataCopyPad(MTE3), 需 S->MTE3
            event_t evS2M3o = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(evS2M3o);
            WaitFlag<HardEvent::S_MTE3>(evS2M3o);
            CopyMeanAndRstd2Gm<float>(meanT2Gm[groupGlobalIdx], rstdT2Gm[groupGlobalIdx], meanT, rstdT, 1, 1);
        } else {
            // T1(half/bf16)输出:ProcessMeanAndRstd 内部 Reg 读 UB(V), 需 S->V
            LocalTensor<T1> meanO = mrOBuf.Get<T1>();
            LocalTensor<T1> rstdO = mrOBuf.Get<T1>()[VL_FP32];
            SetFlag<HardEvent::S_V>(evS2V);
            WaitFlag<HardEvent::S_V>(evS2V);
            ProcessMeanAndRstd<T1>(meanT, meanO, meanGm, rstdT, rstdO, rstdGm, groupGlobalIdx, 1);
        }
    }

    TPipe pipe;
    TQue<TPosition::VECIN, 2> inQue;
    TQue<TPosition::VECOUT, 2> outQue;
    TBuf<> fbuf;    // Pass1 xf / Pass2 中间
    TBuf<> rbuf;    // Pass1 sq / Pass2 yf / rstd 计算 sbuf
    TBuf<> wbuf;    // ReduceSum sharedTmpBuffer
    TBuf<> reduf;   // 归约累加 acc([8]=sumsq [16]=tmp)
    TBuf<> partBuf; // partial(sum,sumsq)写 workspace 暂存
    TBuf<> combBuf; // combine 读回本组全部 partial
    TBuf<> gRawBuf; // gamma 原始(T2)
    TBuf<> bRawBuf; // beta 原始(T2)
    TBuf<> gFBuf;   // gamma float
    TBuf<> bFBuf;   // beta float
    TBuf<> mrFBuf;  // mean/rstd float 暂存(ProcessMeanAndRstd)
    TBuf<> mrOBuf;  // mean/rstd T1 输出暂存

    GlobalTensor<T1> xGm;
    GlobalTensor<T2> gammaGm;
    GlobalTensor<T2> betaGm;
    GlobalTensor<float> quantScaleGm;
    GlobalTensor<int8_t> siluGm;
    GlobalTensor<T1> meanGm;
    GlobalTensor<T1> rstdGm;
    GlobalTensor<T2> meanT2Gm;
    GlobalTensor<T2> rstdT2Gm;
    GlobalTensor<float> wsGm;

    const GroupNormSiluQuantRegbaseTilingData* tiling;
    int64_t blockIdx;
    int64_t numGroups;
    int64_t hwNum;
    int64_t shapeC;
    int64_t shapeD;
    int64_t shapeQuantScale;
    int64_t coresPerGroup;
    int64_t groupGlobalIdx;
    int64_t subIdx;
    int64_t chanPerCore;
    int64_t chanStart;
    int64_t myChan;
    int64_t myElem;
    int64_t xBase;
    int64_t gammaBase;
    int64_t qsBase;
    float eps;
    int64_t groupXBase{0};
    float shiftCScaled{0.0f};
    bool perChannelScale{false};
    bool activateSilu{true};
    bool hasGamma{false};
    bool hasBeta{false};
};

} // namespace GroupNormSiluQuant

#endif // GROUP_NORM_SILU_QUANT_REGBASE_SPLIT_REDUCE_H_
