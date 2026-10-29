/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gru_grad.h
 * \brief GRU 反向 Kernel（dav_3510）—— 基础 API（裸 Mmad/Fixpipe）实现，GM 解耦形态。
 *
 * 架构要点：
 *  - GM 契约：12 输入 / 6 输出 / workspace 段序与 host 侧 GetWorkspaceSize 逐段一致
 *    （前五段输入 dtype + 尾段 fp32 扩展，见 gru_grad_tiling_arch35.cpp）。
 *  - 分块参数：四路 TCubeTiling（host 侧 MultiCoreMatmulTiling 产出）；depthA1/depthB1
 *    由 host 钳制为 2。
 *  - 无 KFC：不使用 matmul::Matmul / REGIST_MATMUL_OBJ / GetSysWorkSpacePtr。
 *  - 同步控制面见 gru_grad_sync.h：时间逆向每步一轮 V->C + C->V，轮次静态配对；
 *    跨簇消费者只有终轮批量 GEMM，由终轮 SyncAll 覆盖。
 *  - 四路 GEMM（C 恒 fp32，输出 dtype 回投由 AIV ConvertOutputs 承担）：
 *    dgateMM 逐步 [B,3H]x[3H,H]->[B,H]；dwIhMM 批量 [3H,TS]x[TS,I]（A=dGiT 转置物化）；
 *    dwHhMM 批量 [3H,TS]x[TS,H]（A=dGhT 物化）；dxMM 批量 [TS,3H]x[3H,I]->[TS,I]。
 */
#ifndef _GRU_GRAD_ARCH35_KERNEL_H_
#define _GRU_GRAD_ARCH35_KERNEL_H_

#include "kernel_operator.h"
#include "../gru_grad_tiling_data.h" // 公用 TilingData（struct GruGradTilingData + CutBatchTiling，本实现全量消费）
#include "gru_grad_sync.h"
#include "gru_grad_cube.h"

using namespace AscendC;

constexpr int64_t GRU_GATE_SIZE = 3;
constexpr int64_t GATE_IDX_RESET = 0;  // [r|z|n] staging 列区首门：reset 门
constexpr int64_t GATE_IDX_UPDATE = 1; // update 门
constexpr int64_t GATE_IDX_NEW = 2;    // new 门
constexpr int64_t AIV_PER_AIC = 2;
constexpr int64_t FLOAT_BYTES = 4;
constexpr int64_t DEFAULT_UB_BUF_ELEMENTS = 4096;
constexpr int64_t L1_ALIGN_BYTES = 512;      // L1/L0C 分形粒度对齐（512B）
constexpr int64_t DG_RESIDENT_MIN_RATIO = 4; // dgate B 侧驻留率至少 1/4 才值得切换
constexpr int64_t ALIGN_32B_FP32 = 8;        // UB fp32 tile 行距对齐（元素）
constexpr int64_t ALIGN_32B_FP32_MASK = 7;
constexpr int64_t BLOCK_BYTES = 32;           // DataCopyPad UB 侧 stride 单位（32B 块）
constexpr int64_t UB_TRANS_BUF_BYTES = 16384; // 转置 pass 的 UB tile 预算（ubTmp）
constexpr int64_t TRANS_GROUP = 16;           // 转置 pass 的 TS 维分组
constexpr int64_t TRANS_STRIP_COLS = 256;     // 条带 run 源行数（staged [8×256]=2048 elem 单次 Gather）
constexpr int64_t TRANS_TILE_COLS = 8;    // 转置列 tile 宽（8 列 fp32 = 32B 块，子带/条带分配粒度）
constexpr int64_t BAND_NUM = 2;           // 带化路径的 N 向等分带数（带宽 hAb=H/2）
constexpr int64_t BAND_H_ALIGN = 128;     // 带化域 H 对齐门槛：hAb=H/2 须 %VL==0 且 >=VL
constexpr int64_t DBUF_SLOTS = 2;         // L1/L0C 双缓冲（乒乓）槽位数
constexpr int64_t RESIDENT_MIN_SLOTS = 2; // dgate B 侧驻留最少槽位（scratch 交替需成对）

// ------------------------------------------------------------------------------------------------
// GateGradsVF —— 门梯度链 VF 寄存器级实现（同仓其他 RNN 类算子的
// __simd_vf__ + asc_vf_call；asc_vf_call 仅 __NPU_ARCH__==3510 展开，其他编译目标为空）。
// 数值与 基线 UB 链逐位一致（同 Sub/Mul 指令序、同操作数配对与舍入点）；
// 门控梯度链的 UB 访问块数显著降低，中间量驻留向量寄存器。
// ------------------------------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void GateGradsVF(__ubuf__ T* drpAddr,  // dst0: drp = dhn·hn·r·(1−r)
                                    __ubuf__ T* dzpAddr,  // dst1: dzp = A·(hp−n)·z
                                    __ubuf__ T* dhnAddr,  // dst2: dhn = A·(1−n²)
                                    __ubuf__ T* dghnAddr, // dst3: dGh_n = r·dhn
                                    __ubuf__ T* dhfhAddr, // dst4: dhFromH = gradH·z
                                    __ubuf__ T* gradHAddr, __ubuf__ T* zAddr, __ubuf__ T* nAddr, __ubuf__ T* hpAddr,
                                    __ubuf__ T* hnAddr, __ubuf__ T* rAddr, uint32_t count, uint32_t oneRepeatSize,
                                    uint16_t repeatTimes);

// ------------------------------------------------------------------------------------------------
// GateGradsScatterVF —— 该计算链 + [r|z|n] staging 布局直写（消除 6 次 UB→UB MTE2
// 装配与 V→M2→M3 双同步，串行链收敛为单 V→M3）。前置条件 hAligned % oneRepeatSize == 0
// （整拍无尾掩码，StoreAlign 地址恒 32B 对齐）；不满足走 原装配路径。
// dGi/dGh staging 各按行写 [drp|dzp|dhn] / [drp|dzp|dGh_n]（drp/dzp 同值双平面双写）；
// dhFromH 就地覆写 hp（chunk 内先载后存、跨 chunk 地址带不相交）。
// ------------------------------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void GateGradsScatterVF(__ubuf__ T* dGiAddr,    // dGi staging [bRows, 3hA]
                                           __ubuf__ T* dGhAddr,    // dGh staging [bRows, 3hA]
                                           __ubuf__ T* dhfhAddr,   // dhFromH 平面（就地覆写 hp）
                                           __ubuf__ T* hpSaveAddr, // hp 保留平面（Tail 消费免 GM 重读）
                                           __ubuf__ T* gradHAddr, __ubuf__ T* zAddr, __ubuf__ T* nAddr,
                                           __ubuf__ T* hpAddr, __ubuf__ T* hnAddr, __ubuf__ T* rAddr, uint32_t bRows,
                                           uint32_t hAligned, uint32_t oneRepeatSize);

// ------------------------------------------------------------------------------------------------
// GateGradsBandVF —— 带化 VF：gradH 折叠 + 全宽 staging 带列直写。
// 相对 GateGradsScatterVF：
//  - 输入 +3：gemmRes（本带 M2 装载于 ubTmp3，带紧凑 [rows, hAb]）、dy（ubNDy 带切片）、
//    dhFromH(t+1)（换出 ubGate 带切片——乒乓换出恰存上步输出，Tail 预取在其后覆写）
//  - 折叠 2 Add：gradH = (gemmRes + dhFromH) + dy（与现行 Accumulate+LoadGradH 同序，
//    IEEE 加法交换律保证逐位一致）后接同 15 算子链
//  - staging 全宽 [rows, 3hA]，本带只写列区 [g*H + bandOff, +hAb)（带间列区不相交，
//    M3 仍全宽单次——免 12 次窄 M3 重构）
//  - dhFromH(t) 就地写 ubHPrev 带列（hp(t) 本带已消费）；hp 保留带列同 scatter
// ------------------------------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void GateGradsBandVF(__ubuf__ T* dGiAddr,     // dGi staging 全宽 [bRows, 3hA]
                                        __ubuf__ T* dGhAddr,     // dGh staging 全宽 [bRows, 3hA]
                                        __ubuf__ T* dhfhOutAddr, // dhFromH(t) 带写出（ubHPrev 带列就地）
                                        __ubuf__ T* hpSaveAddr,  // hp 保留带写（ubDHPrevFromH）
                                        __ubuf__ T* gemmResAddr, // gemmRes 带紧凑 [bRows, hAb]
                                        __ubuf__ T* dyAddr, __ubuf__ T* dhfhInAddr, __ubuf__ T* zAddr,
                                        __ubuf__ T* nAddr, __ubuf__ T* hpAddr, __ubuf__ T* hnAddr, __ubuf__ T* rAddr,
                                        uint32_t bRows, uint32_t hA, uint32_t hAb, uint32_t bandOff,
                                        uint32_t oneRepeatSize);

struct GGemmPlan {
    int64_t M{0};
    int64_t N{0};
    int64_t K{0};
    int64_t singleCoreM{0};
    int64_t singleCoreN{0};
    int64_t usedCoreNum{0};
    int64_t mCoreLoop{0};
    int64_t nCoreLoop{0};
    int64_t mStep{0}; // L0C 子 tile 步长（16 对齐，mStep*nStep <= 64K 元素）
    int64_t nStep{0};
    int64_t kChunk{0}; // L1 K 分块（c0 对齐）
    int64_t l1AElems{0};
    int64_t l1BElems{0};
    int64_t l0CElems{0};
};

template <typename DTYPE>
class GruGradKernel {
public:
    // 512B 向上对齐（L1/L0C 分形粒度）
    __aicore__ inline int64_t AlignUp512(int64_t bytes)
    {
        return (bytes + L1_ALIGN_BYTES - 1) / L1_ALIGN_BYTES * L1_ALIGN_BYTES;
    }

    __aicore__ inline GruGradKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w_input, GM_ADDR w_hidden, GM_ADDR init_h, GM_ADDR output_h,
                                GM_ADDR reset_gate, GM_ADDR update_gate, GM_ADDR new_gate, GM_ADDR h_n, GM_ADDR dy,
                                GM_ADDR dh, GM_ADDR batch_sizes, GM_ADDR dx, GM_ADDR dh_prev, GM_ADDR dw_input,
                                GM_ADDR dw_hidden, GM_ADDR db_input, GM_ADDR db_hidden, const GruGradTilingData* tiling,
                                GM_ADDR workspace)
    {
        this->tiling = tiling;
        this->T = tiling->timeStep;
        this->B = tiling->batch;
        this->H = tiling->hiddenSize;
        this->I = tiling->inputSize;
        this->totalSteps_ = tiling->totalSteps;
        this->vecBTile_ = tiling->singleCoreM;
        this->vecHTile_ = tiling->singleCoreN;

        InitGlobalBuffers(x, w_input, w_hidden, init_h, output_h, reset_gate, update_gate, new_gate, h_n, dy, dh,
                          batch_sizes, dx, dh_prev, dw_input, dw_hidden, db_input, db_hidden, workspace);
        BuildPlans();
        InitVectorBuf();
        this->PlanDgateResident();
        this->PlanWNz();
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            this->CubeProcess();
        }
        if ASCEND_IS_AIV {
            this->VecProcess();
        }
        PipeBarrier<PIPE_ALL>();
    }

    TPipe pipe;

private:
    // -------------------------------------------------------------------------------------------
    // GM 契约（workspace 分段；前六段与基线逐字节一致，尾段为本实现扩展）
    // -------------------------------------------------------------------------------------------
    __aicore__ inline void InitWorkBuffers(__gm__ uint8_t* ws)
    {
        constexpr int64_t dSize = static_cast<int64_t>(sizeof(DTYPE));
        int64_t TS = totalSteps_;
        int64_t off = 0;
        // 注：原实现用 lambda 封装 SetGlobalBuffer；CCE 中 __aicore__ 展开为 [aicore] lambda 属性，
        // 与 setD 的 aicore 调用域冲突导致 fp32/fp16 编译失败（host/aicore 调用域不匹配），改为
        // 私有 __aicore__ 成员函数 SetWorkBuf。
        setD(workGm.dGhGm, ws, off, TS * H * GRU_GATE_SIZE);
        off += TS * H * GRU_GATE_SIZE * dSize;
        setD(workGm.dGiGm, ws, off, TS * H * GRU_GATE_SIZE);
        off += TS * H * GRU_GATE_SIZE * dSize;
        setD(workGm.hPrevWsGm, ws, off, TS * H);
        off += TS * H * dSize;
        // xRevWs [TS,I] 段已删除：本 kernel 零消费（deprecated），host 侧同步移除预留
        setD(workGm.dhPrevWsGm, ws, off, B * H);
        off += B * H * dSize;
        setD(workGm.dhFromHGm, ws, off, B * H);
        off += B * H * dSize;
        // ---- 扩展段（host 侧 GetWorkspaceSize 同序）----
        // dW^T 重排(不变量 #11 B 侧): 只物化 xT[I,TS]/hPrevT[H,TS](体积 6x/3x↓),
        // 复用原 dGiT/dGhT 段(TS*3H ≥ TS*I 和 TS*H)
        setD(xTGm, ws, off, TS * I);
        off += TS * I * dSize;
        setD(hPrevTGm, ws, off, TS * H);
        off += TS * H * dSize;
        dhPrevF32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + off), B * H);
        off += B * H * FLOAT_BYTES;
        dwIF32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + off), H * GRU_GATE_SIZE * I);
        off += H * GRU_GATE_SIZE * I * FLOAT_BYTES;
        dwHF32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + off), H * GRU_GATE_SIZE * H);
        off += H * GRU_GATE_SIZE * H * FLOAT_BYTES;
        dxF32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + off), TS * I);
        off += TS * I * FLOAT_BYTES;
        // wNz 段——w_hh [3H,H] 的 NZ 像(输入 dtype)。host wNzEnabled 时分配 3H*H;
        // kernel 侧 wNzUsable_ 复核失败则不消费(段位闲置)。
        if (this->tiling->wNzEnabled != 0) {
            setD(wNzGm, ws, off, static_cast<int64_t>(H) * GRU_GATE_SIZE * H);
            this->wNzGmValid_ = true;
            off += static_cast<int64_t>(H) * GRU_GATE_SIZE * H * dSize;
        }
        // dgate K 切分部分和段 [aicoreCnt, B, H] fp32——宿主 s1Enabled 时分配
        if (this->tiling->s1Enabled != 0) {
            this->s1PartGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + off),
                                           static_cast<int64_t>(GetBlockNum()) * this->B * this->H);
            this->s1SegValid_ = true;
        }
    }

    // workspace 段绑定辅助：base/off 均为字节粒度（dSize 在调用处折算）
    __aicore__ inline void setD(GlobalTensor<DTYPE>& g, __gm__ uint8_t* base, int64_t off, int64_t elems)
    {
        g.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(base + off), elems);
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR x, GM_ADDR w_input, GM_ADDR w_hidden, GM_ADDR init_h,
                                             GM_ADDR output_h, GM_ADDR reset_gate, GM_ADDR update_gate,
                                             GM_ADDR new_gate, GM_ADDR h_n, GM_ADDR dy, GM_ADDR dh, GM_ADDR batch_sizes,
                                             GM_ADDR dx, GM_ADDR dh_prev, GM_ADDR dw_input, GM_ADDR dw_hidden,
                                             GM_ADDR db_input, GM_ADDR db_hidden, GM_ADDR workspace)
    {
        this->inputGm.xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(x), totalSteps_ * I);
        this->inputGm.wInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(w_input), I * H * GRU_GATE_SIZE);
        this->inputGm.wHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(w_hidden), H * H * GRU_GATE_SIZE);
        this->inputGm.outputHGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(output_h), totalSteps_ * H);
        this->inputGm.dyGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dy), totalSteps_ * H);
        this->inputGm.dhGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dh), B * H);
        this->inputGm.initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(init_h), B * H);
        this->inputGm.resetGateGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(reset_gate), totalSteps_ * H);
        this->inputGm.updateGateGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(update_gate), totalSteps_ * H);
        this->inputGm.newGateGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(new_gate), totalSteps_ * H);
        this->inputGm.hNGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(h_n), totalSteps_ * H);
        this->outputGm.dxGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dx), totalSteps_ * I);
        this->outputGm.dhPrevGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dh_prev), B * H);
        this->outputGm.dwInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dw_input), I * H * GRU_GATE_SIZE);
        this->outputGm.dwHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dw_hidden), H * H * GRU_GATE_SIZE);
        if (this->tiling->isBias == 1) {
            this->outputGm.dbInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(db_input), H * GRU_GATE_SIZE);
            this->outputGm.dbHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(db_hidden), H * GRU_GATE_SIZE);
        }
        if (this->tiling->isSeqLength) {
            this->inputGm.batchSizesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(batch_sizes), T);
        }
        this->InitWorkBuffers(reinterpret_cast<__gm__ uint8_t*>(workspace));
    }

    __aicore__ inline void InitVectorBuf()
    {
        this->pipe.InitBuffer(this->vbGradH, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbHPrev, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbReset, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbUpdate, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbNew, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbHN, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbDHPrevFromH, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbTmp, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbTmp2, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbTmp3, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbGate, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbNReset, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbNUpdate, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbNNew, allocLength * FLOAT_BYTES);
        this->pipe.InitBuffer(this->vbNHN, allocLength * FLOAT_BYTES);
        // VECCALC 池 15×16KB + 8KB = 248KB（ascend950 UB 上限）。
        // 原 16×16KB=256KB 时第 16 块 [248KB,256KB) 与框架保留临时区（counter-mask 模式
        // 的 Duplicate/Cast 掩码暂存）重叠——条件性数据损坏。本缓冲仅承载 dy 预取
        // （原 ubNHPrev 无消费者，复用为 ubNDy），预取超 8KB 域由 pfDyCapOk_ 关闭并
        // 由 LoadGradH 回退 GM 直读。
        this->pipe.InitBuffer(this->vbNHPrev, (allocLength / DBUF_SLOTS) * FLOAT_BYTES); // 半池 8KB=allocLength/2

        this->ubGradH = this->vbGradH.template Get<float>();
        this->ubHPrev = this->vbHPrev.template Get<float>();
        this->ubReset = this->vbReset.template Get<float>();
        this->ubUpdate = this->vbUpdate.template Get<float>();
        this->ubNew = this->vbNew.template Get<float>();
        this->ubHN = this->vbHN.template Get<float>();
        this->ubDHPrevFromH = this->vbDHPrevFromH.template Get<float>();
        this->ubTmp = this->vbTmp.template Get<float>();
        this->ubTmp2 = this->vbTmp2.template Get<float>();
        this->ubTmp3 = this->vbTmp3.template Get<float>();
        this->ubGate = this->vbGate.template Get<float>();
        this->ubNReset = this->vbNReset.template Get<float>();
        this->ubNUpdate = this->vbNUpdate.template Get<float>();
        this->ubNNew = this->vbNNew.template Get<float>();
        this->ubNHN = this->vbNHN.template Get<float>();
        this->ubNHPrev = this->vbNHPrev.template Get<float>();
        // 复用闲置的预取 hPrev 槽位作 dy 预取视图(原 ubNHPrev 无消费者)
        this->ubNDy = this->ubNHPrev;
    }

    // -------------------------------------------------------------------------------------------
    // GEMM 计划：从四路 TCubeTiling 消费分块参数（M/N/singleCoreM/singleCoreN/baseM/baseN/usedCoreNum），
    // 并按 L0C/L1 预算反推子 tile 步长与 K 分块。depthA1/depthB1 已由 host 钳制（raw 路径不消费
    // TPipe 槽位，字段仅作语义不变量保留）。
    // -------------------------------------------------------------------------------------------
    // 按 CUBE_BLOCK(16) 对齐下取整（L0C 子 tile 步长；不足一个分形时取整分形）
    __aicore__ inline int64_t AlignDown16(int64_t v)
    {
        const int64_t blk = static_cast<int64_t>(ggcube::CUBE_BLOCK);
        return v < blk ? blk : (v / blk) * blk;
    }

    __aicore__ inline void SwapTilingMN(TCubeTiling& t)
    {
        int64_t v;
        v = t.M;
        t.M = t.N;
        t.N = v;
        v = t.singleCoreM;
        t.singleCoreM = t.singleCoreN;
        t.singleCoreN = v;
        v = t.baseM;
        t.baseM = t.baseN;
        t.baseN = v;
    }

    __aicore__ inline void BuildOnePlan(GGemmPlan& p, const TCubeTiling& t)
    {
        p.M = t.M;
        p.N = t.N;
        p.K = t.Ka;
        p.singleCoreM = t.singleCoreM > 0 ? t.singleCoreM : t.M;
        p.singleCoreN = t.singleCoreN > 0 ? t.singleCoreN : t.N;
        p.usedCoreNum = t.usedCoreNum > 0 ? t.usedCoreNum : 1;
        p.mCoreLoop = (p.M + p.singleCoreM - 1) / p.singleCoreM;
        p.nCoreLoop = (p.N + p.singleCoreN - 1) / p.singleCoreN;

        int64_t mStep = (t.baseM > 0 && t.baseM <= p.singleCoreM) ? t.baseM : p.singleCoreM;
        int64_t nStep = (t.baseN > 0 && t.baseN <= p.singleCoreN) ? t.baseN : p.singleCoreN;
        // L0C 预算：mStep*nStep <= 64K 元素（fp32）
        nStep = nStep < ggcube::L0C_ELEMS / mStep ? nStep : ggcube::L0C_ELEMS / mStep;
        nStep = AlignDown16(nStep);
        mStep = mStep < ggcube::L0C_ELEMS / nStep ? mStep : ggcube::L0C_ELEMS / nStep;
        mStep = AlignDown16(mStep);
        p.mStep = mStep;
        p.nStep = nStep;

        constexpr int64_t dSize = static_cast<int64_t>(sizeof(DTYPE));
        constexpr int64_t c0 = static_cast<int64_t>(32 / sizeof(DTYPE));
        // ⚠ 片上预算封顶（L0A/L0B 各 64KB）——kChunk 推导需以 L0B 为准。
        // L0B 足迹按【分形数上取整】计：kFrac×nRep×16x16×dsize，其中 kFrac=CeilDiv(k,16)、
        // nRep=CeilDiv(n,16)——kChunk<16 仍占满一个 16 行 k 分形块（）。
        // 故 nStep 先按"kFrac=1 时"封顶：nStep16 ≤ L0B/(256×dsize)（fp32≤1024/fp16≤2048），
        // 再按剩余 kFrac 名额封 kChunk。
        const int64_t nRepCap = static_cast<int64_t>(ggcube::L0B_BYTES) /
                                (static_cast<int64_t>(ggcube::FRACT_ELEMS) * dSize); // 每分形 16x16 元素
        if (nStep > nRepCap * ggcube::CUBE_BLOCK) {
            nStep = nRepCap * ggcube::CUBE_BLOCK;
        }
        // nStep 下限 64（受 N/L0C/nRepCap 封顶）——窄 nStep 使 b 侧 ND2NZ 行宽降至 64B
        // （GM 突发效率差），且 nTile 数与 b 侧搬运 op 数等比膨胀（dx: nStep=16 → 8 nTile × 8 chunk
        // = 64 次 26KB；提升后 2 nTile × 8 chunk = 16 次 64KB）。
        if (nStep < ggcube::CUBE_BLOCK * 4) {
            int64_t nFloor = (p.N < ggcube::CUBE_BLOCK * 4) ?
                                 ((p.N + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK) :
                                 ggcube::CUBE_BLOCK * 4;
            nFloor = nFloor < ggcube::L0C_ELEMS / mStep ? nFloor : ggcube::L0C_ELEMS / mStep;
            nFloor = nFloor < nRepCap * ggcube::CUBE_BLOCK ? nFloor : nRepCap * ggcube::CUBE_BLOCK;
            if (nFloor > nStep) {
                nStep = nFloor;
            }
        }
        p.nStep = nStep;
        const int64_t cubeBlk = static_cast<int64_t>(ggcube::CUBE_BLOCK);
        const int64_t mAl = (mStep + cubeBlk - 1) / cubeBlk * cubeBlk; // M 侧分形对齐
        const int64_t nAl = (nStep + cubeBlk - 1) / cubeBlk * cubeBlk; // N 侧分形对齐
        const int64_t nRep = (nStep + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK;
        const int64_t kFracCap = static_cast<int64_t>(ggcube::L0B_BYTES) /
                                 (static_cast<int64_t>(ggcube::FRACT_ELEMS) * dSize * nRep);
        int64_t kC = ggcube::L1_BUDGET / (dSize * (mAl + nAl));
        // L0A: kC * mAl * dSize <= 64KB（足迹对 kC 线性）
        kC = kC < ggcube::L0A_BYTES / (dSize * mAl) ? kC : ggcube::L0A_BYTES / (dSize * mAl);
        // L0B: kFrac(kC) * nRep * 256 * dSize <= 64KB
        kC = kC < kFracCap * ggcube::CUBE_BLOCK ? kC : kFracCap * ggcube::CUBE_BLOCK;
        kC = (kC / c0) * c0;
        if (kC < c0) {
            kC = c0;
        }
        if (kC > p.K) {
            kC = p.K;
        }
        if (kC < p.K) {
            kC = (kC / cubeBlk) * cubeBlk; // 尾块单独处理，非尾块保持分形对齐（兼 c0 对齐）
            if (kC < cubeBlk) {
                kC = cubeBlk;
            }
        }
        p.kChunk = kC;
        p.l1AElems = ggcube::CeilDiv(static_cast<uint32_t>(kC), ggcube::C0<DTYPE>()) *
                     ggcube::CeilAlign(static_cast<uint32_t>(mStep), ggcube::CUBE_BLOCK) * ggcube::C0<DTYPE>();
        p.l1BElems = ggcube::CeilDiv(static_cast<uint32_t>(nStep), ggcube::C0<DTYPE>()) *
                     ggcube::CeilAlign(static_cast<uint32_t>(kC), ggcube::CUBE_BLOCK) * ggcube::C0<DTYPE>();
        p.l0CElems = ggcube::CeilAlign(static_cast<uint32_t>(mStep), ggcube::CUBE_BLOCK) *
                     ggcube::CeilAlign(static_cast<uint32_t>(nStep), ggcube::CUBE_BLOCK);
    }

    __aicore__ inline void BuildPlans()
    {
        // 带域强制 N 二分——fp32 定长且 H%128==0 时 dgatePlan 的 baseN/singleCoreN
        // 夹到 H/2（dgTileCnt_=2，per-band Fixpipe 提供 B0 提前信号点）。仅触 dgate 一路。
        // H/2≥64 满足 nStep 下限；H/2%16==0 满足对齐；kChunk 由预算自适应（数值容差内）。
        TCubeTiling dgateT = this->tiling->dgateMMParam;
        if constexpr (sizeof(DTYPE) == 4) {
            if (!this->tiling->isSeqLength && (this->H % 128) == 0 && this->H >= 128) {
                const int64_t halfN = this->H / 2;
                if (dgateT.baseN > halfN) {
                    dgateT.baseN = halfN;
                }
                if (dgateT.singleCoreN > halfN) {
                    dgateT.singleCoreN = halfN;
                }
            }
        }
        BuildOnePlan(this->dgatePlan, dgateT);
        { // dW^T = xT @ dGi: M/N 交换消费 host tiling
            TCubeTiling t = this->tiling->dwIhMMParam;
            SwapTilingMN(t);
            BuildOnePlan(this->dwIhPlan, t);
        }
        {
            TCubeTiling t = this->tiling->dwHhMMParam;
            SwapTilingMN(t);
            BuildOnePlan(this->dwHhPlan, t);
        }
        BuildOnePlan(this->dxPlan, this->tiling->dxMMParam);

        // 片上 buffer 取四路计划的最大值，一次分配
        int64_t l1A = dgatePlan.l1AElems;
        l1A = l1A > dwIhPlan.l1AElems ? l1A : dwIhPlan.l1AElems;
        l1A = l1A > dwHhPlan.l1AElems ? l1A : dwHhPlan.l1AElems;
        l1A = l1A > dxPlan.l1AElems ? l1A : dxPlan.l1AElems;
        int64_t l1B = dgatePlan.l1BElems;
        l1B = l1B > dwIhPlan.l1BElems ? l1B : dwIhPlan.l1BElems;
        l1B = l1B > dwHhPlan.l1BElems ? l1B : dwHhPlan.l1BElems;
        l1B = l1B > dxPlan.l1BElems ? l1B : dxPlan.l1BElems;
        int64_t l0C = dgatePlan.l0CElems;
        l0C = l0C > dwIhPlan.l0CElems ? l0C : dwIhPlan.l0CElems;
        l0C = l0C > dwHhPlan.l0CElems ? l0C : dwHhPlan.l0CElems;
        l0C = l0C > dxPlan.l0CElems ? l0C : dxPlan.l0CElems;

        this->l1AElems_ = l1A;
        this->l1BElems_ = l1B;
        this->l0CElems_ = l0C;
        this->l0AElems_ = l1A;
        this->l0BElems_ = l1B;
    }

    // ---------------------------------------------------------------------------------------------
    // dgate B 侧 L1 跨步驻留可行性规划。
    // ）。
    // w_hh 在 T 循环中不变，但其 NZ 像可能超过 L1 预算（无法全驻）——
    // 按 (nt, kChunk) 平铺序驻留前缀槽位，其余 chunk 每步经单 scratch 槽重装；a 侧（dGh 步切片，
    // 体量小）整步突发预装。驻留率不足 25% 或 packed 模式（mExt 逐步可变，回退路径会覆写驻留区）
    // 不启用，回退 原始 路径。
    // ---------------------------------------------------------------------------------------------
    __aicore__ inline void PlanDgateResident()
    {
        const GGemmPlan& p = this->dgatePlan;
        const int64_t dSize = static_cast<int64_t>(sizeof(DTYPE));
        this->dgSlotBBytes_ = AlignUp512(p.l1BElems * dSize);
        this->dgSlotABytes_ = AlignUp512(p.l1AElems * dSize);
        this->dgChunkCnt_ = (p.K > 0 && p.kChunk > 0) ? (p.K + p.kChunk - 1) / p.kChunk : 0;
        this->dgTileCnt_ = (p.N > 0 && p.nStep > 0) ? (p.N + p.nStep - 1) / p.nStep : 0;
        this->dgResident_ = 0;
        if (this->tiling->isSeqLength != 0 || this->dgChunkCnt_ <= 1 || this->dgTileCnt_ < 1) {
            return;
        }
        const int64_t aBurstBytes = this->dgChunkCnt_ * this->dgSlotABytes_;
        if (aBurstBytes + this->dgSlotBBytes_ >= static_cast<int64_t>(ggcube::L1_BUDGET)) {
            return;
        }
        const int64_t avail = static_cast<int64_t>(ggcube::L1_BUDGET) - aBurstBytes - this->dgSlotBBytes_;
        int64_t r = avail / this->dgSlotBBytes_;
        const int64_t total = this->dgChunkCnt_ * this->dgTileCnt_;
        if (r > total) {
            r = total;
        }
        if (r < RESIDENT_MIN_SLOTS || r * DG_RESIDENT_MIN_RATIO < total) { // 驻留率不足则不值得切换
            return;
        }
        this->dgResident_ = r;
    }

    // wNz 路径判定——host 域门 + kernel 复核（驻留失效 / plan 对齐无尾 / kChunk 16 整除）。
    // 通过后 dgate B 侧装载由逐 chunk ND2NZ 变换取预暂存 NZ 像线性拷贝。
    __aicore__ inline void PlanWNz()
    {
        this->wNzUsable_ = false;
        if (!this->wNzGmValid_ || this->tiling->wNzEnabled == 0 || this->tiling->isSeqLength != 0) {
            return;
        }
        const GGemmPlan& p = this->dgatePlan;
        if (this->dgResident_ > 0 || p.K <= 0 || p.kChunk <= 0 || p.N <= 0 || p.nStep <= 0) {
            return;
        }
        if (p.K % p.kChunk != 0 || p.N % p.nStep != 0 || p.kChunk % ggcube::CUBE_BLOCK != 0) {
            return; // 尾块(k/n)与 kChunk<16 分形形态走原 ND2NZ 路径
        }
        this->wNzUsable_ = true;
        // 复核: K 切片数(chunk 数)须 >= AIC 核数(空切片簇会产生未初始化部分和)
        const int64_t chunks = p.K / p.kChunk;
        const int64_t aicCnt = static_cast<int64_t>(GetBlockNum());
        this->s1Usable_ = this->s1SegValid_ && this->tiling->s1Enabled != 0 && chunks >= aicCnt;
    }

    // T 循环前一次性预装 w_hh 驻留槽（(nt, chunk) 平铺序前 dgResident_ 个）
    __aicore__ inline void PreloadDgateB()
    {
        const GGemmPlan& p = this->dgatePlan;
        int64_t loaded = 0;
        for (int64_t nt = 0; nt < this->dgTileCnt_ && loaded < this->dgResident_; ++nt) {
            const int64_t n0 = nt * p.nStep;
            const int64_t nSub = (p.N - n0 < p.nStep) ? (p.N - n0) : p.nStep;
            const int64_t nLd0 = (nSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
            for (int64_t cIdx = 0; cIdx < this->dgChunkCnt_ && loaded < this->dgResident_; ++cIdx) {
                const int64_t k0 = cIdx * p.kChunk;
                const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
                const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                          (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK :
                                          kSub;
                AscendC::LocalTensor<DTYPE> b1(TPosition::B1, loaded * this->dgSlotBBytes_, p.l1BElems);
                ggcube::CopyInNd2Nz<DTYPE>(b1, this->inputGm.wHiddenGm[k0 * this->H + n0], kLoad, nLd0, this->H, kLoad);
                ++loaded;
            }
        }
    }

    // dgate 驻留路径——a 侧整步突发 + B 侧驻留槽直读 / scratch 槽 预发射补装。
    // 同步骨架与基线等价：scratch 的 CopyIn(c+1) 于 Mmad(c) 后预发射（MTE2/M 两管重叠），
    // 下一轮先 WaitMte2ToMte1（等装载齐）再 PIPE_ALL（等 Mmad 退休、L0 可重写）——顺序不得
    // 颠倒，PIPE_ALL 会排空在飞 MTE2 杀死重叠。Mmad 的 K 累加序与 vanilla 逐 chunk 一致，
    // 数值不变。窄事件在 MIX AIC 语境不可靠
    __aicore__ inline void RunDgateResident(GlobalTensor<DTYPE>& aGm, GlobalTensor<float>& cGm, int64_t mExt)
    {
        const GGemmPlan& p = this->dgatePlan;
        const int64_t mSub = ggcube::CUBE_BLOCK; // mExt <= 16，分形整装
        const int64_t aRowPitch = this->H * GRU_GATE_SIZE;
        const int64_t aScrBase = this->dgResident_ * this->dgSlotBBytes_;
        const int64_t bScrOff = aScrBase + this->dgChunkCnt_ * this->dgSlotABytes_;
        const int64_t total = this->dgTileCnt_ * this->dgChunkCnt_;
        AscendC::LocalTensor<DTYPE> b2(TPosition::B2, 0, this->l0BElems_);
        // A4: L0C 双带偏移——每带独立 C tile 区域，消除带间 WAIT_FLAG_FIXP（L0C WAR）。
        // L0C 256KB >> 每带 [16, nStep]×4B，双带仅 ~16KB。带 1 的 MMAD 与带 0 的 Fixpipe
        // 完全并行（不同 L0C 地址无 WAR）。偏移按 512B 对齐（L0C 分形粒度）。
        const int64_t cTileBytes = AlignUp512(mSub * p.nStep * static_cast<int64_t>(sizeof(float)));
        // a 侧全 K 单次 ND2NZ——[mSub, K] 一次入 L1（单 op 替代 dgChunkCnt_ 次 4KB 窄行突发）。
        // 同一全 K 像一次 SplitA 整装 L0A(48KB≤64KB, kStep 跨 k 分形), 各 Mmad 取
        // cIdx×l1AElems 偏移视图——12 次 SplitA/chunk 合并为 1 次/步, Mmad 序不变(数值
        // 逐位一致), 消除 L0A 逐 chunk WAR 与 11 次串行发射(屏障仍由 L0B 的 B 侧 WAR 决定,
        // B 侧 w_hh 远大于 L0B 不可合并)。a1Full 提升作用域供 SplitA 消费。
        const int64_t aFullElems = this->dgChunkCnt_ * p.l1AElems;
        AscendC::LocalTensor<DTYPE> a1Full(TPosition::A1, aScrBase, aFullElems);
        ggcube::CopyInNd2Nz<DTYPE>(a1Full, aGm[0], mSub, p.K, aRowPitch, mSub);
        // 首个 scratch chunk（flat == dgResident_，存在时）随突发一并预发射
        bool scratchInFlight = false;
        if (this->dgResident_ < total) {
            const int64_t nt = this->dgResident_ / this->dgChunkCnt_;
            const int64_t cIdx = this->dgResident_ % this->dgChunkCnt_;
            const int64_t k0 = cIdx * p.kChunk;
            const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
            const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                      (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK :
                                      kSub;
            const int64_t n0 = nt * p.nStep;
            const int64_t nSub = (p.N - n0 < p.nStep) ? (p.N - n0) : p.nStep;
            const int64_t nLd0 = (nSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
            AscendC::LocalTensor<DTYPE> b1(TPosition::B1, bScrOff, p.l1BElems);
            ggcube::CopyInNd2Nz<DTYPE>(b1, this->inputGm.wHiddenGm[k0 * this->H + n0], kLoad, nLd0, this->H, kLoad);
            scratchInFlight = true;
        }
        ggsync::WaitMte2ToMte1(); // 覆盖 a 突发 +（首步）B 预装 + 首个 scratch 预发射
        {
            AscendC::LocalTensor<DTYPE> a2Full(TPosition::A2, 0, aFullElems);
            ggcube::SplitA<DTYPE>(a2Full, a1Full, mSub, p.K);
        }
        for (int64_t nt = 0; nt < this->dgTileCnt_; ++nt) {
            const int64_t n0 = nt * p.nStep;
            const int64_t nSub = (p.N - n0 < p.nStep) ? (p.N - n0) : p.nStep;
            const int64_t nLd0 = (nSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
            // A4: 每带独立 L0C 偏移——带间无 WAR，MMAD(带1) 与 Fixpipe(带0) 并行
            AscendC::LocalTensor<float> c(TPosition::CO1, nt * cTileBytes, this->l0CElems_);
            bool firstK = true;
            for (int64_t cIdx = 0; cIdx < this->dgChunkCnt_; ++cIdx) {
                const int64_t flat = nt * this->dgChunkCnt_ + cIdx;
                const int64_t k0 = cIdx * p.kChunk;
                const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
                const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                          (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK :
                                          kSub;
                if (scratchInFlight) {
                    ggsync::WaitMte2ToMte1(); // scratch(c) 装载完成（须在 PIPE_ALL 前）
                    scratchInFlight = false;
                }
                if (flat > 0) {
                    PipeBarrier<PIPE_ALL>(); // Mmad(c-1) 退休，L0A/L0B/scratch 可重写
                }
                AscendC::LocalTensor<DTYPE> b1(
                    TPosition::B1, (flat < this->dgResident_) ? flat * this->dgSlotBBytes_ : bScrOff, p.l1BElems);
                AscendC::LocalTensor<DTYPE> a2c(TPosition::A2, cIdx * p.l1AElems * static_cast<int64_t>(sizeof(DTYPE)),
                                                p.l1AElems);
                ggcube::SplitBV2<DTYPE>(b2, b1, kLoad, nSub, kLoad); // 单次调用转置(singleK=槽行数)
                ggsync::WaitMte1ToM();
                if (firstK) {
                    ggcube::MmadPlain<float, DTYPE, DTYPE>(c, a2c, b2, mSub, nSub, kSub);
                    firstK = false;
                } else {
                    ggcube::MmadAccum<float, DTYPE, DTYPE>(c, a2c, b2, mSub, nSub, kSub);
                }
                // 预发射下一 chunk（若为 scratch）：与 Mmad 同管外重叠
                if (flat + 1 < total && flat + 1 >= this->dgResident_) {
                    const int64_t nnt = (flat + 1) / this->dgChunkCnt_;
                    const int64_t nc = (flat + 1) % this->dgChunkCnt_;
                    const int64_t nk0 = nc * p.kChunk;
                    const int64_t nkSub = (p.K - nk0 < p.kChunk) ? (p.K - nk0) : p.kChunk;
                    const int64_t nkLoad = (nkSub > ggcube::CUBE_BLOCK) ? (nkSub + ggcube::CUBE_BLOCK - 1) /
                                                                              ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK :
                                                                          nkSub;
                    const int64_t nn0 = nnt * p.nStep;
                    const int64_t nnSub = (p.N - nn0 < p.nStep) ? (p.N - nn0) : p.nStep;
                    const int64_t nnLd0 = (nnSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
                    AscendC::LocalTensor<DTYPE> nb1(TPosition::B1, bScrOff, p.l1BElems);
                    ggcube::CopyInNd2Nz<DTYPE>(nb1, this->inputGm.wHiddenGm[nk0 * this->H + nn0], nkLoad, nnLd0,
                                               this->H, nkLoad);
                    scratchInFlight = true;
                }
            }
            ggsync::WaitMToFix();
            ggcube::FixpipeL0CToGm(cGm[n0], c, mExt, nSub, this->H);
            // A4: WaitFixToM 移除——每带独立 L0C 偏移，带间无 L0C WAR。
            // 带 0 的 Fixpipe 排空不再阻塞带 1 的 MMAD（不同 L0C 区域）。
            // 末带的 WaitFixToM 由 CubeProcess 的 PIPE_ALL + C2V 兜底（L0C 生命周期
            // 跨步复用仍需排空，但排空与带间并行已解耦）。
            // 带 0 提前信号——Fixpipe 发射后即发 B0（不等排空，PIPE_ALL 兜底落地）。
            // 末带追加 B1——早于 FLAG_C2V 一个 PIPE_ALL+set，AIV 预发射 band1 M2。
            if (nt == 0 && this->dgTileCnt_ >= BAND_NUM) { // 仅双带形态存在 band0 提前信号
                PipeBarrier<PIPE_ALL>();
                ggsync::CubeSignalVecBand();
                this->band0Early_ = true;
            } else if (nt == this->dgTileCnt_ - 1 && this->dgTileCnt_ >= BAND_NUM) { // 末带 B1 信号
                PipeBarrier<PIPE_ALL>();
                ggsync::CubeSignalVecBand1();
                this->band1Early_ = true;
            }
        }
    }

    // -------------------------------------------------------------------------------------------
    // 裸 GEMM 执行器：一个 [mExt, nExt] 输出 tile（K 全量），K 分块 + M/N 子 tile 循环。
    // A/B 为 GM(ND)，C 为 GM(fp32, ND)。路径：GM->L1(Nd2Nz)->L0A/L0B(Split)->Mmad->L0C->Fixpipe->GM。
    // -------------------------------------------------------------------------------------------
    __aicore__ inline void RunGemmTiles(const GGemmPlan& p, GlobalTensor<DTYPE>& aGm, int64_t aRowPitch,
                                        GlobalTensor<DTYPE>& bGm, int64_t bRowPitch, GlobalTensor<float>& cGm,
                                        int64_t cRowPitch, int64_t m0, int64_t n0, int64_t mExt, int64_t nExt,
                                        bool batched = false, bool pipelined = false)
    {
        if (mExt <= 0 || nExt <= 0) {
            return;
        }
        // 批路径的 L1 足迹 = SUPER×(chunkA+chunkB), 超预算回退逐 chunk 路径
        constexpr int64_t SUPER = 4;
        if (batched) {
            const int64_t dSize = static_cast<int64_t>(sizeof(DTYPE));
            const int64_t aBytes = AlignUp512(SUPER * p.l1AElems * dSize);
            if (aBytes + SUPER * p.l1BElems * dSize > static_cast<int64_t>(ggcube::L1_BUDGET)) {
                batched = false;
            }
        }
        const int64_t a1Cap = batched ? (SUPER * p.l1AElems) : this->l1AElems_;
        const int64_t b1Cap = batched ? (SUPER * p.l1BElems) : this->l1BElems_;
        AscendC::LocalTensor<DTYPE> a1(TPosition::A1, 0, a1Cap);
        // ⚠ A1/B1 是同一块 L1 的逻辑视角，LocalTensor 构造函数第二参为 L1 内偏移、单位是【字节】
        // （b1 形参: b1(B1, aElems * sizeof(Op), bElems)）。两者都取 0 会互踩；第一版修复
        // 误传元素数（fp32 差 4 倍）仍互踩—— A=0 仍产出非零。按 512B 对齐排 A 之后。
        const int64_t b1OffBytes = this->AlignUp512(a1Cap * static_cast<int64_t>(sizeof(DTYPE)));
        AscendC::LocalTensor<DTYPE> b1(TPosition::B1, b1OffBytes, b1Cap);
        AscendC::LocalTensor<DTYPE> a2(TPosition::A2, 0, this->l0AElems_);
        AscendC::LocalTensor<DTYPE> b2(TPosition::B2, 0, this->l0BElems_);
        AscendC::LocalTensor<float> c(TPosition::CO1, 0, this->l0CElems_);

        const int64_t mTiles = (mExt + p.mStep - 1) / p.mStep;
        const int64_t nTiles = (nExt + p.nStep - 1) / p.nStep;
        for (int64_t mt = 0; mt < mTiles; ++mt) {
            const int64_t mSub0 = mt * p.mStep;
            const int64_t mSubRaw = (mExt - mSub0 < p.mStep) ? (mExt - mSub0) : p.mStep;
            // fp32 Mmad 的 m<16 分形语义未实测（已实测 fp16 m=2 与 m=16/64），为落在实测包络内，
            // 装载/Mmad 恒按 16 行分形整装（GM pad 行读越界只进入 workspace 相邻段，合法且不被泄出），
            // Fixpipe 仅泄出 mSubRaw 个真实行。
            const int64_t mSub = (mSubRaw + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
            for (int64_t nt = 0; nt < nTiles; ++nt) {
                const int64_t nSub0 = nt * p.nStep;
                const int64_t nSub = (nExt - nSub0 < p.nStep) ? (nExt - nSub0) : p.nStep;
                // ⚠ N 侧补满 16 对齐: nSub 非 16 整除时 Nd2Nz 装出半块 NZ n-block, 后续 Mmad 按
                // CeilAlign(n,16) 计算并经 Fixpipe 泄出陈旧 L1 列（读旧值）。
                const int64_t nLd0 = (nSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
                if (pipelined && !batched && p.K > p.kChunk && this->H <= 256) {
                    // 该门: H>256 L1 预算超限
                    // ⚠ 门禁承重：H>256 时 L1 预算超限，勿移除。
                    // 深度-1 流水——L1 双缓冲(消 MTE1→MTE2 WAR 边) + CopyIn(c+1) 预发射(与
                    // Mmad(c) 重叠, MTE2/M 不同管不同存) + Mmad 后窄事件 WaitMToMte1(dW 路径
                    // 已证安全)。单 MTE2 在飞——避开 批装载多描述符突发的未解管线 bug。
                    const int64_t aBuf = p.l1AElems, bBuf = p.l1BElems;
                    const int64_t dS = static_cast<int64_t>(sizeof(DTYPE));
                    const int64_t bOff2 = this->AlignUp512(DBUF_SLOTS * aBuf * dS);
                    if (bOff2 + DBUF_SLOTS * bBuf * dS <= static_cast<int64_t>(ggcube::L1_BUDGET)) {
                        LocalTensor<DTYPE> a1p(TPosition::A1, 0, DBUF_SLOTS * aBuf);
                        LocalTensor<DTYPE> b1p(TPosition::B1, bOff2, DBUF_SLOTS * bBuf);
                        int64_t chunkIdx = 0;
                        bool firstK = true;
                        {
                            const int64_t kSub = (p.K < p.kChunk) ? p.K : p.kChunk;
                            const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                                      (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK *
                                                          ggcube::CUBE_BLOCK :
                                                      kSub;
                            ggcube::CopyInNd2Nz<DTYPE>(a1p[0], aGm[(m0 + mSub0) * aRowPitch], mSub, kSub, aRowPitch,
                                                       mSub);
                            ggcube::CopyInNd2Nz<DTYPE>(b1p[0], bGm[n0 + nSub0], kLoad, nLd0, bRowPitch, kLoad);
                        }
                        for (int64_t k0 = 0; k0 < p.K; k0 += p.kChunk) {
                            const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
                            const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                                      (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK *
                                                          ggcube::CUBE_BLOCK :
                                                      kSub;
                            ggsync::WaitMte2ToMte1();
                            if (chunkIdx > 0) {
                                // Mmad(c-1) 完成后 Split(c) 才可写 L0(L0B 无双缓冲, 64KB 满)。
                                // PIPE_ALL 在此处不抵消预发射: CopyIn(c) 已于上迭代底部发出,
                                // 与 Mmad(c-1) 在 MTE2/M 两管上重叠完成。
                                // ⚠ 窄事件 WaitMToMte1 于本位置实证
                                PipeBarrier<PIPE_ALL>();
                            }
                            const int64_t cur = (chunkIdx % DBUF_SLOTS) * aBuf; // 乒乓当前槽
                            const int64_t curB = (chunkIdx % DBUF_SLOTS) * bBuf;
                            ggcube::SplitA<DTYPE>(a2, a1p[cur], mSub, kSub);
                            ggcube::SplitB<DTYPE>(b2, b1p[curB], kLoad, nSub);
                            ggsync::WaitMte1ToM();
                            if (firstK) {
                                ggcube::MmadPlain<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                                firstK = false;
                            } else {
                                ggcube::MmadAccum<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                            }
                            if (k0 + p.kChunk < p.K) {
                                const int64_t kSubN = (p.K - k0 - p.kChunk < p.kChunk) ? (p.K - k0 - p.kChunk) :
                                                                                         p.kChunk;
                                const int64_t kLoadN = (kSubN > ggcube::CUBE_BLOCK) ?
                                                           (kSubN + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK *
                                                               ggcube::CUBE_BLOCK :
                                                           kSubN;
                                const int64_t nxt = ((chunkIdx + 1) % DBUF_SLOTS) * aBuf; // 预发射下一槽
                                const int64_t nxtB = ((chunkIdx + 1) % DBUF_SLOTS) * bBuf;
                                ggcube::CopyInNd2Nz<DTYPE>(a1p[nxt], aGm[(m0 + mSub0) * aRowPitch + k0 + p.kChunk],
                                                           mSub, kSubN, aRowPitch, mSub);
                                ggcube::CopyInNd2Nz<DTYPE>(b1p[nxtB], bGm[(k0 + p.kChunk) * bRowPitch + n0 + nSub0],
                                                           kLoadN, nLd0, bRowPitch, kLoadN);
                            }
                            ++chunkIdx;
                        }
                    } else {
                        goto vanillaKLoop;
                    }
                    goto gemmDone;
                }
                if (batched) {
                    // 超块批量装载 —— SUPER 个 chunk 的操作数 A/B 一次连发(MTE2 发射共享延迟) +
                    // 单次 WaitMte2ToMte1, 子循环内零 MTE2/零 MTE2 等待; Mmad 后窄事件(M→MTE1),
                    // 超块尾补 MTE1→MTE2(护下一超块 L1 重写)。终轮 GEMM(K=TS, kChunk
                    // L0B 压至 16 → 400 次串行 MTE2 等待)的喂给瓶颈解法。
                    bool firstK = true;
                    for (int64_t sc0 = 0; sc0 < p.K; sc0 += p.kChunk * SUPER) {
                        int64_t nSc = (p.K - sc0 + p.kChunk - 1) / p.kChunk;
                        nSc = (nSc < SUPER) ? nSc : SUPER;
                        for (int64_t ci = 0; ci < nSc; ++ci) {
                            const int64_t k0 = sc0 + ci * p.kChunk;
                            const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
                            const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                                      (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK *
                                                          ggcube::CUBE_BLOCK :
                                                      kSub;
                            ggcube::CopyInNd2Nz<DTYPE>(a1[ci * p.l1AElems], aGm[(m0 + mSub0) * aRowPitch + k0], mSub,
                                                       kSub, aRowPitch, mSub);
                            ggcube::CopyInNd2Nz<DTYPE>(b1[ci * p.l1BElems], bGm[k0 * bRowPitch + n0 + nSub0], kLoad,
                                                       nLd0, bRowPitch, kLoad);
                        }
                        ggsync::WaitMte2ToMte1();
                        for (int64_t ci = 0; ci < nSc; ++ci) {
                            const int64_t k0 = sc0 + ci * p.kChunk;
                            const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
                            const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                                      (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK *
                                                          ggcube::CUBE_BLOCK :
                                                      kSub;
                            ggcube::SplitA<DTYPE>(a2, a1[ci * p.l1AElems], mSub, kSub);
                            ggcube::SplitB<DTYPE>(b2, b1[ci * p.l1BElems], kLoad, nSub);
                            ggsync::WaitMte1ToM();
                            if (firstK) {
                                ggcube::MmadPlain<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                                firstK = false;
                            } else {
                                ggcube::MmadAccum<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                            }
                            // ⚠ M→MTE1 窄事件在 MIX AIC 语境不可靠
                            // dgate 批装载: 换全栅栏。
                            PipeBarrier<PIPE_ALL>();
                        }
                        PipeBarrier<PIPE_ALL>(); // 超块尾全栅栏(MTE1→MTE2 边: 窄事件在此 MIX AIC 语境不可靠,
                                                 // dx 复现)
                    }
                } else {
                vanillaKLoop:
                    for (int64_t k0 = 0; k0 < p.K; k0 += p.kChunk) {
                        const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
                        // B 侧装载 K 填充到 16 倍数：SplitB 的 L1 布局公式仅在被 16 整除的 K 上实测
                        // （k=32/64/128/256）；非整除 K（如 3H=24、TS=21）的分组跨步落在未测区。
                        // Mmad 仍用真实 kSub——"k 恰好按值采用，pad 通道不贡献"（真机实测）。
                        // ⚠ 仅 k>16 时填充：k≤16 的单分形形态已被小 TS 用例实测通过；若强制 16 对齐，
                        // 大 N 时 L0B 足迹 n16*k16*dsize 会
                        // 突破 64KB——）。
                        const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                                  (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK *
                                                      ggcube::CUBE_BLOCK :
                                                  kSub;
                        // GM -> L1（NZ）：srcDValue = 行起点距（元素，实现内部转 gap）
                        ggcube::CopyInNd2Nz<DTYPE>(a1, aGm[(m0 + mSub0) * aRowPitch + k0], mSub, kSub, aRowPitch, mSub);
                        if (pipelined && this->wNzUsable_) {
                            // dgate B 侧(w_hh, 跨 t 不变)读预暂存 NZ 像——线性 GM->L1(消每步
                            // ND2NZ 变换)。槽址 = (nt*chunkCnt + cIdx) * l1Belems, 与 AIV writer 一致。
                            // 位级等价: 像内容 = CopyInNd2Nz 同参输出(一次变换), SplitB/Mmad 不变。
                            const int64_t wnFlat = nt * this->dgChunkCnt_ + k0 / p.kChunk;
                            const int64_t wnOff = wnFlat * p.l1BElems;
                            DataCopy(b1, this->wNzGm[wnOff], static_cast<uint32_t>(kLoad * nLd0));
                        } else {
                            ggcube::CopyInNd2Nz<DTYPE>(b1, bGm[k0 * bRowPitch + n0 + nSub0], kLoad, nLd0, bRowPitch,
                                                       kLoad);
                        }
                        ggsync::WaitMte2ToMte1();
                        // L1 -> L0A/L0B
                        ggcube::SplitA<DTYPE>(a2, a1, mSub, kSub);
                        ggcube::SplitB<DTYPE>(b2, b1, kLoad, nSub);
                        ggsync::WaitMte1ToM();
                        if (k0 == 0) {
                            ggcube::MmadPlain<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                        } else {
                            ggcube::MmadAccum<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                        }
                        PipeBarrier<PIPE_ALL>();
                    }
                }
            gemmDone:
                ggsync::WaitMToFix();
                ggcube::FixpipeL0CToGm(cGm[(m0 + mSub0) * cRowPitch + n0 + nSub0], c, mSubRaw, nSub, cRowPitch);
                ggsync::WaitFixToM();
            }
        }
    }

    // dgate K 维切分 GEMM——全 B 行 × 本簇 K 切片，输出部分和 partials[本簇][B, H]。
    // billed MMAD 由 28 簇各 16 行分形（每核 1-2 真实行）降至 ceil(B/16) 行 × 全 K/28。
    // B 侧走 预暂存 预暂存 NZ 像，A 侧 dGh 全行 ND2NZ。K 累加序为切片内 chunk 序，跨簇按簇序
    // 规约求和（fp32 再结合级差异，数学等价）。
    __aicore__ inline void RunDgateSliceGEMM(int64_t aBase, int64_t kFrom, int64_t kTo)
    {
        const GGemmPlan& p = this->dgatePlan;
        const int64_t aRowPitch = this->H * GRU_GATE_SIZE;
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        GlobalTensor<DTYPE> aGm = this->workGm.dGhGm[aBase];
        GlobalTensor<float> cGm = this->s1PartGm[blockIdx * this->B * this->H];
        AscendC::LocalTensor<DTYPE> a1(TPosition::A1, 0, this->l1AElems_);
        const int64_t b1OffBytes = this->AlignUp512(this->l1AElems_ * static_cast<int64_t>(sizeof(DTYPE)));
        AscendC::LocalTensor<DTYPE> b1(TPosition::B1, b1OffBytes, this->l1BElems_);
        AscendC::LocalTensor<DTYPE> a2(TPosition::A2, 0, this->l0AElems_);
        AscendC::LocalTensor<DTYPE> b2(TPosition::B2, 0, this->l0BElems_);
        AscendC::LocalTensor<float> c(TPosition::CO1, 0, this->l0CElems_);
        const int64_t mTiles = (this->B + p.mStep - 1) / p.mStep;
        const int64_t nTiles = (p.N + p.nStep - 1) / p.nStep;
        for (int64_t mt = 0; mt < mTiles; ++mt) {
            const int64_t mSub0 = mt * p.mStep;
            const int64_t mSubRaw = (this->B - mSub0 < p.mStep) ? (this->B - mSub0) : p.mStep;
            const int64_t mSub = (mSubRaw + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
            for (int64_t nt = 0; nt < nTiles; ++nt) {
                const int64_t nSub0 = nt * p.nStep;
                const int64_t nSub = (p.N - nSub0 < p.nStep) ? (p.N - nSub0) : p.nStep;
                const int64_t nLd0 = (nSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
                for (int64_t k0 = kFrom; k0 < kTo; k0 += p.kChunk) {
                    const int64_t kSub = p.kChunk; // K 切分 域门保证 K%kChunk==0
                    ggcube::CopyInNd2Nz<DTYPE>(a1, aGm[mSub0 * aRowPitch + k0], mSub, kSub, aRowPitch, mSub);
                    const int64_t wnFlat = nt * this->dgChunkCnt_ + k0 / p.kChunk;
                    DataCopy(b1, this->wNzGm[wnFlat * p.l1BElems], static_cast<uint32_t>(kSub * nLd0));
                    ggsync::WaitMte2ToMte1();
                    ggcube::SplitA<DTYPE>(a2, a1, mSub, kSub);
                    ggcube::SplitB<DTYPE>(b2, b1, kSub, nSub);
                    ggsync::WaitMte1ToM();
                    if (k0 == kFrom) {
                        ggcube::MmadPlain<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                    } else {
                        ggcube::MmadAccum<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                    }
                    PipeBarrier<PIPE_ALL>();
                }
                ggsync::WaitMToFix();
                ggcube::FixpipeL0CToGm(cGm[mSub0 * this->H + nSub0], c, mSubRaw, nSub, this->H);
                ggsync::WaitFixToM();
            }
        }
    }

    // -------------------------------------------------------------------------------------------
    // AIC 主流程：时间逆向循环（簇局部 dgate）+ 批量三路 GEMM
    // -------------------------------------------------------------------------------------------
    __aicore__ inline int64_t VecRowStart(int64_t aivIdx)
    {
        const int64_t blockDim = static_cast<int64_t>(GetBlockNum()) * AIV_PER_AIC;
        const int64_t bPerCore = (this->B + blockDim - 1) / blockDim;
        const int64_t rem = this->B % blockDim;
        if (rem == 0) {
            return aivIdx * bPerCore;
        }
        if (aivIdx < rem) {
            return aivIdx * bPerCore;
        }
        return rem * bPerCore + (aivIdx - rem) * (bPerCore - 1);
    }

    __aicore__ inline int64_t VecRowCount(int64_t aivIdx)
    {
        const int64_t blockDim = static_cast<int64_t>(GetBlockNum()) * AIV_PER_AIC;
        const int64_t bPerCore = (this->B + blockDim - 1) / blockDim;
        const int64_t rem = this->B % blockDim;
        int64_t cnt = (rem == 0) ? bPerCore : ((aivIdx < rem) ? bPerCore : bPerCore - 1);
        int64_t start = VecRowStart(aivIdx);
        if (start >= this->B) {
            return 0;
        }
        if (start + cnt > this->B) {
            cnt = this->B - start;
        }
        return cnt;
    }

    // 逐步 dgateMM：AIC c 负责 AIV(2c)/AIV(2c+1) 的行区（簇局部依赖，V2C 栅栏即完备）
    __aicore__ inline void DgateStep(int64_t tIdx)
    {
        const int64_t cluster = static_cast<int64_t>(GetBlockIdx());
        const int64_t aiv0 = cluster * AIV_PER_AIC;
        const int64_t r0 = VecRowStart(aiv0);
        const int64_t r1 = VecRowStart(aiv0 + 1) + VecRowCount(aiv0 + 1);
        int64_t mExt = r1 - r0;
        if (r0 >= this->B || mExt <= 0) {
            return;
        }
        if (r0 + mExt > this->B) {
            mExt = this->B - r0;
        }

        const bool isReverse = (this->tiling->direction != 0);
        const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
        int64_t aBase;
        if (this->tiling->isSeqLength) {
            int64_t compactRow = this->GetCompactRowOffset(gateIdx);
            aBase = compactRow * this->H * GRU_GATE_SIZE;
            int64_t curBatch = this->inputGm.batchSizesGm.GetValue(gateIdx);
            if (r0 >= curBatch) {
                return;
            }
            if (r0 + mExt > curBatch) {
                mExt = curBatch - r0;
            }
        } else {
            aBase = gateIdx * this->B * this->H * GRU_GATE_SIZE;
        }

        GlobalTensor<DTYPE> aGm = this->workGm.dGhGm[aBase + r0 * this->H * GRU_GATE_SIZE];
        GlobalTensor<float> cGm = this->dhPrevF32Gm[r0 * this->H];
        // 已回退: dgate 批装载(SUPER>=2)在三组实验中确定性坏(输入源/工作区中转源均然,
        // dW 同码独过, 根因未定位——kChunk>=64 或 T 循环语境相关); T 循环 GEMM 维持逐 chunk。
        // w_hh B 侧驻留路径优先(单 m-tile 小 M 场景), 大 mExt 回退 深度-1 流水。
        if (this->dgResident_ > 0 && mExt <= ggcube::CUBE_BLOCK) {
            this->RunDgateResident(aGm, cGm, mExt);
            return;
        }
        RunGemmTiles(this->dgatePlan, aGm, this->H * GRU_GATE_SIZE, this->inputGm.wHiddenGm, this->H, cGm, this->H, 0,
                     0, mExt, this->H, false, true); // 深度-1 流水
    }

    __aicore__ inline void RunDxGemm()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t numClusters = static_cast<int64_t>(GetBlockNum());
        const int64_t TS = this->totalSteps_;
        const int64_t bandTS = (TS + numClusters - 1) / numClusters;
        const int64_t dx0 = blockIdx * bandTS;
        if (dx0 < TS) {
            const int64_t dxExt = (TS - dx0 < bandTS) ? TS - dx0 : bandTS;
            if constexpr (sizeof(DTYPE) == 4) {
                // fp32 时 GEMM 直写输出张量, 免 dxF32 中转 + ConvertOnePlane 回投拷贝
                // （m0=dx0 同时索引 a 侧 dGiGm 与 c 侧, dxOut 传全量视图）
                GlobalTensor<float> dxOut;
                dxOut.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(this->outputGm.dxGm.GetPhyAddr(0)),
                                      this->totalSteps_ * this->I);
                RunGemmTiles(this->dxPlan, this->workGm.dGiGm, this->H * GRU_GATE_SIZE, this->inputGm.wInputGm, this->I,
                             dxOut, this->I, dx0, 0, dxExt, this->dxPlan.N, false);
            } else {
                RunGemmTiles(this->dxPlan, this->workGm.dGiGm, this->H * GRU_GATE_SIZE, this->inputGm.wInputGm, this->I,
                             this->dxF32Gm, this->I, dx0, 0, dxExt, this->dxPlan.N, false);
            }
        }
    }

    // dW 2D M×N 分核——终轮已有网格 SyncAll 兜底，无簇内闭包约束，是唯一可自由重分核的 GEMM。
    // 一维 M 带的 3-6 个 nTile 同簇串行（a 侧每 k chunk 随 nT 重复装载 3-6x）+ M-pad
    // （dwHh 1.75x/dwIh 2x）→ 2D tile 网格：nBand=plan.nStep（已满载 L0B），mBand 16 对齐且使
    // tile 总数逼近簇数，tile 索引轮转分簇。每簇单 tile 单 k 链：a 侧随 k 单次装载（消 nT 冗余），
    // chunk-tiles/簇 dwIh 300→100、dwHh 300→50。M<8 时 mBands=1 自然兜底（原 activeClusters=0
    // 整体跳过缺陷同步消除）。
    __aicore__ inline void RunDwGemm2D(const GGemmPlan& p, GlobalTensor<DTYPE>& aGm, int64_t aRowPitch,
                                       GlobalTensor<DTYPE>& bGm, int64_t bRowPitch, GlobalTensor<float>& cGm,
                                       int64_t cRowPitch)
    {
        if (p.M <= 0 || p.N <= 0) {
            return;
        }
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t numClusters = static_cast<int64_t>(GetBlockNum());
        const int64_t nBand = p.nStep;
        const int64_t nBands = (p.N + nBand - 1) / nBand;
        int64_t mBand = (p.M * nBands + numClusters - 1) / numClusters;
        mBand = (mBand + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
        if (mBand < ggcube::CUBE_BLOCK) {
            mBand = ggcube::CUBE_BLOCK;
        }
        const int64_t mBands = (p.M + mBand - 1) / mBand;
        const int64_t totalTiles = mBands * nBands;
        for (int64_t tile = blockIdx; tile < totalTiles; tile += numClusters) {
            const int64_t mb = tile / nBands;
            const int64_t nb = tile % nBands;
            const int64_t m0 = mb * mBand;
            const int64_t mExt = (p.M - m0 < mBand) ? (p.M - m0) : mBand;
            const int64_t n0 = nb * nBand;
            const int64_t nExt = (p.N - n0 < nBand) ? (p.N - n0) : nBand;
            RunGemmTiles(p, aGm, aRowPitch, bGm, bRowPitch, cGm, cRowPitch, m0, n0, mExt, nExt, true);
        }
    }

    __aicore__ inline void RunDwGmms()
    {
        // dW^T 重排(不变量 #11 B 侧): dwIh^T[I,3H] = xT[I,TS] @ dGi[TS,3H]
        //                            dwHh^T[H,3H] = hPrevT[H,TS] @ dGh[TS,3H]
        // B 侧 dGi/dGh ND 直载白拿转置; A 侧为小转置 xT/hPrevT(6x/3x 体积↓)。
        const int64_t TS = this->totalSteps_;
        this->RunDwGemm2D(this->dwIhPlan, this->xTGm, TS, this->workGm.dGiGm, this->H * GRU_GATE_SIZE, this->dwIF32Gm,
                          this->H * GRU_GATE_SIZE);
        this->RunDwGemm2D(this->dwHhPlan, this->hPrevTGm, TS, this->workGm.dGhGm, this->H * GRU_GATE_SIZE,
                          this->dwHF32Gm, this->H * GRU_GATE_SIZE);
    }

    __aicore__ inline void CubeProcess()
    {
        const bool hasRows = VecRowCount(static_cast<int64_t>(GetBlockIdx()) * AIV_PER_AIC) > 0 ||
                             VecRowCount(static_cast<int64_t>(GetBlockIdx()) * AIV_PER_AIC + 1) > 0;
        // w_hh 驻留槽一次性预装（T 循环内不再重复装载）
        if (hasRows && this->dgResident_ > 0) {
            this->PreloadDgateB();
        }
        // K 切分主循环——dgate 全行×本簇 K 切片, 部分和跨簇规约(AIV 侧)。
        // 逐步协议: SyncAll#1(dGh 全簇就绪) → 切片 GEMM → PIPE_ALL(fixpipe 落地, E2/E4 纪律)
        //           → SyncAll#2(部分和发布)。B0/B1/C2V 逐步信号在本域全部旁路(Finals 的
        //           V2C#1/#2 由 AIV Finalize 正常发射, 不受影响)。
        if (this->s1Usable_) {
            const GGemmPlan& p = this->dgatePlan;
            const int64_t chunks = p.K / p.kChunk;
            const int64_t aicCnt = static_cast<int64_t>(GetBlockNum());
            const int64_t bIdx = static_cast<int64_t>(GetBlockIdx());
            // 平衡 chunk 分配: 簇 i 得 [chunks*i/C, chunks*(i+1)/C)——无一空切片
            // (朴素 ceil 均分曾使尾部簇空转, 其 (mt,nt) 循环仍 fixpipe 未初始化 L0C 垃圾进 partial)
            const int64_t c0 = chunks * bIdx / aicCnt;
            const int64_t c1 = chunks * (bIdx + 1) / aicCnt;
            const int64_t kFrom = c0 * p.kChunk;
            const int64_t kTo = c1 * p.kChunk;
            for (int64_t tIdx = this->T - 1; tIdx >= 0; tIdx--) {
                AscendC::SyncAll<false>();
                const int64_t gateIdx = (this->tiling->direction != 0) ? (this->T - 1 - tIdx) : tIdx;
                const int64_t aBase = gateIdx * this->B * this->H * GRU_GATE_SIZE;
                this->RunDgateSliceGEMM(aBase, kFrom, kTo);
                PipeBarrier<PIPE_ALL>();
                AscendC::SyncAll<false>();
            }
            ggsync::CubeWaitVec();
            this->RunDxGemm();      // dx = dGi x w_ih —— 与 AIV 转置重叠(V2C#1)
            ggsync::CubeWaitVec2(); // V2C#2: 本簇 xT/hPrevT 就绪
            this->RunDwGmms();      // dwIh/dwHh —— 需转置产物
            PipeBarrier<PIPE_ALL>();
            ggsync::CubeSignalVec();
            return;
        }
        // 时间逆向循环：每步一轮 V->C（消费一次）+ 一轮 C->V（set 一次）——轮次静态配对
        for (int64_t tIdx = this->T - 1; tIdx >= 0; tIdx--) {
            ggsync::CubeWaitVec();
            this->band0Early_ = false; // 每步复位（RunDgate 的带内置位）
            this->band1Early_ = false; // A1-P3a
            if (hasRows) {
                this->DgateStep(tIdx);
            }
            // 纪律 1——每步每块恰好一次 B0 set：无行块/非带路径（mExt>CUBE_BLOCK 回退、
            // dgResident_==0、单 tile 域）未在 GEMM 内发射者由此补发，与 AIV 的 VecWaitCubeBand
            // 保持轮次静态配对（B0 恒先于 FLAG_C2V 到达，AIV 计算时机不变）。
            if (!this->band0Early_) {
                ggsync::CubeSignalVecBand();
            }
            if (!this->band1Early_) {
                ggsync::CubeSignalVecBand1();
            }
            // Fixpipe 的 GM 写在指令退休后仍可能在 write buffer 中未落地；C2V 置位（PIPE_FIX）
            // 只等指令退休。置位前全停排空，保证消费者（AIV MTE2 读）看到最终数据。
            // ⚠ 消融实证(E2/E4): 本屏障必须 PIPE_ALL——收窄为 PIPE_FIX 或移除均挂死
            //   (flag 协议承重, 非仅数据定序)。
            PipeBarrier<PIPE_ALL>();
            ggsync::CubeSignalVec();
        }
        // 终轮拆两段（dx 重叠转置）:
        //   V2C #1 —— T 循环全部 dGi/dGh 已齐(SyncAll 后), 先跑 dx GEMM(只读 dGi 原布局,
        //             不依赖转置)与 AIV 的 TransposeGiGhT 并行
        //   V2C #2 —— 转置完成后跑 dW 两路(需 dGiT/dGhT)
        ggsync::CubeWaitVec();
        this->RunDxGemm();      // dx = dGi x w_ih —— 与 AIV 转置重叠(V2C#1)
        ggsync::CubeWaitVec2(); // V2C#2: 本簇 xT/hPrevT 就绪
        this->RunDwGmms();      // dwIh/dwHh —— 需转置产物
        PipeBarrier<PIPE_ALL>();
        ggsync::CubeSignalVec();
    }

    // -------------------------------------------------------------------------------------------
    // AIV 主流程
    // -------------------------------------------------------------------------------------------
    // 部分和规约——partials[c][own rows, H] 按簇序 fp32 求和写 dhPrevF32[own rows]。
    // 逐行分段（段长 ≤ allocLength），M2→V / M3→M2 定向同步。调用点位于 SyncAll#2 之后、
    // 下轮 SyncAll#1 之前（单缓冲复用无 WAR）。
    __aicore__ inline void ReduceS1()
    {
        int64_t mStart = 0, mCnt = 0;
        if (!this->GetCoreRows(mStart, mCnt) || mCnt <= 0) {
            return;
        }
        const int64_t aicCnt = static_cast<int64_t>(GetBlockNum());
        const int64_t segMax = this->allocLength;
        for (int64_t r = 0; r < mCnt; ++r) {
            for (int64_t h0 = 0; h0 < this->H; h0 += segMax) {
                const int64_t seg = (this->H - h0 < segMax) ? (this->H - h0) : segMax;
                DataCopy(this->ubTmp, this->s1PartGm[(mStart + r) * this->H + h0], static_cast<uint32_t>(seg));
                SyncM2toV();
                for (int64_t c = 1; c < aicCnt; ++c) {
                    DataCopy(this->ubTmp2, this->s1PartGm[c * this->B * this->H + (mStart + r) * this->H + h0],
                             static_cast<uint32_t>(seg));
                    SyncM2toV();
                    Add(this->ubTmp, this->ubTmp, this->ubTmp2, static_cast<uint32_t>(seg));
                    SyncVtoM2(); // WAR: Add(V 读 ubTmp2) → 下一轮 MTE2 重写 ubTmp2
                }
                SyncVtoM3();
                DataCopy(this->dhPrevF32Gm[(mStart + r) * this->H + h0], this->ubTmp, static_cast<uint32_t>(seg));
                SyncM3toV();
            }
        }
        // MTE3 写(dhPrevF32) → 同核后续 MTE2 读跨管定序：SyncM3toV 仅排 V 管，
        // 不覆盖 MTE2 读者，须全停排空。
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<float> dp;
        dp.SetGlobalBuffer(this->dhPrevF32Gm.GetPhyAddr(mStart * this->H * FLOAT_BYTES), mCnt * this->H);
        DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(dp);
    }

    __aicore__ inline void DcciStepSlices(int64_t tIdx)
    {
        const bool isReverse = (this->tiling->direction != 0);
        const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
        int64_t mStart = 0, mCnt = 0;
        if (!this->GetCoreRows(mStart, mCnt)) {
            return;
        }
        const int64_t cols3H = this->H * GRU_GATE_SIZE;
        // dGi 的跨核读者只有终轮 dx GEMM——逐步 DCCI 延迟到 T 循环后一次性全量刷新
        // （VecProcess 的 V2C#1 前），此处仅刷 dGh（dgate 每步经 V2C 消费）。
        if (this->tiling->isSeqLength) {
            int64_t compactRow = this->GetCompactRowOffset(gateIdx);
            int64_t curBatch = this->inputGm.batchSizesGm.GetValue(gateIdx);
            if (mStart >= curBatch) {
                return;
            }
            if (mStart + mCnt > curBatch) {
                mCnt = curBatch - mStart;
            }
            GlobalTensor<DTYPE> gh;
            gh.SetGlobalBuffer(this->workGm.dGhGm.GetPhyAddr((compactRow + mStart) * cols3H), mCnt * cols3H);
            DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(gh);
            // hp DCCI 移入 ProcessVectorTail(GEMM 窗口)
        } else {
            GlobalTensor<DTYPE> gh;
            gh.SetGlobalBuffer(this->workGm.dGhGm.GetPhyAddr((gateIdx * this->B + mStart) * cols3H), mCnt * cols3H);
            DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(gh);
            // hp DCCI 移入 ProcessVectorTail(GEMM 窗口)
        }
        if (!this->DhPrevUbPath()) { // 快路径下 dhFromHGm 无 GM 读者(Accumulate 直取 UB)
            GlobalTensor<DTYPE> dhf;
            dhf.SetGlobalBuffer(this->workGm.dhFromHGm.GetPhyAddr(mStart * this->H), mCnt * this->H);
            DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(dhf);
        }
    }

    // 一次性写 w_hh [3H,H] 的 dgate-plan NZ 像到 workspace。分形寻址与 SplitBV2 源
    // 布局同构：fractal(kfrac, c0) 槽内偏移 (kfrac + kFrac*c0)*512B，每 c0 列组的
    // [kChunk, c0] GM 步进读恰为 kFrac 个连续分形。槽址 flat = nt*chunkCnt + cIdx，
    // 与消费点一致。借用预循环空闲 TBuf（ubGate/ubTmp/ubTmp2/ubTmp3）批量转写。
    __aicore__ inline void WriteWNzImage()
    {
        const GGemmPlan& p = this->dgatePlan;
        const int64_t aivCnt = static_cast<int64_t>(GetBlockNum()) * AIV_PER_AIC;
        const int64_t aivIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t totalSlots = this->dgTileCnt_ * this->dgChunkCnt_;
        const int64_t kFrac = p.kChunk / ggcube::CUBE_BLOCK;
        const int64_t c0e = ggcube::C0<DTYPE>();
        const int64_t nBlocks = p.nStep / c0e;
        const int64_t sliceElems = p.kChunk * c0e;
        const int64_t dSize = static_cast<int64_t>(sizeof(DTYPE));
        LocalTensor<DTYPE> bufs[4] = {
            this->ubGate.template ReinterpretCast<DTYPE>(), this->ubTmp.template ReinterpretCast<DTYPE>(),
            this->ubTmp2.template ReinterpretCast<DTYPE>(), this->ubTmp3.template ReinterpretCast<DTYPE>()};
        const int64_t ubElems = this->allocLength * FLOAT_BYTES / dSize;
        const int64_t perBuf = ubElems / sliceElems; // 每 TBuf 可容切片数(fp32 4KB 切片 → 4)
        DataCopyPadParams pp;
        pp.isPad = false;
        pp.leftPadding = 0;
        pp.rightPadding = 0;
        pp.paddingValue = 0;
        for (int64_t s = aivIdx; s < totalSlots; s += aivCnt) {
            const int64_t nt = s / this->dgChunkCnt_;
            const int64_t cIdx = s % this->dgChunkCnt_;
            const int64_t k0 = cIdx * p.kChunk;
            const int64_t n0 = nt * p.nStep;
            const int64_t slotBase = s * p.l1BElems;
            for (int64_t g = 0; g < nBlocks;) {
                int64_t done = 0;
                for (int64_t b = 0; b < 4 && g < nBlocks; ++b, g += perBuf) {
                    const int64_t cnt = (nBlocks - g < perBuf) ? (nBlocks - g) : perBuf;
                    for (int64_t j = 0; j < cnt; ++j) {
                        DataCopyParams cp;
                        cp.blockCount = static_cast<uint32_t>(p.kChunk);
                        cp.blockLen = static_cast<uint32_t>(c0e * dSize);
                        cp.srcStride = static_cast<uint32_t>((this->H - c0e) * dSize);
                        cp.dstStride = 0;
                        DataCopyPad(bufs[b][j * sliceElems], this->inputGm.wHiddenGm[k0 * this->H + n0 + (g + j) * c0e],
                                    cp, pp);
                    }
                    done += cnt;
                }
                PipeBarrier<PIPE_ALL>(); // MTE2 批完成(MTE2→MTE3 边)
                int64_t gBase = g - done;
                for (int64_t b = 0; b < 4 && gBase < nBlocks; ++b) {
                    const int64_t cnt = (nBlocks - gBase < perBuf) ? (nBlocks - gBase) : perBuf;
                    for (int64_t j = 0; j < cnt; ++j) {
                        const int64_t gg = gBase + j;
                        DataCopy(this->wNzGm[slotBase + gg * kFrac * (ggcube::CUBE_BLOCK * c0e)],
                                 bufs[b][j * sliceElems], static_cast<uint32_t>(sliceElems));
                    }
                    gBase += cnt;
                }
                PipeBarrier<PIPE_ALL>(); // MTE3 批排空(下一轮 MTE2 重写 TBuf 的 WAR 边)
            }
            // 纪律 5: 跨核读者(AIC MTE2)可见性
            GlobalTensor<DTYPE> view;
            view.SetGlobalBuffer(this->wNzGm.GetPhyAddr(slotBase * dSize), p.l1BElems);
            DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(view);
        }
    }

    __aicore__ inline void VecProcess()
    {
        // 巨型 shape 时先一次性写 w_hh NZ 像(预循环 TBuf 全空闲可借用), SyncAll 全 AIV 背书——
        // 首步 V2C 在其后, AIC 经 CubeWaitVec 消费时像已全局完备。
        if (this->wNzUsable_) {
            this->WriteWNzImage();
            SyncAll();
        }
        // ubGate 常驻 1.0（每 kernel 一次），合并路径免逐步 Duplicate
        Duplicate(this->ubGate, 1.0f, this->allocLength);
        this->InitFastPathCache(); // 快路径判定/行带划分一次算齐（AIV 侧全部消费者）
        if (this->s1Usable_) {     // K 切分主循环（逐步双全局屏障 + 规约）
            for (int64_t tIdx = this->T - 1; tIdx >= 0; tIdx--) {
                this->ProcessVector(tIdx);
                this->DcciStepSlices(tIdx);
                AscendC::SyncAll<false>();     // #1: dGh(tIdx) 全簇就绪 → AIC 切片 GEMM
                this->ProcessVectorTail(tIdx); // GEMM 窗口重叠
                AscendC::SyncAll<false>();     // #2: 部分和全簇就绪
                this->ReduceS1();              // → dhPrevF32[own rows] + DCCI
                this->AccumulateDhPrev(tIdx);
            }
            this->VecProcessFinalize();
            return;
        }
        if (this->bandPath_) { // 带化主循环（首步全 H + 带循环 + 尾步 epilogue）
            this->VecProcessBand();
            return;
        }
        if (this->tiling->isSeqLength) {
            this->InitDhPrev();
        }
        // 时间逆向循环：每步一轮 V->C + 一轮 C->V（双 AIV 都 set / AIC 消费一次；轮次静态配对）
        for (int64_t tIdx = this->T - 1; tIdx >= 0; tIdx--) {
            this->ProcessVector(tIdx);
            this->DcciStepSlices(tIdx); // 纪律 5：dGi/dGh 刷新(hp 已移入 Tail 的 GEMM 窗口)
            ggsync::VecSignalCube();    // V->C（栅栏语义：双 AIV set，AIC 消费一次）
            // GEMM 窗口——StoreHPrev + hp DCCI + 门/dy 预取移入(与 GEMM 重叠);
            // 其 MTE2/MTE3 在 C2V 等待期间异步排空
            this->ProcessVectorTail(tIdx);
            ggsync::VecWaitCubeBand(); // 带 0 提前信号（恒先于 C2V 到达，计算时机不变）
            ggsync::VecWaitCube();     // C->V（dhPrevF32 就绪）。非模板 wait 默认 PIPE_S＝发射流，
                                   // 等待本身阻塞本核后续全部发射，原随后的 PIPE_ALL 冗余，移除。
            this->AccumulateDhPrev(tIdx);
            // 逐步 SyncAll 移除(洁净重试版)。原失败与 vSyncPredone 毒混淆(三处
            // 同步同缺); 毒源修复后分离变量重验: 48 大形状 + 16 边界形状 golden 零失败。
            // WAR 覆盖(单跳 S 阻塞序, 非多跳传递): 读(k) 经 Accumulate 内 SyncM2toV 完成后,
            // S 方发射 V2C(k-1) 的 MTE3 flag → AIC 写(k-1) 不与读(k) 竞争; R6b 时代失败疑因
            // 其基线先于 Tail 同步链引入。栅栏前 AIV 已被握手链近似同步。
            // ⚠ D6(hPrevWs 直读消中转) 曾尝试移除 StoreHPrev 并以 M2→M3/M3→M2 双向窄同步补边——
            // 大形状间歇损坏(1/5~5/5), 存在超出两条已识别 WAR 的隐性耦合(疑乒乓预取与 staging
            // 的跨轮交互), 已整体回退。教训: StoreHPrev 的同步链是 T 循环管序的隐性承重墙。
        }
        this->VecProcessFinalize();
    }

    // 终局段抽取（原 VecProcess T 循环后代码原样迁移）——带化/非带化两路共用。
    __aicore__ inline void VecProcessFinalize()
    {
        // ⚠ T 循环后必须一道全栅栏：TransposeGiGhT 按列带转置、读全部 TS 行——跨簇消费
        // 所有 AIV 的逐步 dGi/dGh 写出（小形状删除后 dW 两路错、
        // ）。逐步 DCCI 已保证可见性，此处只补完成序。
        SyncAll();
        // dGi 一次性全量 DCCI——其唯一跨核读者是终轮 dx GEMM（V2C#1 后），逐步刷新无必要
        {
            GlobalTensor<DTYPE> gi;
            gi.SetGlobalBuffer(this->workGm.dGiGm.GetPhyAddr(0), this->totalSteps_ * this->H * GRU_GATE_SIZE);
            DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(gi);
        }
        ggsync::VecSignalCube(); // V2C#1(FLAG 9): dGi/dGh 全局齐, AIC 先跑 dx(与转置重叠)
        this->TransposeGiGhT();
        // ⚠ 转置尾部 DCCI(缓存清刷)不在 V2C#2 的 PIPE_MTE3 定序内——flag 抢跑 cache clean,
        // AIC 读到未刷写的 xT/hPrevT。
        PipeBarrier<PIPE_ALL>();
        // 2D 分核使 dW 的 A 侧(xT/hPrevT)跨簇消费——V2C#2 仅保证本簇转置完成，
        // 置位前须网格栅栏为全部簇的转置产物背书（沿用终轮 C2V 后 SyncAll 的 AIV 间背书模式；
        // dGi/dGh 侧已由 T 循环后 SyncAll + 全量 DCCI 覆盖）。
        SyncAll();
        ggsync::VecSignalCube2(); // V2C#2(FLAG 10, 独立 ID): 转置全局完成, AIC 跑 dW GEMMs
        this->StoreDhPrev();
        // bias 输出——恒走 ProcessBiasReduce GM 归约（dGi/dGh [TS,3H] 列和 = 全 (T,B) 行贡献）。
        // 原 band+biasInline 域（H∈{128,256}）的 UB 部分和全宽直写在 ≥2 AIV 有行时
        // 构成多写竞争且缺他核行贡献（db 终值不确定），已弃用（原优化自评性能中性）。
        if (this->tiling->isBias == 1) {
            int64_t rows = this->totalSteps_;
            int64_t cols = this->H * GRU_GATE_SIZE;
            this->ProcessBiasReduce(this->workGm.dGiGm, this->outputGm.dbInputGm, rows, cols);
            this->ProcessBiasReduce(this->workGm.dGhGm, this->outputGm.dbHiddenGm, rows, cols);
        }
        ggsync::VecWaitCube(); // 终轮 C->V（本簇 AIC 的 GEMM 完成）
        PipeBarrier<PIPE_ALL>();
        // ⚠ dW 两路 convert 行带在 N 维（cols3H）而 GEMM M 带在 M 维——簇 c 的 convert 读 dwIF32/
        // dwHF32 全部 M 行（跨簇产物），仅本簇 C2V 不足。AIV 间 SyncAll：各 AIV 为本簇 AIC 完成背书，
        // 全员通过即全簇 GEMM 完成（周期 28 零行实证：GEMM 输出正确但 convert 读到未写区）。
        SyncAll();
        this->ConvertOutputs();
    }

    __aicore__ inline int64_t GetCompactRowOffset(int64_t seqIdx)
    {
        int64_t offset = 0;
        for (int64_t i = 0; i < seqIdx; i++) {
            offset += this->inputGm.batchSizesGm.GetValue(i);
        }
        return offset;
    }

    __aicore__ inline bool GetCoreRows(int64_t& mStart, int64_t& mCnt)
    {
        // 行带划分每核恒定（B/blockDim/块序 kernel 内不变）——首调缓存，消逐步
        // 4-5 处重复的整除/取模/分支（ESL: AIV 段标量风暴成分之一）。
        if (!this->coreRowsInit_) {
            int64_t blockDim = static_cast<int64_t>(GetBlockNum()) * AIV_PER_AIC;
            int64_t B_per_core = (this->B + blockDim - 1) / blockDim;
            int64_t rem = this->B % blockDim;
            int64_t blockIdx = GetBlockIdx();
            if (rem == 0 || blockIdx < rem) {
                this->mCnt_ = B_per_core;
                this->mStart_ = blockIdx * B_per_core;
            } else {
                this->mCnt_ = B_per_core - 1;
                this->mStart_ = rem * B_per_core + (blockIdx - rem) * (B_per_core - 1);
            }
            this->coreRowsOk_ = (this->mCnt_ > 0 && this->mStart_ < this->B);
            this->coreRowsInit_ = true;
        }
        mStart = this->mStart_;
        mCnt = this->mCnt_;
        return this->coreRowsOk_;
    }

#include "gru_grad_vector_arch35.h"

private:
    struct InputGm {
        __aicore__ inline InputGm() = default;
        GlobalTensor<DTYPE> xGm;
        GlobalTensor<DTYPE> wInputGm;
        GlobalTensor<DTYPE> wHiddenGm;
        GlobalTensor<DTYPE> outputHGm;
        GlobalTensor<DTYPE> dyGm;
        GlobalTensor<DTYPE> dhGm;
        GlobalTensor<DTYPE> initHGm;
        GlobalTensor<DTYPE> resetGateGm;
        GlobalTensor<DTYPE> updateGateGm;
        GlobalTensor<DTYPE> newGateGm;
        GlobalTensor<DTYPE> hNGm;
        GlobalTensor<int64_t> batchSizesGm;
    };
    struct OutputGm {
        __aicore__ inline OutputGm() = default;
        GlobalTensor<DTYPE> dxGm;
        GlobalTensor<DTYPE> dhPrevGm;
        GlobalTensor<DTYPE> dwInputGm;
        GlobalTensor<DTYPE> dwHiddenGm;
        GlobalTensor<DTYPE> dbInputGm;
        GlobalTensor<DTYPE> dbHiddenGm;
    };
    struct WorkGm {
        __aicore__ inline WorkGm() = default;
        GlobalTensor<DTYPE> dGhGm;
        GlobalTensor<DTYPE> dGiGm;
        GlobalTensor<DTYPE> hPrevWsGm;
        GlobalTensor<DTYPE> dhPrevWsGm;
        GlobalTensor<DTYPE> dhFromHGm;
    };

    InputGm inputGm;
    OutputGm outputGm;
    WorkGm workGm;
    // 扩展 workspace
    GlobalTensor<DTYPE> xTGm;
    GlobalTensor<DTYPE> hPrevTGm;
    GlobalTensor<float> dhPrevF32Gm;
    GlobalTensor<float> dwIF32Gm;
    GlobalTensor<float> dwHF32Gm;
    GlobalTensor<float> dxF32Gm;
    GlobalTensor<DTYPE> wNzGm;    // w_hh NZ 像预暂存段
    bool wNzGmValid_{false};      // host 已分配该段
    bool wNzUsable_{false};       // kernel 复核通过(writer/consumer 双方共同判据)
    GlobalTensor<float> s1PartGm; // dgate K 切分部分和段 [aicoreCnt, B, H]
    bool s1SegValid_{false};      // host 已分配该段
    bool s1Usable_{false};        // 复核通过(K 切片非空 + wNz 域)

    TBuf<TPosition::VECCALC> vbGradH, vbHPrev, vbReset, vbUpdate, vbNew, vbHN, vbDHPrevFromH;
    TBuf<TPosition::VECCALC> vbTmp, vbTmp2, vbTmp3;
    TBuf<TPosition::VECCALC> vbGate;
    // 门平面预取双缓冲(下一拍 r/z/n/hn/hPrev)
    TBuf<TPosition::VECCALC> vbNReset, vbNUpdate, vbNNew, vbNHN, vbNHPrev;
    LocalTensor<float> ubGradH, ubHPrev, ubReset, ubUpdate, ubNew, ubHN, ubDHPrevFromH;
    LocalTensor<float> ubTmp, ubTmp2, ubTmp3;
    LocalTensor<float> ubGate;
    LocalTensor<float> ubNReset, ubNUpdate, ubNNew, ubNHN, ubNHPrev;
    LocalTensor<float> ubNDy; // dy 预取(复用 ubNHPrev 槽位)
    // bias 累加器——ubGate 尾部别名视图（band 路径驻留安全区）
    // 无新 TBuf（VECCALC 池 248KB 已满），[2048, 2048+6hA) 区段与 hp 预取 [0, mCnt*hA) 无交叠
    bool gatesPrefetched{false}; // 门平面预取单次消费标志
    bool dyPrefetched{false};    // dy 预取独立标志（与 gatesPrefetched 解耦，容量门独立）
    // 快路径判定缓存（形状派生量 kernel 内恒定，VecProcess 入口一次算齐）
    bool coreRowsInit_{false}; // GetCoreRows 缓存有效位
    bool coreRowsOk_{false};
    int64_t mStart_{0};
    int64_t mCnt_{0};
    bool fastPathInit_{false};  // 谓词缓存有效位
    bool mergedShapeOk_{false}; // H%alignPara==0（merged/scatter 共同前置）
    bool scatterVOk_{false};    // hiddenSizeAligned % VL == 0（整拍约束）
    int64_t stagingCapRows_{0}; // allocLength / (3*hiddenSizeAligned)
    bool dhPrevUbPath_{false};  // DhPrevUbPath 全判定
    bool pfCapOk_{false};       // 预取容量/单tile判定
    bool pfDyCapOk_{false};     // dy 预取专属容量门（ubNDy 槽位仅 8KB，超域回退 GM 直读）
    bool hAlign8Ok_{false};     // H%alignPara==0（hp 预取门）
    bool vfMte2Dirty_{false};   // Accumulate 同步后是否有新 MTE2 喂 VF（scatter 据此条件同步）
    bool band0Early_{false};    // 本步已发 B0 提前信号（CubeProcess 补发配对）
    bool band1Early_{false};    // 本步已发 B1 末带提前信号（同上补发配对）
    bool bandPath_{false};      // 带化 AIV 主循环分派（InitFastPathCache 一次算齐）
    bool dhFromHInHp_{false};   // dhFromH 常驻缓冲位置（true=ubHPrev 合并路径）
    bool hpPreserved_{false};   // 本步 scatter 已将 hp 保留至 ubDHPrevFromH（Tail 消费）
    bool hpPrefetched{false};   // hp(t-1) 已预取至 ubGate（单次消费，乒乓换入）

    const GruGradTilingData* tiling{nullptr};
    int64_t T{0};
    int64_t B{0};
    int64_t H{0};
    int64_t I{0};
    int64_t vecBTile_{0};
    int64_t vecHTile_{0};
    int64_t allocLength{DEFAULT_UB_BUF_ELEMENTS};
    int64_t totalSteps_{0};
    int64_t alignPara{ALIGN_32B_FP32};
    int64_t alignMask{ALIGN_32B_FP32_MASK};

    GGemmPlan dgatePlan;
    GGemmPlan dwIhPlan;
    GGemmPlan dwHhPlan;
    GGemmPlan dxPlan;
    // dgate B 侧 L1 驻留规划（0 = 回退 原始）
    int64_t dgResident_{0};
    int64_t dgSlotBBytes_{0};
    int64_t dgSlotABytes_{0};
    int64_t dgChunkCnt_{0};
    int64_t dgTileCnt_{0};
    int64_t l1AElems_{0};
    int64_t l1BElems_{0};
    int64_t l0AElems_{0};
    int64_t l0BElems_{0};
    int64_t l0CElems_{0};
};

// GateGradsVF 实现。寄存器调度（与 基线 UB 链逐操作对应）：
//   aVal = gradH·(1−z)（先算后双消费：dhn / dzp）；t1 = dhn = aVal·(1−n²)；
//   t2 = dzp = aVal·(hp−n)·z；t3 = drp = dhn·hn·r·(1−r)；aVal 复用持 dGh_n = r·dhn；
//   t4 = dhFromH = gradH·z。
// ubHN(hn→dGh_n) / ubHPrev(hp→dhFromH) 就地覆写：repeat 内先载后存、跨 repeat 地址带不相交
// （thnn HyTailVF ng→hy 同构，生产验证）。UpdateMask 消费 remaining（尾拍掩码，bn 同款）。
template <typename T>
__simd_vf__ inline void GateGradsVF(__ubuf__ T* drpAddr, __ubuf__ T* dzpAddr, __ubuf__ T* dhnAddr, __ubuf__ T* dghnAddr,
                                    __ubuf__ T* dhfhAddr, __ubuf__ T* gradHAddr, __ubuf__ T* zAddr, __ubuf__ T* nAddr,
                                    __ubuf__ T* hpAddr, __ubuf__ T* hnAddr, __ubuf__ T* rAddr, uint32_t count,
                                    uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> gradHReg, zReg, nReg, hpReg, hnReg, rReg;
    AscendC::Reg::RegTensor<T> oneReg, aVal, t1, t2, t3, t4;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(gradHReg, gradHAddr, aReg);
        AscendC::Reg::LoadAlign(zReg, zAddr, aReg);
        AscendC::Reg::LoadAlign(nReg, nAddr, aReg);
        AscendC::Reg::LoadAlign(hpReg, hpAddr, aReg);
        AscendC::Reg::LoadAlign(hnReg, hnAddr, aReg);
        AscendC::Reg::LoadAlign(rReg, rAddr, aReg);
        AscendC::Reg::Duplicate(oneReg, (T)1.0, mask);
        AscendC::Reg::Sub(aVal, oneReg, zReg, mask);   // 1−z
        AscendC::Reg::Mul(aVal, gradHReg, aVal, mask); // A = gradH·(1−z)
        AscendC::Reg::Mul(t1, nReg, nReg, mask);       // n²
        AscendC::Reg::Sub(t1, oneReg, t1, mask);       // 1−n²
        AscendC::Reg::Mul(t1, aVal, t1, mask);         // dhn = A·(1−n²)
        AscendC::Reg::Sub(t2, hpReg, nReg, mask);      // hp−n
        AscendC::Reg::Mul(t2, t2, zReg, mask);         // (hp−n)·z
        AscendC::Reg::Mul(t2, aVal, t2, mask);         // dzp = A·(hp−n)·z
        AscendC::Reg::Mul(t3, t1, hnReg, mask);        // dhn·hn
        AscendC::Reg::Mul(t3, t3, rReg, mask);         // ·r
        AscendC::Reg::Sub(aVal, oneReg, rReg, mask);   // 1−r（A 已消费）
        AscendC::Reg::Mul(t3, t3, aVal, mask);         // drp = dhn·hn·r·(1−r)
        AscendC::Reg::Mul(aVal, rReg, t1, mask);       // dGh_n = r·dhn
        AscendC::Reg::Mul(t4, gradHReg, zReg, mask);   // dhFromH = gradH·z
        AscendC::Reg::StoreAlign(drpAddr, t3, aReg, mask);
        AscendC::Reg::StoreAlign(dzpAddr, t2, aReg, mask);
        AscendC::Reg::StoreAlign(dhnAddr, t1, aReg, mask);
        AscendC::Reg::StoreAlign(dghnAddr, aVal, aReg, mask);
        AscendC::Reg::StoreAlign(dhfhAddr, t4, aReg, mask);
    }
}

// GateGradsScatterVF 实现。行外层 × 拍内层（chunks = hAligned/VL）：
//   输入/dhFromH 按线性偏移 row*hA + c*VL 寻址；staging 按行基 row*3hA + 门偏 g*hA + c*VL。
//   计算链与 GateGradsVF 逐操作一致（数值逐位等价）；drp/dzp 寄存器双消费双平面直写。
//   全满拍（hA%VL==0 由分派保证）→ CreateMask 常量全满掩码，无 UpdateMask 尾处理。
template <typename T>
__simd_vf__ inline void GateGradsScatterVF(__ubuf__ T* dGiAddr, __ubuf__ T* dGhAddr, __ubuf__ T* dhfhAddr,
                                           __ubuf__ T* hpSaveAddr, __ubuf__ T* gradHAddr, __ubuf__ T* zAddr,
                                           __ubuf__ T* nAddr, __ubuf__ T* hpAddr, __ubuf__ T* hnAddr, __ubuf__ T* rAddr,
                                           uint32_t bRows, uint32_t hAligned, uint32_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<T> gradHReg, zReg, nReg, hpReg, hnReg, rReg;
    AscendC::Reg::RegTensor<T> oneReg, aVal, t1, t2, t3, t4;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<T>();
    AscendC::Reg::Duplicate(oneReg, (T)1.0);
    const uint32_t chunks = hAligned / oneRepeatSize;
    for (uint32_t row = 0; row < bRows; ++row) {
        const uint32_t rowOff = row * hAligned;
        __ubuf__ T* dGiR = dGiAddr + row * GRU_GATE_SIZE * hAligned;
        __ubuf__ T* dGhR = dGhAddr + row * GRU_GATE_SIZE * hAligned;
        for (uint32_t c = 0; c < chunks; ++c) {
            const uint32_t off = rowOff + c * oneRepeatSize;
            const uint32_t cOff = c * oneRepeatSize;
            AscendC::Reg::LoadAlign(gradHReg, gradHAddr + off);
            AscendC::Reg::LoadAlign(zReg, zAddr + off);
            AscendC::Reg::LoadAlign(nReg, nAddr + off);
            AscendC::Reg::LoadAlign(hpReg, hpAddr + off);
            AscendC::Reg::LoadAlign(hnReg, hnAddr + off);
            AscendC::Reg::LoadAlign(rReg, rAddr + off);
            AscendC::Reg::Sub(aVal, oneReg, zReg, mask);                                  // 1−z
            AscendC::Reg::Mul(aVal, gradHReg, aVal, mask);                                // A = gradH·(1−z)
            AscendC::Reg::Mul(t1, nReg, nReg, mask);                                      // n²
            AscendC::Reg::Sub(t1, oneReg, t1, mask);                                      // 1−n²
            AscendC::Reg::Mul(t1, aVal, t1, mask);                                        // dhn = A·(1−n²)
            AscendC::Reg::Sub(t2, hpReg, nReg, mask);                                     // hp−n
            AscendC::Reg::Mul(t2, t2, zReg, mask);                                        // (hp−n)·z
            AscendC::Reg::Mul(t2, aVal, t2, mask);                                        // dzp = A·(hp−n)·z
            AscendC::Reg::Mul(t3, t1, hnReg, mask);                                       // dhn·hn
            AscendC::Reg::Mul(t3, t3, rReg, mask);                                        // ·r
            AscendC::Reg::Sub(aVal, oneReg, rReg, mask);                                  // 1−r（A 已消费）
            AscendC::Reg::Mul(t3, t3, aVal, mask);                                        // drp = dhn·hn·r·(1−r)
            AscendC::Reg::Mul(aVal, rReg, t1, mask);                                      // dGh_n = r·dhn
            AscendC::Reg::Mul(t4, gradHReg, zReg, mask);                                  // dhFromH = gradH·z
            AscendC::Reg::StoreAlign(dGiR + GATE_IDX_RESET * hAligned + cOff, t3, mask);  // 门 r
            AscendC::Reg::StoreAlign(dGiR + GATE_IDX_UPDATE * hAligned + cOff, t2, mask); // 门 z
            AscendC::Reg::StoreAlign(dGiR + GATE_IDX_NEW * hAligned + cOff, t1, mask);    // 门 n
            AscendC::Reg::StoreAlign(dGhR + GATE_IDX_RESET * hAligned + cOff, t3, mask);  // 门 r
            AscendC::Reg::StoreAlign(dGhR + GATE_IDX_UPDATE * hAligned + cOff, t2, mask); // 门 z
            AscendC::Reg::StoreAlign(dGhR + GATE_IDX_NEW * hAligned + cOff, aVal, mask);  // 门 n
            AscendC::Reg::StoreAlign(dhfhAddr + off, t4, mask);
            AscendC::Reg::StoreAlign(hpSaveAddr + off, hpReg, mask); // hp 保留（Tail 免 GM 重读）
        }
    }
}

// GateGradsBandVF 实现。地址口径：
//   全宽输入/输出（dy/dhfhIn/z/n/hp/hn/r/dhfhOut/hpSave，行距 hA）: off = row*hA + bandOff + cOff
//   gemmRes 带紧凑（行距 hAb）: off = row*hAb + cOff
//   staging 全宽（行距 3hA）: 列区 [g*hA + bandOff + cOff)
// 折叠序与现行 Accumulate+LoadGradH 逐位一致：((gemmRes+dhFromH)+dy)，IEEE 加法交换律保证
// 加数对调无损。band0 与 band1 列区/hp 带列均不相交，跨带无 WAR（gemmRes 的 ubTmp3 复用由
// 带间 SyncVtoM2 覆盖——见 VecProcessBand）。
template <typename T>
__simd_vf__ inline void GateGradsBandVF(__ubuf__ T* dGiAddr, __ubuf__ T* dGhAddr, __ubuf__ T* dhfhOutAddr,
                                        __ubuf__ T* hpSaveAddr, __ubuf__ T* gemmResAddr, __ubuf__ T* dyAddr,
                                        __ubuf__ T* dhfhInAddr, __ubuf__ T* zAddr, __ubuf__ T* nAddr,
                                        __ubuf__ T* hpAddr, __ubuf__ T* hnAddr, __ubuf__ T* rAddr, uint32_t bRows,
                                        uint32_t hA, uint32_t hAb, uint32_t bandOff, uint32_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<T> gradHReg, zReg, nReg, hpReg, hnReg, rReg;
    AscendC::Reg::RegTensor<T> gResReg, dyReg, dhInReg;
    AscendC::Reg::RegTensor<T> oneReg, aVal, t1, t2, t3, t4;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<T>();
    AscendC::Reg::Duplicate(oneReg, (T)1.0);
    const uint32_t chunks = hAb / oneRepeatSize;
    for (uint32_t row = 0; row < bRows; ++row) {
        const uint32_t offFull = row * hA + bandOff;
        const uint32_t offComp = row * hAb;
        __ubuf__ T* dGiR = dGiAddr + row * GRU_GATE_SIZE * hA;
        __ubuf__ T* dGhR = dGhAddr + row * GRU_GATE_SIZE * hA;
        for (uint32_t c = 0; c < chunks; ++c) {
            const uint32_t cOff = c * oneRepeatSize;
            AscendC::Reg::LoadAlign(gResReg, gemmResAddr + offComp + cOff); // dhPrevF32(t+1) 带
            AscendC::Reg::LoadAlign(dyReg, dyAddr + offFull + cOff);        // dy(t) 带
            AscendC::Reg::LoadAlign(dhInReg, dhfhInAddr + offFull + cOff);  // dhFromH(t+1) 带
            AscendC::Reg::Add(gradHReg, gResReg, dhInReg, mask);            // gemmRes + dhFromH
            AscendC::Reg::Add(gradHReg, gradHReg, dyReg, mask);             // + dy = gradH(t)
            AscendC::Reg::LoadAlign(zReg, zAddr + offFull + cOff);
            AscendC::Reg::LoadAlign(nReg, nAddr + offFull + cOff);
            AscendC::Reg::LoadAlign(hpReg, hpAddr + offFull + cOff);
            AscendC::Reg::LoadAlign(hnReg, hnAddr + offFull + cOff);
            AscendC::Reg::LoadAlign(rReg, rAddr + offFull + cOff);
            AscendC::Reg::Sub(aVal, oneReg, zReg, mask);   // 1−z
            AscendC::Reg::Mul(aVal, gradHReg, aVal, mask); // A = gradH·(1−z)
            AscendC::Reg::Mul(t1, nReg, nReg, mask);       // n²
            AscendC::Reg::Sub(t1, oneReg, t1, mask);       // 1−n²
            AscendC::Reg::Mul(t1, aVal, t1, mask);         // dhn = A·(1−n²)
            AscendC::Reg::Sub(t2, hpReg, nReg, mask);      // hp−n
            AscendC::Reg::Mul(t2, t2, zReg, mask);         // (hp−n)·z
            AscendC::Reg::Mul(t2, aVal, t2, mask);         // dzp = A·(hp−n)·z
            AscendC::Reg::Mul(t3, t1, hnReg, mask);        // dhn·hn
            AscendC::Reg::Mul(t3, t3, rReg, mask);         // ·r
            AscendC::Reg::Sub(aVal, oneReg, rReg, mask);   // 1−r（A 已消费）
            AscendC::Reg::Mul(t3, t3, aVal, mask);         // drp = dhn·hn·r·(1−r)
            AscendC::Reg::Mul(aVal, rReg, t1, mask);       // dGh_n = r·dhn
            AscendC::Reg::Mul(t4, gradHReg, zReg, mask);   // dhFromH(t) = gradH·z
            AscendC::Reg::StoreAlign(dGiR + GATE_IDX_RESET * hA + bandOff + cOff, t3, mask);  // 门 r
            AscendC::Reg::StoreAlign(dGiR + GATE_IDX_UPDATE * hA + bandOff + cOff, t2, mask); // 门 z
            AscendC::Reg::StoreAlign(dGiR + GATE_IDX_NEW * hA + bandOff + cOff, t1, mask);    // 门 n
            AscendC::Reg::StoreAlign(dGhR + GATE_IDX_RESET * hA + bandOff + cOff, t3, mask);  // 门 r
            AscendC::Reg::StoreAlign(dGhR + GATE_IDX_UPDATE * hA + bandOff + cOff, t2, mask); // 门 z
            AscendC::Reg::StoreAlign(dGhR + GATE_IDX_NEW * hA + bandOff + cOff, aVal, mask);  // 门 n
            AscendC::Reg::StoreAlign(dhfhOutAddr + offFull + cOff, t4, mask);
            AscendC::Reg::StoreAlign(hpSaveAddr + offFull + cOff, hpReg, mask);
        }
    }
}

#endif // _GRU_GRAD_ARCH35_KERNEL_H_
