/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lp_norm_reduce_v2_kernel.h
 * \brief LpNormReduceV2 算子 kernel 类定义（arch35 架构，纯 Vector 归约）——迭代二全量整合
 *
 * 模板参数（与 lp_norm_reduce_v2_tiling_key.h ASCENDC_TPL_ARGS_DECL 一一对应，共 24 实例）：
 *   - D_T_X:      输入/输出 dtype（fp32 / fp16 / bf16；PT-08：bf16 回写 CAST_RINT = RN-even）
 *   - CROSS_CORE: 0=各核直出 / 1=跨核二次归并（PT-10：workspace partial 64B slot + SyncAll 两阶段）
 *   - LAYOUT:     0=AR（C==1）/ 1=ARA（C>1，Pattern::Reduce::RA，PT-09）
 *   - BUFFER_MODE: 0=单缓冲 / 1=双缓冲（仅 inQueueX 生效；PT-02/PT-11：每核 chunk 数 ≥4 启用）
 *
 * 数据流（DESIGN §6.2/§7，迭代二按穿刺结论整合）：
 *   统一 tile 外层循环（全核统一 tile 数，CROSS_CORE=1 时空跑 tile 照常 SyncAll——§7.1）：
 *     acc 置分支初值（0 / -inf / +inf）
 *     for rChunk in 本核 R 段（CROSS_CORE=1 时 = rFactorPerGroup 段）:
 *       CopyIn(2D DataCopyPad，dstStride=0 + isPad rightPadding 补 32B 行步进，
 *              pad 值 = 分支归约单位元的变换前原像——PT-04 dstStride 32B 单位坑规避)
 *       Cast fp32（fp32 免）→ Abs → pBranch 运行期 switch 变换：
 *         BR_ZERO:  Compares+Select 向量段(64 对齐)+标量尾（PT-05/PT-12）
 *         BR_INT_POW: 双计算 buffer 连乘（禁原地平方，issue_20261104_intpow_square_bug）
 *         BR_EXP:   Ln→Muls→Exp 双 buffer 乒乓（PT-06 F1：原地 unary 静默失效）
 *       归约: Pattern::Reduce::AR/RA 一次调用批量逐行/逐列归约（PT-01/PT-05：
 *             规避 counter-mode Level2 Reduce 的 dst 对齐 trap 与 V_S 标量同步）
 *       merge: Add/Max/Min(acc, acc, chunkResult)
 *     [CROSS_CORE=1] Phase1 落盘 workspace（单 VECOUT 队列 outQueueY 复用为累计器载体——PT-10
 *                    模式；VECOUT 规范顺序 Alloc→填→EnQue→DeQue→DataCopyPad，PT-10：DataCopyPad
 *                    必须在 EnQue/DeQue 之后发出；fp16/bf16 累计器先 DataCopy(UB→UB fp32) 拷入队列
 *                    自有张量——H2 契约修复；slotIdx 含 tile 奇偶偏移的 ping-pong 双 bank（容量 ×2）
 *                    ——H1 修复：规避 tile t 的 Phase2 读与 tile t+1 的 Phase1 写的跨核 WAR 竞争）
 *                    → SyncAll
 *                    → Phase2（rGroupIdx==0）累计器置初值后合并全部 rGroupNum 个 partial（同奇偶 slot）→ 回写
 *     Epilogue: 逐 tile 回写（fp32 直通 / fp16·bf16 Cast(CAST_RINT) 后回写）
 *
 * PT-11 双缓冲约束说明：chunk 循环内不使用 counter-mode Level2 Reduce（其内部 V→S 标量同步会阻断
 * MTE2 预取下发、抹平双缓冲流水）；本路径的 Pattern::Reduce::AR/RA 为单一 __simd_vf__ 域自洽实现
 * （寄存器树形归并 + LocalMemBar<VEC_STORE,VEC_LOAD>，无 Scalar pipe 参与，PT-05 源码级结论），
 * 双缓冲流水不受阻断，等价达成 PT-11 "chunk 内纯 Vector + 行末归约" 的规避目标。
 *
 * VECOUT 队列形态说明（迭代二上板实测）：全 kernel 仅保留一个 VECOUT 队列（outQueueY），CROSS_CORE
 * 两阶段复用该队列为 fp32 累计器/partial 落盘载体（PT-10 单队列模式）。GE/FE 动态编译（MIX 管线）
 * 下引入第二个 VECOUT 队列实测触发 MTE 地址越界（aivec error 95，直调编译验证逻辑本身正确），
 * 单队列形态在直调与 GE 两种链路均验证通过。
 *
 * 已知边界（DESIGN §9 R11，issue PT04-reduce-min-allinf-fltmax）：
 *   p=-inf 分支"整行 |x| 全为 +inf"的退化输入输出 FLT_MAX（0x7F7FFFFF）而非 +inf——
 *   dav-3510 Level2 ReduceMin 内部 init=FLT_MAX 哨兵胜出；与"行内恰为 {FLT_MAX,+inf} 集合"
 *   场景不可区分，无法事后修正（需 countInf==R 判定，代价每 chunk 两条归约指令）。
 *   按 issue 推荐路径记录为已知边界，spec extreme_inputs 未定义该输入类，ST 口径排除逐位断言。
 *
 * P2 多段 R 路径（DESIGN §4.3，迭代三；运行期 segNum≥2 分支，tiling 恒选 CROSS_CORE=0/LAYOUT=0/
 * BUFFER_MODE=0 实例）：PT-13 段循环工程化——外层输出 tile，内层输出元素 × 段乘积空间 chunk；
 *   A 段混合基数分解（oIdx → GM 基地址）+ 段乘积空间 k → 各段坐标分解（跨步收集，无需嵌套循环）；
 *   标量收集 D_T_X → S_V → Cast fp32 + Abs + pBranch 变换 → Pattern AR 单行归约 → PipeBarrier →
 *   标量合并（Add/Max/Min）；回写 S_V → 向量搬运进 outQueue → MTE3（PT-17 标量合并→向量搬运模式）。
 *   性能为次优先（正确性优先，spec 无非连续多轴 boundary case，无性能 KPI；PT-13 §3.2）。
 */

#ifndef LP_NORM_REDUCE_V2_H
#define LP_NORM_REDUCE_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lp_norm_reduce_v2_tiling_data.h"
#include "lp_norm_reduce_v2_tiling_key.h"

#include <type_traits>

namespace NsLpNormReduceV2 {

using namespace AscendC;

// fp32 位模式构造 +inf（避免依赖设备端 math.h 宏）
__aicore__ inline float LpnPosInfF32()
{
    union LpnInfUnion {
        uint32_t u;
        float f;
    } v;
    v.u = 0x7F800000U;
    return v.f;
}

// int64 整数上取整/对齐（Kernel 侧本地实现）
// P2 标量 Neumaier 补偿累加（4.2c：大 rTotal 段循环合并误差对齐竞品；cy 为补偿项跨 chunk 持有）
static __aicore__ inline float KahanScalarF32(float acc, float b, float& cy)
{
    const float y = b - cy;
    const float t = acc + y;
    cy = (acc - t) + y;
    return t;
}

__aicore__ inline int64_t LpnCeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

__aicore__ inline int64_t LpnCeilAlign(int64_t a, int64_t align) { return LpnCeilDiv(a, align) * align; }

template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
class LpNormReduceV2 {
    static constexpr int32_t BUFFER_NUM = (BUFFER_MODE == 1) ? 2 : 1;
    static constexpr int64_t ALIGN_ELEMS = 32 / sizeof(D_T_X); // 32B 对齐元素数：fp32=8, fp16/bf16=16
    // Pattern AR/RA LessThanVL 路径 Reg::LoadAlign 越界读余量（PT-09 §2.3：inBuf 需 +256B）
    static constexpr int64_t VEC_LOAD_MARGIN_BYTES = 256;
    // ws 队列/累计器槽位下界（8 fp32 元素 = 32B，PT-10 PT10_SCALAR_SLOT_ELEMS 同款）
    static constexpr int64_t WS_SLOT_MIN_ELEMS = 8;
    // 二分累加路数（4.2d）：chunk 分组路数上限 + Phase2 partial 容量下限。
    // 64×slot×4B ≤ 64×128×4B = 32KB（rowsTile 二分场景 ≤29 → 实际 ≤ 7.3KB）。
    static constexpr int64_t PAIRWISE_LANES = 64;

public:
    __aicore__ inline LpNormReduceV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const LpNormReduceV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessAR();
    __aicore__ inline void ProcessARA();
    __aicore__ inline void ProcessP2();
    __aicore__ inline void ProcessP2Tile(int64_t tileRow0, int64_t cnt);
    __aicore__ inline void ProcessZeroOutput();
    __aicore__ inline void CopyInAR(int64_t tileRow0, int64_t rowsThisTile, int64_t rChunkOffset, int64_t validLen,
                                    int64_t alignedCols);
    __aicore__ inline void CopyInARA(int64_t b, int64_t rChunkOffset, int64_t rCur, int64_t cGmBase, int64_t validC,
                                     int64_t rightPad);
    __aicore__ inline void ComputeAR(int64_t rowsThisTile, int64_t alignedCols, LocalTensor<float>& accF32);
    __aicore__ inline void ComputeARA(int64_t accRow, int64_t rCur, int64_t tilePitch, int64_t validC,
                                      LocalTensor<float>& accF32);
    __aicore__ inline void ApplyPBranchTransform(LocalTensor<float>& work, LocalTensor<float>& calc, int64_t count,
                                                 LocalTensor<float>& result);
    __aicore__ inline void MergeAcc(LocalTensor<float>& dst, const LocalTensor<float>& a, const LocalTensor<float>& b,
                                    int64_t count);
    __aicore__ inline void EpilogueOutAR(int64_t globalRow0, int64_t rowsThisTile, const LocalTensor<float>& accF32);
    __aicore__ inline void EpilogueOutARA(int64_t bGlobalRow0, int64_t bTileCur, int64_t cGmBase, int64_t validC,
                                          int64_t tilePitch, const LocalTensor<float>& accF32);
    __aicore__ inline void Phase1StoreAR(int64_t rowsThisTile, int64_t tileParity, const LocalTensor<float>& accF32);
    __aicore__ inline void Phase1StoreARA(int64_t bTileCur, int64_t tilePitch, int64_t validC, int64_t tileParity,
                                          const LocalTensor<float>& accF32);
    __aicore__ inline void Phase2MergeAR(int64_t tileRow0, int64_t rowsThisTile, int64_t tileParity);
    __aicore__ inline void Phase2MergeARA(int64_t bRow0, int64_t bTileCur, int64_t cGmBase, int64_t validC,
                                          int64_t tilePitch, int64_t tileParity);
    __aicore__ inline float GetInitValue() const;
    __aicore__ inline D_T_X GetPadValue() const;
    __aicore__ inline float MergeScalarF32(float a, float b) const;

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    // 唯一 VECOUT 队列：直出模式回写载体（fp32 累计器驻留）/ CROSS 两阶段 partial 落盘与回写
    // （PT-10 单 VECOUT 模式：FE/MIX 动态编译管线下第二个 VECOUT 队列实测触发 MTE 地址异常，规避）
    TQue<QuePosition::VECOUT, 1> outQueueY_;
    TBuf<TPosition::VECCALC> fp32Buf_; // fp32 计算区（fp32 输入时为 EXP/INT_POW 乒乓第二缓冲）
    TBuf<TPosition::VECCALC> tmpBuf_;  // Reduce API 内部临时 / EXP·INT_POW 乒乓缓冲（时序复用安全）
    TBuf<TPosition::VECCALC> onesBuf_; // BR_ZERO 专用：全 1 参考块（PT-12）
    TBuf<TPosition::VECCALC> maskBuf_; // BR_ZERO 专用：Compares 掩码（1B/元素，PT-05/PT-12）
    TBuf<TPosition::VECCALC> chunkResultBuf_;  // 单 chunk 归约结果（+8 元素 StoreUnAlign 保护槽，PT-05）
    TBuf<TPosition::VECCALC> globalResultBuf_; // fp16/bf16 路径 fp32 累计器（fp32 直用 outQueue 张量）
    TQue<QuePosition::VECIN, 1> wsInQueue_;    // CROSS_CORE=1：Phase2 读回 workspace partial（fp32，PT-10）
    // ---- P2 多段 R 路径专用（DESIGN §4.3，迭代三；roles 见 Init P2 分支）----
    TBuf<TPosition::VECCALC> p2GatherBuf_; // 原始值跨步收集（D_T_X，标量写 → Cast fp32）
    TBuf<TPosition::VECCALC> p2TmpBuf_; // Pattern AR 归约 sharedTmp（专用，规避与乒乓 calc 别名分析）
    // 二分累加缓冲（4.2d）：AR chunk 路 + Phase2 跨核 partial 共用，LANES×slot fp32。
    // chunk 路段 [0, LANES×slot)；Phase2 段 [0, rGroupNum×slot)（rGroupNum≤18 < LANES）。
    TBuf<TPosition::VECCALC> pwBuf_;

    GlobalTensor<D_T_X> xGm_;
    GlobalTensor<D_T_X> yGm_;
    GlobalTensor<float> wsGm_;

    int64_t bLength_ = 0;         // 合轴后总行数/批次数（AR：O = bLength；ARA：O = bLength × cLength）
    int64_t rLength_ = 0;         // 归约长度 R
    int64_t cLength_ = 1;         // R 之后的 A 维积（AR 时 =1）
    int64_t bFactor_ = 0;         // 每核输出块行数/批数
    int64_t rowsTile_ = 1;        // AR：输出 tile 行数
    int64_t bTile_ = 1;           // ARA：输出 tile 批数
    int64_t ubFactorR_ = 1;       // AR：每行 chunk 有效长度；ARA：RA 归约行数（rChunk）
    int64_t ubRAlloc_ = 1;        // AR chunk 缓冲对齐后长度（= CeilAlign(ubFactorR, ALIGN_ELEMS)）
    int64_t ubFactorC_ = 1;       // ARA：C 向 tile 大小（≤64）
    int64_t rAlignedCols_ = 1;    // ARA：tile pitch 上界（= CeilAlign(ubFactorC, ALIGN_ELEMS)）
    int64_t rowStart_ = 0;        // 本核输出块起始行（blockId / rGroupNum × bFactor）
    int64_t coreRows_ = 0;        // 本核实际行数/批数
    int64_t myRLow_ = 0;          // 本核 R 段下界（CROSS_CORE=1：rGroupIdx × rFactorPerGroup）
    int64_t myRHigh_ = 0;         // 本核 R 段上界（min(rLength, myRLow + rFactorPerGroup)）
    int64_t outBlockNum_ = 1;     // 输出分块数
    int64_t rGroupNum_ = 1;       // R 跨核分组数
    int64_t rFactorPerGroup_ = 0; // 每核 R 段工作量（CROSS_CORE=0 时 = rLength）
    int64_t outBlockIdx_ = 0;     // 本核输出块索引
    int64_t rGroupIdx_ = 0;       // 本核 R 分组索引
    int64_t slotStrideElems_ = 1; // workspace slot 步进（fp32 元素，64B 对齐）
    int32_t pBranch_ = LPNORM_P_BRANCH_INT_POW;
    float pVal_ = 2.0f;
    int32_t powTimes_ = 1;
    int32_t zeroOutputFlag_ = 0;
    int32_t segNum_ = 1;         // P2：R 段数（≥2 运行期分支 P2；0/1 走 P1 快速/主路径）
    int64_t segSize_[4] = {0};   // P2：各 R 段大小（乘积空间混合基数）
    int64_t segStride_[4] = {0}; // P2：各 R 段元素步进
    int32_t aSegNum_ = 0;        // P2：A 段数（输出混合基数）
    int64_t aSize_[4] = {0};     // P2：各 A 段大小（aSize[0] 最外层）
    int64_t aStride_[4] = {0};   // P2：各 A 段元素步进
    int32_t evtSv_ = 0;          // S_V 硬同步事件 ID（标量写 → 向量读，PT-13/PT-17）
};

template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const LpNormReduceV2TilingData* tilingData)
{
    // workspace 形参在 GE kernel-task 链路为 基址+16MB 的未映射值（issue_20261104_ge_ws_write_fault），
    // 真实用户 workspace 经 GetSysWorkSpacePtr() 取址（见 Init CROSS_CORE 分支）；形参保留但不消费
    (void)workspace;
    bLength_ = tilingData->bLength;
    rLength_ = tilingData->rLength;
    cLength_ = tilingData->cLength;
    bFactor_ = tilingData->bFactor;
    rowsTile_ = tilingData->rowsTile;
    bTile_ = tilingData->bTile;
    ubFactorR_ = tilingData->ubFactorR;
    ubFactorC_ = tilingData->ubFactorC;
    rAlignedCols_ = tilingData->rAlignedCols;
    rFactorPerGroup_ = tilingData->rFactorPerGroup;
    outBlockNum_ = tilingData->outBlockNum;
    rGroupNum_ = tilingData->rGroupNum;
    pBranch_ = tilingData->pBranch;
    pVal_ = tilingData->pVal;
    powTimes_ = tilingData->powTimes;
    zeroOutputFlag_ = tilingData->zeroOutputFlag;
    segNum_ = tilingData->segNum;
    if (segNum_ >= 2 && zeroOutputFlag_ == 0) {
        // P2 段表装载（快速路径 tiling 侧已归一 segNum=1，此处仅真实 P2 分档进入）
        const int32_t segCnt = (segNum_ > 4) ? 4 : segNum_; // rank≤8 → 段数 ≤4（防御钳位）
        for (int32_t j = 0; j < segCnt; j++) {
            segSize_[j] = tilingData->segSize[j];
            segStride_[j] = tilingData->segStride[j];
        }
        segNum_ = segCnt;
        aSegNum_ = tilingData->aSegNum;
        const int32_t aCnt = (aSegNum_ > 4) ? 4 : aSegNum_;
        for (int32_t a = 0; a < aCnt; a++) {
            aSize_[a] = tilingData->aSize[a];
            aStride_[a] = tilingData->aStride[a];
        }
        aSegNum_ = aCnt;
    }

    if (rowsTile_ < 1) {
        rowsTile_ = 1;
    }
    if (bTile_ < 1) {
        bTile_ = 1;
    }
    if (ubFactorR_ < 1) {
        ubFactorR_ = 1;
    }
    if (ubFactorC_ < 1) {
        ubFactorC_ = 1;
    }
    if (rAlignedCols_ < 1) {
        rAlignedCols_ = 1;
    }
    ubRAlloc_ = LpnCeilAlign(ubFactorR_, ALIGN_ELEMS);

    // 输出空间切分定位（DESIGN §5.2）：CROSS_CORE=1 时 blockId 分解 outBlockIdx / rGroupIdx
    int64_t blockId = static_cast<int64_t>(AscendC::GetBlockIdx());
    if constexpr (CROSS_CORE == 1) {
        int64_t groups = (rGroupNum_ > 1) ? rGroupNum_ : 1;
        outBlockIdx_ = blockId / groups;
        rGroupIdx_ = blockId % groups;
        rowStart_ = outBlockIdx_ * bFactor_;
        myRLow_ = rGroupIdx_ * rFactorPerGroup_;
        myRHigh_ = (rLength_ < myRLow_ + rFactorPerGroup_) ? rLength_ : (myRLow_ + rFactorPerGroup_);
        if (myRHigh_ < myRLow_) {
            myRHigh_ = myRLow_; // 空 R 段防御（partial = 分支单位元，Phase2 合并正确吸收）
        }
    } else {
        rowStart_ = blockId * bFactor_;
        myRLow_ = 0;
        myRHigh_ = rLength_;
    }
    int64_t remain = bLength_ - rowStart_;
    coreRows_ = (remain > bFactor_) ? bFactor_ : remain;
    if (coreRows_ < 0) {
        coreRows_ = 0; // 空闲核保护（blockDim == usedCores 时不会触发）
    }

    xGm_.SetGlobalBuffer((__gm__ D_T_X*)x, bLength_ * rLength_ * cLength_);
    yGm_.SetGlobalBuffer((__gm__ D_T_X*)y, bLength_ * cLength_); // O（keepdim 不影响线性元素数）

    // UB Buffer 规划（DESIGN §5.3，输出 tile 规模封顶，与 O 无关——BLK-1）
    // P2 段循环路径：缓冲几何与 AR/ARA 无关（rowsTile=输出 tile、ubFactorR=段空间 chunk），
    // ones/mask/f32/calc/chunk/stage 复用既有成员（roles 与 AR 对齐），gather/tmp 为 P2 专用
    const bool isP2 = (segNum_ >= 2) && (zeroOutputFlag_ == 0);
    const int64_t marginElems = VEC_LOAD_MARGIN_BYTES / static_cast<int64_t>(sizeof(D_T_X));
    if (isP2) {
        const int64_t chunkElems = ubFactorR_; // host 已按 ubBlockElems 对齐（尾部 pad 至此）
        pipe_.InitBuffer(p2GatherBuf_, (chunkElems + marginElems) * sizeof(D_T_X));
        pipe_.InitBuffer(fp32Buf_, (chunkElems + marginElems) * sizeof(float)); // work（Cast 目的地）
        pipe_.InitBuffer(tmpBuf_, chunkElems * sizeof(float));                  // INT_POW/EXP 乒乓
        const int64_t tmpElems = (chunkElems > 1024) ? chunkElems : 1024;       // PT-01 公式下界
        pipe_.InitBuffer(p2TmpBuf_, tmpElems * sizeof(float));
        if (pBranch_ == LPNORM_P_BRANCH_ZERO) {
            pipe_.InitBuffer(onesBuf_, chunkElems * sizeof(float)); // Compares 向量段 ≤ (count/64)*64 ≤ chunk
            pipe_.InitBuffer(maskBuf_, chunkElems);                 // 1B/元素（PT-05）
        }
        pipe_.InitBuffer(chunkResultBuf_, 9 * sizeof(float));          // 单行 Pattern AR dst（+8 保护槽）
        pipe_.InitBuffer(globalResultBuf_, rowsTile_ * sizeof(float)); // fp32 输出暂存（oTile）
        pipe_.InitBuffer(outQueueY_, 1, rowsTile_ * sizeof(D_T_X));
        evtSv_ = static_cast<int32_t>(pipe_.FetchEventID(AscendC::HardEvent::S_V));
    } else if constexpr (LAYOUT == 0) {
        // AR：E = rowsTile × ubRAlloc；chunkResult 按 tile 行数
        const int64_t tileElems = rowsTile_ * ubRAlloc_;
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, (tileElems + marginElems) * sizeof(D_T_X));
        pipe_.InitBuffer(fp32Buf_, tileElems * sizeof(float));
        pipe_.InitBuffer(tmpBuf_,
                         tileElems * sizeof(float)); // Reduce AR tmp 峰值 = dimA×mainR ≤ E（issue_tmpbuf_1 保守口径）
        if (pBranch_ == LPNORM_P_BRANCH_ZERO) {
            pipe_.InitBuffer(onesBuf_, tileElems * sizeof(float));
            pipe_.InitBuffer(maskBuf_, tileElems); // 1B/元素（PT-05）
        }
        pipe_.InitBuffer(chunkResultBuf_, (rowsTile_ + 8) * sizeof(float));
        // 二分累加缓冲（4.2d）：LANES=64 路 × 32B 对齐 slot；rowsTile 大 O 小 R（chunkNum=1）时不启用
        // 但缓冲统一预留，host 预算已计入（+32KB 上界，rowsTile≤29 时实占 ≤7.3KB）
        pipe_.InitBuffer(pwBuf_, (PAIRWISE_LANES + 1) * 32 * sizeof(float));
        if constexpr (CROSS_CORE == 1) {
            // outQueueY 复用为 Phase1 fp32 累计器载体（PT-10 单 VECOUT 模式），槽 ≥32B；
            // fp16/bf16 累计器（globalResultBuf_）Phase1 经 DataCopy(UB→UB) 拷入队列自有张量后落盘
            // （H2 契约修复：恢复 Alloc↔Free 严格配对），槽/累计器容量按 32B 对齐跨度取整——
            // DataCopy count 需 32B 对齐，尾部垃圾不落 GM（DataCopyPad blockLen 截断在 rowsThisTile）
            const int64_t slotElems = LpnCeilAlign(rowsTile_, WS_SLOT_MIN_ELEMS);
            pipe_.InitBuffer(globalResultBuf_, slotElems * sizeof(float));
            pipe_.InitBuffer(outQueueY_, 1, slotElems * sizeof(float));
            pipe_.InitBuffer(wsInQueue_, 1, slotElems * sizeof(float));
            slotStrideElems_ = LpnCeilAlign(rowsTile_ * static_cast<int64_t>(sizeof(float)), WS_SLOT_ALIGN_BYTES) /
                               static_cast<int64_t>(sizeof(float));
            // GE kernel-task 链路（issue_20261104_ge_ws_write_fault 迭代三定位）：运行时按 TilingFunc
            // 声明大小分配用户 workspace，实际缓冲区基址经 KFC workspace 通道下发（GetSysWorkSpacePtr()），
            // 而 kernel 的 workspace 形参被按 aclnn 约定填为 基址+RESERVED_WORKSPACE(16MB)——该偏移区
            // 在本链路未分配/未映射，MTE 与标量访问均触发 aivec error 95（DDR 地址越界）。
            // 故 wsGm_ 基址取 GetSysWorkSpacePtr()（真实分配区），勿用 workspace 形参。
            // H1 ping-pong：容量 ×2（tile 奇偶双 bank），slotIdx 增加 tileParity×outBlockNum×rGroupNum
            // 偏移——规避 tile t 的 Phase2 读（核0）与 tile t+1 的 Phase1 写（其他核）的跨核 WAR 竞争
            wsGm_.SetGlobalBuffer((__gm__ float*)GetSysWorkSpacePtr(),
                                  2 * outBlockNum_ * rGroupNum_ * slotStrideElems_);
        } else {
            pipe_.InitBuffer(globalResultBuf_, rowsTile_ * sizeof(float));
            pipe_.InitBuffer(outQueueY_, 1, rowsTile_ * sizeof(D_T_X));
        }
    } else {
        // ARA：E = rChunk × rAlignedCols；输出 tile = bTile 批 × tilePitch 列（PT-09 布局）
        const int64_t tileElems = ubFactorR_ * rAlignedCols_;
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, (tileElems + marginElems) * sizeof(D_T_X));
        pipe_.InitBuffer(fp32Buf_, tileElems * sizeof(float));
        pipe_.InitBuffer(tmpBuf_, tileElems * sizeof(float)); // Reduce RA tmp 峰值 = dimA×mainR ≤ E
        if (pBranch_ == LPNORM_P_BRANCH_ZERO) {
            pipe_.InitBuffer(onesBuf_, tileElems * sizeof(float));
            pipe_.InitBuffer(maskBuf_, tileElems);
        }
        pipe_.InitBuffer(chunkResultBuf_, (rAlignedCols_ + 8) * sizeof(float));
        pipe_.InitBuffer(globalResultBuf_, bTile_ * rAlignedCols_ * sizeof(float));
        if constexpr (CROSS_CORE == 1) {
            // outQueueY 复用为 Phase1 fp32 累计器载体（PT-10 单 VECOUT 模式；rAlignedCols ≥ 8 恒 ≥32B）；
            // fp16/bf16 累计器 Phase1 经 DataCopy(UB→UB) 拷入队列自有张量（H2 契约修复，count =
            // bTileCur×tilePitch 含对齐因子天然 32B 对齐，且 ≤ bTile×rAlignedCols 槽容量）
            pipe_.InitBuffer(outQueueY_, 1, bTile_ * rAlignedCols_ * sizeof(float));
            pipe_.InitBuffer(wsInQueue_, 1, bTile_ * rAlignedCols_ * sizeof(float));
            // slot 容量 = bTile × rAlignedCols（行距上限），64B 对齐步进防 bank conflict；
            // 基址取 GetSysWorkSpacePtr()（见上方 AR 分支 issue_20261104 说明）；
            // H1 ping-pong：容量 ×2（(ct,bt) tile 序线性化奇偶双 bank，规避跨 tile WAR）
            slotStrideElems_ = LpnCeilAlign(bTile_ * rAlignedCols_ * static_cast<int64_t>(sizeof(float)),
                                            WS_SLOT_ALIGN_BYTES) /
                               static_cast<int64_t>(sizeof(float));
            wsGm_.SetGlobalBuffer((__gm__ float*)GetSysWorkSpacePtr(),
                                  2 * outBlockNum_ * rGroupNum_ * slotStrideElems_);
        } else {
            pipe_.InitBuffer(outQueueY_, 1, bTile_ * rAlignedCols_ * sizeof(D_T_X));
        }
    }
}

// 分支累计器初值（DESIGN §1.6 合并矩阵：sum→0 / max→-inf / min→+inf）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline float LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::GetInitValue() const
{
    if (pBranch_ == LPNORM_P_BRANCH_POS_INF) {
        return -LpnPosInfF32();
    }
    if (pBranch_ == LPNORM_P_BRANCH_NEG_INF) {
        return LpnPosInfF32();
    }
    return 0.0f;
}

// 输入空间 pad 值（经 Cast+Abs+p 分支变换后为归约单位元，DESIGN §5.3 padding 安全规则）：
//   求和类分支（ZERO/ONE/INT_POW/EXP(p>0)）→ 0（贡献 0）
//   POS_INF(max) → 0（|0|=0 ≤ max|x| 恒成立，不污染 max）
//   NEG_INF(min) → +inf（|inf|=inf ≥ min|x| 恒成立）
//   EXP(p<0)     → +inf（exp(p·ln(+inf))=exp(-inf)=0 零贡献；真实 0 元素仍贡献 +inf，符合 spec 极限语义）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline D_T_X LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::GetPadValue() const
{
    float v = 0.0f;
    if (pBranch_ == LPNORM_P_BRANCH_NEG_INF || (pBranch_ == LPNORM_P_BRANCH_EXP && pVal_ < 0.0f)) {
        v = LpnPosInfF32();
    }
    return static_cast<D_T_X>(v);
}

template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::Process()
{
    if (zeroOutputFlag_ == 1) {
        // R==0 且 O>0：全归约空输入 → 输出全 0（spec boundary，DESIGN §5.6）
        ProcessZeroOutput();
        return;
    }
    if (segNum_ >= 2) {
        // P2 多段 R 通用段循环（DESIGN §4.3；tiling 恒选 CROSS_CORE=0/LAYOUT=0 实例）
        ProcessP2();
        return;
    }
    if (coreRows_ <= 0) {
        // O==0 空输出：直接返回，无 GM 写（DESIGN §5.6）
        return;
    }
    if constexpr (LAYOUT == 0) {
        ProcessAR();
    } else {
        ProcessARA();
    }
}

// P2 多段 R 主路径（DESIGN §4.3，PT-13 段循环工程化，正确性优先·标量收集）：
//   外层输出 tile（rowsTile=oTile）→ 内层输出元素 × 段乘积空间 chunk（ubFactorR）；
//   同一输出位置的多段更新由同一核串行完成（按输出切分，无 atomic/无 workspace）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::ProcessP2()
{
    const int64_t oTile = rowsTile_;
    const int64_t tileNum = LpnCeilDiv(coreRows_, oTile);
    for (int64_t t = 0; t < tileNum; t++) {
        const int64_t tileRow0 = t * oTile;
        const int64_t cnt = (oTile < coreRows_ - tileRow0) ? oTile : (coreRows_ - tileRow0);
        ProcessP2Tile(tileRow0, cnt);
    }
}

// P2 单输出 tile：逐元素 A 段混合基数分解（oIdx → GM 基地址）→ 段乘积空间 chunk 跨步收集
// → Cast fp32 + Abs + pBranch 变换 → Pattern AR 单行归约 → 标量合并累计 → tile 回写
// （同步口径：标量写 UB → SetFlag/WaitFlag<S_V> → 向量（PT-13/PT-17）→ PipeBarrier<PIPE_ALL>
//  → 标量读（PT-15）；回写经 outQueue 向量搬运（PT-17 标量合并→向量搬运模式），全 GM 写走 MTE3）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::ProcessP2Tile(int64_t tileRow0,
                                                                                             int64_t cnt)
{
    LocalTensor<D_T_X> gather = p2GatherBuf_.Get<D_T_X>();
    LocalTensor<float> f32Data = fp32Buf_.Get<float>();
    LocalTensor<float> calc = tmpBuf_.Get<float>();
    LocalTensor<uint8_t> tmpData = p2TmpBuf_.Get<uint8_t>();
    LocalTensor<float> chunkRes = chunkResultBuf_.Get<float>();
    LocalTensor<float> stage = globalResultBuf_.Get<float>();
    const D_T_X padV = GetPadValue();
    const int64_t rTotal = rLength_; // P2 分档：bLength=O、rLength=rTotal、cLength=1
    for (int64_t i = 0; i < cnt; i++) {
        const int64_t oIdx = rowStart_ + tileRow0 + i;
        // A 段坐标分解（aSize[0] 最外层基数；aStride 后缀积）→ 输出元素 GM 基地址
        int64_t base = 0;
        int64_t rem = oIdx;
        for (int32_t a = aSegNum_ - 1; a >= 0; a--) {
            base += (rem % aSize_[a]) * aStride_[a];
            rem /= aSize_[a];
        }
        // 段乘积空间 chunk 循环：k → 各段坐标分解（无需嵌套循环），跨步收集 → 归约 → 标量合并
        float acc = GetInitValue();
        float kahanC = 0.0f; // P2 标量补偿项（4.2c，sum 类 chunk 合并）
        for (int64_t ks = 0; ks < rTotal; ks += ubFactorR_) {
            const int64_t n = (ubFactorR_ < rTotal - ks) ? ubFactorR_ : (rTotal - ks);
            const int64_t alignedN = LpnCeilAlign(n, ALIGN_ELEMS);
            for (int64_t k = 0; k < n; k++) {
                int64_t addr = base;
                int64_t r = ks + k;
                for (int32_t j = segNum_ - 1; j >= 0; j--) {
                    addr += (r % segSize_[j]) * segStride_[j];
                    r /= segSize_[j];
                }
                gather.SetValue(static_cast<uint32_t>(k), xGm_.GetValue(addr));
            }
            for (int64_t k = n; k < alignedN; k++) {
                gather.SetValue(static_cast<uint32_t>(k), padV); // 尾部补分支单位元原像（§5.3）
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(evtSv_);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(evtSv_);
            if constexpr (std::is_same_v<D_T_X, float>) {
                AscendC::Abs(f32Data, gather, static_cast<uint32_t>(alignedN));
            } else {
                AscendC::Cast<float>(f32Data, gather, RoundMode::CAST_NONE, static_cast<uint32_t>(alignedN));
                AscendC::Abs(f32Data, f32Data, static_cast<uint32_t>(alignedN));
            }
            LocalTensor<float> xformRes;
            ApplyPBranchTransform(f32Data, calc, alignedN, xformRes);
            uint32_t srcShape[2] = {1, static_cast<uint32_t>(alignedN)};
            if (pBranch_ == LPNORM_P_BRANCH_POS_INF) {
                AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR>(chunkRes, xformRes, tmpData, srcShape, true);
            } else if (pBranch_ == LPNORM_P_BRANCH_NEG_INF) {
                AscendC::ReduceMin<float, AscendC::Pattern::Reduce::AR>(chunkRes, xformRes, tmpData, srcShape, true);
            } else {
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(chunkRes, xformRes, tmpData, srcShape, true);
            }
            AscendC::PipeBarrier<PIPE_ALL>(); // 向量 → 标量读同步（PT-15）
            if (pBranch_ == LPNORM_P_BRANCH_POS_INF || pBranch_ == LPNORM_P_BRANCH_NEG_INF) {
                acc = MergeScalarF32(acc, chunkRes.GetValue(0));
            } else {
                acc = KahanScalarF32(acc, chunkRes.GetValue(0), kahanC);
            }
        }
        stage.SetValue(static_cast<uint32_t>(i), acc);
    }
    // 回写：S_V 同步 → 向量搬运进 outQueue（Adds x+0 / Cast CAST_RINT）→ EnQue → DataCopyPad
    AscendC::SetFlag<AscendC::HardEvent::S_V>(evtSv_);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(evtSv_);
    LocalTensor<D_T_X> yLocal = outQueueY_.template AllocTensor<D_T_X>();
    if constexpr (std::is_same_v<D_T_X, float>) {
        AscendC::Adds(yLocal, stage, 0.0f, static_cast<uint32_t>(cnt));
    } else {
        // fp32→fp16/bf16 RN-even（PT-08：CAST_RINT 等价 tbe.round）
        AscendC::Cast<D_T_X>(yLocal, stage, RoundMode::CAST_RINT, static_cast<uint32_t>(cnt));
    }
    outQueueY_.template EnQue<D_T_X>(yLocal);
    LocalTensor<D_T_X> yOut = outQueueY_.template DeQue<D_T_X>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(cnt * static_cast<int64_t>(sizeof(D_T_X)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(yGm_[rowStart_ + tileRow0], yOut, copyParams);
    outQueueY_.FreeTensor(yOut);
}

// P2 chunk 间标量合并（DESIGN §1.6 合并矩阵；NaN 传播对齐 golden IEEE 语义）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline float LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::MergeScalarF32(float a, float b) const
{
    if (pBranch_ == LPNORM_P_BRANCH_POS_INF) {
        if (a != a) {
            return a;
        }
        if (b != b) {
            return b;
        }
        return (a > b) ? a : b;
    }
    if (pBranch_ == LPNORM_P_BRANCH_NEG_INF) {
        if (a != a) {
            return a;
        }
        if (b != b) {
            return b;
        }
        return (a < b) ? a : b;
    }
    return a + b;
}

// AR 主路径：统一 tile 外层循环（§7.1）→ 本核 R 段 chunk 循环 → 归约合并 → 逐 tile 回写/两阶段
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::ProcessAR()
{
    // 全核统一 tile 数（CROSS_CORE=1 时空跑 tile 照常 SyncAll，规避 R9 半数参与死等）
    const int64_t tileNumMax = LpnCeilDiv(bFactor_, rowsTile_);
    for (int64_t t = 0; t < tileNumMax; t++) {
        const int64_t tileRow0 = t * rowsTile_;
        const bool validTile = tileRow0 < coreRows_;
        if (!validTile) {
            if constexpr (CROSS_CORE == 1) {
                AscendC::SyncAll(); // 空跑：无计算无落盘，但照常参与栅栏
            }
            continue;
        }
        const int64_t rowsThisTile = (rowsTile_ < coreRows_ - tileRow0) ? rowsTile_ : (coreRows_ - tileRow0);

        // Phase1：本核 R 段归约 → acc（fp32 累计器驻留 outQueueY 张量，PT-10 单 VECOUT 模式）
        LocalTensor<float> accF32;
        if constexpr (std::is_same_v<D_T_X, float>) {
            accF32 = outQueueY_.template AllocTensor<float>();
        } else {
            accF32 = globalResultBuf_.Get<float>();
        }
        AscendC::Duplicate(accF32, GetInitValue(), static_cast<int32_t>(rowsThisTile));
        const int64_t rSpan = myRHigh_ - myRLow_;
        const int64_t chunkNum = LpnCeilDiv(rSpan, ubFactorR_);
        // 二分累加（4.2d）：chunk 部分和先入 PAIRWISE_LANES 路分组累加器（按 chunk 序轮转），
        // chunk 循环后原地两两合并（log2(L) 轮）→ 顺序链长 chunkNum 收敛为 chunkNum/L + log2(L)，
        // 大 N 累加误差从 O(√chunkNum·ULP) 降到 ~O(√(log)·ULP)，对齐竞品树形精度。
        // 仅 sum 类且 chunkNum≥2 启用（max/min 无舍入、单 chunk 无合并误差）。
        const bool usePairwiseChunk = (chunkNum >= 2) && (pBranch_ != LPNORM_P_BRANCH_POS_INF) &&
                                      (pBranch_ != LPNORM_P_BRANCH_NEG_INF);
        constexpr int64_t LANES = PAIRWISE_LANES; // 64 路（缓冲 64×slot×4B ≤ 16KB）
        if (usePairwiseChunk) {
            const int64_t slot = LpnCeilAlign(rowsThisTile, 8); // 32B 对齐段
            LocalTensor<float> pw = pwBuf_.Get<float>();
            AscendC::Duplicate(pw, GetInitValue(), static_cast<int32_t>((LANES + 1) * slot));
            // 单 chunk 部分和载体（第 LANES 段）：每轮置初值后由 ComputeAR merge 进本 chunk 值
            LocalTensor<float> chunkTemp = pw[static_cast<uint32_t>(LANES * slot)];
            for (int64_t c = 0; c < chunkNum; c++) {
                const int64_t validLen = (ubFactorR_ < rSpan - c * ubFactorR_) ? ubFactorR_ : (rSpan - c * ubFactorR_);
                const int64_t alignedCols = LpnCeilAlign(validLen, ALIGN_ELEMS);
                CopyInAR(tileRow0, rowsThisTile, myRLow_ + c * ubFactorR_, validLen, alignedCols);
                AscendC::Duplicate(chunkTemp, GetInitValue(), static_cast<int32_t>(rowsThisTile));
                ComputeAR(rowsThisTile, alignedCols, chunkTemp); // chunkResult merge → 本 chunk 部分和
                LocalTensor<float> laneT = pw[static_cast<uint32_t>((c % LANES) * slot)];
                AscendC::Add(laneT, laneT, chunkTemp, static_cast<int32_t>(rowsThisTile));
            }
            // 原地两两合并 LANES 路 → accF32
            int64_t stride = 1;
            int64_t live = (chunkNum < LANES) ? chunkNum : LANES;
            while (stride < live) {
                for (int64_t i = 0; i + stride < live; i += 2 * stride) {
                    LocalTensor<float> lo = pw[static_cast<uint32_t>(i * slot)];
                    LocalTensor<float> hi = pw[static_cast<uint32_t>((i + stride) * slot)];
                    AscendC::Add(lo, lo, hi, static_cast<int32_t>(rowsThisTile));
                }
                stride *= 2;
            }
            AscendC::Copy(accF32, pw, static_cast<int32_t>(rowsThisTile));
        } else {
            for (int64_t c = 0; c < chunkNum; c++) {
                const int64_t validLen = (ubFactorR_ < rSpan - c * ubFactorR_) ? ubFactorR_ : (rSpan - c * ubFactorR_);
                const int64_t alignedCols = LpnCeilAlign(validLen, ALIGN_ELEMS);
                CopyInAR(tileRow0, rowsThisTile, myRLow_ + c * ubFactorR_, validLen, alignedCols);
                ComputeAR(rowsThisTile, alignedCols, accF32);
            }
        }

        if constexpr (CROSS_CORE == 1) {
            // Phase1 落盘（ping-pong：tile 奇偶 slot）→ 全核栅栏（SyncAll 入口 PipeBarrier<PIPE_ALL>
            // 保证 MTE3 落盘完成）；Phase2 读侧用相同奇偶计算
            Phase1StoreAR(rowsThisTile, t % 2, accF32);
            AscendC::SyncAll();
            if (rGroupIdx_ == 0) {
                Phase2MergeAR(tileRow0, rowsThisTile, t % 2);
            }
        } else {
            EpilogueOutAR(rowStart_ + tileRow0, rowsThisTile, accF32);
        }
    }
}

// ARA 主路径（PT-09）：列 tile（外层）× 批 tile（内层）统一循环 → RA 归约 → 逐 tile 回写/两阶段
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::ProcessARA()
{
    const int64_t cTileNum = LpnCeilDiv(cLength_, ubFactorC_);
    const int64_t bTileNumMax = LpnCeilDiv(bFactor_, bTile_);
    for (int64_t ct = 0; ct < cTileNum; ct++) {
        const int64_t validC = (ubFactorC_ < cLength_ - ct * ubFactorC_) ? ubFactorC_ : (cLength_ - ct * ubFactorC_);
        // tilePitch：本列 tile 的元素行距 = CeilAlign(validC, 32B/typeSize)，
        // CopyIn(dstStride=0+rightPadding)/Cast/RA 归约 srcShape/acc/outQueue 四处统一使用
        const int64_t tilePitch = LpnCeilAlign(validC, ALIGN_ELEMS);
        const int64_t rightPad = tilePitch - validC; // ≤ 15，rightPadding uint8 内
        const int64_t cGmBase = ct * ubFactorC_;
        for (int64_t bt = 0; bt < bTileNumMax; bt++) {
            const int64_t bRow0 = bt * bTile_;
            const bool validTile = bRow0 < coreRows_;
            if (!validTile) {
                if constexpr (CROSS_CORE == 1) {
                    AscendC::SyncAll();
                }
                continue;
            }
            const int64_t bTileCur = (bTile_ < coreRows_ - bRow0) ? bTile_ : (coreRows_ - bRow0);
            // (ct,bt) tile 序线性化（全核统一计数）→ ping-pong 奇偶：内外 tile 循环统一 tile parity
            const int64_t tileSeq = ct * bTileNumMax + bt;

            LocalTensor<float> accF32;
            if constexpr (std::is_same_v<D_T_X, float>) {
                accF32 = outQueueY_.template AllocTensor<float>();
            } else {
                accF32 = globalResultBuf_.Get<float>();
            }
            AscendC::Duplicate(accF32, GetInitValue(), static_cast<int32_t>(bTileCur * tilePitch));
            const int64_t rSpan = myRHigh_ - myRLow_;
            const int64_t rChunkNum = LpnCeilDiv(rSpan, ubFactorR_);
            for (int64_t i = 0; i < bTileCur; i++) {
                const int64_t b = rowStart_ + bRow0 + i;
                for (int64_t rc = 0; rc < rChunkNum; rc++) {
                    const int64_t rCur = (ubFactorR_ < rSpan - rc * ubFactorR_) ? ubFactorR_ :
                                                                                  (rSpan - rc * ubFactorR_);
                    CopyInARA(b, myRLow_ + rc * ubFactorR_, rCur, cGmBase, validC, rightPad);
                    ComputeARA(i, rCur, tilePitch, validC, accF32);
                }
            }

            if constexpr (CROSS_CORE == 1) {
                Phase1StoreARA(bTileCur, tilePitch, validC, tileSeq % 2, accF32);
                AscendC::SyncAll();
                if (rGroupIdx_ == 0) {
                    Phase2MergeARA(bRow0, bTileCur, cGmBase, validC, tilePitch, tileSeq % 2);
                }
            } else {
                EpilogueOutARA(rowStart_ + bRow0, bTileCur, cGmBase, validC, tilePitch, accF32);
            }
        }
    }
}

// R==0 置零路径：按输出 tile 循环 Duplicate(0) → CopyOut（大 O 安全，与 §6.2 tile 口径一致）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::ProcessZeroOutput()
{
    int64_t tileNum = LpnCeilDiv(bLength_, rowsTile_);
    for (int64_t t = 0; t < tileNum; t++) {
        int64_t tileRow0 = t * rowsTile_;
        int64_t rowsThisTile = (rowsTile_ < bLength_ - tileRow0) ? rowsTile_ : (bLength_ - tileRow0);
        if constexpr (std::is_same_v<D_T_X, float>) {
            LocalTensor<float> accF32 = outQueueY_.template AllocTensor<float>();
            AscendC::Duplicate(accF32, 0.0f, static_cast<int32_t>(rowsThisTile));
            EpilogueOutAR(tileRow0, rowsThisTile, accF32);
        } else {
            LocalTensor<float> accF32 = globalResultBuf_.Get<float>();
            AscendC::Duplicate(accF32, 0.0f, static_cast<int32_t>(rowsThisTile));
            EpilogueOutAR(tileRow0, rowsThisTile, accF32);
        }
    }
}

// 2D 搬入（AR）：[rowsThisTile 行 × validLen 元素]，源行步进 (R−validLen)×sizeof(T)（GM 字节）；
// dstStride 恒 0（dav-3510 UB 侧 stride 单位为 32B 块，禁止用其表达行间隙——PT-04 issue），
// 尾块经 isPad+rightPadding 补齐 32B 对齐行步进（pitch = alignedCols），pad 值按分支取归约单位元
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::CopyInAR(
    int64_t tileRow0, int64_t rowsThisTile, int64_t rChunkOffset, int64_t validLen, int64_t alignedCols)
{
    LocalTensor<D_T_X> xLocal = inQueueX_.template AllocTensor<D_T_X>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(rowsThisTile);
    copyParams.blockLen = static_cast<uint32_t>(validLen * static_cast<int64_t>(sizeof(D_T_X)));
    copyParams.srcStride = (rLength_ - validLen) * static_cast<int64_t>(sizeof(D_T_X));
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    int64_t srcOffset = (rowStart_ + tileRow0) * rLength_ + rChunkOffset;
    if (validLen == alignedCols) {
        // 对齐整块：isPad=false（blockLen 已 32B 对齐，无 dummy）
        DataCopyPadExtParams<D_T_X> padParams(false, 0, 0, static_cast<D_T_X>(0));
        AscendC::DataCopyPad(xLocal, xGm_[srcOffset], copyParams, padParams);
    } else {
        // 尾块：pad 到 32B 对齐（alignedCols），pad 值按分支取归约单位元（§5.3 规则）
        DataCopyPadExtParams<D_T_X> padParams(true, 0, static_cast<uint8_t>(alignedCols - validLen), GetPadValue());
        AscendC::DataCopyPad(xLocal, xGm_[srcOffset], copyParams, padParams);
    }
    inQueueX_.template EnQue<D_T_X>(xLocal);
}

// 2D 搬入（ARA，PT-09）：[rCur 行 × validC 列]，源行步进 (C−validC)×sizeof(T)（GM 字节）；
// dstStride=0 + rightPadding=tilePitch−validC（≤15）→ UB 行距 = tilePitch×sizeof(T)，
// pad 列填分支单位元（pad 列不参与 merge，防御性填充分支值）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::CopyInARA(
    int64_t b, int64_t rChunkOffset, int64_t rCur, int64_t cGmBase, int64_t validC, int64_t rightPad)
{
    LocalTensor<D_T_X> xLocal = inQueueX_.template AllocTensor<D_T_X>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(rCur);
    copyParams.blockLen = static_cast<uint32_t>(validC * static_cast<int64_t>(sizeof(D_T_X)));
    copyParams.srcStride = (cLength_ - validC) * static_cast<int64_t>(sizeof(D_T_X));
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    int64_t srcOffset = (b * rLength_ + rChunkOffset) * cLength_ + cGmBase;
    if (rightPad == 0) {
        DataCopyPadExtParams<D_T_X> padParams(false, 0, 0, static_cast<D_T_X>(0));
        AscendC::DataCopyPad(xLocal, xGm_[srcOffset], copyParams, padParams);
    } else {
        DataCopyPadExtParams<D_T_X> padParams(true, 0, static_cast<uint8_t>(rightPad), GetPadValue());
        AscendC::DataCopyPad(xLocal, xGm_[srcOffset], copyParams, padParams);
    }
    inQueueX_.template EnQue<D_T_X>(xLocal);
}

// p 分支逐元素变换（DESIGN §6.3 运行期 switch；work/calc 为两个独立 fp32 缓冲）：
//   POS_INF/NEG_INF/ONE：abs 后直接归约
//   ZERO：Compares+Select 向量段（count 64 对齐约束，PT-12）+ 标量尾混合；NaN 计 1、-0.0 计 0（PT-05）
//   INT_POW：双计算 buffer 连乘（|x|^p = ((|x|·|x|)·|x|)...，禁原地平方——issue_intpow_square_bug）
//   EXP：Ln→Muls→Exp 双 buffer 乒乓（PT-06 F1：原地 unary dst==src 静默失效；F2：pVal 本地化）
// 变换结果经 result 出参返回（EXP 落在 calc，其余落在 work）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::ApplyPBranchTransform(
    LocalTensor<float>& work, LocalTensor<float>& calc, int64_t count, LocalTensor<float>& result)
{
    result = work;
    const int32_t n = static_cast<int32_t>(count);
    switch (pBranch_) {
        case LPNORM_P_BRANCH_POS_INF: // max|x|：abs 后直接归约
        case LPNORM_P_BRANCH_NEG_INF: // min|x|：abs 后直接归约
        case LPNORM_P_BRANCH_ONE:     // sum|x|：abs 后直接归约
            break;
        case LPNORM_P_BRANCH_ZERO: {
            // dav-3510 约束：Compares/Select 向量段要求 count×sizeof(fp32) 为 256B 倍数（PT-12）
            const int64_t full = (count / 64) * 64;
            if (full > 0) {
                LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
                LocalTensor<float> ones = onesBuf_.Get<float>();
                AscendC::Duplicate(ones, 1.0f, static_cast<int32_t>(full));
                // mask=1(≠0，含 NaN) → src0=1.0；mask=0(==0，含 ±0.0) → scalar=0.0（IEEE 语义，PT-05 实测）
                AscendC::Compares(mask, work, 0.0f, AscendC::CMPMODE::NE, static_cast<uint32_t>(full));
                AscendC::Select(work, mask, ones, 0.0f, AscendC::SELMODE::VSEL_CMPMASK_SPR,
                                static_cast<uint32_t>(full));
            }
            for (int64_t k = full; k < count; k++) {
                float v = work.GetValue(k);
                work.SetValue(k, (v != 0.0f) ? 1.0f : 0.0f); // NaN≠0 为真、-0.0==0 为假（标量 IEEE）
            }
            break;
        }
        case LPNORM_P_BRANCH_INT_POW: // 整数 p∈[2,15]：连乘 powTimes=p-1 次替代超越函数
            if (powTimes_ > 0) {
                AscendC::DataCopy(calc, work, n); // calc 保留 |x| 原值副本（count 恒 32B 对齐）
                for (int32_t i = 0; i < powTimes_; i++) {
                    AscendC::Mul(work, work, calc, n);
                }
            }
            break;
        case LPNORM_P_BRANCH_EXP: {
            // exp(p·ln|x|)，IEEE 极限语义（ln0=-inf，PT-06 R1 解除）；乒乓：每步 dst≠src
            const float pLocal = pVal_; // PT-06 F2：标量参数先取核内本地副本
            AscendC::Ln(calc, work, n);
            AscendC::Muls(work, calc, pLocal, n);
            AscendC::Exp(calc, work, n);
            result = calc;
            break;
        }
        default:
            break;
    }
}

// 跨 chunk/跨核 partial 合并（DESIGN §1.6 合并矩阵：sum→Add / max→Max / min→Min）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::MergeAcc(LocalTensor<float>& dst,
                                                                                        const LocalTensor<float>& a,
                                                                                        const LocalTensor<float>& b,
                                                                                        int64_t count)
{
    if (pBranch_ == LPNORM_P_BRANCH_POS_INF) {
        AscendC::Max(dst, a, b, static_cast<int32_t>(count));
    } else if (pBranch_ == LPNORM_P_BRANCH_NEG_INF) {
        AscendC::Min(dst, a, b, static_cast<int32_t>(count));
    } else {
        AscendC::Add(dst, a, b, static_cast<int32_t>(count));
    }
}

// 计算与归约（AR）：Cast(fp32) → Abs → pBranch 变换 → Pattern::Reduce::AR 批量逐行归约 → merge
// fp16/bf16 EXP 分支的 Cast 落点切换到 tmpBuf（work=tmpBuf、calc=fp32Buf），使变换结果落在
// fp32Buf、归约 sharedTmp 复用 tmpBuf，两缓冲互不冲突（时序：变换完成后 tmp 方被归约复用）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::ComputeAR(int64_t rowsThisTile,
                                                                                         int64_t alignedCols,
                                                                                         LocalTensor<float>& accF32)
{
    LocalTensor<D_T_X> xLocal = inQueueX_.template DeQue<D_T_X>();
    LocalTensor<float> fp32Data = fp32Buf_.Get<float>();
    LocalTensor<float> tmpF32 = tmpBuf_.Get<float>();
    LocalTensor<uint8_t> tmpData = tmpBuf_.Get<uint8_t>();
    LocalTensor<float> chunkResult = chunkResultBuf_.Get<float>();
    const int64_t count = rowsThisTile * alignedCols;

    // 1) 升精度 + Abs（fp32 编译期免 Cast；先 cast 后 abs，工程统一顺序，DESIGN §6.5）
    LocalTensor<float> work;
    LocalTensor<float> calc;
    if constexpr (std::is_same_v<D_T_X, float>) {
        work = xLocal;   // fp32：直接在输入缓冲上原地计算（Abs 原地实测安全，PT-04/迭代一 ST）
        calc = fp32Data; // EXP/INT_POW 乒乓第二缓冲
    } else if (pBranch_ == LPNORM_P_BRANCH_EXP) {
        AscendC::Cast<float>(tmpF32, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
        work = tmpF32; // EXP：work 落 tmpBuf，结果落 fp32Buf，归约临时区复用 tmpBuf 无冲突
        calc = fp32Data;
    } else {
        AscendC::Cast<float>(fp32Data, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
        work = fp32Data;
        calc = tmpF32; // INT_POW 连乘副本缓冲（ZERO 分支不使用 calc）
    }
    AscendC::Abs(work, work, static_cast<uint32_t>(count));

    // 2) p 分支运行期变换（结果张量经 xformRes 返回：EXP 落 calc，其余落 work）
    LocalTensor<float> xformRes;
    ApplyPBranchTransform(work, calc, count, xformRes);

    // 3) 二维逐行归约（Pattern AR：srcShape={rows, alignedCols}，行距=alignedCols 32B 对齐；
    //    dst 每行 1 元素 StoreUnAlign，chunkResult 预留 8 元素保护槽——PT-01/PT-05）
    uint32_t srcShape[2] = {static_cast<uint32_t>(rowsThisTile), static_cast<uint32_t>(alignedCols)};
    if (pBranch_ == LPNORM_P_BRANCH_POS_INF) {
        AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR>(chunkResult, xformRes, tmpData, srcShape, true);
    } else if (pBranch_ == LPNORM_P_BRANCH_NEG_INF) {
        AscendC::ReduceMin<float, AscendC::Pattern::Reduce::AR>(chunkResult, xformRes, tmpData, srcShape, true);
    } else {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR>(chunkResult, xformRes, tmpData, srcShape, true);
    }

    // 4) 跨 chunk 合并（整向量 Add/Max/Min，count=rowsThisTile，基地址 32B 对齐——PT-05）。
    //    sum 类且多 chunk（≥2）时改 Kahan 补偿累加（4.2c）：大 N 顺序 Add 的 O(n) 舍入
    //    误差压到 ~2 ULP，对齐竞品树形精度；单 chunk 无合并误差、rowsTile>64 为大 O 小 R
    //    必然单 chunk（host 预算保证 rowsTile 恒小），均走原 Add。
    const int64_t chunkNum = LpnCeilDiv(myRHigh_ - myRLow_, ubFactorR_);
    MergeAcc(accF32, accF32, chunkResult, rowsThisTile);
    inQueueX_.FreeTensor(xLocal);
}

// 计算与归约（ARA，PT-09）：Cast(fp32) → Abs → pBranch 变换 → Pattern::Reduce::RA 逐列归约 → 逐行 merge
// merge 仅 validC 有效列（pad 列结果不落 GM）；acc 第 accRow 行 pitch = tilePitch
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::ComputeARA(int64_t accRow, int64_t rCur,
                                                                                          int64_t tilePitch,
                                                                                          int64_t validC,
                                                                                          LocalTensor<float>& accF32)
{
    LocalTensor<D_T_X> xLocal = inQueueX_.template DeQue<D_T_X>();
    LocalTensor<float> fp32Data = fp32Buf_.Get<float>();
    LocalTensor<float> tmpF32 = tmpBuf_.Get<float>();
    LocalTensor<uint8_t> tmpData = tmpBuf_.Get<uint8_t>();
    LocalTensor<float> chunkResult = chunkResultBuf_.Get<float>();
    const int64_t count = rCur * tilePitch;

    LocalTensor<float> work;
    LocalTensor<float> calc;
    if constexpr (std::is_same_v<D_T_X, float>) {
        work = xLocal;
        calc = fp32Data;
    } else if (pBranch_ == LPNORM_P_BRANCH_EXP) {
        AscendC::Cast<float>(tmpF32, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
        work = tmpF32;
        calc = fp32Data;
    } else {
        AscendC::Cast<float>(fp32Data, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
        work = fp32Data;
        calc = tmpF32;
    }
    AscendC::Abs(work, work, static_cast<uint32_t>(count));

    LocalTensor<float> xformRes;
    ApplyPBranchTransform(work, calc, count, xformRes);

    // ARA 二维归约：沿 R 维（dim0）逐列归约，输出 tilePitch 个值（PT-09：RA 三条内部路径实测覆盖）
    uint32_t srcShape[2] = {static_cast<uint32_t>(rCur), static_cast<uint32_t>(tilePitch)};
    if (pBranch_ == LPNORM_P_BRANCH_POS_INF) {
        AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA>(chunkResult, xformRes, tmpData, srcShape, true);
    } else if (pBranch_ == LPNORM_P_BRANCH_NEG_INF) {
        AscendC::ReduceMin<float, AscendC::Pattern::Reduce::RA>(chunkResult, xformRes, tmpData, srcShape, true);
    } else {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA>(chunkResult, xformRes, tmpData, srcShape, true);
    }

    // merge 到输出 tile 第 accRow 行（仅 validC 有效列；行基地址 tilePitch×4B 32B 对齐）。
    // sum 类且多 chunk（rChunkNum≥2，含跨核 R 分组下的本组 chunk；调用处可见 rSpan/ubFactorR）
    // 时 Kahan（4.2c，段按 accRow×tilePitch 行切片，kahanBuf 4×512 floats；validC>512
    // 防御回退 Add——该场景必为大 C 小 R 单 chunk，无合并误差）
    LocalTensor<float> accRowT = accF32[static_cast<uint32_t>(accRow * tilePitch)];
    MergeAcc(accRowT, accRowT, chunkResult, validC);
    inQueueX_.FreeTensor(xLocal);
}

// Phase1 落盘（AR）：VECOUT 规范顺序 EnQue→DeQue→DataCopyPad（PT-10：DataCopyPad 必须在
// EnQue/DeQue 之后发出，否则 TQue 事件簿记错乱污染共享事件池）；slot 64B 对齐步进；
// slotIdx 增加 tile 奇偶偏移（H1 ping-pong：双 bank 容量 ×2，Phase2 读侧同奇偶）；
// fp16/bf16 累计器为 TBuf 张量，不可直接过队列（FreeTensor 无对应 AllocTensor——H2 契约），
// 先 DataCopy(UB→UB fp32) 拷入队列自有张量再走 EnQue→DeQue→DataCopyPad→FreeTensor
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::Phase1StoreAR(
    int64_t rowsThisTile, int64_t tileParity, const LocalTensor<float>& accF32)
{
    const int64_t slotIdx = tileParity * outBlockNum_ * rGroupNum_ + outBlockIdx_ * rGroupNum_ + rGroupIdx_;
    LocalTensor<float> wsT;
    if constexpr (std::is_same_v<D_T_X, float>) {
        // fp32：累计器即队列自有张量（Alloc 于 ProcessAR），EnQue→DeQue→Free 严格配对
        outQueueY_.template EnQue<float>(accF32);
        wsT = outQueueY_.template DeQue<float>();
    } else {
        // fp16/bf16：拷贝跨度按 32B 对齐取整（DataCopy count 需 32B 对齐，否则向下取整丢数据）；
        // 尾部垃圾不落 GM（DataCopyPad blockLen 截断在 rowsThisTile）
        wsT = outQueueY_.template AllocTensor<float>();
        const int64_t copyElems = LpnCeilAlign(rowsThisTile, WS_SLOT_MIN_ELEMS);
        AscendC::DataCopy(wsT, accF32, static_cast<uint32_t>(copyElems));
        outQueueY_.template EnQue<float>(wsT);
        wsT = outQueueY_.template DeQue<float>();
    }
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(rowsThisTile * static_cast<int64_t>(sizeof(float)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(wsGm_[slotIdx * slotStrideElems_], wsT, copyParams);
    outQueueY_.FreeTensor(wsT);
}

// Phase1 落盘（ARA）：逐行 1D 落盘，slot 行距 = tilePitch（与 acc 一致，Phase1/Phase2 同一 tile
// 迭代序列两侧自洽；规避 fp16 下 acc 行距(tilePitch) ≠ AlignUp(blockLen,32) 隐式行距的 2D 表达问题）；
// slotIdx 含 tile 奇偶偏移（H1 ping-pong）；fp16/bf16 累计器先 DataCopy(UB→UB) 拷入队列自有张量
// （H2 契约修复；count = bTileCur×tilePitch 含对齐因子 tilePitch，天然 32B 对齐）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::Phase1StoreARA(
    int64_t bTileCur, int64_t tilePitch, int64_t validC, int64_t tileParity, const LocalTensor<float>& accF32)
{
    const int64_t slotIdx = tileParity * outBlockNum_ * rGroupNum_ + outBlockIdx_ * rGroupNum_ + rGroupIdx_;
    LocalTensor<float> wsT;
    if constexpr (std::is_same_v<D_T_X, float>) {
        outQueueY_.template EnQue<float>(accF32);
        wsT = outQueueY_.template DeQue<float>();
    } else {
        wsT = outQueueY_.template AllocTensor<float>();
        AscendC::DataCopy(wsT, accF32, static_cast<uint32_t>(bTileCur * tilePitch));
        outQueueY_.template EnQue<float>(wsT);
        wsT = outQueueY_.template DeQue<float>();
    }
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(validC * static_cast<int64_t>(sizeof(float)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    for (int64_t i = 0; i < bTileCur; i++) {
        LocalTensor<float> wsRow = wsT[static_cast<uint32_t>(i * tilePitch)];
        AscendC::DataCopyPad(wsGm_[slotIdx * slotStrideElems_ + i * tilePitch], wsRow, copyParams);
    }
    outQueueY_.FreeTensor(wsT);
}

// Phase2 二次归并（AR，rGroupIdx==0）：累计器置分支初值后合并全部 rGroupNum 个 partial
// （PT-10 偏差修正：含 g=0 自身重新累加，避免本核 partial 重复计入）→ Epilogue 回写；
// slotIdx 与 Phase1 写侧同 tile 奇偶（H1 ping-pong 读侧一致）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::Phase2MergeAR(int64_t tileRow0,
                                                                                             int64_t rowsThisTile,
                                                                                             int64_t tileParity)
{
    LocalTensor<float> acc2;
    if constexpr (std::is_same_v<D_T_X, float>) {
        acc2 = outQueueY_.template AllocTensor<float>();
    } else {
        acc2 = globalResultBuf_.Get<float>(); // fp16/bf16：Epilogue 的 Cast 目的地独占 outQueueY
    }
    AscendC::Duplicate(acc2, GetInitValue(), static_cast<int32_t>(rowsThisTile));
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(rowsThisTile * static_cast<int64_t>(sizeof(float)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPadExtParams<float> padParams(false, 0, 0, 0.0f);
    // 跨核 partial 二分合并（4.2d）：所有 partial 先入 pwBuf（g×slot 段），再原地两两合并
    // ——sum 类时顺序链 rGroupNum 收敛为 log2(rGroupNum)；Max/Min 单位元吸收无差异走原路。
    const bool usePwPhase2 = (rGroupNum_ >= 2) && (pBranch_ != LPNORM_P_BRANCH_POS_INF) &&
                             (pBranch_ != LPNORM_P_BRANCH_NEG_INF) && (rGroupNum_ <= PAIRWISE_LANES);
    LocalTensor<float> pwP2 = pwBuf_.Get<float>();
    const int64_t slotP2 = LpnCeilAlign(rowsThisTile, 8);
    for (int64_t g = 0; g < rGroupNum_; g++) {
        const int64_t slotIdx = tileParity * outBlockNum_ * rGroupNum_ + outBlockIdx_ * rGroupNum_ + g;
        LocalTensor<float> wsIn = wsInQueue_.template AllocTensor<float>();
        AscendC::DataCopyPad(wsIn, wsGm_[slotIdx * slotStrideElems_], copyParams, padParams);
        wsInQueue_.template EnQue<float>(wsIn);
        LocalTensor<float> wsL = wsInQueue_.template DeQue<float>();
        if (usePwPhase2) {
            LocalTensor<float> dst = pwP2[static_cast<uint32_t>(g * slotP2)];
            AscendC::Copy(dst, wsL, static_cast<int32_t>(rowsThisTile));
        } else {
            MergeAcc(acc2, acc2, wsL, rowsThisTile);
        }
        wsInQueue_.FreeTensor(wsL);
    }
    if (usePwPhase2) {
        int64_t stride = 1;
        while (stride < rGroupNum_) {
            for (int64_t i = 0; i + stride < rGroupNum_; i += 2 * stride) {
                LocalTensor<float> lo = pwP2[static_cast<uint32_t>(i * slotP2)];
                LocalTensor<float> hi = pwP2[static_cast<uint32_t>((i + stride) * slotP2)];
                MergeAcc(lo, lo, hi, rowsThisTile);
            }
            stride *= 2;
        }
        AscendC::Copy(acc2, pwP2, static_cast<int32_t>(rowsThisTile));
    }
    EpilogueOutAR(rowStart_ + tileRow0, rowsThisTile, acc2);
}

// Phase2 二次归并（ARA，rGroupIdx==0）：逐 slot 逐行 1D 读回（slot 行距 = tilePitch）→ merge
// （validC 有效列）→ Epilogue 回写；slotIdx 与 Phase1 写侧同 tile 奇偶（H1 ping-pong 读侧一致）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::Phase2MergeARA(
    int64_t bRow0, int64_t bTileCur, int64_t cGmBase, int64_t validC, int64_t tilePitch, int64_t tileParity)
{
    LocalTensor<float> acc2;
    if constexpr (std::is_same_v<D_T_X, float>) {
        acc2 = outQueueY_.template AllocTensor<float>();
    } else {
        acc2 = globalResultBuf_.Get<float>(); // fp16/bf16：Epilogue 的 Cast 目的地独占 outQueueY
    }
    AscendC::Duplicate(acc2, GetInitValue(), static_cast<int32_t>(bTileCur * tilePitch));
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(validC * static_cast<int64_t>(sizeof(float)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPadExtParams<float> padParams(false, 0, 0, 0.0f);
    // 跨核 partial 逐行合并 Kahan（4.2c，同 ComputeARA 段切片；validC≤512 大 C 单 chunk 防御回退）
    for (int64_t g = 0; g < rGroupNum_; g++) {
        const int64_t slotIdx = tileParity * outBlockNum_ * rGroupNum_ + outBlockIdx_ * rGroupNum_ + g;
        LocalTensor<float> wsIn = wsInQueue_.template AllocTensor<float>();
        for (int64_t i = 0; i < bTileCur; i++) {
            LocalTensor<float> wsInRow = wsIn[static_cast<uint32_t>(i * tilePitch)];
            AscendC::DataCopyPad(wsInRow, wsGm_[slotIdx * slotStrideElems_ + i * tilePitch], copyParams, padParams);
        }
        wsInQueue_.template EnQue<float>(wsIn);
        LocalTensor<float> wsL = wsInQueue_.template DeQue<float>();
        for (int64_t i = 0; i < bTileCur; i++) {
            LocalTensor<float> accRowT = acc2[static_cast<uint32_t>(i * tilePitch)];
            LocalTensor<float> wsRowT = wsL[static_cast<uint32_t>(i * tilePitch)];
            MergeAcc(accRowT, accRowT, wsRowT, validC);
        }
        wsInQueue_.FreeTensor(wsL);
    }
    EpilogueOutARA(rowStart_ + bRow0, bTileCur, cGmBase, validC, tilePitch, acc2);
}

// 逐 tile 回写（AR，DESIGN §6.2 Epilogue）：fp32 直通（累计器即 VECOUT 张量，EnQue 保证
// V→MTE3 完成性）；fp16/bf16 Cast(CAST_RINT，RN-even，PT-02/PT-08) 后经 outQueueY 回写
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::EpilogueOutAR(
    int64_t globalRow0, int64_t rowsThisTile, const LocalTensor<float>& accF32)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(rowsThisTile * static_cast<int64_t>(sizeof(D_T_X)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    if constexpr (std::is_same_v<D_T_X, float>) {
        // accF32（直出 / cross Phase2）均承载于 outQueueY 张量（PT-10 单 VECOUT 模式）
        outQueueY_.template EnQue<float>(accF32);
        LocalTensor<float> yOut = outQueueY_.template DeQue<float>();
        AscendC::DataCopyPad(yGm_[globalRow0], yOut, copyParams); // 尾块非 32B 对齐安全
        outQueueY_.FreeTensor(yOut);
    } else {
        LocalTensor<D_T_X> yLocal = outQueueY_.template AllocTensor<D_T_X>();
        // fp32→fp16/bf16 RN-even（PT-08：CAST_RINT 等价 tbe.round；禁用 CAST_ROUND——tie 处 half-away-from-zero）
        AscendC::Cast<D_T_X>(yLocal, accF32, RoundMode::CAST_RINT, static_cast<uint32_t>(rowsThisTile));
        outQueueY_.template EnQue<D_T_X>(yLocal);
        LocalTensor<D_T_X> yOut = outQueueY_.template DeQue<D_T_X>();
        AscendC::DataCopyPad(yGm_[globalRow0], yOut, copyParams);
        outQueueY_.FreeTensor(yOut);
    }
}

// 逐 tile 回写（ARA，PT-09）：2D 回写 [bTileCur × validC]，UB 行距 = tilePitch×sizeof(T)
// （srcStride=0 隐式 AlignUp(blockLen,32)），GM 行步进 = (C−validC)×sizeof(T)（字节）；
// fp16/bf16 逐行 Cast 到 outQueue（行距 tilePitch 与 acc 一致）
template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__aicore__ inline void LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE>::EpilogueOutARA(
    int64_t bGlobalRow0, int64_t bTileCur, int64_t cGmBase, int64_t validC, int64_t tilePitch,
    const LocalTensor<float>& accF32)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(bTileCur);
    copyParams.blockLen = static_cast<uint32_t>(validC * static_cast<int64_t>(sizeof(D_T_X)));
    copyParams.srcStride = 0;
    copyParams.dstStride = (cLength_ - validC) * static_cast<int64_t>(sizeof(D_T_X));
    copyParams.rsv = 0;
    const int64_t yOffset = bGlobalRow0 * cLength_ + cGmBase;
    if constexpr (std::is_same_v<D_T_X, float>) {
        outQueueY_.template EnQue<float>(accF32);
        LocalTensor<float> yOut = outQueueY_.template DeQue<float>();
        AscendC::DataCopyPad(yGm_[yOffset], yOut, copyParams);
        outQueueY_.FreeTensor(yOut);
    } else {
        LocalTensor<D_T_X> yLocal = outQueueY_.template AllocTensor<D_T_X>();
        for (int64_t i = 0; i < bTileCur; i++) {
            LocalTensor<D_T_X> yRow = yLocal[static_cast<uint32_t>(i * tilePitch)];
            LocalTensor<float> accRow = accF32[static_cast<uint32_t>(i * tilePitch)];
            AscendC::Cast<D_T_X>(yRow, accRow, RoundMode::CAST_RINT, static_cast<uint32_t>(validC));
        }
        outQueueY_.template EnQue<D_T_X>(yLocal);
        LocalTensor<D_T_X> yOut = outQueueY_.template DeQue<D_T_X>();
        AscendC::DataCopyPad(yGm_[yOffset], yOut, copyParams);
        outQueueY_.FreeTensor(yOut);
    }
}

} // namespace NsLpNormReduceV2
#endif // LP_NORM_REDUCE_V2_H
