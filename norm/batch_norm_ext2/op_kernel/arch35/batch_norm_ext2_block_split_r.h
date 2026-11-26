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
 * \file batch_norm_ext2_block_split_r.h
 * \brief
 */
#ifndef NORM_BATCH_NORM_EXT2_BLOCK_SPLIT_R_H
#define NORM_BATCH_NORM_EXT2_BLOCK_SPLIT_R_H

#include "batch_norm_ext2_base.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace BatchNormExt2Ops {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::StoreAlign;

// ===================== 末级归约的补偿(TwoSum)求和（BLOCK_SPLIT_R 专用） =====================
// LastFinalize 的末级把 usedCoreNum 个 per-core 部分和（count 加权的 mean / M2）用 BinaryAddVF
// 折半成一列结果，每个 lane 是一个独立通道。这一段是整条归约链的末级：朴素树形加法每层一次
// fp32 舍入都会原样留在 batch_mean 与 batch_var（第 5 个输出 output_reserve_space_2）上，
// 而 batch_mean 随后还要参与 (x-mean) 的近零对消、batch_var 要过 rsqrt，对这段累加误差最敏感。
// 改成补偿求和可把误差从 O(log2(N)·ε·Σ|x|) 压到 O(ε·Σ|x|)：每次加法的低位残差留在 low 里，
// 全部加完后一次性加回，末级只额外舍入一次。该收尾每个通道只跑一次、不在元素热循环里。
// 只在 LastFinalize（末级）启用；CalculateMeanVar 里每核内部的归约保持原样，不改既有口径。
// 与 full_reduce.h 的 CompensatedLastLevelReduce 同一机理，但那里归约的是同一通道的多个
// lane（需要跨 lane 合并残差），这里每个 lane 就是独立通道，残差可逐 lane 直接累加。
constexpr static float BATCH_SPLIT_MAX_FINITE_FP32 = 3.4028234663852886e38F;

// (hi, lo) += value：hi 是舍入后的和，lo 累积每次加法被舍掉的低位残差，使 hi + lo 精确等于
// 真实累加和。Knuth TwoSum 求残差，不要求 |hi| >= |value|，与 Neumaier 的分支补偿逐位等价。
__simd_callee__ inline void BatchSplitAccumulateWithError(RegTensor<float>& hi, RegTensor<float>& lo,
                                                          RegTensor<float>& value, MaskReg& mask)
{
    RegTensor<float> total;
    RegTensor<float> back;
    RegTensor<float> error;
    Add(total, hi, value, mask);
    Sub(back, total, hi, mask);
    Sub(error, total, back, mask);
    Sub(error, hi, error, mask);
    Sub(back, value, back, mask);
    Add(error, error, back, mask);
    Add(lo, lo, error, mask);
    AscendC::Reg::Move(hi, total, mask);
}

// 低位残差非有限即置 0：hi 溢出成 ±inf 时 lo 会退化成 inf - inf = NaN，再加回 hi 会把本该是
// inf 的和污染成 NaN。补偿只在有限域有意义，非有限值按 IEEE 自然传播。
__simd_callee__ inline void BatchSplitClampResidual(RegTensor<float>& lo, MaskReg& mask)
{
    RegTensor<float> zeroReg;
    MaskReg finiteMask;
    Duplicate(zeroReg, 0.0f, mask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::LE>(finiteMask, lo, BATCH_SPLIT_MAX_FINITE_FP32, mask);
    AscendC::Reg::Select(lo, lo, zeroReg, finiteMask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::GE>(finiteMask, lo, -BATCH_SPLIT_MAX_FINITE_FP32, mask);
    AscendC::Reg::Select(lo, lo, zeroReg, finiteMask);
}

// ===================== (hi, lo) / divisor 的「一次正确舍入」除法 =====================
// 朴素写法 `Add(hi, hi, lo); Div(q, hi, d)` 有两处舍入：合并残差一次、真除法一次，
// 合计 1~1.5 ULP —— 而 batch_mean / batch_var 是逐通道标量输出，交叉校验的 rmse 分母
// 就是竞品 fp32 的 0.1~0.4 ULP，两次舍入会直接把比值顶到 1.5~5 倍（实测 d_b_081 cst#2
// 0.69 ULP vs torch 0.31 ULP、d_b_343 5.5 倍）。
// 这里改用「先除得 q0（0.5 ULP），再用 FMA 精确取回 r = hi - q0·d 的余项（|r| 远小于 |hi|，
// 该乘减是单次舍入且结果精确），把余项与 low 一起再除一次加回」—— 补偿除法(classic
// two-step corrected division)。最终结果只剩一次真正影响数值的舍入，等价于对 (hi+lo)/d
// 做一次正确舍入。
//   只借用调用方两个**已空闲**的寄存器（rem/tmp），不新增常驻 RegTensor —— 本模板 finalize
//   已有 ~18 个常驻寄存器，再多占会触发寄存器溢出（实测症状是整图输出 NaN）。
//   q 可与 hi 同寄存器：先把 hi 拷进 rem，再让 Div 原位覆盖 q。
__simd_callee__ inline void BatchSplitCorrectedDivVF(RegTensor<float>& q, RegTensor<float>& hi, RegTensor<float>& lo,
                                                     RegTensor<float>& divisor, RegTensor<float>& rem,
                                                     RegTensor<float>& tmp, MaskReg& mask)
{
    // rem 先存 hi（q 常与 hi 同寄存器，Div 会把 hi 覆盖成商）
    AscendC::Reg::Move(rem, hi, mask);
    Div(q, hi, divisor, mask);     // q = fl(hi/d)，0.5ULP
    Muls(tmp, q, -1.0f, mask);     // tmp = -q
    Mula(rem, tmp, divisor, mask); // rem = hi + (-q)·d = hi - q·d（FMA 单次舍入，|rem| ≪ |hi|）
    Add(rem, rem, lo, mask);       // 并上低位残差（小量，其舍入可忽略）
    // hi 溢出成 ±inf（x 恒定 ±inf / 跨核加权 mean·count 上溢）时，q = ±inf，
    // 上面两步会得到 inf − inf = NaN；非有限修正量必须置 0，让商按 IEEE 自然传播
    // （±inf ÷ n = ±inf），否则 Add(q, q, rem) 会把本该是 ±inf 的 mean 变成 NaN。
    // 与 RAR 侧 RarCompensatedDiv 的 RarClampLaneResidual(corr) 同口径。
    BatchSplitClampResidual(rem, mask);
    Div(rem, rem, divisor, mask);
    Add(q, q, rem, mask);
}

// 双肢版本的补偿除法：同时输出商的双肢 (hiOut, loOut)，使 hiOut + loOut ≈ (hi+lo)/d，
// 且 hiOut = fl((hi+lo)/d)。与 RAR 侧 RarCompensatedDivDd 同口径：
//   q1 = fl(hi/d)；rem = fl(q1·d − hi)（精确）；corr = (lo − rem)/d；hiOut = fl(q1+corr)
//   loOut = (q1 − hiOut) + corr   ← q1−hiOut 由 Sterbenz 精确
// ±inf 时 corr / loOut 退化成 NaN，一律钳成 0（高位肢按 IEEE 传播）。
// 只取 hiOut 时完全等价于上面的 BatchSplitCorrectedDivVF。
__simd_callee__ inline void BatchSplitCorrectedDivDdVF(RegTensor<float>& hiOut, RegTensor<float>& loOut,
                                                       RegTensor<float>& hi, RegTensor<float>& lo,
                                                       RegTensor<float>& divisor, RegTensor<float>& q1,
                                                       RegTensor<float>& rem, RegTensor<float>& corr, MaskReg& mask)
{
    Div(q1, hi, divisor, mask);
    Muls(rem, hi, -1.0f, mask);
    Mula(rem, q1, divisor, mask); // rem = fl(q1·d − hi)，精确
    Sub(rem, lo, rem, mask);      // (hi+lo) − q1·d
    Div(corr, rem, divisor, mask);
    BatchSplitClampResidual(corr, mask);
    Add(hiOut, q1, corr, mask); // 正确舍入的商
    Sub(loOut, q1, hiOut, mask);
    Add(loOut, loOut, corr, mask);
    BatchSplitClampResidual(loOut, mask);
}

// ===================== 除数也是双肢的补偿除法 =====================
// 末级除数是全局 R（有偏方差）与 R−1（无偏方差）。R > 2^24 且为奇数时，fl32(R)≠R：
// 例如 R−1 = 24311189 → fl32 = 24311188（就近舍入到偶数），除数相对偏差恒为 1/n ≈ 0.69 ULP，
// 会把无偏方差系统性带偏（c_s_032 seed7 的 cst#2 rmse=2.07 就是这个）。
// 把除数拆成 (dHi, dLo)：dHi = fl32(d)，dLo = d − dHi（|dLo| ≤ 数个单位，fp32 精确），
// 则真商 = (hi+lo)/(dHi+dLo) ≈ (hi+lo)/dHi − q0·dLo/dHi，于是
//   q0  = fl(hi/dHi)
//   rem = ((hi − q0·dHi) + lo) − q0·dLo      // 第一项 FMA 精确取回，dLo 项是一阶小量
//   q   = q0 + rem/dHi
// dLo == 0 时多出的 `−q0·0` 恒为 ±0：rem 不变、clamp 结果不变 ⇒ 与旧实现在有限值上逐位相同。
__simd_callee__ inline void BatchSplitCorrectedDivDivisorDdVF(RegTensor<float>& q, RegTensor<float>& hi,
                                                              RegTensor<float>& lo, RegTensor<float>& dHi,
                                                              RegTensor<float>& dLo, RegTensor<float>& rem,
                                                              RegTensor<float>& tmp, MaskReg& mask)
{
    Div(q, hi, dHi, mask);
    Muls(rem, hi, -1.0f, mask);
    Mula(rem, q, dHi, mask); // rem = fl(q0·dHi − hi)，精确
    Sub(rem, lo, rem, mask); // (hi+lo) − q0·dHi
    Muls(tmp, q, -1.0f, mask);
    Mula(rem, tmp, dLo, mask); // −q0·dLo（dLo==0 时加的是 −0，不改变 rem）
    BatchSplitClampResidual(rem, mask);
    Div(rem, rem, dHi, mask);
    Add(q, q, rem, mask);
}

__simd_callee__ inline void BatchSplitCorrectedDivDivisorDdDdVF(RegTensor<float>& hiOut, RegTensor<float>& loOut,
                                                                RegTensor<float>& hi, RegTensor<float>& lo,
                                                                RegTensor<float>& dHi, RegTensor<float>& dLo,
                                                                RegTensor<float>& q1, RegTensor<float>& rem,
                                                                RegTensor<float>& corr, MaskReg& mask)
{
    Div(q1, hi, dHi, mask);
    Muls(rem, hi, -1.0f, mask);
    Mula(rem, q1, dHi, mask);    // rem = fl(q1·dHi − hi)，精确
    Sub(rem, lo, rem, mask);     // (hi+lo) − q1·dHi
    Muls(corr, q1, -1.0f, mask); // corr 暂存 −q1
    Mula(rem, corr, dLo, mask);  // −q1·dLo
    BatchSplitClampResidual(rem, mask);
    Div(corr, rem, dHi, mask);
    BatchSplitClampResidual(corr, mask);
    Add(hiOut, q1, corr, mask); // 正确舍入的商
    Sub(loOut, q1, hiOut, mask);
    Add(loOut, loOut, corr, mask);
    BatchSplitClampResidual(loOut, mask);
}

// ===================== fp32 双肢（double-float）原语 =====================
// y 合成链路的误差自由变换：用 (hi, lo) 两个 fp32 表示 hi+lo（hi 为舍入值，|lo| <= 0.5ulp(hi)），
// 使中间量精确到 ~48 bit，最后落 fp32 时才做一次正确舍入。
//   · TwoSum（Knuth，6 条指令，不要求 |a| >= |b|）：s = fl(a+b)，e = (a+b) - s 的精确残差；
//   · TwoProd：p = fl(a*b)，e = fl(a*b - p) —— 用 `Muls(p,-1) + Mula(a,b)` 取回，乘积在 FMA 里
//     精确，故 e 是精确残差；
//   · dd_mul / dd_add：高位肢 + 一阶交叉项后做 QuickTwoSum 归一化。
// 只在 CalcY 的训练分支（逐元素）与 rstd 精化（逐通道）里用，不进统计热循环。
__simd_callee__ inline void BatchSplitTwoSumOut(RegTensor<float>& s, RegTensor<float>& e, RegTensor<float>& a,
                                                RegTensor<float>& b, RegTensor<float>& tmp, MaskReg& mask)
{
    Add(s, a, b, mask);
    Sub(tmp, s, a, mask);
    Sub(e, s, tmp, mask);
    Sub(e, a, e, mask);
    Sub(tmp, b, tmp, mask);
    Add(e, e, tmp, mask);
    // a、b 任一非有限（±inf / NaN）时上面的残差式会退化成 inf−inf=NaN；残差肢置 0，
    // 和按 IEEE 自然传播（如 beta ≡ −inf 时 s 应为 −inf，不能是 −inf+NaN）。
    BatchSplitClampResidual(e, mask);
}

__simd_callee__ inline void BatchSplitTwoProdOut(RegTensor<float>& p, RegTensor<float>& e, RegTensor<float>& a,
                                                 RegTensor<float>& b, MaskReg& mask)
{
    Mul(p, a, b, mask);
    Muls(e, p, -1.0f, mask);
    Mula(e, a, b, mask); // e = fl(a*b - p)，精确残差
    // 乘积溢出成 ±inf 时残差是 inf−inf=NaN；置 0，让高位肢按 IEEE 传播。
    BatchSplitClampResidual(e, mask);
}

__simd_callee__ inline void BatchSplitDdMul(RegTensor<float>& hi, RegTensor<float>& lo, RegTensor<float>& aHi,
                                            RegTensor<float>& aLo, RegTensor<float>& bHi, RegTensor<float>& bLo,
                                            RegTensor<float>& tmp1, RegTensor<float>& tmp2, MaskReg& mask)
{
    Mul(hi, aHi, bHi, mask);
    Muls(lo, hi, -1.0f, mask);
    Mula(lo, aHi, bHi, mask); // 乘法残差
    Mula(lo, aHi, bLo, mask); // 一阶交叉项
    Mula(lo, aLo, bHi, mask); // 一阶交叉项
    // hi 为 ±inf 时残差肢会退化出 NaN（inf−inf、inf×0）；必须先钳成 0，
    // 否则下面 Add(tmp1, hi, lo) 会把高位肢本身也变成 NaN。
    BatchSplitClampResidual(lo, mask);
    // QuickTwoSum 归一化
    Add(tmp1, hi, lo, mask);
    Sub(tmp2, tmp1, hi, mask);
    // hi 为 ±inf 时 tmp1−hi = inf−inf = NaN；钳成 0，避免 Sub(lo, lo, tmp2) 再带回 NaN。
    BatchSplitClampResidual(tmp2, mask);
    Sub(lo, lo, tmp2, mask);
    AscendC::Reg::Move(hi, tmp1, mask);
}

__simd_callee__ inline void BatchSplitDdAdd(RegTensor<float>& hi, RegTensor<float>& lo, RegTensor<float>& bHi,
                                            RegTensor<float>& bLo, RegTensor<float>& t1, RegTensor<float>& t2,
                                            RegTensor<float>& t3, MaskReg& mask)
{
    BatchSplitTwoSumOut(t1, t2, hi, bHi, t3, mask); // (t1, t2) = TwoSum(hi, bHi)，t2 已钳位
    Add(t2, t2, lo, mask);
    Add(t2, t2, bLo, mask);
    // 两个低位肢可能非有限；并入后先钳位，避免后面的 Add(t3, t1, t2) 把高位肢污染成 NaN。
    BatchSplitClampResidual(t2, mask);
    // QuickTwoSum(t1, t2) -> (hi, lo)：t2 在最后一步之前保持为低位肢
    Add(t3, t1, t2, mask);
    Sub(t1, t3, t1, mask);
    // t1（原 s）为 ±inf 时 t3−t1 = NaN；钳成 0，避免污染低位肢。
    BatchSplitClampResidual(t1, mask);
    Sub(lo, t2, t1, mask);
    AscendC::Reg::Move(hi, t3, mask);
}

template <typename T1, typename T2>
class BatchNormExt2BlockSplitR {
public:
    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y) { return (x + y - 1) / y; }

    __aicore__ inline uint32_t CEIL_ALIGN(uint32_t x, uint32_t y) { return (x + y - 1) / y * y; }

    __aicore__ inline BatchNormExt2BlockSplitR(const BatchNormExt2BlockSplitRTilingData* tilingDataIn, TPipe* pipeIn)
    {
        this->pipe = pipeIn;
        tilingData = tilingDataIn;
        // 无偏方差不再乘 Bessel 常数 n/(n-1)：该常数无法在 fp32 中精确表示，其舍入误差会
        // 变成所有通道同向的系统性常数偏差（实测约 -0.3 ULP）。改为在 LastFinalize 里用
        // 真除法直接算 M2/(n-1)（n 与 n-1 均 < 2^24，fp32 精确可表示）。
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
                                GM_ADDR mean_out, GM_ADDR var_out, GM_ADDR batch_mean, GM_ADDR batch_var,
                                GM_ADDR workspace)
    {
        blockIdx = GetBlockIdx();
        usedCoreNum = GetBlockNum();
        int64_t rBlockOffset = 0;
        if (blockIdx < tilingData->formerCoreNums) {
            this->rLoop = tilingData->formerCoreBlockFactor;
            rBlockOffset = blockIdx * this->rLoop * tilingData->rUbFactor;
        } else {
            this->rLoop = tilingData->tailCoreBlockFactor;
            rBlockOffset = (tilingData->formerCoreBlockFactor * tilingData->formerCoreNums +
                            tilingData->tailCoreBlockFactor * (blockIdx - tilingData->formerCoreNums)) *
                           tilingData->rUbFactor;
        }
        uint32_t nowCoreRConut = (blockIdx == (usedCoreNum - 1)) ? tilingData->rUbFactor * rLoop + tilingData->tailR :
                                                                   tilingData->rUbFactor * rLoop;
        uint32_t nowCoreRConutPowOfTow = FindCofFactor(nowCoreRConut);
        uint32_t rPowOfTow = FindCofFactor(tilingData->patternR);
        this->nFactor = static_cast<float>(1) / static_cast<float>(nowCoreRConutPowOfTow);
        this->lastNFactor = static_cast<float>(1) / static_cast<float>(rPowOfTow);
        // mean / var 缩放走「真除法 + 乘回 2 的幂」，所需标量：
        //   nCount / lastNCount       —— 除数 n（本核 r 元素数 / patternR），< 2^24，fp32 精确表示
        //   nPowOfTwo / lastNPowOfTwo —— 累加时预缩放的 2 的幂，除法后乘回（精确）
        //   lastNMinusOne             —— patternR - 1（无偏方差除数）
        // 原来的倒数常数 2^k/n 与 2^m/patternR 已不再使用（其舍入是所有通道同向的常量偏置）。
        this->nCount = static_cast<float>(nowCoreRConut);
        this->nPowOfTwo = static_cast<float>(nowCoreRConutPowOfTow);
        this->lastNCount = static_cast<float>(tilingData->patternR);
        this->lastNPowOfTwo = static_cast<float>(rPowOfTow);
        this->lastNMinusOne = static_cast<float>(tilingData->patternR - 1);
        // 全局除数 R / R−1 的双肢低位肢：R > 2^24 且奇数时 fl32 会就近舍入（如 R−1=24311189→24311188），
        // 造成除数恒偏 1 个单位（≈0.69 ULP）。残差 = d − (int64)fl32(d) ∈ 少数几个整数，fp32 精确。
        this->lastNCountLow = static_cast<float>(tilingData->patternR - static_cast<int64_t>(this->lastNCount));
        this->lastNMinusOneLow = static_cast<float>((tilingData->patternR - 1) -
                                                    static_cast<int64_t>(this->lastNMinusOne));

        xGm.SetGlobalBuffer((__gm__ T1*)x + rBlockOffset * tilingData->patternA);
        betaGm.SetGlobalBuffer((__gm__ T2*)beta);
        gammaGm.SetGlobalBuffer((__gm__ T2*)gamma);
        useRunningMeanVar = tilingData->useRunningMeanVar > 0 ? true : false;
        if (useRunningMeanVar) {
            runningMeanGm.SetGlobalBuffer((__gm__ T2*)mean);
            runningVarGm.SetGlobalBuffer((__gm__ T2*)var);
        }
        yGm.SetGlobalBuffer((__gm__ T1*)y + rBlockOffset * tilingData->patternA);
        batchMeanGm.SetGlobalBuffer((__gm__ float*)batch_mean);
        // batch_var 即第 5 个输出 output_reserve_space_2：训练模式下保存**有偏方差** E[(x-mean)^2]
        batchVarGm.SetGlobalBuffer((__gm__ float*)batch_var);
        runningMeanOutGm.SetGlobalBuffer((__gm__ T2*)mean_out);
        runningVarOutGm.SetGlobalBuffer((__gm__ T2*)var_out);
        meanWsp.SetGlobalBuffer((__gm__ float*)workspace + blockIdx * tilingData->patternAAlign);
        varWsp.SetGlobalBuffer((__gm__ float*)workspace + (usedCoreNum + blockIdx) * tilingData->patternAAlign);
        // 第三块 workspace：per-core 均值的低位肢（末级跨核加权合并需要，保证 mean_hi+mean_lo 精确到
        // ~1e-15）。对应 tiling 侧 workspace 由 usedCoreNums*2 改为 *3。
        meanLowWsp.SetGlobalBuffer((__gm__ float*)workspace + (usedCoreNum * 2 + blockIdx) * tilingData->patternAAlign);
        // 第四块：per-core 有偏方差的低位肢（v = var+eps 的双肢与正确舍入 rsqrt 需要）
        varLowWsp.SetGlobalBuffer((__gm__ float*)workspace + (usedCoreNum * 3 + blockIdx) * tilingData->patternAAlign);
        workspaceGm.SetGlobalBuffer((__gm__ float*)workspace);
        // +64 元素 slack：小 C 重排时按 64 元素窗口读（y 侧尾部行可能读到本行之后最多 63 元素），
        // 留出一段余量避免越界；仅 128/256 B/缓冲。
        pipe->InitBuffer(xQueue, DOUBLE_BUFFER,
                         tilingData->rUbFactor * tilingData->aUbFactor * sizeof(T1) + 64 * sizeof(T1));
        pipe->InitBuffer(yQueue, DOUBLE_BUFFER,
                         tilingData->rUbFactor * tilingData->aUbFactor * sizeof(T1) + 64 * sizeof(T1));
        pipe->InitBuffer(gammaQueue, 1, tilingData->aUbFactor * sizeof(T2));
        pipe->InitBuffer(betaQueue, 1, tilingData->aUbFactor * sizeof(T2));
        pipe->InitBuffer(batchMeanQueue, 1, tilingData->aUbFactor * sizeof(float));
        pipe->InitBuffer(batchVarQueue, 1, tilingData->aUbFactor * sizeof(float));
        // per-core 均值的低位肢：第一阶段 finalize 产出 → 搬进 workspace 第三块
        pipe->InitBuffer(batchMeanLowQueue, 1, tilingData->aUbFactor * sizeof(float));
        // per-core 有偏方差的低位肢：同上 → workspace 第四块
        pipe->InitBuffer(batchVarLowQueue, 1, tilingData->aUbFactor * sizeof(float));
        if (useRunningMeanVar) {
            pipe->InitBuffer(runningMeanInQueue, 1, tilingData->aUbFactor * sizeof(T2));
            pipe->InitBuffer(runningVarInQueue, 1, tilingData->aUbFactor * sizeof(T2));
        }
        pipe->InitBuffer(runningMeanOutQueue, 1, tilingData->aUbFactor * sizeof(T2));
        pipe->InitBuffer(runningVarOutQueue, 1, tilingData->aUbFactor * sizeof(T2));
        pipe->InitBuffer(tmpTbuf1, tilingData->tBufUbFactor * tilingData->aUbFactor * sizeof(float));
        pipe->InitBuffer(tmpTbuf2, tilingData->tBufUbFactor * tilingData->aUbFactor * sizeof(float));
        pipe->InitBuffer(tmpTbuf3, tilingData->tBufUbFactor * tilingData->aUbFactor * sizeof(float));
        pipe->InitBuffer(tmpTbuf4, tilingData->tBufUbFactor * tilingData->aUbFactor * sizeof(float));
        int64_t rUbFactorAlign = CEIL_ALIGN(tilingData->rUbFactor, FP32_BLOCK_ALIGN_SIZE);
        int64_t usedCoreNumAlign = CEIL_ALIGN(usedCoreNum, FP32_BLOCK_ALIGN_SIZE);
        pipe->InitBuffer(countTbuf1, rUbFactorAlign * sizeof(float));
        pipe->InitBuffer(countTbuf2, usedCoreNumAlign * sizeof(float));
        // 第二阶段：把所有核的均值低位肢搬上来（usedCoreNum × aUbFactor 的 fp32）+
        // 末级有偏方差的双肢低位肢（每通道一份）。tiling 已从 UB 预算里扣除这两块的上界。
        pipe->InitBuffer(allMeanLowTbuf, usedCoreNumAlign * tilingData->aUbFactor * sizeof(float));
        pipe->InitBuffer(allVarLowTbuf, usedCoreNumAlign * tilingData->aUbFactor * sizeof(float));
        pipe->InitBuffer(varLowTbuf, tilingData->aUbFactor * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> meanTensor = tmpTbuf1.Get<float>();
        LocalTensor<float> m2Tensor = tmpTbuf2.Get<float>();
        LocalTensor<float> tmpTensor = tmpTbuf3.Get<float>();
        // mean / M2 的低位残差缓冲（Knuth TwoSum 的 lo 肢）
        //   m2LowTensor  —— 独立 buffer（tmpTbuf4），热循环里逐 lane 累积 M2 的加法残差；
        //   meanLowTensor —— 复用 tmpTbuf3：热循环期间它存 mean 的加法残差，
        //                    ProcessWelfordFinalize 一进来就用 MergeWelfordResidual 并回 mean，
        //                    之后 tmpTbuf3 继续作二分归约与 rstd 的暂存，生命周期严格不重叠。
        LocalTensor<float> meanLowTensor = tmpTbuf3.Get<float>();
        LocalTensor<float> m2LowTensor = tmpTbuf4.Get<float>();
        LocalTensor<float> countTensor1 = countTbuf1.Get<float>();
        LocalTensor<float> countTensor2 = countTbuf2.Get<float>();
        int64_t aLoopOffset = 0;
        for (int64_t aUbLoopIdx = 0; aUbLoopIdx < tilingData->aUbLoop; aUbLoopIdx++) {
            currentA = tilingData->aUbFactor;
            currentAAlign = tilingData->aUbFactor;
            if (aUbLoopIdx == (tilingData->aUbLoop - 1)) {
                currentA = tilingData->aUbTail;
                currentAAlign = CEIL_ALIGN(tilingData->aUbTail, T_BLOCK_ALIGN_SIZE);
            }
            aLoopOffset = aUbLoopIdx * tilingData->aUbFactor;
            uint32_t calcLen = tilingData->rUbFactor * currentAAlign;
            currentR = tilingData->rUbFactor;
            MeanM2TensorInit(meanTensor, m2Tensor, meanLowTensor, m2LowTensor, calcLen);
            int64_t xGmOffset = 0;
            int64_t count = 0;
            for (int64_t rUbLoopIdx = 0; rUbLoopIdx < this->rLoop; rUbLoopIdx++) {
                xGmOffset = rUbLoopIdx * tilingData->rUbFactor * tilingData->patternA + aLoopOffset;
                WelfordParallelUpdate(count, meanTensor, m2Tensor, meanLowTensor, m2LowTensor, xGmOffset, calcLen);
            }
            if ((tilingData->tailR != 0) && (blockIdx == (usedCoreNum - 1))) {
                calcLen = tilingData->tailR * currentAAlign;
                currentR = tilingData->tailR;
                xGmOffset = this->rLoop * tilingData->rUbFactor * tilingData->patternA + aLoopOffset;
                WelfordParallelUpdate(count, meanTensor, m2Tensor, meanLowTensor, m2LowTensor, xGmOffset, calcLen);
            }
            CaculateCountBuf(countTensor1, countTensor2);
            LocalTensor<float> localMeanTensor = batchMeanQueue.AllocTensor<float>();
            LocalTensor<float> localMeanLowTensor = batchMeanLowQueue.AllocTensor<float>();
            LocalTensor<float> localVarTensor = batchVarQueue.AllocTensor<float>();
            LocalTensor<float> localVarLowTensor = batchVarLowQueue.AllocTensor<float>();
            ProcessWelfordFinalize(meanTensor, m2Tensor, meanLowTensor, m2LowTensor, countTensor1, localMeanTensor,
                                   localMeanLowTensor, localVarTensor, localVarLowTensor, tmpTensor);
            batchMeanQueue.EnQue(localMeanTensor);
            batchMeanLowQueue.EnQue(localMeanLowTensor);
            batchVarQueue.EnQue(localVarTensor);
            batchVarLowQueue.EnQue(localVarLowTensor);
            localMeanTensor = batchMeanQueue.template DeQue<float>();
            localMeanLowTensor = batchMeanLowQueue.template DeQue<float>();
            localVarTensor = batchVarQueue.template DeQue<float>();
            localVarLowTensor = batchVarLowQueue.template DeQue<float>();
            DataCopy(meanWsp[aLoopOffset], localMeanTensor, currentAAlign);
            DataCopy(meanLowWsp[aLoopOffset], localMeanLowTensor, currentAAlign);
            DataCopy(varWsp[aLoopOffset], localVarTensor, currentAAlign);
            DataCopy(varLowWsp[aLoopOffset], localVarLowTensor, currentAAlign);
            batchMeanQueue.FreeTensor(localMeanTensor);
            batchMeanLowQueue.FreeTensor(localMeanLowTensor);
            batchVarQueue.FreeTensor(localVarTensor);
            batchVarLowQueue.FreeTensor(localVarLowTensor);
        }
        SyncAll();
        LocalTensor<float> allMeanTensor = tmpTbuf1.Get<float>();
        LocalTensor<float> alllVarTensor = tmpTbuf2.Get<float>();
        // 第二阶段的临时缓冲复用（第一阶段结束后这两个 tbuf 已空闲）：
        //   batchMeanLowTensor —— tmpTbuf4：LastFinalize 写出的 batch_mean 低位肢；
        //   rstdLowTensor      —— tmpTbuf1：rstd 双肢精化的低位肢（CopyInAllMeanVar/LastFinalize
        //                         用完 allMean 之后才写，下一轮 aUbLoop 会重新搬入）。
        LocalTensor<float> batchMeanLowTensor = tmpTbuf4.Get<float>();
        LocalTensor<float> rstdLowTensor = tmpTbuf1.Get<float>();
        // 所有核的均值/方差低位肢（独立 TBuf，用于末级跨核加权恢复真均值/真方差）
        LocalTensor<float> allMeanLowTensor = allMeanLowTbuf.Get<float>();
        LocalTensor<float> allVarLowTensor = allVarLowTbuf.Get<float>();
        // 末级有偏方差的双肢低位肢（每通道一份，供 var+eps 双肢与 rstd 精化）
        LocalTensor<float> varLowTensor = varLowTbuf.Get<float>();
        event_t eIdMte2ToVec = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        event_t eIdVecToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        for (int64_t aUbLoopIdx = 0; aUbLoopIdx < tilingData->aUbLoop; aUbLoopIdx++) {
            currentA = tilingData->aUbFactor;
            currentAAlign = tilingData->aUbFactor;
            if (aUbLoopIdx == (tilingData->aUbLoop - 1)) {
                currentA = tilingData->aUbTail;
                currentAAlign = CEIL_ALIGN(tilingData->aUbTail, T_BLOCK_ALIGN_SIZE);
            }
            aLoopOffset = aUbLoopIdx * tilingData->aUbFactor;
            // 反向同步，第2次的搬入需要第1次内存free
            if (aUbLoopIdx > 0) {
                WaitFlag<HardEvent::V_MTE2>(eIdVecToMte2);
            }
            CopyInAllMeanVar(allMeanTensor, alllVarTensor, allMeanLowTensor, allVarLowTensor, aLoopOffset);
            SetFlag<HardEvent::MTE2_V>(eIdMte2ToVec);
            WaitFlag<HardEvent::MTE2_V>(eIdMte2ToVec);
            LocalTensor<float> batchMeanTensor = batchMeanQueue.AllocTensor<float>();
            LocalTensor<float> batchVarTensor = batchVarQueue.AllocTensor<float>();
            LastFinalize(batchMeanTensor, batchVarTensor, allMeanTensor, alllVarTensor, allMeanLowTensor,
                         allVarLowTensor, countTensor2, tmpTensor, batchMeanLowTensor, varLowTensor);
            LocalTensor<T2> gammaTensor = gammaQueue.AllocTensor<T2>();
            LocalTensor<T2> betaTensor = betaQueue.AllocTensor<T2>();
            CopyInGammaBeta(gammaTensor, betaTensor, aLoopOffset);
            gammaQueue.EnQue(gammaTensor);
            betaQueue.EnQue(betaTensor);
            if (blockIdx == 0) {
                LocalTensor<T2> runningMeanInTensor;
                LocalTensor<T2> runningVarInTensor;
                if (useRunningMeanVar) {
                    runningMeanInTensor = runningMeanInQueue.AllocTensor<T2>();
                    runningVarInTensor = runningVarInQueue.AllocTensor<T2>();
                    CopyInRunningMeanVar(runningMeanInTensor, runningVarInTensor, aLoopOffset);
                    runningMeanInQueue.EnQue(runningMeanInTensor);
                    runningVarInQueue.EnQue(runningVarInTensor);
                    runningMeanInTensor = runningMeanInQueue.template DeQue<T2>();
                    runningVarInTensor = runningVarInQueue.template DeQue<T2>();
                }
                LocalTensor<T2> runningMeanOutTensor = runningMeanOutQueue.AllocTensor<T2>();
                LocalTensor<T2> runningVarOutTensor = runningVarOutQueue.AllocTensor<T2>();
                // alllVarTensor(tmpTbuf2) 在 LastFinalize 里被复用为「无偏方差」输出。
                CalculateRunningMeanVar(runningMeanInTensor, runningVarInTensor, runningMeanOutTensor,
                                        runningVarOutTensor, batchMeanTensor, alllVarTensor);
                if (useRunningMeanVar) {
                    runningMeanInQueue.FreeTensor(runningMeanInTensor);
                    runningVarInQueue.FreeTensor(runningVarInTensor);
                }
                runningMeanOutQueue.EnQue(runningMeanOutTensor);
                runningVarOutQueue.EnQue(runningVarOutTensor);
                runningMeanOutTensor = runningMeanOutQueue.template DeQue<T2>();
                runningVarOutTensor = runningVarOutQueue.template DeQue<T2>();
                CopyOutRunningMeanVar(runningMeanOutTensor, runningVarOutTensor, aLoopOffset);
                runningMeanOutQueue.FreeTensor(runningMeanOutTensor);
                runningVarOutQueue.FreeTensor(runningVarOutTensor);
            }
            gammaTensor = gammaQueue.DeQue<T2>();
            betaTensor = betaQueue.DeQue<T2>();
            // 需要等runningMeanVar计算完成后，才能计算成Rstd。
            // batchVarTensor 里是第 5 个输出要保存的**有偏方差**，不能被 rstd 覆盖；
            // 这里把 rstd 写到 tmpTensor(tmpTbuf3) —— LastFinalize 用完后的暂存区，不再另占 UB。
            ComputeRstdNewtonRaphsonExt2<false>(batchVarTensor, tmpTensor, static_cast<uint32_t>(currentA),
                                                tilingData->epsilon, 1.0f, VL_F32);
            // 以现有 NR 结果为种子做一次双肢 Newton 修正：r_hi 仍写 tmpTensor，r_lo 写 tmpTbuf1。
            // 逐通道一次，开销可忽略；训练分支的 y 合成需要 (r_hi, r_lo)。
            // 只服务训练分支：推理分支（useRunningMeanVar != 0）保持原 NR 结果，逐位不动。
            if (!useRunningMeanVar) {
                RefineRstdDd((__ubuf__ float*)batchVarTensor.GetPhyAddr(), (__ubuf__ float*)varLowTensor.GetPhyAddr(),
                             (__ubuf__ float*)tmpTensor.GetPhyAddr(), (__ubuf__ float*)rstdLowTensor.GetPhyAddr(),
                             static_cast<uint32_t>(currentA), tilingData->epsilon);
            }
            int64_t yGmOffset = 0;
            currentR = tilingData->rUbFactor;
            for (int64_t rUbLoopIdx = 0; rUbLoopIdx < this->rLoop; rUbLoopIdx++) {
                yGmOffset = rUbLoopIdx * tilingData->rUbFactor * tilingData->patternA + aLoopOffset;
                NormalizeX(batchMeanTensor, tmpTensor, rstdLowTensor, batchMeanLowTensor, gammaTensor, betaTensor,
                           yGmOffset);
            }
            if ((tilingData->tailR != 0) && (blockIdx == (usedCoreNum - 1))) {
                currentR = tilingData->tailR;
                yGmOffset = this->rLoop * tilingData->rUbFactor * tilingData->patternA + aLoopOffset;
                NormalizeX(batchMeanTensor, tmpTensor, rstdLowTensor, batchMeanLowTensor, gammaTensor, betaTensor,
                           yGmOffset);
            }
            // 反向同步：下一轮 CopyInAllMeanVar 会覆写 tmpTbuf1/2，必须等本轮的
            // rstd 精化与全部 CalcY 读完低位肢（tmpTbuf1 复用为 rstd_low）之后再放行。
            if (aUbLoopIdx < tilingData->aUbLoop - 1) {
                SetFlag<HardEvent::V_MTE2>(eIdVecToMte2);
            }
            gammaQueue.FreeTensor(gammaTensor);
            betaQueue.FreeTensor(betaTensor);
            if (blockIdx == 0) {
                batchMeanQueue.EnQue(batchMeanTensor);
                batchVarQueue.EnQue(batchVarTensor);
                batchMeanTensor = batchMeanQueue.template DeQue<float>();
                batchVarTensor = batchVarQueue.template DeQue<float>();
                CopyOutBatchMeanVar(batchMeanTensor, batchVarTensor, aLoopOffset);
            }
            batchMeanQueue.FreeTensor(batchMeanTensor);
            batchVarQueue.FreeTensor(batchVarTensor);
        }
    }

private:
    __aicore__ inline uint32_t FindCofFactor(uint32_t n)
    {
        // 找到比n大的最邻近的二次幂数, n = 15，结果为16
        if ((n & (n - 1)) != 0) {
            uint32_t temp = n - 1;
            temp |= temp >> 1;
            temp |= temp >> 2;
            temp |= temp >> 4;
            temp |= temp >> 8;
            temp |= temp >> 16;
            return (temp + 1);
        } else {
            return n;
        }
    }

    __aicore__ inline void CaculateCountBuf(LocalTensor<float>& tCountTensor1, LocalTensor<float>& tCountTensor2)
    {
        __ubuf__ float* tmpCountLocal1 = (__ubuf__ float*)tCountTensor1.GetPhyAddr();
        __ubuf__ float* tmpCountLocal2 = (__ubuf__ float*)tCountTensor2.GetPhyAddr();
        float baseAddCount = static_cast<float>(this->rLoop);
        float tailAddCount = static_cast<float>(this->rLoop + 1);
        uint32_t baseNum = tilingData->rUbFactor;
        uint32_t tailNum = (blockIdx == (usedCoreNum - 1)) ? tilingData->tailR : 0;
        uint16_t baseLoopCount = CEIL_DIV(baseNum, VL_F32);
        uint16_t tailLoopCount = CEIL_DIV(tailNum, VL_F32);
        float lastCoreAddCount = static_cast<float>(tilingData->tailR +
                                                    tilingData->tailCoreBlockFactor * tilingData->rUbFactor);
        if (tilingData->tailCoreNums == 0) {
            lastCoreAddCount = static_cast<float>(tilingData->tailR +
                                                  tilingData->formerCoreBlockFactor * tilingData->rUbFactor);
        }
        float tailCoreAddCount = static_cast<float>(tilingData->tailCoreBlockFactor * tilingData->rUbFactor);
        float formerCoreAddCount = static_cast<float>(tilingData->formerCoreBlockFactor * tilingData->rUbFactor);
        uint32_t firstNum = usedCoreNum;
        uint32_t secondNum = usedCoreNum - 1;
        uint32_t thirdNum = usedCoreNum - tilingData->tailCoreNums;
        if (tilingData->tailCoreNums == 0) {
            secondNum = 0;
            thirdNum = usedCoreNum - 1;
        }
        uint16_t fisrstLoopCount = CEIL_DIV(firstNum, VL_F32);
        uint16_t secondLoopCount = CEIL_DIV(secondNum, VL_F32);
        uint16_t thirdLoopCount = CEIL_DIV(thirdNum, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> tmpCount;
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;
            uint32_t sreg1 = baseNum;
            Duplicate(tmpCount, baseAddCount, pregMain);
            for (uint16_t i = 0; i < baseLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                StoreAlign(((__ubuf__ float*)tmpCountLocal1 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg2 = tailNum;
            Duplicate(tmpCount, tailAddCount, pregMain);
            for (uint16_t i = 0; i < tailLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg2);
                StoreAlign(((__ubuf__ float*)tmpCountLocal1 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg3 = firstNum;
            Duplicate(tmpCount, lastCoreAddCount, pregMain);
            for (uint16_t i = 0; i < fisrstLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg3);
                StoreAlign(((__ubuf__ float*)tmpCountLocal2 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg4 = secondNum;
            Duplicate(tmpCount, tailCoreAddCount, pregMain);
            for (uint16_t i = 0; i < secondLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg4);
                StoreAlign(((__ubuf__ float*)tmpCountLocal2 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg5 = thirdNum;
            Duplicate(tmpCount, formerCoreAddCount, pregMain);
            for (uint16_t i = 0; i < thirdLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg5);
                StoreAlign(((__ubuf__ float*)tmpCountLocal2 + i * VL_F32), tmpCount, pregLoop);
            }
        }
    }

    __aicore__ inline void MeanM2TensorInit(LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor,
                                            LocalTensor<float>& meanLowTensor, LocalTensor<float>& m2LowTensor,
                                            uint32_t len)
    {
        __ubuf__ float* meanTensorAddr = (__ubuf__ float*)meanTensor.GetPhyAddr();
        __ubuf__ float* m2TensorAddr = (__ubuf__ float*)m2Tensor.GetPhyAddr();
        __ubuf__ float* meanLowAddr = (__ubuf__ float*)meanLowTensor.GetPhyAddr();
        __ubuf__ float* m2LowAddr = (__ubuf__ float*)m2LowTensor.GetPhyAddr();
        uint16_t loopCount = CEIL_DIV(len, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> tmpMean;
            RegTensor<float> tmpM2;
            MaskReg mask0 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(tmpMean, 0.0, mask0);
            Duplicate(tmpM2, 0.0, mask0);
            MaskReg mask1;
            uint32_t sreg0 = len;
            for (uint16_t i = 0; i < loopCount; i++) {
                mask1 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                StoreAlign(meanTensorAddr + i * VL_F32, tmpMean, mask1);
                StoreAlign(m2TensorAddr + i * VL_F32, tmpM2, mask1);
                // 低位残差与主值同生命周期、同步清零
                StoreAlign(meanLowAddr + i * VL_F32, tmpMean, mask1);
                StoreAlign(m2LowAddr + i * VL_F32, tmpM2, mask1);
            }
        }
    }

    // ===================== 小 C 的 x 搬入：合并成连续大块 + UB 内原地重排 =====================
    // 问题：NHWC 且 currentA 很小（C=1/2 …）时，逐行 DataCopyPad 的 blockLen 只有 2~4 B，
    // MTE2 的块长效率极低（实测等效带宽 ~24 GB/s，而 NCHW 连续布局能到 ~485 GB/s）。
    // 做法：GM 里 currentR 行本来就是连续的（BSR 下 patternA 就是 GM 行步长），
    // 一次 blockCount=1、blockLen=currentR*currentA*sizeof(T1) 搬进队列缓冲的**前缀**
    // （紧凑布局 [R][A]），随后在本块 DeQue 之后用向量单元原地倒序重排成 [R][currentAAlign]。
    // 适用条件（不满足就走原逐行路径，逻辑/数值完全不变）：
    //   · currentA == patternA（aUbLoop==1，GM 行步长等于 currentA）；
    //   · currentA*sizeof(T1) < 32（块长确实病态；大 C 不动）；
    //   · currentA < currentAAlign ≤ 64 且 currentA | currentAAlign（紧凑/对齐两种布局的行宽都是 2 的幂关系）；
    //   · currentR 是 64/currentAAlign（每次 store 覆盖的行数）的整数倍（尾块走原路径）。
    __aicore__ inline bool UseCoalescedX()
    {
        // 【A/B 回退开关 · 当前关闭】设备实测：合并搬入/写回虽然数值正确（out0 与回退版逐位相同），
        // 但在 c_s_007 / d_b_066 / d_b_236 上 cst_bin_perf_us 反而变慢（GPU/NPU 从 0.230/0.087/0.110
        // 降到 0.115/0.073/0.057），说明这三个形状的瓶颈不是小块描述符开销，而是向量计算/低 lane
        // 利用率；重排本身的 Gather/Store 反而加重了计算。故保持关闭。Repack 的 log2 初始化与
        // isPad 语义已修正，若将来换更高效的布局方案可复用。
        return false;
        if (currentA != tilingData->patternA) {
            return false;
        }
        if (currentA * static_cast<int64_t>(sizeof(T1)) >= 32) {
            return false;
        }
        if (currentAAlign <= currentA || currentAAlign > 64 || (currentAAlign % currentA) != 0) {
            return false;
        }
        // currentR 不必是 64/currentAAlign 的整数倍：整组行走合并块，余数行由 CopyInX 退回旧逐行搬入。
        return true;
    }

    // 写回（RepackY + CopyOutY 合并块）：与搬入侧同条件，成对启用。
    __aicore__ inline bool UseCoalescedY() { return UseCoalescedX(); }

    // 原地把紧凑布局重排成对齐布局：dst[row*AAlign + c] = src[row*A + c]。
    // 每次 store 覆盖 G=64/AAlign 行、写满 64 lane（非活跃 lane 补 0，与旧 DataCopyPad 的
    // paddingValue=0 一致）；一个 64 元素的源块覆盖 AAlign/A 个组。
    // 倒序处理：第 g 组的 store 偏移 64g ≥ 第 g 组的源偏移 g·G·A（A<AAlign 时严格大于），
    // 因此写当前组不会破坏尚未读取的（更低地址的）源。
    __aicore__ inline void RepackX(LocalTensor<T1>& x)
    {
        if (!UseCoalescedX()) {
            return;
        }
        __ubuf__ T1* addr = (__ubuf__ T1*)x.GetPhyAddr();
        const uint32_t a = static_cast<uint32_t>(currentA);
        const uint32_t aa = static_cast<uint32_t>(currentAAlign);
        const uint32_t rowsPerStore = 64 / aa;        // G
        const uint32_t groupsPerChunk = aa / a;       // 一个 64 元素源块覆盖的组数
        const uint32_t groupElems = rowsPerStore * a; // 每组的源元素数
        // 只有整组行参与重排；余数行已由 CopyInX 用旧逐行路径直接写进对齐布局尾部
        const uint32_t fullRows = static_cast<uint32_t>(currentR) / rowsPerStore * rowsPerStore;
        const uint32_t groups = fullRows / rowsPerStore;
        const uint32_t chunks = (groups + groupsPerChunk - 1) / groupsPerChunk;
        uint32_t rowShift = 0;
        uint32_t aShift = 0;
        for (uint32_t t = aa; t > 1; t >>= 1) {
            rowShift++;
        }
        for (uint32_t t = a; t > 1; t >>= 1) {
            aShift++;
        }
        const uint32_t colMask = aa - 1;
        __VEC_SCOPE__
        {
            RegTensor<float> src;
            RegTensor<float> vals;
            RegTensor<float> zero;
            RegTensor<uint32_t> seq;
            RegTensor<uint32_t> col;
            RegTensor<uint32_t> row;
            RegTensor<uint32_t> idx;
            MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg active;
            RegTensor<uint32_t> aVec;
            RegTensor<uint32_t> colMaskVec;
            Duplicate(zero, 0.0f, pregAll);
            Duplicate(aVec, a, pregAll);
            Duplicate(colMaskVec, colMask, pregAll);
            AscendC::Reg::Arange((RegTensor<int32_t>&)seq, static_cast<int32_t>(0));
            AscendC::Reg::And(col, seq, colMaskVec, pregAll);
            AscendC::Reg::ShiftRights(row, seq, static_cast<int16_t>(rowShift), pregAll);
            AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::LT>(active, col, aVec, pregAll);
            // VF 作用域内的循环变量必须是 uint16_t：用正序计数 + 反序索引实现倒序处理
            for (uint16_t ci = 0; ci < static_cast<uint16_t>(chunks); ci++) {
                uint32_t c = chunks - 1U - ci;
                LoadTensorForDtypeT<T1>(src, addr, pregAll, c * 64);
                for (uint16_t ti = 0; ti < static_cast<uint16_t>(groupsPerChunk); ti++) {
                    uint32_t t = groupsPerChunk - 1U - ti;
                    uint32_t g = c * groupsPerChunk + t;
                    if (g < groups) {
                        // idx[l] = t*groupElems + row(l)*a + col(l)，全部落在本 64 元素源块内
                        AscendC::Reg::ShiftLefts(idx, row, static_cast<int16_t>(aShift), pregAll);
                        AscendC::Reg::Adds(idx, idx, t * groupElems, pregAll);
                        AscendC::Reg::Add(idx, idx, col, pregAll);
                        AscendC::Reg::Gather(vals, src, idx);
                        Select(vals, vals, zero, active); // 非活跃（padding）lane 写 0
                        if constexpr (IsSameType<T1, half>::value) {
                            RegTensor<half> vHalf;
                            Cast<half, float, castTraitB322B16>(vHalf, vals, pregAll);
                            StoreAlign<half, StoreDist::DIST_PACK_B32>(addr + g * 64, vHalf, pregAll);
                        } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                            RegTensor<bfloat16_t> vBf16;
                            Cast<bfloat16_t, float, castTraitB322B16>(vBf16, vals, pregAll);
                            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(addr + g * 64, vBf16, pregAll);
                        } else {
                            StoreAlign(addr + g * 64, vals, pregAll);
                        }
                    }
                }
            }
        }
    }

    // ===================== 小 C 的 y 写回：UB 内原地压缩 + 合并成连续大块 =====================
    // 与搬入侧成对：仅当 UseCoalescedX() 为真时启用。把 [R][AAlign] 原地压成紧凑 [R][A]
    // （padding lane 不参与），随后由 CopyOutY 一次连续大块写回 GM。
    // 正序覆盖安全性：第 S 次 store 的目标是打包区 [64S, 64S+64)，其自身源区间为
    //   [S·Rs·aa, (S+1)·Rs·aa)，Rs=64/a、aa=currentAAlign；因 a<aa 且 a|aa ⇒ Rs·aa=(64/a)·aa ≥ 128。
    //   S≥1 时 64S+63 < 128S ≤ S·Rs·aa，即当前目标恒低于当前 store 的源起点；更早的 store 目标更小，
    //   因此任何时刻写打包区都不会破坏尚未读取的源。S=0 时先读源寄存器再写 [0,64)。
    // 尾部（packed 不是 64 的整数倍）：逐行把该行的 a 个值 gather 到连续 lane，最后一次带 mask 的连续写。
    __aicore__ inline void RepackY(LocalTensor<T1>& y)
    {
        if (!UseCoalescedY()) {
            return;
        }
        __ubuf__ T1* addr = (__ubuf__ T1*)y.GetPhyAddr();
        const uint32_t a = static_cast<uint32_t>(currentA);
        const uint32_t aa = static_cast<uint32_t>(currentAAlign);
        const uint32_t G = 64 / aa;  // 一个 64 元素源寄存器覆盖的行数
        const uint32_t Rs = 64 / a;  // 一次 64 元素打包 store 覆盖的行数
        const uint32_t gpc = aa / a; // 每次 store 需要的源寄存器数
        const uint32_t groupElems = G * a;
        const uint32_t packed = static_cast<uint32_t>(currentR) * a;
        const uint32_t fullStores = packed / 64;
        const uint32_t remElems = packed % 64;
        // 打包 lane l 分解为 (行 = l / a, 通道 = l % a)；对齐源地址 = 行*aa + 通道
        uint32_t aaLog = 0;
        uint32_t aLog = 0;
        for (uint32_t t = aa; t > 1; t >>= 1) {
            aaLog++;
        }
        for (uint32_t t = a; t > 1; t >>= 1) {
            aLog++;
        }
        const uint32_t aMask = a - 1;
        __VEC_SCOPE__
        {
            RegTensor<float> src;
            RegTensor<float> pack;
            RegTensor<float> vals;
            RegTensor<float> zero;
            RegTensor<uint32_t> seq;
            RegTensor<uint32_t> lk;
            RegTensor<uint32_t> idx;
            RegTensor<uint32_t> row;
            RegTensor<uint32_t> col;
            RegTensor<uint32_t> zeroIdx;
            RegTensor<uint32_t> thr;
            RegTensor<uint32_t> aMaskVec;
            RegTensor<uint32_t> aVec;
            MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg maskT;
            Duplicate(zero, 0.0f, pregAll);
            Duplicate(zeroIdx, static_cast<uint32_t>(0), pregAll);
            Duplicate(thr, groupElems, pregAll);
            Duplicate(aMaskVec, aMask, pregAll);
            Duplicate(aVec, a, pregAll);
            AscendC::Reg::Arange((RegTensor<int32_t>&)seq, static_cast<int32_t>(0));
            for (uint16_t si = 0; si < static_cast<uint16_t>(fullStores); si++) {
                uint32_t s = si;
                Duplicate(pack, 0.0f, pregAll);
                for (uint16_t ti = 0; ti < static_cast<uint16_t>(gpc); ti++) {
                    uint32_t t = ti;
                    LoadTensorForDtypeT<T1>(src, addr, pregAll, (s * Rs + t * G) * aa);
                    // lk = seq - t*groupElems（无符号回绕）；maskT = lk < groupElems
                    AscendC::Reg::Adds(lk, seq, 0U - t * groupElems, pregAll);
                    AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::LT>(maskT, lk, thr, pregAll);
                    AscendC::Reg::ShiftRights(row, lk, static_cast<int16_t>(aLog), pregAll);
                    AscendC::Reg::And(col, lk, aMaskVec, pregAll);
                    AscendC::Reg::ShiftLefts(idx, row, static_cast<int16_t>(aaLog), pregAll);
                    AscendC::Reg::Add(idx, idx, col, pregAll);
                    Select(idx, idx, zeroIdx, maskT); // 无效 lane 索引置 0，避免 Gather 越界
                    AscendC::Reg::Gather(vals, src, idx);
                    Select(pack, vals, pack, maskT);
                }
                if constexpr (IsSameType<T1, half>::value) {
                    RegTensor<half> vHalf;
                    Cast<half, float, castTraitB322B16>(vHalf, pack, pregAll);
                    StoreAlign<half, StoreDist::DIST_PACK_B32>(addr + s * 64, vHalf, pregAll);
                } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                    RegTensor<bfloat16_t> vBf16;
                    Cast<bfloat16_t, float, castTraitB322B16>(vBf16, pack, pregAll);
                    StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(addr + s * 64, vBf16, pregAll);
                } else {
                    StoreAlign(addr + s * 64, pack, pregAll);
                }
            }
            if (remElems > 0) {
                Duplicate(pack, 0.0f, pregAll);
                for (uint16_t ki = 0; ki < static_cast<uint16_t>(remElems / a); ki++) {
                    uint32_t k = ki;
                    LoadTensorForDtypeT<T1>(src, addr, pregAll, (fullStores * Rs + k) * aa);
                    AscendC::Reg::Adds(lk, seq, 0U - k * a, pregAll);
                    AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::LT>(maskT, lk, aVec, pregAll);
                    Select(idx, lk, zeroIdx, maskT);
                    AscendC::Reg::Gather(vals, src, idx);
                    Select(pack, vals, pack, maskT);
                }
                uint32_t tailSreg = remElems;
                MaskReg tailMask = AscendC::MicroAPI::UpdateMask<float>(tailSreg);
                if constexpr (IsSameType<T1, half>::value) {
                    RegTensor<half> vHalf;
                    Cast<half, float, castTraitB322B16>(vHalf, pack, tailMask);
                    StoreAlign<half, StoreDist::DIST_PACK_B32>(addr + fullStores * 64, vHalf, tailMask);
                } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                    RegTensor<bfloat16_t> vBf16;
                    Cast<bfloat16_t, float, castTraitB322B16>(vBf16, pack, tailMask);
                    StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(addr + fullStores * 64, vBf16, tailMask);
                } else {
                    StoreAlign(addr + fullStores * 64, pack, tailMask);
                }
            }
        }
    }

    __aicore__ inline void CopyInX(LocalTensor<T1>& xInUb, int64_t offset)
    {
        if (UseCoalescedX()) {
            const int64_t rowsPerGroup = 64 / currentAAlign;
            const int64_t fullRows = (currentR / rowsPerGroup) * rowsPerGroup;
            if (fullRows > 0) {
                // 一次连续大块：整组行 × currentA 个元素（BSR 下 GM 行步长就是 patternA==currentA）
                DataCopyExtParams coalescedParams;
                coalescedParams.blockCount = 1;
                coalescedParams.blockLen = static_cast<uint32_t>(fullRows * currentA * sizeof(T1));
                coalescedParams.srcStride = 0;
                coalescedParams.dstStride = 0;
                // isPad=true：字节数不是 32B 整数倍（C=1→R*2、C=2→R*4/8），
                // 必须让 MTE2 把 UB 侧块补齐到 32B，否则尾部会被截断/错位。
                DataCopyPadExtParams<T1> coalescedPad;
                coalescedPad.isPad = true;
                coalescedPad.leftPadding = 0;
                coalescedPad.rightPadding = 0;
                coalescedPad.paddingValue = 0;
                DataCopyPad(xInUb, xGm[offset], coalescedParams, coalescedPad);
                if (fullRows < currentR) {
                    // 余数行（< 64/currentAAlign 行）：退回旧逐行搬入，直接写到对齐布局的尾部
                    DataCopyExtParams tailParams;
                    tailParams.blockCount = static_cast<uint16_t>(currentR - fullRows);
                    tailParams.blockLen = currentA * sizeof(T1);
                    tailParams.srcStride = (tilingData->patternA - currentA) * sizeof(T1);
                    tailParams.dstStride = 0;
                    DataCopyPadExtParams<T1> tailPad;
                    tailPad.isPad = true;
                    tailPad.leftPadding = 0;
                    tailPad.rightPadding = 0;
                    tailPad.paddingValue = 0;
                    DataCopyPad(xInUb[fullRows * currentAAlign], xGm[offset + fullRows * tilingData->patternA],
                                tailParams, tailPad);
                }
                return;
            }
        }
        DataCopyPadExtParams<T1> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = (currentA != currentAAlign);
        dataCopyPadExtParams.leftPadding = 0;
        // isPad配置True，rightPadding配置0，表示自动Pad到32B对齐
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = currentR;
        copyInParams.blockLen = currentA * sizeof(T1);
        copyInParams.srcStride = (tilingData->patternA - currentA) * sizeof(T1);
        copyInParams.dstStride = 0;
        DataCopyPad(xInUb, xGm[offset], copyInParams, dataCopyPadExtParams);
    }

    __aicore__ inline void CopyInAllMeanVar(LocalTensor<float>& allMeanTensor, LocalTensor<float>& alllVarTensor,
                                            LocalTensor<float>& allMeanLowTensor, LocalTensor<float>& allVarLowTensor,
                                            int64_t aLoopOffset)
    {
        DataCopyPadExtParams<float> meanVarPadParams;
        meanVarPadParams.isPad = false;
        meanVarPadParams.leftPadding = 0;
        meanVarPadParams.rightPadding = 0;
        meanVarPadParams.paddingValue = 0;
        DataCopyExtParams copyInMeanVarParams;
        copyInMeanVarParams.blockCount = usedCoreNum;
        copyInMeanVarParams.dstStride = 0;
        copyInMeanVarParams.blockLen = currentAAlign * sizeof(float);
        copyInMeanVarParams.srcStride = (tilingData->patternAAlign - currentAAlign) * sizeof(float);
        DataCopyPad(allMeanTensor, workspaceGm[aLoopOffset], copyInMeanVarParams, meanVarPadParams);
        DataCopyPad(alllVarTensor, workspaceGm[usedCoreNum * tilingData->patternAAlign + aLoopOffset],
                    copyInMeanVarParams, meanVarPadParams);
        // 第三块：per-core 均值的低位肢（末级跨核加权合并用；dstStride 与上面一致，按行紧凑排列）
        DataCopyPad(allMeanLowTensor, workspaceGm[usedCoreNum * 2 * tilingData->patternAAlign + aLoopOffset],
                    copyInMeanVarParams, meanVarPadParams);
        // 第四块：per-core 有偏方差的低位肢
        DataCopyPad(allVarLowTensor, workspaceGm[usedCoreNum * 3 * tilingData->patternAAlign + aLoopOffset],
                    copyInMeanVarParams, meanVarPadParams);
    }

    // ===================== 热循环：逐 lane 的 TwoSum 补偿（B 方案，镜像 welford.h） =====================
    // 每个 (r, a) 项是一条长度 = rLoop(+tail) 的 Welford 串行链，mean/M2 的每次加法都在链上留下
    // 一次 fp32 舍入，链长 100+ 时随机游走误差远超 1ULP —— 实测修复前 d_b_343 的方差 rmse 是
    // torch 的 5.5 倍、d_b_062 的 y 在小值域错 26964 个元素（均值需准到 ~1ULP）。故把两处加法
    // 都换成 Knuth TwoSum：
    //   · mean 更新：delta2 = delta1 * (1/count)（乘法自身的舍入无法补偿，但它是小量、相对误差
    //     ~0.5ULP(delta2)，对均值的贡献比加法链小 2 个数量级），(hi, lo) += delta2；
    //   · M2 累加：(hi, lo) += delta4。
    // hi 的自增序列与原来的朴素 Add **逐位相同**（TwoSum 的 total = fl(hi+value)），因此 inf/NaN
    // 传播语义完全不变；lo 只在 finalize 里并回。
    __aicore__ inline void WelfordParallelUpdate(int64_t& count, LocalTensor<float>& meanTensor,
                                                 LocalTensor<float>& m2Tensor, LocalTensor<float>& meanLowTensor,
                                                 LocalTensor<float>& m2LowTensor, int64_t xGmOffset, uint32_t len)
    {
        // copy in x
        LocalTensor<T1> xTensor = xQueue.AllocTensor<T1>();
        CopyInX(xTensor, xGmOffset);
        xQueue.EnQue(xTensor);
        xTensor = xQueue.DeQue<T1>();
        // 小 C：把「紧凑搬入」的 [R][A] 原地重排成消费者期望的 [R][AAlign]（纯搬运，不改算术；
        // 与后续 V 运算同管道、程序序保证可见）
        RepackX(xTensor);
        // ---------
        count += 1;
        float scale = (float)1.0 / static_cast<float>(count);
        __ubuf__ float* meanTensorAddr = (__ubuf__ float*)meanTensor.GetPhyAddr();
        __ubuf__ float* m2TensorAddr = (__ubuf__ float*)m2Tensor.GetPhyAddr();
        __ubuf__ float* meanLowAddr = (__ubuf__ float*)meanLowTensor.GetPhyAddr();
        __ubuf__ float* m2LowAddr = (__ubuf__ float*)m2LowTensor.GetPhyAddr();
        __ubuf__ T1* xTensorAddr = (__ubuf__ T1*)xTensor.GetPhyAddr();
        uint16_t loopCount = CEIL_DIV(len, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> tmpMean;
            RegTensor<float> tmpMeanLow;
            RegTensor<float> tmpM2;
            RegTensor<float> tmpM2Low;
            RegTensor<float> delta1;
            RegTensor<float> delta2;
            RegTensor<float> delta3;
            RegTensor<float> delat4;
            RegTensor<float> zero;
            RegTensor<float> absMean;
            RegTensor<float> sumX1Mean;
            RegTensor<float> scaleReg; // scale 广播到寄存器，供 Mula/Mul 使用
            MaskReg infMeanMask;
            MaskReg nanSumMask;
            MaskReg notNanSumMask;
            MaskReg zeroDeltaMask;
            MaskReg mask0;
            MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(zero, 0.0f, pregAll);
            Duplicate(scaleReg, scale, pregAll);
            uint32_t sreg0 = len;
            for (uint16_t i = 0; i < loopCount; i++) {
                mask0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadTensorForDtypeT<T1>(x1, xTensorAddr, mask0, i * VL_F32);
                LoadAlign(tmpMean, meanTensorAddr + i * VL_F32);
                LoadAlign(tmpMeanLow, meanLowAddr + i * VL_F32);
                LoadAlign(tmpM2, m2TensorAddr + i * VL_F32);
                LoadAlign(tmpM2Low, m2LowAddr + i * VL_F32);
                // delata1 = x1 - mean
                Sub(delta1, x1, tmpMean, mask0);
                // mean 为 ±inf 且 x1 + mean 不是 NaN 时，delta1 置 0，使 mean 保持 ±inf 不变；
                // 异号 inf（x1 = ∓inf、mean = ±inf）时 x1 + mean 未定义（NaN），必须保留 delta1 = ±inf，
                // 让 mean = mean + delta1 * scale 得到 NaN，与 golden 一致；
                // 同号 inf 与有限值 x1 的 x1 + mean = ±inf，仍走清零分支（有限值 - (±inf) = ∓inf 会污染均值）
                AscendC::Reg::Abs(absMean, tmpMean, mask0);
                Compares(infMeanMask, absMean, POS_INF, mask0);
                AscendC::Reg::Add(sumX1Mean, x1, tmpMean, mask0);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nanSumMask, sumX1Mean, sumX1Mean, mask0);
                AscendC::Reg::Not(notNanSumMask, nanSumMask, mask0);
                AscendC::Reg::And(zeroDeltaMask, infMeanMask, notNanSumMask, mask0);
                Select(delta1, zero, delta1, zeroDeltaMask);
                // delta2 = delta1 * scale
                Mul(delta2, delta1, scaleReg, mask0);
                // mean = mean + delta2（TwoSum：加法残差进 meanLow，使 hi+lo 精确等于累加和）
                BatchSplitAccumulateWithError(tmpMean, tmpMeanLow, delta2, mask0);
                StoreAlign(meanTensorAddr + i * VL_F32, tmpMean, mask0);
                StoreAlign(meanLowAddr + i * VL_F32, tmpMeanLow, mask0);
                // delta3 = x1 - mean
                Sub(delta3, x1, tmpMean, mask0);
                // delta4 = delta1 * delta3
                Mul(delat4, delta1, delta3, mask0);
                // M2 = M2 + delta4（TwoSum：加法残差进 m2Low）
                BatchSplitAccumulateWithError(tmpM2, tmpM2Low, delat4, mask0);
                StoreAlign(m2TensorAddr + i * VL_F32, tmpM2, mask0);
                StoreAlign(m2LowAddr + i * VL_F32, tmpM2Low, mask0);
            }
        }
        xQueue.FreeTensor(xTensor);
    }

    // 把逐 lane 的低位残差并回主值：M2/mean ← fl(hi + lo)，只多这一次舍入。
    // 与 welford.h 的 VFMergeVarResidual 同一口径：hi 溢出/NaN 时 lo 会退化成 inf-inf = NaN，
    // 先按非有限置 0，让主值按 IEEE 自然传播。
    __aicore__ inline void MergeWelfordResidual(LocalTensor<float>& hiTensor, LocalTensor<float>& lowTensor,
                                                uint32_t len)
    {
        __ubuf__ float* hiAddr = (__ubuf__ float*)hiTensor.GetPhyAddr();
        __ubuf__ float* lowAddr = (__ubuf__ float*)lowTensor.GetPhyAddr();
        uint16_t loopCount = CEIL_DIV(len, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> hi;
            RegTensor<float> low;
            MaskReg pregLoop;
            uint32_t sreg0 = len;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(hi, hiAddr + i * VL_F32);
                LoadAlign(low, lowAddr + i * VL_F32);
                BatchSplitClampResidual(low, pregLoop);
                Add(hi, hi, low, pregLoop);
                StoreAlign(hiAddr + i * VL_F32, hi, pregLoop);
            }
        }
    }

    __aicore__ inline void ProcessWelfordFinalize(LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor,
                                                  LocalTensor<float>& meanLowTensor, LocalTensor<float>& m2LowTensor,
                                                  LocalTensor<float>& countTensor, LocalTensor<float>& finalMeanTensor,
                                                  LocalTensor<float>& finalMeanLowTensor,
                                                  LocalTensor<float>& finalVarTensor,
                                                  LocalTensor<float>& finalVarLowTensor, LocalTensor<float>& tmpTensor)

    {
        // 先把热循环留下的逐 lane 低位残差并回主值，使后续「按 count 加权的行折半」看到的是
        // 每项都正确舍入的 mean_i / M2_i（误差 0.5ULP，折半后按 1/1108 权重平均 → 可忽略）。
        // 若并列归约，折半树的每一处折叠/回刷都要各带一份残差，改动面大且易漏。
        MergeWelfordResidual(meanTensor, meanLowTensor, tilingData->rUbFactor * currentAAlign);
        MergeWelfordResidual(m2Tensor, m2LowTensor, tilingData->rUbFactor * currentAAlign);
        __ubuf__ float* tmpMeanLocal = (__ubuf__ float*)meanTensor.GetPhyAddr();
        __ubuf__ float* tmpVarLocal = (__ubuf__ float*)m2Tensor.GetPhyAddr();
        __ubuf__ float* tmpCountLocal = (__ubuf__ float*)countTensor.GetPhyAddr();
        __ubuf__ float* batchMeanInUbAddr = (__ubuf__ float*)finalMeanTensor.GetPhyAddr();
        __ubuf__ float* batchMeanLowInUbAddr = (__ubuf__ float*)finalMeanLowTensor.GetPhyAddr();
        __ubuf__ float* batchVarInUbAddr = (__ubuf__ float*)finalVarTensor.GetPhyAddr();
        __ubuf__ float* batchVarLowInUbAddr = (__ubuf__ float*)finalVarLowTensor.GetPhyAddr();
        __ubuf__ float* tmpUbAddr = (__ubuf__ float*)tmpTensor.GetPhyAddr();
        WelfordFinalizeMeanVF(tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr, batchMeanLowInUbAddr,
                              batchVarInUbAddr, tmpUbAddr);
        WelfordFinalizeVarVF(tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr, batchVarInUbAddr,
                             batchVarLowInUbAddr, tmpUbAddr);
    }

    __aicore__ inline void WelfordFinalizeMeanVF(__ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                                 __ubuf__ float* tmpCountLocal, __ubuf__ float* batchMeanInUbAddr,
                                                 __ubuf__ float* batchMeanLowInUbAddr, __ubuf__ float* batchVarInUbAddr,
                                                 __ubuf__ float* binaryAddTmpAddr)
    {
        uint16_t rLoopCount = tilingData->rUbFactor;
        uint16_t aLoopCount = CEIL_DIV(currentA, VL_F32);
        uint32_t rLoopStride = currentAAlign;

        uint16_t remainderLoopCount = (tilingData->rUbFactor - tilingData->binaryAddQuotient) / SCALE_COEF_FOUR;
        uint16_t quotientLoopCount = (tilingData->binaryAddQuotient / SCALE_COEF_FOUR) - remainderLoopCount;
        uint32_t baseLineOffset = SCALE_COEF_FOUR * rLoopStride;
        uint32_t remainderOffset = tilingData->binaryAddQuotient * rLoopStride;
        uint32_t remainderCountOffset = tilingData->binaryAddQuotient;

        uint16_t binaryAddKLoop = tilingData->binaryAddK;
        uint16_t binaryAddInnerLoop = tilingData->binaryAddQuotient / SCALE_COEF_FOUR;
        uint16_t binaryAddLastLoop = tilingData->binaryAddLast;

        float numScale = this->nFactor;
        // mean 缩放改「真除法」：累加时每行已乘 numScale = 1/2^k（2 的幂，精确），故此时 x1 = Σ(count_i·mean_i)/2^k。
        // 先真除 n（= nowCoreRConut，< 2^24 可精确表示）再乘回 2^k（精确），整体等价于一次正确舍入的除法。
        // 原写法 ×(2^k/n) 的倒数常数舍入是所有通道同向的常量偏置（实测 δmean 均值 +6.63e-8），故不再使用。
        // 复用方差路径已在 Init 里建好的 nCount / nPowOfTwo 标量，不新增 buffer。
        float meanDivN = this->nCount;
        float meanPowOfTwo = this->nPowOfTwo;

        uint32_t twoRLoopSize = ROW_TWO_OFFSET * rLoopStride;
        uint32_t threeRLoopSize = ROW_THREE_OFFSET * rLoopStride;
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;

            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            RegTensor<float> rowCount;
            RegTensor<float> nextRowCount;
            RegTensor<float> remCount;
            RegTensor<float> nextRemCount;
            RegTensor<float> nReg;   // 除数 n（本核参与累加的 r 元素数）
            RegTensor<float> addLow; // 整条行折半链的低位残差

            MaskReg pregLoop;
            MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(nReg, meanDivN, pregAll);
            uint32_t sreg0 = currentA;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                Duplicate(addLow, 0.0f, pregLoop);
                // 尾块部分四行，和前面四行相加，最终是一行
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i * SCALE_COEF_FOUR;
                    uint32_t remCountOffset = i * SCALE_COEF_FOUR + remainderCountOffset;
                    // 前两行
                    TwoRowAddForMeanWithTail(x1, tmpMeanLocal, tmpCountLocal, pregLoop, quotOffset, remOffset,
                                             quotOffset + rLoopStride, remOffset + rLoopStride, quotCountOffset,
                                             remCountOffset, quotCountOffset + 1, remCountOffset + 1, rem, nextRow,
                                             remNextRow, rowCount, nextRowCount, remCount, nextRemCount, addLow,
                                             numScale);
                    // 后两行
                    TwoRowAddForMeanWithTail(x2, tmpMeanLocal, tmpCountLocal, pregLoop, quotOffset + twoRLoopSize,
                                             remOffset + twoRLoopSize, quotOffset + threeRLoopSize,
                                             remOffset + threeRLoopSize, quotCountOffset + ROW_TWO_OFFSET,
                                             remCountOffset + ROW_TWO_OFFSET, quotCountOffset + ROW_THREE_OFFSET,
                                             remCountOffset + ROW_THREE_OFFSET, rem, nextRow, remNextRow, rowCount,
                                             nextRowCount, remCount, nextRemCount, addLow, numScale);
                    BatchSplitAccumulateWithError(x1, addLow, x2, pregLoop);
                    StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + i * rLoopStride + aLoopOffset), x1, pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理4行，4行加成1行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = (remainderLoopCount + i) * SCALE_COEF_FOUR;
                    TwoRowAddForMean(x1, tmpMeanLocal, tmpCountLocal, pregLoop, baseOffset, baseOffset + rLoopStride,
                                     baseCountOffset, baseCountOffset + 1, rem, rowCount, nextRowCount, addLow,
                                     numScale);
                    TwoRowAddForMean(x2, tmpMeanLocal, tmpCountLocal, pregLoop, baseOffset + twoRLoopSize,
                                     baseOffset + threeRLoopSize, baseCountOffset + ROW_TWO_OFFSET,
                                     baseCountOffset + ROW_THREE_OFFSET, rem, rowCount, nextRowCount, addLow, numScale);
                    BatchSplitAccumulateWithError(x1, addLow, x2, pregLoop);
                    StoreAlign(
                        ((__ubuf__ float*)binaryAddTmpAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset), x1,
                        pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 最后对2的幂次行 二分累加（补偿版：残差继续累进 addLow，结束时只留下 hi 与 lo）
                CompensatedBinaryAddCarryVF(binaryAddTmpAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop,
                                            binaryAddLastLoop, pregLoop, aLoopOffset, x1, x2, x3, x4, addLow);
                // hi 为 ±inf 时 TwoSum 的残差肢会退化成 inf−inf=NaN；先钳成 0，再交给补偿除法
                // （否则 NaN 会经 lo 参与后续运算，把能按 IEEE 传播成 ±inf 的商污染成 NaN）。
                BatchSplitClampResidual(addLow, pregLoop);
                // 此时 x1(+addLow) = Σ(count_i·mean_i)/2^k。用**双肢**补偿除法 ÷n 后乘回 2^k
                // （2 的幂，逐肢都精确）：hi 落 batchMean，lo（真均值相对 hi 的残差）落
                // batchMeanLow，供末级跨核加权合并恢复真均值（d_b_323 的近零方差通道需要）。
                BatchSplitCorrectedDivDdVF(x1, x2, x1, addLow, nReg, nextRow, remNextRow, remCount, pregLoop);
                Muls(x1, x1, meanPowOfTwo, pregLoop);
                Muls(x2, x2, meanPowOfTwo, pregLoop);
                StoreAlign(((__ubuf__ float*)batchMeanInUbAddr + aLoopOffset), x1, pregLoop);
                StoreAlign(((__ubuf__ float*)batchMeanLowInUbAddr + aLoopOffset), x2, pregLoop);
            }
        }
    }

    __aicore__ inline void WelfordFinalizeVarVF(__ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                                __ubuf__ float* tmpCountLocal, __ubuf__ float* batchMeanInUbAddr,
                                                __ubuf__ float* batchVarInUbAddr, __ubuf__ float* batchVarLowInUbAddr,
                                                __ubuf__ float* binaryAddTmpAddr)
    {
        uint16_t rLoopCount = tilingData->rUbFactor;
        uint16_t aLoopCount = CEIL_DIV(currentA, VL_F32);
        uint32_t rLoopStride = currentAAlign;

        uint16_t remainderLoopCount = (tilingData->rUbFactor - tilingData->binaryAddQuotient) / SCALE_COEF_FOUR;
        uint16_t quotientLoopCount = (tilingData->binaryAddQuotient / SCALE_COEF_FOUR) - remainderLoopCount;
        uint32_t baseLineOffset = SCALE_COEF_FOUR * rLoopStride;
        uint32_t remainderOffset = tilingData->binaryAddQuotient * rLoopStride;
        uint32_t remainderCountOffset = tilingData->binaryAddQuotient;

        uint16_t binaryAddKLoop = tilingData->binaryAddK;
        uint16_t binaryAddInnerLoop = tilingData->binaryAddQuotient / SCALE_COEF_FOUR;
        uint16_t binaryAddLastLoop = tilingData->binaryAddLast;

        float numScale = this->nFactor;
        // 方差：先用真除法除以 n，再乘回 2 的幂 nPowOfTwo（精确），
        // 等价于原来的 ×(2^k/n) 但消除了倒数常数的系统性舍入偏差。
        float varTrueDivN = this->nCount;
        float varPowOfTwo = this->nPowOfTwo;

        uint32_t twoRLoopSize = ROW_TWO_OFFSET * rLoopStride;
        uint32_t threeRLoopSize = ROW_THREE_OFFSET * rLoopStride;
        __VEC_SCOPE__
        {
            RegTensor<float> saveMean;

            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;

            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            RegTensor<float> rowCount;
            RegTensor<float> nextRowCount;
            RegTensor<float> remCount;
            RegTensor<float> nextRemCount;

            RegTensor<float> rowM2;
            RegTensor<float> nextRowM2;
            RegTensor<float> remM2;
            RegTensor<float> nextRemM2;
            RegTensor<float> nReg;
            RegTensor<float> addLow;

            MaskReg pregLoop;
            MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(nReg, varTrueDivN, pregAll);
            uint32_t sreg0 = currentA;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(saveMean, ((__ubuf__ float*)batchMeanInUbAddr + aLoopOffset));
                Duplicate(addLow, 0.0f, pregLoop);
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i * SCALE_COEF_FOUR;
                    uint32_t remCountOffset = i * SCALE_COEF_FOUR + remainderCountOffset;
                    TwoRowAddForVarWithTail(x1, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, quotOffset,
                                            remOffset, quotOffset + rLoopStride, remOffset + rLoopStride,
                                            quotCountOffset, remCountOffset, quotCountOffset + 1, remCountOffset + 1,
                                            saveMean, rem, nextRow, remNextRow, rowCount, nextRowCount, remCount,
                                            nextRemCount, rowM2, nextRowM2, remM2, nextRemM2, addLow, numScale);
                    TwoRowAddForVarWithTail(x2, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop,
                                            quotOffset + twoRLoopSize, remOffset + twoRLoopSize,
                                            quotOffset + threeRLoopSize, remOffset + threeRLoopSize,
                                            quotCountOffset + ROW_TWO_OFFSET, remCountOffset + ROW_TWO_OFFSET,
                                            quotCountOffset + ROW_THREE_OFFSET, remCountOffset + ROW_THREE_OFFSET,
                                            saveMean, rem, nextRow, remNextRow, rowCount, nextRowCount, remCount,
                                            nextRemCount, rowM2, nextRowM2, remM2, nextRemM2, addLow, numScale);
                    BatchSplitAccumulateWithError(x1, addLow, x2, pregLoop);
                    StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + i * rLoopStride + aLoopOffset), x1, pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理4行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = (remainderLoopCount + i) * SCALE_COEF_FOUR;
                    TwoRowAddForVar(x1, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, baseOffset,
                                    baseOffset + rLoopStride, baseCountOffset, baseCountOffset + 1, saveMean, rem,
                                    rowCount, nextRowCount, rowM2, remM2, addLow, numScale);
                    TwoRowAddForVar(x2, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, baseOffset + twoRLoopSize,
                                    baseOffset + threeRLoopSize, baseCountOffset + ROW_TWO_OFFSET,
                                    baseCountOffset + ROW_THREE_OFFSET, saveMean, rem, rowCount, nextRowCount, rowM2,
                                    remM2, addLow, numScale);
                    BatchSplitAccumulateWithError(x1, addLow, x2, pregLoop);
                    StoreAlign(
                        ((__ubuf__ float*)binaryAddTmpAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset), x1,
                        pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                CompensatedBinaryAddCarryVF(binaryAddTmpAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop,
                                            binaryAddLastLoop, pregLoop, aLoopOffset, x1, x2, x3, x4, addLow);
                BatchSplitClampResidual(addLow, pregLoop);
                // 此时 x1(+addLow) = M2_sum/2^k（累加时逐项乘 numScale=1/2^k，精确）。
                // 有偏方差用**双肢**补偿除法：hi 落 batchVar，lo（真值相对 hi 的残差）落
                // batchVarLow，供末级跨核合并 + v=var+eps 双肢 + 正确舍入 rsqrt。
                BatchSplitCorrectedDivDdVF(x1, x2, x1, addLow, nReg, x3, x4, rem, pregLoop);
                Muls(x1, x1, varPowOfTwo, pregLoop);
                Muls(x2, x2, varPowOfTwo, pregLoop);
                StoreAlign(((__ubuf__ float*)batchVarInUbAddr + aLoopOffset), x1, pregLoop);
                StoreAlign(((__ubuf__ float*)batchVarLowInUbAddr + aLoopOffset), x2, pregLoop);
            }
        }
    }

    __aicore__ inline void BinaryAddVF(__ubuf__ float* binaryAddTmpAddr, uint32_t rLoopStride, uint16_t binaryAddKLoop,
                                       uint16_t binaryAddInnerLoop, uint16_t binaryAddLastLoop, MaskReg& pregLoop,
                                       uint32_t offset, RegTensor<float>& x1, RegTensor<float>& x2,
                                       RegTensor<float>& x3, RegTensor<float>& x4)
    {
        uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
        for (uint16_t i = 0; i < binaryAddKLoop; i++) {
            curBinaryAddInnerLoop = curBinaryAddInnerLoop / ROW_FOUR_OFFSET;
            for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
                LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET) * rLoopStride + offset));
                LoadAlign(x2, ((__ubuf__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + 1) * rLoopStride + offset));
                Add(x1, x1, x2, pregLoop);
                LoadAlign(x3, ((__ubuf__ float*)binaryAddTmpAddr +
                               (j * ROW_FOUR_OFFSET + ROW_TWO_OFFSET) * rLoopStride + offset));
                LoadAlign(x4, ((__ubuf__ float*)binaryAddTmpAddr +
                               (j * ROW_FOUR_OFFSET + ROW_THREE_OFFSET) * rLoopStride + offset));
                Add(x3, x3, x4, pregLoop);
                Add(x1, x1, x3, pregLoop);
                StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + j * rLoopStride + offset), x1, pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        for (uint16_t i = 0; i < binaryAddLastLoop; i++) {
            LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + offset));
            LoadAlign(x2, ((__ubuf__ float*)binaryAddTmpAddr + rLoopStride + offset));
            Add(x1, x1, x2, pregLoop);
            StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + offset), x1, pregLoop);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }

    // 与 BinaryAddVF 结构逐位等价的末级归约，只把每一次加法换成补偿加法（Knuth TwoSum）。
    // low 由调用方**预置**（行折半阶段已经累积的残差），本函数继续往里累加，并且**不**把 low
    // 并回 hi —— 只加 hi 等于把补偿丢掉一半，而在这里并回等于又舍入一次；正确做法是把
    // (hi, low) 一起交给末级的 BatchSplitCorrectedDivVF，让合并与除法的舍入合成一次。
    // 结构（4 路折半 + 串行尾加）、读写位置与 BinaryAddVF 完全一致，故调用点的读取口径不变。
    __aicore__ inline void CompensatedBinaryAddCarryVF(__ubuf__ float* binaryAddTmpAddr, uint32_t rLoopStride,
                                                       uint16_t binaryAddKLoop, uint16_t binaryAddInnerLoop,
                                                       uint16_t binaryAddLastLoop, MaskReg& pregLoop, uint32_t offset,
                                                       RegTensor<float>& x1, RegTensor<float>& x2, RegTensor<float>& x3,
                                                       RegTensor<float>& x4, RegTensor<float>& low)
    {
        uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
        for (uint16_t i = 0; i < binaryAddKLoop; i++) {
            curBinaryAddInnerLoop = curBinaryAddInnerLoop / ROW_FOUR_OFFSET;
            for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
                LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET) * rLoopStride + offset));
                LoadAlign(x2, ((__ubuf__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + 1) * rLoopStride + offset));
                BatchSplitAccumulateWithError(x1, low, x2, pregLoop);
                LoadAlign(x3, ((__ubuf__ float*)binaryAddTmpAddr +
                               (j * ROW_FOUR_OFFSET + ROW_TWO_OFFSET) * rLoopStride + offset));
                LoadAlign(x4, ((__ubuf__ float*)binaryAddTmpAddr +
                               (j * ROW_FOUR_OFFSET + ROW_THREE_OFFSET) * rLoopStride + offset));
                BatchSplitAccumulateWithError(x3, low, x4, pregLoop);
                // 与 BinaryAddVF 一致：x1 = (row0+row1) + (row2+row3)，两级加法的残差都进 low。
                BatchSplitAccumulateWithError(x1, low, x3, pregLoop);
                StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + j * rLoopStride + offset), x1, pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        for (uint16_t i = 0; i < binaryAddLastLoop; i++) {
            LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + offset));
            LoadAlign(x2, ((__ubuf__ float*)binaryAddTmpAddr + rLoopStride + offset));
            BatchSplitAccumulateWithError(x1, low, x2, pregLoop);
            StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + offset), x1, pregLoop);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        // 折半结束后最终结果落在 row 0（不论上面的循环是否执行过），显式读回作为 hi 交给调用方。
        LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + offset));
    }

    // 行折半（4 行 → 1 行）的补偿版本：每一次加法换成 Knuth TwoSum，低位残差统一累进调用方的
    // low 寄存器（与 CompensatedBinaryAddCarryVF 同一口径 —— 残差之和 + hi 精确等于真值，
    // 因此整条折半链 + 折半树只需要**一个** low，末级一次性加回）。
    __aicore__ inline void TwoRowAddForMeanWithTail(RegTensor<float>& dst, __ubuf__ float* input,
                                                    __ubuf__ float* tCount, MaskReg& preg, uint32_t offset1,
                                                    uint32_t offset2, uint32_t offset3, uint32_t offset4,
                                                    uint32_t offset5, uint32_t offset6, uint32_t offset7,
                                                    uint32_t offset8, RegTensor<float>& rem, RegTensor<float>& nextRow,
                                                    RegTensor<float>& remNextRow, RegTensor<float>& dstCount,
                                                    RegTensor<float>& remCount, RegTensor<float>& nextRowCount,
                                                    RegTensor<float>& remNextRowCount, RegTensor<float>& low, float n)
    {
        LoadAlign(dst, ((__ubuf__ float*)(input) + (offset1)));
        LoadAlign(rem, ((__ubuf__ float*)(input) + (offset2)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dstCount, ((__ubuf__ float*)(tCount) + (offset5)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount, ((__ubuf__ float*)(tCount) + (offset6)));
        Mul(dst, dst, dstCount, preg);
        Mul(rem, rem, remCount, preg);
        Muls(dst, dst, n, preg);
        Muls(rem, rem, n, preg);
        BatchSplitAccumulateWithError(dst, low, rem, preg);
        LoadAlign(nextRow, ((__ubuf__ float*)(input) + (offset3)));
        LoadAlign(remNextRow, ((__ubuf__ float*)(input) + (offset4)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(nextRowCount, ((__ubuf__ float*)(tCount) + (offset7)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(remNextRowCount, ((__ubuf__ float*)(tCount) + (offset8)));
        Mul(nextRow, nextRow, nextRowCount, preg);
        Mul(remNextRow, remNextRow, remNextRowCount, preg);
        Muls(nextRow, nextRow, n, preg);
        Muls(remNextRow, remNextRow, n, preg);
        BatchSplitAccumulateWithError(nextRow, low, remNextRow, preg);
        BatchSplitAccumulateWithError(dst, low, nextRow, preg);
    }

    __aicore__ inline void TwoRowAddForMean(RegTensor<float>& dst, __ubuf__ float* input, __ubuf__ float* tCount,
                                            MaskReg& preg, uint32_t offset1, uint32_t offset2, uint32_t offset5,
                                            uint32_t offset6, RegTensor<float>& rem, RegTensor<float>& dstCount,
                                            RegTensor<float>& remCount, RegTensor<float>& low, float n)
    {
        LoadAlign(dst, ((__ubuf__ float*)(input) + (offset1)));
        LoadAlign(rem, ((__ubuf__ float*)(input) + (offset2)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dstCount, ((__ubuf__ float*)(tCount) + (offset5)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount, ((__ubuf__ float*)(tCount) + (offset6)));
        Mul(dst, dst, dstCount, preg);
        Mul(rem, rem, remCount, preg);
        Muls(dst, dst, n, preg);
        Muls(rem, rem, n, preg);
        BatchSplitAccumulateWithError(dst, low, rem, preg);
    }

    __aicore__ inline void TwoRowAddForVarWithTail(
        RegTensor<float>& dst, __ubuf__ float* tmpMean, __ubuf__ float* tmpM2, __ubuf__ float* tCount, MaskReg& preg,
        uint32_t offset1, uint32_t offset2, uint32_t offset3, uint32_t offset4, uint32_t offset5, uint32_t offset6,
        uint32_t offset7, uint32_t offset8, RegTensor<float>& mean, RegTensor<float>& rem, RegTensor<float>& nextRow,
        RegTensor<float>& remNextRow, RegTensor<float>& dstCount, RegTensor<float>& remCount,
        RegTensor<float>& nextRowCount, RegTensor<float>& remNextRowCount, RegTensor<float>& dstM2,
        RegTensor<float>& remM2, RegTensor<float>& nextRowM2, RegTensor<float>& remNextRowM2, RegTensor<float>& low,
        float n)
    {
        LoadAlign(dst, ((__ubuf__ float*)(tmpMean) + (offset1)));
        LoadAlign(rem, ((__ubuf__ float*)(tmpMean) + (offset2)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dstCount, ((__ubuf__ float*)(tCount) + (offset5)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount, ((__ubuf__ float*)(tCount) + (offset6)));
        Sub(dst, dst, mean, preg);
        Mul(dst, dst, dst, preg);
        Sub(rem, rem, mean, preg);
        Mul(rem, rem, rem, preg);
        Mul(dst, dst, dstCount, preg);
        Mul(rem, rem, remCount, preg);
        LoadAlign(dstM2, ((__ubuf__ float*)(tmpM2) + (offset1)));
        LoadAlign(remM2, ((__ubuf__ float*)(tmpM2) + (offset2)));
        // 先按 numScale（2 的幂，精确）缩放到与折半树同一域再做 TwoSum：
        // 否则加法残差会以**未缩放**的量级累进 low，与 hi 的域差 2^k 倍，补偿项被放大。
        Muls(dst, dst, n, preg);
        Muls(dstM2, dstM2, n, preg);
        BatchSplitAccumulateWithError(dst, low, dstM2, preg);
        Muls(rem, rem, n, preg);
        Muls(remM2, remM2, n, preg);
        BatchSplitAccumulateWithError(rem, low, remM2, preg);
        BatchSplitAccumulateWithError(dst, low, rem, preg);

        LoadAlign(nextRow, ((__ubuf__ float*)(tmpMean) + (offset3)));
        LoadAlign(remNextRow, ((__ubuf__ float*)(tmpMean) + (offset4)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(nextRowCount, ((__ubuf__ float*)(tCount) + (offset7)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(remNextRowCount, ((__ubuf__ float*)(tCount) + (offset8)));
        Sub(nextRow, nextRow, mean, preg);
        Mul(nextRow, nextRow, nextRow, preg);
        Sub(remNextRow, remNextRow, mean, preg);
        Mul(remNextRow, remNextRow, remNextRow, preg);
        Mul(nextRow, nextRow, nextRowCount, preg);
        Mul(remNextRow, remNextRow, remNextRowCount, preg);
        LoadAlign(nextRowM2, ((__ubuf__ float*)(tmpM2) + (offset3)));
        LoadAlign(remNextRowM2, ((__ubuf__ float*)(tmpM2) + (offset4)));
        Muls(nextRow, nextRow, n, preg);
        Muls(nextRowM2, nextRowM2, n, preg);
        BatchSplitAccumulateWithError(nextRow, low, nextRowM2, preg);
        Muls(remNextRow, remNextRow, n, preg);
        Muls(remNextRowM2, remNextRowM2, n, preg);
        BatchSplitAccumulateWithError(remNextRow, low, remNextRowM2, preg);
        BatchSplitAccumulateWithError(nextRow, low, remNextRow, preg);

        BatchSplitAccumulateWithError(dst, low, nextRow, preg);
    }

    __aicore__ inline void TwoRowAddForVar(RegTensor<float>& dst, __ubuf__ float* tmpMean, __ubuf__ float* tmpM2,
                                           __ubuf__ float* tCount, MaskReg& preg, uint32_t offset1, uint32_t offset2,
                                           uint32_t offset5, uint32_t offset6, RegTensor<float>& mean,
                                           RegTensor<float>& rem, RegTensor<float>& dstCount,
                                           RegTensor<float>& remCount, RegTensor<float>& dstM2, RegTensor<float>& remM2,
                                           RegTensor<float>& low, float n)
    {
        LoadAlign(dst, ((__ubuf__ float*)(tmpMean) + (offset1)));
        LoadAlign(rem, ((__ubuf__ float*)(tmpMean) + (offset2)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dstCount, ((__ubuf__ float*)(tCount) + (offset5)));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount, ((__ubuf__ float*)(tCount) + (offset6)));
        Sub(dst, dst, mean, preg);
        Mul(dst, dst, dst, preg);
        Sub(rem, rem, mean, preg);
        Mul(rem, rem, rem, preg);
        Mul(dst, dst, dstCount, preg);
        Mul(rem, rem, remCount, preg);
        LoadAlign(dstM2, ((__ubuf__ float*)(tmpM2) + (offset1)));
        LoadAlign(remM2, ((__ubuf__ float*)(tmpM2) + (offset2)));
        Muls(dst, dst, n, preg);
        Muls(dstM2, dstM2, n, preg);
        BatchSplitAccumulateWithError(dst, low, dstM2, preg);
        Muls(rem, rem, n, preg);
        Muls(remM2, remM2, n, preg);
        BatchSplitAccumulateWithError(rem, low, remM2, preg);
        BatchSplitAccumulateWithError(dst, low, rem, preg);
    }

    __aicore__ inline void LastFinalize(LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchVarTensor,
                                        LocalTensor<float>& meanTensor, LocalTensor<float>& varTensor,
                                        LocalTensor<float>& allMeanLowTensor, LocalTensor<float>& allVarLowTensor,
                                        LocalTensor<float>& countTensor, LocalTensor<float>& tmpTensor,
                                        LocalTensor<float>& meanLowTensor, LocalTensor<float>& varLowTensor)
    {
        __ubuf__ float* tmpMeanLocal = (__ubuf__ float*)meanTensor.GetPhyAddr();
        __ubuf__ float* tmpVarLocal = (__ubuf__ float*)varTensor.GetPhyAddr();
        __ubuf__ float* tmpCountLocal = (__ubuf__ float*)countTensor.GetPhyAddr();
        __ubuf__ float* allMeanLowAddr = (__ubuf__ float*)allMeanLowTensor.GetPhyAddr();
        __ubuf__ float* allVarLowAddr = (__ubuf__ float*)allVarLowTensor.GetPhyAddr();
        __ubuf__ float* batchMeanTensorAddr = (__ubuf__ float*)batchMeanTensor.GetPhyAddr();
        __ubuf__ float* batchVarTensorAddr = (__ubuf__ float*)batchVarTensor.GetPhyAddr();
        __ubuf__ float* batchMeanLowAddr = (__ubuf__ float*)meanLowTensor.GetPhyAddr();
        __ubuf__ float* batchVarLowAddr = (__ubuf__ float*)varLowTensor.GetPhyAddr();
        __ubuf__ float* tmpUbAddr = (__ubuf__ float*)tmpTensor.GetPhyAddr();
        uint16_t aLoopCount = CEIL_DIV(currentA, VL_F32);
        uint32_t rLoopStride = currentAAlign;
        uint16_t remainderLoopCount = (usedCoreNum - tilingData->lastBinaryAddQuotient);
        uint16_t quotientLoopCount = tilingData->lastBinaryAddQuotient - remainderLoopCount;
        uint32_t baseLineOffset = rLoopStride;
        uint32_t remainderOffset = tilingData->lastBinaryAddQuotient * rLoopStride;
        uint32_t remainderCountOffset = tilingData->lastBinaryAddQuotient;
        uint16_t binaryAddKLoop = tilingData->lastBinaryAddK;
        uint16_t binaryAddInnerLoop = tilingData->lastBinaryAddQuotient;
        uint16_t binaryAddLastLoop = tilingData->lastBinaryAddLast;
        float numScale = this->lastNFactor;
        // 最后一级缩放统一走「真除法 + 乘回 2 的幂」，mean 与 var 共用同一组标量：
        //   累加时每行已乘 numScale = 1/2^m（2 的幂，精确），故 BinaryAddVF 的结果是 Σ/2^m。
        //   ÷patternR（mean 与有偏方差共用）、÷(patternR-1)（无偏方差）后再乘回 2^m（精确），
        //   等价于一次正确舍入的除法；不再乘倒数常数 2^m/patternR 与 Bessel 常数 n/(n-1)。
        float divN = this->lastNCount;
        float divNMinusOne = this->lastNMinusOne;
        float divNLow = this->lastNCountLow;            // R − fl32(R)
        float divNMinusOneLow = this->lastNMinusOneLow; // (R−1) − fl32(R−1)
        float powOfTwo = this->lastNPowOfTwo;
        __VEC_SCOPE__
        {
            RegTensor<float> quot;
            RegTensor<float> rem;
            RegTensor<float> quotCount;
            RegTensor<float> remCount;
            RegTensor<float> oriQuotMean;
            RegTensor<float> oriRemMean;
            RegTensor<float> resMean;
            RegTensor<float> resVar;
            RegTensor<float> resVarLow; // 有偏方差的双肢低位肢（供 var+eps 双肢 / rstd 精化）
            RegTensor<float> resVarUnbiased;
            RegTensor<float> nReg;
            RegTensor<float> nMinusOneReg;
            RegTensor<float> nRegLow;
            RegTensor<float> nMinusOneRegLow;
            RegTensor<float> addLow; // 末级补偿归约的低位残差（每通道独立，逐次归约内复用）
            // 末级低阶修正（精确乘积残差 + per-core 均值低位肢）
            RegTensor<float> lowCorr;
            RegTensor<float> bcCount;
            RegTensor<float> lowRow;

            MaskReg pregLoop;
            MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(nReg, divN, pregAll);
            Duplicate(nMinusOneReg, divNMinusOne, pregAll);
            Duplicate(nRegLow, divNLow, pregAll);
            Duplicate(nMinusOneRegLow, divNMinusOneLow, pregAll);
            uint32_t sreg0 = currentA;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                // 尾块部分按行加至前面
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i;
                    uint32_t remCountOffset = i + remainderCountOffset;
                    LoadAlign(quot, ((__ubuf__ float*)(tmpMeanLocal) + (quotOffset)));
                    LoadAlign(rem, ((__ubuf__ float*)(tmpMeanLocal) + (remOffset)));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                             ((__ubuf__ float*)(tmpCountLocal) + quotCountOffset));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount,
                                                             ((__ubuf__ float*)(tmpCountLocal) + remCountOffset));
                    Mul(quot, quot, quotCount, pregLoop);
                    Mul(rem, rem, remCount, pregLoop);
                    Muls(quot, quot, numScale, pregLoop);
                    Muls(rem, rem, numScale, pregLoop);
                    Add(quot, quot, rem, pregLoop);
                    StoreAlign(((__ubuf__ float*)tmpUbAddr + i * rLoopStride + aLoopOffset), quot, pregLoop);
                }
                // 整块部分除已经叠加了尾块的，需要乘count和scale
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = remainderLoopCount + i;
                    LoadAlign(quot, ((__ubuf__ float*)(tmpMeanLocal) + (baseOffset)));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                             ((__ubuf__ float*)(tmpCountLocal) + baseCountOffset));
                    Mul(quot, quot, quotCount, pregLoop);
                    Muls(quot, quot, numScale, pregLoop);
                    StoreAlign(((__ubuf__ float*)tmpUbAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset),
                               quot, pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 最后对2的幂次行 二分累加
                // 末级换成补偿归约：mean 的错误会原样传进 (x-mean) 的近零对消，最敏感。
                Duplicate(addLow, 0.0f, pregLoop);
                CompensatedBinaryAddCarryVF(tmpUbAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop,
                                            binaryAddLastLoop, pregLoop, aLoopOffset, quot, rem, quotCount, remCount,
                                            addLow);
                // ===== 末级低阶修正（把「每核均值自身 0.5 ULP 舍入」与「乘积舍入」补回来） =====
                // 补偿折半只补了**加法**残差；这里再补两项：
                //   ① TwoProd 取回 mean_i·count_i 的乘积残差（乘积在 FMA 里精确）；
                //   ② 每核均值的除法低位肢 meanLow_i·count_i。
                // 两者都按 numScale（2 的幂，精确）缩放后累进 addLow，使 (quot, addLow) 精确到
                // ~1e-15，末级商与 batchMeanLow 才是真均值的双肢（d_b_323 的近零方差通道需要）。
                Duplicate(lowCorr, 0.0f, pregLoop);
                for (uint16_t i = 0; i < static_cast<uint16_t>(usedCoreNum); i++) {
                    LoadAlign(rem, tmpMeanLocal + static_cast<uint32_t>(i) * rLoopStride + aLoopOffset);
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(bcCount, tmpCountLocal + i);
                    LoadAlign(lowRow, allMeanLowAddr + static_cast<uint32_t>(i) * rLoopStride + aLoopOffset);
                    Mul(quotCount, rem, bcCount, pregLoop);
                    Muls(remCount, quotCount, -1.0f, pregLoop);
                    Mula(remCount, rem, bcCount, pregLoop); // remCount = fl(rem·count − p)，精确
                    Mul(lowRow, lowRow, bcCount, pregLoop);
                    Add(remCount, remCount, lowRow, pregLoop);
                    Muls(remCount, remCount, numScale, pregLoop);
                    Add(lowCorr, lowCorr, remCount, pregLoop);
                }
                Add(addLow, addLow, lowCorr, pregLoop);
                // hi 为 ±inf 时残差肢会退化成 inf−inf=NaN；先钳成 0，让商按 IEEE 自然传播。
                BatchSplitClampResidual(addLow, pregLoop);
                // quot(+addLow) = Σ(count_i·mean_i)/2^m。除以**双肢除数** R 后乘回 2^m（精确）。
                BatchSplitCorrectedDivDivisorDdVF(resMean, quot, addLow, nReg, nRegLow, quotCount, remCount, pregLoop);
                // batch_mean 的低位肢（供 y 的双肢合成；不能随最后一步舍入丢掉）：
                //   q_pre = fl((quot+addLow)/n)，余项 numerator = (quot+addLow) - q_pre·n，
                //   再 ÷n、乘回 2^m（精确）→ mean_lo，使 (mean_hi, mean_lo) 精确到 ~48 bit。
                //   FMA 一次取回 quotient - q_pre·n（乘积在 FMA 里精确）；仅 C 个通道跑一次。
                //   quot 在此之后不再用于 mean（var 段会重新装载），可以原位覆盖。
                Muls(remCount, resMean, -1.0f, pregLoop); // -q_pre
                Mula(quot, remCount, nReg, pregLoop);     // quot - q_pre·n（FMA 单次舍入）
                Add(quot, quot, addLow, pregLoop);        // 并入末级补偿归约的 carry
                // ±inf 商时上面两步是 inf−inf=NaN；钳成 0 → 余项 0 → 低位肢 0，
                // 绝不能把 NaN 传给 CalcY 的 d_lo（会把 y 一起污染成 NaN）。
                BatchSplitClampResidual(quot, pregLoop);
                Div(quot, quot, nReg, pregLoop);             // 余项 ÷ n
                Muls(remCount, quot, powOfTwo, pregLoop);    // 乘回 2^m
                BatchSplitClampResidual(remCount, pregLoop); // 低位肢非有限一律置 0
                StoreAlign(((__ubuf__ float*)batchMeanLowAddr + aLoopOffset), remCount, pregLoop);
                Muls(resMean, resMean, powOfTwo, pregLoop);
                StoreAlign(((__ubuf__ float*)batchMeanTensorAddr + aLoopOffset), resMean, pregLoop);
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i;
                    uint32_t remCountOffset = i + remainderCountOffset;
                    LoadAlign(quot, ((__ubuf__ float*)(tmpVarLocal) + (quotOffset)));
                    LoadAlign(rem, ((__ubuf__ float*)(tmpVarLocal) + (remOffset)));
                    LoadAlign(oriQuotMean, ((__ubuf__ float*)(tmpMeanLocal) + (quotOffset)));
                    LoadAlign(oriRemMean, ((__ubuf__ float*)(tmpMeanLocal) + (remOffset)));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                             ((__ubuf__ float*)(tmpCountLocal) + quotCountOffset));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount,
                                                             ((__ubuf__ float*)(tmpCountLocal) + remCountOffset));
                    Sub(oriQuotMean, oriQuotMean, resMean, pregLoop);
                    Sub(oriRemMean, oriRemMean, resMean, pregLoop);
                    Mul(oriQuotMean, oriQuotMean, oriQuotMean, pregLoop);
                    Mul(oriRemMean, oriRemMean, oriRemMean, pregLoop);
                    Mul(oriQuotMean, oriQuotMean, quotCount, pregLoop);
                    Mul(oriRemMean, oriRemMean, remCount, pregLoop);
                    Mul(quot, quot, quotCount, pregLoop);
                    Mul(rem, rem, remCount, pregLoop);
                    Add(quot, quot, oriQuotMean, pregLoop);
                    Add(rem, rem, oriRemMean, pregLoop);
                    Muls(quot, quot, numScale, pregLoop);
                    Muls(rem, rem, numScale, pregLoop);
                    Add(quot, quot, rem, pregLoop);
                    StoreAlign(((__ubuf__ float*)tmpUbAddr + i * rLoopStride + aLoopOffset), quot, pregLoop);
                }
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = remainderLoopCount + i;
                    LoadAlign(quot, ((__ubuf__ float*)(tmpVarLocal) + (baseOffset)));
                    LoadAlign(oriQuotMean, ((__ubuf__ float*)(tmpMeanLocal) + (baseOffset)));
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                             ((__ubuf__ float*)(tmpCountLocal) + baseCountOffset));
                    Sub(oriQuotMean, oriQuotMean, resMean, pregLoop);
                    Mul(oriQuotMean, oriQuotMean, oriQuotMean, pregLoop);
                    Mul(oriQuotMean, oriQuotMean, quotCount, pregLoop);
                    Mul(quot, quot, quotCount, pregLoop);
                    Add(quot, quot, oriQuotMean, pregLoop);
                    Muls(quot, quot, numScale, pregLoop);
                    StoreAlign(((__ubuf__ float*)tmpUbAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset),
                               quot, pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 最后对2的幂次行 二分累加
                // 末级换成补偿归约：M2 的误差会被 ÷n 与 rsqrt 折算到第 5 个输出。
                Duplicate(addLow, 0.0f, pregLoop);
                CompensatedBinaryAddCarryVF(tmpUbAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop,
                                            binaryAddLastLoop, pregLoop, aLoopOffset, quot, rem, quotCount, remCount,
                                            addLow);
                // ===== 末级低阶修正（M2 侧的乘积舍入） =====
                // 补偿折半只补加法残差；这里补 M2_i·count_i 与 count_i·Δ² 两个乘积的舍入残差
                // （TwoProd 在 FMA 里精确取回），使 (quot, addLow) 精确到 ~1e-15。
                Duplicate(lowCorr, 0.0f, pregLoop);
                for (uint16_t i = 0; i < static_cast<uint16_t>(usedCoreNum); i++) {
                    LoadAlign(rem, tmpVarLocal + static_cast<uint32_t>(i) * rLoopStride + aLoopOffset);
                    LoadAlign(lowRow, tmpMeanLocal + static_cast<uint32_t>(i) * rLoopStride + aLoopOffset);
                    LoadAlign<float, LoadDist::DIST_BRC_B32>(bcCount, tmpCountLocal + i);
                    Mul(quotCount, rem, bcCount, pregLoop);
                    Muls(remCount, quotCount, -1.0f, pregLoop);
                    Mula(remCount, rem, bcCount, pregLoop); // M2_i·count 的精确残差
                    Sub(lowRow, lowRow, resMean, pregLoop);
                    Mul(lowRow, lowRow, lowRow, pregLoop);
                    Mul(quotCount, lowRow, bcCount, pregLoop);
                    Muls(oriRemMean, quotCount, -1.0f, pregLoop);
                    Mula(oriRemMean, lowRow, bcCount, pregLoop); // count·Δ² 的精确残差
                    Add(remCount, remCount, oriRemMean, pregLoop);
                    // 每核方差自身的除法低位肢（M2_sum 残差 = varLow_i · count_i）
                    LoadAlign(lowRow, allVarLowAddr + static_cast<uint32_t>(i) * rLoopStride + aLoopOffset);
                    Mul(lowRow, lowRow, bcCount, pregLoop);
                    Add(remCount, remCount, lowRow, pregLoop);
                    Muls(remCount, remCount, numScale, pregLoop);
                    Add(lowCorr, lowCorr, remCount, pregLoop);
                }
                Add(addLow, addLow, lowCorr, pregLoop);
                // hi 为 ±inf 时残差肢会退化成 inf−inf=NaN；先钳成 0，让商按 IEEE 自然传播。
                BatchSplitClampResidual(addLow, pregLoop);
                // 累加时已乘 numScale = 1/2^m = 1/2^ceil(log2(patternR))（精确），故此时
                // quot(+addLow) = M2_sum/2^m。补偿除法后乘回 2^m（精确），
                // 得到**有偏方差** M2/n（第 5 个输出 output_reserve_space_2 的契约值）
                // 与**无偏方差** M2/(n-1)（供 running var 使用，替代原来乘 Bessel 常数）。
                // 无偏方差的分母是 **R−1**：R−1 为奇数且 >2^24 时 fl32 会就近舍入（差 1 个单位），
                // 必须用双肢除数，否则 out2 带 −0.69 ULP 系统偏差（c_s_032 seed7）。
                BatchSplitCorrectedDivDivisorDdVF(resVarUnbiased, quot, addLow, nMinusOneReg, nMinusOneRegLow,
                                                  quotCount, remCount, pregLoop);
                // 有偏方差用**双肢**补偿除法：hi 落 out4，lo 落 varLow（供 v=var+eps 的双肢与 rstd 精化）
                BatchSplitCorrectedDivDivisorDdDdVF(resVar, resVarLow, quot, addLow, nReg, nRegLow, quotCount, remCount,
                                                    oriRemMean, pregLoop);
                Muls(resVarUnbiased, resVarUnbiased, powOfTwo, pregLoop);
                Muls(resVar, resVar, powOfTwo, pregLoop);
                Muls(resVarLow, resVarLow, powOfTwo, pregLoop);
                StoreAlign(((__ubuf__ float*)batchVarTensorAddr + aLoopOffset), resVar, pregLoop);
                StoreAlign(((__ubuf__ float*)batchVarLowAddr + aLoopOffset), resVarLow, pregLoop);
                // 复用 tmpVarLocal(tmpTbuf2) 存放无偏方差：LastFinalize 对该列的读取已在本次
                // aIndex 内完成，后续 aIndex 只访问自己那一列，故不会互相覆盖。
                StoreAlign(((__ubuf__ float*)tmpVarLocal + aLoopOffset), resVarUnbiased, pregLoop);
            }
        }
    }

    // ===================== rstd 双肢精化（逐通道，只跑一次） =====================
    // 输入 v 也是双肢：(var_hi, var_lo) + eps（TwoSum，精确），因此 v 精确到 ~1e-15，
    // 不再是 v = fl32(var+eps) 那一次 0.25 ULP 的舍入（这是 rstd 偏差的主要来源）。
    // 以 r0（现有 NR 输出）为种子，在双肢里做一次 Newton 修正：
    //   (p,pe) = TwoProd(r0,r0)              → 精确的 r0²
    //   e      = 1 - v·(p,pe)                → 双肢残差
    //   r      = (r0,0) + (0.5·r0,0)·e       → 双肢 rstd（r_hi 正确舍入到 fp32，r_lo 给 y 链）
    // 保留原有 inf/NaN 语义：v == +inf → rstd = 0；v == 0 → rstd = POS_INF（与 NR 实现一致）。
    __aicore__ inline void RefineRstdDd(__ubuf__ float* varAddr, __ubuf__ float* varLowAddr, __ubuf__ float* rstdHiAddr,
                                        __ubuf__ float* rstdLoAddr, uint32_t count, float epsilon)
    {
        static constexpr float POS_INF = 3.40282366920938E+38;
        uint16_t loopCount = static_cast<uint16_t>((count + VL_F32 - 1) / VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> var;
            RegTensor<float> v;
            RegTensor<float> r0;
            RegTensor<float> rHi;
            RegTensor<float> rLo;
            RegTensor<float> pHi;
            RegTensor<float> pLo;
            RegTensor<float> eHi;
            RegTensor<float> eLo;
            RegTensor<float> cHi;
            RegTensor<float> cLo;
            RegTensor<float> halfR0;
            RegTensor<float> t1;
            RegTensor<float> t2;
            RegTensor<float> t3;
            RegTensor<float> t4;
            RegTensor<float> zero;
            RegTensor<float> one;
            RegTensor<float> posInf;
            RegTensor<float> epsReg;
            RegTensor<float> vLo;
            MaskReg pregLoop;
            MaskReg cmpZero;
            MaskReg cmpInf;
            MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(zero, 0.0f, pregAll);
            Duplicate(one, 1.0f, pregAll);
            Duplicate(posInf, POS_INF, pregAll);
            Duplicate(epsReg, epsilon, pregAll);
            uint32_t sreg = count;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg);
                LoadAlign(var, varAddr + i * VL_F32);
                // v = (var_hi, var_lo) + eps（TwoSum 精确；vLo 是 v 的低位肢）
                BatchSplitTwoSumOut(v, vLo, var, epsReg, t1, pregLoop);
                LoadAlign(var, varLowAddr + i * VL_F32);
                Add(vLo, vLo, var, pregLoop);
                BatchSplitClampResidual(vLo, pregLoop);
                LoadAlign(r0, rstdHiAddr + i * VL_F32);
                // (pHi,pLo) = r0²
                BatchSplitTwoProdOut(pHi, pLo, r0, r0, pregLoop);
                // (t1,t2) = v · (pHi,pLo)，v 用双肢
                BatchSplitDdMul(t1, t2, v, vLo, pHi, pLo, t3, t4, pregLoop);
                // (eHi,eLo) = (1,0) - (t1,t2)
                Muls(t3, t1, -1.0f, pregLoop);
                Muls(t4, t2, -1.0f, pregLoop);
                AscendC::Reg::Move(eHi, one, pregLoop);
                AscendC::Reg::Move(eLo, zero, pregLoop);
                BatchSplitDdAdd(eHi, eLo, t3, t4, t1, t2, pHi, pregLoop);
                // (cHi,cLo) = (0.5·r0) · (eHi,eLo)，0.5·r0 精确
                Muls(halfR0, r0, 0.5f, pregLoop);
                BatchSplitDdMul(cHi, cLo, halfR0, zero, eHi, eLo, t1, t2, pregLoop);
                // (rHi,rLo) = (r0,0) + (cHi,cLo)
                AscendC::Reg::Move(rHi, r0, pregLoop);
                AscendC::Reg::Move(rLo, zero, pregLoop);
                BatchSplitDdAdd(rHi, rLo, cHi, cLo, t1, t2, pHi, pregLoop);
                // inf/NaN 兜底（与 ComputeRstdNewtonRaphsonRegExt2 相同）
                Compares(cmpInf, v, POS_INF, pregLoop);
                Select(rHi, zero, rHi, cmpInf);
                Select(rLo, zero, rLo, cmpInf);
                Compares(cmpZero, v, 0.0f, pregLoop);
                Select(rHi, posInf, rHi, cmpZero);
                Select(rLo, zero, rLo, cmpZero);
                // 低位肢非有限一律置 0：v 为 NaN 时 rHi 已是 NaN，rLo 只负责精度，
                // 不能让它携带 NaN 去污染 dd_mul 的交叉项。
                BatchSplitClampResidual(rLo, pregLoop);
                StoreAlign(rstdHiAddr + i * VL_F32, rHi, pregLoop);
                StoreAlign(rstdLoAddr + i * VL_F32, rLo, pregLoop);
            }
        }
    }

    __aicore__ inline void CopyInGammaBeta(LocalTensor<T2>& gammaInUb, LocalTensor<T2>& betaInUb, int64_t gmOffset)
    {
        DataCopyPadExtParams<T2> dataCopyPadExtParamsT;
        dataCopyPadExtParamsT.isPad = false;
        dataCopyPadExtParamsT.leftPadding = 0;
        dataCopyPadExtParamsT.rightPadding = 0;
        dataCopyPadExtParamsT.paddingValue = 0;
        DataCopyExtParams copyInParamsT;
        copyInParamsT.blockCount = 1;
        copyInParamsT.blockLen = currentA * sizeof(T2);
        copyInParamsT.srcStride = 0;
        copyInParamsT.dstStride = 0;
        DataCopyPad(betaInUb, betaGm[gmOffset], copyInParamsT, dataCopyPadExtParamsT);
        DataCopyPad(gammaInUb, gammaGm[gmOffset], copyInParamsT, dataCopyPadExtParamsT);
    }

    __aicore__ inline void CopyInRunningMeanVar(LocalTensor<T2>& runningMeanInUb, LocalTensor<T2>& runningVarInUb,
                                                int64_t gmOffset)
    {
        DataCopyPadExtParams<T2> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentA * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(runningMeanInUb, runningMeanGm[gmOffset], copyInParams, dataCopyPadExtParams);
        DataCopyPad(runningVarInUb, runningVarGm[gmOffset], copyInParams, dataCopyPadExtParams);
    }

    // unbiasedVarTensor 里是 LastFinalize 用真除法算出的无偏方差 M2/(n-1)，
    // running_var = (1-m)*running_var + m * M2/(n-1)。
    // 原实现把 batch_var(=M2/n) 再乘 Bessel 常数 fl(n/(n-1))，该常数舍入会带来系统性偏差。
    __aicore__ inline void CalculateRunningMeanVar(LocalTensor<T2>& runningMeanInUb, LocalTensor<T2>& runningVarInUb,
                                                   LocalTensor<T2>& runningMeanOutUb, LocalTensor<T2>& runningVarOutUb,
                                                   LocalTensor<float>& batchMeanTensor,
                                                   LocalTensor<float>& unbiasedVarTensor)
    {
        __ubuf__ T2* runningMeanInUbAddr = nullptr;
        __ubuf__ T2* runningVarInUbAddr = nullptr;
        if (useRunningMeanVar) {
            runningMeanInUbAddr = (__ubuf__ T2*)runningMeanInUb.GetPhyAddr();
            runningVarInUbAddr = (__ubuf__ T2*)runningVarInUb.GetPhyAddr();
        }
        __ubuf__ T2* runningMeanOutUbAddr = (__ubuf__ T2*)runningMeanOutUb.GetPhyAddr();
        __ubuf__ T2* runningVarOutUbAddr = (__ubuf__ T2*)runningVarOutUb.GetPhyAddr();
        __ubuf__ float* batchMeanTensorAddr = (__ubuf__ float*)batchMeanTensor.GetPhyAddr();
        __ubuf__ float* unbiasedVarTensorAddr = (__ubuf__ float*)unbiasedVarTensor.GetPhyAddr();
        uint16_t aLoop = CEIL_DIV(currentA, VL_F32);

        float m = tilingData->momentum;
        float oneSubM = tilingData->momentumReverse;
        bool vfUseRunningMeanVar = useRunningMeanVar;
        __VEC_SCOPE__
        {
            RegTensor<float> mean;
            RegTensor<float> var;
            RegTensor<float> runningMean;
            RegTensor<float> saveMean;
            RegTensor<float> runningVar;
            RegTensor<float> saveVar;
            MaskReg pregLoop;
            uint32_t sreg2 = currentA;
            for (uint16_t k = 0; k < aLoop; k++) {
                pregLoop = UpdateMask<float>(sreg2);
                // running var（无偏方差已由真除法得到，这里只乘 momentum，不再乘 Bessel 常数）
                LoadAlign(var, ((__ubuf__ float*)unbiasedVarTensorAddr + k * VL_F32));
                Muls(saveVar, var, m, pregLoop);
                if (vfUseRunningMeanVar) {
                    LoadTensorForDtypeT<T2>(runningVar, runningVarInUbAddr, pregLoop, k * VL_F32);
                    Muls(runningVar, runningVar, oneSubM, pregLoop);
                    Add(saveVar, saveVar, runningVar, pregLoop);
                }

                StoreTensorForDtypeT<T2>(runningVarOutUbAddr, saveVar, pregLoop, k * VL_FP32);

                // running mean
                LoadAlign(mean, ((__ubuf__ float*)batchMeanTensorAddr + k * VL_F32));
                Muls(saveMean, mean, m, pregLoop);
                if (vfUseRunningMeanVar) {
                    LoadTensorForDtypeT<T2>(runningMean, runningMeanInUbAddr, pregLoop, k * VL_F32);
                    Muls(runningMean, runningMean, oneSubM, pregLoop);
                    Add(saveMean, saveMean, runningMean, pregLoop);
                }

                StoreTensorForDtypeT<T2>(runningMeanOutUbAddr, saveMean, pregLoop, k * VL_F32);
            }
        }
    }

    __aicore__ inline void CopyOutRunningMeanVar(LocalTensor<T2>& runningMeanOutUb, LocalTensor<T2>& runningVarOutUb,
                                                 int64_t gmOffset)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentA * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(runningMeanOutGm[gmOffset], runningMeanOutUb, copyInParams);
        DataCopyPad(runningVarOutGm[gmOffset], runningVarOutUb, copyInParams);
    }

    // 拷出 batch_mean(output_reserve_space_1) 与**有偏方差**(output_reserve_space_2)。
    __aicore__ inline void CopyOutBatchMeanVar(LocalTensor<float>& batchMeanInUb, LocalTensor<float>& batchVarInUb,
                                               int64_t gmOffset)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentA * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(batchMeanGm[gmOffset], batchMeanInUb, copyInParams);
        DataCopyPad(batchVarGm[gmOffset], batchVarInUb, copyInParams);
    }

    __aicore__ inline void NormalizeX(LocalTensor<float>& batchMeanTensor, LocalTensor<float>& rstdTensor,
                                      LocalTensor<float>& rstdLowTensor, LocalTensor<float>& meanLowTensor,
                                      LocalTensor<T2>& gammaTensor, LocalTensor<T2>& betaTensor, int64_t yGmOffset)
    {
        LocalTensor<T1> xTensor = xQueue.AllocTensor<T1>();
        CopyInX(xTensor, yGmOffset);
        xQueue.EnQue(xTensor);
        xTensor = xQueue.DeQue<T1>();
        RepackX(xTensor); // 小 C：紧凑 [R][A] → 对齐 [R][AAlign]
        LocalTensor<T1> yTensor = yQueue.AllocTensor<T1>();
        CalcY(batchMeanTensor, rstdTensor, rstdLowTensor, meanLowTensor, gammaTensor, betaTensor, xTensor, yTensor);
        // 小 C：把 y 从对齐布局原地压成紧凑 [R][A]，供 CopyOutY 一次连续大块写回（纯置换，不改算术）
        RepackY(yTensor);
        xQueue.FreeTensor(xTensor);
        yQueue.EnQue(yTensor);
        yTensor = yQueue.template DeQue<T1>();
        CopyOutY(yTensor, yGmOffset);
        yQueue.FreeTensor(yTensor);
    }

    // y 的合成（按分支）：
    //   · 训练（useRunningMeanVar == 0，key 600000 唯一可达分支）：误差自由变换（双肢）：
    //       (k_hi,k_lo) = dd_mul((rstd_hi,rstd_lo), (gamma,0))      逐通道、进循环前一次
    //       (d_hi,d_lo) = TwoSum(x, -mean_hi); d_lo -= mean_lo
    //       (p_hi,p_lo) = dd_mul((d_hi,d_lo), (k_hi,k_lo))
    //       (y_hi,y_lo) = TwoSum(p_hi, beta); y_lo += p_lo
    //       y           = fl32(y_hi + y_lo)                          ← 对精确值一次正确舍入
    //     落档只做一次 fp32 -> 声明 dtype 的 Cast。中间量精确到 ~48 bit，使内核 y 与
    //     fl32(y_精确) 逐位一致 —— 三例 fp16 的失败都是「1 ULP 共用标量把 |y| 最大通道
    //     甩过 fp16 边界」，靠调运算顺序只能碰运气，必须把标量与逐元素链路都做成双肢。
    //   · 推理（useRunningMeanVar != 0）：保持原写法 y = (x - mean) * rstd * gamma + beta
    //     （mean/rstd 来自输入，golden 侧运算顺序也不同），不使用低位肢。
    __aicore__ inline void CalcY(LocalTensor<float>& batchMeanTensor, LocalTensor<float>& rstdTensor,
                                 LocalTensor<float>& rstdLowTensor, LocalTensor<float>& meanLowTensor,
                                 LocalTensor<T2>& gammaTensor, LocalTensor<T2>& betaTensor, LocalTensor<T1>& xTensor,
                                 LocalTensor<T1>& yTensor)
    {
        __ubuf__ float* batchMeanTensorAddr = (__ubuf__ float*)batchMeanTensor.GetPhyAddr();
        __ubuf__ float* rstdTensorAddr = (__ubuf__ float*)rstdTensor.GetPhyAddr();
        __ubuf__ float* rstdLowTensorAddr = (__ubuf__ float*)rstdLowTensor.GetPhyAddr();
        __ubuf__ float* meanLowTensorAddr = (__ubuf__ float*)meanLowTensor.GetPhyAddr();
        __ubuf__ T1* xTensorAddr = (__ubuf__ T1*)xTensor.GetPhyAddr();
        __ubuf__ T1* yTensorAddr = (__ubuf__ T1*)yTensor.GetPhyAddr();
        __ubuf__ T2* gammaTensorAddr = (__ubuf__ T2*)gammaTensor.GetPhyAddr();
        __ubuf__ T2* betaTensorAddr = (__ubuf__ T2*)betaTensor.GetPhyAddr();
        uint16_t numLoop = CEIL_DIV(currentA, VL_F32);
        uint16_t lineLoop = currentR;
        bool vfRstdFirst = useRunningMeanVar; // true：推理；false：训练（双肢合成）
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> mean;
            RegTensor<float> meanLo;
            RegTensor<float> negMean;
            RegTensor<float> rstd;
            RegTensor<float> rstdLo;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> kHi;
            RegTensor<float> kLo;
            RegTensor<float> dHi;
            RegTensor<float> dLo;
            RegTensor<float> pHi;
            RegTensor<float> pLo;
            RegTensor<float> y;
            RegTensor<float> yLo;
            RegTensor<float> tmp1;
            RegTensor<float> tmp2;
            RegTensor<float> zero;
            MaskReg mask0;
            MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(zero, 0.0f, pregAll);
            uint32_t sreg0 = currentA;
            for (uint16_t i = 0; i < numLoop; i++) {
                mask0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(mean, batchMeanTensorAddr + i * VL_F32);
                LoadAlign(rstd, rstdTensorAddr + i * VL_F32);
                LoadTensorForDtypeT<T2>(gamma, gammaTensorAddr, mask0, i * VL_F32);
                LoadTensorForDtypeT<T2>(beta, betaTensorAddr, mask0, i * VL_F32);
                if (!vfRstdFirst) {
                    LoadAlign(meanLo, meanLowTensorAddr + i * VL_F32);
                    LoadAlign(rstdLo, rstdLowTensorAddr + i * VL_F32);
                    // 防御：低位肢只能是有限残差，非有限一律置 0，避免 d_lo / k_lo 把 NaN 带进 y。
                    BatchSplitClampResidual(meanLo, mask0);
                    BatchSplitClampResidual(rstdLo, mask0);
                    Muls(negMean, mean, -1.0f, mask0);
                    // k = dd_mul((rstd_hi,rstd_lo), (gamma,0))，逐通道合成一次
                    BatchSplitDdMul(kHi, kLo, rstd, rstdLo, gamma, zero, tmp1, tmp2, mask0);
                    BatchSplitClampResidual(kLo, mask0); // gamma 非有限时 k_lo 退化，置 0
                }
                for (uint16_t j = 0; j < lineLoop; j++) {
                    LoadTensorForDtypeT<T1>(x1, xTensorAddr, mask0, i * VL_F32 + j * currentAAlign);
                    if (vfRstdFirst) {
                        // 推理：y = beta + (x - mean) * rstd * gamma
                        Sub(tmp1, x1, mean, mask0);
                        Mul(tmp1, tmp1, rstd, mask0);
                        AscendC::Reg::Move(y, beta);
                        Mula(y, tmp1, gamma, mask0);
                    } else {
                        // 训练：双肢合成，末步 fl32(y_hi + y_lo) 一次正确舍入
                        BatchSplitTwoSumOut(dHi, dLo, x1, negMean, tmp1, mask0); // d = x - mean_hi（精确残差）
                        Sub(dLo, dLo, meanLo, mask0);                            // 并入 mean 低位肢
                        BatchSplitClampResidual(dLo, mask0);                     // 低位肢非有限 → 0
                        BatchSplitDdMul(pHi, pLo, dHi, dLo, kHi, kLo, tmp1, tmp2, mask0); // p = d * k
                        BatchSplitClampResidual(pLo, mask0);                              // 低位肢非有限 → 0
                        BatchSplitTwoSumOut(y, yLo, pHi, beta, tmp1, mask0);              // y_hi = fl(p_hi+beta)
                        Add(yLo, yLo, pLo, mask0);                                        // 并入 p 的低位肢
                        // 统一规则：低位肢加回高位肢之前钳位。beta ≡ −inf 时 y_hi = −inf、
                        // y_lo 原为 NaN；置 0 后 y = −inf + 0 = −inf（c_s_024 的回归点）。
                        BatchSplitClampResidual(yLo, mask0);
                        Add(y, y, yLo, mask0); // dd -> fp32
                    }
                    if constexpr (IsSameType<T1, half>::value) {
                        RegTensor<half> yFp16;
                        Cast<half, float, castTraitB322B16>(yFp16, y, mask0);
                        StoreAlign<half, StoreDist::DIST_PACK_B32>(yTensorAddr + i * VL_F32 + j * currentAAlign, yFp16,
                                                                   mask0);
                    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, y, mask0);
                        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(yTensorAddr + i * VL_F32 + j * currentAAlign,
                                                                         xBf16, mask0);
                    } else {
                        StoreAlign(yTensorAddr + i * VL_F32 + j * currentAAlign, y, mask0);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOutY(LocalTensor<T1>& yOutUb, int64_t offset)
    {
        if (UseCoalescedY()) {
            // 与搬入侧成对：RepackY 已把 y 原地压成紧凑 [R][A]，一次连续大块写回，
            // 只写本核的 currentR 行（blockLen 精确），不越界。
            DataCopyExtParams coalescedOutParams;
            coalescedOutParams.blockCount = 1;
            coalescedOutParams.blockLen = static_cast<uint32_t>(currentR * currentA * sizeof(T1));
            coalescedOutParams.srcStride = 0;
            coalescedOutParams.dstStride = 0;
            DataCopyPad(yGm[offset], yOutUb, coalescedOutParams);
            return;
        }
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = currentR;
        copyInParams.blockLen = currentA * sizeof(T1);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (tilingData->patternA - currentA) * sizeof(T1);
        DataCopyPad(yGm[offset], yOutUb, copyInParams);
    }

    /* global memory address */
    GlobalTensor<T1> xGm;
    GlobalTensor<T2> betaGm;
    GlobalTensor<T2> gammaGm;
    GlobalTensor<T2> runningMeanGm;
    GlobalTensor<T2> runningVarGm;

    GlobalTensor<T1> yGm;
    GlobalTensor<float> batchMeanGm;
    GlobalTensor<float> batchVarGm;
    GlobalTensor<T2> runningMeanOutGm;
    GlobalTensor<T2> runningVarOutGm;
    GlobalTensor<float> meanWsp;
    GlobalTensor<float> varWsp;
    GlobalTensor<float> meanLowWsp;
    GlobalTensor<float> varLowWsp;
    GlobalTensor<float> workspaceGm;

    const BatchNormExt2BlockSplitRTilingData* tilingData;
    TPipe* pipe;

    /* variable */
    int64_t rLoop = 0;
    int64_t currentA = 0;
    int64_t currentAAlign = 0;
    int64_t currentR = 0;

    float nFactor = 0;
    float lastNFactor = 0;

    // mean / var 的真除法标量（n 与 n-1 均 < 2^24，fp32 精确可表示）
    float nCount = 0;        // 本核 r 元素数 n
    float nPowOfTwo = 0;     // 累加预缩放用的 2 的幂（除法后乘回，精确）
    float lastNCount = 0;    // patternR
    float lastNMinusOne = 0; // patternR - 1
    float lastNPowOfTwo = 0; // 最后一级累加预缩放用的 2 的幂
    // 全局除数 R / R−1 的低位肢（fl32 舍入残差，|值| 很小，fp32 精确）
    float lastNCountLow = 0;
    float lastNMinusOneLow = 0;

    uint32_t usedCoreNum = 0;
    uint32_t blockIdx = 0;
    bool useRunningMeanVar = true;

    static constexpr uint32_t VL_F32 = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = 32;
    static constexpr int64_t T_BLOCK_ALIGN_SIZE = BLOCK_SIZE / sizeof(T1);
    static constexpr int64_t FP32_BLOCK_ALIGN_SIZE = BLOCK_SIZE / sizeof(float);
    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr int64_t SCALE_COEF_FOUR = 4;
    static constexpr uint32_t ROW_TWO_OFFSET = 2;
    static constexpr uint32_t ROW_THREE_OFFSET = 3;
    static constexpr uint32_t ROW_FOUR_OFFSET = 4;

    constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN,
    };

    constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT,
    };

    /* ascendc variable */
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> gammaQueue;
    TQue<QuePosition::VECIN, 1> betaQueue;
    TQue<QuePosition::VECIN, 1> runningMeanInQueue;
    TQue<QuePosition::VECIN, 1> runningVarInQueue;

    TQue<QuePosition::VECOUT, 1> yQueue;
    TQue<QuePosition::VECOUT, 1> batchMeanQueue;
    TQue<QuePosition::VECOUT, 1> batchMeanLowQueue;
    TQue<QuePosition::VECOUT, 1> batchVarLowQueue;
    TQue<QuePosition::VECOUT, 1> batchVarQueue;
    TQue<QuePosition::VECOUT, 1> runningMeanOutQueue;
    TQue<QuePosition::VECOUT, 1> runningVarOutQueue;

    TBuf<TPosition::VECCALC> tmpTbuf1;
    TBuf<TPosition::VECCALC> tmpTbuf2;
    TBuf<TPosition::VECCALC> tmpTbuf3;
    TBuf<TPosition::VECCALC> tmpTbuf4;
    TBuf<TPosition::VECCALC> countTbuf1;
    TBuf<TPosition::VECCALC> countTbuf2;
    // 第二阶段常驻：所有核的 mean/var 低位肢 + 末级有偏方差低位肢
    TBuf<TPosition::VECCALC> allMeanLowTbuf;
    TBuf<TPosition::VECCALC> allVarLowTbuf;
    TBuf<TPosition::VECCALC> varLowTbuf;
};
} // namespace BatchNormExt2Ops

#endif // NORM_BATCH_NORM_EXT2_BLOCK_SPLIT_R_H
