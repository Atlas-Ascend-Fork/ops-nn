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
 * \file gru_grad_cube.h
 * \brief AIC 侧裸路径数据通路：GM(ND) -> L1(NZ) -> L0A/L0B -> MMAD -> L0C -> Fixpipe -> GM(fp32)。
 *
 * 硬件布局事实（dav_3510）：
 *  - L0A=NZ / L0B=ZN / L0C=NZ 是每操作数唯一原生布局（16 组合穷举唯一解），SplitA 直拷、SplitB 必转置。
 *  - C0 = 32/sizeof(T) 个元素（32B 恒定），fp16 16x16 / fp32 16x8；fp32 的 L1->L0B 转置一次 repeat 吃
 *    两个源分形，参数与 fp16 不同，不得合并。
 *  - Mmad 的 m 有下限 2（m=1 静默出错，MmadM 钳位）；k 恰好按值采用，pad 通道不贡献（K 尾免费）。
 *  - Mmad 的 mp.n 必须与 L0C tile 的 N 对齐一致（对齐到 CUBE_BLOCK=16 元素）。
 *  - 本算子为 GM 解耦形态：Fixpipe 直落 GM（CO2Layout::ROW_MAJOR, isToUB=false），不做 L0C->UB 融合，
 *    因此不需要 SSBUF 宏对；C 恒 fp32（输出 dtype 回投由 AIV ConvertOutputs 承担）。
 */
#ifndef _GRU_GRAD_ARCH35_CUBE_H_
#define _GRU_GRAD_ARCH35_CUBE_H_

#include "kernel_operator.h"

namespace ggcube {

constexpr uint32_t CUBE_BLOCK = 16;                       // M 侧分形边长（元素，不随类型变）
constexpr uint32_t C0_BYTES = 32;                         // 分形 C0 字节数恒为 32；C0(元素) = 32/sizeof(T)
constexpr uint32_t FRACT_ELEMS = CUBE_BLOCK * CUBE_BLOCK; // 单分形元素数（16x16，L0A/L0B 足迹按 kFrac*nRep 计）
constexpr uint32_t MMAD_M_MIN = 2;                        // Mmad 的 mp.m 硬件下限（m=1 静默出错）
constexpr uint32_t L1_BUDGET = 448 * 1024;                // L1 512KB，留 64KB 余量给 A1 常驻段与对齐垫
constexpr uint32_t L0A_BYTES = 64 * 1024;                 // L0A 物理容量（真机实测）
constexpr uint32_t L0B_BYTES = 64 * 1024; // L0B 物理容量（同上；溢出无报错——错数/挂死）
constexpr uint32_t L0C_ELEMS = 64 * 1024; // L0C 256KB / 4B

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
__aicore__ inline uint32_t CeilAlign(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }

template <typename T>
__aicore__ inline constexpr uint32_t C0()
{
    return C0_BYTES / sizeof(T);
}

// mp.m 下限 MMAD_M_MIN：m=1 静默出错
__aicore__ inline uint32_t MmadM(uint32_t m) { return (m < MMAD_M_MIN) ? MMAD_M_MIN : m; }

// ------------------------------------------------------------------------------------------------
// GM(ND) -> L1(NZ)。ndNum=1 的标准 ND2NZ；srcDValue = GM 行距（元素），dstNzC0Stride = 父 tile 对齐行数。
//
// -----------------------------------------------------------------------------------------------
template <typename T>
__aicore__ inline void CopyInNd2Nz(const AscendC::LocalTensor<T>& l1, const AscendC::GlobalTensor<T>& gm, uint32_t rows,
                                   uint32_t cols, uint32_t srcRowStride, uint32_t parentRows)
{
    AscendC::Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = rows;
    p.dValue = cols;
    p.srcNdMatrixStride = 0;
    p.srcDValue = srcRowStride;
    p.dstNzC0Stride = CeilAlign(parentRows, CUBE_BLOCK);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 0;
    AscendC::DataCopy(l1, gm, p);
}

// ------------------------------------------------------------------------------------------------
// L1(NZ) -> L0A(NZ)：直拷（两侧同为 NZ，不得"对称地"加转置——16 组合穷举实测）。
// ⚠ mFrac 必须是 tile 的分形行数 CeilDiv(rows, 16)（mFrac =
//   CeilAlign(parentRows,16)/16），且 mStep/srcStride/dstStride 与 Mmad 的 m 四者同源一致。
//   注意：mFrac 需按 CeilDiv(rows, 16) 计算，直接 rows/16 截断在 mSub<16 时 mFrac=0 会使 LoadData 变 no-op，
//   Mmad 乘到 L0A 上一次遗留内容（"L0A survives between launches"，跨 launch/进程确定性
//   复现幽灵值）；mSub=24 时 mFrac=1 只装载 16 行，Mmad 读的尾 8 行同为陈旧数据。
//   rows 为奇数/非 16 倍数时，L1 分形内的 pad 行本就不被 Mmad(m=rows) 消费，无需清零。
// -----------------------------------------------------------------------------------------------
template <typename T>
__aicore__ inline void SplitA(const AscendC::LocalTensor<T>& l0a, const AscendC::LocalTensor<T>& l1, uint32_t rows,
                              uint32_t k)
{
    const uint32_t mFrac = CeilDiv(rows, CUBE_BLOCK);
    AscendC::LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = mFrac;
    p.kStep = CeilDiv(k, C0<T>());
    p.srcStride = mFrac;
    p.dstStride = mFrac;
    p.ifTranspose = false;
    p.sid = 0;
    AscendC::LoadData(l0a, l1, p);
}

// ------------------------------------------------------------------------------------------------
// L1(NZ) -> L0B(ZN)：必须转置。fp16 与 fp32 走不同指令形态，不得统一（真机实测）。
// 入参 l1 为 B^T 视角?? 否——l1 是 B 的自然 [k, n] ND 经 ND2NZ 的 NZ 像（k 行 n 列）。
// -----------------------------------------------------------------------------------------------
template <typename T>
__aicore__ inline void SplitB(const AscendC::LocalTensor<T>& l0b, const AscendC::LocalTensor<T>& l1, uint32_t k,
                              uint32_t n)
{
    const uint32_t kFrac = CeilDiv(k, CUBE_BLOCK);
    const uint32_t nRep = CeilDiv(n, CUBE_BLOCK);
    const uint32_t dstOffset = nRep * CUBE_BLOCK * CUBE_BLOCK;
    const uint32_t srcOffset = CUBE_BLOCK * C0<T>(); // 一个分形，任何 T 恒 512B
    if constexpr (sizeof(T) == 4) {
        // fp32：一次转置粒度 16x16 元素 = 两个 16x8 分形，需 LoadData2dTransposeParamsV2 的分形间隔
        AscendC::LoadData2dTransposeParamsV2 p;
        p.startIndex = 0;
        p.repeatTimes = (uint8_t)nRep;
        constexpr uint32_t FP32_UNITS_PER_TRANS_REPEAT = 2; // 一个 repeat 吃两个 512B 单元（两个 n-block 分形）
        p.srcStride = FP32_UNITS_PER_TRANS_REPEAT * kFrac; // 跨步两个 n-block 分形
        p.dstGap = 0;
        p.dstFracGap = nRep - 1;
        p.srcFracGap = kFrac - 1;
        p.addrMode = 0;
        for (uint32_t i = 0; i < kFrac; ++i) {
            AscendC::LoadDataWithTranspose(l0b[i * dstOffset], l1[i * srcOffset], p);
        }
    } else {
        AscendC::LoadData2DParams p;
        p.startIndex = 0;
        p.repeatTimes = (uint8_t)nRep;
        p.srcStride = kFrac;
        p.dstGap = 0;
        p.ifTranspose = true;
        p.addrMode = 0;
        for (uint32_t i = 0; i < kFrac; ++i) {
            AscendC::LoadData(l0b[i * dstOffset], l1[i * srcOffset], p);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// SplitBV2——L1(NZ of [k, n]) → L0B(ZN) 单次调用转置。
// K_STEP 对齐基 2：fp32 每个分形占 2 个 c0
// LoadDataToL0B(ABLayout::KN): LoadData2DParamsV2.ifTranspose=true, mStep 走 K 分形,
// kStep 走 N 的 32B 块, srcStride/dstStride 按分形间隔——单调用替代 SplitB 的 kFrac
// 循环(48 次/步→12 次/步)。源 NZ 布局两仓同构(Nd2Nz dstNzC0Stride=CeilAlign(rows,16))。
// fp32 kStep 需 CeilAlign(kStep, 2)。
// ------------------------------------------------------------------------------------------------
template <typename T>
__aicore__ inline void SplitBV2(const AscendC::LocalTensor<T>& l0b, const AscendC::LocalTensor<T>& l1, uint32_t k,
                                uint32_t n, uint32_t singleK)
{
    const uint32_t kFrac = CeilDiv(k, CUBE_BLOCK); // K 向 16 元素分形数
    uint32_t nBlocks = CeilDiv(n, C0<T>());        // N 向 32B 块数（fp32 每块 8 元素 / fp16 每块 16 元素）
    constexpr uint32_t KSTEP_BLOCK_ALIGN = 2; // kStep 2 块对齐（16x16 元素粒度；两 dtype 共用本路径）
    nBlocks = CeilAlign(nBlocks, KSTEP_BLOCK_ALIGN);
    AscendC::LoadData2DParamsV2 p;
    p.mStartPosition = 0;
    p.kStartPosition = 0;
    p.mStep = static_cast<uint16_t>(kFrac);
    p.kStep = static_cast<uint16_t>(nBlocks);
    // srcStride: 源 NZ 像(SplitB 同源布局)的分形寻址为 addr(kfrac, c0) = (kfrac + kFrac*c0)*512B
    // —— c0 块(N 的 32B 组)间间隔 = kFrac 个分形单位。源视为 [M=kLoad 行, K=n 列] 的转置装载,
    // 其"源 K 方向分形间隔"即 c0 方向间隔 = kFrac(
    // 其 singleK=行数 且 C0=16 时与此同值; fp32 C0=8 时按布局推导取 kFrac)。
    (void)singleK;
    p.srcStride = static_cast<int32_t>(kFrac);
    p.dstStride = static_cast<uint16_t>((n + CUBE_BLOCK - 1) / CUBE_BLOCK);
    p.ifTranspose = true;
    p.sid = 0;
    AscendC::LoadData(l0b, l1, p);
}

// ------------------------------------------------------------------------------------------------
// MMAD：K 块 0 用 MmadPlain（cmatrixInitVal=true，L0C 由乘积播种），其余用 MmadAccum（累加）。
// 三形仅差两个布尔域，选错不报编译错；本文件封装消除调用侧歧义。
// -----------------------------------------------------------------------------------------------
template <typename TC, typename TA, typename TB>
__aicore__ inline void MmadPlain(const AscendC::LocalTensor<TC>& l0c, const AscendC::LocalTensor<TA>& l0a,
                                 const AscendC::LocalTensor<TB>& l0b, uint32_t m, uint32_t n, uint32_t k)
{
    AscendC::MmadParams mp;
    mp.m = static_cast<uint16_t>(MmadM(m));
    mp.n = static_cast<uint16_t>(CeilAlign(n, CUBE_BLOCK));
    mp.k = static_cast<uint16_t>(k);
    mp.cmatrixSource = false;
    mp.cmatrixInitVal = true;
    AscendC::Mmad(l0c, l0a, l0b, mp);
}

template <typename TC, typename TA, typename TB>
__aicore__ inline void MmadAccum(const AscendC::LocalTensor<TC>& l0c, const AscendC::LocalTensor<TA>& l0a,
                                 const AscendC::LocalTensor<TB>& l0b, uint32_t m, uint32_t n, uint32_t k)
{
    AscendC::MmadParams mp;
    mp.m = static_cast<uint16_t>(MmadM(m));
    mp.n = static_cast<uint16_t>(CeilAlign(n, CUBE_BLOCK));
    mp.k = static_cast<uint16_t>(k);
    mp.cmatrixSource = false;
    mp.cmatrixInitVal = false;
    AscendC::Mmad(l0c, l0a, l0b, mp);
}

// ------------------------------------------------------------------------------------------------
// L0C(NZ) -> GM(ND, fp32)：Fixpipe ROW_MAJOR（isToUB=false 的 GM 形态）。
// srcStride = L0C tile 行数（16 对齐元素）；dstStride = GM C 行距（元素）；GM 解耦形态不做 M 拆分
// （dualDstCtl=0，不涉 DrainToUB 的 splitM 路径）。
// -----------------------------------------------------------------------------------------------
constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_GM = {AscendC::CO2Layout::ROW_MAJOR, false};

__aicore__ inline void FixpipeL0CToGm(const AscendC::GlobalTensor<float>& gm, const AscendC::LocalTensor<float>& l0c,
                                      uint32_t m, uint32_t n, uint32_t gmRowPitch)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp;
    fp.nSize = static_cast<uint16_t>(n);
    fp.mSize = static_cast<uint16_t>(m);
    fp.srcStride = static_cast<uint16_t>(CeilAlign(m, CUBE_BLOCK));
    fp.dstStride = gmRowPitch;
    fp.dualDstCtl = 0;
    fp.subBlockId = false;
    AscendC::Fixpipe<float, float, CFG_ROW_MAJOR_GM>(gm, l0c, fp);
}

} // namespace ggcube

#endif // _GRU_GRAD_ARCH35_CUBE_H_
