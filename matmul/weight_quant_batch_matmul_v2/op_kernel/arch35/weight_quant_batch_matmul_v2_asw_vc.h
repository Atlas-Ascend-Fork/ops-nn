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
 * \file weight_quant_batch_matmul_v2_asw_vc.h
 * \brief ASW-VC kernel（vector 反量化 + 内联手写指令版 mmad），运行于 1952 CV 耦合架构
 *        （MIXCORE，C:V=1:1，无 ASCEND_IS_AIV / ASCEND_IS_AIC 之分，V 与 C 流水同核并行）。
 *        场景：A16W4 pergroup（x fp16 ND，weight int4 FRACTAL_NZ，scale fp16，支持 A/B 转置，
 *        B 非转置时 scale 为 [ceil(k/gs), n]，转置时为 [n, ceil(k/gs)]，antiquantOffset 为可选输入，
 *        形状/dtype 与 scale 一致，按 (w+offset)*scale 反量化）。单核内数据流：
 *        V 路径: GM(int4 NZ + fp16 scale) -> UB -> 反量化(F16) -> MTE3 -> L1(TSCM) B 区
 *        C 路径: GM(x) -> L1(Nd2Nz) -> L0(LoadData2D) -> Mmad(F16xF16) -> L0C -> Fixpipe -> GM。
 *        输出支持 FP16（DEQF16）及 INT8（REQ8/VREQ8，quantScale 使用 uint64 编码）。
 *        A 侧流水 / L0C 乒乓 / bias / 尾块 / 多 batch 逻辑与 ASW kernel 一致；
 *        B 区"反量化写入(MTE3) <-> LoadData2D 读取(MTE1)"的就绪与复用由核内硬事件
 *        （MTE3_MTE1 / MTE1_MTE3）同步，参考 weight_nz_performance 的 CV 耦合写法。
 */

#pragma once

#include "weight_quant_batch_matmul_v2_asw_block.h"
#include "weight_quant_batch_matmul_v2_vf.h"
#include "../weight_quant_batch_matmul_v2_constant.h"
#include "weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "../tool.h"
#include "tool_arch35.h"

using AscendC::BLOCK_CUBE;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;
namespace Reg = AscendC::Reg;
namespace WeightQuantBatchMatmulV2::Arch35 {

// 反量化流水核内 HardEvent flag id（与 cube 侧 flag 分属不同事件类型，互不冲突）
constexpr uint16_t ASWVC_EVT_MTE2_V = 0;      // MTE2 搬运完成（瞬态 set+wait）
constexpr uint16_t ASWVC_EVT_V_MTE2_BUF0 = 1; // 低bit/scale 双 buffer 的 V 读完释放
constexpr uint16_t ASWVC_EVT_V_MTE2_BUF1 = 2;
constexpr uint16_t ASWVC_EVT_V_MTE3 = 3;         // V 写完 staging（瞬态 set+wait）
constexpr uint16_t ASWVC_EVT_MTE3_V_STAGING = 4; // 4/5：MTE3 搬完对应 staging buffer 后释放
// 单个 256B 向量覆盖 8 个 k 行 x 16 个 n 列
constexpr uint64_t ASWVC_K_ROWS_PER_VEC = 8;
// 每个 k 行 16 个 int4 = 8 字节
constexpr uint64_t ASWVC_BYTES_PER_KROW_INT4 = 8;
// int4 每 2 个元素占 1 字节
constexpr uint64_t ASWVC_INT4_ELEMS_PER_BYTE = 2;
// 向量指令作用域标记（CANN 头文件提供；本地兜底为普通作用域）
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

// dav-510r2 后端不支持位宽大于32的除法；K/N方向取值均小于 2^32，统一降位到uint32再除
__aicore__ inline uint32_t DivAsU32(uint64_t a, uint64_t b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return static_cast<uint32_t>(a) / static_cast<uint32_t>(b);
}

__aicore__ inline uint32_t CeilDivAsU32(uint64_t a, uint64_t b) { return DivAsU32(a + b - 1, b); }

template <typename xType, typename wType, bool hasAntiQuantOffset>
__aicore__ inline void ASWVCAntiQuantInt4NzPergroupVf(uint64_t n1Cnt, uint64_t kAlignCur, uint64_t kGmOffset,
                                                      uint64_t groupSize, uint64_t scaleRowPitch,
                                                      __local_mem__ int8_t* weightLowBitAddr,
                                                      __local_mem__ xType* scaleF16Addr,
                                                      __local_mem__ xType* offsetF16Addr,
                                                      __local_mem__ xType* weightF16Addr)
{
    using WRegT = typename RegTensorActualT<wType>::T;
    static constexpr Reg::CastTrait castTraitS4ToF16 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    constexpr uint16_t vecStepBytes = ASWVC_K_ROWS_PER_VEC * ASWVC_BYTES_PER_KROW_INT4;
    constexpr uint16_t vecStepElems = ASWVC_K_ROWS_PER_VEC * BLOCK_CUBE;
    const uint16_t n1Total = static_cast<uint16_t>(n1Cnt);
    const uint16_t kLen = static_cast<uint16_t>(kAlignCur);
    const uint16_t gsSize = static_cast<uint16_t>(groupSize);
    const uint16_t n1StepW = static_cast<uint16_t>(kLen * ASWVC_BYTES_PER_KROW_INT4);
    const uint16_t n1StepF = static_cast<uint16_t>(kLen * BLOCK_CUBE);
    const uint16_t firstEnd = static_cast<uint16_t>(
        Min(kAlignCur, (DivAsU32(kGmOffset, groupSize) + 1) * groupSize - kGmOffset));
    const uint16_t segCnt = (kLen <= firstEnd) ? static_cast<uint16_t>(1) :
                                                 static_cast<uint16_t>(1 + CeilDivAsU32(kLen - firstEnd, groupSize));

    __VEC_SCOPE__
    {
        Reg::RegTensor<xType> scaleVreg;
        Reg::RegTensor<xType> offsetVreg;
        Reg::RegTensor<WRegT> wS4Vreg;
        Reg::RegTensor<xType> wF16Vreg;
        Reg::MaskReg maskAll = Reg::CreateMask<xType, Reg::MaskPattern::ALL>();
        for (uint16_t n1Idx = 0; n1Idx < n1Total; ++n1Idx) {
            __local_mem__ int8_t* wBase = weightLowBitAddr + n1Idx * n1StepW;
            __local_mem__ xType* fBase = weightF16Addr + n1Idx * n1StepF;
            __local_mem__ xType* scaleBase = scaleF16Addr + n1Idx * BLOCK_CUBE;
            __local_mem__ xType* offsetBase = offsetF16Addr + n1Idx * BLOCK_CUBE;
            for (uint16_t s = 0; s < segCnt; ++s) {
                const uint16_t segStart = (s == 0) ? static_cast<uint16_t>(0) :
                                                     static_cast<uint16_t>(firstEnd + (s - 1) * gsSize);
                const uint16_t segBound = static_cast<uint16_t>(firstEnd + s * gsSize);
                const uint16_t segEnd = (segBound < kLen) ? segBound : kLen;
                // DIST_BLK：读取 16 个 n 的该 group scale，广播到 256B 向量
                Reg::DataCopy<xType, Reg::LoadDist::DIST_BLK>(scaleVreg, scaleBase + s * scaleRowPitch);
                if constexpr (hasAntiQuantOffset) {
                    // offset 与 scale 同为 pergroup 行段 [G, N]，按相同方式广播
                    Reg::DataCopy<xType, Reg::LoadDist::DIST_BLK>(offsetVreg, offsetBase + s * scaleRowPitch);
                }
                const uint16_t chainCnt = static_cast<uint16_t>(CeilDivAsU32(segEnd - segStart, ASWVC_K_ROWS_PER_VEC));
                __local_mem__ int8_t* wSeg = wBase + segStart * ASWVC_BYTES_PER_KROW_INT4;
                __local_mem__ xType* fSeg = fBase + segStart * BLOCK_CUBE;
                for (uint16_t c = 0; c < chainCnt; ++c) {
                    // 128 个 int4（8 k行 x 16 n列）= 64 字节连续
                    Reg::DataCopy<WRegT, Reg::LoadDist::DIST_UNPACK4_B8>(
                        wS4Vreg, (__local_mem__ WRegT*)(wSeg + c * vecStepBytes));
                    Reg::Cast<xType, WRegT, castTraitS4ToF16>(wF16Vreg, wS4Vreg, maskAll);
                    if constexpr (hasAntiQuantOffset) {
                        // ANTIQUANT(weight) = (weight + offset) * scale
                        Reg::Add(wF16Vreg, wF16Vreg, offsetVreg, maskAll);
                    }
                    Reg::Mul(wF16Vreg, wF16Vreg, scaleVreg, maskAll);
                    // NZ 布局写 staging：n1 块外层（每块 kAlignCur*16 个元素）、k 行主序、n0 内层
                    Reg::DataCopy<xType, Reg::StoreDist::DIST_NORM_B16>(fSeg + c * vecStepElems, wF16Vreg, maskAll);
                }
            }
        }
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset>
__aicore__ inline void ASWVCAntiQuantInt4NzPergroupTransVf(uint64_t n1Cnt, uint64_t kAlignCur, uint64_t kGmOffset,
                                                           uint64_t groupSize, uint64_t scaleRowPitch,
                                                           __local_mem__ int8_t* weightLowBitAddr,
                                                           __local_mem__ xType* scaleF16Addr,
                                                           __local_mem__ xType* offsetF16Addr,
                                                           __local_mem__ xType* weightF16Addr)
{
    using WRegT = typename RegTensorActualT<wType>::T;
    static constexpr Reg::CastTrait castTraitS4ToF16 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    constexpr uint16_t vecN = BLOCK_CUBE / 2;                // 每个向量广播半个 n0（8 个 n）
    constexpr uint16_t halfFractalElems = vecN * BLOCK_CUBE; // 8n x 16k = 128 个 int4
    constexpr uint16_t halfFractalBytes = halfFractalElems / ASWVC_INT4_ELEMS_PER_BYTE;
    constexpr uint16_t halfCnt = BLOCK_CUBE / vecN; // 每个 16x16 分形含 2 个半区
    constexpr uint16_t fractalBytes = halfCnt * halfFractalBytes;
    constexpr uint16_t fractalElems = halfCnt * halfFractalElems;
    const uint16_t n1Total = static_cast<uint16_t>(n1Cnt);
    const uint16_t k1Total = static_cast<uint16_t>(kAlignCur / BLOCK_CUBE);
    const uint32_t gLo = DivAsU32(kGmOffset, groupSize);

    __VEC_SCOPE__
    {
        Reg::RegTensor<xType> scaleVreg;
        Reg::RegTensor<xType> offsetVreg;
        Reg::RegTensor<WRegT> wS4Vreg;
        Reg::RegTensor<xType> wF16Vreg;
        Reg::MaskReg maskAll = Reg::CreateMask<xType, Reg::MaskPattern::ALL>();
        for (uint16_t k1Idx = 0; k1Idx < k1Total; ++k1Idx) {
            const uint16_t scaleRow = static_cast<uint16_t>(DivAsU32(kGmOffset + k1Idx * BLOCK_CUBE, groupSize) - gLo);
            for (uint16_t n1Idx = 0; n1Idx < n1Total; ++n1Idx) {
                for (uint16_t halfIdx = 0; halfIdx < halfCnt; ++halfIdx) {
                    const uint16_t nOffset = n1Idx * BLOCK_CUBE + halfIdx * vecN;
                    // 8 个 N 的 scale 各广播到连续 16 个 K lane，与转置 NZ 的 (n0, k0) 排布一致
                    Reg::DataCopy<xType, Reg::LoadDist::DIST_E2B_B16>(
                        scaleVreg, scaleF16Addr + scaleRow * scaleRowPitch + nOffset);
                    if constexpr (hasAntiQuantOffset) {
                        Reg::DataCopy<xType, Reg::LoadDist::DIST_E2B_B16>(
                            offsetVreg, offsetF16Addr + scaleRow * scaleRowPitch + nOffset);
                    }
                    const uint32_t fractalIdx = k1Idx * n1Total + n1Idx;
                    Reg::DataCopy<WRegT, Reg::LoadDist::DIST_UNPACK4_B8>(
                        wS4Vreg, (__local_mem__ WRegT*)(weightLowBitAddr + fractalIdx * fractalBytes +
                                                        halfIdx * halfFractalBytes));
                    Reg::Cast<xType, WRegT, castTraitS4ToF16>(wF16Vreg, wS4Vreg, maskAll);
                    if constexpr (hasAntiQuantOffset) {
                        Reg::Add(wF16Vreg, wF16Vreg, offsetVreg, maskAll);
                    }
                    Reg::Mul(wF16Vreg, wF16Vreg, scaleVreg, maskAll);
                    Reg::DataCopy<xType, Reg::StoreDist::DIST_NORM_B16>(
                        weightF16Addr + fractalIdx * fractalElems + halfIdx * halfFractalElems, wF16Vreg, maskAll);
                }
            }
        }
    }
}

#define LOCAL_TEMPLATE_CLASS_PARAMS                                                                               \
    template <typename xType, typename wType, typename scaleType, typename biasType, typename yType, bool aTrans, \
              bool bTrans, QuantType antiQuantType, bool hasAntiQuantOffset, QuantType quantType>
#define LOCAL_TEMPLATE_FUNC_PARAMS \
    xType, wType, scaleType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType

using VCTiling = wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWVCTilingDataParams;

LOCAL_TEMPLATE_CLASS_PARAMS
class WeightQuantBatchMatmulV2ASWVCKernel {
    constexpr static uint64_t C0_SIZE_A = AscendC::AuxGetC0Size<xType>();
    // B 侧 L1/L0 中为反量化后的 fp16（dequantType），C0 按 16 取
    using dequantType = xType;
    constexpr static uint64_t C0_SIZE_DEQ = AscendC::AuxGetC0Size<dequantType>();
    constexpr static uint64_t BLOCK_CUBE_SIZE = AscendC::BLOCK_CUBE;
    // L0A/L0B开启双缓冲，L0C按tiling l0cDB使能乒乓双缓冲
    constexpr static uint64_t HALF_L0A_ELEMS = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_NUM / sizeof(xType);
    constexpr static uint64_t HALF_L0B_ELEMS = AscendC::TOTAL_L0B_SIZE / DOUBLE_BUFFER_NUM / sizeof(dequantType);
    // L0C乒乓时单个半区的元素数（int32累加）
    constexpr static uint64_t HALF_L0C_ELEMS = AscendC::TOTAL_L0C_SIZE / 2 / sizeof(int32_t);
    // CO1(L0C) tensor 寻址跨度（取值与 ASW 模板一致，覆盖乒乓半区偏移）
    constexpr static uint64_t CO1_SPAN_BYTES = 4UL * 1024UL * 1024UL;
    // C2 bias 表空间
    constexpr static uint64_t BIAS_TABLE_SIZE = 4UL * 1024UL;
    // 硬事件flag id（核内本地，不同事件类型的 id 空间相互独立）
    constexpr static uint16_t L1_MTE1_MTE2_FLAG = 0;   // 0~3: A的L1多buffer
    constexpr static uint16_t L1_MTE2_MTE1_FLAG = 0;   // 0~3: A的L1多buffer
    constexpr static uint16_t BIAS_MTE1_MTE2_FLAG = 4; // 避让L1 buffer的0~3
    constexpr static uint16_t L0_M_MTE1_FLAG = 3;      // 2/3: L0双缓冲
    constexpr static uint16_t L0_MTE1_M_FLAG = 2;      // 2/3: L0双缓冲
    constexpr static uint16_t L0C_FIX_M_FLAG = 0;      // 0/1: L0C乒乓双缓冲
    constexpr static uint16_t L0C_M_FIX_FLAG = 0;      // 0/1: L0C乒乓双缓冲
    constexpr static uint16_t QUANT_SCALE_FIX_MTE2_FLAG = 0;
    constexpr static uint16_t QUANT_SCALE_MTE2_FIX_FLAG = 0;
    // bias BT(C2)单缓冲复用，等上一tile读取bias的Mmad完成后再覆写，避让L0的3/4号flag
    constexpr static uint16_t BIAS_BT_M_MTE1_FLAG = 5;
    // B 区（本核 V 流水反量化写入）核内同步 flag：MTE3 写完可读 / MTE1 读完可覆写
    constexpr static uint16_t B_L1_MTE3_MTE1_FLAG = 0; // 瞬态 set+wait
    constexpr static uint16_t B_L1_MTE1_MTE3_FLAG = 0; // 0~3: B的L1多buffer
    // 多buffer flag序号（0~3级buffer对应的偏移）
    constexpr static uint16_t FIRST_FLAG = 1;
    constexpr static uint16_t SECOND_FLAG = 2;
    constexpr static uint16_t THIRD_FLAG = 3;
    // unitflag状态：3=最后一次累加，2=非最后一次累加，硬件自动完成M与FIX的同步
    constexpr static uint32_t FINAL_ACCUMULATION = 3;
    constexpr static uint32_t NON_FINAL_ACCUMULATION = 2;
#if __NPU_ARCH__ == 5102
    // 本kernel为A16W16路径（反量化后 mmad 为 F16xF16），定点位宽取58
    constexpr static uint8_t FIX_SHIFT_VAL_LEN_A16W16 = 58;
    constexpr static int32_t FIX_SHIFT_VAL_MIN_F16XF16 = 34;
    constexpr static int32_t FIX_SHIFT_VAL_MAX_F16XF16 = 43;
#endif

public:
    __aicore__ inline WeightQuantBatchMatmulV2ASWVCKernel()
    {
        // 预设"资源空闲"flag，使首个WaitFlag直接通过；L1最多4buffer，flag 0~3全部预设
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + FIRST_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + SECOND_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + THIRD_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_MTE1_MTE2_FLAG);
        // B 区 L1 buffer（V 流水反量化写入）复用 flag 同样预设
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(B_L1_MTE1_MTE3_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(B_L1_MTE1_MTE3_FLAG + FIRST_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(B_L1_MTE1_MTE3_FLAG + SECOND_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(B_L1_MTE1_MTE3_FLAG + THIRD_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG + 1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(BIAS_BT_M_MTE1_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_M_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_M_FLAG + 1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(QUANT_SCALE_FIX_MTE2_FLAG);
    }

    __aicore__ inline ~WeightQuantBatchMatmulV2ASWVCKernel()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + FIRST_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + SECOND_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + THIRD_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_MTE1_MTE2_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(B_L1_MTE1_MTE3_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(B_L1_MTE1_MTE3_FLAG + FIRST_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(B_L1_MTE1_MTE3_FLAG + SECOND_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(B_L1_MTE1_MTE3_FLAG + THIRD_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG + 1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(BIAS_BT_M_MTE1_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_M_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_M_FLAG + 1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(QUANT_SCALE_FIX_MTE2_FLAG);
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const void* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitMmad();
    __aicore__ inline void SetSingleShape(int32_t singleM, int32_t singleN, int32_t singleK);
    __aicore__ inline void CopyInA1(uint64_t kGmOffset, uint64_t curK, uint64_t l1BufId);
    __aicore__ inline void CopyInBiasL1();
    __aicore__ inline void CopyInA2(uint64_t kL1Offset, uint64_t curK, uint64_t curK0, uint64_t l0BufId);
    __aicore__ inline void CopyInB2(uint64_t kL1Offset, uint64_t curK, uint64_t curK0, uint64_t l0BufId);
    __aicore__ inline void CopyInBiasBt();
    __aicore__ inline void CopyQuantScaleL1();
    __aicore__ inline void MmadCompute(uint64_t curK0, uint64_t l0BufId, uint64_t l0cBufId, bool needBias,
                                       bool needCInit, bool isFinalAccum);
    __aicore__ inline void FixpipeOut(const AscendC::GlobalTensor<yType>& gm, uint64_t l0cBufId);
    __aicore__ inline void Iterate();
    template <bool sync = true>
    __aicore__ inline void GetTensorC(const AscendC::GlobalTensor<yType>& gm);
    __aicore__ inline void ComputeBasicOptiLoop();
    __aicore__ inline void ComputeBmmOptiLoop();
    __aicore__ inline void SetMMParaAndCompute();
    // 产出一个 L1 chunk：nOffset 为本 tile 的 n 起始（16 对齐），singleN 为条带宽度
    __aicore__ inline void DequantChunk(const LocalTensor<xType>& bL1, uint64_t bL1BufElems, uint64_t l1BufId,
                                        uint64_t kGmOffset, uint64_t curK, uint64_t nOffset, uint64_t singleN,
                                        uint64_t batchBOffset);
    __aicore__ inline void CopyWeightNzGmToUb(uint64_t kGmOffset, uint64_t kAlignCur, uint64_t nStart, uint64_t nLen,
                                              uint64_t batchBOffset);
    // scale/offset 共用搬运逻辑：非转置为 [G,N]，转置为 [N,G] 并转换为 UB group-major 布局
    __aicore__ inline void CopyScaleOffsetGmToUb(const GlobalTensor<scaleType>& paramGlobal,
                                                 const LocalTensor<scaleType>& ubParam, uint64_t kGmOffset,
                                                 uint64_t curK, uint64_t nStart, uint64_t nLen, uint64_t batchBOffset);

    uint32_t blockIdx_;
    const VCTiling* tiling_ = nullptr;
    WeightQuantBmmAswBlock<VCTiling> block_;
    GlobalTensor<xType> aGlobal_;
    GlobalTensor<wType> wGlobal_;
    GlobalTensor<scaleType> scaleGlobal_;
    GlobalTensor<scaleType> offsetGlobal_;
    GlobalTensor<uint64_t> quantScaleGlobal_;
    GlobalTensor<yType> cGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<xType> aTileGlobal_;
    GlobalTensor<biasType> biasTileGlobal_;
    GlobalTensor<uint64_t> quantScaleTileGlobal_;

    uint64_t singleM_ = 1;
    uint64_t singleN_ = 1;
    uint64_t singleK_ = 1;
    uint64_t baseM_ = 1;
    uint64_t baseN_ = 1;
    uint64_t baseK_ = 1;
    uint64_t kL1_ = 1; // A/B在L1上单buffer的K方向长度
    uint64_t groupSize_ = 1;
    uint64_t nSize_ = 1;
    uint64_t kAlignTotal_ = 16;     // 按输入 k0 对齐的 K 大小，NZ GM 地址步长
    uint64_t nAlignTotal_ = 16;     // CeilAlign(n, 16)，NZ GM 的 n1 维块数
    uint64_t nMax_ = 16;            // 反量化输出的最大 n 宽度（16 对齐），用于 UB 定容
    uint64_t scaleSingleElems_ = 0; // scale 单 buffer 元素数
    uint64_t vecLoopCnt_ = 0;       // chunk 计数，驱动 UB buffer 轮转
    uint64_t l1BufNum_ = 2;         // L1 buffer数量（2或4），来自tiling l1BufferNum
    uint64_t aL1BufElems_ = 0;      // A单个L1 buffer的元素数
    uint64_t bL1BufElems_ = 0;      // B单个L1 buffer的元素数（反量化后fp16，NZ布局）
    uint64_t bL1ByteOffset_ = 0;    // B在L1上的字节偏移
    uint64_t biasL1ByteOffset_ = 0;
    uint64_t quantScaleL1ByteOffset_ = 0;
    uint64_t quantScalar_ = 0;
    uint64_t l1LoopCnt_ = 0;
    uint64_t l0LoopCnt_ = 0;
    uint64_t l0cPingPong_ = 0; // L0C乒乓计数，每完成一个tile的搬出自增
    bool enableL0cDb_ = false; // L0C双缓冲使能，来自tiling l0cDB
    bool hasBias_ = false;
#if __NPU_ARCH__ == 5102
    uint8_t shiftValue_ = 42;
#endif

    // L1 布局统一使用 TSCM 寻址（V 流水反量化的 MTE3 写与 Nd2Nz/LoadData2D 共享同一物理 L1，
    // 与 reg_base 模板一致）
    AscendC::LocalTensor<xType> aL1_;
    AscendC::LocalTensor<dequantType> bL1_;
    AscendC::LocalTensor<biasType> biasL1_;
    AscendC::LocalTensor<uint64_t> quantScaleL1_;
    LocalTensor<int8_t> ubWeightLowBit_; // int4 原文（NZ 块布局），双 buffer
    LocalTensor<scaleType> ubScale_;     // scale 行段，双 buffer
    LocalTensor<scaleType> ubOffset_;    // offset 行段（hasAntiQuantOffset 时分配），双 buffer
    LocalTensor<xType> ubWeightF16_;     // 反量化结果 staging（紧凑 NZ），双 buffer
    AscendC::LocalTensor<xType> aL0_;
    AscendC::LocalTensor<dequantType> bL0_;
    AscendC::LocalTensor<int32_t> cL0_;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    AscendC::LocalTensor<int32_t> biasBt_;
#else
    AscendC::LocalTensor<biasType> biasBt_;
#endif
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyWeightNzGmToUb(
    uint64_t kGmOffset, uint64_t kAlignCur, uint64_t nStart, uint64_t nLen, uint64_t batchBOffset)
{
    DataCopyExtParams intriParams;
    intriParams.dstStride = 0;
    DataCopyPadExtParams<wType> padParams;
    uint64_t gmOffset = 0;
    if constexpr (bTrans) {
        // GM NZ (k1, n1, n0, k0)：输入 k0 固定为 16，UB 为紧凑 16x16 分形
        const uint64_t batchOffset = batchBOffset * nAlignTotal_ * kAlignTotal_;
        intriParams.blockCount = 1;
        intriParams.blockLen = static_cast<uint32_t>(nLen * ASWVC_BYTES_PER_KROW_INT4);
        intriParams.srcStride = 0;
        for (uint64_t k1Idx = 0; k1Idx < kAlignCur / BLOCK_CUBE; ++k1Idx) {
            gmOffset = batchOffset + (kGmOffset + k1Idx * BLOCK_CUBE) * nAlignTotal_ + nStart * BLOCK_CUBE;
            const uint64_t ubOffset = (vecLoopCnt_ & 1) * (nMax_ * kL1_ / ASWVC_INT4_ELEMS_PER_BYTE) +
                                      k1Idx * nLen * ASWVC_BYTES_PER_KROW_INT4;
            DataCopyPad(ubWeightLowBit_[ubOffset].template ReinterpretCast<wType>(), wGlobal_[gmOffset], intriParams,
                        padParams);
        }
        return;
    } else {
        // GM NZ (n1, k1, k0, n0)：固定 n1 后 K 条带连续，每 n1 搬 kAlignCur*8 字节
        intriParams.blockCount = static_cast<uint16_t>(CeilDiv(nLen, static_cast<uint64_t>(BLOCK_CUBE)));
        intriParams.blockLen = kAlignCur * ASWVC_BYTES_PER_KROW_INT4;
        intriParams.srcStride = (kAlignTotal_ - kAlignCur) * ASWVC_BYTES_PER_KROW_INT4;
        gmOffset = batchBOffset * nAlignTotal_ * kAlignTotal_ + nStart * kAlignTotal_ + kGmOffset * BLOCK_CUBE;
    }
    const uint64_t ubOffset = (vecLoopCnt_ & 1) * (nMax_ * kL1_ / ASWVC_INT4_ELEMS_PER_BYTE);
    DataCopyPad(ubWeightLowBit_[ubOffset].template ReinterpretCast<wType>(), wGlobal_[gmOffset], intriParams,
                padParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyScaleOffsetGmToUb(
    const GlobalTensor<scaleType>& paramGlobal, const LocalTensor<scaleType>& ubParam, uint64_t kGmOffset,
    uint64_t curK, uint64_t nStart, uint64_t nLen, uint64_t batchBOffset)
{
    const uint64_t gLo = DivAsU32(kGmOffset, groupSize_);
    const uint64_t groupCnt = CeilDivAsU32(kGmOffset + curK, groupSize_) - gLo;
    const uint64_t groupNumTotal = CeilDivAsU32(tiling_->k, groupSize_);
    const uint64_t scaleBatchOffset = batchBOffset * nSize_ * groupNumTotal;
    const uint64_t ubBufOffset = (vecLoopCnt_ & 1) * scaleSingleElems_;
    const uint64_t nValid = Min(nLen, nSize_ - nStart);
    DataCopyPadExtParams<scaleType> padParams;
    if constexpr (bTrans) {
        // GM 为 [N, G]，逐 group 将 N 向离散数据收拢为 UB 的 [groupCnt, nLen]
        const uint64_t gmBase = scaleBatchOffset + nStart * groupNumTotal + gLo;
        for (uint64_t groupIdx = 0; groupIdx < groupCnt; ++groupIdx) {
            const uint64_t ubRow = ubBufOffset + groupIdx * nLen;
            for (uint64_t n = 0; n < nValid; ++n) {
                ubParam.SetValue(ubRow + n, paramGlobal.GetValue(gmBase + n * groupNumTotal + groupIdx));
            }
        }
    } else {
        // GM 为 [G, N]，仅搬有效列，避免对齐后的 nLen 造成尾块越界
        DataCopyExtParams intriParams;
        intriParams.blockLen = nValid * sizeof(scaleType);
        intriParams.srcStride = (nSize_ - nValid) * sizeof(scaleType);
        intriParams.dstStride = 0;
        const uint64_t gmOffset = scaleBatchOffset + gLo * nSize_ + nStart;
        if (nValid == nLen) {
            intriParams.blockCount = static_cast<uint16_t>(groupCnt);
            DataCopyPad(ubParam[ubBufOffset], paramGlobal[gmOffset], intriParams, padParams);
        } else {
            intriParams.blockCount = 1;
            intriParams.srcStride = 0;
            for (uint64_t groupIdx = 0; groupIdx < groupCnt; ++groupIdx) {
                DataCopyPad(ubParam[ubBufOffset + groupIdx * nLen], paramGlobal[gmOffset + groupIdx * nSize_],
                            intriParams, padParams);
            }
        }
    }
    for (uint64_t groupIdx = 0; groupIdx < groupCnt; ++groupIdx) {
        const uint64_t ubRow = ubBufOffset + groupIdx * nLen;
        for (uint64_t n = nValid; n < nLen; ++n) {
            ubParam.SetValue(ubRow + n, static_cast<scaleType>(0));
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::DequantChunk(
    const LocalTensor<xType>& bL1, uint64_t bL1BufElems, uint64_t l1BufId, uint64_t kGmOffset, uint64_t curK,
    uint64_t nOffset, uint64_t singleN, uint64_t batchBOffset)
{
    // CV 耦合单核：本核唯一的 V 流水处理整条 n 条带；L1 B 区复用/就绪同步由调用方核内硬事件完成
    uint64_t nLen = CeilAlign(singleN, BLOCK_CUBE);
    uint64_t kAlignCur = CeilAlign(curK, BLOCK_CUBE);
    uint64_t n1Cnt = nLen / BLOCK_CUBE;
    uint64_t bufId = vecLoopCnt_ & 1;
    uint64_t stagingOffset = bufId * nMax_ * kL1_;

    if (vecLoopCnt_ >= DOUBLE_BUFFER_NUM) {
        WaitFlag<HardEvent::V_MTE2>(ASWVC_EVT_V_MTE2_BUF0 + bufId);
    }
    CopyWeightNzGmToUb(kGmOffset, kAlignCur, nOffset, nLen, batchBOffset);
    CopyScaleOffsetGmToUb(scaleGlobal_, ubScale_, kGmOffset, curK, nOffset, nLen, batchBOffset);
    if constexpr (hasAntiQuantOffset) {
        CopyScaleOffsetGmToUb(offsetGlobal_, ubOffset_, kGmOffset, curK, nOffset, nLen, batchBOffset);
    }
    AscendC::SetFlag<HardEvent::MTE2_V>(ASWVC_EVT_MTE2_V);
    AscendC::WaitFlag<HardEvent::MTE2_V>(ASWVC_EVT_MTE2_V);
    if (vecLoopCnt_ >= DOUBLE_BUFFER_NUM) {
        WaitFlag<HardEvent::MTE3_V>(ASWVC_EVT_MTE3_V_STAGING + bufId);
    }
    __local_mem__ xType* offsetF16Addr = nullptr;
    if constexpr (hasAntiQuantOffset) {
        offsetF16Addr = (__local_mem__ xType*)ubOffset_[bufId * scaleSingleElems_].GetPhyAddr();
    }
    if constexpr (bTrans) {
        ASWVCAntiQuantInt4NzPergroupTransVf<xType, wType, hasAntiQuantOffset>(
            n1Cnt, kAlignCur, kGmOffset, groupSize_, nLen,
            (__local_mem__ int8_t*)ubWeightLowBit_[bufId * (nMax_ * kL1_ / ASWVC_INT4_ELEMS_PER_BYTE)].GetPhyAddr(),
            (__local_mem__ xType*)ubScale_[bufId * scaleSingleElems_].GetPhyAddr(), offsetF16Addr,
            (__local_mem__ xType*)ubWeightF16_[stagingOffset].GetPhyAddr());
    } else {
        ASWVCAntiQuantInt4NzPergroupVf<xType, wType, hasAntiQuantOffset>(
            n1Cnt, kAlignCur, kGmOffset, groupSize_, nLen,
            (__local_mem__ int8_t*)ubWeightLowBit_[bufId * (nMax_ * kL1_ / ASWVC_INT4_ELEMS_PER_BYTE)].GetPhyAddr(),
            (__local_mem__ xType*)ubScale_[bufId * scaleSingleElems_].GetPhyAddr(), offsetF16Addr,
            (__local_mem__ xType*)ubWeightF16_[stagingOffset].GetPhyAddr());
    }
    // 避免 GM pad 数据参与 MMAD 累加
    if (kAlignCur > curK) {
        if constexpr (bTrans) {
            const uint64_t k1Idx = kAlignCur / BLOCK_CUBE - 1;
            const uint64_t validK = curK - k1Idx * BLOCK_CUBE;
            for (uint16_t n1Idx = 0; n1Idx < n1Cnt; ++n1Idx) {
                for (uint16_t n0Idx = 0; n0Idx < BLOCK_CUBE; ++n0Idx) {
                    const uint64_t padOffset = stagingOffset + (k1Idx * n1Cnt + n1Idx) * BLOCK_CUBE * BLOCK_CUBE +
                                               n0Idx * BLOCK_CUBE + validK;
                    AscendC::Duplicate(ubWeightF16_[padOffset], (xType)0, static_cast<int32_t>(BLOCK_CUBE - validK));
                }
            }
        } else {
            for (uint16_t n1Idx = 0; n1Idx < n1Cnt; ++n1Idx) {
                AscendC::Duplicate(ubWeightF16_[stagingOffset + n1Idx * kAlignCur * BLOCK_CUBE + curK * BLOCK_CUBE],
                                   (xType)0, static_cast<int32_t>((kAlignCur - curK) * BLOCK_CUBE));
            }
        }
    }
    AscendC::SetFlag<HardEvent::V_MTE2>(ASWVC_EVT_V_MTE2_BUF0 + bufId);
    AscendC::SetFlag<HardEvent::V_MTE3>(ASWVC_EVT_V_MTE3);
    AscendC::WaitFlag<HardEvent::V_MTE3>(ASWVC_EVT_V_MTE3);
    DataCopyParams mte3Params;
    mte3Params.blockCount = 1;
    mte3Params.blockLen = static_cast<uint16_t>((nLen * kAlignCur * sizeof(xType)) / ONE_BLK_SIZE);
    mte3Params.srcStride = 0;
    mte3Params.dstStride = 0;
    uint64_t l1Offset = l1BufId * bL1BufElems;
    DataCopy(bL1[l1Offset], ubWeightF16_[stagingOffset], mte3Params);
    AscendC::SetFlag<HardEvent::MTE3_V>(ASWVC_EVT_MTE3_V_STAGING + bufId);
    vecLoopCnt_++;
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, const void* tilingData)
{
    tiling_ = static_cast<const VCTiling*>(tilingData);
    // CV 耦合架构：cube/vector 同核，核号无需按 task ratio 折算
    blockIdx_ = GetBlockIdx();
    // V/C 路径共用的形状与 L1 布局参数（两侧必须完全一致）
    baseM_ = tiling_->baseM;
    baseN_ = tiling_->baseN;
    baseK_ = tiling_->baseK;
    uint64_t stepKa = static_cast<uint64_t>(tiling_->stepKa);
    uint64_t stepKb = static_cast<uint64_t>(tiling_->stepKb);
    kL1_ = (stepKa < stepKb ? stepKa : stepKb) * baseK_;
    l1BufNum_ = tiling_->l1BufferNum == QUADRUPLE_BUFFER_NUM ? QUADRUPLE_BUFFER_NUM : DOUBLE_BUFFER_NUM;
    aL1BufElems_ = baseM_ * kL1_;
    // B 为反量化后的 fp16 NZ 布局：转置时为 (k1,n1,n0,k0)，否则为 (n1,k1,k0,n0)
    bL1BufElems_ = bTrans ? baseN_ * CeilAlign(kL1_, C0_SIZE_DEQ) : CeilAlign(baseN_, C0_SIZE_DEQ) * kL1_;
    uint64_t aL1Bytes = l1BufNum_ * aL1BufElems_ * sizeof(xType);
    uint64_t bL1Bytes = l1BufNum_ * bL1BufElems_ * sizeof(dequantType);
    bL1ByteOffset_ = aL1Bytes;
    biasL1ByteOffset_ = aL1Bytes + bL1Bytes;
    // L1 布局统一使用 TSCM 寻址（V 流水反量化的 MTE3 写与 Nd2Nz/LoadData2D 共享同一物理 L1，
    // 与 reg_base 模板一致）
    aL1_ = AscendC::LocalTensor<xType>(AscendC::TPosition::TSCM, 0, aL1Bytes);
    bL1_ = AscendC::LocalTensor<dequantType>(AscendC::TPosition::TSCM, bL1ByteOffset_, bL1Bytes);
    block_.Init(tiling_, blockIdx_);
    // V 路径：反量化参数、GM 地址与 UB 布局
    groupSize_ = tiling_->groupSize;
    nSize_ = tiling_->n;
    kAlignTotal_ = CeilAlign(static_cast<uint64_t>(tiling_->k), BLOCK_CUBE);
    nAlignTotal_ = CeilAlign(nSize_, BLOCK_CUBE);
    // GM→UB 时为紧凑 16x16 int4 分形，输出统一为 FP16 k0=16
    nMax_ = CeilAlign(baseN_, BLOCK_CUBE);
    // 一个 chunk 覆盖的最多 group 行数（chunk 起点可能不在 group 边界，+1 兜底）
    scaleSingleElems_ = (CeilDivAsU32(kL1_, groupSize_) + 1) * nMax_;
    // UB 布局（字节）：[F16 staging x2][int4 低bit x2][scale x2][offset x2（hasAntiQuantOffset 时）]
    uint64_t offset = 0;
    uint64_t f16StageBytes = DOUBLE_BUFFER_NUM * nMax_ * kL1_ * sizeof(xType);
    ubWeightF16_ = LocalTensor<xType>(TPosition::LCM, offset, f16StageBytes);
    offset += f16StageBytes;
    uint64_t lowBitBytes = DOUBLE_BUFFER_NUM * (nMax_ * kL1_ / ASWVC_INT4_ELEMS_PER_BYTE);
    ubWeightLowBit_ = LocalTensor<int8_t>(TPosition::LCM, offset, lowBitBytes);
    offset += lowBitBytes;
    uint64_t scaleBytes = DOUBLE_BUFFER_NUM * scaleSingleElems_ * sizeof(scaleType);
    ubScale_ = LocalTensor<scaleType>(TPosition::LCM, offset, scaleBytes);
    offset += scaleBytes;
    if constexpr (hasAntiQuantOffset) {
        ubOffset_ = LocalTensor<scaleType>(TPosition::LCM, offset, scaleBytes);
    }
    vecLoopCnt_ = 0;
    wGlobal_.SetGlobalBuffer((__gm__ wType*)weight);
    scaleGlobal_.SetGlobalBuffer((__gm__ scaleType*)antiquantScale);
    if constexpr (hasAntiQuantOffset) {
        offsetGlobal_.SetGlobalBuffer((__gm__ scaleType*)antiquantOffset);
    }
    if constexpr (AscendC::IsSameType<yType, int8_t>::value) {
        quantScaleGlobal_.SetGlobalBuffer((__gm__ uint64_t*)quantScale);
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    if (tiling_->l2CacheDisable == wqbmmv2_tiling::L2CacheMode::ALL_L2_CACHE_DISABLE ||
        tiling_->l2CacheDisable == wqbmmv2_tiling::L2CacheMode::B_L2_CACHE_DISABLE) {
        wGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
#endif
    // C 路径：x/bias/y GM 地址与 mmad 资源
    aGlobal_.SetGlobalBuffer((__gm__ xType*)x);
    cGlobal_.SetGlobalBuffer((__gm__ yType*)y);
    if (static_cast<bool>(tiling_->isBias)) {
        biasGlobal_.SetGlobalBuffer((__gm__ biasType*)bias);
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    if (tiling_->l2CacheDisable == wqbmmv2_tiling::L2CacheMode::ALL_L2_CACHE_DISABLE ||
        tiling_->l2CacheDisable == wqbmmv2_tiling::L2CacheMode::A_L2_CACHE_DISABLE) {
        aGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
#endif
    InitMmad();
}

// mmad 的 L0/C2 buffer 布局与循环计数初始化（cube 侧）
LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitMmad()
{
    if (static_cast<bool>(tiling_->isBias)) {
        biasL1_ = AscendC::LocalTensor<biasType>(AscendC::TPosition::TSCM, biasL1ByteOffset_,
                                                 baseN_ * sizeof(biasType));
    }
    quantScaleL1ByteOffset_ = biasL1ByteOffset_ +
                              (static_cast<bool>(tiling_->isBias) ? baseN_ * sizeof(biasType) : 0UL);
    if constexpr (AscendC::IsSameType<yType, int8_t>::value && quantType == QuantType::PER_CHANNEL) {
        quantScaleL1_ = AscendC::LocalTensor<uint64_t>(AscendC::TPosition::TSCM, quantScaleL1ByteOffset_,
                                                       baseN_ * sizeof(uint64_t));
    }
    aL0_ = AscendC::LocalTensor<xType>(AscendC::TPosition::A2, 0, AscendC::TOTAL_L0A_SIZE);
    bL0_ = AscendC::LocalTensor<dequantType>(AscendC::TPosition::B2, 0, AscendC::TOTAL_L0B_SIZE);
    cL0_ = AscendC::LocalTensor<int32_t>(AscendC::TPosition::CO1, 0, CO1_SPAN_BYTES);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    biasBt_ = AscendC::LocalTensor<int32_t>(AscendC::TPosition::C2, 0, BIAS_TABLE_SIZE);
#else
    biasBt_ = AscendC::LocalTensor<biasType>(AscendC::TPosition::C2, 0, BIAS_TABLE_SIZE);
#endif
    l1LoopCnt_ = 0;
    l0LoopCnt_ = 0;
    l0cPingPong_ = 0;
    hasBias_ = false;
    // L0C双缓冲：tiling保证l0cDB==2时baseM*baseN*4B*2不超过L0C，此处再防御性校验半区容量
    enableL0cDb_ = tiling_->l0cDB >= DOUBLE_BUFFER_NUM && baseM_ * baseN_ <= HALF_L0C_ELEMS;
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if (tiling_->batchDimAll == 1UL) {
        block_.offset_.batchCOffset = 0;
        block_.offset_.batchAOffset = 0;
        block_.offset_.batchBOffset = 0;
        ComputeBasicOptiLoop();
    } else {
        ComputeBmmOptiLoop();
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeBasicOptiLoop()
{
    for (uint64_t roundIndex = 0; roundIndex < block_.params_.round; ++roundIndex) {
        block_.UpdateBasicIndex(roundIndex);
        block_.UpdateBlockParams(roundIndex);
        if (block_.params_.singleCoreM == 0 || block_.params_.singleCoreN == 0) {
            continue;
        }
        SetSingleShape(block_.params_.singleCoreM, block_.params_.singleCoreN, tiling_->k);
        block_.CalcGMOffset<aTrans, bTrans>();
        SetMMParaAndCompute();
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::ComputeBmmOptiLoop()
{
    uint64_t batchC3C4 = static_cast<uint64_t>(tiling_->batchC3) * tiling_->batchC4;
    uint64_t batchC2C3C4 = tiling_->batchC2 * batchC3C4;
    uint64_t batchB3B4 = static_cast<uint64_t>(tiling_->batchB3) * tiling_->batchB4;
    uint64_t batchB2B3B4 = tiling_->batchB2 * batchB3B4;
    uint64_t batchA3A4 = static_cast<uint64_t>(tiling_->batchA3) * tiling_->batchA4;
    uint64_t batchA2A3A4 = tiling_->batchA2 * batchA3A4;
    uint32_t multiA1C1 = tiling_->batchA1 / tiling_->batchC1;
    uint32_t multiA2C2 = tiling_->batchA2 / tiling_->batchC2;
    uint32_t multiA3C3 = tiling_->batchA3 / tiling_->batchC3;
    uint32_t multiA4C4 = tiling_->batchA4 / tiling_->batchC4;
    uint32_t multiB1C1 = tiling_->batchB1 / tiling_->batchC1;
    uint32_t multiB2C2 = tiling_->batchB2 / tiling_->batchC2;
    uint32_t multiB3C3 = tiling_->batchB3 / tiling_->batchC3;
    uint32_t multiB4C4 = tiling_->batchB4 / tiling_->batchC4;
    for (uint64_t b1Index = 0; b1Index < tiling_->batchC1; ++b1Index) {
        uint64_t batchC1Offset = b1Index * batchC2C3C4;
        uint64_t batchA1Offset = b1Index * batchA2A3A4 * multiA1C1;
        uint64_t batchB1Offset = b1Index * batchB2B3B4 * multiB1C1;
        for (uint64_t b2Index = 0; b2Index < tiling_->batchC2; ++b2Index) {
            uint64_t batchC2Offset = b2Index * batchC3C4 + batchC1Offset;
            uint64_t batchA2Offset = b2Index * batchA3A4 * multiA2C2 + batchA1Offset;
            uint64_t batchB2Offset = b2Index * batchB3B4 * multiB2C2 + batchB1Offset;
            for (uint64_t b3Index = 0; b3Index < tiling_->batchC3; ++b3Index) {
                uint64_t batchC3Offset = b3Index * tiling_->batchC4 + batchC2Offset;
                uint64_t batchA3Offset = b3Index * batchA3A4 * multiA3C3 + batchA2Offset;
                uint64_t batchB3Offset = b3Index * batchB3B4 * multiB3C3 + batchB2Offset;
                for (uint64_t b4Index = 0; b4Index < tiling_->batchC4; ++b4Index) {
                    block_.offset_.batchCOffset = batchC3Offset + b4Index;
                    block_.offset_.batchAOffset = batchA3Offset + b4Index * multiA4C4;
                    block_.offset_.batchBOffset = batchB3Offset + b4Index * multiB4C4;
                    block_.ResetAddressOffsets();
                    ComputeBasicOptiLoop();
                }
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::SetMMParaAndCompute()
{
    if constexpr (AscendC::IsSameType<yType, int8_t>::value) {
        if constexpr (quantType == QuantType::PER_TENSOR) {
            quantScalar_ = quantScaleGlobal_.GetValue(0);
        } else {
            quantScaleTileGlobal_ = quantScaleGlobal_[block_.params_.nIndex * baseN_];
        }
    }
    if (tiling_->isBias) {
        biasTileGlobal_ = biasGlobal_[block_.offset_.offsetBias];
        hasBias_ = true;
    }
#if __NPU_ARCH__ == 5102
    int32_t fixShift = static_cast<int32_t>(tiling_->mmadParam);
    if (fixShift < FIX_SHIFT_VAL_MIN_F16XF16) {
        fixShift = FIX_SHIFT_VAL_MIN_F16XF16;
    } else if (fixShift > FIX_SHIFT_VAL_MAX_F16XF16) {
        fixShift = FIX_SHIFT_VAL_MAX_F16XF16;
    }
    shiftValue_ = static_cast<uint8_t>(fixShift);
#endif
    aTileGlobal_ = aGlobal_[block_.offset_.offsetA];
    Iterate();
    GetTensorC(cGlobal_[block_.offset_.offsetC]);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::SetSingleShape(int32_t singleM,
                                                                                                       int32_t singleN,
                                                                                                       int32_t singleK)
{
    singleM_ = static_cast<uint64_t>(singleM);
    singleN_ = static_cast<uint64_t>(singleN);
    singleK_ = static_cast<uint64_t>(singleK);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyInA1(uint64_t kGmOffset,
                                                                                                 uint64_t curK,
                                                                                                 uint64_t l1BufId)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    uint64_t nDim = aTrans ? curK : singleM_;
    uint64_t dDim = aTrans ? singleM_ : curK;
    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = dDim;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = aTrans ? tiling_->m : tiling_->k;
    nd2nzParams.dstNzC0Stride = CeilAlign(nDim, BLOCK_CUBE_SIZE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    uint64_t gmOffset = aTrans ? kGmOffset * static_cast<uint64_t>(tiling_->m) : kGmOffset;
    AscendC::DataCopy(aL1_[l1BufId * aL1BufElems_], aTileGlobal_[gmOffset], nd2nzParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyInBiasL1()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_MTE1_MTE2_FLAG);
    AscendC::DataCopyPadParams padParams;
    AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(singleN_ * sizeof(biasType)), 0, 0};
    AscendC::DataCopyPad(biasL1_, biasTileGlobal_, biasParam, padParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyInA2(uint64_t kL1Offset,
                                                                                                 uint64_t curK,
                                                                                                 uint64_t curK0,
                                                                                                 uint64_t l0BufId)
{
    uint64_t mL1Align = CeilAlign(singleM_, BLOCK_CUBE_SIZE);
    AscendC::LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;
    if constexpr (!aTrans) {
        // (M, K)
        loadDataParams.mStep = CeilDiv(singleM_, BLOCK_CUBE_SIZE);
        loadDataParams.kStep = CeilDiv(curK0, C0_SIZE_A);
        loadDataParams.srcStride = CeilDiv(singleM_, BLOCK_CUBE_SIZE);
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;
    } else {
        // (K, M)
        loadDataParams.mStep = CeilDiv(curK0, BLOCK_CUBE_SIZE);
        loadDataParams.kStep = CeilDiv(singleM_, C0_SIZE_A);
        loadDataParams.srcStride = CeilDiv(curK, BLOCK_CUBE_SIZE);
        loadDataParams.dstStride = loadDataParams.kStep;
        loadDataParams.ifTranspose = true;
    }
    uint64_t l1BufId = l1LoopCnt_ & (l1BufNum_ - 1);
    uint64_t aL1Offset = l1BufId * aL1BufElems_ + (aTrans ? (kL1Offset * C0_SIZE_A) : (kL1Offset * mL1Align));
    AscendC::LoadData<xType>(aL0_[l0BufId * HALF_L0A_ELEMS], aL1_[aL1Offset], loadDataParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyInB2(uint64_t kL1Offset,
                                                                                                 uint64_t curK,
                                                                                                 uint64_t curK0,
                                                                                                 uint64_t l0BufId)
{
    const uint64_t l1BufId = l1LoopCnt_ & (l1BufNum_ - 1);
    uint64_t bL1Offset = l1BufId * bL1BufElems_;
    AscendC::LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;
    if constexpr (bTrans) {
        // 反量化后的转置 NZ 布局 (k1,n1,n0,k0)，逻辑形状 (N,K)
        const uint64_t nL1Align = CeilAlign(singleN_, BLOCK_CUBE_SIZE);
        loadDataParams.mStep = CeilDiv(singleN_, BLOCK_CUBE_SIZE);
        loadDataParams.kStep = CeilDiv(curK0, C0_SIZE_DEQ);
        loadDataParams.srcStride = CeilDiv(singleN_, BLOCK_CUBE_SIZE);
        loadDataParams.dstStride = loadDataParams.mStep;
        loadDataParams.ifTranspose = false;
        bL1Offset += kL1Offset * nL1Align;
        AscendC::LoadData<dequantType>(bL0_[l0BufId * HALF_L0B_ELEMS], bL1_[bL1Offset], loadDataParams);
        return;
    }

    // 非转置 NZ 布局 (n1,k1,k0,n0)，逻辑形状 (K,N)
    loadDataParams.kStep = CeilDiv(singleN_, C0_SIZE_DEQ);
    loadDataParams.srcStride = CeilDiv(curK, BLOCK_CUBE_SIZE);
    loadDataParams.dstStride = CeilDiv(singleN_, BLOCK_CUBE_SIZE);
    loadDataParams.ifTranspose = true;
    const uint16_t fullMStep = static_cast<uint16_t>(CeilDiv(curK0, BLOCK_CUBE_SIZE));
    // 转置装载进 L0B fractal，b16 场景 mStep 最小为 1
    loadDataParams.mStep = 1;
    uint64_t dstOffset = 0;
    // dst 每次步进 Align(n, 16) * 32B / sizeof(fp16) = 16个k行对应的 L0B block
    const uint64_t dstAddrStride = CeilAlign(singleN_, BLOCK_CUBE_SIZE) * ONE_BLK_SIZE / sizeof(dequantType);
    const uint16_t oriMstartPos = static_cast<uint16_t>(CeilDiv(kL1Offset, BLOCK_CUBE_SIZE));
    for (uint16_t idx = 0; idx < fullMStep; ++idx) {
        loadDataParams.mStartPosition = oriMstartPos + idx;
        AscendC::LoadData<dequantType>(bL0_[l0BufId * HALF_L0B_ELEMS + dstOffset], bL1_[bL1Offset], loadDataParams);
        dstOffset += dstAddrStride;
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyInBiasBt()
{
    // s32场景要对齐到2，因此是align(n / biasC0, 16 / biasC0)
    uint64_t nL1Align = CeilAlign(singleN_, BLOCK_CUBE_SIZE);
    uint64_t btAlign = BLOCK_CUBE_SIZE / AscendC::AuxGetC0Size<biasType>();
    uint16_t burstLen = CeilAlign(nL1Align / AscendC::AuxGetC0Size<biasType>(), btAlign);
    AscendC::DataCopyParams biasParam{1, burstLen, 0, 0};
    // biasBt_为单缓冲，须等上一tile读取bias的Mmad完成后再覆写
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(BIAS_BT_M_MTE1_FLAG);
#if __NPU_ARCH__ == 5102
    // A16W16 路径的 bias 定点位宽
    biasParam.fixShiftVal = FIX_SHIFT_VAL_LEN_A16W16 - shiftValue_;
#endif
    AscendC::DataCopy(biasBt_, biasL1_, biasParam);
    // bias在L1上的空间随MTE1读完成即可释放
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(BIAS_MTE1_MTE2_FLAG);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyQuantScaleL1()
{
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(QUANT_SCALE_FIX_MTE2_FLAG);
    AscendC::DataCopyPadParams padParams;
    AscendC::DataCopyParams scaleParam{1, static_cast<uint16_t>(singleN_ * sizeof(uint64_t)), 0, 0};
    AscendC::DataCopyPad(quantScaleL1_, quantScaleTileGlobal_, scaleParam, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(QUANT_SCALE_MTE2_FIX_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(QUANT_SCALE_MTE2_FIX_FLAG);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::MmadCompute(
    uint64_t curK0, uint64_t l0BufId, uint64_t l0cBufId, bool needBias, bool needCInit, bool isFinalAccum)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = singleM_;
    mmadParams.n = singleN_;
    mmadParams.k = curK0;
    mmadParams.disableGemv = true;
    // L0C乒乓使能时用显式flag同步（unitFlag=0）；未使能时开启unitflag：末次累加置3，硬件自动同步M->FIX
    mmadParams.unitFlag = enableL0cDb_ ? 0 : (isFinalAccum ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION);
    // 每个tile的首个Mmad重新初始化L0C（半区），避免跨tile累加
    mmadParams.cmatrixInitVal = (needCInit && !needBias);
#if __NPU_ARCH__ == 5102
    mmadParams.fixShiftVal = shiftValue_;
#endif
    if (needBias) {
        mmadParams.cmatrixSource = true;
        AscendC::Mmad(cL0_[l0cBufId * HALF_L0C_ELEMS], aL0_[l0BufId * HALF_L0A_ELEMS], bL0_[l0BufId * HALF_L0B_ELEMS],
                      biasBt_, mmadParams);
    } else {
        mmadParams.cmatrixSource = false;
        AscendC::Mmad(cL0_[l0cBufId * HALF_L0C_ELEMS], aL0_[l0BufId * HALF_L0A_ELEMS], bL0_[l0BufId * HALF_L0B_ELEMS],
                      mmadParams);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::FixpipeOut(
    const AscendC::GlobalTensor<yType>& gm, uint64_t l0cBufId)
{
    AscendC::FixpipeParamsC310 fixpipeParams;
    fixpipeParams.nSize = singleN_;
    fixpipeParams.mSize = singleM_;
    fixpipeParams.dstStride = tiling_->n;
    fixpipeParams.srcStride = CeilAlign(singleM_, BLOCK_CUBE_SIZE);
    if constexpr (AscendC::IsSameType<yType, int8_t>::value) {
        if constexpr (quantType == QuantType::PER_TENSOR) {
            fixpipeParams.quantPre = QuantMode_t::REQ8;
            fixpipeParams.deqScalar = quantScalar_;
        } else {
            fixpipeParams.quantPre = QuantMode_t::VREQ8;
        }
    } else {
        fixpipeParams.quantPre = QuantMode_t::DEQF16;
    }
    // L0C乒乓使能时用显式flag同步（unitFlag=0）；未使能时开启unitflag，硬件自动同步FIX->M
    fixpipeParams.unitFlag = enableL0cDb_ ? 0 : FINAL_ACCUMULATION;
    fixpipeParams.params = {1, 1, 1};
#if __NPU_ARCH__ == 5102
    // A16W16 路径：mmad 为 F16xF16，配对 shiftValue_ 补偿定点标度
    fixpipeParams.fixShiftVal = FIX_SHIFT_VAL_LEN_A16W16 - shiftValue_;
#endif
    if constexpr (AscendC::IsSameType<yType, int8_t>::value && quantType == QuantType::PER_CHANNEL) {
        AscendC::Fixpipe(gm, cL0_[l0cBufId * HALF_L0C_ELEMS], quantScaleL1_, fixpipeParams);
    } else {
        AscendC::Fixpipe(gm, cL0_[l0cBufId * HALF_L0C_ELEMS], fixpipeParams);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Iterate()
{
    uint64_t l0cBufId = enableL0cDb_ ? (l0cPingPong_ & 1) : 0;
    if (enableL0cDb_) {
        // 等待两轮前GetTensorC对该半区L0C的fixpipe完成，L0C半区可复用
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_M_FLAG + l0cBufId);
    }
    // 未使能L0C乒乓时，L0C复用由unitflag硬件自动同步（末次累加M->FIX、fixpipe FIX->M）
    uint64_t kL1Iter = CeilDiv(singleK_, kL1_);
    uint64_t kGmOffset = 0;
    bool biasLoaded = false;
    bool needCInit = true;
    for (uint64_t kIter = 0; kIter < kL1Iter; ++kIter) {
        uint64_t curK = (kIter + 1 == kL1Iter) ? (singleK_ - kGmOffset) : kL1_;
        uint64_t l1BufId = l1LoopCnt_ & (l1BufNum_ - 1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(B_L1_MTE1_MTE3_FLAG + l1BufId);
        uint64_t nOffset = block_.params_.nIndex * baseN_;
        DequantChunk(bL1_, bL1BufElems_, l1BufId, kGmOffset, curK, nOffset, singleN_, block_.offset_.batchBOffset);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + l1BufId);
        CopyInA1(kGmOffset, curK, l1BufId);
        bool needBias = hasBias_ && !biasLoaded;
        if (needBias) {
            CopyInBiasL1();
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1_MTE2_MTE1_FLAG + l1BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1_MTE2_MTE1_FLAG + l1BufId);
        // 核内硬事件替代原 AIV/AIC 跨核 flag
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(B_L1_MTE3_MTE1_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(B_L1_MTE3_MTE1_FLAG);
        uint64_t kL0Iter = CeilDiv(curK, baseK_);
        uint64_t kL1Offset = 0;
        for (uint64_t k0Iter = 0; k0Iter < kL0Iter; ++k0Iter) {
            uint64_t curK0 = (k0Iter + 1 == kL0Iter) ? (curK - kL1Offset) : baseK_;
            uint64_t l0BufId = l0LoopCnt_ & 1;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG + l0BufId);
            CopyInA2(kL1Offset, curK, curK0, l0BufId);
            CopyInB2(kL1Offset, curK, curK0, l0BufId);
            if (needBias && k0Iter == 0) {
                CopyInBiasBt();
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0_MTE1_M_FLAG + l0BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0_MTE1_M_FLAG + l0BufId);
            bool isFinalAccum = (kIter + 1 == kL1Iter) && (k0Iter + 1 == kL0Iter);
            MmadCompute(curK0, l0BufId, l0cBufId, needBias && k0Iter == 0, needCInit, isFinalAccum);
            if (needBias && k0Iter == 0) {
                // bias已随本Mmad读入，允许下一tile覆写biasBt_
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(BIAS_BT_M_MTE1_FLAG);
            }
            needCInit = false;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_M_MTE1_FLAG + l0BufId);
            l0LoopCnt_++;
            kL1Offset += curK0;
        }
        // L1 buffer：A区释放给下一轮MTE2，B区释放给下一轮反量化的MTE3（均为核内硬事件）
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_MTE1_MTE2_FLAG + l1BufId);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(B_L1_MTE1_MTE3_FLAG + l1BufId);
        l1LoopCnt_++;
        kGmOffset += curK;
        biasLoaded = biasLoaded || needBias;
    }
    if (enableL0cDb_) {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(L0C_M_FIX_FLAG + l0cBufId);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(L0C_M_FIX_FLAG + l0cBufId);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
template <bool sync>
__aicore__ inline void WeightQuantBatchMatmulV2ASWVCKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::GetTensorC(
    const AscendC::GlobalTensor<yType>& gm)
{
    uint64_t l0cBufId = enableL0cDb_ ? (l0cPingPong_ & 1) : 0;
    if constexpr (AscendC::IsSameType<yType, int8_t>::value && quantType == QuantType::PER_CHANNEL) {
        CopyQuantScaleL1();
    }
    FixpipeOut(gm, l0cBufId);
    if constexpr (AscendC::IsSameType<yType, int8_t>::value && quantType == QuantType::PER_CHANNEL) {
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(QUANT_SCALE_FIX_MTE2_FLAG);
    }
    if (enableL0cDb_) {
        // 标记该半区L0C空闲，供两轮后的Iterate复用
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C_FIX_M_FLAG + l0cBufId);
        l0cPingPong_++;
    }
    // 未使能时L0C复用由fixpipe unitflag硬件自动同步，无需显式SetFlag
}

} // namespace WeightQuantBatchMatmulV2::Arch35
