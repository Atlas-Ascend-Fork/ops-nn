/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// NormalizeBatch kernel 模板类（arch35 四计算单元，RegBase 向量路径）。
//
// 数学语义（对齐 canndev TIK 实现 ops/built-in/tbe/impl/normalize_batch.py）：
//   L  = clamp(seq_len[n], 1, d)                  统计长度（运行时钳制）
//   N  = L（per_feature）/ c*L（all_features）
//   E(x)  = sum(x[0:L]) / N                       统计只用前 L 个元素
//   var   = |E(x^2) - E(x)^2| * N/(N-1)           无偏修正 + abs 防负（N<2 时 var=0）
//   y     = (x - E(x)) / (sqrt(var) + eps)        eps 加在 sqrt 之后；整行 d 都被归一化
//
// 模板（TPL 双 bool 轴）：
//   IS_ALL_FEATURES=0 + IS_BIG_D=0：per_feature 行驻 UB 单遍（x 读 1 次）
//   IS_ALL_FEATURES=0 + IS_BIG_D=1：per_feature d 分段两遍（x 读 2 次）
//   IS_ALL_FEATURES=1 + IS_BIG_D=0：all_features 平面驻 UB 单遍（x 读 1 次）
//   IS_ALL_FEATURES=1 + IS_BIG_D=1：all_features 行批两遍（x 读 2 次；rowStride
//       ≤ eachLoopDNum 时 R 行一批驻留（逐行拷贝背靠背 + 整批一次同步），行自身超预算时逐行 d 分段）
//
// 流水/同步（全显式事件，正确性优先）：
//   CopyIn(MTE2) → TQueBind EnQue/DeQue(MTE2→V) → 具名 VF 函数统计（__simd_vf__
//   + asc_vf_call，体内禁运行时分支）→ V_S → 标量 mean/rstd → VF 归一化 →
//   V_MTE3 → CopyOut(MTE3) → Free → MTE3_MTE2（buffer 复用）
//   seq_len 一次拷入 + MTE2_S 后标量 GetValue。
//
// UB 布局：rowStride = CeilAlign(d, VL) 保证行内 LoadAlign 整 VL 读取不越过行缓冲
// 末尾；[d, rowStride) 与 [L, d) 的垃圾数据由掩码保证不影响语义。

#ifndef OPS_NORM_NORMALIZE_BATCH_KERNEL_H_
#define OPS_NORM_NORMALIZE_BATCH_KERNEL_H_

#include "kernel_operator.h"
#include "normalize_batch_tiling_struct.h"

namespace NsNormalizeBatch {
using namespace AscendC;

// 常量（fp32 路径）
constexpr uint32_t NB_VL_BYTES = 256;                       // VL = 256B（dav 3510）
constexpr uint32_t NB_VL_F32 = NB_VL_BYTES / sizeof(float); // 64 lane / rep
constexpr uint32_t NB_UB_BLOCK_BYTES = 32;                  // UB datablock 32B

// VF 向量函数（具名 __simd_vf__ + asc_vf_call 启动，范式同 bn_training_reduce）
// 约束：体内禁止运行时 if/switch/三目（cmp 指令 + 无法生成硬件循环），循环边界/
// 尾段计数由标量侧调用方算好传入（fullVec 个全向量 + tailVec∈{0,1} 个掩码尾
// 向量）；体内循环不得同时含 UpdateMask 动态掩码与 store——9.3.0 cce 无法为此
// 生成硬件循环（报 infinity loop/E40021，0 起点归纳亦不豁免），此类尾段一律
// 拆标量侧条件调用的无循环直线 VF（尾向量）或挪出循环（统计槽回写）；
// 跨函数的 VF store→load 依赖各函数体内的 LocalMemBar 显式屏障。

// 统计槽清零（每统计单元首段前调用一次；后续段统一走 BRC 累加——0.0f + x 精确
// 无舍入，与「首段直写」逐位等价，以此消灭体内 isFirst 运行时分支）
template <typename T>
__simd_vf__ inline void NbVfClearStatsSlots(__ubuf__ float* sumSlot, __ubuf__ float* sumsqSlot)
{
    AscendC::Reg::RegTensor<float> zeroReg;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    AscendC::Reg::Duplicate(zeroReg, 0.0f, allMask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumSlot, zeroReg, oneMask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumsqSlot, zeroReg, oneMask);
}

// 单段统计累加（[0, validCnt) 连续段）：fullVec 个全向量免掩码直累加 + tailVec
// 个掩码尾向量（掩码外 lane 清零不入和），Reduce 后 BRC 累进标量槽（平移域）
template <typename T>
__simd_vf__ inline void NbVfStatsSegAcc(__ubuf__ T* base, __ubuf__ float* sumSlot, __ubuf__ float* sumsqSlot,
                                        uint16_t fullVec, uint16_t tailVec, uint32_t tailCnt, float negShift)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> sqReg;
    AscendC::Reg::RegTensor<float> sumAcc;
    AscendC::Reg::RegTensor<float> sqAcc;
    AscendC::Reg::RegTensor<float> dstSumReg;
    AscendC::Reg::RegTensor<float> dstSqReg;
    AscendC::Reg::RegTensor<float> accReg;
    AscendC::Reg::RegTensor<float> zeroReg;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    AscendC::Reg::Duplicate(zeroReg, 0.0f, allMask);
    AscendC::Reg::Duplicate(sumAcc, 0.0f, allMask);
    AscendC::Reg::Duplicate(sqAcc, 0.0f, allMask);
    for (uint16_t i = 0; i < fullVec; ++i) { // 全向量：掩码外无垃圾 lane，免 Select
        AscendC::Reg::LoadAlign(xReg, base + static_cast<uint32_t>(i) * NB_VL_F32);
        AscendC::Reg::Adds(xReg, xReg, negShift, allMask); // 平移到 c 邻域
        AscendC::Reg::Add(sumAcc, xReg, sumAcc, allMask);
        AscendC::Reg::Mul(sqReg, xReg, xReg, allMask);
        AscendC::Reg::Add(sqAcc, sqReg, sqAcc, allMask);
    }
    AscendC::Reg::MaskReg tailMask = AscendC::Reg::UpdateMask<float>(tailCnt);
    // 尾向量（tailVec∈{0,1}）：归纳变量从 0 起 + offset 体内计算——VF 体内动态边界循
    // 环须可证 trip count 才能生成硬件循环，非 0 起点写法会被 cce 误判 infinity loop
    for (uint16_t k = 0; k < tailVec; ++k) {
        AscendC::Reg::LoadAlign(xReg, base + (static_cast<uint32_t>(fullVec) + k) * NB_VL_F32);
        AscendC::Reg::Select(xReg, xReg, zeroReg, tailMask);
        AscendC::Reg::Adds(xReg, xReg, negShift, tailMask);
        AscendC::Reg::Select(xReg, xReg, zeroReg, tailMask);
        AscendC::Reg::Add(sumAcc, xReg, sumAcc, allMask);
        AscendC::Reg::Mul(sqReg, xReg, xReg, allMask);
        AscendC::Reg::Add(sqAcc, sqReg, sqAcc, allMask);
    }
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(dstSumReg, sumAcc, allMask);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(dstSqReg, sqAcc, allMask);
    // 跨 VF 函数的 store→load 显式屏障（前段/清零的槽写入排干后再读；缺屏障
    // 偶发读到陈旧槽值，范式同 l2_normalize_grad 的 LocalMemBar）
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(accReg, sumSlot);
    AscendC::Reg::Add(accReg, accReg, dstSumReg, oneMask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumSlot, accReg, oneMask);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(accReg, sumsqSlot);
    AscendC::Reg::Add(accReg, accReg, dstSqReg, oneMask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumsqSlot, accReg, oneMask);
}

// 行批统计累加（BigD all_features 行批路径）：rows 行 padded 槽的 [0, validCnt)，
// 行内按 VL 分段（段掩码跨行复用），整批一次 Reduce + 槽回写
template <typename T>
__simd_vf__ inline void NbVfStatsRowsAcc(__ubuf__ T* base, __ubuf__ float* sumSlot, __ubuf__ float* sumsqSlot,
                                         uint16_t rows, uint32_t rowStrideElems, uint16_t fullVec, uint16_t tailVec,
                                         uint32_t tailCnt, float negShift)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::RegTensor<float> sqReg;
    AscendC::Reg::RegTensor<float> sumAcc;
    AscendC::Reg::RegTensor<float> sqAcc;
    AscendC::Reg::RegTensor<float> dstSumReg;
    AscendC::Reg::RegTensor<float> dstSqReg;
    AscendC::Reg::RegTensor<float> accReg;
    AscendC::Reg::RegTensor<float> zeroReg;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg oneMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    AscendC::Reg::Duplicate(zeroReg, 0.0f, allMask);
    AscendC::Reg::Duplicate(sumAcc, 0.0f, allMask);
    AscendC::Reg::Duplicate(sqAcc, 0.0f, allMask);
    for (uint16_t j = 0; j < fullVec; ++j) {
        for (uint16_t r = 0; r < rows; ++r) {
            AscendC::Reg::LoadAlign(xReg, base + r * rowStrideElems + static_cast<uint32_t>(j) * NB_VL_F32);
            AscendC::Reg::Adds(xReg, xReg, negShift, allMask);
            AscendC::Reg::Add(sumAcc, xReg, sumAcc, allMask);
            AscendC::Reg::Mul(sqReg, xReg, xReg, allMask);
            AscendC::Reg::Add(sqAcc, sqReg, sqAcc, allMask);
        }
    }
    AscendC::Reg::MaskReg tailMask = AscendC::Reg::UpdateMask<float>(tailCnt);
    // 尾向量（tailVec∈{0,1}）：归纳变量从 0 起 + offset 体内计算（同 NbVfStatsSegAcc，
    // 满足 VF 体内硬件循环可证 trip count 约束）
    for (uint16_t k = 0; k < tailVec; ++k) {
        for (uint16_t r = 0; r < rows; ++r) {
            AscendC::Reg::LoadAlign(xReg, base + r * rowStrideElems + (static_cast<uint32_t>(fullVec) + k) * NB_VL_F32);
            AscendC::Reg::Select(xReg, xReg, zeroReg, tailMask);
            AscendC::Reg::Adds(xReg, xReg, negShift, tailMask);
            AscendC::Reg::Select(xReg, xReg, zeroReg, tailMask);
            AscendC::Reg::Add(sumAcc, xReg, sumAcc, allMask);
            AscendC::Reg::Mul(sqReg, xReg, xReg, allMask);
            AscendC::Reg::Add(sqAcc, sqReg, sqAcc, allMask);
        }
    }
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(dstSumReg, sumAcc, allMask);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(dstSqReg, sqAcc, allMask);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(accReg, sumSlot);
    AscendC::Reg::Add(accReg, accReg, dstSumReg, oneMask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumSlot, accReg, oneMask);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(accReg, sumsqSlot);
    AscendC::Reg::Add(accReg, accReg, dstSqReg, oneMask);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(sumsqSlot, accReg, oneMask);
}

// 尾向量归一化（tailVec∈{0,1}，单发直线 VF，无循环体）：VF 体内「UpdateMask
// 动态掩码 + store」的循环 cce 无法生成硬件循环（报 infinity loop/E40021，
// 0 起点归纳亦不豁免），尾向量拆为无循环直线 VF 由标量侧条件调用；指令序列
// 与循环版逐位一致，无精度影响
template <typename T>
__simd_vf__ inline void NbVfNormalizeTail(__ubuf__ T* base, uint32_t tailCnt, float neMeanHi, float neMeanLo,
                                          float rstd)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::MaskReg tailMask = AscendC::Reg::UpdateMask<float>(tailCnt);
    AscendC::Reg::LoadAlign(xReg, base);
    AscendC::Reg::Adds(xReg, xReg, neMeanHi, tailMask);
    AscendC::Reg::Adds(xReg, xReg, neMeanLo, tailMask); // Fast2Sum 残差补偿
    AscendC::Reg::Muls(xReg, xReg, rstd, tailMask);
    AscendC::Reg::StoreAlign(base, xReg, tailMask);
}

// 归一化：base[0:fullVec*VL) 原位 (x+neMeanHi+neMeanLo)*rstd（Fast2Sum 双 float 补偿；
// 整行 d 都处理，含 [L, d) 段；全向量循环免掩码——静态 ALL 掩码下 cce 可生成硬件循环，
// 尾向量由调用方走 NbVfNormalizeTail）
template <typename T>
__simd_vf__ inline void NbVfNormalize(__ubuf__ T* base, uint16_t fullVec, float neMeanHi, float neMeanLo, float rstd)
{
    AscendC::Reg::RegTensor<float> xReg;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    for (uint16_t i = 0; i < fullVec; ++i) {
        AscendC::Reg::LoadAlign(xReg, base + static_cast<uint32_t>(i) * NB_VL_F32);
        AscendC::Reg::Adds(xReg, xReg, neMeanHi, allMask);
        AscendC::Reg::Adds(xReg, xReg, neMeanLo, allMask); // Fast2Sum 残差补偿
        AscendC::Reg::Muls(xReg, xReg, rstd, allMask);
        AscendC::Reg::StoreAlign(base + static_cast<uint32_t>(i) * NB_VL_F32, xReg, allMask);
    }
}

// NormalizeBatchKernel<T, IS_ALL_FEATURES, IS_BIG_D>
//   T 当前仅 float（def 侧只注册 DT_FLOAT；fp16/bf16 扩展时需加 Cast 链）
template <typename T, bool IS_ALL_FEATURES, bool IS_BIG_D>
class NormalizeBatchKernel {
public:
    __aicore__ inline NormalizeBatchKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR seqLen, GM_ADDR y, const NormalizeBatchTilingData* tilingData,
                                TPipe* pipe)
    {
        tilingData_ = tilingData;
        pipe_ = pipe;
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        seqLenGm_.SetGlobalBuffer((__gm__ int32_t*)seqLen);
        yGm_.SetGlobalBuffer((__gm__ T*)y);

        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        nDim_ = tilingData->nDim;
        cDim_ = tilingData->cDim;
        dDim_ = tilingData->dDim;
        rowStride_ = tilingData->rowStride;
        eachLoopDNum_ = tilingData->eachLoopDNum;
        epsilon_ = tilingData->epsilon;

        usedCoreNum_ = static_cast<int64_t>(tilingData->usedCoreNum);
        normalCoreProcessNum_ = tilingData->normalCoreProcessNum;
        // 末核承载尾块；usedCoreNum < 物理核数时多余核 batchCount_ = 0
        batchCount_ = (blockIdx_ == usedCoreNum_ - 1) ? tilingData->tailCoreProcessNum : normalCoreProcessNum_;
        batchStart_ = blockIdx_ * normalCoreProcessNum_;

        // 防御性夹紧（host 侧 FloorDiv 已保证 tail ≥ normal ≥ 1；此夹紧兜底 tiling 回归）：
        // 起始越界则 count 为 0；否则 count 不越过 batch 总量，杜绝 GM 越界读写
        totalBatch_ = IS_ALL_FEATURES ? nDim_ : (nDim_ * cDim_);
        const int64_t remain = totalBatch_ - batchStart_;
        batchCount_ = (remain > 0) ? ((batchCount_ < remain) ? batchCount_ : remain) : 0;

        // UB buffer
        pipe_->InitBuffer(dataQue_, 2, static_cast<uint32_t>(tilingData->dataBufElems * sizeof(T)));
        pipe_->InitBuffer(seqLenBuf_, static_cast<uint32_t>(tilingData->seqLenBufSize));
        pipe_->InitBuffer(scalarBuf_, static_cast<uint32_t>(tilingData->scalarBufSize));
        scalarUb_ = scalarBuf_.Get<float>();

        // seq_len 整体拷入（GM 基址对齐；n*4B 单块），MTE2→S 后标量可读
        seqLenUb_ = seqLenBuf_.Get<int32_t>();
        DataCopyExtParams seqParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(nDim_ * sizeof(int32_t)), 0, 0,
                                       0};
        DataCopyPadExtParams<int32_t> seqPad = {false, 0, 0, 0};
        DataCopyPad(seqLenUb_, seqLenGm_, seqParams, seqPad);
        PipeMte2S();
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }
        for (int64_t b = 0; b < batchCount_; ++b) {
            const int64_t unit = batchStart_ + b;
            if constexpr (IS_ALL_FEATURES) {
                if constexpr (IS_BIG_D) {
                    PlaneTwoPass(unit);
                } else {
                    PlaneOnePass(unit);
                }
            } else {
                const int64_t nIdx = unit / cDim_;
                const int64_t cIdx = unit - nIdx * cDim_;
                if constexpr (IS_BIG_D) {
                    RowTwoPass(nIdx, cIdx);
                } else {
                    RowOnePass(nIdx, cIdx);
                }
            }
        }
    }

private:
    // 流水事件（全显式，正确性优先）
    // 固定事件 ID（范式同 l2_normalize_grad 的 EVENT_IDx）：FetchEventID 只窥探不占位，
    // 多次调用存在 ID 复用不确定性；每通道固定 ID 0，Set/Wait 成对即时完成，复用安全。
    static constexpr event_t NB_EVT_ID0 = static_cast<event_t>(0);

    __aicore__ inline void PipeVS()
    {
        SetFlag<HardEvent::V_S>(NB_EVT_ID0);
        WaitFlag<HardEvent::V_S>(NB_EVT_ID0);
    }

    __aicore__ inline void PipeVMte3()
    {
        SetFlag<HardEvent::V_MTE3>(NB_EVT_ID0);
        WaitFlag<HardEvent::V_MTE3>(NB_EVT_ID0);
    }

    __aicore__ inline void PipeMte3Mte2()
    {
        SetFlag<HardEvent::MTE3_MTE2>(NB_EVT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(NB_EVT_ID0);
    }

    __aicore__ inline void PipeMte2S()
    {
        SetFlag<HardEvent::MTE2_S>(NB_EVT_ID0);
        WaitFlag<HardEvent::MTE2_S>(NB_EVT_ID0);
    }

    // 搬运
    __aicore__ inline void CopyInRange(const LocalTensor<T>& buf, uint32_t ubElemOff, int64_t gmElemOff,
                                       uint32_t elemCnt)
    {
        DataCopyExtParams params = {static_cast<uint16_t>(1), elemCnt * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad(buf[ubElemOff], xGm_[gmElemOff], params, padParams);
    }

    __aicore__ inline void CopyOutRange(const LocalTensor<T>& buf, uint32_t ubElemOff, int64_t gmElemOff,
                                        uint32_t elemCnt)
    {
        DataCopyExtParams params = {static_cast<uint16_t>(1), elemCnt * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[gmElemOff], buf[ubElemOff], params);
    }

    // 行批搬运（BigD all_features 行批路径）
    // R 行的 [0, elemCnt) 背靠背 1D 拷入 VL 对齐 padded 槽（GM 行距 dDim，UB 行距
    // rowStride），整批一次 EnQue/DeQue——逐行路径的耗时主源是每行一次 MTE2→V 同步
    // + 每行 Reduce×2 标量槽往返，行批后同步次数 /rowsPerChunk。
    // 2D DataCopyPad（blockCount>1）在 3510 上触发 VEC_ERROR，故逐行发射、流水
    // 消费（范式同 PlaneOnePass 的平面拷入）。elemCnt 取 len（统计趟）或 dDim
    // （归一化趟，整行）。
    __aicore__ inline void CopyRowsPadIn(const LocalTensor<T>& buf, int64_t gmBase, uint16_t rows, uint32_t elemCnt)
    {
        for (uint16_t r = 0; r < rows; ++r) {
            CopyInRange(buf, r * static_cast<uint32_t>(rowStride_), gmBase + r * dDim_, elemCnt);
        }
    }

    // 行批写回：padded 槽 [0, d) × R 行 → GM 平面（行连续；槽内 [d, rowStride) 不带出）
    __aicore__ inline void CopyRowsPadOut(const LocalTensor<T>& buf, int64_t gmBase, uint16_t rows)
    {
        for (uint16_t r = 0; r < rows; ++r) {
            CopyOutRange(buf, r * static_cast<uint32_t>(rowStride_), gmBase + r * dDim_, static_cast<uint32_t>(dDim_));
        }
    }

    // 统计段累加（平移域）：base[0:validCnt) 的 Σ(x−c) / Σ(x−c)² 累进 UB 标量槽
    // c 为统计单元的平移量（取单元首元素，≈mean）：var = (Σ(x−c)² − (Σ(x−c))²/N)·N/(N−1)
    // 直接算 E(x²)−E(x)² 在 |mean|≫std 时（如 L=1、双元素接近）float32 发生严重抵消，
    // var 仅剩 1~2 位有效数字；平移后两项与方差同量级，抵消消失（Cauchy-Schwarz
    // 保证 S1c²/N ≤ S2c）。
    // 段间累加全程 V 域（BRC 广播读槽 + 加 + 写回）；V→S 回读由调用方每单元做一次。
    __aicore__ inline void StatsFromUb(__ubuf__ T* base, uint32_t validCnt, bool isFirst, float negShift)
    {
        __ubuf__ float* sumSlot = (__ubuf__ float*)scalarUb_.GetPhyAddr();
        __ubuf__ float* sumsqSlot = (__ubuf__ float*)scalarUb_.GetPhyAddr() + (NB_UB_BLOCK_BYTES / sizeof(float));
        if (isFirst) { // 首段前清零 → 后续统一 BRC 累加（0.0f + x 精确，分支留标量侧）
            asc_vf_call<NbVfClearStatsSlots<T>>(sumSlot, sumsqSlot);
        }
        const uint16_t fullVec = static_cast<uint16_t>(validCnt / NB_VL_F32);
        const uint32_t tailCnt = validCnt % NB_VL_F32;
        const uint16_t tailVec = (tailCnt != 0U) ? 1U : 0U;
        asc_vf_call<NbVfStatsSegAcc<T>>(base, sumSlot, sumsqSlot, fullVec, tailVec, tailCnt, negShift);
    }

    // 统计单元收口：V→S 一次回读 sum/sumsq
    __aicore__ inline void ReadStatsSlots(float& sum, float& sumsq)
    {
        PipeVS();
        sum = scalarUb_.GetValue(0);
        sumsq = scalarUb_.GetValue(NB_UB_BLOCK_BYTES / sizeof(float));
    }

    // 行批统计（平移域，BigD all_features 行批路径）
    // R 行 padded 槽的 [0, validCnt) 寄存器级跨行累加，整批一次 Reduce + 槽回写。
    // 与 StatsFromUb 同元素语义（Select 清 → Adds(-shift) → Select 清 → 累加），
    // 区别仅在 fp32 累加分组：lane 累加器跨行常驻、每批一次水平归约（逐行版为每行
    // Reduce×2 + BRC 槽往返），行数多时该标量槽链是主要耗时项。
    // 行内按 VL 分段（vecPerRow 段，rowStride 为 VL 整数倍），段掩码跨行复用。
    __aicore__ inline void StatsRowsFromUb(__ubuf__ T* base, uint16_t rows, uint32_t validCnt, bool isFirst,
                                           float negShift)
    {
        __ubuf__ float* sumSlot = (__ubuf__ float*)scalarUb_.GetPhyAddr();
        __ubuf__ float* sumsqSlot = (__ubuf__ float*)scalarUb_.GetPhyAddr() + (NB_UB_BLOCK_BYTES / sizeof(float));
        if (isFirst) { // 首批前清零 → 统一 BRC 累加（与 StatsFromUb 同口径）
            asc_vf_call<NbVfClearStatsSlots<T>>(sumSlot, sumsqSlot);
        }
        const uint16_t fullVec = static_cast<uint16_t>(validCnt / NB_VL_F32);
        const uint32_t tailCnt = validCnt % NB_VL_F32;
        const uint16_t tailVec = (tailCnt != 0U) ? 1U : 0U;
        asc_vf_call<NbVfStatsRowsAcc<T>>(base, sumSlot, sumsqSlot, rows, static_cast<uint32_t>(rowStride_), fullVec,
                                         tailVec, tailCnt, negShift);
    }

    // 归一化：base[0:cnt) 原位 (x+neMeanHi+neMeanLo)*rstd（整行 d 都处理，含 [L, d) 段）
    // 双 float 补偿加：近常量数据下 (x−mean) 强消没，单条 Adds 的舍入在半 ULP(mean) 量级，
    // 拆 hi/lo 两连 Adds 后舍入降到 ULP(x−mean) 量级（误差随 rstd 放大前先压 3~4 个数量级）。
    __aicore__ inline void NormalizeInUb(__ubuf__ T* base, uint32_t cnt, float neMeanHi, float neMeanLo, float rstd)
    {
        const uint16_t fullVec = static_cast<uint16_t>(cnt / NB_VL_F32);
        const uint32_t tailCnt = cnt % NB_VL_F32;
        asc_vf_call<NbVfNormalize<T>>(base, fullVec, neMeanHi, neMeanLo, rstd);
        if (tailCnt != 0U) { // 尾向量分支留标量侧（VF 体内禁运行时分支），单发直线 VF 处理
            asc_vf_call<NbVfNormalizeTail<T>>(base + static_cast<uint32_t>(fullVec) * NB_VL_F32, tailCnt, neMeanHi,
                                              neMeanLo, rstd);
        }
    }

    // 标量统计（平移域收口）：mean = c + S1c/N；var = (S2c − S1c²/N)·N/(N−1)
    // abs 防负 + N<2 保护 / rstd = 1/(std+eps)（eps 加在 sqrt 之后，对齐 TIK）
    // mean 以 hi/lo 双 float 返回：近常量数据（std≪|mean|）时 rstd 极大，mean 的
    // fp32 表示误差（半 ULP）会被放大数万倍，Fast2Sum 残差消除该误差（范式同
    // l2_normalize_grad double-float）。
    __aicore__ inline void ComputeMeanRstd(float shift, float s1c, float s2c, int64_t n, float& neMeanHi,
                                           float& neMeanLo, float& rstd)
    {
        if (n < 1) {
            n = 1; // ClampSeqLen 已保证 ≥1，双保险
        }
        const float nF = static_cast<float>(n);
        const float r = s1c / nF;
        const float meanHi = shift + r;
        const float meanLo = (shift - meanHi) + r; // Fast2Sum 残差（|shift| ≥ |r| 时精确）
        float var = 0.0f;
        if (n >= 2) {
            var = s2c / nF - (s1c / nF) * (s1c / nF);     // 平移域：两项与 var 同量级，无抵消
            var = (var < 0.0f) ? -var : var;              // abs 防负（对齐 TIK vabs）
            var = var * (nF / static_cast<float>(n - 1)); // 无偏修正 N/(N-1)
        }
        const float stdDev = sqrt(var);
        rstd = 1.0f / (stdDev + epsilon_); // eps 加在 sqrt 之后（对齐 TIK）
        neMeanHi = -meanHi;
        neMeanLo = -meanLo;
    }

    // seq_len 运行时钳制 [1, d]（TIK 无防御，950 版补上）
    __aicore__ inline int64_t ClampSeqLen(int64_t nIdx)
    {
        int64_t len = static_cast<int64_t>(seqLenUb_.GetValue(nIdx));
        if (len < 1) {
            len = 1;
        }
        if (len > dDim_) {
            len = dDim_;
        }
        return len;
    }

    // per_feature：单行 (n, c, :)

    // SmallD：行驻 UB 单遍（x 读 1 次）
    __aicore__ inline void RowOnePass(int64_t nIdx, int64_t cIdx)
    {
        const int64_t gmRow = (nIdx * cDim_ + cIdx) * dDim_;
        const int64_t len = ClampSeqLen(nIdx);
        LocalTensor<T> buf = dataQue_.AllocTensor<T>();
        CopyInRange(buf, 0, gmRow, static_cast<uint32_t>(dDim_));
        dataQue_.EnQue(buf);
        buf = dataQue_.DeQue<T>(); // MTE2→V
        PipeMte2S();               // MTE2→S：标量读首元素作统计平移量
        const float shift = buf.GetValue(0);
        StatsFromUb((__ubuf__ T*)buf.GetPhyAddr(), static_cast<uint32_t>(len), true, -shift);
        float s1c = 0.0f;
        float s2c = 0.0f;
        ReadStatsSlots(s1c, s2c);
        float neMeanHi = 0.0f;
        float neMeanLo = 0.0f;
        float rstd = 0.0f;
        ComputeMeanRstd(shift, s1c, s2c, len, neMeanHi, neMeanLo, rstd);
        NormalizeInUb((__ubuf__ T*)buf.GetPhyAddr(), static_cast<uint32_t>(dDim_), neMeanHi, neMeanLo, rstd);
        PipeVMte3();
        CopyOutRange(buf, 0, gmRow, static_cast<uint32_t>(dDim_));
        dataQue_.FreeTensor(buf);
        PipeMte3Mte2(); // buffer 复用：MTE3 读完成 → 下一行 MTE2 写
    }

    // BigD：d 分段两遍（统计趟 + 归一化趟，x 读 2 次）
    __aicore__ inline void RowTwoPass(int64_t nIdx, int64_t cIdx)
    {
        const int64_t gmRow = (nIdx * cDim_ + cIdx) * dDim_;
        const int64_t len = ClampSeqLen(nIdx);
        // 统计趟：[0, len) 分段累加（平移域；首元素经 MTE2→S 标量读作平移量）
        float shift = 0.0f;
        for (int64_t off = 0; off < len; off += eachLoopDNum_) {
            const uint32_t cnt = static_cast<uint32_t>((len - off) > eachLoopDNum_ ? eachLoopDNum_ : (len - off));
            LocalTensor<T> buf = dataQue_.AllocTensor<T>();
            CopyInRange(buf, 0, gmRow + off, cnt);
            dataQue_.EnQue(buf);
            buf = dataQue_.DeQue<T>();
            if (off == 0) {
                PipeMte2S(); // MTE2→S：标量读首元素
                shift = buf.GetValue(0);
            }
            StatsFromUb((__ubuf__ T*)buf.GetPhyAddr(), cnt, off == 0, -shift);
            dataQue_.FreeTensor(buf);
        }
        float s1c = 0.0f;
        float s2c = 0.0f;
        ReadStatsSlots(s1c, s2c);
        float neMeanHi = 0.0f;
        float neMeanLo = 0.0f;
        float rstd = 0.0f;
        ComputeMeanRstd(shift, s1c, s2c, len, neMeanHi, neMeanLo, rstd);
        // 归一化趟：[0, d) 分段变换并写回
        for (int64_t off = 0; off < dDim_; off += eachLoopDNum_) {
            const uint32_t cnt = static_cast<uint32_t>((dDim_ - off) > eachLoopDNum_ ? eachLoopDNum_ : (dDim_ - off));
            LocalTensor<T> buf = dataQue_.AllocTensor<T>();
            CopyInRange(buf, 0, gmRow + off, cnt);
            dataQue_.EnQue(buf);
            buf = dataQue_.DeQue<T>();
            NormalizeInUb((__ubuf__ T*)buf.GetPhyAddr(), cnt, neMeanHi, neMeanLo, rstd);
            PipeVMte3();
            CopyOutRange(buf, 0, gmRow + off, cnt);
            dataQue_.FreeTensor(buf);
            PipeMte3Mte2();
        }
    }

    // all_features：单平面 (n, :, :)

    // SmallD：平面（c 行）驻 UB 单遍（x 读 1 次）；行距 rowStride 保证 LoadAlign 安全
    __aicore__ inline void PlaneOnePass(int64_t nIdx)
    {
        const int64_t planeBase = nIdx * cDim_ * dDim_;
        const int64_t len = ClampSeqLen(nIdx);
        const int64_t nCnt = cDim_ * len;
        LocalTensor<T> buf = dataQue_.AllocTensor<T>();
        for (int64_t r = 0; r < cDim_; ++r) {
            CopyInRange(buf, static_cast<uint32_t>(r * rowStride_), planeBase + r * dDim_,
                        static_cast<uint32_t>(dDim_));
        }
        dataQue_.EnQue(buf);
        buf = dataQue_.DeQue<T>(); // MTE2→V
        PipeMte2S();               // MTE2→S：标量读平面首元素作统计平移量
        const float shift = buf.GetValue(0);
        for (int64_t r = 0; r < cDim_; ++r) {
            StatsFromUb((__ubuf__ T*)buf.GetPhyAddr() + r * rowStride_, static_cast<uint32_t>(len), r == 0, -shift);
        }
        float s1c = 0.0f;
        float s2c = 0.0f;
        ReadStatsSlots(s1c, s2c);
        float neMeanHi = 0.0f;
        float neMeanLo = 0.0f;
        float rstd = 0.0f;
        ComputeMeanRstd(shift, s1c, s2c, nCnt, neMeanHi, neMeanLo, rstd);
        for (int64_t r = 0; r < cDim_; ++r) {
            NormalizeInUb((__ubuf__ T*)buf.GetPhyAddr() + r * rowStride_, static_cast<uint32_t>(dDim_), neMeanHi,
                          neMeanLo, rstd);
        }
        PipeVMte3();
        for (int64_t r = 0; r < cDim_; ++r) {
            CopyOutRange(buf, static_cast<uint32_t>(r * rowStride_), planeBase + r * dDim_,
                         static_cast<uint32_t>(dDim_));
        }
        dataQue_.FreeTensor(buf);
        PipeMte3Mte2();
    }

    // BigD：两遍（统计趟 + 归一化趟，x 读 2 次）。
    // rowStride ≤ eachLoopDNum（整行驻留）→ 行批路径：R 行一批驻留（同步次数 /rowsPerChunk）+
    // 寄存器级跨行统计 + 批量写回；否则（行自身超预算，如 d=60000）→ 逐行 d 分段路径。
    __aicore__ inline void PlaneTwoPass(int64_t nIdx)
    {
        if (eachLoopDNum_ / rowStride_ >= 1) {
            PlaneTwoPassRows(nIdx);
        } else {
            PlaneTwoPassSeg(nIdx);
        }
    }

    // BigD 行批路径（rowStride ≤ eachLoopDNum）：rowsPerChunk = eachLoopDNum / rowStride 行
    // 一批，一批两拷贝（MTE2 入 / MTE3 出）+ 一次队列同步——逐行路径在 c 大 × d 小时
    // （每行一次 MTE2→V 同步 + 标量槽往返）退化为延迟型串行链；行批后同步次数
    // /rowsPerChunk，拷贝与向量吞吐接管。
    __aicore__ inline void PlaneTwoPassRows(int64_t nIdx)
    {
        const int64_t planeBase = nIdx * cDim_ * dDim_;
        const int64_t len = ClampSeqLen(nIdx);
        const int64_t nCnt = cDim_ * len;
        const int64_t rowsPerChunk = eachLoopDNum_ / rowStride_;
        // 统计趟：行批 [0, len) 驻留 + 跨行累加（平移域；首批首元素经 MTE2→S 标量读作平移量）
        float shift = 0.0f;
        bool firstChunk = true;
        for (int64_t row0 = 0; row0 < cDim_; row0 += rowsPerChunk) {
            const int64_t rowsLeft = cDim_ - row0;
            const uint16_t rows = static_cast<uint16_t>(rowsLeft < rowsPerChunk ? rowsLeft : rowsPerChunk);
            LocalTensor<T> buf = dataQue_.AllocTensor<T>();
            CopyRowsPadIn(buf, planeBase + row0 * dDim_, rows, static_cast<uint32_t>(len));
            dataQue_.EnQue(buf);
            buf = dataQue_.DeQue<T>();
            if (firstChunk) {
                PipeMte2S(); // MTE2→S：标量读平面首元素（行 0 槽 0）
                shift = buf.GetValue(0);
            }
            StatsRowsFromUb((__ubuf__ T*)buf.GetPhyAddr(), rows, static_cast<uint32_t>(len), firstChunk, -shift);
            dataQue_.FreeTensor(buf);
            firstChunk = false;
        }
        float s1c = 0.0f;
        float s2c = 0.0f;
        ReadStatsSlots(s1c, s2c);
        float neMeanHi = 0.0f;
        float neMeanLo = 0.0f;
        float rstd = 0.0f;
        ComputeMeanRstd(shift, s1c, s2c, nCnt, neMeanHi, neMeanLo, rstd);
        // 归一化趟：行批整行驻留 → 逐行变换（槽内 [d, rowStride) 垃圾 lane 同被变换，
        // 但批量写回只带出每行 [0, d)）→ 批量写回
        for (int64_t row0 = 0; row0 < cDim_; row0 += rowsPerChunk) {
            const int64_t rowsLeft = cDim_ - row0;
            const uint16_t rows = static_cast<uint16_t>(rowsLeft < rowsPerChunk ? rowsLeft : rowsPerChunk);
            LocalTensor<T> buf = dataQue_.AllocTensor<T>();
            CopyRowsPadIn(buf, planeBase + row0 * dDim_, rows, static_cast<uint32_t>(dDim_));
            dataQue_.EnQue(buf);
            buf = dataQue_.DeQue<T>();
            __ubuf__ T* base = (__ubuf__ T*)buf.GetPhyAddr();
            for (uint16_t r = 0; r < rows; ++r) {
                NormalizeInUb(base + r * static_cast<uint32_t>(rowStride_), static_cast<uint32_t>(dDim_), neMeanHi,
                              neMeanLo, rstd);
            }
            PipeVMte3();
            CopyRowsPadOut(buf, planeBase + row0 * dDim_, rows);
            dataQue_.FreeTensor(buf);
            PipeMte3Mte2();
        }
    }

    // BigD 逐行 d 分段路径（rowStride > eachLoopDNum：行自身超预算，行批不可行）
    __aicore__ inline void PlaneTwoPassSeg(int64_t nIdx)
    {
        const int64_t planeBase = nIdx * cDim_ * dDim_;
        const int64_t len = ClampSeqLen(nIdx);
        const int64_t nCnt = cDim_ * len;
        // 统计趟：逐行分段累加（平移域；首段元素经 MTE2→S 标量读作平移量）
        float shift = 0.0f;
        for (int64_t r = 0; r < cDim_; ++r) {
            const int64_t rowBase = planeBase + r * dDim_;
            for (int64_t off = 0; off < len; off += eachLoopDNum_) {
                const uint32_t cnt = static_cast<uint32_t>((len - off) > eachLoopDNum_ ? eachLoopDNum_ : (len - off));
                LocalTensor<T> buf = dataQue_.AllocTensor<T>();
                CopyInRange(buf, 0, rowBase + off, cnt);
                dataQue_.EnQue(buf);
                buf = dataQue_.DeQue<T>();
                if (r == 0 && off == 0) {
                    PipeMte2S(); // MTE2→S：标量读首元素
                    shift = buf.GetValue(0);
                }
                StatsFromUb((__ubuf__ T*)buf.GetPhyAddr(), cnt, r == 0 && off == 0, -shift);
                dataQue_.FreeTensor(buf);
            }
        }
        float s1c = 0.0f;
        float s2c = 0.0f;
        ReadStatsSlots(s1c, s2c);
        float neMeanHi = 0.0f;
        float neMeanLo = 0.0f;
        float rstd = 0.0f;
        ComputeMeanRstd(shift, s1c, s2c, nCnt, neMeanHi, neMeanLo, rstd);
        // 归一化趟：逐行分段变换并写回
        for (int64_t r = 0; r < cDim_; ++r) {
            const int64_t rowBase = planeBase + r * dDim_;
            for (int64_t off = 0; off < dDim_; off += eachLoopDNum_) {
                const uint32_t cnt = static_cast<uint32_t>((dDim_ - off) > eachLoopDNum_ ? eachLoopDNum_ :
                                                                                           (dDim_ - off));
                LocalTensor<T> buf = dataQue_.AllocTensor<T>();
                CopyInRange(buf, 0, rowBase + off, cnt);
                dataQue_.EnQue(buf);
                buf = dataQue_.DeQue<T>();
                NormalizeInUb((__ubuf__ T*)buf.GetPhyAddr(), cnt, neMeanHi, neMeanLo, rstd);
                PipeVMte3();
                CopyOutRange(buf, 0, rowBase + off, cnt);
                dataQue_.FreeTensor(buf);
                PipeMte3Mte2();
            }
        }
    }

    // 成员
    TPipe* pipe_ = nullptr;
    const NormalizeBatchTilingData* tilingData_ = nullptr;
    TQue<QuePosition::VECIN, 2> dataQue_; // DeQue 产生 MTE2→V 序；TQueBind<VECIN,VECOUT> 为 MTE2→MTE3
                                          // 直通设计，VF 消费缺 MTE2→V 同步会读到未落盘数据（统计趟竞态）
    TBuf<QuePosition::VECCALC> seqLenBuf_;
    TBuf<QuePosition::VECCALC> scalarBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<int32_t> seqLenGm_;
    GlobalTensor<T> yGm_;
    LocalTensor<int32_t> seqLenUb_;
    LocalTensor<float> scalarUb_;

    int64_t blockIdx_ = 0;
    int64_t nDim_ = 0;
    int64_t cDim_ = 0;
    int64_t dDim_ = 0;
    int64_t rowStride_ = 0;
    int64_t eachLoopDNum_ = 0;
    float epsilon_ = 0.0f;

    int64_t usedCoreNum_ = 0;
    int64_t normalCoreProcessNum_ = 0;
    int64_t batchStart_ = 0;
    int64_t batchCount_ = 0;
    int64_t totalBatch_ = 0;
};

} // namespace NsNormalizeBatch

#endif // OPS_NORM_NORMALIZE_BATCH_KERNEL_H_
