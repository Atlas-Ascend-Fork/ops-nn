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
 * \file dynamic_block_hi_f4_quant_kernel.h
 * \brief Kernel主体类 for DynamicBlockHiF4Quant on ascend960dt (arch35)
 */

#ifndef DYNAMIC_BLOCK_HI_F4_QUANT_KERNEL_H_
#define DYNAMIC_BLOCK_HI_F4_QUANT_KERNEL_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "dynamic_block_hi_f4_quant_tilingdata.h"

namespace DynamicBlockHiF4Quant {
#define FLOAT_OVERFLOW_MODE_CTRL 60
using namespace AscendC;

// Buffer double buffer 数量
constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;

// 基本块尺寸（与 host tiling 保持一致）
constexpr int64_t BLOCK_SIZE_ROW = 64;  // 基本块的高
constexpr int64_t BLOCK_SIZE_COL = 512; // 基本块的宽（按 64*512 粒度切分）

// HiF4 三层缩放架构常量
constexpr int64_t MICRO_BLOCK_SIZE = 4; // Ec: 4*4
constexpr int64_t SUB_BLOCK_SIZE = 8;   // Eb: 8*8
constexpr int64_t BLOCK_SIZE = 64;      // Ea: 64*64

// S1P2 打包：每 2 个 4-bit 元素打包为 1 个 uint8
constexpr int64_t S1P2_PACK_FACTOR = 2;

// 循环参数
constexpr uint16_t LOOP_SIXTEEN = 16;
constexpr uint16_t LOOP_EIGHT = 8;
constexpr uint16_t LOOP_FOUR = 4;
constexpr uint16_t LOOP_TWO = 2;

constexpr uint32_t DIGIT_256 = 256;
constexpr uint32_t DIGIT_128 = 128;
constexpr uint32_t DIGIT_64 = 64;
constexpr uint32_t DIGIT_32 = 32;
constexpr uint32_t DIGIT_16 = 16;
constexpr uint32_t DIGIT_8 = 8;

constexpr float NUMBER_ONE = 1.0;
constexpr float NUMBER_HALF = 0.5;

static constexpr Reg::CastTrait castTraitHalf2Bf16 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

static constexpr AscendC::Reg::CastTrait castTraitE6M2 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};

template <typename T, typename U, AscendC::RoundMode roundMode>
class DynamicBlockHiF4QuantKernel {
public:
    __aicore__ inline DynamicBlockHiF4QuantKernel(const DynamicBlockHiF4QuantTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline ~DynamicBlockHiF4QuantKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2);
    __aicore__ inline void Process();

private:
    // ===== 主流程：搬入 / 计算 / 搬出 =====
    __aicore__ inline void CopyIn(int64_t taskIdx, int64_t rowBlock, int64_t colBlock, int32_t colActual);
    __aicore__ inline void Compute(int64_t rowBlock, int64_t colBlock, int32_t colActual);
    __aicore__ inline void CopyOut(int64_t taskIdx, int64_t rowBlock, int64_t colBlock, int32_t colActual);

    // 以 4*4 为粒度取输入 x 绝对值最大值 -> Vmax16 (16*16 个 BF16)
    __aicore__ inline void ComputeMaxPer4x4(__ubuf__ uint32_t* xAddr, __ubuf__ uint16_t* vmax16Addr, int32_t colActual);

    // 由 Vmax16 计算 8*8 最大值 Vmax8、64*64 最大值 Vmax，并得到 Ea (E6M2) 与 EaUs
    //      EaUs = Ea 每元素复制一份 (8*1 -> 8*2)
    __aicore__ inline void ComputeMaxPer8x8And64x64(__ubuf__ uint16_t* vmax16Addr, __ubuf__ uint16_t* vmax8Addr,
                                                    __ubuf__ uint8_t* eaAddr, __ubuf__ uint16_t* e6m2RecAddr,
                                                    __ubuf__ uint8_t* eaUsAddr);

    // 由 Vmax16/Vmax8/Ea 计算 Eb、Ec、tmpScale（用于后续 x 量化）
    __aicore__ inline void ComputeScaleStageOne(__ubuf__ uint16_t* vmax16Addr, __ubuf__ uint16_t* vmax8Addr,
                                                __ubuf__ uint16_t* e6m2RecAddr, __ubuf__ uint32_t* ebAddr,
                                                __ubuf__ uint8_t* ecAddr, __ubuf__ uint16_t* tmpScaleAddr);

    // 融合计算 scale1 与 scale2，两个均循环 8 次，融合为单循环
    //   - scale1: 由 Eb/Ec/Ea 拼接（-1 轴方向，HIFLOAT4_SCALE 打包，uint32 承载）
    //   - scale2: 由 Eb/Ec/EaUs 拼接（-2 轴方向，HIFLOAT4_SCALE 打包，uint32 承载）
    //             EaUs = Ea 每元素复制一份 (8*1 -> 8*2)
    //             scale2 的循环内仅做位平面累积，循环后做 Interleave/Load/Store
    //   注意：scale2 UB buffer 大小按 uint32 计（InitBuffer 不变），但内部以 uint16 形式使用该空间
    __aicore__ inline void ComputeScaleStageTwo(__ubuf__ uint8_t* eaAddr, __ubuf__ uint8_t* ebAddr,
                                                __ubuf__ uint8_t* ecAddr, __ubuf__ uint16_t* scale1Addr,
                                                __ubuf__ uint16_t* eaUsAddr, __ubuf__ uint16_t* scale2Addr);

    // 由 tmpScale 与输入 x 量化得到 y（S1P2 打包，每 2 个 4-bit -> 1 个 uint8）
    __aicore__ inline void ComputeOutY(__ubuf__ uint32_t* xAddr, __ubuf__ uint16_t* tmpScaleAddr,
                                       __ubuf__ uint32_t* yAddr, int32_t colActual);

    // ===== 内部辅助 =====
    __aicore__ inline void InitParams();
    __aicore__ inline void InitBuffer();
    __aicore__ inline void CalcCoreTaskRange();

private:
    const DynamicBlockHiF4QuantTilingData* tilingData_;
    TPipe* pipe_ = nullptr;

    // ===== TQue / TBuf =====
    // xLocal: VecIn  splitM * splitN * sizeof(X) * Db  -> 64 * 512 * sizeof(T) * 2
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    // yLocal: VecOut splitM * splitN * sizeof(Y) * Db  -> 64 * 512 / 2 * 2 (S1P2 打包)
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    // scale1Local: VecOut blockSizeRow * 8 个 scale * sizeof(uint32_t) * Db (每 64 列 1 个 scale，uint32 承载)
    TQue<QuePosition::VECOUT, DB_BUFFER> scale1Queue_;
    // scale2Local: VecOut 1 * blockSizeCol 个 scale * sizeof(uint32_t) * Db (每 64 行 1 个 scale，uint32 承载)
    TQue<QuePosition::VECOUT, DB_BUFFER> scale2Queue_;

    // 中间 TBuf
    TBuf<TPosition::VECCALC> bufMaxPer4x4_; // Vmax16: 16 * 128 * sizeof(uint16_t)  -> 4 KB
    TBuf<TPosition::VECCALC> bufMaxPer8x8_; // Vmax8 : 8  * 64  * sizeof(uint16_t)  -> 1 KB
    TBuf<TPosition::VECCALC> bufEa_;        // Ea    : 256 * sizeof(uint8_t)        -> 256 B
    TBuf<TPosition::VECCALC> bufEaUs_;      // EaUs  : Ea 每元素复制一份 (8*1 -> 8*2), 16 * sizeof(uint8_t)
    TBuf<TPosition::VECCALC> bufEb_;        // Eb    : 8  * 32  * sizeof(uint8_t)   -> 256 B
    TBuf<TPosition::VECCALC> bufEc_;        // Ec    : 16 * 32  * sizeof(uint8_t)   -> 512 B
    TBuf<TPosition::VECCALC> bufE6m2Rec_;   // E6M2 倒数: 256 * sizeof(uint16_t)
    TBuf<TPosition::VECCALC> bufTmpScale_;  // 量化中间缩放因子: 64 * 512 / 64 * sizeof(uint16_t)

    // ===== GlobalTensor =====
    GlobalTensor<T> xGm_;
    GlobalTensor<uint8_t> yGm_;       // HIFLOAT4 (S1P2 打包)
    GlobalTensor<uint32_t> scale1Gm_; // HIFLOAT4_SCALE (uint32 承载: Ea+Eb+Ec 打包)
    GlobalTensor<uint16_t> scale2Gm_; // HIFLOAT4_SCALE (uint32 承载: Ea+Eb+Ec 打包)

    // ===== Tiling 参数（从 tilingData_ 读入）=====
    int64_t blockIdx_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t blockSizeRow_ = 0;
    int64_t blockSizeCol_ = 0;
    int64_t rowNum_ = 0;
    int64_t colNum_ = 0;
    int64_t rowBlockLoopNum_ = 0;
    int64_t colBlockLoopNum_ = 0;
    int64_t colTileNum_ = 0;
    int64_t normalCoreNum_ = 0;
    int64_t normalCoreBlockNum_ = 0;
    int64_t tailCoreBlockNum_ = 0;

    // ===== 由 scale_alg 决定的 Eb/Ec 比较阈值 =====
    // scale_alg=0(默认): Eb 阈值为 4, Ec 阈值为 2; scale_alg=1: Eb 阈值为 3.75, Ec 阈值为 1.875
    bfloat16_t e8Thresh_ = static_cast<bfloat16_t>(4.0F);
    bfloat16_t e16Thresh_ = static_cast<bfloat16_t>(2.0F);

    // ===== 当前核的任务范围 =====
    int64_t coreTaskStart_ = 0; // 当前核起始任务块号（一维）
    int64_t coreTaskNum_ = 0;   // 当前核处理任务块数量

    // ===== 平台常量 =====
    int64_t ubBlockSize_ = platform::GetUbBlockSize();  // UB 块大小（字节），通常 32
    int64_t oneBlockCountX_ = ubBlockSize_ / sizeof(T); // 一个 UB 块可放 T 元素数
    int64_t oneBlockCountU8_ = ubBlockSize_ / sizeof(uint8_t);

    constexpr static Reg::CastTrait castTrait_ = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, roundMode};
    ;
};

// =====================================================================
// Init：分配 Buffer、计算当前核数据块起始地址、初始化参数
// =====================================================================
template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicBlockHiF4QuantKernel<T, U, roundMode>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale1,
                                                                          GM_ADDR scale2)
{
#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif

    InitParams();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    InitBuffer();
    CalcCoreTaskRange();

    // 设置 GlobalTensor 基地址（合轴后二维 (rowNum_, colNum_) 连续排布）
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    // y 为 HIFLOAT4 (S1P2 打包)，每 2 个 4-bit 元素占 1 字节
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
    // scale1/scale2 为 HIFLOAT4_SCALE，uint32 承载（Ea+Eb+Ec 打包，每元素 4 字节）
    scale1Gm_.SetGlobalBuffer((__gm__ uint32_t*)scale1);
    scale2Gm_.SetGlobalBuffer((__gm__ uint16_t*)scale2);
}

template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicBlockHiF4QuantKernel<T, U, roundMode>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    totalCoreNum_ = tilingData_->totalCoreNum;
    usedCoreNum_ = tilingData_->usedCoreNum;
    ubSize_ = tilingData_->ubSize;
    blockSizeRow_ = tilingData_->blockSizeRow;
    blockSizeCol_ = tilingData_->blockSizeCol;
    rowNum_ = tilingData_->rowNum;
    colNum_ = tilingData_->colNum;
    rowBlockLoopNum_ = tilingData_->rowBlockLoopNum;
    colBlockLoopNum_ = tilingData_->colBlockLoopNum;
    colTileNum_ = tilingData_->colTileNum;
    normalCoreNum_ = tilingData_->normalCoreNum;
    normalCoreBlockNum_ = tilingData_->normalCoreBlockNum;
    tailCoreBlockNum_ = tilingData_->tailCoreBlockNum;

    // 根据 scale_alg 选择 Eb/Ec 比较阈值
    e8Thresh_ = (tilingData_->scaleAlg == 1) ? static_cast<bfloat16_t>(3.75F) : static_cast<bfloat16_t>(4.0F);
    e16Thresh_ = (tilingData_->scaleAlg == 1) ? static_cast<bfloat16_t>(1.875F) : static_cast<bfloat16_t>(2.0F);
}

// 按 64*512 单数据块大小分配 Buffer（DB=2 double buffer）
template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicBlockHiF4QuantKernel<T, U, roundMode>::InitBuffer()
{
    // xLocal: 64 * 512 * sizeof(T) * Db
    pipe_->InitBuffer(inQueue_, DB_BUFFER, blockSizeRow_ * blockSizeCol_ * sizeof(T));
    // yLocal: 64 * 512 / S1P2_PACK_FACTOR * Db (S1P2 打包，2 个元素 1 字节)
    pipe_->InitBuffer(outQueue_, DB_BUFFER, blockSizeRow_ * blockSizeCol_ / S1P2_PACK_FACTOR);
    // scale1Local: 64 * (8 个 scale 元素 + 24 个无效元素) * sizeof(uint32_t) * Db
    pipe_->InitBuffer(scale1Queue_, DB_BUFFER, blockSizeRow_ * (8 + 24) * sizeof(uint32_t));
    // scale2Local: 1 * 512 个 scale 元素 * sizeof(uint32_t) * Db (每 64 行 1 个 scale，1 个 64 行块 -> 1 个，512 列)
    pipe_->InitBuffer(scale2Queue_, DB_BUFFER, blockSizeCol_ * sizeof(uint32_t));

    // 中间 TBuf
    pipe_->InitBuffer(bufMaxPer4x4_, 16 * 128 * sizeof(uint16_t)); // Vmax16: 4 KB
    pipe_->InitBuffer(bufMaxPer8x8_, 8 * 64 * sizeof(uint16_t));   // Vmax8 : 1 KB
    pipe_->InitBuffer(bufEa_, 256 * sizeof(uint8_t));              // Ea    : 256 B
    pipe_->InitBuffer(bufEaUs_, 256 * sizeof(uint8_t));            // EaUs  : Ea 复制一份 8*1 -> 8*2
    pipe_->InitBuffer(bufEb_, 8 * 32 * sizeof(uint8_t));           // Eb    : 256 B
    pipe_->InitBuffer(bufEc_, 16 * 32 * sizeof(uint8_t));          // Ec    : 512 B
    pipe_->InitBuffer(bufE6m2Rec_, 128 * sizeof(uint16_t));        // E6M2 倒数 ：256 B
    pipe_->InitBuffer(bufTmpScale_, 16 * 128 * sizeof(uint16_t));  // tmpScale 用于x的量化：8 KB
}

// 计算当前核处理的一维任务块起止号
// 头核 (blockIdx < normalCoreNum) 处理 normalCoreBlockNum 块，尾核处理 tailCoreBlockNum 块
template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicBlockHiF4QuantKernel<T, U, roundMode>::CalcCoreTaskRange()
{
    if (blockIdx_ < normalCoreNum_) {
        // 头核
        coreTaskStart_ = blockIdx_ * normalCoreBlockNum_;
        coreTaskNum_ = normalCoreBlockNum_;
    } else {
        // 尾核
        coreTaskStart_ = normalCoreNum_ * normalCoreBlockNum_ + (blockIdx_ - normalCoreNum_) * tailCoreBlockNum_;
        coreTaskNum_ = tailCoreBlockNum_;
    }
}

// =====================================================================
// Process：循环处理当前核负责的每个数据块
// =====================================================================
template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicBlockHiF4QuantKernel<T, U, roundMode>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    for (int64_t taskIdx = coreTaskStart_; taskIdx < coreTaskStart_ + coreTaskNum_; ++taskIdx) {
        // 一维任务号 -> (rowBlock, colBlock) 二维坐标
        // rowBlockLoopNum_ 为行方向 64*64 块数，colBlockLoopNum_ 为列方向 512 块数
        int64_t rowBlock = taskIdx / colBlockLoopNum_;
        int64_t colBlock = taskIdx % colBlockLoopNum_;

        // 当前块的列方向实际元素数：尾块用 colTileNum_，否则 blockSizeCol_
        int32_t colActual = (colBlock == colBlockLoopNum_ - 1) ? static_cast<int32_t>(colTileNum_) :
                                                                 static_cast<int32_t>(blockSizeCol_);

        CopyIn(taskIdx, rowBlock, colBlock, colActual);
        Compute(rowBlock, colBlock, colActual);
        CopyOut(taskIdx, rowBlock, colBlock, colActual);
    }
}

// =====================================================================
// CopyIn：搬入一个 64*512 的数据块
// 尾块按实际 colActual 搬入，不足 512 补齐到 512 形成整块
// =====================================================================
template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicBlockHiF4QuantKernel<T, U, roundMode>::CopyIn(int64_t taskIdx, int64_t rowBlock,
                                                                            int64_t colBlock, int32_t colActual)
{
    LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();

    // 当前块在合轴后二维 (rowNum_, colNum_) 中的起始行偏移
    int64_t baseRow = rowBlock * blockSizeRow_;
    int64_t baseCol = colBlock * blockSizeCol_;
    int64_t gmOffset = baseRow * colNum_ + baseCol;
    int64_t rightPad = (blockSizeCol_ - colActual) * sizeof(T) / ubBlockSize_;

    DataCopyExtParams copyParams = {static_cast<uint16_t>(blockSizeRow_), static_cast<uint32_t>(colActual * sizeof(T)),
                                    static_cast<uint32_t>((colNum_ - colActual) * sizeof(T)),
                                    static_cast<uint32_t>(rightPad), 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[gmOffset], copyParams, padParams,
                true); // 使能 enablePreReadIgnoreSync = true, 能较好的提升MTE2的搬运性能，提升算子整体性能

    inQueue_.EnQue(xLocal);
}

// =====================================================================
// Compute：多阶段串联处理一个数据块
// =====================================================================
template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicBlockHiF4QuantKernel<T, U, roundMode>::Compute(int64_t rowBlock, int64_t colBlock,
                                                                             int32_t colActual)
{
    LocalTensor<uint32_t> xLocal = inQueue_.DeQue<uint32_t>();
    __local_mem__ uint32_t* xAddr = (__local_mem__ uint32_t*)xLocal.GetPhyAddr();

    // 分配输出 tensor
    LocalTensor<uint32_t> yLocal = outQueue_.AllocTensor<uint32_t>();
    __local_mem__ uint32_t* yAddr = (__local_mem__ uint32_t*)yLocal.GetPhyAddr();

    LocalTensor<uint16_t> scale1Local = scale1Queue_.AllocTensor<uint16_t>();
    __local_mem__ uint16_t* scale1Addr = (__local_mem__ uint16_t*)scale1Local.GetPhyAddr();

    // scale2 UB buffer 按 uint32 大小分配（InitBuffer 不变），但以 uint16 形式使用
    LocalTensor<uint16_t> scale2Local = scale2Queue_.AllocTensor<uint16_t>();
    __local_mem__ uint16_t* scale2Addr = (__local_mem__ uint16_t*)scale2Local.GetPhyAddr();

    // 取中间 TBuf 地址
    LocalTensor<uint16_t> vmax16Local = bufMaxPer4x4_.Get<uint16_t>();
    auto vmax16Addr = (__local_mem__ uint16_t*)vmax16Local.GetPhyAddr();
    LocalTensor<uint16_t> vmax8Local = bufMaxPer8x8_.Get<uint16_t>();
    auto vmax8Addr = (__local_mem__ uint16_t*)vmax8Local.GetPhyAddr();
    LocalTensor<uint8_t> eaLocal = bufEa_.Get<uint8_t>();
    auto eaAddr = (__local_mem__ uint8_t*)eaLocal.GetPhyAddr();
    LocalTensor<uint8_t> eaUsLocal = bufEaUs_.Get<uint8_t>();
    auto eaUsAddr = (__local_mem__ uint8_t*)eaUsLocal.GetPhyAddr();
    auto eaUsU16Addr = (__local_mem__ uint16_t*)eaUsLocal.GetPhyAddr();
    LocalTensor<uint8_t> ebLocal = bufEb_.Get<uint8_t>();
    auto ebAddr = (__local_mem__ uint8_t*)ebLocal.GetPhyAddr();
    auto ebU32Addr = (__local_mem__ uint32_t*)ebLocal.GetPhyAddr();
    LocalTensor<uint8_t> ecLocal = bufEc_.Get<uint8_t>();
    auto ecAddr = (__local_mem__ uint8_t*)ecLocal.GetPhyAddr();
    LocalTensor<uint16_t> e6m2RecLocal = bufE6m2Rec_.Get<uint16_t>();
    auto e6m2RecAddr = (__local_mem__ uint16_t*)e6m2RecLocal.GetPhyAddr();
    LocalTensor<uint16_t> tmpScaleLocal = bufTmpScale_.Get<uint16_t>();
    auto tmpScaleAddr = (__local_mem__ uint16_t*)tmpScaleLocal.GetPhyAddr();

    // ===== 多阶段串联 =====
    // 以 4*4 为粒度取输入 x 绝对值最大值 -> Vmax16 (16*16)
    ComputeMaxPer4x4(xAddr, vmax16Addr, colActual);

    // 由 Vmax16 计算 8*8 最大值 Vmax8、64*64 最大值 Vmax，并得到 Ea、E6M2_REC、EaUs
    ComputeMaxPer8x8And64x64(vmax16Addr, vmax8Addr, eaAddr, e6m2RecAddr, eaUsAddr);

    // 由 Vmax16/Vmax8/Ea 计算 Eb、Ec、tmpScale
    ComputeScaleStageOne(vmax16Addr, vmax8Addr, e6m2RecAddr, ebU32Addr, ecAddr, tmpScaleAddr);

    // 融合计算 scale1 (-1 轴) 与 scale2 (-2 轴)
    ComputeScaleStageTwo(eaAddr, ebAddr, ecAddr, scale1Addr, eaUsU16Addr, scale2Addr);

    // 由 tmpScale 与输入 x 量化得到 y（S1P2 打包）
    ComputeOutY(xAddr, tmpScaleAddr, yAddr, colActual);

    inQueue_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
    scale1Queue_.EnQue(scale1Local);
    scale2Queue_.EnQue(scale2Local);
}

// =====================================================================
// CopyOut：按实际元素数量搬出 y / scale1 / scale2
// =====================================================================
template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicBlockHiF4QuantKernel<T, U, roundMode>::CopyOut(int64_t taskIdx, int64_t rowBlock,
                                                                             int64_t colBlock, int32_t colActual)
{
    LocalTensor<uint8_t> yLocal = outQueue_.DeQue<uint8_t>();
    LocalTensor<uint32_t> scale1Local = scale1Queue_.DeQue<uint32_t>();
    // scale2 UB 以 uint16 形式使用，DeQue 也按 uint16 取
    LocalTensor<uint16_t> scale2Local = scale2Queue_.DeQue<uint16_t>();

    // 当前块在合轴后二维 (rowNum_, colNum_) 中的起始偏移
    int64_t baseRow = rowBlock * blockSizeRow_;
    int64_t baseCol = colBlock * blockSizeCol_;
    int64_t baseXGmOffset = baseRow * colNum_ + baseCol;

    // ----- y 搬出 (HIFLOAT4 S1P2 打包，每 2 个元素 1 字节) -----
    // yGm 按 S1P2 打包存储：每行 colNum_/2 字节
    int64_t yGmOffset = baseXGmOffset / S1P2_PACK_FACTOR;
    int32_t outBlockLen = colActual / S1P2_PACK_FACTOR * sizeof(uint8_t);
    int32_t ySrcStride = ((blockSizeCol_ - colActual) / S1P2_PACK_FACTOR * sizeof(uint8_t) / ubBlockSize_);
    int32_t yDstStride = (colNum_ - colActual) / S1P2_PACK_FACTOR * sizeof(uint8_t);
    DataCopyExtParams yCopyParams = {static_cast<uint16_t>(blockSizeRow_), static_cast<uint32_t>(outBlockLen),
                                     static_cast<uint32_t>(ySrcStride), static_cast<uint32_t>(yDstStride), 0};
    DataCopyPad(yGm_[yGmOffset], yLocal, yCopyParams);

    // ----- scale1 搬出 (-1 轴方向，每 64 列 1 个 scale，uint32 承载) -----
    // scale1 GM 布局：合轴后 (rowNum_, colNum_/64, 1)，每元素 sizeof(uint32_t)
    // 每行 colNum_/64 个 scale，本块贡献 colActual/64 个
    int64_t scale1ColNum = colNum_ / BLOCK_SIZE;
    int64_t scale1GmRowOffset = baseRow * scale1ColNum;
    int64_t scale1GmColOffset = baseCol / BLOCK_SIZE;
    int64_t scale1OutNum = colActual / BLOCK_SIZE; // 本块每行 scale 数量
    DataCopyExtParams scale1CopyParams = {
        static_cast<uint16_t>(blockSizeRow_), static_cast<uint32_t>(scale1OutNum * sizeof(uint32_t)), 3,
        static_cast<uint32_t>((scale1ColNum - scale1OutNum) * sizeof(uint32_t)),
        0}; // scale1 在UB占用的空间是64*(32+96) 这里srcstride指的就是这个96B的stride=3
    DataCopyPad(scale1Gm_[scale1GmRowOffset + scale1GmColOffset], scale1Local, scale1CopyParams);

    // ----- scale2 搬出 (-2 轴方向，每 64 行 1 个 scale，uint32 承载) -----
    // scale2 GM 布局：合轴后 (rowNum_/64, colNum_, 1)，每元素 sizeof(uint32_t)
    // 本块为 1 个 64 行块、colActual 列
    int64_t scale2RowIdx = rowBlock; // 当前 64 行块的行号
    int64_t scale2GmOffset = scale2RowIdx * colNum_ * 2 + baseCol;
    int64_t scale2OutNum = colActual;
    DataCopyExtParams scale2CopyParams = {
        static_cast<uint16_t>(2),
        static_cast<uint32_t>(scale2OutNum * sizeof(uint16_t)), // scale2搬运分两块，第一块搬运EA+EB组合体，第二块搬运EC
        static_cast<uint32_t>((blockSizeCol_ - scale2OutNum) * sizeof(uint16_t) / ubBlockSize_),
        static_cast<uint32_t>((colNum_ - scale2OutNum) * sizeof(uint16_t)), 0};
    DataCopyPad(scale2Gm_[scale2GmOffset], scale2Local, scale2CopyParams);

    outQueue_.FreeTensor(yLocal);
    scale1Queue_.FreeTensor(scale1Local);
    scale2Queue_.FreeTensor(scale2Local);
}

// 以 4*4 为粒度取输入 x 绝对值最大值 -> Vmax16 (16*16 个 BF16)
template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicBlockHiF4QuantKernel<T, U, roundMode>::ComputeMaxPer4x4(__ubuf__ uint32_t* xAddr,
                                                                                      __ubuf__ uint16_t* vmax16Addr,
                                                                                      int32_t colActual)
{
    // 输入: xAddr 指向 64*512 的 x 数据（BF16/FP16）
    // 输出: vmax16Addr 指向 16*128 个 uint16_t（每 4*4 取绝对值最大值，512 列 -> 128 个）
    uint32_t LOOP_FOUR_STEP_128 = LOOP_FOUR * DIGIT_128;

    __VEC_SCOPE__
    {
        Reg::MaskReg maskU16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        Reg::RegTensor<uint16_t> vRegX_0, vRegX_1, vRegX_2, vRegX_3;
        Reg::RegTensor<uint16_t> expMaskU16;
        Reg::RegTensor<uint16_t> vMax, vMaxTmp;
        Reg::RegTensor<uint16_t> vMax_0, vMax_1;

        Reg::Duplicate(expMaskU16, 0x7fff);

        for (uint16_t i = 0; i < LOOP_SIXTEEN; i++) {
            Reg::Duplicate(vMax, 0x0000);
            for (uint16_t j = 0; j < LOOP_FOUR; j++) {
                Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B32>(
                    (Reg::RegTensor<uint32_t>&)vRegX_0, (Reg::RegTensor<uint32_t>&)vRegX_1,
                    xAddr + i * LOOP_FOUR * DIGIT_256 + j * DIGIT_256);
                Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_DINTLV_B32>(
                    (Reg::RegTensor<uint32_t>&)vRegX_2, (Reg::RegTensor<uint32_t>&)vRegX_3,
                    xAddr + i * LOOP_FOUR * DIGIT_256 + j * DIGIT_256 + DIGIT_128);

                Reg::And(vRegX_0, vRegX_0, expMaskU16, maskU16);
                Reg::And(vRegX_1, vRegX_1, expMaskU16, maskU16);
                Reg::And(vRegX_2, vRegX_2, expMaskU16, maskU16);
                Reg::And(vRegX_3, vRegX_3, expMaskU16, maskU16);

                Reg::Max(vRegX_0, vRegX_0, vRegX_1, maskU16);
                Reg::Max(vRegX_2, vRegX_2, vRegX_3, maskU16);

                Reg::DeInterleave(vRegX_1, vRegX_3, vRegX_0, vRegX_2);
                Reg::Max(vMaxTmp, vRegX_1, vRegX_3, maskU16);
                Reg::Max(vMax, vMax, vMaxTmp, maskU16);
            }
            if constexpr (IsSameType<T, half>::value) {
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)vMax,
                                                             (Reg::RegTensor<T>&)vMax, maskU16);
            }
            Reg::StoreAlign<uint16_t>(vmax16Addr + i * DIGIT_128, vMax, maskU16);
        }
    }
}

// 由 Vmax16 计算 8*8 最大值 Vmax8、64*64 最大值 Vmax，并得到 Ea (E6M2)、E6M2_REC 与 EaUs
//      EaUs = Ea 每元素复制一份 (8*1 -> 8*2)
template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicBlockHiF4QuantKernel<T, U, roundMode>::ComputeMaxPer8x8And64x64(
    __ubuf__ uint16_t* vmax16Addr, __ubuf__ uint16_t* vmax8Addr, __ubuf__ uint8_t* eaAddr,
    __ubuf__ uint16_t* e6m2RecAddr, __ubuf__ uint8_t* eaUsAddr)
{
    // 输入: vmax16Addr (16*128 个 uint16_t)
    // 输出: vmax8Addr  (8*64 个 uint16_t, 每 2 个 Vmax16 取最大)
    //       eaAddr     (256 个 uint8_t, E6M2 编码的 Ea)
    //       e6m2RecAddr(256 个 uint16_t, E6M2 倒数 BF16)
    //       eaUsAddr   (16 个 uint8_t, Ea 每元素复制一份 8*1 -> 8*2)
    uint32_t LOOP_FOUR_STEP_128 = LOOP_FOUR * DIGIT_128;

    __VEC_SCOPE__
    {
        static constexpr AscendC::MicroAPI::CastTrait castTraitReciprocalCast = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

        Reg::MaskReg maskU16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskU8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        Reg::RegTensor<uint16_t> vReg_0, vReg_1, vReg_2, vReg_3;
        Reg::RegTensor<uint16_t> vMax_8, vMax_64;
        Reg::RegTensor<uint16_t> zeroReg;
        Reg::RegTensor<uint16_t> vRegBf16E6M2ScaleBias, vb16_Ea_rec_bf16;
        Reg::RegTensor<uint16_t> vRegE6m2;
        Reg::RegTensor<uint8_t> vregEa;

        Reg::Duplicate(vRegBf16E6M2ScaleBias, 0x3e12);
        Reg::Duplicate(vMax_64, 0);
        Reg::Duplicate(zeroReg, 0);

        for (uint16_t j = 0; j < LOOP_FOUR; j++) {
            Reg::LoadAlign<uint16_t>(vReg_0, vmax16Addr + j * LOOP_FOUR_STEP_128 + 0 * DIGIT_128);
            Reg::LoadAlign<uint16_t>(vReg_1, vmax16Addr + j * LOOP_FOUR_STEP_128 + 1 * DIGIT_128);
            Reg::LoadAlign<uint16_t>(vReg_2, vmax16Addr + j * LOOP_FOUR_STEP_128 + 2 * DIGIT_128);
            Reg::LoadAlign<uint16_t>(vReg_3, vmax16Addr + j * LOOP_FOUR_STEP_128 + 3 * DIGIT_128);

            Reg::Max(vReg_0, vReg_0, vReg_1, maskU16);
            Reg::Max(vReg_2, vReg_2, vReg_3, maskU16);

            Reg::DeInterleave(vReg_1, vReg_3, vReg_0, vReg_2);
            Reg::Max(vMax_8, vReg_1, vReg_3, maskU16);
            Reg::StoreAlign<uint16_t>(vmax8Addr + j * DIGIT_128, vMax_8, maskU16);
            Reg::Max(vMax_64, vMax_64, vMax_8, maskU16);
        }
        Reg::Interleave(vMax_64, zeroReg, vMax_64, zeroReg);
        Reg::Max(vMax_64, vMax_64, zeroReg, maskU16);
        Reg::ReduceMaxWithDataBlock(vMax_64, vMax_64, maskU16);
        Reg::Mul((Reg::RegTensor<bfloat16_t>&)vMax_64, (Reg::RegTensor<bfloat16_t>&)vMax_64,
                 (Reg::RegTensor<bfloat16_t>&)vRegBf16E6M2ScaleBias, maskU16);
        Reg::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((Reg::RegTensor<fp8_e6m2_t>&)vRegE6m2,
                                                         (Reg::RegTensor<bfloat16_t>&)vMax_64,
                                                         maskU16); // 只有前8位才是有效值

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(vregEa, vRegE6m2);
        Reg::StoreAlign<uint8_t>(
            eaAddr, vregEa, maskU8); // 这里编译器直接存的是vRegE6m2，使用了PK_B16模式存出去的，与编码不符，但效果一致
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(
            eaUsAddr, vregEa, vregEa, maskU8); // 这里交织搬出时多了一条MOV指令，用一个寄存器复制了vregEa的内容

        Reg::ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitReciprocalCast>(
            (Reg::RegTensor<bfloat16_t>&)vb16_Ea_rec_bf16, (Reg::RegTensor<fp8_e6m2_t>&)vRegE6m2, maskU8);
        Reg::StoreAlign<uint16_t>(e6m2RecAddr, vb16_Ea_rec_bf16, maskU16);
    }
}

// 由 Vmax16/Vmax8/Ea 计算 Eb、Ec、tmpScale
template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicBlockHiF4QuantKernel<T, U, roundMode>::ComputeScaleStageOne(
    __ubuf__ uint16_t* vmax16Addr, __ubuf__ uint16_t* vmax8Addr, __ubuf__ uint16_t* e6m2RecAddr,
    __ubuf__ uint32_t* ebAddr, __ubuf__ uint8_t* ecAddr, __ubuf__ uint16_t* tmpScaleAddr)
{
    // 输入: vmax16Addr, vmax8Addr, e6m2RecAddr
    // 输出: ebAddr      (8*32 个 uint8_t, E1M0 编码的 Eb)
    //       ecAddr      (16*32 个 uint8_t, E1M0 编码的 Ec)
    //       tmpScaleAddr(量化中间缩放因子)
    uint32_t LOOP_TWO_STEP_256 = LOOP_TWO * DIGIT_256;
    uint32_t LOOP_TWO_STEP_128 = LOOP_TWO * DIGIT_128;
    uint32_t LOOP_TWO_STEP_32 = LOOP_TWO * DIGIT_32;

    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> vRegX_0, vRegX_1, vRegX_2, vRegX_3;
        Reg::RegTensor<bfloat16_t> vb16_Ea_rec_bf16, vMax8, vMax4, vRegTmpScale;
        Reg::RegTensor<bfloat16_t> vRegNumOne, vRegNumHalf;

        Reg::MaskReg maskU16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg mRegE8Comp;
        Reg::MaskReg mRegE8Back;
        Reg::MaskReg mRegE16Comp;

        Reg::Duplicate(vRegNumOne, static_cast<bfloat16_t>(NUMBER_ONE));
        Reg::Duplicate(vRegNumHalf, static_cast<bfloat16_t>(NUMBER_HALF));

        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_E2B_B16>((Reg::RegTensor<uint16_t>&)vb16_Ea_rec_bf16, e6m2RecAddr);

        for (uint16_t i = 0; i < LOOP_EIGHT; i++) {
            Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_US_B16>((Reg::RegTensor<uint16_t>&)vMax8,
                                                                 vmax8Addr + i * DIGIT_64);
            Reg::Mul(vMax8, vMax8, vb16_Ea_rec_bf16, maskU16);
            Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(mRegE8Comp, vMax8, e8Thresh_, maskU16);
            Reg::Select(vMax8, vRegNumHalf, vRegNumOne, mRegE8Comp);
            Reg::Pack<Reg::HighLowPart::LOWEST>(mRegE8Back, mRegE8Comp);
            Reg::StoreAlign<uint32_t, Reg::MaskDist::DIST_PACK>(ebAddr + i * DIGIT_8,
                                                                mRegE8Back); // 存 EB 的 bit 位  64 bits
            for (uint16_t j = 0; j < LOOP_TWO; j++) { // 这里应该手动UnRoll出来好一些？
                Reg::LoadAlign<uint16_t>((Reg::RegTensor<uint16_t>&)vMax4,
                                         vmax16Addr + i * LOOP_TWO_STEP_128 + j * DIGIT_128);
                Reg::Mul(vMax4, vMax4, vb16_Ea_rec_bf16, maskU16);
                Reg::Mul(vMax4, vMax4, vMax8, maskU16);
                Reg::CompareScalar<bfloat16_t, CMPMODE::GE>(mRegE16Comp, vMax4, e16Thresh_, maskU16);
                Reg::StoreAlign<uint8_t, Reg::MaskDist::DIST_PACK>(ecAddr + i * LOOP_TWO_STEP_32 + j * DIGIT_32,
                                                                   mRegE16Comp); // 存 EC 的 bit 位 128bits
                Reg::Select(vMax4, vRegNumHalf, vRegNumOne, mRegE16Comp);
                Reg::Mul(vMax4, vMax4, vMax8, maskU16);
                Reg::Mul(vRegTmpScale, vMax4, vb16_Ea_rec_bf16, maskU16);
                Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(
                    tmpScaleAddr + i * LOOP_TWO_STEP_256 + j * DIGIT_256, (Reg::RegTensor<uint16_t>&)vRegTmpScale,
                    (Reg::RegTensor<uint16_t>&)vRegTmpScale, maskU16); // 存 tmpScale 128 * 2 = 256 个数
            }
        }
    }
}

// 融合计算 scale1 与 scale2：两个均循环 8 次，融合为单循环
//   scale1: 由 Ea/Eb/Ec 拼接 (-1 轴方向，HIFLOAT4_SCALE uint32 承载)
//   scale2: 由 Eb/Ec/EaUs 拼接 (-2 轴方向，HIFLOAT4_SCALE uint32 承载)
//     EaUs = Ea 每元素复制一份 (8*1 -> 8*2)
//     scale2 的循环内做位平面累积 (Load mask + Add + ShiftLeft)，循环后做 Interleave/Load/Store
//   注意：scale2 UB buffer 按 uint32 大小分配（InitBuffer 不变），但内部以 uint16 形式使用该空间
template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicBlockHiF4QuantKernel<T, U, roundMode>::ComputeScaleStageTwo(
    __ubuf__ uint8_t* eaAddr, __ubuf__ uint8_t* ebAddr, __ubuf__ uint8_t* ecAddr, __ubuf__ uint16_t* scale1Addr,
    __ubuf__ uint16_t* eaUsAddr, __ubuf__ uint16_t* scale2Addr)
{
    uint32_t LOOP_TWO_STEP_256 = LOOP_TWO * DIGIT_256;
    uint32_t LOOP_TWO_STEP_32 = LOOP_TWO * DIGIT_32;
    uint32_t LOOP_EIGHT_STEP_32 = LOOP_EIGHT * DIGIT_32;

    __VEC_SCOPE__
    {
        // ===== scale1 寄存器 =====
        Reg::RegTensor<uint8_t> vRegEa, vRegEb, vRegEaEb, vRegEc_0, vRegEc_1;
        // ===== scale2 寄存器（加 S2 后缀避免与 scale1 冲突）=====
        Reg::RegTensor<uint8_t> vRegU8_S2;
        Reg::RegTensor<uint8_t> vRegEa_S2, vRegEb_S2, vRegEcLow_S2, vRegEcHigh_S2;
        Reg::RegTensor<uint16_t> vRegEaEb_0_S2, vRegEaEb_1_S2, vRegEc_0_S2, vRegEc_1_S2;
        Reg::MaskReg mRegEb_S2, mRegEc_0_S2, mRegEc_1_S2;

        Reg::MaskReg maskU16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskU8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        // scale1: Ea 循环外 Load 一次
        Reg::LoadAlign<uint8_t, Reg::LoadDist::DIST_BLK>(vRegEa, eaAddr);
        // scale2: 常量 0x01 用于 mask 累积
        Reg::Duplicate(vRegU8_S2, 0x01);
        Reg::Duplicate(vRegEb_S2, 0x00);
        Reg::Duplicate(vRegEcLow_S2, 0x00);
        Reg::Duplicate(vRegEcHigh_S2, 0x00);

        for (uint16_t i = 0; i < LOOP_EIGHT; i++) {
            // ----- scale1 循环体 (原 VF4) -----
            Reg::LoadAlign<uint8_t, Reg::LoadDist::DIST_BLK>(vRegEb, ebAddr + i * DIGIT_32);
            Reg::Interleave<uint8_t, Reg::HighLowPart::LOWEST>(vRegEaEb, vRegEa, vRegEb);
            Reg::LoadAlign<uint8_t, Reg::LoadDist::DIST_BLK>(vRegEc_0, ecAddr + i * LOOP_TWO_STEP_32 + 0 * DIGIT_32);
            Reg::LoadAlign<uint8_t, Reg::LoadDist::DIST_BLK>(vRegEc_1, ecAddr + i * LOOP_TWO_STEP_32 + 1 * DIGIT_32);
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(
                scale1Addr + i * LOOP_TWO_STEP_256 + 0 * DIGIT_256, (Reg::RegTensor<uint16_t>&)vRegEaEb,
                (Reg::RegTensor<uint16_t>&)vRegEc_0, maskU16);
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(
                scale1Addr + i * LOOP_TWO_STEP_256 + 1 * DIGIT_256, (Reg::RegTensor<uint16_t>&)vRegEaEb,
                (Reg::RegTensor<uint16_t>&)vRegEc_1, maskU16);

            // ----- scale2 循环体 (原 VF5 循环内部) -----
            // Load
            Reg::LoadAlign<uint8_t, Reg::MaskDist::DIST_US>(mRegEb_S2, ebAddr + 0 * DIGIT_32 + i * DIGIT_32);
            Reg::LoadAlign<uint8_t, Reg::MaskDist::DIST_US>(mRegEc_0_S2,
                                                            ecAddr + 0 * LOOP_EIGHT_STEP_32 + i * DIGIT_32);
            Reg::LoadAlign<uint8_t, Reg::MaskDist::DIST_US>(mRegEc_1_S2,
                                                            ecAddr + 1 * LOOP_EIGHT_STEP_32 + i * DIGIT_32);
            // Add
            Reg::Add<uint8_t, Reg::MaskMergeMode::MERGING>(vRegEb_S2, vRegEb_S2, vRegU8_S2, mRegEb_S2);
            Reg::Add<uint8_t, Reg::MaskMergeMode::MERGING>(vRegEcLow_S2, vRegEcLow_S2, vRegU8_S2, mRegEc_0_S2);
            Reg::Add<uint8_t, Reg::MaskMergeMode::MERGING>(vRegEcHigh_S2, vRegEcHigh_S2, vRegU8_S2, mRegEc_1_S2);
            // ShiftLeft vRegU8_S2: 0x01 -> 0x02 -> 0x04 -> ... (每轮翻倍)
            Reg::ShiftLefts(vRegU8_S2, vRegU8_S2, static_cast<int16_t>(1), maskU8);
        }

        // ----- scale2 循环后重排与存储 (原 VF5 循环外) -----
        Reg::Interleave<uint8_t, Reg::HighLowPart::LOWEST>(vRegEb_S2, vRegEb_S2, vRegEb_S2);
        Reg::Interleave((Reg::RegTensor<uint8_t>&)vRegEc_0_S2, (Reg::RegTensor<uint8_t>&)vRegEc_1_S2, vRegEcLow_S2,
                        vRegEcHigh_S2);
        Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_E2B_B16>((Reg::RegTensor<uint16_t>&)vRegEa_S2, eaUsAddr);
        Reg::Interleave((Reg::RegTensor<uint8_t>&)vRegEaEb_0_S2, (Reg::RegTensor<uint8_t>&)vRegEaEb_1_S2, vRegEa_S2,
                        vRegEb_S2);
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(scale2Addr + 0 * DIGIT_256, vRegEaEb_0_S2,
                                                                  vRegEaEb_0_S2, maskU16);
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(scale2Addr + 1 * DIGIT_256, vRegEaEb_1_S2,
                                                                  vRegEaEb_1_S2, maskU16);
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(scale2Addr + 2 * DIGIT_256, vRegEc_0_S2, vRegEc_0_S2,
                                                                  maskU16);
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(scale2Addr + 3 * DIGIT_256, vRegEc_1_S2, vRegEc_1_S2,
                                                                  maskU16);
    }
}

// VF6: 由 tmpScale 与输入 x 量化得到 y（S1P2 打包，每 2 个 4-bit -> 1 个 uint8）
template <typename T, typename U, AscendC::RoundMode roundMode>
__aicore__ inline void DynamicBlockHiF4QuantKernel<T, U, roundMode>::ComputeOutY(__ubuf__ uint32_t* xAddr,
                                                                                 __ubuf__ uint16_t* tmpScaleAddr,
                                                                                 __ubuf__ uint32_t* yAddr,
                                                                                 int32_t colActual)
{
    // 输入: xAddr (64*512), tmpScaleAddr (量化缩放因子)
    // 输出: yAddr  (64*512/2 个 uint8_t, S1P2 打包)
    uint32_t LOOP_FOUR_STEP_64 = LOOP_FOUR * DIGIT_64;
    uint32_t LOOP_FOUR_STEP_256 = LOOP_FOUR * DIGIT_256;

    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> vRegS_0, vRegS_1, vRegS_2, vRegS_3;
        Reg::RegTensor<uint16_t> vRegX_0, vRegX_1, vRegX_2, vRegX_3;
        Reg::RegTensor<bfloat16_t> vRegOutBf16_0, vRegOutBf16_1, vRegOutBf16_2, vRegOutBf16_3;
        Reg::RegTensor<fp4x2_e1m2_t> vRegOut_0, vRegOut_1, vRegOut_2, vRegOut_3;

        Reg::MaskReg maskU16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        for (uint16_t i = 0; i < LOOP_SIXTEEN; i++) {
            Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_US_B16>((Reg::RegTensor<uint16_t>&)vRegS_0,
                                                                 tmpScaleAddr + i * LOOP_FOUR_STEP_64 + 0 * DIGIT_64);
            Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_US_B16>((Reg::RegTensor<uint16_t>&)vRegS_1,
                                                                 tmpScaleAddr + i * LOOP_FOUR_STEP_64 + 1 * DIGIT_64);
            Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_US_B16>((Reg::RegTensor<uint16_t>&)vRegS_2,
                                                                 tmpScaleAddr + i * LOOP_FOUR_STEP_64 + 2 * DIGIT_64);
            Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_US_B16>((Reg::RegTensor<uint16_t>&)vRegS_3,
                                                                 tmpScaleAddr + i * LOOP_FOUR_STEP_64 + 3 * DIGIT_64);

            for (uint16_t j = 0; j < LOOP_FOUR; j++) {
                Reg::LoadAlign<uint32_t>((Reg::RegTensor<uint32_t>&)vRegX_0,
                                         xAddr + i * LOOP_FOUR_STEP_256 + j * LOOP_FOUR_STEP_64 + 0 * DIGIT_64);
                Reg::LoadAlign<uint32_t>((Reg::RegTensor<uint32_t>&)vRegX_1,
                                         xAddr + i * LOOP_FOUR_STEP_256 + j * LOOP_FOUR_STEP_64 + 1 * DIGIT_64);
                Reg::LoadAlign<uint32_t>((Reg::RegTensor<uint32_t>&)vRegX_2,
                                         xAddr + i * LOOP_FOUR_STEP_256 + j * LOOP_FOUR_STEP_64 + 2 * DIGIT_64);
                Reg::LoadAlign<uint32_t>((Reg::RegTensor<uint32_t>&)vRegX_3,
                                         xAddr + i * LOOP_FOUR_STEP_256 + j * LOOP_FOUR_STEP_64 + 3 * DIGIT_64);

                if constexpr (IsSameType<T, half>::value) {
                    Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)vRegX_0,
                                                                 (Reg::RegTensor<T>&)vRegX_0, maskU16);
                    Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)vRegX_1,
                                                                 (Reg::RegTensor<T>&)vRegX_1, maskU16);
                    Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)vRegX_2,
                                                                 (Reg::RegTensor<T>&)vRegX_2, maskU16);
                    Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>((Reg::RegTensor<bfloat16_t>&)vRegX_3,
                                                                 (Reg::RegTensor<T>&)vRegX_3, maskU16);
                }

                Reg::Mul((Reg::RegTensor<bfloat16_t>&)vRegOutBf16_0, (Reg::RegTensor<bfloat16_t>&)vRegX_0, vRegS_0,
                         maskU16);
                Reg::Mul((Reg::RegTensor<bfloat16_t>&)vRegOutBf16_1, (Reg::RegTensor<bfloat16_t>&)vRegX_1, vRegS_1,
                         maskU16);
                Reg::Mul((Reg::RegTensor<bfloat16_t>&)vRegOutBf16_2, (Reg::RegTensor<bfloat16_t>&)vRegX_2, vRegS_2,
                         maskU16);
                Reg::Mul((Reg::RegTensor<bfloat16_t>&)vRegOutBf16_3, (Reg::RegTensor<bfloat16_t>&)vRegX_3, vRegS_3,
                         maskU16);

                Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vRegOut_0, (Reg::RegTensor<bfloat16_t>&)vRegOutBf16_0,
                                                                maskU16);
                Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vRegOut_1, (Reg::RegTensor<bfloat16_t>&)vRegOutBf16_1,
                                                                maskU16);
                Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vRegOut_2, (Reg::RegTensor<bfloat16_t>&)vRegOutBf16_2,
                                                                maskU16);
                Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vRegOut_3, (Reg::RegTensor<bfloat16_t>&)vRegOutBf16_3,
                                                                maskU16);

                Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_PACK4_B32>(
                    yAddr + i * DIGIT_256 + j * DIGIT_64 + 0 * DIGIT_16, (Reg::RegTensor<uint32_t>&)vRegOut_0, maskU16);
                Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_PACK4_B32>(
                    yAddr + i * DIGIT_256 + j * DIGIT_64 + 1 * DIGIT_16, (Reg::RegTensor<uint32_t>&)vRegOut_1, maskU16);
                Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_PACK4_B32>(
                    yAddr + i * DIGIT_256 + j * DIGIT_64 + 2 * DIGIT_16, (Reg::RegTensor<uint32_t>&)vRegOut_2, maskU16);
                Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_PACK4_B32>(
                    yAddr + i * DIGIT_256 + j * DIGIT_64 + 3 * DIGIT_16, (Reg::RegTensor<uint32_t>&)vRegOut_3, maskU16);
            }
        }
    }
}

} // namespace DynamicBlockHiF4Quant

#endif // DYNAMIC_BLOCK_HI_F4_QUANT_KERNEL_H_
