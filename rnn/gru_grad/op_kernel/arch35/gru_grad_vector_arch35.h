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
 * \file gru_grad_vector.h
 * \brief AIV 段：FP32 门控梯度链 + 偏置规约 + 转置物化与输出回投。
 *
 * 本文件被 #include 进 GruGradKernel<DTYPE> 类体。
 *
 * 要点：
 *  - CopyInRow/CopyOutRow 多行搬运的 UB 侧行距按 32B 块表达；blockLen 32B 对齐时走单次
 *    多行快路径，非对齐尾块退化为逐行搬运。
 *  - AccumulateDhPrev 读 dhPrevF32（AIC Fixpipe 直写），与 dhFromHGm 相加后 round 回
 *    输入 dtype 写 dhPrevWs。
 *  - TransposeGiGhT：dGi/dGh [TS,3H] -> [3H,TS] 物化（dW 两路 TransA GEMM 的 A 侧转置，
 *    硬件无 A 侧镜像载入），16 行分组 + 列分块双 DataCopyPad 完成。
 *  - ConvertOutputs：dwI/dwH/dx 的 fp32 workspace -> 输入 dtype 回投。
 */

// ---- 核内管线事件（基线同名同义）------------------------------------------------------------
// VF 寄存器级原语（vgather2 转置用）：AscendC::Reg 由 kernel_operator.h 在 __NPU_ARCH__==3510 下引入
__aicore__ inline void SyncM2toV()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
};

__aicore__ inline void SyncVtoM3()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
};

__aicore__ inline void SyncVtoM2()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
};

__aicore__ inline void SyncM3toV()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventId);
    WaitFlag<HardEvent::MTE3_V>(eventId);
}

__aicore__ inline void SyncM3toM2()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);
};

__aicore__ inline void SyncM2toM3()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventId);
    WaitFlag<HardEvent::MTE2_MTE3>(eventId);
};

// ---- GM<->UB 行搬运（行距修复版）------------------------------------------------------------
// 语义：把 GM 中以 gmStride（元素）为行距的 rows 行、每行 length 个 T 元素，
//       搬到 UB 中以 rowAlign（元素，32B 对齐垫后行距）为行距的 tile。
// fp16 时先落入本 buffer 后半的 half 暂存区（fp32 偏移 allocLength），再 Cast 回 fp32 布局保持。
template <typename T>
__aicore__ inline void CopyInRow(GlobalTensor<T>& gm, const LocalTensor<float>& ub, int64_t gmOffset, int64_t rows,
                                 int64_t length, int64_t rowAlign, int64_t gmStride)
{
    constexpr int64_t dSize = static_cast<int64_t>(sizeof(T));
    const int64_t blockLen = length * dSize;
    const int64_t pitchBytes = rowAlign * static_cast<int64_t>(sizeof(float)); // fp32 布局行距（字节）
    if constexpr (sizeof(T) == 2) {
        LocalTensor<float> ubW = ub; // 视图拷贝：ReinterpretCast 需要非 const 接收者
        // half 暂存区基址：跳过前 allocLength 个 fp32（= allocLength*4 字节，换算成 T 元素）——16KB buffer 后半
        auto tmpUb = ubW.template ReinterpretCast<T>()[allocLength * FLOAT_BYTES / dSize];
        if (blockLen % BLOCK_BYTES == 0) {
            // 快路径：块长 32B 对齐，UB 行距可精确表达（(行距-块长)/32 个 32B 块）
            DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(blockLen),
                                 static_cast<int64_t>(gmStride * dSize), (pitchBytes - blockLen) / BLOCK_BYTES, 0);
            DataCopyPadExtParams<T> pp(true, 0, 0, 0);
            DataCopyPad(tmpUb, gm[gmOffset], cp, pp);
        } else {
            // 尾块非对齐：逐行搬运（行距缺陷只在多行 + 非对齐时存在，逐行构造闭合）
            for (int64_t r = 0; r < rows; ++r) {
                DataCopyExtParams cp(1, static_cast<uint32_t>(blockLen), 0, 0, 0);
                DataCopyPadExtParams<T> pp(true, 0, 0, 0);
                DataCopyPad(tmpUb[r * (pitchBytes / dSize)], gm[gmOffset + r * (gmStride + length)], cp, pp);
            }
        }
        SyncM2toV();
        // 布局保持的位置逐元素 Cast：fp32 tile 行距 pitchBytes 与 half 暂存行距一致
        AscendC::Cast(ub, tmpUb, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(rows * rowAlign));
        PipeBarrier<PIPE_V>();
    } else {
        if (blockLen % BLOCK_BYTES == 0) {
            DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(blockLen),
                                 static_cast<int64_t>(gmStride * dSize), (pitchBytes - blockLen) / BLOCK_BYTES, 0);
            DataCopyPadExtParams<T> pp(true, 0, 0, 0);
            DataCopyPad(ub, gm[gmOffset], cp, pp);
        } else {
            for (int64_t r = 0; r < rows; ++r) {
                DataCopyExtParams cp(1, static_cast<uint32_t>(blockLen), 0, 0, 0);
                DataCopyPadExtParams<T> pp(true, 0, 0, 0);
                DataCopyPad(ub[rowAlign * r], gm[gmOffset + r * (gmStride + length)], cp, pp);
            }
        }
    }
}

// UB（fp32 布局，行距 rowAlign 个 fp32 元素）-> GM（行距 gmStride 个 T 元素）。
// fp16 时先在 ubTmp2 的 half 视图做位置保持 Cast，再按 half 行距搬运。
template <typename T>
__aicore__ inline void CopyOutRow(GlobalTensor<T>& gm, const LocalTensor<float>& ub, int64_t gmOffset, int64_t rows,
                                  int64_t length, int64_t gmStride, int64_t rowAlign, bool vSyncPreDone = false)
{
    constexpr int64_t dSize = static_cast<int64_t>(sizeof(T));
    const int64_t blockLen = length * dSize;
    const int64_t pitchBytes = rowAlign * static_cast<int64_t>(sizeof(float));
    if constexpr (sizeof(T) == 2) {
        LocalTensor<float> ubW = ub;
        auto tmpUb = ubW.template ReinterpretCast<T>();
        AscendC::Cast(tmpUb, ub, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(rows * rowAlign));
        SyncVtoM3();
        const int64_t halfPitch = pitchBytes / dSize; // half 元素行距
        if (blockLen % BLOCK_BYTES == 0) {
            DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(blockLen),
                                 (pitchBytes - blockLen) / BLOCK_BYTES, static_cast<int64_t>(gmStride * dSize), 0);
            DataCopyPad(gm[gmOffset], tmpUb, cp);
        } else {
            for (int64_t r = 0; r < rows; ++r) {
                DataCopyExtParams cp(1, static_cast<uint32_t>(blockLen), 0, 0, 0);
                DataCopyPad(gm[gmOffset + r * (gmStride + length)], tmpUb[r * halfPitch], cp);
            }
        }
    } else {
        // 调用方已建立 V->MTE2->MTE3 传递序时（合并装配路径），内部 V->MTE3 同步为空转，可跳过
        if (!vSyncPreDone) {
            SyncVtoM3();
        }
        if (blockLen % BLOCK_BYTES == 0) {
            DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(blockLen),
                                 (pitchBytes - blockLen) / BLOCK_BYTES, static_cast<int64_t>(gmStride * dSize), 0);
            DataCopyPad(gm[gmOffset], ub, cp);
        } else {
            for (int64_t r = 0; r < rows; ++r) {
                DataCopyExtParams cp(1, static_cast<uint32_t>(blockLen), 0, 0, 0);
                DataCopyPad(gm[gmOffset + r * (gmStride + length)], ub[rowAlign * r], cp);
            }
        }
    }
}

__aicore__ inline void ReadHPrev(int64_t tIdx, int64_t mStart, int64_t bOff, int64_t bRows, int64_t hOff, int64_t hLen,
                                 int64_t hAligned, LocalTensor<float>& ub)
{
    this->vfMte2Dirty_ = true; // 本函数 MTE2 装载喂 VF/后续消费——scatter 需同步
                               // （Tail 侧调用为伪标记, 下步入口复位, 安全）
    const bool isReverse = (this->tiling->direction != 0);
    if (tIdx == 0) {
        CopyInRow(this->inputGm.initHGm, ub, (mStart + bOff) * this->H + hOff, bRows, hLen, hAligned, 0);
        return;
    }
    int64_t prevIdx = isReverse ? (this->T - tIdx) : (tIdx - 1);
    if (!this->tiling->isSeqLength) {
        CopyInRow(this->inputGm.outputHGm, ub, prevIdx * this->B * this->H + (mStart + bOff) * this->H + hOff, bRows,
                  hLen, hAligned, 0);
        return;
    }
    int64_t compactOutputRow = this->GetCompactRowOffset(prevIdx);
    int64_t bsPrev = this->inputGm.batchSizesGm.GetValue(prevIdx);
    int64_t rowStart = mStart + bOff; // 全局 batch 起点
    if (rowStart >= bsPrev) {
        CopyInRow(this->inputGm.initHGm, ub, rowStart * this->H + hOff, bRows, hLen, hAligned, 0);
    } else if (rowStart + bRows <= bsPrev) {
        CopyInRow(this->inputGm.outputHGm, ub, (compactOutputRow + rowStart) * this->H + hOff, bRows, hLen, hAligned,
                  0);
    } else {
        int64_t nActive = bsPrev - rowStart;
        int64_t nNew = bRows - nActive;
        CopyInRow(this->inputGm.outputHGm, ub, (compactOutputRow + rowStart) * this->H + hOff, nActive, hLen, hAligned,
                  0);
        auto ubNewPart = ub[nActive * hAligned];
        CopyInRow(this->inputGm.initHGm, ubNewPart, (rowStart + nActive) * this->H + hOff, nNew, hLen, hAligned, 0);
    }
}

// 快路径判定缓存——形状派生量（H 对齐性、staging 容量、tile 数）kernel 内恒定，
// VecProcess 入口一次算齐，逐步谓词退化为单次 bool 读（消每步乘除比较重算）。
__aicore__ inline void InitFastPathCache()
{
    if (this->fastPathInit_) {
        return;
    }
    this->hAlign8Ok_ = (this->H % this->alignPara) == 0;
    this->mergedShapeOk_ = this->hAlign8Ok_;
    const int64_t hAFull = ((this->H + this->alignMask) / this->alignPara) * this->alignPara;
    const int64_t vfVL = static_cast<int64_t>(AscendC::GetVecLen() / sizeof(float));
    this->scatterVOk_ = (hAFull % vfVL) == 0;
    this->stagingCapRows_ = this->allocLength / (GRU_GATE_SIZE * hAFull); // staging [rows, 3hA] 双面容量
    int64_t mStart = 0, mCnt = 0;
    if (this->GetCoreRows(mStart, mCnt) && mCnt > 0) {
        const int64_t hTiles = (this->H + this->vecHTile_ - 1) / this->vecHTile_;
        const int64_t bTiles = (mCnt + this->vecBTile_ - 1) / this->vecBTile_;
        const int64_t pfRows = (mCnt < this->vecBTile_) ? mCnt : this->vecBTile_;
        this->dhPrevUbPath_ = !this->tiling->isSeqLength && hTiles == 1 && bTiles == 1;
        this->pfCapOk_ = bTiles == 1 && pfRows * hAFull <= this->allocLength;
        // ubNDy 槽位仅 8KB——dy 预取超域时由 LoadGradH 回退 GM 直读
        this->pfDyCapOk_ = this->pfCapOk_ && pfRows * hAFull <= this->allocLength / DBUF_SLOTS; // 半池 8KB 上限
        // 带化分派——scatter 域之上加 T>=2（首步无 GEMM 前驱走全 H 路径）与 H%128==0
        // （带宽 hAb=H/2 需 %VL==0 且 >=VL）。预取命中由 pfCapOk_ && mCnt>0 && !seq && T>=2
        // 构造性保证（Tail(t) 对 t>0 恒预取 t-1）。
        this->bandPath_ = this->dhPrevUbPath_ && this->pfCapOk_ && this->mergedShapeOk_ && this->scatterVOk_ &&
                          this->T >= BAND_NUM && (this->H % BAND_H_ALIGN) == 0 && mCnt > 0 &&
                          mCnt <= this->stagingCapRows_; // T>=2: 首步无 GEMM 前驱走全 H 路径
    }
    this->fastPathInit_ = true;
}

// dhPrev UB 驻留快路径判定 —— 单 b/H tile + 定长 + 非变长 + fp32(半精度变体保 round-back 契约 §5.2)。
__aicore__ inline bool DhPrevUbPath()
{
    if constexpr (sizeof(DTYPE) == 4) {
        return this->dhPrevUbPath_;
    } else {
        return false;
    }
}

// grad_h_t = dy[t] + dh_next
__aicore__ inline void LoadGradH(int64_t tIdx, int64_t ghOff, int64_t batchHOff, int64_t bRows, int64_t hLen,
                                 int64_t hAligned)
{
    if (tIdx != T - 1 && this->DhPrevUbPath()) {
        if (this->dyPrefetched) {
            this->dyPrefetched = false; // 单次消费（与 gatesPrefetched 同语义）
            Add(this->ubGradH, this->ubGradH, this->ubNDy, bRows * hAligned);
        } else {
            CopyInRow(this->inputGm.dyGm, this->ubTmp, ghOff, bRows, hLen, hAligned, 0);
            SyncM2toV();
            Add(this->ubGradH, this->ubGradH, this->ubTmp, bRows * hAligned);
        }
        return;
    }
    CopyInRow(this->inputGm.dyGm, this->ubGradH, ghOff, bRows, hLen, hAligned, 0);
    if (tIdx == T - 1) {
        CopyInRow(this->inputGm.dhGm, this->ubTmp, batchHOff, bRows, hLen, hAligned, 0);
    } else {
        CopyInRow(this->workGm.dhPrevWsGm, this->ubTmp, batchHOff, bRows, hLen, hAligned, 0);
    }
    SyncM2toV();
    Add(this->ubGradH, this->ubGradH, this->ubTmp, bRows * hAligned);
}

__aicore__ inline void LoadGatesAndHPrev(int64_t tIdx, int64_t ghOff, int64_t mStart, int64_t bOff, int64_t bRows,
                                         int64_t hOff, int64_t hLen, int64_t hAligned)
{
    if (this->gatesPrefetched && !this->tiling->isSeqLength && this->H <= this->vecHTile_ &&
        (this->H % this->alignPara) == 0) {
        // 视图乒乓——计算直读预取缓冲, 免 4 面 UB->UB 中转 DataCopy + 同步。
        // 条件: 单 h/b tile 且 H 对齐(预取 rowAlign=H 与计算 hAligned 间距一致);
        // 不满足回退原中转路径。乒乓: 交换后 ubN* 指向旧计算缓冲, 供下步预取写入。
        LocalTensor<float> tv;
        tv = this->ubUpdate;
        this->ubUpdate = this->ubNUpdate;
        this->ubNUpdate = tv;
        tv = this->ubNew;
        this->ubNew = this->ubNNew;
        this->ubNNew = tv;
        tv = this->ubReset;
        this->ubReset = this->ubNReset;
        this->ubNReset = tv;
        tv = this->ubHN;
        this->ubHN = this->ubNHN;
        this->ubNHN = tv;
        this->gatesPrefetched = false; // 单次消费语义保持
    } else if (this->gatesPrefetched && !this->tiling->isSeqLength) {
        // 消费预取: UB->UB 拷贝(MTE2 发射, 低于 GM 访存开销), r/z/n/hn 四面
        // MTE2 喂 VF——标记脏（scatter 处同步）
        this->vfMte2Dirty_ = true;
        const int64_t cpCount = bRows * hAligned;
        if ((cpCount * FLOAT_BYTES) % BLOCK_BYTES == 0) {
            DataCopyParams dp(1, static_cast<uint16_t>(cpCount * FLOAT_BYTES / BLOCK_BYTES), 0, 0);
            DataCopy(this->ubUpdate, this->ubNUpdate, dp);
            DataCopy(this->ubNew, this->ubNNew, dp);
            DataCopy(this->ubReset, this->ubNReset, dp);
            DataCopy(this->ubHN, this->ubNHN, dp);
        } else {
            for (int64_t i = 0; i < cpCount; ++i) {
                this->ubUpdate.SetValue(i, this->ubNUpdate.GetValue(i));
                this->ubNew.SetValue(i, this->ubNNew.GetValue(i));
                this->ubReset.SetValue(i, this->ubNReset.GetValue(i));
                this->ubHN.SetValue(i, this->ubNHN.GetValue(i));
            }
        }
        this->gatesPrefetched = false; // 单次消费
    } else {
        // GM->UB 门装载 MTE2 喂 VF——标记脏（scatter 处同步）
        this->vfMte2Dirty_ = true;
        CopyInRow(this->inputGm.updateGateGm, this->ubUpdate, ghOff, bRows, hLen, hAligned, 0);
        CopyInRow(this->inputGm.newGateGm, this->ubNew, ghOff, bRows, hLen, hAligned, 0);
        CopyInRow(this->inputGm.resetGateGm, this->ubReset, ghOff, bRows, hLen, hAligned, 0);
        CopyInRow(this->inputGm.hNGm, this->ubHN, ghOff, bRows, hLen, hAligned, 0);
    }
    // hp 预取消费——乒乓换入（与门控视图乒乓同构）。换出的旧 ubHPrev 缓冲
    // （持上步 dhFromH，Accumulate 已消费）成为 ubGate，供 Tail 预取下一拍 hp。
    // 预取仅在合并路径域发射（ubGate 的 1.0 常量仅逐门遗留路径消费）。
    if (this->hpPrefetched && !this->tiling->isSeqLength && this->H <= this->vecHTile_ &&
        (this->H % this->alignPara) == 0) {
        LocalTensor<float> tv = this->ubHPrev;
        this->ubHPrev = this->ubGate;
        this->ubGate = tv;
        this->hpPrefetched = false; // 单次消费语义保持
    } else {
        this->ReadHPrev(tIdx, mStart, bOff, bRows, hOff, hLen, hAligned, this->ubHPrev);
    }
}

// d_i_new = grad_h_t * (1-z) * (1-n^2)  -> dGi[new]
__aicore__ inline void ComputeDINew(int64_t gmOffGi, int64_t bRows, int64_t hLen, int64_t gmGateStride,
                                    int64_t hAligned)
{
    SyncM2toV();
    Duplicate(this->ubGate, 1.0f, bRows * hAligned);
    Sub(this->ubTmp, this->ubGate, this->ubUpdate, bRows * hAligned);
    Mul(this->ubTmp2, this->ubGradH, this->ubTmp, bRows * hAligned);
    Mul(this->ubTmp, this->ubNew, this->ubNew, bRows * hAligned);
    Sub(this->ubTmp, this->ubGate, this->ubTmp, bRows * hAligned);
    Mul(this->ubDHPrevFromH, this->ubTmp2, this->ubTmp, bRows * hAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dGiGm, this->ubDHPrevFromH, gmOffGi + GATE_IDX_NEW * H, bRows, hLen,
               this->H * GRU_GATE_SIZE - hLen, hAligned);
    SyncM3toV();
}

// d_reset = (d_i_new * h_n) * r * (1-r)  -> dGi[reset], dGh[reset]
__aicore__ inline void ComputeDReset(int64_t gmOffGi, int64_t bRows, int64_t hLen, int64_t gmGateStride,
                                     int64_t hAligned)
{
    Mul(this->ubTmp, this->ubDHPrevFromH, this->ubHN, bRows * hAligned);
    Mul(this->ubTmp2, this->ubTmp, this->ubReset, bRows * hAligned);
    Sub(this->ubTmp, this->ubGate, this->ubReset, bRows * hAligned);
    Mul(this->ubTmp3, this->ubTmp2, this->ubTmp, bRows * hAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dGiGm, this->ubTmp3, gmOffGi, bRows, hLen, this->H * GRU_GATE_SIZE - hLen, hAligned);
    CopyOutRow(this->workGm.dGhGm, this->ubTmp3, gmOffGi, bRows, hLen, this->H * GRU_GATE_SIZE - hLen, hAligned);
    SyncM3toV();
}

// d_update = grad_h_t * (hp-n) * z * (1-z)  -> dGi[update], dGh[update]
__aicore__ inline void ComputeDUpdate(int64_t gmOffGi, int64_t bRows, int64_t hLen, int64_t gmGateStride,
                                      int64_t hAligned)
{
    Sub(this->ubTmp, this->ubHPrev, this->ubNew, bRows * hAligned);
    Mul(this->ubTmp2, this->ubGradH, this->ubTmp, bRows * hAligned);
    Mul(this->ubTmp, this->ubTmp2, this->ubUpdate, bRows * hAligned);
    Sub(this->ubTmp2, this->ubGate, this->ubUpdate, bRows * hAligned);
    Mul(this->ubTmp3, this->ubTmp, this->ubTmp2, bRows * hAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dGiGm, this->ubTmp3, gmOffGi + H, bRows, hLen, this->H * GRU_GATE_SIZE - hLen, hAligned);
    CopyOutRow(this->workGm.dGhGm, this->ubTmp3, gmOffGi + H, bRows, hLen, this->H * GRU_GATE_SIZE - hLen, hAligned);
    SyncM3toV();
}

// d_gh[new] = d_i_new * r;  dh_prev_from_h = grad_h_t * z  -> dhFromH
__aicore__ inline void ComputeDGhNewAndDhFromH(int64_t gmOffGi, int64_t batchHOff, int64_t bRows, int64_t hLen,
                                               int64_t gmGateStride, int64_t hAligned)
{
    Mul(this->ubTmp3, this->ubDHPrevFromH, this->ubReset, bRows * hAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dGhGm, this->ubTmp3, gmOffGi + GATE_IDX_NEW * H, bRows, hLen, gmGateStride, hAligned);
    SyncM3toV();
    Mul(this->ubDHPrevFromH, this->ubGradH, this->ubUpdate, bRows * hAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dhFromHGm, this->ubDHPrevFromH, batchHOff, bRows, hLen, 0, hAligned);
    SyncM3toV();
}

__aicore__ inline void ProcessVectorHTile(int64_t tIdx, int64_t gateIdx, int64_t mStart, int64_t bOff, int64_t bRows,
                                          int64_t ht, int64_t compactGateRow, int64_t gm3H, int64_t gmGateStride,
                                          int64_t hTiles)
{
    int64_t hLen = this->vecHTile_;
    int64_t hOff = ht * this->vecHTile_;
    if (ht == hTiles - 1) {
        hLen = H - hOff;
    }
    int64_t hAligned = ((hLen + this->alignMask) / this->alignPara) * this->alignPara; // dtype 感知行距
    int64_t blkAligned = bRows * hAligned;
    // 复位脏标记——Accumulate 的 SyncM2toV（上迭代末）已覆盖此前全部 MTE2（含
    // 上步 Tail 预取，MTE2 管内序）；此后仅 LoadGates/ReadHPrev 的 MTE2 分支重标脏。
    this->vfMte2Dirty_ = false;

    // ⚠ hTiles>=2（H>4096）时的跨 tile WAR：本 tile 的 MTE2 装载（写 ubGradH/ubTmp/ubUpdate 等）
    // 与上一 tile 仍在 V 管线排队的计算（读/写同名 UB）无序——V 标志链只护 MTE3 不护 MTE2。
    // （H4097/4104 实测）：上一 tile 的 V 写晚到覆盖本 tile 的 dh 装载（ubTmp 写写冲突）
    // → tile-1 全域输出错；上一 tile 的 V 读被本 tile 装载提前覆盖 → tile-0 区 h=0..7 散点错。
    // 单 tile（H<=4096）不跨 tile 故从未触发；步进循环有 SyncAll 全栅栏护住。
    SyncVtoM2(); // 恢复无条件（单 tile 跳过依赖的传递序推理不可靠）

    int64_t ghOff, gmOffGi;
    if (this->tiling->isSeqLength) {
        ghOff = (compactGateRow + mStart + bOff) * H + hOff;
        gmOffGi = (compactGateRow + mStart + bOff) * gm3H + hOff;
    } else {
        ghOff = gateIdx * B * H + (mStart + bOff) * H + hOff;
        gmOffGi = gateIdx * B * gm3H + (mStart + bOff) * gm3H + hOff;
    }
    int64_t batchHOff = (mStart + bOff) * H + hOff;

    this->LoadGradH(tIdx, ghOff, batchHOff, bRows, hLen, hAligned);
    this->LoadGatesAndHPrev(tIdx, ghOff, mStart, bOff, bRows, hOff, hLen, hAligned);
    const bool mergedGrads = this->GateGradsMergedOk(bRows, hLen, hAligned);
    this->dhFromHInHp_ = mergedGrads; // dhFromH 常驻缓冲按路径记录（供 Accumulate 快路径）
    if (this->GateGradsScatterOk(bRows, hLen, hAligned)) {
        // VF scatter 直写 staging（hA%VL==0 整拍域）——装配 MTE2 与双同步出链
        this->ComputeGateGradsScatter(gmOffGi, batchHOff, bRows, hLen, hAligned);
    } else if (mergedGrads) {
        // 合并路径——单 tile hOff==0，gmOffGi 即整行 [0,3H) 起点
        this->ComputeGateGradsMerged(gmOffGi, batchHOff, bRows, hLen, hAligned);
    } else {
        this->ComputeDINew(gmOffGi, bRows, hLen, gmGateStride, hAligned);
        this->ComputeDReset(gmOffGi, bRows, hLen, gmGateStride, hAligned);
        this->ComputeDUpdate(gmOffGi, bRows, hLen, gmGateStride, hAligned);
        this->ComputeDGhNewAndDhFromH(gmOffGi, batchHOff, bRows, hLen, gmGateStride, hAligned);
    }
}

__aicore__ inline void StoreHPrevHTile(int64_t tIdx, int64_t gateIdx, int64_t mStart, int64_t bOff, int64_t bRows,
                                       int64_t ht, int64_t compactGateRow, int64_t hTiles, bool hpInUb)
{
    int64_t hLen = this->vecHTile_;
    int64_t hOff = ht * this->vecHTile_;
    if (ht == hTiles - 1) {
        hLen = H - hOff;
    }
    int64_t hAligned = ((hLen + this->alignMask) / this->alignPara) * this->alignPara;

    // hpInUb 分支无 V 写/无 MTE2（仅 MTE3 读 V 写过的 ubDHPrevFromH，其可见性由
    // scatter 内 SyncVtoM3 已建立；MTE3 同管读读内序）——顶部 M3→V 防护仅为非 hpInUb
    // 分支的 Duplicate(V 写 ubTmp2) 所需，移入该分支。
    int64_t hPrevWsOff;
    if (this->tiling->isSeqLength) {
        hPrevWsOff = (compactGateRow + mStart + bOff) * H + hOff;
    } else {
        hPrevWsOff = (gateIdx * B + mStart + bOff) * H + hOff;
    }
    if (hpInUb) {
        // hp 已由 VF scatter 保留至 ubDHPrevFromH（本步 scatter 单 tile 域）——
        // 免 ReadHPrev GM 重载。V→M3 序：scatter 内 SyncVtoM3 已建立（其后 MTE3 同序）。
        CopyOutRow(this->workGm.hPrevWsGm, this->ubDHPrevFromH, hPrevWsOff, bRows, hLen, 0, hAligned);
    } else {
        // 跨 tile WAR：本 tile Duplicate(V 写 ubTmp2) 需等上一 tile CopyOutRow(MTE3 读 ubTmp2) 排空
        SyncM3toV();
        Duplicate(this->ubTmp2, 0.0f, bRows * hAligned);
        SyncVtoM2();
        this->ReadHPrev(tIdx, mStart, bOff, bRows, hOff, hLen, hAligned, this->ubTmp2);
        SyncM2toM3();
        CopyOutRow(this->workGm.hPrevWsGm, this->ubTmp2, hPrevWsOff, bRows, hLen, 0, hAligned);
    }
    SyncM3toM2(); // 下一 hTile 的 ReadHPrev（MTE2 写 ubTmp2）与本写（MTE3 读）防同型 WAR
}

__aicore__ inline void ProcessVector(int64_t tIdx)
{
    const int64_t bTile = this->vecBTile_, hTile = this->vecHTile_;
    const bool isReverse = (this->tiling->direction != 0);
    const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
    const int64_t gm3H = this->H * GRU_GATE_SIZE;
    const int64_t gmGateStride = this->H * GATE_IDX_NEW; // 弃用: 消费点已改 3H-hLen(GAP 正确值)

    int64_t compactGateRow = 0, curBatch = this->B;
    if (this->tiling->isSeqLength) {
        compactGateRow = this->GetCompactRowOffset(gateIdx);
        curBatch = this->inputGm.batchSizesGm.GetValue(gateIdx);
    }

    int64_t mStart = 0, mCnt = 0;
    if (!this->GetCoreRows(mStart, mCnt)) {
        return;
    }
    if (this->tiling->isSeqLength) {
        if (mStart >= curBatch) {
            return;
        }
        if (mStart + mCnt > curBatch) {
            mCnt = curBatch - mStart;
        }
    }

    int64_t bTiles = (mCnt + bTile - 1) / bTile;
    int64_t hTiles = (H + hTile - 1) / hTile;
    for (int64_t bt = 0; bt < bTiles; bt++) {
        int64_t bRows = bTile;
        int64_t bOff = bt * bTile;
        if (bt == bTiles - 1) {
            bRows = mCnt - bOff;
        }
        for (int64_t ht = 0; ht < hTiles; ht++) {
            this->ProcessVectorHTile(tIdx, gateIdx, mStart, bOff, bRows, ht, compactGateRow, gm3H, gmGateStride,
                                     hTiles);
        }
    }
}

// 步尾段——V2C 置位后、C2V 等待前的 GEMM 窗口内执行(原在计算体内, 挂串行链):
// StoreHPrev(重读 GM + 写 hPrevWs, 仅终轮转置跨簇消费) + hp DCCI + 门/dy 预取。
__aicore__ inline void ProcessVectorTail(int64_t tIdx)
{
    const int64_t bTile = this->vecBTile_, hTile = this->vecHTile_;
    const bool isReverse = (this->tiling->direction != 0);
    const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;

    int64_t compactGateRow = 0, curBatch = this->B;
    if (this->tiling->isSeqLength) {
        compactGateRow = this->GetCompactRowOffset(gateIdx);
        curBatch = this->inputGm.batchSizesGm.GetValue(gateIdx);
    }
    int64_t mStart = 0, mCnt = 0;
    if (!this->GetCoreRows(mStart, mCnt)) {
        return;
    }
    if (this->tiling->isSeqLength) {
        if (mStart >= curBatch) {
            return;
        }
        if (mStart + mCnt > curBatch) {
            mCnt = curBatch - mStart;
        }
    }
    {
        int64_t bTiles = (mCnt + bTile - 1) / bTile;
        int64_t hTiles = (H + hTile - 1) / hTile;
        // hp 保留消费仅限单 tile 域——多 bTile 时 VF 保留面被逐 tile 覆写,
        // 仅末 tile 数据有效; 回退 ReadHPrev GM 重载
        const bool hpInUb = this->hpPreserved_ && bTiles == 1 && hTiles == 1;
        // 单次消费即复位（与 gatesPrefetched 同语义）：变长模式下 mCnt 逐步变化，
        // scatter 适配可能跨 stagingCapRows_ 翻转——翻转后的逐门路径已把 ubDHPrevFromH
        // 覆写为 dhFromH，历史标志若不清零会误将 dhFromH 当 hp 直写 hPrevWs
        this->hpPreserved_ = false;
        for (int64_t bt = 0; bt < bTiles; bt++) {
            int64_t bRows = bTile;
            int64_t bOff = bt * bTile;
            if (bt == bTiles - 1) {
                bRows = mCnt - bOff;
            }
            for (int64_t ht = 0; ht < hTiles; ht++) {
                this->StoreHPrevHTile(tIdx, gateIdx, mStart, bOff, bRows, ht, compactGateRow, hTiles, hpInUb);
            }
        }
        // hp DCCI(自 DcciStepSlices 移入窗口: hPrevWs 的跨簇读者仅终轮转置, SyncAll+V2C#2 兜底,
        // 逐步刷新仅为写缓冲落地; 置于本写之后语义不变)
        {
            GlobalTensor<DTYPE> hp;
            if (this->tiling->isSeqLength) {
                hp.SetGlobalBuffer(this->workGm.hPrevWsGm.GetPhyAddr((compactGateRow + mStart) * this->H),
                                   mCnt * this->H);
            } else {
                hp.SetGlobalBuffer(this->workGm.hPrevWsGm.GetPhyAddr((gateIdx * this->B + mStart) * this->H),
                                   mCnt * this->H);
            }
            DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(hp);
        }
    }
    // 门平面预取: 下一拍(tIdx-1)的 r/z/n/hn/dy 的 MTE2 装载(乒乓目标 = 非别名缓冲)
    if (tIdx > 0) {
        const int64_t nextGateIdx = isReverse ? (this->T - tIdx) : (tIdx - 1);
        int64_t nextGhOff;
        if (this->tiling->isSeqLength) {
            // packed 模式预取从简(逐 timestep 偏移不同, 预取意义有限) —— 不预取
        } else {
            nextGhOff = nextGateIdx * B * H;
        }
        if (!this->tiling->isSeqLength) {
            // ⚠ 预取仅限单 bTile 且整行落 UB 预算:
            //   1) pfBTiles>=2 时各 tile 重复写 ubN*[0] 互踩——消费端(LoadGatesAndHPrev/LoadGradH)
            //      恒从缓冲头读 bRows 行, 后一 tile 覆盖前一 tile 首行 → 本 AIV 首行门控错
            //      (H);
            //   2) bRows*H 超 allocLength 时 MTE2 越界写 UB(原 预算门)。
            // 容量/单tile判定缓存; 7 面装载共享一份 cp（逐步恒定, 原 7×15-20
            // 条标量装配 → 1×; fp32 + 块长 32B 对齐域, 与 CopyInRow 快路径逐参一致）。
            const int64_t hAlignedFull = ((this->H + this->alignMask) / this->alignPara) * this->alignPara;
            if (this->pfCapOk_) {
                const int64_t pitchBytes = hAlignedFull * static_cast<int64_t>(sizeof(float));
                if constexpr (sizeof(DTYPE) == 4) {
                    // fp32 共参快路径（块长 32B 对齐前提与 CopyInRow 内部一致）
                    if ((this->H * FLOAT_BYTES) % BLOCK_BYTES == 0) {
                        DataCopyExtParams cp(static_cast<uint16_t>(mCnt), static_cast<uint32_t>(this->H * FLOAT_BYTES),
                                             0, (pitchBytes - this->H * FLOAT_BYTES) / BLOCK_BYTES, 0);
                        DataCopyPadExtParams<DTYPE> pp(true, 0, 0, 0);
                        DataCopyPad(this->ubNUpdate, this->inputGm.updateGateGm[nextGhOff + mStart * H], cp, pp);
                        DataCopyPad(this->ubNNew, this->inputGm.newGateGm[nextGhOff + mStart * H], cp, pp);
                        DataCopyPad(this->ubNReset, this->inputGm.resetGateGm[nextGhOff + mStart * H], cp, pp);
                        DataCopyPad(this->ubNHN, this->inputGm.hNGm[nextGhOff + mStart * H], cp, pp);
                        if (this->pfDyCapOk_) { // ubNDy 槽位 8KB，超域不预取（GM 回退）
                            DataCopyPad(this->ubNDy, this->inputGm.dyGm[nextGhOff + mStart * H], cp, pp);
                            this->dyPrefetched = true;
                        }
                        this->gatesPrefetched = true;
                        if (this->hAlign8Ok_) {
                            const int64_t nextT = tIdx - 1;
                            if (nextT == 0) {
                                DataCopyPad(this->ubGate, this->inputGm.initHGm[mStart * this->H], cp, pp);
                            } else {
                                const int64_t prevIdx = isReverse ? (this->T - nextT) : (nextT - 1);
                                DataCopyPad(this->ubGate,
                                            this->inputGm.outputHGm[prevIdx * this->B * this->H + mStart * this->H], cp,
                                            pp);
                            }
                            this->hpPrefetched = true;
                        }
                    } else {
                        this->PrefetchFallback(tIdx, isReverse, nextGhOff, mStart, mCnt, hAlignedFull);
                    }
                } else {
                    this->PrefetchFallback(tIdx, isReverse, nextGhOff, mStart, mCnt, hAlignedFull);
                }
            }
        }
    }
}

// 预取回退路径（fp16 或 fp32 非对齐块长）——与原 CopyInRow×7 行为逐参一致。
__aicore__ inline void PrefetchFallback(int64_t tIdx, bool isReverse, int64_t nextGhOff, int64_t mStart, int64_t mCnt,
                                        int64_t hAlignedFull)
{
    CopyInRow(this->inputGm.updateGateGm, this->ubNUpdate, nextGhOff + mStart * H, mCnt, this->H, hAlignedFull, 0);
    CopyInRow(this->inputGm.newGateGm, this->ubNNew, nextGhOff + mStart * H, mCnt, this->H, hAlignedFull, 0);
    CopyInRow(this->inputGm.resetGateGm, this->ubNReset, nextGhOff + mStart * H, mCnt, this->H, hAlignedFull, 0);
    CopyInRow(this->inputGm.hNGm, this->ubNHN, nextGhOff + mStart * H, mCnt, this->H, hAlignedFull, 0);
    // dy 同窗预取(下一拍 LoadGradH UB 直取)；8KB 槽位容量门
    if (this->pfDyCapOk_) {
        CopyInRow(this->inputGm.dyGm, this->ubNDy, nextGhOff + mStart * H, mCnt, this->H, hAlignedFull, 0);
        this->dyPrefetched = true;
    }
    this->gatesPrefetched = true;
    // hp(t-1) 同窗预取至 ubGate（合并路径域——ubGate 的 1.0 常量仅逐门
    // 遗留路径消费, 其 ComputeDINew 入口自重填）。hp 源与 ReadHPrev 同构:
    // nextT==0 → initHGm, 否则 outputHGm[prevIdx(nextT)]。
    if (this->hAlign8Ok_) {
        const int64_t nextT = tIdx - 1;
        if (nextT == 0) {
            CopyInRow(this->inputGm.initHGm, this->ubGate, mStart * this->H, mCnt, this->H, hAlignedFull, 0);
        } else {
            const int64_t prevIdx = isReverse ? (this->T - nextT) : (nextT - 1);
            CopyInRow(this->inputGm.outputHGm, this->ubGate, prevIdx * this->B * this->H + mStart * this->H, mCnt,
                      this->H, hAlignedFull, 0);
        }
        this->hpPrefetched = true;
    }
}

// 预取视图乒乓消费（门控 ×4 + hp）——带路径每迭代一次。
// ⚠ hp 换出缓冲（ubGate）恰持 dhFromH(t+1)（上步 VF 就地写入），本迭代带折叠消费，
// Tail 预取在其后覆写——乒乓自洽，无需新缓冲。预取命中为构造性保证（bandPath_ 谓词）。
__aicore__ inline void ConsumePrefetchSwaps()
{
    LocalTensor<float> tv;
    tv = this->ubUpdate;
    this->ubUpdate = this->ubNUpdate;
    this->ubNUpdate = tv;
    tv = this->ubNew;
    this->ubNew = this->ubNNew;
    this->ubNNew = tv;
    tv = this->ubReset;
    this->ubReset = this->ubNReset;
    this->ubNReset = tv;
    tv = this->ubHN;
    this->ubHN = this->ubNHN;
    this->ubNHN = tv;
    this->gatesPrefetched = false;
    LocalTensor<float> th = this->ubHPrev;
    this->ubHPrev = this->ubGate;
    this->ubGate = th;
    this->hpPrefetched = false;
}

// 带化主循环——步周期 = G' + 2L + A'/2 的收益兑现段。
// 时间线: [Wait B0 → 带0 (与 AIC 带1 GEMM 重叠) → Wait C2V → 带1 → M3/DCCI/V2C → Tail]
// 首步 (T-1) 无 GEMM 前驱 → 现行全 H 路径；尾步 epilogue 消费 GEMM(0) 的 B0/C2V 并以
// 现行 AccumulateDhPrev(0) 物化 dhPrev(0)（flag 平衡: 每步 B0/C2V 各 1 set 1 wait×2AIV）。
__aicore__ inline void VecProcessBand()
{
    this->dhFromHInHp_ = true;
    const int64_t mStart = this->mStart_, mCnt = this->mCnt_;
    const int64_t hA = ((this->H + this->alignMask) / this->alignPara) * this->alignPara;
    const int64_t hAb = this->H / BAND_NUM; // 带宽 = H 二分
    const uint32_t vfVL = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
    // 偏置输出恒走 ProcessBiasReduce GM 归约：原 biasInline UB 部分和全宽直写
    // 在多 AIV 有行时构成多写竞争且缺他核行贡献（db=全 (T,B) 行求和），已弃用。
    // 全宽 M3 的 cp 跨迭代恒定——外提（逐步免 ~10 条标量装配）
    DataCopyExtParams cpBand(static_cast<uint16_t>(mCnt), static_cast<uint32_t>(GRU_GATE_SIZE * this->H * FLOAT_BYTES),
                             0, 0, 0); // [rows,3H] 全宽
    // 首步: 现行全 H 路径（dy+dh 首步装载 + 非预取门装载 + scatter + M3）
    this->ProcessVector(this->T - 1);
    this->DcciStepSlices(this->T - 1);
    if (this->s1Usable_) {
        // 带化主循环——B0/B1/C2V 信号替换为 SyncAll<false>×2 + 规约。
        // 迭代序: #2(t+1) → Reduce(t+1)[dhPrevF32 物化] → 带VF/M3/Dcci → #1(t) → Tail(t)。
        AscendC::SyncAll<false>();
        this->ProcessVectorTail(this->T - 1);
        for (int64_t tIdx = this->T - 2; tIdx >= 0;
             tIdx--) { // 自次末步 T-2 起倒推（首步 T-1 无 GEMM 前驱，已单独处理）
            const bool isReverse = (this->tiling->direction != 0);
            const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
            const int64_t gmOffGi = (gateIdx * this->B + mStart) * this->H * GRU_GATE_SIZE;
            AscendC::SyncAll<false>(); // #2(t+1): 部分和全簇就绪
            this->ReduceS1();
            if (mCnt > 0) {
                this->ConsumePrefetchSwaps();
                CopyInRow(this->dhPrevF32Gm, this->ubTmp3, mStart * this->H, mCnt, hAb, hAb, this->H - hAb);
                SyncM2toV();
                asc_vf_call<GateGradsBandVF<float>>(
                    (__ubuf__ float*)this->ubTmp.GetPhyAddr(), (__ubuf__ float*)this->ubTmp2.GetPhyAddr(),
                    (__ubuf__ float*)this->ubHPrev.GetPhyAddr(), (__ubuf__ float*)this->ubDHPrevFromH.GetPhyAddr(),
                    (__ubuf__ float*)this->ubTmp3.GetPhyAddr(), (__ubuf__ float*)this->ubNDy.GetPhyAddr(),
                    (__ubuf__ float*)this->ubGate.GetPhyAddr(), (__ubuf__ float*)this->ubUpdate.GetPhyAddr(),
                    (__ubuf__ float*)this->ubNew.GetPhyAddr(), (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),
                    (__ubuf__ float*)this->ubHN.GetPhyAddr(), (__ubuf__ float*)this->ubReset.GetPhyAddr(),
                    static_cast<uint32_t>(mCnt), static_cast<uint32_t>(hA), static_cast<uint32_t>(hAb), 0, vfVL);
                SyncVtoM2();
                CopyInRow(this->dhPrevF32Gm, this->ubTmp3, mStart * this->H + hAb, mCnt, hAb, hAb, this->H - hAb);
                SyncM2toV();
                asc_vf_call<GateGradsBandVF<float>>(
                    (__ubuf__ float*)this->ubTmp.GetPhyAddr(), (__ubuf__ float*)this->ubTmp2.GetPhyAddr(),
                    (__ubuf__ float*)this->ubHPrev.GetPhyAddr(), (__ubuf__ float*)this->ubDHPrevFromH.GetPhyAddr(),
                    (__ubuf__ float*)this->ubTmp3.GetPhyAddr(), (__ubuf__ float*)this->ubNDy.GetPhyAddr(),
                    (__ubuf__ float*)this->ubGate.GetPhyAddr(), (__ubuf__ float*)this->ubUpdate.GetPhyAddr(),
                    (__ubuf__ float*)this->ubNew.GetPhyAddr(), (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),
                    (__ubuf__ float*)this->ubHN.GetPhyAddr(), (__ubuf__ float*)this->ubReset.GetPhyAddr(),
                    static_cast<uint32_t>(mCnt), static_cast<uint32_t>(hA), static_cast<uint32_t>(hAb),
                    static_cast<uint32_t>(hAb), vfVL);
                this->hpPreserved_ = true;
                SyncVtoM3();
                if constexpr (sizeof(DTYPE) == 4) {
                    DataCopyPad(this->workGm.dGiGm[gmOffGi], this->ubTmp, cpBand);
                    DataCopyPad(this->workGm.dGhGm[gmOffGi], this->ubTmp2, cpBand);
                }
                SyncM3toV();
            }
            this->DcciStepSlices(tIdx);
            AscendC::SyncAll<false>(); // #1(t): dGh(t) 全簇就绪 → AIC 切片 GEMM
            this->ProcessVectorTail(tIdx);
        }
        AscendC::SyncAll<false>(); // #2(0)
        this->ReduceS1();
        this->AccumulateDhPrev(0);
        this->VecProcessFinalize();
        return;
    }
    ggsync::VecSignalCube();
    this->ProcessVectorTail(this->T - 1);
    for (int64_t tIdx = this->T - 2; tIdx >= 0; tIdx--) { // 自次末步 T-2 起倒推（首步 T-1 无 GEMM 前驱，已单独处理）
        const bool isReverse = (this->tiling->direction != 0);
        const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
        const int64_t gmOffGi = (gateIdx * this->B + mStart) * this->H * GRU_GATE_SIZE;
        ggsync::VecWaitCubeBand(); // B0: GEMM(t+1) 带0 就绪
        if (mCnt > 0) {
            this->ConsumePrefetchSwaps();
            // 带 0: gemmRes M2 + 折叠 VF（与 AIC 带1 GEMM 重叠窗口）
            CopyInRow(this->dhPrevF32Gm, this->ubTmp3, mStart * this->H, mCnt, hAb, hAb, this->H - hAb);
            SyncM2toV();
            asc_vf_call<GateGradsBandVF<float>>(
                (__ubuf__ float*)this->ubTmp.GetPhyAddr(),         // dGi staging 全宽
                (__ubuf__ float*)this->ubTmp2.GetPhyAddr(),        // dGh staging 全宽
                (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),       // dhFromH(t) 带写出（就地）
                (__ubuf__ float*)this->ubDHPrevFromH.GetPhyAddr(), // hp 保留带写
                (__ubuf__ float*)this->ubTmp3.GetPhyAddr(),        // gemmRes 带紧凑
                (__ubuf__ float*)this->ubNDy.GetPhyAddr(),         // dy(t) 带
                (__ubuf__ float*)this->ubGate.GetPhyAddr(),        // dhFromH(t+1) 带（换出缓冲）
                (__ubuf__ float*)this->ubUpdate.GetPhyAddr(),      // z
                (__ubuf__ float*)this->ubNew.GetPhyAddr(),         // n
                (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),       // hp(t)
                (__ubuf__ float*)this->ubHN.GetPhyAddr(),          // hn
                (__ubuf__ float*)this->ubReset.GetPhyAddr(),       // r
                static_cast<uint32_t>(mCnt), static_cast<uint32_t>(hA), static_cast<uint32_t>(hAb), 0, vfVL);
            SyncVtoM2(); // 带0 VF 的 ubTmp3 读 → 带1 M2 写 WAR（V→M2 定向）
        }
        ggsync::VecWaitCubeBand1(); // B1——末带 Fixpipe 落地（早于 C2V）
        if (mCnt > 0) {
            // 带 1 gemmRes M2 预发射：装载延迟藏进 B1→C2V 间隔（band1 关键段缩短）
            CopyInRow(this->dhPrevF32Gm, this->ubTmp3, mStart * this->H + hAb, mCnt, hAb, hAb, this->H - hAb);
        }
        ggsync::VecWaitCube(); // C2V: GEMM(t+1) 全就绪
        if (mCnt > 0) {
            // 带 1: 折叠 VF + 全宽 M3（staging 两带列区均已写齐）
            SyncM2toV();
            asc_vf_call<GateGradsBandVF<float>>(
                (__ubuf__ float*)this->ubTmp.GetPhyAddr(), (__ubuf__ float*)this->ubTmp2.GetPhyAddr(),
                (__ubuf__ float*)this->ubHPrev.GetPhyAddr(), (__ubuf__ float*)this->ubDHPrevFromH.GetPhyAddr(),
                (__ubuf__ float*)this->ubTmp3.GetPhyAddr(), (__ubuf__ float*)this->ubNDy.GetPhyAddr(),
                (__ubuf__ float*)this->ubGate.GetPhyAddr(), (__ubuf__ float*)this->ubUpdate.GetPhyAddr(),
                (__ubuf__ float*)this->ubNew.GetPhyAddr(), (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),
                (__ubuf__ float*)this->ubHN.GetPhyAddr(), (__ubuf__ float*)this->ubReset.GetPhyAddr(),
                static_cast<uint32_t>(mCnt), static_cast<uint32_t>(hA), static_cast<uint32_t>(hAb),
                static_cast<uint32_t>(hAb), vfVL);
            this->hpPreserved_ = true;
            SyncVtoM3();
            if constexpr (sizeof(DTYPE) == 4) {
                // 全宽 staging 单 M3（H%8==0 ⟹ hA==H，块长=3H 连续）
                DataCopyPad(this->workGm.dGiGm[gmOffGi], this->ubTmp, cpBand);
                DataCopyPad(this->workGm.dGhGm[gmOffGi], this->ubTmp2, cpBand);
            }
            SyncM3toV();
        }
        // bias 累加——从 staging 直积到 UB 累加器(ubGate 尾部别名)。
        // 位于 SyncM3toV(MTE3 已读完 staging)与 V2C(S 管不等 V 管)之间,
        // V 管空闲窗口内零关键路径; 管内序保证 Add 在 scatter 后执行。
        this->DcciStepSlices(tIdx);
        ggsync::VecSignalCube();
        this->ProcessVectorTail(tIdx);
    }
    // 尾步 epilogue: 消费 GEMM(0) 的 B0/B1/C2V（flag 平衡）+ dhPrev(0) 物化（现行 Accumulate 复用，
    // 读 ubHPrev 的 dhFromH(0) + dhPrevF32(0) 全量 M2）
    ggsync::VecWaitCubeBand();
    ggsync::VecWaitCubeBand1();
    ggsync::VecWaitCube();
    this->AccumulateDhPrev(0);
    this->VecProcessFinalize();
}

__aicore__ inline void InitDhPrev()
{
    if (!this->tiling->isSeqLength) {
        return;
    }
    const int64_t bTile = this->vecBTile_, hTile = this->vecHTile_;
    int64_t mStart = 0, mCnt = 0;
    if (!this->GetCoreRows(mStart, mCnt)) {
        return;
    }
    int64_t bTiles = (mCnt + bTile - 1) / bTile;
    int64_t hTiles = (H + hTile - 1) / hTile;
    for (int64_t bT = 0; bT < bTiles; bT++) {
        int64_t bOff = bT * bTile, bRows = bTile;
        if (bT == bTiles - 1)
            bRows = mCnt - bOff;
        for (int64_t hT = 0; hT < hTiles; hT++) {
            int64_t hOff = hT * hTile, hLen = hTile;
            if (hT == hTiles - 1)
                hLen = H - hOff;
            int64_t hAligned = ((hLen + this->alignMask) / this->alignPara) * this->alignPara;
            CopyInRow(this->inputGm.dhGm, this->ubTmp, (mStart + bOff) * H + hOff, bRows, hLen, hAligned, 0);
            SyncM2toM3();
            CopyOutRow(this->workGm.dhPrevWsGm, this->ubTmp, (mStart + bOff) * H + hOff, bRows, hLen, 0, hAligned);
            SyncM3toM2();
        }
    }
}

// ------------------------------------------------------------------------------------------------
// 门梯度合并计算（fp32 单 tile 快路径）。
// 原实现 4 个 Compute* 逐门 SyncVtoM3/CopyOut/SyncM3toV（7 次 MTE3 + ~10 次同步/步），
// 且 r/z 门 dGi/dGh 两平面同值逐门双写。本函数一条 V 链算出五个平面
// （dhn/drp/dzp/dGh_n/dhFromH），UB 内 2D scatter 装配 [r|z|n] 行布局后
// dGi/dGh 各一次 [rows,3H] 连续写出 + dhFromH 一次（3 次 MTE3 + 4 次同步）。
// 数值与逐门路径逐位一致（同一 V 指令序列、同一 GM 目标，仅写出编排变化）。
// ------------------------------------------------------------------------------------------------
__aicore__ inline bool GateGradsMergedOk(int64_t bRows, int64_t hLen, int64_t hAligned)
{
    if constexpr (sizeof(DTYPE) != 4) {
        return false;
    } else {
        // 仅单 hTile（hLen==H）：多 tile 时 r/z/n 门列区带间隔不连续，装配布局不成立。
        // 且要求 3H 行距 32B 对齐（H%8==0）：整行 [rows,3H] 写出走多行快路径；
        // 非对齐 H 的 per-row DataCopyPad 尾块会越界写破坏下一行首元素（H=85 实证，
        // GM 行距 3H 无对齐余量），回退原逐门路径（其尾块溢出落在本行 padding 内无害）。
        // 形状侧判定 Init 缓存；容量比较改预除上界（hLen==H 时 hAligned==hiddenSizeAligned）。
        return hLen == this->H && this->mergedShapeOk_ && bRows <= this->stagingCapRows_;
    }
}

__aicore__ inline void ComputeGateGradsMerged(int64_t gmOffGi, int64_t batchHOff, int64_t bRows, int64_t hLen,
                                              int64_t hAligned)
{
    const int64_t cnt = bRows * hAligned;
    SyncM2toV(); // 门/梯度 MTE2 装载完成
    // VF 寄存器级重写——基线多组 UB 往返 V 指令收敛为少数载入/存储 + 寄存器算子
    // 中间量全程驻留向量寄存器不落 UB。
    // 数值与 基线链逐位一致（GateGradsVF 内同指令序同操作数配对）；ubHN(hn→dGh_n)与
    // ubHPrev(hp→dhFromH) 就地覆写，下游 staging/快路径消费契约不变。
    // 可用性判据：asc_vf_call 若在本编译目标未展开（非 3510），输出缓冲持旧值 →
    {
        const uint32_t vfCnt = static_cast<uint32_t>(cnt);
        const uint32_t vfVL = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
        const uint16_t vfRep = static_cast<uint16_t>(
            AscendC::CeilDivision(static_cast<int32_t>(vfCnt), static_cast<int32_t>(vfVL)));
        asc_vf_call<GateGradsVF<float>>((__ubuf__ float*)this->ubTmp2.GetPhyAddr(),        // drp = dhn·hn·r·(1−r)
                                        (__ubuf__ float*)this->ubTmp3.GetPhyAddr(),        // dzp = A·(hp−n)·z
                                        (__ubuf__ float*)this->ubDHPrevFromH.GetPhyAddr(), // dhn = A·(1−n²)
                                        (__ubuf__ float*)this->ubHN.GetPhyAddr(), // dGh_n = r·dhn（就地覆写 hn）
                                        (__ubuf__ float*)this->ubHPrev.GetPhyAddr(), // dhFromH = gradH·z（就地覆写 hp）
                                        (__ubuf__ float*)this->ubGradH.GetPhyAddr(),  // gradH
                                        (__ubuf__ float*)this->ubUpdate.GetPhyAddr(), // z
                                        (__ubuf__ float*)this->ubNew.GetPhyAddr(),    // n
                                        (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),  // hp
                                        (__ubuf__ float*)this->ubHN.GetPhyAddr(),     // hn
                                        (__ubuf__ float*)this->ubReset.GetPhyAddr(),  // r
                                        vfCnt, vfVL, vfRep);
    }
    SyncVtoM2();
    // UB 装配: staging_dGi(ubTmp) = [drp|dzp|dhn], staging_dGh(ubReset) = [drp|dzp|dGh_n]
    // （staging_dGh 由 ubGate 迁至 ubReset：ubGate 现专持 1.0 常量不被覆写）
    const uint16_t blkLen = static_cast<uint16_t>(hAligned * FLOAT_BYTES / BLOCK_BYTES);
    // dstGap：z/n 两门列区跨距 = (GRU_GATE_SIZE-1)*hAligned 个 fp32（32B 块单位）
    const uint16_t dstGap = static_cast<uint16_t>((GRU_GATE_SIZE - 1) * hAligned * FLOAT_BYTES / BLOCK_BYTES);
    const DataCopyParams dp(static_cast<uint16_t>(bRows), blkLen, 0, dstGap);
    DataCopy(this->ubTmp[GATE_IDX_RESET * hAligned], this->ubTmp2, dp);      // 门 r
    DataCopy(this->ubTmp[GATE_IDX_UPDATE * hAligned], this->ubTmp3, dp);     // 门 z
    DataCopy(this->ubTmp[GATE_IDX_NEW * hAligned], this->ubDHPrevFromH, dp); // 门 n
    DataCopy(this->ubReset[GATE_IDX_RESET * hAligned], this->ubTmp2, dp);    // 门 r
    DataCopy(this->ubReset[GATE_IDX_UPDATE * hAligned], this->ubTmp3, dp);   // 门 z
    DataCopy(this->ubReset[GATE_IDX_NEW * hAligned], this->ubHN, dp);        // 门 n
    SyncM2toM3();
    // 整行连续写出: [rows, 3H] 平面一次 MTE3（列偏移 3×hOff 已含在 gmOffGi）。
    // V->MTE2->MTE3 传递序已建立，CopyOutRow 内部同步经 vSyncPreDone 跳过。
    CopyOutRow(this->workGm.dGiGm, this->ubTmp, gmOffGi, bRows, GRU_GATE_SIZE * hLen,
               this->H * GRU_GATE_SIZE - GRU_GATE_SIZE * hLen, GRU_GATE_SIZE * hAligned);
    CopyOutRow(this->workGm.dGhGm, this->ubReset, gmOffGi, bRows, GRU_GATE_SIZE * hLen,
               this->H * GRU_GATE_SIZE - GRU_GATE_SIZE * hLen, GRU_GATE_SIZE * hAligned);
    CopyOutRow(this->workGm.dhFromHGm, this->ubHPrev, batchHOff, bRows, hLen, 0, hAligned);
    SyncM3toV();
}

// ------------------------------------------------------------------------------------------------
// VF scatter 直写路径——GateGradsScatterVF 计算链 + [r|z|n] staging 布局直写一体完成。
// 相对逐门装配路径：6 次 UB→UB MTE2 DataCopy 与 SyncVtoM2/SyncM2toM3 双事件对出链，
// 串行链收敛为 MTE2 装载 → VF(scatter) → 单 SyncVtoM3 → 3 次 MTE3。
// dGh staging 自 ubReset 迁至 ubTmp2（ubReset 持输入 r 至 VF 消费完毕，散写同缓冲会踩
// 未消费行——首行散写范围 [0,3hA) 覆盖次行 r 输入区 [hA,2hA) 的行序冲突）。
// 数值与逐门路径逐位一致（同指令序）；dhFromH 仍落 ubHPrev（快路径契约保持）。
// ------------------------------------------------------------------------------------------------
__aicore__ inline bool GateGradsScatterOk(int64_t bRows, int64_t hLen, int64_t hAligned)
{
    if constexpr (sizeof(DTYPE) != 4) {
        return false;
    } else {
        // scatter 前置（单 hTile + H%8==0 + staging 容量）之上加整拍约束 hA%VL==0：
        // scatter 循环无尾拍掩码（CreateMask 全满），StoreAlign 地址恒 32B 对齐。
        // 非整拍域回退逐门路径。整拍判定 Init 缓存。
        return GateGradsMergedOk(bRows, hLen, hAligned) && this->scatterVOk_;
    }
}

__aicore__ inline void ComputeGateGradsScatter(int64_t gmOffGi, int64_t batchHOff, int64_t bRows, int64_t hLen,
                                               int64_t hAligned)
{
    // 快路径（预取视图乒乓 + hp 乒乓消费）下 Accumulate 同步后无新 MTE2——
    // 其 SyncM2toV 传递覆盖本 VF 全部输入（gradH=V 链内序、门/hp=上步预取 MTE2 管内序），
    // 免重复事件对；LoadGates/ReadHPrev 的 MTE2 分支已标脏，此处补同步。
    if (this->vfMte2Dirty_) {
        SyncM2toV(); // 门/梯度 MTE2 装载完成
    }
    {
        const uint32_t vfVL = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
        asc_vf_call<GateGradsScatterVF<float>>(
            (__ubuf__ float*)this->ubTmp.GetPhyAddr(),         // dGi staging [drp|dzp|dhn]
            (__ubuf__ float*)this->ubTmp2.GetPhyAddr(),        // dGh staging [drp|dzp|dGh_n]
            (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),       // dhFromH（就地覆写 hp）
            (__ubuf__ float*)this->ubDHPrevFromH.GetPhyAddr(), // hp 保留面（Tail 消费）
            (__ubuf__ float*)this->ubGradH.GetPhyAddr(),       // gradH
            (__ubuf__ float*)this->ubUpdate.GetPhyAddr(),      // z
            (__ubuf__ float*)this->ubNew.GetPhyAddr(),         // n
            (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),       // hp
            (__ubuf__ float*)this->ubHN.GetPhyAddr(),          // hn
            (__ubuf__ float*)this->ubReset.GetPhyAddr(),       // r
            static_cast<uint32_t>(bRows), static_cast<uint32_t>(hAligned), vfVL);
    }
    this->hpPreserved_ = true; // 本步 hp 已保留（Tail 单 tile 域消费）
    SyncVtoM3();               // VF scatter → MTE3（替代 V→M2→M3 传递链）
    // 整行连续写出: [rows, 3H] 平面一次 MTE3；vSyncPreDone——上方已建立 V→M3 序。
    // dGi/dGh 同参共享一份 cp（bRows/块长/行距逐步恒定, 原 2× 装配 → 1×）。
    if constexpr (sizeof(DTYPE) == 4) {
        const int64_t blockLen = GRU_GATE_SIZE * hLen * FLOAT_BYTES;
        const int64_t pitchBytes = GRU_GATE_SIZE * hAligned * FLOAT_BYTES;
        DataCopyExtParams cp(static_cast<uint16_t>(bRows), static_cast<uint32_t>(blockLen),
                             (pitchBytes - blockLen) / BLOCK_BYTES,
                             static_cast<int64_t>(this->H * GRU_GATE_SIZE - GRU_GATE_SIZE * hLen) * FLOAT_BYTES, 0);
        DataCopyPad(this->workGm.dGiGm[gmOffGi], this->ubTmp, cp);
        DataCopyPad(this->workGm.dGhGm[gmOffGi], this->ubTmp2, cp);
    } else {
        CopyOutRow(this->workGm.dGiGm, this->ubTmp, gmOffGi, bRows, GRU_GATE_SIZE * hLen,
                   this->H * GRU_GATE_SIZE - GRU_GATE_SIZE * hLen, GRU_GATE_SIZE * hAligned, true);
        CopyOutRow(this->workGm.dGhGm, this->ubTmp2, gmOffGi, bRows, GRU_GATE_SIZE * hLen,
                   this->H * GRU_GATE_SIZE - GRU_GATE_SIZE * hLen, GRU_GATE_SIZE * hAligned, true);
    }
    // DhPrevUbPath 快路径下 dhFromHGm 无 GM 读者（Accumulate 直取 ubHPrev、
    // DcciStepSlices 已跳过）——MTE3 盲写出链。慢路径（多 bTile/变长）保留 GM 写。
    if (!this->DhPrevUbPath()) {
        CopyOutRow(this->workGm.dhFromHGm, this->ubHPrev, batchHOff, bRows, hLen, 0, hAligned, true);
    }
    SyncM3toV();
}

// grad_h_prev = dgateMM_result(fp32) + dh_prev_from_h -> round 回输入 dtype 写 dhPrevWs
__aicore__ inline void AccumulateDhPrev(int64_t tIdx)
{
    const int64_t bTile = this->vecBTile_, hTile = this->vecHTile_;
    int64_t mStart = 0, mCnt = 0;
    if (!this->GetCoreRows(mStart, mCnt)) {
        return;
    }
    if (this->tiling->isSeqLength) {
        const bool isReverse = (this->tiling->direction != 0);
        int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
        int64_t curBatch = this->inputGm.batchSizesGm.GetValue(gateIdx);
        if (mStart >= curBatch) {
            return;
        }
        if (mStart + mCnt > curBatch) {
            mCnt = curBatch - mStart;
        }
    }
    if (this->DhPrevUbPath()) {
        const int64_t hAligned = ((this->H + this->alignMask) / this->alignPara) * this->alignPara;
        CopyInRow(this->dhPrevF32Gm, this->ubTmp, mStart * this->H, mCnt, this->H, hAligned, 0);
        SyncM2toV();
        // dhFromH 常驻缓冲按计算路径选择（合并路径在 ubHPrev，旧路径在 ubDHPrevFromH）
        Add(this->ubGradH, this->dhFromHInHp_ ? this->ubHPrev : this->ubDHPrevFromH, this->ubTmp, mCnt * hAligned);
        return;
    }
    int64_t bTiles = (mCnt + bTile - 1) / bTile;
    int64_t hTiles = (H + hTile - 1) / hTile;
    for (int64_t bT = 0; bT < bTiles; bT++) {
        int64_t bOff = bT * bTile, bRows = bTile;
        if (bT == bTiles - 1)
            bRows = mCnt - bOff;
        for (int64_t hT = 0; hT < hTiles; hT++) {
            int64_t hOff = hT * hTile, hLen = hTile;
            if (hT == hTiles - 1)
                hLen = H - hOff;
            int64_t hAligned = ((hLen + this->alignMask) / this->alignPara) * this->alignPara;
            // dhPrevF32 为 fp32 workspace：直接 fp32 搬入（rowAlign 为 fp32 元素距）
            CopyInRow(this->dhPrevF32Gm, this->ubTmp, (mStart + bOff) * H + hOff, bRows, hLen, hAligned, 0);
            CopyInRow(this->workGm.dhFromHGm, this->ubTmp2, (mStart + bOff) * H + hOff, bRows, hLen, hAligned, 0);
            SyncM2toV();
            Add(this->ubTmp3, this->ubTmp, this->ubTmp2, bRows * hAligned);
            SyncVtoM3();
            CopyOutRow(this->workGm.dhPrevWsGm, this->ubTmp3, (mStart + bOff) * H + hOff, bRows, hLen, 0, hAligned);
            SyncM3toM2();
        }
    }
}

__aicore__ inline void StoreDhPrev()
{
    const int64_t bTile = this->vecBTile_, hTile = this->vecHTile_;
    int64_t mStart = 0, mCnt = 0;
    if (!this->GetCoreRows(mStart, mCnt)) {
        return;
    }
    if (this->DhPrevUbPath()) {
        const int64_t hAligned = ((this->H + this->alignMask) / this->alignPara) * this->alignPara;
        CopyOutRow(this->outputGm.dhPrevGm, this->ubGradH, mStart * this->H, mCnt, this->H, 0, hAligned);
        return;
    }
    int64_t bTiles = (mCnt + bTile - 1) / bTile;
    int64_t hTiles = (H + hTile - 1) / hTile;
    for (int64_t bT = 0; bT < bTiles; bT++) {
        int64_t bOff = bT * bTile, bRows = bTile;
        if (bT == bTiles - 1)
            bRows = mCnt - bOff;
        for (int64_t hT = 0; hT < hTiles; hT++) {
            int64_t hOff = hT * hTile, hLen = hTile;
            if (hT == hTiles - 1)
                hLen = H - hOff;
            int64_t hAligned = ((hLen + this->alignMask) / this->alignPara) * this->alignPara;
            CopyInRow(this->workGm.dhPrevWsGm, this->ubTmp, (mStart + bOff) * H + hOff, bRows, hLen, hAligned, 0);
            SyncM2toM3();
            CopyOutRow(this->outputGm.dhPrevGm, this->ubTmp, (mStart + bOff) * H + hOff, bRows, hLen, 0, hAligned);
            SyncM3toM2();
        }
    }
}

// ------------------------------------------------------------------------------------------------
// TransposeGiGhT：dGi/dGh [TS,3H] -> dGiT/dGhT [3H,TS]（dW GEMM 的 A 侧，输入 dtype 存储）。
// 16 行分组（TS 维）× 列逐根处理：
//   对每个输出行 m（=源列 m）：CopyInRow 按列收集 rCnt 个元素（GM 行距 cols，length=1 逐元素
//   fallback 路径）落 UB（行距 8 fp32 元素），再 CopyOutRow 按行散布到 dstGm[m*TS + r0..]。
// 注意：写侧须按 dst tile 行基址取数——若按源行基址取数，落进 dst 的是源的
// "行片段"而非"列"，dGiT 实为 dGi 的行拷贝（f2a ws 实证 dGiT[0:4]==dGi[0:4]），dW 两路 A 侧全错。
// 每任务写完即对本块 DataCacheCleanAndInvalid（cv-sync 纪律 5：生产者侧刷新）。
// ------------------------------------------------------------------------------------------------
// A3: 条带 Gather 偏移表（字节粒度，固定与 run 无关）: staged[r*SC+m] = loaded[m*8+r]。
// 载入区 [rowsThis×8] 行距 8；staged 区 [8×SC] 行距 SC；尾 run 的 m≥rowsThis 项聚到
// 暂存垃圾尾（M3 按 rowsThis 截断不写出）。
__aicore__ inline void InitTransGatherOffsets()
{
    auto offView = this->ubTmp2.template ReinterpretCast<uint32_t>();
    for (int64_t r = 0; r < TRANS_TILE_COLS; ++r) {
        for (int64_t m = 0; m < TRANS_STRIP_COLS; ++m) {
            // 字节粒度偏移: staged[r*SC+m] = loaded[m*TRANS_TILE_COLS+r]
            offView.SetValue(r * TRANS_STRIP_COLS + m, static_cast<uint32_t>((m * TRANS_TILE_COLS + r) * FLOAT_BYTES));
        }
    }
}

__aicore__ inline void TransposeOnePlane(GlobalTensor<DTYPE>& srcGm, GlobalTensor<DTYPE>& dstGm, int64_t rows,
                                         int64_t cols, int64_t colStart, int64_t colEnd, int64_t localIdx,
                                         int64_t pairStride, int64_t dstRowPitch = 0)
{
    // 列带归属：本簇仅处理源列 [colStart, colEnd)（= dst 行带），生产/消费同簇闭环。
    // dstRowPitch: dst 平面行距——默认等于 rows(整面转置); D6 分段直读时传全量 TS
    // (dst 为 [cols, TS] 大平面的列偏移视图, 行距与段内行数不同)。
    const int64_t dstPitch = (dstRowPitch > 0) ? dstRowPitch : rows;
    const int64_t groups = (rows + TRANS_GROUP - 1) / TRANS_GROUP;
    const int64_t bandCols = colEnd - colStart;
    // ---- fp32 标量混合转置：[16 行 x 8 列] tile -------------------------------------------
    // 原 per-column 路径为大量微搬运+事件，MTE3 成为瓶颈。
    // 新路径：1 次多块 MTE2 载入（8 列 = 32B 块，快路径，srcStride 为间隙语义 = (cols-8)*4）
    //         + 片内标量转置（目的行 16 元连续，标量 store 写合并；标量访存按 4-8 周期/元素
    //           估算 ~153K 元素/AIV ≈ 1ms，对比 MTE 微事务 ~17ms）。
    // fp16 保持原路径（CopyInRow 多行 half 暂存存在 pitch 错配，标量 SetValue 路径后续可平移）。
    if constexpr (sizeof(DTYPE) == 4) {
        // ⚠ rows 非 16 整除(I=520 实证: Gather 批次残留 6-9 行 [0,8) 位零元素, 根因未定位)——
        // 整体回退逐列 PROVEN 路径(慢但正确); 16 对齐 shape 保持 v4+Gather。
        if (rows % TRANS_GROUP != 0) {
            const int64_t totalCols = colEnd - colStart;
            for (int64_t task = localIdx; task < groups * totalCols; task += pairStride) {
                const int64_t g = task / totalCols;
                const int64_t m = colStart + task % totalCols;
                const int64_t r0 = g * TRANS_GROUP;
                const int64_t rCnt = (rows - r0 < TRANS_GROUP) ? (rows - r0) : TRANS_GROUP;
                CopyInRow(srcGm, this->ubTmp, r0 * cols + m, rCnt, 1, this->alignPara, cols - 1);
                SyncM2toV();
                CopyOutRow(dstGm, this->ubTmp, m * dstPitch + r0, rCnt, 1, 0, this->alignPara);
                SyncM3toM2();
            }
            GlobalTensor<DTYPE> dstView;
            dstView.SetGlobalBuffer(dstGm.GetPhyAddr(colStart * dstPitch), (colEnd - colStart) * dstPitch);
            DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(dstView);
            return;
        }
        constexpr int64_t TILE_C = TRANS_TILE_COLS; // 8 列 tile（fp32 恰为 32B 块）
        constexpr int64_t STAGE_OFF = 2048;
        constexpr int64_t SC = TRANS_STRIP_COLS;
        // A3: 条带化转置——[8 列 × SC 行] run: 1 MTE2 + 1 Gather + 1 M3。
        // 消融实证 v4 每-tile 微事务结构使转置占内核 49.2%(3.8ms, 有效带宽仅 ~10GB/s);
        // 条带合并后微事务数 /31(xT) ~ /16(hPrevT), 且 8 对齐列带分配消除 4B 尾列中继
        // (bench: xT band=5 列全中继 / hPrevT band=10 列 2 列中继——微事务风暴主源)。
        // 同步链: [load → M2→V → Gather → V→M3 → M3 → M3→M2], 逐 run 的 M3→M2 经
        // [MTE3→MTE2→V] 传递序护下 run Gather 对暂存区的覆写（与 v4 批结构同构）。
        const int64_t fullSub = bandCols / TILE_C;
        const int64_t tailCols = bandCols % TILE_C;
        const int64_t runsPerSub = (rows + SC - 1) / SC;
        const int64_t totalRuns = fullSub * runsPerSub;
        for (int64_t run = localIdx; run < totalRuns; run += pairStride) {
            const int64_t sb = run / runsPerSub;
            const int64_t rc = run % runsPerSub;
            const int64_t c0 = colStart + sb * TILE_C;
            const int64_t r0 = rc * SC;
            const int64_t rowsThis = (rows - r0 < SC) ? (rows - r0) : SC;
            // 载入 [rowsThis × 8]（块长 32B 多行快路径，单 MTE2），行距 8 落 ubTmp[0,2048)
            CopyInRow(srcGm, this->ubTmp, r0 * cols + c0, rowsThis, TILE_C, TILE_C, cols - TILE_C);
            SyncM2toV();
            // 条带置换: staged[r*SC+m] = loaded[m*8+r]（表固定; 尾 run 垃圾尾 M3 截断）
            Gather(this->ubTmp[STAGE_OFF], this->ubTmp, this->ubTmp2.template ReinterpretCast<uint32_t>(),
                   static_cast<uint32_t>(0), static_cast<uint32_t>(TILE_C * SC));
            SyncVtoM3();
            // 条带写出: dst 行 [c0, c0+8) × 列 [r0, r0+rowsThis)——单 M3
            // (UB 侧 srcStride 单位 32B 块——CopyOutRow 同款; GM 侧字节)
            DataCopyExtParams cp(static_cast<uint16_t>(TILE_C), static_cast<uint32_t>(rowsThis * FLOAT_BYTES),
                                 static_cast<int64_t>((SC - rowsThis) * FLOAT_BYTES / BLOCK_BYTES),
                                 static_cast<int64_t>(dstPitch - rowsThis) * FLOAT_BYTES, 0);
            DataCopyPad(dstGm[c0 * dstPitch + r0], this->ubTmp[STAGE_OFF], cp);
            SyncM3toM2();
        }
        // 尾列（band 非 8 整除时）：MTE3 写出(SetValue 标量 GM 写实证不可靠——H=42/85/86/129/
        // 170/171 + TS%16==0 尾列 16 元窗口零化/错位, dwHh 错而 dwIh/dx/dh0 全对)。
        constexpr int64_t TAIL_GRP = 8; // 尾列中继槽位数（组内 8 列各占独立槽，与 TRANS_TILE_COLS 同宽）
        const int64_t slotElems = TRANS_GROUP * this->alignPara;
        int64_t pendCnt[TAIL_GRP];
        int64_t pendR0[TAIL_GRP];
        int64_t pendM[TAIL_GRP];
        int64_t staged = 0;
        for (int64_t task = localIdx; task < groups * tailCols; task += pairStride) {
            const int64_t g = task / tailCols;
            const int64_t m = colStart + fullSub * TILE_C + task % tailCols;
            const int64_t r0 = g * TRANS_GROUP;
            const int64_t rCnt = (rows - r0 < TRANS_GROUP) ? (rows - r0) : TRANS_GROUP;
            // [性能] 批量中继: 组内 8 列各占独立槽位(staged 顺序发射裸 MTE2), 一次 MTE2->MTE3
            // 直通窄事件后顺序发射裸 MTE3 写出, 组尾一次 MTE3->MTE2 排空(护下组槽位复用)。
            // 原泛化期逐列 CopyInRow+SyncM2toV+CopyOutRow(SyncVtoM3)+SyncM3toM2 = 3 事件/列,
            // hPrevT 尾列跨簇×多行组场景的尾列微搬运
            // (bench 7.3→11.7ms 主因); SetValue 旧版快但标量 GM 写实证不可靠(H=42/85/86/
            // 129/170/171 尾列 16 元窗口零化/错位)——保持 MTE3 写出语义, 仅重组事件结构。
            if (staged == TAIL_GRP) {
                // 纯 MTE 中继: MTE2->MTE3 直通窄事件。V 标量紧致化(GetValue/SetValue)两式实证
                // 均非确定竞态(492 rel 0.2-0.8 / 219 rel 7e+13, 加 PIPE_ALL 无效——标量写 UB
                // 对 MTE3 读疑似可见性问题非定序问题), 弃用; 16 条 4B 微事务/列为已证安全代价。
                SyncM2toM3();
                for (int64_t i = 0; i < staged; ++i) {
                    const LocalTensor<float> slot = this->ubTmp3[i * slotElems];
                    for (int64_t r = 0; r < pendCnt[i]; ++r) {
                        DataCopyExtParams cp(1, static_cast<uint32_t>(FLOAT_BYTES), 0, 0, 0);
                        DataCopyPad(dstGm[pendM[i] * dstPitch + pendR0[i] + r], slot[r * this->alignPara], cp);
                    }
                }
                SyncM3toM2();
                staged = 0;
            }
            {
                const LocalTensor<float> slot = this->ubTmp3[staged * slotElems];
                for (int64_t r = 0; r < rCnt; ++r) {
                    DataCopyExtParams cp(1, static_cast<uint32_t>(FLOAT_BYTES), 0, 0, 0);
                    DataCopyPadExtParams<float> pp(true, 0, 0, 0);
                    DataCopyPad(slot[r * this->alignPara], srcGm[r0 * cols + m + r * cols], cp, pp);
                }
                pendCnt[staged] = rCnt;
                pendR0[staged] = r0;
                pendM[staged] = m;
                ++staged;
            }
        }
        if (staged > 0) {
            SyncM2toM3();
            for (int64_t i = 0; i < staged; ++i) {
                const LocalTensor<float> slot = this->ubTmp3[i * slotElems];
                for (int64_t r = 0; r < pendCnt[i]; ++r) {
                    DataCopyExtParams cp(1, static_cast<uint32_t>(FLOAT_BYTES), 0, 0, 0);
                    DataCopyPad(dstGm[pendM[i] * dstPitch + pendR0[i] + r], slot[r * this->alignPara], cp);
                }
            }
            SyncM3toM2();
        }
        // 纪律 5：本核写过的 dst 行带（[colStart, colEnd) x [0, rows) 连续区），终轮 V2C 前一次刷新
        if (bandCols > 0) {
            GlobalTensor<DTYPE> dstView;
            dstView.SetGlobalBuffer(dstGm.GetPhyAddr(colStart * rows), bandCols * rows);
            DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(dstView);
        }
        return;
    }
    // ---- fp16：原 per-column 路径（MTE 微事务，待标量路径平移） ---------------------------
    const int64_t totalTasks = groups * bandCols;
    for (int64_t task = localIdx; task < totalTasks; task += pairStride) {
        const int64_t g = task / bandCols;
        const int64_t m = colStart + task % bandCols;
        const int64_t r0 = g * TRANS_GROUP;
        const int64_t rCnt = (rows - r0 < TRANS_GROUP) ? (rows - r0) : TRANS_GROUP;
        CopyInRow(srcGm, this->ubTmp, r0 * cols + m, rCnt, 1, this->alignPara, cols - 1);
        SyncM2toV();
        CopyOutRow(dstGm, this->ubTmp, m * dstPitch + r0, rCnt, 1, 0, this->alignPara);
        SyncM3toM2();
        GlobalTensor<DTYPE> dstView;
        dstView.SetGlobalBuffer(dstGm.GetPhyAddr(m * rows + r0), rCnt);
        DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(dstView);
    }
}

__aicore__ inline void TransposeGiGhT()
{
    const int64_t ts = this->totalSteps_;
    const int64_t cluster = static_cast<int64_t>(GetBlockIdx()) / AIV_PER_AIC;
    const int64_t localIdx = static_cast<int64_t>(GetBlockIdx()) % AIV_PER_AIC;
    const int64_t numClusters = static_cast<int64_t>(GetBlockNum());
    this->InitTransGatherOffsets();
    // dW^T 重排: 只转置 x[TS,I]->xT[I,TS] 与 hPrevWs[TS,H]->hPrevT[H,TS](体积 6x/3x↓)
    // A3: 8 列子带对齐分配——消非对齐列带的 4B 尾列中继（旧任意宽 band 使 bench 的 xT
    // band=5 列全走中继 / hPrevT band=10 列 2 列中继，为微事务风暴主源）。子带数
    // subBands=cols/8 均衡分布到活跃簇；真尾列（cols%8）归最后活跃簇走中继路径。
    {
        const int64_t cols = this->I;
        const int64_t subBands = cols / TRANS_TILE_COLS; // 8 列子带
        int64_t activeClusters = (numClusters < subBands) ? numClusters : subBands;
        activeClusters = (activeClusters < 1) ? 1 : activeClusters;
        const int64_t per = subBands / activeClusters;
        const int64_t rem = subBands % activeClusters;
        if (cluster < activeClusters) {
            const int64_t sb0 = cluster * per + (cluster < rem ? cluster : rem);
            const int64_t sbCnt = per + (cluster < rem ? 1 : 0);
            const int64_t c0 = sb0 * TRANS_TILE_COLS;
            int64_t c1 = c0 + sbCnt * TRANS_TILE_COLS;
            if (cluster == activeClusters - 1) {
                c1 = cols; // 真尾列归最后活跃簇
            }
            if (c0 < c1) {
                this->TransposeOnePlane(this->inputGm.xGm, this->xTGm, ts, cols, c0, c1, localIdx, AIV_PER_AIC);
            }
        }
    }
    {
        const int64_t cols = this->H;
        const int64_t subBands = cols / TRANS_TILE_COLS; // 8 列子带
        int64_t activeClusters = (numClusters < subBands) ? numClusters : subBands;
        activeClusters = (activeClusters < 1) ? 1 : activeClusters;
        const int64_t per = subBands / activeClusters;
        const int64_t rem = subBands % activeClusters;
        if (cluster < activeClusters) {
            const int64_t sb0 = cluster * per + (cluster < rem ? cluster : rem);
            const int64_t sbCnt = per + (cluster < rem ? 1 : 0);
            const int64_t c0 = sb0 * TRANS_TILE_COLS;
            int64_t c1 = c0 + sbCnt * TRANS_TILE_COLS;
            if (cluster == activeClusters - 1) {
                c1 = cols;
            }
            if (c0 < c1) {
                this->TransposeOnePlane(this->workGm.hPrevWsGm, this->hPrevTGm, ts, cols, c0, c1, localIdx,
                                        AIV_PER_AIC);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// ConvertOutputs：fp32 workspace -> 输入 dtype（dwI/dwH/dx）。
// 按 AIV 行切片；每行 CopyInRow(fp32) + CopyOutRow(DTYPE)。
// ------------------------------------------------------------------------------------------------
__aicore__ inline void ConvertOnePlane(GlobalTensor<float>& srcGm, GlobalTensor<DTYPE>& dstGm, int64_t rows,
                                       int64_t cols, int64_t rowStart, int64_t rowEnd, int64_t localIdx,
                                       int64_t pairStride)
{
    // 行带归属：与 RunBatchedGmms 的 C 平面行带一致，同簇生产同簇消费。
    // fp32 行分批：每块 rowsPerBlock 行整块读写（多行 32B 对齐快路径，事件数随块数而非行数下降；
    // 原 fp32 逐行 3 事件 × ~12.8K 行/核是尾部大头之一）。
    // ⚠ 仅 fp32：fp16 的 CopyInRow/CopyOutRow 多行 staging 存在 pitch 错配（half 暂存行距
    // pitchBytes/dSize=2*cols 与 Cast 的连续读模式不符），保持逐行。
    // ⚠ 宽行(cols>allocLength, 即 I>4096)：rowsPerBlock=allocLength/cAl=0 → 首块 rCnt<0 直接
    // break, 平面从未回投(I=)；fp16 逐行同宽越界 ubTmp。
    // 修复: 列方向按 chunk(allocLength 内 alignPara 倍数)分块, 单行单块逐块回投。
    const int64_t cAl = (cols + this->alignMask) / this->alignPara * this->alignPara;
    if (cAl > this->allocLength) {
        const int64_t chunk = (this->allocLength / this->alignPara) * this->alignPara;
        for (int64_t r = rowStart + localIdx; r < rowEnd; r += pairStride) {
            for (int64_t c0 = 0; c0 < cols; c0 += chunk) {
                const int64_t cCnt = (cols - c0 < chunk) ? (cols - c0) : chunk;
                const int64_t cAligned = ((cCnt + this->alignMask) / this->alignPara) * this->alignPara;
                CopyInRow(srcGm, this->ubTmp, r * cols + c0, 1, cCnt, cAligned, 0);
                SyncM2toV();
                CopyOutRow(dstGm, this->ubTmp, r * cols + c0, 1, cCnt, 0, cAligned);
                SyncM3toM2();
            }
        }
        return;
    }
    if constexpr (sizeof(DTYPE) == 4) {
        // ⚠ UB 行距必须 32B 对齐: 原rowAlign=cols 在 cols%8≠0 时多行慢路径 ub[cols*r] 未对齐
        // → MTE 未对齐 UB 访问 → AIV 陷阱 → AIC 等待超时(
        const int64_t rowsPerBlock = this->allocLength / (cAl > 0 ? cAl : 1);
        const int64_t blockStride = rowsPerBlock * pairStride;
        for (int64_t r0 = rowStart + localIdx * rowsPerBlock; r0 < rowEnd; r0 += blockStride) {
            const int64_t rCnt = (rowEnd - r0 < rowsPerBlock) ? (rowEnd - r0) : rowsPerBlock;
            if (rCnt <= 0) {
                break;
            }
            CopyInRow(srcGm, this->ubTmp, r0 * cols, rCnt, cols, cAl, 0);
            SyncM2toV();
            CopyOutRow(dstGm, this->ubTmp, r0 * cols, rCnt, cols, 0, cAl);
            SyncM3toM2();
        }
    } else {
        for (int64_t r = rowStart + localIdx; r < rowEnd; r += pairStride) {
            CopyInRow(srcGm, this->ubTmp, r * cols, 1, cols, cols, 0);
            SyncM2toV();
            CopyOutRow(dstGm, this->ubTmp, r * cols, 1, cols, 0, cols);
            // ⚠ 下一迭代 CopyInRow（MTE2 写 ubTmp）与本迭代 CopyOutRow（MTE3 读 ubTmp）分属不同 pipe，
            // SyncM3toV 只拦 V 管线、拦不住 MTE2 队列——确定性 WAR（）。补 MTE3->MTE2。
            SyncM3toM2();
        }
    }
}

// 转置回投: src 为 dW^T [srcRows, srcCols](fp32), dst 为 dW [srcCols, srcRows](DTYPE)
__aicore__ inline void ConvertTransposedPlane(GlobalTensor<float>& srcGm, GlobalTensor<DTYPE>& dstGm, int64_t srcRows,
                                              int64_t srcCols, int64_t rowStart, int64_t rowEnd, int64_t localIdx,
                                              int64_t pairStride)
{
    // 逐列分块转置回投：fp32 src [srcRows, srcCols] -> DTYPE dst [srcCols, srcRows]。
    // ⚠ 原批量行分块版缺陷（已废弃）：staging 基址 ubTmp[k*srcRows] 与元素行距 alignPara(8) 不匹配，
    //   列 k 驻留区间 [k*srcRows, k*srcRows + srcRows*8) 必然覆写列 k+1 的基址区（r≥srcRows/8 全被踩），
    //   且 rCnt 列 × pitch8 总跨度超 ubTmp 。
    //   改为 HEAD TransposeOnePlane 同构：逐列 + 块高 stageRows，单列驻留 [0, stageRows*8) 恰满预算。
    const int64_t stageRows = this->allocLength / this->alignPara; // pitch8 下单列驻留上限（=512）
    for (int64_t j = rowStart + localIdx; j < rowEnd; j += pairStride) {
        for (int64_t r0 = 0; r0 < srcRows; r0 += stageRows) {
            const int64_t rCnt = (srcRows - r0 < stageRows) ? (srcRows - r0) : stageRows;
            // 按列收集：src 第 j 列的 [r0, r0+rCnt)（fp32 直读，fp16 由 CopyOutRow 出向 Cast）
            CopyInRow(srcGm, this->ubTmp, j + r0 * srcCols, rCnt, 1, this->alignPara, srcCols - 1);
            SyncM2toV();
            // 按行散布：dst 第 j 行的 [r0, r0+rCnt)，GM 行距 1（连续）
            CopyOutRow(dstGm, this->ubTmp, j * srcRows + r0, rCnt, 1, 0, this->alignPara);
            SyncM3toM2();
        }
    }
}

__aicore__ inline void ConvertOutputs()
{
    const int64_t cols3H = this->H * GRU_GATE_SIZE;
    const int64_t cluster = static_cast<int64_t>(GetBlockIdx()) / AIV_PER_AIC;
    const int64_t localIdx = static_cast<int64_t>(GetBlockIdx()) % AIV_PER_AIC;
    const int64_t numClusters = static_cast<int64_t>(GetBlockNum());
    // A6: 8 列子带对齐分配——消 dW 转置回投的尾列 4B 微事务中继
    // (原 band3H=ceil(768/28)=28 非 8 倍数 → 4 尾列/簇 × 28 簇 × 2 面 ≈ 682 微 MTE3, ESL 实证)
    const int64_t subBands3H = cols3H / TRANS_TILE_COLS; // 8 列子带
    int64_t activeClusters = (numClusters < subBands3H) ? numClusters : subBands3H;
    activeClusters = (activeClusters < 1) ? 1 : activeClusters;
    const int64_t per = subBands3H / activeClusters;
    const int64_t rem = subBands3H % activeClusters;
    if (cluster < activeClusters) {
        const int64_t sb0 = cluster * per + (cluster < rem ? cluster : rem);
        const int64_t sbCnt = per + (cluster < rem ? 1 : 0);
        const int64_t r0 = sb0 * TRANS_TILE_COLS;
        int64_t r1 = r0 + sbCnt * TRANS_TILE_COLS;
        if (cluster == activeClusters - 1) {
            r1 = cols3H; // 真尾列(cols3H%8) 归最后活跃簇
        }
        if (r0 < r1) {
            if constexpr (sizeof(DTYPE) == 4) {
                this->InitTransGatherOffsets(); // ubTmp2 偏移表可能被 BiasReduce 覆盖, 重初始化
                this->TransposeOnePlane(this->dwIF32Gm, this->outputGm.dwInputGm, this->I, cols3H, r0, r1, localIdx,
                                        AIV_PER_AIC);
                this->TransposeOnePlane(this->dwHF32Gm, this->outputGm.dwHiddenGm, this->H, cols3H, r0, r1, localIdx,
                                        AIV_PER_AIC);
            } else {
                this->ConvertTransposedPlane(this->dwIF32Gm, this->outputGm.dwInputGm, this->I, cols3H, r0, r1,
                                             localIdx, AIV_PER_AIC);
                this->ConvertTransposedPlane(this->dwHF32Gm, this->outputGm.dwHiddenGm, this->H, cols3H, r0, r1,
                                             localIdx, AIV_PER_AIC);
            }
        }
    }
    // fp32 时 dx 由 GEMM 直写输出(RunDxGemm), 跳过回投
    if constexpr (sizeof(DTYPE) != 4) {
        const int64_t TS = this->totalSteps_;
        const int64_t bandTS = (TS + numClusters - 1) / numClusters;
        const int64_t d0 = cluster * bandTS;
        if (d0 < TS) {
            const int64_t d1 = (TS - d0 < bandTS) ? TS : (d0 + bandTS);
            this->ConvertOnePlane(this->dxF32Gm, this->outputGm.dxGm, TS, this->I, d0, d1, localIdx, AIV_PER_AIC);
        }
    }
}

__aicore__ inline void ProcessBiasReduce(GlobalTensor<DTYPE>& srcGm, GlobalTensor<DTYPE>& dstGm, int64_t rows,
                                         int64_t cols)
{
    SyncM3toV();
    int64_t nReduceCnt = this->tiling->nReduceCnt;
    int64_t singleCoreReduceN = this->tiling->singleCoreReduceN;
    if (nReduceCnt <= 0)
        return;
    if (static_cast<int64_t>(GetBlockIdx()) >= nReduceCnt)
        return;
    int64_t nIdx = GetBlockIdx();
    int64_t nStart = nIdx * singleCoreReduceN;
    int64_t nCnt = (nIdx == nReduceCnt - 1) ? this->tiling->singleCoreReduceNTail : singleCoreReduceN;
    if (nStart >= cols || nCnt <= 0)
        return;
    if (nStart + nCnt > cols)
        nCnt = cols - nStart;

    int64_t maxNOnce = allocLength;

    for (int64_t cStart = 0; cStart < nCnt; cStart += maxNOnce) {
        int64_t cCnt = (cStart + maxNOnce > nCnt) ? (nCnt - cStart) : maxNOnce;
        int64_t cAligned = ((cCnt + this->alignMask) / this->alignPara) * this->alignPara;
        int64_t cGlobalOff = nStart + cStart;

        Duplicate(this->ubTmp, 0.0f, cAligned);
        SyncVtoM2();
        // fp32 行分批：每批 ROW_BATCH 行整块载入（多行快路径）后 UB 内 V 管逐行累加
        // （950PR 同管 RAW 互锁免事件），事件数降 ROW_BATCH 倍——原逐行 2 事件
        // × 2 平面是尾部 scalar 开销大头。fp16 受 CopyInRow 多行 staging pitch 限制保持逐行。
        if constexpr (sizeof(DTYPE) == 4) {
            const int64_t rowBatch = (this->allocLength / cAligned) > 0 ? (this->allocLength / cAligned) : 1;
            for (int64_t r0 = 0; r0 < rows; r0 += rowBatch) {
                const int64_t rCnt = (rows - r0 < rowBatch) ? (rows - r0) : rowBatch;
                // ⚠ 快路径 srcStride 为【间隙】语义(字节): 列切片行距 cols, 块长 cCnt,
                // 间隙 = (cols - cCnt) 元素。gmStride=cols(全 pitch)或 0(连续)皆错——
                // dx 回披(整行连续, 间隙 0)与转置(慢路径, 元素 pitch)是两套语义, 勿混。
                CopyInRow(srcGm, this->ubTmp2, r0 * cols + cGlobalOff, rCnt, cCnt, cAligned, cols - cCnt);
                SyncM2toV();
                for (int64_t r = 0; r < rCnt; ++r) {
                    Add(this->ubTmp, this->ubTmp, this->ubTmp2[r * cAligned], cAligned);
                }
                SyncVtoM2();
            }
        } else {
            for (int64_t r = 0; r < rows; ++r) {
                CopyInRow(srcGm, this->ubTmp2, r * cols + cGlobalOff, 1, cCnt, cAligned, 0);
                SyncM2toV();
                Add(this->ubTmp, this->ubTmp, this->ubTmp2, cAligned);
                SyncVtoM2();
            }
        }
        SyncVtoM3();
        CopyOutRow(dstGm, this->ubTmp, cGlobalOff, 1, cCnt, 0, cAligned);
        SyncM3toV();
    }
}
