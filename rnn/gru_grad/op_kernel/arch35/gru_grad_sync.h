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
 * \file gru_grad_sync.h
 * \brief GRU 反向（dav_3510）跨核/核内同步控制面。
 *
 * 无 KFC 后全部同步显式化，遵循以下同步纪律：
 *  1. 每个 AIV 参加每一轮握手，包括没有活干的（本文件两侧循环形状静态配对）。
 *  2. V->C 是栅栏不是计数信号量：双 AIV set、AIC 恰好消费一次（1:1 / 2:2 均实测死锁）。
 *  3. 轮次静态配对：AIC/AIV 两侧同一 for 形状，任何 continue 不得跳过 flag。
 *  4. 跨核 flag 只约束所在 pipe：CubeWaitVec 后补 PipeBarrier<PIPE_ALL>。
 *  5. flag 不刷新缓存：AIV 写 GM 后、set 前，对写过的 GM 切片 DataCacheCleanAndInvalid。
 *  6. 一轮握手兼任 WAR 栅栏（AIC 第 t+1 步写 dhPrevF32 前，AIV 第 t 步 Accumulate 已完成）。
 *
 * 方向不对称（实测，不得"对称化"）：
 *  - C->V：普通 set/wait，AIC set 一次、双 AIV 都能等到（PIPE_FIX 发射）。
 *  - V->C：栅栏语义，双 AIV 都 set、AIC 消费一次（PIPE_MTE3 发射 / PIPE_M 等待）。
 */
#ifndef _GRU_GRAD_ARCH35_SYNC_H_
#define _GRU_GRAD_ARCH35_SYNC_H_

#include "kernel_operator.h"

namespace ggsync {

// 跨核事件 ID（与基线实现的 SYNC_* 同值，便于对照）
constexpr uint16_t FLAG_C2V = 8;   // cube -> vector：dhPrevF32 就绪
constexpr uint16_t FLAG_V2C = 9;   // vector -> cube：dGi/dGh/hPrevWs 步切片就绪（栅栏）
constexpr uint16_t FLAG_V2C2 = 10; // vector -> cube 终轮第二栅栏：本簇转置完成(与 V2C 分离,
                                   // 同 flag 双 set 双 wait 实证误配对——wait#2 吃旧信号致 dW GEMM 抢跑)
constexpr uint16_t FLAG_C2V_B0 = 5; // 带提前信号——band0 Fixpipe 落地即发（早于 FLAG_C2V）;
                                    // 槽位与保留 ID 无冲突
constexpr uint16_t
    AIV_SUBBLOCK_FLAG_OFFSET = 16; // intra_block 通道 AIV 子块 flag 槽位自动偏移步长（AIV0 用 F、AIV1 用 F+16）
constexpr uint16_t FLAG_C2V_B1 = 6; // 末带提前信号——末带 Fixpipe 落地即发（早于
                                    // FLAG_C2V 一个 PIPE_ALL+set），AIV 据此预发射 band1 gemmRes M2

// CrossCore{Set,Wait}Flag<modeId, pipe> 的通道语义（值与 AscendC 官方一致）：
constexpr uint8_t MODE_INTRA_BLOCK = 4; // 块内通道（AIC<->本块双 AIV，子块自动槽位偏移；同 AscendC INTRA_MODE=4）
constexpr uint8_t MODE_CROSS_CORE = 2; // 跨核全核通道（mode-2 全核消息，SyncAll 同款；终轮全簇转置栅栏用）

// ---- 核内：pipe 间全停（功能优先，成对 Set/Wait）--------------------------------------------
template <AscendC::HardEvent E>
__aicore__ inline void PipeWait(event_t id = EVENT_ID0)
{
    AscendC::SetFlag<E>(id);
    AscendC::WaitFlag<E>(id);
}

__aicore__ inline void WaitMte2ToMte1() { PipeWait<AscendC::HardEvent::MTE2_MTE1>(); } // GM->L1 完成后再 L1->L0
__aicore__ inline void WaitMte1ToM() { PipeWait<AscendC::HardEvent::MTE1_M>(); }       // L1->L0 完成后再 Mmad
__aicore__ inline void WaitMToFix() { PipeWait<AscendC::HardEvent::M_FIX>(); }         // Mmad 完成后再 Fixpipe
__aicore__ inline void WaitFixToM() { PipeWait<AscendC::HardEvent::FIX_M>(); }         // 排空后再复用 L0C
// Mmad 对 L0A/L0B 的读完成 -> 下一 k 块的 SplitA/SplitB(MTE1) 才可复用（窄于 PIPE_ALL 的定向 WAR；
// GruGrad 参考核 fine-sync 变体真机实证等价，）
__aicore__ inline void WaitMToMte1() { PipeWait<AscendC::HardEvent::M_MTE1>(); }
// SplitA/SplitB(MTE1) 对 L1 a1/b1 的读完成 -> 下一 k 块的 CopyInNd2Nz(MTE2 重写 L1) 才可发射。
// 原 PIPE_ALL 顺带覆盖的 MTE1->MTE2 WAR；首轮收窄漏掉此边导致梯度全错（），补齐。
__aicore__ inline void WaitMte1ToMte2() { PipeWait<AscendC::HardEvent::MTE1_MTE2>(); }

// ---- 跨核：CUBE -> VECTOR（普通 set/wait）--------------------------------------------------
// C2V/V2C 换用 intra_block 块内硬件通道（MODE_INTRA_BLOCK）——ffts mode-2 跨核消息实测
// 占步周期约一半；intra_block 为 SyncAllImpl 内部
// AIC<->AIV 同款原语（块内信号量，子块自动槽位偏移）。
// 槽位语义（SyncAllImpl/WaitPreTaskEndV1Impl 权威用法推导）：set/wait 自动按 AIV
// 子块偏移 flagId，AIC 为块主显式双槽（ID+16 偏移，与保留 ID 无冲突）。
__aicore__ inline void CubeSignalVec()
{
    AscendC::CrossCoreSetFlag<MODE_INTRA_BLOCK, PIPE_FIX>(FLAG_C2V);
    AscendC::CrossCoreSetFlag<MODE_INTRA_BLOCK, PIPE_FIX>(FLAG_C2V + AIV_SUBBLOCK_FLAG_OFFSET);
}
__aicore__ inline void VecWaitCube() { AscendC::CrossCoreWaitFlag<MODE_INTRA_BLOCK, PIPE_S>(FLAG_C2V); }

// 带提前信号（band0 Fixpipe+PIPE_ALL 后发射；末带仍走 FLAG_C2V）。
// 语义：仅提前，不改变 FLAG_C2V 的数据完备门——AIV 先 Wait(B0) 再 Wait(C2V)，
// B0 恒先于 C2V 到达（AIC 程序序），计算时机不变。每步每块恰好 1 set / 2 AIV 各 1 wait。
__aicore__ inline void CubeSignalVecBand()
{
    AscendC::CrossCoreSetFlag<MODE_INTRA_BLOCK, PIPE_FIX>(FLAG_C2V_B0);
    AscendC::CrossCoreSetFlag<MODE_INTRA_BLOCK, PIPE_FIX>(FLAG_C2V_B0 + AIV_SUBBLOCK_FLAG_OFFSET);
}
__aicore__ inline void VecWaitCubeBand() { AscendC::CrossCoreWaitFlag<MODE_INTRA_BLOCK, PIPE_S>(FLAG_C2V_B0); }

// 末带提前信号（末带 Fixpipe+PIPE_ALL 后发射，早于 FLAG_C2V）。
// AIV 据此预发射 band1 的 gemmRes M2——装载延迟藏进 B1→C2V 间隔（band1 关键段缩短）。
__aicore__ inline void CubeSignalVecBand1()
{
    AscendC::CrossCoreSetFlag<MODE_INTRA_BLOCK, PIPE_FIX>(FLAG_C2V_B1);
    AscendC::CrossCoreSetFlag<MODE_INTRA_BLOCK, PIPE_FIX>(FLAG_C2V_B1 + AIV_SUBBLOCK_FLAG_OFFSET);
}
__aicore__ inline void VecWaitCubeBand1() { AscendC::CrossCoreWaitFlag<MODE_INTRA_BLOCK, PIPE_S>(FLAG_C2V_B1); }

// ---- 跨核：VECTOR -> CUBE（栅栏：双 AIV set，AIC 消费一次）---------------------------------
__aicore__ inline void VecSignalCube() { AscendC::CrossCoreSetFlag<MODE_INTRA_BLOCK, PIPE_MTE3>(FLAG_V2C); }
__aicore__ inline void CubeWaitVec()
{
    AscendC::CrossCoreWaitFlag<MODE_INTRA_BLOCK, PIPE_S>(FLAG_V2C);
    AscendC::CrossCoreWaitFlag<MODE_INTRA_BLOCK, PIPE_S>(FLAG_V2C + AIV_SUBBLOCK_FLAG_OFFSET);
    // 纪律 4：wait 后 AIC 随后的 MTE1/MTE2 访存可越过先发射，必须补全栅栏
    AscendC::PipeBarrier<PIPE_ALL>();
}
// 终轮第二栅栏（独立 flag ID，语义同 V2C）
__aicore__ inline void VecSignalCube2() { AscendC::CrossCoreSetFlag<MODE_CROSS_CORE, PIPE_MTE3>(FLAG_V2C2); }
__aicore__ inline void CubeWaitVec2()
{
    AscendC::CrossCoreWaitFlag<MODE_CROSS_CORE, PIPE_M>(FLAG_V2C2);
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace ggsync

#endif // _GRU_GRAD_ARCH35_SYNC_H_
