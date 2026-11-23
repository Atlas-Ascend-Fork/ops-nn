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
 * \file block_mmad_fma_mul.h
 * \brief fp32高精度模式vector FMA矩阵乘: C[i][j:j+baseN] = bias[j] + A*B, regbase大小256B, fp32一次处理64个数;
 *        支持A/B转置输入: transA为A tile原生[K,baseM]取向驻留 + VF广播装载编译期步长交换(零开销);
 *        transB为CopyIn阶段两步直转(staging单缓冲原生[N,K]取向搬入 + DataCopyGather转置pass -> BT[K,baseN]),
 *        staging由pass即时释放复用(独立事件槽, MTE2/V流水重叠保持), BT行距恒baseN;
 *        baseN四档(64/128/192/256, 单n-tile装下N的最紧档, n>256多n-tile恒256):
 *        baseN=64为(MR=8,NS=1), 其余为MR=4/NS=baseN/64, 由host的tiling策略产出经tilingdata下发,
 *        Process按(mr_, nsTile)查表分发到编译期模板(多n-tile尾tile按tailN裁剪NS, 尾块<64按整块发射);
 *        B/C的UB行距恒baseN(运行时值, *4B均为256B整数倍), 全掩码, 地址由指针偏移计算;
 *        VF计算体对全布局统一为NN形态(B恒行连续装载), 4布局每k步发射槽逐指令相同;
 *        bias经C乒乓预初始化承载: operator()将CPing填充为bias行广播(无bias清零)、CPong清零
 *        (统一移入k循环j==0: 单k组(小k)免Add免Out时上一tile CopyOut直读CPing, 预填充须等其读完再写,
 *        否则V/MTE3跨管竞态污染上一tile输出), FmaMulKernel全部k组统一"从UB装载C"路径, VF体内零运行时分支
 *        (VF硬约束, 实测分支使编译器对整个计算体保守化, 破坏MTE2/V流水, 性能显著劣化);
 *        transB免Out: CPing兼任全局累加器、CPong为工作块, flush原地累加进CPing后CopyOut直读,
 *        单staging+免B乒乓+免Out使transB宽路径UB预算218KB(248KB内), 转置/非转置同享宽路径性能;
 *        分块累加(高精度, 等价multi_mul的reduceSum后再累加): 多k组按tiling下发的链长singleCoreK周期
 *        flush(flushGroups_个k组, 两条C缓冲链各累计singleCoreK个k元素)——块结果加到全局累加器
 *        (非transB为Out/transB为CPing)后清零重累, 链内乘积只与从小长大的部分和相加, 仅块和碰全局
 *        累加器, 消除K个乘积逐个累到不断长大的结果上的大数吃小数; 链长host按sqrt(K)选优
 *        (误差界(K/L+L)eps在L=sqrt(K)最小), flush点/加法顺序tiling期固定, 结果可复现;
 *        地址计算AddrReg硬件化: B/A装载走(base, AddrReg)双段寻址, VAG装入步长后由VLOOP硬件
 *        每迭代自动增量游标, k循环体内全VF指令、零scalar地址算术
 */

#pragma once
#include "./block_mmad.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_,
          class TileCopy_>
class BlockMmad<MatmulFmaMul<>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using A_T = typename AType::T;
    using B_T = typename BType::T;
    using C_T = typename CType::T;
    // 转置标记由MatmulType编译期推导(builder的LayoutA/LayoutB tag -> TagToTrans -> isTrans)
    static constexpr bool TRANS_A = AType_::isTrans;
    static constexpr bool TRANS_B = BType_::isTrans;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t baseK_{0};
    uint64_t tailM_{0};
    uint64_t tailN_{0};
    uint64_t tailK_{0};
    uint64_t currentK_{0};
    uint64_t loopK_{0};
    bool hasBias_{false};
    // 计算体模板参数(tilingdata下发): MR/NS为host选型产出, kernel按其分发到编译期模板实例
    uint64_t mr_{4};
    uint64_t ns_{1};
    // 分块累加flush周期(k组数, Init由singleCoreK按布局折算): 相邻两次flush间——transB工作块CPong
    // 单缓冲链P组(P = singleCoreK/baseK), 非transB乒乓CPing/CPong各链P/2组(P = 2*singleCoreK/baseK,
    // singleCoreK为baseK倍数时P恒为偶数, 乒乓链对称); 两条链长均为singleCoreK个k元素
    uint64_t flushGroups_{0};
    // transB专用: BT转置tile与gather索引表的UB偏移(Init时按tiling尺寸计算)
    uint64_t ubOffsetBT_{0};
    uint64_t ubOffsetIdx_{0};
    // bias行(128宽, 按tile的n段填充, 无bias为Init预清零的零行)的UB偏移
    uint64_t ubOffsetBias_{0};

    __aicore__ inline BlockMmad()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0x0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0x1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0x2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0x0);
    }

    __aicore__ inline ~BlockMmad()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0x0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0x1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0x2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0x0);
    }

public:
    __aicore__ inline void Init(const TupleShape& shape, const TupleShape& blockInfo, uint64_t tailK, uint64_t baseK,
                                uint64_t loopK, bool hasBias, uint64_t mr, uint64_t ns, uint64_t singleCoreK)
    {
        m_ = Get<DIMENSION_M>(shape);
        n_ = Get<DIMENSION_N>(shape);
        k_ = Get<DIMENSION_K>(shape);
        baseM_ = Get<0>(blockInfo);
        baseN_ = Get<1>(blockInfo);
        baseK_ = baseK;
        tailM_ = baseM_;
        tailN_ = baseN_;
        tailK_ = tailK;
        currentK_ = baseK_;
        loopK_ = loopK;
        hasBias_ = hasBias;
        mr_ = mr;
        ns_ = ns;
        // singleCoreK为单C缓冲一个flush周期内累计的k元素数(host按sqrt(K)选优, 向上对齐baseK倍数);
        // 防御性钳到[baseK, k_](k_ < baseK的loopK==1单k组路径不经flush, 仅保证折算不越界)
        if (singleCoreK < baseK_) {
            singleCoreK = baseK_;
        }
        flushGroups_ = TRANS_B ? singleCoreK / baseK_ : singleCoreK * 2U / baseK_;
        if (flushGroups_ == 0U) {
            flushGroups_ = 1U;
        }
        // bias行接在UB布局末尾(偏移公式与operator()一致), 仅作GM->UB的搬运暂存,
        // 由operator()的预填充阶段将其广播到CPing(无bias则CPing直接清零)
        if constexpr (TRANS_B) {
            InitTranspose();
            ubOffsetBias_ = ubOffsetIdx_ + baseN_;
        } else {
            const uint64_t bTileElems = baseK_ * baseN_;
            ubOffsetBias_ = baseM_ * baseK_ * 2 + bTileElems * 2 + baseM_ * baseN_ * 3;
        }
    }

    __aicore__ inline void SetTailM(int64_t tailM) { tailM_ = tailM; }

    __aicore__ inline void SetTailN(int64_t tailN) { tailN_ = tailN; }

    // transB初始化: BT转置tile与gather索引表(UB布局末尾, 偏移公式与operator()一致),
    // 每核一次, 表Init后只读跨tile复用; SetValue写入与VF体内装载同为Vector管序(硬件保序), 无需跨管同步
    __aicore__ inline void InitTranspose()
    {
        const uint64_t bTileElems = baseN_ * FMA_BSTRIDE;
        // transB布局(单staging免Out): APing + staging + CPing + APong + CPong + BT + idx (+ bias由Init定位)
        ubOffsetBT_ = baseM_ * baseK_ * 2 + bTileElems + baseM_ * baseN_ * 2;
        ubOffsetIdx_ = ubOffsetBT_ + baseK_ * baseN_;
        // idx[lane j] = j * FMA_BSTRIDE(元素粒度, DataCopyGather per-lane索引单位为元素);
        // 表项数=baseN_(128窄2块/256宽4块, 每块64 lane)
        AscendC::LocalTensor<uint32_t> idxTab = ubLocal_.ReinterpretCast<uint32_t>()[ubOffsetIdx_];
        for (uint32_t j = 0U; j < baseN_; ++j) {
            idxTab.SetValue(j, j * FMA_BSTRIDE);
        }
    }

    // 搬入A矩阵分块(tailM, currentK); transA时A存储为[k, m]行优先, tile以[K, baseM]原生取向驻留,
    // 行内右补零至alignM, VF广播地址k*baseM+m由FmaVf编译期步长交换, 与NN指令数逐条相同
    __aicore__ inline void CopyInA(const AscendC::GlobalTensor<float>& aGlobal,
                                   const AscendC::LocalTensor<float>& aubLocal)
    {
        if constexpr (TRANS_A) {
            // M尾垃圾列[tailM, alignM)补零, [alignM, baseM)驻留旧值均不读; K尾垃圾行不搬运
            uint32_t alignM = AscendC::CeilAlign(tailM_, ALIGN_NUM);
            uint8_t rightPadding = static_cast<uint8_t>(alignM - tailM_);
            AscendC::DataCopyPadExtParams<float> copyPadParams{true, 0, rightPadding, 0};
            uint32_t srcStride = static_cast<uint32_t>((m_ - tailM_) * sizeof(float));
            uint32_t dstStride = static_cast<uint32_t>((baseM_ - alignM) / ALIGN_NUM);
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(currentK_),
                                                  static_cast<uint32_t>(tailM_ * sizeof(float)), srcStride, dstStride,
                                                  0};
            AscendC::DataCopyPad(aubLocal, aGlobal, copyParams, copyPadParams);
        } else {
            uint32_t alignK = AscendC::CeilAlign(currentK_, ALIGN_NUM);
            uint8_t rightPadding = static_cast<uint8_t>(alignK - currentK_);
            AscendC::DataCopyPadExtParams<float> copyPadParams{true, 0, rightPadding, 0};
            uint32_t srcStride = static_cast<uint32_t>((k_ - currentK_) * sizeof(float));
            uint32_t dstStride = static_cast<uint32_t>((baseK_ - alignK) / ALIGN_NUM);
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(tailM_),
                                                  static_cast<uint32_t>(currentK_ * sizeof(float)), srcStride,
                                                  dstStride, 0};
            AscendC::DataCopyPad(aubLocal, aGlobal, copyParams, copyPadParams);
        }
    }

    // 搬入B矩阵分块; transB时B存储为[n, k]行优先, 原生[N, BSTRIDE行距]取向搬入staging缓冲(此为
    // 两步直转第一步, 仅MTE2原生搬入), 转置由Process内TransposeTileVfT转pass完成;
    // N尾垃圾行不搬运(驻留旧值, gather索引恒在staging界内安全); K尾垃圾列右补零防御不读
    __aicore__ inline void CopyInB(const AscendC::GlobalTensor<float>& bGlobal,
                                   const AscendC::LocalTensor<float>& bubLocal)
    {
        if constexpr (TRANS_B) {
            uint32_t padK = AscendC::CeilAlign(currentK_, ALIGN_NUM);
            uint8_t rightPadding = static_cast<uint8_t>(padK - currentK_);
            AscendC::DataCopyPadExtParams<float> copyPadParams{true, 0, rightPadding, 0};
            uint32_t srcStride = static_cast<uint32_t>((k_ - currentK_) * sizeof(float));
            uint32_t dstStride = static_cast<uint32_t>((FMA_BSTRIDE - padK) / ALIGN_NUM);
            for (uint32_t row = 0U; row < tailN_; row += FMA_COPY_ROWS) {
                uint32_t rows = (tailN_ - row < FMA_COPY_ROWS) ? (tailN_ - row) : FMA_COPY_ROWS;
                AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(rows),
                                                      static_cast<uint32_t>(currentK_ * sizeof(float)), srcStride,
                                                      dstStride, 0};
                AscendC::DataCopyPad(bubLocal[row * FMA_BSTRIDE], bGlobal[row * k_], copyParams, copyPadParams);
            }
        } else {
            // B为(k, n)行优先, 每行n方向tailN个连续数据, 行宽为baseN_
            uint32_t alignN = AscendC::CeilAlign(tailN_, ALIGN_NUM);
            uint8_t rightPadding = static_cast<uint8_t>(alignN - tailN_);
            AscendC::DataCopyPadExtParams<float> copyPadParams{true, 0, rightPadding, 0};
            uint32_t srcStride = static_cast<uint32_t>((n_ - tailN_) * sizeof(float));
            uint32_t dstStride = static_cast<uint32_t>((baseN_ - alignN) / ALIGN_NUM);
            AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(currentK_),
                                                  static_cast<uint32_t>(tailN_ * sizeof(float)), srcStride, dstStride,
                                                  0};
            AscendC::DataCopyPad(bubLocal, bGlobal, copyParams, copyPadParams);
        }
    }

    // 搬入bias列向量[n]的[nBase, nBase+tailN)段 -> UB单行(baseN宽);
    // 尾部只补零至8对齐(rightPadding*4B<=32B为MTE2 pad硬件上限, 与CopyInB同款模式,
    // 超限会触发DataCopyPad断言trap), [alignN, baseN)驻留旧值不读: bias垃圾列仅落CopyOut裁剪区, 无害;
    // (bias与转置布局无关, 仅与C列对应); 每tile按自身n段搬一次, 同tile的k组经预填充复用
    __aicore__ inline void CopyInBias(const AscendC::GlobalTensor<float>& biasGlobal,
                                      const AscendC::LocalTensor<float>& biasLocal)
    {
        uint32_t alignN = AscendC::CeilAlign(tailN_, ALIGN_NUM);
        uint8_t rightPadding = static_cast<uint8_t>(alignN - tailN_);
        AscendC::DataCopyPadExtParams<float> copyPadParams{true, 0, rightPadding, 0.0f};
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailN_ * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(biasLocal, biasGlobal, copyParams, copyPadParams);
    }

    // bias行广播填充CPing执行体: bias行分NBC块(NBC=baseN/64, 行宽=NBC*64)装载到寄存器后逐行写回,
    // 行递增走AddrReg硬件游标(步长baseN运行时), 循环体零scalar地址计算;
    // 在operator()预填充阶段调用(V管线), 与k循环首组的MTE2搬入并行, 墙钟时间被完全隐藏;
    // V管程序序保证填充先于FmaMulKernel首组从CPing装载, 无需跨管同步; VF硬约束: 单层uint16_t循环
    template <uint32_t NBC>
    static __simd_vf__ inline void BiasFillVfT(__ubuf__ float* dstPtr, __ubuf__ float* biasPtr, uint64_t rows,
                                               uint32_t rowStride)
    {
        AscendC::Reg::RegTensor<float> regTensorB[NBC];
        AscendC::Reg::MaskReg pregAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
#pragma unroll
        for (uint32_t nb = 0; nb < NBC; ++nb) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(regTensorB[nb], biasPtr + nb * FMA_SUB_N);
        }
        for (uint16_t r = 0; r < static_cast<uint16_t>(rows); ++r) {
            AscendC::Reg::AddrReg rAddr = AscendC::Reg::CreateAddrReg<float>(r, rowStride);
#pragma unroll
            for (uint32_t nb = 0; nb < NBC; ++nb) {
                AscendC::Reg::DataCopy<float>(dstPtr + nb * FMA_SUB_N, regTensorB[nb], rAddr, pregAll);
            }
        }
    }

    // 统一计算体: MR行 x NS个64宽子块寄存器分块, B/C的UB行距=baseN(tiling下发的运行时值, 4档
    // 64/128/192/256, 行距*4B均为256B整数倍满足LoadAlign/StoreAlign约束);
    // 每k步发射 NS条B装载 + MR条A广播装载(DIST_BRC_B32) + MR*NS条MulAddDst, 装载/FMA = 1/NS + 1/MR,
    // 寄存器 = MR + NS + MR*NS((4,4)时24个, 低于regbase干净预算25);
    // (MR,NS)由Process标量流按tiling下发的(mr_, nsTile)查表分发到编译期模板(VF体内免运行时分支):
    // baseN=64 -> (8,1); baseN>=128 -> MR=4, NS=baseN/64(多n-tile尾tile由kernel按tailN裁剪nsTile);
    // N尾块<64部分按整块发射(CopyOut裁剪不读, 垃圾无害);
    // 地址计算AddrReg硬件化(VAG游标 + VLOOP自动增量, 消化k循环体内全部scalar算术):
    //   B/A装载走(base, AddrReg)双段式, base为组内循环不变量(m组层预置), k递增由VAG装入游标后
    //   VLOOP每迭代硬件自动增量, k循环体全VF指令零scalar计算;
    // transA: A tile原生[K, baseM]取向驻留, 广播装载编译期步长交换(addr(m,k) = m*aMStride + k*aKStride,
    //         NN:(aStride,1) / TN:(1,aStride)), 指令数与发射槽和NN逐指令相同(零开销);
    // B恒为[K,N]取向(transB转置已在Process转pass完成), 行装载恒为行连续LoadDist, 全布局统一NN形态;
    // VF体内严禁运行时分支(实测会使编译器对整个计算体保守化, 破坏MTE2/V流水): 所有k组一律
    // 从UB装载C——首k组装载到的即operator()预填充的CPing(有bias为bias行广播, 无bias为0),
    // 次k组为预清零的CPong, 其余k组为UB累积值(乒乓接力, 多k组由调用方按周期flush清零重累分块有界),
    // bias语义完全由预填充内容承载
    template <bool TRANS_A, uint32_t MR, uint32_t NS>
    static __simd_vf__ inline void FmaMulKernel(__ubuf__ float* cubPtr, __ubuf__ float* aubPtr, __ubuf__ float* bubPtr,
                                                uint64_t mAub, uint64_t kBub, uint64_t aStride, uint32_t bRowStride)
    {
        const uint32_t aMStride = TRANS_A ? 1U : static_cast<uint32_t>(aStride);
        const uint32_t aKStride = TRANS_A ? static_cast<uint32_t>(aStride) : 1U;
        AscendC::Reg::RegTensor<float> regA[MR];
        AscendC::Reg::RegTensor<float> regB[NS];
        AscendC::Reg::RegTensor<float> regC[MR][NS];
        AscendC::Reg::MaskReg pregAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        uint16_t mNum = static_cast<uint16_t>(mAub);
        // m方向MR行一组, 尾部不足由后续逐行循环处理
        uint16_t mIdx = 0;
        for (; mIdx + MR <= mNum; mIdx += MR) {
            __ubuf__ float* cBase = cubPtr + mIdx * bRowStride;
#pragma unroll
            for (uint32_t idx = 0; idx < MR * NS; ++idx) {
                const uint32_t r = idx / NS;
                const uint32_t nb = idx % NS;
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    regC[r][nb], cBase + r * bRowStride + nb * FMA_SUB_N);
            }
            __ubuf__ float* aRowBase = aubPtr + static_cast<uint32_t>(mIdx) * aMStride;
            for (uint16_t kIdx = 0; kIdx < static_cast<uint16_t>(kBub); ++kIdx) {
                AscendC::Reg::AddrReg bAddr = AscendC::Reg::CreateAddrReg<float>(kIdx, bRowStride);
                AscendC::Reg::AddrReg aAddr = AscendC::Reg::CreateAddrReg<float>(kIdx, aKStride);
#pragma unroll
                for (uint32_t nb = 0; nb < NS; ++nb) {
                    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(regB[nb], bubPtr + nb * FMA_SUB_N,
                                                                                      bAddr);
                }
#pragma unroll
                for (uint32_t r = 0; r < MR; ++r) {
                    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                        regA[r], aRowBase + r * aMStride, aAddr);
                }
#pragma unroll
                for (uint32_t idx = 0; idx < MR * NS; ++idx) {
                    const uint32_t r = idx / NS;
                    const uint32_t nb = idx % NS;
                    AscendC::Reg::MulAddDst(regC[r][nb], regA[r], regB[nb], pregAll);
                }
            }
#pragma unroll
            for (uint32_t idx = 0; idx < MR * NS; ++idx) {
                const uint32_t r = idx / NS;
                const uint32_t nb = idx % NS;
                AscendC::Reg::StoreAlign<float>(cBase + r * bRowStride + nb * FMA_SUB_N, regC[r][nb], pregAll);
            }
        }
        // m方向尾部不足MR行, 逐行计算(所有k组统一UB装载路径)
        for (; mIdx < mNum; ++mIdx) {
            __ubuf__ float* cRow = cubPtr + mIdx * bRowStride;
#pragma unroll
            for (uint32_t nb = 0; nb < NS; ++nb) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(regC[0][nb], cRow + nb * FMA_SUB_N);
            }
            __ubuf__ float* aRow = aubPtr + static_cast<uint32_t>(mIdx) * aMStride;
            for (uint16_t kIdx = 0; kIdx < static_cast<uint16_t>(kBub); ++kIdx) {
                AscendC::Reg::AddrReg bAddr = AscendC::Reg::CreateAddrReg<float>(kIdx, bRowStride);
                AscendC::Reg::AddrReg aAddr = AscendC::Reg::CreateAddrReg<float>(kIdx, aKStride);
#pragma unroll
                for (uint32_t nb = 0; nb < NS; ++nb) {
                    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_NORM>(regB[nb], bubPtr + nb * FMA_SUB_N,
                                                                                      bAddr);
                }
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(regA[0], aRow, aAddr);
#pragma unroll
                for (uint32_t nb = 0; nb < NS; ++nb) {
                    AscendC::Reg::MulAddDst(regC[0][nb], regA[0], regB[nb], pregAll);
                }
            }
#pragma unroll
            for (uint32_t nb = 0; nb < NS; ++nb) {
                AscendC::Reg::StoreAlign<float>(cRow + nb * FMA_SUB_N, regC[0][nb], pregAll);
            }
        }
    }

    // 转置pass执行体(两步直转第二步): staging[N, K]tile(行距FMA_BSTRIDE) -> BT[K, N]tile(行距NB*64);
    // 每行k: NB条DataCopyGather列收集 + NB条DataCopy写回, t[nb][lane i] = staging[(nb*64+i)*BSTRIDE + k]
    // = B[k][nb*64+i], B32数据配U32索引(per-lane元素粒度); NB为编译期块数(128窄2块/宽路径256为4块,
    // Process按tiling的baseN运行时分发); N尾垃圾行复制为BT垃圾列(索引恒在staging界内, 越界读驻留旧值
    // 安全), K尾循环界与FmaVf同kBub不读; VF硬约束: 单层uint16_t循环/无分支, NB*2个RegTensor
    template <uint32_t NB>
    static __simd_vf__ inline void TransposeTileVfT(__ubuf__ float* dstPtr, __ubuf__ float* srcPtr,
                                                    __ubuf__ uint32_t* idxPtr, uint64_t kBub)
    {
        AscendC::Reg::RegTensor<uint32_t> regIdx[NB];
        AscendC::Reg::RegTensor<float> regT[NB];
        AscendC::Reg::MaskReg pregAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        // idx[nb][lane i] = (nb*64+i)*BSTRIDE, 表由InitTranspose一次性生成(项数=baseN_)
#pragma unroll
        for (uint32_t nb = 0; nb < NB; ++nb) {
            AscendC::Reg::DataCopy<uint32_t, AscendC::Reg::LoadDist::DIST_NORM>(regIdx[nb], idxPtr + nb * FMA_VL_FP32);
        }
        constexpr uint32_t rowPitch = NB * FMA_VL_FP32;
        for (uint16_t kIdx = 0; kIdx < static_cast<uint16_t>(kBub); ++kIdx) {
            // BT[k][n] = staging[n][k]: base逐行k滑动(列内元素粒度), idx跨行收集
#pragma unroll
            for (uint32_t nb = 0; nb < NB; ++nb) {
                AscendC::Reg::DataCopyGather(regT[nb], srcPtr + kIdx, regIdx[nb], pregAll);
            }
#pragma unroll
            for (uint32_t nb = 0; nb < NB; ++nb) {
                AscendC::Reg::DataCopy<float>(dstPtr + kIdx * rowPitch + nb * FMA_VL_FP32, regT[nb], pregAll);
            }
        }
    }

    // 寄存体内核调度(VF体内无类成员访问与运行时分支, 分支在编译期/标量流消除):
    //   transB=1: 先转置pass(TransposeTileVfT<NB>: staging列收集 -> BT[K,N], NB=baseN/64为1~4,
    //             按tiling的baseN运行时分发)再FMA读BT; staging单缓冲: pass入队后立即SetFlag(V_MTE2,0x2)
    //             释放(标志于pass排空时置位), 下一k组的CopyInB无需等FMA, MTE2/V流水重叠保持
    //             (槽位0x2独立于A乒乓的0x0/0x1); 连续VF_CALL硬件保序: pass写BT先于FMA读BT;
    //   计算体分发: MR由tiling下发(mr_), NS按本tile的tailN裁剪(nsTile, 多n-tile尾tile),
    //             查表分发到编译期模板实例; C自UB装载累积(多k组由调用方按周期分块flush有界)
    __aicore__ inline void Process(uint64_t ubOffsetA, uint64_t ubOffsetB, uint64_t ubOffsetC)
    {
        AscendC::LocalTensor<float> aubLocal = ubLocal_[ubOffsetA];
        AscendC::LocalTensor<float> bubLocal = ubLocal_[ubOffsetB];
        AscendC::LocalTensor<float> cubLocal = ubLocal_[ubOffsetC];
        __ubuf__ float* cubPtr = (__ubuf__ float*)cubLocal.GetPhyAddr();
        __ubuf__ float* bPtr = (__ubuf__ float*)bubLocal.GetPhyAddr();
        __ubuf__ float* aubPtr = (__ubuf__ float*)aubLocal.GetPhyAddr();
        // A tile行距: NN为baseK, TN为baseM([K, baseM]原生取向)
        const uint64_t aStride = TRANS_A ? baseM_ : baseK_;
        if constexpr (TRANS_B) {
            __ubuf__ float* btPtr = (__ubuf__ float*)ubLocal_[ubOffsetBT_].GetPhyAddr();
            __ubuf__ uint32_t* idxPtr = (__ubuf__ uint32_t*)ubLocal_.ReinterpretCast<uint32_t>()[ubOffsetIdx_]
                                            .GetPhyAddr();
            switch (baseN_ / FMA_VL_FP32) {
                case 1:
                    AscendC::VF_CALL<TransposeTileVfT<1>>(btPtr, bPtr, idxPtr, currentK_);
                    break;
                case 2:
                    AscendC::VF_CALL<TransposeTileVfT<2>>(btPtr, bPtr, idxPtr, currentK_);
                    break;
                case 3:
                    AscendC::VF_CALL<TransposeTileVfT<3>>(btPtr, bPtr, idxPtr, currentK_);
                    break;
                default:
                    AscendC::VF_CALL<TransposeTileVfT<4>>(btPtr, bPtr, idxPtr, currentK_);
                    break;
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0x2);
            bPtr = btPtr;
        }
        // NS按本tile的tailN裁剪(单tile时nsTile=ns_, 多n-tile尾tile被裁小); 尾块<64按整块发射
        const uint64_t nsTail = (tailN_ + FMA_SUB_N - 1) / FMA_SUB_N;
        const uint32_t nsTile = static_cast<uint32_t>(ns_ < nsTail ? ns_ : nsTail);
        const uint32_t bRowStride = static_cast<uint32_t>(baseN_);
        if (mr_ == FMA_MR) {
            // baseN=64档: (MR=8, NS=1)
            AscendC::VF_CALL<FmaMulKernel<TRANS_A, FMA_MR, 1>>(cubPtr, aubPtr, bPtr, tailM_, currentK_, aStride,
                                                               bRowStride);
        } else {
            switch (nsTile) {
                case 1:
                    AscendC::VF_CALL<FmaMulKernel<TRANS_A, FMA_MR_WIDE, 1>>(cubPtr, aubPtr, bPtr, tailM_, currentK_,
                                                                            aStride, bRowStride);
                    break;
                case 2:
                    AscendC::VF_CALL<FmaMulKernel<TRANS_A, FMA_MR_WIDE, 2>>(cubPtr, aubPtr, bPtr, tailM_, currentK_,
                                                                            aStride, bRowStride);
                    break;
                case 3:
                    AscendC::VF_CALL<FmaMulKernel<TRANS_A, FMA_MR_WIDE, 3>>(cubPtr, aubPtr, bPtr, tailM_, currentK_,
                                                                            aStride, bRowStride);
                    break;
                default:
                    AscendC::VF_CALL<FmaMulKernel<TRANS_A, FMA_MR_WIDE, 4>>(cubPtr, aubPtr, bPtr, tailM_, currentK_,
                                                                            aStride, bRowStride);
                    break;
            }
        }
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<float>& cGlobal,
                                   const AscendC::LocalTensor<float>& outubLocal)
    {
        uint16_t blockCount = static_cast<uint16_t>(tailM_);
        uint32_t blockLen = static_cast<uint32_t>(tailN_ * sizeof(float));
        if ((baseN_ - tailN_) % ALIGN_NUM == 0) {
            // 常规2D搬运: UB侧行距(srcStride)以32B为单位, 要求(baseN-tailN)被8整除(即tailN%8==0)
            uint32_t srcStride = static_cast<uint32_t>((baseN_ - tailN_) / ALIGN_NUM);
            uint32_t dstStride = static_cast<uint32_t>((n_ - tailN_) * sizeof(float));
            AscendC::DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
            AscendC::DataCopyPad(cGlobal, outubLocal, copyParams);
        } else {
            // n-tail tile的tailN非8对齐: srcStride无法按32B单位表达, 退化逐行1D搬运(仅尾tile走此路径)
            for (uint32_t row = 0; row < tailM_; ++row) {
                AscendC::DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
                AscendC::DataCopyPad(cGlobal[row * n_], outubLocal[row * baseN_], copyParams);
            }
        }
    }

    __aicore__ inline void operator()(const AscendC::GlobalTensor<float>& cGlobal,
                                      const AscendC::GlobalTensor<float>& aGlobal,
                                      const AscendC::GlobalTensor<float>& bGlobal,
                                      const AscendC::GlobalTensor<float>& biasGlobal)
    {
        // transB: B乒乓为单staging缓冲([N, BSTRIDE行距], pass即时释放复用), 无BPong无Out,
        //         分块累加: CPing兼任全局累加器, CPong为工作块, 周期flush(CPing += CPong),
        //         CopyOut直读CPing, BT/idx/bias由Init偏移公式定位;
        // 非transB: A/B乒乓 + C乒乓块 + Out(全局累加器), 末尾bias行(偏移公式与Init一致):
        //   NN窄: 2x16KB(A) + 2x32KB(B) + 2x32KB(C) + 32KB(Out) + 0.5KB(bias) = 192.5KB;
        //   NN宽: 2x8KB(A) + 2x64KB(B) + 2x32KB(C) + 32KB(Out) + 1KB(bias) = 241KB;
        //   NT/TT窄: 2x16KB(A) + 36KB(staging) + 2x32KB(C) + 32KB(BT) + 2KB = 165KB;
        //   NT/TT宽: 2x8KB(A) + 72KB(staging) + 2x32KB(C) + 64KB(BT) + 2KB = 218KB(248KB内)
        const uint64_t bTileElems = TRANS_B ? baseN_ * FMA_BSTRIDE : baseK_ * baseN_;
        uint64_t ubOffsetAPing = 0;
        uint64_t ubOffsetBPing = baseM_ * baseK_ + ubOffsetAPing;
        // for C accumulate ping
        uint64_t ubOffsetCPing = bTileElems + ubOffsetBPing;
        uint64_t ubOffsetAPong = baseM_ * baseN_ + ubOffsetCPing;
        uint64_t ubOffsetBPong = baseM_ * baseK_ + ubOffsetAPong;
        // for C accumulate pong: transB免BPong(单staging)
        uint64_t ubOffsetCPong = TRANS_B ? (baseM_ * baseK_ + ubOffsetAPong) : (bTileElems + ubOffsetBPong);
        [[maybe_unused]] uint64_t ubOffsetOut = baseM_ * baseN_ + ubOffsetCPong;
        uint64_t ubOffsetA[] = {ubOffsetAPing, ubOffsetAPong};
        uint64_t ubOffsetB[] = {ubOffsetBPing, ubOffsetBPong};
        uint64_t ubOffsetC[] = {ubOffsetCPing, ubOffsetCPong};

        // C乒乓预初始化(FmaMulKernel免运行时分支的代价与前提):
        //   CPing: 有bias时装入bias行并广播填充全baseM行(首k组C寄存器初值=bias), 无bias清零;
        //   CPong: 清零(次k组从CPong装载需0初值; 单k组免Add免Out直读CPing, CPong不读免清);
        // 预填充统一移入k循环j==0(置于CopyIn之后)再发射: 单k组(小k)路径上一tile的CopyOut直读CPing,
        // 入口预填充(V管写CPing)与该MTE3读同址跨管竞态, 会污染上一tile的搬出数据(小k精度问题根因),
        // 须等其读完再写; 此处仅先行载入bias行(GM->UB搬运暂存, 无跨管读者), MTE2完成后标量流等待,
        // 保证k循环内BiasFillVfT发射时bias行已就绪; 填充与首组CopyInA/B(MTE2)两管线并行,
        // V管程序序保证填充先于FmaMulKernel首组从CPing装载
        if (hasBias_) {
            CopyInBias(biasGlobal, ubLocal_[ubOffsetBias_]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0x0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0x0);
        }
        for (uint64_t j = 0; j < loopK_; ++j) {
            currentK_ = (j + 1 == loopK_) ? tailK_ : baseK_;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(j & 0x1);
            if constexpr (TRANS_A) {
                // A为[k, m]行优先, 第j轮k分块位于第j * baseK_行, m分块为列偏移(由调度器计算)
                CopyInA(aGlobal[j * baseK_ * m_], ubLocal_[ubOffsetA[j & 0x1]]);
            } else {
                CopyInA(aGlobal[j * baseK_], ubLocal_[ubOffsetA[j & 0x1]]);
            }
            if constexpr (TRANS_B) {
                // B为[n, k]行优先, 第j轮k分块为行内偏移, 原生搬入单staging后由Process转pass转成[K, N];
                // 等pass(上一k组)释放(槽位0x2, pass排空即置位), A仍走乒乓槽位j&1
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0x2);
                CopyInB(bGlobal[j * baseK_], ubLocal_[ubOffsetBPing]);
            } else {
                // B为(k, n)行优先, 第j轮k分块位于第j * baseK_行
                CopyInB(bGlobal[j * baseK_ * n_], ubLocal_[ubOffsetB[j & 0x1]]);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(j & 0x1);
            if constexpr (TRANS_B) {
                if (j == 0U) {
                    // 免Out: 上一tile CopyOut读CPing(单k组直读结果/多k组读全局累加器), 首个k组前等其读完;
                    // 置于CopyIn之后使MTE2搬入与上一tile CopyOut(MTE3)并行, 残余暴露仅为两者时差
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0x0);
                    if (hasBias_) {
                        __ubuf__ float* cpingPtr = (__ubuf__ float*)ubLocal_[ubOffsetCPing].GetPhyAddr();
                        __ubuf__ float* biasPtr = (__ubuf__ float*)ubLocal_[ubOffsetBias_].GetPhyAddr();
                        switch (baseN_ / FMA_SUB_N) {
                            case 1:
                                AscendC::VF_CALL<BiasFillVfT<1>>(cpingPtr, biasPtr, baseM_,
                                                                 static_cast<uint32_t>(baseN_));
                                break;
                            case 2:
                                AscendC::VF_CALL<BiasFillVfT<2>>(cpingPtr, biasPtr, baseM_,
                                                                 static_cast<uint32_t>(baseN_));
                                break;
                            case 3:
                                AscendC::VF_CALL<BiasFillVfT<3>>(cpingPtr, biasPtr, baseM_,
                                                                 static_cast<uint32_t>(baseN_));
                                break;
                            default:
                                AscendC::VF_CALL<BiasFillVfT<4>>(cpingPtr, biasPtr, baseM_,
                                                                 static_cast<uint32_t>(baseN_));
                                break;
                        }
                    } else {
                        AscendC::Duplicate<float>(ubLocal_[ubOffsetCPing], 0, static_cast<int32_t>(baseM_ * baseN_));
                    }
                    // CPong清零: 多k组为工作块0初值(各k组Process均累加其上, 周期flush后清零重累),
                    // 单k组免Out直读CPing, CPong不读无需清
                    if (loopK_ > 1U) {
                        AscendC::Duplicate<float>(ubLocal_[ubOffsetCPong], 0, static_cast<int32_t>(baseM_ * baseN_));
                    }
                }
            } else {
                // CPing/CPong预填充(与transB同款移入此处): 单k组免Add免Out路径CopyOut直读CPing,
                // 先等上一tile的CopyOut(MTE3)读完CPing再预填充; 多k组CopyOut读Out, CPing/CPong
                // 无MTE3读者免等待(Out的MTE3_V标志由k循环内首次flush的Wait配对消费)
                if (j == 0U) {
                    if (loopK_ == 1U) {
                        // 置于CopyIn之后使MTE2搬入与上一tile CopyOut(MTE3)并行, 残余暴露仅为两者时差
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0x0);
                    }
                    if (hasBias_) {
                        __ubuf__ float* cpingPtr = (__ubuf__ float*)ubLocal_[ubOffsetCPing].GetPhyAddr();
                        __ubuf__ float* biasPtr = (__ubuf__ float*)ubLocal_[ubOffsetBias_].GetPhyAddr();
                        switch (baseN_ / FMA_SUB_N) {
                            case 1:
                                AscendC::VF_CALL<BiasFillVfT<1>>(cpingPtr, biasPtr, baseM_,
                                                                 static_cast<uint32_t>(baseN_));
                                break;
                            case 2:
                                AscendC::VF_CALL<BiasFillVfT<2>>(cpingPtr, biasPtr, baseM_,
                                                                 static_cast<uint32_t>(baseN_));
                                break;
                            case 3:
                                AscendC::VF_CALL<BiasFillVfT<3>>(cpingPtr, biasPtr, baseM_,
                                                                 static_cast<uint32_t>(baseN_));
                                break;
                            default:
                                AscendC::VF_CALL<BiasFillVfT<4>>(cpingPtr, biasPtr, baseM_,
                                                                 static_cast<uint32_t>(baseN_));
                                break;
                        }
                    } else {
                        AscendC::Duplicate<float>(ubLocal_[ubOffsetCPing], 0, static_cast<int32_t>(baseM_ * baseN_));
                    }
                    // CPong清零: 多k组为次组块的0初值(周期flush后清零重累), 单k组免Out直读CPing,
                    // CPong不读无需清
                    if (loopK_ > 1U) {
                        AscendC::Duplicate<float>(ubLocal_[ubOffsetCPong], 0, static_cast<int32_t>(baseM_ * baseN_));
                    }
                }
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(j & 0x1);
            // 微指令完成(tailM, currentK) * (currentK, tailN) -> (tailM, tailN), 累加到本组C缓冲;
            // transB多k组: C目标恒为工作块CPong(CPing为全局累加器), 其余布局为C[j&1]乒乓块
            Process(ubOffsetA[j & 0x1], TRANS_B ? ubOffsetBPing : ubOffsetB[j & 0x1],
                    (TRANS_B && loopK_ > 1U) ? ubOffsetCPong : ubOffsetC[j & 0x1]);
            // A矩阵和B矩阵使用完毕，可以搬新的进来了
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(j & 0x1);
            // 分块累加flush(多k组, 周期flushGroups_组=单C缓冲singleCoreK个k元素, host按sqrt(K)选优下发):
            // 周期末块结果加到全局累加器后清零重累, 等价multi_mul的reduceSum后再累加——链内乘积只与
            // 从小长大的部分和相加, 只有块和碰全局累加器, 消除"K个乘积逐个加到不断长大的结果上"的
            // 大数吃小数; Add/Duplicate均为V管操作, V程序序保证先于本组Process写完/后于下组Process装载,
            // 免新增事件标志; flush点与加法顺序tiling期固定, 结果可复现
            if (loopK_ > 1U && ((j + 1U) % flushGroups_) == 0U) {
                if constexpr (TRANS_B) {
                    // transB免Out(UB预算218KB): CPing兼任全局累加器(j==0装bias/0初值, MTE3保护由j==0
                    // 的Wait完成), CPong为工作块, 原地累加dst==src0后工作块清零
                    AscendC::Add(ubLocal_[ubOffsetCPing], ubLocal_[ubOffsetCPing], ubLocal_[ubOffsetCPong],
                                 static_cast<int32_t>(baseM_ * baseN_));
                    AscendC::Duplicate<float>(ubLocal_[ubOffsetCPong], 0, static_cast<int32_t>(baseM_ * baseN_));
                } else {
                    if (j + 1U == flushGroups_) {
                        // 首次flush直接定义Out(=CPing+CPong)免清零; Out为上一tile CopyOut读对象,
                        // Wait置于此处保持上一tile CopyOut与本tile首个flush周期内的搬入/计算两管线并行
                        // (与上一tile CopyOut后的Set配对1:1, 替代原多k组末尾Add前的Wait)
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0x0);
                        AscendC::Add(ubLocal_[ubOffsetOut], ubLocal_[ubOffsetCPing], ubLocal_[ubOffsetCPong],
                                     static_cast<int32_t>(baseM_ * baseN_));
                    } else {
                        AscendC::Add(ubLocal_[ubOffsetOut], ubLocal_[ubOffsetOut], ubLocal_[ubOffsetCPing],
                                     static_cast<int32_t>(baseM_ * baseN_));
                        AscendC::Add(ubLocal_[ubOffsetOut], ubLocal_[ubOffsetOut], ubLocal_[ubOffsetCPong],
                                     static_cast<int32_t>(baseM_ * baseN_));
                    }
                    // 块清零: 下一周期从0重累(bias仅由j==0预填充进入一次)
                    AscendC::Duplicate<float>(ubLocal_[ubOffsetCPing], 0, static_cast<int32_t>(baseM_ * baseN_));
                    AscendC::Duplicate<float>(ubLocal_[ubOffsetCPong], 0, static_cast<int32_t>(baseM_ * baseN_));
                }
            }
        }
        if constexpr (TRANS_B) {
            if (loopK_ == 1U) {
                // 单k组: CPing即最终结果(含预填充的bias初值), 免Add直接CopyOut
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0x0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0x0);
                CopyOut(cGlobal, ubLocal_[ubOffsetCPing]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0x0);
            } else {
                // 分块累加: 各周期flush已将CPong工作块逐块累入CPing(全局累加器, 含bias初值);
                // 尾段(不足一个flush周期, 含K小于周期的整链情形)的CPong工作链补累后CopyOut直读CPing;
                // CPing的MTE3保护已由k循环j==0的Wait(MTE3_V)完成(上一tile CopyOut读CPing), 免二次等待
                if ((loopK_ % flushGroups_) != 0U) {
                    AscendC::Add(ubLocal_[ubOffsetCPing], ubLocal_[ubOffsetCPing], ubLocal_[ubOffsetCPong],
                                 static_cast<int32_t>(baseM_ * baseN_));
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0x0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0x0);
                CopyOut(cGlobal, ubLocal_[ubOffsetCPing]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0x0);
            }
        } else {
            if (loopK_ == 1U) {
                // 单k组: CPing即最终结果(含预填充的bias初值), 免Add与CPong
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0x0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0x0);
                CopyOut(cGlobal, ubLocal_[ubOffsetCPing]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0x0);
            } else {
                // 分块累加: 各周期flush已将块和逐次累入Out; 尾段(不足一个flush周期)补累——flush后
                // 两缓冲已清零, 尾段仅单缓冲有值时另一缓冲加0无害
                if (loopK_ < flushGroups_) {
                    // K不足单个flush周期(链长singleCoreK覆盖大半K): 无周期flush, 末尾一次性合并定义Out
                    // (=CPing+CPong); Out首次写需等上一tile CopyOut(MTE3)读完(与周期flush的Wait互斥,
                    // 每tile恰好消费一次MTE3_V标志)
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0x0);
                    AscendC::Add(ubLocal_[ubOffsetOut], ubLocal_[ubOffsetCPing], ubLocal_[ubOffsetCPong],
                                 static_cast<int32_t>(baseM_ * baseN_));
                } else if ((loopK_ % flushGroups_) != 0U) {
                    AscendC::Add(ubLocal_[ubOffsetOut], ubLocal_[ubOffsetOut], ubLocal_[ubOffsetCPing],
                                 static_cast<int32_t>(baseM_ * baseN_));
                    AscendC::Add(ubLocal_[ubOffsetOut], ubLocal_[ubOffsetOut], ubLocal_[ubOffsetCPong],
                                 static_cast<int32_t>(baseM_ * baseN_));
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0x0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0x0);
                CopyOut(cGlobal, ubLocal_[ubOffsetOut]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0x0);
            }
        }
    }

private:
    constexpr static uint16_t DIMENSION_M = 0;
    constexpr static uint16_t DIMENSION_N = 1;
    constexpr static uint16_t DIMENSION_K = 2;
    // regbase大小256B, fp32一次只能做64个数
    constexpr static uint32_t FMA_VL_FP32 = 64;
    // baseN=64档的MR(m方向寄存器分块行数): (MR=8, NS=1), 寄存器17个
    constexpr static uint32_t FMA_MR = 8;
    // baseN>=128档的MR: (MR=4, NS=baseN/64), 寄存器(NS=4时)24个, 低于regbase干净预算25;
    // NS为编译期模板参数(1~4, Process按tiling下发的ns与tile的tailN裁剪分发, VF体内免运行时分支)
    constexpr static uint32_t FMA_SUB_N = FMA_VL_FP32;
    constexpr static uint32_t FMA_MR_WIDE = 4;
    constexpr static uint64_t ALIGN_NUM = 8;
    // transB staging行距(元素数): 72 = bank错开(288B行距, 128B bank相位错开32B,
    // 转置pass gather冲突64路->16路); 约束: >= baseK且%8 == 0
    constexpr static uint32_t FMA_BSTRIDE = 72;
    // strided 2D DataCopyPad单批行数上限分块(staging按n方向分批搬入)
    constexpr static uint32_t FMA_COPY_ROWS = 32;
    AscendC::LocalTensor<float> ubLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
