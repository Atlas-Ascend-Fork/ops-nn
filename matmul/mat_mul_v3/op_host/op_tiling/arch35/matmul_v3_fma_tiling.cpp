/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_fma_tiling.cpp
 * \brief fp32高精度模式vector FMA矩阵乘tiling模板:
 *        C[i][j:j+128] += A[i][k] * B[k][j:j+128], regbase大小256B, fp32一次处理64个数, NR=2寄存器分块行宽128
 */

#include <cmath>

#include "matmul_v3_fma_tiling.h"
#include "matmul_tiling_registry.h"
#include "matmul_v3_tiling_strategy.h"
#include "matmul/common/op_host/math_util_nn.h"

namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;

MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3FmaTiling, DAV_3510, TO_FMA_MUL);
MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3FmaTiling, DAV_9201, TO_FMA_MUL);
MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3FmaTiling, DAV_9202, TO_FMA_MUL);

bool MatMulV3FmaTiling::IsCapable()
{
    // 未使能高精度模式
    if (!args_.isForceGrpAccForFp32) {
        return false;
    }
    if (MatMulV3TilingHelper::IsSelfNonContiguous(context_)) {
        return false;
    }
    if (args_.aFormat != ge::FORMAT_ND || args_.bFormat != ge::FORMAT_ND) {
        return false;
    }
    // 仅支持fp32 bias
    if (args_.hasBias && args_.biasType != ge::DT_FLOAT) {
        return false;
    }
    // A和B都是fp32
    if (args_.aType != ge::DT_FLOAT || args_.bType != ge::DT_FLOAT) {
        return false;
    }
    return true;
}

ge::graphStatus MatMulV3FmaTiling::DoOpTiling()
{
    uint64_t m = args_.mValue;
    uint64_t n = args_.nValue;
    // baseN档+满波tile行数联合选型(参数随档产出经tilingdata下发):
    // baseN初选: 单n-tile装下N的最紧档 min(256, ceil(n/64)*64), n>256时由后续按(轮数,波数)选档
    // 决定最终档(256/192/128), 满宽tile宽即baseN, 尾tile由kernel按tailN裁剪NS(tilingdata的ns为
    // 档位上限); baseN=64为(MR=8,NS=1), 其余MR=4/NS=baseN/64;
    // 满波tile行数取UB上限(元素粒度, 布局公式与block_mmad_fma_mul.h的operator()一致):
    //   NN/TN: 2*A(baseM*baseK) + 2*B(baseK*baseN) + 2*C + Out(3*baseM*baseN) + bias(baseN)
    //   NT/TT: 2*A + staging(baseN*BSTRIDE) + 2*C(2*baseM*baseN) + BT(baseK*baseN) + idx(baseN) + bias(baseN);
    // 8对齐为transA的A tile行距硬约束(CopyInA的dstStride以32B为单位), 下限8
    constexpr uint64_t bytesPerElem = 4UL;
    auto setBaseNTier = [&](uint64_t baseNTier) {
        runInfo_.baseN = baseNTier;
        runInfo_.mr = (baseNTier == FMA_SUB_UNIT) ? FMA_MR_NARROW : FMA_MR_WIDE;
        runInfo_.ns = baseNTier / FMA_SUB_UNIT;
        const uint64_t fixedElems = args_.isBTrans ?
                                        (baseNTier * FMA_BSTRIDE + FMA_BASE_K * baseNTier + 2UL * baseNTier) :
                                        (2UL * FMA_BASE_K * baseNTier + baseNTier);
        const uint64_t slopeElems = args_.isBTrans ? (2UL * FMA_BASE_K + 2UL * baseNTier) :
                                                     (2UL * FMA_BASE_K + 3UL * baseNTier);
        const uint64_t ubCap = (FMA_UB_BUDGET / bytesPerElem - fixedElems) / slopeElems;
        runInfo_.baseM = std::max(FMA_BASE_M_ALIGN, ubCap / FMA_BASE_M_ALIGN * FMA_BASE_M_ALIGN);
    };
    setBaseNTier(std::min(FMA_BASE_N_MAX, ops::CeilDiv(n, FMA_SUB_UNIT) * FMA_SUB_UNIT));
    // N超档位上限时按(轮数,波数)字典序选档: 轮数nCore=ceil(N/baseN)决定A的GM重读倍数——B驻L2时
    // 加轮无收益, 轮数锁定ceil(N/256); B超L2时轮数放开(降档增大baseM、mTile变小, 省B重读GM流量)。
    // 波数=ceil(mTile×nCore/aivNum)近似正比墙钟, 候选中取波数最少档, 波数不减时维持256(防欠填充
    // 回退), 同波数取大档(保小tile与低A重读); 64档A重读过重不入候选
    if (n > FMA_BASE_N_MAX) {
        const bool bResidentL2 = args_.kValue * n * bytesPerElem <= FMA_SPREAD_B_LIMIT;
        const uint64_t minRounds = ops::CeilDiv(n, FMA_BASE_N_MAX);
        auto tileWaves = [&](uint64_t baseNTier) {
            return ops::CeilDiv(ops::CeilDiv(m, runInfo_.baseM) * ops::CeilDiv(n, baseNTier), compileInfo_.aivNum);
        };
        setBaseNTier(FMA_BASE_N_MAX);
        uint64_t bestTier = FMA_BASE_N_MAX;
        uint64_t bestWaves = tileWaves(FMA_BASE_N_MAX);
        for (uint64_t tier = FMA_BASE_N_MAX - FMA_SUB_UNIT; tier >= FMA_SUB_UNIT * 2; tier -= FMA_SUB_UNIT) {
            if (bResidentL2 && ops::CeilDiv(n, tier) != minRounds) {
                continue;
            }
            setBaseNTier(tier);
            const uint64_t waves = tileWaves(tier);
            if (waves < bestWaves) {
                bestWaves = waves;
                bestTier = tier;
            }
        }
        setBaseNTier(bestTier);
    }
    // 大B低核占用降档至64: tile数不足核数时n向并发流数少, MTE2单核带宽受限, 降档使流数×nCore×4、
    // 每核B切片缩至1/4; 已满核占用或B驻L2时不降
    if (args_.kValue * n * bytesPerElem > FMA_SPREAD_B_LIMIT) {
        const uint64_t nCoreCur = ops::CeilDiv(n, runInfo_.baseN);
        const uint64_t mTilesCur = ops::CeilDiv(m, runInfo_.baseM);
        if (mTilesCur * nCoreCur < compileInfo_.aivNum) {
            setBaseNTier(FMA_SUB_UNIT);
        }
    }
    runInfo_.baseK = FMA_BASE_K;
    // 分块累加链长singleCoreK(单C缓冲相邻两次flush间累计的k元素数L): 误差界(K/L + L)*eps在L≈sqrt(K)最优,
    // K个乘积逐个累到大数的误差降为(K/L)*eps; L向上对齐baseK倍数(非transB的flush周期2L/baseK恒为偶数,
    // 乒乓双缓冲链对称), 上限K/下限baseK(K不足baseK的loopK==1单k组路径不经flush); 链长tiling期固定,
    // flush点与加法顺序固定, 结果可复现(同cube基础模板deterministic切K的设计动机)
    uint64_t sqrtK = static_cast<uint64_t>(std::sqrt(static_cast<double>(args_.kValue)));
    while (sqrtK * sqrtK > args_.kValue) {
        --sqrtK;
    }
    while ((sqrtK + 1UL) * (sqrtK + 1UL) <= args_.kValue) {
        ++sqrtK;
    }
    runInfo_.singleCoreK = std::min(std::max(ops::CeilAlign(sqrtK, FMA_BASE_K), FMA_BASE_K), args_.kValue);
    // 尾波均摊: F个满波(每波aivNum个baseM行tile)后, 剩余行按baseMTail切tile填满尾波所有核,
    // 消除尾波空转, 墙钟恒定≈F*baseM+baseMTail; baseM选择与墙钟无关(大baseM只减少tile数即B重读),
    // 故满波恒取UB上限
    const uint64_t aivNum = compileInfo_.aivNum;
    const uint64_t nCore = ops::CeilDiv(n, runInfo_.baseN);
    const uint64_t fullWaves = m / (aivNum * runInfo_.baseM);
    const uint64_t remainRows = m - fullWaves * aivNum * runInfo_.baseM;
    runInfo_.mTileFull = fullWaves * aivNum;
    if (remainRows == 0UL) {
        runInfo_.baseMTail = runInfo_.baseM; // 恰好整除: 无尾波
    } else {
        // 尾波单核m槽位(多n-tile时每核承担nCore个tile); baseMTail=ceil(R/槽位), buffer约束≤baseM;
        // 同n-切片碎片tile并发读被L2吸收, 均摊纯赚核占用
        const uint64_t mSlots = std::max(1UL, aivNum / nCore);
        uint64_t baseMTail = std::min(ops::CeilDiv(remainRows, mSlots), runInfo_.baseM);
        // 尾行慢路径: 非MR倍数的余行走逐行循环(单行耗时约为整行的47%), 对齐到MR倍数则tile变大,
        // 仅当省下的慢行时间超过对齐增量时对齐划算
        const uint64_t rem = baseMTail % runInfo_.mr;
        const uint64_t delta = (rem == 0UL) ? 0UL : (runInfo_.mr - rem);
        if (delta > 0UL && 47UL * rem > 100UL * delta && baseMTail + delta <= runInfo_.baseM) {
            baseMTail += delta;
        }
        runInfo_.baseMTail = baseMTail;
    }
    const uint64_t spreadTiles = remainRows == 0UL ? 0UL : ops::CeilDiv(remainRows, runInfo_.baseMTail);
    const uint64_t mTileNum = runInfo_.mTileFull + spreadTiles;
    runInfo_.usedCoreNum = std::min(mTileNum * nCore, aivNum);
    return ge::GRAPH_SUCCESS;
}

uint64_t MatMulV3FmaTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .SetBatchModel(MatMulV3BatchModel::BATCH_MODEL)
        .SetModel(MatMulV3Model::FMA_MUL)
        .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
        .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
        .GetTilingKey();
}

ge::graphStatus MatMulV3FmaTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<MatMulFmaTilingData>(tiling);
}

ge::graphStatus MatMulV3FmaTiling::GetTilingDataProcess(MatMulFmaTilingData& tilingData) const
{
    // 前7字段复用基类填充, 追加计算体模板参数与分块累加链长
    MatMulV3BaseTiling::GetTilingDataProcess(tilingData);
    tilingData.mr = runInfo_.mr;
    tilingData.ns = runInfo_.ns;
    tilingData.mTileFull = runInfo_.mTileFull;
    tilingData.baseMTail = runInfo_.baseMTail;
    tilingData.singleCoreK = static_cast<uint32_t>(runInfo_.singleCoreK);
    return ge::GRAPH_SUCCESS;
}
} // namespace matmul_v3_advanced
} // namespace optiling
