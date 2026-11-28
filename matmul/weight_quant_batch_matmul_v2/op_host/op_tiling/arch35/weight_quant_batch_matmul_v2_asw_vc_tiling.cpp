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
 * \file weight_quant_batch_matmul_v2_asw_vc_tiling.cpp
 * \brief ASW-VC 模板 tiling：独立实现 baseM/baseN/baseK 搜索、L1/L0 预算、尾块与 batch 参数计算。
 */

#include "weight_quant_batch_matmul_v2_asw_vc_tiling.h"

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "../weight_quant_batch_matmul_v2_tiling_key.h"
#include "matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "platform/platform_infos_def.h"
#include "../../../op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_key.h"

using namespace platform_ascendc;

namespace optiling {
namespace weight_quant_batch_matmul_v2 {
namespace {
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t L1_ALIGN_SIZE = 32;
constexpr uint64_t L2_ALIGN_SIZE = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_64 = 64;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint64_t BASIC_BLOCK_K_32_BYTE = 32;
constexpr uint64_t BASIC_BLOCK_K_128_BYTE = 128;
constexpr uint64_t BASIC_BLOCK_K_256_BYTE = 256;
constexpr uint64_t BASIC_BLOCK_K_512_BYTE = 512;
constexpr uint64_t L1_SINGLE_SIZE_LIMIT = 48 * 1024UL;
constexpr uint64_t MAX_STEP_K = 8;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_FOUR = 4;
constexpr uint32_t NUM_EIGHT = 8;
constexpr uint32_t DB_SIZE = 2;
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint64_t KB_SIZE = 1024;
constexpr uint64_t THOUSAND_NUM = 1000;
constexpr uint64_t INT4_ELEMS_PER_BYTE = 2;
constexpr int32_t ASW_VC_PRIORITY = 10; // 介于 ITERBATCH(9) 与 ASW(11) 之间
constexpr double BALANCE_RATE_EDGE = 0.9;
constexpr double CUBE_BOUND_RATIO = 0.85;
constexpr double EPSILON = 1e-9;
constexpr int64_t DEFAULT_CUBE_FREQ = 1650; // MHz，平台信息缺失时的兜底值
constexpr int64_t DEFAULT_AI_CORE_CNT = 32;
constexpr int64_t DEFAULT_DDR_RATE = 31;
constexpr int64_t DEFAULT_L2_RATE = 100;
constexpr int32_t FIX_SHIFT_VAL_LEN_A16W8 = 29;
constexpr int32_t FIX_SHIFT_VAL_LEN_A16W16 = 58;
constexpr int32_t FIX_SHIFT_VAL_MIN_F16XF16 = 34;
constexpr int32_t FIX_SHIFT_VAL_MAX_F16XF16 = 43;

int64_t StrToInt64WithDefault(const std::string& str, int64_t defaultValue)
{
    if (str.empty()) {
        return defaultValue;
    }
    char* endPtr = nullptr;
    errno = 0;
    long long parsed = std::strtoll(str.c_str(), &endPtr, 10);
    if (errno != 0 || endPtr == str.c_str() || *endPtr != '\0' || parsed < 0) {
        return defaultValue;
    }
    return static_cast<int64_t>(parsed);
}

double GetCoreFreq(fe::PlatFormInfos* platformInfo)
{
    std::string freqStr = std::to_string(DEFAULT_CUBE_FREQ);
    platformInfo->GetPlatformRes("AICoreSpec", "cube_freq", freqStr);
    return StrToInt64WithDefault(freqStr, DEFAULT_CUBE_FREQ) / static_cast<double>(THOUSAND_NUM);
}
} // namespace

// ASW-VC 仅承接 A16W4 pergroup：weight int4 FRACTAL_NZ + scale fp16，支持 A/B 转置
bool WeightQuantBatchMatmulV2TilingASWVC::IsCapable()
{
    if (matmulInfoPtr_->bDtype != ge::DT_INT4 || matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ ||
        matmulInfoPtr_->antiQuantType != QuantType::PER_GROUP ||
        matmulInfoPtr_->antiQuantScaleDtype != ge::DT_FLOAT16) {
        OP_LOGI(opName_, "the ASW-VC template only supports A16W4 pergroup with FRACTAL_NZ weight and fp16 scale");
        return false;
    }
    if (matmulInfoPtr_->cDtype == ge::DT_INT8 && matmulInfoPtr_->quantType != QuantType::PER_TENSOR &&
        matmulInfoPtr_->quantType != QuantType::PER_CHANNEL) {
        OP_LOGI(opName_, "the ASW-VC template requires per-tensor/per-channel quant scale for int8 output");
        return false;
    }
    const uint64_t weightC0 = matmulInfoPtr_->c0Size == 0UL ? CUBE_BLOCK : matmulInfoPtr_->c0Size;
    if (weightC0 != CUBE_BLOCK) {
        OP_LOGI(opName_, "the ASW-VC template does not support weight C0 %lu for transB %d", weightC0,
                static_cast<int32_t>(matmulInfoPtr_->transB));
        return false;
    }
    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWVC::InstantiateTilingData()
{
    if (vcTilingData_ == nullptr) {
        try {
            vcTilingData_ = std::make_unique<wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWVCTilingDataParams>();
        } catch (std::bad_alloc&) {
            OP_LOGE(opName_, "tiling data memory allocation failed");
            return ge::GRAPH_FAILED;
        }
    }
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < vcTilingDataSize_,
                    OP_LOGE(opName_, "insufficient tiling data capacity %zu, ASW-VC size %zu",
                            context_->GetRawTilingData()->GetCapacity(), vcTilingDataSize_),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWVC::DoOpTiling()
{
    OP_LOGD(opName_, "DoOpTiling of ASW-VC tiling strategy.");
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    OP_LOGE(opName_, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(matmulInfoPtr_->groupSize == 0UL, OP_LOGE(opName_, "groupSize must be greater than 0"),
                    return ge::GRAPH_FAILED);
    ResetBaseTiling();
    OP_TILING_CHECK(CalRebalanceBlock() == ge::GRAPH_FAILED, OP_LOGE(opName_, "failed to calculate rebalance block"),
                    return ge::GRAPH_FAILED);
    CalL1Tiling();
    CalL1BufferNum();
    MapRunInfoToTiling();
    SetTilingData();
    SetBatchParams();
    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2TilingASWVC::ResetBaseTiling()
{
    runInfo_.usedCoreNum = compileInfoPtr_->aicNum;
    runInfo_.baseM = BASIC_BLOCK_SIZE_256;
    runInfo_.baseN = BASIC_BLOCK_SIZE_256;
    runInfo_.baseK = GetShapeWithDataType(BASIC_BLOCK_K_128_BYTE, matmulInfoPtr_->aDtype);
    runInfo_.stepM = 1;
    runInfo_.stepN = 1;
    runInfo_.stepKa = 1;
    runInfo_.stepKb = 1;
    runInfo_.dbL0c = 1;
    runInfo_.l1BufferNum = DB_SIZE;
    runInfo_.mBlockCnt = 1;
    runInfo_.nBlockCnt = 1;
    runInfo_.cubeBoundParam = 0.0;
    runInfo_.cubeBoundEdge = 0.0;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWVC::CalRebalanceBlock()
{
    fe::PlatFormInfos* platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "platformInfo is null"), return ge::GRAPH_FAILED);
    const PlatformPerf platformPerf = GetPlatformPerf(platformInfo);
    const BaseBlockLimits limits = CalcBaseBlockLimits(platformPerf);
    SearchBaseBlock(limits);
    OptimizeBaseBlockForParallelism(limits);
    CalBaseK();
    OP_TILING_CHECK(ApplyUbLimit() == ge::GRAPH_FAILED, OP_LOGE(opName_, "failed to apply UB constraint"),
                    return ge::GRAPH_FAILED);
    UpdateBlockCounts();
    UpdateL0cBufferMode();
    return ge::GRAPH_SUCCESS;
}

WeightQuantBatchMatmulV2TilingASWVC::PlatformPerf WeightQuantBatchMatmulV2TilingASWVC::GetPlatformPerf(
    fe::PlatFormInfos* platformInfo) const
{
    std::string coreCntStr = std::to_string(DEFAULT_AI_CORE_CNT);
    std::string ddrRateStr = std::to_string(DEFAULT_DDR_RATE);
    std::string l2RateStr = std::to_string(DEFAULT_L2_RATE);
    platformInfo->GetPlatformRes("SoCInfo", "ai_core_cnt", coreCntStr);
    platformInfo->GetPlatformRes("AICoreMemoryRates", "ddr_rate", ddrRateStr);
    platformInfo->GetPlatformRes("AICoreMemoryRates", "l2_rate", l2RateStr);
    const double coreFreq = GetCoreFreq(platformInfo);
    const int64_t coreCnt = StrToInt64WithDefault(coreCntStr, DEFAULT_AI_CORE_CNT);
    return {coreFreq, coreFreq * coreCnt * StrToInt64WithDefault(ddrRateStr, DEFAULT_DDR_RATE) / KB_SIZE,
            coreFreq * coreCnt * StrToInt64WithDefault(l2RateStr, DEFAULT_L2_RATE) / KB_SIZE};
}

WeightQuantBatchMatmulV2TilingASWVC::BaseBlockLimits WeightQuantBatchMatmulV2TilingASWVC::CalcBaseBlockLimits(
    const PlatformPerf& platformPerf)
{
    const uint64_t mSize = matmulInfoPtr_->mSize;
    const uint64_t nSize = matmulInfoPtr_->nSize;
    const uint64_t kSize = matmulInfoPtr_->kSize;
    const uint64_t aDtypeSize = static_cast<uint64_t>(ge::GetSizeByDataType(matmulInfoPtr_->aDtype));
    const uint64_t baseMNBufferLimit = compileInfoPtr_->l0cSize;
    const double cubeMemRatio = (static_cast<double>(mSize) + nSize) / (static_cast<double>(mSize) * nSize);
    const double computePower = platformPerf.coreFreq * NUM_EIGHT * compileInfoPtr_->aicNum;
    const double inputSize = static_cast<double>(matmulInfoPtr_->batchY) *
                             static_cast<double>(GetSizeWithDataType(mSize * kSize, matmulInfoPtr_->aDtype) +
                                                 GetSizeWithDataType(nSize * kSize, matmulInfoPtr_->aDtype));
    const double l2CacheUsage = compileInfoPtr_->l2Size == 0 ? 1.0 : std::max(inputSize / compileInfoPtr_->l2Size, 1.0);
    runInfo_.cubeBoundEdge = (platformPerf.l2BW / computePower) +
                             l2CacheUsage * (1 - platformPerf.l2BW / platformPerf.hbmBW) * cubeMemRatio -
                             (1 + platformPerf.l2BW / platformPerf.hbmBW) / kSize;

    const uint64_t baseMBest = std::min(ops::CeilAlign(mSize, static_cast<uint64_t>(CUBE_BLOCK)),
                                        static_cast<uint64_t>(BASIC_BLOCK_SIZE_256));
    const uint64_t baseNBest = std::max(
        static_cast<uint64_t>(CUBE_BLOCK),
        std::min(ops::CeilAlign(nSize, static_cast<uint64_t>(CUBE_BLOCK)),
                 ops::FloorAlign(baseMNBufferLimit / DATA_SIZE_L0C / baseMBest, static_cast<uint64_t>(CUBE_BLOCK))));
    const double cubeBoundParamBest = (1.0 / baseMBest) + (1.0 / baseNBest);
    const bool isMemoryBound = cubeBoundParamBest > runInfo_.cubeBoundEdge;
    const uint64_t innerAlignUnit = isMemoryBound ? BASIC_BLOCK_SIZE_128 : BASIC_BLOCK_SIZE_64;
    const double fixpBoundEdge = (static_cast<double>(mSize) * nSize * platformPerf.hbmBW) /
                                 ((mSize + nSize) * platformPerf.l2BW);
    const uint64_t baseNAlignUnit = static_cast<double>(kSize) < fixpBoundEdge ?
                                        GetShapeWithDataType(BASIC_BLOCK_K_256_BYTE, matmulInfoPtr_->bDtype) :
                                        (matmulInfoPtr_->transB ?
                                             CUBE_BLOCK :
                                             GetShapeWithDataType(innerAlignUnit, matmulInfoPtr_->bDtype));
    const uint64_t baseMAlignUnit = matmulInfoPtr_->transA ? innerAlignUnit / aDtypeSize : CUBE_BLOCK;
    return {GetMaxBaseWithLimit(baseMNBufferLimit, baseMAlignUnit, false, isMemoryBound),
            GetMaxBaseWithLimit(baseMNBufferLimit, baseNAlignUnit, true, isMemoryBound), baseMAlignUnit, baseNAlignUnit,
            baseMNBufferLimit};
}

void WeightQuantBatchMatmulV2TilingASWVC::SearchBaseBlock(const BaseBlockLimits& limits)
{
    runInfo_.baseM = std::max(static_cast<uint64_t>(CUBE_BLOCK),
                              std::min(limits.maxBaseM, static_cast<uint64_t>(BASIC_BLOCK_SIZE_256)));
    runInfo_.baseN = std::max(
        static_cast<uint64_t>(CUBE_BLOCK),
        std::min(limits.maxBaseN,
                 ops::FloorAlign(limits.baseMNBufferLimit / DATA_SIZE_L0C / runInfo_.baseM, limits.baseNAlignUnit)));
    runInfo_.cubeBoundParam = (1.0 / runInfo_.baseM) + (1.0 / runInfo_.baseN);
    runInfo_.cubeBoundEdge *= CUBE_BOUND_RATIO;
    double balanceRate = GetBalanceRateWithTail(runInfo_.baseM, runInfo_.baseN);

    for (uint64_t curBaseM = limits.maxBaseM; curBaseM >= 1 && curBaseM <= limits.maxBaseM;
         curBaseM -= limits.baseMAlignUnit) {
        const uint64_t curMaxBaseN = std::min(
            limits.maxBaseN,
            ops::FloorAlign(limits.baseMNBufferLimit / DATA_SIZE_L0C / curBaseM, limits.baseNAlignUnit));
        for (uint64_t curBaseN = curMaxBaseN; curBaseN >= 1 && curBaseN <= curMaxBaseN;
             curBaseN -= limits.baseNAlignUnit) {
            const double curCubeBoundParam = (1.0 / curBaseM) + (1.0 / curBaseN);
            const double curBalanceRate = GetBalanceRateWithTail(curBaseM, curBaseN);
            const bool skipCond = balanceRate >= BALANCE_RATE_EDGE && curCubeBoundParam > runInfo_.cubeBoundParam &&
                                  curCubeBoundParam > runInfo_.cubeBoundEdge && runInfo_.cubeBoundEdge > 0;
            if (skipCond) {
                continue;
            }
            const bool cubeBoundCond = curCubeBoundParam <= runInfo_.cubeBoundEdge && curBalanceRate > balanceRate;
            const bool balanceCond = ((curCubeBoundParam / curBalanceRate) < (runInfo_.cubeBoundParam / balanceRate)) ||
                                     ((std::abs(curCubeBoundParam / curBalanceRate -
                                                runInfo_.cubeBoundParam / balanceRate) < EPSILON) &&
                                      curBalanceRate > balanceRate);
            if (cubeBoundCond || balanceCond) {
                if (cubeBoundCond) {
                    runInfo_.cubeBoundEdge = curCubeBoundParam;
                }
                runInfo_.baseM = curBaseM;
                runInfo_.baseN = curBaseN;
                runInfo_.cubeBoundParam = curCubeBoundParam;
                balanceRate = curBalanceRate;
            }
        }
    }
    runInfo_.baseM = std::min(ops::CeilAlign(matmulInfoPtr_->mSize, static_cast<uint64_t>(CUBE_BLOCK)), runInfo_.baseM);
    runInfo_.baseN = std::min(ops::CeilAlign(matmulInfoPtr_->nSize, static_cast<uint64_t>(CUBE_BLOCK)), runInfo_.baseN);
}

void WeightQuantBatchMatmulV2TilingASWVC::OptimizeBaseBlockForParallelism(const BaseBlockLimits& limits)
{
    const uint64_t mSize = matmulInfoPtr_->mSize;
    const uint64_t nSize = matmulInfoPtr_->nSize;
    const uint64_t batchNum = std::max(matmulInfoPtr_->batchY, static_cast<uint64_t>(1));
    const uint64_t alignMFull = ops::CeilAlign(mSize, static_cast<uint64_t>(CUBE_BLOCK));
    if (alignMFull > limits.maxBaseM) {
        return;
    }

    const uint64_t coreCnt = std::max(static_cast<uint64_t>(compileInfoPtr_->aicNum), static_cast<uint64_t>(1));
    const uint64_t batchParallel = std::min(batchNum, coreCnt);
    const uint64_t nBlockUpper = ops::CeilDiv(nSize, static_cast<uint64_t>(CUBE_BLOCK));
    const uint64_t wantNBlocks = std::min(std::max(coreCnt / batchParallel, static_cast<uint64_t>(1)), nBlockUpper);
    uint64_t candN = ops::CeilAlign(ops::CeilDiv(nSize, wantNBlocks), static_cast<uint64_t>(CUBE_BLOCK));
    const uint64_t nFloor = std::min(nSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_64));
    candN = std::max(candN,
                     matmulInfoPtr_->transB ? ops::CeilAlign(nFloor, static_cast<uint64_t>(CUBE_BLOCK)) : nFloor);
    const uint64_t candNMax = std::min(
        limits.maxBaseN,
        ops::FloorAlign(limits.baseMNBufferLimit / DATA_SIZE_L0C / alignMFull, static_cast<uint64_t>(CUBE_BLOCK)));
    candN = std::min(candN, candNMax);

    const uint64_t oldBlocks = batchNum * ops::CeilDiv(mSize, runInfo_.baseM) * ops::CeilDiv(nSize, runInfo_.baseN);
    const uint64_t newBlocks = batchNum * ops::CeilDiv(nSize, candN);
    if (candN >= CUBE_BLOCK && newBlocks >= oldBlocks) {
        runInfo_.baseM = alignMFull;
        runInfo_.baseN = candN;
    }
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWVC::ApplyUbLimit()
{
    while (runInfo_.baseN > CUBE_BLOCK && CalcUbSize(runInfo_.baseN, runInfo_.baseK) > compileInfoPtr_->ubSize) {
        runInfo_.baseN -= CUBE_BLOCK;
    }
    const uint64_t minUbSize = CalcUbSize(runInfo_.baseN, runInfo_.baseK);
    OP_TILING_CHECK(minUbSize > compileInfoPtr_->ubSize,
                    OP_LOGE(opName_, "minimum UB size %lu exceeds available UB size %lu, baseN %lu, baseK %lu",
                            minUbSize, compileInfoPtr_->ubSize, runInfo_.baseN, runInfo_.baseK),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2TilingASWVC::UpdateBlockCounts()
{
    runInfo_.mBlockCnt = ops::CeilDiv(matmulInfoPtr_->mSize, runInfo_.baseM);
    runInfo_.nBlockCnt = ops::CeilDiv(matmulInfoPtr_->nSize, runInfo_.baseN);
    runInfo_.usedCoreNum = std::min(matmulInfoPtr_->batchY * runInfo_.mBlockCnt * runInfo_.nBlockCnt,
                                    runInfo_.usedCoreNum);
}

void WeightQuantBatchMatmulV2TilingASWVC::UpdateL0cBufferMode()
{
    runInfo_.dbL0c = (runInfo_.baseM * runInfo_.baseN * DATA_SIZE_L0C * DB_SIZE <= compileInfoPtr_->l0cSize &&
                      CheckAntiQuantScale(runInfo_.baseN, DB_SIZE)) ?
                         DB_SIZE :
                         1;
}

uint64_t WeightQuantBatchMatmulV2TilingASWVC::GetMaxBaseWithLimit(uint64_t baseMNBufferLimit, uint64_t baseAlignUnit,
                                                                  bool isRightMatrix, bool isMemoryBound) const
{
    ge::DataType dtype = matmulInfoPtr_->aDtype;
    uint64_t l0Size = isRightMatrix ? compileInfoPtr_->l0bSize : compileInfoPtr_->l0aSize;
    uint64_t shapeValue = isRightMatrix ? matmulInfoPtr_->nSize : matmulInfoPtr_->mSize;
    uint64_t kAlignValue = ops::CeilAlign(matmulInfoPtr_->kSize, static_cast<uint64_t>(CUBE_BLOCK));
    uint64_t kLimitValue = isMemoryBound ? CUBE_BLOCK : GetShapeWithDataType(BASIC_BLOCK_K_128_BYTE, dtype);
    uint64_t minKL0 = GetSizeWithDataType(std::min(kLimitValue, kAlignValue), dtype);
    OP_TILING_CHECK(minKL0 == 0, OP_LOGE(opName_, "minKL0 is 0"), return 0);
    uint64_t maxBaseMNWithBuffer = baseMNBufferLimit / DATA_SIZE_L0C / CUBE_BLOCK;
    uint64_t maxBaseBlock = std::min(l0Size / DB_SIZE / minKL0, maxBaseMNWithBuffer);
    const uint64_t kAlignUnit = (!matmulInfoPtr_->transA || matmulInfoPtr_->transB) ?
                                    GetShapeWithDataType(BASIC_BLOCK_K_512_BYTE, dtype) :
                                    CUBE_BLOCK;
    uint64_t maxBaseMNWithKInner = compileInfoPtr_->l1Size /
                                   (NUM_TWO * DB_SIZE * GetSizeWithDataType(std::min(kAlignUnit, kAlignValue), dtype));
    maxBaseBlock = std::min(maxBaseBlock, maxBaseMNWithKInner);
    maxBaseBlock = std::min(ops::CeilAlign(shapeValue, baseAlignUnit), ops::FloorAlign(maxBaseBlock, baseAlignUnit));
    if (shapeValue < baseAlignUnit) {
        maxBaseBlock = std::min(maxBaseBlock, ops::CeilAlign(shapeValue, static_cast<uint64_t>(CUBE_BLOCK)));
    }
    return maxBaseBlock;
}

double WeightQuantBatchMatmulV2TilingASWVC::GetBalanceRateWithTail(uint64_t baseM, uint64_t baseN) const
{
    uint64_t batchNum = matmulInfoPtr_->batchY;
    uint64_t totalRound = batchNum * ops::CeilDiv(matmulInfoPtr_->mSize, baseM) *
                          ops::CeilDiv(matmulInfoPtr_->nSize, baseN);
    uint64_t mainRound = ops::CeilDiv(totalRound, runInfo_.usedCoreNum) - 1;
    if (matmulInfoPtr_->mSize <= CUBE_BLOCK) {
        baseM = matmulInfoPtr_->mSize;
    }
    if (matmulInfoPtr_->nSize <= CUBE_BLOCK) {
        baseN = matmulInfoPtr_->nSize;
    }
    return (static_cast<double>(batchNum) * matmulInfoPtr_->mSize * matmulInfoPtr_->nSize / runInfo_.usedCoreNum) /
           ((mainRound + 1) * baseM * baseN);
}

void WeightQuantBatchMatmulV2TilingASWVC::CalBaseK()
{
    uint64_t aDtypeSize = static_cast<uint64_t>(ge::GetSizeByDataType(matmulInfoPtr_->aDtype));
    uint64_t kC0B = GetShapeWithDataType(BASIC_BLOCK_K_32_BYTE, matmulInfoPtr_->bDtype);
    uint64_t kValueAlign = ops::CeilAlign(matmulInfoPtr_->kSize, kC0B);
    auto getMaxBaseK = [this]() {
        uint64_t baseMax = std::max(runInfo_.baseM, runInfo_.baseN);
        uint64_t maxKL0A = GetShapeWithDataType(compileInfoPtr_->l0aSize / DB_SIZE / baseMax, matmulInfoPtr_->aDtype);
        uint64_t maxKL0B = GetShapeWithDataType(compileInfoPtr_->l0bSize / DB_SIZE / baseMax, matmulInfoPtr_->aDtype);
        return std::min(maxKL0A, maxKL0B);
    };
    uint64_t maxBaseK = getMaxBaseK();
    while (maxBaseK < kC0B && std::max(runInfo_.baseM, runInfo_.baseN) > static_cast<uint64_t>(CUBE_BLOCK)) {
        if (runInfo_.baseM >= runInfo_.baseN) {
            runInfo_.baseM = std::max(ops::FloorAlign(runInfo_.baseM / NUM_TWO, static_cast<uint64_t>(CUBE_BLOCK)),
                                      static_cast<uint64_t>(CUBE_BLOCK));
        } else {
            runInfo_.baseN = std::max(ops::FloorAlign(runInfo_.baseN / NUM_TWO, static_cast<uint64_t>(CUBE_BLOCK)),
                                      static_cast<uint64_t>(CUBE_BLOCK));
        }
        maxBaseK = getMaxBaseK();
    }
    if (maxBaseK < kC0B) {
        runInfo_.baseK = kC0B;
        return;
    }
    if (kValueAlign <= maxBaseK) {
        runInfo_.baseK = kValueAlign;
        return;
    }
    if (matmulInfoPtr_->transA && !matmulInfoPtr_->transB) {
        runInfo_.baseK = ops::FloorAlign(maxBaseK, kC0B);
        return;
    }
    if (maxBaseK * aDtypeSize >= BASIC_BLOCK_K_256_BYTE) {
        runInfo_.baseK = ops::FloorAlign(maxBaseK,
                                         GetShapeWithDataType(BASIC_BLOCK_K_256_BYTE, matmulInfoPtr_->aDtype));
        return;
    }
    constexpr std::array<uint64_t, 4> baseKCandidate = {128, 64, 32, 16};
    for (uint64_t baseK : baseKCandidate) {
        if (baseK >= kC0B && maxBaseK >= baseK) {
            runInfo_.baseK = baseK;
            return;
        }
    }
}

void WeightQuantBatchMatmulV2TilingASWVC::CalL1Tiling()
{
    const uint64_t totalL1Size = compileInfoPtr_->l1Size - GetFixedL1Size();
    const uint64_t maxStepK = std::min(ops::CeilDiv(matmulInfoPtr_->kSize, runInfo_.baseK), MAX_STEP_K);
    const bool isKInner = !matmulInfoPtr_->transA || matmulInfoPtr_->transB;
    const uint64_t kAlignUnit = isKInner ? GetShapeWithDataType(BASIC_BLOCK_K_512_BYTE, matmulInfoPtr_->aDtype) :
                                           CUBE_BLOCK;
    uint64_t resKL1 = 0;
    uint64_t singleMteSize = 0;
    for (uint64_t stepK = 1; stepK <= maxStepK; stepK++) {
        const uint64_t curKL1 = runInfo_.baseK * stepK;
        const auto [aL1Size, bL1Size] = CalcL1TensorSizes(curKL1, curKL1);
        if (CalcUbSize(runInfo_.baseN, curKL1) > compileInfoPtr_->ubSize ||
            (aL1Size + bL1Size) * DB_SIZE > totalL1Size ||
            std::max(aL1Size, bL1Size) * DB_SIZE * NUM_TWO > compileInfoPtr_->l1Size) {
            break;
        }
        const bool condNoRes = resKL1 == 0;
        const bool condKAlign256B = curKL1 % GetShapeWithDataType(BASIC_BLOCK_K_256_BYTE, matmulInfoPtr_->aDtype) == 0;
        const bool condKAlign = resKL1 % kAlignUnit != 0 &&
                                (condKAlign256B || (!condKAlign256B && singleMteSize < L1_SINGLE_SIZE_LIMIT));
        const bool condMteSize = resKL1 % kAlignUnit == 0 && curKL1 % kAlignUnit == 0 &&
                                 singleMteSize < L1_SINGLE_SIZE_LIMIT;
        if (condNoRes || condKAlign || condMteSize) {
            resKL1 = curKL1;
            singleMteSize = std::max(aL1Size, bL1Size);
        }
    }
    runInfo_.stepKa = resKL1 / runInfo_.baseK;
    runInfo_.stepKb = runInfo_.stepKa;
}

std::pair<uint64_t, uint64_t> WeightQuantBatchMatmulV2TilingASWVC::CalcL1TensorSizes(uint64_t kL1A, uint64_t kL1B) const
{
    const uint64_t c0B = GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->aDtype);
    const uint64_t bL1Elems = matmulInfoPtr_->transB ? runInfo_.baseN * ops::CeilAlign(kL1B, c0B) :
                                                       ops::CeilAlign(runInfo_.baseN, c0B) * kL1B;
    return {GetSizeWithDataType(runInfo_.baseM * kL1A, matmulInfoPtr_->aDtype),
            GetSizeWithDataType(bL1Elems, matmulInfoPtr_->aDtype)};
}

uint64_t WeightQuantBatchMatmulV2TilingASWVC::GetFixedL1Size() const
{
    const uint64_t biasSize = matmulInfoPtr_->hasBias ?
                                  runInfo_.baseN * ge::GetSizeByDataType(matmulInfoPtr_->biasDtype) * DB_SIZE :
                                  0UL;
    const uint64_t quantScaleSize = matmulInfoPtr_->cDtype == ge::DT_INT8 &&
                                            matmulInfoPtr_->quantType == QuantType::PER_CHANNEL ?
                                        runInfo_.baseN * sizeof(uint64_t) :
                                        0UL;
    return biasSize + quantScaleSize;
}

void WeightQuantBatchMatmulV2TilingASWVC::CalL1BufferNum()
{
    const uint64_t kL1A = runInfo_.baseK * runInfo_.stepKa;
    const uint64_t kL1B = runInfo_.baseK * runInfo_.stepKb;
    const auto [aL1Size, bL1Size] = CalcL1TensorSizes(kL1A, kL1B);
    const uint64_t abL1TensorSize = aL1Size + bL1Size;
    runInfo_.l1BufferNum = abL1TensorSize * NUM_FOUR + GetFixedL1Size() <= compileInfoPtr_->l1Size ? NUM_FOUR : DB_SIZE;
}

uint64_t WeightQuantBatchMatmulV2TilingASWVC::CalcUbSize(uint64_t baseN, uint64_t kL1) const
{
    uint64_t nMax = ops::CeilAlign(baseN, CUBE_BLOCK);
    // chunk 起点可能不在 group 边界，+1 兜底最多覆盖的 group 行数
    uint64_t scaleSingleElems = (ops::CeilDiv(kL1, matmulInfoPtr_->groupSize) + 1UL) * nMax;
    uint64_t stagingBytes = DB_SIZE * GetSizeWithDataType(nMax * kL1, matmulInfoPtr_->aDtype);
    uint64_t lowBitBytes = DB_SIZE * (nMax * kL1 / INT4_ELEMS_PER_BYTE);
    uint64_t scaleBytes = DB_SIZE * GetSizeWithDataType(scaleSingleElems, matmulInfoPtr_->antiQuantScaleDtype);
    uint64_t offsetBytes = matmulInfoPtr_->hasAntiQuantOffset ? scaleBytes : 0UL;
    return stagingBytes + lowBitBytes + scaleBytes + offsetBytes;
}

void WeightQuantBatchMatmulV2TilingASWVC::MapRunInfoToTiling()
{
    basicTiling_.usedCoreNum = static_cast<uint32_t>(runInfo_.usedCoreNum);
    basicTiling_.baseM = static_cast<uint32_t>(runInfo_.baseM);
    basicTiling_.baseN = static_cast<uint32_t>(runInfo_.baseN);
    basicTiling_.baseK = static_cast<uint32_t>(runInfo_.baseK);
    basicTiling_.stepKa = static_cast<uint32_t>(runInfo_.stepKa);
    basicTiling_.stepKb = static_cast<uint32_t>(runInfo_.stepKb);
    basicTiling_.dbL0c = static_cast<uint32_t>(runInfo_.dbL0c);

    adaptiveWin_.mBlockCnt = runInfo_.mBlockCnt;
    adaptiveWin_.nBlockCnt = runInfo_.nBlockCnt;
}

uint64_t WeightQuantBatchMatmulV2TilingASWVC::GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    bool is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_INT4);
    if (is4BitInput) {
        return shapeSize + shapeSize;
    }
    return shapeSize / static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
}

uint64_t WeightQuantBatchMatmulV2TilingASWVC::GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    bool is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_INT4);
    if (is4BitInput) {
        OP_TILING_CHECK(
            shapeSize % 2 != 0,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "elements", std::to_string(shapeSize).c_str(),
                                                      "When the dtype of the input is FLOAT4 or INT4, the shape size "
                                                      "of the input must be a positive even number"),
            return 0);
        return shapeSize / 2;
    }
    return shapeSize * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
}

bool WeightQuantBatchMatmulV2TilingASWVC::CheckAntiQuantScale(uint64_t baseN, uint64_t dbL0c) const
{
    uint64_t maxScaleBaseN = dbL0c != DB_SIZE ? BASIC_BLOCK_SIZE_512 : BASIC_BLOCK_SIZE_256;
    return baseN <= maxScaleBaseN;
}

wqbmmv2_tiling::L2CacheMode WeightQuantBatchMatmulV2TilingASWVC::SetDisableL2cache(uint32_t mL1, uint32_t kaL1,
                                                                                   uint32_t kbL1, uint32_t nL1) const
{
    if (!enableUncache_) {
        OP_LOGD(opName_, "enable_uncache is not set to 1, L2 uncache disabled.");
        return wqbmmv2_tiling::L2CacheMode::L2_CACHE_DEFAULT;
    }
    uint64_t totalSize = matmulInfoPtr_->mSize * matmulInfoPtr_->nSize * ge::GetSizeByDataType(matmulInfoPtr_->cDtype) +
                         matmulInfoPtr_->mSize * matmulInfoPtr_->kSize * ge::GetSizeByDataType(matmulInfoPtr_->aDtype) +
                         matmulInfoPtr_->kSize * matmulInfoPtr_->nSize * ge::GetSizeByDataType(matmulInfoPtr_->bDtype);
    wqbmmv2_tiling::L2CacheMode cacheMode = wqbmmv2_tiling::L2CacheMode::L2_CACHE_DEFAULT;
    const uint64_t innerB = matmulInfoPtr_->transB ? matmulInfoPtr_->kSize : matmulInfoPtr_->nSize;
    const bool flagB = matmulInfoPtr_->transB ?
                           (GetSizeWithDataType(static_cast<uint64_t>(kbL1), matmulInfoPtr_->bDtype) % L2_ALIGN_SIZE ==
                            0UL) :
                           (GetSizeWithDataType(static_cast<uint64_t>(nL1), matmulInfoPtr_->bDtype) % L2_ALIGN_SIZE ==
                            0UL);
    bool rightNotL2Cache = basicTiling_.baseM >= matmulInfoPtr_->mSize && adaptiveWin_.mBlockCnt <= 1UL &&
                           GetSizeWithDataType(innerB, matmulInfoPtr_->bDtype) % L2_ALIGN_SIZE == 0UL && flagB;
    if (totalSize < compileInfoPtr_->l2Size) {
        if (rightNotL2Cache) {
            cacheMode = wqbmmv2_tiling::L2CacheMode::B_L2_CACHE_DISABLE;
        }
        OP_LOGD(opName_, "L2 cache params: totalSize:%lu, flagB:%d, rightNotL2Cache:%d, cacheMode:%d.", totalSize,
                static_cast<int32_t>(flagB), static_cast<int32_t>(rightNotL2Cache), static_cast<int32_t>(cacheMode));
        return cacheMode;
    }
    const uint64_t innerA = matmulInfoPtr_->transA ? matmulInfoPtr_->mSize : matmulInfoPtr_->kSize;
    const bool flagA = matmulInfoPtr_->transA ?
                           (GetSizeWithDataType(static_cast<uint64_t>(mL1), matmulInfoPtr_->aDtype) % L2_ALIGN_SIZE ==
                            0UL) :
                           (GetSizeWithDataType(static_cast<uint64_t>(kaL1), matmulInfoPtr_->aDtype) % L2_ALIGN_SIZE ==
                            0UL);
    bool leftNotL2Cache = basicTiling_.baseN >= matmulInfoPtr_->nSize && adaptiveWin_.nBlockCnt <= 1UL &&
                          GetSizeWithDataType(innerA, matmulInfoPtr_->aDtype) % L2_ALIGN_SIZE == 0UL && flagA;
    if (leftNotL2Cache && rightNotL2Cache) {
        cacheMode = wqbmmv2_tiling::L2CacheMode::ALL_L2_CACHE_DISABLE;
    } else if (leftNotL2Cache) {
        cacheMode = wqbmmv2_tiling::L2CacheMode::A_L2_CACHE_DISABLE;
    } else if (rightNotL2Cache) {
        cacheMode = wqbmmv2_tiling::L2CacheMode::B_L2_CACHE_DISABLE;
    }
    OP_LOGD(opName_,
            "L2 cache params: totalSize:%lu, flagA:%d, flagB:%d, leftNotL2Cache:%d, rightNotL2Cache:%d, cacheMode:%d.",
            totalSize, static_cast<int32_t>(flagA), static_cast<int32_t>(flagB), static_cast<int32_t>(leftNotL2Cache),
            static_cast<int32_t>(rightNotL2Cache), static_cast<int32_t>(cacheMode));
    return cacheMode;
}

void WeightQuantBatchMatmulV2TilingASWVC::SetTilingData()
{
    vcTilingData_->mTailCnt = 1;
    vcTilingData_->nTailCnt = 1;
    vcTilingData_->mmadParam = GetVcMmadParam();
    vcTilingData_->l2CacheDisable = SetDisableL2cache(basicTiling_.baseM, basicTiling_.stepKa * basicTiling_.baseK,
                                                      basicTiling_.stepKb * basicTiling_.baseK, basicTiling_.baseN);
}

uint8_t WeightQuantBatchMatmulV2TilingASWVC::GetVcMmadParam() const
{
    const int32_t a16w8Shift = static_cast<int32_t>(static_cast<uint8_t>(shiftValue_));
    int32_t vcFixShift = FIX_SHIFT_VAL_LEN_A16W16 - (FIX_SHIFT_VAL_LEN_A16W8 - a16w8Shift);
    if (vcFixShift < FIX_SHIFT_VAL_MIN_F16XF16 || vcFixShift > FIX_SHIFT_VAL_MAX_F16XF16) {
        OP_LOGW(opName_,
                "asw-vc fixShiftVal %d converted from a16w8 shift %d is out of valid range [%d, %d], "
                "clamped for f16xf16 mmad",
                vcFixShift, a16w8Shift, FIX_SHIFT_VAL_MIN_F16XF16, FIX_SHIFT_VAL_MAX_F16XF16);
        vcFixShift = vcFixShift < FIX_SHIFT_VAL_MIN_F16XF16 ? FIX_SHIFT_VAL_MIN_F16XF16 : FIX_SHIFT_VAL_MAX_F16XF16;
    }
    return static_cast<uint8_t>(vcFixShift);
}

void WeightQuantBatchMatmulV2TilingASWVC::SetBatchParams()
{
    vcTilingData_->batchDimAll = static_cast<uint32_t>(matmulInfoPtr_->batchY);
    vcTilingData_->batchA1 = matmulInfoPtr_->batchX0;
    vcTilingData_->batchA2 = matmulInfoPtr_->batchX1;
    vcTilingData_->batchA3 = matmulInfoPtr_->batchX2;
    vcTilingData_->batchA4 = matmulInfoPtr_->batchX3;
    vcTilingData_->batchB1 = matmulInfoPtr_->batchWeight0;
    vcTilingData_->batchB2 = matmulInfoPtr_->batchWeight1;
    vcTilingData_->batchB3 = matmulInfoPtr_->batchWeight2;
    vcTilingData_->batchB4 = matmulInfoPtr_->batchWeight3;
    vcTilingData_->batchC1 = matmulInfoPtr_->batchY0;
    vcTilingData_->batchC2 = matmulInfoPtr_->batchY1;
    vcTilingData_->batchC3 = matmulInfoPtr_->batchY2;
    vcTilingData_->batchC4 = matmulInfoPtr_->batchY3;
    vcTilingData_->biasWithBatch = static_cast<uint64_t>(matmulInfoPtr_->biasWithBatch);
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWVC::DoLibApiTiling()
{
    vcTilingData_->m = matmulInfoPtr_->mSize;
    vcTilingData_->n = matmulInfoPtr_->nSize;
    vcTilingData_->k = matmulInfoPtr_->kSize;
    vcTilingData_->usedCoreNum = basicTiling_.usedCoreNum;
    vcTilingData_->baseM = basicTiling_.baseM;
    vcTilingData_->baseN = basicTiling_.baseN;
    vcTilingData_->baseK = basicTiling_.baseK;
    vcTilingData_->stepKa = basicTiling_.stepKa;
    vcTilingData_->stepKb = basicTiling_.stepKb;
    vcTilingData_->isBias = matmulInfoPtr_->hasBias;
    vcTilingData_->groupSize = static_cast<uint32_t>(matmulInfoPtr_->groupSize);
    // baseM*baseN*4B*2不超过L0C时开启L0C双缓冲（另含scale约束，见CheckAntiQuantScale）
    vcTilingData_->l0cDB = static_cast<uint8_t>(basicTiling_.dbL0c);
    vcTilingData_->l1BufferNum = static_cast<uint8_t>(runInfo_.l1BufferNum);
    LogTilingData();
    return ge::GRAPH_SUCCESS;
}

void WeightQuantBatchMatmulV2TilingASWVC::LogTilingData() const
{
    OP_LOGD(opName_,
            "ASW-VC tiling: core=%u, shape=[%u, %u, %u], base=[%u, %u, %u], tail=[%u, %u], "
            "step=[%u, %u], mmad=%u, l1BufferNum=%u, l0cDB=%u, l2CacheDisable=%u, isBias=%u, groupSize=%u",
            vcTilingData_->usedCoreNum, vcTilingData_->m, vcTilingData_->n, vcTilingData_->k, vcTilingData_->baseM,
            vcTilingData_->baseN, vcTilingData_->baseK, vcTilingData_->mTailCnt, vcTilingData_->nTailCnt,
            vcTilingData_->stepKa, vcTilingData_->stepKb, static_cast<uint32_t>(vcTilingData_->mmadParam),
            static_cast<uint32_t>(vcTilingData_->l1BufferNum), static_cast<uint32_t>(vcTilingData_->l0cDB),
            static_cast<uint32_t>(vcTilingData_->l2CacheDisable), vcTilingData_->isBias, vcTilingData_->groupSize);
    OP_LOGD(opName_,
            "ASW-VC batch: dim=%u, A=[%lu, %lu, %lu, %lu], B=[%lu, %lu, %lu, %lu], "
            "C=[%lu, %lu, %lu, %lu], biasWithBatch=%lu",
            vcTilingData_->batchDimAll, vcTilingData_->batchA1, vcTilingData_->batchA2, vcTilingData_->batchA3,
            vcTilingData_->batchA4, vcTilingData_->batchB1, vcTilingData_->batchB2, vcTilingData_->batchB3,
            vcTilingData_->batchB4, vcTilingData_->batchC1, vcTilingData_->batchC2, vcTilingData_->batchC3,
            vcTilingData_->batchC4, vcTilingData_->biasWithBatch);
    OP_LOGD(opName_, "ASW-VC tilingKey=%lu, tilingDataSize=%zu", GetTilingKey(), vcTilingDataSize_);
}

uint64_t WeightQuantBatchMatmulV2TilingASWVC::GetTilingKey() const
{
    uint64_t socVersionType = WQBMMV2_SOC_SUPPORT_MMAD_S8S4;
    uint64_t subSocVersionType = WQBMMV2_DEFAULT;
    uint64_t antiquantScenario = WQBMMV2_DEFAULT;
    // 反量化在 vector 核完成（ASW 为 FIXPIPE_ANTIQUANT），算法大类归 VECTOR_ANTIQUANT
    uint64_t algorithm = WQBMMV2_ALGO_VECTOR_ANTIQUANT;
    uint64_t subAlgorithm = WQBMMV2_SUB_ALGO_ASW_VC;
    uint64_t templateCustom = static_cast<uint64_t>(Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_2);
    uint64_t apiConstexpr = 0UL;
    bool transA = matmulInfoPtr_->transA;
    bool transB = matmulInfoPtr_->transB;
    uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
    uint64_t quantType = matmulInfoPtr_->cDtype == ge::DT_INT8 ? static_cast<uint64_t>(matmulInfoPtr_->quantType) :
                                                                 static_cast<uint64_t>(QuantType::NONE);
    bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    bool hasBias = matmulInfoPtr_->hasBias;
    bool isBiasFp32 = matmulInfoPtr_->biasDtype == ge::DT_FLOAT && matmulInfoPtr_->hasBias;
    bool isWeightNz = true;
    uint64_t tilingKey = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm,
                                            subAlgorithm, templateCustom, apiConstexpr, transA, transB, antiquantType,
                                            quantType, hasAntiquantOffset, hasBias, isBiasFp32, isWeightNz);
    return tilingKey;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWVC::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus WeightQuantBatchMatmulV2TilingASWVC::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", vcTilingDataSize_);
    OP_TILING_CHECK(vcTilingDataSize_ % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] is not aligned to 8", vcTilingDataSize_),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           static_cast<const void*>(vcTilingData_.get()), vcTilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    // 本平台每个 block 含 1 AIC + 1 AIV（MIX_AIC_1_1），blockDim 与 ASW 一致为 usedCoreNum 个 block
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(vcTilingDataSize_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2TilingASWVC, ASW_VC_PRIORITY);
} // namespace weight_quant_batch_matmul_v2
} // namespace optiling
