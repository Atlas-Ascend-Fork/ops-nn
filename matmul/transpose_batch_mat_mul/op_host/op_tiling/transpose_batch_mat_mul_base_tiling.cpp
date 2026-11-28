/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_batch_mat_mul_base_tiling.cpp
 * \brief
 */
#include <string>

#include "transpose_batch_mat_mul_base_tiling.h"
#include "util/math_util.h"
#include "log/log.h"
#include "op_host/tiling_key.h"
#include "error_util.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "matmul/common/op_host/log_format_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"
#include "graph/utils/type_utils.h"
#include "../../op_kernel/transpose_batch_mat_mul_tiling_key.h"

using namespace optiling::transpose_batch_mat_mul;

namespace {
constexpr size_t ATTR_NUM = 5;
constexpr size_t HF32_ATTR_INDEX = 3;
constexpr size_t ALLOW_DIM = 3;
constexpr size_t BATCH_IDX = 0;
constexpr size_t M_IDX = 1;
constexpr size_t KA_IDX = 2;
constexpr size_t KB_IDX = 1;
constexpr size_t N_IDX = 2;
constexpr size_t BIAS_IDX = 2;
constexpr size_t SCALE_IDX = 3;
constexpr uint64_t BLOCK_CUBE = 16;
constexpr uint64_t NO_BATCH_SHAPE_DIM = 2;
constexpr uint64_t ONE_BATCH_SHAPE_DIM = 3;
constexpr uint64_t TWO_BATCH_SHAPE_DIM = 4;
constexpr uint64_t THREE_BATCH_SHAPE_DIM = 5;
constexpr uint64_t FOUR_BATCH_SHAPE_DIM = 6;
constexpr uint64_t DEFAULT_SIZE = 32;
constexpr uint64_t NUM_TWO = 2;
constexpr uint64_t BASIC_BLOCK_SIZE_32 = 32;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint64_t BASE_N_MAX = 512;
constexpr uint64_t BASE_BLOCK_MAX_SIZE = 1024;
constexpr uint64_t L0_BASE_BLOCK_MAX_BYTES = 2048;
constexpr uint64_t L0_MN_MAX_BYTES = 32768;
constexpr double CORE_UTILIZATION_THRESHOLD = 0.6;
constexpr int64_t SUPPORTED_INNER_AXIS = 65536;
constexpr uint64_t PERM_X1_1_0_2_CODE = 213UL;
constexpr uint64_t PERM_0_1_2_CODE = 123UL;
constexpr uint64_t PERM_0_2_1_CODE = 132UL;
} // namespace

namespace optiling {
namespace transpose_batch_mat_mul {
template <typename T>
T GetAlignNumWithDataType(T size, ge::DataType dtype)
{
    return size / static_cast<T>(ge::GetSizeByDataType(dtype));
}

static inline uint64_t LastPower2(uint64_t n)
{
    n |= n >> 1;  // spread the highest set bit to 2 bits
    n |= n >> 2;  // spread the highest set bit to 4 bits
    n |= n >> 4;  // spread the highest set bit to 8 bits
    n |= n >> 8;  // spread the highest set bit to 16 bits
    n |= n >> 16; // spread the highest set bit to 32 bits
    n |= n >> 32; // spread the highest set bit to 64 bits
    return (n & ~(n >> 1));
}

static inline uint64_t NextPower2(uint64_t n)
{
    uint64_t lastPower2 = LastPower2(n);
    return (n == lastPower2) ? lastPower2 : (lastPower2 << 1);
}

static inline bool IsValidPermList(const gert::ContinuousVector* permList)
{
    if ((permList == nullptr) || (permList->GetData() == nullptr) || (permList->GetSize() != ALLOW_DIM)) {
        return false;
    }
    const int64_t* perm = static_cast<const int64_t*>(permList->GetData());
    for (size_t i = 0; i < ALLOW_DIM; ++i) {
        if ((perm[i] < 0) || (perm[i] >= static_cast<int64_t>(ALLOW_DIM))) {
            return false;
        }
    }
    return true;
}

ge::graphStatus TransposeBatchMatMulBaseTiling::GetShapeAttrsInfo() // check whether the input attributes are supported
{
    args_.opName = context_->GetNodeName();
    OP_TILING_CHECK(args_.opName == nullptr, CUBE_INNER_ERR_REPORT("TransposeBatchMatMul", "get op name invalid"),
                    return ge::GRAPH_FAILED);
    OP_LOGI(args_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    if ((CheckArgs() != ge::GRAPH_SUCCESS) || (GetArgs() != ge::GRAPH_SUCCESS)) {
        OP_LOGE_WITHOUT_REPORT(args_.opName, "CheckArgs or GetArgs failed, see the preceding error log.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulBaseTiling::DoLibApiTiling()
{
    auto ret = MatmulV3BaseTiling::DoLibApiTiling();
    tbmmTilingData_.multiBatchInfo.batchUsedCoreNum = tbmmTilingData_.matmulTiling.matmulTiling.usedCoreNum;
    tbmmTilingData_.multiBatchInfo.aBatchDim3 = static_cast<uint32_t>(batchInfo_.batchA3);
    tbmmTilingData_.multiBatchInfo.aBatchDim2 = static_cast<uint32_t>(batchInfo_.batchA2);
    tbmmTilingData_.multiBatchInfo.aBatchDim1 = static_cast<uint32_t>(batchInfo_.batchA1);
    tbmmTilingData_.multiBatchInfo.aBatchDim0 = static_cast<uint32_t>(batchInfo_.batchA0);
    tbmmTilingData_.multiBatchInfo.bBatchDim3 = static_cast<uint32_t>(batchInfo_.batchB3);
    tbmmTilingData_.multiBatchInfo.bBatchDim2 = static_cast<uint32_t>(batchInfo_.batchB2);
    tbmmTilingData_.multiBatchInfo.bBatchDim1 = static_cast<uint32_t>(batchInfo_.batchB1);
    tbmmTilingData_.multiBatchInfo.bBatchDim0 = static_cast<uint32_t>(batchInfo_.batchB0);
    tbmmTilingData_.multiBatchInfo.cBatchDim3 = static_cast<uint32_t>(batchInfo_.batchC3);
    tbmmTilingData_.multiBatchInfo.cBatchDim2 = static_cast<uint32_t>(batchInfo_.batchC2);
    tbmmTilingData_.multiBatchInfo.cBatchDim1 = static_cast<uint32_t>(batchInfo_.batchC1);
    tbmmTilingData_.multiBatchInfo.cBatchDim0 = static_cast<uint32_t>(batchInfo_.batchC0);

    aBatchDimAll_ = batchInfo_.batchA0 * batchInfo_.batchA1 * batchInfo_.batchA2 * batchInfo_.batchA3;
    bBatchDimAll_ = batchInfo_.batchB0 * batchInfo_.batchB1 * batchInfo_.batchB2 * batchInfo_.batchB3;
    cBatchDimAll_ = batchInfo_.batchC0 * batchInfo_.batchC1 * batchInfo_.batchC2 * batchInfo_.batchC3;
    tbmmTilingData_.multiBatchInfo.aBatchDimAll = static_cast<uint32_t>(aBatchDimAll_);
    tbmmTilingData_.multiBatchInfo.bBatchDimAll = static_cast<uint32_t>(bBatchDimAll_);
    tbmmTilingData_.multiBatchInfo.cBatchDimAll = static_cast<uint32_t>(cBatchDimAll_);
    tbmmTilingData_.multiBatchInfo.batchTileBlock = static_cast<uint32_t>(cBatchDimAll_);
    if (CheckBMMTilingDataIsVaild()) {
        return ge::GRAPH_FAILED;
    }
    tbmmTilingData_.multiBatchInfo.biasWithBatch = static_cast<uint32_t>(batchInfo_.biasWithBatch);
    tbmmTilingData_.multiBatchInfo.mOri = static_cast<uint32_t>(args_.mOriValue);
    DoCommonTiling();
    tbmmTilingData_.matmulTiling.tileL2cacheTiling.mTileCntL2 = 1;
    tbmmTilingData_.matmulTiling.tileL2cacheTiling.nTileCntL2 = 1;

    tbmmTilingData_.matmulTiling.matmulRunInfo.transA = transA_;
    tbmmTilingData_.matmulTiling.matmulRunInfo.transB = transB_;
    tbmmTilingData_.batchSplitFactor = batchSplitFactor_;
    return ret;
}

/*
 * Algorithm to calculate the best (baseM, baseN) that gives even workload amongst iterations.
 * Parameter `divisor` is used to control the starting point of the algorithm.
 * Choosing different starting point can sometimes get better performence.
 * The starting point of `divisor = 2` is half of that of `divisor = 1`
 */
static void CalcBaseMN(uint64_t& baseM, uint64_t& baseN, const matmul_v3::MatmulV3Args& args, uint64_t divisor = 1UL)
{
    uint64_t dtypeSize = GetSizeByDataType(args.aType);
    // step 1: calc baseM
    auto getBestBaseM = [dtypeSize, divisor](bool transX1, uint64_t m, uint64_t n, uint64_t k) -> uint64_t {
        if (!transX1) {
            // Limit baseM to 2048B to avoid a too-small baseK; when k > 32, limit it to 1024B.
            uint64_t maxM = k >= BASIC_BLOCK_SIZE_32 ? BASE_BLOCK_MAX_SIZE : L0_BASE_BLOCK_MAX_BYTES;
            maxM /= (dtypeSize * divisor);
            uint64_t mTimes = ops::CeilDiv(m, maxM);
            uint64_t singleTimeM = ops::CeilDiv(m, mTimes);
            return ops::CeilAlign(singleTimeM, BLOCK_CUBE);
        }
        // baseN does not exceed 512. At least 32; keep baseM*n >= 32768B when n is small.
        const uint64_t nCalc = std::min(n, BASE_N_MAX);
        const uint64_t minBaseM = std::max(L0_MN_MAX_BYTES / dtypeSize / nCalc, BASIC_BLOCK_SIZE_32);
        // L0 limit is 2048B; try 1024B for fp16. At least 16.
        const uint64_t maxBaseM = std::min(NextPower2(m), (L0_BASE_BLOCK_MAX_BYTES / dtypeSize / divisor));
        uint64_t bestBaseM = std::max(maxBaseM, BLOCK_CUBE);
        for (uint64_t candidate = bestBaseM; candidate >= minBaseM; candidate >>= uint64_t(1)) {
            if (m % candidate == uint64_t(0)) {
                break;
            }
            if (ops::CeilAlign(m, candidate) < ops::CeilAlign(m, bestBaseM)) {
                bestBaseM = candidate;
            }
        }
        return bestBaseM;
    };
    baseM = getBestBaseM(args.isATrans, args.mValue, args.nValue, args.kValue);
    // step 2: calc baseN
    auto getBestBaseN = [dtypeSize, baseM, divisor](bool transX2, uint64_t n, uint64_t k) -> uint64_t {
        const uint64_t minBaseN = std::max(std::min(L0_MN_MAX_BYTES / dtypeSize / baseM, BASE_N_MAX / dtypeSize),
                                           BASIC_BLOCK_SIZE_32);
        const uint64_t dtypeBaseNMax = L0_BASE_BLOCK_MAX_BYTES / dtypeSize;
        const uint64_t maxBaseN = (transX2 && k >= BASIC_BLOCK_SIZE_32) ? std::min(BASE_BLOCK_MAX_SIZE, dtypeBaseNMax) :
                                                                          dtypeBaseNMax / divisor;
        uint64_t bestBaseN = std::max(std::min(NextPower2(n), maxBaseN), BLOCK_CUBE);
        for (uint64_t candidate = bestBaseN; candidate >= minBaseN; candidate >>= uint64_t(1)) {
            if (n % candidate == uint64_t(0)) {
                break;
            }
            if (ops::CeilAlign(n, candidate) < ops::CeilAlign(n, bestBaseN)) {
                bestBaseN = candidate;
            }
        }
        return std::min(LastPower2(L0_MN_MAX_BYTES / baseM), bestBaseN); // L0C limits baseM * baseN <= 32768
    };
    baseN = getBestBaseN(args.isBTrans, args.nValue, args.kValue);
}

static void TuneBaseMN(matmul_v3::MatmulV3RunInfo& runInfo, const matmul_v3::MatmulV3Args& args, uint64_t batchC,
                       uint64_t aicNum)
{
    uint64_t& oriBaseM = runInfo.baseM;
    uint64_t& oriBaseN = runInfo.baseN;
    CalcBaseMN(oriBaseM, oriBaseN, args);
    if (args.aType == ge::DT_FLOAT) {
        return;
    }
    // for bf16 fp16 try a different set of thresholds
    uint64_t newBaseM;
    uint64_t newBaseN;
    CalcBaseMN(newBaseM, newBaseN, args, 2UL); // choose half of the starting point by setting divisor = 2
    // evaluate core utilization
    auto getCoreUtilization = [args, batchC, aicNum](uint64_t baseM, uint64_t baseN) -> double {
        uint64_t cnt = ops::CeilDiv(args.mValue, baseM) * ops::CeilDiv(args.nValue, baseN) * batchC;
        return static_cast<double>(cnt) / ops::CeilAlign(cnt, aicNum);
    };
    double oriCoreUtil = getCoreUtilization(oriBaseM, oriBaseN);
    // CORE_UTILIZATION_THRESHOLD is an empirical value.
    if ((oriCoreUtil < CORE_UTILIZATION_THRESHOLD) && (getCoreUtilization(newBaseM, newBaseN) > oriCoreUtil)) {
        oriBaseM = newBaseM;
        oriBaseN = newBaseN;
        return;
    }
    // evaluate data copy workload
    auto getDataCopyRatio = [](uint64_t baseM, uint64_t baseN) -> double {
        return 1.0 / static_cast<double>(baseM) + 1.0 / static_cast<double>(baseN);
    };
    if (getDataCopyRatio(newBaseM, newBaseN) < getDataCopyRatio(oriBaseM, oriBaseN)) {
        oriBaseM = newBaseM;
        oriBaseN = newBaseN;
    }
}

static void TuneBaseMKN(matmul_v3::MatmulV3RunInfo& runInfo, const matmul_v3::MatmulV3Args& args, uint64_t batchC,
                        uint64_t aicNum)
{
    uint64_t& baseM = runInfo.baseM;
    uint64_t& baseN = runInfo.baseN;
    uint64_t& baseK = runInfo.baseK;
    OP_LOGD(args.opName, "before DoCommonTiling baseM, baseN, baseK[%lu, %lu, %lu]", baseM, baseN, baseK);
    bool isSmallShape = (args.mValue < BASIC_BLOCK_SIZE_256 || args.nValue <= BASIC_BLOCK_SIZE_256);
    bool useBaseBlock = ((baseM == BASIC_BLOCK_SIZE_128 || baseM == BASIC_BLOCK_SIZE_256) &&
                         (baseN == BASIC_BLOCK_SIZE_128 || baseN == BASIC_BLOCK_SIZE_256));
    bool isL2Tile = (runInfo.l2Info.mTile > 1 || runInfo.l2Info.nTile > 1);
    if ((!isSmallShape && useBaseBlock) || isL2Tile) {
        return;
    }

    TuneBaseMN(runInfo, args, batchC, aicNum);
    uint64_t maxBaseK = L0_MN_MAX_BYTES / GetSizeByDataType(args.aType) / std::max(baseM, baseN); // L0AB limit
    baseK = std::max(std::min(ops::FloorAlign(maxBaseK, BLOCK_CUBE), ops::CeilAlign(args.kValue, BLOCK_CUBE)),
                     BLOCK_CUBE);
    OP_LOGD(args.opName, "after DoCommonTiling baseM, baseN, baseK[%lu, %lu, %lu]", baseM, baseN, baseK);
}

void TransposeBatchMatMulBaseTiling::ResetBasicBlock(uint64_t tempBaseM, uint64_t tempBaseN)
{
    OP_TILING_CHECK(tempBaseM == 0 && tempBaseN == 0,
                    OP_LOGW(args_.opName, "tempBaseM == 0 && tempBaseN == 0 is invalid"), return);
    uint64_t baseKAlignNum = (!args_.isATrans && args_.isBTrans) ?
                                 GetAlignNumWithDataType(BASIC_BLOCK_SIZE_256, args_.aType) :
                                 BLOCK_CUBE;
    uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(args_.kValue), baseKAlignNum);
    uint64_t maxBaseK = GetAlignNumWithDataType(compileInfo_.l0ASize / NUM_TWO, args_.aType) /
                        std::max(tempBaseM, tempBaseN);
    if (maxBaseK >= baseKAlignNum) {
        runInfo_.baseM = tempBaseM;
        runInfo_.baseN = tempBaseN;
        maxBaseK = ops::FloorAlign(maxBaseK, baseKAlignNum);
        runInfo_.baseK = std::min(kValueAlign, maxBaseK);
    }
}

void TransposeBatchMatMulBaseTiling::BaseLoadBalance()
{
    uint64_t baseMAlignNum = args_.isATrans ? GetAlignNumWithDataType(BASIC_BLOCK_SIZE_256, args_.aType) : BLOCK_CUBE;
    uint64_t baseNAlignNum = !args_.isBTrans ? GetAlignNumWithDataType(BASIC_BLOCK_SIZE_256, args_.aType) : BLOCK_CUBE;
    uint64_t mMaxTile = ops::CeilDiv(args_.mValue, baseMAlignNum);
    uint64_t nMaxTile = ops::CeilDiv(args_.nValue, baseNAlignNum);
    uint64_t tempBaseM = runInfo_.baseM;
    uint64_t tempBaseN = runInfo_.baseN;
    uint64_t coreNumMN = compileInfo_.aicNum / batchInfo_.batchC;
    if (mMaxTile * nMaxTile >= coreNumMN || (!args_.isATrans && args_.isBTrans)) {
        uint64_t mCore = ops::CeilDiv(args_.mValue, runInfo_.baseM);
        uint64_t nCore = ops::CeilDiv(args_.nValue, runInfo_.baseN);
        if (mMaxTile < nMaxTile || (mMaxTile == nMaxTile && baseNAlignNum == BLOCK_CUBE)) {
            tempBaseM = ops::CeilAlign(ops::CeilDiv(args_.mValue, mCore), baseMAlignNum);
            mCore = ops::CeilDiv(args_.mValue, tempBaseM);
            nCore = ops::FloorDiv(coreNumMN, mCore);
            tempBaseN = ops::CeilAlign(ops::CeilDiv(args_.nValue, nCore), baseNAlignNum);
        } else {
            tempBaseN = ops::CeilAlign(ops::CeilDiv(args_.nValue, nCore), baseNAlignNum);
            nCore = ops::CeilDiv(args_.nValue, tempBaseN);
            mCore = ops::FloorDiv(coreNumMN, nCore);
            tempBaseM = ops::CeilAlign(ops::CeilDiv(args_.mValue, mCore), baseMAlignNum);
        }

        while (tempBaseN >= tempBaseM * NUM_TWO && nCore < coreNumMN / NUM_TWO && tempBaseN != baseNAlignNum) {
            nCore *= NUM_TWO;
            mCore = ops::FloorDiv(coreNumMN, nCore);
            tempBaseM = ops::CeilAlign(ops::CeilDiv(args_.mValue, mCore), baseMAlignNum);
            tempBaseN = ops::CeilAlign(ops::CeilDiv(args_.nValue, nCore), baseNAlignNum);
            mCore = ops::CeilDiv(args_.mValue, static_cast<uint64_t>(tempBaseM));
            nCore = ops::CeilDiv(args_.nValue, static_cast<uint64_t>(tempBaseN));
        }

        while (tempBaseM >= tempBaseN * NUM_TWO && mCore < coreNumMN / NUM_TWO && tempBaseM != baseMAlignNum) {
            mCore *= NUM_TWO;
            nCore = ops::FloorDiv(coreNumMN, mCore);
            tempBaseM = ops::CeilAlign(ops::CeilDiv(args_.mValue, mCore), baseMAlignNum);
            tempBaseN = ops::CeilAlign(ops::CeilDiv(args_.nValue, nCore), baseNAlignNum);
            mCore = ops::CeilDiv(args_.mValue, static_cast<uint64_t>(tempBaseM));
            nCore = ops::CeilDiv(args_.nValue, static_cast<uint64_t>(tempBaseN));
        }
        ResetBasicBlock(tempBaseM, tempBaseN);
    }
}

void TransposeBatchMatMulBaseTiling::DoCommonTiling()
{
    TuneBaseMKN(runInfo_, args_, cBatchDimAll_, compileInfo_.aicNum);
    uint64_t mCnt = ops::CeilDiv(args_.mValue, runInfo_.baseM);
    uint64_t nCnt = ops::CeilDiv(args_.nValue, runInfo_.baseN);
    if (mCnt * nCnt * batchInfo_.batchC < compileInfo_.aicNum) {
        BaseLoadBalance();
    }
    uint64_t baseM = runInfo_.baseM;
    uint64_t baseN = runInfo_.baseN;
    uint64_t baseK = runInfo_.baseK;
    OP_TILING_CHECK(
        (baseM == 0UL) || (baseN == 0UL) || (baseK == 0UL),
        CUBE_INNER_ERR_REPORT(args_.opName, "invalid base block, baseM:%lu, baseN:%lu, baseK:%lu", baseM, baseN, baseK),
        return);
    constexpr uint64_t l1ReservedBytes = 256;
    constexpr uint64_t maxBaseNWithBias = 256;
    constexpr uint64_t l1BiasTableBytes = 1024;
    constexpr uint64_t l1DepthAlign = 4;
    bool hasQuant = context_->GetOptionalInputShape(3) != nullptr; // scale is the 3rd input
    uint64_t totalL1Size = compileInfo_.l1Size + l1ReservedBytes;  // 256B is reserved for rpc, not used here
    if (hasQuant) {
        totalL1Size -= cBatchDimAll_ * args_.nValue * sizeof(uint64_t);
        OP_LOGI(args_.opName, "Quant function is activated.");
    }
    if (args_.hasBias) {
        totalL1Size -= l1BiasTableBytes;
        baseN = std::min(maxBaseNWithBias, baseN); // baseN must not exceed 256 when bias exists
    }
    // With double buffering, L1 can store 65536 fp32 elements.
    uint64_t depthA1 = (totalL1Size / NUM_TWO / aDtypeSize_ / (baseM * baseK) / l1DepthAlign) * l1DepthAlign;
    uint64_t depthB1 = (totalL1Size / NUM_TWO / aDtypeSize_ / (baseN * baseK) / l1DepthAlign) * l1DepthAlign;
    depthA1 = std::max(NUM_TWO, depthA1);
    depthB1 = std::max(NUM_TWO, depthB1);
    uint64_t stepKa = depthA1 / NUM_TWO;
    uint64_t stepKb = depthB1 / NUM_TWO;
    if (stepKa > stepKb) {
        stepKa = stepKa / stepKb * stepKb;
        depthA1 = stepKa * NUM_TWO;
    } else {
        stepKb = stepKb / stepKa * stepKa;
        depthB1 = stepKb * NUM_TWO;
    }
    tbmmTilingData_.matmulTiling.matmulTiling.depthA1 = depthA1;
    tbmmTilingData_.matmulTiling.matmulTiling.depthB1 = depthB1;
    tbmmTilingData_.matmulTiling.matmulTiling.stepKa = stepKa;
    tbmmTilingData_.matmulTiling.matmulTiling.stepKb = stepKb;
    tbmmTilingData_.matmulTiling.matmulTiling.stepM = 1;
    tbmmTilingData_.matmulTiling.matmulTiling.stepN = 1;
    tbmmTilingData_.matmulTiling.matmulTiling.singleCoreM = std::min(args_.mValue, baseM);
    tbmmTilingData_.matmulTiling.matmulTiling.singleCoreN = std::min(args_.nValue, baseN);
    tbmmTilingData_.matmulTiling.matmulTiling.singleCoreK = args_.kValue;
    tbmmTilingData_.matmulTiling.matmulTiling.baseM = baseM;
    tbmmTilingData_.matmulTiling.matmulTiling.baseN = baseN;
    tbmmTilingData_.matmulTiling.matmulTiling.baseK = baseK;
    uint64_t batchSplitMode = batchSplitFactor_ > 1 ? 1 : 0;
    uint64_t ppMatmulMode = 0;
    uint64_t permX1 = 0;
    uint64_t permX2 = 0;
    if (transA_ == PERM_X1_1_0_2_CODE) {
        // 2 is the lexicographic code of permList, 2 -> [1,0,2].
        permX1 = 2;
    } else if (transA_ == PERM_0_1_2_CODE) {
        permX1 = 0;
    }
    tilingKey_ = GET_TPL_TILING_KEY(batchSplitMode, ppMatmulMode, permX1, permX2, );
}

bool TransposeBatchMatMulBaseTiling::CheckBMMTilingDataIsVaild() const
{
    return (optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchA3, args_.opName, "batchInfo_.batchA3") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchA2, args_.opName, "batchInfo_.batchA2") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchA1, args_.opName, "batchInfo_.batchA1") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchA0, args_.opName, "batchInfo_.batchA0") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchB3, args_.opName, "batchInfo_.batchB3") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchB2, args_.opName, "batchInfo_.batchB2") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchB1, args_.opName, "batchInfo_.batchB1") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchB0, args_.opName, "batchInfo_.batchB0") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchC3, args_.opName, "batchInfo_.batchC3") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchC2, args_.opName, "batchInfo_.batchC2") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchC1, args_.opName, "batchInfo_.batchC1") ||
            optiling::matmul_v3::CheckNumberIsValid(batchInfo_.batchC0, args_.opName, "batchInfo_.batchC0") ||
            optiling::matmul_v3::CheckNumberIsValid(aBatchDimAll_, args_.opName, "batchInfo_.aBatchDimAll") ||
            optiling::matmul_v3::CheckNumberIsValid(bBatchDimAll_, args_.opName, "batchInfo_.bBatchDimAll") ||
            optiling::matmul_v3::CheckNumberIsValid(cBatchDimAll_, args_.opName, "batchInfo_.cBatchDimAll"));
}

ge::graphStatus TransposeBatchMatMulBaseTiling::PostTiling()
{
    size_t tilingDataSize = sizeof(TBMMTilingData);
    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(args_.opName, "tiling data size[%zu] is not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           static_cast<void*>(&tbmmTilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d, requiredBytes=%zu, availableBytes=%zu", ret,
                tilingDataSize, context_->GetRawTilingData()->GetCapacity());
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    context_->SetBlockDim(compileInfo_.aicNum);
    workspaceSize_ = std::max(workspaceSize_, DEFAULT_SIZE);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "workspaces is null"),
                    return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

uint64_t TransposeBatchMatMulBaseTiling::GetTilingKey() const { return tilingKey_; }

ge::graphStatus TransposeBatchMatMulBaseTiling::CheckArgs()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    if (context_->GetOptionalInputShape(BIAS_IDX) != nullptr) { // bias is input index 2
        args_.hasBias = true;
        if (context_->GetOptionalInputDesc(BIAS_IDX) == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "bias", "bias cannot be empty");
            return ge::GRAPH_FAILED;
        }
    }
    auto* shapeScale = context_->GetOptionalInputShape(SCALE_IDX);
    if (shapeScale != nullptr) {
        const gert::Shape& scaleShape = shapeScale->GetShape();
        OP_TILING_CHECK(scaleShape.GetDimNum() == 0,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(args_.opName, "scale", "0",
                                                                 "The shape dim of scale must be greater than 0"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            scaleShape.GetDim(0) >= SUPPORTED_INNER_AXIS,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                args_.opName, "scale", std::to_string(scaleShape.GetDim(0)).c_str(),
                Ops::NN::FormatString("The first dimension of %s must be less than %ld", "scale", SUPPORTED_INNER_AXIS)
                    .c_str()),
            return ge::GRAPH_FAILED);
    }
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_,
                                    attrs->GetAttrPointer<bool>(ATTR_NUM - 2)); // check the second-to-last attribute
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1));
    }
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    return ge::GRAPH_SUCCESS;
}

static inline void GetFormat(const gert::TilingContext& context, optiling::matmul_v3::MatmulV3Args& args)
{
    ge::Format formatA = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(0)->GetStorageFormat()));
    ge::Format formatB = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(1)->GetStorageFormat()));
    ge::Format formatOut = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetOutputDesc(0)->GetStorageFormat()));
    args.aFormat = (formatA != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatA;
    args.bFormat = (formatB != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatB;
    args.outFormat = (formatOut != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatOut;
}

static inline void GetDtype(const gert::TilingContext& context, optiling::matmul_v3::MatmulV3Args& args)
{
    // op_impl_mode_enum: 0x1: default 0x2: high_performance 0x4: high_precision 0x8: super_performance
    // 0x10: support_of_bound_index 0x20: enable_float_32_execution 0x40: enable_hi_float_32_execution
    if (context.GetAttrs()->GetAttrNum() >= HF32_ATTR_INDEX + 1) {
        args.isHf32 = *context.GetAttrs()->GetAttrPointer<bool>(HF32_ATTR_INDEX) ? 1 : 0;
    }
    OP_LOGD(args.opName, "Hf32 flag is: %d", args.isHf32);

    args.aType = context.GetInputDesc(0)->GetDataType();
    args.bType = context.GetInputDesc(1)->GetDataType();
    args.cType = context.GetOutputDesc(0)->GetDataType();
    if (args.hasBias) {
        args.biasType = context.GetOptionalInputDesc(BIAS_IDX)->GetDataType();
    }
}

static ge::graphStatus OpSpecificCheck(const optiling::matmul_v3::MatmulV3Args& args)
{
    // format check
    OP_TILING_CHECK((args.aFormat == ge::FORMAT_FRACTAL_NZ) || (args.outFormat == ge::FORMAT_FRACTAL_NZ),
                    OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                        args.opName, "x1, y",
                        Ops::NN::FormatString("%s, %s", ge::TypeUtils::FormatToSerialString(args.aFormat).c_str(),
                                              ge::TypeUtils::FormatToSerialString(args.outFormat).c_str())
                            .c_str(),
                        Ops::NN::FormatString("The formats of %s must be %s", "x1, y", "ND").c_str()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK((args.bFormat == ge::FORMAT_FRACTAL_NZ) && args.hasBias,
                    OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(args.opName, "x2", "FRACTAL_NZ",
                                                           Ops::NN::FormatString("When optional parameter %s exists, "
                                                                                 "the format of %s cannot be %s",
                                                                                 "bias", "x2", "FRACTAL_NZ")
                                                               .c_str()),
                    return ge::GRAPH_FAILED);
    // dtype check
    std::vector<ge::DataType> dtype = {args.aType, args.bType, args.cType};
    if (args.hasBias) {
        dtype.push_back(args.biasType);
    }

    auto isValidDtype = [&args](const std::vector<ge::DataType>& dtypeList) -> ge::graphStatus {
        const std::vector<std::vector<ge::DataType> > dtypeSuportList = {
            // x1,              x2,             y,              bias
            {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
            {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT},
            {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT},
            {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
            {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT}};
        for (auto& supported : dtypeSuportList) {
            if (std::equal(dtypeList.begin(), dtypeList.end(), supported.begin())) {
                return ge::GRAPH_SUCCESS;
            }
        }
        std::string paramNames = "x1, x2, y";
        std::string dtypeStr = ge::TypeUtils::DataTypeToSerialString(args.aType) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(args.bType) + ", " +
                               ge::TypeUtils::DataTypeToSerialString(args.cType);
        if (args.hasBias) {
            paramNames += ", bias";
            dtypeStr += ", " + ge::TypeUtils::DataTypeToSerialString(args.biasType);
        }
        std::string supportedStr;
        for (const auto& supported : dtypeSuportList) {
            if (!supportedStr.empty()) {
                supportedStr += ", ";
            }
            supportedStr += "(";
            const size_t supportedNum = args.hasBias ? supported.size() : supported.size() - 1;
            for (size_t i = 0; i < supportedNum; ++i) {
                if (i != 0) {
                    supportedStr += ", ";
                }
                supportedStr += ge::TypeUtils::DataTypeToSerialString(supported[i]);
            }
            supportedStr += ")";
        }
        const std::string reason = "The dtypes of " + paramNames + " must be within the range " + supportedStr;
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(args.opName, paramNames.c_str(), dtypeStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    };
    return isValidDtype(dtype);
}

ge::graphStatus TransposeBatchMatMulBaseTiling::GetShapeMKN(const gert::Shape& aShape, const gert::Shape& bShape)
{
    const int64_t* aPerm = static_cast<const int64_t*>(aPermList_->GetData());
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList_->GetData());
    const int64_t m = aShape[aPerm[M_IDX]];
    const int64_t kA = aShape[aPerm[KA_IDX]];
    const int64_t kB = bShape[bPerm[KB_IDX]];
    const int64_t n = bShape[bPerm[N_IDX]];
    args_.isATrans = aPerm[M_IDX] > aPerm[KA_IDX];
    args_.isBTrans = bPerm[KB_IDX] > bPerm[N_IDX];

    OP_TILING_CHECK(
        kA != kB,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args_.opName, "x1, x2",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(aShape).c_str(), Ops::Base::ToString(bShape).c_str())
                .c_str(),
            "The k-axis of x1 and x2 must be equal"),
        return ge::GRAPH_FAILED);

    auto isValidDimValue = [](int64_t dim) -> bool { return (dim > 0) && (dim <= INT32_MAX); };
    if (!isValidDimValue(m) || !isValidDimValue(kA) || !isValidDimValue(n)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args_.opName, "x1, x2",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(aShape).c_str(), Ops::Base::ToString(bShape).c_str())
                .c_str(),
            Ops::NN::FormatString("m, k and n of %s must be within the range (0, INT32_MAX]", "x1, x2").c_str());
        return ge::GRAPH_FAILED;
    }
    args_.mValue = static_cast<uint64_t>(m);
    args_.kValue = static_cast<uint64_t>(kA);
    args_.nValue = static_cast<uint64_t>(n);
    args_.mOriValue = args_.mValue;
    args_.nOriValue = args_.nValue;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulBaseTiling::GetShapeBatch(const gert::Shape& aShape, const gert::Shape& bShape)
{
    const int64_t* aPerm = static_cast<const int64_t*>(aPermList_->GetData());
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList_->GetData());
    const int64_t batchA = aShape[aPerm[BATCH_IDX]];
    const int64_t batchB = bShape[bPerm[BATCH_IDX]];
    OP_TILING_CHECK(
        (batchA <= 0) || (batchB <= 0),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args_.opName, "x1, x2",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(aShape).c_str(), Ops::Base::ToString(bShape).c_str())
                .c_str(),
            "The batch-axis of x1 and x2 must be positive"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        batchA != batchB,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args_.opName, "x1, x2",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(aShape).c_str(), Ops::Base::ToString(bShape).c_str())
                .c_str(),
            "The batch-axis of x1 and x2 must be equal"),
        return ge::GRAPH_FAILED);
    batchInfo_.batchA3 = batchA;
    batchInfo_.batchB3 = batchA;
    batchInfo_.batchC3 = batchA;
    batchInfo_.batchA = batchA;
    batchInfo_.batchB = batchA;
    batchInfo_.batchC = batchA;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulBaseTiling::GetShapeBias()
{
    if (args_.hasBias) {
        const gert::Shape& biasShape = context_->GetInputShape(BIAS_IDX)->GetOriginShape();
        OP_TILING_CHECK(biasShape.GetDimNum() == 0,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(args_.opName, "bias", "0",
                                                                 "The shape dim of bias must be greater than 0"),
                        return ge::GRAPH_FAILED);
        const int64_t biasValue = biasShape.GetDim(biasShape.GetDimNum() - 1);
        if (args_.nOriValue != static_cast<uint64_t>(biasValue)) {
            const std::string reason = "The current value is not within the valid range. The valid range is [" +
                                       std::to_string(args_.nOriValue) + ", " + std::to_string(args_.nOriValue) + "]";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(args_.opName, "bias", std::to_string(biasValue).c_str(),
                                                  reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    batchInfo_.biasWithBatch = false;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulBaseTiling::GetShape()
{
    const gert::Shape& aShape = context_->GetInputShape(0)->GetOriginShape();
    const gert::Shape& bShape = context_->GetInputShape(1)->GetOriginShape();
    const gert::Shape& cShape = context_->GetOutputShape(0)->GetOriginShape();
    const size_t aDimNum = aShape.GetDimNum();
    const size_t bDimNum = bShape.GetDimNum();
    const size_t cDimNum = cShape.GetDimNum();
    OP_TILING_CHECK(
        (aDimNum != ALLOW_DIM) || (bDimNum != ALLOW_DIM) || (cDimNum != ALLOW_DIM),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            args_.opName, "x1, x2, y", Ops::NN::FormatString("%zu, %zu, %zu", aDimNum, bDimNum, cDimNum).c_str(),
            Ops::NN::FormatString("The shape dims of %s must be %zu", "x1, x2, y", ALLOW_DIM).c_str()),
        return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    aPermList_ = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    bPermList_ = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, aPermList_);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, bPermList_);
    OP_TILING_CHECK(
        !IsValidPermList(aPermList_) || !IsValidPermList(bPermList_),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(args_.opName, "permX1, permX2", "invalid perm list",
                                               "Each perm must have 3 elements and each element must be in [0, 2]"),
        return ge::GRAPH_FAILED);

    if (GetShapeMKN(aShape, bShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(args_.opName, "get m/k/n failed");
        return ge::GRAPH_FAILED;
    }
    if (GetShapeBatch(aShape, bShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(args_.opName, "get batch failed");
        return ge::GRAPH_FAILED;
    }
    if (GetShapeBias() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(args_.opName, "get bias failed");
        return ge::GRAPH_FAILED;
    }

    // Pass perm through transA/transB. Encode each element as value + 1 to avoid leading zeros,
    // for example {1, 0, 2} becomes {2, 1, 3} and then 213.
    const int64_t* permX1 = static_cast<const int64_t*>(aPermList_->GetData());
    transA_ = 0UL;
    for (uint32_t i = 0; i < aPermList_->GetSize(); i++) {
        transA_ = transA_ * 10 + permX1[i] + 1; // shift by one decimal digit
    }
    const int64_t* permX2 = static_cast<const int64_t*>(bPermList_->GetData());
    transB_ = 0UL;
    for (uint32_t i = 0; i < bPermList_->GetSize(); i++) {
        transB_ = transB_ * 10 + permX2[i] + 1; // shift by one decimal digit
    }
    OP_LOGI(args_.opName, "transA: %lu, transB: %lu", transA_, transB_);

    uint64_t inputBatch = batchInfo_.batchC;
    uint64_t inputK = args_.kValue;
    uint64_t inputN = args_.nValue;
    bool supportBatchM = true;

    if (transA_ == PERM_X1_1_0_2_CODE) {
        supportBatchM = (inputBatch * inputK < SUPPORTED_INNER_AXIS);
    } else if (transA_ == PERM_0_1_2_CODE) {
        supportBatchM = (inputK < SUPPORTED_INNER_AXIS);
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            args_.opName, "permX1", Ops::NN::FormatString("%lu", transA_).c_str(),
            Ops::NN::FormatString("The value of %s must be %s", "permX1", "{[1,0,2], [0,1,2]}").c_str());
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK((transB_ != PERM_0_1_2_CODE) && (transB_ != PERM_0_2_1_CODE),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        args_.opName, "permX2", Ops::NN::FormatString("%lu", transB_).c_str(),
                        Ops::NN::FormatString("The value of %s must be %s", "permX2", "{[0,1,2], [0,2,1]}").c_str()),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        !supportBatchM,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args_.opName, "x1, x2",
            Ops::NN::FormatString("b, m, n, k = %lu, %lu, %lu, %lu", inputBatch, args_.mValue, inputN, inputK).c_str(),
            Ops::NN::FormatString("The inner axis of %s must be less than %ld", "x1", SUPPORTED_INNER_AXIS).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulBaseTiling::GetArgs()
{
    GetFormat(*context_, args_);
    GetDtype(*context_, args_);
    // get batch_split_factor
    if (context_->GetAttrs()->GetAttrNum() >= ATTR_NUM) {
        batchSplitFactor_ = std::max(*(context_->GetAttrs()->GetAttrPointer<int32_t>(ATTR_NUM - 1)), 1);
    }
    if (GetShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(args_.opName, "get shape failed");
        return ge::GRAPH_FAILED;
    }
    if (OpSpecificCheck(args_) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(args_.opName, "format and dtype check failed");
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(
        (args_.bFormat == ge::FORMAT_FRACTAL_NZ) &&
            ((transA_ != PERM_X1_1_0_2_CODE) || (transB_ != PERM_0_1_2_CODE) || batchSplitFactor_ > 1),
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
            args_.opName, "x2", "FRACTAL_NZ",
            "When x2 is FRACTAL_NZ, perm_x1, perm_x2 and batch_split_factor must be [1,0,2], [0,1,2] and 1"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace transpose_batch_mat_mul
} // namespace optiling
