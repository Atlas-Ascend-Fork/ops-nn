/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rms_norm_tiling_arch35.cpp
 * \brief RmsNorm Op Tiling
 */
#include <iostream>
#include <limits>
#include "rms_norm_tiling.h"

namespace optiling {
namespace {
// 每核保留足够的向量工作量，摊薄转置和跨核归约开销。
constexpr uint64_t TRANS_PHYSICAL_LARGE_MIN_VL_PER_CORE = 8;
constexpr uint64_t SPLIT_AR_MIN_PARALLEL_R_BLOCKS = 2;
constexpr uint64_t SPLIT_AR_MIN_DB_TILES_PER_CORE = 8;
// With A <= this work threshold, the R-core planner keeps each non-tail
// R block at least one full UB tile long.
constexpr uint64_t SPLIT_AR_MAX_A_ROWS = SPLIT_AR_MIN_DB_TILES_PER_CORE;
constexpr uint64_t AR_CACHE_LEVEL_COUNT = std::numeric_limits<uint64_t>::digits;
constexpr uint32_t BATCH_INVARIANT_LEVEL = 3;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;

template <typename T>
T RegbaseCeilDiv(T x, T y)
{
    // Keep the overflow-safe variant local to regbase tiling.
    return y == 0 ? x : x / y + static_cast<T>(x % y != 0);
}

bool IsTransposeCapable(uint64_t numRow, uint64_t numCol, uint32_t curElementByte, uint64_t ubBlockSize)
{
    // 这里只判断短 R 转置能力，A 向性能门槛由后续函数判断。
    if (numRow == 0 || numCol == 0 || curElementByte == 0 || ubBlockSize == 0) {
        return false;
    }
    uint64_t maxR = ubBlockSize / curElementByte;
    return maxR != 0 && numCol <= maxR;
}

bool IsPhysicalTransposePreferred(uint64_t blockFactor, uint64_t vlfp32)
{
    if (blockFactor == 0 || vlfp32 == 0 ||
        vlfp32 > std::numeric_limits<uint64_t>::max() / TRANS_PHYSICAL_LARGE_MIN_VL_PER_CORE) {
        return false;
    }
    uint64_t minRowsPerCore = vlfp32 * TRANS_PHYSICAL_LARGE_MIN_VL_PER_CORE;
    return blockFactor > minRowsPerCore;
}

void SaveTransTilingData(gert::TilingContext* context, uint32_t numCore, uint32_t usedCoreNum,
                         uint64_t targetRowsPerCore, uint64_t numRow, uint64_t numCol, uint64_t rAligned,
                         uint64_t rTileBase, uint64_t tileALen, uint64_t tileATail, uint64_t totalTiles,
                         uint64_t tilesPerCore, float epsilon, float avgFactor)
{
    RMSNormRegbaseTransTilingData tiling;
    tiling.set_numRow(numRow);
    tiling.set_numCol(numCol);
    tiling.set_rAligned(rAligned);
    tiling.set_rTileBase(rTileBase);
    tiling.set_tileALen(tileALen);
    tiling.set_epsilon(epsilon);
    tiling.set_avgFactor(avgFactor);
    context->SetBlockDim(usedCoreNum);
    OP_LOGI(context,
            "TransTiling numCore: %u, numRow: %lu, numCol: %lu, rAligned: %lu, rTileBase: %lu, "
            "tileALen: %lu, tileATail: %lu, totalTiles: %lu, tilesPerCore: %lu, usedCoreNum: %u, "
            "targetRowsPerCore: %lu",
            numCore, numRow, numCol, rAligned, rTileBase, tileALen, tileATail, totalTiles, tilesPerCore, usedCoreNum,
            targetRowsPerCore);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->SetTilingKey(RMSNORM_REGBASE_TRANSPOSE);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
}

ge::graphStatus TilingReduceEmpty(gert::TilingContext* context, uint64_t numRow, uint32_t numCore, uint64_t ubSize,
                                  uint64_t ubBlockSize, float epsilon)
{
    // 空归约仅需写 A 个 rstd，按每核至少 32 KiB 输出控制启动开销。
    constexpr uint64_t singleCoreMinBytes = 32UL * 1024UL;
    constexpr uint64_t floatBytes = sizeof(float);
    OP_CHECK_IF(numCore == 0, OP_LOGE(context, "The number of AIV cores cannot be zero."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(numRow > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE(context, "numRow exceeds the int64_t range."), return ge::GRAPH_FAILED);
    uint64_t elemsPerBlock = ubBlockSize / floatBytes;
    OP_CHECK_IF(elemsPerBlock == 0, OP_LOGE(context, "The UB block size is smaller than one float element."),
                return ge::GRAPH_FAILED);
    uint64_t rowsPerMinBlock = singleCoreMinBytes / floatBytes;
    uint64_t blockNum = std::min(RegbaseCeilDiv(numRow, rowsPerMinBlock), static_cast<uint64_t>(numCore));
    blockNum = std::max(blockNum, 1UL);
    uint64_t usedCoreNum = blockNum;
    uint64_t rowsPerCore = Ops::Base::CeilAlign(RegbaseCeilDiv(numRow, usedCoreNum), elemsPerBlock);
    usedCoreNum = std::max(RegbaseCeilDiv(numRow, rowsPerCore), 1UL);
    uint64_t rowsPerTailCore = numRow == 0 ? 0 : numRow - (usedCoreNum - 1) * rowsPerCore;
    uint64_t tailCoreStartIndex = usedCoreNum - 1;
    uint64_t perLoopMax = ubSize / floatBytes;
    uint64_t chunk = std::min(perLoopMax, rowsPerCore);
    // 常规循环按 UB 数据块对齐，最后一轮由 DataCopyPad 精确处理尾部。
    uint64_t rowsPerLoop = chunk / elemsPerBlock * elemsPerBlock;
    if (rowsPerLoop == 0) {
        rowsPerLoop = std::max(rowsPerCore, 1UL);
    }

    RMSNormRegbaseReduceEmptyTilingData tiling;
    tiling.set_numRow(numRow);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_rowsPerTailCore(rowsPerTailCore);
    tiling.set_tailCoreStartIndex(tailCoreStartIndex);
    tiling.set_rowsPerLoop(rowsPerLoop);
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    OP_LOGI(context,
            "TilingData(ReduceEmpty) numRow: %lu, usedCoreNum: %lu, rowsPerCore: %lu, rowsPerTailCore: %lu, "
            "tailCoreStartIndex: %lu, rowsPerLoop: %lu, epsilon: %f",
            numRow, usedCoreNum, rowsPerCore, rowsPerTailCore, tailCoreStartIndex, rowsPerLoop, epsilon);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->SetTilingKey(RMSNORM_REGBASE_REDUCE_EMPTY);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingTranspose(gert::TilingContext* context, uint64_t numRow, uint64_t numCol, uint32_t numCore,
                                uint64_t ubSize, uint32_t curElementByte, uint32_t gammaElementByte,
                                uint64_t ubBlockSize, uint64_t vlfp32, float epsilon, float avgFactor)
{
    // UB 中转为 [R,A]，GM 搬运仍使用真实 R。
    uint64_t rTileBase = ubBlockSize / curElementByte;
    uint64_t aTileAlign = ubBlockSize / sizeof(uint16_t);
    if (rTileBase == 0 || aTileAlign == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t rAligned = RegbaseCeilDiv(numCol, rTileBase) * rTileBase;
    // 按 Kernel buffer 清单反推单个 A tile 上限。
    uint64_t fixedBufSize = rAligned * gammaElementByte; // gammaQueue，类型转换在计算 VF 内完成。
    uint64_t perARowBufSize = rAligned * (curElementByte * DOUBLE_BUFFER_NUM * MULTI_FACTOR_2 + FLOAT_BYTE_SIZE) +
                              FLOAT_BYTE_SIZE * DOUBLE_BUFFER_NUM;
    if (ubSize <= fixedBufSize) {
        return ge::GRAPH_FAILED;
    }
    uint64_t maxTileALen = (ubSize - fixedBufSize) / perARowBufSize;
    maxTileALen = maxTileALen / aTileAlign * aTileAlign;
    // repeatTimes 为 uint8_t，超出部分由 Kernel 外层循环处理。
    maxTileALen = std::min(maxTileALen, static_cast<uint64_t>(std::numeric_limits<uint8_t>::max()) * aTileAlign);
    if (maxTileALen < aTileAlign) {
        return ge::GRAPH_FAILED;
    }

    // 每核约分配两个 FP32 VL 的 A 行。
    uint64_t targetRowsPerCore = vlfp32 * MULTI_FACTOR_2;
    uint64_t usedCoreNum64 = RegbaseCeilDiv(numRow, targetRowsPerCore);
    // numRow/numCore/vlfp32 均为正，无需下界兜底。
    usedCoreNum64 = std::min({usedCoreNum64, numRow, static_cast<uint64_t>(numCore)});
    uint32_t usedCoreNum = static_cast<uint32_t>(usedCoreNum64);
    uint64_t rowsPerCoreMax = RegbaseCeilDiv(numRow, static_cast<uint64_t>(usedCoreNum));
    // R=一个 B16 block 时保留至少两个 tile 的流水机会。
    uint64_t targetTilesPerCore = (numCol == aTileAlign && rowsPerCoreMax > targetRowsPerCore) ? MULTI_FACTOR_2 : 1;
    targetTilesPerCore = std::max(targetTilesPerCore, RegbaseCeilDiv(rowsPerCoreMax, maxTileALen));
    uint64_t tileALen = RegbaseCeilDiv(rowsPerCoreMax, targetTilesPerCore);
    tileALen = RegbaseCeilDiv(tileALen, aTileAlign) * aTileAlign;
    // 短 R 时保留至少两个 FP32 VL 的 A 轴展开空间。
    if (numCol < aTileAlign && tileALen <= vlfp32 && maxTileALen >= vlfp32 * MULTI_FACTOR_2) {
        tileALen = vlfp32 * MULTI_FACTOR_2;
    }
    tileALen = std::min(tileALen, maxTileALen);
    uint64_t aOuter = RegbaseCeilDiv(numRow, tileALen);
    uint64_t tileATail = numRow - tileALen * (aOuter - 1);
    uint64_t tilesPerCore = RegbaseCeilDiv(rowsPerCoreMax, tileALen);

    SaveTransTilingData(context, numCore, usedCoreNum, targetRowsPerCore, numRow, numCol, rAligned, rTileBase, tileALen,
                        tileATail, aOuter, tilesPerCore, epsilon, avgFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSplitAR(gert::TilingContext* context, uint64_t numRow, uint64_t numCol, uint32_t numCore,
                              uint64_t ubSize, uint32_t xBytes, uint32_t gammaBytes, uint64_t blockSize,
                              uint64_t vectorLength, float epsilon)
{
    const uint64_t maxRBlockNum = std::min(static_cast<uint64_t>(numCore), vectorLength);
    // x/y 的核间分块按 UB block 对齐；FP16/BF16 输入对齐时，FP32 gamma 也满足该对齐。
    const uint64_t dataPerBlock = blockSize / xBytes;
    const uint64_t numRowAlign = Ops::Base::CeilAlign(numRow, blockSize / sizeof(float));
    const uint64_t combElements = Ops::Base::CeilAlign(maxRBlockNum * numRowAlign, vectorLength);
    const uint64_t fixedBufSize = (numRowAlign * MULTI_FACTOR_2 + combElements + vectorLength) * sizeof(float) +
                                  blockSize * AR_CACHE_LEVEL_COUNT;
    if (numRow == 0 || maxRBlockNum < SPLIT_AR_MIN_PARALLEL_R_BLOCKS || ubSize <= fixedBufSize) {
        return ge::GRAPH_FAILED;
    }
    // x/gamma/y queues + FP32 input and reduction work buffers.
    const uint64_t perElementBytes = DOUBLE_BUFFER_NUM * (xBytes * MULTI_FACTOR_2 + gammaBytes) +
                                     sizeof(float) * MULTI_FACTOR_2;
    const uint64_t binaryAddMaxLen = vectorLength * vectorLength * MULTI_FACTOR_2 * MULTI_FACTOR_2;
    uint64_t ubFactorMax = std::min((ubSize - fixedBufSize) / perElementBytes, binaryAddMaxLen);
    ubFactorMax = ubFactorMax / dataPerBlock * dataPerBlock;
    if (ubFactorMax == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t ubFactor = 1ULL << (std::numeric_limits<uint64_t>::digits - 1 - __builtin_clzll(ubFactorMax));
    const uint64_t perCoreWorkMin = ubFactor * SPLIT_AR_MIN_DB_TILES_PER_CORE;
    const uint64_t rBlocksPerRowByWork = numCol / perCoreWorkMin;
    if (rBlocksPerRowByWork < SPLIT_AR_MIN_PARALLEL_R_BLOCKS) {
        return ge::GRAPH_FAILED;
    }
    uint64_t rBlockNum = maxRBlockNum;
    if (rBlocksPerRowByWork < RegbaseCeilDiv(maxRBlockNum, numRow)) {
        rBlockNum = rBlocksPerRowByWork * numRow;
    }
    uint64_t rBlockFactor = Ops::Base::CeilAlign(RegbaseCeilDiv(numCol, rBlockNum), dataPerBlock);
    // 多行场景按完整 UB tile 切 R，避免各行重复处理极短的尾 tile。
    // 单行保留原切分，避免新增尾核的同步开销超过减少尾 tile 的收益。
    const uint64_t tileAlignedFactor = rBlockFactor / ubFactor * ubFactor;
    if (numRow > 1 && tileAlignedFactor != 0 && RegbaseCeilDiv(numCol, tileAlignedFactor) <= maxRBlockNum) {
        rBlockFactor = tileAlignedFactor;
    }
    rBlockNum = RegbaseCeilDiv(numCol, rBlockFactor);
    if (rBlockNum < SPLIT_AR_MIN_PARALLEL_R_BLOCKS || rBlockNum > maxRBlockNum) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t tailR = numCol - (rBlockNum - 1) * rBlockFactor;
    const uint64_t userWorkspaceSize = rBlockNum * numRowAlign * sizeof(float);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const uint64_t systemWorkspaceSize = static_cast<uint64_t>(
        platform_ascendc::PlatformAscendC(platformInfo).GetLibApiWorkSpaceSize());
    OP_CHECK_IF(userWorkspaceSize > std::numeric_limits<size_t>::max() - systemWorkspaceSize,
                OP_LOGE(context, "The SplitAR workspace size overflows size_t."), return ge::GRAPH_FAILED);
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    OP_CHECK_IF(context->SetScheduleMode(BATCH_MODE_SCHEDULE) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Failed to set the SplitAR schedule mode."), return ge::GRAPH_FAILED);
    workspace[0] = systemWorkspaceSize + userWorkspaceSize;

    RMSNormRegbaseSplitARTilingData tiling;
    tiling.set_numRow(numRow);
    tiling.set_numCol(numCol);
    tiling.set_rBlockFactor(rBlockFactor);
    tiling.set_rBlockNum(rBlockNum);
    tiling.set_tailR(tailR);
    tiling.set_ubFactor(ubFactor);
    tiling.set_numRowAlign(numRowAlign);
    tiling.set_epsilon(epsilon);
    tiling.set_avgFactor(1.0f / static_cast<float>(numCol));
    OP_LOGI(context,
            "SplitAR numRow: %lu, numCol: %lu, rBlockNum: %lu, rBlockFactor: %lu, tailR: %lu, "
            "ubFactor: %lu, numRowAlign: %lu, userWorkspaceSize: %lu",
            numRow, numCol, rBlockNum, rBlockFactor, tailR, ubFactor, numRowAlign, userWorkspaceSize);
    context->SetBlockDim(static_cast<uint32_t>(rBlockNum));
    context->SetTilingKey(RMSNORM_REGBASE_SPLIT_AR);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus TilingArch354RmsNorm(gert::TilingContext* context, uint64_t numRow, uint64_t numCol, uint32_t numCore,
                                     uint64_t ubSize, ge::DataType xDataType, ge::DataType gammaDataType, float epsilon,
                                     RMSNormTilingData& rmsNormTilingData,
                                     RMSNormArch35TilingData& rmsNormArch35TilingData)
{
    OP_CHECK_IF(numCore == 0 || ubSize <= RETAINED_SIZE_1K,
                OP_LOGE(context, "Invalid AIV core count or insufficient UB size."), return ge::GRAPH_FAILED);
    const uint64_t blockSize = Ops::Base::GetUbBlockSize(context);
    const uint64_t vectorLength = Ops::Base::GetVRegSize(context) / FLOAT_BYTE_SIZE;
    OP_CHECK_IF(blockSize < sizeof(float) || vectorLength < MULTI_FACTOR_2,
                OP_LOGE(context, "Invalid UB block size or vector length."), return ge::GRAPH_FAILED);
    const bool isRmsNorm = std::string(context->GetNodeType()) == "RmsNorm";
    if (isRmsNorm) {
        // 仅 SplitAR 使用用户工作区，并在该分支申请系统预留及 partial sum 空间。
        size_t* workspace = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
        workspace[0] = 0;
    }
    if (isRmsNorm && (numRow == 0 || numCol == 0)) {
        return TilingReduceEmpty(context, numRow, numCore, ubSize, blockSize, epsilon);
    }
    auto* platformInfo = context->GetPlatformInfo();
    const bool isAscend950 = platformInfo != nullptr &&
                             platform_ascendc::PlatformAscendC(platformInfo).GetSocVersion() ==
                                 platform_ascendc::SocVersion::ASCEND950;
    const bool enableNewTemplates = isRmsNorm && isAscend950 &&
                                    context->GetDeterministicLevel() != BATCH_INVARIANT_LEVEL;
    const uint64_t blockFactor = RegbaseCeilDiv(numRow, static_cast<uint64_t>(numCore));
    const uint32_t useCoreNum = static_cast<uint32_t>(RegbaseCeilDiv(numRow, blockFactor));
    context->SetBlockDim(useCoreNum);

    auto dtypeByteIterator_x = dTypeByteMap.find(xDataType);
    OP_TILING_CHECK(dtypeByteIterator_x == dTypeByteMap.end(),
                    OPS_REPORT_VECTOR_INNER_ERR(context, "Fail to get dtype factor."), return ge::GRAPH_FAILED);
    uint32_t curElementByteX = dtypeByteIterator_x->second;
    auto dtypeByteIterator_gamma = dTypeByteMap.find(gammaDataType);
    OP_TILING_CHECK(dtypeByteIterator_gamma == dTypeByteMap.end(),
                    OPS_REPORT_VECTOR_INNER_ERR(context, "Fail to get dtype factor."), return ge::GRAPH_FAILED);
    uint32_t curElementByteGamma = dtypeByteIterator_gamma->second;

    auto curElementByte = std::max(curElementByteX, curElementByteGamma);
    OP_TILING_CHECK(curElementByte == 0, OPS_REPORT_VECTOR_INNER_ERR(context, "curElementByte is zero."),
                    return ge::GRAPH_FAILED);
    if (enableNewTemplates && IsTransposeCapable(numRow, numCol, curElementByteX, blockSize) &&
        IsPhysicalTransposePreferred(blockFactor, vectorLength)) {
        if (TilingTranspose(context, numRow, numCol, numCore, ubSize, curElementByteX, curElementByteGamma, blockSize,
                            vectorLength, epsilon, 1.0f / static_cast<float>(numCol)) == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
    }
    const uint64_t dataPerBlockX = blockSize / curElementByteX;
    const uint64_t numColAlignFullLoad = RegbaseCeilDiv(numCol, dataPerBlockX) * dataPerBlockX;
    const uint64_t fullLoadRMax = vectorLength * vectorLength * MULTI_FACTOR_2 * MULTI_FACTOR_2;
    uint64_t powerFactor = 0;
    int64_t ubFactor = 0;
    uint32_t tilingKey;
    // Only size a full row when it is within the full-load reduction limit.
    if (numColAlignFullLoad > 0 && numColAlignFullLoad <= fullLoadRMax) {
        powerFactor = 1ULL << (std::numeric_limits<uint64_t>::digits - 1 - __builtin_clzll(numColAlignFullLoad - 1));
        const uint64_t firstVcaddLength = RegbaseCeilDiv(RegbaseCeilDiv(powerFactor, vectorLength), blockSize) *
                                          blockSize;
        const uint64_t gammaRowBytes = numColAlignFullLoad * curElementByteGamma;
        if (ubSize > RETAINED_SIZE_1K + gammaRowBytes) {
            ubFactor = (ubSize - RETAINED_SIZE_1K - gammaRowBytes) /
                       (numColAlignFullLoad * curElementByteX * MULTI_FACTOR_2 * DOUBLE_BUFFER_NUM +
                        FLOAT_BYTE_SIZE * (DOUBLE_BUFFER_NUM + X_REDUCE_TMP_NUM) + firstVcaddLength);
        }
    }
    if (ubFactor >= 1 && numColAlignFullLoad <= fullLoadRMax) {
        float avgFactor = (numCol == 0ULL) ? 0.0f : 1.0f / static_cast<float>(numCol);
        uint64_t colFlodFactor = powerFactor; // 二分累加折叠点
        uint64_t lastBlockFactor = numRow - (useCoreNum - 1) * blockFactor;
        ubFactor = std::min(ubFactor, static_cast<int64_t>(blockFactor));
        tilingKey = RMSNORM_REGBASE_NORMAL;
        rmsNormArch35TilingData.set_num_row(numRow);
        rmsNormArch35TilingData.set_num_col(numCol);
        rmsNormArch35TilingData.set_num_col_align(numColAlignFullLoad);
        rmsNormArch35TilingData.set_block_factor(blockFactor);
        rmsNormArch35TilingData.set_col_flod_factor(colFlodFactor);
        rmsNormArch35TilingData.set_ub_factor(ubFactor);
        rmsNormArch35TilingData.set_epsilon(epsilon);
        rmsNormArch35TilingData.set_avg_factor(avgFactor);
        rmsNormArch35TilingData.set_last_block_factor(lastBlockFactor);
        OP_LOGD(
            context,
            "TilingData[%u] numCore: %u, ubSize: %lu, numRow: %lu, numCol: %lu, numColAlignFullLoad: %lu "
            "blockFactor: %lu, colFlodFactor: %lu, ubFactor: %ld, epsilon: %f, avgFactor: %f, lastBlockFactor: %lu. ",
            tilingKey, useCoreNum, ubSize, numRow, numCol, numColAlignFullLoad, blockFactor, colFlodFactor, ubFactor,
            epsilon, avgFactor, lastBlockFactor);
        context->SetTilingKey(tilingKey);
        rmsNormArch35TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                                             context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(rmsNormArch35TilingData.GetDataSize());
    } else {
        // 限制 A 的行数，保证每行有足够的 R 向并行工作量。
        const uint64_t maxSplitARRows = std::min(SPLIT_AR_MAX_A_ROWS,
                                                 static_cast<uint64_t>(numCore) / SPLIT_AR_MIN_PARALLEL_R_BLOCKS);
        if (enableNewTemplates && numRow > 0 && numRow <= maxSplitARRows &&
            TilingSplitAR(context, numRow, numCol, numCore, ubSize, curElementByteX, curElementByteGamma, blockSize,
                          vectorLength, epsilon) == ge::GRAPH_SUCCESS) {
            return ge::GRAPH_SUCCESS;
        }
        OP_CHECK_IF(numCol > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(context, "The split-D reduction length exceeds uint32_t."), return ge::GRAPH_FAILED);
        uint64_t rowFactor = vectorLength;
        uint64_t numColAlign = RegbaseCeilDiv(numCol * curElementByte, static_cast<uint64_t>(ALING_FACTOR_512)) *
                               ALING_FACTOR_512 / curElementByte;
        OP_CHECK_IF(numColAlign > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(context, "The aligned split-D reduction length exceeds uint32_t."),
                    return ge::GRAPH_FAILED);
        ComputeTotalBufSizeParam computeTotalBufSizeParam{.bufferNum = DOUBLE_BUFFER_NUM,
                                                          .dtype = xDataType,
                                                          .dtypeSizeX = curElementByteX,
                                                          .dtypeSizeGamma = curElementByteGamma,
                                                          .length = static_cast<uint32_t>(numColAlign),
                                                          .split = false};
        uint32_t total = ComputeTotalBufSize(computeTotalBufSizeParam);
        uint32_t multiNNum = static_cast<uint32_t>(ubSize) / static_cast<uint32_t>(total);
        uint64_t colBufferLength{0};
        uint64_t ubLoop{0};
        uint32_t isNddma = 1;
        ubFactor = 1UL;
        computeTotalBufSizeParam.length = static_cast<uint32_t>(ubFactor) * MULTI_FACTOR_2;
        computeTotalBufSizeParam.split = true;
        while (ComputeTotalBufSize(computeTotalBufSizeParam) < ubSize) {
            ubFactor *= MULTI_FACTOR_2;
            computeTotalBufSizeParam.length = static_cast<uint32_t>(ubFactor) * MULTI_FACTOR_2;
        }
        ubLoop = 1UL;
        while (ubLoop * MULTI_FACTOR_2 * ubFactor <= numCol) {
            ubLoop *= MULTI_FACTOR_2;
        }
        colBufferLength = ubFactor;
        tilingKey = RMSNORM_REGBASE_SPLIT;
        if (numCol >= NDDMA_BETTER_STAGE) {
            isNddma = 0U;
        }
        rmsNormTilingData.set_num_row(numRow);
        rmsNormTilingData.set_num_col(numCol);
        rmsNormTilingData.set_num_col_align(numColAlign);
        rmsNormTilingData.set_block_factor(blockFactor);
        rmsNormTilingData.set_last_block_factor(0);
        rmsNormTilingData.set_row_factor(rowFactor);
        rmsNormTilingData.set_ub_factor(ubFactor);
        rmsNormTilingData.set_epsilon(epsilon);
        rmsNormTilingData.set_ub_loop(ubLoop);
        rmsNormTilingData.set_col_buffer_length(colBufferLength);
        rmsNormTilingData.set_multi_n_num(multiNNum);
        rmsNormTilingData.set_is_nddma(isNddma);

        OP_LOGD(context,
                "TilingData[%u] numCore: %u, ubSize: %lu, numRow: %lu, numCol: %lu, numColAlign: %lu, "
                "col_buffer_length: %u, "
                "blockFactor: %lu, rowFactor: %u, ubFactor: %u, epsilon: %f, ubLoop: %u, multi_n_num: %u, isNddma: %u.",
                tilingKey, numCore, ubSize, rmsNormTilingData.get_num_row(), rmsNormTilingData.get_num_col(),
                rmsNormTilingData.get_num_col_align(), rmsNormTilingData.get_col_buffer_length(),
                rmsNormTilingData.get_block_factor(), rmsNormTilingData.get_row_factor(),
                rmsNormTilingData.get_ub_factor(), rmsNormTilingData.get_epsilon(), rmsNormTilingData.get_ub_loop(),
                rmsNormTilingData.get_multi_n_num(), rmsNormTilingData.get_is_nddma());

        context->SetTilingKey(tilingKey);
        rmsNormTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                                       context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(rmsNormTilingData.GetDataSize());
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
