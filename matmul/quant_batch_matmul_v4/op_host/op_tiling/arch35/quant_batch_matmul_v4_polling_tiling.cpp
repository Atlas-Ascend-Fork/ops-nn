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
 * \file quant_batch_matmul_v4_polling_tiling.cpp
 * \brief
 */

#include "quant_batch_matmul_v4_polling_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

void QuantBatchMatmulV4PollingTiling::SetShape(const uint64_t mSize, const uint64_t nSize, const uint64_t kSize,
                                               const uint64_t groupSize)
{
    basicBlockParam_.mSize = mSize;
    basicBlockParam_.nSize = nSize;
    basicBlockParam_.kSize = kSize;
    basicBlockParam_.groupSize = groupSize;
    basicBlockParam_.singleM = 0;
    basicBlockParam_.singleN = 0;
    basicBlockParam_.mDim = 0;
    basicBlockParam_.nDim = 0;
    basicBlockParam_.baseM = 0;
    basicBlockParam_.baseN = 0;
    basicBlockParam_.baseK = 0;
}

void QuantBatchMatmulV4PollingTiling::SetDtypeBits(const uint64_t aDtypeBits, const uint64_t bDtypeBits,
                                                   const uint64_t x1ScaleDtypeBits, const uint64_t x2ScaleDtypeBits)
{
    basicBlockParam_.aDtypeBits = aDtypeBits;
    basicBlockParam_.bDtypeBits = bDtypeBits;
    basicBlockParam_.x2ScaleDtypeBits = x2ScaleDtypeBits;
    basicBlockParam_.x1ScaleDtypeBits = x1ScaleDtypeBits;
}

void QuantBatchMatmulV4PollingTiling::SetLimitParams(const uint64_t alignSizeN, const uint64_t maxBaseMSize,
                                                     const uint64_t maxBaseNSize, const uint64_t maxMNSize)
{
    limitParam_.alignSizeN = alignSizeN;
    limitParam_.maxBaseMSize = maxBaseMSize;
    limitParam_.maxBaseNSize = maxBaseNSize;
    limitParam_.maxMNSize = maxMNSize;
}

void QuantBatchMatmulV4PollingTiling::SetPlatformInfo(const PlatformInfo& platformInfos)
{
    platformInfos_ = platformInfos;
}

bool QuantBatchMatmulV4PollingTiling::GetPollingTilingSolution()
{
    optionalResults_.clear();
    CalculateBasicBlockTiling();
    if (!optionalResults_.empty()) {
        sort(optionalResults_.begin(), optionalResults_.end());
        basicBlockParam_ = optionalResults_.front();
        return true;
    }
    return false;
}

ge::graphStatus QuantBatchMatmulV4PollingTiling::ComputeTiling()
{
    OP_TILING_CHECK(!basicBlockParam_.isValid() || !platformInfos_.isValid(),
                    OP_LOGE("quant_batch_matmul_v4", "basicBlockParam or platformInfos not properly initialized"),
                    return ge::GRAPH_FAILED);

    if (!GetPollingTilingSolution()) {
        // 设置默认解
        basicBlockParam_.baseM = std::min(BASIC_BLOCK_SIZE_128, ops::CeilAlign(basicBlockParam_.mSize, CUBE_BLOCK));
        basicBlockParam_.baseN = std::min(BASIC_BLOCK_SIZE_128, ops::CeilAlign(basicBlockParam_.nSize, CUBE_BLOCK));
        basicBlockParam_.baseK = std::min(static_cast<uint64_t>(BASIC_BLOCK_SIZE_128), basicBlockParam_.groupSize);
        basicBlockParam_.mDim = 1;
        uint64_t mCount = ops::CeilDiv(basicBlockParam_.mSize, BASIC_BLOCK_SIZE_128);
        uint64_t nCount = ops::CeilDiv(basicBlockParam_.nSize, BASIC_BLOCK_SIZE_128);
        basicBlockParam_.nDim = std::min(mCount * nCount, static_cast<uint64_t>(platformInfos_.aicNum));
    }
    basicBlockParam_.enablePreAlloc = !ClosePreAllocCond();
    return ge::GRAPH_SUCCESS;
}

void QuantBatchMatmulV4PollingTiling::GetBasicBlockSolution()
{
    uint64_t baseMMax = std::min(limitParam_.maxBaseMSize, basicBlockParam_.singleM);
    for (uint64_t baseM = CUBE_BLOCK; baseM <= baseMMax; baseM += CUBE_BLOCK) {
        uint64_t baseNMax = std::min(limitParam_.maxBaseNSize,
                                     ops::CeilAlign(limitParam_.maxMNSize / baseM, limitParam_.alignSizeN));
        baseNMax = std::min(baseNMax, basicBlockParam_.singleN);
        for (uint64_t baseN = baseNMax; baseN >= limitParam_.alignSizeN; baseN -= limitParam_.alignSizeN) {
            if (baseN * baseM > limitParam_.maxMNSize) {
                continue;
            }
            basicBlockParam_.baseK = std::min(static_cast<uint64_t>(BASIC_BLOCK_SIZE_128), basicBlockParam_.groupSize);

            if (!CheckBufferSize(baseM, baseN)) {
                continue;
            }
            basicBlockParam_.baseM = baseM;
            basicBlockParam_.baseN = baseN;
            GetMte2Cost();
            optionalResults_.push_back(basicBlockParam_);
        }
    }
}

double QuantBatchMatmulV4PollingTiling::BWUtilization(const uint64_t mDim, const uint64_t nDim) const
{
    // 当前认为带宽利用率和实际使用的核数成正比
    return (mDim * nDim) / static_cast<double>(platformInfos_.aicNum);
}

void QuantBatchMatmulV4PollingTiling::GetMte2Cost()
{
    double mte2Cost = basicBlockParam_.mSize *
                          (basicBlockParam_.nDim * ops::CeilDiv(basicBlockParam_.singleN, basicBlockParam_.baseN) - 1) *
                          basicBlockParam_.aDtypeBits +
                      basicBlockParam_.mSize * basicBlockParam_.aDtypeBits * platformInfos_.l2HbmBWRatio +
                      basicBlockParam_.nSize *
                          (basicBlockParam_.mDim * ops::CeilDiv(basicBlockParam_.singleM, basicBlockParam_.baseM) - 1) *
                          basicBlockParam_.bDtypeBits +
                      basicBlockParam_.nSize * basicBlockParam_.bDtypeBits * platformInfos_.l2HbmBWRatio;
    // 降低总核数较少时被选中的概率
    basicBlockParam_.mte2Cost = mte2Cost / BYTE_BITS / BWUtilization(basicBlockParam_.mDim, basicBlockParam_.nDim);
}

bool QuantBatchMatmulV4PollingTiling::CheckBufferSize(const uint64_t baseM, const uint64_t baseN)
{
    uint64_t aL0DbDataSize = baseM * basicBlockParam_.baseK * basicBlockParam_.aDtypeBits / BYTE_BITS * DB_SIZE;
    uint64_t bL0DbDataSize = baseN * basicBlockParam_.baseK * basicBlockParam_.bDtypeBits / BYTE_BITS * DB_SIZE;
    if (aL0DbDataSize > platformInfos_.l0ASize || bL0DbDataSize > platformInfos_.l0BSize) {
        return false;
    }
    if (MAX_PERF_STEPKB * aL0DbDataSize + MAX_PERF_STEPKB * bL0DbDataSize > platformInfos_.l1Size) {
        return false;
    }
    // 第一个16: 4个bank的空间 * sizeof(float32/int32)
    // 最后一个2：sizeof(yType)
    // 空间组成：4个bank + pertokenScale + antiquantScale + out
    uint64_t confirmedUbSize = 16 * baseM * baseN + baseM * basicBlockParam_.x1ScaleDtypeBits / BYTE_BITS +
                               baseM * baseN * 2;
    uint64_t minDbScaleUbSize = baseN * MAX_PERF_STEPKB * basicBlockParam_.x2ScaleDtypeBits / BYTE_BITS *
                                DB_SIZE; // 最小只载2倍kL1的大小

    if (confirmedUbSize + minDbScaleUbSize > platformInfos_.ubSize) {
        return false;
    }
    // 计算剩余空间实际可以载入多少倍kL1的antiquatScale, 单次最多载入16K
    uint64_t freeUbSize = std::min(platformInfos_.ubSize - confirmedUbSize, MAX_SCALE_SIZE * DB_SIZE);
    uint64_t scaleFactor = freeUbSize / minDbScaleUbSize;
    basicBlockParam_.scaleFactor = std::min(
        std::max(ops::FloorDiv(scaleFactor, MAX_PERF_STEPKB), 1UL),
        ops::CeilDiv(basicBlockParam_.kSize / basicBlockParam_.groupSize, MAX_PERF_STEPKB));
    return true;
}

void QuantBatchMatmulV4PollingTiling::CalculateBasicBlockTiling()
{
    uint64_t mDimMax = std::min(ops::CeilDiv(basicBlockParam_.mSize, CUBE_BLOCK),
                                static_cast<uint64_t>(platformInfos_.aicNum));
    for (uint64_t mDim = mDimMax; mDim >= 1; mDim--) {
        uint64_t nDimMax = std::min(ops::CeilDiv(basicBlockParam_.nSize, CUBE_BLOCK), platformInfos_.aicNum / mDim);
        for (uint64_t nDim = nDimMax; nDim >= 1; nDim--) {
            if ((!optionalResults_.empty()) &&
                // 0.8：当已有可选解时剪枝分核情况小于0.8倍总核数的解
                mDim * nDim < 0.8 * platformInfos_.aicNum) {
                continue;
            }
            basicBlockParam_.mDim = mDim;
            basicBlockParam_.nDim = nDim;
            basicBlockParam_.singleM = ops::CeilAlign(ops::CeilDiv(basicBlockParam_.mSize, mDim), CUBE_BLOCK);
            basicBlockParam_.singleN = ops::CeilAlign(ops::CeilDiv(basicBlockParam_.nSize, nDim),
                                                      limitParam_.alignSizeN);
            uint64_t mDimFixed = ops::CeilDiv(basicBlockParam_.mSize, basicBlockParam_.singleM);
            uint64_t nDimFixed = ops::CeilDiv(basicBlockParam_.nSize, basicBlockParam_.singleN);
            if (mDimFixed != mDim || nDimFixed != nDim) {
                continue;
            }
            GetBasicBlockSolution();
        }
    }
}

bool QuantBatchMatmulV4PollingTiling::ClosePreAllocCond()
{
    int64_t actualKSize = std::min(basicBlockParam_.kSize, basicBlockParam_.baseK);
    int64_t actualMSize = std::min(basicBlockParam_.mSize, basicBlockParam_.baseM);
    int64_t actualNSize = std::min(basicBlockParam_.nSize, basicBlockParam_.baseN);
    int64_t aL1Size = actualMSize * actualKSize * MAX_PERF_STEPKB * basicBlockParam_.aDtypeBits / BYTE_BITS;
    int64_t bL1Size = actualNSize * actualKSize * MAX_PERF_STEPKB * basicBlockParam_.bDtypeBits / BYTE_BITS;
    // AL1 + BL1载入数据量较大(128 * 1024 B,经验值)时，开启prealloc会导致后2轮的mte2抢占前面的ostd,
    // 导致前面的带宽跟不上cube计算 当K较小时，单轮计算量较小， 如果不开启prealloc,
    // 下一轮MN的计算的MTE2无法提前发出，会导致mte2断流
    bool condition = aL1Size + bL1Size >= 128 * 1024 && basicBlockParam_.kSize > 1024;
    return condition;
}
} // namespace optiling
