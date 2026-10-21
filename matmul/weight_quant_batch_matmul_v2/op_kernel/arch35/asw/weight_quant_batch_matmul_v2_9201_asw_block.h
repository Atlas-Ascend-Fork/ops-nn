/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* !
 * \file weight_quant_batch_matmul_v2_9201_asw_block.h
 * \brief
 */
#pragma once

#include "../tool_arch35.h"
#include "basic_api/kernel_basic_intf.h"
#include "lib/matmul_intf.h"
#include "weight_quant_batch_matmul_v2_adaptive_sliding_window_common.h"

namespace WeightQuantBatchMatmulV2::Arch35 {
struct ASWTilingParam {
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t mCnt;     // M 方向的基本块数
    uint64_t nCnt;     // N 方向的基本块数
    uint64_t totalCnt; // 基本块总数, totalCnt = mCnt * nCnt
    uint64_t mCoreNum; // 单个滑动窗口 M 方向长度，单位：基本块个数
    uint64_t mTailCoreNum;
    uint64_t mBaseTail; // M 方向最后一个基本块（尾块）大小，单位：元素个数
    uint64_t nBaseTail; // N 方向最后一个基本块（尾块）大小，单位：元素个数
    uint64_t mIndex;    // 基本块相对于 [0, 1, ..., mCnt - 1]（即 M 方向）的索引号
    uint64_t nIndex;    // 基本块相对于 [0, 1, ..., nCnt - 1]（即 M 方向）的索引号
    uint64_t mSplitAddrOffset;
    uint64_t nSplitAddrOffset;
    uint64_t totalTailTile;
    uint64_t mainRow;
    uint64_t round; // 滑动窗口轮数
    uint64_t index; // 基本块相对于 [0, 1, ..., totalCnt - 1] 的索引号
    uint64_t kSize; // 原始矩阵k的大小
};

struct ASWOffsetParam {
    uint64_t offsetM;
    uint64_t offsetN;
    uint64_t offsetA;
    uint64_t offsetB;
    uint64_t offsetC;
    uint64_t offsetScale;
    uint64_t scaleScalar;
    uint64_t offsetBias;
};

class WeightQuantBmmAswBlock {
public:
    __aicore__ inline WeightQuantBmmAswBlock() {}
    __aicore__ inline void Init(const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData* tilingData,
                                uint32_t blockIdx);
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx);
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx);
    template <bool aTrans, bool bTrans, typename wType, CubeFormat formatX2 = CubeFormat::ND, bool scaleTrans = false,
              QuantType antiQuantType = QuantType::PER_CHANNEL>
    __aicore__ inline void CalcGMOffset();
    __aicore__ inline void ResetAddressOffsets();

public:
    ASWTilingParam params_;
    ASWOffsetParam offset_;
    const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData* tilingData_;
    bool evadeDuplicateBlock_;

private:
    const uint64_t WINDOW_LEN = 4;
    const uint64_t MX_GROUP_SIZE = 32;
    uint32_t blockIdx_;
};

__aicore__ inline void WeightQuantBmmAswBlock::Init(
    const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData* tilingData, uint32_t blockIdx)
{
    params_.mSplitAddrOffset = 0;
    params_.nSplitAddrOffset = 0;
    blockIdx_ = blockIdx;
    tilingData_ = tilingData;
    params_.kSize = tilingData_->matmulTiling.Ka;
    evadeDuplicateBlock_ = static_cast<bool>(tilingData_->evadeDuplicateBlock);
    params_.mCnt = CeilDiv(static_cast<uint64_t>(tilingData_->matmulTiling.M),
                           static_cast<uint64_t>(tilingData_->matmulTiling.baseM));
    if (evadeDuplicateBlock_) {
        params_.nCnt = CeilDiv(static_cast<uint64_t>(tilingData_->matmulTiling.N -
                                                     tilingData_->matmulTiling.baseN * tilingData_->mainTileNum),
                               static_cast<uint64_t>(tilingData_->lastRoundBaseN)) +
                       tilingData_->mainTileNum;
    } else {
        params_.nCnt = CeilDiv(static_cast<uint64_t>(tilingData_->matmulTiling.N),
                               static_cast<uint64_t>(tilingData_->matmulTiling.baseN));
    }
    params_.totalCnt = params_.mCnt * params_.nCnt;
    params_.mBaseTail = tilingData_->matmulTiling.M - (params_.mCnt - 1) * tilingData_->matmulTiling.baseM;
    params_.nBaseTail = tilingData_->matmulTiling.N - (params_.nCnt - 1) * tilingData_->matmulTiling.baseN;
    params_.totalTailTile = tilingData_->mTailTile * tilingData_->nTailTile;
    params_.round = CeilDiv(params_.totalCnt, static_cast<uint64_t>(tilingData_->matmulTiling.usedCoreNum));
    params_.mCoreNum = Min(WINDOW_LEN, params_.mCnt);
    params_.mainRow = params_.mCnt / params_.mCoreNum - 1;
    params_.mTailCoreNum = params_.mCnt - params_.mCoreNum * params_.mainRow;
}

__aicore__ inline void WeightQuantBmmAswBlock::UpdateBasicIndex(uint64_t roundIdx)
{
    uint64_t newBlockIdx = (roundIdx == params_.round - 1) ? (blockIdx_ / params_.totalTailTile) : blockIdx_;
    params_.index = newBlockIdx + roundIdx * tilingData_->matmulTiling.usedCoreNum;
    uint64_t rowIdx = params_.index / params_.nCnt / params_.mCoreNum;
    if (rowIdx < params_.mainRow) {
        params_.mIndex = rowIdx * params_.mCoreNum + params_.index % params_.mCoreNum;
        params_.nIndex = (params_.index / params_.mCoreNum) % params_.nCnt;
    } else {
        rowIdx = params_.mainRow;
        uint64_t tailIndex = params_.index - params_.mainRow * params_.mCoreNum * params_.nCnt;
        params_.mIndex = params_.mainRow * params_.mCoreNum + tailIndex % params_.mTailCoreNum;
        params_.nIndex = (tailIndex / params_.mTailCoreNum) % params_.nCnt;
    }

    if (rowIdx & 1) {
        params_.nIndex = params_.nCnt - 1 - params_.nIndex;
    }
}

__aicore__ inline void WeightQuantBmmAswBlock::UpdateBlockParams(uint64_t roundIdx)
{
    params_.singleCoreM = params_.mIndex != (params_.mCnt - 1) ? tilingData_->matmulTiling.baseM : params_.mBaseTail;
    if (evadeDuplicateBlock_) {
        if (params_.nIndex < tilingData_->mainTileNum) {
            params_.singleCoreN = tilingData_->matmulTiling.baseN;
        } else if (params_.nIndex < params_.nCnt - 1) {
            params_.singleCoreN = tilingData_->lastRoundBaseN;
        } else {
            params_.singleCoreN = tilingData_->lastRoundBaseNTail;
        }
    } else {
        params_.singleCoreN = params_.nIndex != (params_.nCnt - 1) ? tilingData_->matmulTiling.baseN :
                                                                     params_.nBaseTail;
    }
    if (tilingData_->mTailTile == 1 && tilingData_->nTailTile == 1) {
        return;
    }
    if (roundIdx == params_.round - 1) {
        uint64_t singleCoreMSplit = (params_.singleCoreM + tilingData_->mTailTile - 1) / tilingData_->mTailTile;
        uint64_t singleCoreNSplit = (params_.singleCoreN + tilingData_->nTailTile - 1) / tilingData_->nTailTile;
        uint64_t mSplitIdx = (blockIdx_ % params_.totalTailTile) % tilingData_->mTailTile;
        uint64_t nSplitIdx = (blockIdx_ % params_.totalTailTile) / tilingData_->mTailTile;
        params_.mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        params_.nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (params_.mSplitAddrOffset >= params_.singleCoreM || params_.nSplitAddrOffset >= params_.singleCoreN) {
            params_.singleCoreM = 0;
            params_.singleCoreN = 0;
            return;
        }
        if (params_.mSplitAddrOffset + singleCoreMSplit > params_.singleCoreM) {
            params_.singleCoreM = params_.singleCoreM - singleCoreMSplit * mSplitIdx;
        } else {
            params_.singleCoreM = singleCoreMSplit;
        }
        if (params_.nSplitAddrOffset + singleCoreNSplit > params_.singleCoreN) {
            params_.singleCoreN = params_.singleCoreN - singleCoreNSplit * nSplitIdx;
        } else {
            params_.singleCoreN = singleCoreNSplit;
        }
    }
}

__aicore__ inline void WeightQuantBmmAswBlock::ResetAddressOffsets()
{
    params_.mSplitAddrOffset = 0;
    params_.nSplitAddrOffset = 0;
}

template <bool aTrans, bool bTrans, typename wType, CubeFormat formatX2, bool scaleTrans, QuantType antiQuantType>
__aicore__ inline void WeightQuantBmmAswBlock::CalcGMOffset()
{
    offset_.offsetM = params_.mIndex * tilingData_->matmulTiling.baseM + params_.mSplitAddrOffset;
    if (evadeDuplicateBlock_) {
        if (params_.nIndex < tilingData_->mainTileNum) {
            offset_.offsetN = params_.nIndex * tilingData_->matmulTiling.baseN;
        } else {
            offset_.offsetN = tilingData_->mainTileNum * tilingData_->matmulTiling.baseN +
                              (params_.nIndex - tilingData_->mainTileNum) * tilingData_->lastRoundBaseN;
        }
    } else {
        offset_.offsetN = params_.nIndex * tilingData_->matmulTiling.baseN + params_.nSplitAddrOffset;
    }
    if constexpr (aTrans) {
        offset_.offsetA = offset_.offsetM;
    } else {
        offset_.offsetA = offset_.offsetM * tilingData_->matmulTiling.Ka;
    }

    if constexpr (bTrans) {
        if constexpr (formatX2 == CubeFormat::ND) {
            offset_.offsetB = offset_.offsetN * tilingData_->matmulTiling.Kb;
        } else {
            offset_.offsetB = offset_.offsetN * C0<wType>::value;
        }
    } else {
        offset_.offsetB = offset_.offsetN;
    }

    if constexpr (scaleTrans && antiQuantType == QuantType::MX) {
        offset_.offsetScale = offset_.offsetN * params_.kSize / MX_GROUP_SIZE;
    } else {
        offset_.offsetScale = offset_.offsetN;
    }

    offset_.offsetC = offset_.offsetM * tilingData_->matmulTiling.N + offset_.offsetN;
    offset_.offsetBias = offset_.offsetN;
}

} // namespace WeightQuantBatchMatmulV2::Arch35
