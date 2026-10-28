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
 * \file conv2d_basic_block_strategy.h
 * \brief
 */
#ifndef CONV2D_BASIC_BLOCK_STRATEGY_H
#define CONV2D_BASIC_BLOCK_STRATEGY_H

#include "conv_basic_block_strategy.h"
#include "conv_basic_block.h"
#include "conv_base_numblocks_decision.h"

namespace conv_tiling {

using optiling::conv_ops_tiling::IsDAV920Arch;

class Conv2DBasicBlockStrategy : public ConvBasicBlockStrategy {
public:
    bool SupportsInnerBatch() const override { return true; }

    uint32_t GetDoCut(const BBCommonParams& /* params */) const override { return 1; }

    uint64_t GetFwCutForGroup(const BBCommonParams& params, const BasicBlockInfo& bbInfo) const override
    {
        return params.shapeInfo.batch * ConvCeilDiv(params.shapeInfo.ho * params.shapeInfo.wo, bbInfo.mTile) *
               ConvCeilDiv(params.shapeInfo.co, bbInfo.nTile);
    }

    uint64_t CalcInputAI(const BBCommonParams& params, const BasicBlockInfo& bbInfo) const override
    {
        return CONST_VALUE_2 * params.shapeInfo.batch * params.shapeInfo.wo * params.shapeInfo.ho *
               params.shapeInfo.co * params.shapeInfo.kh * params.shapeInfo.kw /
               (params.shapeInfo.batch * params.shapeInfo.wi * params.shapeInfo.hi * bbInfo.nCut +
                params.shapeInfo.kh * params.shapeInfo.kw * params.shapeInfo.co * bbInfo.mCut) /
               dtypeSizeTab.at(params.descInfo.fMapDtype);
    }

    uint64_t CalcOutputAI(const BBCommonParams& params) const override
    {
        uint64_t dtypeSize = dtypeSizeTab.at(params.descInfo.outDtype);
        uint64_t ci = params.shapeInfo.ci;
        if (IsDAV920Arch(params.platformInfo.npuArch)) {
            uint64_t ciAlign = params.flagInfo.enableC04Flag ? C04_CIN_SIZE : C0_SIZE / dtypeSize;
            ci = ConvAlignB(ci, ciAlign);
        }
        return CONST_VALUE_2 * ci * params.shapeInfo.kh * params.shapeInfo.kw / dtypeSize;
    }

    void CalcL0InitParams(const BBCommonParams& params, BasicBlockInfo& bbInfo, uint64_t availableL1Size) const override
    {
        CalcL0InitParamsCommon(params, bbInfo, availableL1Size);
    }

    void ExecuteBatchMDimDecision(const BBCommonParams& params, BasicBlockInfo& bbInfo) const override
    {
        auto candidates = GenerateBatchMDimCandidates(params, bbInfo);
        uint32_t minCost = numeric_limits<uint32_t>::max();
        pair<uint32_t, uint32_t> optDecision;
        uint32_t mCut = bbInfo.mCut;
        uint32_t mCutAlignM0 = static_cast<uint32_t>(
            ConvCeilDiv(params.shapeInfo.ho * params.shapeInfo.wo, params.convOpsConstParams.m0));
        uint32_t mForCost = IsDAV920Arch(params.platformInfo.npuArch) ? mCutAlignM0 : mCut;
        uint32_t batchCut = ConvCeilDiv(params.shapeInfo.batch, params.innerBatch);
        for (const auto& candidate : candidates) {
            uint32_t mDim = candidate.first;
            uint32_t batchDim = candidate.second;
            uint32_t cost1 = ConvCeilDiv(batchCut, batchDim) * ConvCeilDiv(mForCost, mDim);
            float cost2 = 1.0f / static_cast<float>(batchDim);
            float cost = cost1 + cost2;
            if (cost < minCost) {
                minCost = static_cast<uint32_t>(cost);
                optDecision = candidate;
            }
        }
        bbInfo.mDim = optDecision.first;
        bbInfo.batchDim = optDecision.second;
    }

    void SetCandidateSingleOutputShape(const BBCommonParams& params, BasicBlockInfo& bbInfo,
                                       uint32_t nDim) const override
    {
        uint64_t curCo = params.flagInfo.convGroupType != ConvGroupType::NORMAL_CONV ?
                             (params.flagInfo.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
                                  params.oriGroupInfo.coPerGroup :
                                  params.optGroupInfo.coutOpt) :
                             params.shapeInfo.co;
        uint64_t n0 = static_cast<uint64_t>(CUBE_MKN_MAP.GetMKN(dtypeMap.at(params.descInfo.fMapDtype), MKN_N_IDX));
        uint64_t singleCoreCo = ConvAlignB(ConvCeilDiv(ConvAlignB(curCo, n0), nDim), n0);
        int64_t singleCoreBatch = static_cast<int64_t>(ConvCeilDiv(params.shapeInfo.batch, bbInfo.batchDim));
        params.tilingBase->SetSingleOutputShapeCoMBatch(static_cast<int64_t>(singleCoreCo), static_cast<int64_t>(0),
                                                        singleCoreBatch);
    }

private:
    unordered_set<pair<uint32_t, uint32_t>, BBPairHash> GenerateBatchMDimCandidates(const BBCommonParams& params,
                                                                                    const BasicBlockInfo& bbInfo) const
    {
        uint32_t cores = bbInfo.fDim;
        uint64_t mCut = static_cast<uint64_t>(bbInfo.mCut);
        uint32_t batchCut = ConvCeilDiv(params.shapeInfo.batch, params.innerBatch);
        uint32_t calCut1 = static_cast<uint32_t>(
            sqrt(static_cast<double>(cores * batchCut) / static_cast<double>(mCut)));
        uint32_t calCut2 = static_cast<uint32_t>(
            sqrt(static_cast<double>(static_cast<uint64_t>(cores) * mCut) / static_cast<double>(batchCut)));
        return ConvBasicBlock::GenerateCandidates(cores, mCut, batchCut, calCut1, calCut2, params.tilingBase->nodeType);
    }
};

} // namespace conv_tiling

#endif
