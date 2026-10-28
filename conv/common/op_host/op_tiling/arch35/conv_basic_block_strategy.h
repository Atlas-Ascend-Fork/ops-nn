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
 * \file conv_basic_block_strategy.h
 * \brief
 */

#ifndef CONV_BASIC_BLOCK_STRATEGY_H
#define CONV_BASIC_BLOCK_STRATEGY_H

#include <cstdint>
#include "conv_api_tiling_base.h"
#include "conv_base_numblocks_decision.h"

namespace conv_tiling {

using optiling::conv_ops_tiling::ConvAlignB;
using optiling::conv_ops_tiling::ConvAscendcFeatureFlag;
using optiling::conv_ops_tiling::ConvCeilDiv;
using optiling::conv_ops_tiling::ConvInferHiL1;
using optiling::conv_ops_tiling::CUBE_MKN_MAP;
using optiling::conv_ops_tiling::FixpipeInfo;
using optiling::conv_ops_tiling::MKN_N_IDX;

constexpr uint32_t COUT_LIMIT_128 = 128;

struct BBCommonParams {
    const ConvAscendcShapesInfo& shapeInfo;
    const ConvAscendcDescInfo& descInfo;
    const ConvAscendcAttrInfo& attrInfo;
    const ConvOpsConstParams& convOpsConstParams;
    const ConvAscendcTilingFlag& flagInfo;
    ConvAscendcFeatureFlag featureFlagInfo;
    const PlatformInfo& platformInfo;
    uint32_t aicoreNum;
    uint64_t l1Size;
    uint64_t l0cSize;
    const ConvOriGroupInfo& oriGroupInfo;
    const ConvOptGroupInfo& optGroupInfo;
    const FixpipeInfo& fixpipeInfo;
    ConvTilingBase* tilingBase;
    mutable bool enableInnerBatch;
    mutable uint32_t innerBatch;
};

class ConvBasicBlockStrategy {
public:
    virtual ~ConvBasicBlockStrategy() = default;

    virtual bool SupportsInnerBatch() const { return false; }

    virtual uint32_t GetDoCut(const BBCommonParams& params) const = 0;

    virtual uint64_t GetFwCutForGroup(const BBCommonParams& params, const BasicBlockInfo& bbInfo) const = 0;
    virtual uint64_t GetFwCut(const BBCommonParams& params, const BasicBlockInfo& bbInfo) const
    {
        return params.shapeInfo.batch * bbInfo.mCut;
    }

    virtual uint64_t CalcInputAI(const BBCommonParams& params, const BasicBlockInfo& bbInfo) const = 0;

    virtual uint64_t CalcOutputAI(const BBCommonParams& params) const = 0;

    virtual void CalcL0InitParams(const BBCommonParams& params, BasicBlockInfo& bbInfo,
                                  uint64_t availableL1Size) const = 0;

    void CalcL0InitParamsCommon(const BBCommonParams& params, BasicBlockInfo& bbInfo, uint64_t availableL1Size) const
    {
        uint64_t orgCo = params.flagInfo.convGroupType != ConvGroupType::NORMAL_CONV ?
                             (params.flagInfo.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
                                  params.oriGroupInfo.coPerGroup :
                                  params.optGroupInfo.coutOpt) :
                             params.shapeInfo.co;
        uint64_t biasSize = 0;
        if (params.flagInfo.hasBias) {
            biasSize = dtypeSizeTab.at(params.descInfo.biasDtype) * orgCo;
        }
        uint64_t scaleSize = static_cast<uint64_t>(params.fixpipeInfo.channelWiseCoeff * FP16_DTYPE_SIZE * orgCo);
        uint64_t availL1 = availableL1Size - biasSize - scaleSize;

        uint64_t hoAL1 = ConvCeilDiv(bbInfo.mTile, params.shapeInfo.wo) + CONST_VALUE_2;
        uint64_t hiAL1 = ConvInferHiL1(hoAL1, params.shapeInfo.hi, params.shapeInfo.kh, params.attrInfo.dilationH,
                                       params.attrInfo.strideH);
        bbInfo.mIn = hiAL1 * params.shapeInfo.wi;
        bbInfo.nKernelHKernelW = bbInfo.nTile * params.shapeInfo.kh * params.shapeInfo.kw;
        bbInfo.cinL1 = availL1 / dtypeSizeTab.at(params.descInfo.fMapDtype) / CONST_VALUE_2 /
                       (bbInfo.mIn * params.innerBatch + params.shapeInfo.kh * params.shapeInfo.kw * bbInfo.nTile) /
                       params.convOpsConstParams.k0 * params.convOpsConstParams.k0;
        bbInfo.cinL1 = bbInfo.cinL1 < params.convOpsConstParams.k0 ? params.convOpsConstParams.k0 : bbInfo.cinL1;
        bbInfo.mCut = ConvCeilDiv(params.shapeInfo.wo * params.shapeInfo.ho, bbInfo.mTile);
        bbInfo.nCut = ConvCeilDiv(orgCo, bbInfo.nTile);
        bbInfo.batch = params.shapeInfo.batch;
        bbInfo.aicoreNum = params.aicoreNum;
    }

    virtual void ExecuteBatchMDimDecision(const BBCommonParams& params, BasicBlockInfo& bbInfo) const = 0;

    virtual bool ShouldAdjustMnTile(const BBCommonParams& params, const BasicBlockInfo& bbInfo) const
    {
        if (bbInfo.mCut * params.shapeInfo.batch >= params.aicoreNum && bbInfo.nTile <= COUT_LIMIT_128 &&
            params.attrInfo.groups == 1 && params.shapeInfo.kh == 1 && params.shapeInfo.kw == 1 && bbInfo.nDim > 1) {
            return false;
        }
        return true;
    }

    virtual bool ShouldSkipBasicBlock(const BBCommonParams& params) const
    {
        return params.shapeInfo.hi <= CONST_VALUE_2;
    }

    virtual bool ShouldUseBasicBlockMN(const BBCommonParams& params, uint64_t basicBlockM, uint64_t basicBlockN) const
    {
        uint64_t orgCo = params.flagInfo.convGroupType != ConvGroupType::NORMAL_CONV ?
                             (params.flagInfo.convGroupType == ConvGroupType::ORI_GROUP_CONV ?
                                  params.oriGroupInfo.coPerGroup :
                                  params.optGroupInfo.coutOpt) :
                             params.shapeInfo.co;
        uint64_t group = params.attrInfo.groups;
        if (params.flagInfo.convGroupType == ConvGroupType::OPT_GROUP_CONV) {
            group = params.optGroupInfo.groupOpt;
        }
        uint64_t mCut = ConvCeilDiv(params.shapeInfo.wo * params.shapeInfo.ho, basicBlockM);
        uint64_t nCut = ConvCeilDiv(orgCo, basicBlockN);
        return (mCut * nCut * params.shapeInfo.batch * params.shapeInfo.dout * group > params.aicoreNum);
    }

    virtual void SetCandidateSingleOutputShape(const BBCommonParams& params, BasicBlockInfo& bbInfo,
                                               uint32_t nDim) const = 0;

    virtual uint64_t CalcBasicBlockNum1(const BBCommonParams& params, const BasicBlockInfo& bbInfo) const
    {
        if (!params.enableInnerBatch) {
            return ConvCeilDiv(params.shapeInfo.batch * bbInfo.mCut, bbInfo.fDim);
        }
        return ConvCeilDiv(ConvCeilDiv(ConvCeilDiv(params.shapeInfo.batch *
                                                       ConvCeilDiv(params.shapeInfo.ho * params.shapeInfo.wo,
                                                                   params.convOpsConstParams.m0) *
                                                       params.convOpsConstParams.m0,
                                                   bbInfo.mTile),
                                       params.innerBatch),
                           bbInfo.fDim);
    }
};

} // namespace conv_tiling

#endif
