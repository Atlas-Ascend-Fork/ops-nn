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
 * \file conv2_dv2.h
 * \brief
 */

#ifndef CONV_2D_ASWT_H
#define CONV_2D_ASWT_H

#include "kernel_tiling/kernel_tiling.h"
#include "../../common/arch35/conv_common.h"
#include "conv2d_v2_api.h"

namespace ConvAWST {

using namespace AscendC;
using namespace conv2d;

constexpr static uint8_t TAIL_REGION0 = 0;
constexpr static uint8_t TAIL_REGION1 = 1;
constexpr static uint8_t TAIL_REGION2 = 2;

struct ConvAWSTParams {
    uint64_t fmStartAddr;
    uint64_t weightStartAddr;
    uint64_t outputStartAddr;
    uint64_t biasStartAddr;
    uint32_t singleBlockN;
    uint32_t singleBlockM;
    uint64_t curMOffset;
};

__aicore__ inline void CalCutsAndTailBlock(uint32_t oriLen, uint32_t baseBlock, uint32_t& cuts, uint32_t& tailBlock)
{
    cuts = CeilDiv(oriLen, baseBlock);
    tailBlock = oriLen % baseBlock;
    if (!tailBlock)
        tailBlock = baseBlock;
}

__aicore__ inline void CalcStartAddr(uint64_t curBatchOffset, uint64_t curNOffset, uint64_t orgM,
                                     const Ops::NN::Conv2dV2::Conv2DTilingData& convTilingData, ConvAWSTParams& params)
{
    params.fmStartAddr = curBatchOffset * convTilingData.cin * convTilingData.hin * convTilingData.win;
    params.weightStartAddr = curNOffset * convTilingData.cin * convTilingData.kh * convTilingData.kw;
    params.outputStartAddr = curBatchOffset * convTilingData.cout * orgM + curNOffset * orgM + params.curMOffset;
    params.biasStartAddr = curNOffset;
}

template <uint8_t regionType, int8_t IsASWT>
__aicore__ inline uint32_t CalcTailRegionMOffset(uint32_t allStandardCuts, uint32_t mWindowsAllCuts, uint32_t orgM,
                                                 uint16_t mWindows, uint16_t baseM)
{
    uint64_t curMoffset = (allStandardCuts / mWindowsAllCuts) * mWindows * baseM;
    if constexpr (regionType == TAIL_REGION1) {
        if constexpr (!(IsASWT & CONV_ASWT_TAIL_REGION_2)) {
            uint32_t mCuts = CeilDiv(orgM, baseM);
            mWindows = mCuts % mWindows == 0 ? mWindows : mCuts % mWindows;
        }
        curMoffset += ((allStandardCuts % mWindowsAllCuts) % mWindows) * baseM;
    } else if constexpr (regionType == TAIL_REGION2) {
        curMoffset = CeilDiv(allStandardCuts, mWindowsAllCuts) * mWindows * baseM;
    }
    return curMoffset;
}

template <uint8_t regionType>
__aicore__ inline uint32_t CalcTailRegionCurCuts(uint16_t curCoreIdx, const Ops::NN::Conv2dV2::Conv2DTilingData& info)
{
    if constexpr (regionType == TAIL_REGION0) {
        return curCoreIdx;
    } else if constexpr (regionType == TAIL_REGION1) {
        return curCoreIdx - info.tailRegionCuts[TAIL_REGION0];
    } else {
        return curCoreIdx - info.tailRegionCuts[TAIL_REGION0] - info.tailRegionCuts[TAIL_REGION1];
    }
}

template <uint8_t regionType>
__aicore__ inline uint8_t IsIllegalCore(uint16_t curCoreIdx, const Ops::NN::Conv2dV2::Conv2DTilingData& info)
{
    if constexpr (regionType == TAIL_REGION0) {
        return curCoreIdx >= info.tailRegionCuts[TAIL_REGION0];
    } else if constexpr (regionType == TAIL_REGION1) {
        return curCoreIdx < info.tailRegionCuts[TAIL_REGION0] ||
               curCoreIdx >= info.tailRegionCuts[TAIL_REGION0] + info.tailRegionCuts[TAIL_REGION1];
    } else {
        return curCoreIdx < info.tailRegionCuts[TAIL_REGION0] + info.tailRegionCuts[TAIL_REGION1] ||
               curCoreIdx >= info.tailRegionCuts[TAIL_REGION0] + info.tailRegionCuts[TAIL_REGION1] +
                                 info.tailRegionCuts[TAIL_REGION2];
    }
}

template <uint8_t regionType, int8_t IsASWT>
__aicore__ inline void CalcTailRegionParams(const Ops::NN::Conv2dV2::Conv2DTilingData& convTilingData, uint32_t orgM,
                                            uint32_t allStandardCuts, uint32_t mWindowsAllCuts, uint16_t baseM,
                                            bool nReverse, uint16_t curCoreIdx, ConvAWSTParams& params)
{
    uint32_t mCuts, nCuts, adaptTailBaseM, adaptTailBaseN;
    CalCutsAndTailBlock(convTilingData.tailRegionOrgM[regionType], convTilingData.adaptBaseM, mCuts, adaptTailBaseM);
    CalCutsAndTailBlock(convTilingData.tailRegionOrgN[regionType], convTilingData.adaptBaseN, nCuts, adaptTailBaseN);

    uint32_t mOffset = CalcTailRegionMOffset<regionType, IsASWT>(allStandardCuts, mWindowsAllCuts, orgM,
                                                                 convTilingData.mWindows, baseM);

    uint32_t curCuts = CalcTailRegionCurCuts<regionType>(curCoreIdx, convTilingData);
    uint32_t curBatchOffset = curCuts / (mCuts * nCuts);
    if (nReverse)
        curBatchOffset = convTilingData.batch - curBatchOffset - 1;

    uint32_t curN = (curCuts / mCuts) % nCuts;
    uint32_t curM = curCuts % mCuts;
    uint32_t curNOffset = curN * convTilingData.adaptBaseN;

    if (nReverse) {
        curNOffset = convTilingData.cout > (curN + 1) * convTilingData.adaptBaseN ?
                         convTilingData.cout - (curN + 1) * convTilingData.adaptBaseN :
                         0;
        if constexpr (regionType < TAIL_REGION2) {
            if (curCuts >= convTilingData.tailRegionCuts[regionType] - mCuts) {
                uint32_t tailNs = convTilingData.tailRegionBatchTailNs[regionType];
                uint32_t curNStart = curN * convTilingData.adaptBaseN;
                curNOffset = convTilingData.cout > curNStart + tailNs ? convTilingData.cout - curNStart - tailNs : 0;
            }
        }
    }

    params.curMOffset = mOffset + curM * convTilingData.adaptBaseM;
    params.singleBlockM = curM == mCuts - 1 ? adaptTailBaseM : convTilingData.adaptBaseM;
    params.singleBlockN = curN == nCuts - 1 ? adaptTailBaseN : convTilingData.adaptBaseN;

    if constexpr (regionType < TAIL_REGION2) {
        if (curCuts >= convTilingData.tailRegionCuts[regionType] - mCuts) {
            params.singleBlockN = convTilingData.tailRegionBatchTailNs[regionType];
        }
    }

    CalcStartAddr(curBatchOffset, curNOffset, orgM, convTilingData, params);
}

__aicore__ inline void CalcStandardRegionParams(uint32_t curCuts,
                                                const Ops::NN::Conv2dV2::Conv2DTilingData& convTilingData,
                                                uint32_t orgM, uint32_t orgN, uint16_t baseM, uint16_t baseN,
                                                uint32_t mCuts, uint32_t nCuts, uint16_t mWindows,
                                                uint32_t mWindowsAllCuts, bool nReverse, uint16_t tailBaseM,
                                                uint16_t tailBaseN, ConvAWSTParams& params)
{
    uint32_t preMWindowsOffset = (curCuts / mWindowsAllCuts) * mWindows;
    uint32_t curMWindows = min(mCuts - preMWindowsOffset, static_cast<uint32_t>(mWindows));

    uint32_t windowOffset = curCuts % mWindowsAllCuts;
    uint32_t windowMOffset = windowOffset % curMWindows;
    if (nReverse)
        windowOffset = curMWindows * convTilingData.batch * nCuts - windowOffset - 1;

    uint32_t curCutIndex = (windowOffset / curMWindows) * mCuts + windowMOffset + preMWindowsOffset;

    uint32_t curBatch = curCutIndex / (mCuts * nCuts);
    uint32_t curN = (curCutIndex / mCuts) % nCuts;
    uint32_t curM = curCutIndex % mCuts;
    uint32_t curNOffset = curN * baseN;

    params.singleBlockM = curM == mCuts - 1 ? tailBaseM : baseM;
    params.singleBlockN = curN == nCuts - 1 ? tailBaseN : baseN;
    params.curMOffset = curM * baseM;

    CalcStartAddr(curBatch, curNOffset, orgM, convTilingData, params);
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void ASWTConv2dKernelImpl(
    GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const Ops::NN::Conv2dV2::Conv2DTilingData& convTilingData,
    const ConvAWSTParams& params, Conv2d<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>& conv)
{
    using FMAP_T = typename FMAP_TYPE::T;
    using WEIGHT_T = typename WEIGHT_TYPE::T;
    using OUTPUT_T = typename OUTPUT_TYPE::T;
    using BIAS_T = typename BIAS_TYPE::T;
    GlobalTensor<FMAP_T> fmapGm;
    GlobalTensor<WEIGHT_T> filterGm;
    GlobalTensor<OUTPUT_T> outputGm;
    GlobalTensor<BIAS_T> biasGm;

    fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ FMAP_T*>(x + params.fmStartAddr * sizeof(FMAP_T)));
    filterGm.SetGlobalBuffer(reinterpret_cast<__gm__ WEIGHT_T*>(filter + params.weightStartAddr * sizeof(WEIGHT_T)));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ OUTPUT_T*>(y + params.outputStartAddr * sizeof(OUTPUT_T)));
    if (convTilingData.hasBias) {
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_T*>(bias + params.biasStartAddr * sizeof(BIAS_T)));
    }

    conv.SetSingleOutputShape(params.singleBlockN, params.singleBlockM, 1);
    conv.SetFmapStartPosition(params.curMOffset);
    conv.SetWeight(filterGm);
    if (convTilingData.hasBias) {
        conv.SetBias(biasGm);
    }
    conv.SetFmap(fmapGm);
    conv.IterateAll(outputGm);
    conv.End();
}

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG,
          int8_t IsASWT>
__aicore__ inline void ASWTRunConv2dKernel(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                           const Ops::NN::Conv2dV2::Conv2DTilingData& conv2dTilingData)
{
    Conv2d<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG> conv;
    const Ops::NN::Conv2dV2::Conv2DTilingData& convTilingData = conv2dTilingData;
    conv.Init(&convTilingData);

    uint16_t curCoreIdx = static_cast<uint16_t>(GetBlockIdx());
    uint16_t coreNum = static_cast<uint16_t>(GetBlockNum());

    uint32_t orgM = convTilingData.hout * convTilingData.wout;
    uint32_t orgN = convTilingData.cout;
    uint16_t baseN = convTilingData.nL0;
    uint16_t baseM = convTilingData.hoL0;
    uint16_t tailBaseN = orgN % baseN ? orgN % baseN : baseN;
    uint16_t tailBaseM = orgM % baseM ? orgM % baseM : baseM;
    uint32_t mCuts = CeilDiv(orgM, baseM);
    uint32_t nCuts = CeilDiv(orgN, baseN);
    uint16_t mWindows = convTilingData.mWindows;
    uint32_t allStandardCuts = ((convTilingData.batch * mCuts * nCuts) / coreNum) * coreNum;
    uint32_t mWindowsAllCuts = mWindows * nCuts * convTilingData.batch;

    bool nReverse = false;
    ConvAWSTParams params;

    uint32_t standardLoop = allStandardCuts / coreNum;
    for (uint32_t i = 0; i < standardLoop; i++) {
        uint32_t curCuts = i * coreNum + curCoreIdx;
        nReverse = (curCuts / mWindowsAllCuts) % 2 != 0;

        CalcStandardRegionParams(curCuts, convTilingData, orgM, orgN, baseM, baseN, mCuts, nCuts, mWindows,
                                 mWindowsAllCuts, nReverse, tailBaseM, tailBaseN, params);
        ASWTConv2dKernelImpl<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>(
            x, filter, bias, y, convTilingData, params, conv);
    }

    nReverse = (allStandardCuts / mWindowsAllCuts) % 2 == 0;

    if constexpr (IsASWT & CONV_ASWT_TAIL_REGION_0) {
        if (!IsIllegalCore<TAIL_REGION0>(curCoreIdx, convTilingData)) {
            CalcTailRegionParams<TAIL_REGION0, IsASWT>(convTilingData, orgM, allStandardCuts, mWindowsAllCuts, baseM,
                                                       nReverse, curCoreIdx, params);
            ASWTConv2dKernelImpl<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>(
                x, filter, bias, y, convTilingData, params, conv);
        }
    }
    if constexpr (IsASWT & CONV_ASWT_TAIL_REGION_1) {
        if (!IsIllegalCore<TAIL_REGION1>(curCoreIdx, convTilingData)) {
            CalcTailRegionParams<TAIL_REGION1, IsASWT>(convTilingData, orgM, allStandardCuts, mWindowsAllCuts, baseM,
                                                       nReverse, curCoreIdx, params);
            ASWTConv2dKernelImpl<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>(
                x, filter, bias, y, convTilingData, params, conv);
        }
    }
    if constexpr (IsASWT & CONV_ASWT_TAIL_REGION_2) {
        if (!IsIllegalCore<TAIL_REGION2>(curCoreIdx, convTilingData)) {
            CalcTailRegionParams<TAIL_REGION2, IsASWT>(convTilingData, orgM, allStandardCuts, mWindowsAllCuts, baseM,
                                                       nReverse, curCoreIdx, params);
            ASWTConv2dKernelImpl<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG>(
                x, filter, bias, y, convTilingData, params, conv);
        }
    }
}

#define CONV_ASWT_REGION_CASE(REGION)                                                                      \
    case REGION:                                                                                           \
        ASWTRunConv2dKernel<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG, REGION>( \
            x, filter, bias, y, convTilingData);                                                           \
        break;

template <class FMAP_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class SCALE_TYPE, class CONV_CFG>
__aicore__ inline void ASWTRunConv2dKernelByRegion(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                                   const Ops::NN::Conv2dV2::Conv2DTilingData& convTilingData)
{
    int8_t isAswt = CONV_ASWT_DISABLE;
    isAswt |= convTilingData.tailRegionCuts[TAIL_REGION0] != 0 ? CONV_ASWT_TAIL_REGION_0 : CONV_ASWT_DISABLE;
    isAswt |= convTilingData.tailRegionCuts[TAIL_REGION1] != 0 ? CONV_ASWT_TAIL_REGION_1 : CONV_ASWT_DISABLE;
    isAswt |= convTilingData.tailRegionCuts[TAIL_REGION2] != 0 ? CONV_ASWT_TAIL_REGION_2 : CONV_ASWT_DISABLE;
    switch (isAswt) {
        CONV_ASWT_REGION_CASE(CONV_ASWT_TAIL_REGION_0)
        CONV_ASWT_REGION_CASE(CONV_ASWT_TAIL_REGION_1)
        CONV_ASWT_REGION_CASE(CONV_ASWT_TAIL_REGION_2)
        CONV_ASWT_REGION_CASE(CONV_ASWT_TAIL_REGION_0_1)
        CONV_ASWT_REGION_CASE(CONV_ASWT_TAIL_REGION_1_2)
        default:
            ASWTRunConv2dKernel<FMAP_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, SCALE_TYPE, CONV_CFG,
                                CONV_ASWT_DISABLE>(x, filter, bias, y, convTilingData);
            break;
    }
}

#undef CONV_ASWT_REGION_CASE

} // namespace ConvAWST
#endif // CONV_2D_ASWT_H
