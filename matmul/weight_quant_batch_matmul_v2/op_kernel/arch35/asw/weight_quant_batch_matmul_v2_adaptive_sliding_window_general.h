/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

/*!
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window_general.h
 * \brief
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "lib/matmul_intf.h"
#include "../../../common/arch35/weight_quant_batch_matmul_cube_compute_tools.h"
#include "../tool_arch35.h"
#include "../../weight_quant_batch_matmul_v2_constant.h"
#include "weight_quant_batch_matmul_v2_9201_asw_block.h"
#include "weight_quant_batch_matmul_v2_adaptive_sliding_window_common.h"

#ifndef LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
#define LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL                                                                       \
    template <typename xType, typename wType, typename scaleType, typename biasType, typename yType, bool bTrans, \
              QuantType antiQuantType, bool hasAntiQuantOffset, QuantType quantType, CubeFormat weightFormat,     \
              bool hasBias>
#endif
#ifndef LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL
#define LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL                                                                        \
    xType, wType, scaleType, biasType, yType, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat, \
        hasBias
#endif

using AscendC::AIC;
using AscendC::AIV;
using AscendC::BLOCK_CUBE;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

namespace MicroAPI = AscendC::MicroAPI;

namespace WeightQuantBatchMatmulV2::Arch35 {
template <typename xType, typename wType, typename scaleType>
struct ScaleQuantParam {
    uint16_t singleCoreN;
    uint16_t kCount;
    uint16_t singleCoreK;
    uint64_t kUbSize;
    uint64_t nUbSize;
    uint64_t groupKNum;
    scaleType scaleValue;
    scaleType offsetValue;

    __local_mem__ xType* weightF16Addr;
    __local_mem__ wType* weightAddr;
    __local_mem__ scaleType* scaleAddr;
    __local_mem__ scaleType* antiQuantOffsetAddr;
};

constexpr static uint16_t VL_F16_ELEMENTS = 128;
constexpr static uint16_t VL_WEIGHT_ELEMENTS = 64;

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
class WeightQuantBatchMatmulV2ASWGeneralKernel {
public:
    __aicore__ inline WeightQuantBatchMatmulV2ASWGeneralKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const void* tilingData);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitUbBuffs();
    __aicore__ inline void EndSync();
    __aicore__ inline void InitCubeBuffs();

    uint32_t blockIdx_;
    const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData* tiling_;

    GlobalTensor<xType> aGlobal_;
    GlobalTensor<wType> bGlobal_;
    GlobalTensor<yType> cGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<scaleType> antiquantScaleGlobal_;
    GlobalTensor<scaleType> antiquantOffsetGlobal_;

    WeightQuantBatchMatmulV2::Arch35::WeightQuantBmmAswBlock block_;

private:
    uint64_t sumLoopNum_ = 0;
    uint64_t totalLoop_ = 0;
    uint64_t roundDbFlag_ = 0;
    uint64_t l1DbFlag_ = 0;
    uint64_t l0DbFlag_ = 0;
    uint64_t kBlockSize_ = 0;
    uint64_t totalKBlocks_ = 0;
    uint64_t weightUbInElements_ = 0;
    uint64_t scaleElements_ = 0;
    uint64_t weightUbOutElements_ = 0;
    uint64_t kUbSize_ = 0;
    uint64_t nUbSize_ = 0;
    scaleType scaleValue_ = 0;
    scaleType offsetValue_ = 0;

    LocalTensor<xType> aL1PingLocalTensor_;
    LocalTensor<xType> aL1PongLocalTensor_;
    LocalTensor<xType> bL1PingLocalTensor_;
    LocalTensor<xType> bL1PongLocalTensor_;
    LocalTensor<biasType> biasPingLocalTensor_;
    LocalTensor<biasType> biasPongLocalTensor_;

    LocalTensor<int8_t> weightUbInTensor_;
    LocalTensor<xType> weightUbOutTensor_;
    LocalTensor<scaleType> scaleLocalTensor_;
    LocalTensor<scaleType> offsetLocalTensor_;

    LocalTensor<float> cL0Tensor_;
    LocalTensor<xType> aL0Tensor_;
    LocalTensor<xType> bL0Tensor_;
    LocalTensor<float> biasBtTensor_;

    __aicore__ inline void CopyBias2L1();
    __aicore__ inline void AntiquantCompute();
    __aicore__ inline void GetAToL1(uint64_t kOutIdx);
    __aicore__ inline void CopyWeightGm2UB(uint64_t kOutIdx);
    __aicore__ inline void CopyWeightUB2L1();
    __aicore__ inline void IterateMatmul(uint64_t kOutIdx, uint64_t kInnerIdx, uint64_t kOffset);
    __aicore__ inline void GetTensorC(const GlobalTensor<yType>& outTensor);
    __aicore__ inline void CopyScaleAndOffsetToUB(uint64_t kOutIdx);
    __aicore__ inline void MatmulCompute();
    __aicore__ inline void InitEventIds();
    __aicore__ inline void SetAivToAic();
    __aicore__ inline void WaitAivToAic();
    __aicore__ inline void SetAicToAiv();
    __aicore__ inline void WaitAicToAiv();
};

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, const void* tilingData)
{
    tiling_ = static_cast<const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData*>(tilingData);
    blockIdx_ = GetBlockIdx();
    UpdateGlobalAddr(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace);
    kBlockSize_ = tiling_->matmulTiling.stepKa * tiling_->matmulTiling.baseK;
    totalKBlocks_ = CeilDiv<uint64_t>(tiling_->matmulTiling.Ka, kBlockSize_);
    sumLoopNum_ = block_.params_.round * totalKBlocks_;
    InitCubeBuffs();
    InitEventIds();
    if ASCEND_IS_AIV {
        InitUbBuffs();
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::InitEventIds()
{
    if ASCEND_IS_AIV {
        for (int idx = 0; idx < DOUBLE_BUFFER; ++idx) {
            SetFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            SetFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
        }
    }
    if ASCEND_IS_AIC {
        for (int idx = 0; idx < DOUBLE_BUFFER; ++idx) {
            SetFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::InitCubeBuffs()
{
    uint64_t l1BufOffsetPart1 = 0;
    uint64_t l1BufOffsetPart2 = L1_BUFFER_HALF_SIZE;
    uint64_t bL1DataSizes = tiling_->matmulTiling.baseN * tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseK *
                            tiling_->matmulTiling.stepKb;
    uint64_t bL1DataBytes = bL1DataSizes * sizeof(xType);

    bL1PingLocalTensor_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart1, bL1DataSizes);
    l1BufOffsetPart1 += bL1DataBytes;
    bL1PongLocalTensor_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart2, bL1DataSizes);
    l1BufOffsetPart2 += bL1DataBytes;

    uint64_t aL1DataSize = tiling_->matmulTiling.baseM * tiling_->matmulTiling.stepM * tiling_->matmulTiling.baseK *
                           tiling_->matmulTiling.stepKa;

    aL1PingLocalTensor_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart1, aL1DataSize);
    l1BufOffsetPart1 += aL1DataSize * sizeof(xType);
    aL1PongLocalTensor_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart2, aL1DataSize);
    l1BufOffsetPart2 += aL1DataSize * sizeof(xType);
    if constexpr (hasBias) {
        biasPingLocalTensor_ = AscendC::LocalTensor<biasType>(AscendC::QuePosition::TSCM, l1BufOffsetPart1,
                                                              tiling_->matmulTiling.baseN);
        l1BufOffsetPart1 += tiling_->matmulTiling.baseN * sizeof(biasType);

        biasPongLocalTensor_ = AscendC::LocalTensor<biasType>(AscendC::QuePosition::TSCM, l1BufOffsetPart2,
                                                              tiling_->matmulTiling.baseN);
        l1BufOffsetPart2 += tiling_->matmulTiling.baseN * sizeof(biasType);
        biasBtTensor_ = AscendC::LocalTensor<float>(TPosition::C2, 0, BIAS_TABLE_SIZE / sizeof(float));
    }

    cL0Tensor_ = AscendC::LocalTensor<float>(TPosition::CO1, 0, L0C_BUFFER_SIZE / sizeof(float));
    aL0Tensor_ = AscendC::LocalTensor<xType>(TPosition::A2, 0, L0A_BUFFER_SIZE / sizeof(xType));
    // 特性：mmad运算之前均转为F16类型
    bL0Tensor_ = AscendC::LocalTensor<xType>(TPosition::B2, 0, L0B_BUFFER_SIZE / sizeof(xType));
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::InitUbBuffs()
{
    if constexpr (antiQuantType == QuantType::PER_GROUP) {
        scaleElements_ = tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseN *
                         CeilDiv<uint64_t>(tiling_->matmulTiling.baseK * tiling_->matmulTiling.stepKb,
                                           tiling_->groupSize);
        weightUbOutElements_ = tiling_->matmulTiling.baseN * tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseK *
                               tiling_->matmulTiling.stepKb;
    } else {
        // 解ub Bank冲突 n方向需要留出一行来
        weightUbOutElements_ = (tiling_->matmulTiling.baseN + 1) * tiling_->matmulTiling.stepN *
                               tiling_->matmulTiling.baseK * tiling_->matmulTiling.stepKb;
        if constexpr (antiQuantType == QuantType::PER_CHANNEL) {
            scaleElements_ = tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseN;
        }
    }
    weightUbOutTensor_ = AscendC::LocalTensor<xType>(AscendC::TPosition::VECIN, 0,
                                                     weightUbOutElements_ * DOUBLE_BUFFER);
    uint64_t weightOutSize = weightUbOutElements_ * sizeof(xType);
    scaleLocalTensor_ = AscendC::LocalTensor<scaleType>(AscendC::TPosition::VECIN, weightOutSize * DOUBLE_BUFFER,
                                                        scaleElements_ * DOUBLE_BUFFER);
    uint64_t scaleSize = scaleElements_ * sizeof(scaleType);
    if constexpr (hasAntiQuantOffset) {
        offsetLocalTensor_ = AscendC::LocalTensor<scaleType>(
            AscendC::TPosition::VECIN, (weightOutSize + scaleSize) * DOUBLE_BUFFER, scaleElements_ * DOUBLE_BUFFER);
        scaleSize += scaleElements_ * sizeof(scaleType);
    }
    weightUbInElements_ = tiling_->matmulTiling.baseN * tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseK *
                          tiling_->matmulTiling.stepKb;
    if constexpr (AscendC::IsSupportB4<wType>()) {
        weightUbInElements_ = weightUbInElements_ >> 1;
    }
    weightUbInTensor_ = AscendC::LocalTensor<int8_t>(
        AscendC::TPosition::VECIN, (weightOutSize + scaleSize) * DOUBLE_BUFFER, weightUbInElements_ * DOUBLE_BUFFER);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::UpdateGlobalAddr(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace)
{
    block_.Init(tiling_, blockIdx_);

    if constexpr (antiQuantType == QuantType::PER_TENSOR) { // pertensor
        scaleValue_ = *((__gm__ scaleType*)antiquantScale);
        if constexpr (hasAntiQuantOffset) {
            offsetValue_ = *((__gm__ scaleType*)antiquantOffset);
        }
    } else {
        antiquantScaleGlobal_.SetGlobalBuffer((__gm__ scaleType*)antiquantScale);
        if constexpr (hasAntiQuantOffset) {
            antiquantOffsetGlobal_.SetGlobalBuffer((__gm__ scaleType*)antiquantOffset);
        }
    }
    // update global buffer
    aGlobal_.SetGlobalBuffer((__gm__ xType*)x);
    bGlobal_.SetGlobalBuffer((__gm__ wType*)weight);
    cGlobal_.SetGlobalBuffer((__gm__ yType*)y);
    if constexpr (hasBias) {
        biasGlobal_.SetGlobalBuffer((__gm__ biasType*)bias);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::Process()
{
    for (uint64_t roundIndex = 0; roundIndex < block_.params_.round; ++roundIndex) {
        block_.UpdateBasicIndex(roundIndex);

        if (block_.evadeDuplicateBlock_ && block_.params_.index >= block_.params_.nCnt) {
            continue;
        }
        // 1. Set single core param
        block_.UpdateBlockParams(roundIndex);
        if (block_.params_.singleCoreM <= 0 || block_.params_.singleCoreN <= 0) {
            continue;
        }
        // 2. compute offset
        block_.template CalcGMOffset<false, bTrans, wType, weightFormat>();

        // 3. set offset and compute
        MatmulCompute();
        roundDbFlag_ ^= 1;
    }
    EndSync();
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::MatmulCompute()
{
    uint64_t kOffset = 0;
    for (uint64_t kOutIdx = 0; kOutIdx < totalKBlocks_; kOutIdx++) {
        if ASCEND_IS_AIV {
            WaitFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
            CopyWeightGm2UB(kOutIdx);
            CopyScaleAndOffsetToUB(kOutIdx);
            if (totalLoop_ > 1) {
                WaitAicToAiv();
            }
            AntiquantCompute();
            SetFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]); // VF结束 可以再搬
            SetFlag<HardEvent::V_MTE3>(MTE_SYNC_FLAG_ID_6);
            WaitFlag<HardEvent::V_MTE3>(MTE_SYNC_FLAG_ID_6); // 等VF计算完成 开始拷贝
            CopyWeightUB2L1();
            SetFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]); // 告诉VF前序拷贝完成 可以算别的了
            SetAivToAic();                                                 // 拷贝完成 核间同步
        } else {
            WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
            uint64_t curKSize = Min<uint64_t>(tiling_->matmulTiling.Ka - kOutIdx * kBlockSize_, kBlockSize_);
            uint64_t kInnerBlockCount = CeilDiv<uint64_t>(curKSize, tiling_->matmulTiling.baseK);
            if (kOutIdx == 0 && hasBias) {
                CopyBias2L1();
                SetFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_FLAG_ID_6);
            }
            GetAToL1(kOutIdx);
            WaitAivToAic();
            SetFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
            for (uint64_t kInnerIdx = 0; kInnerIdx < kInnerBlockCount; kInnerIdx++) {
                IterateMatmul(kOutIdx, kInnerIdx, kOffset);
                kOffset += tiling_->matmulTiling.baseK;
            }
            SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
            if (sumLoopNum_ > 2 && totalLoop_ < sumLoopNum_ - 2) {
                SetAicToAiv();
            }
        }
        l1DbFlag_ = l1DbFlag_ ^ 1;
        totalLoop_++;
    }
    GetTensorC(cGlobal_[block_.offset_.offsetC]);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::CopyBias2L1()
{
    AscendC::DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = block_.params_.singleCoreN * sizeof(biasType);
    params.srcStride = (tiling_->matmulTiling.N - block_.params_.singleCoreN) * sizeof(biasType);
    params.dstStride = 0;
    AscendC::DataCopyPadExtParams<biasType> padParams;
    AscendC::DataCopyPad((roundDbFlag_ == IDX_0) ? biasPingLocalTensor_ : biasPongLocalTensor_,
                         biasGlobal_[block_.offset_.offsetBias], params, padParams, tiling_->enablePreAlloc);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void
WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::CopyScaleAndOffsetToUB(uint64_t kOutIdx)
{
    if (antiQuantType == QuantType::PER_TENSOR || (antiQuantType == QuantType::PER_CHANNEL && kOutIdx > 0)) {
        return;
    }
    AscendC::DataCopyExtParams dataCopyParams;
    AscendC::DataCopyPadExtParams<scaleType> padParams;
    if constexpr (antiQuantType == QuantType::PER_CHANNEL) {
        dataCopyParams.blockCount = 1;
        // 32B对齐
        dataCopyParams.blockLen = sizeof(scaleType) * nUbSize_;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        if constexpr (hasAntiQuantOffset) {
            DataCopyPad(offsetLocalTensor_[tiling_->matmulTiling.baseN * tiling_->matmulTiling.stepN * roundDbFlag_],
                        antiquantOffsetGlobal_[block_.offset_.offsetN], dataCopyParams, padParams,
                        tiling_->enablePreAlloc);
        }
        DataCopyPad(scaleLocalTensor_[tiling_->matmulTiling.baseN * tiling_->matmulTiling.stepN * roundDbFlag_],
                    antiquantScaleGlobal_[block_.offset_.offsetN], dataCopyParams, padParams, tiling_->enablePreAlloc);
    }
    if constexpr (antiQuantType == QuantType::PER_GROUP) {
        dataCopyParams.blockCount = CeilDiv<uint64_t>(kUbSize_, tiling_->groupSize);
        // 32B对齐
        dataCopyParams.blockLen = sizeof(scaleType) * nUbSize_;
        dataCopyParams.srcStride = (tiling_->matmulTiling.N - nUbSize_) * sizeof(scaleType);
        dataCopyParams.dstStride = (tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseN - nUbSize_) *
                                   sizeof(scaleType) / ONE_BLK_SIZE;
        uint64_t gmScakeOffset = (kOutIdx * kBlockSize_ / tiling_->groupSize) * tiling_->matmulTiling.N +
                                 block_.offset_.offsetN;
        if constexpr (hasAntiQuantOffset) {
            DataCopyPad(offsetLocalTensor_[scaleElements_ * l1DbFlag_], antiquantOffsetGlobal_[gmScakeOffset],
                        dataCopyParams, padParams, tiling_->enablePreAlloc);
        }
        DataCopyPad(scaleLocalTensor_[scaleElements_ * l1DbFlag_], antiquantScaleGlobal_[gmScakeOffset], dataCopyParams,
                    padParams, tiling_->enablePreAlloc);
    }
}

template <typename xType, typename wType, typename scaleType, QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void AntiquantComputeNKVF(const ScaleQuantParam<xType, wType, scaleType>& param)
{
    MicroAPI::RegTensor<wType> weightB8RegTensor;
    MicroAPI::RegTensor<scaleType> scaleRegTensor;
    MicroAPI::RegTensor<xType> weightF16RegTensor;
    MicroAPI::RegTensor<half> weightFp16RegTensor;
    MicroAPI::RegTensor<scaleType> antiQuantOffsetReg;
    __local_mem__ xType* weightF16Addr = param.weightF16Addr;
    __local_mem__ wType* weightB8Addr = param.weightAddr;
    __local_mem__ scaleType* scaleAddr = param.scaleAddr;
    __local_mem__ scaleType* antiQuantOffsetAddr = 0;
    if constexpr (hasAntiQuantOffset) {
        antiQuantOffsetAddr = param.antiQuantOffsetAddr;
    }
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
    static constexpr MicroAPI::CastTrait S8_TO_FP16_TRAIT_ODD = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 AscendC::RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait FP16_TO_BF16_TRAIT = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                               MicroAPI::MaskMergeMode::ZEROING,
                                                               AscendC::RoundMode::CAST_RINT};

    if constexpr (antiQuantType == QuantType::PER_TENSOR) {
        MicroAPI::Duplicate(scaleRegTensor, param.scaleValue);
    }
    // baseN取值一般会大于k/128,把n的循环放在内轴
    for (uint16_t kIdx = 0; kIdx < param.kCount; kIdx++) {
        for (uint16_t nIdx = 0; nIdx < param.nUbSize; nIdx++) {
            if constexpr (antiQuantType == QuantType::PER_CHANNEL) {
                // k循环从0开始 n循环每次+1
                MicroAPI::AddrReg scaleAddrReg = MicroAPI::CreateAddrReg<scaleType>(kIdx, 0, nIdx, 1);
                MicroAPI::DataCopy<scaleType, MicroAPI::LoadDist::DIST_BRC_B16>(scaleRegTensor, scaleAddr,
                                                                                scaleAddrReg);
                if constexpr (hasAntiQuantOffset) {
                    MicroAPI::DataCopy<scaleType, MicroAPI::LoadDist::DIST_BRC_B16>(antiQuantOffsetReg,
                                                                                    antiQuantOffsetAddr, scaleAddrReg);
                }
            }
            MicroAPI::AddrReg weightB8AddrReg = MicroAPI::CreateAddrReg<wType>(kIdx, VL_F16_ELEMENTS, nIdx,
                                                                               param.singleCoreK);
            MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(weightB8RegTensor, weightB8Addr,
                                                                          weightB8AddrReg);
            if constexpr (IsSameType<xType, bfloat16_t>::value) {
                MicroAPI::Cast<half, wType, S8_TO_FP16_TRAIT_ODD>(weightFp16RegTensor, weightB8RegTensor, maskAll);
                MicroAPI::Cast<xType, half, FP16_TO_BF16_TRAIT>(weightF16RegTensor, weightFp16RegTensor, maskAll);
            } else {
                MicroAPI::Cast<xType, wType, S8_TO_FP16_TRAIT_ODD>(weightF16RegTensor, weightB8RegTensor, maskAll);
            }
            if constexpr (hasAntiQuantOffset) {
                if constexpr (antiQuantType == QuantType::PER_TENSOR) {
                    MicroAPI::Adds(weightF16RegTensor, weightF16RegTensor, param.offsetValue, maskAll);
                } else {
                    MicroAPI::Add(weightF16RegTensor, weightF16RegTensor, antiQuantOffsetReg, maskAll);
                }
            }
            MicroAPI::Mul(weightF16RegTensor, weightF16RegTensor, scaleRegTensor, maskAll);
            MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                weightF16Addr, weightF16RegTensor, CeilAlign(param.nUbSize, BLOCK_CUBE) + 1, 1, maskAll);
        }
        // k方向循环需要跨越128*( N方向对齐分形 + 1) - 前nUbSize个已读分形
        weightF16Addr += VL_F16_ELEMENTS * (CeilAlign(param.nUbSize, BLOCK_CUBE) + 1) - BLOCK_CUBE * param.nUbSize;
    }
}

template <typename xType, typename wType, typename scaleType, QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void AntiquantComputeKNVF(const ScaleQuantParam<xType, wType, scaleType>& param)
{
    MicroAPI::RegTensor<wType> weightB8RegTensor;
    MicroAPI::RegTensor<scaleType> scaleRegTensor;
    MicroAPI::RegTensor<xType> weightF16RegTensor;
    MicroAPI::RegTensor<half> weightFp16RegTensor;
    MicroAPI::RegTensor<scaleType> antiQuantOffsetReg;
    __local_mem__ xType* weightF16Addr = param.weightF16Addr;
    __local_mem__ wType* weightB8Addr = param.weightAddr;
    __local_mem__ scaleType* scaleAddr = param.scaleAddr;
    __local_mem__ scaleType* antiQuantOffsetAddr = 0;
    if constexpr (hasAntiQuantOffset) {
        antiQuantOffsetAddr = param.antiQuantOffsetAddr;
    }
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
    static constexpr MicroAPI::CastTrait S8_TO_FP16_TRAIT_ODD = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 AscendC::RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait FP16_TO_BF16_TRAIT = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                               MicroAPI::MaskMergeMode::ZEROING,
                                                               AscendC::RoundMode::CAST_RINT};

    if constexpr (antiQuantType == QuantType::PER_TENSOR) {
        MicroAPI::Duplicate(scaleRegTensor, param.scaleValue);
    }
    if constexpr (antiQuantType == QuantType::PER_CHANNEL) {
        if constexpr (hasAntiQuantOffset) {
            MicroAPI::DataCopy<scaleType, MicroAPI::LoadDist::DIST_NORM>(antiQuantOffsetReg, antiQuantOffsetAddr);
        }
        MicroAPI::DataCopy<scaleType, MicroAPI::LoadDist::DIST_NORM>(scaleRegTensor, scaleAddr);
    }
    for (uint16_t kIdx = 0; kIdx < param.kUbSize; kIdx++) {
        MicroAPI::AddrReg weightB8AddrReg = MicroAPI::CreateAddrReg<wType>(kIdx, param.singleCoreN);
        MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(weightB8RegTensor, weightB8Addr, weightB8AddrReg);
        if constexpr (IsSameType<xType, bfloat16_t>::value) {
            MicroAPI::Cast<half, wType, S8_TO_FP16_TRAIT_ODD>(weightFp16RegTensor, weightB8RegTensor, maskAll);
            MicroAPI::Cast<xType, half, FP16_TO_BF16_TRAIT>(weightF16RegTensor, weightFp16RegTensor, maskAll);
        } else {
            MicroAPI::Cast<xType, wType, S8_TO_FP16_TRAIT_ODD>(weightF16RegTensor, weightB8RegTensor, maskAll);
        }
        if constexpr (hasAntiQuantOffset) {
            if constexpr (antiQuantType == QuantType::PER_TENSOR) {
                MicroAPI::Adds(weightF16RegTensor, weightF16RegTensor, param.offsetValue, maskAll);
            } else {
                MicroAPI::Add(weightF16RegTensor, weightF16RegTensor, antiQuantOffsetReg, maskAll);
            }
        }
        MicroAPI::Mul(weightF16RegTensor, weightF16RegTensor, scaleRegTensor, maskAll);
        MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            weightF16Addr, weightF16RegTensor, CeilAlign(param.kUbSize, BLOCK_CUBE) + 1, 1, maskAll);
    }
}

template <typename xType, typename wType, typename scaleType, QuantType antiQuantType, bool hasAntiQuantOffset>
__aicore__ inline void AntiquantComputeKNPerCVF(const ScaleQuantParam<xType, wType, scaleType>& param)
{
    MicroAPI::RegTensor<int4x2_t> weightB4RegTensor;
    MicroAPI::RegTensor<int4x2_t> weightB4RegTensor2;
    MicroAPI::RegTensor<xType> weightF16RegTensor;
    MicroAPI::RegTensor<xType> weightF16RegTensor2;
    MicroAPI::RegTensor<scaleType> scaleRegTensor;
    MicroAPI::RegTensor<scaleType> antiQuantOffsetReg;
    __local_mem__ xType* weightF16Addr = param.weightF16Addr;
    __local_mem__ int4x2_t* weightB4Addr = (__local_mem__ int4x2_t*)param.weightAddr;
    __local_mem__ scaleType* scaleAddr = param.scaleAddr;
    __local_mem__ scaleType* antiQuantOffsetAddr = 0;
    if constexpr (hasAntiQuantOffset) {
        antiQuantOffsetAddr = param.antiQuantOffsetAddr;
    }
    uint16_t nCount = CeilDiv<uint16_t>(param.nUbSize, BLOCK_CUBE);
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
    static constexpr MicroAPI::CastTrait CAST_S4_TO_F16_TRAIT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 AscendC::RoundMode::UNKNOWN};

    if constexpr (antiQuantType == QuantType::PER_TENSOR) {
        MicroAPI::Duplicate(scaleRegTensor, param.scaleValue);
    }
    for (uint16_t nIdx = 0; nIdx < nCount; nIdx++) {
        if constexpr (antiQuantType == QuantType::PER_CHANNEL) {
            MicroAPI::AddrReg scaleAddrReg = MicroAPI::CreateAddrReg<scaleType>(nIdx, BLOCK_CUBE);
            MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(scaleRegTensor, scaleAddr, scaleAddrReg);
            if constexpr (hasAntiQuantOffset) {
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(antiQuantOffsetReg, antiQuantOffsetAddr,
                                                                        scaleAddrReg);
            }
        }

        for (uint16_t kIdx = 0; kIdx < param.kCount; kIdx++) {
            AddrReg weightB4AddrReg = MicroAPI::CreateAddrReg<int4x2_t>(
                nIdx, CeilDiv<uint64_t>(param.singleCoreK, BLOCK_CUBE) * VL_F16_ELEMENTS, kIdx, VL_F16_ELEMENTS);
            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(weightB4RegTensor, weightB4Addr,
                                                                              weightB4AddrReg);
            MicroAPI::DataCopy<int4x2_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                weightB4RegTensor2, weightB4Addr + VL_WEIGHT_ELEMENTS, weightB4AddrReg);
            MicroAPI::Cast<xType, int4x2_t, CAST_S4_TO_F16_TRAIT>(weightF16RegTensor, weightB4RegTensor, maskAll);
            MicroAPI::Cast<xType, int4x2_t, CAST_S4_TO_F16_TRAIT>(weightF16RegTensor2, weightB4RegTensor2, maskAll);
            if constexpr (hasAntiQuantOffset) {
                if constexpr (antiQuantType == QuantType::PER_CHANNEL) {
                    MicroAPI::Add(weightF16RegTensor, weightF16RegTensor, antiQuantOffsetReg, maskAll);
                    MicroAPI::Add(weightF16RegTensor2, weightF16RegTensor2, antiQuantOffsetReg, maskAll);
                } else {
                    MicroAPI::Adds(weightF16RegTensor, weightF16RegTensor, param.offsetValue, maskAll);
                    MicroAPI::Adds(weightF16RegTensor2, weightF16RegTensor2, param.offsetValue, maskAll);
                }
            }
            MicroAPI::Mul(weightF16RegTensor, weightF16RegTensor, scaleRegTensor, maskAll);
            MicroAPI::Mul(weightF16RegTensor2, weightF16RegTensor2, scaleRegTensor, maskAll);
            // 512实际上为跨1024B 512 * sizeof（F16） = 1024B
            AddrReg weightF16PhyAddrReg = MicroAPI::CreateAddrReg<xType>(
                nIdx, CeilDiv<uint64_t>(param.kUbSize, BLOCK_CUBE) * 512, kIdx, 512);
            MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(weightF16Addr, weightF16RegTensor,
                                                                          weightF16PhyAddrReg, maskAll);
            MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(
                weightF16Addr + VL_F16_ELEMENTS, weightF16RegTensor2, weightF16PhyAddrReg, maskAll);
        }
    }
}

template <typename xType, typename wType, typename scaleType, bool hasAntiQuantOffset>
__aicore__ inline void AntiquantComputeKNB4PerGVF(const ScaleQuantParam<xType, wType, scaleType>& param)
{
    using bType = std::conditional_t<IsSameType<wType, fp4x2_e2m1_t>::value, wType, int4x2_t>;
    using fp16Type = std::conditional_t<IsSameType<wType, fp4x2_e2m1_t>::value, bfloat16_t, xType>;
    MicroAPI::RegTensor<bType> weightB4RegTensor;
    MicroAPI::RegTensor<bType> weightB4RegTensor2;
    MicroAPI::RegTensor<fp16Type> weightB16RegTensor;
    MicroAPI::RegTensor<fp16Type> weightB16RegTensor2;
    MicroAPI::RegTensor<xType> weightF16RegTensor;
    MicroAPI::RegTensor<xType> weightF16RegTensor2;
    MicroAPI::RegTensor<scaleType> scaleRegTensor;
    MicroAPI::RegTensor<scaleType> antiQuantOffsetReg;
    __local_mem__ xType* weightF16Addr = param.weightF16Addr;
    __local_mem__ bType* weightB4Addr = (__local_mem__ bType*)param.weightAddr;
    __local_mem__ scaleType* scaleAddr = param.scaleAddr;
    __local_mem__ scaleType* antiQuantOffsetAddr = 0;
    if constexpr (hasAntiQuantOffset) {
        antiQuantOffsetAddr = param.antiQuantOffsetAddr;
    }
    uint16_t nCount = CeilDiv<uint16_t>(param.nUbSize, BLOCK_CUBE);
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
    static constexpr MicroAPI::CastTrait CAST_S4_TO_F16_TRAIT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 AscendC::RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait BF16_TO_FP16_TRAIT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                               MicroAPI::MaskMergeMode::ZEROING,
                                                               AscendC::RoundMode::CAST_RINT};
    for (uint16_t nIdx = 0; nIdx < nCount; nIdx++) {
        for (uint16_t kIdx = 0; kIdx < param.kCount; kIdx++) {
            MicroAPI::AddrReg scaleAddrReg = MicroAPI::CreateAddrReg<scaleType>(nIdx, BLOCK_CUBE, kIdx,
                                                                                param.singleCoreN);
            MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(scaleRegTensor, scaleAddr, scaleAddrReg);
            if constexpr (hasAntiQuantOffset) {
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(antiQuantOffsetReg, antiQuantOffsetAddr,
                                                                        scaleAddrReg);
            }
            for (uint16_t groupNum = 0; groupNum < param.groupKNum; groupNum++) {
                AddrReg weightB4AddrReg = MicroAPI::CreateAddrReg<bType>(
                    nIdx, CeilDiv<uint64_t>(param.singleCoreK, BLOCK_CUBE) * VL_F16_ELEMENTS, kIdx,
                    VL_F16_ELEMENTS * param.groupKNum, groupNum, VL_F16_ELEMENTS);
                MicroAPI::DataCopy<bType, MicroAPI::LoadDist::DIST_UNPACK4_B8>(weightB4RegTensor, weightB4Addr,
                                                                               weightB4AddrReg);
                MicroAPI::DataCopy<bType, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                    weightB4RegTensor2, weightB4Addr + VL_WEIGHT_ELEMENTS, weightB4AddrReg);
                if constexpr (IsSameType<xType, fp16Type>::value) {
                    MicroAPI::Cast<xType, bType, CAST_S4_TO_F16_TRAIT>(weightF16RegTensor, weightB4RegTensor, maskAll);
                    MicroAPI::Cast<xType, bType, CAST_S4_TO_F16_TRAIT>(weightF16RegTensor2, weightB4RegTensor2,
                                                                       maskAll);
                } else {
                    MicroAPI::Cast<bfloat16_t, bType, CAST_S4_TO_F16_TRAIT>(weightB16RegTensor, weightB4RegTensor,
                                                                            maskAll);
                    MicroAPI::Cast<bfloat16_t, bType, CAST_S4_TO_F16_TRAIT>(weightB16RegTensor2, weightB4RegTensor2,
                                                                            maskAll);
                    MicroAPI::Cast<xType, bfloat16_t, BF16_TO_FP16_TRAIT>(weightF16RegTensor, weightB16RegTensor,
                                                                          maskAll);
                    MicroAPI::Cast<xType, bfloat16_t, BF16_TO_FP16_TRAIT>(weightF16RegTensor2, weightB16RegTensor2,
                                                                          maskAll);
                }
                if constexpr (hasAntiQuantOffset) {
                    MicroAPI::Add(weightF16RegTensor, weightF16RegTensor, antiQuantOffsetReg, maskAll);
                    MicroAPI::Add(weightF16RegTensor2, weightF16RegTensor2, antiQuantOffsetReg, maskAll);
                }
                MicroAPI::Mul(weightF16RegTensor, weightF16RegTensor, scaleRegTensor, maskAll);
                MicroAPI::Mul(weightF16RegTensor2, weightF16RegTensor2, scaleRegTensor, maskAll);
                // 512实际上为跨1024B 512 * sizeof（F16） = 1024B
                AddrReg weightF16PhyAddrReg = MicroAPI::CreateAddrReg<xType>(
                    nIdx, CeilDiv<uint64_t>(param.kUbSize, BLOCK_CUBE) * 512, kIdx, 512 * param.groupKNum, groupNum,
                    512);
                MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(weightF16Addr, weightF16RegTensor,
                                                                              weightF16PhyAddrReg, maskAll);
                MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(
                    weightF16Addr + VL_F16_ELEMENTS, weightF16RegTensor2, weightF16PhyAddrReg, maskAll);
            }
        }
    }
}

template <typename xType, typename wType, typename scaleType, bool hasAntiQuantOffset>
__aicore__ inline void AntiquantComputeKNB8PerGVF(const ScaleQuantParam<xType, wType, scaleType>& param)
{
    MicroAPI::RegTensor<wType> weightB8RegTensor;
    MicroAPI::RegTensor<wType> weightB8RegTensor2;
    MicroAPI::RegTensor<half> weightFp16RegTensor;
    MicroAPI::RegTensor<half> weightFp16RegTensor2;
    MicroAPI::RegTensor<xType> weightF16RegTensor;
    MicroAPI::RegTensor<xType> weightF16RegTensor2;
    MicroAPI::RegTensor<scaleType> scaleRegTensor;
    MicroAPI::RegTensor<scaleType> antiQuantOffsetReg;
    __local_mem__ xType* weightF16Addr = param.weightF16Addr;
    __local_mem__ wType* weightB8Addr = param.weightAddr;
    __local_mem__ scaleType* scaleAddr = param.scaleAddr;
    __local_mem__ scaleType* antiQuantOffsetAddr = 0;
    if constexpr (hasAntiQuantOffset) {
        antiQuantOffsetAddr = param.antiQuantOffsetAddr;
    }
    uint16_t nCount = CeilDiv<uint16_t>(param.nUbSize, BLOCK_CUBE);
    // 256 双寄存器处理的wType数据个数 VL_F16_ELEMENTS*2
    uint16_t vfProcessElement = 256;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
    static constexpr MicroAPI::CastTrait S8_TO_FP16_TRAIT_ODD = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                                 MicroAPI::MaskMergeMode::ZEROING,
                                                                 AscendC::RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait FP16_TO_BF16_TRAIT = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                                               MicroAPI::MaskMergeMode::ZEROING,
                                                               AscendC::RoundMode::CAST_RINT};
    for (uint16_t nIdx = 0; nIdx < nCount; nIdx++) {
        for (uint16_t kIdx = 0; kIdx < param.kCount; kIdx++) {
            MicroAPI::AddrReg scaleAddrReg = MicroAPI::CreateAddrReg<scaleType>(nIdx, BLOCK_CUBE, kIdx,
                                                                                param.singleCoreN);
            MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(scaleRegTensor, scaleAddr, scaleAddrReg);
            if constexpr (hasAntiQuantOffset) {
                MicroAPI::DataCopy<xType, MicroAPI::LoadDist::DIST_BLK>(antiQuantOffsetReg, antiQuantOffsetAddr,
                                                                        scaleAddrReg);
            }
            for (uint16_t groupNum = 0; groupNum < param.groupKNum; groupNum++) {
                AddrReg weightB8AddrReg = MicroAPI::CreateAddrReg<wType>(
                    nIdx, CeilDiv<uint64_t>(param.singleCoreK, BLOCK_CUBE) * vfProcessElement, kIdx,
                    vfProcessElement * param.groupKNum, groupNum, vfProcessElement);
                MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(weightB8RegTensor, weightB8Addr,
                                                                              weightB8AddrReg);
                MicroAPI::DataCopy<wType, MicroAPI::LoadDist::DIST_UNPACK_B8>(
                    weightB8RegTensor2, weightB8Addr + VL_F16_ELEMENTS, weightB8AddrReg);
                if constexpr (IsSameType<xType, half>::value) {
                    MicroAPI::Cast<xType, wType, S8_TO_FP16_TRAIT_ODD>(weightF16RegTensor, weightB8RegTensor, maskAll);
                    MicroAPI::Cast<xType, wType, S8_TO_FP16_TRAIT_ODD>(weightF16RegTensor2, weightB8RegTensor2,
                                                                       maskAll);
                } else {
                    MicroAPI::Cast<half, wType, S8_TO_FP16_TRAIT_ODD>(weightFp16RegTensor, weightB8RegTensor, maskAll);
                    MicroAPI::Cast<half, wType, S8_TO_FP16_TRAIT_ODD>(weightFp16RegTensor2, weightB8RegTensor2,
                                                                      maskAll);
                    MicroAPI::Cast<xType, half, FP16_TO_BF16_TRAIT>(weightF16RegTensor, weightFp16RegTensor, maskAll);
                    MicroAPI::Cast<xType, half, FP16_TO_BF16_TRAIT>(weightF16RegTensor2, weightFp16RegTensor2, maskAll);
                }
                if constexpr (hasAntiQuantOffset) {
                    MicroAPI::Add(weightF16RegTensor, weightF16RegTensor, antiQuantOffsetReg, maskAll);
                    MicroAPI::Add(weightF16RegTensor2, weightF16RegTensor2, antiQuantOffsetReg, maskAll);
                }
                MicroAPI::Mul(weightF16RegTensor, weightF16RegTensor, scaleRegTensor, maskAll);
                MicroAPI::Mul(weightF16RegTensor2, weightF16RegTensor2, scaleRegTensor, maskAll);
                // 512实际上为跨1024B 512 * sizeof（F16） = 1024B
                AddrReg weightF16PhyAddrReg = MicroAPI::CreateAddrReg<xType>(
                    nIdx, CeilDiv<uint64_t>(param.kUbSize, BLOCK_CUBE) * 512, kIdx, 512 * param.groupKNum, groupNum,
                    512);
                MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(weightF16Addr, weightF16RegTensor,
                                                                              weightF16PhyAddrReg, maskAll);
                MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_NORM_B16>(
                    weightF16Addr + VL_F16_ELEMENTS, weightF16RegTensor2, weightF16PhyAddrReg, maskAll);
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::AntiquantCompute()
{
    ScaleQuantParam<xType, wType, scaleType> param;
    param.singleCoreN = tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseN;
    param.singleCoreK = tiling_->matmulTiling.stepKb * tiling_->matmulTiling.baseK;
    param.kUbSize = kUbSize_;
    param.nUbSize = nUbSize_;
    param.groupKNum = tiling_->groupSize / BLOCK_CUBE;
    // VF每次处理128个数字
    param.kCount = CeilDiv<uint16_t>(kUbSize_, 128);
    if constexpr (weightFormat == CubeFormat::NZ) {
        // 256:512B /sizeof(F16)
        param.weightF16Addr = (__local_mem__ xType*)weightUbOutTensor_[256 * l1DbFlag_].GetPhyAddr();
    } else {
        param.weightF16Addr = (__local_mem__ xType*)weightUbOutTensor_[weightUbOutElements_ * l1DbFlag_].GetPhyAddr();
    }
    param.weightAddr = (__local_mem__ wType*)weightUbInTensor_[weightUbInElements_ * l1DbFlag_].GetPhyAddr();

    if constexpr (antiQuantType == QuantType::PER_CHANNEL) {
        param.scaleAddr = (__local_mem__ scaleType*)scaleLocalTensor_[tiling_->matmulTiling.baseN *
                                                                      tiling_->matmulTiling.stepN * roundDbFlag_]
                              .GetPhyAddr();
        if constexpr (hasAntiQuantOffset) {
            param.antiQuantOffsetAddr = (__local_mem__ scaleType*)
                                            offsetLocalTensor_[tiling_->matmulTiling.baseN *
                                                               tiling_->matmulTiling.stepN * roundDbFlag_]
                                                .GetPhyAddr();
        }
    }
    if constexpr (antiQuantType == QuantType::PER_GROUP) {
        param.kCount = CeilDiv<uint16_t>(kUbSize_, tiling_->groupSize);
        param.scaleAddr = (__local_mem__ scaleType*)scaleLocalTensor_[scaleElements_ * l1DbFlag_].GetPhyAddr();
        if constexpr (hasAntiQuantOffset) {
            param.antiQuantOffsetAddr = (__local_mem__ scaleType*)offsetLocalTensor_[scaleElements_ * l1DbFlag_]
                                            .GetPhyAddr();
        }
    }
    if constexpr (antiQuantType == QuantType::PER_TENSOR) {
        param.scaleValue = scaleValue_;
        param.offsetValue = offsetValue_;
    }
    SetFlag<HardEvent::MTE2_V>(MTE_SYNC_FLAG_ID_6);
    WaitFlag<HardEvent::MTE2_V>(MTE_SYNC_FLAG_ID_6); // 拷贝 等待

    WaitFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
    if constexpr (bTrans) {
        AscendC::VF_CALL<AntiquantComputeNKVF<xType, wType, scaleType, antiQuantType, hasAntiQuantOffset>>(param);
    } else {
        if constexpr (antiQuantType == QuantType::PER_GROUP) {
            if constexpr (AscendC::IsSupportB4<wType>()) {
                AscendC::VF_CALL<AntiquantComputeKNB4PerGVF<xType, wType, scaleType, hasAntiQuantOffset>>(param);
            } else {
                AscendC::VF_CALL<AntiquantComputeKNB8PerGVF<xType, wType, scaleType, hasAntiQuantOffset>>(param);
            }
        } else {
            if constexpr (IsSameType<wType, int4b_t>::value && weightFormat == CubeFormat::NZ) {
                param.kCount = CeilDiv<uint16_t>(kUbSize_, BLOCK_CUBE);
                AscendC::VF_CALL<AntiquantComputeKNPerCVF<xType, wType, scaleType, antiQuantType, hasAntiQuantOffset>>(
                    param);
            } else {
                AscendC::VF_CALL<AntiquantComputeKNVF<xType, wType, scaleType, antiQuantType, hasAntiQuantOffset>>(
                    param);
            }
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::CopyWeightGm2UB(
    uint64_t kOutIdx)
{
    uint64_t singleNSize = tiling_->matmulTiling.stepN * tiling_->matmulTiling.baseN;
    uint64_t kOffset = kOutIdx * kBlockSize_;
    kUbSize_ = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, kBlockSize_);
    nUbSize_ = Min<uint64_t>(tiling_->matmulTiling.N - block_.offset_.offsetN, block_.params_.singleCoreN);
    uint64_t weightGmOffset;
    AscendC::DataCopyExtParams dataCopyParams;
    AscendC::DataCopyPadExtParams<wType> padParams;
    if constexpr (bTrans) {
        dataCopyParams.blockCount = nUbSize_;
        dataCopyParams.blockLen = kUbSize_ * sizeof(wType);
        dataCopyParams.srcStride = (tiling_->matmulTiling.Ka - kUbSize_) * sizeof(wType);
        dataCopyParams.dstStride = (kBlockSize_ - kUbSize_) * sizeof(wType) / ONE_BLK_SIZE;
        weightGmOffset = (block_.offset_.offsetN * tiling_->matmulTiling.Kb + kOffset);
    } else {
        if constexpr (weightFormat == CubeFormat::NZ) {
            dataCopyParams.blockCount = CeilDiv<uint16_t>(nUbSize_, BLOCK_CUBE);
            if constexpr (AscendC::IsSupportB4<wType>()) {
                dataCopyParams.blockLen = kUbSize_ * BLOCK_CUBE >> 1;
                dataCopyParams.srcStride = (CeilAlign(tiling_->matmulTiling.Kb, BLOCK_CUBE) - kUbSize_) * BLOCK_CUBE >>
                                           1;
                dataCopyParams.dstStride = ((kBlockSize_ - kUbSize_) * BLOCK_CUBE / ONE_BLK_SIZE) >> 1;
            } else {
                dataCopyParams.blockLen = kUbSize_ * BLOCK_CUBE * sizeof(wType);
                dataCopyParams.srcStride = (tiling_->matmulTiling.Kb - kUbSize_) * BLOCK_CUBE * sizeof(wType);
                dataCopyParams.dstStride = ((kBlockSize_ - kUbSize_) * BLOCK_CUBE / ONE_BLK_SIZE) * sizeof(wType);
            }
            weightGmOffset = block_.offset_.offsetN * CeilAlign(tiling_->matmulTiling.Kb, BLOCK_CUBE) +
                             kOffset * BLOCK_CUBE;
        } else {
            dataCopyParams.blockCount = kUbSize_;
            dataCopyParams.blockLen = nUbSize_ * sizeof(wType);
            dataCopyParams.srcStride = (tiling_->matmulTiling.N - nUbSize_) * sizeof(wType);
            dataCopyParams.dstStride = (singleNSize - nUbSize_) * sizeof(wType) / ONE_BLK_SIZE;
            weightGmOffset = block_.offset_.offsetN + tiling_->matmulTiling.N * kOffset;
        }
    }
    DataCopyPad(weightUbInTensor_[weightUbInElements_ * l1DbFlag_].template ReinterpretCast<wType>(),
                bGlobal_[weightGmOffset], dataCopyParams, padParams, tiling_->enablePreAlloc);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::CopyWeightUB2L1()
{
    AscendC::DataCopyParams dataCopyParams;
    if constexpr (weightFormat == CubeFormat::ND) {
        if constexpr (bTrans) {
            dataCopyParams.blockLen = CeilAlign(nUbSize_, BLOCK_CUBE);
            dataCopyParams.blockCount = CeilDiv<uint16_t>(kUbSize_, BLOCK_CUBE);
        } else {
            dataCopyParams.blockLen = CeilAlign(kUbSize_, BLOCK_CUBE);
            dataCopyParams.blockCount = CeilDiv<uint16_t>(nUbSize_, BLOCK_CUBE);
        }
        // 1:消除解bank的32B
        dataCopyParams.srcStride = 1;
        dataCopyParams.dstStride = 0;
        AscendC::DataCopy((l1DbFlag_ == IDX_0) ? bL1PingLocalTensor_ : bL1PongLocalTensor_,
                          weightUbOutTensor_[weightUbOutElements_ * l1DbFlag_], dataCopyParams);
    } else {
        dataCopyParams.blockLen = BLOCK_CUBE;
        dataCopyParams.blockCount = CeilDiv<uint64_t>(kUbSize_, BLOCK_CUBE) * CeilDiv<uint64_t>(nUbSize_, BLOCK_CUBE);
        // BLOCK_CUBE*BLOCK_CUBE*sizeof(F16)/32B
        dataCopyParams.srcStride = BLOCK_CUBE;
        dataCopyParams.dstStride = 0;
        AscendC::DataCopy(
            // 256:512B /sizeof(F16)
            (l1DbFlag_ == IDX_0) ? bL1PingLocalTensor_ : bL1PongLocalTensor_, weightUbOutTensor_[256 * l1DbFlag_],
            dataCopyParams);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::IterateMatmul(
    uint64_t kOutIdx, uint64_t kInnerIdx, uint64_t kOffset)
{
    if ASCEND_IS_AIC {
        WeightQuantBatchMatmul::Arch35::L0CopyAndCalcParams l0CopyAndCalcParams;
        l0CopyAndCalcParams.mL0Size = Min<uint64_t>(tiling_->matmulTiling.M - block_.offset_.offsetM,
                                                    block_.params_.singleCoreM);
        l0CopyAndCalcParams.nL0Size = Min<uint64_t>(tiling_->matmulTiling.N - block_.offset_.offsetN,
                                                    block_.params_.singleCoreN);
        l0CopyAndCalcParams.nL1Size = l0CopyAndCalcParams.nL0Size;
        l0CopyAndCalcParams.mL1Size = l0CopyAndCalcParams.mL0Size;
        l0CopyAndCalcParams.kL1Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOutIdx * kBlockSize_, kBlockSize_);
        l0CopyAndCalcParams.kL0Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, tiling_->matmulTiling.baseK);
        l0CopyAndCalcParams.isFirstKLoop = kOutIdx == 0 && kInnerIdx == 0;
        l0CopyAndCalcParams.isLastKLoop = tiling_->matmulTiling.Ka - kOffset <= tiling_->matmulTiling.baseK;

        uint64_t aL1Offset = kInnerIdx * tiling_->matmulTiling.baseK *
                             CeilAlign<uint64_t>(l0CopyAndCalcParams.mL0Size, BLOCK_CUBE);
        uint64_t bL1Offset = 0;
        if constexpr (bTrans) {
            bL1Offset = kInnerIdx * tiling_->matmulTiling.baseK *
                        CeilAlign<uint64_t>(l0CopyAndCalcParams.nL0Size, BLOCK_CUBE);
        } else {
            bL1Offset = kInnerIdx * tiling_->matmulTiling.baseK * BLOCK_CUBE;
        }
        WaitFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l0DbFlag_]);
        if (l0CopyAndCalcParams.isFirstKLoop && hasBias) {
            WaitFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_FLAG_ID_6);
            WeightQuantBatchMatmul::Arch35::CopyBiasL1ToBT(
                biasBtTensor_[roundDbFlag_ << 9], (roundDbFlag_ == IDX_0) ? biasPingLocalTensor_ : biasPongLocalTensor_,
                l0CopyAndCalcParams.nL0Size); // 左移9位，biastable的一半
        }
        // 14: 根据aType取L0A的一半位置
        LocalTensor<xType> aL0Tensor = aL0Tensor_[l0DbFlag_ << 14];
        WeightQuantBatchMatmul::Arch35::LoadAL1ToL0<xType, xType>(
            aL0Tensor, (l1DbFlag_ == IDX_0) ? aL1PingLocalTensor_[aL1Offset] : aL1PongLocalTensor_[aL1Offset],
            l0CopyAndCalcParams);
        // 14: weight在VF量化后变成xType
        LocalTensor<xType> bL0Tensor = bL0Tensor_[l0DbFlag_ << 14];
        WeightQuantBatchMatmul::Arch35::LoadBL1ToL0<xType, xType, bTrans>(
            bL0Tensor, (l1DbFlag_ == IDX_0) ? bL1PingLocalTensor_[bL1Offset] : bL1PongLocalTensor_[bL1Offset],
            l0CopyAndCalcParams);
        SetFlag<HardEvent::MTE1_M>(MTE_SYNC_FLAG_ID_6);
        WaitFlag<HardEvent::MTE1_M>(MTE_SYNC_FLAG_ID_6);
        // 15: 根据cType取L0C的一半位置
        LocalTensor<float> cL0Tensor = cL0Tensor_[roundDbFlag_ << 15];
        if (l0CopyAndCalcParams.isFirstKLoop && hasBias) {
            // 9: baisTable一半位置
            WeightQuantBatchMatmul::Arch35::MmadCompute<float, xType, xType, float>(
                cL0Tensor, aL0Tensor, bL0Tensor, biasBtTensor_[roundDbFlag_ << 9], l0CopyAndCalcParams);
        } else {
            WeightQuantBatchMatmul::Arch35::MmadCompute<float, xType, xType>(cL0Tensor, aL0Tensor, bL0Tensor,
                                                                             l0CopyAndCalcParams);
        }
        SetFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l0DbFlag_]);
        l0DbFlag_ ^= 1;
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::GetAToL1(
    uint64_t kOutIdx)
{
    // A 矩阵 K 方向搬入 L1 Size 计算
    uint64_t kOffset = kOutIdx * kBlockSize_;
    uint64_t kL1Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, kBlockSize_);
    // A矩阵 M 方向搬入L1 Size计算
    uint64_t mL1Size = Min<uint64_t>(tiling_->matmulTiling.M - block_.offset_.offsetM, block_.params_.singleCoreM);
    // A 矩阵在 GM 上搬运的偏移计算：mOffset * K + kOffset
    uint64_t aGmOffset = (block_.offset_.offsetM * tiling_->matmulTiling.Ka + kOffset);

    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = mL1Size;
    nd2nzParams.dValue = kL1Size;
    nd2nzParams.srcDValue = tiling_->matmulTiling.Ka;
    nd2nzParams.srcNdMatrixStride = 0;

    nd2nzParams.dstNzC0Stride = CeilAlign<uint64_t>(nd2nzParams.nValue, BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    // 最后一个参数enablePreReadIgnoreSync = true表示开启PreAllocation
    DataCopy((l1DbFlag_ == IDX_0) ? aL1PingLocalTensor_ : aL1PongLocalTensor_, aGlobal_[aGmOffset], nd2nzParams,
             tiling_->enablePreAlloc);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::GetTensorC(
    const GlobalTensor<yType>& outTensor)
{
    if ASCEND_IS_AIV {
        return;
    }
    LocalTensor<float> cL0Tensor = cL0Tensor_[(roundDbFlag_) << 15];
    WeightQuantBatchMatmul::Arch35::FixL0CToDstParams fixL0CToDstParams;
    fixL0CToDstParams.mL0Size = block_.params_.singleCoreM;
    fixL0CToDstParams.nL0Size = block_.params_.singleCoreN;
    fixL0CToDstParams.outNSize = tiling_->matmulTiling.N;
    WeightQuantBatchMatmul::Arch35::FixL0CToDst<yType, float>(outTensor, cL0Tensor, fixL0CToDstParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::EndSync()
{
    if ASCEND_IS_AIV {
        for (int idx = 0; idx < DOUBLE_BUFFER; ++idx) {
            WaitFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            WaitFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
        }
    }

    if ASCEND_IS_AIC {
        for (int idx = 0; idx < DOUBLE_BUFFER; ++idx) {
            WaitFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::SetAivToAic()
{
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::WaitAivToAic()
{
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
}
LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::SetAicToAiv()
{
    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
}

LOCAL_TEMPLATE_CLASS_PARAMS_GENERAL
__aicore__ inline void WeightQuantBatchMatmulV2ASWGeneralKernel<LOCAL_TEMPLATE_FUNC_PARAMS_GENERAL>::WaitAicToAiv()
{
    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
}

} // namespace WeightQuantBatchMatmulV2::Arch35
