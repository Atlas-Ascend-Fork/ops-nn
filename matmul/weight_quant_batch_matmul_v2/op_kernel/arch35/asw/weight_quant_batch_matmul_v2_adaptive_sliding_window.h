/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window.h
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

#ifndef LOCAL_TEMPLATE_CLASS_PARAMS_ASW
#define LOCAL_TEMPLATE_CLASS_PARAMS_ASW                                                                                \
    template <typename xType, typename wType, typename biasType, typename yType, bool bTrans, QuantType antiQuantType, \
              bool hasAntiQuantOffset, QuantType quantType, CubeFormat weightFormat, bool hasBias>
#endif
#ifndef LOCAL_TEMPLATE_FUNC_PARAMS_ASW
#define LOCAL_TEMPLATE_FUNC_PARAMS_ASW \
    xType, wType, biasType, yType, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat, hasBias
#endif

using AscendC::AIC;
using AscendC::AIV;
using AscendC::BLOCK_CUBE;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::fp8_e8m0_t;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;

namespace MicroAPI = AscendC::MicroAPI;

namespace WeightQuantBatchMatmulV2::Arch35 {

template <typename yType, typename biasType>
struct BiasAddParam {
    uint16_t mCount;
    uint16_t nCount;
    uint16_t ubAlignN;

    __local_mem__ float* cAddr;
    __local_mem__ yType* yAddr;
    __local_mem__ biasType* biasAddr;
};

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
class WeightQuantBatchMatmulV2ASWKernel {
public:
    __aicore__ inline WeightQuantBatchMatmulV2ASWKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const void* tilingData);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void AddBias(uint64_t roundIndex);
    __aicore__ inline void InitUbBuffs();
    __aicore__ inline void EndSync();
    __aicore__ inline void InitCubeBuffs();

    uint32_t blockIdx_;
    const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData* tiling_;

    GlobalTensor<xType> aGlobal_;
    GlobalTensor<wType> bGlobal_;
    GlobalTensor<yType> cGlobal_;
    GlobalTensor<uint64_t> scaleGlobal_;
    GlobalTensor<biasType> biasGlobal_;

    LocalTensor<float> cLocalTensor_;
    LocalTensor<yType> yLocalTensor_;
    LocalTensor<biasType> biasLocalTensor_;

    uint64_t ubElements_ = 0;
    uint64_t ubCTensorSize_ = 0;

    WeightQuantBatchMatmulV2::Arch35::WeightQuantBmmAswBlock block_;

private:
    uint64_t roundDbFlag_ = 0;
    uint64_t l1DbFlag_ = 0;
    uint64_t l0DbFlag_ = 0;
    uint64_t kBlockSize_ = 0;
    uint64_t totalKBlocks_ = 0;

    GlobalTensor<uint8_t> wGlobalU8_;
    GlobalTensor<uint64_t> antiquantScaleGlobal_;

    LocalTensor<xType> aL1PingLocalTensor_;
    LocalTensor<xType> aL1PongLocalTensor_;
    LocalTensor<wType> bL1PingLocalTensor_;
    LocalTensor<wType> bL1PongLocalTensor_;
    LocalTensor<uint64_t> scalePingLocalTensor_;
    LocalTensor<uint64_t> scalePongLocalTensor_;
    LocalTensor<fp8_e8m0_t> scaleWL1LocalBuf_;

    LocalTensor<float> cL0Tensor_;
    LocalTensor<xType> aL0Tensor_;
    LocalTensor<wType> bL0Tensor_;

    __aicore__ inline void ProcessBaseBlockCompute(uint64_t roundIndex);
    __aicore__ inline void GetABToL1(uint64_t kOutIdx);
    __aicore__ inline void CopyXNd2Nz(uint64_t kOutIdx);
    __aicore__ inline void CopyWeightGm2L1(uint64_t kOutIdx);
    __aicore__ inline void IterateMatmul(uint64_t kOutIdx, uint64_t kInnerIdx, uint64_t kOffset);
    __aicore__ inline void GetTensorC(const GlobalTensor<yType>& outTensor);
    __aicore__ inline void GetTensorC(const LocalTensor<float>& outTensor);
    __aicore__ inline void CopyScaleToL1();
    __aicore__ inline void MatmulCompute(uint64_t roundIndex);
    __aicore__ inline void InitEventIds();
};

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, const void* tilingData)
{
    tiling_ = static_cast<const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData*>(tilingData);
    blockIdx_ = GetBlockIdx();
    UpdateGlobalAddr(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace);

    if constexpr (hasBias) {
        InitUbBuffs();
        SetFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[0]);
        SetFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[1]);
        SetFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[0]);
        SetFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[1]);
    }

    if ASCEND_IS_AIC {
        InitCubeBuffs();
        InitEventIds();
        wGlobalU8_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(weight));
        antiquantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(antiquantScale));
        kBlockSize_ = tiling_->matmulTiling.stepKa * tiling_->matmulTiling.baseK;
        totalKBlocks_ = CeilDiv<uint64_t>(tiling_->matmulTiling.Ka, kBlockSize_);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::InitCubeBuffs()
{
    uint64_t l1BufOffsetPart1 = 0;
    uint64_t l1BufOffsetPart2 = L1_BUFFER_HALF_SIZE;
    // 内轴 C0 对齐，外轴 16 对齐，baseN 在 tiling 时已保证 16 对齐
    uint64_t bL1DataSizes = (bTrans ? tiling_->matmulTiling.baseN :
                                      CeilAlign<uint64_t>(tiling_->matmulTiling.baseN, C0<wType>::value)) *
                            tiling_->matmulTiling.stepN *
                            CeilAlign<uint64_t>(tiling_->matmulTiling.baseK, bTrans ? C0<wType>::value : BLOCK_CUBE) *
                            tiling_->matmulTiling.stepKb;
    uint64_t bL1DataBytes = bL1DataSizes;
    if constexpr (AscendC::IsSupportB4<wType>()) {
        bL1DataBytes = bL1DataSizes >> 1;
    }
    // 内轴 C0 对齐，外轴 16 对齐，baseM 在 tiling 时已保证 16 对齐
    uint64_t aL1DataSize = tiling_->matmulTiling.baseM * tiling_->matmulTiling.stepM *
                           CeilAlign<uint64_t>(tiling_->matmulTiling.baseK, C0<xType>::value) *
                           tiling_->matmulTiling.stepKa;
    uint64_t scaleBytes = sizeof(uint64_t) * tiling_->matmulTiling.baseN;

    bL1PingLocalTensor_ = AscendC::LocalTensor<wType>(AscendC::QuePosition::TSCM, l1BufOffsetPart1, bL1DataSizes);
    l1BufOffsetPart1 += bL1DataBytes;
    bL1PongLocalTensor_ = AscendC::LocalTensor<wType>(AscendC::QuePosition::TSCM, l1BufOffsetPart2, bL1DataSizes);
    l1BufOffsetPart2 += bL1DataBytes;

    aL1PingLocalTensor_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart1, aL1DataSize);
    l1BufOffsetPart1 += aL1DataSize * sizeof(xType);
    aL1PongLocalTensor_ = AscendC::LocalTensor<xType>(AscendC::QuePosition::TSCM, l1BufOffsetPart2, aL1DataSize);
    l1BufOffsetPart2 += aL1DataSize * sizeof(xType);

    if constexpr (antiQuantType == QuantType::PER_CHANNEL) {
        scalePingLocalTensor_ = AscendC::LocalTensor<uint64_t>(AscendC::QuePosition::TSCM, l1BufOffsetPart1,
                                                               tiling_->matmulTiling.baseN);
        l1BufOffsetPart1 += scaleBytes;
        scalePongLocalTensor_ = AscendC::LocalTensor<uint64_t>(AscendC::QuePosition::TSCM, l1BufOffsetPart2,
                                                               tiling_->matmulTiling.baseN);
        l1BufOffsetPart2 += scaleBytes;
    }

    if constexpr (IsSameType<wType, fp4x2_e2m1_t>::value) {
        uint64_t sacleWL1DataSize = tiling_->matmulTiling.stepKa * tiling_->matmulTiling.baseN *
                                    tiling_->matmulTiling.baseK / static_cast<uint64_t>(32); // 32: Mx 固定的groupSize
        scaleWL1LocalBuf_ = AscendC::LocalTensor<fp8_e8m0_t>(AscendC::QuePosition::TSCM, l1BufOffsetPart1,
                                                             sacleWL1DataSize);
        // 32：每次迭代处理blockNum * 32B，blockNum = sacleWL1DataBytes / 32，32639 => 0111111101111111
        AscendC::InitConstValue(scaleWL1LocalBuf_.template ReinterpretCast<uint16_t>(),
                                {1, static_cast<uint16_t>(sacleWL1DataSize * sizeof(fp8_e8m0_t) / 32), 0, 32639U});
        l1BufOffsetPart1 += sacleWL1DataSize * sizeof(fp8_e8m0_t);
    }

    cL0Tensor_ = AscendC::LocalTensor<float>(TPosition::CO1, 0, L0C_BUFFER_SIZE / sizeof(float));
    aL0Tensor_ = AscendC::LocalTensor<xType>(TPosition::A2, 0, L0A_BUFFER_SIZE / sizeof(xType));
    if constexpr (AscendC::IsSupportB4<wType>()) {
        bL0Tensor_ = AscendC::LocalTensor<wType>(TPosition::B2, 0, L0B_BUFFER_SIZE << 1);
    } else {
        bL0Tensor_ = AscendC::LocalTensor<wType>(TPosition::B2, 0, L0B_BUFFER_SIZE / sizeof(wType));
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::InitEventIds()
{
    for (int idx = 0; idx < DOUBLE_BUFFER; ++idx) {
        SetFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
        SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
        if constexpr (!bTrans) {
            SetFlag<HardEvent::FIX_M>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::InitUbBuffs()
{
    uint64_t ubBufOffset = 0;
    ubElements_ = tiling_->matmulTiling.baseM * tiling_->matmulTiling.baseN;
    // 64 多分配256B空间，避免非对齐时fixpipe载入时越界
    ubCTensorSize_ = ubElements_ + 64;
    cLocalTensor_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, ubBufOffset, ubCTensorSize_ * DOUBLE_BUFFER);
    ubBufOffset += ubCTensorSize_ * DOUBLE_BUFFER * sizeof(float);
    yLocalTensor_ = AscendC::LocalTensor<yType>(AscendC::TPosition::VECOUT, ubBufOffset, ubElements_ * DOUBLE_BUFFER);
    ubBufOffset += ubElements_ * DOUBLE_BUFFER * sizeof(yType);
    // 64 多分配256B空间，避免非对齐时最后一轮VF载入时越界
    biasLocalTensor_ = AscendC::LocalTensor<biasType>(AscendC::TPosition::VECIN, ubBufOffset,
                                                      tiling_->matmulTiling.baseN * DOUBLE_BUFFER + 64);
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::UpdateGlobalAddr(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace)
{
    block_.Init(tiling_, blockIdx_);

    if constexpr (antiQuantType == QuantType::PER_TENSOR) { // pertensor
        block_.offset_.scaleScalar = *((__gm__ uint64_t*)antiquantScale);
    } else {
        scaleGlobal_.SetGlobalBuffer((__gm__ uint64_t*)antiquantScale);
    }

    // update global buffer
    aGlobal_.SetGlobalBuffer((__gm__ xType*)x);
    bGlobal_.SetGlobalBuffer((__gm__ wType*)weight);
    cGlobal_.SetGlobalBuffer((__gm__ yType*)y);
    if constexpr (hasBias) {
        biasGlobal_.SetGlobalBuffer((__gm__ biasType*)bias);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::Process()
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
        if constexpr (hasBias) {
            if ASCEND_IS_AIC {
                if (roundIndex > 1) {
                    WaitForVector();
                }
                MatmulCompute(roundIndex);
                NotifyVector();
            }

            if ASCEND_IS_AIV {
                WaitForCube();
                AddBias(roundIndex);
                if (block_.params_.round > 2 && roundIndex < block_.params_.round - 2) {
                    NotifyCube();
                }
            }
        } else {
            MatmulCompute(roundIndex);
        }
        roundDbFlag_ ^= 1;
    }
    EndSync();
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::MatmulCompute(
    uint64_t roundIndex)
{
    ProcessBaseBlockCompute(roundIndex);
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::ProcessBaseBlockCompute(
    uint64_t roundIndex)
{
    uint64_t kOffset = 0;
    if constexpr (!bTrans) {
        WaitFlag<HardEvent::FIX_M>(MTE_SYNC_DB_FLAG_IDS_1[roundDbFlag_]);
    }
    for (uint64_t kOutIdx = 0; kOutIdx < totalKBlocks_; kOutIdx++) {
        GetABToL1(kOutIdx);

        uint64_t curKSize = Min<uint64_t>(tiling_->matmulTiling.Ka - kOutIdx * kBlockSize_, kBlockSize_);
        uint64_t kInnerBlockCount = CeilDiv<uint64_t>(curKSize, tiling_->matmulTiling.baseK);

        SetFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);

        for (uint64_t kInnerIdx = 0; kInnerIdx < kInnerBlockCount; kInnerIdx++) {
            IterateMatmul(kOutIdx, kInnerIdx, kOffset);
            kOffset += tiling_->matmulTiling.baseK;
        }
        SetFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);

        l1DbFlag_ = l1DbFlag_ ^ 1;
    }
    if constexpr (antiQuantType == QuantType::PER_CHANNEL) {
        CopyScaleToL1();
    }
    if constexpr (!bTrans) {
        SetFlag<HardEvent::M_FIX>(MTE_SYNC_DB_FLAG_IDS_1[roundDbFlag_]);
        WaitFlag<HardEvent::M_FIX>(MTE_SYNC_DB_FLAG_IDS_1[roundDbFlag_]);
    }
    if constexpr (hasBias) {
        GetTensorC(cLocalTensor_[ubCTensorSize_ * (roundIndex & 1)]);
    } else {
        GetTensorC(cGlobal_[block_.offset_.offsetC]);
    }
    if constexpr (!bTrans) {
        SetFlag<HardEvent::FIX_M>(MTE_SYNC_DB_FLAG_IDS_1[roundDbFlag_]);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::CopyScaleToL1()
{
    AscendC::DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    // 32B对齐
    dataCopyParams.blockLen = CeilDiv<uint64_t>(sizeof(uint64_t) * block_.params_.singleCoreN, 32);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    LocalTensor<uint64_t> scaleLocalTensor = roundDbFlag_ == 0 ? scalePingLocalTensor_ : scalePongLocalTensor_;

    SetFlag<HardEvent::FIX_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[roundDbFlag_]);
    WaitFlag<HardEvent::FIX_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[roundDbFlag_]);
    DataCopy(scaleLocalTensor, antiquantScaleGlobal_[block_.offset_.offsetN], dataCopyParams, tiling_->enablePreAlloc);
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::GetABToL1(uint64_t kOutIdx)
{
    WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[l1DbFlag_]);
    CopyXNd2Nz(kOutIdx);
    CopyWeightGm2L1(kOutIdx);
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::CopyXNd2Nz(uint64_t kOutIdx)
{
    // A 矩阵 K 方向搬入 L1 Size 计算
    uint64_t singleKSize = tiling_->matmulTiling.stepKa * tiling_->matmulTiling.baseK;
    uint64_t kOffset = kOutIdx * singleKSize;
    uint64_t kL1Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, singleKSize);
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

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::CopyWeightGm2L1(
    uint64_t kOutIdx)
{
    // B 矩阵 K 方向搬入 L1 Size 计算
    uint64_t singleKSize = tiling_->matmulTiling.stepKb * tiling_->matmulTiling.baseK;
    uint64_t kOffset = kOutIdx * singleKSize;
    uint64_t kL1Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, singleKSize);
    // B 矩阵 N 方向搬入L1 Size计算
    uint64_t nL1Size = Min<uint64_t>(tiling_->matmulTiling.N - block_.offset_.offsetN, block_.params_.singleCoreN);
    uint64_t weightGmOffset;
    if constexpr (weightFormat == CubeFormat::ND) {
        weightGmOffset = bTrans ? (block_.offset_.offsetN * tiling_->matmulTiling.Kb + kOffset) :
                                  (kOffset * tiling_->matmulTiling.N + block_.offset_.offsetN);
    } else {
        // 转置的 B 矩阵在 GM 上搬运的偏移计算 (int4 需要除以 2)：kOffset * problem_n + nOffset * C0
        weightGmOffset = (kOffset * tiling_->matmulTiling.N + block_.offset_.offsetN * C0<wType>::value);
    }
    if constexpr (AscendC::IsSupportB4<wType>()) {
        weightGmOffset = weightGmOffset >> 1;
    }
    if constexpr (weightFormat == CubeFormat::NZ) {
        AscendC::DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = CeilDiv<uint64_t>(kL1Size, C0<wType>::value);
        dataCopyParams.blockLen = nL1Size;
        dataCopyParams.srcStride = tiling_->matmulTiling.N - nL1Size;
        dataCopyParams.dstStride = CeilAlign<uint64_t>(nL1Size, static_cast<uint64_t>(AscendC::BLOCK_CUBE)) - nL1Size;
        AscendC::DataCopy((l1DbFlag_ == IDX_0) ? bL1PingLocalTensor_.template ReinterpretCast<uint8_t>() :
                                                 bL1PongLocalTensor_.template ReinterpretCast<uint8_t>(),
                          wGlobalU8_[weightGmOffset], dataCopyParams, tiling_->enablePreAlloc);
    } else {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = bTrans ? nL1Size : kL1Size;
        nd2nzParams.dValue = bTrans ? kL1Size : nL1Size;
        nd2nzParams.srcDValue = bTrans ? tiling_->matmulTiling.Kb : tiling_->matmulTiling.N;
        if constexpr (AscendC::IsSupportB4<wType>()) {
            nd2nzParams.dValue = nd2nzParams.dValue >> 1;
            nd2nzParams.srcDValue = nd2nzParams.srcDValue >> 1;
        }
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.dstNzC0Stride = CeilAlign<uint64_t>(nd2nzParams.nValue, BLOCK_CUBE);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        DataCopy((l1DbFlag_ == IDX_0) ? bL1PingLocalTensor_.template ReinterpretCast<uint8_t>() :
                                        bL1PongLocalTensor_.template ReinterpretCast<uint8_t>(),
                 wGlobalU8_[weightGmOffset], nd2nzParams, tiling_->enablePreAlloc);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::IterateMatmul(
    uint64_t kOutIdx, uint64_t kInnerIdx, uint64_t kOffset)
{
    if ASCEND_IS_AIC {
        WeightQuantBatchMatmul::Arch35::L0CopyAndCalcParams l0CopyAndCalcParams;
        l0CopyAndCalcParams.mL0Size = block_.params_.singleCoreM;
        l0CopyAndCalcParams.nL0Size = CeilAlign<uint64_t>(block_.params_.singleCoreN,
                                                          bTrans ? BLOCK_CUBE : C0<wType>::value);
        l0CopyAndCalcParams.nL1Size = l0CopyAndCalcParams.nL0Size;
        l0CopyAndCalcParams.mL1Size = l0CopyAndCalcParams.mL0Size;
        l0CopyAndCalcParams.kL1Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOutIdx * kBlockSize_, kBlockSize_);
        l0CopyAndCalcParams.kL0Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOffset, tiling_->matmulTiling.baseK);
        l0CopyAndCalcParams.isFirstKLoop = kOutIdx == 0 && kInnerIdx == 0;
        l0CopyAndCalcParams.isLastKLoop = tiling_->matmulTiling.Ka - kOffset <= tiling_->matmulTiling.baseK;
        uint64_t aL1Offset = kInnerIdx * tiling_->matmulTiling.baseK *
                             CeilAlign<uint64_t>(l0CopyAndCalcParams.mL0Size, BLOCK_CUBE);
        uint64_t bL1Offset = bTrans ? kInnerIdx * tiling_->matmulTiling.baseK *
                                          CeilAlign<uint64_t>(l0CopyAndCalcParams.nL0Size, BLOCK_CUBE) :
                                      kInnerIdx * tiling_->matmulTiling.baseK * C0<wType>::value;
        WaitFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l0DbFlag_]);
        LocalTensor<xType> aL0Tensor = aL0Tensor_[l0DbFlag_ << 14];
        WeightQuantBatchMatmul::Arch35::LoadAL1ToL0<xType, xType>(
            aL0Tensor, (l1DbFlag_ == IDX_0) ? aL1PingLocalTensor_[aL1Offset] : aL1PongLocalTensor_[aL1Offset],
            l0CopyAndCalcParams);
        LocalTensor<wType> bL0Tensor;
        if constexpr (AscendC::IsSupportB4<wType>()) {
            bL0Tensor = bL0Tensor_[l0DbFlag_ << 16];
        } else {
            bL0Tensor = bL0Tensor_[l0DbFlag_ << 15];
        }
        if constexpr (IsSameType<wType, fp4x2_e2m1_t>::value) {
            l0CopyAndCalcParams.scaleKL1Size = kBlockSize_ / 32;
            WeightQuantBatchMatmul::Arch35::LoadBAndScaleL1ToL0<wType, wType, fp8_e8m0_t>(
                bL0Tensor, (l1DbFlag_ == IDX_0) ? bL1PingLocalTensor_[bL1Offset] : bL1PongLocalTensor_[bL1Offset],
                scaleWL1LocalBuf_, l0CopyAndCalcParams);
        } else {
            WeightQuantBatchMatmul::Arch35::LoadBL1ToL0<wType, wType, bTrans>(
                bL0Tensor, (l1DbFlag_ == IDX_0) ? bL1PingLocalTensor_[bL1Offset] : bL1PongLocalTensor_[bL1Offset],
                l0CopyAndCalcParams);
        }

        LocalTensor<float> cL0Tensor = cL0Tensor_[roundDbFlag_ << 15];
        SetFlag<HardEvent::MTE1_M>(MTE_SYNC_FLAG_ID_6);
        WaitFlag<HardEvent::MTE1_M>(MTE_SYNC_FLAG_ID_6);
        if constexpr (bTrans) {
            WeightQuantBatchMatmul::Arch35::MmadCompute<float, xType, wType>(cL0Tensor, aL0Tensor, bL0Tensor,
                                                                             l0CopyAndCalcParams);
        } else {
            WeightQuantBatchMatmul::Arch35::MmadCompute<float, xType, wType, false>(cL0Tensor, aL0Tensor, bL0Tensor,
                                                                                    l0CopyAndCalcParams);
        }
        SetFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[l0DbFlag_]);

        l0DbFlag_ ^= 1;
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::GetTensorC(
    const GlobalTensor<yType>& outTensor)
{
    LocalTensor<float> cL0Tensor = cL0Tensor_[(roundDbFlag_) << 15];
    WeightQuantBatchMatmul::Arch35::FixL0CToDstParams fixL0CToDstParams;
    fixL0CToDstParams.mL0Size = block_.params_.singleCoreM;
    fixL0CToDstParams.nL0Size = block_.params_.singleCoreN;
    fixL0CToDstParams.outNSize = tiling_->matmulTiling.N;
    if constexpr (antiQuantType == QuantType::PER_CHANNEL) {
        SetFlag<HardEvent::MTE2_FIX>(MTE_SYNC_DB_FLAG_IDS_2[roundDbFlag_]);
        WaitFlag<HardEvent::MTE2_FIX>(MTE_SYNC_DB_FLAG_IDS_2[roundDbFlag_]);
        WeightQuantBatchMatmul::Arch35::FixL0CToDst<yType, float, bTrans>(
            outTensor, cL0Tensor, roundDbFlag_ == 0 ? scalePingLocalTensor_ : scalePongLocalTensor_, fixL0CToDstParams);
    } else {
        WeightQuantBatchMatmul::Arch35::FixL0CToDst<yType, float, bTrans>(
            outTensor, cL0Tensor, block_.offset_.scaleScalar, fixL0CToDstParams);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::GetTensorC(
    const LocalTensor<float>& outTensor)
{
    LocalTensor<float> cL0Tensor = cL0Tensor_[roundDbFlag_ << 15];
    WeightQuantBatchMatmul::Arch35::FixL0CToDstParams fixL0CToDstParams;
    fixL0CToDstParams.mL0Size = block_.params_.singleCoreM;
    fixL0CToDstParams.nL0Size = CeilAlign<uint64_t>(block_.params_.singleCoreN, BLOCK_CUBE);
    fixL0CToDstParams.outNSize = CeilAlign<uint64_t>(block_.params_.singleCoreN, BLOCK_CUBE);
    if constexpr (antiQuantType == QuantType::PER_CHANNEL) {
        SetFlag<HardEvent::MTE2_FIX>(MTE_SYNC_DB_FLAG_IDS_2[roundDbFlag_]);
        WaitFlag<HardEvent::MTE2_FIX>(MTE_SYNC_DB_FLAG_IDS_2[roundDbFlag_]);
        WeightQuantBatchMatmul::Arch35::FixL0CToDst<float, float, false, bTrans>(
            outTensor, cL0Tensor, roundDbFlag_ == 0 ? scalePingLocalTensor_ : scalePongLocalTensor_, fixL0CToDstParams);
    } else {
        WeightQuantBatchMatmul::Arch35::FixL0CToDst<float, float, false, bTrans>(
            outTensor, cL0Tensor, block_.offset_.scaleScalar, fixL0CToDstParams);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::EndSync()
{
    if ASCEND_IS_AIV {
        WaitFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[0]);
        WaitFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[1]);
        WaitFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[0]);
        WaitFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[1]);
    }

    if ASCEND_IS_AIC {
        for (int idx = 0; idx < DOUBLE_BUFFER; ++idx) {
            WaitFlag<HardEvent::M_MTE1>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            WaitFlag<HardEvent::MTE1_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            if constexpr (!bTrans) {
                WaitFlag<HardEvent::FIX_M>(MTE_SYNC_DB_FLAG_IDS_1[idx]);
            }
        }
    }
}

template <typename yType, typename biasType>
__aicore__ inline void AddBiasVf(const BiasAddParam<yType, biasType>& param)
{
    MicroAPI::RegTensor<float> cFloatRegTensor;

    MicroAPI::RegTensor<biasType> biasRegTensor;
    MicroAPI::RegTensor<float> biasFloatRegTensor;

    MicroAPI::RegTensor<yType> addResRegTensor;
    MicroAPI::RegTensor<float> addResFloatRegTensor;

    MicroAPI::MaskReg yTypeCalcMask = MicroAPI::CreateMask<yType, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg floatCastMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    __local_mem__ float* cAddr = param.cAddr;
    __local_mem__ yType* yAddr = param.yAddr;
    __local_mem__ biasType* biasAddr = param.biasAddr;

    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    uint16_t vLF32Elements = 64;
    for (uint16_t nIdx = 0; nIdx < param.nCount; nIdx++) {
        uint32_t nTileSize = Min<uint16_t>(param.ubAlignN - nIdx * vLF32Elements, vLF32Elements);
        MicroAPI::MaskReg yTypeCopyMask = MicroAPI::UpdateMask<float>(nTileSize);

        MicroAPI::AddrReg biasAddrReg = MicroAPI::CreateAddrReg<biasType>(nIdx, vLF32Elements);
        if constexpr (IsSameType<biasType, float>::value) {
            MicroAPI::DataCopy<biasType, MicroAPI::LoadDist::DIST_NORM>(biasFloatRegTensor, biasAddr, biasAddrReg);
        } else {
            MicroAPI::DataCopy<biasType, MicroAPI::LoadDist::DIST_US_B16>(biasRegTensor, biasAddr, biasAddrReg);
            MicroAPI::Cast<float, yType, castTrait>(biasFloatRegTensor, biasRegTensor, yTypeCalcMask);
        }
        for (uint16_t mIdx = 0; mIdx < param.mCount; mIdx++) {
            MicroAPI::AddrReg cAddrReg = MicroAPI::CreateAddrReg<float>(nIdx, vLF32Elements, mIdx, param.ubAlignN);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(cFloatRegTensor, cAddr, cAddrReg);
            MicroAPI::Add(addResFloatRegTensor, cFloatRegTensor, biasFloatRegTensor, yTypeCalcMask);
            MicroAPI::Cast<yType, float, castTrait>(addResRegTensor, addResFloatRegTensor, floatCastMask);
            MicroAPI::AddrReg yAddrReg = MicroAPI::CreateAddrReg<yType>(nIdx, vLF32Elements, mIdx, param.ubAlignN);
            MicroAPI::DataCopy<yType, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr, addResRegTensor, yAddrReg,
                                                                          yTypeCopyMask);
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW
__aicore__ inline void WeightQuantBatchMatmulV2ASWKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW>::AddBias(uint64_t roundIndex)
{
    LocalTensor<yType> yLocalTensor = yLocalTensor_[ubElements_ * (roundIndex & 1)];
    LocalTensor<biasType> biasLocalTensor = biasLocalTensor_[tiling_->matmulTiling.baseN * (roundIndex & 1)];
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = block_.params_.singleCoreN * sizeof(biasType);
    params.srcStride = (tiling_->matmulTiling.N - block_.params_.singleCoreN) * sizeof(biasType);
    params.dstStride = 0;
    DataCopyPadExtParams<biasType> padParams;
    WaitFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[roundIndex & 1]);
    DataCopyPad(biasLocalTensor, biasGlobal_[block_.offset_.offsetBias], params, padParams, true);

    BiasAddParam<yType, biasType> param;
    param.mCount = block_.params_.singleCoreM;
    param.ubAlignN = CeilAlign<uint64_t>(block_.params_.singleCoreN, BLOCK_CUBE);
    // VF每次处理64个数字
    param.nCount = CeilDiv<uint16_t>(param.ubAlignN, 64);
    param.cAddr = (__local_mem__ float*)cLocalTensor_[ubCTensorSize_ * (roundIndex & 1)].GetPhyAddr();
    param.yAddr = (__local_mem__ yType*)yLocalTensor.GetPhyAddr();
    param.biasAddr = (__local_mem__ biasType*)biasLocalTensor.GetPhyAddr();

    SetFlag<HardEvent::MTE2_V>(MTE_SYNC_FLAG_ID_6);
    WaitFlag<HardEvent::MTE2_V>(MTE_SYNC_FLAG_ID_6);

    WaitFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[roundIndex & 1]);

    AscendC::VF_CALL<AddBiasVf<yType, biasType>>(param);
    SetFlag<HardEvent::V_MTE2>(MTE_SYNC_DB_FLAG_IDS_1[roundIndex & 1]);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = block_.params_.singleCoreN * sizeof(yType);
    dataCopyParams.blockCount = block_.params_.singleCoreM;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (tiling_->matmulTiling.N - block_.params_.singleCoreN) * sizeof(yType);

    SetFlag<HardEvent::V_MTE3>(MTE_SYNC_FLAG_ID_6);
    WaitFlag<HardEvent::V_MTE3>(MTE_SYNC_FLAG_ID_6);

    DataCopyPad(cGlobal_[block_.offset_.offsetC], yLocalTensor, dataCopyParams);
    SetFlag<HardEvent::MTE3_V>(MTE_SYNC_DB_FLAG_IDS_1[roundIndex & 1]);
}
} // namespace WeightQuantBatchMatmulV2::Arch35
