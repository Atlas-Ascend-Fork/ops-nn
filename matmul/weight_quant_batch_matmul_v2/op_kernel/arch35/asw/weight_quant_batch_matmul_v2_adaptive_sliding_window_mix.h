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
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window_mix.h
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

using AscendC::BLOCK_CUBE;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::int4b_t;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TEventID;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::WaitFlag;
namespace MicroAPI = AscendC::MicroAPI;
using AscendC::fp8_e8m0_t;
using AscendC::IsSameType;
using matmul::MatmulImpl;
using matmul::MatmulType;
using FixpipeParams = AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR>;
using WeightQuantBatchMatmul::Arch35::FixL0CToDstParams;
using WeightQuantBatchMatmul::Arch35::L0CopyAndCalcParams;

#ifndef LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
#define LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX                                                                            \
    template <typename xType, typename wType, typename biasType, typename yType, bool bTrans, QuantType antiQuantType, \
              bool hasAntiQuantOffset, QuantType quantType, bool hasBias, bool useL14Buff, bool useUb4Buff>
#endif
#ifndef LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX
#define LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX \
    xType, wType, biasType, yType, bTrans, antiQuantType, hasAntiQuantOffset, quantType, hasBias, useL14Buff, useUb4Buff
#endif

namespace WeightQuantBatchMatmulV2::Arch35 {

template <typename T>
__aicore__ inline constexpr bool IsSupportB8Type()
{
    return AscendC::SupportType<T, int8_t, fp8_e4m3fn_t>();
}

template <typename yType, typename biasType = yType>
struct ScaleAccParam {
    uint64_t mCount;
    uint64_t nCount;

    __local_mem__ float* addAddr;
    __local_mem__ float* accAddr;
    __local_mem__ yType* scaleAddr;
    __local_mem__ biasType* biasAddr;
    __local_mem__ float* destUbAddr;

    __local_mem__ yType* castDestAddr;
};

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
class WeightQuantBatchMatmulV2ASWMixKernel {
public:
    __aicore__ inline WeightQuantBatchMatmulV2ASWMixKernel(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const void* tilingData, TPipe* tPipe);
    __aicore__ inline void InitInputs(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR y);
    __aicore__ inline void InitL1Buffs();
    __aicore__ inline void InitL0Buffs();
    __aicore__ inline void InitUbDoubleBuffs();
    __aicore__ inline void InitUbQuadrupleBuffs();

    __aicore__ inline void Process();
    __aicore__ inline void ProcessBaseBlockCompute(uint64_t mOffset, uint64_t nOffset, uint64_t kBlockSize,
                                                   uint64_t totalKBlocks);
    __aicore__ inline void GetABToL1(uint64_t mOffset, uint64_t nOffset, uint64_t kOffset);
    __aicore__ inline void CopyXNd2Nz(uint64_t nOffset, uint64_t kOffset);
    __aicore__ inline void CopyWeightGm2L1(uint64_t nOffset, uint64_t kOffset);
    __aicore__ inline void GetABToL1QuadrupleBuff(uint64_t mOffset, uint64_t nOffset, uint64_t kOutIdx);
    __aicore__ inline void ProcessSingleBlock(uint64_t mOffset, uint64_t nOffset, uint64_t kInnerIdx);
    __aicore__ inline void CopyResToGM(uint64_t mOffset, uint64_t nOffset);
    __aicore__ inline void UpdateMmParams(uint64_t mOffset, uint64_t nOffset, L0CopyAndCalcParams& l0CopyAndCalcParams,
                                          FixL0CToDstParams& fixL0CToDstParams);
    __aicore__ inline void IterateMatmul(uint64_t mOffset, uint64_t nOffset, uint64_t kOutIdx, uint64_t kInnerIdx,
                                         uint64_t singleK);
    __aicore__ inline void IterateMatmulForQuadrupleBuff(uint64_t kInnerIdx,
                                                         const L0CopyAndCalcParams& l0CopyAndCalcParams,
                                                         const FixL0CToDstParams& fixL0CToDstParams);
    __aicore__ inline void CastAndCopyToGm(uint64_t nOffset, uint64_t kOffset);
    __aicore__ inline void CopyScaleToUB(uint64_t nOffset, uint64_t kOffset);
    __aicore__ inline void CopyBiasToUB(uint64_t nOffset);
    __aicore__ inline void GetABL1ToL0();
    __aicore__ inline void LoadToL0AB(const LocalTensor<xType>& aL1Tensor, const LocalTensor<wType>& bL1Tensor);
    __aicore__ inline void GetTensorC(const LocalTensor<float>& dstTensor, uint64_t tailM, uint64_t tailN,
                                      uint64_t tailMAlign, uint64_t tailNAlign);

public:
    using inputXType = MatmulType<TPosition::TSCM, CubeFormat::NZ, xType, false, LayoutMode::NONE, false,
                                  TPosition::GM>;
    using inputWType = MatmulType<TPosition::TSCM, CubeFormat::NZ, wType, bTrans, LayoutMode::NONE, false,
                                  TPosition::GM>;
    using cType = MatmulType<TPosition::VECIN, CubeFormat::ND, float>;
    using inputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, yType>;
    TPipe* pipe_;
    const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData* tiling_;

protected:
    static constexpr int32_t IDX_0 = 0;
    static constexpr int32_t IDX_1 = 1;
    static constexpr int32_t IDX_2 = 2;
    static constexpr int32_t IDX_3 = 3;
    static constexpr uint64_t L1_BUFFER_SIZE = 512 * 1024;
    static constexpr uint64_t L0A_BUFFER_SIZE = 64 * 1024;
    static constexpr uint64_t L0B_BUFFER_SIZE = 64 * 1024;
    static constexpr uint64_t L0C_BUFFER_SIZE = 256 * 1024;
    static constexpr uint64_t BIAS_TABLE_SIZE = 4096;
    static constexpr uint64_t L1_BUFFER_HALF_SIZE = 256 * 1024;
    static constexpr uint64_t L1_BUFFER_NUM = useL14Buff ? QUADRUPLE_BUFFER_NUM : DOUBLE_BUFFER;
    static constexpr uint64_t L1_BUFFER_MASK = L1_BUFFER_NUM - 1;
    static constexpr uint64_t PIPE_V_TO_FIX_SYNC_START = useUb4Buff ? 3 : 1;
    int32_t curBlockIdx_;

    uint64_t dbFlag_ = 0;
    uint64_t scaleDbFlag_ = 0;
    uint64_t l0BufIdx_ = 0;
    uint8_t curL1BufIdx_ = 0;
    uint64_t l0LoadKSize_ = 0;
    GlobalTensor<xType> xGlobal_;
    GlobalTensor<uint8_t> wGlobalU8_;
    GlobalTensor<xType> antiquantScaleGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<yType> yGlobal_;
    WeightQuantBatchMatmulV2::Arch35::WeightQuantBmmAswBlock block_;

    // A、B矩阵L1默认开Double Buffer
    LocalTensor<xType> aL1LocalBuf0_;
    LocalTensor<xType> aL1LocalBuf1_;
    LocalTensor<xType> aL1LocalBuf2_;
    LocalTensor<xType> aL1LocalBuf3_;
    LocalTensor<wType> bL1LocalBuf0_;
    LocalTensor<wType> bL1LocalBuf1_;
    LocalTensor<wType> bL1LocalBuf2_;
    LocalTensor<wType> bL1LocalBuf3_;
    LocalTensor<fp8_e8m0_t> scaleWL1LocalBuf_;

    LocalTensor<float> cL0Tensor_;
    LocalTensor<xType> aL0Tensor_;
    LocalTensor<wType> bL0Tensor_;
    LocalTensor<float> biasBtTensor_;

    // MTE1_MTE2同步：A、B矩阵用同一组
    AscendC::TEventID eventIdsMte1ToMte2_[QUADRUPLE_BUFFER_NUM];

    // MTE2_MTE1同步：A、B矩阵用同一组
    AscendC::TEventID eventIdsMte2ToMte1_[QUADRUPLE_BUFFER_NUM];

    TEventID eventIdUbPingVToMte2_;
    TEventID eventIdUbPongVToMte2_;
    TEventID eventIdScaleMte2ToV_;

    TEventID eventIdBiasVToMte2_;
    TEventID eventIdBiasMte2ToV_;

    LocalTensor<xType> scalePingLocalTensor_;
    LocalTensor<xType> scalePongLocalTensor_;
    LocalTensor<biasType> biasUbTensor_;

    // FixPipe等Vector的核间同步总次数
    uint64_t pipeVToFixSyncCount_ = 0;

    // 放在 bank0 和 bank1 的累加和结果
    LocalTensor<float> accLocalTensor_;
    // 放在 bank2 和 bank3 的 fixpipe 输出结果
    LocalTensor<float> addLocalTensor_;
    uint64_t add2Offset_ = 0;

    LocalTensor<yType> castPingLocalTensor_;
    LocalTensor<yType> castPongLocalTensor_;

    LocalTensor<yType> resUbTensor_;

    uint64_t kIterateIdx_ = 0;
    uint64_t singleCoreKCount_ = 0;
    TEventID eventIdsMToMte1_[2];

    uint64_t bL1DataSizes_;
    uint64_t aL1DataSizes_;

    uint64_t curMSize_;
    uint64_t curNSize_;
    uint64_t curMAlignSize_;
    uint64_t curNAlignSize_;
    uint64_t curKL1Size_;
    uint64_t singleKSize_;
};

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, const void* tilingData, TPipe* tPipe)
{
    tiling_ = static_cast<const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData*>(tilingData);
    pipe_ = tPipe;

    curBlockIdx_ = GetBlockIdx();

    InitInputs(x, weight, antiquantScale, bias, y);
    InitL1Buffs();
    InitL0Buffs();
    if constexpr (useUb4Buff) {
        InitUbQuadrupleBuffs();
    } else {
        InitUbDoubleBuffs();
    }

    // 申请 AL1 和 BL1 的 MTE1_MTE2同步事件ID
    if ASCEND_IS_AIC {
        for (uint64_t idx = 0; idx < L1_BUFFER_NUM; idx++) {
            eventIdsMte1ToMte2_[idx] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_[idx]);

            eventIdsMte2ToMte1_[idx] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>();
        }
    }

    if ASCEND_IS_AIV {
        eventIdUbPingVToMte2_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(eventIdUbPingVToMte2_);
        eventIdUbPongVToMte2_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(eventIdUbPongVToMte2_);
        eventIdScaleMte2ToV_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        if constexpr (hasBias) {
            eventIdBiasVToMte2_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
            SetFlag<HardEvent::V_MTE2>(eventIdBiasVToMte2_);
            eventIdBiasMte2ToV_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        }
    }

    singleCoreKCount_ = CeilDiv<uint64_t>(tiling_->matmulTiling.Ka, tiling_->matmulTiling.baseK);
    if constexpr (useUb4Buff) {
        // 4为double buff，最后四次不用设置同步
        pipeVToFixSyncCount_ = singleCoreKCount_ * block_.params_.round > 4 ?
                                   singleCoreKCount_ * block_.params_.round - 4 :
                                   0;
    } else {
        // 2为double buff，最后两次不用设置同步
        pipeVToFixSyncCount_ = singleCoreKCount_ * block_.params_.round > 2 ?
                                   singleCoreKCount_ * block_.params_.round - 2 :
                                   0;
    }
}

/*
 *  功能：申请各输入的 global buffer
 */
LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::InitInputs(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR y)
{
    block_.Init(tiling_, curBlockIdx_);

    // 申请 Global Buffer
    if ASCEND_IS_AIC {
        xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x));
        wGlobalU8_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(weight));
    }

    if ASCEND_IS_AIV {
        antiquantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(antiquantScale));
        if constexpr (hasBias) {
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType*>(bias));
        }
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y));
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::InitL1Buffs()
{
    singleKSize_ = tiling_->matmulTiling.baseK * tiling_->matmulTiling.stepKa;
    if ASCEND_IS_AIV {
        return;
    }

    // 在 L1 Buffer 上为 AL1 和 BL1 申请空间
    uint64_t l1BufOffsetPart1 = 0;
    uint64_t l1BufOffsetPart2 = L1_BUFFER_HALF_SIZE;

    /* |BL0|BL2|AL0|AL2|BL1|BL3|AL1|AL3| */
    // B矩阵L1 Buff分配
    uint64_t bL1DataSizes = tiling_->matmulTiling.baseN * tiling_->matmulTiling.stepN * singleKSize_;
    uint64_t bL1DataBytes;
    if constexpr (IsSupportB8Type<wType>()) {
        bL1DataBytes = bL1DataSizes;
    } else {
        bL1DataBytes = bL1DataSizes >> 1; // BL1 是 bit4 类型
    }

    bL1LocalBuf0_ = LocalTensor<wType>(TPosition::TSCM, l1BufOffsetPart1, bL1DataSizes);
    l1BufOffsetPart1 += bL1DataBytes;

    bL1LocalBuf1_ = LocalTensor<wType>(TPosition::TSCM, l1BufOffsetPart2, bL1DataSizes);
    l1BufOffsetPart2 += bL1DataBytes;

    if constexpr (useL14Buff) {
        bL1LocalBuf2_ = LocalTensor<wType>(TPosition::TSCM, l1BufOffsetPart1, bL1DataSizes);
        l1BufOffsetPart1 += bL1DataBytes;

        bL1LocalBuf3_ = LocalTensor<wType>(TPosition::TSCM, l1BufOffsetPart2, bL1DataSizes);
        l1BufOffsetPart2 += bL1DataBytes;
    }

    // A矩阵L1 Buff分配
    uint64_t aL1DataSizes = tiling_->matmulTiling.baseM * tiling_->matmulTiling.stepM * singleKSize_;
    uint64_t aL1DataBytes = aL1DataSizes * sizeof(xType);

    aL1LocalBuf0_ = LocalTensor<xType>(TPosition::TSCM, l1BufOffsetPart1, aL1DataSizes);
    l1BufOffsetPart1 += aL1DataBytes;

    aL1LocalBuf1_ = LocalTensor<xType>(TPosition::TSCM, l1BufOffsetPart2, aL1DataSizes);
    l1BufOffsetPart2 += aL1DataBytes;

    if constexpr (useL14Buff) {
        aL1LocalBuf2_ = LocalTensor<xType>(TPosition::TSCM, l1BufOffsetPart1, aL1DataSizes);
        l1BufOffsetPart1 += aL1DataBytes;

        aL1LocalBuf3_ = LocalTensor<xType>(TPosition::TSCM, l1BufOffsetPart2, aL1DataSizes);
        l1BufOffsetPart2 += aL1DataBytes;
    }

    if constexpr (IsSameType<wType, fp4x2_e2m1_t>::value) {
        uint64_t scaleWL1DataSizes = tiling_->matmulTiling.baseN * tiling_->matmulTiling.baseK /
                                     static_cast<uint64_t>(32); // mx 规定 groupsize 为 32
        if constexpr (useL14Buff) {
            scaleWL1DataSizes *= tiling_->matmulTiling.stepKb;
        }
        uint64_t sacleWL1DataBytes = scaleWL1DataSizes * sizeof(fp8_e8m0_t);
        scaleWL1LocalBuf_ = LocalTensor<fp8_e8m0_t>(TPosition::TSCM, l1BufOffsetPart2, scaleWL1DataSizes);
        // 32：每次迭代处理blockNum * 32B，blockNum = sacleWL1DataBytes / 32，32639 => 0111111101111111
        AscendC::InitConstValue(scaleWL1LocalBuf_.template ReinterpretCast<uint16_t>(),
                                {1, static_cast<uint16_t>(sacleWL1DataBytes / 32), 0, 32639U});
        l1BufOffsetPart2 += sacleWL1DataBytes;
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::InitL0Buffs()
{
    if ASCEND_IS_AIV {
        return;
    }

    aL0Tensor_ = LocalTensor<xType>(AscendC::TPosition::A2, 0, L0A_BUFFER_SIZE / sizeof(xType));
    if constexpr (IsSupportB8Type<wType>()) {
        bL0Tensor_ = LocalTensor<wType>(AscendC::TPosition::B2, 0, L0B_BUFFER_SIZE);
    } else {
        bL0Tensor_ = LocalTensor<wType>(AscendC::TPosition::B2, 0, L0B_BUFFER_SIZE * 2);
    }
    cL0Tensor_ = LocalTensor<float>(AscendC::TPosition::CO1, 0, L0C_BUFFER_SIZE / sizeof(float));
    biasBtTensor_ = LocalTensor<float>(AscendC::TPosition::C2, 0, BIAS_TABLE_SIZE / sizeof(float));

    eventIdsMToMte1_[0] = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();
    eventIdsMToMte1_[1] = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();

    SetFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[0]);
    SetFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[1]);
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::InitUbDoubleBuffs()
{
    TBuf<QuePosition::VECIN> bankBuf;
    // 第一个 4 指 4 个 buffer，第二个 4 指 buffer 中数据的类型为 float，每个 float 是 4 个字节
    uint64_t bankUbSize = 4 * tiling_->matmulTiling.baseM * tiling_->matmulTiling.baseN * 4;
    pipe_->InitBuffer(bankBuf, bankUbSize);
    // accPingLocalTensor_起始下标：accLocalTensor_[0]，放在 bank0
    // accPongLocalTensor_起始下标：accLocalTensor_[64]，放在 bank1
    accLocalTensor_ = bankBuf.template GetWithOffset<float>(bankUbSize / 4, 0);
    // add0LocalTensor_起始下标：addLocalTensor_[0]，放在 bank2
    // add1LocalTensor_起始下标：addLocalTensor_[64]，放在 bank3
    addLocalTensor_ = bankBuf.template GetWithOffset<float>((bankUbSize - 512) / 4, 512);

    // 在 UB 上为 antiquant_scale 申请空间
    TBuf<QuePosition::VECIN> scaleBuf;
    // 2 指 ping pong buffer，申请 2 份空间
    uint64_t scaleUbSize = 2 * tiling_->matmulTiling.baseN * tiling_->matmulTiling.stepKa * sizeof(xType);
    pipe_->InitBuffer(scaleBuf, scaleUbSize);
    scalePingLocalTensor_ = scaleBuf.template GetWithOffset<xType>(scaleUbSize / 2, 0);
    scalePongLocalTensor_ = scaleBuf.template GetWithOffset<xType>(scaleUbSize / 2, scaleUbSize / 2);

    if constexpr (hasBias) {
        TBuf<QuePosition::VECIN> biasBuf;
        uint64_t biasUbSize = tiling_->matmulTiling.baseN * sizeof(biasType);
        pipe_->InitBuffer(biasBuf, biasUbSize);
        biasUbTensor_ = biasBuf.Get<biasType>();
    }

    TBuf<QuePosition::VECOUT> outBuf;
    // 2 指的是float16/bfloat16数据类型的空间大小
    pipe_->InitBuffer(outBuf, tiling_->matmulTiling.baseM * tiling_->matmulTiling.baseN * 2);
    resUbTensor_ = outBuf.Get<yType>();
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::InitUbQuadrupleBuffs()
{
    TBuf<QuePosition::VECIN> bankBuf;
    // 第一个 4 指 4 个buffer，第二个 4 指 buffer 中数据的类型为 float（4 字节），最后 2 是使能 4 buffer 需要 2 倍空间
    uint64_t bankUbSize = 4 * tiling_->matmulTiling.baseM * tiling_->matmulTiling.baseN * 4 * 2;
    uint64_t ubBufOffset = bankUbSize / 2;
    pipe_->InitBuffer(bankBuf, bankUbSize);
    // GetWithOffset 第一个参数size单位为元素个数，第二个参数bufOffset单位为字节
    // accPingLocalTensor_起始下标：accLocalTensor_[0]，放在 bank0
    // accPongLocalTensor_起始下标：accLocalTensor_[64]，放在 bank1
    accLocalTensor_ = bankBuf.template GetWithOffset<float>(bankUbSize / 4, 0);
    add2Offset_ = ubBufOffset / 4;
    // add0LocalTensor_起始下标：addLocalTensor_[0]，放在 bank2
    // add1LocalTensor_起始下标：addLocalTensor_[64]，放在 bank3
    // add2LocalTensor_起始下标：addLocalTensor_[add2Offset_]，放在 bank2
    // add3LocalTensor_起始下标：addLocalTensor_[add2Offset_ + 64]，放在 bank3
    addLocalTensor_ = bankBuf.template GetWithOffset<float>((bankUbSize - 512) / 4, 512);

    // 在 UB 上为 antiquant_scale 申请空间
    // 256 指 2 个 bank (256B * 2) 所能容纳的 b16 元素个数；第一个 2 指多倍载入，减少scale搬入与fixpipe的写UB冲突；
    // 第二个 2 指虽 scale 只存储在 bank0 和 bank1，但需要连续分配ub空间，只能附带 bank2 和 bank3 分配，空间大小翻倍
    uint64_t scaleUbSize = 256 * tiling_->matmulTiling.stepKa * 2 * sizeof(xType) * 2;
    scalePingLocalTensor_ = bankBuf.template GetWithOffset<xType>(scaleUbSize / 2, ubBufOffset);
    ubBufOffset += scaleUbSize;
    scalePongLocalTensor_ = bankBuf.template GetWithOffset<xType>(scaleUbSize / 2, ubBufOffset);

    if constexpr (hasBias) {
        ubBufOffset += scaleUbSize;
        // 第一个 4: bias 只存储在 bank0，但需要连续分配ub空间，附带申请 bank1、bank2 和 bank3；第二个 4 指
        // sizeof(float32)
        uint64_t biasUbSize = 4 * tiling_->matmulTiling.baseN * 4;
        biasUbTensor_ = bankBuf.template GetWithOffset<biasType>(biasUbSize, ubBufOffset);
    }

    TBuf<QuePosition::VECOUT> outBuf;
    // 2 指的是float16/bfloat16数据类型的空间大小
    pipe_->InitBuffer(outBuf, tiling_->matmulTiling.baseM * tiling_->matmulTiling.baseN * 2);
    resUbTensor_ = outBuf.Get<yType>();
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::Process()
{
    if (curBlockIdx_ >= tiling_->matmulTiling.usedCoreNum) {
        return;
    }

    uint64_t totalKBlocks = CeilDiv<uint64_t>(tiling_->matmulTiling.Ka, singleKSize_);
    for (uint64_t roundIndex = 0; roundIndex < block_.params_.round; ++roundIndex) {
        block_.UpdateBasicIndex(roundIndex);
        // 设置核内参数： singleCoreM 和 singleCoreN
        block_.UpdateBlockParams(roundIndex);
        if (block_.params_.singleCoreM <= 0 || block_.params_.singleCoreN <= 0) {
            continue;
        }
        block_.template CalcGMOffset<false, bTrans, wType, CubeFormat::NZ>();
        ProcessBaseBlockCompute(block_.offset_.offsetM, block_.offset_.offsetN, singleKSize_, totalKBlocks);
    }

    if ASCEND_IS_AIC {
        for (uint64_t idx = 0; idx < L1_BUFFER_NUM; idx++) {
            WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_[idx]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_[idx]);

            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdsMte2ToMte1_[idx]);
        }
        WaitFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[0]);
        WaitFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdsMToMte1_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdsMToMte1_[1]);
    }

    if ASCEND_IS_AIV {
        WaitFlag<HardEvent::V_MTE2>(eventIdUbPingVToMte2_);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdUbPingVToMte2_);

        WaitFlag<HardEvent::V_MTE2>(eventIdUbPongVToMte2_);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdUbPongVToMte2_);

        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdScaleMte2ToV_);

        if constexpr (hasBias) {
            WaitFlag<HardEvent::V_MTE2>(eventIdBiasVToMte2_);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdBiasVToMte2_);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdBiasMte2ToV_);
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::UpdateMmParams(
    uint64_t mOffset, uint64_t nOffset, L0CopyAndCalcParams& l0CopyAndCalcParams, FixL0CToDstParams& fixL0CToDstParams)
{
    if ASCEND_IS_AIV {
        return;
    }
    curMSize_ = Min<uint64_t>(tiling_->matmulTiling.M - mOffset, tiling_->matmulTiling.baseM);
    curNSize_ = Min<uint64_t>(tiling_->matmulTiling.N - nOffset, tiling_->matmulTiling.baseN);
    curMAlignSize_ = CeilAlign<uint64_t>(curMSize_, BLOCK_CUBE);
    curNAlignSize_ = CeilAlign<uint64_t>(curNSize_, BLOCK_CUBE);

    uint64_t tailNAlign64 = CeilAlign<uint64_t>(curNAlignSize_, 64UL); // 在L0上N方向64对齐，使能unitflag

    l0CopyAndCalcParams.mL0Size = curMSize_;
    l0CopyAndCalcParams.kL0Size = tiling_->matmulTiling.baseK;
    l0CopyAndCalcParams.nL0Size = tailNAlign64;
    l0CopyAndCalcParams.isFirstKLoop = true;
    l0CopyAndCalcParams.isLastKLoop = true;

    fixL0CToDstParams.nL0Size = curNSize_;
    fixL0CToDstParams.mL0Size = curMSize_;
    // 解ubBank冲突需跳过1024B 1024B/sizeof(fp32)=256，单bank可存放元素 256B/sizeof(fp32)=64
    fixL0CToDstParams.outNSize = 256;
    fixL0CToDstParams.splitNSize = 64UL;
    fixL0CToDstParams.dstNdStride = curMSize_ * 256;
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::CopyScaleToUB(
    uint64_t nOffset, uint64_t kOffset)
{
    if constexpr (useUb4Buff) {
        if (dbFlag_ != IDX_0) {
            return;
        }
    }
    // 将scale从GM搬运到UB中
    scaleDbFlag_ == IDX_0 ? WaitFlag<HardEvent::V_MTE2>(eventIdUbPingVToMte2_) :
                            WaitFlag<HardEvent::V_MTE2>(eventIdUbPongVToMte2_);
    AscendC::DataCopyParams dataCopyParams;
    if constexpr (useUb4Buff) {
        // 2：scale 多倍载入
        dataCopyParams.blockCount = Min<uint64_t>((tiling_->matmulTiling.Ka - kOffset) / tiling_->matmulTiling.baseK,
                                                  tiling_->matmulTiling.stepKa * 2) *
                                    tiling_->matmulTiling.baseK / tiling_->groupSize;
    } else {
        dataCopyParams.blockCount = Min<uint64_t>((tiling_->matmulTiling.Ka - kOffset) / tiling_->matmulTiling.baseK,
                                                  tiling_->matmulTiling.stepKa) *
                                    tiling_->matmulTiling.baseK / tiling_->groupSize;
    }

    uint64_t curN = Min<uint64_t>(tiling_->matmulTiling.N - nOffset, tiling_->matmulTiling.baseN);
    // 32B对齐
    dataCopyParams.blockLen = CeilDiv<uint64_t>(sizeof(xType) * curN, 32);
    // 32B对齐
    dataCopyParams.srcStride = CeilDiv<uint64_t>(sizeof(xType) * (tiling_->matmulTiling.N - curN), 32);
    if constexpr (useUb4Buff) {
        // 32B对齐，1024 指 4 个 bank 的宽度
        dataCopyParams.dstStride = CeilDiv<uint64_t>(1024 - sizeof(xType) * curN, 32);
    } else {
        dataCopyParams.dstStride = 0;
    }
    // 最后一个参数enablePreReadIgnoreSync = true表示开启PreAllocation
    DataCopy(scaleDbFlag_ == IDX_0 ? scalePingLocalTensor_ : scalePongLocalTensor_,
             antiquantScaleGlobal_[(kOffset / tiling_->groupSize) * tiling_->matmulTiling.N + nOffset], dataCopyParams,
             tiling_->enablePreAlloc);

    SetFlag<HardEvent::MTE2_V>(eventIdScaleMte2ToV_);
    WaitFlag<HardEvent::MTE2_V>(eventIdScaleMte2ToV_);
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::CopyBiasToUB(
    uint64_t nOffset)
{
    if ASCEND_IS_AIC {
        return;
    }

    WaitFlag<HardEvent::V_MTE2>(eventIdBiasVToMte2_);
    // 将bias从GM搬运到UB中
    uint64_t curN = Min<uint64_t>(tiling_->matmulTiling.N - nOffset, tiling_->matmulTiling.baseN);
    if constexpr (useUb4Buff) {
        uint64_t biasPerLen = 64; // bias 在 ub 上每 64 个元素存一行
        AscendC::DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = CeilDiv<uint64_t>(curN, biasPerLen);
        // 32B对齐
        dataCopyParams.blockLen = CeilDiv<uint64_t>(biasPerLen * sizeof(biasType), 32);
        dataCopyParams.srcStride = 0;
        // 32B对齐，1024 指 4 个 bank 的宽度
        dataCopyParams.dstStride = CeilDiv<uint64_t>(1024 - biasPerLen * sizeof(biasType), 32);
        // 最后一个参数enablePreReadIgnoreSync = true表示开启PreAllocation
        DataCopy(biasUbTensor_, biasGlobal_[nOffset], dataCopyParams, true);
    } else {
        // 函数内部已开启PreAllocation
        DataCopyPad2D(biasUbTensor_, biasGlobal_[nOffset], 1, curN, tiling_->matmulTiling.N);
    }

    SetFlag<HardEvent::MTE2_V>(eventIdBiasMte2ToV_);
}

template <typename yType, typename biasType, bool hasBias, bool useUb4Buff>
__aicore__ inline void CastToYType(const ScaleAccParam<yType, biasType>& param)
{
    uint16_t floatStride = 256;
    uint16_t b16Stride = 512;
    uint16_t stride = 64;

    MicroAPI::RegTensor<float> accRegTensor;

    MicroAPI::RegTensor<biasType> biasRegTensor;
    MicroAPI::RegTensor<float> biasFloatRegTensor;
    __local_mem__ biasType* biasUbAddr = param.biasAddr;

    MicroAPI::RegTensor<yType> castRegTensor;

    MicroAPI::MaskReg castMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg datacopyToUbMask = MicroAPI::CreateMask<yType, MicroAPI::MaskPattern::ALL>();

    __local_mem__ yType* destUbAddr = param.castDestAddr;
    __local_mem__ float* accUbAddr = param.accAddr;

    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    if (param.mCount == 1) {
        for (uint16_t nIdx = 0; nIdx < param.nCount; nIdx++) {
            if constexpr (hasBias) {
                MicroAPI::AddrReg biasReg;
                if constexpr (useUb4Buff) {
                    if constexpr (IsSameType<biasType, float>::value) {
                        biasReg = MicroAPI::CreateAddrReg<biasType>(nIdx, floatStride);
                    } else {
                        biasReg = MicroAPI::CreateAddrReg<biasType>(nIdx, b16Stride);
                    }
                } else {
                    biasReg = MicroAPI::CreateAddrReg<biasType>(nIdx, stride);
                }
                if constexpr (IsSameType<biasType, float>::value) {
                    MicroAPI::DataCopy<biasType, MicroAPI::LoadDist::DIST_NORM>(biasFloatRegTensor, biasUbAddr,
                                                                                biasReg);
                } else {
                    MicroAPI::DataCopy<biasType, MicroAPI::LoadDist::DIST_US_B16>(biasRegTensor, biasUbAddr, biasReg);
                    MicroAPI::Cast<float, biasType, castTrait>(biasFloatRegTensor, biasRegTensor, castMask);
                }
            }
            MicroAPI::AddrReg accReg = MicroAPI::CreateAddrReg<float>(nIdx, floatStride);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(accRegTensor, accUbAddr, accReg);
            if constexpr (hasBias) {
                MicroAPI::Add(accRegTensor, biasFloatRegTensor, accRegTensor, castMask);
            }
            MicroAPI::Cast<yType, float, castTrait>(castRegTensor, accRegTensor, castMask);
            MicroAPI::AddrReg castReg = MicroAPI::CreateAddrReg<yType>(nIdx, stride);
            MicroAPI::DataCopy<yType, MicroAPI::StoreDist::DIST_PACK_B32>(destUbAddr, castRegTensor, castReg,
                                                                          datacopyToUbMask);
        }
    }

    if (param.nCount == 1) {
        if constexpr (hasBias) {
            if constexpr (IsSameType<biasType, float>::value) {
                MicroAPI::DataCopy<biasType, MicroAPI::LoadDist::DIST_NORM>(biasFloatRegTensor, biasUbAddr);
            } else {
                MicroAPI::DataCopy<biasType, MicroAPI::LoadDist::DIST_US_B16>(biasRegTensor, biasUbAddr);
                MicroAPI::Cast<float, biasType, castTrait>(biasFloatRegTensor, biasRegTensor, castMask);
            }
        }
        for (uint16_t mIdx = 0; mIdx < param.mCount; mIdx++) {
            MicroAPI::AddrReg accReg = MicroAPI::CreateAddrReg<float>(mIdx, floatStride);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(accRegTensor, accUbAddr, accReg);
            if constexpr (hasBias) {
                MicroAPI::Add(accRegTensor, biasFloatRegTensor, accRegTensor, castMask);
            }
            MicroAPI::Cast<yType, float, castTrait>(castRegTensor, accRegTensor, castMask);
            MicroAPI::AddrReg castReg = MicroAPI::CreateAddrReg<yType>(mIdx, stride);
            MicroAPI::DataCopy<yType, MicroAPI::StoreDist::DIST_PACK_B32>(destUbAddr, castRegTensor, castReg,
                                                                          datacopyToUbMask);
        }
        return;
    }

    for (uint16_t nIdx = 0; nIdx < param.nCount; nIdx++) {
        if constexpr (hasBias) {
            MicroAPI::AddrReg biasReg;
            if constexpr (useUb4Buff) {
                if constexpr (IsSameType<biasType, float>::value) {
                    biasReg = MicroAPI::CreateAddrReg<biasType>(nIdx, floatStride);
                } else {
                    biasReg = MicroAPI::CreateAddrReg<biasType>(nIdx, b16Stride);
                }
            } else {
                biasReg = MicroAPI::CreateAddrReg<biasType>(nIdx, stride);
            }
            if constexpr (IsSameType<biasType, float>::value) {
                MicroAPI::DataCopy<biasType, MicroAPI::LoadDist::DIST_NORM>(biasFloatRegTensor, biasUbAddr, biasReg);
            } else {
                MicroAPI::DataCopy<biasType, MicroAPI::LoadDist::DIST_US_B16>(biasRegTensor, biasUbAddr, biasReg);
                MicroAPI::Cast<float, biasType, castTrait>(biasFloatRegTensor, biasRegTensor, castMask);
            }
        }
        for (uint16_t mIdx = 0; mIdx < param.mCount; mIdx++) {
            MicroAPI::AddrReg accReg = MicroAPI::CreateAddrReg<float>(nIdx, floatStride, mIdx,
                                                                      param.nCount * floatStride);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(accRegTensor, accUbAddr, accReg);
            if constexpr (hasBias) {
                MicroAPI::Add(accRegTensor, biasFloatRegTensor, accRegTensor, castMask);
            }
            MicroAPI::Cast<yType, float, castTrait>(castRegTensor, accRegTensor, castMask);
            MicroAPI::AddrReg castReg = MicroAPI::CreateAddrReg<yType>(nIdx, stride, mIdx, param.nCount * stride);
            MicroAPI::DataCopy<yType, MicroAPI::StoreDist::DIST_PACK_B32>(destUbAddr, castRegTensor, castReg,
                                                                          datacopyToUbMask);
        }
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::CastAndCopyToGm(
    uint64_t mOffset, uint64_t nOffset)
{
    if ASCEND_IS_AIC {
        return;
    }

    // 设置累加结果处理的参数
    ScaleAccParam<yType, biasType> resParam;
    resParam.mCount = Min<uint64_t>(tiling_->matmulTiling.baseM, tiling_->matmulTiling.M - mOffset);
    uint64_t curN = Min<uint64_t>(tiling_->matmulTiling.baseN, tiling_->matmulTiling.N - nOffset);
    // 64：每一次处理64个数字
    resParam.nCount = CeilDiv<uint64_t>(curN, 64);

    resParam.accAddr = (__local_mem__ float*)accLocalTensor_[((kIterateIdx_ & 1) ^ 1) << 6].GetPhyAddr();

    resParam.castDestAddr = (__local_mem__ yType*)resUbTensor_.GetPhyAddr();

    if constexpr (hasBias) {
        resParam.biasAddr = (__local_mem__ biasType*)biasUbTensor_.GetPhyAddr();
        WaitFlag<HardEvent::MTE2_V>(eventIdBiasMte2ToV_);
    }

    event_t eventIDMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIDMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIDMte3ToV);

    AscendC::VF_CALL<CastToYType<yType, biasType, hasBias, useUb4Buff>>(resParam);
    event_t eventIDVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMte3);
    // 将结果由 UB 搬运至 GM
    CopyResToGM(mOffset, nOffset);
    if constexpr (hasBias) {
        SetFlag<HardEvent::V_MTE2>(eventIdBiasVToMte2_);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void
WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::ProcessBaseBlockCompute(uint64_t mOffset,
                                                                                                  uint64_t nOffset,
                                                                                                  uint64_t kBlockSize,
                                                                                                  uint64_t totalKBlocks)
{
    if constexpr (hasBias) {
        CopyBiasToUB(nOffset);
    }

    L0CopyAndCalcParams l0CopyAndCalcParams;
    FixL0CToDstParams fixL0CToDstParams;
    UpdateMmParams(mOffset, nOffset, l0CopyAndCalcParams, fixL0CToDstParams);

    for (uint64_t kOutIdx = 0; kOutIdx < totalKBlocks; kOutIdx++) {
        uint64_t kOutOffset = kOutIdx * kBlockSize;
        curKL1Size_ = Min<uint64_t>(tiling_->matmulTiling.Ka - kOutOffset, singleKSize_);
        if ASCEND_IS_AIC {
            if constexpr (useL14Buff) {
                GetABToL1QuadrupleBuff(mOffset, nOffset, kOutIdx);
                GetABL1ToL0();
            } else {
                GetABToL1(mOffset, nOffset, kOutOffset);
                WaitFlag<HardEvent::MTE2_MTE1>(eventIdsMte2ToMte1_[curL1BufIdx_]);
            }
        }

        if ASCEND_IS_AIV {
            CopyScaleToUB(nOffset, kOutOffset);
        }

        // K 内部循环，因为需要在 Vector Core 上累加，所以需要保证每次 Matmul 的 K size 在同一个 group 内
        uint64_t kInnerBlockCount = CeilDiv<uint64_t>(curKL1Size_, tiling_->matmulTiling.baseK);
        for (uint64_t kInnerIdx = 0; kInnerIdx < kInnerBlockCount; kInnerIdx++) {
            if ASCEND_IS_AIC {
                // FixPipe交替写入UB的多块buffer，开头不用等Vector同步
                if (kIterateIdx_ > PIPE_V_TO_FIX_SYNC_START) {
                    WaitForVector();
                }

                if constexpr (useL14Buff) {
                    IterateMatmulForQuadrupleBuff(kInnerIdx, l0CopyAndCalcParams, fixL0CToDstParams);
                } else {
                    IterateMatmul(mOffset, nOffset, kOutIdx, kInnerIdx, kBlockSize);
                }
                NotifyVector();
            }
            if ASCEND_IS_AIV {
                WaitForCube();
                ProcessSingleBlock(mOffset, nOffset, kInnerIdx);

                // FixPipe交替写入UB的多块buffer，最后baseK不需要设置核间同步标记防止不匹配
                if (kIterateIdx_ < pipeVToFixSyncCount_) {
                    NotifyCube();
                }
            }
            kIterateIdx_++;
        }

        if ASCEND_IS_AIC {
            // L1 2 or 4 buffer切换
            SetFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_[curL1BufIdx_]);
            curL1BufIdx_ = (curL1BufIdx_ + 1) & L1_BUFFER_MASK;
            if constexpr (useL14Buff) {
                // L0AB double buffer 切换
                SetFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[l0BufIdx_]);
                l0BufIdx_ ^= 1;
            }
        }

        if ASCEND_IS_AIV {
            // 一次 K 方向计算结束，ping pong 切换
            if constexpr (useUb4Buff) {
                dbFlag_ = dbFlag_ ^ 1;
                if (dbFlag_ == IDX_0 || kOutIdx == totalKBlocks - 1) {
                    scaleDbFlag_ == IDX_0 ? SetFlag<HardEvent::V_MTE2>(eventIdUbPingVToMte2_) :
                                            SetFlag<HardEvent::V_MTE2>(eventIdUbPongVToMte2_);
                    scaleDbFlag_ = scaleDbFlag_ ^ 1;
                }
                if (kOutIdx == totalKBlocks - 1) {
                    dbFlag_ = IDX_0;
                }
            } else {
                scaleDbFlag_ == IDX_0 ? SetFlag<HardEvent::V_MTE2>(eventIdUbPingVToMte2_) :
                                        SetFlag<HardEvent::V_MTE2>(eventIdUbPongVToMte2_);
                scaleDbFlag_ = scaleDbFlag_ ^ 1;
            }
        }
    }
    curL1BufIdx_ = 0; // 下一轮计算需要从第0个buff开始，保证尾块场景无问题
    CastAndCopyToGm(mOffset, nOffset);
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::GetABToL1(
    uint64_t mOffset, uint64_t nOffset, uint64_t kOffset)
{
    l0LoadKSize_ = tiling_->matmulTiling.baseK;

    // MTE2 等待 MTE1 完成，即等待 L1 buffer 可写入。A、B 从 GM 搬运至 L1 时共用一组同步
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_[curL1BufIdx_]);

    CopyXNd2Nz(mOffset, kOffset);
    CopyWeightGm2L1(nOffset, kOffset);

    SetFlag<HardEvent::MTE2_MTE1>(eventIdsMte2ToMte1_[curL1BufIdx_]);
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::CopyXNd2Nz(
    uint64_t mOffset, uint64_t kOffset)
{
    // A 矩阵在 GM 上搬运的偏移计算：mOffset * K + kOffset
    uint64_t aGmOffset = (mOffset * tiling_->matmulTiling.Ka + kOffset);

    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = curMSize_;
    nd2nzParams.dValue = curKL1Size_;
    nd2nzParams.srcDValue = tiling_->matmulTiling.Ka;
    nd2nzParams.srcNdMatrixStride = 0;

    nd2nzParams.dstNzC0Stride = CeilAlign<uint64_t>(nd2nzParams.nValue, BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    // 最后一个参数enablePreReadIgnoreSync = true表示开启PreAllocation
    DataCopy((curL1BufIdx_ == 0) ? aL1LocalBuf0_ : aL1LocalBuf1_, xGlobal_[aGmOffset], nd2nzParams,
             tiling_->enablePreAlloc);
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::CopyWeightGm2L1(
    uint64_t nOffset, uint64_t kOffset)
{
    uint64_t weightGmOffset;
    if constexpr (IsSupportB8Type<wType>()) {
        // 转置的 B 矩阵在 GM 上搬运的偏移计算：kOffset * problem_n + nOffset * K0_INT8
        weightGmOffset = (kOffset * tiling_->matmulTiling.N + nOffset * C0<wType>::value);
    } else {
        // 转置的 B 矩阵在 GM 上搬运的偏移计算 (int4 需要除以 2)：kOffset * problem_n + nOffset * K0_INT4
        weightGmOffset = (kOffset * tiling_->matmulTiling.N + nOffset * C0<wType>::value) / 2;
    }

    AscendC::DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = CeilDiv<uint64_t>(curKL1Size_, C0<wType>::value);
    dataCopyParams.blockLen = curNSize_;
    dataCopyParams.srcStride = tiling_->matmulTiling.N - curNSize_;
    dataCopyParams.dstStride = 0;

    // 最后一个参数enablePreReadIgnoreSync = true表示开启PreAllocation
    AscendC::DataCopy((curL1BufIdx_ == 0) ? bL1LocalBuf0_.template ReinterpretCast<uint8_t>() :
                                            bL1LocalBuf1_.template ReinterpretCast<uint8_t>(),
                      wGlobalU8_[weightGmOffset], dataCopyParams, tiling_->enablePreAlloc);
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::GetABToL1QuadrupleBuff(
    uint64_t mOffset, uint64_t nOffset, uint64_t kOutIdx)
{
    l0LoadKSize_ = curKL1Size_;

    // L1每次4 buff连续搬入
    if (kOutIdx & L1_BUFFER_MASK) {
        return;
    }

    // A 矩阵 K 方向搬入 L1 Size 计算
    uint64_t kOffset = kOutIdx * singleKSize_;
    // A 矩阵在 GM 上搬运的偏移计算：mOffset * K + kOffset
    uint64_t aGmOffset = mOffset * tiling_->matmulTiling.Ka + kOffset;

    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = curMSize_;
    nd2nzParams.dValue = curKL1Size_;
    nd2nzParams.srcDValue = tiling_->matmulTiling.Ka;
    nd2nzParams.srcNdMatrixStride = 0;

    nd2nzParams.dstNzC0Stride = CeilAlign<uint64_t>(nd2nzParams.nValue, BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    uint64_t weightGmOffset;
    uint64_t weightGmPerBuffOffset;
    weightGmOffset = (kOffset * tiling_->matmulTiling.N + nOffset * C0<wType>::value);
    if constexpr (IsSupportB8Type<wType>()) {
        weightGmPerBuffOffset = (singleKSize_ * tiling_->matmulTiling.N);
    } else {
        // 转置的 B 矩阵在 GM 上搬运的偏移计算 (int4 需要除以 2)
        weightGmOffset = weightGmOffset >> 1;
        weightGmPerBuffOffset = (singleKSize_ * tiling_->matmulTiling.N) / 2;
    }

    AscendC::DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = CeilDiv<uint64_t>(curKL1Size_, C0<wType>::value);
    dataCopyParams.blockLen = curNSize_;
    dataCopyParams.srcStride = tiling_->matmulTiling.N - curNSize_;
    dataCopyParams.dstStride = 0;

    /* 每次L1按照4 buff搬运 */
    uint64_t blockCount = Min<uint64_t>(CeilDiv(tiling_->matmulTiling.Ka - kOffset, singleKSize_),
                                        QUADRUPLE_BUFFER_NUM);
    if (blockCount > IDX_0) {
        // MTE2 等待 MTE1 完成，即等待 L1 buffer 可写入。A、B 从 GM 搬运至 L1 时共用一组同步
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_[0]);
        // 最后一个参数enablePreReadIgnoreSync = true表示开启PreAllocation
        DataCopy(aL1LocalBuf0_, xGlobal_[aGmOffset], nd2nzParams, tiling_->enablePreAlloc);
        AscendC::DataCopy(bL1LocalBuf0_.template ReinterpretCast<uint8_t>(), wGlobalU8_[weightGmOffset], dataCopyParams,
                          tiling_->enablePreAlloc);
        SetFlag<HardEvent::MTE2_MTE1>(eventIdsMte2ToMte1_[0]);
    }

    if (blockCount > IDX_1) {
        aGmOffset += singleKSize_;
        weightGmOffset += weightGmPerBuffOffset;
        // MTE2 等待 MTE1 完成，即等待 L1 buffer 可写入。A、B 从 GM 搬运至 L1 时共用一组同步
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_[1]);
        // 最后一个参数enablePreReadIgnoreSync = true表示开启PreAllocation
        DataCopy(aL1LocalBuf1_, xGlobal_[aGmOffset], nd2nzParams, tiling_->enablePreAlloc);
        AscendC::DataCopy(bL1LocalBuf1_.template ReinterpretCast<uint8_t>(), wGlobalU8_[weightGmOffset], dataCopyParams,
                          tiling_->enablePreAlloc);
        SetFlag<HardEvent::MTE2_MTE1>(eventIdsMte2ToMte1_[1]);
    }

    if (blockCount > IDX_2) {
        aGmOffset += singleKSize_;
        weightGmOffset += weightGmPerBuffOffset;
        // MTE2 等待 MTE1 完成，即等待 L1 buffer 可写入。A、B 从 GM 搬运至 L1 时共用一组同步
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_[2]);
        // 最后一个参数enablePreReadIgnoreSync = true表示开启PreAllocation
        DataCopy(aL1LocalBuf2_, xGlobal_[aGmOffset], nd2nzParams, tiling_->enablePreAlloc);
        AscendC::DataCopy(bL1LocalBuf2_.template ReinterpretCast<uint8_t>(), wGlobalU8_[weightGmOffset], dataCopyParams,
                          tiling_->enablePreAlloc);
        SetFlag<HardEvent::MTE2_MTE1>(eventIdsMte2ToMte1_[2]);
    }

    if (blockCount > IDX_3) {
        aGmOffset += singleKSize_;
        weightGmOffset += weightGmPerBuffOffset;
        // MTE2 等待 MTE1 完成，即等待 L1 buffer 可写入。A、B 从 GM 搬运至 L1 时共用一组同步
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdsMte1ToMte2_[3]);
        // 最后一个参数enablePreReadIgnoreSync = true表示开启PreAllocation
        DataCopy(aL1LocalBuf3_, xGlobal_[aGmOffset], nd2nzParams, tiling_->enablePreAlloc);
        AscendC::DataCopy(bL1LocalBuf3_.template ReinterpretCast<uint8_t>(), wGlobalU8_[weightGmOffset], dataCopyParams,
                          tiling_->enablePreAlloc);
        SetFlag<HardEvent::MTE2_MTE1>(eventIdsMte2ToMte1_[3]);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::GetABL1ToL0()
{
    // 等待 A 与 B 矩阵由 GM 搬入 L1 完成后，启动搬入L0
    WaitFlag<HardEvent::MTE2_MTE1>(eventIdsMte2ToMte1_[curL1BufIdx_]);
    // L0一次载入一整块L1 buff的数据
    if (curL1BufIdx_ == IDX_0) {
        LoadToL0AB(aL1LocalBuf0_, bL1LocalBuf0_);
    } else if (curL1BufIdx_ == IDX_1) {
        LoadToL0AB(aL1LocalBuf1_, bL1LocalBuf1_);
    } else if (curL1BufIdx_ == IDX_2) {
        LoadToL0AB(aL1LocalBuf2_, bL1LocalBuf2_);
    } else {
        LoadToL0AB(aL1LocalBuf3_, bL1LocalBuf3_);
    }

    SetFlag<HardEvent::MTE1_M>(l0BufIdx_ & 1);
    WaitFlag<HardEvent::MTE1_M>(l0BufIdx_ & 1);
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::LoadToL0AB(
    const LocalTensor<xType>& aL1Tensor, const LocalTensor<wType>& bL1Tensor)
{
    // L0A 容量64KB，A 矩阵为 16 bits，可存放 2^15 个数，Ping下标为0，Pong下标为 2^14
    LocalTensor<xType> aL0Tensor = aL0Tensor_[(l0BufIdx_ & 1) << 14];
    LocalTensor<wType> bL0Tensor;
    if constexpr (IsSupportB8Type<wType>()) {
        // L0B 容量64KB，B 矩阵为 8 bits，可存放 2^16 个数，Ping下标为0，Pong下标为 2^15
        bL0Tensor = bL0Tensor_[(l0BufIdx_ & 1) << 15];
    } else {
        // L0B 容量64KB，B 矩阵为 4 bits，可存放 2^17 个数，Ping下标为0，Pong下标为 2^16
        bL0Tensor = bL0Tensor_[(l0BufIdx_ & 1) << 16];
    }

    L0CopyAndCalcParams l0CopyAndCalcParams;
    l0CopyAndCalcParams.mL0Size = curMSize_;
    l0CopyAndCalcParams.mL1Size = curMAlignSize_;
    l0CopyAndCalcParams.kL0Size = l0LoadKSize_;
    l0CopyAndCalcParams.kL1Size = curKL1Size_;
    l0CopyAndCalcParams.nL0Size = curNAlignSize_;
    l0CopyAndCalcParams.nL1Size = curNSize_;

    WaitFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[l0BufIdx_]);
    WeightQuantBatchMatmul::Arch35::LoadAL1ToL0<xType, xType>(aL0Tensor, aL1Tensor, l0CopyAndCalcParams);

    if constexpr (IsSameType<wType, fp4x2_e2m1_t>::value) {
        if constexpr (useL14Buff) {
            uint64_t tailNAlign64 = CeilAlign<uint64_t>(curNAlignSize_, 64UL); // 在L0上N方向64对齐，使能unitflag
            AscendC::LoadData2DParamsV2 bL0Load2dParams;
            bL0Load2dParams.mStartPosition = 0;
            bL0Load2dParams.kStartPosition = 0;
            bL0Load2dParams.mStep = curNAlignSize_ / static_cast<uint64_t>(BLOCK_CUBE);
            if constexpr (IsSupportB8Type<wType>()) {
                bL0Load2dParams.kStep = CeilDiv(l0LoadKSize_, 32UL);
            } else {
                bL0Load2dParams.kStep = CeilDiv(l0LoadKSize_, 32UL) >> 1;
            }
            bL0Load2dParams.srcStride = CeilDiv(curNAlignSize_ * 32, 512UL);
            bL0Load2dParams.dstStride = CeilDiv(tailNAlign64 * 32, 512UL);

            AscendC::LoadData2DMxParams scaleWLoad2dMxParams;
            scaleWLoad2dMxParams.xStartPosition = 0;
            scaleWLoad2dMxParams.yStartPosition = 0;
            scaleWLoad2dMxParams.xStep = curNAlignSize_ / static_cast<uint64_t>(BLOCK_CUBE);
            // 32为MX量化的group-size，2为一个分形在K方向上的字节数，2*32UL=64UL
            scaleWLoad2dMxParams.yStep = CeilDiv(l0LoadKSize_, 64UL) << 1;
            scaleWLoad2dMxParams.srcStride = CeilDiv(l0LoadKSize_, 64UL);
            scaleWLoad2dMxParams.dstStride = scaleWLoad2dMxParams.srcStride
                                             << 1; // L0B和L0B_MX的数据量是16:1的关系，buffer大小是8:1的关系

            AscendC::LoadData(bL0Tensor, bL1Tensor, scaleWL1LocalBuf_, bL0Load2dParams, scaleWLoad2dMxParams);
        } else {
            l0CopyAndCalcParams.scaleKL1Size = l0LoadKSize_ / 32UL;
            WeightQuantBatchMatmul::Arch35::LoadBAndScaleL1ToL0<wType, wType, fp8_e8m0_t>(
                bL0Tensor, bL1Tensor, scaleWL1LocalBuf_, l0CopyAndCalcParams);
        }
    } else {
        l0CopyAndCalcParams.nL0Size = CeilAlign<uint64_t>(curNSize_,
                                                          64UL); // 在L0上N方向64对齐，原先的curNAlignSize_为16对齐
        WeightQuantBatchMatmul::Arch35::LoadBL1ToL0<wType, wType>(bL0Tensor, bL1Tensor, l0CopyAndCalcParams);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::GetTensorC(
    const LocalTensor<float>& dstTensor, uint64_t tailM, uint64_t tailN, uint64_t tailMAlign, uint64_t tailNAlign)
{
    uint64_t tailK = Min<uint64_t>(
        tiling_->matmulTiling.Ka - (kIterateIdx_ % singleCoreKCount_) * tiling_->matmulTiling.baseK,
        tiling_->matmulTiling.baseK);
    // L0A 容量64KB，A 矩阵为 16 bits，可存放 2^15 个数，Ping下标为0，Pong下标为 2^14
    LocalTensor<xType> aL0Tensor = aL0Tensor_[(kIterateIdx_ & 1) << 14];
    LocalTensor<wType> bL0Tensor;
    if constexpr (IsSupportB8Type<wType>()) {
        // L0B 容量64KB，B 矩阵为 8 bits，可存放 2^16 个数，Ping下标为0，Pong下标为 2^15
        bL0Tensor = bL0Tensor_[(kIterateIdx_ & 1) << 15];
    } else {
        // L0B 容量64KB，B 矩阵为 4 bits，可存放 2^17 个数，Ping下标为0，Pong下标为 2^16
        bL0Tensor = bL0Tensor_[(kIterateIdx_ & 1) << 16];
    }
    // L0C 容量256KB，C 矩阵为 float32，可存放 2^16 个数
    LocalTensor<float> cL0Tensor;
    if constexpr (useUb4Buff) {
        // buf0下标为0，buf1下标为 2^14，buf2下标为 2^15，buf3下标为 2^14 + 2^15
        cL0Tensor = cL0Tensor_[(kIterateIdx_ & 3) << 14];
    } else {
        // Ping下标为0，Pong下标为 2^15
        cL0Tensor = cL0Tensor_[(kIterateIdx_ & 1) << 15];
    }

    uint64_t tailNAlign64 = CeilAlign<uint64_t>(tailNAlign, 64UL); // 在L0上N方向64对齐，使能unitflag
    L0CopyAndCalcParams l0CopyAndCalcParams;
    l0CopyAndCalcParams.mL0Size = tailMAlign;
    l0CopyAndCalcParams.kL0Size = tailK;
    l0CopyAndCalcParams.nL0Size = tailNAlign64;
    l0CopyAndCalcParams.isFirstKLoop = true;
    l0CopyAndCalcParams.isLastKLoop = true;

    WaitFlag<HardEvent::MTE1_M>(l0BufIdx_ & 1);
    WeightQuantBatchMatmul::Arch35::MmadCompute<float, xType, wType>(cL0Tensor, aL0Tensor, bL0Tensor,
                                                                     l0CopyAndCalcParams);
    SetFlag<HardEvent::M_MTE1>(eventIdsMToMte1_[l0BufIdx_]);

    FixL0CToDstParams fixL0CToDstParams;
    fixL0CToDstParams.nL0Size = tailN;
    fixL0CToDstParams.mL0Size = tailM;
    fixL0CToDstParams.outNSize = 256;
    fixL0CToDstParams.splitNSize = 64UL;
    fixL0CToDstParams.dstNdStride = tailM * 256;
    WeightQuantBatchMatmul::Arch35::FixL0CToDst<float, float, true>(dstTensor, cL0Tensor, fixL0CToDstParams);
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::IterateMatmul(
    uint64_t mOffset, uint64_t nOffset, uint64_t kOutIdx, uint64_t kInnerIdx, uint64_t singleK)
{
    if ASCEND_IS_AIC {
        uint64_t tailM = Min<uint64_t>(tiling_->matmulTiling.M - mOffset, tiling_->matmulTiling.baseM);
        uint64_t tailN = Min<uint64_t>(tiling_->matmulTiling.N - nOffset, tiling_->matmulTiling.baseN);
        uint64_t kL1Size = Min<uint64_t>(tiling_->matmulTiling.Ka - kOutIdx * singleK, singleK);
        uint64_t tailMAlign = CeilAlign<uint64_t>(tailM, BLOCK_CUBE);
        uint64_t tailNAlign = CeilAlign<uint64_t>(tailN, BLOCK_CUBE);
        uint64_t aL1Offset = kInnerIdx * tiling_->matmulTiling.baseK * tailMAlign;
        // 以下 bL1Offset 计算公式按照 B 矩阵转置为前提
        uint64_t bL1Offset = kInnerIdx * tiling_->matmulTiling.baseK * tailNAlign;
        LoadToL0AB((curL1BufIdx_ == IDX_0) ? aL1LocalBuf0_[aL1Offset] : aL1LocalBuf1_[aL1Offset],
                   (curL1BufIdx_ == IDX_0) ? bL1LocalBuf0_[bL1Offset] : bL1LocalBuf1_[bL1Offset]);
        SetFlag<HardEvent::MTE1_M>(l0BufIdx_ & 1);
        if constexpr (useUb4Buff) {
            GetTensorC(addLocalTensor_[(((kIterateIdx_ & 2) == 0) ? 0 : add2Offset_) + ((kIterateIdx_ & 1) << 6)],
                       tailM, tailN, tailM, tailNAlign);
        } else {
            GetTensorC(addLocalTensor_[(kIterateIdx_ & 1) << 6], tailM, tailN, tailM, tailNAlign);
        }
        l0BufIdx_ ^= 1;
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void
WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::IterateMatmulForQuadrupleBuff(
    uint64_t kInnerIdx, const L0CopyAndCalcParams& l0CopyAndCalcParams, const FixL0CToDstParams& fixL0CToDstParams)
{
    if ASCEND_IS_AIC {
        uint64_t aL0Offset = kInnerIdx * tiling_->matmulTiling.baseK * curMAlignSize_;
        // 以下 bL0Offset 计算公式按照 B 矩阵转置为前提
        uint64_t tailNAlign64 = CeilAlign<uint64_t>(curNAlignSize_, 64UL); // 在L0上N方向64对齐，使能unitflag
        uint64_t bL0Offset = kInnerIdx * tiling_->matmulTiling.baseK * tailNAlign64;
        // L0B 容量64KB:B 矩阵为 8 bits，可存放 2^16 个数，Ping下标为0，Pong下标为 2^15; B 矩阵为 4 bits，可存放 2^17
        // 个数，Ping下标为0，Pong下标为 2^16
        constexpr uint8_t bL0BitOffset = IsSupportB8Type<wType>() ? 15 : 16;
        LocalTensor<xType> aL0Tensor = aL0Tensor_[((l0BufIdx_ & 1) << 14) + aL0Offset];
        LocalTensor<wType> bL0Tensor = bL0Tensor_[((l0BufIdx_ & 1) << bL0BitOffset) + bL0Offset];
        uint64_t l0cOffsetBit = 16 - (L1_BUFFER_NUM >> 1); // 16:L0C 256K/sizeof(float32)
        LocalTensor<float> cL0Tensor = cL0Tensor_[(kIterateIdx_ & L1_BUFFER_MASK) << l0cOffsetBit];

        WeightQuantBatchMatmul::Arch35::MmadCompute<float, xType, wType>(cL0Tensor, aL0Tensor, bL0Tensor,
                                                                         l0CopyAndCalcParams);
        WeightQuantBatchMatmul::Arch35::FixL0CToDst<float, float, true>(addLocalTensor_[(kIterateIdx_ & 1) << 6],
                                                                        cL0Tensor, fixL0CToDstParams);
    }
}

// 量化单个block并且搬运到目标地址
template <typename yType>
__aicore__ inline void ScaleSingleBlockVf(const ScaleAccParam<yType>& param)
{
    const uint16_t mCount = static_cast<uint16_t>(param.mCount);
    const uint16_t nCount = static_cast<uint16_t>(param.nCount);
    uint16_t mFloatStride = 256;
    uint16_t stride = 64;

    MicroAPI::RegTensor<yType> scaleRegTensor;
    MicroAPI::RegTensor<float> scaleFloatRegTensor;
    __local_mem__ yType* scaleAddr = param.scaleAddr;

    MicroAPI::RegTensor<float> addRegTensor;
    __local_mem__ float* addUbAddr = param.addAddr;

    MicroAPI::RegTensor<float> mulResFloatRegTensor;
    MicroAPI::MaskReg datacopyToUbMask = MicroAPI::CreateMask<yType, MicroAPI::MaskPattern::ALL>();

    __local_mem__ float* destUbAddr = param.destUbAddr;

    MicroAPI::MaskReg scaleCastMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    if (param.mCount == 1) {
        for (uint16_t nIdx = 0; nIdx < nCount; nIdx++) {
            MicroAPI::AddrReg scaleReg = MicroAPI::CreateAddrReg<yType>(nIdx, stride);
            MicroAPI::DataCopy<yType, MicroAPI::LoadDist::DIST_US_B16>(scaleRegTensor, scaleAddr, scaleReg);
            MicroAPI::Cast<float, yType, castTrait>(scaleFloatRegTensor, scaleRegTensor, scaleCastMask);
            MicroAPI::AddrReg addReg = MicroAPI::CreateAddrReg<float>(nIdx, mFloatStride);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(addRegTensor, addUbAddr, addReg);
            MicroAPI::Mul(mulResFloatRegTensor, scaleFloatRegTensor, addRegTensor, scaleCastMask);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(destUbAddr, mulResFloatRegTensor, addReg,
                                                                          datacopyToUbMask);
        }
        return;
    }

    if (param.nCount == 1) {
        MicroAPI::DataCopy<yType, MicroAPI::LoadDist::DIST_US_B16>(scaleRegTensor, scaleAddr);
        MicroAPI::Cast<float, yType, castTrait>(scaleFloatRegTensor, scaleRegTensor, scaleCastMask);
        for (uint16_t mIdx = 0; mIdx < mCount; mIdx++) {
            MicroAPI::AddrReg addReg = MicroAPI::CreateAddrReg<float>(mIdx, mFloatStride);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(addRegTensor, addUbAddr, addReg);
            MicroAPI::Mul(mulResFloatRegTensor, scaleFloatRegTensor, addRegTensor, scaleCastMask);
            MicroAPI::AddrReg destReg = MicroAPI::CreateAddrReg<float>(mIdx, mFloatStride);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(destUbAddr, mulResFloatRegTensor, destReg,
                                                                          datacopyToUbMask);
        }
        return;
    }

    for (uint16_t nIdx = 0; nIdx < nCount; nIdx++) {
        MicroAPI::AddrReg scaleReg = MicroAPI::CreateAddrReg<yType>(nIdx, stride);
        MicroAPI::DataCopy<yType, MicroAPI::LoadDist::DIST_US_B16>(scaleRegTensor, scaleAddr, scaleReg);
        MicroAPI::Cast<float, yType, castTrait>(scaleFloatRegTensor, scaleRegTensor, scaleCastMask);
        for (uint16_t mIdx = 0; mIdx < mCount; mIdx++) {
            MicroAPI::AddrReg addReg = MicroAPI::CreateAddrReg<float>(nIdx, param.mCount * mFloatStride, mIdx,
                                                                      mFloatStride);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(addRegTensor, addUbAddr, addReg);
            MicroAPI::Mul(mulResFloatRegTensor, scaleFloatRegTensor, addRegTensor, scaleCastMask);
            MicroAPI::AddrReg destReg = MicroAPI::CreateAddrReg<float>(nIdx, mFloatStride, mIdx,
                                                                       param.nCount * mFloatStride);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(destUbAddr, mulResFloatRegTensor, destReg,
                                                                          datacopyToUbMask);
        }
    }
}

template <typename yType>
__aicore__ inline void AccSingleBlockVf(const ScaleAccParam<yType>& param)
{
    const uint16_t mCount = static_cast<uint16_t>(param.mCount);
    const uint16_t nCount = static_cast<uint16_t>(param.nCount);
    uint16_t mFloatStride = 256;
    uint16_t stride = 64;

    MicroAPI::RegTensor<yType> scaleRegTensor;
    MicroAPI::RegTensor<float> scaleFloatRegTensor;
    __local_mem__ yType* scaleAddr = param.scaleAddr;

    MicroAPI::RegTensor<float> addRegTensor;
    __local_mem__ float* addUbAddr = param.addAddr;

    MicroAPI::RegTensor<float> accRegTensor;
    __local_mem__ float* accUbAddr = param.accAddr;

    MicroAPI::RegTensor<float> mulResFloatRegTensor;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    __local_mem__ float* destUbAddr = param.destUbAddr;

    MicroAPI::MaskReg scaleCastMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    static constexpr MicroAPI::CastTrait castTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                      MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

    if (param.mCount == 1) {
        for (uint16_t nIdx = 0; nIdx < nCount; nIdx++) {
            MicroAPI::AddrReg scaleReg = MicroAPI::CreateAddrReg<yType>(nIdx, stride);
            MicroAPI::DataCopy<yType, MicroAPI::LoadDist::DIST_US_B16>(scaleRegTensor, scaleAddr, scaleReg);
            MicroAPI::Cast<float, yType, castTrait>(scaleFloatRegTensor, scaleRegTensor, scaleCastMask);

            MicroAPI::AddrReg addReg = MicroAPI::CreateAddrReg<float>(nIdx, mFloatStride);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(addRegTensor, addUbAddr, addReg);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(accRegTensor, accUbAddr, addReg);

            MicroAPI::MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(accRegTensor, scaleFloatRegTensor,
                                                                         addRegTensor, mask);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(destUbAddr, accRegTensor, addReg, mask);
        }
        return;
    }

    if (param.nCount == 1) {
        MicroAPI::DataCopy<yType, MicroAPI::LoadDist::DIST_US_B16>(scaleRegTensor, scaleAddr);
        MicroAPI::Cast<float, yType, castTrait>(scaleFloatRegTensor, scaleRegTensor, scaleCastMask);
        for (uint16_t mIdx = 0; mIdx < mCount; mIdx++) {
            MicroAPI::AddrReg addReg = MicroAPI::CreateAddrReg<float>(mIdx, mFloatStride);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(addRegTensor, addUbAddr, addReg);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(accRegTensor, accUbAddr, addReg);

            MicroAPI::MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(accRegTensor, scaleFloatRegTensor,
                                                                         addRegTensor, mask);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(destUbAddr, accRegTensor, addReg, mask);
        }

        return;
    }

    for (uint16_t nIdx = 0; nIdx < nCount; nIdx++) {
        MicroAPI::AddrReg scaleReg = MicroAPI::CreateAddrReg<yType>(nIdx, stride);
        MicroAPI::DataCopy<yType, MicroAPI::LoadDist::DIST_US_B16>(scaleRegTensor, scaleAddr, scaleReg);
        MicroAPI::Cast<float, yType, castTrait>(scaleFloatRegTensor, scaleRegTensor, scaleCastMask);
        for (uint16_t mIdx = 0; mIdx < mCount; mIdx++) {
            MicroAPI::AddrReg addReg = MicroAPI::CreateAddrReg<float>(nIdx, param.mCount * mFloatStride, mIdx,
                                                                      mFloatStride);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(addRegTensor, addUbAddr, addReg);

            MicroAPI::AddrReg destReg = MicroAPI::CreateAddrReg<float>(nIdx, mFloatStride, mIdx,
                                                                       param.nCount * mFloatStride);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(accRegTensor, accUbAddr, destReg);

            MicroAPI::MulAddDst<float, MicroAPI::MaskMergeMode::ZEROING>(accRegTensor, scaleFloatRegTensor,
                                                                         addRegTensor, mask);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(destUbAddr, accRegTensor, destReg, mask);
        }
    }
}

// 计算单个k方向的所有baseM * baseN基本块的和
LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::ProcessSingleBlock(
    uint64_t mOffset, uint64_t nOffset, uint64_t kInnerIdx)
{
    ScaleAccParam<yType> accParam;
    accParam.mCount = Min<uint64_t>(tiling_->matmulTiling.baseM, tiling_->matmulTiling.M - mOffset);

    uint64_t curN = Min<uint64_t>(tiling_->matmulTiling.baseN, tiling_->matmulTiling.N - nOffset);
    // 每一次处理64个数字
    accParam.nCount = CeilDiv<uint64_t>(curN, 64);

    LocalTensor<xType> scaleTensor = scaleDbFlag_ == IDX_0 ? scalePingLocalTensor_ : scalePongLocalTensor_;
    uint64_t scaleOffset = 0;
    if constexpr (useUb4Buff) {
        // 512：直接跨 4 个 bank 取下一个 scale，512 = 1024 / sizeof(b16)
        scaleOffset = ((dbFlag_ * tiling_->matmulTiling.stepKa + kInnerIdx) * tiling_->matmulTiling.baseK) /
                      tiling_->groupSize * 512;
    } else {
        scaleOffset = (kInnerIdx * tiling_->matmulTiling.baseK) / tiling_->groupSize * curN;
    }
    accParam.scaleAddr = (__local_mem__ xType*)scaleTensor[scaleOffset].GetPhyAddr();

    if constexpr (useUb4Buff) {
        accParam.addAddr = (__local_mem__ float*)addLocalTensor_[(((kIterateIdx_ & 2) == 0) ? 0 : add2Offset_) +
                                                                 ((kIterateIdx_ & 1) << 6)]
                               .GetPhyAddr();
    } else {
        accParam.addAddr = (__local_mem__ float*)addLocalTensor_[(kIterateIdx_ & 1) << 6].GetPhyAddr();
    }
    accParam.accAddr = (__local_mem__ float*)accLocalTensor_[((kIterateIdx_ & 1) ^ 1) << 6].GetPhyAddr();
    accParam.destUbAddr = (__local_mem__ float*)accLocalTensor_[(kIterateIdx_ & 1) << 6].GetPhyAddr();

    // 如果是第0块，只做scale量化
    if (unlikely(kIterateIdx_ % singleCoreKCount_ == 0)) {
        AscendC::VF_CALL<ScaleSingleBlockVf<yType>>(accParam);
    } else {
        AscendC::VF_CALL<AccSingleBlockVf<yType>>(accParam);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS_ASW_MIX
__aicore__ inline void WeightQuantBatchMatmulV2ASWMixKernel<LOCAL_TEMPLATE_FUNC_PARAMS_ASW_MIX>::CopyResToGM(
    uint64_t mOffset, uint64_t nOffset)
{
    DataCopyParams dataCopyParams;
    constexpr uint64_t dataBlockSize = 32;
    uint64_t curN = Min<uint64_t>(tiling_->matmulTiling.N - nOffset, tiling_->matmulTiling.baseN);
    uint64_t curNAlign64 = CeilAlign<uint64_t>(curN, 64);
    uint64_t baseBlockCount = curNAlign64 * sizeof(yType) / dataBlockSize;
    uint64_t blockLen = CeilDiv(curN * 2, dataBlockSize);
    dataCopyParams.blockLen = blockLen;
    dataCopyParams.blockCount = Min<uint64_t>(tiling_->matmulTiling.baseM, tiling_->matmulTiling.M - mOffset);
    dataCopyParams.srcStride = baseBlockCount - blockLen;
    dataCopyParams.dstStride = CeilDiv(tiling_->matmulTiling.N * 2 - curN * 2, dataBlockSize);
    uint64_t outOffset = mOffset * tiling_->matmulTiling.N + nOffset;

    DataCopy(yGlobal_[outOffset], resUbTensor_, dataCopyParams);
}
} // namespace WeightQuantBatchMatmulV2::Arch35
