/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file smooth_l1_loss_v2_dag.h
 * \brief
 */

#ifndef SMOOTH_L1_LOSS_V2_DAG_H
#define SMOOTH_L1_LOSS_V2_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"
#ifdef __CCE_AICORE__
#include "kernel_operator.h"
#endif

namespace SmoothL1LossV2 {

using namespace Ops::Base;

constexpr int SCALAR_INDEX_2 = 2;
constexpr int SCALAR_INDEX_3 = 3;

#ifdef __CCE_AICORE__
using namespace AscendC;

constexpr static MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                   MicroAPI::MaskMergeMode::ZEROING,
                                                   RoundMode::UNKNOWN}; // bf16/fp16 --float

constexpr static MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                   MicroAPI::MaskMergeMode::ZEROING,
                                                   RoundMode::CAST_RINT}; // float---bf16/fp16

// |xi - yi| < sigma ? (xi - yi) ** 2 * 0.5 / sigma : |xi - yi| - 0.5 * sigma
// multiplyValue = 1 / sigma, addsValue = -0.5 * sigma
template <typename T, typename U = float, typename OutT = T>
__simd_vf__ inline void CalcSmoothL1LossV2VF(__ubuf__ T* predictAddr, __ubuf__ T* labelAddr, __ubuf__ OutT* resultAddr,
                                             U sigmaUp, U multiplyValueUp, U addsValueUp, uint16_t repeatTimes,
                                             uint32_t oneRepeat, uint32_t totalLen)
{
    MicroAPI::MaskReg pregUp;
    MicroAPI::MaskReg regMask;
    MicroAPI::RegTensor<U> regPredict;
    MicroAPI::RegTensor<U> regLabel;
    MicroAPI::RegTensor<U> regSub;
    MicroAPI::RegTensor<U> regAbs;
    MicroAPI::RegTensor<U> regL2Term;
    MicroAPI::RegTensor<U> regL1Term;
    MicroAPI::RegTensor<U> regResult;

    for (uint16_t loop = 0; loop < repeatTimes; loop++) {
        pregUp = MicroAPI::UpdateMask<U>(totalLen);

        // 加载 predict / label：fp16/bf16 走 LoadAlign + UnPack + Cast 宽化路径，fp32 直接 LoadAlign
        if constexpr (IsSameType<U, float>::value && !IsSameType<T, U>::value) {
            MicroAPI::RegTensor<T> regTmp;
            MicroAPI::RegTensor<T> regCopyIn;
            MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regCopyIn, predictAddr, (int32_t)oneRepeat);
            MicroAPI::UnPack((MicroAPI::RegTensor<int32_t>&)regTmp, (MicroAPI::RegTensor<int16_t>&)regCopyIn);
            MicroAPI::Cast<U, T, castTrait0>(regPredict, regTmp, pregUp);
            MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regCopyIn, labelAddr, (int32_t)oneRepeat);
            MicroAPI::UnPack((MicroAPI::RegTensor<int32_t>&)regTmp, (MicroAPI::RegTensor<int16_t>&)regCopyIn);
            MicroAPI::Cast<U, T, castTrait0>(regLabel, regTmp, pregUp);
        } else {
            MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regPredict, predictAddr,
                                                                            (int32_t)oneRepeat);
            MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regLabel, labelAddr, (int32_t)oneRepeat);
        }

        MicroAPI::Sub(regSub, regPredict, regLabel, pregUp);
        MicroAPI::Abs(regAbs, regSub, pregUp);
        MicroAPI::Muls(regL2Term, regSub, 0.5f, pregUp);
        MicroAPI::Mul(regL2Term, regSub, regL2Term, pregUp);
        MicroAPI::Muls(regL2Term, regL2Term, multiplyValueUp, pregUp);
        MicroAPI::Adds(regL1Term, regAbs, addsValueUp, pregUp);
        MicroAPI::Compares<U, CMPMODE::LT>(regMask, regAbs, sigmaUp, pregUp);
        MicroAPI::Select(regResult, regL2Term, regL1Term, regMask);

        // 存储结果：fp16/bf16 走 Cast + Pack + MaskPack + StoreAlign 窄化路径，fp32 直接 StoreAlign
        if constexpr (IsSameType<U, float>::value && !IsSameType<OutT, U>::value) {
            MicroAPI::RegTensor<OutT> regTmp;
            MicroAPI::RegTensor<OutT> regCopyOut;
            MicroAPI::MaskReg pregT;
            MicroAPI::Cast<OutT, U, castTrait1>(regTmp, regResult, pregUp);
            MicroAPI::Pack((MicroAPI::RegTensor<uint16_t>&)regCopyOut, (MicroAPI::RegTensor<uint32_t>&)regTmp);
            MicroAPI::MaskPack(pregT, pregUp);
            MicroAPI::StoreAlign<OutT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(resultAddr, regCopyOut,
                                                                                (int32_t)oneRepeat, pregT);
        } else {
            MicroAPI::StoreAlign<OutT, MicroAPI::PostLiteral::POST_MODE_UPDATE>(resultAddr, regResult,
                                                                                (int32_t)oneRepeat, pregUp);
        }
    }
}
#endif

// 自定义 VF 节点：CopyIn0/CopyIn1 -> CalcOp -> CopyOut，将 Cast/Sub/Abs/Muls/Mul/Adds/Compare/Select
// 细粒度节点融合为单节点，降低 DAG 活跃节点数、提升 UB tile 大小。
// 输出类型 OutT：elemwise 场景为 T（回归原始类型），sum/mean 场景为 PromteT（供 ReduceSumOp 消费）。
template <typename T, typename U = float, typename OutT = T>
struct CalcSmoothL1LossV2Op : public Ops::Base::Vec::ElemwiseQuinaryOP<OutT, T, T, U, U, U> {
    __aicore__ inline CalcSmoothL1LossV2Op(Ops::Base::LocalTensor<OutT>& y, Ops::Base::LocalTensor<T>& predict,
                                           Ops::Base::LocalTensor<T>& label, U& sigma, U& multiplyValue, U& addsValue,
                                           int32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t oneRepeat = Ops::Base::GetVRegSize() / sizeof(U);
        uint32_t totalLen = count;
        uint16_t repeatTimes = (uint16_t)Ops::Base::CeilDiv<uint32_t>(totalLen, oneRepeat);

        __ubuf__ T* predictAddr = (__ubuf__ T*)predict.GetPhyAddr();
        __ubuf__ T* labelAddr = (__ubuf__ T*)label.GetPhyAddr();
        __ubuf__ OutT* resultAddr = (__ubuf__ OutT*)y.GetPhyAddr();

        CalcSmoothL1LossV2VF<T, U, OutT>(predictAddr, labelAddr, resultAddr, sigma, multiplyValue, addsValue,
                                         repeatTimes, oneRepeat, totalLen);
#endif
    }
};

template <typename T, typename PromteT = float>
struct SmoothL1LossV2OpDag {
    // 通过Compute构造计算图
    // |xi - yi| < sigma ? (xi - yi) ** 2  * 0.5 / sigma  :  |xi - yi| - 0.5 * sigma
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using CalcOp = Bind<CalcSmoothL1LossV2Op<T, PromteT, T>, OpCopyIn0, OpCopyIn1, Placeholder::Var<PromteT, 0>,
                        Placeholder::Var<PromteT, 1>, Placeholder::Var<PromteT, SCALAR_INDEX_2>>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CalcOp>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    // 指定计算顺序
    using OpDag = DAGSch<Outputs>;
};

template <typename T, typename PromteT>
struct SmoothL1LossV2SumDag {
    // 通过Compute构造计算图
    // |xi - yi| < sigma ? (xi - yi) ** 2  * 0.5 / sigma  :  |xi - yi| - 0.5 * sigma
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using CalcOp = Bind<CalcSmoothL1LossV2Op<T, PromteT, PromteT>, OpCopyIn0, OpCopyIn1, Placeholder::Var<PromteT, 0>,
                        Placeholder::Var<PromteT, 1>, Placeholder::Var<PromteT, SCALAR_INDEX_2>>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, CalcOp>;
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename PromteT>
struct SmoothL1LossV2MeanDag {
    // 通过Compute构造计算图
    // |xi - yi| < sigma ? (xi - yi) ** 2  * 0.5 / sigma  :  |xi - yi| - 0.5 * sigma
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using CalcOp = Bind<CalcSmoothL1LossV2Op<T, PromteT, PromteT>, OpCopyIn0, OpCopyIn1, Placeholder::Var<PromteT, 0>,
                        Placeholder::Var<PromteT, 1>, Placeholder::Var<PromteT, SCALAR_INDEX_2>>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, CalcOp>;
    using Mul0 = Bind<Vec::Muls<PromteT>, ReduceOp0, Placeholder::Var<PromteT, SCALAR_INDEX_3>>;
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, Mul0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace SmoothL1LossV2

#endif // SMOOTH_L1_LOSS_V2_DAG_H
