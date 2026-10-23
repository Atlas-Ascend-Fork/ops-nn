/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file threshold_grad_v2_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_THRESHOLD_GRAD_V2_DAG_H
#define CANN_CUSTOM_OPS_THRESHOLD_GRAD_V2_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace ThresholdGradV2Op {
using namespace Ops::Base;

constexpr int COMPARE_MODE_LE = 3;
constexpr int SELECT_MODE_TENSOR = 2;
#if 0
template <typename U>
struct ThresholdGradV28BDag {
    using const_zero = MAKE_CONST(float, 0.0);
    using data_zero = Bind<Vec::Duplicate<float>, const_zero>;
    using OpCopyInGrad = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyInGradHalf = Bind<Vec::Cast<half, U, 0>, OpCopyInGrad>;
    using OpCopyInGradCast = Bind<Vec::Cast<float, half, 0>, OpCopyInGradHalf>;
    using OpCopyInSelf = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyInSelfHalf = Bind<Vec::Cast<half, U, 0>, OpCopyInSelf>;
    using OpCopyInSelfCast = Bind<Vec::Cast<float, half, 0>, OpCopyInSelfHalf>;
    using OpCopyInThreshold = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using OpCopyInThresholdHalf = Bind<Vec::Cast<half, U, 0>, OpCopyInThreshold>;
    using OpCopyInThresholdCast = Bind<Vec::Cast<float, half, 0>, OpCopyInThresholdHalf>;
    using Compare = Bind<Vec::Compare<uint8_t, float, COMPARE_MODE_LE>, OpCopyInSelfCast, OpCopyInThresholdCast>;
    using Select = Bind<Vec::Select<uint8_t, float, SELECT_MODE_TENSOR>, Compare, data_zero, OpCopyInGradCast>;
    using SelectHalf = Bind<Vec::Cast<half, float, 1>, Select>;
    using SelectCast = Bind<Vec::Cast<U, half, 1>, SelectHalf>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, SelectCast>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U>
struct ThresholdGradV2Int32Dag {
    using const_zero = MAKE_CONST(float, 0.0);
    using data_zero = Bind<Vec::Duplicate<float>, const_zero>;
    using OpCopyInGrad = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyInGradCast = Bind<Vec::Cast<float, U, 1>, OpCopyInGrad>;
    using OpCopyInSelf = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyInSelfCast = Bind<Vec::Cast<float, U, 1>, OpCopyInSelf>;
    using OpCopyInThreshold = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using OpCopyInThresholdCast = Bind<Vec::Cast<float, U, 1>, OpCopyInThreshold>;
    using Compare = Bind<Vec::Compare<uint8_t, float, COMPARE_MODE_LE>, OpCopyInSelfCast, OpCopyInThresholdCast>;
    using Select = Bind<Vec::Select<uint8_t, float, SELECT_MODE_TENSOR>, Compare, data_zero, OpCopyInGradCast>;
    using SelectCast = Bind<Vec::Cast<U, float, 1>, Select>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, SelectCast>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U>
struct ThresholdGradV2Dag2 {
    using const_zero = MAKE_CONST(float, 0.0);
    using data_zero = Bind<Vec::Duplicate<float>, const_zero>;
    using OpCopyInGrad = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyInGradCast = Bind<Vec::Cast<float, U, 0>, OpCopyInGrad>;
    using OpCopyInSelf = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyInSelfCast = Bind<Vec::Cast<float, U, 0>, OpCopyInSelf>;
    using OpCopyInThreshold = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using OpCopyInThresholdCast = Bind<Vec::Cast<float, U, 0>, OpCopyInThreshold>;
    using Compare = Bind<Vec::Compare<uint8_t, float, COMPARE_MODE_LE>, OpCopyInSelfCast, OpCopyInThresholdCast>;
    using Select = Bind<Vec::Select<uint8_t, float, SELECT_MODE_TENSOR>, Compare, data_zero, OpCopyInGradCast>;
    using SelectCast = Bind<Vec::Cast<U, float, 1>, Select>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, SelectCast>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
#endif
template <typename U>
struct ThresholdGradV2Dag {
    using const_zero = MAKE_CONST(U, 0.0);
    using data_zero = Bind<Vec::Duplicate<U>, const_zero>;
    using OpCopyInGrad = Bind<Vec::CopyInBrc<U>, Placeholder::In0<U>>;
    using OpCopyInSelf = Bind<Vec::CopyInBrc<U>, Placeholder::In1<U>>;
    using OpCopyInThreshold = Bind<Vec::CopyInBrc<U>, Placeholder::In2<U>>;
    using Compare = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_LE>, OpCopyInSelf, OpCopyInThreshold>;
    using Select = Bind<Vec::Select<uint8_t, U, SELECT_MODE_TENSOR>, Compare, data_zero, OpCopyInGrad>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, Select>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace ThresholdGradV2Op
#endif // CANN_CUSTOM_OPS_THRESHOLD_GRAD_V2_DAG_H
