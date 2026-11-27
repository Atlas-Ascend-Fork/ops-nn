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
 * \file foreach_log2_flat_regbase.h
 * \brief Register-only compute policy for ForeachLog2 on Ascend950 (arch35 flat transport).
 *        Computes y = log2(x) for every tensor of the list. FP32 and FP16 use the native
 *        dav_3510 Log2 (the FP16 form converts to FP32, takes vln, multiplies by 1/ln2 and
 *        converts back - the same FP32 evaluation the retired SIMT path performed); BF16 is
 *        promoted to FP32 and rounded back by the shared unary policy, because Log2Impl only
 *        admits half/float.
 */

#ifndef FOREACH_LOG2_FLAT_REGBASE_H
#define FOREACH_LOG2_FLAT_REGBASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../foreach_utils/arch35/foreach_unary_regbase_common.h"

namespace ForeachLog2 {
using namespace AscendC;

// The default intrinsic follows the compiler's FTZ setting. Preserve positive
// subnormal inputs explicitly so they have the same finite log2 result as the
// original SIMT path. The SDK supplies the normalization and compensation.
constexpr Reg::Log2SpecificMode LOG2_PRESERVE_SUBNORMAL = {Reg::MaskMergeMode::ZEROING,
                                                           Log2Algo::PRECISION_1ULP_FTZ_FALSE};

struct Log2Operation {
    static constexpr bool kPromoteHalf = false;

    template <typename T>
    __simd_callee__ static inline void Apply(Reg::RegTensor<T, Reg::RegTraitNumOne>& dstReg,
                                             Reg::RegTensor<T, Reg::RegTraitNumOne>& srcReg, Reg::MaskReg& mask)
    {
        Reg::Log2<T, &LOG2_PRESERVE_SUBNORMAL>(dstReg, srcReg, mask);
    }
};

template <typename T, typename Tiling>
using ForeachLog2Flat = ForeachUnaryRegbase::UnaryApplyKernel<T, Tiling, Log2Operation>;
} // namespace ForeachLog2

#endif // FOREACH_LOG2_FLAT_REGBASE_H
