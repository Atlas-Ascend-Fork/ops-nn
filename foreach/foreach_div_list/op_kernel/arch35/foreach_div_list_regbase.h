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
 * \file foreach_div_list_regbase.h
 * \brief arch35 (DAV_3510) register-VF compute policy for ForeachDivList.
 *        The transport (3 depth-2 queues, DataCopyPad in/out, per-tensor cursor,
 *        empty-tensor skip) is the shared ForeachFlatRegbase::FlatFlowKernel.
 *        This file only owns the per-tile math: every dtype is unpacked into
 *        float lanes and divided by one hardware vdiv (1 ULP). Subnormal
 *        operands are pre-normalized by 2^24 with a per-lane select (the
 *        quotient is restored by a matching power-of-two output scale), so
 *        legal FP32/BF16 subnormal inputs keep their finite quotient while the
 *        common normal path stays on the fast vdiv. Packed back with a
 *        mask-decremented preg so tile/tensor tails never write outside the
 *        current tile.
 */

#ifndef FOREACH_DIV_LIST_REGBASE_H
#define FOREACH_DIV_LIST_REGBASE_H

#include "kernel_operator.h"
#if __has_include("../../foreach_utils/arch35/foreach_flat_flow_regbase.h")
#include "../../foreach_utils/arch35/foreach_flat_flow_regbase.h"
#else
#include "../../../foreach_utils/op_kernel/arch35/foreach_flat_flow_regbase.h"
#endif
#if __has_include("op_kernel/load_store_utils.h")
#include "op_kernel/load_store_utils.h"
#else
#include "../../../common/inc/op_kernel/load_store_utils.h"
#endif

namespace NsForeachDivList {
using namespace AscendC;

// Hardware vdiv (1 ULP, FTZ) is the fast path; subnormal operands are scaled
// into the normal range before the divide and the quotient is scaled back, so
// the reviewer's 0x0014440B/0x0053E3CF pair yields 0.2416 instead of 0/0=NaN
// without paying the long software IEEE-754 sequence per tile.
constexpr AscendC::Reg::DivSpecificMode FOREACH_DIV_LIST_FAST_DIV = {AscendC::Reg::MaskMergeMode::ZEROING, false,
                                                                     AscendC::DivAlgo::PRECISION_1ULP_FTZ_TRUE};
constexpr float FOREACH_DIV_MIN_NORMAL = 1.1754943508222875e-38f; // 2^-126
constexpr float FOREACH_DIV_SCALE_UP = 16777216.0f;               // 2^24
constexpr float FOREACH_DIV_SCALE_DOWN = 5.9604644775390625e-08f; // 2^-24

template <typename T>
struct DivListPolicy {
    static constexpr bool kUsesExtraUb = false;

    // Released addc_list pattern: an __aicore__ call site owning one __VEC_SCOPE__.
    // LoadOneTensorForDtypeT unpacks 16-bit with DIST_UNPACK_B16, fp32 is a plain
    // load; StoreOneTensorForDtypeT packs with CAST_RINT under the same preg, so
    // the decremented mask covers the tile/tensor tail for every dtype.
    __aicore__ static inline void Run(LocalTensor<T>* inputs, LocalTensor<T> output, float unusedScalar,
                                      uint32_t dataCount)
    {
        (void)unusedScalar;
        __local_mem__ T* x1Addr = reinterpret_cast<__local_mem__ T*>(inputs[0].GetPhyAddr());
        __local_mem__ T* x2Addr = reinterpret_cast<__local_mem__ T*>(inputs[1].GetPhyAddr());
        __local_mem__ T* outAddr = reinterpret_cast<__local_mem__ T*>(output.GetPhyAddr());
        constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
        uint16_t repeatTimes = static_cast<uint16_t>((dataCount + VL - 1) / VL);
        uint32_t sreg = dataCount;
        __VEC_SCOPE__
        {
            Reg::MaskReg mask;
            Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::RegTensor<float, Reg::RegTraitNumOne> x1Reg;
            Reg::RegTensor<float, Reg::RegTraitNumOne> x2Reg;
            Reg::RegTensor<float, Reg::RegTraitNumOne> outReg;
            Reg::RegTensor<float, Reg::RegTraitNumOne> abs1Reg;
            Reg::RegTensor<float, Reg::RegTraitNumOne> abs2Reg;
            Reg::RegTensor<float, Reg::RegTraitNumOne> x1Scaled;
            Reg::RegTensor<float, Reg::RegTraitNumOne> x2Scaled;
            Reg::RegTensor<float, Reg::RegTraitNumOne> scale1;
            Reg::RegTensor<float, Reg::RegTraitNumOne> scale2;
            Reg::RegTensor<float, Reg::RegTraitNumOne> outScale;
            Reg::RegTensor<float, Reg::RegTraitNumOne> oneReg;
            Reg::RegTensor<float, Reg::RegTraitNumOne> scaleUpReg;
            Reg::RegTensor<float, Reg::RegTraitNumOne> scaleDownReg;
            Reg::Duplicate(oneReg, 1.0f, all);
            Reg::Muls(scaleUpReg, oneReg, FOREACH_DIV_SCALE_UP, all);
            Reg::Muls(scaleDownReg, oneReg, FOREACH_DIV_SCALE_DOWN, all);
            for (uint16_t i = 0; i < repeatTimes; ++i) {
                mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(sreg);
                ops::LoadOneTensorForDtypeT<T>(x1Addr, x1Reg, mask, i * VL);
                ops::LoadOneTensorForDtypeT<T>(x2Addr, x2Reg, mask, i * VL);
                // Normalize subnormal operands by 2^24; the common normal path
                // keeps scale=1 and runs the single hardware vdiv.
                Reg::Abs(abs1Reg, x1Reg, mask);
                Reg::Abs(abs2Reg, x2Reg, mask);
                Reg::MaskReg sub1;
                Reg::MaskReg sub2;
                Reg::MaskReg not1;
                Reg::MaskReg not2;
                Reg::MaskReg only1;
                Reg::MaskReg only2;
                Reg::Compares<float, CMPMODE::LT>(sub1, abs1Reg, FOREACH_DIV_MIN_NORMAL, mask);
                Reg::Compares<float, CMPMODE::LT>(sub2, abs2Reg, FOREACH_DIV_MIN_NORMAL, mask);
                Reg::Select(scale1, scaleUpReg, oneReg, sub1);
                Reg::Select(scale2, scaleUpReg, oneReg, sub2);
                Reg::Mul(x1Scaled, x1Reg, scale1, mask);
                Reg::Mul(x2Scaled, x2Reg, scale2, mask);
                Reg::Div<float, &FOREACH_DIV_LIST_FAST_DIV>(outReg, x1Scaled, x2Scaled, mask);
                // Restore the quotient: only-x1-scaled -> x 2^-24, only-x2-scaled -> x 2^24.
                Reg::Not(not1, sub1, mask);
                Reg::Not(not2, sub2, mask);
                Reg::And(only1, sub1, not2, mask);
                Reg::And(only2, not1, sub2, mask);
                Reg::Select(outScale, scaleDownReg, oneReg, only1);
                Reg::Select(outScale, scaleUpReg, outScale, only2);
                Reg::Mul(outReg, outReg, outScale, mask);
                ops::StoreOneTensorForDtypeT<T>(outAddr, outReg, mask, i * VL);
            }
        }
    }
};

template <typename T, typename Tiling>
using ForeachDivListRegbase = ForeachFlatRegbase::FlatFlowKernel<T, float, Tiling, 2, DivListPolicy<T>>;
} // namespace NsForeachDivList
#endif // FOREACH_DIV_LIST_REGBASE_H
