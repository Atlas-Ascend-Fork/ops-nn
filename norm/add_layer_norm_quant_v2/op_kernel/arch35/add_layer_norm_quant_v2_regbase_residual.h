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
 * \file add_layer_norm_quant_v2_regbase_residual.h
 * \brief ascend950 (arch35/regbase) x-load helpers local to AddLayerNormQuantV2.
 *
 *        The shared arch35 family loader (add_layer_norm_regbase_common.h
 *        LoadInputsToRegWithBias) accumulates x = (x2 + bias) + x1, while the 910-family
 *        kernels of this operator and the frozen golden compute x = (x1 + x2) + bias.
 *        Float addition is not associative, so x (and every downstream output derived
 *        from it, including the int8 quantized ones) must be summed in exactly that
 *        order to stay comparable. The shared loader is deliberately left untouched:
 *        any edit to it recompiles add_layer_norm / add_layer_norm_quant arch35, which
 *        this change cannot cover-verify. Same approach as the existing precedent
 *        norm/quantize_add_layer_norm/op_kernel/arch35/quantize_add_layer_norm_regbase_residual.h.
 *
 *        The helpers below are local copies of the shared originals with only the
 *        association order corrected -- keep them in sync with
 *        add_layer_norm_regbase_common.h otherwise. They are __simd_callee__ helpers so
 *        they can be called from the __VEC_SCOPE__ bodies of this operator (cf.
 *        add_rms_norm_quant arch35 QuantDiv, also a callee called from __VEC_SCOPE__).
 */

#ifndef ADD_LAYER_NORM_QUANT_V2_REGBASE_RESIDUAL_H
#define ADD_LAYER_NORM_QUANT_V2_REGBASE_RESIDUAL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../../add_layer_norm/arch35/add_layer_norm_regbase_common.h"

namespace AddLayerNormQuantRegbase {
using namespace AddLayerNorm;
using namespace AscendC;

// load one operand and cast to fp32; same instruction shape as the inline blocks of the shared loader
template <typename T>
__simd_callee__ inline void LoadInputToFp32V2(__ubuf__ T* addr, uint32_t offset, RegTensor<float>& dst, MaskReg& preg)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dst, (__ubuf__ T*)addr + offset);
    } else {
        RegTensor<T> raw;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, (__ubuf__ T*)addr + offset);
        Cast<float, T, castTraitB162B32>(dst, raw, preg);
    }
}

// x1+x2 first, then + bias: matches the 910-family kernels and the frozen golden.
// Local variant of AddLayerNorm::LoadInputsToRegWithBias (which is (x2 + bias) + x1).
template <typename X1_TYPE, typename X2_TYPE, typename BIAS_TYPE>
__simd_callee__ inline void LoadInputsToRegWithBiasX1X2First(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X2_TYPE* x2Addr,
                                                             __ubuf__ BIAS_TYPE* biasAddr, RegTensor<float>& dst,
                                                             MaskReg& preg, uint32_t offset0, uint32_t offset1,
                                                             uint32_t offset2)
{
    RegTensor<float> x1Fp32, x2Fp32, biasFp32;

    LoadInputToFp32V2(x1Addr, offset0, x1Fp32, preg);
    LoadInputToFp32V2(x2Addr, offset1, x2Fp32, preg);
    Add<float>(dst, x1Fp32, x2Fp32, preg);
    LoadInputToFp32V2(biasAddr, offset2, biasFp32, preg);
    Add<float>(dst, dst, biasFp32, preg);
}

// dispatcher mirroring AddLayerNorm::LoadInputsToReg. The bias-none path is a plain
// two-operand add (no association-order question) and is inlined via LoadInputToFp32V2
// with the exact instruction sequence of AddLayerNorm::LoadInputsToRegWithBiasNone.
template <typename X1_TYPE, typename X2_TYPE, typename BIAS_TYPE, int TILING_KEY>
__simd_callee__ inline void LoadInputsToRegX1X2First(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X2_TYPE* x2Addr,
                                                     __ubuf__ BIAS_TYPE* biasAddr, RegTensor<float>& dst, MaskReg& preg,
                                                     uint32_t offset0, uint32_t offset1, uint32_t offset2)
{
    if constexpr (IS_BIAS_NONE) {
        RegTensor<float> x1Fp32, x2Fp32;
        LoadInputToFp32V2(x1Addr, offset0, x1Fp32, preg);
        LoadInputToFp32V2(x2Addr, offset1, x2Fp32, preg);
        Add<float>(dst, x1Fp32, x2Fp32, preg);
    } else {
        LoadInputsToRegWithBiasX1X2First<X1_TYPE, X2_TYPE, BIAS_TYPE>(x1Addr, x2Addr, biasAddr, dst, preg, offset0,
                                                                      offset1, offset2);
    }
}

// local copy of AddLayerNorm::VFWelfordParallelUpdateCommon with the x order fixed;
// the register instruction sequence is a verbatim copy of the shared original except
// for the loader call and the address extraction done by the caller-side wrapper.
template <bool INIT, typename X1_TYPE, typename X2_TYPE, typename BIAS_TYPE, int TILING_KEY>
__aicore__ inline void VFWelfordParallelUpdateX1X2First(__ubuf__ X1_TYPE* x1Local, __ubuf__ X2_TYPE* x2Local,
                                                        __ubuf__ BIAS_TYPE* biasLocal, __ubuf__ BIAS_TYPE* xOutLocal,
                                                        __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                                        uint64_t calLen, uint16_t loopCount, float scale)
{
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> tmpMean;
        RegTensor<float> tmpVar;
        RegTensor<float> delta1;
        RegTensor<float> delta2;
        RegTensor<float> delat4;
        RegTensor<float> delta3;
        MaskReg pregLoop;
        uint32_t sreg0 = calLen;
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadInputsToRegX1X2First<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(x1Local, x2Local, biasLocal, x1, pregLoop,
                                                                              i * VL_FP32, i * VL_FP32, i * VL_FP32);
            StoreRegToOutput(xOutLocal, x1, pregLoop, i * VL_FP32);
            if constexpr (INIT) {
                Duplicate(tmpMean, 0.0, pregLoop);
            } else {
                LoadAlign(tmpMean, tmpMeanLocal + i * VL_FP32);
            }
            Sub(delta1, x1, tmpMean, pregLoop);
            Muls(delta2, delta1, scale, pregLoop);
            Add(tmpMean, tmpMean, delta2, pregLoop);
            StoreAlign(tmpMeanLocal + i * VL_FP32, tmpMean, pregLoop);

            if constexpr (INIT) {
                Duplicate(tmpVar, 0.0, pregLoop);
            } else {
                LoadAlign(tmpVar, tmpVarLocal + i * VL_FP32);
            }
            Sub(delta3, x1, tmpMean, pregLoop);
            Mul(delat4, delta1, delta3, pregLoop);
            Add(tmpVar, tmpVar, delat4, pregLoop);
            StoreAlign(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
        }
    }
}

} // namespace AddLayerNormQuantRegbase

#endif // ADD_LAYER_NORM_QUANT_V2_REGBASE_RESIDUAL_H
