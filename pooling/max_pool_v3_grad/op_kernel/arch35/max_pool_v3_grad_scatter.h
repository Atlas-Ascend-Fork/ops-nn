/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool_v3_grad_scatter.h
 * \brief NHWC backward scatter primitives.
 */

#ifndef MAX_POOL_V3_GRAD_SCATTER_H_
#define MAX_POOL_V3_GRAD_SCATTER_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../pool_grad_common/arch35/pool_grad_index_common.h"
#include "pool_utils/arch35/compute/pool_fast_div.h"
#include "pool_utils/arch35/compute/pool_grad_scatter_compute.h"

using namespace AscendC;
constexpr int64_t DOUBLE_HEAD = 2;
using ComputeType = float;
// Cast traits are supplied by PoolUtils::Compute.

namespace MaxPoolV3GradNHWCNameSpace {

template <typename T2>
__simd_callee__ inline Reg::MaskReg GenT2Mask(uint32_t& maskCount)
{
    Reg::MaskReg reg;
    if constexpr (std::is_same<T2, int64_t>::value) {
        reg = AscendC::Reg::UpdateMask<T2, AscendC::Reg::RegTraitNumTwo>(maskCount);
    } else {
        reg = AscendC::Reg::UpdateMask<T2>(maskCount);
    }
    return reg;
}

template <typename T>
__simd_callee__ inline void GradientAcc(__local_mem__ ComputeType* yAddr, Reg::RegTensor<ComputeType>& gradReg,
                                        Reg::RegTensor<T>& argmaxReg, Reg::MaskReg& pregArgmax)
{
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    AscendC::Reg::RegTensor<ComputeType> scatterAccResReg;
    AscendC::Reg::DataCopyGather(scatterAccResReg, yAddr, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
    AscendC::Reg::Add(scatterAccResReg, scatterAccResReg, gradReg, pregArgmax);
    AscendC::Reg::DataCopyScatter(yAddr, scatterAccResReg, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
}

template <typename T1, typename T2>
__simd_callee__ inline void GetConCurrentInput(Reg::RegTensor<int32_t>& argmaxReg, Reg::RegTensor<ComputeType>& gradReg,
                                               __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
                                               Reg::RegTensor<uint32_t>& parallelRegIndex,
                                               Reg::RegTensor<uint32_t>& parallelRegGrad, Reg::MaskReg& pregT1,
                                               Reg::MaskReg& pregT2)
{
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        AscendC::Reg::RegTensor<T1> gradRegT1;
        AscendC::Reg::RegTensor<uint16_t> parallelRegGradU16;
        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Cast<uint16_t, uint32_t, PoolUtils::Compute::castTraitU32U16>(
            parallelRegGradU16, (AscendC::Reg::RegTensor<uint32_t>&)parallelRegGrad, allMaskU32);
        AscendC::Reg::Pack(parallelRegGradU16, (AscendC::Reg::RegTensor<int32_t>&)parallelRegGradU16);
        AscendC::Reg::DataCopyGather(gradRegT1, gradAddr, parallelRegGradU16, pregT1);
        AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint32_t>&)gradRegT1,
                             (AscendC::Reg::RegTensor<uint16_t>&)gradRegT1);
        AscendC::Reg::Cast<ComputeType, T1, PoolUtils::Compute::castTraitT1ComputeType>(gradReg, gradRegT1, allMaskU32);
    } else {
        AscendC::Reg::DataCopyGather(gradReg, gradAddr, parallelRegGrad, pregT1);
    }

    if constexpr (std::is_same<T2, int32_t>::value) {
        AscendC::Reg::DataCopyGather(argmaxReg, argmaxAddr, parallelRegIndex, pregT2);
    } else if constexpr (std::is_same<T2, int64_t>::value) {
        AscendC::Reg::RegTensor<T2, AscendC::Reg::RegTraitNumTwo> argmaxRegTwo;
        AscendC::Reg::DataCopyGather(argmaxRegTwo, argmaxAddr, parallelRegIndex, pregT2);
        argmaxReg = (AscendC::Reg::RegTensor<int32_t>&)argmaxRegTwo.reg[0];
    }
}

template <typename T1, typename T2>
__simd_callee__ inline void GetConCurrentInput(Reg::RegTensor<int32_t>& argmaxReg, Reg::RegTensor<ComputeType>& gradReg,
                                               __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
                                               Reg::RegTensor<int32_t>& parallelRegIndex,
                                               Reg::RegTensor<int32_t>& parallelRegGrad, Reg::MaskReg& pregT1,
                                               Reg::MaskReg& pregT2)
{
    GetConCurrentInput<T1, T2>(argmaxReg, gradReg, gradAddr, argmaxAddr, (Reg::RegTensor<uint32_t>&)parallelRegIndex,
                               (Reg::RegTensor<uint32_t>&)parallelRegGrad, pregT1, pregT2);
}

struct DivMagic {
    uint32_t magic;
    int16_t shift;
};

__aicore__ inline DivMagic PrecomputeDiv(uint32_t divisor)
{
    DivMagic dm;
    uint32_t m = 0;
    uint32_t s = 0;
    GetUintDivMagicAndShift<uint32_t>(m, s, divisor);
    dm.magic = m;
    dm.shift = static_cast<int16_t>(s);
    return dm;
}

__simd_callee__ inline void FastDivInt32(Reg::RegTensor<int32_t>& res, Reg::RegTensor<int32_t>& src, const DivMagic& dm)
{
    Reg::RegTensor<uint32_t> tmp;
    Reg::RegTensor<uint32_t> magicReg;
    Reg::Duplicate(magicReg, dm.magic);
    Reg::MaskReg allMask = Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    PoolUtils::Compute::FastDivImpl(tmp, (Reg::RegTensor<uint32_t>&)src, magicReg, dm.shift, allMask);
    res = (Reg::RegTensor<int32_t>&)tmp;
}

// Argmax stores flattened spatial indices; cIncReg supplies the channel index.
// Decode using the full input width, then form offsets in tile layout [N][H][W][cAligned].
// The caller subtracts the tile origin through baseOffset.
template <const uint32_t IS_MUL_C = 0>
__simd_callee__ inline void IndexConvNhwcFastDiv(Reg::RegTensor<int32_t>& argmaxReg, Reg::RegTensor<uint32_t>& hTmpReg,
                                                 Reg::RegTensor<uint32_t>& wTmpReg, Reg::RegTensor<uint32_t>& magicWReg,
                                                 int16_t shiftW, int32_t wOutputActual, int32_t wInput,
                                                 int32_t baseOffset, int32_t cOutputAligned,
                                                 Reg::RegTensor<int32_t>& cIncReg)
{
    Reg::MaskReg allMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> remU32;

    // h = argmax / wInput
    PoolUtils::Compute::FastDivImpl(hTmpReg, (Reg::RegTensor<uint32_t>&)argmaxReg, magicWReg, shiftW, allMask);
    // w = argmax - h * wInput
    Reg::Muls(wTmpReg, hTmpReg, uint32_t(wInput), allMask);
    Reg::Sub(wTmpReg, (Reg::RegTensor<uint32_t>&)argmaxReg, wTmpReg, allMask);

    // Output offset: w * cStride + h * wOutputActual * cStride + cInc + baseOffset.
    Reg::Muls(argmaxReg, (Reg::RegTensor<int32_t>&)wTmpReg, int32_t(cOutputAligned), allMask);
    Reg::RegTensor<int32_t> hwTmpIndexReg;
    Reg::Muls(hwTmpIndexReg, (Reg::RegTensor<int32_t>&)hTmpReg, int32_t(wOutputActual * cOutputAligned), allMask);
    Reg::Add(argmaxReg, argmaxReg, hwTmpIndexReg, allMask);
    Reg::Add(argmaxReg, argmaxReg, cIncReg, allMask);
    Reg::Adds(argmaxReg, argmaxReg, baseOffset, allMask);

    if constexpr (IS_MUL_C == 1) {
        // DoMulNCNhwcFastDiv resolves channel remainders before this call.
    }
}

__simd_callee__ inline void FilterMask2D(Reg::MaskReg& preg, Reg::RegTensor<uint32_t>& hTmpReg,
                                         Reg::RegTensor<uint32_t>& wTmpReg, Reg::RegTensor<int32_t>& hLowerReg,
                                         Reg::RegTensor<int32_t>& wLowerReg, Reg::RegTensor<int32_t>& hUpperReg,
                                         Reg::RegTensor<int32_t>& wUpperReg)
{
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg hMask;
    AscendC::Reg::MaskReg wMask;
    AscendC::Reg::Compare<int32_t, CMPMODE::GE>(hMask, (AscendC::Reg::RegTensor<int32_t>&)hTmpReg, hLowerReg, allMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GE>(wMask, (AscendC::Reg::RegTensor<int32_t>&)wTmpReg, wLowerReg, allMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GT>(hMask, hUpperReg, (AscendC::Reg::RegTensor<int32_t>&)hTmpReg, hMask);
    AscendC::Reg::Compare<int32_t, CMPMODE::GT>(wMask, wUpperReg, (AscendC::Reg::RegTensor<int32_t>&)wTmpReg, wMask);
    AscendC::Reg::MaskAnd(hMask, hMask, wMask, allMask);
    AscendC::Reg::MaskAnd(preg, preg, hMask, allMask);
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__simd_callee__ inline void DoSingleNCNhwcFastDiv(
    __local_mem__ ComputeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ T2* argmaxAddr,
    Reg::RegTensor<int32_t>& parallelRegIndex, Reg::RegTensor<int32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
    Reg::RegTensor<uint32_t>& magicWReg, int16_t shiftW, int32_t wOutputActual, int32_t wInput, int32_t baseOffset,
    int32_t cOutputAligned, Reg::RegTensor<int32_t>& cIncReg, Reg::RegTensor<int32_t>& hLowerReg,
    Reg::RegTensor<int32_t>& wLowerReg, Reg::RegTensor<int32_t>& hUpperReg, Reg::RegTensor<int32_t>& wUpperReg)
{
    AscendC::Reg::RegTensor<ComputeType> gradReg;
    AscendC::Reg::RegTensor<int32_t> argmaxReg;
    AscendC::Reg::RegTensor<uint32_t> hTmpReg;
    AscendC::Reg::RegTensor<uint32_t> wTmpReg;

    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = GenT2Mask<T2>(maskT2);

    GetConCurrentInput<T1, T2>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, pregT1,
                               pregT2);
    IndexConvNhwcFastDiv<0>(argmaxReg, hTmpReg, wTmpReg, magicWReg, shiftW, wOutputActual, wInput, baseOffset,
                            cOutputAligned, cIncReg);
    if constexpr (std::is_same<T2, int32_t>::value) {
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMask2D(pregT2, hTmpReg, wTmpReg, hLowerReg, wLowerReg, hUpperReg, wUpperReg);
        }
        GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregT2);
    } else {
        uint32_t argmaxMask = argmaxMaskCount;
        AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMask2D(pregArgmax, hTmpReg, wTmpReg, hLowerReg, wLowerReg, hUpperReg, wUpperReg);
        }
        GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregArgmax);
    }
}

template <typename T1, typename T2, const uint32_t IS_CHECK_RANGE>
__simd_callee__ inline void DoMulNCNhwcFastDiv(__local_mem__ ComputeType* yAddr, __local_mem__ T1* gradAddr,
                                               __local_mem__ T2* argmaxAddr, Reg::RegTensor<int32_t>& parallelRegIndex,
                                               Reg::RegTensor<int32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
                                               Reg::RegTensor<uint32_t>& magicWReg, int16_t shiftW,
                                               int32_t wOutputActual, int32_t wInput, int32_t baseOffset,
                                               int32_t cOutputAligned, int32_t cOutputActual, const DivMagic& divC,
                                               Reg::RegTensor<int32_t>& cIncReg, Reg::RegTensor<int32_t>& hLowerReg,
                                               Reg::RegTensor<int32_t>& wLowerReg, Reg::RegTensor<int32_t>& hUpperReg,
                                               Reg::RegTensor<int32_t>& wUpperReg)
{
    // Precompute divC outside the VF to avoid division and 64-bit arithmetic in the vector domain.
    AscendC::Reg::RegTensor<ComputeType> gradReg;
    AscendC::Reg::RegTensor<int32_t> argmaxReg;
    AscendC::Reg::RegTensor<uint32_t> hTmpReg;
    AscendC::Reg::RegTensor<uint32_t> wTmpReg;
    uint32_t maskT1 = argmaxMaskCount;
    uint32_t maskT2 = argmaxMaskCount;
    AscendC::Reg::MaskReg pregT1 = AscendC::Reg::UpdateMask<T1>(maskT1);
    AscendC::Reg::MaskReg pregT2 = GenT2Mask<T2>(maskT2);
    GetConCurrentInput<T1, T2>(argmaxReg, gradReg, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, pregT1,
                               pregT2);

    // Reduce channel lane indices modulo cOutputActual.
    Reg::RegTensor<int32_t> cIdxReg;
    Reg::RegTensor<int32_t> cIncModReg;
    FastDivInt32(cIdxReg, cIncReg, divC);
    Reg::MaskReg allMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::Muls(cIncModReg, cIdxReg, cOutputActual, allMask);
    Reg::Sub(cIncModReg, cIncReg, cIncModReg, allMask);

    IndexConvNhwcFastDiv<1>(argmaxReg, hTmpReg, wTmpReg, magicWReg, shiftW, wOutputActual, wInput, baseOffset,
                            cOutputAligned, cIncModReg);

    if constexpr (std::is_same<T2, int32_t>::value) {
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMask2D(pregT2, hTmpReg, wTmpReg, hLowerReg, wLowerReg, hUpperReg, wUpperReg);
        }
        GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregT2);
    } else {
        uint32_t argmaxMask = argmaxMaskCount;
        AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMask2D(pregArgmax, hTmpReg, wTmpReg, hLowerReg, wLowerReg, hUpperReg, wUpperReg);
        }
        GradientAcc<int32_t>(yAddr, gradReg, argmaxReg, pregArgmax);
    }
}

} // namespace MaxPoolV3GradNHWCNameSpace

#endif // MAX_POOL_V3_GRAD_SCATTER_H_
