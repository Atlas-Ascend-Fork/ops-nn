/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL_V3_GRAD_NHWC_INT_H_
#define MAX_POOL_V3_GRAD_NHWC_INT_H_

#include <limits>
#include "kernel_operator.h"
#include "pool_utils/arch35/compute/pool_fast_div.h"
#include "../../pool_grad_common/arch35/pool_grad_index_common.h"

namespace MaxPoolV3GradNHWCIntNameSpace {

using namespace AscendC;

template <typename T>
constexpr uint32_t INTEGER_LANES = platform::GetVRegSize() /
                                   (sizeof(T) > sizeof(int32_t) ? sizeof(T) : sizeof(int32_t));

template <typename T>
__simd_callee__ inline void DuplicateLowestValueVF(Reg::RegTensor<T>& value)
{
    static_assert(std::is_integral<T>::value, "Integer NHWC kernels require an integer dtype");
    Reg::Duplicate(value, std::numeric_limits<T>::lowest());
}

template <typename T>
__simd_callee__ inline Reg::MaskReg ValueMaskToIndexMask(Reg::MaskReg valueMask)
{
    if constexpr (sizeof(T) == 1) {
        Reg::MaskReg halfMask;
        Reg::MaskReg indexMask;
        Reg::UnPack(halfMask, valueMask);
        Reg::UnPack(indexMask, halfMask);
        return indexMask;
    } else if constexpr (sizeof(T) == 2) {
        Reg::MaskReg indexMask;
        Reg::UnPack(indexMask, valueMask);
        return indexMask;
    } else if constexpr (sizeof(T) == 8) {
        Reg::MaskReg indexMask;
        Reg::Pack(indexMask, valueMask);
        return indexMask;
    } else {
        return valueMask;
    }
}

template <typename T>
__simd_callee__ inline Reg::MaskReg IndexMaskToValueMask(Reg::MaskReg indexMask)
{
    if constexpr (sizeof(T) == 1) {
        Reg::MaskReg halfMask;
        Reg::MaskReg valueMask;
        Reg::Pack(halfMask, indexMask);
        Reg::Pack(valueMask, halfMask);
        return valueMask;
    } else if constexpr (sizeof(T) == 2) {
        Reg::MaskReg valueMask;
        Reg::Pack(valueMask, indexMask);
        return valueMask;
    } else if constexpr (sizeof(T) == 8) {
        Reg::MaskReg valueMask;
        Reg::UnPack(valueMask, indexMask);
        return valueMask;
    } else {
        return indexMask;
    }
}

template <typename T, bool STORE>
__simd_callee__ inline void IntegerGatherScatter(Reg::RegTensor<T>& value, __ubuf__ T* address,
                                                 Reg::RegTensor<uint32_t>& index, Reg::MaskReg mask)
{
    if constexpr (sizeof(T) >= sizeof(uint32_t)) {
        if constexpr (STORE) {
            Reg::DataCopyScatter(address, value, index, mask);
        } else {
            Reg::DataCopyGather(value, address, index, mask);
        }
    } else {
        Reg::RegTensor<uint16_t> packedIndex;
        Reg::Pack(packedIndex, (Reg::RegTensor<int32_t>&)index);
        if constexpr (STORE && sizeof(T) == 1) {
            // Byte scatter consumes the even byte in each 16-bit index slot.
            Reg::RegTensor<uint16_t> expanded;
            Reg::UnPack(expanded, (Reg::RegTensor<uint8_t>&)value);
            Reg::MaskReg scatterMask;
            Reg::UnPack(scatterMask, mask);
            Reg::DataCopyScatter(address, (Reg::RegTensor<T>&)expanded, packedIndex, scatterMask);
        } else if constexpr (STORE) {
            Reg::DataCopyScatter(address, value, packedIndex, mask);
        } else if constexpr (sizeof(T) == 1) {
            // Byte gather returns raw bytes in the low half of 16-bit slots.
            // Pack the slots without converting any numeric values.
            Reg::MaskReg gatherMask;
            Reg::UnPack(gatherMask, mask);
            Reg::DataCopyGather((Reg::RegTensor<uint16_t>&)value, address, packedIndex, gatherMask);
            Reg::Pack((Reg::RegTensor<uint8_t>&)value, (Reg::RegTensor<uint16_t>&)value);
        } else {
            Reg::DataCopyGather(value, address, packedIndex, mask);
        }
    }
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

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_callee__ inline void DoSingleNCNhwcFastDiv(
    __local_mem__ T* yAddr, __local_mem__ T* gradAddr, __local_mem__ INDEX_T* argmaxAddr,
    Reg::RegTensor<int32_t>& parallelRegIndex, Reg::RegTensor<int32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
    Reg::RegTensor<uint32_t>& magicWReg, int16_t shiftW, int32_t wOutputActual, int32_t wInput, int32_t baseOffset,
    int32_t cOutputAligned, Reg::RegTensor<int32_t>& cIncReg, Reg::RegTensor<int32_t>& hLowerReg,
    Reg::RegTensor<int32_t>& wLowerReg, Reg::RegTensor<int32_t>& hUpperReg, Reg::RegTensor<int32_t>& wUpperReg)
{
    static_assert(std::is_same<INDEX_T, int32_t>::value, "NHWC SIMD argmax is int32");
    Reg::RegTensor<T> gradient;
    Reg::RegTensor<T> sum;
    Reg::RegTensor<int32_t> selected;
    Reg::RegTensor<uint32_t> hIndex;
    Reg::RegTensor<uint32_t> wIndex;
    Reg::MaskReg indexMask = Reg::UpdateMask<int32_t>(argmaxMaskCount);
    Reg::MaskReg valueMask = IndexMaskToValueMask<T>(indexMask);
    Reg::DataCopyGather(selected, argmaxAddr, (Reg::RegTensor<uint32_t>&)parallelRegIndex, indexMask);
    IntegerGatherScatter<T, false>(gradient, gradAddr, (Reg::RegTensor<uint32_t>&)parallelRegGrad, valueMask);
    IndexConvNhwcFastDiv<0>(selected, hIndex, wIndex, magicWReg, shiftW, wOutputActual, wInput, baseOffset,
                            cOutputAligned, cIncReg);
    if constexpr (IS_CHECK_RANGE != 0) {
        FilterMask2D(indexMask, hIndex, wIndex, hLowerReg, wLowerReg, hUpperReg, wUpperReg);
        valueMask = IndexMaskToValueMask<T>(indexMask);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    IntegerGatherScatter<T, false>(sum, yAddr, (Reg::RegTensor<uint32_t>&)selected, valueMask);
    Reg::Add(sum, sum, gradient, valueMask);
    IntegerGatherScatter<T, true>(sum, yAddr, (Reg::RegTensor<uint32_t>&)selected, valueMask);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_callee__ inline void DoMulNCNhwcFastDiv(
    __local_mem__ T* yAddr, __local_mem__ T* gradAddr, __local_mem__ INDEX_T* argmaxAddr,
    Reg::RegTensor<int32_t>& parallelRegIndex, Reg::RegTensor<int32_t>& parallelRegGrad, uint32_t argmaxMaskCount,
    Reg::RegTensor<uint32_t>& magicWReg, int16_t shiftW, int32_t wOutputActual, int32_t wInput, int32_t baseOffset,
    int32_t cOutputAligned, int32_t cOutputActual, const DivMagic& divC, Reg::RegTensor<int32_t>& cIncReg,
    Reg::RegTensor<int32_t>& hLowerReg, Reg::RegTensor<int32_t>& wLowerReg, Reg::RegTensor<int32_t>& hUpperReg,
    Reg::RegTensor<int32_t>& wUpperReg)
{
    Reg::RegTensor<int32_t> channelGroup;
    Reg::RegTensor<int32_t> channelOffset;
    FastDivInt32(channelGroup, cIncReg, divC);
    Reg::MaskReg allMask = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::Muls(channelOffset, channelGroup, cOutputActual, allMask);
    Reg::Sub(channelOffset, cIncReg, channelOffset, allMask);
    // Windows in the same vector are separated by ceil(kernel / stride),
    // so each active (window, channel) lane scatters to a distinct element.
    DoSingleNCNhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, argmaxMaskCount, magicWReg, shiftW,
        wOutputActual, wInput, baseOffset, cOutputAligned, channelOffset, hLowerReg, wLowerReg, hUpperReg, wUpperReg);
}

} // namespace MaxPoolV3GradNHWCIntNameSpace
#endif // MAX_POOL_V3_GRAD_NHWC_INT_H_
