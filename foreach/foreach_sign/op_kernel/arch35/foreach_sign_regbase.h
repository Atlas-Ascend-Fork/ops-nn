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
 * \file foreach_sign_regbase.h
 * \brief RegBase (register) kernel implementation for foreach_sign
 */

#ifndef FOREACH_SIGN_REGBASE_H
#define FOREACH_SIGN_REGBASE_H

#include "../../foreach_utils/arch35/foreach_regbase_unary.h"

#define FOREACH_TILING_KEY_INT8 10007
#define FOREACH_TILING_KEY_INT64 10010

namespace ForeachSign {
using namespace AscendC;

namespace SignInternal {
template <typename T, typename RegT, const Reg::RegTrait& trait = Reg::RegTraitNumOne>
__simd_vf__ inline void SignCoreVF(__ubuf__ T* dstUb, __ubuf__ T* srcUb, uint32_t calCount, uint16_t repeatTime,
                                   uint32_t vlSize)
{
    Reg::MaskReg signMask;
    Reg::MaskReg cmpMaskLt;
    Reg::MaskReg cmpMaskGt;
    RegT brcZeroReg;
    RegT brcOneReg;
    RegT brcNegOneReg;
    RegT srcReg;
    RegT selReg0;
    RegT selReg1;
    Reg::Duplicate(brcZeroReg, 0);
    Reg::Duplicate(brcOneReg, 1);
    Reg::Duplicate(brcNegOneReg, -1);
    for (uint16_t i = 0; i < repeatTime; ++i) {
        signMask = Reg::UpdateMask<T, trait>(calCount);
        Reg::LoadAlign(srcReg, srcUb + i * vlSize);
        Reg::CompareScalar<T, CMPMODE::LT>(cmpMaskLt, srcReg, 0, signMask);
        Reg::CompareScalar<T, CMPMODE::GT>(cmpMaskGt, srcReg, 0, signMask);
        Reg::Select(selReg0, brcNegOneReg, brcZeroReg, cmpMaskLt);
        Reg::Select(selReg1, brcOneReg, selReg0, cmpMaskGt);
        if constexpr (sizeof(T) == 1) {
            Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B8>(dstUb + i * vlSize, selReg1, signMask);
        } else if constexpr (sizeof(T) == 2) {
            Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>(dstUb + i * vlSize, selReg1, signMask);
        } else if constexpr (sizeof(T) == 4) {
            Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(dstUb + i * vlSize, selReg1, signMask);
        } else {
            Reg::StoreAlign(dstUb + i * vlSize, selReg1, signMask);
        }
    }
}

template <typename T>
__aicore__ inline void SignLaunch(__ubuf__ T* dstUb, __ubuf__ T* srcUb, uint32_t calCount)
{
    if constexpr (sizeof(T) == 8) {
        using RegT = Reg::RegTensor<T, Reg::RegTraitNumTwo>;
        constexpr int32_t vlSize = static_cast<int32_t>(GetVecLen() / sizeof(T) * 2);
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, vlSize));
        SignCoreVF<T, RegT, Reg::RegTraitNumTwo>(dstUb, srcUb, calCount, repeatTime, vlSize);
    } else {
        using RegT = Reg::RegTensor<T>;
        constexpr int32_t vlSize = static_cast<int32_t>(GetVecLen() / sizeof(T));
        const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, vlSize));
        SignCoreVF<T, RegT>(dstUb, srcUb, calCount, repeatTime, vlSize);
    }
}
} // namespace SignInternal

template <typename T, typename Tiling>
class ForeachSignRegbase : public ForeachRegbaseUnary<T, Tiling, ForeachSignRegbase<T, Tiling>> {
public:
    using Base = ForeachRegbaseUnary<T, Tiling, ForeachSignRegbase<T, Tiling>>;
    using Base::Process;

    __aicore__ inline ForeachSignRegbase() : Base(*this) {}

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, const Tiling* tilingData,
                                TPipe* tPipe)
    {
        Base::Init(inputs, outputs, workspace, tilingData, tPipe);
    }

    __aicore__ inline void Compute(LocalTensor<T> inLocal, LocalTensor<T> outLocal, int64_t dataCount)
    {
        __ubuf__ T* srcUb = reinterpret_cast<__ubuf__ T*>(inLocal.GetPhyAddr());
        __ubuf__ T* dstUb = reinterpret_cast<__ubuf__ T*>(outLocal.GetPhyAddr());
        SignInternal::SignLaunch<T>(dstUb, srcUb, static_cast<uint32_t>(dataCount));
    }
};

} // namespace ForeachSign

#endif // FOREACH_SIGN_REGBASE_H
