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
 * \file conv3d_dimension_strategy.cpp
 * \brief
 */

#include "conv3d_dimension_strategy.h"

namespace conv_tiling {

uint64_t Conv3DAPIStrategy::ValidateAndAdjustKL1(uint64_t kValue, uint64_t scaleFactor) const
{
    if (scaleFactor == 0 || kValue == 0) {
        return kValue;
    }
    if (bbMode_ == nullptr || bbMode_->GetTilingIns() == nullptr) {
        return kValue;
    }

    uint64_t c0 = bbMode_->GetTilingIns()->cubeInfo.k0;
    uint64_t c1 = bbMode_->GetTilingIns()->shapeInfo.singleCi1;
    if (c0 == 0 || c1 == 0) {
        return kValue;
    }
    uint64_t cin = c0 * c1;
    uint64_t cinPerDin = kValue / scaleFactor;

    if (cinPerDin >= cin) {
        return cin * scaleFactor;
    }

    uint64_t best = c0;
    for (uint64_t f = 1; f <= c1; ++f) {
        if (c1 % f != 0) {
            continue;
        }
        uint64_t amount = c0 * f;
        if (amount > cinPerDin) {
            break;
        }
        best = amount;
    }
    return best;
}

uint64_t Conv3DAPIStrategy::CalcDinFromCinSlice(uint64_t cinSlice, uint64_t cin) const
{
    if (cin == 0 || cinSlice == 0) {
        return 1;
    }
    if (cin % cinSlice == 0) {
        return 1;
    }
    if (bbMode_ == nullptr || bbMode_->GetTilingIns() == nullptr) {
        return 1;
    }
    uint64_t kd = bbMode_->GetTilingIns()->shapeInfo.orgkD;
    uint64_t m = cinSlice / cin;
    if (m >= 1 && m <= kd) {
        return m;
    }
    if (m > kd) {
        return kd;
    }
    return 1;
}

} // namespace conv_tiling
