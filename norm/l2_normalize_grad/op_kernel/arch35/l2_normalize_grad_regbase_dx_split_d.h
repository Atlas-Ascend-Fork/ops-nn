/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been technically reviewed for functional accuracy.
 */

/*!
 * \file l2_normalize_grad_regbase_dx_split_d.h
 * \brief L2NormalizeGrad outer-parallel split-D kernel (TilingKey 7010).
 */
#ifndef L2_NORMALIZE_GRAD_REGBASE_DX_SPLIT_D_H
#define L2_NORMALIZE_GRAD_REGBASE_DX_SPLIT_D_H

#include "l2_normalize_grad_regbase_dx_split_d_base.h"

namespace L2NormalizeGrad {

template <typename T_X>
class RegbaseDxSplitD : public RegbaseDxSplitDBase<T_X> {
    using Base = RegbaseDxSplitDBase<T_X>;
    using Base::cols_;
    using Base::FormerProcess;
    using Base::InitCommon;
    using Base::InitSplitDBuffers;
    using Base::LatterProcess;
    using Base::tasksPerCore_;
    using Base::tiling_;
    using Base::usedCoreNum_;

public:
    __aicore__ inline RegbaseDxSplitD(TPipe* pipe, const L2NormalizeGradTilingData* tilingData) : Base(pipe, tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR dy, GM_ADDR dx)
    {
        rows_ = tiling_->outer;
        cols_ = tiling_->dimLen;
        if (!InitCommon(x, y, dy, dx, cols_)) {
            return;
        }
        InitSplitDBuffers();
    }

    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        int64_t blockTail = rows_ - (usedCoreNum_ - 1) * tasksPerCore_;
        int64_t calcRowNum = coreIdx == usedCoreNum_ - 1 ? blockTail : tasksPerCore_;
        for (int64_t rowIdx = 0; rowIdx < calcRowNum; rowIdx++) {
            FormerProcess(rowIdx);
            LatterProcess(rowIdx);
        }
    }

private:
    int64_t rows_;
};
} // namespace L2NormalizeGrad
#endif // L2_NORMALIZE_GRAD_REGBASE_DX_SPLIT_D_H
