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
 * \file max_pool3d_grad_ksize_one_tiling.cpp
 * \brief MaxPool3DGrad ksize=1 专用 tiling 实现 (NCDHW).
 */

#include "max_pool3d_grad_ksize_one_tiling.h"

#include <algorithm>

#include "op_host/tiling_templates_registry.h"
#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"

namespace optiling {

bool MaxPool3DGradKsizeOneTiling::IsCapable()
{
    if (inputData.dKernel != 1 || inputData.hKernel != 1 || inputData.wKernel != 1) {
        return false;
    }
    if (inputData.inputFormat != ge::Format::FORMAT_NCDHW) {
        return false;
    }

    // scatter 索引寄存器为 16bit, shape/stride 超 65535 会静默截断导致错位, 回退通用路径
    constexpr int64_t KSIZE_ONE_MAX_U16 = 65535;
    if (inputData.dX > KSIZE_ONE_MAX_U16 || inputData.hX > KSIZE_ONE_MAX_U16 || inputData.wX > KSIZE_ONE_MAX_U16 ||
        inputData.dStride > KSIZE_ONE_MAX_U16 || inputData.hStride > KSIZE_ONE_MAX_U16 ||
        inputData.wStride > KSIZE_ONE_MAX_U16) {
        return false;
    }

    // 最小 stride 对齐切片(1 highAxis × stride×stride×stride)能否装入 UB, 放不下回退通用 NCDHW 路径
    common_->InitializationVars(context_, ubSize_, coreNum_);
    auto& split = common_->GetSplitData();
    split.highAxisInner = 1;
    split.dOutputInner = inputData.dStride;
    split.hOutputInner = inputData.hStride;
    split.wOutputInner = inputData.wStride;
    common_->DoBufferCalculate();
    if (common_->GetSplitData().totalBufferSize > common_->GetBaseData().availableUb) {
        OP_LOGI(context_, "ksize_one minimal tile exceeds UB, fallback");
        return false;
    }
    return true;
}

ge::graphStatus MaxPool3DGradKsizeOneTiling::DoOpTiling() { return common_->DoOpTiling(context_, ubSize_, coreNum_); }

ge::graphStatus MaxPool3DGradKsizeOneTiling::PostTiling() { return common_->PostTiling(context_); }

uint64_t MaxPool3DGradKsizeOneTiling::GetTilingKey() const { return common_->GetTilingKey(); }

REGISTER_TILING_TEMPLATE("MaxPool3DGrad", MaxPool3DGradKsizeOneTiling, 0);

} // namespace optiling
