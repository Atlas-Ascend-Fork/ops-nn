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
 * \file pool3d_grad_ksize_one_tiling_common.cpp
 * \brief MaxPool3DGrad ksize=1 公共 tiling 实现 (NCDHW), 复用 small kernel 轴切块.
 */

#include "pool3d_grad_ksize_one_tiling_common.h"

#include <algorithm>
#include <limits>

#include "op_common/op_host/util/platform_util.h"
#include "util/math_util.h"

namespace optiling {
using namespace Pool3DGradNameSpace;

static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();

uint64_t Pool3DGradKsizeOneCommonTiling::GetTilingKey() const
{
    uint32_t idxDtype = TPL_INT32;
    uint32_t isSimt = 0;
    uint32_t isChannelLast = 0;
    uint32_t isCheckRange = 0;
    uint32_t useINT64Index = 0;
    uint32_t isKsizeOne = 1;
    return GET_TPL_TILING_KEY(idxDtype, isSimt, isChannelLast, isCheckRange, useINT64Index, isKsizeOne);
}

void Pool3DGradKsizeOneCommonTiling::DoBufferCalculate()
{
    Pool3DGradNCDHWSplitInfo& split = GetSplitData();
    Pool3DGradNCDHWBaseInfo& base = GetBaseData();

    // kernel=1: 输出轴上 strided 位置映射为连续的 grad 子块
    int64_t dGradInner = Ops::Base::CeilDiv(split.dOutputInner + inputData_->dKernel - 1, inputData_->dStride);
    int64_t hGradInner = Ops::Base::CeilDiv(split.hOutputInner + inputData_->hKernel - 1, inputData_->hStride);
    int64_t wGradInner = Ops::Base::CeilDiv(split.wOutputInner + inputData_->wKernel - 1, inputData_->wStride);
    // grad / y 行 stride 均按 32 字节块对齐, 与 small kernel 一致
    int64_t wGradInnerAligned = Ops::Base::CeilAlign(wGradInner, base.maxDataNumInOneBlock);
    int64_t wOutputInnerAligned = Ops::Base::CeilAlign(split.wOutputInner, base.maxDataNumInOneBlock);

    int64_t gradPlaneDHW = dGradInner * hGradInner * wGradInnerAligned;
    int64_t outputPlaneDHW = split.dOutputInner * split.hOutputInner * wOutputInnerAligned;

    split.gradBufferSize = split.highAxisInner * gradPlaneDHW * base.inputBytes;
    split.outputBufferSize = split.highAxisInner * outputPlaneDHW * base.inputBytes;
    split.argmaxBufferSize = 0;
    split.inputBufferSize = 0;
    split.totalBufferSize = (split.gradBufferSize + split.outputBufferSize) * DOUBLE_BUFFER;
}

bool Pool3DGradKsizeOneCommonTiling::IsMeetUBSize()
{
    DoBufferCalculate();
    Pool3DGradNCDHWSplitInfo& split = GetSplitData();
    Pool3DGradNCDHWBaseInfo& base = GetBaseData();
    if (base.inputBytes == FLOAT16_SIZE) {
        // fp16 散落索引为 uint16(上限 65535): grad 块(扁平索引)与 y 块(散落偏移)均需 <= 65535 元素,
        // 避免大 UB(如 384KB) 场景下 uint16 索引溢出
        return split.totalBufferSize <= base.availableUb &&
               split.gradBufferSize <= MAX_INPUT_ELEMENTS * base.inputBytes &&
               split.outputBufferSize <= MAX_INPUT_ELEMENTS * base.inputBytes;
    }
    return split.totalBufferSize <= base.availableUb;
}

ge::graphStatus Pool3DGradKsizeOneCommonTiling::DoOpTiling(gert::TilingContext* context, int64_t ubSize,
                                                           int64_t coreNum)
{
    InitializationVars(context, ubSize, coreNum);
    return Pool3DGradNCDHWSmallKernelCommonTiling::DoOpTiling(context);
}

ge::graphStatus Pool3DGradKsizeOneCommonTiling::PostTiling(gert::TilingContext* context)
{
    return Pool3DGradNCDHWSmallKernelCommonTiling::PostTiling(context, GetTilingKey());
}

} // namespace optiling
