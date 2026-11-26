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
 * \file pool3d_grad_ksize_one_tiling_common.h
 * \brief MaxPool3DGrad ksize=1 (kD=kH=kW=1) 公共 tiling (NCDHW).
 *        复用 small kernel 的轴切块逻辑, 仅重写 buffer 计算 (无 argmax/forward, y 为 strided scatter).
 */

#ifndef POOL3D_GRAD_KSIZE_ONE_TILING_COMMON_H
#define POOL3D_GRAD_KSIZE_ONE_TILING_COMMON_H

#include "pool3d_grad_ncdhw_small_kernel_tiling.h"

namespace optiling {

class Pool3DGradKsizeOneCommonTiling : public Pool3DGradNCDHWSmallKernelCommonTiling {
public:
    explicit Pool3DGradKsizeOneCommonTiling(Pool3DGradNCDHWInputInfo* input)
        : Pool3DGradNCDHWSmallKernelCommonTiling(input), inputData_(input)
    {}

    ~Pool3DGradKsizeOneCommonTiling() override = default;

    ge::graphStatus DoOpTiling(gert::TilingContext* context, int64_t ubSize, int64_t coreNum);
    ge::graphStatus PostTiling(gert::TilingContext* context);
    uint64_t GetTilingKey() const;

    void DoBufferCalculate() override;
    bool IsMeetUBSize() override;

private:
    Pool3DGradNCDHWInputInfo* inputData_;
};

} // namespace optiling

#endif // POOL3D_GRAD_KSIZE_ONE_TILING_COMMON_H
