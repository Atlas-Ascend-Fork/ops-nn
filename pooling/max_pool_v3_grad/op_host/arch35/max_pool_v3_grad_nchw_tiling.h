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
 * \file max_pool_v3_grad_nchw_tiling.h
 * \brief NCHW small-kernel tiling template for MaxPoolV3Grad.
 */

#ifndef MAX_POOL_V3_GRAD_NCHW_TILING_H
#define MAX_POOL_V3_GRAD_NCHW_TILING_H

#include "max_pool_v3_grad_tiling_base.h"
#include "../../../pool_grad_common/op_host/arch35/max_pool_grad_nchw_tiling_common.h"

namespace optiling {

// 最小 tile 面积（元素数），避免小 shape 被切成 1x1
constexpr int64_t MIN_TILE_AREA_ELEMS = 32;

class MaxPoolV3GradNCHWTilingHelper : public MaxPoolGradNCHWTilingCommon {
public:
    explicit MaxPoolV3GradNCHWTilingHelper(MaxPoolGradWithArgmaxInputInfoCommon* input)
        : MaxPoolGradNCHWTilingCommon(input)
    {}

protected:
    void DoBufferCalculate() override;
    void SetTilingData(gert::TilingContext* context, uint64_t key) override;
    int64_t GetMinTileArea() const override { return MIN_TILE_AREA_ELEMS; }
    bool PreferHSplitFirst() const override { return baseData.hProBatchSize <= baseData.wProBatchSize; }
};

class MaxPoolV3GradNCHWTiling : public MaxPoolV3GradTilingBase {
public:
    explicit MaxPoolV3GradNCHWTiling(gert::TilingContext* context) : MaxPoolV3GradTilingBase(context) {}
    ~MaxPoolV3GradNCHWTiling() override = default;

private:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    MaxPoolV3GradNCHWTilingHelper commonTiling{&inputData};
    MaxPoolGradWithArgmaxHardwareInfo hwInfo{};
};

} // namespace optiling
#endif // MAX_POOL_V3_GRAD_NCHW_TILING_H
