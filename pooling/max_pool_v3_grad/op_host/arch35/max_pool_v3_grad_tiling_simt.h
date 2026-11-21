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
 * \file max_pool_v3_grad_tiling_simt.h
 * \brief SIMT tiling template for MaxPoolV3Grad.
 */

#ifndef MAX_POOL_V3_GRAD_TILING_SIMT_H
#define MAX_POOL_V3_GRAD_TILING_SIMT_H

#include "max_pool_v3_grad_tiling_base.h"

namespace optiling {

class MaxPoolV3GradSimtTiling : public MaxPoolV3GradTilingBase {
public:
    explicit MaxPoolV3GradSimtTiling(gert::TilingContext* context) : MaxPoolV3GradTilingBase(context) {}
    ~MaxPoolV3GradSimtTiling() override = default;

private:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    int32_t needCoreNum_{1};
    bool needInt64Indices_{false};
};

} // namespace optiling
#endif // MAX_POOL_V3_GRAD_TILING_SIMT_H
