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
 * \file max_pool3d_grad_ksize_one_tiling.h
 * \brief MaxPool3DGrad ksize=1 (kD=kH=kW=1) 专用 tiling (NCDHW).
 *        复用 MaxPool3DGradTilingBase 的参数解析, tiling 数学下沉到公共 Pool3DGradKsizeOneCommonTiling.
 */

#ifndef MAX_POOL3D_GRAD_KSIZE_ONE_TILING_H_
#define MAX_POOL3D_GRAD_KSIZE_ONE_TILING_H_

#include <memory>

#include "max_pool3d_grad_tiling.h"
#include "../../../pool_grad_common/op_host/arch35/pool3d_grad_ksize_one_tiling_common.h"

namespace optiling {

class MaxPool3DGradKsizeOneTiling : public MaxPool3DGradTilingBase {
public:
    explicit MaxPool3DGradKsizeOneTiling(gert::TilingContext* context)
        : MaxPool3DGradTilingBase(context), common_(std::make_unique<Pool3DGradKsizeOneCommonTiling>(&inputData))
    {}

    ~MaxPool3DGradKsizeOneTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    std::unique_ptr<Pool3DGradKsizeOneCommonTiling> common_;
};

} // namespace optiling

#endif // MAX_POOL3D_GRAD_KSIZE_ONE_TILING_H_
