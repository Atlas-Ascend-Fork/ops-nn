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
 * \file max_pool_v3_grad_tiling.cpp
 * \brief Tiling entry for MaxPoolV3Grad.
 */

#include "exe_graph/runtime/tiling_parse_context.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "max_pool_v3_grad_tiling_base.h"

using Ops::NN::Optiling::TilingRegistry;

namespace optiling {

static ge::graphStatus Tiling4MaxPoolV3Grad(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4MaxPoolV3Grad(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaxPoolV3Grad)
    .Tiling(Tiling4MaxPoolV3Grad)
    .TilingParse<MaxPoolV3GradCompileInfo>(TilingPrepare4MaxPoolV3Grad);

} // namespace optiling
