/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "foreach_sub_scalar_regbase.h"
#include "foreach_sub_scalar_tiling_key.h"
#include "../../foreach_utils/arch35/foreach_flat_tiling_data.h"
using namespace AscendC;
template <uint32_t schMode>
__global__ __aicore__ void foreach_sub_scalar(GM_ADDR x, GM_ADDR scalar, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachFlatTilingData);
    GET_TILING_DATA_WITH_STRUCT(ForeachFlatTilingData, td, tiling);
    (void)workspace;
    TPipe pipe;
    ForeachSubScalar::FlatKernel<DTYPE_X, DTYPE_SCALAR, ForeachFlatTilingData> op;
    op.Init(&td, &pipe);
    op.InitScalar(scalar);
    op.Process(x, y);
    pipe.Destroy();
}
