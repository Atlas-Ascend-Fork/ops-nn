/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "foreach_mul_list_regbase.h"
#include "foreach_mul_list_tiling_key.h"
#include "../../foreach_utils/arch35/foreach_flat_tiling_data.h"
using namespace AscendC;
template <uint32_t schMode>
__global__ __aicore__ void foreach_mul_list(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachFlatTilingData);
    GET_TILING_DATA_WITH_STRUCT(ForeachFlatTilingData, td, tiling);
    (void)workspace;
    TPipe pipe;
    ForeachMulList::ForeachMulListRegbase<DTYPE_X1, int32_t, ForeachFlatTilingData> op;
    op.Init(&td, &pipe);
    op.Process(x1, x2, y);
    pipe.Destroy();
}
