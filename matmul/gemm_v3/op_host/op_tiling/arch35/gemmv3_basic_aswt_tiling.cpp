/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gemmv3_basic_aswt_tiling.cpp
 * \brief GemmV3 A5/A6 Blaze(TENSOR_LEVEL) ASWT tiling implementation.
 */

#include "gemmv3_basic_aswt_tiling.h"

#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"

namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;

// A5/A6: GemmV3 走 Blaze(TENSOR_LEVEL) ASWT；arch35 仅支持 ND，无需 HIGH_LEVEL 回退
MM_REGISTER_TILING_TEMPLATE(GemmV3, GemmV3BasicAswtTiling, DAV_3510, BASIC_ASWT);
MM_REGISTER_TILING_TEMPLATE(GemmV3, GemmV3BasicAswtTiling, DAV_9201, BASIC_ASWT);
MM_REGISTER_TILING_TEMPLATE(GemmV3, GemmV3BasicAswtTiling, DAV_9202, BASIC_ASWT);

ge::graphStatus GemmV3BasicAswtTiling::DoOpTiling()
{
    // 复用 ASWT 切分；GemmV3 只注册 TENSOR_LEVEL + 非全载 + on-the-fly + BASIC 组合，
    // 跳过基类全载/fixpipe/切K 分支，直接固定模板参数。
    ge::graphStatus ret = MatMulV3AswTiling::DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    if (compileInfo_.npuArch != NpuArch::DAV_9201 && compileInfo_.npuArch != NpuArch::DAV_9202) {
        // 非 fixpipe 优化场景：ATrans(A转置、B不转置)时按 L1 容量调整 stepK，避免 A/B 同时驻留 L1 超限
        if (args_.isATrans && !args_.isBTrans) {
            uint64_t remainSizeForAL1BL1 = args_.hasBias ? (compileInfo_.l1Size - BIAS_TABLE_NUM * DATA_SIZE_FP32) :
                                                           compileInfo_.l1Size;
            runInfo_.stepKa = remainSizeForAL1BL1 / NUM_TWO / ((runInfo_.baseM + runInfo_.baseN) * runInfo_.baseK) /
                              args_.aDtypeSize;
            runInfo_.stepKb = runInfo_.stepKa;
            runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
            runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE;
        }
    }
    runInfo_.splitKBase = args_.kValue;

    fullLoad_ = MatMulV3FullLoad::NONE_FULL_LOAD;
    l0C2Out_ = MatMulV3L0C2Out::ON_THE_FLY;
    model_ = MatMulV3Model::BASIC;
    apiLevel_ = MatMulV3ApiLevel::TENSOR_LEVEL;
    return ge::GRAPH_SUCCESS;
}
} // namespace matmul_v3_advanced
} // namespace optiling
