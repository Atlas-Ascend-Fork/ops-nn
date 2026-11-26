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
 * \file gemmv3_basic_aswt_tiling.h
 * \brief GemmV3 A5/A6 Blaze(TENSOR_LEVEL) ASWT tiling.
 */
#pragma once

#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.h"

namespace optiling {
namespace matmul_v3_advanced {
// GemmV3 只注册了 ASWT 非全载 on-the-fly 的 TENSOR_LEVEL 单组合，
// 复用 MatMulV3BasicAswtTiling 的切分与 tiling data，但固定模板参数。
class GemmV3BasicAswtTiling : public MatMulV3BasicAswtTiling {
public:
    GemmV3BasicAswtTiling(gert::TilingContext* context, MatMulTilingCfg& cfg)
        : MatMulV3BasicAswtTiling(context, cfg) {};
    ~GemmV3BasicAswtTiling() override = default;

protected:
    // Blaze(TENSOR_LEVEL) 分支仅支持 ND 输入（kernel 侧 layoutB 无 NZ 分支），
    bool IsCapable() override { return args_.aFormat == ge::FORMAT_ND && args_.bFormat == ge::FORMAT_ND; }
    ge::graphStatus DoOpTiling() override;
};
} // namespace matmul_v3_advanced
} // namespace optiling
