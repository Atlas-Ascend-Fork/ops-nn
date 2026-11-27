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
 * \file foreach_sign_regbase_tiling.h
 * \brief Sign-owned RegBase tiling: keeps the legacy arch35 validation contract
 *        (same checks, error macros and accepted inputs) on top of the shared
 *        ForeachRegbaseTiling split and transport.
 */

#ifndef FOREACH_SIGN_REGBASE_TILING_H
#define FOREACH_SIGN_REGBASE_TILING_H

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "graph/utils/type_utils.h"
#include "../../../foreach_utils/op_host/foreach_regbase_tiling.h"

namespace optiling {

class ForeachSignRegbaseTiling : public ForeachRegbaseTilingUnary {
public:
    explicit ForeachSignRegbaseTiling(gert::TilingContext* context) : ForeachRegbaseTilingUnary(context) {}
    ~ForeachSignRegbaseTiling() override = default;

    void Reset(gert::TilingContext* context) override { ForeachRegbaseTiling::Reset(context); }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
};

} // namespace optiling

#endif // FOREACH_SIGN_REGBASE_TILING_H
