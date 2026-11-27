/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "../rotate_dual_level_quant_common.h"

namespace optiling {
struct RotateDualLevelQuantCompileInfo {
    bool initialized = false;
    bool isA5 = false;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint64_t l1Size = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
    uint64_t l0CSize = 0;
    uint64_t ubSize = 0;
};

class RotateDualLevelQuantTiling {
public:
    explicit RotateDualLevelQuantTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    bool ReadPlatform();
    bool ReadAndValidateInputs();
    void MakeLogicalTiling();
    bool MakeKernelTiling();
    ge::graphStatus SaveTiling();

    gert::TilingContext* context_;
    RotateDualLevelQuantCompileInfo compileInfo_;
    RotateDualLevelQuantNS::Attributes attrs_;
    RotateDualLevelQuantNS::ShapeInfo shapeInfo_;
    RotateDualLevelQuantNS::TargetType target_ = RotateDualLevelQuantNS::TargetType::UNSUPPORTED;
    RotateDualLevelQuantNS::RotateDualLevelQuantTilingData data_;
};
} // namespace optiling
