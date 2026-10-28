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

#include "op_host/tiling_base.h"
#include "confidence_head_fused_compile_info.h"
#include "../../../op_kernel/arch35/confidence_head_fused_tiling_data.h"

namespace optiling::confidence_head_fused {

class ConfidenceHeadFusedTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit ConfidenceHeadFusedTiling(gert::TilingContext* context) : TilingBaseClass(context) {}
    ~ConfidenceHeadFusedTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus CheckArgs();
    ge::graphStatus CheckDtypeValid();
    ge::graphStatus CheckShapeValid();
    ge::graphStatus CheckFormat();
    ge::graphStatus GetArgs();
    ConfidenceHeadFusedCompileInfo platformInfo_{};
    CHF::ConfidenceHeadFusedTilingData tilingData_{};
};

} // namespace optiling::confidence_head_fused
