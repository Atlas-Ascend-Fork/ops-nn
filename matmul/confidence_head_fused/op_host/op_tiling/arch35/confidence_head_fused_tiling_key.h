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

#include "../../../op_kernel/arch35/confidence_head_fused_kernel_tiling_key.h"

namespace optiling::confidence_head_fused {

class ConfidenceHeadFusedTilingKey {
public:
    ConfidenceHeadFusedTilingKey& SetHasMarkov(bool hasMarkov)
    {
        tilingKey_ = hasMarkov ? CONFIDENCE_HEAD_FUSED_WITH_MARKOV : CONFIDENCE_HEAD_FUSED_NO_MARKOV;
        return *this;
    }

    uint64_t GetTilingKey() const;

private:
    uint64_t tilingKey_{CONFIDENCE_HEAD_FUSED_NO_MARKOV};
};

} // namespace optiling::confidence_head_fused
