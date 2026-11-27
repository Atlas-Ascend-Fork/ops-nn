/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FOREACH_PERF_FLAT_TILING_H
#define FOREACH_PERF_FLAT_TILING_H
#include "foreach_flat_regbase_tiling.h"

namespace optiling::ForeachPerfFlat {
inline bool IsTarget(gert::TilingContext* context)
{
    auto info = context->GetPlatformInfo();
    return info != nullptr &&
           platform_ascendc::PlatformAscendC(info).GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
}

// Called only after the operator's original shape/dtype/scalar validation.
template <uint32_t FLOWS>
ge::graphStatus Build(gert::TilingContext* context, uint64_t key)
{
    auto node = context->GetComputeNodeInfo();
    auto instances = node == nullptr ? nullptr : node->GetInputInstanceInfo(0);
    if (instances == nullptr)
        return ge::GRAPH_FAILED;
    const int64_t count = instances->GetInstanceNum();
    if (count == 0) {
        auto td = context->GetTilingData<ForeachFlatTilingData>();
        auto ws = context->GetWorkspaceSizes(1);
        if (td == nullptr || ws == nullptr)
            return ge::GRAPH_FAILED;
        *td = ForeachFlatTilingData{};
        td->tileElems = 32;
        td->tensorEndOffsetList[0] = -1;
        ws[0] = 32;
        context->SetBlockDim(1);
        context->SetTilingKey(key);
        return ge::GRAPH_SUCCESS;
    }
    auto result = ForeachFlatRegbaseTiling::Build<FLOWS>(context);
    if (result != ge::GRAPH_SUCCESS)
        return result;
    auto td = context->GetTilingData<ForeachFlatTilingData>();
    int64_t maximum = 0;
    for (int64_t i = 0; i < count; ++i)
        maximum = std::max(maximum, td->tensorDataCountList[i]);
    // Tiny lists need parallelism across tensors rather than one serial DMA chain.
    // The element-balanced route remains in use when any tensor is large.
    if (maximum <= 32768) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const int64_t cores = std::min<int64_t>(
            count, std::min<int64_t>(platform.GetCoreNumAiv(), FOREACH_FLAT_MAX_CORE_COUNT));
        if (cores <= 0)
            return ge::GRAPH_FAILED;
        int64_t cursor = 0;
        for (int64_t core = 0; core < cores; ++core) {
            const int64_t share = count / cores + static_cast<int64_t>(core < count % cores);
            const int64_t end = cursor + share - 1;
            td->tensorStartList[core] = static_cast<uint16_t>(cursor);
            td->tensorEndList[core] = static_cast<uint16_t>(end);
            td->tensorStartOffsetList[core] = 0;
            td->tensorEndOffsetList[core] = td->tensorDataCountList[end] - 1;
            cursor = end + 1;
        }
        context->SetBlockDim(static_cast<uint32_t>(cores));
    }
    context->SetTilingKey(key);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling::ForeachPerfFlat
#endif
