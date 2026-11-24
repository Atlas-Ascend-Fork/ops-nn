/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_GRAPH_FUSION_PASS_FIXPIPE_FUSION_REPORT_UTILS_H_
#define COMMON_GRAPH_FUSION_PASS_FIXPIPE_FUSION_REPORT_UTILS_H_

#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "version/ge-compiler_version.h"

#if GE_COMPILER_VERSION_NUM >= 90100000U
namespace ge {
namespace fusion {
// Weak mirror of ge/fusion/graph_fuse_inspector_utils.h (the fusion_result.json report api),
// the single maintenance point of the fixpipe common passes: the header name is prefixed with
// the fixpipe domain so it never duplicates the conv domain mirror, the plugin still loads
// when the runtime libge does not export these symbols. Keep signatures in sync with the GE
// header and check the address against nullptr before calling.
class GraphFuseInspectorUtils {
public:
    static Status ReportFuse(const std::vector<GNode>& nodesBeforeFuse, const std::vector<GNode>& nodesAfterFuse,
                             CustomPassContext& ctx) __attribute__((weak));
#if GE_COMPILER_VERSION_NUM >= 90200000U
    static Status ReportMatch(const std::vector<GNode>& matchedNodes, CustomPassContext& ctx) __attribute__((weak));
#endif
};
} // namespace fusion
} // namespace ge
#endif

namespace ops {
namespace fusion {
// report one structure match of a fusion pass, counted regardless of the following fusion
// conditions, the match_times statistic lands in fusion_result.json (non fatal on failure)
inline ge::Status ReportMatchNodes(const std::vector<ge::GNode>& matchedNodes, ge::CustomPassContext& ctx)
{
#if GE_COMPILER_VERSION_NUM >= 90200000U
    if (ge::fusion::GraphFuseInspectorUtils::ReportMatch != nullptr) {
        return ge::fusion::GraphFuseInspectorUtils::ReportMatch(matchedNodes, ctx);
    }
#else
    (void)matchedNodes;
    (void)ctx;
#endif
    return ge::SUCCESS;
}

// report one fusion effect after the graph surgery and while the absorbed nodes are still
// referenced, the effect_times statistic lands in fusion_result.json (non fatal on failure)
inline ge::Status ReportFuseNodes(const std::vector<ge::GNode>& nodesBeforeFuse,
                                  const std::vector<ge::GNode>& nodesAfterFuse, ge::CustomPassContext& ctx)
{
#if GE_COMPILER_VERSION_NUM >= 90100000U
    if (ge::fusion::GraphFuseInspectorUtils::ReportFuse != nullptr) {
        return ge::fusion::GraphFuseInspectorUtils::ReportFuse(nodesBeforeFuse, nodesAfterFuse, ctx);
    }
#else
    (void)nodesBeforeFuse;
    (void)nodesAfterFuse;
    (void)ctx;
#endif
    return ge::SUCCESS;
}
} // namespace fusion
} // namespace ops
#endif // COMMON_GRAPH_FUSION_PASS_FIXPIPE_FUSION_REPORT_UTILS_H_
