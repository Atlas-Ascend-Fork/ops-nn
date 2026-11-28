/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_batch_mat_mul_tiling.cpp
 * \brief
 */
#include "transpose_batch_mat_mul_tiling.h"

#include <type_traits>

#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_platform_common.h"
#include "transpose_batch_mat_mul_simplifiedkey.h"
#include "./arch35/transpose_batch_mat_mul_tiling_advanced.h"
#include "transpose_batch_mat_mul_base_tiling.h"
#include "transpose_batch_mat_mul_einsum_tiling.h"
#include "platform/platform_infos_def.h"
#include "op_cache_tiling.h"

#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"

using namespace optiling::transpose_batch_mat_mul;
using namespace optiling::matmul_v3;
using Ops::NN::TilingPrepareForOpCache;
using Ops::NN::Optiling::TilingRegistry;

namespace optiling {

REGISTER_TILING_TEMPLATE("TransposeBatchMatMul", TransposeBatchMatMulBaseTiling, 0);

static bool IsPlatformHardwareInfoValid(gert::TilingContext* context)
{
    if (context == nullptr) {
        return false;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return false;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0UL;
    uint64_t l1Size = 0UL;
    uint64_t l0aSize = 0UL;
    uint64_t l0bSize = 0UL;
    uint64_t l0cSize = 0UL;
    uint64_t l2Size = 0UL;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
    // HOST-3: platform parameters must be range-checked before use (unsigned types compare with == 0).
    // The AIV core count is deliberately not required here: TransposeBatchMatMul only uses the vector
    // cores in the MIX/ON_THE_FLY scenarios, and the arch35 tiling validates aivNum itself when it
    // actually consumes it. Requiring a non-zero aivNum at this entry rejected cube-only platforms
    // that used to tile fine.
    return (platform.GetCoreNumAic() != 0U) && (ubSize != 0UL) && (l1Size != 0UL) && (l0aSize != 0UL) &&
           (l0bSize != 0UL) && (l0cSize != 0UL) && (l2Size != 0UL);
}

static ge::graphStatus TransposeBatchMatMulTilingFunc(gert::TilingContext* context)
{
    // The context has to be validated before anything dereferences it: the platform check below reports
    // through the operator name, so a null context would crash instead of returning GRAPH_FAILED.
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("TransposeBatchMatMul", "context is null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        !IsPlatformHardwareInfoValid(context),
        CUBE_INNER_ERR_REPORT("TransposeBatchMatMul", "invalid platform info: core number or buffer size is zero"),
        return ge::GRAPH_FAILED);
    if (IsAdvancedSocVersion(context)) {
        return transpose_batch_mat_mul_advanced::TransposeBatchMatMulTiling(context).DoTiling();
    }
    if (IsPpMatmulEinsumMode(context) == ge::GRAPH_SUCCESS) {
        return TransposeBatchMatMulEinsumTiling(context).DoTiling();
    }
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForTransposeBatchMatMul(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("TransposeBatchMatMul", "context is null"),
                    return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
                    return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<MatmulV3CompileInfo>();
    OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
    std::string val;
    std::string dataMoveL12Bt;
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", dataMoveL12Bt);
    compileInfoPtr->supportL0c2out = !val.empty();
    compileInfoPtr->supportL12BtBf16 = (dataMoveL12Bt.find("bf16") != std::string::npos);
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    // HOST-4: runtime platform parameters (core numbers and on-chip buffer sizes) must be obtained in
    // the Tiling phase, not cached here. Both tiling paths refresh them at Tiling time:
    // MatmulV3BaseTiling::InitCompileInfo() during construction for the base path, and the arch35
    // tiling classes for the advanced path. Only compile-time information is kept in this structure.
    compileInfoPtr->btSize = compileInfoPtr->supportL0c2out ? 1024UL : 0UL;                      // 1024 is btSize
    compileInfoPtr->btSize = compileInfoPtr->supportL12BtBf16 ? 4096UL : compileInfoPtr->btSize; // 4096 is btSize

    TilingPrepareForOpCache(context);
    OP_LOGI(context->GetNodeName(),
            "compile info success soc:%d, npu arch:%u, l1Size:%lu, l2Size:%lu, coreNum:%lu, supportL0c2out:%d, "
            "supportL12BtBf16:%d",
            static_cast<int>(compileInfoPtr->socVersion), compileInfoPtr->npuArch, compileInfoPtr->l1Size,
            compileInfoPtr->l2Size, compileInfoPtr->aicNum, compileInfoPtr->supportL0c2out,
            compileInfoPtr->supportL12BtBf16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TransposeBatchMatMul)
    .Tiling(TransposeBatchMatMulTilingFunc)
    .TilingParse<MatmulV3CompileInfo>(TilingPrepareForTransposeBatchMatMul)
    .GenSimplifiedKey(transpose_batch_mat_mul::GenSimplifiedKey);
} // namespace optiling
