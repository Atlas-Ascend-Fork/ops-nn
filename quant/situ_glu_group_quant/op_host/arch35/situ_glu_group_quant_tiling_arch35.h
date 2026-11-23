/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file situ_glu_group_quant_tiling_arch35.h
 * \brief Tiling for SituGluGroupQuant, ported from cann-recipes-infer
 *        TilingForSituGluGroupQuant (whole-row single chunk, row-level core
 *        dispatch, sparse validRows resolved in-kernel), ops-nn class style.
 */

#ifndef QUANT_SITU_GLU_GROUP_QUANT_TILING_ARCH35_H
#define QUANT_SITU_GLU_GROUP_QUANT_TILING_ARCH35_H

#include <cstdint>
#include <string>
#include <set>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/situ_glu_group_quant_tiling_data.h"

namespace optiling {

// ==================== CompileInfo ====================
// Cached platform info, so tiling still works when the SuperKernel JIT /
// tiling-sink path calls TilingFor* with a null runtime PlatformInfo.
// Field set mirrors recipes SituGluGroupQuantCompileInfo (incl. socVersion).
struct SituGluGroupQuantCompileInfo {
    int64_t totalCoreNum{0};
    int64_t ubSize{0};
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND950;
};

// ==================== Input Info ====================
struct SituGluGroupQuantInputInfo {
    int64_t dimNum{0};
    int64_t totalRows{1}; // product of leading dims (may include padding tail)
    int64_t lastDim{0};   // 2*EPR
};

// ==================== Output Info ====================
// Output dtypes are fixed by the def registration (y = e4m3fn, y_scale = e8m0);
// ValidateOutput is an early-error defense in depth (same rejection set as the
// def combination / simplifiedKey matching, one layer earlier, owner decision).
struct SituGluGroupQuantOutputInfo {
    ge::DataType yDtype{ge::DT_UNDEFINED};
    ge::DataType y_scaleDtype{ge::DT_UNDEFINED};
};

// ==================== Attr Params ====================
struct SituGluGroupQuantAttrParam {
    float beta{1.0f};
    // <= 0 disables the up-branch tanh soft-clamp (up passthrough, TPL
    // dimension); > 0 enables it. Default 0.0 (disabled).
    float linearBeta{0.0f};
    bool activateLeft{false}; // false -> gate second half (situ normal), true -> gate first half (recipes)
    // Extension attrs (situ_mx_quant caliber): validated-only for now — each
    // supports exactly one value, none reaches the kernel.
    int64_t axis{-1};
    int64_t quantMode{0};
    int64_t dstType{36};
    float dstTypeMax{0.0f};
    std::string roundMode{"rint"};
    // group_list_type: 0=cumsum (reserved), 1=count (only supported). Default 0,
    // so a missing attr is rejected together with an explicit 0.
    int64_t groupListType{0};
};

// ==================== Tiling Class ====================
class SituGluGroupQuantRegbaseTiling {
public:
    explicit SituGluGroupQuantRegbaseTiling(gert::TilingContext* context) : context_(context) {};

    ge::graphStatus GetNpuInfo();
    ge::graphStatus ParseAttrs();
    ge::graphStatus ValidateInput();
    ge::graphStatus ValidateOutput();
    ge::graphStatus CalculateTiling();
    ge::graphStatus FillTilingData();
    void SetTilingKeyAndCore();
    void PrintTilingData() const;

private:
    SituGluGroupQuantCompileInfo compileInfo_;
    SituGluGroupQuantInputInfo inputInfo_;
    SituGluGroupQuantOutputInfo outputInfo_;
    SituGluGroupQuantAttrParam attrParam_;

    // tiling results (recipes caliber)
    int64_t epr_ = 0;             // EPR = lastDim / 2
    int64_t totalElements_ = 0;   // totalRows * EPR
    int64_t chunkElements_ = 0;   // min(EPR, MAX_TILE_COLS) = whole-row chunk
    int64_t groupIndexCount_ = 0; // group_index length (values stay on device)
    int64_t fullRowsMode_ = 0;    // 1 = optional group_index absent -> all rows valid
    int64_t hasWeight_ = 0;       // 1 = optional weight input instantiated
    int64_t scaleColNum_ = 0;     // e8m0 scales per row (even aligned)
    int64_t rowsPerBatch_ = 1;    // K-row batching (perf dim; 1 = legacy per-row cycle)
    int64_t blockDim_ = 1;        // min(coreNum, totalRows)

    gert::TilingContext* context_ = nullptr;
    SituGluGroupQuantTilingData* tilingData_ = nullptr;
};

// ==================== Main function declarations ====================
ge::graphStatus Tiling4SituGluGroupQuant(gert::TilingContext* context);
ge::graphStatus TilingPrepare4SituGluGroupQuant(gert::TilingParseContext* context);

} // namespace optiling
#endif // QUANT_SITU_GLU_GROUP_QUANT_TILING_ARCH35_H
