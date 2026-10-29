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
 * \file swiglu_hi_f4_quant_tiling.h
 * \brief Tiling data structure and parameters for SwiGLU + HiFloat4 quantization
 */

#ifndef SWIGLU_HI_F4_QUANT_TILING_H
#define SWIGLU_HI_F4_QUANT_TILING_H

#include <cstdint>
#include <vector>
#include <string>
#include <set>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "../../op_kernel/arch92/swiglu_hi_f4_quant_tiling_data.h"

namespace optiling {

struct SwigluHiF4QuantCompileInfo {
    int64_t totalCoreNum{0};
    int64_t ubSize{0};
};

struct SwigluHiF4QuantInputInfo {
    ge::DataType xDtype{ge::DT_UNDEFINED};
    int64_t dimNum{0};
    int64_t inputDimNeg3{1};
    int64_t inputDimNeg2{1};
    int64_t inputDimNeg1{1};
    int64_t groupIndexType{0}; // 0=不存在, 1=int32, 2=int64
    int64_t groupIndexNum{0};  // group_index 的第0维 shape，不存在时为 0
};

struct SwigluHiF4QuantOutputInfo {
    ge::DataType yDtype{ge::DT_UNDEFINED};
    ge::DataType scaleDtype{ge::DT_UNDEFINED};
    int64_t outputYDimNeg1{0};
};

struct SwigluHiF4QuantAttrParam {
    int64_t activateDim{-1};
    bool activateLeft{false};
    int64_t swigluMode{0};
    float clampLimit{7.0f};
    float gluAlpha{1.702f};
    float gluBias{1.0f};
    int64_t axis{-1};
    int64_t blockSize{64};
    int64_t scaleAlg{0}; // 0=mse_first(有偏,阈值4/2), 1=unbiased(无偏,阈值3.75/1.875)
};

struct SwigluHiF4QuantTilingResult {
    int64_t basicDimNeg1{512};
    int64_t basicDimNeg2{1};
    int64_t usedCoreNum{0};
    int64_t isFullLoad{0};
    int64_t maxBasicNumUbDimNeg1{0};
    int64_t maxBasicNumUbDimNeg2{0};
    int64_t ubLoopPerRow{0};
    int64_t ubTailPerRow{0};
    // Inter-core split
    int64_t frontCoreNum{0};
    int64_t frontCoreBasicNumDimNeg2{0};
    int64_t frontCoreLoopTimes{0};
    int64_t frontCoreLastLoopBasicNum{0};
    int64_t tailCoreBasicNumDimNeg2{0};
    int64_t tailCoreLoopTimes{0};
    int64_t tailCoreLastLoopBasicNum{0};
};

class SwigluHiF4QuantRegbaseTiling {
public:
    explicit SwigluHiF4QuantRegbaseTiling(gert::TilingContext* context) : context_(context) {}

    ge::graphStatus GetNpuInfo();
    ge::graphStatus ValidateInput();
    ge::graphStatus ParseAttrs();
    ge::graphStatus ValidateOutput();
    ge::graphStatus PreProcess();
    ge::graphStatus CalculateTiling();
    ge::graphStatus FillTilingData();
    ge::graphStatus SetParams();
    void PrintTilingData() const;

private:
    int64_t HiFp4GetubFactorNeg1();
    void SetTilingParamsNeg1();

private:
    gert::TilingContext* context_{nullptr};
    SwigluHiF4QuantTilingData* tilingData_{nullptr};

    SwigluHiF4QuantCompileInfo compileInfo_;
    SwigluHiF4QuantInputInfo inputInfo_;
    SwigluHiF4QuantOutputInfo outputInfo_;
    SwigluHiF4QuantAttrParam attrParam_;
    SwigluHiF4QuantTilingResult tilingResult_;
};

ge::graphStatus Tiling4SwigluHiF4Quant(gert::TilingContext* context);
ge::graphStatus TilingPrepare4SwigluHiF4Quant(gert::TilingParseContext* context);

} // namespace optiling
#endif // SWIGLU_HI_F4_QUANT_TILING_H
