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
 * \file anti_hi_f4_quant_tiling_arch35.h
 * \brief
 */

#ifndef OPS_NN_ANTI_HI_F4_QUANT_TILING_ARCH35_H
#define OPS_NN_ANTI_HI_F4_QUANT_TILING_ARCH35_H

#include <cstdint>
#include <vector>
#include <string>
#include <set>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/anti_hi_f4_quant_tilingdata.h"

namespace optiling {

struct AntiHiF4QuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct AntiHiF4QuantTilingParam {
    int64_t totalCoreNum{0};
    int64_t usedCoreNum{0};
    int64_t ubSize{0};
    uint32_t vfLen{0};
    uint32_t workspaceSize{0};
    int64_t tilingKey{0};
    int64_t dstType{27};
    uint64_t quantMode{0};
    int64_t xDimNeg1{0};
    int64_t xDimNeg2{0};

    // -1轴
    int64_t groupNum{0};
    int64_t ubFormer{0};
    int64_t blockFormer{0};
    int64_t blockTail{0};
    int64_t ubLoopOfFormerBlock{0};
    int64_t ubLoopOfTailBlock{0};
    int64_t ubTailOfFormerBlock{0};
    int64_t ubTailOfTailBlock{0};

    // -2轴
    int64_t splitBlockColSize{0};
    int64_t splitBlockRowSize{0};
    int64_t splitBlockColNum{0};
    int64_t splitBlockRowNum{0};
    int64_t tailBlockRowSize{0};
    int64_t tailBlockColSize{0};
    int64_t totalTaskNum{0};
    int64_t headCoreLoopNum{0};
    int64_t tailCoreLoopNum{0};
    int64_t headCoreNum{0};
    int64_t ubBlockNum{0};
    int64_t taskNumPerBatch{0};
    int64_t xDimNeg3{1};
};

class AntiHiF4QuantNotTailTiling {
public:
    explicit AntiHiF4QuantNotTailTiling(gert::TilingContext* context, const AntiHiF4QuantTilingParam& tilingParams)
        : context_(context), tilingParams_(tilingParams), tilingData_{}
    {}
    ~AntiHiF4QuantNotTailTiling() {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus SetTilingData();

    void PrintTilingData();
    void CalcMaxUbAvailable(const int64_t blockH, const int64_t blockW);
    void SplitCoreNotTail();

private:
    gert::TilingContext* context_ = nullptr;
    AntiHiF4QuantTilingParam tilingParams_;
    AntiHiF4QuantNotTailTilingData tilingData_{};
};

class AntiHiF4QuantTailTiling {
public:
    explicit AntiHiF4QuantTailTiling(gert::TilingContext* context, const AntiHiF4QuantTilingParam& tilingParams)
        : context_(context), tilingParams_(tilingParams), tilingData_{}
    {}
    ~AntiHiF4QuantTailTiling() {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus SetTilingData();

    void splitCoreTail();
    void PrintTilingData();

private:
    gert::TilingContext* context_ = nullptr;
    AntiHiF4QuantTilingParam tilingParams_;
    AntiHiF4QuantTailTilingData tilingData_{};
};

} // namespace optiling
#endif // OPS_NN_ANTI_HI_F4_QUANT_TILING_ARCH35_H
