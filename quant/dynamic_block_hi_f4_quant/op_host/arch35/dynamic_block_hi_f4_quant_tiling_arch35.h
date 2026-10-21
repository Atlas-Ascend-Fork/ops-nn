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
 * \file dynamic_block_hi_f4_quant_tiling_arch35.h
 * \brief Tiling header for DynamicBlockHiF4Quant on ascend960dt (arch35)
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_BLOCK_HI_F4_QUANT_REGBASE_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_BLOCK_HI_F4_QUANT_REGBASE_TILING_H
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/dynamic_block_hi_f4_quant_tilingdata.h"

namespace optiling {
using namespace Ops::NN::Optiling;

struct DynamicBlockHiF4QuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct DynamicBlockHiF4QuantTilingParam {
    int64_t tilingKey{0};
    int64_t totalCoreNum{0};
    int64_t usedCoreNum{0};
    int64_t ubSize{0};
    int64_t roundMode{0};
    int64_t scaleAlg{0};           // 缩放算法选择：0-默认(阈值为4/2)，1-阈值3.75/1.875
    int64_t blockSizeRow{0};       // 基本块的高，固定 64
    int64_t blockSizeCol{0};       // 基本块的宽，固定 512
    int64_t rowNum{0};             // 合轴后的行数 (-1 轴之前所有轴乘积，含 -2 轴)
    int64_t colNum{0};             // -1 轴大小
    int64_t rowBlockLoopNum{0};    // 行方向 64*64 基本块数量 = ceil(rowNum/64)
    int64_t colBlockLoopNum{0};    // 列方向 512 粒度块数量 = ceil(colNum/512)
    int64_t colTileNum{0};         // 列方向尾块长度
    int64_t normalCoreNum{0};      // 头核数量
    int64_t normalCoreBlockNum{0}; // 头核处理基本块的数量
    int64_t tailCoreBlockNum{0};   // 尾核处理基本块的数量
    int64_t workspaceSize{0};
};

enum class RoundModeList {
    MODE_ROUND = 0,
    MODE_FLOOR = 1,
    MODE_CEIL = 2,
    MODE_TRUNC = 3,
    MODE_RINT = 4,
    MODE_HYBRID = 5,
    MODE_UNDEFINED = -1,
};
// 本算子 round_mode 仅支持 "rint" 与 "trunc" 两种模式

class DynamicBlockHiF4QuantTiling {
public:
    explicit DynamicBlockHiF4QuantTiling(gert::TilingContext* context) : context_(context) {}
    ~DynamicBlockHiF4QuantTiling() {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetAttr();
    ge::graphStatus CheckDtype() const;
    ge::graphStatus CheckShape() const;
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus SetTilingParams();
    ge::graphStatus MergeAxis();

    void SetTilingKey();
    void SetTilingData();
    void PrintTilingData();
    void SplitCore();

    RoundModeList GetRoundMode(const std::string& roundMode);

private:
    uint64_t roundMode_ = 0;
    gert::TilingContext* context_ = nullptr;
    DynamicBlockHiF4QuantTilingData tilingData{};
    DynamicBlockHiF4QuantTilingParam tilingParams{};
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_BLOCK_HI_F4_QUANT_REGBASE_TILING_H
