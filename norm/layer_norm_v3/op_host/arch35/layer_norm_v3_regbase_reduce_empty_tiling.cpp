/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "layer_norm_v3_tiling.h"
#include "op_common/op_host/util/math_util.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
using Ops::Base::CeilDiv;

namespace {
constexpr uint64_t BFP32_SIZE = sizeof(float);
constexpr uint64_t REDUCE_EMPTY_KEY = 1100;
constexpr uint64_t MIN_BYTES_PER_CORE = 32 * 1024;
constexpr uint64_t NUM_TWO = 2;
} // namespace

bool LayerNormV3RegBaseReduceEmptyTiling::IsCapable() { return commonParams.isRegBase && commonParams.rowSize == 0; }

uint64_t LayerNormV3RegBaseReduceEmptyTiling::GetTilingKey() const { return REDUCE_EMPTY_KEY; }

ge::graphStatus LayerNormV3RegBaseReduceEmptyTiling::DoOpTiling()
{
    const uint64_t outputBytes = commonParams.paramDtype == ge::DT_FLOAT ? BFP32_SIZE : BFP32_SIZE / NUM_TWO;
    const uint64_t blockBytes = commonParams.blockSize;
    const uint64_t elementsPerBlock = blockBytes / outputBytes;
    const uint64_t rows = commonParams.colSize;

    const uint64_t desiredCores = std::min(commonParams.coreNum, CeilDiv(rows * outputBytes, MIN_BYTES_PER_CORE));
    const uint64_t rowsPerCore = CeilDiv(CeilDiv(rows, desiredCores), elementsPerBlock) * elementsPerBlock;
    const uint64_t usedCores = CeilDiv(rows, rowsPerCore);
    const uint64_t maxTileRows = commonParams.ubSizePlatForm / (NUM_TWO * outputBytes);
    const uint64_t rowsPerTile = std::min(rowsPerCore, maxTileRows / elementsPerBlock * elementsPerBlock);
    td_.set_rows(rows);
    td_.set_rowsPerCore(rowsPerCore);
    td_.set_rowsPerTile(static_cast<uint32_t>(rowsPerTile));
    td_.set_usedCoreNum(static_cast<uint32_t>(usedCores));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3RegBaseReduceEmptyTiling::PostTiling()
{
    auto* rawTilingData = context_->GetRawTilingData();
    context_->SetBlockDim(td_.get_usedCoreNum());
    td_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(td_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

// Dispatch empty reductions before normal templates.
REGISTER_OPS_TILING_TEMPLATE(LayerNormV3, LayerNormV3RegBaseReduceEmptyTiling, 25);
REGISTER_OPS_TILING_TEMPLATE(LayerNorm, LayerNormV3RegBaseReduceEmptyTiling, 25);
} // namespace optiling
