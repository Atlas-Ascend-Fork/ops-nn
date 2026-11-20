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
using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

namespace {
// Keep parameter mode values in sync with the transpose kernel.
constexpr uint32_t TRANSPOSE_PARAM_SAME = 0;
constexpr uint32_t TRANSPOSE_PARAM_BRC_P = 1;
constexpr uint32_t TRANSPOSE_PARAM_TILE_PARAMS = 2;
constexpr uint32_t TRANSPOSE_PARAM_FULL_PARAMS = 3;
constexpr uint64_t NUM_TWO = 2;
constexpr uint64_t BFP32_SIZE = sizeof(float);
constexpr uint64_t B16_SIZE = BFP32_SIZE / NUM_TWO;
constexpr uint64_t TRANSPOSE_R_LIMIT = 16;
constexpr uint64_t TRANSPOSE_MIN_VL_PER_CORE = 8;
constexpr uint64_t TRANSPOSE_ROWS = 16;
constexpr uint64_t TRANSPOSE_BLOCK_BYTES = 32;
constexpr uint64_t TRANSPOSE_MAX_REPEATS = 255;
constexpr uint64_t TRANSPOSE_KEY_BASE = 1000;
constexpr uint64_t BUFFER_COUNT = NUM_TWO;

} // namespace

bool LayerNormV3RegBaseTransposeTiling::IsCapable()
{
    return commonParams.isRegBase && commonParams.rowSize <= TRANSPOSE_R_LIMIT &&
           CeilDiv(commonParams.colSize, commonParams.coreNum) > TRANSPOSE_MIN_VL_PER_CORE * commonParams.vlFp32;
}

uint64_t LayerNormV3RegBaseTransposeTiling::GetTilingKey() const
{
    return TRANSPOSE_KEY_BASE + GetDTypeKey(commonParams.tensorDtype, commonParams.paramDtype);
}

ge::graphStatus LayerNormV3RegBaseTransposeTiling::DoOpTiling()
{
    const auto xType = commonParams.tensorDtype;
    const auto paramType = commonParams.paramDtype;
    const uint64_t xBytes = xType == ge::DT_FLOAT ? BFP32_SIZE : B16_SIZE;
    const uint64_t paramBytes = paramType == ge::DT_FLOAT ? BFP32_SIZE : B16_SIZE;
    const uint64_t r = commonParams.rowSize;
    const uint64_t blockBytes = commonParams.blockSize;
    // TransDataTo5HD consumes 16 rows and 16 B16 / 8 B32 columns per repeat.
    const uint64_t rAlignment = TRANSPOSE_BLOCK_BYTES / xBytes;
    const uint64_t aAlignment = commonParams.vlFp32;
    const uint64_t rAlign = CeilAlign(r, rAlignment);
    uint64_t paramAlign = CeilAlign(r * paramBytes, blockBytes) / paramBytes;
    // FP32 input/centered data, AR output double buffers, shared reduce/RA-output storage,
    // and two slots for each FP32 statistic. Keep this budget consistent with Init.
    const uint64_t tempBytesPerA = rAlign * BFP32_SIZE;
    const uint64_t bytesPerA = BUFFER_COUNT * (r * BFP32_SIZE + rAlign * xBytes) + tempBytesPerA +
                               NUM_TWO * BUFFER_COUNT * BFP32_SIZE;
    const uint64_t paramsSize = NUM_TWO * paramAlign * paramBytes;
    const uint64_t rowsPerCore = CeilDiv(commonParams.colSize, commonParams.coreNum);
    const uint64_t repeatLimit = TRANSPOSE_ROWS * TRANSPOSE_MAX_REPEATS / aAlignment * aAlignment;
    const uint64_t b = commonParams.paramsToNormSize;
    uint64_t paramStride = 0;
    uint64_t maxTileA = 0;
    auto paramMode = commonParams.normToParamsSize == 1 ? TRANSPOSE_PARAM_SAME : TRANSPOSE_PARAM_BRC_P;
    if (b == 1) {
        maxTileA = std::min((commonParams.ubSizePlatForm - paramsSize) / bytesPerA, repeatLimit);
    } else {
        paramMode = TRANSPOSE_PARAM_TILE_PARAMS;
        const uint64_t tileBytesPerA = bytesPerA + NUM_TWO * BUFFER_COUNT * r * paramBytes;
        maxTileA = std::min(commonParams.ubSizePlatForm / tileBytesPerA, repeatLimit);
        // Keep one compact AR period only when it fits alongside an entire B-row x tile.
        // Bound B before multiplying it, and include the per-core tile limit in admission.
        if (b <= repeatLimit && b <= CeilAlign(rowsPerCore, aAlignment)) {
            const uint64_t fullParamAlign = CeilAlign(b * r * paramBytes, blockBytes) / paramBytes;
            const uint64_t fullParamsSize = NUM_TWO * fullParamAlign * paramBytes;
            if (fullParamsSize < commonParams.ubSizePlatForm) {
                const uint64_t fullTileA = std::min((commonParams.ubSizePlatForm - fullParamsSize) / bytesPerA,
                                                    repeatLimit);
                if (b <= fullTileA / aAlignment * aAlignment) {
                    paramMode = TRANSPOSE_PARAM_FULL_PARAMS;
                    paramAlign = fullParamAlign;
                    maxTileA = fullTileA;
                }
            }
        }
    }
    const uint64_t tileA = std::min(maxTileA / aAlignment * aAlignment, CeilAlign(rowsPerCore, aAlignment));
    if (tileA == 0) {
        return ge::GRAPH_PARAM_INVALID;
    }
    if (paramMode == TRANSPOSE_PARAM_TILE_PARAMS) {
        paramStride = tileA;
        paramAlign = r * tileA;
    }
    td_.set_a(commonParams.colSize);
    td_.set_aBlockFactor(rowsPerCore);
    td_.set_tileA(tileA);
    td_.set_r(r);
    td_.set_rAlign(rAlign);
    td_.set_paramAlign(paramAlign);
    td_.set_usedCoreNum(CeilDiv(commonParams.colSize, rowsPerCore));
    td_.set_epsilon(commonParams.eps);
    td_.set_paramRows(b);
    td_.set_paramR(r / commonParams.normToParamsSize);
    td_.set_paramStride(paramStride);
    td_.set_paramMode(paramMode);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LayerNormV3RegBaseTransposeTiling::PostTiling()
{
    context_->SetBlockDim(td_.get_usedCoreNum());
    td_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

// Keep the R=1 no-reduce template ahead of this path.
REGISTER_OPS_TILING_TEMPLATE(LayerNormV3, LayerNormV3RegBaseTransposeTiling, 125);
REGISTER_OPS_TILING_TEMPLATE(LayerNorm, LayerNormV3RegBaseTransposeTiling, 125);
} // namespace optiling
