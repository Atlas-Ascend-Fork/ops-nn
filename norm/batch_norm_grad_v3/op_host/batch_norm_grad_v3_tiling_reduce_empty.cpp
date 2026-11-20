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
 * \file batch_norm_grad_v3_tiling_reduce_empty.cpp
 * \brief Tiling for training with an empty reduction and non-empty parameter gradients.
 */

#include <algorithm>
#include <string>
#include <vector>
#include "batch_norm_grad_v3_tiling.h"

namespace optiling {
namespace {
constexpr uint64_t REDUCE_EMPTY_KEY = 60000000;
constexpr uint64_t MIN_BYTES_PER_CORE = 32 * 1024;
constexpr uint64_t UB_BUDGET_DIVISOR = 2;

void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "BatchNormGradV3", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}
} // namespace

class BatchNormGradV3ReduceEmptyTiling : public BatchNormGradV3TilingBase {
public:
    explicit BatchNormGradV3ReduceEmptyTiling(gert::TilingContext* context)
        : BatchNormGradV3TilingBase(context, tilingData_.baseTilingData)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override { return REDUCE_EMPTY_KEY; }
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    BatchNormGradV3ReduceEmptyTilingData tilingData_;
};

bool BatchNormGradV3ReduceEmptyTiling::IsCapable() { return aDim > 0 && (r1Dim == 0 || r0Dim == 0); }

ge::graphStatus BatchNormGradV3ReduceEmptyTiling::DoOpTiling()
{
    const uint64_t elementBytes = weightDtype == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    OP_CHECK_IF(coreNum == 0 || blockSize < elementBytes || ubSize / UB_BUDGET_DIVISOR < blockSize,
                ReportTilingStageError(context_->GetNodeName(), "Invalid AIV core count or UB configuration"),
                return ge::GRAPH_FAILED);
    const uint64_t channels = static_cast<uint64_t>(aDim);
    const uint64_t elementsPerBlock = blockSize / elementBytes;
    const uint64_t desiredCores = std::min<uint64_t>(coreNum,
                                                     Ops::Base::CeilDiv(channels, MIN_BYTES_PER_CORE / elementBytes));
    const uint64_t channelsPerCore = Ops::Base::CeilAlign(Ops::Base::CeilDiv(channels, desiredCores), elementsPerBlock);
    const uint64_t usedCores = Ops::Base::CeilDiv(channels, channelsPerCore);
    const uint64_t maxTileChannels = ubSize / (UB_BUDGET_DIVISOR * elementBytes);
    const uint64_t tileChannels = std::min(channelsPerCore, Ops::Base::FloorAlign(maxTileChannels, elementsPerBlock));

    baseTilingData.set_aDim(aDim);
    baseTilingData.set_formerBlockDim(channelsPerCore);
    baseTilingData.set_blockNum(usedCores);
    tilingData_.set_tileChannels(tileChannels);
    tilingData_.set_outputMask(outputMask_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3ReduceEmptyTiling::GetWorkspaceSize()
{
    auto* workspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspace);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradV3ReduceEmptyTiling::PostTiling()
{
    auto* rawData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawData);
    OP_CHECK_IF(rawData->GetCapacity() < tilingData_.GetDataSize(),
                ReportTilingStageError(context_->GetNodeName(), "Empty reduction tiling buffer is too small"),
                return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(rawData->GetData(), rawData->GetCapacity());
    rawData->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(baseTilingData.get_blockNum());
    return ge::GRAPH_SUCCESS;
}

// Try the empty reduction after the legacy templates (1000–1100) and before RAR split-core (2000).
REGISTER_TILING_TEMPLATE("BatchNormGradV3", BatchNormGradV3ReduceEmptyTiling, 1500);
} // namespace optiling
