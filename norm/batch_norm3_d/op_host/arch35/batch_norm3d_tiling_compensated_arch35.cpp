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
#include <limits>
#include "batch_norm3d_tiling.h"

namespace optiling {
class BatchNorm3DCompensatedTiling : public BatchNorm3DRegbaseTilingBase {
public:
    explicit BatchNorm3DCompensatedTiling(gert::TilingContext* context) : BatchNorm3DRegbaseTilingBase(context) {}

protected:
    bool IsCapable() override
    {
        // FP16 RA still exposes FP32 statistics; near-zero means and affine
        // cancellation require the same compensated reduction as FP32 input.
        return xDtype_ == ge::DT_FLOAT || (xDtype_ == ge::DT_FLOAT16 && r0_ == 1);
    }

    ge::graphStatus DoOpTiling() override
    {
        data_.r1 = r1_;
        data_.r0 = r0_;
        data_.a = a_;
        const int64_t channelTile = r0_ == 1 && a_ > 1 ? 64 : 1;
        const int64_t channelGroups = (a_ + channelTile - 1) / channelTile;
        const int64_t reductionTiles = (r1_ * r0_ + 4095) / 4096;
        data_.reduceBlocks = std::min<int64_t>(std::max<int64_t>(aicoreParams_.blockDim / channelGroups, 1),
                                               reductionTiles);
        usedCoreNums_ = std::min<int64_t>(channelGroups, aicoreParams_.blockDim) * data_.reduceBlocks;
        data_.blockNum = usedCoreNums_;
        const double count = static_cast<double>(r1_) * static_cast<double>(r0_);
        const double inverse = 1.0 / count;
        data_.invCount = static_cast<float>(inverse);
        data_.invCountLow = static_cast<float>(inverse - static_cast<double>(data_.invCount));
        data_.varianceCorrection = count > 1.0 ? static_cast<float>(count / (count - 1.0)) :
                                                 std::numeric_limits<float>::quiet_NaN();
        data_.epsilon = epsilon_;
        data_.momentum = exponentialAvgFactor_;
        data_.useRunningMeanVar = useRunningMeanVar_;
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override { return 700000; }

    ge::graphStatus PostTiling() override
    {
        context_->SetBlockDim(usedCoreNums_);
        if (data_.reduceBlocks > 1) {
            OP_CHECK_IF(context_->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context_->GetNodeName(), "Failed to set batch schedule mode"), return ge::GRAPH_FAILED);
        }
        auto* workspace = context_->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, workspace);
        workspace[0] = workspaceSize_ + usedCoreNums_ * 2 * 64 * sizeof(float);
        auto* output = context_->GetTilingData<BatchNorm3DCompensatedTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, output);
        *output = data_;
        return ge::GRAPH_SUCCESS;
    }

private:
    BatchNorm3DCompensatedTilingData data_;
};
REGISTER_OPS_TILING_TEMPLATE(BatchNorm3D, BatchNorm3DCompensatedTiling, 9000);
} // namespace optiling
