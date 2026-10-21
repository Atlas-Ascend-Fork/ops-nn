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
 * \file quant_batch_matmul_v4_pergroup_arch92_tiling.h
 * \brief
 */
#pragma once

#include "quant_batch_matmul_v4_tiling.h"
#include "quant_batch_matmul_v4_polling_tiling.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v4_tiling_data_apt.h"

namespace optiling {
using namespace matmul_v4;

struct QuantBatchMatmulPerGroupInfo {
    bool transA;
    bool transB;
    bool weightNz;
    bool hasAntiQuantOffset = false;
    uint32_t libApiWorkSpaceSize;
    uint64_t groupSize;
    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t x1ScaleShapeInner;
    uint64_t x1ScaleShapeOuter;
    uint64_t x2ScaleShapeInner;
    uint64_t x2ScaleShapeOuter;

    uint32_t useCoreNum;
    ge::DataType aDtype;
    ge::DataType bDtype;
    ge::DataType cDtype;
    ge::DataType x1ScaleDtype;
    ge::DataType x2ScaleDtype;
    ge::DataType yOffsetDtype;
    QuantType antiQuantType = QuantType::PER_GROUP;

    ge::Format bFormat = ge::FORMAT_ND;
    const char* opName;
};

constexpr uint64_t BASE_N_ALIGN_SIZE = 64UL;
constexpr uint64_t MAX_BASE_M_BLOCK_SIZE = 128UL;
constexpr uint64_t MAX_BASE_N_BLOCK_SIZE = 512UL;
constexpr uint64_t MAX_BASE_MN_SIZE = 32768UL;

class QuantBatchMatmulV4PerGroupTiling : public TilingBaseClass {
public:
    explicit QuantBatchMatmulV4PerGroupTiling(gert::TilingContext* context) : TilingBaseClass(context) {}
    ~QuantBatchMatmulV4PerGroupTiling() override = default;

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }

    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus InstantiateTilingData();

    QuantBatchMatmulPerGroupInfo inputParams_;
    QuantBatchMatmulV4CompileInfo compileInfo_;
    std::unique_ptr<qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams> tilingDataManager_;
    qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams* tilingData_ = nullptr;
    size_t tilingDataSize_ = sizeof(qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams);

    QuantBatchMatmulV4PollingTiling pollingTiling_;
    bool isVectorAntiquant_ = false;
    bool isS8S4_ = false;
    bool isFp8Fp4_ = false;

private:
    bool AnalyzeScaleInputs();
    bool AnalyzeYScaleInput();
    bool AnalyzeAttrs();
    bool AnalyzeInputs();
    uint64_t GetFp8Fp4BaseK() const;
    bool IsGroupSizeSupported(uint64_t groupSize, const std::vector<uint64_t>& groupSizeSupportList) const;
    ge::graphStatus AnalyzeContext();
};
} // namespace optiling
