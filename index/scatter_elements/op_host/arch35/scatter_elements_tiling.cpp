/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "../../../scatter_elements_v2/op_host/arch35/scatter_elements_v2_asc_tiling.h"
#include "../../../scatter_elements_v2/op_host/arch35/scatter_elements_v2_base_tiling.h"

namespace optiling {
ge::graphStatus TilingArch35PrepareForScatterElementsV2(gert::TilingParseContext* context);

BEGIN_TILING_DATA_DEF(ScatterElementsCopyTilingData)
TILING_DATA_FIELD_DEF(uint64_t, dataBytes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ScatterElements_4000000, ScatterElementsCopyTilingData)

REGISTER_TILING_DATA_CLASS(ScatterElements, ScatterElementsV2AscTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElements_3000001, ScatterElementsV2SmallShapeTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElements_3000002, ScatterElementsV2SmallShapeTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElements_3000004, ScatterElementsV2SmallShapeTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElements_3000008, ScatterElementsV2SmallShapeTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElements_3001001, ScatterElementsV2SmallShapeTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElements_3001002, ScatterElementsV2SmallShapeTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElements_3001004, ScatterElementsV2SmallShapeTilingData)
REGISTER_TILING_DATA_CLASS(ScatterElements_3001008, ScatterElementsV2SmallShapeTilingData)

static ge::graphStatus TilingScatterElements(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE_WITHOUT_REPORT("ScatterElements", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    const auto* input = context->GetInputShape(0);
    const auto* indices = context->GetInputShape(1);
    const auto* updates = context->GetInputShape(2);
    const auto* inputDesc = context->GetInputDesc(0);
    const auto* compileInfo = static_cast<const ScatterElementsV2CompileInfoArch35*>(context->GetCompileInfo());
    OP_CHECK_IF(input == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "data"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(indices == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "indices"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(updates == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "updates"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "data"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Missing compile information"),
                return ge::GRAPH_FAILED);
    const auto& dataShape = input->GetStorageShape();
    const auto& indexShape = indices->GetStorageShape();
    const auto& updateShape = updates->GetStorageShape();
    const auto rank = dataShape.GetDimNum();
    // V1 accepts tensor updates. V2's scalar broadcast must not extend V1's input contract.
    if (rank != indexShape.GetDimNum() || rank != updateShape.GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "data, indices, updates",
            (std::to_string(rank) + ", " + std::to_string(indexShape.GetDimNum()) + ", " +
             std::to_string(updateShape.GetDimNum()))
                .c_str(),
            "The input ranks must match");
        return ge::GRAPH_FAILED;
    }
    for (size_t d = 0; d < rank; ++d) {
        if (indexShape.GetDim(d) > updateShape.GetDim(d)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "indices, updates",
                (Ops::Base::ToString(indexShape) + ", " + Ops::Base::ToString(updateShape)).c_str(),
                ("indices must not exceed updates at dimension " + std::to_string(d)).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    auto* raw = context->GetRawTilingData();
    OP_CHECK_IF(raw == nullptr, OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Missing raw tiling data"),
                return ge::GRAPH_FAILED);
    constexpr uint32_t COPY_BYTES_PER_CORE = 16 * 1024;
    const int64_t elementBytes = ge::GetSizeByDataType(inputDesc->GetDataType());
    if (elementBytes <= 0 || compileInfo->totalCoreNum <= 0) {
        OP_LOGE_WITHOUT_REPORT(
            context->GetNodeName(), "Invalid copy tiling inputs: dtype=%d, elementBytes=%ld, cores=%ld",
            static_cast<int>(inputDesc->GetDataType()), elementBytes, static_cast<int64_t>(compileInfo->totalCoreNum));
        return ge::GRAPH_FAILED;
    }
    const int64_t dataElements = dataShape.GetShapeSize();
    const int64_t copyCores = std::min(Ops::Base::CeilDiv(dataElements, COPY_BYTES_PER_CORE / elementBytes),
                                       static_cast<int64_t>(compileInfo->totalCoreNum));
    if (indexShape.GetShapeSize() == 0) {
        // Validate empty inputs with the common validators before taking the copy-only route.
        class EmptyScatterElementsTiling : public ScatterElementsV2AscTiling {
        public:
            using ScatterElementsV2AscTiling::GetShapeAttrsInfo;
            using ScatterElementsV2AscTiling::ScatterElementsV2AscTiling;
        };
        EmptyScatterElementsTiling validate(context);
        auto status = validate.GetShapeAttrsInfo();
        if (status != ge::GRAPH_SUCCESS) {
            return status;
        }
        auto* workspaces = context->GetWorkspaceSizes(1);
        OP_CHECK_IF(workspaces == nullptr,
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Missing workspace size output"),
                    return ge::GRAPH_FAILED);
        ScatterElementsCopyTilingData copyTiling;
        copyTiling.set_dataBytes(static_cast<uint64_t>(dataElements) * elementBytes);
        if (raw->GetCapacity() < copyTiling.GetDataSize()) {
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                   "Copy tiling buffer too small: requiredBytes=%zu, availableBytes=%zu",
                                   copyTiling.GetDataSize(), raw->GetCapacity());
            return ge::GRAPH_FAILED;
        }
        copyTiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
        raw->SetDataSize(copyTiling.GetDataSize());
        workspaces[0] = 0;
        context->SetTilingKey(4000000);
        context->SetBlockDim(std::max(copyCores, int64_t{1}));
        context->SetScheduleMode(1);
        return context->SetDynUBufSize(COPY_BYTES_PER_CORE);
    }

    ScatterElementsV2AscTiling tiling(context);
    auto status = tiling.DoTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    if (context->GetTilingKey() / 1000000 == 3) {
        // Reserve the system workspace required by the input copy barrier.
        auto* workspaces = context->GetWorkspaceSizes(1);
        OP_CHECK_IF(workspaces == nullptr,
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Missing workspace size output"),
                    return ge::GRAPH_FAILED);
        workspaces[0] = std::max<size_t>(workspaces[0], compileInfo->workspaceSize);
    }
    // V2's single-core SortInvoke expects one caller.
    const int64_t blockDim = std::max<int64_t>(context->GetBlockDim(), copyCores);
    context->SetBlockDim(tiling.IsSingleCoreSort() ? 1 : blockDim);
    return context->SetDynUBufSize(std::max(context->GetDynUBufSize(), COPY_BYTES_PER_CORE));
}

static ge::graphStatus TilingPrepareForScatterElements(gert::TilingParseContext* context)
{
    auto status = TilingArch35PrepareForScatterElementsV2(context);
    if (status == ge::GRAPH_SUCCESS) {
        auto* compileInfo = context->GetCompiledInfo<ScatterElementsV2CompileInfoArch35>();
        OP_CHECK_IF(compileInfo == nullptr,
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Missing compile information"),
                    return ge::GRAPH_FAILED);
        compileInfo->workspaceSize = platform_ascendc::PlatformAscendC(context->GetPlatformInfo())
                                         .GetLibApiWorkSpaceSize();
    }
    return status;
}

IMPL_OP_OPTILING(ScatterElements)
    .Tiling(TilingScatterElements)
    .TilingParse<ScatterElementsV2CompileInfoArch35>(TilingPrepareForScatterElements);
} // namespace optiling
