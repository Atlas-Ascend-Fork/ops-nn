/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rotate_dual_level_quant_tiling.h"
#include <algorithm>
#include <limits>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "securec.h"

namespace optiling {
namespace {
namespace RDLQ = ::RotateDualLevelQuantNS;

uint64_t CeilDivPositive(uint64_t value, uint64_t divisor) { return value / divisor + (value % divisor != 0); }

uint64_t FixpipeReceiveBytes(uint64_t rows, uint64_t pitch)
{
    const uint64_t bytes = rows * pitch * sizeof(uint16_t) + RDLQ::FIXPIPE_READ_GUARD_BYTES;
    return CeilDivPositive(bytes, RDLQ::UB_ALIGNMENT_BYTES) * RDLQ::UB_ALIGNMENT_BYTES;
}

uint64_t QuantWorkspaceBytes(uint64_t rows)
{
    const uint64_t bytes = RDLQ::MX_WORKSPACE_BYTES + rows * RDLQ::LEVEL0_BLOCK_SIZE * sizeof(uint16_t) +
                           RDLQ::COLLECT_READ_GUARD_BYTES + rows * RDLQ::LEVEL0_SCALE_ROW_BYTES;
    return CeilDivPositive(bytes, RDLQ::UB_ALIGNMENT_BYTES) * RDLQ::UB_ALIGNMENT_BYTES;
}

void FillPlatform(const platform_ascendc::PlatformAscendC& platform, RotateDualLevelQuantCompileInfo& info)
{
    info.isA5 = platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    info.aicNum = platform.GetCoreNumAic();
    info.aivNum = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, info.l1Size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, info.l0ASize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, info.l0BSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, info.l0CSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);
    info.initialized = true;
}

std::vector<int64_t> Dimensions(const gert::Shape& shape)
{
    std::vector<int64_t> dims;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        dims.push_back(shape.GetDim(i));
    }
    return dims;
}

RDLQ::TargetType Target(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT4_E2M1:
            return RDLQ::TargetType::FP4_E2M1;
        case ge::DT_FLOAT8_E4M3FN:
            return RDLQ::TargetType::FP8_E4M3FN;
        case ge::DT_FLOAT8_E5M2:
            return RDLQ::TargetType::FP8_E5M2;
        default:
            return RDLQ::TargetType::UNSUPPORTED;
    }
}
} // namespace

bool RotateDualLevelQuantTiling::ReadPlatform()
{
    if (context_->GetPlatformInfo() != nullptr) {
        const platform_ascendc::PlatformAscendC platform(context_->GetPlatformInfo());
        FillPlatform(platform, compileInfo_);
    } else if (context_->GetCompileInfo() != nullptr) {
        compileInfo_ = *static_cast<const RotateDualLevelQuantCompileInfo*>(context_->GetCompileInfo());
    }
    if (!compileInfo_.initialized || !compileInfo_.isA5 || compileInfo_.aicNum == 0 ||
        compileInfo_.aivNum != RDLQ::AIV_NUM_RATIO * compileInfo_.aicNum || compileInfo_.l1Size == 0 ||
        compileInfo_.l0ASize == 0 || compileInfo_.l0BSize == 0 || compileInfo_.l0CSize == 0 ||
        compileInfo_.ubSize == 0) {
        OP_LOGE(context_->GetNodeName(), "Ascend950 with valid memory sizes and AIC:AIV=1:2 is required.");
        return false;
    }
    return true;
}

bool RotateDualLevelQuantTiling::ReadAndValidateInputs()
{
    const auto* attrs = context_->GetAttrs();
    if (attrs == nullptr || attrs->GetAttrNum() < 9 || context_->GetInputShape(0) == nullptr ||
        context_->GetInputShape(1) == nullptr || context_->GetInputDesc(0) == nullptr ||
        context_->GetInputDesc(1) == nullptr) {
        OP_LOGE(context_->GetNodeName(), "Input descriptors, shapes and nine internal attributes are required.");
        return false;
    }
    const auto* dtype = attrs->GetAttrPointer<int64_t>(0);
    const auto* axis = attrs->GetAttrPointer<int64_t>(1);
    const auto* roundMode = attrs->GetAttrPointer<char>(2);
    const auto* scaleAlg = attrs->GetAttrPointer<int64_t>(3);
    const auto* dstTypeMax = attrs->GetAttrPointer<float>(4);
    const auto* trans = attrs->GetAttrPointer<bool>(5);
    const auto* groupType = attrs->GetAttrPointer<int64_t>(6);
    const auto* level0BlockSize = attrs->GetAttrPointer<int64_t>(7);
    const auto* level1BlockSize = attrs->GetAttrPointer<int64_t>(8);
    if (dtype == nullptr || axis == nullptr || roundMode == nullptr || scaleAlg == nullptr || dstTypeMax == nullptr ||
        trans == nullptr || groupType == nullptr || level0BlockSize == nullptr || level1BlockSize == nullptr) {
        return false;
    }
    attrs_ = {*axis, roundMode, *scaleAlg, *dstTypeMax, *trans, *groupType, *level0BlockSize, *level1BlockSize};
    for (size_t i = 0; i < 3; ++i) {
        if (context_->GetOutputDesc(i) == nullptr || context_->GetOutputShape(i) == nullptr ||
            ge::GetPrimaryFormat(context_->GetOutputDesc(i)->GetStorageFormat()) != ge::FORMAT_ND) {
            OP_LOGE(context_->GetNodeName(), "Three ND output descriptors and shapes are required.");
            return false;
        }
    }
    const auto xType = context_->GetInputDesc(0)->GetDataType();
    if ((xType != ge::DT_FLOAT16 && xType != ge::DT_BF16) || context_->GetInputDesc(1)->GetDataType() != xType ||
        context_->GetOutputDesc(1)->GetDataType() != ge::DT_FLOAT ||
        context_->GetOutputDesc(2)->GetDataType() != ge::DT_FLOAT8_E8M0 ||
        *dtype != context_->GetOutputDesc(0)->GetDataType()) {
        OP_LOGE(context_->GetNodeName(), "Dtypes must match the FP16/BF16 -> FP4/FP8,FP32,E8M0 contract.");
        return false;
    }
    for (size_t i = 0; i < 4; ++i) {
        const auto* desc = i < 2 ? context_->GetInputDesc(i) : context_->GetOptionalInputDesc(i);
        if (desc != nullptr && ge::GetPrimaryFormat(desc->GetStorageFormat()) != ge::FORMAT_ND) {
            OP_LOGE(context_->GetNodeName(), "All inputs must use ND format.");
            return false;
        }
    }
    const auto* alpha = context_->GetOptionalInputDesc(2);
    const auto* alphaShape = context_->GetOptionalInputShape(2);
    if (alpha != nullptr && (alpha->GetDataType() != ge::DT_BF16 || alphaShape == nullptr ||
                             Dimensions(alphaShape->GetStorageShape()) != std::vector<int64_t>{1})) {
        OP_LOGE(context_->GetNodeName(), "alpha must be BF16 [1].");
        return false;
    }
    const auto* groups = context_->GetOptionalInputDesc(3);
    const auto* groupShape = context_->GetOptionalInputShape(3);
    if (groups != nullptr && (groups->GetDataType() != ge::DT_INT64 || groupShape == nullptr)) {
        OP_LOGE(context_->GetNodeName(), "group_list must be INT64 with a shape.");
        return false;
    }
    const auto x = Dimensions(context_->GetInputShape(0)->GetStorageShape());
    const auto rotation = Dimensions(context_->GetInputShape(1)->GetStorageShape());
    const auto groupDims = groups == nullptr ? std::vector<int64_t>{} : Dimensions(groupShape->GetStorageShape());
    target_ = Target(context_->GetOutputDesc(0)->GetDataType());
    std::string reason;
    if (!RDLQ::ValidateShapes(x, rotation, groups == nullptr ? nullptr : &groupDims, false, shapeInfo_, reason) ||
        !RDLQ::ValidateAttributes(attrs_, x.size(), target_, reason)) {
        OP_LOGE(context_->GetNodeName(), "%s", reason.c_str());
        return false;
    }
    if (Dimensions(context_->GetOutputShape(0)->GetStorageShape()) != x ||
        Dimensions(context_->GetOutputShape(1)->GetStorageShape()) != shapeInfo_.level0Scale ||
        Dimensions(context_->GetOutputShape(2)->GetStorageShape()) != shapeInfo_.level1Scale) {
        OP_LOGE(context_->GetNodeName(), "Output shapes must be x.shape, [...,ceil(N/512)], [...,ceil(N/64),2].");
        return false;
    }
    data_.groupMode = groups != nullptr;
    data_.groupNum = groups == nullptr ? 0 : static_cast<uint32_t>(groupDims.front());
    data_.hasAlpha = alpha != nullptr;
    data_.yDtype = static_cast<int32_t>(*dtype);
    return true;
}

void RotateDualLevelQuantTiling::MakeLogicalTiling()
{
    data_.M = static_cast<uint32_t>(shapeInfo_.rows);
    data_.N = static_cast<uint32_t>(shapeInfo_.n);
    data_.rotK = static_cast<uint32_t>(shapeInfo_.rotK);
    data_.rotationBlocksPerRow = data_.N / data_.rotK;
    data_.rotationMatricesPerGroup = static_cast<uint32_t>(shapeInfo_.rotationsPerGroup);
    data_.level0BlockSize = static_cast<uint32_t>(attrs_.level0BlockSize);
    data_.level1BlockSize = static_cast<uint32_t>(attrs_.level1BlockSize);
    data_.level0ScalesPerRow = static_cast<uint32_t>(shapeInfo_.level0Scale.back());
    data_.level1ScalePairsPerRow = static_cast<uint32_t>(shapeInfo_.level1Scale[shapeInfo_.level1Scale.size() - 2]);
    data_.level1ScalesPerRow = data_.level1ScalePairsPerRow * RDLQ::SCALE_PAIR_SIZE;
    data_.lastLevel0BlockSize = data_.N == 0 ? 0 : (data_.N - 1) % data_.level0BlockSize + 1;
    data_.logicalBlockCount = static_cast<uint64_t>(data_.M) * data_.level0ScalesPerRow;
    data_.groupListType = static_cast<int32_t>(attrs_.groupListType);
    data_.axis = static_cast<int32_t>(attrs_.axis);
    data_.roundMode = attrs_.roundMode == "rint" ? 4 : (attrs_.roundMode == "round" ? 0 : 1);
    data_.scaleAlg = static_cast<int32_t>(attrs_.scaleAlg);
    data_.level0MaxValue = RDLQ::TargetMax(target_);
    data_.level0InvMaxValue = 1.0f / data_.level0MaxValue;
    data_.dstTypeMax = static_cast<float>(attrs_.dstTypeMax);
    data_.invDstTypeMax = data_.dstTypeMax == 0.0f ? 0.0f : 1.0f / data_.dstTypeMax;
    data_.usedCoreNum = shapeInfo_.empty ? 1 :
                                           (data_.groupMode ? compileInfo_.aicNum :
                                                              static_cast<uint32_t>(std::min<uint64_t>(
                                                                  compileInfo_.aicNum, data_.logicalBlockCount)));
}

bool RotateDualLevelQuantTiling::MakeKernelTiling()
{
    if (shapeInfo_.empty) {
        return true;
    }
    const auto invalidTiling = [this]() {
        OP_LOGE(context_->GetNodeName(),
                "Platform memory cannot hold a 16-row Cube tile and complete level0/MX workspace.");
        return false;
    };
    const auto& resources = compileInfo_;
    auto& data = data_;
    const uint64_t k = data.rotK;
    if (resources.aicNum == 0 || resources.ubSize > std::numeric_limits<uint32_t>::max() || data.M == 0 ||
        data.N == 0 || (k != 32 && k != 64 && k != 128 && k != 256) || data.N % k != 0) {
        return invalidTiling();
    }
    const uint64_t inputPitch = CeilDivPositive(k, RDLQ::FIXPIPE_ROW_PITCH_GRAN) * RDLQ::FIXPIPE_ROW_PITCH_GRAN;
    // The common Fixpipe Block keeps two L0B stages, including for K=256.
    const uint64_t maxBaseK = resources.l0BSize / (RDLQ::ROTATION_BUFFER_COUNT * k * sizeof(uint16_t));
    const uint64_t baseK = std::min(k, maxBaseK / RDLQ::CUBE_ALIGNMENT * RDLQ::CUBE_ALIGNMENT);
    if (baseK == 0) {
        return invalidTiling();
    }
    const uint64_t level0Columns = CeilDivPositive(data.N, RDLQ::LEVEL0_BLOCK_SIZE);
    const uint64_t targetRows = CeilDivPositive(
                                    CeilDivPositive(static_cast<uint64_t>(data.M) * level0Columns, resources.aicNum),
                                    RDLQ::CUBE_ALIGNMENT) *
                                RDLQ::CUBE_ALIGNMENT;
    uint64_t selectedRows = 0;
    for (uint64_t rows = RDLQ::CUBE_ALIGNMENT; rows <= RDLQ::MAX_ROWS_PER_AIV * RDLQ::AIV_NUM_RATIO;
         rows += RDLQ::CUBE_ALIGNMENT) {
        const uint64_t localRows = CeilDivPositive(rows, RDLQ::AIV_NUM_RATIO);
        const bool fits = RDLQ::ROTATION_BUFFER_COUNT * (rows * k + k * k) * sizeof(uint16_t) <= resources.l1Size &&
                          RDLQ::ROTATION_BUFFER_COUNT * rows * baseK * sizeof(uint16_t) <= resources.l0ASize &&
                          RDLQ::ROTATION_BUFFER_COUNT * rows * k * sizeof(float) <= resources.l0CSize &&
                          FixpipeReceiveBytes(localRows, inputPitch) + QuantWorkspaceBytes(localRows) <=
                              resources.ubSize;
        if (fits && rows <= targetRows) {
            selectedRows = rows;
        }
    }
    if (selectedRows == 0) {
        return invalidTiling();
    }

    const uint64_t localRows = CeilDivPositive(selectedRows, RDLQ::AIV_NUM_RATIO);
    const uint64_t receiveBytes = FixpipeReceiveBytes(localRows, inputPitch);
    const uint64_t workspaceBytes = QuantWorkspaceBytes(localRows);
    const uint64_t privateEnd = receiveBytes + workspaceBytes;
    const uint64_t tasks = CeilDivPositive(data.M, selectedRows) * level0Columns;
    const uint32_t cores = data.groupMode ? resources.aicNum :
                                            static_cast<uint32_t>(std::min<uint64_t>(resources.aicNum, tasks));
    // Select the row tile from single-slot capacity first. Do not shrink it just to enable UBDB.
    // Even one 512-column task can produce several K-wide Fixpipe chunks.
    const bool hasMultipleFixpipeResults = data.groupMode || data.N > data.rotK || tasks > cores;
    const uint32_t ubDB = hasMultipleFixpipeResults && privateEnd <= resources.ubSize / 2 ? 2 : 1;
    data.aivNumRatio = RDLQ::AIV_NUM_RATIO;
    data.splitM = 1;
    data.mL1 = data.baseM = static_cast<uint32_t>(selectedRows);
    data.nL1 = data.baseN = data.kL1 = data.rotK;
    data.baseK = static_cast<uint32_t>(baseK);
    data.maxRowsPerAiv = static_cast<uint32_t>(localRows);
    data.ubDB = ubDB;
    data.ubPitchGran = RDLQ::FIXPIPE_ROW_PITCH_GRAN;
    data.fixpipeReceiveBytes = static_cast<uint32_t>(receiveBytes);
    data.quantWorkspaceBytes = static_cast<uint32_t>(workspaceBytes);
    data.mxWorkspaceBytes = RDLQ::MX_WORKSPACE_BYTES;
    data.collectUbOffset = data.mxWorkspaceBytes;
    data.collectUbBytes = static_cast<uint32_t>(localRows * RDLQ::LEVEL0_BLOCK_SIZE * sizeof(uint16_t) +
                                                RDLQ::COLLECT_READ_GUARD_BYTES);
    data.level0ScaleUbOffset = data.collectUbOffset + data.collectUbBytes;
    data.level0ScaleUbBytes = static_cast<uint32_t>(localRows * RDLQ::LEVEL0_SCALE_ROW_BYTES);
    data.totalUbBytes = static_cast<uint32_t>(ubDB == 2 ? std::max(privateEnd, resources.ubSize / 2 + receiveBytes) :
                                                          privateEnd);
    data.usedCoreNum = cores;
    data.taskCount = data.groupMode ? 0 : tasks;
    data.userWorkspaceSize = 0;
    return true;
}

ge::graphStatus RotateDualLevelQuantTiling::SaveTiling()
{
    auto* raw = context_->GetRawTilingData();
    if (raw == nullptr || raw->GetData() == nullptr || raw->GetCapacity() < sizeof(data_) ||
        memcpy_s(raw->GetData(), raw->GetCapacity(), &data_, sizeof(data_)) != EOK) {
        OP_LOGE(context_->GetNodeName(), "Tiling data buffer is missing or too small.");
        return ge::GRAPH_FAILED;
    }
    auto* workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = shapeInfo_.empty ? 0 : RDLQ::SYSTEM_WORKSPACE_BYTES + data_.userWorkspaceSize;
    raw->SetDataSize(sizeof(data_));
    context_->SetTilingKey(data_.groupMode ? RDLQ::GROUPED_TILING_KEY : RDLQ::ORDINARY_TILING_KEY);
    context_->SetBlockDim(data_.usedCoreNum);
    context_->SetScheduleMode(1); // Static scheduling, matching the A5 RotateQuant outer contract.
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateDualLevelQuantTiling::DoTiling()
{
    if (context_ == nullptr || !ReadPlatform() || !ReadAndValidateInputs()) {
        return ge::GRAPH_FAILED;
    }
    MakeLogicalTiling();
    if (!MakeKernelTiling()) {
        return ge::GRAPH_FAILED;
    }
    return SaveTiling();
}

static ge::graphStatus Tiling(gert::TilingContext* context) { return RotateDualLevelQuantTiling(context).DoTiling(); }

static ge::graphStatus TilingParse(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto* info = context->GetCompiledInfo<RotateDualLevelQuantCompileInfo>();
    if (info == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // Leave an explicit invalid cache when platform metadata is unavailable.
    *info = RotateDualLevelQuantCompileInfo{};
    if (context->GetPlatformInfo() != nullptr) {
        const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
        FillPlatform(platform, *info);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RotateDualLevelQuant).Tiling(Tiling).TilingParse<RotateDualLevelQuantCompileInfo>(TilingParse);
} // namespace optiling
