/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file group_norm_swish_grad_tiling.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "platform/soc_spec.h"
#include "group_norm_swish_grad_tiling.h"
#include <string>
#include <vector>
#include "base/err_msg.h"
#include "util/shape_util.h"
#include "base/err_msg.h"

using namespace std;

namespace {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
} // namespace

namespace {
constexpr int64_t FP32_MODE = 0;
constexpr int64_t FP16_MODE = 1;
constexpr int64_t BF16_MODE = 2;
constexpr int64_t MODE_0 = 0;
constexpr int64_t MODE_1 = 1;
constexpr int64_t MODE_3 = 3;
constexpr int64_t INPUT_0 = 0;
constexpr int64_t INPUT_1 = 1;
constexpr int64_t INPUT_2 = 2;
constexpr int64_t INPUT_3 = 3;
constexpr int64_t INPUT_4 = 4;
constexpr int64_t INPUT_5 = 5;
constexpr int64_t DIM0 = 0;
constexpr int64_t DIM1 = 1;
constexpr int64_t DIM2 = 2;
constexpr int64_t DIM3 = 3;
constexpr int64_t NUM_GROUPS_IDX = 0;
constexpr int64_t SWISH_SCALE_IDX = 2;
constexpr int64_t DGAMMA_IS_REQUIRE_IDX = 3;
constexpr int64_t DBETA_IS_REQUIRE_IDX = 4;
constexpr int64_t UPPER_CARRYING_LIMIT = 4000;
constexpr uint64_t WORKSPACE_REVERSE = 16 * 1024 * 1024;
constexpr uint64_t TEN = 10;
constexpr uint64_t UB_COPIES_1 = 8;
constexpr uint64_t UB_COPIES_2 = 4;
constexpr uint64_t WORKSPACE_COPIES = 2;
constexpr uint64_t BLOCK_BYTES = 32;
constexpr uint64_t RESERVE_SAPCE = 1024;
constexpr uint64_t FLOAT_DTYPE_BYTES = 4;
constexpr uint64_t BFLOAT16_DTYPE_BYTES = 2;
constexpr uint64_t FLOAT16_DTYPE_BYTES = 2;
constexpr uint64_t EIGHT_BLOCK = 8;
constexpr uint64_t SPLIT_COUNT = 2;
constexpr uint64_t STEP_SIZE = 64;
constexpr uint32_t BATCH_MODE = 1;
constexpr uint64_t DATA_COPY_MAX_BLOCK_COUNT = 65535;
constexpr uint64_t STAGE2_LEGACY_REDUCE_BUFFER_COUNT = 1;
constexpr uint64_t STAGE2_KAHAN_BUFFER_COUNT = 3;
} // namespace

namespace optiling {

static bool IsArch3510(gert::TilingContext* tilingContext)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    return ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510;
}

class GroupNormSwishGradTiling {
public:
    explicit GroupNormSwishGradTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    ge::graphStatus SetTilingKeyMode(ge::DataType dtypeStr, uint64_t isDeterministicKey) const;
    ge::graphStatus ComputeAllocUBStage2(uint64_t coreBatchCounts, uint64_t availableSpace);
    ge::graphStatus CalDeterministicFp32Stage2(uint64_t availableSpace, uint64_t stage2ReduceBufferCount);
    ge::graphStatus CalDeterministicCastStage2(uint64_t availableSpace, uint64_t stage2ReduceBufferCount);
    void CalNonDeterministicCastStage2();
    void SetStage2CastInfo(uint64_t castEleNum);
    uint64_t GetDataTypeSize(ge::DataType dtypeStr) const;
    uint64_t GetElePerBlock(uint64_t dtypeBytes) const;
    uint64_t Ceil(uint64_t a, uint64_t b) const;
    uint64_t DivCeil(uint64_t a, uint64_t b) const;
    uint64_t Floor(uint64_t a, uint64_t b) const;
    ge::graphStatus CalStage2TilingInfo(ge::DataType dtypeStr, uint64_t isDeterministicKey, bool useKahanStage2);
    ge::graphStatus CalStage1TilingInfo(uint64_t reserveSpace);
    bool CheckInputDtype();
    bool CheckInputShape();
    bool PlanStepCoreUsage();
    GroupNormSwishGradTilingData tilingData;
    const GroupNormSwishGradCompileInfo* compileInfo = nullptr;
    ge::DataType dtypeStr;
    uint64_t dtypeBytes = 0;
    uint64_t elePerBlock = 0;
    std::unique_ptr<GroupNormSwishGradTilingCalculationParameters> tilingParams;
    gert::TilingContext* tilingContext = nullptr;
    const gert::RuntimeAttrs* attrs = nullptr;
};

bool GroupNormSwishGradTiling::CheckInputDtype()
{
    if ((tilingContext->GetInputDesc(INPUT_0) == nullptr || tilingContext->GetInputDesc(INPUT_1) == nullptr ||
         tilingContext->GetInputDesc(INPUT_2) == nullptr || tilingContext->GetInputDesc(INPUT_3) == nullptr ||
         tilingContext->GetInputDesc(INPUT_4) == nullptr || tilingContext->GetInputDesc(INPUT_5) == nullptr)) {
        const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                           "group_norm_swish_grad" :
                                           tilingContext->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       "Required input descriptors are unavailable in the Tiling context; check input "
                                       "registration and context construction.");
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return false;
    };
    auto inputDesc = tilingContext->GetInputDesc(INPUT_0);
    OP_CHECK_IF(inputDesc == nullptr, std::printf("nullptr error!"), return false);
    this->dtypeStr = inputDesc->GetDataType();
    OP_CHECK_IF(!(this->dtypeStr == tilingContext->GetInputDesc(INPUT_1)->GetDataType() &&
                  this->dtypeStr == tilingContext->GetInputDesc(INPUT_2)->GetDataType() &&
                  this->dtypeStr == tilingContext->GetInputDesc(INPUT_3)->GetDataType() &&
                  this->dtypeStr == tilingContext->GetInputDesc(INPUT_4)->GetDataType() &&
                  this->dtypeStr == tilingContext->GetInputDesc(INPUT_5)->GetDataType()),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    tilingContext->GetNodeName(), "dy,mean,rstd,x,gamma,beta",
                    std::to_string(static_cast<int>(tilingContext->GetInputDesc(INPUT_0)->GetDataType())) + "," +
                        std::to_string(static_cast<int>(tilingContext->GetInputDesc(INPUT_1)->GetDataType())) + "," +
                        std::to_string(static_cast<int>(tilingContext->GetInputDesc(INPUT_2)->GetDataType())) + "," +
                        std::to_string(static_cast<int>(tilingContext->GetInputDesc(INPUT_3)->GetDataType())) + "," +
                        std::to_string(static_cast<int>(tilingContext->GetInputDesc(INPUT_4)->GetDataType())) + "," +
                        std::to_string(static_cast<int>(tilingContext->GetInputDesc(INPUT_5)->GetDataType())),
                    "Input dtypeBytes is not same"),
                return false);
    this->dtypeBytes = GetDataTypeSize(this->dtypeStr);
    OP_CHECK_IF((this->dtypeBytes == 0), OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "dtypeBytes is zero"),
                return false);
    // Calculate elePerBlock based on the value of dtypeBytes
    this->elePerBlock = GetElePerBlock(this->dtypeBytes);
    if ((this->elePerBlock == 0)) {
        const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                           "group_norm_swish_grad" :
                                           tilingContext->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       std::string("The validated dtype produced no elements per block; dtypeBytes=") +
                                       std::to_string(dtypeBytes));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return false;
    };
    return true;
}

bool GroupNormSwishGradTiling::CheckInputShape()
{
    auto InputShape0 = tilingContext->GetInputShape(INPUT_0);
    OP_CHECK_IF(InputShape0 == nullptr, std::printf("nullptr error!"), return false);
    auto InputShape1 = tilingContext->GetInputShape(INPUT_1);
    OP_CHECK_IF(InputShape1 == nullptr, std::printf("nullptr error!"), return false);
    auto InputShape2 = tilingContext->GetInputShape(INPUT_2);
    OP_CHECK_IF(InputShape2 == nullptr, std::printf("nullptr error!"), return false);
    auto InputShape3 = tilingContext->GetInputShape(INPUT_3);
    OP_CHECK_IF(InputShape3 == nullptr, std::printf("nullptr error!"), return false);
    auto InputShape4 = tilingContext->GetInputShape(INPUT_4);
    OP_CHECK_IF(InputShape4 == nullptr, std::printf("nullptr error!"), return false);
    auto InputShape5 = tilingContext->GetInputShape(INPUT_5);
    OP_CHECK_IF(InputShape5 == nullptr, std::printf("nullptr error!"), return false);
    const gert::Shape dyShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    const gert::Shape meanShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    const gert::Shape rstdShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    const gert::Shape xShape = tilingContext->GetInputShape(INPUT_3)->GetStorageShape();
    const gert::Shape gammaShape = tilingContext->GetInputShape(INPUT_4)->GetStorageShape();
    const gert::Shape betaShape = tilingContext->GetInputShape(INPUT_5)->GetStorageShape();
    auto dimNum = dyShape.GetDimNum();
    OP_CHECK_IF(
        dyShape != xShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "dy,x",
                                               Ops::Base::ToString(dyShape) + "," + Ops::Base::ToString(xShape),
                                               "Input shape check Failed, dy shape should be same with x shape"),
        return false);
    attrs = tilingContext->GetAttrs();
    if ((attrs == nullptr)) {
        const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                           "group_norm_swish_grad" :
                                           tilingContext->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       "The Tiling runtime attribute container is unavailable; check operator "
                                       "registration and context construction.");
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return false;
    };
    tilingParams->g = *(attrs->GetAttrPointer<int32_t>(NUM_GROUPS_IDX));
    OP_CHECK_IF(
        static_cast<uint64_t>(meanShape.GetDim(DIM1)) != tilingParams->g,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            tilingContext->GetNodeName(), "num_groups", std::to_string(tilingParams->g),
            std::string("num_groups must equal mean.shape[1], expected=") + std::to_string(meanShape.GetDim(DIM1))),
        return false);
    tilingParams->n = dyShape.GetDim(DIM0);
    tilingParams->c = dyShape.GetDim(DIM1);
    OP_CHECK_IF(static_cast<uint64_t>(meanShape.GetDim(DIM0)) != tilingParams->n,
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    tilingContext->GetNodeName(), "mean,dy",
                    std::to_string(meanShape.GetDim(DIM0)) + "," + std::to_string(tilingParams->n),
                    "Check shape failed, mean.shape[0] should be same with N(x.shape[0])"),
                return false);
    OP_CHECK_IF(
        meanShape != rstdShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "mean,rstd",
                                               Ops::Base::ToString(meanShape) + "," + Ops::Base::ToString(rstdShape),
                                               "Check shape failed, Shape of mean and rstd not same."),
        return false);
    OP_CHECK_IF((meanShape.GetDimNum() != 2 || rstdShape.GetDimNum() != 2),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    tilingContext->GetNodeName(), "mean,rstd",
                    std::to_string(meanShape.GetDimNum()) + "," + std::to_string(rstdShape.GetDimNum()),
                    "Check shape failed, Dim of mean and rstd should be 2."),
                return false);
    OP_CHECK_IF(
        (gammaShape.GetDimNum() != 1 || static_cast<uint64_t>(gammaShape.GetDim(DIM0)) != tilingParams->c),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext->GetNodeName(), "gamma", Ops::Base::ToString(gammaShape),
                                              "Check shape failed, Shape of gamma should be (C,)."),
        return false);
    OP_CHECK_IF(
        (betaShape.GetDimNum() != 1 || static_cast<uint64_t>(betaShape.GetDim(DIM0)) != tilingParams->c),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(tilingContext->GetNodeName(), "beta", Ops::Base::ToString(betaShape),
                                              "Check shape failed, Shape of beta should be (C,)."),
        return false);
    tilingParams->nxg = tilingParams->n * tilingParams->g;
    OP_CHECK_IF(tilingParams->nxg == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    tilingContext->GetNodeName(), "num_groups", std::to_string(tilingParams->g),
                    "N and num_groups must have a nonzero product; check the input batch dimension and num_groups."),
                return false);
    tilingParams->channelPerGroup = tilingParams->c / tilingParams->g;
    // C / G must not be zero, and C must be an integer multiple of G
    OP_CHECK_IF((tilingParams->channelPerGroup == 0 || tilingParams->c % tilingParams->g != 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    tilingContext->GetNodeName(), "num_groups", std::to_string(tilingParams->g),
                    "num_groups must divide the input channel dimension and cannot exceed it."),
                return false);
    tilingParams->hxw = 1;
    for (uint64_t dimIdx = 2; dimIdx < dimNum; dimIdx++) {
        tilingParams->hxw *= dyShape.GetDim(dimIdx);
    }
    OP_CHECK_IF(
        tilingParams->hxw == 0,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "dy", std::to_string(tilingParams->hxw),
                                                  "Check shape failed, HxW should not be 0."),
        return false);
    return true;
}

ge::graphStatus GroupNormSwishGradTiling::ComputeAllocUBStage2(uint64_t coreBatchCounts, uint64_t availableSpace)
{
    if (tilingParams->castEleNum == 0) {
        const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                           "group_norm_swish_grad" :
                                           tilingContext->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       std::string("Stage2 computed a zero cast tile; castEleNum=") +
                                       std::to_string(tilingParams->castEleNum));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return ge::GRAPH_FAILED;
    };
    uint64_t coreBatchCapacity = availableSpace / (tilingParams->castEleNum * FLOAT_DTYPE_BYTES);
    coreBatchCapacity = std::min(coreBatchCapacity, DATA_COPY_MAX_BLOCK_COUNT);
    tilingParams->coreBatchParts = std::min(coreBatchCapacity, coreBatchCounts);
    if (tilingParams->coreBatchParts == 0) {
        const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                           "group_norm_swish_grad" :
                                           tilingContext->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       std::string("Stage2 cannot assign a batch tile; availableBytes=") +
                                       std::to_string(availableSpace) +
                                       ", batchCount=" + std::to_string(coreBatchCounts));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return ge::GRAPH_FAILED;
    };
    tilingParams->coreBatchPartsTailRepeat = (coreBatchCounts % tilingParams->coreBatchParts == 0) ?
                                                 tilingParams->coreBatchParts :
                                                 coreBatchCounts % tilingParams->coreBatchParts;
    tilingParams->repeatTime4Stage2 = DivCeil(coreBatchCounts, tilingParams->coreBatchParts);
    return ge::GRAPH_SUCCESS;
}

void GroupNormSwishGradTiling::SetStage2CastInfo(uint64_t castEleNum)
{
    // Common C-axis split metadata used by both stage2 reduce and cast-only paths.
    tilingParams->castEleNum = castEleNum;
    tilingParams->stage2CoreUsed = DivCeil(tilingParams->c, tilingParams->castEleNum);
    tilingParams->tailCastNum = (tilingParams->stage2CoreUsed == 1) ?
                                    tilingParams->c :
                                    tilingParams->c - (tilingParams->stage2CoreUsed - 1) * tilingParams->castEleNum;
}

ge::graphStatus GroupNormSwishGradTiling::CalDeterministicFp32Stage2(uint64_t availableSpace,
                                                                     uint64_t stage2ReduceBufferCount)
{
    // FP32 deterministic stage2 only needs reduce buffers plus row input space.
    uint64_t preferredCastEleNum = Ceil(DivCeil(tilingParams->c, tilingParams->coreNumUsed / SPLIT_COUNT), STEP_SIZE);
    uint64_t maxCastEleNumByUb = Floor(availableSpace / ((stage2ReduceBufferCount + 1) * FLOAT_DTYPE_BYTES), STEP_SIZE);
    if (maxCastEleNumByUb == 0) {
        const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                           "group_norm_swish_grad" :
                                           tilingContext->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       std::string("No Stage2 cast tile fits the available UB; availableBytes=") +
                                       std::to_string(availableSpace) +
                                       ", reduceBuffers=" + std::to_string(stage2ReduceBufferCount));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return ge::GRAPH_FAILED;
    };

    SetStage2CastInfo(std::min(preferredCastEleNum, maxCastEleNumByUb));
    uint64_t stage2ReduceSpace = tilingParams->castEleNum * stage2ReduceBufferCount * FLOAT_DTYPE_BYTES;
    uint64_t coreBatchCounts = DivCeil(DivCeil(tilingParams->n, SPLIT_COUNT), SPLIT_COUNT);
    OP_CHECK_IF(ComputeAllocUBStage2(coreBatchCounts, availableSpace - stage2ReduceSpace) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "fail to Alloc UB for Stage2"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormSwishGradTiling::CalDeterministicCastStage2(uint64_t availableSpace,
                                                                     uint64_t stage2ReduceBufferCount)
{
    // Low-precision deterministic stage2 reserves cast input/output space in addition to reduce buffers.
    uint64_t preferredCastEleNum = Ceil(DivCeil(tilingParams->c, tilingParams->coreNumUsed), STEP_SIZE);
    uint64_t maxCastEleNumByUb = Floor(
        availableSpace / (stage2ReduceBufferCount * FLOAT_DTYPE_BYTES + FLOAT16_DTYPE_BYTES + FLOAT_DTYPE_BYTES),
        STEP_SIZE);
    if (maxCastEleNumByUb == 0) {
        const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                           "group_norm_swish_grad" :
                                           tilingContext->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       std::string("No Stage2 cast tile fits the available UB; availableBytes=") +
                                       std::to_string(availableSpace) +
                                       ", reduceBuffers=" + std::to_string(stage2ReduceBufferCount));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return ge::GRAPH_FAILED;
    };

    SetStage2CastInfo(std::min(preferredCastEleNum, maxCastEleNumByUb));
    uint64_t stage2ReduceCastSpace = tilingParams->castEleNum *
                                     (stage2ReduceBufferCount * FLOAT_DTYPE_BYTES + FLOAT16_DTYPE_BYTES);
    uint64_t coreBatchCounts = DivCeil(tilingParams->n, SPLIT_COUNT);
    OP_CHECK_IF(ComputeAllocUBStage2(coreBatchCounts, availableSpace - stage2ReduceCastSpace) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "fail to Alloc UB for Stage2"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void GroupNormSwishGradTiling::CalNonDeterministicCastStage2()
{
    // Non-deterministic low-precision stage2 only casts the partial result back by channel split.
    uint64_t castEleNum = Ceil(DivCeil(tilingParams->c, tilingParams->coreNumUsed), STEP_SIZE);
    SetStage2CastInfo(castEleNum);
    tilingParams->workSpaceSize = tilingParams->c;
}

ge::graphStatus GroupNormSwishGradTiling::CalStage2TilingInfo(ge::DataType dtypeStrLocal, uint64_t isDeterministicKey,
                                                              bool useKahanStage2)
{
    size_t* currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    uint64_t ubSizePlatForm;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    size_t availableSpace = (ubSizePlatForm - RESERVE_SAPCE) / SPLIT_COUNT;
    if ((currentWorkSpace == nullptr)) {
        const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                           "group_norm_swish_grad" :
                                           tilingContext->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       "TilingContext workspace output container is unavailable; check the Tiling "
                                       "output contract, not device memory allocation.");
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return ge::GRAPH_FAILED;
    };
    size_t usrWorkspaceSize = 0;
    if (isDeterministicKey == 1) {
        // Kahan compensation increases per-output reduce buffers, so castEleNum must be capped by UB here.
        const uint64_t stage2ReduceBufferCount = useKahanStage2 ? STAGE2_KAHAN_BUFFER_COUNT :
                                                                  STAGE2_LEGACY_REDUCE_BUFFER_COUNT;
        tilingParams->workSpaceSize = DivCeil(tilingParams->n, SPLIT_COUNT) * tilingParams->c;
        usrWorkspaceSize = WORKSPACE_COPIES * tilingParams->workSpaceSize * FLOAT_DTYPE_BYTES;
        if (dtypeStrLocal == ge::DT_FLOAT) {
            OP_CHECK_IF(CalDeterministicFp32Stage2(availableSpace, stage2ReduceBufferCount) != ge::GRAPH_SUCCESS,
                        OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "fail to calculate fp32 Stage2 tiling"),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(CalDeterministicCastStage2(availableSpace, stage2ReduceBufferCount) != ge::GRAPH_SUCCESS,
                        OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "fail to calculate cast Stage2 tiling"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        if (dtypeStrLocal == ge::DT_FLOAT) {
            // no stage2 task
            usrWorkspaceSize = 0;
        } else {
            // task Cast
            CalNonDeterministicCastStage2();
            usrWorkspaceSize = WORKSPACE_COPIES * tilingParams->workSpaceSize * FLOAT_DTYPE_BYTES;
        }
    }
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    currentWorkSpace[0] = sysWorkspaceSize + usrWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormSwishGradTiling::CalStage1TilingInfo(uint64_t reserveSpace)
{
    uint64_t unalignedExtraSpace = (tilingParams->hxw % EIGHT_BLOCK == 0 || tilingParams->channelPerGroup == 1) ?
                                       0 :
                                       UB_COPIES_2 * Ceil(tilingParams->hxw, elePerBlock) * FLOAT_DTYPE_BYTES;
    // Prevent wrapping errors in uint32 subtraction
    uint64_t ubSizePlatForm;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    OP_CHECK_IF(ubSizePlatForm < reserveSpace,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    tilingContext->GetNodeName(), "x",
                    Ops::Base::ToString(tilingContext->GetInputShape(INPUT_3)->GetStorageShape()),
                    std::string("The per-group workload exceeds UB capacity: required reservedBytes=") +
                        std::to_string(reserveSpace) + ", availableBytes=" + std::to_string(ubSizePlatForm) +
                        "; reduce the per-group workload."),
                return ge::GRAPH_FAILED);
    tilingParams->mode0UbCapGNum = (ubSizePlatForm < reserveSpace + unalignedExtraSpace) ?
                                       0 :
                                       (ubSizePlatForm - reserveSpace - unalignedExtraSpace) /
                                           (Ceil(tilingParams->channelPerGroup * tilingParams->hxw, elePerBlock) *
                                            dtypeBytes * UB_COPIES_1);
    tilingParams->mode1UbCapCNum = (ubSizePlatForm - reserveSpace) /
                                   (Ceil(tilingParams->hxw, elePerBlock) * dtypeBytes * UB_COPIES_1);
    if (tilingParams->mode1UbCapCNum > 0) {
        tilingParams->mode1UbIterCNum = Ceil(tilingParams->channelPerGroup, tilingParams->mode1UbCapCNum) /
                                        tilingParams->mode1UbCapCNum;
        tilingParams->mode1UbTailCNum = (tilingParams->mode1UbIterCNum * tilingParams->mode1UbCapCNum -
                                             tilingParams->channelPerGroup ==
                                         0) ?
                                            tilingParams->mode1UbCapCNum :
                                            (tilingParams->channelPerGroup -
                                             ((tilingParams->mode1UbIterCNum - 1) * tilingParams->mode1UbCapCNum));
    }
    if (tilingParams->mode0UbCapGNum > 0) {
        tilingParams->tilingKey = MODE_0;
    } else if (tilingParams->mode0UbCapGNum <= 0 && tilingParams->mode1UbCapCNum == 1) {
        tilingParams->tilingKey = MODE_1;
    } else {
        tilingParams->tilingKey = MODE_3;
        tilingParams->mode2UbCapacityEle = Floor((ubSizePlatForm - reserveSpace) / (dtypeBytes * UB_COPIES_1),
                                                 elePerBlock * EIGHT_BLOCK);
        if (tilingParams->mode2UbCapacityEle == 0) {
            const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                               "group_norm_swish_grad" :
                                               tilingContext->GetNodeName());
            const std::string msgReason = (std::string(__func__) + ": " +
                                           std::string("Stage1 mode2 computed zero UB capacity; availableBytes=") +
                                           std::to_string(ubSizePlatForm) +
                                           ", reservedBytes=" + std::to_string(reserveSpace));
            OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
            REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                      (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
            return ge::GRAPH_FAILED;
        };
        tilingParams->mode2UbIterationNum = Ceil(tilingParams->hxw, tilingParams->mode2UbCapacityEle) /
                                            tilingParams->mode2UbCapacityEle;
        tilingParams->mode2UbTailNum = (tilingParams->hxw -
                                            tilingParams->mode2UbIterationNum * tilingParams->mode2UbCapacityEle ==
                                        0) ?
                                           tilingParams->mode2UbCapacityEle :
                                           (tilingParams->hxw -
                                            (tilingParams->mode2UbIterationNum - 1) * tilingParams->mode2UbCapacityEle);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormSwishGradTiling::SetTilingKeyMode(ge::DataType dtypeStrLocal,
                                                           uint64_t isDeterministicKey) const
{
    switch (dtypeStrLocal) {
        case ge::DT_BF16:
            tilingContext->SetTilingKey(BF16_MODE + isDeterministicKey * TEN);
            return ge::GRAPH_SUCCESS;
        case ge::DT_FLOAT16:
            tilingContext->SetTilingKey(FP16_MODE + isDeterministicKey * TEN);
            return ge::GRAPH_SUCCESS;
        case ge::DT_FLOAT:
            tilingContext->SetTilingKey(FP32_MODE + isDeterministicKey * TEN);
            return ge::GRAPH_SUCCESS;
        default:
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "dy",
                                                  std::to_string(static_cast<int>(dtypeStrLocal)),
                                                  "inputdtype must be in [float32, float16, bfloat16]");
            return ge::GRAPH_FAILED;
    }
}

uint64_t GroupNormSwishGradTiling::GetDataTypeSize(ge::DataType dtypeStrLocal) const
{
    switch (dtypeStrLocal) {
        case ge::DT_FLOAT:
            return FLOAT_DTYPE_BYTES;
        case ge::DT_BF16:
            return BFLOAT16_DTYPE_BYTES + FLOAT_DTYPE_BYTES;
        case ge::DT_FLOAT16:
            return FLOAT16_DTYPE_BYTES + FLOAT_DTYPE_BYTES;
        default:
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(tilingContext->GetNodeName(), "dy",
                                                  std::to_string(static_cast<int>(dtypeStrLocal)),
                                                  "inputdtype must be in [float32, float16, bfloat16]");
            return 0;
    }
}

uint64_t GroupNormSwishGradTiling::GetElePerBlock(uint64_t dtypeBytesLocal) const
{
    switch (dtypeBytesLocal) {
        case FLOAT_DTYPE_BYTES:
            return BLOCK_BYTES / FLOAT_DTYPE_BYTES;
        case FLOAT16_DTYPE_BYTES + FLOAT_DTYPE_BYTES:
            return BLOCK_BYTES / FLOAT16_DTYPE_BYTES;
        default:
            return 0;
    }
}

uint64_t GroupNormSwishGradTiling::Ceil(uint64_t a, uint64_t b) const
{
    if (b == 0) {
        const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                           "group_norm_swish_grad" :
                                           tilingContext->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       std::string("Tiling arithmetic denominator is zero; numerator=") +
                                       std::to_string(a) + ", denominator=" + std::to_string(b));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return 0;
    };
    return ((a - 1) / b + 1) * b;
}

uint64_t GroupNormSwishGradTiling::DivCeil(uint64_t a, uint64_t b) const
{
    if (b == 0) {
        const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                           "group_norm_swish_grad" :
                                           tilingContext->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       std::string("Tiling arithmetic denominator is zero; numerator=") +
                                       std::to_string(a) + ", denominator=" + std::to_string(b));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return 0;
    };
    return (a - 1) / b + 1;
}

uint64_t GroupNormSwishGradTiling::Floor(uint64_t a, uint64_t b) const
{
    if (b == 0) {
        const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                           "group_norm_swish_grad" :
                                           tilingContext->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       std::string("Tiling arithmetic denominator is zero; numerator=") +
                                       std::to_string(a) + ", denominator=" + std::to_string(b));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return 0;
    };
    return a / b * b;
}

bool GroupNormSwishGradTiling::PlanStepCoreUsage()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint64_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    tilingParams->taskNumPerCore = Ceil(tilingParams->nxg, totalCoreNum) / totalCoreNum;
    tilingParams->coreNumUsed = (tilingParams->nxg - 1) / tilingParams->taskNumPerCore + 1;
    if (tilingParams->coreNumUsed == 0) {
        const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                           "group_norm_swish_grad" :
                                           tilingContext->GetNodeName());
        const std::string msgReason = (std::string(__func__) + ": " +
                                       std::string("Core scheduling produced no usable cores; platformCores=") +
                                       std::to_string(totalCoreNum) + ", tasks=" + std::to_string(tilingParams->nxg));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return false;
    };
    tilingParams->taskNumPerTailCore = tilingParams->taskNumPerCore;
    tilingParams->tailCore = tilingParams->coreNumUsed;
    if (tilingParams->nxg % tilingParams->coreNumUsed != 0) {
        tilingParams->taskNumPerTailCore = tilingParams->taskNumPerCore - 1;
        tilingParams->tailCore = tilingParams->nxg % tilingParams->coreNumUsed;
    }
    return true;
}

ge::graphStatus GroupNormSwishGradTiling::Init()
{
    OP_LOGD(tilingContext, "Tiling initing.");
    try {
        tilingParams = std::make_unique<GroupNormSwishGradTilingCalculationParameters>();
    } catch (const std::bad_alloc& e) {
        do {
            const std::string requestedBytes = std::to_string(sizeof(GroupNormSwishGradTilingCalculationParameters));
            OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(),
                                   "Host allocation of Tiling parameters failed: requestedBytes=%s, "
                                   "allocator=std::make_unique, reason=%s.",
                                   requestedBytes.c_str(), e.what());
            const std::vector<const char*> keys = {"buf_size", "alloc_interface"};
            const std::vector<const char*> values = {requestedBytes.c_str(),
                                                     "std::make_unique<GroupNormSwishGradTilingCalculationParameters>"};
            REPORT_PREDEFINED_ERR_MSG("EH0010", keys, values);
        } while (0);
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(tilingParams == nullptr,
                VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "failed to instantiate tilingParams"),
                return ge::GRAPH_FAILED);
    // Get Inputs dtype
    OP_CHECK_IF(!CheckInputDtype(), OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "InputDtype Check Failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckInputShape(),
        do {
            if (tilingContext->GetInputShape(INPUT_0) == nullptr || tilingContext->GetInputShape(INPUT_1) == nullptr ||
                tilingContext->GetInputShape(INPUT_2) == nullptr || tilingContext->GetInputShape(INPUT_3) == nullptr ||
                tilingContext->GetInputShape(INPUT_4) == nullptr || tilingContext->GetInputShape(INPUT_5) == nullptr) {
                const std::string msgOpName = (tilingContext == nullptr || tilingContext->GetNodeName() == nullptr ?
                                                   "group_norm_swish_grad" :
                                                   tilingContext->GetNodeName());
                OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                                       "Input shape metadata is unavailable during Tiling context validation.");
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        msgOpName.c_str(), "Input shape metadata is unavailable during Tiling context validation."}));
            } else {
                OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "InputShape Check Failed.");
            }
        } while (0),
        return ge::GRAPH_FAILED);
    // Allocate computing core
    uint64_t channelPerGroupOnceProcess = Ceil(tilingParams->channelPerGroup, elePerBlock);
    // Check channelPerGroup not exceeding the operator's current carrying capacity.
    OP_CHECK_IF(
        (channelPerGroupOnceProcess > UPPER_CARRYING_LIMIT),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            tilingContext->GetNodeName(), "x",
            Ops::Base::ToString(tilingContext->GetInputShape(INPUT_3)->GetStorageShape()),
            std::string("Aligned channels per group exceed the implementation limit: actual=") +
                std::to_string(channelPerGroupOnceProcess) + ", maximum=" + std::to_string(UPPER_CARRYING_LIMIT)),
        return ge::GRAPH_FAILED);
    uint64_t reserveSpace = RESERVE_SAPCE + channelPerGroupOnceProcess * this->dtypeBytes * UB_COPIES_1;
    // The function PlanStepCoreUsage is to plan the usage of cores for each step.
    OP_CHECK_IF(!PlanStepCoreUsage(), OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "PlanStepCoreUsage failed."),
                return ge::GRAPH_FAILED);
    // Select UB allocation mode for stage_1
    OP_CHECK_IF(CalStage1TilingInfo(reserveSpace) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "fail to calculate Stage1 TilingInfo"),
                return ge::GRAPH_FAILED);
    // Because the accumulation axis is the N axis, there is no deterministic problem when N<=2.
    // On A5, use deterministic accumulation by default; A2/A3 still depend on the deterministic attribute.
    uint64_t isDeterministicKey = 0;
    const bool isArch3510 = IsArch3510(tilingContext);
    if (tilingParams->n > 2) {
        const auto deterministic = tilingContext->GetDeterministic();
        // A5 defaults to deterministic accumulation. A2/A3 still follow the deterministic attribute.
        isDeterministicKey = isArch3510 ? 1 : (deterministic == 1 ? 1 : 0);
    }
    OP_CHECK_IF(
        !(isDeterministicKey == 0 || isDeterministicKey == 1),
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(), "Error: isDeterministicKey must be 0 or 1!"),
        return ge::GRAPH_FAILED);
    // Set schedule mode to BATCH_MODE when isDeterministicKey is true.
    if (isDeterministicKey) {
        tilingContext->SetScheduleMode(BATCH_MODE);
    }
    // Set TilingKey mode
    OP_CHECK_IF(SetTilingKeyMode(this->dtypeStr, isDeterministicKey) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "fail to Set TilingKey"), return ge::GRAPH_FAILED);
    // Calculate workspace space
    OP_CHECK_IF(CalStage2TilingInfo(this->dtypeStr, isDeterministicKey, isArch3510) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "fail to calculate Stage2 TilingInfo"),
                return ge::GRAPH_FAILED);
    // Get ATTR
    tilingParams->dgammaIsRequire = static_cast<uint64_t>(*(attrs->GetAttrPointer<bool>(DGAMMA_IS_REQUIRE_IDX)));
    tilingParams->dbetaIsRequire = static_cast<uint64_t>(*(attrs->GetAttrPointer<bool>(DBETA_IS_REQUIRE_IDX)));
    tilingParams->swishScale = *(attrs->GetAttrPointer<float>(SWISH_SCALE_IDX));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupNormSwishGradTiling::SetKernelTiling()
{
    OP_LOGD(tilingContext, "Tiling start.");
    tilingContext->SetBlockDim(tilingParams->coreNumUsed);
    tilingData.set_Tiling_key(tilingParams->tilingKey);                              // 0
    tilingData.set_N(tilingParams->n);                                               // 1
    tilingData.set_C(tilingParams->c);                                               // 2
    tilingData.set_HXW(tilingParams->hxw);                                           // 3
    tilingData.set_G(tilingParams->g);                                               // 4
    tilingData.set_NXG(tilingParams->nxg);                                           // 5
    tilingData.set_C_G(tilingParams->channelPerGroup);                               // 6
    tilingData.set_task_num_per_core(tilingParams->taskNumPerCore);                  // 7
    tilingData.set_task_num_per_tail_core(tilingParams->taskNumPerTailCore);         // 8
    tilingData.set_tail_core(tilingParams->tailCore);                                // 9
    tilingData.set_mode1_ub_cap_C_num(tilingParams->mode1UbCapCNum);                 // 10
    tilingData.set_mode1_ub_iter_C_num(tilingParams->mode1UbIterCNum);               // 11
    tilingData.set_mode1_ub_tail_C_num(tilingParams->mode1UbTailCNum);               // 12
    tilingData.set_mode2_ub_capacity_ele(tilingParams->mode2UbCapacityEle);          // 13
    tilingData.set_mode2_ub_iteration_num(tilingParams->mode2UbIterationNum);        // 14
    tilingData.set_mode2_ub_tail_num(tilingParams->mode2UbTailNum);                  // 15
    tilingData.set_workSpaceSize(tilingParams->workSpaceSize);                       // 16
    tilingData.set_stage2CoreUsed(tilingParams->stage2CoreUsed);                     // 17
    tilingData.set_castEleNum(tilingParams->castEleNum);                             // 18
    tilingData.set_tailCastNum(tilingParams->tailCastNum);                           // 19
    tilingData.set_coreBatchParts(tilingParams->coreBatchParts);                     // 20
    tilingData.set_coreBatchPartsTailRepeat(tilingParams->coreBatchPartsTailRepeat); // 21
    tilingData.set_repeatTime4Stage2(tilingParams->repeatTime4Stage2);               // 22
    tilingData.set_dgamma_is_require(tilingParams->dgammaIsRequire);                 // 23
    tilingData.set_dbeta_is_require(tilingParams->dbetaIsRequire);                   // 24
    tilingData.set_swish_scale(tilingParams->swishScale);                            // 25
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    TilingDataPrint();
    OP_LOGD(tilingContext, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void GroupNormSwishGradTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext, "tilingKey:               %ld.", tilingParams->tilingKey);
    OP_LOGD(tilingContext, "N:                       %ld.", tilingParams->n);
    OP_LOGD(tilingContext, "C:                       %ld.", tilingParams->c);
    OP_LOGD(tilingContext, "HXW:                     %ld.", tilingParams->hxw);
    OP_LOGD(tilingContext, "G:                       %ld.", tilingParams->g);
    OP_LOGD(tilingContext, "NXG:                     %ld.", tilingParams->nxg);
    OP_LOGD(tilingContext, "channelPerGroup:         %ld.", tilingParams->channelPerGroup);
    OP_LOGD(tilingContext, "taskNumPerCore:          %ld.", tilingParams->taskNumPerCore);
    OP_LOGD(tilingContext, "taskNumPerTailCore:      %ld.", tilingParams->taskNumPerTailCore);
    OP_LOGD(tilingContext, "tailCore:                %ld.", tilingParams->tailCore);
    OP_LOGD(tilingContext, "mode1UbCapCNum:          %ld.", tilingParams->mode1UbCapCNum);
    OP_LOGD(tilingContext, "mode1UbIterCNum:         %ld.", tilingParams->mode1UbIterCNum);
    OP_LOGD(tilingContext, "mode1UbTailCNum:         %ld.", tilingParams->mode1UbTailCNum);
    OP_LOGD(tilingContext, "mode2UbCapacityEle:      %ld.", tilingParams->mode2UbCapacityEle);
    OP_LOGD(tilingContext, "mode2UbIterationNum:     %ld.", tilingParams->mode2UbIterationNum);
    OP_LOGD(tilingContext, "mode2UbTailNum:          %ld.", tilingParams->mode2UbTailNum);
    OP_LOGD(tilingContext, "workSpaceSize:           %ld.", tilingParams->workSpaceSize);
    OP_LOGD(tilingContext, "stage2CoreUsed:          %ld.", tilingParams->stage2CoreUsed);
    OP_LOGD(tilingContext, "castEleNum:              %ld.", tilingParams->castEleNum);
    OP_LOGD(tilingContext, "tailCastNum:             %ld.", tilingParams->tailCastNum);
    OP_LOGD(tilingContext, "coreBatchParts:          %ld.", tilingParams->coreBatchParts);
    OP_LOGD(tilingContext, "coreBatchPartsTailRepeat:%ld.", tilingParams->coreBatchPartsTailRepeat);
    OP_LOGD(tilingContext, "repeatTime4Stage2        %ld.", tilingParams->repeatTime4Stage2);
    OP_LOGD(tilingContext, "dgammaIsRequire:         %ld.", tilingParams->dgammaIsRequire);
    OP_LOGD(tilingContext, "dbetaIsRequire:          %ld.", tilingParams->dbetaIsRequire);
    OP_LOGD(tilingContext, "swishScale:              %f.", tilingParams->swishScale);
}

static ge::graphStatus TilingGroupNormSwishGrad(gert::TilingContext* context)
{
    GroupNormSwishGradTiling tilingObject(context);
    // Upstream behavior: Init() result is intentionally not checked here; tiling_3/4/5 rely on this.
    tilingObject.Init();
    return tilingObject.SetKernelTiling();
}

static ge::graphStatus TilingPrepareForGroupNormSwishGrad(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupNormSwishGrad)
    .Tiling(TilingGroupNormSwishGrad)
    .TilingParse<GroupNormSwishGradCompileInfo>(TilingPrepareForGroupNormSwishGrad);
} // namespace optiling
