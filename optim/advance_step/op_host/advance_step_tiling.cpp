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
 * \file advance_step_tiling.cpp
 * \brief
 */
#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "platform/platform_infos_def.h"
#include "advance_step_tiling_def.h"
#include <string>
#include "util/shape_util.h"
#include "base/err_msg.h"

namespace optiling {

static constexpr uint16_t SOC_VERSION_310 = 200;
static constexpr uint16_t SOC_VERSION_910 = 220;
static constexpr int64_t ONE_BLOCK_BYTE = 32;
static constexpr int64_t UB_RESERVED_BYTE = 768;
static constexpr int32_t RESERVED_WORKSPACE_SIZE = 32 * 1024 * 1024;
static constexpr int64_t IDX_INPUT__INPUT_TOKENS = 0;
static constexpr int64_t IDX_INPUT__SAMPLED_TOKEN_IDS = 1;
static constexpr int64_t IDX_INPUT__INPUT_POSITIONS = 2;
static constexpr int64_t IDX_INPUT__SEQ_LENS = 3;
static constexpr int64_t IDX_INPUT__SLOT_MAPPING = 4;
static constexpr int64_t IDX_INPUT__BLOCK_TABLES = 5;
static constexpr int64_t IDX_OPTIONAL_INPUT__SPEC_TOKEN = 6;
static constexpr int64_t IDX_OPTIONAL_INPUT__ACCEPTED_NUM = 7;
static constexpr int64_t IDX_ATTR__NUM_SEQS = 0;
static constexpr int64_t IDX_ATTR__NUM_QUERIES = 1;
static constexpr int64_t IDX_ATTR__BLOCK_SIZE = 2;
static constexpr int64_t MAX_NUMSEQS = 200000000;

inline static int64_t AlignBytes(int64_t elementNum, int64_t bytes = sizeof(int64_t))
{
    return (elementNum * bytes + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
}

inline static int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        int64_t rtTmp = ((x % y != 0) ? (x / y + 1) : (x / y));
        int64_t rtCd = (y != 0 && x != 0) ? rtTmp : x;
        return rtCd;
    } else {
        return 0;
    }
}

inline static int64_t FloorDiv(int64_t x, int64_t y) { return y == 0 ? x : x / y; }

static bool CheckOptionalShapeExisting(const gert::StorageShape* storageShape)
{
    OP_CHECK_IF(nullptr == storageShape, OP_LOGD("CheckOptionalShapeExisting: Get nullptr storageShape"), return false);
    int64_t smoothShapeSize = storageShape->GetOriginShape().GetShapeSize();
    OP_CHECK_IF((smoothShapeSize <= 0), OP_LOGD("CheckOptionalShapeExisting", "Get empty storageShape"), return false);
    return true;
}

bool AdvanceStepTilingHelper::GetBaseInfo()
{
    OP_LOGD(context_, "Enter GetBaseInfo");
    auto platformInfo = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    this->aivNum_ = platformInfo.GetCoreNumAiv();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, this->ubSize_);
    OP_LOGD(context_, "End GetBaseInfo");
    return true;
}

bool AdvanceStepTilingHelper::CheckAttrs()
{
    OP_LOGD(context_, "Enter CheckAttr");
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    if (nullptr == attrs) {
        const std::string msgOpName = std::string(Ops::Base::GetOpInfo(context_));
        const std::string msgReason = (std::string(__func__) + ": " + "Invalid RuntimeAttrs, tiling failed.");
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return false;
    };
    this->numSeqs_ = *(attrs->GetInt(IDX_ATTR__NUM_SEQS));
    this->numQueries_ = *(attrs->GetInt(IDX_ATTR__NUM_QUERIES));
    this->blockSize_ = *(attrs->GetInt(IDX_ATTR__BLOCK_SIZE));
    OP_CHECK_IF(
        this->numSeqs_ < 1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num_seqs", std::to_string(this->numSeqs_),
                                              "Non-positive attr num_seqs, tiling failed."),
        return false);
    OP_CHECK_IF(
        this->numQueries_ < 1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num_queries", std::to_string(this->numQueries_),
                                              "Non-positive attr num_queries, tiling failed."),
        return false);
    OP_CHECK_IF(
        this->blockSize_ < 1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "block_size", std::to_string(this->blockSize_),
                                              "Non-positive attr block_size, tiling failed."),
        return false);
    OP_CHECK_IF(this->numSeqs_ >= MAX_NUMSEQS,
                OP_LOGE(context_, "numSeqs %ld should be smaller than 200000000.", this->numSeqs_), return false);
    OP_LOGD(context_, "End CheckAttr");
    return true;
}

bool AdvanceStepTilingHelper::GetInputsOutputs()
{
    auto inputTokensShapePtr = context_->GetInputShape(IDX_INPUT__INPUT_TOKENS); // [num_seqs * (1 + spec_num),]
    auto sampledTokenIdsShapePtr = context_->GetInputShape(IDX_INPUT__SAMPLED_TOKEN_IDS); // [num_seqs, (1 + spec_num)]
    auto inputPositionsShapePtr = context_->GetInputShape(IDX_INPUT__INPUT_POSITIONS); // [num_seqs * (1 + spec_num),]
    auto seqLensShapePtr = context_->GetInputShape(IDX_INPUT__SEQ_LENS);               // [num_seqs * (1 + spec_num),]
    auto slotMappingShapePtr = context_->GetInputShape(IDX_INPUT__SLOT_MAPPING);       // [num_seqs * (1 + spec_num),]
    auto blockTablesShapePtr = context_->GetInputShape(IDX_INPUT__BLOCK_TABLES);       // [num_seqs, seq_lens_maxblock]
    OP_CHECK_IF(inputTokensShapePtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "advance_step" :
                                                                                     (context_)->GetNodeName()),
                    "input_tokens"),
                return false);
    OP_CHECK_IF(sampledTokenIdsShapePtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "advance_step" :
                                                                                     (context_)->GetNodeName()),
                    "sampled_token_ids"),
                return false);
    OP_CHECK_IF(inputPositionsShapePtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "advance_step" :
                                                                                     (context_)->GetNodeName()),
                    "input_positions"),
                return false);
    OP_CHECK_IF(seqLensShapePtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "advance_step" :
                                                                                     (context_)->GetNodeName()),
                    "seq_lens"),
                return false);
    OP_CHECK_IF(slotMappingShapePtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "advance_step" :
                                                                                     (context_)->GetNodeName()),
                    "slot_mapping"),
                return false);
    OP_CHECK_IF(blockTablesShapePtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "advance_step" :
                                                                                     (context_)->GetNodeName()),
                    "block_tables"),
                return false);
    this->inputTokensShape_ = &inputTokensShapePtr->GetStorageShape();
    this->sampledTokenIdsShape_ = &sampledTokenIdsShapePtr->GetStorageShape();
    this->inputPositionsShape_ = &inputPositionsShapePtr->GetStorageShape();
    this->seqLensShape_ = &seqLensShapePtr->GetStorageShape();
    this->slotMappingShape_ = &slotMappingShapePtr->GetStorageShape();
    this->blockTablesShape_ = &blockTablesShapePtr->GetStorageShape();

    auto specTokenShapePtr = context_->GetOptionalInputShape(IDX_OPTIONAL_INPUT__SPEC_TOKEN); // [num_seqs, spec_num]
    auto acceptedNumShapePtr = context_->GetOptionalInputShape(IDX_OPTIONAL_INPUT__ACCEPTED_NUM); // [num_seqs, ]
    this->specTokenExist = CheckOptionalShapeExisting(specTokenShapePtr);
    this->acceptedNumExist = CheckOptionalShapeExisting(acceptedNumShapePtr);
    this->specTokenShape_ = specTokenExist ? &specTokenShapePtr->GetStorageShape() : nullptr;
    this->acceptedNumShape_ = acceptedNumExist ? &acceptedNumShapePtr->GetStorageShape() : nullptr;

    return true;
}

bool AdvanceStepTilingHelper::CheckOptionalInputs()
{
    OP_LOGD(context_, "Enter CheckOptionalInputs");
    auto attrs = context_->GetAttrs();
    this->numSeqs_ = *(attrs->GetInt(IDX_ATTR__NUM_SEQS));
    // check optional inputs if exist
    OP_CHECK_IF(this->specTokenExist &&
                    (this->specTokenShape_->GetDimNum() != 2 || this->specTokenShape_->GetDim(0) != this->numSeqs_),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "spec_token",
                                                      Ops::Base::ToString(*this->specTokenShape_),
                                                      "specToken shape is not [num_seqs, spec_num]."),
                return false);
    this->specNum_ = this->specTokenExist ? this->specTokenShape_->GetDim(1) : 0;
    this->tokenEachReqs_ = this->specNum_ + 1;
    OP_CHECK_IF(this->acceptedNumExist &&
                    (this->acceptedNumShape_->GetDimNum() != 1 || this->acceptedNumShape_->GetDim(0) != this->numSeqs_),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "accepted_num",
                                                      Ops::Base::ToString(*this->acceptedNumShape_),
                                                      "acceptedNum shape is not [num_seqs, ]."),
                return false);
    OP_LOGD(context_, "End CheckOptionalInputs");
    return true;
}

bool AdvanceStepTilingHelper::CheckInputOutputShape()
{
    OP_LOGD(context_, "Enter CheckInputOutputShape");
    OP_CHECK_IF(this->numSeqs_ != this->numQueries_,
                OP_LOGE(context_, "numSeqs %ld should be equal to numQueries %ld.", this->numSeqs_, this->numQueries_),
                return false);
    OP_CHECK_IF(this->inputTokensShape_->GetDimNum() != 1 ||
                    this->inputTokensShape_->GetDim(0) != this->numSeqs_ * this->tokenEachReqs_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "input_tokens",
                                                      Ops::Base::ToString(*this->inputTokensShape_),
                                                      "inputTokens shape is not [num_seqs * (1 + spec_num),]."),
                return false);
    OP_CHECK_IF(this->sampledTokenIdsShape_->GetDimNum() != 2 ||
                    this->sampledTokenIdsShape_->GetDim(0) != this->numSeqs_ ||
                    this->sampledTokenIdsShape_->GetDim(1) != this->tokenEachReqs_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "sampled_token_ids",
                                                      Ops::Base::ToString(*this->sampledTokenIdsShape_),
                                                      "sampledTokenIds shape is not [num_seqs, (1 + spec_num)]."),
                return false);
    OP_CHECK_IF(*this->inputPositionsShape_ != *this->inputTokensShape_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "input_positions",
                                                      Ops::Base::ToString(*this->inputPositionsShape_),
                                                      "inputPosisiongs shape is not [num_seqs * (1 + spec_num),]."),
                return false);
    OP_CHECK_IF(*this->seqLensShape_ != *this->inputTokensShape_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "seq_lens",
                                                      Ops::Base::ToString(*this->seqLensShape_),
                                                      "seqLens shape is not [num_seqs * (1 + spec_num),]."),
                return false);
    OP_CHECK_IF(*this->slotMappingShape_ != *this->inputTokensShape_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "slot_mapping",
                                                      Ops::Base::ToString(*this->slotMappingShape_),
                                                      "slotMapping shape is not [num_seqs * (1 + spec_num),]."),
                return false);
    OP_CHECK_IF(this->blockTablesShape_->GetDimNum() != 2 || this->blockTablesShape_->GetDim(0) != this->numSeqs_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "block_tables",
                                                      Ops::Base::ToString(*this->blockTablesShape_),
                                                      "blockTables shape 1st dim is not num_seqs."),
                return false);
    this->blockTablesStride_ = this->blockTablesShape_->GetDim(1);
    OP_CHECK_IF(this->blockTablesStride_ < 1,
                OP_LOGE(context_, "blockTables shape 2nd dim must be positive, empty tensor is not supported."),
                return false);
    OP_LOGD(context_, "End CheckInputOutputShape");
    return true;
}

bool AdvanceStepTilingHelper::CheckInputOutputFormat() const
{
    OP_LOGD(context_, "Enter CheckInputOutputFormat");
    // need no more null check for InputDesc
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__INPUT_TOKENS)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "inputTokensFormat is not ND."), return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__SAMPLED_TOKEN_IDS)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "sampledTokenIdsFormat is not ND."), return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__INPUT_POSITIONS)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "inputPositionsFormat is not ND."), return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__SEQ_LENS)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "seqLensFormat is not ND."), return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__SLOT_MAPPING)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "slotMappingFormat is not ND."), return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__BLOCK_TABLES)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "blockTablesFormat is not ND."), return false);
    // check OptionalInputFormat if exist
    OP_CHECK_IF(this->specTokenExist &&
                    context_->GetOptionalInputDesc(IDX_OPTIONAL_INPUT__SPEC_TOKEN)->GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE(context_, "specTokenFormat is not ND."), return false);
    OP_CHECK_IF(
        this->acceptedNumExist &&
            context_->GetOptionalInputDesc(IDX_OPTIONAL_INPUT__ACCEPTED_NUM)->GetStorageFormat() != ge::FORMAT_ND,
        OP_LOGE(context_, "acceptedNumFormat is not ND."), return false);
    OP_LOGD(context_, "End CheckInputOutputFormat");
    return true;
}

bool AdvanceStepTilingHelper::CheckInputOutputDType() const
{
    OP_LOGD(context_, "Enter CheckInputOutputDType");
    // need no more null check for InputDesc
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__INPUT_TOKENS)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "input_tokens",
                    std::to_string(static_cast<int>(context_->GetInputDesc(IDX_INPUT__INPUT_TOKENS)->GetDataType())),
                    "inputTokensDType is not DT_INT64."),
                return false);
    OP_CHECK_IF(
        context_->GetInputDesc(IDX_INPUT__SAMPLED_TOKEN_IDS)->GetDataType() != ge::DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "sampled_token_ids",
            std::to_string(static_cast<int>(context_->GetInputDesc(IDX_INPUT__SAMPLED_TOKEN_IDS)->GetDataType())),
            "sampledTokenIdsDType is not DT_INT64."),
        return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__INPUT_POSITIONS)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "input_positions",
                    std::to_string(static_cast<int>(context_->GetInputDesc(IDX_INPUT__INPUT_POSITIONS)->GetDataType())),
                    "inputPositionsDType is not DT_INT64."),
                return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__SEQ_LENS)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "seq_lens",
                    std::to_string(static_cast<int>(context_->GetInputDesc(IDX_INPUT__SEQ_LENS)->GetDataType())),
                    "seqLensDType is not DT_INT64."),
                return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__SLOT_MAPPING)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "slot_mapping",
                    std::to_string(static_cast<int>(context_->GetInputDesc(IDX_INPUT__SLOT_MAPPING)->GetDataType())),
                    "slotMappingDType is not DT_INT64."),
                return false);
    OP_CHECK_IF(context_->GetInputDesc(IDX_INPUT__BLOCK_TABLES)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "block_tables",
                    std::to_string(static_cast<int>(context_->GetInputDesc(IDX_INPUT__BLOCK_TABLES)->GetDataType())),
                    "blockTablesDType is not DT_INT64."),
                return false);
    // check OptionalInputDType if exist
    OP_CHECK_IF(this->specTokenExist &&
                    context_->GetOptionalInputDesc(IDX_OPTIONAL_INPUT__SPEC_TOKEN)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "spec_token",
                    std::to_string(static_cast<int>(
                        context_->GetOptionalInputDesc(IDX_OPTIONAL_INPUT__SPEC_TOKEN)->GetDataType())),
                    "specTokenDType is not DT_INT64."),
                return false);
    OP_CHECK_IF(this->acceptedNumExist &&
                    context_->GetOptionalInputDesc(IDX_OPTIONAL_INPUT__ACCEPTED_NUM)->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "accepted_num",
                    std::to_string(static_cast<int>(
                        context_->GetOptionalInputDesc(IDX_OPTIONAL_INPUT__ACCEPTED_NUM)->GetDataType())),
                    "acceptedNumDType is not DT_INT64."),
                return false);
    OP_LOGD(context_, "End CheckInputOutputDType");
    return true;
}

bool AdvanceStepTilingHelper::Tiling4Seqs()
{
    OP_LOGD(context_, "Enter Tiling4Seqs");
    this->perCoreSeqs_ = CeilDiv(this->numSeqs_, static_cast<int64_t>(this->aivNum_));
    if (this->perCoreSeqs_ <= 0) {
        const std::string msgOpName = std::string(Ops::Base::GetOpInfo(context_));
        const std::string msgReason = (std::string(__func__) + ": " +
                                       std::string("Tiling produced no sequences per core: num_seqs=") +
                                       std::to_string(this->numSeqs_) +
                                       ", platformCores=" + std::to_string(this->aivNum_));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return false;
    };
    this->needCoreNum_ = CeilDiv(this->numSeqs_, this->perCoreSeqs_);
    this->lastCoreSeqs_ = this->numSeqs_ - (this->needCoreNum_ - 1) * this->perCoreSeqs_;

    // input oneline buffers, use 4 int64 buffers and 1 int32 buffer with an addtional block
    int64_t perCoreInBufSize = 4 * AlignBytes(this->tokenEachReqs_ +
                                              ONE_BLOCK_BYTE / static_cast<int64_t>(sizeof(int64_t))) +
                               AlignBytes(this->tokenEachReqs_ + ONE_BLOCK_BYTE / static_cast<int64_t>(sizeof(int64_t)),
                                          static_cast<int64_t>(sizeof(int32_t)));
    // buffers to calc ReduceMin
    int64_t perCoreTempBufSize = AlignBytes(1 + this->tokenEachReqs_, static_cast<int64_t>(sizeof(float)));
    // calc: ubRequired = perCoreInBufSize + tempBufSize + UB_RESERVED_BYTE
    int64_t perCoreAvaliableSize = static_cast<int64_t>(this->ubSize_) - perCoreTempBufSize - UB_RESERVED_BYTE;
    OP_CHECK_IF(perCoreAvaliableSize <= perCoreInBufSize,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context_->GetNodeName(), "spec_token", std::to_string(this->specNum_),
                    std::string("The per-sequence tile exceeds UB capacity: inputBytes=") +
                        std::to_string(perCoreInBufSize) + ", availableBytes=" + std::to_string(perCoreAvaliableSize)),
                return false);
    // calc perLoopSeqs
    this->perLoopMaxSeqs_ = FloorDiv(perCoreAvaliableSize, perCoreInBufSize);
    // only supports full-load
    this->ubTilingPolicy_ = UB_TILING_POLICY::FULL_LOAD;
    OP_LOGD(context_, "End Tiling4Seqs");
    return true;
}

ge::graphStatus AdvanceStepTilingHelper::DoTiling()
{
    OP_LOGD(context_, "Enter DoTiling");
    OP_CHECK_IF(!GetBaseInfo(), OP_LOGE_WITHOUT_REPORT(context_, "GetBaseInfo falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckAttrs(), OP_LOGE_WITHOUT_REPORT(context_, "CheckAttrs falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!GetInputsOutputs(), OP_LOGE_WITHOUT_REPORT(context_, "GetInputs falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputOutputFormat(),
                OP_LOGE_WITHOUT_REPORT(context_, "CheckInputOutputFormat falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputOutputDType(),
                OP_LOGE_WITHOUT_REPORT(context_, "CheckInputOutputDType falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckOptionalInputs(), OP_LOGE_WITHOUT_REPORT(context_, "CheckOptionalInputs falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(this->specTokenExist && this->acceptedNumExist),
                OP_LOGI(context_, "No optional inputs, use legacy tiling."), return this->Tiling4AdvanceStepLegacy());
    OP_CHECK_IF(!CheckInputOutputShape(),
                OP_LOGE_WITHOUT_REPORT(context_, "CheckInputOutputShape falied, return false"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!Tiling4Seqs(), OP_LOGE_WITHOUT_REPORT(context_, "DoBlockTiling falied, return false"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context_, "End DoTiling");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdvanceStepTilingHelper::PostTiling(AdvanceStepTilingData* tilingData) const
{
    OP_LOGD(context_, "Enter PostTiling");
    // set tiling data
    context_->SetBlockDim(this->needCoreNum_);
    tilingData->set_needCoreNum(this->needCoreNum_);
    tilingData->set_blockTablesStride(this->blockTablesStride_);
    tilingData->set_numSeqs(this->numSeqs_);
    tilingData->set_numQueries(this->numQueries_);
    tilingData->set_blockSize(this->blockSize_);
    tilingData->set_perCoreSeqs(this->perCoreSeqs_);
    tilingData->set_lastCoreSeqs(this->lastCoreSeqs_);
    tilingData->set_perLoopMaxSeqs(this->perLoopMaxSeqs_);
    tilingData->set_specNum(this->specNum_);
    OP_LOGD("AdvanceStepTilingHelper::PostTiling", "Tilingdata perCoreSeqs: %ld, lastCoreSeqs: %ld, specNum: %ld",
            this->perCoreSeqs_, this->lastCoreSeqs_, this->specNum_);

    // set tiling key
    uint32_t tilingKey = static_cast<uint32_t>(this->ubTilingPolicy_);
    context_->SetTilingKey(tilingKey);
    if (tilingKey == 1) {
        tilingData->set_needCoreNum(this->aivNum_);
    }
    tilingData->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData->GetDataSize());

    // set workspace size
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(RESERVED_WORKSPACE_SIZE);

    OP_LOGD(context_, "End PostTiling");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdvanceStepTilingHelper::Tiling4AdvanceStepLegacy()
{
    auto input0_storage_shape = context_->GetInputShape(0)->GetStorageShape();
    auto input1_storage_shape = context_->GetInputShape(1)->GetStorageShape();
    auto input2_storage_shape = context_->GetInputShape(2)->GetStorageShape();
    auto input3_storage_shape = context_->GetInputShape(3)->GetStorageShape();
    auto input4_storage_shape = context_->GetInputShape(4)->GetStorageShape();
    auto input5_storage_shape = context_->GetInputShape(5)->GetStorageShape();
    OP_CHECK_IF(this->numSeqs_ <= this->numQueries_,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    "CheckAdvTiling", "num_seqs,num_queries",
                    std::to_string(this->numSeqs_) + "," + std::to_string(this->numQueries_),
                    ("numSeqs " + std::to_string(this->numSeqs_) + " should not be smaller or equal to numQueries " +
                     std::to_string(this->numQueries_) + " ")),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(input0_storage_shape.GetDim(0) != this->numSeqs_,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "input_tokens", std::to_string(input0_storage_shape.GetDim(0)),
                    ("numSeqs " + std::to_string(this->numSeqs_) + " should equal to inputTokens's dim " +
                     std::to_string(input0_storage_shape.GetDim(0)) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input2_storage_shape.GetDim(0) != this->numSeqs_,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "input_positions", std::to_string(input2_storage_shape.GetDim(0)),
                    ("numSeqs " + std::to_string(this->numSeqs_) + " should equal to inputPositions's dim " +
                     std::to_string(input2_storage_shape.GetDim(0)) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input3_storage_shape.GetDim(0) != this->numSeqs_,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "seq_lens", std::to_string(input3_storage_shape.GetDim(0)),
                    ("numSeqs " + std::to_string(this->numSeqs_) + " should equal to seqLens's dim " +
                     std::to_string(input3_storage_shape.GetDim(0)) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input4_storage_shape.GetDim(0) != this->numSeqs_,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "slot_mapping", std::to_string(input4_storage_shape.GetDim(0)),
                    ("numSeqs " + std::to_string(this->numSeqs_) + " should equal to slotMapping's dim " +
                     std::to_string(input4_storage_shape.GetDim(0)) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(input5_storage_shape.GetDim(0) != this->numSeqs_,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    "CheckAdvTiling", "block_tables", std::to_string(input5_storage_shape.GetDim(0)),
                    ("numSeqs " + std::to_string(this->numSeqs_) + " should equal to blockTables's first dim " +
                     std::to_string(input5_storage_shape.GetDim(0)) + " ")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        input1_storage_shape.GetDim(0) != this->numQueries_,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            "CheckAdvTiling", "sampled_token_ids", std::to_string(input1_storage_shape.GetDim(0)),
            ("numQueries " + std::to_string(this->numQueries_) + " should equal to sampledTokenIds's first dim " +
             std::to_string(input1_storage_shape.GetDim(0)) + " ")),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(input1_storage_shape.GetDim(1) != 1,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("CheckAdvTiling", "sampled_token_ids",
                                                          std::to_string(input1_storage_shape.GetDim(1)),
                                                          "sampledTokenIds's last dim should be 1"),
                return ge::GRAPH_FAILED);
    this->blockTablesStride_ = input5_storage_shape.GetShapeSize() / input5_storage_shape.GetDim(0);

    this->ubTilingPolicy_ = UB_TILING_POLICY::ONE_CORE_NO_SPEC;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4AdvanceStep(gert::TilingContext* context)
{
    if (nullptr == context) {
        OP_LOGE_WITHOUT_REPORT("AdvanceStep", "%s", "Context is null");
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"AdvanceStep", "Context is null"}));
        return ge::GRAPH_FAILED;
    };
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for Arch35 architecture");
    OP_LOGI(context, "Enter Tiling4AdvanceStep");
    AdvanceStepTilingData tiling;
    AdvanceStepTilingHelper instanceAdvanceStepTilingHelper(context);
    auto status = instanceAdvanceStepTilingHelper.DoTiling();
    OP_CHECK_IF(status != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(context, "DoTiling Failed, return Failed."),
                return ge::GRAPH_FAILED);
    status = instanceAdvanceStepTilingHelper.PostTiling(&tiling);

    OP_LOGI(context, "End Tiling4AdvanceStep");
    return status;
}

static ge::graphStatus TilingPrepare4AdvanceStep([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdvanceStep).Tiling(Tiling4AdvanceStep).TilingParse<AdvanceStepCompileInfo>(TilingPrepare4AdvanceStep);
} // namespace optiling
