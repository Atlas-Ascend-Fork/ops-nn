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
 * \file ctc_loss_v3_grad.cpp
 * \brief
 */
#include "ctc_loss_v3_grad_tiling.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

using namespace ge;
using namespace std;

namespace optiling {
namespace {
ge::graphStatus ReportTilingFailure(const char* opName, const std::string& reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    return ge::GRAPH_FAILED;
}
} // namespace
constexpr uint64_t RESERVE_SAPCE = 1.5 * 1024;
constexpr int64_t SELECT_SPACE = 8 * 1024;
constexpr uint32_t FLOAT_DTYPE_BYTES = 4;
constexpr int64_t T_DIM = 0;
constexpr int64_t N_DIM = 1;
constexpr int64_t C_DIM = 2;
constexpr int64_t BATCH_DIM = 0;
constexpr int64_t ALPHA_T_DIM = 1;
constexpr int64_t ALPHA_DIM = 2;
constexpr int64_t S_DIM = 2;
constexpr int64_t S_COE = 2;
constexpr int64_t ALIGN = 8;
constexpr int64_t INPUT_GRAD_OUT_IDX = 0;
constexpr int64_t INPUT_LOG_PROB_IDX = 1;
constexpr int64_t INPUT_TRAGETS_IDX = 2;
constexpr int64_t INPUT_INPUT_LENGTHS_IDX = 3;
constexpr int64_t INPUT_TARGET_LENGTHS_IDX = 4;
constexpr int64_t INPUT_LOSS_IDX = 5;
constexpr int64_t INPUT_LOG_ALPHA_IDX = 6;
constexpr int64_t LOSS_DIM_NUM = 1;
constexpr int64_t GRAD_DIM_NUM = 3;
constexpr int64_t TARGETS_DIM_NUM = 2;
constexpr int64_t GRAD_RESERVED_UB = 40 * 1024;
constexpr int64_t MAX_SYMBOL_SET = 200000;
constexpr int64_t MAX_LABEL_LEN = 1000;
constexpr int64_t MAX_BATCH = 10000;
constexpr int64_t LARGE_BATCH = 1500;
constexpr int64_t NUM_PER_REPEAT = 64;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t FLOAT_DSIZE = 4;
constexpr int64_t INT64_DSIZE = 8;
constexpr int64_t MASKNUM_PER_BYTES = 8;
constexpr int64_t WORKSPACE_ALPHA_NUM = 4;
constexpr int64_t TARGETS_GM_NUM = 2;
constexpr int64_t TARGETS_UB_NUM = 4;
constexpr int64_t TARGETS_LENGTH_UB_NUM = 5;
constexpr int64_t MASK_UB_NUM = 5;
constexpr int64_t TARGETS_LENGTH_MASK_UB_NUM = 2;
constexpr int64_t ALPHA_LENGTH_NUM = 8;
const std::string OP_NAME = "CTCLossV3Grad";

class CTCLossV3GradTiling {
public:
    explicit CTCLossV3GradTiling(gert::TilingContext* ctx) : context(ctx) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    bool CheckShapeInfo();
    bool CheckShapeInfoForN();
    bool CheckShapeInfoForCT();
    void PrintTilingData() const;
    void TilingDataPrint();

private:
    CTCLossV3GradTilingData tilingData;
    gert::TilingContext* context = nullptr;
    int64_t sysWorkspaceSize = 16 * 1024 * 1024;
    int64_t useWorkspaceSize = 0;
    int64_t coreUsed = 48;
    int64_t sliceLength = 1;
    int64_t sliceLengthTail = 1;
    int64_t probSliceNum = 1;
    int64_t alphaLength = 0;

    int64_t maxInputLength = 1;
    int64_t sDimRange = 1;
    int64_t maxTargetLength = 1;
    int64_t symbolSet = 1;

    int64_t batchSize = 192 * 1024;
    int64_t targetsDimNum = 1;
    int64_t targetsNum = 1;
    int64_t taskPerCore = 1;
    int64_t taskTailCore = 1;
    int64_t BLANK = 0;
    int64_t zeroInfinity = 0;
    int64_t targetDsize = 0;
    int64_t gradDsize = 0;
    int64_t gradOutN = 0;
    int64_t targetsN = 0;
    int64_t inputLengthsN = 0;
    int64_t targetLengthsN = 0;
    int64_t lossN = 0;
    int64_t logAlphaN = 0;
    int64_t gradN = 0;
    int64_t gradT = 0;
    int64_t gradC = 0;
    int64_t logAlphaT = 0;
};

bool CTCLossV3GradTiling::CheckShapeInfoForN()
{
    // Check the shape info for some inputs, focusing mainly on batchSize.
    auto nodeName = context->GetNodeName();
    auto const gradOutShape = context->GetInputShape(INPUT_GRAD_OUT_IDX);
    OP_CHECK_IF(gradOutShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "gradOutShape",
                                                         "failed to get the input shape of grad_out from the context"),
                return false);
    auto const gradOutShapeVal = gradOutShape->GetStorageShape();
    OP_CHECK_IF(gradOutShapeVal.GetDimNum() != LOSS_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "grad_out",
                                             (std::to_string(gradOutShapeVal.GetDimNum()) + "D").c_str(), "1D"),
                return false);
    gradOutN = gradOutShapeVal.GetDim(BATCH_DIM);

    auto const inputLengthsShape = context->GetInputShape(INPUT_INPUT_LENGTHS_IDX);
    OP_CHECK_IF(inputLengthsShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "inputLengthsShape",
                                                         "failed to get the input shape of input_lengths from the "
                                                         "context"),
                return false);
    auto const inputLengthsShapeVal = inputLengthsShape->GetStorageShape();
    OP_CHECK_IF(inputLengthsShapeVal.GetDimNum() != LOSS_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "input_lengths",
                                             (std::to_string(inputLengthsShapeVal.GetDimNum()) + "D").c_str(), "1D"),
                return false);
    inputLengthsN = inputLengthsShapeVal.GetDim(BATCH_DIM);

    auto const targetLengthsShape = context->GetInputShape(INPUT_TARGET_LENGTHS_IDX);
    OP_CHECK_IF(targetLengthsShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "targetLengthsShape",
                                                         "failed to get the input shape of target_lengths from the "
                                                         "context"),
                return false);
    auto const targetLengthsShapeVal = targetLengthsShape->GetStorageShape();
    OP_CHECK_IF(targetLengthsShapeVal.GetDimNum() != LOSS_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "target_lengths",
                                             (std::to_string(targetLengthsShapeVal.GetDimNum()) + "D").c_str(), "1D"),
                return false);
    targetLengthsN = targetLengthsShapeVal.GetDim(BATCH_DIM);

    auto const lossShape = context->GetInputShape(INPUT_LOSS_IDX);
    OP_CHECK_IF(lossShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "lossShape",
                                                         "failed to get the input shape of target_lengths from the "
                                                         "context"),
                return false);
    auto const lossShapeVal = lossShape->GetStorageShape();
    OP_CHECK_IF(lossShapeVal.GetDimNum() != LOSS_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "neg_log_likelihood",
                                             (std::to_string(lossShapeVal.GetDimNum()) + "D").c_str(), "1D"),
                return false);
    lossN = lossShapeVal.GetDim(BATCH_DIM);
    return true;
}

bool CTCLossV3GradTiling::CheckShapeInfoForCT()
{
    // Check the shape information for some inputs and outputs, focusing mainly on symbolset and time.
    auto nodeName = context->GetNodeName();
    auto const logProbsShape = context->GetInputShape(INPUT_LOG_PROB_IDX);
    OP_CHECK_IF(logProbsShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "logProbsShape",
                                                         "failed to get the input shape of log_probs from the "
                                                         "context"),
                return false);
    auto const logProbsShapeVal = logProbsShape->GetStorageShape();
    OP_CHECK_IF(logProbsShapeVal.GetDimNum() != GRAD_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "log_probs",
                                             (std::to_string(logProbsShapeVal.GetDimNum()) + "D").c_str(), "3D"),
                return false);
    maxInputLength = logProbsShapeVal.GetDim(T_DIM);
    batchSize = logProbsShapeVal.GetDim(N_DIM);
    symbolSet = logProbsShapeVal.GetDim(C_DIM);

    auto const targetsShape = context->GetInputShape(INPUT_TRAGETS_IDX);
    OP_CHECK_IF(targetsShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "targetsShape",
                                                         "failed to get the input shape of targets from the context"),
                return false);
    auto const targetsShapeVal = targetsShape->GetStorageShape();
    targetsDimNum = targetsShapeVal.GetDimNum();
    OP_CHECK_IF(
        targetsDimNum != LOSS_DIM_NUM && targetsDimNum != TARGETS_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "targets", (std::to_string(targetsDimNum) + "D").c_str(), "1D or 2D"),
        return false);

    targetsN = targetsNum = targetsShapeVal.GetDim(BATCH_DIM);
    if (targetsDimNum > LOSS_DIM_NUM) {
        targetsNum = targetsNum * targetsShapeVal.GetDim(1);
        sDimRange = targetsShapeVal.GetDim(1);
    }

    auto const logAlphaShape = context->GetInputShape(INPUT_LOG_ALPHA_IDX);
    OP_CHECK_IF(logAlphaShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "logAlphaShape",
                                                         "failed to get the input shape of log_alpha from the "
                                                         "context"),
                return false);
    auto const logAlphaShapeVal = logAlphaShape->GetStorageShape();
    OP_CHECK_IF(logAlphaShapeVal.GetDimNum() != GRAD_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "log_alpha",
                                             (std::to_string(logAlphaShapeVal.GetDimNum()) + "D").c_str(), "3D"),
                return false);
    logAlphaN = logAlphaShapeVal.GetDim(BATCH_DIM);
    logAlphaT = logAlphaShapeVal.GetDim(ALPHA_T_DIM);
    alphaLength = logAlphaShapeVal.GetDim(ALPHA_DIM);
    maxTargetLength = (alphaLength - 1) / S_COE;

    auto const gradShape = context->GetOutputShape(0);
    OP_CHECK_IF(gradShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "gradShape",
                                                         "failed to get the output shape of grad from the context"),
                return false);
    auto const gradShapeVal = gradShape->GetStorageShape();
    OP_CHECK_IF(
        gradShapeVal.GetDimNum() != GRAD_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "grad", (std::to_string(gradShapeVal.GetDimNum()) + "D").c_str(), "3D"),
        return false);
    gradN = gradShapeVal.GetDim(N_DIM);
    gradT = gradShapeVal.GetDim(T_DIM);
    gradC = gradShapeVal.GetDim(C_DIM);
    return true;
}

bool CTCLossV3GradTiling::CheckShapeInfo()
{
    // Check the shape information for all the inputs and outputs.
    auto nodeName = context->GetNodeName();
    if (!CheckShapeInfoForN() || !CheckShapeInfoForCT()) {
        return false;
    }

    bool NCheck = gradOutN == batchSize && batchSize == inputLengthsN && inputLengthsN == targetLengthsN &&
                  targetLengthsN == lossN && lossN == logAlphaN && logAlphaN == gradN;
    NCheck = targetsDimNum > 1 ? (NCheck && (batchSize == targetsN)) : NCheck;
    OP_CHECK_IF(
        !NCheck,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            nodeName,
            "n-dim of grad_out, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, "
            "log_alpha and grad",
            "grad_out:" + std::to_string(gradOutN) + ", log_probs:" + std::to_string(batchSize) +
                ", targets:" + std::to_string(targetsN) + ", input_lengths:" + std::to_string(inputLengthsN) +
                ", target_lengths:" + std::to_string(targetLengthsN) + ", neg_log_likelihood:" + std::to_string(lossN) +
                ", log_alpha:" + std::to_string(logAlphaN) + ", grad:" + std::to_string(gradN),
            "the n-dim of grad_out, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, "
            "log_alpha and grad should be the same"),
        return false);

    OP_CHECK_IF(batchSize > MAX_BATCH,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "log_probs", std::to_string(batchSize),
                                                      "the n-dim of log_probs should be less than or equal to " +
                                                          std::to_string(MAX_BATCH) +
                                                          ", AICPU is recommended for larger batch size"),
                return false);

    bool TCheck = maxInputLength == gradT && gradT == logAlphaT;
    OP_CHECK_IF(!TCheck,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(nodeName, "t-dim of log_probs, grad and log_alpha",
                                                       "log_probs:" + std::to_string(maxInputLength) +
                                                           ", grad:" + std::to_string(gradT) +
                                                           ", log_alpha:" + std::to_string(logAlphaT),
                                                       "the t-dim of log_probs, grad and log_alpha should be the same"),
                return false);

    bool CCheck = symbolSet == gradC;
    OP_CHECK_IF(!CCheck,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    nodeName, "c-dim of log_probs and grad",
                    "log_probs:" + std::to_string(symbolSet) + ", grad:" + std::to_string(gradC),
                    "the c-dim of log_probs and grad should be the same"),
                return false);

    OP_CHECK_IF(symbolSet > MAX_SYMBOL_SET || maxTargetLength > MAX_LABEL_LEN,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    nodeName, "c-dim of log_probs and max target length of log_alpha",
                    std::to_string(symbolSet) + ", " + std::to_string(maxTargetLength),
                    "the c-dim of log_probs should be less than or equal to " + std::to_string(MAX_SYMBOL_SET) +
                        " and the max target length of log_alpha should be less than or equal to " +
                        std::to_string(MAX_LABEL_LEN) + ", AICPU is recommended for larger values"),
                return false);
    return true;
}

ge::graphStatus CTCLossV3GradTiling::Init()
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(context, "CTCLossV3Grad tiling starts running");
    auto compileInfo = static_cast<const CTCLossV3GradCompileInfo*>(context->GetCompileInfo());

    uint32_t coreNum = compileInfo->coreNum;
    int64_t ubSize = compileInfo->ubSizePlatForm - RESERVE_SAPCE - SELECT_SPACE;
    if (!CheckShapeInfo()) {
        return ge::GRAPH_FAILED;
    }
    auto targetDtype = context->GetInputDesc(INPUT_TRAGETS_IDX)->GetDataType();
    auto gradDtype = context->GetInputDesc(0)->GetDataType();
    targetDsize = GetSizeByDataType(targetDtype);
    gradDsize = GetSizeByDataType(gradDtype);

    auto* attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "attrs", "failed to get the attrs from the context"),
                return ge::GRAPH_FAILED);
    const auto* blankPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_IF(blankPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "blankPtr",
                                                         "failed to get the attr pointer of blank from the attrs"),
                return ge::GRAPH_FAILED);
    BLANK = *blankPtr;
    OP_CHECK_IF(
        BLANK < 0 || BLANK >= symbolSet,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "blank", std::to_string(BLANK),
                                              "blank should be in the range [0, " + std::to_string(symbolSet) + ")"),
        return false);
    const auto* zeroInfinityPtr = attrs->GetAttrPointer<int64_t>(2);
    OP_CHECK_IF(zeroInfinityPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "zeroInfinityPtr",
                                                         "failed to get the attr pointer of zero_infinity from the "
                                                         "attrs"),
                return ge::GRAPH_FAILED);
    zeroInfinity = *zeroInfinityPtr;
    coreUsed = batchSize > coreNum ? coreNum : batchSize;
    taskPerCore = batchSize / coreNum;
    taskTailCore = batchSize % coreNum;

    int64_t alphaLengthAlign = Ops::Base::CeilAlign(alphaLength, NUM_PER_REPEAT);
    int64_t maxTargetLengthAlign = Ops::Base::CeilAlign(maxTargetLength, NUM_PER_REPEAT);
    int64_t targetLengthPlusAlign = Ops::Base::CeilAlign(maxTargetLength + 1, NUM_PER_REPEAT);
    // Calculate the ubsize needed for targetsLengths and alphaLength
    int64_t needSize = alphaLength * FLOAT_DSIZE + alphaLengthAlign * FLOAT_DSIZE * ALPHA_LENGTH_NUM +
                       Ops::Base::CeilAlign(alphaLengthAlign / MASKNUM_PER_BYTES, BLOCK_SIZE) * MASK_UB_NUM +
                       targetLengthPlusAlign * INT64_DSIZE * TARGETS_UB_NUM +
                       maxTargetLengthAlign * (FLOAT_DSIZE * TARGETS_LENGTH_UB_NUM + INT64_DSIZE) +
                       Ops::Base::CeilAlign(maxTargetLengthAlign / MASKNUM_PER_BYTES, BLOCK_SIZE) *
                           TARGETS_LENGTH_MASK_UB_NUM;
    needSize = batchSize <= LARGE_BATCH ? (needSize + batchSize * (targetDsize + INT64_DSIZE)) : needSize;
    sliceLength = (ubSize - needSize) / INT64_DSIZE; // NeedSize is smaller than ubSize
    sliceLength = sliceLength > symbolSet ? symbolSet : sliceLength;
    sliceLength = Ops::Base::CeilAlign(sliceLength, NUM_PER_REPEAT);
    probSliceNum = sliceLength == 0 ? 0 : (symbolSet - 1 + sliceLength) / sliceLength;
    sliceLengthTail = symbolSet - (probSliceNum - 1) * sliceLength;
    size_t sysWorkspaceSizeNeed = compileInfo->sysWorkspaceSize;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = batchSize > LARGE_BATCH ?
                              (sysWorkspaceSizeNeed + alphaLength * FLOAT_DSIZE * WORKSPACE_ALPHA_NUM * batchSize +
                               FLOAT_DSIZE * batchSize * TARGETS_GM_NUM) :
                              (sysWorkspaceSizeNeed + alphaLength * FLOAT_DSIZE * WORKSPACE_ALPHA_NUM * batchSize);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CTCLossV3GradTiling::RunKernelTiling()
{
    context->SetTilingKey(0); // only one tilingKey now
    context->SetBlockDim(coreUsed);
    tilingData.set_sliceLength(sliceLength);
    tilingData.set_sliceLengthTail(sliceLengthTail);
    tilingData.set_probSliceNum(probSliceNum);
    tilingData.set_alphaLength(alphaLength);
    tilingData.set_maxInputLength(maxInputLength);
    tilingData.set_symbolSet(symbolSet);
    tilingData.set_batchSize(batchSize);
    tilingData.set_sDimRange(sDimRange);
    tilingData.set_targetsDimNum(targetsDimNum);
    tilingData.set_targetsNum(targetsNum);
    tilingData.set_taskPerCore(taskPerCore);
    tilingData.set_taskTailCore(taskTailCore);
    tilingData.set_BLANK(BLANK);
    tilingData.set_zeroInfinity(zeroInfinity);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void CTCLossV3GradTiling::PrintTilingData() const
{
    OP_LOGD(context, "Start printing");
    OP_LOGD(context, "sliceLength is %ld.", sliceLength);
    OP_LOGD(context, "sliceLengthTail is %ld.", sliceLengthTail);
    OP_LOGD(context, "probSliceNum is %ld.", probSliceNum);
    OP_LOGD(context, "alphaLength is %ld.", alphaLength);
    OP_LOGD(context, "maxInputLength is %ld.", maxInputLength);
    OP_LOGD(context, "symbolSet is %ld.", symbolSet);
    OP_LOGD(context, "batchSize is %ld.", batchSize);
    OP_LOGD(context, "targetsDimNum is %ld.", targetsDimNum);
    OP_LOGD(context, "sDimRange is %ld.", sDimRange);
    OP_LOGD(context, "targetsNum is %ld.", targetsNum);
    OP_LOGD(context, "taskPerCore is %ld.", taskPerCore);
    OP_LOGD(context, "taskTailCore is %ld.", taskTailCore);
    OP_LOGD(context, "BLANK is %ld.", BLANK);
    OP_LOGD(context, "zeroInfinity is %ld.", zeroInfinity);
    OP_LOGD(context, "End printing");
    OP_LOGD(context, "CTCLossV3Grad tiling end running");
}

static ge::graphStatus TilingFunc4CTCLossV3Grad(gert::TilingContext* context)
{
    CTCLossV3GradTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepare4CTCLossV3Grad(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        return ReportTilingFailure("CtcLossV3Grad", "context is nullptr.");
    }
    auto nodeName = context->GetNodeName();
    OP_LOGD(context, "TilingPrepare4CTCLossV3Grad start.");
    auto compileInfo = context->GetCompiledInfo<CTCLossV3GradCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "compileInfo",
                                                         "failed to get the compiled info from the context"),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "platformInfo",
                                                         "failed to get the platform info from the context"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), ReportTilingFailure(nodeName, "Failed to get core num."),
                return ge::GRAPH_FAILED);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF((compileInfo->sysWorkspaceSize <= 0U),
                ReportTilingFailure(nodeName, "sysWorkspaceSize should be greater than zero"), return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0), ReportTilingFailure(nodeName, "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "TilingPrepare4CTCLossV3Grad end.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CTCLossV3Grad)
    .Tiling(TilingFunc4CTCLossV3Grad)
    .TilingParse<CTCLossV3GradCompileInfo>(TilingPrepare4CTCLossV3Grad);
} // namespace optiling
