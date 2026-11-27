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
 * \file rotate_quant_tiling.cpp
 * \brief Arch35 tiling implementation of the RotateQuant operator.
 */
#include "../../../../common/inc/op_host/op_const_def_nn.h"
#include "rotate_quant_tiling.h"
#include <algorithm>
#include <array>
#include <limits>
#include "error_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "matmul/common/op_host/log_format_util.h"
#include "util/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"
#include "platform/soc_spec.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_key.h"
#include "op_host/tiling_templates_registry.h"
#include "op_common/op_host/util/op_const_def.h"

namespace {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t ROT_INDEX = 1;
constexpr uint32_t ALPHA_INDEX = 2;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t SCALE_INDEX = 1;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t MAX_X_DIM_NUM = 7;
constexpr uint32_t MIN_X_DIM_NUM = 1;
constexpr uint32_t MIN_ROT_DIM_NUM = 2;
constexpr uint32_t MAX_ROT_DIM_NUM = 3;
constexpr int64_t MIN_GROUP_NUM = 1;
constexpr int64_t MAX_GROUP_NUM = 1024;
constexpr uint32_t MIN_M = 16;
constexpr uint32_t INIT_ACCUMULATE_VALUE = 1;
constexpr uint32_t EPILOGUE_ELEMENTS = 16384;
constexpr int64_t MX_SCALE_LAST_SIZE = 2;
constexpr int64_t HIF4_SCALE_LAST_DIM = 1;
constexpr int64_t MX_SCALE_CEIL_NUM = 64;
constexpr int64_t SYS_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
constexpr uint64_t DEFAULT_TILING_KEY = 0;
constexpr uint64_t GROUPED_TILING_KEY = 1;
constexpr uint32_t WORKSPACE_NUM = 1;
constexpr uint32_t AIC_AIV_RATIO_950 = 2;

constexpr uint32_t YDTYPE_ATTR_INDEX = 0;
constexpr uint32_t AXIS_ATTR_INDEX = 1;
constexpr uint32_t ROUND_MODE_ATTR_INDEX = 2;
constexpr uint32_t SCALE_ALG_ATTR_INDEX = 3;
constexpr uint32_t DST_TYPE_MAX_ATTR_INDEX = 4;
constexpr uint32_t TRANS_ATTR_INDEX = 5;

constexpr int32_t MIN_SCALE_ALG = 0;
constexpr int32_t MAX_SCALE_ALG = 2;
constexpr int32_t SCALE_ALG_PER_CHANNEL = 1;
constexpr float DST_TYPE_MAX_LOWER = 6.0f;
constexpr float DST_TYPE_MAX_UPPER = 12.0f;
constexpr float DST_TYPE_MAX_DEFAULT = 0.0f;
constexpr float INVERSE_UNIT = 1.0f;
constexpr int32_t DEFAULT_AXIS = -1;

constexpr int64_t ROT_K_SIZE_32 = 32;
constexpr int64_t ROT_K_SIZE_64 = 64;
constexpr int64_t ROT_K_SIZE_128 = 128;
constexpr int64_t ROT_K_SIZE_256 = 256;
constexpr std::array<int64_t, 4> SUPPORTED_ROT_K_SIZES = {ROT_K_SIZE_32, ROT_K_SIZE_64, ROT_K_SIZE_128, ROT_K_SIZE_256};

constexpr uint32_t ALPHA_DIM_NUM = 1;
constexpr uint32_t ALPHA_DIM_SIZE = 1;

constexpr uint64_t ROT_TYPE_MULTI = 0;
constexpr uint64_t ROT_TYPE_SINGLE = 1;

constexpr uint32_t ELEMENT_BYTES_16BIT = 2;
constexpr uint32_t ELEMENT_BYTES_32BIT = 4;
constexpr uint32_t MX_GROUP_SIZE = 32;
constexpr uint32_t MX_SCALE_ROW_BYTES = 32;
// Match the fixed BF16 input, FP8-sized output and MX scratch regions in the epilogue.
constexpr uint32_t MX_SCALE_ELEMENTS = EPILOGUE_ELEMENTS / MX_GROUP_SIZE;
constexpr uint32_t MX_FIXED_UB_BYTES = EPILOGUE_ELEMENTS * (ELEMENT_BYTES_16BIT + sizeof(int8_t)) +
                                       MX_SCALE_ELEMENTS * (3 * ELEMENT_BYTES_16BIT + sizeof(int8_t));
constexpr uint32_t SINGLE_ROTATION_B_SIZE = 1;
constexpr uint32_t SCHEDULE_MODE_STATIC = 1;
constexpr int32_t TEMPLATE_PRIORITY = 0;

enum class RoundModeList {
    MODE_ROUND = 0,
    MODE_FLOOR = 1,
    MODE_CEIL = 2,
    MODE_TRUNC = 3,
    MODE_RINT = 4,
    MODE_HYBRID = 5,
    MODE_UNDEFINED = -1,
};

RoundModeList GetRoundMode(const std::string& roundMode)
{
    if (roundMode == "rint") {
        return RoundModeList::MODE_RINT;
    } else if (roundMode == "round") {
        return RoundModeList::MODE_ROUND;
    } else if (roundMode == "floor") {
        return RoundModeList::MODE_FLOOR;
    }
    return RoundModeList::MODE_UNDEFINED;
}
} // namespace

namespace Ops {
namespace NN {
namespace RotateQuant {

namespace {
constexpr bool IsDav920Series(NpuArch npuArch)
{
    return npuArch == Ops::Base::DAV_9201 || npuArch == Ops::Base::DAV_9202;
}
} // namespace

ge::graphStatus RotateQuantAptTiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

bool RotateQuantAptTiling::IsCapable()
{
    if (compileInfo_.aicNum == 0) {
        OP_LOGE(context_->GetNodeName(), "RotateQuant is only supported for aicNum:aivNum=1:2, aicNum=%lu, aivNum=%lu",
                compileInfo_.aicNum, compileInfo_.aivNum);
        return false;
    }
    // DAV_9201/DAV_9202: AIC:AIV = 1:1
    if (IsDav920Series(compileInfo_.npuArch)) {
        if (compileInfo_.aivNum != compileInfo_.aicNum) {
            OP_LOGE(context_->GetNodeName(),
                    "RotateQuant is only supported for aicNum:aivNum=1:1 on DAV_9201/DAV_9202, aicNum=%lu, aivNum=%lu",
                    compileInfo_.aicNum, compileInfo_.aivNum);
            return false;
        }
        return true;
    }
    if (compileInfo_.aivNum != (compileInfo_.aicNum * AIC_AIV_RATIO_950)) {
        OP_LOGE(context_->GetNodeName(), "RotateQuant is only supported for aicNum:aivNum=1:2, aicNum=%lu, aivNum=%lu",
                compileInfo_.aicNum, compileInfo_.aivNum);
        return false;
    }
    return true;
}

void RotateQuantAptTiling::InitCompileInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("RotateQuantApt", "context_ is null");
        return;
    }
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }

    const auto& ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo_.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo_.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo_.l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo_.l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo_.npuArch = ascendcPlatform.GetCurNpuArch();

    if (compileInfo_.aicNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "aicNum",
                                              std::to_string(compileInfo_.aicNum).c_str(),
                                              "the value of aicNum must be greater than 0");
        return;
    }
    OP_LOGI("RotateQuantApt", "parse compile info success aicNum:%lu, aivNum:%lu, npuArch:%d", compileInfo_.aicNum,
            compileInfo_.aivNum, static_cast<int>(compileInfo_.npuArch));
    OP_LOGD(context_->GetNodeName(), "RotateQuantApt InitCompileInfo Success");
}

ge::graphStatus RotateQuantAptTiling::GetShapeAttrsInfo()
{
    tilingDataSize_ = sizeof(RotateQuantAptOpt::RotateQuantAptTilingData);
    if (inputParams_.initFlag) {
        OP_LOGD(inputParams_.opName, "No need to get shape and attrs from tiling context again");
        return ge::GRAPH_SUCCESS;
    }

    inputParams_.opName = context_->GetNodeName();
    OP_LOGI(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(
        CheckContext() != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(inputParams_.opName, "context", "the tiling context is invalid"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(AnalyzeAttrs() != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(inputParams_.opName, "attrs", "the attrs are invalid"),
                    return ge::GRAPH_FAILED);
    // Defensive value access for failure diagnostics: null-check the desc/shape pointers first (print "null" when
    // absent) instead of relying on the implicit coupling with CheckContext's preceding null check, so the error
    // path never dereferences possibly-null context data and crashes instead of returning a failure.
    auto dtypeOf = [this](bool isInput, uint32_t idx) -> std::string {
        const auto* desc = isInput ? context_->GetInputDesc(idx) : context_->GetOutputDesc(idx);
        return desc == nullptr ? std::string("null") : ge::TypeUtils::DataTypeToSerialString(desc->GetDataType());
    };
    auto shapeOf = [this](bool isInput, uint32_t idx) -> std::string {
        const auto* shape = isInput ? context_->GetInputShape(idx) : context_->GetOutputShape(idx);
        return shape == nullptr ? std::string("null") : Shape2String(shape->GetStorageShape());
    };
    OP_TILING_CHECK(
        AnalyzeDtype() != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            inputParams_.opName, "x, rotation, y, scale",
            Ops::NN::FormatString("%s, %s, %s, %s", dtypeOf(true, X_INDEX).c_str(), dtypeOf(true, ROT_INDEX).c_str(),
                                  dtypeOf(false, Y_INDEX).c_str(), dtypeOf(false, SCALE_INDEX).c_str())
                .c_str(),
            "the dtypes of x, rotation, y and scale are invalid"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(AnalyzeGroups() != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(inputParams_.opName, "groups", "the groups are invalid"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        AnalyzeShapes() != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            inputParams_.opName, "x, rotation, y, scale",
            Ops::NN::FormatString("%s, %s, %s, %s", shapeOf(true, X_INDEX).c_str(), shapeOf(true, ROT_INDEX).c_str(),
                                  shapeOf(false, Y_INDEX).c_str(), shapeOf(false, SCALE_INDEX).c_str())
                .c_str(),
            "the shapes of x, rotation, y and scale are invalid"),
        return ge::GRAPH_FAILED);

    OP_LOGD(inputParams_.opName, "Input params: MNK[%ld, %ld, %ld], nKRatio[%ld].", inputParams_.M, inputParams_.N,
            inputParams_.K, inputParams_.nKRatio);

    inputParams_.initFlag = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::CheckContext()
{
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(X_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(ROT_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ROT_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(Y_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(Y_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(SCALE_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(SCALE_INDEX));
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "context tiling data capacity %zu < actual tiling data size %zu.",
                              context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::ParseAttrValues()
{
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const int64_t* yDtypePtr = attrs->GetAttrPointer<int64_t>(YDTYPE_ATTR_INDEX);
    if (yDtypePtr != nullptr) {
        inputParams_.yDtype = static_cast<ge::DataType>(*yDtypePtr);
    }

    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(AXIS_ATTR_INDEX);
    if (axisPtr != nullptr) {
        inputParams_.axis = static_cast<int32_t>(*axisPtr);
    }

    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<char>(ROUND_MODE_ATTR_INDEX));
    std::string roundModeStr = attrs->GetAttrPointer<char>(ROUND_MODE_ATTR_INDEX);
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    OP_CHECK_IF((roundMode == RoundModeList::MODE_UNDEFINED),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "round_mode", roundModeStr.c_str(),
                                                      "the value of round_mode must be one of {rint, floor, round}"),
                return ge::GRAPH_FAILED);
    inputParams_.roundMode = static_cast<int32_t>(roundMode);

    const int64_t* scaleAlgPtr = attrs->GetAttrPointer<int64_t>(SCALE_ALG_ATTR_INDEX);
    if (scaleAlgPtr != nullptr) {
        inputParams_.scaleAlg = static_cast<int32_t>(*scaleAlgPtr);
        OP_TILING_CHECK(
            inputParams_.scaleAlg < MIN_SCALE_ALG || inputParams_.scaleAlg > MAX_SCALE_ALG,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                inputParams_.opName, "scaleAlg", std::to_string(inputParams_.scaleAlg).c_str(),
                Ops::NN::FormatString("the value of scaleAlg must be in [%d, %d]", MIN_SCALE_ALG, MAX_SCALE_ALG)
                    .c_str()),
            return ge::GRAPH_FAILED);
    }

    tilingData_.invDstTypeMax = DST_TYPE_MAX_DEFAULT;
    const float* dstTypeMaxPtr = attrs->GetAttrPointer<float>(DST_TYPE_MAX_ATTR_INDEX);
    if (dstTypeMaxPtr != nullptr) {
        inputParams_.dstTypeMax = *dstTypeMaxPtr;
    }

    const bool* transPtr = attrs->GetAttrPointer<bool>(TRANS_ATTR_INDEX);
    if (transPtr != nullptr) {
        inputParams_.trans = *transPtr;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::AnalyzeAttrs()
{
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGD(inputParams_.opName, "No attrs provided, using default values");
        return ge::GRAPH_SUCCESS;
    }

    OP_TILING_CHECK(ParseAttrValues() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Failed to parse attrs."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.trans == true,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "trans", "true", "the value of trans must be false"),
        return ge::GRAPH_FAILED);

    const auto& xShapeForAxis = context_->GetInputShape(X_INDEX)->GetStorageShape();
    OP_TILING_CHECK(
        inputParams_.axis != DEFAULT_AXIS && inputParams_.axis != static_cast<int32_t>(xShapeForAxis.GetDimNum() - 1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "axis", std::to_string(inputParams_.axis).c_str(),
                                              Ops::NN::FormatString("the value of axis must be %d or %d", DEFAULT_AXIS,
                                                                    static_cast<int32_t>(xShapeForAxis.GetDimNum() - 1))
                                                  .c_str()),
        return ge::GRAPH_FAILED);

    OP_LOGD(inputParams_.opName, "Attrs: yDtype=%d, axis=%d, roundMode=%d, scaleAlg=%d, dstTypeMax=%f, trans=%d",
            inputParams_.yDtype, inputParams_.axis, inputParams_.roundMode, inputParams_.scaleAlg,
            inputParams_.dstTypeMax, inputParams_.trans);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::CheckAlphaInput()
{
    auto computeNodeInfo = context_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, computeNodeInfo);
    inputParams_.hasAlpha = false;
    auto alphaDesc = context_->GetOptionalInputDesc(ALPHA_INDEX);
    if (alphaDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    // The invalid party is y's output dtype (hif4 does not support alpha input), so the log attributes the error
    // to y rather than alpha.
    OP_TILING_CHECK(inputParams_.yDtype == Ops::NN::DT_HIFLOAT4,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "y", ge::TypeUtils::DataTypeToSerialString(inputParams_.yDtype).c_str(),
                        "alpha is only supported for mxfp4/mxfp8 output, not for hif4"),
                    return ge::GRAPH_FAILED);
    auto alphaDtype = alphaDesc->GetDataType();
    OP_TILING_CHECK(alphaDtype != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "alpha",
                                                          ge::TypeUtils::DataTypeToSerialString(alphaDtype).c_str(),
                                                          "the dtype of alpha must be bf16"),
                    return ge::GRAPH_FAILED);
    auto alphaShape = context_->GetOptionalInputShape(ALPHA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, alphaShape);
    auto alphaStorageShape = alphaShape->GetStorageShape();
    OP_TILING_CHECK(
        alphaStorageShape.GetDimNum() != ALPHA_DIM_NUM || alphaStorageShape.GetDim(0) != ALPHA_DIM_SIZE,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "alpha", Shape2String(alphaStorageShape).c_str(),
                                              "the shape of alpha must be (1,)"),
        return ge::GRAPH_FAILED);
    inputParams_.hasAlpha = true;
    OP_LOGD(inputParams_.opName, "hasAlpha detected");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::AnalyzeDtype()
{
    inputParams_.dataType = context_->GetInputDesc(X_INDEX)->GetDataType();
    auto rotDtype = context_->GetInputDesc(ROT_INDEX)->GetDataType();
    auto yDtype = context_->GetOutputDesc(Y_INDEX)->GetDataType();
    auto scaleDtype = context_->GetOutputDesc(SCALE_INDEX)->GetDataType();

    OP_TILING_CHECK(inputParams_.dataType != ge::DT_FLOAT16 && inputParams_.dataType != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "x", ge::TypeUtils::DataTypeToSerialString(inputParams_.dataType).c_str(),
                        "the dtype of x must be float16 or bfloat16"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        inputParams_.dataType != rotDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            inputParams_.opName, "x, rotation",
            Ops::NN::FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.dataType).c_str(),
                                  ge::TypeUtils::DataTypeToSerialString(rotDtype).c_str())
                .c_str(),
            "the dtypes of x and rotation must be the same"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(yDtype != ge::DT_FLOAT4_E2M1 && yDtype != ge::DT_FLOAT8_E4M3FN && yDtype != ge::DT_FLOAT8_E5M2 &&
                        yDtype != Ops::NN::DT_HIFLOAT4,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "y", ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                        "the dtype of y must be one of [FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT4]"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yDtype == Ops::NN::DT_HIFLOAT4 && !IsDav920Series(compileInfo_.npuArch),
                    OP_LOGE(inputParams_.opName, "HIFLOAT4 output is only supported on DAV_9201/DAV_9202."),
                    return ge::GRAPH_FAILED);
    if (yDtype == ge::DT_FLOAT4_E2M1) {
        OP_TILING_CHECK(inputParams_.scaleAlg != MIN_SCALE_ALG && inputParams_.scaleAlg != MAX_SCALE_ALG,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "y", ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                            Ops::NN::FormatString("when the dtype of y is FLOAT4_E2M1, the value of scaleAlg must be "
                                                  "%d or %d",
                                                  MIN_SCALE_ALG, MAX_SCALE_ALG)
                                .c_str()),
                        return ge::GRAPH_FAILED);
    } else if (yDtype == Ops::NN::DT_HIFLOAT4) {
        OP_TILING_CHECK(inputParams_.scaleAlg != MIN_SCALE_ALG,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "y", ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                            Ops::NN::FormatString("when the dtype of y is HIFLOAT4, the value of scaleAlg must be %d",
                                                  MIN_SCALE_ALG)
                                .c_str()),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(inputParams_.roundMode != static_cast<int32_t>(RoundModeList::MODE_RINT),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "round_mode", std::to_string(inputParams_.roundMode).c_str(),
                            "when the dtype of y is HIFLOAT4, the value of round_mode must be rint"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(inputParams_.roundMode != static_cast<int32_t>(RoundModeList::MODE_RINT),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "round_mode", std::to_string(inputParams_.roundMode).c_str(),
                            "when the dtype of y is FLOAT8_E4M3/FLOAT8_E5M2, the value of round_mode must be rint"),
                        return ge::GRAPH_FAILED);

        OP_TILING_CHECK(
            inputParams_.scaleAlg != MIN_SCALE_ALG && inputParams_.scaleAlg != SCALE_ALG_PER_CHANNEL,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "y", ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                Ops::NN::FormatString("when the dtype of y is FLOAT8_E4M3/FLOAT8_E5M2, the value of scaleAlg must "
                                      "be %d or %d",
                                      MIN_SCALE_ALG, SCALE_ALG_PER_CHANNEL)
                    .c_str()),
            return ge::GRAPH_FAILED);
    }

    if (yDtype == ge::DT_FLOAT4_E2M1 && inputParams_.scaleAlg == MAX_SCALE_ALG) {
        OP_TILING_CHECK(inputParams_.dstTypeMax < DST_TYPE_MAX_LOWER || inputParams_.dstTypeMax > DST_TYPE_MAX_UPPER,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "y", ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                            Ops::NN::FormatString("when the dtype of y is FLOAT4_E2M1 and the value of scaleAlg is %d, "
                                                  "the value of dstTypeMax must be in [%f, %f]",
                                                  MAX_SCALE_ALG, DST_TYPE_MAX_LOWER, DST_TYPE_MAX_UPPER)
                                .c_str()),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            inputParams_.dstTypeMax != DST_TYPE_MAX_DEFAULT,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                inputParams_.opName, "dstTypeMax", std::to_string(inputParams_.dstTypeMax).c_str(),
                Ops::NN::FormatString("the value of dstTypeMax must equal %f", DST_TYPE_MAX_DEFAULT).c_str()),
            return ge::GRAPH_FAILED);
    }

    if (yDtype == Ops::NN::DT_HIFLOAT4) {
        OP_TILING_CHECK(scaleDtype != Ops::NN::DT_HIFLOAT4_SCALE,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "scale", ge::TypeUtils::DataTypeToSerialString(scaleDtype).c_str(),
                            "the dtype of scale must be HIFLOAT4_SCALE for hif4 output"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(scaleDtype != ge::DT_FLOAT8_E8M0,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "scale",
                                                              ge::TypeUtils::DataTypeToSerialString(scaleDtype).c_str(),
                                                              "the dtype of scale must be FLOAT8_E8M0 for dav3510"),
                        return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(
        CheckAlphaInput() != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "alpha", "invalid", "the alpha input is invalid"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::ValidateRotShape(const gert::Shape& rotShape)
{
    const size_t rotRank = rotShape.GetDimNum();
    const size_t minRotRank = inputParams_.groupMode ? 3 : MIN_ROT_DIM_NUM;
    const size_t maxRotRank = inputParams_.groupMode ? 4 : MAX_ROT_DIM_NUM;
    OP_TILING_CHECK(
        rotRank < minRotRank || rotRank > maxRotRank,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            inputParams_.opName, "rotation", (std::to_string(rotRank) + "D").c_str(),
            Ops::NN::FormatString("the shape dim of rotation must be in [%zu, %zu]", minRotRank, maxRotRank).c_str()),
        return ge::GRAPH_FAILED);

    inputParams_.K = rotShape.GetDim(rotShape.GetDimNum() - 1);
    if (inputParams_.yDtype == Ops::NN::DT_HIFLOAT4) {
        OP_TILING_CHECK(rotShape.GetDimNum() != MIN_ROT_DIM_NUM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            inputParams_.opName, "rotation", (std::to_string(rotShape.GetDimNum()) + "D").c_str(),
                            "for hif4 output, the shape dim of rotation must be 2"),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(inputParams_.K != ROT_K_SIZE_64 && inputParams_.K != ROT_K_SIZE_128,
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                            inputParams_.opName, "rotation", std::to_string(inputParams_.K).c_str(),
                            "for hif4 output, the last dim size of rotation must be 64 or 128"),
                        return ge::GRAPH_FAILED);
    }
    auto platformInfo = context_->GetPlatformInfo();
    const bool isAscend950 = platformInfo != nullptr &&
                             platform_ascendc::PlatformAscendC(platformInfo).GetSocVersion() ==
                                 platform_ascendc::SocVersion::ASCEND950;
    const bool isSupportedK = isAscend950 ? std::find(SUPPORTED_ROT_K_SIZES.begin(), SUPPORTED_ROT_K_SIZES.end(),
                                                      inputParams_.K) != SUPPORTED_ROT_K_SIZES.end() :
                                            (inputParams_.K == ROT_K_SIZE_32 || inputParams_.K == ROT_K_SIZE_64 ||
                                             inputParams_.K == ROT_K_SIZE_128 || inputParams_.K == ROT_K_SIZE_256);
    OP_TILING_CHECK(!isSupportedK,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        inputParams_.opName, "rotation", std::to_string(inputParams_.K).c_str(),
                        "the last dim size of rotation must be one of [32, 64, 128, 256]"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(inputParams_.N % inputParams_.K != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "N",
                        (std::to_string(inputParams_.N) + " (K=" + std::to_string(inputParams_.K) + ")").c_str(),
                        "the value of N must be divisible by K"),
                    return ge::GRAPH_FAILED);

    inputParams_.nKRatio = inputParams_.N / inputParams_.K;

    if (inputParams_.groupMode) {
        OP_TILING_CHECK(rotShape.GetDim(rotRank - 2) != inputParams_.K ||
                            (rotRank == 4 && rotShape.GetDim(1) != inputParams_.nKRatio),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            inputParams_.opName, "rotation", Shape2String(rotShape).c_str(),
                            "the shape of rotation must be [E,K,K] or [E,N/K,K,K] in grouped mode"),
                        return ge::GRAPH_FAILED);
        inputParams_.rotType = (rotRank == 4 && inputParams_.nKRatio > 1) ? ROT_TYPE_MULTI : ROT_TYPE_SINGLE;
        return ge::GRAPH_SUCCESS;
    }

    if (rotShape.GetDimNum() == MIN_ROT_DIM_NUM) {
        inputParams_.rotType = ROT_TYPE_SINGLE;
        OP_TILING_CHECK(
            rotShape.GetDim(0) != inputParams_.K,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                inputParams_.opName, "rotation", Shape2String(rotShape).c_str(),
                Ops::NN::FormatString("the shape of rotation must be (%ld, %ld)", inputParams_.K, inputParams_.K)
                    .c_str()),
            return ge::GRAPH_FAILED);
    } else {
        inputParams_.rotType = (inputParams_.nKRatio > 1) ? ROT_TYPE_MULTI : ROT_TYPE_SINGLE;
        OP_TILING_CHECK(rotShape.GetDim(0) != inputParams_.nKRatio || rotShape.GetDim(1) != inputParams_.K,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            inputParams_.opName, "rotation", Shape2String(rotShape).c_str(),
                            Ops::NN::FormatString("the shape of rotation must be (%ld, %ld, %ld)", inputParams_.nKRatio,
                                                  inputParams_.K, inputParams_.K)
                                .c_str()),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::ValidateScaleShape(const gert::Shape& mxscaleShape, const gert::Shape& scaleShape)
{
    // The kernel writes the scale output by flat offset and only requires matching total element counts, so callers
    // may pass a flat view with an equal total (e.g. 1D [M*CeilDiv(N,64)*2]).
    OP_CHECK_IF(mxscaleShape.GetShapeSize() != scaleShape.GetShapeSize(),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    inputParams_.opName, "scale", std::to_string(scaleShape.GetShapeSize()).c_str(),
                    Ops::NN::FormatString("the element size of scale must be %ld (derived shape %s)",
                                          mxscaleShape.GetShapeSize(), Shape2String(mxscaleShape).c_str())
                        .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::AnalyzeGroups()
{
    constexpr size_t GROUP_INDEX = 3;
    constexpr size_t GROUP_TYPE_INDEX = 6;
    auto nodeInfo = context_->GetComputeNodeInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, nodeInfo);
    if (context_->GetOptionalInputDesc(GROUP_INDEX) == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto platformInfo = context_->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    const auto& platform = platform_ascendc::PlatformAscendC(platformInfo);
    OP_TILING_CHECK(
        platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND950 && !IsDav920Series(compileInfo_.npuArch),
        OP_LOGE(inputParams_.opName, "Grouped RotateQuant is only supported on Ascend950, DAV_9201 and DAV_9202."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(inputParams_.yDtype == Ops::NN::DT_HIFLOAT4,
                    OP_LOGE(inputParams_.opName, "Grouped RotateQuant requires MX FP4/FP8 output."),
                    return ge::GRAPH_FAILED);
    auto groups = context_->GetOptionalInputShape(GROUP_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, groups);
    const auto& shape = groups->GetStorageShape();
    const auto& xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    const auto& rotShape = context_->GetInputShape(ROT_INDEX)->GetStorageShape();
    OP_TILING_CHECK(
        context_->GetOptionalInputDesc(GROUP_INDEX)->GetDataType() != ge::DT_INT64 || shape.GetDimNum() != 1 ||
            xShape.GetDimNum() != 2 || (rotShape.GetDimNum() != 3 && rotShape.GetDimNum() != 4),
        OP_LOGE(inputParams_.opName, "Grouped mode requires int64[E], x[M,N], rotation[E,K,K] or [E,N/K,K,K]."),
        return ge::GRAPH_FAILED);
    int64_t groupsNum = shape.GetDim(0);
    OP_TILING_CHECK(groupsNum < MIN_GROUP_NUM || groupsNum > MAX_GROUP_NUM || rotShape.GetDim(0) != groupsNum,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opName, "groups", Shape2String(shape).c_str(),
                        Ops::NN::FormatString("the group count must be in [%ld, %ld] and equal rotation dim0",
                                              MIN_GROUP_NUM, MAX_GROUP_NUM)
                            .c_str()),
                    return ge::GRAPH_FAILED);
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > GROUP_TYPE_INDEX) {
        auto type = attrs->GetAttrPointer<int64_t>(GROUP_TYPE_INDEX);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, type);
        inputParams_.groupListType = *type;
    }
    OP_TILING_CHECK(inputParams_.groupListType != 0 && inputParams_.groupListType != 1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "group_list_type",
                                                          std::to_string(inputParams_.groupListType).c_str(),
                                                          "the value of group_list_type must be 0 or 1"),
                    return ge::GRAPH_FAILED);
    inputParams_.groupMode = true;
    inputParams_.groupNum = static_cast<uint32_t>(groupsNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::AnalyzeShapes()
{
    const auto& xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    const auto& rotShape = context_->GetInputShape(ROT_INDEX)->GetStorageShape();

    OP_TILING_CHECK(
        xShape.GetDimNum() < MIN_X_DIM_NUM || xShape.GetDimNum() > MAX_X_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            inputParams_.opName, "x", (std::to_string(xShape.GetDimNum()) + "D").c_str(),
            Ops::NN::FormatString("the shape dim of x must be in [%d, %d]", MIN_X_DIM_NUM, MAX_X_DIM_NUM).c_str()),
        return ge::GRAPH_FAILED);

    // These bounds also keep M*N, folded M*(N/K) and grouped rotation offsets
    // within int64_t: E <= 1024 and K <= 256 are validated by AnalyzeGroups/ValidateRotShape.
    constexpr int64_t maxShapeDim = std::numeric_limits<int32_t>::max();
    inputParams_.M = INIT_ACCUMULATE_VALUE;
    for (size_t i = 0; i < xShape.GetDimNum() - 1; i++) {
        const int64_t dim = xShape.GetDim(i);
        OP_TILING_CHECK(dim <= 0 || inputParams_.M > maxShapeDim / dim,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            inputParams_.opName, "x", Shape2String(xShape).c_str(),
                            Ops::NN::FormatString("the flattened M of x must be in [1, %ld]", maxShapeDim).c_str()),
                        return ge::GRAPH_FAILED);
        inputParams_.M *= dim;
    }
    inputParams_.N = xShape.GetDim(xShape.GetDimNum() - 1);

    OP_TILING_CHECK(inputParams_.M <= 0 || inputParams_.N <= 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "M, N",
                        (std::to_string(inputParams_.M) + ", " + std::to_string(inputParams_.N)).c_str(),
                        "the values of M and N must be positive"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(inputParams_.M > maxShapeDim || inputParams_.N > maxShapeDim,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "x", Shape2String(xShape).c_str(),
                                                          "the values of M and N must not exceed INT32_MAX"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        ValidateRotShape(rotShape) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "rotation", Shape2String(rotShape).c_str(),
                                              "the shape of rotation is invalid"),
        return ge::GRAPH_FAILED);

    // Verify the total element count of the output scale matches the inferred result (flat equal-total views are
    // allowed): M*CeilDiv(N,64)*lastDim. DAV_9201/DAV_9202 hif4 output: scale is (*, CeilDiv(N,64), 1); MXFP output:
    // (*, CeilDiv(N,64), 2).
    const auto& scaleShape = context_->GetOutputShape(SCALE_INDEX)->GetStorageShape();
    gert::Shape mxScaleShape = xShape;
    mxScaleShape.SetDim(mxScaleShape.GetDimNum() - 1, Ops::Base::CeilDiv(inputParams_.N, MX_SCALE_CEIL_NUM));
    int64_t scaleLastDim = (inputParams_.yDtype == Ops::NN::DT_HIFLOAT4) ? HIF4_SCALE_LAST_DIM : MX_SCALE_LAST_SIZE;
    mxScaleShape.AppendDim(scaleLastDim);
    OP_TILING_CHECK(
        ValidateScaleShape(mxScaleShape, scaleShape) != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "scale", Shape2String(scaleShape).c_str(),
                                              "the shape of scale is invalid"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void RotateQuantAptTiling::PrintTilingData()
{
    auto& tiling = tilingData_;
    OP_LOGD(inputParams_.opName, "=== RotateQuantApt Tiling Data ===");
    OP_LOGD(inputParams_.opName, "M:            [%u]", tiling.M);
    OP_LOGD(inputParams_.opName, "N:            [%u]", tiling.N);
    OP_LOGD(inputParams_.opName, "K:            [%u]", tiling.K);
    OP_LOGD(inputParams_.opName, "nKRatio:      [%u]", tiling.nKRatio);
    OP_LOGD(inputParams_.opName, "mL1:          [%u]", tiling.mL1);
    OP_LOGD(inputParams_.opName, "tailML1:      [%u]", tiling.tailML1);
    OP_LOGD(inputParams_.opName, "nL1:          [%u]", tiling.nL1);
    OP_LOGD(inputParams_.opName, "kL1:          [%u]", tiling.kL1);
    OP_LOGD(inputParams_.opName, "baseM:        [%u]", tiling.baseM);
    OP_LOGD(inputParams_.opName, "baseN:        [%u]", tiling.baseN);
    OP_LOGD(inputParams_.opName, "baseK:        [%u]", tiling.baseK);
    OP_LOGD(inputParams_.opName, "tailML1:      [%u]", tiling.tailML1);
    OP_LOGD(inputParams_.opName, "dstTypeMax:   [%f]", tiling.dstTypeMax);
    OP_LOGD(inputParams_.opName, "invDstTypeMax:[%f]", tiling.invDstTypeMax);
    OP_LOGD(inputParams_.opName, "axis:         [%d]", tiling.axis);
    OP_LOGD(inputParams_.opName, "roundMode:    [%d]", tiling.roundMode);
    OP_LOGD(inputParams_.opName, "scaleAlg:     [%d]", tiling.scaleAlg);
    OP_LOGD(inputParams_.opName, "trans:        [%u]", tiling.trans);
    OP_LOGD(inputParams_.opName, "hasAlpha:     [%u]", tiling.hasAlpha);
    OP_LOGD(inputParams_.opName, "groupMode:    [%u]", tiling.groupMode);
    OP_LOGD(inputParams_.opName, "groupNum:     [%u]", tiling.groupNum);
    OP_LOGD(inputParams_.opName, "groupListType:[%d]", tiling.groupListType);
    OP_LOGD(inputParams_.opName, "ubDB: [%u]", tiling.ubDB);
    OP_LOGD(inputParams_.opName, "=== End Tiling Data ===");
}

ge::graphStatus RotateQuantAptTiling::CalcTilingData()
{
    tilingData_.M = static_cast<uint32_t>(inputParams_.M);
    tilingData_.N = static_cast<uint32_t>(inputParams_.N);
    tilingData_.K = static_cast<uint32_t>(inputParams_.K);
    tilingData_.B = (inputParams_.rotType == ROT_TYPE_MULTI) ? static_cast<uint32_t>(inputParams_.nKRatio) :
                                                               SINGLE_ROTATION_B_SIZE;
    tilingData_.nKRatio = static_cast<uint32_t>(inputParams_.nKRatio);
    tilingData_.groupMode = inputParams_.groupMode;
    tilingData_.groupNum = inputParams_.groupNum;
    tilingData_.groupListType = static_cast<int32_t>(inputParams_.groupListType);

    tilingData_.dstTypeMax = inputParams_.dstTypeMax;
    if (tilingData_.dstTypeMax > DST_TYPE_MAX_DEFAULT) {
        tilingData_.invDstTypeMax = INVERSE_UNIT / tilingData_.dstTypeMax;
    } else {
        tilingData_.invDstTypeMax = DST_TYPE_MAX_DEFAULT;
    }

    tilingData_.axis = inputParams_.axis;
    tilingData_.roundMode = inputParams_.roundMode;
    tilingData_.scaleAlg = inputParams_.scaleAlg;
    tilingData_.trans = static_cast<uint32_t>(inputParams_.trans);
    tilingData_.hasAlpha = static_cast<uint32_t>(inputParams_.hasAlpha);
    return ge::GRAPH_SUCCESS;
}

bool RotateQuantAptTiling::IsHiFloat4Output() const
{
    return context_->GetOutputDesc(Y_INDEX)->GetDataType() == Ops::NN::DT_HIFLOAT4;
}

uint64_t RotateQuantAptTiling::GetEpilogueWorkspaceBytes(uint32_t maxRows) const
{
    return MX_FIXED_UB_BYTES + static_cast<uint64_t>(maxRows) * MX_SCALE_ROW_BYTES;
}

bool RotateQuantAptTiling::SetBaseBlockTiling()
{
    const uint64_t rot = tilingData_.K;
    const uint64_t rotBytes = rot * rot * ELEMENT_BYTES_16BIT;
    if (compileInfo_.l1Size <= DOUBLE_BUFFER * rotBytes || compileInfo_.ubSize < GetEpilogueWorkspaceBytes(MIN_M)) {
        return false;
    }

    // The common Tensor API Block has two L0/L1 stages on both supported SoCs.
    uint64_t maxBaseK = compileInfo_.l0BSize / DOUBLE_BUFFER / rot / ELEMENT_BYTES_16BIT;
    maxBaseK = maxBaseK / MIN_M * MIN_M;
    if (maxBaseK == 0) {
        return false;
    }
    tilingData_.baseK = static_cast<uint32_t>(std::min(rot, maxBaseK));
    tilingData_.baseN = tilingData_.nL1;
    const uint64_t l1Rows = (compileInfo_.l1Size - DOUBLE_BUFFER * rotBytes) /
                            (DOUBLE_BUFFER * rot * ELEMENT_BYTES_16BIT);
    const uint64_t l0ARows = compileInfo_.l0ASize / DOUBLE_BUFFER / tilingData_.baseK / ELEMENT_BYTES_16BIT;
    const uint64_t l0CRows = compileInfo_.l0CSize / DOUBLE_BUFFER / rot / ELEMENT_BYTES_32BIT;
    const uint64_t ubRows = (compileInfo_.ubSize - MX_FIXED_UB_BYTES) / MX_SCALE_ROW_BYTES;
    // Emit a 16-aligned whole tile that fits both L0 stages, both operand L1
    // stages and the complete epilogue workspace; the kernel consumes it directly.
    const uint64_t maxRows = std::min(
        {l1Rows, l0ARows, l0CRows, ubRows, static_cast<uint64_t>(EPILOGUE_ELEMENTS) / rot});
    tilingData_.baseM = static_cast<uint32_t>(maxRows / MIN_M * MIN_M);
    return tilingData_.baseM != 0;
}

bool RotateQuantAptTiling::SetHiFloat4MatmulTiling()
{
    // Keep the mainline A6 Basic API's single-buffer L0 and shared-rotation tiling.
    tilingData_.baseK = tilingData_.kL1;
    tilingData_.baseN = tilingData_.nL1;
    const uint32_t baseML0A = compileInfo_.l0ASize / tilingData_.baseK / ELEMENT_BYTES_16BIT;
    const uint32_t baseML0C = compileInfo_.l0CSize / tilingData_.baseN / ELEMENT_BYTES_32BIT;
    tilingData_.baseM = std::max(std::min(baseML0A, baseML0C), 1U);

    const uint32_t totalCoreNum = static_cast<uint32_t>(compileInfo_.aicNum);
    const uint64_t totalEffectiveM = static_cast<uint64_t>(tilingData_.M) * tilingData_.nKRatio;
    OP_TILING_CHECK(totalEffectiveM > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "M * (N / K)", std::to_string(totalEffectiveM).c_str(),
                        "the value of M * (N / K) must not exceed UINT32_MAX for ordinary shared rotation"),
                    return false);
    const uint32_t mL1Max = static_cast<uint32_t>(
        (compileInfo_.l1Size - tilingData_.kL1 * tilingData_.nL1 * ELEMENT_BYTES_16BIT) /
        (tilingData_.kL1 * ELEMENT_BYTES_16BIT));
    uint64_t mL1PerCore = (totalEffectiveM + totalCoreNum - 1) / totalCoreNum;
    mL1PerCore = (mL1PerCore + MIN_M - 1) / MIN_M * MIN_M;
    tilingData_.mL1 = static_cast<uint32_t>(
        std::min({static_cast<uint64_t>(mL1Max), mL1PerCore, static_cast<uint64_t>(tilingData_.baseM)}));
    const uint64_t tileNum = (totalEffectiveM + tilingData_.mL1 - 1) / tilingData_.mL1;
    inputParams_.realCoreNum = static_cast<uint32_t>(std::min(tileNum, static_cast<uint64_t>(totalCoreNum)));
    const uint32_t remainderM = static_cast<uint32_t>(totalEffectiveM % tilingData_.mL1);
    tilingData_.tailML1 = remainderM == 0 ? tilingData_.mL1 : remainderM;
    return true;
}

bool RotateQuantAptTiling::SetMatmulTiling()
{
    tilingData_.kL1 = tilingData_.K;
    tilingData_.nL1 = tilingData_.K;
    // Match the MIX launch profile; the Tensor API wrapper selects split-M delivery from it.
    tilingData_.aivNumRatio = IsDav920Series(compileInfo_.npuArch) ? 1 : AIC_AIV_RATIO_950;
    if (IsHiFloat4Output()) {
        return SetHiFloat4MatmulTiling();
    }
    if (!SetBaseBlockTiling()) {
        return false;
    }

    uint32_t totalCoreNum = static_cast<uint32_t>(compileInfo_.aicNum);
    if (inputParams_.groupMode) {
        // Actual group lengths live on device. Launch all cores; the scheduler
        // determines active cores and each group tail without crossing boundaries.
        uint64_t totalRows = static_cast<uint64_t>(tilingData_.M) * tilingData_.nKRatio;
        uint64_t perCore = (totalRows + totalCoreNum - 1) / totalCoreNum;
        if (tilingData_.B > SINGLE_ROTATION_B_SIZE) {
            // Each M tile produces B tasks, one per N-axis rotation.
            uint32_t mLoops = std::max(totalCoreNum / tilingData_.B, 1U);
            perCore = (static_cast<uint64_t>(tilingData_.M) + mLoops - 1) / mLoops;
        }
        perCore = (perCore + MIN_M - 1) / MIN_M * MIN_M;
        uint64_t rotBytes = static_cast<uint64_t>(tilingData_.K) * tilingData_.K * ELEMENT_BYTES_16BIT;
        if (compileInfo_.l1Size <= DOUBLE_BUFFER * rotBytes) {
            return false;
        }
        uint64_t l1Rows = (compileInfo_.l1Size - DOUBLE_BUFFER * rotBytes) /
                          (DOUBLE_BUFFER * tilingData_.K * ELEMENT_BYTES_16BIT);
        uint64_t rows = std::min({perCore, l1Rows, static_cast<uint64_t>(tilingData_.baseM),
                                  static_cast<uint64_t>(EPILOGUE_ELEMENTS / tilingData_.K)});
        if (rows == 0) {
            return false;
        }
        tilingData_.mL1 = static_cast<uint32_t>(rows);
        tilingData_.tailML1 = tilingData_.mL1;
        inputParams_.realCoreNum = totalCoreNum;
        return true;
    }
    if (tilingData_.B == SINGLE_ROTATION_B_SIZE) {
        const uint64_t totalEffectiveM = static_cast<uint64_t>(tilingData_.M) * tilingData_.nKRatio;
        // The retained CMCT scheduler stores the flattened row count in uint32_t.
        OP_TILING_CHECK(totalEffectiveM > std::numeric_limits<uint32_t>::max(),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "M * (N / K)", std::to_string(totalEffectiveM).c_str(),
                            "the value of M * (N / K) must not exceed UINT32_MAX for ordinary shared rotation"),
                        return false);
        uint64_t mL1PerCore = (totalEffectiveM + totalCoreNum - 1) / totalCoreNum;
        mL1PerCore = (mL1PerCore + MIN_M - 1) / MIN_M * MIN_M;
        tilingData_.mL1 = static_cast<uint32_t>(std::min(mL1PerCore, static_cast<uint64_t>(tilingData_.baseM)));
        const uint64_t tileNum = (totalEffectiveM + tilingData_.mL1 - 1) / tilingData_.mL1;
        inputParams_.realCoreNum = static_cast<uint32_t>(std::min(tileNum, static_cast<uint64_t>(totalCoreNum)));
        uint32_t remainderM = static_cast<uint32_t>(totalEffectiveM % tilingData_.mL1);
        tilingData_.tailML1 = (remainderM == 0) ? tilingData_.mL1 : remainderM;
    } else {
        tilingData_.mL1 = tilingData_.baseM;
        uint32_t mL1LoopNum = Ops::Base::CeilDiv(tilingData_.M, tilingData_.mL1);
        if (mL1LoopNum * tilingData_.B < totalCoreNum) {
            // When mL1 equals baseM and not all cores are used, adjust the mL1 size.
            mL1LoopNum = totalCoreNum / tilingData_.B;
            tilingData_.mL1 = Ops::Base::CeilDiv(tilingData_.M, mL1LoopNum);
            tilingData_.mL1 = Ops::Base::CeilAlign(tilingData_.mL1, MIN_M); // aligned to 16
            mL1LoopNum = Ops::Base::CeilDiv(tilingData_.M, tilingData_.mL1);
        }
        uint32_t remainderM = tilingData_.M % tilingData_.mL1;
        tilingData_.tailML1 = (remainderM == 0) ? tilingData_.mL1 : remainderM;
        inputParams_.realCoreNum = std::min(mL1LoopNum * tilingData_.B, totalCoreNum);
    }
    return true;
}

void RotateQuantAptTiling::SetUbDB()
{
    tilingData_.ubDB = 1;
    if (IsHiFloat4Output()) {
        return;
    }
    if (!inputParams_.groupMode) {
        const uint64_t effectiveM = tilingData_.B == SINGLE_ROTATION_B_SIZE ?
                                        static_cast<uint64_t>(tilingData_.M) * tilingData_.nKRatio :
                                        tilingData_.M;
        const uint64_t tileNum = (effectiveM + tilingData_.mL1 - 1) / tilingData_.mL1 * tilingData_.B;
        // Actual group lengths stay on device; only ordinary mode has an exact Host tile count.
        if (tileNum <= inputParams_.realCoreNum) {
            return;
        }
    }
    // MX workspace sizes (52736 + 32*mL1) are 32-byte aligned.
    // Use the BlockMmad half-UB slots; each slot must fit the full epilogue workspace.
    const uint64_t workspaceBytes = GetEpilogueWorkspaceBytes(tilingData_.mL1);
    if (workspaceBytes <= compileInfo_.ubSize / DOUBLE_BUFFER) {
        tilingData_.ubDB = DOUBLE_BUFFER;
    }
}

ge::graphStatus RotateQuantAptTiling::DoOpTiling()
{
    OP_TILING_CHECK(CalcTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "CalcTilingData failed."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!SetMatmulTiling(), OP_LOGE(inputParams_.opName, "SetMatmulTiling failed."),
                    return ge::GRAPH_FAILED);

    SetUbDB();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

// Select ordinary/grouped scheduling; input and output dtypes remain binary metadata.
uint64_t RotateQuantAptTiling::GetTilingKey() const
{
    return inputParams_.groupMode ? GROUPED_TILING_KEY : DEFAULT_TILING_KEY;
}

ge::graphStatus RotateQuantAptTiling::GetWorkspaceSize()
{
    workspaceSize_ = SYS_WORKSPACE_SIZE;
    if (IsHiFloat4Output()) {
        workspaceSize_ += static_cast<size_t>(inputParams_.M) * static_cast<size_t>(inputParams_.N) *
                          ELEMENT_BYTES_32BIT;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateQuantAptTiling::PostTiling()
{
    context_->SetBlockDim(inputParams_.realCoreNum);
    context_->SetScheduleMode(SCHEDULE_MODE_STATIC);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData_), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(WORKSPACE_NUM);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

// Register the tiling class (950 and DAV_9201/DAV_9202 reuse the same arch35 tiling class, registered per arch).
REGISTER_TILING_TEMPLATE_WITH_ARCH(RotateQuant, RotateQuantAptTiling, static_cast<int32_t>(Ops::Base::DAV_3510),
                                   TEMPLATE_PRIORITY);
static Ops::NN::Optiling::RegisterArch g_rotateQuantAptTiling9201Register = Ops::NN::Optiling::RegisterArch(
                                                                                "RotateQuant")
                                                                                .tiling<RotateQuantAptTiling>(
                                                                                    TEMPLATE_PRIORITY,
                                                                                    static_cast<int32_t>(
                                                                                        Ops::Base::DAV_9201));
static Ops::NN::Optiling::RegisterArch g_rotateQuantAptTiling9202Register = Ops::NN::Optiling::RegisterArch(
                                                                                "RotateQuant")
                                                                                .tiling<RotateQuantAptTiling>(
                                                                                    TEMPLATE_PRIORITY,
                                                                                    static_cast<int32_t>(
                                                                                        Ops::Base::DAV_9202));

static ge::graphStatus RotateQuantAptTilingFunc(gert::TilingContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "RotateQuant", "TilingContext is null!");
    return Ops::NN::Optiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForRotateQuantApt(gert::TilingParseContext* context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "RotateQuant", "TilingParseContext is null!");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RotateQuant)
    .Tiling(RotateQuantAptTilingFunc)
    .TilingParse<RotateQuantAptCompileInfo>(TilingParseForRotateQuantApt);

} // namespace RotateQuant
} // namespace NN
} // namespace Ops
