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
 * \file swiglu_hi_f4_quant_tiling_arch92.cpp
 * \brief Tiling implementation for SwiGLU + HiFloat4 quantization
 */

#include <cmath>
#include <sstream>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/platform_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "../../../../../common/inc/op_host/op_const_def_nn.h"
#include "../swiglu_hi_f4_quant_tiling.h"
#include "../../../op_kernel/arch92/swiglu_hi_f4_quant_tiling_data.h"

using namespace std;
using namespace ge;
using namespace AscendC;

namespace optiling {
// ==================== 常量定义 ====================
constexpr int64_t INDEX_ATTR_ACTIVATE_DIM = 0;
constexpr int64_t INDEX_ATTR_ACTIVATE_LEFT = 1;
constexpr int64_t INDEX_ATTR_SWIGLU_MODE = 2;
constexpr int64_t INDEX_ATTR_CLAMP_LIMIT = 3;
constexpr int64_t INDEX_ATTR_GLU_ALPHA = 4;
constexpr int64_t INDEX_ATTR_GLU_BIAS = 5;
constexpr int64_t INDEX_ATTR_AXIS = 6;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 7;
constexpr int64_t LIMIT_GRPUP_INDEX = 512;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DIGIT_EIGHT = 8;
constexpr int64_t SIXTEEN = 16;
constexpr int64_t DIGIT_64 = 64;
constexpr int64_t DIGIT_128 = 128;
constexpr int64_t DIGIT_256 = 256;
constexpr int64_t DIGIT_512 = 512;
constexpr int64_t QUANT_BLOCK_SIZE = 64;
constexpr int64_t BLOCK_PER_GROUP = 1;
constexpr int64_t UINT32_BYTES_SIZE = 4;
constexpr int64_t UINT16_BYTES_SIZE = 2;
constexpr int64_t UINT8_BYTES_SIZE = 1;
constexpr int64_t BYTES_OF_INPUT = 2;
constexpr int64_t BYTES_OF_TMP_TYPE = 1;
constexpr int64_t MAX_BYTES_OF_OUTPUT = 1;
constexpr int64_t BYTES_OF_HIFLOAT4_SCALE = 4;
constexpr size_t MAX_DIM_NUM = 7;

constexpr int64_t RESERVED_UB_SIZE = 4096;

constexpr int64_t HIFP4_PER_CORE_MIN_GROUP_NUM = 32;
constexpr int64_t HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM = 8;
constexpr int64_t HIFP4_OUTPUT_REG_ONELOOP = 2;
constexpr int64_t HIFP4_MAXPER8_REG_ONELOOP = 16;
constexpr int64_t HIFP4_DB_NUM = 2;
constexpr int64_t HIFP4_NUM_4 = 4;
constexpr int64_t HIFP4_MID_UB_RES_NUM = 7;
constexpr int64_t UB_BLOCK_SIZE = 32;
constexpr int64_t HIFP4_ONE_GROUP_INPUT_BYTES = 256;
constexpr int64_t HIFP4_ONE_GROUP_OUTPUT_BYTES = 32;
constexpr int64_t HIFP4_ONE_GROUP_OUTPUT_SCALE_BYTES = 4;
constexpr int64_t HIFP4_ONE_GROUP_MID_UB_RES_BYTES = 117;
constexpr int64_t HIFP4_ONE_GROUP_SWIGLU_BYTES = 128;
constexpr int64_t HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM = 256;

constexpr int64_t TILING_KEY_BASE = 1000;

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {Ops::NN::DT_HIFLOAT4};
const std::set<ge::DataType> SCALE_SUPPORT_DTYPE_SET = {Ops::NN::DT_HIFLOAT4_SCALE};

// ==================== 辅助函数 ====================
template <typename T>
static std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

// ==================== SwigluHiF4QuantRegbaseTiling 类方法实现 ====================

ge::graphStatus SwigluHiF4QuantRegbaseTiling::GetNpuInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo_.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo_.totalCoreNum <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo_.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo_.ubSize <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get UB size."),
                return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "CompileInfo: totalCoreNum=%ld, ubSize=%ld", compileInfo_.totalCoreNum,
            compileInfo_.ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluHiF4QuantRegbaseTiling::ParseAttrs()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    // Get axis
    auto* attrAxis = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrAxis);
    attrParam_.axis = (attrAxis != nullptr) ? static_cast<int64_t>(*attrAxis) : -1;

    // Convert positive axis to negative index
    if (attrParam_.axis >= 0) {
        attrParam_.axis -= inputInfo_.dimNum;
    }
    OP_CHECK_IF(attrParam_.axis != -1 && attrParam_.axis != -2,
                OP_LOGE(context_->GetNodeName(), "The axis should be -1/-2, please check."), return ge::GRAPH_FAILED);

    attrParam_.blockSize = QUANT_BLOCK_SIZE;

    // Get activate_dim
    auto* attrActivateDim = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_ACTIVATE_DIM);
    attrParam_.activateDim = (attrActivateDim != nullptr) ? static_cast<int64_t>(*attrActivateDim) : -1;
    OP_LOGD(context_->GetNodeName(), "attr activate_dim = %ld", attrParam_.activateDim);

    if (attrParam_.activateDim >= 0) {
        attrParam_.activateDim -= inputInfo_.dimNum;
    }
    OP_CHECK_IF(attrParam_.activateDim != -1 && attrParam_.activateDim != -2,
                OP_LOGE(context_->GetNodeName(), "The activateDim should be -1/-2, please check."),
                return ge::GRAPH_FAILED);

    // Get activate_left
    auto* attrActivateLeft = attrs->GetAttrPointer<bool>(INDEX_ATTR_ACTIVATE_LEFT);
    attrParam_.activateLeft = (attrActivateLeft != nullptr) ? *attrActivateLeft : false;
    OP_LOGD(context_->GetNodeName(), "attr activate_left = %d", attrParam_.activateLeft);

    // Get swiglu_mode
    auto* attrSwigluMode = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SWIGLU_MODE);
    attrParam_.swigluMode = (attrSwigluMode != nullptr) ? static_cast<int64_t>(*attrSwigluMode) : 0;
    OP_LOGD(context_->GetNodeName(), "attr swiglu_mode = %ld", attrParam_.swigluMode);

    // Get clamp_limit
    auto* attrClampLimit = attrs->GetAttrPointer<float>(INDEX_ATTR_CLAMP_LIMIT);
    attrParam_.clampLimit = (attrClampLimit != nullptr) ? *attrClampLimit : 7.0f;
    OP_CHECK_IF(
        (attrParam_.swigluMode == 1) && (attrParam_.clampLimit <= 0.0f),
        OP_LOGE(context_->GetNodeName(), "swigluMode == 1, clampLimit must > 0, but is %f", attrParam_.clampLimit),
        return ge::GRAPH_FAILED);

    // Get glu_alpha
    auto* attrGluAlpha = attrs->GetAttrPointer<float>(INDEX_ATTR_GLU_ALPHA);
    attrParam_.gluAlpha = (attrGluAlpha != nullptr) ? *attrGluAlpha : 1.702f;

    // Get glu_bias
    auto* attrGluBias = attrs->GetAttrPointer<float>(INDEX_ATTR_GLU_BIAS);
    attrParam_.gluBias = (attrGluBias != nullptr) ? *attrGluBias : 1.0f;

    // Get scale_alg: 0=mse_first(有偏, Eb/Ec阈值为4/2), 1=unbiased(无偏, Eb/Ec阈值为3.75/1.875)
    auto* attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    attrParam_.scaleAlg = (attrScaleAlg != nullptr) ? static_cast<int64_t>(*attrScaleAlg) : 0;
    OP_CHECK_IF(attrParam_.scaleAlg != 0 && attrParam_.scaleAlg != 1,
                OP_LOGE(context_->GetNodeName(),
                        "The scale_alg only support 0(mse_first) or 1(unbiased), but is %ld, please check.",
                        attrParam_.scaleAlg),
                return ge::GRAPH_FAILED);

    // 当前仅支持activate_dim=-1 axis=-1组合
    OP_CHECK_IF(attrParam_.activateDim != -1 && attrParam_.axis != -1,
                OP_LOGE(context_->GetNodeName(), "Only support activate_dim=-1 and axis=-1, please check."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluHiF4QuantRegbaseTiling::ValidateInput()
{
    // Check x dtype
    auto inputXPtr = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
                OP_LOGE(context_->GetNodeName(),
                        "Input x's data type only support FLOAT16 and BFLOAT16 currently, but x is [%s], please check.",
                        Ops::Base::ToString(xDtype).c_str()),
                return ge::GRAPH_FAILED);
    inputInfo_.xDtype = xDtype;

    // Get input shape
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    inputInfo_.dimNum = static_cast<int64_t>(xShape.GetDimNum());

    OP_CHECK_IF(
        inputInfo_.dimNum < 2 || inputInfo_.dimNum > static_cast<int64_t>(MAX_DIM_NUM),
        OP_LOGE(context_->GetNodeName(), "The shape is invalid, Input x rank[%ld] should be in [2, 7], please check.",
                inputInfo_.dimNum),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(xShape.GetDim(xShape.GetDimNum() - 1) % DIGIT_128 != 0,
                OP_LOGE(context_->GetNodeName(), "The last axis of input x should be divisible by 128, please check."),
                return ge::GRAPH_FAILED);

    inputInfo_.inputDimNeg1 = xShape.GetDim(xShape.GetDimNum() - 1);

    // Detect optional input group_index
    inputInfo_.groupIndexNum = 0; // 初始值为 0
    auto groupIndexDesc = context_->GetOptionalInputDesc(1);
    if (groupIndexDesc != nullptr) {
        auto groupIndexDtype = groupIndexDesc->GetDataType();
        if (groupIndexDtype == ge::DT_INT32) {
            inputInfo_.groupIndexType = 1; // groupIndexType 设为1，表示输入groupIndex为int32类型
            OP_LOGI(context_->GetNodeName(), "group_index exists with type int32");
        } else if (groupIndexDtype == ge::DT_INT64) {
            inputInfo_.groupIndexType = 2; // groupIndexType 设为2，表示输入groupIndex为int64类型
            OP_LOGI(context_->GetNodeName(), "group_index exists with type int64");
        } else {
            OP_LOGE(context_->GetNodeName(), "group_index has unsupported dtype %d", static_cast<int>(groupIndexDtype));
            return ge::GRAPH_FAILED;
        }

        // 获取 group_index 的 shape 并校验维度必须为 1
        auto groupIndexShape = context_->GetOptionalInputShape(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, groupIndexShape);
        size_t groupIndexDimNum = groupIndexShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(groupIndexDimNum != 1,
                    OP_LOGE(context_->GetNodeName(), "group_index dimension must be 1, but got %zu", groupIndexDimNum),
                    return ge::GRAPH_FAILED);
        inputInfo_.groupIndexNum = groupIndexShape->GetStorageShape().GetDim(0);
        OP_LOGI(context_->GetNodeName(), "group_index exists with shape[0]=%ld", inputInfo_.groupIndexNum);

        // 校验 group_index shape必须满足 0 <shape[0] <= 512
        OP_CHECK_IF(inputInfo_.groupIndexNum > LIMIT_GRPUP_INDEX || inputInfo_.groupIndexNum <= 0,
                    OP_LOGE(context_->GetNodeName(), "group_index shape[0] must be <= 512 and > 0, but got %ld.",
                            inputInfo_.groupIndexNum),
                    return ge::GRAPH_FAILED);
    } else {
        inputInfo_.groupIndexType = 0;
        OP_LOGI(context_->GetNodeName(), "group_index does not exist");
    }

    OP_LOGI(context_->GetNodeName(), "ValidateInput: xDtype=%d, dimNum=%ld, inputDimNeg1=%ld",
            static_cast<int>(inputInfo_.xDtype), inputInfo_.dimNum, inputInfo_.inputDimNeg1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluHiF4QuantRegbaseTiling::ValidateOutput()
{
    // Check y dtype
    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_SET.count(yDtype) == 0,
        OP_LOGE(context_->GetNodeName(), "Output y's data type only support HIFLOAT4, but y is [%s], please check.",
                Ops::Base::ToString(yDtype).c_str()),
        return ge::GRAPH_FAILED);
    outputInfo_.yDtype = yDtype;

    // Check scale dtype
    auto outputScalePtr = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputScalePtr);
    auto scaleDtype = outputScalePtr->GetDataType();
    OP_CHECK_IF(SCALE_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
                OP_LOGE(context_->GetNodeName(),
                        "Output scale's data type must be DT_HIFLOAT4_SCALE, but scale is [%s], please check.",
                        Ops::Base::ToString(scaleDtype).c_str()),
                return ge::GRAPH_FAILED);
    outputInfo_.scaleDtype = scaleDtype;

    // Check output shapes
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(xShape.GetDim(xShape.GetDimNum() - 1) != yShape.GetDim(yShape.GetDimNum() - 1) * 2,
                OP_LOGE(context_->GetNodeName(),
                        "The shape of output y:%s must be half of the input x:%s on activate dim, please check.",
                        Shape2String(yShape).c_str(), Shape2String(xShape).c_str()),
                return ge::GRAPH_FAILED);

    outputInfo_.outputYDimNeg1 = yShape.GetDim(yShape.GetDimNum() - 1);

    // Check scale shape
    auto scaleShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
    auto scaleShape = scaleShapePtr->GetStorageShape();

    auto correctScaleShape = yShape;
    if (attrParam_.axis == -1) {
        correctScaleShape.SetDim(yShape.GetDimNum() - 1,
                                 Ops::Base::CeilDiv(yShape.GetDim(yShape.GetDimNum() - 1), DIGIT_64));
        correctScaleShape.AppendDim(1);
        OP_CHECK_IF(scaleShape != correctScaleShape,
                    OP_LOGE(context_->GetNodeName(),
                            "The shape of output scale %s is incorrect, correct shape is %s, please check.",
                            Shape2String(scaleShape).c_str(), Shape2String(correctScaleShape).c_str()),
                    return ge::GRAPH_FAILED);
    }

    OP_LOGI(context_->GetNodeName(), "ValidateOutput: yDtype=%d, scaleDtype=%d, outputYDimNeg1=%ld",
            static_cast<int>(outputInfo_.yDtype), static_cast<int>(outputInfo_.scaleDtype), outputInfo_.outputYDimNeg1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluHiF4QuantRegbaseTiling::PreProcess()
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    if (attrParam_.activateDim == -1 && attrParam_.axis == -1) {
        int64_t mergedDim0 = 1;
        int64_t mergedDim1 = 1;
        for (size_t i = 0; i < xShape.GetDimNum() - 1; i++) {
            mergedDim1 *= xShape.GetDim(i);
        }
        inputInfo_.inputDimNeg3 = mergedDim0;
        inputInfo_.inputDimNeg2 = mergedDim1;
    } else {
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(context_->GetNodeName(),
            "Merge shape in PreProcess: inputDimNeg3=%ld, inputDimNeg2=%ld, inputDimNeg1=%ld, outputYDimNeg1=%ld",
            inputInfo_.inputDimNeg3, inputInfo_.inputDimNeg2, inputInfo_.inputDimNeg1, outputInfo_.outputYDimNeg1);
    return ge::GRAPH_SUCCESS;
}

// ==================== 核心 Tiling 计算 ===================

int64_t SwigluHiF4QuantRegbaseTiling::HiFp4GetubFactorNeg1()
{
    int64_t vlByteSize = static_cast<int64_t>(256);
    int64_t ubReserveAlignBuf = HIFP4_MID_UB_RES_NUM * UB_BLOCK_SIZE * HIFP4_DB_NUM +
                                HIFP4_DB_NUM * vlByteSize * DIGIT_FOUR + RESERVED_UB_SIZE * 4;
    int64_t ubCap = (compileInfo_.ubSize - ubReserveAlignBuf) / HIFP4_DB_NUM /
                    (HIFP4_ONE_GROUP_MID_UB_RES_BYTES + HIFP4_ONE_GROUP_INPUT_BYTES + HIFP4_ONE_GROUP_OUTPUT_BYTES +
                     HIFP4_ONE_GROUP_OUTPUT_SCALE_BYTES + HIFP4_ONE_GROUP_SWIGLU_BYTES);
    return ubCap / HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM;
}

void SwigluHiF4QuantRegbaseTiling::SetTilingParamsNeg1()
{
    // Set basic block parameters according to axis
    tilingResult_.basicDimNeg1 = DIGIT_512;
    tilingResult_.basicDimNeg2 = 1;

    int64_t ubTotalBasicBlock = HiFp4GetubFactorNeg1();
    int64_t basicNumNeg1 = Ops::Base::CeilDiv(outputInfo_.outputYDimNeg1, tilingResult_.basicDimNeg1);
    if (ubTotalBasicBlock >= basicNumNeg1) {
        // Full-load scenario: entire row fits in UB
        tilingResult_.maxBasicNumUbDimNeg1 = basicNumNeg1;
        tilingResult_.maxBasicNumUbDimNeg2 = Ops::Base::FloorDiv(ubTotalBasicBlock, basicNumNeg1);
        tilingResult_.ubLoopPerRow = 1;
        tilingResult_.ubTailPerRow = outputInfo_.outputYDimNeg1;
        tilingResult_.isFullLoad = 1;
    } else {
        // Non-full-load scenario
        tilingResult_.maxBasicNumUbDimNeg1 = ubTotalBasicBlock;
        tilingResult_.maxBasicNumUbDimNeg2 = 1;
        tilingResult_.ubLoopPerRow = Ops::Base::CeilDiv(basicNumNeg1, tilingResult_.maxBasicNumUbDimNeg1);
        tilingResult_.ubTailPerRow = outputInfo_.outputYDimNeg1 - (tilingResult_.ubLoopPerRow - 1) *
                                                                      tilingResult_.maxBasicNumUbDimNeg1 *
                                                                      tilingResult_.basicDimNeg1;
        tilingResult_.isFullLoad = 0;
    }

    // Calculate inter-core split strategy
    int64_t basicNumNeg2 = Ops::Base::CeilDiv(inputInfo_.inputDimNeg2, tilingResult_.basicDimNeg2);
    tilingResult_.usedCoreNum = std::min(basicNumNeg2, compileInfo_.totalCoreNum);

    // Tail core parameters
    tilingResult_.tailCoreBasicNumDimNeg2 = Ops::Base::FloorDiv(basicNumNeg2, tilingResult_.usedCoreNum);
    tilingResult_.tailCoreLoopTimes = Ops::Base::CeilDiv(tilingResult_.tailCoreBasicNumDimNeg2,
                                                         tilingResult_.maxBasicNumUbDimNeg2);
    tilingResult_.tailCoreLastLoopBasicNum = tilingResult_.tailCoreBasicNumDimNeg2 -
                                             (tilingResult_.tailCoreLoopTimes - 1) * tilingResult_.maxBasicNumUbDimNeg2;

    // Front core parameters
    tilingResult_.frontCoreNum = basicNumNeg2 % tilingResult_.usedCoreNum;
    tilingResult_.frontCoreBasicNumDimNeg2 = tilingResult_.tailCoreBasicNumDimNeg2 + 1;
    tilingResult_.frontCoreLoopTimes = Ops::Base::CeilDiv(tilingResult_.frontCoreBasicNumDimNeg2,
                                                          tilingResult_.maxBasicNumUbDimNeg2);
    tilingResult_.frontCoreLastLoopBasicNum = tilingResult_.frontCoreBasicNumDimNeg2 -
                                              (tilingResult_.frontCoreLoopTimes - 1) *
                                                  tilingResult_.maxBasicNumUbDimNeg2;
}

// ==================== 计算编排 ====================

ge::graphStatus SwigluHiF4QuantRegbaseTiling::CalculateTiling()
{
    if (attrParam_.axis == -1) {
        SetTilingParamsNeg1();
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluHiF4QuantRegbaseTiling::FillTilingData()
{
    tilingData_ = context_->GetTilingData<SwigluHiF4QuantTilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(memset_s(tilingData_, sizeof(SwigluHiF4QuantTilingData), 0, sizeof(SwigluHiF4QuantTilingData)) != EOK,
                OP_LOGE(context_->GetNodeName(), "memset tilingData failed"), return ge::GRAPH_FAILED);

    // Basic parameters
    tilingData_->usedCoreNum = tilingResult_.usedCoreNum;

    // Data shape
    tilingData_->inputDimNeg1 = inputInfo_.inputDimNeg1;
    tilingData_->inputDimNeg2 = inputInfo_.inputDimNeg2;
    tilingData_->outputYDimNeg1 = outputInfo_.outputYDimNeg1;
    tilingData_->groupIndexNum = inputInfo_.groupIndexNum;

    // Memory allocation
    tilingData_->basicDimNeg1 = tilingResult_.basicDimNeg1;
    tilingData_->basicDimNeg2 = tilingResult_.basicDimNeg2;
    tilingData_->maxBasicNumUbDimNeg1 = tilingResult_.maxBasicNumUbDimNeg1;
    tilingData_->maxBasicNumUbDimNeg2 = tilingResult_.maxBasicNumUbDimNeg2;
    tilingData_->ubLoopPerRow = tilingResult_.ubLoopPerRow;
    tilingData_->ubTailPerRow = tilingResult_.ubTailPerRow;

    // Inter-core split
    tilingData_->frontCoreNum = tilingResult_.frontCoreNum;
    tilingData_->frontCoreBasicNumDimNeg2 = tilingResult_.frontCoreBasicNumDimNeg2;
    tilingData_->frontCoreLoopTimes = tilingResult_.frontCoreLoopTimes;
    tilingData_->frontCoreLastLoopBasicNum = tilingResult_.frontCoreLastLoopBasicNum;
    tilingData_->tailCoreBasicNumDimNeg2 = tilingResult_.tailCoreBasicNumDimNeg2;
    tilingData_->tailCoreLoopTimes = tilingResult_.tailCoreLoopTimes;
    tilingData_->tailCoreLastLoopBasicNum = tilingResult_.tailCoreLastLoopBasicNum;

    // SwiGLU attributes
    tilingData_->swigluMode = attrParam_.swigluMode;
    tilingData_->activateLeft = attrParam_.activateLeft ? 1 : 0;
    tilingData_->clampLimit = attrParam_.clampLimit;
    tilingData_->gluAlpha = attrParam_.gluAlpha;
    tilingData_->gluBias = attrParam_.gluBias;
    tilingData_->scaleAlg = attrParam_.scaleAlg;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluHiF4QuantRegbaseTiling::SetParams()
{
    // Set TilingKey
    int64_t activateDimIndex = (attrParam_.activateDim == -1) ? 0 : 1;
    int64_t axisIndex = (attrParam_.axis == -1) ? 0 : 1;
    int64_t tilingKey = TILING_KEY_BASE + inputInfo_.groupIndexType * 100 + activateDimIndex * 10 + axisIndex;

    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(tilingResult_.usedCoreNum);

    // Set workspace
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t usrWorkspaceSize = 0;
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = usrWorkspaceSize + sysWorkspaceSize;

    OP_LOGI(context_->GetNodeName(), "SetParams done. TilingKey=%ld, BlockDim=%ld", tilingKey,
            tilingResult_.usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

void SwigluHiF4QuantRegbaseTiling::PrintTilingData() const
{
    OP_LOGI(context_->GetNodeName(),
            "TilingData: usedCoreNum=%ld, "
            "inputDimNeg2=%ld, inputDimNeg1=%ld, outputYDimNeg1=%ld, "
            "groupIndexNum=%ld, basicDimNeg1=%ld, basicDimNeg2=%ld, "
            "maxBasicNumUbDimNeg1=%ld, maxBasicNumUbDimNeg2=%ld, "
            "ubLoopPerRow=%ld, ubTailPerRow=%ld, "
            "frontCoreNum=%ld, frontCoreBasicNumDimNeg2=%ld, "
            "frontCoreLoopTimes=%ld, frontCoreLastLoopBasicNum=%ld, "
            "tailCoreBasicNumDimNeg2=%ld, tailCoreLoopTimes=%ld, tailCoreLastLoopBasicNum=%ld, "
            "swigluMode=%ld, activateLeft=%ld, clampLimit=%f, gluAlpha=%f, gluBias=%f, scaleAlg=%ld",
            tilingData_->usedCoreNum, tilingData_->inputDimNeg2, tilingData_->inputDimNeg1, tilingData_->outputYDimNeg1,
            tilingData_->groupIndexNum, tilingData_->basicDimNeg1, tilingData_->basicDimNeg2,
            tilingData_->maxBasicNumUbDimNeg1, tilingData_->maxBasicNumUbDimNeg2, tilingData_->ubLoopPerRow,
            tilingData_->ubTailPerRow, tilingData_->frontCoreNum, tilingData_->frontCoreBasicNumDimNeg2,
            tilingData_->frontCoreLoopTimes, tilingData_->frontCoreLastLoopBasicNum,
            tilingData_->tailCoreBasicNumDimNeg2, tilingData_->tailCoreLoopTimes, tilingData_->tailCoreLastLoopBasicNum,
            tilingData_->swigluMode, tilingData_->activateLeft, tilingData_->clampLimit, tilingData_->gluAlpha,
            tilingData_->gluBias, tilingData_->scaleAlg);
}

} // namespace optiling
