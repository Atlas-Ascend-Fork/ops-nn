/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dynamic_block_hi_f4_quant_tiling_arch35.cpp
 * \brief Tiling implementation for DynamicBlockHiF4Quant on ascend960dt (arch35)
 */

#include "dynamic_block_hi_f4_quant_tiling_arch35.h"
#include "../../op_kernel/arch35/dynamic_block_hi_f4_quant_tilingdata.h"
#include <cmath>
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "../../op_kernel/arch35/dynamic_block_hi_f4_quant_struct.h"
#include "../../../../common/inc/op_host/op_const_def_nn.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace DynamicBlockHiF4QuantOp;
namespace optiling {

constexpr int64_t INDEX_ATTR_ROUND_MODE = 0;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 1;
constexpr int64_t DIGIT_ZERO = 0;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_64 = 64;
constexpr int64_t DIGIT_512 = 512;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;     // fp16/bf16
constexpr int64_t BYTES_OF_HIFLOAT4 = 1;       // 使用时除以 2（每 byte 打包 2 个 4-bit 元素）
constexpr int64_t BYTES_OF_HIFLOAT4_SCALE = 4; // 每 Block 4 字节 (Ea + Eb + Ec 打包)
constexpr int64_t RESERVED_UB_SIZE = 4096;
constexpr size_t MAX_DIM_NUM = 7;
constexpr size_t MIN_DIM_NUM = 2;

// Buffer 分配参考设计说明书 2.4.2.2 节 (blockSizeRow=64, blockSizeCol=512, Db=2)
constexpr int64_t BLOCK_SIZE_ROW = 64;  // 基本块的高
constexpr int64_t BLOCK_SIZE_COL = 512; // 基本块的宽（按 64*512 粒度切分）

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_HIFLOAT4};
const std::set<ge::DataType> SCALE_SUPPORT_DTYPE_SET = {Ops::NN::DT_HIFLOAT4_SCALE};

template <typename T>
std::string Shape2String(const T& shape)
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

RoundModeList DynamicBlockHiF4QuantTiling::GetRoundMode(const std::string& roundMode)
{
    if (roundMode == "rint") {
        return RoundModeList::MODE_RINT;
    } else if (roundMode == "trunc") {
        return RoundModeList::MODE_TRUNC;
    }
    return RoundModeList::MODE_UNDEFINED;
}

ge::graphStatus DynamicBlockHiF4QuantTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrRoundMode);
    std::string roundModeStr = attrRoundMode;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    OP_CHECK_IF((roundMode == RoundModeList::MODE_UNDEFINED),
                OP_LOGE(context_->GetNodeName(), "invalid round_mode:%s; round_mode should be one of {rint, trunc}",
                        attrRoundMode),
                return ge::GRAPH_FAILED);
    tilingParams.roundMode = static_cast<int64_t>(roundMode);
    OP_LOGD(context_->GetNodeName(), "The attr round_mode is %s, roundMode enum is %ld", roundModeStr.c_str(),
            tilingParams.roundMode);

    auto* attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    int64_t scaleAlg = DIGIT_ZERO;
    if (attrScaleAlg != nullptr) {
        scaleAlg = *attrScaleAlg;
    }
    OP_CHECK_IF((scaleAlg != 0 && scaleAlg != 1),
                OP_LOGE(context_->GetNodeName(), "invalid scale_alg:%ld; scale_alg should be 0 or 1", scaleAlg),
                return ge::GRAPH_FAILED);
    tilingParams.scaleAlg = scaleAlg;
    OP_LOGD(context_->GetNodeName(), "The attr scale_alg is %ld", scaleAlg);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockHiF4QuantTiling::CheckDtype() const
{
    auto inputXPtr = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_SET.count(xDtype) == 0,
                OP_LOGE(context_->GetNodeName(),
                        "Input x's data type only support FLOAT16 and BFLOAT16 currently, but x is [%s], please check.",
                        Ops::Base::ToString(xDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        Y_SUPPORT_DTYPE_SET.count(yDtype) == 0,
        OP_LOGE(context_->GetNodeName(), "Output y's data type only support HIFLOAT4, but y is [%s], please check.",
                Ops::Base::ToString(yDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputScale1Ptr = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputScale1Ptr);
    auto scale1Dtype = outputScale1Ptr->GetDataType();
    OP_CHECK_IF(SCALE_SUPPORT_DTYPE_SET.count(scale1Dtype) == 0,
                OP_LOGE(context_->GetNodeName(),
                        "Output scale1's data type only support HIFLOAT4_SCALE, but scale1 is [%s], please check.",
                        Ops::Base::ToString(scale1Dtype).c_str()),
                return ge::GRAPH_FAILED);

    auto outputScale2Ptr = context_->GetOutputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputScale2Ptr);
    auto scale2Dtype = outputScale2Ptr->GetDataType();
    OP_CHECK_IF(SCALE_SUPPORT_DTYPE_SET.count(scale2Dtype) == 0,
                OP_LOGE(context_->GetNodeName(),
                        "Output scale2's data type only support HIFLOAT4_SCALE, but scale2 is [%s], please check.",
                        Ops::Base::ToString(scale2Dtype).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockHiF4QuantTiling::CheckShape() const
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    size_t xSizeNum = xShape.GetShapeSize();
    if (xSizeNum == 0ULL) {
        OP_LOGE(context_->GetNodeName(), "dynamic_block_hi_f4_quant does not support empty tensor.");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        xShape.GetDimNum() < MIN_DIM_NUM || xShape.GetDimNum() > MAX_DIM_NUM,
        OP_LOGE(context_->GetNodeName(), "The shape is invalid, Input x rank[%zu] should be in [2, 7], please check.",
                static_cast<int64_t>(xShape.GetDimNum())),
        return ge::GRAPH_FAILED);

    int64_t lastDim = xShape.GetDim(xShape.GetDimNum() - 1);
    int64_t nonLastDim = xShape.GetDim(xShape.GetDimNum() - 2);
    OP_CHECK_IF(lastDim % DIGIT_64 != 0 || nonLastDim % DIGIT_64 != 0,
                OP_LOGE(context_->GetNodeName(),
                        "Both the -1 axis and -2 axis of input x must be multiples of 64. "
                        "The current axis -1 is [%lld] and axis -2 is [%lld], please check.",
                        lastDim, nonLastDim),
                return ge::GRAPH_FAILED);

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();
    OP_CHECK_IF(xShape != yShape,
                OP_LOGE(context_->GetNodeName(),
                        "The shape of output y[%s] must be same with shape of input x[%s], please check.",
                        Shape2String(yShape).c_str(), Shape2String(xShape).c_str()),
                return ge::GRAPH_FAILED);

    // scale1: rank = rank(x)+1, scale1.shape[:-2]=x.shape[:-1],
    //         scale1.shape[-2]=x.shape[-1]//64, scale1.shape[-1]=1
    auto scale1ShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scale1ShapePtr);
    auto scale1Shape = scale1ShapePtr->GetStorageShape();
    auto newScale1Shape = xShape;
    newScale1Shape.SetDim(xShape.GetDimNum() - 1, Ops::Base::CeilDiv(xShape.GetDim(xShape.GetDimNum() - 1), DIGIT_64));
    newScale1Shape.AppendDim(DIGIT_ONE);
    OP_CHECK_IF(newScale1Shape != scale1Shape,
                OP_LOGE(context_->GetNodeName(),
                        "The shape of output scale1[%s] is incorrect, correct shape is [%s], please check.",
                        Shape2String(scale1Shape).c_str(), Shape2String(newScale1Shape).c_str()),
                return ge::GRAPH_FAILED);

    // scale2: rank = rank(x)+1, scale2.shape[:-3]=x.shape[:-2],
    //         scale2.shape[-3]=x.shape[-2]//64, scale2.shape[-2]=x.shape[-1], scale2.shape[-1]=1
    auto scale2ShapePtr = context_->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scale2ShapePtr);
    auto scale2Shape = scale2ShapePtr->GetStorageShape();
    auto newScale2Shape = xShape;
    newScale2Shape.SetDim(xShape.GetDimNum() - DIGIT_TWO,
                          Ops::Base::CeilDiv(xShape.GetDim(xShape.GetDimNum() - DIGIT_TWO), DIGIT_64));
    newScale2Shape.AppendDim(DIGIT_ONE);
    OP_CHECK_IF(newScale2Shape != scale2Shape,
                OP_LOGE(context_->GetNodeName(),
                        "The shape of output scale2[%s] is incorrect, correct shape is [%s], please check.",
                        Shape2String(scale2Shape).c_str(), Shape2String(newScale2Shape).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockHiF4QuantTiling::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicBlockHiF4QuantTiling GetPlatformInfo.");

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParams.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParams.totalCoreNum <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParams.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((tilingParams.ubSize <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    tilingParams.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

// 合轴：将 -1 轴之前所有轴（含 -2 轴）合轴，得到二维 Shape(M, N)
// 由于 -2 轴已 64 整数倍对齐，无需保留 batch 概念
ge::graphStatus DynamicBlockHiF4QuantTiling::MergeAxis()
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    // -1 轴之前所有轴相乘得到 rowNum
    tilingParams.rowNum = 1;
    for (size_t i = 0; i < xShape.GetDimNum() - 1; i++) {
        tilingParams.rowNum *= xShape.GetDim(i);
    }
    tilingParams.colNum = xShape.GetDim(xShape.GetDimNum() - 1);

    return ge::GRAPH_SUCCESS;
}

// 按 64*512 粒度切分并均匀分到每个核上
void DynamicBlockHiF4QuantTiling::SplitCore()
{
    // 行方向基本块（64*64）数量
    tilingParams.rowBlockLoopNum = Ops::Base::CeilDiv(tilingParams.rowNum, BLOCK_SIZE_ROW);
    // 列方向 512 粒度块数量
    tilingParams.colBlockLoopNum = Ops::Base::CeilDiv(tilingParams.colNum, BLOCK_SIZE_COL);
    // 总数据块数量 (局部变量，不入 TilingData)
    int64_t totalTaskNum = tilingParams.rowBlockLoopNum * tilingParams.colBlockLoopNum;

    // 列方向尾块长度
    tilingParams.colTileNum = (tilingParams.colNum % BLOCK_SIZE_COL == 0) ? BLOCK_SIZE_COL :
                                                                            tilingParams.colNum % BLOCK_SIZE_COL;

    // 均匀分到每个核上去
    tilingParams.usedCoreNum = std::min(tilingParams.totalCoreNum, totalTaskNum);
    tilingParams.usedCoreNum = tilingParams.usedCoreNum == 0 ? 1 : tilingParams.usedCoreNum;
    // 头核数量 = totalTaskNum % totalCoreNum
    tilingParams.normalCoreNum = totalTaskNum % tilingParams.usedCoreNum;
    // 头核处理基本块的数量 = ceil(totalTaskNum / usedCoreNum)
    tilingParams.normalCoreBlockNum = Ops::Base::CeilDiv(totalTaskNum, tilingParams.usedCoreNum);
    // 尾核处理基本块的数量 = floor(totalTaskNum / usedCoreNum)
    tilingParams.tailCoreBlockNum = totalTaskNum / tilingParams.usedCoreNum;
}

ge::graphStatus DynamicBlockHiF4QuantTiling::SetTilingParams()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicBlockHiF4QuantTiling SetTilingParams.");

    tilingParams.blockSizeRow = BLOCK_SIZE_ROW;
    tilingParams.blockSizeCol = BLOCK_SIZE_COL;

    MergeAxis();
    SplitCore();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockHiF4QuantTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicBlockHiF4QuantTiling DoTiling.");

    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The platforminfo get failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The data type check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The attr get failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The shape check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetTilingParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "DynamicBlockHiF4QuantTiling SetTilingParams failed"),
                return ge::GRAPH_FAILED);

    SetTilingKey();
    SetTilingData();
    PrintTilingData();

    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %ld.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    auto rawTilingData = context_->GetRawTilingData();
    uint64_t tilingDataSize = sizeof(tilingData);
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), reinterpret_cast<void*>(&tilingData),
                           tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "Tiling DataSize Greater than capacity, please check.");
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParams.workspaceSize;

    return ge::GRAPH_SUCCESS;
}

void DynamicBlockHiF4QuantTiling::SetTilingKey()
{
    if (tilingParams.roundMode == static_cast<int64_t>(RoundModeList::MODE_RINT)) {
        roundMode_ = TPL_RINT;
    } else if (tilingParams.roundMode == static_cast<int64_t>(RoundModeList::MODE_TRUNC)) {
        roundMode_ = TPL_TRUNC;
    }
    int64_t tilingKey = GET_TPL_TILING_KEY(roundMode_);
    tilingParams.tilingKey = tilingKey;
    OP_LOGD(context_->GetNodeName(), "roundMode is %ld, tilingKey is %ld", roundMode_, tilingKey);
    context_->SetTilingKey(tilingKey);
}

void DynamicBlockHiF4QuantTiling::SetTilingData()
{
    tilingData.tilingKey = tilingParams.tilingKey;
    tilingData.totalCoreNum = tilingParams.totalCoreNum;
    tilingData.usedCoreNum = tilingParams.usedCoreNum;
    tilingData.ubSize = tilingParams.ubSize;
    tilingData.roundMode = tilingParams.roundMode;
    tilingData.scaleAlg = tilingParams.scaleAlg;
    tilingData.blockSizeRow = tilingParams.blockSizeRow;
    tilingData.blockSizeCol = tilingParams.blockSizeCol;
    tilingData.rowNum = tilingParams.rowNum;
    tilingData.colNum = tilingParams.colNum;
    tilingData.rowBlockLoopNum = tilingParams.rowBlockLoopNum;
    tilingData.colBlockLoopNum = tilingParams.colBlockLoopNum;
    tilingData.colTileNum = tilingParams.colTileNum;
    tilingData.normalCoreNum = tilingParams.normalCoreNum;
    tilingData.normalCoreBlockNum = tilingParams.normalCoreBlockNum;
    tilingData.tailCoreBlockNum = tilingParams.tailCoreBlockNum;
}

void DynamicBlockHiF4QuantTiling::PrintTilingData()
{
    OP_LOGD(
        context_->GetNodeName(),
        "TilingData tilingKey: %ld, totalCoreNum: %ld, usedCoreNum: %ld, ubSize: %ld, roundMode: %ld, scaleAlg: %ld, "
        "blockSizeRow: %ld, blockSizeCol: %ld, rowNum: %ld, colNum: %ld, "
        "rowBlockLoopNum: %ld, colBlockLoopNum: %ld, colTileNum: %ld, "
        "normalCoreNum: %ld, normalCoreBlockNum: %ld, tailCoreBlockNum: %ld",
        tilingData.tilingKey, tilingData.totalCoreNum, tilingData.usedCoreNum, tilingData.ubSize, tilingData.roundMode,
        tilingData.scaleAlg, tilingData.blockSizeRow, tilingData.blockSizeCol, tilingData.rowNum, tilingData.colNum,
        tilingData.rowBlockLoopNum, tilingData.colBlockLoopNum, tilingData.colTileNum, tilingData.normalCoreNum,
        tilingData.normalCoreBlockNum, tilingData.tailCoreBlockNum);
}

static ge::graphStatus TilingForDynamicBlockHiF4Quant(gert::TilingContext* context)
{
    OP_LOGD("DynamicBlockHiF4QuantTiling", "Enter TilingForDynamicBlockHiF4Quant");
    OP_CHECK_IF(context == nullptr, OP_LOGE("DynamicBlockHiF4QuantTiling", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    optiling::DynamicBlockHiF4QuantTiling blockHiF4QuantTiling(context);
    return blockHiF4QuantTiling.DoTiling();
}

static ge::graphStatus TilingPrepareForDynamicBlockHiF4Quant(gert::TilingParseContext* context)
{
    OP_LOGD("DynamicBlockHiF4QuantTiling", "Enter TilingPrepareForDynamicBlockHiF4Quant");
    OP_CHECK_IF(context == nullptr, OP_LOGE("DynamicBlockHiF4QuantTiling", "TilingParse context is null."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicBlockHiF4Quant)
    .Tiling(TilingForDynamicBlockHiF4Quant)
    .TilingParse<DynamicBlockHiF4QuantCompileInfo>(TilingPrepareForDynamicBlockHiF4Quant);

} // namespace optiling
