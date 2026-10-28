/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dynamic_block_quant_i8_tiling.cpp
 * \brief
 */
#include "dynamic_block_quant_tiling.h"
#include "dynamic_block_quant_i8_tiling.h"

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_infos_def.h"
#include "error_util.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
// Tiling阶段框架契约类失败(框架对象为空/平台信息异常)统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "DynamicBlockQuant", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

namespace dynamic_block_quant_i8 {
const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_I8_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_I8_SET = {ge::DT_INT8};
const std::set<ge::DataType> SCALE_SUPPORT_I8_DTYPE_SET = {ge::DT_FLOAT};
constexpr int64_t INDEX_ATTR_MIN_SCALE = 0;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 1;
constexpr int64_t INDEX_ATTR_DST_DTYPE = 2;
constexpr int64_t INDEX_ATTR_BLOCK_SIZE_ROW = 3;
constexpr int64_t INDEX_ATTR_BLOCK_SIZE_COL = 4;
constexpr int64_t INDEX_ATTR_DST_DTYPE_MAX = 5;
constexpr int64_t BLOCK_SIZE_1 = 1;
constexpr int64_t BLOCK_SIZE_128 = 128;
constexpr int64_t DIM_TWO = 2;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t NUM_TEN = 10;

class RowTilingData {
public:
    int64_t coreNum = 0;
    int64_t perCoreBlock = 0;
    int64_t tailBlock = 0;
    int64_t extraCoreBlock = 0;
    RowTilingData(int64_t curCoreNum, int64_t totalBlock)
    {
        this->coreNum = curCoreNum;
        this->perCoreBlock = curCoreNum != 0 ? (totalBlock / curCoreNum) : totalBlock;
        this->tailBlock = totalBlock - curCoreNum * this->perCoreBlock;
        this->extraCoreBlock = this->perCoreBlock + 1;
    }
    int64_t GetCurCoreBlockNum(int64_t coreIdx)
    {
        return coreIdx >= this->tailBlock ? this->perCoreBlock : this->extraCoreBlock;
    }
};

static RoundModeList GetRoundMode(const std::string& roundMode)
{
    if (roundMode == "rint") {
        return RoundModeList::MODE_RINT;
    }
    return RoundModeList::MODE_UNDEFINED;
}

ge::graphStatus DynamicBlockQuantI8::CheckParams(DynamicBlockQuantTilingParam& tilingParam)
{
    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(context, "The dtype check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr(tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(context, "The attr get failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape(tilingParam) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context, "The shape check failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockQuantI8::CheckDtype()
{
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_IF(inputXPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                         "the input desc of x is missing in the tiling context; check "
                                                         "that the graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(INPUT_SUPPORT_DTYPE_I8_SET.count(xDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(xDtype),
                                                      "The dtype of x must be DT_FLOAT16 or DT_BF16"),
                return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_IF(outputYPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "y",
                                                         "the output desc of y is missing in the tiling context; check "
                                                         "that the graph node provides all required outputs"),
                return ge::GRAPH_FAILED);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(Y_SUPPORT_DTYPE_I8_SET.count(yDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(yDtype),
                                          "DT_INT8"),
                return ge::GRAPH_FAILED);

    auto outputScalePtr = context->GetOutputDesc(1);
    OP_CHECK_IF(outputScalePtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "scale",
                                                         "the output desc of scale is missing in the tiling context; "
                                                         "check that the graph node provides all required outputs"),
                return ge::GRAPH_FAILED);
    auto scaleDtype = outputScalePtr->GetDataType();
    OP_CHECK_IF(SCALE_SUPPORT_I8_DTYPE_SET.count(scaleDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "scale",
                                          ge::TypeUtils::DataTypeToSerialString(scaleDtype), "DT_FLOAT"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockQuantI8::GetAttr(DynamicBlockQuantTilingParam& tilingParam)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        ReportTilingStageError(context->GetNodeName(),
                               "GetShapeAttrsInfo: GetAttrs() returned null; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    auto* attrMinScale = attrs->GetAttrPointer<float>(INDEX_ATTR_MIN_SCALE);
    OP_CHECK_IF(
        attrMinScale == nullptr,
        ReportTilingStageError(context->GetNodeName(), "the optional attr min_scale is missing in the tiling context; "
                                                       "set the min_scale attribute or rely on its registered default"),
        return ge::GRAPH_FAILED);
    tilingParam.minScale = static_cast<float>(*attrMinScale);
    OP_LOGD(context, "The attr minScale is %f", tilingParam.minScale);
    OP_CHECK_IF(
        (tilingParam.minScale < 0.0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "min_scale", std::to_string(tilingParam.minScale),
                                              "The value of min_scale must be greater than or equal to 0"),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_IF(outputYPtr == nullptr,
                ReportTilingStageError(context->GetNodeName(),
                                       "the output shape of y is missing in the tiling context; check that InferShape "
                                       "executed and the framework propagated the output shapes"),
                return ge::GRAPH_FAILED);
    auto yDtype = outputYPtr->GetDataType();

    auto* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    OP_CHECK_IF(attrRoundMode == nullptr,
                ReportTilingStageError(context->GetNodeName(),
                                       "the optional attr round_mode is missing in the tiling context; set the "
                                       "round_mode attribute or rely on its registered default"),
                return ge::GRAPH_FAILED);
    std::string roundModeStr = attrRoundMode;
    RoundModeList roundMode = GetRoundMode(roundModeStr);
    tilingParam.roundMode = static_cast<int64_t>(roundMode);

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_DTYPE);
    OP_CHECK_IF(
        attrDstType == nullptr,
        ReportTilingStageError(context->GetNodeName(), "the optional attr dst_type is missing in the tiling context; "
                                                       "set the dst_type attribute or rely on its registered default"),
        return ge::GRAPH_FAILED);
    tilingParam.dstType = static_cast<int64_t>(*attrDstType);

    OP_CHECK_IF(tilingParam.dstType != static_cast<int64_t>(yDtype),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "y, dst_type",
                    ge::TypeUtils::DataTypeToSerialString(yDtype) + ", " + std::to_string(tilingParam.dstType),
                    "The dtypes of y and dst_type must be the same"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (tilingParam.dstType != 2),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "dst_type", std::to_string(tilingParam.dstType), "DT_INT8"),
        return ge::GRAPH_FAILED);

    auto* attrDstTypeMax = attrs->GetAttrPointer<float>(INDEX_ATTR_DST_DTYPE_MAX);
    tilingParam.dstTypeMax = (attrDstTypeMax != nullptr) ? static_cast<float>(*attrDstTypeMax) : 0;
    OP_CHECK_IF(!Ops::Base::IsFloatEqual(tilingParam.dstTypeMax, 0.0f),
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "dst_type_max",
                                          std::to_string(tilingParam.dstTypeMax), "0.0"),
                return ge::GRAPH_FAILED);

    return GetAttrBlock(tilingParam);
}

ge::graphStatus DynamicBlockQuantI8::GetAttrBlock(DynamicBlockQuantTilingParam& tilingParam)
{
    auto* attrs = context->GetAttrs();
    // 与GetAttr一致先校验属性容器: 若GetAttrs()返回空, 后续解引用将在任何业务校验前崩溃
    OP_CHECK_IF(attrs == nullptr,
                ReportTilingStageError(context->GetNodeName(),
                                       "the attrs container of the tiling context is null; check the TilingContext "
                                       "construction"),
                return ge::GRAPH_FAILED);
    auto* attrBlockSizeRow = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE_ROW);
    OP_CHECK_IF(attrBlockSizeRow == nullptr,
                ReportTilingStageError(context->GetNodeName(),
                                       "the optional attr row_block_size is missing in the tiling context; set the "
                                       "row_block_size attribute or rely on its registered default"),
                return ge::GRAPH_FAILED);
    tilingParam.blockSizeRow = static_cast<int64_t>(*attrBlockSizeRow);
    OP_CHECK_IF(tilingParam.blockSizeRow != BLOCK_SIZE_1,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "row_block_size",
                                          std::to_string(tilingParam.blockSizeRow), "1"),
                return ge::GRAPH_FAILED);

    auto* attrBlockSizeCol = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_BLOCK_SIZE_COL);
    OP_CHECK_IF(attrBlockSizeCol == nullptr,
                ReportTilingStageError(context->GetNodeName(),
                                       "the optional attr col_block_size is missing in the tiling context; set the "
                                       "col_block_size attribute or rely on its registered default"),
                return ge::GRAPH_FAILED);
    tilingParam.blockSizeCol = static_cast<int64_t>(*attrBlockSizeCol);
    OP_CHECK_IF(tilingParam.blockSizeCol != BLOCK_SIZE_128,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "col_block_size",
                                          std::to_string(tilingParam.blockSizeCol), "128"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockQuantI8::CheckShape(DynamicBlockQuantTilingParam& tilingParam)
{
    auto outputYPtr = context->GetOutputShape(0);
    OP_CHECK_IF(outputYPtr == nullptr,
                ReportTilingStageError(context->GetNodeName(),
                                       "the output shape of y is missing in the tiling context; check that InferShape "
                                       "executed and the framework propagated the output shapes"),
                return ge::GRAPH_FAILED);
    auto yShape = outputYPtr->GetStorageShape();

    auto scaleShapePtr = context->GetOutputShape(1);
    OP_CHECK_IF(scaleShapePtr == nullptr,
                ReportTilingStageError(context->GetNodeName(),
                                       "the output shape of scale is missing in the tiling context; check that "
                                       "InferShape executed and the framework propagated the output shapes"),
                return ge::GRAPH_FAILED);
    auto scaleShape = scaleShapePtr->GetStorageShape();

    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_IF(xShapePtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                         "the input shape of x is missing in the tiling context; check "
                                                         "that the graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    auto xShape = xShapePtr->GetStorageShape();

    OP_CHECK_IF(xShape != yShape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, y",
                                                       Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(yShape),
                                                       "The shapes of x and y must be the same"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        static_cast<int64_t>(xShape.GetDimNum() != DIM_TWO && xShape.GetDimNum() != DIM_THREE),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(xShape.GetDimNum()),
                                                 "The shape dim of x must be 2 or 3"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(static_cast<int64_t>(xShape.GetDimNum()) != static_cast<int64_t>(scaleShape.GetDimNum()),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "x, scale",
                    std::to_string(xShape.GetDimNum()) + ", " + std::to_string(scaleShape.GetDimNum()),
                    "The shape dims of x and scale must be the same"),
                return ge::GRAPH_FAILED);
    if (xShape.GetDimNum() == DIM_TWO) {
        OP_CHECK_IF(
            (static_cast<int64_t>(scaleShape.GetDim(0)) !=
             Ops::Base::CeilDiv(xShape.GetDim(0), tilingParam.blockSizeRow)) ||
                (static_cast<int64_t>(scaleShape.GetDim(1)) !=
                 Ops::Base::CeilDiv(xShape.GetDim(1), tilingParam.blockSizeCol)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "x, scale",
                Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(scaleShape),
                "The shape of scale must be [ceil(x_shape[0]/row_block_size), ceil(x_shape[1]/col_block_size)]"),
            return ge::GRAPH_FAILED);
    } else if (xShape.GetDimNum() == DIM_THREE) {
        OP_CHECK_IF(
            (static_cast<int64_t>(scaleShape.GetDim(0)) != xShape.GetDim(0)) ||
                (static_cast<int64_t>(scaleShape.GetDim(1)) !=
                 Ops::Base::CeilDiv(xShape.GetDim(1), tilingParam.blockSizeRow)) ||
                (static_cast<int64_t>(scaleShape.GetDim(DIM_TWO)) !=
                 Ops::Base::CeilDiv(xShape.GetDim(DIM_TWO), tilingParam.blockSizeCol)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, scale",
                                                   Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(scaleShape),
                                                   "The shape of scale must be [x_shape[0], "
                                                   "ceil(x_shape[1]/row_block_size), ceil(x_shape[2]/col_block_size)]"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicBlockQuantI8::DoTiling(DynamicBlockQuantTilingParam& tilingParam)
{
    auto xShapePtr = context->GetInputShape(0);
    OP_CHECK_IF(xShapePtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                         "the input shape of x is missing in the tiling context; check "
                                                         "that the graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    auto xShape = xShapePtr->GetStorageShape();
    CalcTilingData(tilingParam, xShape);

    // 获取输入/输出数据类型
    auto inputXPtr = context->GetInputDesc(0);
    OP_CHECK_IF(inputXPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                         "the input desc of x is missing in the tiling context; check "
                                                         "that the graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    auto inDtype = inputXPtr->GetDataType();
    auto outputYPtr = context->GetOutputDesc(0);
    OP_CHECK_IF(outputYPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "y",
                                                         "the output desc of y is missing in the tiling context; check "
                                                         "that the graph node provides all required outputs"),
                return ge::GRAPH_FAILED);
    auto outDtype = outputYPtr->GetDataType();
    CalcTilingKey(inDtype, tilingParam);
    return ge::GRAPH_SUCCESS;
}

void DynamicBlockQuantI8::CalcTilingData(DynamicBlockQuantTilingParam& tilingParam, const gert::Shape& xShape)
{
    if (xShape.GetDimNum() == DIM_TWO) {
        tilingParam.rowNum = xShape.GetDim(0);
        tilingParam.colNum = xShape.GetDim(1);
    } else {
        tilingParam.rowNum = xShape.GetDim(0) * xShape.GetDim(1);
        tilingParam.colNum = xShape.GetDim(DIM_TWO);
    }

    // 每一列的block数量
    int64_t singleRowBlockNum = Ops::Base::CeilDiv(tilingParam.colNum, tilingParam.blockSizeCol);
    // 先按行切分
    perCoreRowNum = tilingParam.rowNum / tilingParam.totalCoreNum;
    int64_t tailStartRowIdx = perCoreRowNum * tilingParam.totalCoreNum;

    int64_t tailRowNum = tilingParam.rowNum - tailStartRowIdx;

    // 计算剩余的block数量
    int64_t tailBlockNum = singleRowBlockNum * tailRowNum;

    // 尾块开始的行idx
    int64_t curRowIdx = tailStartRowIdx;
    int64_t curColBlockIdx = 0;
    // 若剩余block数量小于总核数
    if (tailBlockNum <= tilingParam.totalCoreNum) {
        for (int64_t i = 0; i < tailBlockNum; i++) {
            if (curColBlockIdx >= singleRowBlockNum) {
                curRowIdx++;
                curColBlockIdx = 0;
            }
            tailRowList[i] = static_cast<uint32_t>(curRowIdx);
            tailColBlockStartList[i] = static_cast<uint32_t>(curColBlockIdx);
            tailColBlockEndList[i] = static_cast<uint32_t>(++curColBlockIdx);
        }
        tilingParam.usedCoreNum = perCoreRowNum > 0 ? tilingParam.totalCoreNum : tailBlockNum;

        return;
    }

    // 尾块中每一行至少使用的核数
    int64_t tailRowCoreNum = tilingParam.totalCoreNum / tailRowNum;
    // 须额外多使用一个核的行数
    int64_t tailExtraCoreRowNum = tilingParam.totalCoreNum % tailRowNum;

    // 当前行开始位置使用的核总idx
    int64_t curRowCoreBaseIdx = 0;

    RowTilingData extraRowTilingData(tailRowCoreNum + 1, singleRowBlockNum);
    RowTilingData rowTilingData(tailRowCoreNum, singleRowBlockNum);
    RowTilingData* curRowTilingData = &extraRowTilingData;

    int64_t curCoreBlockNum = 0;
    for (int64_t i = 0; i < tailRowNum; i++) {
        curColBlockIdx = 0;
        curRowTilingData = i >= tailExtraCoreRowNum ? &rowTilingData : &extraRowTilingData;
        for (int64_t j = 0; j < curRowTilingData->coreNum; j++) {
            tailRowList[curRowCoreBaseIdx + j] = static_cast<uint32_t>(tailStartRowIdx + i);
            curCoreBlockNum = curRowTilingData->GetCurCoreBlockNum(j);
            tailColBlockStartList[curRowCoreBaseIdx + j] = static_cast<uint32_t>(curColBlockIdx);
            curColBlockIdx += curCoreBlockNum;
            tailColBlockEndList[curRowCoreBaseIdx + j] = static_cast<uint32_t>(curColBlockIdx);
        }
        curRowCoreBaseIdx += curRowTilingData->coreNum;
    }

    tilingParam.usedCoreNum = tilingParam.totalCoreNum;
}

void DynamicBlockQuantI8::CalcTilingKey(DataType inputType, DynamicBlockQuantTilingParam& tilingParam)
{
    // 十分位为0、1，分别表示输入为float16，bfloat16
    int64_t tenDigit = inputType == DT_FLOAT16 ? 0 : 1;
    int64_t digit = 0;
    tilingParam.tilingKey = tenDigit * NUM_TEN + digit;
}

void DynamicBlockQuantI8::SetTilingData(DynamicBlockQuantTilingData& tilingData,
                                        const DynamicBlockQuantTilingParam& tilingParam)
{
    tilingData.set_tilingKey(tilingParam.tilingKey);
    tilingData.set_totalCoreNum(tilingParam.totalCoreNum);
    tilingData.set_ubSize(tilingParam.ubSize);
    tilingData.set_minScale(tilingParam.minScale);
    tilingData.set_roundMode(tilingParam.roundMode);
    tilingData.set_dstType(tilingParam.dstType);
    tilingData.set_blockSizeRow(tilingParam.blockSizeRow);
    tilingData.set_blockSizeCol(tilingParam.blockSizeCol);
    tilingData.set_dstTypeMax(tilingParam.dstTypeMax);
    tilingData.set_rowNum(tilingParam.rowNum);
    tilingData.set_colNum(tilingParam.colNum);
    tilingData.set_usedCoreNum(tilingParam.usedCoreNum);
    tilingData.set_tailRowList(tailRowList);
    tilingData.set_tailColBlockStartList(tailColBlockStartList);
    tilingData.set_tailColBlockEndList(tailColBlockEndList);
    tilingData.set_perCoreRowNum(perCoreRowNum);
}
} // namespace dynamic_block_quant_i8
} // namespace optiling
