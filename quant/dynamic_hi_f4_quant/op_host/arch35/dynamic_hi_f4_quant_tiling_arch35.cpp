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
 * \file dynamic_hi_f4_quant_tiling.cpp
 * \brief
 */
#include "dynamic_hi_f4_quant_tiling_arch35.h"
#include "../../op_kernel/arch35/dynamic_hi_f4_quant_tilingdata.h"
#include <cmath>
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "../../op_kernel/arch35/dynamic_hi_f4_quant_struct.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "../../../../common/inc/op_host/op_const_def_nn.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace DynamicHiF4QuantOp;
namespace optiling {
constexpr int64_t SECOND_LAST = -2;
constexpr int64_t INDEX_ATTR_AXIS = 0;
constexpr int64_t INDEX_ATTR_SCALE_ALG = 1;
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
constexpr int64_t BLOCK_SIZE = 64;
constexpr int64_t BLOCK_PER_GROUP = 1;
constexpr int64_t UINT32_BYTES_SIZE = 4;
constexpr int64_t UINT16_BYTES_SIZE = 2;
constexpr int64_t UINT8_BYTES_SIZE = 1;
constexpr int64_t BYTES_OF_INPUT_TYPE = 2;
constexpr int64_t BYTES_OF_TMP_TYPE = 1;
constexpr int64_t MAX_BYTES_OF_OUTPUT_TYPE = 1;
constexpr int64_t BYTES_OF_HIFLOAT4 = 1; // 使用的时候除以2
constexpr int64_t BYTES_OF_HIFLOAT4_SCALE = 4;
constexpr size_t MAX_DIM_NUM = 7;

constexpr int64_t RESERVED_UB_SIZE = 4096;

constexpr int64_t HIFP4_PER_CORE_MIN_GROUP_NUM = 32; // 换算字节数： HIFP4_PER_CORE_MIN_GROUP_NUM * 64(group elements
                                                     // num) * 2(dtype width)
constexpr int64_t HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM = 8;
constexpr int64_t HIFP4_OUTPUT_REG_ONELOOP = 2;
constexpr int64_t HIFP4_MAXPER8_REG_ONELOOP = 16;
constexpr int64_t HIFP4_DB_NUM = 2; // 1: 不开db  2: 开db
constexpr int64_t HIFP4_NUM_4 = 4;
constexpr int64_t HIFP4_MID_UB_RES_NUM = 6;
constexpr int64_t HIFP4_UB_BLOCK_SIZE = 32;               // unit: byte
constexpr int64_t HIFP4_ONE_GROUP_INPUT_BYTES = 64 * 2;   // only fp16 bf16 now
constexpr int64_t HIFP4_ONE_GROUP_OUTPUT_BYTES = 32;      // fp4: 64 * 0.5
constexpr int64_t HIFP4_ONE_GROUP_OUTPUT_SCALE_BYTES = 4; // scale is fp32
constexpr int64_t HIFP4_ONE_GROUP_MID_UB_RES_BYTES = 119; // E8:16bit(性能考虑，u16存储) E1_8:8bit E1_16:16bit
                                                          // Maxper4:32B Maxper8:16B Maxper64:2B scale:64B
constexpr int64_t HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM = 256; // scale 输出的计算，一次reg循环计算256个group的scale

const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_HIFLOAT4};
const std::set<ge::DataType> Y_SUPPORT_DTYPE_FP4_SET = {ge::DT_HIFLOAT4};
const std::set<ge::DataType> OUTPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT, Ops::NN::DT_HIFLOAT4_SCALE};

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

ge::graphStatus DynamicHiF4QuantTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto* attrAxis = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrAxis);
    tilingParams.axis = static_cast<int64_t>(*attrAxis);
    OP_LOGD(context_->GetNodeName(), "The attr axis is %ld", tilingParams.axis);
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    tilingParams.axis = tilingParams.axis < 0 ? tilingParams.axis : tilingParams.axis - xShape.GetDimNum();
    OP_CHECK_IF(tilingParams.axis != -1 && tilingParams.axis != -2,
                OP_LOGE(context_->GetNodeName(), "The axis should be -1/-2, please check."), return ge::GRAPH_FAILED);

    auto* attrScaleAlg = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_SCALE_ALG);
    int64_t scaleAlg = 0;
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

ge::graphStatus DynamicHiF4QuantTiling::CheckDtype() const
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

    auto outputScalePtr = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputScalePtr);
    auto scaleDtype = outputScalePtr->GetDataType();
    OP_CHECK_IF(OUTPUT_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
                OP_LOGE(context_->GetNodeName(),
                        "Output scale's data type only support FLOAT32 currently, but scale is [%s], please check.",
                        Ops::Base::ToString(scaleDtype).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicHiF4QuantTiling::CheckShape() const
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape.GetDimNum() < 2 || xShape.GetDimNum() > MAX_DIM_NUM,
        OP_LOGE(context_->GetNodeName(), "The shape is invalid, Input x rank[%zu] should be in [2, 7], please check.",
                static_cast<int64_t>(xShape.GetDimNum())),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();

    OP_CHECK_IF(xShape.GetDim(xShape.GetDimNum() - 1) % DIGIT_TWO != 0 && Y_SUPPORT_DTYPE_FP4_SET.count(yDtype) != 0,
                OP_LOGE(context_->GetNodeName(),
                        "When output y's data type is HIFLOAT4, the last axis must be a multiple of 64, please check."),
                return ge::GRAPH_FAILED);

    auto yShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(xShape != yShape,
                OP_LOGE(context_->GetNodeName(),
                        "The shape of output y[%s] must be same with shape of input x[%s], please check.",
                        Shape2String(yShape).c_str(), Shape2String(xShape).c_str()),
                return ge::GRAPH_FAILED);

    auto ScaleShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ScaleShapePtr);
    auto ScaleShape = ScaleShapePtr->GetStorageShape();

    auto newScaleShape = xShape;
    int64_t lastDim = xShape.GetDim(xShape.GetDimNum() - 1);
    int64_t nonLastDim = xShape.GetDim(xShape.GetDimNum() - 2);
    if (tilingParams.axis == -1) {
        newScaleShape.SetDim(xShape.GetDimNum() - 1,
                             Ops::Base::CeilDiv(xShape.GetDim(xShape.GetDimNum() - 1), DIGIT_64));
        newScaleShape.AppendDim(DIGIT_ONE);
        OP_CHECK_IF(newScaleShape != ScaleShape,
                    OP_LOGE(context_->GetNodeName(),
                            "The shape of output scale [%s] is incorrect, correct shape is [%s], please check.",
                            Shape2String(ScaleShape).c_str(), Shape2String(newScaleShape).c_str()),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            lastDim % DIGIT_64 != 0,
            OP_LOGE(context_->GetNodeName(),
                    "When the quantization axis is the last axis, the size of the last axis must be a multiple of 64."
                    "The current last axis size is [%lld] please check.",
                    lastDim),
            return ge::GRAPH_FAILED);
    } else if (tilingParams.axis == SECOND_LAST) {
        newScaleShape.SetDim(xShape.GetDimNum() - DIGIT_TWO,
                             Ops::Base::CeilDiv(xShape.GetDim(xShape.GetDimNum() - DIGIT_TWO), DIGIT_64));
        newScaleShape.AppendDim(DIGIT_ONE);
        OP_CHECK_IF(newScaleShape != ScaleShape,
                    OP_LOGE(context_->GetNodeName(),
                            "The shape of output scale [%s] is incorrect, correct shape is [%s], please check.",
                            Shape2String(ScaleShape).c_str(), Shape2String(newScaleShape).c_str()),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            lastDim % DIGIT_64 != 0 || nonLastDim % DIGIT_64 != 0,
            OP_LOGE(
                context_->GetNodeName(),
                "When the quantization axis is axis -2, the sizes of both axis -2 and the last axis must be multiples "
                "of 64."
                "The current axis -1 is [%lld] and axis -2 is [%lld], please check.",
                lastDim, nonLastDim),
            return ge::GRAPH_FAILED);
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicHiF4QuantTiling::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicHiF4QuantTiling GetPlatformInfo.");

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParams.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParams.totalCoreNum <= 0), OP_LOGE(context_->GetNodeName(), "Failed to core num."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParams.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((tilingParams.ubSize <= 0), OP_LOGE(context_->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    tilingParams.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

// 合轴
ge::graphStatus DynamicHiF4QuantTiling::MergeAxis()
{
    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    for (size_t i = 0; i < xShape.GetDimNum() - DIGIT_TWO; i++) {
        tilingParams.dim0 *= xShape.GetDim(i);
    }
    tilingParams.dimNeg2 = xShape.GetDim(xShape.GetDimNum() - DIGIT_TWO);
    tilingParams.dimNeg1 = xShape.GetDim(xShape.GetDimNum() - 1);

    return ge::GRAPH_SUCCESS;
}

// 切块
void DynamicHiF4QuantTiling::SplitCore(int64_t blockW, int64_t blockSize)
{
    uint64_t xUbSize = blockSize * blockW * BYTES_OF_INPUT_TYPE;
    uint64_t yUbSize = blockSize * blockW * BYTES_OF_HIFLOAT4 / DIGIT_TWO;
    uint64_t scaleUbSize = blockW * BYTES_OF_HIFLOAT4_SCALE;
    uint64_t tmpscaleUbSize = blockSize * blockW / DIGIT_FOUR * BYTES_OF_INPUT_TYPE;
    uint64_t maxper4UbSize = blockSize * blockW / DIGIT_FOUR * BYTES_OF_INPUT_TYPE;
    uint64_t maxper8UbSize = blockSize * blockW / DIGIT_EIGHT * BYTES_OF_INPUT_TYPE;
    uint64_t e6m2UbSize = blockW * BYTES_OF_INPUT_TYPE;
    uint64_t e1_8UbSize = blockW * BYTES_OF_INPUT_TYPE;
    uint64_t e1_16UbSize = blockW * BYTES_OF_INPUT_TYPE * DIGIT_TWO;

    tilingParams.groupPerUb = (tilingParams.ubSize - RESERVED_UB_SIZE) / N_BUFFER /
                              (xUbSize + yUbSize + scaleUbSize + tmpscaleUbSize + maxper4UbSize + maxper8UbSize +
                               e6m2UbSize + e1_8UbSize + e1_16UbSize);

    tilingParams.splitBlockH = blockSize * tilingParams.groupPerUb;
    tilingParams.dimNeg2SplitBlockNum = Ops::Base::CeilDiv(tilingParams.dimNeg2, tilingParams.splitBlockH);
    tilingParams.dimNeg1BlockNum = Ops::Base::CeilDiv(tilingParams.dimNeg1, blockW);
    tilingParams.totalTaskNum = tilingParams.dim0 * tilingParams.dimNeg2SplitBlockNum * tilingParams.dimNeg1BlockNum;
}

ge::graphStatus DynamicHiF4QuantTiling::SetTilingParamsNeg2()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicHiF4QuantTiling SetTilingParams.");

    MergeAxis();

    tilingParams.blockSize = BLOCK_SIZE;
    tilingParams.blockW = DIGIT_128;
    SplitCore(tilingParams.blockW, tilingParams.blockSize);
    tilingParams.dimNeg2Tail = tilingParams.dimNeg2 % tilingParams.splitBlockH == 0 ?
                                   tilingParams.splitBlockH :
                                   tilingParams.dimNeg2 % tilingParams.splitBlockH;
    tilingParams.dimNeg1Tail = tilingParams.dimNeg1 % tilingParams.blockW == 0 ?
                                   tilingParams.blockW :
                                   tilingParams.dimNeg1 % tilingParams.blockW;

    tilingParams.dimNeg1IsPad = tilingParams.dimNeg1 % tilingParams.blockSize == 0 ? 0 : 1; // 0
    tilingParams.dimNeg1IsOdd = Ops::Base::CeilDiv(tilingParams.dimNeg1, tilingParams.blockSize) % DIGIT_TWO == 0 ?
                                    0 :
                                    1; // 尾轴scale奇数列补0 0
    tilingParams.dimNeg2IsOdd = Ops::Base::CeilDiv(tilingParams.dimNeg2, tilingParams.blockSize) % DIGIT_TWO == 0 ?
                                    0 :
                                    1; // 非尾轴scale奇数行补0  0

    tilingParams.blockCountPerBatch = tilingParams.dimNeg2SplitBlockNum * tilingParams.dimNeg1BlockNum;
    tilingParams.scale1ColCountPerBatch = Ops::Base::CeilDiv(tilingParams.dimNeg1, tilingParams.blockSize);
    tilingParams.scale2RowCountPerBatch = Ops::Base::CeilDiv(tilingParams.dimNeg2, tilingParams.blockSize);

    tilingParams.usedCoreNum = std::min(tilingParams.totalCoreNum, tilingParams.totalTaskNum);
    tilingParams.blockPerHeadCore = Ops::Base::CeilDiv(tilingParams.totalTaskNum, tilingParams.totalCoreNum);
    tilingParams.blockPerTailCore = tilingParams.totalTaskNum / tilingParams.totalCoreNum;
    tilingParams.headCoreNum = tilingParams.totalTaskNum % tilingParams.totalCoreNum;

    return ge::GRAPH_SUCCESS;
}

int64_t DynamicHiF4QuantTiling::HiF4GetUbFormerNeg1()
{
    // unit：byte  fp4输出，因为ubGroupFormer 16对齐，其que大小天然对齐的。
    // 中间结果E8 E1_16 E1_8要先保存ub，再计算scale输出，三个中间结果32B对齐
    // scale输出，是4个VL交织出来的，输出需要256B * 4 对齐
    int64_t vlByteSize = static_cast<int64_t>(Ops::Base::GetVRegSize(context_)); // 这里有问题
    int64_t ubReserveAlignBuf = HIFP4_MID_UB_RES_NUM * HIFP4_UB_BLOCK_SIZE * HIFP4_DB_NUM +
                                HIFP4_DB_NUM * vlByteSize * DIGIT_FOUR + RESERVED_UB_SIZE * 4;
    int64_t ubCap = (tilingParams.ubSize - ubReserveAlignBuf) / HIFP4_DB_NUM /
                    (HIFP4_ONE_GROUP_MID_UB_RES_BYTES + HIFP4_ONE_GROUP_INPUT_BYTES + HIFP4_ONE_GROUP_OUTPUT_BYTES +
                     HIFP4_ONE_GROUP_OUTPUT_SCALE_BYTES); // 每64个数计算中所有输出和中间变量占用空间的字节数
    return ubCap / HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM * HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM; //
}

ge::graphStatus DynamicHiF4QuantTiling::SetTilingParamsNeg1()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicHiF4QuantTiling SetTilingParamsNeg1.");
    const gert::StorageShape* xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    uint64_t totalshape = xShape.GetShapeSize();

    tilingParams.groupNum = static_cast<int64_t>(Ops::Base::CeilDiv(totalshape, static_cast<uint64_t>(DIGIT_64))); //
    tilingParams.usedCoreNum = Ops::Base::CeilDiv(tilingParams.groupNum,
                                                  static_cast<int64_t>(HIFP4_PER_CORE_MIN_GROUP_NUM));
    tilingParams.usedCoreNum = tilingParams.usedCoreNum > tilingParams.totalCoreNum ? tilingParams.totalCoreNum :
                                                                                      tilingParams.usedCoreNum;

    tilingParams.ubFormer = HiF4GetUbFormerNeg1(); // 单位是group个数

    tilingParams.blockFormer = Ops::Base::CeilDiv(tilingParams.groupNum,
                                                  tilingParams.usedCoreNum); // 分核 整核处理多少group
    tilingParams.usedCoreNum = static_cast<int64_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(tilingParams.groupNum), tilingParams.blockFormer));
    tilingParams.blockTail = tilingParams.groupNum -
                             (tilingParams.usedCoreNum - 1) * tilingParams.blockFormer; // 分核 尾核处理多少group
    // yztodo : 这四个也要用int64传递
    tilingParams.ubLoopOfFormerBlock = Ops::Base::CeilDiv(tilingParams.blockFormer,
                                                          tilingParams.ubFormer); // 在ub上刷几次数据
    tilingParams.ubLoopOfTailBlock = Ops::Base::CeilDiv(tilingParams.blockTail, tilingParams.ubFormer);
    tilingParams.ubTailOfFormerBlock = tilingParams.blockFormer -
                                       (tilingParams.ubLoopOfFormerBlock - 1) * tilingParams.ubFormer; // group的数量
    tilingParams.ubTailOfTailBlock = tilingParams.blockTail -
                                     (tilingParams.ubLoopOfTailBlock - 1) * tilingParams.ubFormer;

    tilingParams.ubFormerRegLoop = Ops::Base::CeilDiv(tilingParams.ubFormer, HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM);
    tilingParams.ubTailOfFormerRegLoop = Ops::Base::CeilDiv(tilingParams.ubTailOfFormerBlock,
                                                            HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM); // 正常核尾块
    tilingParams.ubTailOfTailRegLoop = Ops::Base::CeilDiv(tilingParams.ubTailOfTailBlock,
                                                          HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM); // 尾核尾块

    tilingParams.ubMaxPer8RegLoop = Ops::Base::CeilDiv(tilingParams.ubFormer, HIFP4_MAXPER8_REG_ONELOOP);
    tilingParams.ubTailOfMaxPer8RegLoop = Ops::Base::CeilDiv(tilingParams.ubTailOfFormerBlock,
                                                             HIFP4_MAXPER8_REG_ONELOOP); // 正常核尾块
    tilingParams.ubTailOfTailMaxPer8RegLoop = Ops::Base::CeilDiv(tilingParams.ubTailOfTailBlock,
                                                                 HIFP4_MAXPER8_REG_ONELOOP); // 尾核尾块

    tilingParams.ubFormerScaleRegLoop = Ops::Base::CeilDiv(tilingParams.ubFormer,
                                                           HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM);
    tilingParams.ubTailOfFormerScaleRegLoop = Ops::Base::CeilDiv(tilingParams.ubTailOfFormerBlock,
                                                                 HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM);
    tilingParams.ubTailOfTailScaleRegLoop = Ops::Base::CeilDiv(tilingParams.ubTailOfTailBlock,
                                                               HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM);

    tilingParams.ubFormerOutRegLoop = Ops::Base::CeilDiv(tilingParams.ubFormer, HIFP4_OUTPUT_REG_ONELOOP);
    tilingParams.ubTailOfOutRegLoop = Ops::Base::CeilDiv(tilingParams.ubTailOfFormerBlock, HIFP4_OUTPUT_REG_ONELOOP);
    tilingParams.ubTailOfTailOutRegLoop = Ops::Base::CeilDiv(tilingParams.ubTailOfTailBlock, HIFP4_OUTPUT_REG_ONELOOP);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DynamicHiF4QuantTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter DynamicHiF4QuantTiling DoTiling.");

    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The platforminfo get failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The data type check failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The attr get failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The shape check failed."),
                return ge::GRAPH_FAILED);

    if (tilingParams.axis == -1) {
        SetTilingParamsNeg1();
        SetTilingKey();
        SetTilingDataNeg1();
    } else {
        SetTilingParamsNeg2();
        SetTilingKey();
        SetTilingDataNeg2();
    }
    PrintTilingData();

    uint64_t tilingDataSize = sizeof(tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    auto rawTilingData = context_->GetRawTilingData();
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), reinterpret_cast<void*>(&tilingData),
                           tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %lu.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParams.workspaceSize;

    return ge::GRAPH_SUCCESS;
}

void DynamicHiF4QuantTiling::SetTilingKey()
{
    if (tilingParams.axis == -1) {
        quantAxis_ = TPL_QUANT_TAIL_AXIS;
    } else {
        quantAxis_ = TPL_QUANT_NO_TAIL_AXIS;
    }
    tilingParams.tilingKey = GET_TPL_TILING_KEY(quantAxis_);
    context_->SetTilingKey(tilingParams.tilingKey);
}

void DynamicHiF4QuantTiling::SetTilingDataNeg1()
{
    tilingData.totalCoreNum = tilingParams.totalCoreNum;
    tilingData.usedCoreNum = tilingParams.usedCoreNum;
    tilingData.groupNum = tilingParams.blockFormer;
    tilingData.alignGroupNum = tilingParams.ubFormer;
    tilingData.rowPerHeadCore = tilingParams.ubLoopOfFormerBlock;
    tilingData.rowPerTailCore = tilingParams.ubLoopOfTailBlock;
    tilingData.multiRowNumHeadCore = tilingParams.ubTailOfFormerBlock;
    tilingData.multiRowNumTailCore = tilingParams.ubTailOfTailBlock;
    tilingData.innerLoopEle = tilingParams.ubFormerRegLoop;
    tilingData.innerLoopTimes = tilingParams.ubTailOfFormerRegLoop;
    tilingData.innerLoopTail = tilingParams.ubTailOfTailRegLoop;
    tilingData.max8LoopEle = tilingParams.ubMaxPer8RegLoop;
    tilingData.max8LoopTimes = tilingParams.ubTailOfMaxPer8RegLoop;
    tilingData.max8LoopTail = tilingParams.ubTailOfTailMaxPer8RegLoop;
    tilingData.scaleLoopEle = tilingParams.ubFormerScaleRegLoop;
    tilingData.scaleLoopTimes = tilingParams.ubTailOfFormerScaleRegLoop;
    tilingData.scaleLoopTail = tilingParams.ubTailOfTailScaleRegLoop;
    tilingData.outLoopEle = tilingParams.ubFormerOutRegLoop;
    tilingData.outLoopTimes = tilingParams.ubTailOfOutRegLoop;
    tilingData.outLoopTail = tilingParams.ubTailOfTailOutRegLoop;
    tilingData.scaleAlg = tilingParams.scaleAlg;
}

void DynamicHiF4QuantTiling::SetTilingDataNeg2()
{
    tilingData.totalCoreNum = tilingParams.totalCoreNum;
    tilingData.usedCoreNum = tilingParams.usedCoreNum;
    tilingData.dim0 = tilingParams.dim0;
    tilingData.dimNeg2 = tilingParams.dimNeg2;
    tilingData.dimNeg1 = tilingParams.dimNeg1;
    tilingData.blockW = tilingParams.blockW;
    tilingData.tilingKey = tilingParams.tilingKey;
    tilingData.splitBlockH = tilingParams.splitBlockH;
    tilingData.dimNeg2Tail = tilingParams.dimNeg2Tail;
    tilingData.dimNeg1Tail = tilingParams.dimNeg1Tail;
    tilingData.dimNeg2SplitBlockNum = tilingParams.dimNeg2SplitBlockNum;
    tilingData.dimNeg1BlockNum = tilingParams.dimNeg1BlockNum;
    tilingData.blockPerHeadCore = tilingParams.blockPerHeadCore;
    tilingData.blockPerTailCore = tilingParams.blockPerTailCore;
    tilingData.headCoreNum = tilingParams.headCoreNum;
    tilingData.dimNeg2IsOdd = tilingParams.dimNeg2IsOdd;
    tilingData.dimNeg1IsOdd = tilingParams.dimNeg1IsOdd;
    tilingData.dimNeg1IsPad = tilingParams.dimNeg1IsPad;
    tilingData.blockCountPerBatch = tilingParams.blockCountPerBatch;
    tilingData.scale1ColCountPerBatch = tilingParams.scale1ColCountPerBatch;
    tilingData.scale2RowCountPerBatch = tilingParams.scale2RowCountPerBatch;
    tilingData.scaleAlg = tilingParams.scaleAlg;
}

void DynamicHiF4QuantTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(),
            "TilingData totalCoreNum: %ld, usedCoreNum: %ld, "
            "dim0: %ld, dimNeg2: %ld, dimNeg1: %ld, blockW: %ld, "
            "tilingKey: %ld, splitBlockH: %ld, dimNeg2Tail: %ld, "
            "dimNeg1Tail: %ld, dimNeg2SplitBlockNum: %ld, dimNeg1BlockNum: %ld, "
            "blockPerHeadCore: %ld, blockPerTailCore: %ld, headCoreNum: %ld, "
            "dimNeg2IsOdd: %ld, dimNeg1IsOdd: %ld, dimNeg1IsPad: %ld, blockCountPerBatch: %ld, "
            "scale1ColCountPerBatch: %ld, scale2RowCountPerBatch: %ld, "
            "rowPerHeadCore: %ld, rowPerTailCore: %ld, "
            "multiRowNumHeadCore: %ld, multiRowNumTailCore: %ld, "
            "groupNum: %ld, alignGroupNum: %ld, "
            "innerLoopEle: %ld, innerLoopTimes: %ld, innerLoopTail: %ld, "
            "max8LoopEle: %ld, max8LoopTimes: %ld, max8LoopTail: %ld, "
            "scaleLoopEle: %ld, outLoopTimes: %ld, scaleLoopTail: %ld, "
            "outLoopEle: %ld, outLoopTimes: %ld, outLoopTail: %ld, scaleAlg: %ld, ",
            tilingData.totalCoreNum, tilingData.usedCoreNum, tilingData.dim0, tilingData.dimNeg2, tilingData.dimNeg1,
            tilingData.blockW, tilingData.tilingKey, tilingData.splitBlockH, tilingData.dimNeg2Tail,
            tilingData.dimNeg1Tail, tilingData.dimNeg2SplitBlockNum, tilingData.dimNeg1BlockNum,
            tilingData.blockPerHeadCore, tilingData.blockPerTailCore, tilingData.headCoreNum, tilingData.dimNeg2IsOdd,
            tilingData.dimNeg1IsOdd, tilingData.dimNeg1IsPad, tilingData.blockCountPerBatch,
            tilingData.scale1ColCountPerBatch, tilingData.scale2RowCountPerBatch, tilingData.rowPerHeadCore,
            tilingData.rowPerTailCore, tilingData.multiRowNumHeadCore, tilingData.multiRowNumTailCore,
            tilingData.groupNum, tilingData.alignGroupNum, tilingData.innerLoopEle, tilingData.innerLoopTimes,
            tilingData.innerLoopTail, tilingData.max8LoopEle, tilingData.max8LoopTimes, tilingData.max8LoopTail,
            tilingData.scaleLoopEle, tilingData.scaleLoopTimes, tilingData.outLoopEle, tilingData.outLoopEle,
            tilingData.outLoopTimes, tilingData.outLoopTail, tilingData.scaleAlg);
}

static ge::graphStatus TilingForDynamicHiF4Quant(gert::TilingContext* context_)
{
    OP_LOGD("DynamicHiF4QuantTiling", "Enter TilingForDynamicHiF4QuantTiling");
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("DynamicHiF4QuantTiling", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    optiling::DynamicHiF4QuantTiling hifloat4Tiling(context_);
    return hifloat4Tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForDynamicHiF4Quant(gert::TilingParseContext* context_)
{
    OP_LOGD("DynamicHiF4QuantTiling", "Enter TilingPrepareForDynamicHiF4QuantTiling");
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("DynamicHiF4QuantTiling", "TilingParse context is null."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DynamicHiF4Quant)
    .Tiling(TilingForDynamicHiF4Quant)
    .TilingParse<DynamicHiF4QuantCompileInfo>(TilingPrepareForDynamicHiF4Quant);
} // namespace optiling
