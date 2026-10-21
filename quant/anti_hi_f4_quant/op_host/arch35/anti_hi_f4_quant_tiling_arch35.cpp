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
 * \file anti_hi_f4_quant_tiling_arch35.cpp
 * \brief
 */

#include "anti_hi_f4_quant_tiling_arch35.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "../../op_kernel/arch35/anti_hi_f4_quant_tilingdata.h"
#include "../../op_kernel/arch35/anti_hi_f4_quant_struct.h"
#include "../../../../common/inc/op_host/op_const_def_nn.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace AntiHiF4QuantOp;

namespace optiling {
constexpr int64_t INDEX_INPUT_X = 0;
constexpr int64_t INDEX_INPUT_SCALE = 1;
constexpr int64_t INDEX_OUTPUT_Y = 0;
constexpr size_t INDEX_ATTR_DST_TYPE = 0;
constexpr int64_t BLOCK_SIZE = 64;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t X_NUM_OF_ONE_BYTE = 2;
constexpr int64_t MIN_DIM_NUM = 2;
constexpr int64_t MAX_DIM_NUM = 7;
constexpr int64_t BYTE_OF_INPUT_SCALE = 4;
constexpr int64_t BYTE_OF_OUTPUT_Y = 2;
const int64_t DST_TYPE_FLOAT16 = 1;
const int64_t DST_TYPE_BF16 = 27;

constexpr int64_t N_BUFFER = 2;

const std::set<ge::DataType> INPUT_X_SUPPORT_DTYPE_SET = {ge::DT_HIFLOAT4};
const std::set<ge::DataType> INPUT_SCALE_SUPPORT_DTYPE_SET = {Ops::NN::DT_HIFLOAT4_SCALE};

/*
 * CheckDtype - 校验输入/输出张量的数据类型
 *   x:     必须为 DT_HIFLOAT4 (S1P2 编码 4-bit)
 *   scale: 必须为 DT_HIFLOAT4_SCALE (三层缩放因子, float32 物理存储)
 *   y:     必须为 DT_FLOAT16 或 DT_BF16
 */
static ge::graphStatus CheckDtype(const gert::TilingContext* context)
{
    auto inputXPtr = context->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto xDtype = inputXPtr->GetDataType();
    OP_CHECK_IF(INPUT_X_SUPPORT_DTYPE_SET.count(xDtype) == 0,
                OP_LOGE(context->GetNodeName(),
                        "Input x\'s data type only support HIFLOAT4 currently, but x is [%s], please check.",
                        Ops::Base::ToString(xDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto inputScalePtr = context->GetInputDesc(INDEX_INPUT_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputScalePtr);
    auto scaleDtype = inputScalePtr->GetDataType();
    OP_CHECK_IF(
        INPUT_SCALE_SUPPORT_DTYPE_SET.count(scaleDtype) == 0,
        OP_LOGE(context->GetNodeName(),
                "Input scale\'s data type only support HIFLOAT4_SCALE currently, but scale is [%s], please check.",
                Ops::Base::ToString(scaleDtype).c_str()),
        return ge::GRAPH_FAILED);

    auto outputYPtr = context->GetOutputDesc(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();
    OP_CHECK_IF(
        yDtype != ge::DT_FLOAT16 && yDtype != ge::DT_BF16,
        OP_LOGE(context->GetNodeName(),
                "Output y\'s data type only support FLOAT16 and BFLOAT16 currently, but y is [%s], please check.",
                Ops::Base::ToString(yDtype).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * CheckShape - 校验 x/scale/y 的 shape 合法性, 并推导 quantMode
 *
 * 正向算子 DynamicHiF4Quant 的 scale 输出 shape 规则 (infershape):
 *   scale = x_shape (复制)
 *   scale[axis] = ceil(x[axis] / 64)   (量化轴被压缩为 block 粒度)
 *   scale.AppendDim(1)                  (末尾追加对齐维)
 *
 * 因此 scale 恒为 rank(x)+1 维, scale[-1] == 1:
 *   axis=-1 (PER_CHANNEL): scale = [x[0],..., x[-3], ceil(x[-1]/64), 1]
 *   axis=-2 (PER_HEAD):    scale = [x[0],..., ceil(x[-2]/64), x[-1], 1]
 *
 * 本函数通过 scale[-2] 的值反推量化轴:
 *   scale[-2] == x[-1]             → axis=-2 (PER_HEAD)
 *   scale[-2] == ceil(x[-1] / 64)  → axis=-1 (PER_CHANNEL)
 */
static ge::graphStatus CheckShape(AntiHiF4QuantTilingParam& tilingParams, const gert::TilingContext* context)
{
    auto xShapePtr = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    // x 维度范围 [2, 7]
    OP_CHECK_IF(xShape.GetDimNum() < MIN_DIM_NUM || xShape.GetDimNum() > MAX_DIM_NUM,
                OP_LOGE(context->GetNodeName(), "Input x rank[%zu] should be in [%lld, %lld], please check.",
                        xShape.GetDimNum(), MIN_DIM_NUM, MAX_DIM_NUM),
                return ge::GRAPH_FAILED);

    tilingParams.xDimNeg1 = xShape.GetDim(xShape.GetDimNum() - 1);

    // x -1 轴必须为 64 的倍数: HiF4 以 64 元素为最小 Block 粒度
    OP_CHECK_IF(tilingParams.xDimNeg1 % BLOCK_SIZE != 0,
                OP_LOGE(context->GetNodeName(),
                        "The last axis must be a multiple of 64, the current last axis size is [%lld] please check.",
                        tilingParams.xDimNeg1),
                return ge::GRAPH_FAILED);

    tilingParams.xDimNeg2 = xShape.GetDim(xShape.GetDimNum() - NUM_TWO);
    // x -2 轴也必须为 64 的倍数
    // 原因: axis=-2 量化模式下, scale 以 64 为粒度对 x[-2] 进行分组,
    // 因此 x[-2] 必须能被 BLOCK_SIZE 整除, 否则无法正确索引 scale
    OP_CHECK_IF(tilingParams.xDimNeg2 % BLOCK_SIZE != 0,
                OP_LOGE(context->GetNodeName(),
                        "The second last axis must be a multiple of 64, the current size is [%lld], please check.",
                        tilingParams.xDimNeg2),
                return ge::GRAPH_FAILED);

    auto scaleShapePtr = context->GetInputShape(INDEX_INPUT_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShapePtr);
    auto scaleShape = scaleShapePtr->GetStorageShape();

    // scale 维度范围: [MIN_DIM_NUM+1, MAX_DIM_NUM+1] = [3, 8]
    // 因为 scale.ndim == x.ndim + 1 (来自 DynamicHiF4Quant 的 AppendDim(1))
    OP_CHECK_IF(scaleShape.GetDimNum() < MIN_DIM_NUM + 1 || scaleShape.GetDimNum() > MAX_DIM_NUM + 1,
                OP_LOGE(context->GetNodeName(), "Input scale rank[%zu] should be in [%lld, %lld], please check.",
                        scaleShape.GetDimNum(), MIN_DIM_NUM + 1, MAX_DIM_NUM + 1),
                return ge::GRAPH_FAILED);

    // scale 末维固定为1，由正向算子 DynamicHiF4Quant 的 AppendDim(1) 产生
    OP_CHECK_IF(scaleShape.GetDim(scaleShape.GetDimNum() - 1) != 1,
                OP_LOGE(context->GetNodeName(),
                        "scale shape [%s] last dim should be 1 (from DynamicHiF4Quant AppendDim), please check.",
                        Shape2String(scaleShape).c_str()),
                return ge::GRAPH_FAILED);

    // scale 与 x 的维度关系: scale.ndim == x.ndim + 1
    OP_CHECK_IF(scaleShape.GetDimNum() != xShape.GetDimNum() + 1,
                OP_LOGE(context->GetNodeName(), "scale rank[%zu] should be x rank[%zu] + 1, please check.",
                        scaleShape.GetDimNum(), xShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // scale 必须与 x 共享前 n-2 维 (两种 axis 下都成立)
    // axis=-1: scale = [x[0],...,x[-3], ceil(x[-1]/64), 1] → 前 n-2 维完全一致
    // axis=-2: scale = [x[0],...,ceil(x[-2]/64), x[-1], 1] → 前 n-3 维一致
    // 统一检查: 跳过 scale 的最后两维及末尾 AppendDim(1), 前 n-2 维必须匹配 x 对应位置
    OP_CHECK_IF(
        xShape.GetDimNum() > 2 && [&]() -> bool {
            for (size_t i = 0; i < xShape.GetDimNum() - 2; ++i) {
                if (scaleShape.GetDim(i) != xShape.GetDim(i)) {
                    return true;
                }
            }
            return false;
        }(),
        OP_LOGE(context->GetNodeName(),
                "scale shape [%s] leading dims (except last 2) should match x [%s], please check.",
                Shape2String(scaleShape).c_str(), Shape2String(xShape).c_str()),
        return ge::GRAPH_FAILED);

    // 根据 scale[-2] 的值判定量化轴:
    //   axis=-1 (PER_CHANNEL): scale = [x[0], ..., x[-3], ceil(x[-1]/64), 1]
    //     → scale[-2] == ceil(x[-1]/64), scale[-2] != x[-1]
    //   axis=-2 (PER_HEAD):    scale = [x[0], ..., ceil(x[-2]/64), x[-1], 1]
    //     → scale[-2] == x[-1], scale[-3] == ceil(x[-2]/64)
    int64_t scaleNeg2 = scaleShape.GetDim(scaleShape.GetDimNum() - 2);
    int64_t scaleNeg3 = scaleShape.GetDim(scaleShape.GetDimNum() - 3);
    // xDimNeg1/xDimNeg2 已通过 BLOCK_SIZE 整除校验, 直接除即可得到 block 数
    if (scaleNeg2 == tilingParams.xDimNeg1 && scaleNeg3 == tilingParams.xDimNeg2 / BLOCK_SIZE) {
        // axis=-2 (PER_HEAD): scale[-2] == x[-1], scale[-3] == x[-2]/64
        tilingParams.quantMode = 1;
    } else if (scaleNeg2 == tilingParams.xDimNeg1 / BLOCK_SIZE && scaleNeg3 == tilingParams.xDimNeg2) {
        // axis=-1 (PER_CHANNEL): scale[-2] == x[-1]/64
        tilingParams.quantMode = 0;
    } else {
        OP_LOGE(context->GetNodeName(),
                "Cannot determine quant axis from scale shape [%s] and x shape [%s], "
                "scale[-2]=%lld is neither x[-1]/64=%lld nor x[-1]=%lld, please check.",
                Shape2String(scaleShape).c_str(), Shape2String(xShape).c_str(), scaleNeg2,
                tilingParams.xDimNeg1 / BLOCK_SIZE, tilingParams.xDimNeg1);
        return ge::GRAPH_FAILED;
    }

    // y.shape 必须与 x.shape 完全一致 (反量化恢复原始 shape)
    auto yShapePtr = context->GetOutputShape(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    auto yShape = yShapePtr->GetStorageShape();

    OP_CHECK_IF(
        xShape != yShape,
        OP_LOGE(context->GetNodeName(), "The shape of output y:%s must be same with shape of input x:%s, please check.",
                Shape2String(yShape).c_str(), Shape2String(xShape).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * GetPlatformInfo - 获取硬件平台参数
 *   totalCoreNum:  AIV 可用核数, 用于后续分核计算
 *   ubSize:        单核 UB 大小 (bytes), 用于确定单次处理的 block 数
 *   workspaceSize: API 库 workspace 大小
 *   vfLen:         Vector 寄存器长度 (bytes), 影响 vector 指令的元素粒度
 */
static ge::graphStatus GetPlatformInfo(AntiHiF4QuantTilingParam& tilingParams, const gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter AntiHiF4QuantTiling GetPlatformInfo.");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    tilingParams.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((tilingParams.totalCoreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingParams.ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((tilingParams.ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);

    tilingParams.workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    tilingParams.vfLen = Ops::Base::GetVRegSize(context);

    return ge::GRAPH_SUCCESS;
}

/*
 * GetAttr - 读取并校验 dst_type 属性
 *   dst_type 决定反量化的目标精度: 1=FLOAT16, 27=BFLOAT16
 *   同时校验 dst_type 与 y 的实际 dtype 是否一致
 */
static ge::graphStatus GetAttr(AntiHiF4QuantTilingParam& tilingParams, const gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* dstTypePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstTypePtr);

    // dst_type 合法值校验
    int64_t dstType = *dstTypePtr;
    OP_CHECK_IF(dstType != DST_TYPE_FLOAT16 && dstType != DST_TYPE_BF16,
                OP_LOGE(context->GetNodeName(),
                        "Invalid attr dst_type: [%lld], must be in {1(FLOAT16), 27(BFLOAT16)}, please check.", dstType),
                return ge::GRAPH_FAILED);

    tilingParams.dstType = dstType;

    // 交叉校验: dst_type 属性值必须与 InferShape 推导出的 y dtype 一致
    auto outputYPtr = context->GetOutputDesc(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYPtr);
    auto yDtype = outputYPtr->GetDataType();

    if (dstType == DST_TYPE_FLOAT16) {
        OP_CHECK_IF(yDtype != ge::DT_FLOAT16,
                    OP_LOGE(context->GetNodeName(), "dst_type is FLOAT16(1), but output y dtype is [%s], please check.",
                            Ops::Base::ToString(yDtype).c_str()),
                    return ge::GRAPH_FAILED);
    } else if (dstType == DST_TYPE_BF16) {
        OP_CHECK_IF(
            yDtype != ge::DT_BF16,
            OP_LOGE(context->GetNodeName(), "dst_type is BFLOAT16(27), but output y dtype is [%s], please check.",
                    Ops::Base::ToString(yDtype).c_str()),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MergeAxis(AntiHiF4QuantTilingParam& tilingParams, const gert::TilingContext* context)
{
    auto xShapePtr = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();

    int64_t dimNum = xShape.GetDimNum();

    if (tilingParams.quantMode == 0) {
        for (int64_t i = 0; i < dimNum - NUM_TWO; i++) {
            tilingParams.xDimNeg2 *= xShape.GetDim(i);
        }
    } else {
        for (int64_t i = 0; i < dimNum - NUM_TWO; i++) {
            tilingParams.xDimNeg3 *= xShape.GetDim(i);
        }
    }

    return ge::GRAPH_SUCCESS;
}

static void SetTilingKey(AntiHiF4QuantTilingParam& tilingParams, gert::TilingContext* context)
{
    int64_t tilingKey = GET_TPL_TILING_KEY(tilingParams.quantMode);
    tilingParams.tilingKey = tilingKey;
    context->SetTilingKey(tilingKey);
}

static ge::graphStatus TilingForAntiHiF4Quant(gert::TilingContext* context)
{
    OP_LOGD("AntiHiF4QuantTiling", "Enter TilingForAntiHiF4Quant");

    OP_CHECK_IF(context == nullptr, OP_LOGE("AntiHiF4QuantTiling", "Tiling context is null."), return ge::GRAPH_FAILED);

    AntiHiF4QuantTilingParam tilingParams;

    OP_CHECK_IF(GetPlatformInfo(tilingParams, context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The platforminfo get failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckDtype(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The data type check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckShape(tilingParams, context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The shape check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr(tilingParams, context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The attr check failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(MergeAxis(tilingParams, context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "The merge axis failed."), return ge::GRAPH_FAILED);

    SetTilingKey(tilingParams, context);

    if (tilingParams.quantMode == 0) {
        AntiHiF4QuantTailTiling tailTiling(context, tilingParams);
        return tailTiling.DoTiling();
    } else {
        AntiHiF4QuantNotTailTiling notTailTiling(context, tilingParams);
        return notTailTiling.DoTiling();
    }
}

static ge::graphStatus TilingPrepareForAntiHiF4Quant(gert::TilingParseContext* context)
{
    OP_LOGD("AntiHiF4QuantTiling", "Enter TilingPrepareForAntiHiF4Quant");

    OP_CHECK_IF(context == nullptr, OP_LOGE("AntiHiF4QuantTiling", "TilingParse context is null."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AntiHiF4Quant)
    .Tiling(TilingForAntiHiF4Quant)
    .TilingParse<AntiHiF4QuantCompileInfo>(TilingPrepareForAntiHiF4Quant);

} // namespace optiling
