/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Host-side tiling for NormalizeBatch on arch35
//   (ascend950 / ascend960dt / ascend960pr / ascend350).
//   主要步骤：
//     GetPlatformInfo       平台参数（coreNum/ubSize/blockSize/vRegSize，禁写死）
//     GetInputInfo          输入校验（dtype → format → 维度 → seq_len 一致性 → 输出一致）
//     ReadAndValidateAttrs  normalize_type（per_feature/all_features）/ epsilon（>0）
//     ComputeSplit          batch 切核 + UB 切分（SmallD/BigD 模板判定 + eachLoopDNum）
//     FillAndLogTilingData + SetTilingKey（GET_TPL_TILING_KEY 双 bool 轴）+ SetBlockDim
//
//   切分模型：seq_len 为运行时 GM 数据，tiling 只见静态 (n, c, d)——batch 维
//   （per_feature: n*c 行 / all_features: n 平面）均分切核，统计长度 L 核内钳制。
//   模板判定：
//     SmallD：单遍数据（per_feature 行 rowStride / all_features 平面 c*rowStride）×4B ≤ UB 预算
//     BigD ：放不下 → d 分段两遍，eachLoopDNum = FloorAlign(预算, VL)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "register/op_def_registry.h"                             // IMPL_OP_OPTILING
#include "op_common/log/log.h"                                    // OP_LOGE, OP_LOGI, OP_CHECK_*
#include "op_common/op_host/util/platform_util.h"                 // PlatformAscendC, GetUbBlockSize/GetVRegSize
#include "op_common/op_host/util/math_util.h"                     // Ops::Base::CeilAlign/FloorAlign/CeilDiv
#include "graph/utils/type_utils.h"                               // TypeUtils::DataTypeToSerialString
#include "securec.h"                                              // memset_s
#include "../../op_kernel/arch35/normalize_batch_tiling_struct.h" // NormalizeBatchTilingData
#include "../../op_kernel/arch35/normalize_batch_struct.h"        // TPL 声明（GET_TPL_TILING_KEY）
#include "normalize_batch_tiling_arch35.h"                        // NormalizeBatchCompileInfo

namespace optiling {

namespace {
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t INPUT_SEQ_LEN_IDX = 1;
constexpr size_t OUTPUT_Y_IDX = 0;
constexpr size_t ATTR_NORMALIZE_TYPE_IDX = 0; // attr 按索引访问：normalize_type=0，epsilon=1
constexpr size_t ATTR_EPSILON_IDX = 1;

constexpr int64_t FP32_BYTES = 4; // fp32 单元素字节数（当前唯一支持 dtype）
constexpr int64_t INT32_BYTES = 4;
constexpr int64_t EXPECTED_RANK = 3;           // (n, c, d)
constexpr int64_t MIN_D_DIM = 2;               // d>1（对齐 TIK check_params）
constexpr int64_t SCALAR_BUF_BYTES = 64;       // 标量槽（sum/sumsq 各 32B）
constexpr int64_t RESERVE_UB_BYTES = 8 * 1024; // TPipe/框架保留预算
constexpr int64_t DATA_QUEUE_DEPTH = 2;        // dataQue_ 双缓冲
constexpr float DEFAULT_EPSILON = 1e-5f;       // attr 默认值（对齐 proto）

const char* const NORMALIZE_TYPE_PER_FEATURE = "per_feature";
const char* const NORMALIZE_TYPE_ALL_FEATURES = "all_features";
} // namespace

// tiling 全程工作状态
struct NormalizeBatchCtx {
    // 平台参数
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t blockSize = 0;
    int64_t vRegSize = 0;

    // 输入信息
    ge::DataType xDtype = ge::DT_UNDEFINED;
    std::vector<int64_t> xShape; // (n, c, d)

    // attr
    bool isAllFeatures = false;
    float epsilon = DEFAULT_EPSILON;

    // 派生量（ComputeSplit 填充）
    int64_t nDim = 0;
    int64_t cDim = 0;
    int64_t dDim = 0;
    int64_t vlElems = 0;       // VL 元素数（fp32 256B/4 = 64）
    int64_t rowStride = 0;     // CeilAlign(d, vlElems)
    int64_t eachLoopDNum = 0;  // BigD 分段（VL 对齐）
    int64_t dataBufElems = 0;  // 主数据 buffer 元素数
    int64_t seqLenBufSize = 0; // seq_len buffer 字节
    bool isBigD = false;
    int64_t batch = 0; // per_feature: n*c / all_features: n
    int64_t usedCoreNum = 0;
    int64_t normalCoreProcessNum = 0;
    int64_t tailCoreProcessNum = 0;
};

// GetPlatformInfo — 平台硬件参数获取
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, NormalizeBatchCtx& ctx)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ctx.coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(ctx.coreNum == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.coreNum",
                                                         "GetCoreNumAiv returned 0"),
                return ge::GRAPH_FAILED);

    uint64_t ub = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ctx.ubSize = static_cast<int64_t>(ub);
    OP_CHECK_IF(ctx.ubSize == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.ubSize",
                                                         "GetCoreMemSize(UB) returned 0"),
                return ge::GRAPH_FAILED);

    ctx.blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));
    ctx.vRegSize = static_cast<int64_t>(Ops::Base::GetVRegSize(context));
    OP_CHECK_IF(ctx.blockSize <= 0 || ctx.vRegSize <= 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.blockSize/vRegSize",
                                                         "invalid platform geometry"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// GetInputInfo — 输入 x / seq_len 与输出 y 校验
//   dtype → format → 维度 → seq_len 形状一致性 → 输出一致。
static ge::graphStatus GetInputInfo(gert::TilingContext* context, NormalizeBatchCtx& ctx)
{
    // x: rank-3 (n, c, d)，float32，ND
    auto xShapePtr = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xS = xShapePtr->GetStorageShape();
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ctx.xDtype = xDesc->GetDataType();
    OP_CHECK_IF(ctx.xDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(ctx.xDtype).c_str(),
                                                      "only DT_FLOAT is supported"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xDesc->GetStorageFormat() != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context->GetNodeName(), "x",
                                               std::to_string(static_cast<int64_t>(xDesc->GetStorageFormat())).c_str(),
                                               "only FORMAT_ND is supported"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(xS.GetDimNum() != EXPECTED_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                         std::to_string(static_cast<int64_t>(xS.GetDimNum())).c_str(),
                                                         "input_x rank must be 3 (n, c, d)"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < xS.GetDimNum(); ++i) {
        OP_CHECK_IF(xS.GetDim(i) < 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(xS.GetDim(i)).c_str(),
                        "dim[" + std::to_string(i) + "] is negative (dynamic shape must be instantiated)"),
                    return ge::GRAPH_FAILED);
    }
    ctx.xShape.clear();
    ctx.xShape.reserve(xS.GetDimNum());
    for (size_t i = 0; i < xS.GetDimNum(); ++i) {
        ctx.xShape.push_back(xS.GetDim(i));
    }
    ctx.nDim = ctx.xShape[0];
    ctx.cDim = ctx.xShape[1];
    ctx.dDim = ctx.xShape[2];
    OP_CHECK_IF(ctx.nDim < 1 || ctx.cDim < 1,
                OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "x",
                                          ("[" + std::to_string(ctx.nDim) + ", " + std::to_string(ctx.cDim) + ", " +
                                           std::to_string(ctx.dDim) + "]")
                                              .c_str(),
                                          "n and c must be >= 1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ctx.dDim < MIN_D_DIM,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "d", std::to_string(ctx.dDim).c_str(),
                                                      "d must be > 1 (aligned with the TIK reference)"),
                return ge::GRAPH_FAILED);

    // seq_len: rank-1 (n,)，int32，ND
    auto seqShapePtr = context->GetInputShape(INPUT_SEQ_LEN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqShapePtr);
    const gert::Shape& seqS = seqShapePtr->GetStorageShape();
    const gert::CompileTimeTensorDesc* seqDesc = context->GetInputDesc(INPUT_SEQ_LEN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqDesc);
    OP_CHECK_IF(
        seqDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "seq_len",
                                              ge::TypeUtils::DataTypeToSerialString(seqDesc->GetDataType()).c_str(),
                                              "only DT_INT32 is supported"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        seqS.GetDimNum() != 1 || seqS.GetDim(0) != ctx.nDim,
        OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "seq_len",
                                  ("[" + std::to_string(seqS.GetDimNum() == 0 ? 0 : seqS.GetDim(0)) + "]").c_str(),
                                  ("rank-1 shape (" + std::to_string(ctx.nDim) + ",) matching input_x n").c_str()),
        return ge::GRAPH_FAILED);

    // y: dtype/format/shape 与 x 一致
    const gert::CompileTimeTensorDesc* yDesc = context->GetOutputDesc(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_IF(
        yDesc->GetDataType() != ctx.xDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "output_y",
                                              ge::TypeUtils::DataTypeToSerialString(yDesc->GetDataType()).c_str(),
                                              "output_y dtype must be the same as input_x dtype"),
        return ge::GRAPH_FAILED);
    auto yShapePtr = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    const gert::Shape& yS = yShapePtr->GetStorageShape();
    OP_CHECK_IF(yS.GetDimNum() != EXPECTED_RANK || yS.GetDim(0) != ctx.nDim || yS.GetDim(1) != ctx.cDim ||
                    yS.GetDim(2) != ctx.dDim,
                OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "output_y",
                                          ("[" + std::to_string(yS.GetDimNum()) + " dims]").c_str(),
                                          "output_y shape must be identical to input_x shape"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ReadAndValidateAttrs — normalize_type / epsilon 校验
static ge::graphStatus ReadAndValidateAttrs(gert::TilingContext* context, NormalizeBatchCtx& ctx)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* normalizeType = attrs->GetAttrPointer<char>(ATTR_NORMALIZE_TYPE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, normalizeType);
    if (strcmp(normalizeType, NORMALIZE_TYPE_PER_FEATURE) == 0) {
        ctx.isAllFeatures = false;
    } else if (strcmp(normalizeType, NORMALIZE_TYPE_ALL_FEATURES) == 0) {
        ctx.isAllFeatures = true;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "normalize_type", normalizeType,
                                              "must be \"per_feature\" or \"all_features\"");
        return ge::GRAPH_FAILED;
    }
    const float* epsilonPtr = attrs->GetAttrPointer<float>(ATTR_EPSILON_IDX);
    ctx.epsilon = (epsilonPtr != nullptr) ? (*epsilonPtr) : DEFAULT_EPSILON;
    // isfinite 拦 +Inf：IEEE 下 +Inf > 0 为真，仅 !(eps > 0) 会放行 Inf 进 tiling，
    // kernel 侧 rstd = 1/(stdDev+Inf) = 0 使输出整体退化为 ±0（对齐 bn_infer_grad
    // / bn_training_update 的 epsilon 校验惯例）
    OP_CHECK_IF(
        !(std::isfinite(ctx.epsilon) && ctx.epsilon > 0.0f),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "epsilon", std::to_string(ctx.epsilon).c_str(),
                                              "epsilon must be a finite float greater than 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ComputeSplit — batch 切核 + UB 切分 + SmallD/BigD 模板判定
static ge::graphStatus ComputeSplit(gert::TilingContext* context, NormalizeBatchCtx& ctx)
{
    ctx.vlElems = ctx.vRegSize / FP32_BYTES;
    OP_CHECK_IF(ctx.vlElems <= 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.vRegSize",
                                                         "vRegSize must cover at least one fp32 element"),
                return ge::GRAPH_FAILED);

    // UB 行距：CeilAlign(d, VL) —— 保证行内 LoadAlign 整 VL 读取不越行缓冲末尾
    ctx.rowStride = Ops::Base::CeilAlign(ctx.dDim, ctx.vlElems);

    // seq_len buffer（32B 对齐）
    ctx.seqLenBufSize = Ops::Base::CeilAlign(ctx.nDim * INT32_BYTES, ctx.blockSize);

    // 主数据 buffer 预算
    const int64_t budgetBytes = ctx.ubSize - ctx.seqLenBufSize - SCALAR_BUF_BYTES - RESERVE_UB_BYTES;
    OP_CHECK_IF(budgetBytes <= 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "ubSize", "UB budget is negative after seq_len/scalar/reserve deduction"),
                return ge::GRAPH_FAILED);

    const int64_t budgetElems = budgetBytes / FP32_BYTES;
    // 统一按双缓冲队列折算预算：kernel 侧 InitBuffer(dataQue_, 2, ...) 实占 2 份，
    // SmallD/BigD 两条路径的判定口径必须一致，否则单份过界时运行期 UB 溢出
    const int64_t queueBudgetElems = budgetElems / DATA_QUEUE_DEPTH;
    if (ctx.isAllFeatures) {
        // all_features：平面（c 行）驻 UB 单遍
        ctx.isBigD = (ctx.cDim * ctx.rowStride) > queueBudgetElems;
    } else {
        // per_feature：单行驻 UB 单遍
        ctx.isBigD = ctx.rowStride > queueBudgetElems;
    }

    if (!ctx.isBigD) {
        ctx.dataBufElems = ctx.isAllFeatures ? (ctx.cDim * ctx.rowStride) : ctx.rowStride;
        ctx.eachLoopDNum = ctx.vlElems; // SmallD 不消费，填合法值便于日志
    } else {
        // BigD：d 分段（双缓冲队列，单段 ≤ 预算一半）
        ctx.eachLoopDNum = Ops::Base::FloorAlign(queueBudgetElems, ctx.vlElems);
        OP_CHECK_IF(ctx.eachLoopDNum < ctx.vlElems,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "ubSize",
                        "UB budget too small for one VL-aligned BigD chunk (need at least " +
                            std::to_string(ctx.vlElems * FP32_BYTES * DATA_QUEUE_DEPTH) + " bytes)"),
                    return ge::GRAPH_FAILED);
        ctx.dataBufElems = ctx.eachLoopDNum;
    }

    // batch 切核：FloorDiv + 尾核吸收余数（tail = q + r ∈ [q, q + used-1]）。
    // 禁用 CeilDiv：batch∈(used, 2*used) 时 CeilDiv*(used-1) ≥ batch 会使 tail ≤ 0，
    // 非末核起始越界 → GM 越界读写（反例：batch=64/used=56 时 CeilDiv 得 tail=-46）。
    ctx.batch = ctx.isAllFeatures ? ctx.nDim : (ctx.nDim * ctx.cDim);
    ctx.usedCoreNum = std::min(ctx.batch, ctx.coreNum);
    ctx.normalCoreProcessNum = ctx.batch / ctx.usedCoreNum; // FloorDiv（used ≤ batch 恒 ≥ 1）
    ctx.tailCoreProcessNum = ctx.batch - (ctx.usedCoreNum - 1) * ctx.normalCoreProcessNum;
    return ge::GRAPH_SUCCESS;
}

// FillAndLogTilingData — 填充 + 日志
static ge::graphStatus FillAndLogTilingData(gert::TilingContext* context, const NormalizeBatchCtx& ctx)
{
    auto* td = context->GetTilingData<NormalizeBatchTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(memset_s(td, sizeof(NormalizeBatchTilingData), 0, sizeof(NormalizeBatchTilingData)) != EOK,
                OP_LOGE(context->GetNodeName(), "Memset tilingdata error"), return ge::GRAPH_FAILED);

    td->nDim = ctx.nDim;
    td->cDim = ctx.cDim;
    td->dDim = ctx.dDim;
    td->usedCoreNum = static_cast<int32_t>(ctx.usedCoreNum);
    td->normalCoreProcessNum = ctx.normalCoreProcessNum;
    td->tailCoreProcessNum = ctx.tailCoreProcessNum;
    td->rowStride = ctx.rowStride;
    td->eachLoopDNum = ctx.eachLoopDNum;
    td->dataBufElems = ctx.dataBufElems;
    td->seqLenBufSize = ctx.seqLenBufSize;
    td->scalarBufSize = SCALAR_BUF_BYTES;
    td->epsilon = ctx.epsilon;

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0U; // batch 维切核、核间零依赖，无 workspace

    OP_LOGI(context->GetNodeName(),
            "tiling: isAllFeatures=%d, isBigD=%d, shape=[%ld, %ld, %ld], batch=%ld, "
            "usedCoreNum=%ld, normal=%ld, tail=%ld, rowStride=%ld, eachLoopDNum=%ld, dataBufElems=%ld",
            static_cast<int>(ctx.isAllFeatures), static_cast<int>(ctx.isBigD), ctx.nDim, ctx.cDim, ctx.dDim, ctx.batch,
            ctx.usedCoreNum, ctx.normalCoreProcessNum, ctx.tailCoreProcessNum, ctx.rowStride, ctx.eachLoopDNum,
            ctx.dataBufElems);
    return ge::GRAPH_SUCCESS;
}

// TilingFuncNormalizeBatch — tiling 入口
static ge::graphStatus TilingFuncNormalizeBatch(gert::TilingContext* context)
{
    NormalizeBatchCtx ctx;
    OP_CHECK_IF(GetPlatformInfo(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputInfo(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadAndValidateAttrs(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeSplit(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(FillAndLogTilingData(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // SetTilingKey（框架宏按模板实参位序生成）与 SetBlockDim（必须大于等于 1）
    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(ctx.isAllFeatures ? 1U : 0U),
                                                  static_cast<uint64_t>(ctx.isBigD ? 1U : 0U));
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(static_cast<uint32_t>(std::max(ctx.usedCoreNum, 1L)));
    return ge::GRAPH_SUCCESS;
}

// TilingParseForNormalizeBatch — 编译期准备（空 CompileInfo，恒成功）
static ge::graphStatus TilingParseForNormalizeBatch([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// IMPL_OP_OPTILING(NormalizeBatch) — Host 侧注册
//   seq_len 的“值”不影响 tiling（仅核内消费），无需 TilingInputsDataDependency。
IMPL_OP_OPTILING(NormalizeBatch)
    .Tiling(TilingFuncNormalizeBatch)
    .TilingParse<NormalizeBatchCompileInfo>(TilingParseForNormalizeBatch);

} // namespace optiling
