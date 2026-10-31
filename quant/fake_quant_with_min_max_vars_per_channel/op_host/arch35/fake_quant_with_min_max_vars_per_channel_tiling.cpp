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
 * \file fake_quant_with_min_max_vars_per_channel_tiling.cpp
 * \brief FakeQuantWithMinMaxVarsPerChannel Tiling 实现（arch35 / ascend950 / v2.2 方案 A）
 *
 * 函数拆分（DESIGN v2.2 §3.4）：
 *   - SelectMode             : 校验 + 固定返回 PER_MODE=2
 *   - MergeInputShape        : (...,d_last) → (M, N)
 *   - CalcNativePCBlock      : 二路竞争切核（actCoreNum0 vs actCoreNum1）
 *   - CalcNativePCUB         : 行相关 + 行无关字节模型 (v2.2：行无关仅 min/max DB)
 *                              多行/单行二路决策
 *   - WriteTilingData        : 填 TilingData
 *   - FakeQuantWithMinMaxVarsPerChannelTilingFunc : 入口
 *
 * v2.2 关键变更（基于 DESIGN v2.2 §3.4.6）：
 *   - 行无关字节模型从 v2.1 的 (min+max+5×scale)×BUF_NUM×L + 16KB(Floor tmp) 收敛到
 *     仅 (min+max)×BUF_NUM×align_up(L,64) = 16L
 *   - RESERVED_UB 从 24 KB（含 8 KB Select + 16 KB Floor）回调到 1 KB（仅基础 stack/event/SPR）
 *   - 删除 kPersistentVectors / kOneVecScaleVectors / kFloorTmpBytes 常量
 */

#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/fake_quant_with_min_max_vars_per_channel_tiling_data.h"
#include "../../op_kernel/arch35/fake_quant_with_min_max_vars_per_channel_tiling_key.h"

namespace optiling {

constexpr int64_t kCacheLineBytes = 32; // ascend950 cacheLine
constexpr int64_t kVL = 64;             // VL = 64 (fp32 = 256B)
constexpr int64_t kBufNum = 2;          // Double Buffer
constexpr size_t kMaxRank = 8U;         // README 约束：x rank ∈ [1, 8]
// [v2.2 方案 A] RESERVED_UB 回调到 1 KB：取消 Select/Floor 高阶 API，无需 24 KB 预留
constexpr int64_t kReservedUB = 1 * 1024;
constexpr int64_t DEFAULT_UB_SIZE = 192 * 1024;

constexpr size_t WORKSPACE_NUM = 1;
constexpr uint32_t WS_SYS_SIZE = 0U;

// ----------------------------------------------------------------------------
// TilingResult 中间结构体（DESIGN §3.4.1）
// ----------------------------------------------------------------------------
struct TilingResult {
    int64_t numCore = 0;
    int64_t blockAxis = 0;
    int64_t blockFactor = 0;
    int64_t blockTailFactor = 0;
    int64_t blockUnion = 1;
    int64_t ubAxis = 0;
    int64_t baseN = 1;
    int64_t baseLen = 0;
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    int64_t dim2 = 0;
    int64_t headNum = 0;
    int64_t tailDim = 0;
};

// ----------------------------------------------------------------------------
// 通用工具
// ----------------------------------------------------------------------------
// CeilDiv 用除法式实现（a/b + 余数进位），避免 (a+b-1) 在 a 接近 INT64_MAX 时的
// 有符号溢出（P1-04：被除数侧运算前保护）
static inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b <= 0) {
        return 0;
    }
    int64_t q = a / b;
    return (a % b != 0) ? q + 1 : q;
}

// AlignUp 在向上取整会越过 INT64_MAX 时返回原值 a（对齐是性能提示，不静默产生
// 溢出 UB；P1-04），b<=0 时维持原语义
static inline int64_t AlignUp(int64_t a, int64_t b)
{
    if (b <= 0) {
        return a;
    }
    int64_t r = a % b;
    if (r == 0) {
        return a;
    }
    if (a > INT64_MAX - (b - r)) {
        return a;
    }
    return (a / b + 1) * b;
}

// GetCoreNum: 平衡核数（DESIGN §3.4.5）
//   taskNum 个任务，分配到 coreNum 个核上时，实际活跃的核数
static inline int64_t GetCoreNum(int64_t taskNum, int64_t coreNum)
{
    if (taskNum <= 0 || coreNum <= 0) {
        return 1;
    }
    int64_t perCore = CeilDiv(taskNum, coreNum);
    if (perCore <= 0) {
        perCore = 1;
    }
    return CeilDiv(taskNum, perCore);
}

// ----------------------------------------------------------------------------
// 平台信息
// ----------------------------------------------------------------------------
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum <= 0, OP_LOGE(context, "FakeQuantPC: coreNum invalid"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    if (*ubSize == 0) {
        *ubSize = DEFAULT_UB_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

// ----------------------------------------------------------------------------
// SelectMode（DESIGN §3.4.3）：校验 + 固定返回 PER_MODE=2
// ----------------------------------------------------------------------------
static int64_t SelectMode(gert::TilingContext* context, ge::DataType xDtype, const gert::Shape& xShape,
                          const gert::Shape& minShape, const gert::Shape& maxShape, int64_t numBits)
{
    OP_CHECK_IF(xDtype != ge::DT_FLOAT,
                OP_LOGE(context, "FakeQuantPC: only fp32 supported, got dtype=%d", static_cast<int>(xDtype)),
                return -1);

    OP_CHECK_IF(xShape.GetDimNum() < 1, OP_LOGE(context, "FakeQuantPC: x rank must >= 1"), return -1);
    OP_CHECK_IF(xShape.GetDimNum() > kMaxRank,
                OP_LOGE(context, "FakeQuantPC: x rank %zu exceeds max %zu", xShape.GetDimNum(), kMaxRank), return -1);
    // 未知维守卫：x 任一维为负（含 -1/-2 未知标记）时维度语义未定，不得当作 0/空张量
    // 静默成功（否则 min/max 长度同为 -1 时可穿过相等校验，下发 blockFactor=0、
    // tailDim=-1 的空计划且不报错，非空输入不会被写入）。
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        OP_CHECK_IF(xShape.GetDim(i) < 0,
                    OP_LOGE(context, "FakeQuantPC: x dim[%zu]=%ld is not a concrete size (negative/unknown marker)", i,
                            xShape.GetDim(i)),
                    return -1);
    }
    OP_CHECK_IF(minShape.GetDimNum() != 1,
                OP_LOGE(context, "FakeQuantPC: min rank must be 1, got %zu", minShape.GetDimNum()), return -1);
    OP_CHECK_IF(maxShape.GetDimNum() != 1,
                OP_LOGE(context, "FakeQuantPC: max rank must be 1, got %zu", maxShape.GetDimNum()), return -1);
    OP_CHECK_IF(minShape.GetDim(0) != maxShape.GetDim(0), OP_LOGE(context, "FakeQuantPC: min/max shape mismatch"),
                return -1);
    OP_CHECK_IF(minShape.GetDim(0) < 0,
                OP_LOGE(context, "FakeQuantPC: min/max length %ld is not a concrete size", minShape.GetDim(0)),
                return -1);

    int64_t xTail = xShape.GetDim(xShape.GetDimNum() - 1);
    OP_CHECK_IF(minShape.GetDim(0) != xTail,
                OP_LOGE(context, "FakeQuantPC: min/max length (%ld) != x tail dim (%ld)", minShape.GetDim(0), xTail),
                return -1);

    OP_CHECK_IF(numBits < 2 || numBits > 16, OP_LOGE(context, "FakeQuantPC: num_bits must in [2,16], got %ld", numBits),
                return -1);

    return 2;
}

// ----------------------------------------------------------------------------
// MergeInputShape（DESIGN §3.4.4）：(..., d_last) → (M, N)
// ----------------------------------------------------------------------------
// 返回 false 表示前置维连乘溢出 INT64_MAX（P1-05：如 x=[3037000500,3037000500,8]
// 各维合法但乘积越界），调用方必须拒绝而不是静默走空 tensor 分支
static bool MergeInputShape(const gert::Shape& xShape, int64_t& M, int64_t& N)
{
    size_t rank = xShape.GetDimNum();
    N = xShape.GetDim(rank - 1);
    M = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        int64_t d = xShape.GetDim(i);
        d = (d > 0) ? d : 0;
        if (M != 0 && d > INT64_MAX / M) {
            return false;
        }
        M *= d;
    }
    return true;
}

// ----------------------------------------------------------------------------
// CalcNativePCBlock（DESIGN §3.4.5）：二路竞争
// ----------------------------------------------------------------------------
static void CalcNativePCBlock(int64_t M, int64_t N, int64_t coreNum, int64_t dtypeSize, TilingResult& res)
{
    if (dtypeSize <= 0) {
        dtypeSize = sizeof(float);
    }
    int64_t elemPerCacheLine = kCacheLineBytes / dtypeSize; // fp32 → 8
    if (elemPerCacheLine <= 0) {
        elemPerCacheLine = 1;
    }

    int64_t cacheLineNumN = CeilDiv(N, elemPerCacheLine);

    int64_t actCoreNum0 = GetCoreNum(M, coreNum);             // 切 M
    int64_t actCoreNum1 = GetCoreNum(cacheLineNumN, coreNum); // 切 N（按 cacheLine）

    if (actCoreNum0 >= actCoreNum1) {
        // 切 M 维
        res.blockAxis = 0;
        res.numCore = actCoreNum0;
        if (actCoreNum0 <= 0) {
            actCoreNum0 = 1;
            res.numCore = 1;
        }
        res.blockFactor = CeilDiv(M, actCoreNum0);
        if (res.blockFactor <= 0) {
            res.blockFactor = 1;
        }
        int64_t consumed = res.blockFactor * (actCoreNum0 - 1);
        res.blockTailFactor = M - consumed;
        if (res.blockTailFactor <= 0) {
            // 退化为单核
            res.numCore = 1;
            res.blockFactor = M;
            res.blockTailFactor = M;
        }
    } else {
        // 切 N 维（按 cacheLine 粒度）
        res.blockAxis = 1;
        res.numCore = actCoreNum1;
        if (actCoreNum1 <= 0) {
            actCoreNum1 = 1;
            res.numCore = 1;
        }
        int64_t cellsPerCore = CeilDiv(cacheLineNumN, actCoreNum1);
        if (cellsPerCore <= 0) {
            cellsPerCore = 1;
        }
        res.blockFactor = cellsPerCore * elemPerCacheLine;
        int64_t consumed = res.blockFactor * (actCoreNum1 - 1);
        res.blockTailFactor = N - consumed;
        if (res.blockTailFactor <= 0) {
            res.numCore = 1;
            res.blockFactor = N;
            res.blockTailFactor = N;
        }
    }
    res.blockUnion = 1;
}

// ----------------------------------------------------------------------------
// CalcNativePCUB（DESIGN v2.2 §3.4.6）：行相关 + 行无关字节模型 (方案 A)
//
// v2.2 行无关字节模型：
//   行相关 (per baseN × baseLen): (xBytes + yBytes) × BUF_NUM = 4 × dtypeSize
//   行无关 (per baseLen):          (minBytes + maxBytes) × BUF_NUM = 4 × dtypeSize
//                                  仅 min/max DB queue，无任何常驻 TBuf
//   fp32 时单行模式 baseN=1：bytesPerLen_single = (4 + 4) × 4 = 32 B/elem
//
// 对比 v2.1：消除 5 个 scale TBuf + 1 个 oneVec TBuf + Floor 16 KB tmp + Select 8 KB 预留
// 释放空间 = (5+1) × align_up(L,64) × 4 + 16 KB + 8 KB = 24L + 24 KB
// ----------------------------------------------------------------------------
static void CalcNativePCUB(TilingResult& res, uint64_t ubSize, int64_t dtypeSize)
{
    int64_t available = static_cast<int64_t>(ubSize) - kReservedUB;
    if (available <= 0) {
        available = static_cast<int64_t>(ubSize);
    }

    // 单行模式 baseN=1 下每 element 字节消耗：
    //   row-related: (1+1) × BUF_NUM × 1 = 4    (x + y DB)
    //   row-indep:   (1+1) × BUF_NUM     = 4    (min + max DB)
    //   合计: 8 vectors × dtypeSize = 32 B/elem (fp32)
    int64_t bytesPerLen_single = (1 + 1) * kBufNum * 1 // x + y DB (baseN=1)
                                 + (1 + 1) * kBufNum;  // min + max DB
    bytesPerLen_single *= dtypeSize;

    int64_t maxBaseRaw = available / bytesPerLen_single;
    int64_t maxBase = (maxBaseRaw / kVL) * kVL;
    if (maxBase < kVL) {
        maxBase = kVL;
    }

    // 计算尾轴方向工作宽度
    int64_t blockBase = (res.blockAxis == 0) ? res.dim1 : res.blockFactor;
    if (blockBase <= 0) {
        blockBase = 1;
    }
    int64_t blockBaseAligned = AlignUp(blockBase, kVL);

    if (blockBaseAligned <= maxBase / 2 && blockBaseAligned > 0) {
        // ---- 多行模式 (ubAxis = 0) ----
        res.baseLen = blockBaseAligned;
        // [v2.2] 行无关字节：仅 (min + max) × BUF_NUM × baseLen
        int64_t channelBytes = (1 + 1) * kBufNum * res.baseLen * dtypeSize;
        int64_t leftBytes = available - channelBytes;
        if (leftBytes <= 0) {
            leftBytes = 0;
        }
        int64_t rowBytesPerN = (1 + 1) * kBufNum * res.baseLen * dtypeSize; // x+y DB
        int64_t maxN = (rowBytesPerN > 0) ? (leftBytes / rowBytesPerN) : 1;
        if (maxN < 1) {
            maxN = 1;
        }

        int64_t blockInnerSize = (res.blockAxis == 0) ? res.blockFactor : res.dim0;
        if (blockInnerSize <= 0) {
            blockInnerSize = 1;
        }

        res.ubAxis = 0;
        res.baseN = std::min(maxN, blockInnerSize);
    } else {
        // ---- 单行模式 (ubAxis = 1) ----
        res.ubAxis = 1;
        res.baseN = 1;
        res.baseLen = std::min(blockBaseAligned, maxBase);
        if (res.baseLen <= 0) {
            res.baseLen = kVL;
        }
    }
}

// ----------------------------------------------------------------------------
// WriteTilingData（DESIGN §3.4.7）
// ----------------------------------------------------------------------------
static void WriteTilingData(FakeQuantWithMinMaxVarsPerChannelTilingData* td, const TilingResult& res, float quantMin,
                            float quantMax, int64_t numBits, bool narrowRange)
{
    td->numCore = res.numCore;
    td->blockAxis = res.blockAxis;
    td->blockFactor = res.blockFactor;
    td->blockTailFactor = res.blockTailFactor;
    td->blockUnion = res.blockUnion;

    td->ubAxis = res.ubAxis;
    td->baseN = res.baseN;
    td->baseLen = res.baseLen;

    td->axis = -1;
    td->dim0 = res.dim0;
    td->dim1 = res.dim1;
    td->dim2 = res.dim2;

    td->hasZeroPoint = 0;
    td->headNum = res.headNum;
    td->tailDim = res.tailDim;

    td->quantMin = quantMin;
    td->quantMax = quantMax;
    // numBits 已在 SelectMode 校验为 [2,16]，此处收窄到 int32 存盘安全
    td->numBits = static_cast<int32_t>(numBits);
    td->narrowRange = narrowRange ? 1 : 0;
}

static void LogTilingData(gert::TilingContext* context, const FakeQuantWithMinMaxVarsPerChannelTilingData* tiling)
{
    OP_LOGI(context->GetNodeName(),
            "FakeQuantWithMinMaxVarsPerChannel TilingData: numCore=%ld blockAxis=%ld blockFactor=%ld "
            "blockTailFactor=%ld blockUnion=%ld ubAxis=%ld baseN=%ld baseLen=%ld axis=%ld dim0=%ld dim1=%ld "
            "dim2=%ld hasZeroPoint=%ld headNum=%ld tailDim=%ld quantMin=%f quantMax=%f numBits=%d narrowRange=%d",
            tiling->numCore, tiling->blockAxis, tiling->blockFactor, tiling->blockTailFactor, tiling->blockUnion,
            tiling->ubAxis, tiling->baseN, tiling->baseLen, tiling->axis, tiling->dim0, tiling->dim1, tiling->dim2,
            tiling->hasZeroPoint, tiling->headNum, tiling->tailDim, tiling->quantMin, tiling->quantMax, tiling->numBits,
            tiling->narrowRange);
}

// ----------------------------------------------------------------------------
// Tiling 入口
// ----------------------------------------------------------------------------
static ge::graphStatus FakeQuantWithMinMaxVarsPerChannelTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for Arch35 architecture");
    // 1. 平台信息
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "FakeQuantPC: GetPlatformInfo error"), return ge::GRAPH_FAILED);
    // 2. shape / dtype / attr
    auto xShapeRT = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapeRT);
    auto minShapeRT = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, minShapeRT);
    auto maxShapeRT = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxShapeRT);
    auto xShape = xShapeRT->GetStorageShape();
    auto minShape = minShapeRT->GetStorageShape();
    auto maxShape = maxShapeRT->GetStorageShape();

    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ge::DataType xDtype = xDesc->GetDataType();

    // min/max 同样是输入，tiling 门独立校验其 dtype（不再只依赖 Host 侧或注册匹配）
    auto minDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, minDesc);
    auto maxDesc = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxDesc);
    OP_CHECK_IF(
        minDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(context, "FakeQuantPC: min must be fp32, got dtype=%d", static_cast<int>(minDesc->GetDataType())),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        maxDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(context, "FakeQuantPC: max must be fp32, got dtype=%d", static_cast<int>(maxDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    // num_bits 注册为 Int(int64)（op_graph/..._proto.h），必须按 int64 读取，
    // 否则大值会被截断为 int32 而绕过下方 [2,16] 校验。
    int64_t numBits = 8;
    bool narrowRange = false;
    if (attrs->GetAttrNum() >= 1) {
        const int64_t* numBitsPtr = attrs->GetAttrPointer<int64_t>(0);
        if (numBitsPtr != nullptr) {
            numBits = *numBitsPtr;
        }
    }
    if (attrs->GetAttrNum() >= 2) {
        const bool* nrPtr = attrs->GetAttrPointer<bool>(1);
        if (nrPtr != nullptr) {
            narrowRange = *nrPtr;
        }
    }

    // 3. SelectMode
    int64_t perMode = SelectMode(context, xDtype, xShape, minShape, maxShape, numBits);
    OP_CHECK_IF(perMode != 2, OP_LOGE(context, "FakeQuantPC: SelectMode failed"), return ge::GRAPH_FAILED);

    // 4. MergeInputShape（连乘溢出即拒绝，不静默走空 tensor 分支）
    int64_t M = 0, N = 0;
    OP_CHECK_IF(!MergeInputShape(xShape, M, N),
                OP_LOGE(context, "FakeQuantPC: merged rows of x leading dims overflow int64"), return ge::GRAPH_FAILED);
    // P1-1: 元素总数 M*N 也须在 int64 内（kernel 用 (baseRow+n)*tailDim_ 计算偏移，
    // x=[8, 2^61] 时 M=8 通过检查但 M*N 偏移越界 INT64_MAX）
    OP_CHECK_IF(M != 0 && N > INT64_MAX / M,
                OP_LOGE(context, "FakeQuantPC: element count %ld*%ld overflows int64 row offset", M, N),
                return ge::GRAPH_FAILED);
    // P1-1(fix7)：Kernel Process 中 (myHeadRows_+baseN_-1) 先加后除，M 接近
    // INT64_MAX 时加法越过 int64（单 AIV 配置 numCore=1 → blockFactor=M）。
    // 给 M 留出 baseN 上界余量（kMaxBaseN=190，与 kernel UB 块行数一致）。
    OP_CHECK_IF(M > INT64_MAX - 256,
                OP_LOGE(context, "FakeQuantPC: merged rows %ld leaves no headroom for row-block arithmetic", M),
                return ge::GRAPH_FAILED);

    // 5. TilingData 初始化
    FakeQuantWithMinMaxVarsPerChannelTilingData*
        tiling = context->GetTilingData<FakeQuantWithMinMaxVarsPerChannelTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    float quantMin = narrowRange ? 1.0f : 0.0f;
    float quantMax = static_cast<float>((1LL << numBits) - 1);

    // 6. 空 tensor 退化
    if (M <= 0 || N <= 0) {
        TilingResult emptyRes;
        emptyRes.numCore = 1;
        emptyRes.blockAxis = 0;
        emptyRes.dim0 = M;
        emptyRes.dim1 = N;
        emptyRes.headNum = M;
        emptyRes.tailDim = N;
        WriteTilingData(tiling, emptyRes, quantMin, quantMax, numBits, narrowRange);
        context->SetBlockDim(1);
        size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
        OP_CHECK_NULL_WITH_CONTEXT(context, ws);
        ws[0] = WS_SYS_SIZE;
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(FQ_TPL_DT_FP32), static_cast<uint32_t>(2),
                              static_cast<uint32_t>(0), static_cast<uint32_t>(0));
        LogTilingData(context, tiling);
        return ge::GRAPH_SUCCESS;
    }

    // 7. CalcNativePCBlock
    TilingResult res;
    res.dim0 = M;
    res.dim1 = N;
    res.headNum = M;
    res.tailDim = N;
    CalcNativePCBlock(M, N, coreNum, sizeof(float), res);

    // 8. CalcNativePCUB
    CalcNativePCUB(res, ubSize, sizeof(float));

    // 9. WriteTilingData
    WriteTilingData(tiling, res, quantMin, quantMax, numBits, narrowRange);
    LogTilingData(context, tiling);

    // 10. BlockDim
    context->SetBlockDim(static_cast<uint32_t>(res.numCore));

    // 11. Workspace
    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = WS_SYS_SIZE;

    // 12. TilingKey 四维（D_T_X=FQ_TPL_DT_FP32, MODE=2, HAS_ZP=0, ROUND_MODE=0）
    //     注：D_T_X 已改用整数占位（见 tiling_key.h），不再从 ge::DataType 强转，
    //         避免 kernel UT/CPU 仿真编译时依赖 C_DT_FLOAT 符号。
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(FQ_TPL_DT_FP32), static_cast<uint32_t>(2),
                          static_cast<uint32_t>(0), static_cast<uint32_t>(0));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForFakeQuantWithMinMaxVarsPerChannel(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct FakeQuantWithMinMaxVarsPerChannelCompileInfo {};

IMPL_OP_OPTILING(FakeQuantWithMinMaxVarsPerChannel)
    .Tiling(FakeQuantWithMinMaxVarsPerChannelTilingFunc)
    .TilingParse<FakeQuantWithMinMaxVarsPerChannelCompileInfo>(TilingParseForFakeQuantWithMinMaxVarsPerChannel);

} // namespace optiling
