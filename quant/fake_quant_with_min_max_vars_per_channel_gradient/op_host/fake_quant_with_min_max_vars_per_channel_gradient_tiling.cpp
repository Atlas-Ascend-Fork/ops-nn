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
 * \file fake_quant_with_min_max_vars_per_channel_gradient_tiling.cpp
 * \brief Host tiling（arch35，新写法）：
 *        - TilingData 为 POD 结构体（op_kernel/arch35/..._tilingdata.h），memcpy 直写 raw tiling；
 *        - 模板参数（ASCENDC_TPL）取代旧 SetTilingKey(1001)/TILING_KEY_IS 机制：
 *          MODE=NORMAL（常规模板）/ MODE=EMPTY（空 tensor 专用模板）；
 *        - 空 tensor（numel==0）走退化分支：填安全 tiling 值 + MODE=EMPTY，无需新增运行时 tilingkey；
 *        - 入口校验：rank∈[1,8]、gradients 与 x 同形、min/max 1-D 且长度==C；
 *          分核策略带 512MB 合并 workspace 上限（超限由行切分回退 d-chunk 切分）。
 */

#include "fake_quant_with_min_max_vars_per_channel_gradient_tiling.h"

#include <cstring>

#include "../op_kernel/arch35/fake_quant_with_min_max_vars_per_channel_gradient_tilingdata.h"
#include "../op_kernel/arch35/fake_quant_with_min_max_vars_per_channel_gradient_tiling_key.h"

using namespace FakeQuantWMMVPCG;

namespace optiling {
constexpr uint32_t VL_FP32 = 64;
constexpr uint32_t RESERVED_UB = 8 * 1024; // 8 KB scratchpad
// 9 FP32 buffers all sized by dTileLen: nudgedMin/Max + bpMinAcc/Max + bpMinComp/MaxComp + g + x + bpx = 9 * 4 bytes.
constexpr uint32_t BYTES_PER_DTILE_ELEM = 36;
constexpr size_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t SPLIT_MODE_ROWS = 0;
constexpr uint32_t SPLIT_MODE_DCHUNKS = 1;
constexpr uint32_t EMPTY_DTILE_LEN = VL_FP32; // 空 tensor 模板占位 dTileLen（空模板不使用分块）
constexpr size_t ATTR_IDX_NUM_BITS = 0;
constexpr size_t ATTR_IDX_NARROW_RANGE = 1;
constexpr size_t INPUT_IDX_GRADIENTS = 0;
constexpr size_t INPUT_IDX_X = 1;
constexpr size_t INPUT_IDX_MIN = 2;
constexpr size_t INPUT_IDX_MAX = 3;
constexpr uint32_t MAX_RANK = 8;
// Upper bound for the cross-core bp_min/bp_max merge workspace of split_mode=0 (2 * cores *
// ceil(C, VL) * 4B). Above it the row split is abandoned for the bounded d-chunk split.
constexpr uint64_t MAX_ROW_MERGE_WORKSPACE_BYTES = 512ull * 1024ull * 1024ull;

class FakeQuantWithMinMaxVarsPerChannelGradientTiling {
public:
    explicit FakeQuantWithMinMaxVarsPerChannelGradientTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;

private:
    FakeQuantWithMinMaxVarsPerChannelGradientTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    uint32_t usedCoreNum_ = 0;
    uint32_t rowsPerCore_ = 0;
    uint32_t tailRows_ = 0;
    uint32_t totalRows_ = 0;
    uint32_t channelNum_ = 0;
    uint32_t tileLen_ = 0;
    int32_t quantMin_ = 0;
    int32_t quantMax_ = 0;
    uint32_t dTileLen_ = 0;
    uint32_t numDChunks_ = 0;
    uint32_t splitMode_ = 0;
    uint32_t chunksPerCore_ = 0;
    uint32_t tailChunks_ = 0;
    uint64_t ubSizePlatForm_ = 0;
    uint32_t coreNumPlatform_ = 0;
    bool isEmpty_ = false;   // true: 空 tensor（numel==0），下发 MODE=EMPTY
    uint64_t tilingKey_ = 0; // 由 GET_TPL_TILING_KEY 生成的模板参数 key
};

ge::graphStatus FakeQuantWithMinMaxVarsPerChannelGradientTiling::Init()
{
    OP_LOGD(tilingContext->GetNodeName(), "Tiling init begin.");

    auto compileInfo = tilingContext->GetCompileInfo<FakeQuantWithMinMaxVarsPerChannelGradientCompileInfo>();
    OP_CHECK_IF((compileInfo == nullptr), OP_LOGE(tilingContext, "compileInfo is nullptr"), return ge::GRAPH_FAILED);
    coreNumPlatform_ = static_cast<uint32_t>(compileInfo->totalCoreNum);
    ubSizePlatForm_ = compileInfo->ubSizePlatForm;
    if (coreNumPlatform_ == 0 || ubSizePlatForm_ == 0) {
        OP_LOGE(tilingContext, "platform info invalid: coreNum=%u, ubSize=%lu", coreNumPlatform_, ubSizePlatForm_);
        return ge::GRAPH_FAILED;
    }

    auto xShapePtr = tilingContext->GetInputShape(INPUT_IDX_X);
    OP_CHECK_IF((xShapePtr == nullptr), OP_LOGE(tilingContext, "x shape is nullptr"), return ge::GRAPH_FAILED);
    auto xShape = xShapePtr->GetStorageShape();
    size_t dimNum = xShape.GetDimNum();
    if (dimNum == 0) {
        OP_LOGE(tilingContext, "x rank is 0");
        return ge::GRAPH_FAILED;
    }
    if (dimNum > MAX_RANK) {
        OP_LOGE(tilingContext, "x rank %zu exceeds max %u", dimNum, MAX_RANK);
        return ge::GRAPH_FAILED;
    }
    int64_t numel = xShape.GetShapeSize();
    int64_t channelNum = xShape.GetDim(static_cast<int64_t>(dimNum) - 1);

    // Attrs（空 tensor 分支也需要 quantMin/quantMax，故提前解析）
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_IF((attrs == nullptr), OP_LOGE(tilingContext, "attrs is nullptr"), return ge::GRAPH_FAILED);
    int64_t numBits = 8;
    bool narrowRange = false;
    // num_bits is registered as Int (int64). Read it as int64 and range-check before narrowing:
    // reading it as int32 would only look at the low 32 bits and let out-of-range values whose
    // low word lands in [2, 16] bypass the check.
    const int64_t* numBitsPtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_NUM_BITS);
    if (numBitsPtr != nullptr) {
        numBits = *numBitsPtr;
    }
    const bool* narrowPtr = attrs->GetAttrPointer<bool>(ATTR_IDX_NARROW_RANGE);
    if (narrowPtr != nullptr) {
        narrowRange = *narrowPtr;
    }
    if (numBits < 2 || numBits > 16) {
        OP_LOGE(tilingContext, "num_bits %ld out of range [2,16]", numBits);
        return ge::GRAPH_FAILED;
    }
    quantMin_ = narrowRange ? 1 : 0;
    quantMax_ = static_cast<int32_t>((static_cast<int64_t>(1) << numBits) - 1);

    // ---- 空 tensor 退化：走 MODE=EMPTY 专用模板（kernel 侧 EmptyKernel 输出 bp_min/bp_max=0）----
    if (numel == 0) {
        isEmpty_ = true;
        usedCoreNum_ = 1;
        rowsPerCore_ = 0;
        tailRows_ = 0;
        totalRows_ = 0;
        channelNum_ = (channelNum > 0) ? static_cast<uint32_t>(channelNum) : 0;
        dTileLen_ = EMPTY_DTILE_LEN;
        tileLen_ = EMPTY_DTILE_LEN;
        numDChunks_ = (channelNum_ + EMPTY_DTILE_LEN - 1) / EMPTY_DTILE_LEN;
        splitMode_ = SPLIT_MODE_ROWS;
        chunksPerCore_ = 0;
        tailChunks_ = 0;

        size_t* emptyWorkSpace = tilingContext->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext, emptyWorkSpace);
        emptyWorkSpace[0] = SYS_WORKSPACE_SIZE; // 空模板无跨核归约，仅系统 workspace
        tilingContext->SetScheduleMode(1);
        OP_LOGI(tilingContext->GetNodeName(),
                "empty tensor detected: channelNum=%u, use MODE=EMPTY template (single core).", channelNum_);
        return ge::GRAPH_SUCCESS;
    }

    if (channelNum <= 0) {
        OP_LOGE(tilingContext, "channelNum <= 0");
        return ge::GRAPH_FAILED;
    }
    if (numel <= 0 || (numel % channelNum) != 0) {
        OP_LOGE(tilingContext, "x numel %ld not divisible by channelNum %ld", numel, channelNum);
        return ge::GRAPH_FAILED;
    }
    // Cross-input constraints (README): gradients shape must equal x shape; min/max must be 1-D
    // with length equal to the channel C. The kernel indexes gradGm_/xGm_/minGm_/maxGm_ linearly
    // by channelNum_, so a mismatch would read/write out of range. Reject here (the only entry,
    // since this op excludes the aclnn path). Empty tensors (numel==0) take the MODE=EMPTY branch
    // above and are exempt — they write no channel-indexed data.
    auto gradShapePtr = tilingContext->GetInputShape(INPUT_IDX_GRADIENTS);
    auto minShapePtr = tilingContext->GetInputShape(INPUT_IDX_MIN);
    auto maxShapePtr = tilingContext->GetInputShape(INPUT_IDX_MAX);
    OP_CHECK_IF((gradShapePtr == nullptr || minShapePtr == nullptr || maxShapePtr == nullptr),
                OP_LOGE(tilingContext, "gradients/min/max shape is nullptr"), return ge::GRAPH_FAILED);
    auto gradShape = gradShapePtr->GetStorageShape();
    auto minShape = minShapePtr->GetStorageShape();
    auto maxShape = maxShapePtr->GetStorageShape();
    if (gradShape.GetDimNum() != dimNum) {
        OP_LOGE(tilingContext, "gradients rank %zu != x rank %zu", gradShape.GetDimNum(), dimNum);
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0U; i < dimNum; ++i) {
        if (gradShape.GetDim(static_cast<int64_t>(i)) != xShape.GetDim(static_cast<int64_t>(i))) {
            OP_LOGE(tilingContext, "gradients shape mismatch with x at dim %zu", i);
            return ge::GRAPH_FAILED;
        }
    }
    if (minShape.GetDimNum() != 1U || maxShape.GetDimNum() != 1U) {
        OP_LOGE(tilingContext, "min/max must be 1-D, got min rank %zu max rank %zu", minShape.GetDimNum(),
                maxShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (minShape.GetDim(0) != channelNum || maxShape.GetDim(0) != channelNum) {
        OP_LOGE(tilingContext, "min/max length (%ld/%ld) must equal channel C (%ld)", minShape.GetDim(0),
                maxShape.GetDim(0), channelNum);
        return ge::GRAPH_FAILED;
    }
    int64_t totalRows = numel / channelNum;
    if (totalRows <= 0) {
        OP_LOGE(tilingContext, "totalRows <= 0");
        return ge::GRAPH_FAILED;
    }
    // Guard the narrowing into the uint32 TilingData fields. All later arithmetic that adds to
    // channelNum_ (ceil to VL, d-chunk count) is done in 64-bit, so representing channelNum in
    // uint32 is sufficient here; a silent truncation would still corrupt the split decisions.
    constexpr int64_t kUint32Max = static_cast<int64_t>(0xFFFFFFFFu);
    if (totalRows > kUint32Max || channelNum > kUint32Max) {
        OP_LOGE(tilingContext, "totalRows %ld or channelNum %ld exceeds uint32 range", totalRows, channelNum);
        return ge::GRAPH_FAILED;
    }
    totalRows_ = static_cast<uint32_t>(totalRows);
    channelNum_ = static_cast<uint32_t>(channelNum);

    // D-chunk size: every UB-resident buffer is sized by dTileLen, no longer by D.
    // freeBytes = ubSize - RESERVED_UB; dTileLen = floor(freeBytes / BYTES_PER_DTILE_ELEM), align down to VL=64.
    int64_t freeBytes = static_cast<int64_t>(ubSizePlatForm_) - static_cast<int64_t>(RESERVED_UB);
    if (freeBytes <= 0) {
        OP_LOGE(tilingContext, "UB %lu too small (reserved %u)", ubSizePlatForm_, RESERVED_UB);
        return ge::GRAPH_FAILED;
    }
    int64_t rawDTileLen = freeBytes / static_cast<int64_t>(BYTES_PER_DTILE_ELEM);
    int64_t alignedDTileLen = (rawDTileLen / VL_FP32) * VL_FP32;
    if (alignedDTileLen <= 0) {
        OP_LOGE(tilingContext, "dTileLen too small after alignment");
        return ge::GRAPH_FAILED;
    }
    // Cap dTileLen at ceil(D, VL): no benefit to going larger than D itself.
    // 64-bit intermediates on purpose: channelNum_ + VL_FP32 and channelNum_ + dTileLen_ wrap in
    // uint32 once channelNum_ approaches UINT32_MAX, which would collapse dCap/numDChunks to 0 and
    // make the d-chunk division below divide by zero.
    const uint64_t dCapWide = ((static_cast<uint64_t>(channelNum_) + VL_FP32 - 1) / VL_FP32) * VL_FP32;
    const int64_t dCap = dCapWide > static_cast<uint64_t>(kUint32Max) ? kUint32Max : static_cast<int64_t>(dCapWide);
    if (alignedDTileLen > dCap) {
        alignedDTileLen = dCap;
    }
    dTileLen_ = static_cast<uint32_t>(alignedDTileLen);
    numDChunks_ = static_cast<uint32_t>((static_cast<uint64_t>(channelNum_) + dTileLen_ - 1) / dTileLen_);
    // Keep tileLen_ (legacy field) identical to dTileLen_ — the kernel now uses dTileLen
    // everywhere, but old field is preserved in TilingData for compatibility.
    tileLen_ = dTileLen_;

    // Multi-core split strategy:
    //   N > 1 and bounded merge workspace -> split rows across cores (split_mode=0)
    //   otherwise                         -> split d-chunks across cores (split_mode=1)
    // split_mode=0 needs a 2*usedCoreNum*ceil(C,VL)*4B cross-core bp_min/bp_max merge workspace
    // that grows unbounded with the full channel axis; when it would exceed the cap, fall back to
    // the bounded d-chunk split (each core owns disjoint channels and writes GM directly).
    const uint32_t rowSplitCores = totalRows_ < coreNumPlatform_ ? totalRows_ : coreNumPlatform_;
    // 64-bit: ceil(channelNum_, VL) can wrap in uint32 near UINT32_MAX and would understate the
    // merge workspace, wrongly selecting the row split.
    const uint64_t rowWsStride = ((static_cast<uint64_t>(channelNum_) + VL_FP32 - 1) / VL_FP32) * VL_FP32;
    const uint64_t rowMergeWorkspaceBytes = static_cast<uint64_t>(2) * rowSplitCores * rowWsStride *
                                            static_cast<uint64_t>(sizeof(float));
    if (totalRows_ > 1 && rowMergeWorkspaceBytes <= MAX_ROW_MERGE_WORKSPACE_BYTES) {
        splitMode_ = SPLIT_MODE_ROWS;
        usedCoreNum_ = rowSplitCores;
        rowsPerCore_ = totalRows_ / usedCoreNum_;
        tailRows_ = totalRows_ % usedCoreNum_;
        chunksPerCore_ = numDChunks_;
        tailChunks_ = 0;
    } else {
        splitMode_ = SPLIT_MODE_DCHUNKS;
        usedCoreNum_ = numDChunks_ < coreNumPlatform_ ? numDChunks_ : coreNumPlatform_;
        if (usedCoreNum_ == 0) {
            usedCoreNum_ = 1;
        }
        chunksPerCore_ = numDChunks_ / usedCoreNum_;
        tailChunks_ = numDChunks_ % usedCoreNum_;
        rowsPerCore_ = totalRows_; // every core processes all rows
        tailRows_ = 0;
    }
    if (usedCoreNum_ == 0) {
        OP_LOGE(tilingContext, "usedCoreNum is 0");
        return ge::GRAPH_FAILED;
    }

    // WHY: split_mode=0 cross-core reduce per channel needs a 2*usedCoreNum*ceil(D,VL) workspace
    // for bp_min/bp_max merge; split_mode=1 partitions disjoint d-chunks across cores so each
    // core writes its own channel range directly to bp_min/bp_max GM, no cross-core reduce
    // workspace is required (avoids GB-scale workspace when channelNum is huge).
    size_t userWorkspaceBytes = 0;
    if (splitMode_ == SPLIT_MODE_ROWS) {
        uint32_t wsStride = ((channelNum_ + VL_FP32 - 1) / VL_FP32) * VL_FP32;
        userWorkspaceBytes = static_cast<size_t>(2) * usedCoreNum_ * wsStride * sizeof(float);
    }
    size_t* currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkSpace);
    currentWorkSpace[0] = SYS_WORKSPACE_SIZE + userWorkspaceBytes;
    tilingContext->SetScheduleMode(1);
    OP_LOGD(tilingContext->GetNodeName(),
            "Tiling init done. totalRows=%u channelNum=%u usedCore=%u splitMode=%u "
            "rowsPerCore=%u tailRows=%u dTileLen=%u numDChunks=%u chunksPerCore=%u tailChunks=%u "
            "qmin=%d qmax=%d",
            totalRows_, channelNum_, usedCoreNum_, splitMode_, rowsPerCore_, tailRows_, dTileLen_, numDChunks_,
            chunksPerCore_, tailChunks_, quantMin_, quantMax_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeQuantWithMinMaxVarsPerChannelGradientTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext->GetNodeName(), "RunKernelTiling start.");
    tilingContext->SetBlockDim(usedCoreNum_ > 0 ? usedCoreNum_ : 1);
    // 模板参数下发（新写法）：MODE=EMPTY 走空 tensor 专用模板；MODE=NORMAL 走 regbase 常规模板。
    tilingKey_ = GET_TPL_TILING_KEY(isEmpty_ ? FQPCG_TPL_MODE_EMPTY : FQPCG_TPL_MODE_NORMAL);
    tilingContext->SetTilingKey(tilingKey_);

    tilingData.usedCoreNum = usedCoreNum_;
    tilingData.rowsPerCore = rowsPerCore_;
    tilingData.tailRows = tailRows_;
    tilingData.totalRows = totalRows_;
    tilingData.channelNum = channelNum_;
    tilingData.tileLen = tileLen_;
    tilingData.quantMin = quantMin_;
    tilingData.quantMax = quantMax_;
    tilingData.dTileLen = dTileLen_;
    tilingData.numDChunks = numDChunks_;
    tilingData.splitMode = splitMode_;
    tilingData.chunksPerCore = chunksPerCore_;
    tilingData.tailChunks = tailChunks_;

    auto rawTiling = tilingContext->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, rawTiling);
    const size_t dataSize = sizeof(tilingData);
    errno_t ret = memcpy_s(rawTiling->GetData(), rawTiling->GetCapacity(), reinterpret_cast<void*>(&tilingData),
                           dataSize);
    if (ret != EOK) {
        OP_LOGE(tilingContext->GetNodeName(), "memcpy_s tiling failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    rawTiling->SetDataSize(dataSize);
    TilingDataPrint();
    OP_LOGD(tilingContext->GetNodeName(), "RunKernelTiling end. tilingKey=%lu empty=%d", tilingKey_,
            static_cast<int32_t>(isEmpty_));
    return ge::GRAPH_SUCCESS;
}

void FakeQuantWithMinMaxVarsPerChannelGradientTiling::TilingDataPrint() const
{
    OP_LOGI(tilingContext->GetNodeName(),
            "FakeQuantWithMinMaxVarsPerChannelGradient TilingData: usedCoreNum=%u rowsPerCore=%u tailRows=%u "
            "totalRows=%u channelNum=%u tileLen=%u quantMin=%d quantMax=%d dTileLen=%u numDChunks=%u "
            "splitMode=%u chunksPerCore=%u tailChunks=%u",
            usedCoreNum_, rowsPerCore_, tailRows_, totalRows_, channelNum_, tileLen_, quantMin_, quantMax_, dTileLen_,
            numDChunks_, splitMode_, chunksPerCore_, tailChunks_);
    OP_LOGD(tilingContext->GetNodeName(), "usedCoreNum: %u", usedCoreNum_);
    OP_LOGD(tilingContext->GetNodeName(), "rowsPerCore: %u", rowsPerCore_);
    OP_LOGD(tilingContext->GetNodeName(), "tailRows: %u", tailRows_);
    OP_LOGD(tilingContext->GetNodeName(), "totalRows: %u", totalRows_);
    OP_LOGD(tilingContext->GetNodeName(), "channelNum: %u", channelNum_);
    OP_LOGD(tilingContext->GetNodeName(), "tileLen: %u", tileLen_);
    OP_LOGD(tilingContext->GetNodeName(), "quantMin: %d", quantMin_);
    OP_LOGD(tilingContext->GetNodeName(), "quantMax: %d", quantMax_);
    OP_LOGD(tilingContext->GetNodeName(), "dTileLen: %u", dTileLen_);
    OP_LOGD(tilingContext->GetNodeName(), "numDChunks: %u", numDChunks_);
    OP_LOGD(tilingContext->GetNodeName(), "splitMode: %u", splitMode_);
    OP_LOGD(tilingContext->GetNodeName(), "chunksPerCore: %u", chunksPerCore_);
    OP_LOGD(tilingContext->GetNodeName(), "tailChunks: %u", tailChunks_);
}

static ge::graphStatus TilingFakeQuantWithMinMaxVarsPerChannelGradient(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for Arch35 architecture");
    FakeQuantWithMinMaxVarsPerChannelGradientTiling tilingObject(context);
    auto status = tilingObject.Init();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    return tilingObject.RunKernelTiling();
}

static ge::graphStatus TilingPrepareForFakeQuantWithMinMaxVarsPerChannelGradient(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForFakeQuantWithMinMaxVarsPerChannelGradient start.");
    auto compileInfo = context->GetCompiledInfo<FakeQuantWithMinMaxVarsPerChannelGradientCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_CHECK_IF((compileInfo->totalCoreNum <= 0 || compileInfo->ubSizePlatForm <= 0),
                OP_LOGE(context->GetNodeName(), "GetHardwareInfo failed, totalCoreNum:%d, ubSize:%lu.",
                        compileInfo->totalCoreNum, compileInfo->ubSizePlatForm),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "TilingPrepareForFakeQuantWithMinMaxVarsPerChannelGradient end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FakeQuantWithMinMaxVarsPerChannelGradient)
    .Tiling(TilingFakeQuantWithMinMaxVarsPerChannelGradient)
    .TilingParse<FakeQuantWithMinMaxVarsPerChannelGradientCompileInfo>(
        TilingPrepareForFakeQuantWithMinMaxVarsPerChannelGradient);

} // namespace optiling
