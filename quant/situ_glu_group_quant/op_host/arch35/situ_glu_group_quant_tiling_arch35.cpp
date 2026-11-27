/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file situ_glu_group_quant_tiling_arch35.cpp
 * \brief Tiling implementation for SituGluGroupQuant, ported from
 *        cann-recipes-infer (recipes caliber + ops-nn error reporting style).
 */

#include "situ_glu_group_quant_tiling_arch35.h"
#include "quant/situ_glu_group_quant/op_kernel/arch35/situ_glu_group_quant_tiling_key.h"

#include <cmath>
#include <sstream>
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"
#include "util/math_util.h"
#include "error_util.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace SituGluGroupQuantOp;

namespace optiling {
// ==================== Constants ====================
constexpr int64_t INDEX_ATTR_BETA = 0;
constexpr int64_t INDEX_ATTR_LINEAR_BETA = 1;
constexpr int64_t INDEX_ATTR_ACTIVATE_LEFT = 2;
constexpr int64_t INDEX_ATTR_AXIS = 3;
constexpr int64_t INDEX_ATTR_QUANT_MODE = 4;
constexpr int64_t INDEX_ATTR_DST_TYPE = 5;
constexpr int64_t INDEX_ATTR_DST_TYPE_MAX = 6;
constexpr int64_t INDEX_ATTR_ROUND_MODE = 7;
constexpr int64_t INDEX_ATTR_GROUP_LIST_TYPE = 8;

// dst_type enumeration (ACL dtype numbers, situ_mx_quant caliber)
constexpr int64_t DTYPE_FP8_E5M2 = 35;   // DT_FLOAT8_E5M2
constexpr int64_t DTYPE_FP8_E4M3FN = 36; // DT_FLOAT8_E4M3FN
constexpr int64_t DTYPE_FP4_E2M1 = 40;   // DT_FLOAT4_E2M1
constexpr int64_t DTYPE_FP4_E1M2 = 41;   // DT_FLOAT4_E1M2

constexpr int64_t INDEX_INPUT_WEIGHT = 1;
constexpr int64_t INDEX_INPUT_GROUP_INDEX = 2;

// Whole-row chunk: min(EPR, 3072). 3072 = 96 * 32 keeps the quant group (32)
// boundaries aligned to the chunk. The kernel CopyIn moves the whole row into a
// chunk-sized buffer, so EPR must not exceed this cap (onboard-found overrun).
constexpr int64_t MAX_TILE_COLS = 3072;
constexpr int64_t QUANT_BLOCK_SIZE = 32;
// Minimum EPR: groupCount >= 4 keeps the masked PACK_B16 scale store effective;
// below it the store silently drops the sub-8B payload and y_scale stays
// uninitialized (onboard-found at EPR=64, garbage identical on recipes).
constexpr int64_t MIN_TILE_COLS = 128;
// K-row batching (performance dimension, no tiling-key change): amortize the
// per-row copy-descriptor / VF-call / queue-cycle fixed costs when EPR is
// small. Gated to the region where the per-row path measurably under-performs
// the K-batch pipeline (EPR <= 2048, widened from 1024 after the two-path
// tanh register-ization); larger EPR keeps the proven per-row pipeline.
constexpr int64_t K_BATCH_EPR_MAX = 2048;
// Loose guard only (NOT a tuning knob): K itself is UB-budget driven — the
// same "fill as much as fits" philosophy as situ_glu (ubMaxBatch) and
// dynamic_mx_quant (ubFactor). Prevents pathological blockCount sizes only.
constexpr int64_t K_BATCH_MAX = 128;
// Vector access quantum of the scale-side buffers: one vector register = 128
// uint16 lanes; packed e8m0 scale bytes / uint16 maxexp / recip are padded per
// row up to this lane count (kernel InitBuffer uses CeilAlignI(x, VF_LEN_16)).
constexpr int64_t SCALE_ALIGN_LANES = 128;
// sizeof(T) of x: both supported dtypes (bf16/fp16) are 2 bytes — must stay
// in sync with the kernel template T if a new dtype is ever added. Per-slot
// byte accounting of the InitBuffer ledger uses this, NOT the recipes
// BYTES_PER_ELEMENT (that 16 B/elem constant is a WHOLE-chunk total budget
// across all buffers; mixing it into per-slot terms inflates perRow ~6.8x
// and silently shrinks K from 99/33/17 to ~20/5/2 at EPR=128/512/1024).
constexpr int64_t X_ELEM_BYTES = 2;
// K==1 且 chunk 非 256 整数倍时 kernel 会把 actBuf/yQueue 补到 256 元素 VF
// 量子（gc%8 越界防护，e24b45de8）——drift guard 需计入该垫区。
constexpr int64_t VF_WRITE_QUANT = 256;
constexpr int64_t UB_RESERVE_RATIO_NUM = 7;
constexpr int64_t UB_RESERVE_RATIO_DEN = 8;

const set<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E4M3FN};
const set<ge::DataType> SCALE_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};

// ==================== Helper Functions ====================
template <typename T>
static string Shape2String(const T& shape)
{
    ostringstream oss;
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

// ==================== Class Methods ====================
ge::graphStatus SituGluGroupQuantRegbaseTiling::GetNpuInfo()
{
    OP_LOGD(context_->GetNodeName(), "GetNpuInfo begin.");
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        // Fallback to the compile-time cached platform info (TilingPrepare), so
        // that tiling still works when the SuperKernel JIT / tiling-sink path
        // calls with a null runtime PlatformInfo (recipes caliber).
        const auto* compileInfo = context_->GetCompileInfo<SituGluGroupQuantCompileInfo>();
        if (compileInfo != nullptr && compileInfo->totalCoreNum > 0 && compileInfo->ubSize > 0) {
            compileInfo_.totalCoreNum = compileInfo->totalCoreNum;
            compileInfo_.ubSize = compileInfo->ubSize;
            OP_LOGI(context_->GetNodeName(), "GetNpuInfo uses compile-time cached info: totalCoreNum=%ld, ubSize=%ld",
                    compileInfo_.totalCoreNum, compileInfo_.ubSize);
            return ge::GRAPH_SUCCESS;
        }
        const std::string msgOpName = std::string(((context_) == nullptr || (context_)->GetNodeName() == nullptr ?
                                                       "situ_glu_group_quant" :
                                                       (context_)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Tiling framework object platformInfo is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object platformInfo is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo_.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if ((compileInfo_.totalCoreNum <= 0)) {
        const std::string msgOpName = (context_ == nullptr || context_->GetNodeName() == nullptr ?
                                           "situ_glu_group_quant" :
                                           context_->GetNodeName());
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", "Failed to get core num.");
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), "Failed to get core num."}));
        return ge::GRAPH_FAILED;
    };
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo_.ubSize = static_cast<int64_t>(ubSize);
    if ((compileInfo_.ubSize <= 0)) {
        const std::string msgOpName = (context_ == nullptr || context_->GetNodeName() == nullptr ?
                                           "situ_glu_group_quant" :
                                           context_->GetNodeName());
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", "Failed to get UB size.");
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), "Failed to get UB size."}));
        return ge::GRAPH_FAILED;
    };
    OP_LOGI(context_->GetNodeName(), "CompileInfo: totalCoreNum=%ld, ubSize=%ld", compileInfo_.totalCoreNum,
            compileInfo_.ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGroupQuantRegbaseTiling::ParseAttrs()
{
    auto* attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(((context_) == nullptr || (context_)->GetNodeName() == nullptr ?
                                                       "situ_glu_group_quant" :
                                                       (context_)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(
            msgOpName.c_str(), "%s",
            "Tiling framework object attrs is unavailable; check the operator registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    auto* attrBeta = attrs->GetAttrPointer<float>(INDEX_ATTR_BETA);
    attrParam_.beta = (attrBeta != nullptr) ? *attrBeta : 1.0f;
    OP_CHECK_IF((attrParam_.beta <= 0.0f),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "beta", std::to_string(attrParam_.beta),
                    ("beta must be greater than 0, but got " + std::to_string(attrParam_.beta) + ".")),
                return ge::GRAPH_FAILED);

    auto* attrLinearBeta = attrs->GetAttrPointer<float>(INDEX_ATTR_LINEAR_BETA);
    // <= 0 disables the up-branch tanh soft-clamp (up passthrough, TPL
    // dimension); > 0 enables it. Any float is accepted — the disable/enable
    // split is a compile-time TPL dimension, not a validation failure.
    attrParam_.linearBeta = (attrLinearBeta != nullptr) ? *attrLinearBeta : 0.0f;
    OP_LOGI(context_->GetNodeName(), "ParseAttrs: linearBeta=%f (disabled: %d)", attrParam_.linearBeta,
            static_cast<int>(attrParam_.linearBeta <= 0.0f));

    auto* attrActivateLeft = attrs->GetAttrPointer<bool>(INDEX_ATTR_ACTIVATE_LEFT);
    attrParam_.activateLeft = (attrActivateLeft != nullptr) ? *attrActivateLeft : false;
    OP_LOGI(context_->GetNodeName(), "ParseAttrs: activateLeft=%d", static_cast<int>(attrParam_.activateLeft));

    // ---- Extension attrs (situ_mx_quant caliber): validated-only, single supported value each ----
    auto* attrAxis = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    attrParam_.axis = (attrAxis != nullptr) ? static_cast<int64_t>(*attrAxis) : -1;
    OP_CHECK_IF((attrParam_.axis != -1),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "axis", std::to_string(attrParam_.axis),
                    ("Only axis=-1 is supported currently, but got " + std::to_string(attrParam_.axis) + ".")),
                return ge::GRAPH_FAILED);

    auto* attrQuantMode = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_QUANT_MODE);
    attrParam_.quantMode = (attrQuantMode != nullptr) ? static_cast<int64_t>(*attrQuantMode) : 0;
    OP_CHECK_IF((attrParam_.quantMode != 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "quant_mode", std::to_string(attrParam_.quantMode),
                    ("Only quant_mode=0 (mx quantization) is supported currently, but got " +
                     std::to_string(attrParam_.quantMode) + ".")),
                return ge::GRAPH_FAILED);

    auto* attrDstType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
    attrParam_.dstType = (attrDstType != nullptr) ? static_cast<int64_t>(*attrDstType) : DTYPE_FP8_E4M3FN;
    OP_CHECK_IF(
        (attrParam_.dstType != DTYPE_FP8_E5M2) && (attrParam_.dstType != DTYPE_FP8_E4M3FN) &&
            (attrParam_.dstType != DTYPE_FP4_E2M1) && (attrParam_.dstType != DTYPE_FP4_E1M2),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dst_type", std::to_string(attrParam_.dstType),
                                              ("Invalid dst_type: " + std::to_string(attrParam_.dstType) +
                                               ", only 35(E5M2), 36(E4M3FN), 40(E2M1), 41(E1M2) supported.")),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (attrParam_.dstType != DTYPE_FP8_E4M3FN),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dst_type", std::to_string(attrParam_.dstType),
                                              ("Only dst_type=36 (FP8_E4M3FN) is supported currently, but got " +
                                               std::to_string(attrParam_.dstType) + ".")),
        return ge::GRAPH_FAILED);

    auto* attrDstTypeMax = attrs->GetAttrPointer<float>(INDEX_ATTR_DST_TYPE_MAX);
    attrParam_.dstTypeMax = (attrDstTypeMax != nullptr) ? *attrDstTypeMax : 0.0f;
    OP_CHECK_IF((attrParam_.dstTypeMax < 0.0f),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "dst_type_max", std::to_string(attrParam_.dstTypeMax),
                    ("dst_type_max must be >= 0, but got " + std::to_string(attrParam_.dstTypeMax) + ".")),
                return ge::GRAPH_FAILED);

    const char* attrRoundMode = attrs->GetAttrPointer<char>(INDEX_ATTR_ROUND_MODE);
    std::string roundModeStr = (attrRoundMode != nullptr) ? attrRoundMode : "rint";
    OP_CHECK_IF((roundModeStr != "rint" && roundModeStr != "round" && roundModeStr != "floor"),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "round_mode", roundModeStr,
                    ("Invalid round_mode: " + roundModeStr + ", only rint, round, floor supported.")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((roundModeStr != "rint"),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "round_mode", roundModeStr,
                    ("Only round_mode=rint is supported currently, but got " + roundModeStr + ".")),
                return ge::GRAPH_FAILED);
    attrParam_.roundMode = roundModeStr;

    // group_list_type: 0=cumsum (reserved), 1=count (current). The default is 0,
    // so an omitted attr is rejected the same way as an explicit 0 — cumsum is
    // not implemented yet and callers must explicitly opt into count mode.
    auto* attrGroupListType = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_GROUP_LIST_TYPE);
    attrParam_.groupListType = (attrGroupListType != nullptr) ? static_cast<int64_t>(*attrGroupListType) : 0;
    OP_CHECK_IF((attrParam_.groupListType != 0 && attrParam_.groupListType != 1),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "group_list_type", std::to_string(attrParam_.groupListType),
                    ("Invalid group_list_type: " + std::to_string(attrParam_.groupListType) +
                     ", only 0(cumsum), 1(count) supported.")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((attrParam_.groupListType != 1),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "group_list_type", std::to_string(attrParam_.groupListType),
                    ("group_list_type=0 (cumsum mode) is reserved and not supported yet; "
                     "explicitly pass group_list_type=1 (count mode) instead.")),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGroupQuantRegbaseTiling::ValidateInput()
{
    // recipes caliber (TilingForSituGluGroupQuant input checks, kept verbatim
    // in semantics): shape non-null, rank >= 1, last dim positive and even.
    // No empty-tensor rejection: a zero-size leading dim (e.g. [0, 128]) passes
    // and degenerates to blockDim/validRows of 0, exactly like recipes. No input
    // dtype check either (dtype legality is enforced by the def registration).
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_IF(xShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context_) == nullptr || (context_)->GetNodeName() == nullptr ? "situ_glu_group_quant" :
                                                                                     (context_)->GetNodeName()),
                    "x"),
                return ge::GRAPH_FAILED);
    int64_t dimNum = static_cast<int64_t>(xShape->GetStorageShape().GetDimNum());
    OP_LOGI(context_->GetNodeName(), "Input x shape = %s", Shape2String(xShape->GetStorageShape()).c_str());
    inputInfo_.dimNum = dimNum;
    inputInfo_.lastDim = xShape->GetStorageShape().GetDim(dimNum - 1);
    OP_CHECK_IF(
        (dimNum < 1 || inputInfo_.lastDim <= 0 || inputInfo_.lastDim % 2 != 0),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context_->GetNodeName(), "x", std::to_string(inputInfo_.lastDim),
            ("x last dimension must be positive and even, but got " + std::to_string(inputInfo_.lastDim) + ".")),
        return ge::GRAPH_FAILED);

    // group_index is OPTIONAL (owner decision): absent -> whole-tensor mode
    // (fullRowsMode = 1, all rows processed); present -> recipes grouped semantics.
    // Emptiness of a PRESENT tensor is not rejected (recipes caliber).
    auto groupIndexShape = context_->GetOptionalInputShape(INDEX_INPUT_GROUP_INDEX);
    if (groupIndexShape != nullptr) {
        groupIndexCount_ = groupIndexShape->GetStorageShape().GetShapeSize();
        fullRowsMode_ = 0;
    } else {
        // optional input not instantiated: whole-tensor mode, all rows valid
        groupIndexCount_ = 0;
        fullRowsMode_ = 1;
    }

    // Optional weight (owner decision): per-token FP32 scalar scale, length N
    // (= totalRows, global row index). Only instantiation is recorded here;
    // the numel contract (== totalRows, owner spec: "weight element count equals
    // the flattened input row count") is enforced below — without it the kernel's
    // per-row weightRawPtr_[row] read walks out of bounds for a short weight
    // (edge-suite finding E1, UB onboard) and silently ignores a long one.
    const auto* weightShape = context_->GetOptionalInputShape(INDEX_INPUT_WEIGHT);
    hasWeight_ = (weightShape != nullptr) ? 1 : 0;

    // Collapse leading dims into totalRows (flattened total rows, may include padding tail)
    int64_t totalRows = 1;
    for (int64_t i = 0; i < dimNum - 1; i++) {
        totalRows *= xShape->GetStorageShape().GetDim(i);
    }
    inputInfo_.totalRows = totalRows;

    if (hasWeight_) {
        int64_t weightNumel = 1;
        const int64_t weightDimNum = weightShape->GetStorageShape().GetDimNum();
        for (int64_t i = 0; i < weightDimNum; i++) {
            weightNumel *= weightShape->GetStorageShape().GetDim(i);
        }
        OP_CHECK_IF((weightNumel != inputInfo_.totalRows),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context_->GetNodeName(), "weight", std::to_string(weightNumel),
                        ("weight element count must equal the flattened input row count (totalRows=" +
                         std::to_string(inputInfo_.totalRows) + "), but got " + std::to_string(weightNumel) +
                         " (per-token scale indexed by global row).")),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGroupQuantRegbaseTiling::ValidateOutput()
{
    // Output dtypes are fixed (y = e4m3fn, y_scale = e8m0). recipes itself does
    // not validate outputs in tiling; this is an early-error defense in depth —
    // the same inputs are rejected by the def combination (graph mode) and the
    // binary simplifiedKey (eager) anyway, one layer earlier here (owner decision,
    // same caliber as the sibling ops like grouped_dynamic_mx_quant).
    auto yDtype = context_->GetOutputDesc(0)->GetDataType();
    OP_CHECK_IF(
        (Y_SUPPORT_DTYPE_SET.find(yDtype) == Y_SUPPORT_DTYPE_SET.end()),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y", std::to_string(static_cast<int>(yDtype)),
                                              ("Output y dtype " + std::to_string(static_cast<int>(yDtype)) +
                                               " is not supported. Only FLOAT8_E4M3FN is supported.")),
        return ge::GRAPH_FAILED);
    outputInfo_.yDtype = yDtype;

    auto y_scaleDtype = context_->GetOutputDesc(1)->GetDataType();
    OP_CHECK_IF((SCALE_SUPPORT_DTYPE_SET.find(y_scaleDtype) == SCALE_SUPPORT_DTYPE_SET.end()),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "y_scale", std::to_string(static_cast<int>(y_scaleDtype)),
                    ("Output y_scale dtype " + std::to_string(static_cast<int>(y_scaleDtype)) +
                     " is not supported. Only FLOAT8_E8M0 is supported.")),
                return ge::GRAPH_FAILED);
    outputInfo_.y_scaleDtype = y_scaleDtype;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGroupQuantRegbaseTiling::CalculateTiling()
{
    epr_ = inputInfo_.lastDim / 2;
    totalElements_ = inputInfo_.totalRows * epr_;
    chunkElements_ = std::min(epr_, MAX_TILE_COLS);

    // The kernel CopyIn moves the whole row (epr*2 elements) into a chunk-sized
    // buffer — chunk_ == epr_ is the kernel's own assumption (see CopyIn in the
    // ported base.h). An EPR beyond MAX_TILE_COLS therefore overruns UB
    // (onboard-found: EPR=4096 yields ~98% wrong y via UB trampling). recipes
    // never guards this: its UB budget check uses the clamped chunk, so it can
    // never fire beyond the cap. Guard here per owner decision (fix-forward of
    // the upstream granularity bug, the single deliberate deviation from the
    // recipes tiling).
    OP_CHECK_IF((epr_ > MAX_TILE_COLS),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context_->GetNodeName(), "x", std::to_string(inputInfo_.lastDim),
                    ("EPR (x last dim / 2) must not exceed " + std::to_string(MAX_TILE_COLS) +
                     " (whole-row chunk cap), but got " + std::to_string(epr_) + ".")),
                return ge::GRAPH_FAILED);

    // Owner decision (onboard-verified fix-forward, deliberate deviation from
    // recipes which crashes / silently corrupts on both points):
    //   1. EPR not a multiple of 32: the CopyIn else-branch moves epr*2 bytes
    //      (not 32B-aligned; onboard-found EPR=129 faults the AIV with MTE
    //      error 80 on both this port and recipes), and the kernel
    //      groupCount = EPR/32 truncates the tail group.
    //   2. EPR < 128: groupCount < 4 lets the masked PACK_B16 scale store drop
    //      the sub-8B payload — onboard-found EPR=64 leaves y_scale
    //      uninitialized on both this port and recipes.
    OP_CHECK_IF((epr_ % QUANT_BLOCK_SIZE != 0),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context_->GetNodeName(), "x", std::to_string(inputInfo_.lastDim),
                    ("EPR (x last dim / 2) must be a multiple of " + std::to_string(QUANT_BLOCK_SIZE) + ", but got " +
                     std::to_string(epr_) + ".")),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((epr_ < MIN_TILE_COLS),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context_->GetNodeName(), "x", std::to_string(inputInfo_.lastDim),
                    ("EPR (x last dim / 2) must be at least " + std::to_string(MIN_TILE_COLS) +
                     " (keeps the y_scale store effective), but got " + std::to_string(epr_) + ".")),
                return ge::GRAPH_FAILED);

    // e8m0 scales per row: ceil(EPR/32) groups, rounded up to even (interleaved layout)
    int64_t groupCount = (epr_ + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE;
    scaleColNum_ = (groupCount + 1) / 2 * 2;

    // K-row batching budget, per the ACTUAL kernel InitBuffer ledger (TPipe:
    // arg3 = per-slot bytes, total = num x len; source-verified):
    //   gateupQueue  DB2 : 2 slots x (2*EPR*T)   = 4*EPR*T
    //   yQueue       DB2 : 2 slots x EPR*1B      = 2*EPR
    //   mxScaleQueue DB2 : 2 x align(gc,128)*1B  (PACK_B16: 1 byte per group)
    //   actBuf       TBuf: EPR*T
    //   maxExpBuf    TBuf: align(gc,128)*2B      (uint16 per group)
    //   recipBuf     TBuf: align(gc,128)*2B
    //   weightQueue  DB2 (hasWeight only): 2 x align(4K,32)B <= 8K+64
    // Odd gc (scaleColNum = gc+1): K-batching is allowed too — the per-batch
    // scale copy-out uses the per-row Compact block path (1-byte-granularity
    // strides, swiglu_mx_quant caliber) instead of the flat contiguous copy.
    // UB budget note: within the K-batch region (EPR <= 2048, gc <= 64) the
    // 128-lane alignment absorbs the +1 tail byte — align(gc+1,128) ==
    // align(gc,128) — so the InitBuffer ledger and this budget are unchanged.
    // hasWeight does not disable batching — weights ride a dedicated UB queue
    // (CopyIn batch-copies the K contiguous fp32 scalars; the activation VF
    // loops rows reading them from UB, one call per batch).
    rowsPerBatch_ = 1;
    // 逐槽字节记账（sizeof(T) 口径，与 InitBuffer 一致）：
    //   gateupQueue DB2: 2 slots × (2·EPR·T) = 4·EPR·T
    //   yQueue      DB2: 2 × EPR·1B          = 2·EPR
    //   actBuf      TBuf: EPR·T
    //   mxScaleQueue DB2: 2·align(gc,128)·1B
    //   maxExpBuf + recipBuf TBuf: 2·align(gc,128)·2B
    // => perRow = (4+1)·EPR·T + 2·EPR + 6·alignGc = 12·EPR + 6·alignGc (T=2)
    const int64_t alignGc = (groupCount + SCALE_ALIGN_LANES - 1) / SCALE_ALIGN_LANES * SCALE_ALIGN_LANES;
    const int64_t perRowBase = 4 * epr_ * X_ELEM_BYTES + 2 * epr_ + epr_ * X_ELEM_BYTES + 2 * alignGc +
                               2 * (alignGc * 2);
    if (compileInfo_.ubSize > 0) {
        const int64_t budget = compileInfo_.ubSize * UB_RESERVE_RATIO_NUM / UB_RESERVE_RATIO_DEN;
        if (epr_ <= K_BATCH_EPR_MAX) {
            int64_t k;
            if (hasWeight_) {
                k = (budget - 64) / (perRowBase + 8); // weightQueue amortized: <= 8B/row + 64B
            } else {
                k = budget / perRowBase;
            }
            rowsPerBatch_ = k < 1 ? 1 : (k > K_BATCH_MAX ? K_BATCH_MAX : k);
            // 负载均衡约束：行数允许时保持批数 >= 2x 核数。UB 驱动的大 K 在
            // 中等行数下会使批数低于核数——部分核整批空转（实测 rows=4096、
            // cores≈50、EPR=128 时 K=96 仅 43 批，比 K=19 的 216 批慢 48%；
            // 而 rows=16384 批数充裕时大 K 净收益 3.31x）。行数不足时退化为
            // K=1，由 blockDim 按行均摊。
            if (compileInfo_.totalCoreNum > 0 && inputInfo_.totalRows > 0) {
                const int64_t kBalance = std::max<int64_t>(1, inputInfo_.totalRows / (compileInfo_.totalCoreNum * 2));
                rowsPerBatch_ = std::min(rowsPerBatch_, kBalance);
            }
            // VF 窗口对齐约束（必须在均衡收缩之后执行——均衡改 K 会破坏对齐）：
            // 量化 VF（ComputeMaxExpOcpVF/ComputeDataVF）每轮处理固定 8 组
            // = 256 元素，LoopMask 是否约束 Load/Store 的内存访问宽度无文档
            // 背书（实测掩码生效、满批零污染，但不赌未定义行为）。收缩 K 使
            // 满批组数 gc*K 为 8 的倍数——最后一条 VF 窗口恰好收在 actBuf/
            // yQueue 槽尾内；奇 gc 时 K 向下取 8 的倍数（偶 gc 不受影响）。
            if (rowsPerBatch_ > 1) {
                const int64_t kAlign = (rowsPerBatch_ / 8) * 8;
                if (kAlign >= 1 && (groupCount * rowsPerBatch_) % 8 != 0) {
                    rowsPerBatch_ = kAlign;
                }
            }
            OP_LOGI(context_->GetNodeName(),
                    "K-row batching (budget-driven): EPR=%ld, perRow=%ld, hasWeight=%ld, "
                    "K=%ld",
                    epr_, perRowBase, hasWeight_, rowsPerBatch_);
        }
        // Drift guard（全 EPR 生效，含 >1024 的 K=1 区域——恢复 recipes 整块
        // 校验的防御本意）：派生 K 必须满足完整 InitBuffer 台账。K==1 且
        // chunk 非 256 整数倍时 kernel 会 pad actBuf/yQueue 到 256 量子
        // （e24b45de8 越界防护），垫区一并计入。
        int64_t padBytes = 0;
        if (rowsPerBatch_ == 1 && epr_ % VF_WRITE_QUANT != 0) {
            const int64_t padded = (epr_ + VF_WRITE_QUANT - 1) / VF_WRITE_QUANT * VF_WRITE_QUANT;
            // actBuf(T) + yQueue(1B) + gateupQueue(两半区, 2*T)
            padBytes = (padded - epr_) * (X_ELEM_BYTES + 1 + 2 * X_ELEM_BYTES);
        }
        const int64_t weightBytes = hasWeight_ ? 2 * ((rowsPerBatch_ * 4 + 31) / 32 * 32) : 0;
        OP_CHECK_IF((rowsPerBatch_ * perRowBase + padBytes + weightBytes > budget),
                    OP_LOGE(context_->GetNodeName(), "K-row batching ledger overflow: internal error"),
                    return ge::GRAPH_FAILED);
    }

    // (The recipes single-chunk UB check — chunk × 16 B/elem total budget ≤
    // ubSize × 7/8 — is subsumed by the drift guard above: perRowBase accounts
    // every per-row buffer by exact per-slot bytes, and the guard now runs for
    // ALL EPR, including the >1024 K=1 region.)

    // row-level core dispatch: blockDim = min(coreNum, totalRows)
    blockDim_ = std::min(compileInfo_.totalCoreNum, inputInfo_.totalRows);
    if (blockDim_ < 1) {
        blockDim_ = 1;
    }
    OP_LOGI(context_->GetNodeName(),
            "Tiling: totalRows=%ld, EPR=%ld, chunk=%ld, scaleColNum=%ld, groupIndexCount=%ld, "
            "blockDim=%ld, fullRowsMode=%ld",
            inputInfo_.totalRows, epr_, chunkElements_, scaleColNum_, groupIndexCount_, blockDim_, fullRowsMode_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGroupQuantRegbaseTiling::FillTilingData()
{
    tilingData_ = context_->GetTilingData<SituGluGroupQuantTilingData>();
    if (tilingData_ == nullptr) {
        const std::string msgOpName = (context_ == nullptr || context_->GetNodeName() == nullptr ?
                                           "situ_glu_group_quant" :
                                           context_->GetNodeName());
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", "get tilingdata ptr failed");
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), "get tilingdata ptr failed"}));
        return ge::GRAPH_FAILED;
    };
    if ((memset_s(tilingData_, sizeof(SituGluGroupQuantTilingData), 0, sizeof(SituGluGroupQuantTilingData)) != EOK)) {
        const std::string msgOpName = (context_ == nullptr || context_->GetNodeName() == nullptr ?
                                           "situ_glu_group_quant" :
                                           context_->GetNodeName());
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", "memset tilingData failed");
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), "memset tilingData failed"}));
        return ge::GRAPH_FAILED;
    };

    // recipes field names (kernel body references tiling_->xxx verbatim)
    tilingData_->totalElements = totalElements_;
    tilingData_->outputElementsPerRow = epr_;
    tilingData_->groupIndexCount = groupIndexCount_;
    tilingData_->fullRowsMode = fullRowsMode_;
    tilingData_->chunkElements = chunkElements_;
    tilingData_->beta = attrParam_.beta;
    tilingData_->linearBeta = attrParam_.linearBeta;
    tilingData_->blockSize = static_cast<int32_t>(QUANT_BLOCK_SIZE);
    tilingData_->scaleColNum = scaleColNum_;
    tilingData_->rowsPerBatch = rowsPerBatch_;
    tilingData_->activateLeft = attrParam_.activateLeft ? 1 : 0;
    tilingData_->hasWeight = hasWeight_;
    return ge::GRAPH_SUCCESS;
}

void SituGluGroupQuantRegbaseTiling::SetTilingKeyAndCore()
{
    // TPL dimensions (swiglu_mx_quant groupIndex/axisLast/roundMode caliber):
    // optional-input presence + activate_left + linear_beta disable are all
    // compile-time selected; the kernel bodies are runtime-branch-free.
    const uint64_t groupIndexMode = (fullRowsMode_ == 0) ? TPL_HAS_GROUP_INDEX : TPL_NO_GROUP_INDEX;
    const uint64_t weightMode = (hasWeight_ != 0) ? TPL_HAS_WEIGHT : TPL_NO_WEIGHT;
    const uint64_t activateLeftMode = (attrParam_.activateLeft) ? TPL_ACTIVATE_LEFT : TPL_ACTIVATE_NOT_LEFT;
    const uint64_t groupListTypeMode = (attrParam_.groupListType == 1) ? TPL_GROUP_LIST_COUNT : TPL_GROUP_LIST_CUMSUM;
    const uint64_t linearBetaMode = (attrParam_.linearBeta <= 0.0f) ? TPL_LINEAR_DISABLED : TPL_LINEAR_ENABLED;
    const int64_t tilingKey = GET_TPL_TILING_KEY(groupIndexMode, weightMode, activateLeftMode, groupListTypeMode,
                                                 linearBetaMode);
    OP_LOGI(context_->GetNodeName(),
            "groupIndexMode=%lu, weightMode=%lu, activateLeftMode=%lu, groupListTypeMode=%lu, "
            "linearBetaMode=%lu, tilingKey=%ld",
            groupIndexMode, weightMode, activateLeftMode, groupListTypeMode, linearBetaMode, tilingKey);
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(blockDim_);
}

void SituGluGroupQuantRegbaseTiling::PrintTilingData() const
{
    OP_LOGI(context_->GetNodeName(),
            "TilingData: totalElements=%ld, outputElementsPerRow=%ld, groupIndexCount=%ld, chunkElements=%ld, "
            "beta=%f, linearBeta=%f, blockSize=%d, scaleColNum=%ld, fullRowsMode=%ld, activateLeft=%ld, "
            "hasWeight=%ld",
            tilingData_->totalElements, tilingData_->outputElementsPerRow, tilingData_->groupIndexCount,
            tilingData_->chunkElements, tilingData_->beta, tilingData_->linearBeta, tilingData_->blockSize,
            tilingData_->scaleColNum, tilingData_->fullRowsMode, tilingData_->activateLeft, tilingData_->hasWeight);
}

// ==================== Entry Functions ====================
ge::graphStatus Tiling4SituGluGroupQuant(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do Tiling4SituGluGroupQuant");
    SituGluGroupQuantRegbaseTiling tiling(context);

    OP_CHECK_IF(tiling.GetNpuInfo() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "GetNpuInfo failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.ParseAttrs() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "ParseAttrs failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.ValidateInput() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "ValidateInput failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.ValidateOutput() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "ValidateOutput failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.CalculateTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "CalculateTiling failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.FillTilingData() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "FillTilingData failed"), return ge::GRAPH_FAILED);
    tiling.SetTilingKeyAndCore();
    tiling.PrintTilingData();

    // Workspace: the kernel keeps all intermediates in UB and calls no libapi
    // (no sysWorkspace dependency), so set 0 directly per the latest caliber.
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "situ_glu_group_quant" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Tiling framework object currentWorkspace is unavailable; check the operator "
                               "registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object currentWorkspace is unavailable; "
                                                         "check the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    *currentWorkspace = 0; // SituGluGroupQuant kernel does not need workspace

    OP_LOGI(context->GetNodeName(), "End to do Tiling4SituGluGroupQuant");
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus TilingPrepare4SituGluGroupQuant(gert::TilingParseContext* context)
{
    // TilingParse: cache platform info at compile time for the SuperKernel / GE
    // JIT path, so TilingFor* has coreNum/ubSize even when PlatformInfo is null
    // at run. (recipes TilingPrepareForSituGluGroupQuant, kept verbatim.)
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SituGluGroupQuant", "Tiling parse context is null"),
               return ge::GRAPH_FAILED);
    auto* compileInfo = context->GetCompiledInfo<SituGluGroupQuantCompileInfo>();
    OPS_ERR_IF(compileInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SituGluGroupQuant", "Compile info is null"),
               return ge::GRAPH_FAILED);
    auto* platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr,
               OPS_REPORT_VECTOR_INNER_ERR("SituGluGroupQuant", "Tiling parse platform info is null"),
               return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    compileInfo->totalCoreNum = static_cast<int64_t>(platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    compileInfo->socVersion = platform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}
// ==================== Registration ====================
IMPL_OP_OPTILING(SituGluGroupQuant)
    .Tiling(Tiling4SituGluGroupQuant)
    .TilingParse<SituGluGroupQuantCompileInfo>(TilingPrepare4SituGluGroupQuant);

} // namespace optiling
