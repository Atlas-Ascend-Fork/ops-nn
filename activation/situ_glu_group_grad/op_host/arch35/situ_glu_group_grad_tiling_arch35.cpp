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
 * \file situ_glu_group_grad_tiling_arch35.cpp
 * \brief Tiling implementation for SituGluGroupGrad Arch35 (Ascend 950), with optional weight / group_index.
 * Tiling structure mirrors SituGluGroup (which mirrors SwigluGroup).
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <graph/utils/type_utils.h>
#include "base/err_msg.h"
#include "activation/situ_glu_group_grad/op_kernel/arch35/situ_glu_group_grad_tiling_key.h"
#include "situ_glu_group_grad_tiling_arch35.h"

using namespace ge;
namespace optiling {
namespace {
constexpr int64_t GRADY_INDEX = 0;
constexpr int64_t X_INDEX = 1;
constexpr int64_t WEIGHT_INDEX = 2;
constexpr int64_t GROUP_INDEX_INDEX = 3;
constexpr int64_t GRADX_INDEX = 0;
constexpr int64_t GRADWEIGHT_INDEX = 1;
constexpr int64_t DIM_ATTR_INDEX = 0;
constexpr int64_t BETA_ATTR_INDEX = 1;
constexpr int64_t LINEAR_BETA_ATTR_INDEX = 2;
constexpr int64_t ACTIVATE_LEFT_ATTR_INDEX = 3;
constexpr int64_t GROUP_LIST_TYPE_ATTR_INDEX = 4;

// group_list_type: how group_index expresses the group sizes.
constexpr int64_t GROUP_LIST_TYPE_CUMSUM = 0;
constexpr int64_t GROUP_LIST_TYPE_COUNT = 1;

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t B16_BYTES = 2;
constexpr int64_t B32_BYTES = 4;
constexpr int64_t B32_ALIGN_NUM = BLOCK_SIZE / B32_BYTES;
// Ascend950 cacheline size is 512B. Split d-factor by one cacheline so each chunk is cacheline
// aligned; the element count per cacheline depends on the x dtype size.
constexpr int64_t ASCEND950_CACHE_LINE_BYTES = 512;
// The group_index groupIndexFactor is aligned down to this many elements (16 * 8B = 128B).
constexpr int64_t GROUP_INDEX_ALIGN_ELEMS = 16;
// The VF tail chunk is loaded with a full vector-length LoadAlign (not masked), which may read up
// to (VL - 1) elements past the aligned row end. Keep a small reserve so the last buffer can never
// over-read past UB, and leave headroom for any framework/stack usage.
constexpr uint64_t UB_RESERVE_BYTES = 8 * 1024;
constexpr size_t MAX_DIM_NUM = 8;
// Initial split factor used when shrinking groupIndexFactor: halving starts from 2 chunks.
constexpr int64_t MIN_GROUP_INDEX_SPLIT_FACTOR = 2;
// Number of per-row UB buffers used by the kernel: grad_y + gate + up + grad_gate + grad_up.
constexpr int64_t UB_BUFFER_NUM = 5;

constexpr float BETA_DEFAULT = 1.0f;
constexpr float LINEAR_BETA_DEFAULT = 0.0f;

constexpr int64_t INT64_MAX_VALUE = std::numeric_limits<int64_t>::max();
constexpr int64_t UINT32_MAX_VALUE = std::numeric_limits<uint32_t>::max();

// The Tiling layer is the only layer that observes these platform facts, so it owns the
// single formal first report instead of only writing a PLOG. EZ1008 is Execution_Error
// ("Failed to execute operator %s. Reason: %s."), which matches a failed tiling execution.
void ReportTilingFailure(const char* opName, const char* reason)
{
    REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                              (std::vector<const char*>{opName, reason}));
}

// Overflow-checked arithmetic. Signed overflow is undefined behavior, and a post-multiply sign
// check is not a valid overflow guard (a wrapped product can land on a positive value), so every
// shape-product / UB-size multiplication is guarded *before* the multiply.
bool MulOverflow(int64_t a, int64_t b, int64_t& out) { return __builtin_mul_overflow(a, b, &out); }
bool AddOverflow(int64_t a, int64_t b, int64_t& out) { return __builtin_add_overflow(a, b, &out); }

// Saturating variants used by the UB-budget math: on overflow they return INT64_MAX, which makes
// the following "<= ubSize_" comparison fail safe (treat the tile as too large) instead of wrapping.
int64_t SaturatingMul(int64_t a, int64_t b)
{
    int64_t out = 0;
    if (MulOverflow(a, b, out)) {
        return INT64_MAX_VALUE;
    }
    return out;
}
int64_t SaturatingAdd(int64_t a, int64_t b)
{
    int64_t out = 0;
    if (AddOverflow(a, b, out)) {
        return INT64_MAX_VALUE;
    }
    return out;
}

int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y <= 0) {
        return x;
    }
    int64_t numerator = 0;
    if (AddOverflow(x, y - 1, numerator)) {
        return INT64_MAX_VALUE;
    }
    return numerator / y;
}
int64_t DownAlign(int64_t x, int64_t y)
{
    if (y <= 0) {
        return x;
    }
    return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y) { return SaturatingMul(CeilDiv(x, y), y); }

// Returns false when a dim is negative or the element-count product overflows int64.
bool ShapeElementNum(const gert::Shape& shape, int64_t& elementNum)
{
    elementNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        int64_t dim = shape.GetDim(i);
        if (dim < 0 || MulOverflow(elementNum, dim, elementNum)) {
            return false;
        }
    }
    return true;
}

class SituGluGroupGradArch35Tiling {
public:
    explicit SituGluGroupGradArch35Tiling(gert::TilingContext* context) : context_(context) {}

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    gert::TilingContext* context_ = nullptr;
    SituGluGroupGradTilingData tilingData_;

private:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus CheckGradYInfo();
    ge::graphStatus CheckWeightInfo();
    ge::graphStatus CheckGroupIndexInfo();
    ge::graphStatus CheckOutputInfo();
    ge::graphStatus CalcGroupIndexTiling();
    ge::graphStatus CalcOpTiling();
    void InitCoreTiling();
    void SetFullDTiling();
    void SetSplitDTiling();
    void SetRowLoopTiling();
    int64_t AddWeightSize(int64_t totalSize, int64_t rowFactor) const;
    int64_t CalcTotalSize(int64_t rowFactor, int64_t dFactor) const;
    void CalcDAndRowFactorTiling(int64_t rowOnceLoop, int64_t dStep);
    void SetEmptyTiling();
    void FillTilingData();
    void PrintTilingInfo();

    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t workspaceSize_ = 0;
    uint64_t usedCoreNums_ = 0;
    int64_t xDims_ = 0;
    int64_t cutDim_ = 0;
    int64_t cutDimSize_ = 0;
    int64_t bs_ = 1;
    int64_t d_ = 1;
    int64_t splitD_ = 0;
    int64_t xElemBytes_ = B16_BYTES;
    ge::DataType xDtype_ = ge::DT_FLOAT;
    float beta_ = BETA_DEFAULT;
    float linearBeta_ = LINEAR_BETA_DEFAULT;
    int64_t activateLeft_ = 1;
    int64_t groupListType_ = GROUP_LIST_TYPE_CUMSUM;
    bool hasWeight_ = false;
    bool hasGroupIndex_ = false;
    int64_t groupIndexElemNum_ = 0;
    int64_t groupIndexLoop_ = 0;
    int64_t groupIndexFactor_ = 0;
    int64_t tailGroupIndexFactor_ = 0;
    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowFactor_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t tailRowFactorOfTailBlock_ = 0;
    int64_t dLoop_ = 0;
    int64_t dFactor_ = 0;
    int64_t tailDFactor_ = 0;
};

ge::graphStatus SituGluGroupGradArch35Tiling::Init() { return GetPlatformInfo(); }

ge::graphStatus SituGluGroupGradArch35Tiling::GetPlatformInfo()
{
    // Platform parameters are queried directly in Tiling, never passed through
    // TilingParse -> CompileInfo (compile-time values can drift from the runtime environment).
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "platform info is null");
        ReportTilingFailure(context_->GetNodeName(), "platform info is null");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    // Reject illegal platform values before they are used as divisors / buffer sizes.
    if (coreNum_ == 0) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "core num must be greater than 0, but got %lu", coreNum_);
        ReportTilingFailure(context_->GetNodeName(), "core num must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (ubSize_ == 0) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "ubSize must be greater than 0, but got %lu", ubSize_);
        ReportTilingFailure(context_->GetNodeName(), "ubSize must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (ubSize_ <= UB_RESERVE_BYTES) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "ubSize %lu is too small, it must be greater than %lu", ubSize_,
                               UB_RESERVE_BYTES);
        ReportTilingFailure(context_->GetNodeName(), "ubSize is too small");
        return ge::GRAPH_FAILED;
    }
    // Downstream tiling and the kernel buffer pool use only the usable UB (total minus reserve).
    ubSize_ -= UB_RESERVE_BYTES;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGroupGradArch35Tiling::GetShapeAttrsInfo()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto* attrDim = attrs->GetAttrPointer<int64_t>(DIM_ATTR_INDEX);
    cutDim_ = attrDim == nullptr ? -1 : *attrDim;
    auto* attrBeta = attrs->GetAttrPointer<float>(BETA_ATTR_INDEX);
    beta_ = attrBeta == nullptr ? BETA_DEFAULT : *attrBeta;
    auto* attrLinearBeta = attrs->GetAttrPointer<float>(LINEAR_BETA_ATTR_INDEX);
    linearBeta_ = attrLinearBeta == nullptr ? LINEAR_BETA_DEFAULT : *attrLinearBeta;
    auto* attrActivateLeft = attrs->GetAttrPointer<bool>(ACTIVATE_LEFT_ATTR_INDEX);
    activateLeft_ = (attrActivateLeft == nullptr || *attrActivateLeft) ? 1 : 0;
    auto* attrGroupListType = attrs->GetAttrPointer<int64_t>(GROUP_LIST_TYPE_ATTR_INDEX);
    groupListType_ = attrGroupListType == nullptr ? GROUP_LIST_TYPE_CUMSUM : *attrGroupListType;
    // beta is a divisor in the SiTU formula (beta * tanh(gate / beta)); 0 would divide by zero
    // and produce inf/NaN, so reject it explicitly instead of silently degrading the output.
    // NaN/Inf must also be rejected: 1/inf folds to 0 and 0*inf yields NaN downstream.
    OP_CHECK_IF(!std::isfinite(beta_) || beta_ == 0.0f,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "beta", std::to_string(beta_),
                                                      "attr beta must be finite and non-zero"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!std::isfinite(linearBeta_),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "linear_beta",
                                                      std::to_string(linearBeta_), "attr linear_beta must be finite"),
                return ge::GRAPH_FAILED);
    // group_list_type selects how the kernel reduces group_index to the total row count: the last
    // element (cumsum) or the sum of all elements (count). Any other value has no defined meaning.
    OP_CHECK_IF((groupListType_ != GROUP_LIST_TYPE_CUMSUM && groupListType_ != GROUP_LIST_TYPE_COUNT),
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "group_list_type", std::to_string(groupListType_),
                                          "in [0, 1]"),
                return ge::GRAPH_FAILED);

    auto shapeX = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);
    const gert::Shape& xStorageShape = shapeX->GetStorageShape();
    xDims_ = static_cast<int64_t>(xStorageShape.GetDimNum());
    OP_CHECK_IF((xDims_ < 1 || xDims_ > static_cast<int64_t>(MAX_DIM_NUM)),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", std::to_string(xDims_), "[1, 8]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((cutDim_ > (xDims_ - 1) || cutDim_ < -1 * xDims_),
                OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "dim", std::to_string(cutDim_),
                                          "in [-" + std::to_string(xDims_) + ", " + std::to_string(xDims_ - 1) + "]"),
                return ge::GRAPH_FAILED);
    cutDim_ = cutDim_ < 0 ? (cutDim_ + xDims_) : cutDim_;
    cutDimSize_ = xStorageShape.GetDim(cutDim_);
    // Merge x to 2D: bs = product of dims before cutDim, d = product of dims from cutDim on.
    // Guard every multiplication: a wrapped product must be rejected before it can drive tiling.
    bs_ = 1;
    for (int64_t i = 0; i < cutDim_; i++) {
        int64_t dim = xStorageShape.GetDim(i);
        OP_CHECK_IF((dim < 0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(dim),
                                                          "x dims must be non-negative"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(MulOverflow(bs_, dim, bs_),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(dim),
                                                          "product of x dims before dim axis overflows int64"),
                    return ge::GRAPH_FAILED);
    }
    d_ = 1;
    for (int64_t i = cutDim_; i < xDims_; i++) {
        int64_t dim = xStorageShape.GetDim(i);
        OP_CHECK_IF((dim < 0),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(dim),
                                                          "x dims must be non-negative"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(MulOverflow(d_, dim, d_),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(dim),
                                                          "product of x dims from dim axis on overflows int64"),
                    return ge::GRAPH_FAILED);
    }
    // Empty tensors are supported (bs == 0 or d == 0). The total element count must still fit in
    // int64: the kernel derives GM offsets from bs * d, so a wrapped product would corrupt them.
    int64_t totalElementNum = 0;
    OP_CHECK_IF(MulOverflow(bs_, d_, totalElementNum),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x",
                                                      std::to_string(bs_) + "*" + std::to_string(d_),
                                                      "total element count of x overflows int64"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((cutDimSize_ % NUM_TWO != 0),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "x", std::to_string(cutDimSize_),
                                                          "size of x at dim should be divisible by 2"),
                return ge::GRAPH_FAILED);

    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype_ = xDesc->GetDataType();
    OP_CHECK_IF(
        (xDtype_ != ge::DT_FLOAT16 && xDtype_ != ge::DT_BF16 && xDtype_ != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str(),
                                  "FLOAT16, BFLOAT16, FLOAT"),
        return ge::GRAPH_FAILED);
    xElemBytes_ = (xDtype_ == ge::DT_FLOAT) ? B32_BYTES : B16_BYTES;

    if (CheckGradYInfo() == ge::GRAPH_FAILED || CheckWeightInfo() == ge::GRAPH_FAILED ||
        CheckGroupIndexInfo() == ge::GRAPH_FAILED || CheckOutputInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    splitD_ = d_ / NUM_TWO;
    // DataCopyPad src/dstStride is uint32_t. The kernel passes (d - curDFactor) and
    // (splitD - curDFactor) element counts that are scaled by sizeof(T) inside the copy params;
    // reject shapes whose row span in bytes cannot be represented so the stride is never truncated.
    OP_CHECK_IF((d_ > UINT32_MAX_VALUE / xElemBytes_) || (splitD_ > UINT32_MAX_VALUE / xElemBytes_),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "x",
                                                      std::to_string(d_) + ", split=" + std::to_string(splitD_),
                                                      "row span in bytes must fit in uint32 for DataCopyPad stride"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// grad_y must match x on every dim except cutDim, which is halved; dtype must equal x.
ge::graphStatus SituGluGroupGradArch35Tiling::CheckGradYInfo()
{
    auto shapeGradY = context_->GetInputShape(GRADY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeGradY);
    const gert::Shape& gradYStorageShape = shapeGradY->GetStorageShape();
    OP_CHECK_IF((static_cast<int64_t>(gradYStorageShape.GetDimNum()) != xDims_),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad_y",
                                             std::to_string(gradYStorageShape.GetDimNum()), std::to_string(xDims_)),
                return ge::GRAPH_FAILED);
    auto shapeX = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);
    const gert::Shape& xStorageShape = shapeX->GetStorageShape();
    for (int64_t i = 0; i < xDims_; ++i) {
        int64_t expectedDim = (i == cutDim_) ? cutDimSize_ / NUM_TWO : xStorageShape.GetDim(i);
        OP_CHECK_IF(
            (gradYStorageShape.GetDim(i) != expectedDim),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "grad_y", std::to_string(gradYStorageShape.GetDim(i)),
                (i == cutDim_) ? "grad_y at dim should be x at dim divided by 2 (" + std::to_string(expectedDim) + ")" :
                                 "grad_y at non-cut dim should equal x"),
            return ge::GRAPH_FAILED);
    }
    auto gradYDesc = context_->GetInputDesc(GRADY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradYDesc);
    OP_CHECK_IF((gradYDesc->GetDataType() != xDtype_),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grad_y",
                                          ge::TypeUtils::DataTypeToSerialString(gradYDesc->GetDataType()).c_str(),
                                          ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGroupGradArch35Tiling::CheckWeightInfo()
{
    auto weightDesc = context_->GetOptionalInputDesc(WEIGHT_INDEX);
    if (weightDesc != nullptr) {
        auto weightDtype = weightDesc->GetDataType();
        OP_CHECK_IF((weightDtype != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "weight",
                                              ge::TypeUtils::DataTypeToSerialString(weightDtype).c_str(), "FLOAT"),
                    return ge::GRAPH_FAILED);
        auto weightShape = context_->GetOptionalInputShape(WEIGHT_INDEX);
        if (weightShape != nullptr) {
            auto weightStorageShape = weightShape->GetStorageShape();
            auto weightDimNum = weightStorageShape.GetDimNum();
            OP_CHECK_IF(
                (weightDimNum < 1 || weightDimNum > MAX_DIM_NUM),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "weight", std::to_string(weightDimNum), "[1, 8]"),
                return ge::GRAPH_FAILED);
            int64_t weightElementNum = 0;
            OP_CHECK_IF(!ShapeElementNum(weightStorageShape, weightElementNum),
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                            context_->GetNodeName(), "weight", std::to_string(weightElementNum),
                            "weight element count overflows int64 or has a negative dim"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((weightElementNum != bs_),
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                            context_->GetNodeName(), "weight", std::to_string(weightElementNum),
                            "weight element num should equal the product of x dims before the dim axis (" +
                                std::to_string(bs_) + ")"),
                        return ge::GRAPH_FAILED);
            hasWeight_ = true;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGroupGradArch35Tiling::CheckGroupIndexInfo()
{
    auto groupIndexDesc = context_->GetOptionalInputDesc(GROUP_INDEX_INDEX);
    if (groupIndexDesc != nullptr) {
        auto groupIndexDtype = groupIndexDesc->GetDataType();
        OP_CHECK_IF((groupIndexDtype != ge::DT_INT64),
                    OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "group_index",
                                              ge::TypeUtils::DataTypeToSerialString(groupIndexDtype).c_str(), "INT64"),
                    return ge::GRAPH_FAILED);
        auto groupIndexShape = context_->GetOptionalInputShape(GROUP_INDEX_INDEX);
        if (groupIndexShape != nullptr) {
            auto groupIndexStorageShape = groupIndexShape->GetStorageShape();
            OP_CHECK_IF((groupIndexStorageShape.GetDimNum() != 1),
                        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "group_index",
                                                     std::to_string(groupIndexStorageShape.GetDimNum()), "1"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(!ShapeElementNum(groupIndexStorageShape, groupIndexElemNum_),
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "group_index",
                                                                  std::to_string(groupIndexElemNum_),
                                                                  "group_index element count overflows int64"),
                        return ge::GRAPH_FAILED);
            // Empty tensor is not supported: a passed group_index must have a positive element count.
            OP_CHECK_IF((groupIndexElemNum_ <= 0),
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "group_index",
                                                                  std::to_string(groupIndexElemNum_),
                                                                  "group_index must be non-empty"),
                        return ge::GRAPH_FAILED);
            hasGroupIndex_ = true;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGroupGradArch35Tiling::CheckOutputInfo()
{
    // grad_x: same dtype and shape as x.
    auto gradXDesc = context_->GetOutputDesc(GRADX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXDesc);
    OP_CHECK_IF((gradXDesc->GetDataType() != xDtype_),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grad_x",
                                          ge::TypeUtils::DataTypeToSerialString(gradXDesc->GetDataType()).c_str(),
                                          "the same as input x"),
                return ge::GRAPH_FAILED);
    auto gradXShape = context_->GetOutputShape(GRADX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradXShape);
    auto gradXStorageShape = gradXShape->GetStorageShape();
    OP_CHECK_IF((static_cast<int64_t>(gradXStorageShape.GetDimNum()) != xDims_),
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad_x",
                                             std::to_string(gradXStorageShape.GetDimNum()), std::to_string(xDims_)),
                return ge::GRAPH_FAILED);
    auto shapeX = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);
    const gert::Shape& xStorageShape = shapeX->GetStorageShape();
    for (int64_t i = 0; i < xDims_; ++i) {
        OP_CHECK_IF((gradXStorageShape.GetDim(i) != xStorageShape.GetDim(i)),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grad_x",
                                                          std::to_string(gradXStorageShape.GetDim(i)),
                                                          "grad_x should have the same shape as x"),
                    return ge::GRAPH_FAILED);
    }

    // grad_weight: required float32 output. When weight is present its element count must equal bs;
    // otherwise it must be empty (numel == 0).
    auto gradWeightDesc = context_->GetOutputDesc(GRADWEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradWeightDesc);
    OP_CHECK_IF((gradWeightDesc->GetDataType() != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grad_weight",
                                          ge::TypeUtils::DataTypeToSerialString(gradWeightDesc->GetDataType()).c_str(),
                                          "FLOAT"),
                return ge::GRAPH_FAILED);
    auto gradWeightShape = context_->GetOutputShape(GRADWEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradWeightShape);
    const gert::Shape& gradWeightStorage = gradWeightShape->GetStorageShape();
    int64_t gradWeightElementNum = 0;
    OP_CHECK_IF(!ShapeElementNum(gradWeightStorage, gradWeightElementNum),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "grad_weight",
                                                          std::to_string(gradWeightElementNum),
                                                          "grad_weight element count overflows int64 or has a neg dim"),
                return ge::GRAPH_FAILED);
    if (hasWeight_) {
        OP_CHECK_IF((gradWeightElementNum != bs_),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context_->GetNodeName(), "grad_weight", std::to_string(gradWeightElementNum),
                        "grad_weight element num should equal the product of x dims before the dim axis (" +
                            std::to_string(bs_) + ")"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((gradWeightElementNum != 0),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "grad_weight",
                                                              std::to_string(gradWeightElementNum),
                                                              "grad_weight must be empty when weight is not provided"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SituGluGroupGradArch35Tiling::CalcGroupIndexTiling()
{
    // cumsum mode reads only the last group_index element on device, so the chunked UB reduce that
    // groupIndexLoop / groupIndexFactor describe is never entered and is not computed here.
    if (!hasGroupIndex_ || groupListType_ == GROUP_LIST_TYPE_CUMSUM) {
        return ge::GRAPH_SUCCESS;
    }

    const int64_t int64Align = BLOCK_SIZE / static_cast<int64_t>(sizeof(int64_t));
    groupIndexFactor_ = groupIndexElemNum_;
    int64_t groupIndexSize = SaturatingMul(SaturatingMul(RoundUp(groupIndexFactor_, int64Align), DOUBLE_BUFFER),
                                           static_cast<int64_t>(sizeof(int64_t)));
    int64_t groupIndexSumSize = BLOCK_SIZE;
    if (SaturatingAdd(groupIndexSize, groupIndexSumSize) <= static_cast<int64_t>(ubSize_)) {
        groupIndexLoop_ = 1;
        tailGroupIndexFactor_ = groupIndexFactor_;
        return ge::GRAPH_SUCCESS;
    }

    int64_t base = MIN_GROUP_INDEX_SPLIT_FACTOR;
    int64_t maxBase = std::max(groupIndexElemNum_, base);
    while (base <= maxBase) {
        groupIndexFactor_ = CeilDiv(groupIndexElemNum_, base);
        groupIndexSize = SaturatingMul(SaturatingMul(RoundUp(groupIndexFactor_, int64Align), DOUBLE_BUFFER),
                                       static_cast<int64_t>(sizeof(int64_t)));
        if (SaturatingAdd(groupIndexSize, groupIndexSumSize) < static_cast<int64_t>(ubSize_)) {
            break;
        }
        if (base >= maxBase) {
            break;
        }
        base++;
    }
    groupIndexFactor_ = std::max(groupIndexFactor_, static_cast<int64_t>(1));
    if (groupIndexFactor_ > GROUP_INDEX_ALIGN_ELEMS) {
        groupIndexFactor_ = DownAlign(groupIndexFactor_, GROUP_INDEX_ALIGN_ELEMS);
        groupIndexFactor_ = std::max(groupIndexFactor_, static_cast<int64_t>(1));
    }
    groupIndexLoop_ = CeilDiv(groupIndexElemNum_, groupIndexFactor_);
    tailGroupIndexFactor_ = groupIndexElemNum_ % groupIndexFactor_ == 0 ? groupIndexFactor_ :
                                                                          groupIndexElemNum_ % groupIndexFactor_;
    return ge::GRAPH_SUCCESS;
}

void SituGluGroupGradArch35Tiling::InitCoreTiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;
}

void SituGluGroupGradArch35Tiling::SetFullDTiling()
{
    dLoop_ = 1;
    dFactor_ = splitD_;
    tailDFactor_ = dFactor_;
}

void SituGluGroupGradArch35Tiling::SetSplitDTiling()
{
    dLoop_ = CeilDiv(splitD_, dFactor_);
    tailDFactor_ = splitD_ % dFactor_ == 0 ? dFactor_ : splitD_ % dFactor_;
}

void SituGluGroupGradArch35Tiling::SetRowLoopTiling()
{
    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;
}

int64_t SituGluGroupGradArch35Tiling::AddWeightSize(int64_t totalSize, int64_t rowFactor) const
{
    if (!hasWeight_) {
        return totalSize;
    }
    // weight queue (double buffered, fp32) plus the grad_weight per-row accumulator (single, fp32).
    int64_t weightBytes = SaturatingMul(RoundUp(rowFactor, B32_ALIGN_NUM), B32_BYTES);
    weightBytes = SaturatingMul(weightBytes, DOUBLE_BUFFER);
    int64_t accBytes = SaturatingMul(RoundUp(rowFactor, B32_ALIGN_NUM), B32_BYTES);
    return SaturatingAdd(SaturatingAdd(totalSize, weightBytes), accBytes);
}

int64_t SituGluGroupGradArch35Tiling::CalcTotalSize(int64_t rowFactor, int64_t dFactor) const
{
    // grad_y, gate, up, grad_gate and grad_up share the x dtype (2 bytes for fp16/bf16, 4 bytes for
    // float32). The backward is computed by the VF (RegBase) path, so no fp32 UB scratch is needed
    // for the activation itself; only the weight queue and the grad_weight accumulator are fp32.
    // Saturating math: an overflowing byte count is reported as INT64_MAX so the UB-budget
    // comparison rejects the tile instead of accepting a wrapped (too small) size.
    int64_t alignNum = BLOCK_SIZE / xElemBytes_;
    int64_t bufBytes = SaturatingMul(rowFactor, RoundUp(dFactor, alignNum));
    bufBytes = SaturatingMul(bufBytes, xElemBytes_);
    bufBytes = SaturatingMul(bufBytes, DOUBLE_BUFFER);
    int64_t totalSize = SaturatingMul(bufBytes, UB_BUFFER_NUM); // grad_y + gate + up + grad_gate + grad_up
    return AddWeightSize(totalSize, rowFactor);
}

void SituGluGroupGradArch35Tiling::CalcDAndRowFactorTiling(int64_t rowOnceLoop, int64_t dStep)
{
    rowFactor_ = rowOnceLoop;
    if (CalcTotalSize(rowOnceLoop, splitD_) <= static_cast<int64_t>(ubSize_)) {
        SetFullDTiling();
    } else {
        int64_t base = 1;
        int64_t maxBase = CeilDiv(splitD_, dStep);
        dFactor_ = dStep;
        while (base <= maxBase) {
            // Base * dStep stays small because the budget check below breaks the loop well before
            // base can grow large; keep the multiply guarded so an absurd splitD_ cannot wrap.
            int64_t tryDFactor = SaturatingMul(base, dStep);
            if (tryDFactor > splitD_) {
                tryDFactor = splitD_;
            }
            if (CalcTotalSize(rowOnceLoop, tryDFactor) > static_cast<int64_t>(ubSize_)) {
                break;
            }
            dFactor_ = tryDFactor;
            if (tryDFactor == splitD_) {
                break;
            }
            if (base >= maxBase) {
                break;
            }
            base++;
        }
        SetSplitDTiling();
    }

    if (dFactor_ == splitD_) {
        int64_t lo = 1;
        int64_t hi = rowOfFormerBlock_;
        while (lo < hi) {
            // Round up without (hi - lo + 1), which can overflow when hi is near INT64_MAX.
            int64_t delta = hi - lo;
            int64_t mid = lo + CeilDiv(delta, NUM_TWO);
            if (CalcTotalSize(mid, dFactor_) <= static_cast<int64_t>(ubSize_)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        rowFactor_ = lo;
    }

    SetRowLoopTiling();
}

void SituGluGroupGradArch35Tiling::SetEmptyTiling()
{
    // Zero workload: one core is launched whose row loop count is 0, so the kernel returns without
    // touching global memory. rowFactor/dFactor stay positive to keep kernel queue buffers valid.
    // Count-mode group-index tiling (groupIndexFactor/groupIndexLoop) is still computed by CalcGroupIndexTiling so
    // the kernel derives realBs = min(groupRows, bs) = 0 and exits from the group path as well, where groupRows is
    // group_index[-1] in cumsum mode and sum(group_index) in count mode. The kernel also returns directly when
    // splitD is 0, which covers d == 0 with a non-empty group_index (realBs > 0 recomputes non-zero row loops in
    // that path); grad_weight is an empty sum there, so the kernel zero-fills its valid rows before returning.
    rowOfFormerBlock_ = 0;
    rowOfTailBlock_ = 0;
    rowLoopOfFormerBlock_ = 0;
    rowLoopOfTailBlock_ = 0;
    tailRowFactorOfFormerBlock_ = 0;
    tailRowFactorOfTailBlock_ = 0;
    rowFactor_ = 1;
    dLoop_ = 1;
    // The kernel still allocates the grad_y/gate/up/grad_gate/grad_up queue buffers from
    // rowFactor*dFactor during Init even though Process returns immediately for the zero-workload
    // case, and it also allocates the weight queue / grad_weight accumulator when weight is present.
    // Cap dFactor by the UB budget left after those buffers so a huge splitD can never overrun the
    // buffer pool. AddWeightSize is used here too, keeping this path on the same budget accounting.
    int64_t budget = static_cast<int64_t>(ubSize_);
    const int64_t weightBytes = AddWeightSize(0, rowFactor_);
    budget = (weightBytes < budget) ? (budget - weightBytes) : 0;
    const int64_t dPerRowBytes = DOUBLE_BUFFER * UB_BUFFER_NUM * xElemBytes_;
    int64_t maxDFactor = budget / dPerRowBytes;
    maxDFactor -= BLOCK_SIZE / xElemBytes_; // headroom for the per-row RoundUp alignment
    maxDFactor = std::max(maxDFactor, static_cast<int64_t>(1));
    dFactor_ = std::min(std::max(splitD_, static_cast<int64_t>(1)), maxDFactor);
    tailDFactor_ = dFactor_;
    usedCoreNums_ = 1;
}

ge::graphStatus SituGluGroupGradArch35Tiling::CalcOpTiling()
{
    if (CalcGroupIndexTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (bs_ == 0 || d_ == 0) {
        SetEmptyTiling();
        return ge::GRAPH_SUCCESS;
    }
    InitCoreTiling();
    // Start from one row per tile; CalcDAndRowFactorTiling grows rowFactor afterwards if it fits.
    int64_t rowOnceLoop = 1;
    int64_t dChunk = ASCEND950_CACHE_LINE_BYTES / xElemBytes_; // elements per cacheline
    CalcDAndRowFactorTiling(rowOnceLoop, dChunk);
    return ge::GRAPH_SUCCESS;
}

void SituGluGroupGradArch35Tiling::FillTilingData()
{
    tilingData_.set_bs(bs_);
    tilingData_.set_d(d_);
    tilingData_.set_splitD(splitD_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_beta(beta_);
    tilingData_.set_linearBeta(linearBeta_);
    tilingData_.set_activateLeft(activateLeft_);
    tilingData_.set_groupIndexElemNum(groupIndexElemNum_);
    tilingData_.set_ubSize(ubSize_);
    tilingData_.set_groupIndexLoop(groupIndexLoop_);
    tilingData_.set_groupIndexFactor(groupIndexFactor_);
    tilingData_.set_tailGroupIndexFactor(tailGroupIndexFactor_);
    tilingData_.set_coreNum(coreNum_);
    tilingData_.set_hasWeight(hasWeight_ ? 1 : 0);
    tilingData_.set_hasGroupIndex(hasGroupIndex_ ? 1 : 0);
    tilingData_.set_groupListType(groupListType_);
    tilingData_.set_hasGradW(hasWeight_ ? 1 : 0);
}

void SituGluGroupGradArch35Tiling::PrintTilingInfo()
{
    OP_LOGD(context_->GetNodeName(),
            "SituGluGroupGradArch35Tiling shape: bs:%ld, d:%ld, splitD:%ld, groupIndexElemNum:%ld, hasGroupIndex:%d, "
            "hasWeight:%d, hasGradW:%d.",
            bs_, d_, splitD_, groupIndexElemNum_, static_cast<int32_t>(hasGroupIndex_),
            static_cast<int32_t>(hasWeight_), static_cast<int32_t>(hasWeight_));
    OP_LOGD(context_->GetNodeName(),
            "SituGluGroupGradArch35Tiling rowTiling: rowOfFormerBlock:%ld, rowOfTailBlock:%ld, "
            "rowLoopOfFormerBlock:%ld, rowLoopOfTailBlock:%ld, rowFactor:%ld, tailRowFactorOfFormerBlock:%ld, "
            "tailRowFactorOfTailBlock:%ld.",
            rowOfFormerBlock_, rowOfTailBlock_, rowLoopOfFormerBlock_, rowLoopOfTailBlock_, rowFactor_,
            tailRowFactorOfFormerBlock_, tailRowFactorOfTailBlock_);
    OP_LOGD(context_->GetNodeName(), "SituGluGroupGradArch35Tiling dTiling: dLoop:%ld, dFactor:%ld, tailDFactor:%ld.",
            dLoop_, dFactor_, tailDFactor_);
    OP_LOGD(context_->GetNodeName(),
            "SituGluGroupGradArch35Tiling groupIndexTiling: groupIndexLoop:%ld, groupIndexFactor:%ld, "
            "tailGroupIndexFactor:%ld, coreNum:%lu, ubSize:%luB.",
            groupIndexLoop_, groupIndexFactor_, tailGroupIndexFactor_, coreNum_, ubSize_);
    OP_LOGD(context_->GetNodeName(),
            "SituGluGroupGradArch35Tiling attrs: beta:%f, linearBeta:%f, activateLeft:%ld, groupListType:%ld.", beta_,
            linearBeta_, activateLeft_, groupListType_);
}

ge::graphStatus SituGluGroupGradArch35Tiling::DoTiling()
{
    if (GetShapeAttrsInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    FillTilingData();
    PrintTilingInfo();

    // The mode axis is a compile-time template argument (no hardcoded TilingKey branch in the
    // kernel); the data type is not part of the key, the build registers DTYPE_X per binary.
    context_->SetTilingKey(GET_TPL_TILING_KEY(SITU_GLU_GROUP_GRAD_TPL_SCH_MODE_0));
    if (hasGroupIndex_) {
        context_->SetBlockDim(coreNum_);
    } else {
        context_->SetBlockDim(usedCoreNums_);
    }
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    auto* rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    tilingData_.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData_.GetDataSize());

    OP_LOGD(context_->GetNodeName(), "SituGluGroupGradArch35Tiling done: blockDim:%lu, workspaceSize:%ldB.",
            hasGroupIndex_ ? coreNum_ : usedCoreNums_, workspaceSize_);
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus Tiling4SituGluGroupGradArch35(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE_WITHOUT_REPORT("SituGluGroupGrad", "Tiling context is null");
        ReportTilingFailure("SituGluGroupGrad", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    SituGluGroupGradArch35Tiling tiling(context);
    if (tiling.Init() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForSituGluGroupGrad(gert::TilingParseContext* context)
{
    // No compile-info carries platform parameters; Tiling queries them directly.
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SituGluGroupGrad)
    .Tiling(Tiling4SituGluGroupGradArch35)
    .TilingParse<SituGluGroupGradCompileInfo>(TilingPrepareForSituGluGroupGrad);
} // namespace optiling
