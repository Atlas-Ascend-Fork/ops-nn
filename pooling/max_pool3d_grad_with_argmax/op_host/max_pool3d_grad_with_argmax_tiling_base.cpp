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
 * \file max_pool3d_grad_with_argmax_tiling_base.cpp
 * \brief
 */

#include "log/log.h"
#include "max_pool3d_grad_with_argmax_tiling.h"

namespace optiling {

constexpr uint64_t D_SHAPE_INDEX = 2;
constexpr uint64_t H_SHAPE_INDEX = 3;
constexpr uint64_t W_SHAPE_INDEX = 4;
constexpr uint64_t D_ATTR_INDEX = 0;
constexpr uint64_t H_ATTR_INDEX = 1;
constexpr uint64_t W_ATTR_INDEX = 2;

bool MaxPool3DGradWithArgmaxTilingBase::CheckInputShape()
{
    const gert::StorageShape* xShape = context_->GetInputShape(X_INDEX);
    const gert::StorageShape* gradShape = context_->GetInputShape(GRAD_INDEX);
    const gert::StorageShape* argmaxShape = context_->GetInputShape(ARGMAX_INDEX);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gradDimNum = gradShape->GetStorageShape().GetDimNum();
    size_t argmaxDimNum = argmaxShape->GetStorageShape().GetDimNum();

    // xDimNum should be 5(format:NCDHW)
    OP_CHECK_IF(
        (xDimNum != NCDHW_DIM_NUM) || (gradDimNum != NCDHW_DIM_NUM) || (argmaxDimNum != NCDHW_DIM_NUM),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "x, grad, argmax",
            (std::to_string(xDimNum) + ", " + std::to_string(gradDimNum) + ", " + std::to_string(argmaxDimNum)).c_str(),
            "all input ranks must be 5 (NCDHW)"),
        return false);
    for (uint32_t i = 0; i < xDimNum; i++) {
        OP_CHECK_IF(xShape->GetStorageShape().GetDim(i) == 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x",
                                                          Ops::Base::ToString(xShape->GetStorageShape()).c_str(),
                                                          "input x must not contain zero dimensions"),
                    return false);
    }

    // gradShape&argmaxShape's shape should be equal
    for (size_t i = 0; i < xDimNum; i++) {
        uint64_t gradDimValue = gradShape->GetStorageShape().GetDim(i);
        uint64_t argmaxDimValue = argmaxShape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(gradDimValue != argmaxDimValue,
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        context_->GetNodeName(), "grad, argmax",
                        (std::to_string(gradDimValue) + ", " + std::to_string(argmaxDimValue)).c_str(),
                        (std::string("dimensions must match at axis ") + std::to_string(i)).c_str()),
                    return false);
    }

    // Input NCDim should be equal
    for (size_t i = 0; i < NC_DIM_NUM; i++) {
        uint64_t xDimValue = xShape->GetStorageShape().GetDim(i);
        uint64_t gradDimValue = gradShape->GetStorageShape().GetDim(i);
        OP_CHECK_IF((gradDimValue != xDimValue),
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        context_->GetNodeName(), "grad, x",
                        (std::to_string(gradDimValue) + ", " + std::to_string(xDimValue)).c_str(),
                        (std::string("dimensions must match at axis ") + std::to_string(i)).c_str()),
                    return false);
    }

    return true;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::CheckInputDtype()
{
    OP_CHECK_IF(
        context_->GetInputDesc(X_INDEX) == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "x",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetInputDesc(GRAD_INDEX) == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "grad",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetInputDesc(ARGMAX_INDEX) == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "argmax",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOutputDesc(Y_INDEX) == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{Ops::Base::GetOpInfo(context_).c_str(),
                                              "context_->GetOutputDesc(Y_INDEX) is null; Tiling cannot continue"})),
                return ge::GRAPH_FAILED);
    auto xDataType = context_->GetInputDesc(X_INDEX)->GetDataType();
    auto gradDataType = context_->GetInputDesc(GRAD_INDEX)->GetDataType();
    auto argmaxDataType = context_->GetInputDesc(ARGMAX_INDEX)->GetDataType();
    auto yOutDataType = context_->GetOutputDesc(Y_INDEX)->GetDataType();

    OP_CHECK_IF(xDataType != gradDataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "x, grad",
                    (Ops::Base::ToString(xDataType) + ", " + Ops::Base::ToString(gradDataType)).c_str(),
                    "x and grad must have the same dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDataType != yOutDataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "x, y",
                    (Ops::Base::ToString(xDataType) + ", " + Ops::Base::ToString(yOutDataType)).c_str(),
                    "x and y must have the same dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (xDataType != ge::DT_FLOAT) && (xDataType != ge::DT_FLOAT16) && (xDataType != ge::DT_BF16),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xDataType).c_str(),
                                              "supported dtypes are float32, float16 and bfloat16"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(argmaxDataType != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "argmax",
                                          Ops::Base::ToString(argmaxDataType).c_str(), "int32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::CheckAttrShape()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "attributes",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    int32_t kSizeDimNum = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetSize();
    int32_t stridesDimNum = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetSize();
    int32_t padsDimNum = attrs->GetListInt(PADS_ATTR_INDEX)->GetSize();
    int32_t dilationsDimNum = attrs->GetListInt(DILATION_ATTR_INDEX)->GetSize();

    // Check attr dim num
    OP_CHECK_IF(
        (kSizeDimNum != DHW_DIM_NUM) && (kSizeDimNum != 1),
        OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "ksize", std::to_string(kSizeDimNum).c_str(), "1 or 3"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((stridesDimNum != DHW_DIM_NUM) && (stridesDimNum != 1) && (stridesDimNum != 0),
                OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "strides", std::to_string(stridesDimNum).c_str(),
                                             "0, 1 or 3"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (padsDimNum != DHW_DIM_NUM) && (padsDimNum != 1),
        OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "pads", std::to_string(padsDimNum).c_str(), "1 or 3"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((dilationsDimNum != DHW_DIM_NUM) && (dilationsDimNum != 1),
                OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "dilation",
                                             std::to_string(dilationsDimNum).c_str(), "1 or 3"),
                return ge::GRAPH_FAILED);

    // Check attr value bigger than 0
    auto kSizeVector = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetData();
    auto stridesVector = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetData();
    auto padsVector = attrs->GetListInt(PADS_ATTR_INDEX)->GetData();
    auto dilationsVector = attrs->GetListInt(DILATION_ATTR_INDEX)->GetData();
    for (uint32_t i = 0; i < (uint32_t)kSizeDimNum; i++) {
        OP_CHECK_IF((kSizeVector[i] <= 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ksize",
                                                          std::to_string(kSizeVector[i]).c_str(),
                                                          "every element must be greater than 0"),
                    return ge::GRAPH_FAILED);
    }
    for (uint32_t i = 0; i < (uint32_t)stridesDimNum; i++) {
        OP_CHECK_IF((stridesVector[i] <= 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "strides",
                                                          std::to_string(stridesVector[i]).c_str(),
                                                          "every element must be greater than 0"),
                    return ge::GRAPH_FAILED);
    }
    for (uint32_t i = 0; i < (uint32_t)padsDimNum; i++) {
        OP_CHECK_IF((padsVector[i] < 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "pads",
                                                          std::to_string(padsVector[i]).c_str(),
                                                          "every element must be nonnegative"),
                    return ge::GRAPH_FAILED);
    }
    for (uint32_t i = 0; i < (uint32_t)dilationsDimNum; i++) {
        OP_CHECK_IF((dilationsVector[i] <= 0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dilation",
                                                          std::to_string(dilationsVector[i]).c_str(),
                                                          "every element must be greater than 0"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::SetInputParams()
{
    const gert::Shape xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    const gert::Shape gradShape = context_->GetInputShape(GRAD_INDEX)->GetStorageShape();
    uint64_t n = xShape.GetDim(0);
    uint64_t c = xShape.GetDim(1);
    maxPoolGradParams.ncDim = n * c;
    maxPoolGradParams.diDim = xShape.GetDim(D_SHAPE_INDEX);
    maxPoolGradParams.hiDim = xShape.GetDim(H_SHAPE_INDEX);
    maxPoolGradParams.wiDim = xShape.GetDim(W_SHAPE_INDEX);
    maxPoolGradParams.doDim = gradShape.GetDim(D_SHAPE_INDEX);
    maxPoolGradParams.hoDim = gradShape.GetDim(H_SHAPE_INDEX);
    maxPoolGradParams.woDim = gradShape.GetDim(W_SHAPE_INDEX);
    maxPoolGradParams.vl = NUM_PER_REP_B32;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::SetAttrParams()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "attributes",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    int32_t kSizeDimNum = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetSize();
    int32_t stridesDimNum = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetSize();
    int32_t padsDimNum = attrs->GetListInt(PADS_ATTR_INDEX)->GetSize();
    int32_t dilationsDimNum = attrs->GetListInt(DILATION_ATTR_INDEX)->GetSize();
    auto kSizeVector = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetData();
    auto stridesVector = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetData();
    auto padsVector = attrs->GetListInt(PADS_ATTR_INDEX)->GetData();
    auto dilationsVector = attrs->GetListInt(DILATION_ATTR_INDEX)->GetData();
    bool ceilMode = *attrs->GetBool(CEIL_MODE_ATTR_INDEX);
    maxPoolGradParams.ceilMode = ceilMode;
    maxPoolGradParams.kd = kSizeVector[D_ATTR_INDEX];
    maxPoolGradParams.kh = (kSizeDimNum == 1) ? maxPoolGradParams.kd : kSizeVector[H_ATTR_INDEX];
    maxPoolGradParams.kw = (kSizeDimNum == 1) ? maxPoolGradParams.kd : kSizeVector[W_ATTR_INDEX];
    if (stridesDimNum == 0) {
        maxPoolGradParams.sd = maxPoolGradParams.kd;
        maxPoolGradParams.sh = maxPoolGradParams.kd;
        maxPoolGradParams.sw = maxPoolGradParams.kd;
    } else {
        maxPoolGradParams.sd = stridesVector[D_ATTR_INDEX];
        maxPoolGradParams.sh = (stridesDimNum == 1) ? maxPoolGradParams.sd : stridesVector[H_ATTR_INDEX];
        maxPoolGradParams.sw = (stridesDimNum == 1) ? maxPoolGradParams.sd : stridesVector[W_ATTR_INDEX];
    }
    maxPoolGradParams.pDTop = padsVector[D_ATTR_INDEX];
    maxPoolGradParams.pHTop = (padsDimNum == 1) ? maxPoolGradParams.pDTop : padsVector[H_ATTR_INDEX];
    maxPoolGradParams.pWTop = (padsDimNum == 1) ? maxPoolGradParams.pDTop : padsVector[W_ATTR_INDEX];
    maxPoolGradParams.dilationD = dilationsVector[D_ATTR_INDEX];
    maxPoolGradParams.dilationH = (dilationsDimNum == 1) ? maxPoolGradParams.dilationD : dilationsVector[H_ATTR_INDEX];
    maxPoolGradParams.dilationW = (dilationsDimNum == 1) ? maxPoolGradParams.dilationD : dilationsVector[W_ATTR_INDEX];
    bool isDOverLap = maxPoolGradParams.kd > maxPoolGradParams.sd ? true : false;
    bool isHOverLap = maxPoolGradParams.kh > maxPoolGradParams.sh ? true : false;
    bool isWOverLap = maxPoolGradParams.kw > maxPoolGradParams.sw ? true : false;
    maxPoolGradParams.isOverLap = (isDOverLap || isHOverLap || isWOverLap);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::CheckInputValid()
{
    const uint64_t kd = maxPoolGradParams.kd;
    const uint64_t kh = maxPoolGradParams.kh;
    const uint64_t kw = maxPoolGradParams.kw;
    const uint64_t sd = maxPoolGradParams.sd;
    const uint64_t sh = maxPoolGradParams.sh;
    const uint64_t sw = maxPoolGradParams.sw;
    const uint64_t pDTop = maxPoolGradParams.pDTop;
    const uint64_t pHTop = maxPoolGradParams.pHTop;
    const uint64_t pWTop = maxPoolGradParams.pWTop;
    const uint64_t dilationD = maxPoolGradParams.dilationD;
    const uint64_t dilationH = maxPoolGradParams.dilationH;
    const uint64_t dilationW = maxPoolGradParams.dilationW;

    // check 1
    OP_CHECK_IF((pDTop > (kd / 2)) || (pHTop > (kh / 2)) || (pWTop > (kw / 2)),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "pads, ksize, dilation",
                    (std::to_string(pDTop) + ", " + std::to_string(pHTop) + ", " + std::to_string(pWTop)).c_str(),
                    "pads must not exceed half the kernel size"),
                return ge::GRAPH_FAILED);
    // check 2
    OP_CHECK_IF((pDTop > ((kd - 1) * dilationD + 1) / 2) || (pHTop > ((kh - 1) * dilationH + 1) / 2) ||
                    (pWTop > ((kw - 1) * dilationW + 1) / 2),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "pads, ksize, dilation",
                    (std::to_string(pDTop) + ", " + std::to_string(pHTop) + ", " + std::to_string(pWTop)).c_str(),
                    "pads must not exceed half the effective kernel size"),
                return ge::GRAPH_FAILED);
    // check 3
    // Check outerDim invaild
    int64_t doExpected, hoExpected, woExpected;
    if (maxPoolGradParams.ceilMode) {
        doExpected = Ops::Base::CeilDiv((maxPoolGradParams.diDim + NUM_TWO * pDTop + sd - dilationD * (kd - 1) - 1),
                                        sd);
        hoExpected = Ops::Base::CeilDiv((maxPoolGradParams.hiDim + NUM_TWO * pHTop + sh - dilationH * (kh - 1) - 1),
                                        sh);
        woExpected = Ops::Base::CeilDiv((maxPoolGradParams.wiDim + NUM_TWO * pWTop + sw - dilationW * (kw - 1) - 1),
                                        sw);
    } else {
        doExpected = (maxPoolGradParams.diDim + NUM_TWO * pDTop + sd - dilationD * (kd - 1) - 1) / sd;
        hoExpected = (maxPoolGradParams.hiDim + NUM_TWO * pHTop + sh - dilationH * (kh - 1) - 1) / sh;
        woExpected = (maxPoolGradParams.wiDim + NUM_TWO * pWTop + sw - dilationW * (kw - 1) - 1) / sw;
    }
    doExpected = ((doExpected - 1) * sd >= maxPoolGradParams.diDim + pDTop) ? doExpected - 1 : doExpected;
    hoExpected = ((hoExpected - 1) * sh >= maxPoolGradParams.hiDim + pHTop) ? hoExpected - 1 : hoExpected;
    woExpected = ((woExpected - 1) * sw >= maxPoolGradParams.wiDim + pWTop) ? woExpected - 1 : woExpected;
    OP_CHECK_IF((doExpected <= 0) || (static_cast<uint64_t>(doExpected) != maxPoolGradParams.doDim) ||
                    (hoExpected <= 0) || (static_cast<uint64_t>(hoExpected) != maxPoolGradParams.hoDim) ||
                    (woExpected <= 0) || (static_cast<uint64_t>(woExpected) != maxPoolGradParams.woDim),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "grad",
                    Ops::Base::ToString(context_->GetInputShape(GRAD_INDEX)->GetStorageShape()).c_str(),
                    (std::string("expected D/H/W: ") + std::to_string(doExpected) + ", " + std::to_string(hoExpected) +
                     ", " + std::to_string(woExpected))
                        .c_str()),
                return ge::GRAPH_FAILED);
    // check 4
    // Check index range
    OP_CHECK_IF(maxPoolGradParams.diDim * maxPoolGradParams.hiDim * maxPoolGradParams.wiDim > MAX_INT32,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context_->GetNodeName(), "x",
                    std::to_string(maxPoolGradParams.diDim * maxPoolGradParams.hiDim * maxPoolGradParams.wiDim).c_str(),
                    "D * H * W must not exceed INT32_MAX"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MaxPool3DGradWithArgmaxTilingBase::SetOtherInputParams()
{
    auto xDataType = context_->GetInputDesc(X_INDEX)->GetDataType();
    uint64_t xDtypeSize = GetSizeByDataType(xDataType);
    maxPoolGradParams.xDtypeSize = xDtypeSize;
    maxPoolGradParams.indexDtypeSize = GetSizeByDataType(ge::DT_INT32);

    uint64_t dLenCalFromDo = (maxPoolGradParams.doDim - 1) * maxPoolGradParams.sd + maxPoolGradParams.kd;
    uint64_t dLenCalFromDi = maxPoolGradParams.diDim + maxPoolGradParams.pDTop;
    maxPoolGradParams.pDBottom = dLenCalFromDo < dLenCalFromDi ? 0 : dLenCalFromDo - dLenCalFromDi;
    uint64_t hLenCalFromHo = (maxPoolGradParams.hoDim - 1) * maxPoolGradParams.sh + maxPoolGradParams.kh;
    uint64_t hLenCalFromHi = maxPoolGradParams.hiDim + maxPoolGradParams.pHTop;
    maxPoolGradParams.pHBottom = hLenCalFromHo < hLenCalFromHi ? 0 : hLenCalFromHo - hLenCalFromHi;
    uint64_t wLenCalFromWo = (maxPoolGradParams.woDim - 1) * maxPoolGradParams.sw + maxPoolGradParams.kw;
    uint64_t wLenCalFromWi = maxPoolGradParams.wiDim + maxPoolGradParams.pWTop;
    maxPoolGradParams.pWBottom = wLenCalFromWo < wLenCalFromWi ? 0 : wLenCalFromWo - wLenCalFromWi;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "Enter MaxPool3DGradWithArgmaxTilingBase GetShapeAttrsInfo.");
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckInputDtype(),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "The input dtype is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputShape(),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "The input relationship is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckAttrShape(),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "The attr shape is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != SetInputParams(),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Set input shape failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != SetAttrParams(),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Set attr shape failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckInputValid(),
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "The input shape is invalid."),
                return ge::GRAPH_FAILED);
    SetOtherInputParams();
    return ge::GRAPH_SUCCESS;
}

bool MaxPool3DGradWithArgmaxTilingBase::IsCapable() { return false; }

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const Tiling4MaxPool3DGradWithArgmaxCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfo == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{context_->GetNodeName(), "compile info is null"})),
                return ge::GRAPH_FAILED);
    maxPoolGradParams.totalCoreNum = compileInfo->totalCoreNum;
    maxPoolGradParams.maxUbSize = compileInfo->maxUbSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::GetWorkspaceSize()
{
    context_->SetBlockDim(maxPoolGradParams.usedCoreNum);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t usrWorkspaceSize = maxPoolGradParams.workspaceSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize + sysWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBase::PostTiling() { return ge::GRAPH_SUCCESS; }

void MaxPool3DGradWithArgmaxTilingBase::SetCntTailTilingParams()
{
    maxPoolGradParams.ncCnt = Ops::Base::CeilDiv(maxPoolGradParams.ncDim, maxPoolGradParams.baseNc);
    maxPoolGradParams.doCnt = Ops::Base::CeilDiv(maxPoolGradParams.doDim, maxPoolGradParams.baseDo);
    maxPoolGradParams.hoCnt = Ops::Base::CeilDiv(maxPoolGradParams.hoDim, maxPoolGradParams.baseHo);
    maxPoolGradParams.woCnt = Ops::Base::CeilDiv(maxPoolGradParams.woDim, maxPoolGradParams.baseWo);
    maxPoolGradParams.ncTail = maxPoolGradParams.ncDim - (maxPoolGradParams.ncCnt - 1) * maxPoolGradParams.baseNc;
    maxPoolGradParams.doTail = maxPoolGradParams.doDim - (maxPoolGradParams.doCnt - 1) * maxPoolGradParams.baseDo;
    maxPoolGradParams.hoTail = maxPoolGradParams.hoDim - (maxPoolGradParams.hoCnt - 1) * maxPoolGradParams.baseHo;
    maxPoolGradParams.woTail = maxPoolGradParams.woDim - (maxPoolGradParams.woCnt - 1) * maxPoolGradParams.baseWo;
    maxPoolGradParams.totalCnt = maxPoolGradParams.ncCnt * maxPoolGradParams.doCnt * maxPoolGradParams.hoCnt *
                                 maxPoolGradParams.woCnt;
}

void MaxPool3DGradWithArgmaxTilingBase::SetBaseTilingData()
{
    tilingData.set_ncDim(maxPoolGradParams.ncDim);
    tilingData.set_diDim(maxPoolGradParams.diDim);
    tilingData.set_hiDim(maxPoolGradParams.hiDim);
    tilingData.set_wiDim(maxPoolGradParams.wiDim);
    tilingData.set_doDim(maxPoolGradParams.doDim);
    tilingData.set_hoDim(maxPoolGradParams.hoDim);
    tilingData.set_woDim(maxPoolGradParams.woDim);
    tilingData.set_kd(maxPoolGradParams.kd);
    tilingData.set_kh(maxPoolGradParams.kh);
    tilingData.set_kw(maxPoolGradParams.kw);
    tilingData.set_sd(maxPoolGradParams.sd);
    tilingData.set_sh(maxPoolGradParams.sh);
    tilingData.set_sw(maxPoolGradParams.sw);
    tilingData.set_padDTop(maxPoolGradParams.pDTop);
    tilingData.set_padDBottom(maxPoolGradParams.pDBottom);
    tilingData.set_padHTop(maxPoolGradParams.pHTop);
    tilingData.set_padHBottom(maxPoolGradParams.pHBottom);
    tilingData.set_padWTop(maxPoolGradParams.pWTop);
    tilingData.set_padWBottom(maxPoolGradParams.pWBottom);
    tilingData.set_baseNc(maxPoolGradParams.baseNc);
    tilingData.set_baseDo(maxPoolGradParams.baseDo);
    tilingData.set_baseHo(maxPoolGradParams.baseHo);
    tilingData.set_baseWo(maxPoolGradParams.baseWo);
    tilingData.set_ncTail(maxPoolGradParams.ncTail);
    tilingData.set_doTail(maxPoolGradParams.doTail);
    tilingData.set_hoTail(maxPoolGradParams.hoTail);
    tilingData.set_woTail(maxPoolGradParams.woTail);
    tilingData.set_ncCnt(maxPoolGradParams.ncCnt);
    tilingData.set_doCnt(maxPoolGradParams.doCnt);
    tilingData.set_hoCnt(maxPoolGradParams.hoCnt);
    tilingData.set_woCnt(maxPoolGradParams.woCnt);
    tilingData.set_totalCnt(maxPoolGradParams.totalCnt);
    tilingData.set_usedCoreNum(maxPoolGradParams.usedCoreNum);
    tilingData.set_totalUBSize(maxPoolGradParams.maxUbSize);
}

void MaxPool3DGradWithArgmaxTilingBase::PrintTilingData()
{
    maxPoolGradParams.isInitOutput = (maxPoolGradParams.doDim * maxPoolGradParams.kd <
                                      maxPoolGradParams.diDim + maxPoolGradParams.pDTop + maxPoolGradParams.pDBottom) ||
                                     (maxPoolGradParams.hoDim * maxPoolGradParams.kh <
                                      maxPoolGradParams.hiDim + maxPoolGradParams.pHTop + maxPoolGradParams.pHBottom) ||
                                     (maxPoolGradParams.woDim * maxPoolGradParams.kw <
                                      maxPoolGradParams.wiDim + maxPoolGradParams.pWTop + maxPoolGradParams.pWBottom) ||
                                     (maxPoolGradParams.doDim - 1) * maxPoolGradParams.sd + maxPoolGradParams.kd <
                                         maxPoolGradParams.diDim + maxPoolGradParams.pDTop ||
                                     (maxPoolGradParams.hoDim - 1) * maxPoolGradParams.sh + maxPoolGradParams.kh <
                                         maxPoolGradParams.hiDim + maxPoolGradParams.pHTop ||
                                     (maxPoolGradParams.woDim - 1) * maxPoolGradParams.sw + maxPoolGradParams.kw <
                                         maxPoolGradParams.wiDim + maxPoolGradParams.pWTop ||
                                     maxPoolGradParams.isOverLap;
    OP_LOGI(context_->GetNodeName(),
            "TilingData nc: %lu, di: %lu, hi: %lu, wi: %lu do: %lu, ho: %lu, wo: %lu, "
            "kd: %lu, kh: %lu, kw: %lu sd: %lu, sh: %lu, sw: %lu, "
            "pDTop: %lu, pHTop: %lu, pWTop: %lu pDBottom: %lu, pHBottom: %lu, pWBottom: %lu, "
            "baseNc: %lu, baseDo: %lu, baseHo: %lu, baseWo: %lu ncTail: %lu, doTail: %lu, hoTail: %lu, woTail: %lu, "
            "ncCnt: %lu, doCnt: %lu, hoCnt: %lu, woCnt: %lu totalCnt: %lu, usedCoreNum: %lu, totalUBSize: %lu, "
            "ceilMode: %d, isOverLap: %d.",
            tilingData.get_ncDim(), tilingData.get_diDim(), tilingData.get_hiDim(), tilingData.get_wiDim(),
            tilingData.get_doDim(), tilingData.get_hoDim(), tilingData.get_woDim(), tilingData.get_kd(),
            tilingData.get_kh(), tilingData.get_kw(), tilingData.get_sd(), tilingData.get_sh(), tilingData.get_sw(),
            tilingData.get_padDTop(), tilingData.get_padHTop(), tilingData.get_padWTop(), tilingData.get_padDBottom(),
            tilingData.get_padHBottom(), tilingData.get_padWBottom(), tilingData.get_baseNc(), tilingData.get_baseDo(),
            tilingData.get_baseHo(), tilingData.get_baseWo(), tilingData.get_ncTail(), tilingData.get_doTail(),
            tilingData.get_hoTail(), tilingData.get_woTail(), tilingData.get_ncCnt(), tilingData.get_doCnt(),
            tilingData.get_hoCnt(), tilingData.get_woCnt(), tilingData.get_totalCnt(), tilingData.get_usedCoreNum(),
            tilingData.get_totalUBSize(), maxPoolGradParams.ceilMode, maxPoolGradParams.isOverLap);
}

uint64_t MaxPool3DGradWithArgmaxTilingBase::GetTilingKey() const
{
    uint64_t tilingKey = maxPoolGradParams.tilingType;
    if (maxPoolGradParams.isOverLap) {
        tilingKey += TILING_OVERLAP;
    }
    if (TILING_TYPE_CUTK == maxPoolGradParams.tilingType) {
        tilingKey += maxPoolGradParams.ubCutAxis;
    }
    OP_LOGI(context_->GetNodeName(), "TilingKey is %lu.", tilingKey);
    return tilingKey;
}
} // namespace optiling
