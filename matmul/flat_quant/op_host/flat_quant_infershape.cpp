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
 * \file flat_quant_infershape.cpp
 * \brief InferShape and InferDataType registration for the FlatQuant operator.
 */
#include "common/inc/op_host/op_const_def_nn.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"
#include "op_common/op_host/util/op_const_def.h"
#include "acl/acl_platform.h"

using namespace ge;
namespace ops {
static const int32_t DTYPE_FLOAT4_E2M1 = 40;
static const int32_t DTYPE_HIFLOAT4 = 42;
static constexpr size_t FLATQUANT_K_IDX = 0;
static constexpr size_t FLATQUANT_M_IDX = 1;
static constexpr size_t FLATQUANT_N_IDX = 2;
static constexpr size_t FLATQUANT_ATTRS_NUM_TWO = 2;
static constexpr size_t FLATQUANT_ATTRS_NUM_THREE = 3;
static constexpr size_t FLATQUANT_MX_N_OUT_DIM = 2;
static constexpr size_t FLATQUANT_HIF4_N_OUT_DIM = 1;
static constexpr size_t FLATQUANT_N_IS_EVEN = 2;
static constexpr size_t FLATQUANT_X_DIM_NUM = 3;
static constexpr size_t FLATQUANT_DOUBLE_CEIL_SIZE = 64;
static const size_t ATTR_INDEX_OF_DST_DTYPE = 1;
static const size_t ATTR_INDEX_OF_DST_TYPE_MAX = 2;
constexpr float ZERO_FLOAT = 0.0f;
constexpr float SIX_FLOAT = 6.0f;
constexpr float TWELVE_FLOAT = 12.0f;
constexpr int64_t UNKNOWN_DIM = -1;

static bool IsFloatEqual(float a, float b) { return std::abs(a - b) <= std::numeric_limits<float>::epsilon(); }

static bool IsFlatQuantMxFp4DavidSupport()
{
    char buf[64] = {0};
    aclError ret = aclplatformGetDeviceInfo(ACL_PLATFORM_NPU_ARCH, buf, sizeof(buf));
    if (ret != ACL_SUCCESS) {
        OP_LOGE("IsFlatQuantMxFp4DavidSupport", "get npu arch failed.");
        return false;
    }
    return (std::atoi(buf) == static_cast<int32_t>(Ops::Base::DAV_3510) ||
            std::atoi(buf) == static_cast<int32_t>(Ops::Base::DAV_9201) ||
            std::atoi(buf) == static_cast<int32_t>(Ops::Base::DAV_9202));
}

static ge::graphStatus InferShape4FlatQuant(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        OP_LOGE_WITHOUT_REPORT("FlatQuant", "InferShape failed: context is nullptr");
        return GRAPH_FAILED;
    }
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    gert::Shape* qScaleShape = context->GetOutputShape(1);
    if ((xShape == nullptr) || (outShape == nullptr) || (qScaleShape == nullptr)) {
        OP_LOGE_WITHOUT_REPORT("FlatQuant", "InferShape failed: xShape=%p, outShape=%p, qScaleShape=%p",
                               static_cast<const void*>(xShape), static_cast<void*>(outShape),
                               static_cast<void*>(qScaleShape));
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "FlatQuantInferShape begin");
    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITHOUT_REPORT("FlatQuant", "InferShape failed: attrs is nullptr");
        return GRAPH_FAILED;
    }

    // Dynamic rank: propagate unknown rank to both outputs by convention; a known rank must be exactly 3 dims [K, M,
    // N].
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*outShape);
        Ops::Base::SetUnknownRank(*qScaleShape);
        return GRAPH_SUCCESS;
    }
    OP_CHECK_IF(
        xShape->GetDimNum() != FLATQUANT_X_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xShape->GetDimNum()).c_str(), "3"),
        return GRAPH_FAILED);
    // Each dim must be positive or the unknown dim (-1) for dynamic shapes, aligned with tiling CheckShapes.
    for (size_t i = 0; i < FLATQUANT_X_DIM_NUM; i++) {
        const int64_t dim = xShape->GetDim(i);
        OP_CHECK_IF(
            dim <= 0 && dim != UNKNOWN_DIM,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(*xShape).c_str(),
                                                  "each dim (K, M, N) of x must be positive or -1 for dynamic shapes"),
            return GRAPH_FAILED);
    }

    if (IsFlatQuantMxFp4DavidSupport()) { // Only for dav_3510
        // Validate the dst_type_max parameter if provided.
        const float* dstTypeMax = nullptr;
        if (attrs->GetAttrNum() >= FLATQUANT_ATTRS_NUM_THREE) {
            dstTypeMax = attrs->GetAttrPointer<float>(ATTR_INDEX_OF_DST_TYPE_MAX);
        }
        if (dstTypeMax != nullptr) {
            OP_CHECK_IF(
                !IsFloatEqual(*dstTypeMax, ZERO_FLOAT) && !(*dstTypeMax >= SIX_FLOAT && *dstTypeMax <= TWELVE_FLOAT),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "dst_type_max", std::to_string(*dstTypeMax).c_str(),
                    "the value of attr dst_type_max must be 0 or in the range [6, 12]"),
                return GRAPH_FAILED);
        }

        const int32_t* outxDtype = nullptr;
        if (attrs->GetAttrNum() >= FLATQUANT_ATTRS_NUM_TWO) {
            outxDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_DTYPE);
        }
        OP_CHECK_IF(
            xShape->GetDim(FLATQUANT_N_IDX) % FLATQUANT_N_IS_EVEN != 0 &&
                xShape->GetDim(FLATQUANT_N_IDX) != UNKNOWN_DIM, // allow unknown dim (-1) in dynamic shapes
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(*xShape).c_str(),
                                                  "the N dim of x must be an even number"),
            return ge::GRAPH_FAILED);

        if (outxDtype != nullptr) {
            int32_t dstDtype = *outxDtype;
            if (dstDtype == DTYPE_FLOAT4_E2M1 || dstDtype == DTYPE_HIFLOAT4) { // dst_dtype == 40 (mxfp4) / 42 (hif4)
                const int64_t mDim = xShape->GetDim(FLATQUANT_M_IDX);
                const int64_t nDim = xShape->GetDim(FLATQUANT_N_IDX);
                // Keep derived dims as -1 when either M/N is an unknown dim (-1); compute only when both
                // are known, avoiding negative values and unsigned wrap-around producing negative or huge dims.
                const bool hasUnknownDim = (mDim == UNKNOWN_DIM) || (nDim == UNKNOWN_DIM);
                const int64_t mnDim = hasUnknownDim ? UNKNOWN_DIM : mDim * nDim;
                const int64_t ceilMNDim = hasUnknownDim ?
                                              UNKNOWN_DIM :
                                              (mnDim + static_cast<int64_t>(FLATQUANT_DOUBLE_CEIL_SIZE) - 1) /
                                                  static_cast<int64_t>(FLATQUANT_DOUBLE_CEIL_SIZE);
                const size_t scaleLastDim = (dstDtype == DTYPE_FLOAT4_E2M1) ? FLATQUANT_MX_N_OUT_DIM :
                                                                              FLATQUANT_HIF4_N_OUT_DIM;
                outShape->SetDimNum(0);
                outShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
                outShape->AppendDim(mnDim);

                qScaleShape->SetDimNum(0);
                qScaleShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
                qScaleShape->AppendDim(ceilMNDim);
                qScaleShape->AppendDim(scaleLastDim);
            } else { // default
                outShape->SetDimNum(0);
                outShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
                outShape->AppendDim(xShape->GetDim(FLATQUANT_M_IDX));
                outShape->AppendDim(xShape->GetDim(FLATQUANT_N_IDX));

                qScaleShape->SetDimNum(0);
                qScaleShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
            }
        } else {
            outShape->SetDimNum(0);
            outShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
            outShape->AppendDim(xShape->GetDim(FLATQUANT_M_IDX));
            outShape->AppendDim(xShape->GetDim(FLATQUANT_N_IDX));

            qScaleShape->SetDimNum(0);
            qScaleShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
        }
    } else { // For 910B
        outShape->SetDimNum(0);
        outShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
        outShape->AppendDim(xShape->GetDim(FLATQUANT_M_IDX));
        outShape->AppendDim(xShape->GetDim(FLATQUANT_N_IDX));
        qScaleShape->SetDimNum(0);
        qScaleShape->AppendDim(xShape->GetDim(FLATQUANT_K_IDX));
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4FlatQuant(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE_WITHOUT_REPORT("FlatQuant", "InferDataType failed: context is nullptr");
        return GRAPH_FAILED;
    }
    OP_LOGD(context, "FlatQuantInferDataType begin");
    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITHOUT_REPORT("FlatQuant", "InferDataType failed: attrs is nullptr");
        return GRAPH_FAILED;
    }

    if (IsFlatQuantMxFp4DavidSupport()) { // For dav_3510
        const int32_t* outxDtype = nullptr;
        if (attrs->GetAttrNum() >= FLATQUANT_ATTRS_NUM_TWO) {
            outxDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_DTYPE);
        }
        if (outxDtype != nullptr) {
            int32_t dstDtype = *outxDtype;
            if (dstDtype == DTYPE_FLOAT4_E2M1) {
                context->SetOutputDataType(0, ge::DT_FLOAT4_E2M1);
                context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
            } else if (dstDtype == DTYPE_HIFLOAT4) {
                context->SetOutputDataType(0, Ops::NN::DT_HIFLOAT4);
                context->SetOutputDataType(1, Ops::NN::DT_HIFLOAT4_SCALE);
            } else {
                context->SetOutputDataType(0, ge::DT_INT4);
                context->SetOutputDataType(1, ge::DT_FLOAT);
            }
        } else {
            context->SetOutputDataType(0, ge::DT_INT4);
            context->SetOutputDataType(1, ge::DT_FLOAT);
        }
    } else { // For 910B
        context->SetOutputDataType(0, ge::DT_INT4);
        context->SetOutputDataType(1, ge::DT_FLOAT);
    }

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlatQuant).InferShape(InferShape4FlatQuant).InferDataType(InferDataType4FlatQuant);
} // namespace ops
