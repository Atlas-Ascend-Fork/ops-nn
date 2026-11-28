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
 * \file matmul_to_weight_quant_batch_matmul_fusion.cpp
 * \brief MatmulToWeightQuantBatchMatmulFusion
 *
 * Fusion pattern: fuse AscendAntiQuantV2 (+Transpose) + MatMul/BatchMatMul (+Add) (+AscendQuant[V2])
 * into A16W4 per-group WeightQuantBatchMatmulV2.
 *
 * Supported weight scenarios:
 *   1. Non-transposed FRACTAL_NZ weights are passed through, including dynamic weights.
 *   2. ND/NCHW/NHWC constant weights are packed into non-transposed FRACTAL_NZ constants.
 *   3. Transposed constant weights are packed into non-transposed FRACTAL_NZ constants;
 *      transposed dynamic weights are not fused.
 *
 *                  int4_weight   antiquant_scale   [antiquant_offset]
 *                       \              |                  /
 *                        AscendAntiQuantV2 (per-group dequantization)
 *                                   |
 *                            (optional) Transpose
 *                                   |
 *         fp16_input ---------------+
 *              \                   /
 *               MatMul/MatMulV2/MatMulV3/BatchMatMul/BatchMatMulV2/BatchMatMulV3
 *                                   |
 *                        (optional) Add  --- const_bias(N,)
 *                                   |
 *                     (optional) AscendQuant[V2] --- scale [offset]
 *                                   |
 *                              out (fp16/int8)
 *
 *            -------->
 *
 * fp16_input   int4_weight   antiquant_scale(fp16)   [antiquant_offset(fp16)]   [bias]
 *     \             |               |                       |                     /
 *      WeightQuantBatchMatmulV2 (attrs: transpose_x, transpose_weight=false, group_size, dtype)
 *                                   |
 *                              out (fp16/int8)
 *
 * INT8 output fusion:
 *
 *   MatMul/BatchMatMul (fp16)
 *                  |
 *           AscendQuant[V2] --- scale [offset]
 *                  |
 *               out (int8)
 *
 *            -------->
 *
 *   fp16_input   int4_weight   antiquant_scale   [antiquant_offset]   quant_scale(uint64)   [bias]
 *        \             |               |                    |                 |                /
 *         WeightQuantBatchMatmulV2 (output quantization fused, dtype=INT8)
 *                                   |
 *                                out (int8)
 *
 * quant_scale(uint64) encodes the quantization scale and integer offset.
 *
 * A following Add with an (N,) constant is folded into bias only when it is the sole MatMul consumer.
 * INT8 output quantization is absorbed only when AscendQuant[V2] is the sole consumer and its scale/offset
 * are readable constants or attributes. Quant scale and integer offset are encoded into quant_scale(uint64);
 * unsupported quantization remains as an independent node after the FP16 fusion.
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/inc/op_graph/cube_utils/cube_fp16_t.h"
#include "common/op_graph/fusion_pass/weight_quant_fusion_utils.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "securec.h"
#include "version/ge-compiler_version.h"
#include "matmul_to_weight_quant_batch_matmul_fusion.h"

namespace ops {
namespace {
const char* const kPassName = "MatmulToWeightQuantBatchMatmulFusion";
const char* const kAntiQuantType = "AscendAntiQuantV2";
const char* const kAddType = "Add";
const char* const kConstType = "Const";
const char* const kTransposeType = "Transpose";
const char* const kTransposeDType = "TransposeD";
const char* const kQuantType = "AscendQuant";
const char* const kQuantV2Type = "AscendQuantV2";

constexpr int64_t kXPort = 0;
constexpr int64_t kWeightPort = 1;
constexpr int64_t kScalePort = 2;
constexpr int64_t kOffsetPort = 3;
constexpr int64_t kQuantScalePort = 4;
constexpr int64_t kBiasPort = 6;
constexpr int64_t kAntiWeightPort = 0;
constexpr int64_t kAntiScalePort = 1;
constexpr int64_t kAntiOffsetPort = 2;
constexpr int64_t kMmBiasPort = 2;
constexpr int64_t kQuantNodeScalePort = 1;
constexpr int64_t kQuantNodeOffsetPort = 2;
constexpr int64_t kNzBlock = 16;
constexpr int64_t kMinGroupSize = 32;
constexpr size_t kMinXDimNum = 2;
constexpr size_t kMaxXDimNum = 6;
constexpr size_t kWeightDimNum = 2;
constexpr size_t kNzDimNum = 4;
constexpr size_t kPerTensorParamSize = 1;
constexpr uint32_t kDeqScaleMask = 0xFFFFE000U;
constexpr uint64_t kQuantOffsetValueShift = 37UL;
constexpr uint64_t kQuantOffsetValueMask = 0x00001FE000000000UL;
constexpr uint64_t kQuantOffsetSignMask = 0x0000200000000000UL;
constexpr uint64_t kQuantSignedFlagMask = 0x0000400000000000UL;
constexpr float kQuantOffsetMin = -128.0F;
constexpr float kQuantOffsetMax = 127.0F;
constexpr int32_t kDav510ArchType = 5102;
constexpr int64_t kDefaultOutputDType = -1; // WeightQuantBatchMatmulV2 "dtype" 属性的缺省值
constexpr size_t kPermDimNum = 2;           // 二维转置 perm 的元素个数
constexpr size_t kInt4ElemsPerByte = 2;
constexpr uint8_t kInt4NibbleMask = 0x0FU;
constexpr uint8_t kInt4HighNibbleMask = 0xF0U;
constexpr uint8_t kInt4BitWidth = 4;

struct FusionCtx {
    ge::GNodePtr nodeMatmul;
    ge::GNodePtr nodeAntiQuant;
    ge::GNodePtr nodeXTranspose;
    ge::GNodePtr nodeWeightTranspose;
    ge::GNodePtr nodeAdd;
    ge::GNodePtr nodeQuant;
    ge::GNodePtr nodeWeight;
    ge::GNodePtr nodeScale;
    ge::GNodePtr nodeOffset;
    ge::GNodePtr nodeAddBias;
    ge::GNodePtr nodeQuantScale;
    ge::GNodePtr nodeQuantOffset;
    ge::TensorDesc xDesc;
    ge::TensorDesc weightDesc;
    ge::TensorDesc outDesc;
    ge::TensorDesc biasDesc;
    std::vector<uint16_t> scaleData;
    std::vector<uint16_t> offsetData;
    std::vector<uint16_t> biasData;
    std::vector<uint8_t> packedWeight;
    std::vector<uint64_t> quantScaleData;
    int64_t kDim = 0;
    int64_t nDim = 0;
    int64_t groupSize = 0;
    int64_t weightInputIndex = -1;
    int64_t biasInputIndex = -1;
    bool transposeX = false;
    bool transposeWeight = false;
    bool inlineWeight = false;
    bool hasOffset = false;
    bool hasExternalBias = false;
};

std::string GetNodeType(const ge::GNode& node)
{
    ge::AscendString type;
    if (node.GetType(type) != ge::GRAPH_SUCCESS) {
        return "";
    }
    return type.GetString();
}

bool IsMatMulType(const ge::AscendString& type)
{
    const std::string value(type.GetString());
    return value == "MatMul" || value == "MatMulV2" || value == "MatMulV3" || value == "BatchMatMul" ||
           value == "BatchMatMulV2" || value == "BatchMatMulV3";
}

bool IsBatchMatMulType(const ge::AscendString& type)
{
    const std::string value(type.GetString());
    return value == "BatchMatMul" || value == "BatchMatMulV2" || value == "BatchMatMulV3";
}

bool IsConst(const ge::GNodePtr& node) { return node != nullptr && GetNodeType(*node) == kConstType; }

bool GetConstTensor(const ge::GNodePtr& node, ge::Tensor& tensor)
{
    return IsConst(node) && node->GetAttr(ge::AscendString("value"), tensor) == ge::GRAPH_SUCCESS &&
           tensor.GetData() != nullptr;
}

bool CheckPlatform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) ==
        ge::SUCCESS) {
        const auto mmad = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_mmad");
        const bool supportS8S4 = mmad != platformInfo.ai_core_intrinsic_dtype_map.end() &&
                                 std::find(mmad->second.begin(), mmad->second.end(), "s8s4") != mmad->second.end();
        const std::string soc = platformInfo.str_info.short_soc_version;
        const bool supportSoc = soc.find("MC62") != std::string::npos ||
                                platformInfo.soc_info.arch_type == kDav510ArchType;
        if (!supportS8S4 || !supportSoc) {
            OP_LOGD(kPassName, "A16W4 fusion requires MC62 and mmad s8s4 support");
            return false;
        }
        return true;
    }
    OP_LOGD(kPassName, "Failed to get platform info");
    return false;
}

bool GetTransposePerm(const ge::GNode& node, std::vector<int64_t>& perm)
{
    const std::string type = GetNodeType(node);
    if (type == kTransposeDType) {
        return node.GetAttr(ge::AscendString("perm"), perm) == ge::GRAPH_SUCCESS;
    }
    if (type != kTransposeType) {
        return false;
    }
    ge::Tensor tensor;
    if (!GetConstTensor(node.GetInDataNodesAndPortIndexs(1).first, tensor)) {
        return false;
    }
    const auto dtype = tensor.GetTensorDesc().GetDataType();
    if (dtype == ge::DT_INT32) {
        if (tensor.GetSize() < kPermDimNum * sizeof(int32_t)) {
            return false;
        }
        const auto* data = reinterpret_cast<const int32_t*>(tensor.GetData());
        perm = {data[0], data[1]};
        return true;
    }
    if (dtype == ge::DT_INT64) {
        if (tensor.GetSize() < kPermDimNum * sizeof(int64_t)) {
            return false;
        }
        const auto* data = reinterpret_cast<const int64_t*>(tensor.GetData());
        perm = {data[0], data[1]};
        return true;
    }
    return false;
}

bool IsFusibleTranspose(const ge::GNodePtr& node)
{
    if (node == nullptr || node->GetOutDataNodesAndPortIndexs(0).size() != 1 || !node->GetOutControlNodes().empty()) {
        return false;
    }
    std::vector<int64_t> perm;
    return GetTransposePerm(*node, perm) && perm == std::vector<int64_t>{1, 0};
}

bool MatchXInput(FusionCtx& ctx)
{
    auto input = ctx.nodeMatmul->GetInDataNodesAndPortIndexs(kXPort).first;
    if (input == nullptr) {
        return false;
    }
    const std::string type = GetNodeType(*input);
    if (type == kTransposeType || type == kTransposeDType) {
        if (ctx.transposeX || !IsFusibleTranspose(input)) {
            return false;
        }
        ctx.nodeXTranspose = input;
        ctx.transposeX = true;
        return input->GetInputDesc(0, ctx.xDesc) == ge::GRAPH_SUCCESS;
    }
    return ctx.nodeMatmul->GetInputDesc(kXPort, ctx.xDesc) == ge::GRAPH_SUCCESS;
}

bool MatchAntiQuant(FusionCtx& ctx)
{
    auto input = ctx.nodeMatmul->GetInDataNodesAndPortIndexs(kWeightPort).first;
    if (input == nullptr) {
        return false;
    }
    const std::string type = GetNodeType(*input);
    if (type == kTransposeType || type == kTransposeDType) {
        if (ctx.transposeWeight || !IsFusibleTranspose(input)) {
            return false;
        }
        ctx.nodeWeightTranspose = input;
        ctx.transposeWeight = true;
        input = input->GetInDataNodesAndPortIndexs(0).first;
    }
    if (input == nullptr || GetNodeType(*input) != kAntiQuantType ||
        input->GetOutDataNodesAndPortIndexs(0).size() != 1) {
        return false;
    }
    ctx.nodeAntiQuant = input;
    ctx.nodeWeight = input->GetInDataNodesAndPortIndexs(kAntiWeightPort).first;
    ctx.nodeScale = input->GetInDataNodesAndPortIndexs(kAntiScalePort).first;
    ctx.nodeOffset = input->GetInDataNodesAndPortIndexs(kAntiOffsetPort).first;
    return ctx.nodeWeight != nullptr && ctx.nodeScale != nullptr;
}

int64_t CeilDiv16(int64_t value) { return value / kNzBlock + ((value % kNzBlock) == 0 ? 0 : 1); }

bool GetElementNum(const std::vector<int64_t>& dims, size_t& elementNum)
{
    size_t result = 1;
    for (const auto dim : dims) {
        if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / result) {
            return false;
        }
        result *= static_cast<size_t>(dim);
    }
    elementNum = result;
    return true;
}

bool CheckWeightDesc(FusionCtx& ctx)
{
    if (ctx.nodeAntiQuant->GetInputDesc(kAntiWeightPort, ctx.weightDesc) != ge::GRAPH_SUCCESS ||
        ctx.weightDesc.GetDataType() != ge::DT_INT4) {
        return false;
    }
    const auto origin = ctx.weightDesc.GetOriginShape().GetDims();
    if (origin.size() != kWeightDimNum || origin[0] <= 0 || origin[1] <= 0) {
        return false;
    }
    const int64_t kDim = ctx.transposeWeight ? origin[1] : origin[0];
    const int64_t nDim = ctx.transposeWeight ? origin[0] : origin[1];
    if (kDim != ctx.kDim || nDim != ctx.nDim) {
        return false;
    }

    const auto storage = ctx.weightDesc.GetShape().GetDims();
    const auto format = ctx.weightDesc.GetFormat();
    const bool isNdLike = (format == ge::FORMAT_ND || format == ge::FORMAT_NCHW || format == ge::FORMAT_NHWC);
    if (isNdLike) {
        if (storage != origin || !IsConst(ctx.nodeWeight)) {
            return false;
        }
        ctx.inlineWeight = true;
        return true;
    }
    if (format != ge::FORMAT_FRACTAL_NZ || storage.size() != kNzDimNum || storage[2] != kNzBlock ||
        storage[3] != kNzBlock) {
        return false;
    }
    const std::vector<int64_t> expected = ctx.transposeWeight ?
                                              std::vector<int64_t>{CeilDiv16(ctx.kDim), CeilDiv16(ctx.nDim), kNzBlock,
                                                                   kNzBlock} :
                                              std::vector<int64_t>{CeilDiv16(ctx.nDim), CeilDiv16(ctx.kDim), kNzBlock,
                                                                   kNzBlock};
    if (storage != expected) {
        return false;
    }
    if (ctx.transposeWeight) {
        if (!IsConst(ctx.nodeWeight)) {
            OP_LOGD(kPassName, "Transposed dynamic weight is not fused");
            return false;
        }
        ctx.inlineWeight = true;
    }
    return true;
}

bool CheckShapesAndAttrs(FusionCtx& ctx)
{
    ge::AscendString mmType;
    if (ctx.nodeMatmul->GetType(mmType) != ge::GRAPH_SUCCESS) {
        return false;
    }
    const bool isBatch = IsBatchMatMulType(mmType);
    (void)ctx.nodeMatmul->GetAttr(ge::AscendString(isBatch ? "adj_x1" : "transpose_x1"), ctx.transposeX);
    (void)ctx.nodeMatmul->GetAttr(ge::AscendString(isBatch ? "adj_x2" : "transpose_x2"), ctx.transposeWeight);
    if (!MatchXInput(ctx) || !MatchAntiQuant(ctx)) {
        return false;
    }

    const auto xDims = ctx.xDesc.GetOriginShape().GetDims();
    if (xDims.size() < kMinXDimNum || xDims.size() > kMaxXDimNum ||
        (ctx.xDesc.GetFormat() != ge::FORMAT_ND && ctx.xDesc.GetFormat() != ge::FORMAT_NCHW)) {
        return false;
    }
    size_t xElementNum = 0;
    if (!GetElementNum(xDims, xElementNum) || ctx.xDesc.GetDataType() != ge::DT_FLOAT16) {
        return false;
    }
    const size_t last = xDims.size() - 1;
    ctx.kDim = ctx.transposeX ? xDims[last - 1] : xDims[last];

    const auto antiWeightDims = ctx.nodeAntiQuant->GetInputDesc(kAntiWeightPort, ctx.weightDesc) == ge::GRAPH_SUCCESS ?
                                    ctx.weightDesc.GetOriginShape().GetDims() :
                                    std::vector<int64_t>();
    if (antiWeightDims.size() != kWeightDimNum) {
        return false;
    }
    ctx.nDim = ctx.transposeWeight ? antiWeightDims[0] : antiWeightDims[1];

    ge::TensorDesc antiOutDesc;
    ge::TensorDesc mmWeightDesc;
    if (ctx.nodeAntiQuant->GetOutputDesc(0, antiOutDesc) != ge::GRAPH_SUCCESS ||
        ctx.nodeMatmul->GetInputDesc(kWeightPort, mmWeightDesc) != ge::GRAPH_SUCCESS ||
        antiOutDesc.GetDataType() != ge::DT_FLOAT16 || mmWeightDesc.GetDataType() != ge::DT_FLOAT16) {
        return false;
    }
    bool sqrtMode = false;
    int64_t dstType = ge::DT_FLOAT16;
    (void)ctx.nodeAntiQuant->GetAttr(ge::AscendString("sqrt_mode"), sqrtMode);
    (void)ctx.nodeAntiQuant->GetAttr(ge::AscendString("dst_type"), dstType);
    if (sqrtMode || dstType != ge::DT_FLOAT16) {
        return false;
    }

    ge::TensorDesc scaleDesc;
    if (ctx.nodeAntiQuant->GetInputDesc(kAntiScalePort, scaleDesc) != ge::GRAPH_SUCCESS ||
        (scaleDesc.GetDataType() != ge::DT_FLOAT16 && scaleDesc.GetDataType() != ge::DT_FLOAT)) {
        return false;
    }
    const auto scaleDims = scaleDesc.GetOriginShape().GetDims();
    if (scaleDims.size() != kWeightDimNum || scaleDims[0] <= 0 || scaleDims[1] != ctx.nDim ||
        ctx.kDim % scaleDims[0] != 0) {
        return false;
    }
    ctx.groupSize = ctx.kDim / scaleDims[0];
    if (ctx.groupSize < kMinGroupSize || ctx.groupSize >= ctx.kDim || ctx.groupSize % kMinGroupSize != 0) {
        return false;
    }
    ctx.hasOffset = ctx.nodeOffset != nullptr;
    if (ctx.hasOffset) {
        ge::TensorDesc offsetDesc;
        if (ctx.nodeAntiQuant->GetInputDesc(kAntiOffsetPort, offsetDesc) != ge::GRAPH_SUCCESS ||
            (offsetDesc.GetDataType() != ge::DT_FLOAT16 && offsetDesc.GetDataType() != ge::DT_FLOAT) ||
            offsetDesc.GetOriginShape().GetDims() != scaleDims) {
            return false;
        }
    }
    if (!CheckWeightDesc(ctx)) {
        return false;
    }

    auto biasPeer = ctx.nodeMatmul->GetInDataNodesAndPortIndexs(kMmBiasPort).first;
    ctx.hasExternalBias = biasPeer != nullptr;
    if (ctx.hasExternalBias) {
        if (ctx.nodeMatmul->GetInputDesc(kMmBiasPort, ctx.biasDesc) != ge::GRAPH_SUCCESS ||
            ctx.biasDesc.GetDataType() != ge::DT_FLOAT16 ||
            ctx.biasDesc.GetOriginShape().GetDims() != std::vector<int64_t>{ctx.nDim}) {
            return false;
        }
    }

    const auto consumers = ctx.nodeMatmul->GetOutDataNodesAndPortIndexs(0);
    if (!ctx.hasExternalBias && consumers.size() == 1 && consumers.front().first != nullptr &&
        GetNodeType(*consumers.front().first) == kAddType) {
        auto add = consumers.front().first;
        const int64_t chainPort = consumers.front().second;
        if (chainPort == 0 || chainPort == 1) {
            auto bias = add->GetInDataNodesAndPortIndexs(1 - chainPort).first;
            ge::TensorDesc biasDesc;
            ge::TensorDesc addOutDesc;
            if (IsConst(bias) && add->GetInputDesc(1 - chainPort, biasDesc) == ge::GRAPH_SUCCESS &&
                add->GetOutputDesc(0, addOutDesc) == ge::GRAPH_SUCCESS &&
                (biasDesc.GetDataType() == ge::DT_FLOAT16 || biasDesc.GetDataType() == ge::DT_FLOAT) &&
                biasDesc.GetOriginShape().GetDims() == std::vector<int64_t>{ctx.nDim} &&
                addOutDesc.GetDataType() == ge::DT_FLOAT16) {
                ctx.nodeAdd = add;
                ctx.nodeAddBias = bias;
                ctx.outDesc = addOutDesc;
            }
        }
    }
    if (ctx.nodeAdd == nullptr && (ctx.nodeMatmul->GetOutputDesc(0, ctx.outDesc) != ge::GRAPH_SUCCESS ||
                                   ctx.outDesc.GetDataType() != ge::DT_FLOAT16)) {
        return false;
    }
    const auto outDims = ctx.outDesc.GetOriginShape().GetDims();
    return !outDims.empty() && outDims.back() == ctx.nDim;
}

bool ReadFp16Const(const ge::GNodePtr& node, const std::vector<int64_t>& expectedDims, std::vector<uint16_t>& result)
{
    ge::Tensor tensor;
    if (!GetConstTensor(node, tensor)) {
        return false;
    }
    const auto& tensorDesc = tensor.GetTensorDesc();
    if (tensorDesc.GetShape().GetDims() != expectedDims) {
        return false;
    }
    size_t elementNum = 0;
    if (!GetElementNum(expectedDims, elementNum) || elementNum > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return false;
    }
    const auto dtype = tensorDesc.GetDataType();
    const size_t elementSize = dtype == ge::DT_FLOAT16 ? sizeof(uint16_t) : sizeof(float);
    if ((dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT) || tensor.GetSize() < elementNum * elementSize) {
        return false;
    }
    result.resize(elementNum);
    if (dtype == ge::DT_FLOAT16) {
        const auto* data = reinterpret_cast<const uint16_t*>(tensor.GetData());
        std::copy(data, data + elementNum, result.begin());
    } else {
        const auto* data = reinterpret_cast<const float*>(tensor.GetData());
        for (size_t i = 0; i < elementNum; ++i) {
            fp16_t value(data[i]);
            result[i] = value.val;
        }
    }
    return true;
}

float DecodeDeqScale(uint64_t encoded)
{
    const uint32_t bits = static_cast<uint32_t>(encoded);
    float value = 0.0F;
    if (memcpy_s(&value, sizeof(value), &bits, sizeof(bits)) != EOK) {
        return 0.0F;
    }
    return value;
}

uint64_t EncodeDeqScale(float value)
{
    uint32_t bits = 0U;
    if (memcpy_s(&bits, sizeof(bits), &value, sizeof(value)) != EOK) {
        return 0UL;
    }
    return static_cast<uint64_t>(bits & kDeqScaleMask);
}

bool CheckQuantOffsetValue(float offset)
{
    return std::isfinite(offset) && offset == std::round(offset) && offset >= kQuantOffsetMin &&
           offset <= kQuantOffsetMax;
}

uint64_t EncodeQuantScale(float scale, float offset)
{
    uint64_t encoded = EncodeDeqScale(scale);
    if (offset == 0.0F) {
        return encoded;
    }
    const auto offsetInt = static_cast<int64_t>(offset);
    encoded |= (static_cast<uint64_t>(offsetInt) << kQuantOffsetValueShift) & kQuantOffsetValueMask;
    if (offsetInt < 0) {
        encoded |= kQuantOffsetSignMask;
    }
    return encoded | kQuantSignedFlagMask;
}

bool ReadConstFloats(const ge::GNodePtr& node, std::vector<float>& values)
{
    ge::Tensor tensor;
    if (!GetConstTensor(node, tensor)) {
        return false;
    }
    const auto dtype = tensor.GetTensorDesc().GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_UINT64) {
        return false;
    }
    const size_t elementSize = dtype == ge::DT_UINT64 ? sizeof(uint64_t) :
                                                        (dtype == ge::DT_FLOAT16 ? sizeof(uint16_t) : sizeof(float));
    if (tensor.GetSize() == 0UL || tensor.GetSize() % elementSize != 0UL) {
        return false;
    }
    const size_t elementNum = tensor.GetSize() / elementSize;
    if (dtype == ge::DT_FLOAT) {
        const auto* data = reinterpret_cast<const float*>(tensor.GetData());
        values.assign(data, data + elementNum);
    } else if (dtype == ge::DT_FLOAT16) {
        const auto* data = reinterpret_cast<const uint16_t*>(tensor.GetData());
        values.resize(elementNum);
        for (size_t i = 0; i < elementNum; ++i) {
            ops::tagFp16 half(data[i]);
            values[i] = half.ToFloat();
        }
    } else {
        const auto* data = reinterpret_cast<const uint64_t*>(tensor.GetData());
        values.resize(elementNum);
        for (size_t i = 0; i < elementNum; ++i) {
            values[i] = DecodeDeqScale(data[i]);
        }
    }
    return true;
}

bool CheckQuantSemantics(const ge::GNode& node)
{
    bool sqrtMode = false;
    (void)node.GetAttr(ge::AscendString("sqrt_mode"), sqrtMode);
    if (sqrtMode) {
        return false;
    }
    ge::AscendString roundMode;
    if (node.GetAttr(ge::AscendString("round_mode"), roundMode) == ge::GRAPH_SUCCESS) {
        std::string mode(roundMode.GetString());
        std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
        return mode.empty() || mode == "round";
    }
    return true;
}

bool ReadV1QuantParams(const ge::GNode& quantNode, std::vector<float>& quantScale, std::vector<float>& quantOffset)
{
    float scale = 0.0F;
    float offset = 0.0F;
    if (quantNode.GetAttr(ge::AscendString("scale"), scale) != ge::GRAPH_SUCCESS ||
        quantNode.GetAttr(ge::AscendString("offset"), offset) != ge::GRAPH_SUCCESS || !CheckQuantOffsetValue(offset)) {
        return false;
    }
    quantScale = {scale};
    quantOffset = {offset};
    return true;
}

bool CheckQuantParamShape(const ge::GNodePtr& node, int64_t nDim)
{
    ge::Tensor tensor;
    if (!GetConstTensor(node, tensor)) {
        return false;
    }
    const auto dims = tensor.GetTensorDesc().GetShape().GetDims();
    if (dims.empty() || (dims.back() != 1 && dims.back() != nDim)) {
        return false;
    }
    return std::all_of(dims.begin(), dims.end() - 1, [](int64_t dim) { return dim == 1; });
}

bool CheckQuantAxis(const ge::GNode& quantNode, int64_t outRank, bool perChannel)
{
    if (!perChannel) {
        return true;
    }
    int64_t axis = -1;
    (void)quantNode.GetAttr(ge::AscendString("axis"), axis);
    return axis == -1 || axis == outRank - 1;
}

bool ReadV2QuantParams(const ge::GNodePtr& quantNode, int64_t nDim, std::vector<float>& quantScale,
                       std::vector<float>& quantOffset)
{
    auto scaleNode = quantNode->GetInDataNodesAndPortIndexs(kQuantNodeScalePort).first;
    if (!ReadConstFloats(scaleNode, quantScale) || quantScale.empty() ||
        (quantScale.size() != kPerTensorParamSize && static_cast<int64_t>(quantScale.size()) != nDim) ||
        !CheckQuantParamShape(scaleNode, nDim)) {
        return false;
    }
    auto offsetNode = quantNode->GetInDataNodesAndPortIndexs(kQuantNodeOffsetPort).first;
    if (offsetNode == nullptr) {
        return true;
    }
    if (!ReadConstFloats(offsetNode, quantOffset) || quantOffset.empty() ||
        (quantOffset.size() != kPerTensorParamSize && static_cast<int64_t>(quantOffset.size()) != nDim) ||
        !CheckQuantParamShape(offsetNode, nDim)) {
        return false;
    }
    return std::all_of(quantOffset.begin(), quantOffset.end(), CheckQuantOffsetValue);
}

bool EncodeQuantParams(const std::vector<float>& quantScale, const std::vector<float>& quantOffset, int64_t nDim,
                       std::vector<uint64_t>& result)
{
    if (quantScale.empty() ||
        (quantScale.size() != kPerTensorParamSize && static_cast<int64_t>(quantScale.size()) != nDim) ||
        (!quantOffset.empty() && quantOffset.size() != kPerTensorParamSize &&
         static_cast<int64_t>(quantOffset.size()) != nDim)) {
        return false;
    }
    const size_t resultSize = std::max(quantScale.size(), quantOffset.size());
    result.resize(resultSize);
    for (size_t i = 0; i < resultSize; ++i) {
        const float scale = quantScale[i < quantScale.size() ? i : 0UL];
        const float offset = quantOffset.empty() ? 0.0F : quantOffset[i < quantOffset.size() ? i : 0UL];
        if (!std::isfinite(scale) || scale <= 0.0F || !CheckQuantOffsetValue(offset)) {
            return false;
        }
        result[i] = EncodeQuantScale(scale, offset);
    }
    return true;
}

bool MatchQuantFusion(FusionCtx& ctx)
{
    const auto outputNode = ctx.nodeAdd != nullptr ? ctx.nodeAdd : ctx.nodeMatmul;
    const auto consumers = outputNode->GetOutDataNodesAndPortIndexs(0);
    if (consumers.size() != 1 || consumers.front().first == nullptr) {
        return false;
    }
    auto quantNode = consumers.front().first;
    const std::string type = GetNodeType(*quantNode);
    if (type != kQuantType && type != kQuantV2Type) {
        return false;
    }
    ge::TensorDesc quantOutDesc;
    if (quantNode->GetOutputDesc(0, quantOutDesc) != ge::GRAPH_SUCCESS || quantOutDesc.GetDataType() != ge::DT_INT8 ||
        quantOutDesc.GetOriginShape().GetDims() != ctx.outDesc.GetOriginShape().GetDims() ||
        !CheckQuantSemantics(*quantNode)) {
        return false;
    }
    std::vector<float> quantScale;
    std::vector<float> quantOffset;
    const bool paramsValid = type == kQuantType ? ReadV1QuantParams(*quantNode, quantScale, quantOffset) :
                                                  ReadV2QuantParams(quantNode, ctx.nDim, quantScale, quantOffset);
    const bool perChannel = quantScale.size() > kPerTensorParamSize || quantOffset.size() > kPerTensorParamSize;
    if (!paramsValid ||
        (type == kQuantV2Type &&
         !CheckQuantAxis(*quantNode, static_cast<int64_t>(quantOutDesc.GetOriginShape().GetDims().size()),
                         perChannel)) ||
        !EncodeQuantParams(quantScale, quantOffset, ctx.nDim, ctx.quantScaleData)) {
        return false;
    }
    ctx.nodeQuant = quantNode;
    if (type == kQuantV2Type) {
        ctx.nodeQuantScale = quantNode->GetInDataNodesAndPortIndexs(kQuantNodeScalePort).first;
        ctx.nodeQuantOffset = quantNode->GetInDataNodesAndPortIndexs(kQuantNodeOffsetPort).first;
    }
    ctx.outDesc = quantOutDesc;
    return true;
}

uint8_t GetInt4(const uint8_t* data, size_t index)
{
    const uint8_t value = data[index / kInt4ElemsPerByte];
    return (index % kInt4ElemsPerByte == 0) ? (value & kInt4NibbleMask) : ((value >> kInt4BitWidth) & kInt4NibbleMask);
}

void SetInt4(std::vector<uint8_t>& data, size_t index, uint8_t value)
{
    uint8_t& target = data[index / kInt4ElemsPerByte];
    if (index % kInt4ElemsPerByte == 0) {
        target = (target & kInt4HighNibbleMask) | (value & kInt4NibbleMask);
    } else {
        target = (target & kInt4NibbleMask) | ((value & kInt4NibbleMask) << kInt4BitWidth);
    }
}

bool PackWeightToNz(FusionCtx& ctx)
{
    ge::Tensor tensor;
    if (!GetConstTensor(ctx.nodeWeight, tensor) || tensor.GetTensorDesc().GetDataType() != ge::DT_INT4) {
        return false;
    }
    size_t sourceElementNum = 0;
    if (!GetElementNum(ctx.weightDesc.GetShape().GetDims(), sourceElementNum) ||
        tensor.GetSize() < (sourceElementNum + kInt4ElemsPerByte - 1) / kInt4ElemsPerByte) {
        return false;
    }
    const int64_t n1 = CeilDiv16(ctx.nDim);
    const int64_t k1 = CeilDiv16(ctx.kDim);
    const std::vector<int64_t> nzDims{n1, k1, kNzBlock, kNzBlock};
    size_t nzElementNum = 0;
    if (!GetElementNum(nzDims, nzElementNum)) {
        return false;
    }
    ctx.packedWeight.assign(nzElementNum / kInt4ElemsPerByte, 0U);
    const auto* source = tensor.GetData();
    const bool sourceNz = ctx.weightDesc.GetFormat() == ge::FORMAT_FRACTAL_NZ;
    for (int64_t k = 0; k < ctx.kDim; ++k) {
        for (int64_t n = 0; n < ctx.nDim; ++n) {
            size_t sourceIndex = 0;
            if (ctx.transposeWeight) {
                sourceIndex = sourceNz ? static_cast<size_t>(
                                             ((k / kNzBlock * n1 + n / kNzBlock) * kNzBlock + n % kNzBlock) * kNzBlock +
                                             k % kNzBlock) :
                                         static_cast<size_t>(n * ctx.kDim + k);
            } else {
                sourceIndex = sourceNz ? static_cast<size_t>(
                                             ((n / kNzBlock * k1 + k / kNzBlock) * kNzBlock + k % kNzBlock) * kNzBlock +
                                             n % kNzBlock) :
                                         static_cast<size_t>(k * ctx.nDim + n);
            }
            if (sourceIndex >= sourceElementNum) {
                return false;
            }
            const size_t targetIndex = static_cast<size_t>(
                ((n / kNzBlock * k1 + k / kNzBlock) * kNzBlock + k % kNzBlock) * kNzBlock + n % kNzBlock);
            SetInt4(ctx.packedWeight, targetIndex, GetInt4(source, sourceIndex));
        }
    }
    return true;
}

bool PrepareConstants(FusionCtx& ctx)
{
    const std::vector<int64_t> scaleDims{ctx.kDim / ctx.groupSize, ctx.nDim};
    if (!ReadFp16Const(ctx.nodeScale, scaleDims, ctx.scaleData)) {
        OP_LOGD(kPassName, "antiquant_scale must be a readable FP16/FP32 constant");
        return false;
    }
    if (ctx.hasOffset && !ReadFp16Const(ctx.nodeOffset, scaleDims, ctx.offsetData)) {
        OP_LOGD(kPassName, "antiquant_offset must be a readable FP16/FP32 constant");
        return false;
    }
    if (ctx.nodeAdd != nullptr && !ReadFp16Const(ctx.nodeAddBias, {ctx.nDim}, ctx.biasData)) {
        return false;
    }
    return !ctx.inlineWeight || PackWeightToNz(ctx);
}

void SetDesc(ge::TensorDesc& target, const ge::TensorDesc& source)
{
    target.SetDataType(source.GetDataType());
    target.SetShape(source.GetShape());
    target.SetFormat(source.GetFormat());
    target.SetOriginShape(source.GetOriginShape());
    target.SetOriginFormat(source.GetOriginFormat());
}

ge::TensorDesc MakeDesc(const std::vector<int64_t>& storageDims, const std::vector<int64_t>& originDims,
                        ge::DataType dtype, ge::Format format, ge::Format originFormat = ge::FORMAT_ND)
{
    ge::TensorDesc desc;
    desc.SetDataType(dtype);
    desc.SetShape(ge::Shape(storageDims));
    desc.SetFormat(format);
    desc.SetOriginShape(ge::Shape(originDims));
    desc.SetOriginFormat(originFormat);
    return desc;
}

bool SetProducerDesc(const ge::es::EsTensorHolder& tensor, const ge::TensorDesc& desc)
{
    auto* producer = tensor.GetProducer();
    if (producer == nullptr) {
        return false;
    }
    ge::TensorDesc target;
    if (producer->GetOutputDesc(0, target) != ge::GRAPH_SUCCESS) {
        return false;
    }
    SetDesc(target, desc);
    return producer->UpdateOutputDesc(0, target) == ge::GRAPH_SUCCESS;
}

bool SetConsumerDesc(ge::GNode* consumer, int64_t port, const ge::TensorDesc& desc)
{
    if (consumer == nullptr) {
        return false;
    }
    ge::TensorDesc target;
    if (consumer->GetInputDesc(port, target) != ge::GRAPH_SUCCESS) {
        return false;
    }
    SetDesc(target, desc);
    return consumer->UpdateInputDesc(port, target) == ge::GRAPH_SUCCESS;
}

ge::es::EsTensorHolder CreateInt4NzConst(ge::es::EsGraphBuilder& builder, const FusionCtx& ctx,
                                         const ge::TensorDesc& desc)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    if (graph == nullptr) {
        return ge::es::EsTensorHolder(nullptr);
    }
    ge::Tensor tensor(desc, ctx.packedWeight);
    ge::AttrValue value;
    if (value.SetAttrValue(tensor) != ge::GRAPH_SUCCESS) {
        return ge::es::EsTensorHolder(nullptr);
    }
    auto node = ge::es::CompliantNodeBuilder(graph)
                    .OpType(kConstType)
                    .Name("weight_nz_const")
                    .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({{"value", ge::es::CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR", value}})
                    .InstanceOutputShape("y", desc.GetShape().GetDims())
                    .InstanceOutputDataType("y", ge::DT_INT4)
                    .InstanceOutputFormat("y", ge::FORMAT_FRACTAL_NZ)
                    .Build();
    return ge::es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

ge::fusion::GraphUniqPtr BuildReplacementGraph(FusionCtx& ctx)
{
    auto builder = ge::es::EsGraphBuilder("anti_quant_v2_matmul_replacement");
    auto rX = builder.CreateInput(0, "x", ctx.xDesc.GetDataType(), ctx.xDesc.GetFormat(),
                                  ctx.xDesc.GetShape().GetDims());
    int64_t nextInputIndex = 1;

    const std::vector<int64_t> nzDims{CeilDiv16(ctx.nDim), CeilDiv16(ctx.kDim), kNzBlock, kNzBlock};
    const auto inlineWeightDesc = MakeDesc(nzDims, {ctx.kDim, ctx.nDim}, ge::DT_INT4, ge::FORMAT_FRACTAL_NZ);
    ge::es::EsTensorHolder rWeight(nullptr);
    if (ctx.inlineWeight) {
        rWeight = CreateInt4NzConst(builder, ctx, inlineWeightDesc);
    } else {
        ctx.weightInputIndex = nextInputIndex++;
        rWeight = builder.CreateInput(ctx.weightInputIndex, "weight", ctx.weightDesc.GetDataType(),
                                      ctx.weightDesc.GetFormat(), ctx.weightDesc.GetShape().GetDims());
    }

    const std::vector<int64_t> scaleDims{ctx.kDim / ctx.groupSize, ctx.nDim};
    const auto scaleDesc = MakeDesc(scaleDims, scaleDims, ge::DT_FLOAT16, ge::FORMAT_ND);
    auto rScale = builder.CreateConst(ctx.scaleData, scaleDims, ge::DT_FLOAT16, ge::FORMAT_ND);
    ge::es::EsTensorHolder rOffset(nullptr);
    if (ctx.hasOffset) {
        rOffset = builder.CreateConst(ctx.offsetData, scaleDims, ge::DT_FLOAT16, ge::FORMAT_ND);
    }
    ge::es::EsTensorHolder rQuantScale(nullptr);
    ge::TensorDesc quantScaleDesc;
    if (ctx.nodeQuant != nullptr) {
        const std::vector<int64_t> quantScaleDims{static_cast<int64_t>(ctx.quantScaleData.size())};
        quantScaleDesc = MakeDesc(quantScaleDims, quantScaleDims, ge::DT_UINT64, ge::FORMAT_ND);
        rQuantScale = builder.CreateConst(ctx.quantScaleData, quantScaleDims, ge::DT_UINT64, ge::FORMAT_ND);
    }

    ge::es::EsTensorHolder rBias(nullptr);
    auto biasDesc = MakeDesc({ctx.nDim}, {ctx.nDim}, ge::DT_FLOAT16, ge::FORMAT_ND);
    if (ctx.nodeAdd != nullptr) {
        rBias = builder.CreateConst(ctx.biasData, {ctx.nDim}, ge::DT_FLOAT16, ge::FORMAT_ND);
    } else if (ctx.hasExternalBias) {
        ctx.biasInputIndex = nextInputIndex++;
        rBias = builder.CreateInput(ctx.biasInputIndex, "bias", ctx.biasDesc.GetDataType(), ctx.biasDesc.GetFormat(),
                                    ctx.biasDesc.GetShape().GetDims());
        biasDesc = ctx.biasDesc;
    }
    if (rX.GetProducer() == nullptr || rWeight.GetProducer() == nullptr || rScale.GetProducer() == nullptr ||
        (ctx.hasOffset && rOffset.GetProducer() == nullptr) ||
        (ctx.nodeQuant != nullptr && rQuantScale.GetProducer() == nullptr) ||
        ((ctx.nodeAdd != nullptr || ctx.hasExternalBias) && rBias.GetProducer() == nullptr)) {
        return nullptr;
    }

    const int64_t outputDtype = ctx.nodeQuant != nullptr ? static_cast<int64_t>(ge::DT_INT8) : kDefaultOutputDType;
    auto rOut = ge::es::WeightQuantBatchMatmulV2(rX, rWeight, rScale, rOffset, rQuantScale, nullptr, rBias,
                                                 ctx.transposeX, false, ctx.groupSize, outputDtype);
    auto* fusedNode = rOut.GetProducer();
    if (fusedNode == nullptr || !SetProducerDesc(rX, ctx.xDesc) ||
        !SetProducerDesc(rWeight, ctx.inlineWeight ? inlineWeightDesc : ctx.weightDesc) ||
        !SetProducerDesc(rScale, scaleDesc) || (ctx.hasOffset && !SetProducerDesc(rOffset, scaleDesc)) ||
        (ctx.nodeQuant != nullptr && !SetProducerDesc(rQuantScale, quantScaleDesc)) ||
        ((ctx.nodeAdd != nullptr || ctx.hasExternalBias) && !SetProducerDesc(rBias, biasDesc)) ||
        !SetProducerDesc(rOut, ctx.outDesc) || !SetConsumerDesc(fusedNode, kXPort, ctx.xDesc) ||
        !SetConsumerDesc(fusedNode, kWeightPort, ctx.inlineWeight ? inlineWeightDesc : ctx.weightDesc) ||
        !SetConsumerDesc(fusedNode, kScalePort, scaleDesc) ||
        (ctx.hasOffset && !SetConsumerDesc(fusedNode, kOffsetPort, scaleDesc)) ||
        (ctx.nodeQuant != nullptr && !SetConsumerDesc(fusedNode, kQuantScalePort, quantScaleDesc)) ||
        ((ctx.nodeAdd != nullptr || ctx.hasExternalBias) && !SetConsumerDesc(fusedNode, kBiasPort, biasDesc))) {
        return nullptr;
    }
    return builder.BuildAndReset({rOut});
}

std::unique_ptr<ge::fusion::SubgraphBoundary> BuildBoundary(const FusionCtx& ctx)
{
    auto boundary = std::make_unique<ge::fusion::SubgraphBoundary>();
    ge::fusion::SubgraphInput xInput;
    if (ctx.nodeXTranspose != nullptr) {
        xInput.AddInput({*ctx.nodeXTranspose, 0});
    } else {
        xInput.AddInput({*ctx.nodeMatmul, kXPort});
    }
    if (boundary->AddInput(0, std::move(xInput)) != ge::SUCCESS) {
        return nullptr;
    }
    if (!ctx.inlineWeight) {
        ge::fusion::SubgraphInput weightInput;
        weightInput.AddInput({*ctx.nodeAntiQuant, kAntiWeightPort});
        if (boundary->AddInput(ctx.weightInputIndex, std::move(weightInput)) != ge::SUCCESS) {
            return nullptr;
        }
    }
    if (ctx.hasExternalBias) {
        ge::fusion::SubgraphInput biasInput;
        biasInput.AddInput({*ctx.nodeMatmul, kMmBiasPort});
        if (boundary->AddInput(ctx.biasInputIndex, std::move(biasInput)) != ge::SUCCESS) {
            return nullptr;
        }
    }
    auto outputNode = ctx.nodeQuant != nullptr ? ctx.nodeQuant :
                                                 (ctx.nodeAdd != nullptr ? ctx.nodeAdd : ctx.nodeMatmul);
    ge::fusion::SubgraphOutput output({*outputNode, 0});
    if (boundary->AddOutput(0, std::move(output)) != ge::SUCCESS) {
        return nullptr;
    }
    return boundary;
}

void AddNodeToRemove(std::vector<ge::GNodePtr>& nodes, const ge::GNodePtr& node)
{
    if (node != nullptr && std::find(nodes.begin(), nodes.end(), node) == nodes.end()) {
        nodes.emplace_back(node);
    }
}

void AddOrphanConstToRemove(std::vector<ge::GNodePtr>& nodes, const ge::GNodePtr& node)
{
    if (IsConst(node) && node->GetOutDataNodesAndPortIndexs(0).size() <= 1) {
        AddNodeToRemove(nodes, node);
    }
}

std::vector<ge::GNodePtr> CollectNodesToRemove(const FusionCtx& ctx)
{
    std::vector<ge::GNodePtr> nodes;
    AddNodeToRemove(nodes, ctx.nodeQuant);
    AddNodeToRemove(nodes, ctx.nodeAdd);
    AddNodeToRemove(nodes, ctx.nodeMatmul);
    AddNodeToRemove(nodes, ctx.nodeWeightTranspose);
    AddNodeToRemove(nodes, ctx.nodeXTranspose);
    AddNodeToRemove(nodes, ctx.nodeAntiQuant);
    AddOrphanConstToRemove(nodes, ctx.nodeScale);
    AddOrphanConstToRemove(nodes, ctx.nodeOffset);
    AddOrphanConstToRemove(nodes, ctx.nodeAddBias);
    if (ctx.nodeQuant != nullptr) {
        AddOrphanConstToRemove(nodes, ctx.nodeQuantScale);
        AddOrphanConstToRemove(nodes, ctx.nodeQuantOffset);
    }
    if (ctx.inlineWeight) {
        AddOrphanConstToRemove(nodes, ctx.nodeWeight);
    }
    return nodes;
}

ge::Status ReplaceSubgraph(const ge::fusion::SubgraphBoundary& boundary, const ge::Graph& replacement,
                           ge::CustomPassContext& passContext)
{
    passContext.SetPassName(ge::AscendString(kPassName));
#if GE_COMPILER_VERSION_NUM >= 90100000
    using ReplaceWithCtxFn = ge::Status (*)(const ge::fusion::SubgraphBoundary&, const ge::Graph&,
                                            ge::CustomPassContext&);
    auto replaceWithCtx = static_cast<ReplaceWithCtxFn>(&ge::fusion::SubgraphRewriter::Replace);
    if (replaceWithCtx != nullptr) {
        return replaceWithCtx(boundary, replacement, passContext);
    }
#endif
    return ge::fusion::SubgraphRewriter::Replace(boundary, replacement);
}

ge::Status ProcessSingleNodeFusion(const ge::GraphPtr& graph, const ge::GNode& matmulNode,
                                   ge::CustomPassContext& passContext)
{
    FusionCtx ctx;
    ctx.nodeMatmul = std::make_shared<ge::GNode>(matmulNode);
    if (!CheckShapesAndAttrs(ctx)) {
        return ge::GRAPH_NOT_CHANGED;
    }
    try {
        (void)MatchQuantFusion(ctx);
        if (!PrepareConstants(ctx)) {
            return ge::GRAPH_NOT_CHANGED;
        }
    } catch (const std::bad_alloc&) {
        OP_LOGW(kPassName, "Failed to allocate constant folding buffer");
        return ge::GRAPH_NOT_CHANGED;
    }

    auto replacement = BuildReplacementGraph(ctx);
    auto boundary = replacement == nullptr ? nullptr : BuildBoundary(ctx);
    if (replacement == nullptr || boundary == nullptr) {
        OP_LOGW(kPassName, "Failed to build replacement graph or boundary");
        return ge::GRAPH_NOT_CHANGED;
    }
    auto nodesToRemove = CollectNodesToRemove(ctx);
    std::unordered_set<std::string> removeNames;
    for (const auto& node : nodesToRemove) {
        if (node == nullptr) {
            continue;
        }
        ge::AscendString nodeName;
        if (node->GetName(nodeName) == ge::GRAPH_SUCCESS && nodeName.GetString() != nullptr) {
            removeNames.insert(std::string(nodeName.GetString()));
        }
    }
    if (ReplaceSubgraph(*boundary, *replacement, passContext) != ge::SUCCESS) {
        OP_LOGW(kPassName, "SubgraphRewriter::Replace failed");
        return ge::GRAPH_NOT_CHANGED;
    }
    if (!removeNames.empty()) {
        for (auto& node : graph->GetDirectNode()) {
            ge::AscendString nodeName;
            if (node.GetName(nodeName) != ge::GRAPH_SUCCESS || nodeName.GetString() == nullptr ||
                removeNames.count(std::string(nodeName.GetString())) == 0U) {
                continue;
            }
            if (graph->RemoveNode(node) != ge::GRAPH_SUCCESS) {
                OP_LOGW(kPassName, "Failed to remove old node after fusion");
            }
        }
    }
    OP_LOGI(kPassName, "Fused AscendAntiQuantV2 and MatMul into WeightQuantBatchMatmulV2");
    return ge::SUCCESS;
}
} // namespace

ge::Status MatmulToWeightQuantBatchMatmulFusion::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    if (!CheckPlatform() || graph == nullptr || !graph->IsValid()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    std::vector<ge::GNode> matmulNodes;
    for (auto& node : graph->GetDirectNode()) {
        ge::AscendString type;
        if (node.GetType(type) == ge::GRAPH_SUCCESS && IsMatMulType(type)) {
            matmulNodes.emplace_back(node);
        }
    }
    bool changed = false;
    for (auto& node : matmulNodes) {
        const auto status = ProcessSingleNodeFusion(graph, node, passContext);
        if (status == ge::SUCCESS) {
            changed = true;
        } else if (status != ge::GRAPH_NOT_CHANGED) {
            return status;
        }
    }
    return changed ? ge::SUCCESS : ge::GRAPH_NOT_CHANGED;
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(MatmulToWeightQuantBatchMatmulFusion).Stage(ge::CustomPassStage::kAfterOriginGraphOptimize);
#endif

} // namespace ops
