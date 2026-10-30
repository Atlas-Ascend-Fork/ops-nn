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
 * \file fused_weight_quant_mat_mul_fusion_pass.cpp
 * \brief FusedWeightQuantMatMulFusionPass
 *
 * Fusion pattern（子模式 a，y FP16）: fuse WeightQuantBatchMatmulV2 + Relu into FusedWeightQuantMatMul.
 *
 *   x  weight  antiquant_scale [bias]
 *    \    |         |             /
 *     WeightQuantBatchMatmulV2 ──┘
 *               │
 *             Relu ──> (唯一 consumer)
 * ─────────────────────────────────>
 *   x  weight  antiquant_scale [bias]
 *    \    |         |             /
 *     FusedWeightQuantMatMul (fused_op_type="relu")
 *               │
 *          y (fp16)
 *
 * Fusion pattern（子模式 b，y INT8 输出量化）:
 * fuse WeightQuantBatchMatmulV2 + Relu + AscendQuant/AscendQuantV2。
 *
 *   x  weight  antiquant_scale(Const) [bias]
 *    \    |         |             /
 *     WeightQuantBatchMatmulV2(y fp16) ──┘
 *               │
 *             Relu ──> (唯一 consumer)
 *               │
 *           AscendQuant[V2](y int8) ── scale [offset]
 * ─────────────────────────────────>
 *   x  weight  antiquant_scale'(Const, 合并编码) [bias]
 *    \    |         |             /
 *     FusedWeightQuantMatMul (fused_op_type="relu")
 *               │
 *          y (int8)
 *
 * 输出量化由 pass 常量折叠承载（底座 wqbmm_v2 无独立 quant 输入通路，y=INT8 由
 * antiquant_scale 的 uint64 合并编码承载）：
 *   combined[i] = encode(decode(antiquant_scale[i]) * s_q)
 *   decode/encode 对应 fixpipe deq scale 编码（fp32 位模式截断高 19 位，
 *   对齐 cmct/kernel_qbmm_cube.h 的 DEQ_SCALE_MUL）。
 *   非对称量化：offset 为量化码域加法项，不参与乘法合并，独立编入
 *   uint64 高位（bits 37-44 值 + bit45 符号 + bit46 signed 标志，对齐老代
 *   MatmulQuantToFixpipeFusionPass 的复合编码先例）；对称量化（offset 为 0）保持
 *   高位全零的既有编码不变。
 *
 * 保守准入（图模式无 fallback，不满足任一条件放弃融合，Relu 保持独立节点）：
 *   1. 平台门槛：支持 mmad s8s4
 *   2. WQBMMV2：x FP16 / weight INT8 且 ND / y FP16（INT8 输出场景不融合——
 *      relu(quant(mm)) 与 quant(relu(mm)) 语义漂移）
 *   3. WQBMMV2 可选输入 antiquant_offset/quant_scale/quant_offset/pertoken_scale 均未接入
 *      （底座 checker 本就要求为空，pass 预判做双保险；FE 节点输入布局与注册布局
 *      不一致，端口按 _input_name_key 名字映射解析）
 *   4. antiquant_scale 为 UINT64 且 PerChannel（group_size == 0，shape (n)/(1,n)/(n,1)）
 *   5. bias 未接入或 dtype ∈ {FP16, INT32}
 *   6. Relu 是 WQBMMV2 输出的唯一 consumer
 *   7. 透传属性 transpose_x/transpose_weight/antiquant_group_size/dtype，忽略 inner_precise
 *      （融合算子固定高精度模式），fused_op_type 固定 "relu"；子模式 b 的 dtype 设为 INT8
 *   8. 子模式 b 追加（任一不满足 → fallback 子模式 a，AscendQuant 保持独立节点）：
 *      a) Relu 的唯一 consumer 为 AscendQuant/AscendQuantV2 且其输出 dtype 为 INT8；
 *      b) sqrt_mode 为 false、round_mode 为默认 round（其他舍入模式与 fixpipe REQ8
 *         舍入行为不一致，不融合）；
 *      c) offset（非对称量化）：V1 offset 属性 / V2 offset Const 输入，
 *         须为整数零点且值域 [-128, 127]（编入 uint64 高位，见上方折叠说明；per-tensor
 *         标量/(1,) 或 per-channel (n,) 与 scale 同款广播）；非整数/超值域/非 Const →
 *         fallback a；对称量化（offset 为 0/未接入）编码与增补前完全一致；
 *      d) AscendQuant(V1) scale 属性为正有限值；AscendQuantV2 scale 输入为 Const
 *         （FP32 或 UINT64 编码，shape (1,) per-tensor 或 (n,) per-channel）；
 *      e) WQBMMV2 的 antiquant_scale 为 Const（uint64 (n)）——折叠前提；
 *      f) 折叠结果有限（s_q > 0，乘法结果无溢出）
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "log/log.h"
#include "securec.h"
#include "platform/platform_info.h"
#include "version/ge-compiler_version.h"
// weight_quant_fusion_utils.h 声明 ge::fusion::SubgraphRewriter 的弱符号兼容版本
// （3 参 Replace 运行时无强定义时为空，调用方回退 2 参版本），为本文件依赖的实际来源，
// 勿按未使用头清理；亦不可与 ge/fusion/graph_rewriter.h 同时包含（类重定义）
#include "matmul/common/op_graph/fusion_pass/weight_quant_fusion_utils.h"
#include "matmul/fused_weight_quant_mat_mul/common/fused_weight_quant_mat_mul_constants.h"
#include "fused_weight_quant_mat_mul_fusion_pass.h"

namespace ops {
using ge::fusion::GraphUniqPtr;
using ge::fusion::SubgraphBoundary;
using ge::fusion::SubgraphInput;
using ge::fusion::SubgraphOutput;
using ge::fusion::SubgraphRewriter;

static const char* kFusedOpType = "FusedWeightQuantMatMulFusionPass";
static const char* kOpTypeRelu = "Relu";
static const char* kOpTypeWqbmmV2 = "WeightQuantBatchMatmulV2";
static const char* kOpTypeAscendQuant = "AscendQuant";
static const char* kOpTypeAscendQuantV2 = "AscendQuantV2";
static const char* kOpTypeConst = "Const";
static const char* kFusedOpName = "FusedWeightQuantMatMul";

// AscendQuantV2 输入布局（对齐 es_nn es_AscendQuantV2 原型）：x(0)/scale(1)/offset(2)
// （与上方 WQBMMV2 的 quant_scale(4)/quant_offset(5) 输入端口区分命名）
constexpr int64_t QUANT_NODE_SCALE_PORT = 1;
constexpr int64_t QUANT_NODE_OFFSET_PORT = 2;

// 输入索引与算子注册布局（ops-info/es_nn 原型，亦为融合目标算子 FusedWeightQuantMatMul 的布局）
// 严格对齐：x=0, weight=1, antiquant_scale=2, antiquant_offset=3, quant_scale=4, quant_offset=5, bias=6
// 注意 FE 生成的 WQBMMV2 节点为另一布局：x=0, weight=1, antiquant_scale=2, antiquant_offset=3,
// bias=4, pertoken_scale=5（无 quant_scale/quant_offset/bias=6 输入），节点属性 _input_name_key /
// _input_name_value 记录名字→端口映射，源节点可选输入端口须按名字映射解析（见 ResolveInputPorts）；
// BIAS_PORT 仅用于向融合目标算子接线
constexpr int64_t X_PORT = 0;
constexpr int64_t WEIGHT_PORT = 1;
constexpr int64_t ANTI_QUANT_SCALE_PORT = 2;
constexpr int64_t ANTI_QUANT_OFFSET_PORT = 3;
constexpr int64_t QUANT_SCALE_PORT = 4;
constexpr int64_t QUANT_OFFSET_PORT = 5;
constexpr int64_t BIAS_PORT = 6;

constexpr size_t kMinWeightDimNum = 2;
constexpr size_t kPerTensorParamSize = 1;
constexpr size_t kExpectedConsumerNum = 1;

static bool CheckPlatform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) !=
        ge::SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to get platform info, skip fusion pass");
        return false;
    }
    const auto mmadIter = platformInfo.ai_core_intrinsic_dtype_map.find("Intrinsic_mmad");
    const bool supportS8S4 = mmadIter != platformInfo.ai_core_intrinsic_dtype_map.end() &&
                             std::find(mmadIter->second.begin(), mmadIter->second.end(), "s8s4") !=
                                 mmadIter->second.end();
    if (!supportS8S4) {
        OP_LOGD(kFusedOpType, "The fusion pass requires mmad s8s4 support");
    }
    return supportS8S4;
}

static bool GetNodeInputDesc(const ge::GNode& node, int64_t port, ge::TensorDesc& desc)
{
    return node.GetInputDesc(port, desc) == ge::GRAPH_SUCCESS;
}

// 可选输入未接入判定：无上游数据节点即视为未接入
static bool IsOptionalInputAbsent(const ge::GNode& node, int64_t port)
{
    auto peer = node.GetInDataNodesAndPortIndexs(port);
    return peer.first == nullptr;
}

// 解析源节点（WQBMMV2）可选输入端口。FE 生成的节点输入布局（x=0, weight=1, antiquant_scale=2,
// antiquant_offset=3, bias=4, pertoken_scale=5，共 6 输入，无 quant 输入）与算子注册布局
// （quant_scale=4, quant_offset=5, bias=6，共 7 输入，无 pertoken_scale）不一致，判别顺序：
// 1)节点带 _input_name_key / _input_name_value 名字映射属性时按名字解析（FE 后续阶段写入，
//   权威）——映射未列出的输入视为该布局不存在，端口置 -1；
// 2)否则按节点 IR 名字解析（GetInputIndexByName）：es_nn/注册布局节点（UT 构图、es 图）按名
//   可得 bias=6 / quant_scale=4 / quant_offset=5，pertoken_scale 不存在；bias 端口叠加物理
//   连接回退链（IR 名端口 → 端口 6 → 端口 4）：若 IR 名解析指向无连接边的端口（如 FE 布局
//   节点按注册 IR 误得 bias=6），依次探测端口 6/4 上的实际连接边，保证 FE 布局 bias（端口 4）
//   不丢失；pertoken_scale 名字不可解析但端口 5 存在连接边时按 pertoken 处理（端口 5 上的
//   连接边无论语义为 pertoken_scale 还是 quant_offset 均应拒绝融合，保守无害）；
// 3)名字解析完全不可用（无任何 IR 名）时按输入数量兜底：GetInputsSize < 7 视为 FE 布局。
//   注意 GetInputsSize 语义为已连接输入数而非 IR 槽位数（实测 7 槽位仅连 4 输入的 es 布局
//   节点返回 4，按数量判别会误判 FE 布局致 bias 边丢失），故仅作兜底。
// 注意 GetInputIndexByName 为非 const 接口（内部含名字缓存），形参须为非 const 引用
static void ResolveInputPorts(ge::GNode& node, int64_t& biasPort, int64_t& pertokenPort, int64_t& quantScalePort,
                              int64_t& quantOffsetPort)
{
    std::vector<ge::AscendString> names;
    std::vector<int64_t> ports;
    const bool hasNameMap = node.GetAttr(ge::AscendString("_input_name_key"), names) == ge::GRAPH_SUCCESS &&
                            node.GetAttr(ge::AscendString("_input_name_value"), ports) == ge::GRAPH_SUCCESS &&
                            !names.empty() && names.size() == ports.size();
    if (!hasNameMap) {
        constexpr int64_t kFeLayoutBiasPort = 4;
        constexpr int64_t kFeLayoutPertokenPort = 5;
        int32_t biasIdx = -1;
        int32_t pertokenIdx = -1;
        int32_t quantScaleIdx = -1;
        int32_t quantOffsetIdx = -1;
        (void)node.GetInputIndexByName(ge::AscendString("bias"), biasIdx);
        (void)node.GetInputIndexByName(ge::AscendString("pertoken_scale"), pertokenIdx);
        (void)node.GetInputIndexByName(ge::AscendString("quant_scale"), quantScaleIdx);
        (void)node.GetInputIndexByName(ge::AscendString("quant_offset"), quantOffsetIdx);
        if (biasIdx >= 0 || pertokenIdx >= 0 || quantScaleIdx >= 0 || quantOffsetIdx >= 0) {
            biasPort = (biasIdx >= 0) ? biasIdx : -1;
            pertokenPort = (pertokenIdx >= 0) ? pertokenIdx : -1;
            quantScalePort = (quantScaleIdx >= 0) ? quantScaleIdx : -1;
            quantOffsetPort = (quantOffsetIdx >= 0) ? quantOffsetIdx : -1;
            // bias 物理回退链：IR 名端口无连接边时依次探测端口 6（注册布局）/ 4（FE 布局）
            if (biasPort < 0 || node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(biasPort)).first == nullptr) {
                if (node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(BIAS_PORT)).first != nullptr) {
                    biasPort = BIAS_PORT;
                } else if (node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(kFeLayoutBiasPort)).first != nullptr) {
                    biasPort = kFeLayoutBiasPort;
                }
            }
            // pertoken 兜底：名字不可解析但端口 5 有连接边（pertoken_scale/quant_offset 两种
            // 语义均须拒绝），交给可选输入缺位校验保守拒绝
            if (pertokenPort < 0 &&
                node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(kFeLayoutPertokenPort)).first != nullptr) {
                pertokenPort = kFeLayoutPertokenPort;
            }
            return;
        }
        if (node.GetInputsSize() < static_cast<size_t>(BIAS_PORT + 1)) {
            biasPort = kFeLayoutBiasPort;
            pertokenPort = kFeLayoutPertokenPort;
            quantScalePort = -1;
            quantOffsetPort = -1;
        } else {
            biasPort = BIAS_PORT;
            pertokenPort = -1;
            quantScalePort = QUANT_SCALE_PORT;
            quantOffsetPort = QUANT_OFFSET_PORT;
        }
        return;
    }
    biasPort = -1;
    pertokenPort = -1;
    quantScalePort = -1;
    quantOffsetPort = -1;
    for (size_t i = 0; i < names.size(); i++) {
        const std::string name(names[i].GetString());
        if (name == "bias") {
            biasPort = ports[i];
        } else if (name == "pertoken_scale") {
            pertokenPort = ports[i];
        } else if (name == "quant_scale") {
            quantScalePort = ports[i];
        } else if (name == "quant_offset") {
            quantOffsetPort = ports[i];
        }
    }
}

struct FusedCtx {
    ge::GNodePtr nodeWqbmm;
    ge::GNodePtr nodeRelu;
    bool transX = false;
    bool transWeight = false;
    int64_t groupSize = 0;
    int64_t dtype = -1;
    // 源节点（WQBMMV2）的 bias 输入端口：FE 节点布局为 4，注册/es_nn 布局为 6，-1 表示无该输入，
    // 由 _input_name_key 名字映射解析（见 ResolveInputPorts）
    int64_t biasPort = BIAS_PORT;
    // 子模式 b（y INT8 输出量化）：Relu 后接的量化节点与折叠结果；nodeQuant 为空即子模式 a
    ge::GNodePtr nodeQuant;
    int64_t nDim = 0;
    std::vector<uint64_t> foldedScale;
};

static bool CheckWqbmmDtype(const ge::GNode& node)
{
    ge::TensorDesc xDesc;
    ge::TensorDesc wDesc;
    ge::TensorDesc scaleDesc;
    ge::TensorDesc yDesc;
    if (!GetNodeInputDesc(node, X_PORT, xDesc) || !GetNodeInputDesc(node, WEIGHT_PORT, wDesc) ||
        !GetNodeInputDesc(node, ANTI_QUANT_SCALE_PORT, scaleDesc)) {
        OP_LOGD(kFusedOpType, "Failed to get input desc of WQBMMV2 node");
        return false;
    }
    if (node.GetOutputDesc(0, yDesc) != ge::GRAPH_SUCCESS) {
        OP_LOGD(kFusedOpType, "Failed to get output desc of WQBMMV2 node");
        return false;
    }
    if (xDesc.GetDataType() != ge::DT_FLOAT16) {
        OP_LOGD(kFusedOpType, "The dtype of x is %d, must be float16", static_cast<int32_t>(xDesc.GetDataType()));
        return false;
    }
    if (wDesc.GetDataType() != ge::DT_INT8) {
        OP_LOGD(kFusedOpType, "The dtype of weight is %d, must be int8", static_cast<int32_t>(wDesc.GetDataType()));
        return false;
    }
    if (scaleDesc.GetDataType() != ge::DT_UINT64) {
        OP_LOGD(kFusedOpType, "The dtype of antiquant_scale is %d, must be uint64",
                static_cast<int32_t>(scaleDesc.GetDataType()));
        return false;
    }
    if (yDesc.GetDataType() != ge::DT_FLOAT16) {
        OP_LOGD(kFusedOpType, "The dtype of y is %d, must be float16 (int8 output is not fused)",
                static_cast<int32_t>(yDesc.GetDataType()));
        return false;
    }
    auto wFormat = wDesc.GetFormat();
    if (wFormat != ge::FORMAT_ND && wFormat != ge::FORMAT_NCHW) {
        OP_LOGD(kFusedOpType, "The format of weight is %d, must be ND", static_cast<int32_t>(wFormat));
        return false;
    }
    return true;
}

// antiquant_scale PerChannel 校验：group_size == 0 且 scale shape 为 (n)/(1,n)/(n,1)
static bool CheckAntiQuantScale(const ge::GNode& node, bool transWeight, int64_t& nDim)
{
    // antiquant_group_size 为 optional 属性（缺省 0，即 per-channel），FE 节点不物化带默认值的属性
    int64_t groupSize = 0;
    (void)node.GetAttr(ge::AscendString("antiquant_group_size"), groupSize);
    if (groupSize != 0) {
        OP_LOGD(kFusedOpType, "antiquant_group_size is %ld, only per-channel (0) is fused", groupSize);
        return false;
    }
    ge::TensorDesc wDesc;
    if (!GetNodeInputDesc(node, WEIGHT_PORT, wDesc)) {
        return false;
    }
    auto wShape = wDesc.GetOriginShape().GetDims();
    if (wShape.size() < kMinWeightDimNum) {
        OP_LOGD(kFusedOpType, "The dim num of weight is %zu, must be >= 2", wShape.size());
        return false;
    }
    nDim = transWeight ? wShape[wShape.size() - kMinWeightDimNum] : wShape.back();
    ge::TensorDesc scaleDesc;
    if (!GetNodeInputDesc(node, ANTI_QUANT_SCALE_PORT, scaleDesc)) {
        return false;
    }
    if (nDim <= 0) {
        return true;
    }
    const auto scaleShape = scaleDesc.GetOriginShape().GetDims();
    const bool vectorMatch = scaleShape == std::vector<int64_t>{nDim};
    const bool matrixMatch = transWeight ? scaleShape == std::vector<int64_t>{nDim, 1} :
                                           scaleShape == std::vector<int64_t>{1, nDim};
    if (!vectorMatch && !matrixMatch) {
        OP_LOGD(kFusedOpType, "antiquant_scale dim num is %zu, must be per-channel (n)/(1,n)/(n,1) with n[%ld]",
                scaleShape.size(), nDim);
        return false;
    }
    return true;
}

// ===== 子模式 b（y INT8 输出量化）：常量折叠辅助 =====

// fixpipe deq scale 编码：fp32 位模式截断高 19 位（低 13 位尾数清零，低 32 位存放），
// 对齐 cmct/kernel_qbmm_cube.h:40,193 的 DEQ_SCALE_MUL 处理（"fixpipe只能取高19位"）
constexpr uint32_t DEQ_SCALE_MASK = 0xFFFFE000U;
constexpr uint64_t DEQ_SCALE_HIGH_BITS_SHIFT = sizeof(uint32_t) * 8UL;

static float DecodeDeqScale(uint64_t encoded)
{
    // 截断值仍是合法 fp32（低尾数位为零），按位解释即无损解码
    uint32_t bits = static_cast<uint32_t>(encoded);
    float value = 0.0F;
    errno_t ret = memcpy_s(&value, sizeof(value), &bits, sizeof(bits));
    if (ret != EOK) {
        OP_LOGE(kFusedOpType, "Failed to decode deq scale, memcpy_s ret=%d", ret);
        return 0.0F;
    }
    return value;
}

static uint64_t EncodeDeqScale(float value)
{
    uint32_t bits = 0U;
    errno_t ret = memcpy_s(&bits, sizeof(bits), &value, sizeof(value));
    if (ret != EOK) {
        OP_LOGE(kFusedOpType, "Failed to encode deq scale, memcpy_s ret=%d", ret);
        return 0UL;
    }
    return static_cast<uint64_t>(bits & DEQ_SCALE_MASK);
}

// 非对称量化 offset 高位编码（对齐老代 MatmulQuantToFixpipeFusionPass 的复合 uint64 编码契约，
// 位号以 mask 为准，老 pass 源码注释位号有误）：bits 0-31 为截断 fp32 scale（DEQ_SCALE_MASK 契约），
// offset 为量化码域加法项，编码进 bits 37-44（值，整型补码）+ bit 45（符号，负 offset 置位）
// + bit 46（signed 标志，恒置位）。offset 编码与 scale 折叠方向无关（加法项独立于乘除约定）。
constexpr uint64_t kQuantOffsetValueShift = 37UL;
constexpr uint64_t kQuantOffsetValueMask = 0x00001FE000000000UL; // bits 37-44
constexpr uint64_t kQuantOffsetSignMask = 0x0000200000000000UL;  // bit 45
constexpr uint64_t kQuantSignedFlagMask = 0x0000400000000000UL;  // bit 46
// offset 值域：int8 零点范围（覆盖 INT8 非对称量化典型场景）
constexpr float kQuantOffsetMin = -128.0F;
constexpr float kQuantOffsetMax = 127.0F;

// offset 合法性：有限、整数（硬件 offset 字段为整型码偏移，非整数需 round，引入 ≤0.5 code
// 误差，按保守准入原则拒绝回退 a）、int8 零点值域内
static bool CheckQuantOffsetValue(float offset)
{
    if (!std::isfinite(offset) || offset != std::round(offset) || offset < kQuantOffsetMin ||
        offset > kQuantOffsetMax) {
        OP_LOGD(kFusedOpType, "quant offset %f is not an integer within [%d, %d], skip quant fusion", offset,
                static_cast<int32_t>(kQuantOffsetMin), static_cast<int32_t>(kQuantOffsetMax));
        return false;
    }
    return true;
}

// offset 编码进 uint64 高位（低 32 位为合并 scale）：值取整型低 8 bit（负数为补码）左移 37，
// 负数另置 bit45 符号位，bit46 signed 标志恒置（对齐老代 pass 的 ProcessScaleElement 编码）。
// 对称量化（offset 为 0）保持高位全零的既有编码不变（已交付路径零回归风险）
static uint64_t EncodeDeqScaleWithOffset(float combined, float offset)
{
    uint64_t encoded = EncodeDeqScale(combined);
    if (offset == 0.0F) {
        return encoded;
    }
    auto offsetInt = static_cast<int64_t>(offset);
    encoded |= (static_cast<uint64_t>(offsetInt) << kQuantOffsetValueShift) & kQuantOffsetValueMask;
    if (offsetInt < 0) {
        encoded |= kQuantOffsetSignMask;
    }
    encoded |= kQuantSignedFlagMask;
    return encoded;
}

// 读取 Const 节点的 value tensor（对齐 anti_quant_matmul_fusion_pass 的 GetConstValue 前置条件）
static bool GetConstTensor(const ge::GNodePtr& node, ge::Tensor& tensor)
{
    if (node == nullptr) {
        return false;
    }
    ge::AscendString type;
    if (node->GetType(type) != ge::GRAPH_SUCCESS || std::string(type.GetString()) != kOpTypeConst) {
        return false;
    }
    if (node->GetAttr(ge::AscendString("value"), tensor) != ge::GRAPH_SUCCESS) {
        OP_LOGD(kFusedOpType, "Failed to get const value attr");
        return false;
    }
    return tensor.GetData() != nullptr;
}

// 读取 WQBMMV2 的 antiquant_scale Const 数据（uint64，size == n）
static bool ReadAntiQuantScaleConst(const ge::GNode& wqbmmNode, int64_t nDim, std::vector<uint64_t>& data)
{
    if (nDim <= 0) {
        OP_LOGD(kFusedOpType, "antiquant_scale dimension must be positive, but is [%ld]", nDim);
        return false;
    }
    auto peer = wqbmmNode.GetInDataNodesAndPortIndexs(ANTI_QUANT_SCALE_PORT);
    ge::Tensor tensor;
    if (!GetConstTensor(peer.first, tensor)) {
        OP_LOGD(kFusedOpType, "antiquant_scale is not a Const node, skip quant fusion");
        return false;
    }
    if (tensor.GetTensorDesc().GetDataType() != ge::DT_UINT64) {
        OP_LOGD(kFusedOpType, "antiquant_scale const dtype is not uint64, skip quant fusion");
        return false;
    }
    const size_t elementNum = static_cast<size_t>(nDim);
    const size_t dataSize = sizeof(uint64_t) * elementNum;
    if (tensor.GetSize() < dataSize) {
        OP_LOGD(kFusedOpType, "antiquant_scale const size %zu less than n[%ld]", tensor.GetSize(), nDim);
        return false;
    }
    data.resize(elementNum);
    errno_t ret = memcpy_s(data.data(), dataSize, tensor.GetData(), dataSize);
    if (ret != EOK) {
        OP_LOGE(kFusedOpType, "Failed to read antiquant_scale const, memcpy_s ret=%d", ret);
        return false;
    }
    return true;
}

// 量化节点数值语义校验：sqrt_mode 关闭 + round_mode 为默认 round（V1/V2 同名属性）
static bool CheckQuantSemantics(const ge::GNode& node)
{
    bool sqrtMode = false;
    (void)node.GetAttr(ge::AscendString("sqrt_mode"), sqrtMode); // 缺省 false
    if (sqrtMode) {
        OP_LOGD(kFusedOpType, "sqrt_mode is true, skip quant fusion");
        return false;
    }
    ge::AscendString roundMode;
    if (node.GetAttr(ge::AscendString("round_mode"), roundMode) == ge::GRAPH_SUCCESS) {
        std::string mode(roundMode.GetString());
        std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
        if (!mode.empty() && mode != "round") {
            OP_LOGD(kFusedOpType, "round_mode is [%s], only default round is fused", roundMode.GetString());
            return false;
        }
    }
    return true;
}

// 读取 Const 的全量 fp32 数值（DT_FLOAT 直读 / DT_UINT64 按 deq scale 编码解码）
static bool ReadConstFloats(const ge::GNodePtr& node, std::vector<float>& values)
{
    ge::Tensor tensor;
    if (!GetConstTensor(node, tensor)) {
        return false;
    }
    auto dtype = tensor.GetTensorDesc().GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_UINT64) {
        OP_LOGD(kFusedOpType, "const dtype %d is not float/uint64", static_cast<int32_t>(dtype));
        return false;
    }
    size_t elemSize = (dtype == ge::DT_UINT64) ? sizeof(uint64_t) : sizeof(float);
    if (tensor.GetSize() == 0UL || tensor.GetSize() % elemSize != 0UL) {
        OP_LOGD(kFusedOpType, "const tensor size %zu is not aligned to elem size %zu", tensor.GetSize(), elemSize);
        return false;
    }
    size_t elemNum = tensor.GetSize() / elemSize;
    if (dtype == ge::DT_FLOAT) {
        const float* ptr = reinterpret_cast<const float*>(tensor.GetData());
        values.assign(ptr, ptr + elemNum);
    } else {
        const uint64_t* ptr = reinterpret_cast<const uint64_t*>(tensor.GetData());
        values.resize(elemNum);
        for (size_t i = 0; i < elemNum; i++) {
            values[i] = DecodeDeqScale(ptr[i]);
        }
    }
    return true;
}

// 折叠：combined[i] = s_w[i] * s_q[i]（per-tensor s_q 广播）；offset（量化码域加法项）不参与
// 乘法合并，独立编入 uint64 高位（per-tensor 广播同 s_q）。
// 上游 antiquant_scale 高位（bits 32-63）须为零：已带高位编码（如老代 pass 产物）的表项
// 不做二次折叠（重编码会静默丢弃上游高位字段），保守拒绝
static bool FoldQuantScale(const std::vector<uint64_t>& antiScale, const std::vector<float>& quantScale,
                           const std::vector<float>& quantOffset, std::vector<uint64_t>& folded)
{
    folded.assign(antiScale.size(), 0UL);
    for (size_t i = 0; i < antiScale.size(); i++) {
        if ((antiScale[i] >> DEQ_SCALE_HIGH_BITS_SHIFT) != 0UL) {
            OP_LOGD(kFusedOpType, "antiquant_scale[%zu] has non-zero high bits, skip quant fusion", i);
            return false;
        }
        float sW = DecodeDeqScale(antiScale[i]);
        float sQ = quantScale[i < quantScale.size() ? i : 0UL];
        float oQ = quantOffset.empty() ? 0.0F : quantOffset[i < quantOffset.size() ? i : 0UL];
        if (!std::isfinite(sW) || !std::isfinite(sQ) || sQ <= 0.0F) {
            OP_LOGD(kFusedOpType, "invalid scale value (sW=%f, sQ=%f), skip quant fusion", sW, sQ);
            return false;
        }
        float combined = sW * sQ;
        if (!std::isfinite(combined)) {
            OP_LOGD(kFusedOpType, "folded scale is not finite, skip quant fusion");
            return false;
        }
        folded[i] = EncodeDeqScaleWithOffset(combined, oQ);
    }
    return true;
}

// V1（AscendQuant）：scale/offset 为标量属性（offset 支持非对称：整数零点编入高位，
// 非整数/超值域回退 a）。scale/offset 均为 REQUIRED 属性（见 ascend_quant_def），读取失败
// 即属性缺失或类型异常，按对称量化（offset=0）假定继续会静默丢弃非零 offset，故保守回退子模式 a
static bool ReadV1QuantParams(const ge::GNode& quantNode, std::vector<float>& quantScale,
                              std::vector<float>& quantOffset)
{
    float scale = 0.0F;
    float offset = 0.0F;
    if (quantNode.GetAttr(ge::AscendString("scale"), scale) != ge::GRAPH_SUCCESS ||
        quantNode.GetAttr(ge::AscendString("offset"), offset) != ge::GRAPH_SUCCESS) {
        OP_LOGD(kFusedOpType, "Failed to get scale/offset attr of AscendQuant, skip quant fusion");
        return false;
    }
    if (!CheckQuantOffsetValue(offset)) {
        return false;
    }
    quantScale = {scale};
    quantOffset = {offset};
    return true;
}

// V2（AscendQuantV2）：scale 为输入（须 Const，FP32/UINT64 编码，(1,) per-tensor 或 (n,) per-channel）；
// offset（非对称量化）：未接入（对称）或 Const（FP32/UINT64 编码，(1,) 或 (n,) 与 scale 同款
// 广播规则），整数零点编入 uint64 高位
static bool ReadV2QuantParams(const ge::GNode& quantNode, int64_t nDim, std::vector<float>& quantScale,
                              std::vector<float>& quantOffset)
{
    auto scalePeer = quantNode.GetInDataNodesAndPortIndexs(QUANT_NODE_SCALE_PORT);
    if (!ReadConstFloats(scalePeer.first, quantScale)) {
        OP_LOGD(kFusedOpType, "quant scale is not a readable Const, skip quant fusion");
        return false;
    }
    if (quantScale.empty() ||
        (quantScale.size() != kPerTensorParamSize && static_cast<int64_t>(quantScale.size()) != nDim)) {
        OP_LOGD(kFusedOpType, "quant scale size %zu is neither 1 nor n[%ld]", quantScale.size(), nDim);
        return false;
    }
    auto offsetPeer = quantNode.GetInDataNodesAndPortIndexs(QUANT_NODE_OFFSET_PORT);
    if (offsetPeer.first != nullptr) {
        if (!ReadConstFloats(offsetPeer.first, quantOffset)) {
            OP_LOGD(kFusedOpType, "quant offset is not a readable Const, skip quant fusion");
            return false;
        }
        if (quantOffset.empty() ||
            (quantOffset.size() != kPerTensorParamSize && static_cast<int64_t>(quantOffset.size()) != nDim)) {
            OP_LOGD(kFusedOpType, "quant offset size %zu is neither 1 nor n[%ld]", quantOffset.size(), nDim);
            return false;
        }
        for (auto v : quantOffset) {
            if (!CheckQuantOffsetValue(v)) {
                return false;
            }
        }
    }
    return true;
}

// 子模式 b 匹配：Relu 唯一 consumer 为 AscendQuant/AscendQuantV2 且输出 INT8 时做常量折叠。
// 任一条件不满足返回 false（调用方 fallback 子模式 a）
static bool MatchQuantFusion(FusedCtx& ctx)
{
    auto consumers = ctx.nodeRelu->GetOutDataNodesAndPortIndexs(0);
    if (consumers.size() != kExpectedConsumerNum || consumers.front().first == nullptr) {
        return false;
    }
    ge::GNodePtr quantNode = consumers.front().first;
    ge::AscendString type;
    if (quantNode->GetType(type) != ge::GRAPH_SUCCESS) {
        return false;
    }
    std::string typeStr(type.GetString());
    if (typeStr != kOpTypeAscendQuant && typeStr != kOpTypeAscendQuantV2) {
        return false;
    }
    ge::TensorDesc quantOutDesc;
    if (quantNode->GetOutputDesc(0, quantOutDesc) != ge::GRAPH_SUCCESS || quantOutDesc.GetDataType() != ge::DT_INT8) {
        OP_LOGD(kFusedOpType, "The output dtype of quant node is not int8, skip quant fusion");
        return false;
    }
    if (!CheckQuantSemantics(*quantNode)) {
        return false;
    }
    std::vector<float> quantScale;
    std::vector<float> quantOffset;
    if (typeStr == kOpTypeAscendQuant) {
        if (!ReadV1QuantParams(*quantNode, quantScale, quantOffset)) {
            return false;
        }
    } else if (!ReadV2QuantParams(*quantNode, ctx.nDim, quantScale, quantOffset)) {
        return false;
    }
    // antiquant_scale 须为 Const（uint64 (n)），折叠为合并编码
    std::vector<uint64_t> antiScale;
    if (!ReadAntiQuantScaleConst(*ctx.nodeWqbmm, ctx.nDim, antiScale)) {
        return false;
    }
    if (!FoldQuantScale(antiScale, quantScale, quantOffset, ctx.foldedScale)) {
        return false;
    }
    ctx.nodeQuant = quantNode;
    return true;
}

static bool MatchAndValidate(const ge::GNode& wqbmmNode, FusedCtx& ctx)
{
    ctx.nodeWqbmm = std::make_shared<ge::GNode>(wqbmmNode);
    // Relu 须为 WQBMMV2 输出的唯一 consumer
    auto consumers = ctx.nodeWqbmm->GetOutDataNodesAndPortIndexs(0);
    if (consumers.size() != kExpectedConsumerNum || consumers.front().first == nullptr) {
        OP_LOGD(kFusedOpType, "The output of WQBMMV2 has no consumer or more than one consumer");
        return false;
    }
    ge::GNodePtr nextNode = consumers.front().first;
    ge::AscendString nextType;
    if (nextNode->GetType(nextType) != ge::GRAPH_SUCCESS) {
        return false;
    }
    if (std::string(nextType.GetString()) != kOpTypeRelu) {
        OP_LOGD(kFusedOpType, "The only consumer of WQBMMV2 is [%s], not Relu", nextType.GetString());
        return false;
    }
    ctx.nodeRelu = nextNode;

    if (!CheckWqbmmDtype(*ctx.nodeWqbmm)) {
        return false;
    }
    // 可选输入：antiquant_offset/quant_scale/quant_offset/pertoken_scale 必须未接入（端口按布局解析）
    int64_t pertokenPort = -1;
    int64_t quantScalePort = QUANT_SCALE_PORT;
    int64_t quantOffsetPort = QUANT_OFFSET_PORT;
    ResolveInputPorts(*ctx.nodeWqbmm, ctx.biasPort, pertokenPort, quantScalePort, quantOffsetPort);
    if (!IsOptionalInputAbsent(*ctx.nodeWqbmm, ANTI_QUANT_OFFSET_PORT) ||
        (quantScalePort >= 0 && !IsOptionalInputAbsent(*ctx.nodeWqbmm, quantScalePort)) ||
        (quantOffsetPort >= 0 && !IsOptionalInputAbsent(*ctx.nodeWqbmm, quantOffsetPort)) ||
        (pertokenPort >= 0 && !IsOptionalInputAbsent(*ctx.nodeWqbmm, pertokenPort))) {
        OP_LOGD(kFusedOpType, "antiquant_offset/quant_scale/quant_offset/pertoken_scale must be absent for fusion");
        return false;
    }
    // bias：未接入或 FP16/INT32（源端口按布局解析，FE 节点为 4，注册/es_nn 布局为 6）
    ge::GNodePtr biasNode = nullptr;
    if (ctx.biasPort >= 0) {
        biasNode = ctx.nodeWqbmm->GetInDataNodesAndPortIndexs(static_cast<int32_t>(ctx.biasPort)).first;
    }
    if (biasNode != nullptr) {
        ge::TensorDesc biasDesc;
        if (!GetNodeInputDesc(*ctx.nodeWqbmm, ctx.biasPort, biasDesc)) {
            return false;
        }
        auto biasDtype = biasDesc.GetDataType();
        if (biasDtype != ge::DT_FLOAT16 && biasDtype != ge::DT_INT32) {
            OP_LOGD(kFusedOpType, "The dtype of bias is %d, must be float16/int32", static_cast<int32_t>(biasDtype));
            return false;
        }
    }
    // 透传属性：transpose_x/transpose_weight/antiquant_group_size 均为 optional 属性（ops-info 默认
    // false/false/0），FE 生成的节点不物化带默认值的属性，GetAttr 失败按默认值处理
    ctx.transX = false;
    ctx.transWeight = false;
    ctx.groupSize = 0;
    (void)ctx.nodeWqbmm->GetAttr(ge::AscendString("transpose_x"), ctx.transX);
    (void)ctx.nodeWqbmm->GetAttr(ge::AscendString("transpose_weight"), ctx.transWeight);
    (void)ctx.nodeWqbmm->GetAttr(ge::AscendString("antiquant_group_size"), ctx.groupSize);
    ctx.dtype = -1;
    (void)ctx.nodeWqbmm->GetAttr(ge::AscendString("dtype"), ctx.dtype); // dtype 属性缺省 -1
    if (!CheckAntiQuantScale(*ctx.nodeWqbmm, ctx.transWeight, ctx.nDim)) {
        return false;
    }
    return true;
}

static void SetTensorDesc(ge::TensorDesc& desc, const ge::DataType& dtype, const ge::Shape& shape,
                          const ge::Format& format)
{
    desc.SetDataType(dtype);
    desc.SetShape(shape);
    desc.SetFormat(format);
    desc.SetOriginShape(shape);
    desc.SetOriginFormat(format);
}

static bool SetProducerOutputDesc(ge::GNode* producer, const ge::DataType& dtype, const ge::Shape& shape,
                                  const ge::Format& format)
{
    if (producer == nullptr) {
        return false;
    }
    ge::TensorDesc desc;
    if (producer->GetOutputDesc(0, desc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    SetTensorDesc(desc, dtype, shape, format);
    return producer->UpdateOutputDesc(0, desc) == ge::GRAPH_SUCCESS;
}

static bool SetConsumerInputDesc(ge::GNode* consumer, int64_t port, const ge::DataType& dtype, const ge::Shape& shape,
                                 const ge::Format& format)
{
    if (consumer == nullptr) {
        return false;
    }
    ge::TensorDesc desc;
    if (consumer->GetInputDesc(port, desc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    SetTensorDesc(desc, dtype, shape, format);
    return consumer->UpdateInputDesc(port, desc) == ge::GRAPH_SUCCESS;
}

// 替换子图构建的源 desc 集合（es 建点仅连边不落 desc，收集一次、双侧补齐复用）
struct ReplacementDescs {
    ge::TensorDesc xDesc;
    ge::TensorDesc wDesc;
    ge::TensorDesc scaleDesc;
    ge::TensorDesc biasDesc;
    // 子模式 a 取 Relu 输出（FP16），子模式 b 取量化节点输出（INT8）——relu/量化均不改 shape
    ge::TensorDesc outDesc;
    bool hasBias = false;
};

// 收集源节点输入输出 desc（bias desc 一次获取，收敛原三处重复读取）
static bool CollectSourceDescs(const FusedCtx& ctx, ReplacementDescs& descs)
{
    if (ctx.nodeWqbmm->GetInputDesc(X_PORT, descs.xDesc) != ge::GRAPH_SUCCESS ||
        ctx.nodeWqbmm->GetInputDesc(WEIGHT_PORT, descs.wDesc) != ge::GRAPH_SUCCESS ||
        ctx.nodeWqbmm->GetInputDesc(ANTI_QUANT_SCALE_PORT, descs.scaleDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    if (ctx.nodeQuant != nullptr) {
        if (ctx.nodeQuant->GetOutputDesc(0, descs.outDesc) != ge::GRAPH_SUCCESS) {
            return false;
        }
    } else if (ctx.nodeRelu->GetOutputDesc(0, descs.outDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    descs.hasBias = ctx.biasPort >= 0 &&
                    ctx.nodeWqbmm->GetInDataNodesAndPortIndexs(static_cast<int32_t>(ctx.biasPort)).first != nullptr;
    return !descs.hasBias || ctx.nodeWqbmm->GetInputDesc(ctx.biasPort, descs.biasDesc) == ge::GRAPH_SUCCESS;
}

// 建融合节点并连边/置属性（无 es 生成接口，走 CompliantNodeBuilder 通用建点）。
// 属性：透传 transpose_x/transpose_weight/antiquant_group_size/dtype（忽略 inner_precise，
// 融合算子固定高精度模式），fused_op_type 固定 relu；子模式 b 的 dtype 设为 INT8
static bool BuildFusedNode(ge::Graph& graph, const FusedCtx& ctx, bool hasBias, const ge::es::EsTensorHolder& rX,
                           const ge::es::EsTensorHolder& rW, const ge::es::EsTensorHolder& rScale,
                           const ge::es::EsTensorHolder& rBias, ge::GNode& fusedNode)
{
    fusedNode = ge::es::CompliantNodeBuilder(&graph)
                    .OpType(kFusedOpName)
                    .Name(kFusedOpName)
                    .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"weight", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"antiquant_scale", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"antiquant_offset", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                  {"quant_scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                  {"quant_offset", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                  {"bias", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                    .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .Build();
    if (ge::es::AddEdgeAndUpdatePeerDesc(graph, *rX.GetProducer(), rX.GetProducerOutIndex(), fusedNode, X_PORT) !=
            ge::GRAPH_SUCCESS ||
        ge::es::AddEdgeAndUpdatePeerDesc(graph, *rW.GetProducer(), rW.GetProducerOutIndex(), fusedNode, WEIGHT_PORT) !=
            ge::GRAPH_SUCCESS ||
        ge::es::AddEdgeAndUpdatePeerDesc(graph, *rScale.GetProducer(), rScale.GetProducerOutIndex(), fusedNode,
                                         ANTI_QUANT_SCALE_PORT) != ge::GRAPH_SUCCESS ||
        (hasBias && ge::es::AddEdgeAndUpdatePeerDesc(graph, *rBias.GetProducer(), rBias.GetProducerOutIndex(),
                                                     fusedNode, BIAS_PORT) != ge::GRAPH_SUCCESS)) {
        return false;
    }
    bool transX = ctx.transX;
    bool transWeight = ctx.transWeight;
    int64_t groupSize = ctx.groupSize;
    int64_t dtype = ctx.nodeQuant != nullptr ? static_cast<int64_t>(ge::DT_INT8) : ctx.dtype;
    ge::AscendString fusedOpType(fused_weight_quant_mat_mul::FUSED_OP_TYPE_RELU);
    return fusedNode.SetAttr(ge::AscendString("transpose_x"), transX) == ge::GRAPH_SUCCESS &&
           fusedNode.SetAttr(ge::AscendString("transpose_weight"), transWeight) == ge::GRAPH_SUCCESS &&
           fusedNode.SetAttr(ge::AscendString("antiquant_group_size"), groupSize) == ge::GRAPH_SUCCESS &&
           fusedNode.SetAttr(ge::AscendString("dtype"), dtype) == ge::GRAPH_SUCCESS &&
           fusedNode.SetAttr(ge::AscendString("fused_op_type"), fusedOpType) == ge::GRAPH_SUCCESS;
}

// desc 双侧补齐：对子图输入 producer 与融合节点输入补齐 desc（dtype/shape/format/origin
// 四元组），漏了会导致后续 inferShape/内存分配用错格式；子模式 b 的 scale desc 强制为
// DT_UINT64 + Shape{n} + FORMAT_ND（与折叠 Const 一致）
static bool UpdateFusedNodeDescs(const FusedCtx& ctx, ge::GNode& fusedNode, const ReplacementDescs& descs,
                                 const ge::es::EsTensorHolder& rX, const ge::es::EsTensorHolder& rW,
                                 const ge::es::EsTensorHolder& rScale, const ge::es::EsTensorHolder& rBias)
{
    if (!SetProducerOutputDesc(rX.GetProducer(), descs.xDesc.GetDataType(), descs.xDesc.GetShape(),
                               descs.xDesc.GetFormat()) ||
        !SetProducerOutputDesc(rW.GetProducer(), descs.wDesc.GetDataType(), descs.wDesc.GetShape(),
                               descs.wDesc.GetFormat())) {
        return false;
    }
    const bool scaleUpdated = ctx.nodeQuant != nullptr ?
                                  SetProducerOutputDesc(rScale.GetProducer(), ge::DT_UINT64, ge::Shape({ctx.nDim}),
                                                        ge::FORMAT_ND) :
                                  SetProducerOutputDesc(rScale.GetProducer(), descs.scaleDesc.GetDataType(),
                                                        descs.scaleDesc.GetShape(), descs.scaleDesc.GetFormat());
    if (!scaleUpdated ||
        (descs.hasBias && !SetProducerOutputDesc(rBias.GetProducer(), descs.biasDesc.GetDataType(),
                                                 descs.biasDesc.GetShape(), descs.biasDesc.GetFormat())) ||
        !SetProducerOutputDesc(&fusedNode, descs.outDesc.GetDataType(), descs.outDesc.GetShape(),
                               descs.outDesc.GetFormat()) ||
        !SetConsumerInputDesc(&fusedNode, X_PORT, descs.xDesc.GetDataType(), descs.xDesc.GetShape(),
                              descs.xDesc.GetFormat()) ||
        !SetConsumerInputDesc(&fusedNode, WEIGHT_PORT, descs.wDesc.GetDataType(), descs.wDesc.GetShape(),
                              descs.wDesc.GetFormat())) {
        return false;
    }
    const bool scaleInputUpdated = ctx.nodeQuant != nullptr ?
                                       SetConsumerInputDesc(&fusedNode, ANTI_QUANT_SCALE_PORT, ge::DT_UINT64,
                                                            ge::Shape({ctx.nDim}), ge::FORMAT_ND) :
                                       SetConsumerInputDesc(&fusedNode, ANTI_QUANT_SCALE_PORT,
                                                            descs.scaleDesc.GetDataType(), descs.scaleDesc.GetShape(),
                                                            descs.scaleDesc.GetFormat());
    if (!scaleInputUpdated) {
        return false;
    }
    return !descs.hasBias || SetConsumerInputDesc(&fusedNode, BIAS_PORT, descs.biasDesc.GetDataType(),
                                                  descs.biasDesc.GetShape(), descs.biasDesc.GetFormat());
}

// 构建替换子图：子模式 a——antiquant_scale 取子图输入（外部边直连）；
// 子模式 b——折叠为内部 Const（合并编码），输出 desc 取量化节点输出（INT8）
static GraphUniqPtr BuildReplacementGraph(const FusedCtx& ctx)
{
    ReplacementDescs descs;
    if (!CollectSourceDescs(ctx, descs)) {
        return nullptr;
    }

    auto builder = ge::es::EsGraphBuilder("replacement");
    int64_t idx = 0;
    auto rX = builder.CreateInput(idx++, "x", descs.xDesc.GetDataType(), descs.xDesc.GetFormat(),
                                  descs.xDesc.GetShape().GetDims());
    auto rW = builder.CreateInput(idx++, "weight", descs.wDesc.GetDataType(), descs.wDesc.GetFormat(),
                                  descs.wDesc.GetShape().GetDims());
    // 子模式 a：scale 为子图输入；子模式 b：折叠为内部 Const（边界不含 scale）
    ge::es::EsTensorHolder rScale(nullptr);
    if (ctx.nodeQuant != nullptr) {
        rScale = builder.CreateConst(ctx.foldedScale, {ctx.nDim}, ge::DT_UINT64, ge::FORMAT_ND);
    } else {
        rScale = builder.CreateInput(idx++, "antiquant_scale", descs.scaleDesc.GetDataType(),
                                     descs.scaleDesc.GetFormat(), descs.scaleDesc.GetShape().GetDims());
    }
    ge::es::EsTensorHolder rBias(nullptr);
    if (descs.hasBias) {
        rBias = builder.CreateInput(idx++, "bias", descs.biasDesc.GetDataType(), descs.biasDesc.GetFormat(),
                                    descs.biasDesc.GetShape().GetDims());
    }

    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    if (graph == nullptr) {
        return nullptr;
    }
    ge::GNode fusedNode;
    if (!BuildFusedNode(*graph, ctx, descs.hasBias, rX, rW, rScale, rBias, fusedNode) ||
        !UpdateFusedNodeDescs(ctx, fusedNode, descs, rX, rW, rScale, rBias)) {
        return nullptr;
    }

    auto yOut = ge::es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(fusedNode, 0));
    return builder.BuildAndReset({yOut});
}

static std::unique_ptr<SubgraphBoundary> BuildBoundary(const FusedCtx& ctx)
{
    constexpr int64_t kQuantFusionBiasInputIndex = 2;
    constexpr int64_t kReluFusionBiasInputIndex = 3;
    auto boundary = std::make_unique<SubgraphBoundary>();

    ge::fusion::SubgraphInput input0;
    input0.AddInput({*ctx.nodeWqbmm, X_PORT});
    if (boundary->AddInput(X_PORT, std::move(input0)) != ge::SUCCESS) {
        return nullptr;
    }
    ge::fusion::SubgraphInput input1;
    input1.AddInput({*ctx.nodeWqbmm, WEIGHT_PORT});
    if (boundary->AddInput(WEIGHT_PORT, std::move(input1)) != ge::SUCCESS) {
        return nullptr;
    }
    if (ctx.nodeQuant == nullptr) {
        // 子模式 a：antiquant_scale 外部边直连为子图输入
        ge::fusion::SubgraphInput input2;
        input2.AddInput({*ctx.nodeWqbmm, ANTI_QUANT_SCALE_PORT});
        if (boundary->AddInput(ANTI_QUANT_SCALE_PORT, std::move(input2)) != ge::SUCCESS) {
            return nullptr;
        }
    }
    // 子模式 b：antiquant_scale 由内部 Const（合并编码）承载，不进边界
    ge::GNodePtr biasNode = nullptr;
    if (ctx.biasPort >= 0) {
        biasNode = ctx.nodeWqbmm->GetInDataNodesAndPortIndexs(static_cast<int32_t>(ctx.biasPort)).first;
    }
    if (biasNode != nullptr) {
        const int64_t biasIdx = (ctx.nodeQuant != nullptr) ? kQuantFusionBiasInputIndex : kReluFusionBiasInputIndex;
        ge::fusion::SubgraphInput inputBias;
        inputBias.AddInput({*ctx.nodeWqbmm, ctx.biasPort});
        if (boundary->AddInput(biasIdx, std::move(inputBias)) != ge::SUCCESS) {
            return nullptr;
        }
    }
    // 融合输出边界：子模式 a 取 Relu 输出（fp16），子模式 b 取量化节点输出（int8）
    if (ctx.nodeQuant != nullptr) {
        ge::fusion::SubgraphOutput output({*ctx.nodeQuant, 0});
        if (boundary->AddOutput(0, std::move(output)) != ge::SUCCESS) {
            return nullptr;
        }
    } else {
        ge::fusion::SubgraphOutput output({*ctx.nodeRelu, 0});
        if (boundary->AddOutput(0, std::move(output)) != ge::SUCCESS) {
            return nullptr;
        }
    }
    return boundary;
}

static ge::Status ProcessSingleNodeFusion(const ge::GraphPtr& graph, const ge::GNode& wqbmmNode,
                                          ge::CustomPassContext& passContext)
{
    FusedCtx ctx;
    if (!MatchAndValidate(wqbmmNode, ctx)) {
        return ge::GRAPH_NOT_CHANGED;
    }
    // 子模式 b 尝试（y INT8 输出量化）：条件不满足时 fallback 子模式 a（nodeQuant 保持空）
    (void)MatchQuantFusion(ctx);
    auto replacement = BuildReplacementGraph(ctx);
    if (replacement == nullptr) {
        OP_LOGW(kFusedOpType, "Failed to build replacement graph");
        return ge::GRAPH_NOT_CHANGED;
    }
    auto boundary = BuildBoundary(ctx);
    if (boundary == nullptr) {
        OP_LOGW(kFusedOpType, "Failed to build boundary");
        return ge::GRAPH_NOT_CHANGED;
    }

    passContext.SetPassName(ge::AscendString(kFusedOpType));
#if GE_COMPILER_VERSION_NUM >= 90100000
    // GE 9.1.0+ 运行时存在带 CustomPassContext 的 Replace 强符号，优先使用（对齐
    // anti_quant_matmul_fusion_pass 的版本兼容写法）
    using ReplaceWithCtxFn = ge::Status (*)(const ge::fusion::SubgraphBoundary&, const ge::Graph&,
                                            ge::CustomPassContext&);
    auto replaceWithCtx = static_cast<ReplaceWithCtxFn>(&ge::fusion::SubgraphRewriter::Replace);
    if (replaceWithCtx != nullptr) {
        if (replaceWithCtx(*boundary, *replacement, passContext) != ge::SUCCESS) {
            OP_LOGW(kFusedOpType, "SubgraphRewriter::Replace failed");
            return ge::GRAPH_NOT_CHANGED;
        }
    } else if (ge::fusion::SubgraphRewriter::Replace(*boundary, *replacement) != ge::SUCCESS) {
        OP_LOGW(kFusedOpType, "SubgraphRewriter::Replace failed");
        return ge::GRAPH_NOT_CHANGED;
    }
#else
    if (ge::fusion::SubgraphRewriter::Replace(*boundary, *replacement) != ge::SUCCESS) {
        OP_LOGW(kFusedOpType, "SubgraphRewriter::Replace failed");
        return ge::GRAPH_NOT_CHANGED;
    }
#endif

    // 移除旧节点：子模式 a 移除 Relu 与 WQBMMV2；子模式 b 另移除量化节点
    // （Replace 成功后各节点输出均已被替换；旧 antiquant_scale Const 在 b 模式下成为孤立
    // 节点，由 GE 图清理回收，可能仍被其他节点共享引用，不做显式删除）
    if (ctx.nodeQuant != nullptr && graph->RemoveNode(*ctx.nodeQuant) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to remove quant node after fusion");
    }
    if (graph->RemoveNode(*ctx.nodeRelu) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to remove Relu node after fusion");
    }
    if (graph->RemoveNode(*ctx.nodeWqbmm) != ge::GRAPH_SUCCESS) {
        OP_LOGW(kFusedOpType, "Failed to remove WQBMMV2 node after fusion");
    }
    if (ctx.nodeQuant != nullptr) {
        OP_LOGI(kFusedOpType,
                "Fuse WeightQuantBatchMatmulV2 + Relu + AscendQuant into FusedWeightQuantMatMul(y int8) succeeded");
    } else {
        OP_LOGI(kFusedOpType, "Fuse WeightQuantBatchMatmulV2 + Relu into FusedWeightQuantMatMul succeeded");
    }
    return ge::SUCCESS;
}

ge::Status FusedWeightQuantMatMulFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    if (!CheckPlatform()) {
        return ge::GRAPH_NOT_CHANGED;
    }
    if (graph == nullptr || !graph->IsValid()) {
        OP_LOGW(kFusedOpType, "Graph is null or invalid, skip fusion pass.");
        return ge::GRAPH_NOT_CHANGED;
    }
    OP_LOGD(kFusedOpType, "Enter FusedWeightQuantMatMulFusionPass");

    std::vector<ge::GNode> wqbmmNodes;
    for (auto& node : graph->GetDirectNode()) {
        ge::AscendString opType;
        if (node.GetType(opType) != ge::GRAPH_SUCCESS) {
            continue;
        }
        if (std::string(opType.GetString()) != kOpTypeWqbmmV2) {
            continue;
        }
        wqbmmNodes.emplace_back(node);
    }
    if (wqbmmNodes.empty()) {
        OP_LOGD(kFusedOpType, "No matched WeightQuantBatchMatmulV2 node found, exit fusion pass");
        return ge::GRAPH_NOT_CHANGED;
    }
    bool changed = false;
    for (auto& node : wqbmmNodes) {
        auto status = ProcessSingleNodeFusion(graph, node, passContext);
        if (status == ge::SUCCESS) {
            changed = true;
            continue;
        }
        if (status != ge::GRAPH_NOT_CHANGED) {
            return status;
        }
    }
    OP_LOGD(kFusedOpType, "FusedWeightQuantMatMulFusionPass completed");
    return changed ? ge::SUCCESS : ge::GRAPH_NOT_CHANGED;
}

#if GE_COMPILER_VERSION_NUM >= 90100000
REG_FUSION_PASS(FusedWeightQuantMatMulFusionPass).Stage(ge::CustomPassStage::kAfterOriginGraphOptimize);
#endif

} // namespace ops
