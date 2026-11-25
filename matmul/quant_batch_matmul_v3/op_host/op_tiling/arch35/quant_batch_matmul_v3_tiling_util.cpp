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
 * \file quant_batch_matmul_v3_tiling_util.cpp
 * \brief
 */
#include "quant_batch_matmul_v3_tiling_util.h"

#include <new>
#include "error_util.h"
#include "platform/platform_infos_def.h"
#include "quant_batch_matmul_v3_checker.h"
#include "quant_batch_matmul_v3_checker_for_mmads8s4.h"
#include "quant_batch_matmul_v3_checker_arch_9201.h"
#include "op_common/op_host/util/op_const_def.h"

namespace optiling {
namespace {
// Minimum HBM bandwidth of DAV_3510 series, in TB/s.
constexpr double HBM_BW = 1.4;
// Minimum L2 bandwidth of DAV_3510 series, in TB/s.
constexpr double L2_BW = 5.2;
// MXFP4 Cube throughput of 32-core DAV_3510 series, in TFLOPS.
constexpr double MXFP4_CUBE_THROUGHPUT = 1728.0;
// MXFP8 Cube throughput of 32-core DAV_3510 series, in TFLOPS.
constexpr double MXFP8_CUBE_THROUGHPUT = 864.0;
constexpr double MMAD_OPS_PER_DOT = 2.0;
constexpr uint64_t MMAD_BLOCK_SIZE = 256UL;
constexpr uint64_t MX_SCALE_GROUP_SIZE = 32UL;
// 带宽换算系数：TB/s × 1e6 = B/us。
constexpr double SECONDS_TO_US = 1e6;
constexpr uint64_t L2_CACHE_SIZE = 128UL * 1024UL * 1024UL;
constexpr uint64_t L2_CACHE_THRESHOLD_PERCENT = 80UL;
constexpr uint64_t PERCENT_BASE = 100UL;
constexpr uint64_t L2_CACHE_THRESHOLD_SIZE = L2_CACHE_SIZE * L2_CACHE_THRESHOLD_PERCENT / PERCENT_BASE;
constexpr uint64_t MXFP8_L0C_PINGPONG_INNER_AXIS_ALIGN_SIZE = 128UL;
constexpr uint64_t MXFP4_L0C_PINGPONG_INNER_AXIS_ALIGN_SIZE = 256UL;

template <typename Checker>
bool CheckDtypeWithChecker(gert::TilingContext* context, const QuantBatchMatmulInfo& inputParams)
{
    auto* qmmV3Checker = new (std::nothrow) Checker(context, inputParams);
    OP_TILING_CHECK(qmmV3Checker == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams.opName, "Failed to instantiate qmmV3Checker."), return false);
    bool res = qmmV3Checker->CheckDtype();
    delete qmmV3Checker;
    return res;
}

template <typename Checker>
bool CheckShapeWithChecker(gert::TilingContext* context, const QuantBatchMatmulInfo& inputParams,
                           const std::vector<gert::Shape*>& mandatoryShape, const gert::StorageShape* biasShape,
                           const gert::StorageShape* pertokenShape, const std::vector<int64_t>& dimValueOfMKN)
{
    auto* qmmV3Checker = new (std::nothrow) Checker(context, inputParams);
    OP_TILING_CHECK(qmmV3Checker == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams.opName, "Failed to instantiate qmmV3Checker."), return false);
    bool res = qmmV3Checker->CheckShape(mandatoryShape, biasShape, pertokenShape, dimValueOfMKN);
    delete qmmV3Checker;
    return res;
}
bool IsMxCubeBound(const QuantBatchMatmulInfo& inputParams)
{
    const uint64_t m = inputParams.mSize;
    const uint64_t n = inputParams.nSize;
    const uint64_t k = inputParams.kSize;
    const bool isMxfp4Input = inputParams.aDtype == ge::DT_FLOAT4_E2M1;
    const double cubeThroughput = isMxfp4Input ? MXFP4_CUBE_THROUGHPUT : MXFP8_CUBE_THROUGHPUT;
    const uint64_t aInputCost = m * k;
    const uint64_t bInputCost = n * k;
    const uint64_t aScaleCost = ops::CeilDiv(aInputCost, MX_SCALE_GROUP_SIZE);
    const uint64_t bScaleCost = ops::CeilDiv(bInputCost, MX_SCALE_GROUP_SIZE);
    const uint64_t copyOutBytes = GetSizeWithDataType(m * n, inputParams.cDtype);
    const uint64_t hbmBytes = GetSizeWithDataType(aInputCost, inputParams.aDtype) +
                              GetSizeWithDataType(aScaleCost, inputParams.perTokenScaleDtype) +
                              GetSizeWithDataType(bInputCost, inputParams.bDtype) +
                              GetSizeWithDataType(bScaleCost, inputParams.scaleDtype);
    const uint64_t aL2Cost = m * k * (ops::CeilDiv(n, MMAD_BLOCK_SIZE) - 1UL);
    const uint64_t bL2Cost = n * k * (ops::CeilDiv(m, MMAD_BLOCK_SIZE) - 1UL);
    const uint64_t l2Bytes = GetSizeWithDataType(aL2Cost, inputParams.aDtype) +
                             GetSizeWithDataType(ops::CeilDiv(aL2Cost, MX_SCALE_GROUP_SIZE),
                                                 inputParams.perTokenScaleDtype) +
                             GetSizeWithDataType(bL2Cost, inputParams.bDtype) +
                             GetSizeWithDataType(ops::CeilDiv(bL2Cost, MX_SCALE_GROUP_SIZE), inputParams.scaleDtype);
    const uint64_t l2Footprint = hbmBytes + copyOutBytes;
    const double copyOutBandwidth = l2Footprint < L2_CACHE_THRESHOLD_SIZE ? L2_BW : HBM_BW;
    const double transferCost = static_cast<double>(hbmBytes) / HBM_BW + static_cast<double>(l2Bytes) / L2_BW +
                                static_cast<double>(copyOutBytes) / copyOutBandwidth;
    const double computeCost = MMAD_OPS_PER_DOT * static_cast<double>(m) * static_cast<double>(n) *
                               static_cast<double>(k) / cubeThroughput;
    return computeCost > transferCost;
}

// 9201 用 compileInfo.socSpec 的真实平台规格动态判定 cube bound（静态常量按 950 校准，在 9201 上会失真），
// 与 EstimateMte2Time/EstimateMmadTime 同口径。
bool IsMxCubeBound(const QuantBatchMatmulInfo& inputParams, const SocSpecifyParams& socParams, uint32_t aicNum)
{
    const uint64_t m = inputParams.mSize;
    const uint64_t n = inputParams.nSize;
    const uint64_t k = inputParams.kSize;
    if (!HasPerGroupScale(inputParams) || aicNum == 0U) {
        return false;
    }
    // MX 每 32 K 元素一组 scale，hif4 每 64 K 元素一组 scale。
    const uint64_t scaleGroupSize = GetScaleGroupSize(inputParams.scaleDtype);
    const uint64_t aInputCost = m * k;
    const uint64_t bInputCost = n * k;
    const uint64_t aScaleCost = ops::CeilDiv(aInputCost, scaleGroupSize);
    const uint64_t bScaleCost = ops::CeilDiv(bInputCost, scaleGroupSize);
    const uint64_t copyOutBytes = GetSizeWithDataType(m * n, inputParams.cDtype);
    const uint64_t hbmBytes = GetSizeWithDataType(aInputCost, inputParams.aDtype) +
                              GetSizeWithDataType(aScaleCost, inputParams.perTokenScaleDtype) +
                              GetSizeWithDataType(bInputCost, inputParams.bDtype) +
                              GetSizeWithDataType(bScaleCost, inputParams.scaleDtype);
    const uint64_t aL2Cost = m * k * (ops::CeilDiv(n, MMAD_BLOCK_SIZE) - 1UL);
    const uint64_t bL2Cost = n * k * (ops::CeilDiv(m, MMAD_BLOCK_SIZE) - 1UL);
    const uint64_t l2Bytes = GetSizeWithDataType(aL2Cost, inputParams.aDtype) +
                             GetSizeWithDataType(ops::CeilDiv(aL2Cost, scaleGroupSize),
                                                 inputParams.perTokenScaleDtype) +
                             GetSizeWithDataType(bL2Cost, inputParams.bDtype) +
                             GetSizeWithDataType(ops::CeilDiv(bL2Cost, scaleGroupSize), inputParams.scaleDtype);
    const uint64_t l2Footprint = hbmBytes + copyOutBytes;
    const double copyOutBandwidth = l2Footprint < L2_CACHE_THRESHOLD_SIZE ? socParams.l2Bw : socParams.hbmBw;
    // MTE2 访存耗时（us）：带宽 TB/s × SECONDS_TO_US → B/us。
    const double transferCost = static_cast<double>(hbmBytes) / (socParams.hbmBw * SECONDS_TO_US) +
                                static_cast<double>(l2Bytes) / (socParams.l2Bw * SECONDS_TO_US) +
                                static_cast<double>(copyOutBytes) / (copyOutBandwidth * SECONDS_TO_US);

    // MMAD 计算耗时（us）：整片分片次数 / 整片频率（freqMHz × aicNum）。
    const uint64_t mFracM = ops::CeilDiv(m, socParams.mmadFracM);
    const uint64_t nFracN = ops::CeilDiv(n, socParams.mmadFracN);
    const uint64_t kFracK = ops::CeilDiv(GetSizeWithDataType(k, inputParams.aDtype), socParams.mmadFracK);
    const double freqMhz = static_cast<double>(socParams.freq) * static_cast<double>(GHZ_TO_MHZ);
    const double computeCost = static_cast<double>(mFracM * nFracN * kFracK) / (freqMhz * static_cast<double>(aicNum));
    return computeCost > transferCost;
}

// 平台规格默认值（平台信息缺失时回退）：v120/950 相近量级。
constexpr uint64_t LB_DEFAULT_FREQ_MHZ = 1650UL;
constexpr uint64_t LB_DEFAULT_DDR_RATE = 31UL;
constexpr uint64_t LB_DEFAULT_L2_RATE = 100UL;
constexpr uint64_t LB_DEFAULT_CORE_CNT = 32UL;
constexpr uint64_t LB_KB_SIZE = 1024UL;

// 平台规格字符串转数值，非法/负值/未完全解析时回退默认值。
int64_t LbStrToIntWithDefault(const std::string& str, int64_t defaultValue)
{
    if (str.empty()) {
        return defaultValue;
    }
    try {
        size_t pos = 0;
        long long parsed = std::stoll(str, &pos, 10); // 10: 十进制
        if (pos != str.size() || parsed < 0) {
            return defaultValue;
        }
        return static_cast<int64_t>(parsed);
    } catch (const std::exception&) {
        return defaultValue;
    }
}

// 平台规格字符串转正数值：在 LbStrToIntWithDefault 基础上，解析结果为 0 同样视为非法并回退默认值，
// 保证解析出的 hbmBw/l2Bw 恒大于 0。
int64_t LbStrToPositiveIntWithDefault(const std::string& str, int64_t defaultValue)
{
    int64_t parsed = LbStrToIntWithDefault(str, defaultValue);
    return parsed > 0 ? parsed : defaultValue;
}

} // namespace

void FillSocSpecifyParamsFromPlatform(fe::PlatFormInfos* platformInfo, SocSpecifyParams& outInfo)
{
    if (platformInfo == nullptr) {
        return;
    }
    std::string coreCntStr = std::to_string(LB_DEFAULT_CORE_CNT);
    std::string ddrRateStr = std::to_string(LB_DEFAULT_DDR_RATE);
    std::string l2RateStr = std::to_string(LB_DEFAULT_L2_RATE);
    std::string freqStr = std::to_string(LB_DEFAULT_FREQ_MHZ);

    (void)platformInfo->GetPlatformRes("SoCInfo", "ai_core_cnt", coreCntStr);
    (void)platformInfo->GetPlatformRes("AICoreMemoryRates", "ddr_rate", ddrRateStr);
    (void)platformInfo->GetPlatformRes("AICoreMemoryRates", "l2_rate", l2RateStr);
    (void)platformInfo->GetPlatformRes("AICoreSpec", "cube_freq", freqStr);

    outInfo.freq = static_cast<double>(LbStrToPositiveIntWithDefault(freqStr, LB_DEFAULT_FREQ_MHZ)) /
                   static_cast<double>(GHZ_TO_MHZ);
    // 解析结果（含字符串 "0"）为 0 时统一回退默认规格，保证 hbmBw/l2Bw 恒大于 0。
    int64_t coreCnt = LbStrToPositiveIntWithDefault(coreCntStr, LB_DEFAULT_CORE_CNT);
    int64_t ddrRate = LbStrToPositiveIntWithDefault(ddrRateStr, LB_DEFAULT_DDR_RATE);
    int64_t l2Rate = LbStrToPositiveIntWithDefault(l2RateStr, LB_DEFAULT_L2_RATE);
    outInfo.hbmBw = outInfo.freq * static_cast<double>(coreCnt) * static_cast<double>(ddrRate) / LB_KB_SIZE;
    outInfo.l2Bw = outInfo.freq * static_cast<double>(coreCnt) * static_cast<double>(l2Rate) / LB_KB_SIZE;
}

bool GetSocSpecAdjustBalanceInfo(const QuantBatchMatmulInfo& inputParams,
                                 const QuantBatchMatmulV3CompileInfo& compileInfo, SocSpecifyParams& outInfo)
{
    // socSpec 未被 TilingParse/InitCompileInfo 填充时保守返回 false，等价旧版拿不到 platformInfo 的行为。
    if (compileInfo.socSpec.freq <= 0.0 || compileInfo.socSpec.hbmBw <= 0.0 || compileInfo.socSpec.l2Bw <= 0.0) {
        return false;
    }
    outInfo = compileInfo.socSpec;
    // 9202 对 hif4 逐组量化与 mxfp4 输入的 mmad N 方向碎片翻倍修正。
    if (compileInfo.npuArch == NpuArch::DAV_9202 &&
        (inputParams.isHif4PerGroup || (inputParams.isMxPerGroup && inputParams.aDtype == ge::DT_FLOAT4_E2M1))) {
        outInfo.mmadFracN *= 2UL;
    }
    outInfo.CalSingleCoreComputePower(inputParams.aDtype);
    return true;
}

MxL0CPingpongEligibility GetMxL0CPingpongEligibility(const QuantBatchMatmulInfo& inputParams,
                                                     const QuantBatchMatmulV3CompileInfo& compileInfo,
                                                     bool tensorApiEnabled, uint64_t totalWinCnt)
{
    const NpuArch npuArch = compileInfo.npuArch;
    // MX L0C ping-pong 模板在 DAV_3510/DAV_9201 生效。
    if ((npuArch != Ops::Base::DAV_3510 && npuArch != Ops::Base::DAV_9201) || !tensorApiEnabled ||
        !IsMxBasicApiCapable(inputParams)) {
        return {};
    }

    // 9201 上 mxfp4（A/B 均 4bit）在 ping-pong 模板下性能未达预期，暂回退常规 blaze 模板；
    // 3510 上该组合保持 shape 级判定不受影响。
    if (npuArch == Ops::Base::DAV_9201 &&
        ((inputParams.aDtype == ge::DT_FLOAT4_E2M1 && inputParams.bDtype == ge::DT_FLOAT4_E2M1) ||
         (inputParams.aDtype == ge::DT_HIFLOAT4 && inputParams.bDtype == ge::DT_HIFLOAT4))) {
        return {};
    }

    // 9201 仅 mxfp8 + cube bound + 每核多轮的完整条件集才走入 l0cDB，各门槛对 explicitPlan
    // 与 legacyTail 同步生效；3510 保持 master 行为（门槛仅关闭 legacy 尾）。
    const bool isDav9201 = npuArch == Ops::Base::DAV_9201;

    // The legacy tail path has additional gates and also controls the original scale window adjustment.
    if (inputParams.scaleDtype != ge::DT_FLOAT8_E8M0 ||
        (inputParams.cDtype != ge::DT_FLOAT16 && inputParams.cDtype != ge::DT_BF16)) {
        return {!isDav9201, false};
    }
    const uint64_t aInnerAxis = inputParams.transA ? inputParams.mSize : inputParams.kSize;
    const uint64_t bInnerAxis = inputParams.transB ? inputParams.kSize : inputParams.nSize;
    const uint64_t innerAxisAlignSize = inputParams.aDtype == ge::DT_FLOAT4_E2M1 ?
                                            MXFP4_L0C_PINGPONG_INNER_AXIS_ALIGN_SIZE :
                                            MXFP8_L0C_PINGPONG_INNER_AXIS_ALIGN_SIZE;
    const bool isInnerAxisAligned = GetSizeWithDataType(aInnerAxis, inputParams.aDtype) % innerAxisAlignSize == 0UL &&
                                    GetSizeWithDataType(bInnerAxis, inputParams.bDtype) % innerAxisAlignSize == 0UL;
    if (!isInnerAxisAligned) {
        // 9201 内轴不对齐不走入；3510 的 SelectMxL0TilePlan 内部另有对齐判定。
        return {!isDav9201, false};
    }
    // 3510 沿用静态估算；9201 用 compileInfo.socSpec 的真实平台规格动态判定 cube bound，
    // socSpec 未填充时保守不进。
    if (isDav9201) {
        // 9201 仅每核多轮（totalWinCnt > 1）的用例走入 L0C ping-pong：单轮时 L0C 双缓冲
        // 无下一轮计算可流水掩盖，只剩半幅 L0C 与 stepK 收敛的代价，回退常规 blaze 模板；
        // totalWinCnt == 0 表示调用方无窗口信息，跳过轮数判定。
        if (totalWinCnt == 1UL) {
            return {false, false};
        }
        SocSpecifyParams socParams;
        if (!GetSocSpecAdjustBalanceInfo(inputParams, compileInfo, socParams)) {
            return {false, false};
        }
        const bool isCubeBound = IsMxCubeBound(inputParams, socParams, compileInfo.aicNum);
        return {isCubeBound, isCubeBound};
    }
    return {true, IsMxCubeBound(inputParams)};
}

void QuantBatchMatMulV3TilingUtil::SetBasicTilingData(const QuantBatchMatmulInfo& inputParams,
                                                      const BasicRunInfoTiling& basicTiling,
                                                      DequantBmm::QuantBatchMatmulV3TilingDataParams& tilingData)
{
    SetCommonTilingData(inputParams, tilingData);
    tilingData.params.ubCalcN = basicTiling.ubCalcN;
    tilingData.params.ubCalcM = basicTiling.ubCalcM;
    tilingData.params.isPerTensor = static_cast<uint32_t>(inputParams.isPerTensor);
    tilingData.params.isPertoken = static_cast<uint32_t>(inputParams.isPertoken);
    tilingData.params.isDoubleScale = static_cast<uint32_t>(inputParams.isDoubleScale);
}

void QuantBatchMatMulV3TilingUtil::SetBasicLibApiTiling(const QuantBatchMatmulInfo& inputParams,
                                                        const BasicRunInfoTiling& basicTiling,
                                                        DequantBmm::QuantBatchMatmulV3TilingDataParams& tilingData)
{
    tilingData.matmulTiling.M = inputParams.mSize;
    tilingData.matmulTiling.N = inputParams.nSize;
    tilingData.matmulTiling.Ka = inputParams.kSize;
    tilingData.matmulTiling.Kb = inputParams.kSize;
    tilingData.matmulTiling.usedCoreNum = basicTiling.usedCoreNum;
    tilingData.matmulTiling.singleCoreM = basicTiling.singleCoreM;
    tilingData.matmulTiling.singleCoreN = basicTiling.singleCoreN;
    tilingData.matmulTiling.singleCoreK = basicTiling.singleCoreK;
    tilingData.matmulTiling.baseM = basicTiling.baseM;
    tilingData.matmulTiling.baseN = basicTiling.baseN;
    tilingData.matmulTiling.baseK = basicTiling.baseK;
    tilingData.matmulTiling.depthA1 = basicTiling.depthA1;
    tilingData.matmulTiling.depthB1 = basicTiling.depthB1;
    tilingData.matmulTiling.stepM = basicTiling.stepM;
    tilingData.matmulTiling.stepN = basicTiling.stepN;
    tilingData.matmulTiling.stepKa = basicTiling.stepKa;
    tilingData.matmulTiling.stepKb = basicTiling.stepKb;
    tilingData.matmulTiling.isBias = inputParams.hasBias ? 1 : 0;
    tilingData.matmulTiling.iterateOrder = basicTiling.iterateOrder;
    tilingData.matmulTiling.dbL0A = 2; // Double buffer flag: 1 disables DB, 2 enables DB.
    tilingData.matmulTiling.dbL0B = 2; // Double buffer flag: 1 disables DB, 2 enables DB.
    tilingData.matmulTiling.dbL0C = basicTiling.dbL0c;
}

bool QuantBatchMatMulV3TilingUtil::CheckDtype(gert::TilingContext* context, const QuantBatchMatmulInfo& inputParams,
                                              const QuantBatchMatmulV3CompileInfo& compileInfo)
{
    switch (compileInfo.npuArch) {
        case Ops::Base::DAV_3510:
            return CheckDtypeWithChecker<QuantBatchMatmulV3Checker>(context, inputParams);
        case Ops::Base::DAV_9201:
        case Ops::Base::DAV_9202:
            return CheckDtypeWithChecker<QuantBatchMatmulV3CheckerArch9201>(context, inputParams);
        case NpuArch::DAV_RESV:
            return CheckDtypeWithChecker<QuantBatchMatmulV3Checker4MmadS8S4>(context, inputParams);
        default:
            CUBE_INNER_ERR_REPORT(inputParams.opName,
                                  "Failed to find CheckDtype function for current NPU architecture.");
            return false;
    }
}

bool QuantBatchMatMulV3TilingUtil::CheckShape(gert::TilingContext* context, const QuantBatchMatmulInfo& inputParams,
                                              const QuantBatchMatmulV3CompileInfo& compileInfo,
                                              const std::vector<gert::Shape*>& mandatoryShape,
                                              const gert::StorageShape* biasShape,
                                              const gert::StorageShape* pertokenShape,
                                              const std::vector<int64_t>& dimValueOfMKN)
{
    switch (compileInfo.npuArch) {
        case Ops::Base::DAV_3510:
            return CheckShapeWithChecker<QuantBatchMatmulV3Checker>(context, inputParams, mandatoryShape, biasShape,
                                                                    pertokenShape, dimValueOfMKN);
        case Ops::Base::DAV_9201:
        case Ops::Base::DAV_9202:
            return CheckShapeWithChecker<QuantBatchMatmulV3CheckerArch9201>(context, inputParams, mandatoryShape,
                                                                            biasShape, pertokenShape, dimValueOfMKN);
        case NpuArch::DAV_RESV:
            return CheckShapeWithChecker<QuantBatchMatmulV3Checker4MmadS8S4>(context, inputParams, mandatoryShape,
                                                                             biasShape, pertokenShape, dimValueOfMKN);
        default:
            CUBE_INNER_ERR_REPORT(inputParams.opName,
                                  "Failed to find CheckShape function for current NPU architecture.");
            return false;
    }
}

} // namespace optiling
