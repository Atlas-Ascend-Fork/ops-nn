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
 * \file weight_quant_batch_matmul_v2_asw_vc_tiling.h
 * \brief ASW-VC（adaptive sliding window - vector cube）模板 tiling。
 *        场景：A16W4 pergroup（x fp16 ND，weight int4 FRACTAL_NZ，antiquant_scale fp16），
 *        vector 核先做 pergroup 反量化写入 L1(TSCM)，cube 核再做 F16×F16 matmul。
 *        输出平铺的 WeightQuantBatchMatmulV2ASWVCTilingDataParams（供 ASW-VC kernel 使用）。
 */

#pragma once

#include <utility>

#include "weight_quant_batch_matmul_v2_adaptive_sliding_window_tiling.h"
#include "matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_data.h"

namespace optiling {
namespace weight_quant_batch_matmul_v2 {

class WeightQuantBatchMatmulV2TilingASWVC : public WeightQuantBatchMatmulV2Tiling {
public:
    explicit WeightQuantBatchMatmulV2TilingASWVC(gert::TilingContext* context) : WeightQuantBatchMatmulV2Tiling(context)
    {
        if (context->GetCompileInfo() == nullptr) {
            InitCompileInfo();
        }
    }
    ~WeightQuantBatchMatmulV2TilingASWVC() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    struct PlatformPerf {
        double coreFreq;
        double hbmBW;
        double l2BW;
    };

    struct BaseBlockLimits {
        uint64_t maxBaseM;
        uint64_t maxBaseN;
        uint64_t baseMAlignUnit;
        uint64_t baseNAlignUnit;
        uint64_t baseMNBufferLimit;
    };

    ge::graphStatus InstantiateTilingData();
    void ResetBaseTiling();
    ge::graphStatus CalRebalanceBlock();
    PlatformPerf GetPlatformPerf(fe::PlatFormInfos* platformInfo) const;
    BaseBlockLimits CalcBaseBlockLimits(const PlatformPerf& platformPerf);
    void SearchBaseBlock(const BaseBlockLimits& limits);
    void OptimizeBaseBlockForParallelism(const BaseBlockLimits& limits);
    ge::graphStatus ApplyUbLimit();
    void UpdateBlockCounts();
    void UpdateL0cBufferMode();
    uint64_t GetMaxBaseWithLimit(uint64_t baseMNBufferLimit, uint64_t baseAlignUnit, bool isRightMatrix,
                                 bool isMemoryBound) const;
    double GetBalanceRateWithTail(uint64_t baseM, uint64_t baseN) const;
    void CalBaseK();
    void CalL1Tiling();
    void CalL1BufferNum();
    std::pair<uint64_t, uint64_t> CalcL1TensorSizes(uint64_t kL1A, uint64_t kL1B) const;
    uint64_t GetFixedL1Size() const;
    uint64_t CalcUbSize(uint64_t baseN, uint64_t kL1) const;
    void MapRunInfoToTiling();
    bool CheckAntiQuantScale(uint64_t baseN, uint64_t dbL0c = 1) const;
    void SetTilingData();
    void SetBatchParams();
    uint8_t GetVcMmadParam() const;
    void LogTilingData() const;
    wqbmmv2_tiling::L2CacheMode SetDisableL2cache(uint32_t mL1, uint32_t kaL1, uint32_t kbL1, uint32_t nL1) const;
    uint64_t GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype) const;
    uint64_t GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) const;

    std::unique_ptr<wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWVCTilingDataParams> vcTilingData_;
    size_t vcTilingDataSize_ = sizeof(wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWVCTilingDataParams);
    BasicTiling basicTiling_;
    ASWRunInfo runInfo_;
    AdaptiveSlidingWindow adaptiveWin_;
};
} // namespace weight_quant_batch_matmul_v2
} // namespace optiling
