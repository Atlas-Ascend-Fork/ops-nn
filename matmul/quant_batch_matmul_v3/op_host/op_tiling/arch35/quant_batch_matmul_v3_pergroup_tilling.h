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
 * \file quant_batch_matmul_v3_pergroup_tilling.h
 * \brief A8W8/A8W4 X2 Per-Group tiling strategy for DAV_RESV.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "../quant_batch_matmul_v3_tiling_base.h"
#include "quant_batch_matmul_v3_tiling_util.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"
// Provides the QuantBatchMatmulV3Arch35TilingKey namespace plus TPL_* / GET_TPL_TILING_KEY
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_apt_tiling_key.h"

namespace optiling {

// Per-group quantization along K: one integer matmul per group, dequantized on the fly by FixPipe,
// then accumulated in FP32 in the AIV UB. Only targets with supportMmadS8S4 use this strategy.
class X2PerGroupTiling : public QuantBatchMatmulV3TilingBase {
public:
    explicit X2PerGroupTiling(gert::TilingContext* context);
    ~X2PerGroupTiling() override = default;

    // DoTiling(): GetShapeAttrsInfo -> GetPlatformInfo -> IsCapable -> DoOpTiling -> DoLibApiTiling ->
    // GetWorkspaceSize -> PostTiling -> GetTilingKey. First two are swapped vs some headers' numbering.
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

protected:
    bool IsCapable() override;
    bool CheckDtype() const override;
    bool CheckShape(const std::vector<gert::Shape*>& mandatoryShape, const gert::StorageShape* biasShape,
                    const gert::StorageShape* pertokenShape, const std::vector<int64_t>& dimValueOfMKN) const override;

private:
    // False when clearing the raw tiling data fails. The constructor cannot report that, so the
    // result is kept in resetFailed_ and DoOpTiling() fails before it writes into a dirty buffer.
    bool Reset();
    // DoOpTiling() helpers, in call order. CheckCoreNum() comes first because this strategy requires
    // AIC:AIV = 1:1 and the base class offers no hook for that check.
    bool CheckCoreNum() const;
    bool CalcBaseBlock();
    // baseK must divide groupSizeK so a K chunk never straddles two scales.
    uint64_t CalcBaseK(uint64_t baseKFromCalculator) const;
    void CalcUsedCoreNum();
    bool CalcL1Tiling();
    struct ResidentLayout {
        uint64_t aPanelBytes, bPanelBytes;
    };
    bool CanUseResidentLayout() const;
    ResidentLayout GetResidentLayout() const;
    // Retain one A panel across N tiles, with four B windows and two scale buffers.
    bool CanAFullLoad(uint64_t kAL1 = UINT64_MAX, uint64_t kBL1 = UINT64_MAX) const;
    uint64_t CalcAFullLoadStepK() const;
    // Retain one B panel across M tiles, with four A windows and two scale buffers.
    bool CanBFullLoad(uint64_t kBL1 = UINT64_MAX, uint64_t kAL1 = UINT64_MAX) const;
    uint64_t CalcBFullLoadStepK() const;
    uint64_t CalcFullLoadStepK(uint64_t residentBytes, uint64_t baseWindowBytes) const;
    // Priority: A -> B -> streaming. Broadcast caching is selected by the kernel.
    void SelectReuseMode(uint64_t kAL1 = UINT64_MAX, uint64_t kBL1 = UINT64_MAX);
    // Select the TilingKey from the final emitted K windows.
    void SelectKernelType();

    // Solve stepK from the L1 budget: whole groups per kL1 tile, or a factor of one group when even
    // a single group does not fit.
    uint64_t CalcL1StepK() const;
    void SetQuantMode();
    uint64_t GetBatchMode() const;
    uint64_t GetKernelType() const;
    uint64_t GetApiLevel() const;

    DequantBmm::QuantBatchMatmulV3BasicAPITilingData tilingDataSelf_;
    DequantBmm::QuantBatchMatmulV3BasicAPITilingData& tilingData_;
    BasicRunInfoTiling basicTiling_;
    uint64_t mCnt_ = 1UL; // number of base blocks along M
    uint64_t nCnt_ = 1UL; // number of base blocks along N
    bool resetFailed_ = false;
    bool bFullLoad_ = false;
    bool aFullLoad_ = false;
};
} // namespace optiling
