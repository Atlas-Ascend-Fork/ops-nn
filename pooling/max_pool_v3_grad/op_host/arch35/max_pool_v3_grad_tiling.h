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
 * \file max_pool_v3_grad_tiling.h
 * \brief MaxPoolV3Grad NHWC tiling declarations.
 */

#ifndef MAX_POOL_V3_GRAD_TILING_H_
#define MAX_POOL_V3_GRAD_TILING_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "utils/extern_math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "max_pool_v3_grad_tiling_base.h"

namespace optiling {

using MaxPoolV3GradNHWCNameSpace::MaxPoolV3GradNHWCTilingData;

// Pooling input metadata after resolving the data format.
struct MaxPoolV3GradNHWCInputInfo {
    int64_t nX{1};
    int64_t cX{1};
    int64_t hX{1};
    int64_t wX{1};
    int64_t hGrad{1};
    int64_t wGrad{1};
    int64_t hKernel{1};
    int64_t wKernel{1};
    int64_t hStride{1};
    int64_t wStride{1};
    int64_t padH{0};
    int64_t padW{0};
    int64_t padHBack{0};
    int64_t padWBack{0};
    int64_t inputDataCount{0};
    ge::DataType inputDtype{ge::DataType::DT_FLOAT};
    ge::Format inputFormat{ge::Format::FORMAT_NCHW};
};

struct MaxPoolV3GradNHWCBaseInfo {
    int64_t vRegSize{0};
    int64_t ubBlockSize{0};
    int64_t inputBytes{0};
    int64_t indexBytes{0};
    int64_t availableUb{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t maxDataNumInOneBlock{0};
    int64_t proDataNumInOneBeat{0};
    int64_t moveDataNumCacheLine{0};
    int64_t hProBatchSize{0};
    int64_t wProBatchSize{0};
    int64_t isPad{0};
    int64_t isOverlap{0};
};

struct MaxPoolV3GradNHWCSplitInfo {
    int64_t isCheckRange{0};

    int64_t nOutputInner{1};
    int64_t nOutputTail{1};
    int64_t nOutputOuter{1};

    int64_t hOutputInner{1};
    int64_t hOutputTail{1};
    int64_t hOutputOuter{1};

    int64_t wOutputInner{1};
    int64_t wOutputTail{1};
    int64_t wOutputOuter{1};

    int64_t cOutputInner{1};
    int64_t cOutputTail{1};
    int64_t cOutputOuter{1};

    int64_t normalCoreProcessNum{0};
    int64_t tailCoreProcessNum{0};
    int64_t usedCoreNum{0};
    int64_t totalBaseBlockNum{0};

    int64_t inputBufferSize{0};
    int64_t outputBufferSize{0};
    int64_t gradBufferSize{0};
    int64_t argmaxBufferSize{0};
    int64_t totalBufferSize{0};
    int64_t isBigKernel{0};
};

// NHWC tile search using the common pooling-gradient helpers.
class MaxPoolV3GradNHWCTilingImpl {
public:
    explicit MaxPoolV3GradNHWCTilingImpl(MaxPoolV3GradNHWCInputInfo* input) : inputData_(input) {}
    ~MaxPoolV3GradNHWCTilingImpl() = default;

    void InitializationVars(gert::TilingContext* context, int64_t ubSize, int64_t coreNum);
    ge::graphStatus DoOpTiling(gert::TilingContext* context);
    ge::graphStatus PostTiling(gert::TilingContext* context, uint64_t key);
    MaxPoolV3GradNHWCSplitInfo& GetSplitData() { return splitData_; }
    MaxPoolV3GradNHWCBaseInfo& GetBaseData() { return baseData_; }
    void DoBufferCalculate();
    ge::graphStatus SetTilingData(gert::TilingContext* context);
    bool IsMeetUBSize();
    bool IsMeetTargetCoreNum() const;

private:
    void SearchBestTiling();
    void DoUBTiling();
    void DoBlockTiling();
    bool TrySplitN();
    bool TrySplitAlignH();
    bool TrySplitAlignW();
    bool TrySplitAlignC();
    void SplitUnalignHWC();
    void DynamicAdjustmentHW();
    // Contract an axis until the normal core workload would increase.
    void SearchOuterSingle(int64_t& inner, int64_t step);
    // Keep tile boundaries on the stride grid unless each window contains one element.
    void SearchOuterSingleAligned(int64_t& inner, int64_t stride, int64_t kernelSize);
    // Adjust N, H and W in that order to improve core utilization.
    // Keep the contiguous channel axis unchanged to preserve vectorization.
    void AdjustInnerSplitForMultiCore();

    MaxPoolV3GradNHWCBaseInfo baseData_;
    MaxPoolV3GradNHWCSplitInfo splitData_;
    MaxPoolV3GradNHWCInputInfo* inputData_;
};

class MaxPoolV3GradNHWCSmallKernelTiling : public MaxPoolV3GradTilingBase {
public:
    explicit MaxPoolV3GradNHWCSmallKernelTiling(gert::TilingContext* context)
        : MaxPoolV3GradTilingBase(context), nhwcBase_(&nhwcInputData_)
    {}

    ~MaxPoolV3GradNHWCSmallKernelTiling() override = default;

private:
    MaxPoolV3GradNHWCInputInfo nhwcInputData_;
    MaxPoolV3GradNHWCTilingImpl nhwcBase_;
    int64_t isCheckRange_ = 0;

    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

} // namespace optiling

#endif // MAX_POOL_V3_GRAD_TILING_H_
