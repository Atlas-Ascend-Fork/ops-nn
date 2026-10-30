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
 * \file cross_entropy_sum_exp_and_index_logit_tiling_arch35.h
 * \brief A5 (ascend950) tiling data struct
 */
#ifndef OPS_BUILT_IN_OP_TILING_ARCH35_CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_H
#define OPS_BUILT_IN_OP_TILING_ARCH35_CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_H

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "graph/types.h"
#include "loss/cross_entropy_sum_exp_and_index_logit/op_kernel/arch35/cross_entropy_sum_exp_and_index_logit_struct.h"

namespace optiling {

struct CrossEntropySumExpAndIndexLogitCompileInfo {
    int32_t totalCoreNum = 40;
    uint64_t ubSizePlatForm = 0;
};

class CrossEntropySumExpAndIndexLogitTiling {
public:
    explicit CrossEntropySumExpAndIndexLogitTiling(gert::TilingContext* context, bool isInplace = false)
        : tilingContext(context), isInplace_(isInplace) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

private:
    void TilingDataSet();
    void TilingDataPrint() const;
    bool CheckParam();
    bool CoreSplit();
    bool UbSplit();
    CrossEntropySumExpAndIndexLogitRegBaseTilingData* tilingData_{nullptr};
    gert::TilingContext* tilingContext = nullptr;
    bool isInplace_ = false;
    int32_t coreNum = 1;
    uint64_t ubSize = 0;
    uint64_t workspaceSize = 0;
    ge::DataType logitsType_ = ge::DT_FLOAT;
    ge::DataType globalMaxType_ = ge::DT_FLOAT;
    int64_t N_ = 0;
    int64_t vLocal_ = 0;
    int64_t vocabStart_ = 0;
    int64_t vocabEnd_ = 0;
    int64_t usedCores_ = 0;
    int64_t headCoreNum_ = 0;
    int64_t tokensPerCore_ = 0;
    int64_t tokensPerCoreTail_ = 0;
    int64_t headBlockNum_ = 0;
    int64_t tailBlockNum_ = 0;
    int64_t rowBlockMax_ = 0;
    int64_t reduceTmpBytes_ = 0;
    int64_t vTile_ = 0;
    int64_t vLoopNum_ = 0;
    int64_t lastVTile_ = 0;
    int64_t logitsRowStride_ = 0;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_ARCH35_CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_H
