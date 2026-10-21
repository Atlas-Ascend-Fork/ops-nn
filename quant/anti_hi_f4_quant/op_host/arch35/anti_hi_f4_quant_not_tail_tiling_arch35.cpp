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
 * \file anti_hi_f4_quant_not_tail_tiling_arch35.cpp
 * \brief
 */

#include "anti_hi_f4_quant_tiling_arch35.h"
#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "../../op_kernel/arch35/anti_hi_f4_quant_tilingdata.h"
#include "../../op_kernel/arch35/anti_hi_f4_quant_struct.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace AntiHiF4QuantOp;

namespace optiling {

constexpr int64_t BLOCK_SIZE = 64;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t X_NUM_OF_ONE_BYTE = 2;
constexpr int64_t BYTE_OF_INPUT_SCALE = 4;
constexpr int64_t BYTE_OF_OUTPUT_Y = 2;

constexpr int64_t N_BUFFER = 2;

void AntiHiF4QuantNotTailTiling::CalcMaxUbAvailable(const int64_t blockH, const int64_t blockW)
{
    int64_t xUbSize = blockH * blockW / X_NUM_OF_ONE_BYTE * N_BUFFER;
    int64_t scaleUbSize = blockW * BYTE_OF_INPUT_SCALE * N_BUFFER;
    int64_t yUbSize = blockH * blockW * BYTE_OF_OUTPUT_Y * N_BUFFER;

    tilingParams_.ubBlockNum = tilingParams_.ubSize / (xUbSize + scaleUbSize + yUbSize);
}

void AntiHiF4QuantNotTailTiling::SplitCoreNotTail()
{
    tilingParams_.splitBlockColSize = tilingParams_.ubBlockNum * BLOCK_SIZE;

    tilingParams_.splitBlockColNum = Ops::Base::CeilDiv(tilingParams_.xDimNeg2, tilingParams_.splitBlockColSize);
    tilingParams_.splitBlockRowNum = Ops::Base::CeilDiv(tilingParams_.xDimNeg1, tilingParams_.splitBlockRowSize);

    tilingParams_.totalTaskNum = tilingParams_.xDimNeg3 * tilingParams_.splitBlockColNum *
                                 tilingParams_.splitBlockRowNum;
}

ge::graphStatus AntiHiF4QuantNotTailTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter AntiHiF4QuantNotTailTiling DoTiling.");

    tilingParams_.splitBlockRowSize = tilingParams_.vfLen / BYTE_OF_OUTPUT_Y;
    CalcMaxUbAvailable(BLOCK_SIZE, tilingParams_.splitBlockRowSize);
    SplitCoreNotTail();
    while (tilingParams_.totalTaskNum < tilingParams_.totalCoreNum && tilingParams_.ubBlockNum > 1) {
        tilingParams_.ubBlockNum = Ops::Base::CeilDiv(tilingParams_.ubBlockNum, NUM_TWO);
        SplitCoreNotTail();
    }
    tilingParams_.tailBlockRowSize = tilingParams_.xDimNeg1 % tilingParams_.splitBlockRowSize == 0 ?
                                         tilingParams_.splitBlockRowSize :
                                         tilingParams_.xDimNeg1 % tilingParams_.splitBlockRowSize;
    tilingParams_.tailBlockColSize = tilingParams_.xDimNeg2 % tilingParams_.splitBlockColSize == 0 ?
                                         tilingParams_.splitBlockColSize :
                                         tilingParams_.xDimNeg2 % tilingParams_.splitBlockColSize;
    tilingParams_.usedCoreNum = min(tilingParams_.totalCoreNum, tilingParams_.totalTaskNum);
    tilingParams_.headCoreLoopNum = Ops::Base::CeilDiv(tilingParams_.totalTaskNum, tilingParams_.usedCoreNum);
    tilingParams_.tailCoreLoopNum = tilingParams_.totalTaskNum / tilingParams_.usedCoreNum;
    tilingParams_.headCoreNum = tilingParams_.totalTaskNum % tilingParams_.usedCoreNum;
    tilingParams_.taskNumPerBatch = tilingParams_.splitBlockColNum * tilingParams_.splitBlockRowNum;

    OP_CHECK_IF(SetTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The tilingData_ set failed."),
                return ge::GRAPH_FAILED);

    PrintTilingData();

    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %ld.", tilingParams_.usedCoreNum);
    context_->SetBlockDim(tilingParams_.usedCoreNum);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParams_.workspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AntiHiF4QuantNotTailTiling::SetTilingData()
{
    tilingData_.tilingKey = tilingParams_.tilingKey;
    tilingData_.totalCoreNum = tilingParams_.totalCoreNum;
    tilingData_.usedCoreNum = tilingParams_.usedCoreNum;
    tilingData_.dstType = tilingParams_.dstType;
    tilingData_.xDimNeg1 = tilingParams_.xDimNeg1;
    tilingData_.xDimNeg2 = tilingParams_.xDimNeg2;
    tilingData_.xDimNeg3 = tilingParams_.xDimNeg3;
    tilingData_.splitBlockColSize = tilingParams_.splitBlockColSize;
    tilingData_.splitBlockRowSize = tilingParams_.splitBlockRowSize;
    tilingData_.tailBlockRowSize = tilingParams_.tailBlockRowSize;
    tilingData_.tailBlockColSize = tilingParams_.tailBlockColSize;
    tilingData_.splitBlockColNum = tilingParams_.splitBlockColNum;
    tilingData_.splitBlockRowNum = tilingParams_.splitBlockRowNum;
    tilingData_.headCoreLoopNum = tilingParams_.headCoreLoopNum;
    tilingData_.tailCoreLoopNum = tilingParams_.tailCoreLoopNum;
    tilingData_.headCoreNum = tilingParams_.headCoreNum;
    tilingData_.totalTaskNum = tilingParams_.totalTaskNum;
    tilingData_.ubBlockNum = tilingParams_.ubBlockNum;
    tilingData_.taskNumPerBatch = tilingParams_.taskNumPerBatch;

    uint64_t tilingDataSize = sizeof(tilingData_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    auto rawTilingData = context_->GetRawTilingData();
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

void AntiHiF4QuantNotTailTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(),
            "TilingData tilingKey: %ld, totalCoreNum: %ld, usedCoreNum: %ld, dstType: %ld, xDimNeg1: %ld, "
            "xDimNeg2: %ld, xDimNeg3: %ld, splitBlockColSize: %ld, splitBlockRowSize: %ld, tailBlockRowSize: %ld, "
            "tailBlockColSize: %ld, splitBlockColNum: %ld, splitBlockRowNum: %ld, "
            "headCoreLoopNum: %ld, tailCoreLoopNum: %ld, headCoreNum: %ld, "
            "totalTaskNum: %ld, ubBlockNum: %ld, taskNumPerBatch: %ld",
            tilingData_.tilingKey, tilingData_.totalCoreNum, tilingData_.usedCoreNum, tilingData_.dstType,
            tilingData_.xDimNeg1, tilingData_.xDimNeg2, tilingData_.xDimNeg3, tilingData_.splitBlockColSize,
            tilingData_.splitBlockRowSize, tilingData_.tailBlockRowSize, tilingData_.tailBlockColSize,
            tilingData_.splitBlockColNum, tilingData_.splitBlockRowNum, tilingData_.headCoreLoopNum,
            tilingData_.tailCoreLoopNum, tilingData_.headCoreNum, tilingData_.totalTaskNum, tilingData_.ubBlockNum,
            tilingData_.taskNumPerBatch);
}

} // namespace optiling
