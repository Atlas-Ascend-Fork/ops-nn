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
 * \file anti_hi_f4_quant_tail_tiling_arch35.cpp
 * \brief
 */
#include "anti_hi_f4_quant_tiling_arch35.h"
#include "../../op_kernel/arch35/anti_hi_f4_quant_tilingdata.h"
#include "../../op_kernel/arch35/anti_hi_f4_quant_struct.h"
#include <cmath>
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "error_util.h"

using namespace std;
using namespace ge;
using namespace AscendC;
using namespace AntiHiF4QuantOp;
namespace optiling {

// ========== 常量（对标 DynamicHiF4Quant） ==========
constexpr int64_t N_BUFFER = 2;
constexpr int64_t BLOCK_SIZE = 64;

constexpr int64_t RESERVED_UB_SIZE = 4096;
constexpr int64_t HIFP4_PER_CORE_MIN_GROUP_NUM = 32;
constexpr int64_t HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM = 8;
constexpr int64_t HIFP4_OUTPUT_REG_ONELOOP = 2;
constexpr int64_t HIFP4_DB_NUM = 2;

// 反量化每 group 的 UB 字节消耗（对标 DynamicHiF4Quant 的 HIFP4_ONE_GROUP_*）
// x: 32B (64 fp4 packed), scale: 4B (1 float), y: 128B (64 BF16), scaleTmp: 0B (不需要额外中间buffer)
constexpr int64_t HIFP4_ONE_GROUP_INPUT_BYTES = 32;
constexpr int64_t HIFP4_ONE_GROUP_SCALE_BYTES = 4;
constexpr int64_t HIFP4_ONE_GROUP_OUTPUT_BYTES = 128;
constexpr int64_t HIFP4_ONE_GROUP_MID_UB_RES_BYTES = 64; // 反量化不需要中间 maxper4/maxper8/e8 等中间buffer

void AntiHiF4QuantTailTiling::splitCoreTail()
{
    tilingParams_.groupNum = Ops::Base::CeilDiv(tilingParams_.xDimNeg1 * tilingParams_.xDimNeg2, BLOCK_SIZE);

    tilingParams_.usedCoreNum = Ops::Base::CeilDiv(tilingParams_.groupNum, HIFP4_PER_CORE_MIN_GROUP_NUM);
    tilingParams_.usedCoreNum = tilingParams_.usedCoreNum > tilingParams_.totalCoreNum ? tilingParams_.totalCoreNum :
                                                                                         tilingParams_.usedCoreNum;

    int64_t ubReserveAlignBuf = RESERVED_UB_SIZE * 4;
    int64_t ubCap = (tilingParams_.ubSize - ubReserveAlignBuf) / HIFP4_DB_NUM /
                    (HIFP4_ONE_GROUP_MID_UB_RES_BYTES + HIFP4_ONE_GROUP_INPUT_BYTES + HIFP4_ONE_GROUP_SCALE_BYTES +
                     HIFP4_ONE_GROUP_OUTPUT_BYTES);
    tilingParams_.ubFormer = ubCap / HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM * HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM;

    tilingParams_.blockFormer = Ops::Base::CeilDiv(tilingParams_.groupNum, tilingParams_.usedCoreNum);
    tilingParams_.usedCoreNum = Ops::Base::CeilDiv(tilingParams_.groupNum, tilingParams_.blockFormer);
    tilingParams_.blockTail = tilingParams_.groupNum - (tilingParams_.usedCoreNum - 1) * tilingParams_.blockFormer;

    tilingParams_.ubLoopOfFormerBlock = Ops::Base::CeilDiv(tilingParams_.blockFormer, tilingParams_.ubFormer);
    tilingParams_.ubLoopOfTailBlock = Ops::Base::CeilDiv(tilingParams_.blockTail, tilingParams_.ubFormer);
    tilingParams_.ubTailOfFormerBlock = tilingParams_.blockFormer -
                                        (tilingParams_.ubLoopOfFormerBlock - 1) * tilingParams_.ubFormer;
    tilingParams_.ubTailOfTailBlock = tilingParams_.blockTail -
                                      (tilingParams_.ubLoopOfTailBlock - 1) * tilingParams_.ubFormer;
}

ge::graphStatus AntiHiF4QuantTailTiling::SetTilingData()
{
    tilingData_.totalCoreNum = tilingParams_.totalCoreNum;
    tilingData_.usedCoreNum = tilingParams_.usedCoreNum;
    tilingData_.tilingKey = tilingParams_.tilingKey;

    tilingData_.groupNum = tilingParams_.blockFormer;
    tilingData_.alignGroupNum = tilingParams_.ubFormer;
    tilingData_.rowPerHeadCore = tilingParams_.ubLoopOfFormerBlock;
    tilingData_.rowPerTailCore = tilingParams_.ubLoopOfTailBlock;
    tilingData_.multiRowNumHeadCore = tilingParams_.ubTailOfFormerBlock;
    tilingData_.multiRowNumTailCore = tilingParams_.ubTailOfTailBlock;

    tilingData_.scaleLoopEle = Ops::Base::CeilDiv(tilingParams_.ubFormer, HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM);
    tilingData_.scaleLoopTimes = Ops::Base::CeilDiv(tilingParams_.ubTailOfFormerBlock,
                                                    HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM);
    tilingData_.scaleLoopTail = Ops::Base::CeilDiv(tilingParams_.ubTailOfTailBlock, HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM);
    tilingData_.outLoopEle = Ops::Base::CeilDiv(tilingParams_.ubFormer, HIFP4_OUTPUT_REG_ONELOOP);
    tilingData_.outLoopTimes = Ops::Base::CeilDiv(tilingParams_.ubTailOfFormerBlock, HIFP4_OUTPUT_REG_ONELOOP);
    tilingData_.outLoopTail = Ops::Base::CeilDiv(tilingParams_.ubTailOfTailBlock, HIFP4_OUTPUT_REG_ONELOOP);
    tilingData_.dstType = tilingParams_.dstType;

    uint64_t tilingDataSize = sizeof(tilingData_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    auto rawTilingData = context_->GetRawTilingData();
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                           reinterpret_cast<void*>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

void AntiHiF4QuantTailTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(),
            "AntiHiF4Quant Tiling: totalCoreNum=%ld, usedCoreNum=%ld, tilingKey=%ld, "
            "groupNum=%ld, alignGroupNum=%ld, rowPerHeadCore=%ld, rowPerTailCore=%ld, "
            "multiRowNumHeadCore=%ld, multiRowNumTailCore=%ld, "
            "scaleLoopEle=%ld, scaleLoopTimes=%ld, scaleLoopTail=%ld, "
            "outLoopEle=%ld, outLoopTimes=%ld, outLoopTail=%ld, dstType=%ld",
            tilingData_.totalCoreNum, tilingData_.usedCoreNum, tilingData_.tilingKey, tilingData_.groupNum,
            tilingData_.alignGroupNum, tilingData_.rowPerHeadCore, tilingData_.rowPerTailCore,
            tilingData_.multiRowNumHeadCore, tilingData_.multiRowNumTailCore, tilingData_.scaleLoopEle,
            tilingData_.scaleLoopTimes, tilingData_.scaleLoopTail, tilingData_.outLoopEle, tilingData_.outLoopTimes,
            tilingData_.outLoopTail, tilingData_.dstType);
}

ge::graphStatus AntiHiF4QuantTailTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "Enter AntiHiF4QuantTailTiling DoTiling.");

    splitCoreTail();

    OP_CHECK_IF(SetTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "The tilingData_ set failed."),
                return ge::GRAPH_FAILED);
    PrintTilingData();

    OP_LOGD(context_->GetNodeName(), "Tiling usedCoreNum is %ld.", tilingParams_.usedCoreNum);
    context_->SetBlockDim(tilingParams_.usedCoreNum);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
