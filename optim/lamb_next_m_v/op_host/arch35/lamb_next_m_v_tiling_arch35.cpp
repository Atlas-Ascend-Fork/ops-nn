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
 * \file lamb_next_m_v_tiling_arch35.cpp
 * \brief lamb_next_m_v_tiling_arch35 source file
 */

#include "lamb_next_m_v_tiling_arch35.h"
#include <graph/utils/type_utils.h>
#include <securec.h>
#include <algorithm>
#include <cstdint>
#include <vector>
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace ge;

namespace optiling {

constexpr static uint64_t LAMB_NEXT_M_V_TILING_PRIORITY = 0;
constexpr static uint64_t LNMV_TILING_KEY_FP32 = 100;
constexpr static uint64_t LNMV_TILING_KEY_FP16 = 200;
constexpr static uint64_t TILING_KEY_FP32 = 100;
constexpr static uint64_t TILING_KEY_FP16 = 200;
constexpr static int32_t INPUT_NUM = 13;
constexpr static int32_t OUTPUT_NUM = 4;

namespace {
// 形状边界/溢出防护(算子内)。
// 这些校验原先写在共用头 optim/lamb_apply_common/op_host/arch35/lamb_brc_tiling_plan.h 内,
// 会影响 lamb 族全部算子; 为把影响收敛在本算子目录内, 改为在 LambNextMV host 侧前置校验,
// 仍沿用原公共接口 BuildLambBrcPlan<13,4>() 完成规划。折叠分组与共用实现保持一致, 逐轴校验:
// rank 上限、uint32 表示域、折叠乘法 int64 不溢出、元素总数 uint64 不溢出; 任一越界即拒绝,
// 避免 td.outShape/td.inShape 静默截断与有符号溢出 UB。
ge::graphStatus CheckLambNextMVBrcBounds(gert::TilingContext* context, uint32_t nin)
{
    auto outShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShapePtr);
    const gert::Shape& outShape = outShapePtr->GetStorageShape();
    size_t rank = outShape.GetDimNum();
    OP_CHECK_IF(
        rank > LAMB_BRC_MAX_DIM,
        OP_LOGE(context->GetNodeName(), "output rank %zu exceeds max supported rank %u", rank, LAMB_BRC_MAX_DIM),
        return ge::GRAPH_FAILED);

    std::vector<int64_t> rawOut(rank);
    for (size_t d = 0; d < rank; d++) {
        rawOut[d] = outShape.GetDim(d);
        OP_CHECK_IF(rawOut[d] < 0 || static_cast<uint64_t>(rawOut[d]) > UINT32_MAX,
                    OP_LOGE(context->GetNodeName(), "output dim %zu (%ld) exceeds uint32 range", d, rawOut[d]),
                    return ge::GRAPH_FAILED);
    }

    std::vector<std::vector<int64_t>> rawIn(nin, std::vector<int64_t>(rank, 1));
    for (uint32_t i = 0; i < nin; i++) {
        auto sp = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, sp);
        const gert::Shape& s = sp->GetStorageShape();
        size_t r = s.GetDimNum();
        OP_CHECK_IF(r > rank, OP_LOGE(context->GetNodeName(), "input %u rank %zu exceeds output rank %zu", i, r, rank),
                    return ge::GRAPH_FAILED);
        for (size_t d = 0; d < r; d++) { // 右对齐补维
            rawIn[i][rank - r + d] = s.GetDim(d);
        }
    }

    std::vector<int64_t> co;
    std::vector<std::vector<int64_t>> ci(nin);
    for (size_t d = 0; d < rank; d++) {
        if (rawOut[d] == 1) {
            continue;
        }
        bool merge = !co.empty();
        for (uint32_t i = 0; i < nin && merge; i++) {
            merge = ((ci[i].back() == 1) == (rawIn[i][d] == 1));
        }
        if (merge) {
            OP_CHECK_IF((rawOut[d] != 0 && co.back() > INT64_MAX / rawOut[d]) ||
                            (co.back() != 0 && rawOut[d] > INT64_MAX / co.back()),
                        OP_LOGE(context->GetNodeName(), "collapsed output dim (%ld x %ld) exceeds int64 range",
                                co.back(), rawOut[d]),
                        return ge::GRAPH_FAILED);
            co.back() *= rawOut[d];
            for (uint32_t i = 0; i < nin; i++) {
                OP_CHECK_IF((rawIn[i][d] != 0 && ci[i].back() > INT64_MAX / rawIn[i][d]) ||
                                (ci[i].back() != 0 && rawIn[i][d] > INT64_MAX / ci[i].back()),
                            OP_LOGE(context->GetNodeName(), "collapsed input %u dim exceeds int64 range", i),
                            return ge::GRAPH_FAILED);
                ci[i].back() *= rawIn[i][d];
            }
        } else {
            co.push_back(rawOut[d]);
            for (uint32_t i = 0; i < nin; i++) {
                ci[i].push_back(rawIn[i][d]);
            }
        }
    }
    if (co.empty()) { // 全 1 / 标量输出
        co.push_back(1);
        for (uint32_t i = 0; i < nin; i++) {
            ci[i].push_back(1);
        }
    }

    uint64_t total = 1;
    for (size_t d = 0; d < co.size(); d++) {
        OP_CHECK_IF(co[d] < 0 || static_cast<uint64_t>(co[d]) > UINT32_MAX,
                    OP_LOGE(context->GetNodeName(), "collapsed output dim %zu (%ld) exceeds uint32 range", d, co[d]),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(total != 0 && static_cast<uint64_t>(co[d]) > UINT64_MAX / total,
                    OP_LOGE(context->GetNodeName(), "total element count exceeds uint64 range"),
                    return ge::GRAPH_FAILED);
        total *= static_cast<uint64_t>(co[d]);
    }
    for (uint32_t i = 0; i < nin; i++) {
        for (size_t d = 0; d < ci[i].size(); d++) {
            OP_CHECK_IF(ci[i][d] < 0 || static_cast<uint64_t>(ci[i][d]) > UINT32_MAX,
                        OP_LOGE(context->GetNodeName(), "collapsed dim %zu of input %u (%ld) exceeds uint32 range", d,
                                i, ci[i][d]),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus TilingPrepareForLambNextMV(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<LambNextMVCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LambNextMVTiling::GetShapeAttrsInfo()
{
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    static const char* kInputNames[] = {"input_mul3",     "input_mul2", "input_realdiv1", "input_mul1", "input_mul0",
                                        "input_realdiv0", "input_mul4", "mul0_x",         "mul1_sub",   "mul2_x",
                                        "mul3_sub1",      "mul4_x",     "add2_y"};
    static const char* kOutputNames[] = {"y1", "y2", "y3", "y4"};
    for (int32_t inputIdx = 1; inputIdx < INPUT_NUM; inputIdx++) {
        auto inputDesc = context_->GetInputDesc(inputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);

        auto curDtype = inputDesc->GetDataType();
        if (curDtype != input0DType) {
            std::string paramNames = std::string(kInputNames[inputIdx]) + " and input0";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context_->GetNodeName(), paramNames.c_str(),
                (Ops::Base::ToString(curDtype) + " and " + Ops::Base::ToString(input0DType)).c_str(),
                "Their dtypes should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    for (int32_t outputIdx = 0; outputIdx < OUTPUT_NUM; outputIdx++) {
        auto outputDesc = context_->GetOutputDesc(outputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);

        auto curDtype = outputDesc->GetDataType();
        if (curDtype != input0DType) {
            std::string paramNames = std::string(kOutputNames[outputIdx]) + " and input0";
            std::string incorrectDtypes = Ops::Base::ToString(curDtype) + " and " + Ops::Base::ToString(input0DType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), paramNames.c_str(), incorrectDtypes.c_str(),
                                                   "Their dtypes should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool LambNextMVTiling::IsCapable() { return true; }

ge::graphStatus LambNextMVTiling::DoOpTiling()
{
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);

    ge::DataType input0DType = input0Desc->GetDataType();
    uint32_t dtSize = 0;
    if (input0DType == ge::DT_FLOAT) {
        tilingKey = TILING_KEY_FP32;
        dtSize = sizeof(float);
    } else if (input0DType == ge::DT_FLOAT16) {
        tilingKey = TILING_KEY_FP16;
        dtSize = sizeof(uint16_t);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "input_mul3", Ops::Base::ToString(input0DType).c_str(),
                                  "fp16 or fp32");
        return ge::GRAPH_FAILED;
    }

    using PlanTiling = LambBrcTilingData<13, 4>;
    td_ = PlanTiling{};
    // 空进空出: 输出 0 元素时 tiling 全 0, kernel 按 usedCoreNum=0 直接退出。
    auto outShape0 = context_->GetOutputShape(0);
    if (outShape0 == nullptr || outShape0->GetStorageShape().GetShapeSize() != 0) {
        // 先做本算子内的形状边界/溢出前置校验(原共用头的校验迁移至此), 再走原公共规划接口。
        OP_CHECK_IF(CheckLambNextMVBrcBounds(context_, static_cast<uint32_t>(INPUT_NUM)) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "broadcast shape bounds check failed"), return ge::GRAPH_FAILED);
        // 先取返回值再判: 模板实参里的逗号会被预处理器当成 OP_CHECK_IF 的参数分隔符。
        ge::graphStatus planRet = BuildLambBrcPlan<13, 4>(context_, coreNum_, ubSize_, dtSize, td_);
        OP_CHECK_IF(planRet != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "build broadcast plan failed"),
                    return ge::GRAPH_FAILED);
    }

    auto ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &td_, sizeof(td_));
    OP_CHECK_IF(ret != EOK, OP_LOGE(context_->GetNodeName(), "copy tiling data failed, ret %d", ret),
                return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    context_->SetBlockDim(std::max<uint32_t>(td_.usedCoreNum, 1));
    size_t* ws = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ws);
    ws[0] = 0U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LambNextMVTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t LambNextMVTiling::GetTilingKey() const { return tilingKey; }

ge::graphStatus LambNextMVTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LambNextMVTiling::PostTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LambNextMVTiling::GetPlatformInfo()
{
    auto compileInfo = static_cast<const LambNextMVCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF(coreNum_ == 0 || ubSize_ == 0,
                OP_LOGE(context_->GetNodeName(), "invalid platform info: coreNum %lu ubSize %lu", coreNum_, ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForLambNextMV(gert::TilingContext* context)
{
    OP_LOGD("LambNextMVTiling", "Enter TilingForLambNextMV");
    if (context == nullptr) {
        OP_LOGE("LambNextMVTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = static_cast<const LambNextMVCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc LambNextMVTiling");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

IMPL_OP_OPTILING(LambNextMV).Tiling(TilingForLambNextMV).TilingParse<LambNextMVCompileInfo>(TilingPrepareForLambNextMV);

REGISTER_OPS_TILING_TEMPLATE(LambNextMV, LambNextMVTiling, LAMB_NEXT_M_V_TILING_PRIORITY);
} // namespace optiling
