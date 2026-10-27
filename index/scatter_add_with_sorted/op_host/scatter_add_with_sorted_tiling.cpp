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
 * \file scatter_add_with_sorted_tiling.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "scatter_add_with_sorted_tiling.h"
#include "op_host/tiling_util.h"
#include "index/scatter_add_with_sorted/op_host/scatter_add_with_sorted_tiling_base.h"

using namespace std;
using Ops::NN::Optiling::TilingRegistry;
using namespace AscendC;

namespace {
const int DT_FLOAT32_TYPE = 1;
const int DT_FLOAT16_TYPE = 2;
const int DT_INT32_TYPE = 3;
const int DT_UINT8_TYPE = 4;
const int DT_INT8_TYPE = 5;
const int DT_BF16_TYPE = 6;

const int ADD_MODE = 10;

const int SIZE_OF_FP16 = 2;
const int SIZE_OF_FP32 = 4;
const int SIZE_OF_INT32 = 4;
const int SIZE_OF_UINT8 = 1;
const int SIZE_OF_INT8 = 1;
const int SIZE_OF_BF16 = 2;

const int BUFFER_NUM = 2;
const int MAX_SIZE = 64;

const int INPUT_0 = 0;
const int INPUT_1 = 1;
const int INPUT_2 = 2;
const int INPUT_3 = 3;
} // namespace

namespace optiling {
class ScatterAddWithSortedTiling {
public:
    explicit ScatterAddWithSortedTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    void TilingDataPrint() const;

private:
    ScatterAddWithSortedTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    uint32_t tilingKey = 0;
    uint64_t usedCoreNum = 0;
    uint64_t extraTaskCore = 0;
    uint64_t eachCount = 1;
    uint64_t lastCount = 1;
    uint64_t inputCount = 1;
    uint64_t indicesCount = 1;
    uint64_t updatesCount = 1;
    uint64_t inputOneTime = 0;
    uint64_t updatesOneTime = 0;
    uint64_t updatesAlign = 0;
    uint64_t updatesLoop = 0;
    uint64_t updatesEach = 0;
    uint64_t updatesLast = 0;
    uint64_t eachNum = 0;
    uint64_t eachLoop = 0;
    uint64_t eachTail = 0;
    uint64_t lastNum = 0;
    uint64_t lastLoop = 0;
    uint64_t lastTail = 0;
    uint64_t maxSize = 0;
    uint64_t dataAlign = 32;
    uint64_t workspaceSize = 0;
    uint64_t max_ub = 20480;
};

ge::graphStatus ScatterAddWithSortedTiling::Init()
{
    if (tilingContext == nullptr) {
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {
            "ScatterAddWithSorted",
            "ScatterAddWithSortedTiling::Init: tiling context is null; check the registered tiling caller"};
        REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = tilingContext->GetCompileInfo<ScatterAddWithSortedCompileInfo>();
    if (compileInfo == nullptr) {
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {
            tilingContext->GetNodeName(),
            "ScatterAddWithSortedTiling::Init: compile info is null; check TilingParse registration and execution"};
        REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
        return ge::GRAPH_FAILED;
    }
    const uint32_t coreNum = static_cast<uint32_t>(compileInfo->totalCoreNum);
    if (coreNum == 0) {
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {
            tilingContext->GetNodeName(),
            "ScatterAddWithSortedTiling::Init: platform coreNum is 0; check platform information and TilingParse"};
        REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
        return ge::GRAPH_FAILED;
    }
    workspaceSize = compileInfo->workspaceSize;
    auto ubSizePlatForm = compileInfo->ubSizePlatForm;
    max_ub = ubSizePlatForm / max_ub * max_ub / BUFFER_NUM;
    OP_LOGD(tilingContext, "ubSizePlatForm: %lu.", ubSizePlatForm);

    auto attrs = tilingContext->GetAttrs();
    if (attrs == nullptr || tilingContext->GetRawTilingData() == nullptr) {
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {tilingContext->GetNodeName(),
                                                 "ScatterAddWithSortedTiling::Init: attributes or raw tiling storage "
                                                 "is null; check tiling context construction"};
        REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
        return ge::GRAPH_FAILED;
    }
    const char* inputNames[] = {"var", "value", "sorted_index"};
    for (size_t i = 0; i < sizeof(inputNames) / sizeof(inputNames[0]); ++i) {
        OP_CHECK_IF(tilingContext->GetInputShape(i) == nullptr || tilingContext->GetInputDesc(i) == nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(tilingContext->GetNodeName(), inputNames[i]), return ge::GRAPH_FAILED);
    }

    const char* reduce = attrs->GetAttrPointer<char>(0);
    OP_CHECK_IF(reduce == nullptr, OP_LOGE_WITH_INVALID_INPUT(tilingContext->GetNodeName(), "reduction"),
                return ge::GRAPH_FAILED);
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    auto valueDtype = tilingContext->GetInputDesc(INPUT_1)->GetDataType();
    uint32_t inputSize = 0;
    uint32_t isAdd = 0;
    if (strcmp(reduce, "add") == 0) {
        isAdd = 1;
    }

    if (ge::DT_FLOAT == inputDtype) {
        tilingKey = DT_FLOAT32_TYPE + isAdd * ADD_MODE;
        inputSize = SIZE_OF_FP32 + SIZE_OF_FP32 + SIZE_OF_FP32;
        dataAlign = dataAlign / SIZE_OF_FP32;
    } else if (ge::DT_FLOAT16 == inputDtype) {
        tilingKey = DT_FLOAT16_TYPE + isAdd * ADD_MODE;
        inputSize = SIZE_OF_FP16 + SIZE_OF_FP16 + SIZE_OF_FP16;
        dataAlign = dataAlign / SIZE_OF_FP16;
    } else if (ge::DT_INT32 == inputDtype) {
        tilingKey = DT_INT32_TYPE + isAdd * ADD_MODE;
        inputSize = SIZE_OF_INT32;
    } else if (ge::DT_UINT8 == inputDtype) {
        tilingKey = DT_UINT8_TYPE + isAdd * ADD_MODE;
        inputSize = SIZE_OF_UINT8;
    } else if (ge::DT_INT8 == inputDtype) {
        tilingKey = DT_INT8_TYPE + isAdd * ADD_MODE;
        inputSize = SIZE_OF_INT8;
    } else if (ge::DT_BF16 == inputDtype) {
        tilingKey = DT_BF16_TYPE + isAdd * ADD_MODE;
        inputSize = SIZE_OF_BF16 + SIZE_OF_BF16 + SIZE_OF_FP16;
        dataAlign = dataAlign / SIZE_OF_BF16;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            tilingContext->GetNodeName(), "var", Ops::Base::ToString(inputDtype).c_str(),
            "supported dtypes on this tiling path are FLOAT, FLOAT16, INT32, UINT8, INT8 and BF16");
        return ge::GRAPH_FAILED;
    }

    if (inputDtype != valueDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            tilingContext->GetNodeName(), "var, value",
            (Ops::Base::ToString(inputDtype) + ", " + Ops::Base::ToString(valueDtype)).c_str(),
            "var and value must have the same dtype");
        return ge::GRAPH_FAILED;
    }

    if (ge::DT_INT32 != tilingContext->GetInputDesc(INPUT_2)->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "sorted_index",
                                  Ops::Base::ToString(tilingContext->GetInputDesc(INPUT_2)->GetDataType()).c_str(),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }

    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();

    auto inputDimNum = inputShape.GetDimNum();
    if (inputDimNum != updatesShape.GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            tilingContext->GetNodeName(), "var, value",
            (std::to_string(inputDimNum) + ", " + std::to_string(updatesShape.GetDimNum())).c_str(),
            "var and value must have the same number of dimensions on this tiling path");
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < inputDimNum; ++i) {
        auto dimInput = inputShape.GetDim(i);
        auto dimUpdates = updatesShape.GetDim(i);
        if (i == inputDimNum - 1) {
            inputOneTime = dimInput;
            updatesOneTime = dimUpdates;
        }
        inputCount *= dimInput;
        updatesCount *= dimUpdates;
    }
    indicesCount = indicesShape.GetDim(0);
    if (inputOneTime == 0 || updatesOneTime == 0 || inputCount == 0 || updatesCount == 0 || indicesCount == 0) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            tilingContext->GetNodeName(), "var, value, sorted_index",
            (Ops::Base::ToString(inputShape) + ", " + Ops::Base::ToString(updatesShape) + ", " +
             Ops::Base::ToString(indicesShape))
                .c_str(),
            "var, value and sorted_index must be nonempty on this tiling path");
        return ge::GRAPH_FAILED;
    }

    uint32_t inputDataAlign = dataAlign / inputSize;

    // atomic_add scene，self dtype in int32, int8, uint8
    if (tilingContext->GetInputDesc(INPUT_3) == nullptr) {
        if (indicesCount < coreNum) {
            usedCoreNum = indicesCount;
            eachNum = 1;
        } else {
            usedCoreNum = coreNum;
            eachNum = indicesCount / coreNum;
            extraTaskCore = indicesCount - eachNum * coreNum;
        }
        updatesEach = max_ub / inputSize;
        updatesEach = updatesEach > updatesOneTime ? updatesOneTime : updatesEach;
        updatesLoop = (updatesOneTime - 1) / updatesEach + 1;
        updatesLast = updatesOneTime - updatesEach * (updatesLoop - 1);
        updatesAlign = ((updatesEach - 1) / inputDataAlign + 1) * inputDataAlign;

        OP_LOGD(tilingContext, "Tiling inited.");
        return ge::GRAPH_SUCCESS;
    }

    if (ge::DT_INT32 != tilingContext->GetInputDesc(INPUT_3)->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "pos",
                                  Ops::Base::ToString(tilingContext->GetInputDesc(INPUT_3)->GetDataType()).c_str(),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }

    inputSize += (SIZE_OF_FP32 + SIZE_OF_FP32) / BUFFER_NUM;
    updatesAlign = (updatesOneTime + dataAlign - 1) / dataAlign * dataAlign;
    if (updatesAlign * inputSize >= max_ub) {
        maxSize = MAX_SIZE;
        max_ub = max_ub - maxSize * (SIZE_OF_INT32 + SIZE_OF_INT32);
        updatesEach = max_ub / inputSize;
        updatesEach = updatesEach > updatesOneTime ? updatesOneTime : updatesEach;
        updatesLoop = (updatesOneTime - 1) / updatesEach + 1;
        updatesLast = updatesOneTime - updatesEach * (updatesLoop - 1);
        updatesAlign = (updatesEach + dataAlign - 1) / dataAlign * dataAlign;
    } else {
        maxSize = (max_ub - updatesAlign * inputSize) / (SIZE_OF_INT32 + SIZE_OF_INT32);
        updatesEach = updatesLast = updatesOneTime;
        updatesLoop = 1;
    }

    usedCoreNum = indicesCount < coreNum ? indicesCount : coreNum;
    eachCount = (indicesCount + usedCoreNum - 1) / usedCoreNum;
    usedCoreNum = (indicesCount + eachCount - 1) / eachCount;
    lastCount = indicesCount - eachCount * (usedCoreNum - 1);

    eachNum = eachCount;
    eachLoop = 1;
    eachTail = eachCount;
    if (eachCount > maxSize) {
        eachNum = maxSize;
        eachLoop = (eachCount + maxSize - 1) / maxSize;
        eachTail = eachCount - (eachLoop - 1) * eachNum;
    }

    lastNum = lastCount;
    lastLoop = 1;
    lastTail = lastCount;
    if (lastCount > maxSize) {
        lastNum = maxSize;
        lastLoop = (lastCount + maxSize - 1) / maxSize;
        lastTail = lastCount - (lastLoop - 1) * lastNum;
    }

    OP_LOGD(tilingContext, "Tiling inited.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddWithSortedTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext, "Tiling start.");

    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_extraTaskCore(extraTaskCore);
    tilingData.set_eachCount(eachCount);
    tilingData.set_lastCount(lastCount);
    tilingData.set_inputCount(inputCount);
    tilingData.set_indicesCount(indicesCount);
    tilingData.set_updatesCount(updatesCount);
    tilingData.set_inputOneTime(inputOneTime);
    tilingData.set_updatesOneTime(updatesOneTime);
    tilingData.set_updatesAlign(updatesAlign);
    tilingData.set_maxSize(maxSize);
    tilingData.set_eachNum(eachNum);
    tilingData.set_eachLoop(eachLoop);
    tilingData.set_eachTail(eachTail);
    tilingData.set_lastNum(lastNum);
    tilingData.set_lastLoop(lastLoop);
    tilingData.set_lastTail(lastTail);
    tilingData.set_updatesLoop(updatesLoop);
    tilingData.set_updatesEach(updatesEach);
    tilingData.set_updatesLast(updatesLast);
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(usedCoreNum);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {
            tilingContext->GetNodeName(), "ScatterAddWithSortedTiling::RunKernelTiling: workspace-size storage is "
                                          "null; expected one size slot; check tiling context construction"};
        REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = workspaceSize;
    TilingDataPrint();
    OP_LOGD(tilingContext, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

void ScatterAddWithSortedTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext, "usedCoreNum: %lu.", usedCoreNum);
    OP_LOGD(tilingContext, "extraTaskCore: %lu.", extraTaskCore);
    OP_LOGD(tilingContext, "eachCount: %lu.", eachCount);
    OP_LOGD(tilingContext, "lastCount: %lu.", lastCount);
    OP_LOGD(tilingContext, "inputCount: %lu.", inputCount);
    OP_LOGD(tilingContext, "indicesCount: %lu.", indicesCount);
    OP_LOGD(tilingContext, "updatesCount: %lu.", updatesCount);
    OP_LOGD(tilingContext, "inputOneTime: %lu.", inputOneTime);
    OP_LOGD(tilingContext, "updatesOneTime: %lu.", updatesOneTime);
    OP_LOGD(tilingContext, "updatesAlign: %lu.", updatesAlign);
    OP_LOGD(tilingContext, "maxSize: %lu.", maxSize);
    OP_LOGD(tilingContext, "eachNum: %lu.", eachNum);
    OP_LOGD(tilingContext, "eachLoop: %lu.", eachLoop);
    OP_LOGD(tilingContext, "eachTail: %lu.", eachTail);
    OP_LOGD(tilingContext, "lastNum: %lu.", lastNum);
    OP_LOGD(tilingContext, "lastLoop: %lu.", lastLoop);
    OP_LOGD(tilingContext, "lastTail: %lu.", lastTail);
    OP_LOGD(tilingContext, "updatesLoop: %lu.", updatesLoop);
    OP_LOGD(tilingContext, "updatesEach: %lu.", updatesEach);
    OP_LOGD(tilingContext, "updatesLast: %lu.", updatesLast);
    OP_LOGD(tilingContext, "tilingKey: %u.", tilingKey);
    OP_LOGD(tilingContext, "max_ub: %lu.", max_ub);
}

ge::graphStatus ScatterAddWithSortedTilingForAscendC(gert::TilingContext* context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingScatterAddWithSorted(gert::TilingContext* context)
{
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGD(context->GetNodeName(), "ScatterAddWithSorted is ascendc. running ascendc tiling.");
        return ScatterAddWithSortedTilingForAscendC(context);
    }

    ScatterAddWithSortedTiling tilingObject(context);
    const auto ret = tilingObject.Init();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "ScatterAddWithSortedTiling::Init failed, rc=%u; propagate the first error.",
                               ret);
        return ret;
    }
    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepareForScatterAddWithSorted(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForScatterAddWithSorted start.");
    auto compileInfo = context->GetCompiledInfo<ScatterAddWithSortedCompileInfo>();
    if (compileInfo == nullptr) {
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {context->GetNodeName(),
                                                 "TilingPrepareForScatterAddWithSorted: compile info storage is null; "
                                                 "check TilingParse context construction"};
        REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {
            context->GetNodeName(),
            "TilingPrepareForScatterAddWithSorted: platform info is null; check platform initialization"};
        REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    if (compileInfo->ubSizePlatForm <= 0) {
        const std::string reason = "TilingPrepareForScatterAddWithSorted: UB size is " +
                                   std::to_string(compileInfo->ubSizePlatForm) +
                                   "; expected > 0 bytes; check platform initialization";
        const std::vector<const char*> keys = {"opName", "reason"};
        const std::vector<const char*> values = {context->GetNodeName(), reason.c_str()};
        REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context, "total_ub_size is %lu.", totalUbSize);
    OP_LOGD(context, "TilingPrepareForScatterAddWithSorted end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScatterAddWithSorted)
    .Tiling(TilingScatterAddWithSorted)
    .TilingParse<ScatterAddWithSortedCompileInfo>(TilingPrepareForScatterAddWithSorted);
} // namespace optiling
