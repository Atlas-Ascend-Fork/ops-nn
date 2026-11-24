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
 * \file multi_add_rms_norm_dynamic_quant_tiling.h
 * \brief
 */
#ifndef OPS_NORM_MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_OP_HOST_H_
#define OPS_NORM_MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_OP_HOST_H_

#include <string>
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {
using std::string;
BEGIN_TILING_DATA_DEF(MultiAddRmsNormDynamicQuantTilingData)
TILING_DATA_FIELD_DEF(uint64_t, useCore);
TILING_DATA_FIELD_DEF(uint64_t, numFirstDim);
TILING_DATA_FIELD_DEF(uint64_t, numLastDim);
TILING_DATA_FIELD_DEF(uint64_t, numLastDimAligned);
TILING_DATA_FIELD_DEF(uint64_t, firstDimPerCore);
TILING_DATA_FIELD_DEF(uint64_t, firstDimPerCoreTail);
TILING_DATA_FIELD_DEF(uint64_t, firstDimPerLoop);
TILING_DATA_FIELD_DEF(uint64_t, lastDimLoopNum);
TILING_DATA_FIELD_DEF(uint64_t, lastDimSliceLen);
TILING_DATA_FIELD_DEF(uint64_t, lastDimSliceLenTail);
TILING_DATA_FIELD_DEF(uint32_t, smoothNum);
TILING_DATA_FIELD_DEF(uint32_t, x1Num);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MultiAddRmsNormDynamicQuant, MultiAddRmsNormDynamicQuantTilingData)

BEGIN_TILING_DATA_DEF(MultiAddRmsNormDynamicQuantRegbaseTilingData)
TILING_DATA_FIELD_DEF(uint64_t, numM);
TILING_DATA_FIELD_DEF(uint64_t, numN);
TILING_DATA_FIELD_DEF(uint64_t, baseM);
TILING_DATA_FIELD_DEF(uint64_t, baseN);
TILING_DATA_FIELD_DEF(uint64_t, baseNDtypeAlign);
TILING_DATA_FIELD_DEF(uint64_t, baseNReduceAlign);
TILING_DATA_FIELD_DEF(uint64_t, powerSplit);
TILING_DATA_FIELD_DEF(uint64_t, powerLoop);
TILING_DATA_FIELD_DEF(uint64_t, mPerCore);
TILING_DATA_FIELD_DEF(uint64_t, mLastCore);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
TILING_DATA_FIELD_DEF(uint32_t, hasSmoothScale1);
TILING_DATA_FIELD_DEF(uint32_t, hasSmoothScale2);
TILING_DATA_FIELD_DEF(uint32_t, hasBeta);
TILING_DATA_FIELD_DEF(uint32_t, x1Num);
// N 维跨核切分(issue#43): host 计算分区后由 tiling 字段直传(单一事实源, 与 mPerCore 等
// 既有字段同模式), kernel 照单消费、不再自行推导。nSplitNum==1 表示关闭(走原单核路径)。
TILING_DATA_FIELD_DEF(uint64_t, nSplitNum);
TILING_DATA_FIELD_DEF(uint64_t, nSegLen);
TILING_DATA_FIELD_DEF(uint64_t, nSegTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MultiAddRmsNormDynamicQuant_100, MultiAddRmsNormDynamicQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(MultiAddRmsNormDynamicQuant_102, MultiAddRmsNormDynamicQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(MultiAddRmsNormDynamicQuant_103, MultiAddRmsNormDynamicQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(MultiAddRmsNormDynamicQuant_104, MultiAddRmsNormDynamicQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(MultiAddRmsNormDynamicQuant_199, MultiAddRmsNormDynamicQuantRegbaseTilingData)

constexpr uint32_t TILING_TYPE_PERF = 0;
// 1 号位(NORMAL/key 101)已下线: 不可达死档, 见 tiling_arch35.cpp SetTilingParams 注释。编号保留不复用。
constexpr uint32_t TILING_TYPE_SINGLE_ROW = 2;
constexpr uint32_t TILING_TYPE_SPILT = 3;
// N-split(issue#43): split 内核的按 N 分段多核形态。独立 key 而非在 103 内分支, 因为
// 任务类型必须**编译期静态可判**(预编译按 TILING_KEY_IS(<数字>) 提取 __enable_feature_*,
// 不接受运行时数据条件) —— 104 静态绑定 MIX_AIV_1_0 以启用 SyncAll 跨核同步资源
// (intercoreSync), 103 保持 AIV_ONLY 不动, M>1 形态零波及。
constexpr uint32_t TILING_TYPE_NSPLIT = 4;
constexpr uint32_t TILING_OFFSET_REGBASE = 100;
constexpr uint64_t TILING_KEY_UNRUN = 199;

struct MultiAddRmsNormDynamicQuantCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};

enum class UB_TILING_POLICY : uint8_t { NORMAL, SINGLE_ROW, SLICE_D };

class MultiAddRmsNormDynamicQuantTilingHelper {
public:
    explicit MultiAddRmsNormDynamicQuantTilingHelper(gert::TilingContext* context) : context_(context) {}

    ~MultiAddRmsNormDynamicQuantTilingHelper() = default;
    bool DoTiling();
    void SetTilingDataAndTilingKeyAndWorkSpace(MultiAddRmsNormDynamicQuantTilingData* tiling);

private:
    bool GetBaseInfo();
    bool GetShapeInfo();
    bool DoBlockTiling();
    bool DoUbTiling();
    bool CheckInputOutputShape();
    bool CheckInputOutputDType();

    bool CheckUbNormalTiling();
    bool CheckUbSingleRowTiling();
    bool CheckUbSliceDTiling();

    gert::TilingContext* context_;

    ge::DataType xDtype_{ge::DataType::DT_FLOAT16};
    uint64_t x1Num_{1};
    uint64_t dtSize_{2};
    uint64_t socCoreNums_{1};
    uint64_t ubSize_{1};
    uint64_t sysWorkspaceSize_{1};

    uint64_t useCore_{1};
    uint64_t numFirstDim_{1};
    uint64_t numLastDim_{1};
    uint64_t numLastDimAligned_{1};
    uint64_t firstDimPerCore_{1};
    uint64_t firstDimPerCoreTail_{1};
    uint64_t firstDimPerLoop_{1};
    uint64_t lastDimSliceLen_{1};
    uint64_t lastDimLoopNum_{1};
    uint64_t lastDimSliceLenTail_{1};
    float eps_{1e-6};
    float avgFactor_{0.0};
    uint32_t smoothNum_{0};
    UB_TILING_POLICY ubTilingPolicy_{UB_TILING_POLICY::SINGLE_ROW};
};

struct MultiAddRmsNormDynamicQuantRegbaseTilingParams {
    // Platform
    uint64_t maxUbSize{0};
    uint64_t totalCoreNum{0};
    uint64_t vecLength{0};
    // Input Info
    uint64_t numM{0};
    uint64_t numN{0};
    uint64_t xDtypeSize{0};
    uint64_t xDtypeAlignNum{0};
    uint64_t xReduceAlignNum{0};
    // Cal params
    uint64_t baseM{0};
    uint64_t baseN{0};
    uint64_t baseNB8Align{0};
    uint64_t baseNDtypeAlign{0};
    uint64_t baseNReduceAlign{0};
    uint64_t reduceBufLenAlign{0};
    uint64_t powerSplit{0};
    uint64_t powerLoop{0};
    uint64_t mPerCore{0};
    uint64_t mLastCore{0};
    uint64_t usedCoreNum{0};
    // N 维跨核切分
    uint64_t nSplitNum{1};
    uint64_t nSegLen{0};
    uint64_t nSegTail{0};
    // Workspace
    uint64_t workspaceSize{0};
    // Tiling key parmas
    uint64_t tilingType{0};

    float epsilon{0};
    float avgFactor{0};
    uint32_t quantBufCnt{0};
    uint32_t x1Num{1}; // multi-add TensorList 长度(1~5)
    bool hasSmoothScale1{false};
    bool hasSmoothScale2{false};
    bool hasBeta{false};
    bool hasY2Scale2{false};
    bool needGetCompileInfo{false};
    bool needRun{true};
};

class MultiAddRmsNormDynamicQuantRegbaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit MultiAddRmsNormDynamicQuantRegbaseTiling(gert::TilingContext* tilingContext)
        : Ops::NN::Optiling::TilingBaseClass(tilingContext)
    {}
    ~MultiAddRmsNormDynamicQuantRegbaseTiling() override {}

    const string nodeName = "MultiAddRmsNormDynamicQuantRegbase";
    MultiAddRmsNormDynamicQuantRegbaseTilingData tilingData;
    MultiAddRmsNormDynamicQuantRegbaseTilingParams tilingParams;

    ge::graphStatus CheckDtypeVaild(ge::DataType& srcDtype, std::vector<ge::DataType>& supportDtypeList,
                                    string srcName);
    bool CheckShapeNull();
    bool CheckOptionalInput();
    bool CheckInputShapeDim();
    bool CheckInputShapeValue();
    bool CheckInputDtype();
    bool CheckOutputDtype();
    ge::graphStatus SetInputParams();
    uint64_t CalUBTotalSize(uint64_t baseN, const uint32_t tilingType);
    int64_t CalFullLoadBaseM(uint64_t baseN, int64_t& tmpPower);
    uint64_t CalUsedSize(uint64_t baseM, uint64_t baseNB8Align, uint64_t baseNB32Align, uint64_t baseNDtypeAlign,
                         int64_t firstVcaddLength);
    ge::graphStatus SetTilingParams();
    void SetTilingData();
    void PrintTilingData();

private:
    bool TryPerfTiling();
    bool TrySingleRowTiling();
    bool TrySplitTiling();
    void TryNSplitTiling();

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

} // namespace optiling

#endif // OPS_NORM_MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_OP_HOST_H_
