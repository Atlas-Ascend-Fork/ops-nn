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
 * \file batch_norm_ext2_tiling_rar_block_split_r_arch35.cpp
 * \brief
 */
#include "batch_norm_ext2_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_RAR_BLOCK_SPLIT_R = 250000;

constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;

constexpr int64_t NUM_TWO = 2;
constexpr int64_t BLOCK_SPLIT_R0_INDEX = 2;
constexpr int64_t BLOCK_SPLIT_R1_INDEX = 0;
constexpr int64_t UP_SPLIT_R0_INDEX = 2;
constexpr int64_t WEIGHT_INPUT_IDX = 1;

constexpr int64_t DOUBLE_BUFFER_NUM = 2;
// 逐通道 fp32 小缓冲个数（batchMean / batchMeanLow / batchVar / batchVarLow 四个队列各一份）
constexpr int64_t MEAN_VAR_BUFFER_NUM = 4;
// 双肢（y 合成 + rstd）额外需要的逐通道 fp32 缓冲：
// 全局均值低位肢 / 有偏方差低位肢 / 无偏方差(out2) / rstd 高位肢 / rstd 低位肢
constexpr int64_t DD_FP32_BUFFER_NUM = 5;
// ubFactor 每元素需要的 fp32 逐 lane 缓冲个数：mean / M2 / mean 的 TwoSum 补偿残差 / count。
// mean 的残差缓冲（kernel 侧 meanLowTbuf）是缺陷 2 的 mean 精度修复引入的，2→3 同步让出
// 4 B/元素的 UB，kernel 与 tiling 的 ubFactor 口径必须一致，否则会 UB 越界。
constexpr int64_t TMP_MEAN_VAR_BUFFER_NUM = 3;
constexpr int64_t COUNT_BUFFER_NUM = 1;
constexpr int64_t RUNNING_MEAN_VAR_BUFFER_DEFAULT_NUM = 2;
constexpr int64_t BETA_GAMMA_BUFFER_NUM = 2;
// UB 预算的安全余量：下面的 ubSizeCanUse 按 buffer 的「名义字节数」求和，而 InitBuffer 每块
// 缓冲还会按 32B 对齐向上取整，AscendC 自身也可能预留少量 UBUF。本模板的逐通道缓冲已增至
// 4 个队列 + 9 个 TBuf（见 saveMeanVarSize），名义余量只剩百字节量级，不足以覆盖这些对齐
// 开销——按名义值把 UB 占满会让内核落到「算得下但可能越界写」的风险区。这里显式扣除余量，
// 与 block_split_r 模板的 UB_RESERVE_SIZE 同口径，保证 ubFactor 计算前仍留有对齐空间。
constexpr int64_t UB_RESERVE_SIZE = 32L * 1024L;

constexpr int64_t BINARY_ADD_COEF = 2;
constexpr int64_t BINARY_ADD_COEF_FOUR = 4;
constexpr int64_t WSP_RESERVED_SIZE = 16L * 1024L * 1024L;
// 每个核在 workspace 里占的 patternAAlign 行数：mean / var / mean-low / var-low
constexpr int64_t MEAN_AND_VAR_NODE_NUM = 4;
constexpr int64_t FIRST_VCADD_RESULT_MAX_NUM = 128;
} // namespace

static int64_t FindBinaryQuotient(int64_t len)
{
    int64_t binaryQuotient = 1;
    while (binaryQuotient <= len) {
        binaryQuotient *= BINARY_ADD_COEF;
    }
    binaryQuotient /= BINARY_ADD_COEF;
    return binaryQuotient;
}

namespace optiling {
class BatchNormExt2RARBlockSplitRTiling : public BatchNormExt2RegbaseTilingBase {
public:
    explicit BatchNormExt2RARBlockSplitRTiling(gert::TilingContext* context) : BatchNormExt2RegbaseTilingBase(context)
    {}
    ~BatchNormExt2RARBlockSplitRTiling() override = default;

    void Reset(gert::TilingContext* context) override
    {
        BatchNormExt2RegbaseTilingBase::Reset(context);
        usedCoreNums_ = 0;
    }

protected:
    bool IsCapable() override
    {
        OP_LOGI(context_->GetNodeName(),
                "BatchNormExt2RARBlockSplitRTiling IsCapable: enter IsCapable function to judge.");
        // 不按 format 过滤：基类已把五种 format 统一折算成 (r1_, a_, r0_)，本模板往下只用这三个值，
        // 同 dims 的 ND 与 NCHW 在这里不可区分。
        if (a_ * NUM_TWO >= static_cast<int64_t>(aicoreParams_.blockDim)) {
            return false;
        }
        OP_LOGI(context_->GetNodeName(),
                "BatchNormExt2RARBlockSplitRTiling IsCapable true! ready to use BatchNormExt2RARBlockSplitRTiling "
                "(tilingkey: "
                "%lu) !",
                TILINGKEY_RAR_BLOCK_SPLIT_R);
        return true;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void SetInputInfo();
    bool BinaryAddTiling(const int64_t binaryAddNum, int64_t& binaryAddK, int64_t& binaryAddLast);
    int64_t UbSplit(int64_t r1BlockInner, int64_t r0BlockInner, int64_t ubFactor, int64_t& ubSplitAxis,
                    int64_t& ubOuter, int64_t& ubInner, int64_t eleNumPerBlock);

private:
    BatchNormExt2RARBlockSplitRTilingData batchNormTilingData;
};

void BatchNormExt2RARBlockSplitRTiling::SetInputInfo()
{
    // dim
    batchNormTilingData.set_patternR1(r1_);
    batchNormTilingData.set_patternA(a_);
    batchNormTilingData.set_patternR0(r0_);

    // attr
    batchNormTilingData.set_epsilon(epsilon_);
    batchNormTilingData.set_momentum(exponentialAvgFactor_);
    batchNormTilingData.set_momentumReverse(1 - exponentialAvgFactor_);
}

bool BatchNormExt2RARBlockSplitRTiling::BinaryAddTiling(const int64_t binaryAddNum, int64_t& binaryAddK,
                                                        int64_t& binaryAddLast)
{
    binaryAddK = 0;
    int64_t curBinaryAddNum = 1;
    while (curBinaryAddNum < binaryAddNum) {
        binaryAddK++;
        curBinaryAddNum *= BINARY_ADD_COEF_FOUR;
    }
    if (curBinaryAddNum == binaryAddNum) {
        binaryAddLast = 0;
    } else if (curBinaryAddNum == binaryAddNum * BINARY_ADD_COEF) {
        binaryAddK = binaryAddK - 1;
        binaryAddLast = 1;
    } else {
        OP_LOGI(context_->GetNodeName(), "BinaryAddTiling binaryAddNum %ld case not supported", binaryAddNum);
        return false;
    }
    return true;
}

int64_t BatchNormExt2RARBlockSplitRTiling::UbSplit(int64_t r1BlockInner, int64_t r0BlockInner, int64_t ubFactor,
                                                   int64_t& ubSplitAxis, int64_t& ubOuter, int64_t& ubInner,
                                                   int64_t eleNumPerBlock)
{
    int64_t r0BlockInnerAlign = Ops::Base::CeilAlign(r0BlockInner, eleNumPerBlock);

    if (ubFactor <= r0BlockInnerAlign) {
        // 1. ub只切分到R0轴

        ubSplitAxis = UP_SPLIT_R0_INDEX;
        ubInner = Ops::Base::FloorAlign(ubFactor, eleNumPerBlock); // 考虑r0需要对齐，避免ub用超
        ubOuter = Ops::Base::CeilDiv(r0BlockInner, ubInner);
        ubInner = Ops::Base::CeilDiv(r0BlockInner, ubOuter);
        ubInner = Ops::Base::CeilAlign(ubInner, eleNumPerBlock);

        return ubInner;
    } else if (ubFactor <= r0BlockInnerAlign * a_ || r1BlockInner == 1) {
        // 2. A轴参与切分
        // 考虑r0需要对齐，且ubFactor大小的限制（ubFactor如果对于r0BlockInner对齐后的空间不够的话，ub切分轴需要在a轴）

        ubSplitAxis = 1;
        ubInner = std::min(Ops::Base::FloorDiv(ubFactor, r0BlockInnerAlign), a_); // 考虑r0需要对齐，避免ub用超
        ubOuter = Ops::Base::CeilDiv(a_, ubInner);

        return r0BlockInner;
    } else {
        // 3. ub切分到R1轴

        ubSplitAxis = 0;
        // 计算在ubFactor限制下，最大的对齐R空间，并对齐到32B
        int64_t maxAlignedRNum = Ops::Base::FloorDiv(ubFactor, a_);
        maxAlignedRNum = Ops::Base::FloorAlign(maxAlignedRNum, eleNumPerBlock);

        // 继续计算上述32B对齐的最大R可用空间对应的ubInner
        int64_t maxUbInnerFromAligned = Ops::Base::FloorDiv(maxAlignedRNum, r0BlockInner);

        ubInner = std::min(maxUbInnerFromAligned, r1BlockInner);

        // 确保至少要处理1个元素
        ubInner = ubInner > 1 ? ubInner : 1;

        ubOuter = Ops::Base::CeilDiv(r1BlockInner, ubInner);
        ubInner = Ops::Base::CeilDiv(r1BlockInner, ubOuter);

        return ubInner * r0BlockInner;
    }
}

ge::graphStatus BatchNormExt2RARBlockSplitRTiling::DoOpTiling()
{
    SetInputInfo();

    int64_t blockSplitAxis = 0;
    int64_t formerBlockOuter = 0;
    int64_t tailBlockOuter = 0;
    int64_t blockInner = 0;

    // 多核切分R轴：采用均分方式，划分formerCore和tailCore
    if (r1_ < static_cast<int64_t>(aicoreParams_.blockDim)) {
        // R0参与多核切分
        blockSplitAxis = BLOCK_SPLIT_R0_INDEX;
        int64_t blockOuter = std::min(Ops::Base::FloorDiv(static_cast<int64_t>(aicoreParams_.blockDim), r1_), r0_);
        blockInner = Ops::Base::CeilDiv(r0_, blockOuter);
        tailBlockOuter = blockInner * blockOuter - r0_;
        formerBlockOuter = blockOuter - tailBlockOuter;
        usedCoreNums_ = r1_ * blockOuter;
    } else {
        // R1即满足多核
        blockSplitAxis = BLOCK_SPLIT_R1_INDEX;
        blockInner = Ops::Base::CeilDiv(r1_, static_cast<int64_t>(aicoreParams_.blockDim));
        int64_t blockOuter = Ops::Base::CeilDiv(r1_, blockInner);
        tailBlockOuter = blockInner * blockOuter - r1_;
        formerBlockOuter = blockOuter - tailBlockOuter;
        usedCoreNums_ = blockOuter;
    }

    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: blockSplitAxis is %lu !",
            blockSplitAxis);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: formerBlockOuter is %lu !",
            formerBlockOuter);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: tailBlockOuter is %lu !",
            tailBlockOuter);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: blockInner is %lu !", blockInner);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: usedCoreNums_ is %lu !",
            usedCoreNums_);

    // 计算ubfactor
    // Step1: ubsize减去固定ubffer大小部分，得到ubSizeCanUse
    int64_t elemSize = FP32_BYTE;
    int64_t T2ElemSize = FP32_BYTE;
    if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
        elemSize = FP16_BYTE;
    }
    auto weightDesc = context_->GetInputDesc(WEIGHT_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    auto weightDataType = weightDesc->GetDataType();
    if (weightDataType == ge::DT_FLOAT16 || weightDataType == ge::DT_BF16) {
        T2ElemSize = FP16_BYTE;
    }

    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: class member blockSize_ is %lu !",
            blockSize_);
    int64_t fp32EleNumPerBlock = blockSize_ / FP32_BYTE;
    int64_t patternAAlign = Ops::Base::CeilAlign(a_, fp32EleNumPerBlock);

    int64_t runningMeanVarSize = Ops::Base::CeilAlign(a_, blockSize_ / T2ElemSize) * T2ElemSize *
                                 RUNNING_MEAN_VAR_BUFFER_DEFAULT_NUM;
    if (useRunningMeanVar_) {
        runningMeanVarSize *= NUM_TWO;
    }
    int64_t saveMeanVarSize = patternAAlign * static_cast<int64_t>(sizeof(float)) *
                              (MEAN_VAR_BUFFER_NUM + DD_FP32_BUFFER_NUM);
    int64_t betaGammaSize = Ops::Base::CeilAlign(a_, blockSize_ / T2ElemSize) * T2ElemSize * BETA_GAMMA_BUFFER_NUM;

    int64_t blockDimAlignSize = Ops::Base::CeilAlign(static_cast<int64_t>(aicoreParams_.blockDim), fp32EleNumPerBlock) *
                                FP32_BYTE;
    int64_t tmpbuffer3AlignSize = std::max(static_cast<int64_t>(aicoreParams_.blockDim) * patternAAlign,
                                           static_cast<int64_t>(
                                               Ops::Base::CeilAlign(FIRST_VCADD_RESULT_MAX_NUM, fp32EleNumPerBlock))) *
                                  FP32_BYTE;

    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: runningMeanVarSize is %lu !",
            runningMeanVarSize);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: saveMeanVarSize is %lu !",
            saveMeanVarSize);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: betaGammaSize is %lu !",
            betaGammaSize);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: blockDimAlignSize is %lu !",
            blockDimAlignSize);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: tmpbuffer3AlignSize is %lu !",
            tmpbuffer3AlignSize);

    int64_t xSizePerUbFactor = elemSize * DOUBLE_BUFFER_NUM;
    int64_t ySizePerUbFactor = elemSize * DOUBLE_BUFFER_NUM;
    int64_t tmpMeanVarPerUbFactor = static_cast<int64_t>(sizeof(float)) * TMP_MEAN_VAR_BUFFER_NUM;
    int64_t countBuffer1PerUbFactor = static_cast<int64_t>(sizeof(float)) * COUNT_BUFFER_NUM;

    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: xSizePerUbFactor is %lu !",
            xSizePerUbFactor);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: ySizePerUbFactor is %lu !",
            ySizePerUbFactor);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: tmpMeanVarPerUbFactor is %lu !",
            tmpMeanVarPerUbFactor);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: countBuffer1PerUbFactor is %lu !",
            countBuffer1PerUbFactor);

    int64_t ubSizeCanUse = static_cast<int64_t>(aicoreParams_.ubSize) - runningMeanVarSize - saveMeanVarSize -
                           betaGammaSize - blockDimAlignSize - tmpbuffer3AlignSize - UB_RESERVE_SIZE;
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: ubSizeCanUse is %lu !",
            ubSizeCanUse);
    OP_CHECK_IF(ubSizeCanUse <= 0, OP_LOGI(context_->GetNodeName(), "ubSizeCanUse is not a positive number."),
                return ge::GRAPH_PARAM_INVALID);

    // Step2: 根据ubSizeCanUse，算出ubFactor，并进行对齐操作
    int64_t ubFactor = ubSizeCanUse /
                       (xSizePerUbFactor + ySizePerUbFactor + tmpMeanVarPerUbFactor + countBuffer1PerUbFactor);

    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: direct obtained ubFactor is %lu !",
            ubFactor);

    ubFactor = Ops::Base::FloorAlign(ubFactor, blockSize_ / elemSize);

    OP_CHECK_IF(ubFactor == 0, OP_LOGI(context_->GetNodeName(), "ubFactor is 0."), return ge::GRAPH_PARAM_INVALID);
    batchNormTilingData.set_ubFactor(ubFactor);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: ubFactor is %lu !\n", ubFactor);

    // 根据ubFactor计算ub切分参数
    int64_t formerCoreR1BlockInner = 0;
    int64_t formerCoreR0BlockInner = 0;
    int64_t tailCoreR1BlockInner = 0;
    int64_t tailCoreR0BlockInner = 0;

    int64_t formerCoreUbSplitAxis = 0;
    int64_t formerCoreUbOuter = 0;
    int64_t formerCoreUbInner = 0;
    int64_t tailCoreUbSplitAxis = 0;
    int64_t tailCoreUbOuter = 0;
    int64_t tailCoreUbInner = 0;

    if (blockSplitAxis == BLOCK_SPLIT_R1_INDEX) {
        formerCoreR1BlockInner = blockInner;
        tailCoreR1BlockInner = blockInner - 1;
        formerCoreR0BlockInner = r0_;
        tailCoreR0BlockInner = r0_;
    } else {
        formerCoreR1BlockInner = 1;
        tailCoreR1BlockInner = 1;
        formerCoreR0BlockInner = blockInner;
        tailCoreR0BlockInner = blockInner - 1;
    }
    int64_t formerCoreProcessRNum = 0;
    int64_t tailCoreProcessRNum = 0;
    formerCoreProcessRNum = UbSplit(formerCoreR1BlockInner, formerCoreR0BlockInner, ubFactor, formerCoreUbSplitAxis,
                                    formerCoreUbOuter, formerCoreUbInner, blockSize_ / elemSize);
    tailCoreProcessRNum = UbSplit(tailCoreR1BlockInner, tailCoreR0BlockInner, ubFactor, tailCoreUbSplitAxis,
                                  tailCoreUbOuter, tailCoreUbInner, blockSize_ / elemSize);

    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: formerCoreUbSplitAxis is %lu !",
            formerCoreUbSplitAxis);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: formerCoreUbOuter is %lu !",
            formerCoreUbOuter);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: formerCoreUbInner is %lu !\n",
            formerCoreUbInner);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: tailCoreUbSplitAxis is %lu !",
            tailCoreUbSplitAxis);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: tailCoreUbOuter is %lu !",
            tailCoreUbOuter);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: tailCoreUbInner is %lu !\n\n",
            tailCoreUbInner);

    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: formerCoreProcessRNum is %lu !",
            formerCoreProcessRNum);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: tailCoreProcessRNum is %lu !\n",
            tailCoreProcessRNum);

    int64_t formerCoreBinaryAddQuotient = FindBinaryQuotient(formerCoreProcessRNum);
    int64_t tailCoreBinaryAddQuotient = FindBinaryQuotient(tailCoreProcessRNum);

    OP_LOGI(context_->GetNodeName(),
            "BatchNormExt2RARBlockSplitRTiling DoOpTiling: formerCoreBinaryAddQuotient is %lu !",
            formerCoreBinaryAddQuotient);
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling: tailCoreBinaryAddQuotient is %lu !",
            tailCoreBinaryAddQuotient);

    int64_t lastBinaryAddQuotient = FindBinaryQuotient(usedCoreNums_);
    int64_t lastBinaryAddK = 0;
    int64_t lastBinaryAddLast = 0;
    auto res = BinaryAddTiling(lastBinaryAddQuotient, lastBinaryAddK, lastBinaryAddLast);
    OP_CHECK_IF(res == false,
                OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling BinaryAddTiling param invalid"),
                return ge::GRAPH_PARAM_INVALID);

    // 全局除数 R（及 R-1）的双肢：host 侧用 double 算 R，再取 fl32(R) 与精确余量。
    // |R - fl32(R)| ≤ 1（R < 2^25 时），落 fp32 精确。
    double rTotalD = static_cast<double>(r1_) * static_cast<double>(r0_);
    float nHi = static_cast<float>(rTotalD);
    float nLo = static_cast<float>(rTotalD - static_cast<double>(nHi));
    double rMinusOneD = rTotalD - 1.0;
    float nMinusOneHi = static_cast<float>(rMinusOneD);
    float nMinusOneLo = static_cast<float>(rMinusOneD - static_cast<double>(nMinusOneHi));
    batchNormTilingData.set_nHi(nHi);
    batchNormTilingData.set_nLo(nLo);
    batchNormTilingData.set_nMinusOneHi(nMinusOneHi);
    batchNormTilingData.set_nMinusOneLo(nMinusOneLo);
    OP_LOGI(context_->GetNodeName(), "RAR tiling divisor R=%.0f -> (%.9g, %.9g) ; R-1 -> (%.9g, %.9g)", rTotalD, nHi,
            nLo, nMinusOneHi, nMinusOneLo);

    batchNormTilingData.set_patternAAlign(patternAAlign);
    batchNormTilingData.set_blockSplitAxis(blockSplitAxis);
    batchNormTilingData.set_formerBlockOuter(formerBlockOuter);
    batchNormTilingData.set_tailBlockOuter(tailBlockOuter);
    batchNormTilingData.set_blockInner(blockInner);
    batchNormTilingData.set_formerCoreUbSplitAxis(formerCoreUbSplitAxis);
    batchNormTilingData.set_formerCoreUbOuter(formerCoreUbOuter);
    batchNormTilingData.set_formerCoreUbInner(formerCoreUbInner);
    batchNormTilingData.set_tailCoreUbSplitAxis(tailCoreUbSplitAxis);
    batchNormTilingData.set_tailCoreUbOuter(tailCoreUbOuter);
    batchNormTilingData.set_tailCoreUbInner(tailCoreUbInner);
    batchNormTilingData.set_formerCoreBinaryAddQuotient(formerCoreBinaryAddQuotient);
    batchNormTilingData.set_tailCoreBinaryAddQuotient(tailCoreBinaryAddQuotient);
    batchNormTilingData.set_lastBinaryAddQuotient(lastBinaryAddQuotient);
    batchNormTilingData.set_lastBinaryAddK(lastBinaryAddK);
    batchNormTilingData.set_lastBinaryAddLast(lastBinaryAddLast);
    batchNormTilingData.set_useRunningMeanVar(useRunningMeanVar_);

    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling DoOpTiling SUCCESS!");
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormExt2RARBlockSplitRTiling::GetTilingKey() const
{
    OP_LOGI(context_->GetNodeName(), "BatchNormExt2RARBlockSplitRTiling GetTilingKey tilingkey is %lu !",
            TILINGKEY_RAR_BLOCK_SPLIT_R);
    return TILINGKEY_RAR_BLOCK_SPLIT_R;
}

ge::graphStatus BatchNormExt2RARBlockSplitRTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = WSP_RESERVED_SIZE +
                          usedCoreNums_ * MEAN_AND_VAR_NODE_NUM * batchNormTilingData.get_patternAAlign() * FP32_BYTE;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(batchNormTilingData.GetDataSize() > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        batchNormTilingData.GetDataSize(), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    batchNormTilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(batchNormTilingData.GetDataSize());
    uint32_t batch_mode = 1U;
    auto ret = context_->SetScheduleMode(batch_mode);

    return ret;
}

REGISTER_TILING_TEMPLATE("BatchNormExt2", BatchNormExt2RARBlockSplitRTiling, 25000);
} // namespace optiling
