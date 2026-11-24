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
 * \file multi_add_rms_norm_dynamic_quant_regbase_split_reduce.h
 * \brief
 */
#ifndef MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_SPILT_REDUCE_H_
#define MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_SPILT_REDUCE_H_

#include "multi_add_rms_norm_dynamic_quant_regbase_common.h"

namespace MultiAddRmsNormDynamicQuant {

// 跨核合并槽: 每个值 1 个 16 fp32(64B, 恰一个 cacheline)的广播槽, 每核 4 个值
// (Σx²hi/Σx²lo/max1/max2)。写入全走 Duplicate(标量操作数, 向量指令) —— 不做标量直写 UB,
// 标量↔向量序由 V_S/S_V 事件对表达(renorm_global.h 同款), 不使用 PIPE_ALL。
constexpr uint64_t PART_SLOT_ELEMS = 16;
constexpr uint64_t PART_SLOTS_PER_CORE = 4;
// 暂存面积数(y1Tmp / xOutTmp / y2Tmp), 与 host GetWorkspaceSize 的 WORKSPACE_COUNT 同源。
constexpr uint64_t NSPLIT_AREA_CNT = 3;
// 合并槽区基址对齐粒度(64B = 16 个 float), 与 host 的对齐表达式同源。
constexpr uint64_t PART_BASE_ALIGN_ELEMS = 16;

template <typename T_X, typename T_Y>
class KernelMultiAddRmsNormDynamicQuantRegbaseSpiltReduce {
    using T_SMOOTH_SCALE = T_X;

public:
    __aicore__ inline KernelMultiAddRmsNormDynamicQuantRegbaseSpiltReduce(TPipe* pipe) { pipe_ = pipe; }

    // 相对参照:去 beta 输入(multi_add 无 beta,hasBeta_ 恒 false);增 y 输出(RmsNorm 结果);x1 为 TensorList;
    // 末尾保留 workspace 参(两阶段 split reduce 用)。
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooathScale1, GM_ADDR smooathScale2,
                                GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2,
                                GM_ADDR workspace, const MultiAddRmsNormDynamicQuantRegbaseTilingData* tilingData)
    {
        numM_ = tilingData->numM;
        numN_ = tilingData->numN;
        baseM_ = tilingData->baseM;
        baseN_ = tilingData->baseN;
        baseNDtypeAlign_ = tilingData->baseNDtypeAlign;
        baseNReduceAlign_ = tilingData->baseNReduceAlign;
        powerSplit_ = tilingData->powerSplit;
        powerLoop_ = tilingData->powerLoop;
        mPerCore_ = tilingData->mPerCore;
        mLastCore_ = tilingData->mLastCore;
        epsilon_ = tilingData->epsilon;
        avgFactor_ = tilingData->avgFactor;

        hasSmoothScale1_ = tilingData->hasSmoothScale1;
        hasSmoothScale2_ = tilingData->hasSmoothScale2;
        hasBeta_ = false; // multi_add 无 beta
        hasY2Scale2_ = hasSmoothScale2_;
        x1Num_ = tilingData->x1Num; // multi-add TensorList 长度(1~5)

        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();
        oriOverflowMode_ = GetOverflowMode<T_Y>();

        // 行×列二维划分(issue#43 全量修复): host 下发 nSplitNum = M*K2(总块数), kernel 同源推导
        //   K2 = nSplitNum/numM(整除), 本核负责 (rowIdx_, segIdx_) = (blockIdx/K2, blockIdx%K2),
        //   只处理该行 [segStart_, segStart_+numN_) 一段。numM==1 时 K2=nSplitNum, 与 v15 完全一致。
        // numN_ 覆写为段长; powerLoop_ 按段长本地重算。oriNumN_ 保留整行长度, 供 workspace
        // 三面积基址(每面积 M 行)与合并槽寻址。不变量同 v15(段长>=ps; 决策全局一致防 SyncAll 死锁)。
        nSplitNum_ = tilingData->nSplitNum;
        nSegLen_ = tilingData->nSegLen;
        nSegTail_ = tilingData->nSegTail;
        oriNumN_ = numN_;
        if (nSplitNum_ > 1) {
            uint64_t rows = numM_;
            k2_ = nSplitNum_ / rows;
            // 一致性防御(各核同 tiling → 同决策, 不会分叉死锁): 块数失配/不整除/段长越域
            // (历史缺陷: 尾段无符号回绕成巨值)→ 整体回落单核。
            if (k2_ < 2 || nSplitNum_ != rows * k2_ || nSplitNum_ != static_cast<uint64_t>(blockNum_) ||
                nSegLen_ < powerSplit_ || nSegLen_ > oriNumN_ || nSegTail_ < powerSplit_ || nSegTail_ > oriNumN_) {
                // 不能回落单核 legacy: 此时实际起核数是 numM*K2, 而 legacy 按
                // blockIdx_*mPerCore_*numN_ 定位, blockIdx_ >= numM 的核会算出张量界外偏移,
                // 把"结果不可信"升级成 GM 越界写。标记非法, Process 直接空转返回,
                // 由 host tiling 的校验保证不下发非法 tiling。
                invalidTiling_ = true;
            }
        }
        if (nSplitNum_ > 1) {
            uint64_t myIdx = static_cast<uint64_t>(blockIdx_);
            rowIdx_ = myIdx / k2_;
            uint64_t segIdx = myIdx % k2_;
            rowBase_ = rowIdx_ * oriNumN_;
            segStart_ = segIdx * nSegLen_;
            numN_ = (segIdx == k2_ - 1) ? nSegTail_ : nSegLen_;
            powerLoop_ = 1;
            while (powerLoop_ * CONST_FACTOR_2 * powerSplit_ <= numN_) {
                powerLoop_ *= CONST_FACTOR_2;
            }
        }

        CalBlockTail();
        InitBuffer(x1, x2, gamma, smooathScale1, smooathScale2, y1, y2, x, y, scale1, scale2, workspace);
    }

    __aicore__ inline void CalBlockTail()
    {
        mCore_ = blockIdx_ == (blockNum_ - 1) ? mLastCore_ : mPerCore_;
        nCnt_ = CeilDiv(numN_, baseN_);
        tailN_ = numN_ - (nCnt_ - 1) * baseN_;

        tailNDtypeAlign_ = CeilAlign(tailN_, BLOCK_SIZE / sizeof(T_X));
        baseNB8Align_ = CeilAlign(baseN_, B8_BLOCK_NUM);
        baseNB32Align_ = CeilAlign(baseN_, B32_BLOCK_NUM);
        uint64_t reduceSumBufLen = baseNReduceAlign_ / (2 * V_LENGTH);
        reduceSumBufAlign_ = CeilAlign(reduceSumBufLen, B32_BLOCK_NUM);
    }

    __aicore__ inline void InitBuffer(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooathScale1,
                                      GM_ADDR smooathScale2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR y,
                                      GM_ADDR scale1, GM_ADDR scale2, GM_ADDR workspace)
    {
        // nSplit: 输入/输出按段重绑(gmOffset=segStart_, gmLen=段长); 暂存区三面积基址改为
        // i*oriNumN_(原为 i*blockNum_*numN_, 两语义在 M 分组下同值), 各核只写自己段的槽。
        bool nSplit = nSplitNum_ > 1;
        // nSplit 2D: 行/列输入输出按 (rowBase_+segStart_, 段长) 重绑; gamma/smooth 按列段共享;
        // scale 按行(每行 group-0 核独写); 暂存面积基址 = i*M*oriNumN_(每面积 M 整行)。
        uint64_t gmOffset = nSplit ? (rowBase_ + segStart_) : static_cast<uint64_t>(blockIdx_) * mPerCore_ * numN_;
        uint64_t gmLen = nSplit ? numN_ : mCore_ * numN_;
        uint64_t gammaOffset = nSplit ? segStart_ : 0;
        uint64_t scalesGmOffset = nSplit ? rowIdx_ : static_cast<uint64_t>(blockIdx_) * mPerCore_;
        uint64_t workSpaceOffset = nSplit ? (rowBase_ + segStart_) : static_cast<uint64_t>(blockIdx_) * numN_;
        uint64_t workSpaceStart = nSplit ? oriNumN_ * numM_ : static_cast<uint64_t>(blockNum_) * numN_;
        // x1 为 TensorList:逐个 tensor 取地址(参照 ascend910b kernel GetTensorAddr<T>(i, x1))
        for (uint32_t i = 0; i < x1Num_; ++i) {
            x1GmList_[i].SetGlobalBuffer(GetTensorAddr<T_X>(i, x1) + gmOffset, gmLen);
        }
        x2Gm_.SetGlobalBuffer((__gm__ T_X*)x2 + gmOffset, gmLen);
        gammaGm_.SetGlobalBuffer((__gm__ T_X*)gamma + gammaOffset, numN_);
        y1Gm_.SetGlobalBuffer((__gm__ T_Y*)y1 + gmOffset, gmLen);
        xGm_.SetGlobalBuffer((__gm__ T_X*)x + gmOffset, gmLen);
        yGm_.SetGlobalBuffer((__gm__ T_X*)y + gmOffset, gmLen); // 增:y 输出(RmsNorm 结果)
        scale1Gm_.SetGlobalBuffer((__gm__ float*)scale1 + scalesGmOffset, mCore_);
        y1TmpGm_.SetGlobalBuffer((__gm__ float*)workspace + workSpaceOffset, numN_);
        xOutTmpGm_.SetGlobalBuffer((__gm__ float*)workspace + workSpaceStart + workSpaceOffset, numN_);
        if (hasSmoothScale1_) {
            smoothScale1Gm_.SetGlobalBuffer((__gm__ T_SMOOTH_SCALE*)smooathScale1 + gammaOffset, numN_);
        }
        if (hasSmoothScale2_) {
            smoothScale2Gm_.SetGlobalBuffer((__gm__ T_SMOOTH_SCALE*)smooathScale2 + gammaOffset, numN_);
        }
        if (hasY2Scale2_) {
            y2Gm_.SetGlobalBuffer((__gm__ T_Y*)y2 + gmOffset, gmLen);
            scale2Gm_.SetGlobalBuffer((__gm__ float*)scale2 + scalesGmOffset, mCore_);
            y2TmpGm_.SetGlobalBuffer((__gm__ float*)workspace + 2 * workSpaceStart + workSpaceOffset, numN_);
        } else {
            // ss2 缺省: y2/s2 退化为 1 元素占位输出(infershape 固定 6 输出)。必须绑定合法
            // 窗口 —— GlobalTensor 默认构造不初始化地址, 不绑定则 WritePlaceholderY2Scale2
            // 的 SetValue 把 0 写到残留地址(逐核不同、随代码形态漂移): _055 实证其砸穿了
            // 每核 x 段首元素低字节(62 处, golden&0xFF00)。
            y2Gm_.SetGlobalBuffer((__gm__ T_Y*)y2, 1);
            scale2Gm_.SetGlobalBuffer((__gm__ float*)scale2, 1);
        }

        uint64_t ubFactorQuant = CeilAlign(baseN_, BLOCK_SIZE / sizeof(T_SMOOTH_SCALE));
        uint64_t ubFactorRstd = B32_BLOCK_NUM;
        pipe_->InitBuffer(inQueueX1_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(inQueueX2_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueX_, 1, baseNReduceAlign_ * sizeof(T_X));
        pipe_->InitBuffer(outQueueY_, 1, baseNDtypeAlign_ * sizeof(T_X)); // 增:y(RmsNorm 结果)输出
        pipe_->InitBuffer(x1AccBuf_,
                          baseNReduceAlign_ * sizeof(float)); // 增:multi-add Σx1 fp32 累加缓冲(对齐 ascend910b 全 fp32)
        pipe_->InitBuffer(x2Fp32Buf_, baseNReduceAlign_ * sizeof(float)); // 增:x2 fp32 缓冲(与 Σx1 同以 fp32 喂 reduce)
        pipe_->InitBuffer(inQueueGamma_, 1, baseNDtypeAlign_ * sizeof(T_X));
        if (hasBeta_) {
            pipe_->InitBuffer(inQueueBeta_, 1, baseNDtypeAlign_ * sizeof(T_X));
        }
        pipe_->InitBuffer(outQueueY1_, 1, baseNB8Align_ * sizeof(T_Y));
        pipe_->InitBuffer(outQueueScale1_, 1, ubFactorRstd * sizeof(float));
        if (hasSmoothScale1_) {
            pipe_->InitBuffer(inQueueSmoothScale1_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if (hasSmoothScale2_) {
            pipe_->InitBuffer(inQueueSmoothScale2_, 1, ubFactorQuant * sizeof(T_SMOOTH_SCALE));
        }
        if (hasY2Scale2_) {
            pipe_->InitBuffer(outQueueY2_, 1, baseNB8Align_ * sizeof(T_Y));
            pipe_->InitBuffer(outQueueScale2_, 1, ubFactorRstd * sizeof(float));
        }

        pipe_->InitBuffer(xOutTmpBuf_, baseNReduceAlign_ * sizeof(float));
        pipe_->InitBuffer(y1TmpBuf_, baseNB32Align_ * sizeof(float));
        if (hasY2Scale2_) {
            pipe_->InitBuffer(y2TmpBuf_, baseNB32Align_ * sizeof(float));
        }
        pipe_->InitBuffer(rstdBuf_, ubFactorRstd * sizeof(float));
        // 横向规约用 accAddr+stride 做整 VL 载入(无掩码, 固定读 V_LENGTH 个), 故每段必须留
        // 2*V_LENGTH 的地址空间, 否则 lo 段尾部越界读 UB。
        pipe_->InitBuffer(msAccBuf_, NUM_FOUR_DF * V_LENGTH * sizeof(float)); // Σx²: hi[0,2V) lo[2V,4V)
        pipe_->InitBuffer(reduceSumBuf_, reduceSumBufAlign_ * sizeof(float));
        pipe_->InitBuffer(level1Buf_, RmsNorm::ONCE_VECTOR_SIZE * sizeof(float));
        pipe_->InitBuffer(level2Buf_, RmsNorm::ONCE_VECTOR_SIZE * sizeof(float));
        pipe_->InitBuffer(level3Buf_, RmsNorm::ONCE_VECTOR_SIZE * sizeof(float));
        pipe_->InitBuffer(tempBuf_, V_LENGTH * sizeof(float));
        if (nSplit) {
            // 跨核合并槽: 每核 4 个 64B 广播槽(写视图 partGm_ + 全量读视图 partGmAll_),
            // 布局/配方与 renorm_global.h 的 A5 跨核规约一致(host N_SPLIT_PARTIAL_CORE_BYTES 同源)。
            pipe_->InitBuffer(partOutBuf_, PART_SLOTS_PER_CORE * PART_SLOT_ELEMS * sizeof(float));
            pipe_->InitBuffer(partInBuf_, PART_SLOTS_PER_CORE * PART_SLOT_ELEMS * sizeof(float));
            // 槽区基址必须 64B(16 个 float) 对齐: 每核 4 个 64B 槽靠 DCCI 逐槽 clean/invalidate,
            // 基址落在 cacheline 中间会让相邻两核的槽共享同一 cacheline, 互相刷掉未落地的数据。
            // 暂存三面积共 3*M*N 个 float, 仅当 M*N % 16 == 0 时天然对齐(反例: M=18,N=30406)。
            // host GetWorkspaceSize 必须用同一个对齐表达式, 否则两侧基址或总量对不上。
            const uint64_t partBase = (NSPLIT_AREA_CNT * oriNumN_ * numM_ + PART_BASE_ALIGN_ELEMS - 1) /
                                      PART_BASE_ALIGN_ELEMS * PART_BASE_ALIGN_ELEMS;
            partGmAll_.SetGlobalBuffer((__gm__ float*)workspace + partBase,
                                       nSplitNum_ * PART_SLOTS_PER_CORE * PART_SLOT_ELEMS);
            partGm_.SetGlobalBuffer((__gm__ float*)workspace + partBase +
                                        static_cast<uint64_t>(blockIdx_) * PART_SLOTS_PER_CORE * PART_SLOT_ELEMS,
                                    PART_SLOTS_PER_CORE * PART_SLOT_ELEMS);
        }
    }

    __aicore__ inline void Process()
    {
        if (invalidTiling_) {
            return;
        }
        if (nSplitNum_ > 1) {
            ProcessNSplit();
            // ss2 缺省: y2/s2 占位输出写 0 定值(修复 deterministic FAIL)。
            // 必须在 ProcessNSplit 之后写: 放在入口会扰动每核首段 x 的前 32B
            // (_055 seed42 实证 62 段×32 元素被清零; v22.6 无此写全过)。
            if (!hasY2Scale2_) {
                WritePlaceholderY2Scale2(y2Gm_, scale2Gm_);
            }
            return;
        }
        uint64_t powerMain = powerSplit_ * powerLoop_;
        uint64_t powerTail = numN_ - powerMain;
        uint64_t powerTailLoop = powerTail / powerSplit_;
        uint64_t powerTailTail = powerTail % powerSplit_;
        uint64_t masterLoop = powerTailTail != 0 ? 1 : 0;
        masterLoop = powerLoop_ - powerTailLoop - masterLoop;
        LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
        LocalTensor<float> msAccLocal = msAccBuf_.Get<float>();

        for (uint64_t mIdx = 0; mIdx < mCore_; mIdx++) {
            uint64_t xGmOffset = mIdx * numN_;

            LocalTensor<float> scale1Local = outQueueScale1_.AllocTensor<float>();
            LocalTensor<float> scale2Local;
            if (hasY2Scale2_) {
                scale2Local = outQueueScale2_.AllocTensor<float>();
            }
            Duplicate(scale1Local, 0.0f, B32_BLOCK_NUM);
            if (hasY2Scale2_) {
                Duplicate(scale2Local, 0.0f, B32_BLOCK_NUM);
            }
            // Process rstd & Copy out xOut(x1+x2)
            // IS_RSTD=false: 共享规约的单段 fp32 Σx² 不再用于 rstd —— 它的舍入(~1 ULP)会经 rstd
            // 乘进每个元素, 把 scale 推过 0.5 ULP; 补偿链缺这一级则后两级的补偿也失效。
            // 改为: ComputeFormerHandle 里逐 slice 做补偿累加, 这里再出两段式 rstd。
            Duplicate(msAccLocal, 0.0f, NUM_FOUR_DF * V_LENGTH); // 每行清零(含越界余量区)
            PipeBarrier<PIPE_V>();
            SubProcessRstd<false>(rstdLocal, 0, xGmOffset, masterLoop, powerTailLoop, powerTailTail, avgFactor_,
                                  epsilon_);
            PipeBarrier<PIPE_V>();
            FinalizeRstd(rstdLocal, avgFactor_, epsilon_);
            PipeBarrier<PIPE_V>();
            // Process reducemax & Store (rmsout*smooth) to workSpace
            // row_max 走单段 fp32: max 本身不产生舍入误差, 对它做两段式补偿无收益, 而补偿式
            // r = hi - s0*qmax 在 hi 非有限时退化为 inf-inf = NaN, 使 scale 输出 NaN(公式应得 inf)。
            SubProcessReduceMax(scale1Local, scale2Local, rstdLocal, mIdx);
            // Copy out y(ComputeY 要用 row_max,故必须在 SubProcessScale 之前)
            PipeBarrier<PIPE_V>();
            SubProcessY(scale1Local, scale2Local, mIdx);
            // Process scale:row_max -> 真实 scale(GM 语义不变)
            PipeBarrier<PIPE_V>();
            SubProcessScale(scale1Local, scale2Local);
            // Copy out scale
            outQueueScale1_.EnQue<float>(scale1Local);
            if (hasY2Scale2_) {
                outQueueScale2_.EnQue<float>(scale2Local);
            }
            CopyOutScale(scale1Gm_, outQueueScale1_, mIdx, 1);
            if (hasY2Scale2_) {
                CopyOutScale(scale2Gm_, outQueueScale2_, mIdx, 1);
            }
        }
        // ss2 缺省: y2/s2 占位输出写 0 定值(与 perf/single_row/nSplit 分支一致, 修 deterministic
        // 残值)。放在主循环之后, 避免扰动每核首段 x 的前 32B(_055 实证)。
        if (!hasY2Scale2_) {
            WritePlaceholderY2Scale2(y2Gm_, scale2Gm_);
        }
    }

    // N-split 多核主流程(仅 numM==1): 每核处理一段, 两轮 SyncAll 栅障交换跨核规约量。
    // 数值语义与单核版逐段对齐: Phase A/B 只改 Σx² 的求和划分(每核段内补偿累加 + 跨核
    // 固定 k 序两段式合并), row_max 为精确 max(可结合); x 输出与 rstd 无关, 逐位不变。
    __aicore__ inline void ProcessNSplit()
    {
        uint64_t powerMain = powerSplit_ * powerLoop_;
        uint64_t powerTail = numN_ - powerMain;
        uint64_t powerTailLoop = powerTail / powerSplit_;
        uint64_t powerTailTail = powerTail % powerSplit_;
        uint64_t masterLoop = powerTailTail != 0 ? 1 : 0;
        masterLoop = powerLoop_ - powerTailLoop - masterLoop;

        LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
        LocalTensor<float> msAccLocal = msAccBuf_.Get<float>();

        LocalTensor<float> scale1Local = outQueueScale1_.AllocTensor<float>();
        LocalTensor<float> scale2Local;
        if (hasY2Scale2_) {
            scale2Local = outQueueScale2_.AllocTensor<float>();
        }

        // Phase A: 段内 Σx² 补偿累加 + x 输出 + fp32 x 暂存(与 legacy 同一条 SubProcessRstd)
        Duplicate(scale1Local, 0.0f, B32_BLOCK_NUM);
        if (hasY2Scale2_) {
            Duplicate(scale2Local, 0.0f, B32_BLOCK_NUM);
        }
        Duplicate(msAccLocal, 0.0f, NUM_FOUR_DF * V_LENGTH);
        PipeBarrier<PIPE_V>();
        SubProcessRstd<false>(rstdLocal, 0, 0, masterLoop, powerTailLoop, powerTailTail, avgFactor_, epsilon_);
        PipeBarrier<PIPE_V>();
        ReduceMsAccToPair();
        VecToScalarSync(); // V→S: 随后标量 GetValue 读向量写入
        float msHi = msAccLocal.GetValue(0);
        float msLo = msAccLocal.GetValue(NUM_TWO_DF * V_LENGTH);
        WriteMsPartial(msHi, msLo);
        SyncAll();

        // Phase B: 每核读全部槽, 固定 k 序两段式合并 → rstd(各核逐位一致, 确定性保持)。
        // rstd 由 RstdScalarFromHi 以 FinalizeRstd 尾段同款向量指令算出(相同 hi → 逐位一致),
        // 再以 Duplicate(标量操作数)广播落 UB。
        float msHiMerged = MergeMsToHi();
        Duplicate(rstdLocal, msHiMerged, B32_BLOCK_NUM);
        PipeBarrier<PIPE_V>();
        float rstd = RstdScalarFromHi(rstdLocal, avgFactor_, epsilon_);
        Duplicate(rstdLocal, rstd, B32_BLOCK_NUM);
        PipeBarrier<PIPE_V>();

        // Phase C: 段内 row_max + y 输出 + inp 暂存(与 legacy 同一条 SubProcessReduceMax)
        SubProcessReduceMax(scale1Local, scale2Local, rstdLocal, 0);
        VecToScalarSync(); // V→S
        float segMax1 = scale1Local.GetValue(0);
        float segMax2 = hasY2Scale2_ ? scale2Local.GetValue(0) : 0.0f;
        WriteMaxPartial(segMax1, segMax2);
        SyncAll();

        // Phase D: 合并全局 row_max → scale 换算 → 段内量化输出; scale 全行仅 1 元素, 0 号核独写
        float gmax1 = 0.0f;
        float gmax2 = 0.0f;
        MergeMax(gmax1, gmax2);
        Duplicate(scale1Local, gmax1, B32_BLOCK_NUM);
        if (hasY2Scale2_) {
            Duplicate(scale2Local, gmax2, B32_BLOCK_NUM);
        }
        PipeBarrier<PIPE_V>();
        // 顺序与 legacy Process 严格一致: ComputeY 要求 scaleLocal[0] 仍是 row_max
        // (den = row_max*PRE/127), 先 Y 后 Scale —— 若先换算成 scale(row_max/127),
        // 分母小 127 倍, y1 全行饱和(实测 62% 错位、边界元素 ±127 饱和)。
        SubProcessY(scale1Local, scale2Local, 0);
        PipeBarrier<PIPE_V>();
        SubProcessScale(scale1Local, scale2Local);
        // 二维: 每行的 seg-0 核独写该行 scale(scale1Gm_ 已按 rowIdx_ 绑定, 长度 1)
        if (k2_ > 0 && static_cast<uint64_t>(blockIdx_) % k2_ == 0) {
            outQueueScale1_.EnQue<float>(scale1Local);
            CopyOutScale(scale1Gm_, outQueueScale1_, 0, 1);
            if (hasY2Scale2_) {
                outQueueScale2_.EnQue<float>(scale2Local);
                CopyOutScale(scale2Gm_, outQueueScale2_, 0, 1);
            }
        }
    }

    // 把 msAccBuf_ 横向规约成 (hi@acc[0], lo@acc[2V]) 一对(FinalizeRstd 前半段原样拆出,
    // 单核路径与 N-split 路径共用, 保证两种划分下横向规约自身逐位一致)。
    __aicore__ inline void ReduceMsAccToPair()
    {
        __local_mem__ float* accAddr = (__ubuf__ float*)msAccBuf_.Get<float>().GetPhyAddr();
        // 成对规约只做到 stride=B32_BLOCK_NUM(8): 整 VL 载入要求 32B 对齐(=8 个 float),
        // stride=4/2/1 会产生非对齐载入, 触发 aivec 报错。
        for (uint32_t stride = V_LENGTH / NUM_TWO_DF; stride >= B32_BLOCK_NUM; stride >>= 1) {
            uint32_t cnt = stride;
            __VEC_SCOPE__
            {
                RegTensor<float> aHi, aLo, bHi, bLo, sHi, sLo, tHi, eReg;
                MaskReg m = UpdateMask<float>(cnt);
                DataCopy<float>(aHi, accAddr);
                DataCopy<float>(bHi, accAddr + stride);
                DataCopy<float>(aLo, accAddr + NUM_TWO_DF * V_LENGTH);
                DataCopy<float>(bLo, accAddr + NUM_TWO_DF * V_LENGTH + stride);
                DfTwoSum(tHi, eReg, aHi, bHi, m);
                Add(eReg, eReg, aLo, m);
                Add(eReg, eReg, bLo, m);
                DfTwoSum(sHi, sLo, tHi, eReg, m);
                DataCopy<float>(accAddr, sHi, m);
                DataCopy<float>(accAddr + NUM_TWO_DF * V_LENGTH, sLo, m);
            }
        }
        // 末 8 个 lane: 广播载入(允许元素粒度偏移, 不受 32B 对齐限制)逐个两段式合并
        __VEC_SCOPE__
        {
            RegTensor<float> sHi, sLo, aHi, aLo, tHi, eReg, s2, e2;
            MaskReg m1 = CreateMask<float, MaskPattern::VL1>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(sHi, accAddr);
            DataCopy<float, LoadDist::DIST_BRC_B32>(sLo, accAddr + NUM_TWO_DF * V_LENGTH);
            for (uint16_t k = 1; k < (uint16_t)B32_BLOCK_NUM; k++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(aHi, accAddr + k);
                DataCopy<float, LoadDist::DIST_BRC_B32>(aLo, accAddr + NUM_TWO_DF * V_LENGTH + k);
                DfTwoSum(tHi, eReg, sHi, aHi, m1);
                Add(eReg, eReg, sLo, m1);
                Add(eReg, eReg, aLo, m1);
                DfTwoSum(s2, e2, tHi, eReg, m1);
                Muls(sHi, s2, static_cast<float>(1.0), m1);
                Muls(sLo, e2, static_cast<float>(1.0), m1);
            }
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(accAddr, sHi, m1);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(accAddr + NUM_TWO_DF * V_LENGTH, sLo, m1);
        }
    }

private:
    // 标量↔向量序(renorm_global.h 同款事件对, 不使用 PIPE_ALL):
    //   VecToScalarSync: 向量写入 → 随后标量 GetValue
    //   ScalarToVecSync: 标量读取段结束 → 随后向量指令可安全复用该 UB
    __aicore__ inline void VecToScalarSync()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(ev);
        WaitFlag<HardEvent::V_S>(ev);
    }

    __aicore__ inline void ScalarToVecSync()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(ev);
        WaitFlag<HardEvent::S_V>(ev);
    }

    // 标量 TwoSum(与 DfTwoSum 同式): 跨核合并槽按 k 序标量读取后逐对合并, 各核执行
    // 相同指令序列 → 结果逐位一致(确定性)。残差非有限即置 0, 同 DfClampResidual 语义。
    static __aicore__ inline void ScalarTwoSum(float a, float b, float& s, float& e)
    {
        s = a + b;
        float bb = s - a;
        float aa = s - bb;
        float da = a - aa;
        float db = b - bb;
        e = da + db;
        if (e != e || e > MAX_FINITE_FP32 || e < -MAX_FINITE_FP32) {
            e = 0.0f;
        }
    }

    // 写本核合并槽并发布(单写单读协议): ms 对(槽0/1)只在 Phase A 写一次, max 对(槽2/3)只在
    // Phase C 写一次 —— 槽不复用, 读侧 L1 无旧行可驻留。若复用槽位, 第二轮读会命中读侧
    // 第一轮留下的陈旧 L1 行(写侧 DCCI 只清写侧缓存, 实测致 max 读旧值 → y1 成片饱和)。
    // 写入全走 Duplicate(标量操作数, 纯向量写), V_MTE3 事件对 + 写侧逐槽 DCCI(renorm 配方)。
    __aicore__ inline void WriteMsPartial(float msHi, float msLo)
    {
        LocalTensor<float> slot = partOutBuf_.Get<float>();
        Duplicate(slot, msHi, PART_SLOT_ELEMS);
        Duplicate(slot[PART_SLOT_ELEMS], msLo, PART_SLOT_ELEMS);
        PublishSlots(0, NUM_TWO_DF);
    }

    __aicore__ inline void WriteMaxPartial(float max1, float max2)
    {
        LocalTensor<float> slot = partOutBuf_.Get<float>();
        Duplicate(slot, max1, PART_SLOT_ELEMS);
        Duplicate(slot[PART_SLOT_ELEMS], hasY2Scale2_ ? max2 : (float)0.0, PART_SLOT_ELEMS);
        PublishSlots(NUM_TWO_DF, NUM_TWO_DF);
    }

    __aicore__ inline void PublishSlots(uint64_t slotIdx, uint64_t slotCnt)
    {
        LocalTensor<float> slot = partOutBuf_.Get<float>();
        PipeBarrier<PIPE_V>();
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopy(partGm_[slotIdx * PART_SLOT_ELEMS], slot, slotCnt * PART_SLOT_ELEMS);
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        for (uint64_t v = 0; v < slotCnt; ++v) {
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                partGm_[(slotIdx + v) * PART_SLOT_ELEMS]);
        }
    }

    // 读第 k 核指定槽对(A5 上跨核向量读可能脏 lane, 故整槽 GM→UB 后标量取值, 同 renorm)。
    __aicore__ inline void ReadSlotPair(uint64_t k, uint64_t slotIdx, float& v0, float& v1)
    {
        LocalTensor<float> slot = partInBuf_.Get<float>();
        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
        DataCopy(slot, partGmAll_[k * PART_SLOTS_PER_CORE * PART_SLOT_ELEMS + slotIdx * PART_SLOT_ELEMS],
                 NUM_TWO_DF * PART_SLOT_ELEMS);
        SetFlag<HardEvent::MTE2_V>(eventMTE2V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V);
        VecToScalarSync(); // MTE2→V→S: 槽载入完成后标量取值
        v0 = slot.GetValue(0);
        v1 = slot.GetValue(PART_SLOT_ELEMS);
        ScalarToVecSync();
    }

    // Σx² 跨核合并(固定 k 序两段式链, 与 FinalizeRstd 逐 slice 累加的补偿结构同型),
    // 返回合并后的高位段(低位段无下游消费者, 同单核路径)。
    __aicore__ inline float MergeMsToHi()
    {
        float accHi = 0.0f;
        float accLo = 0.0f;
        // 只合并本行组的 k2 个槽(全局槽号 rowIdx_*k2_ 起); 每核读自己行的槽, 固定序确定性
        for (uint64_t k = rowIdx_ * k2_; k < (rowIdx_ + 1) * k2_; ++k) {
            float hi, lo;
            ReadSlotPair(k, 0, hi, lo);
            if (k == rowIdx_ * k2_) {
                accHi = hi;
                accLo = lo;
                continue;
            }
            float s, e, s2, e2;
            ScalarTwoSum(accHi, hi, s, e);
            e += accLo;
            e += lo;
            ScalarTwoSum(s, e, s2, e2);
            accHi = s2;
            accLo = e2;
        }
        return accHi;
    }

    // rstd 尾段: 与 FinalizeRstd 尾段**同一套向量指令**(单 lane Mul/Add/Sqrt/Div, 只出高位段),
    // srcLocal[0] 存 msHi —— 给定相同 hi 时两种路径的 rstd 逐位一致。AICore 无标量 sqrtf
    // (__host__ 函数), 标量侧只做 TwoSum 链, 开方/取倒一律走向量单元。
    __aicore__ inline float RstdScalarFromHi(LocalTensor<float>& srcLocal, float avgFactor, float epsilon)
    {
        __local_mem__ float* addr = (__ubuf__ float*)srcLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> msHi, avgReg, tHi, epsReg, oneReg, r0, tmp;
            MaskReg m1 = CreateMask<float, MaskPattern::VL1>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(msHi, addr);
            Duplicate(avgReg, avgFactor, m1);
            Duplicate(epsReg, epsilon, m1);
            Duplicate(oneReg, static_cast<float>(1.0), m1);
            Mul(tHi, msHi, avgReg, m1);
            Add(tHi, tHi, epsReg, m1);
            Sqrt(tmp, tHi, m1);
            Div(r0, oneReg, tmp, m1);
            // t 非有限守卫: Σx² 在 fp32 溢出成 inf 时(x 取到 ±3.4e38 的用例, _072 实证),
            // 未校正的 1/sqrt(inf)=0 已是数学正确值(golden 同样溢出, 得 y=0/scale=0);
            // 而 Markstein 残差 d = t - s0² = inf - inf = NaN 会把它毁成 NaN, 整张 y 被污染。
            // 故留一份未校正值, 校正后按 t 的有限性择一: t 有限时逐位不变(校正照常生效),
            // t=inf 取 0, t=NaN 取 1/sqrt(NaN)=NaN(NaN 传播语义不变)。
            RegTensor<float> r0Keep;
            MaskReg mFiniteT;
            Muls(r0Keep, r0, static_cast<float>(1.0), m1);
            CompareScalar<float, CMPMODE::LT>(mFiniteT, tHi, static_cast<float>(INFINITY), m1);
            // 向量 Sqrt 与 Div 均非正确舍入(_055 实证: sqrt 偏 +1 ULP, 经 1/s 再偏 1 ULP,
            // rstd 连锁使 row_max/scale 各偏 1 ULP)。两级 Markstein 校正到正确舍入
            // (sqrt 校正与 NR+Markstein 除法校正各经 30/50 万随机样本 0 误判):
            //   d = t - s0^2(FMA 精确);  s1 = s0 + d*(0.5*q0)   (q0=1/s0 兼作引导)
            //   e = 1 - s1*q0';  q1 = q0' + q0'*e;  r = 1 - s1*q1;  rstd = q1 + q1*r
            {
                RegTensor<float> negS, eReg, q1Reg, halfReg, dReg, s1Reg;
                Muls(negS, tmp, static_cast<float>(-1.0), m1);
                Muls(dReg, tmp, static_cast<float>(1.0), m1);
                FusedMulDstAdd(dReg, negS, tHi, m1); // d = t - s0*s0
                Muls(halfReg, r0, static_cast<float>(0.5), m1);
                Muls(s1Reg, dReg, static_cast<float>(1.0), m1);
                FusedMulDstAdd(s1Reg, halfReg, tmp, m1); // s1 = s0 + d*0.5*q0(正确舍入 sqrt)
                Div(r0, oneReg, s1Reg, m1);              // q0' = 1/s1(~1 ULP)
                Muls(negS, s1Reg, static_cast<float>(-1.0), m1);
                Muls(eReg, r0, static_cast<float>(1.0), m1);
                FusedMulDstAdd(eReg, negS, oneReg, m1); // e = 1 - s1*q0'
                Muls(q1Reg, eReg, static_cast<float>(1.0), m1);
                FusedMulDstAdd(q1Reg, r0, r0, m1); // q1 = q0' + q0'*e(NR)
                Muls(eReg, q1Reg, static_cast<float>(1.0), m1);
                FusedMulDstAdd(eReg, negS, oneReg, m1); // e = 1 - s1*q1
                FusedMulDstAdd(eReg, q1Reg, q1Reg, m1); // rstd = q1 + q1*e(正确舍入)
                Select(eReg, eReg, r0Keep, mFiniteT);   // t 非有限 -> 取未校正值
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(addr, eReg, m1);
            }
        }
        VecToScalarSync();
        return srcLocal.GetValue(0);
    }

    // row_max 跨核合并: max 精确可结合, 与合并次序无关。NaN 语义对齐段内向量 Max
    // (NaN 传播): 标量 `>` 对 NaN 恒 false 会把含 NaN 段的最大值丢成 0, 使 scale 输出
    // 与 golden(含 NaN 输入用例)不符 —— 任一段为 NaN 即全局 NaN。
    __aicore__ inline void MergeMax(float& gmax1, float& gmax2)
    {
        gmax1 = 0.0f;
        gmax2 = 0.0f;
        for (uint64_t k = rowIdx_ * k2_; k < (rowIdx_ + 1) * k2_; ++k) {
            float m1, m2;
            ReadSlotPair(k, NUM_TWO_DF, m1, m2);
            if (m1 > gmax1 || m1 != m1) {
                gmax1 = m1;
            }
            if (hasY2Scale2_ && (m2 > gmax2 || m2 != m2)) {
                gmax2 = m2;
            }
        }
    }

    __aicore__ inline void CopyInGamma(uint64_t gmOffset, uint64_t blockLen)
    {
        LocalTensor<T_X> gammaLocal = inQueueGamma_.AllocTensor<T_X>();
        RmsNorm::DataCopyImpl<T_X>(gammaLocal, gammaGm_[gmOffset], 1, blockLen);
        inQueueGamma_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyInQuant(uint64_t gmOffset, uint64_t blockLen)
    {
        if (hasSmoothScale1_) {
            LocalTensor<T_SMOOTH_SCALE> smoothScale1Local = inQueueSmoothScale1_.AllocTensor<T_SMOOTH_SCALE>();
            RmsNorm::DataCopyImpl<T_SMOOTH_SCALE>(smoothScale1Local, smoothScale1Gm_[gmOffset], 1, blockLen);
            inQueueSmoothScale1_.EnQue(smoothScale1Local);
        }
        if (hasSmoothScale2_) {
            LocalTensor<T_SMOOTH_SCALE> smoothScale2Local = inQueueSmoothScale2_.AllocTensor<T_SMOOTH_SCALE>();
            RmsNorm::DataCopyImpl<T_SMOOTH_SCALE>(smoothScale2Local, smoothScale2Gm_[gmOffset], 1, blockLen);
            inQueueSmoothScale2_.EnQue(smoothScale2Local);
        }
    }

    __aicore__ inline void CopyInBeta(uint64_t gmOffset, uint64_t blockLen)
    {
        LocalTensor<T_X> betaLocal = inQueueBeta_.AllocTensor<T_X>();
        RmsNorm::DataCopyImpl<T_X>(betaLocal, betaGm_[gmOffset], 1, blockLen);
        inQueueBeta_.EnQue(betaLocal);
    }

    /**
     * @brief Copy in xGm_[srcGmOffset:srcGmOffset+count] and reduce to dst[dstOffset].
     */
    // 竞态修复v7(2026-11-18): 共享 CopyInX 恒走 DataCopyPad(isPad=true), 对齐长度下
    // pad 通路首块行为可疑(v31 曾因变体映射错误未在 fp16 上生效过)。blockLen 已 32B
    // 对齐时改无 pad 直拷。RmsNorm:: 限定名避免与 AscendC 内部 DataCopyImpl 撞名。
    __aicore__ inline void CopyInXNoPad(TQue<QuePosition::VECIN, 1>& inQueue, GlobalTensor<T_X>& srcGm,
                                        uint64_t gmOffset, uint32_t blockLen)
    {
        LocalTensor<T_X> xLocal = inQueue.AllocTensor<T_X>();
        RmsNorm::DataCopyImpl<T_X>(xLocal, srcGm[gmOffset], 1, blockLen);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void ComputeFormerHandle(LocalTensor<float>& dstLocal, uint64_t srcGmOffset, uint64_t dstOffset,
                                               uint64_t workSpaceOffset, uint32_t count, uint32_t powerSplit)
    {
        LocalTensor<float> xOutTmpLocal = xOutTmpBuf_.Get<float>();
        uint32_t calCount = Aligned((uint64_t)(count * sizeof(T_X)), ALIGN_32_FACTOR) / sizeof(T_X);
        // multi-add:按 ascend910b 的次序在 fp32 累加(canndev normal_kernel CopyInX1X2AndAddX1X2
        // 「1.将x1的第一个tensor和x2相加 2.将x1剩余的相加」),即 x = (x1[0]+x2)+x1[1]+..+x1[n-1]。
        // fp32 加法不满足结合律,写成 (Σx1)+x2 会在相消点产生 1~2 个 ULP 的偏差。
        // ReduceSumRstd 内部会做 x1Local+x2Local,故把 A2 次序的「末项」留给它:
        //   n==1: x1Local = x1[0],                        x2Local = x2
        //   n>=2: x1Local = (x1[0]+x2)+x1[1]+..+x1[n-2],  x2Local = x1[n-1]
        // 两种情形下 ReduceSumRstd 的那次加法都恰好是 A2 的最后一步。
        LocalTensor<float> x1Local = x1AccBuf_.Get<float>();
        LocalTensor<float> x2Local = x2Fp32Buf_.Get<float>(); // 中间项的 cast 暂存,末项落定后即为第二操作数
        for (uint32_t i = 0; i < x1Num_; ++i) {
            if (calCount == count) {
                CopyInXNoPad(inQueueX1_, x1GmList_[i], srcGmOffset, count);
            } else {
                CopyInX(inQueueX1_, x1GmList_[i], srcGmOffset, count, 0, calCount - count);
            }
            LocalTensor<T_X> x1In = inQueueX1_.DeQue<T_X>();
            if (i == 0) {
                Cast(x1Local, x1In, RoundMode::CAST_NONE, calCount); // x = x1[0].to(fp32)
                PipeBarrier<PIPE_V>();
                inQueueX1_.FreeTensor(x1In);
                if (x1Num_ > 1) { // x1[0] + x2 紧随其后
                    if (calCount == count) {
                        CopyInXNoPad(inQueueX2_, x2Gm_, srcGmOffset, count);
                    } else {
                        CopyInX(inQueueX2_, x2Gm_, srcGmOffset, count, 0, calCount - count);
                    }
                    LocalTensor<T_X> x2First = inQueueX2_.DeQue<T_X>();
                    Cast(x2Local, x2First, RoundMode::CAST_NONE, calCount);
                    PipeBarrier<PIPE_V>();
                    inQueueX2_.FreeTensor(x2First);
                    Add(x1Local, x1Local, x2Local, calCount);
                    PipeBarrier<PIPE_V>();
                }
                continue;
            }
            Cast(x2Local, x1In, RoundMode::CAST_NONE, calCount); // x1[i].to(fp32)
            PipeBarrier<PIPE_V>();
            inQueueX1_.FreeTensor(x1In);
            if (i + 1 == x1Num_) {
                break; // 末项 x1[n-1] 留在 x2Local,交给 ReduceSumRstd 完成最后一次加法
            }
            Add(x1Local, x1Local, x2Local, calCount); // x += x1[i]
            PipeBarrier<PIPE_V>();
        }
        if (x1Num_ == 1) { // 只有一个 x1 时,第二操作数就是 x2
            if (calCount == count) {
                CopyInXNoPad(inQueueX2_, x2Gm_, srcGmOffset, count);
            } else {
                CopyInX(inQueueX2_, x2Gm_, srcGmOffset, count, 0, calCount - count);
            }
            LocalTensor<T_X> x2In = inQueueX2_.DeQue<T_X>();
            Cast(x2Local, x2In, RoundMode::CAST_NONE, calCount);
            PipeBarrier<PIPE_V>();
            inQueueX2_.FreeTensor(x2In);
        }

        LocalTensor<float> workLocal = reduceSumBuf_.Get<float>();
        uint32_t calNum = Aligned((uint64_t)(count * sizeof(T_X)), ALIGN_512_FACTOR) / sizeof(T_X);
        uint64_t dupLen = calNum - calCount;
        if (dupLen > 0) {
            Duplicate(x1Local[calCount], (float)0.0, dupLen);
            Duplicate(x2Local[calCount], (float)0.0, dupLen);
        }

        // ReduceSumRstd<float,HAS_XOUT=false,HAS_XOUT_FP32=true,IS_RSTD=false>:x1/x2 均 fp32
        //   => xOutTmpLocal = Σx1+x2(fp32 x),dstLocal = Σ((Σx1+x2)^2)(fp32)。
        //   HAS_XOUT=false(x 输出需 T_X,单独 Cast);dummy xOut 传 x1Local(U=float 下 HAS_XOUT=false 不写)。
        NormCommon::ReduceSumRstd<float, false, true, false>(dstLocal, x1Local, xOutTmpLocal, x1Local, x2Local,
                                                             workLocal, dstOffset, calNum, powerSplit);

        AccumSqCompensated(xOutTmpLocal, count);
        PipeBarrier<PIPE_V>();
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        RmsNorm::DataCopyImpl<float>(xOutTmpGm_[workSpaceOffset], xOutTmpLocal, 1, count);
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);

        LocalTensor<T_X> xOutLocal = outQueueX_.AllocTensor<T_X>();
        Cast(xOutLocal, xOutTmpLocal, RoundMode::CAST_RINT,
             calCount); // x 输出 = fp32 x.to(T_X)(round-nearest-even 对齐 ascend910b)
        outQueueX_.EnQue<T_X>(xOutLocal);
        // x 拷出前显式 V→MTE3 依赖: fp32 workspace 拷出(下方)有事件对且从不撕裂;
        // x 路径仅靠队列槽隐式序, 新构建链时序下每核首段 lane0 出现字节级撕裂(62/62,
        // ours=golden&0xFF00)。补上与本应等价的显式边。
        event_t eventVMTE3X = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3X);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3X);
        CopyOutX(xGm_, outQueueX_, srcGmOffset, count);
    }

    // 把本 slice 的 x(fp32) 的平方以补偿求和累加进 msAccBuf_。掩码外 zeroing, 尾块不污染。
    __aicore__ inline void AccumSqCompensated(LocalTensor<float>& xFp32Local, uint32_t count)
    {
        __local_mem__ float* xAddr = (__ubuf__ float*)xFp32Local.GetPhyAddr();
        __local_mem__ float* accAddr = (__ubuf__ float*)msAccBuf_.Get<float>().GetPhyAddr();
        uint16_t repeatTimes = (uint16_t)CeilDivision(count, (uint32_t)V_LENGTH);
        uint32_t calCount = count;
        __VEC_SCOPE__
        {
            RegTensor<float> accHi, accLo, xReg, sqHi, sqLo, tHi, eReg;
            MaskReg maskFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg maskReg;
            DataCopy<float>(accHi, accAddr);
            DataCopy<float>(accLo, accAddr + NUM_TWO_DF * V_LENGTH);
            for (uint16_t i = 0; i < repeatTimes; i++) {
                maskReg = UpdateMask<float>(calCount);
                DataCopy<float>(xReg, xAddr + i * V_LENGTH);
                DfTwoProd(sqHi, sqLo, xReg, xReg, maskReg); // x^2 = sqHi + sqLo(精确)
                DfTwoSum(tHi, eReg, accHi, sqHi, maskFull);
                Add(eReg, eReg, accLo, maskFull);
                Add(eReg, eReg, sqLo, maskFull);
                DfTwoSum(accHi, accLo, tHi, eReg, maskFull);
            }
            DataCopy<float>(accAddr, accHi, maskFull);
            DataCopy<float>(accAddr + NUM_TWO_DF * V_LENGTH, accLo, maskFull);
        }
    }

    // 横向补偿规约(V_LENGTH lane -> 1) + t = ms*avg+eps + rstd = 1/sqrt(t)。
    // 横向合并保留补偿: 普通树形求和会把逐 slice 累加积累的低位段丢掉, 而合并过程中的
    // 残差会回灌 hi, 所以那部分是活的; 只有最终的低位段无下游消费者。
    __aicore__ inline void FinalizeRstd(LocalTensor<float>& rstdLocal, float avgFactor, float epsilon)
    {
        ReduceMsAccToPair();
        __local_mem__ float* accAddr = (__ubuf__ float*)msAccBuf_.Get<float>().GetPhyAddr();

        __local_mem__ float* rAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> msHi, avgReg, tHi, epsReg, oneReg, r0, tmp;
            MaskReg m1 = CreateMask<float, MaskPattern::VL1>();
            DataCopy<float, LoadDist::DIST_BRC_B32>(msHi, accAddr);
            Duplicate(avgReg, avgFactor, m1);
            Duplicate(epsReg, epsilon, m1);
            Duplicate(oneReg, static_cast<float>(1.0), m1);
            // rstd 只出高位段: 下游 ComputeYOut/ComputeReduceMax 的接口只吃单值 rstd,
            // 低位段无处落脚。实测 rstd_lo/rstd_hi ≈ 0.29 ULP, 即便加回 fp32 也被舍掉,
            // 所以这里不做两段式 t —— fl(ms_hi*avg) 与 fl(t+eps) 同原两段式的高位段逐位相同。
            // 注意: 逐 slice 累加与横向合并的补偿仍然保留, 那两处的残差会回灌 hi, 是活的。
            Mul(tHi, msHi, avgReg, m1);
            Add(tHi, tHi, epsReg, m1);
            Sqrt(tmp, tHi, m1);
            Div(r0, oneReg, tmp, m1); // r0 = 1/sqrt(t)
            // t 非有限守卫(与 RstdScalarFromHi 同款, 两路径必须逐位一致): 见该函数注释。
            RegTensor<float> r0Keep;
            MaskReg mFiniteT;
            Muls(r0Keep, r0, static_cast<float>(1.0), m1);
            CompareScalar<float, CMPMODE::LT>(mFiniteT, tHi, static_cast<float>(INFINITY), m1);
            // 与 RstdScalarFromHi 完全相同的两级 Markstein 校正(两路径必须逐位一致):
            // sqrt 校正 + NR+Markstein 除法校正, 均校正到正确舍入。
            {
                RegTensor<float> negS, eReg, q1Reg, halfReg, dReg, s1Reg;
                Muls(negS, tmp, static_cast<float>(-1.0), m1);
                Muls(dReg, tmp, static_cast<float>(1.0), m1);
                FusedMulDstAdd(dReg, negS, tHi, m1); // d = t - s0*s0
                Muls(halfReg, r0, static_cast<float>(0.5), m1);
                Muls(s1Reg, dReg, static_cast<float>(1.0), m1);
                FusedMulDstAdd(s1Reg, halfReg, tmp, m1); // s1 = 正确舍入 sqrt
                Div(r0, oneReg, s1Reg, m1);              // q0' = 1/s1
                Muls(negS, s1Reg, static_cast<float>(-1.0), m1);
                Muls(eReg, r0, static_cast<float>(1.0), m1);
                FusedMulDstAdd(eReg, negS, oneReg, m1); // e = 1 - s1*q0'
                Muls(q1Reg, eReg, static_cast<float>(1.0), m1);
                FusedMulDstAdd(q1Reg, r0, r0, m1); // NR
                Muls(eReg, q1Reg, static_cast<float>(1.0), m1);
                FusedMulDstAdd(eReg, negS, oneReg, m1);
                FusedMulDstAdd(eReg, q1Reg, q1Reg, m1); // 正确舍入 rstd
                Select(eReg, eReg, r0Keep, mFiniteT);   // t 非有限 -> 取未校正值
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rAddr, eReg, m1);
            }
        }
    }

    template <bool IS_RSTD>
    __aicore__ inline void SubProcessRstd(LocalTensor<float> dstLocal, uint64_t position, uint64_t curRow,
                                          uint64_t masterLoop, uint64_t tailLoop, uint64_t tail, float avgFactor = 1.0f,
                                          float epsilon = 0.0f)
    {
        uint64_t workSpaceOffset{0};
        uint64_t offset{curRow};
        uint32_t level1{0};
        uint32_t level2{0};
        uint32_t level3{0};
        LocalTensor<float> level1Local = level1Buf_.Get<float>();
        LocalTensor<float> level2Local = level2Buf_.Get<float>();
        LocalTensor<float> level3Local = level3Buf_.Get<float>();
        LocalTensor<float> tempLocal = tempBuf_.Get<float>();
        Duplicate(level1Local, (float)0.0, RmsNorm::ONCE_VECTOR_SIZE);
        Duplicate(level2Local, (float)0.0, RmsNorm::ONCE_VECTOR_SIZE);
        Duplicate(level3Local, (float)0.0, RmsNorm::ONCE_VECTOR_SIZE);
        // Stage1: Cal TailLoop
        // Use TailBlock as MainBlock and reduce 2 nearby Mainblock twice.
        for (uint32_t repeat = 0; repeat < tailLoop; repeat++) {
            // Tail
            ComputeFormerHandle(tempLocal, offset, 0, workSpaceOffset, powerSplit_, powerSplit_);
            offset += powerSplit_;
            workSpaceOffset += powerSplit_;
            // Main
            ComputeFormerHandle(tempLocal, offset, 1, workSpaceOffset, powerSplit_, powerSplit_);
            offset += powerSplit_;
            workSpaceOffset += powerSplit_;

            RmsNorm::ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
            level1 += 1;
            RmsNorm::ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        }
        // Stage2: Cal TailTail
        // Use TailBlock as MainBlock and reduce 2 nearby Mainblock twice.
        uint64_t powerSplitHalf = powerSplit_ / CONST_FACTOR_2;
        if (tail > 0 && tail <= powerSplitHalf) {
            // Tail
            ComputeFormerHandle(tempLocal, offset, 0, workSpaceOffset, powerSplitHalf + tail, powerSplitHalf);
            offset += powerSplitHalf + tail;
            workSpaceOffset += powerSplitHalf + tail;
            // Main
            ComputeFormerHandle(tempLocal, offset, 1, workSpaceOffset, powerSplitHalf, powerSplitHalf);
            offset += powerSplitHalf;
            workSpaceOffset += powerSplitHalf;
        } else if (tail > powerSplitHalf) {
            // Half Main & Half Tail
            ComputeFormerHandle(tempLocal, offset, 0, workSpaceOffset, powerSplit_, powerSplit_);
            offset += powerSplit_;
            workSpaceOffset += powerSplit_;
            // Other Tail
            ComputeFormerHandle(tempLocal, offset, 1, workSpaceOffset, tail, powerSplitHalf);
            offset += tail;
            workSpaceOffset += tail;
        }
        if (tail > 0) {
            RmsNorm::ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
            level1 += 1;
            RmsNorm::ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        }
        // Stage3: Cal MasterLoop
        for (uint32_t repeat = 0; repeat < masterLoop; repeat++) {
            ComputeFormerHandle(level1Local, offset, level1, workSpaceOffset, powerSplit_, powerSplit_);
            offset += powerSplit_;
            workSpaceOffset += powerSplit_;
            level1 += 1;
            RmsNorm::ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        }
        ComputeMultiLevelRstd<IS_RSTD>(dstLocal, position, level1Local, level2Local, level3Local, level1, level2,
                                       avgFactor, epsilon);
    }

    __aicore__ inline void SubProcessReduceMax(LocalTensor<float>& scale1Local, LocalTensor<float>& scale2Local,
                                               LocalTensor<float>& rstdLocal, uint64_t mIdx)
    {
        LocalTensor<float> xOutTmpLocal = xOutTmpBuf_.Get<float>();
        LocalTensor<float> y1TmpLocal = y1TmpBuf_.Get<float>();
        LocalTensor<float> y2TmpLocal;
        if (hasY2Scale2_) {
            y2TmpLocal = y2TmpBuf_.Get<float>();
        }
        for (uint32_t nIdx = 0; nIdx < nCnt_; nIdx++) {
            uint64_t realN = (nIdx == nCnt_ - 1) ? tailN_ : baseN_;
            uint64_t realNDtypeAlign = (nIdx == nCnt_ - 1) ? tailNDtypeAlign_ : baseNDtypeAlign_;
            CopyInGamma(nIdx * baseN_, realN);
            CopyInQuant(nIdx * baseN_, realN);
            if (hasBeta_) {
                CopyInBeta(nIdx * baseN_, realN);
            }
            LocalTensor<T_X> gammaLocal = inQueueGamma_.DeQue<T_X>();
            LocalTensor<T_SMOOTH_SCALE> smoothScale1Local;
            LocalTensor<T_SMOOTH_SCALE> smoothScale2Local;
            if (hasSmoothScale1_) {
                smoothScale1Local = inQueueSmoothScale1_.DeQue<T_SMOOTH_SCALE>();
            }
            if (hasSmoothScale2_) {
                smoothScale2Local = inQueueSmoothScale2_.DeQue<T_SMOOTH_SCALE>();
            }
            LocalTensor<T_X> betaLocal;
            if (hasBeta_) {
                betaLocal = inQueueBeta_.DeQue<T_X>();
            }

            uint64_t gmOffsetX = mIdx * numN_ + nIdx * baseN_;
            uint64_t gmOffsetYTmp = nIdx * baseN_;
            uint64_t gmOffsetScale = 0;
            DataCopyPadExtParams<float> padParams{
                true,                    // isPad
                static_cast<uint8_t>(0), // leftPadding
                static_cast<uint8_t>(0), // rightPadding
                static_cast<float>(0.0)  // paddingValue
            };
            event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::V_MTE2>(eventVMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
            RmsNorm::DataCopyImpl<float>(xOutTmpLocal, xOutTmpGm_[gmOffsetYTmp], 1, realN, 0, 0, padParams);
            SetFlag<HardEvent::MTE2_V>(eventMTE2V);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V);

            SetOverflowMode<T_Y>(0);
            // row_max 的规约元素数必须是真实 N(与 PERF 路径传 baseN_ 一致), 不能传 32B 对齐长度:
            // 对齐段里 [realN, realNDtypeAlign) 是 padding —— x 的 padding 被 DataCopyPad 显式置 0,
            // 但 gamma/smooth_scale 的搬入(CopyInQuant/CopyInGamma)不补零, padding 留着上一段残留值。
            // 残留为有限值时 0*stale=0 无害, 残留为 ±inf 时 0*(±inf)=NaN, 再经 MergeMax 的 NaN
            // 传播把整行 scale 变成 NaN(_180 实证: ss1 恒为 -inf, 设备 scale1=NaN 而 golden=+inf)。
            DispatchReduceMax<T_X, T_SMOOTH_SCALE, T_Y>(scale1Local, y1TmpLocal, xOutTmpLocal, rstdLocal, gammaLocal,
                                                        betaLocal, smoothScale1Local, 0, realN, hasSmoothScale1_,
                                                        hasBeta_);
            if (hasY2Scale2_) {
                DispatchReduceMax<T_X, T_SMOOTH_SCALE, T_Y>(scale2Local, y2TmpLocal, xOutTmpLocal, rstdLocal,
                                                            gammaLocal, betaLocal, smoothScale2Local, 0, realN,
                                                            hasSmoothScale2_, hasBeta_);
            }
            SetOverflowMode<T_Y>(oriOverflowMode_);

            // y(RmsNorm 结果,量化前)= xOut * rstd * gamma;两阶段里只在此(reducemax 阶段)按 n 分块各写一次。
            LocalTensor<T_X> yLocal = outQueueY_.AllocTensor<T_X>();
            ComputeYOut<T_X, T_X>(yLocal, xOutTmpLocal, rstdLocal, gammaLocal, 0, baseNDtypeAlign_);
            outQueueY_.EnQue<T_X>(yLocal);
            CopyOutY(yGm_, outQueueY_, gmOffsetX, realN);

            inQueueGamma_.FreeTensor(gammaLocal);
            if (hasSmoothScale1_) {
                inQueueSmoothScale1_.FreeTensor(smoothScale1Local);
            }
            if (hasSmoothScale2_) {
                inQueueSmoothScale2_.FreeTensor(smoothScale2Local);
            }
            if (hasBeta_) {
                inQueueBeta_.FreeTensor(betaLocal);
            }

            // Copy (rmsOut * smooth) to workspace
            event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::V_MTE3>(eventVMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
            RmsNorm::DataCopyImpl<float>(y1TmpGm_[gmOffsetYTmp], y1TmpLocal, 1, realN);
            if (hasY2Scale2_) {
                RmsNorm::DataCopyImpl<float>(y2TmpGm_[gmOffsetYTmp], y2TmpLocal, 1, realN);
            }
            SetFlag<HardEvent::MTE3_V>(eventMTE3V);
            WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        }
    }

    __aicore__ inline void SubProcessScale(LocalTensor<float>& scale1Local, LocalTensor<float>& scale2Local)
    {
        ComputeScale<T_Y>(scale1Local, 0);
        if (hasY2Scale2_) {
            ComputeScale<T_Y>(scale2Local, 0);
        }
    }

    __aicore__ inline void SubProcessY(LocalTensor<float>& scale1Local, LocalTensor<float>& scale2Local, uint64_t mIdx)
    {
        LocalTensor<float> y1TmpLocal = y1TmpBuf_.Get<float>();
        LocalTensor<float> y2TmpLocal;
        if (hasY2Scale2_) {
            y2TmpLocal = y2TmpBuf_.Get<float>();
        }

        for (uint32_t nIdx = 0; nIdx < nCnt_; nIdx++) {
            uint64_t realN = (nIdx == nCnt_ - 1) ? tailN_ : baseN_;
            uint64_t realNDtypeAlign = (nIdx == nCnt_ - 1) ? tailNDtypeAlign_ : baseNDtypeAlign_;
            uint64_t gmOffsetXY = mIdx * numN_ + nIdx * baseN_;
            uint64_t gmOffsetYTmp = nIdx * baseN_;

            LocalTensor<T_Y> y1Local = outQueueY1_.AllocTensor<T_Y>();
            LocalTensor<T_Y> y2Local;
            if (hasY2Scale2_) {
                y2Local = outQueueY2_.AllocTensor<T_Y>();
            }
            // Copy (rmsOut * smooth) from workspace
            DataCopyPadExtParams<float> padParams{
                true,                    // isPad
                static_cast<uint8_t>(0), // leftPadding
                static_cast<uint8_t>(0), // rightPadding
                static_cast<float>(0.0)  // paddingValue
            };
            event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::V_MTE2>(eventVMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
            RmsNorm::DataCopyImpl<float>(y1TmpLocal, y1TmpGm_[gmOffsetYTmp], 1, realN, 0, 0, padParams);
            if (hasY2Scale2_) {
                RmsNorm::DataCopyImpl<float>(y2TmpLocal, y2TmpGm_[gmOffsetYTmp], 1, realN, 0, 0, padParams);
            }
            SetFlag<HardEvent::MTE2_V>(eventMTE2V);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V);

            // y1Tmp/y2Tmp 里已是量化前的 inp(含 smooth), ComputeY 只需 inp 与 row_max,
            // 不再按 slice 重取 smooth_scale。
            ComputeY<T_Y>(y1Local, scale1Local, y1TmpLocal, 0, baseNDtypeAlign_);
            if (hasY2Scale2_) {
                ComputeY<T_Y>(y2Local, scale2Local, y2TmpLocal, 0, baseNDtypeAlign_);
            }

            outQueueY1_.EnQue<T_Y>(y1Local);
            if (hasY2Scale2_) {
                outQueueY2_.EnQue<T_Y>(y2Local);
            }
            CopyOutY(y1Gm_, outQueueY1_, gmOffsetXY, realN);
            if (hasY2Scale2_) {
                CopyOutY(y2Gm_, outQueueY2_, gmOffsetXY, realN);
            }
        }
    }

    template <typename T_GAMMA, typename T_SMOOTH, typename T_YB8>
    __aicore__ inline void DispatchReduceMax(LocalTensor<float>& scaleLocal, LocalTensor<float>& yTmpLocal,
                                             LocalTensor<float>& xLocal, LocalTensor<float>& rstdLocal,
                                             LocalTensor<T_GAMMA>& gammaLocal, LocalTensor<T_GAMMA>& betaLocal,
                                             LocalTensor<T_SMOOTH>& smoothScaleLocal, uint32_t rstdScaleOffset,
                                             uint32_t calCount, bool hasSmoothScale, bool hasBeta)
    {
        if (hasSmoothScale) {
            if (hasBeta) {
                ComputeReduceMax<float, T_GAMMA, T_SMOOTH, true, true, T_YB8>(scaleLocal, yTmpLocal, xLocal, rstdLocal,
                                                                              gammaLocal, betaLocal, smoothScaleLocal,
                                                                              rstdScaleOffset, calCount);
            } else {
                ComputeReduceMax<float, T_GAMMA, T_SMOOTH, true, false, T_YB8>(scaleLocal, yTmpLocal, xLocal, rstdLocal,
                                                                               gammaLocal, betaLocal, smoothScaleLocal,
                                                                               rstdScaleOffset, calCount);
            }
        } else {
            if (hasBeta) {
                ComputeReduceMax<float, T_GAMMA, T_SMOOTH, false, true, T_YB8>(scaleLocal, yTmpLocal, xLocal, rstdLocal,
                                                                               gammaLocal, betaLocal, smoothScaleLocal,
                                                                               rstdScaleOffset, calCount);
            } else {
                ComputeReduceMax<float, T_GAMMA, T_SMOOTH, false, false, T_YB8>(
                    scaleLocal, yTmpLocal, xLocal, rstdLocal, gammaLocal, betaLocal, smoothScaleLocal, rstdScaleOffset,
                    calCount);
            }
        }
    }

private:
    TPipe* pipe_ = nullptr;
    // GM Buffer
    GlobalTensor<float> scale1Gm_;
    GlobalTensor<float> scale2Gm_;
    GlobalTensor<float> y1TmpGm_;
    GlobalTensor<float> y2TmpGm_;
    GlobalTensor<float> xOutTmpGm_;
    static constexpr uint32_t MAX_X1_NUM = 5; // x1 TensorList 最大长度(0~4 个 Add → 1~5 个 tensor)
    GlobalTensor<T_X> x1GmList_[MAX_X1_NUM];  // multi-add:x1 列表各 tensor
    GlobalTensor<T_X> x2Gm_;
    GlobalTensor<T_X> gammaGm_;
    GlobalTensor<T_X> betaGm_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_X> yGm_; // 增:y 输出(RmsNorm 结果)
    GlobalTensor<T_SMOOTH_SCALE> smoothScale1Gm_;
    GlobalTensor<T_SMOOTH_SCALE> smoothScale2Gm_;
    GlobalTensor<T_Y> y1Gm_;
    GlobalTensor<T_Y> y2Gm_;

    // UB Buffer
    TQue<QuePosition::VECIN, 1> inQueueX1_;
    TQue<QuePosition::VECIN, 1> inQueueX2_;
    TQue<QuePosition::VECIN, 1> inQueueGamma_;
    TQue<QuePosition::VECIN, 1> inQueueBeta_;
    TQue<QuePosition::VECIN, 1> inQueueSmoothScale1_;
    TQue<QuePosition::VECIN, 1> inQueueSmoothScale2_;
    TQue<QuePosition::VECOUT, 1> outQueueY1_;
    TQue<QuePosition::VECOUT, 1> outQueueY2_;
    TQue<QuePosition::VECOUT, 1> outQueueY_; // 增:y(RmsNorm 结果)输出
    TQue<QuePosition::VECOUT, 1> outQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueScale1_;
    TQue<QuePosition::VECOUT, 1> outQueueScale2_;
    TBuf<TPosition::VECCALC> rstdBuf_;
    TBuf<TPosition::VECCALC> reduceSumBuf_;
    TBuf<TPosition::VECCALC> level1Buf_;
    TBuf<TPosition::VECCALC> level2Buf_;
    TBuf<TPosition::VECCALC> level3Buf_;
    TBuf<TPosition::VECCALC> tempBuf_;
    TBuf<TPosition::VECCALC> y1TmpBuf_;
    TBuf<TPosition::VECCALC> y2TmpBuf_;
    TBuf<TPosition::VECCALC> xOutTmpBuf_;
    TBuf<TPosition::VECCALC> x1AccBuf_;   // 增:multi-add Σx1 fp32 累加缓冲
    TBuf<TPosition::VECCALC> x2Fp32Buf_;  // 增:x2 fp32 缓冲(与 Σx1 同以 fp32 喂 reduce)
    TBuf<TPosition::VECCALC> msAccBuf_;   // 增:Σx² 补偿累加(前 V_LENGTH lane=hi, 后 V_LENGTH=lo)
    TBuf<TPosition::VECCALC> partOutBuf_; // 增:N-split 本核合并槽写出缓冲(64B)
    TBuf<TPosition::VECCALC> partInBuf_;  // 增:N-split 他核合并槽读入缓冲(64B)
    GlobalTensor<float> partGm_;          // 增:N-split 本核合并槽(写视图)
    GlobalTensor<float> partGmAll_;       // 增:N-split 全部合并槽(读视图)

    // Tiling data
    uint64_t numN_{0};
    uint64_t numM_{0};
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t baseNDtypeAlign_{0};
    uint64_t tailNDtypeAlign_{0};
    uint64_t baseNReduceAlign_{0};
    uint64_t reduceSumBufAlign_{0};
    uint64_t powerSplit_{0};
    uint64_t powerLoop_{0};
    uint64_t mPerCore_{0};
    uint64_t mLastCore_{0};
    float epsilon_{0};
    float avgFactor_{0};
    uint32_t x1Num_{1}; // multi-add TensorList 长度(1~5)
    bool hasSmoothScale1_{false};
    bool hasSmoothScale2_{false};
    bool hasBeta_{false};
    bool hasY2Scale2_{false};
    // Platform
    int64_t blockIdx_{0};
    int64_t blockNum_{0};
    // N 维跨核切分(issue#43)
    uint64_t nSplitNum_{1};
    bool invalidTiling_{false}; // tiling 自洽校验失败: Process 空转返回, 不做任何 GM 写
    uint64_t nSegLen_{0};
    uint64_t nSegTail_{0};
    uint64_t oriNumN_{0}; // 整行长度(split 前的 numN, 供 workspace 面积/合并槽寻址)
    uint64_t segStart_{0};
    uint64_t k2_{0};      // 二维: 每行列段数(=nSplitNum/numM; numM==1 时=nSplitNum)
    uint64_t rowIdx_{0};  // 二维: 本核行号
    uint64_t rowBase_{0}; // 二维: 本核行基址(=rowIdx_*oriNumN)
    // Cal params
    uint64_t mCore_;
    uint64_t nCnt_;
    uint64_t tailN_;
    uint64_t baseNB8Align_;
    uint64_t baseNB32Align_;
    // Other
    int64_t oriOverflowMode_{0};
};
} // namespace MultiAddRmsNormDynamicQuant
#endif // _MULTI_ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_SPILT_REDUCE_H_
