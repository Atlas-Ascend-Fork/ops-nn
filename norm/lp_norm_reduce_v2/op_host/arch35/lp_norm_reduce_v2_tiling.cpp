/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lp_norm_reduce_v2_tiling.cpp
 * \brief LpNormReduceV2 Tiling 实现（arch35）——迭代二全量分档整合
 *
 * 流程（DESIGN §4/§5，迭代二按穿刺结论整合）：
 *   1. 平台信息（ubSize / AIV coreNum）
 *   2. 属性与 shape 读取（p / axes / keepdim；dtype / numel / rank 校验）
 *   3. pBranch 判定（全 6 分支一次到位，±inf 别名判定，NaN 拒绝；DESIGN §1.6/§6.3）
 *   4. 合轴段表（标记 A/R → 消除 size=1 → 相邻同类型合并 → 段表 + P1 (B,R,C)；0 段退化→(B=O,R=1,C=1)
 *      走 AR；≥2 段 → P2 段表路径，DESIGN §4.1/§4.3，迭代三）
 *   5. 空张量 / R==0 快速路径（§5.6；R==0 时 bLength 展平为 O=oTotal，覆盖 P1/P2 形态）
 *   6. LAYOUT 判定：C==1 → AR(0)；C>1 → ARA(1)（DESIGN §4.2）
 *   7. 输出空间切分（outBlockNum = min(bLen, coreNum)）+ CROSS_CORE 判定（§5.2）：
 *      usedCores×2 ≤ coreNum 且 R ≥ 2×minRPerCore → rGroupNum = min(coreNum/usedCores,
 *      CeilDiv(R, minRPerCore))；rGroupNum<2 回退直出（SUG-5）；workspace 超限回退直出
 *      （workspace 容量 ×2：tile 奇偶 ping-pong 双 bank——H1 跨 tile WAR 竞争修复）
 *   8. BUFFER_MODE 判定（PT-02/PT-11）：两遍求解——先按单缓冲解 tile 参数估计每核 chunk 数，
 *      ≥4 chunk 启用双缓冲（每核 ≤2 chunk 无收益），再按最终 bufNum 复解
 *   9. UB 两步闭式求解（§5.3 BLK-2；BR_ZERO 分支预算含 Compares/Select 辅助块 +5B/元素，
 *      PT-12；CROSS_CORE 预算含 wsOut/wsIn 队列 +8B/行）
 *  10. TilingKey 选择（dtype × CROSS_CORE × LAYOUT × BUFFER_MODE，24 实例全量）+ workspace 声明
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "base/err_msg.h"
#include "graph/utils/type_utils.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/lp_norm_reduce_v2_tiling_data.h"
#include "../../op_kernel/arch35/lp_norm_reduce_v2_tiling_key.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

constexpr uint32_t WORKSPACE_NUM = 1;
constexpr int64_t MAX_NUMEL = 2147483647;      // numel(x) ≤ 2^31-1（spec global_constraints）
constexpr int64_t MAX_RANK = 8;                // rank(x) ≤ 8
constexpr int64_t MAX_COPY_BLOCK_COUNT = 4095; // 2D DataCopyPad blockCount 上限
// CROSS_CORE rGroup 起评（DESIGN §5.2：minRPerCore 起评值 32768 元素，实测调优）
constexpr int64_t MIN_R_PER_CORE = 32768;
// 双缓冲启用判据（PT-02/PT-11 实测：每核 chunk 数 ≥4 有正收益，≥8 显著；≤2 无收益）
constexpr int64_t MIN_CHUNKS_FOR_DOUBLE_BUFFER = 4;
// workspace 上限（REQUIREMENTS §8.2 数 MB 量级约束；超限回退直出——防御，正常分档远小于此值）
constexpr int64_t MAX_WORKSPACE_BYTES = 16 * 1024 * 1024;
// ARA C 向 tile 上限（DESIGN §5.2：a0TileBase=64，tileA0Len ≤ 64）
constexpr int64_t ARA_C_TILE_MAX = 64;
// P2 段空间 chunk 上限（DESIGN §4.3/PT-13 §3.2.4：UB 收集预算显式建模——chunk × 全 p 分支缓冲
// ≤ 176KB（fp32+ZERO 最坏：gather/f32/calc/tmp/ones 5×32KB + mask 8KB + stage/queue 4KB），
// 恒低于 950 UB 可用上界；rTotal 超出时按 chunk 分片串行合并，正确性不受影响）
constexpr int64_t P2_CHUNK_CAP = 8192;
// P2 输出 tile 上限（回写批量，PT-13 LP13_O_TILE 同款）
constexpr int64_t P2_O_TILE_CAP = 512;
// 跨核路由开关（迭代三已修复 GE 链路 workspace 故障，issues/issue_20261104_ge_ws_write_fault.md）：
//   迭代二曾因 GE 链路 kernel 写 workspace 触发 aivec error 95 而默认关闭；迭代三上板定位根因 =
//   GE kernel-task 链路的 kernel workspace 形参被填为 真实缓冲区基址+16MB（aclnn 约定残留，该偏移
//   区未分配/未映射），与 workspace 写本身无关（迭代二实验受 FE kernel cache / 包 staging 陈旧产物
//   污染，"直调正常/GE 崩"二分结论有效但归因有误）。kernel 侧已改为经 GetSysWorkSpacePtr() 取
//   真实分配区基址（op_kernel/arch35/lp_norm_reduce_v2.h Init CROSS_CORE 分支），GE 链路实测通过，
//   跨核路由恢复默认开启。
//   门控两态 UT：tests/ut 以 -DLPNORM_CROSS_CORE_ROUTING_ENABLED=1 单独编译 crosscore 测试
//   二进制验证显式启用态路由（宏默认已改 1，生产构建与主 UT 二进制同为准核启用态）。
#ifndef LPNORM_CROSS_CORE_ROUTING_ENABLED
#define LPNORM_CROSS_CORE_ROUTING_ENABLED 1
#endif
constexpr bool ENABLE_CROSS_CORE_ROUTING = (LPNORM_CROSS_CORE_ROUTING_ENABLED != 0);
constexpr int64_t UB_FIXED_RESERVE = 8192; // TQue/TPipe 系统开销 + 安全余量（字节）
// ±inf 别名：2147483647.0 / -2147483648.0 在 fp32 属性存储下舍入为 ±2^31（DESIGN §1.6）
constexpr float POS_INF_ALIAS = 2147483648.0f;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape; // rank=0 标量按 shape {1} 退化处理（合轴后 O=1, R=1）
    }
    return in_shape;
}

// 获取平台信息 ubSize / AIV coreNum（internal_error 防御，DESIGN §8）
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    if (unlikely(*coreNum == 0)) {
        OP_LOGE_WITHOUT_REPORT("LpNormReduceV2", "TilingFunc: platform AIV coreNum is 0.");
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"LpNormReduceV2", "platform AIV coreNum is 0"}));
        return ge::GRAPH_FAILED;
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    if (unlikely(*ubSize == 0)) {
        OP_LOGE_WITHOUT_REPORT("LpNormReduceV2", "TilingFunc: platform UB size is 0.");
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"LpNormReduceV2", "platform UB size is 0"}));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// workspace 声明：即使大小为 0 也必须调用 GetWorkspaceSizes 声明 slot，
// 否则运行时不会为 kernel 分配 workspace 指针（CROSS_CORE=1 时按分档结果覆写实值）
static ge::graphStatus DeclareWorkspace(gert::TilingContext* context, size_t bytes)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = bytes;
    return ge::GRAPH_SUCCESS;
}

// p 分支判定（判定优先级对齐 spec.formula：+inf 别名 > -inf 别名 > 0 > 1 > 整数[2,15] > 其他有限浮点）
static ge::graphStatus JudgePBranch(gert::TilingContext* context, float p, int32_t* pBranch, int32_t* powTimes)
{
    OP_CHECK_IF(
        std::isnan(p),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "p", "NaN",
                                              "p supports +inf, -inf and any finite float value; NaN is not supported"),
        return ge::GRAPH_FAILED);
    const float posInf = std::numeric_limits<float>::infinity();
    if (p == posInf || p == POS_INF_ALIAS) {
        *pBranch = LPNORM_P_BRANCH_POS_INF; // max|x|
    } else if (p == -posInf || p == -POS_INF_ALIAS) {
        *pBranch = LPNORM_P_BRANCH_NEG_INF; // min|x|
    } else if (p == 0.0f) {
        *pBranch = LPNORM_P_BRANCH_ZERO; // 非零元素计数
    } else if (p == 1.0f) {
        *pBranch = LPNORM_P_BRANCH_ONE; // sum|x|
    } else if (p == std::floor(p) && p >= 2.0f && p <= 15.0f) {
        *pBranch = LPNORM_P_BRANCH_INT_POW; // 连乘 sum|x|^p
        *powTimes = static_cast<int32_t>(p) - 1;
    } else {
        *pBranch = LPNORM_P_BRANCH_EXP; // sum exp(p·ln|x|)
    }
    return ge::GRAPH_SUCCESS;
}

// 合轴结果（DESIGN §4.1/§4.3，迭代三段表化）：标记 A/R → 消除 size=1 → 相邻同类型合并，
// 输出交替段表（stride=后缀积）；P1（rSegNum≤1）整理为单 R 段 (B, R, C)，
// rSegNum==0 退化（归约轴全 size=1 消除）→ (B=O, R=1, C=1) 恒等归约走 AR（PT-17 §2.3.1）
struct MergedShape {
    int64_t b = 1;
    int64_t r = 1;
    int64_t c = 1;
    int32_t rSegNum = 0; // R 段数：0=全 A 退化 / 1=P1 / ≥2=P2
    int32_t aSegNum = 0; // A 段数（rank≤8 → A/R 段各 ≤4）
    int64_t rSize[4] = {0};
    int64_t rStride[4] = {0};
    int64_t aSize[4] = {0};
    int64_t aStride[4] = {0};
    int64_t rTotal = 1; // Π rSize（每输出元素归约总数；0 段退化 =1）
    int64_t oTotal = 1; // Π aSize（输出元素总数 O；含 P2 中段 A=0 场景）
};

// 段表构建（DESIGN §4.1 全步骤：消除只看 size 不看标记 → 消除后相邻同类型合并，顺序不可交换，
// PT-17 case08）：stride[s] = 段后所有段大小积（元素步进）；A 表 aSize[0] 为输出最外层基数
static ge::graphStatus BuildMergedSegments(gert::TilingContext* context, const gert::Shape& xShape,
                                           const gert::TypedContinuousVector<int64_t>* axes, MergedShape* out)
{
    const int64_t rank = static_cast<int64_t>(xShape.GetDimNum());
    std::set<int64_t> axesSet;
    if (axes->GetSize() == 0UL) {
        for (int64_t i = 0; i < rank; i++) {
            axesSet.insert(i);
        }
    } else {
        OP_CHECK_IF(static_cast<int64_t>(axes->GetSize()) > rank,
                    OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON(context->GetNodeName(), "axes",
                                                             std::to_string(axes->GetSize()).c_str(),
                                                             "axes size must not exceed rank of x"),
                    return ge::GRAPH_FAILED);
        for (size_t i = 0; i < axes->GetSize(); i++) {
            int64_t a = (axes->GetData())[i];
            if (a < 0) {
                a += rank; // 负索引归一化（对齐 InferShape4LpNorm）
            }
            OP_CHECK_IF(a < 0 || a >= rank,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "axes", std::to_string((axes->GetData())[i]).c_str(),
                            "axis is out of range [-rank(x), rank(x)) after negative-index normalization"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(axesSet.count(a) > 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axes", std::to_string(a).c_str(),
                                                              "duplicate elements in axes are not allowed"),
                        return ge::GRAPH_FAILED);
            axesSet.insert(a);
        }
    }

    // 1、段序列：消除 size=1（不看标记）→ 相邻同类型合并（消除后）
    constexpr int32_t SEG_MAX = 8;
    int64_t segSize[SEG_MAX] = {0};
    bool segIsReduce[SEG_MAX] = {false};
    int32_t segCnt = 0;
    for (int64_t i = 0; i < rank; i++) {
        const int64_t dimSize = xShape.GetDim(i);
        if (dimSize == 1) {
            continue; // size=1 冗余维消除（不影响地址计算与归约语义）
        }
        const bool isReduce = axesSet.count(i) > 0;
        if (segCnt > 0 && segIsReduce[segCnt - 1] == isReduce) {
            segSize[segCnt - 1] *= dimSize; // 相邻同类型段合并
        } else {
            segIsReduce[segCnt] = isReduce;
            segSize[segCnt] = dimSize;
            segCnt++;
        }
    }
    // 2、后缀积 stride：stride[s] = Σ_{t>s} segSize[t]（元素步进）
    int64_t segStride[SEG_MAX] = {0};
    int64_t suffixProd = 1;
    for (int32_t s = segCnt - 1; s >= 0; s--) {
        segStride[s] = suffixProd;
        suffixProd *= segSize[s];
    }
    // 3、拆分 A/R 段表 + P1 (B, R, C) 整理（b=R 段前 A 积 / c=R 段后 A 积）
    bool metReduce = false;
    int64_t bProd = 1;
    int64_t cProd = 1;
    for (int32_t s = 0; s < segCnt; s++) {
        if (segIsReduce[s]) {
            OP_CHECK_IF(out->rSegNum >= 4, OP_LOGE(context, "LpNormReduceV2: R segment count exceeds 4."),
                        return ge::GRAPH_FAILED);
            out->rSize[out->rSegNum] = segSize[s];
            out->rStride[out->rSegNum] = segStride[s];
            out->rSegNum++;
            out->rTotal *= segSize[s];
            if (!metReduce) {
                out->r = segSize[s]; // 首个（P1 唯一）R 段
                metReduce = true;
            }
        } else {
            OP_CHECK_IF(out->aSegNum >= 4, OP_LOGE(context, "LpNormReduceV2: A segment count exceeds 4."),
                        return ge::GRAPH_FAILED);
            out->aSize[out->aSegNum] = segSize[s];
            out->aStride[out->aSegNum] = segStride[s];
            out->aSegNum++;
            out->oTotal *= segSize[s];
            if (metReduce) {
                cProd *= segSize[s];
            } else {
                bProd *= segSize[s];
            }
        }
    }
    out->b = bProd;
    out->c = cProd;
    if (out->rSegNum == 0) {
        // 0 段 R 退化（PT-17 §2.3.1）：R=1 恒等归约，(B=O, R=1, C=1) 走 AR（kernel 无需特判）
        out->r = 1;
        out->b = out->oTotal;
        out->c = 1;
    }
    return ge::GRAPH_SUCCESS;
}

// AR 布局 UB 两步闭式求解（DESIGN §5.3 BLK-2；tmpBuf 按 E 元素最坏占用量化——保守覆盖 3510
// AR 归约 tmp 峰值 dimA×mainR ≤ E 的所有 mainR 取值，issue_20261104_reduce_tmpbuf_1 口径）：
//   ① 先解行 chunk 上界 ubR（按单行最小 tile 开销）
//   ② 再解 rowsTile（行批量吃满余量），输出 tile 缓冲与 O 无关（BLK-1）
// 预算项：in：bufNum×typeSize + fp32Buf：4 + tmp：4 + [ZERO] ones+mask：5；每行输出侧：
//   chunkResult+globalResult：8 + outQueue：typeSize + [CROSS] wsOut+wsIn：8
// 固定额外开销（必须显式入账，否则 cap-bound 场景余量 < 额外量时 TPipe 分配越过可用 UB 上限，
// 表现为 MTE 指令地址越界 aivec error 95——迭代二上板实测定位）：
//   inQueue 每 buffer 槽 +256B VEC LoadAlign 越界读余量（PT-09）×bufNum + chunkResult 8 元素
//   StoreUnAlign 保护槽（PT-05）等 ≈ 288B 实测额外量 → fixedExtra = 256×bufNum + 1024
//   （含安全余量：迭代一实测可用上界 245644B（253952 − 系统 ~8.3KB），本式保证 kernel 分配
//     ≤ ubAvail − 736B，恒低于该上界 ≥600B）
static void SolveUbTilingAR(uint64_t ubSize, int64_t rSpan, int64_t typeSize, int64_t ubBlockElems, int64_t bufNum,
                            int64_t bFactor, bool isZeroBranch, bool isCrossCore, int32_t pBranch, int64_t* ubR,
                            int64_t* rowsTile)
{
    const int64_t ubAvail = static_cast<int64_t>(ubSize) - UB_FIXED_RESERVE -
                            (256 * bufNum + 1024 + 8192 + 8192); // +8KB kahanBuf（4.2c）
    const int64_t perElemBytes = bufNum * typeSize + 8 + (isZeroBranch ? 5 : 0);
    const int64_t outPerRowBytes = 8 + typeSize + (isCrossCore ? 8 : 0);
    int64_t ubRCap = (ubAvail - outPerRowBytes) / perElemBytes;
    // sum 类分支（非 ±inf 非计数）chunk 内 Pattern 树形深度收敛（4.2c 配套）：树形舍入误差
    // ~log2(chunk)×ULP/2，chunk=20K 时 ~0.03（L1 cross_check 比值线超限实测）；cap 1024 后
    // log2=10 层 ≈0.008，与竞品分块树形同级。±inf/计数无累加舍入不受影响；性能代价为
    // 大 R 场景 MTE 搬运块变小（后续可两级树形恢复，DESIGN §9.1 债务 D4）。
    if (!isZeroBranch && pBranch != LPNORM_P_BRANCH_POS_INF && pBranch != LPNORM_P_BRANCH_NEG_INF) {
        ubRCap = std::min<int64_t>(ubRCap, 1024);
    }
    if (ubRCap < 1) {
        ubRCap = 1; // 防御（正常 UB 容量远大于单块）
    }
    int64_t localUbR;
    if (rSpan < ubBlockElems) {
        localUbR = rSpan; // 短行：chunk 即整段（缓冲区按对齐后长度分配）
    } else {
        localUbR = std::min(rSpan, ubRCap);
        localUbR = FloorAlign(localUbR, ubBlockElems);
        if (localUbR < ubBlockElems) {
            localUbR = ubBlockElems; // 防御（正常 UB 容量远大于单块）
        }
    }
    const int64_t ubRAlloc = CeilAlign(localUbR, ubBlockElems);
    const int64_t perRowBytes = ubRAlloc * perElemBytes + outPerRowBytes;
    int64_t localRowsTile = ubAvail / perRowBytes;
    localRowsTile = std::max<int64_t>(localRowsTile, 1);
    localRowsTile = std::min(localRowsTile, bFactor);
    localRowsTile = std::min(localRowsTile, MAX_COPY_BLOCK_COUNT); // 2D DataCopyPad blockCount ≤ 4095
    *ubR = localUbR;
    *rowsTile = localRowsTile;
}

// ARA 布局 UB 两步闭式求解（DESIGN §5.3 / PT-09 布局）：
//   ubFactorC = min(64, C)（tileA0Len）；rAlignedCols = CeilAlign(ubFactorC, 32B/typeSize)
//   ① rChunk（RA 归约行数，≤4095 blockCount 上限）② bTile（批方向吃满余量，≤4095）
// 预算项同 AR（每批行输出侧按 rAlignedCols 缩放）
static void SolveUbTilingARA(uint64_t ubSize, int64_t rSpan, int64_t cLength, int64_t typeSize, int64_t ubBlockElems,
                             int64_t bufNum, int64_t bFactor, bool isZeroBranch, bool isCrossCore, int64_t* rChunk,
                             int64_t* bTile, int64_t* ubFactorC, int64_t* rAlignedCols)
{
    const int64_t ubAvail = static_cast<int64_t>(ubSize) - UB_FIXED_RESERVE -
                            (256 * bufNum + 1024 + 8192 + 8192); // +8KB kahanBuf（4.2c）
    const int64_t localUbFactorC = std::min(ARA_C_TILE_MAX, cLength);
    const int64_t localAlignedCols = CeilAlign(localUbFactorC, ubBlockElems);
    const int64_t perElemBytes = bufNum * typeSize + 8 + (isZeroBranch ? 5 : 0);
    const int64_t outPerBatchBytes = localAlignedCols * (8 + typeSize + (isCrossCore ? 8 : 0));
    int64_t rChunkCap = (ubAvail - outPerBatchBytes) / (localAlignedCols * perElemBytes);
    if (rChunkCap < 1) {
        rChunkCap = 1; // 防御
    }
    int64_t localRChunk = std::min(rSpan, rChunkCap);
    localRChunk = std::min(localRChunk, MAX_COPY_BLOCK_COUNT); // 2D DataCopyPad blockCount ≤ 4095
    int64_t bTileCap = (ubAvail - localRChunk * localAlignedCols * perElemBytes - outPerBatchBytes) / outPerBatchBytes;
    int64_t localBTile = std::max<int64_t>(bTileCap, 1);
    localBTile = std::min(localBTile, bFactor);
    localBTile = std::min(localBTile, MAX_COPY_BLOCK_COUNT); // CopyOut blockCount ≤ 4095
    *rChunk = localRChunk;
    *bTile = localBTile;
    *ubFactorC = localUbFactorC;
    *rAlignedCols = localAlignedCols;
}

// TilingKey 选择（参数顺序与 ASCENDC_TPL_ARGS_DECL 一致：D_T_X / CROSS_CORE / LAYOUT / BUFFER_MODE；
// key 位布局：dtype | CROSS_CORE<<8 | LAYOUT<<16 | BUFFER_MODE<<24）
static inline void SelectTilingKey(gert::TilingContext* context, ge::DataType dataType, uint64_t crossCore,
                                   uint64_t layout, uint64_t bufferMode)
{
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType), crossCore, layout, bufferMode);
}

// tiling 分发入口
static ge::graphStatus LpNormReduceV2TilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT("LpNormReduceV2", "TilingFunc: GetPlatformInfo failed; "
                                                         "propagate lower-layer first error."),
                return ge::GRAPH_FAILED);

    // 2、获取 shape、dtype、属性信息
    auto inputX = context->GetInputShape(0); // const gert::StorageShape*
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    const gert::Shape inputShapeX = EnsureNotScalar(inputX->GetStorageShape()); // rank=0 → {1}
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    OP_CHECK_IF(supportedDtype.count(dataType) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(dataType),
                                                      "only FLOAT16, FLOAT and BFLOAT16 are supported"),
                return ge::GRAPH_FAILED);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* pAttr = attrs->GetAttrPointer<float>(0); // 属性索引 0：p
    OP_CHECK_NULL_WITH_CONTEXT(context, pAttr);
    const auto* axesAttr = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(1); // 属性索引 1：axes
    OP_CHECK_NULL_WITH_CONTEXT(context, axesAttr);
    // epsilon（索引 3）不参与计算（对齐 TBE），keepdim（索引 2）仅影响输出 shape

    const int64_t rank = static_cast<int64_t>(inputShapeX.GetDimNum());
    OP_CHECK_IF(rank > MAX_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(rank).c_str(),
                                                         "rank of x must be in [0, 8]"),
                return ge::GRAPH_FAILED);
    const int64_t numel = inputShapeX.GetShapeSize();
    OP_CHECK_IF(numel > MAX_NUMEL,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "x", std::to_string(numel).c_str(),
                                                          "total elements must not exceed 2147483647"),
                return ge::GRAPH_FAILED);

    // 3、设置 TilingData（空间由 kernel 侧 REGISTER_TILING_DEFAULT 决定）
    LpNormReduceV2TilingData* tiling = context->GetTilingData<LpNormReduceV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    if (unlikely(memset_s(tiling, sizeof(LpNormReduceV2TilingData), 0, sizeof(LpNormReduceV2TilingData)) != EOK)) {
        OP_LOGE_WITHOUT_REPORT("LpNormReduceV2", "TilingFunc: memset tiling data failed.");
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"LpNormReduceV2", "memset tiling data failed"}));
        return ge::GRAPH_FAILED;
    }

    // 5、pBranch 判定（全 6 分支一次到位；pBranch/pVal/powTimes 运行期下发 kernel）
    int32_t pBranch = LPNORM_P_BRANCH_INT_POW;
    int32_t powTimes = 1;
    OP_CHECK_IF(JudgePBranch(context, *pAttr, &pBranch, &powTimes) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT("LpNormReduceV2", "TilingFunc: JudgePBranch failed; "
                                                         "propagate lower-layer first error."),
                return ge::GRAPH_FAILED);
    tiling->pBranch = pBranch;
    tiling->pVal = *pAttr;
    tiling->powTimes = powTimes;
    const bool isZeroBranch = (pBranch == LPNORM_P_BRANCH_ZERO);

    // 6、合轴段表构建（DESIGN §4.1/§4.3：P1 (B,R,C) / P2 段表 / 0 段退化）
    MergedShape ms;
    OP_CHECK_IF(BuildMergedSegments(context, inputShapeX, axesAttr, &ms) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT("LpNormReduceV2", "TilingFunc: BuildMergedSegments failed; "
                                                         "propagate lower-layer first error."),
                return ge::GRAPH_FAILED);
    const int64_t bLen = ms.b;
    const int64_t rLen = ms.r;
    const int64_t cLen = ms.c;
    tiling->bLength = bLen;
    tiling->rLength = rLen;
    tiling->cLength = cLen;
    tiling->segNum = ms.rSegNum; // 实际 R 段数（0 退化 / 1 P1 / ≥2 P2）
    const int64_t typeSize = (dataType == ge::DT_FLOAT) ? 4 : 2;
    const int64_t ubBlockElems = 32 / typeSize; // 32B 对齐元素数：fp32=8, fp16/bf16=16

    // 7、空张量快速路径（DESIGN §5.6；oTotal 为全 A 段积，覆盖 P2 中段 A=0 场景）
    if (ms.oTotal == 0) {
        // O==0（非归约轴为 0 → 输出空 Tensor）：blockDim=1 空跑，无 GM 写
        tiling->segNum = 1; // 快速路径按 AR 空跑口径（kernel segNum≥2 才走 P2）
        tiling->coreNum = 1;
        tiling->outBlockNum = 1;
        tiling->rGroupNum = 1;
        tiling->bFactor = 0;
        tiling->rowsTile = 1;
        tiling->bTile = 1;
        tiling->ubFactorR = 1;
        tiling->ubFactorC = 1;
        tiling->rAlignedCols = 1;
        tiling->rFactorPerGroup = 0;
        tiling->zeroOutputFlag = 0;
        context->SetBlockDim(1);
        SelectTilingKey(context, dataType, 0, 0, 0);
        OP_CHECK_IF(DeclareWorkspace(context, 0U) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT("LpNormReduceV2", "TilingFunc: DeclareWorkspace failed; "
                                                             "propagate lower-layer first error."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (ms.rTotal == 0) {
        // R==0 且 O>0（归约轴为空，如 shape [0,4] axes=[] 或 [2,0,3] axes=[1]）：
        // 输出全 0（spec boundary）。bLength 展平为 O=oTotal（线性输出元素数，覆盖 P1/P2 形态）
        int64_t solveR = 1; // dummy R：仅保证缓冲区有界
        int64_t solveB = 1;
        SolveUbTilingAR(ubSize, solveR, typeSize, ubBlockElems, 1, ms.oTotal, false, false, pBranch, &solveR, &solveB);
        tiling->segNum = 1; // 快速路径按 AR 置零口径（kernel segNum≥2 才走 P2）
        tiling->bLength = ms.oTotal;
        tiling->rLength = 0; // 归约空集语义（kernel 置零路径不读 rLength，字段口径统一）
        tiling->cLength = 1; // bLength 已展平为 O，输出缓冲口径 bLength×cLength = O
        tiling->coreNum = 1;
        tiling->outBlockNum = 1;
        tiling->rGroupNum = 1;
        tiling->bFactor = ms.oTotal;
        tiling->rowsTile = solveB;
        tiling->bTile = 1;
        tiling->ubFactorR = solveR;
        tiling->ubFactorC = 1;
        tiling->rAlignedCols = 1;
        tiling->rFactorPerGroup = 0;
        tiling->zeroOutputFlag = 1;
        context->SetBlockDim(1);
        SelectTilingKey(context, dataType, 0, 0, 0);
        OP_CHECK_IF(DeclareWorkspace(context, 0U) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT("LpNormReduceV2", "TilingFunc: DeclareWorkspace failed; "
                                                             "propagate lower-layer first error."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    // 8、P2 多段 R 通用路径（DESIGN §4.3，迭代三；PT-13 段循环工程化，正确性优先）：
    //    段表直传 TilingData，按输出切核（CROSS_CORE=0 直出、LAYOUT=0 实例运行期分支）；
    //    bLength=O、rLength=rTotal（段乘积空间）、rowsTile=输出 tile、ubFactorR=段空间 chunk
    if (ms.rSegNum >= 2) {
        for (int32_t j = 0; j < ms.rSegNum; j++) {
            tiling->segSize[j] = ms.rSize[j];
            tiling->segStride[j] = ms.rStride[j];
        }
        tiling->aSegNum = ms.aSegNum;
        for (int32_t a = 0; a < ms.aSegNum; a++) {
            tiling->aSize[a] = ms.aSize[a];
            tiling->aStride[a] = ms.aStride[a];
        }
        const int64_t oTotal = ms.oTotal;
        const int64_t outBlockNum = std::min(oTotal, coreNum);
        const int64_t bFactor = CeilDiv(oTotal, outBlockNum);
        const int64_t usedCores = CeilDiv(oTotal, bFactor);
        const int64_t chunkCap = std::min(ms.rTotal, P2_CHUNK_CAP);
        const int64_t ubFactorR = CeilAlign(chunkCap, ubBlockElems);     // 段空间 chunk（对齐，尾部 pad）
        const int64_t oTile = std::min<int64_t>(P2_O_TILE_CAP, bFactor); // 输出 tile（回写批量）
        tiling->bLength = oTotal;
        tiling->rLength = ms.rTotal;
        tiling->cLength = 1;
        tiling->coreNum = usedCores;
        tiling->outBlockNum = outBlockNum;
        tiling->rGroupNum = 1;
        tiling->bFactor = bFactor;
        tiling->rowsTile = oTile;
        tiling->bTile = 1;
        tiling->ubFactorR = ubFactorR;
        tiling->ubFactorC = 1;
        tiling->rAlignedCols = 1;
        tiling->rFactorPerGroup = ms.rTotal;
        tiling->zeroOutputFlag = 0;
        context->SetBlockDim(usedCores);
        // key=(dtype, CROSS_CORE=0, LAYOUT=0, BUFFER_MODE=0)：P2 不经 inQueue/MTE2 与 workspace
        SelectTilingKey(context, dataType, 0, 0, 0);
        OP_CHECK_IF(DeclareWorkspace(context, 0U) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT("LpNormReduceV2", "TilingFunc: DeclareWorkspace failed; "
                                                             "propagate lower-layer first error."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    // 9、LAYOUT 判定（DESIGN §4.2）：C==1 → AR；C>1 → ARA（bf16 迭代二放开，PT-08 CAST_RINT 验证）
    const int64_t layout = (cLen == 1) ? 0 : 1;

    // 10、输出空间切分（两级切分之输出优先，DESIGN §5.2）
    const int64_t outBlockNum = std::min(bLen, coreNum);
    const int64_t bFactor = CeilDiv(bLen, outBlockNum);
    const int64_t usedCores = CeilDiv(bLen, bFactor);

    // 11、CROSS_CORE 判定：输出分块无法吃满核数（usedCores×2 ≤ coreNum）且 R 足够大（≥2×起评值）
    // （ENABLE_CROSS_CORE_ROUTING=false 时恒直出——见常量定义处注释与关联 issue）
    int64_t rGroupNum = 1;
    bool crossCore = false;
    if (ENABLE_CROSS_CORE_ROUTING && usedCores * 2 <= coreNum && rLen >= 2 * MIN_R_PER_CORE) {
        const int64_t maxGroups = coreNum / usedCores;
        const int64_t groupsByR = CeilDiv(rLen, MIN_R_PER_CORE);
        int64_t groups = std::min(maxGroups, groupsByR);
        if (groups >= 2) {
            rGroupNum = groups;
            crossCore = true;
        } // rGroupNum < 2 → 回退直出（SUG-5：跳过无收益的二次归并空跑）
    }
    int64_t rFactorPerGroup = crossCore ? CeilDiv(rLen, rGroupNum) : rLen;

    // 12/13、BUFFER_MODE 两遍求解 + workspace 上限校验（超限回退直出重解）
    int64_t ubFactorR = 1;
    int64_t rowsTile = 1;
    int64_t bTile = 1;
    int64_t ubFactorC = 1;
    int64_t rAlignedCols = 1;
    uint64_t bufferMode = 0;
    int64_t wsBytes = 0;
    for (int32_t attempt = 0; attempt < 2; attempt++) {
        const int64_t rSpan = crossCore ? rFactorPerGroup : rLen;
        // 第一遍：单缓冲求解 → 估计每核 chunk 数（tile 数 × 行内 chunk 数，PT-11 口径）
        int64_t estChunks = 1;
        if (layout == 0) {
            int64_t ubR1 = 1;
            int64_t rowsTile1 = 1;
            SolveUbTilingAR(ubSize, rSpan, typeSize, ubBlockElems, 1, bFactor, isZeroBranch, crossCore, pBranch, &ubR1,
                            &rowsTile1);
            estChunks = CeilDiv(bFactor, rowsTile1) * CeilDiv(rSpan, ubR1);
        } else {
            int64_t rChunk1 = 1;
            int64_t bTile1 = 1;
            int64_t ubFactorC1 = 1;
            int64_t rAlignedCols1 = 1;
            SolveUbTilingARA(ubSize, rSpan, cLen, typeSize, ubBlockElems, 1, bFactor, isZeroBranch, crossCore, &rChunk1,
                             &bTile1, &ubFactorC1, &rAlignedCols1);
            estChunks = CeilDiv(bFactor, bTile1) * CeilDiv(cLen, ubFactorC1) * CeilDiv(rSpan, rChunk1);
        }
        bufferMode = (estChunks >= MIN_CHUNKS_FOR_DOUBLE_BUFFER) ? 1 : 0;
        const int64_t bufNum = (bufferMode == 1) ? 2 : 1;

        // 第二遍：按最终 bufNum 求解 tile 参数
        if (layout == 0) {
            SolveUbTilingAR(ubSize, rSpan, typeSize, ubBlockElems, bufNum, bFactor, isZeroBranch, crossCore, pBranch,
                            &ubFactorR, &rowsTile);
            bTile = 1;
            ubFactorC = 1;
            rAlignedCols = 1;
        } else {
            SolveUbTilingARA(ubSize, rSpan, cLen, typeSize, ubBlockElems, bufNum, bFactor, isZeroBranch, crossCore,
                             &ubFactorR, &bTile, &ubFactorC, &rAlignedCols);
            rowsTile = 1;
        }

        // workspace 上限校验（CROSS_CORE=1）：slot = tile 输出元素数（AR：rowsTile；ARA：bTile×rAlignedCols）
        // （H1 ping-pong：容量 ×2——tile 奇偶双 bank，规避 tile t 的 Phase2 读与 tile t+1 的
        // Phase1 写跨核 WAR 竞争；16MB 上限回退直出逻辑不变，超限仍回退 rGroupNum=1 重解）
        if (crossCore) {
            const int64_t tileOutElems = (layout == 0) ? rowsTile : bTile * rAlignedCols;
            const int64_t slotBytes = CeilAlign(tileOutElems * static_cast<int64_t>(sizeof(float)),
                                                WS_SLOT_ALIGN_BYTES);
            wsBytes = 2 * outBlockNum * rGroupNum * slotBytes;
            if (wsBytes <= MAX_WORKSPACE_BYTES) {
                break; // 分档成立
            }
            // 回退直出（SUG-5 同款防御）：rGroupNum=1 重解
            OP_LOGI(context, "LpNormReduceV2: workspace %ld bytes exceeds limit %ld, fallback to direct mode.", wsBytes,
                    MAX_WORKSPACE_BYTES);
            crossCore = false;
            rGroupNum = 1;
            rFactorPerGroup = rLen;
            wsBytes = 0;
            continue; // 第二次 attempt 必然通过（直出无 workspace）
        }
        break;
    }

    tiling->coreNum = crossCore ? usedCores * rGroupNum : usedCores;
    tiling->outBlockNum = outBlockNum;
    tiling->rGroupNum = rGroupNum;
    tiling->bFactor = bFactor;
    tiling->rowsTile = rowsTile;
    tiling->bTile = bTile;
    tiling->ubFactorR = ubFactorR;
    tiling->ubFactorC = ubFactorC;
    tiling->rAlignedCols = rAlignedCols;
    tiling->rFactorPerGroup = rFactorPerGroup;
    tiling->zeroOutputFlag = 0;
    context->SetBlockDim(tiling->coreNum);

    // 14、设置 TilingKey（映射到 lp_norm_reduce_v2_regbase.cpp kernel 入口模板参数）+ workspace 声明
    // （单次声明：提前按 0 声明再覆写实值会在 FE 动态编译链路下丢失 workspace 分配——迭代二上板实测）
    SelectTilingKey(context, dataType, crossCore ? 1U : 0U, static_cast<uint64_t>(layout), bufferMode);
    OP_CHECK_IF(DeclareWorkspace(context, static_cast<size_t>(wsBytes)) != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT("LpNormReduceV2", "TilingFunc: DeclareWorkspace failed; "
                                                         "propagate lower-layer first error."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLpNormReduceV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct LpNormReduceV2CompileInfo {}; // 必须定义, 入图场景依赖

// tiling 注册入口
IMPL_OP_OPTILING(LpNormReduceV2)
    .Tiling(LpNormReduceV2TilingFunc)
    .TilingParse<LpNormReduceV2CompileInfo>(TilingParseForLpNormReduceV2);

} // namespace optiling
