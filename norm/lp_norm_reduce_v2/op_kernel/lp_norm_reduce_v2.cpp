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
 * \file lp_norm_reduce_v2.cpp
 * \brief LpNormReduceV2 算子 kernel 入口（arch35 架构：Ascend 950PR / 950DT，DAV_3510）
 *        （nn 仓约定：op_kernel 根目录以 <snake>.cpp 命名；源工程 regbase.cpp 迁移重命名，主体不变）
 *
 * 模板参数（与 lp_norm_reduce_v2_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义对应）：
 *   - D_T_X:      输入 dtype（fp32 / fp16 / bf16）
 *   - CROSS_CORE: 0=各核直出 / 1=跨核二次归并
 *   - LAYOUT:     0=AR（尾轴/全归约/前缀归约）/ 1=ARA（非尾轴归约）
 *   - BUFFER_MODE: 0=单缓冲 / 1=双缓冲
 *
 * workspace ABI 说明（GE kernel-task 链路，迭代三上板定位修复，
 * issues/issue_20261104_ge_ws_write_fault.md）：
 *   - FE 动态编译产物按 framework-launch ABI 编译；运行时按 TilingFunc 声明大小分配用户 workspace，
 *     实际缓冲区基址经 KFC workspace 通道下发（kernel 侧 GetSysWorkSpacePtr() 可取到，上板 dump 实证
 *     形参 = GetSysWorkSpacePtr() + RESERVED_WORKSPACE(16MB)）；
 *   - 该 +16MB 偏移是 aclnn 执行器约定（executor 分配 [16MB sys][user] 连续区）在 GE kernel-task
 *     链路的残留——本链路未在形参前实际分配 16MB 保留区，形参值落入未映射 DDR，任何 MTE/标量访问
 *     均触发 aivec error 95（DDR 地址越界）；GetUserWorkspace()（= GetSysWorkSpacePtr()+16MB）同因失效；
 *   - 故 kernel 侧跨核 partial 交换的 wsGm_ 基址取 GetSysWorkSpacePtr()（真实分配区，容量与
 *     TilingFunc 声明一致），workspace 形参不消费；
 *   - 若后续接入 aclnn 执行器链路（executor 会真实分配 [16MB sys][user]），需按该链路改回
 *     GetUserWorkspace/形参语义并单独适配验证。
 */

#include "arch35/lp_norm_reduce_v2_kernel.h"

template <typename D_T_X, int CROSS_CORE, int LAYOUT, int BUFFER_MODE>
__global__ __aicore__ void lp_norm_reduce_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LpNormReduceV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(LpNormReduceV2TilingData, tilingData, tiling);
    (void)workspace; // GE kernel-task 链路下形参非真实用户 workspace 基址（见文件头说明）
    NsLpNormReduceV2::LpNormReduceV2<D_T_X, CROSS_CORE, LAYOUT, BUFFER_MODE> op;
    op.Init(x, y, workspace, &tilingData);
    op.Process();
}
