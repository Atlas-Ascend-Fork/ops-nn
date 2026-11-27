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
 * \file lp_norm_reduce_v2_tiling_key.h
 * \brief LpNormReduceV2 TilingKey 模板参数定义（4 个正交维度：dtype × CROSS_CORE × LAYOUT × BUFFER_MODE）
 *
 * ✅ 必须使用 ASCENDC_TPL_ARGS_DECL 模板编程方式
 * ❌ 禁止使用废弃的 TILING_KEY_IS 宏
 *
 * p 分支不进 TilingKey（DESIGN §6.3）：pBranch/pVal/powTimes 为 TilingData 运行期字段。
 */

#ifndef __LP_NORM_REDUCE_V2_TILING_KEY_H__
#define __LP_NORM_REDUCE_V2_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// DECL：4 维正交参数（与 kernel 入口模板参数一一对应）
//   D_T_X      输入 dtype（fp16 / bf16 / fp32，绑定输入 0）
//   CROSS_CORE 0=各核直出 / 1=跨核（workspace partial + SyncAll 二次归并）
//   LAYOUT     0=AR（C==1，尾轴/全归约/前缀）/ 1=ARA（C>1，非尾轴，Pattern::Reduce::RA）
//   BUFFER_MODE 0=单缓冲 / 1=双缓冲（仅 inQueueX 生效）
ASCENDC_TPL_ARGS_DECL(LpNormReduceV2,
                      ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_BF16, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
                      ASCENDC_TPL_UINT_DECL(CROSS_CORE, 8, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(LAYOUT, 8, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1));

// SEL：迭代二全量 24 实例 = {fp32, fp16, bf16} × CROSS_CORE{0,1} × LAYOUT{AR,ARA} × BUFFER_MODE{0,1}
// （PLAN.md §2.1；TilingKey 位布局：dtype | CROSS_CORE<<8 | LAYOUT<<16 | BUFFER_MODE<<24）
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
                                     ASCENDC_TPL_UINT_SEL(CROSS_CORE, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_UINT_SEL(BUFFER_MODE, ASCENDC_TPL_UI_LIST, 1)));

#endif // __LP_NORM_REDUCE_V2_TILING_KEY_H__
