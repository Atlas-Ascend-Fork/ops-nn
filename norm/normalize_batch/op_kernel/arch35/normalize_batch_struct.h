/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Ascend C template parameter (TPL) declarations for NormalizeBatch.
//   TilingKey 模板参数按双 bool 轴声明（isAllFeatures / isBigD），与 kernel
//   entry 的函数模板参数一一对应：
//   - isAllFeatures：normalize_type == "all_features"（同一 n 的所有 c 共享统计量）
//   - isBigD：单遍数据（行/平面）放不进 UB → d 分段两遍（统计趟 + 归一化趟）
//   dtype（float32）由框架按 REG_OP 输入名经 DTYPE_INPUT_X 编译期实例化，不进 TilingKey。
//   tilingKey 数值由框架宏 GET_TPL_TILING_KEY(isAllFeatures, isBigD) 生成。
//
//   TPL_SEL_0 (0,0) per_feature + SmallD：行驻 UB 单遍（x 读 1 次）
//   TPL_SEL_1 (0,1) per_feature + BigD  ：d 分段两遍（x 读 2 次）
//   TPL_SEL_2 (1,0) all_features + SmallD：平面驻 UB 单遍（x 读 1 次）
//   TPL_SEL_3 (1,1) all_features + BigD  ：行批两遍（x 读 2 次；整行驻留时 R 行
//             逐行拷贝背靠背 + 整批一次同步驻留，行自身超预算时逐行 d 分段）

#ifndef OPS_NORM_NORMALIZE_BATCH_TILING_KEY_H_
#define OPS_NORM_NORMALIZE_BATCH_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h" // ASCENDC_TPL macros

ASCENDC_TPL_ARGS_DECL(NormalizeBatch, ASCENDC_TPL_BOOL_DECL(isAllFeatures, 0, 1), ASCENDC_TPL_BOOL_DECL(isBigD, 0, 1));

ASCENDC_TPL_SEL(
    // per_feature + SmallD：行驻 UB 单遍
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isAllFeatures, 0), ASCENDC_TPL_BOOL_SEL(isBigD, 0)),
    // per_feature + BigD：d 分段两遍
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isAllFeatures, 0), ASCENDC_TPL_BOOL_SEL(isBigD, 1)),
    // all_features + SmallD：平面驻 UB 单遍
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isAllFeatures, 1), ASCENDC_TPL_BOOL_SEL(isBigD, 0)),
    // all_features + BigD：行 × d 分段两遍
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isAllFeatures, 1), ASCENDC_TPL_BOOL_SEL(isBigD, 1)));

#endif // OPS_NORM_NORMALIZE_BATCH_TILING_KEY_H_
