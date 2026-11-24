/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TARGET_VALUE_RANGE_H
#define TARGET_VALUE_RANGE_H

#include <cstdint>
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"

// target 值域校验（host 侧）。target 的取值范围是 [0, C-1]（见 aclnnNLLLoss.md），属于 device 数据，
// host 侧静态校验覆盖不到；越界 target 在 legacy kernel（arch22）与 arch35 release kernel 下
// 均会静默产生错误 loss/梯度（详见 issue #9235）。
// 语义对齐 PyTorch "Target X is out of bounds"，ignore_index 元素豁免（与 PyTorch 一致）。

// 纯 host 校验：逐元素检查 hostBuf 中 targetNumel 个 target（按 dtype 解释）是否落在
// [0, nClasses)，ignoreIndex 豁免。越界报 ACLNN_ERR_PARAM_INVALID 并返回 false。
// opapi 库以 -fvisibility=hidden 编译，非 aclnn* 接口符号默认不导出，
// 需显式 default visibility 供 opapi UT 链接。
__attribute__((visibility("default"))) bool ValidateTargetValues(const void* hostBuf, int64_t targetNumel,
                                                                 op::DataType dtype, int64_t nClasses,
                                                                 int64_t ignoreIndex);

// 入口封装：target 需连续、零偏移、地址可解析才会做 D2H 回读校验；否则跳过（行为与修复前一致）。
// 回读失败（如 UT 等无 device 数据环境）降级为跳过校验。
__attribute__((visibility("default"))) bool CheckTargetValueRange(const aclTensor* target, int64_t nClasses,
                                                                  int64_t ignoreIndex);

#endif // TARGET_VALUE_RANGE_H
