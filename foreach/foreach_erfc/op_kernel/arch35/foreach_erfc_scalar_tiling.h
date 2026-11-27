/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FOREACH_ERFC_SCALAR_TILING_H
#define FOREACH_ERFC_SCALAR_TILING_H
#include <cstdint>
// Ordinary inputsTensorUbSize is a positive UB element count below this bit.
// A flagged value stores only the scalar tensor count in the remaining bits.
constexpr uint32_t FOREACH_ERFC_SCALAR_FLAG = uint32_t(1) << 31;
#endif
