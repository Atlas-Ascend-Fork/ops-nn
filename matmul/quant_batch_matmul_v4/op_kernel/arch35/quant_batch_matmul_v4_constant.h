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
 * \file quant_batch_matmul_v4_constant.h
 * \brief
 */
#pragma once

namespace QuantBatchMatmulV4 {
enum class QuantType : uint32_t {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
    MX = 4,
};

constexpr int32_t IDX_0 = 0;
constexpr int32_t OFFSET_FOR_4BITS = 128;
constexpr int32_t GROUP_SIZE_32 = 32;
constexpr int32_t GROUP_SIZE_HIF4 = 64;
constexpr int32_t SCALE_COPY_GROUP_SIZE = 2;
constexpr int32_t SCALE_COPY_DEFAULT_STRIDE = 0;
constexpr int32_t SCALE_COPY_DEFAULT_NS_STRIDE = 1;
constexpr int64_t ONE_BLK_ELEM_B4 = 64;
} // namespace QuantBatchMatmulV4
