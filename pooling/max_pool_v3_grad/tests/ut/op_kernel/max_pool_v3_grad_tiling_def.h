/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool_v3_grad_tiling_def.h
 * \brief kernel UT tiling 宏定义(tiling 结构体由 kernel 侧 max_pool_grad_struct.h 提供,
 *        与 max_pool3d_grad_with_argmax 的 def.h 同构)
 */

#ifndef _GE_MAX_POOL_V3_GRAD_TILING_DEF_H_
#define _GE_MAX_POOL_V3_GRAD_TILING_DEF_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"
#include "securec.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

template <typename T>
inline void InitTilingData(uint8_t* tiling, T* constData)
{
    const auto ret = memcpy_s(constData, sizeof(T), tiling, sizeof(T));
    if (ret != EOK) {
        (void)fprintf(stderr, "InitTilingData memcpy_s failed, ret=%d.\n", static_cast<int>(ret));
        std::abort();
    }
}

#define REGISTER_TILINGDATA_SIZE(tiling_struct, counter)

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#endif
