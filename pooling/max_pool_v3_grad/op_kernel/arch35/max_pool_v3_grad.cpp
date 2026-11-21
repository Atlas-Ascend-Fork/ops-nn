/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool_v3_grad.cpp
 * \brief Kernel entry for MaxPoolV3Grad.
 *
 * MaxPoolV3Grad only adds attributes on top of MaxPoolGrad.  The host tiling
 * path normalizes those attributes to the common SIMT, NCHW small/big-kernel or
 * NHWC fast-path tiling data, so the MaxPoolGrad implementations are reused
 * as-is, including their NaN-propagate max-selection semantics.
 *
 * NHWC 快速通道(参考 max_pool3d_grad NDHWC 实现去除 D 维): kernelMode = TPL_NHWC_KERNEL,
 * 运行时按 tilingData.isBigKernel 分流小C小kernel / 小C大kernel(共用 BackwardBase 反向),
 * 小kernel 内部再按 isBigC 分流大C标量通道。
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../pool_grad_common/arch35/max_pool_grad_struct.h"
#include "../pool_grad_common/arch35/max_pool_grad_simt.h"
#include "max_pool_v3_grad_nchw_small_kernel.h"
#include "max_pool_v3_grad_nchw_big_kernel.h"

#include "max_pool_v3_grad_nhwc_impl_gather.h"
#include "max_pool_v3_grad_nhwc_big_kernel.h"
#include "max_pool_v3_grad_nhwc_int_impl_gather.h"
#include "max_pool_v3_grad_nhwc_int_big_kernel.h"

using namespace AscendC;
using MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData;
using MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData;
using MaxPoolV3GradNHWCNameSpace::MaxPoolV3GradNHWCBigKernel;
using MaxPoolV3GradNHWCNameSpace::MaxPoolV3GradNHWCSmallKernel;
using MaxPoolV3GradNHWCNameSpace::MaxPoolV3GradNHWCTilingData;
using namespace PoolGradNameSpace;
using MaxPoolV3GradNCHWBigKernelNameSpace::PoolGradNCHWBigKernelV3;
using MaxPoolV3GradNCHWSmallKernelNameSpace::PoolGradNCHWSmallKernelV3;

// REGISTER_TILING_DEFAULT 决定框架为 tiling 数据预留的默认长度。各 KERNEL_MODE
// 通过 GET_TILING_DATA_WITH_STRUCT 读取的结构体不能超过该长度，否则会越界。
static_assert(sizeof(MaxPoolV3GradNHWCTilingData) >= sizeof(MaxPoolGradWithArgmaxNCHWTilingCommonData),
              "default tiling registration must cover NCHW tiling data");
static_assert(sizeof(MaxPoolV3GradNHWCTilingData) >= sizeof(MaxPoolGradWithArgmaxSimtTilingCommonData),
              "default tiling registration must cover SIMT tiling data");

template <uint64_t KERNEL_MODE = TPL_SIMT_KERNEL, uint64_t FORMAT = TPL_NCHW_FORMAT, uint64_t INDICES_DTYPE = TPL_INT32,
          uint64_t IS_CHECK_RANGE = TPL_NO_CHECK_RANGE>
__global__ __aicore__ void max_pool_v3_grad(GM_ADDR orig_input, GM_ADDR orig_output, GM_ADDR grad, GM_ADDR out_grad,
                                            GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MaxPoolV3GradNHWCTilingData);

    // 大小核分发在tiling侧完成: kernelMode按ksize元素数阈值(KSIZE_THRESHOLD=128)
    // 下发TPL_NCHW_BIG_KERNEL或TPL_NCHW_SMALL_KERNEL, kernel入口按KERNEL_MODE静态分发。
    static_assert(KERNEL_MODE == TPL_SIMT_KERNEL || KERNEL_MODE == TPL_NCHW_SMALL_KERNEL ||
                      KERNEL_MODE == TPL_NCHW_BIG_KERNEL || KERNEL_MODE == TPL_NHWC_KERNEL,
                  "MaxPoolV3Grad supports SIMT, NCHW small-kernel, NCHW big-kernel or NHWC fast-path mode");

    if constexpr (KERNEL_MODE == TPL_SIMT_KERNEL) {
        if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
            return;
        }
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxSimtTilingCommonData, tilingData, tiling);

        if constexpr (INDICES_DTYPE == TPL_INT32) {
            MaxPoolGrad::MaxPoolGradSIMT<DTYPE_ORIG_INPUT, int32_t, FORMAT> op(&pipe, &tilingData);
            op.Init(orig_input, orig_output, grad, out_grad, workspace);
            op.Process();
        } else {
            MaxPoolGrad::MaxPoolGradSIMT<DTYPE_ORIG_INPUT, int64_t, FORMAT> op(&pipe, &tilingData);
            op.Init(orig_input, orig_output, grad, out_grad, workspace);
            op.Process();
        }
    } else if constexpr (KERNEL_MODE == TPL_NHWC_KERNEL && FORMAT == TPL_NHWC_FORMAT && INDICES_DTYPE == TPL_INT32 &&
                         std::is_integral_v<DTYPE_ORIG_INPUT>) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolV3GradNHWCTilingData, tilingData, tiling);
        if (tilingData.isBigKernel) {
            MaxPoolV3GradNHWCIntNameSpace::MaxPoolV3GradNHWCBigKernel<DTYPE_ORIG_INPUT, int32_t, IS_CHECK_RANGE> op;
            op.Init(orig_input, orig_output, grad, out_grad, tilingData, pipe);
            op.Process();
        } else {
            MaxPoolV3GradNHWCIntNameSpace::MaxPoolV3GradNHWCSmallKernel<DTYPE_ORIG_INPUT, int32_t, IS_CHECK_RANGE> op;
            op.Init(orig_input, orig_output, grad, out_grad, tilingData, pipe);
            op.Process();
        }
    } else if constexpr (KERNEL_MODE == TPL_NHWC_KERNEL && FORMAT == TPL_NHWC_FORMAT && INDICES_DTYPE == TPL_INT32 &&
                         (std::is_same_v<DTYPE_ORIG_INPUT, float> || std::is_same_v<DTYPE_ORIG_INPUT, half> ||
                          std::is_same_v<DTYPE_ORIG_INPUT, bfloat16_t>)) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolV3GradNHWCTilingData, tilingData, tiling);
        if (tilingData.isBigKernel) {
            MaxPoolV3GradNHWCBigKernel<DTYPE_ORIG_INPUT, int32_t, IS_CHECK_RANGE> op;
            op.Init(orig_input, orig_output, grad, out_grad, tilingData, pipe);
            op.Process();
        } else {
            MaxPoolV3GradNHWCSmallKernel<DTYPE_ORIG_INPUT, int32_t, IS_CHECK_RANGE> op;
            op.Init(orig_input, orig_output, grad, out_grad, tilingData, pipe);
            op.Process();
        }
    } else if constexpr (KERNEL_MODE == TPL_NCHW_SMALL_KERNEL) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNCHWTilingCommonData, tilingData, tiling);

        if constexpr (IS_CHECK_RANGE == TPL_CHECK_RANGE) {
            PoolGradNCHWSmallKernelV3<DTYPE_ORIG_INPUT, true> op(pipe, tilingData);
            op.Init(orig_input, orig_output, grad, out_grad);
            op.Process();
        } else if constexpr (IS_CHECK_RANGE == TPL_NO_CHECK_RANGE) {
            PoolGradNCHWSmallKernelV3<DTYPE_ORIG_INPUT, false> op(pipe, tilingData);
            op.Init(orig_input, orig_output, grad, out_grad);
            op.Process();
        }
    } else if constexpr (KERNEL_MODE == TPL_NCHW_BIG_KERNEL) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxNCHWTilingCommonData, tilingData, tiling);

        if constexpr (IS_CHECK_RANGE == TPL_CHECK_RANGE) {
            PoolGradNCHWBigKernelV3<DTYPE_ORIG_INPUT, true> op(pipe, tilingData);
            op.Init(orig_input, orig_output, grad, out_grad);
            op.Process();
        } else if constexpr (IS_CHECK_RANGE == TPL_NO_CHECK_RANGE) {
            PoolGradNCHWBigKernelV3<DTYPE_ORIG_INPUT, false> op(pipe, tilingData);
            op.Init(orig_input, orig_output, grad, out_grad);
            op.Process();
        }
    }
}
