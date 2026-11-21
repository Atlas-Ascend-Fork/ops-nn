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
 * \file max_pool_grad_struct.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_STRUCT_H
#define MAX_POOL_GRAD_STRUCT_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "ascendc/host_api/tiling/template_argument.h"

namespace PoolGradNameSpace {

#define TPL_NCHW_SMALL_KERNEL 1
#define TPL_NCHW_BIG_KERNEL 2
#define TPL_SIMT_KERNEL 3
#define TPL_NHWC_KERNEL 4

#define TPL_NCHW_FORMAT 0
#define TPL_NHWC_FORMAT 1
#define TPL_INT64 0
#define TPL_INT32 1
#define TPL_NO_CHECK_RANGE 0
#define TPL_CHECK_RANGE 1

ASCENDC_TPL_ARGS_DECL(MaxPoolGrad,
                      ASCENDC_TPL_UINT_DECL(kernelMode, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, TPL_NCHW_SMALL_KERNEL,
                                            TPL_NCHW_BIG_KERNEL, TPL_SIMT_KERNEL, TPL_NHWC_KERNEL),
                      ASCENDC_TPL_UINT_DECL(format, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, TPL_NCHW_FORMAT,
                                            TPL_NHWC_FORMAT),
                      ASCENDC_TPL_UINT_DECL(indicesDtype, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_INT64, TPL_INT32),
                      ASCENDC_TPL_UINT_DECL(isCheckRange, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, TPL_NO_CHECK_RANGE,
                                            TPL_CHECK_RANGE));

ASCENDC_TPL_SEL(
    // SIMT kernel - format can be NCHW or NHWC
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(kernelMode, ASCENDC_TPL_UI_LIST, TPL_SIMT_KERNEL),
                         ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_NCHW_FORMAT, TPL_NHWC_FORMAT),
                         ASCENDC_TPL_UINT_SEL(indicesDtype, ASCENDC_TPL_UI_LIST, TPL_INT64, TPL_INT32),
                         ASCENDC_TPL_UINT_SEL(isCheckRange, ASCENDC_TPL_UI_LIST, TPL_NO_CHECK_RANGE),
                         ASCENDC_TPL_TILING_STRUCT_SEL(MaxPoolGradWithArgmaxSimtTilingCommonData)),

    // SIMD kernel - format must be NCHW
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(kernelMode, ASCENDC_TPL_UI_LIST, TPL_NCHW_BIG_KERNEL,
                                              TPL_NCHW_SMALL_KERNEL),
                         ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_NCHW_FORMAT),
                         ASCENDC_TPL_UINT_SEL(indicesDtype, ASCENDC_TPL_UI_LIST, TPL_INT64, TPL_INT32),
                         ASCENDC_TPL_UINT_SEL(isCheckRange, ASCENDC_TPL_UI_LIST, TPL_NO_CHECK_RANGE, TPL_CHECK_RANGE),
                         ASCENDC_TPL_TILING_STRUCT_SEL(MaxPoolGradWithArgmaxNCHWTilingCommonData)),

    // NHWC SIMD kernel (MaxPoolV3Grad) - format must be NHWC, argmax 固定 int32
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                         ASCENDC_TPL_UINT_SEL(kernelMode, ASCENDC_TPL_UI_LIST, TPL_NHWC_KERNEL),
                         ASCENDC_TPL_UINT_SEL(format, ASCENDC_TPL_UI_LIST, TPL_NHWC_FORMAT),
                         ASCENDC_TPL_UINT_SEL(indicesDtype, ASCENDC_TPL_UI_LIST, TPL_INT32),
                         ASCENDC_TPL_UINT_SEL(isCheckRange, ASCENDC_TPL_UI_LIST, TPL_NO_CHECK_RANGE, TPL_CHECK_RANGE),
                         ASCENDC_TPL_TILING_STRUCT_SEL(MaxPoolV3GradNHWCTilingData)));

} // namespace PoolGradNameSpace

namespace MaxPoolGradNCHWBackwardBaseNameSpace {
// NCHW 反向 help buffer 的单一来源，host tiling 与 kernel InitBuffer 共用。
constexpr int32_t HELP_BUFFER = 4096;
} // namespace MaxPoolGradNCHWBackwardBaseNameSpace

// MaxPoolV3Grad NHWC 快速通道 tiling data 结构
// (参考 max_pool3d_grad 的 Pool3DGradNDHWCTilingData, 去除 D 维; 需与本文件 TPL 声明同处,
//  生成代码会对 TPL_SEL 结构体取 sizeof)
namespace MaxPoolV3GradNHWCNameSpace {

// 空间维 tiling 数据: 参考 Pool3DGradNCDHWTilingData 去 D 维/去 dilation(MaxPoolV3 无 dilation 属性)
struct MaxPoolV3GradSpatialTilingData {
    int64_t hArgmax = 0; // grad(池化输出) H
    int64_t wArgmax = 0; // grad(池化输出) W
    int64_t hOutput = 0; // orig_input H
    int64_t wOutput = 0; // orig_input W
    int64_t hKernel = 0;
    int64_t wKernel = 0;
    int64_t hStride = 0;
    int64_t wStride = 0;
    int64_t padH = 0;
    int64_t padW = 0;
    int64_t padHBack = 0;
    int64_t padWBack = 0;

    int64_t highAxisInner = 0; // N 轴切分
    int64_t highAxisTail = 0;
    int64_t highAxisOuter = 0;
    int64_t hOutputInner = 0; // 输入 H 轴切分
    int64_t hOutputTail = 0;
    int64_t hOutputOuter = 0;
    int64_t wOutputInner = 0; // 输入 W 轴切分
    int64_t wOutputTail = 0;
    int64_t wOutputOuter = 0;

    int64_t normalCoreProcessNum = 0;
    int64_t tailCoreProcessNum = 0;
    int64_t usedCoreNum = 0;

    int64_t inputBufferSize = 0;  // 前向输入窗 tile 缓冲(字节)
    int64_t outputBufferSize = 0; // 反向输出 tile 缓冲(字节, fp32)
    int64_t gradBufferSize = 0;   // 反向 grad tile 缓冲(字节)
    int64_t argmaxBufferSize = 0; // argmax tile 缓冲(字节, int32)

    int64_t hProBatchSize = 0;
    int64_t wProBatchSize = 0;

    int64_t isBigKernel = 0;
};

// NHWC tiling 数据: 参考 Pool3DGradNDHWCTilingData, C 轴为最内层连续轴
struct MaxPoolV3GradNHWCTilingData {
    MaxPoolV3GradSpatialTilingData base;

    int64_t cDim = 0; // 输入 C 完整维度
    int64_t cOutputInner = 0;
    int64_t cOutputTail = 0;
    int64_t cOutputOuter = 0;
    int64_t isBigKernel = 0;
};

} // namespace MaxPoolV3GradNHWCNameSpace
#endif // MAX_POOL_GRAD_STRUCT_H
