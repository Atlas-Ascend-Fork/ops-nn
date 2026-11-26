/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTIONBACKWARD_OP_H_
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTIONBACKWARD_OP_H_

#include "opdev/op_executor.h"
namespace l0op {
const int CONV1DINPUTDIM = 3;
const int CONV2DINPUTDIM = 4;
const int CONV3DINPUTDIM = 5;
const int32_t CONV_NON_SPATIAL_DIM_NUM = 2;
const int32_t CONV2D_OUTPUT_PADDING_4 = 4;
const int32_t CONV2D_OUTPUT_PADDING_SPATIAL_GROUP = 2;
const int32_t CONV2D_OUTPUT_PADDING_H_IDX = 0;
const int32_t CONV2D_OUTPUT_PADDING_W_IDX = 1;
const int64_t LIFTED_OUTPUT_PADDING_D = 0;

struct ConvBackpropParams {
    const aclTensor* input;
    const aclTensor* weight;
    const aclTensor* outBackprop;
    const aclIntArray* stride;
    const aclIntArray* padding;
    const aclIntArray* dilation;
    int groups;
};

struct ConvolutionBackwardInputTensor {
    const aclTensor* gradOutput;
    const aclTensor* input;
    const aclTensor* weight;
};

struct ConvolutionBackwardParams {
    const aclIntArray* biasSizes;
    const aclIntArray* stride;
    const aclIntArray* padding;
    const aclIntArray* dilation;
    const bool transposed;
    const aclIntArray* outputPadding;
    const int groups;
    const aclBoolArray* outputMask;
    const int8_t cubeMathType;
};

// transposed场景outputPadding按空间序(spatialOrder)取值。空间维布局(2D为2元素/3D为3元素)按序直取；
// 2D场景4元素为[pt, pb, pl, pr]方向布局(与padding 4元素约定一致)，H取pt=[0]、W取pl=[2]，
// 即spatialOrder*CONV2D_OUTPUT_PADDING_SPATIAL_GROUP取每对首个。checker形状校验与属性转换层共用本取值，保证两层索引一致。
inline int64_t GetTransposedOutputPadding(const aclIntArray* outputPadding, int64_t spatialOrder)
{
    const int64_t padIdx = (outputPadding->Size() == static_cast<uint64_t>(CONV2D_OUTPUT_PADDING_4)) ?
                               spatialOrder * CONV2D_OUTPUT_PADDING_SPATIAL_GROUP :
                               spatialOrder;
    return (*outputPadding)[padIdx];
}

// Check if need to swap D and H dimensions for Conv3D backward
bool NeedSwapDHForConv3DBackward(const ConvolutionBackwardInputTensor& inputTensor,
                                 const ConvolutionBackwardParams& params);

struct tagAdaptParam {
    aclIntArray* adaptStride{nullptr};
    aclIntArray* adaptDilation{nullptr};
    aclIntArray* adaptPad{nullptr};
    bool vecMode{false};
};
using AdaptParam = struct tagAdaptParam;
// Conv2dBackpropInput
// 5HD->FZ with Fp16
const aclTensor* Conv2DBackpropInputFp162Fp16(const aclTensor* input, const aclTensor* weight,
                                              const aclTensor* outBackprop, const aclIntArray* stride,
                                              const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                              aclOpExecutor* executor);
// 1971 5HD->FZ with Fp32
const aclTensor* Conv2DBackpropInputFp322Fp32(const aclTensor* input, const aclTensor* weight,
                                              const aclTensor* outBackprop, const aclIntArray* stride,
                                              const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                              aclOpExecutor* executor);
// 1971 5HD->FZ with Hf32
const aclTensor* Conv2DBackpropInputHf32(const aclTensor* input, const aclTensor* weight, const aclTensor* outBackprop,
                                         const aclIntArray* stride, const aclIntArray* padding,
                                         const aclIntArray* dilation, int groups, aclOpExecutor* executor);
// 1971 5HD->FZ with Bf16
const aclTensor* Conv2DBackpropInputBf162Bf16(const aclTensor* input, const aclTensor* weight,
                                              const aclTensor* outBackprop, const aclIntArray* stride,
                                              const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                              aclOpExecutor* executor);
// 1971 5HD->FZ with input dataType and Hf32Flag.
const aclTensor* Conv2DBackpropInput(const aclTensor* input, const aclTensor* weight, const aclTensor* outBackprop,
                                     const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation,
                                     int groups, aclOpExecutor* executor, bool useHf32Flag, op::DataType dataType);
// Conv2dBackpropFilter
// 5HD->FZ with Fp16
const aclTensor* Conv2DBackpropFilterFp162Fp32(const aclTensor* input, const aclTensor* weight,
                                               const aclTensor* outBackprop, const aclIntArray* stride,
                                               const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                               aclOpExecutor* executor);

// 1971 5HD->FZ with Fp32
const aclTensor* Conv2DBackpropFilterFp322Fp32(const aclTensor* input, const aclTensor* weight,
                                               const aclTensor* outBackprop, const aclIntArray* stride,
                                               const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                               aclOpExecutor* executor);
// 1971 5HD->FZ with Hf32
const aclTensor* Conv2DBackpropFilterHf32(const aclTensor* input, const aclTensor* weight, const aclTensor* outBackprop,
                                          const aclIntArray* stride, const aclIntArray* padding,
                                          const aclIntArray* dilation, int groups, aclOpExecutor* executor);
// 1971 5HD->FZ with Bf16
const aclTensor* Conv2DBackpropFilterBf162Fp32(const aclTensor* input, const aclTensor* weight,
                                               const aclTensor* outBackprop, const aclIntArray* stride,
                                               const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                               aclOpExecutor* executor);

// Conv3dBackpropFilter
// 6HD->FZ_3D with Fp16
const aclTensor* Conv3DBackpropFilterFp162Fp32(const aclTensor* input, const aclTensor* weight,
                                               const aclTensor* outBackprop, const aclIntArray* stride,
                                               const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                               const aclIntArray* outputPadding, aclOpExecutor* executor);

// 1971 6HD->FZ_3D with Fp32
const aclTensor* Conv3DBackpropFilterFp322Fp32(const aclTensor* input, const aclTensor* weight,
                                               const aclTensor* outBackprop, const aclIntArray* stride,
                                               const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                               const aclIntArray* outputPadding, aclOpExecutor* executor);

// 1971 6HD->FZ_3D with Bf16
const aclTensor* Conv3DBackpropFilterBf162Fp32(const aclTensor* input, const aclTensor* weight,
                                               const aclTensor* outBackprop, const aclIntArray* stride,
                                               const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                               const aclIntArray* outputPadding, aclOpExecutor* executor);

// 1971 6HD->FZ_3D with Hf32
const aclTensor* Conv3DBackpropFilterHf32(const aclTensor* input, const aclTensor* weight, const aclTensor* outBackprop,
                                          const aclIntArray* stride, const aclIntArray* padding,
                                          const aclIntArray* dilation, int groups, const aclIntArray* outputPadding,
                                          aclOpExecutor* executor);

// Conv3dBackpropInput
// 6HD->FZ with Fp16
const aclTensor* Conv3DBackpropInputFp162Fp16(const aclTensor* input, const aclTensor* weight,
                                              const aclTensor* outBackprop, const aclIntArray* stride,
                                              const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                              aclOpExecutor* executor, AdaptParam* adptParams);

// Ascend950: adapt Tensor between 3D and 2D. Other platform will return directly.
const aclTensor* AdapterTensor(const aclTensor* tensor, aclOpExecutor* executor, size_t inputDim, size_t targetDim);

// 1971 6HD->FZ with Fp32
const aclTensor* Conv3DBackpropInputFp322Fp32(const aclTensor* input, const aclTensor* weight,
                                              const aclTensor* outBackprop, const aclIntArray* stride,
                                              const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                              aclOpExecutor* executor, AdaptParam* adptParams);
// 1971 6HD->FZ with Hf32
const aclTensor* Conv3DBackpropInputHf32(const aclTensor* input, const aclTensor* weight, const aclTensor* outBackprop,
                                         const aclIntArray* stride, const aclIntArray* padding,
                                         const aclIntArray* dilation, int groups, aclOpExecutor* executor,
                                         AdaptParam* adptParams);
// 1971 6HD->FZ with Bf16
const aclTensor* Conv3DBackpropInputBf162Bf16(const aclTensor* input, const aclTensor* weight,
                                              const aclTensor* outBackprop, const aclIntArray* stride,
                                              const aclIntArray* padding, const aclIntArray* dilation, int groups,
                                              aclOpExecutor* executor, AdaptParam* adptParams);
// Conv3dBackpropFilter
// 1971 6HD->FZ_3D with Bf16/FP16
// 1982 5HD->FZ_3D with Bf16/FP16

const aclTensor* Conv3DBackpropFilter(ConvolutionBackwardInputTensor& inputTensor, ConvolutionBackwardParams& params,
                                      aclOpExecutor* executor, bool use_hf32);
// Conv3dBackpropInput
// 1971 6HD->FZ with Bf16/FP16
// 91 5HD->FZ with Bf16/FP16
const aclTensor* Conv3DBackpropInput(ConvolutionBackwardInputTensor& inputTensor, ConvolutionBackwardParams& params,
                                     aclOpExecutor* executor, bool use_hf32, AdaptParam* adptParams);
bool IsConv3DBackpropInputV2(const ConvBackpropParams& params);

bool IsConv3DBackpropFilterV2(const ConvBackpropParams& params);

bool IsInputTransdataWhiteListCase(const ConvBackpropParams& params);

bool IsConv2DBackpropInputToCastCase(const ConvBackpropParams& params);

bool IsConv2DBackpropInputTo3DCase(const ConvBackpropParams& params);

bool IsConv2DBpFilterTo3Dcase(const ConvBackpropParams& params);

bool CheckN2HEnable(const aclTensor* weight, const aclTensor*& output, aclIntArray* stride5, aclIntArray* dilation5,
                    aclIntArray* pad6, int groups);

bool CheckWeightPreTransposeEnable(const aclTensor* weight, const aclTensor* input, const aclIntArray* stride5,
                                   int groups);

aclnnStatus N2HOptimize(const aclTensor*& weight, const aclTensor*& outBackprop, AdaptParam* adptParams,
                        aclOpExecutor* executor);

void GetConv3DBackpropAdapterParam(const aclTensor* input, const aclIntArray* stride, const aclIntArray* padding,
                                   const aclIntArray* dilation, aclOpExecutor* executor, AdaptParam* params);
} // namespace l0op

#endif // OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTIONBACKWARD_OP_H_
