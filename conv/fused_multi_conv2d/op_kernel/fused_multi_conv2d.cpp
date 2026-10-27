/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define K_MAX_SHAPE_DIM 0

#include "arch35/fused_multi_conv2d_impl.h"

using namespace AscendC;

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

// 解析列表地址；list 为 TensorList 描述，index 为张量索引，表头偏移单位为字节。
__aicore__ inline GM_ADDR TensorListAt(GM_ADDR list, uint32_t index)
{
    if (list == nullptr) {
        return nullptr;
    }
    __gm__ uint64_t* descriptor = reinterpret_cast<__gm__ uint64_t*>(list);
    __gm__ uint64_t* addresses = descriptor + (*descriptor >> 3);
    return reinterpret_cast<GM_ADDR>(addresses[index]);
}

// 解码主机参数；tilingData 包含几何、缓存地址、两层定标及偏置/激活开关。
template <typename TilingData>
__aicore__ inline FusedMultiConv2d::KernelArgs BuildKernelArguments(const TilingData& tilingData)
{
    FusedMultiConv2d::KernelArgs kernelArguments;

#define FC2D_GET_FIELD(member, name) kernelArguments.geometry.member = (int)tilingData.name;
    FC2D_FLAT_FIELDS(FC2D_GET_FIELD)
#undef FC2D_GET_FIELD

    kernelArguments.fixedShift1 = (uint8_t)tilingData.fixedShift1;
    kernelArguments.fixedShift2 = (uint8_t)tilingData.fixedShift2;
    kernelArguments.hasBias1 = tilingData.hasBias1 != 0;
    kernelArguments.hasBias2 = tilingData.hasBias2 != 0;
    kernelArguments.relu1 = tilingData.relu1 != 0;
    kernelArguments.relu2 = tilingData.relu2 != 0;

    return kernelArguments;
}

// 执行融合卷积；x/y 为输入/输出，filters/bias/scale 为两层权重/偏置/定标列表，
// relu_weight 为预留输入，workspace 为系统工作区，tiling 为主机参数。
extern "C" __global__ __aicore__ void fused_multi_conv2d(GM_ADDR x, GM_ADDR filters, GM_ADDR bias, GM_ADDR scale,
                                                         GM_ADDR relu_weight, GM_ADDR y, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    (void)relu_weight;
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);

    GM_ADDR filter1 = TensorListAt(filters, 0);
    GM_ADDR filter2 = TensorListAt(filters, 1);

    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR bias1 = (tilingData.hasBias1 != 0) ? TensorListAt(bias, 0) : nullptr;
    GM_ADDR bias2 = (tilingData.hasBias2 != 0) ? TensorListAt(bias, 1) : nullptr;

    GM_ADDR dequantScale2 = nullptr;
    GM_ADDR quantScale1 = nullptr;
    GM_ADDR quantScale2 = nullptr;
    if constexpr (FusedMultiConv2d::DefaultCfg::NEEDS_Q1_TABLE) {
        quantScale1 = TensorListAt(scale, 0);
        if constexpr (FusedMultiConv2d::DefaultCfg::NEEDS_DEQ_TABLE) {
            dequantScale2 = TensorListAt(scale, 1);
        } else {
            quantScale2 = TensorListAt(scale, 1);
        }
    }

    const auto kernelArguments = BuildKernelArguments(tilingData);

    FusedMultiConv2d::FusedMultiConv2dKernel<FusedMultiConv2d::DefaultCfg>(
        x, filter1, bias1, filter2, bias2, dequantScale2, quantScale1, quantScale2, y, kernelArguments);
}
