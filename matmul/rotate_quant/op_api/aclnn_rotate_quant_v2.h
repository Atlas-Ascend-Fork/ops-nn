/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_ROTATE_QUANT_V2_H
#define ACLNN_ROTATE_QUANT_V2_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRotateQuantV2的第一段接口，计算并获取workspace大小。
 * @domain aclnn_ops_infer
 *
 * 对输入x进行旋转变换、可选的clamp操作及MX量化，支持Ascend 950PR&950DT系列产品。
 * groupList为nullptr时沿用非分组语义；非空时按M轴分组，各组使用对应的rotation。
 * MX量化仍沿最后一维每32个元素计算一个scale，alpha、axis、roundMode、scaleAlg及dstTypeMax在各组间共用。
 *
 * @param [in] x: 待旋转量化的输入张量，必选参数。
 * Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16，支持非连续的Tensor，数据格式支持ND。
 * groupList为nullptr时，shape为(*, N)，维度范围为[1, 7]，支持空Tensor。
 * groupList非空时，shape为(M, N)，仅支持二维；M可为0，N必须大于0，且必须是K的整数倍。
 * @param [in] rotation: 旋转矩阵，必选参数，数据类型与x保持一致。
 * Device侧的aclTensor，支持非连续的Tensor，数据格式支持ND。
 * groupList为nullptr时，shape为(K, K)或(N/K, K, K)，支持空Tensor。
 * groupList非空时，shape为(E, K, K)或(E, N/K, K, K)，不支持空Tensor，E须与groupList的长度一致。
 * 最后两维长度相同，K支持32、64、128、256，x最后一维长度N必须是K的整数倍。
 * @param [in] alpha: clamp需要限制的范围的缩放系数，可选参数。
 * Device侧的aclTensor，shape为(1,)，数据类型为BFLOAT16，数据格式支持ND。
 * 当值在(0.0, 1.0)内时执行clamp；传入nullptr或值不在该区间时，不做clamp处理。
 * @param [in] groupList: 指定M轴分组情况的分组列表，可选参数，分组量化特性仅支持Ascend950。
 * Device侧的aclTensor，shape为(E,)，维度为1，数据类型为INT64，数据格式支持ND，支持非连续的Tensor。
 * 传入nullptr时不进行M轴分组；非空时不支持空Tensor，分组数须满足1 <= E <= 1024。
 * 分组从M轴第0行连续划分，支持行数为0的空组，空组仍占据对应rotation的组下标。
 * 允许只覆盖M的连续前缀，未覆盖行的yOut和scaleOut（包括这些行的scale padding）保持调用前的内容。
 * 全部组为空时不更新输出；M为0时，各组行数均须为0。
 * @param [in] groupListType: 表示groupList的分组表示方式，仅支持0和1。
 * 0表示各组行数的累计和（cumsum）：元素须非负、单调不减，最后一项须小于等于M。
 * 1表示各组行数（count）：元素须非负，所有元素之和须小于等于M。
 * groupList为nullptr时仍须传入0或1，此时不进行M轴分组。
 * @param [in] axis: 表示量化发生的轴（reduce轴）。
 * 仅支持-1或D-1，D为x的维数；分组场景仅支持-1或1。
 * @param [in] roundMode: 表示数据转换的模式。
 * 支持"rint"、"round"、"floor"，传入nullptr时采用"rint"。
 * 当yOut的数据类型为FLOAT8_E4M3FN或FLOAT8_E5M2时，仅支持"rint"。
 * @param [in] scaleAlg: 表示scale的计算算法，默认值为0。
 * 当yOut的数据类型为FLOAT4_E2M1时，仅支持0和2；为FLOAT8_E4M3FN或FLOAT8_E5M2时，仅支持0和1。
 * @param [in] dstTypeMax: 表示量化目标类型的最大值。
 * 当scaleAlg为2时，取值必须在[6.0, 12.0]范围内；其余场景仅支持0.0。
 * @param [in] trans: 表示输出y是否转置，目前仅支持false。
 * @param [out] yOut: 量化后的输出张量，需由调用方预先分配。
 * Device侧的aclTensor，数据类型支持FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2，数据格式支持ND。
 * shape与x相同，分组场景为(M, N)；其数据类型决定量化输出类型，无需单独传入y_dtype。
 * @param [out] scaleOut: 动态量化计算出的缩放系数，需由调用方预先分配。
 * Device侧的aclTensor，数据类型为FLOAT8_E8M0，数据格式支持ND。
 * shape为(*, CeilDiv(N, 64), 2)，维度范围为[2, 8]；分组场景为(M, CeilDiv(N, 64), 2)。
 * @param [out] workspaceSize: 返回用户需要在NPU Device侧申请的workspace大小，单位为字节。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnRotateQuantV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* rotation, const aclTensor* alpha, const aclTensor* groupList,
    int64_t groupListType, int64_t axis, char* roundMode, int64_t scaleAlg, double dstTypeMax, bool trans,
    aclTensor* yOut, aclTensor* scaleOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnRotateQuantV2的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在NPU Device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在NPU Device侧申请的workspace大小，由aclnnRotateQuantV2GetWorkspaceSize获取。
 * @param [in] executor: 第一段接口返回的op执行器，包含算子计算流程，调用本接口后不再可用。
 * @param [in] stream: 指定执行任务的acl Stream流。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnRotateQuantV2(void* workspace, uint64_t workspaceSize,
                                                                      aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_ROTATE_QUANT_V2_H
