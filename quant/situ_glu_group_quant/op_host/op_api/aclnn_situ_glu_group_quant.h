/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_SITU_GLU_GROUP_QUANT_H_
#define OP_API_INC_SITU_GLU_GROUP_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSituGluGroupQuant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能（移植自 cann-recipes-infer SituGluGroupQuant）：
 * 稀疏SiTU门控激活与MXFP8（e4m3fn，group_size=32）尾轴动态量化的融合算子，
 * 面向MoE分组（grouped）场景。输入x最后一维为拼接的gate/up两半（2*EPR）；
 * 激活结果驻留片上（不回写GM），随后按MX算法量化为y(fp8_e4m3fn)+y_scale(e8m0)。
 * 与两个小算子串联 npu_situ_and_mul_sparse -> npu_dynamic_mx_quant(dst=e4m3fn)
 * 逐位一致（有效行范围内）。
 *
 * @param [in] x: npu device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16，
 * shape支持1维及以上，最后一维必须为正偶数（EPR=x.shape[-1]/2，EPR须为32的倍数且在
 * [128, 3072]内，由tiling防呆校验），
 * 空Tensor（零长维度）退化为0行计算，不拒收，数据格式支持ND/NCL/NCHW/NCDHW。
 * @param [in] weightOptional: npu device侧的aclTensor，可选输入，数据类型FLOAT32，
 * 每个元素为对应token（行）的缩放因子，元素个数 = N（与x展开后的总行数一致，
 * 按全局行号索引）。存在时激活结果按 act = T(fp32(act) * w[row]) 缩放后
 * 再做MX量化；不传（nullptr）则不做缩放。
 * @param [in] groupIndexOptional: npu device侧的aclTensor，可选输入，数据类型INT64，
 * 不传（nullptr）时为全行模式（所有行参与计算）；传入时第i个元素为专家i分到的
 * token行数，kernel逐元素累加得到有效行数
 * validRows=min(sum(group_index), totalRows)，仅处理前validRows行，
 * 其余padding行不被写出。
 * @param [in] beta: host侧的double类型，SiTU激活gate支路的beta参数，必须大于0。默认1.0。
 * @param [in] linearBeta: host侧的double类型，SiTU激活up支路软限幅参数：
 * up = linearBeta*tanh(up/linearBeta)。<=0时禁用软限幅（up直通参与相乘），
 * >0时启用（up输出幅度被钳位在±linearBeta内）。默认0.0（禁用）。
 * @param [in] activateLeft: host侧的bool类型，gate/up方向选择：
 * true时gate取x前半部分、up取后半部分（recipes原始方向）；
 * false（默认）时gate取x后半部分、up取前半部分（situ normal默认方向）。
 * @param [in] axis: host侧的int64_t类型，量化轴。当前仅支持-1（尾轴），默认-1。
 * @param [in] quantMode: host侧的int64_t类型，量化模式。当前仅支持0（MX量化），默认0。
 * @param [in] dstType: host侧的int64_t类型，目标量化数据类型：
 * 35(FLOAT8_E5M2)、36(FLOAT8_E4M3FN)、40(FLOAT4_E2M1)、41(FLOAT4_E1M2)。
 * 当前仅支持36（FLOAT8_E4M3FN），默认36。
 * @param [in] dstTypeMax: host侧的double类型，目标数据类型数值上限，取值>=0，默认0。
 * @param [in] roundModeOptional: host侧的char*类型，舍入模式：rint/round/floor。
 * 当前仅支持rint，默认rint。
 * @param [in] groupListType: host侧的int64_t类型，group_index的语义模式：
 * 0=cumsum模式（元素为组结束偏移，预留未实现，会被拒绝），
 * 1=count模式（元素为每组的行数，当前语义）。默认0，当前必须显式传1。
 * @param [out] yOut: npu device侧的aclTensor，量化后的输出，数据类型FLOAT8_E4M3FN，
 * shape为x.shape[:-1]+[EPR]。
 * @param [out] yScaleOut: npu device侧的aclTensor，MX量化的scale，
 * 数据类型FLOAT8_E8M0，shape为x.shape[:-1]+[ceil(ceil(EPR/32)/2), 2]。
 * @param [out] workspaceSize: host侧的uint64_t类型，计算workspace大小后回传给用户。
 * @param [out] executor: aclOpExecutor类型，返回给用户的执行器。
 * @retval aclnnStatus 返回0表示成功，其他值表示失败。
 *
 * @see aclnnSituGluGroupQuant
 */
ACLNN_API aclnnStatus aclnnSituGluGroupQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* weightOptional,
                                                             const aclTensor* groupIndexOptional, double beta,
                                                             double linearBeta, bool activateLeft, int64_t axis,
                                                             int64_t quantMode, int64_t dstType, double dstTypeMax,
                                                             const char* roundModeOptional, int64_t groupListType,
                                                             const aclTensor* yOut, const aclTensor* yScaleOut,
                                                             uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnSituGluGroupQuant的第二段接口，执行器下发。
 * @domain aclnn_ops_infer
 *
 * @param [in] workspace: host侧的void*类型，device侧内存地址，用于存放执行期间
 * 算子内部使用的临时空间。
 * @param [in] workspaceSize: host侧的uint64_t类型，第一段接口计算出的workspace大小。
 * @param [in] executor: aclOpExecutor类型，第一段接口返回的执行器。
 * @param [in] stream: aclrtStream类型，指定执行任务的stream。
 * @retval aclnnStatus 返回0表示成功，其他值表示失败。
 */
ACLNN_API aclnnStatus aclnnSituGluGroupQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif // OP_API_INC_SITU_GLU_GROUP_QUANT_H_
