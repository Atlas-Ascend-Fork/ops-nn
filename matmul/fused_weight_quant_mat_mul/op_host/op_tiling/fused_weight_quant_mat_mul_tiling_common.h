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
 * \file fused_weight_quant_mat_mul_tiling_common.h
 * \brief FusedWeightQuantMatMul tiling 公共定义。
 *        属性布局：transpose_x(0)/transpose_weight(1)/antiquant_group_size(2)/dtype(3)/fused_op_type(4)。
 *        与底座前 4 个属性索引一致；索引 4 为 String（fused_op_type），而底座 checker/GetAttrs
 *        在索引 4 读 int64 的 inner_precise（GetAttrPointer 为裸 reinterpret_cast，无类型校验），
 *        因此底座按索引读属性的代码不可直接复用，须使用本文件的融合属性解析。
 */

#ifndef OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_TILING_COMMON_H
#define OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_TILING_COMMON_H

#include <memory>
#include "matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling.h"

namespace optiling {
namespace fused_weight_quant_mat_mul {

// 融合 epilogue 类型，与 kernel 侧 WeightQuantBatchMatmulV2::Arch35::FusedOpType 枚举值对齐
enum class FusedOpType : uint64_t {
    NONE = 0UL,
    RELU = 1UL,
};

// 本算子属性索引（前 4 个与底座一致；无 inner_precise，融合算子固定高精度模式等价 inner_precise=0）
constexpr uint32_t TRANSPOSE_X_ATTR_INDEX = 0UL;
constexpr uint32_t TRANSPOSE_WEIGHT_ATTR_INDEX = 1UL;
constexpr uint32_t ANTIQUANT_GROUP_SIZE_ATTR_INDEX = 2UL;
constexpr uint32_t FUSED_OP_TYPE_ATTR_INDEX = 4UL;

// 输入索引与底座 WeightQuantBatchMatmulV2 严格对齐
constexpr uint32_t BIAS_INDEX = 6UL;

constexpr size_t MM_SHAPE_LEN_ND = 2UL;
constexpr uint64_t ORI_SHIFT_VALUE = 13UL; // 对齐底座 ORI_SHIFT_VALUE 默认值（未设置 shiftValue 时的计算值）

// 融合版 GetShapeAttrsInfo：复用底座 GetDtype/GetInputs/GetBatchInfo（输入索引 0-6 对齐，直接可用），
// 属性解析按本算子布局执行（读 fused_op_type@4，不读 inner_precise），shiftValue 固定默认值。
ge::graphStatus GetFusedShapeAttrsInfo(gert::TilingContext* context,
                                       std::unique_ptr<WeightQuantBatchMatmulInfo>& matmulInfoPtr, const char*& opName,
                                       uint64_t& fusedOpType, uint32_t& shiftValue);

} // namespace fused_weight_quant_mat_mul
} // namespace optiling

#endif // OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_TILING_COMMON_H
