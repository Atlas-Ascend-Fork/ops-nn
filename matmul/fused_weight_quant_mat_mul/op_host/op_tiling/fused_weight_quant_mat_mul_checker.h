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
 * \file fused_weight_quant_mat_mul_checker.h
 * \brief FusedWeightQuantMatMul（supportMmadS8S4 平台）参数合法性校验。
 *        dtype/shape 校验逻辑对齐底座 WeightQuantBatchMatmulV2Checker4MmadS8S4；
 *        属性读取按本算子布局适配（索引 4 为 String fused_op_type，底座在索引 4 读 int64
 *        inner_precise，GetAttrPointer 无类型校验不可复用）；叠加一期收紧：
 *        y ∈ {FP16, INT8}、antiquant 仅 PerChannel、fused_op_type ∈ {"relu"}。
 */

#ifndef OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_CHECKER_H
#define OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_CHECKER_H

#include "fused_weight_quant_mat_mul_tiling_common.h"

namespace optiling {
namespace fused_weight_quant_mat_mul {

class FusedWeightQuantMatMulChecker {
public:
    explicit FusedWeightQuantMatMulChecker(gert::TilingContext* context) : context_(context) {}

    ~FusedWeightQuantMatMulChecker() = default;

    ge::graphStatus Check();

private:
    ge::graphStatus CheckContext();
    bool CheckDtype();
    bool CheckAttr();
    bool CheckShape();
    bool CheckInputShape(const gert::StorageShape* xShape, const gert::StorageShape* weightShape);
    bool CheckAntiQuantShape(const gert::StorageShape* antiQuantScaleShape,
                             const gert::StorageShape* antiQuantOffsetShape);
    bool CheckBatchInfo(const gert::Shape& xShape, const gert::Shape& weightShape) const;

    gert::TilingContext* context_ = nullptr;
    WeightQuantBatchMatmulInfo inputParams_;
};
} // namespace fused_weight_quant_mat_mul
} // namespace optiling

#endif // OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_CHECKER_H
