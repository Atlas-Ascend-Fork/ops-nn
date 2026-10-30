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
 * \file fused_weight_quant_mat_mul_iterbatch_tiling.h
 * \brief FusedWeightQuantMatMul IterBatch tiling：继承底座 WeightQuantBatchMatmulV2IterbatchTiling
 *        （其已继承 ASW tiling），batch 融合判断与切分算法零拷贝复用；仅覆写属性解析与
 *        GetTilingKey（追加 FUSED_OPTYPE 编码位）。
 */

#ifndef OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_ITERBATCH_TILING_H
#define OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_ITERBATCH_TILING_H

#include "../fused_weight_quant_mat_mul_tiling_common.h"
#include "matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_iterbatch_tiling.h"

namespace optiling {
namespace fused_weight_quant_mat_mul {

class FusedWeightQuantMatMulIterbatchTiling
    : public weight_quant_batch_matmul_v2::WeightQuantBatchMatmulV2IterbatchTiling {
public:
    explicit FusedWeightQuantMatMulIterbatchTiling(gert::TilingContext* context)
        : weight_quant_batch_matmul_v2::WeightQuantBatchMatmulV2IterbatchTiling(context)
    {}
    ~FusedWeightQuantMatMulIterbatchTiling() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    uint64_t GetTilingKey() const override;

    uint64_t fusedOpType_ = static_cast<uint64_t>(FusedOpType::RELU);
};
} // namespace fused_weight_quant_mat_mul
} // namespace optiling

#endif // OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_ITERBATCH_TILING_H
