/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// gn_training_update_package/op_graph/gn_training_update_graph_infer.cpp
// =============================================================================
//
// ROLE: Graph-level data type inference for the GnTrainingUpdate operator.
//   When the Graph Engine (GE) constructs a computational graph, it needs to
//   know the output data types of each node before execution. This file
//   registers a type inference function that tells GE the output dtypes.
//
//   Semantics (design/InferShapeDtype.md "InferDataType" table,
//   spec/spec.yaml outputs.dtype_rule / dtype_policy):
//     y.dtype              = x.dtype  (float16 or float32, follows input 0;
//                                      dtype_policy.promotion: same_as_first_input)
//     batch_mean.dtype     = float32  (copied from sum, whose dtype_set is [float32])
//     batch_variance.dtype = float32  (same as batch_mean)
//   No cross-input promotion exists; the statistics/affine inputs are always
//   float32, so copying sum's dtype yields exactly the declared combinations.
//
//   The INFER_SHAPE equivalent is in op_host/gn_training_update_infershape.cpp.
//
//   No rt1.0 (V1) infer-shape is registered here on purpose: V1 is the legacy
//   ge::InferShapeFuncRegister mechanism. The graph route is served by the V2
//   IMPL_OP_INFERSHAPE callbacks in op_host/gn_training_update_infershape.cpp;
//   the canndev built-in legacy V1 for this op type is isolated by
//   TILING_REG_FLAG in newer packages.
//
// CONTENTS:
//   - InferDataTypeForGnTrainingUpdate() — the type inference function
//   - IMPL_OP(...).InferDataType(...) — registration
//
// REGISTRATION NAME: the graph-IR prototype is REG_OP(GNTrainingUpdate)
//   (frozen, verbatim from the official IR), while the OpDef / ACLNN op type
//   is GnTrainingUpdate (read-only). The impl registry is keyed by exact op
//   type string, so the same function is registered under BOTH names:
//   GnTrainingUpdate serves the ACLNN/two-stage path, GNTrainingUpdate serves
//   graph nodes whose type is the canonical REG_OP name.
//
//   InferFormat / InferShapeRange callbacks are not registered: all tensors
//   are fixed FORMAT_ND (OpDef UnknownShapeFormat ND), so the framework
//   default inference is sufficient and the operator interface does not
//   require them.
//
// =============================================================================

#include "register/op_impl_registry.h" // IMPL_OP macro for operator registration
#include "op_common/log/log.h"         // OP_LOGE
#include <vector>

using namespace ge;

namespace ops {

// ---------------------------------------------------------------------------
// Shared input-validation helpers (design/InferShapeDtype.md "broadcast 对齐规则",
// "dtype 组合" and "边界处理"). Violations on statically-known facts are
// rejected (GRAPH_FAILED) so illegal graphs fail at compile time instead of
// executing. Dynamic dims (any dim < 0, including the unknown-rank {-2}
// marker) are wildcards: a comparison involving an unknown dim is skipped.
// ---------------------------------------------------------------------------
namespace {

// dtype contract: x in {float16, float32}; sum/square_sum and every present
// optional input (scale/offset/mean/variance) must be float32. An absent
// optional input reports DT_UNDEFINED and is skipped.
ge::graphStatus CheckInputDtypes(const std::vector<ge::DataType>& dtypes, const char* op_name)
{
    const ge::DataType x_dtype = dtypes[0];
    if (x_dtype != ge::DT_FLOAT16 && x_dtype != ge::DT_FLOAT) {
        OP_LOGE(op_name, "x dtype must be float16 or float32.");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 1; i < dtypes.size(); ++i) {
        const ge::DataType dt = dtypes[i];
        if (dt == ge::DT_UNDEFINED) {
            continue; // absent optional input
        }
        if (dt != ge::DT_FLOAT) {
            OP_LOGE(op_name, "input %zu dtype must be float32.", i);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

// ---------------------------------------------------------------------------
// InferDataTypeForGnTrainingUpdate(context) — data type inference callback
//
// Output dtype derivation:
//   output 0 (y)              <- input 0 (x)   dtype (fp16 or fp32)
//   output 1 (batch_mean)     <- input 1 (sum) dtype (always fp32)
//   output 2 (batch_variance) <- input 1 (sum) dtype (always fp32)
//
// Parameters:
//   context — InferDataTypeContext that provides access to input dtypes
//             and allows setting output dtypes
//
// Returns:
//   ge::graphStatus — ge::GRAPH_SUCCESS on success
// ---------------------------------------------------------------------------
static ge::graphStatus InferDataTypeForGnTrainingUpdate(gert::InferDataTypeContext* context)
{
    // dtype contract validation (design/InferShapeDtype.md "dtype 组合").
    // Required inputs by packed index; OPTIONAL_INPUT ports by IR-prototype
    // index via GetOptionalInputDataType (absent -> DT_UNDEFINED, skipped).
    std::vector<ge::DataType> input_dtypes;
    for (size_t i = 0; i <= 2; ++i) { // x, sum, square_sum
        input_dtypes.push_back(context->GetInputDataType(i));
    }
    for (size_t ir = 3; ir <= 6; ++ir) { // scale, offset, mean, variance
        input_dtypes.push_back(context->GetOptionalInputDataType(ir));
    }
    if (CheckInputDtypes(input_dtypes, "GnTrainingUpdate") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // y.dtype = x.dtype (same_as_first_input)
    const ge::DataType x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x_dtype);

    // batch_mean.dtype = batch_variance.dtype = sum.dtype (float32)
    const ge::DataType sum_dtype = context->GetInputDataType(1);
    context->SetOutputDataType(1, sum_dtype);
    context->SetOutputDataType(2, sum_dtype);
    return ge::GRAPH_SUCCESS;
}

// Register the type inference function. Dual registration: GnTrainingUpdate
// (OpDef/ACLNN op type) and GNTrainingUpdate (canonical REG_OP graph type) —
// see the file header for the rationale.
IMPL_OP(GnTrainingUpdate).InferDataType(InferDataTypeForGnTrainingUpdate);
IMPL_OP(GNTrainingUpdate).InferDataType(InferDataTypeForGnTrainingUpdate);
} // namespace ops
