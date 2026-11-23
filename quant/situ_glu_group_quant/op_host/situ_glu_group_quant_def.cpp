/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file situ_glu_group_quant_def.cpp
 * \brief SituGluGroupQuant operator definition, ported from cann-recipes-infer.
 *
 * Fusion of situ_and_mul_sparse (sparse SiTU gated activation) and the MXFP8
 * (e4m3fn, blocksize=32) tail-axis dynamic quantization of dynamic_mx_quant.
 * Input x holds the gate/up halves concatenated on the last axis (2*EPR);
 * the fused op computes the sparse SiTU activation act[.., EPR] in registers
 * (never spilled to GM) and quantizes it to y(fp8_e4m3fn) + y_scale(e8m0)
 * with group_size=32 along the last axis. Bit-identical to the two small ops
 * chained (npu_situ_and_mul_sparse -> npu_dynamic_mx_quant, dst=e4m3fn).
 */

#include <register/op_def_registry.h>

namespace ops {

class SituGluGroupQuant : public OpDef {
public:
    explicit SituGluGroupQuant(const char* name) : OpDef(name)
    {
        // dtype 组合（2种）：
        // x(FLOAT16/BFLOAT16) → y(FLOAT8_E4M3FN)，y_scale(FLOAT8_E8M0)，group_index(INT64)
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // Optional input (owner decision): per-token FP32 scalar scale, length N
        // (= totalRows, global row index). Present -> act = T(fp32(act) * w[row])
        // applied between activation and quantization; absent -> no scaling.
        this->Input("weight")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // Optional input (owner decision): absent -> whole-tensor mode (all rows
        // processed, validRows = totalRows); present -> recipes grouped semantics.
        this->Input("group_index")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("beta").AttrType(OPTIONAL).Float(1.0f);
        // linear_beta (owner decision, TPL dimension): up-branch soft-clamp
        // threshold (up = linear_beta * tanh(up/linear_beta)). <= 0 disables
        // the soft-clamp (up passthrough); > 0 enables it. Default 0.0
        // (disabled) — callers must opt in explicitly.
        this->Attr("linear_beta").AttrType(OPTIONAL).Float(0.0f);
        // activate_left (owner decision, mirrors situ_mx_quant normal path):
        //   true  -> gate = first half, up = second half (recipes original caliber)
        //   false -> gate = second half, up = first half (situ normal default, new)
        this->Attr("activate_left").AttrType(OPTIONAL).Bool(false);
        // Extension attrs (situ_mx_quant caliber): validated-only for now —
        // each currently supports exactly one value, so none of them reaches
        // the tiling data or the kernel (single-variant code path).
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("dst_type").AttrType(OPTIONAL).Int(ge::DT_FLOAT8_E4M3FN);
        this->Attr("dst_type_max").AttrType(OPTIONAL).Float(0.0f);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        // group_list_type: 0=cumsum mode (RESERVED, rejected by tiling until
        // implemented), 1=count mode (current semantics). Default 0 points at
        // the future mode; callers must explicitly pass 1 today.
        this->Attr("group_list_type").AttrType(OPTIONAL).Int(0);

        // Ascend 950 (arch35) configuration using Regbase; kernel entry file
        // follows the {ops}.cpp convention (KERNEL_SRC default, wiki caliber).
        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true).DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend950", regbaseCfg);
    }
};

OP_ADD(SituGluGroupQuant);

} // namespace ops
