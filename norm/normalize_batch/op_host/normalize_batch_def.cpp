/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Operator definition (OpDef) for NormalizeBatch (arch35: ascend950/ascend960dt/ascend960pr/ascend350).
//   公开原型：
//   - Input  input_x  : REQUIRED, {DT_FLOAT}, FORMAT_ND, shape (n, c, d)
//   - Input  seq_len  : REQUIRED, {DT_INT32}, FORMAT_ND, shape (n,)
//   - Output output_y : REQUIRED, {DT_FLOAT}, FORMAT_ND, shape (n, c, d)
//   - Attr   normalize_type : REQUIRED String（"per_feature" | "all_features"）
//   - Attr   epsilon        : OPTIONAL Float，默认 1e-5
//   - AICore config: arch35 四计算单元共用同一 config（对齐 canndev TIK 版口径，仅 float32）

#include "register/op_def_registry.h"

namespace ops {

class NormalizeBatch : public OpDef {
public:
    explicit NormalizeBatch(const char* name) : OpDef(name)
    {
        // AutoContiguous：非连续 view 输入/输出由框架自动物化为连续布局，
        // kernel 侧恒按连续 ND 布局消费/产出。
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("seq_len")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("output_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("normalize_type").AttrType(REQUIRED).String();
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5f);

        OpAICoreConfig aicoreConfig;
        // 动态 shape/rank 放行口径与 infershape（UNKNOWN_RANK 透传）、binary json（shape=-2）
        // 保持一致（范式同 l2_normalize）：DynamicCompileStaticFlag(true) 保证 tiling 前
        // 动态维已静态实例化，tiling 侧的负维/非 3 维防御校验兜底
        // 注：不配置 opFile.value——kernel 入口统一为缺省 {ops}.cpp（op_kernel/normalize_batch.cpp），
        // 该字段仅用于绑定非默认入口（如 *_apt.cpp 变体），冗余配置会误导自动生成流程
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false);
        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().AddConfig("ascend960dt", aicoreConfig);
        this->AICore().AddConfig("ascend960pr", aicoreConfig);
        this->AICore().AddConfig("ascend350", aicoreConfig);
    }
};

OP_ADD(NormalizeBatch);
} // namespace ops
