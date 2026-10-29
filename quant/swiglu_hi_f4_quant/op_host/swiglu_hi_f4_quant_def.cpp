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
 * \file swiglu_hi_f4_quant_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "../../../common/inc/op_host/op_const_def_nn.h"

namespace ops {

class SwigluHiF4Quant : public OpDef {
public:
    explicit SwigluHiF4Quant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("group_index")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({Ops::NN::DT_HIFLOAT4, Ops::NN::DT_HIFLOAT4})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("scale")
            .ParamType(REQUIRED)
            .DataType({Ops::NN::DT_HIFLOAT4_SCALE, Ops::NN::DT_HIFLOAT4_SCALE})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("activate_dim").AttrType(OPTIONAL).Int(-1);
        this->Attr("activate_left").AttrType(OPTIONAL).Bool(false);
        this->Attr("swiglu_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("clamp_limit").AttrType(OPTIONAL).Float(7.0f);
        this->Attr("glu_alpha").AttrType(OPTIONAL).Float(1.702f);
        this->Attr("glu_bias").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("scale_alg").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true).DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend960dt", regbaseCfg);
        this->AICore().AddConfig("ascend960pr", regbaseCfg);
    }
};

OP_ADD(SwigluHiF4Quant);

} // namespace ops
