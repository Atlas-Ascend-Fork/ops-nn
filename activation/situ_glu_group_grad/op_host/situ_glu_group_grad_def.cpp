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
 * \file situ_glu_group_grad_def.cpp
 * \brief OpDef registration for SituGluGroupGrad (grad_y/x/weight/group_index -> grad_x/grad_weight, ascend950).
 */
#include <register/op_def_registry.h>
#include "op_common/op_host/util/op_arch_helper.h"

namespace ops {
constexpr float DEFAULT_BETA = 1.0;
constexpr float DEFAULT_LINEAR_BETA = 0.0;
// 0: group_index is a cumsum list (its last element is the total row count);
// 1: group_index is a count list (the sum of its elements is the total row count).
constexpr int64_t DEFAULT_GROUP_LIST_TYPE = 0;

static const std::vector<ge::DataType> xDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
static const std::vector<ge::DataType> weightDtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
static const std::vector<ge::DataType> groupIndexDtype = {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
static const std::vector<ge::Format> xFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
static const std::vector<ge::Format> xUnknownFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class SituGluGroupGrad : public OpDef {
public:
    explicit SituGluGroupGrad(const char* name) : OpDef(name)
    {
        // SituGluGroupGrad is an ascend950-only op: it exposes the optional weight / group_index inputs and the
        // (required) grad_weight output in the op prototype itself, so no A2 config is registered.
        this->Input("grad_y")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xUnknownFormat)
            .AutoContiguous();
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xUnknownFormat)
            .AutoContiguous();
        this->Input("weight")
            .ParamType(OPTIONAL)
            .DataType(weightDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xUnknownFormat)
            .AutoContiguous();
        this->Input("group_index")
            .ParamType(OPTIONAL)
            .DataType(groupIndexDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xUnknownFormat)
            .AutoContiguous();
        this->Output("grad_x").ParamType(REQUIRED).DataType(xDtype).Format(xFormat).UnknownShapeFormat(xUnknownFormat);
        this->Output("grad_weight")
            .ParamType(REQUIRED)
            .DataType(weightDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xUnknownFormat);
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);
        this->Attr("beta").AttrType(OPTIONAL).Float(DEFAULT_BETA);
        this->Attr("linear_beta").AttrType(OPTIONAL).Float(DEFAULT_LINEAR_BETA);
        this->Attr("activate_left").AttrType(OPTIONAL).Bool(true);
        this->Attr("group_list_type").AttrType(OPTIONAL).Int(DEFAULT_GROUP_LIST_TYPE);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "situ_glu_group_grad");
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510, aicoreConfig);
    }
};
OP_ADD(SituGluGroupGrad);
} // namespace ops
