/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"
#include "op_common/op_host/util/op_arch_helper.h"

namespace ops {
class IndexByTensor : public OpDef {
public:
    explicit IndexByTensor(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> valueTypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT64,
                                                      ge::DT_INT32, ge::DT_BOOL,    ge::DT_UINT8,
                                                      ge::DT_INT8,  ge::DT_BF16,    ge::DT_COMPLEX64};
        const std::vector<ge::Format> formats(valueTypes.size(), ge::FORMAT_ND);
        const std::vector<ge::DataType> indexTypes(valueTypes.size(), ge::DT_INT64);

        this->Input("x").ParamType(REQUIRED).DataType(valueTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("indices").ParamType(DYNAMIC).DataType(indexTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("y").ParamType(REQUIRED).DataType(valueTypes).Format(formats).UnknownShapeFormat(formats);
        this->Attr("indices_mask").AttrType(OPTIONAL).ListInt({});

        // Static graphs lower to Index; other graphs use the registered ACLNN fallback.
        // IndexByTensor has no independent tiling or kernel, so GE must not compile it as a TBE operator.
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "aclnn_only");
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510, config);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9201, config);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9202, config);
    }
};

OP_ADD(IndexByTensor);
} // namespace ops
