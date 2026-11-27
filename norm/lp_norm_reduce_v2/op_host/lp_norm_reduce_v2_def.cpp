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
 * \file lp_norm_reduce_v2_def.cpp
 * \brief LpNormReduceV2 算子定义，声明输入输出和算子配置
 *
 * 与 LpNormReduce（V1）的核心差异：属性 p 为 **Float**（默认 2.0，IEEE 754，
 * 支持 ±inf 与任意有限浮点，勿回退 V1 的 Int 哨兵口径）。
 * 多 soc 交付（同 DAV_3510/arch35 家族，复用同一 AICore 配置），不注册 910b。
 */

#include "register/op_def_registry.h"

namespace ops {
class LpNormReduceV2 : public OpDef {
public:
    explicit LpNormReduceV2(const char* name) : OpDef(name)
    {
        this->Input("x")                                           // 输入x定义
            .ParamType(REQUIRED)                                   // 必选输入
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}) // 支持数据类型
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}) // 支持format格式（与 dtype 一一配对）
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}) // 未确定大小shape对应format格式
            .AutoContiguous(); // 内存自动连续化（contiguous: optional）
        this->Output("y")      // 输出y定义（dtype 与 x 一致，InferDataType 推导）
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 属性注册（属性索引：0=p, 1=axes, 2=keepdim, 3=epsilon，与 proto/infershape/tiling 一致）
        this->Attr("p").AttrType(OPTIONAL).Float(static_cast<float>(2.0)); // 范数阶数（IEEE 754，支持 ±inf）
        this->Attr("axes").AttrType(OPTIONAL).ListInt(std::vector<int64_t>({})); // 归约轴（空 = 全维度，支持负索引）
        this->Attr("keepdim").AttrType(OPTIONAL).Bool(false);                    // 是否保留被归约维度
        this->Attr("epsilon").AttrType(OPTIONAL).Float(static_cast<float>(1e-12)); // 预留参数，不参与计算

        // 各 soc 同属 DAV_3510（arch35），复用同一 AICore 配置
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "lp_norm_reduce_v2"); // kernel 文件绑定：op_kernel/lp_norm_reduce_v2.cpp
        this->AICore().AddConfig("ascend950", aicoreConfig950);
        this->AICore().AddConfig("ascend350", aicoreConfig950);
        this->AICore().AddConfig("ascend960dt", aicoreConfig950);
        this->AICore().AddConfig("ascend960pr", aicoreConfig950);
    }
};
OP_ADD(LpNormReduceV2); // 添加算子信息库
} // namespace ops
