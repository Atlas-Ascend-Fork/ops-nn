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
 * \file fused_weight_quant_mat_mul_def.cpp
 * \brief FusedWeightQuantMatMul OpDef 注册。
 *        原型注册完整能力面（proto 层）；本文件注册已实现 dtype 组合
 *        （A16W8：x fp16 × weight int8 × antiquant_scale uint64 × y {fp16,int8} × bias{fp16,int32}），
 *        A16W4 / 独立 quant 输入等组合待后续阶段实现时再扩展注册。
 *        y int8 为输出量化（2026-10-09 增补）：antiquant_scale 语义随 y dtype 变化——
 *        y INT8 时其 uint64 编码为"反量化 ÷ 输出量化"合并 scale（由融合 pass 常量折叠生成，
 *        与底座 wqbmm_v2 的 y=INT8 契约一致）；y FP16 时为纯反量化 scale。
 *        非对称量化（2026-10-20 增补）：y INT8 且 AscendQuant 带整数零点 offset（[-128, 127]）时，
 *        offset 编入同一 uint64 的高位（bits 37-44 值 + bit45 符号 + bit46 signed 标志，对齐老代
 *        MatmulQuantToFixpipeFusionPass 复合编码先例；对称量化高位全零不变）。
 *        仅图模式。
 *        注意：各输入的 dtype/format 列表长度必须一致（矩形，每组代表一个 dtype 组合），
 *        否则 ini 生成器会丢弃全部 dtype/format 信息导致二进制编译失败。
 */

#include "register/op_def_registry.h"

namespace ops {
class FusedWeightQuantMatMul : public OpDef {
public:
    explicit FusedWeightQuantMatMul(const char* name) : OpDef(name)
    {
        // 属性定义与 proto（fused_weight_quant_mat_mul_proto.h）严格对齐：
        // 不注册 inner_precise（融合算子固定高精度模式，融合 pass 透传时忽略原节点该属性）
        this->Attr("transpose_x").AttrType(OPTIONAL).Bool(false);
        this->Attr("transpose_weight").AttrType(OPTIONAL).Bool(false);
        this->Attr("antiquant_group_size").AttrType(OPTIONAL).Int(0);
        this->Attr("dtype").AttrType(OPTIONAL).Int(-1);
        this->Attr("fused_op_type").AttrType(OPTIONAL).String("relu");

        OpAICoreConfig aicoreConfig;
        // 组合0/1：y fp16 × bias fp16/int32；组合2/3：y int8（输出量化）× bias fp16/int32
        // （bias 缺省场景由 HAS_BIAS=0 实例化覆盖）
        aicoreConfig.Input("x").ParamType(REQUIRED).DataType({}).Format({}).UnknownShapeFormat({});
        aicoreConfig.Input("weight").ParamType(REQUIRED).DataType({}).Format({}).UnknownShapeFormat({});
        aicoreConfig.Input("antiquant_scale").ParamType(REQUIRED).DataType({}).Format({}).UnknownShapeFormat({});
        // 不支持 antiquant_offset/quant_scale/quant_offset（checker 强制为空），
        // dtype 仅为编译占位，与底座行为一致（quant 输入为后续独立通路预留槽位）
        aicoreConfig.Input("antiquant_offset").ParamType(OPTIONAL).DataType({}).Format({}).UnknownShapeFormat({});
        aicoreConfig.Input("quant_scale").ParamType(OPTIONAL).DataType({}).Format({}).UnknownShapeFormat({});
        aicoreConfig.Input("quant_offset").ParamType(OPTIONAL).DataType({}).Format({}).UnknownShapeFormat({});
        aicoreConfig.Input("bias").ParamType(OPTIONAL).DataType({}).Format({}).UnknownShapeFormat({});
        aicoreConfig.Output("y").ParamType(REQUIRED).DataType({}).Format({}).UnknownShapeFormat({});
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
    }
};

OP_ADD(FusedWeightQuantMatMul);
} // namespace ops
