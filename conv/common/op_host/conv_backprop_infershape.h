/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_backprop_infershape.h
 * \brief
 */

#ifndef OPS_CONV_BACKPROP_INFERSHAPE_H_
#define OPS_CONV_BACKPROP_INFERSHAPE_H_

#include <array>
#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/infer_shape_context.h>
#include <exe_graph/runtime/infer_shape_range_context.h>
#include <log/log.h>
#include <util/shape_util.h>
#include <register/op_impl_registry.h>
#include "error_util.h"

namespace Ops {
namespace NN {
namespace Conv {
ge::graphStatus InferShapeForConvBackprop(gert::InferShapeContext* context, size_t const_tensor_idx,
                                          const char* const_tensor_name, size_t dim_num);
ge::graphStatus PartialInferInputShapeWhenConstInvisible(gert::InferShapeContext* context, size_t filter_index = 1,
                                                         int64_t channel_multiplier = 1);
ge::graphStatus InferShapeRangeForConvBackprop(gert::InferShapeRangeContext* context, size_t filter_index,
                                               int64_t channel_multiplier);
// Conv2DBackpropFilterV2 和 Conv2DBackpropFilterV3 共用
ge::graphStatus InferShapeForConv2DBackpropFilter(gert::InferShapeContext* context);
ge::graphStatus PartialInferFilterShapeWhenConstInvisible(gert::InferShapeContext* context);
// Conv2DBackpropFilterV2 和 Conv2DBackpropFilterV3 共用
ge::graphStatus InferDataTypeForConv2DBackpropFilter(gert::InferDataTypeContext* context);
// Conv2DBackpropInputV2 和 Conv3DBackpropInputV2 共用
ge::graphStatus InferDataTypeForConvBackpropInputV2(gert::InferDataTypeContext* context);
// Conv2DTransposeV2 和 Conv3DTransposeV2 共用
bool CheckOutputAllZero(const gert::Shape* shape);
bool CheckOutputAllZeroFrom2D(const gert::InferShapeContext* context, const gert::Shape* shape, int32_t d_index);
// Conv2DTransposeV2 和 Conv3DTransposeV2 共用
ge::graphStatus InferDataTypeForConvTransposeV2(gert::InferDataTypeContext* context);
ge::graphStatus InferShapeForConvBackpropExtend3D(gert::InferShapeContext* context, size_t const_tensor_idx,
                                                  const char* const_tensor_name);
ge::graphStatus SetOutputShapeDim(const gert::InferShapeContext* context, const gert::Tensor* const_tensor,
                                  gert::Shape* y_shape);
int32_t GetConvBackpropIndex(ge::Format format);

// 新旧模板属性列名集合(按def声明顺序)。模板增删属性只改集合，槽位与布局属性总数由集合推导。
// inline constexpr std::array: 无隐式堆分配与动态初始化，多cpp包含共享同一只读实例。
inline constexpr std::array<const char*, 9> C2D_BP_FILTER_LAYOUT = {"strides", "pads",           "dilations",
                                                                    "groups",  "data_format",    "output_padding",
                                                                    "padding", "from_depthwise", "_op_impl_mode_enum"};
inline constexpr std::array<const char*, 8> C2D_BP_FILTER_V2_LAYOUT = {
    "strides", "pads", "dilations", "groups", "data_format", "padding", "from_depthwise", "use_hf32"};
inline constexpr std::array<const char*, 8> C3D_BP_FILTER_LAYOUT = {
    "strides", "pads", "dilations", "groups", "data_format", "output_padding", "padding", "_op_impl_mode_enum"};
inline constexpr std::array<const char*, 10> C3D_BP_FILTER_V2_LAYOUT = {
    "strides",        "pads",    "dilations",          "groups",        "data_format", "enable_hf32",
    "output_padding", "padding", "_op_impl_mode_enum", "from_depthwise"};

// 布列集合以(首指针,列数)形态解引用：布局槽位下标与布局属性总数由 MakeBpParamIdx 从集合推导填入。
struct BpParamIdx {
    const char* const* layout = nullptr;
    size_t layoutSize = 0;
    const char* const* fallbackLayout = nullptr;
    size_t fallbackLayoutSize = 0;
    int32_t outputPaddingSize = 0;
};

template <size_t layoutSize, size_t fallbackSize>
constexpr BpParamIdx MakeBpParamIdx(const std::array<const char*, layoutSize>& layout,
                                    const std::array<const char*, fallbackSize>& fallbackLayout,
                                    int32_t outputPaddingSize)
{
    return {layout.data(), layout.size(), fallbackLayout.data(), fallbackLayout.size(), outputPaddingSize};
}

template <size_t layoutSize>
constexpr BpParamIdx MakeBpParamIdx(const std::array<const char*, layoutSize>& layout, int32_t outputPaddingSize)
{
    return {layout.data(), layout.size(), nullptr, 0, outputPaddingSize};
}

int32_t AttrIndexIn(const char* const* layout, size_t layoutSize, const char* attrName);
ge::graphStatus CheckOutputPaddingListSize(gert::InferShapeContext* context, const BpParamIdx& bpParamIdx);
const bool* GetFromDepthwiseAttr(const gert::RuntimeAttrs* runtimeAttrs, const BpParamIdx& bpParamIdx);

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // OPS_CONV_BACKPROP_INFERSHAPE_H_
