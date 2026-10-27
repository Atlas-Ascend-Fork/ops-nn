/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_MULTI_CONV2D_ATTRS_H_
#define FUSED_MULTI_CONV2D_ATTRS_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace FusedMultiConv2dAttrs {

enum AttributeIndex : size_t {
    ATTR_BIAS_MASK,
    ATTR_DTYPE,
    ATTR_STRIDES,
    ATTR_PADS,
    ATTR_DILATIONS,
    ATTR_GROUPS,
    ATTR_DATA_FORMAT,
    ATTR_OFFSET_X,
    ATTR_ROUND_MODE,
    ATTR_PAD_MODE,
    ATTR_ENABLE_HF32,
    ATTR_ENABLE_RELU,
    PUBLIC_ATTR_COUNT,
    ATTR_ASCENDC_OP_PARA_SIZE = PUBLIC_ATTR_COUNT,
    ATTR_FIXED_SHIFT,
    RUNTIME_ATTR_COUNT,
};

enum class DataFormat : int64_t { NCHW = 0, NHWC = 1 };
enum class RoundMode : int64_t { RINT = 0, ROUND = 1 };
enum class PadMode : int64_t { SPECIFIC = 0, SAME = 1, VALID = 2, SAME_UPPER = 3, SAME_LOWER = 4, NOTSET = 5 };

// 根据属性名 name 查找格式、舍入或补边属性的索引，其他名称返回公开属性总数。
inline size_t MappedEnumAttrIndex(const char* name)
{
    if (std::strcmp(name, "data_format") == 0) {
        return ATTR_DATA_FORMAT;
    }
    if (std::strcmp(name, "round_mode") == 0) {
        return ATTR_ROUND_MODE;
    }
    if (std::strcmp(name, "pad_mode") == 0) {
        return ATTR_PAD_MODE;
    }
    return PUBLIC_ATTR_COUNT;
}

// 将 attrIndex 指定属性的枚举值 value 转为名称，未知值返回 nullptr。
inline const char* MappedEnumValueName(size_t attrIndex, int64_t value)
{
    if (attrIndex == ATTR_DATA_FORMAT) {
        switch (static_cast<DataFormat>(value)) {
            case DataFormat::NCHW:
                return "NCHW";
            case DataFormat::NHWC:
                return "NHWC";
        }
    } else if (attrIndex == ATTR_ROUND_MODE) {
        switch (static_cast<RoundMode>(value)) {
            case RoundMode::RINT:
                return "rint";
            case RoundMode::ROUND:
                return "round";
        }
    } else if (attrIndex == ATTR_PAD_MODE) {
        switch (static_cast<PadMode>(value)) {
            case PadMode::SPECIFIC:
                return "SPECIFIC";
            case PadMode::SAME:
                return "SAME";
            case PadMode::VALID:
                return "VALID";
            case PadMode::SAME_UPPER:
                return "SAME_UPPER";
            case PadMode::SAME_LOWER:
                return "SAME_LOWER";
            case PadMode::NOTSET:
                return "NOTSET";
        }
    }
    return nullptr;
}

constexpr const char* FIXED_SHIFT_ATTR = "fixed_shift_value";
constexpr int64_t DEFAULT_FIXED_SHIFT = 0;

} // namespace FusedMultiConv2dAttrs

#endif
