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
 * \file max_pool_v3_grad_tiling_base.h
 * \brief Common tiling base for MaxPoolV3Grad templates.
 */

#ifndef MAX_POOL_V3_GRAD_TILING_BASE_H
#define MAX_POOL_V3_GRAD_TILING_BASE_H

#include <cstdint>
#include <string>

#include "log/log.h"
#include "op_host/tiling_base.h"
#include "platform/platform_ascendc.h"
#include "../../../pool_grad_common/op_kernel/arch35/max_pool_grad_struct.h"
#include "../../../pool_grad_common/op_kernel/arch35/max_pool_grad_with_argmax_struct_common.h"
#include "../../../pool_grad_common/op_host/arch35/max_pool_grad_with_argmax_tiling_common.h"
#include "../../../pool_grad_common/op_host/arch35/util.h"

namespace optiling {

using namespace MaxPoolGradWithArgmaxNHWCNameSpace;
using namespace PoolGradNameSpace;

constexpr uint64_t DCACHE_SIZE = 128UL * 1024UL;
constexpr int64_t PER_CORE_MIN_ELEMENTS = 1024;
constexpr int64_t INT32_SIZE = 4;
constexpr int64_t INT64_SIZE = 8;
constexpr int64_t FLOAT32_SIZE = 4;
constexpr int64_t KSIZE_THRESHOLD = 128;
constexpr int64_t KERNEL_OFFSET = 1;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint32_t BATCH_MODE = 1;

constexpr int32_t ATTR_INDEX_KSIZE = 0;
constexpr int32_t ATTR_INDEX_STRIDES = 1;
constexpr int32_t ATTR_INDEX_PADDING_MODE = 2;
constexpr int32_t ATTR_INDEX_PADS = 3;
constexpr int32_t ATTR_INDEX_DATA_FORMAT = 4;
constexpr int32_t ATTR_INDEX_GLOBAL_POOLING = 5;
constexpr int32_t ATTR_INDEX_CEIL_MODE = 6;

constexpr size_t ATTR_LIST_SIZE = 4;
constexpr size_t ORIG_INPUT_INDEX = 0;
constexpr size_t ORIG_OUTPUT_INDEX = 1;
constexpr size_t GRAD_INDEX = 2;
constexpr size_t OUT_GRAD_INDEX = 0;

struct MaxPoolV3GradCompileInfo {};

class MaxPoolV3GradTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit MaxPoolV3GradTilingBase(gert::TilingContext* context) : TilingBaseClass(context) {}
    ~MaxPoolV3GradTilingBase() override = default;

    MaxPoolGradWithArgmaxInputInfoCommon inputData;
    int64_t coreNum_{0};
    int64_t ubSize_{0};
    int64_t inputHW_{0};
    int64_t padBottom_{0};
    int64_t padRight_{0};
    int64_t totalInputElements_{0};
    int64_t totalOutputElements_{0};

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override;
};

} // namespace optiling
#endif // MAX_POOL_V3_GRAD_TILING_BASE_H
