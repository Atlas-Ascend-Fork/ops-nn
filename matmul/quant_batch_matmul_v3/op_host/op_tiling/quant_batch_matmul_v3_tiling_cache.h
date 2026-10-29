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
 * \file quant_batch_matmul_v3_tiling_cache.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_V3_TILING_CACHE_H
#define QUANT_BATCH_MATMUL_V3_TILING_CACHE_H

#include "quant_batch_matmul_v3_tiling_cache_key.h"
#include "common/op_host/op_tiling/tiling_cache.h"

namespace optiling {
class QuantBatchMatmulV3HashItem {
public:
    explicit QuantBatchMatmulV3HashItem(const QuantBatchMatmulInfo& params,
                                        const Ops::NN::Optiling::AiCoreParams& aicoreParams)
        : hashKey_(params, aicoreParams)
    {}
    const QuantBatchMatmulV3HashInput& input() const { return hashKey_; }
    const BasicTiling& GetTiling() const { return tiling_; }
    void SetTiling(const BasicTiling& tiling) { tiling_ = tiling; }

private:
    QuantBatchMatmulV3HashInput hashKey_;
    BasicTiling tiling_;
};

using MMBasicTilingHash = Ops::NN::TilingCache<QuantBatchMatmulV3HashInput, QuantBatchMatmulV3HashItem>;
} // namespace optiling
#endif // QUANT_BATCH_MATMUL_V3_TILING_CACHE_H
