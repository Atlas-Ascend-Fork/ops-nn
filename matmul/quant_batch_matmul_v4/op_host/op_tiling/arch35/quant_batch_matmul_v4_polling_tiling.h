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
 * \file quant_batch_matmul_v4_polling_tiling.h
 * \brief
 */
#pragma once

#include <cstdint>
#include <vector>

#include "op_host/tiling_templates_registry.h"
#include "common/inc/error_util.h"
#include "common/op_host/math_util_nn.h"
#include "tiling/tiling_api.h"
#include "op_cache_tiling.h"
#include "register/tilingdata_base.h"

#include <iostream>
#include <sstream>
#include <string>

namespace optiling {
struct PollingBasicBlockParam {
    uint64_t mSize = 0;
    uint64_t nSize = 0;
    uint64_t kSize = 0;
    uint64_t groupSize = 0;
    uint64_t singleM = 0;
    uint64_t singleN = 0;
    uint64_t mDim = 0;
    uint64_t nDim = 0;
    uint64_t baseM = 0;
    uint64_t baseN = 0;
    uint64_t baseK = 0;
    uint64_t aDtypeBits = 0;
    uint64_t bDtypeBits = 0;
    uint64_t x2ScaleDtypeBits = 0;
    uint64_t x1ScaleDtypeBits = 0;
    uint64_t scaleFactor = 0;
    double mte2Cost = 0;
    bool enablePreAlloc = true; // MTE2是否开启PreAllocation

    bool operator<(const PollingBasicBlockParam& other) const
    {
        if (mte2Cost != other.mte2Cost) {
            return mte2Cost < other.mte2Cost;
        }
        if (singleM % baseM != 0) { // 优先选择核内无尾块的解
            return false;
        }
        if (ops::CeilDiv(singleN, baseN) != ops::CeilDiv(other.singleN, other.baseN)) {
            return ops::CeilDiv(singleN, baseN) < ops::CeilDiv(other.singleN, other.baseN);
        }
        if (baseN != other.baseN) {
            return baseN < other.baseN;
        }
        return baseM > other.baseM;
    }

    bool isValid() const
    {
        return mSize != 0 && nSize != 0 && kSize != 0 && groupSize != 0 && aDtypeBits != 0 && bDtypeBits != 0 &&
               x1ScaleDtypeBits != 0 && x2ScaleDtypeBits != 0;
    }
    std::string ToString() const
    {
        std::stringstream ss;
        ss << "PollingBasicBlockParam { "
           << "mSize: " << mSize << ", "
           << "nSize: " << nSize << ", "
           << "kSize: " << kSize << ", "
           << "groupSize: " << groupSize << ", "
           << "singleM: " << singleM << ", "
           << "singleN: " << singleN << ", "
           << "mDim: " << mDim << ", "
           << "nDim: " << nDim << ", "
           << "baseM: " << baseM << ", "
           << "baseN: " << baseN << ", "
           << "baseK: " << baseK << ", "
           << "aDtypeBits: " << aDtypeBits << ", "
           << "bDtypeBits: " << bDtypeBits << ", "
           << "x2ScaleDtypeBits: " << x2ScaleDtypeBits << ", "
           << "x1ScaleDtypeBits: " << x1ScaleDtypeBits << ", "
           << "scaleFactor: " << scaleFactor << ", "
           << "mte2Cost: " << mte2Cost << " }";
        return ss.str();
    }
};

struct PlatformInfo {
    uint64_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
    uint64_t l0CSize = 0;
    double l2HbmBWRatio = 0;
    bool isValid() const
    {
        return aicNum != 0 && ubSize != 0 && l1Size != 0 && l0ASize != 0 && l0BSize != 0 && l0CSize != 0 &&
               l2HbmBWRatio > 1e-6;
    }
    std::string ToString() const
    {
        std::stringstream ss;
        ss << "PlatformInfos { "
           << "aicNum: " << aicNum << ", "
           << "ubSize: " << ubSize << ", "
           << "l1Size: " << l1Size << ", "
           << "l0ASize: " << l0ASize << ", "
           << "l0BSize: " << l0BSize << ", "
           << "l0CSize: " << l0CSize << ", "
           << "l2HbmBWRatio: " << l2HbmBWRatio << " }";
        return ss.str();
    }
};

struct PollingLimitParam {
    uint64_t alignSizeN = 0;
    uint64_t maxBaseMSize = 0;
    uint64_t maxBaseNSize = 0;
    uint64_t maxMNSize = 0;
};

constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128UL;
constexpr uint64_t BYTE_BITS = 8UL;
constexpr uint64_t CUBE_BLOCK = 16UL;
constexpr uint64_t MAX_PERF_STEPKB = 4UL;
constexpr uint64_t MAX_SCALE_SIZE = 16384UL;
constexpr uint64_t DB_SIZE = 2UL;

class QuantBatchMatmulV4PollingTiling {
public:
    QuantBatchMatmulV4PollingTiling() {}
    ~QuantBatchMatmulV4PollingTiling() = default;

    void SetShape(const uint64_t mSize, const uint64_t nSize, const uint64_t kSize, const uint64_t groupSize);
    void SetDtypeBits(const uint64_t aDtypeBits, const uint64_t bDtypeBits, const uint64_t x1ScaleDtypeBits,
                      const uint64_t x2ScaleDtypeBits);
    void SetPlatformInfo(const PlatformInfo& platformInfos);
    void SetLimitParams(const uint64_t alignSizeN, const uint64_t maxBaseMSize, const uint64_t maxBaseNSize,
                        const uint64_t maxMNSize);
    ge::graphStatus ComputeTiling();
    const PollingBasicBlockParam& GetTilingResult() const { return basicBlockParam_; }

protected:
    PollingBasicBlockParam basicBlockParam_;
    std::vector<PollingBasicBlockParam> optionalResults_;
    PlatformInfo platformInfos_;
    PollingLimitParam limitParam_;

private:
    void GetBasicBlockSolution();
    void CalculateBasicBlockTiling();
    bool CheckBufferSize(const uint64_t baseM, const uint64_t baseN);
    void GetMte2Cost();
    double BWUtilization(const uint64_t mDim, const uint64_t nDim) const;
    bool GetPollingTilingSolution();
    bool ClosePreAllocCond();
};
} // namespace optiling
