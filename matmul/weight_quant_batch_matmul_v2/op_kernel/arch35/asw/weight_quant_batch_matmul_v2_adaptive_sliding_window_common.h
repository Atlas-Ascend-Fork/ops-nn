/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window_common.h
 * \brief
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "../../weight_quant_batch_matmul_v2_constant.h"

namespace WeightQuantBatchMatmulV2::Arch35 {

static constexpr uint64_t L1_BUFFER_HALF_SIZE = 256 * 1024;
static constexpr uint64_t L1_BUFFER_SIZE = 512 * 1024;
static constexpr uint64_t L0A_BUFFER_SIZE = 64 * 1024;
static constexpr uint64_t L0B_BUFFER_SIZE = 64 * 1024;
static constexpr uint64_t L0C_BUFFER_SIZE = 256 * 1024;
static constexpr uint64_t BIAS_TABLE_SIZE = 4 * 1024;

static constexpr uint64_t IDX_0 = 0;
static constexpr uint64_t SCALE_COPY_GROUP_SIZE = 2;

static constexpr uint8_t DOUBLE_BUFFER = 2;
static constexpr uint8_t MTE_SYNC_FLAG_ID_4 = 4;
static constexpr uint8_t MTE_SYNC_FLAG_ID_5 = 5;
static constexpr uint8_t MTE_SYNC_FLAG_ID_6 = 6;
static constexpr uint8_t MTE_SYNC_FLAG_ID_7 = 7;
static constexpr uint8_t MTE_SYNC_FLAG_ID_8 = 8;
static constexpr uint8_t MTE_SYNC_DB_FLAG_IDS_1[DOUBLE_BUFFER] = {MTE_SYNC_FLAG_ID_4, MTE_SYNC_FLAG_ID_5};
static constexpr uint8_t MTE_SYNC_DB_FLAG_IDS_2[DOUBLE_BUFFER] = {MTE_SYNC_FLAG_ID_7, MTE_SYNC_FLAG_ID_8};

template <typename T>
struct C0 {
    static constexpr uint64_t value = AscendC::IsSupportB4<T>() ? 64L : (32L / sizeof(T));
};

template <typename DtypeWeight>
constexpr uint64_t GetMxGroupSize()
{
    if (IsSameType<DtypeWeight, hifloat4x2_t>::value) {
        // hif4类型的groupSize是64
        return 64UL;
    } else {
        // 其他类型的groupSize是32
        return 32UL;
    }
}

__aicore__ inline void NotifyVector() { AscendC::CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIV_AIC_FLAG); }

__aicore__ inline void WaitForCube() { AscendC::CrossCoreWaitFlag<SYNC_MODE4, PIPE_V>(SYNC_AIV_AIC_FLAG); }

__aicore__ inline void WaitForVector() { AscendC::CrossCoreWaitFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIC_AIV_FLAG); }

__aicore__ inline void NotifyCube() { AscendC::CrossCoreSetFlag<SYNC_MODE4, PIPE_V>(SYNC_AIC_AIV_FLAG); }

} // namespace WeightQuantBatchMatmulV2::Arch35
