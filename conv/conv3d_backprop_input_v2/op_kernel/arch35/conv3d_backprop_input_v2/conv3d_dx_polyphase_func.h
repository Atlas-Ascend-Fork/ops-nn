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
 * \file conv3d_dx_polyphase_func.h
 * \brief 一维 Polyphase 公共定义、内存布局与模式入口。
 */
#ifndef CONV3D_DX_POLYPHASE_FUNC_H
#define CONV3D_DX_POLYPHASE_FUNC_H

#include <cstdint>
#include "kernel_operator.h"
#include "utils/std/algorithm.h"
#include "op_common/op_kernel/math_util.h"
#include "op_common/op_kernel/platform_util.h"
#include "../convolution_3d_backprop/polyphase_impl/conv_bp_input_sub_func_polyphase_data_move.h"

namespace AscendC {

enum class PolyphaseMode : uint8_t { PHASED = 0, DIRECT_DENSE = 1 };
enum class PolyphaseStrategy : uint8_t { ONE_PHASE, SERIAL_DEPTHWISE, PARALLEL_DEPTHWISE, DENSE_SEGMENTED };

constexpr uint8_t POLYPHASE_SYNC_MODE = 4U;
// 同步模式 4 中，AIC flag 16 + bank 对应 AIV1 的 bank flag。
constexpr uint8_t POLYPHASE_AIV1_FLAG_BASE = 16U;
constexpr uint8_t POLYPHASE_ALL_PHASE_B1 = 1U;
constexpr uint32_t POLYPHASE_C0 = Convolution3DBackpropFunc::PolyphaseDataMove::C0;

// DIRECT 全驻留位与 host tiling 保持一致。
constexpr uint8_t POLYPHASE_A_FULL = 1U;
constexpr uint8_t POLYPHASE_B_FULL = 2U;
constexpr uint32_t POLYPHASE_DOUBLE_BUFFER = 2U;
// 滤波器打包使用独立 raw ND 和 packed NZ 缓冲。
constexpr uint32_t POLYPHASE_FILTER_PACK_BUFFERS = 2U;
constexpr uint32_t POLYPHASE_NDDMA_2D = 2U;
constexpr uint32_t POLYPHASE_NDDMA_3D = 3U;
constexpr uint32_t POLYPHASE_NDDMA_4D = 4U;
constexpr uint32_t POLYPHASE_NDDMA_GM_STRIDE_BITS = 40U;
constexpr uint32_t POLYPHASE_GATHER_INDEX_LIMIT = static_cast<uint32_t>(UINT16_MAX) + 1U;
constexpr uint32_t POLYPHASE_INTERLEAVE_STRIDE = 2U;
constexpr uint32_t POLYPHASE_GATHER_STRIDE3 = 3U;
constexpr uint32_t POLYPHASE_GATHER_STRIDE4 = 4U;
constexpr uint32_t POLYPHASE_GATHER_STRIDE5 = 5U;
constexpr event_t POLYPHASE_FILTER_EVENT = static_cast<event_t>(2U);
constexpr event_t POLYPHASE_DIRECT_L1_EVENT = static_cast<event_t>(7U);
constexpr uint32_t POLYPHASE_GROUP_PACK_REQUEST = 3U;
constexpr uint32_t POLYPHASE_GROUP_PACK_READY = 4U;
constexpr uint32_t POLYPHASE_OUTPUT_STORE_DONE = 2U;

struct PolyphaseMemoryLayout {
    uint64_t a1Bytes = 0U;
    uint64_t phaseB1Bytes = 0U;
    uint64_t b1BankBytes = 0U;
    uint64_t phaseOutputOffsetBytes = 0U;
    uint64_t outputOffsetBytes = 0U;
    uint64_t biasOffsetBytes = 0U;
    uint64_t indexOffsetBytes = 0U;
    uint64_t outputBankPitchBytes = 0U;
};

struct PolyphaseWorkGeometry {
    uint32_t batchIdx = 0;
    uint32_t nStart = 0;
    uint32_t curN = 0;
    uint64_t uStart = 0;
    uint32_t curJ = 0;
    uint32_t localA1W = 0;
    int64_t xStart = 0;
};

template <typename T>
class Conv3dPolyphase {
public:
    using L0cT = float;
    using DataMove = Convolution3DBackpropFunc::PolyphaseDataMove;

    __aicore__ inline Conv3dPolyphase() = default;

    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR dedy, GM_ADDR y,
                                const Conv3DBackpropInputArch35TilingData& tilingData, GM_ADDR bias = nullptr)
    {
        tiling_ = &tilingData;
        filterGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(filter));
        dedyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dedy));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        if (bias != nullptr) {
            biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
        }
    }

    __aicore__ inline void ProcessDx()
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if (tiling_->enableFullLoad != POLYPHASE_ALL_PHASE_B1) {
            ProcessTranspose();
            return;
        }
#endif
        ProcessDxAllPhases();
    }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    __aicore__ inline void ProcessTranspose()
    {
        if ASCEND_IS_AIC_SCALAR {
            ProcessRole<PolyphaseRole::CUBE>();
        }
        if ASCEND_IS_AIV_SCALAR {
            if (GetSubBlockIdx() == 0U) {
                ProcessRole<PolyphaseRole::OUTPUT>();
            } else {
                ProcessRole<PolyphaseRole::FILTER>();
            }
        }
    }
#endif

protected:
    // AIV0 输出；AIV1 在 PHASED 打包滤波器，在 DIRECT 输出分片。
    enum class PolyphaseRole { CUBE, OUTPUT, FILTER };

    using PhasedStrategy = PolyphaseStrategy;

    struct PhasedCoreState {
        uint64_t filterSegmentOrdinal = 0U;
        uint64_t denseKTileOrdinal = 0U;
        uint32_t filterPhaseOrdinal = 0U;
        bool filterInitialized = false;
        bool outputInitialized = false;
    };

    struct PhasedTaskDescriptor {
        PolyphaseWorkGeometry geometry;
        uint32_t phaseGroup;
        uint64_t round;
    };

    struct PhaseSegment {
        uint64_t first;
        uint32_t length;
        uint32_t bank;
        int64_t xStart;
    };

    // 公共任务几何与内存布局
    __aicore__ inline uint64_t GetRounds(uint64_t blockIdx) const
    {
        const uint64_t items = static_cast<uint64_t>(tiling_->batch) * tiling_->nDim * tiling_->mDim;
        if (items == 0U || tiling_->coreNum == 0U || tiling_->coreNum > items || blockIdx >= tiling_->coreNum) {
            return 0U;
        }
        return items / tiling_->coreNum + static_cast<uint64_t>(blockIdx < items % tiling_->coreNum);
    }

    __aicore__ inline void DecodeGeometry(uint64_t basic, PolyphaseWorkGeometry& geometry) const
    {
        const uint64_t perBatch = static_cast<uint64_t>(tiling_->nDim) * tiling_->mDim;
        const uint64_t remainder = basic % perBatch;
        geometry.batchIdx = static_cast<uint32_t>(basic / perBatch);
        geometry.nStart = static_cast<uint32_t>(remainder / tiling_->mDim) * tiling_->baseN;
        geometry.curN = Std::min(tiling_->baseN, tiling_->cin - geometry.nStart);
        geometry.uStart = remainder % tiling_->mDim * tiling_->singleCoreM;
        const uint64_t j = static_cast<uint64_t>(tiling_->wo) + tiling_->stepKa - 1U;
        geometry.curJ = Std::min(static_cast<uint32_t>(tiling_->singleCoreM),
                                 static_cast<uint32_t>(j - geometry.uStart));
        geometry.localA1W = Ops::Base::CeilAlign<uint32_t>(geometry.curJ, POLYPHASE_C0) + tiling_->stepKa - 1U;
        const uint64_t halo = tiling_->stepKa - 1U;
        geometry.xStart = geometry.uStart >= halo ? static_cast<int64_t>(geometry.uStart - halo) :
                                                    -static_cast<int64_t>(halo - geometry.uStart);
    }

    __aicore__ inline bool HasBias() const { return tiling_->isBiasFullLoad != 0; }

    __aicore__ inline void PrepareMemoryLayout(bool dense)
    {
        memoryLayout_ = {};
        const uint64_t stride = tiling_->stepKb;
        const uint64_t q = tiling_->stepKa;
        const uint64_t baseM = tiling_->baseM;
        const uint64_t stripe = Ops::Base::CeilAlign<uint64_t>(tiling_->singleCoreM, POLYPHASE_C0);
        const uint64_t channels = Ops::Base::CeilAlign<uint64_t>(dense ? tiling_->cout : POLYPHASE_C0, POLYPHASE_C0);
        memoryLayout_.a1Bytes = Ops::Base::CeilAlign<uint64_t>(channels * (stripe + q - 1U) * sizeof(T),
                                                               Ops::Base::GetUbBlockSize());
        const uint64_t compact = channels * q;
        const uint64_t resident = tiling_->enableFullLoad == POLYPHASE_ALL_PHASE_B1 ? stride : 1U;
        memoryLayout_.phaseB1Bytes = Ops::Base::CeilAlign<uint64_t>(POLYPHASE_C0 * compact * sizeof(T),
                                                                    Ops::Base::GetUbBlockSize());
        memoryLayout_.b1BankBytes = memoryLayout_.phaseB1Bytes * resident;
        const uint64_t outputM = tiling_->enableFullLoad == POLYPHASE_ALL_PHASE_B1 ? baseM : stripe;
        const uint64_t phaseElements = stride * POLYPHASE_C0 * outputM;
        const uint64_t outputElements = stride == POLYPHASE_INTERLEAVE_STRIDE ?
                                            stride * outputM :
                                            Ops::Base::CeilAlign<uint64_t>(stride * outputM,
                                                                           (Ops::Base::GetVRegSize() / sizeof(T)));
        uint64_t cursor = Ops::Base::CeilAlign<uint64_t>(phaseElements * (dense ? sizeof(float) : sizeof(T)),
                                                         Ops::Base::GetUbBlockSize());
        memoryLayout_.phaseOutputOffsetBytes = dense ? cursor : 0U;
        if (dense) {
            cursor += Ops::Base::CeilAlign<uint64_t>(phaseElements * sizeof(T), Ops::Base::GetUbBlockSize());
        }
        memoryLayout_.outputOffsetBytes = cursor;
        cursor += Ops::Base::CeilAlign<uint64_t>(POLYPHASE_C0 * outputElements * sizeof(T),
                                                 Ops::Base::GetUbBlockSize());
        memoryLayout_.biasOffsetBytes = cursor;
        cursor += dense ? POLYPHASE_C0 * sizeof(float) : 0U;
        memoryLayout_.indexOffsetBytes = cursor;
        if (stride != POLYPHASE_INTERLEAVE_STRIDE) {
            cursor += Ops::Base::CeilAlign<uint64_t>(outputElements * sizeof(uint16_t), Ops::Base::GetUbBlockSize());
        }
        memoryLayout_.outputBankPitchBytes = Ops::Base::CeilAlign<uint64_t>(cursor, Ops::Base::GetUbBlockSize());
    }

    __aicore__ inline void ComputeMatmul(LocalTensor<L0cT>& acc, const LocalTensor<T>& a, const LocalTensor<T>& b,
                                         uint32_t m, uint32_t n, uint32_t k, bool firstK) const
    {
        MmadParams params;
        params.m = static_cast<uint16_t>(m);
        params.n = static_cast<uint16_t>(n);
        params.k = static_cast<uint16_t>(k);
        params.cmatrixInitVal = firstK;
        params.cmatrixSource = false;
        params.isBias = false;
        Mmad(acc, a, b, params);
    }

    __aicore__ inline static int64_t FloorDivSigned(int64_t value, int64_t divisor)
    {
        return value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor);
    }

    __aicore__ inline static int64_t CeilDivSigned(int64_t value, int64_t divisor)
    {
        return -FloorDivSigned(-value, divisor);
    }
#include "../convolution_3d_backprop/polyphase_impl/conv_bp_input_sub_func_polyphase_transform.h"
#include "../convolution_3d_backprop/polyphase_impl/conv_bp_input_sub_func_polyphase_pipeline.h"
#include "../convolution_3d_backprop/polyphase_impl/conv_bp_input_sub_func_polyphase_direct_dense.h"
#include "../convolution_3d_backprop/polyphase_impl/conv_bp_input_sub_func_polyphase_depthwise_segmented.h"
#include "../convolution_3d_backprop/polyphase_impl/conv_bp_input_sub_func_polyphase_dense_segmented.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "../convolution_3d_backprop/polyphase_impl/conv_bp_input_sub_func_polyphase_transpose_phased.h"

    template <PolyphaseRole role>
    __aicore__ inline void ProcessRole()
    {
        switch (static_cast<PolyphaseMode>(tiling_->polyphaseMode)) {
            case PolyphaseMode::PHASED:
                DispatchPhased<role>();
                break;
            case PolyphaseMode::DIRECT_DENSE:
                if (tiling_->polyphaseDualAiv != 0) {
                    ProcessDirectDense<role, true>();
                } else {
                    ProcessDirectDense<role, false>();
                }
                break;
            default:
                break;
        }
    }
#endif

    const Conv3DBackpropInputArch35TilingData* tiling_ = nullptr;
    GlobalTensor<T> filterGm_;
    GlobalTensor<T> dedyGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> biasGm_;
    PolyphaseMemoryLayout memoryLayout_;
};

} // namespace AscendC

#endif // CONV3D_DX_POLYPHASE_FUNC_H
