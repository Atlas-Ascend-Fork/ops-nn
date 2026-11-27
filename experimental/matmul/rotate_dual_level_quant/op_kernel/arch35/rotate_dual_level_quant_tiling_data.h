/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>

namespace RotateDualLevelQuantNS {
constexpr uint32_t LEVEL0_BLOCK_SIZE = 512;
constexpr uint32_t LEVEL1_BLOCK_SIZE = 32;
constexpr uint32_t SCALE_PAIR_SIZE = 2;
constexpr uint64_t ORDINARY_TILING_KEY = 0;
constexpr uint64_t GROUPED_TILING_KEY = 1;
constexpr uint32_t CUBE_ALIGNMENT = 16;
constexpr uint32_t MAX_ROWS_PER_AIV = 32;
constexpr uint32_t AIV_NUM_RATIO = 2;
constexpr uint32_t ROTATION_BUFFER_COUNT = 2;
constexpr uint32_t MX_WORKSPACE_BYTES = 54 * 1024;
constexpr uint32_t UB_ALIGNMENT_BYTES = 32;
constexpr uint32_t FIXPIPE_ROW_PITCH_GRAN = UB_ALIGNMENT_BYTES / sizeof(uint16_t);
constexpr uint32_t FIXPIPE_READ_GUARD_BYTES = 512;
constexpr uint32_t COLLECT_READ_GUARD_BYTES = 512;
constexpr uint32_t LEVEL0_SCALE_ROW_BYTES = 32;
constexpr uint64_t SYSTEM_WORKSPACE_BYTES = 16UL * 1024UL * 1024UL;

// Host/device contract for the ops-tensor components. A complete task owns
// mL1 original rows and one 512-wide column block; each Cube micro-tile is only rotK wide.
#pragma pack(push, 8)
struct alignas(8) RotateDualLevelQuantTilingData {
    uint32_t M = 0; // Product of input dimensions preceding the last dimension.
    uint32_t N = 0;
    uint32_t rotK = 0;
    uint32_t rotationBlocksPerRow = 0;     // N / rotK.
    uint32_t rotationMatricesPerGroup = 1; // Shared rotation or N/rotK rotations.
    uint32_t level0BlockSize = LEVEL0_BLOCK_SIZE;
    uint32_t level1BlockSize = LEVEL1_BLOCK_SIZE;
    uint32_t level0ScalesPerRow = 0;     // CeilDiv(N, 512).
    uint32_t level1ScalePairsPerRow = 0; // CeilDiv(N, 64).
    uint32_t level1ScalesPerRow = 0;     // Includes the possible padding byte.
    uint32_t lastLevel0BlockSize = 0;
    uint32_t groupMode = 0;
    uint32_t groupNum = 0;
    int32_t groupListType = 0;
    int32_t yDtype = 0;
    int32_t axis = -1;
    int32_t roundMode = 4; // AscendC CAST_RINT=4, CAST_ROUND=0, CAST_FLOOR=1.
    int32_t scaleAlg = 0;
    uint32_t hasAlpha = 0;
    uint32_t usedCoreNum = 0;
    uint32_t aivNumRatio = AIV_NUM_RATIO;
    uint32_t splitM = 1; // Partition original rows, never the folded M*(N/rotK) view.
    uint32_t mL1 = 0;    // Full original-row capacity before Split-M, 16-aligned.
    uint32_t nL1 = 0;
    uint32_t kL1 = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t maxRowsPerAiv = 0; // CeilDiv(mL1, 2); a one-row task uses AIV0 only.
    uint32_t ubDB = 1;          // Number of K-wide Fixpipe receive slots; their bases are 0 and TOTAL_UB_SIZE/2.
    uint32_t ubPitchGran = FIXPIPE_ROW_PITCH_GRAN; // BF16 elements, matching the common Block row pitch.
    uint32_t fixpipeReceiveBytes = 0;              // Includes a readable guard; also the fixed private workspace base.
    uint32_t quantWorkspaceBytes = 0;              // One private workspace, independent of the receive slot index.
    // All offsets below are relative to fixpipeReceiveBytes in each AIV.
    uint32_t mxWorkspaceBytes = MX_WORKSPACE_BYTES; // Includes the in-place BF16 normalization input.
    uint32_t collectUbOffset = 0;
    uint32_t collectUbBytes = 0; // [rotation slice][maxRowsPerAiv][rotK], BF16, plus readable guard.
    uint32_t level0ScaleUbOffset = 0;
    uint32_t level0ScaleUbBytes = 0; // One FP32 value per 32-byte UB row.
    uint32_t totalUbBytes = 0;       // Highest reserved UB end address, including any gap before receive slot 1.
    float level0MaxValue = 6.0f;     // Actual destination maximum: 6, 448 or 57344.
    float level0InvMaxValue = 1.0f / 6.0f;
    float dstTypeMax = 0.0f; // Existing V2 attribute; applies ONLY to level1 quantization.
    float invDstTypeMax = 0.0f;
    uint64_t logicalBlockCount = 0; // Per-row level0 blocks M*CeilDiv(N,512), not the task count.
    uint64_t taskCount = 0;         // Exact ordinary count; grouped count is resolved from device groupList.
    uint64_t userWorkspaceSize = 0; // Intermediate results stay in UB; system workspace is separate.
};
#pragma pack(pop)

static_assert(sizeof(RotateDualLevelQuantTilingData) % 8 == 0, "Tiling data must be 8-byte aligned");
} // namespace RotateDualLevelQuantNS
