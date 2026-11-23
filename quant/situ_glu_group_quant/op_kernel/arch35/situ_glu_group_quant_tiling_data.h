/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file situ_glu_group_quant_tiling_data.h
 * \brief Tiling data structure for SituGluGroupQuant, ported verbatim from
 *        cann-recipes-infer (field names and types kept so the ported kernel
 *        body can reference tiling_->xxx verbatim).
 */

#ifndef SITU_GLU_GROUP_QUANT_TILING_DATA_H
#define SITU_GLU_GROUP_QUANT_TILING_DATA_H

#include <cstdint>

struct SituGluGroupQuantTilingData {
    // Field order follows the tiling-data convention: int64 first, then int32,
    // then float (no type mixing). Names and types are unchanged — the kernel
    // body and the host tiling reference fields by name, and the UT kernel test
    // includes this header directly, so the layout stays in sync by construction.
    // --- int64 ---
    int64_t totalElements;        // total activation elements = totalRows * EPR
    int64_t outputElementsPerRow; // EPR = x_last / 2
    int64_t groupIndexCount;      // number of experts (group_index length)
    int64_t fullRowsMode;         // [host-diag-only] 1 = group_index absent: all rows valid; 0 = grouped
    int64_t chunkElements;        // min(EPR, MAX_TILE_COLS) = whole-row chunk
    int64_t scaleColNum;          // e8m0 scales per row = ceil(EPR/32) rounded up to even
    int64_t rowsPerBatch;         // K-row batching (perf dim, NOT a tiling-key dim): rows per
                                  // copy+VF batch in the kernel; 1 = legacy per-row cycle.
                                  // Host guarantees 1 for hasWeight (per-row wVal scalar in
                                  // the activation VF), for odd groupCount (scaleColNum !=
                                  // groupCount: per-row GM scale padding breaks flat batch
                                  // copies), and for EPR > K_BATCH_EPR_MAX.
    int64_t activateLeft;         // [host-diag-only] 1 = gate first half (recipes); 0 = gate second half (situ normal)
    int64_t hasWeight;            // [host-diag-only] 1 = optional weight input present (per-token fp32 scale)
    // --- int32 ---
    int32_t highPrecision; // [host-diag-only] recipes high_precision round-trip switch
    int32_t blockSize;     // [host-diag-only] group_size, fixed 32
    // --- float ---
    float beta;
    float linearBeta;
    // [host-diag-only] fields: the kernel no longer reads them — presence /
    // direction / precision are compile-time via the TPL tiling-key dimensions.
    // The host still fills them for logging and UT asserts.
};
#endif // SITU_GLU_GROUP_QUANT_TILING_DATA_H
