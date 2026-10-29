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
 * \file swiglu_hi_f4_quant_tiling_data.h
 * \brief Tiling data structure for SwiGLU + HiFloat4 quantization
 */

#ifndef SWIGLU_HI_F4_QUANT_TILING_DATA_H_
#define SWIGLU_HI_F4_QUANT_TILING_DATA_H_

struct SwigluHiF4QuantTilingData {
    // Basic parameters
    int64_t usedCoreNum; // Actual number of cores used

    // Data shape information
    int64_t inputDimNeg2;   // Input x -2 axis dimension (after merge)      inputDimY
    int64_t inputDimNeg1;   // Input x -1 axis dimension (after merge)      inputDimX
    int64_t outputYDimNeg1; // Output y -1 axis dimension                   outputDimX
    int64_t groupIndexNum;
    // Memory allocation parameters
    int64_t basicDimNeg1;         // Basic block -1 axis element count         basicX
    int64_t basicDimNeg2;         // Basic block -2 axis element count         basicY
    int64_t maxBasicNumUbDimNeg1; // Number of basic blocks in -1 axis per UB iteration   maxBasicNumUbDimX
    int64_t maxBasicNumUbDimNeg2; // Number of basic blocks in -2 axis per UB iteration   maxBasicNumUbDimY
    int64_t ubLoopPerRow;         // UB loops per row for non-full-load scenario
    int64_t ubTailPerRow;         // Tail elements per row for non-full-load scenario

    // Inter-core split parameters
    int64_t frontCoreNum;              // Front cores: number of cores that process one extra block
    int64_t frontCoreBasicNumDimNeg2;  // Front cores: basic blocks in -2 axis           frontCoreBasicNumDimY
    int64_t frontCoreLoopTimes;        // Front cores: loop times
    int64_t frontCoreLastLoopBasicNum; // Front cores: basic blocks in last loop
    int64_t tailCoreBasicNumDimNeg2;   // Tail core: basic blocks in -2 axis             tailCoreBasicNumDimY
    int64_t tailCoreLoopTimes;         // Tail core: loop times
    int64_t tailCoreLastLoopBasicNum;  // Tail core: basic blocks in last loop

    // SwiGLU related parameters (reserved, using float type as per README)
    int64_t swigluMode;   // SwiGLU or variant mode (reserved)
    int64_t activateLeft; // SwiGLU left/right activation (reserved)
    float clampLimit;
    float gluAlpha;
    float gluBias;
    int64_t scaleAlg; // 0=mse_first(有偏): Eb/Ec阈值4/2; 1=unbiased(无偏): Eb/Ec阈值3.75/1.875
};

#endif // SWIGLU_HI_F4_QUANT_TILING_DATA_H_
