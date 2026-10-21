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
 * \file anti_hi_f4_quant_common.h
 * \brief
 */

#ifndef OPS_NN_ANTI_HI_F4_QUANT_COMMON_H
#define OPS_NN_ANTI_HI_F4_QUANT_COMMON_H
namespace AntiHiF4Quant {
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t BLOCK_SIZE = 64;

// -2 轴
constexpr int64_t NUM_TWO = 2;
constexpr int64_t BYTE_OF_INPUT_SCALE = 4;
constexpr int64_t BYTE_OF_OUTPUT_Y = 2;
constexpr int64_t X_NUM_OF_ONE_BYTE = 2;
constexpr int64_t BF16_BIAS = 127;
constexpr int16_t BF16_SHAFT_LEFT_NUM = 7;
constexpr int16_t SHIFT_RIGHT_NUM = 1;
constexpr int64_t UB_BLOCK_SIZE = 32;
// -1 轴
constexpr int64_t DQ_HIFP4_GROUP_BYTE_NUM = 32; // 64 * 0.5 byte = 32 字节（x: fp4x2 packed）
constexpr int64_t DQ_HIFP4_SCALE_BYTE_NUM = 4;  // 每 group 4 字节 scale (Ea+Eb+Ec packed)
constexpr int64_t DQ_HIFP4_SCALE_FP32_NUM = 1;  // scale 按 float 存储 = 4 bytes = 1 个 float
constexpr int64_t DQ_HIFP4_BF16_PER_GROUP = 64; // 反量化后每 group 64 个 BF16 元素
constexpr int64_t DQ_HIFP4_SCALE_ALIGN = 32;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DIGIT_EIGHT = 8;

static constexpr MicroAPI::CastTrait castTraitE6M2ToBF16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                            MicroAPI::MaskMergeMode::ZEROING,
                                                            AscendC::RoundMode::UNKNOWN};
static constexpr MicroAPI::CastTrait castTraitB4ToB16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                         MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr MicroAPI::CastTrait castTraitBF16ToFP16 = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                                            MicroAPI::MaskMergeMode::ZEROING,
                                                            AscendC::RoundMode::CAST_RINT};
} // namespace AntiHiF4Quant
#endif
