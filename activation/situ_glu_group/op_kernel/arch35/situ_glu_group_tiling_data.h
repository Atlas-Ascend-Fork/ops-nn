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
 * \file situ_glu_group_tiling_data.h
 * \brief Device-side mirror of optiling::SituGluGroupTilingData.
 *
 * The field order and types must match op_host/arch35/situ_glu_group_tiling_arch35.h exactly: the
 * host fills this struct and the kernel reads it through GET_TILING_DATA_WITH_STRUCT.
 */
#ifndef SITU_GLU_GROUP_TILING_DATA_H_
#define SITU_GLU_GROUP_TILING_DATA_H_

#include <cstdint>

struct SituGluGroupTilingData {
    int64_t bs;
    int64_t d;
    int64_t splitD;
    int64_t rowFactor;
    int64_t dLoop;
    int64_t dFactor;
    int64_t tailDFactor;
    float beta;
    float linearBeta;
    int64_t activateLeft;
    int64_t ubSize;
    int64_t groupIndexLoop;
    int64_t groupIndexFactor;
    int64_t tailGroupIndexFactor;
    int64_t groupIndexElemNum;
    int64_t hasWeight;
    int64_t hasGroupIndex;
    int64_t groupListType;
};

#endif // SITU_GLU_GROUP_TILING_DATA_H_
