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
 * \file bn3_d_training_update_tiling_key.h
 * \brief BN3DTrainingUpdate arch35 TilingKey。
 */

#ifndef BN3_D_TRAINING_UPDATE_TILING_KEY_H
#define BN3_D_TRAINING_UPDATE_TILING_KEY_H

// x 的 format 为 NCHW / NCDHW(通道轴 = 1)。
#define BN3D_TRAINING_UPDATE_KEY_CHANNEL_FIRST 2000
// x 的 format 为 NHWC / NDHWC(通道轴 = 末轴)。
#define BN3D_TRAINING_UPDATE_KEY_CHANNEL_LAST 3000
// 周期展平路径:通道数 C <= 2 且通道周期可整除展开块(C 与上两条路径按 TilingKey 严格隔离)。
// 覆盖 channel-last(C = 1/2)与 channel-first(空间块 S 极小、周期 P = C·S 短)两类长尾形状。
#define BN3D_TRAINING_UPDATE_KEY_FLAT 4000

#endif // BN3_D_TRAINING_UPDATE_TILING_KEY_H
