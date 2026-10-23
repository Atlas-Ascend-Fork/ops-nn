/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../test_add_rms_norm_cast_regbase.h"
#undef DTYPE_X1
#define DTYPE_X1 bfloat16_t
#define ADD_RMS_ALIAS_SUITE AddRmsNormCastRecomputeAliasBf16950
#define ADD_RMS_RECOMPUTE_SUITE AddRmsNormCastRecomputeBf16950
#define add_rms_norm_cast add_rms_norm_cast_recompute_bf16
#include "test_add_rms_norm_cast_apt_spilt_recompute.h"
#undef add_rms_norm_cast
#undef ADD_RMS_RECOMPUTE_SUITE
#undef ADD_RMS_ALIAS_SUITE
