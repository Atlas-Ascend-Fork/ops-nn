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

#include "ascendc/host_api/tiling/template_argument.h"
#include "gemm_symm_tiling_data.h"

#define GEMM_SYMM_TRANS_FALSE 0
#define GEMM_SYMM_TRANS_TRUE 1
#define GEMM_SYMM_HIGH_PREC_FALSE 0
#define GEMM_SYMM_HIGH_PREC_TRUE 1

ASCENDC_TPL_ARGS_DECL(GemmSymm, ASCENDC_TPL_BOOL_DECL(TRANS_A, GEMM_SYMM_TRANS_FALSE, GEMM_SYMM_TRANS_TRUE),
                      ASCENDC_TPL_BOOL_DECL(TRANS_B, GEMM_SYMM_TRANS_FALSE, GEMM_SYMM_TRANS_TRUE),
                      ASCENDC_TPL_BOOL_DECL(HIGH_PREC, GEMM_SYMM_HIGH_PREC_FALSE, GEMM_SYMM_HIGH_PREC_TRUE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_BOOL_SEL(TRANS_A, GEMM_SYMM_TRANS_FALSE, GEMM_SYMM_TRANS_TRUE),
                                     ASCENDC_TPL_BOOL_SEL(TRANS_B, GEMM_SYMM_TRANS_FALSE, GEMM_SYMM_TRANS_TRUE),
                                     ASCENDC_TPL_BOOL_SEL(HIGH_PREC, GEMM_SYMM_HIGH_PREC_FALSE,
                                                          GEMM_SYMM_HIGH_PREC_TRUE),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(GemmSymmTilingData)));
