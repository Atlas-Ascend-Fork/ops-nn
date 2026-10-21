/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dynamic_hi_f4_quant_struct.h
 * \brief Template argument definitions for quantAxis selection (tail axis vs non-tail axis)
 */
#ifndef DYNAMIC_HIF4_QUANT_STRUCT_H_
#define DYNAMIC_HIF4_QUANT_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"
#define TPL_QUANT_TAIL_AXIS 0
#define TPL_QUANT_NO_TAIL_AXIS 1
namespace DynamicHiF4QuantOp {
ASCENDC_TPL_ARGS_DECL(DynamicHiF4Quant, ASCENDC_TPL_UINT_DECL(quantAxis, 1, ASCENDC_TPL_UI_LIST, TPL_QUANT_TAIL_AXIS,
                                                              TPL_QUANT_NO_TAIL_AXIS));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(quantAxis, ASCENDC_TPL_UI_LIST, TPL_QUANT_TAIL_AXIS,
                                                          TPL_QUANT_NO_TAIL_AXIS)));

} // namespace DynamicHiF4QuantOp

#endif // DYNAMIC_HIF4_QUANT_STRUCT_H_
