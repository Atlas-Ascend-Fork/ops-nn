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
 * \file dynamic_block_hi_f4_quant_struct.h
 * \brief Template argument definitions for round_mode selection
 * \note round_mode 仅支持 "rint" 与 "trunc" 两种取整模式
 */
#ifndef DYNAMIC_BLOCK_HI_F4_QUANT_STRUCT_H_
#define DYNAMIC_BLOCK_HI_F4_QUANT_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_RINT 4
#define TPL_TRUNC 3

namespace DynamicBlockHiF4QuantOp {
ASCENDC_TPL_ARGS_DECL(DynamicBlockHiF4Quant,
                      ASCENDC_TPL_UINT_DECL(roundMode, 2, ASCENDC_TPL_UI_LIST, TPL_RINT, TPL_TRUNC));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(roundMode, ASCENDC_TPL_UI_LIST, TPL_RINT, TPL_TRUNC)));

} // namespace DynamicBlockHiF4QuantOp

#endif // DYNAMIC_BLOCK_HI_F4_QUANT_STRUCT_H_
