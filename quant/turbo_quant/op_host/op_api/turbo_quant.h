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
 * \file turbo_quant.h
 * \brief level0 API of TurboQuant
 */
#ifndef OP_API_TURBO_QUANT_H
#define OP_API_TURBO_QUANT_H

#include "aclnn_kernels/contiguous.h"

namespace l0op {
std::tuple<aclTensor*, aclTensor*> TurboQuant(const aclTensor* latent, const aclTensor* centroids,
                                              aclOpExecutor* executor);
}

#endif // OP_API_TURBO_QUANT_H
