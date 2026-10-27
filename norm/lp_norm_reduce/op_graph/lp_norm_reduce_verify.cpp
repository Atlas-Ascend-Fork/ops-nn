/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lp_norm_reduce_proto.h"
#include "graph/utils/type_utils.h"
#include "op_common/log/log.h"

namespace ge {
// Verification precedes dtype inference and storage format selection. An
// output's default dtype is not an explicit declaration, and a graph's input
// format need not be the selected kernel's storage format. Validate the input
// dtype here; the selected implementation checks the final descriptors.
IMPLEMT_VERIFIER(LpNormReduce, VerifyLpNormReduce)
{
    const auto x = op.GetInputDesc("x");
    const std::string nodeName = op.GetName();
    if (x.GetDataType() != DT_FLOAT16 && x.GetDataType() != DT_FLOAT && x.GetDataType() != DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName.c_str(), "x", TypeUtils::DataTypeToSerialString(x.GetDataType()),
                                              "x must use float16, float32 or bfloat16");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(LpNormReduce, VerifyLpNormReduce);
} // namespace ge
