# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Match the typed pipeline without an externally injected version macro.
AddOpTestCase(rotate_dual_level_quant "ascend950pr_9599"
    "-DDTYPE_X=bfloat16_t -DDTYPE_Y=fp4x2_e2m1_t -DDTYPE_LEVEL0_SCALE=float -DDTYPE_LEVEL1_SCALE=fp8_e8m0_t -DENABLE_CV_COMM_VIA_SSBUF=true"
    UT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/arch35)
