# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# tiling UT 平台桩固定为 Ascend950（UB 253952 / 56 核 / NpuArch 3510），算法对 arch35 各 soc 通用；
# 门控仅需保证当前构建 soc 属于本算子注册的计算单元
if((UT_TEST_ALL OR OP_HOST_UT)
   AND (ASCEND_COMPUTE_UNIT MATCHES "ascend950"
        OR ASCEND_COMPUTE_UNIT MATCHES "ascend960dt"
        OR ASCEND_COMPUTE_UNIT MATCHES "ascend960pr"
        OR ASCEND_COMPUTE_UNIT MATCHES "ascend350"))
    add_modules_ut_sources(HOSTNAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
