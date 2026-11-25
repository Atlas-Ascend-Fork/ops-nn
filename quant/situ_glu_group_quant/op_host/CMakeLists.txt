# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# ascend350 与 ascend950 同为 dav_3510（arch35），tiling 直接复用；ascend960dt
# 为 dav_9201，仓库惯例（situ_mx_quant 等 mx quant 族）同样复用 arch35 tiling。
set(SUPPORT_COMPUTE_UNIT "ascend950" "ascend350" "ascend960dt")
set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35")
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE situ_glu_group_quant
                    ACLNNTYPE aclnn_inner COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR}
                    DISABLE_IN_OPP TRUE)
