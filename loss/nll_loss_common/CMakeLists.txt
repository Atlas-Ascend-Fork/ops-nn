# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# nll_loss 与 nll_loss_grad 共享的 op_api 公共代码（target 值域校验）。
# 参照 pool_3d_common：首行不带 OPTYPE（自动取目录名 nll_loss_common，供 DEPENDENCIES 解析），
# 次行 OPTYPE common 注册头文件；vendor 单算子分包时依赖目录源文件一并编入 opapi 库。
set(SUPPORT_COMPUTE_UNIT "dav_3510" "dav_9201" "dav_9202")
set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35")
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE)
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE common ACLNNTYPE aclnn_exclude)
