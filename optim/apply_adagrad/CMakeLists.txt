# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# GEIR 原型壳算子：仅注册算子IR（op_graph），无 op_host/op_kernel 实现；
# GEIR 图中的 ApplyAdagrad 节点由 ApplyAdagradToInplaceFusionPass 改写为
# InplaceApplyAdagrad 执行，仅面向 A5（Ascend950）GEIR 通路，不适配 960。
set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE apply_adagrad ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR} DISABLE_IN_OPP TRUE DEPENDENCIES inplace_apply_adagrad)
