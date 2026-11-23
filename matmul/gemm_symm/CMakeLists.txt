# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_all_modules_sources(
    ACLNNTYPE aclnn_exclude
)

add_kernel_sources(
    KERNEL_SRC arch35/gemm_symm.cpp
    COMPUTE_UNITS ascend950
    AUTO_SYNC false
)

# Include paths contain '/', and gen_ops_compile_ini.py splits OPTIONS on '/'.
# Keep them in custom_compile_options.ini, which uses ',' and ';' as separators.
# add_ops_compile_options() only wraps that write and fails on some internal trees.
file(APPEND "${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}"
    "GemmSymm,,-I${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/arch35;-I${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/arch35/catlass\n"
)
