# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 与算子根目录 CMakeLists.txt 保持一致：CANN < 9.3.0 不编译本算子，UT 也不再编译。
set(SWIGLU_HI_F4_QUANT_MIN_CANN_VERSION "9.3.0")
set(SWIGLU_HI_F4_QUANT_CANN_VERSION "")
if(EXISTS "${ASCEND_CANN_PACKAGE_PATH}/compiler/version.info")
    file(READ "${ASCEND_CANN_PACKAGE_PATH}/compiler/version.info" SWIGLU_HI_F4_QUANT_VERSION_INFO)
    string(REGEX MATCH "Version=([0-9]+\\.[0-9]+\\.[0-9]+)" SWIGLU_HI_F4_QUANT_VERSION_MATCH
           "${SWIGLU_HI_F4_QUANT_VERSION_INFO}")
    if(SWIGLU_HI_F4_QUANT_VERSION_MATCH)
        set(SWIGLU_HI_F4_QUANT_CANN_VERSION "${CMAKE_MATCH_1}")
    endif()
endif()
if(SWIGLU_HI_F4_QUANT_CANN_VERSION AND
   SWIGLU_HI_F4_QUANT_CANN_VERSION VERSION_LESS SWIGLU_HI_F4_QUANT_MIN_CANN_VERSION)
    message(STATUS "swiglu_hi_f4_quant: CANN ${SWIGLU_HI_F4_QUANT_CANN_VERSION} < "
                   "${SWIGLU_HI_F4_QUANT_MIN_CANN_VERSION}, skip ut.")
    return()
endif()

if(ENABLE_TEST)
    add_subdirectory(ut)
endif()
