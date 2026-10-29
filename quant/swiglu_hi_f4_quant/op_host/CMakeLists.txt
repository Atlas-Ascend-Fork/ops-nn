# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 算子输出 y/scale 为 HIFLOAT4/HIFLOAT4_SCALE，只有 CANN >= 9.3.0 的 opbuild 支持；
# 低版本工具链不编译本算子，否则 opbuild 校验会报
# "Dtype of y of op SwigluHiF4Quant is not support."。
# 同样的判断在 op_graph/、tests/ 的 CMakeLists.txt 中各有一份，修改时需同步。
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
if(NOT SWIGLU_HI_F4_QUANT_CANN_VERSION)
    message(WARNING "swiglu_hi_f4_quant: CANN version not detected, build it anyway.")
endif()
if(SWIGLU_HI_F4_QUANT_CANN_VERSION AND
   SWIGLU_HI_F4_QUANT_CANN_VERSION VERSION_LESS SWIGLU_HI_F4_QUANT_MIN_CANN_VERSION)
    message(STATUS "swiglu_hi_f4_quant: CANN ${SWIGLU_HI_F4_QUANT_CANN_VERSION} < "
                   "${SWIGLU_HI_F4_QUANT_MIN_CANN_VERSION}, skip building this operator.")
    # 跳过编译后算子不会写入 ops_config.txt，而 CI 会按变更文件从该文件反查算子依赖
    # （scripts/util/dependency_parser.py，查不到会直接报错退出）。
    # 这里只补一条依赖图记录，不添加任何编译目标。
    file(APPEND ${ASCEND_SUB_CONFIG_PATH}
         "OP_CATEGORY;${op_category};OP_NAME;swiglu_hi_f4_quant;OPTYPE;swiglu_hi_f4_quant;ACLNNTYPE;aclnn\n")
    return()
endif()

get_filename_component(OP_ROOT ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${OP_ROOT}
                    OPTYPE swiglu_hi_f4_quant ACLNNTYPE aclnn DISABLE_IN_OPP TRUE)
