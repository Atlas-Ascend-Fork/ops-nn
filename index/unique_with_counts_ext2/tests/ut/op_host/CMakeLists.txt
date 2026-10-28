#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(HOSTNAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    add_modules_ut_sources(HOSTNAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
if(UT_TEST_ALL OR OP_API_UT)
    add_modules_ut_sources(HOSTNAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})

    set(UNIQUE_DIM_ERROR_CODE_UT unique_dim_error_code_ut)
    add_executable(${UNIQUE_DIM_ERROR_CODE_UT}
        ${CMAKE_CURRENT_SOURCE_DIR}/unique_dim_error_code_ut.cpp
        $<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>
        ${UT_PATH}/op_api/stub/opdev/platform.cpp
        ${UT_PATH}/op_api/stub/opdev/op_executor.cpp)
    target_include_directories(${UNIQUE_DIM_ERROR_CODE_UT} PRIVATE
        ${JSON_INCLUDE}
        ${ASCEND_DIR}/include
        ${ASCEND_DIR}/pkg_inc
        ${ASCEND_DIR}/include/aclnn
        ${ASCEND_DIR}/include/aclnnop
        ${OPS_NN_COMMON_INC}
        ${UT_PATH}/op_api/stub
        ${OPAPI_INCLUDE})
    target_compile_definitions(${UNIQUE_DIM_ERROR_CODE_UT} PRIVATE LOG_CPP)
    target_compile_options(${UNIQUE_DIM_ERROR_CODE_UT} PRIVATE ${UT_DEBUG_FLAG} -fno-access-control)
    target_link_directories(${UNIQUE_DIM_ERROR_CODE_UT} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)
    target_link_libraries(${UNIQUE_DIM_ERROR_CODE_UT} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        $<BUILD_INTERFACE:dlog_headers>
        -Wl,--push-state,--no-as-needed
        nnopbase
        error_manager
        -Wl,--pop-state
        opapi_math_stub
        graph
        c_sec
        unified_dlog
        gtest
        platform
        register
        opp_registry
        rt2_registry)
    add_dependencies(${UNIQUE_DIM_ERROR_CODE_UT} json)
    add_dependencies(${OP_API_MODULE_NAME}_cases_obj ${UNIQUE_DIM_ERROR_CODE_UT})

    if(ENABLE_UT_EXEC)
        set(UNIQUE_DIM_ERROR_CODE_UT_ENV
            "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}"
            "LD_PRELOAD=${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64/libopapi_math.so")
        if(ENABLE_ASAN)
            execute_process(
                COMMAND ${CMAKE_C_COMPILER} -print-file-name=libasan.so
                OUTPUT_VARIABLE UNIQUE_DIM_ERROR_CODE_UT_LIBASAN
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE UNIQUE_DIM_ERROR_CODE_UT_ASAN_RESULT)
            execute_process(
                COMMAND ${CMAKE_C_COMPILER} -print-file-name=libstdc++.so
                OUTPUT_VARIABLE UNIQUE_DIM_ERROR_CODE_UT_LIBSTDC
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE UNIQUE_DIM_ERROR_CODE_UT_STDC_RESULT)
            if(NOT UNIQUE_DIM_ERROR_CODE_UT_ASAN_RESULT EQUAL 0 OR
               NOT UNIQUE_DIM_ERROR_CODE_UT_STDC_RESULT EQUAL 0)
                message(FATAL_ERROR "Cannot find compiler ASAN runtimes for UniqueDim error-code UT")
            endif()
            list(REMOVE_ITEM UNIQUE_DIM_ERROR_CODE_UT_ENV
                "LD_PRELOAD=${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64/libopapi_math.so")
            list(APPEND UNIQUE_DIM_ERROR_CODE_UT_ENV
                "LD_PRELOAD=${UNIQUE_DIM_ERROR_CODE_UT_LIBASAN}:${UNIQUE_DIM_ERROR_CODE_UT_LIBSTDC}:${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64/libopapi_math.so"
                "ASAN_OPTIONS=detect_leaks=0")
        endif()
        add_custom_command(TARGET ${UNIQUE_DIM_ERROR_CODE_UT} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E env ${UNIQUE_DIM_ERROR_CODE_UT_ENV}
                $<TARGET_FILE:${UNIQUE_DIM_ERROR_CODE_UT}>
            COMMENT "Run UniqueDim error-code UT"
            VERBATIM)
    endif()
endif()
