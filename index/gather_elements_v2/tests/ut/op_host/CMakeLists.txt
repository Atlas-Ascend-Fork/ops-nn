# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(HOSTNAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    add_modules_ut_sources(HOSTNAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    # Load the error registration owner only when these host tests are selected.
    target_link_libraries(${OP_TILING_MODULE_NAME}_static_lib INTERFACE
        -Wl,--push-state,--no-as-needed nnopbase -Wl,--pop-state)
endif()
if(UT_TEST_ALL OR OP_API_UT)
    add_modules_ut_sources(HOSTNAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    # Keep error-manager assertions separate from the shared API test executable.
    set(GATHER_ELEMENTS_V2_ERROR_UT gather_elements_v2_error_code_ut)
    add_executable(${GATHER_ELEMENTS_V2_ERROR_UT}
        ${CMAKE_CURRENT_SOURCE_DIR}/gather_elements_v2_error_code_ut.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/op_api/gather_elements_v2.cpp
        ${UT_PATH}/op_api/stub/opdev/platform.cpp
        ${UT_PATH}/op_api/stub/opdev/op_executor.cpp)
    target_include_directories(${GATHER_ELEMENTS_V2_ERROR_UT} PRIVATE
        ${JSON_INCLUDE}
        ${OPS_NN_COMMON_INC}
        ${UT_PATH}/op_api/stub
        ${ASCEND_DIR}/include
        ${ASCEND_DIR}/pkg_inc
        ${ASCEND_DIR}/include/aclnn
        ${ASCEND_DIR}/include/aclnnop
        ${OPAPI_INCLUDE})
    target_compile_definitions(${GATHER_ELEMENTS_V2_ERROR_UT} PRIVATE LOG_CPP)
    target_compile_options(${GATHER_ELEMENTS_V2_ERROR_UT} PRIVATE ${UT_DEBUG_FLAG} -fno-access-control)
    target_link_directories(${GATHER_ELEMENTS_V2_ERROR_UT} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)
    target_link_libraries(${GATHER_ELEMENTS_V2_ERROR_UT} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        $<BUILD_INTERFACE:dlog_headers>
        -Wl,--push-state,--no-as-needed
        nnopbase
        error_manager
        -Wl,--pop-state
        graph
        gtest
        platform
        register
        opp_registry
        rt2_registry)
    add_dependencies(${GATHER_ELEMENTS_V2_ERROR_UT} json)
    # The usual nn_op_api_ut build entry must also build these isolated cases.
    # The isolated target does not consume the shared cases object or executable.
    add_dependencies(${OP_API_MODULE_NAME}_cases_obj ${GATHER_ELEMENTS_V2_ERROR_UT})

    if(ENABLE_UT_EXEC)
        set(GATHER_ELEMENTS_V2_ERROR_UT_ENV "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}")
        if(ENABLE_ASAN)
            execute_process(
                COMMAND ${CMAKE_C_COMPILER} -print-file-name=libasan.so
                OUTPUT_VARIABLE GATHER_ELEMENTS_V2_ERROR_UT_LIBASAN
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE GATHER_ELEMENTS_V2_ERROR_UT_ASAN_RESULT)
            execute_process(
                COMMAND ${CMAKE_C_COMPILER} -print-file-name=libstdc++.so
                OUTPUT_VARIABLE GATHER_ELEMENTS_V2_ERROR_UT_LIBSTDC
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE GATHER_ELEMENTS_V2_ERROR_UT_STDC_RESULT)
            if(NOT GATHER_ELEMENTS_V2_ERROR_UT_ASAN_RESULT EQUAL 0 OR
               NOT GATHER_ELEMENTS_V2_ERROR_UT_STDC_RESULT EQUAL 0)
                message(FATAL_ERROR "Cannot find compiler ASAN runtimes for GatherElementsV2 error-code UT")
            endif()
            list(APPEND GATHER_ELEMENTS_V2_ERROR_UT_ENV
                "LD_PRELOAD=${GATHER_ELEMENTS_V2_ERROR_UT_LIBASAN}:${GATHER_ELEMENTS_V2_ERROR_UT_LIBSTDC}"
                "ASAN_OPTIONS=detect_leaks=0")
        endif()
        add_custom_command(TARGET ${GATHER_ELEMENTS_V2_ERROR_UT} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E env ${GATHER_ELEMENTS_V2_ERROR_UT_ENV}
                $<TARGET_FILE:${GATHER_ELEMENTS_V2_ERROR_UT}>
            COMMENT "Run GatherElementsV2 error-code UT"
            VERBATIM)
    endif()
endif()
