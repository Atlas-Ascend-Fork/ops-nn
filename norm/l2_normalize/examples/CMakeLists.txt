# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# This standalone example must not add installed-CANN dependencies to the host build.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    return()
endif()

cmake_minimum_required(VERSION 3.18)
project(GE_IR_EXAMPLE LANGUAGES CXX)

set(ASCEND_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_PATH)
    message(FATAL_ERROR "Source the CANN set_env.sh before configuring this example.")
endif()
find_library(GRAPH_LIBRARY NAMES libgraph.so PATHS "${ASCEND_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)
find_library(GE_RUNNER_LIBRARY NAMES libge_runner.so PATHS "${ASCEND_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)
find_library(GRAPH_BASE_LIBRARY NAMES libgraph_base.so PATHS "${ASCEND_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)
find_library(GE_COMPILER_LIBRARY NAMES libge_compiler.so PATHS "${ASCEND_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)

add_executable(test_geir_l2_normalize test_geir_l2_normalize.cpp)
target_include_directories(test_geir_l2_normalize PRIVATE
    "${ASCEND_PATH}/include/graph"
    "${ASCEND_PATH}/include/ge"
    "${ASCEND_PATH}/opp/built-in/op_proto/inc"
    "${ASCEND_PATH}/include")
target_link_libraries(test_geir_l2_normalize PRIVATE
    ${GRAPH_LIBRARY} ${GE_RUNNER_LIBRARY} ${GRAPH_BASE_LIBRARY} ${GE_COMPILER_LIBRARY})
