# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 仅收集本目录顶层源文件；子目录 cube_utils/fusion_pass 由各自 CMakeLists 独立编入，避免源文件重复编译
file(GLOB OP_GRAPH_COMMON_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
if(OP_GRAPH_COMMON_SRCS)
    add_graph_plugin_modules()
    target_include_directories(${GRAPH_PLUGIN_NAME}_obj PRIVATE
        ${OPS_NN_DIR}/common/stub/inc/op_graph
    )
    # 仅编译 soc 含 ascend950 时注入平台宏，桩注册内容才编入
    if("ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
        set_source_files_properties(${OP_GRAPH_COMMON_SRCS} PROPERTIES
            COMPILE_DEFINITIONS OP_GRAPH_NN_ASCEND950=1
        )
    endif()
    target_sources(${GRAPH_PLUGIN_NAME}_obj PRIVATE ${OP_GRAPH_COMMON_SRCS})
endif()
