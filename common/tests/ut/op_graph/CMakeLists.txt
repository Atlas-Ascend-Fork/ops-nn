# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_GRAPH_UT)
    if(NOT TARGET ${OP_GRAPH_MODULE_NAME}_cases_obj)
        add_op_graph_ut_modules(${OP_GRAPH_MODULE_NAME})
    endif()
    target_sources(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test_bias_optimize_quant_rollback_base.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fixpipe_ability_process_pass.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fixpipe_fusion_pass.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fixpipe_pre_quant_pass.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fixpipe_relu_quant_pass.cpp)
    # fe 融合 pass 桩登记基线 UT：注册代码仅 ascend950 构建经 graph_plugin 对象编入（与产品门控一致），
    # 非 950 构建无注册代码，基线 UT 同步不编入；stub 强符号覆盖 libregister 同名实现
    if("ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
        target_include_directories(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
            ${PROJECT_SOURCE_DIR}/common/stub/inc/op_graph)
        target_sources(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
            ${PROJECT_SOURCE_DIR}/tests/ut/common/fe_fusion_pass_registry_stub.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test_legacy_fusion_pass_stub_registration.cpp)
    endif()
endif()
