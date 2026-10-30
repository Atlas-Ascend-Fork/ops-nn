# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# FusedWeightQuantMatMul：WeightQuantBatchMatmulV2 + Relu 融合算子（仅图模式）

add_all_modules_sources(
    ACLNNTYPE aclnn_exclude
    DEPENDENCIES weight_quant_batch_matmul_v2
)

if ((UT_TEST_ALL OR OP_GRAPH_UT) AND TARGET ${OP_GRAPH_MODULE_NAME}_cases_obj)
    target_include_directories(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE ${PROJECT_SOURCE_DIR})
endif()

add_kernel_sources(
    KERNEL_SRC arch35/fused_weight_quant_mat_mul.cpp
    COMPUTE_UNITS mc62
)
