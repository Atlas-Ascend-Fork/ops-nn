# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# MaxPoolV3Grad NHWC 快速通道 kernel 精度用例(三分支: 大C/小C小kernel/小C大kernel, 含 pad 重叠/多 tile 过滤)。
# 当前环境 tikicpulib CPU 仿真器无法从测试 TU 执行 Reg:: 向量通路(最小探针亦失败, 与
# max_pool3d_grad 未提供 kernel UT 同因), 暂不启用; 仿真器修复后取消下方注释即可。
# if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
#     AddOpTestCase(max_pool_v3_grad "ascend950pr_9599" "-DDTYPE_ORIG_INPUT=float" pool_grad_common)
# endif()
