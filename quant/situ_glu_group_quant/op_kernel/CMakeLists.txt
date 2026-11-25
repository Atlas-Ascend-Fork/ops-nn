# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 算子 kernel 配置项（wiki: 算子kernel配置项指南）：
# KERNEL_SRC 缺省为 {ops}.cpp，此处显式声明入口；
# COMPUTE_UNITS 显式声明本算子支持 ascend950/ascend350/ascend960dt。
# 注意：ascend950 与 ascend350 同属 dav_3510，同一 arch 下声明形式须统一
# （逐个声明 soc，不能与 npuArch 形式混用），故这里全部采用 soc 形式。
add_kernel_sources(
    KERNEL_SRC situ_glu_group_quant.cpp
    COMPUTE_UNITS ascend950 ascend350 ascend960dt
)
