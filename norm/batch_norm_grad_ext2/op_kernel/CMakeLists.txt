# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_kernel_sources(
    KERNEL_SRC arch35/batch_norm_grad_ext2.cpp
    COMPUTE_UNITS dav_3510
    AUTO_SYNC false
    # training 的 dx 改用双 fp32（dd）精确链后寄存器压力上升、栈帧逼近 scbzi 的 [-512, 511]
    # 立即数范围（dd 链只用 4 个 scratch 已把压力压到可编译）。该选项同仓 avg_pool*、
    # avg_pool3_d_grad 也在用；RAFullLoad(20000000) 的全量 dd 仍会越界，故该模板本轮未改。
    OPTIONS "--cce-long-scbz=true"
)
