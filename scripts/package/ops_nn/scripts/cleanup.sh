#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# 本脚本经makeself --cleanup参数调用（ops_nn.xml的<cleanup>字段由打包器映射为该参数）：
# .run安装脚本install.sh执行完毕后，或安装过程中收到HUP/INT/QUIT/TERM信号被中断时，
# makeself将CWD置为其运行期解压临时目录（如$TMPROOT/makeself.<rand>.<date>.<pid>）并执行本脚本，
# 用于清理安装过程的临时解压产物。安全防护：仅当目标非空、非根目录、且目录名匹配makeself
# 临时目录命名模式（makeself.*/selfgz*）时才执行删除，防止误在非临时目录（家目录/数据盘等）
# 手动执行造成数据灾难性丢失。

set -e

target_dir="$(pwd -P)"

if [ -z "${target_dir}" ] || [ "${target_dir}" = "/" ]; then
    echo "[ERROR] cleanup.sh: invalid target directory '${target_dir}', abort." >&2
    exit 1
fi
case "$(basename "${target_dir}")" in
    makeself.*|selfgz*) ;;
    *)
        echo "[ERROR] cleanup.sh: current directory '${target_dir}' is not a makeself temporary directory, abort." \
             " This script removes the extraction temporary directory of the .run installer and is meant to be" \
             " invoked by the package manager (makeself --cleanup), not run manually." >&2
        exit 1
        ;;
esac

echo "[INFO] cleanup.sh: removing makeself temporary directory: ${target_dir}"
rm -rf -- "${target_dir}"
