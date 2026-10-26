/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_OP_GRAPH_ARCH_UTILS_H_
#define OPS_NN_OP_GRAPH_ARCH_UTILS_H_

#include <set>
#include <string>

#include "error_util.h"
#include "platform/platform_info.h"

namespace ops {

/**
 * @brief Check whether the current NpuArch is supported.
 *
 * @param pass_name Fusion pass name used in diagnostic logs.
 * @param supported_archs Supported NpuArch values.
 * @return true if the current NpuArch is supported, false otherwise or on failure.
 */
inline bool CheckNpuArch(const std::string& pass_name, const std::set<std::string>& supported_archs)
{
    fe::PlatFormInfos platform_info;
    fe::OptionalInfos optional_info;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_D(pass_name.c_str(), "Failed to get platform info.");
        return false;
    }
    std::string arch;
    if (!platform_info.GetPlatformRes("version", "NpuArch", arch)) {
        OPS_LOG_D(pass_name.c_str(), "NpuArch is unavailable.");
        return false;
    }
    if (supported_archs.count(arch) == 0U) {
        OPS_LOG_D(pass_name.c_str(), "Unsupported NpuArch=%s.", arch.c_str());
        return false;
    }
    return true;
}

} // namespace ops

#endif // OPS_NN_OP_GRAPH_ARCH_UTILS_H_
