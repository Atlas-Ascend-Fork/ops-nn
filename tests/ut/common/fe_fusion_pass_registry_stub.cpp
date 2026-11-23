/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fe_fusion_pass_registry_stub.cpp
 * \brief 桩登记 UT 的 fe 注册接口 stub：强符号覆盖 libregister.so 同名实现，记录构造调用，
 *        并提供环境变量可控的 aclsysGetVersionNum 强符号（覆盖桩头弱声明），用于门控分层验证。
 */

#include "fe_fusion_pass_registry_ut_stub.h"

#include <cstdlib>
#include <map>

namespace fe {

FusionPassRegistrar::FusionPassRegistrar(const GraphFusionPassType& pass_type, const std::string& pass_name,
                                         GraphPass* (*create_fn)(), PassAttr attr)
{
    ut_stub_fe::GetRecords().push_back(ut_stub_fe::RegRecord{pass_type, pass_name, create_fn, attr});
}

FusionPassRegistry& FusionPassRegistry::GetInstance()
{
    static FusionPassRegistry instance;
    return instance;
}

std::map<std::string, PassDesc> FusionPassRegistry::GetPassDesc(const GraphFusionPassType& pass_type)
{
    std::map<std::string, PassDesc> result;
    for (const auto& record : ut_stub_fe::GetRecords()) {
        if (record.type == pass_type) {
            PassDesc desc{};
            desc.attr = record.attr;
            desc.create_fn = record.create_fn;
            result[record.name] = desc;
        }
    }
    return result;
}

} // namespace fe

// 强符号覆盖桩头的弱符号声明：环境变量控制版本号/查询失败，供门控分层用例（exec 子进程）注入场景
//   LEGACY_STUB_UT_GE_VER      返回的 ge-compiler 版本号，默认 90300000（全部门槛满足）
//   LEGACY_STUB_UT_QUERY_FAIL  置 1 时模拟查询失败（返回非 ACL_SUCCESS），桩按不满足处理、不登记
extern "C" int32_t aclsysGetVersionNum(char* pkgName, int32_t* versionNum)
{
    (void)pkgName;
    if (versionNum == nullptr) {
        return -1;
    }
    *versionNum = 0;
    const char* queryFail = std::getenv("LEGACY_STUB_UT_QUERY_FAIL");
    if (queryFail != nullptr && queryFail[0] == '1') {
        return -1;
    }
    const char* version = std::getenv("LEGACY_STUB_UT_GE_VER");
    *versionNum = (version != nullptr) ? static_cast<int32_t>(std::atoi(version)) : 90300000;
    return 0;
}

// 强符号定义 OriginDataType 探测符号（桩头弱声明处带 asm 别名，定义自动继承），UT 内 API 恒可用
extern "C" void stub_GetOriginDataType() {}

extern "C" void stub_SetOriginDataType() {}
