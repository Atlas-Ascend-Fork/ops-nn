/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FE_FUSION_PASS_REGISTRY_UT_STUB_H_
#define FE_FUSION_PASS_REGISTRY_UT_STUB_H_

#include <cstdint>
#include <string>
#include <vector>

#include "fe_fusion_pass_registry_stub.h"

// 桩登记 UT 的注册记录接口：fe_fusion_pass_registry_stub.cpp（UT stub 实现）记录每次
// FusionPassRegistrar 构造，测试经 GetRecords() 断言登记表基线与门控行为
namespace ut_stub_fe {

struct RegRecord {
    fe::GraphFusionPassType type;
    std::string name;
    fe::GraphPass* (*create_fn)();
    fe::PassAttr attr;
};

std::vector<RegRecord>& GetRecords();

} // namespace ut_stub_fe

#endif // FE_FUSION_PASS_REGISTRY_UT_STUB_H_
