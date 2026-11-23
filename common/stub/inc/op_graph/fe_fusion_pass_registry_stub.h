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
 * \file fe_fusion_pass_registry_stub.h
 * \brief GE fe 融合 pass 注册接口的最小桩声明：只声明不实现，运行期由动态链接器绑定 libregister.so
 */

#ifndef FE_FUSION_PASS_REGISTRY_STUB_H_
#define FE_FUSION_PASS_REGISTRY_STUB_H_

#include <cstdint>
#include <map>
#include <string>

#include "log/log.h"

namespace fe {

// pass 阶段类型，枚举值必须与 GE 的 fe::GraphFusionPassType 一致
// （来源：cann/ge graph_fusion_pass_base.h）
enum GraphFusionPassType {
    BUILT_IN_GRAPH_PASS = 0,
    BUILT_IN_VECTOR_CORE_GRAPH_PASS,
    CUSTOM_AI_CORE_GRAPH_PASS,
    CUSTOM_VECTOR_CORE_GRAPH_PASS,
    SECOND_ROUND_BUILT_IN_GRAPH_PASS,
    BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS,
    BUILT_IN_PREPARE_GRAPH_PASS,
    BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS,
    BUILT_IN_TF_TAG_NO_CONST_FODING_GRAPH_PASS,
    BUILT_IN_TF_MERGE_SUB_GRAPH_PASS,
    BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS,
    BUILT_IN_EN_ISA_ARCH_V300_AND_V220_GRAPH_PASS,
    BUILT_IN_EN_ISA_ARCH_V100_GRAPH_PASS,
    BUILT_IN_EN_ISA_ARCH_V200_GRAPH_PASS,
    BUILT_IN_DELETE_NO_CONST_FOLDING_GRAPH_PASS,
    BUILT_IN_AFTER_MULTI_DIMS_PASS,
    BUILT_IN_AFTER_OPTIMIZE_STAGE1,
    BUILT_IN_AFTER_OP_JUDGE,
    BUILT_IN_AFTER_BUFFER_OPTIMIZE,
    GRAPH_FUSION_PASS_TYPE_RESERVED
};

using PassAttr = uint64_t;

class GraphPass;

struct PassDesc {
    PassAttr attr;
    GraphPass* (*create_fn)();
};

class FusionPassRegistry {
public:
    static FusionPassRegistry& GetInstance();
    std::map<std::string, PassDesc> GetPassDesc(const GraphFusionPassType& pass_type);
};

// 构造函数有副作用：构造即完成登记
class FusionPassRegistrar {
public:
    FusionPassRegistrar(const GraphFusionPassType& pass_type, const std::string& pass_name, GraphPass* (*create_fn)(),
                        PassAttr attr);
    ~FusionPassRegistrar() {}
};

} // namespace fe

// 弱符号：查询 ge-compiler 版本；环境里没有该函数时值为 nullptr，不报错
extern "C" {
__attribute__((weak)) int32_t aclsysGetVersionNum(char* pkgName, int32_t* versionNum);
}

namespace fe {

constexpr int32_t kStubGeCompilerVersion900 = 90000000;
constexpr int32_t kStubGeCompilerVersion901 = 90100000;
constexpr int32_t kStubGeCompilerVersion902 = 90200000;
// ACL_SUCCESS 的值（0）；桩头不引 acl 头文件，避免与本文件的弱符号声明在同一编译单元冲突
constexpr int32_t kStubAclSuccess = 0;
constexpr char kStubLogModule[] = "fe_fusion_pass_stub";
// 查询失败/弱符号未解析时返回的哨兵值（合法版本号恒 >= 0，不会与哨兵冲突）
constexpr int32_t kStubVersionInvalid = -1;

// 仅查询一次并缓存 ge-compiler 版本号（静态缓存），避免 47 条 STUB_REG_PASS 重复查询与重复打日志
inline int32_t StubGetGeCompilerVersion()
{
    static const int32_t version = []() {
        if (aclsysGetVersionNum == nullptr) {
            // 运行环境未提供该接口（弱符号未解析）：按查询失败处理，不登记、不报错
            return kStubVersionInvalid;
        }
        int32_t ver = 0;
        if (aclsysGetVersionNum(const_cast<char*>("ge-compiler"), &ver) != kStubAclSuccess) {
            OP_LOGW(kStubLogModule, "Failed to get ge-compiler version, skip stub registration.");
            return kStubVersionInvalid;
        }
        return ver;
    }();
    return version;
}

// 每条登记的运行期门槛：与对应 nn 实现 REG_FUSION_PASS 的运行期 Stage 判定阈值一致
// （实现为 IsTargetVersion()/IsGeVersionSupported()/Get*Stage() 判定型的，运行版本不足时 Stage 降级
//  为 kAfterInferShape，桩必须同样不登记，否则 fe 桥接查不到实现会报 "pattern fusion is nullptr"）
inline bool StubIsGeCompilerVersionAtLeast(int32_t minVersion) { return StubGetGeCompilerVersion() >= minVersion; }

} // namespace fe

// 弱符号探测 ge::TensorDesc 的 OriginDataType 公共 API 是否存在（运行环境导出）。
// MatmulQuantToFixpipeFusionPass 的运行期 Stage 判定除版本外还依赖该 API（同版本号的不同 CANN 包
// API 可用性可能不同，版本号无法单独判定），桩登记需复刻同一探测，声明照抄实现的 asm 别名弱符号。
extern "C" {
void stub_GetOriginDataType() __asm__("_ZNK2ge10TensorDesc17GetOriginDataTypeEv") __attribute__((weak));
void stub_SetOriginDataType() __asm__("_ZN2ge10TensorDesc17SetOriginDataTypeENS_8DataTypeE") __attribute__((weak));
}

namespace fe {

inline bool StubFixpipePublicApiAvailable()
{
    return stub_GetOriginDataType != nullptr && stub_SetOriginDataType != nullptr;
}

} // namespace fe

// 运行版本满足才构造 registrar（不满足则 nullptr，不登记）；工厂写死 nullptr，创建由 REG_FUSION_PASS 负责
#define STUB_REG_PASS(pass_name, pass_type, attr, min_ver) \
    STUB_REG_PASS_UNIQ_HELPER(__COUNTER__, pass_name, pass_type, attr, min_ver)

#define STUB_REG_PASS_UNIQ_HELPER(ctr, pass_name, pass_type, attr, min_ver) \
    STUB_REG_PASS_UNIQ(ctr, pass_name, pass_type, attr, min_ver)

#define STUB_REG_PASS_UNIQ(ctr, pass_name, pass_type, attr, min_ver)                                                  \
    static ::fe::FusionPassRegistrar* stub_reg_##ctr                                                                  \
        __attribute__((unused)) = ::fe::StubIsGeCompilerVersionAtLeast(min_ver) ?                                     \
                                      new ::fe::FusionPassRegistrar(                                                  \
                                          pass_type, pass_name, []() -> ::fe::GraphPass* { return nullptr; }, attr) : \
                                      nullptr

#endif // FE_FUSION_PASS_REGISTRY_STUB_H_
