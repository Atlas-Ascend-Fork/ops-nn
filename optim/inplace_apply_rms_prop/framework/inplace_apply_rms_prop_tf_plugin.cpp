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
 * \file inplace_apply_rms_prop_tf_plugin.cpp
 * \brief TF ApplyRMSProp / ResourceApplyRMSProp -> InplaceApplyRMSProp plugin mapping.
 *
 * TensorFlow hands the RMSProp state update over as one of two operator variants:
 *   - ApplyRMSProp         : V1 operator, ref inputs, one data output (updated var)
 *   - ResourceApplyRMSProp : resource variables, no data output
 * Both names are claimed here. The parser is the framework AutoMappingByOpFn, so
 * every same-named input/attribute is forwarded as-is and the V1 data output is
 * mapped to output 0 of the destination operator.
 *
 * The destination operator follows the compilation target SoC, resolved once when
 * this plugin library is loaded:
 *   - Ascend950 series             -> InplaceApplyRMSProp
 *   - other SoCs                   -> built-in ApplyRMSProp (historical mapping)
 *   - SoC cannot be determined     -> OP_LOGE + built-in ApplyRMSProp (never silent)
 *
 * rtGetSocVersion() is queried first: it returns the SoC version configured through
 * rtSetSocVersion(), i.e. the compilation target selected by ATC --soc_version, which
 * is what an offline conversion without a device must follow. Only when it is not
 * available (e.g. rtSetSocVersion() was never called) is aclrtGetSocName() used as the
 * device SoC fallback. A failure of both queries keeps the historical mapping but is
 * reported at ERROR level, so an Ascend950 target is never swapped out silently.
 */
#include <cstdint>
#include <string>
#include <vector>

#include "acl/acl_base_rt.h" /* aclrtGetSocName */
#include "log/log.h"
#include "register/register.h"

// Declared locally: this plugin does not include a runtime header. The 4-byte return
// value is rtError_t (0 == RT_ERROR_NONE), identical to the installed libruntime ABI.
extern "C" int rtGetSocVersion(char* socVersion, uint32_t socVersionLen);

// Contract forwarded unchanged to the destination operator:
//   inputs : var, ms, mom, lr, rho, momentum, epsilon, grad
//            (FLOAT / FLOAT16 / BFLOAT16, all identical; scalars are 0-D or {1})
//   attrs  : use_locking
//   outputs: var (V1 ref output) plus the destination state outputs ms/mom
// InplaceApplyRMSProp keeps var/ms/mom as explicit outputs instead of writing the
// state back through the input addresses, so the mapping relies on the framework
// ref-port alias between the TF ref variables and the destination outputs.

namespace domi {
namespace {

constexpr char kLogTag[] = "InplaceApplyRMSProp";
constexpr char kInplaceOpType[] = "InplaceApplyRMSProp";
constexpr char kBuiltinOpType[] = "ApplyRMSProp";
constexpr char kTargetSocPrefix[] = "Ascend950";
constexpr size_t kSocVersionLen = 128;

std::string QueryTargetSocVersion()
{
    char socVersion[kSocVersionLen] = {0};
    if (rtGetSocVersion(socVersion, static_cast<uint32_t>(sizeof(socVersion))) == 0 && socVersion[0] != '\0') {
        return std::string(socVersion);
    }
    const char* deviceSoc = aclrtGetSocName();
    return deviceSoc == nullptr ? std::string() : std::string(deviceSoc);
}

bool SelectInplaceMapping()
{
    const std::string soc = QueryTargetSocVersion();
    if (soc.empty()) {
        // Neither a configured target SoC nor a device is available, so the routing decision
        // cannot be made. Keep the historical mapping and fail loudly: silently swapping an
        // Ascend950 target back to the built-in operator would hide a broken offline build.
        OP_LOGE(kLogTag,
                "Target SoC cannot be determined (rtGetSocVersion and aclrtGetSocName both failed); TF "
                "ApplyRMSProp/ResourceApplyRMSProp keeps the built-in ApplyRMSProp mapping. Pass --soc_version "
                "for offline conversion so the Ascend950 routing is applied.");
        return false;
    }
    if (soc.compare(0, sizeof(kTargetSocPrefix) - 1, kTargetSocPrefix) == 0) {
        return true;
    }
    OP_LOGW(kLogTag,
            "Target SoC %s is not Ascend950; TF ApplyRMSProp/ResourceApplyRMSProp keeps the built-in "
            "ApplyRMSProp mapping.",
            soc.c_str());
    return false;
}

const bool gUseInplaceMapping = SelectInplaceMapping();

} // namespace

static OpReceiver registerInplaceApplyRmsProp = OpRegistrationData(gUseInplaceMapping ? kInplaceOpType : kBuiltinOpType)
                                                    .FrameworkType(TENSORFLOW)
                                                    .OriginOpType(std::vector<ge::AscendString>{"ApplyRMSProp",
                                                                                                "ResourceApplyRMSProp"})
                                                    .ParseParamsByOperatorFn(AutoMappingByOpFn)
                                                    .ImplyType(ImplyType::TVM);
} // namespace domi
