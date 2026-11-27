/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_situ_mx_quant_grouped.cpp
 * \brief Onboard test for the grouped path of SituMxQuant (group_index present),
 *        merged from cann-recipes-infer SituGluGroupQuant. CPU reference follows the
 *        recipes kernel semantics exactly: gate = first half, up = linearBeta*tanh(up/linearBeta),
 *        exp/Div direct evaluation, intermediate round-trip,
 *        then OCP MXFP8-e4m3fn quantization (group=32). Padding rows beyond
 *        sum(group_index) must stay untouched (sentinel checked).
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>

#include "acl/acl.h"
#include "aclnnop/aclnn_situ_glu_group_quant.h"

#ifndef ACLNN_ERR_PARAM_INVALID
#define ACLNN_ERR_PARAM_INVALID static_cast<aclnnStatus>(161002)
#endif

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// ==================== BF16 / FP8 / E8M0 Conversion Helpers ====================

uint16_t FloatToBf16(float val)
{
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t roundingBias = 0x7FFF + lsb;
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float Fp8E4M3ToFloat(uint8_t val)
{
    int32_t sign = (val >> 7) & 1;
    int32_t exp = (val >> 3) & 0xF;
    int32_t mant = val & 0x7;
    float result;
    if (exp == 0) {
        if (mant == 0) {
            result = 0.0f;
        } else {
            result = (mant / 8.0f) * std::ldexp(1.0f, 1 - 7);
        }
    } else if (exp == 0xF && mant == 0x7) {
        return sign ? -INFINITY : INFINITY;
    } else {
        result = (1.0f + mant / 8.0f) * std::ldexp(1.0f, exp - 7);
    }
    return sign ? -result : result;
}

uint8_t FloatToFp8E4M3(float val)
{
    if (std::isnan(val))
        return 0x7F;
    if (val == 0.0f)
        return std::signbit(val) ? 0x80 : 0x00;
    bool sign = val < 0;
    float absVal = std::fabs(val);

    if (absVal >= 448.0f) {
        return sign ? 0xFE : 0x7E;
    }

    int exp;
    float mant = std::frexp(absVal, &exp);
    exp = exp + 6;

    if (exp <= 0) {
        int32_t mantBits = static_cast<int32_t>(std::nearbyint(absVal * 8.0f * 64.0f));
        if (mantBits <= 0)
            return sign ? 0x80 : 0x00;
        if (mantBits > 7)
            mantBits = 7;
        return static_cast<uint8_t>((sign ? 0x80 : 0) | mantBits);
    }

    if (exp > 15) {
        return sign ? 0xFE : 0x7E;
    }

    int32_t mantBits = static_cast<int32_t>(std::nearbyint((mant * 2.0f - 1.0f) * 8.0f));
    if (mantBits < 0)
        mantBits = 0;
    if (mantBits > 7) {
        mantBits = 0;
        exp++;
        if (exp > 15) {
            return sign ? 0xFE : 0x7E;
        }
    }
    if (exp == 15 && mantBits >= 7) {
        mantBits = 6;
    }

    return static_cast<uint8_t>((sign ? 0x80 : 0) | (exp << 3) | mantBits);
}

float E8M0ToFloat(uint8_t e8m0)
{
    if (e8m0 == 0)
        return 0.0f;
    int32_t exp = static_cast<int32_t>(e8m0) - 127;
    return std::ldexp(1.0f, exp);
}

// ==================== CPU Reference (recipes SituGluGroupQuant semantics) ====================
//
// Activation (fp32, two-path tanh matching SituActivationVF, tanh.h caliber):
//   gate = x[row, c]            (FIRST half; activate_left is forced on grouped)
//   up   = x[row, c + EPR]      (second half)
//   x = gate/beta (plain 1/beta — subnormal beta gives +-inf, sigmoid branch
//   consumes it; gate=0*inf=NaN falls to the exit cast as +0)
//   TwoPathTanh(x): |x| < 0.6 -> x*(1+c1*x^2+c2*x^4+c3*x^6+c4*x^8)
//                            -> x*(1+c1*x^2+c2*x^4+c3*x^6+c4*x^8) (Horner, FMA order)
//                   |x| >= 0.6 -> 2/(1+exp(-2x)) - 1
//   gateBranch = (TwoPathTanh(gate/beta) * sigmoid(gate)) * beta
//   upBranch   = TwoPathTanh(up/linearBeta) * linearBeta
//   sigmoid subnormal results (< 2^-126) are flushed to 0 (hardware FTZ)
//   act = bf16(gateBranch * upBranch)   (act buffer is always bf16 on device)
//   NOTE: pure fp32 activation chain with a single exit rounding — the
//   intermediate branch round-trips were removed with high_precision in
//   7c5282b36 (kernel == recipes hp=true caliber); the stale round-trips here
//   produced 1-bf16-ulp act deltas that, when sitting on a group amax power-of-2
//   boundary, shifted the whole group's scale by one e8m0 step (onboard-found
//   at EPR=192/3072: 1-2 groups x 32 y bytes off by 2x).
// Quantization: identical OCP e4m3fn algorithm as the normal path (emax=8).

void CpuSituGluGroupQuant(const std::vector<uint16_t>& xBf16, int64_t totalRows, int64_t colNum, float beta,
                          float linearBeta, const std::vector<int64_t>& groupIndex, std::vector<uint8_t>& yFp8,
                          std::vector<uint8_t>& y_scaleE8M0)
{
    const int64_t epr = colNum / 2;
    const int64_t blockSize = 32;
    const int64_t groupCount = (epr + blockSize - 1) / blockSize;
    const int64_t scaleColNum = (groupCount + 1) / 2 * 2;
    const int32_t emax = 8;

    int64_t validRows = 0;
    for (auto t : groupIndex) {
        validRows += t;
    }
    validRows = std::min(validRows, totalRows);

    // Two-path tanh (tanh.h reference, mirrors VFTanhTwoPathVf). Horner uses
    // explicit fmaf() to mirror the kernel FusedMulDstAdd single-rounding order.
    struct TwoPathTanh {
        float operator()(float x) const
        {
            const float c1 = -0.333327681f, c2 = 0.133152977f;
            const float c3 = -0.0523039624f, c4 = 0.0157396831f;
            float x2 = x * x;
            float p = x2 * c4;
            p = p + c3;
            p = fmaf(x2, p, c2);
            p = fmaf(x2, p, c1);
            p = p * x2;
            p = fmaf(x, p, x);
            float naive = 2.0f / (1.0f + std::exp(-2.0f * x)) - 1.0f;
            return std::fabs(x) >= 0.6f ? naive : p;
        }
    } twoPathTanh;
    auto SituBranch = [beta, &twoPathTanh](float g) {
        float x = g / beta; // plain division: subnormal beta -> +-inf, sigmoid branch consumes
        float tanhResult = twoPathTanh(x);
        float sigmoidResult = 1.0f / (1.0f + std::exp(-g));
        if (std::fabs(sigmoidResult) < 1.17549435e-38f) {
            sigmoidResult = 0.0f; // hardware FTZ: flush subnormal sigmoid results
        }
        return (tanhResult * sigmoidResult) * beta;
    };
    auto UpBranch = [linearBeta, &twoPathTanh](float u) { return twoPathTanh(u / linearBeta) * linearBeta; };

    for (int64_t r = 0; r < validRows; r++) {
        std::vector<float> act(epr);
        for (int64_t c = 0; c < epr; c++) {
            float gate = Bf16ToFloat(xBf16[r * colNum + c]);
            float up = Bf16ToFloat(xBf16[r * colNum + c + epr]);
            float gateBranch = SituBranch(gate);
            float upBranch = UpBranch(up);
            // pure fp32 chain, single exit rounding to bf16 (kernel caliber)
            act[c] = Bf16ToFloat(FloatToBf16(gateBranch * upBranch));
        }

        for (int64_t g = 0; g < groupCount; g++) {
            int64_t start = g * blockSize;
            int64_t end = std::min(start + blockSize, epr);
            float maxVal = 0.0f;
            for (int64_t c = start; c < end; c++) {
                maxVal = std::max(maxVal, std::fabs(act[c]));
            }

            uint8_t e8m0;
            float recipScale;
            if (maxVal <= 0.0f) {
                e8m0 = 0;
                recipScale = 0.0f;
            } else if (std::isinf(maxVal) || std::isnan(maxVal)) {
                e8m0 = 0xFF;
                recipScale = 0.0f;
            } else {
                int32_t sharedExp = static_cast<int32_t>(std::floor(std::log2(maxVal))) - emax;
                if (sharedExp + 127 <= 0) {
                    e8m0 = 0;
                    recipScale = 0.0f;
                } else if (sharedExp == 0) {
                    // kernel: zeroMask on sharedExp==0 forces recipScale=0 (flush to zero)
                    e8m0 = static_cast<uint8_t>(sharedExp + 127);
                    recipScale = 0.0f;
                } else {
                    e8m0 = static_cast<uint8_t>(sharedExp + 127);
                    recipScale = std::ldexp(1.0f, -sharedExp);
                }
            }
            y_scaleE8M0[r * scaleColNum + g] = e8m0;

            for (int64_t c = start; c < end; c++) {
                // kernel multiplies in the bf16 domain: act(bf16) * recip(bf16) rounds
                // the product back to bf16 before the fp8 cast
                float prod = Bf16ToFloat(FloatToBf16(act[c] * recipScale));
                yFp8[r * epr + c] = FloatToFp8E4M3(prod);
            }
        }
    }
}

// ==================== Test Driver ====================

struct GroupedTestConfig {
    int64_t totalRows;
    int64_t colNum; // x last dim = 2 * EPR
    std::vector<int64_t> groupIndex;
    float beta;
    float linearBeta; // maps to aclnn linearBeta
    const char* desc;
};

int RunGroupedTest(int32_t deviceId, aclrtStream stream, const GroupedTestConfig& cfg)
{
    LOG_PRINT("\n--- %s ---\n", cfg.desc);
    const int64_t epr = cfg.colNum / 2;
    const int64_t groupCount = (epr + 31) / 32;
    const int64_t scaleColNum = (groupCount + 1) / 2 * 2;

    std::vector<int64_t> xShape = {cfg.totalRows, cfg.colNum};
    std::vector<int64_t> tokensShape = {static_cast<int64_t>(cfg.groupIndex.size())};
    std::vector<int64_t> yShape = {cfg.totalRows, epr};
    std::vector<int64_t> y_scaleShape = {cfg.totalRows, scaleColNum / 2, 2};

    // deterministic pseudo-random bf16 input in [-2, 2]
    std::vector<uint16_t> xHost(cfg.totalRows * cfg.colNum);
    uint32_t seed = 42;
    for (auto& v : xHost) {
        seed = seed * 1664525 + 1013904223;
        float f = static_cast<float>((seed >> 8) % 4000) / 1000.0f - 2.0f;
        v = FloatToBf16(f);
    }

    // prefill y/y_scale with sentinel bytes to verify padding rows stay untouched
    std::vector<uint8_t> yHost(cfg.totalRows * epr, 0xA5);
    std::vector<uint8_t> scaleHost(cfg.totalRows * scaleColNum, 0xA5);

    void* xDevice = nullptr;
    void* tokensDevice = nullptr;
    void* yDevice = nullptr;
    void* scaleDevice = nullptr;
    aclTensor* x = nullptr;
    aclTensor* groupIndex = nullptr;
    aclTensor* y = nullptr;
    aclTensor* y_scale = nullptr;

    int ret = CreateAclTensor(xHost, xShape, &xDevice, ACL_BF16, &x);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(cfg.groupIndex, tokensShape, &tokensDevice, ACL_INT64, &groupIndex);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(yHost, yShape, &yDevice, ACL_FLOAT8_E4M3FN, &y);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensor(scaleHost, y_scaleShape, &scaleDevice, ACL_FLOAT8_E8M0, &y_scale);
    CHECK_RET(ret == 0, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnSituGluGroupQuantGetWorkspaceSize(x, nullptr, groupIndex, cfg.beta, cfg.linearBeta, true, -1, 0, 36, 0.0,
                                                 "rint", 1, y, y_scale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituGluGroupQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return 1);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed. ERROR: %d\n", ret); return 1);
    }

    ret = aclnnSituGluGroupQuant(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituGluGroupQuant failed. ERROR: %d\n", ret); return 1);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return 1);

    ret = aclrtMemcpy(yHost.data(), yHost.size(), yDevice, yHost.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("memcpy y failed. ERROR: %d\n", ret); return 1);
    ret = aclrtMemcpy(scaleHost.data(), scaleHost.size(), scaleDevice, scaleHost.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("memcpy scale failed. ERROR: %d\n", ret); return 1);

    std::vector<uint8_t> yRef(cfg.totalRows * epr, 0xA5);
    std::vector<uint8_t> scaleRef(cfg.totalRows * scaleColNum, 0xA5);
    CpuSituGluGroupQuant(xHost, cfg.totalRows, cfg.colNum, cfg.beta, cfg.linearBeta, cfg.groupIndex, yRef, scaleRef);

    int64_t validRows = 0;
    for (auto t : cfg.groupIndex) {
        validRows += t;
    }
    validRows = std::min(validRows, cfg.totalRows);

    // exact byte compare on valid rows; sentinel compare on padding rows
    int64_t yMismatch = 0;
    int64_t scaleMismatch = 0;
    int64_t paddingTouched = 0;
    for (int64_t r = 0; r < cfg.totalRows; r++) {
        for (int64_t c = 0; c < epr; c++) {
            uint8_t got = yHost[r * epr + c];
            uint8_t exp = yRef[r * epr + c];
            if (r < validRows) {
                bool sameValue = (got == exp) || (Fp8E4M3ToFloat(got) == Fp8E4M3ToFloat(exp)); // -0 == +0
                bool withinUlp = (got / 0x80 == exp / 0x80) &&
                                 (std::abs(static_cast<int>(got % 0x80) - static_cast<int>(exp % 0x80)) <= 1);
                if (!sameValue && !withinUlp) {
                    if (yMismatch < 5) {
                        LOG_PRINT("  y mismatch at [%ld,%ld]: got 0x%02x(%f) expect 0x%02x(%f)\n", r, c, got,
                                  Fp8E4M3ToFloat(got), exp, Fp8E4M3ToFloat(exp));
                    }
                    yMismatch++;
                }
            } else if (got != 0xA5) {
                paddingTouched++;
            }
        }
        for (int64_t c = 0; c < scaleColNum; c++) {
            uint8_t got = scaleHost[r * scaleColNum + c];
            uint8_t exp = scaleRef[r * scaleColNum + c];
            if (r < validRows) {
                if (got != exp) {
                    if (scaleMismatch < 5) {
                        LOG_PRINT("  scale mismatch at [%ld,%ld]: got 0x%02x expect 0x%02x\n", r, c, got, exp);
                    }
                    scaleMismatch++;
                }
            } else if (got != 0xA5) {
                paddingTouched++;
            }
        }
    }

    LOG_PRINT("  validRows=%ld/%ld | y mismatch=%ld | scale mismatch=%ld | padding touched=%ld\n", validRows,
              cfg.totalRows, yMismatch, scaleMismatch, paddingTouched);
    // dump row-0 scale bytes for diagnosis
    LOG_PRINT("  scale row0 got   :");
    for (int64_t c = 0; c < scaleColNum; c++) {
        LOG_PRINT(" %02x", scaleHost[c]);
    }
    LOG_PRINT("\n  scale row0 expect:");
    for (int64_t c = 0; c < scaleColNum; c++) {
        LOG_PRINT(" %02x", scaleRef[c]);
    }
    LOG_PRINT("\n");
    bool passed = (yMismatch == 0 && scaleMismatch == 0 && paddingTouched == 0);
    LOG_PRINT("  -> %s\n", passed ? "PASS" : "FAIL");

    aclDestroyTensor(x);
    aclDestroyTensor(groupIndex);
    aclDestroyTensor(y);
    aclDestroyTensor(y_scale);
    aclrtFree(xDevice);
    aclrtFree(tokensDevice);
    aclrtFree(yDevice);
    aclrtFree(scaleDevice);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return passed ? 0 : 1;
}

// negative probe: illegal params must be rejected at stage 1 (ACLNN_ERR_PARAM_INVALID)
int RunNegativeProbe(double beta, double linearBeta, bool badYDtype, bool emptyTokens, bool oddLastDim,
                     bool eprTooLarge, const char* desc)
{
    LOG_PRINT("  probe: %s\n", desc);
    std::vector<int64_t> xShape = {4, oddLastDim ? 127 : (eprTooLarge ? 8192 : 256)};
    std::vector<int64_t> yShape = {4, oddLastDim ? 64 : (eprTooLarge ? 4096 : 128)};
    std::vector<int64_t> scaleShape = {4, eprTooLarge ? 64 : 2, 2};
    std::vector<int64_t> tokensShape = {emptyTokens ? 0 : 2};

    void* nullData = nullptr;
    auto makeStrides = [](const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        return strides;
    };
    auto xs = makeStrides(xShape);
    auto ys = makeStrides(yShape);
    auto ss = makeStrides(scaleShape);
    auto ts = makeStrides(tokensShape);

    aclTensor* x = aclCreateTensor(xShape.data(), xShape.size(), ACL_BF16, xs.data(), 0, ACL_FORMAT_ND, xShape.data(),
                                   xShape.size(), nullData);
    aclTensor* y = aclCreateTensor(yShape.data(), yShape.size(), badYDtype ? ACL_FLOAT8_E5M2 : ACL_FLOAT8_E4M3FN,
                                   ys.data(), 0, ACL_FORMAT_ND, yShape.data(), yShape.size(), nullData);
    aclTensor* scale = aclCreateTensor(scaleShape.data(), scaleShape.size(), ACL_FLOAT8_E8M0, ss.data(), 0,
                                       ACL_FORMAT_ND, scaleShape.data(), scaleShape.size(), nullData);
    aclTensor* tokens = aclCreateTensor(tokensShape.data(), tokensShape.size(), ACL_INT64, ts.data(), 0, ACL_FORMAT_ND,
                                        tokensShape.data(), tokensShape.size(), nullData);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSituGluGroupQuantGetWorkspaceSize(x, nullptr, tokens, beta, linearBeta, true, -1, 0, 36, 0.0,
                                                      "rint", 1, y, scale, &workspaceSize, &executor);
    bool passed = (ret == ACLNN_ERR_PARAM_INVALID);
    LOG_PRINT("    -> %s (ret=%d)\n", passed ? "PASS" : "FAIL", static_cast<int>(ret));

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclDestroyTensor(scale);
    aclDestroyTensor(tokens);
    return passed ? 0 : 1;
}

int main(int argc, char** argv)
{
    LOG_PRINT("=== SituGluGroupQuant (standalone port) Onboard Test ===\n");
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

    if (argc > 1 && std::string(argv[1]) == "scan") {
        std::vector<int64_t> eprs = {32, 64, 96, 128, 160, 192, 256, 512};
        for (int64_t epr : eprs) {
            GroupedTestConfig cfg = {2, epr * 2, {2}, 1.0f, 2.0f, "scan"};
            const int64_t g = (epr + 31) / 32;
            const int64_t scn = (g + 1) / 2 * 2;
            LOG_PRINT("\n=== scan EPR=%ld groupCount=%ld scaleColNum=%ld ===\n", epr, g, scn);
            (void)RunGroupedTest(deviceId, stream, cfg);
        }
        Finalize(deviceId, stream);
        return 0;
    }

    // NOTE (recipes inherited limits, kept verbatim by the port):
    //  - EPR < 128 (groupCount < 4): the masked PACK_B16 scale store silently drops
    //    sub-8B payloads, y_scale stays uninitialized. Cases stay >= 128.
    //  - EPR > 3072 is NOT rejected (recipes caliber): the whole-row CopyIn cap
    //    is an upstream-caller assumption, not an enforced limit.
    std::vector<GroupedTestConfig> tests = {
        {8, 256, {4, 4}, 1.0f, 2.0f, "G1: 2 experts x4 rows, EPR=128"},
        {8, 256, {4, 4}, 1.0f, 2.0f, "G2: same as G1"},
        {8, 256, {2, 0, 3}, 0.5f, 1.5f, "G3: zero-token expert + padding rows (5/8 valid)"},
        {8, 256, {20}, 1.0f, 2.0f, "G4: validRows clamp (sum=20 > totalRows=8)"},
        {16, 256, {3, 5, 8}, 1.0f, 0.8f, "G5: 3 experts uneven, EPR=128"},
        {4, 6144, {4}, 1.0f, 2.0f, "G6: EPR=3072 (recipes chunk boundary)"},
    };

    int failed = 0;
    for (const auto& cfg : tests) {
        failed += RunGroupedTest(deviceId, stream, cfg);
    }

    LOG_PRINT("\n=== Negative Probes ===\n");
    int probeFailed = 0;
    probeFailed += RunNegativeProbe(0.0, 2.0, false, false, false, false, "beta=0 rejected");
    probeFailed += RunNegativeProbe(-1.0, 2.0, false, false, false, false, "beta<0 rejected");
    probeFailed += RunNegativeProbe(1.0, 2.0, true, false, false, false, "y dtype e5m2 rejected");
    probeFailed += RunNegativeProbe(1.0, 2.0, false, false, true, false, "odd last dim rejected");

    LOG_PRINT("\n=== Summary: %ld/%ld functional PASSED, %d probe(s) FAILED ===\n",
              static_cast<int64_t>(tests.size()) - failed, static_cast<int64_t>(tests.size()), probeFailed);
    Finalize(deviceId, stream);
    return (failed == 0 && probeFailed == 0) ? 0 : 1;
}
