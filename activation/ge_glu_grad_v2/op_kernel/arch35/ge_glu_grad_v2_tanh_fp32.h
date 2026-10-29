/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ge_glu_grad_v2_tanh_fp32.h
 * \brief
 */
#ifndef GE_GLU_GRAD_V2_TANH_FP32_H_
#define GE_GLU_GRAD_V2_TANH_FP32_H_

#include "ge_glu_grad_v2_tanh_base.h"

namespace GeGluGradV2Tanh {
using namespace AscendC;

class GeGluGradV2TanhFP32 : public GeGluGradV2TanhBase<float> {
public:
    __aicore__ inline GeGluGradV2TanhFP32(GM_ADDR dy, GM_ADDR x, GM_ADDR gelu, GM_ADDR dx,
                                          const GeGluGradV2TilingData* tilingDataPtr, TPipe* tPipe)
        : GeGluGradV2TanhBase<float>(dy, x, gelu, dx, tilingDataPtr, tPipe){};
    __aicore__ inline void Init();

    __aicore__ inline void Process(bool perfMode = false)
    {
        if (perfMode) {
            ProcessPerf<GeGluGradV2TanhFP32, &GeGluGradV2TanhFP32::ComputeLeftHalf,
                        &GeGluGradV2TanhFP32::ComputeRightHalf>(this);
            return;
        }

        ProcessFused<GeGluGradV2TanhFP32, &GeGluGradV2TanhFP32::ComputeFused>(this);
    };

private:
    __aicore__ inline void ComputeLeftHalf(const int64_t& realProcCount);
    __aicore__ inline void ComputeRightHalf(const int64_t& realProcCount);
    __aicore__ inline void ComputeFused(const int64_t& realProcCount);
};

__aicore__ inline void GeGluGradV2TanhFP32::Init()
{
    pipe->InitBuffer(inQueueDY, DB_BUFFER, maxProcCount * sizeof(float));
    pipe->InitBuffer(inQueueGelu, DB_BUFFER, maxProcCount * sizeof(float));
    pipe->InitBuffer(inQueueX1, DB_BUFFER, maxProcCount * sizeof(float));
    pipe->InitBuffer(inQueueX2, DB_BUFFER, maxProcCount * sizeof(float));

    pipe->InitBuffer(outQueueDX1, DB_BUFFER, maxProcCount * sizeof(float));
    pipe->InitBuffer(outQueueDX2, DB_BUFFER, maxProcCount * sizeof(float));

    pipe->InitBuffer(resultTempBuf0, maxProcCount * sizeof(float));
    pipe->InitBuffer(resultTempBuf1, maxProcCount * sizeof(float));
    pipe->InitBuffer(resultTempBuf2, maxProcCount * sizeof(float));
    pipe->InitBuffer(resultTempBuf3, maxProcCount * sizeof(float));
    pipe->InitBuffer(resultTempBuf4, maxProcCount * sizeof(float));
}

__aicore__ inline void GeGluGradV2TanhFP32::ComputeLeftHalf(const int64_t& realProcCount)
{
    LocalTensor<float> ubDY = inQueueDY.DeQue<float>();
    LocalTensor<float> ubGelu = inQueueGelu.DeQue<float>();
    LocalTensor<float> ubX1 = inQueueX1.DeQue<float>();
    LocalTensor<float> xBufLeft = resultTempBuf0.Get<float>();
    LocalTensor<float> outLocalLeft = outQueueDX1.AllocTensor<float>();
    Mul(outLocalLeft, ubGelu, ubDY, realProcCount); // dx1 = gelu * dy
    Mul(xBufLeft, ubX1, ubDY, realProcCount);       // x1 = x1 * dy
    inQueueGelu.FreeTensor(ubGelu);
    inQueueDY.FreeTensor(ubDY);
    inQueueX1.FreeTensor(ubX1);
    outQueueDX1.EnQue(outLocalLeft);
}

__aicore__ inline void GeGluGradV2TanhFP32::ComputeRightHalf(const int64_t& realProcCount)
{
    LocalTensor<float> ubX2 = inQueueX2.DeQue<float>();
    LocalTensor<float> outLocalRight = outQueueDX2.AllocTensor<float>();
    LocalTensor<float> xBufLeft = resultTempBuf0.Get<float>();
    ComputeGeluGrad(outLocalRight, xBufLeft, ubX2, realProcCount);
    outQueueDX2.EnQue(outLocalRight);
    inQueueX2.FreeTensor(ubX2);
}

__aicore__ inline void GeGluGradV2TanhFP32::ComputeFused(const int64_t& realProcCount)
{
#ifndef __CCE_AICORE__
    ComputeLeftHalf(realProcCount);
    ComputeRightHalf(realProcCount);
    return;
#else
    LocalTensor<float> ubDY = inQueueDY.DeQue<float>();
    LocalTensor<float> ubGelu = inQueueGelu.DeQue<float>();
    LocalTensor<float> ubX1 = inQueueX1.DeQue<float>();
    LocalTensor<float> ubX2 = inQueueX2.DeQue<float>();
    LocalTensor<float> outLocalLeft = outQueueDX1.AllocTensor<float>();
    LocalTensor<float> outLocalRight = outQueueDX2.AllocTensor<float>();

    __ubuf__ float* dyPtr = (__ubuf__ float*)ubDY.GetPhyAddr();
    __ubuf__ float* geluPtr = (__ubuf__ float*)ubGelu.GetPhyAddr();
    __ubuf__ float* x1Ptr = (__ubuf__ float*)ubX1.GetPhyAddr();
    __ubuf__ float* x2Ptr = (__ubuf__ float*)ubX2.GetPhyAddr();
    __ubuf__ float* outLeftPtr = (__ubuf__ float*)outLocalLeft.GetPhyAddr();
    __ubuf__ float* outRightPtr = (__ubuf__ float*)outLocalRight.GetPhyAddr();

    constexpr uint32_t vl = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t loopNum = (static_cast<uint32_t>(realProcCount) + vl - 1) / vl;
    uint32_t count = static_cast<uint32_t>(realProcCount);

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vDy;
        Reg::RegTensor<float> vGelu;
        Reg::RegTensor<float> vGate;
        Reg::RegTensor<float> vAct;
        Reg::RegTensor<float> vGateDy;
        Reg::RegTensor<float> vDxGate;
        Reg::RegTensor<float> vX2Sq;
        Reg::RegTensor<float> vP;
        Reg::RegTensor<float> vG2;
        Reg::RegTensor<float> vD;
        Reg::RegTensor<float> vT;
        Reg::RegTensor<float> vResp;
        Reg::RegTensor<float> vOne;
        Reg::RegTensor<float> vDxAct;
        Reg::MaskReg mask;
        Reg::Duplicate(vOne, 1.0f);
        for (uint16_t loopIdx = 0; loopIdx < loopNum; ++loopIdx) {
            mask = Reg::UpdateMask<float>(count);
            Reg::DataCopy(vDy, dyPtr + loopIdx * vl);
            Reg::DataCopy(vGelu, geluPtr + loopIdx * vl);
            Reg::DataCopy(vGate, x1Ptr + loopIdx * vl);
            Reg::DataCopy(vAct, x2Ptr + loopIdx * vl);

            Reg::Mul(vDxGate, vGelu, vDy, mask);
            Reg::Mul(vGateDy, vGate, vDy, mask);

            Reg::Mul(vX2Sq, vAct, vAct, mask);
            Reg::Muls(vP, vX2Sq, COEFFICIENT_A1, mask);
            Reg::Adds(vP, vP, COEFFICIENT_A2, mask);
            Reg::Mul(vP, vP, vAct, mask);
            Reg::Exp(vP, vP, mask);

            Reg::Muls(vG2, vX2Sq, COEFFICIENT_A3, mask);
            Reg::Adds(vG2, vG2, COEFFICIENT_A4, mask);

            Reg::Adds(vT, vP, 1.0f, mask);
            Reg::Div(vD, vOne, vT, mask);
            // Preserve the original derivative and exceptional-value semantics:
            // ((d - 1) * act * g2 + 1) * d, where g2 has negative coefficients.
            Reg::Adds(vResp, vD, -1.0f, mask);
            Reg::Mul(vResp, vResp, vAct, mask);
            Reg::Mul(vResp, vResp, vG2, mask);
            Reg::Adds(vResp, vResp, 1.0f, mask);
            Reg::Mul(vResp, vResp, vD, mask);
            Reg::Mul(vDxAct, vGateDy, vResp, mask);

            Reg::StoreAlign(outLeftPtr + loopIdx * vl, vDxGate, mask);
            Reg::StoreAlign(outRightPtr + loopIdx * vl, vDxAct, mask);
        }
    }

    inQueueDY.FreeTensor(ubDY);
    inQueueGelu.FreeTensor(ubGelu);
    inQueueX1.FreeTensor(ubX1);
    inQueueX2.FreeTensor(ubX2);
    outQueueDX1.EnQue(outLocalLeft);
    outQueueDX2.EnQue(outLocalRight);
#endif
}

} // namespace GeGluGradV2Tanh

#endif // GE_GLU_GRAD_V2_TANH_FP32_H_
