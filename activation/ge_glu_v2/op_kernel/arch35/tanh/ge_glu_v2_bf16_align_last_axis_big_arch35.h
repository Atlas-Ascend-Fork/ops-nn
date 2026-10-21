/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GeGluV2_BFLOAT16_LAST_BIG_ARCH35_H
#define GeGluV2_BFLOAT16_LAST_BIG_ARCH35_H

#include "../ge_glu_v2_base.h"

namespace GeGluV2 {
using namespace AscendC;

template <typename T>
class GeGluV2Bf16AlignLastAxisBig : public GeGluV2Base<T> {
public:
    __aicore__ inline GeGluV2Bf16AlignLastAxisBig(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace,
                                const GeGluV2TilingData* tilingData);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum = 2;

private:
    __aicore__ inline void CopyInX(const int64_t& idx_x, const int64_t& idx_y, const int64_t& length);
    __aicore__ inline void ComputeGeluAndMul(const int64_t& ub_num);
    __aicore__ inline void CopyOutGelu(const int64_t& idx_x, const int64_t& idx_y, const int64_t& length);
    __aicore__ inline void CopyOutMul(const int64_t& idx_x, const int64_t& idx_y, const int64_t& length);
    __aicore__ inline void ProcessPerCore();
    __aicore__ inline void ProcessLastCore();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueX1;
    TQue<QuePosition::VECIN, bufferNum> inQueueX2;
    TQue<QuePosition::VECOUT, bufferNum> outQueueGelu;
    TQue<QuePosition::VECOUT, bufferNum> outQueueMul;
};

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace,
                                                            const GeGluV2TilingData* tilingData)
{
    this->BaseInit(x, y, gelu, tilingData, true);
    pipe.InitBuffer(inQueueX1, bufferNum, this->m_tilingData.bufSize * sizeof(T));
    pipe.InitBuffer(inQueueX2, bufferNum, this->m_tilingData.bufSize * sizeof(T));
    pipe.InitBuffer(outQueueGelu, bufferNum, this->m_tilingData.bufSize * sizeof(T));
    pipe.InitBuffer(outQueueMul, bufferNum, this->m_tilingData.bufSize * sizeof(T));
}

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::Process()
{
    if (this->blockIdx >= this->m_tilingData.realCoreNum) {
        return;
    }

    ProcessPerCore();
}

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::ProcessPerCore()
{
    int64_t totalLoop = this->m_tilingData.loopNum;
    int64_t coreNum = this->m_tilingData.realCoreNum;
    int64_t tilesPerCore = totalLoop / coreNum;
    int64_t remainder = totalLoop % coreNum;

    int64_t startZ;
    int64_t endZ;
    if (this->blockIdx < remainder) {
        startZ = this->blockIdx * (tilesPerCore + 1);
        endZ = startZ + tilesPerCore + 1;
    } else {
        startZ = remainder * (tilesPerCore + 1) + (this->blockIdx - remainder) * tilesPerCore;
        endZ = startZ + tilesPerCore;
    }

    for (int64_t z = startZ; z < endZ; z++) {
        int64_t idx_x = 0;
        int64_t idx_y = 0;
        int64_t length = 0;
        if (this->m_tilingData.tailLoopNum != 0) {
            idx_x = z / (this->m_tilingData.group + 1);
            idx_y = z % (this->m_tilingData.group + 1);
            length = idx_y == this->m_tilingData.group ? this->m_tilingData.tailLoopNum : this->m_tilingData.splitSize;
        } else {
            idx_x = z / this->m_tilingData.group;
            idx_y = z % this->m_tilingData.group;
            length = this->m_tilingData.splitSize;
        }

        CopyInX(idx_x, idx_y, length);
        ComputeGeluAndMul(length);
        CopyOutGelu(idx_x, idx_y, length);
        CopyOutMul(idx_x, idx_y, length);
    }
}

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::CopyInX(const int64_t& idx_x, const int64_t& idx_y,
                                                               const int64_t& length)
{
    LocalTensor<T> ubX1 = inQueueX1.AllocTensor<T>();
    LocalTensor<T> ubX2 = inQueueX2.AllocTensor<T>();
    this->CopyInXAlignLastBig(idx_x, idx_y, length, ubX1, ubX2);

    inQueueX1.EnQue(ubX1);
    inQueueX2.EnQue(ubX2);
}

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::ComputeGeluAndMul(const int64_t& ub_num)
{
    LocalTensor<T> ubX2 = inQueueX2.DeQue<T>();
    LocalTensor<T> ubX1 = inQueueX1.DeQue<T>();
    LocalTensor<T> gelu_out = outQueueGelu.AllocTensor<T>();
    LocalTensor<T> mul_out = outQueueMul.AllocTensor<T>();

    __ubuf__ T* x1Addr = (__ubuf__ T*)ubX1.GetPhyAddr();
    __ubuf__ T* x2Addr = (__ubuf__ T*)ubX2.GetPhyAddr();
    __ubuf__ T* geluAddr = (__ubuf__ T*)gelu_out.GetPhyAddr();
    __ubuf__ T* mulAddr = (__ubuf__ T*)mul_out.GetPhyAddr();

    AscendC::VF_CALL<GeGluV2ComputeVf<T, T>>(x1Addr, x2Addr, geluAddr, mulAddr, static_cast<uint32_t>(ub_num));

    inQueueX2.FreeTensor(ubX2);
    inQueueX1.FreeTensor(ubX1);
    outQueueGelu.EnQue(gelu_out);
    outQueueMul.EnQue(mul_out);
}

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::CopyOutGelu(const int64_t& idx_x, const int64_t& idx_y,
                                                                   const int64_t& length)
{
    LocalTensor<T> outLocalGelu = outQueueGelu.DeQue<T>();
    this->CopyOutGeluBaseLastBig(idx_x, idx_y, length, outLocalGelu);
    outQueueGelu.FreeTensor(outLocalGelu);
}

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::CopyOutMul(const int64_t& idx_x, const int64_t& idx_y,
                                                                  const int64_t& length)
{
    LocalTensor<T> outLocalMul = outQueueMul.DeQue<T>();
    this->CopyOutMulBaseLastBig(idx_x, idx_y, length, outLocalMul);
    outQueueMul.FreeTensor(outLocalMul);
}
} // namespace GeGluV2
#endif // GeGluV2_BFLOAT16_LAST_BIG_ARCH35_H
