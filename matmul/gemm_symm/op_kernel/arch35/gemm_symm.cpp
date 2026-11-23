/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "gemm_symm_tiling_key.h"

#include "gemm_symm_epilogue.hpp"
#include "gemm_symm_block_mmad.hpp"
#include "gemm_symm_scheduler.hpp"
#include "gemm_symm_kernel.hpp"

using namespace Catlass;
using namespace tla;

#ifndef DTYPE_A
#define DTYPE_A bfloat16_t
#endif

template <bool HighPrec>
using GemmSymmL1Shape = AscendC::Std::conditional_t<HighPrec, Shape<Int<192>, Int<192>, Int<256>>,
                                                    Shape<Int<256>, Int<256>, Int<128>>>;
template <bool HighPrec>
using GemmSymmL0Shape = AscendC::Std::conditional_t<HighPrec, Shape<Int<192>, Int<192>, Int<64>>,
                                                    Shape<Int<256>, Int<256>, Int<64>>>;

template <bool TransA, bool TransB, bool HighPrec>
__aicore__ inline void GemmSymmLaunch(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR y, GemmSymmTilingData const& td)
{
    using ElementX = DTYPE_A;
    using ElementD = DTYPE_A;
    using ElementP = float;
    using LayoutTagA = AscendC::Std::conditional_t<TransA, layout::ColumnMajor, layout::RowMajor>;
    using LayoutTagB = AscendC::Std::conditional_t<TransB, layout::ColumnMajor, layout::RowMajor>;
    using L1TileShape = GemmSymmL1Shape<HighPrec>;
    using L0TileShape = GemmSymmL0Shape<HighPrec>;
    using TileCopy = Gemm::Tile::PackedTileCopyTla<Arch::Ascend950, ElementX, LayoutTagA, ElementX, LayoutTagB,
                                                   ElementP, layout::RowMajor>;
    using BlockMmad = Gemm::Block::BlockMmadSyrkTla<L1TileShape, L0TileShape, ElementX, ElementP, TileCopy, false>;
    using BlockEpilogue = Epilogue::Block::BlockEpilogueSyrkAxpby<Arch::Ascend950, ElementD, ElementD>;
    using BlockScheduler = Gemm::Block::SyrkBlockScheduler;
    using MatmulKernel = Gemm::Kernel::BasicSyrkWorkspaceTla<BlockMmad, BlockEpilogue, BlockScheduler, 2, !HighPrec,
                                                             HighPrec>;

    const float alpha = td.k == 0 ? 0.0F : td.alpha;
    const float beta = td.hasC == 0 ? 0.0F : td.beta;
    const uint32_t m = td.m;
    const uint32_t k = td.k;
    typename MatmulKernel::Params params{td.batch,
                                         GemmCoord{m, m, k},
                                         a,
                                         td.hasC == 0 ? nullptr : c,
                                         y,
                                         nullptr,
                                         tla::MakeLayout<ElementX, LayoutTagA>(m, k),
                                         tla::MakeLayout<ElementX, LayoutTagB>(k, m),
                                         tla::MakeLayout<ElementD, layout::RowMajor>(m, m),
                                         tla::MakeLayout<ElementD, layout::RowMajor>(m, m),
                                         static_cast<int64_t>(m) * k,
                                         static_cast<int64_t>(m) * m,
                                         static_cast<int64_t>(m) * m,
                                         alpha,
                                         beta,
                                         b,
                                         static_cast<int64_t>(m) * k};
    MatmulKernel kernel;
    kernel(params);
}

template <bool TransA, bool TransB, bool HighPrec>
__global__ __aicore__ void gemm_symm(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(GemmSymmTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    GET_TILING_DATA_WITH_STRUCT(GemmSymmTilingData, td, tiling);
    if (td.m == 0 || td.batch == 0 || td.usedCoreNum == 0) {
        return;
    }
    GemmSymmLaunch<TransA, TransB, HighPrec>(a, b, c, y, td);
}
