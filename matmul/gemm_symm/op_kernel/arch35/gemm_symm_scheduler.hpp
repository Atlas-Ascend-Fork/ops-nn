/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_SYRK_H
#define CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_SYRK_H

#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Block {

struct SyrkBlockScheduler {
    static constexpr uint32_t kRowAlign = 16;

    GemmCoord problemShape;
    MatrixCoord tileMN;
    uint32_t triangleTasks{0};
    uint32_t tasksPerBatch{0};
    uint32_t coreNum{0};
    uint32_t mainTasks{0};
    uint32_t tailTiles{0};
    uint32_t tailSplit{1};

    CATLASS_HOST_DEVICE
    SyrkBlockScheduler(GemmCoord const& problemShape_, MatrixCoord const& tileMN_, uint32_t coreNum_ = 0)
        : problemShape(problemShape_), tileMN(tileMN_), coreNum(coreNum_)
    {
        uint32_t tileCount = CeilDiv(problemShape.m(), tileMN.row());
        triangleTasks = tileCount * (tileCount + 1) / 2;
        mainTasks = triangleTasks;
        tailSplit = 1;
        if (coreNum > 1 && triangleTasks > coreNum) {
            tailTiles = triangleTasks % coreNum;
            mainTasks = triangleTasks - tailTiles;
            if (tailTiles > 0) {
                uint32_t parts = coreNum / tailTiles;
                uint32_t minRows = tileMN.row();
                for (uint32_t i = 0; i < tailTiles; ++i) {
                    uint32_t tileRow = 0;
                    uint32_t tileCol = 0;
                    Locate(mainTasks + i, tileRow, tileCol);
                    minRows = Min(minRows, TileRows(tileRow));
                }
                uint32_t maxParts = minRows / kRowAlign;
                if (maxParts == 0) {
                    maxParts = 1;
                }
                tailSplit = Min(parts, maxParts);
                if (tailSplit <= 1) {
                    tailSplit = 1;
                    mainTasks = triangleTasks;
                    tailTiles = 0;
                }
            }
        }
        tasksPerBatch = mainTasks + tailTiles * tailSplit;
    }

    CATLASS_HOST_DEVICE
    uint32_t GetCoreLoops() const { return tasksPerBatch; }

    CATLASS_HOST_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx) const { return tasksPerBatch == 0 ? 0 : taskIdx / tasksPerBatch; }

    CATLASS_HOST_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx) const
    {
        uint32_t tileRow = 0;
        uint32_t tileCol = 0;
        uint32_t rowInTile = 0;
        uint32_t rows = 0;
        Decode(taskIdx % tasksPerBatch, tileRow, tileCol, rowInTile, rows);
        return GemmCoord{tileRow, tileCol, 0};
    }

    CATLASS_HOST_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord const& blockCoord) const
    {
        return GemmCoord{TileRows(blockCoord.m()), TileCols(blockCoord.n()), problemShape.k()};
    }

    CATLASS_HOST_DEVICE
    GemmCoord GetTaskShape(uint32_t taskIdx) const
    {
        uint32_t tileRow = 0;
        uint32_t tileCol = 0;
        uint32_t rowInTile = 0;
        uint32_t rows = 0;
        Decode(taskIdx % tasksPerBatch, tileRow, tileCol, rowInTile, rows);
        return GemmCoord{rows, TileCols(tileCol), problemShape.k()};
    }

    CATLASS_HOST_DEVICE
    MatrixCoord GetElementOffset(uint32_t taskIdx) const
    {
        uint32_t tileRow = 0;
        uint32_t tileCol = 0;
        uint32_t rowInTile = 0;
        uint32_t rows = 0;
        Decode(taskIdx % tasksPerBatch, tileRow, tileCol, rowInTile, rows);
        return MatrixCoord{tileRow * tileMN.row() + rowInTile, tileCol * tileMN.column()};
    }

    CATLASS_HOST_DEVICE
    uint32_t TileRows(uint32_t tileRow) const
    {
        uint32_t start = tileRow * tileMN.row();
        return start >= problemShape.m() ? 0 : Min(tileMN.row(), problemShape.m() - start);
    }

    CATLASS_HOST_DEVICE
    uint32_t TileCols(uint32_t tileCol) const
    {
        uint32_t start = tileCol * tileMN.column();
        return start >= problemShape.n() ? 0 : Min(tileMN.column(), problemShape.n() - start);
    }

    CATLASS_HOST_DEVICE
    uint32_t TileCount() const { return CeilDiv(problemShape.m(), tileMN.row()); }

    CATLASS_HOST_DEVICE
    void Locate(uint32_t triangleIdx, uint32_t& tileRow, uint32_t& tileCol) const
    {
        uint32_t tileCount = TileCount();
        uint32_t remaining = triangleIdx;
        for (uint32_t col = 0; col < tileCount; ++col) {
            uint32_t length = tileCount - col;
            if (remaining < length) {
                tileCol = col;
                tileRow = (col % 2 == 0) ? (col + remaining) : (tileCount - 1 - remaining);
                return;
            }
            remaining -= length;
        }
        tileRow = 0;
        tileCol = 0;
    }

    CATLASS_HOST_DEVICE
    void Decode(uint32_t index, uint32_t& tileRow, uint32_t& tileCol, uint32_t& rowInTile, uint32_t& rows) const
    {
        uint32_t triangleIdx = index;
        uint32_t part = 0;
        if (tailSplit > 1 && index >= mainTasks) {
            uint32_t tailIndex = index - mainTasks;
            triangleIdx = mainTasks + tailIndex / tailSplit;
            part = tailIndex % tailSplit;
        }
        Locate(triangleIdx, tileRow, tileCol);
        uint32_t tileRows = TileRows(tileRow);
        if (tailSplit > 1 && index >= mainTasks) {
            SplitRows(tileRows, tailSplit, part, rowInTile, rows);
        } else {
            rowInTile = 0;
            rows = tileRows;
        }
    }

    CATLASS_HOST_DEVICE
    static void SplitRows(uint32_t length, uint32_t parts, uint32_t part, uint32_t& offset, uint32_t& size)
    {
        uint32_t quanta = length / kRowAlign;
        uint32_t leftover = length % kRowAlign;
        uint32_t base = quanta / parts;
        uint32_t extra = quanta % parts;
        uint32_t mine = base + (part < extra ? 1u : 0u);
        uint32_t before = part * base + (part < extra ? part : extra);
        offset = before * kRowAlign;
        size = mine * kRowAlign;
        if (part + 1 == parts) {
            size += leftover;
        }
    }
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_SYRK_H
