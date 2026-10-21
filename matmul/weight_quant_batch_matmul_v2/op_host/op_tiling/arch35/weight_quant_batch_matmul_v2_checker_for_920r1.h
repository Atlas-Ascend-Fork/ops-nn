/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_checker_for_920r1.h
 * \brief
 */
#pragma once
#include "../weight_quant_batch_matmul_v2_tiling.h"

namespace optiling {

class WeightQuantBatchMatmulV2CheckerFor920R1 {
public:
    WeightQuantBatchMatmulV2CheckerFor920R1(gert::TilingContext* context) : context_(context) {}

    ~WeightQuantBatchMatmulV2CheckerFor920R1() {};

    ge::graphStatus Check();

private:
    bool isA16MxFp4_ = false;
    bool isA16HiF4_ = false;
    bool isA16W4C016PerG_ = false;
    bool isNZC016_ = false;
    ge::graphStatus CheckContext();
    bool CheckDtype();
    bool CheckAttr();
    bool CheckShape();
    bool CheckInputShape(const gert::StorageShape* xShape, const gert::StorageShape* weightShape);
    bool CheckInputDimNum(const gert::StorageShape* xShape, const gert::StorageShape* weightShape);
    bool CheckInputSize(const gert::StorageShape* xShape, const gert::StorageShape* weightShape);
    bool CheckAntiQuantShape(const gert::StorageShape* antiQuantScaleShape,
                             const gert::StorageShape* antiQuantOffsetShape);
    bool CheckShapeAdvanced();
    gert::TilingContext* context_ = nullptr;
    WeightQuantBatchMatmulInfo inputParams_;
};
} // namespace optiling
