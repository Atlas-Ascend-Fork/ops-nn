/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file einsum_infershape.cpp
 * \brief Einsum infershape 2.0：equation解析、省略号(...)广播、未知维(-1)传播
 *        （自canndev op_proto/matrix_calculation_einsum_gemm.cc迁移；gert无shape range接口，
 *        未知维按-1传播统一处理，原range交集逻辑不再需要）
 */

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t EINSUM_EQUATION_IDX = 0; // attr: equation(String)
constexpr size_t EINSUM_Y_IDX = 0;        // output: y
constexpr size_t EINSUM_MAX_INPUT_NUM = 2;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t kEinsumOffsetLength = 3; // "..."

bool IsEllipsis(const std::string& oriStr) { return oriStr.find("...") != std::string::npos; }

void TrimWhitespace(std::string& s)
{
    size_t index = 0;
    if (!s.empty()) {
        while ((index = s.find(' ', index)) != std::string::npos) {
            s.erase(index, 1);
        }
    }
}

bool SplitEquation(const std::string& eqn, std::string& inEqu, std::string& outEqu)
{
    size_t pos = eqn.find("->");
    if (pos == std::string::npos) {
        return false;
    }
    inEqu = eqn.substr(0, pos);
    outEqu = eqn.substr(pos + 2);
    return true;
}

void GetInSplitEquationList(const std::string& inEqu, std::vector<std::string>& inEquList)
{
    std::stringstream equStream(inEqu);
    std::string term;
    while (!equStream.eof()) {
        std::getline(equStream, term, ',');
        inEquList.push_back(term);
    }
}

// 建立输入标签与维度的映射；同一标签在不同输入出现时维度必须一致（未知维-1可与任意值相容）
bool InsertLabelToMap(const char label, const int64_t shape, std::map<char, int64_t>& equMap)
{
    if (!isalpha(label)) {
        OP_LOGE("Einsum", "Input label should be A-Za-z, which is [%c].", label);
        return false;
    }
    auto it = equMap.find(label);
    if (it == equMap.end()) {
        equMap[label] = shape;
        OP_LOGD("Einsum", "The shape of input label [%c] is [%ld].", label, shape);
        return true;
    }
    if (it->second != shape && it->second != UNKNOWN_DIM && shape != UNKNOWN_DIM) {
        OP_LOGE("Einsum", "Input label [%c] has different shape, which is [%ld] and [%ld].", label, it->second, shape);
        return false;
    }
    if (it->second == UNKNOWN_DIM && shape != UNKNOWN_DIM) {
        it->second = shape;
    }
    return true;
}

// 含省略号场景：省略号对应维收集进集合，其余标签逐项映射
bool MapWithEllipsis(const std::string& equTemp, const std::vector<int64_t>& equTensorTemp,
                     std::map<char, int64_t>& equMap, std::set<std::vector<int64_t>>& ellipsisSet)
{
    int64_t equTempSize = static_cast<int64_t>(equTemp.size());
    int64_t equTensorTempSize = static_cast<int64_t>(equTensorTemp.size());
    int64_t ellSize = equTensorTempSize + kEinsumOffsetLength - equTempSize;
    std::vector<int64_t> ellList;

    auto startIndex = equTemp.find_first_of(".");
    for (size_t i = 0; i < startIndex; i++) {
        if (!InsertLabelToMap(equTemp[i], equTensorTemp[i], equMap)) {
            return false;
        }
    }
    if (ellSize > 0) {
        // 各输入省略号维数必须一致：与已登记集合比对须在插入前进行；插入后begin()可能恰好指向
        // 新元素自身，用其size自比对会漏判，后续按begin()->size()遍历将产生越界读
        if (!ellipsisSet.empty() && ellSize != static_cast<int64_t>(ellipsisSet.begin()->size())) {
            return false;
        }
        for (int64_t j = 0; j < ellSize; j++) {
            ellList.push_back(equTensorTemp[static_cast<size_t>(j) + startIndex]);
        }
        ellipsisSet.insert(ellList);
    }
    for (int64_t k = static_cast<int64_t>(startIndex); k < (equTensorTempSize - ellSize); k++) {
        size_t equTempIdx = static_cast<size_t>(k) + static_cast<size_t>(kEinsumOffsetLength);
        size_t equTensorTempIdx = static_cast<size_t>(k) + static_cast<size_t>(ellSize);
        if (!InsertLabelToMap(equTemp[equTempIdx], equTensorTemp[equTensorTempIdx], equMap)) {
            return false;
        }
    }
    return true;
}

// 不含省略号场景
bool MapWithoutEllipsis(const std::string& equTemp, const std::vector<int64_t>& equTensorTemp,
                        std::map<char, int64_t>& equMap)
{
    for (size_t i = 0; i < equTemp.size(); i++) {
        if (!InsertLabelToMap(equTemp[i], equTensorTemp[i], equMap)) {
            return false;
        }
    }
    return true;
}

// 输出含省略号：省略号位置展开为广播后的省略号维，其余标签取映射维度
void OutputWithEllipsis(std::vector<int64_t>& outputShape, const std::string& outEqu,
                        const std::map<char, int64_t>& equMap, const std::vector<int64_t>& ellList)
{
    size_t startIndex = outEqu.find_first_of(".");
    for (size_t i = 0; i < startIndex; i++) {
        outputShape.push_back(equMap.at(outEqu[i]));
    }
    for (size_t i = 0; i < ellList.size(); i++) {
        outputShape.push_back(ellList[i]);
    }
    for (size_t i = (startIndex + static_cast<size_t>(kEinsumOffsetLength)); i < outEqu.size(); i++) {
        outputShape.push_back(equMap.at(outEqu[i]));
    }
}

// 输出不含省略号
void OutputWithoutEllipsis(std::vector<int64_t>& outputShape, const std::string& outEqu,
                           const std::map<char, int64_t>& equMap)
{
    for (size_t i = 0; i < outEqu.size(); i++) {
        outputShape.push_back(equMap.at(outEqu[i]));
    }
}

// "..."/重复标签合法性；重复标签是否出现在输入或输出
bool CheckEllipsisAndDuplicatedLabel(std::string partEqn, bool& strideFlag)
{
    size_t startIndex = partEqn.find("...");
    if (startIndex != std::string::npos) {
        partEqn.erase(startIndex, static_cast<size_t>(kEinsumOffsetLength));
        // 方程中只能出现一个省略号
        if (partEqn.find_first_of(".") != std::string::npos) {
            return false;
        }
    }
    if (!strideFlag) {
        std::set<char> labelSet(partEqn.begin(), partEqn.end());
        strideFlag = (labelSet.size() != partEqn.size());
    }
    return true;
}

bool CheckEquation(const char* opName, const std::string& eqn, size_t inputNum, std::vector<std::string>& inEquList,
                   std::string& outEqu)
{
    std::string inEqu;
    bool inputStrideFlag = false;
    bool outputInflateFlag = false;
    if (!SplitEquation(eqn, inEqu, outEqu)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "equation", eqn.c_str(),
                                              "the value must contain \"->\" to split input and output terms");
        return false;
    }

    GetInSplitEquationList(inEqu, inEquList);
    for (auto& inEquTmp : inEquList) {
        OP_LOGD(opName, "Input equation is [%s].", inEquTmp.c_str());
        if (inEquTmp.empty()) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "equation", eqn.c_str(),
                                                  "the value must not contain an empty input term");
            return false;
        }
        if (!CheckEllipsisAndDuplicatedLabel(inEquTmp, inputStrideFlag)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "equation", eqn.c_str(),
                                                  "the value must contain at most one \"...\" per input term");
            return false;
        }
    }

    if (!CheckEllipsisAndDuplicatedLabel(outEqu, outputInflateFlag)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "equation", eqn.c_str(),
                                              "the value must contain at most one \"...\" in the output term");
        return false;
    }

    // "aab, bc->aac"这类输入输出同时带重复标签的方程非法
    if (outputInflateFlag && inputStrideFlag) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName, "equation", eqn.c_str(),
            "the value must not contain duplicated labels in both the input and output terms");
        return false;
    }

    if (inEquList.size() != inputNum) {
        const std::string reason = "the value must be " + std::to_string(inEquList.size()) +
                                   " to match the input terms of the equation";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "N", std::to_string(inputNum).c_str(), reason.c_str());
        return false;
    }
    return true;
}

// 省略号维广播：对应位取已知维度的最大值；1与任意维广播；未知维-1与任意维广播结果仍未知
// （-1传播到输出，由运行时拿到真实值后再校验广播合法性；不能用max把-1与1的组合固化成1，
// 也不能断言-1与k(≠1)的结果为k——编译期无法保证图运行时合法，下游按动态shape处理更安全）
// 集合内各向量长度由MapWithEllipsis在插入前统一校验；本函数入口再做等长兜底校验，
// 循环体内对每个元素显式做边界判断，彻底杜绝vector裸下标越界读（UB）
bool GetEllipsisRes(const std::set<std::vector<int64_t>>& ellipsisSet, std::vector<int64_t>& ellipsisOut)
{
    if (ellipsisSet.empty()) {
        return true;
    }
    // 兜底校验各输入省略号维数一致（MapWithEllipsis插入前已保证，此处防御性冗余）：
    // 长度基准取集合首元素而非字典序begin()——二者在等长前提下等价，等长性由本循环自身建立
    const size_t vecSize = ellipsisSet.begin()->size();
    for (const auto& vecItem : ellipsisSet) {
        if (vecItem.size() != vecSize) {
            return false;
        }
    }
    for (size_t i = 0; i < vecSize; i++) {
        int64_t shapeTmp = ellipsisSet.begin()->at(i);
        for (const auto& vecItem : ellipsisSet) {
            // 显式边界判断：等长校验失效时走报错分支而非UB
            if (i >= vecItem.size()) {
                return false;
            }
            if (vecItem[i] == UNKNOWN_DIM || shapeTmp == UNKNOWN_DIM) {
                shapeTmp = UNKNOWN_DIM;
            } else if (vecItem[i] == 1 || shapeTmp == 1) {
                shapeTmp = std::max(vecItem[i], shapeTmp);
            } else if (vecItem[i] != shapeTmp) {
                return false;
            }
        }
        ellipsisOut.push_back(shapeTmp);
    }
    OP_LOGD("Einsum", "The output ... shape size is [%zu].", ellipsisOut.size());
    return true;
}

// 输出项中的标签必须都在输入项中出现过
bool CheckOutputNewLabel(const std::string& outEqu, const std::map<char, int64_t>& equMap)
{
    std::string outEquBak(outEqu);
    size_t ellipsisPos = outEquBak.find("...");
    if (ellipsisPos != std::string::npos) {
        outEquBak.erase(ellipsisPos, static_cast<size_t>(kEinsumOffsetLength));
    }
    for (auto i : outEquBak) {
        if (equMap.find(i) == equMap.end()) {
            return false;
        }
    }
    return true;
}

bool EinsumComputeOutputShape(const char* opName, const std::string& eqn,
                              const std::vector<std::vector<int64_t>>& tensorList, std::vector<int64_t>& outputShape)
{
    std::map<char, int64_t> equMap;
    std::vector<int64_t> ellipsisOut;
    std::set<std::vector<int64_t>> ellipsisSet;
    std::vector<std::string> inEquList;
    std::string outEqu;

    if (!CheckEquation(opName, eqn, tensorList.size(), inEquList, outEqu)) {
        OP_LOGE(opName, "Equation is wrong.");
        return false;
    }

    for (size_t i = 0; i < inEquList.size(); i++) {
        std::string equTemp = inEquList[i];
        auto equTempSize = equTemp.size();
        std::vector<int64_t> equTensorTemp = tensorList[i];
        auto equTensorTempSize = equTensorTemp.size();
        if (IsEllipsis(equTemp)) {
            // "...ab, bc->...ac"且tensor只比equation少3维以上为非法
            if (equTensorTempSize + static_cast<size_t>(kEinsumOffsetLength) < equTempSize) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x", "",
                                                      "the input tensor size does not match the "
                                                      "equation term");
                return false;
            }
            if (!MapWithEllipsis(equTemp, equTensorTemp, equMap, ellipsisSet)) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x", "", "the input label and shape do not match");
                return false;
            }
        } else {
            if (equTensorTempSize != equTempSize) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x", "",
                                                      "the input tensor size does not match the "
                                                      "equation term");
                return false;
            }
            if (!MapWithoutEllipsis(equTemp, equTensorTemp, equMap)) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x", "", "the input label and shape do not match");
                return false;
            }
        }
    }

    if (!CheckOutputNewLabel(outEqu, equMap)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName, "equation", eqn.c_str(),
            "the value must not contain output labels that do not appear in the input terms");
        return false;
    }

    if (!GetEllipsisRes(ellipsisSet, ellipsisOut)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x", "", "the shapes corresponding to \"...\" do not match");
        return false;
    }

    if (IsEllipsis(outEqu)) {
        OutputWithEllipsis(outputShape, outEqu, equMap, ellipsisOut);
    } else {
        if (!ellipsisOut.empty()) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "equation", eqn.c_str(),
                                                  "the value must contain \"...\" in the output term");
            return false;
        }
        OutputWithoutEllipsis(outputShape, outEqu, equMap);
    }
    return true;
}

static ge::graphStatus InferShapeForEinsum(gert::InferShapeContext* context)
{
    const char* opName = context->GetNodeName();
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(opName, "Get attrs failed.");
        return ge::GRAPH_FAILED;
    }
    const char* equationPtr = attrs->GetAttrPointer<char>(EINSUM_EQUATION_IDX);
    if (equationPtr == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "equation", "null", "the value must be a valid equation string");
        return ge::GRAPH_FAILED;
    }
    std::string equation = equationPtr;
    TrimWhitespace(equation);

    const size_t inputNum = context->GetComputeNodeInputNum();
    if (inputNum == 0 || inputNum > EINSUM_MAX_INPUT_NUM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "N", std::to_string(inputNum).c_str(),
                                              "the value must be in the range [1, 2]");
        return ge::GRAPH_FAILED;
    }

    std::vector<std::vector<int64_t>> tensorList;
    tensorList.reserve(inputNum);
    for (size_t i = 0; i < inputNum; i++) {
        const auto* shape = context->GetInputShape(i);
        if (shape == nullptr) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x", "null", "the input shape must not be null");
            return ge::GRAPH_FAILED;
        }
        std::vector<int64_t> dims;
        dims.reserve(shape->GetDimNum());
        for (size_t j = 0; j < shape->GetDimNum(); j++) {
            dims.push_back(shape->GetDim(j));
        }
        tensorList.push_back(std::move(dims));
    }

    std::vector<int64_t> outputShape;
    if (!EinsumComputeOutputShape(opName, equation, tensorList, outputShape)) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(opName, "The output shape size is [%zu].", outputShape.size());
    auto* yShape = context->GetOutputShape(EINSUM_Y_IDX);
    if (yShape == nullptr) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "y", "null", "the output shape must not be null");
        return ge::GRAPH_FAILED;
    }
    yShape->SetDimNum(outputShape.size());
    for (size_t i = 0; i < outputShape.size(); i++) {
        yShape->SetDim(i, outputShape[i]);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP(Einsum).InferShape(InferShapeForEinsum);
} // namespace ops
