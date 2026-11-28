# CeluV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Activation function Celu (Continuously Differentiable Exponential Linear Units).
- Formula:

$$
CELU(x) = \max(0,x) + \min(0, \alpha \ast (\exp(x/\alpha) - 1))
$$

where:

- x: Input tensor

- alpha: Activation coefficient in the CELU formula, with the default value 1.0

## Parameters

<table style="table-layout: auto; width: 100%">
<thead>
    <tr>
    <th style="white-space: nowrap">Parameter name</th>
    <th style="white-space: nowrap">Input/Output</th>
    <th style="white-space: nowrap">Description</th>
    <th style="white-space: nowrap">Data type</th>
    <th style="white-space: nowrap">Data format</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>x</td>
    <td>Input</td>
    <td>'x' in the formula, the input tensor of the CELU activation function. Empty tensors are supported. Non-contiguous tensor inputs are automatically converted to contiguous tensors by the framework before computation.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>alpha</td>
    <td>Attribute</td>
    <td>Activation coefficient in the CELU formula, with the default value 1.0. It cannot be 0.</td>
    <td>FLOAT</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y</td>
    <td>Output</td>
    <td>Output tensor of x computed by CELU, with the shape the same as the input x. Non-contiguous tensor outputs are automatically converted to contiguous tensors by the framework before computation.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
    </tr>
</tbody></table>

- <term>Atlas training products</term>, <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>: BFLOAT16 is not supported.
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas training products</term>, <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>: alpha within the subnormal domain (|α|<1e-7) is rejected.

## Constraints

- alpha: alpha cannot be 0.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_celu.cpp](examples/test_aclnn_celu.cpp) | Calls the CeluV2 operator through the [aclnnCelu&aclnnInplaceCelu](docs/aclnnCelu&aclnnInplaceCelu_en.md) API. |
| GE graph mode | - | Calls the CeluV2 operator by constructing a graph with the [operator IR](op_graph/celu_v2_proto.h). |
