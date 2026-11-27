# MseLoss

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Computes the mean squared error between each element of the input and the target.

- Formula:

  When `reduction` is `none`:

  $$
    \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
    l_n = \left( x_n - y_n \right)^2,
  $$

  where `x` is predict, `y` is label, and `N` is the number of elements. If `reduction` is not `none`, then:

  $$
    \ell(x, y) =
    \begin{cases}
        \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
    \end{cases}
  $$

## Parameters

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>predict</td>
      <td>Input</td>
      <td>Input x in the formula, representing the predicted value.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>Input</td>
      <td>Input y in the formula, representing the target value.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output ℓ(x,y) in the formula. When reduction is none, the shape is the same as that of the broadcast result of predict and label; when reduction is mean or sum, it is a 0-dimensional scalar.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Optional attribute</td>
      <td>Specifies the reduction to apply to the output. The value can be none, mean, or sum, and the default value is "mean". none indicates that no reduction is applied, mean indicates that the sum of the output is divided by the number of elements in the output, and sum indicates that the output is summed.</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

- <term>Atlas inference products</term>, <term>Atlas training products</term>: BFLOAT16 is not supported by the aclnn APIs.

## Constraints

None

## Invocation Description

| Invocation method | Sample code | Description |
|--------------|--------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_mse_loss](./examples/arch35/test_aclnn_mse_loss.cpp) | Invokes the MseLoss operator through the [aclnnMseLoss](./docs/aclnnMseLoss_en.md) interface. |
| aclnn API | [test_aclnn_mse_loss_out](./examples/arch35/test_aclnn_mse_loss_out.cpp) | Invokes the MseLoss operator through the [aclnnMseLossOut](./docs/aclnnMseLossOut_en.md) interface. |
| GE graph mode | - | Invokes the MseLoss operator by building a graph with the [operator IR](./op_graph/mse_loss_proto.h). |
