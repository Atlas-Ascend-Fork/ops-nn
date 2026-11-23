# SmoothL1LossV2

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

- Operator function: computes the SmoothL1 loss function.
- Formula:

  For the loss function with a batch of N, when `reduction` is none, this function is defined as:

  $$
  \ell(x,y) = L = \{l_1,\dots,l_N\}^\top
  $$

  where $l_n$ is:

  $$
  l_n = \begin{cases}
  0.5(x_n-y_n)^2/sigma, & if |x_n-y_n| < sigma \\
  |x_n-y_n| - 0.5*sigma, &  otherwise
  \end{cases}
  $$

  where sigma is the threshold attribute of this operator (the parameter of the same name on the PyTorch and aclnn interface side is beta), whose value must be non-negative; when sigma is 0, the element-wise loss degenerates to the pure L1 term $|x_n-y_n|$.

  If `reduction` is `mean` or `sum`:

  $$
  \ell(x,y)=\begin{cases}
  mean(L), & \text{if reduction} = \text{mean}\\
  sum(L), & \text{if reduction} = \text{sum}
  \end{cases}
  $$

- where:
  - Xn: predictions
  - Yn: labels/truth

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
    <td>'x' in the formula, indicating the predicted value input used to compute the SmoothL1 loss. Its data type must be the same as that of label.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>label</td>
    <td>Input</td>
    <td>'y' in the formula, indicating the true label value input that predict needs to fit. Its data type must be the same as that of predict.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>loss</td>
    <td>Output</td>
    <td>Indicates the loss value output obtained by computing predict and label through smooth_l1_loss_v2. Its data type is the same as that of predict.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sigma</td>
    <td>Optional attribute</td>
    <td><ul><li>Threshold parameter that controls the boundary between MSE and L1 loss in smooth_l1_loss_v2. The value must be non-negative.</li><li>The default value is 1.0.</li></ul></td>
    <td>FLOAT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>reduction</td>
    <td>Optional attribute</td>
    <td><ul><li>Reduction parameter that indicates performing mean/sum/no reduction on the element-wise loss values.</li><li>The value can be none, mean, or sum.</li><li>The default value is "mean".</li></ul></td>
    <td>STRING</td>
    <td>-</td>
  </tr>

</tbody></table>

## Constraints

- reduction supports only none, mean, and sum; other values are rejected.
- sigma must be non-negative (sigma >= 0); when sigma is 0, the element-wise loss degenerates to |x-y|.
- predict and label must have the same data type. In the GE graph mode path, predict and label must have the same shape (broadcast is not supported); in the aclnn path, the shapes of self and target must satisfy the broadcast relationship.
- When reduction is mean or sum, the output is a rank-0 scalar (0-dimension); when reduction is none, the output shape is the same as that of predict.
- Empty tensor semantics: in the aclnn path, when self or target is an empty tensor, the behavior differs by reduction - none returns success directly, mean fills the output with NaN, and sum fills the output with 0; in the GE graph mode path, when reduction is none and the input contains a zero-length axis, the current tiling implementation does not support it, which is a known limitation.
- Dynamic shape (unknown dimension represented by -1) and dynamic rank (represented by -2) compilation are supported; rank-0 scalar inputs are not supported in the GE graph mode path.
- Special value behavior: NaN in the input propagates element-wise to the corresponding output according to the formula; for Inf and extreme-value inputs in low-precision types such as FLOAT16 and BFLOAT16, the output may saturate to the maximum finite value of the data type (differing from the inf result of high-precision reference computation), which is a known precision behavior.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_smooth_l1_loss_v2.cpp](examples/test_aclnn_smooth_l1_loss_v2.cpp) | Invokes the SmoothL1Loss operator through the [aclnnSmoothL1Loss](docs/aclnnSmoothL1Loss_en.md) interface. The sample only demonstrates the interface invocation flow. |
