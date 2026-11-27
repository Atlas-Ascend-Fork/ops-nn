# NllLossGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Negative log likelihood loss backward.
- Formula:

$$
x\_grad[n][i] =
\begin{cases}
    \displaystyle -y\_grad[n] \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{none} \\
    \displaystyle -\frac{y\_grad}{\text{total\_weight}} \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{mean} \\
    \displaystyle -y\_grad \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{sum} \\
    0, & \text{if } i \neq \text{target}
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
    <td>x</td>
    <td>Input</td>
    <td>Input tensor x in the formulas. The shape can be (C), (N, C) or (N, C, H, W), where N indicates the batch size and C indicates the number of classes; when the shape is (N, C, H, W), H and W must be positive integers.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y_grad</td>
    <td>Input</td>
    <td>Gradient of the forward result, corresponding to y_grad in the formulas. When reduction is "none": if x is (N, C), the shape is (N); if x is (C,), the shape is (); if x is (N, C, H, W), the shape is (N, H, W). When reduction is "mean" or "sum", the shape is a single-element tensor (0D or 1D with one element). The data type is the same as x.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>target</td>
    <td>Input</td>
    <td>Represents the ground-truth label; it is target in the formulas. When x is (N, C), the shape is (N); when x is (C,), the shape is (); when x is (N, C, H, W), the shape is (N, H, W). Each element must be in [0, C-1] or equal to ignore_index.</td>
    <td>INT32, INT64, UINT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>Input</td>
    <td>Represents the scaling weight of each class; it is weight in the formulas. The shape is (C,) and the number of elements equals the number of classes C of x. The data type is the same as x.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>total_weight</td>
    <td>Input</td>
    <td>It is total_weight in the formulas. The number of elements must be 1 (shape (1,)). The data type is the same as x.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>reduction</td>
    <td>Optional attribute</td>
    <td><ul><li>Specifies the reduction to apply to the output; it is reduction in the formulas.</li><li>The value can be "none", "mean" or "sum".</li><li>The default value is "mean"; omitting the attribute is equivalent to "mean" (aclnn reduction=1).</li></ul></td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ignore_index</td>
    <td>Optional attribute</td>
    <td><ul><li>Specifies a target value that is ignored and does not affect the input gradient.</li><li>The default value is -100; omitting the attribute is equivalent to -100.</li><li>Elements in target equal to this value do not participate in the gradient computation.</li></ul></td>
    <td>INT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>x_grad</td>
    <td>Output</td>
    <td>It is x_grad[n][i] in the formulas. The shape is the same as x and the data type is the same as x.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- The value of reduction can only be "none", "mean" or "sum" (aclnn reduction=0/1/2); other values return an error.
- Each element of target must be in [0, C-1] or equal to ignore_index, where C is the number of classes of x (the size of the last dimension of x).
- The number of elements in weight must equal C.
- The number of elements in total_weight must be 1.
- The shape of y_grad varies with reduction and the shape of x: when reduction is "none", y_grad is (N) if x is (N, C), () if x is (C,), and (N, H, W) if x is (N, C, H, W); when reduction is "mean" or "sum", y_grad is a single-element tensor.
- x, y_grad, weight, total_weight and x_grad must have the same data type.
- x supports ranks of 1D (C), 2D (N, C) and 4D (N, C, H, W) only; other ranks return an error; when x is 4D, H and W must be positive integers.
- Dynamic shape support: -1 in a shape denotes an unknown dimension; unknown-rank input (marked by -2) is supported, in which case the inferred output shape is unknown rank (-2).
- Only the ND data format is supported.

## Invocation Description

| Invocation method | Sample code | Description |
|--------------|--------------|--------------|
| aclnn API | [test_aclnn_nll_loss_grad](./examples/test_aclnn_nll_loss_grad.cpp) | Invokes the NLLLossGrad operator through the [aclnnNLLLossBackward](./docs/aclnnNLLLossBackward_en.md) interface, applicable when x is (C) or (N, C). |
| aclnn API | [test_aclnn_nll_loss_grad_2d](./examples/test_aclnn_nll_loss_grad_2d.cpp) | Invokes the NLLLossGrad operator through the [aclnnNLLLoss2dBackward](./docs/aclnnNLLLoss2dBackward_en.md) interface, applicable when x is (N, C, H, W). |
| GE graph mode | - | Invokes the NLLLossGrad operator by constructing a graph with the [operator IR definition](./op_graph/nll_loss_grad_proto.h). |

The sample code is compiled and run through build.sh at the repository root, as shown below (the --soc option is optional; specifying ascend950 additionally runs the sample files under the arch35 directory):

```bash
bash build.sh --run_example nll_loss_grad eager cust --example_name=nll_loss_grad --soc=ascend950
bash build.sh --run_example nll_loss_grad eager cust --example_name=nll_loss_grad_2d --soc=ascend950
```

For more usage, see the [Quick Invocation Guide](../../docs/zh/invocation/quick_op_invocation.md).
