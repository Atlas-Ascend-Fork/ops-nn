# SituGluGroupGrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------- | ------|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                             |    ×    |
| <term>Atlas training products</term>                              |    ×    |

## Function

- Operator function: the backward gradient computation of the SiTU Gated Linear Unit activation. Given the upstream gradient grad_y and the forward input x, it computes the gradient grad_x of x and the gradient grad_weight of weight. When the optional input weight is passed, grad_x is weighted per row and grad_weight is computed; when weight is absent, grad_weight is an empty tensor (shape [0]). When the optional input group_index is passed, only the first realBs = min(groupRows, number of rows) rows are computed, where groupRows is group_index[-1] in cumsum mode (group_list_type 0, the default) or sum(group_index) in count mode (group_list_type 1). Supported only on Ascend 950PR&950DT products.

- Calculation formula or calculation process:

  For a given input tensor x (dimensions [..., cut], where cut is even, let h = cut // 2) and upstream gradient grad_y (dimensions [..., h]):

  1. Splits x into gate and up based on activate_left (each of shape [..., h]).

  2. Computes the forward intermediates:

     $$
     t = \tanh\left(\frac{gate}{\beta}\right), \quad s = \sigma(gate), \quad situ\_a = \beta \cdot t \cdot s
     $$

     When linear_beta > 0: $up' = linear\_beta \cdot \tanh(up / linear\_beta)$, otherwise $up' = up$.

  3. Product rule:

     $$
     grad\_situ\_a = grad\_y \cdot up', \quad grad\_up' = grad\_y \cdot situ\_a
     $$

  4. Gate gradient (w is the scalar of the corresponding row of the optional input weight; w = 1 when weight is absent):

     $$
     grad\_gate = grad\_situ\_a \cdot s \cdot \left[ (1 - t^2) + \beta \cdot t \cdot (1 - s) \right] \cdot w
     $$

  5. Up gradient:

     - When linear_beta > 0: $grad\_up = grad\_up' \cdot (1 - \tanh^2(up / linear\_beta)) \cdot w$
     - When linear_beta ≤ 0: $grad\_up = grad\_up' \cdot w$

  6. Concatenates grad_gate and grad_up into grad_x based on activate_left (shape identical to x).

  7. Weight gradient (output grad_weight; an empty tensor when weight is absent), summed over h for each row n (note: not multiplied by w):

     $$
     grad\_w_n = \sum_{h} grad\_y_{n,h} \cdot situ\_a_{n,h} \cdot up'_{n,h}
     $$

  8. Row masking (optional input group_index): only the first realBs = min(groupRows, number of rows) rows are computed and written back; the remaining rows are neither read nor written. groupRows is selected by group_list_type: group_index[-1] in cumsum mode (0, the default) and sum(group_index) in count mode (1).

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
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
      <td>grad_y</td>
      <td>Input</td>
      <td>Upstream gradient of back-propagation. The size of the dim dimension is half of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Forward input. The dim dimension is evenly split into the gate and up parts. The data type must be the same as grad_y.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Optional input</td>
      <td>Per-row routing weight of the forward. The number of elements equals the product of the dimensions of x before the dim axis. Supported only on Ascend 950PR&950DT products.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>Optional input</td>
      <td>Number of tokens of each group, whose meaning is selected by group_list_type. A 1-D non-empty tensor with non-negative elements. Supported only on Ascend 950PR&950DT products.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td><ul><li>Dimension to split, which must be consistent with the forward SituGlu.</li><li>The value range is [-x.dim(), x.dim()-1].</li><li>The default value is -1.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Attribute</td>
      <td><ul><li>Scaling coefficient of the SiTU gate part, which must be consistent with the forward SituGlu.</li><li>It must be a non-zero value; passing 0 returns a parameter validation failure.</li><li>The default value is 1.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linear_beta</td>
      <td>Attribute</td>
      <td><ul><li>Scaling coefficient of the linear tanh on the up path, which must be consistent with the forward SituGlu.</li><li>The default value is 0.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>Attribute</td>
      <td><ul><li>Whether gate is the front half when x is split, which must be consistent with the forward SituGlu.</li><li>The default value is true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_list_type</td>
      <td>Attribute</td>
      <td><ul><li>How group_index expresses the group sizes.</li><li>The value can only be 0 or 1: 0 is cumsum mode, where group_index holds the cumulative row counts and the total is group_index[-1]; 1 is count mode, where group_index holds the per-group row counts and the total is sum(group_index).</li><li>The default value is 0.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>Output</td>
      <td>Gradient of the input x. The shape is exactly the same as x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_weight</td>
      <td>Output</td>
      <td>Gradient of weight: the shape is the same as weight when weight is passed, and an empty tensor (shape [0]) when weight is absent. Supported only on Ascend 950PR&950DT products.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Non-contiguous tensors are supported.
- beta, linear_beta, activate_left and dim must be consistent with the values used in the forward SituGlu call; otherwise the gradient result is incorrect.
- beta must be a non-zero value; passing 0 returns a parameter validation failure.
- Supported only on Ascend 950PR&950DT products (weight / group_index are optional inputs and grad_weight is a required output).
- When group_index is passed, rows whose index ≥ realBs in grad_x and grad_weight are not written; the caller must clear them explicitly if strict zeros are required.
- group_list_type must take a value from [0, 1]; any other value returns a parameter validation failure.
- The values of group_index must meet the following conditions, otherwise the output is not guaranteed to be as expected:
  - When group_list_type is 0, group_index must be a non-negative, monotonically non-decreasing sequence, representing the cumsum result (cumulative sum) of the size of each group after grouping. The last value must be less than or equal to the number of rows.
  - When group_list_type is 1, group_index must be a non-negative sequence, representing the size of each group after grouping. The sum of the values must be less than or equal to the number of rows.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_situ_glu_group_grad](./examples/test_aclnn_situ_glu_group_grad.cpp) | Calls the SituGluGroupGrad operator through the [aclnnSituGluGroupGrad](./docs/aclnnSituGluGroupGrad_en.md) interface (including the optional inputs weight/group_index). |
| GE graph mode | [test_geir_situ_glu_group_grad](./examples/test_geir_situ_glu_group_grad.cpp)  | Calls the SituGluGroupGrad operator by building a graph through [operator IR](./op_graph/situ_glu_group_grad_proto.h).         |
| PyTorch API |  -  | Calls the SituGluGroupGrad operator through the [situ_glu_group_grad](./docs/torchapi_situ_glu_group_grad_en.md) interface.         |
