# SituGluGroup

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

- Operator function: SiTU Gated Linear Unit activation function. It splits the input tensor x along the specified dimension into the gate and up halves, and computes the output by the SiTU formula.

- Calculation formula or calculation process:

  For a given input tensor x whose dimensions are [a, b, c, d, e, f, g, ...], the operator SituGluGroup performs the following computations:

  1. Folds x based on the input parameter dim into [pre, cut], where cut must be an even number. Let h = cut // 2.

  2. Splits x into the front and back parts based on the input parameter activate_left:

     - When activate_left is true (default):

     $$
     gate = x[:, :h], \quad up = x[:, h:]
     $$

     - When activate_left is false:

     $$
     gate = x[:, h:], \quad up = x[:, :h]
     $$

  3. Performs the SiTU computation based on the input parameters beta and linear_beta:

     $$
     situ\_a = \beta \cdot \tanh\left(\frac{gate}{\beta}\right) \cdot \text{sigmoid}(gate)
     $$

     When linear_beta > 0:

     $$
     up = linear\_beta \cdot \tanh\left(\frac{up}{linear\_beta}\right)
     $$

  4. Output:

     $$
     y = situ\_a \cdot up
     $$

  5. Optional: when weight is passed, each row (indexed by the dimensions before the dim axis) is multiplied by the corresponding scalar:

     $$
     y[i] = y[i] \cdot weight[i]
     $$

  6. Optional: when group_index is passed, the number of valid rows realBs is determined by its semantics (specified by group_list_type). In count mode (group_list_type=1), realBs = min(sum(group_index), number of rows); in cumsum mode (group_list_type=0), only the last element is consumed, realBs = min(group_index[-1], number of rows) (intermediate elements are not used and monotonicity is not required). Only the first realBs rows are computed and written.

  7. Reshapes the output tensor y so that the number of dimensions is the same as that of x before folding, the size on the dim axis is half of x, and the other dimensions are the same as x.

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
      <td>x</td>
      <td>Input</td>
      <td>Input x in the formula. The dim dimension is evenly split into the gate and up parts.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td><ul><li>Dimension to split.</li><li>The value range is [-x.dim(), x.dim()-1].</li><li>The default value is -1.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Attribute</td>
      <td><ul><li>Scaling coefficient of the SiTU gate part, controlling the strength of the tanh nonlinearity.</li><li>It must be a non-zero value; passing 0 returns a parameter validation failure.</li><li>The default value is 1.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linear_beta</td>
      <td>Attribute</td>
      <td><ul><li>Scaling coefficient of the linear tanh on the up path.</li><li>When it is greater than 0, the bounding transformation is applied to up; when it is less than or equal to 0, up is passed through directly.</li><li>The default value is 0.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>Attribute</td>
      <td><ul><li>Whether gate is the front half when x is split.</li><li>true means that gate is the front half and up is the back half; false means that gate is the back half and up is the front half.</li><li>The default value is true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_list_type</td>
      <td>Attribute</td>
      <td><ul><li>Semantics of group_index.</li><li>1 is count mode (each element is the row count of that group; the sum is used); 0 is cumsum mode (each element is the cumulative end offset of that group on the row axis; only the last element is used).</li><li>The default value is 0; both 0 and 1 are supported.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Optional input</td>
      <td><ul><li>Per-row weight scalar, multiplied into the SiTU activation result.</li><li>The number of elements equals the product of the dimensions of x before the dim axis; when dim is -1, it equals the product of all dimensions except the last one.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>Optional input</td>
      <td><ul><li>Grouped token numbers whose semantics are selected by group_list_type. A 1-D non-empty tensor whose elements must be greater than or equal to 0.</li><li>count mode (1): sum(group_index) is the number of valid rows; cumsum mode (0): only the last element is used as the number of valid rows; intermediate elements are not used and monotonicity is not required.</li><li>When passed, only the first realBs rows are computed and written; the remaining rows are not written (their content is undefined), and the caller must apply masking. If realBs exceeds the row count, it is clamped to the row count.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>y in the formula, the SiTU activation result. The size of the dim dimension is half of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Non-contiguous tensors are supported.
- weight and group_index are optional inputs.
- When group_index is passed, its elements must be greater than or equal to 0: the total is sum(group_index) in count mode, and the last element in cumsum mode (intermediate elements are not used and monotonicity is not required). A zero total means realBs = 0: no row is written and the whole output is undefined. If the total exceeds the row count, realBs is clamped to the row count. Only the first realBs rows are written; the content of the remaining rows is undefined, and the caller must apply masking.
- group_list_type supports 0 (cumsum mode) and 1 (count mode); the default is 0. It has no effect when group_index is not passed; callers passing count-semantics data must explicitly pass 1. Any other value returns a parameter validation failure.
- This operator supports both Ascend 950 and Ascend 350 (they share the same arch35 implementation and have identical support).
- In count mode, sum(group_index) is accumulated with saturation at the row count and cannot overflow int64; negative elements make the operator terminate with an error.
- beta must be a non-zero value; passing 0 returns a parameter validation failure.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_situ_glu_group](./examples/test_aclnn_situ_glu_group.cpp) | Calls the SituGluGroup operator through the [aclnnSituGluGroup](./docs/aclnnSituGluGroup_en.md) interface (including the optional inputs weight/group_index). |
| GE graph mode | -  | Calls the SituGluGroup operator by building a graph through [operator IR](./op_graph/situ_glu_group_proto.h).         |
| PyTorch API |  -  | Calls the SituGluGroup operator through the [situ_glu_group](./docs/torchapi_situ_glu_group_en.md) interface.         |
