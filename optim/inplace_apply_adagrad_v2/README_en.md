# InplaceApplyAdagradV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- **Operator function**: Perform a single-step parameter update of the Adagrad V2 optimizer. The operator adaptively adjusts the learning rate of each parameter by accumulating the sum of squares of historical gradients. Compared with V1, V2 adds the epsilon parameter in the denominator for numerical stability protection. The operator has inplace semantics: the var and accum outputs share device memory with the inputs.

- **Formula**:

  $$
  \begin{aligned}
  accum\_new &= accum + grad^2 \\
  var\_new   &= var - lr \times \frac{grad}{\sqrt{accum\_new} + \epsilon}
  \end{aligned}
  $$

  Here `lr` is the learning rate, and `epsilon` is a small constant for numerical stability. The update of `accum` is controlled by the update_slots parameter: when update_slots=false, accum_new = accum (the accumulated sum is not updated).

## Parameters

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
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
    <td>var</td>
    <td>Input/Output (inplace)</td>
    <td>Parameter tensor to be updated. It should come from Variable and corresponds to var in the formula.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>Input/Output (inplace)</td>
    <td>Accumulated sum of squared gradients. It should come from Variable and corresponds to accum in the formula. The shape/dtype must be the same as those of var.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Input</td>
    <td>Learning rate, corresponding to lr in the formula. <strong>A single-element tensor</strong>.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>Input</td>
    <td>Small constant for numerical stability, corresponding to epsilon in the formula, added outside the sqrt. <strong>A single-element tensor</strong>.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>Input</td>
    <td>Current gradient tensor, corresponding to grad in the formula. The shape/dtype must be the same as those of var.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>update_slots</td>
    <td>Attribute</td>
    <td>Indicates whether to update accum. Default value: true. When this parameter is set to false, accum remains unchanged.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- Input tensors var, accum, and grad must have the same shape and data type.
- `lr` and `epsilon` must be single-element tensors.

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph mode | [test_geir_inplace_apply_adagrad_v2](./examples/arch35/test_geir_inplace_apply_adagrad_v2.cpp) | Invoke the InplaceApplyAdagradV2 operator through graph construction using the [operator IR](./op_graph/inplace_apply_adagrad_v2_proto.h). |
