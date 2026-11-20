# ApplyAdagrad

English | [简体中文](./README.md)

<!-- codespell:ignore accum adagrad -->

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

> Starting from Ascend 950 this operator is succeeded by
> [InplaceApplyAdagrad](../inplace_apply_adagrad/README_en.md); only the GEIR prototype is retained and the GEIR path
> actually invokes the InplaceApplyAdagrad implementation.

## Function Description

- **Operator function**: performs a single Adagrad optimizer parameter update. The learning rate of each parameter is
  adapted to the current gradient, updating the parameter `var` and the squared-gradient accumulator `accum`.
- Calculation formula:

  $$
  \begin{aligned}
  accum_{new} &=
  \begin{cases}
  accum + grad^2, & \text{update\_slots} = true \\
  accum, & \text{update\_slots} = false
  \end{cases} \\
  var_{new} &= var - lr \times \frac{grad}{\sqrt{accum_{new}}}
  \end{aligned}
  $$

- **Implementation**: this operator registers only the operator IR (`op_graph/apply_adagrad_proto.h`) and the graph
  inference, providing no dedicated op_host tiling or op_kernel implementation. During GEIR graph compilation the
  `ApplyAdagradToInplaceFusionPass` (`op_graph/fusion_pass/`, a DecomposePass that rewrites nodes by operator type)
  rewrites each `ApplyAdagrad` node in the graph into an `InplaceApplyAdagrad` node, so the computation, tiling and
  AICore kernel are all reused from [InplaceApplyAdagrad](../inplace_apply_adagrad/README_en.md):
  - `var`/`accum`/`lr`/`grad` and `update_slots`/`use_locking` are forwarded to `InplaceApplyAdagrad` unchanged;
  - the `ApplyAdagrad` output `var` maps to the `InplaceApplyAdagrad` output `var`; `accum` is updated in place by
    `InplaceApplyAdagrad`.

## Parameters

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 600px">
<col style="width: 220px">
<col style="width: 170px">
</colgroup>
<thead>
<tr>
<th>Parameter</th>
<th>Input/Output/Attribute</th>
<th>Description</th>
<th>Data Type</th>
<th>Format</th>
</tr>
</thead>
<tbody>
<tr>
<td>var</td>
<td>Input / Output</td>
<td>Parameter tensor to update; should come from a Variable. The output var shares device memory with the input var.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>accum</td>
<td>Input</td>
<td>Squared-gradient accumulator; should come from a Variable. Must match the shape/dtype of var; updated in place by the operator.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>lr</td>
<td>Input</td>
<td>Learning rate. Supports a rank-0 scalar or a {1}-shaped single-element tensor, same dtype as var.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>grad</td>
<td>Input</td>
<td>Current gradient tensor. Must match the shape/dtype of var.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>update_slots</td>
<td>Attribute</td>
<td>Whether to update accum, default true. When false, accum stays unchanged.</td>
<td>BOOL</td>
<td>-</td>
</tr>
<tr>
<td>use_locking</td>
<td>Attribute</td>
<td>Whether to lock during update, default false, compatibility only.</td>
<td>BOOL</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- No aclnn interface is provided; only GE graph mode invocation is supported.
- Support relationship: on Ascend 950PR&950DT products, `ApplyAdagrad` nodes in GEIR graphs are rewritten into
  `InplaceApplyAdagrad` by this repository's `ApplyAdagradToInplaceFusionPass` during graph compilation (arch35
  implementation; inputs and attributes are forwarded, the output var is mapped back to this operator, and accum is
  updated in place); other products are served by the built-in `ApplyAdagrad` implementations of their environments.
- On Ascend 950, TensorFlow `ApplyAdagrad` / `ResourceApplyAdagrad` are mapped directly to InplaceApplyAdagrad at the
  adapter layer by [inplace_apply_adagrad_tf_plugin.cpp](../inplace_apply_adagrad/framework/inplace_apply_adagrad_tf_plugin.cpp),
  which is independent of the GEIR rewrite path of this operator; other products go through the built-in TF adapter.

## Invocation Description

| Invocation method | Sample code | Description |
|:---------|:---------|:-----|
| GE graph mode | [test_geir_apply_adagrad.cpp](examples/test_geir_apply_adagrad.cpp) | Calls the ApplyAdagrad operator by building the graph through the [operator IR](op_graph/apply_adagrad_proto.h). |
