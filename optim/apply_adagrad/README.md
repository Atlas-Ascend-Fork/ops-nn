# ApplyAdagrad

<!-- codespell:ignore accum adagrad -->

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     √    |
|  <term>Atlas推理系列产品</term>    |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

> 自 Ascend 950 起本算子由 [InplaceApplyAdagrad](../inplace_apply_adagrad/README.md) 替代，当前仅保留GEIR原型，GEIR通路实际调用InplaceApplyAdagrad的实现。

## 功能说明

- **算子功能**：执行Adagrad优化器的单步参数更新。根据当前梯度自适应调整每个参数的学习率，更新模型参数 `var` 与梯度平方累积量 `accum`。
- 计算公式：

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

- **实现方式**：本算子仅注册算子IR（`op_graph/apply_adagrad_proto.h`）与图推导，不提供独立的 op_host tiling 与 op_kernel 实现。GEIR 图编译时由 `ApplyAdagradToInplaceFusionPass`（`op_graph/fusion_pass/`，DecomposePass 机制，按算子类型匹配改写）将图中 `ApplyAdagrad` 节点改写为 `InplaceApplyAdagrad` 节点执行，计算、tiling 与 AICore kernel 均复用 [InplaceApplyAdagrad](../inplace_apply_adagrad/README.md)：
  - `var`/`accum`/`lr`/`grad` 与 `update_slots`/`use_locking` 原样传递给 `InplaceApplyAdagrad`；
  - `ApplyAdagrad` 的输出 `var` 对应 `InplaceApplyAdagrad` 的输出 `var`；`accum` 由 `InplaceApplyAdagrad` 原地更新。

## 参数说明

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
<th>参数名</th>
<th>输入/输出/属性</th>
<th>描述</th>
<th>数据类型</th>
<th>数据格式</th>
</tr>
</thead>
<tbody>
<tr>
<td>var</td>
<td>输入 / 输出</td>
<td>待更新的参数张量，应来自Variable。输出var与输入var共享Device内存。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>accum</td>
<td>输入</td>
<td>梯度平方累积量，应来自Variable。shape/dtype必须与var一致，由算子原地更新。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>lr</td>
<td>输入</td>
<td>学习率。支持rank-0标量或shape为{1}的单元素Tensor，dtype与var一致。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>grad</td>
<td>输入</td>
<td>当前梯度Tensor。shape/dtype必须与var一致。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>update_slots</td>
<td>属性</td>
<td>是否更新accum，默认true。为false时accum保持不变。</td>
<td>BOOL</td>
<td>-</td>
</tr>
<tr>
<td>use_locking</td>
<td>属性</td>
<td>是否在更新时加锁，默认false，仅作兼容属性。</td>
<td>BOOL</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- 本算子不提供aclnn对外接口，仅支持GE图模式调用。
- 支持关系：Ascend 950PR&950DT系列产品上，GEIR图中的 `ApplyAdagrad` 节点由本仓 `ApplyAdagradToInplaceFusionPass` 在图编译期改写为 `InplaceApplyAdagrad` 执行（arch35 实现，输入/属性透传、输出var映射回本算子、accum原地更新）；其余产品由各自环境的内置 `ApplyAdagrad` 实现支持。
- TensorFlow `ApplyAdagrad` / `ResourceApplyAdagrad` 的接入在 Ascend 950 上由 [inplace_apply_adagrad_tf_plugin.cpp](../inplace_apply_adagrad/framework/inplace_apply_adagrad_tf_plugin.cpp) 在适配层直接映射为InplaceApplyAdagrad，与本算子的GEIR改写通路互不影响；其余产品走内置TF适配。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:---------|:---------|:-----|
| GE图模式 | [test_geir_apply_adagrad.cpp](examples/test_geir_apply_adagrad.cpp) | 通过 [算子IR](op_graph/apply_adagrad_proto.h) 构图方式调用ApplyAdagrad算子。 |
