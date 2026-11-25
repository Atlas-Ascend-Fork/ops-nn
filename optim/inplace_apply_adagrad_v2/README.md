# InplaceApplyAdagradV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     ×    |
|  <term>Atlas A2系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- **算子功能**：执行Adagrad V2优化器的单步参数更新。通过累积历史梯度的平方和来自适应调整每个参数的学习率，V2相比V1在分母添加epsilon参数提供数值稳定性保护。Inplace语义，var和accum输出与输入共享Device内存。

- **计算公式**：

  $$
  \begin{aligned}
  accum\_new &= accum + grad^2 \\
  var\_new   &= var - lr \times \frac{grad}{\sqrt{accum\_new} + \epsilon}
  \end{aligned}
  $$

  其中`lr`为学习率，`epsilon`用于数值稳定性的小常数，`accum`更新受update_slots参数控制：当update_slots=false时，accum_new = accum（不更新累积和）。

## 参数说明

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>var</td>
    <td>输入/输出(inplace)</td>
    <td>待更新的参数张量，应来自Variable。对应公式中的var。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>输入/输出(inplace)</td>
    <td>梯度平方累积和，应来自Variable。对应公式中的accum。shape/dtype必须与var一致。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率，对应公式中的lr。<strong>单元素Tensor</strong>。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>输入</td>
    <td>用于数值稳定性的小常数，对应公式中的epsilon，加在sqrt之外。<strong>单元素Tensor</strong>。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>当前梯度Tensor，对应公式中的grad。shape/dtype 必须与var一致。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>update_slots</td>
    <td>属性</td>
    <td>是否更新accum。默认true。当设为false时，accum保持不变。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- 输入张量var、accum、grad必须具有相同的形状和数据类型
- `lr`、`epsilon`须为单元素Tensor

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式 | [test_geir_inplace_apply_adagrad_v2](./examples/arch35/test_geir_inplace_apply_adagrad_v2.cpp) | 通过 [算子IR](./op_graph/inplace_apply_adagrad_v2_proto.h) 构图方式调用 InplaceApplyAdagradV2 算子。 |
