# MseLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：计算输入和目标中每个元素之间的均方误差。

- 计算公式：

  当`reduction`为`none`时：

  $$
    \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
    l_n = \left( x_n - y_n \right)^2,
  $$

  其中`x`是predict，`y`是label，`N`是元素个数。如果`reduction`不是`none`，那么：

  $$
    \ell(x, y) =
    \begin{cases}
        \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
    \end{cases}
  $$

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
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
      <td>predict</td>
      <td>输入</td>
      <td>公式中的输入x，表示预测值。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>输入</td>
      <td>公式中的输入y，表示目标值。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出ℓ(x,y)，reduction为none时shape与predict、label做broadcast关系后的shape一致，reduction为mean或sum时为0维标量。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>可选属性</td>
      <td>指定要应用到输出的缩减，取值为none、mean或sum，默认值为"mean"。none表示不应用缩减，mean表示输出的总和将除以输出中的元素数，sum表示输出将被求和。</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

- <term>Atlas推理系列产品</term>、<term>Atlas训练系列产品</term>：aclnn接口数据类型不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|--------------|--------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_mse_loss](./examples/arch35/test_aclnn_mse_loss.cpp) | 通过[aclnnMseLoss](./docs/aclnnMseLoss.md)接口方式调用MseLoss算子。 |
| aclnn API | [test_aclnn_mse_loss_out](./examples/arch35/test_aclnn_mse_loss_out.cpp) | 通过[aclnnMseLossOut](./docs/aclnnMseLossOut.md)接口方式调用MseLoss算子。 |
| GE图模式 | - | 通过[算子IR](./op_graph/mse_loss_proto.h)构图方式调用MseLoss算子。 |
