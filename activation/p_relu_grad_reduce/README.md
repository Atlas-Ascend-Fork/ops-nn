# PReluGradReduce

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     ×    |
|  <term>Atlas A2系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：对 PReluGradUpdate 生成的中间结果 `updates` 按照 `weights` 相对 `features` 的广播维度求和，输出权重梯度 `da`。
- 计算公式：`da = ReduceSum(updates, axes)`；`axes` 为 `weights` 在 `features` 中广播扩展的维度。标量权重时对所有维度求和。
- `updates` 的元素由 PReluGradUpdate 根据 `features` 与 `grads` 计算；本算子只负责归约，不计算 `gradInput` 或生成 `updates`。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1305px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 247px">
  <col style="width: 108px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>grads</td>
      <td>输入</td>
      <td>prelu正向的输出y反向传播的梯度</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  <tr>
      <td>features</td>
      <td>输入</td>
      <td>prelu正向的输入features</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
  </tr>
  <tr>
      <td>weights</td>
      <td>输入</td>
      <td>prelu正向的输入weights</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
  </tr>
  <tr>
      <td>updates</td>
      <td>输入</td>
      <td>PReluGradUpdate 输出的权重梯度中间结果</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
  </tr>
  <tr>
      <td>da</td>
      <td>输出</td>
      <td>weights的梯度</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
  </tr>
  </tbody>
  </table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_prelu_backward](../p_relu_grad_update/examples/test_aclnn_prelu_backward.cpp) | 通过[aclnnPReluBackward](../p_relu_grad_update/docs/aclnnPreluBackward.md)接口方式调用PReluGradReduce算子。 |
