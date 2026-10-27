# ScaledMaskedSoftmaxV2

## 支持的产品型号

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Kirin X90处理器系列产品</term> | √ |
|  <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：将输入x先乘以scale，再把mask为true的位置置为-inf，最后沿最后一维做softmax。mask为false的位置保留缩放后的值。
- 实现说明：公式中mask为true时置为$-\infty$，实际实现填充的是对应计算精度下的数据类型最小值（dtype min），而非IEEE $-inf$。具体取值与计算路径一致：FLOAT16路径使用`float16`最小值（对应位型`0xFBFF`），FLOAT与BFLOAT16路径在float32上计算时使用`float32`最小值（对应位型`0xFF7FFFFF`）。该取值经softmax的`exp`后同样趋近于0，与置为$-\infty$在数值效果上等价。
- 计算公式：

  $$
  x_i' = -\infty, \quad mask_i = true
  $$

  $$
  x_i' = scale \times x_i, \quad mask_i = false
  $$

  $$
  y = Softmax(x', dim = -1)
  $$

  $$
  Softmax(X_i) ={e^{X_i - max(X, dim=-1)} \over \sum{e^{X_i - max(X, dim=-1)}}}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>x</td>
      <td>输入</td>
      <td>公式中的输入x。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>bool掩码。true的位置在softmax前置为-inf（实际为dtype最小值，见功能说明中的实现说明），false的位置保留scale * x。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>属性</td>
      <td>默认值为1.0。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fixedTriuMask</td>
      <td>属性</td>
      <td>仅支持false</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的y。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030处理器系列产品: 不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_scaled_masked_softmax](./examples/test_aclnn_scaled_masked_softmax.cpp) | 通过[aclnnScaledMaskedSoftmax](./docs/aclnnScaledMaskedSoftmax.md)接口方式调用aclnnScaledMaskedSoftmax算子。    |
