# ThresholdGradV2

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

- 算子功能：完成threshold正向操作的反向传播梯度计算。

- 计算公式：

  $$
      out_i =
      \begin{cases}
      gradients_i, \quad features_i > threshold_i\\
      0,  \quad features_i \leq threshold_i
      \end{cases}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1430px">
  <colgroup>
    <col style="width: 127px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
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
      <td>gradients</td>
      <td>输入</td>
      <td>上游梯度，与features、threshold广播兼容。</td>
      <td>FLOAT16、FLOAT、BF16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>features</td>
      <td>输入</td>
      <td>正向输入，决定门控掩码。</td>
      <td>FLOAT16、FLOAT、BF16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>输入</td>
      <td>阈值tensor，与gradients、features广播兼容。</td>
      <td>FLOAT16、FLOAT、BF16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops</td>
      <td>输出</td>
      <td>反向梯度，dtype与gradients一致，shape为三者广播后形状。</td>
      <td>FLOAT16、FLOAT、BF16、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- gradients、features、threshold、backprops四者dtype一致。
- gradients、features、threshold三者shape需满足broadcast关系，backprops的shape为广播后的形状。

## 调用说明

| 调用方式   | 调用样例                                                | 说明                            |
| :--------- | :------------------------------------------------------ | :------------------------------ |
| GEIR调用 | [test_geir_threshold_grad_v2.cpp](./examples/test_geir_threshold_grad_v2.cpp) | 通过GE IR图方式调用threshold_grad_v2算子。 |
