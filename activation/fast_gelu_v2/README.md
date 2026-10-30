# FastGeluV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     √    |
|  <term>Atlas推理系列产品</term>     |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：计算FastGeluV2激活函数。
- 计算公式：

$$
FastGeluV2(x_i) = x_i \times (sgn(x_i) \times [-0.1444 \times (clip(0.7071 \times |x_i|,\ max=1.769) - 1.769)^2 + 0.5] + 0.5)
$$

其中：

- x: 输入张量
- $sgn(x) = (x + 0.000000000001) / |x + 0.000000000001|$
- $clip(v, max)$ 表示将v的取值限制在max以内

## 参数说明

<table style="table-layout: auto; width: 100%">
<thead>
    <tr>
    <th style="white-space: nowrap">参数名</th>
    <th style="white-space: nowrap">输入/输出</th>
    <th style="white-space: nowrap">描述</th>
    <th style="white-space: nowrap">数据类型</th>
    <th style="white-space: nowrap">数据格式</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>x</td>
    <td>输入</td>
    <td>公式中的'x'，表示FastGeluV2激活函数的输入张量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>表示x经FastGeluV2计算得到的输出张量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式  | [test_geir_fast_gelu_v2.cpp](examples/arch35/test_geir_fast_gelu_v2.cpp) | 通过[算子IR](./op_graph/fast_gelu_v2_proto.h)以GE IR图模式方式调用FastGeluV2算子。 |
