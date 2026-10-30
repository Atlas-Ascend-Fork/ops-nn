# SoftplusV2

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：Softplus激活函数，当$\beta \cdot x$不超过阈值时输出$\frac{1}{\beta}\log(1+\exp(\beta \cdot x))$，否则直接输出$x$。
- 计算公式：

$$
out = \begin{cases}
\displaystyle \frac{1}{\beta}\log(1+\exp(\beta \cdot x)), & \beta \cdot x \le threshold \\
x, & \beta \cdot x > threshold
\end{cases}
$$

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
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
</tr>
</thead>
<tbody>
<tr>
<td>x</td>
<td>输入</td>
<td>待进行Softplus计算的输入张量，公式中的x。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>输出</td>
<td>Softplus计算后的输出张量，公式中的out，shape与x一致。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>beta</td>
<td>可选属性</td>
<td>公式中的beta，默认值为1.0。</td>
<td>FLOAT</td>
<td>-</td>
</tr>
<tr>
<td>threshold</td>
<td>可选属性</td>
<td>公式中的threshold，默认值为20.0。</td>
<td>FLOAT</td>
<td>-</td>
</tr>
</tbody>
</table>

- <term>Atlas推理系列产品</term>、<term>Atlas训练系列产品</term>：数据类型不支持BFLOAT16。

## 约束说明

- 输入 x 的维度上限为 8（rank ≤ 8，与 aclnn 接口一致）；超过 8 维时 InferShape 与 Tiling/GetWorkspaceSize 均拒绝执行。
- 输入 x 的单个维度长度不超过 INT32_MAX；元素总数（各维度乘积）须在 int64 表示范围内，溢出时 Tiling 拒绝执行。
- 标量（0 维）输入按 1 个元素处理，空张量（元素数为 0）直接返回。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| aclnn API | [test_aclnn_softplus](./examples/test_aclnn_softplus.cpp) | 通过aclnnSoftplusGetWorkspaceSize、aclnnSoftplus接口调用SoftplusV2算子，接口说明参见[aclnnSoftplus](./docs/aclnnSoftplus.md)。 |
| GE图模式 | - | 通过[算子IR定义](./op_graph/softplus_v2_proto.h)中的REG_OP(SoftplusV2)构图调用SoftplusV2算子。 |
