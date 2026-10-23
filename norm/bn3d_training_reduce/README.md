# BN3DTrainingReduce

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     √    |
|  <term>Atlas推理系列产品</term>    |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：BN3DTrainingReduce用于三维Batch Normalization（批归一化）训练前向的规约阶段，与BN3DTrainingUpdate配对使用。算子按通道分别计算输入`x`的和与平方和。
- 计算公式（以5D NCDHW为例）：

  $$
  sum_{c} = \sum_{n=0}^{N-1} \sum_{d=0}^{D-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x_{(n,c,d,h,w)}
  $$

  $$
  squareSum_{c} = \sum_{n=0}^{N-1} \sum_{d=0}^{D-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x_{(n,c,d,h,w)}^2
  $$

  其中$x$为输入特征图，$sum$、$squareSum$为per-C的统计量。`square_sum`是逐元素平方后求和，不是$(\sum x)^2$。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>公式中的输入<code>x</code>，shape要求详见<a href="#约束说明">约束说明</a>。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCDHW、NDHWC、NDC1HWC0</td>
    </tr>
    <tr>
      <td>sum</td>
      <td>输出</td>
      <td>公式中的<code>sum</code>，shape要求详见<a href="#约束说明">约束说明</a>。</td>
      <td>FLOAT32</td>
      <td>NCDHW、NDHWC、NDC1HWC0</td>
    </tr>
    <tr>
      <td>square_sum</td>
      <td>输出</td>
      <td>公式中的<code>squareSum</code>，shape与数据格式和<code>sum</code>一致。</td>
      <td>FLOAT32</td>
      <td>NCDHW、NDHWC、NDC1HWC0</td>
    </tr>
  </tbody></table>

## 约束说明

- 参数表中的数据格式指原始格式，具体存储格式由图编译流程选择；输入`x`的数据格式和shape维度数必须同时满足对应产品的要求，否则报错。
- 空Tensor：通道轴大小为0时输出为空，其他轴允许为0；通道轴不为0时，N轴和各空间规约轴大小必须大于0。
- 支持动态shape与动态rank：图编译期用`-1`表示未知维度、`[-2]`表示未知rank，执行期以具体shape重新校验，具体shape不允许含负值维度。rank未知时，原始格式为NCDHW、NDHWC的输出shape为[-1]；NDC1HWC0推理分支的输出shape为[1,1,-1,1,1,-1]，此推理规则不扩大各产品支持的原始格式集合。
- 输入含NaN、Inf时，按IEEE 754语义传播到对应通道的两个输出；有限值平方或累加溢出FLOAT32时，允许输出Inf。
- <term>Ascend 950PR&950DT系列产品</term>：
  - 输入数据类型：支持FLOAT16、FLOAT32、BFLOAT16。
  - 输入格式和shape：NCDHW为2～5维，shape为[N, C, ...]；NDHWC为5维，shape为[N, D, H, W, C]。
  - 输出格式和shape：两个通道向量输出均使用NCDHW格式，shape为[C]。
  - 不支持以NDC1HWC0作为输入原始格式直接构图。
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  - 输入数据类型：支持FLOAT16、FLOAT32、BFLOAT16。
  - NCDHW输入为2～5维，shape为[N, C, ...]；两个输出的shape均为[C]。
  - NDHWC输入为5维，shape为[N, D, H, W, C]；两个输出的shape均为[C]。
  - NDC1HWC0输入为6维，shape为[N, D, C1, H, W, C0]；两个输出的shape均为[1, 1, C1, 1, 1, C0]。
- <term>Atlas训练系列产品</term>、<term>Atlas推理系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>：
  - 输入数据类型：支持FLOAT16、FLOAT32，不支持BFLOAT16。

## 调用说明

本算子是图模式内部算子，仅提供算子IR构图入口，不提供aclnn对外接口。

| 调用方式   | 调用样例           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用  | [test_geir_bn3d_training_reduce](examples/arch35/test_geir_bn3d_training_reduce.cpp) | 通过[算子IR](op_graph/bn3d_training_reduce_proto.h)构图方式调用BN3DTrainingReduce算子。 |
