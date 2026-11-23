# SmoothL1LossV2

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

- 算子功能：计算SmoothL1损失函数。
- 计算公式:

  Batch为N的损失函数，当`reduction`为none时，此函数定义为：

  $$
  \ell(x,y) = L = \{l_1,\dots,l_N\}^\top
  $$

  其中的$l_n$为：

  $$
  l_n = \begin{cases}
  0.5(x_n-y_n)^2/sigma, & if |x_n-y_n| < sigma \\
  |x_n-y_n| - 0.5*sigma, &  otherwise
  \end{cases}
  $$

  其中sigma为本算子的阈值属性（PyTorch与aclnn接口侧的同名参数为beta），取值非负；当sigma=0时，逐元素损失退化为纯L1项$|x_n-y_n|$。

  如果`reduction`为`mean`或`sum`时，

  $$
  \ell(x,y)=\begin{cases}
  mean(L), & \text{if reduction} = \text{mean}\\
  sum(L), & \text{if reduction} = \text{sum}
  \end{cases}
  $$

- 其中：
  - Xn: predictions
  - Yn: labels/truth

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
    <td>公式中的'x'，表示用于计算SmoothL1损失的预测值输入，数据类型须与label一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>label</td>
    <td>输入</td>
    <td>公式中的'y'，表示predict需要拟合的真实标签值输入，数据类型须与predict一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>loss</td>
    <td>输出</td>
    <td>表示predict与label经smooth_l1_loss_v2计算得到的损失值输出，数据类型与predict一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sigma</td>
    <td>可选属性</td>
    <td><ul><li>控制smooth_l1_loss_v2中MSE与L1损失分界边界的阈值参数，取值须非负。</li><li>默认值为1.0。</li></ul></td>
    <td>FLOAT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>reduction</td>
    <td>可选属性</td>
    <td><ul><li>表示对逐元素损失值进行均值/求和/无归约的归约参数。</li><li>取值为none、mean或sum。</li><li>默认值为"mean"。</li></ul></td>
    <td>STRING</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- reduction仅支持none、mean、sum三种取值，其他取值会被拒绝。
- sigma取值须非负（sigma>=0）；sigma=0时逐元素损失退化为|x-y|。
- predict与label的数据类型必须一致；GE图模式通路下predict与label的shape必须一致（不支持broadcast），aclnn通路下self与target的shape须满足broadcast关系。
- reduction为mean或sum时，输出为rank-0标量（0维）；reduction为none时，输出shape与predict一致。
- 空Tensor语义：aclnn通路下，self或target为空Tensor时按reduction差异化处理——none时直接返回成功、mean时输出为NaN、sum时输出为0；GE图模式通路下，reduction为none且输入存在零长度轴时，当前tiling实现不支持，属已知限制。
- 支持动态Shape（未知维度以-1表示）与动态Rank（以-2表示）编译；GE图模式通路下rank-0标量输入不被支持。
- 特殊值行为：输入中的NaN会按公式逐元素传播到对应输出；Inf及极值输入在FLOAT16、BFLOAT16等低精度类型下，输出可能饱和为该数据类型的最大有限值（与高精度参考计算的inf结果存在差异），属已知精度行为。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_smooth_l1_loss_v2.cpp](examples/test_aclnn_smooth_l1_loss_v2.cpp) | 通过[aclnnSmoothL1Loss](docs/aclnnSmoothL1Loss.md)接口方式调用SmoothL1Loss算子，样例仅演示接口调用流程。 |
