# BNLL

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

- 算子功能：激活函数BNLL（Binomial Normal Log Likelihood），即数值稳定版本的softplus。
- 计算公式：

$$
BNLL(x) = \ln(1 + e^x)
$$

等价分段实现（保证数值稳定性）：

$$
y_i = \begin{cases} x_i + \ln(1 + e^{-x_i}) & \text{if } x_i > 0 \\ \ln(1 + e^{x_i}) & \text{otherwise} \end{cases}
$$

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
    <td>公式中的'x'，表示BNLL激活函数的输入张量。支持空Tensor，维度支持0-8。</td>
    <td>FLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>表示x经BNLL计算得到的输出张量，shape和dtype与输入x一致。</td>
    <td>FLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE图模式  | [test_geir_bnll.cpp](examples/test_geir_bnll.cpp) | 通过GE IR接口方式调用BNLL算子。 |


## Golden 的 Host 回归

NumPy Kernel golden 使用 `max(x, 0) + log1p(exp(-abs(x)))`，指数输入始终非正，避免 `np.where` 同时求值未选分支时产生溢出；保留原有 float64 中间精度及输入 dtype/shape，正负无穷和 NaN 按公式传播。

运行 `python3 activation/bnll/tests/assets/test_bnll_golden_branch_overflow.py` 可检查真实注册 golden，在 RuntimeWarning 作为错误和 NumPy overflow/invalid 严格模式下验证大值、普通值、标量、矩阵及空输入，使用 CPU Torch `logaddexp` 独立参考。仅验证 Host golden；不代表 TTK 设备执行、CANN 构建、NPU 算子精度或性能验证。
