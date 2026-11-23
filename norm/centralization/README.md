# Centralization

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

对输入张量 `x`，沿 `axes` 指定的轴计算均值，再从每个元素中减去对应的均值。

$$
y = x - \operatorname{mean}(x, \mathrm{axes})
$$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `x` | 输入 | 待中心化的张量，秩为 1 至 8。 | FLOAT、FLOAT16 | ND |
| `axes` | 属性 | 归约轴列表。每个轴的取值范围为 `[-rank(x), rank(x))`，重复的轴只计算一次。GE 图中省略此属性时，默认沿最后一个轴归约。 | LIST_INT | - |
| `y` | 输出 | 中心化结果，形状和数据类型与 `x` 相同。 | FLOAT、FLOAT16 | ND |

显式传入 `axes=[]` 时，输出为 `x - x`。有限元素的结果为 0，NaN 和无穷元素的结果为 NaN。

## 约束说明

<term>Ascend 950PR&950DT系列产品</term>支持空张量，也支持在建图时使用动态形状或动态秩。执行时，输入张量的各维度必须已确定。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE 图模式 | [test_geir_centralization](examples/test_geir_centralization.cpp) | 使用 `ge::op::Centralization` 构图。 |
| TensorFlow 图模式 | - | 使用自定义算子 `Centralization`，将 `axes` 作为整数列表属性传入。 |
