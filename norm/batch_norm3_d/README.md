# BatchNorm3D

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

BatchNorm3D对5D输入张量按通道做批归一化，支持训练和推理模式。

设每个通道包含$M=N\times D\times H\times W$个元素。训练模式计算通道均值$\mu$和有偏方差$v$：

$$
\mu=\frac{1}{M}\sum_{i=1}^{M}x_i,\qquad
v=\frac{1}{M}\sum_{i=1}^{M}(x_i-\mu)^2
$$

训练和推理模式均按以下公式计算输出；推理模式的$\mu$和$v$分别取自输入`mean`和`variance`。

$$
y=(x-\mu)\frac{1}{\sqrt{v+\text{epsilon}}}\times\text{scale}+\text{offset}
$$

训练模式下，当$M>1$时，`batch_variance`输出无偏方差$vM/(M-1)$。$M=1$时的结果和`reserve_space_2`的含义见[约束说明](#约束说明)。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| x | 输入 | 逻辑shape为5D的输入张量，通道数为C；存储格式约束见下文。 | FLOAT16、FLOAT32 | NCDHW、NDHWC、NDC1HWC0 |
| scale | 输入 | 每通道的缩放系数。 | FLOAT32 | ND、NDC1HWC0 |
| offset | 输入 | 每通道的偏置，shape与scale一致。 | FLOAT32 | ND、NDC1HWC0 |
| mean | 可选输入 | 每通道的推理均值，shape与scale一致。推理时必填；训练时的传入要求见下文。 | FLOAT32 | ND、NDC1HWC0 |
| variance | 可选输入 | 每通道的推理方差，shape与scale一致。推理时必填；训练时的传入要求见下文。 | FLOAT32 | ND、NDC1HWC0 |
| y | 输出 | 归一化结果，shape、数据类型和数据格式与x一致。 | FLOAT16、FLOAT32 | NCDHW、NDHWC、NDC1HWC0 |
| batch_mean | 输出 | shape与scale一致。训练时为批均值，推理时为输入mean。 | FLOAT32 | ND、NDC1HWC0 |
| batch_variance | 输出 | shape与scale一致。训练时为批方差统计量，推理时为输入variance。 | FLOAT32 | ND、NDC1HWC0 |
| reserve_space_1 | 输出 | shape与scale一致。训练时保存批均值供反向计算使用，推理时为输入mean。 | FLOAT32 | ND、NDC1HWC0 |
| reserve_space_2 | 输出 | shape与scale一致。训练时保存辅助统计量，具体含义见下文；推理时为输入variance。 | FLOAT32 | ND、NDC1HWC0 |
| epsilon | 可选属性 | 平方根运算前加到方差上的值，默认值为0.0001。建议使用有限正数。 | FLOAT32 | - |
| data_format | 可选属性 | x的逻辑数据格式，支持NCDHW、NDHWC，默认值为NCDHW。 | STRING | - |
| is_training | 可选属性 | true表示训练，false表示推理，默认值为true。 | BOOL | - |

## 约束说明

- 运行时输入的每一维必须大于0，不支持空张量；NCDHW的通道轴为1，NDHWC的通道轴为4。
- 输入按声明格式连续存储，不提供自定义stride接口。
- epsilon未设置数值域拒绝检查；非正值或非有限值可能导致除零、无穷或NaN。推理时建议variance非负。
- <term>Ascend 950PR&950DT系列产品</term>：
  - x、y使用NCDHW或NDHWC格式，秩为5；data_format必须与x的格式一致。其余张量使用ND格式，均为一维，长度等于C。
  - 训练模式下mean、variance可省略；传入时必须满足上述规格，数值不参与计算。五个输出均为必选输出。
  - 训练模式下`reserve_space_2`为倒标准差$1/\sqrt{v+\text{epsilon}}$；当$M=1$时，`batch_variance`为NaN。
  - 动态图中的`-1`表示待确定维度，`[-2]`表示待确定秩；执行时必须给出满足上述约束的具体shape。形状、类型推断回调不作为输入合法性判定接口。
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>、<term>Atlas推理系列产品</term>、<term>Atlas训练系列产品</term>：
  - 使用CANN内置实现，执行时所有张量采用NDC1HWC0存储格式。x、y的存储shape为`[N,D,C1,H,W,C0]`，通道张量的存储shape为`[1,1,C1,1,1,C0]`，C1、C0与x对应维度一致；逻辑格式支持NCDHW、NDHWC。
  - 训练模式下mean、variance必须省略。
  - 训练模式下`reserve_space_2`为有偏方差$v$；有限输入且$M=1$时，`batch_variance`为0。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| GE图模式 | [test_geir_batch_norm3d](examples/test_geir_batch_norm3d.cpp) | 通过[算子IR](op_graph/batch_norm3d_proto.h)构图调用。 |
