# BatchNormGradV3

简体中文 | [English](README_en.md)

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | √ |

上表为标准ACLNN接口的支持情况。Fast接口见[aclnnFastBatchNormBackward](docs/aclnnFastBatchNormBackward.md)，直接GE图调用的产品差异见[约束说明](#约束说明)。

## 功能说明

BatchNormGradV3用于计算批归一化的输入梯度、缩放参数梯度和偏置参数梯度。训练模式使用前向保存的统计量。计算公式见[aclnnBatchNormBackward](docs/aclnnBatchNormBackward.md#功能说明)。

## 参数说明

下表为GE图算子参数。C为通道数。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| dy | 输入 | 上游梯度。 | FLOAT32、FLOAT16、BFLOAT16 | NCHW、NHWC、NCDHW、NDHWC |
| x | 输入 | 前向输入，shape、数据类型和数据格式与dy一致。 | FLOAT32、FLOAT16、BFLOAT16 | NCHW、NHWC、NCDHW、NDHWC |
| weight | 输入 | 每通道的缩放参数，shape为`[C]`。 | FLOAT32、FLOAT16、BFLOAT16 | ND |
| running_mean | 输入 | 运行均值，shape为`[C]`。 | FLOAT32、FLOAT16、BFLOAT16 | ND |
| running_var | 输入 | 运行方差，shape为`[C]`。 | FLOAT32、FLOAT16、BFLOAT16 | ND |
| save_mean | 输入 | 训练使用的保存均值，shape为`[C]`。 | FLOAT32 | ND |
| save_rstd | 输入 | 训练使用的保存统计量，shape为`[C]`，产品差异见约束说明。 | FLOAT32 | ND |
| dx | 输出 | 输入梯度，shape、数据类型和数据格式与dy一致。 | FLOAT32、FLOAT16、BFLOAT16 | NCHW、NHWC、NCDHW、NDHWC |
| dweight | 输出 | 缩放参数梯度，数据类型与weight一致。 | FLOAT32、FLOAT16、BFLOAT16 | ND |
| dbias | 输出 | 偏置参数梯度，数据类型与weight一致。 | FLOAT32、FLOAT16、BFLOAT16 | ND |
| is_training | 可选属性 | true表示训练，false表示推理，默认值为true。 | BOOL | - |
| epsilon | 可选属性 | 添加到方差中的值，用于提高数值稳定性，默认值为1e-5。建议使用较小的有限正数。 | FLOAT32 | - |
| output_mask | 可选属性 | 长度为3的布尔数组，依次对应dx、dweight、dbias，默认值为`{true, true, true}`。 | ListBool | - |

## 约束说明

- NCHW、NCDHW的通道轴为第2维，NHWC、NDHWC的通道轴为最后一维。
- 直接GE构图时需要连接所有必选输入。ACLNN包装层可填充辅助参数空指针的默认值，并将支持的低维或高维输入调整为底层布局。ACLNN的shape和格式约束见[接口文档](docs/aclnnBatchNormBackward.md)。
- <term>Ascend 950PR&950DT系列产品</term>：
  - 直接GE调用时使用4维NCHW/NHWC或5维NCDHW/NDHWC。训练时save_rstd为前向保存的标准差倒数。
  - 训练支持非通道轴为空的输入。当C>0且归约元素数为0时，dx为空Tensor，启用的dweight、dbias保留shape为`[C]`并置零；未启用的输出不写入。
  - 推理模式的GE Kernel计算dx。标准ACLNN接口在推理时还会按outputMask计算gradWeight和gradBias。
  - 形状推导按weight的shape推导dweight和dbias。空输入推理时，dx保留dy的shape，dweight和dbias推导为`[0]`。
  - GE图定义中`-1`表示待确定维度，`[-2]`表示待确定秩。执行时需提供符合布局约束的具体shape。使用未知秩时，应显式设置与实际4维或5维输入匹配的数据格式。
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：直接GE调用时使用4维NCHW或5维NCDHW，save_rstd传入保存的方差。推理Kernel计算三个梯度，output_mask使用默认值。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn API | [test_aclnn_batch_norm_backward](./examples/test_aclnn_batch_norm_backward.cpp) | 通过[aclnnBatchNormBackward](./docs/aclnnBatchNormBackward.md)接口方式调用batch_norm_grad_v3算子。 |
| aclnn API | [test_aclnn_fast_batch_norm_backward](./examples/arch22/test_aclnn_fast_batch_norm_backward.cpp) | 通过[aclnnFastBatchNormBackward](./docs/aclnnFastBatchNormBackward.md)接口方式调用batch_norm_grad_v3算子。 |
| GE图模式 | [test_geir_batch_norm_grad_v3](./examples/test_geir_batch_norm_grad_v3.cpp) | 通过[算子IR](op_graph/batch_norm_grad_v3_proto.h)构图方式调用BatchNormGradV3算子。 |
| PyTorch API | [test_torch_batch_norm_grad_v3](./examples/test_torch_batch_norm_grad_v3.py) | 通过PyTorch的`native_batch_norm_backward`调用，运行命令见样例。 |
