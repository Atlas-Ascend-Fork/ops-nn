# ZConfusionSoftmaxGradFusionPass

## 融合模式

该融合规则将Mul、ReduceSum、Sub组成的子图识别并融合为ConfusionSoftmaxGrad算子，计算公式为：

```text
mul = grad * x
sum = ReduceSum(mul, axes=[-1], keep_dims=true)
y = grad - sum
```

- 融合前：

  ![Mul、ReduceSum、Sub子图](../../../docs/zh/figures/z_confusion_softmax_grad_fusion_pass_1.png)

  **融合为**

  ![ConfusionSoftmaxGrad替换子图](../../../docs/zh/figures/z_confusion_softmax_grad_fusion_pass_2.png)

融合后，原Mul的第一个输入映射到ConfusionSoftmaxGrad的`grad`，第二个输入映射到`x`，融合节点的`y`输出替换原Sub的输出。融合节点沿用原Sub输出的数据类型、格式、shape、原始格式和原始shape。

## 使用约束

- 输入和连接约束：
  - Mul的第一个输入必须与Sub的第一个输入复用同一张量`grad`，第二个输入为`x`。
  - Sub的第二个输入必须为ReduceSum的结果，不能与第一个输入交换。
  - 原Sub的输出数据边数量必须为1，否则跳过融合。
  - Mul、ReduceSum等内部节点的连接应符合上述模式，不能在融合后丢失子图外部依赖。

- 数据格式和shape约束：
  - `grad`与`x`的shape必须完全相同，不能通过广播匹配不同shape的输入；输入必须为非标量。
  - Mul的输入和输出必须具有已知shape；维度包含`-1`或未知rank（`{-2}`）时跳过融合。
  - Mul输出的最后一维长度不能超过30000；超过该长度时跳过融合。
  - 根据[ConfusionSoftmaxGrad算子定义](../op_graph/confusion_softmax_grad_proto.h)，输入数据格式必须为ND。

- ReduceSum参数约束：
  - `axes`必须是第1号输入上的单元素INT32或INT64常量；运行时输入的axes不支持融合。
  - 归约轴必须是合法的最后一维，支持`-1`及`rank-1`的写法，不支持越界轴。
  - axes常量允许被子图外其他节点共享，融合后保留这些外部连接。
  - `keep_dims`属性必须存在且为true。

- 数据类型约束：
  - `grad`与`x`必须具有相同的数据类型。
  - 数据类型仅支持FLOAT16、FLOAT32、BFLOAT16。

- 替换图必须能够完成shape推导；推导失败时融合失败。

## 支持的型号

<!-- npu="A3" id1 -->
<term>Atlas A3系列产品</term>
<!-- end id1 -->

<!-- npu="950" id2 -->
<term>Ascend 950PR&950DT系列产品</term>
<!-- end id2 -->
