# NllLossGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     ×    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：负对数似然损失反向。
- 计算公式：

$$
x\_grad[n][i] =
\begin{cases}
    \displaystyle -y\_grad[n] \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{none} \\
    \displaystyle -\frac{y\_grad}{\text{total\_weight}} \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{mean} \\
    \displaystyle -y\_grad \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{sum} \\
    0, & \text{if } i \neq \text{target}
\end{cases}
$$

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
    <td>x</td>
    <td>输入</td>
    <td>公式中的输入x，输入概率tensor。shape支持(C)、(N, C)或(N, C, H, W)，其中N表示batch size，C表示类别数；当shape为(N, C, H, W)时，H、W必须为正整数。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y_grad</td>
    <td>输入</td>
    <td>正向结果的梯度，对应公式中的y_grad。reduction为"none"时，x为(N, C)则shape为(N)，x为(C,)则shape为()，x为(N, C, H, W)则shape为(N, H, W)；reduction为"mean"或"sum"时，shape为单元素tensor（0D或元素个数为1的1D）。数据类型与x保持一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>target</td>
    <td>输入</td>
    <td>表示真实标签，公式中的target。x为(N, C)时shape为(N)，x为(C,)时shape为()，x为(N, C, H, W)时shape为(N, H, W)。每个元素的取值范围为[0, C-1]或等于ignore_index。</td>
    <td>INT32、INT64、UINT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>输入</td>
    <td>表示每个类别的缩放权重，公式中的weight。shape为(C,)，元素个数等于x的类别数C。数据类型与x保持一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>total_weight</td>
    <td>输入</td>
    <td>公式中的total_weight。元素个数必须为1（shape为(1,)）。数据类型与x保持一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>reduction</td>
    <td>可选属性</td>
    <td><ul><li>指定要应用到输出的缩减，对应公式中的reduction。</li><li>取值为"none"、"mean"或"sum"。</li><li>默认值为"mean"，省略该属性时等价于"mean"（对应aclnn接口reduction=1）。</li></ul></td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ignore_index</td>
    <td>可选属性</td>
    <td><ul><li>指定一个被忽略且不影响输入梯度的目标值。</li><li>默认值为-100，省略该属性时等价于-100。</li><li>target中等于该值的元素不参与梯度计算。</li></ul></td>
    <td>INT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>x_grad</td>
    <td>输出</td>
    <td>公式中的x_grad[n][i]。shape与x一致，数据类型与x保持一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- reduction取值只能为"none"、"mean"或"sum"（对应aclnn接口的reduction=0/1/2），其它取值返回错误。
- target每个元素的取值范围为[0, C-1]或等于ignore_index，其中C为x的类别数（x为1D/2D时为最后一维大小，x为4D时为第2维大小）。
- weight的元素个数必须等于C。
- total_weight的元素个数必须为1。
- y_grad的形状随reduction与x的形状变化：reduction为"none"时，x为(N, C)则y_grad为(N)，x为(C,)则为()，x为(N, C, H, W)则为(N, H, W)；reduction为"mean"或"sum"时，y_grad为单元素tensor。
- x、y_grad、weight、total_weight、x_grad的数据类型必须一致。
- x支持的维度边界为1D (C)、2D (N, C)、4D (N, C, H, W)，其它维度返回错误；当x为4D时，H、W必须为正整数。
- 动态shape支持：shape中可用-1表示未知维度；支持动态rank输入（以-2标记未知秩），此时输出shape推导为未知秩(-2)。
- 数据格式仅支持ND。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|--------------|--------------|--------------|
| aclnn API | [test_aclnn_nll_loss_grad](./examples/test_aclnn_nll_loss_grad.cpp) | 通过[aclnnNLLLossBackward](./docs/aclnnNLLLossBackward.md)接口方式调用NLLLossGrad算子，适用于x为(C)或(N, C)的场景。 |
| aclnn API | [test_aclnn_nll_loss_grad_2d](./examples/test_aclnn_nll_loss_grad_2d.cpp) | 通过[aclnnNLLLoss2dBackward](./docs/aclnnNLLLoss2dBackward.md)接口方式调用NLLLossGrad算子，适用于x为(N, C, H, W)的场景。 |
| GE图模式 | - | 通过[算子IR定义](./op_graph/nll_loss_grad_proto.h)构图方式调用NLLLossGrad算子。 |

样例代码的编译与运行在仓库根目录下通过build.sh完成，命令如下（--soc可选，指定ascend950时会额外运行arch35目录下的示例文件）：

```bash
bash build.sh --run_example nll_loss_grad eager cust --example_name=nll_loss_grad --soc=ascend950
bash build.sh --run_example nll_loss_grad eager cust --example_name=nll_loss_grad_2d --soc=ascend950
```

更多用法参考[快速调用指南](../../docs/zh/invocation/quick_op_invocation.md)。
