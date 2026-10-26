# InplaceApplyRMSProp

## 产品支持情况

| 产品 | 是否支持 |
|:-----------------------------------------|:------:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

`InplaceApplyRMSProp`是RMSProp自适应学习率优化器的参数更新算子。算子
维护梯度的滑动均方根和动量缓冲区，在每次迭代中更新`var`、`ms`和
`mom`三个状态张量。三个显式输出分别返回同名输入的更新结果。

本算子当前仅提供AscendC GE图模式通路，不提供独立的`aclnn`接口。

计算公式如下：

$$
\begin{aligned}
\text{ms}_t &= \rho \cdot \text{ms}_{t-1} + (1 - \rho) \cdot \text{grad}^2 \\
\text{mom}_t &= \text{momentum} \cdot \text{mom}_{t-1} +
\frac{\text{lr} \cdot \text{grad}}{\sqrt{\text{ms}_t + \epsilon}} \\
\text{var}_t &= \text{var}_{t-1} - \text{mom}_t
\end{aligned}
$$

其中`lr`为学习率，`rho`为衰减系数，`momentum`为动量系数，`epsilon`
为数值稳定性常数。

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
<td>var</td>
<td>输入</td>
<td>待更新的权重张量，对应公式中的var，就地更新；shape、dtype与ms、mom、grad一致。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>ms</td>
<td>输入</td>
<td>梯度平方滑动平均状态，对应公式中的ms，就地更新；调用方应保证非负，实现不做数值校验。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>mom</td>
<td>输入</td>
<td>动量缓冲区，对应公式中的mom，就地更新。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>lr</td>
<td>输入</td>
<td>学习率，对应公式中的lr。shape为{1}的一维ND标量Tensor。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>rho</td>
<td>输入</td>
<td>衰减系数，对应公式中的rho。shape为{1}的一维ND标量Tensor；建议取值[0, 1)，实现不做数值校验。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>momentum</td>
<td>输入</td>
<td>动量系数，对应公式中的momentum。shape为{1}的一维ND标量Tensor。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>epsilon</td>
<td>输入</td>
<td>数值稳定性常数，对应公式中的epsilon。shape为{1}的一维ND标量Tensor。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>grad</td>
<td>输入</td>
<td>当前梯度张量，对应公式中的grad；shape、dtype与var一致。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>var</td>
<td>输出</td>
<td>更新后的参数，对应输入var，就地写回var。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>ms</td>
<td>输出</td>
<td>更新后的梯度平方滑动平均，对应输入ms，就地写回ms。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>mom</td>
<td>输出</td>
<td>更新后的动量缓冲区，对应输入mom，就地写回mom。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>use_locking</td>
<td>可选属性</td>
<td>是否加锁更新，默认false；当前实现仅作兼容占位，不引入额外互斥锁。</td>
<td>BOOL</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- `var`、`ms`、`mom`、`grad`的shape、dtype必须完全一致，支持rank1-8；调用方应保证`ms`非负，实现不做数值校验，负值可能使平方根计算产生NaN。
- 输入shape标记`-1`（未知维）与`-2`（未知rank）已按动态shape/rank登记并支持Shape推导传播；但执行通路要求最终解析为rank∈[1,8]且各维非负的具体ND shape，未解析的`-1`/`-2`会在Tiling阶段被拒绝。
- 空Tensor仅适用于状态张量：当`var`（及与其形状一致的`ms`、`mom`、`grad`）`numel == 0`时走空路径、不执行更新；`lr`、`rho`、`momentum`、`epsilon`必须仍为shape {1}，`numel == 0`的标量输入会被拒绝。
- `lr`、`rho`、`momentum`、`epsilon`必须为shape{1}的ND标量张量，dtype与`var`一致；`rho`建议取值[0, 1)，实现不做范围校验。
- 非连续（view strides/offset）Tensor未声明支持：调用方应保证`var`、`ms`、`mom`、`grad`与四个标量输入为连续内存（ND格式），不满足时行为未定义。
- 输入非法时按下列规则拒绝：rank超出[1,8]报`EZ0012`；状态张量shape不一致报`EZ0010`；标量shape非{1}、逐维为负或元素数非法报`EZ0009`；`var` dtype非法报`EZ0020`，跨输入dtype不一致报`EZ0021`。
- `epsilon`在计算域按FP32处理；当其值小于等于`FLT_MIN`或为NaN时，kernel使用`FLT_MIN`作为下限。
- `FLOAT`、`BFLOAT16`路径的`grad²`在FP32域计算：二者指数范围相同，当`|grad| > sqrt(FLT_MAX)≈1.84e19`时该中间量在FP32溢出为`inf`，`ms`、`mom`、`var`可能偏离数学定义值；`FLOAT16`的`|grad| ≤ 65504`，不存在该溢出。
- 支持`FLOAT`、`FLOAT16`、`BFLOAT16`；FP16/BF16路径在FP32域完成计算后再转换回目标dtype。
- `use_locking`不改变计算结果，当前实现不引入额外互斥锁。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | [test_geir_inplace_apply_rms_prop](./examples/test_geir_inplace_apply_rms_prop.cpp) | 通过GE图模式调用，算子IR定义见[算子IR](./op_graph/inplace_apply_rms_prop_proto.h)。 |

当前不提供ACLNN、PyTorch、ONNX或融合规则接口；TensorFlow原生RMSProp接口与本算子的三显式输出原型不一致，因此不作为有效调用通路。
