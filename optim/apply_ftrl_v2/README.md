# ApplyFtrlV2

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

ApplyFtrlV2 实现 FTRL（Follow The Regularized Leader）优化算法的 V2 版本参数更新规则。该算子是纯逐元素计算算子，在模型训练的每步迭代中根据梯度、学习率和正则化参数就地更新权重变量（var）、梯度平方累加器（accum）和线性累加器（linear）。适用于大规模稀疏特征的在线学习场景，如广告点击率（CTR）预估、推荐系统等。

**V2 与 V1（ApplyFtrl）的区别**：V2 新增 `l2_shrinkage` 参数，支持 shrinkage-type L2 正则化。V2 在梯度上叠加 `2 * l2_shrinkage * var` 的修正量，等价于对损失函数直接添加 L2 惩罚项（`Loss += l2_shrinkage * ||w||²`）。当 `l2_shrinkage = 0` 时，V2 退化为 V1。

**计算公式**（7 步更新）：

$$
\begin{aligned}
\hat{g} &= g + 2 \cdot \lambda_s \cdot w \\
n' &= n + g^2 \\
\sigma &= \frac{{n'}^{-p} - n^{-p}}{\eta} \\
z &\leftarrow z + \hat{g} - \sigma \cdot w \\
x &= \lambda_1 \cdot \text{sign}(z) - z \\
y &= \frac{{n'}^{-p}}{\eta} + 2 \cdot \lambda_2 \\
w &\leftarrow \begin{cases} x / y & \text{if } |z| > \lambda_1 \\ 0 & \text{otherwise} \end{cases} \\
n &\leftarrow n'
\end{aligned}
$$

其中：$w$=var（权重），$n$=accum（梯度平方累积），$z$=linear（线性累加器），$g$=grad（梯度），$\eta$=lr（学习率），$\lambda_1$=l1，$\lambda_2$=l2，$\lambda_s$=l2_shrinkage，$p$=lr_power。

## 参数说明

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
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
    <td>var</td>
    <td>输入</td>
    <td>模型权重张量，对应公式中 $w$。更新结果经输出端口 var 返回；输出是否复用输入 var 的存储由图编译器决定。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>输入</td>
    <td>梯度平方累加器，对应公式中 $n$。shape/dtype必须与var一致；Kernel内将更新结果写回输入GM地址（in-place）。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>linear</td>
    <td>输入</td>
    <td>线性累加器，对应公式中 $z$。shape/dtype必须与var一致；Kernel内将更新结果写回输入GM地址（in-place）。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>当前步梯度，对应公式中 $g$。shape/dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率（0-d 或 shape [1] 张量），对应公式中 $\eta$。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l1</td>
    <td>输入</td>
    <td>L1 正则化强度（0-d 或 shape [1] 张量），对应公式中 $\lambda_1$。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l2</td>
    <td>输入</td>
    <td>L2 正则化强度（0-d 或 shape [1] 张量），对应公式中 $\lambda_2$。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l2_shrinkage</td>
    <td>输入</td>
    <td>L2 shrinkage 强度（0-d 或 shape [1] 张量），对应公式中 $\lambda_s$（V2 新增）。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr_power</td>
    <td>输入</td>
    <td>学习率衰减幂次（0-d 或 shape [1] 张量），对应公式中 $p$。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>属性</td>
    <td>是否在更新时加锁。默认false。当前实现不读取该属性、不提供加锁保护，仅作语义占位。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var</td>
    <td>输出</td>
    <td>更新后的var Tensor，对应公式中 $w$。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

> **注**：accum、linear 为 in-place 更新（与 TensorFlow 一致），更新结果直接写回输入地址，不作为显式输出端口返回；var 为唯一显式输出。

## 约束说明

- **数据类型一致性**：var、accum、linear、grad 四个 tensor 的数据类型必须一致；lr、l1、l2、l2_shrinkage、lr_power 五个 scalar 的数据类型必须与 tensor 一致（Tiling 入口逐输入校验，不一致报错拒绝）。
- **Shape 约束**：var、accum、linear、grad 的 shape 必须完全相同（Tiling 入口逐维校验）；lr、l1、l2、l2_shrinkage、lr_power 必须为标量（0-d tensor 或恰好 1 个元素的张量；空标量与多元素张量被拒绝）；tensor 维度范围 0-8 维（Tiling 入口拒绝 9 维及以上）。
- **规模上限**：var 元素总数不超过 2^32-512（TilingData 计数字段为 uint32，超限输入显式报错，不做静默截断）。
- **参数值域**：lr > 0（学习率必须为正数）、l1 >= 0、l2 >= 0、l2_shrinkage >= 0、lr_power <= 0。超参为 device tensor，算子**不做运行时值域校验**（由调用方保证）；域外取值按 IEEE 浮点规则传播 Inf/NaN（例如 lr=0 时 linear 输出 ∓Inf/NaN、var 通常输出 NaN（x/y 为 Inf/Inf），仅 linear 为 NaN 的元素（如 var=0 使 Inf·0=NaN）输出 0；|linear| ≤ l1 时 var 输出 0；负 accum 配合分数幂次输出 NaN）。
- **In-place 语义**：accum、linear 就地更新，执行后输入数据被覆盖；var 经独立输出端口返回（不保证与输入共享存储）。
- **FP16/BF16 计算**：FP16 和 BF16 输入在 kernel 内部提升到 FP32 计算，结果 cast 回原始类型输出。
- **确定性**：默认确定性实现。该算子为纯逐元素操作，无归约或排序，计算完全确定性。
- **不支持 `multiply_linear_by_lr` 属性**：以 TensorFlow ApplyFtrlV2 标准路径为准（`multiply_linear_by_lr=false`）。
- **空 Tensor**：支持空 Tensor（numel=0），kernel 跳过计算。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|:---------|:---------|:-----|
| GE图模式 | [test_geir_apply_ftrl_v2](./examples/arch35/test_geir_apply_ftrl_v2.cpp) | 通过 [算子IR](./op_graph/apply_ftrl_v2_proto.h) 构图方式调用 ApplyFtrlV2 算子。 |
