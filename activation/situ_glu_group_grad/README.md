# SituGluGroupGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | ------|
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    ×     |
| <term>Atlas A2系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×    |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：SiTU门控线性单元（SiTU Gated Linear Unit）激活函数的反向梯度计算。根据上游梯度grad_y和前向输入x，计算输入x的梯度grad_x与weight的梯度grad_weight；传入可选输入weight时按行对grad_x加权并计算grad_weight，未传入weight时grad_weight为空Tensor（shape [0]）；传入可选输入group_index时仅计算前realBs = min(总行数, 行数)行，总行数由group_list_type决定（cumsum模式取group_index[-1]，count模式取sum(group_index)）。仅Ascend 950PR&950DT系列产品支持。

- 计算公式或计算流程：

  对给定的输入张量x（维度为[..., cut]，cut为偶数，令h = cut // 2）和上游梯度grad_y（维度为[..., h]）：

  1. 根据activate_left切分x为gate和up（各shape [..., h]）。

  2. 计算前向中间量：

     $$
     t = \tanh\left(\frac{gate}{\beta}\right), \quad s = \sigma(gate), \quad situ\_a = \beta \cdot t \cdot s
     $$

     当linear_beta > 0时：$up' = linear\_beta \cdot \tanh(up / linear\_beta)$，否则$up' = up$。

  3. 乘积法则：

     $$
     grad\_situ\_a = grad\_y \cdot up', \quad grad\_up' = grad\_y \cdot situ\_a
     $$

  4. gate梯度（w为可选输入weight对应行的标量，不传weight时w = 1）：

     $$
     grad\_gate = grad\_situ\_a \cdot s \cdot \left[ (1 - t^2) + \beta \cdot t \cdot (1 - s) \right] \cdot w
     $$

  5. up梯度：

     - linear_beta > 0时：$grad\_up = grad\_up' \cdot (1 - \tanh^2(up / linear\_beta)) \cdot w$
     - linear_beta ≤ 0时：$grad\_up = grad\_up' \cdot w$

  6. 根据activate_left将grad_gate、grad_up拼接为grad_x（shape与x一致）。

  7. weight梯度（输出grad_weight，未传入weight时grad_weight为空Tensor），对每行n沿h求和（注意不乘w）：

     $$
     grad\_w_n = \sum_{h} grad\_y_{n,h} \cdot situ\_a_{n,h} \cdot up'_{n,h}
     $$

   8. 行掩码（可选输入group_index）：仅前realBs = min(总行数, 行数)行被计算并写回，其余行不读不写。总行数由group_list_type决定：cumsum模式（0，默认）为group_index[-1]，count模式（1）为sum(group_index)。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
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
      <td>grad_y</td>
      <td>输入</td>
      <td>反向传播上游梯度，dim维度大小为x的一半。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>前向输入，dim维度会被均分为gate与up两部分。数据类型需与grad_y一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>可选输入</td>
      <td>前向的每行路由权重，元素个数等于x在dim轴之前各维度的乘积。仅Ascend 950PR&950DT系列产品支持。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>可选输入</td>
      <td>各组token数，一维非空、元素非负，含义由group_list_type决定。仅Ascend 950PR&950DT系列产品支持。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td><ul><li>切分维度，需与前向SituGlu一致。</li><li>取值范围为[-x.dim(), x.dim()-1]。</li><li>默认值为-1。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>属性</td>
      <td><ul><li>SiTU门控部分的缩放系数，需与前向SituGlu保持一致。</li><li>必须为非零值，传入0会返回参数校验失败。</li><li>默认值为1.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linear_beta</td>
      <td>属性</td>
      <td><ul><li>up路径线性tanh的缩放系数，需与前向SituGlu保持一致。</li><li>默认值为0.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>属性</td>
      <td><ul><li>切分x时gate是否为前半部分，需与前向SituGlu保持一致。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_list_type</td>
      <td>属性</td>
      <td><ul><li>group_index的分组表示方式。</li><li>取值只能为0或1：0为cumsum模式，group_index为各组行数的累计和，总行数为group_index[-1]；1为count模式，group_index为各组行数，总行数为sum(group_index)。</li><li>默认值为0。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>输出</td>
      <td>输入x的梯度，shape与x完全一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_weight</td>
      <td>输出</td>
      <td>weight的梯度：传入weight时shape与weight一致；未传入weight时为空Tensor（shape [0]）。仅Ascend 950PR&950DT系列产品支持。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 支持非连续Tensor。
- beta、linear_beta、activate_left、dim需与前向SituGlu调用时的取值一致，否则梯度结果不正确。
- beta必须为非零值，传入0会返回参数校验失败。
- 仅Ascend 950PR&950DT系列产品支持（weight / group_index为可选输入，grad_weight为必选输出）。
- 传入group_index时，grad_x与grad_weight中行号 ≥ realBs的行不会被写入，调用方如需严格零值请自行清零。
- group_list_type必须从[0, 1]中取值，传入其他值会返回参数校验失败。
- group_index的数值需要满足以下条件，否则无法保证输出是否符合预期：
  - 当group_list_type为0时，group_index必须为非负单调非递减数列，表示分组后每组大小的cumsum结果（累计和），最后一个值应小于等于行数。
  - 当group_list_type为1时，group_index必须为非负数列，表示分组后每组大小，数值的总和应小于等于行数。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_situ_glu_group_grad](./examples/test_aclnn_situ_glu_group_grad.cpp) | 通过[aclnnSituGluGroupGrad](./docs/aclnnSituGluGroupGrad.md)接口调用SituGluGroupGrad算子（含可选输入weight/group_index）。 |
| GE图模式 | [test_geir_situ_glu_group_grad](./examples/test_geir_situ_glu_group_grad.cpp)  | 通过[算子IR](./op_graph/situ_glu_group_grad_proto.h)构图方式调用SituGluGroupGrad算子。         |
| PyTorch API |  -  | 通过[situ_glu_group_grad](./docs/torchapi_situ_glu_group_grad.md)接口调用SituGluGroupGrad算子。         |
