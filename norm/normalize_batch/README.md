# NormalizeBatch

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | √ |

> 说明：算子注册`ascend950`、`ascend960dt`、`ascend960pr`等计算单元，共用同一套arch35 AscendC实现。

## 功能说明

- 算子功能：变长序列批归一化。按`seq_len[n]`给出的每个batch的归一化数据个数，对输入张量`x`（shape为`(n, c, d)`）的每个`(n, c)`行（或`n`平面）做标准化，输出`y`与`x`同shape、同dtype。统计（均值/方差）只使用每行前`L`个元素，但整行`d`个元素都会被归一化。
- 计算公式：

  $$
  L = \min(\max(seq\_len_n,\ 1),\ d)
  $$

  $$
  N = \begin{cases} L & \text{per\_feature（每(n,c)行独立统计）} \\ c \cdot L & \text{all\_features（同n的所有c共享统计）} \end{cases}
  $$

  $$
  mean = \frac{1}{N}\sum_{k=0}^{L-1} x_{n,\cdot,k} \qquad var = \left|\,E(x^2) - E(x)^2\,\right| \cdot \frac{N}{N-1}
  $$

  $$
  y = \frac{x - mean}{\sqrt{var} + \epsilon}
  $$

  其中`epsilon`加在`sqrt`之后（先开方再加，非加在方差之下）；方差带无偏修正`N/(N-1)`并取绝对值防负；`N < 2`时`var = 0`；`seq_len`在算子内部钳制到`[1, d]`。

## 参数说明

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>input_x</td>
      <td>输入</td>
      <td>表示被归一化的输入特征张量，即公式中的<code>x</code>，shape为<code>(n, c, d)</code>。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seq_len</td>
      <td>输入</td>
      <td>表示每个batch的归一化数据个数，即公式中的<code>seq_len</code>，shape为<code>(n,)</code>；取值在算子内部钳制到<code>[1, d]</code>。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_y</td>
      <td>输出</td>
      <td>表示归一化结果张量，即公式中的<code>y</code>，shape和数据类型与<code>input_x</code>一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normalize_type</td>
      <td>属性</td>
      <td>表示归一化的统计范围：<code>per_feature</code>为每<code>(n, c)</code>行独立统计（<code>N = L</code>）；<code>all_features</code>为同一<code>n</code>的所有<code>c</code>共享统计量（<code>N = c * L</code>）。必填，不支持其他取值。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td>表示防除零的下限，即公式中的<code>epsilon</code>，加在标准差之后，默认值为<code>1e-5</code>，要求大于0。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- `input_x`仅支持FLOAT32，数据格式仅支持ND，维度为3（`n, c, d`），其中`n >= 1`、`c >= 1`、`d > 1`（`d`维为统计长度维度，要求大于1）；`output_y.shape = input_x.shape`。
- `seq_len`仅支持INT32，数据格式仅支持ND，维度为1（`(n,)`）；运行时取值钳制到`[1, d]`。
- `normalize_type`必填，仅支持`per_feature`或`all_features`。
- `epsilon`为FLOAT32类型且必须大于0，默认值`1e-5`；加在`sqrt(var)`之后。
- 在<term>Ascend 950PR&950DT系列产品</term>上，本算子为非原地实现，执行后输入`input_x`保持不变；调用方应保证`output_y`与`input_x`底层内存不重叠。
- 在<term>Ascend 950PR&950DT系列产品</term>上，统计与归一化全程FLOAT32，内部采用平移域统计与补偿归一化，近常量输入（`std << |mean|`）下的精度优于逐项直接计算。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|----|------|
| GE图模式 | [test_geir_normalize_batch](examples/test_geir_normalize_batch.cpp) | 通过[算子IR](op_graph/normalize_batch_proto.h)构图方式调用NormalizeBatch算子。 |
