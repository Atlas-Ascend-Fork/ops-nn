# DataFormatDimMap

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term>| √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：根据源数据格式和目标数据格式的维度映射关系，将输入的维度索引转换为目标格式下的对应维度索引。
- 计算示例：输入维度索引`[0, 1, 2, 3]`，源格式`NHWC`，目标格式`NCHW`，输出`[0, 2, 3, 1]`。即NHWC的第1维(H)对应NCHW的第2维，第2维(W)对应第3维，第3维(C)对应第1维。
- 负索引示例：输入维度索引支持负数，映射语义为回绕取模，`-i` 等价于 `(格式长度 - i)`。如源格式`NHWC`、目标格式`NCHW`时，输入`[-1, -2, -3, -4]`与`[3, 2, 1, 0]`的输出相同，均为`[1, 3, 2, 0]`。

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
    <td>维度索引张量，取值范围为[-N, N)，N为src_format的维度数（4维格式N=4，5维格式N=5，即x取值范围为[-4, 4)或[-5, 5)）。支持负数索引，-i表示倒数第i维（如-1表示最后一维）。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>src_format</td>
    <td>可选属性</td>
    <td><ul><li>源数据格式，枚举值："NCHW"、"NHWC"、"NCDHW"、"NDHWC"。</li><li>默认值为"NHWC"。</li></ul></td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dst_format</td>
    <td>可选属性</td>
    <td><ul><li>目标数据格式，枚举值："NCHW"、"NHWC"、"NCDHW"、"NDHWC"。</li><li>默认值为"NCHW"。</li></ul></td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>映射后的维度索引张量，数据类型与shape均与输入x一致。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- src_format和dst_format仅支持枚举取值："NCHW"、"NHWC"（4维）、"NCDHW"、"NDHWC"（5维）。
- src_format和dst_format必须包含相同的字符集（即相同的维度字母，顺序可不同），即两者需同为4维格式或同为5维格式。
- 输入x中元素的取值范围为[-N, N)，N为src_format的维度数（4维格式N=4，5维格式N=5）；超出该范围的输入不会被报错拒绝，将按与负索引相同的回绕取模周期语义参与计算，但正确性不作保证。
- 输入x的shape维度数不超过8。
- 输入x的总元素数以int64表示范围为上限，shape各维乘积溢出int64的输入将被报错拒绝。
- 输出y的数据类型必须与输入x一致，shape必须与输入x相同；不一致的输入将被算子在tiling阶段报错拒绝。
- 支持动态shape与动态rank：未知秩（-2）输入的输出同为未知秩；含未知维（-1）的shape须以运行期真实维度代入后执行，无法提供真实维度的输入将被拒绝。
- 其他非ND的公有格式（如NCHW、NHWC）标签的输入不触发算子侧报错，按存储布局逐元素映射，与ND等价；私有格式（NC1HWC0、FRACTAL_Z、FRACTAL_NZ、NDC1HWC0、NC1HWC0_C04、FRACTAL_Z_3D）输入将被报错拒绝。
- 非连续布局的输入由框架自动插入连续化处理后参与计算（算子声明AutoContiguous）。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式 | [test_geir_data_format_dim_map](./examples/test_geir_data_format_dim_map.cpp) | 通过[算子IR](./op_graph/data_format_dim_map_proto.h)构图方式调用DataFormatDimMap算子。 |
