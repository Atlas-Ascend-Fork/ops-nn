# DeformableOffsets

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>      |    √     |
| <term>Atlas A2系列产品</term>       |    √     |

## 功能说明

- 算子功能：用于计算变形卷积（Deformable Convolution）输出的函数。通过引入偏移参数offsets，使得卷积核在输入特征图上的位置可以动态调整，从而适配不规则的几何变化。
- 计算公式：

  假定输入input的shape是[N, inH, inW, inC]，输出deformOut的shape为[N, outH\*K_H, outW\*K_W, inC]，根据已有属性计算outH、outW：

  $$
  outH = (inH + pads[0] + pads[1] - ((K\_H - 1) * dilations[1] + 1)) // strides[1] + 1
  $$

  $$
  outW = (inW + pads[2] + pads[3] - ((K\_W - 1) * dilations[2] + 1)) // strides[2] + 1
  $$

  标准卷积计算采样点下标：

  $$
  x = -pads[2] + ow*strides[2] + kw*dilations[2], ow的取值为(0, outW-1), kw的取值为(0, K\_W-1)
  $$

  $$
  y = -pads[0] + oh*strides[1] + kh*dilations[1], oh的取值为(0, outH-1), kh的取值为(0, K\_H-1)
  $$

  根据传入的`offsets`，进行变形卷积，计算偏移后的下标：

  $$
  (x,y) = (x + offsetX, y + offsetY)
  $$

  使用双线性插值计算偏移后点的值：

  $$
  (x_{0}, y_{0}) = (int(x), int(y)) \\
  (x_{1}, y_{1}) = (x_{0} + 1, y_{0} + 1)
  $$

  $$
  weight_{00} = (x_{1} - x) * (y_{1} - y) \\
  weight_{01} = (x_{1} - x) * (y - y_{0}) \\
  weight_{10} = (x - x_{0}) * (y_{1} - y) \\
  weight_{11} = (x - x_{0}) * (y - y_{0}) \\
  $$

  $$
  deformOut(x, y) = mask * (weight_{00} * input(x0, y0) + weight_{01} * input(x0, y1) + weight_{10} * input(x1, y0) + weight_{11} * input(x1, y1))
  $$

  其中，`input`表示输入张量（参数表中对应`x`），`offsets`表示偏移值，`deformOut`表示输出张量（参数表中对应`y`），`strides`指定高度和宽度方向上的步幅，`pads`指定高度和宽度方向上的填充，`ksize`指定卷积核大小（对应公式中的`K_H`、`K_W`），`dilations`指定每个维度上的膨胀因子，`offsetX`和`offsetY`表示`offsets`中的偏移分量，`mask`为`offsets`中对应卷积核位置的第三个分量（modulated=true时恒存在）。偏移后的采样点(x, y)及其插值四邻域点可能落在输入图像边界之外，kernel实际按“越界取0”处理。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>表示输入的原始数据。对应公式中的input。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>输入</td>
      <td>表示偏移值。对应公式中的offsets。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示变形卷积的输出。对应公式中的deformOut。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>必选属性</td>
      <td>表示指定卷积核在高度和宽度方向上的步幅，即(1, stridesH, stridesW, 1)。对应公式中的strides。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>必选属性</td>
      <td>指定在输入的高度和宽度方向上添加的填充。对应公式中的pads。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>必选属性</td>
      <td>指定卷积核的大小。对应公式中的K_H、K_W。size为2(K_H, K_W)，各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilations</td>
      <td>可选属性</td>
      <td>指定每个维度上的膨胀因子，即(1, dilationsH, dilationsW, 1)。默认为(1, 1, 1, 1)。对应公式中的dilations。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td>指定输入x的数据格式。默认为"NHWC"，当前仅支持"NHWC"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformable_groups</td>
      <td>可选属性</td>
      <td>指定输入x在C轴上的分组数。默认为1。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>可选属性</td>
      <td>指定变形卷积的版本，表示offsets中是否包含掩码。若为true，offsets中包含掩码；若为false，则不包含。默认为true，当前仅支持true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- `x`、`offsets`、`y`三者数据类型必须一致。
- `x`、`offsets`的shape均须为4维（N、H、W、C）。
- `ksize`各元素须大于0；`strides`的H/W分量（NHWC下第1、2个元素）必须大于0，N/C分量（第0、3个元素）必须为1；`dilations`的N/C分量（第0、3个元素）必须为1。
- `modulated`仅支持true（默认值true），即`offsets`中每个采样点包含高度偏移、宽度偏移、掩码（mask）共3个分量。
- `offsets`的shape须满足（NHWC）：
  - N维与`x`的N维一致；
  - H/W维等于按功能说明中的公式，由`x`与`pads`、`strides`、`ksize`、`dilations`推导出的卷积输出高/宽；
  - C维等于`deformable_groups * ksize[0] * ksize[1] * 3`（3为modulated=true时每个采样点的分量数）；
  - `deformable_groups`属性值须与由`offsets`的C维推导出的分组数一致，且`x`的C维须能被`deformable_groups`整除。
- 输出`y`的shape为(N, outH * ksize[0], outW * ksize[1], C)，其中N、C维与`x`一致。
- 不支持空tensor：`x`的C维必须大于0，且推导出的卷积输出高/宽均大于0。

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式 | [test_geir_deformable_offsets](./examples/test_geir_deformable_offsets.cpp)   | 通过[算子IR](./op_graph/deformable_offsets_proto.h)构图方式调用DeformableOffsets算子。 |
