# DeformableOffsetsGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>      |    √     |
| <term>Atlas A2系列产品</term>       |    √     |

## 功能说明

- 算子功能：DeformableOffsets算子的目的是根据offsets（kernel采样点的偏移值）来收集用于卷积的特征采样点，并对其进行重组，方便Conv2d算子进行卷积计算。而DeformableOffsetsGrad即为这一过程的反向：以卷积反向传播得到的、相对于DeformableOffsets前向输出y的梯度`grad`为输入，结合前向的输入特征`x`与偏移值`offsets`，计算并输出`x`的梯度`grad_x`与`offsets`的梯度`grad_offsets`。

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
      <td>grad</td>
      <td>输入</td>
      <td>相对于DeformableOffsets前向输出y的梯度。shape为[N, outH*K_H, outW*K_W, C]，其中outH/outW由约束说明中的输出尺寸公式计算，C与x的通道数一致。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>DeformableOffsets前向的输入特征（原始输入数据）。shape为[N, inH, inW, C]。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>输入</td>
      <td>kernel采样点的偏移值。shape为[N, outH, outW, deformable_groups*K_H*K_W*3]，每个采样点含x/y偏移与mask共3个分量（modulated=true时），outH/outW由约束说明中的输出尺寸公式计算。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>输出</td>
      <td>x的梯度。shape与x相同：[N, inH, inW, C]。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>grad_offsets</td>
      <td>输出</td>
      <td>相对于输入offsets的梯度。shape与offsets相同：[N, outH, outW, deformable_groups*K_H*K_W*3]。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>必填属性</td>
      <td>滑动窗口的步长，包含4个整数的列表。data_format为"NHWC"时取值为(1, stridesH, stridesW, 1)，为"NCHW"时为(1, 1, stridesH, stridesW)；stridesH/stridesW不可为0，N维strides[0]必须为1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>必填属性</td>
      <td>填充维度，包含4个整数的列表，[padTop, padBottom, padLeft, padRight]。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>必填属性</td>
      <td>卷积核的大小，包含2个整数的列表，(K_H, K_W)。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilations</td>
      <td>可选属性</td>
      <td>每个维度的伸缩因子，包含4个整数的列表。默认值[1, 1, 1, 1]。N维dilations[0]必须为1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td>数据格式，取值仅支持"NCHW"或"NHWC"。OpDef注册的默认值为"NHWC"，但算子IR（proto）注册的默认值为"NCHW"，两者不一致；经GE图模式构图调用时不要依赖默认值，请显式指定该属性。该属性仅决定strides/dilations列表的下标解释顺序，输入张量的存储格式固定为NHWC。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformable_groups</td>
      <td>可选属性</td>
      <td>C轴上的分组数，默认值1。必须大于0，且必须等于offsets通道数/(K_H*K_W*3)（modulated=true时每个采样点含3个偏移分量）。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>可选属性</td>
      <td>布尔值，指定DeformableConv2D的版本，true表示v2，false表示v1，默认值true。当前仅支持true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入`grad`、`x`、`offsets`的shape必须为4D。
- **不支持空tensor**：`grad`、`x`、`offsets`任一输入的元素总数为0（任一维度为0）时，算子在Tiling阶段直接拒绝执行并通过日志上抛报错，不会得到空输出。
- 输入张量的存储格式固定为NHWC。
- `strides`、`pads`、`dilations`的长度必须为4，`ksize`的长度必须为2。
- `offsets`的H/W分别等于按输出尺寸公式计算出的outH/outW。
- `grad`的H/W必须分别等于`offsets`的H/W乘以`ksize`的对应元素；输出`grad_offsets`的H/W必须与`offsets`的H/W一致。
- `grad`、`x`、`offsets`、`grad_offsets`的batch维N必须一致。
- 按下方输出尺寸公式计算得到的outH/outW必须大于0；`x`的通道维C必须大于0。
- 索引路径与容量说明：`grad`/`x`/`offsets`任一张量元素总数不超过INT32上限（2147483647）时走int32索引内核；任一超过该上限时自动切换int64索引内核。单张量最大元素数受设备内存约束。
- 违反上述任一约束的输入在图编译/Tiling阶段被拒绝并经错误日志上抛；本算子无独立公开错误码。

**输出尺寸公式**（与[算子IR](./op_graph/deformable_offsets_grad_proto.h)定义一致，`//`表示整除）：

$$
outH = (inH + pads[0] + pads[1] - (dilationsH * (K_H - 1) + 1)) // stridesH + 1
$$

$$
outW = (inW + pads[2] + pads[3] - (dilationsW * (K_W - 1) + 1)) // stridesW + 1
$$

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| GE图模式 | [test_geir_deformable_offsets_grad](./examples/arch35/test_geir_deformable_offsets_grad.cpp)   | 通过[算子IR](./op_graph/deformable_offsets_grad_proto.h)构图方式调用DeformableOffsetsGrad算子。 |

- 算子已注册InferShape/InferDataType推导回调（`grad_x`的shape/dtype跟随`x`，`grad_offsets`跟随`offsets`），并开启动态shape/动态rank支持，可在GE中动态构图调用。
- 本算子无公开aclnnAPI入口（构建配置为`ACLNNTYPE aclnn_exclude`）；`op_api/`目录交付的L0级接口仅供算子内部级联复用，不对外承诺。
