# Dilation2D

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>   |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：形态学膨胀（Morphological Dilation）操作的二维实现。对输入图像的每个通道独立执行滑窗操作：在每个窗口位置，取输入值与滤波器值之和的最大值作为输出。

- 计算公式：

$$
y[b, oh, ow, c] = \max_{fh, fw} (x[b, oh \times stride_h + fh \times rate_h - pad\_top, ow \times stride_w + fw \times rate_w - pad\_left, c] + filter[fh, fw, c])
$$

其中 $pad\_top$、$pad\_left$ 由 padding 模式决定：SAME/CALCULATED 模式按下方公式计算，VALID 模式为 0。超出输入边界的位置视为负无穷，不参与 max 比较。

## 输出尺寸计算

记有效滤波器尺寸 $filter\_h\_eff = fH + (fH-1) \times (rate\_h-1)$，$filter\_w\_eff$ 同理。

- SAME：$oH = \lceil H / stride\_h \rceil$，$oW = \lceil W / stride\_w \rceil$；pad 按 $pad_h = \max((oH-1) \times stride\_h + filter\_h\_eff - H, 0)$ 计算后上下均分（$pad\_top = pad\_h // 2$）。
- VALID：$oH = \max(0, \lfloor (H - filter\_h\_eff + stride\_h) / stride\_h \rfloor)$，$oW$ 同理；无 padding。
- CALCULATED：使用 pads 显式指定 $[H\_top, H\_bottom, W\_left, W\_right]$，记 $hNum = H + H\_top + H\_bottom - filter\_h\_eff$：
  - ceil_mode=false（floor 口径）：$oH = \max(0, \lfloor hNum / stride\_h \rfloor + 1)$；$hNum < 0$ 时输出为空（$oH = 0$）。
  - ceil_mode=true（ceil 口径）：$oH = \max(0, \lceil hNum / stride\_h \rceil + 1)$。
  - $oW$ 同理。

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
      <td>4D输入张量，NHWC或NCHW格式。NHWC时filter shape为(fH,fW,C)，NCHW时为(C,fH,fW)。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>filter</td>
      <td>输入</td>
      <td>3D滤波器张量，C维度与x的C一致。NHWC时shape为(fH,fW,C)，NCHW时为(C,fH,fW)。</td>
      <td>与x相同</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>4D输出张量，oH/oW由输入shape+属性推导。</td>
      <td>与x相同</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>可选属性</td>
      <td>4个正整数，滑窗步幅。NHWC时为[sN, sH, sW, sC]，NCHW时为[sN, sC, sH, sW]。取值范围[1, 2147483647]。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rates</td>
      <td>可选属性</td>
      <td>4个正整数，膨胀率。NHWC时为[rN, rH, rW, rC]，NCHW时为[rN, rC, rH, rW]。取值范围[1, 2147483647]。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>可选属性</td>
      <td>padding模式，"SAME"、"VALID"或"CALCULATED"，默认"SAME"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>可选属性</td>
      <td>4个非负整数，显式padding值[H_top, H_bottom, W_left, W_right]，默认{0,0,0,0}，仅CALCULATED模式生效。取值范围[0, 2147483647]。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>可选属性</td>
      <td>输出尺寸计算使用ceil还是floor，默认false。仅CALCULATED模式生效。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td>数据格式，"NHWC"或"NCHW"，默认"NHWC"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入x必须为4D张量，filter必须为3D张量。
- x和filter的dtype必须一致，仅支持FLOAT16。
- filter的通道数（C维度）必须与x的通道数一致。NHWC时filter shape为(fH,fW,C)，NCHW时为(C,fH,fW)。
- 输入维度与strides/rates/pads取值均不超过2147483647（int32上限），超出的输入会被拒绝。
- strides和rates必须为4个正整数。NHWC时strides[0]=strides[3]=1, rates[0]=rates[3]=1；NCHW时strides[0]=strides[1]=1, rates[0]=rates[1]=1。
- CALCULATED模式下pads必须小于有效滤波器尺寸，pads格式为[H_top, H_bottom, W_left, W_right]。
- 超出输入边界的位置视为负无穷，不参与max比较。
- 支持SAME、VALID和CALCULATED三种padding模式。
- 支持动态shape：输入含-1（未知维度）时按可计算维度推导输出，无法计算的维度输出-1。
- 仅支持Ascend 950。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE图模式 | [test_geir_dilation2_d](./examples/arch35/test_geir_dilation2_d.cpp) | 通过[算子IR](./op_graph/dilation2_d_proto.h)构图方式调用Dilation2D算子。 |
