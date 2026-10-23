# AvgPool1d

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                       |    √     |
| <term>Atlas A3系列产品</term>     |    ×     |
| <term>Atlas A2系列产品</term>      |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：对输入tensor的最后一维做一维平均池化（1D average pooling），支持不对称padding、ceil_mode输出长度修正和count_include_pad分母选择。

- 计算公式：

设输入x的shape为(..., L)，pads=[pad_left, pad_right]，padding后长度L_padded = L + pad_left + pad_right。输出长度L_out按以下公式计算：

$$
L_{out} = \begin{cases} \left\lfloor \frac{L_{padded} - ksize}{strides} \right\rfloor + 1, & ceil\_mode = False \\ \left\lceil \frac{L_{padded} - ksize}{strides} \right\rceil + 1, & ceil\_mode = True \end{cases}
$$

当pad_left > 0且最后一个窗口的起始位置落在padding区时（即(L_out - 1) * strides >= L + pad_left），L_out减1。若按公式计算得到的L_out <= 0，则按0处理并输出空tensor（不报错）。

每个输出窗口的值为：

$$
y[..., w] = \frac{\sum_{i=0}^{ksize-1} x_{padded}[..., w \times strides + i]}{divisor}
$$

其中divisor在count_include_pad=True时为窗口在padded空间的实际元素数（ceil_mode下可能小于ksize），在count_include_pad=False时为窗口与原始数据区的交集大小（最小为1）。

## 参数说明

<table><thead>
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
    <td>待进行一维平均池化计算的入参，shape为(..., L)，L为池化方向维度长度。</td>
    <td>float16、float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>一维平均池化计算结果，shape为(..., L_out)，与x同dtype。</td>
    <td>float16、float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ksize</td>
    <td>属性</td>
    <td>池化窗口大小。</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>strides</td>
    <td>属性</td>
    <td>池化窗口滑动步长。</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>pads</td>
    <td>属性</td>
    <td>左右padding大小，格式为[pad_left, pad_right]，支持不对称padding。</td>
    <td>ListInt</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ceil_mode</td>
    <td>属性</td>
    <td>为True时用ceil模式计算输出长度，为False时用floor模式。默认False。</td>
    <td>bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>count_include_pad</td>
    <td>属性</td>
    <td>为True时分母含padding元素，为False时仅含有效元素。默认False。</td>
    <td>bool</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- 输入x的rank需大于等于3，支持3D~8D。
- ksize需大于等于1。
- strides需大于等于1（stride=0将报错）。
- pads的元素需大于等于0，长度为2。
- 输入x仅支持float16、float32类型。
- 数据格式仅支持ND。
- 输出y与输入x同dtype。
- 支持空tensor输入（L=0时输出空tensor）。
- 若按公式计算得到的L_out <= 0（例如ksize大于L_padded），L_out按0处理并输出空tensor，算子不报错；PyTorch在同类配置下会报错（Output size is too small），此处为有意偏离。
- 当pad_left > 0且最后一个窗口的起始位置落在padding区时（即(L_out - 1) * strides >= L + pad_left），L_out减1；pad_left为0时不做该修正（与PyTorch的ceil_mode行为有意不同：保留该末窗口可避免PyTorch在pad_l=0时的NaN问题）。
- pad_left为0且最后一个窗口完全落在原始数据区之外时，该窗口保留在输出中并按补零窗口计算（无有效元素，结果为0）。
- divisor在count_include_pad=False时取窗口与原始数据区的交集大小，最小为1，不会出现除零错误。
- 输入x与输出y需为连续tensor；算子声明AutoContiguous，非连续输入由框架自动转换为连续tensor。
- 确定性实现：采用单线程内顺序累加，不使用原子操作，相同输入多次执行结果一致。
- NaN输入按IEEE 754规则传播，Inf输入按IEEE 754规则传播。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>样例代码</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE图模式</td>
    <td><a href="./examples/test_geir_avg_pool1_d.cpp">test_geir_avg_pool1_d</a></td>
    <td>通过<a href="./op_graph/avg_pool1_d_proto.h">avg_pool1_d_proto.h</a>定义算子IR构图方式调用AvgPool1D算子，详见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>。</td>
  </tr>
</tbody>
</table>
