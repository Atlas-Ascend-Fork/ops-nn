# MaxPoolExt2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>   |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    x     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 接口功能：
对于4维的输入张量，进行最大池化（max pooling）操作。支持NHWC和NCHW两种数据格式，支持SAME和VALID两种填充模式。
- 计算公式：
  - 当padding="SAME"时，out tensor的shape中H和W维度推导公式：

    $$
    [H_{out}, W_{out}]=[\lceil{\frac{H_{in}}{s_h}}\rceil, \lceil{\frac{W_{in}}{s_w}}\rceil]
    $$

  - 当padding="VALID"时，out tensor的shape中H和W维度推导公式：

    $$
    [H_{out}, W_{out}]=[\lceil{\frac{H_{in} - k_h + 1}{s_h}}\rceil, \lceil{\frac{W_{in} - k_w + 1}{s_w}}\rceil]
    $$

    当 k_h > H_in + 1（或 k_w > W_in + 1）时，公式结果为负，输出对应维度按 0 处理（与 Tiling 实现的 max(0, ·) 截断一致）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
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
      <td>x</td>
      <td>输入</td>
      <td>输入tensor，维度为4，公式中的x。</td>
      <td>FLOAT16、FLOAT、INT8、INT16、INT32、INT64、UINT8、UINT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出tensor，与x同dtype，维度为4。</td>
      <td>FLOAT16、FLOAT、INT8、INT16、INT32、INT64、UINT8、UINT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>最大池化的窗口大小，长度为4的列表，对应公式中的k_h和k_w。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>窗口移动的步长，长度为4的列表，对应公式中的s_h和s_w。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>属性</td>
      <td>填充模式，取值为"SAME"或"VALID"。SAME模式表示输出大小为输入大小除以步长向上取整，VALID模式表示输出大小为输入大小减去窗口大小加1再除以步长向上取整。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>数据格式，支持"NHWC"或"NCHW"，默认为"NHWC"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- **值域限制说明：**
  - x：输入tensor维度必须为4。
  - ksize：长度为4的列表。NHWC 格式下 ksize[0]=1、ksize[3]=1，空间轴为 ksize[1]（H）和ksize[2]（W）；NCHW 格式下 ksize[0]=1、ksize[1]=1，空间轴为 ksize[2]（H）和ksize[3]（W）。空间轴取值必须大于0。 <!-- codespell:ignore ksize -->
  - strides：长度为4的列表。NHWC 格式下 strides[0]=1、strides[3]=1，空间轴为 strides[1]（H）和strides[2]（W）；NCHW 格式下 strides[0]=1、strides[1]=1，空间轴为 strides[2]（H）和strides[3]（W）。空间轴取值必须大于0。
  - padding：取值必须为"SAME"或"VALID"。
  - data_format：取值必须为"NHWC"或"NCHW"，默认为"NHWC"。
  - 输入x与输出y的数据格式（tensor format）仅支持ND（算子注册契约，见 op_host/max_pool_ext2_def.cpp 的 Format 声明）；其他 format（如 FRACTAL_NZ）会被算子宿主侧拒绝。
  - `data_format` 属性只决定逻辑轴序（NHWC/NCHW），不改变 tensor format 的 ND 存储契约；不要用 NCHW/NHWC 作为 storage format 传入。
  - 输出y的dtype与输入x的dtype相同；输出y的shape由算子根据输入shape与属性推导（恒为4维），调用方不可独立设置输出 shape。
  - IR 原型 [op_graph/max_pool_ext2_proto.h](./op_graph/max_pool_ext2_proto.h) 的 REG_OP 在参数表 8 种 dtype 之外额外声明 DOUBLE（FLOAT64）与 QINT8（图侧原型注册范围，宽于交付实现）；当前 ascend950/ascend350 后端 OpDef 仅交付参数表中 dtype，上述两种类型无对应实现，图模式构图后无可用 Kernel。
  - 必选参数（x、ksize、strides、padding）缺失由框架层在构图/编译阶段拒绝，不会到达算子边界。 <!-- codespell:ignore ksize -->
- **特殊值语义（与同族 max_pool_with_argmax 的 nan_prop 默认一致）：** 忽略 NaN——窗口内 NaN 不参与最大值比较，取非 NaN 值的最大值；窗口内全为 NaN 时结果为 -inf。Inf 正常参与比较；+0.0 与 -0.0 按 IEEE 754 比较相等。
- **动态shape说明：** 支持动态shape输入。输入单个维度可标记为 -1（未知维度），此时输出对应维度推导结果为 -1；未知rank输入输出按 4 维未知 shape 处理。

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
    <td><a href="./examples/test_geir_max_pool_ext2.cpp">test_geir_max_pool_ext2</a></td>
    <td>算子IR定义：<a href="./op_graph/max_pool_ext2_proto.h">op_graph/max_pool_ext2_proto.h</a>。构建与运行：环境需已安装 CANN-toolkit 与编译好的算子包，在仓库根目录执行 <code>bash build.sh --run_example max_pool_ext2 graph</code>（编译并执行样例，输入数据由样例内置生成，覆盖各 dtype 与静态/动态 shape，通过标准为输出存在且数量正确）；详见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子快速调用</a>。</td>
  </tr>
</tbody>
</table>
