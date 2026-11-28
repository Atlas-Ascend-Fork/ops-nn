# NllLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

  - 接口功能：计算负对数似然损失值。

  - 计算公式：

    当`reduction`为`none`时，

    $$
    \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
    l_n = - w_{y_n} x_{n,y_n}, \quad
    w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not=   \text{ignoreIndex}\},
    $$

    其中$x$是self, $y$是target, $w$是weight，$N$是batch的大小. 如果`reduction`不是`'none'`,那么：

    $$
    \ell(x, y) = \begin{cases}
        \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
        \text{if reduction} = \text{`mean';}\\
        \sum_{n=1}^N l_n,  &
        \text{if reduction} = \text{`sum'.}
    \end{cases}
    $$

    $$
    totalWeight = \sum_{n=1}^N w_{y_n}
    $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 910px"><colgroup>
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
        <td><ul><li>待进行nll_loss计算的入参，公式中的x。</li><li>shape支持(C,)、(N, C)或(N, C, H, W)，其中N表示batch大小，C表示类别数。</li></ul></td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>target</td>
        <td>输入</td>
        <td><ul><li>待进行nll_loss计算的入参，公式中的target，表示真实标签。</li><li>shape为()、(N,)或(N, H, W)，与x的维度对应。</li><li>元素取值范围为[0, C-1]，取值为ignore_index的元素不参与损失计算。</li></ul></td>
        <td>INT32、INT64、UINT8</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>可选输入</td>
        <td><ul><li>待进行nll_loss计算的入参，公式中的weight，表示每个类别的缩放权重。</li><li>shape为(C,)。</li></ul></td>
        <td>FLOAT、FLOAT16、BFLOAT16（与self一致）</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td><ul><li>待进行nll_loss计算的出参，公式中的L。</li><li>reduction为none时：x为(N, C)时shape为(N,)，x为(N, C, H, W)时shape为(N, H, W)，x为(C,)时shape为()（0维标量）；reduction为mean或sum时shape为(1,)。</li></ul></td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>total_weight</td>
        <td>输出</td>
        <td><ul><li>待进行nll_loss计算的出参，公式中的totalWeight，即各类别权重之和。</li><li>shape为(1,)，仅在reduction为mean或sum时有效，reduction为none时取值无效。</li></ul></td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>reduction</td>
        <td>可选属性</td>
        <td><ul><li>指定应用到输出的缩减方式：none表示不缩减；mean表示输出为各元素损失之和除以totalWeight；sum表示输出为各元素损失之和。</li><li>默认值为"mean"。</li></ul></td>
        <td>STRING</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ignore_index</td>
        <td>可选属性</td>
        <td><ul><li>指定一个被忽略且不影响损失计算的目标值。</li><li>默认值为-100，应取[0, C-1]之外且与target数据类型可比较的整数值。</li></ul></td>
        <td>INT64</td>
        <td>-</td>
      </tr>
    </tbody></table>

## 约束说明

  - target元素取值范围为[0, C-1]；取值为ignore_index的元素不参与损失计算。当前实现未对越界取值做显式校验，越界取值的行为不在公开契约内。
  - ignore_index取值为整数，默认-100；应取[0, C-1]之外且与target数据类型可比较的值，否则会跳过对应类别的损失计算。
  - 输入x仅支持1D (C,)、2D (N, C)或4D (N, C, H, W)；target相应为0D、1D (N,)或3D (N, H, W)，且各维度与x对应一致。
  - reduction取值为none、mean或sum，默认值为mean。
  - 输出total_weight仅在reduction为mean或sum时有效，reduction为none时取值无效。
  - 数据格式仅支持ND。
  - 空输入场景：输入x元素个数为0时，reduction为mean输出NaN、为sum输出0，total_weight为0；reduction为none时输出不写入。
  - x类别维C为0且target非空时属非法输入。

## 调用说明

  | 调用方式 | 调用样例                                                                   | 说明                                                           |
  |--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
  | aclnn API | [test_aclnn_nll_loss](./examples/test_aclnn_nll_loss.cpp) | 通过[aclnnNLLLoss](./docs/aclnnNLLLoss.md)接口方式调用NLLLoss算子。 |
  | aclnn API | [test_aclnn_nll_loss_2d](./examples/test_aclnn_nll_loss_2d.cpp) | 通过[aclnnNLLLoss2d](./docs/aclnnNLLLoss2d.md)接口方式调用NLLLoss算子。 |
  | GE图模式 | - | 通过[算子IR](./op_graph/nll_loss_proto.h)构图方式调用NLLLoss算子。 |

## UT测试

  - Host侧UT（含InferShape用例与arch35 Tiling用例）的构建命令：

    ```bash
    ./build.sh -u --ophost --noexec --soc=ascend950 -O0 -j8
    ```

    其中`--soc`用于指定算子编译目标Soc（支持ascend950等）；若不指定`--soc`，arch35 Tiling用例（tests/ut/op_host/arch35/test_nll_loss_tiling.cpp）不会被编译执行，仅运行InferShape等Host用例。去掉`--noexec`可在编译后直接执行用例。
