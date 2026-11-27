# aclnnRotateDualLevelQuant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：对输入 x 进行矩阵乘旋转变换，执行可选的 clamp 操作，再沿尾轴依次进行
  level0 量化和 level1 MX 量化。
  输出量化结果 yOut、FP32 一级尺度 level0ScaleOut 和 E8M0 二级尺度 level1ScaleOut。
  yOut 支持 FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2。
- 可选功能：groupList 用于沿 M 轴划分连续行组，各组使用对应的旋转矩阵；
  alpha 用于控制每 32 个元素的 clamp 范围。
- 本接口沿用 Ascend 950 上的旋转、分组和可选 clamp 规则，在 MX 量化前增加每 512 个元素的一级归一化，并额外返回 FP32 一级尺度。旋转与分组规则可参见 [aclnnRotateQuantV2](../../../../matmul/rotate_quant/docs/aclnnRotateQuantV2.md)。
- 计算公式：先计算旋转及 clamp 后的数据 V，再按每 512 个元素计算一级尺度及归一化中间值 temp，
  最后对 temp 每 32 个元素计算一个二级 MX 尺度并生成 yOut。

记 x 的维数为 D、shape 为 `(d0,...,d[D-2],N)`，令 `X=x.reshape(M,N)`，
M 为前导维的乘积，一维输入时 M=1。K 为旋转矩阵的阶数，B=N/K 为每行的旋转块数。
以下公式描述有效数据为有限数、块最大绝对值非零且尺度处于常规范围内的计算。零值、非有限中间值和尺度边界见[特殊数值与尺度边界](#特殊数值与尺度边界)。

### 1. 旋转变换

当所有行和列块共享一个 `(K,K)` 旋转矩阵 R 时，执行右乘旋转：

$$
U=\left(X.\operatorname{reshape}(MB,K)\ @\ R\right).\operatorname{reshape}(M,N)
$$

对普通及分组模式，统一写为：

$$
U_{m,bK+c}=\sum_{i=0}^{K-1}X_{m,bK+i}\,R_{e(m),b,i,c},\qquad 0\le c<K
$$

其中，m 为本次处理的原始行号，`0≤b<B`。普通模式处理全部 M 行，分组模式处理 groupList 指定的连续前缀。
`R[e(m),b]` 的选择规则如下：

| 模式 | rotation 的 shape | 旋转矩阵的选择 |
|---|---|---|
| 普通模式 | `(K,K)` | 所有行及所有列块共享一个矩阵 |
| 普通模式 | `(B,K,K)` | 第 b 个列块使用 rotation[b] |
| M 轴分组模式 | `(E,K,K)` | 第 e 组的所有列块共享 rotation[e] |
| M 轴分组模式 | `(E,B,K,K)` | 第 e 组、第 b 个列块使用 rotation[e,b] |

普通模式由 `groupList=nullptr` 指定，可令 e(m)=0；分组模式由 groupList 描述连续行组，
e(m) 表示第 m 行所属的组。
矩阵乘中间结果采用 BF16 数据域，记 `Z=cast_BF16(U)`；即使 x 和 rotation 为 FLOAT16，Z 仍为 BF16。

### 2. 可选 clamp

每行沿 N 每 32 个元素为一组。记第 h 个组为
`H[h]=[32*h,min(32*(h+1),N))`，计算：

$$
a^{Z}_{m,h}=\max_{j\in H[h]}|Z_{m,j}|,\qquad
L_{m,h}=\operatorname{cast}_{\mathrm{BF16}}\left(\alpha\,a^{Z}_{m,h}\right)
$$

当 alpha 不为 nullptr 且数值在 `(0.0,1.0)` 内时：

$$
V_{m,j}=\operatorname{clamp}\left(Z_{m,j},-L_{m,h(j)},L_{m,h(j)}\right)
$$

否则 `V=Z`。其中 `h(j)=floor(j/32)`。V 为进入二级量化前的 BF16 数据。
clamp 发生在 level0 之前，level0 以 V 为输入。

### 3. 二级量化

记 `k0=level0BlockSize=512`、`k1=level1BlockSize=32`，分别为一级量化与二级 MX 量化的块大小。
对每行的一个一级块，先计算一级尺度 s0 和归一化中间值 temp：

$$
\{V_{m,j}\}_{j\in G[g]}
\xrightarrow{\mathrm{level0}}
\left\{s_{0,m,g},\{temp_{m,j}\}_{j\in G[g]}\right\}
$$

再将 temp 划分为二级块：

$$
\{temp_{m,j}\}_{j\in H[h]}
\xrightarrow{\mathrm{level1\ MX}}
\left\{s_{1,m,h},\{y_{m,j}\}_{j\in H[h]}\right\}
$$

#### 3.1 level0：一级量化

每行沿原始 N 轴按 level0BlockSize 个元素划分一级块，记
`G[g]=[k0*g,min(k0*(g+1),N))`，其中 k0=512。最后一个块不足 512 时，仅统计有效元素。

$$
input\_max_{m,g}=\max_{j\in G[g]}|V_{m,j}|
$$

$$
\widehat{s}_{0,m,g}=\begin{cases}
\mathrm{RN}_{32}(input\_max_{m,g}\cdot\mathrm{RN}_{32}(1/6)), & \mathrm{FP4\ E2M1}\\
\mathrm{RN}_{32}(input\_max_{m,g}/C_{\mathrm{MAX}}), & \mathrm{FP8}
\end{cases},\qquad
s_{0,m,g}=\begin{cases}
0, & \widehat{s}_{0,m,g}<2^{-126}\\
\widehat{s}_{0,m,g}, & \text{其他情况}
\end{cases}
$$

$$
temp_{m,j}=\operatorname{cast}_{\mathrm{BF16}}
\left(\frac{V_{m,j}}{s_{0,m,g(j)}}\right),\qquad g(j)=\lfloor j/k_0\rfloor
$$

一级尺度 `s0` 以 FLOAT32 写入 level0ScaleOut。V 除以对应的一级尺度后转换为 BF16，得到第二级量化的输入 temp。
RN₃₂ 表示 FP32 最近偶数舍入。FP4 一级尺度沿用 DynamicDualLevelMxQuant 的 FP32 倒数乘法，两种 FP8 的一级尺度直接按 FP32 除法计算。
FP8 归一化使用正确舍入的 FP32 商，再按最近偶数舍入转换为 BF16。
一级分母由 yOut 的类型决定，FP8 场景使用对应类型的最大有限值：

| yOut 数据类型 | 一级分母 CMAX | scaleAlg=0 使用的 emax |
|---|---:|---:|
| FLOAT4_E2M1 | 6 | 2 |
| FLOAT8_E4M3FN | 448 | 8 |
| FLOAT8_E5M2 | 57344 | 15 |

一级块可以包含多个 K 宽旋转块。512/32 两级分块始终沿原始 N 轴进行，每行独立计算，
不按旋转块重新划分一级块，也不跨原始行或 M 轴分组边界归约。

#### 3.2 level1：MX 量化

对 temp 每行沿 N 每 level1BlockSize 个元素计算一个共享的 E8M0 尺度。
当前 k1=32，二级块采用前述 H[h] 的划分。记：

$$
a_{m,h}=\max_{j\in H[h]}|temp_{m,j}|
$$

根据 scaleAlg 选择第二级尺度 `s1`，以 E8M0 写入 level1ScaleOut。
计算量化结果时，先将 temp 与内部 BF16 乘数 `r` 相乘，乘积转换为 BF16 后，再按 roundMode 转换为目标类型。
FP8 的最终转换采用饱和模式。

$$
y_{m,j}=\operatorname{cast}_{\mathrm{yOut.dtype}}
\left(\operatorname{cast}_{\mathrm{BF16}}\left(temp_{m,j}\cdot r_{m,h(j)}\right),\operatorname{roundMode}\right)
$$

常规范围内，`r` 为 `s1` 的 BF16 倒数。编码 0 的处理随量化算法而异，见[特殊数值与尺度边界](#特殊数值与尺度边界)。

**场景一：scaleAlg=0，支持 FP4 和 FP8。**

按块最大绝对值的指数计算共享尺度：

$$
e_{m,h}=\left\lfloor\log_2(a_{m,h})\right\rfloor-emax
$$

$$
s_{1,m,h}=2^{e_{m,h}}
$$

emax 为上表中目标类型的最大正规数指数。该算法按指数选取尺度，不能保证所有小块均不发生饱和。

**场景二：scaleAlg=1，仅支持 FLOAT8_E4M3FN 和 FLOAT8_E5M2。**

先按 FP8 最大有限值计算浮点尺度，再向上选择可表示的二次幂尺度。
对常规有限正尺度，有：

$$
u_{m,h}=\mathrm{RN}_{32}(a_{m,h}\cdot\mathrm{RN}_{32}(1/C_{\mathrm{MAX}})),\qquad
s_{1,m,h}=2^{\lceil\log_2(u_{m,h})\rceil}
$$

FP32 中间尺度为次正规数时，按[特殊数值与尺度边界](#特殊数值与尺度边界)中的规则选取 E8M0 编码。

**场景三：scaleAlg=2，仅支持 FLOAT4_E2M1。**

ACLNN 先校验双精度 dstTypeMax 在 [6.0,12.0] 内且有限，再转为 FP32 分母 d。以下分派和计算均以 d 为准。

- d 为 **6.0 或 7.0**：根据块最大值的 BF16 尾数字段选择共享指数。
  对常规 BF16 正规数块最大值，写作 `a=2^p*(1+f)`，其中 `0<=f<1`，有：

  $$
  \delta=\begin{cases}
  1,&d=6.0\ \text{且}\ f>1/2\\
  1,&d=7.0\ \text{且}\ f>3/4\\
  0,&\text{其他情况}
  \end{cases}
  $$

  $$
  s_1=2^{p-2+\delta}
  $$

  等价地，BF16 的 7 位尾数字段分别大于 64 或 96 时，对共享指数加一。

- d 为 [6.0,12.0] 内的其他值时，先直接计算正确舍入的 FP32 商，再选择 E8M0 二次幂尺度。
  对常规有限正规数尺度，有：

  $$
  u_{m,h}=\mathrm{RN}_{32}(a_{m,h}/d),\qquad
  s_{1,m,h}=2^{\lceil\log_2(u_{m,h})\rceil}
  $$

`scaleAlg`、`dstTypeMax` 仅影响第二级尺度计算，不改变一级分母 CMAX。
roundMode 控制最终 y 的类型转换，不改变两个 BF16 中间结果的类型。

#### 3.3 量化输出

量化数据 y 按原始元素位置写入 yOut；s0 和 s1 分别按 512、32 元素的块位置写入
level0ScaleOut 和 level1ScaleOut。概念上的恢复关系为：

$$
\hat V_{m,j}=s_{0,m,g(j)}\,s_{1,m,h(j)}\,y_{m,j}
$$

该关系包含 BF16 中间转换、尺度存储和最终低精度量化造成的误差。
它恢复的是旋转及可选 clamp 后的数据 V，不直接恢复原始输入 x。

以 N=576、K=64 为例，每行包含 9 个旋转块，一级量化仍按原始 N 轴划分：

| 一级块 | 原始列范围（含端点） | 覆盖的旋转块 | 对应二级尺度数 |
|---|---|---|---:|
| g=0 | 0～511 | b=0～7，共 8 个 K 宽块 | 16 |
| g=1 | 512～575 | b=8，共 1 个 K 宽块 | 2 |

因此每行有 2 个一级尺度和 18 个有效二级尺度，后者按 9 对 E8M0 编码存放。

## 函数原型

本算子采用[两段式接口](../../../../docs/zh/context/two_phase_api.md)：
先调用 `aclnnRotateDualLevelQuantGetWorkspaceSize` 获取所需 workspace 大小及执行器，
再调用 `aclnnRotateDualLevelQuant` 执行。

```cpp
aclnnStatus aclnnRotateDualLevelQuantGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* rotation,
    const aclTensor* alpha,
    const aclTensor* groupList,
    int64_t groupListType,
    int64_t axis,
    char* roundMode,
    int64_t scaleAlg,
    double dstTypeMax,
    bool trans,
    int64_t level0BlockSize,
    int64_t level1BlockSize,
    aclTensor* yOut,
    aclTensor* level0ScaleOut,
    aclTensor* level1ScaleOut,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```cpp
aclnnStatus aclnnRotateDualLevelQuant(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## aclnnRotateDualLevelQuantGetWorkspaceSize

- **参数说明**

所有 Tensor 均为 Device 侧 Tensor，数据格式为 ND。三路输出须由调用方预先分配。

<table style="table-layout: fixed; width: 1500px">
  <colgroup>
    <col style="width: 250px">
    <col style="width: 100px">
    <col style="width: 220px">
    <col style="width: 420px">
    <col style="width: 180px">
    <col style="width: 80px">
    <col style="width: 135px">
    <col style="width: 115px">
  </colgroup>
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">x（const aclTensor*）</td>
      <td>必选输入</td>
      <td>待旋转量化的输入张量。</td>
      <td>支持满足 shape 约束的空 Tensor。各维及前导维合并后的 M 均不得超过 INT32_MAX。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>普通模式 1～7 维；分组模式为 (M,N)，仅限二维。</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">rotation（const aclTensor*）</td>
      <td>必选输入</td>
      <td>旋转矩阵。</td>
      <td>数据类型与 x 一致。普通模式支持满足 shape 约束的空 Tensor；分组模式不支持空 Tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>普通模式 2 或 3 维；分组模式 3 或 4 维，具体 shape 见约束说明。</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">alpha（const aclTensor*）</td>
      <td>可选输入</td>
      <td>clamp 范围的缩放系数。</td>
      <td>可传 nullptr。不支持空 Tensor。仅当传入 Tensor 且数值在 (0.0, 1.0) 内时启用 clamp。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>(1,)，一维。</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupList（const aclTensor*）</td>
      <td>可选输入</td>
      <td>M 轴连续行分组列表，各组使用对应 rotation。</td>
      <td>nullptr 表示普通模式；非空指针时不支持空 Tensor，1≤E≤1024。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(E,)，一维。</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupListType（int64_t）</td>
      <td>输入</td>
      <td>groupList 的表示方式。</td>
      <td>0 表示累计和 cumsum，1 表示各组行数 count；groupList 为 nullptr 时也必须取 0 或 1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">axis（int64_t）</td>
      <td>输入</td>
      <td>level0、level1 量化共同使用的尾轴。</td>
      <td>仅支持 -1 或 D-1，其中 D 为 x 的维数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">roundMode（char*）</td>
      <td>输入</td>
      <td>level1 阶段最终 y 的类型转换模式。</td>
      <td>可取 rint、round、floor。传入 nullptr 时采用 rint。与 yOut 类型的组合见约束说明。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">scaleAlg（int64_t）</td>
      <td>输入</td>
      <td>level1 MX 尺度的计算算法。</td>
      <td>可取 0、1、2。与 yOut 类型的组合见约束说明。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">dstTypeMax（double）</td>
      <td>输入</td>
      <td>scaleAlg=2 时计算第二级尺度使用的范围参数。</td>
      <td>与 scaleAlg 的组合见约束说明。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">trans（bool）</td>
      <td>输入</td>
      <td>是否转置量化结果 y。</td>
      <td>仅支持 false。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">level0BlockSize（int64_t）</td>
      <td>输入</td>
      <td>第一级量化块大小，对应公式中的 k0。</td>
      <td>须显式传入 512。其他取值返回 ACLNN_ERR_PARAM_INVALID（161002），空输入也适用。对应的 Graph 属性 <code>level0_block_size</code> 默认值为 512。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">level1BlockSize（int64_t）</td>
      <td>输入</td>
      <td>第二级 MX 量化块大小，对应公式中的 k1。</td>
      <td>须显式传入 32。其他取值返回 ACLNN_ERR_PARAM_INVALID（161002），空输入也适用。对应的 Graph 属性 <code>level1_block_size</code> 默认值为 32。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">yOut（aclTensor*）</td>
      <td>必选输出</td>
      <td>经过旋转、可选 clamp、level0 及 level1 后的最终量化结果。</td>
      <td>shape 与 x 一致，数据类型决定量化的目标类型。x 为合法空 Tensor 时，须传入同 shape 的空 Tensor。</td>
      <td>FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>与 x 相同，1～7 维。</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">level0ScaleOut（aclTensor*）</td>
      <td>必选输出</td>
      <td>由旋转及 clamp 后的数据 V 计算的 FP32 一级尺度，对应公式中的 s0。</td>
      <td>shape 见输出布局。x 为合法空 Tensor 时，须传入满足 shape 约束的空 Tensor。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>与 x 的维数相同，1～7 维。</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">level1ScaleOut（aclTensor*）</td>
      <td>必选输出</td>
      <td>由归一化中间值 temp 计算的 E8M0 二级尺度，对应公式中的 s1。</td>
      <td>shape 见输出布局。x 为合法空 Tensor 时，须传入满足 shape 约束的空 Tensor。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>比 x 多一维，2～8 维。</td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">workspaceSize（uint64_t*）</td>
      <td>必选输出</td>
      <td>返回本次调用所需的 Device workspace 字节数。</td>
      <td>不可传 nullptr，按返回值申请 workspace。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">executor（aclOpExecutor**）</td>
      <td>必选输出</td>
      <td>返回包含算子计算流程的执行器。</td>
      <td>不可传 nullptr；传给第二段接口，调用后不可重复使用。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

- **返回值**

返回 aclnnStatus 状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。
本接口的主要返回情况如下：

<table style="table-layout: fixed; width: 1000px">
  <colgroup>
    <col style="width: 320px">
    <col style="width: 100px">
    <col style="width: 580px">
  </colgroup>
  <thead>
    <tr><th>返回值</th><th style="white-space: nowrap">错误码</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>ACLNN_SUCCESS</td><td>0</td><td>成功创建执行计划。</td></tr>
    <tr><td>ACLNN_ERR_PARAM_NULLPTR</td><td>161001</td><td>x、rotation、三路输出、workspaceSize 或 executor 存在非法空指针。</td></tr>
    <tr><td>ACLNN_ERR_PARAM_INVALID</td><td>161002</td><td>Tensor 的数据类型、格式、shape 或标量参数取值不符合约束。</td></tr>
    <tr><td>ACLNN_ERR_RUNTIME_ERROR</td><td>361001</td><td>当前运行平台不支持本接口。</td></tr>
    <tr><td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td><td>561101</td><td>内部创建执行器失败。</td></tr>
    <tr><td>ACLNN_ERR_INNER_NULLPTR</td><td>561103</td><td>内部准备输入或输出、添加算子任务或添加输出回写任务失败。</td></tr>
  </tbody>
</table>

当 workspaceSize 和 executor 两个指针本身均有效时，接口先将其指向的值设为 0 和 nullptr；
后续校验或计划创建失败时，不返回可执行的计划。第一段返回失败后，不应调用第二段接口。

## aclnnRotateDualLevelQuant

- **参数说明**

<table style="table-layout: fixed; width: 1000px">
  <colgroup>
    <col style="width: 220px">
    <col style="width: 110px">
    <col style="width: 670px">
  </colgroup>
  <thead>
    <tr><th style="white-space: nowrap">参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td style="white-space: nowrap">workspace（void*）</td><td>输入</td><td>Device workspace 地址。workspaceSize 为 0 时可传 nullptr。</td></tr>
    <tr><td style="white-space: nowrap">workspaceSize（uint64_t）</td><td>输入</td><td>已申请的 workspace 字节数，由第一段接口返回。</td></tr>
    <tr><td style="white-space: nowrap">executor（aclOpExecutor*）</td><td>输入</td><td>第一段成功返回的执行器。须为有效指针，每个执行器只能提交一次。</td></tr>
    <tr><td style="white-space: nowrap">stream（aclrtStream）</td><td>输入</td><td>指定执行任务的 Stream。</td></tr>
  </tbody>
</table>

- **返回值**

返回 aclnnStatus 状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。
executor 为 nullptr，或 workspaceSize 非零而 workspace 为 nullptr 时，返回
`ACLNN_ERR_PARAM_NULLPTR`（161001）。
第二段接口提交任务后，读取输出或释放相关资源前应同步等待 Stream 上的任务完成。

## 约束说明

### 输入与属性组合

- x 和 rotation 必须同时为 FLOAT16 或同时为 BFLOAT16。
- K 只能取 32、64、128、256，rotation 最后两维须相等，N 必须为 K 的整数倍。
  因此非空输入的 N 也是 32 的整数倍，但不要求 N 为 512 的整数倍。
- 普通模式下 x 为 `(*,N)`，维数为 1～7；rotation 为 `(K,K)` 或 `(N/K,K,K)`。
- 分组模式下 x 为 `(M,N)`，仅限二维，M≥0、N>0；rotation 为 `(E,K,K)` 或 `(E,N/K,K,K)`。
  E 必须等于 groupList 的长度。
- ACLNN 调用须使用具体的非负 shape。
- yOut 为 FLOAT4_E2M1 时，roundMode 支持 rint、round、floor，scaleAlg 支持 0、2。
  yOut 为两种 FP8 时，roundMode 仅支持 rint，scaleAlg 支持 0、1。
- scaleAlg=2 时，dstTypeMax 为 `[6.0,12.0]` 内的有限值。其他情况仅支持 0.0。

支持的 Tensor 类型组合如下，alpha 和 groupList 均可按条件省略：

| x | rotation | alpha | groupList | yOut | level0ScaleOut | level1ScaleOut |
|---|---|---|---|---|---|---|
| FLOAT16 | FLOAT16 | BFLOAT16 | INT64 | FLOAT4_E2M1 | FLOAT32 | FLOAT8_E8M0 |
| BFLOAT16 | BFLOAT16 | BFLOAT16 | INT64 | FLOAT4_E2M1 | FLOAT32 | FLOAT8_E8M0 |
| FLOAT16 | FLOAT16 | BFLOAT16 | INT64 | FLOAT8_E4M3FN | FLOAT32 | FLOAT8_E8M0 |
| BFLOAT16 | BFLOAT16 | BFLOAT16 | INT64 | FLOAT8_E4M3FN | FLOAT32 | FLOAT8_E8M0 |
| FLOAT16 | FLOAT16 | BFLOAT16 | INT64 | FLOAT8_E5M2 | FLOAT32 | FLOAT8_E8M0 |
| BFLOAT16 | BFLOAT16 | BFLOAT16 | INT64 | FLOAT8_E5M2 | FLOAT32 | FLOAT8_E8M0 |

### 分组约束

- groupListType=0：列表为非负、单调不减的累计行数，最后一个值不超过 M。
- groupListType=1：列表元素为非负的各组行数，总和不超过 M。
- 分组从第 0 行开始连续划分。允许空组，空组仍占用 rotation 的组下标；
  例如 count `[0,2,1]` 的两段有效行分别使用 rotation[1] 和 rotation[2]。
- 允许仅覆盖 M 的连续前缀，未覆盖行的 yOut、level0ScaleOut、level1ScaleOut 及其 padding
  均保持调用前内容。全部组为空时不更新输出；M=0 时各组行数应全部为 0。
- groupList 的设备数据值由调用方保证满足上述条件，Host 仅检查其类型、格式及 shape。
  列表值不合法时，不保证输出符合预期。
- M 轴分组用于选择旋转矩阵；每行沿 N 轴的 512/32 两级量化粒度不随 groupList 改变。
  一个一级块内最多有 16 个完整的 32 元素二级块，末尾不足 512 时仅处理其有效二级块。

### 输出 shape 与尺度布局

三路输出的顺序固定为 yOut、level0ScaleOut、level1ScaleOut。
令 `q0=ceil(N/level0BlockSize)`、`q1=ceil(N/level1BlockSize)`，当前分别为 `ceil(N/512)`、`ceil(N/32)`，有：

$$
\operatorname{shape}(yOut)=\operatorname{shape}(x)
$$

$$
\operatorname{shape}(level0ScaleOut)=(d_0,\ldots,d_{D-2},q_0)
$$

$$
\operatorname{shape}(level1ScaleOut)
=(d_0,\ldots,d_{D-2},\lceil q_1/2\rceil,2)
=(d_0,\ldots,d_{D-2},\lceil N/64\rceil,2)
$$

- level0ScaleOut 的维数与 x 相同；level1ScaleOut 的维数比 x 多 1。
- 每行第 h 个有效二级尺度存放于 `level1ScaleOut[...,h/2,h%2]`，其中 `h/2` 采用整数除法。
- q1 为奇数时，最后一个存储字节为 padding，填充值为 E8M0 **编码 0**；
  编码 0 表示数值 `2^-127`，不表示浮点数值零，padding 不参与有效数据恢复。
- padding 仅写入本次处理的行；未覆盖行的原有 padding 内容保持不变。

例如：

| x shape | K | level0ScaleOut shape | level1ScaleOut shape | 每行有效二级尺度/存储字节 |
|---|---:|---|---|---|
| `(4,576)` | 64 | `(4,2)` | `(4,9,2)` | 18/18，无 padding |
| `(4,544)` | 32 | `(4,2)` | `(4,9,2)` | 17/18，最后 1 字节为 padding |
| `(512,)` | 128 | `(1,)` | `(8,2)` | 16/16，无 padding |

### 特殊数值与尺度边界

- 一级量化的最大绝对值只统计 `V` 中的有限值，无有限值时取 0。
  若上式的候选尺度小于 `2^-126`，则将一级尺度写为正零。归一化时保留 `V` 中零元素的符号。
- 二级量化中，若 `temp` 的 32 元素块全为零，则输出 E8M0 编码 0，内部 BF16 乘数为 0。
  若用于计算二级尺度的块最大绝对值为 NaN 或 Inf，则写入 E8M0 的 NaN 编码 255。
  含非有限中间值时，常规量化公式和恢复公式不适用。
- 二级尺度的最小可表示值为 `2^-127`，对应 E8M0 编码 0。有限块在尺度下界的行为如下。

  | 量化算法 | 尺度下界及乘数 |
  |---|---|
  | scaleAlg=0，FP4 | 块最大绝对值为 BF16 正规数且共享指数不大于 −127 时，编码为 0，乘数为 `2^127`。块最大绝对值的 BF16 指数域为 0 时，编码和乘数均为 0。 |
  | scaleAlg=0，FP8 | 共享指数不大于 −127 时，编码和乘数均为 0。 |
  | scaleAlg=1，FP8 | FP32 中间尺度 `u` 为次正规数时，`u > 2^-127` 取编码 1，否则取编码 0。块最大绝对值非零时，编码 0 对应的乘数为 `2^127`。 |
  | scaleAlg=2，FP4，`d` 为 6.0 或 7.0 | 尺度达到下界时，编码和乘数均为 0。 |
  | scaleAlg=2，FP4，其他 `d` | FP32 中间尺度 `u` 为次正规数时，取编码 0。块最大绝对值非零时，编码 0 对应的乘数为 `2^127`。 |

### 空 Tensor 与 workspace

- 普通模式支持满足全部 metadata 约束的空 x/rotation；输出 shape 仍按上述公式计算。
  空 rotation 仅可能出现在合法的 `(0,K,K)` 列块布局，此时 N=0。
- 分组模式允许 M=0，但 N>0，groupList 和 rotation 仍须非空且满足分组约束。
- 空输入仍须通过必选指针、dtype、ND 格式、shape 和属性检查，成功后返回零 workspace 的空执行计划。
- 非空计算的 workspace 还可能包含输入连续化和输出回写所需的空间。调用方须按第一段接口返回的字节数分配。

### 确定性说明

- aclnnRotateDualLevelQuant 默认确定性实现。

## 调用示例

完整示例见 [test_aclnn_rotate_dual_level_quant.cpp](../examples/test_aclnn_rotate_dual_level_quant.cpp)。
算子源码位于 `experimental/matmul/rotate_dual_level_quant`，在项目根目录执行以下命令构建自定义算子包：

```bash
bash build.sh --pkg --experimental --soc=ascend950 --ops=rotate_dual_level_quant --vendor_name=custom
```

安装生成的自定义算子包后，使用包内导出的头文件和 `libcust_opapi.so` 编译示例。
通用步骤参见[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

示例使用 FLOAT16 输入，N=576、K=64，通过命令行选择输出类型和模式：

| 参数 | 取值 | 含义 |
|---|---|---|
| 第一个参数 | fp4 / e4m3 / e5m2 | 选择 FLOAT4_E2M1 / FLOAT8_E4M3FN / FLOAT8_E5M2 |
| 第二个参数 | ordinary | M=4，groupList 为 nullptr，共享 `(64,64)` 旋转矩阵 |
| 第二个参数 | count | M=4，count 为 `[2,1]`，只处理前 3 行 |
| 第二个参数 | cumsum | M=4，累计行数为 `[2,3]`，只处理前 3 行 |
| 第二个参数 | empty | M=0，演示合法空输入及空执行计划 |

调用方式：

```bash
./test_aclnn_rotate_dual_level_quant fp4 ordinary
./test_aclnn_rotate_dual_level_quant e4m3 count
./test_aclnn_rotate_dual_level_quant e5m2 cumsum
./test_aclnn_rotate_dual_level_quant fp4 empty
```

前三条分别演示普通、count 分组和 cumsum 分组的非空计算；最后一条演示空输入调用流程。
以下代码与 examples 目录中的完整示例保持一致：

<!-- example-source: ../examples/test_aclnn_rotate_dual_level_quant.cpp -->

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rotate_dual_level_quant.h"

namespace {
void Check(int status, const char* api)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(api) + " failed, status=" + std::to_string(status));
    }
}

struct Resources {
    int32_t deviceId = 0;
    bool aclReady = false;
    bool deviceReady = false;
    aclrtStream stream = nullptr;
    void* workspace = nullptr;
    std::vector<void*> buffers;
    std::vector<aclTensor*> tensors;

    ~Resources()
    {
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        for (auto* tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        for (void* buffer : buffers) {
            aclrtFree(buffer);
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceReady) {
            aclrtResetDevice(deviceId);
        }
        if (aclReady) {
            aclFinalize();
        }
    }
};

aclTensor* CreateTensor(Resources& resources, const std::vector<int64_t>& shape, aclDataType dtype, const void* initial,
                        size_t bytes, void*& address)
{
    // Give empty tensors a valid backing address without copying nonexistent elements.
    Check(aclrtMalloc(&address, std::max<size_t>(bytes, 1), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    resources.buffers.push_back(address);
    if (bytes != 0) {
        Check(aclrtMemcpy(address, bytes, initial, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * shape[i - 1];
    }
    auto* tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                   shape.size(), address);
    if (tensor == nullptr) {
        throw std::runtime_error("aclCreateTensor failed");
    }
    resources.tensors.push_back(tensor);
    return tensor;
}
} // namespace

int main(int argc, char** argv)
{
    Resources resources;
    try {
        if (argc > 3) {
            throw std::runtime_error("Usage: example [fp4|e4m3|e5m2] [ordinary|count|cumsum|empty]");
        }
        const std::string type = argc > 1 ? argv[1] : "fp4";
        const std::string mode = argc > 2 ? argv[2] : "ordinary";
        if (type != "fp4" && type != "e4m3" && type != "e5m2") {
            throw std::runtime_error("Unknown output dtype");
        }
        if (mode != "ordinary" && mode != "count" && mode != "cumsum" && mode != "empty") {
            throw std::runtime_error("Unknown grouping mode");
        }
        resources.buffers.reserve(8);
        resources.tensors.reserve(8);
        Check(aclInit(nullptr), "aclInit");
        resources.aclReady = true;
        Check(aclrtSetDevice(resources.deviceId), "aclrtSetDevice");
        resources.deviceReady = true;
        Check(aclrtCreateStream(&resources.stream), "aclrtCreateStream");
        const char* name = aclrtGetSocName();
        std::string soc = name == nullptr ? "" : name;
        std::transform(soc.begin(), soc.end(), soc.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (soc.rfind("ascend950", 0) != 0) {
            throw std::runtime_error("This example requires Ascend950 (A5)");
        }

        const bool grouped = mode == "count" || mode == "cumsum";
        const int64_t m = mode == "empty" ? 0 : 4;
        const int64_t n = 576; // Demonstrates a 512-wide block plus a 64-element tail.
        const int64_t k = 64;
        const int64_t e = grouped ? 2 : 1;
        const int64_t level0BlockSize = 512;
        const int64_t level1BlockSize = 32;
        const int64_t level0Count = (n + level0BlockSize - 1) / level0BlockSize;
        const int64_t level1Pairs = (n + level1BlockSize * 2 - 1) / (level1BlockSize * 2);
        const auto dtype = type == "fp4" ? ACL_FLOAT4_E2M1 : (type == "e4m3" ? ACL_FLOAT8_E4M3FN : ACL_FLOAT8_E5M2);
        const size_t yBytes = static_cast<size_t>(m * n) / (type == "fp4" ? 2 : 1);
        std::vector<uint16_t> xData(static_cast<size_t>(m * n), 0x3c00); // FP16 1.0.
        std::vector<uint16_t> rotData(static_cast<size_t>(e * k * k), 0);
        for (int64_t group = 0; group < e; ++group) {
            for (int64_t i = 0; i < k; ++i) {
                rotData[static_cast<size_t>(group * k * k + i * k + i)] = group == 0 ? 0x3c00 : 0xbc00;
            }
        }
        std::vector<uint8_t> yData(yBytes, 0);
        std::vector<float> level0Data(static_cast<size_t>(m * level0Count), -1.0f);
        std::vector<uint8_t> level1Data(static_cast<size_t>(m * level1Pairs * 2), 0);
        // Groups cover rows [0,3); row 3 retains initial y/level0/level1 values.
        std::vector<int64_t> groupData = mode == "cumsum" ? std::vector<int64_t>{2, 3} : std::vector<int64_t>{2, 1};
        void* xAddress = nullptr;
        void* rotAddress = nullptr;
        void* yAddress = nullptr;
        void* level0Address = nullptr;
        void* level1Address = nullptr;
        void* groupsAddress = nullptr;
        auto* x = CreateTensor(resources, {m, n}, ACL_FLOAT16, xData.data(), xData.size() * sizeof(uint16_t), xAddress);
        const std::vector<int64_t> rotShape = grouped ? std::vector<int64_t>{e, k, k} : std::vector<int64_t>{k, k};
        auto* rotation = CreateTensor(resources, rotShape, ACL_FLOAT16, rotData.data(),
                                      rotData.size() * sizeof(uint16_t), rotAddress);
        auto* y = CreateTensor(resources, {m, n}, dtype, yData.data(), yBytes, yAddress);
        auto* level0 = CreateTensor(resources, {m, level0Count}, ACL_FLOAT, level0Data.data(),
                                    level0Data.size() * sizeof(float), level0Address);
        auto* level1 = CreateTensor(resources, {m, level1Pairs, 2}, ACL_FLOAT8_E8M0, level1Data.data(),
                                    level1Data.size(), level1Address);
        aclTensor* groups = grouped ? CreateTensor(resources, {e}, ACL_INT64, groupData.data(),
                                                   groupData.size() * sizeof(int64_t), groupsAddress) :
                                      nullptr;
        char roundMode[] = "rint";
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        const auto status = aclnnRotateDualLevelQuantGetWorkspaceSize(
            x, rotation, nullptr, groups, mode == "cumsum" ? 0 : 1, -1, roundMode, 0, 0.0, false, level0BlockSize,
            level1BlockSize, y, level0, level1, &workspaceSize, &executor);
        if (status != ACL_SUCCESS) {
            std::fprintf(stderr, "aclnnRotateDualLevelQuantGetWorkspaceSize failed, status=%d.\n",
                         static_cast<int>(status));
            return 1;
        }
        if (workspaceSize != 0) {
            Check(aclrtMalloc(&resources.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "workspace allocation");
        }
        Check(aclnnRotateDualLevelQuant(resources.workspace, workspaceSize, executor, resources.stream),
              "aclnnRotateDualLevelQuant");
        Check(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream");
        if (m == 0) {
            std::printf("Empty input: execution completed with workspaceSize=%llu.\n",
                        static_cast<unsigned long long>(workspaceSize));
            return 0;
        }
        Check(aclrtMemcpy(yData.data(), yBytes, yAddress, yBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy y");
        Check(aclrtMemcpy(level0Data.data(), level0Data.size() * sizeof(float), level0Address,
                          level0Data.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
              "copy level0");
        Check(aclrtMemcpy(level1Data.data(), level1Data.size(), level1Address, level1Data.size(),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "copy level1");
        for (size_t i = 0; i < std::min<size_t>(16, yData.size()); ++i) {
            std::printf("y byte[%zu]=0x%02x\n", i, static_cast<unsigned>(yData[i]));
        }
        for (size_t i = 0; i < level0Data.size(); ++i) {
            std::printf("level0[%zu]=%g\n", i, level0Data[i]);
        }
        for (size_t i = 0; i < std::min<size_t>(18, level1Data.size()); ++i) {
            const uint8_t code = level1Data[i];
            std::printf("level1[%zu]: code=%u, value=%g\n", i, static_cast<unsigned>(code),
                        code == 255 ? std::nan("") : std::ldexp(1.0, static_cast<int>(code) - 127));
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
```
