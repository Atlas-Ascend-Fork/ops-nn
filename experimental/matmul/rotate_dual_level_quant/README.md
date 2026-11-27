# RotateDualLevelQuant

## 产品支持情况

| 产品 | 是否支持 |
|---|:---:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：对输入 x 进行旋转变换和可选的 clamp 操作，再沿尾轴执行两级动态量化，
  输出低精度结果 y、FP32 一级尺度 level0_scale 和 E8M0 二级尺度 level1_scale。
- 旋转及分组相关输入和属性沿用 [RotateQuantV2](../../../matmul/rotate_quant/docs/aclnnRotateQuantV2.md) 的 A5 接口，
  支持普通旋转和按 M 轴分组选择旋转矩阵。y 支持 FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2。
- 两级量化的分块顺序参照 [DynamicDualLevelMxQuant](../../../quant/dynamic_dual_level_mx_quant/README.md)。
  第一级沿原始尾轴每 512 个元素计算一个 FP32 尺度，归一化后再按每 32 个元素执行 MX 量化。
  通过 level0_block_size、level1_block_size 指定两级量化块大小，默认值及当前唯一支持值分别为 512、32。

### 计算公式

记 x 的尾维为 N，前导维合并为 M，一维输入时 M=1；旋转矩阵阶数为 K，B=N/K。
两级量化块大小由 level0_block_size、level1_block_size 指定，当前分别只支持 512、32。
以下公式描述有限数据、块最大绝对值非零且尺度处于可表示范围内的计算。

#### 1. 旋转变换

将每行沿 N 划分为 B 个 K 元素块，执行右乘旋转：

$$
Z_{m,bK:(b+1)K}
=\operatorname{cast}_{\mathrm{BF16}}\left(X_{m,bK:(b+1)K}R_{e(m),b}\right)
$$

普通模式使用共享矩阵或每列块独立的矩阵；分组模式下，e(m) 为第 m 行所属的组，
使用对应组的共享矩阵或列块矩阵。旋转结果以 BF16 域进入后续处理。

#### 2. 可选 clamp

记每行第 h 个 32 元素块为 `H[h]=[32*h,min(32*(h+1),N))`，计算：

$$
a^Z_{m,h}=\max_{j\in H[h]}|Z_{m,j}|,\qquad L_{m,h}=\alpha\,a^Z_{m,h}
$$

alpha 非空且数值在 `(0.0,1.0)` 内时：

$$
z_{m,j}=\operatorname{clamp}\left(Z_{m,j},-L_{m,h(j)},L_{m,h(j)}\right),\qquad h(j)=\lfloor j/32\rfloor
$$

否则 `z=Z`。一级量化使用 clamp 处理后的 BF16 数据 z。

#### 3. 一级量化

记每行第 g 个一级块为 `G[g]=[512*g,min(512*(g+1),N))`，计算：

$$
A_{m,g}=\max_{j\in G[g]}|z_{m,j}|
$$

$$
\widehat{s}_{0,m,g}=\begin{cases}
\mathrm{RN}_{32}(A_{m,g}\cdot\mathrm{RN}_{32}(1/6)), & \mathrm{FP4\ E2M1}\\
\mathrm{RN}_{32}(A_{m,g}/C_{\mathrm{MAX}}), & \mathrm{FP8}
\end{cases},\qquad
s_{0,m,g}=\begin{cases}
0, & \widehat{s}_{0,m,g}<2^{-126}\\
\widehat{s}_{0,m,g}, & \text{其他}
\end{cases}
$$

$$
t_{m,j}=\operatorname{cast}_{\mathrm{BF16}}
\left(\frac{z_{m,j}}{s_{0,m,g(j)}}\right),\qquad g(j)=\lfloor j/512\rfloor
$$

其中 RN₃₂ 表示 FP32 最近偶数舍入。一级最大值仅统计有限的 BF16 绝对值，无有限值时取 0。
FP4 一级尺度沿用 DynamicDualLevelMxQuant 的 FP32 倒数乘法。两种 FP8 的一级尺度直接按 FP32 除法计算。
次正规一级尺度写为正零，归一化后恢复原输入零的符号。

s0 写入 level0_scale，t 作为第二级量化的输入。CMAX 由 y 的数据类型决定：

| y 数据类型 | 一级分母 CMAX | scale_alg=0 使用的 emax |
|---|---:|---:|
| FLOAT4_E2M1 | 6 | 2 |
| FLOAT8_E4M3FN | 448 | 8 |
| FLOAT8_E5M2 | 57344 | 15 |

一级块可包含多个 K 宽旋转块；不足 512 的尾块仅统计有效元素。
两级分块始终沿原始 N 轴、每行独立进行，不跨原始行或 M 轴分组边界归约。

#### 4. 二级 MX 量化

对 t 每 32 个元素计算一个 E8M0 尺度。记 `a[m,h]=max(abs(t[m,H[h]]))`。

- scale_alg=0，支持三种输出类型：

  $$
  e_{m,h}=\left\lfloor\log_2(a_{m,h})\right\rfloor-emax,\qquad s_{1,m,h}=2^{e_{m,h}}
  $$

  对于 FP4 E2M1 的有限正规块最大值 a，最小有效尺度为 2⁻¹²⁷。
  当 floor(log₂(a)) − 2 ≤ −127 时，有效 E8M0 码为 0，内部乘数为 2¹²⁷。
  若该块 BF16 最大绝对值的原始指数域为零（含全零块或仅有零和次正规数的块），则尺度码和内部乘数均为零。
  尾部对齐的填充字节由有效组数识别。

- scale_alg=1，仅支持两种 FP8。先按目标类型最大有限值计算浮点尺度，
  在常规正规数范围内向上选择二次幂尺度：

  $$
  u_{m,h}=\mathrm{RN}_{32}(a_{m,h}\cdot\mathrm{RN}_{32}(1/C_{\mathrm{MAX}})),\qquad
  s_{1,m,h}=2^{\lceil\log_2(u_{m,h})\rceil}
  $$

- scale_alg=2，仅支持 FLOAT4_E2M1，dst_type_max 为 `[6.0,12.0]` 内的有限值。
  将合法的 dst_type_max 转成 FP32 分母 d。d 为 6.0 或 7.0 时，沿用 RotateQuantV2 的 BF16 尾数判定规则；
  d 取其他值时，先直接计算正确舍入的 FP32 商 a/d，再按该商选择 E8M0 尺度。
  常规正规数公式、尾数阈值和适用范围见[接口文档](docs/aclnnRotateDualLevelQuant.md)。

选定 s1 后，执行最终类型转换：

$$
y_{m,j}=\operatorname{cast}_{\mathrm{y.dtype}}
\left(\frac{t_{m,j}}{s_{1,m,h(j)}},\operatorname{round\_mode}\right)
$$

s1 写入 level1_scale。scale_alg、dst_type_max 仅影响第二级尺度，不改变一级分母。
scale_alg=0 仍可能发生饱和。两级尺度的概念恢复关系为：

$$
\hat z_{m,j}=s_{0,m,g(j)}\,s_{1,m,h(j)}\,y_{m,j}
$$

该关系包含 BF16 中间转换及最终量化的误差，恢复对象为旋转及可选 clamp 后的数据 z。

## 参数说明

下表采用[算子 IR](op_graph/rotate_dual_level_quant_proto.h)中的参数名称。
ACLNN 中 `rot` 对应 `rotation`，`group_list` 对应 `groupList`，三路输出依次对应
`yOut`、`level0ScaleOut`、`level1ScaleOut`。IR 属性采用 snake_case，ACLNN 属性使用对应的 camelCase 名称。
`y_dtype` 为 IR 属性；ACLNN 从调用方预分配的 yOut 读取目标类型，无须额外传入。

<table>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待旋转量化的张量。普通模式为 (*,N)，分组模式为 (M,N)。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rot</td>
      <td>输入</td>
      <td>旋转矩阵，数据类型与 x 一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>可选输入</td>
      <td>clamp 范围的缩放系数，shape 为 (1,)；可传 nullptr。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_list</td>
      <td>可选输入</td>
      <td>M 轴连续行的分组列表，shape 为 (E,)；nullptr 表示普通模式。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>量化结果，shape 与 x 相同。</td>
      <td>FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>level0_scale</td>
      <td>输出</td>
      <td>第一级量化尺度，尾维为 ceil(N/512)，其他维度与 x 一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>level1_scale</td>
      <td>输出</td>
      <td>第二级 MX 尺度，尾两维为 (ceil(N/64),2)，其他维度与 x 的前导维一致。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_dtype</td>
      <td>可选属性</td>
      <td>指定 y 的数据类型，默认 DT_FLOAT4_E2M1；ACLNN 由 yOut 的 dtype 指定。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td>量化发生的轴，默认 -1，仅支持 -1 或 x 最后一维的索引。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>可选属性</td>
      <td>最终 y 的类型转换模式，默认 rint；FP4 支持 rint、round、floor，FP8 仅支持 rint。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>可选属性</td>
      <td>第二级尺度算法，默认 0；FP4 支持 0、2，FP8 支持 0、1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>可选属性</td>
      <td>第二级尺度范围参数，默认 0.0；scale_alg=2 时须在 [6.0,12.0] 内，其他场景仅支持 0.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>trans</td>
      <td>可选属性</td>
      <td>是否转置输出 y，默认 false，当前仅支持 false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_list_type</td>
      <td>可选属性</td>
      <td>分组列表的表示方式，默认 0；0=cumsum，1=count，group_list 为空时也必须为 0 或 1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>level0_block_size</td>
      <td>可选属性</td>
      <td>第一级量化块大小，默认 512，当前仅支持 512；ACLNN 参数名为 level0BlockSize。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>level1_block_size</td>
      <td>可选属性</td>
      <td>第二级 MX 量化块大小，默认 32，当前仅支持 32；ACLNN 参数名为 level1BlockSize。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

上述默认值为 IR 属性默认值；ACLNN 需按函数原型显式传入标量属性，roundMode 可传 nullptr 以采用 rint。
类型、shape 和返回值的完整说明见 [aclnnRotateDualLevelQuant](docs/aclnnRotateDualLevelQuant.md)。

## 约束说明

### 输入及属性

- 所有 Tensor 均须为 ND 格式，支持非连续 Tensor；x 和 rot 必须同时为 FLOAT16 或同时为 BFLOAT16。
- 普通模式下 x 为 `(*,N)`，维数为 1～7；rot 为 `(K,K)` 或 `(N/K,K,K)`。
- 分组模式下 x 为 `(M,N)`，仅支持二维，M≥0、N>0；rot 为 `(E,K,K)` 或 `(E,N/K,K,K)`，不能为空 Tensor。
- K 只能取 32、64、128、256，rot 最后两维必须相等，N 必须为 K 的整数倍，不要求 N 为 512 的整数倍。
- ACLNN 调用须使用具体的非负 shape；x 各维及合轴后的 M 不得超过 INT32_MAX。
- alpha 非空时须为 BFLOAT16 的 `(1,)` Tensor；nullptr 或数值不在 `(0.0,1.0)` 内时不执行 clamp。
- axis 仅支持 -1 或 D-1，其中 D 为 x 的维数；分组模式仅支持 -1 或 1。trans 仅支持 false。
- level0_block_size 仅支持 512，level1_block_size 仅支持 32；合法空输入也须满足此约束。
- round_mode、scale_alg 须满足参数表中的 dtype 组合约束。
  scale_alg=2 时 dst_type_max 须为 `[6.0,12.0]` 内的有限值，其他场景仅支持 0.0。

### M 轴分组

- group_list 非空时，须为 INT64 的 `(E,)` Tensor，`1≤E≤1024`，不能为空 Tensor；rot 的首维须等于 E。
- group_list_type=0：列表为非负、单调不减的累计行数，最后一个值不超过 M。
- group_list_type=1：列表为非负的各组行数，总和不超过 M。
- 分组从第 0 行开始连续划分，支持空组，空组仍占用 rot 的组下标。
  允许仅覆盖 M 的连续前缀，未覆盖行的三路输出及尺度 padding 均保持调用前内容。
- 全部组为空时不更新输出；M=0 时，各组行数应全部为 0。
- group_list 的设备数据值由调用方保证合法，Host 仅检查 metadata；列表值不合法时不保证输出符合预期。
- M 轴分组仅改变对应行使用的旋转矩阵，沿 N 轴的 512/32 两级量化粒度不变。

### 输出与空 Tensor

- 输出顺序固定为 y、level0_scale、level1_scale，三路输出均须预先分配。
- 对于 `x.shape=(*,N)`，`y.shape=(*,N)`，`level0_scale.shape=(*,ceil(N/level0_block_size))`，
  `level1_scale.shape=(*,ceil(N/(2*level1_block_size)),2)`；前导维保持不变。
  当前两级块大小仅支持 512、32，因此尺度尾维分别为 `ceil(N/512)`、`(ceil(N/64),2)`。
- level0_scale 与 x 的维数相同，为 1～7 维；level1_scale 比 x 多一维，为 2～8 维。
- 每行有效二级尺度数为 `ceil(N/32)`。有效尺度数为奇数时，最后一个字节为 padding，填 E8M0 编码 0。
  编码 0 表示数值 `2^-127`，不表示浮点数值零，padding 不参与有效数据恢复。
  未覆盖行的 padding 同样保留原内容。
- 例如 x 为 `(4,576)`、K=64 时，两级尺度 shape 分别为 `(4,2)`、`(4,9,2)`。
  x 为 `(4,544)`、K=32 时，尺度 shape 相同，但每行第二级尺度的最后 1 字节为 padding。
- 普通模式支持满足全部 metadata 约束的空输入；空 rot 仅可能为 `(0,K,K)`，此时 N=0。
  分组模式支持 M=0，但 N>0，group_list 和 rot 仍须非空。合法空输入的输出保持调用前内容。

## 调用说明

<table>
  <thead>
    <tr>
      <th>调用方式</th>
      <th>样例代码</th>
      <th>说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>aclnn 接口</td>
      <td><a href="examples/test_aclnn_rotate_dual_level_quant.cpp">test_aclnn_rotate_dual_level_quant</a></td>
      <td>通过 <a href="docs/aclnnRotateDualLevelQuant.md">aclnnRotateDualLevelQuant</a> 两段式接口调用。</td>
    </tr>
    <tr>
      <td>Torch 接口</td>
      <td><a href="docs/torchapi_rotate_dual_level_quant.md">rotate_dual_level_quant</a></td>
      <td>通过 torch_extension 提供 eager 执行和 Meta 推导，返回 y、level0_scale、level1_scale。</td>
    </tr>
    <tr>
      <td>图模式</td>
      <td>-</td>
      <td>通过 <a href="op_graph/rotate_dual_level_quant_proto.h">算子 IR</a> 构图，支持普通及分组计算。</td>
    </tr>
  </tbody>
</table>

ACLNN 先调用 `aclnnRotateDualLevelQuantGetWorkspaceSize` 获取 workspace 大小与执行器，
再调用 `aclnnRotateDualLevelQuant`。

算子位于 `experimental/matmul/rotate_dual_level_quant`，在项目根目录执行以下命令构建自定义算子包：

若需联编本地两个仓库的修改，请在同一终端指定配套的 ops-tensor 源码目录，并检出其锁定的 Tensor API 子模块。例如，两个仓库平行放置时可以执行

```bash
export OPS_TENSOR_SOURCE_DIR="$(cd ../ops-tensor_dual_level_rotate_quant && pwd)"
git -C "$OPS_TENSOR_SOURCE_DIR" submodule update --init -- include/tensor_api
```

```bash
bash build.sh --pkg --experimental --soc=ascend950 --ops=rotate_dual_level_quant --vendor_name=custom
```

安装生成的自定义算子包后，编译和运行示例时使用包内导出的头文件和 `libcust_opapi.so`。
通用步骤参见[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

示例支持输出类型 `fp4/e4m3/e5m2`，以及 `ordinary/count/cumsum/empty` 模式：

```bash
./test_aclnn_rotate_dual_level_quant fp4 ordinary
./test_aclnn_rotate_dual_level_quant e4m3 count
./test_aclnn_rotate_dual_level_quant e5m2 cumsum
./test_aclnn_rotate_dual_level_quant fp4 empty
```

前三条分别演示普通、count 分组和 cumsum 分组的非空计算；最后一条演示合法空输入及空执行计划。
非空计算的中间结果保存在 UB，用户 GM scratch 为零；Host 预留 16MiB 系统 workspace。
ACLNN 返回的实际 workspace 还可能包含连续化及输出回写所需空间，调用方须按接口返回值分配。
