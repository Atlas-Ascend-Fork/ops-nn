# UnionSparseGen

简体中文 | [English](./README_en.md)

## 产品支持情况

| 产品 | 是否支持 |
|---|:---:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：将同一请求内的多行KV（Key/Value，键和值）块列表合并为升序去重并集，同时给出并集块数和逐KV位置的查询位掩码，用于稀疏注意力元数据预处理。

- 计算公式：

设每组查询数为$W$、KV块大小为$Y$。各请求按有效查询长度连续分组，每组最多$W$个查询，不跨请求合并。

以下公式描述单个稀疏头的一组查询。组内有$n$个有效查询，请求$b$的有效查询长度和有效KV长度分别为$Q_b$、$V_b$；$q_i$是组内查询$i$在该请求内从0开始的位置。掩码类型$m$决定查询$i$可见KV位置的右边界$E_i$，该边界本身不包含在可见范围内：

| 掩码类型$m$ | 可见边界$E_i$ |
|---|---|
| 0：非因果 | $E_i=V_b$ |
| 1：右下对齐因果 | $E_i=V_b-Q_b+q_i+1$ |

每个查询按逐行块数读取输入前缀，长度限制在[0,K]内，$K$为每行块索引容量；负编号和完全超出有效KV范围的块编号不参与计算。块编号相对于该请求的KV起点。选中块集合$S_i$还受尾块模式$R$控制：

| 尾块模式$R$ | 选中块集合$S_i$ |
|---|---|
| 0：显式列表 | 保留全部合法显式块，因果掩码全为0的块也保留。 |
| 1：自动补尾 | 显式块只保留编号小于$\lfloor\max(E_i,0)/Y\rfloor$的完整可见块；当$E_i>0$且$E_i$不能被$Y$整除时，再加入尾块$\lfloor E_i/Y\rfloor$。 |

升序去重并集$D$及实际并集块数$C$为：

$$
D=\operatorname{sort}\left(\bigcup_{i=0}^{n-1}S_i\right),\qquad C=|D|.
$$

设$d_r=D[r]$，块内偏移$t$对应的KV位置为$p=Yd_r+t$，其中$0\le r<C$、$0\le t<Y$。按并集块顺序展开的查询位掩码$M$为：

$$
M[rY+t]=\sum_{i=0}^{n-1}2^i\,\mathbf{1}[d_r\in S_i]\,\mathbf{1}[p<E_i].
$$

$\mathbf{1}[\cdot]$表示条件成立取1，否则取0。因此，$M$的第$i$位表示查询$i$是否选择且可见该KV位置。非因果模式同样保留查询选择关系和有效KV边界；因果边界不大于0时，对应查询位均为0。尾块模式为0时，两种掩码类型不改变块并集和块数；模式为1时，可见边界也参与选块，因而可能改变并集和块数。

输入shape分别为[H,T,K]和[H,T]，其中$H$为稀疏头数，$T$为查询存储行数，$B$为请求数。输出shape依次为[H,G,U]、[H,G]和[H,G,UY]。公共Torch入口根据CPU长度计算输出组数$G$、最大逻辑块数$L_{\max}$和每组输出容量$U$：

$$
\begin{aligned}
G&=\sum_{b=0}^{B-1}\left\lceil\frac{Q_b}{W}\right\rceil,\\
L_{\max}&=\max\left(1,\left\lceil\frac{V_0}{Y}\right\rceil,\ldots,\left\lceil\frac{V_{B-1}}{Y}\right\rceil\right),\\
U&=\min\bigl(W(K+R),L_{\max}\bigr).
\end{aligned}
$$

无请求时$G=0$、$L_{\max}=1$。ACLNN直调使用调用方预分配的输出，并按约束说明校验$G$和$U$。GE图模式根据输入shape和属性推导$G$和$U$。每组只在前$C$个位置写入$D$，其余块索引填-1；额外输出组的count及未使用、越界KV位置的掩码填0。有效长度的来源和组合要求见约束说明。

## 参数说明

下表对应算子原型。ACLNN和PyTorch调用层的长度输入与几何参数要求见各自接口文档。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|---|---|---|---|---|
| sparse_block_idx | 输入 | 用于构造选中块集合$S_i$的块索引，shape为[H,T,K]。 | INT32 | ND |
| sparse_block_count | 输入 | 构造$S_i$时读取的有效前缀长度，shape为[H,T]。 | INT32 | ND |
| cu_sqlen_q | 可选输入 | 查询存储前缀和，shape为[B+1]；用于查询存储定位，未提供实际查询长度时还决定$Q_b$。 | INT64 | ND |
| cu_sqlen_kv | 可选输入 | KV存储前缀和，shape为[B+1]；未提供实际KV长度时决定$V_b$。 | INT64 | ND |
| seq_used_q | 可选输入 | 实际查询长度$Q_b$，shape为[B]，优先于查询前缀和差分。 | INT32 | ND |
| seq_used_kv | 可选输入 | 实际KV长度$V_b$，shape为[B]，优先于KV前缀和差分。 | INT32 | ND |
| union_sparse_idx | 输出 | 升序去重块并集$D$，shape为[H,G,U]，补齐值为-1。 | INT32 | ND |
| union_sparse_count | 输出 | 并集块数$C$，shape为[H,G]。 | INT32 | ND |
| exact_mask | 输出 | 逐KV位置的查询位掩码$M$，shape为[H,G,UY]，补齐值为0。 | INT32 | ND |
| block_shape | 可选属性 | 查询/KV块大小，默认[1,128]，支持[1,Y]，Y为4、8、16或128；第二个元素对应$Y$。 | ListInt | - |
| union_size | 可选属性 | 对应$W$，默认8且仅支持8。 | INT64 | - |
| mask_type | 可选属性 | 对应$m$，默认0；支持0（非因果）和1（右下对齐因果）。 | INT64 | - |
| totalq_union_num | 可选属性 | 输出组数$G$提示。默认-1，tiling使用输出$G$；其他值须与输出$G$一致。 | INT64 | - |
| max_logical_blocks | 可选属性 | 排序及成员表的逻辑块上限$L$。默认-1，按KV长度来源推导$L$；有效正值在GE中同时用于确定$U$。 | INT64 | - |
| max_q_seqlen | 可选属性 | 默认-1；提供Q长度Tensor时忽略，否则作为定长Q槽宽，组合要求见约束说明。 | INT64 | - |
| max_kv_seqlen | 可选属性 | 默认-1；提供KV长度Tensor时忽略，否则作为每个请求的有效KV长度$V_b$。 | INT64 | - |
| layout_sparse_pattern | 可选属性 | 稀疏模式布局，默认4且仅支持4，表示输入采用[H,T,K]布局。 | INT64 | - |
| is_input_idx_sorted | 可选属性 | 有效输入前缀的排序承诺，默认True；False允许无序输入。 | BOOL | - |
| residual_block_mode | 可选属性 | 对应$R$，默认0；支持0（显式列表）或1（筛选完整可见块并补尾）。 | INT64 | - |

## 约束说明

- 支持block_shape=[1,Y]，Y为4、8、16或128；union_size=8、mask_type为0或1、layout_sparse_pattern=4、is_input_idx_sorted为True或False、residual_block_mode为0或1。块形状恰有两个元素。
- 稀疏输入为[H,T,K]和[H,T]；H>0、K>0，维度非负。块编号相对于请求的KV起点；负编号和超出有效KV逻辑域的编号忽略。声明有序时有效前缀须非降序；无序输入同样支持重复编号。
- 输入、输出元素个数、字节数及访问地址范围均须能由INT64表示。尾块、mask与padding规则见功能说明。

- 块索引、块数、实际长度及输出为INT32，前缀为INT64，所有Tensor采用ND格式。ACLNN非空Tensor均位于同一设备；输入支持非连续Tensor，输出支持各元素存储位置不重叠的非连续Tensor。调用方提供正确的stride、storageOffset和足够的底层存储，算子按输出视图回写。

Q、KV独立选择长度来源。下表中的cu、seqUsed、max分别表示对应侧的前缀和、实际长度及max属性；提供长度Tensor时，max的任意int64值均被忽略。

| cu | seqUsed | 有效长度 | max |
|---|---|---|---|
| 提供 | 提供 | seqUsed[b]，不得超过cu相邻差分 | 忽略 |
| 提供 | 省略 | cu相邻差分 | 忽略 |
| 省略 | 提供 | seqUsed[b] | 忽略 |
| 省略 | 省略 | Q取max与剩余查询行数的较小值；KV取max | 必须非负 |

- Q有前缀和时，前缀从0开始、非降序、末项为T，请求起点为cu[b]。KV前缀同样从0开始、非降序；KV长度不受Q槽宽限制。
- Q只有seqUsed时，由其shape确定B；B>0时要求T能被B整除，槽宽为T/B，请求b从b×槽宽开始，seqUsedQ[b]不得超过槽宽。
- Q两个长度Tensor均省略且maxQ>0时，槽宽为maxQ，B=ceil(T/maxQ)，请求b的有效Q为min(maxQ,T-b×maxQ)，允许末槽不足。maxQ=0时必须T=0；若有KV长度Tensor则沿用其B，否则B=0。
- 所有已传长度Tensor须对应相同B，cu为[B+1]，seqUsed为[B]；定长Q的B还须与KV长度描述符一致。B=0时必须T=0，实际有效组数为0。
- 有效长度非负；实际长度与前缀同时提供时不得超过对应差分容量。定长KV须满足ceil(maxKV/Y)不大于INT32_MAX。不要求有效KV长度大于或等于有效Q长度。默认max=-1仅在被忽略时合法。

- 公共Torch入口使用功能说明中的G、Lmax、U公式，按实际长度分配输出；低层Torch及Fake/Meta仍须显式提供准确的G和Lmax，几何默认值-1仅用于检测遗漏，不请求自动推导。
- ACLNN直调以 `unionSparseIdx.shape[1]` 为输出组数 $G$。`totalqUnionNum=-1` 时使用该组数，其他值须等于 $G$。
- `maxLogicalBlocks` 为排序及成员表规划上限 $L$。正值须在 `[1, INT32_MAX]` 内并覆盖有效KV逻辑域。取 `-1` 时，提供KV长度Tensor则使用 `INT32_MAX`，定长KV使用 `max(1, ceil(maxKvSeqlen/Y))`。
- ACLNN按 `U=min(W*(K+R),L)` 在GetWorkspaceSize阶段校验预分配的三个输出形状 `[H,G,U]`、`[H,G]`、`[H,G,U*Y]`，包括 $G=0$ 的情况。$U$ 表示输出存储容量。
- GE图模式的InferShape只读取Host上的输入shape和属性，不读取长度Tensor内容，不引入Device到Host同步。totalq_union_num>0时直接作为G，否则G=floor((T+7B)/8)；max_logical_blocks>0时直接作为U，否则U=W(K+R)。B由长度Tensor的shape或定长Q规则确定。GE的 `max_logical_blocks` 正值同时用作 $U$ 和规划上限 $L$。传入 `-1` 时，$U$ 使用上述默认公式，$L$ 按变长或定长KV规则推导。tiling使用输出 $G$ 处理 `totalq_union_num=-1`，其他取值须与输出 $G$ 一致。
- G须覆盖全部实际有效组，U须覆盖每组实际并集。额外输出组填idx=-1、count=0、mask=0。G不足不保证报错，可能遗漏有效组；kernel检测到实际并集超过U时触发设备错误，可能在执行或流同步时暴露，不能保证在GetWorkspaceSize阶段拦截。该检查不校验G。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|---|---|---|
| aclnn API | [test_aclnn_union_sparse_gen](./examples/test_aclnn_union_sparse_gen.cpp) | 通过[aclnnUnionSparseGen](./docs/aclnnUnionSparseGen.md)接口调用算子。 |
| GE图模式 | - | 通过[算子IR](./op_graph/union_sparse_gen_proto.h)构图。 |
| PyTorch API | - | 通过[union_sparse_gen](./docs/torchapi_union_sparse_gen.md)接口调用算子。 |
