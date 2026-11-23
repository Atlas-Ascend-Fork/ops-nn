# union_sparse_gen

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
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

- 接口功能：将同一请求内每组最多8个查询选择的KV（Key/Value，键和值）块合并，返回升序去重块索引、实际并集块数和逐KV位置的查询位掩码，用于稀疏注意力元数据预处理。

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

无请求时$G=0$、$L_{\max}=1$。PyTorch公共入口根据上述公式预分配输出，并向ACLNN传入$G$和$L_{\max}$。ACLNN使用这些参数校验输出形状。每组只在前$C$个位置写入$D$，其余块索引填-1；额外输出组的count及未使用、越界KV位置的掩码填0。有效长度的来源和组合要求见约束说明。

## 函数原型

```python
cann_ops_nn.union_sparse_gen(
    sparse_block_idx,
    sparse_block_count,
    cu_sqlen_q=None,
    cu_sqlen_kv=None,
    seq_used_q=None,
    seq_used_kv=None,
    *,
    block_shape=(1, 128),
    union_size=8,
    mask_type=0,
    max_q_seqlen=-1,
    max_kv_seqlen=-1,
    layout_sparse_pattern=4,
    is_input_idx_sorted=True,
    residual_block_mode=0,
) -> Tuple[Tensor, Tensor, Tensor]
```

完整构建包的模块名为`cann_ops_nn`；选择性构建包的模块名为`cann_ops_nn_custom`，接口参数一致。`*`之后的属性须以关键字传入。四个长度Tensor可选，但需满足[约束说明](#约束说明)中的组合要求。

稀疏输入和输出位于NPU（Neural Processing Unit，神经网络处理器），长度元数据位于CPU（Central Processing Unit，中央处理器）。

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| sparse_block_idx | Tensor | 必选 | 设备上的块索引，用于构造选中块集合$S_i$，shape为[H,T,K]。 | int32 | 3 |
| sparse_block_count | Tensor | 必选 | 与块索引位于同一设备，指定构造$S_i$时读取的有效前缀长度，shape为[H,T]。 | int32 | 2 |
| cu_sqlen_q | Tensor | 可选 | CPU上的查询前缀和，shape为[B+1]，用于查询存储定位；默认None，省略时按等宽槽位定位。 | int32、int64 | 1 |
| cu_sqlen_kv | Tensor | 可选 | CPU上的KV前缀和，shape为[B+1]；默认None，省略时按约束说明选择实际长度或定长属性。 | int32、int64 | 1 |
| seq_used_q | Tensor | 可选 | CPU上的实际查询长度$Q_b$，shape为[B]；默认None，存在时优先于查询前缀和差分。 | int32、int64 | 1 |
| seq_used_kv | Tensor | 可选 | CPU上的实际KV长度$V_b$，shape为[B]；默认None，存在时优先于KV前缀和差分。 | int32、int64 | 1 |
| block_shape | Tuple[int,int] | 可选 | 查询/KV块大小，默认(1,128)，支持(1,Y)，Y为4、8、16或128；第二个元素对应$Y$。 | - | - |
| union_size | int | 可选 | 对应$W$，默认8且仅支持8。 | - | - |
| mask_type | int | 可选 | 对应$m$，默认0；支持0（非因果）和1（右下对齐因果）。 | - | - |
| max_q_seqlen | int | 可选 | 默认-1；提供Q长度Tensor时忽略，否则作为定长Q槽宽，组合要求见约束说明。 | - | - |
| max_kv_seqlen | int | 可选 | 默认-1；提供KV长度Tensor时忽略，否则作为每个请求的有效KV长度$V_b$。 | - | - |
| layout_sparse_pattern | int | 可选 | 默认4且仅支持4，表示输入稀疏模式采用[H,T,K]布局。 | - | - |
| is_input_idx_sorted | bool | 可选 | 有效前缀排序承诺，默认True；False允许无序输入。 | - | - |
| residual_block_mode | int | 可选 | 对应$R$，默认0；支持0（显式列表）或1（筛选完整可见块并补尾）。 | - | - |

## 返回值说明

按下表顺序返回三个设备上的Tensor：

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| union_sparse_idx | Tensor | 必选 | 对应$D$，shape为[H,G,U]；升序去重块索引，未使用位置填-1。 | int32 | 3 |
| union_sparse_count | Tensor | 必选 | 对应$C$，shape为[H,G]；保存各组实际并集块数。 | int32 | 2 |
| exact_mask | Tensor | 必选 | 对应$M$，shape为[H,G,UY]；第$i$位对应组内查询$i$，未使用和越界KV位置为0。 | int32 | 3 |

## 约束说明

- 仅支持`block_shape=[1,Y]`（`Y∈{4,8,16,128}`）、`union_size=8`、`mask_type=0/1`、`layout_sparse_pattern=4`、`sorted=true/false`、`residual=0/1`。
- `is_input_idx_sorted=true`时输入有效前缀须非降序，false允许无序，两者均允许重复，长度由sparse_block_count指定；非法块ID忽略。`residual=0` 保留合法显式块。`residual=1` 沿用旧版逐 query 选择规则：显式块仅保留完整位于可见范围内的块（`d < floor(max(V,0)/Y)`），并在 `V>0` 且 `V%Y!=0` 时加入不完整尾块 `floor(V/Y)`，然后合并去重；尾块同时参与 unionSparseIdx、unionSparseCount 和 exactMask。`maskType=1` 时 `V=KV-Q+query局部位置+1`，`maskType=0` 时 V 为该请求实际有效 KV 长度。完整尾块不自动补入，已选择的尾块不会重复计数。

- Q、KV 独立选择长度来源：seqUsed 优先，其次为 cu 相邻差分；对应 cu、seqUsed 任意一个非空时，max 的任意 int64 值均被忽略。两者均为空时才使用 max 作为每个请求的定长有效长度，要求 `max≥0`。

  | cuX | seqUsedX | 有效长度 | maxX |
  | --- | --- | --- | --- |
  | 传入 | 传入 | `seqUsedX[b]`，不得超过 cuX 差分容量 | 完全忽略 |
  | 传入 | nullptr | `cuX[b+1]−cuX[b]` | 完全忽略 |
  | nullptr | 传入 | `seqUsedX[b]` | 完全忽略 |
  | nullptr | nullptr | KV 为 maxKV；Q 为 `min(maxQ,T-b×maxQ)` | 定长槽宽/长度，必须非负 |

  Q 有 cu 时通过前缀定位。Q 只有 seqUsed 时，B 由其长度确定，要求 `T%B=0`，`slot=T/B`，请求 b 从 `b×slot` 开始，`seqUsedQ[b]≤slot`。Q 两个长度 Tensor 都为空时，`slot=maxQ`，`B=ceil(T/maxQ)`，有效Q=`min(maxQ,T-b×maxQ)`，允许最后一槽不足 maxQ。例如 T=10、maxQ=6 时 Q 长度为 [6,4]。

  KV 只有 seqUsed 时不受 Q slot 限制。所有已传长度 Tensor 必须具有相同 B；cu 形状为 [B+1]，seqUsed 为 [B]，定长 Q 时 KV 描述符的 B 必须等于 `ceil(T/maxQ)`。

  四个 Tensor 均为空时，正 maxQ 通过 `ceil(T/maxQ)` 确定 B；`maxQ=0` 时必须 `T=0`，并按空 batch 处理。有 KV 长度 Tensor 时可确定非空 B，允许 `maxQ=0`、`T=0`。`B=0` 时 `T=G=0`。定长 KV 要求 `ceil(maxKV/Y)≤INT32_MAX`；不要求 `KV≥Q`。默认 `max=-1` 仅在被忽略时合法。

- CPU长度校验后，cu转为NPU INT64，seqUsed转为NPU INT32。不新增Q/KV布局参数。
- 公共入口根据CPU长度计算G/Lmax/U并分配输出，调用成本包含校验和长度下发；首次调用可能JIT编译C++桥接，需可用的C++编译器与ninja；缺少ninja时，在当前Python环境执行`python -m pip install ninja`。

## 确定性计算

默认支持确定性计算。

## 调用示例

```python
import torch
import torch_npu

try:
    from cann_ops_nn import union_sparse_gen
except ImportError:
    from cann_ops_nn_custom import union_sparse_gen

# 稀疏输入在NPU，长度元数据在CPU。
idx = torch.tensor([[[0], [0]]], dtype=torch.int32, device="npu:0")
count = torch.ones((1, 2), dtype=torch.int32, device="npu:0")
cu_q = torch.tensor([0, 2], dtype=torch.int64)
cu_kv = torch.tensor([0, 128], dtype=torch.int64)

# 右下对齐因果模式：两个查询分别可见127、128个KV位置。
union_idx, union_count, mask = union_sparse_gen(
    idx, count, cu_q, cu_kv, mask_type=1,
)
assert union_idx.cpu().tolist() == [[[0]]]
assert union_count.cpu().tolist() == [[1]]
assert torch.all(mask[0, 0, :127] == 3)
assert mask[0, 0, 127].item() == 2

# 不传前缀和时使用等宽查询槽位，实际长度仍由seq_used指定。
used_q = torch.tensor([2], dtype=torch.int32)
used_kv = torch.tensor([128], dtype=torch.int32)
_, _, noncausal_mask = union_sparse_gen(
    idx, count, seq_used_q=used_q, seq_used_kv=used_kv,
    max_q_seqlen=2, mask_type=0,
)
assert torch.all(noncausal_mask == 3)
```
