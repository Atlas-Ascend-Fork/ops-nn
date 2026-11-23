# aclnnUnionSparseGen

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

- 接口功能：生成稀疏注意力所需的升序KV（Key/Value，键和值）块并集、并集块数和逐KV位置的查询位掩码，不执行注意力计算。

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

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnUnionSparseGenGetWorkspaceSize”获取所需workspace大小及包含计算流程的执行器，再调用“aclnnUnionSparseGen”执行计算。

```cpp
aclnnStatus aclnnUnionSparseGenGetWorkspaceSize(
  const aclTensor* sparseBlockIdx,
  const aclTensor* sparseBlockCount,
  const aclTensor* cuSqlenQOptional,
  const aclTensor* cuSqlenKvOptional,
  const aclTensor* seqUsedQOptional,
  const aclTensor* seqUsedKvOptional,
  const aclIntArray* blockShape,
  int64_t          unionSize,
  int64_t          maskType,
  int64_t          maxQSeqlen,
  int64_t          maxKvSeqlen,
  int64_t          layoutSparsePattern,
  bool             isInputIdxSorted,
  int64_t          residualBlockMode,
  int64_t          totalqUnionNum,
  int64_t          maxLogicalBlocks,
  aclTensor*       unionSparseIdx,
  aclTensor*       unionSparseCount,
  aclTensor*       exactMask,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnUnionSparseGen(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnUnionSparseGenGetWorkspaceSize

- **参数说明**

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|---|---|---|---|---|---|---|---|
| sparseBlockIdx（const aclTensor*） | 输入 | 每个查询选择的逻辑KV块索引，用于构造选中块集合$S_i$。 | shape为[H,T,K]，支持空Tensor。 | INT32 | ND | 3 | √ |
| sparseBlockCount（const aclTensor*） | 输入 | 逐行有效前缀长度，用于确定$S_i$的读取范围。 | shape为[H,T]，支持空Tensor；计算时将有效块数限制在[0,K]内。 | INT32 | ND | 2 | √ |
| cuSqlenQOptional（const aclTensor*） | 可选输入 | 查询存储前缀和，用于查询存储定位；未提供seqUsedQOptional时，其差分决定$Q_b$。 | shape为[B+1]；传nullptr时使用等宽槽位，组合要求见约束说明。 | INT64 | ND | 1 | √ |
| cuSqlenKvOptional（const aclTensor*） | 可选输入 | KV存储前缀和，未提供seqUsedKvOptional时，其差分决定$V_b$。 | shape为[B+1]；省略时按约束说明选择实际长度或定长属性。 | INT64 | ND | 1 | √ |
| seqUsedQOptional（const aclTensor*） | 可选输入 | 各请求实际查询长度，对应$Q_b$，优先于查询前缀和差分。 | shape为[B]，支持空Tensor；省略时按约束说明选择前缀差分或定长属性。 | INT32 | ND | 1 | √ |
| seqUsedKvOptional（const aclTensor*） | 可选输入 | 各请求实际KV长度，对应$V_b$，优先于KV前缀和差分。 | shape为[B]，支持空Tensor；省略时按约束说明选择前缀差分或定长属性。 | INT32 | ND | 1 | √ |
| blockShape（const aclIntArray*） | 输入 | 查询/KV块大小；第二个元素对应$Y$。 | 必须为[1,Y]，Y为4、8、16或128，不能为nullptr。 | INT64 | - | - | - |
| unionSize（int64_t） | 输入 | 每组查询数，对应$W$。 | 仅支持8。 | - | - | - | - |
| maskType（int64_t） | 输入 | 掩码模式，对应$m$，决定可见边界$E_i$。 | 支持0（非因果）和1（右下对齐因果）。 | - | - | - | - |
| maxQSeqlen（int64_t） | 输入 | 定长Q槽宽。 | 提供Q长度Tensor时忽略，否则必须非负；组合要求见约束说明。 | - | - | - | - |
| maxKvSeqlen（int64_t） | 输入 | 定长KV有效长度$V_b$。 | 提供KV长度Tensor时忽略，否则必须非负。 | - | - | - | - |
| layoutSparsePattern（int64_t） | 输入 | 稀疏模式布局，表示输入采用[H,T,K]布局。 | 仅支持4。 | - | - | - | - |
| isInputIdxSorted（bool） | 输入 | 对构造$S_i$的输入有效前缀的排序承诺。 | 支持true或false；true表示有效前缀非降序。 | - | - | - | - |
| residualBlockMode（int64_t） | 输入 | 选中块集合$S_i$的来源模式。 | 对应$R$，支持0（显式列表）或1（筛选完整可见块并补尾）。 | - | - | - | - |
| totalqUnionNum（int64_t） | 输入 | 输出组数$G$提示。 | 取-1时使用预分配输出的$G$，其他值须与$G$一致。 | - | - | - | - |
| maxLogicalBlocks（int64_t） | 输入 | 排序及成员表的逻辑块上限$L$。 | 支持-1或[1, INT32_MAX]。ACLNN按$U=\min(W(K+R),L)$校验输出容量；推导规则见约束说明。 | - | - | - | - |
| unionSparseIdx（aclTensor*） | 输出 | 升序去重并集，对应$D$。 | 预分配shape为[H,G,U]，支持空Tensor，补齐值为-1。 | INT32 | ND | 3 | √ |
| unionSparseCount（aclTensor*） | 输出 | 每个输出组的实际并集块数，对应$C$。 | 预分配shape为[H,G]，支持空Tensor。 | INT32 | ND | 2 | √ |
| exactMask（aclTensor*） | 输出 | 逐KV位置的查询位掩码，对应$M$。 | 预分配shape为[H,G,UY]，支持空Tensor；第$i$位对应组内查询$i$。 | INT32 | ND | 3 | √ |
| workspaceSize（uint64_t*） | 输出 | 返回计算需要的设备工作空间字节数，不参与公式。 | 由第一段接口返回。 | - | - | - | - |
| executor（aclOpExecutor**） | 输出 | 返回包含计算流程的执行器，不参与公式。 | 由第一段接口返回，供第二段接口使用。 | - | - | - | - |

- **返回值**

aclnnStatus：返回状态码，参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。第一段接口检查参数描述和属性，可能返回以下错误：

| 返回值 | 错误码 | 描述 |
|---|---|---|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选输入、输出、blockShape、workspaceSize或executor为空指针。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 数据类型、格式、维度、属性、长度组合、输出shape或地址范围不符合约束。 |

## aclnnUnionSparseGen

- **参数说明**

| 参数名 | 输入/输出 | 描述 |
|---|---|---|
| workspace（void*） | 输入 | 设备工作空间地址，大小按第一段接口返回的workspaceSize申请。 |
| workspaceSize（uint64_t） | 输入 | 第一段接口返回的设备工作空间字节数。 |
| executor（aclOpExecutor*） | 输入 | 第一段接口返回的执行器。 |
| stream（aclrtStream） | 输入 | 执行计算的流。 |

- **返回值**

aclnnStatus：返回状态码，参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 仅支持`blockShape=[1,Y]`, `Y∈{4,8,16,128}`、`unionSize=8`、`maskType∈{0,1}`、`layoutSparsePattern=4`、`isInputIdxSorted=true/false`、`residualBlockMode=0/1`；不开放窗口或Union消费mask模式。blockShape非空且恰有两个元素。
- cu前缀为INT64，其余输入、输出Tensor为INT32，均位于同一Device，数据格式为ND。`K>0`，维度非负，输入输出的维度及INT32字节范围均须能由INT64表示。
- 支持非连续输入，ACLNN在计算前将其转连续；支持各元素存储位置不重叠的非连续输出，按输出的shape、stride和storageOffset回写。调用方须提供合法的stride及足够的底层存储。
- sparseBlockIdx为[H,T,K]，sparseBlockCount为[H,T]。`isInputIdxSorted=true`时每行有效前缀须非降序；false允许无序，两者均允许重复。块ID相对于请求的KV起点；负ID及超过该请求有效KV逻辑域的ID忽略。`residual=0` 保留合法显式块。`residual=1` 按 query 选择块：显式块仅保留完整位于可见范围内的块（`d < floor(max(V,0)/Y)`），并在 `V>0` 且 `V%Y!=0` 时加入不完整尾块 `floor(V/Y)`，然后合并去重；尾块同时参与 unionSparseIdx、unionSparseCount 和 exactMask。`maskType=1` 时 `V=KV-Q+query局部位置+1`，`maskType=0` 时 V 为该请求实际有效 KV 长度。完整尾块不自动补入，已选择的尾块不会重复计数。
- cuSqlenQOptional非空时为[B+1]，从0开始、非降序、末项为T；请求存储起点为cuQ[b]，Q容量为相邻差分。seqUsedQOptional非空时为[B]，其值须在对应Q容量内。

  Q、KV 独立选择长度来源：seqUsed 优先，其次为 cu 相邻差分；对应 cu、seqUsed 任意一个非空时，max 的任意 int64 值均被忽略。两者均为空时才使用 max 作为每个请求的定长有效长度，要求 `max≥0`。

  | cuX | seqUsedX | 有效长度 | maxX |
  | --- | --- | --- | --- |
  | 传入 | 传入 | `seqUsedX[b]`，不得超过 cuX 差分容量 | 完全忽略 |
  | 传入 | nullptr | `cuX[b+1]−cuX[b]` | 完全忽略 |
  | nullptr | 传入 | `seqUsedX[b]` | 完全忽略 |
  | nullptr | nullptr | `Q=min(maxQ,剩余Q行)`，KV=maxKV | 定长槽宽/长度，必须非负 |

  Q 有 cu 时通过前缀定位。Q 只有 seqUsed 时，B 由其长度确定，要求 `T%B=0`，`slot=T/B`，请求 b 从 `b×slot` 开始，`seqUsedQ[b]≤slot`。Q 两个长度 Tensor 都为空时，`slot=maxQ`，`B=ceil(T/slot)`，请求 b 从 `b×slot` 开始，有效Q=`min(slot,T-b×slot)`，允许最后一槽不足 slot。

  KV 只有 seqUsed 时不受 Q slot 限制。所有已传长度 Tensor 必须具有相同 B；cu 形状为 [B+1]，seqUsed 为 [B]。

  四个 Tensor 均为空时，正 maxQ 通过 `ceil(T/maxQ)` 确定 B，最后不足一槽也计为一个 batch；`maxQ=0` 时必须 `T=0`，并按空 batch 处理。有 KV 长度 Tensor 时可确定非空 B，允许 `maxQ=0`、`T=0`。`B=0` 时 `T=0`、实际组数为0；显式提供额外输出组时全部填充为空。定长 KV 要求 `ceil(maxKV/Y)≤INT32_MAX`；不要求 `KV≥Q`。默认 `max=-1` 仅在被忽略时合法。

- ACLNN 的 G、L 和 U 按上述规则确定，支持 `G=0` 空输出；空输出也校验属性与 shape。
- 容量由公式自动确定，不能静默截断。`residual=0` 时`residual=0` 时两种mask类型下并集和count相同；尾部越界token及padding mask均为0。位掩码采用INT32位模式；本轮W8仅使用低8位，不开放W32。
- aclnnUnionSparseGen默认确定性实现。

## 确定性计算

默认支持确定性计算。

## 调用示例

示例代码如下，编译和执行过程参见[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_union_sparse_gen.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)       \
  do {                                \
    printf(message, ##__VA_ARGS__);    \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                           ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return 1);
  return 0;
}

int main(int argc, char**) {
  // 无参数时按cu前缀定位；传任意参数（例如 ./example fixed）时运行固定槽位调用。
  const bool fixedSlots = argc > 1;
  // 1. device/stream初始化。
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出：H=1，Q=2，KV=128，K=1，Y=128，W=8，G=1，Lmax=1，U=1。
  std::vector<int64_t> sparseShape = {1, 2, 1};
  std::vector<int64_t> countShape = {1, 2};
  std::vector<int64_t> lengthShape = {2};
  std::vector<int64_t> usedShape = {1};
  std::vector<int64_t> unionShape = {1, 1, 1};
  std::vector<int64_t> unionCountShape = {1, 1};
  std::vector<int64_t> maskShape = {1, 1, 128};
  std::vector<int32_t> sparseHostData = {0, 0};
  std::vector<int32_t> countHostData = {1, 1};
  std::vector<int64_t> cuQHostData = {0, 2};
  std::vector<int64_t> cuKvHostData = {0, 128};
  std::vector<int32_t> seqQHostData = {2};
  std::vector<int32_t> seqKvHostData = {128};
  std::vector<int32_t> unionHostData(1, 0);
  std::vector<int32_t> unionCountHostData(1, 0);
  std::vector<int32_t> maskHostData(128, 0);

  void* sparseDeviceAddr = nullptr;
  void* countDeviceAddr = nullptr;
  void* cuQDeviceAddr = nullptr;
  void* cuKvDeviceAddr = nullptr;
  void* seqQDeviceAddr = nullptr;
  void* seqKvDeviceAddr = nullptr;
  void* unionDeviceAddr = nullptr;
  void* unionCountDeviceAddr = nullptr;
  void* maskDeviceAddr = nullptr;
  aclTensor* sparseBlockIdx = nullptr;
  aclTensor* sparseBlockCount = nullptr;
  aclTensor* cuSqlenQ = nullptr;
  aclTensor* cuSqlenKv = nullptr;
  aclTensor* seqUsedQ = nullptr;
  aclTensor* seqUsedKv = nullptr;
  aclTensor* unionSparseIdx = nullptr;
  aclTensor* unionSparseCount = nullptr;
  aclTensor* exactMask = nullptr;

  ret = CreateAclTensor(sparseHostData, sparseShape, &sparseDeviceAddr, ACL_INT32, &sparseBlockIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(countHostData, countShape, &countDeviceAddr, ACL_INT32, &sparseBlockCount);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuQHostData, lengthShape, &cuQDeviceAddr, ACL_INT64, &cuSqlenQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuKvHostData, lengthShape, &cuKvDeviceAddr, ACL_INT64, &cuSqlenKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(seqQHostData, usedShape, &seqQDeviceAddr, ACL_INT32, &seqUsedQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(seqKvHostData, usedShape, &seqKvDeviceAddr, ACL_INT32, &seqUsedKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(unionHostData, unionShape, &unionDeviceAddr, ACL_INT32, &unionSparseIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(unionCountHostData, unionCountShape, &unionCountDeviceAddr,
                        ACL_INT32, &unionSparseCount);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, ACL_INT32, &exactMask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用两段式接口；maskType=1表示causal。
  // 固定槽位不传cuQ/cuKV，使用seqUsedQ/seqUsedKv；槽宽由T/B确定，maxQSeqlen被忽略。
  // 前缀模式和固定槽位模式的有效Q/KV相同，三个输出也相同。
  const int64_t shapeValues[] = {1, 128};
  auto* blockShape = aclCreateIntArray(shapeValues, 2);
  CHECK_RET(blockShape != nullptr, return 1);
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnUnionSparseGenGetWorkspaceSize(
      sparseBlockIdx, sparseBlockCount, fixedSlots ? nullptr : cuSqlenQ,
      fixedSlots ? nullptr : cuSqlenKv, seqUsedQ, seqUsedKv,
      blockShape, 8, int64_t{1}, fixedSlots ? 2 : -1, -1, 4, true, 0, 1, 1,
      unionSparseIdx, unionSparseCount, exactMask, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnUnionSparseGenGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnUnionSparseGen(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUnionSparseGen failed. ERROR: %d\n", ret); return ret);

  // 4. 同步执行结果并将输出拷贝到Host。
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  aclDestroyIntArray(blockShape);
  ret = aclrtMemcpy(unionHostData.data(), sizeof(int32_t), unionDeviceAddr,
                    sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemcpy(unionCountHostData.data(), sizeof(int32_t), unionCountDeviceAddr,
                    sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemcpy(maskHostData.data(), maskHostData.size() * sizeof(int32_t), maskDeviceAddr,
                    maskHostData.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  LOG_PRINT("unionSparseIdx=[%d], unionSparseCount=[%d]\n", unionHostData[0], unionCountHostData[0]);
  LOG_PRINT("exactMask first=%d, last=%d\n", maskHostData.front(), maskHostData.back());
  // 预期输出：unionSparseIdx=[0]，unionSparseCount=[1]，exactMask前127项为3，最后一项为2。

  bool correct = unionHostData[0] == 0 && unionCountHostData[0] == 1;
  for (size_t i = 0; i < maskHostData.size(); ++i) {
    correct = correct && maskHostData[i] == (i + 1 == maskHostData.size() ? 2 : 3);
  }

  // 5. 释放资源。
  aclDestroyTensor(sparseBlockIdx);
  aclDestroyTensor(sparseBlockCount);
  aclDestroyTensor(cuSqlenQ);
  aclDestroyTensor(cuSqlenKv);
  aclDestroyTensor(seqUsedQ);
  aclDestroyTensor(seqUsedKv);
  aclDestroyTensor(unionSparseIdx);
  aclDestroyTensor(unionSparseCount);
  aclDestroyTensor(exactMask);
  aclrtFree(sparseDeviceAddr);
  aclrtFree(countDeviceAddr);
  aclrtFree(cuQDeviceAddr);
  aclrtFree(cuKvDeviceAddr);
  aclrtFree(seqQDeviceAddr);
  aclrtFree(seqKvDeviceAddr);
  aclrtFree(unionDeviceAddr);
  aclrtFree(unionCountDeviceAddr);
  aclrtFree(maskDeviceAddr);
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return correct ? 0 : 1;
}
```
