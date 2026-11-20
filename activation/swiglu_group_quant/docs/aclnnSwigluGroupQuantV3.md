# aclnnSwigluGroupQuantV3

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

### 接口功能

`aclnnSwigluGroupQuantV3`实现SwiGLU激活、可选weight加权和量化融合计算，在
[aclnnSwigluGroupQuantV2](aclnnSwigluGroupQuantV2.md)的基础上新增`groupListType`参数，
用于指定`groupIndexOptional`的列表格式：

- **groupListType=1**：一维INT64 count列表`[G]`，每个元素表示对应组的token数。
- **groupListType=2**：INT64二元组列表`[E, 2]`，每行为`[group_id, count]`，仅第二列count参与计算；支持非连续输入，由ACLNN自动转为连续布局。
- **groupListType=其他值不支持**。类型2的group_id不参与求和、索引或token重排。

V3保留各量化模式的计算和参数约束。`quantMode=0/1`支持有效token前缀截断，
`quantMode=2/3`支持按组HiFloat8量化；`quantMode=5`仍不支持`groupIndexOptional`和`scaleOptional`。
在支持HiFloat4的平台和CANN版本上，`quantMode=4`同样支持count/pair格式的有效token前缀截断。

> **与V1/V2的关系**：V3在`bias`之后、输出参数之前新增`int64_t groupListType`。
> V1/V2保持原函数签名，并自动使用`groupListType=1`；V3调用时需显式传入该参数。

### 计算公式

将输入记为`[..., D]`，令$H=D/2$，$T$为除最后一维外所有维度的乘积。
以下公式将输入按token展平为$[T,D]$描述，输出保留原有前导维度。

**基础计算流程**

```txt
步骤一：GroupIndex处理（可选）→ 确定有效token数或各组token区间
步骤二：输入切分、Clamp处理（可选）与SwiGLU激活
步骤三：Weight加权（可选）
步骤四：按quantMode执行量化
```

<details>
<summary><strong>步骤一：GroupIndex处理（可选）</strong></summary>

定义组数$L$及每组token数$c_g$：

$$
c_g =
\begin{cases}
\text{groupIndexOptional}[g], & \text{groupListType}=1,\quad L=G \\
\text{groupIndexOptional}[g,1], & \text{groupListType}=2,\quad L=E
\end{cases}
$$

count必须非负。类型2中的第一列group_id仅为标识，不作为输入或scale的访问下标。

**quantMode=0/1/4：有效token前缀截断**

$$
T_{\text{valid}} = \min\!\left(\sum_{g=0}^{L-1} c_g,\ T\right)
$$

仅处理前$T_{\text{valid}}$个token，只保证该前缀的输出有效，输出shape保持不变。
未提供`groupIndexOptional`时，$T_{\text{valid}}=T$。

例如，类型1的`[2, 3, 0]`与类型2的`[[7, 2], [1, 3], [9, 0]]`都表示前5个token有效
（假设$T\geq5$）。Block/MX/HiFloat4的类型2归约要求count及其总和处于int32非负范围。

**quantMode=2/3：按组HiFloat8量化**

各组按列表顺序占据连续token区间：

$$
b_0=0,\qquad b_{g+1}=b_g+c_g,\qquad I_g=[b_g,b_{g+1})
$$

- 静态量化（mode 2）：第$g$组使用输入`scaleOptional[g]`作为invScale。
- 动态量化（mode 3）：独立统计第$g$组的量化scale，并写入`yScaleOut[g]`。
- 类型2的scale按pair的行顺序对应，长度为$E$；不使用group_id作为下标。
- count总和不得超过$T$。未覆盖的尾部token的量化输出填0；启用`outputOrigin`时，尾部仍输出weight加权前的SwiGLU结果。
- 未提供`groupIndexOptional`时，将所有token作为一组处理。

**quantMode=5**

仅允许`groupListType=1`且`groupIndexOptional=nullptr`，全部$T$行参与计算。
该模式未开放共享内核的cumsum分组分支。

</details>

<details>
<summary><strong>步骤二：输入切分、Clamp与SwiGLU激活</strong></summary>

输入沿最后一维切分为两部分：

$$
A[t,h]=x[t,h],\qquad B[t,h]=x[t,h+H],\qquad 0\leq h<H
$$

当`clampLimit > 0`时，令$c=\text{clampLimit}$：

$$
A'=\min(A,c),\qquad B'=\min(\max(B,-c),c)
$$

`clampLimit=-1.0`时不启用clamp，即$A'=A$、$B'=B$。
注意门控分支$A$只限制上界，不会将负数截断为0。

`quantMode=0/1/2/3/4`使用标准SwiGLU：

$$
Y_{\text{act}}=A'\cdot\sigma(A')\cdot B',\qquad
\sigma(z)=\frac{1}{1+e^{-z}}
$$

`quantMode=5`支持带`alpha`和`bias`的变体：

$$
Y_{\text{act}}=A'\cdot\sigma(\alpha A')\cdot(B'+\beta)
$$

其中$\alpha=\text{alpha}$、$\beta=\text{bias}$。
当`alpha=1.0`、`bias=0.0`时，激活的数学公式退化为标准SwiGLU。
mode 5的激活结果会舍入到输入的FLOAT16/BFLOAT16类型，mode 1的FP8路径则保留FP32结果参与量化，
因此两者不保证逐位一致。

</details>

<details>
<summary><strong>步骤三：Weight加权（可选）</strong></summary>

提供`weightOptional`时，对每个token的激活结果加权：

$$
Y_{\text{weighted}}[t,h]=Y_{\text{act}}[t,h]\cdot w[t]
$$

未提供weight时，$Y_{\text{weighted}}=Y_{\text{act}}$。
mode 5支持FLOAT16、BFLOAT16或FLOAT32的weight，乘法按FP32计算后再次舍入到输入类型。
其他模式的weight为FLOAT32。

`outputOrigin=true`时，`yOriginOut`输出weight加权前的SwiGLU结果，数据类型与`x`一致。

</details>

<details>
<summary><strong>步骤四：量化计算</strong></summary>

- **quantMode=0**：Block FP8量化，特征维按128个元素计算一个FLOAT32 scale。
- **quantMode=1**：MX FP8/FP4量化，特征维按32个元素计算一个FLOAT8_E8M0 scale。
- **quantMode=2**：HiFloat8静态量化，使用`scaleOptional`中的invScale。
- **quantMode=3**：HiFloat8动态量化，按组或按整个输入计算FLOAT32 scale并写入`yScaleOut`。
- **quantMode=4**：HiFloat4量化，特征维按64个元素分组；仅在支持的平台及CANN版本上可用。
- **quantMode=5**：CuBLAS兼容MX FP8量化，特征维按32个元素计算一个FLOAT8_E8M0 scale，无amax下限。

mode 1的FP8路径在计算scale前将amax限制为不小于$10^{-4}$；mode 5不设置该下限，
全零块的scale编码和量化乘数均为0。两者均使用E8M0 scale，但中间舍入和边界值处理不同。

原有量化模式的计算详见[aclnnSwigluGroupQuant](aclnnSwigluGroupQuant.md)，
mode 5的MX量化计算详见[aclnnSwigluGroupQuantV2](aclnnSwigluGroupQuantV2.md#计算公式)。
`groupListType`只改变groupIndex的读取格式，不改变对应模式的量化算法。

</details>

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSwigluGroupQuantV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwigluGroupQuantV3”接口执行计算。

```Cpp
aclnnStatus aclnnSwigluGroupQuantV3GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightOptional,
  const aclTensor *groupIndexOptional,
  const aclTensor *scaleOptional,
  int64_t          dstType,
  int64_t          quantMode,
  int64_t          blockSize,
  bool             roundScale,
  double           clampLimit,
  double           dstTypeMax,
  bool             outputOrigin,
  double           alpha,
  double           bias,
  int64_t          groupListType,
  const aclTensor *yOut,
  const aclTensor *yScaleOut,
  const aclTensor *yOriginOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwigluGroupQuantV3(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnSwigluGroupQuantV3GetWorkspaceSize

- **参数说明：**

  x、weightOptional和groupIndexOptional支持非连续Tensor。ACLNN接口会在执行算子前自动将其转为连续布局，内核按转换后的连续数据访问；连续化所需的workspace包含在GetWorkspaceSize返回值中。

  <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
  <col style="width: 260px">
  <col style="width: 110px">
  <col style="width: 220px">
  <col style="width: 420px">
  <col style="width: 180px">
  <col style="width: 90px">
  <col style="width: 100px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>SwiGLU输入。</td>
      <td><ul><li>shape为[..., D]，T为前导维度的乘积。</li><li>quantMode=5时必须为二维[T, D]。</li><li>非空输入：mode 1/5要求D &gt;= 64且D % 64 == 0；mode 0/4要求D &gt;= 256且D % 256 == 0；mode 2/3要求D为正偶数。</li><li>mode 1/4为2-7维，其余为2-8维（mode 5仅支持二维）。</li><li>mode 0/1/4/5支持FLOAT16、BFLOAT16；mode 2/3还支持FLOAT32。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2-8（见使用说明）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>每个token的weight，用于SwiGLU结果加权。</td>
      <td><ul><li>可传nullptr；提供时元素个数必须为T，维度为1-8。</li><li>mode 5支持FLOAT16、BFLOAT16、FLOAT32，其他模式仅支持FLOAT32。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>各组token数或[group_id, count]列表。</td>
      <td><ul><li>groupListType=1：一维INT64 count列表[G]。</li><li>groupListType=2：必填，INT64二元组表[E, 2]，仅第二列count参与计算；非连续输入由ACLNN自动转连续。</li><li>提供时必须为非空Tensor，count必须非负；group_id不参与索引或重排。</li><li>mode 0/1/4按count总和确定有效token前缀；mode 2/3按count划分组区间。</li><li>mode 5不支持，必须传nullptr。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1或2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>静态量化输入invScale。</td>
      <td><ul><li>仅mode 2使用，且该模式必须提供。</li><li>无groupIndex时shape为[1]；groupListType=1时为[G]；groupListType=2时为[E]，按pair行顺序对应。</li><li>mode 3的scale由数据动态计算，不读取该输入。mode 5必须传nullptr。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dstType（int64_t）</td>
      <td>输入</td>
      <td>目标量化数据类型。</td>
      <td><ul><li>mode 0/5支持FLOAT8_E5M2、FLOAT8_E4M3FN。</li><li>mode 1还支持FLOAT4_E2M1、FLOAT4_E1M2。</li><li>mode 2/3固定输出HIFLOAT8；mode 4固定输出HIFLOAT4，dstType不生效。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td><ul><li>0：Block FP8；1：MX FP8/FP4；2：HiFloat8静态量化；3：HiFloat8动态量化；5：MX Quant V2。</li><li>4：HiFloat4，仅在支持的平台和CANN版本上可用。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize（int64_t）</td>
      <td>输入</td>
      <td>特征维量化块大小。</td>
      <td><ul><li>0表示使用该模式的默认块大小。</li><li>mode 0支持0或128；mode 1/5支持0或32。mode 2/3/4不使用该参数。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundScale（bool）</td>
      <td>输入</td>
      <td>是否将scale取整为2的幂。</td>
      <td><ul><li>mode 1/5必须为true。</li><li>mode 2/3/4不使用该参数。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clampLimit（double）</td>
      <td>输入</td>
      <td>SwiGLU计算前的clamp阈值。</td>
      <td><ul><li>-1.0表示不启用clamp；启用时必须大于0。</li><li>mode 5要求有限值。门控分支只限制上界，线性分支限制在[-clampLimit, clampLimit]。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax（double）</td>
      <td>输入</td>
      <td>HiFloat8动态量化的最大有限值。</td>
      <td><ul><li>mode 3用于计算scale；mode 2/3要求该值大于0。</li><li>mode 0/1/4/5不使用该参数。算子属性默认值为15.0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputOrigin（bool）</td>
      <td>输入</td>
      <td>是否输出weight加权前的SwiGLU结果。</td>
      <td><ul><li>true时yOriginOut有效；false时不写入yOriginOut内容。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha（double）</td>
      <td>输入</td>
      <td>SwiGLU激活中Sigmoid输入的缩放系数。</td>
      <td><ul><li>mode 5要求有限正数。</li><li>mode 0/1/4必须使用1.0；mode 2/3不使用该参数。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias（double）</td>
      <td>输入</td>
      <td>SwiGLU激活中线性分支的偏置。</td>
      <td><ul><li>mode 5要求有限数。</li><li>mode 0/1/4必须使用0.0；mode 2/3不使用该参数。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType（int64_t）</td>
      <td>输入</td>
      <td><b>V3新增。</b>groupIndexOptional的列表格式。</td>
      <td><ul><li>支持1（count列表）和2（[group_id, count]二元组）。0（cumsum模式）及其他值不支持。</li><li>类型2必须提供shape为[E, 2]的groupIndexOptional；支持非连续输入，由ACLNN自动转连续。</li><li>mode 5仅支持取1且groupIndexOptional必须为nullptr。</li><li>V3调用需显式传入。V1/V2兼容入口使用默认值1。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut（aclTensor*）</td>
      <td>输出</td>
      <td>量化结果。</td>
      <td><ul><li>shape为[..., D/2]。</li><li>mode 0/1/5的数据类型与dstType一致；mode 2/3为HIFLOAT8；mode 4为HIFLOAT4。</li><li>mode 0/1/4提供groupIndex时，仅保证有效token前缀的输出。</li></ul></td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2、HIFLOAT8、HIFLOAT4</td>
      <td>ND</td>
      <td>与x相同</td>
      <td>×</td>
    </tr>
    <tr>
      <td>yScaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>量化scale。</td>
      <td><ul><li>mode 0：shape为[..., ceil((D/2)/128)]，FLOAT32。</li><li>mode 1/5：shape为[..., ceil(ceil((D/2)/32)/2), 2]，FLOAT8_E8M0。</li><li>mode 2不输出scale，shape为[0]。</li><li>mode 3：无groupIndex时shape为[1]，count模式为[G]，pair模式为[E]，FLOAT32。</li><li>mode 4：shape为[..., ceil((D/2)/64), 1]，使用目标平台的HiFloat4 scale类型。</li></ul></td>
      <td>FLOAT32、FLOAT8_E8M0、HiFloat4 scale类型</td>
      <td>ND</td>
      <td>1-8（依模式而定）</td>
      <td>×</td>
    </tr>
    <tr>
      <td>yOriginOut（aclTensor*）</td>
      <td>输出</td>
      <td>weight加权前的SwiGLU结果。</td>
      <td><ul><li>outputOrigin=true时shape为[..., D/2]，数据类型与x一致。</li><li>不支持空指针。mode 5在outputOrigin=false时shape为[0]；其他模式保留[..., D/2]形状但内容无效。</li></ul></td>
      <td>与x相同</td>
      <td>ND</td>
      <td>与x相同；空输出为1维</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x、yOut、yScaleOut、yOriginOut、workspaceSize或executor存在空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入或输出数据类型、格式等不满足接口约束。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>输入输出shape或属性不满足模式约束；groupListType不是1或2；类型2未提供groupIndexOptional或其shape不是[E, 2]；类型1的groupIndexOptional不是一维；groupIndexOptional为空Tensor；mode 5非法传入groupIndexOptional/scaleOptional；或mode 2缺失scaleOptional等。具体以参数说明和错误日志为准。</td>
    </tr>
  </tbody></table>

## aclnnSwigluGroupQuantV3

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSwigluGroupQuantV3GetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnSwigluGroupQuantV3沿用各量化模式的确定性实现。
- **非连续输入**：x、weightOptional、groupIndexOptional支持由ACLNN自动转连续，调用方无需预先自行连续化；连续化后仍须满足对应模式的shape和数据类型约束。
- **groupListType**：仅支持1和2。类型1为`[G]` count列表；类型2为`[E, 2]`二元组表，必须提供groupIndexOptional，非连续输入由ACLNN自动转连续。0（cumsum边界列表）及其他值不支持。
- **count约束**：count非负，提供的groupIndexOptional不得为空Tensor。类型2中group_id不参与计算。Block/MX/HiFloat4的pair归约使用int32，要求各count及其总和不超过$2^{31}-1$。
- **分组语义**：mode 0/1/4仅按count总和截断有效token前缀；mode 2/3按列表顺序划分组区间，count总和不得超过$T$，scale按组序号而非group_id对应。
- **mode 5**：groupListType必须为1，groupIndexOptional和scaleOptional必须为nullptr；输入为二维`[T,D]`，非空输入的`D >= 64`且`D % 64 == 0`；blockSize为0或32，roundScale为true。
- **mode 0/1/4**：mode 1非空输入的`D >= 64`且`D % 64 == 0`；mode 0/4非空输入的`D >= 256`且`D % 256 == 0`。mode 1/4的输入维度为2-7维；mode 0为2-8维。支持空输入Tensor，传入的groupIndexOptional本身仍须非空。
- **mode 2/3**：输入支持FLOAT32、FLOAT16、BFLOAT16，weight为FLOAT32，不支持空输入Tensor。mode 2必须提供FLOAT32 invScale；mode 3自行计算并输出FLOAT32 scale。
- **mode 4**：HiFloat4仅在目标平台、CANN版本和编译配置均支持时可用，具体规格以对应实现及算子包为准。
- **alpha/bias**：mode 5要求alpha为有限正数、bias为有限数；mode 0/1/4使用alpha=1.0、bias=0.0；mode 2/3不使用这两个参数。
- **yOriginOut**：启用时输出weight加权前的SwiGLU激活结果，数据类型与x一致；即使outputOrigin=false，也须提供合法的Tensor描述。各模式的shape约束见参数表。
- **接口兼容**：V1/V2签名保持不变，使用groupListType=1。V3仅新增列表格式参数，不改变原有量化模式的算法。Torch接口暂不开放该参数。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。
对应源码为[test_aclnn_swiglu_group_quant_v3.cpp](../examples/test_aclnn_swiglu_group_quant_v3.cpp)。

示例使用`quantMode=0`、`groupListType=2`和`[[7,20],[1,12]]`，在64个输入token中仅处理前32个token。
group_id不会改变输入顺序或作为token下标。

```Cpp
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_swiglu_group_quant_v3.h"

namespace SwigluExample {
inline int64_t Numel(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        if (dim <= 0 || size > std::numeric_limits<int64_t>::max() / dim) {
            throw std::invalid_argument("tensor shape must be positive and fit int64");
        }
        size *= dim;
    }
    return size;
}

inline bool CheckHardwareSupport(const char* operatorName)
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        std::cout << "Warning: cannot get SOC name, skip hardware check" << std::endl;
        return true;
    }
    std::cout << "Current SOC: " << socName << std::endl;
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }
    std::cout << "Warning: " << operatorName << " only supports Ascend950, current SOC '" << socName
              << "' is not supported. Skip test." << std::endl;
    return false;
}

class Session {
public:
    explicit Session(int& status) : status_(status) {}
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session()
    {
        if (pending_) {
            Record(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
        }
        if (executor_ != nullptr) {
            Record(aclDestroyAclOpExecutor(executor_), "aclDestroyAclOpExecutor");
        }
        if (workspace_ != nullptr) {
            Record(aclrtFree(workspace_), "aclrtFree(workspace)");
        }
        for (auto& tensor : tensors_) {
            if (tensor->descriptor != nullptr) {
                Record(aclDestroyTensor(tensor->descriptor), "aclDestroyTensor");
            }
            if (tensor->device != nullptr) {
                Record(aclrtFree(tensor->device), "aclrtFree(tensor)");
            }
        }
        if (stream_ != nullptr) {
            Record(aclrtDestroyStream(stream_), "aclrtDestroyStream");
        }
        if (deviceSet_) {
            Record(aclrtResetDevice(deviceId_), "aclrtResetDevice");
        }
        if (initialized_) {
            Record(aclFinalize(), "aclFinalize");
        }
    }

    void Check(int result, const char* operation)
    {
        if (result != ACL_SUCCESS) {
            Record(result, operation);
            throw std::runtime_error(operation);
        }
    }

    void Init(int32_t deviceId)
    {
        deviceId_ = deviceId;
        Check(aclInit(nullptr), "aclInit");
        initialized_ = true;
        Check(aclrtSetDevice(deviceId_), "aclrtSetDevice");
        deviceSet_ = true;
        Check(aclrtCreateStream(&stream_), "aclrtCreateStream");
    }

    template <typename T>
    aclTensor* CreateTensor(const std::vector<T>& host, const std::vector<int64_t>& shape, aclDataType dtype)
    {
        const int64_t elements = Numel(shape);
        if (static_cast<uint64_t>(elements) != host.size() ||
            host.size() > std::numeric_limits<size_t>::max() / sizeof(T) || aclDataTypeSize(dtype) != sizeof(T)) {
            throw std::invalid_argument("tensor shape, storage or dtype size mismatch");
        }
        const size_t bytes = host.size() * sizeof(T);
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size(); i > 1; --i) {
            strides[i - 2] = strides[i - 1] * shape[i - 1];
        }
        tensors_.push_back(std::make_unique<TensorResource>());
        auto& tensor = *tensors_.back();
        Check(aclrtMalloc(&tensor.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(tensor)");
        Check(aclrtMemcpy(tensor.device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy");
        tensor.descriptor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                            shape.data(), shape.size(), tensor.device);
        Check(tensor.descriptor == nullptr ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS, "aclCreateTensor");
        return tensor.descriptor;
    }

    aclOpExecutor** ExecutorAddress() { return &executor_; }

    aclOpExecutor* PrepareExecutor()
    {
        Check(aclSetAclOpExecutorRepeatable(executor_), "aclSetAclOpExecutorRepeatable");
        return executor_;
    }

    void* AllocateWorkspace(uint64_t bytes)
    {
        if (bytes > 0) {
            Check(aclrtMalloc(&workspace_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(workspace)");
        }
        return workspace_;
    }

    aclrtStream StreamForLaunch()
    {
        pending_ = true;
        return stream_;
    }

    void Synchronize()
    {
        Check(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
        pending_ = false;
    }

private:
    struct TensorResource {
        void* device = nullptr;
        aclTensor* descriptor = nullptr;
    };
    void Record(int result, const char* operation)
    {
        if (result != ACL_SUCCESS) {
            std::cerr << operation << " failed: " << result << std::endl;
            if (status_ == ACL_SUCCESS) {
                status_ = result;
            }
        }
    }
    int& status_;
    int32_t deviceId_ = 0;
    bool initialized_ = false;
    bool deviceSet_ = false;
    bool pending_ = false;
    aclrtStream stream_ = nullptr;
    aclOpExecutor* executor_ = nullptr;
    void* workspace_ = nullptr;
    std::vector<std::unique_ptr<TensorResource>> tensors_;
};
} // namespace SwigluExample

using SwigluExample::CheckHardwareSupport;
using SwigluExample::Numel;

int main()
{
    int status = ACL_SUCCESS;
    {
        SwigluExample::Session session(status);
        try {
            session.Init(0);
            if (!CheckHardwareSupport("SwigluGroupQuantV3")) {
                std::cout << "\n=== Test SKIPPED (hardware not supported) ===" << std::endl;
                return ACL_SUCCESS;
            }
            const std::vector<int64_t> xShape{64, 256};
            const std::vector<int64_t> yShape{64, 128};
            const std::vector<int64_t> scaleShape{64, 1};
            std::vector<uint16_t> xHost(Numel(xShape), 0x3C00); // FP16 1.0
            std::vector<uint8_t> yHost(Numel(yShape), 0);
            std::vector<float> scaleHost(Numel(scaleShape), 0.0f);
            const std::vector<int64_t> groupHost{7, 20, 1, 12};
            std::vector<uint16_t> originHost(Numel(yShape), 0);

            aclTensor* x = session.CreateTensor(xHost, xShape, ACL_FLOAT16);
            aclTensor* y = session.CreateTensor(yHost, yShape, ACL_FLOAT8_E4M3FN);
            aclTensor* scale = session.CreateTensor(scaleHost, scaleShape, ACL_FLOAT);
            aclTensor* origin = session.CreateTensor(originHost, yShape, ACL_FLOAT16);

            aclTensor* group = session.CreateTensor(groupHost, {2, 2}, ACL_INT64);

            uint64_t workspaceSize = 0;
            auto ret = aclnnSwigluGroupQuantV3GetWorkspaceSize(x, nullptr, group, nullptr, ACL_FLOAT8_E4M3FN, 0, 128,
                                                               false, -1.0, 15.0, true, 1.0, 0.0, 2, y, scale, origin,
                                                               &workspaceSize, session.ExecutorAddress());
            session.Check(ret, "aclnnSwigluGroupQuantV3");
            aclOpExecutor* executor = session.PrepareExecutor();
            void* workspace = session.AllocateWorkspace(workspaceSize);
            ret = aclnnSwigluGroupQuantV3(workspace, workspaceSize, executor, session.StreamForLaunch());
            session.Check(ret, "aclnnSwigluGroupQuantV3");
            session.Synchronize();

        } catch (const std::exception& error) {
            std::cerr << error.what() << std::endl;
            if (status == ACL_SUCCESS) {
                status = ACL_ERROR_INVALID_PARAM;
            }
        }
    }
    if (status != ACL_SUCCESS) {
        return 1;
    }
    std::cout << "aclnnSwigluGroupQuantV3 example succeeded" << std::endl;
    return 0;
}
```
