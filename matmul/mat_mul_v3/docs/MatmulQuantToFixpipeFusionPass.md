# MatmulQuantToFixpipeFusionPass

## 融合模式

该融合规则利用矩阵乘结果写出时的Fixpipe后处理能力，将矩阵乘及其后的反量化或重新量化合并为一个算子，减少独立的后处理节点。矩阵乘节点的类型为`MatMul`、`MatMulV2`、`BatchMatMul`或`BatchMatMulV2`。下文的“矩阵乘节点”均指这四种类型。

本规则识别融合前计算图（下文称“源图”）中的以下输入组合，并按输入类型确定融合目标。“仅权重量化（weight-only）”指第一个数据输入为浮点类型、第二个数据输入为INT8权重的场景。

| 矩阵乘节点的第一个输入 | 矩阵乘节点的第二个输入 | 融合后的算子类型 |
| ---- | ---- | ---- |
| INT8 | INT8 | `QuantBatchMatmulV3` |
| FLOAT16 | INT8 | `WeightQuantBatchMatmulV2`，须满足仅权重量化场景的额外平台要求 |
| FLOAT32 | INT8 | 可识别此类源图，但当前`WeightQuantBatchMatmulV2`不支持FLOAT32输入，不能将其视为设备可执行的组合 |

融合规则的识别范围和目标算子的执行规格需要同时满足。当前`WeightQuantBatchMatmulV2`的第一个输入`x`只支持FLOAT16、BF16；其中BF16不在本融合规则的源图识别范围内。具体组合还须满足[数据类型与格式](#数据类型与格式)及[平台能力要求](#平台能力要求)。

以下三图仅展示INT8×INT8输入、INT32矩阵乘结果的典型场景。仅权重量化也匹配下述三种节点连接模式，目标为`WeightQuantBatchMatmulV2`，端口映射见[输入与输出的对应关系](#输入与输出的对应关系)。图中使用以下符号。

- `x1`、`x2`：矩阵乘的两个数据输入；`bias`：可选偏置，`optional`表示可以不提供该输入；`y`：图中融合前后对应的输出张量。
- `scale`：缩放参数张量的图示名称。在反量化图中对应`AscendDequant.deq_scale`输入，在重新量化图中对应`AscendRequant.req_scale`输入。`UINT64`表示该张量的数据按算子要求编码。
- `Const(scale)`：以常量节点提供的缩放参数张量；`scale_fused`：三节点融合后生成的缩放参数常量的图示名称，不是新增的算子参数。
- 第三图的`s`、`o`分别表示`AscendQuant.scale`、`AscendQuant.offset`两个标量属性值，与上述`scale`张量是不同的对象。
- 框中的数据类型表示对应张量或节点输出的数据类型。融合后`QuantBatchMatmulV3`的输入索引从0开始：0、1为数据输入，2为`scale`，4为`bias`。

### 矩阵乘与AscendDequant融合

矩阵乘结果直接输入`AscendDequant`时，将两个节点融合为表中对应的矩阵乘算子。`AscendDequant.deq_scale`张量连接到融合后的缩放参数输入，融合输出为FLOAT16。

![矩阵乘与AscendDequant融合，缩放参数和可选bias连接到融合节点](../../../docs/zh/figures/matmul_quant_to_fixpipe_fusion_pass_1.png)

该两节点模式不要求`deq_scale`为编译期常量。`deq_scale`由运行期节点生成时，保留该生成节点及其输入，将生成的张量连接到融合后的矩阵乘算子。

### 矩阵乘与AscendRequant融合

矩阵乘结果直接输入`AscendRequant`时，将两个节点融合为表中对应的矩阵乘算子。`AscendRequant.req_scale`张量连接到融合后的缩放参数输入。融合规则读取源`AscendRequant`的输出描述，接受INT8、FLOAT16或FLOAT32，并将读取到的数据类型写入融合节点的输出描述和`dtype`属性。

独立`AscendRequant`的算子IR仅声明INT8输出。实际执行还须符合融合目标的输出规格，例如`WeightQuantBatchMatmulV2`的输出规格为FLOAT16、BF16或INT8。

`req_scale`可以由运行期节点生成，该生成节点及其输入在融合后保留。

下图以INT8输出的典型场景为例，融合前后输出数据类型一致。

![矩阵乘与AscendRequant融合，以INT8输出为例](../../../docs/zh/figures/matmul_quant_to_fixpipe_fusion_pass_2.png)

### 矩阵乘、AscendDequant与AscendQuant融合

矩阵乘后依次连接`AscendDequant`和`AscendQuant`，且满足[使用约束](#使用约束)以及下述常量和输出连接要求时，将三个节点融合为表中对应的矩阵乘算子，输出为INT8。

三节点融合要求在编译期能够取得`AscendDequant.deq_scale`的常量值，以及`AscendQuant.scale`、`AscendQuant.offset`的属性值。本规则按[三节点 scale 编码](#三节点-scale-编码)生成一维UINT64常量张量，并将其连接到融合后的缩放参数输入。图中的`scale_fused`表示这个常量张量。

图中的`offset=o`是`AscendQuant`的属性，不是`QuantBatchMatmulV3.offset`输入。三节点融合将该偏移信息编码在新生成的缩放参数中，不单独连接`offset`输入，也不为参数合并新增运行期乘法或加法节点。

![三个节点融合，合并缩放和偏移参数并生成UINT64常量](../../../docs/zh/figures/matmul_quant_to_fixpipe_fusion_pass_3.png)

本规则先尝试三节点模式。三节点模式未匹配或未通过校验时，还会检查`矩阵乘 → AscendDequant`两节点模式。两节点模式满足自身的[输出连接](#输出连接与共享场景)和[平台能力](#平台能力要求)等条件时，只合并矩阵乘与`AscendDequant`，`AscendQuant`仍为下游节点并保留其属性。

#### 三节点 scale 编码

编码时从原`deq_scale`常量的原始字节中每次读取8字节，按UINT64元素解释。位编号从0开始，`float_bits`表示读取FLOAT32的原始位模式。

| 步骤 | 运算 | 结果 |
| ---- | ---- | ---- |
| 解码原缩放因子 | 取原元素低32位并与`0xFFFFE000`按位与，再按FLOAT32解释 | 清除低13位，舍弃原元素高32位中的字段 |
| 合并缩放因子 | 解码后的FLOAT32值乘以`AscendQuant.scale` | 得到FLOAT32乘积 |
| 截断乘积位模式 | `float_bits(乘积) & 0xFFFFE000` | 再清除低13位，将保留的位写入第13–31位 |
| 偏移取整 | 对`AscendQuant.offset`执行`std::round`，校验后转为`int64_t` | 取最接近的整数，半整数向远离零的方向取整。取整结果为非有限数或超出`int64_t`范围时，三节点校验不通过 |
| 编码偏移值 | 将取整值转为UINT64后左移37位，再与`0x00001FE000000000`按位与 | 取整值的低8位写入第37–44位 |
| 编码偏移符号 | 原`AscendQuant.offset < 0`时合入`0x0000200000000000` | 第45位依据原值设置，例如`-0.25`取整为0时第45位仍为1 |
| 设置有符号标志 | 合入`0x0000400000000000` | 第46位恒为1 |

各字段按位或得到新的UINT64元素，其余位为0。若原参数含有`L`个UINT64元素，则生成形状为`[L]`、格式为ND的常量。两节点融合直接复用源缩放参数。

### 输入与输出的对应关系

输入端口从0编号。融合后的数据输入、缩放参数和可选偏置按下表连接。

| 融合前的输入 | QuantBatchMatmulV3输入 | WeightQuantBatchMatmulV2输入 | 对应关系 |
| ---- | ---- | ---- | ---- |
| 矩阵乘的第一个数据输入（input 0） | `x1`（input 0） | `x`（input 0） | 使用同一个输入张量 |
| 矩阵乘的第二个数据输入（input 1） | `x2`（input 1） | `weight`（input 1） | 使用同一个输入张量 |
| `AscendDequant.deq_scale`或`AscendRequant.req_scale`（input 1） | `scale`（input 2） | `antiquant_scale`（input 2） | 两节点模式使用融合前的张量，三节点模式使用合并后的常量张量 |
| 矩阵乘的可选`bias`（input 2） | `bias`（input 4） | `bias`（input 6） | 已连接时沿用原偏置的生产者、输出端口和输入描述 |

#### bias 与可选输入

| 源矩阵乘的 bias 状态 | 融合规则的处理 |
| ---- | ---- |
| 缺少`bias`端口，且input 2无数据边 | 跳过该融合候选 |
| `bias`端口为input 2，但未连接 | 可继续匹配，目标`bias`端口保留为空描述 |
| input 2已连接bias | 可继续匹配，按上表将同一个bias张量接到目标端口 |

目标节点构图时保留未连接的可选端口，使用空描述（`DT_UNDEFINED`、`FORMAT_RESERVED`）。本规则留空`QuantBatchMatmulV3`的`offset`（input 3）和`pertoken_scale`（input 5），以及`WeightQuantBatchMatmulV2`的`antiquant_offset`（input 3）、`quant_scale`（input 4）和`quant_offset`（input 5）。源`bias`未连接时，对应的目标`bias`端口同样留空。

#### 转置属性与输出

两个数据输入的转置属性按下表对应。融合保留属性值，不新增转置节点，也不将属性取反。

| 矩阵乘节点类型 | 融合前的两路转置属性 | QuantBatchMatmulV3的对应属性 | WeightQuantBatchMatmulV2的对应属性 |
| ---- | ---- | ---- | ---- |
| `MatMul`、`MatMulV2` | `transpose_x1`、`transpose_x2` | `transpose_x1`、`transpose_x2` | `transpose_x`、`transpose_weight` |
| `BatchMatMul`、`BatchMatMulV2` | `adj_x1`、`adj_x2` | `transpose_x1`、`transpose_x2` | `transpose_x`、`transpose_weight` |

融合后的输出形状与融合前的矩阵乘结果形状一致，数据格式为ND。矩阵乘与`AscendDequant`的两节点融合固定输出FLOAT16；矩阵乘与`AscendRequant`的两节点融合沿用源`AscendRequant`的输出数据类型，取值为INT8、FLOAT16或FLOAT32；三节点融合输出INT8。下游节点改为使用该输出张量。

## 使用约束

### 数据类型与格式

| 检查项 | 要求 |
| ---- | ---- |
| 矩阵乘节点的输入 | input 0和input 1须连接数据，类型符合[融合模式](#融合模式)的输入识别范围。除input 2的可选`bias`外，其他输入不得连接数据。bias端口的三种状态见[bias 与可选输入](#bias-与可选输入) |
| INT8×INT8输入的数据格式 | 第一个输入为ND、NCHW、NHWC、HWCN、CHWN、NDHWC、NCDHW、DHWCN或DHWNC；第二个输入支持这些格式及FRACTAL_NZ |
| 仅权重量化输入的数据格式 | 两个数据输入均为ND、NCHW、NHWC、HWCN、CHWN、NDHWC、NCDHW、DHWCN或DHWNC |
| `AscendDequant`两节点融合的输出 | 融合后固定为FLOAT16；即使融合前`AscendDequant`输出为FLOAT32，融合后也不会保留该输出类型 |
| `AscendRequant`两节点融合的输出 | 融合规则沿用源`AscendRequant`输出描述中的INT8、FLOAT16或FLOAT32数据类型，执行规格见[矩阵乘与AscendRequant融合](#矩阵乘与ascendrequant融合) |
| 三节点融合 | `AscendQuant`输出为INT8。编译期须能读取UINT64编码的`deq_scale`常量和`AscendQuant.scale`、`AscendQuant.offset`，并按[三节点 scale 编码](#三节点-scale-编码)生成缩放参数。生成的参数须符合目标算子在所用平台上的输入要求 |

输出类型需要与下游节点的输入要求一致。尤其是`AscendDequant`的两节点融合，不能假定融合后仍提供FLOAT32结果。

上述格式和类型是融合规则分别检查的条件，不能任意交叉组合。图中数据类型、形状、缩放参数和偏置的组合还须符合[QuantBatchMatmulV3](../../quant_batch_matmul_v3/README.md)或[WeightQuantBatchMatmulV2](../../weight_quant_batch_matmul_v2/README.md)的算子约束。

输入形状中存在未知维度时，不会仅因维度值未知而跳过本规则。动态维度和批次维度广播能否执行，仍须满足目标算子及所用安装包的支持范围。

### 后处理属性

下表列出保持后处理语义的构图前提，需要由用户保证。融合 pass 不以这些属性值作为准入条件，设置为其他值时仍可能发生融合。

| 被合并的节点 | 构图时需保证的属性值 |
| ---- | ---- |
| `AscendDequant` | `relu_flag=false`、`sqrt_mode=false` |
| `AscendRequant` | `relu_flag=false` |
| `AscendQuant`，仅三节点融合 | `sqrt_mode=false`、`round_mode="Round"` |

被合并节点上的ReLU、平方缩放和其他舍入模式不会随融合保留。回退到两节点融合时，留在图中的`AscendQuant`仍按自身属性执行。

### 输出连接与共享场景

- 矩阵乘的输出张量只能接入一个下游输入，且该输入属于所匹配的`AscendDequant`或`AscendRequant`。例如，矩阵乘结果同时作为图输出并输入`AscendDequant`时，不融合这两个节点，以保留矩阵乘结果。
- 三节点模式要求`AscendDequant`的输出只连接到所匹配的`AscendQuant`。当该输出被多个`AscendQuant`共享，或还有其他用途（例如同时作为图输出）时，跳过三节点模式。
- 两节点模式允许末端`AscendDequant`的输出被多个节点使用。共享场景满足两节点融合条件时，只合并矩阵乘与`AscendDequant`，并将原消费者重连到融合节点。该融合节点的输出描述为FLOAT16。

### 平台能力要求

- 平台须具有`Intrinsic_fix_pipe_l0c2out`能力，且不具有`Intrinsic_data_move_l0c2ub`能力。
- 若平台具有`Intrinsic_fix_pipe_l0c2ub`能力，还须具有支持BF16的`Intrinsic_data_move_l12bt`能力。
- 三节点模式和仅权重量化的两节点模式另要求平台能力表中的`Intrinsic_mmad`包含`s8s4`。`s8s4`为平台能力标识，源输入类型见[融合模式](#融合模式)。

平台满足前两项通用条件但缺少`s8s4`能力时，INT8×INT8的三节点图仍可在满足其他两节点条件的情况下只融合矩阵乘与`AscendDequant`，保留`AscendQuant`。仅权重量化的两节点也依赖该能力，因此在此情况下两种模式都不会融合。

未发生融合时，节点仍留在计算图中，需要按各独立算子的规格检查其输入类型、数据格式和形状。例如，独立`AscendDequant`的`x`输入要求INT32；如果保留的矩阵乘结果为FLOAT16，不能直接作为该输入执行。

## 支持的型号

以下为INT8两节点基本模式的适用产品。三节点模式和仅权重量化场景还须满足前述额外平台能力要求，不能仅凭产品名称判断所有模式均可融合。

<!-- npu="910b" id1 -->
Atlas A2系列产品
<!-- end id1 -->

<!-- npu="A3" id2 -->
Atlas A3系列产品
<!-- end id2 -->

<!-- npu="950" id3 -->
Ascend 950PR&950DT系列产品
<!-- end id3 -->
