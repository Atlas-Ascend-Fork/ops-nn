# LpNormReduceV2

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                   |    √     |
| <term>Atlas A3系列产品</term> |    ×     |
| <term>Atlas A2系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                  |    ×     |
| <term>Atlas推理系列产品</term>                          |    ×     |
| <term>Atlas训练系列产品</term>                           |    ×     |

## 功能说明

- 算子功能：计算Lp范数的归约段。对输入`x`逐元素取绝对值后，沿属性`axes`指定的轴做归约，归约方式由属性`p`决定。本算子只做归约，不做最后的$\frac{1}{p}$次开方；完整Lp范数由LpNormReduceV2与匹配的LpNormUpdateV2算子串联得到。
- 与LpNormReduce（V1）的核心差异：属性`p`为**FLOAT**类型（V1为INT），支持±inf与任意有限浮点阶数，判定优先级从上到下为：

| 优先级 | p取值 | 计算公式 |
|-------|--------|---------|
| 1 | p = +inf（含浮点别名2147483647.0/2147483648.0） | $y = \max\limits_{i \in S}{a_i}$ |
| 2 | p = −inf（含浮点别名−2147483648.0） | $y = \min\limits_{i \in S}{a_i}$ |
| 3 | p = 0 | $y = \sum_{i \in S}{[a_i \neq 0]}$（非零元素个数） |
| 4 | p = 1 | $y = \sum_{i \in S}{a_i}$ |
| 5 | p为整数且2≤p≤15 | $y = \sum_{i \in S}{a_i^{p}}$（连乘实现） |
| 6 | 其他有限浮点p | $y = \sum_{i \in S}{\exp(p \cdot \ln a_i)}$ |

  其中$a_i = \lvert x_i \rvert$，$S$为规范化后的归约轴集合（`axes`为空表示全轴，负轴按`axis + rank`折算，重复轴报错）。

- 升精度计算语义：FLOAT16、BFLOAT16输入先转换到FLOAT32计算（绝对值与归约均在FLOAT32内完成），结果回写原数据类型——FLOAT16就近取整转回，BFLOAT16按round-to-nearest-even回写；FLOAT32输入直接计算。p=±inf与p=0分支为精确运算（输入不含NaN时与参考实现逐位一致）。
- 示例：x shape为`(4, 32, 16, 70)`，`axes = [1]`，`keepdim = true`，`p = 2.0`时，输出y shape为`(4, 1, 16, 70)`，每个输出元素等于输出位置$(i, 0, k, l)$沿轴1的32个输入元素的平方和（被归约的轴1在输出中保留为长度1）。

## 参数说明

| 参数名  | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ------- | -------------- | ---- | -------- | -------- |
| x       | 必选输入       | 公式中的`x`，表示待归约张量。rank范围为0到8。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| y       | 必选输出       | 公式中的输出`y`。数据类型与`x`一致，shape由`axes`与`keepdim`决定。 | FLOAT、FLOAT16、BFLOAT16 | ND |
| p       | 可选属性       | 公式中的`p`，表示范数的浮点阶数，默认值为2.0。遵循IEEE 754取值范围，支持±inf（含fp32存储别名2147483647.0/2147483648.0与−2147483648.0）和任意有限浮点值；不支持NaN。 | FLOAT | - |
| axes    | 可选属性       | 公式中的归约轴集合$S$。默认值为空列表。空列表表示对所有轴归约；取值范围为[-rank(`x`), rank(`x`))，支持负轴；元素不允许重复、数量不超过rank(`x`)。 | LISTINT | - |
| keepdim | 可选属性       | 决定输出张量是否保留归约轴。为true时归约轴保留为长度1；为false时删除归约轴，全轴归约时输出为0维标量。默认值为false。 | BOOL | - |
| epsilon | 可选属性       | 预留参数，不参与计算。对齐TBE（Tensor Boost Engine，张量加速引擎）参考实现，由后续LpNormUpdateV2消费。默认值为1e-12。 | FLOAT | - |

## 约束说明

- 本算子是LpNormReduce的浮点p演进版本，仅支持GE（Graph Engine，图引擎）图模式与Kernel兼容路径调用，不提供aclnn接口。
- 输入总元素数不超过$2^{31}-1$。
- 输入含NaN时不做过滤：NaN按IEEE 754语义参与计算（p=0分支中NaN计为非零元素；p=±inf分支中含NaN的归约结果为NaN）。
- p∈[2,15]整数分支中间幂值$|x|^p$可能超出FLOAT32上限得到inf，不做饱和保护（与参考实现一致）。
- 有限p的归约域为空时（如shape含0维）输出0；若零长度维仅位于非归约轴，输出为空Tensor。
- 输出shape推导：keepdim=true时`y.shape[i] = (i ∈ axes) ? 1 : x.shape[i]`；keepdim=false时删除被归约维度；axes为空且keepdim=false时输出为0维标量。
- 已知边界（p=−inf分支）：整行$|x|$全为+inf的退化输入输出FLT_MAX而非+inf（底层归约取最小值指令的初始哨兵值胜出，与{FLT_MAX,+inf}混合集合不可区分）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| GE图模式 | -        | 通过[LpNormReduceV2算子IR（Intermediate Representation，中间表示）定义](op_graph/lp_norm_reduce_v2_proto.h)构图调用；仓内[`../lp_norm_v2/op_api/lp_norm_reduce_v2.cpp`](../lp_norm_v2/op_api/lp_norm_reduce_v2.cpp)提供构图辅助。 |

## 构建与测试

```bash
# 编译算子（需指定soc）
bash build.sh --ops=lp_norm_reduce_v2 --soc=ascend950

# 运行UT（op_host infershape/tiling + op_graph inferdtype，编译后自动执行）
bash build.sh --ops=lp_norm_reduce_v2 --soc=ascend950 -u
```

- UT（Unit Test，单元测试）：`tests/ut/`（infershape 26例 + arch35 tiling 90例 + 跨核路由tiling 16例 + op_graph InferDataType 6例，经仓内共享Host UT runner执行）。
- ST（System Test，系统测试）：`tests/st/`（`ttk_geir_lp_norm_reduce_v2.csv` 31例 + `arch35/ttk_kernel_lp_norm_reduce_v2.csv` 93例，覆盖6个p分支×3种数据类型×多种归约布局与跨核场景）。三方精度参考见`tests/assets/golden.py`，基于torch官方Lp范数接口`torch.linalg.vector_norm`（ord=inf/-inf/0/1直接映射各分支，任意p经`norm^p`还原幂和）。
