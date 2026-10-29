# GeluV2ONNXFusionPass

## 融合模式

该融合将小算子组合融合成单个GeluV2算子。erf场景支持3种拓扑（RealDiv + 0.5常量两种位置、TrueDiv），每种按Add/Mul输入顺序展开8个变体共24个模式；tanh场景匹配内核计算的完整近似公式，按κ/β/1.0常量位置×外部乘法顺序展开32个变体（0.5常量仅匹配先乘形式），合计56个模式。

计算公式：

- erf场景（approximate="none"）：

  $$
  gelu(x) = 0.5 \times x \times (1 + erf(x \div \sqrt{2}))
  $$

- tanh场景（approximate="tanh"，与内核一致的完整近似）：

  $$
  gelu(x) = 0.5 \times x \times (1 + tanh(\sqrt{2 \div \pi} \times (x + 0.044715 \times x^3)))
  $$

### erf场景（approximate="none"）

场景一：div节点为RealDiv（除法形式），0.5常量先与x相乘：

![](../../../docs/zh/figures/gelu_v2_onnx_fusion_pass_erf_1.png)

场景二：div节点为RealDiv（除法形式），0.5常量位于末级Mul：

![](../../../docs/zh/figures/gelu_v2_onnx_fusion_pass_erf_2.png)

场景三：div节点为Mul（等价除法TrueDiv），0.5常量先与x相乘：

![](../../../docs/zh/figures/gelu_v2_onnx_fusion_pass_erf_3.png)

### tanh场景（approximate="tanh"）

场景四：完整近似公式，内层链为`Mul(x,x) → Mul(·,x) → Mul(·,κ) → Add(x,·) → Mul(β,·) → Tanh`，外部与erf场景相同的0.5乘法链（0.5常量位置固定为先乘，输入顺序按32个变体枚举）。下图为PyTorch导出规范形（κ/β/1.0常量在第一输入、0.5在mul0），常量标注在连接线上：

![](../../../docs/zh/figures/gelu_v2_onnx_fusion_pass_tanh.png)

## 使用约束

- 数据类型约束：
  - 输入x支持FLOAT16、FLOAT32、BFLOAT16。
  - 公式常量（erf的div常量、tanh的κ/β常量、1.0、0.5）必须是FLOAT16或FLOAT32标量。

- 常量取值约束：
  - erf div常量按场景匹配（FLOAT32容差0.00001，FLOAT16容差0.001）：

    | 场景 | RealDiv（除法形式） | TrueDiv（Mul形式） |
    | ---- | ---- | ---- |
    | erf（approximate="none"） | 1.41421356（√2） | 0.70710678（1/√2） |

  - tanh内层常量：β = 0.79788456（√(2/π)，容差同上）、κ = 0.044715（容差0.0001）。

  - Add常量必须等于1.0，Mul常量必须等于0.5。
  - 常量值不允许为NaN（NaN与任何容差比较均为假，解析阶段显式拒绝）。

- 拓扑约束：
  - 中间节点的输出只能被子图内的下一个节点消费；任一中间结果被子图外的节点复用时跳过融合（tanh场景中间节点为8个：squareMul/cubeMul/kappaMul/innerAdd/betaMul/Tanh/add/mul0）。
  - erf div节点要求x位于第一个输入、常量位于第二个输入；TrueDiv（Mul形式）场景下x不能是标量。
  - tanh内层链固定：`squareMul=Mul(x,x)`、`cubeMul=Mul(squareMul,x)`（左结合）、内层Add（innerAdd）为`Add(x, κx³)`（读序），0.5常量固定先乘（mul0族）；κ/β/1.0常量位置与外部乘法顺序不限。
  - erf的Add/Mul节点常量输入位置不限（支持两种导出顺序）。

- 支持动态shape场景，输出shape与输入x一致。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
