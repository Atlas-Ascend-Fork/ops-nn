# confidence_head_fused

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

- 接口功能：将hidden与可选的markov_emb沿最后一维逻辑拼接，与一维weight做矩阵向量乘，再加可选的逐行bias，经temperature缩放、可选截断和Sigmoid后，输出每行的FP32置信度。底层封装aclnnConfidenceHeadFused。
- 计算公式：

  $$
  features = \begin{cases}
  hidden, & markov\_emb\text{为None} \\
  \operatorname{cat}(hidden, markov\_emb, dim=-1), & \text{其他情况}
  \end{cases}
  $$

  $$
  z = features @ weight + bias,\qquad u=-z/T_{\mathrm{eff}}
  $$

  bias为None时取0；bias的shape为(N,)，三维输入时在batch之间广播。temperature为None或当前值为+0/-0时，有效温度$T_{\mathrm{eff}}=1$；其余情况使用当前值，允许负值，不修改输入Tensor。

  定义$C=\operatorname{FP32}(\mathrm{INT32\_MAX})=2147483648.0$。clamp_low、clamp_high分别转为FP32的L、H；为None时，L取-C、H取C。计算顺序为：

  ```text
  if L != -C: u = max(u, L)
  if H != C:  u = min(u, H)
  out = 1 / (1 + exp(u))
  ```

  未传入某侧clamp时不做该侧截断；显式传入并转为相同约定值时也跳过该侧，显式0为有效阈值。截断作用于指数输入u，计算使用FP32累加及激活。

## 函数原型

```python
cann_ops_nn.confidence_head_fused(
    hidden,
    weight,
    *,
    markov_emb=None,
    temperature=None,
    bias=None,
    clamp_low=None,
    clamp_high=None,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| hidden | Tensor | 必选 | 待投影的特征，最后一维为K1。 | torch.float16、torch.bfloat16 | (N, K1) 或 (B, N, K1) |
| weight | Tensor | 必选 | 所有行与batch共用的一维投影权重，长度为K1+K2。 | 与hidden一致 | (K1+K2,) |
| markov_emb | Tensor | 可选 | 沿最后一维与hidden逻辑拼接；前导维度与hidden一致。默认值为None，此时K2=0。 | 与hidden一致 | (N, K2) 或 (B, N, K2) |
| temperature | Tensor | 可选 | 温度系数。默认值为None；未传入或当前值为正负零时按1计算，其他值按当前值计算。 | torch.float32 | (1,) |
| bias | Tensor | 可选 | 逐行偏置，在batch间广播。默认值为None。 | torch.float32 | (N,) |
| clamp_low | float | 可选 | 指数输入下限，内部转为FP32。默认值为None，表示不做下限截断。 | - | - |
| clamp_high | float | 可选 | 指数输入上限，内部转为FP32。默认值为None，表示不做上限截断。 | - | - |

## 返回值说明

| 输出名 | 输出类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| out | Tensor | 每行的置信度，shape为hidden去除最后一维。 | torch.float32 | (N,) 或 (B, N) |

## 约束说明

- 当前支持Torch单算子模式调用。
- 所有Tensor须在同一NPU设备，可选Tensor可以传入None。支持合法基础格式及非连续视图，不支持FRACTAL_NZ。
- hidden、weight、markov_emb必须同为torch.float16或同为torch.bfloat16。weight不支持batch维度，bias只支持一维 (N,)。
- 各维度、展平后的行数B*N和K1+K2均须位于 [0,UINT32_MAX]；非空GEMV受权重全载L1及分块所需片上容量限制。
- temperature由Kernel读取当前值，允许负值，不修改输入Tensor。
- clamp作用于指数输入 -z/T_eff。约定下限 -2147483648.0、上限2147483648.0表示跳过对应截断；显式0为有效阈值。两侧均启用时先max下限、再min上限，不要求下限小于上限。
- K1+K2=0时仍执行bias、温度缩放、可选截断和Sigmoid；B=0或N=0时返回空输出。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用（eager）

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  hidden = torch.randn((2, 129, 33), dtype=torch.float16).npu()
  weight = torch.randn((40,), dtype=torch.float16).npu()
  markov_emb = torch.randn((2, 129, 7), dtype=torch.float16).npu()
  temperature = torch.tensor([2.0], dtype=torch.float32).npu()
  bias = torch.randn((129,), dtype=torch.float32).npu()

  out = cann_ops_nn.confidence_head_fused(
      hidden,
      weight,
      markov_emb=markov_emb,
      temperature=temperature,
      bias=bias,
      clamp_low=-60.0,
      clamp_high=60.0,
  )
  print(out.shape, out.dtype)
  ```
