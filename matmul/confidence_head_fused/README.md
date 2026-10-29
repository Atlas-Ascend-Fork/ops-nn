# ConfidenceHeadFused

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

将hidden与可选markov_emb沿最后一维逻辑拼接，与一维weight做矩阵向量乘，加上可选bias，经温度缩放、可选截断和Sigmoid后，输出FP32置信度。

```text
features = cat([hidden, markov_emb], dim=-1)  # markov_emb 缺省时为 hidden
z = features @ weight + bias                # bias 缺省时为 0
T_eff = 1 if temperature is None or temperature[0] == 0 else temperature[0]
u = -z / T_eff
if clamp_low is enabled:  u = max(u, clamp_low)
if clamp_high is enabled: u = min(u, clamp_high)
out = 1 / (1 + exp(u))
```

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| hidden | 输入 | 特征，shape为 [N,K1] 或 [B,N,K1]。 | FLOAT16、BFLOAT16 | ND |
| weight | 输入 | 所有行共用的投影权重，shape为 [K1+K2]，类型与hidden一致。 | FLOAT16、BFLOAT16 | ND |
| markov_emb | 可选输入 | shape为 [N,K2] 或 [B,N,K2]，前导维度、类型与hidden一致。缺省时K2=0。 | FLOAT16、BFLOAT16 | ND |
| temperature | 可选输入 | shape为 [1]；缺省或当前值为正负零时按1计算。 | FLOAT32 | ND |
| bias | 可选输入 | shape为 [N]，在batch间广播。 | FLOAT32 | ND |
| clamp_low | 可选属性 | 指数输入下限，未传入时不做下限截断。 | Float | - |
| clamp_high | 可选属性 | 指数输入上限，未传入时不做上限截断。 | Float | - |
| out | 输出 | shape为 [N] 或 [B,N]。 | FLOAT32 | ND |

## 约束说明

- 支持合法基础格式，不支持FRACTAL_NZ。ACLNN内部将输入转连续并统一为ND，支持合法非连续输入及输出视图。
- hidden、weight、markov_emb必须同为FLOAT16或同为BFLOAT16；temperature、bias、out为FLOAT32。
- 各维度、B*N和K1+K2须位于 [0,UINT32_MAX]；非空GEMV受权重全载L1及分块所需片上容量限制。
- temperature由Kernel读取当前值，允许负值，不修改输入Tensor。
- clamp作用于 -z/T_eff，先执行下限截断，再执行上限截断。下限 -2147483648.0、上限2147483648.0为FP32约定值，表示跳过对应截断；显式传入0时正常截断。
- K1+K2=0时仍执行后处理；B=0或N=0时输出为空。

## 调用说明

- <term>Ascend 950PR&950DT系列产品</term>：

  | 调用方式 | 样例代码 | 说明 |
  | --- | --- | --- |
  | aclnn接口 | [test_aclnn_confidence_head_fused](examples/arch35/test_aclnn_confidence_head_fused.cpp) | 通过<br>[aclnnConfidenceHeadFused](docs/aclnnConfidenceHeadFused.md)<br>接口调用ConfidenceHeadFused算子。 |
  | PyTorch API | [调用示例](docs/torchapi_confidence_head_fused.md#调用示例) | 通过<br>[confidence_head_fused](docs/torchapi_confidence_head_fused.md)<br>接口调用ConfidenceHeadFused算子。 |
