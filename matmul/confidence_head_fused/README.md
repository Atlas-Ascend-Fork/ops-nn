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

将 hidden 与可选 markov_emb 沿最后一维逻辑拼接，与一维 weight 做矩阵向量乘，加上可选 bias，经温度缩放、可选截断和 Sigmoid 后，输出 FP32 置信度。

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
| hidden | 输入 | 特征，shape 为 [N,K1] 或 [B,N,K1]。 | FLOAT16、BFLOAT16 | ND |
| weight | 输入 | 所有行共用的投影权重，shape 为 [K1+K2]，类型与 hidden 一致。 | FLOAT16、BFLOAT16 | ND |
| markov_emb | 可选输入 | shape 为 [N,K2] 或 [B,N,K2]，前导维度、类型与 hidden 一致。缺省时 K2=0。 | FLOAT16、BFLOAT16 | ND |
| temperature | 可选输入 | shape 为 [1]；缺省或当前值为正负零时按 1 计算。 | FLOAT32 | ND |
| bias | 可选输入 | shape 为 [N]，在 batch 间广播。 | FLOAT32 | ND |
| clamp_low | 可选属性 | 指数输入下限，未传入时不做下限截断。 | Float | - |
| clamp_high | 可选属性 | 指数输入上限，未传入时不做上限截断。 | Float | - |
| out | 输出 | shape 为 [N] 或 [B,N]。 | FLOAT32 | ND |

## 约束说明

- 支持合法基础格式，不支持 FRACTAL_NZ。ACLNN 内部将输入转连续并统一为 ND，支持合法非连续输入及输出视图。
- hidden、weight、markov_emb 必须同为 FLOAT16 或同为 BFLOAT16；temperature、bias、out 为 FLOAT32。
- 各维度、B*N 和 K1+K2 须位于 [0,UINT32_MAX]；非空 GEMV 受权重全载 L1 及分块所需片上容量限制。
- temperature 由 Kernel 读取当前值，允许负值，不修改输入 Tensor。
- clamp 作用于 -z/T_eff，先执行下限截断，再执行上限截断。下限 -2147483648.0、上限 2147483648.0 为 FP32 约定值，表示跳过对应截断；显式传入 0 时正常截断。
- K1+K2=0 时仍执行后处理；B=0 或 N=0 时输出为空。

## 调用方式

| 调用方式 | 接口 |
| --- | --- |
| ACLNN | [aclnnConfidenceHeadFused](docs/aclnnConfidenceHeadFused.md)（[调用示例](examples/arch35/test_aclnn_confidence_head_fused.cpp)） |
| Torch 单算子 | [confidence_head_fused](docs/torchapi_confidence_head_fused.md) |
