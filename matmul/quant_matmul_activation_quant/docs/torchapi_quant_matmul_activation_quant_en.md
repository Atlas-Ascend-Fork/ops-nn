# quant_matmul_activation_quant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:00:11.342Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description:

  Fuses quantized matrix multiplication, activation, and dynamic quantization, encapsulating `aclnnQuantMatmulActivationQuantWeightNz`. Currently, the gelu activation and MX quantization mode are supported. Inputs `x1` and `x2` are P8 quantized matrices. Required input `x2_scale` and optional input `x1_scale` are MX quantization scale factors, and `bias` is the bias term. The matrix multiplication result is dynamically quantized after passing through the activation function, outputting the quantized result `y` and quantization scale `y_scale`. The M/N/K dimensions and transpose flags are automatically derived from the last two dimensions of `x1` and `x2`; explicit passing is not required.

- Formula:

  - QuantMatmul MX quantization mode:

    $$
    matmulOut[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice))* (x1Scale[m/gsM, j] * x2Scale[j, n/gsN]))+bias[n]
    $$

    where gsM, gsN, and gsK denote groupSizeM, groupSizeN, and groupSizeK respectively; x1Slice denotes the vector of length groupSizeK in row m of x1, and x2Slice denotes the vector of length groupSizeK in column n of x2; slices on the K axis both start at j*groupSizeK, where j takes values in [0, kLoops), kLoops = ceil(K / groupSizeK), and K is the length of the K axis. The last slice can be shorter than groupSizeK.

  - Activation formula:

    - gelu_tanh (high-performance approximation):
    $$
    activationOut=GELU(matmulOut)=matmulOut × Φ(matmulOut)=0.5 * matmulOut * (1 + tanh( \sqrt{2 / \pi} * (matmulOut + 0.044715 * matmulOut^{3})))
    $$

    - gelu_erf:
    $$
    activationOut=GELU(matmulOut)=0.5 * matmulOut * (1 + erf(matmulOut / \sqrt{2}))
    $$

  - Dynamic quantization formula:

    - **Use case 1: when scale_alg is 0**:
      - Group the input activationOut along the last axis by $k = 32$ values. A group of k values $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized into $\{mxscale1, \{P_i\}_{i=1}^{k}\}, k = 32$

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
      mxscale = 2^{shared\_exp}\\
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space k\\
      $$

      - The quantized $P_{i}$ values form the output yOut according to the corresponding positions of $V_{i}$, and mxscale is output as yScaleOut for each group along the last axis.

      - emax: The exponent of the largest normal number of the corresponding data type.

          | DataType | Emax |
          | :-----------: | :--: |
          |  FLOAT4_E2M1  |  2   |
          | FLOAT8_E4M3FN |  8   |
          |  FLOAT8_E5M2  |  15  |

    - **Scenario 2, when scale_alg is 1, only FP8 types are involved**:
      - The input activationOut is divided into blocks of $k = 32$ elements along the tail axis, and a block scale factor $S_{fp32}^b$ is computed for each block separately. All elements in the block are then mapped to the target low-precision type FP8 using the same $S_{fp32}^b$. If the last block contains fewer than $k = 32$ elements, the missing values are treated as 0 and the block is processed as a complete block.
      - Find the maximum absolute value of the values in the block:

        $$
        Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
        $$

      - FP32 is mapped to within the range representable by the target data type FP8, where $Amax(DType)$ is the maximum value representable by the target precision:

        $$
        S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
        $$

      - The block scale factor $S_{fp32}^b$ is converted to the scale value $S_{ue8m0}^b$ representable in FP8 format
      - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point scale factor $S_{fp32}^b$ of the block
      - To avoid overflow during quantization, the exponent is rounded up and kept within the range representable by FP8:

        $$
        E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a subnormal number, and} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{ otherwise} \end{cases}
        $$

      - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
      - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
      - The final step applied to quantization: for each element in a block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$, and the final quantized output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ denotes the scale factor of the block, that is, $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ denotes the quantized data in the block.

    - **Scenario 3: when scale_alg is 2, only the FP4_E2M1 type is involved**:
      - When dstTypeMax = 0.0/6.0/7.0:
        - The input x is grouped along the axis dimension in groups of k = blocksize elements, and each group of k numbers  $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize:
        $$
        shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{ if} the one or two bits preceding the high bit of the mantissa \text{ are 1, and the mantissa is not all 0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{ otherwise} \end{cases} \\
        $$
        $$
        P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
        $$
        - ​The quantized $P_{i}$ form the output yOut according to the positions of the corresponding $V_{i}$, and mxscale forms the output mxscaleOut according to the groups along the axis dimension.
      - When dstTypeMax != 0.0/6.0/7.0:
        - The long vector is divided into blocks, each with a length of k. A block scale factor $S_{fp32}^b$ is computed separately for each block, and then all elements in the block are mapped to the target low-precision type using the same $S_{fp32}^b$. If the last block has fewer than k elements, the missing values are treated as 0, and the block is processed as a complete one.
        - Find the maximum absolute value in the block:
        $$
        Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
        $$
        - Map FP32 to within the range representable by the target data type, where if dst_max_value=0, $Amax(DType)$ is the maximum value representable by the target precision; if dst_max_value!=0, $Amax(DType)$ is the value passed in for dst_max_value.
        $$
        S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
        $$
        - Convert the block quantization scale factor $S_{fp32}^b$ to a scale value $S_{ue8m0}^b$ representable in the FP8 format.
        - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point quantization scale factor $S_{fp32}^b$ of the block.
        - To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:
          $$
          E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b, & \text{ otherwise} \end{cases}
          $$
        - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
        - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
        - Apply to the final quantization step. For each element within a block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$. The final quantized output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ represents the block scale factor, i.e., $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ represents the quantized data within the block.
        - ​The quantized $P_{i}$ are combined into the output yOut according to the positions of the corresponding $V_{i}$, and mxscale is combined into the output mxscaleOut according to the corresponding groupings along the axis dimension.

## Function Prototype

```python
cann_ops_nn.quant_matmul_activation_quant(x1, x2, x2_scale, *, x1_scale=None, bias=None,
    output_dtype=None, x1_dtype=None, x2_dtype=None, x1scale_dtype=None, x2scale_dtype=None,
    group_sizes=None, activation_type="gelu_tanh", quant_mode="mx", round_mode="rint",
    scale_alg=0, dst_type_max=0.0) -> (Tensor y, Tensor y_scale)
```

## Mandatory

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| `x1` | Tensor | Required | Left matrix of the matrix multiplication operation. The data format is ND. The last two dimensions are `(M, K)` or `(K, M)`, determined by automatic derivation from matching with the dimensions of `x2`. | torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2 | 2-6 dimensions, `(..., M, K)` or `(..., K, M)` |
| `x2` | Tensor | Required | Right matrix of the matrix multiplication operation. The data format is ND or FRACTAL_NZ. The last two dimensions are `(K, N)` or `(N, K)`, determined by automatic derivation from matching with the dimensions of `x1`. | torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2 | 2-6 dimensions, `(..., K, N)` or `(..., N, K)` |
| `x2_scale` | Tensor | Required | MX quantization scale factor of x2 in the matrix multiplication computation. The data format is ND. The batch dimension must be the same as that of `x2`. | torch.float8_e8m0fnu | `(..., K//64, N, 2)` or `(..., N, K//64, 2)` (depending on the direction of `x2`) |
| `x1_scale` | Tensor | Optional | MX quantization scale factor of x1 in the matrix multiplication computation. The data format is ND. The batch dimension must be the same as that of `x1`. | torch.float8_e8m0fnu | `(..., M, K//64, 2)` or `(..., K//64, M, 2)` (depending on the direction of `x1`) |
| `bias` | Tensor | Optional | Bias accumulated after the matrix multiplication operation. The data format is ND. | float32 | `(N,)` |
| `output_dtype` | int | Optional | Data type enumeration value of the output `y`. Supports torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2, etc. Default value: None (indicating the same type as `x1`). | int | - |
| `x1_dtype` | int | Optional | Data type enumeration value of `x1`. When not passed in, determined by automatic derivation according to the scalar_type of `x1`. | int | - |
| `x2_dtype` | int | Optional | Data type enumeration value of `x2`. When not passed in, determined by automatic derivation according to the scalar_type of `x2`. | int | - |
| `x1scale_dtype` | int | Optional | Data type enumeration value of `x1_scale`. When not passed in, determined by automatic derivation according to the scalar_type of `x1_scale`. | int | - |
| `x2scale_dtype` | int | Optional | Data type enumeration value of `x2_scale`. When not passed in, determined by automatic derivation according to the scalar_type of `x2_scale`. | int | - |
| `group_sizes` | List[int] | Optional | Group quantization sizes `[groupSizeM, groupSizeN, groupSizeK]`. The value range of each element is [0, 65535]. | list | `(3,)` |
| `activation_type` | str | Optional | Activation function type. Supports `"gelu_tanh"` and `"gelu_erf"`. Default value: `"gelu_tanh"`. | string | - |
| `quant_mode` | str | Optional | Quantization mode. Currently supports `"mx"`. Default value: `"mx"`. | string | - |
| `round_mode` | str | Optional | Rounding mode. If `output_dtype` is FLOAT4_E2M1, supported values are `"rint"`, `"floor"`, and `"round"`; if `output_dtype` is FLOAT8_E4M3FN/FLOAT8_E5M2, only `"rint"` is supported. Default value: `"rint"`. | string | - |
| `scale_alg` | int | Optional | Scale algorithm. If `output_dtype` is FLOAT4_E2M1, supported values are 0 and 2; 0 indicates scenario 1, and 2 indicates scenario 3 (enabling `dst_type_max`); if `output_dtype` is FLOAT8_E4M3FN/FLOAT8_E5M2, supported values are 0 and 1; 0 indicates scenario 1, and 1 indicates scenario 2. Default value: 0. | int | - |
| `dst_type_max` | float | Optional | Maximum value of the target data type, used for quantization range control. Does not take effect when `scale_alg` is 0 or 1; simply pass in 0.0. If `scale_alg` is 2, supported values are 0.0 and 6.0-12.0; 0.0 indicates using the default maximum value of the target type. Default value: 0.0. | float32 | - |

## Return Values

| Output | Output Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| `y` | Tensor | Required | Matrix multiplication operation and activation computation result after dynamic quantization. | torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2 | `(..., M, N)`; |
| `y_scale` | Tensor | Required | Quantization scale corresponding to each group after dynamic quantization. The last dimension is fixed to 2. | torch.float8_e8m0fnu | `(..., M, CeilDiv(N, 64), 2)` |

## Constraints

- This API supports training and inference use cases.
- This API supports calls in single-operator mode.
- Empty tensors are not supported.
- Contiguous tensors are supported. Non-contiguous tensors support only the use case where the last two dimensions are transposed.
- `x1` supports 2-6 dimensions. When `x2` is in NZ format, it supports 4-8 dimensions; when `x2` is in ND format, it supports 2-6 dimensions.
- When `x2` is in NZ format, only the data type torch.float8_e4m3fn is supported.
- When `K` or `N` is 1, the weightNz feature cannot be used. This API does not support this use case.
- The batch dimensions of `x1` and `x2` (dimensions other than the last two dimensions) support broadcasting (right-aligned). For example, if `x1=(1,M,K)` and `x2=(8,K,N)`, the output is `(8,M,N)`.
- If `x1_scale` and `x2_scale` are passed in, the number of their batch dimensions (dimensions other than the last three dimensions) and the value of each dimension must be exactly the same as the corresponding `x1` and `x2`; if `x1` has no batch dimension (2D), `x1_scale` and `x2_scale` must be 3D.
- The last dimension of `x1_scale` and `x2_scale` must be 2.
- If `group_sizes` is passed in, it must contain three elements `[groupSizeM, groupSizeN, groupSizeK]`, each within the range [0, 65535]. Currently, only [0, 0, 0] and [1, 1, 32] are supported in MX scenarios.
- The data type of `y` is determined by the data type of `x1`, and the two must be consistent.
- The input and output support the following data type combinations:

  | x1 | x2 | x1_scale | x2_scale | Bias | y | y_scale |
  |---------------|---------------|-------------|-------------|------------------|---------------------------|-------------|
  | torch.float8_e4m3fn | torch.float8_e4m3fn | torch.float8_e8m0fnu | torch.float8_e8m0fnu | None/torch.float32  | torch.float8_e4m3fn | torch.float8_e8m0fnu |
  | torch.float8_e5m2   | torch.float8_e4m3fn | torch.float8_e8m0fnu | torch.float8_e8m0fnu | None/torch.float32  | torch.float8_e5m2 | torch.float8_e8m0fnu |
  | torch.float8_e5m2   | torch.float8_e5m2 | torch.float8_e8m0fnu | torch.float8_e8m0fnu | None/torch.float32  | torch.float8_e5m2 | torch.float8_e8m0fnu |
  | torch.float8_e4m3fn | torch.float8_e5m2 | torch.float8_e8m0fnu | torch.float8_e8m0fnu | None/torch.float32  | torch.float8_e4m3fn | torch.float8_e8m0fnu |
  | torch.float4_e2m1fn_x2 | torch.float4_e2m1fn_x2 | torch.float8_e8m0fnu | torch.float8_e8m0fnu | None/torch.float32  | torch.float4_e2m1fn_x2 | torch.float8_e8m0fnu |

- MXFP4 scenario constraints (the actual data types of `x1`, `x2`, and `y` are all `torch.float4_e2m1fn_x2`):
  - Currently, `x1`, `x2`, and `y` are all packed as uint8. The last axis of the shape before packing must be even.
  - If `x2` is in NZ format, transposition is not supported for `x1`.
  - `scale_alg` supports only the values 0 and 2.
  - If `scale_alg` is 2, the supported values of `dst_type_max` are 0.0 and 6.0-12.0.
  - `round_mode` supports `"rint"`, `"floor"`, and `"round"`.

## Deterministic Computation

Defaults to a deterministic computation.

## Example

- Call in single-operator mode

  - FP8 scenario example:

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_nn

    m, k, n = 5, 64, 128
    group_size = 32
    # x1 physical shape (M, K); x2 physical shape (K, N)
    x1 = torch.randn(m, k, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
    x2 = torch.randn(k, n, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
    x2_nz = torch_npu.npu_format_cast(x2, 29) # 29 is the NZ format
    x1_scale = torch.ones(m, math.ceil(k / group_size / 2), 2, dtype=torch.float8_e8m0fnu).npu()
    x2_scale = torch.ones(math.ceil(k / group_size / 2), n, 2, dtype=torch.float8_e8m0fnu).npu()

    y, y_scale = torch.ops.cann_ops_nn.quant_matmul_activation_quant(
        x1, x2_nz, x2_scale, x1_scale=x1_scale, bias=None,
        activation_type="gelu_tanh", quant_mode="mx", round_mode="rint",
        scale_alg=0, dst_type_max=0.0)
    print("y: ", y.cpu())
    print("y_scale: ", y_scale.cpu())
    ```

  - FP4 scenario example:

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_nn

    m, k, n = 5, 64, 128
    group_size = 32
    # x1 physical shape (M, K//2); x2 physical shape (K, N//2), FP4 dual-nibble packed into uint8, halving the last dimension
    x1 = torch.randn(m, k // 2, dtype=torch.float32).to(torch.uint8).npu()
    x2 = torch.randn(k, n // 2, dtype=torch.float32).to(torch.uint8).npu()
    x2_nz = torch_npu.npu_format_cast(x2, 29) # 29 is the NZ format
    x1_scale = torch.ones(m, math.ceil(k / group_size / 2), 2, dtype=torch.float8_e8m0fnu).npu()
    x2_scale = torch.ones(math.ceil(k / group_size / 2), n, 2, dtype=torch.float8_e8m0fnu).npu()

    y, y_scale = torch.ops.cann_ops_nn.quant_matmul_activation_quant(
        x1, x2_nz, x2_scale, x1_scale=x1_scale, bias=None,
        output_dtype=torch_npu.float4_e2m1fn_x2,
        x1_dtype=torch_npu.float4_e2m1fn_x2,
        x2_dtype=torch_npu.float4_e2m1fn_x2,
        activation_type="gelu_tanh", quant_mode="mx", round_mode="rint",
        scale_alg=0, dst_type_max=0.0)
    # y physical shape (M, N//2), FP4 dual-nibble packed into uint8, halving the last dimension, y_scale physical shape(M, CeilDiv(N, 64), 2)
    print("y: ", y.cpu())
    print("y_scale: ", y_scale.cpu())
    ```
