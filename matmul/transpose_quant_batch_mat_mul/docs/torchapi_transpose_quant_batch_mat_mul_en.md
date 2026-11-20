# transpose_quant_batch_mat_mul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:12:58.207Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
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

- Description: Performs quantized matrix multiplication on tensors x1 and x2. The bottom layer encapsulates aclnnTransposeQuantBatchMatMul; when x2 is in FRACTAL_NZ format, it encapsulates aclnnTransposeQuantBatchMatMulWeightNz. The API supports four quantization modes: MX (MXFP8 and MXFP4), FP8 (K-C), HIFP8, and INT8 W8A8 (K-C).
- Four quantization modes:

  | Quantization Mode | x1/x2 Data Type | x1_scale | x2_scale | K/N Constraints |
  | --- | --- | --- | --- | --- |
  | MX | torch.float8_e4m3fn (MXFP8) or torch.uint8 (MXFP4, x1_dtype/x2_dtype set to float4_e2m1fn_x2) | torch.float8_e8m0fnu, 4D | torch.float8_e8m0fnu, 4D | K is a multiple of 64; N is unlimited |
  | FP8 (K-C) | torch.float8_e4m3fn and torch.float8_e5m2 (any combination) | torch.float32, 1D (M,) | torch.float32, 1D (N,) | K=512, N=128 |
  | HIFP8 | torch.uint8 (hifloat8 storage, x1_dtype/x2_dtype set to torch_npu.hifloat8) | None | torch.uint64 or torch.int64 (interpreted as bits; the lower 32 bits are the float32 bit pattern), 1D (N,), per-channel FIXPIPE dequant scale | K and N are unlimited and must be positive |
  | INT8 W8A8 (K-C) | torch.int8 | torch.float32, 2D (M, B) | torch.float32, 2D (B, N) | K is a multiple of 32; N is a multiple of 16 |

- Formula (taking each slice along the batch dimension as an example):

  - MX quantization:

    $$
    y[m, n] = \sum_{j=0}^{K/64-1} \sum_{s=0}^{1} \left(\left(\sum_{k=0}^{31} x1[m, j \times 64 + s \times 32 + k] \cdot x2^{(p)}[j \times 64 + s \times 32 + k, n]\right) \cdot x1Scale[m, j, s] \cdot x2Scale^{(p)}[j, n, s]\right)
    $$

    where K is the K-axis length of the matrix multiplication, x1Scale and x2Scale are MX scale factors encoded in torch.float8_e8m0fnu, and the last dimension s (0 or 1) corresponds to the two 32-element sub-blocks in a K/64 group. The indices of $x2^{(p)}$ and $x2Scale^{(p)}$ depend on perm_x2: when perm_x2 is [0, 1, 2], $x2^{(p)}[...] = x2[j \times 64 + s \times 32 + k, n]$ and $x2Scale^{(p)}[j, n, s] = x2Scale[j, n, s]$; when perm_x2 is [0, 2, 1], $x2^{(p)}[...] = x2[n, j \times 64 + s \times 32 + k]$ and $x2Scale^{(p)}[j, n, s] = x2Scale[n, j, s]$.

  - FP8 (K-C):

    $$
    y[m, n] = \left(\sum_{k=0}^{K-1} x1[m, k] \cdot x2[k, n]\right) \cdot x1Scale[m] \cdot x2Scale[n]
    $$

  - HIFP8:

    $$
    y[m, n] = \left(\sum_{k=0}^{K-1} x1[m, k] \cdot x2[k, n]\right) \cdot x2Scale[n]
    $$

    where x1 and x2 are HIFLOAT8 and x2Scale is the Fixpipe per-channel dequant scale.

  - INT8 W8A8 (K-C):

    $$
    y[m, b, n] = \mathrm{bf16}\left(\mathrm{fp32}\left(\sum_{k} x1[m, b, k] \cdot x2[b, k, n]\right) \cdot x1Scale[m, b] \cdot x2Scale[b, n]\right)
    $$

    dtype supports only 27 (torch.bfloat16). perm_x2 supports only [0, 1, 2].

- Example: Assume that the shape of x1 is (M, B, K), the shape of x2 is (B, K, N), and the shape of output y is (M, B, N).

## Function Prototype

```python
cann_ops_nn.transpose_quant_batch_mat_mul(
    x1,
    x2,
    *,
    dtype,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    group_sizes=None,
    perm_x1=None,
    perm_x2=None,
    perm_y=None,
    batch_split_factor=None,
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
) -> Tensor
```

## Mandatory

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| x1 | Tensor | Required | Left matrix in the matrix multiplication, with shape (M, B, K). In MXFP4 use cases, the last dimension is the physical length K/2 after FP4 packing. In HIFP8 use cases, the storage type is torch.uint8 and x1_dtype is set to HIFLOAT8. | torch.float8_e4m3fn/torch.float8_e5m2/torch.int8; in MXFP4 and HIFP8 use cases, the actual storage type (such as torch.uint8), with the logical type specified by x1_dtype | 3D, (M, B, K) |
| x2 | Tensor | Required | Right matrix in the matrix multiplication, with the same data type and K-axis length as x1. The shape is (B, K, N) when perm_x2 is [0, 1, 2] and (B, N, K) when perm_x2 is [0, 2, 1]. In MXFP4 use cases, the last dimension is the physical length (N/2 or K/2) after FP4 packing. | Same as x1 | 3D |
| dtype | int | Required | Data type enumeration value of output y: 1 indicates torch.float16, 27 indicates torch.bfloat16, and 34 indicates HIFLOAT8 (supported only in T-C quantization; carried as torch.uint8 on the torch side). INT8 W8A8 supports only 27. | int64 | - |
| bias | Tensor | Optional | Bias added after the matrix multiplication. Reserved parameter, currently not supported. You must pass in None. | - | - |
| x1_scale | Tensor | Optional | Quantization scale factor of x1. Required in MX use cases; required in FP8 (K-C) use cases as the per-token scale; required in INT8 W8A8 use cases with shape (M, B); not used in HIFP8 and should be None. | MX: torch.float8_e8m0fnu; FP8 and INT8 W8A8: torch.float32 | MX: 4D, (M, B, K/64, 2); FP8: 1D, (M,); INT8 W8A8: 2D, (M, B) |
| x2_scale | Tensor | Optional | Quantization scale factor of x2. Required in MX use cases; required in FP8 (K-C) use cases as the per-channel scale; required in INT8 W8A8 use cases with shape (B, N); required in HIFP8 use cases as the per-channel FIXPIPE dequant scale. | MX: torch.float8_e8m0fnu; FP8 and INT8 W8A8: torch.float32; HIFP8: torch.uint64 or torch.int64 (interpreted as uint64 bits) | MX: 4D, (B, K/64, N, 2) when perm_x2 is [0, 1, 2] and (B, N, K/64, 2) when perm_x2 is [0, 2, 1]; FP8: 1D, (N,); INT8 W8A8: 2D, (B, N); HIFP8: 1D, (N,) |
| group_sizes | List[int] | Optional | Quantization group sizes [groupSizeM, groupSizeN, groupSizeK], each element in the range [0, 65535]. In MX use cases, groupSizeM and groupSizeN support only 0 or 1 (when the value is 0, the API infers it from the scale shape), and groupSizeK supports only 32; in FP8 (K-C) use cases, pass [0, 0, 0]; in HIFP8 use cases, pass [0, 0, 0]. | int64 | - |
| perm_x1 | List[int] | Optional | Transpose order of x1. Only [1, 0, 2] is supported. Default value: [1, 0, 2]. | int64 | - |
| perm_x2 | List[int] | Optional | Transpose order of x2. [0, 1, 2] and [0, 2, 1] are supported. Default value: [0, 1, 2]. FP8 (K-C) and INT8 W8A8 support only [0, 1, 2]. | int64 | - |
| perm_y | List[int] | Optional | Transpose order of the output matrix. Only [1, 0, 2] is supported. Default value: [1, 0, 2]. | int64 | - |
| batch_split_factor | int | Optional | Split size of dimension B of the output matrix. Currently only the value 1 is supported. Default value: 1. | int64 | - |
| x1_dtype | int | Optional | Data type enumeration value of x1. When not passed, it is auto-inferred according to the data type of x1. In MXFP4 use cases, this parameter must be passed in and set to torch_npu.float4_e2m1fn_x2; in HIFP8 use cases, this parameter must be passed in and set to torch_npu.hifloat8. | int64 | - |
| x2_dtype | int | Optional | Data type enumeration value of x2. When not passed, it is auto-inferred according to the data type of x2. In MXFP4 use cases, this parameter must be passed in and set to torch_npu.float4_e2m1fn_x2; in HIFP8 use cases, this parameter must be passed in and set to torch_npu.hifloat8. | int64 | - |
| x1_scale_dtype | int | Optional | Data type enumeration value of x1_scale. When not passed, it is auto-inferred according to the data type of x1_scale. | int64 | - |
| x2_scale_dtype | int | Optional | Data type enumeration value of x2_scale. When not passed, it is auto-inferred according to the data type of x2_scale. | int64 | - |


## Return Values

| Output | Output Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- |
| y | Tensor | Output of the quantized matrix multiplication. When dtype is 34, the output is HIFLOAT8 (T-C quantization only) and is carried as torch.uint8; the user must interpret it as HIFLOAT8. | torch.float16 when dtype is 1, torch.bfloat16 when dtype is 27, torch.uint8 when dtype is 34 | (M, B, N) |

## Constraints

- The API supports single-operator mode and TorchAir graph mode.
- In TorchAir graph mode, three quantization modes share the same GE Converter, which dispatches automatically based on the input/scale data types: MXFP8 (torch.float8_e4m3fn + torch.float8_e8m0fnu scale) and MXFP4 (torch.uint8 + x1_dtype/x2_dtype set to float4_e2m1fn_x2 + torch.float8_e8m0fnu scale) can coexist in the same graph; FP8 (K-C, torch.float32 scale) and HIFP8 (x1_dtype/x2_dtype set to hifloat8 + torch.uint64/int64 scale) are also supported by the same Converter. In graph mode, MXFP4 and HIFP8 still require x1_dtype/x2_dtype, with the same semantics as single-operator mode. Graph mode does not include INT8 W8A8 K-C; that mode is supported only in single-operator mode.
- x1 and x2 must be NPU tensors; x1_scale and x2_scale must be passed as required by the quantization mode (see "Mode-specific constraints" below).
- Only 3D tensors are supported: the shape of x1 is (M, B, K), the shape of x2 is (B, K, N) or (B, N, K), the batch and K dimensions of x1 and x2 must be the same, and batch dimension broadcasting is not supported.
- perm_x1 only supports [1, 0, 2]; perm_y only supports [1, 0, 2]; perm_x2 supports [0, 1, 2] and [0, 2, 1] (FP8 (K-C) supports only [0, 1, 2]).
- batch_split_factor currently supports only the value 1 (validated by Tiling in single-operator mode and by the GE Converter in graph mode).
- bias is a reserved parameter and is currently not supported.
- Empty tensors are not supported.
- When dtype is 34, the output is HIFLOAT8 (T-C quantization only), carried as torch.uint8 on the torch side; the user must interpret it as HIFLOAT8.

Mode-specific constraints:

- **MX**:
  - x1 and x2 must have the same data type: torch.float8_e4m3fn in MXFP8 use cases and torch.uint8 in MXFP4 use cases.
  - x1_scale and x2_scale must have the data type torch.float8_e8m0fnu.
  - K only supports multiples of 64.
  - The shape of x1_scale must be (M, B, K/64, 2); the shape of x2_scale must be (B, K/64, N, 2) when perm_x2 is [0, 1, 2] and (B, N, K/64, 2) when perm_x2 is [0, 2, 1]; the last dimension must be 2.
  - group_sizes must be explicitly passed in, groupSizeM and groupSizeN must be 0 or 1, and groupSizeK must be 32.
  - In MXFP4 use cases, data is stored with two FLOAT4_E2M1 values packed together (the last dimension of the tensor is the physical length, which is half of the logical length): the last dimension of x1 is K/2; the last dimension of x2 is N/2 when perm_x2 is [0, 1, 2] and K/2 when perm_x2 is [0, 2, 1]. In this case, the FP4 type cannot be automatically inferred from the actual storage type of the tensor (such as torch.uint8). Specify torch_npu.float4_e2m1fn_x2 through x1_dtype and x2_dtype.
  - Only x2 supports the FRACTAL_NZ format (MXFP8).
- **FP8 (K-C)**:
  - x1 and x2 support torch.float8_e4m3fn and torch.float8_e5m2 in any combination.
  - x1_scale must be torch.float32 with shape (M,); x2_scale must be torch.float32 with shape (N,).
  - K supports only 512, and N supports only 128.
  - group_sizes must be [0, 0, 0].
- **HIFP8**:
  - x1 and x2 are stored as torch.uint8 and must be specified as torch_npu.hifloat8 through x1_dtype and x2_dtype.
  - x2_scale must be torch.uint64 or torch.int64 with shape (N,); both are interpreted as 64-bit **bit patterns**, and int64 is not a numeric value. The **lower 32 bits contain the torch.float32 bit pattern** (1.0 is encoded as 0x000000003F800000), and the upper 32 bits are ignored. Do not pass a numeric integer (for example, do not pass 2 for scale=2); otherwise 2 is reinterpreted as a float32 denormal bit pattern, producing incorrect results without a warning.
  - x1_scale is None and does not participate in computation.
  - K and N have no special alignment requirements (they only need to be positive). perm_x2 supports [0, 1, 2] and [0, 2, 1].
  - x2 supports the FRACTAL_NZ format (the same as MXFP8; the view shape is unchanged, and only the physical layout differs).
  - dtype supports 1 (fp16), 27 (bf16), and 34 (HIFLOAT8). When dtype=34 (T-C), the output is HIFLOAT8, carried as torch.uint8 on the torch side; the user must interpret it as HIFLOAT8. Both ND and NZ x2 are supported.
  - group_sizes must be [0, 0, 0].
<!-- npu="A3,910b" id7 -->
- **INT8 W8A8 (K-C)**:
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term> are supported.
  - x1 and x2 are torch.int8.
  - x1_scale and x2_scale must be torch.float32. x1_scale has shape (M, B), and x2_scale has shape (B, N).
  - K is a multiple of 32, and N is a multiple of 16. On Atlas A2, B*K must be less than 65536.
  - perm_x2 supports only [0, 1, 2]. batch_split_factor is 1, and bias must be None.
  - dtype supports only 27 (torch.bfloat16). When x2 is FRACTAL_NZ, use the WeightNz API.
<!-- end id7 -->

## Deterministic Computation

Defaults to a deterministic computation.

## Example

- Single-operator mode call (eager)

  - MXFP8 use case example:

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    M, B, K, N = 64, 16, 128, 256
    x1 = torch.randn(M, B, K).to(torch.float8_e4m3fn).npu()
    x2 = torch.randn(B, K, N).to(torch.float8_e4m3fn).npu()
    x1_scale = torch.ones(M, B, K // 64, 2, dtype=torch.float8_e8m0fnu).npu()
    x2_scale = torch.ones(B, K // 64, N, 2, dtype=torch.float8_e8m0fnu).npu()

    y = cann_ops_nn.transpose_quant_batch_mat_mul(
        x1,
        x2,
        dtype=27,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        group_sizes=[1, 1, 32],
    )
    print(y.shape, y.dtype)
    ```

  - MXFP4 use case example:

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    M, B, K, N = 64, 16, 128, 256
    # FP4 is stored in a packed format of two FLOAT4_E2M1 values, and the last dimension of the tensor is the physical length (half of the logical length).
    x1 = torch.randint(0, 256, (M, B, K // 2), dtype=torch.uint8).npu()
    x2 = torch.randint(0, 256, (B, K, N // 2), dtype=torch.uint8).npu()
    x1_scale = torch.ones(M, B, K // 64, 2, dtype=torch.float8_e8m0fnu).npu()
    x2_scale = torch.ones(B, K // 64, N, 2, dtype=torch.float8_e8m0fnu).npu()

    y = cann_ops_nn.transpose_quant_batch_mat_mul(
        x1,
        x2,
        dtype=27,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        group_sizes=[1, 1, 32],
        x1_dtype=torch_npu.float4_e2m1fn_x2,
        x2_dtype=torch_npu.float4_e2m1fn_x2,
    )
    print(y.shape, y.dtype)
    ```
