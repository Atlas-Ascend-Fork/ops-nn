# add_rms_norm_dynamic_quant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T19:47:34.945Z -->

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

    Fuses Add, RMS Normalization, and MX dynamic quantization. Wraps aclnnAddRmsNormDynamicMxQuantV2 (the V2 API is always called internally; when x3 is not passed, the behavior is consistent with aclnnAddRmsNormDynamicMxQuant). Adds inputs x1 and x2 (with optional x3, supporting three-way residual addition), applies RmsNorm normalization (with an optional beta bias), and then performs dynamic MX quantization in groups of blocksize=32 along the trailing axis of the input. Outputs the quantized result y, the addition result x, the block quantization scale mxscale, and the reciprocal of the standard deviation rstd.

- Formula:

  When x3 is not passed:

  $$
  x=x_{1}+x_{2}
  $$

  When x3 is passed in (accumulated in the order shown in parentheses, matching residual addition semantics):

  $$
  x=(x_{3}+x_{1})+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  When scale_alg is 0 (OCP implementation):

  - Group the RmsNorm output y along the trailing axis into groups of k = 32 numbers. Each group of k numbers $\{V_i\}_{i=1}^{k}$ is dynamically quantized to $\{mxscale,\{P_i\}_{i=1}^{k}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \quad i = 1, 2, \ldots, k
    $$

  - emax: Exponent bits of the maximum normal number for the corresponding data type.

    | DataType | Emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

  When scale_alg is 1 (cuBLAS implementation), only FP8 types are involved:

  - Divide the long vector into blocks, each of length k. Compute a block scaling factor $S_{fp32}^b$ for each block independently, and then map all elements in the block to the target low-precision type FP8 using the same $S_{fp32}^b$.
  - Find the maximum absolute value of the values in the block:

    $$
    Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
    $$

  - Map FP32 to the range representable by the target data type FP8:

    $$
    S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
    $$

  - Convert to the scaling value $S_{ue8m0}^b$ representable in the FP8 format.
  - Extract the unbiased exponent $E_{int}^b$ and mantissa $M_{fixp}^b$ from the floating-point scaling factor $S_{fp32}^b$ of the block.
  - To prevent overflow during quantization, round up the exponent:

    $$
    E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a normal number, and} E_{int}^b < 254 \text{ and} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{ if} S_{fp32}^b \text{ is a subnormal number, and} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{ otherwise} \end{cases}
    $$

  - Calculate the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
  - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
  - Apply the final step to quantization: $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$

## Function Prototype

```python
cann_ops_nn.add_rms_norm_dynamic_quant(x1, x2, gamma, beta=None, x3=None, epsilon=1e-6, scale_alg=0, round_mode="rint", dst_type=40, output_rstd=False)
    -> (Tensor y, Tensor x, Tensor mxscale, Tensor rstd)
```

## Mandatory

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Parameter type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x1</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Source data tensor in the normalization process, x1 in the formula.</td>
        <td>float16, bfloat16</td>
        <td>(..., D)</td>
    </tr>
    <tr>
        <td>x2</td>
        <td>Tensor</td>
        <td>required</td>
        <td>Source data tensor in the normalization process, x2 in the formula. The shape and data type must be consistent with x1.</td>
        <td>float16, bfloat16</td>
        <td>(..., D)</td>
    </tr>
    <tr>
        <td>gamma</td>
        <td>Tensor</td>
        <td>required</td>
        <td>Weight tensor in the normalization process, gamma in the formula. The shape must be consistent with the last dimension of x1.</td>
        <td>float16, bfloat16, float32</td>
        <td>(D,)</td>
    </tr>
    <tr>
        <td>beta</td>
        <td>Tensor</td>
        <td>optional</td>
        <td>Bias term in the normalization process, beta in the formula. The shape and data type must be consistent with gamma.</td>
        <td>float16, bfloat16, float32</td>
        <td>(D,)</td>
    </tr>
    <tr>
        <td>x3</td>
        <td>Tensor</td>
        <td>optional</td>
        <td>Optional input of the three-way residual addition, which is x3 in the formula. The shape and data type must be consistent with x1; when not passed, x = x1 + x2 is calculated.</td>
        <td>float16, bfloat16</td>
        <td>(..., D)</td>
    </tr>
    <tr>
        <td>epsilon</td>
        <td>float</td>
        <td>optional</td>
        <td>Epsilon in the formula, used to prevent division-by-zero errors. Default value: 1e-6.</td>
        <td>float32</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scale_alg</td>
        <td>int</td>
        <td>optional</td>
        <td>Calculation method of mxscale, which is scale_alg in the formula. Value 0 indicates the Open Compute Project (OCP) implementation, and value 1 indicates the cuBLAS implementation (only supported for FP8 output). Only value 0 is supported when dst_type is 40/41 (FP4). Default value: 0.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>round_mode</td>
        <td>str</td>
        <td>optional</td>
        <td>Data conversion mode, round_mode in the formula. Only "rint" is supported when dst_type is 35/36 (FP8); "rint", "floor", and "round" are supported when dst_type is 40/41 (FP4). Default value: "rint".</td>
        <td>str</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type</td>
        <td>int</td>
        <td>optional</td>
        <td>Data type enumeration value of output y, DType in the formula. Value range: {35, 36, 40, 41}, corresponding to FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, and FLOAT4_E1M2 respectively. Default value: 40.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output_rstd</td>
        <td>bool</td>
        <td>Optional</td>
        <td>Specifies whether to output valid rstd. If False, rstd is an invalid placeholder output (content undefined). Default value: False.</td>
        <td>bool</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## Return Values

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Parameter Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>y</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Normalized and quantized result, corresponding to Pi and di in the formula. When dst_type is 40/41 (FP4), dtype is uint8 and two FP4 values are packed into one byte (4 bits for each FP4 value); the trailing axis of the shape is half of the trailing axis of x1. When dst_type is 35/36 (FP8), dtype is float8_e5m2 or float8_e4m3fn respectively, and the shape is consistent with x1.</td>
        <td>uint8 (FP4), float8_e5m2, and float8_e4m3fn</td>
        <td>FP4: (..., D/2); FP8: (..., D)</td>
    </tr>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Addition result of x1+x2 (+x3), corresponding to x in the formula. The shape and data type are consistent with those of x1.</td>
        <td>float16 and bfloat16</td>
        <td>(..., D)</td>
    </tr>
    <tr>
        <td>mxscale</td>
        <td>Tensor</td>
        <td>required</td>
        <td>Quantization scale corresponding to each group (blocksize=32), which is mxscale and Sb in the formula, encoded in E8M0. In single-operator mode, the data type is float8_e8m0fnu; in graph mode (torchair), the data type is uint8. Both share the same byte layout and can be converted to each other via view(torch.uint8). The shape is x1.shape[:-1] concatenated with (ceil(ceil(D/32)/2), 2), that is, the rank is 1 greater than that of x1, the second-to-last dimension is half of the number of blocks rounded up to an even number, and the last dimension is 2.</td>
        <td>float8_e8m0fnu, uint8</td>
        <td>(..., ceil(ceil(D/32)/2), 2)</td>
    </tr>
    <tr>
        <td>rstd</td>
        <td>Tensor</td>
        <td>required</td>
        <td>Reciprocal of the standard deviation after normalization, which is the reciprocal of Rms(x) in the formula. Outputs valid values when output_rstd is True; outputs an invalid placeholder (content undefined) when output_rstd is False. The shape is x1.shape[:-1] concatenated with (1,).</td>
        <td>float32</td>
        <td>(..., 1)</td>
    </tr>
</tbody>
</table>

## Constraints

- The API supports single-operator mode and TorchAir graph mode calls.
- x1 supports 1 to 7 dimensions; the shape and data type of x2 must be consistent with those of x1; the shape and data type of x3 (if passed) must be consistent with those of x1.
- gamma must be a 1D tensor, with its length equal to the last dimension of x1 and its data type consistent with that of x1 or float32; the shape and data type of beta (if passed) must be consistent with those of gamma.
- When dst_type is 40/41 (FP4), the trailing axis of x1 must be even, and scale_alg supports only 0.
- When scale_alg is 1 (cuBLAS implementation), only FP8 output is supported.
- Strict validation of parameter types (eager direct call, i.e., when calling `cann_ops_nn.ops.add_rms_norm_dynamic_quant` or importing this function directly): x1/x2/gamma must be `torch.Tensor`, and beta/x3 must be `torch.Tensor` or `None`; among the scalar parameters, dst_type and scale_alg accept only native Python int (bool is rejected; `enum.IntEnum` is accepted because it is a subclass of int), round_mode accepts only exact str, output_rstd accepts only exact bool, and epsilon accepts only exact int/float. Third-party types such as torch.dtype and numpy scalars (including subclasses of Python built-in types such as `np.float64` and `np.str_`) throw a TypeError that names the parameter directly at the entry (for example, `dst_type=torch.float8_e5m2` is rejected), without implicit conversion or normalization.
- In graph mode (torch.compile) or when called through `torch.ops.cann_ops_nn.add_rms_norm_dynamic_quant`, scalar parameters are first type-converted by the torch dispatcher according to the schema: torch.dtype is converted by the framework to its internal enum ordinal (for example, `torch.float8_e5m2 -> 23` and `torch.int4 -> 40`). The entry type interception does not take effect on this path; dst_type value range validation serves as the fallback. To avoid ambiguity, explicitly pass int enum values (35/36/40/41).
- mxscale output shape constraints: rank(mxscale) = rank(x1) + 1, mxscale.shape[-2] = ceil(ceil(x1.shape[-1] / 32) / 2), mxscale.shape[-1] = 2, and other dimensions are consistent with those of x1.

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Single-operator mode call (eager)

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    rows, hidden = 4, 64
    x1 = torch.randn(rows, hidden, dtype=torch.float16).npu()
    x2 = torch.randn(rows, hidden, dtype=torch.float16).npu()
    gamma = torch.ones(hidden, dtype=torch.float16).npu()

    # Without optional inputs, outputs FP8_E4M3FN quantized result
    # y, x, mxscale, rstd = cann_ops_nn.add_rms_norm_dynamic_quant(x1, x2, gamma, dst_type=36)

    # With optional inputs beta, x3 (three-way residual addition), and rstd output
    beta = torch.zeros(hidden, dtype=torch.float16).npu()
    x3 = torch.randn(rows, hidden, dtype=torch.float16).npu()
    y, x, mxscale, rstd = cann_ops_nn.add_rms_norm_dynamic_quant(
        x1, x2, gamma, beta=beta, x3=x3, epsilon=1e-6,
        scale_alg=0, round_mode="rint", dst_type=36, output_rstd=True)

    # FP4 output (dst_type=40/41): y is uint8, two FP4 values packed into one byte, trailing axis halved
    # y, x, mxscale, rstd = cann_ops_nn.add_rms_norm_dynamic_quant(
    #     x1, x2, gamma, dst_type=40, round_mode="rint")

    # mxscale is float8_e8m0fnu in single-operator mode and uint8 in graph mode; to handle it uniformly as uint8:
    # mxscale_u8 = mxscale.view(torch.uint8)

    print("y: ", y)
    print("x: ", x)
    print("mxscale: ", mxscale)
    print("rstd: ", rstd)
    ```

- Graph mode (torchair) call

    ```python
    import torch, torch_npu, cann_ops_nn

    @torch.compile(backend="npu")
    def func(x1, x2, gamma):
        return cann_ops_nn.add_rms_norm_dynamic_quant(x1, x2, gamma, dst_type=36)

    x1 = torch.randn(4, 64, dtype=torch.float16).npu()
    x2 = torch.randn(4, 64, dtype=torch.float16).npu()
    gamma = torch.ones(64, dtype=torch.float16).npu()
    y, x, mxscale, rstd = func(x1, x2, gamma)
    print("y: ", y)
    print("mxscale: ", mxscale)
    ```
