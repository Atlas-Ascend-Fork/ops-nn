# grouped_dynamic_mx_quant_with_dual_axis

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:34:54.924Z -->

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

    Performs dual-axis MX quantization based on x and group_index in one pass, and outputs y1, mxscale1, y2, and mxscale2. y1 and mxscale1 are the quantization results along the last dimension, and y2 and mxscale2 are the quantization results along the second-to-last dimension.

- Calculation description:

    Quantization is performed with 32 elements as one MX block. When scale_alg=1, a scale of the FLOAT8_E8M0 type is generated for each block, and the reciprocal of the scale is used to convert the elements in the block to the data type specified by dst_type.

- Formula:

  For the input data in an MX block:

  $$
  D^{b}=[d^{i}]_{i=1}^{k},\quad k=32
  $$

  Calculate the maximum absolute value of the elements in the block:

  $$
  Amax(D^{b})=\max_i(|d^{i}|)
  $$

  When scale_alg=1, calculate the FP32 scale based on the maximum representable value of the target FP8 type:

  $$
  S_{fp32}^{b}=\frac{Amax(D^{b})}{Amax(dst_type)}
  $$

  Extract the unbiased exponent $E_{int}^{b}$ and the mantissa $M_{fixp}^{b}$ from the block scale factor $S_{fp32}^{b}$.

  To ensure no overflow during quantization, round up the exponent and keep it within the representable range of FP8:

  $$
  E_{int}^{b} = \begin{cases}
    E_{int}^{b} + 1, & \text{ if} S_{fp32}^{b} \text{ is a normal number and} E_{int}^{b} < 254 \text{ and} M_{fixp}^{b} > 0 \\
    E_{int}^{b} + 1, & \text{ if} S_{fp32}^{b} \text{ is a subnormal number and} M_{fixp}^{b} > 0.5 \\
    E_{int}^{b}, & \text{ otherwise}
  \end{cases}
  $$

  Calculate the block scale factor:

  $$
  S_{ue8m0}^{b}=2^{E_{int}^{b}}
  $$

  Compute the block conversion factor:

  $$
  R_{fp32}^{b}=\frac{1}{fp32(S_{ue8m0}^{b})}
  $$

  Perform quantization on each element in the block:

  $$
  p_i=cast\_to\_dst\_type(d^{i}\times R_{fp32}^{b}, round_mode)
  $$

  Output the scale and quantization result corresponding to the block:

  $$
  (S_{ue8m0}^{b},[p_i]_{i=1}^{k})
  $$

## Function Prototype

```python
cann_ops_nn.grouped_dynamic_mx_quant_with_dual_axis(
    x, group_index, *, round_mode="rint", scale_alg=1, dst_type=24, dst_type_max=0)
    -> (Tensor y1, Tensor mxscale1, Tensor y2, Tensor mxscale2)
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
    <th>Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Input tensor to be quantized, corresponding to input data D in the formula.</td>
        <td>torch.float16, torch.bfloat16</td>
        <td>(M, N)</td>
    </tr>
    <tr>
        <td>group_index</td>
        <td>Tensor</td>
        <td>required</td>
        <td>Quantization group index, describing group boundaries in cumsum form. G is the number of groups; each element value is non-negative and non-decreasing, and the last element must equal M.</td>
        <td>torch.int64</td>
        <td>(G,)</td>
    </tr>
    <tr>
        <td>round_mode</td>
        <td>str</td>
        <td>optional</td>
        <td>Rounding mode used to convert quantization results to the target output type. Only "rint" is supported; default value is "rint".</td>
        <td>str</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scale_alg</td>
        <td>int</td>
        <td>optional</td>
        <td>Algorithm for scale computation. Only 1 is supported. Default value is 1.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type</td>
        <td>int</td>
        <td>optional</td>
        <td>Target output data type of y1 and y2. 23 indicates torch.float8_e5m2, 24 indicates torch.float8_e4m3fn. Default value is 24.</td>
        <td>int</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type_max</td>
        <td>float</td>
        <td>optional</td>
        <td>Reserved parameter. Only 0.0 is supported. The default value is 0.0.</td>
        <td>float</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## Returns

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
    <th>Output</th>
    <th>Output Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (Shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>y1</td>
        <td>Tensor</td>
        <td>required</td>
        <td>Quantization result of <code>x</code> along the -1 axis. The shape is consistent with that of <code>x</code>.</td>
        <td>torch.float8_e4m3fn, torch.float8_e5m2</td>
        <td>(M, N)</td>
    </tr>
    <tr>
        <td>mxscale1</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Quantization scale corresponding to each group along the -1 axis. Every two scales in the last dimension are stored in pairs. One scale is generated for every 32 elements along the N dimension. Even padding is applied, with the padding value set to 0.</td>
        <td>torch.uint8 (FLOAT8_E8M0)</td>
        <td>(M, ceil(N/64), 2)</td>
    </tr>
    <tr>
        <td>y2</td>
        <td>Tensor</td>
        <td>required</td>
        <td>Quantization result of <code>x</code> along the -2 axis. The shape is the same as <code>x</code>.</td>
        <td>torch.float8_e4m3fn, torch.float8_e5m2</td>
        <td>(M, N)</td>
    </tr>
    <tr>
        <td>mxscale2</td>
        <td>Tensor</td>
        <td>required</td>
        <td>Quantization scale corresponding to each group along the -2 axis. One scale is generated for every 32 elements along the second-to-last dimension, with even padding. The output requires interleaving for every two rows of data.</td>
        <td>torch.uint8 (FLOAT8_E8M0)</td>
        <td>(floor(M/64)+G, N, 2)</td>
    </tr>
</tbody>
</table>

## Constraints

- This API is only supported for training use cases and is not supported for inference use cases.
- This API is only supported for single-operator mode calls and is not supported for TorchAir graph mode calls.
- Data type constraints:
  - `x` supports only torch.float16 and torch.bfloat16.
  - `group_index` supports only torch.int64.
  - The data types of `y1` and `y2` are specified by `dst_type`. Currently, torch.float8_e4m3fn (24) and torch.float8_e5m2 (23) are supported.
  - The data types of `mxscale1` and `mxscale2` are torch.uint8 (FLOAT8_E8M0).
- Shape constraints:
  - `x` must be 2-dimensional. M (dimension 0) can be 0, and N must be greater than 0 and divisible by 64 (N % 64 == 0).
  - Empty tensors are not supported for `group_index`. It must be 1-dimensional, each element value must be non-negative and non-decreasing, and the last element must equal M; when M=0, the last element is 0.
  - The shapes of `y1` and `y2` must be consistent with that of `x`, both (M, N).
  - The shape of `mxscale1` is (M, ceil(N/64), 2).
  - The shape of `mxscale2` is (floor(M/64)+G, N, 2), where G is the number of elements in group_index.
- The 0th dimension M of x can be 0; N and the length of group_index must still be greater than 0.

## Deterministic Computation

Defaults to a deterministic computation.

## Calling Instructions

- Single-operator mode call (eager):

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    M = 256
    N = 1024

    # Construct the NPU inputs.
    x = torch.randn(M, N, dtype=torch.bfloat16).npu()
    group_index = torch.tensor([64, 128, 192, 256], dtype=torch.int64).npu()

    # Call the dual-axis MX quantization API.
    y1, mxscale1, y2, mxscale2 = (
        torch.ops.cann_ops_nn.grouped_dynamic_mx_quant_with_dual_axis(
            x,
            group_index,
            round_mode="rint",
            scale_alg=1,
            dst_type=torch.float8_e4m3fn,
            dst_type_max=0.0,
        )
    )
    # View the shapes of the four outputs.
    print("y1 shape: ", y1.shape)
    print("mxscale1 shape: ", mxscale1.shape)
    print("y2 shape: ", y2.shape)
    print("mxscale2 shape: ", mxscale2.shape)
    ```
