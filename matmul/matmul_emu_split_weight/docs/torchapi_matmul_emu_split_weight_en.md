# matmul_emu_split_weight

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:20:20.682Z -->

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

- Description:

  Uses dual-path BF16 GEMM fusion to emulate FP32 precision matrix multiplication. The FP32 weights are split offline into high-order BF16 and low-order residual BF16. During inference, two BF16 GEMMs are executed and a linear combination is performed. Activations remain BF16 throughout. Both matrix multiplications run on the Cube. The underlying API is `aclnnMatmulEmuSplitWeight`.

- Formula:

  $$
  result = X \times W_{high} + scale \times (X \times W_{low})
  $$

  Where $X$ is the BF16 activation matrix, $W_{high}$ is the high-order weight obtained by truncating the FP32 weight to BF16, $W_{low}$ is the low-order residual weight obtained by dividing the FP32 weight residual by scale and truncating it to BF16, and $scale$ is the scaling factor (typically 1/256=0.00390625).

- Offline weight splitting formula:

  $$
  W_{high} = \text{BF16}(W_{fp32})
  $$

  $$
  W_{low} = \text{BF16}\left(\frac{W_{fp32} - W_{high}.float()}{scale}\right)
  $$

## Function Prototype

```python
cann_ops_nn.matmul_emu_split_weight(x, w_high, w_low, *, w_low_scale=0.00390625, y_dtype=0)
    -> Tensor
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
    <th>Shape (Dimension)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Activation matrix, corresponding to X in the formula. The K dimension must be consistent with the K dimension of w_high/w_low.</td>
        <td>bfloat16</td>
        <td>(M, K)</td>
    </tr>
    <tr>
        <td>w_high</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>High-order weight, corresponding to W_high in the formula. Obtained by truncating the FP32 weight to BF16.</td>
        <td>bfloat16</td>
        <td>(K, N)</td>
    </tr>
    <tr>
        <td>w_low</td>
        <td>Tensor</td>
        <td>required</td>
        <td>Low-order residual weight, corresponding to W_low in the formula. Obtained by dividing the FP32 weight residual by scale and then truncating to BF16. The shape must be exactly the same as w_high.</td>
        <td>bfloat16</td>
        <td>(K, N)</td>
    </tr>
    <tr>
        <td>w_low_scale</td>
        <td>float</td>
        <td>optional</td>
        <td>Scale factor, corresponding to scale in the formula. Only 1/256 (0.00390625) is supported. Defaults to 0.00390625.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>y_dtype</td>
        <td>int</td>
        <td>optional</td>
        <td>Output data type identifier. Only 0 (FP32) is supported. Defaults to 0.</td>
        <td>-</td>
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
    <th>Output Name</th>
    <th>Output Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>result</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Computed output matrix, corresponding to result in the formula. Matrix multiplication result of FP32 precision emulated by dual-path BF16 GEMM fusion.</td>
        <td>float32</td>
        <td>(M, N)</td>
    </tr>
</tbody>
</table>

## Constraints

- This API supports use in inference scenarios.
- x, w_high, and w_low must be of bfloat16 data type. Only ND data format is supported.
- x, w_high, w_low, and y must be 2-dimensional.
- The shape of x is (M, K), the shape of w_high is (K, N), and the shape of w_low must be exactly the same as w_high.
- The K dimension of x must match the K dimension of w_high/w_low.
- M, K, and N are in the range (0, INT32_MAX].
- Only 1/256 (0.00390625) is supported for w_low_scale; other values cause an error.
- Only 0 (FP32) is supported for y_dtype; other values cause an error.
- This API does not accept a transpose attribute. If transposition is required, permute the tensor before passing it in. The operator supports non-contiguous tensors, so no extra contiguous operation is needed.

## Deterministic Computation

Defaults to a deterministic computation.

## Example

- Single-operator mode call (eager)

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  M, K, N = 128, 256, 128
  x = torch.randn(M, K, dtype=torch.bfloat16).npu()
  w_high = torch.randn(K, N, dtype=torch.bfloat16).npu()
  w_low = torch.randn(K, N, dtype=torch.bfloat16).npu()

  result = cann_ops_nn.matmul_emu_split_weight(x, w_high, w_low)
  print("result: ", result)
  ```
