# flat_quant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:01:30.493Z -->

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

  Performs two Kronecker product small-matrix multiplications on matrix x in sequence, and then quantizes the multiplication result. The underlying implementation calls `aclnnFlatQuantV3`.

- Formula:

  $$
  \text{intermediate} = x \cdot \text{kronecker\_p2}
  $$

  $$
  \text{result} = \text{kronecker\_p1} \cdot \text{intermediate}
  $$

  $$
  \text{scale} = \text{compute\_quant\_scale}(\text{result}, \text{clip\_ratio}, \text{dst\_type\_max})
  $$

  $$
  \text{out} = \text{quantize}(\text{result}, \text{scale})
  $$

## Function Prototype

```python
cann_ops_nn.flat_quant(x, kronecker_p1, kronecker_p2, clip_ratio=1.0, dst_dtype=torch.quint4x2, dst_type_max=0.0, group_list=None, group_list_type=0)
    -> (Tensor out, Tensor quant_scale)
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
    <th>Parameter Type</th>
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
        <td>required</td>
        <td>Input matrix, the data tensor for Kronecker product multiplication and quantization. N2 must be a multiple of 8 (when dst_dtype = torch.quint4x2) or an even number (when dst_dtype = torch_npu.float4_e2m1fn_x2).</td>
        <td>float16, bfloat16</td>
        <td>(M, N1, N2)</td>
    </tr>
    <tr>
        <td>kronecker_p1</td>
        <td>Tensor</td>
        <td>required</td>
        <td>First Kronecker product matrix, whose N1 dimension is the same as that of x.</td>
        <td>float16, bfloat16</td>
        <td>(N1, N1)</td>
    </tr>
    <tr>
        <td>kronecker_p2</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>The second Kronecker product matrix. Its N2 dimension is the same as the N2 dimension of x.</td>
        <td>float16, bfloat16</td>
        <td>(N2, N2)</td>
    </tr>
    <tr>
        <td>clip_ratio</td>
        <td>float</td>
        <td>Optional</td>
        <td>Quantization clipping ratio, used to control the clipping ratio in quantization. Range: (0, 1]. Default value: 1.0.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_dtype</td>
        <td>int</td>
        <td>optional</td>
        <td>Specifies the output data type. Supported values are torch.quint4x2 (default) and torch_npu.float4_e2m1fn_x2.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type_max</td>
        <td>float</td>
        <td>Optional</td>
        <td>Maximum value of the target data type. The input value can only be 0 or a value in the range [6, 12]. Default value: 0.0.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group_list</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Grouping list used for grouped quantization. Supports 1D or 2D input. Default value: None.</td>
        <td>int64</td>
        <td>(G,) or (G, 2)</td>
    </tr>
    <tr>
        <td>group_list_type</td>
        <td>int</td>
        <td>optional</td>
        <td>Grouping mode of the group_list input. Value range: 0-2. Default value: 0.</td>
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
    <th>Output</th>
    <th>Output Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>out</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Quantized output. If dst_dtype = torch.quint4x2, the data type is int32 and the shape is (M, N1, N2//8); if dst_dtype = torch_npu.float4_e2m1fn_x2, the data type is uint8 and the shape is (M, N1 * N2//2).</td>
        <td>int32, uint8</td>
        <td>(M, N1, N2//8) or (M, N1 * N2//2)</td>
    </tr>
    <tr>
        <td>quant_scale</td>
        <td>Tensor</td>
        <td>required</td>
        <td>Quantization scale factor. If dst_dtype = torch.quint4x2, the data type is float32 and the shape is (M,); if dst_dtype = torch_npu.float4_e2m1fn_x2, the data type is uint8 and the shape is (M, ceildiv(N1*N2,64), 2).</td>
        <td>float32, uint8</td>
        <td>(M,) or (M, ceildiv(N1*N2,64), 2)</td>
    </tr>
</tbody>
</table>

## Constraints

- This API can be used in both training and inference use cases.
- This API supports calls in single-operator mode and TorchAir graph mode.
- The N2 dimension of input x must meet the following conditions:
  - If dst_dtype = torch.quint4x2, N2 must be a multiple of 8.
  - If dst_dtype = torch_npu.float4_e2m1fn_x2, N2 must be an even number.
- The data types of inputs kronecker_p1 and kronecker_p2 must be the same as that of x.
- dst_dtype supports torch.quint4x2 (default) and torch_npu.float4_e2m1fn_x2. The outputs are described as follows:
  - If dtype is torch.quint4x2, output out is of type int32, formed by concatenating 8 int4 values. Unpack the values manually to view them. Output quant_scale is of type float32.
  - If dtype is torch_npu.float4_e2m1fn_x2, output out is of type uint8, formed by concatenating 2 float4_e2m1fn_x2 values. Unpack the values manually to view them. Output quant_scale is of type uint8. Convert it to float8_e8m0fnu manually to view the actual values.
- The value range of group_list_type is 0-2. If group_list_type is 0 or 1, the shape of group_list is (G,); if group_list_type is 2, the shape of group_list is (G, 2). G indicates the number of groups and must be less than or equal to 1024.
- group_list must meet the following conditions:
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: group_list supports only None input.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
    - If group_list is not None, the values of the group_list input must meet the following conditions. Otherwise, the output cannot be guaranteed to meet expectations:
      - When group_list_type is 0, group_list must be a non-negative monotonically non-decreasing sequence, representing the cumsum result (cumulative sum) of the size of each group after grouping. The last value must be less than or equal to the first dimension of the tensor in x.
      - When group_list_type is 1, group_list must be a non-negative sequence, representing the size of each group after grouping. The sum of the values must be less than or equal to the first dimension of the tensor in x.
      - When group_list_type is 2, group_list must be a non-negative sequence arranged as [[groupIdx0, groupSize0], [groupIdx1, groupSize1]...], where groupSize is the size of each group after grouping. The sum of the values in the second column must be less than or equal to the first dimension of the tensor in x.
  <!-- end id8 -->

## Deterministic Computation

Defaults to a deterministic computation.

## API call

- Single-operator mode call (eager)

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  M, N1, N2 = 128, 16, 16
  x = torch.randn(M, N1, N2, dtype=torch.float16).npu()
  kronecker_p1 = torch.randn(N1, N1, dtype=torch.float16).npu()
  kronecker_p2 = torch.randn(N2, N2, dtype=torch.float16).npu()

  # int32 output (default)
  out, quant_scale = cann_ops_nn.flat_quant(x, kronecker_p1, kronecker_p2)
  print("out: ", out)
  print("quant_scale: ", quant_scale)
  ```

  With optional inputs:

  ```python
  group_list = torch.tensor([0, 16, 32, 48, 64], dtype=torch.int64).npu()
  out, quant_scale = cann_ops_nn.flat_quant(
      x, kronecker_p1, kronecker_p2,
      group_list=group_list,
      clip_ratio=0.95,
      group_list_type=0
  )
  ```

- Graph mode (torchair) call

  ```python
  import torch, torch_npu, cann_ops_nn

  @torch.compile(backend="npu")
  def func(x, p1, p2):
      return cann_ops_nn.flat_quant(x, p1, p2)

  M, N1, N2 = 128, 16, 16
  x = torch.randn(M, N1, N2, dtype=torch.float16).npu()
  kronecker_p1 = torch.randn(N1, N1, dtype=torch.float16).npu()
  kronecker_p2 = torch.randn(N2, N2, dtype=torch.float16).npu()
  out, quant_scale = func(x, kronecker_p1, kronecker_p2)
  print("out: ", out)
  print("quant_scale: ", quant_scale)
  ```
