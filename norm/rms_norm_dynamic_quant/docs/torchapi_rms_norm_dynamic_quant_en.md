# rms_norm_dynamic_quant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:31:45.413Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

    Fuses RMS Normalization with symmetric dynamic quantization (INT8), encapsulating aclnnRmsNormDynamicQuant. The input x is first normalized by RMSNorm (optionally scaled by smooth_scales and offset by beta), and then dynamically quantized to INT8 row by row. The output is the INT8 quantization result and the FLOAT32 scale factor.

- Formula:

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma + beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - If smooth_scales is not provided, the rmsnorm output is quantized directly:

  $$
   scaleOut=row\_max(abs(y))/max\_val
  $$

  $$
   yOut=round(y/scaleOut)
  $$

  - If smooth_scales is input, apply smooth scaling before quantization:

  $$
    input = y\cdot smooth\_scales
  $$

  $$
   scaleOut=row\_max(abs(input))/max\_val
  $$

  $$
   yOut=round(input/scaleOut)
  $$

  where row\_max denotes the maximum value of each row. max\_val is 127 for INT8.

## Function Prototype

```python
cann_ops_nn.rms_norm_dynamic_quant(x, gamma, smooth_scales=None, beta=None, *, epsilon=1e-6, dst_type=2)
    -> (Tensor y, Tensor scale)
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
        <td>Source data tensor to be normalized, which is x in the formula.</td>
        <td>float16, bfloat16</td>
        <td>(..., D)</td>
    </tr>
    <tr>
        <td>gamma</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Weight tensor in the normalization process, gamma in the formula. The shape must be consistent with the last dimension of x.</td>
        <td>float16, bfloat16</td>
        <td>(D,)</td>
    </tr>
    <tr>
        <td>smooth_scales</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>smoothScale tensor used in quantization, smooth_scales in the formula. The shape and dtype must be consistent with gamma.</td>
        <td>float16, bfloat16</td>
        <td>(D,)</td>
    </tr>
    <tr>
        <td>beta</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Bias term added during normalization, which is beta in the formula. Its shape and dtype must be consistent with gamma.</td>
        <td>float16, bfloat16</td>
        <td>(D,)</td>
    </tr>
    <tr>
        <td>epsilon</td>
        <td>float</td>
        <td>Optional</td>
        <td>Used to prevent division by zero errors. epsilon in the formula, must be greater than zero. Default value: 1e-6.</td>
        <td>float32</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type</td>
        <td>int</td>
        <td>Optional</td>
        <td>Enumeration value of the data type of output y. Currently, only INT8 quantization is supported. Default value: 2.</td>
        <td>int</td>
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
    <th>Type</th>
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
        <td>Quantization output, corresponding to yOut in the formula, with a shape consistent with x.</td>
        <td>int8</td>
        <td>(..., D)</td>
    </tr>
    <tr>
        <td>scale</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Quantization scale factor, corresponding to scaleOut in the formula, with a shape of x excluding the last dimension.</td>
        <td>float32</td>
        <td>(...,)</td>
    </tr>
</tbody>
</table>

## Constraints

- This API can be used in training and inference tasks.
- This API can be called in single-operator mode and TorchAir graph mode.
- The last dimension of input `x` must be less than or equal to 8192. Otherwise, precision issues may occur.
- `x` does not support all-zero input.

## Deterministic Computation

Defaults to a deterministic computation.

## Example

- Call in single-operator mode (eager)

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    rows, hidden = 16, 64
    x = torch.randn(rows, hidden, dtype=torch.float16).npu()
    gamma = torch.ones(hidden, dtype=torch.float16).npu()
    # Without optional inputs
    # y, scale = cann_ops_nn.rms_norm_dynamic_quant(x, gamma)

    smooth = torch.ones(hidden, dtype=torch.float16).npu()
    beta = torch.zeros(hidden, dtype=torch.float16).npu()
    # With optional inputs
    y, scale = cann_ops_nn.rms_norm_dynamic_quant(
        x, gamma, smooth_scales=smooth, beta=beta, epsilon=1e-5)

    print("y: ", y)
    print("scale: ", scale)
    ```

- Call in graph mode (torchair)

    ```python
    import torch, torch_npu, cann_ops_nn

    @torch.compile(backend="npu")
    def func(x, gamma):
        return cann_ops_nn.rms_norm_dynamic_quant(x, gamma)

    x = torch.randn(4, 64, dtype=torch.float16).npu()
    gamma = torch.ones(64, dtype=torch.float16).npu()
    y, scale = func(x, gamma)
    print("y: ", y)
    print("scale: ", scale)
    ```
