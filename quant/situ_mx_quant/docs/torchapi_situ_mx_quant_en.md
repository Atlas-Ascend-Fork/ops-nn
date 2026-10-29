# situ_mx_quant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:38:25.561Z -->

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

- Description: Fuses the Situ activation function with dynamic MX (Microscaling) quantization into one operator. The Situ activation is applied to the input x, and then the activation result is quantized with MX quantization to output the quantized result and scale. The bottom layer encapsulates aclnnSituMxQuant.

- Formula:

  1. Situ activation:

    $$
    situ_a = \beta \times \tanh(gate / \beta) \times sigmoid(gate)
    $$

    When linear_beta > 0:

    $$
    up = linear\_beta \times \tanh(up / linear\_beta)
    $$

    $$
    situOut = situ_a \times up
    $$

    When activate_left is true, gate takes the first half of x and up takes the second half; when activate_left is false, gate takes the second half of x and up takes the first half.
  2. MX quantization (OCP algorithm):

    $$
    shared\_exp = floor(log2(max(|situOut_i|))) - emax
    $$

    $$
    y\_scale = 2^{shared\_exp}  (E8M0)
    $$

    $$
    y = cast\_to\_fp8(situOut / y\_scale)
    $$

## Function Prototype

```python
cann_ops_nn.situ_mx_quant(
    x,
    *,
    beta=1.0,
    linear_beta=0.0,
    activate_left=False,
    dst_type=36,
    round_mode="rint",
) -> (Tensor, Tensor)
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
        <td>Input data, corresponding to x in the formula. Non-contiguous tensors are not supported. The ND data format is supported.</td>
        <td>float16, bfloat16</td>
        <td>1-7 dimensions, with the last dimension being 2H.</td>
    </tr>
    <tr>
        <td>beta</td>
        <td>float</td>
        <td>Optional</td>
        <td>The beta parameter of Situ activation, corresponding to β in the formula. Defaults to 1.0.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>linear_beta</td>
        <td>float</td>
        <td>Optional</td>
        <td>The linear_beta parameter of Situ activation, corresponding to linear_beta in the formula. Defaults to 0.0.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>activate_left</td>
        <td>bool</td>
        <td>Optional</td>
        <td>Indicates whether gate takes the first half or the second half of x, corresponding to activate_left in the corresponding formula. Default: false.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dst_type</td>
        <td>int</td>
        <td>Optional</td>
        <td>Specifies the data type of output y, corresponding to the target type of cast_to_fp8 in the formula. Currently, only {35, 36} are supported, corresponding to the data types of output y {35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}. Default: 36.</td>
        <td>INT</td>
        <td>-</td>
    </tr>
    <tr>
        <td>round_mode</td>
        <td>str</td>
        <td>Optional</td>
        <td>Specifies the rounding mode of quantization, that is, the rounding mode of cast in the formula. Supported values are "rint", "round", and "floor". Default: "rint".</td>
        <td>-</td>
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
        <td>y</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Quantization output, corresponding to y in the formula. Non-contiguous tensors are not supported, and the supported data format is ND.</td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
        <td>x.shape[:-1]+[H]</td>
    </tr>
    <tr>
        <td>y_scale</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>MX quantization scale (E8M0 format), corresponding to y_scale in the formula. Non-contiguous tensors are not supported, and the supported data format is ND.</td>
        <td>FLOAT8_E8M0</td>
        <td>x.shape[:-1]+[ceil(H/64), 2]</td>
    </tr>
</tbody>
</table>

where H = x.shape[-1] / 2.

## Constraints

- x must be an NPU tensor.
- The number of dimensions of x must be greater than or equal to 1, and the last dimension must be an even number.
- The data type of x must be torch.float16 or torch.bfloat16.
- beta must be greater than 0.
- dst_type supports 36 (FLOAT8_E4M3FN) or 35 (FLOAT8_E5M2).
- round_mode must be "rint".
- axis is fixed to -1 (last dimension). Other axes are not currently supported.
- Empty tensors and non-contiguous tensors are not supported.
- The shape constraints of y_scale are as follows:
  - H = x.shape[-1] / 2
  - scaleNum = ceil(H / 64)
  - y_scale.shape = x.shape[:-1] + [scaleNum, 2]
- This API supports both training and inference use cases.
- This API supports calls in single-operator mode and TorchAir graph mode.

## Deterministic Computation

Defaults to a deterministic computation.

## Example

- Call in single-operator mode (eager)

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    # FP16 -> FP8_E4M3FN
    x = torch.randn(16, 128, dtype=torch.float16).npu()
    y, y_scale = cann_ops_nn.situ_mx_quant(
        x, beta=1.0, dst_type=36, round_mode="rint",
    )
    print("y:============", y.shape, y.cpu())
    print("y_scale:============", y_scale.shape, y_scale.cpu().view(torch.uint8))

    # BF16 -> FP8_E5M2 + linear_beta
    x_bf16 = torch.randn(32, 256, dtype=torch.bfloat16).npu()
    y_bf16, y_scale_bf16 = cann_ops_nn.situ_mx_quant(
        x_bf16, beta=0.5, linear_beta=2.0, activate_left=False, dst_type=35, round_mode="rint",
    )
    print("y_bf16:============", y_bf16.shape, y_bf16.cpu())
    ```

- Call in graph mode (torchair)

    ```python
    import torch
    import torch_npu
    import torchair
    import cann_ops_nn


    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return cann_ops_nn.situ_mx_quant(
                x, beta=1.0, dst_type=36, round_mode="rint",
            )


    def situ_mx_quant_test():
        x = torch.randn(16, 128, dtype=torch.bfloat16).npu()
        model = NetModel()
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        y, y_scale = model(x)
        print("y:============", y.shape, y.cpu())
        print("y_scale:============", y_scale.shape, y_scale.cpu().view(torch.uint8))


    if __name__ == "__main__":
        situ_mx_quant_test()
    ```
