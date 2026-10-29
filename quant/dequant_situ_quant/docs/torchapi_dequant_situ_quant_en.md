# dequant_situ_quant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:14:38.372Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Does not support
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Does not support
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Does not support
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Does not support
<!-- end id6 -->

## Function

- Description: Performs fused computation of dequantization (Dequant), Situ activation, and quantization (Quant) on input x. The underlying implementation calls aclnnDequantSituQuant.

- Formula:

  1. The dequantization path varies depending on the data type of input x:

  - INT8 path:

    $$
    dequantOut_i = cast\_to\_float(x_i) \times weight\_scale_i + bias_i
    $$

  - INT32 path:

    $$
    dequantOut_i = cast\_to\_float(x_i) \times weight\_scale_i \times activation\_scale_i + bias_i
    $$

  - BFLOAT16/FLOAT16 path (pre-dequantization):

    $$
    dequantOut_i = cast\_to\_float(x_i)
    $$
  2. Situ activation:

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

    When activate_left is true, gate takes the first half of dequantOut and up takes the second half. When activate_left is false, gate takes the second half of dequantOut and up takes the first half.
  3. Quantization:

  - static mode:

    $$
    y_i = trunc(situOut_i / quant\_scale_i + quant\_offset_i)
    $$

  - dynamic mode:

    $$
    scale_i = absmax(situOut_i) / 127
    $$

    $$
    y_i = trunc(situOut_i / scale_i)
    $$

## Function Prototype

```python
cann_ops_nn.dequant_situ_quant(
    x,
    *,
    weight_scale=None,
    activation_scale=None,
    bias=None,
    quant_scale=None,
    quant_offset=None,
    group_index=None,
    beta=4.0,
    linear_beta=25.0,
    activate_left=True,
    quant_type="dynamic",
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
        <td>Input data, corresponding to x in the formula. Does not support non-contiguous tensors. Supports the ND data format.</td>
        <td>int8, int32, bfloat16, float16</td>
        <td>INT8: 2-8 dimensions; others: 2 dimensions, with the last dimension being 2H</td>
    </tr>
    <tr>
        <td>weight_scale</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Dequantization weight scale, corresponding to weight_scale in the formula. Does not support non-contiguous. Supports ND data format.</td>
        <td>float32</td>
        <td>1 dimension, shape (2H,) or (1,)</td>
    </tr>
    <tr>
        <td>activation_scale</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Dequantization activation scale, corresponding to activation_scale in the formula. Does not support non-contiguous. Supports ND data format.</td>
        <td>float32</td>
        <td>1 dimension, shape (1,)</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Dequantization bias, which is the bias in the corresponding formula. Does not support non-contiguous tensors; supports the ND data format.</td>
        <td>float32</td>
        <td>1 dimension, shape (2H,) or (1,)</td>
    </tr>
    <tr>
        <td>quant_scale</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Quantization scale, corresponding to quant_scale in the formula. Does not support non-contiguous tensors; supports the ND data format.</td>
        <td>float32</td>
        <td>1D, shape (H,) or (1,)</td>
    </tr>
    <tr>
        <td>quant_offset</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Quantization offset, corresponding to quant_offset in the formula. Does not support non-contiguous tensors; supports the ND data format.</td>
        <td>float32</td>
        <td>1D, shape (H,) or (1,)</td>
    </tr>
    <tr>
        <td>group_index</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>MoE group index. Does not support non-contiguous tensors; ND data format is supported.</td>
        <td>int64</td>
        <td>1 dimension, shape (K,)</td>
    </tr>
    <tr>
        <td>beta</td>
        <td>float</td>
        <td>Optional</td>
        <td>Beta parameter for Situ activation, corresponds to β in the formula. Defaults to 4.0.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>linear_beta</td>
        <td>float</td>
        <td>Optional</td>
        <td>linear_beta parameter of Situ activation, corresponding to linear_beta in the formula. Defaults to 25.0.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>activate_left</td>
        <td>bool</td>
        <td>Optional</td>
        <td>Indicates whether gate takes the first half or the second half of dequantOut, corresponding to activate_left in the corresponding formula. Defaults to true.</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quant_type</td>
        <td>str</td>
        <td>Optional</td>
        <td>Quantization mode, corresponding to the static/dynamic mode in the corresponding formula. Supports "static" and "dynamic". Defaults to "dynamic".</td>
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
        <td>y</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Quantization output, corresponding to y in the formula. Does not support non-contiguous tensors. Only the ND data format is supported.</td>
        <td>int8</td>
        <td>INT8: x.shape[:-1]+[H]; other: [M, H]</td>
    </tr>
    <tr>
        <td>y_scale</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Dynamic quantization scale, corresponding to scale in the formula (meaningful only in dynamic mode). Does not support non-contiguous tensors. Only the ND data format is supported.</td>
        <td>float32</td>
        <td>INT8: x.shape[:-1]; other: [M]</td>
    </tr>
</tbody>
</table>

where H = x.shape[-1] / 2.

## Constraints

- x and the optional tensors must all be NPU tensors; None can be passed for an optional tensor.
- The last dimension of x must be an even number. If the data type of x is INT8, the number of dimensions must be ≥ 2; if the data type of x is INT32/BF16/FLOAT16, the number of dimensions must be 2.
- beta cannot be 0.
- quant_scale is required when quant_type is "static".
- Empty tensors are supported (an empty output is returned directly when x.size(0) == 0).
- The constraints on quant_type are as follows:

  | quant_type | Meaning | quant_scale | quant_offset |
  | ---- | ---- | ---- | ---- |
  | "static" | Static quantization | Required | Optional |
  | "dynamic" | Dynamic quantization | Optional (used as smoothScale) | Not effective |

- The input constraints for each data type are as follows:

  | X Data Type | Required Input | Optional Input | Prohibited Input |
  | ---- | ---- | ---- | ---- |
  | INT8 | x, weight_scale | bias, quant_scale, quant_offset | activation_scale, group_index |
  | INT32 | x, weight_scale, activation_scale | bias, group_index | quant_scale, quant_offset |
  | BF16 | x | — | weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index |
  | FLOAT16 | x | — | weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index |

- This API can be used in training and inference use cases.
- This API can be called in single-operator mode and TorchAir graph mode.

## Deterministic Computation

Defaults to a deterministic computation.

## Example

- Single-operator mode (eager) call

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    # INT8, dynamic quantization
    x = torch.randint(-127, 127, (16, 64), dtype=torch.int8).npu()
    weight_scale = torch.full((64,), 0.1, dtype=torch.float32).npu()

    y, y_scale = cann_ops_nn.dequant_situ_quant(
        x, weight_scale=weight_scale,
        beta=4.0, linear_beta=25.0, activate_left=True, quant_type="dynamic",
    )
    print("y:============", y.shape, y.cpu())
    print("y_scale:============", y_scale.shape, y_scale.cpu())

    # BF16, pre-dequantization
    x_bf16 = torch.randn(32, 128, dtype=torch.bfloat16).npu()
    y_bf16, y_scale_bf16 = cann_ops_nn.dequant_situ_quant(
        x_bf16, beta=4.0, linear_beta=25.0, activate_left=True, quant_type="dynamic",
    )
    print("y_bf16:============", y_bf16.shape, y_bf16.cpu())
    ```

- Graph mode (torchair) call

    ```python
    import torch
    import torch_npu
    import torchair
    import cann_ops_nn


    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, weight_scale, quant_scale):
            return cann_ops_nn.dequant_situ_quant(
                x, weight_scale=weight_scale, quant_scale=quant_scale,
                beta=4.0, linear_beta=25.0, activate_left=True, quant_type="static",
            )


    def dequant_situ_quant_test():
        x = torch.randint(-127, 127, (16, 64), dtype=torch.int8).npu()
        weight_scale = torch.full((64,), 0.1, dtype=torch.float32).npu()
        quant_scale = torch.tensor([1.0], dtype=torch.float32).npu()
        model = NetModel()
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        y, y_scale = model(x, weight_scale, quant_scale)
        print("y:============", y.shape, y.cpu())
        print("y_scale:============", y_scale.shape, y_scale.cpu())


    if __name__ == "__main__":
        dequant_situ_quant_test()
    ```
