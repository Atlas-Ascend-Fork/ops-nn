# situ_glu

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T14:01:44.004Z -->

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

  SiTU Gated Linear Unit activation function. Splits the input tensor `x` along the specified dimension `dim` into two halves: gate and up, and computes the output using the SiTU formula. The bottom layer encapsulates `aclnnSituGlu`.

- Formula:

  For a given input tensor `x`, compute the SiTU output `y` as follows:

  1. Merge the axes based on `dim` so that the shape after merging is `[pre, cut]`, where `cut` must be even. Let `h = cut // 2`.

  2. Split `x` into the front and rear parts based on `activate_left`:

     - `activate_left=True` (default): `gate = x[..., :h]`, `up = x[..., h:]`
     - `activate_left=False`: `gate = x[..., h:]`, `up = x[..., :h]`

  3. Perform the SiTU computation based on `beta` and `linear_beta`:

     $$
     situ\_a = \beta \cdot \tanh\left(\frac{gate}{\beta}\right) \cdot \text{sigmoid}(gate)
     $$

     When `linear_beta > 0`:

     $$
     up = linear\_beta \cdot \tanh\left(\frac{up}{linear\_beta}\right)
     $$

     When `linear_beta <= 0`, `up` is directly passed through.

  4. Output:

     $$
     y = situ\_a \cdot up
     $$

  5. The dimensions of output tensor `y` are the same as those of `x`, the size on the `dim` axis is half of that of `x`, and the other dimensions are the same as `x`.

## Function Prototype

```python
cann_ops_nn.situ_glu(
    x,
    *,
    dim=-1,
    beta=1.0,
    linear_beta=0.0,
    activate_left=True
) -> Tensor
```

## Parameters

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | Required | SiTU input. The `dim` dimension is evenly split into the gate part and the up part. | float16, float32, bfloat16 | 1 to 8 dimensions; the `dim` dimension must be an even number |
| `dim` | int | Optional | Dimension along which the input is split. Value range: `[-x.dim(), x.dim()-1]`. Default: `-1`. | - | - |
| `beta` | float | Optional | Scaling factor of the gate part in SiTU, which controls the intensity of the tanh nonlinearity. A non-zero positive value is recommended. Default: `1.0`. | - | - |
| `linear_beta` | float | Optional | Scaling factor for the linear tanh of the up path. If the value is greater than 0, a bounded transformation is applied to up; if the value is less than or equal to 0, up is passed through directly. Default: `0.0`. | - | - |
| `activate_left` | bool | Optional | Specifies whether gate is the first half when `x` is split. `True` indicates that gate is the first half and up is the second half; `False` indicates that gate is the second half and up is the first half. Default: `True`. | - | - |

## Return Values

| Name | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- |
| `y` | Tensor | SiTU activation result. | Same as `x` | Same as `x`, but the size of the `dim` dimension is `x.shape[dim] // 2` |

## Constraints

- This API supports both training and inference use cases.
- This API can be called in both single-operator mode and TorchAir graph mode.
- When `dim` is not the last dimension, if the combination of trailing dimensions does not meet half-row 32B alignment, the API falls back to the Long-H path. The result is correct but performance slightly degrades.

## Deterministic Computation

- Defaults to a deterministic computation.

## Example

- Single-operator mode call:

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  x = torch.randn(2, 32, dtype=torch.float32).npu()

  # Call with default parameters.
  y = cann_ops_nn.situ_glu(x)
  print(y.shape, y.dtype)  # torch.Size([2, 16]) torch.float32

  # Specify all parameters.
  y = cann_ops_nn.situ_glu(x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True)
  print(y.shape, y.dtype)
  ```

- TorchAir graph mode call:

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn

  class Model(torch.nn.Module):
      def forward(self, x):
          return cann_ops_nn.situ_glu(x, dim=-1, beta=1.0)

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  x = torch.randn(2, 32, dtype=torch.float32).npu()
  y = model(x)
  ```
