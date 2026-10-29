# situ_glu_grad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T14:14:47.618Z -->

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

  Backward gradient operator of [situ_glu](../../situ_glu/docs/torchapi_situ_glu.md). Given the forward output gradient `grad_y` and the forward input `x`, this operator differentiates `x` using the SiTU gated linear unit formula and returns the input gradient `grad_x`. The underlying implementation encapsulates `aclnnSituGluGrad`.

- Formula:

  For the given input tensor `x`, merge the axes based on the input parameter `dim`. After merging, the dimensions are `[pre, cut]`, where `cut` must be even. Let `h = cut // 2`. Given the forward output gradient $\text{grad\_y}$ (whose shape is the same as the forward output `y`, that is, the `dim` axis is half of `x`), compute the input gradient `grad_x` in the following steps:

  1. Split `x` into front and back parts to obtain `gate` and `up` based on `activate_left` (consistent with the forward; each has shape `[..., h]`):

     - `activate_left=True` (default): `gate = x[..., :h]`, `up = x[..., h:]`
     - `activate_left=False`: `gate = x[..., h:]`, `up = x[..., :h]`

  2. Compute the forward intermediates:

     $$
     t_g = \tanh\left(\frac{gate}{\beta}\right),\quad s_g = \text{sigmoid}(gate),\quad \text{situ\_a} = \beta \cdot t_g \cdot s_g
     $$

     When `linear_beta > 0`:

     $$
     t_u = \tanh\left(\frac{up}{linear\_beta}\right),\quad up' = linear\_beta \cdot t_u
     $$

     When `linear_beta <= 0`, $up' = up$. The forward output is $y = \text{situ\_a} \cdot up'$.

  3. Product rule:

     $$
     \text{grad\_situ\_a} = \text{grad\_y} \cdot up'
     $$

     $$
     \text{grad\_up'} = \text{grad\_y} \cdot \text{situ\_a}
     $$

  4. gate gradient:

     $$
     \text{grad\_gate} = \text{grad\_situ\_a} \cdot s_g \cdot \left((1 - t_g^2) + \beta \cdot t_g \cdot (1 - s_g)\right)
     $$

  5. up gradient:

     - If `linear_beta > 0`: $\text{grad\_up} = \text{grad\_up'} \cdot (1 - t_u^2)$
     - If `linear_beta <= 0`: $\text{grad\_up} = \text{grad\_up'}$

  6. Concatenate `grad_gate` and `grad_up` into `grad_x` based on `activate_left` (shape consistent with `x`):

     - If `activate_left=True`: $\text{grad\_x} = \text{concat}([\text{grad\_gate}, \text{grad\_up}], \text{dim})$
     - `activate_left=False`: $\text{grad\_x} = \text{concat}([\text{grad\_up}, \text{grad\_gate}], \text{dim})$

## Function Prototype

```python
cann_ops_nn.situ_glu_grad(
    grad_y,
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
| `grad_y` | Tensor | Required | Gradient of the forward output `y`. | float16, float32, bfloat16 | Same as `x`, but the size of the `dim` dimension is `x.shape[dim] // 2`. |
| `x` | Tensor | Required | Forward input. The `dim` dimension is evenly split into the gate and up parts. The data type of `grad_y` must be consistent with that of `x`. | float16, float32, bfloat16 | 1 to 8 dimensions, and the size of the `dim` dimension is even. |
| `dim` | int | Optional | Dimension along which to split. Value range: `[-x.dim(), x.dim()-1]`. Default value: `-1`. Must match the forward `situ_glu`. | - | - |
| `beta` | float | Optional | Scale factor of the SiTU gate part. Must remain consistent with the forward `situ_glu`. Default value: `1.0`. | - | - |
| `linear_beta` | float | Optional | Scale factor of the linear tanh in the up path. Must remain consistent with the forward `situ_glu`. Default value: `0.0`. | - | - |
| `activate_left` | bool | Optional | Whether the gate is the first half when `x` is split. Must remain consistent with the forward `situ_glu`. Default value: `True`. | - | - |

## Returns

| Name | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- |
| `grad_x` | Tensor | Gradient of input `x`. | Same as `x` | Exactly the same as `x` |

## Constraints

- This API supports calls in single-operator mode and TorchAir graph mode.
- If `dim` is not the last dimension and the combination of trailing dimensions does not meet half-row 32B alignment, the implementation falls back to the Long-H path. The result is correct but performance slightly degrades.
- The values of `beta`, `linear_beta`, `activate_left`, and `dim` must match the forward `situ_glu` call. Otherwise, the gradient result is incorrect.

## Deterministic Computation

- Defaults to a deterministic computation.

## Example

- Call in single-operator mode:

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  x = torch.randn(2, 32, dtype=torch.float32).npu()
  grad_y = torch.randn(2, 16, dtype=torch.float32).npu()

  # Call with the same parameters as the forward.
  grad_x = cann_ops_nn.situ_glu_grad(
      grad_y, x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True
  )
  print(grad_x.shape, grad_x.dtype)  # torch.Size([2, 32]) torch.float32
  ```

- Call in TorchAir graph mode:

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn

  class Model(torch.nn.Module):
      def forward(self, grad_y, x):
          return cann_ops_nn.situ_glu_grad(
              grad_y, x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True
          )

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  x = torch.randn(2, 32, dtype=torch.float32).npu()
  grad_y = torch.randn(2, 16, dtype=torch.float32).npu()
  grad_x = model(grad_y, x)
  ```
