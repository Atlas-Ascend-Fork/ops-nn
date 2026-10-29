# swiglu_group_backward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T15:29:52.649Z -->

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

  Performs backward gradient computation for the ClampedSwiglu activation function. Recomputes the clamp mask and sigmoid from the upstream gradient `grad_output` and the forward input `x`, and outputs `grad_x` and the optional `grad_weight`. The underlying implementation calls `aclnnSwigluGroupGrad`.

- Formula:

  Forward decomposition: `x` is split into two halves along the hidden dimension to obtain gate (`g`) and up (`u`); the optional clamp produces `g̃ = min(c, g)` and `ũ = clip(u, −c, c)`; `SiLU(g̃) = g̃·σ(g̃)`; `y = SiLU(g̃)·ũ·w_t`.

  $$
  silu'(g̃) = s + f − f·s
  $$

  $$
  dg = grad\_output \cdot silu'(g̃) \cdot ũ \cdot w_t \cdot I(g < c) \cdot m_r
  $$

  $$
  du = grad\_output \cdot f \cdot w_t \cdot I(−c < u < c) \cdot m_r
  $$

  $$
  grad\_weight = \Sigma(grad\_output \cdot y\_origin) \text{ along hidden dim}
  $$

  where `I` is the indicator function for the open interval (mask=0 at boundary values), `m_r` is the group_index mask, and `w_t` is the broadcast of weight.

## Function Prototype

```python
torch.ops.cann_ops_nn.swiglu_group_backward(
    grad_output,
    x,
    *,
    weight=None,
    y_origin=None,
    group_index=None,
    clamp_limit=-1.0,
) -> (Tensor, Tensor?)
```

## Parameters

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| `grad_output`| Tensor | Required | Upstream gradient. | `torch.float16`, `torch.bfloat16`, `torch.float32`| 1-8 dimensions |
| `x`| Tensor | Required | Forward input, containing the gate and up branches. | `torch.float16`, `torch.bfloat16`, `torch.float32`| 1-8 dimensions; the last dimension is `2 * grad_output.shape[-1]`. |
| `weight`| Tensor | Optional | MoE top-k routing weights. Defaults to all ones. | `torch.float32`| The number of elements equals the number of elements in `grad_output` excluding the last dimension. |
| `y_origin`| Tensor | Optional | Forward output y. When weight is provided, y has already been multiplied by the weight. | Same as `grad_output` | The last dimension is the same as `grad_output`; the number of elements in non-last dimensions equals that of `grad_output`. |
| `group_index`| Tensor | Optional | Token count index of each group. Defaults to all rows being valid. | `torch.int64`| 1 dimension |
| `clamp_limit`| float | Optional | Clamp limit. The default `-1.0` indicates no clamping. Must be greater than 0 when enabled. | - | - |

## Return Values

| Name | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- |
|`grad_x`| Tensor | Gradient of x. | Same as `grad_output` | Same as `x` |
| `grad_weight`| Tensor | Gradient of weight, returned only when both weight and y_origin are provided. | `torch.float32`| Same as `weight` |

## Constraints

- This API supports calls in single-operator mode and TorchAir graph mode.
- `grad_output`, `x`, `weight`, `y_origin`, and `group_index` must all be NPU tensors; optional tensors can be set to `None`.
- `weight` and `y_origin` must be provided together or left empty together.
- The total number of elements in the non-tail axes of `grad_output` equals that of `x`, and `x.shape[-1]` must equal `2 * grad_output.shape[-1]`.
- When `group_index` is not empty, it must be a 1-D non-empty tensor (G > 0).
- `clamp_limit` must be `-1.0` or greater than `0.0`; `-1.0` indicates no clamping.

## Deterministic Computation

Defaults to a deterministic computation.

## Example

- Call in single-operator mode:

  ```python
  import torch
  import torch_npu
  import cann_ops_nn.ops

  grad_output = torch.randn(8, 128, dtype=torch.float16).npu()
  x = torch.randn(8, 256, dtype=torch.float16).npu()
  weight = torch.ones(8, 1, dtype=torch.float32).npu()
  y_origin = torch.randn(8, 128, dtype=torch.float16).npu()
  group_index = torch.tensor([8], dtype=torch.int64).npu()

  grad_x, grad_weight = torch.ops.cann_ops_nn.swiglu_group_backward(
      grad_output, x, weight=weight, y_origin=y_origin,
      group_index=group_index, clamp_limit=7.0)
  print(grad_x.shape, grad_x.dtype)
  print(grad_weight.shape, grad_weight.dtype)
  ```

- Call in graph mode (torchair):

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn.ops

  class Model(torch.nn.Module):
      def forward(self, grad_output, x):
          grad_x, _ = torch.ops.cann_ops_nn.swiglu_group_backward(
              grad_output, x, clamp_limit=-1.0)
          return grad_x

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  grad_output = torch.randn(8, 128, dtype=torch.float16).npu()
  x = torch.randn(8, 256, dtype=torch.float16).npu()
  grad_x = model(grad_output, x)
  ```
