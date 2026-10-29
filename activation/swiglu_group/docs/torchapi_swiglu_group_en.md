# swiglu_group

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T15:17:07.084Z -->

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

  Performs SwiGLU activation on input `x`. Supports optional per-token weights and count mode grouping. The underlying implementation encapsulates `aclnnSwigluGroup`.

- Formula:

  Let `D` be the size of the last dimension of input `x`, `A = x[..., :D/2]`, and `B = x[..., D/2:]`.

  $$
  y = silu(A) \times B = \frac{A}{1 + e^{-A}} \times B
  $$

  When `clamp_limit > 0`, `A` and `B` are clamped first. When `weight` is not empty, the result is multiplied by `weight` token by token.

## Function Prototype

```python
cann_ops_nn.swiglu_group(
    x,
    *,
    weight=None,
    group_index=None,
    clamp_limit=-1.0,
) -> Tensor
```

## Parameters

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | Required | SwiGLU input. The last dimension is evenly split into two parts. | `torch.float16`, `torch.bfloat16`, `torch.float32` | 1-8 dimensions, with the last dimension being an even number |
| `weight` | Tensor | Optional | Per-token weight. When not empty, it is multiplied to the SwiGLU result. | `torch.float32` | 1-8 dimensions, with the number of elements equal to that of `x` excluding the last dimension |
| `group_index` | Tensor | Optional | Number of tokens grouped in count mode. | `torch.int64` | 1 dimension |
| `clamp_limit` | float | Optional | Clamps A and B before activation. The default value is -1.0, indicating that no clamping is performed. If clamping is enabled, the value must be greater than 0. | - | - |

## Return Values

| Name | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- |
| `y` | Tensor | SwiGLU activation result. | Same as `x` | Last dimension is `x.shape[-1] // 2`, and other dimensions are the same as `x`. |

## Constraints

- This API can be called in single-operator mode and TorchAir graph mode.
- `x`, `weight`, and `group_index` must be NPU tensors; optional tensors can be set to `None`.
- The last dimension of `x` must be even.
- `x` and `weight` support empty tensors, but an empty group_index alone is not supported.
- Non-contiguous tensors are not supported.
- `group_index` indicates the number of tokens in each group. All elements must be greater than or equal to 0.

## Deterministic Computation

Defaults to a deterministic computation.

## Example

- Call in single-operator mode:

  ```python
  import torch
  import torch_npu
  import cann_ops_nn.ops

  x = torch.randn(8, 512, dtype=torch.float16).npu()
  weight = torch.ones(8, dtype=torch.float32).npu()
  group_index = torch.tensor([4, 4], dtype=torch.int64).npu()

  y = torch.ops.cann_ops_nn.swiglu_group(
      x, weight=weight, group_index=group_index, clamp_limit=-1.0)
  print(y.shape, y.dtype)
  ```

- Call in graph mode (torchair):

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn.ops

  class Model(torch.nn.Module):
      def forward(self, x):
          return torch.ops.cann_ops_nn.swiglu_group(x, clamp_limit=-1.0)

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  x = torch.randn(8, 512, dtype=torch.float16).npu()
  y = model(x)
  ```
