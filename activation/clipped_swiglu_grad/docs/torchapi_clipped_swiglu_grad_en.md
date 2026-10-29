# clipped_swiglu_grad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T09:02:16.415Z -->

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

  `ClippedSwigluGrad` is the backward gradient computation operator of `ClippedSwiglu`. Based on the gradient `grad_y` of the forward output `y`, the forward input `x`, and the optional group index `group_index`, it computes the gradient `grad_x` of the input `x`. The underlying implementation is `aclnnClippedSwigluGrad`.

- Calculation Process:

  1. Merge the axes of `x` based on the parameter `dim`. The dimension of `x` after merging is `[pre, dim2H]`; the dimension of `grad_y` after merging is `[pre, dim2H // 2]`.
  2. Filter the `pre` axis of `x` based on `group_index`. Only the first `group` rows participate in the computation, and the gradient of the remaining rows is 0.
  3. Split `x` into two parts `x0` and `x1` based on `interleaved` (same as the forward pass).
  4. Recompute the forward intermediate quantities:
     - $\mathbf{x}_0' = \min(\mathbf{x}_0, \text{limit})$
     - $\mathbf{x}_1' = \min(\max(\mathbf{x}_1, -\text{limit}), \text{limit})$
     - $s = \sigma(\alpha \times \mathbf{x}_0') = \frac{1}{1 + e^{-\alpha \times \mathbf{x}_0'}}$
  5. Compute the backward gradients:
     - $\text{mask}_{x_0} = (\mathbf{x}_0 \le \text{limit})$
     - $\text{mask}_{x_1} = (-\text{limit} \le \mathbf{x}_1 \le \text{limit})$
     - $\mathbf{grad}_{x_0} = \text{grad\_y} \times (\mathbf{x}_1' + \text{bias}) \times s \times (1 + \alpha \times \mathbf{x}_0' \times (1 - s)) \times \text{mask}_{x_0}$
     - $\mathbf{grad}_{x_1} = \text{grad\_y} \times \mathbf{x}_0' \times s \times \text{mask}_{x_1}$
  6. Scatter `grad_x0` and `grad_x1` back to `dx` (corresponding to the splitting method), and reshape the output to the same dimensions as `x`.

## Function Prototype

```python
cann_ops_nn.clipped_swiglu_grad(
    grad_y,
    x,
    *,
    group_index=None,
    dim=-1,
    alpha=1.702,
    limit=7.0,
    bias=1.0,
    interleaved=True,
    clamp_mode=0,
) -> Tensor
```

## Parameters

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| `grad_y` | Tensor | Required | Gradient of the forward output `y`, with the `dim` dimension being half of `x`. | `torch.float16`, `torch.bfloat16`, `torch.float32` | Same as `x`, but the `dim` dimension is half of the corresponding dimension of `x` |
| `x` | Tensor | Required | Forward input, used to recompute the forward intermediate quantities. | `torch.float16`, `torch.bfloat16`, `torch.float32` | 1 to 8 dimensions, with an even size in the `dim` dimension |
| `group_index` | Tensor | Optional | Group index. The `i`-th element indicates the number of batches processed by the `i`-th group. | `torch.int64` | 1 dimension, with no more than 8192 elements |
| `dim` | int | Optional | Index of the dimension for axis merging and splitting. Value range: `[-x.dim(), x.dim()-1]`. Default: `-1`. | - | - |
| `alpha` | float | Optional | SwiGLU activation coefficient. Default: `1.702`. | - | - |
| `limit` | float | Optional | Clipping threshold. The value must be greater than 0. Default: `7.0`. | - | - |
| `bias` | float | Optional | Bias of the linear computation. Default: `1.0`. | - | - |
| `interleaved` | bool | Optional | Splitting method. `true` indicates odd-even splitting, and `false` indicates front-back splitting. Default: `true`. | - | - |
| `clamp_mode` | int | Optional | Clamp position control. Default: `0`. Currently, only `0` (clamp before silu) is supported. `1` (clamp after silu) will be supported in later versions. | - | - |

## Return Values

| Name | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- |
| `grad_x` | Tensor | Gradient of input `x`. | Same as `x` | Same as `x` |

## Constraints

- This API supports single-operator mode and TorchAir graph mode calls.
- `grad_y` and `x` must both be NPU tensors; `group_index` can be `None`.
- The data types of `grad_y` and `x` must be the same.
- The rank of `x` must be greater than 0, and the size of the `dim` dimension must be an even number.
- The size of the `dim` dimension of `grad_y` must be half of the corresponding dimension of `x`, and the other dimensions must be the same as those of `x`.
- `limit` must be greater than 0.
- Currently, `clamp_mode` supports only the value `0`.
- Empty tensors and non-contiguous tensors are not supported.

## Deterministic Computation

Defaults to a deterministic computation.

## API Call

- Single-operator mode call:

  ```python
  import torch
  import torch_npu
  import cann_ops_nn.ops

  x = torch.randn(8, 512, dtype=torch.float16).npu()
  grad_y = torch.randn(8, 256, dtype=torch.float16).npu()

  grad_x = torch.ops.cann_ops_nn.clipped_swiglu_grad(
      grad_y, x, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True)
  print(grad_x.shape, grad_x.dtype)
  ```

- Graph mode (torchair) call:

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn.ops

  class Model(torch.nn.Module):
      def forward(self, grad_y, x):
          return torch.ops.cann_ops_nn.clipped_swiglu_grad(grad_y, x)

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  grad_y = torch.randn(8, 256, dtype=torch.float16).npu()
  x = torch.randn(8, 512, dtype=torch.float16).npu()
  grad_x = model(grad_y, x)
  ```
