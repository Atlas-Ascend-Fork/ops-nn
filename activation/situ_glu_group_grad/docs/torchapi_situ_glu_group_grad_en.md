# situ_glu_group_grad

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

- Interface function:

  The backward gradient operator of [situ_glu](../../situ_glu/docs/torchapi_situ_glu_en.md). Given the forward output gradient `grad_y` and the forward input `x`, it differentiates `x` by the SiTU gated linear unit formula and returns the input gradient `grad_x`. It wraps `aclnnSituGluGroupGrad` underneath.

- Calculation formula:

  For a given input tensor `x`, folds it based on the input parameter `dim` into `[pre, cut]`, where `cut` must be even. Let `h = cut // 2`. Given the forward output gradient $\text{grad\_y}$ (whose shape matches the forward output `y`, i.e. the `dim` axis is half of `x`), the input gradient `grad_x` is computed as follows:

  1. Splits `x` into the front and back parts based on `activate_left` to obtain `gate` and `up` (consistent with the forward, each of shape `[..., h]`):

     - `activate_left=True` (default): `gate = x[..., :h]`, `up = x[..., h:]`
     - `activate_left=False`: `gate = x[..., h:]`, `up = x[..., :h]`

  2. Computes the forward intermediates:

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

  4. Gate gradient:

     $$
     \text{grad\_gate} = \text{grad\_situ\_a} \cdot s_g \cdot \left((1 - t_g^2) + \beta \cdot t_g \cdot (1 - s_g)\right)
     $$

  5. Up gradient:

     - When `linear_beta > 0`: $\text{grad\_up} = \text{grad\_up'} \cdot (1 - t_u^2)$
     - When `linear_beta <= 0`: $\text{grad\_up} = \text{grad\_up'}$

  6. Concatenates `grad_gate` and `grad_up` into `grad_x` according to `activate_left` (shape identical to `x`):

     - `activate_left=True`: $\text{grad\_x} = \text{concat}([\text{grad\_gate}, \text{grad\_up}], \text{dim})$
     - `activate_left=False`: $\text{grad\_x} = \text{concat}([\text{grad\_up}, \text{grad\_gate}], \text{dim})$

  7. When the optional input `weight` (per-row scalar $w_n$) is passed, grad_gate/grad_up are further multiplied per row by $w_n$ (chain rule, forward $y_n = w_n \cdot \text{situ\_a} \cdot up'$), and the output `grad_weight` is computed at the same time (note: not multiplied by $w_n$):

     $$
     \text{grad\_w}_n = \sum_{h} \text{grad\_y}_{n,h} \cdot \text{situ\_a}_{n,h} \cdot up'_{n,h}
     $$

  8. When the optional input `group_index` is passed: $\text{realBs} = \min(\text{groupRows}, \text{number of rows})$, and only the first realBs rows are computed and written back. groupRows is selected by `group_list_type`: `group_index[-1]` in cumsum mode (0, the default) and $\sum \text{group\_index}$ in count mode (1).

## Function Prototype

```python
cann_ops_nn.situ_glu_group_grad(
    grad_y,
    x,
    *,
    weight=None,
    group_index=None,
    dim=-1,
    beta=1.0,
    linear_beta=0.0,
    activate_left=True,
    group_list_type=0
) -> (Tensor, Tensor)
```

## Parameters

| Parameter name | Parameter type | Optional/Required | Description | Data type | Dimension (shape) |
| --- | --- | --- | --- | --- | --- |
| `grad_y` | Tensor | Required | Gradient of the forward output `y`. | float16, float32, bfloat16 | Same as `x`, but the size of the `dim` dimension is `x.shape[dim] // 2` |
| `x` | Tensor | Required | Forward input. The `dim` dimension is evenly split into the gate and up parts. The data types of `grad_y` and `x` must be the same. | float16, float32, bfloat16 | 1-8 dimensions, the `dim` dimension is even |
| `weight` | Tensor | Optional | Per-row routing weight of the forward. The number of elements must equal the product of the dimensions of `x` before the `dim` axis. Supported only on Ascend 950PR&950DT products. The default value is `None`. | float32 | 1-8 dimensions |
| `group_index` | Tensor | Optional | Number of tokens of each group, whose meaning is selected by `group_list_type`. A 1-D non-empty tensor with non-negative elements. Only the first `min(groupRows, number of rows)` rows are computed and written back. Supported only on Ascend 950PR&950DT products. The default value is `None`. | int64 | 1 dimension |
| `dim` | int | Optional | Dimension to split. The value range is `[-x.dim(), x.dim()-1]`. The default value is `-1`. It must be consistent with the forward `situ_glu`. | - | - |
| `beta` | float | Optional | Scaling coefficient of the SiTU gate part. It must be consistent with the forward `situ_glu`. The default value is `1.0`. | - | - |
| `linear_beta` | float | Optional | Scaling coefficient of the linear tanh on the up path. It must be consistent with the forward `situ_glu`. The default value is `0.0`. | - | - |
| `activate_left` | bool | Optional | Whether gate is the front half when `x` is split. It must be consistent with the forward `situ_glu`. The default value is `True`. | - | - |
| `group_list_type` | int | Optional | How `group_index` expresses the group sizes. The value can only be 0 or 1: 0 is cumsum mode, where `group_index` holds the cumulative row counts and the total is `group_index[-1]`; 1 is count mode, where `group_index` holds the per-group row counts and the total is `sum(group_index)`. The default value is `0`. | - | - |

## Return Values

Returns the tuple `(grad_x, grad_weight)`.

| Parameter name | Parameter type | Description | Data type | Dimension (shape) |
| --- | --- | --- | --- | --- |
| `grad_x` | Tensor | Gradient of the input `x`. | Same as `x` | Exactly the same as `x` |
| `grad_weight` | Tensor | Gradient of the input `weight`: the shape is the same as `weight` when `weight` is passed, and an empty tensor (shape `[0]`) when it is absent. | float32 | Same as `weight`, or empty `[0]` |

## Constraints

- This interface supports both single-operator mode and TorchAir graph mode.
- When `dim` is not the last dimension, if the trailing dimension combination does not satisfy half-row 32B alignment, it falls back to the Long-H path; the result is correct but performance is slightly lower.
- `beta`, `linear_beta`, `activate_left` and `dim` must be consistent with the values used in the forward `situ_glu` call; otherwise the gradient result is incorrect.
- Supported only on Ascend 950PR&950DT products (`weight` and `group_index` are optional inputs, and the return value `grad_weight` is a required output).
- When `group_index` is passed, rows whose index ≥ `realBs` in `grad_x` and `grad_weight` are not written; clear the corresponding rows explicitly if strict zeros are required.
- `group_list_type` must take a value from [0, 1]; any other value returns a parameter validation failure.
- The values of `group_index` must meet the following conditions, otherwise the output is not guaranteed to be as expected:
  - When `group_list_type` is 0, `group_index` must be a non-negative, monotonically non-decreasing sequence, representing the cumsum result (cumulative sum) of the size of each group after grouping. The last value must be less than or equal to the number of rows.
  - When `group_list_type` is 1, `group_index` must be a non-negative sequence, representing the size of each group after grouping. The sum of the values must be less than or equal to the number of rows.

## Deterministic Computation

- Deterministic computation is supported by default.

## Example

- Single-operator mode:

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  x = torch.randn(2, 32, dtype=torch.float32).npu()
  grad_y = torch.randn(2, 16, dtype=torch.float32).npu()

  # Call with parameters consistent with the forward
  grad_x, grad_weight = cann_ops_nn.situ_glu_group_grad(
      grad_y, x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True
  )
  # grad_weight is an empty tensor (shape [0]) when weight is absent
  print(grad_x.shape, grad_x.dtype, grad_weight.shape)  # torch.Size([2, 32]) torch.float32 torch.Size([0])

  # Pass weight and group_index (supported only on Ascend 950PR&950DT products)
  weight = torch.rand(2, dtype=torch.float32).npu()
  # group_list_type=0 (the default, cumsum mode): the total row count is group_index[-1]=2, so all 2 rows are computed
  group_index = torch.tensor([1, 2], dtype=torch.int64).npu()
  grad_x, grad_weight = cann_ops_nn.situ_glu_group_grad(
      grad_y, x, weight=weight, group_index=group_index,
      dim=-1, beta=1.0, linear_beta=0.0, activate_left=True, group_list_type=0
  )
  print(grad_weight.shape, grad_weight.dtype)  # torch.Size([2]) torch.float32
  ```

- TorchAir graph mode:

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn

  class Model(torch.nn.Module):
      def forward(self, grad_y, x):
          grad_x, _ = cann_ops_nn.situ_glu_group_grad(
              grad_y, x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True
          )
          return grad_x

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  x = torch.randn(2, 32, dtype=torch.float32).npu()
  grad_y = torch.randn(2, 16, dtype=torch.float32).npu()
  grad_x = model(grad_y, x)
  ```
