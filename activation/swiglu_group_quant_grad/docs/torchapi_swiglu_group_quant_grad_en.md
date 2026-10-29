# swiglu_group_quant_backward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T16:02:56.933Z -->

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

 `SwigluGroupQuantGrad` is the backward gradient computation operator of `SwigluGroupQuant`. Based on the gradient `grad_y` of the quantized output `y`, the forward input `x`, and optional forward auxiliary tensors, it computes the gradient `grad_x` of the input `x` and the gradient `grad_weight` of the optional weight `weight` through backward propagation. The underlying implementation calls `aclnnSwigluGroupQuantGrad`.

- Calculation Process:

  1. Receives the gradient `grad_y` of the forward quantized output and the forward input `x`.
  2. Backpropagates gradients along the quantization path and the SwiGLU activation path based on the input segmentation of the forward SwiGLU and the quantization parameters.
  3. Output `grad_x` (same shape/dtype as `x`) and `grad_weight` (same shape as `weight`; an empty Tensor if `weight` is not provided).

## Function Prototype

```python
cann_ops_nn.swiglu_group_quant_backward(
    grad_y,
    x,
    *,
    weight=None,
    y_origin=None,
    group_index=None,
    clamp_limit=-1.0,
) -> (Tensor, Tensor)
```

## Parameters

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
|`grad_y`| Tensor | Required | Gradient input tensor, i.e., the gradient of the forward quantized output `y`. |`torch.float16`, `torch.bfloat16`, `torch.float32`| ND, consistent with forward output `y` |
|`x`| Tensor | Required | Input tensor of the forward computation. |`torch.float16`, `torch.bfloat16`, `torch.float32`| ND, consistent with forward input `x` |
|`weight`| Tensor | Optional | Topk weight tensor used in the forward computation. |`torch.float32`| ND, consistent with forward `weight` |
|`y_origin`| Tensor | Optional | SwiGLU output tensor before forward quantization. |`torch.float16`, `torch.bfloat16`, `torch.float32`| ND, consistent with forward `y_origin` |
|`group_index`| Tensor | Optional | Group index tensor used in forward dynamic quantization. |`torch.int64`| ND, consistent with forward `group_index` |
|`clamp_limit`| float | Optional | Clamp value of the gradient mask. The default `-1.0` indicates no clamping. If set to a positive value, the SwiGLU input is clamped before activation. Must be `-1.0` or greater than `0`, and must be consistent with forward `clamp_limit`. | - | - |

## Return Values

| Name | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- |
|`grad_x`| Tensor | Gradient of input `x`, with the same data type as `x`. | Same as `x` | ND, same as `x` |
|`grad_weight`| Tensor | Gradient of weight `weight`. If `weight` is not provided, an empty tensor with shape `[0]` is returned. |`torch.float32`| ND, same as `weight`, or `[0]`|

## Constraints

- This API supports both single-operator mode and TorchAir graph mode.
-`grad_y` and `x` must be NPU tensors; optional tensors can be passed as `None`.
- The rank of input `x` must be greater than 0, and the last dimension must be an even number.
-`clamp_limit` uses the default value `-1.0` when not enabled; when enabled, it must be greater than `0` and must be consistent with forward `clamp_limit`.
- Empty tensors and non-contiguous tensors are not supported.

## Deterministic Computation

Defaults to a deterministic computation.

## Invocation

- Calling in single-operator mode:

 ```python
  import torch
  import torch_npu
  import cann_ops_nn.ops

  grad_y = torch.randn(8, 256, dtype=torch.float16).npu()
  x = torch.randn(8, 512, dtype=torch.float16).npu()
  weight = torch.randn(256, dtype=torch.float32).npu()

  grad_x, grad_weight = cann_ops_nn.swiglu_group_quant_backward(
      grad_y, x, weight=weight, clamp_limit=-1.0
  )
 ```

- Calling in graph mode (torchair):

 ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn.ops

  class Model(torch.nn.Module):
      def forward(self, grad_y, x, weight):
          grad_x, grad_weight = cann_ops_nn.swiglu_group_quant_backward(
              grad_y, x, weight=weight
          )
          return grad_x, grad_weight

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  grad_y = torch.randn(8, 256, dtype=torch.float16).npu()
  x = torch.randn(8, 512, dtype=torch.float16).npu()
  weight = torch.randn(256, dtype=torch.float32).npu()
  grad_x, grad_weight = model(grad_y, x, weight)
 ```
