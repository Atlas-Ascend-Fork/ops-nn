# BatchNormGradV3

English | [简体中文](./README.md)

## Applicable Products

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

The table lists support for the standard ACLNN interface. See [aclnnFastBatchNormBackward](docs/aclnnFastBatchNormBackward_en.md) for Fast API support and the constraints below for direct GE graph invocation.

## Function

BatchNormGradV3 computes the input, scale, and bias gradients for batch normalization. Training uses saved forward statistics. For the formulas, see [aclnnBatchNormBackward](docs/aclnnBatchNormBackward_en.md#function).

## Parameters

The following parameters apply to the GE graph operator. C is the channel count.

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
| --- | --- | --- | --- | --- |
| dy | Input | Upstream gradient. | FLOAT32, FLOAT16, BFLOAT16 | NCHW, NHWC, NCDHW, NDHWC |
| x | Input | Forward input, with the same shape, data type, and format as dy. | FLOAT32, FLOAT16, BFLOAT16 | NCHW, NHWC, NCDHW, NDHWC |
| weight | Input | Per-channel scale, with shape `[C]`. | FLOAT32, FLOAT16, BFLOAT16 | ND |
| running_mean | Input | Running mean, with shape `[C]`. | FLOAT32, FLOAT16, BFLOAT16 | ND |
| running_var | Input | Running variance, with shape `[C]`. | FLOAT32, FLOAT16, BFLOAT16 | ND |
| save_mean | Input | Saved mean for training, with shape `[C]`. | FLOAT32 | ND |
| save_rstd | Input | Saved statistic for training, with shape `[C]`. See the product-specific constraints. | FLOAT32 | ND |
| dx | Output | Input gradient, with the same shape, data type, and format as dy. | FLOAT32, FLOAT16, BFLOAT16 | NCHW, NHWC, NCDHW, NDHWC |
| dweight | Output | Scale gradient, with data type inferred from weight. | FLOAT32, FLOAT16, BFLOAT16 | ND |
| dbias | Output | Bias gradient, with data type inferred from weight. | FLOAT32, FLOAT16, BFLOAT16 | ND |
| is_training | Optional attribute | true for training and false for inference. Default: true. | BOOL | - |
| epsilon | Optional attribute | Value added to the variance to improve numerical stability. Default: 1e-5. A small, finite positive value is recommended. | FLOAT32 | - |
| output_mask | Optional attribute | Three booleans in the order dx, dweight, dbias. Default: `{true, true, true}`. | ListBool | - |

## Constraints

- The channel axis is the second dimension in NCHW and NCDHW and the last dimension in NHWC and NDHWC.
- Connect all required GE input tensors when building the graph. The ACLNN wrapper supplies defaults for its optional null pointers and adapts supported lower-rank and higher-rank inputs to the kernel layouts. See the [ACLNN API specification](docs/aclnnBatchNormBackward_en.md) for its shape and format requirements.
- <term>Ascend 950PR&950DT products</term>:
  - The direct GE operator uses 4D NCHW/NHWC or 5D NCDHW/NDHWC. Training takes the saved reciprocal standard deviation as save_rstd.
  - Training supports inputs with an empty non-channel axis. When C > 0 and the reduction has zero elements, dx is empty, and enabled dweight and dbias outputs retain shape `[C]` and are set to zero. Disabled outputs are not written.
  - In inference, the GE kernel computes dx. The standard ACLNN interface also computes gradWeight and gradBias in inference according to outputMask.
  - Shape inference derives dweight and dbias from the shape of weight. For empty-input inference, dx retains the shape of dy, and dweight and dbias are inferred as `[0]`.
  - In a GE graph definition, `-1` denotes an unknown dimension, and `[-2]` denotes unknown rank. At execution, provide a concrete shape that meets the layout constraints. For unknown rank, specify a format compatible with the actual 4D or 5D shape.
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The direct GE operator uses 4D NCHW or 5D NCDHW. Pass the saved variance as save_rstd. The inference kernel computes all three gradients. Use the default output_mask.

## Invocation Description

| Invocation Method | Sample Code | Description |
| --- | --- | --- |
| aclnn API | [test\_aclnn\_batch\_norm\_backward](./examples/test_aclnn_batch_norm_backward.cpp) | Invokes the batch_norm_grad_v3 operator through the [aclnnBatchNormBackward](./docs/aclnnBatchNormBackward_en.md) APIs |
| aclnn API | [test\_aclnn\_fast\_batch\_norm\_backward](./examples/arch22/test_aclnn_fast_batch_norm_backward.cpp) | Invokes the batch_norm_grad_v3 operator through the [aclnnFastBatchNormBackward](./docs/aclnnFastBatchNormBackward_en.md) APIs |
| GE graph invocation | [test_geir_batch_norm_grad_v3](./examples/test_geir_batch_norm_grad_v3.cpp) | Invokes BatchNormGradV3 by building a graph with the [operator IR](op_graph/batch_norm_grad_v3_proto.h). |
| PyTorch API | [test_torch_batch_norm_grad_v3](./examples/test_torch_batch_norm_grad_v3.py) | Invokes PyTorch's `native_batch_norm_backward`. See the sample for the command. |
