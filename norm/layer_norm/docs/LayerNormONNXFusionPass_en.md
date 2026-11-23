# LayerNormONNXFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-11T12:41:18.764Z pushedAt=2026-11-16T06:28:26.358Z nl=2 -->

## Description

Identifies the combination of small operators imported from ONNX, such as ReduceMean, Sub, Pow (or Square), Add, Sqrt, and Div, and fuses them into a LayerNorm operator. For the scenario with scale and offset, the trailing Mul and Add are also fused.

- Scenario 1: with gamma and beta

  ![layer_norm_onnx_fusion_pass_1](../../../docs/en/figures/layer_norm_onnx_fusion_pass_1.png)

  **After:**

  ![layer_norm_onnx_fusion_pass_7](../../../docs/en/figures/layer_norm_onnx_fusion_pass_7.png)

- Scenario 2: without gamma and beta

  ![layer_norm_onnx_fusion_pass_2](../../../docs/en/figures/layer_norm_onnx_fusion_pass_2.png)

  **After:**

  ![layer_norm_onnx_fusion_pass_8](../../../docs/en/figures/layer_norm_onnx_fusion_pass_8.png)

- Scenario 3: Cast before the square computation, with gamma and beta

  ![layer_norm_onnx_fusion_pass_3](../../../docs/en/figures/layer_norm_onnx_fusion_pass_3.png)

  **After:**

  ![layer_norm_onnx_fusion_pass_7](../../../docs/en/figures/layer_norm_onnx_fusion_pass_7.png)

- Scenario 4: Cast before the square computation, without gamma and beta

  ![layer_norm_onnx_fusion_pass_4](../../../docs/en/figures/layer_norm_onnx_fusion_pass_4.png)

  **After:**

  ![layer_norm_onnx_fusion_pass_8](../../../docs/en/figures/layer_norm_onnx_fusion_pass_8.png)

- Scenario 5: Max and Min in sequence before the square root, without Cast, gamma, and beta

  ![layer_norm_onnx_fusion_pass_5](../../../docs/en/figures/layer_norm_onnx_fusion_pass_5.png)

  **After:**

  ![layer_norm_onnx_fusion_pass_8](../../../docs/en/figures/layer_norm_onnx_fusion_pass_8.png)

- Scenario 6: Min and Max are applied in sequence before the square root, without Cast, gamma, or beta

  ![layer_norm_onnx_fusion_pass_6](../../../docs/en/figures/layer_norm_onnx_fusion_pass_6.png)

  **After:**

  ![layer_norm_onnx_fusion_pass_8](../../../docs/en/figures/layer_norm_onnx_fusion_pass_8.png)

In the scenarios above, the exponent input exp of Pow is 2. You can also replace it with Square, which has no exp input. The division node in Scenarios 3 and 4 supports only Div. The division node in Scenarios 1, 2, 5, and 6 supports Div or RealDiv. The two inputs of Add\_0 can be swapped. When gamma and beta are present, the two inputs of Mul\_0 and Add\_1 can also be swapped separately.

After fusion, x serves as the x input of LayerNorm. When gamma and beta are present, the original inputs are reused. When they are absent, a gamma constant of all ones and a beta constant of all zeros are created, with the shape being the suffix shape corresponding to the normalization axis. The epsilon constant is converted to the epsilon attribute of LayerNorm. begin\_norm\_axis takes the starting axis of normalization, and begin\_params\_axis takes the negative value of the beta dimension. Only the y output of LayerNorm is used to replace the atomic graph output. The mean and variance outputs are not additionally connected.

## Constraints

- The dimension of x must be greater than or equal to 1.
- The two ReduceMean nodes compute along the same normalization axis, and keep\_dims is true. axes uses INT32 or INT64 constants. The normalization axis must be a non-empty axis sequence that increases consecutively and ends with the last dimension. The current implementation converts and checks the axes of ReduceMean\_0. When dimension 0 is involved, use the corresponding negative axis instead of the axis value 0.
- The first input of Sub must be x, and the second input must be the result of ReduceMean\_0. Their positions cannot be swapped. The square branch and the division numerator must reuse the same Sub result.
- In Scenario 3 and 4 with Cast, the division node supports only Div. Cast is located between Sub and Pow (or Square), and the target data type is FLOAT32. The numerator of Div still directly uses the Sub result. In other scenarios, the division node supports Div or RealDiv.
- epsilon must be a FLOAT32 or FLOAT16 constant with a scalar or [1] shape. Its value must be close to 0 and not greater than 0.1. The exponent input exp of Pow must also be a scalar or [1] constant of these two data types, with a value of 2.
- With Max and Min, the second input of Max is 0, and the second input of Min is inf (positive infinity). Both are FLOAT32 or FLOAT16 scalars or [1] constants. The constant inputs of the two operators do not support swapping.
- With gamma and beta, when both inputs of the corresponding Mul or Add have static shapes, gamma or beta must be a constant. Its number of dimensions equals the number of normalization axes, and the length of each dimension matches the corresponding normalization dimension of x. When the corresponding node has a dynamic shape input, the current implementation skips the constant and shape checks for that parameter.
- Without gamma and beta, x does not support a dynamic shape. The total number of elements in the normalization suffix shape must be greater than 0. The output data type of the division node supports only FLOAT32 and FLOAT16, which are used to create the gamma and beta constants.
- Internal nodes of the fused subgraph cannot be reused by other branches outside the subgraph. The subgraph exit can have multiple consumers. The pattern without gamma and beta uses the division node as the exit, and the pattern with gamma and beta uses Add\_1 as the exit.

## Applicable Products

<!-- npu="950" id1 -->
Ascend 950PR&950DT products
<!-- end id1 -->
