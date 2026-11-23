# LayerNormONNXMULFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-11T12:40:50.599Z pushedAt=2026-11-16T06:28:26.352Z nl=2 -->

## Description

Identifies and fuses small operators imported from ONNX, such as ReduceMean, two independent Sub, Mul (or Pow), Add, Sqrt (or Pow), and Div, into a LayerNormV3 operator. For scenarios with scale and shift, the trailing Mul and Add are also fused.

- Scenario 1: Use Mul to compute square, with gamma and beta.

  ![layer_norm_onnx_mul_fusion_pass_1](../../../docs/en/figures/layer_norm_onnx_mul_fusion_pass_1.png)

  **After:**

  ![layer_norm_onnx_mul_fusion_pass_3](../../../docs/en/figures/layer_norm_onnx_mul_fusion_pass_3.png)

- Scenario 2: Use Mul to compute square, without gamma and beta.

  ![layer_norm_onnx_mul_fusion_pass_2](../../../docs/en/figures/layer_norm_onnx_mul_fusion_pass_2.png)

  **After:**

  ![layer_norm_onnx_mul_fusion_pass_4](../../../docs/en/figures/layer_norm_onnx_mul_fusion_pass_4.png)

- Scenario 3: Uses Pow to compute square, with gamma and beta.

  ![layer_norm_onnx_mul_fusion_pass_5](../../../docs/en/figures/layer_norm_onnx_mul_fusion_pass_5.png)

  **After:**

  ![layer_norm_onnx_mul_fusion_pass_3](../../../docs/en/figures/layer_norm_onnx_mul_fusion_pass_3.png)

- Scenario 4: Uses Pow to compute square, without gamma and beta.

  ![layer_norm_onnx_mul_fusion_pass_6](../../../docs/en/figures/layer_norm_onnx_mul_fusion_pass_6.png)

  **After:**

  ![layer_norm_onnx_mul_fusion_pass_4](../../../docs/en/figures/layer_norm_onnx_mul_fusion_pass_4.png)

In Scenarios 1 and 2, both inputs of Mul\_0 are the result of Sub\_1, used to compute square. In Scenarios 3 and 4, Pow uses the result of Sub\_1 as the base, and the exponent input exp is 2. In all the above scenarios, Sqrt can be replaced with Pow whose exp is 0.5. The two inputs of Add\_0 can be swapped. When gamma and beta are present, the two inputs of Mul\_1 and Add\_1 can also be swapped respectively.

After fusion, the inputs of LayerNormV3 are x, gamma, and beta. When gamma and beta are present, the original inputs are reused. When they are absent, an all-ones gamma constant and an all-zeros beta constant are created, both with shape [length of the last dimension of x]. The epsilon constant is converted to the epsilon attribute of LayerNormV3. begin\_norm\_axis retains the original axis value of ReduceMean\_0, and begin\_params\_axis is fixed to -1. Only the y output of LayerNormV3 is used to replace the original subgraph output. The mean and rstd outputs are not additionally connected.

## Constraints

- The dimension of x must be greater than or equal to 1. Only normalization along the last dimension is supported.
- The axes of the two ReduceMean operators must be INT32 or INT64 constants, each containing only one element, and the original axis values must be the same. The axis value is -1 or the dimension of x minus 1. The keep\_dims of both ReduceMean operators must be true.
- Sub\_0 and Sub\_1 must be two independent nodes. The same Sub cannot be reused. The first input of both Sub operators is x, and the second input is the result of ReduceMean\_0. The positions cannot be swapped. Sub\_0 is used as the numerator of Div, and Sub\_1 is used to compute square.
- epsilon must be a FLOAT32 or FLOAT16 constant with a scalar or [1] shape.
- When Pow is used, the exponent input exp must be a FLOAT32 or FLOAT16 scalar or [1] constant. The exp of the square branch is 2, and the exp of the square root branch is 0.5.
- When gamma and beta are present, both must be one-dimensional tensors of the same length. They can be constants or non-constants.
- When gamma and beta are absent, the last dimension of x must be a known positive number. The data type of the Div output supports only FLOAT32 and FLOAT16, which are used to create gamma and beta constants. Other dimensions can be dynamic.
- The internal nodes of the fused subgraph cannot be reused by other branches outside the subgraph. The subgraph output can have multiple consumers. The pattern without gamma and beta uses Div as the output, and the pattern with gamma and beta uses Add\_1 as the output.

## Applicable Products

<!-- npu="950" id1 -->
Ascend 950PR&950DT products
<!-- end id1 -->
