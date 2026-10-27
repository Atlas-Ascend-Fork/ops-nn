# WeightQuantBatchMatmulV2TransposeFusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T17:10:14.756Z pushedAt=2026-10-13T01:47:29.770Z -->

## Fusion Patterns

<!-- npu="950,A3,910b" id1 -->
Fusion pattern 1: Delete the Transpose/TransposeD nodes before the x and/or weight inputs of the WeightQuantBatchMatmulV2 operator from the graph, and write the transpose information to the transpose_x and transpose_weight attributes of the operator. When the weight input is connected to a Transpose node, also delete the Transpose/TransposeD nodes before the antiquant_scale and antiquant_offset inputs from the graph. See the following figure.

![](../../../docs/en/figures/WeightQuantBatchMatmulV2TransposeFusionPass_1.png)

Fusion pattern 2: Delete the Transpose/TransposeD node before the weight input of the WeightQuantBatchMatmulV2 operator from the graph, and write the transpose information to the transpose_weight attribute of the operator. Also delete the simple Reshape nodes before the antiquant_scale and antiquant_offset inputs from the graph. See the following figure.

![](../../../docs/en/figures/WeightQuantBatchMatmulV2TransposeFusionPass_2.png)

This fusion mode is supported on the following products.

<!-- npu="910b" id2 -->
Atlas A2 products
<!-- end id2 -->

<!-- npu="A3" id3 -->
Atlas A3 products
<!-- end id3 -->

<!-- npu="950" id4 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id4 -->
<!-- end id1 -->

## Constraints

- At least one of the x and weight inputs must be connected to a Transpose/TransposeD node; otherwise, fusion is not triggered.
- The Transpose/TransposeD or Reshape nodes connected to antiquant_scale and antiquant_offset are processed only when the weight node is connected to a Transpose node (fusion mode 1 processes Transpose/TransposeD, and fusion mode 2 processes simple Reshape).
- In fusion mode 2, the Reshape nodes of antiquant_scale and antiquant_offset must be simple Reshapes (a dimension of the input shape is 1).
- This fusion rule cannot be disabled; disabling it causes functional issues.
- The data type of x supports only FLOAT16 and BF16; the data type of weight supports INT4, INT8, INT32, FLOAT, FLOAT4_E2M1, FLOAT8_E4M3FN, and HIFLOAT8; the output data type supports only FLOAT16, BF16, and INT8.
- The input shapes of x and weight must be 2D.
