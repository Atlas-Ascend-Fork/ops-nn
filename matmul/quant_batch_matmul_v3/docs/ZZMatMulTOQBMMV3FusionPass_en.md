# ZZMatMulTOQBMMV3FusionPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-12T17:10:34.905Z pushedAt=2026-10-13T01:47:29.775Z -->

## Fusion Pattern

This fusion converts the MatMul/MatMulV2/MatMulV3 and BatchMatMul/BatchMatMulV2/BatchMatMulV3 operators into the QuantBatchMatMulV3 operator, and inserts a constant with a scale of 1.0 as the scale input. The bias input, transpose attribute, and data type of output y remain unchanged.

![](../../../docs/en/figures/ZZMatMulTOQBMMV3FusionPass_1.png)

## Constraints

- The MatMul type includes BatchMatMul/BatchMatMulV2/MatMul/MatMulV2.
- The input supports the HIFLOAT8 data type.
<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Support the combination of INT8 input and INT32 output.
<!-- end id1 -->

## Applicable Products

<!-- npu="950" id2 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id2 -->
